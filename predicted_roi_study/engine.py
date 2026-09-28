"""Execution engine for the two-stage, strict predicted-ROI experiment.

Nothing in this module executes on import.  Every public stage is guarded by
the immutable protocol DAG. Before all four models x five seeds have a
validation-only primary lock, test files may only be byte-hashed by the
integrity preflight; they are never decoded into a model tensor or inferred.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import special
from torch.utils.data import DataLoader

from .ablations import (
    AblationInputBuilder,
    AblationScope,
    AblationTask,
    PREDICTED_MASK_ARMS,
    build_ablation_plan,
    build_ablation_test_schedule,
    build_ablation_validation_schedule,
    build_analytical_eye_variants,
    collect_ablation_validation_locks,
    open_ablation_test_gate,
    validation_lock_path,
    verify_ablation_validation_lock,
    write_ablation_cache,
    write_ablation_validation_lock,
)

from .config import (
    PROJECT_ROOT,
    resolve_project_path,
    sha256_file,
    verify_locked_sources,
    verify_pretrained_weights,
)
from .data import (
    ImageFrameDataset,
    ROICacheDataset,
    SegmentationFrameDataset,
    build_classifier_training_selection,
    build_inner_fold_assignment,
    classifier_loss_weights,
    crossfit_frames,
    dataframe_sha256,
    read_manifest,
    split_frames,
    strict_boolean_mask,
    training_spatial_prior,
    validate_manifest,
)
from .metrics import (
    aggregate_eyes_to_patients,
    aggregate_frames_to_eyes,
    apply_temperature_scaling,
    clean_json,
    conditional_binary_metrics,
    delong_auc_comparison,
    fit_temperature_on_probabilities,
    flatten_metrics,
    holm_adjust,
    mcnemar_exact_comparison,
    segmentation_metrics,
    select_binary_threshold,
)
from .models import (
    MODEL_SPECIFIC_CLASSIFIER_ARCHITECTURE_IDS,
    build_roi_classifier,
    build_segmenter,
    segmentation_loss,
)
from .protocol import (
    ProtocolGateError,
    assert_prerequisites,
    code_fingerprint,
    open_test_access,
    output_root,
    prepare_reused_splits,
    read_json,
    save_json_atomic,
    stage_receipt_path,
    unit_root,
    verify_stage_receipt,
    write_stage_receipt,
)
from .reporting import export_evaluation, export_five_seed_summary
from .roi import DEFAULT_NEUTRAL_RGB, ROIPolicy, extract_roi_tensor
from .qualitative import (
    render_common_segmentation_gallery,
    render_failure_gallery,
    select_failure_examples,
    verify_common_selection_lock,
    write_failure_selection_manifest,
)
from .resource_reporting import (
    FLOPS_RATIONALE,
    collect_compute_and_provenance,
    environment_inventory,
    parameter_inventory,
)


def _save_csv_atomic(path: str | Path, frame: pd.DataFrame) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig", lineterminator="\n")
    temporary.replace(destination)
    return destination


def _completed_stage_receipt_or_none(
    cfg: Mapping[str, Any],
    stage: str,
    *,
    model: str | None = None,
    seed: int | None = None,
) -> dict[str, Any] | None:
    """Return a valid immutable receipt, or ``None`` only when none exists.

    A present but stale/corrupt receipt is deliberately not treated as an
    incomplete stage.  That fail-closed distinction prevents a clean resume
    from silently rebuilding over a previously attested result.
    """

    receipt_path = stage_receipt_path(cfg, stage, model=model, seed=seed)
    if not receipt_path.exists():
        return None
    return verify_stage_receipt(cfg, stage, model=model, seed=seed)


def _torch_save_atomic(path: str | Path, value: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(destination)
    return destination


def _npz_atomic(path: str | Path, **arrays: np.ndarray) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)
    return destination


def _relative(path: str | Path) -> str:
    return str(Path(path).resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")


def _frame_identity_sha256(row: Any) -> str:
    value = (
        f"{row.patient_id}|{row.case_id}|{row.side}|{row.frame_id}"
    ).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _frame_token(row: Any) -> str:
    return _frame_identity_sha256(row)[:24]


def _seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(4)


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(value: Mapping[str, Any]) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"])
    if torch.cuda.is_available() and value.get("cuda"):
        torch.cuda.set_rng_state_all(value["cuda"])


def _device_required() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("Clinical training requires CUDA; no silent CPU fallback is allowed")
    return torch.device("cuda")


def _amp_context(cfg: Mapping[str, Any], device: torch.device):
    enabled = bool(cfg["training"]["amp"]) and device.type == "cuda"
    if cfg["training"]["amp_dtype"] != "bfloat16":
        raise ValueError("The locked mixed-precision policy is CUDA bfloat16")
    if enabled and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This CUDA device does not support the locked bfloat16 policy")
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)


def _loader(
    dataset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=generator,
    )


def _to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _write_progress(cfg: Mapping[str, Any], **fields: Any) -> None:
    save_json_atomic(
        output_root(cfg) / "progress.json",
        {"pid": os.getpid(), "updated_unix": time.time(), **clean_json(fields)},
    )


def _model_image_size(cfg: Mapping[str, Any]) -> int:
    height, width = (int(value) for value in cfg["dataset"]["image_size"])
    if height != width:
        raise ValueError("The four locked segmenters require a square input")
    return height


def _iter_masks(dataset_root: Path, rows: pd.DataFrame) -> Iterable[np.ndarray]:
    for row in rows.itertuples(index=False):
        with Image.open(dataset_root / str(row.output_mask)) as mask:
            yield np.asarray(mask) > 0


def _base_roi_policy(
    cfg: Mapping[str, Any], dataset_root: Path, train_rows: pd.DataFrame
) -> ROIPolicy:
    roi = cfg["roi"]
    cleanup = roi["component_cleanup"]
    minimum, maximum = roi["minimum_area"], roi["maximum_area"]
    return ROIPolicy.fit_from_gt(
        _iter_masks(dataset_root, train_rows),
        threshold=0.5,
        quantile_low=float(minimum["quantile"]),
        quantile_high=float(maximum["quantile"]),
        lower_scale=float(minimum["multiplier"]),
        upper_scale=float(maximum["multiplier"]),
        dominance_ratio=1.5,
        reject_border=True,
        border_margin_pixels=0,
        closing_iterations=int(cleanup["morphological_closing_iterations"]),
        max_hole_area_pixels=int(cleanup["maximum_hole_area_pixels"]),
    )


def _set_spatial_prior_if_needed(
    model: torch.nn.Module,
    dataset_root: Path,
    train_rows: pd.DataFrame,
    destination: Path,
) -> Path | None:
    if not hasattr(model, "set_spatial_prior"):
        return None
    prior_dataset = SegmentationFrameDataset(dataset_root, train_rows)
    prior = training_spatial_prior(prior_dataset)
    destination = Path(destination)
    if destination.exists():
        persisted = torch.load(destination, map_location="cpu", weights_only=True)
        if not torch.is_tensor(persisted) or not torch.equal(persisted, prior):
            raise ProtocolGateError(
                f"Existing training spatial prior conflicts with current fit rows: {destination}"
            )
        model.set_spatial_prior(persisted)
        return destination
    model.set_spatial_prior(prior)
    return _torch_save_atomic(destination, prior)


def _build_segmenter_for_fit(
    cfg: Mapping[str, Any], model_name: str, dataset_root: Path, train_rows: pd.DataFrame,
    work_dir: Path, *, pretrained: bool,
) -> tuple[torch.nn.Module, Path | None]:
    model = build_segmenter(model_name, pretrained=pretrained, image_size=_model_image_size(cfg))
    prior_path = _set_spatial_prior_if_needed(
        model, dataset_root, train_rows, work_dir / "training_spatial_prior.pt"
    )
    return model, prior_path


@torch.inference_mode()
def _validate_segmenter(
    cfg: Mapping[str, Any], model: torch.nn.Module, dataset: SegmentationFrameDataset,
    *, threshold: float = 0.5,
) -> dict[str, float]:
    model.eval()
    device = next(model.parameters()).device
    records: list[dict[str, Any]] = []
    batches = _loader(
        dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        num_workers=int(cfg["training"]["num_workers"]),
        shuffle=False,
        seed=0,
    )
    for batch in batches:
        indices = batch["index"].tolist()
        batch = _to_device(batch, device)
        with _amp_context(cfg, device):
            output = model(batch["image"])
        probability = output["seg_logits"].float().sigmoid().cpu().numpy()[:, 0]
        ground_truth = batch["mask"].bool().cpu().numpy()[:, 0]
        if not np.isfinite(probability).all():
            raise FloatingPointError("Non-finite validation segmentation probabilities")
        for local, index in enumerate(indices):
            pred, gt = probability[local] >= threshold, ground_truth[local]
            intersection = int((pred & gt).sum())
            union = int((pred | gt).sum())
            frame_row = dataset.rows.iloc[index]
            records.append(
                {
                    "patient_id": frame_row.patient_id,
                    "case_id": frame_row.case_id,
                    "side": frame_row.side,
                    "dice": 2.0 * intersection / max(int(pred.sum() + gt.sum()), 1),
                    "iou": intersection / max(union, 1),
                }
            )
    frames = pd.DataFrame(records)
    eyes = frames.groupby(
        ["patient_id", "case_id", "side"], sort=False
    )[["dice", "iou"]].mean()
    return {"eye_dice": float(eyes.dice.mean()), "eye_iou": float(eyes.iou.mean())}


def _capture_model_info(model: torch.nn.Module, model_name: str, pretrained: bool) -> dict[str, Any]:
    base = getattr(model, "base_model", model)
    disabled = int(getattr(model, "legacy_classification_parameters_frozen", 0))
    value: dict[str, Any] = {
        "name": model_name,
        "adapter_class": type(model).__name__,
        "base_class": type(base).__name__,
        "pretrained_requested": bool(pretrained),
        **parameter_inventory(
            model, explicitly_disabled_legacy_parameters=disabled
        ),
        # Backward-compatible aliases, never labeled as active-graph counts.
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "legacy_classification_parameters_frozen": disabled,
        "flops_status": "not_reported",
        "flops_rationale": FLOPS_RATIONALE,
        "output_contract": "seg_logits only; no cls_logits",
    }
    for attribute in ("pretraining_info", "pretrained_provenance", "inference_policy", "detach_roi"):
        if hasattr(base, attribute):
            value[attribute] = clean_json(getattr(base, attribute))
    return value


def _verify_segmenter_initialization_provenance(
    cfg: Mapping[str, Any], model_name: str, model_info: Mapping[str, Any]
) -> dict[str, Any]:
    specification = dict(cfg["segmentation"]["initialization"][model_name])
    observed = model_info.get(
        "pretraining_info", model_info.get("pretrained_provenance", {})
    )
    if not isinstance(observed, Mapping):
        observed = {}
    checkpoint_value = specification.get("checkpoint")
    if checkpoint_value is None:
        loaded = bool(observed.get("enabled", observed.get("pretrained", False)))
        if loaded:
            raise ProtocolGateError(
                f"{model_name} loaded pretrained bytes despite random-init lock"
            )
        return {
            "status": "verified_random_initialization",
            "configured": specification,
            "observed": dict(observed),
        }
    configured_path = resolve_project_path(checkpoint_value, must_exist=True).resolve()
    configured_sha = str(specification["sha256"])
    observed_sha = str(observed.get("sha256", ""))
    observed_path_value = observed.get("path")
    if observed_sha != configured_sha or observed_path_value is None:
        raise ProtocolGateError(
            f"{model_name} reported pretrained provenance does not match config"
        )
    observed_path = Path(str(observed_path_value)).resolve()
    if observed_path != configured_path or sha256_file(observed_path) != configured_sha:
        raise ProtocolGateError(
            f"{model_name} consumed a different pretrained checkpoint"
        )
    return {
        "status": "verified_pretrained_checkpoint",
        "configured": specification,
        "observed": dict(observed),
    }


def _train_segmenter_model(
    cfg: Mapping[str, Any],
    model_name: str,
    seed: int,
    dataset_root: Path,
    train_rows: pd.DataFrame,
    work_dir: Path,
    *,
    outer_seed: int,
    inner_fold: int | None,
    validation_rows: pd.DataFrame | None,
    fixed_epochs: int | None,
    pretrained: bool = True,
) -> dict[str, Any]:
    """Train one full or inner segmenter with resumable epoch checkpoints."""

    if validation_rows is not None and (inner_fold is not None or fixed_epochs is not None):
        raise ProtocolGateError("Outer segmenter fit cannot carry an inner fold/fixed epoch policy")
    if validation_rows is None and (inner_fold is None or fixed_epochs is None):
        raise ProtocolGateError("Inner OOF segmenter fit requires an inner fold and fixed epochs")
    offsets = cfg["training"]["seed_offsets"]
    training_seed = int(seed) + int(offsets["segmentation"])
    augmentation_seed = int(seed) + int(offsets["augmentation"])
    max_epochs = int(fixed_epochs if fixed_epochs is not None else cfg["training"]["max_epochs"])
    fit_role = "outer_full_validation_selected" if validation_rows is not None else "inner_oof_fixed_epoch"
    fingerprint = code_fingerprint()["sha256"]
    checkpoint_identity = {
        "schema": 1,
        "model": model_name,
        "fit_role": fit_role,
        "outer_seed": int(outer_seed),
        "inner_fold": int(inner_fold) if inner_fold is not None else None,
        "fit_seed_argument": int(seed),
        "training_seed": training_seed,
        "augmentation_seed": augmentation_seed,
        "train_index_sha256": dataframe_sha256(train_rows),
        "validation_index_sha256": (
            dataframe_sha256(validation_rows) if validation_rows is not None else None
        ),
        "fixed_epochs": int(fixed_epochs) if fixed_epochs is not None else None,
        "maximum_epochs": max_epochs,
        "pretrained_requested": bool(pretrained),
        "initialization_policy": dict(cfg["segmentation"]["initialization"][model_name]),
        "config_sha256": cfg["_runtime"]["config_sha256"],
        "code_sha256": fingerprint,
    }
    last_path = work_dir / "last.pt"
    selected_path = work_dir / (
        "selected.pt" if validation_rows is not None else "final.pt"
    )
    if selected_path.exists() and not last_path.exists():
        raise ProtocolGateError(f"Orphan segmenter checkpoint without resumable state: {selected_path}")
    resume_state: Mapping[str, Any] | None = None
    if last_path.exists():
        loaded_state = torch.load(last_path, map_location="cpu", weights_only=False)
        if not isinstance(loaded_state, Mapping):
            raise ProtocolGateError(f"Malformed segmenter resume checkpoint: {last_path}")
        if (
            loaded_state.get("config_sha256") != cfg["_runtime"]["config_sha256"]
            or loaded_state.get("code_sha256") != fingerprint
            or loaded_state.get("checkpoint_identity") != checkpoint_identity
        ):
            raise ProtocolGateError(f"Resume segmenter/input provenance mismatch: {last_path}")
        resume_state = loaded_state
        if validation_rows is not None and int(resume_state.get("best_epoch", -1)) >= 0:
            if not selected_path.is_file():
                raise ProtocolGateError(
                    f"Selected segmenter checkpoint is missing beside resumable state: {selected_path}"
                )
        if selected_path.exists():
            selected_header = torch.load(selected_path, map_location="cpu", weights_only=True)
            if (
                not isinstance(selected_header, Mapping)
                or selected_header.get("checkpoint_identity") != checkpoint_identity
                or int(selected_header.get("epoch", -2))
                != int(resume_state.get("best_epoch", -1))
            ):
                raise ProtocolGateError(
                    f"Selected segmenter checkpoint conflicts with resumable state: {selected_path}"
                )

    device = _device_required()
    work_dir.mkdir(parents=True, exist_ok=True)
    _seed_everything(training_seed)
    train_dataset = SegmentationFrameDataset(
        dataset_root, train_rows, cfg["training"]["augmentation"], seed=augmentation_seed
    )
    validation_dataset = (
        SegmentationFrameDataset(dataset_root, validation_rows) if validation_rows is not None else None
    )
    model, prior_path = _build_segmenter_for_fit(
        cfg, model_name, dataset_root, train_rows, work_dir, pretrained=pretrained
    )
    model_info_path = work_dir / "model_info.json"
    model_info = _capture_model_info(model, model_name, pretrained)
    model_info["checkpoint_identity"] = checkpoint_identity
    model_info["training_spatial_prior_sha256"] = (
        sha256_file(prior_path) if prior_path else None
    )
    save_json_atomic(model_info_path, model_info)
    model.to(device)
    learning_rate = float(cfg["segmentation"]["fixed_hyperparameters"][model_name]["learning_rate"])
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(cfg["training"]["max_epochs"])
    )
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    patience = int(cfg["training"]["patience"])
    best_key = (-math.inf, -math.inf)
    best_epoch = -1
    stale = 0
    first_epoch = 0
    history: list[dict[str, Any]] = []
    if resume_state is not None:
        model.load_state_dict(resume_state["model"], strict=True)
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        scaler.load_state_dict(resume_state["scaler"])
        first_epoch = int(resume_state["epoch"]) + 1
        best_key = tuple(float(value) for value in resume_state["best_key"])
        best_epoch = int(resume_state["best_epoch"])
        stale = int(resume_state["stale"])
        history = list(resume_state["history"])
        _restore_rng_state(resume_state["rng"])

    for epoch in range(first_epoch, max_epochs):
        if validation_dataset is not None and stale >= patience:
            break
        train_dataset.epoch = epoch
        model.train()
        if hasattr(model, "set_epoch"):
            model.set_epoch(epoch, int(cfg["training"]["max_epochs"]))
        batches = _loader(
            train_dataset,
            batch_size=int(cfg["training"]["batch_size"]),
            num_workers=int(cfg["training"]["num_workers"]),
            shuffle=True,
            seed=training_seed + epoch,
        )
        optimizer.zero_grad(set_to_none=True)
        loss_sum, seen = 0.0, 0
        started = time.perf_counter()
        for step, batch in enumerate(batches):
            batch = _to_device(batch, device)
            group_start = (step // int(cfg["training"]["gradient_accumulation"])) * int(
                cfg["training"]["gradient_accumulation"]
            )
            denominator = min(
                int(cfg["training"]["gradient_accumulation"]), len(batches) - group_start
            )
            with _amp_context(cfg, device):
                outputs = model(batch["image"])
                loss = segmentation_loss(model, outputs, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite segmentation loss: {model_name}/seed {seed}")
            scaler.scale(loss / denominator).backward()
            batch_size = int(batch["image"].shape[0])
            loss_sum += float(loss.detach()) * batch_size
            seen += batch_size
            if (step + 1) % int(cfg["training"]["gradient_accumulation"]) == 0 or step + 1 == len(batches):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(cfg["training"]["gradient_clip"]), error_if_nonfinite=True
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        validation = (
            _validate_segmenter(cfg, model, validation_dataset, threshold=0.5)
            if validation_dataset is not None
            else {"eye_dice": np.nan, "eye_iou": np.nan}
        )
        key = (float(validation["eye_dice"]), float(validation["eye_iou"]))
        if validation_dataset is not None:
            if not np.isfinite(key).all():
                raise RuntimeError("Validation eye Dice/IoU must be defined for segmentation early stopping")
            if key > best_key:
                best_key, best_epoch, stale = key, epoch, 0
                _torch_save_atomic(
                    selected_path,
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "validation": validation,
                        "checkpoint_identity": checkpoint_identity,
                    },
                )
            else:
                stale += 1
        else:
            best_epoch = epoch
        scheduler.step()
        event = {
            "epoch": epoch + 1,
            "train_loss": loss_sum / max(seen, 1),
            "validation_eye_dice_05": validation["eye_dice"],
            "validation_eye_iou_05": validation["eye_iou"],
            "seconds": time.perf_counter() - started,
            "learning_rate_after_step": scheduler.get_last_lr()[0],
        }
        history.append(event)
        _torch_save_atomic(
            last_path,
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best_key": best_key,
                "best_epoch": best_epoch,
                "stale": stale,
                "history": history,
                "rng": _rng_state(),
                "config_sha256": cfg["_runtime"]["config_sha256"],
                "code_sha256": fingerprint,
                "checkpoint_identity": checkpoint_identity,
            },
        )
        _save_csv_atomic(work_dir / "history.csv", pd.DataFrame(history))
        _write_progress(
            cfg,
            phase="segmentation_training",
            model=model_name,
            seed=seed,
            epoch=epoch + 1,
            target_epochs=max_epochs,
            best_epoch=best_epoch + 1,
            stale_epochs=stale,
            validation=validation,
        )
    if max_epochs <= first_epoch and not last_path.exists():
        raise RuntimeError("No segmentation epoch was executed")
    if validation_dataset is None:
        _torch_save_atomic(
            selected_path,
            {
                "model": model.state_dict(),
                "epoch": best_epoch,
                "checkpoint_identity": checkpoint_identity,
            },
        )
    else:
        if not selected_path.is_file():
            raise RuntimeError("Segmentation training produced no selected checkpoint")
        selected = torch.load(selected_path, map_location="cpu", weights_only=True)
        if selected.get("checkpoint_identity") != checkpoint_identity:
            raise ProtocolGateError(
                f"Selected segmenter checkpoint provenance mismatch: {selected_path}"
            )
        model.load_state_dict(selected["model"], strict=True)
    return {
        "model": model,
        "checkpoint": selected_path,
        "history": work_dir / "history.csv",
        "prior": prior_path,
        "best_epoch": int(best_epoch),
        "best_validation": {"eye_dice": best_key[0], "eye_iou": best_key[1]},
        "learning_rate": learning_rate,
        "training_seed": training_seed,
        "model_info": model_info_path,
        "checkpoint_identity": checkpoint_identity,
    }


@torch.inference_mode()
def _infer_probability_maps(
    cfg: Mapping[str, Any], model: torch.nn.Module, dataset_root: Path, rows: pd.DataFrame,
    destination: Path,
) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    dataset = ImageFrameDataset(dataset_root, rows)
    device = next(model.parameters()).device
    model.eval()
    records: list[dict[str, Any]] = []
    batches = _loader(
        dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        num_workers=int(cfg["training"]["num_workers"]),
        shuffle=False,
        seed=0,
    )
    for batch in batches:
        indices = batch["index"].tolist()
        images = batch["image"].to(device, non_blocking=True)
        with _amp_context(cfg, device):
            outputs = model(images)
        scores = outputs["seg_logits"].float().sigmoid().cpu().numpy()[:, 0]
        if not np.isfinite(scores).all():
            raise FloatingPointError("Non-finite segmentation probability map")
        for local, index in enumerate(indices):
            row = rows.iloc[index]
            path = destination / f"{_frame_token(row)}.npz"
            frame_identity_sha256 = _frame_identity_sha256(row)
            _npz_atomic(
                path,
                probability=scores[local].astype(np.float32, copy=False),
                frame_identity_sha256=np.frombuffer(
                    bytes.fromhex(frame_identity_sha256), dtype=np.uint8
                ).copy(),
            )
            records.append(
                {
                    "patient_id": row.patient_id,
                    "case_id": row.case_id,
                    "side": row.side,
                    "frame_id": str(row.frame_id),
                    "frame_identity_sha256": frame_identity_sha256,
                    "label": int(row.label_binary),
                    "label_3class": int(row.label_3class),
                    "map_path": _relative(path),
                    "map_sha256": sha256_file(path),
                }
            )
    index_path = destination / "index.csv"
    _save_csv_atomic(index_path, pd.DataFrame(records))
    return index_path


def _verify_file_index(index_path: Path, path_column: str, hash_column: str) -> pd.DataFrame:
    table = pd.read_csv(index_path, encoding="utf-8-sig", dtype={"frame_id": str})
    required = {
        "patient_id", "case_id", "side", "frame_id", "frame_identity_sha256",
        path_column, hash_column,
    }
    if not required <= set(table):
        raise ProtocolGateError(
            f"Probability-map index lacks columns: {sorted(required - set(table))}"
        )
    identity_columns = ["patient_id", "case_id", "side", "frame_id"]
    if table.duplicated(identity_columns).any():
        raise ProtocolGateError("Probability-map index repeats a frame identity")
    expected_parent = index_path.parent.resolve()
    for row in table.itertuples(index=False):
        path = resolve_project_path(getattr(row, path_column), must_exist=True)
        identity_sha256 = _frame_identity_sha256(row)
        if (
            str(row.frame_identity_sha256) != identity_sha256
            or path.parent != expected_parent
            or path.name != f"{identity_sha256[:24]}.npz"
        ):
            raise ProtocolGateError(
                f"Probability map path/identity is not bound to its frame: {path}"
            )
        if sha256_file(path) != getattr(row, hash_column):
            raise ProtocolGateError(f"Cached artifact changed: {path}")
        _verify_npz_frame_identity(path, identity_sha256)
    return table


def _select_roi_policy(
    cfg: Mapping[str, Any], base_policy: ROIPolicy, dataset_root: Path,
    validation_rows: pd.DataFrame, map_index_path: Path,
) -> tuple[ROIPolicy, dict[str, Any]]:
    maps = _verify_file_index(map_index_path, "map_path", "map_sha256")
    keyed = maps.set_index(["patient_id", "case_id", "side", "frame_id"])
    if keyed.index.duplicated().any():
        raise ProtocolGateError("Validation probability-map index repeats a frame identity")
    candidates: list[dict[str, Any]] = []
    thresholds = [float(value) for value in cfg["segmentation"]["threshold_candidates"]]
    ratios = [float(value) for value in cfg["roi"]["dominance"]["candidate_ratios"]]
    minimum_coverage = float(cfg["roi"]["dominance"]["minimum_validation_coverage"])
    for threshold in thresholds:
        policies = {
            ratio: replace(base_policy, threshold=threshold, dominance_ratio=ratio)
            for ratio in ratios
        }
        per_ratio: dict[float, list[dict[str, Any]]] = {ratio: [] for ratio in ratios}
        for row in validation_rows.itertuples(index=False):
            cached = keyed.loc[
                (row.patient_id, row.case_id, row.side, str(row.frame_id))
            ]
            with np.load(resolve_project_path(cached.map_path, must_exist=True), allow_pickle=False) as handle:
                probability = handle["probability"].astype(np.float32, copy=False)
            with Image.open(dataset_root / str(row.output_mask)) as handle:
                gt = np.asarray(handle) > 0
            for ratio, policy in policies.items():
                result = policy.postprocess_probability(probability)
                prediction = result.mask if result.valid else np.zeros_like(gt, dtype=bool)
                metric = segmentation_metrics(prediction, gt, distances=False)
                per_ratio[ratio].append(
                    {
                        "patient_id": row.patient_id,
                        "case_id": row.case_id,
                        "side": row.side,
                        "valid": result.valid,
                        "dice": metric["dice"],
                        "iou": metric["iou"],
                        "roi_hit": metric["roi_hit"],
                    }
                )
        for ratio, records in per_ratio.items():
            frame = pd.DataFrame(records)
            grouped = frame.groupby(
                ["patient_id", "case_id", "side"], sort=False
            )
            eye_valid_count = grouped.valid.sum()
            eye_coverage = float((eye_valid_count >= int(cfg["aggregation"]["minimum_valid_frames"])).mean())
            candidates.append(
                {
                    "threshold": threshold,
                    "dominance_ratio": ratio,
                    "postprocessed_eye_dice": float(grouped.dice.mean().mean()),
                    "postprocessed_eye_iou": float(grouped.iou.mean().mean()),
                    "eye_coverage": eye_coverage,
                    "frame_roi_hit_rate": float(frame.roi_hit.mean()),
                    "coverage_constraint_satisfied": eye_coverage >= minimum_coverage,
                }
            )
    feasible = [row for row in candidates if row["coverage_constraint_satisfied"]]
    if feasible:
        pool = feasible
        fallback_used = False
        key = lambda row: (
            -row["postprocessed_eye_dice"],
            -row["postprocessed_eye_iou"],
            -row["eye_coverage"],
            abs(row["threshold"] - 0.5),
            abs(row["dominance_ratio"] - 1.5),
            row["threshold"],
            row["dominance_ratio"],
        )
    else:
        pool = candidates
        fallback_used = True
        key = lambda row: (
            -row["eye_coverage"],
            -row["postprocessed_eye_dice"],
            -row["postprocessed_eye_iou"],
            abs(row["threshold"] - 0.5),
            abs(row["dominance_ratio"] - 1.5),
            row["threshold"],
            row["dominance_ratio"],
        )
    selected = min(pool, key=key)
    policy = replace(
        base_policy,
        threshold=float(selected["threshold"]),
        dominance_ratio=float(selected["dominance_ratio"]),
    )
    return policy, {
        "selected": selected,
        "candidates": candidates,
        "minimum_eye_coverage": minimum_coverage,
        "coverage_constraint_satisfied": bool(selected["coverage_constraint_satisfied"]),
        "infeasible_fallback_used": fallback_used,
        "selection_partition": "validation",
        "test_used": False,
        "tie_break": "objective, IoU, coverage, closest to defaults, smaller values",
    }


def prepare(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and byte-copy the five old outer splits; no images are inferred."""

    return prepare_reused_splits(cfg)


def preflight(cfg: Mapping[str, Any], model: str) -> dict[str, Any]:
    """Run a train-partition-only software/GPU/weight smoke test."""

    assert_prerequisites(cfg, "preflight", model=model)
    existing = _completed_stage_receipt_or_none(cfg, "preflight", model=model)
    if existing is not None:
        return {"status": "already_complete", "receipt": existing}
    device = _device_required()
    source_hashes = verify_locked_sources(cfg)
    dataset_audit = validate_manifest(cfg, verify_pixels=True)
    crossfit_audit: dict[str, Any] = {}
    for outer_seed in cfg["split_seeds"]:
        assignment = build_inner_fold_assignment(cfg, int(outer_seed))
        heldout_sets = [
            set(assignment.loc[assignment.inner_fold == fold, "patient_id"].astype(str))
            for fold in range(int(cfg["cross_fitting"]["folds"]))
        ]
        if any(left & right for i, left in enumerate(heldout_sets) for right in heldout_sets[i + 1 :]):
            raise ProtocolGateError(f"Cross-fit held-out patient overlap for seed {outer_seed}")
        _, outer_train_rows = split_frames(cfg, int(outer_seed), "train")
        expected_patients = set(outer_train_rows.patient_id.astype(str))
        if set().union(*heldout_sets) != expected_patients:
            raise ProtocolGateError(f"Cross-fit folds do not partition outer train for seed {outer_seed}")
        crossfit_audit[str(outer_seed)] = {
            "patients": len(expected_patients),
            "fold_sizes": [len(group) for group in heldout_sets],
            "assignment_sha256": dataframe_sha256(assignment),
            "pairwise_heldout_disjoint": True,
            "outer_validation_or_test_patients": 0,
        }
    required_free_gib = float(cfg["preflight"]["minimum_free_output_disk_gib"])
    free_gib = shutil.disk_usage(output_root(cfg).parent).free / 2**30
    if free_gib < required_free_gib:
        raise ProtocolGateError(
            f"Preflight requires at least {required_free_gib:.1f} GiB free output space; observed {free_gib:.1f} GiB"
        )
    pretrained_hashes = verify_pretrained_weights(cfg)
    weights = cfg["classifier"]["secondary"]["pretrained_weights"]
    weight_path = resolve_project_path(weights["cache_path"], must_exist=True)
    if sha256_file(weight_path) != weights["sha256"]:
        raise ProtocolGateError("The local ResNet-18 ImageNet checkpoint failed SHA-256 verification")
    dataset_root, rows = split_frames(cfg, int(cfg["split_seeds"][0]), "train")
    # The executable model contract uses no clinical image. Dataset pixels are
    # separately checked for integrity; this forward/backward path is fully
    # synthetic and cannot become performance evidence.
    image_size = _model_image_size(cfg)
    generator = torch.Generator(device="cpu").manual_seed(20260909)
    synthetic_image = torch.rand((1, 3, image_size, image_size), generator=generator)
    synthetic_mask = torch.zeros((1, 1, image_size, image_size), dtype=torch.float32)
    synthetic_mask[
        :, :, image_size // 3: 2 * image_size // 3,
        image_size // 3: 2 * image_size // 3,
    ] = 1.0
    batch = _to_device(
        {
            "image": synthetic_image,
            "mask": synthetic_mask,
            "label": torch.zeros(1, dtype=torch.long),
            "index": torch.zeros(1, dtype=torch.long),
        },
        device,
    )
    segmenter, _ = _build_segmenter_for_fit(
        cfg, model, dataset_root, rows, output_root(cfg) / "verification" / model, pretrained=True
    )
    segmenter_model_info = _capture_model_info(segmenter, model, pretrained=True)
    segmenter_initialization_audit = _verify_segmenter_initialization_provenance(
        cfg, model, segmenter_model_info
    )
    segmenter.to(device).train()
    with _amp_context(cfg, device):
        output = segmenter(batch["image"])
        loss = segmentation_loss(segmenter, output, batch)
    loss.backward()
    if not torch.isfinite(loss) or output["seg_logits"].shape != batch["mask"].shape:
        raise RuntimeError("Segmenter smoke contract failed")
    base = getattr(segmenter, "base_model", segmenter)
    legacy_classifier_gradients = [
        name
        for name, parameter in base.named_parameters()
        if "classif" in name.lower() and parameter.grad is not None
    ]
    if legacy_classifier_gradients:
        raise RuntimeError(
            "Segmentation-only backward reached legacy classifier parameters: "
            + ", ".join(legacy_classifier_gradients[:8])
        )
    segmenter_loss_value = float(loss.detach())
    segmentation_shape = list(output["seg_logits"].shape)
    del segmenter, base, output, loss
    gc.collect()
    torch.cuda.empty_cache()
    # A single synthetic hard ROI is shared byte-for-byte by both classifier
    # strategies. Each branch receives an independent forward/backward audit.
    image = batch["image"][0]
    mask = torch.zeros(image.shape[-2:], dtype=torch.bool, device=device)
    h, w = mask.shape
    mask[h // 3: 2 * h // 3, w // 3: 2 * w // 3] = True
    roi = extract_roi_tensor(
        image,
        mask,
        target_size=cfg["roi"]["input_size"],
        neutral=DEFAULT_NEUTRAL_RGB,
        return_metadata=True,
    )
    randomized = torch.rand_like(image)
    mutated_image = torch.where(mask.unsqueeze(0), image, randomized)
    mutated_roi = extract_roi_tensor(
        mutated_image,
        mask,
        target_size=cfg["roi"]["input_size"],
        neutral=DEFAULT_NEUTRAL_RGB,
        return_metadata=True,
    )
    invariance_tolerance = float(
        cfg["roi"]["anti_leakage_invariance_test"]["maximum_absolute_logit_difference"]
    )
    if (
        not torch.equal(roi.tensor, mutated_roi.tensor)
        or not torch.equal(roi.output_mask, mutated_roi.output_mask)
        or not torch.equal(roi.geometry, mutated_roi.geometry)
    ):
        raise RuntimeError("Outside-ROI classifier invariance smoke check failed")
    classifier_audits: dict[str, Any] = {}
    for strategy in _classifier_strategy_names(cfg):
        classifier = _build_classifier_from_config(
            cfg,
            pretrained=True,
            classifier_strategy=strategy,
            model_name=model,
        ).to(device)
        trainability_audit = (
            _audit_primary_classifier_trainability(cfg, model, classifier)
            if strategy == PRIMARY_CLASSIFIER_STRATEGY else {
                "status": "passed",
                "classifier_strategy": strategy,
                "policy": "all_resnet18_parameters_trainable",
                "observed_trainable_parameter_count": int(
                    sum(p.numel() for p in classifier.parameters() if p.requires_grad)
                ),
                "observed_frozen_parameter_count": int(
                    sum(p.numel() for p in classifier.parameters() if not p.requires_grad)
                ),
            }
        )
        classifier.eval()
        classifier.zero_grad(set_to_none=True)
        roi_batch = {
            "image": roi.tensor.unsqueeze(0),
            "roi_mask": roi.output_mask.unsqueeze(0),
        }
        mutated_batch = {
            "image": mutated_roi.tensor.unsqueeze(0),
            "roi_mask": mutated_roi.output_mask.unsqueeze(0),
        }
        logits = _classifier_forward(
            classifier, roi_batch, classifier_strategy=strategy
        )
        classifier_loss = F.cross_entropy(logits.float(), batch["label"])
        classifier_loss.backward()
        finite_gradients = all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in classifier.parameters()
        )
        if (
            logits.shape != (1, 2)
            or not torch.isfinite(logits).all()
            or not torch.isfinite(classifier_loss)
            or not finite_gradients
        ):
            raise RuntimeError(f"ROI classifier forward/backward contract failed: {strategy}")
        classifier.zero_grad(set_to_none=True)
        with torch.inference_mode():
            logits_a = _classifier_forward(
                classifier, roi_batch, classifier_strategy=strategy
            ).float()
            logits_b = _classifier_forward(
                classifier, mutated_batch, classifier_strategy=strategy
            ).float()
        invariance_difference = float((logits_a - logits_b).abs().max().cpu())
        if invariance_difference > invariance_tolerance:
            raise RuntimeError(
                f"Outside-ROI classifier invariance smoke check failed: {strategy}"
            )
        classifier_audits[strategy] = {
            "classifier_strategy": strategy,
            "classifier_family": model,
            "classifier_logits_shape": list(logits.shape),
            "finite_forward": True,
            "finite_backward": True,
            "strict_hard_mask_supplied": True,
            "model_internal_hard_mask_argument": bool(
                strategy == PRIMARY_CLASSIFIER_STRATEGY
            ),
            "input_contract": "strict_roi_masked_pixels_no_explicit_geometry_vector",
            "outside_roi_invariance_maximum_logit_difference": invariance_difference,
            "outside_roi_invariance_tolerance": invariance_tolerance,
            "outside_roi_invariance_passed": True,
            "pretrained_provenance": classifier.pretrained_provenance,
            "parameter_info": (
                classifier.parameter_info()
                if hasattr(classifier, "parameter_info") else parameter_inventory(classifier)
            ),
            "trainability_policy_audit": trainability_audit,
        }
        del classifier, logits, logits_a, logits_b, classifier_loss
        gc.collect()
        torch.cuda.empty_cache()
    eligibility_rows: list[dict[str, Any]] = []
    for eye_index, (valid_count, label) in enumerate(((3, 0), (4, 0), (7, 1))):
        for frame_index in range(7):
            is_valid = frame_index < valid_count
            patient_id = f"synthetic_patient_{eye_index}"
            case_id = f"synthetic_eye_{eye_index}"
            side = "SAG" if eye_index % 2 == 0 else "SOL"
            identity_sha256 = hashlib.sha256(
                f"{patient_id}|{case_id}|{side}|{frame_index}".encode("utf-8")
            ).hexdigest()
            eligibility_rows.append(
                {
                    "patient_id": patient_id,
                    "case_id": case_id,
                    "side": side,
                    "frame_id": str(frame_index),
                    "label": label,
                    "label_3class": label,
                    "frame_identity_sha256": identity_sha256,
                    "roi_valid": bool(is_valid),
                    "abstention_reason": "" if is_valid else "synthetic_invalid_roi",
                    "cache_path": "synthetic_not_loaded.npz" if is_valid else np.nan,
                }
            )
    eligibility_source = pd.DataFrame(eligibility_rows)
    eligibility_source_hash = dataframe_sha256(eligibility_source)
    synthetic_optimization, synthetic_ledger = build_classifier_training_selection(
        eligibility_source,
        frames_per_eye=int(cfg["dataset"]["frames_per_eye"]),
        minimum_valid_frames=int(cfg["aggregation"]["minimum_valid_frames"]),
    )
    synthetic_dataset = ROICacheDataset(synthetic_optimization)
    ledger_by_eye = synthetic_ledger.set_index("case_id")
    eligibility_passed = bool(
        dataframe_sha256(eligibility_source) == eligibility_source_hash
        and len(synthetic_dataset) == 11
        and not bool(ledger_by_eye.loc["synthetic_eye_0", "eye_training_eligible"])
        and bool(ledger_by_eye.loc["synthetic_eye_1", "eye_training_eligible"])
        and bool(ledger_by_eye.loc["synthetic_eye_2", "eye_training_eligible"])
        and int(ledger_by_eye.loc["synthetic_eye_0", "optimization_frame_count"]) == 0
        and int(ledger_by_eye.loc["synthetic_eye_1", "optimization_frame_count"]) == 4
        and int(ledger_by_eye.loc["synthetic_eye_2", "optimization_frame_count"]) == 7
    )
    if not eligibility_passed:
        raise RuntimeError("Classifier 3/7 versus 4/7 training eligibility smoke check failed")
    eligibility_audit = {
        "status": "passed",
        "canonical_roi_index_immutable": True,
        "three_of_seven_eye_excluded": True,
        "four_of_seven_eye_included": True,
        "seven_of_seven_eye_included": True,
        "valid_but_ineligible_cache_rows_never_relabelled": True,
        "valid_but_ineligible_cache_rows_never_enter_dataset": True,
        "frames_per_eye": int(cfg["dataset"]["frames_per_eye"]),
        "minimum_valid_frames": int(cfg["aggregation"]["minimum_valid_frames"]),
        "optimization_frame_count": int(len(synthetic_optimization)),
        "optimization_index_sha256": dataframe_sha256(synthetic_optimization),
        "eligibility_ledger_sha256": dataframe_sha256(synthetic_ledger),
        "identical_selection_for_both_classifier_strategies": True,
    }
    edge_policy = ROIPolicy(
        min_area_pixels=10,
        max_area_pixels=200,
        image_shape=(32, 32),
        threshold=0.5,
        dominance_ratio=1.5,
        reject_border=True,
    )
    edge_maps: dict[str, np.ndarray] = {
        "empty": np.zeros((32, 32), dtype=np.float32),
        "tiny": np.pad(np.ones((2, 2), dtype=np.float32), ((15, 15), (15, 15))),
        "oversize": np.pad(np.ones((20, 20), dtype=np.float32), ((6, 6), (6, 6))),
        "border": np.pad(np.ones((4, 4), dtype=np.float32), ((0, 28), (12, 16))),
    }
    multi = np.zeros((32, 32), dtype=np.float32)
    multi[5:9, 5:9] = 0.9
    multi[20:24, 20:24] = 0.9
    edge_maps["multi_ambiguous"] = multi
    edge_results = {
        expected: edge_policy.postprocess_probability(score).status.value
        for expected, score in edge_maps.items()
    }
    if any(observed != expected for expected, observed in edge_results.items()):
        raise RuntimeError(f"ROI component edge-case smoke failed: {edge_results}")
    report_path = output_root(cfg) / "verification" / f"{model}.json"
    save_json_atomic(
        report_path,
        {
            "purpose": "train-only software/GPU smoke; not performance evidence",
            "model": model,
            "partition": "synthetic_model_contract; train metadata/GT used only where required for initialization prior",
            "cuda": torch.cuda.get_device_name(0),
            "python": sys.version,
            "torch": torch.__version__,
            "torchvision": importlib.metadata.version("torchvision"),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "bfloat16_supported": torch.cuda.is_bf16_supported(),
            "environment": environment_inventory(),
            "free_output_disk_gib": free_gib,
            "minimum_free_output_disk_gib": required_free_gib,
            "locked_source_hashes": source_hashes,
            "pretrained_weight_hashes": pretrained_hashes,
            "dataset_integrity": dataset_audit,
            "crossfit_patient_disjointness": crossfit_audit,
            "segmenter_loss": segmenter_loss_value,
            "segmentation_shape": segmentation_shape,
            "primary_classifier_strategy": PRIMARY_CLASSIFIER_STRATEGY,
            "classifier_strategies": list(_classifier_strategy_names(cfg)),
            "classifier_audits": classifier_audits,
            "classifier_training_eligibility_audit": eligibility_audit,
            "required_prelaunch_audits": [
                {
                    "round": 1,
                    "name": "dual_classifier_contract_and_backward",
                    "scope": f"{model}_both_strategies",
                    "executed_scope": f"{model}_both_strategies",
                    "aggregate_required_scope": "all_four_models_both_strategies",
                    "status": "passed",
                    "strategies": {
                        name: {
                            "finite_forward": audit["finite_forward"],
                            "finite_backward": audit["finite_backward"],
                            "classifier_logits_shape": audit["classifier_logits_shape"],
                            "strict_hard_mask_supplied": audit["strict_hard_mask_supplied"],
                        }
                        for name, audit in classifier_audits.items()
                    },
                },
                {
                    "round": 2,
                    "name": "synthetic_strict_roi_end_to_end_and_outside_roi_invariance",
                    "scope": f"{model}_both_strategies",
                    "executed_scope": f"{model}_both_strategies",
                    "aggregate_required_scope": "all_four_models_both_strategies",
                    "status": "passed",
                    "strict_roi_tensor_shared_by_both_strategies": True,
                    "classifier_training_eligibility_contract_passed": True,
                    "strategies": {
                        name: {
                            "outside_roi_invariance_passed": audit[
                                "outside_roi_invariance_passed"
                            ],
                            "maximum_absolute_logit_difference": audit[
                                "outside_roi_invariance_maximum_logit_difference"
                            ],
                        }
                        for name, audit in classifier_audits.items()
                    },
                },
            ],
            "roi_edge_case_statuses": edge_results,
            "outside_roi_invariance_tolerance": invariance_tolerance,
            "classifier_weight_provenance": {
                strategy: audit["pretrained_provenance"]
                for strategy, audit in classifier_audits.items()
            },
            "segmenter_model_info": segmenter_model_info,
            "segmenter_initialization_audit": segmenter_initialization_audit,
            "legacy_classifier_gradients": legacy_classifier_gradients,
            "network_download_during_run": False,
            "passed": True,
            "test_files_byte_hashed_for_integrity": True,
            "test_images_decoded_for_model_or_inferred": False,
            "clinical_images_decoded_for_model_contract": False,
        },
    )
    del batch, image, roi, mutated_roi, mutated_image
    gc.collect()
    torch.cuda.empty_cache()
    weight_artifacts = [weight_path]
    for specification in cfg["classifier"]["primary"][
        "initialization_by_segmenter"
    ].values():
        if specification.get("checkpoint") is not None:
            candidate = resolve_project_path(specification["checkpoint"], must_exist=True)
            if candidate not in weight_artifacts:
                weight_artifacts.append(candidate)
    for specification in cfg["segmentation"]["initialization"].values():
        if specification["checkpoint"] is not None:
            candidate = resolve_project_path(specification["checkpoint"], must_exist=True)
            if candidate not in weight_artifacts:
                weight_artifacts.append(candidate)
    receipt = write_stage_receipt(
        cfg, "preflight", model=model, artifacts=[report_path, *weight_artifacts]
    )
    return {"status": "complete", "receipt": str(receipt), "artifacts": [str(report_path)]}


def train_segmenter(cfg: Mapping[str, Any], model: str, seed: int) -> dict[str, Any]:
    """Fit one outer segmenter and lock its post-processing using validation only."""

    assert_prerequisites(cfg, "train-segmenters", model=model, seed=seed)
    existing = _completed_stage_receipt_or_none(
        cfg, "train-segmenters", model=model, seed=seed
    )
    if existing is not None:
        return {"status": "already_complete", "receipt": existing}
    run = unit_root(cfg, model, seed)
    segmenter_dir = run / "segmenter"
    dataset_root, train_rows = split_frames(cfg, seed, "train")
    _, validation_rows = split_frames(cfg, seed, "validation")
    if set(train_rows.patient_id.astype(str)) & set(validation_rows.patient_id.astype(str)):
        raise RuntimeError("Outer patient leakage between training and validation")
    base_policy = _base_roi_policy(cfg, dataset_root, train_rows)
    trained = _train_segmenter_model(
        cfg,
        model,
        seed,
        dataset_root,
        train_rows,
        segmenter_dir,
        outer_seed=seed,
        inner_fold=None,
        validation_rows=validation_rows,
        fixed_epochs=None,
    )
    map_index = _infer_probability_maps(
        cfg,
        trained["model"],
        dataset_root,
        validation_rows,
        segmenter_dir / "validation_probability_maps",
    )
    policy, selection = _select_roi_policy(
        cfg, base_policy, dataset_root, validation_rows, map_index
    )
    lock_path = segmenter_dir / "segmenter_lock.json"
    save_json_atomic(
        lock_path,
        {
            "model": model,
            "seed": seed,
            "stage": "segmentation_only",
            "classifier_loss_weight": 0.0,
            "best_epoch_zero_based": trained["best_epoch"],
            "epochs_trained_for_selected_checkpoint": trained["best_epoch"] + 1,
            "best_validation_raw_threshold_05": trained["best_validation"],
            "checkpoint": _relative(trained["checkpoint"]),
            "checkpoint_sha256": sha256_file(trained["checkpoint"]),
            "checkpoint_identity": trained["checkpoint_identity"],
            "training_spatial_prior": _relative(trained["prior"]) if trained["prior"] else None,
            "training_spatial_prior_sha256": sha256_file(trained["prior"]) if trained["prior"] else None,
            "roi_policy": policy.to_dict(),
            "roi_policy_selection": selection,
            "area_bounds_source": "outer training GT only",
            "threshold_and_dominance_source": "outer validation only",
            "test_used": False,
            "config_sha256": cfg["_runtime"]["config_sha256"],
            "code_sha256": code_fingerprint()["sha256"],
        },
    )
    artifacts = [
        trained["checkpoint"],
        trained["history"],
        trained["model_info"],
        map_index,
        lock_path,
    ]
    if trained["prior"]:
        artifacts.append(trained["prior"])
    receipt = write_stage_receipt(
        cfg,
        "train-segmenters",
        model=model,
        seed=seed,
        artifacts=artifacts,
        metadata={
            "best_epoch": trained["best_epoch"] + 1,
            "roi_policy": policy.to_dict(),
            "coverage_constraint_satisfied": selection["coverage_constraint_satisfied"],
            "test_images_loaded": False,
        },
    )
    del trained["model"]
    gc.collect()
    torch.cuda.empty_cache()
    return {"status": "complete", "receipt": str(receipt), "artifacts": [str(p) for p in artifacts]}


def _verify_segmenter_checkpoint_identity(
    cfg: Mapping[str, Any],
    model_name: str,
    lock: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    outer_seed: int,
    inner_fold: int | None,
) -> dict[str, Any]:
    identity = lock.get("checkpoint_identity")
    if not isinstance(identity, Mapping):
        raise ProtocolGateError("Segmenter lock lacks a structured checkpoint identity")
    expected_role = (
        "outer_full_validation_selected" if inner_fold is None else "inner_oof_fixed_epoch"
    )
    expected_fields = {
        "model": model_name,
        "fit_role": expected_role,
        "outer_seed": int(outer_seed),
        "inner_fold": int(inner_fold) if inner_fold is not None else None,
        "config_sha256": cfg["_runtime"]["config_sha256"],
        "code_sha256": code_fingerprint()["sha256"],
    }
    if any(identity.get(key) != value for key, value in expected_fields.items()):
        raise ProtocolGateError("Segmenter checkpoint identity conflicts with the requested fit")
    if payload.get("checkpoint_identity") != dict(identity):
        raise ProtocolGateError("Segmenter checkpoint payload and lock identities differ")
    return dict(identity)


def _load_segmenter_from_lock(
    cfg: Mapping[str, Any], model_name: str, lock: Mapping[str, Any]
) -> torch.nn.Module:
    checkpoint_path = resolve_project_path(lock["checkpoint"], must_exist=True)
    if sha256_file(checkpoint_path) != lock["checkpoint_sha256"]:
        raise ProtocolGateError(f"Locked segmenter checkpoint changed: {checkpoint_path}")
    model = build_segmenter(
        model_name, pretrained=False, image_size=_model_image_size(cfg)
    )
    prior_value = lock.get("training_spatial_prior")
    if prior_value:
        prior_path = resolve_project_path(prior_value, must_exist=True)
        if sha256_file(prior_path) != lock["training_spatial_prior_sha256"]:
            raise ProtocolGateError(f"Locked spatial prior changed: {prior_path}")
        model.set_spatial_prior(torch.load(prior_path, map_location="cpu", weights_only=True))
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    _verify_segmenter_checkpoint_identity(
        cfg,
        model_name,
        lock,
        payload,
        outer_seed=int(lock["seed"]),
        inner_fold=None,
    )
    model.load_state_dict(payload["model"], strict=True)
    return model.to(_device_required()).eval()


def _cache_one_roi(
    cfg: Mapping[str, Any],
    row: Any,
    image: torch.Tensor,
    probability: np.ndarray,
    policy: ROIPolicy,
    cache_dir: Path,
    audit_dir: Path | None,
    *,
    audit_probability: bool = True,
) -> dict[str, Any]:
    result = policy.postprocess_probability(probability)
    record: dict[str, Any] = {
        "patient_id": row.patient_id,
        "case_id": row.case_id,
        "side": row.side,
        "frame_id": str(row.frame_id),
        "label": int(row.label_binary),
        "label_3class": int(row.label_3class),
        "frame_identity_sha256": _frame_identity_sha256(row),
        "roi_valid": bool(result.valid),
        "abstention_reason": "" if result.valid else result.status.value,
        "probability": np.nan,
        "logit_0": np.nan,
        "logit_1": np.nan,
        "cache_path": None,
        "cache_sha256": None,
        "audit_path": None,
        "audit_sha256": None,
        "component_count": len(result.components),
        "selected_area_pixels": int(result.area_pixels),
        "selected_area_fraction": float(result.area_pixels / probability.size),
        "selected_bbox_xyxy": (
            json.dumps(list(result.bbox_xyxy)) if result.bbox_xyxy is not None else ""
        ),
        "dominance_ratio_observed": result.dominance_ratio_observed,
        "roi_confidence": (
            float(result.selected_component.mean_probability)
            if result.valid and result.selected_component is not None
            else np.nan
        ),
    }
    token = _frame_token(row)
    if result.valid:
        extraction = extract_roi_tensor(
            image,
            result.mask,
            target_size=cfg["roi"]["input_size"],
            neutral=DEFAULT_NEUTRAL_RGB,
            return_metadata=True,
        )
        cache_path = cache_dir / f"{token}.npz"
        _npz_atomic(
            cache_path,
            image=extraction.tensor.detach().cpu().numpy().astype(np.float32, copy=False),
            mask=extraction.output_mask.detach().cpu().numpy().astype(np.uint8, copy=False),
            source_mask=result.mask.astype(np.uint8, copy=False),
            geometry=extraction.geometry.detach().cpu().numpy().astype(np.float32, copy=False),
            frame_identity_sha256=np.frombuffer(
                bytes.fromhex(record["frame_identity_sha256"]), dtype=np.uint8
            ).copy(),
        )
        record["cache_path"] = _relative(cache_path)
        record["cache_sha256"] = sha256_file(cache_path)
    if audit_dir is not None:
        audit_path = audit_dir / f"{token}.npz"
        audit_arrays = {
            "thresholded_mask": result.hard_mask.astype(np.uint8, copy=False),
            "selected_mask": result.mask.astype(np.uint8, copy=False),
            "frame_identity_sha256": np.frombuffer(
                bytes.fromhex(record["frame_identity_sha256"]), dtype=np.uint8
            ).copy(),
        }
        if audit_probability:
            audit_arrays["probability"] = probability.astype(np.float32, copy=False)
        _npz_atomic(audit_path, **audit_arrays)
        record["audit_path"] = _relative(audit_path)
        record["audit_sha256"] = sha256_file(audit_path)
    return record


def _cache_rois_from_probability_index(
    cfg: Mapping[str, Any],
    dataset_root: Path,
    rows: pd.DataFrame,
    probability_index_path: Path,
    policy: ROIPolicy,
    destination: Path,
    *,
    save_audit: bool,
) -> Path:
    maps = _verify_file_index(probability_index_path, "map_path", "map_sha256")
    map_lookup = maps.set_index(["patient_id", "case_id", "side", "frame_id"])
    if map_lookup.index.duplicated().any():
        raise ProtocolGateError("Probability-map index repeats a frame identity")
    images = ImageFrameDataset(dataset_root, rows)
    cache_dir = destination / "tensors"
    audit_dir = destination / "audit" if save_audit else None
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows.itertuples(index=False)):
        key = (row.patient_id, row.case_id, row.side, str(row.frame_id))
        if key not in map_lookup.index:
            raise RuntimeError(f"Missing validation probability map: {key}")
        map_row = map_lookup.loc[key]
        with np.load(resolve_project_path(map_row.map_path, must_exist=True), allow_pickle=False) as cached:
            probability = cached["probability"].astype(np.float32, copy=False)
        records.append(
            _cache_one_roi(
                cfg, row, images[index]["image"], probability, policy, cache_dir, audit_dir
            )
        )
    index_path = destination / "index.csv"
    _save_csv_atomic(index_path, pd.DataFrame(records))
    return index_path


@torch.inference_mode()
def _infer_and_cache_rois(
    cfg: Mapping[str, Any],
    model: torch.nn.Module,
    dataset_root: Path,
    rows: pd.DataFrame,
    policy: ROIPolicy,
    destination: Path,
    *,
    save_audit: bool,
    audit_probability: bool = True,
) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    cache_dir = destination / "tensors"
    audit_dir = destination / "audit" if save_audit else None
    dataset = ImageFrameDataset(dataset_root, rows)
    device = next(model.parameters()).device
    model.eval()
    records: list[dict[str, Any]] = []
    batches = _loader(
        dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        num_workers=int(cfg["training"]["num_workers"]),
        shuffle=False,
        seed=0,
    )
    for batch in batches:
        indices = batch["index"].tolist()
        source_images = batch["image"]
        images = source_images.to(device, non_blocking=True)
        with _amp_context(cfg, device):
            outputs = model(images)
        probabilities = outputs["seg_logits"].float().sigmoid().cpu().numpy()[:, 0]
        if not np.isfinite(probabilities).all():
            raise FloatingPointError("Non-finite segmentation probability during ROI construction")
        for local, index in enumerate(indices):
            row = rows.iloc[index]
            records.append(
                _cache_one_roi(
                    cfg,
                    row,
                    source_images[local],
                    probabilities[local],
                    policy,
                    cache_dir,
                    audit_dir,
                    audit_probability=audit_probability,
                )
            )
    index_path = destination / "index.csv"
    _save_csv_atomic(index_path, pd.DataFrame(records))
    return index_path


def _verify_npz_frame_identity(path: Path, expected_sha256: str) -> None:
    with np.load(path, allow_pickle=False) as cached:
        if "frame_identity_sha256" not in cached.files:
            raise ProtocolGateError(f"ROI artifact lacks its embedded frame identity: {path}")
        encoded = np.asarray(cached["frame_identity_sha256"])
    if encoded.dtype != np.uint8 or encoded.shape != (32,):
        raise ProtocolGateError(f"ROI artifact has a malformed embedded frame identity: {path}")
    if encoded.tobytes().hex() != expected_sha256:
        raise ProtocolGateError(f"ROI artifact belongs to a different frame: {path}")


def _verify_roi_index(
    cfg: Mapping[str, Any], index_path: Path, expected_rows: pd.DataFrame
) -> pd.DataFrame:
    index = pd.read_csv(index_path, encoding="utf-8-sig", dtype={"frame_id": str})
    required = {
        "patient_id", "case_id", "side", "frame_id", "label", "label_3class",
        "frame_identity_sha256", "roi_valid", "abstention_reason", "cache_path",
        "cache_sha256", "roi_confidence",
    }
    if not required <= set(index):
        raise ProtocolGateError(f"ROI index lacks columns: {sorted(required - set(index))}")
    identity_columns = ["patient_id", "case_id", "side", "frame_id"]
    if index.duplicated(identity_columns).any():
        raise ProtocolGateError("ROI cache index repeats an eye/frame identity")
    if expected_rows.duplicated(identity_columns).any():
        raise ProtocolGateError("Expected split rows repeat an eye/frame identity")
    expected_signatures = sorted(
        zip(
            expected_rows.patient_id.astype(str),
            expected_rows.case_id.astype(str),
            expected_rows.side.astype(str),
            expected_rows.frame_id.astype(str),
            expected_rows.label_binary.astype(int),
            expected_rows.label_3class.astype(int),
            strict=True,
        )
    )
    observed_signatures = sorted(
        zip(
            index.patient_id.astype(str),
            index.case_id.astype(str),
            index.side.astype(str),
            index.frame_id.astype(str),
            index.label.astype(int),
            index.label_3class.astype(int),
            strict=True,
        )
    )
    if observed_signatures != expected_signatures:
        raise ProtocolGateError(
            "ROI cache index frame/side/label identities differ from the expected split"
        )
    valid_mask = strict_boolean_mask(index["roi_valid"], name="roi_valid")
    index["roi_valid"] = valid_mask.to_numpy(dtype=bool)
    combined_oof = index_path.name == "train_oof_index.csv"
    if combined_oof and "oof_fold" not in index:
        raise ProtocolGateError("Combined OOF ROI index lacks its fold provenance")
    if combined_oof:
        fold_values = pd.to_numeric(index["oof_fold"], errors="coerce")
        if (
            fold_values.isna().any()
            or not np.equal(fold_values.to_numpy(float), fold_values.to_numpy(int)).all()
            or set(fold_values.astype(int))
            != set(range(int(cfg["cross_fitting"]["folds"])))
        ):
            raise ProtocolGateError("Combined OOF ROI index has invalid fold coverage")
    for row, roi_is_valid in zip(index.itertuples(index=False), valid_mask, strict=True):
        if combined_oof:
            try:
                fold = int(row.oof_fold)
            except (TypeError, ValueError) as error:
                raise ProtocolGateError("OOF ROI row has an invalid fold identity") from error
            if fold < 0 or fold >= int(cfg["cross_fitting"]["folds"]):
                raise ProtocolGateError("OOF ROI row refers to an undeclared fold")
            cache_parent = (
                index_path.parent.parent
                / "crossfit"
                / f"fold_{fold}"
                / "heldout_rois"
            )
        else:
            cache_parent = index_path.parent
        expected_cache_root = (cache_parent / "tensors").resolve()
        expected_audit_root = (cache_parent / "audit").resolve()
        identity_sha256 = _frame_identity_sha256(row)
        if row.frame_identity_sha256 != identity_sha256:
            raise ProtocolGateError("ROI index carries a mismatched frame identity digest")
        expected_name = f"{identity_sha256[:24]}.npz"
        if bool(roi_is_valid):
            if not np.isfinite(row.roi_confidence) or not 0 <= float(row.roi_confidence) <= 1:
                raise ProtocolGateError("A valid ROI must carry a finite mean-probability confidence")
            path = resolve_project_path(row.cache_path, must_exist=True)
            if path.parent != expected_cache_root or path.name != expected_name:
                raise ProtocolGateError(f"ROI tensor cache path is not bound to its frame: {path}")
            if sha256_file(path) != row.cache_sha256:
                raise ProtocolGateError(f"ROI tensor cache changed: {path}")
            _verify_npz_frame_identity(path, identity_sha256)
        elif pd.notna(row.cache_path) or pd.notna(row.cache_sha256) or pd.notna(row.roi_confidence):
            raise ProtocolGateError("An abstained frame must not have a classifier tensor or confidence")
        if hasattr(row, "audit_path") and pd.notna(row.audit_path):
            path = resolve_project_path(row.audit_path, must_exist=True)
            if path.parent != expected_audit_root or path.name != expected_name:
                raise ProtocolGateError(f"ROI audit path is not bound to its frame: {path}")
            if sha256_file(path) != row.audit_sha256:
                raise ProtocolGateError(f"ROI audit cache changed: {path}")
            _verify_npz_frame_identity(path, identity_sha256)
    return index


def build_rois(cfg: Mapping[str, Any], model: str, seed: int) -> dict[str, Any]:
    """Build OOF train ROIs plus outer-validation ROIs; test remains inaccessible."""

    assert_prerequisites(cfg, "build-rois", model=model, seed=seed)
    existing = _completed_stage_receipt_or_none(cfg, "build-rois", model=model, seed=seed)
    if existing is not None:
        return {"status": "already_complete", "receipt": existing}
    run = unit_root(cfg, model, seed)
    roi_root = run / "roi"
    segmenter_lock_path = run / "segmenter" / "segmenter_lock.json"
    segmenter_lock = read_json(segmenter_lock_path)
    outer_checkpoint = resolve_project_path(segmenter_lock["checkpoint"], must_exist=True)
    if sha256_file(outer_checkpoint) != segmenter_lock["checkpoint_sha256"]:
        raise ProtocolGateError(f"Outer segmenter checkpoint changed: {outer_checkpoint}")
    outer_payload = torch.load(outer_checkpoint, map_location="cpu", weights_only=True)
    _verify_segmenter_checkpoint_identity(
        cfg,
        model,
        segmenter_lock,
        outer_payload,
        outer_seed=seed,
        inner_fold=None,
    )
    outer_policy = ROIPolicy.from_dict(segmenter_lock["roi_policy"])
    dataset_root, outer_train = split_frames(cfg, seed, "train")
    _, validation_rows = split_frames(cfg, seed, "validation")
    assignments = build_inner_fold_assignment(cfg, seed)
    fold_indices: list[pd.DataFrame] = []
    fold_artifacts: list[Path] = []
    selected_epochs = int(segmenter_lock["epochs_trained_for_selected_checkpoint"])
    if selected_epochs < 1:
        raise ProtocolGateError("Outer segmentation lock has an invalid epoch count")

    for fold in range(int(cfg["cross_fitting"]["folds"])):
        fold_dir = run / "crossfit" / f"fold_{fold}"
        fold_lock_path = fold_dir / "fold_lock.json"
        _, fit_rows = crossfit_frames(cfg, seed, fold, "fit")
        _, heldout_rows = crossfit_frames(cfg, seed, fold, "heldout")
        if set(fit_rows.patient_id.astype(str)) & set(heldout_rows.patient_id.astype(str)):
            raise RuntimeError("Inner cross-fit patient leakage")
        if fold_lock_path.exists():
            fold_lock = read_json(fold_lock_path)
            if (
                fold_lock.get("config_sha256") != cfg["_runtime"]["config_sha256"]
                or fold_lock.get("code_sha256") != code_fingerprint()["sha256"]
            ):
                raise ProtocolGateError(f"Stale cross-fit lock: {fold_lock_path}")
            index_path = resolve_project_path(fold_lock["heldout_roi_index"], must_exist=True)
            if sha256_file(index_path) != fold_lock["heldout_roi_index_sha256"]:
                raise ProtocolGateError(f"Cross-fit ROI index changed: {index_path}")
            _verify_roi_index(cfg, index_path, heldout_rows)
            checkpoint = resolve_project_path(fold_lock["checkpoint"], must_exist=True)
            if sha256_file(checkpoint) != fold_lock["checkpoint_sha256"]:
                raise ProtocolGateError(f"Cross-fit checkpoint changed: {checkpoint}")
            checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            identity = _verify_segmenter_checkpoint_identity(
                cfg,
                model,
                fold_lock,
                checkpoint_payload,
                outer_seed=seed,
                inner_fold=fold,
            )
            if (
                identity.get("train_index_sha256") != dataframe_sha256(fit_rows)
                or identity.get("fixed_epochs") != selected_epochs
                or identity.get("maximum_epochs") != selected_epochs
            ):
                raise ProtocolGateError(f"Cross-fit checkpoint inputs changed: {checkpoint}")
        else:
            fold_policy = _base_roi_policy(cfg, dataset_root, fit_rows)
            fold_policy = replace(
                fold_policy,
                threshold=outer_policy.threshold,
                dominance_ratio=outer_policy.dominance_ratio,
            )
            # The formula is deterministic and recorded; held-out labels/masks
            # are never used for training, early stopping, or quality bounds.
            fold_training_seed = int(seed) + (fold + 1) * 100_003
            trained = _train_segmenter_model(
                cfg,
                model,
                fold_training_seed,
                dataset_root,
                fit_rows,
                fold_dir / "segmenter",
                outer_seed=seed,
                inner_fold=fold,
                validation_rows=None,
                fixed_epochs=selected_epochs,
            )
            index_path = _infer_and_cache_rois(
                cfg,
                trained["model"],
                dataset_root,
                heldout_rows,
                fold_policy,
                fold_dir / "heldout_rois",
                # Store only the two hard audit rasters for later ablations.
                # The scalar component confidence is in the index; omitting
                # float probability maps keeps the OOF runs disk-bounded.
                save_audit=True,
                audit_probability=False,
            )
            _verify_roi_index(cfg, index_path, heldout_rows)
            save_json_atomic(
                fold_lock_path,
                {
                    "model": model,
                    "outer_seed": seed,
                    "inner_fold": fold,
                    "fit_seed_argument": fold_training_seed,
                    "fit_patients": sorted(fit_rows.patient_id.astype(str).unique()),
                    "heldout_patients": sorted(heldout_rows.patient_id.astype(str).unique()),
                    "heldout_masks_or_labels_used_for_segmenter_fit_or_roi_policy": False,
                    "epochs_inherited_from_outer_validation_lock": selected_epochs,
                    "roi_policy": fold_policy.to_dict(),
                    "checkpoint": _relative(trained["checkpoint"]),
                    "checkpoint_sha256": sha256_file(trained["checkpoint"]),
                    "checkpoint_identity": trained["checkpoint_identity"],
                    "heldout_roi_index": _relative(index_path),
                    "heldout_roi_index_sha256": sha256_file(index_path),
                    "config_sha256": cfg["_runtime"]["config_sha256"],
                    "code_sha256": code_fingerprint()["sha256"],
                },
            )
            checkpoint = trained["checkpoint"]
            del trained["model"]
            gc.collect()
            torch.cuda.empty_cache()
        fold_table = pd.read_csv(index_path, encoding="utf-8-sig", dtype={"frame_id": str})
        fold_table["oof_fold"] = fold
        fold_indices.append(fold_table)
        fold_model_info = fold_dir / "segmenter" / "model_info.json"
        if not fold_model_info.is_file():
            raise ProtocolGateError(
                f"Cross-fit model provenance is missing: {fold_model_info}"
            )
        fold_artifacts.extend(
            [Path(index_path), Path(checkpoint), fold_lock_path, fold_model_info]
        )

    train_index = pd.concat(fold_indices, ignore_index=True).sort_values(
        ["patient_id", "case_id", "side", "frame_id"], kind="stable"
    ).reset_index(drop=True)
    train_index_path = roi_root / "train_oof_index.csv"
    _save_csv_atomic(train_index_path, train_index)
    _verify_roi_index(cfg, train_index_path, outer_train)

    validation_map_index = run / "segmenter" / "validation_probability_maps" / "index.csv"
    validation_index_path = _cache_rois_from_probability_index(
        cfg,
        dataset_root,
        validation_rows,
        validation_map_index,
        outer_policy,
        roi_root / "validation",
        save_audit=True,
    )
    validation_index = _verify_roi_index(cfg, validation_index_path, validation_rows)
    gc.collect()
    torch.cuda.empty_cache()

    overview_path = roi_root / "crossfit_overview.json"
    save_json_atomic(
        overview_path,
        {
            "model": model,
            "seed": seed,
            "folds": int(cfg["cross_fitting"]["folds"]),
            "outer_training_frames": len(outer_train),
            "oof_frames": len(train_index),
            "oof_unique_patients": int(train_index.patient_id.nunique()),
            "validation_frames": len(validation_index),
            "train_valid_roi_frames": int(train_index.roi_valid.astype(bool).sum()),
            "validation_valid_roi_frames": int(validation_index.roi_valid.astype(bool).sum()),
            "outer_selected_epochs_used_by_every_inner_fit": selected_epochs,
            "training_roi_source": "out-of-fold predicted masks",
            "validation_roi_source": "outer full-fit predicted masks",
            "ground_truth_or_full_image_fallback": False,
            "assignment_sha256": dataframe_sha256(assignments),
            "test_images_loaded": False,
        },
    )
    artifacts = [train_index_path, validation_index_path, overview_path, *fold_artifacts]
    receipt = write_stage_receipt(
        cfg,
        "build-rois",
        model=model,
        seed=seed,
        artifacts=artifacts,
        metadata={
            "oof_segmenter_fits": int(cfg["cross_fitting"]["folds"]),
            "train_valid_roi_frames": int(train_index.roi_valid.astype(bool).sum()),
            "validation_valid_roi_frames": int(validation_index.roi_valid.astype(bool).sum()),
            "test_images_loaded": False,
        },
    )
    return {"status": "complete", "receipt": str(receipt), "artifacts": [str(path) for path in artifacts]}


PRIMARY_CLASSIFIER_STRATEGY = "model_specific"
STANDARDIZED_CLASSIFIER_STRATEGY = "standardized_resnet18"


def _classifier_strategy_specs(cfg: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    classifier = cfg["classifier"]
    if "primary" in classifier and "secondary" in classifier:
        specs = {
            str(classifier["primary"]["name"]): dict(classifier["primary"]),
            str(classifier["secondary"]["name"]): dict(classifier["secondary"]),
        }
        order = tuple(str(value) for value in classifier["strategy_order"])
        if order != tuple(specs) or order != (
            PRIMARY_CLASSIFIER_STRATEGY,
            STANDARDIZED_CLASSIFIER_STRATEGY,
        ):
            raise ProtocolGateError("Classifier strategy order/configuration is not the locked dual-branch protocol")
        return specs
    # Compatibility for isolated unit fixtures created before the dual-branch
    # config lock.  Clinical load_config() always provides the explicit specs.
    return {
        PRIMARY_CLASSIFIER_STRATEGY: {"name": PRIMARY_CLASSIFIER_STRATEGY, "seed_offset": 0},
        STANDARDIZED_CLASSIFIER_STRATEGY: {
            "name": STANDARDIZED_CLASSIFIER_STRATEGY,
            "seed_offset": 500_009,
        },
    }


def _classifier_strategy_names(cfg: Mapping[str, Any]) -> tuple[str, str]:
    specs = _classifier_strategy_specs(cfg)
    return tuple(specs)  # type: ignore[return-value]


def _classifier_work_dir(run: Path, strategy: str) -> Path:
    return run / "classifiers" / str(strategy)


def _classifier_training_seed(cfg: Mapping[str, Any], seed: int, strategy: str) -> int:
    specs = _classifier_strategy_specs(cfg)
    if strategy not in specs:
        raise ProtocolGateError(f"Unknown classifier strategy: {strategy}")
    return (
        int(seed)
        + int(cfg["training"]["seed_offsets"]["classifier"])
        + int(specs[strategy].get("seed_offset", 0))
    )


def _build_classifier_from_config(
    cfg: Mapping[str, Any], *, pretrained: bool, geometry_features: int = 0,
    classifier_strategy: str = STANDARDIZED_CLASSIFIER_STRATEGY,
    model_name: str | None = None,
) -> torch.nn.Module:
    if classifier_strategy == STANDARDIZED_CLASSIFIER_STRATEGY:
        weights = cfg["classifier"].get("pretrained_weights") or cfg["classifier"][
            "secondary"
        ]["pretrained_weights"]
        weight_path = (
            resolve_project_path(weights["cache_path"], must_exist=True)
            if pretrained else None
        )
        return build_roi_classifier(
            backbone="resnet18",
            pretrained=pretrained,
            num_classes=int(cfg["classifier"]["classes"]),
            dropout=float(cfg["classifier"].get("dropout", 0.2)),
            geometry_features=int(geometry_features),
            weights_path=weight_path,
            expected_sha256=weights["sha256"] if pretrained else None,
        )
    if classifier_strategy != PRIMARY_CLASSIFIER_STRATEGY or model_name is None:
        raise ProtocolGateError(f"Unsupported classifier strategy/model: {classifier_strategy}/{model_name}")
    from .models import build_strict_roi_classifier

    initialization = cfg["classifier"]["primary"]["initialization_by_segmenter"][
        model_name
    ]
    use_pretrained = bool(pretrained and initialization.get("pretrained"))
    checkpoint_value = initialization.get("checkpoint") if use_pretrained else None
    checkpoint = (
        resolve_project_path(checkpoint_value, must_exist=True)
        if use_pretrained else None
    )
    return build_strict_roi_classifier(
        model_name,
        pretrained=use_pretrained,
        weights_path=checkpoint,
        expected_sha256=initialization.get("sha256") if use_pretrained else None,
        dropout=None,
        geometry_features=int(geometry_features),
        lightweight=False,
    )


def _classifier_forward(
    classifier: torch.nn.Module,
    batch: Mapping[str, Any],
    *,
    classifier_strategy: str,
    geometry_features: int = 0,
) -> torch.Tensor:
    if "roi_mask" not in batch or batch["roi_mask"] is None:
        raise ProtocolGateError(
            f"{classifier_strategy} inference requires the explicit cached hard ROI mask"
        )
    images = batch["image"]
    hard_mask = batch["roi_mask"].bool()
    if images.ndim != 4 or hard_mask.shape != images.shape[:1] + images.shape[-2:]:
        raise ProtocolGateError("Classifier image/hard-mask batch shapes disagree")
    neutral = torch.as_tensor(
        DEFAULT_NEUTRAL_RGB, dtype=images.dtype, device=images.device
    ).view(1, 3, 1, 1)
    strict_images = torch.where(hard_mask.unsqueeze(1), images, neutral)
    geometry = batch.get("geometry") if geometry_features else None
    if classifier_strategy == PRIMARY_CLASSIFIER_STRATEGY:
        return classifier(
            strict_images, geometry=geometry, hard_mask=hard_mask
        )
    return classifier(strict_images, geometry)


def _audit_primary_classifier_trainability(
    cfg: Mapping[str, Any], model_name: str, classifier: torch.nn.Module
) -> dict[str, Any]:
    """Fail closed when observed trainability diverges from the locked policy."""

    primary = cfg["classifier"]["primary"]
    declared = dict(primary["trainability_by_segmenter"][model_name])
    named = list(classifier.named_parameters())
    trainable_ids = {id(parameter) for _, parameter in named if parameter.requires_grad}
    frozen_ids = {id(parameter) for _, parameter in named if not parameter.requires_grad}
    failures: list[str] = []
    if getattr(classifier, "model_family", None) != model_name:
        failures.append("model_family_mismatch")
    if getattr(classifier, "classifier_strategy", None) != PRIMARY_CLASSIFIER_STRATEGY:
        failures.append("classifier_strategy_mismatch")
    expected_trainable_ids: set[int]
    expected_frozen_ids: set[int]
    if model_name == "yolo26":
        native_head = classifier.feature_extractor.model[-1]
        expected_frozen_ids = {id(parameter) for parameter in native_head.parameters()}
        expected_trainable_ids = {id(parameter) for parameter in classifier.parameters()} - expected_frozen_ids
        if not expected_frozen_ids:
            failures.append("native_yolo_head_not_found")
    elif model_name == "sam2_unet":
        expected_trainable_ids = {
            id(parameter)
            for name, parameter in named
            if ".prompt_learn." in name or name.startswith("classification_head.")
        }
        expected_frozen_ids = {id(parameter) for parameter in classifier.parameters()} - expected_trainable_ids
        if not any(".prompt_learn." in name for name, _ in named):
            failures.append("sam_adapter_prompt_learn_not_found")
        if not expected_frozen_ids:
            failures.append("sam_original_hiera_trunk_not_frozen")
    else:
        expected_trainable_ids = {id(parameter) for parameter in classifier.parameters()}
        expected_frozen_ids = set()
    if trainable_ids != expected_trainable_ids:
        failures.append("trainable_parameter_set_mismatch")
    if frozen_ids != expected_frozen_ids:
        failures.append("frozen_parameter_set_mismatch")
    if not bool(primary.get("fitted_segmenter_frozen_during_classifier_training")):
        failures.append("fitted_segmenter_separation_not_declared")
    audit = {
        "status": "passed" if not failures else "failed",
        "classifier_strategy": PRIMARY_CLASSIFIER_STRATEGY,
        "classifier_family": model_name,
        "declared_policy": declared,
        "observed_trainable_parameter_count": int(
            sum(parameter.numel() for _, parameter in named if parameter.requires_grad)
        ),
        "observed_frozen_parameter_count": int(
            sum(parameter.numel() for _, parameter in named if not parameter.requires_grad)
        ),
        "observed_trainable_parameter_tensors": int(len(trainable_ids)),
        "observed_frozen_parameter_tensors": int(len(frozen_ids)),
        "fitted_segmenter_object_reused_by_classifier": False,
        "failures": failures,
    }
    if failures and bool(primary.get("require_parameter_audit_against_trainability_policy")):
        raise ProtocolGateError(
            f"Primary classifier trainability policy failed for {model_name}: {failures}"
        )
    return audit


@torch.inference_mode()
def _infer_classifier_frames(
    cfg: Mapping[str, Any], classifier: torch.nn.Module, roi_index: pd.DataFrame,
    *, geometry_features: int = 0,
    classifier_strategy: str = STANDARDIZED_CLASSIFIER_STRATEGY,
) -> pd.DataFrame:
    dataset = ROICacheDataset(
        roi_index, normalize=False, include_geometry=bool(geometry_features)
    )
    result = roi_index.copy()
    result["probability"] = np.nan
    result["logit_0"] = np.nan
    result["logit_1"] = np.nan
    result["classifier_strategy"] = str(classifier_strategy)
    if not len(dataset):
        return result
    device = next(classifier.parameters()).device
    classifier.eval()
    predictions: dict[tuple[str, str, str, str], tuple[float, float, float]] = {}
    batches = _loader(
        dataset,
        batch_size=int(cfg["classifier"]["batch_size"]),
        num_workers=int(cfg["training"]["num_workers"]),
        shuffle=False,
        seed=0,
    )
    for batch in batches:
        indices = batch["index"].tolist()
        batch = _to_device(batch, device)
        with _amp_context(cfg, device):
            logits = _classifier_forward(
                classifier,
                batch,
                classifier_strategy=classifier_strategy,
                geometry_features=geometry_features,
            )
        logits = logits.float().cpu().numpy()
        probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()[:, 1]
        if not np.isfinite(logits).all() or not np.isfinite(probabilities).all():
            raise FloatingPointError("Non-finite ROI classifier inference output")
        for local, index in enumerate(indices):
            row = dataset.rows.iloc[index]
            key = (
                str(row.patient_id),
                str(row.case_id),
                str(row.side),
                str(row.frame_id),
            )
            if key in predictions:
                raise ProtocolGateError(
                    f"Classifier dataset repeats a full frame identity: {key}"
                )
            predictions[key] = (
                float(probabilities[local]), float(logits[local, 0]), float(logits[local, 1])
            )
    result_valid = strict_boolean_mask(result["roi_valid"], name="roi_valid")
    for position, (index, row) in enumerate(result.iterrows()):
        key = (
            str(row.patient_id),
            str(row.case_id),
            str(row.side),
            str(row.frame_id),
        )
        if bool(result_valid.iloc[position]):
            if key not in predictions:
                raise RuntimeError(f"Missing classifier output for valid ROI {key}")
            result.loc[index, ["probability", "logit_0", "logit_1"]] = predictions[key]
        elif key in predictions:
            raise RuntimeError("Classifier was invoked for an invalid ROI")
    return result


def _classifier_validation_monitor(cfg: Mapping[str, Any], frames: pd.DataFrame) -> dict[str, Any]:
    eyes = aggregate_frames_to_eyes(
        frames,
        threshold=0.5,
        frames_per_eye=int(cfg["dataset"]["frames_per_eye"]),
        min_valid_frames=int(cfg["aggregation"]["minimum_valid_frames"]),
    )
    valid = eyes.evaluable.to_numpy(bool)
    if not valid.any():
        return {
            "monitor_status": "non_evaluable",
            "eye_auroc": np.nan,
            "eye_nll": np.nan,
            "evaluable_eyes": 0,
            "eye_coverage": 0.0,
            "key": (0.0, -math.inf, -math.inf),
        }
    metrics = conditional_binary_metrics(
        eyes.loc[valid, "label"], eyes.loc[valid, "probability"], threshold=0.5
    )
    auroc, nll = float(metrics["auroc"]), float(metrics["nll"])
    if np.isfinite(auroc):
        status = "auroc"
        key = (2.0, auroc, -nll)
    else:
        status = "negative_nll_fallback"
        key = (1.0, -nll, -math.inf)
    return {
        "monitor_status": status,
        "eye_auroc": auroc,
        "eye_nll": nll,
        "evaluable_eyes": int(valid.sum()),
        "eye_coverage": float(valid.mean()),
        "key": key,
    }


def _persist_classifier_training_selection(
    cfg: Mapping[str, Any],
    train_index: pd.DataFrame,
    *,
    raw_index_path: Path,
    destination: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Create one shared, auditable optimization selection for a classifier fit."""

    raw_index_path = raw_index_path.resolve()
    if not raw_index_path.is_file():
        raise ProtocolGateError(f"Classifier raw training ROI index is missing: {raw_index_path}")
    raw_before = dataframe_sha256(train_index)
    optimization_index, eligibility_ledger = build_classifier_training_selection(
        train_index,
        frames_per_eye=int(cfg["dataset"]["frames_per_eye"]),
        minimum_valid_frames=int(cfg["aggregation"]["minimum_valid_frames"]),
    )
    if dataframe_sha256(train_index) != raw_before:
        raise ProtocolGateError("Classifier eligibility derivation mutated the canonical ROI index")

    destination.mkdir(parents=True, exist_ok=True)
    optimization_path = (destination / "training_optimization_index.csv").resolve()
    eligibility_path = (destination / "training_eye_eligibility.csv").resolve()
    expected_optimization_sha = dataframe_sha256(optimization_index)
    expected_eligibility_sha = dataframe_sha256(eligibility_ledger)
    if optimization_path.exists():
        persisted_optimization = pd.read_csv(
            optimization_path, encoding="utf-8-sig", dtype={"frame_id": str}
        )
        if dataframe_sha256(persisted_optimization) != expected_optimization_sha:
            raise ProtocolGateError(
                "Existing classifier optimization index conflicts with canonical OOF ROIs"
            )
    else:
        _save_csv_atomic(optimization_path, optimization_index)
        persisted_optimization = pd.read_csv(
            optimization_path, encoding="utf-8-sig", dtype={"frame_id": str}
        )
    if eligibility_path.exists():
        persisted_eligibility = pd.read_csv(eligibility_path, encoding="utf-8-sig")
        if dataframe_sha256(persisted_eligibility) != expected_eligibility_sha:
            raise ProtocolGateError(
                "Existing classifier eye eligibility ledger conflicts with canonical OOF ROIs"
            )
    else:
        _save_csv_atomic(eligibility_path, eligibility_ledger)
        persisted_eligibility = pd.read_csv(eligibility_path, encoding="utf-8-sig")
    if (
        dataframe_sha256(persisted_optimization) != expected_optimization_sha
        or dataframe_sha256(persisted_eligibility) != expected_eligibility_sha
    ):
        raise ProtocolGateError("Classifier training selection changed during CSV persistence")
    optimization_index = persisted_optimization
    eligibility_ledger = persisted_eligibility
    eligible = strict_boolean_mask(
        eligibility_ledger["eye_training_eligible"], name="eye_training_eligible"
    )
    provenance = {
        "training_roi_index": _relative(raw_index_path),
        "training_roi_index_sha256": sha256_file(raw_index_path),
        "training_roi_dataframe_sha256": raw_before,
        "training_optimization_index": _relative(optimization_path),
        "training_optimization_index_sha256": sha256_file(optimization_path),
        "training_optimization_dataframe_sha256": expected_optimization_sha,
        "training_eye_eligibility": _relative(eligibility_path),
        "training_eye_eligibility_sha256": sha256_file(eligibility_path),
        "training_eye_eligibility_dataframe_sha256": expected_eligibility_sha,
        "frames_per_eye": int(cfg["dataset"]["frames_per_eye"]),
        "minimum_valid_frames": int(cfg["aggregation"]["minimum_valid_frames"]),
        "eligible_training_eye_count": int(eligible.sum()),
        "ineligible_training_eye_count": int((~eligible).sum()),
        "optimization_frame_count": int(len(optimization_index)),
    }
    return optimization_index, eligibility_ledger, provenance


def _verify_classifier_training_selection(
    cfg: Mapping[str, Any],
    train_index: pd.DataFrame,
    optimization_index: pd.DataFrame,
    eligibility_ledger: pd.DataFrame,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed unless a supplied optimization selection is reproducible."""

    expected_optimization, expected_ledger = build_classifier_training_selection(
        train_index,
        frames_per_eye=int(cfg["dataset"]["frames_per_eye"]),
        minimum_valid_frames=int(cfg["aggregation"]["minimum_valid_frames"]),
    )
    hashes = {
        "training_roi_dataframe_sha256": dataframe_sha256(train_index),
        "training_optimization_dataframe_sha256": dataframe_sha256(optimization_index),
        "training_eye_eligibility_dataframe_sha256": dataframe_sha256(eligibility_ledger),
        "frames_per_eye": int(cfg["dataset"]["frames_per_eye"]),
        "minimum_valid_frames": int(cfg["aggregation"]["minimum_valid_frames"]),
    }
    if (
        hashes["training_optimization_dataframe_sha256"]
        != dataframe_sha256(expected_optimization)
        or hashes["training_eye_eligibility_dataframe_sha256"]
        != dataframe_sha256(expected_ledger)
    ):
        raise ProtocolGateError(
            "Classifier optimization rows/eligibility ledger are not reproducible from canonical OOF ROIs"
        )
    for field, observed in hashes.items():
        if provenance.get(field) != observed:
            raise ProtocolGateError(f"Classifier training selection provenance mismatch: {field}")
    for path_field in (
        "training_roi_index",
        "training_optimization_index",
        "training_eye_eligibility",
    ):
        path = resolve_project_path(str(provenance.get(path_field)), must_exist=True)
        if sha256_file(path) != provenance.get(f"{path_field}_sha256"):
            raise ProtocolGateError(
                f"Classifier training selection artifact changed: {path_field}"
            )
    eligible = strict_boolean_mask(
        eligibility_ledger["eye_training_eligible"], name="eye_training_eligible"
    )
    expected_counts = {
        "eligible_training_eye_count": int(eligible.sum()),
        "ineligible_training_eye_count": int((~eligible).sum()),
        "optimization_frame_count": int(len(optimization_index)),
    }
    for field, observed in expected_counts.items():
        if provenance.get(field) != observed:
            raise ProtocolGateError(f"Classifier training selection count mismatch: {field}")
    return {**hashes, **expected_counts}


def _train_roi_classifier_model(
    cfg: Mapping[str, Any], model_name: str, seed: int, train_index: pd.DataFrame,
    validation_index: pd.DataFrame, work_dir: Path, *, geometry_features: int = 0,
    optimization_index: pd.DataFrame,
    eligibility_ledger: pd.DataFrame,
    training_selection_provenance: Mapping[str, Any],
    classifier_input: str = "strict_roi_masked_pixels_no_explicit_geometry_vector",
    freeze_image_encoder: bool = False,
    augmentation: Mapping[str, Any] | None = None,
    classifier_strategy: str = STANDARDIZED_CLASSIFIER_STRATEGY,
    classifier_family: str | None = None,
) -> dict[str, Any]:
    minimum_valid = int(cfg["aggregation"]["minimum_valid_frames"])
    selection_contract = _verify_classifier_training_selection(
        cfg,
        train_index,
        optimization_index,
        eligibility_ledger,
        training_selection_provenance,
    )
    offsets = cfg["training"]["seed_offsets"]
    strategy_spec = _classifier_strategy_specs(cfg)[classifier_strategy]
    strategy_seed_offset = int(strategy_spec.get("seed_offset", 0))
    classifier_augmentation_seed = (
        int(seed) + int(offsets["augmentation"]) + strategy_seed_offset
    )
    train_dataset = ROICacheDataset(
        optimization_index,
        normalize=False,
        include_geometry=bool(geometry_features),
        augmentation=cfg["training"]["augmentation"] if augmentation is None else augmentation,
        seed=classifier_augmentation_seed,
    )
    training_seed = _classifier_training_seed(cfg, seed, classifier_strategy)
    fingerprint = code_fingerprint()["sha256"]
    validation_index_sha256 = dataframe_sha256(validation_index)
    checkpoint_selection_contract = {
        **selection_contract,
        "training_roi_index_sha256": training_selection_provenance[
            "training_roi_index_sha256"
        ],
        "training_optimization_index_sha256": training_selection_provenance[
            "training_optimization_index_sha256"
        ],
        "training_eye_eligibility_sha256": training_selection_provenance[
            "training_eye_eligibility_sha256"
        ],
    }
    checkpoint_identity = {
        "schema": 1,
        "segmenter_condition": model_name,
        "outer_seed": int(seed),
        "classifier_strategy": classifier_strategy,
        "classifier_family": classifier_family,
        "classifier_architecture_id": (
            strategy_spec.get("architecture_by_segmenter", {}).get(classifier_family)
            if classifier_strategy == PRIMARY_CLASSIFIER_STRATEGY
            else strategy_spec.get("architecture", "torchvision_resnet18")
        ),
        "training_seed": training_seed,
        "augmentation_seed": classifier_augmentation_seed,
        "classifier_input": classifier_input,
        "geometry_features": int(geometry_features),
        "image_encoder_frozen": bool(freeze_image_encoder),
        "augmentation": (
            dict(cfg["training"]["augmentation"])
            if augmentation is None
            else dict(augmentation)
        ),
        "training_selection_contract": checkpoint_selection_contract,
        "validation_index_sha256": validation_index_sha256,
        "config_sha256": cfg["_runtime"]["config_sha256"],
        "code_sha256": fingerprint,
    }
    last_path = work_dir / "last.pt"
    selected_path = work_dir / "selected.pt"
    resume_state: Mapping[str, Any] | None = None
    if selected_path.exists() and not last_path.exists():
        raise ProtocolGateError(
            f"Orphan classifier checkpoint without resumable state: {selected_path}"
        )
    if last_path.exists():
        candidate = torch.load(last_path, map_location="cpu", weights_only=False)
        if not isinstance(candidate, Mapping):
            raise ProtocolGateError(f"Malformed classifier resume checkpoint: {last_path}")
        required_resume_fields = {
            "model", "optimizer", "scheduler", "scaler", "epoch", "best_key",
            "best_epoch", "best_monitor_status", "stale", "history", "rng",
            "terminal_non_evaluable", "config_sha256", "code_sha256",
            "checkpoint_identity",
        }
        if not required_resume_fields <= set(candidate):
            raise ProtocolGateError(
                f"Classifier resume checkpoint lacks fields: "
                f"{sorted(required_resume_fields - set(candidate))}"
            )
        if (
            candidate.get("config_sha256") != cfg["_runtime"]["config_sha256"]
            or candidate.get("code_sha256") != fingerprint
        ):
            raise ProtocolGateError(f"Resume code/config mismatch: {last_path}")
        if candidate.get("checkpoint_identity") != checkpoint_identity:
            raise ProtocolGateError(
                f"Resume classifier/input provenance mismatch: {last_path}"
            )
        resumed_best_epoch = int(candidate.get("best_epoch", -1))
        terminal_marker = candidate["terminal_non_evaluable"]
        if not isinstance(terminal_marker, (bool, np.bool_)):
            raise ProtocolGateError(
                f"Classifier resume has a non-boolean terminal marker: {last_path}"
            )
        if bool(terminal_marker) and (
            resumed_best_epoch >= 0 or selected_path.exists()
        ):
            raise ProtocolGateError(
                f"Non-evaluable classifier resume conflicts with a selected state: {last_path}"
            )
        if resumed_best_epoch >= 0:
            if not selected_path.is_file():
                raise ProtocolGateError(
                    f"Classifier resume names a selected epoch but its checkpoint is missing: "
                    f"{selected_path}"
                )
            selected_header = torch.load(
                selected_path, map_location="cpu", weights_only=True
            )
            if not isinstance(selected_header, Mapping) or "model" not in selected_header:
                raise ProtocolGateError(
                    f"Malformed selected classifier checkpoint: {selected_path}"
                )
            if selected_header.get("checkpoint_identity") != checkpoint_identity:
                raise ProtocolGateError(
                    f"Selected classifier checkpoint provenance mismatch: {selected_path}"
                )
            if int(selected_header.get("epoch", -1)) != resumed_best_epoch:
                raise ProtocolGateError(
                    f"Selected classifier epoch conflicts with resumable state: {selected_path}"
                )
        elif selected_path.exists():
            raise ProtocolGateError(
                f"Classifier resume has no selected epoch but a selected checkpoint exists: "
                f"{selected_path}"
            )
        resume_state = candidate

    valid_labels = train_dataset.rows.label.astype(int).unique()
    if len(train_dataset) == 0 or set(valid_labels) != {0, 1}:
        if resume_state is not None or selected_path.exists():
            raise ProtocolGateError(
                "Existing classifier checkpoints conflict with the current untrainable "
                "strict-ROI selection"
            )
        work_dir.mkdir(parents=True, exist_ok=True)
        status_path = work_dir / "classifier_unavailable.json"
        save_json_atomic(
            status_path,
            {
                "available": False,
                "reason": "OOF predicted ROIs do not contain trainable frames from both classes",
                "valid_training_frames": len(train_dataset),
                "valid_training_labels": sorted(int(value) for value in valid_labels),
                "monitor_status": "non_evaluable",
            },
        )
        validation_frames = validation_index.copy()
        validation_frames["segmentation_roi_valid"] = strict_boolean_mask(
            validation_frames["roi_valid"], name="roi_valid"
        ).to_numpy(dtype=bool)
        validation_frames["segmentation_abstention_reason"] = validation_frames[
            "abstention_reason"
        ]
        validation_frames["roi_valid"] = False
        validation_frames["abstention_reason"] = "classifier_not_trainable"
        validation_frames[["probability", "logit_0", "logit_1"]] = np.nan
        validation_frames["classifier_strategy"] = str(classifier_strategy)
        validation_path = work_dir / "validation_frames_raw.csv"
        _save_csv_atomic(validation_path, validation_frames)
        model_info_path = work_dir / "model_info.json"
        save_json_atomic(
            model_info_path,
            {
                "model": model_name,
                "outer_seed": int(seed),
                "segmenter_condition": model_name,
                "classifier_strategy": classifier_strategy,
                "classifier_role": strategy_spec.get("role"),
                "classifier_estimand_id": strategy_spec.get("estimand_id"),
                "classifier_family": classifier_family,
                "architecture": checkpoint_identity["classifier_architecture_id"],
                "available": False,
                "reason": "OOF predicted ROIs do not contain trainable frames from both classes",
                "separate_weights_per_model_seed": True,
                "strict_roi_input": True,
                "training_selection": dict(training_selection_provenance),
                "checkpoint_identity": checkpoint_identity,
            },
        )
        return {
            "available": False,
            "checkpoint": None,
            "history": status_path,
            "validation_frames": validation_path,
            "monitor_status": "non_evaluable",
            "best_epoch": None,
            "model_info": model_info_path,
            "checkpoint_identity": checkpoint_identity,
        }

    device = _device_required()
    work_dir.mkdir(parents=True, exist_ok=True)
    _seed_everything(training_seed)
    classifier = _build_classifier_from_config(
        cfg,
        pretrained=True,
        geometry_features=geometry_features,
        classifier_strategy=classifier_strategy,
        model_name=classifier_family,
    ).to(device)
    if (
        classifier_strategy == PRIMARY_CLASSIFIER_STRATEGY
        and classifier_family is not None
        and not model_name.startswith("ablation:")
    ):
        trainability_audit = _audit_primary_classifier_trainability(
            cfg, classifier_family, classifier
        )
    elif classifier_strategy == PRIMARY_CLASSIFIER_STRATEGY:
        trainability_audit = {
            "status": "ablation_override",
            "classifier_family": classifier_family,
            "reason": "Predeclared geometry/freeze ablations may alter the primary trainability set",
        }
    else:
        trainability_audit = {
            "status": "passed",
            "classifier_strategy": classifier_strategy,
            "policy": "all_resnet18_parameters_trainable",
        }
    if freeze_image_encoder:
        classifier.encoder.requires_grad_(False)
        classifier.encoder.eval()
    model_info_path = work_dir / "model_info.json"
    family_architecture = (
        strategy_spec.get("architecture_by_segmenter", {}).get(classifier_family)
        if classifier_strategy == PRIMARY_CLASSIFIER_STRATEGY
        else strategy_spec.get("architecture", "torchvision_resnet18")
    )
    extra_parameter_info = (
        classifier.parameter_info() if hasattr(classifier, "parameter_info") else {}
    )
    if classifier_strategy == PRIMARY_CLASSIFIER_STRATEGY:
        expected_architecture = MODEL_SPECIFIC_CLASSIFIER_ARCHITECTURE_IDS.get(
            str(classifier_family)
        )
        if (
            extra_parameter_info.get("family") != classifier_family
            or extra_parameter_info.get("strategy") != PRIMARY_CLASSIFIER_STRATEGY
            or extra_parameter_info.get("architecture_id") != expected_architecture
            or family_architecture != expected_architecture
        ):
            raise ProtocolGateError(
                "Model-specific classifier builder does not match the locked family/"
                "architecture contract"
            )
    model_info_record = {
            "model": model_name,
            "outer_seed": int(seed),
            "segmenter_condition": model_name,
            "classifier_strategy": classifier_strategy,
            "classifier_role": strategy_spec.get("role"),
            "classifier_estimand_id": strategy_spec.get("estimand_id"),
            "classifier_family": classifier_family,
            "architecture": family_architecture,
            "primary_features": classifier_input,
            "classifier_mode": (
                "geometry_only_frozen_encoder" if freeze_image_encoder
                else "appearance_plus_explicit_geometry" if geometry_features
                else "strict_roi_masked_pixels_no_explicit_geometry_vector"
            ),
            "geometry_features": int(geometry_features),
            "image_encoder_frozen": bool(freeze_image_encoder),
            **parameter_inventory(
                classifier,
                explicitly_disabled_legacy_parameters=int(
                    getattr(classifier, "explicitly_disabled_parameters", 0)
                ),
            ),
            "parameters": sum(parameter.numel() for parameter in classifier.parameters()),
            "trainable_parameters": sum(
                parameter.numel() for parameter in classifier.parameters() if parameter.requires_grad
            ),
            "flops_status": "not_reported",
            "flops_rationale": FLOPS_RATIONALE,
            "pretrained_provenance": classifier.pretrained_provenance,
            "family_parameter_info": extra_parameter_info,
            "trainability_policy_audit": trainability_audit,
            "normalization_owned_by_model": True,
            "shared_architecture_across_all_segmenters": bool(
                classifier_strategy == STANDARDIZED_CLASSIFIER_STRATEGY
            ),
            "separate_weights_per_model_seed": True,
            "minimum_valid_training_frames_per_eye": minimum_valid,
            "eligible_training_eyes": int(
                training_selection_provenance["eligible_training_eye_count"]
            ),
            "ineligible_training_eyes": int(
                training_selection_provenance["ineligible_training_eye_count"]
            ),
            "optimization_training_frames": int(
                training_selection_provenance["optimization_frame_count"]
            ),
            "training_selection": dict(training_selection_provenance),
            "checkpoint_identity": checkpoint_identity,
            "augmentation": cfg["training"]["augmentation"] if augmentation is None else augmentation,
            "augmentation_stage": "after strict hard masking; neutral background restored",
            "augmentation_seed": classifier_augmentation_seed,
    }
    optimizer = torch.optim.AdamW(
        (parameter for parameter in classifier.parameters() if parameter.requires_grad),
        lr=float(cfg["classifier"]["learning_rate"]),
        weight_decay=float(cfg["classifier"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(cfg["training"]["max_epochs"])
    )
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    # ROICacheDataset filters invalid rows and then applies a stable sort.  Build
    # the weights from that exact optimization order so ``batch["index"]`` can
    # never address a weight belonging to a different frame.
    sample_weights = torch.as_tensor(
        classifier_loss_weights(train_dataset.rows), dtype=torch.float32, device=device
    )
    if int(cfg["classifier"]["gradient_accumulation"]) != 1:
        raise ValueError("The locked classifier weighting proof requires gradient_accumulation=1")
    max_epochs = int(cfg["training"]["max_epochs"])
    patience = int(cfg["training"]["patience"])
    best_key = (-math.inf, -math.inf, -math.inf)
    best_epoch = -1
    best_monitor_status = "non_evaluable"
    stale = 0
    terminal_non_evaluable = False
    first_epoch = 0
    history: list[dict[str, Any]] = []
    if resume_state is not None:
        state = resume_state
        classifier.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        first_epoch = int(state["epoch"]) + 1
        best_key = tuple(float(value) for value in state["best_key"])
        best_epoch = int(state["best_epoch"])
        best_monitor_status = str(state["best_monitor_status"])
        stale = int(state["stale"])
        terminal_non_evaluable = bool(state["terminal_non_evaluable"])
        history = list(state["history"])
        _restore_rng_state(state["rng"])
    save_json_atomic(model_info_path, model_info_record)

    for epoch in range(first_epoch, max_epochs):
        if stale >= patience or terminal_non_evaluable:
            break
        classifier.train()
        if freeze_image_encoder:
            classifier.encoder.eval()
        train_dataset.epoch = epoch
        batches = _loader(
            train_dataset,
            batch_size=int(cfg["classifier"]["batch_size"]),
            num_workers=int(cfg["training"]["num_workers"]),
            shuffle=True,
            seed=training_seed + epoch,
        )
        optimizer.zero_grad(set_to_none=True)
        loss_sum, weight_sum = 0.0, 0.0
        started = time.perf_counter()
        for step, batch in enumerate(batches):
            batch = _to_device(batch, device)
            accumulation = int(cfg["classifier"]["gradient_accumulation"])
            group_start = (step // accumulation) * accumulation
            denominator = min(accumulation, len(batches) - group_start)
            with _amp_context(cfg, device):
                logits = _classifier_forward(
                    classifier,
                    batch,
                    classifier_strategy=classifier_strategy,
                    geometry_features=geometry_features,
                )
                losses = F.cross_entropy(logits.float(), batch["label"], reduction="none")
                weights = sample_weights[batch["index"]]
                # Global sample weights have mean one.  A fixed denominator
                # (also for the final short batch) makes every frame's planned
                # contribution independent of random minibatch composition.
                loss = (losses * weights).sum() / float(cfg["classifier"]["batch_size"])
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite ROI classifier loss")
            scaler.scale(loss / denominator).backward()
            loss_sum += float((losses.detach() * weights).sum())
            weight_sum += float(weights.sum())
            if (step + 1) % accumulation == 0 or step + 1 == len(batches):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    classifier.parameters(),
                    float(cfg["training"]["gradient_clip"]),
                    error_if_nonfinite=True,
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        validation_frames = _infer_classifier_frames(
            cfg,
            classifier,
            validation_index,
            geometry_features=geometry_features,
            classifier_strategy=classifier_strategy,
        )
        monitor = _classifier_validation_monitor(cfg, validation_frames)
        key = tuple(float(value) for value in monitor.pop("key"))
        terminal_non_evaluable = monitor["monitor_status"] == "non_evaluable"
        if not terminal_non_evaluable and key > best_key:
            best_key, best_epoch, stale = key, epoch, 0
            best_monitor_status = str(monitor["monitor_status"])
            _torch_save_atomic(
                selected_path,
                {
                    "model": classifier.state_dict(),
                    "epoch": epoch,
                    "validation_monitor": monitor,
                    "segmenter_condition": model_name,
                    "outer_seed": int(seed),
                    "classifier_strategy": classifier_strategy,
                    "classifier_family": classifier_family,
                    "training_selection_contract": checkpoint_selection_contract,
                    "checkpoint_identity": checkpoint_identity,
                },
            )
        else:
            stale += 1
        scheduler.step()
        event = {
            "epoch": epoch + 1,
            "train_weighted_loss": loss_sum / max(weight_sum, 1e-12),
            **monitor,
            "seconds": time.perf_counter() - started,
            "learning_rate_after_step": scheduler.get_last_lr()[0],
        }
        history.append(event)
        _torch_save_atomic(
            last_path,
            {
                "model": classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best_key": best_key,
                "best_epoch": best_epoch,
                "best_monitor_status": best_monitor_status,
                "stale": stale,
                "terminal_non_evaluable": terminal_non_evaluable,
                "history": history,
                "rng": _rng_state(),
                "config_sha256": cfg["_runtime"]["config_sha256"],
                "code_sha256": fingerprint,
                "classifier_strategy": classifier_strategy,
                "classifier_family": classifier_family,
                "training_selection_contract": checkpoint_selection_contract,
                "validation_index_sha256": validation_index_sha256,
                "checkpoint_identity": checkpoint_identity,
            },
        )
        _save_csv_atomic(work_dir / "history.csv", pd.DataFrame(history))
        _write_progress(
            cfg,
            phase="classifier_training",
            model=model_name,
            seed=seed,
            epoch=epoch + 1,
            best_epoch=best_epoch + 1,
            stale_epochs=stale,
            validation=monitor,
        )
        if terminal_non_evaluable:
            # The locked protocol forbids selecting a diagnostic checkpoint
            # without any evaluable validation eye.
            break
    if not selected_path.is_file():
        _save_csv_atomic(work_dir / "history.csv", pd.DataFrame(history))
        status_path = work_dir / "classifier_unavailable.json"
        save_json_atomic(
            status_path,
            {
                "available": False,
                "reason": "No evaluable validation eye; early stopping monitor undefined",
                "monitor_status": "non_evaluable",
            },
        )
        validation_frames = validation_index.copy()
        validation_frames["segmentation_roi_valid"] = strict_boolean_mask(
            validation_frames["roi_valid"], name="roi_valid"
        ).to_numpy(dtype=bool)
        validation_frames["segmentation_abstention_reason"] = validation_frames[
            "abstention_reason"
        ]
        validation_frames["roi_valid"] = False
        validation_frames["abstention_reason"] = "classifier_validation_non_evaluable"
        validation_frames[["probability", "logit_0", "logit_1"]] = np.nan
        validation_frames["classifier_strategy"] = str(classifier_strategy)
        validation_path = work_dir / "validation_frames_raw.csv"
        _save_csv_atomic(validation_path, validation_frames)
        return {
            "available": False,
            "checkpoint": None,
            "history": work_dir / "history.csv",
            "validation_frames": validation_path,
            "monitor_status": "non_evaluable",
            "best_epoch": None,
            "model_info": model_info_path,
            "checkpoint_identity": checkpoint_identity,
        }
    selected = torch.load(selected_path, map_location="cpu", weights_only=True)
    if selected.get("checkpoint_identity") != checkpoint_identity:
        raise ProtocolGateError(f"Selected classifier checkpoint provenance mismatch: {selected_path}")
    if int(selected.get("epoch", -1)) != int(best_epoch):
        raise ProtocolGateError(
            f"Selected classifier epoch conflicts with training state: {selected_path}"
        )
    classifier.load_state_dict(selected["model"], strict=True)
    validation_frames = _infer_classifier_frames(
        cfg,
        classifier,
        validation_index,
        geometry_features=geometry_features,
        classifier_strategy=classifier_strategy,
    )
    validation_path = work_dir / "validation_frames_raw.csv"
    _save_csv_atomic(validation_path, validation_frames)
    return {
        "available": True,
        "checkpoint": selected_path,
        "history": work_dir / "history.csv",
        "validation_frames": validation_path,
        "monitor_status": best_monitor_status,
        "best_epoch": best_epoch,
        "model_info": model_info_path,
        "checkpoint_identity": checkpoint_identity,
    }


def train_classifier(cfg: Mapping[str, Any], model: str, seed: int) -> dict[str, Any]:
    """Fit both locked classifier strategies from the identical ROI caches."""

    assert_prerequisites(cfg, "train-classifiers", model=model, seed=seed)
    existing = _completed_stage_receipt_or_none(
        cfg, "train-classifiers", model=model, seed=seed
    )
    if existing is not None:
        return {"status": "already_complete", "receipt": existing}
    run = unit_root(cfg, model, seed)
    _, train_rows = split_frames(cfg, seed, "train")
    _, validation_rows = split_frames(cfg, seed, "validation")
    train_index_path = run / "roi" / "train_oof_index.csv"
    validation_index_path = run / "roi" / "validation" / "index.csv"
    train_index = _verify_roi_index(cfg, train_index_path, train_rows)
    validation_index = _verify_roi_index(cfg, validation_index_path, validation_rows)
    optimization_index, eligibility_ledger, selection_provenance = (
        _persist_classifier_training_selection(
            cfg,
            train_index,
            raw_index_path=train_index_path,
            destination=run / "classifiers" / "shared",
        )
    )
    optimization_index_path = resolve_project_path(
        selection_provenance["training_optimization_index"], must_exist=True
    )
    eligibility_ledger_path = resolve_project_path(
        selection_provenance["training_eye_eligibility"], must_exist=True
    )
    artifacts: list[Path] = [
        train_index_path,
        validation_index_path,
        optimization_index_path,
        eligibility_ledger_path,
    ]
    branch_manifest: dict[str, Any] = {}
    specs = _classifier_strategy_specs(cfg)
    for strategy in _classifier_strategy_names(cfg):
        work_dir = _classifier_work_dir(run, strategy)
        result = _train_roi_classifier_model(
            cfg,
            model,
            seed,
            train_index,
            validation_index,
            work_dir,
            optimization_index=optimization_index,
            eligibility_ledger=eligibility_ledger,
            training_selection_provenance=selection_provenance,
            classifier_strategy=strategy,
            classifier_family=model,
        )
        history_path = Path(result["history"])
        validation_frames_path = Path(result["validation_frames"])
        model_info_path = Path(result["model_info"])
        checkpoint_path = Path(result["checkpoint"]) if result["checkpoint"] else None
        status_path = work_dir / "training_result.json"
        save_json_atomic(
            status_path,
            {
                "available": bool(result["available"]),
                "model": model,
                "seed": int(seed),
                "classifier_strategy": strategy,
                "classifier_role": specs[strategy].get("role"),
                "classifier_estimand_id": specs[strategy].get("estimand_id"),
                "classifier_family": model,
                "monitor_status": result["monitor_status"],
                "best_epoch_zero_based": result["best_epoch"],
                "checkpoint": _relative(checkpoint_path) if checkpoint_path else None,
                "checkpoint_sha256": sha256_file(checkpoint_path) if checkpoint_path else None,
                "checkpoint_identity": result["checkpoint_identity"],
                "history": _relative(history_path),
                "history_sha256": sha256_file(history_path),
                "validation_frames": _relative(validation_frames_path),
                "validation_frames_sha256": sha256_file(validation_frames_path),
                "model_info": _relative(model_info_path),
                "model_info_sha256": sha256_file(model_info_path),
                **selection_provenance,
                "validation_roi_index": _relative(validation_index_path),
                "validation_roi_index_sha256": sha256_file(validation_index_path),
                "training_roi_source": "patient-grouped five-fold OOF predicted masks",
                "validation_roi_source": "outer full-fit predicted masks",
                "classifier_input": "hard-masked tight ROI only",
                "full_image_or_gt_fallback": False,
                "separate_weights_per_model_seed": True,
                "test_used": False,
            },
        )
        strategy_artifacts = [
            history_path,
            validation_frames_path,
            model_info_path,
            status_path,
        ]
        if checkpoint_path is not None:
            strategy_artifacts.append(checkpoint_path)
        artifacts.extend(strategy_artifacts)
        branch_manifest[strategy] = {
            "model": model,
            "seed": int(seed),
            "classifier_strategy": strategy,
            "classifier_family": model,
            "role": specs[strategy].get("role"),
            "estimand_id": specs[strategy].get("estimand_id"),
            "training_result": _relative(status_path),
            "training_result_sha256": sha256_file(status_path),
            "checkpoint_identity": result["checkpoint_identity"],
            "training_optimization_index": selection_provenance[
                "training_optimization_index"
            ],
            "training_optimization_index_sha256": selection_provenance[
                "training_optimization_index_sha256"
            ],
            "training_eye_eligibility": selection_provenance[
                "training_eye_eligibility"
            ],
            "training_eye_eligibility_sha256": selection_provenance[
                "training_eye_eligibility_sha256"
            ],
            "artifacts": [
                {"path": _relative(path), "sha256": sha256_file(path)}
                for path in strategy_artifacts
            ],
        }
    manifest_path = run / "classifiers" / "training_manifest.json"
    save_json_atomic(
        manifest_path,
        {
            "schema": 2,
            "model": model,
            "seed": int(seed),
            "primary_classifier_strategy": PRIMARY_CLASSIFIER_STRATEGY,
            "secondary_classifier_strategy": STANDARDIZED_CLASSIFIER_STRATEGY,
            "identical_oof_predicted_roi_inputs": True,
            "identical_training_optimization_index": True,
            "identical_training_eye_eligibility_ledger": True,
            **selection_provenance,
            "validation_roi_index": _relative(validation_index_path),
            "validation_roi_index_sha256": sha256_file(validation_index_path),
            "strategies": branch_manifest,
            "test_used": False,
        },
    )
    artifacts.insert(0, manifest_path)
    receipt = write_stage_receipt(
        cfg,
        "train-classifiers",
        model=model,
        seed=seed,
        artifacts=artifacts,
        metadata={
            "primary_classifier_strategy": PRIMARY_CLASSIFIER_STRATEGY,
            "classifier_strategies": list(_classifier_strategy_names(cfg)),
            "classifier_strategy_count": 2,
            "identical_oof_predicted_roi_inputs": True,
            "identical_training_optimization_index": True,
            "identical_training_eye_eligibility_ledger": True,
            "training_optimization_index_sha256": selection_provenance[
                "training_optimization_index_sha256"
            ],
            "training_eye_eligibility_sha256": selection_provenance[
                "training_eye_eligibility_sha256"
            ],
            "strategies": {
                name: {
                    "model": model,
                    "seed": int(seed),
                    "classifier_strategy": name,
                    "classifier_family": model,
                    "role": specs[name].get("role"),
                    "estimand_id": specs[name].get("estimand_id"),
                    "available": read_json(
                        _classifier_work_dir(run, name) / "training_result.json"
                    )["available"],
                    "monitor_status": read_json(
                        _classifier_work_dir(run, name) / "training_result.json"
                    )["monitor_status"],
                }
                for name in _classifier_strategy_names(cfg)
            },
            "test_images_loaded": False,
        },
    )
    return {"status": "complete", "receipt": str(receipt), "artifacts": [str(path) for path in artifacts]}


def _lock_calibration_and_threshold(
    cfg: Mapping[str, Any], table: pd.DataFrame, *, level: str
) -> dict[str, Any]:
    valid = table.evaluable.to_numpy(bool)
    labels = table.loc[valid, "label"].astype(int).to_numpy()
    probabilities = table.loc[valid, "probability"].astype(float).to_numpy()
    class_count = int(np.unique(labels).size) if len(labels) else 0
    if class_count < 2:
        return {
            "level": level,
            "status": "unavailable",
            "classification_threshold_status": "unavailable",
            "calibration_status": "unavailable",
            "reason": "no_evaluable_units" if class_count == 0 else "one_evaluable_class_only",
            "evaluable_units": int(valid.sum()),
            "evaluable_class_count": class_count,
            "temperature": None,
            "temperature_fit": None,
            "threshold": None,
            "threshold_selection": None,
        }
    epsilon = float(cfg["calibration"]["probability_clip_epsilon"])
    minimum_temperature, maximum_temperature = (
        float(value) for value in cfg["calibration"]["temperature_bounds"]
    )
    temperature_fit = fit_temperature_on_probabilities(
        labels,
        probabilities,
        epsilon=epsilon,
        min_temperature=minimum_temperature,
        max_temperature=maximum_temperature,
    )
    calibration_ok = str(temperature_fit["status"]).startswith("ok") and np.isfinite(
        temperature_fit["temperature"]
    )
    if calibration_ok:
        temperature = float(temperature_fit["temperature"])
        threshold_probabilities = apply_temperature_scaling(
            special.logit(np.clip(probabilities, epsilon, 1 - epsilon)), temperature
        )
        threshold_scale = "temperature_scaled"
    else:
        temperature = None
        threshold_probabilities = probabilities
        threshold_scale = "raw_due_to_calibration_unavailable"
    selection = select_binary_threshold(
        labels,
        threshold_probabilities,
        grid=cfg["classifier"]["threshold_candidates"],
        objective=cfg["classifier"]["threshold_selection_metric"],
    )
    return {
        "level": level,
        "status": "locked" if calibration_ok else "calibration_unavailable",
        "classification_threshold_status": "locked",
        "calibration_status": "locked" if calibration_ok else "unavailable",
        "reason": None if calibration_ok else str(temperature_fit["status"]),
        "evaluable_units": int(valid.sum()),
        "evaluable_class_count": class_count,
        "temperature": temperature,
        "temperature_fit": temperature_fit,
        "threshold": float(selection["threshold"]),
        "threshold_probability_scale": threshold_scale,
        "threshold_selection": selection,
    }


def _load_classifier_checkpoint(
    cfg: Mapping[str, Any], checkpoint_path: Path, *, geometry_features: int = 0,
    freeze_image_encoder: bool = False,
    classifier_strategy: str = STANDARDIZED_CLASSIFIER_STRATEGY,
    classifier_family: str | None = None,
) -> torch.nn.Module:
    classifier = _build_classifier_from_config(
        cfg,
        pretrained=False,
        geometry_features=geometry_features,
        classifier_strategy=classifier_strategy,
        model_name=classifier_family,
    )
    if freeze_image_encoder:
        classifier.encoder.requires_grad_(False)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    classifier.load_state_dict(payload["model"], strict=True)
    return classifier.to(_device_required()).eval()


@torch.inference_mode()
def _outside_roi_invariance_audit(
    cfg: Mapping[str, Any], classifier: torch.nn.Module, validation_index: pd.DataFrame,
    *, classifier_strategy: str = STANDARDIZED_CLASSIFIER_STRATEGY,
    geometry_features: int = 0,
) -> dict[str, Any]:
    validity = strict_boolean_mask(validation_index["roi_valid"], name="roi_valid")
    valid = validation_index.loc[validity]
    tolerance = float(
        cfg["roi"]["anti_leakage_invariance_test"]["maximum_absolute_logit_difference"]
    )
    if valid.empty:
        return {
            "status": "no_valid_validation_roi",
            "passed": None,
            "maximum_absolute_logit_difference": None,
            "tolerance": tolerance,
        }
    row = valid.iloc[0]
    cache_path = resolve_project_path(row.cache_path, must_exist=True)
    with np.load(cache_path, allow_pickle=False) as cached:
        source_mask = torch.from_numpy(cached["source_mask"].astype(bool, copy=False).copy())
    dataset_root, manifest = read_manifest(cfg)
    matched = manifest[
        (manifest.patient_id.astype(str) == str(row.patient_id))
        & (manifest.case_id.astype(str) == str(row.case_id))
        & (manifest.side.astype(str) == str(row.side))
        & (manifest.frame_id.astype(str) == str(row.frame_id))
    ]
    if len(matched) != 1:
        raise RuntimeError("Could not recover the validation source image for invariance audit")
    with Image.open(dataset_root / str(matched.iloc[0].output_image)) as handle:
        source = torch.from_numpy(
            np.asarray(handle.convert("RGB"), dtype=np.float32).transpose(2, 0, 1).copy() / 255.0
        )
    generator = torch.Generator().manual_seed(20260909)
    randomized = torch.rand(source.shape, generator=generator, dtype=source.dtype)
    mutated = torch.where(source_mask.unsqueeze(0), source, randomized)
    first = extract_roi_tensor(
        source,
        source_mask,
        target_size=cfg["roi"]["input_size"],
        neutral=DEFAULT_NEUTRAL_RGB,
        return_metadata=True,
    )
    second = extract_roi_tensor(
        mutated,
        source_mask,
        target_size=cfg["roi"]["input_size"],
        neutral=DEFAULT_NEUTRAL_RGB,
        return_metadata=True,
    )
    tensor_equal = bool(torch.equal(first.tensor, second.tensor))
    mask_equal = bool(torch.equal(first.output_mask, second.output_mask))
    geometry_equal = bool(torch.equal(first.geometry, second.geometry))
    device = next(classifier.parameters()).device
    first_batch = {
        "image": first.tensor.unsqueeze(0).to(device),
        "roi_mask": first.output_mask.unsqueeze(0).to(device),
        "geometry": first.geometry.unsqueeze(0).to(device),
    }
    second_batch = {
        "image": second.tensor.unsqueeze(0).to(device),
        "roi_mask": second.output_mask.unsqueeze(0).to(device),
        "geometry": second.geometry.unsqueeze(0).to(device),
    }
    logits_a = _classifier_forward(
        classifier,
        first_batch,
        classifier_strategy=classifier_strategy,
        geometry_features=geometry_features,
    ).float()
    logits_b = _classifier_forward(
        classifier,
        second_batch,
        classifier_strategy=classifier_strategy,
        geometry_features=geometry_features,
    ).float()
    maximum = float((logits_a - logits_b).abs().max().cpu())
    return {
        "status": "tested",
        "passed": bool(tensor_equal and mask_equal and geometry_equal and maximum <= tolerance),
        "classifier_strategy": classifier_strategy,
        "roi_tensor_exactly_equal": tensor_equal,
        "roi_hard_mask_exactly_equal": mask_equal,
        "roi_geometry_exactly_equal": geometry_equal,
        "maximum_absolute_logit_difference": maximum,
        "tolerance": tolerance,
        "sample": {
            "patient_id": str(row.patient_id),
            "case_id": str(row.case_id),
            "frame_id": str(row.frame_id),
        },
        "operation": "randomized every source pixel outside the fixed predicted hard mask",
    }


def _verified_classifier_training_result(
    cfg: Mapping[str, Any], run: Path, model: str, seed: int, strategy: str
) -> tuple[dict[str, Any], dict[str, Path | None]]:
    """Verify the branch manifest before it can contribute to a lock."""

    path = _classifier_work_dir(run, strategy) / "training_result.json"
    value = read_json(path)
    if (
        value.get("model") != model
        or value.get("seed") != int(seed)
        or value.get("classifier_strategy") != strategy
        or value.get("classifier_family") != model
    ):
        raise ProtocolGateError(f"Classifier training identity mismatch: {path}")
    paths: dict[str, Path | None] = {"training_result": path.resolve()}
    for name in ("history", "validation_frames", "model_info"):
        artifact = resolve_project_path(value[name], must_exist=True)
        if sha256_file(artifact) != value[f"{name}_sha256"]:
            raise ProtocolGateError(f"Classifier {name} changed before validation lock: {artifact}")
        paths[name] = artifact
    checkpoint_identity = value.get("checkpoint_identity")
    if not isinstance(checkpoint_identity, Mapping):
        raise ProtocolGateError(f"Classifier training result lacks checkpoint identity: {path}")
    expected_identity = {
        "segmenter_condition": model,
        "outer_seed": int(seed),
        "classifier_strategy": strategy,
        "classifier_family": model,
        "config_sha256": cfg["_runtime"]["config_sha256"],
        "code_sha256": code_fingerprint()["sha256"],
    }
    if any(checkpoint_identity.get(key) != expected for key, expected in expected_identity.items()):
        raise ProtocolGateError(f"Classifier checkpoint identity conflicts with its branch: {path}")
    model_info_path = paths["model_info"]
    assert model_info_path is not None
    if read_json(model_info_path).get("checkpoint_identity") != dict(checkpoint_identity):
        raise ProtocolGateError(f"Classifier model provenance identity mismatch: {model_info_path}")
    for name in (
        "training_roi_index",
        "training_optimization_index",
        "training_eye_eligibility",
        "validation_roi_index",
    ):
        artifact = resolve_project_path(value[name], must_exist=True)
        if sha256_file(artifact) != value[f"{name}_sha256"]:
            raise ProtocolGateError(f"Classifier ROI source changed before validation lock: {artifact}")
        paths[name] = artifact
    raw_training_path = paths["training_roi_index"]
    optimization_path = paths["training_optimization_index"]
    eligibility_path = paths["training_eye_eligibility"]
    assert raw_training_path is not None
    assert optimization_path is not None
    assert eligibility_path is not None
    _, expected_training_rows = split_frames(cfg, seed, "train")
    _, expected_validation_rows = split_frames(cfg, seed, "validation")
    raw_training = _verify_roi_index(cfg, raw_training_path, expected_training_rows)
    validation_roi_path = paths["validation_roi_index"]
    assert validation_roi_path is not None
    _verify_roi_index(cfg, validation_roi_path, expected_validation_rows)
    optimization = pd.read_csv(
        optimization_path, encoding="utf-8-sig", dtype={"frame_id": str}
    )
    eligibility = pd.read_csv(eligibility_path, encoding="utf-8-sig")
    _verify_classifier_training_selection(
        cfg, raw_training, optimization, eligibility, value
    )
    checkpoint: Path | None = None
    if value.get("checkpoint") is not None:
        checkpoint = resolve_project_path(value["checkpoint"], must_exist=True)
        if sha256_file(checkpoint) != value.get("checkpoint_sha256"):
            raise ProtocolGateError(f"Classifier checkpoint changed before validation lock: {checkpoint}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload.get("checkpoint_identity") != dict(checkpoint_identity):
            raise ProtocolGateError(f"Classifier checkpoint payload identity mismatch: {checkpoint}")
    if bool(value.get("available")) != bool(checkpoint is not None):
        raise ProtocolGateError(f"Classifier availability/checkpoint mismatch: {path}")
    paths["checkpoint"] = checkpoint
    return value, paths


def _lock_one_classifier_strategy(
    cfg: Mapping[str, Any], model: str, seed: int, run: Path, strategy: str,
    *, roi_grid_status: str,
) -> tuple[dict[str, Any], list[Path]]:
    training, paths = _verified_classifier_training_result(
        cfg, run, model, seed, strategy
    )
    validation_frames_path = paths["validation_frames"]
    assert validation_frames_path is not None
    validation_frames = pd.read_csv(
        validation_frames_path, encoding="utf-8-sig", dtype={"frame_id": str}
    )
    if "classifier_strategy" not in validation_frames or not (
        validation_frames["classifier_strategy"].astype(str) == strategy
    ).all():
        raise ProtocolGateError(f"Validation frames are not uniquely labeled {strategy}")
    eyes = aggregate_frames_to_eyes(
        validation_frames,
        threshold=0.5,
        frames_per_eye=int(cfg["dataset"]["frames_per_eye"]),
        min_valid_frames=int(cfg["aggregation"]["minimum_valid_frames"]),
    )
    patients = aggregate_eyes_to_patients(eyes.copy(), threshold=0.5)
    eye_lock = _lock_calibration_and_threshold(cfg, eyes, level="eye")
    patient_lock = _lock_calibration_and_threshold(cfg, patients, level="patient")
    evaluable_eye_count = int(eyes.evaluable.astype(bool).sum())
    evaluable_eye_class_count = int(
        eyes.loc[eyes.evaluable.astype(bool), "label"].nunique()
    )
    expected_monitor = {0: "non_evaluable", 1: "negative_nll_fallback", 2: "auroc"}.get(
        evaluable_eye_class_count
    )
    monitor_status = str(training["monitor_status"])
    if expected_monitor is None or monitor_status != expected_monitor:
        raise ProtocolGateError(
            f"{strategy} monitor status {monitor_status!r} conflicts with validation data "
            f"({expected_monitor!r})"
        )
    checkpoint_path = paths["checkpoint"]
    classifier: torch.nn.Module | None = None
    if checkpoint_path is not None:
        classifier = _load_classifier_checkpoint(
            cfg,
            checkpoint_path,
            classifier_strategy=strategy,
            classifier_family=model,
        )
        _, expected_validation_rows = split_frames(cfg, seed, "validation")
        validation_index = _verify_roi_index(
            cfg,
            run / "roi" / "validation" / "index.csv",
            expected_validation_rows,
        )
        invariance = _outside_roi_invariance_audit(
            cfg,
            classifier,
            validation_index,
            classifier_strategy=strategy,
        )
        if invariance["passed"] is False:
            raise ProtocolGateError(f"Outside-ROI invariance audit failed for {strategy}")
    else:
        invariance = {
            "status": "classifier_unavailable",
            "passed": None,
            "classifier_strategy": strategy,
            "reason": "No trainable/validation-selectable classifier checkpoint",
        }
    invariance_path = run / "lock" / f"outside_roi_invariance_{strategy}.json"
    save_json_atomic(invariance_path, invariance)
    operational = bool(
        checkpoint_path is not None
        and eye_lock["classification_threshold_status"] == "locked"
    )
    calibrated = bool(operational and eye_lock["calibration_status"] == "locked")
    status_reasons = {
        "classification_threshold_status": (
            str(eye_lock["reason"])
            if eye_lock["classification_threshold_status"] == "unavailable"
            else None
        ),
        "calibration_status": (
            str(eye_lock["reason"])
            if eye_lock["calibration_status"] == "unavailable"
            else None
        ),
        "roi_grid_status": (
            "minimum validation eye coverage was infeasible"
            if roi_grid_status == "infeasible_fallback" else None
        ),
        "eye_level": eye_lock["reason"],
        "patient_level": patient_lock["reason"],
    }
    specs = _classifier_strategy_specs(cfg)
    record = {
        "model": model,
        "seed": int(seed),
        "classifier_strategy": strategy,
        "classifier_role": specs[strategy].get("role"),
        "classifier_estimand_id": specs[strategy].get("estimand_id"),
        "classifier_family": model,
        "classifier_checkpoint": _relative(checkpoint_path) if checkpoint_path else None,
        "classifier_checkpoint_sha256": sha256_file(checkpoint_path) if checkpoint_path else None,
        "classifier_checkpoint_identity": training["checkpoint_identity"],
        "classifier_history": _relative(paths["history"]),
        "classifier_history_sha256": sha256_file(paths["history"]),
        "classifier_provenance": _relative(paths["model_info"]),
        "classifier_provenance_sha256": sha256_file(paths["model_info"]),
        "classifier_training_result": _relative(paths["training_result"]),
        "classifier_training_result_sha256": sha256_file(paths["training_result"]),
        "training_roi_index": training["training_roi_index"],
        "training_roi_index_sha256": training["training_roi_index_sha256"],
        "training_roi_dataframe_sha256": training[
            "training_roi_dataframe_sha256"
        ],
        "training_optimization_index": training["training_optimization_index"],
        "training_optimization_index_sha256": training[
            "training_optimization_index_sha256"
        ],
        "training_optimization_dataframe_sha256": training[
            "training_optimization_dataframe_sha256"
        ],
        "training_eye_eligibility": training["training_eye_eligibility"],
        "training_eye_eligibility_sha256": training[
            "training_eye_eligibility_sha256"
        ],
        "training_eye_eligibility_dataframe_sha256": training[
            "training_eye_eligibility_dataframe_sha256"
        ],
        "frames_per_eye": training["frames_per_eye"],
        "minimum_valid_frames": training["minimum_valid_frames"],
        "eligible_training_eye_count": training["eligible_training_eye_count"],
        "ineligible_training_eye_count": training["ineligible_training_eye_count"],
        "optimization_frame_count": training["optimization_frame_count"],
        "validation_frames": _relative(validation_frames_path),
        "validation_frames_sha256": sha256_file(validation_frames_path),
        "monitor_status": monitor_status,
        "eye": eye_lock,
        "patient": patient_lock,
        "outside_roi_invariance": invariance,
        "outside_roi_invariance_path": _relative(invariance_path),
        "outside_roi_invariance_sha256": sha256_file(invariance_path),
        "classification_threshold_status": eye_lock["classification_threshold_status"],
        "calibration_status": eye_lock["calibration_status"],
        "operational_system_evaluable": operational,
        "calibrated_system_evaluable": calibrated,
        "validation_evaluable_eye_count": evaluable_eye_count,
        "validation_evaluable_class_count": evaluable_eye_class_count,
        "status_reasons": status_reasons,
        "test_used": False,
    }
    artifacts = [
        path for path in paths.values() if path is not None
    ] + [invariance_path.resolve()]
    del classifier
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return record, list(dict.fromkeys(Path(path).resolve() for path in artifacts))


def lock_validation(cfg: Mapping[str, Any], model: str, seed: int) -> dict[str, Any]:
    """Independently lock both classifier branches without test access."""

    assert_prerequisites(cfg, "lock", model=model, seed=seed)
    existing = _completed_stage_receipt_or_none(cfg, "lock", model=model, seed=seed)
    if existing is not None:
        return {"status": "already_complete", "receipt": existing}
    run = unit_root(cfg, model, seed)
    segmenter_lock_path = run / "segmenter" / "segmenter_lock.json"
    segmenter_lock = read_json(segmenter_lock_path)
    roi_grid_status = (
        "feasible"
        if segmenter_lock["roi_policy_selection"]["coverage_constraint_satisfied"]
        else "infeasible_fallback"
    )
    strategy_locks: dict[str, dict[str, Any]] = {}
    classifier_dependencies: list[Path] = []
    for strategy in _classifier_strategy_names(cfg):
        strategy_lock, strategy_artifacts = _lock_one_classifier_strategy(
            cfg, model, seed, run, strategy, roi_grid_status=roi_grid_status
        )
        strategy_locks[strategy] = strategy_lock
        classifier_dependencies.extend(strategy_artifacts)
    primary = strategy_locks[PRIMARY_CLASSIFIER_STRATEGY]
    secondary = strategy_locks[STANDARDIZED_CLASSIFIER_STRATEGY]
    lock_path = run / "lock" / "primary_lock.json"
    save_json_atomic(
        lock_path,
        {
            "schema": 2,
            "model": model,
            "seed": seed,
            "analysis": "strict_predicted_roi_primary_with_standardized_secondary",
            "primary_classifier_strategy": PRIMARY_CLASSIFIER_STRATEGY,
            "secondary_classifier_strategy": STANDARDIZED_CLASSIFIER_STRATEGY,
            "classifier_strategies": strategy_locks,
            # Primary aliases remain explicit for the global 20-unit gate and
            # cannot be populated by the standardized secondary branch.
            "primary_system_evaluable": primary["operational_system_evaluable"],
            "operational_system_evaluable": primary["operational_system_evaluable"],
            "calibrated_system_evaluable": primary["calibrated_system_evaluable"],
            "classifier_checkpoint": primary["classifier_checkpoint"],
            "classifier_checkpoint_sha256": primary["classifier_checkpoint_sha256"],
            "segmenter_checkpoint": segmenter_lock["checkpoint"],
            "segmenter_checkpoint_sha256": segmenter_lock["checkpoint_sha256"],
            "segmenter_lock_sha256": sha256_file(segmenter_lock_path),
            "roi_policy": segmenter_lock["roi_policy"],
            "roi_grid_status": roi_grid_status,
            "monitor_status": primary["monitor_status"],
            "eye": primary["eye"],
            "patient": primary["patient"],
            "outside_roi_invariance": primary["outside_roi_invariance"],
            "classification_threshold_status": primary["classification_threshold_status"],
            "calibration_status": primary["calibration_status"],
            "validation_evaluable_eye_count": primary["validation_evaluable_eye_count"],
            "validation_evaluable_class_count": primary["validation_evaluable_class_count"],
            "status_reasons": primary["status_reasons"],
            "secondary_operational_system_evaluable": secondary["operational_system_evaluable"],
            "test_used_for_training_selection_calibration_or_thresholds": False,
            "config_sha256": cfg["_runtime"]["config_sha256"],
            "code_sha256": code_fingerprint()["sha256"],
        },
    )
    metadata = {
        "primary_classifier_strategy": PRIMARY_CLASSIFIER_STRATEGY,
        "secondary_classifier_strategy": STANDARDIZED_CLASSIFIER_STRATEGY,
        "classifier_strategy_count": 2,
        "monitor_status": primary["monitor_status"],
        "classification_threshold_status": primary["classification_threshold_status"],
        "calibration_status": primary["calibration_status"],
        "roi_grid_status": roi_grid_status,
        "validation_evaluable_eye_count": primary["validation_evaluable_eye_count"],
        "validation_evaluable_class_count": primary["validation_evaluable_class_count"],
        "status_reasons": primary["status_reasons"],
        "classifier_strategies": {
            strategy: {
                "role": value["classifier_role"],
                "estimand_id": value["classifier_estimand_id"],
                "monitor_status": value["monitor_status"],
                "classification_threshold_status": value["classification_threshold_status"],
                "calibration_status": value["calibration_status"],
                "operational_system_evaluable": value["operational_system_evaluable"],
                "validation_evaluable_eye_count": value["validation_evaluable_eye_count"],
                "validation_evaluable_class_count": value[
                    "validation_evaluable_class_count"
                ],
            }
            for strategy, value in strategy_locks.items()
        },
    }
    locked_dependencies: list[Path] = [
        lock_path,
        *classifier_dependencies,
        segmenter_lock_path,
        run / "segmenter" / "model_info.json",
        resolve_project_path(segmenter_lock["checkpoint"], must_exist=True),
        run / "roi" / "train_oof_index.csv",
        run / "roi" / "validation" / "index.csv",
        run / "classifiers" / "training_manifest.json",
    ]
    if segmenter_lock.get("training_spatial_prior"):
        locked_dependencies.append(
            resolve_project_path(segmenter_lock["training_spatial_prior"], must_exist=True)
        )
    for fold in range(int(cfg["cross_fitting"]["folds"])):
        fold_lock_path = run / "crossfit" / f"fold_{fold}" / "fold_lock.json"
        fold_lock = read_json(fold_lock_path)
        locked_dependencies.extend(
            [
                fold_lock_path,
                resolve_project_path(fold_lock["checkpoint"], must_exist=True),
                resolve_project_path(fold_lock["heldout_roi_index"], must_exist=True),
                run / "crossfit" / f"fold_{fold}" / "segmenter" / "model_info.json",
            ]
        )
    locked_dependencies = list(dict.fromkeys(path.resolve() for path in locked_dependencies))
    receipt = write_stage_receipt(
        cfg,
        "lock",
        model=model,
        seed=seed,
        artifacts=locked_dependencies,
        metadata=metadata,
    )
    return {
        "status": "complete",
        "receipt": str(receipt),
        "artifacts": [str(path) for path in locked_dependencies],
        "metadata": metadata,
    }


def _attach_retrospective_segmentation_metrics(
    cfg: Mapping[str, Any], dataset_root: Path, source_rows: pd.DataFrame,
    roi_index: pd.DataFrame,
) -> pd.DataFrame:
    """Add GT-based metrics after inference; never changes ROI validity/probability."""

    result = roi_index.copy()
    lookup = source_rows.copy()
    lookup["frame_id"] = lookup.frame_id.astype(str)
    lookup = lookup.set_index(["patient_id", "case_id", "side", "frame_id"])
    if lookup.index.duplicated().any():
        raise ProtocolGateError("Retrospective GT lookup repeats a frame identity")
    records: list[dict[str, Any]] = []
    tolerance = float(cfg["segmentation"].get("surface_dice_tolerance_pixels", 2.0))
    hit_threshold = float(cfg["evaluation"]["localized_success"]["frame_hit_iou_threshold"])
    for row in result.itertuples(index=False):
        source = lookup.loc[
            (row.patient_id, row.case_id, row.side, str(row.frame_id))
        ]
        with Image.open(dataset_root / str(source.output_mask)) as handle:
            ground_truth = np.asarray(handle) > 0
        if pd.isna(row.audit_path):
            raise ProtocolGateError("Retrospective segmentation evaluation requires an audit raster")
        audit_path = resolve_project_path(row.audit_path, must_exist=True)
        if sha256_file(audit_path) != row.audit_sha256:
            raise ProtocolGateError(f"Audit raster changed: {audit_path}")
        with np.load(audit_path, allow_pickle=False) as cached:
            raw = cached["thresholded_mask"].astype(bool, copy=False)
            postprocessed = cached["selected_mask"].astype(bool, copy=False)
        post = segmentation_metrics(
            postprocessed,
            ground_truth,
            roi_iou=hit_threshold,
            distances=True,
            surface_tolerance_px=tolerance,
        )
        raw_metrics = segmentation_metrics(
            raw,
            ground_truth,
            roi_iou=hit_threshold,
            distances=True,
            surface_tolerance_px=tolerance,
        )
        records.append({**post, **{f"raw_{key}": value for key, value in raw_metrics.items()}})
    metrics = pd.DataFrame(records, index=result.index)
    for column in metrics:
        result[column] = metrics[column]
    result["gt_used_for_inference_or_abstention"] = False
    return result


def _evaluation_decision_thresholds(cfg: Mapping[str, Any]) -> np.ndarray:
    specification = cfg["evaluation"]["decision_curve"]
    start, stop, step = (
        float(specification["threshold_start"]),
        float(specification["threshold_stop"]),
        float(specification["threshold_step"]),
    )
    values = np.arange(start, stop + step / 2.0, step)
    return np.round(values, 10)


def evaluate(cfg: Mapping[str, Any], model: str, seed: int) -> dict[str, Any]:
    """Evaluate both frozen classifier branches after the global 20-unit gate."""

    assert_prerequisites(cfg, "evaluate", model=model, seed=seed)
    existing = _completed_stage_receipt_or_none(cfg, "evaluate", model=model, seed=seed)
    if existing is not None:
        return {"status": "already_complete", "receipt": existing}
    access_path = open_test_access(cfg)
    run = unit_root(cfg, model, seed)
    evaluation_root = run / "evaluation"
    attempt_path = evaluation_root / "attempt.json"
    lock_path = run / "lock" / "primary_lock.json"
    lock = read_json(lock_path)
    attempt = {
        "model": model,
        "seed": seed,
        "config_sha256": cfg["_runtime"]["config_sha256"],
        "code_sha256": code_fingerprint()["sha256"],
        "primary_lock_sha256": sha256_file(lock_path),
        "test_access_sha256": sha256_file(access_path),
        "primary_classifier_strategy": PRIMARY_CLASSIFIER_STRATEGY,
        "classifier_strategy_checkpoint_sha256": {
            name: lock["classifier_strategies"][name].get("classifier_checkpoint_sha256")
            for name in _classifier_strategy_names(cfg)
        },
    }
    if attempt_path.exists():
        previous = read_json(attempt_path)
        if {key: previous.get(key) for key in attempt} != attempt:
            raise ProtocolGateError("A different code/config/lock test attempt already exists")
    else:
        save_json_atomic(attempt_path, {**attempt, "status": "started"})

    segmenter_lock_path = run / "segmenter" / "segmenter_lock.json"
    if sha256_file(segmenter_lock_path) != lock.get("segmenter_lock_sha256"):
        raise ProtocolGateError("Segmenter lock changed after the primary validation lock")
    segmenter_lock = read_json(segmenter_lock_path)
    if (
        segmenter_lock.get("checkpoint") != lock.get("segmenter_checkpoint")
        or segmenter_lock.get("checkpoint_sha256") != lock.get("segmenter_checkpoint_sha256")
        or segmenter_lock.get("roi_policy") != lock.get("roi_policy")
    ):
        raise ProtocolGateError("Primary lock and segmenter lock disagree")
    policy = ROIPolicy.from_dict(lock["roi_policy"])
    segmenter = _load_segmenter_from_lock(cfg, model, segmenter_lock)
    dataset_root, test_rows = split_frames(cfg, seed, "test")
    test_roi_index_path = _infer_and_cache_rois(
        cfg,
        segmenter,
        dataset_root,
        test_rows,
        policy,
        evaluation_root / "test_rois",
        save_audit=True,
    )
    roi_index = _verify_roi_index(cfg, test_roi_index_path, test_rows)
    del segmenter
    gc.collect()
    torch.cuda.empty_cache()
    roi_index["segmentation_roi_valid"] = roi_index.roi_valid.astype(bool)
    roi_index["segmentation_abstention_reason"] = roi_index.abstention_reason
    base_frames = _attach_retrospective_segmentation_metrics(
        cfg, dataset_root, test_rows, roi_index
    )
    if lock.get("primary_classifier_strategy") != PRIMARY_CLASSIFIER_STRATEGY:
        raise ProtocolGateError("The locked primary branch is not model_specific")
    strategy_results: dict[str, Any] = {}
    artifacts: list[Path] = [attempt_path, test_roi_index_path, lock_path, access_path]
    artifact_names = [
        "frames.csv", "eyes.csv", "patients.csv", "metrics.json",
        "calibration_curves.csv", "calibration_reliability_bands.csv",
        "decision_curves.csv", "risk_coverage_curves.csv",
        "bootstrap_patient_cluster_ci.csv", "segmentation_patient_cluster_ci.csv",
        "confusion_2x3.csv", "confusion_conditional_2x2.csv",
        "abstention_reasons.csv", "classification_by_label_3class.csv",
    ]
    for strategy in _classifier_strategy_names(cfg):
        branch = lock["classifier_strategies"][strategy]
        frames = base_frames.copy()
        # Even if calibration is unavailable, evaluate the immutable raw
        # checkpoint. Thresholded outputs remain unavailable unless locked.
        if branch["classifier_checkpoint"]:
            checkpoint = resolve_project_path(
                branch["classifier_checkpoint"], must_exist=True
            )
            if sha256_file(checkpoint) != branch["classifier_checkpoint_sha256"]:
                raise ProtocolGateError(
                    f"{strategy} checkpoint changed after validation lock"
                )
            classifier = _load_classifier_checkpoint(
                cfg,
                checkpoint,
                classifier_strategy=strategy,
                classifier_family=model,
            )
            frames = _infer_classifier_frames(
                cfg, classifier, frames, classifier_strategy=strategy
            )
            del classifier
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            artifacts.append(checkpoint)
        else:
            frames["roi_valid"] = False
            frames["abstention_reason"] = "classifier_unavailable"
            frames[["probability", "logit_0", "logit_1"]] = np.nan
            frames["classifier_strategy"] = strategy
        destination = evaluation_root / strategy
        strategy_seed_offset = int(
            _classifier_strategy_specs(cfg)[strategy].get("seed_offset", 0)
        )
        bundle = export_evaluation(
            frames,
            destination,
            eye_threshold=branch["eye"]["threshold"],
            patient_threshold=branch["patient"]["threshold"],
            eye_temperature=branch["eye"]["temperature"],
            patient_temperature=branch["patient"]["temperature"],
            eye_threshold_probability_scale=branch["eye"].get(
                "threshold_probability_scale"
            ),
            patient_threshold_probability_scale=branch["patient"].get(
                "threshold_probability_scale"
            ),
            min_valid_frames=int(cfg["aggregation"]["minimum_valid_frames"]),
            frames_per_eye=int(cfg["dataset"]["frames_per_eye"]),
            calibration_bins=int(cfg["calibration"]["ece_bins"]),
            bootstrap_draws=int(cfg["statistics"]["bootstrap_draws"]),
            bootstrap_seed=(
                int(cfg["statistics"]["bootstrap_seed"])
                + int(cfg["training"]["seed_offsets"]["bootstrap"])
                + int(seed)
                + strategy_seed_offset
            ),
            decision_thresholds=_evaluation_decision_thresholds(cfg),
            mean_segmentation_columns=[
                *cfg["segmentation"]["metrics"],
                *[f"raw_{metric}" for metric in cfg["segmentation"]["metrics"]],
            ],
            make_plots=True,
            classifier_strategy=strategy,
        )
        strategy_results[strategy] = {
            "model": model,
            "seed": int(seed),
            "classifier_strategy": strategy,
            "classifier_family": model,
            "role": branch.get("classifier_role"),
            "classifier_estimand_id": branch.get("classifier_estimand_id"),
            "is_primary_estimand": bool(strategy == PRIMARY_CLASSIFIER_STRATEGY),
            "test_frames": len(frames),
            "test_eyes": len(bundle["eyes"]),
            "test_patients": len(bundle["patients"]),
            "diagnostic_valid_frames": int(frames.roi_valid.sum()),
            "operational_system_evaluable": bool(
                branch["operational_system_evaluable"]
            ),
        }
        artifacts.extend(
            destination / name
            for name in artifact_names
            if (destination / name).is_file()
        )
        artifacts.extend(sorted(destination.glob("classification_calibration_decision_curves.*")))
        artifacts.extend(sorted(destination.glob("calibration_density_and_risk_coverage.*")))
        for provenance_key in (
            "classifier_history", "classifier_provenance", "classifier_training_result"
        ):
            artifacts.append(resolve_project_path(branch[provenance_key], must_exist=True))
    save_json_atomic(
        attempt_path,
        {
            **attempt,
            "status": "complete",
            "test_frames": len(base_frames),
            "segmentation_valid_frames": int(base_frames.segmentation_roi_valid.sum()),
            "classifier_strategies": strategy_results,
        },
    )
    artifacts = list(dict.fromkeys(path.resolve() for path in artifacts))
    receipt = write_stage_receipt(
        cfg,
        "evaluate",
        model=model,
        seed=seed,
        artifacts=artifacts,
        metadata={
            "test_frames": len(base_frames),
            "primary_classifier_strategy": PRIMARY_CLASSIFIER_STRATEGY,
            "classifier_strategies": strategy_results,
            "primary_system_evaluable": bool(lock["primary_system_evaluable"]),
            "calibrated_system_evaluable": bool(
                lock.get("calibrated_system_evaluable", False)
            ),
            "test_selection_or_refitting": False,
        },
    )
    return {"status": "complete", "receipt": str(receipt), "artifacts": [str(path) for path in artifacts]}


# ---------------------------------------------------------------------------
# Predeclared ablation execution
# ---------------------------------------------------------------------------


def _ablation_path(root: Path, category: str, task: AblationTask) -> Path:
    return root.joinpath(category, *task.key.split("/"))


def _immutable_engine_json(path: Path, value: Mapping[str, Any]) -> Path:
    """Write a small immutable execution artifact using protocol JSON rules."""

    cleaned = clean_json(dict(value))
    if path.exists():
        if read_json(path) != cleaned:
            raise ProtocolGateError(f"Refusing to replace immutable ablation artifact: {path}")
        return path
    save_json_atomic(path, cleaned)
    return path


def _verify_ablation_validation_bundle(
    root: Path, task: AblationTask, lock: Mapping[str, Any]
) -> tuple[Path, list[Path]]:
    """Bind every terminal validation lock to its complete execution bundle."""

    run = _ablation_path(root, "runs", task)
    result_path = run / "validation" / "result.json"
    if not result_path.is_file():
        raise ProtocolGateError(
            f"Ablation validation lock lacks its execution result: {task.key}"
        )
    result = read_json(result_path)
    if (
        result.get("task") != task.to_dict()
        or result.get("test_decoded_or_inferred") is not False
    ):
        raise ProtocolGateError(f"Ablation execution result identity failed: {task.key}")

    if task.requires_classifier_fit:
        pairs = (
            ("train_index_path", "train_index_sha256"),
            ("training_optimization_index_path", "training_optimization_index_sha256"),
            ("training_eye_eligibility_path", "training_eye_eligibility_sha256"),
            ("validation_index_path", "validation_index_sha256"),
            ("validation_frames_path", "validation_frames_sha256"),
            ("classifier_history_path", "classifier_history_sha256"),
            ("classifier_provenance_path", "classifier_provenance_sha256"),
            ("eye_table_path", "eye_table_sha256"),
            ("patient_table_path", "patient_table_sha256"),
        )
        identity = result.get("checkpoint_identity")
        if (
            result.get("classifier_strategy") != PRIMARY_CLASSIFIER_STRATEGY
            or result.get("classifier_family") != task.model
            or result.get("validation_frames_sha256")
            != lock.get("validation_source_sha256")
            or result.get("checkpoint_sha256") != lock.get("checkpoint_sha256")
            or not isinstance(result.get("available"), bool)
            or bool(result.get("available"))
            != (lock.get("checkpoint_path") is not None)
            or not isinstance(identity, Mapping)
            or identity.get("segmenter_condition") != f"ablation:{task.key}"
            or identity.get("outer_seed") != int(task.seed)
            or identity.get("classifier_strategy") != PRIMARY_CLASSIFIER_STRATEGY
            or identity.get("classifier_family") != task.model
        ):
            raise ProtocolGateError(
                f"Fitted ablation result conflicts with its terminal lock: {task.key}"
            )
    else:
        pairs = (
            ("primary_validation_frames_path", "primary_validation_frames_sha256"),
            ("validation_source_path", "validation_source_sha256"),
            ("patient_table_path", "patient_table_sha256"),
        )
        if (
            result.get("analytical_reuse") is not True
            or result.get("classifier_refit") is not False
            or result.get("validation_source_sha256")
            != lock.get("validation_source_sha256")
            or lock.get("checkpoint_path") is not None
        ):
            raise ProtocolGateError(
                f"Analytical ablation result conflicts with its terminal lock: {task.key}"
            )

    artifacts = [result_path.resolve()]
    resolved_by_field: dict[str, Path] = {}
    for path_field, hash_field in pairs:
        value = result.get(path_field)
        digest = result.get(hash_field)
        if not isinstance(value, str) or not isinstance(digest, str):
            raise ProtocolGateError(
                f"Ablation result lacks {path_field}/{hash_field}: {task.key}"
            )
        artifact = resolve_project_path(value, must_exist=True)
        if sha256_file(artifact) != digest:
            raise ProtocolGateError(f"Ablation provenance artifact changed: {artifact}")
        resolved_by_field[path_field] = artifact
        artifacts.append(artifact)

    lock_validation_path = lock.get("validation_source_path")
    if not isinstance(lock_validation_path, str):
        raise ProtocolGateError(f"Ablation lock lacks a validation source path: {task.key}")
    result_validation_field = (
        "validation_frames_path" if task.requires_classifier_fit
        else "validation_source_path"
    )
    if resolved_by_field[result_validation_field] != Path(lock_validation_path).resolve():
        raise ProtocolGateError(
            f"Ablation result and lock point to different validation sources: {task.key}"
        )

    if task.requires_classifier_fit and lock.get("checkpoint_path") is not None:
        checkpoint = Path(str(lock["checkpoint_path"])).resolve()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload.get("checkpoint_identity") != identity:
            raise ProtocolGateError(
                f"Ablation checkpoint identity conflicts with its task: {task.key}"
            )
        artifacts.append(checkpoint)
    return result_path.resolve(), list(dict.fromkeys(artifacts))


def _primary_roi_index_for_partition(
    cfg: Mapping[str, Any], model: str, seed: int, partition: str,
    expected_rows: pd.DataFrame,
) -> pd.DataFrame:
    run = unit_root(cfg, model, seed)
    paths = {
        "train": run / "roi" / "train_oof_index.csv",
        "validation": run / "roi" / "validation" / "index.csv",
        "test": run / "evaluation" / "test_rois" / "index.csv",
    }
    if partition not in paths:
        raise ValueError("partition must be train, validation, or test")
    index = _verify_roi_index(cfg, paths[partition], expected_rows)
    if "audit_path" not in index or index.audit_path.isna().any():
        raise ProtocolGateError(
            f"{model}/seed_{seed}/{partition} lacks hard-mask audit rasters required "
            "by the predeclared ablations; rebuild the ROI stage with this code version"
        )
    return index


def _ablation_primary_lookup(
    index: pd.DataFrame,
) -> dict[tuple[str, str, str, str], Any]:
    lookup: dict[tuple[str, str, str, str], Any] = {}
    for row in index.itertuples(index=False):
        key = (
            str(row.patient_id),
            str(row.case_id),
            str(row.side),
            str(row.frame_id),
        )
        if key in lookup:
            raise ProtocolGateError(f"Duplicate primary ROI audit key: {key}")
        lookup[key] = row
    return lookup


def _build_ablation_cache_index(
    cfg: Mapping[str, Any],
    plan: Any,
    task: AblationTask,
    partition: str,
    dataset_root: Path,
    source_rows: pd.DataFrame,
    *,
    cache_root: Path,
    index_path: Path,
    primary_index: pd.DataFrame | None = None,
    test_gate: Any | None = None,
) -> pd.DataFrame:
    """Materialize one task-bounded cache without reading undeclared inputs."""

    policy = None
    lookup: dict[tuple[str, str, str, str], Any] = {}
    if task.arm in PREDICTED_MASK_ARMS:
        if task.model is None or primary_index is None:
            raise ValueError("A predicted-mask ablation requires its primary ROI index")
        segmenter_lock = read_json(unit_root(cfg, task.model, task.seed) / "segmenter" / "segmenter_lock.json")
        policy = ROIPolicy.from_dict(segmenter_lock["roi_policy"])
        lookup = _ablation_primary_lookup(primary_index)
    elif primary_index is not None:
        raise ValueError("A non-predicted-mask ablation cannot receive a primary ROI index")

    builder = AblationInputBuilder(
        plan,
        task,
        partition,
        roi_policy=policy,
        test_gate=test_gate,
        target_size=cfg["roi"]["input_size"],
        neutral=DEFAULT_NEUTRAL_RGB,
    )
    images = ImageFrameDataset(dataset_root, source_rows)
    records: list[dict[str, Any]] = []
    for position, row in enumerate(source_rows.itertuples(index=False)):
        key = (
            str(row.patient_id),
            str(row.case_id),
            str(row.side),
            str(row.frame_id),
        )
        sample_key = "|".join(key)
        frame_identity_sha256 = _frame_identity_sha256(row)
        kwargs: dict[str, Any] = {}
        source_audit_path: str | None = None
        source_audit_sha256: str | None = None
        if task.arm == "gt_roi_oracle":
            with Image.open(dataset_root / str(row.output_mask)) as handle:
                kwargs["gt_mask"] = np.asarray(handle.convert("L")) > 0
        elif task.arm in PREDICTED_MASK_ARMS:
            if key not in lookup:
                raise ProtocolGateError(f"Primary ROI audit is missing ablation frame {key}")
            primary = lookup[key]
            audit_path = resolve_project_path(primary.audit_path, must_exist=True)
            if sha256_file(audit_path) != str(primary.audit_sha256):
                raise ProtocolGateError(f"Primary ROI audit changed: {audit_path}")
            with np.load(audit_path, allow_pickle=False) as audit:
                if not {"thresholded_mask", "selected_mask"} <= set(audit.files):
                    raise ProtocolGateError(f"ROI audit lacks locked hard rasters: {audit_path}")
                kwargs.update(
                    predicted_hard_mask=audit["thresholded_mask"].astype(bool, copy=True),
                    selected_mask=audit["selected_mask"].astype(bool, copy=True),
                    predicted_status=(
                        "valid" if bool(primary.roi_valid) else str(primary.abstention_reason)
                    ),
                    roi_confidence=(
                        float(primary.roi_confidence)
                        if pd.notna(primary.roi_confidence) else None
                    ),
                )
            source_audit_path = _relative(audit_path)
            source_audit_sha256 = str(primary.audit_sha256)

        item = builder.build(
            images[position]["image"], sample_key=sample_key, **kwargs
        )
        cache = write_ablation_cache(cache_root, task, partition, sample_key, item)
        if cache["frame_identity_sha256"] != frame_identity_sha256:
            raise ProtocolGateError("Ablation cache frame identity drifted during write")
        cache_path = _relative(cache["cache_path"]) if cache["cache_path"] else None
        records.append(
            {
                "patient_id": row.patient_id,
                "case_id": row.case_id,
                "side": row.side,
                "frame_id": str(row.frame_id),
                "frame_identity_sha256": frame_identity_sha256,
                "label": int(row.label_binary),
                "label_3class": int(row.label_3class),
                "split": partition,
                "model": task.model if task.model is not None else "shared",
                "seed": int(task.seed),
                "arm": task.arm,
                "variant": task.variant,
                "roi_valid": bool(item.valid),
                "abstention_reason": "" if item.valid else item.status,
                "roi_confidence": (
                    float(item.roi_confidence) if item.roi_confidence is not None else np.nan
                ),
                "classifier_mode": item.classifier_mode,
                "cache_path": cache_path,
                "cache_sha256": cache["cache_sha256"],
                "source_audit_path": source_audit_path,
                "source_audit_sha256": source_audit_sha256,
                "probability": np.nan,
                "logit_0": np.nan,
                "logit_1": np.nan,
            }
        )
    result = pd.DataFrame(records).sort_values(
        ["patient_id", "case_id", "side", "frame_id"], kind="stable"
    ).reset_index(drop=True)
    expected_keys = {
        (
            str(row.patient_id),
            str(row.case_id),
            str(row.side),
            str(row.frame_id),
        )
        for row in source_rows.itertuples(index=False)
    }
    observed_keys = set(
        zip(
            result.patient_id.astype(str),
            result.case_id.astype(str),
            result.side.astype(str),
            result.frame_id.astype(str),
        )
    )
    if observed_keys != expected_keys or len(result) != len(source_rows):
        raise ProtocolGateError("Ablation cache index does not cover the requested partition")
    _save_csv_atomic(index_path, result)
    return result


def _remove_ablation_scratch(cache_root: Path, task: AblationTask) -> None:
    target = cache_root.joinpath(*task.key.split("/")).resolve()
    allowed = cache_root.resolve()
    try:
        target.relative_to(allowed)
    except ValueError as error:
        raise RuntimeError(f"Refusing unsafe ablation scratch removal: {target}") from error
    if target != allowed and target.exists():
        shutil.rmtree(target)


def _ablation_geometry_settings(task: AblationTask) -> tuple[int, bool, Mapping[str, Any] | None]:
    geometry_features = 4 if task.arm in {"geometry_only", "appearance_plus_geometry"} else 0
    freeze_encoder = task.arm == "geometry_only"
    # A spatial/brightness transform of a constant neutral tensor would reveal
    # the mask outline in geometry-only; mask-only also requires an exact
    # binary input.  Their predeclared signal isolation therefore takes
    # precedence over appearance augmentation.
    augmentation: Mapping[str, Any] | None = (
        {} if task.arm in {"geometry_only", "mask_only"} else None
    )
    return geometry_features, freeze_encoder, augmentation


def _ablation_level_lock(value: Mapping[str, Any]) -> dict[str, Any]:
    status = str(value["status"])
    if status == "unavailable":
        return {
            "status": "unavailable",
            "threshold": None,
            "temperature": None,
            "unavailable_reason": str(value["reason"]),
        }
    if status == "calibration_unavailable":
        return {
            "status": "calibration_unavailable",
            "classification_threshold_status": "locked",
            "calibration_status": "unavailable",
            "threshold": float(value["threshold"]),
            "temperature": None,
            "threshold_probability_scale": "raw_due_to_calibration_unavailable",
            "unavailable_reason": str(value["reason"]),
        }
    if status != "locked":
        raise ProtocolGateError(f"Unexpected ablation validation lock status: {status}")
    return {
        "status": "locked",
        "classification_threshold_status": "locked",
        "calibration_status": "locked",
        "threshold": float(value["threshold"]),
        "temperature": float(value["temperature"]),
        "threshold_probability_scale": "temperature_scaled",
        "unavailable_reason": None,
    }


def _fit_and_lock_ablation_task(
    cfg: Mapping[str, Any], plan: Any, task: AblationTask, root: Path,
) -> Path:
    config_sha256 = str(cfg["_runtime"]["canonical_config_sha256"])
    lock_root = root / "validation_locks"
    lock_path = validation_lock_path(lock_root, task)
    if lock_path.is_file():
        existing_lock = verify_ablation_validation_lock(
            lock_path, task, plan_sha256=plan.sha256, config_sha256=config_sha256
        )
        _verify_ablation_validation_bundle(root, task, existing_lock)
        return lock_path

    dataset_root, train_rows = split_frames(cfg, task.seed, "train")
    _, validation_rows = split_frames(cfg, task.seed, "validation")
    primary_train = primary_validation = None
    if task.arm in PREDICTED_MASK_ARMS:
        assert task.model is not None
        primary_train = _primary_roi_index_for_partition(
            cfg, task.model, task.seed, "train", train_rows
        )
        primary_validation = _primary_roi_index_for_partition(
            cfg, task.model, task.seed, "validation", validation_rows
        )

    run = _ablation_path(root, "runs", task)
    cache_root = root / "scratch" / "cache"
    try:
        train_index = _build_ablation_cache_index(
            cfg, plan, task, "train", dataset_root, train_rows,
            cache_root=cache_root,
            index_path=run / "inputs" / "train_index.csv",
            primary_index=primary_train,
        )
        validation_index = _build_ablation_cache_index(
            cfg, plan, task, "validation", dataset_root, validation_rows,
            cache_root=cache_root,
            index_path=run / "inputs" / "validation_index.csv",
            primary_index=primary_validation,
        )
        optimization_index, eligibility_ledger, selection_provenance = (
            _persist_classifier_training_selection(
                cfg,
                train_index,
                raw_index_path=run / "inputs" / "train_index.csv",
                destination=run / "inputs" / "classifier_selection",
            )
        )
        geometry_features, freeze_encoder, augmentation = _ablation_geometry_settings(task)
        trained = _train_roi_classifier_model(
            cfg,
            f"ablation:{task.key}",
            task.seed,
            train_index,
            validation_index,
            run / "classifier",
            optimization_index=optimization_index,
            eligibility_ledger=eligibility_ledger,
            training_selection_provenance=selection_provenance,
            geometry_features=geometry_features,
            classifier_input=str(task.arm),
            freeze_image_encoder=freeze_encoder,
            augmentation=augmentation,
            classifier_strategy=PRIMARY_CLASSIFIER_STRATEGY,
            classifier_family=task.model,
        )
        validation_source = Path(trained["validation_frames"])
        frames = pd.read_csv(
            validation_source, encoding="utf-8-sig", dtype={"frame_id": str}
        )
        eyes = aggregate_frames_to_eyes(
            frames,
            threshold=0.5,
            frames_per_eye=int(cfg["dataset"]["frames_per_eye"]),
            min_valid_frames=int(cfg["aggregation"]["minimum_valid_frames"]),
        )
        patients = aggregate_eyes_to_patients(eyes.copy(), threshold=0.5)
        eye_path = _save_csv_atomic(run / "validation" / "eyes_raw.csv", eyes)
        patient_path = _save_csv_atomic(run / "validation" / "patients_raw.csv", patients)
        eye_lock = _ablation_level_lock(
            _lock_calibration_and_threshold(cfg, eyes, level="eye")
        )
        patient_lock = _ablation_level_lock(
            _lock_calibration_and_threshold(cfg, patients, level="patient")
        )
        checkpoint = Path(trained["checkpoint"]) if trained["checkpoint"] else None
        result_path = _immutable_engine_json(
            run / "validation" / "result.json",
            {
                "task": task.to_dict(),
                "available": bool(trained["available"]),
                "classifier_mode": (
                    "geometry_only_frozen_encoder" if freeze_encoder
                    else "appearance_plus_explicit_geometry" if geometry_features
                    else "strict_roi_masked_pixels_no_explicit_geometry_vector"
                ),
                "classifier_strategy": PRIMARY_CLASSIFIER_STRATEGY,
                "classifier_family": task.model,
                "geometry_features": geometry_features,
                "image_encoder_frozen": freeze_encoder,
                "augmentation": cfg["training"]["augmentation"] if augmentation is None else augmentation,
                "checkpoint_sha256": sha256_file(checkpoint) if checkpoint else None,
                "checkpoint_identity": trained["checkpoint_identity"],
                "train_index_path": _relative(run / "inputs" / "train_index.csv"),
                "train_index_sha256": sha256_file(run / "inputs" / "train_index.csv"),
                "training_optimization_index_path": selection_provenance[
                    "training_optimization_index"
                ],
                "training_optimization_index_sha256": selection_provenance[
                    "training_optimization_index_sha256"
                ],
                "training_eye_eligibility_path": selection_provenance[
                    "training_eye_eligibility"
                ],
                "training_eye_eligibility_sha256": selection_provenance[
                    "training_eye_eligibility_sha256"
                ],
                "minimum_valid_frames": selection_provenance["minimum_valid_frames"],
                "frames_per_eye": selection_provenance["frames_per_eye"],
                "validation_index_path": _relative(
                    run / "inputs" / "validation_index.csv"
                ),
                "validation_index_sha256": sha256_file(run / "inputs" / "validation_index.csv"),
                "validation_frames_path": _relative(validation_source),
                "validation_frames_sha256": sha256_file(validation_source),
                "classifier_history_path": _relative(Path(trained["history"])),
                "classifier_history_sha256": sha256_file(Path(trained["history"])),
                "classifier_provenance_path": _relative(Path(trained["model_info"])),
                "classifier_provenance_sha256": sha256_file(Path(trained["model_info"])),
                "eye_table_path": _relative(eye_path),
                "eye_table_sha256": sha256_file(eye_path),
                "patient_table_path": _relative(patient_path),
                "patient_table_sha256": sha256_file(patient_path),
                "test_decoded_or_inferred": False,
            },
        )
        lock_path = write_ablation_validation_lock(
            lock_root,
            task,
            plan_sha256=plan.sha256,
            config_sha256=config_sha256,
            validation_source_sha256=sha256_file(validation_source),
            validation_source_path=validation_source.resolve(),
            checkpoint_sha256=sha256_file(checkpoint) if checkpoint else None,
            checkpoint_path=checkpoint.resolve() if checkpoint else None,
            level_locks={"eye": eye_lock, "patient": patient_lock},
        )
        locked = verify_ablation_validation_lock(
            lock_path, task, plan_sha256=plan.sha256, config_sha256=config_sha256
        )
        verified_result, _ = _verify_ablation_validation_bundle(root, task, locked)
        if verified_result != result_path.resolve():
            raise ProtocolGateError("Ablation result path changed during terminal lock")
        return lock_path
    finally:
        _remove_ablation_scratch(cache_root, task)


def _lock_analytical_ablation_batch(
    cfg: Mapping[str, Any], plan: Any, tasks: Iterable[AblationTask], root: Path,
) -> list[Path]:
    selected = tuple(tasks)
    if not selected:
        return []
    first = selected[0]
    assert first.model is not None
    primary_path = (
        _classifier_work_dir(unit_root(cfg, first.model, first.seed), PRIMARY_CLASSIFIER_STRATEGY)
        / "validation_frames_raw.csv"
    )
    frames = pd.read_csv(primary_path, encoding="utf-8-sig", dtype={"frame_id": str})
    frames["model"] = first.model
    frames["seed"] = int(first.seed)
    variants = build_analytical_eye_variants(frames, cfg, plan=plan, partition="validation")
    lock_root = root / "validation_locks"
    config_sha256 = str(cfg["_runtime"]["canonical_config_sha256"])
    paths: list[Path] = []
    for task in selected:
        lock_path = validation_lock_path(lock_root, task)
        if lock_path.is_file():
            existing_lock = verify_ablation_validation_lock(
                lock_path, task, plan_sha256=plan.sha256, config_sha256=config_sha256
            )
            _verify_ablation_validation_bundle(root, task, existing_lock)
            paths.append(lock_path)
            continue
        key = f"{task.arm}/{task.variant}"
        if key not in variants:
            raise ProtocolGateError(f"Missing analytical validation variant: {key}")
        eyes = variants[key].copy()
        patients = aggregate_eyes_to_patients(eyes.copy(), threshold=0.5)
        run = _ablation_path(root, "runs", task)
        source_path = _save_csv_atomic(run / "validation" / "eyes_raw.csv", eyes)
        patient_path = _save_csv_atomic(run / "validation" / "patients_raw.csv", patients)
        result_path = _immutable_engine_json(
            run / "validation" / "result.json",
            {
                "task": task.to_dict(),
                "analytical_reuse": True,
                "classifier_refit": False,
                "primary_validation_frames_path": _relative(primary_path),
                "primary_validation_frames_sha256": sha256_file(primary_path),
                "validation_source_path": _relative(source_path),
                "validation_source_sha256": sha256_file(source_path),
                "patient_table_path": _relative(patient_path),
                "patient_table_sha256": sha256_file(patient_path),
                "test_decoded_or_inferred": False,
            },
        )
        lock_path = write_ablation_validation_lock(
            lock_root,
            task,
            plan_sha256=plan.sha256,
            config_sha256=config_sha256,
            validation_source_sha256=sha256_file(source_path),
            validation_source_path=source_path.resolve(),
            level_locks={
                "eye": _ablation_level_lock(
                    _lock_calibration_and_threshold(cfg, eyes, level="eye")
                ),
                "patient": _ablation_level_lock(
                    _lock_calibration_and_threshold(cfg, patients, level="patient")
                ),
            },
        )
        locked = verify_ablation_validation_lock(
            lock_path, task, plan_sha256=plan.sha256, config_sha256=config_sha256
        )
        verified_result, _ = _verify_ablation_validation_bundle(root, task, locked)
        if verified_result != result_path.resolve():
            raise ProtocolGateError("Analytical ablation result path changed during lock")
        paths.append(lock_path)
    return paths


def lock_ablations(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Fit 180 model-specific classifiers and freeze all 320 decisions.

    This stage never calls ``split_frames(..., 'test')`` and never reads a
    primary test prediction table.  Task caches are deleted immediately after
    each classifier's immutable checkpoint and validation source are locked.
    """

    assert_prerequisites(cfg, "lock-ablations")
    existing = _completed_stage_receipt_or_none(cfg, "lock-ablations")
    if existing is not None:
        return {"status": "already_complete", "receipt": existing}
    plan = build_ablation_plan(cfg)
    root = output_root(cfg) / "ablations"
    for batch in build_ablation_validation_schedule(plan):
        if batch.phase == "lock_analytical":
            _lock_analytical_ablation_batch(cfg, plan, batch.tasks, root)
        else:
            for task in batch.tasks:
                _fit_and_lock_ablation_task(cfg, plan, task, root)

    config_sha256 = str(cfg["_runtime"]["canonical_config_sha256"])
    lock_root = root / "validation_locks"
    lock_hashes = collect_ablation_validation_locks(
        plan, lock_root, config_sha256=config_sha256
    )
    if len(lock_hashes) != int(cfg["ablations"]["expected_validation_locks"]):
        raise ProtocolGateError("Ablation validation lock cardinality does not match the protocol")
    entries: list[dict[str, Any]] = []
    artifacts: list[Path] = []
    for task in plan.tasks:
        path = validation_lock_path(lock_root, task)
        value = verify_ablation_validation_lock(
            path, task, plan_sha256=plan.sha256, config_sha256=config_sha256
        )
        result_path, bundle_artifacts = _verify_ablation_validation_bundle(
            root, task, value
        )
        entries.append(
            {
                "task_key": task.key,
                "lock_path": _relative(path),
                "lock_sha256": lock_hashes[task.key],
                "checkpoint_sha256": value.get("checkpoint_sha256"),
                "validation_source_sha256": value["validation_source_sha256"],
                "execution_result_sha256": sha256_file(result_path),
            }
        )
        artifacts.extend([path, *bundle_artifacts])
    manifest_path = _immutable_engine_json(
        root / "state" / "validation_lock_manifest.json",
        {
            "schema": 1,
            "plan": plan.summary(),
            "config_sha256": config_sha256,
            "lock_count": len(entries),
            "classifier_fit_count": plan.summary()["classifier_fit_count"],
            "analytical_lock_count": plan.summary()["analytical_task_count"],
            "all_validation_locks_before_any_ablation_test": True,
            "test_predictions_generated": False,
            "entries": entries,
        },
    )
    artifacts.insert(0, manifest_path)
    # De-duplicate paths while preserving stable receipt order.
    unique_artifacts = list(dict.fromkeys(Path(path).resolve() for path in artifacts))
    receipt = write_stage_receipt(
        cfg,
        "lock-ablations",
        artifacts=unique_artifacts,
        metadata={
            "ablation_validation_lock_count": len(lock_hashes),
            "all_ablation_validation_locks_complete": True,
            "test_predictions_generated": False,
            "lock_manifest_sha256": sha256_file(manifest_path),
            "classifier_fit_count": plan.summary()["classifier_fit_count"],
            "analytical_reuse_lock_count": plan.summary()["analytical_task_count"],
        },
    )
    return {
        "status": "complete",
        "receipt": str(receipt),
        "artifacts": [str(path) for path in unique_artifacts],
        "metadata": plan.summary(),
    }


def _ablation_lock_values(lock: Mapping[str, Any], level: str) -> tuple[float | None, float | None]:
    value = lock["level_locks"][level]
    threshold = value.get("threshold")
    temperature = value.get("temperature")
    return (
        float(threshold) if threshold is not None else None,
        float(temperature) if temperature is not None else None,
    )


def _ablation_export(
    cfg: Mapping[str, Any],
    frames: pd.DataFrame,
    destination: Path,
    lock: Mapping[str, Any],
    *,
    seed: int,
    min_valid_frames: int,
) -> dict[str, Any]:
    eye_threshold, eye_temperature = _ablation_lock_values(lock, "eye")
    patient_threshold, patient_temperature = _ablation_lock_values(lock, "patient")
    return export_evaluation(
        frames,
        destination,
        eye_threshold=eye_threshold,
        patient_threshold=patient_threshold,
        eye_temperature=eye_temperature,
        patient_temperature=patient_temperature,
        eye_threshold_probability_scale=lock["level_locks"]["eye"].get(
            "threshold_probability_scale"
        ),
        patient_threshold_probability_scale=lock["level_locks"]["patient"].get(
            "threshold_probability_scale"
        ),
        min_valid_frames=int(min_valid_frames),
        frames_per_eye=int(cfg["dataset"]["frames_per_eye"]),
        calibration_bins=int(cfg["calibration"]["ece_bins"]),
        bootstrap_draws=int(cfg["statistics"]["bootstrap_draws"]),
        bootstrap_seed=(
            int(cfg["statistics"]["bootstrap_seed"])
            + int(cfg["training"]["seed_offsets"]["bootstrap"])
            + int(seed)
        ),
        decision_thresholds=_evaluation_decision_thresholds(cfg),
        mean_segmentation_columns=(),
        # Hundreds of repeated publication panels are intentionally omitted;
        # the immutable tables contain every value needed for final composite
        # ablation figures.
        make_plots=False,
        classifier_strategy=PRIMARY_CLASSIFIER_STRATEGY,
    )


_ABLATION_EXPORT_FILENAMES = (
    "frames.csv",
    "eyes.csv",
    "patients.csv",
    "metrics.json",
    "calibration_curves.csv",
    "calibration_reliability_bands.csv",
    "decision_curves.csv",
    "risk_coverage_curves.csv",
    "bootstrap_patient_cluster_ci.csv",
    "segmentation_patient_cluster_ci.csv",
    "confusion_2x3.csv",
    "confusion_conditional_2x2.csv",
    "abstention_reasons.csv",
    "classification_by_label_3class.csv",
)


def _ablation_test_artifact_records(
    destination: Path, task: AblationTask
) -> list[dict[str, Any]]:
    """Hash every task output consumed by summaries or later audit."""

    names = [*_ABLATION_EXPORT_FILENAMES]
    if task.requires_classifier_fit:
        names.append("test_index.csv")
    records: list[dict[str, Any]] = []
    for name in names:
        path = (destination / name).resolve()
        if not path.is_file():
            raise ProtocolGateError(f"Missing completed ablation task artifact: {path}")
        records.append(
            {
                "name": name,
                "path": str(path),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return records


def _verified_ablation_test_artifacts(
    result_path: Path, task: AblationTask, value: Mapping[str, Any] | None = None
) -> list[Path]:
    """Verify the immutable per-task artifact manifest and return its paths."""

    result = dict(value) if value is not None else read_json(result_path)
    if (
        result.get("schema") != 1
        or result.get("status") != "complete"
        or result.get("task") != task.to_dict()
    ):
        raise ProtocolGateError(f"Malformed completed ablation result: {result_path}")
    records = result.get("task_artifacts")
    if not isinstance(records, list):
        raise ProtocolGateError(f"Ablation result lacks a task artifact manifest: {result_path}")
    expected_names = [*_ABLATION_EXPORT_FILENAMES]
    if task.requires_classifier_fit:
        expected_names.append("test_index.csv")
    observed_names = [record.get("name") for record in records if isinstance(record, Mapping)]
    if observed_names != expected_names or len(records) != len(expected_names):
        raise ProtocolGateError(f"Ablation task artifact inventory changed: {result_path}")
    verified: list[Path] = []
    for name, record in zip(expected_names, records, strict=True):
        if not isinstance(record, Mapping):
            raise ProtocolGateError(f"Malformed ablation task artifact record: {result_path}")
        expected_path = (result_path.parent / name).resolve()
        artifact_path = Path(str(record.get("path", ""))).resolve()
        if artifact_path != expected_path or not artifact_path.is_file():
            raise ProtocolGateError(f"Ablation task artifact path changed: {expected_path}")
        if (
            artifact_path.stat().st_size != record.get("bytes")
            or sha256_file(artifact_path) != record.get("sha256")
        ):
            raise ProtocolGateError(f"Ablation task artifact changed: {artifact_path}")
        verified.append(artifact_path)
    metrics_record = records[expected_names.index("metrics.json")]
    metrics_path = Path(str(result.get("metrics_path", ""))).resolve()
    if (
        metrics_path != verified[expected_names.index("metrics.json")]
        or result.get("metrics_sha256") != metrics_record.get("sha256")
    ):
        raise ProtocolGateError(f"Ablation metrics pointer disagrees with artifact manifest: {result_path}")
    return verified


def _completed_ablation_test_result(
    path: Path, task: AblationTask, gate_sha256: str, lock_sha256: str
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = read_json(path)
    if (
        value.get("task") != task.to_dict()
        or value.get("ablation_test_gate_sha256") != gate_sha256
        or value.get("validation_lock_sha256") != lock_sha256
        or value.get("status") != "complete"
    ):
        raise ProtocolGateError(f"Ablation test result changed after first evaluation: {path}")
    _verified_ablation_test_artifacts(path, task, value)
    return value


def _evaluate_fitted_ablation_task(
    cfg: Mapping[str, Any], plan: Any, task: AblationTask, root: Path, gate: Any,
) -> Path:
    lock_path = validation_lock_path(root / "validation_locks", task)
    lock = verify_ablation_validation_lock(
        lock_path,
        task,
        plan_sha256=plan.sha256,
        config_sha256=str(cfg["_runtime"]["canonical_config_sha256"]),
    )
    destination = _ablation_path(root, "test", task)
    result_path = destination / "result.json"
    lock_sha256 = sha256_file(lock_path)
    if _completed_ablation_test_result(
        result_path, task, gate.sha256, lock_sha256
    ) is not None:
        return result_path

    dataset_root, test_rows = split_frames(cfg, task.seed, "test")
    primary_test = None
    if task.arm in PREDICTED_MASK_ARMS:
        assert task.model is not None
        primary_test = _primary_roi_index_for_partition(
            cfg, task.model, task.seed, "test", test_rows
        )
    cache_root = root / "scratch" / "cache"
    try:
        test_index = _build_ablation_cache_index(
            cfg,
            plan,
            task,
            "test",
            dataset_root,
            test_rows,
            cache_root=cache_root,
            index_path=destination / "test_index.csv",
            primary_index=primary_test,
            test_gate=gate,
        )
        checkpoint_value = lock.get("checkpoint_path")
        geometry_features, freeze_encoder, _ = _ablation_geometry_settings(task)
        if checkpoint_value:
            checkpoint = Path(checkpoint_value)
            if sha256_file(checkpoint) != lock["checkpoint_sha256"]:
                raise ProtocolGateError(f"Ablation classifier changed after validation lock: {checkpoint}")
            classifier = _load_classifier_checkpoint(
                cfg,
                checkpoint,
                geometry_features=geometry_features,
                freeze_image_encoder=freeze_encoder,
                classifier_strategy=PRIMARY_CLASSIFIER_STRATEGY,
                classifier_family=task.model,
            )
            frames = _infer_classifier_frames(
                cfg,
                classifier,
                test_index,
                geometry_features=geometry_features,
                classifier_strategy=PRIMARY_CLASSIFIER_STRATEGY,
            )
            del classifier
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            frames = test_index.copy()
            frames["roi_valid"] = False
            frames["abstention_reason"] = "validation_lock_classifier_unavailable"
            frames[["probability", "logit_0", "logit_1"]] = np.nan
            frames["classifier_strategy"] = PRIMARY_CLASSIFIER_STRATEGY
        frames["ablation_task"] = task.key
        bundle = _ablation_export(
            cfg,
            frames,
            destination,
            lock,
            seed=task.seed,
            min_valid_frames=int(cfg["aggregation"]["minimum_valid_frames"]),
        )
        metrics_path = destination / "metrics.json"
        task_artifacts = _ablation_test_artifact_records(destination, task)
        _immutable_engine_json(
            result_path,
            {
                "schema": 1,
                "status": "complete",
                "task": task.to_dict(),
                "ablation_test_gate_sha256": gate.sha256,
                "validation_lock_sha256": lock_sha256,
                "test_index_sha256": sha256_file(destination / "test_index.csv"),
                "metrics_path": str(metrics_path.resolve()),
                "metrics_sha256": sha256_file(metrics_path),
                "task_artifacts": task_artifacts,
                "test_frames": len(frames),
                "test_eyes": len(bundle["eyes"]),
                "test_patients": len(bundle["patients"]),
                "test_selection_or_refitting": False,
                "task_cache_deleted_after_inference": True,
            },
        )
        return result_path
    finally:
        _remove_ablation_scratch(cache_root, task)


def _analytical_carrier_frames(
    primary_frames: pd.DataFrame, analytical_eyes: pd.DataFrame
) -> pd.DataFrame:
    """Carry an already aggregated eye score through the common reporter.

    Every valid frame of an evaluable eye receives the fixed analytical eye
    score.  Re-averaging therefore reproduces the locked score exactly while
    retaining original frame-validity counts and patient/eye membership.
    These carrier values are explicitly labelled and are never presented as
    independent frame predictions.
    """

    result = primary_frames.copy()
    result["source_primary_frame_probability"] = result["probability"]
    result["analytical_probability_carrier"] = True
    lookup = analytical_eyes.set_index("case_id")
    if lookup.index.duplicated().any():
        raise ProtocolGateError("Analytical eye table contains duplicate eyes")
    if set(result.case_id.astype(str)) != set(lookup.index.astype(str)):
        raise ProtocolGateError("Analytical eye table is not aligned to primary test frames")
    # Normalize lookup keys without mutating the source table.
    by_case = {str(index): row for index, row in lookup.iterrows()}
    for case_id, indices in result.groupby(result.case_id.astype(str)).groups.items():
        eye = by_case[str(case_id)]
        if bool(eye.evaluable):
            valid = result.loc[indices, "roi_valid"].astype(bool)
            result.loc[np.asarray(indices)[valid.to_numpy()], "probability"] = float(eye.probability)
    return result


def _evaluate_analytical_ablation_task(
    cfg: Mapping[str, Any], plan: Any, task: AblationTask, root: Path, gate: Any,
    primary_frames: pd.DataFrame, variants: Mapping[str, pd.DataFrame],
) -> Path:
    lock_path = validation_lock_path(root / "validation_locks", task)
    lock = verify_ablation_validation_lock(
        lock_path,
        task,
        plan_sha256=plan.sha256,
        config_sha256=str(cfg["_runtime"]["canonical_config_sha256"]),
    )
    destination = _ablation_path(root, "test", task)
    result_path = destination / "result.json"
    lock_sha256 = sha256_file(lock_path)
    if _completed_ablation_test_result(
        result_path, task, gate.sha256, lock_sha256
    ) is not None:
        return result_path
    key = f"{task.arm}/{task.variant}"
    analytical_eyes = variants[key]
    frames = _analytical_carrier_frames(primary_frames, analytical_eyes)
    frames["ablation_task"] = task.key
    minimum = (
        int(task.variant)
        if task.arm == "minimum_valid_frames"
        else int(cfg["aggregation"]["minimum_valid_frames"])
    )
    bundle = _ablation_export(
        cfg, frames, destination, lock, seed=task.seed, min_valid_frames=minimum
    )
    exported = bundle["eyes"].set_index("case_id")
    expected = analytical_eyes.set_index("case_id")
    common = expected.evaluable.astype(bool) & exported.evaluable.astype(bool)
    if common.any() and not np.allclose(
        exported.loc[common, "probability_raw"].to_numpy(float),
        expected.loc[common, "probability"].to_numpy(float),
        atol=1e-12,
        rtol=0.0,
    ):
        raise ProtocolGateError("Analytical carrier did not reproduce the locked eye scores")
    metrics_path = destination / "metrics.json"
    task_artifacts = _ablation_test_artifact_records(destination, task)
    _immutable_engine_json(
        result_path,
        {
            "schema": 1,
            "status": "complete",
            "task": task.to_dict(),
            "ablation_test_gate_sha256": gate.sha256,
            "validation_lock_sha256": lock_sha256,
            "primary_test_frames_sha256": dataframe_sha256(primary_frames),
            "metrics_path": str(metrics_path.resolve()),
            "metrics_sha256": sha256_file(metrics_path),
            "task_artifacts": task_artifacts,
            "test_frames": len(frames),
            "test_eyes": len(bundle["eyes"]),
            "test_patients": len(bundle["patients"]),
            "classifier_refit": False,
            "analytical_probability_reuse": True,
            "test_selection_or_refitting": False,
        },
    )
    return result_path


def _ablation_primary_contrasts(
    cfg: Mapping[str, Any], root: Path, result_paths: Iterable[Path]
) -> Path:
    """Compare every predeclared arm with its aligned deployable primary."""

    rows: list[dict[str, Any]] = []
    draws = int(cfg["statistics"]["bootstrap_draws"])
    base_seed = int(cfg["statistics"]["bootstrap_seed"])
    for task_index, result_path in enumerate(result_paths):
        result = read_json(result_path)
        task = result["task"]
        comparators = (
            list(cfg["models"])
            if task.get("model") is None
            else [str(task["model"])]
        )
        for model_index, model_name in enumerate(comparators):
            for level_index, level in enumerate(("eye", "patient")):
                reference = pd.read_csv(
                    unit_root(cfg, model_name, int(task["seed"]))
                    / "evaluation"
                    / PRIMARY_CLASSIFIER_STRATEGY
                    / f"{level}s.csv",
                    encoding="utf-8-sig",
                )
                comparator = pd.read_csv(
                    Path(result_path).parent / f"{level}s.csv",
                    encoding="utf-8-sig",
                )
                reference, comparator = _aligned_unit_pair(
                    reference, comparator, level=level
                )
                effect = _paired_locked_primary_effect(
                    reference,
                    comparator,
                    draws=draws,
                    seed=(
                        base_seed
                        + int(task["seed"]) * 30_011
                        + task_index * 31
                        + model_index * 3
                        + level_index
                    ),
                )
                rows.append(
                    {
                        "model": model_name,
                        "seed": int(task["seed"]),
                        "level": level,
                        "reference": "strict_predicted_roi_primary",
                        "ablation_arm": str(task["arm"]),
                        "variant": (
                            str(task["variant"])
                            if task.get("variant") is not None
                            else "none"
                        ),
                        "contrast": "primary - ablation",
                        "metric": "failure_inclusive.balanced_accuracy",
                        "test_selection_or_refitting": False,
                        **effect,
                    }
                )
    table = pd.DataFrame(rows)
    table["family_id"] = (
        "ablation_"
        + table.model.astype(str)
        + "_"
        + table.level.astype(str)
        + "_failure_aware_balanced_accuracy"
    )
    table["planned_family_size"] = 0
    table["p_value_holm"] = np.nan
    table["reject_holm_0_05"] = False
    for family_id, indices in table.groupby("family_id").groups.items():
        family_size = int(len(indices))
        table.loc[indices, "planned_family_size"] = family_size
        adjusted, rejected = holm_adjust(
            table.loc[indices, "p_value_randomization_two_sided"],
            alpha=float(cfg["statistics"]["alpha"]),
            planned_family_size=family_size,
        )
        table.loc[indices, "p_value_holm"] = adjusted
        table.loc[indices, "reject_holm_0_05"] = rejected
    path = root / "summary" / "paired_primary_vs_ablation_patient_cluster_holm.csv"
    _save_csv_atomic(path, table)
    return path


def _ablation_summary(
    cfg: Mapping[str, Any], plan: Any, root: Path, result_paths: Iterable[Path]
) -> list[Path]:
    result_paths = list(result_paths)
    tasks_by_key = {task.key: task for task in plan.tasks}
    for result_path in result_paths:
        result = read_json(result_path)
        task_key = result.get("task", {}).get("key")
        if task_key not in tasks_by_key:
            raise ProtocolGateError(f"Unknown task in ablation result: {result_path}")
        _verified_ablation_test_artifacts(
            result_path, tasks_by_key[task_key], result
        )
    rows: list[dict[str, Any]] = []
    for result_path in result_paths:
        result = read_json(result_path)
        task_value = result["task"]
        metrics = read_json(Path(result["metrics_path"]))
        for level in ("eye", "patient"):
            scope_metrics = metrics[level]["ALL"]
            operational = scope_metrics.get(
                "primary_operational", scope_metrics["primary_calibrated"]
            )
            rows.append(
                {
                    "model": task_value.get("model") or "shared",
                    "classifier_strategy": PRIMARY_CLASSIFIER_STRATEGY,
                    "seed": int(task_value["seed"]),
                    "level": level,
                    "scope": "ALL",
                    "arm": task_value["arm"],
                    "variant": (
                        str(task_value["variant"])
                        if task_value.get("variant") is not None else "none"
                    ),
                    **flatten_metrics(operational),
                }
            )
    table = pd.DataFrame(rows)
    destination = root / "summary"
    per_seed_path = _save_csv_atomic(destination / "classification_per_seed.csv", table)
    preferred = [
        "coverage", "coverage_negative", "coverage_positive", "selective_risk",
        "failure_aware_aurc", "conditional.auroc", "conditional.average_precision",
        "conditional.accuracy", "conditional.balanced_accuracy", "conditional.sensitivity",
        "conditional.specificity", "conditional.ppv", "conditional.npv", "conditional.f1",
        "conditional.mcc", "conditional.brier", "conditional.nll",
        "failure_inclusive.accuracy", "failure_inclusive.balanced_accuracy",
        "failure_inclusive.sensitivity", "failure_inclusive.specificity",
        "calibration.ece_equal_width", "calibration.ece_equal_mass",
        "calibration.calibration_intercept", "calibration.calibration_slope",
    ]
    metric_columns = [column for column in preferred if column in table]
    summary_path = destination / "classification_five_seed_mean_sd.csv"
    export_five_seed_summary(
        table,
        summary_path,
        metric_columns=metric_columns,
        group_columns=(
            "model", "classifier_strategy", "level", "scope", "arm", "variant"
        ),
    )
    interpretation_path = _immutable_engine_json(
        destination / "interpretation.json",
        {
            "schema": 1,
            "task_count": len(plan.tasks),
            "classifier_fit_count": plan.summary()["classifier_fit_count"],
            "shared_arms_fitted_once_per_seed": False,
            "all_learned_arms_use_model_specific_classifier_family_per_model_seed": True,
            "analytical_arms_reuse_primary_frame_probabilities": True,
            "seed_summary": "arithmetic mean plus sample standard deviation over five seed estimates",
            "inference_warning": (
                "The five holdouts overlap and are not independent cohorts; mean plus/minus SD is descriptive."
            ),
            "primary_summary_is_separate_and_unchanged": True,
        },
    )
    contrast_path = _ablation_primary_contrasts(cfg, root, result_paths)
    return [per_seed_path, summary_path, contrast_path, interpretation_path]


def evaluate_ablations(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate all 320 frozen tasks only after the one global gate opens."""

    assert_prerequisites(cfg, "evaluate-ablations")
    existing = _completed_stage_receipt_or_none(cfg, "evaluate-ablations")
    if existing is not None:
        return {"status": "already_complete", "receipt": existing}
    plan = build_ablation_plan(cfg)
    root = output_root(cfg) / "ablations"
    # This is deliberately the first operation capable of authorizing an
    # ablation test decode. It verifies all 320 validation locks together.
    gate = open_ablation_test_gate(cfg, plan=plan, lock_root=root / "validation_locks")
    result_paths: list[Path] = []
    for batch in build_ablation_test_schedule(plan, gate):
        if batch.phase == "evaluate_shared_once":
            for task in batch.tasks:
                result_paths.append(
                    _evaluate_fitted_ablation_task(cfg, plan, task, root, gate)
                )
            continue
        assert batch.model is not None
        primary_path = (
            unit_root(cfg, batch.model, batch.seed)
            / "evaluation"
            / PRIMARY_CLASSIFIER_STRATEGY
            / "frames.csv"
        )
        primary_frames = pd.read_csv(
            primary_path, encoding="utf-8-sig", dtype={"frame_id": str}
        )
        primary_frames["model"] = batch.model
        primary_frames["seed"] = int(batch.seed)
        variants = build_analytical_eye_variants(
            primary_frames,
            cfg,
            partition="test",
            plan=plan,
            test_gate=gate,
        )
        for task in batch.tasks:
            if task.scope is AblationScope.ANALYTICAL_MODEL_SEED:
                result_paths.append(
                    _evaluate_analytical_ablation_task(
                        cfg, plan, task, root, gate, primary_frames, variants
                    )
                )
            else:
                result_paths.append(
                    _evaluate_fitted_ablation_task(cfg, plan, task, root, gate)
                )
    if len(result_paths) != int(cfg["ablations"]["expected_test_evaluations"]):
        raise ProtocolGateError("Ablation test result cardinality does not match the protocol")
    if len({path.resolve() for path in result_paths}) != len(plan.tasks):
        raise ProtocolGateError("Each ablation task must have exactly one test result")

    path_by_key = {read_json(path)["task"]["key"]: path for path in result_paths}
    if set(path_by_key) != {task.key for task in plan.tasks}:
        raise ProtocolGateError("Ablation result keys do not exactly cover the frozen plan")
    task_artifact_paths: list[Path] = []
    for task in plan.tasks:
        result_path = path_by_key[task.key]
        result = read_json(result_path)
        task_artifact_paths.extend(
            _verified_ablation_test_artifacts(result_path, task, result)
        )
    summary_paths = _ablation_summary(cfg, plan, root, result_paths)
    # Schedules and plan order are both deterministic but differ; key the
    # manifest explicitly so schedule order cannot hide misalignment.
    entries = [
        {
            "task_key": task.key,
            "result_path": _relative(path_by_key[task.key]),
            "result_sha256": sha256_file(path_by_key[task.key]),
        }
        for task in plan.tasks
    ]
    manifest_path = _immutable_engine_json(
        root / "state" / "test_evaluation_manifest.json",
        {
            "schema": 1,
            "plan_sha256": plan.sha256,
            "ablation_test_gate_sha256": gate.sha256,
            "test_evaluation_count": len(entries),
            "test_selection_or_refitting": False,
            "entries": entries,
        },
    )
    from .protocol import stage_receipt_path

    ablation_lock_receipt = stage_receipt_path(cfg, "lock-ablations")
    # Re-verify after summary generation, then attest every task table/index
    # that fed the summaries and paired contrasts in the global receipt.
    task_artifact_paths = []
    for task in plan.tasks:
        result_path = path_by_key[task.key]
        result = read_json(result_path)
        task_artifact_paths.extend(
            _verified_ablation_test_artifacts(result_path, task, result)
        )
    artifacts = [
        gate.path,
        manifest_path,
        *result_paths,
        *task_artifact_paths,
        *summary_paths,
    ]
    unique_artifacts = list(dict.fromkeys(Path(path).resolve() for path in artifacts))
    receipt = write_stage_receipt(
        cfg,
        "evaluate-ablations",
        artifacts=unique_artifacts,
        metadata={
            "ablation_test_evaluation_count": len(result_paths),
            "all_ablation_test_evaluations_complete": True,
            "separate_immutable_ablation_summary": True,
            "primary_summary_modified": False,
            "ablation_lock_receipt_sha256": sha256_file(ablation_lock_receipt),
            "test_selection_or_refitting": False,
        },
    )
    return {
        "status": "complete",
        "receipt": str(receipt),
        "artifacts": [str(path) for path in unique_artifacts],
        "metadata": {"task_count": len(result_paths), "plan_sha256": plan.sha256},
    }


def _per_seed_primary_metrics(cfg: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    classification_rows: list[dict[str, Any]] = []
    segmentation_rows: list[dict[str, Any]] = []
    for model_name in cfg["models"]:
        for seed in cfg["split_seeds"]:
            evaluation_root = unit_root(cfg, model_name, seed) / "evaluation"
            for strategy in _classifier_strategy_names(cfg):
                directory = evaluation_root / strategy
                metrics = read_json(directory / "metrics.json")
                for level in ("eye", "patient"):
                    scope_metrics = metrics[level]["ALL"]
                    values = scope_metrics["primary_operational"]
                    classification_rows.append(
                        {
                            "model": model_name,
                            "seed": int(seed),
                            "classifier_strategy": strategy,
                            "classifier_role": _classifier_strategy_specs(cfg)[strategy].get(
                                "role"
                            ),
                            "classifier_estimand_id": _classifier_strategy_specs(cfg)[strategy].get(
                                "estimand_id"
                            ),
                            "is_primary_estimand": bool(
                                strategy == PRIMARY_CLASSIFIER_STRATEGY
                            ),
                            "level": level,
                            "scope": "ALL",
                            "arm": "predicted_roi",
                            "operational_probability_scale": scope_metrics[
                                "operating_thresholds"
                            ]["probability_scale"],
                            **flatten_metrics(values),
                            **flatten_metrics(scope_metrics["raw"], "raw"),
                            **flatten_metrics(
                                scope_metrics["primary_calibrated"],
                                "temperature_scaled",
                            ),
                        }
                    )
            directory = evaluation_root / PRIMARY_CLASSIFIER_STRATEGY
            level_tables = {
                "frame": pd.read_csv(
                    directory / "frames.csv", encoding="utf-8-sig", dtype={"frame_id": str}
                ),
                "eye": pd.read_csv(directory / "eyes.csv", encoding="utf-8-sig"),
                "patient": pd.read_csv(directory / "patients.csv", encoding="utf-8-sig"),
            }
            metric_names = [
                *cfg["segmentation"]["metrics"],
                *[f"raw_{metric}" for metric in cfg["segmentation"]["metrics"]],
            ]
            for level, table in level_tables.items():
                if level == "frame":
                    evaluable = table.segmentation_roi_valid.astype(bool)
                else:
                    evaluable = table.segmentation_roi_evaluable.astype(bool)
                row: dict[str, Any] = {
                    "model": model_name,
                    "seed": int(seed),
                    "classifier_strategy": PRIMARY_CLASSIFIER_STRATEGY,
                    "level": level,
                    "scope": "ALL",
                    "n_units": int(len(table)),
                    "n_roi_evaluable": int(evaluable.sum()),
                    "roi_coverage": float(evaluable.mean()),
                }
                for metric in metric_names:
                    if metric in table:
                        values = pd.to_numeric(table[metric], errors="coerce")
                        row[metric] = float(values.mean())
                        row[f"{metric}.n_nonmissing"] = int(values.notna().sum())
                segmentation_rows.append(row)
    return pd.DataFrame(classification_rows), pd.DataFrame(segmentation_rows)


def _aligned_unit_pair(
    left: pd.DataFrame, right: pd.DataFrame, *, level: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align a model pair and prove that identities and labels are unchanged."""

    keys = ["patient_id"] if level == "patient" else ["patient_id", "case_id", "side"]
    identities = [*keys, "label"]
    if "label_3class" in left and "label_3class" in right:
        identities.append("label_3class")
    for name, table in (("left", left), ("right", right)):
        missing = set(identities) - set(table)
        if missing or table.duplicated(keys).any():
            raise ProtocolGateError(
                f"Malformed {level} comparison table ({name}); missing={sorted(missing)}"
            )
    a = left.copy()
    b = right.copy()
    for column in keys:
        a[column] = a[column].astype(str)
        b[column] = b[column].astype(str)
    a = a.sort_values(keys, kind="stable").reset_index(drop=True)
    b = b.sort_values(keys, kind="stable").reset_index(drop=True)
    if len(a) != len(b):
        raise ProtocolGateError("Model comparison tables have different unit counts")
    for column in identities:
        if not np.array_equal(a[column].astype(str).to_numpy(), b[column].astype(str).to_numpy()):
            raise ProtocolGateError(f"Model comparison mismatch in {column}")
    return a, b


def _failure_aware_balanced_accuracy_from_prediction(table: pd.DataFrame) -> float:
    labels = table.label.to_numpy(int)
    prediction = table.prediction.to_numpy(int)
    sensitivity = (
        float(((prediction == 1) & (labels == 1)).sum()) / int((labels == 1).sum())
        if int((labels == 1).sum()) else np.nan
    )
    specificity = (
        float(((prediction == 0) & (labels == 0)).sum()) / int((labels == 0).sum())
        if int((labels == 0).sum()) else np.nan
    )
    return (
        float((sensitivity + specificity) / 2)
        if np.isfinite(sensitivity) and np.isfinite(specificity)
        else np.nan
    )


def _bca_interval(
    estimate: float, bootstrap: np.ndarray, jackknife: np.ndarray
) -> tuple[float, float, str]:
    finite = bootstrap[np.isfinite(bootstrap)]
    low = float(np.percentile(finite, 2.5)) if len(finite) else np.nan
    high = float(np.percentile(finite, 97.5)) if len(finite) else np.nan
    if (
        len(finite) == 0
        or len(jackknife) < 3
        or not np.isfinite(estimate)
        or not np.isfinite(jackknife).all()
    ):
        return low, high, "percentile"
    proportion = (
        np.sum(finite < estimate) + 0.5 * np.sum(finite == estimate)
    ) / len(finite)
    proportion = float(
        np.clip(proportion, 1 / (2 * len(finite)), 1 - 1 / (2 * len(finite)))
    )
    z0 = float(special.ndtri(proportion))
    deviations = float(jackknife.mean()) - jackknife
    denominator = 6 * float(np.sum(deviations**2) ** 1.5)
    acceleration = (
        float(np.sum(deviations**3)) / denominator if denominator > 0 else 0.0
    )
    adjusted: list[float] = []
    for alpha_value in (0.025, 0.975):
        z_alpha = float(special.ndtri(alpha_value))
        divisor = 1 - acceleration * (z0 + z_alpha)
        if abs(divisor) <= 1e-12:
            return low, high, "percentile"
        adjusted.append(float(special.ndtr(z0 + (z0 + z_alpha) / divisor)))
    if not np.isfinite(adjusted).all() or adjusted[0] > adjusted[1]:
        return low, high, "percentile"
    low, high = map(float, np.quantile(finite, np.clip(adjusted, 0, 1)))
    return low, high, "bca"


def _paired_locked_primary_effect(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    """Paired patient bootstrap CI plus patient-label-swap randomization p."""

    patient_rows: list[dict[str, Any]] = []
    for patient_id, a_group in left.groupby("patient_id", sort=True):
        b_group = right[right.patient_id == patient_id]
        if len(a_group) != len(b_group) or a_group.label.nunique() != 1:
            raise ProtocolGateError("Paired comparison changed a patient's unit count or label")
        label = int(a_group.label.iloc[0])
        a_fraction = float((a_group.prediction.to_numpy(int) == label).mean())
        b_fraction = float((b_group.prediction.to_numpy(int) == label).mean())
        patient_rows.append(
            {
                "patient_id": str(patient_id),
                "label": label,
                "difference": a_fraction - b_fraction,
                "unit_count": len(a_group),
            }
        )
    patients = pd.DataFrame(patient_rows)
    if set(patients.label.astype(int)) != {0, 1}:
        raise ProtocolGateError("Primary paired comparison requires both patient label strata")

    def balanced_difference(frame: pd.DataFrame) -> float:
        return float(
            0.5
            * sum(
                frame.loc[frame.label == label, "difference"].mean()
                for label in (0, 1)
            )
        )

    estimate = balanced_difference(patients)
    rng = np.random.default_rng(seed)
    strata = {
        label: patients.loc[patients.label == label, "difference"].to_numpy(float)
        for label in (0, 1)
    }
    bootstrap_array = 0.5 * sum(
        values[
            rng.integers(0, len(values), size=(draws, len(values)))
        ].mean(axis=1)
        for values in strata.values()
    )
    jackknife = np.asarray(
        [
            balanced_difference(patients.drop(index=index))
            for index in patients.index
        ],
        dtype=float,
    )
    low, high, ci_method = _bca_interval(
        estimate, bootstrap_array, jackknife
    )

    # Swap the two complete model outcomes within whole patients.  This is a
    # valid paired randomization null and preserves both eyes of a patient.
    randomized_by_stratum: list[np.ndarray] = []
    for values in strata.values():
        signs = (rng.integers(0, 2, size=(draws, len(values))) * 2 - 1).astype(float)
        randomized_by_stratum.append((signs * values).mean(axis=1))
    randomized_array = 0.5 * sum(randomized_by_stratum)
    valid_randomized = randomized_array[np.isfinite(randomized_array)]
    p_value = (
        (1 + int((np.abs(valid_randomized) >= abs(estimate) - 1e-15).sum()))
        / (len(valid_randomized) + 1)
        if len(valid_randomized) and np.isfinite(estimate)
        else np.nan
    )
    return {
        "metric": "failure_inclusive.balanced_accuracy",
        "estimate": float(estimate),
        "low": low,
        "high": high,
        "ci_method": ci_method,
        "bootstrap_se": float(bootstrap_array.std(ddof=1)),
        "valid_bootstrap_draws": int(np.isfinite(bootstrap_array).sum()),
        "requested_draws": int(draws),
        "p_value_randomization_two_sided": float(p_value),
        "valid_randomization_draws": int(len(valid_randomized)),
        "p_value_method": "whole-patient paired model-label-swap Monte Carlo; plus-one correction",
    }


def _primary_model_comparisons(cfg: Mapping[str, Any], destination: Path) -> list[Path]:
    """Paired four-model comparisons, kept separate within each strategy."""

    cluster_rows: list[dict[str, Any]] = []
    classical_rows: list[dict[str, Any]] = []
    draws = int(cfg["statistics"]["bootstrap_draws"])
    base_seed = int(cfg["statistics"]["bootstrap_seed"])
    pair_count = math.comb(len(cfg["models"]), 2)
    planned_family_size = pair_count * len(cfg["split_seeds"])
    for strategy_index, strategy in enumerate(_classifier_strategy_names(cfg)):
        strategy_spec = _classifier_strategy_specs(cfg)[strategy]
        for seed in cfg["split_seeds"]:
            tables = {
                level: {
                    model_name: pd.read_csv(
                        unit_root(cfg, model_name, seed)
                        / "evaluation"
                        / strategy
                        / f"{level}s.csv",
                        encoding="utf-8-sig",
                        dtype={"frame_id": str},
                    )
                    for model_name in cfg["models"]
                }
                for level in ("eye", "patient")
            }
            for pair_index, (model_a, model_b) in enumerate(
                itertools.combinations(cfg["models"], 2)
            ):
                eyes_a, eyes_b = _aligned_unit_pair(
                    tables["eye"][model_a], tables["eye"][model_b], level="eye"
                )
                primary = _paired_locked_primary_effect(
                    eyes_a,
                    eyes_b,
                    draws=draws,
                    seed=(
                        base_seed + int(seed) * 101 + pair_index
                        + strategy_index * 1_000_003
                    ),
                )
                cluster_rows.append(
                    {
                        "seed": int(seed),
                        "classifier_strategy": strategy,
                        "classifier_role": strategy_spec.get("role"),
                        "classifier_estimand_id": strategy_spec.get("estimand_id"),
                        "is_primary_estimand": strategy == PRIMARY_CLASSIFIER_STRATEGY,
                        "level": "eye",
                        "model_a": model_a,
                        "model_b": model_b,
                        "contrast": f"{model_a} - {model_b}",
                        "family_id": (
                            f"{strategy}_eye_failure_aware_balanced_accuracy"
                        ),
                        "planned_family_size": planned_family_size,
                        **primary,
                    }
                )
                for level in ("eye", "patient"):
                    a, b = _aligned_unit_pair(
                        tables[level][model_a], tables[level][model_b], level=level
                    )
                    delong = delong_auc_comparison(
                        a.label,
                        a.probability_raw,
                        b.probability_raw,
                        evaluable_a=a.roi_evaluable,
                        evaluable_b=b.roi_evaluable,
                    )
                    classical_rows.append(
                        {
                            "seed": int(seed),
                            "classifier_strategy": strategy,
                            "classifier_role": strategy_spec.get("role"),
                            "classifier_estimand_id": strategy_spec.get("estimand_id"),
                            "is_primary_estimand": strategy == PRIMARY_CLASSIFIER_STRATEGY,
                            "level": level,
                            "model_a": model_a,
                            "model_b": model_b,
                            "test": "delong_raw_common_roi_evaluable_descriptive",
                            "effect": delong["difference_a_minus_b"],
                            "p_value": delong["p_value"],
                            "n_common_evaluable": delong["n_common_evaluable"],
                            "status": delong["status"],
                            "note": delong["assumption_note"],
                        }
                    )
                    common = a.evaluable.astype(bool) & b.evaluable.astype(bool)
                    if common.any():
                        mcnemar = mcnemar_exact_comparison(
                            a.label,
                            a.prediction.replace(-1, np.nan),
                            b.prediction.replace(-1, np.nan),
                            threshold_a=0.5,
                            threshold_b=0.5,
                            evaluable_a=common,
                            evaluable_b=common,
                        )
                    else:
                        mcnemar = {
                            "conditional_accuracy_difference_a_minus_b": np.nan,
                            "p_value_exact_two_sided": np.nan,
                            "n_common_evaluable": 0,
                            "status": "undefined_no_common_evaluable",
                            "assumption_note": "No common operationally evaluable units.",
                        }
                    classical_rows.append(
                        {
                            "seed": int(seed),
                            "classifier_strategy": strategy,
                            "classifier_role": strategy_spec.get("role"),
                            "classifier_estimand_id": strategy_spec.get("estimand_id"),
                            "is_primary_estimand": strategy == PRIMARY_CLASSIFIER_STRATEGY,
                            "level": level,
                            "model_a": model_a,
                            "model_b": model_b,
                            "test": "mcnemar_locked_predictions_common_evaluable_descriptive",
                            "effect": mcnemar[
                                "conditional_accuracy_difference_a_minus_b"
                            ],
                            "p_value": mcnemar["p_value_exact_two_sided"],
                            "n_common_evaluable": mcnemar["n_common_evaluable"],
                            "status": mcnemar["status"],
                            "note": mcnemar["assumption_note"],
                        }
                    )
    cluster_table = pd.DataFrame(cluster_rows)
    cluster_table["p_value_holm"] = np.nan
    cluster_table["reject_holm_0_05"] = False
    for _, indices in cluster_table.groupby("family_id").groups.items():
        adjusted, rejected = holm_adjust(
            cluster_table.loc[indices, "p_value_randomization_two_sided"],
            alpha=float(cfg["statistics"]["alpha"]),
            planned_family_size=planned_family_size,
        )
        cluster_table.loc[indices, "p_value_holm"] = adjusted
        cluster_table.loc[indices, "reject_holm_0_05"] = rejected
    classical_table = pd.DataFrame(classical_rows)
    classical_table["p_value_holm"] = np.nan
    classical_table["reject_holm_0_05"] = False
    for (_, test, level), indices in classical_table.groupby(
        ["classifier_strategy", "test", "level"]
    ).groups.items():
        adjusted, rejected = holm_adjust(
            classical_table.loc[indices, "p_value"],
            alpha=float(cfg["statistics"]["alpha"]),
            planned_family_size=planned_family_size,
        )
        classical_table.loc[indices, "p_value_holm"] = adjusted
        classical_table.loc[indices, "reject_holm_0_05"] = rejected
    cluster_path = destination / "paired_model_patient_cluster_bootstrap.csv"
    classical_path = destination / "paired_model_classical_sensitivity_holm.csv"
    _save_csv_atomic(cluster_path, cluster_table)
    _save_csv_atomic(classical_path, classical_table)
    return [cluster_path, classical_path]


def _within_segmenter_classifier_strategy_comparisons(
    cfg: Mapping[str, Any], destination: Path
) -> list[Path]:
    """Paired E1-vs-E2 effects within each segmenter and frozen split."""

    cluster_rows: list[dict[str, Any]] = []
    classical_rows: list[dict[str, Any]] = []
    draws = int(cfg["statistics"]["bootstrap_draws"])
    base_seed = int(cfg["statistics"]["bootstrap_seed"])
    family_size = len(cfg["models"]) * len(cfg["split_seeds"])
    for model_index, model_name in enumerate(cfg["models"]):
        for seed in cfg["split_seeds"]:
            tables = {
                strategy: {
                    level: pd.read_csv(
                        unit_root(cfg, model_name, seed)
                        / "evaluation"
                        / strategy
                        / f"{level}s.csv",
                        encoding="utf-8-sig",
                    )
                    for level in ("eye", "patient")
                }
                for strategy in _classifier_strategy_names(cfg)
            }
            for level_index, level in enumerate(("eye", "patient")):
                primary, standardized = _aligned_unit_pair(
                    tables[PRIMARY_CLASSIFIER_STRATEGY][level],
                    tables[STANDARDIZED_CLASSIFIER_STRATEGY][level],
                    level=level,
                )
                effect = _paired_locked_primary_effect(
                    primary,
                    standardized,
                    draws=draws,
                    seed=(
                        base_seed + int(seed) * 90_011 + model_index * 101 + level_index
                    ),
                )
                cluster_rows.append(
                    {
                        "model": model_name,
                        "seed": int(seed),
                        "level": level,
                        "strategy_a": PRIMARY_CLASSIFIER_STRATEGY,
                        "strategy_b": STANDARDIZED_CLASSIFIER_STRATEGY,
                        "contrast": "model_specific - standardized_resnet18",
                        "estimand": "E3_model_specific_minus_standardized_within_segmenter",
                        "family_id": f"classifier_strategy_{level}_failure_aware_balanced_accuracy",
                        "planned_family_size": family_size,
                        **effect,
                    }
                )
                delong = delong_auc_comparison(
                    primary.label,
                    primary.probability_raw,
                    standardized.probability_raw,
                    evaluable_a=primary.roi_evaluable,
                    evaluable_b=standardized.roi_evaluable,
                )
                classical_rows.append(
                    {
                        "model": model_name,
                        "seed": int(seed),
                        "level": level,
                        "strategy_a": PRIMARY_CLASSIFIER_STRATEGY,
                        "strategy_b": STANDARDIZED_CLASSIFIER_STRATEGY,
                        "contrast": "model_specific - standardized_resnet18",
                        "estimand": "E3_model_specific_minus_standardized_within_segmenter",
                        "test": "delong_raw_common_roi_evaluable_descriptive",
                        "effect": delong["difference_a_minus_b"],
                        "p_value": delong["p_value"],
                        "n_common_evaluable": delong["n_common_evaluable"],
                        "status": delong["status"],
                        "note": delong["assumption_note"],
                    }
                )
                common = (
                    primary.evaluable.astype(bool)
                    & standardized.evaluable.astype(bool)
                )
                if common.any():
                    mcnemar = mcnemar_exact_comparison(
                        primary.label,
                        primary.prediction.replace(-1, np.nan),
                        standardized.prediction.replace(-1, np.nan),
                        threshold_a=0.5,
                        threshold_b=0.5,
                        evaluable_a=common,
                        evaluable_b=common,
                    )
                else:
                    mcnemar = {
                        "conditional_accuracy_difference_a_minus_b": np.nan,
                        "p_value_exact_two_sided": np.nan,
                        "n_common_evaluable": 0,
                        "status": "undefined_no_common_evaluable",
                        "assumption_note": "No common operationally evaluable units.",
                    }
                classical_rows.append(
                    {
                        "model": model_name,
                        "seed": int(seed),
                        "level": level,
                        "strategy_a": PRIMARY_CLASSIFIER_STRATEGY,
                        "strategy_b": STANDARDIZED_CLASSIFIER_STRATEGY,
                        "contrast": "model_specific - standardized_resnet18",
                        "estimand": "E3_model_specific_minus_standardized_within_segmenter",
                        "test": "mcnemar_locked_predictions_common_evaluable_descriptive",
                        "effect": mcnemar[
                            "conditional_accuracy_difference_a_minus_b"
                        ],
                        "p_value": mcnemar["p_value_exact_two_sided"],
                        "n_common_evaluable": mcnemar["n_common_evaluable"],
                        "status": mcnemar["status"],
                        "note": mcnemar["assumption_note"],
                    }
                )
    cluster = pd.DataFrame(cluster_rows)
    cluster["p_value_holm"] = np.nan
    cluster["reject_holm_0_05"] = False
    for _, indices in cluster.groupby("family_id").groups.items():
        adjusted, rejected = holm_adjust(
            cluster.loc[indices, "p_value_randomization_two_sided"],
            alpha=float(cfg["statistics"]["alpha"]),
            planned_family_size=family_size,
        )
        cluster.loc[indices, "p_value_holm"] = adjusted
        cluster.loc[indices, "reject_holm_0_05"] = rejected
    classical = pd.DataFrame(classical_rows)
    classical["p_value_holm"] = np.nan
    classical["reject_holm_0_05"] = False
    for (_, _), indices in classical.groupby(["test", "level"]).groups.items():
        adjusted, rejected = holm_adjust(
            classical.loc[indices, "p_value"],
            alpha=float(cfg["statistics"]["alpha"]),
            planned_family_size=family_size,
        )
        classical.loc[indices, "p_value_holm"] = adjusted
        classical.loc[indices, "reject_holm_0_05"] = rejected
    cluster_path = destination / "paired_within_segmenter_classifier_strategy_patient_cluster_holm.csv"
    classical_path = destination / "paired_within_segmenter_classifier_strategy_classical_holm.csv"
    _save_csv_atomic(cluster_path, cluster)
    _save_csv_atomic(classical_path, classical)
    return [cluster_path, classical_path]


def _aligned_frame_pair(
    left: pd.DataFrame, right: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["patient_id", "case_id", "side", "frame_id"]
    identities = [*keys, "label", "label_3class"]
    for name, table in (("left", left), ("right", right)):
        missing = set(identities) - set(table)
        if missing or table.duplicated(keys).any():
            raise ProtocolGateError(
                f"Malformed frame comparison table ({name}); missing={sorted(missing)}"
            )
    a, b = left.copy(), right.copy()
    for column in keys:
        a[column] = a[column].astype(str)
        b[column] = b[column].astype(str)
    a = a.sort_values(keys, kind="stable").reset_index(drop=True)
    b = b.sort_values(keys, kind="stable").reset_index(drop=True)
    if len(a) != len(b):
        raise ProtocolGateError("Segmentation comparison frame counts differ")
    for column in identities:
        if not np.array_equal(a[column].astype(str).to_numpy(), b[column].astype(str).to_numpy()):
            raise ProtocolGateError(f"Segmentation comparison mismatch in {column}")
    return a, b


def _paired_patient_continuous_effect(
    patient_differences: pd.DataFrame, *, draws: int, seed: int
) -> dict[str, Any]:
    data = patient_differences.dropna(subset=["difference"]).reset_index(drop=True)
    estimate = float(data.difference.mean()) if len(data) else np.nan
    if not len(data) or data.label.nunique() < 2:
        return {
            "estimate": estimate,
            "low": np.nan,
            "high": np.nan,
            "ci_method": "undefined_requires_both_label_strata",
            "bootstrap_se": np.nan,
            "valid_bootstrap_draws": 0,
            "p_value_randomization_two_sided": np.nan,
            "valid_randomization_draws": 0,
        }
    rng = np.random.default_rng(seed)
    strata = [
        group.difference.to_numpy(float)
        for _, group in data.groupby("label", sort=True)
    ]
    weights = [len(values) / len(data) for values in strata]
    bootstrap_array = sum(
        weight
        * values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
        for weight, values in zip(weights, strata, strict=True)
    )
    jackknife = np.asarray(
        [float(data.drop(index=index).difference.mean()) for index in data.index],
        dtype=float,
    )
    low, high, method = _bca_interval(estimate, bootstrap_array, jackknife)
    signs = (rng.integers(0, 2, size=(draws, len(data))) * 2 - 1).astype(float)
    randomized = (signs * data.difference.to_numpy(float)).mean(axis=1)
    p_value = (
        1 + int((np.abs(randomized) >= abs(estimate) - 1e-15).sum())
    ) / (draws + 1)
    return {
        "estimate": estimate,
        "low": low,
        "high": high,
        "ci_method": method,
        "bootstrap_se": float(bootstrap_array.std(ddof=1)),
        "valid_bootstrap_draws": int(np.isfinite(bootstrap_array).sum()),
        "p_value_randomization_two_sided": float(p_value),
        "valid_randomization_draws": int(draws),
    }


def _segmentation_model_comparisons(
    cfg: Mapping[str, Any], destination: Path
) -> Path:
    """Patient-cluster paired effects for raw and post-processed masks."""

    core_metrics = list(cfg["statistics"]["paired_segmentation_metrics"])
    draws = int(cfg["statistics"]["bootstrap_draws"])
    base_seed = int(cfg["statistics"]["bootstrap_seed"])
    family_size = math.comb(len(cfg["models"]), 2) * len(cfg["split_seeds"])
    rows: list[dict[str, Any]] = []
    for seed in cfg["split_seeds"]:
        tables = {
            model_name: pd.read_csv(
                unit_root(cfg, model_name, seed)
                / "evaluation"
                / PRIMARY_CLASSIFIER_STRATEGY
                / "frames.csv",
                encoding="utf-8-sig",
                dtype={"frame_id": str},
            )
            for model_name in cfg["models"]
        }
        for pair_index, (model_a, model_b) in enumerate(
            itertools.combinations(cfg["models"], 2)
        ):
            a, b = _aligned_frame_pair(tables[model_a], tables[model_b])
            prefixes = {"postprocessed": "", "raw": "raw_"}
            for representation in cfg["statistics"]["paired_segmentation_representations"]:
                prefix = prefixes[str(representation)]
                for metric_index, metric in enumerate(core_metrics):
                    column = f"{prefix}{metric}"
                    av = pd.to_numeric(a[column], errors="coerce")
                    bv = pd.to_numeric(b[column], errors="coerce")
                    common = av.notna() & bv.notna()
                    paired = pd.DataFrame(
                        {
                            "patient_id": a.loc[common, "patient_id"].astype(str),
                            "label": a.loc[common, "label"].astype(int),
                            "difference": (av[common] - bv[common]).astype(float),
                        }
                    )
                    patient = (
                        paired.groupby(["patient_id", "label"], as_index=False)
                        .difference.mean()
                    )
                    effect = _paired_patient_continuous_effect(
                        patient,
                        draws=draws,
                        seed=(
                            base_seed
                            + int(seed) * 10_007
                            + pair_index * 101
                            + metric_index * 2
                            + int(representation == "raw")
                        ),
                    )
                    rows.append(
                        {
                            "seed": int(seed),
                            "model_a": model_a,
                            "model_b": model_b,
                            "contrast": f"{model_a} - {model_b}",
                            "representation": representation,
                            "metric": metric,
                            "family_id": f"segmentation_{representation}_{metric}",
                            "planned_family_size": family_size,
                            "n_total_frames": int(len(a)),
                            "n_common_metric_frames": int(common.sum()),
                            "n_patients_with_common_metric": int(patient.patient_id.nunique()),
                            **effect,
                        }
                    )
    table = pd.DataFrame(rows)
    table["p_value_holm"] = np.nan
    table["reject_holm_0_05"] = False
    for _, indices in table.groupby("family_id").groups.items():
        adjusted, rejected = holm_adjust(
            table.loc[indices, "p_value_randomization_two_sided"],
            alpha=float(cfg["statistics"]["alpha"]),
            planned_family_size=family_size,
        )
        table.loc[indices, "p_value_holm"] = adjusted
        table.loc[indices, "reject_holm_0_05"] = rejected
    path = destination / "paired_segmentation_patient_cluster_effects_holm.csv"
    _save_csv_atomic(path, table)
    return path


def _calibration_loss_model_comparisons(
    cfg: Mapping[str, Any], destination: Path
) -> Path:
    """Paired Brier/NLL model effects in separate E1 and E2 families."""

    draws = int(cfg["statistics"]["bootstrap_draws"])
    base_seed = int(cfg["statistics"]["bootstrap_seed"])
    epsilon = float(cfg["calibration"]["probability_clip_epsilon"])
    family_size = math.comb(len(cfg["models"]), 2) * len(cfg["split_seeds"])
    rows: list[dict[str, Any]] = []
    for strategy_index, strategy in enumerate(_classifier_strategy_names(cfg)):
        strategy_spec = _classifier_strategy_specs(cfg)[strategy]
        for seed in cfg["split_seeds"]:
            tables = {
                level: {
                    model_name: pd.read_csv(
                        unit_root(cfg, model_name, seed)
                        / "evaluation" / strategy / f"{level}s.csv",
                        encoding="utf-8-sig",
                    )
                    for model_name in cfg["models"]
                }
                for level in ("eye", "patient")
            }
            for pair_index, (model_a, model_b) in enumerate(
                itertools.combinations(cfg["models"], 2)
            ):
                for level_index, level in enumerate(("eye", "patient")):
                    a, b = _aligned_unit_pair(
                        tables[level][model_a], tables[level][model_b], level=level
                    )
                    scales = (
                        ("raw", "probability_raw", "roi_evaluable"),
                        ("temperature_scaled", "probability_calibrated", "calibrated_evaluable"),
                    )
                    for scale_index, (scale, probability_column, validity_column) in enumerate(scales):
                        pa = pd.to_numeric(a[probability_column], errors="coerce")
                        pb = pd.to_numeric(b[probability_column], errors="coerce")
                        common = (
                            a[validity_column].astype(bool)
                            & b[validity_column].astype(bool)
                            & pa.notna() & pb.notna()
                        )
                        labels = a.loc[common, "label"].to_numpy(float)
                        values_a = pa[common].to_numpy(float)
                        values_b = pb[common].to_numpy(float)
                        for metric_index, metric in enumerate(
                            cfg["statistics"]["paired_calibration_loss_tests"]
                        ):
                            if metric == "brier":
                                differences = (values_a - labels) ** 2 - (values_b - labels) ** 2
                            else:
                                clipped_a = np.clip(values_a, epsilon, 1 - epsilon)
                                clipped_b = np.clip(values_b, epsilon, 1 - epsilon)
                                losses_a = -(labels * np.log(clipped_a) + (1 - labels) * np.log1p(-clipped_a))
                                losses_b = -(labels * np.log(clipped_b) + (1 - labels) * np.log1p(-clipped_b))
                                differences = losses_a - losses_b
                            paired = pd.DataFrame(
                                {
                                    "patient_id": a.loc[common, "patient_id"].astype(str).to_numpy(),
                                    "label": a.loc[common, "label"].astype(int).to_numpy(),
                                    "difference": differences,
                                }
                            )
                            patient = paired.groupby(
                                ["patient_id", "label"], as_index=False
                            ).difference.mean()
                            effect = _paired_patient_continuous_effect(
                                patient,
                                draws=draws,
                                seed=(
                                    base_seed + int(seed) * 20_011 + pair_index * 211
                                    + level_index * 17 + scale_index * 5 + metric_index
                                    + strategy_index * 1_000_003
                                ),
                            )
                            rows.append(
                                {
                                    "seed": int(seed),
                                    "classifier_strategy": strategy,
                                    "classifier_role": strategy_spec.get("role"),
                                    "classifier_estimand_id": strategy_spec.get("estimand_id"),
                                    "is_primary_estimand": strategy == PRIMARY_CLASSIFIER_STRATEGY,
                                    "level": level,
                                    "model_a": model_a,
                                    "model_b": model_b,
                                    "contrast": f"{model_a} - {model_b}",
                                    "probability_scale": scale,
                                    "metric": metric,
                                    "effect_direction": "positive_means_model_a_has_worse_loss",
                                    "family_id": f"{strategy}_calibration_loss_{level}_{scale}_{metric}",
                                    "planned_family_size": family_size,
                                    "n_total_units": int(len(a)),
                                    "n_common_score_units": int(common.sum()),
                                    "n_patients_with_common_scores": int(patient.patient_id.nunique()),
                                    **effect,
                                }
                            )
    table = pd.DataFrame(rows)
    table["p_value_holm"] = np.nan
    table["reject_holm_0_05"] = False
    for _, indices in table.groupby("family_id").groups.items():
        adjusted, rejected = holm_adjust(
            table.loc[indices, "p_value_randomization_two_sided"],
            alpha=float(cfg["statistics"]["alpha"]),
            planned_family_size=family_size,
        )
        table.loc[indices, "p_value_holm"] = adjusted
        table.loc[indices, "reject_holm_0_05"] = rejected
    path = destination / "paired_calibration_loss_patient_cluster_effects_holm.csv"
    _save_csv_atomic(path, table)
    return path


def _primary_qualitative_outputs(
    cfg: Mapping[str, Any], destination: Path
) -> list[Path]:
    """Render prelocked common examples and clearly labeled failure examples."""

    source_rows: dict[int, pd.DataFrame] = {}
    model_frames: dict[int, dict[str, pd.DataFrame]] = {}
    dataset_roots: set[Path] = set()
    for seed in cfg["split_seeds"]:
        dataset_root, rows = split_frames(cfg, int(seed), "test")
        dataset_roots.add(dataset_root.resolve())
        source_rows[int(seed)] = rows
        model_frames[int(seed)] = {
            model_name: pd.read_csv(
                unit_root(cfg, model_name, int(seed))
                / "evaluation"
                / PRIMARY_CLASSIFIER_STRATEGY
                / "frames.csv",
                encoding="utf-8-sig",
                dtype={"frame_id": str},
            )
            for model_name in cfg["models"]
        }
    if len(dataset_roots) != 1:
        raise ProtocolGateError("Qualitative rendering requires one locked dataset root")
    dataset_root = next(iter(dataset_roots))
    qualitative_root = destination / "qualitative"
    common_lock_path = output_root(cfg) / "qualitative" / "common_selection_lock.json"
    verify_common_selection_lock(common_lock_path, source_rows)
    common = render_common_segmentation_gallery(
        common_lock_path,
        source_rows,
        model_frames,
        dataset_root=dataset_root,
        output_dir=qualitative_root / "common_model_blind",
        models=cfg["models"],
    )
    failure_selection = select_failure_examples(
        source_rows,
        model_frames,
        models=cfg["models"],
        statuses=cfg["reporting"]["failure_gallery"],
        include_unobserved=True,
    )
    failure_manifest_path = qualitative_root / "failure_selection_manifest.json"
    write_failure_selection_manifest(
        failure_manifest_path,
        failure_selection,
        source_rows_by_seed=source_rows,
    )
    failures = render_failure_gallery(
        source_rows,
        model_frames,
        failure_selection,
        dataset_root=dataset_root,
        output_dir=qualitative_root / "outcome_conditioned_failures",
        models=cfg["models"],
    )
    artifacts = [
        common_lock_path,
        common.index_path,
        common.metadata_path,
        *common.figures,
        failure_manifest_path,
        failures.index_path,
        failures.metadata_path,
        *failures.figures,
    ]
    # Per-figure sidecars carry source/audit hashes and interpretation labels.
    artifacts.extend((qualitative_root / "common_model_blind").glob("*.json"))
    artifacts.extend(
        (qualitative_root / "outcome_conditioned_failures").glob("*.json")
    )
    return list(dict.fromkeys(Path(path).resolve() for path in artifacts))


def summarize(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Create descriptive five-seed tables only after all primary evaluations."""

    assert_prerequisites(cfg, "summarize")
    existing = _completed_stage_receipt_or_none(cfg, "summarize")
    if existing is not None:
        return {"status": "already_complete", "receipt": existing}
    destination = output_root(cfg) / "summary"
    destination.mkdir(parents=True, exist_ok=True)
    classification, segmentation = _per_seed_primary_metrics(cfg)
    per_seed_classification_path = destination / "classification_per_seed.csv"
    per_seed_segmentation_path = destination / "segmentation_per_seed.csv"
    _save_csv_atomic(per_seed_classification_path, classification)
    _save_csv_atomic(per_seed_segmentation_path, segmentation)
    preferred = [
        "coverage",
        "coverage_negative",
        "coverage_positive",
        "selective_risk",
        "failure_aware_aurc",
        "conditional.auroc",
        "conditional.average_precision",
        "conditional.accuracy",
        "conditional.balanced_accuracy",
        "conditional.sensitivity",
        "conditional.specificity",
        "conditional.ppv",
        "conditional.npv",
        "conditional.f1",
        "conditional.mcc",
        "conditional.brier",
        "conditional.nll",
        "failure_inclusive.accuracy",
        "failure_inclusive.balanced_accuracy",
        "failure_inclusive.sensitivity",
        "failure_inclusive.specificity",
        "calibration.ece_equal_width",
        "calibration.ece_equal_mass",
        "calibration.calibration_intercept",
        "calibration.calibration_slope",
    ]
    classification_metrics = [
        candidate
        for metric in preferred
        for candidate in (
            metric,
            f"raw.{metric}",
            f"temperature_scaled.{metric}",
        )
        if candidate in classification
    ]
    classification_summary_path = destination / "classification_five_seed_mean_sd.csv"
    export_five_seed_summary(
        classification,
        classification_summary_path,
        metric_columns=classification_metrics,
        group_columns=(
            "model", "classifier_strategy", "classifier_role",
            "classifier_estimand_id", "is_primary_estimand", "level", "scope", "arm",
        ),
    )
    strategy_summary_paths: list[Path] = []
    for strategy in _classifier_strategy_names(cfg):
        strategy_rows = classification.loc[
            classification.classifier_strategy == strategy
        ].copy()
        role = "primary" if strategy == PRIMARY_CLASSIFIER_STRATEGY else "secondary"
        per_strategy_path = destination / f"classification_{role}_{strategy}_per_seed.csv"
        five_seed_strategy_path = (
            destination / f"classification_{role}_{strategy}_five_seed_mean_sd.csv"
        )
        _save_csv_atomic(per_strategy_path, strategy_rows)
        export_five_seed_summary(
            strategy_rows,
            five_seed_strategy_path,
            metric_columns=classification_metrics,
            group_columns=(
                "model", "classifier_strategy", "classifier_role",
                "classifier_estimand_id", "is_primary_estimand", "level", "scope", "arm",
            ),
        )
        strategy_summary_paths.extend([per_strategy_path, five_seed_strategy_path])
    segmentation_metrics_columns = [
        column
        for column in [
            "roi_coverage",
            *cfg["segmentation"]["metrics"],
            *[f"raw_{m}" for m in cfg["segmentation"]["metrics"]],
        ]
        if column in segmentation
    ]
    segmentation_summary_path = destination / "segmentation_five_seed_mean_sd.csv"
    export_five_seed_summary(
        segmentation,
        segmentation_summary_path,
        metric_columns=segmentation_metrics_columns,
        group_columns=("model", "classifier_strategy", "level", "scope"),
    )
    comparison_paths = _primary_model_comparisons(cfg, destination)
    strategy_comparison_paths = _within_segmenter_classifier_strategy_comparisons(
        cfg, destination
    )
    segmentation_comparison_path = _segmentation_model_comparisons(cfg, destination)
    calibration_comparison_path = _calibration_loss_model_comparisons(
        cfg, destination
    )
    qualitative_paths = _primary_qualitative_outputs(cfg, destination)
    compute_table, compute_details = collect_compute_and_provenance(
        cfg, output_root=output_root(cfg)
    )
    compute_table_path = destination / "compute_and_provenance.csv"
    compute_details_path = destination / "compute_and_provenance.json"
    _save_csv_atomic(compute_table_path, compute_table)
    save_json_atomic(compute_details_path, compute_details)
    note_path = destination / "interpretation.json"
    save_json_atomic(
        note_path,
        {
            "seed_summary": "arithmetic mean plus sample standard deviation over five seed-level estimates",
            "inference_warning": (
                "The five test holdouts overlap and are not five independent cohorts. "
                "Mean plus/minus SD is descriptive; patient-cluster paired intervals within seed are primary."
            ),
            "primary_level": "eye",
            "patient_level": "secondary and lower-powered",
            "E1_primary_estimand": (
                "model_specific classifier family on strict predicted ROI; all rankings use E1 only"
            ),
            "E2_secondary_estimand": (
                "standardized_resnet18 on the identical strict predicted ROI cache; reported separately"
            ),
            "E3_paired_strategy_estimand": (
                "within-segmenter model_specific minus standardized_resnet18 paired comparison"
            ),
            "classifier_strategy_pooling_for_rankings": False,
            "demographic_subgroups": (
                "Unavailable: the locked manifest contains side and original diagnosis but no age/sex metadata."
            ),
        },
    )
    artifacts = [
        per_seed_classification_path,
        per_seed_segmentation_path,
        classification_summary_path,
        *strategy_summary_paths,
        segmentation_summary_path,
        note_path,
        *comparison_paths,
        *strategy_comparison_paths,
        segmentation_comparison_path,
        calibration_comparison_path,
        compute_table_path,
        compute_details_path,
        *qualitative_paths,
    ]
    receipt = write_stage_receipt(
        cfg,
        "summarize",
        artifacts=artifacts,
        metadata={
            "models": len(cfg["models"]),
            "seeds": len(cfg["split_seeds"]),
            "classifier_strategies": list(_classifier_strategy_names(cfg)),
            "primary_classifier_strategy": PRIMARY_CLASSIFIER_STRATEGY,
            "primary_secondary_pooled": False,
            "overlapping_holdouts_treated_as_independent": False,
        },
    )
    return {"status": "complete", "receipt": str(receipt), "artifacts": [str(path) for path in artifacts]}
