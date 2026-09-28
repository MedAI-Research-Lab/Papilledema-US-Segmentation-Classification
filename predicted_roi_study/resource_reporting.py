"""Transparent compute, parameter and runtime provenance reporting.

Generic FLOP profilers undercount dynamic/functional segmentation operators and
omit CPU ROI post-processing.  This module therefore reports measured training
time and explicit parameter inventories, and marks FLOPs as not reported rather
than emitting a misleading number.
"""
from __future__ import annotations

import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .config import resolve_project_path, sha256_file


FLOPS_RATIONALE = (
    "Not reported: dynamic YOLO instance paths, functional attention/upsampling, "
    "CPU connected-component post-processing, and abstention-dependent classifier "
    "execution make generic profiler totals incomplete and non-comparable."
)


def parameter_inventory(
    model: torch.nn.Module, *, explicitly_disabled_legacy_parameters: int = 0
) -> dict[str, int]:
    """Return semantically labeled, internally consistent parameter counts."""

    registered = int(sum(parameter.numel() for parameter in model.parameters()))
    trainable = int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
    )
    frozen = registered - trainable
    disabled = int(explicitly_disabled_legacy_parameters)
    if disabled < 0 or disabled > frozen:
        raise ValueError(
            "explicitly disabled legacy parameters must be a subset of frozen parameters"
        )
    return {
        "registered_parameters": registered,
        "trainable_updateable_parameters": trainable,
        "frozen_parameters": frozen,
        "explicitly_disabled_legacy_parameters": disabled,
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_inventory(
    *, packages: Sequence[str] = (
        "torchvision",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "Pillow",
        "timm",
        "ultralytics",
    ),
) -> dict[str, Any]:
    """Capture reproducibility-relevant software, accelerator and RNG settings."""

    cuda_available = bool(torch.cuda.is_available())
    cuda: dict[str, Any] = {
        "available": cuda_available,
        "torch_cuda_runtime": torch.version.cuda,
        "cudnn_version": (
            int(torch.backends.cudnn.version())
            if torch.backends.cudnn.is_available()
            and torch.backends.cudnn.version() is not None
            else None
        ),
    }
    if cuda_available:
        properties = torch.cuda.get_device_properties(0)
        cuda.update(
            {
                "device_name": properties.name,
                "device_capability": list(torch.cuda.get_device_capability(0)),
                "total_memory_bytes": int(properties.total_memory),
                "bfloat16_supported": bool(torch.cuda.is_bf16_supported()),
            }
        )
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "packages": {name: _package_version(name) for name in packages},
        "cuda": cuda,
        "determinism": {
            "deterministic_algorithms_enabled": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
    }


def history_summary(path: str | Path) -> dict[str, float | int | None]:
    """Summarize recorded epoch wall time without inventing missing timings."""

    source = Path(path)
    if not source.is_file():
        return {"epochs_recorded": 0, "seconds": None}
    frame = pd.read_csv(source, encoding="utf-8-sig")
    if "seconds" not in frame:
        return {"epochs_recorded": int(len(frame)), "seconds": None}
    seconds = pd.to_numeric(frame.seconds, errors="coerce")
    return {
        "epochs_recorded": int(len(frame)),
        "seconds": float(seconds.sum()) if seconds.notna().any() else None,
    }


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def collect_compute_and_provenance(
    cfg: Mapping[str, Any], *, output_root: str | Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Collect one compute/provenance row per strategy, model and outer seed."""

    root = Path(output_root)
    rows: list[dict[str, Any]] = []
    for model_name in cfg["models"]:
        for seed in cfg["split_seeds"]:
            run = root / "runs" / model_name / f"seed_{int(seed)}"
            segmenter_info_path = run / "segmenter" / "model_info.json"
            segmenter_lock_path = run / "segmenter" / "segmenter_lock.json"
            if not segmenter_info_path.is_file() or not segmenter_lock_path.is_file():
                raise FileNotFoundError(
                    f"Missing completed segmenter provenance for {model_name}/seed_{seed}"
                )
            segmenter_info = json.loads(segmenter_info_path.read_text(encoding="utf-8"))
            segmenter_lock = json.loads(segmenter_lock_path.read_text(encoding="utf-8"))
            outer_history = history_summary(run / "segmenter" / "history.csv")
            oof_histories = [
                history_summary(
                    run / "crossfit" / f"fold_{fold}" / "segmenter" / "history.csv"
                )
                for fold in range(int(cfg["cross_fitting"]["folds"]))
            ]
            checkpoint = resolve_project_path(
                segmenter_lock["checkpoint"], must_exist=True
            )
            if sha256_file(checkpoint) != segmenter_lock["checkpoint_sha256"]:
                raise RuntimeError("Segmenter checkpoint changed before compute summary")
            oof_seconds_values = [
                item["seconds"] for item in oof_histories if item["seconds"] is not None
            ]
            strategies = (cfg["classifier"]["primary"], cfg["classifier"]["secondary"])
            for strategy_spec in strategies:
                strategy = str(strategy_spec["name"])
                classifier_dir = run / "classifiers" / strategy
                classifier_history = history_summary(classifier_dir / "history.csv")
                classifier_info_path = classifier_dir / "model_info.json"
                classifier_info = (
                    json.loads(classifier_info_path.read_text(encoding="utf-8"))
                    if classifier_info_path.is_file() else {}
                )
                training_result_path = classifier_dir / "training_result.json"
                training_result = (
                    json.loads(training_result_path.read_text(encoding="utf-8"))
                    if training_result_path.is_file() else {}
                )
                classifier_checkpoint: Path | None = None
                if training_result.get("checkpoint"):
                    classifier_checkpoint = resolve_project_path(
                        training_result["checkpoint"], must_exist=True
                    )
                    if sha256_file(classifier_checkpoint) != training_result.get(
                        "checkpoint_sha256"
                    ):
                        raise RuntimeError(
                            "Classifier checkpoint changed before compute summary"
                        )
                rows.append({
                    "model": model_name,
                    "seed": int(seed),
                    "classifier_strategy": strategy,
                    "classifier_role": strategy_spec.get("role"),
                    "classifier_estimand_id": strategy_spec.get("estimand_id"),
                    "is_primary_estimand": strategy == cfg["classifier"]["primary"]["name"],
                    "segmenter_registered_parameters": segmenter_info.get(
                        "registered_parameters", segmenter_info.get("parameters")
                    ),
                    "segmenter_trainable_updateable_parameters": segmenter_info.get(
                        "trainable_updateable_parameters",
                        segmenter_info.get("trainable_parameters"),
                    ),
                    "segmenter_frozen_parameters": segmenter_info.get(
                        "frozen_parameters"
                    ),
                    "segmenter_explicitly_disabled_legacy_parameters": segmenter_info.get(
                        "explicitly_disabled_legacy_parameters",
                        segmenter_info.get("legacy_classification_parameters_frozen"),
                    ),
                    "classifier_registered_parameters": classifier_info.get(
                        "registered_parameters", classifier_info.get("parameters")
                    ),
                    "classifier_trainable_updateable_parameters": classifier_info.get(
                        "trainable_updateable_parameters",
                        classifier_info.get("trainable_parameters"),
                    ),
                    "classifier_frozen_parameters": classifier_info.get(
                        "frozen_parameters"
                    ),
                    "classifier_explicitly_disabled_parameters": classifier_info.get(
                        "explicitly_disabled_parameters",
                        classifier_info.get("explicitly_disabled_legacy_parameters"),
                    ),
                    "outer_segmenter_epochs": outer_history["epochs_recorded"],
                    "outer_segmenter_seconds": outer_history["seconds"],
                    "oof_segmenter_fit_count": len(oof_histories),
                    "oof_segmenter_epochs_total": int(
                        sum(int(item["epochs_recorded"]) for item in oof_histories)
                    ),
                    "oof_segmenter_seconds_total": (
                        float(sum(oof_seconds_values))
                        if len(oof_seconds_values) == len(oof_histories)
                        else None
                    ),
                    "classifier_epochs": classifier_history["epochs_recorded"],
                    "classifier_seconds": classifier_history["seconds"],
                    "classifier_checkpoint_bytes": (
                        int(classifier_checkpoint.stat().st_size)
                        if classifier_checkpoint is not None else None
                    ),
                    "classifier_checkpoint_sha256": (
                        training_result.get("checkpoint_sha256")
                        if classifier_checkpoint is not None else None
                    ),
                    "segmenter_checkpoint_bytes": int(checkpoint.stat().st_size),
                    "segmenter_checkpoint_sha256": segmenter_lock[
                        "checkpoint_sha256"
                    ],
                    "segmentation_initialization": _json_text(
                        cfg["segmentation"]["initialization"][model_name]
                    ),
                    "outer_segmenter_rng_seed": int(seed)
                    + int(cfg["training"]["seed_offsets"]["segmentation"]),
                    "classifier_rng_seed": (
                        int(seed)
                        + int(cfg["training"]["seed_offsets"]["classifier"])
                        + int(strategy_spec.get("seed_offset", 0))
                    ),
                    "oof_fit_seed_rule": "outer_seed + (fold+1)*100003, then segmentation offset inside trainer",
                    "segmenter_reported_pretraining": _json_text(
                        segmenter_info.get(
                            "pretraining_info",
                            segmenter_info.get("pretrained_provenance", {}),
                        )
                    ),
                    "classifier_reported_pretraining": _json_text(
                        classifier_info.get("pretrained_provenance", {})
                    ),
                    "classifier_declared_initialization": _json_text(
                        cfg["classifier"]["primary"]["initialization_by_segmenter"][model_name]
                        if strategy == cfg["classifier"]["primary"]["name"]
                        else cfg["classifier"]["secondary"]["pretrained_weights"]
                    ),
                    "flops_status": "not_reported",
                    "flops_rationale": FLOPS_RATIONALE,
                    "timing_scope": "recorded_training_epoch_wall_time; excludes data preparation and reporting",
                })
    preflight_reports: dict[str, Any] = {}
    for model_name in cfg["models"]:
        path = root / "verification" / f"{model_name}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing preflight environment report: {path}")
        preflight_reports[model_name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "report": json.loads(path.read_text(encoding="utf-8")),
        }
    details = {
        "schema": 1,
        "environment": environment_inventory(),
        "segmentation_initialization": cfg["segmentation"]["initialization"],
        "classifier_primary": cfg["classifier"]["primary"],
        "classifier_secondary": cfg["classifier"]["secondary"],
        "classifier_strategy_order": cfg["classifier"]["strategy_order"],
        "preflight_reports": preflight_reports,
        "flops_status": "not_reported",
        "flops_rationale": FLOPS_RATIONALE,
        "latency_status": "not_measured_by_training_run",
        "latency_note": (
            "A separate fixed-hardware batch-1 benchmark is required; training "
            "epoch wall time is not presented as clinical end-to-end latency."
        ),
        "overlapping_seed_note": (
            "Five split seeds reuse overlapping patients and are not independent cohorts."
        ),
    }
    return pd.DataFrame(rows), details


__all__ = [
    "FLOPS_RATIONALE",
    "collect_compute_and_provenance",
    "environment_inventory",
    "history_summary",
    "parameter_inventory",
]
