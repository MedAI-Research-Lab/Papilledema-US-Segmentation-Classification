"""Auditable execution engine for the locked three-class predicted-ROI study.

The binary study is an immutable upstream segmentation/ROI generator.  This
module trains only new three-logit diagnostic heads.  Development imports are
limited to patient-OOF training ROIs and validation ROIs; source test ROIs
cannot be opened until every model/seed/strategy validation lock exists.
"""

from __future__ import annotations

import contextlib
import hashlib
import itertools
import json
import math
import os
import platform
import random
import shutil
import sys
import time
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# CuBLAS requires this setting before the first CUDA context is created for
# deterministic GEMM on CUDA >= 10.2.  The launcher sets it as well; this
# in-process default keeps direct CLI invocations under the same contract.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import metrics
from .config import (
    CLASSIFIER_STRATEGIES,
    CLASS_NAMES,
    CLASS_ORDER,
    EXPECTED_MODELS,
    EXPECTED_SEEDS,
    config_without_runtime,
    resolve_project_path,
    sha256_file,
    upstream_artifact_paths,
    verify_frozen_source_anchors,
    verify_split_sources,
)
from .data import (
    IMAGE_NET_MEAN,
    FrozenUpstreamArtifacts,
    ThreeClassROICacheDataset,
    audit_frozen_upstream_artifacts,
    discover_frozen_upstream_artifacts,
    strict_boolean_mask,
    threeclass_classifier_loss_weights,
)
from .models import build_roi_classifier, build_threeclass_roi_classifier
from .publication_exports import (
    PUBLICATION_TABLE_NAMES,
    generate_publication_outputs,
    localization_vectors,
    segmentation_only_audit,
    validate_publication_outputs,
    write_publication_contracts,
)
from .publication_figures import generate_q1_figures, validate_q1_figures
from .protocol import (
    ProtocolGateError,
    assert_all_upstream_imports,
    assert_all_validation_locks,
    assert_test_access_open,
    deferred_test_import_receipt_path,
    open_test_access,
    output_root,
    read_json,
    save_json_atomic,
    unit_root,
    upstream_import_receipt_path,
    utc_now,
    validation_lock_path,
    verify_deferred_test_import_receipt,
    verify_upstream_import_receipt,
    verify_validation_lock,
    write_deferred_test_import_receipt,
    write_upstream_import_receipt,
    write_validation_lock,
)


PROBABILITY_COLUMNS = ("probability_0", "probability_1", "probability_2")
LOGIT_COLUMNS = ("logit_0", "logit_1", "logit_2")
IDENTITY_COLUMNS = ("patient_id", "case_id", "side", "frame_id")

DEFAULT_TRAINING = {
    "learning_rate": 1.0e-4,
    "weight_decay": 1.0e-4,
    "batch_size": 16,
    "gradient_accumulation": 1,
    "gradient_clip": 1.0,
    "num_workers": 0,
    "amp": True,
    "amp_dtype": "bfloat16",
    "augmentation": {
        "rotation_degrees": 7.0,
        "translation_fraction": 0.03,
        "scale": [0.95, 1.05],
        "brightness": [0.9, 1.1],
    },
    "seed_offsets": {
        "classifier": 40_000,
        "augmentation": 20_000,
        "bootstrap": 50_000,
        "standardized_resnet18": 500_009,
    },
}

DEFAULT_MODEL_INITIALIZATION: dict[str, dict[str, Any]] = {
    "yolo26": {
        "pretrained": True,
        "checkpoint": "binary_study/models/vendor/yolo26_sources/weights/yolo26s-seg.pt",
        "sha256": "3da1d83e31caec96f9300eb4064f4f62882c133c7c264d63dfe61a7c197837a4",
        "source": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-seg.pt",
    },
    "vit_method2": {
        "pretrained": False,
        "checkpoint": None,
        "sha256": None,
        "source": "random_initialization_no_compatible_locked_checkpoint",
    },
    "emcad": {
        "pretrained": True,
        "checkpoint": "binary_study/weights/emcad_joint/pvt_v2_b0.pth",
        "sha256": "fbb931ad59ab4d64e3f4370991326803e3af882fcaa877b81550d368d1fbab1f",
        "source": "https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b0.pth",
    },
    "sam2_unet": {
        "pretrained": True,
        "checkpoint": "binary_study/weights/sam2_unet/sam2_hiera_tiny.pt",
        "sha256": "65b50056e05bcb13694174f51bb6da89c894b57b75ccdf0ba6352c597c5d1125",
        "source": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt",
    },
}

DEFAULT_RESNET_INITIALIZATION = {
    "pretrained": True,
    "checkpoint": "predicted_roi_study/weights/resnet18-f37072fd.pth",
    "sha256": "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
    "source": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
}


def _clean(value: Any) -> Any:
    return metrics.clean_json(value)


def _save_json(path: str | Path, value: Any) -> Path:
    return save_json_atomic(path, _clean(value))


def _save_csv(path: str | Path, table: pd.DataFrame) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    table.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(destination)
    return destination


def _torch_save(path: str | Path, value: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(destination)
    return destination


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        _clean(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _code_sha256() -> str:
    package = Path(__file__).resolve().parent
    records = []
    for path in sorted(package.glob("*.py"), key=lambda item: item.name):
        records.append((path.name, sha256_file(path)))
    return _canonical_sha256(records)


def _installed_version(distribution: str) -> str | None:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return None


def _runtime_environment() -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    gpu: dict[str, Any] | None = None
    if cuda_available:
        properties = torch.cuda.get_device_properties(0)
        gpu = {
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "compute_capability": [int(properties.major), int(properties.minor)],
            "device_count": int(torch.cuda.device_count()),
        }
    return {
        "python": {
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "executable": str(Path(sys.executable).resolve()),
        },
        "operating_system": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": {
            name: _installed_version(name)
            for name in (
                "torch",
                "torchvision",
                "numpy",
                "pandas",
                "scikit-learn",
                "scipy",
                "matplotlib",
                "ultralytics",
                "timm",
            )
        },
        "accelerator": {
            "cuda_available": cuda_available,
            "torch_cuda_runtime": torch.version.cuda,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cudnn_version": (
                int(torch.backends.cudnn.version())
                if torch.backends.cudnn.is_available()
                else None
            ),
            "gpu": gpu,
        },
    }


def _write_progress(cfg: Mapping[str, Any], **fields: Any) -> Path:
    payload = {
        "study_id": cfg["study_id"],
        "process_id": os.getpid(),
        "updated_utc": utc_now(),
        **fields,
    }
    return _save_json(output_root(cfg) / "orchestration" / "progress.json", payload)


def _artifact_records(paths: Mapping[str, str | Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for role, raw in paths.items():
        path = Path(raw).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append(
            {
                "role": str(role),
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return records


def _generic_receipt_path(
    cfg: Mapping[str, Any],
    stage: str,
    *,
    model: str | None = None,
    seed: int | None = None,
    strategy: str | None = None,
) -> Path:
    destination = output_root(cfg) / "state" / "receipts" / stage
    if model is not None:
        destination /= model
    if seed is not None:
        destination /= f"seed_{seed}"
    if strategy is not None:
        destination /= strategy
    return destination.with_suffix(".json")


def _verify_generic_receipt(
    cfg: Mapping[str, Any],
    stage: str,
    *,
    model: str | None = None,
    seed: int | None = None,
    strategy: str | None = None,
) -> dict[str, Any]:
    """Verify a completed stage without rewriting any attested artifact."""

    destination = _generic_receipt_path(
        cfg, stage, model=model, seed=seed, strategy=strategy
    )
    if not destination.is_file():
        raise ProtocolGateError(f"Missing {stage} receipt: {destination}")
    payload = read_json(destination)
    expected = {
        "schema_version": 1,
        "stage": stage,
        "study_id": cfg["study_id"],
        "config_sha256": cfg["config_sha256"],
        "code_sha256": _code_sha256(),
        "model": model,
        "seed": seed,
        "classifier_strategy": strategy,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ProtocolGateError(f"{stage} receipt identity mismatch: {key}")
    records = payload.get("artifacts")
    if not isinstance(records, list) or not records:
        raise ProtocolGateError(f"{stage} receipt contains no artifacts")
    roles: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ProtocolGateError(f"Malformed artifact record in {stage} receipt")
        role = record.get("role")
        if not isinstance(role, str) or not role or role in roles:
            raise ProtocolGateError(f"Duplicate or malformed artifact role in {stage} receipt")
        roles.add(role)
        path = Path(str(record.get("path", ""))).resolve()
        if not path.is_file():
            raise ProtocolGateError(f"Attested artifact is missing: {path}")
        if path.stat().st_size != record.get("size_bytes"):
            raise ProtocolGateError(f"Attested artifact size changed: {path}")
        if sha256_file(path) != record.get("sha256"):
            raise ProtocolGateError(f"Attested artifact hash changed: {path}")
    return payload


def _generic_receipt(
    cfg: Mapping[str, Any],
    stage: str,
    *,
    model: str | None = None,
    seed: int | None = None,
    strategy: str | None = None,
    artifacts: Mapping[str, str | Path],
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    destination = _generic_receipt_path(
        cfg, stage, model=model, seed=seed, strategy=strategy
    )
    payload = {
        "schema_version": 1,
        "stage": stage,
        "study_id": cfg["study_id"],
        "config_sha256": cfg["config_sha256"],
        "code_sha256": _code_sha256(),
        "model": model,
        "seed": seed,
        "classifier_strategy": strategy,
        "artifacts": _artifact_records(artifacts),
        "metadata": dict(metadata or {}),
        "completed_utc": utc_now(),
    }
    if destination.exists():
        previous = read_json(destination)
        left, right = dict(previous), dict(payload)
        left.pop("completed_utc", None)
        right.pop("completed_utc", None)
        if left != right:
            raise ProtocolGateError(f"Refusing to overwrite non-identical receipt: {destination}")
        return destination
    return _save_json(destination, payload)


def _final_provenance_payload(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Inventory every terminal receipt through audit without a hash cycle."""

    receipt_root = output_root(cfg) / "state" / "receipts"
    final_receipt = _generic_receipt_path(cfg, "final-provenance").resolve()
    records: list[dict[str, Any]] = []
    for path in sorted(receipt_root.rglob("*.json"), key=lambda item: item.as_posix()):
        if path.resolve() == final_receipt:
            continue
        payload = read_json(path)
        records.append(
            {
                "receipt": path.relative_to(receipt_root).as_posix(),
                "stage": payload.get("stage"),
                "model": payload.get("model"),
                "seed": payload.get("seed"),
                "classifier_strategy": payload.get("classifier_strategy"),
                "sha256": sha256_file(path),
                "size_bytes": int(path.stat().st_size),
            }
        )
    if not records or not any(record["stage"] == "audit" for record in records):
        raise ProtocolGateError(
            "Final provenance snapshot requires the completed audit receipt"
        )
    return {
        "schema_version": 1,
        "study_id": cfg["study_id"],
        "config_sha256": cfg["config_sha256"],
        "code_sha256": _code_sha256(),
        "scope": "all_output_receipts_through_final_audit",
        "attestation": (
            "The final-provenance receipt attests this snapshot and is excluded "
            "from its own inventory to avoid a circular hash dependency."
        ),
        "receipt_count": len(records),
        "receipts": records,
    }


def _write_or_verify_final_provenance(cfg: Mapping[str, Any]) -> Path:
    path = output_root(cfg) / "summary" / "final_provenance_snapshot.json"
    expected = _final_provenance_payload(cfg)
    if path.is_file():
        if read_json(path) != expected:
            raise ProtocolGateError("Final provenance snapshot no longer matches receipts")
    else:
        _save_json(path, expected)
    receipt = _generic_receipt(
        cfg,
        "final-provenance",
        artifacts={"final_provenance_snapshot": path},
        metadata={
            "includes_audit_receipt": True,
            "circular_hash_dependency": False,
        },
    )
    _verify_generic_receipt(cfg, "final-provenance")
    if not receipt.is_file():
        raise ProtocolGateError("Final provenance receipt was not persisted")
    return path


def _training_settings(cfg: Mapping[str, Any]) -> dict[str, Any]:
    settings = dict(DEFAULT_TRAINING)
    locked = cfg.get("training")
    if isinstance(locked, Mapping):
        for key, value in locked.items():
            settings[key] = value
    settings["maximum_epochs"] = int(cfg["classifier"]["maximum_epochs"])
    settings["patience"] = int(cfg["classifier"]["patience"])
    return settings


def _classifier_initialization(
    cfg: Mapping[str, Any], model: str, strategy: str
) -> dict[str, Any]:
    classifier = cfg["classifier"]
    if strategy == "model_specific":
        supplied = classifier.get("initialization_by_segmenter", {})
        spec = dict(supplied.get(model, DEFAULT_MODEL_INITIALIZATION[model]))
    elif strategy == "standardized_resnet18":
        spec = dict(classifier.get("standardized_initialization", DEFAULT_RESNET_INITIALIZATION))
    else:
        raise ProtocolGateError(f"Unknown classifier strategy: {strategy}")
    if bool(spec.get("pretrained")):
        checkpoint = resolve_project_path(spec["checkpoint"], must_exist=True)
        actual = sha256_file(checkpoint)
        if actual != spec.get("sha256"):
            raise ProtocolGateError(
                f"Classifier initialization SHA-256 mismatch for {model}/{strategy}"
            )
        spec["checkpoint"] = str(checkpoint)
        spec["observed_sha256"] = actual
    return spec


def _seed_everything(seed: int, *, warn_only: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
        # Flash and memory-efficient scaled-dot-product attention may select
        # non-deterministic CUDA kernels.  The math backend is slower but is
        # compatible with the publication run's strict reproducibility gate.
        cuda_backend = torch.backends.cuda
        if hasattr(cuda_backend, "enable_flash_sdp"):
            cuda_backend.enable_flash_sdp(False)
        if hasattr(cuda_backend, "enable_mem_efficient_sdp"):
            cuda_backend.enable_mem_efficient_sdp(False)
        if hasattr(cuda_backend, "enable_math_sdp"):
            cuda_backend.enable_math_sdp(True)
        if hasattr(cuda_backend, "enable_cudnn_sdp"):
            cuda_backend.enable_cudnn_sdp(False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=bool(warn_only))


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _device_required() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("The locked clinical run requires a CUDA GPU")
    return torch.device("cuda")


def _amp_context(settings: Mapping[str, Any], device: torch.device):
    if not bool(settings.get("amp", True)):
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if settings.get("amp_dtype") == "bfloat16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def _loader(
    dataset: ThreeClassROICacheDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=True,
        persistent_workers=bool(num_workers),
        generator=generator,
        drop_last=False,
    )


def _to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _build_classifier(
    cfg: Mapping[str, Any],
    model: str,
    strategy: str,
    *,
    pretrained: bool,
    lightweight: bool = False,
) -> torch.nn.Module:
    initialization = _classifier_initialization(cfg, model, strategy) if pretrained else {}
    if strategy == "model_specific":
        use_pretrained = bool(initialization.get("pretrained", False)) if pretrained else False
        return build_threeclass_roi_classifier(
            model,
            pretrained=use_pretrained,
            weights_path=initialization.get("checkpoint") if use_pretrained else None,
            expected_sha256=initialization.get("sha256") if use_pretrained else None,
            geometry_features=0,
            lightweight=lightweight,
            num_classes=3,
        )
    if strategy == "standardized_resnet18":
        use_pretrained = bool(initialization.get("pretrained", False)) if pretrained else False
        return build_roi_classifier(
            "resnet18",
            pretrained=use_pretrained,
            weights_path=initialization.get("checkpoint") if use_pretrained else None,
            expected_sha256=initialization.get("sha256") if use_pretrained else None,
            geometry_features=0,
            num_classes=3,
            dropout=0.2,
        )
    raise ProtocolGateError(f"Unknown classifier strategy: {strategy}")


def _classifier_forward(
    classifier: torch.nn.Module,
    batch: Mapping[str, Any],
    *,
    strategy: str,
) -> torch.Tensor:
    images = batch["image"]
    hard_mask = batch["roi_mask"].bool()
    if images.ndim != 4 or tuple(hard_mask.shape) != (
        len(images),
        images.shape[-2],
        images.shape[-1],
    ):
        raise ProtocolGateError("Classifier image and hard-mask shapes disagree")
    neutral = IMAGE_NET_MEAN.to(device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    strict_images = torch.where(hard_mask.unsqueeze(1), images, neutral)
    if strategy == "model_specific":
        logits = classifier(strict_images, geometry=None, hard_mask=hard_mask)
    elif strategy == "standardized_resnet18":
        logits = classifier(strict_images, None)
    else:
        raise ProtocolGateError(f"Unknown classifier strategy: {strategy}")
    if tuple(logits.shape) != (len(images), 3) or not bool(torch.isfinite(logits).all()):
        raise RuntimeError("Three-class classifier must return finite Bx3 logits")
    return logits.float()


def _strategy_directory(
    cfg: Mapping[str, Any], model: str, seed: int, strategy: str
) -> Path:
    return unit_root(cfg, model, seed) / "classifiers" / strategy


def _training_seed(
    cfg: Mapping[str, Any], model: str, seed: int, strategy: str
) -> int:
    settings = _training_settings(cfg)
    offsets = settings["seed_offsets"]
    value = int(seed) + int(offsets["classifier"])
    if strategy == "standardized_resnet18":
        value += int(offsets["standardized_resnet18"])
    value += 1009 * tuple(cfg["models"]).index(model)
    return value


def _read_roi_index(path: str | Path) -> pd.DataFrame:
    required = {
        *IDENTITY_COLUMNS,
        "label_3class",
        "frame_identity_sha256",
        "roi_valid",
        "abstention_reason",
        "cache_path",
        "cache_sha256",
    }
    header = pd.read_csv(path, encoding="utf-8-sig", nrows=0)
    missing = required - set(header)
    if missing:
        raise ValueError(f"ROI index is missing columns: {sorted(missing)}")
    # Deliberately project only the three-class ROI contract.  The completed
    # binary experiment's logits, probabilities, thresholds and predictions
    # are prohibited inputs to this extension.
    table = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={"frame_id": str},
        usecols=sorted(required),
    )
    table["roi_valid"] = strict_boolean_mask(table["roi_valid"], name="roi_valid")
    labels = pd.to_numeric(table["label_3class"], errors="raise").astype(int)
    if not labels.isin(CLASS_ORDER).all():
        raise ValueError("label_3class must be 0, 1, or 2")
    table["label_3class"] = labels
    if table.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError("ROI index contains duplicate frame identities")
    return table


def _audit_locked_manifest(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed on the immutable 91-patient manifest contract."""

    specification = cfg["dataset"]
    path = resolve_project_path(specification["manifest"], must_exist=True)
    observed_sha256 = sha256_file(path)
    if observed_sha256 != specification["manifest_sha256"]:
        raise ProtocolGateError("Locked dataset manifest SHA-256 mismatch")
    required = {
        "patient_id",
        "case_id",
        "side",
        "frame_id",
        "label_3class",
        "output_image",
        "output_mask",
        "output_image_sha256",
        "output_mask_sha256",
    }
    header = pd.read_csv(path, encoding="utf-8-sig", nrows=0)
    missing = required - set(header)
    if missing:
        raise ProtocolGateError(
            f"Locked dataset manifest is missing columns: {sorted(missing)}"
        )
    manifest = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={"frame_id": str},
        usecols=sorted(required),
    )
    if len(manifest) != int(specification["frames"]):
        raise ProtocolGateError("Locked dataset manifest frame count mismatch")
    identity = ["patient_id", "case_id", "side", "frame_id"]
    if manifest[identity].isna().any().any() or manifest.duplicated(identity).any():
        raise ProtocolGateError("Locked manifest contains missing or duplicate frame identities")
    labels = pd.to_numeric(manifest["label_3class"], errors="raise").astype(int)
    if not labels.isin(CLASS_ORDER).all():
        raise ProtocolGateError("Locked manifest contains an invalid three-class label")
    manifest["label_3class"] = labels
    if manifest.groupby("patient_id")["label_3class"].nunique().max() != 1:
        raise ProtocolGateError("Locked manifest has inconsistent patient labels")
    if manifest.groupby("case_id")["patient_id"].nunique().max() != 1:
        raise ProtocolGateError("Locked manifest maps an eye to multiple patients")
    if manifest.groupby("case_id")["side"].nunique().max() != 1:
        raise ProtocolGateError("Locked manifest maps an eye to multiple sides")
    eye_sizes = manifest.groupby(["patient_id", "case_id", "side"], sort=False).agg(
        rows=("frame_id", "size"), unique_frames=("frame_id", "nunique")
    )
    expected_frames_per_eye = int(specification["frames_per_eye"])
    if not (
        eye_sizes["rows"].eq(expected_frames_per_eye)
        & eye_sizes["unique_frames"].eq(expected_frames_per_eye)
    ).all():
        raise ProtocolGateError("Each locked eye must contain exactly seven unique frames")
    patient_eye_counts = (
        manifest[["patient_id", "case_id"]]
        .drop_duplicates()
        .groupby("patient_id", sort=False)
        .size()
    )
    if not patient_eye_counts.eq(int(specification["eyes_per_patient"])).all():
        raise ProtocolGateError("Each locked patient must contain exactly two eyes")
    patients = manifest[["patient_id", "label_3class"]].drop_duplicates()
    observed_patient_counts = {
        str(int(label)): int(count)
        for label, count in patients["label_3class"].value_counts().sort_index().items()
    }
    expected_patient_counts = {
        str(key): int(value)
        for key, value in specification["patient_counts_by_class"].items()
    }
    counts = {
        "patients": int(manifest["patient_id"].nunique()),
        "eyes": int(manifest["case_id"].nunique()),
        "frames": int(len(manifest)),
    }
    for key in ("patients", "eyes", "frames"):
        if counts[key] != int(specification[key]):
            raise ProtocolGateError(f"Locked manifest {key} count mismatch")
    if observed_patient_counts != expected_patient_counts:
        raise ProtocolGateError("Locked manifest patient class counts mismatch")
    for column in (
        "output_image_sha256",
        "output_mask_sha256",
    ):
        if not manifest[column].astype(str).str.fullmatch(r"[0-9a-fA-F]{64}").all():
            raise ProtocolGateError(f"Locked manifest contains malformed {column} values")
    return {
        "path": str(path),
        "sha256": observed_sha256,
        **counts,
        "frames_per_eye": expected_frames_per_eye,
        "eyes_per_patient": int(specification["eyes_per_patient"]),
        "patient_counts_by_class": observed_patient_counts,
        "unique_frame_identity_count": int(
            manifest[identity].drop_duplicates().shape[0]
        ),
        "status": "passed",
    }


def _split_membership(cfg: Mapping[str, Any], seed: int) -> pd.DataFrame:
    path = resolve_project_path(
        cfg["split_policy"]["sources"][str(seed)]["path"], must_exist=True
    )
    table = pd.read_csv(path, encoding="utf-8-sig")
    required = {"patient_id", "label_3class", "split", "seed"}
    if not required <= set(table):
        raise ValueError(f"Locked split file is missing {sorted(required - set(table))}")
    if set(table["split"]) != {"train", "validation", "test"}:
        raise ValueError("Locked split file has unexpected partition names")
    if table["patient_id"].duplicated().any() or set(table["seed"].astype(int)) != {int(seed)}:
        raise ValueError("Locked split membership is malformed")
    return table


def _verify_partition_rows(
    cfg: Mapping[str, Any], table: pd.DataFrame, seed: int, partition: str
) -> None:
    split = _split_membership(cfg, seed)
    expected = split.loc[split["split"] == partition, ["patient_id", "label_3class"]]
    observed = table[["patient_id", "label_3class"]].drop_duplicates()
    if observed["patient_id"].duplicated().any():
        raise ValueError("ROI partition has inconsistent patient labels")
    merged = expected.merge(
        observed, on=["patient_id", "label_3class"], how="outer", indicator=True
    )
    if not merged["_merge"].eq("both").all():
        raise ProtocolGateError(
            f"Frozen ROI {partition} membership disagrees with locked seed {seed} split"
        )


def _training_selection(
    table: pd.DataFrame, *, minimum_valid_frames: int = 4, frames_per_eye: int = 7
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    eligible_keys: set[tuple[str, str, str]] = set()
    for identity, group in table.groupby(["patient_id", "case_id", "side"], sort=True):
        if len(group) != frames_per_eye or group["frame_id"].nunique() != frames_per_eye:
            raise ValueError(f"Eye {identity!r} does not contain exactly seven frames")
        if group["label_3class"].nunique() != 1:
            raise ValueError(f"Eye {identity!r} has inconsistent labels")
        valid_count = int(group["roi_valid"].sum())
        eligible = valid_count >= int(minimum_valid_frames)
        if eligible:
            eligible_keys.add(tuple(map(str, identity)))
        rows.append(
            {
                "patient_id": identity[0],
                "case_id": identity[1],
                "side": identity[2],
                "label_3class": int(group["label_3class"].iloc[0]),
                "n_frames": int(len(group)),
                "n_valid_frames": valid_count,
                "minimum_valid_frames": int(minimum_valid_frames),
                "training_eligible": bool(eligible),
                "optimization_frames": valid_count if eligible else 0,
                "exclusion_reason": "" if eligible else "insufficient_valid_frames",
            }
        )
    eye_key = table[["patient_id", "case_id", "side"]].astype(str).apply(tuple, axis=1)
    selected = table.loc[
        table["roi_valid"] & eye_key.isin(eligible_keys)
    ].sort_values(list(IDENTITY_COLUMNS), kind="stable").reset_index(drop=True)
    ledger = pd.DataFrame(rows).sort_values(
        ["patient_id", "case_id", "side"], kind="stable"
    ).reset_index(drop=True)
    if len(selected) != int(ledger["optimization_frames"].sum()):
        raise RuntimeError("Training selection ledger does not match selected frames")
    patient_classes = selected[["patient_id", "label_3class"]].drop_duplicates()
    counts = patient_classes["label_3class"].value_counts().reindex(CLASS_ORDER, fill_value=0)
    if (counts == 0).any():
        raise ProtocolGateError("All three classes require ROI-bearing training patients")
    return selected, ledger


def prepare(cfg: Mapping[str, Any]) -> dict[str, Any]:
    root = output_root(cfg)
    existing = _generic_receipt_path(cfg, "prepare")
    if existing.is_file():
        _verify_generic_receipt(cfg, "prepare")
        return {
            "status": "already_complete",
            "receipt": str(existing),
            "output": str(root),
        }
    root.mkdir(parents=True, exist_ok=True)
    for name in ("provenance", "orchestration", "state/receipts", "runs", "summary", "figures", "tables"):
        (root / name).mkdir(parents=True, exist_ok=True)
    anchors = verify_frozen_source_anchors(cfg)
    splits = verify_split_sources(cfg)
    manifest_audit = _audit_locked_manifest(cfg)
    config_snapshot = root / "provenance" / "config_threeclass_roi.snapshot.json"
    _save_json(config_snapshot, config_without_runtime(cfg))
    source_protocol = Path(__file__).resolve().parent / "PROTOCOL.md"
    protocol_snapshot = root / "provenance" / "PROTOCOL.snapshot.md"
    if source_protocol.is_file():
        shutil.copyfile(source_protocol, protocol_snapshot)
    else:
        protocol_snapshot.write_text(
            "Three-class protocol text was not yet available.\n", encoding="utf-8"
        )
    code_inventory = {
        path.name: sha256_file(path)
        for path in sorted(Path(__file__).resolve().parent.glob("*.py"))
    }
    code_path = _save_json(root / "provenance" / "code_inventory.json", code_inventory)
    manifest_audit_path = _save_json(
        root / "provenance" / "locked_manifest_audit.json", manifest_audit
    )
    runtime_environment_path = _save_json(
        root / "provenance" / "runtime_environment.json", _runtime_environment()
    )
    design = {
        "study_id": cfg["study_id"],
        "analysis_status": cfg["analysis_status"],
        "classes": cfg["classes"],
        "models": cfg["models"],
        "seeds": cfg["split_seeds"],
        "classifier_fits": cfg["classifier"]["expected_fits"],
        "upstream_segmentation_retrained": False,
        "prior_test_use_disclosure": cfg["prior_test_use"]["required_disclosure"],
        "confirmatory_claim_allowed": False,
        "test_access_open": False,
        "created_utc": utc_now(),
    }
    design_path = _save_json(root / "provenance" / "study_design.json", design)
    publication_contracts = write_publication_contracts(cfg)
    receipt = _generic_receipt(
        cfg,
        "prepare",
        artifacts={
            "config_snapshot": config_snapshot,
            "protocol_snapshot": protocol_snapshot,
            "code_inventory": code_path,
            "locked_manifest_audit": manifest_audit_path,
            "runtime_environment": runtime_environment_path,
            "study_design": design_path,
            **publication_contracts,
        },
        metadata={
            "source_anchors": anchors,
            "split_hashes": splits,
            "manifest_audit": manifest_audit,
            "test_data_read": False,
        },
    )
    _write_progress(cfg, phase="prepared", completed_units=0, target_units=40)
    return {"status": "complete", "receipt": str(receipt), "output": str(root)}


def import_upstream(cfg: Mapping[str, Any], model: str, seed: int) -> dict[str, Any]:
    _verify_generic_receipt(cfg, "prepare")
    completed = _generic_receipt_path(
        cfg, "audit-upstream-development", model=model, seed=seed
    )
    if completed.is_file():
        _verify_generic_receipt(
            cfg, "audit-upstream-development", model=model, seed=seed
        )
        verify_upstream_import_receipt(cfg, model, seed)
        return {
            "status": "already_complete",
            "receipt": str(upstream_import_receipt_path(cfg, model, seed)),
            "audit": str(unit_root(cfg, model, seed) / "upstream" / "development_import_audit.json"),
        }
    paths = upstream_artifact_paths(cfg, model, seed, phase="development")
    artifact = discover_frozen_upstream_artifacts(
        cfg["frozen_upstream"]["source_output"],
        families=[model],
        seeds=[seed],
        require_test=False,
    )[0]
    audit = audit_frozen_upstream_artifacts(
        artifact,
        verify_referenced_files=True,
        verify_payload_identity=True,
    )
    train = _read_roi_index(artifact.training_oof_index)
    validation = _read_roi_index(artifact.validation_index)
    _verify_partition_rows(cfg, train, seed, "train")
    _verify_partition_rows(cfg, validation, seed, "validation")
    if set(train["patient_id"]) & set(validation["patient_id"]):
        raise ProtocolGateError("Patient leakage between OOF training and validation ROI imports")
    audit_path = _save_json(
        unit_root(cfg, model, seed) / "upstream" / "development_import_audit.json",
        audit,
    )
    receipt = write_upstream_import_receipt(cfg, model, seed, artifact_paths=paths)
    _generic_receipt(
        cfg,
        "audit-upstream-development",
        model=model,
        seed=seed,
        artifacts={"audit": audit_path, "protocol_import_receipt": receipt},
        metadata={
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_artifacts_read": False,
        },
    )
    _write_progress(cfg, phase="upstream_imported", model=model, seed=seed)
    return {"status": "complete", "receipt": str(receipt), "audit": str(audit_path)}


def _outside_roi_invariance(
    classifier: torch.nn.Module,
    sample: Mapping[str, torch.Tensor],
    *,
    strategy: str,
    device: torch.device,
) -> dict[str, Any]:
    classifier.eval()
    image = sample["image"].unsqueeze(0).to(device)
    mask = sample["roi_mask"].unsqueeze(0).to(device)
    perturbed = image.clone()
    noise = torch.rand_like(perturbed)
    perturbed = torch.where(mask.unsqueeze(1), perturbed, noise)
    batch_a = {"image": image, "roi_mask": mask}
    batch_b = {"image": perturbed, "roi_mask": mask}
    with torch.inference_mode():
        left = _classifier_forward(classifier, batch_a, strategy=strategy)
        right = _classifier_forward(classifier, batch_b, strategy=strategy)
    maximum = float((left - right).abs().max().cpu())
    if maximum > 1e-6:
        raise ProtocolGateError(
            f"Outside-ROI invariance failed for {strategy}: max difference {maximum}"
        )
    return {"passed": True, "maximum_absolute_logit_difference": maximum}


def preflight(cfg: Mapping[str, Any], model: str) -> dict[str, Any]:
    _verify_generic_receipt(cfg, "prepare")
    completed = _generic_receipt_path(cfg, "preflight", model=model)
    if completed.is_file():
        _verify_generic_receipt(cfg, "preflight", model=model)
        return {
            "status": "already_complete",
            "receipt": str(completed),
            "report": str(output_root(cfg) / "preflight" / f"{model}.json"),
        }
    seed = int(cfg["split_seeds"][0])
    verify_upstream_import_receipt(cfg, model, seed)
    path = upstream_artifact_paths(cfg, model, seed, phase="development")[
        "validation_roi_index"
    ]
    table = _read_roi_index(path)
    dataset = ThreeClassROICacheDataset(table)
    if len(dataset) < 1:
        raise ProtocolGateError("Preflight requires at least one valid validation ROI")
    settings = _training_settings(cfg)
    # Establish the deterministic CUDA contract before device discovery can
    # initialize a CUDA context.  Each strategy is reseeded again below.
    _seed_everything(
        _training_seed(cfg, model, seed, CLASSIFIER_STRATEGIES[0]),
        warn_only=bool(settings["deterministic_algorithms_warn_only"]),
    )
    device = _device_required()
    results: dict[str, Any] = {
        "model": model,
        "seed": seed,
        "cuda_device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "torch_cuda_runtime": torch.version.cuda,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_version": (
            int(torch.backends.cudnn.version())
            if torch.backends.cudnn.is_available()
            else None
        ),
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_algorithms_warn_only": bool(
            settings["deterministic_algorithms_warn_only"]
        ),
        "strategies": {},
        "test_artifacts_read": False,
    }
    for strategy in CLASSIFIER_STRATEGIES:
        preflight_seed = _training_seed(cfg, model, seed, strategy)
        _seed_everything(
            preflight_seed,
            warn_only=bool(settings["deterministic_algorithms_warn_only"]),
        )
        torch.cuda.reset_peak_memory_stats(device)
        classifier = _build_classifier(
            cfg, model, strategy, pretrained=True, lightweight=False
        ).to(device)
        sample = dataset[0]
        invariance = _outside_roi_invariance(
            classifier, sample, strategy=strategy, device=device
        )
        classifier.train()
        configured_batch_size = int(settings["batch_size"])
        if len(dataset) < configured_batch_size:
            raise ProtocolGateError(
                f"Preflight requires at least the configured batch size "
                f"({configured_batch_size}) for {model}/{strategy}"
            )
        loader = _loader(
            dataset,
            batch_size=configured_batch_size,
            shuffle=False,
            seed=preflight_seed,
            num_workers=0,
        )
        batch = _to_device(next(iter(loader)), device)
        with _amp_context(settings, device):
            logits = _classifier_forward(classifier, batch, strategy=strategy)
        loss = F.cross_entropy(logits, batch["label"])
        loss.backward()
        trainable = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
        registered = sum(p.numel() for p in classifier.parameters())
        nonzero_gradients = sum(
            int(p.grad is not None and bool(torch.isfinite(p.grad).all()))
            for p in classifier.parameters()
            if p.requires_grad
        )
        if nonzero_gradients == 0:
            raise ProtocolGateError(f"No finite training gradient for {model}/{strategy}")
        results["strategies"][strategy] = {
            "output_shape": list(logits.shape),
            "configured_batch_size_tested": configured_batch_size,
            "deterministic_preflight_seed": preflight_seed,
            "loss": float(loss.detach().cpu()),
            "registered_parameters": int(registered),
            "trainable_parameters": int(trainable),
            "finite_gradient_parameter_tensors": int(nonzero_gradients),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "outside_roi_invariance": invariance,
            "initialization": _classifier_initialization(cfg, model, strategy),
        }
        del classifier, logits, loss
        torch.cuda.empty_cache()
    report = _save_json(output_root(cfg) / "preflight" / f"{model}.json", results)
    receipt = _generic_receipt(
        cfg,
        "preflight",
        model=model,
        artifacts={"report": report},
        metadata={"test_artifacts_read": False},
    )
    _write_progress(cfg, phase="preflight_complete", model=model, seed=seed)
    return {"status": "complete", "receipt": str(receipt), "report": str(report)}


def _infer_frames(
    cfg: Mapping[str, Any],
    classifier: torch.nn.Module,
    index: pd.DataFrame,
    *,
    model: str,
    seed: int,
    strategy: str,
    split: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    settings = _training_settings(cfg)
    device = next(classifier.parameters()).device
    dataset = ThreeClassROICacheDataset(index)
    loader = _loader(
        dataset,
        batch_size=int(settings["batch_size"]),
        shuffle=False,
        seed=_training_seed(cfg, model, seed, strategy),
        num_workers=int(settings["num_workers"]),
    )
    classifier.eval()
    logits_parts: list[np.ndarray] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            batch = _to_device(batch, device)
            with _amp_context(settings, device):
                logits = _classifier_forward(classifier, batch, strategy=strategy)
            logits_parts.append(logits.float().cpu().numpy())
    elapsed = time.perf_counter() - started
    raw_logits = (
        np.concatenate(logits_parts, axis=0)
        if logits_parts
        else np.empty((0, len(CLASS_ORDER)), dtype=np.float32)
    )
    raw_probability = torch.softmax(torch.from_numpy(raw_logits), dim=1).numpy()
    if len(raw_logits) != len(dataset.rows):
        raise RuntimeError("Classifier prediction count differs from valid ROI rows")
    predictions = dataset.rows[[*IDENTITY_COLUMNS, "frame_identity_sha256"]].copy()
    for offset, column in enumerate(LOGIT_COLUMNS):
        predictions[column] = raw_logits[:, offset]
    for offset, column in enumerate(PROBABILITY_COLUMNS):
        predictions[column] = raw_probability[:, offset]
    predictions["prediction"] = np.argmax(raw_probability, axis=1).astype(int)
    predictions["confidence"] = raw_probability.max(axis=1)
    output = index.merge(
        predictions,
        on=[*IDENTITY_COLUMNS, "frame_identity_sha256"],
        how="left",
        validate="one_to_one",
    )
    valid = output["roi_valid"].to_numpy(dtype=bool)
    if output.loc[valid, list(PROBABILITY_COLUMNS)].isna().any().any():
        raise RuntimeError("A valid ROI frame lacks classifier probabilities")
    if output.loc[~valid, list(PROBABILITY_COLUMNS)].notna().any().any():
        raise RuntimeError("An invalid ROI frame received classifier probabilities")
    output.loc[~valid, "prediction"] = metrics.ABSTAIN
    output["prediction"] = output["prediction"].astype(int)
    output["split"] = split
    output["model"] = model
    output["seed"] = int(seed)
    output["strategy"] = strategy
    timing = {
        "valid_frames": int(len(dataset)),
        "all_intended_frames": int(len(index)),
        "inference_seconds": float(elapsed),
        "milliseconds_per_valid_frame": (
            float(1000.0 * elapsed / len(dataset)) if len(dataset) else np.nan
        ),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    return output, timing


def _aggregate_frames(frames: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    eyes = metrics.aggregate_frames_to_eyes(
        frames,
        frames_per_eye=7,
        min_valid_frames=4,
        validity_column="roi_valid",
        probability_columns=PROBABILITY_COLUMNS,
        label_column="label_3class",
    )
    patients = metrics.aggregate_eyes_to_patients(
        eyes,
        probability_columns=PROBABILITY_COLUMNS,
        label_column="label_3class",
    )
    return eyes, patients


def _selection_monitor(patients: pd.DataFrame) -> dict[str, Any]:
    result = metrics.selective_multiclass_metrics(
        patients["label_3class"],
        patients.loc[:, PROBABILITY_COLUMNS].to_numpy(float),
        evaluable=patients["evaluable"].to_numpy(dtype=bool),
    )
    evaluable = patients.loc[patients["evaluable"]]
    all_classes = set(evaluable["label_3class"].astype(int)) == set(CLASS_ORDER)
    if not len(evaluable) or not all_classes:
        return {
            "status": "non_evaluable_missing_validation_class",
            "macro_nll": np.nan,
            "failure_aware_balanced_accuracy": result[
                "failure_aware_balanced_accuracy"
            ],
            "key": (-math.inf, -math.inf),
            "metrics": result,
        }
    macro_nll = float(result["conditional"]["macro"]["nll_true_class"])
    failure_ba = float(result["failure_aware_balanced_accuracy"])
    if not all(np.isfinite([macro_nll, failure_ba])):
        raise FloatingPointError("Validation monitor contains a non-finite value")
    return {
        "status": "evaluable",
        "macro_nll": macro_nll,
        "failure_aware_balanced_accuracy": failure_ba,
        # These are the only two criteria declared by the locked protocol.
        # Training uses strict tuple comparison, so an exact tie keeps the
        # already-selected (and therefore earlier) epoch.
        "key": (-macro_nll, failure_ba),
        "metrics": result,
    }


def _globally_normalized_weighted_loss(
    losses: torch.Tensor,
    weights: torch.Tensor,
    global_weight_mean: torch.Tensor,
) -> torch.Tensor:
    """Return an unbiased minibatch estimate of the locked global objective.

    ``threeclass_classifier_loss_weights`` fixes weights over the complete
    optimization set so that classes, then patients, then eyes have equal
    total mass.  Re-normalizing by the *minibatch* weight sum would change that
    target according to batch composition.  Dividing the minibatch mean by the
    fold-fixed global mean preserves the declared global weighted-risk
    objective while retaining ordinary stochastic minibatch updates.
    """

    if losses.ndim != 1 or weights.ndim != 1 or losses.shape != weights.shape:
        raise ValueError("Losses and weights must be equal-length vectors")
    if global_weight_mean.numel() != 1:
        raise ValueError("global_weight_mean must be scalar")
    return (losses * weights).mean() / global_weight_mean


def _checkpoint_fingerprint(
    cfg: Mapping[str, Any], model: str, seed: int, strategy: str,
    train_index: Path, validation_index: Path,
) -> dict[str, Any]:
    return {
        "study_id": cfg["study_id"],
        "config_sha256": cfg["config_sha256"],
        "code_sha256": _code_sha256(),
        "model": model,
        "seed": int(seed),
        "strategy": strategy,
        "train_index_sha256": sha256_file(train_index),
        "validation_index_sha256": sha256_file(validation_index),
        "classifier_outputs": 3,
    }


def train_classifier(
    cfg: Mapping[str, Any], model: str, seed: int, strategy: str
) -> dict[str, Any]:
    _verify_generic_receipt(cfg, "prepare")
    verify_upstream_import_receipt(cfg, model, seed)
    if strategy not in tuple(cfg["classifier"]["strategy_order"]):
        raise ProtocolGateError(f"Unknown strategy: {strategy}")
    training_receipt_path = _generic_receipt_path(
        cfg,
        "train-classifier",
        model=model,
        seed=seed,
        strategy=strategy,
    )
    if training_receipt_path.is_file():
        _verify_generic_receipt(
            cfg,
            "train-classifier",
            model=model,
            seed=seed,
            strategy=strategy,
        )
        return {"status": "already_complete", "receipt": str(training_receipt_path)}
    source = upstream_artifact_paths(cfg, model, seed, phase="development")
    train_index_path = source["train_oof_roi_index"]
    validation_index_path = source["validation_roi_index"]
    train_index = _read_roi_index(train_index_path)
    validation_index = _read_roi_index(validation_index_path)
    _verify_partition_rows(cfg, train_index, seed, "train")
    _verify_partition_rows(cfg, validation_index, seed, "validation")
    selected, ledger = _training_selection(train_index)
    work = _strategy_directory(cfg, model, seed, strategy)
    work.mkdir(parents=True, exist_ok=True)
    selected_index_path = _save_csv(
        unit_root(cfg, model, seed) / "classifiers" / "shared" / "training_optimization_index.csv",
        selected,
    )
    ledger_path = _save_csv(
        unit_root(cfg, model, seed) / "classifiers" / "shared" / "training_eye_eligibility.csv",
        ledger,
    )
    selected_path = work / "selected.pt"
    last_path = work / "last.pt"
    history_path = work / "history.csv"
    validation_frames_path = work / "validation_frames_raw.csv"
    validation_eyes_path = work / "validation_eyes_raw.csv"
    validation_patients_path = work / "validation_patients_raw.csv"
    model_info_path = work / "model_info.json"
    settings = _training_settings(cfg)
    training_seed = _training_seed(cfg, model, seed, strategy)
    _seed_everything(
        training_seed,
        warn_only=bool(settings["deterministic_algorithms_warn_only"]),
    )
    device = _device_required()
    torch.cuda.reset_peak_memory_stats(device)
    classifier = _build_classifier(cfg, model, strategy, pretrained=True).to(device)
    registered_parameters = int(sum(p.numel() for p in classifier.parameters()))
    trainable_parameters = int(
        sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    )
    initialization = _classifier_initialization(cfg, model, strategy)
    model_info = {
        "model": model,
        "seed": int(seed),
        "classifier_strategy": strategy,
        "classifier_outputs": 3,
        "class_order": list(CLASS_ORDER),
        "class_names": CLASS_NAMES,
        "registered_parameters": registered_parameters,
        "trainable_parameters": trainable_parameters,
        "frozen_parameters": registered_parameters - trainable_parameters,
        "parameter_info": (
            classifier.parameter_info() if hasattr(classifier, "parameter_info") else {}
        ),
        "initialization": initialization,
        "training_seed": training_seed,
        "strict_roi_input": True,
        "explicit_geometry_features": False,
        "upstream_segmenter_frozen": True,
        "new_random_three_logit_head": True,
        "loss_weight_normalization": (
            "minibatch_mean_divided_by_fold_fixed_global_weight_mean"
        ),
        "settings": settings,
        "test_data_read": False,
    }
    _save_json(model_info_path, model_info)

    train_dataset = ThreeClassROICacheDataset(
        selected,
        augmentation=settings["augmentation"],
        seed=training_seed + int(settings["seed_offsets"]["augmentation"]),
    )
    if len(train_dataset) != len(selected):
        raise RuntimeError("Training dataset changed the locked optimization row set")
    sample_weights = torch.as_tensor(
        threeclass_classifier_loss_weights(selected), dtype=torch.float32, device=device
    )
    global_weight_mean = sample_weights.mean().detach()
    if not bool(torch.isfinite(sample_weights).all()) or bool(
        (sample_weights <= 0).any()
    ):
        raise ProtocolGateError(
            "Classifier loss weights must be finite and strictly positive"
        )
    if not bool(torch.isfinite(global_weight_mean)) or float(global_weight_mean) <= 0:
        raise ProtocolGateError("Global classifier weight mean is invalid")
    optimizer = torch.optim.AdamW(
        (p for p in classifier.parameters() if p.requires_grad),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(settings["maximum_epochs"])
    )
    fingerprint = _checkpoint_fingerprint(
        cfg, model, seed, strategy, train_index_path, validation_index_path
    )
    history: list[dict[str, Any]] = []
    first_epoch = 0
    best_epoch = -1
    best_key = (-math.inf, -math.inf)
    stale = 0
    if last_path.is_file():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        if state.get("fingerprint") != fingerprint:
            raise ProtocolGateError("Resume checkpoint fingerprint differs from current run")
        classifier.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history = list(state["history"])
        first_epoch = int(state["epoch"]) + 1
        best_epoch = int(state["best_epoch"])
        best_key = tuple(float(value) for value in state["best_key"])
        stale = int(state["stale"])
        _restore_rng_state(state["rng"])

    maximum_epochs = int(settings["maximum_epochs"])
    patience = int(settings["patience"])
    for epoch in range(first_epoch, maximum_epochs):
        if stale >= patience:
            break
        classifier.train()
        train_dataset.epoch = epoch
        loader = _loader(
            train_dataset,
            batch_size=int(settings["batch_size"]),
            shuffle=True,
            seed=training_seed + epoch,
            num_workers=int(settings["num_workers"]),
        )
        optimizer.zero_grad(set_to_none=True)
        weighted_loss_sum = 0.0
        weight_sum = 0.0
        started = time.perf_counter()
        for batch in loader:
            batch = _to_device(batch, device)
            with _amp_context(settings, device):
                logits = _classifier_forward(classifier, batch, strategy=strategy)
                losses = F.cross_entropy(logits, batch["label"], reduction="none")
                weights = sample_weights[batch["index"]]
                loss = _globally_normalized_weighted_loss(
                    losses, weights, global_weight_mean
                )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite classifier training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                classifier.parameters(),
                float(settings["gradient_clip"]),
                error_if_nonfinite=True,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            weighted_loss_sum += float((losses.detach() * weights).sum().cpu())
            weight_sum += float(weights.sum().cpu())
        scheduler.step()
        validation_frames, _ = _infer_frames(
            cfg,
            classifier,
            validation_index,
            model=model,
            seed=seed,
            strategy=strategy,
            split="validation",
        )
        validation_eyes, validation_patients = _aggregate_frames(validation_frames)
        monitor = _selection_monitor(validation_patients)
        key = tuple(float(value) for value in monitor.pop("key"))
        improved = key > best_key
        if improved:
            best_key = key
            best_epoch = epoch
            stale = 0
            _torch_save(
                selected_path,
                {
                    "model": classifier.state_dict(),
                    "epoch": epoch,
                    "monitor": _clean(monitor),
                    "fingerprint": fingerprint,
                    "classifier_outputs": 3,
                },
            )
        else:
            stale += 1
        event = {
            "epoch": epoch + 1,
            "train_weighted_loss": weighted_loss_sum / max(weight_sum, 1e-12),
            "validation_patient_macro_nll": monitor["macro_nll"],
            "validation_patient_failure_aware_balanced_accuracy": monitor[
                "failure_aware_balanced_accuracy"
            ],
            "validation_monitor_status": monitor["status"],
            "best_epoch": best_epoch + 1,
            "stale_epochs": stale,
            "learning_rate_after_step": scheduler.get_last_lr()[0],
            "seconds": time.perf_counter() - started,
        }
        history.append(event)
        _save_csv(history_path, pd.DataFrame(history))
        _torch_save(
            last_path,
            {
                "model": classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_key": best_key,
                "stale": stale,
                "history": history,
                "rng": _rng_state(),
                "fingerprint": fingerprint,
            },
        )
        _write_progress(
            cfg,
            phase="classifier_training",
            model=model,
            seed=seed,
            strategy=strategy,
            epoch=epoch + 1,
            best_epoch=best_epoch + 1,
            stale_epochs=stale,
            validation_patient_macro_nll=monitor["macro_nll"],
            validation_patient_failure_aware_balanced_accuracy=monitor[
                "failure_aware_balanced_accuracy"
            ],
        )
    if not selected_path.is_file() or best_epoch < 0:
        raise ProtocolGateError("No evaluable validation checkpoint was selected")
    selected_checkpoint = torch.load(selected_path, map_location="cpu", weights_only=False)
    if selected_checkpoint.get("fingerprint") != fingerprint:
        raise ProtocolGateError("Selected checkpoint fingerprint mismatch")
    classifier.load_state_dict(selected_checkpoint["model"], strict=True)
    validation_frames, timing = _infer_frames(
        cfg,
        classifier,
        validation_index,
        model=model,
        seed=seed,
        strategy=strategy,
        split="validation",
    )
    validation_eyes, validation_patients = _aggregate_frames(validation_frames)
    _save_csv(validation_frames_path, validation_frames)
    _save_csv(validation_eyes_path, validation_eyes)
    _save_csv(validation_patients_path, validation_patients)
    model_info["selected_epoch"] = best_epoch + 1
    model_info["training_epochs_completed"] = len(history)
    model_info["early_stopped"] = len(history) < maximum_epochs
    model_info["peak_gpu_memory_bytes"] = int(torch.cuda.max_memory_allocated(device))
    model_info["validation_inference"] = timing
    _save_json(model_info_path, model_info)
    receipt = _generic_receipt(
        cfg,
        "train-classifier",
        model=model,
        seed=seed,
        strategy=strategy,
        artifacts={
            "selected_checkpoint": selected_path,
            "history": history_path,
            "validation_frames_raw": validation_frames_path,
            "validation_eyes_raw": validation_eyes_path,
            "validation_patients_raw": validation_patients_path,
            "model_info": model_info_path,
            "training_optimization_index": selected_index_path,
            "training_eye_eligibility": ledger_path,
        },
        metadata={
            "best_epoch": best_epoch + 1,
            "test_data_read": False,
            "training_rows": len(selected),
            "eligible_training_eyes": int(ledger["training_eligible"].sum()),
        },
    )
    del classifier
    torch.cuda.empty_cache()
    _write_progress(
        cfg,
        phase="classifier_training_complete",
        model=model,
        seed=seed,
        strategy=strategy,
        best_epoch=best_epoch + 1,
    )
    return {"status": "complete", "receipt": str(receipt), "checkpoint": str(selected_path)}


def _calibration_record(result: Mapping[str, Any]) -> dict[str, Any]:
    status = str(result.get("status", ""))
    temperature = result.get("temperature")
    available = (
        status.startswith("ok")
        and temperature is not None
        and np.isfinite(temperature)
        and not isinstance(temperature, bool)
        and float(temperature) > 0.0
    )
    return {
        "status": "available" if available else "unavailable",
        "temperature": float(temperature) if available else None,
        "reason": None if available else status or "calibration_fit_failed",
        "fit": _clean(result),
    }


def _calibrate_units(table: pd.DataFrame, temperature: float | None) -> pd.DataFrame:
    if (
        temperature is None
        or isinstance(temperature, bool)
        or not np.isfinite(float(temperature))
        or float(temperature) <= 0.0
    ):
        raise ProtocolGateError(
            "Calibration is unavailable; the fail-closed protocol forbids test access"
        )
    result = table.copy()
    valid = result["evaluable"].to_numpy(dtype=bool)
    probability = result.loc[:, PROBABILITY_COLUMNS].to_numpy(float)
    calibrated = np.full_like(probability, np.nan, dtype=float)
    if temperature is not None and valid.any():
        pseudo_logits = np.log(np.clip(probability[valid], 1e-7, 1.0))
        calibrated[valid] = metrics.apply_temperature_scaling(pseudo_logits, temperature)
    for offset, column in enumerate(PROBABILITY_COLUMNS):
        result[column] = calibrated[:, offset]
    result["prediction"] = metrics.ABSTAIN
    result.loc[valid & np.isfinite(calibrated).all(axis=1), "prediction"] = np.argmax(
        calibrated[valid & np.isfinite(calibrated).all(axis=1)], axis=1
    )
    result["prediction"] = result["prediction"].astype(int)
    result["confidence"] = np.where(
        valid & np.isfinite(calibrated).all(axis=1), np.nanmax(calibrated, axis=1), np.nan
    )
    result["evaluable"] = valid & np.isfinite(calibrated).all(axis=1)
    return result


def lock_validation(
    cfg: Mapping[str, Any], model: str, seed: int, strategy: str
) -> dict[str, Any]:
    _verify_generic_receipt(cfg, "prepare")
    verify_upstream_import_receipt(cfg, model, seed)
    lock_path = validation_lock_path(cfg, model, seed, strategy)
    if lock_path.is_file():
        lock = verify_validation_lock(cfg, model, seed, strategy)
        return {
            "status": "already_complete",
            "lock": str(lock_path),
            "calibration": lock["metadata"]["calibration"],
        }
    _verify_generic_receipt(
        cfg,
        "train-classifier",
        model=model,
        seed=seed,
        strategy=strategy,
    )
    work = _strategy_directory(cfg, model, seed, strategy)
    selected_path = work / "selected.pt"
    history_path = work / "history.csv"
    raw_frames_path = work / "validation_frames_raw.csv"
    raw_eyes_path = work / "validation_eyes_raw.csv"
    raw_patients_path = work / "validation_patients_raw.csv"
    model_info_path = work / "model_info.json"
    for path in (
        selected_path,
        history_path,
        raw_frames_path,
        raw_eyes_path,
        raw_patients_path,
        model_info_path,
    ):
        if not path.is_file():
            raise ProtocolGateError(f"Training artifact missing before validation lock: {path}")
    raw_eyes = pd.read_csv(raw_eyes_path)
    raw_patients = pd.read_csv(raw_patients_path)
    calibration: dict[str, Any] = {}
    for level, table in (("eye", raw_eyes), ("patient", raw_patients)):
        valid = table["evaluable"].astype(bool).to_numpy()
        fit = metrics.fit_temperature_on_probabilities(
            table.loc[valid, "label_3class"].astype(int),
            table.loc[valid, PROBABILITY_COLUMNS].to_numpy(float),
            require_all_classes=True,
        )
        record = _calibration_record(fit)
        calibration[level] = record
        if record["status"] != "available":
            raise ProtocolGateError(
                f"{level.capitalize()} calibration is unavailable "
                f"({record['reason']}); validation cannot be locked and global "
                "test access remains closed"
            )

    calibrated_paths: dict[str, Path] = {}
    raw_metrics: dict[str, Any] = {}
    calibrated_metrics: dict[str, Any] = {}
    for level, table in (("eye", raw_eyes), ("patient", raw_patients)):
        record = calibration[level]
        calibrated = _calibrate_units(table, record["temperature"])
        path = _save_csv(work / f"validation_{level}s_calibrated.csv", calibrated)
        calibrated_paths[level] = path
        raw_metrics[level] = metrics.selective_multiclass_metrics(
            table["label_3class"],
            table.loc[:, PROBABILITY_COLUMNS].to_numpy(float),
            evaluable=table["evaluable"].astype(bool).to_numpy(),
        )
        calibrated_metrics[level] = metrics.selective_multiclass_metrics(
            calibrated["label_3class"],
            calibrated.loc[:, PROBABILITY_COLUMNS].to_numpy(float),
            evaluable=calibrated["evaluable"].astype(bool).to_numpy(),
        )
    calibration_path = _save_json(work / "calibration_lock.json", calibration)
    validation_metrics_path = _save_json(
        work / "validation_metrics.json",
        {"raw": raw_metrics, "calibrated": calibrated_metrics},
    )
    monitor = _selection_monitor(raw_patients)
    metadata = {
        "classifier_strategy": strategy,
        "classifier_outputs": 3,
        "class_order": list(CLASS_ORDER),
        "training_roi_source": "frozen_out_of_fold_predicted_roi",
        "selection_partition": "validation",
        "primary_prediction_level": "patient",
        "primary_endpoint": "three_class_failure_aware_balanced_accuracy",
        "decision_rule": "argmax",
        "primary_abstention_policy": "structural_roi_gate_only",
        "test_data_read": False,
        "test_selection_or_refitting": False,
        "binary_head_threshold_or_calibrator_reused": False,
        "monitor_status": monitor["status"],
        "validation_patient_macro_nll": monitor["macro_nll"],
        "validation_patient_failure_aware_balanced_accuracy": monitor[
            "failure_aware_balanced_accuracy"
        ],
        "calibration": {
            level: {
                "status": calibration[level]["status"],
                "temperature": calibration[level]["temperature"],
                "reason": calibration[level]["reason"],
            }
            for level in ("eye", "patient")
        },
    }
    training_receipt = (
        output_root(cfg)
        / "state"
        / "receipts"
        / "train-classifier"
        / model
        / f"seed_{seed}"
        / f"{strategy}.json"
    )
    lock = write_validation_lock(
        cfg,
        model,
        seed,
        strategy,
        artifacts={
            "selected_checkpoint": selected_path,
            "history": history_path,
            "validation_frames_raw": raw_frames_path,
            "validation_eyes_raw": raw_eyes_path,
            "validation_patients_raw": raw_patients_path,
            "validation_eyes_calibrated": calibrated_paths["eye"],
            "validation_patients_calibrated": calibrated_paths["patient"],
            "calibration": calibration_path,
            "validation_metrics": validation_metrics_path,
            "model_info": model_info_path,
            "training_receipt": training_receipt,
        },
        metadata=metadata,
    )
    _write_progress(
        cfg, phase="validation_locked", model=model, seed=seed, strategy=strategy
    )
    return {"status": "complete", "lock": str(lock), "calibration": calibration}


def _load_selected_classifier(
    cfg: Mapping[str, Any], model: str, seed: int, strategy: str, device: torch.device
) -> torch.nn.Module:
    verify_validation_lock(cfg, model, seed, strategy)
    classifier = _build_classifier(cfg, model, strategy, pretrained=False).to(device)
    checkpoint = torch.load(
        _strategy_directory(cfg, model, seed, strategy) / "selected.pt",
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("classifier_outputs") != 3:
        raise ProtocolGateError("Selected checkpoint is not a three-logit classifier")
    classifier.load_state_dict(checkpoint["model"], strict=True)
    classifier.eval()
    return classifier


def _write_confusion_table(
    destination: Path, matrix: Sequence[Sequence[int]], proportions: Sequence[Sequence[float]]
) -> Path:
    rows = []
    for actual, class_name in CLASS_NAMES.items():
        for predicted, predicted_name in enumerate(
            [CLASS_NAMES[0], CLASS_NAMES[1], CLASS_NAMES[2], "abstain"]
        ):
            rows.append(
                {
                    "actual_class": int(actual),
                    "actual_name": class_name,
                    "predicted_class": int(predicted) if predicted < 3 else metrics.ABSTAIN,
                    "predicted_name": predicted_name,
                    "count": int(matrix[actual][predicted]),
                    "row_proportion": float(proportions[actual][predicted]),
                }
            )
    return _save_csv(destination, pd.DataFrame(rows))


def _calibration_rows(metric: Mapping[str, Any], *, level: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    conditional = metric.get("conditional", {})
    for class_name, values in conditional.get("per_class", {}).items():
        for strategy, key in (
            ("uniform", "calibration_bins_uniform"),
            ("quantile", "calibration_bins_quantile"),
        ):
            for row in values.get(key, []):
                rows.append(
                    {"level": level, "scope": "one_vs_rest", "class": class_name, "binning": strategy, **row}
                )
    for strategy, key in (
        ("uniform", "top_label_calibration_bins_uniform"),
        ("quantile", "top_label_calibration_bins_quantile"),
    ):
        for row in conditional.get(key, []):
            rows.append(
                {"level": level, "scope": "top_label", "class": "all", "binning": strategy, **row}
            )
    return pd.DataFrame(rows)


def _segmentation_class_table(
    cfg: Mapping[str, Any], model: str, seed: int
) -> tuple[pd.DataFrame, Path]:
    """Summarize the clean post-gate segmentation-only audit."""

    frame, path = segmentation_only_audit(cfg, model, seed)
    required = {
        "patient_id",
        "case_id",
        "label_3class",
        "segmentation_roi_valid",
        "dice",
        "iou",
        "roi_hit",
    }
    if not required <= set(frame):
        raise ProtocolGateError("Clean segmentation-only frame audit lacks required columns")
    frame["segmentation_roi_valid"] = frame["segmentation_roi_valid"].astype(bool)
    rows: list[dict[str, Any]] = []
    for class_index, group in frame.groupby("label_3class", sort=True):
        valid = group["segmentation_roi_valid"].to_numpy(bool)
        row: dict[str, Any] = {
            "model": model,
            "seed": int(seed),
            "class_index": int(class_index),
            "class_name": CLASS_NAMES[int(class_index)],
            "frames": int(len(group)),
            "eyes": int(group["case_id"].nunique()),
            "patients": int(group["patient_id"].nunique()),
            "roi_coverage": float(valid.mean()),
            "roi_failures": int((~valid).sum()),
        }
        for column in ("dice", "iou", "roi_hit"):
            values = pd.to_numeric(group[column], errors="coerce")
            row[f"{column}_all_mean"] = float(values.mean())
            row[f"{column}_all_sd"] = float(values.std(ddof=1))
            row[f"{column}_valid_mean"] = float(values.loc[valid].mean())
        rows.append(row)
    return pd.DataFrame(rows), path


def evaluate(
    cfg: Mapping[str, Any], model: str, seed: int, strategy: str
) -> dict[str, Any]:
    assert_test_access_open(cfg)
    verify_validation_lock(cfg, model, seed, strategy)
    completed = _generic_receipt_path(
        cfg, "evaluate", model=model, seed=seed, strategy=strategy
    )
    if completed.is_file():
        _verify_generic_receipt(
            cfg, "evaluate", model=model, seed=seed, strategy=strategy
        )
        verify_deferred_test_import_receipt(cfg, model, seed)
        return {
            "status": "already_complete",
            "receipt": str(completed),
            "metrics": str(
                unit_root(cfg, model, seed) / "evaluation" / strategy / "metrics.json"
            ),
        }
    deferred = upstream_artifact_paths(cfg, model, seed, phase="test")
    receipt_path = write_deferred_test_import_receipt(
        cfg, model, seed, artifact_paths=deferred
    )
    verify_deferred_test_import_receipt(cfg, model, seed)
    shared_audit_receipt = _generic_receipt_path(
        cfg, "audit-upstream-test", model=model, seed=seed
    )
    shared_audit_path = (
        unit_root(cfg, model, seed) / "upstream" / "test_import_audit.json"
    )
    artifact = discover_frozen_upstream_artifacts(
        cfg["frozen_upstream"]["source_output"],
        families=[model],
        seeds=[seed],
        require_test=True,
    )[0]
    if shared_audit_receipt.is_file():
        _verify_generic_receipt(
            cfg, "audit-upstream-test", model=model, seed=seed
        )
        audit = read_json(shared_audit_path)
    else:
        # Test cache bytes and embedded identities are verified only after the
        # global lock gate, once per model x seed and reused by both strategies.
        audit = audit_frozen_upstream_artifacts(
            artifact,
            verify_referenced_files=True,
            verify_payload_identity=True,
        )
        _save_json(shared_audit_path, audit)
        _generic_receipt(
            cfg,
            "audit-upstream-test",
            model=model,
            seed=seed,
            artifacts={
                "audit": shared_audit_path,
                "deferred_test_import_receipt": deferred_test_import_receipt_path(
                    cfg, model, seed
                ),
            },
            metadata={"test_gate_verified": True, "strategies_reusing_audit": 2},
        )
    test_index = _read_roi_index(artifact.test_index)
    _verify_partition_rows(cfg, test_index, seed, "test")
    device = _device_required()
    torch.cuda.reset_peak_memory_stats(device)
    classifier = _load_selected_classifier(cfg, model, seed, strategy, device)
    frames, timing = _infer_frames(
        cfg,
        classifier,
        test_index,
        model=model,
        seed=seed,
        strategy=strategy,
        split="test",
    )
    raw_eyes, raw_patients = _aggregate_frames(frames)
    eye_localized, patient_localized = localization_vectors(
        cfg, model, seed, raw_eyes, raw_patients
    )
    # Retrospective reference-overlap audit only.  These flags are attached
    # after inference and never affect ROI validity, calibration, prediction,
    # model selection, or structural abstention.
    raw_eyes["localized"] = eye_localized
    raw_patients["localized"] = patient_localized
    calibration = read_json(
        _strategy_directory(cfg, model, seed, strategy) / "calibration_lock.json"
    )
    calibrated_eyes = _calibrate_units(raw_eyes, calibration["eye"]["temperature"])
    calibrated_patients = _calibrate_units(
        raw_patients, calibration["patient"]["temperature"]
    )
    tables = {
        "raw": {"eye": raw_eyes, "patient": raw_patients},
        "calibrated": {"eye": calibrated_eyes, "patient": calibrated_patients},
    }
    evaluation_metrics: dict[str, Any] = {}
    evaluation_root = unit_root(cfg, model, seed) / "evaluation" / strategy
    evaluation_root.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Path] = {
        "frames_raw": _save_csv(evaluation_root / "frames_raw.csv", frames),
        "deferred_test_import_receipt": receipt_path,
        "upstream_test_audit": shared_audit_path,
    }
    calibration_tables: list[pd.DataFrame] = []
    risk_tables: list[pd.DataFrame] = []
    for state, levels in tables.items():
        evaluation_metrics[state] = {}
        for level, table in levels.items():
            table_path = _save_csv(evaluation_root / f"{level}s_{state}.csv", table)
            artifacts[f"{level}s_{state}"] = table_path
            result = metrics.selective_multiclass_metrics(
                table["label_3class"],
                table.loc[:, PROBABILITY_COLUMNS].to_numpy(float),
                evaluable=table["evaluable"].astype(bool).to_numpy(),
                localized_success=table["localized"].astype(bool).to_numpy(),
            )
            evaluation_metrics[state][level] = result
            confusion_path = _write_confusion_table(
                evaluation_root / f"confusion_3x4_{level}_{state}.csv",
                result["confusion_matrix_3x4"],
                result["confusion_matrix_3x4_row_proportions"],
            )
            artifacts[f"confusion_{level}_{state}"] = confusion_path
            calibration_rows = _calibration_rows(result, level=level)
            if len(calibration_rows):
                calibration_rows["probability_state"] = state
                calibration_tables.append(calibration_rows)
            curve = metrics.multiclass_risk_coverage_curve(
                table["label_3class"],
                table.loc[:, PROBABILITY_COLUMNS].to_numpy(float),
                evaluable=table["evaluable"].astype(bool).to_numpy(),
                include_abstentions_as_failures=True,
            )
            curve["level"] = level
            curve["probability_state"] = state
            risk_tables.append(curve)
    metrics_path = _save_json(evaluation_root / "metrics.json", evaluation_metrics)
    artifacts["metrics"] = metrics_path
    artifacts["calibration_curves"] = _save_csv(
        evaluation_root / "calibration_curves.csv",
        pd.concat(calibration_tables, ignore_index=True) if calibration_tables else pd.DataFrame(),
    )
    artifacts["risk_coverage_curves"] = _save_csv(
        evaluation_root / "risk_coverage_curves.csv",
        pd.concat(risk_tables, ignore_index=True) if risk_tables else pd.DataFrame(),
    )
    draws = int(cfg["statistics"]["bootstrap_draws"])
    bootstrap = {
        "patient_calibrated": metrics.patient_cluster_bootstrap_ci(
            calibrated_patients,
            draws=draws,
            seed=int(seed) + int(_training_settings(cfg)["seed_offsets"]["bootstrap"]),
            ci_method="bca",
        ),
        "eye_calibrated": metrics.patient_cluster_bootstrap_ci(
            calibrated_eyes,
            draws=draws,
            seed=int(seed) + int(_training_settings(cfg)["seed_offsets"]["bootstrap"]) + 1,
            ci_method="bca",
        ),
    }
    artifacts["bootstrap_ci"] = _save_json(
        evaluation_root / "bootstrap_patient_cluster_ci.json", bootstrap
    )
    segmentation, segmentation_source = _segmentation_class_table(cfg, model, seed)
    artifacts["segmentation_only_frame_audit"] = segmentation_source
    artifacts["segmentation_only_frame_audit_receipt"] = (
        segmentation_source.with_suffix(".receipt.json")
    )
    artifacts["segmentation_by_class"] = _save_csv(
        evaluation_root / "segmentation_by_class.csv", segmentation
    )
    artifacts["resource_timing"] = _save_json(
        evaluation_root / "resource_timing.json",
        {
            **timing,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "model": model,
            "seed": int(seed),
            "strategy": strategy,
        },
    )
    result_receipt = _generic_receipt(
        cfg,
        "evaluate",
        model=model,
        seed=seed,
        strategy=strategy,
        artifacts=artifacts,
        metadata={
            "primary_level": "patient",
            "primary_probability_state": "calibrated",
            "test_selection_or_refitting": False,
            "source_segmentation_frame_audit": str(segmentation_source),
            "analysis_status": cfg["analysis_status"],
            "prior_test_use_disclosure": cfg["prior_test_use"]["required_disclosure"],
        },
    )
    del classifier
    torch.cuda.empty_cache()
    _write_progress(
        cfg, phase="evaluation_complete", model=model, seed=seed, strategy=strategy
    )
    return {"status": "complete", "receipt": str(result_receipt), "metrics": str(metrics_path)}


def _flatten_numeric(
    value: Mapping[str, Any], *, prefix: str = ""
) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            result.update(_flatten_numeric(item, prefix=name))
        elif isinstance(item, (int, float, np.integer, np.floating)) and not isinstance(item, bool):
            result[name] = float(item)
    return result


def _holm_adjust(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    m = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (m - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def _failure_aware_ba(table: pd.DataFrame) -> float:
    result = metrics.selective_multiclass_metrics(
        table["label_3class"],
        table.loc[:, PROBABILITY_COLUMNS].to_numpy(float),
        evaluable=table["evaluable"].astype(bool).to_numpy(),
    )
    return float(result["failure_aware_balanced_accuracy"])


def _paired_primary_comparisons(
    cfg: Mapping[str, Any], all_patients: pd.DataFrame
) -> pd.DataFrame:
    from scipy import stats as scipy_stats

    primary = all_patients.loc[
        (all_patients["strategy"] == cfg["classifier"]["primary_strategy"])
        & (all_patients["probability_state"] == "calibrated")
    ].copy()
    draws = int(cfg["statistics"]["bootstrap_draws"])
    if draws != 5000:
        raise ProtocolGateError("Primary paired comparisons require exactly 5000 draws")
    models = tuple(str(model) for model in cfg["models"])
    seeds = tuple(int(seed) for seed in cfg["split_seeds"])
    pair_count = math.comb(len(models), 2)
    family_size = pair_count * len(seeds)
    if family_size != 30:
        raise ProtocolGateError(
            "The primary Holm family must contain 6 unordered model pairs "
            "within each of 5 seeds (30 comparisons)"
        )
    required = {
        "model",
        "seed",
        "patient_id",
        "label_3class",
        "evaluable",
        "strategy",
        "probability_state",
        *PROBABILITY_COLUMNS,
    }
    missing_columns = required - set(primary)
    if missing_columns:
        raise ProtocolGateError(
            f"Primary paired comparison columns are missing: {sorted(missing_columns)}"
        )

    family_id = "primary_patient_failure_aware_balanced_accuracy_30_test_holm"
    training = cfg.get("training", {})
    configured_offsets = (
        training.get("seed_offsets", {}) if isinstance(training, Mapping) else {}
    )
    bootstrap_offset = int(
        configured_offsets.get(
            "bootstrap", DEFAULT_TRAINING["seed_offsets"]["bootstrap"]
        )
    )

    def comparison_seed(
        namespace: str, split_seed: int, left: str, right: str
    ) -> int:
        material = {
            "namespace": namespace,
            "study_id": cfg.get("study_id"),
            "config_sha256": cfg.get("config_sha256"),
            "bootstrap_seed_offset": bootstrap_offset,
            "bootstrap_draws": draws,
            "models": models,
            "split_seeds": seeds,
            "split_seed": int(split_seed),
            "left_model": left,
            "right_model": right,
        }
        digest = hashlib.sha256(
            json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).digest()
        return int.from_bytes(digest[:8], "little") % (2**32)

    def evaluable_flags(values: pd.Series, *, model: str, seed: int) -> np.ndarray:
        flags: list[bool] = []
        for value in values.tolist():
            if pd.isna(value):
                flags.append(False)
            elif isinstance(value, (bool, np.bool_)):
                flags.append(bool(value))
            elif isinstance(value, (int, np.integer)) and int(value) in (0, 1):
                flags.append(bool(value))
            elif isinstance(value, str) and value.strip().lower() in {"true", "false"}:
                flags.append(value.strip().lower() == "true")
            else:
                raise ProtocolGateError(
                    f"Malformed evaluable flag for {model}, seed {seed}: {value!r}"
                )
        return np.asarray(flags, dtype=bool)

    def correct_or_failure(
        aligned: pd.DataFrame, suffix: str, *, model: str, seed: int
    ) -> np.ndarray:
        probability = aligned.loc[
            :, [f"{column}_{suffix}" for column in PROBABILITY_COLUMNS]
        ].to_numpy(float)
        evaluable = evaluable_flags(
            aligned[f"evaluable_{suffix}"], model=model, seed=seed
        )
        usable = evaluable & np.isfinite(probability).all(axis=1)
        prediction = np.full(len(aligned), metrics.ABSTAIN, dtype=int)
        prediction[usable] = np.argmax(probability[usable], axis=1).astype(int)
        # Structural abstention and any missing probability remain ABSTAIN and
        # therefore contribute zero recall for the patient's true class.
        return (prediction == aligned["label_3class"].to_numpy(int)).astype(float)

    def balanced_difference(differences: np.ndarray, labels: np.ndarray) -> float:
        class_means: list[float] = []
        for class_index in CLASS_ORDER:
            values = differences[labels == int(class_index)]
            if not len(values):
                return np.nan
            class_means.append(float(values.mean()))
        return float(np.mean(class_means))

    def bca_interval(
        estimate: float, distribution: np.ndarray, jackknife: np.ndarray
    ) -> tuple[float, float, str, str | None]:
        finite = distribution[np.isfinite(distribution)]
        low = float(np.quantile(finite, 0.025)) if len(finite) else np.nan
        high = float(np.quantile(finite, 0.975)) if len(finite) else np.nan
        if (
            not len(finite)
            or len(jackknife) < 3
            or not np.isfinite(estimate)
            or not np.isfinite(jackknife).all()
        ):
            return low, high, "percentile", "bca_not_identified"
        proportion = (
            np.sum(finite < estimate) + 0.5 * np.sum(finite == estimate)
        ) / len(finite)
        proportion = float(
            np.clip(proportion, 1 / (2 * len(finite)), 1 - 1 / (2 * len(finite)))
        )
        z0 = float(scipy_stats.norm.ppf(proportion))
        deviations = float(jackknife.mean()) - jackknife
        denominator = 6 * float(np.sum(deviations**2) ** 1.5)
        acceleration = (
            float(np.sum(deviations**3)) / denominator if denominator > 0 else 0.0
        )
        adjusted: list[float] = []
        for alpha in (0.025, 0.975):
            z_alpha = float(scipy_stats.norm.ppf(alpha))
            divisor = 1 - acceleration * (z0 + z_alpha)
            if abs(divisor) <= 1e-12:
                return low, high, "percentile", "zero_bca_denominator"
            adjusted.append(
                float(scipy_stats.norm.cdf(z0 + (z0 + z_alpha) / divisor))
            )
        if not np.isfinite(adjusted).all() or adjusted[0] > adjusted[1]:
            return low, high, "percentile", "invalid_bca_quantiles"
        low, high = map(float, np.quantile(finite, np.clip(adjusted, 0, 1)))
        return low, high, "bca", None

    rows: list[dict[str, Any]] = []
    for split_seed in seeds:
        seed_primary = primary.loc[primary["seed"].astype(int) == split_seed].copy()
        for pair_index, (left, right) in enumerate(itertools.combinations(models, 2)):
            left_table = seed_primary.loc[seed_primary["model"].astype(str) == left].copy()
            right_table = seed_primary.loc[seed_primary["model"].astype(str) == right].copy()
            left_table["patient_id"] = left_table["patient_id"].astype(str)
            right_table["patient_id"] = right_table["patient_id"].astype(str)
            if left_table["patient_id"].duplicated().any() or right_table[
                "patient_id"
            ].duplicated().any():
                raise ProtocolGateError(
                    f"Patient pairing is not one-to-one for {left} versus {right}, "
                    f"seed {split_seed}"
                )
            aligned = left_table.merge(
                right_table,
                on="patient_id",
                suffixes=("_left", "_right"),
                how="inner",
                validate="one_to_one",
                sort=True,
            )
            if (
                not len(aligned)
                or len(aligned) != len(left_table)
                or len(aligned) != len(right_table)
            ):
                raise ProtocolGateError(
                    f"Same-seed patient pairing failed for {left} versus {right}, "
                    f"seed {split_seed}"
                )
            left_labels = aligned["label_3class_left"].to_numpy(int)
            right_labels = aligned["label_3class_right"].to_numpy(int)
            if not np.array_equal(left_labels, right_labels):
                raise ProtocolGateError(
                    f"Patient labels differ for {left} versus {right}, seed {split_seed}"
                )
            aligned["label_3class"] = left_labels
            labels = left_labels
            if set(labels.tolist()) != set(int(value) for value in CLASS_ORDER):
                raise ProtocolGateError(
                    f"All three label strata are required for seed {split_seed}"
                )
            left_correct = correct_or_failure(
                aligned, "left", model=left, seed=split_seed
            )
            right_correct = correct_or_failure(
                aligned, "right", model=right, seed=split_seed
            )
            differences = left_correct - right_correct
            estimate = balanced_difference(differences, labels)

            bootstrap_seed = comparison_seed("paired_bootstrap", split_seed, left, right)
            bootstrap_rng = np.random.default_rng(bootstrap_seed)
            bootstrap_components: list[np.ndarray] = []
            class_counts: dict[int, int] = {}
            for class_index in CLASS_ORDER:
                values = differences[labels == int(class_index)]
                class_counts[int(class_index)] = int(len(values))
                indices = bootstrap_rng.integers(
                    0, len(values), size=(draws, len(values))
                )
                bootstrap_components.append(values[indices].mean(axis=1))
            distribution = np.mean(np.stack(bootstrap_components, axis=0), axis=0)
            jackknife = np.asarray(
                [
                    balanced_difference(
                        np.delete(differences, index), np.delete(labels, index)
                    )
                    for index in range(len(aligned))
                ],
                dtype=float,
            )
            ci_low, ci_high, ci_method, ci_fallback_reason = bca_interval(
                estimate, distribution, jackknife
            )

            randomization_seed = comparison_seed(
                "paired_sign_flip", split_seed, left, right
            )
            randomization_rng = np.random.default_rng(randomization_seed)
            randomized_components: list[np.ndarray] = []
            for class_index in CLASS_ORDER:
                values = differences[labels == int(class_index)]
                signs = (
                    randomization_rng.integers(
                        0, 2, size=(draws, len(values)), dtype=np.int8
                    )
                    * 2
                    - 1
                ).astype(float)
                randomized_components.append((signs * values).mean(axis=1))
            randomization_distribution = np.mean(
                np.stack(randomized_components, axis=0), axis=0
            )
            finite_randomized = randomization_distribution[
                np.isfinite(randomization_distribution)
            ]
            p_value = float(
                (
                    1
                    + np.sum(
                        np.abs(finite_randomized) >= abs(estimate) - 1e-15
                    )
                )
                / (len(finite_randomized) + 1)
            )
            rows.append(
                {
                    "seed": int(split_seed),
                    "left_model": left,
                    "right_model": right,
                    "contrast": f"{left} - {right}",
                    "metric": "patient_failure_aware_balanced_accuracy",
                    "difference_left_minus_right": float(estimate),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "ci_method": ci_method,
                    "ci_fallback_reason": ci_fallback_reason,
                    "confidence_level": 0.95,
                    "bootstrap_se": float(distribution.std(ddof=1)),
                    "bootstrap_draws": draws,
                    "valid_bootstrap_draws": int(np.isfinite(distribution).sum()),
                    "bootstrap_seed": bootstrap_seed,
                    "bootstrap_stratified_by_class": True,
                    "paired": True,
                    "resampling_unit": "patient_within_seed",
                    "n_paired_patients": int(len(aligned)),
                    "n_class_0": class_counts[0],
                    "n_class_1": class_counts[1],
                    "n_class_2": class_counts[2],
                    "p_value_randomization_two_sided": p_value,
                    "p_value_two_sided": p_value,
                    "p_value_method": (
                        "whole-patient paired model-label-swap sign-flip "
                        "Monte Carlo randomization with plus-one correction"
                    ),
                    "randomization_draws": draws,
                    "valid_randomization_draws": int(len(finite_randomized)),
                    "randomization_seed": randomization_seed,
                    "family_id": family_id,
                    "planned_family_size": family_size,
                    "analysis_status": cfg.get(
                        "analysis_status", "exploratory_post_hoc_internal"
                    ),
                    "exploratory": True,
                    "descriptive": False,
                    "confirmatory": False,
                    "pooled_across_seeds": False,
                    "overlapping_seed_holdouts_not_independent": True,
                    "pair_index_within_seed": pair_index,
                }
            )
    result = pd.DataFrame(rows)
    if len(result) != family_size:
        raise ProtocolGateError(
            f"Primary paired comparison family has {len(result)} rows, expected {family_size}"
        )
    result["p_value_holm"] = _holm_adjust(
        result["p_value_randomization_two_sided"]
    )
    result["reject_holm_0_05"] = result["p_value_holm"] <= 0.05
    return result


def _publication_figures(
    cfg: Mapping[str, Any], per_seed: pd.DataFrame, all_patients: pd.DataFrame,
    segmentation: pd.DataFrame,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper")
    figure_root = output_root(cfg) / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    primary = per_seed.loc[
        (per_seed["level"] == "patient")
        & (per_seed["probability_state"] == "calibrated")
    ].copy()
    metric = "failure_aware_balanced_accuracy"
    if metric in primary:
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
        sns.barplot(data=primary, x="model", y=metric, hue="strategy", errorbar="sd", ax=axes[0])
        axes[0].set_ylim(0, 1)
        axes[0].set_title("Patient-level failure-aware balanced accuracy")
        axes[0].tick_params(axis="x", rotation=25)
        sns.barplot(data=primary, x="model", y="coverage", hue="strategy", errorbar="sd", ax=axes[1])
        axes[1].set_ylim(0, 1)
        axes[1].set_title("Patient-level operational coverage")
        axes[1].tick_params(axis="x", rotation=25)
        for extension in ("png", "pdf", "svg"):
            path = figure_root / f"patient_primary_performance.{extension}"
            figure.savefig(path, dpi=300, bbox_inches="tight")
            outputs.append(path)
        plt.close(figure)

    if len(segmentation):
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
        sns.barplot(
            data=segmentation,
            x="model",
            y="dice_all_mean",
            hue="class_name",
            errorbar="sd",
            ax=axes[0],
        )
        axes[0].set_title("Class-conditional segmentation Dice")
        axes[0].tick_params(axis="x", rotation=25)
        sns.barplot(
            data=segmentation,
            x="model",
            y="roi_coverage",
            hue="class_name",
            errorbar="sd",
            ax=axes[1],
        )
        axes[1].set_title("Class-conditional ROI coverage")
        axes[1].set_ylim(0, 1)
        axes[1].tick_params(axis="x", rotation=25)
        for extension in ("png", "pdf", "svg"):
            path = figure_root / f"segmentation_class_conditional.{extension}"
            figure.savefig(path, dpi=300, bbox_inches="tight")
            outputs.append(path)
        plt.close(figure)

    primary_patients = all_patients.loc[
        (all_patients["strategy"] == cfg["classifier"]["primary_strategy"])
        & (all_patients["probability_state"] == "calibrated")
    ]
    for model in cfg["models"]:
        table = primary_patients.loc[primary_patients["model"] == model]
        result = metrics.selective_multiclass_metrics(
            table["label_3class"],
            table.loc[:, PROBABILITY_COLUMNS].to_numpy(float),
            evaluable=table["evaluable"].astype(bool).to_numpy(),
        )
        matrix = np.asarray(result["confusion_matrix_3x4"], dtype=int)
        figure, axis = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[CLASS_NAMES[0], CLASS_NAMES[1], CLASS_NAMES[2], "abstain"],
            yticklabels=[CLASS_NAMES[0], CLASS_NAMES[1], CLASS_NAMES[2]],
            ax=axis,
        )
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Reference standard")
        axis.set_title(f"{model}: patient-level 3×4 confusion (five seeds, descriptive)")
        for extension in ("png", "pdf", "svg"):
            path = figure_root / f"confusion_3x4_{model}.{extension}"
            figure.savefig(path, dpi=300, bbox_inches="tight")
            outputs.append(path)
        plt.close(figure)
    return outputs


def summarize(cfg: Mapping[str, Any]) -> dict[str, Any]:
    assert_test_access_open(cfg)
    completed = _generic_receipt_path(cfg, "summarize")
    if completed.is_file():
        _verify_generic_receipt(cfg, "summarize")
        validate_publication_outputs(cfg)
        validate_q1_figures(cfg)
        payload = read_json(completed)
        return {
            "status": "already_complete",
            "receipt": str(completed),
            "artifacts": [record["path"] for record in payload["artifacts"]],
        }
    rows: list[dict[str, Any]] = []
    patient_tables: list[pd.DataFrame] = []
    segmentation_tables: list[pd.DataFrame] = []
    for model in cfg["models"]:
        for seed in cfg["split_seeds"]:
            for strategy in cfg["classifier"]["strategy_order"]:
                evaluation_root = unit_root(cfg, model, seed) / "evaluation" / strategy
                receipt = (
                    output_root(cfg)
                    / "state"
                    / "receipts"
                    / "evaluate"
                    / model
                    / f"seed_{seed}"
                    / f"{strategy}.json"
                )
                if not receipt.is_file():
                    raise ProtocolGateError(f"Missing evaluation receipt: {receipt}")
                _verify_generic_receipt(
                    cfg,
                    "evaluate",
                    model=model,
                    seed=int(seed),
                    strategy=strategy,
                )
                metric_payload = read_json(evaluation_root / "metrics.json")
                for state in ("raw", "calibrated"):
                    for level in ("eye", "patient"):
                        flattened = _flatten_numeric(metric_payload[state][level])
                        rows.append(
                            {
                                "model": model,
                                "seed": int(seed),
                                "strategy": strategy,
                                "probability_state": state,
                                "level": level,
                                **flattened,
                            }
                        )
                        table = pd.read_csv(evaluation_root / f"{level}s_{state}.csv")
                        table["model"] = model
                        table["seed"] = int(seed)
                        table["strategy"] = strategy
                        table["probability_state"] = state
                        if level == "patient":
                            patient_tables.append(table)
                if strategy == cfg["classifier"]["primary_strategy"]:
                    segmentation_tables.append(
                        pd.read_csv(evaluation_root / "segmentation_by_class.csv")
                    )
    per_seed = pd.DataFrame(rows)
    summary_keys = ["model", "strategy", "probability_state", "level"]
    numeric = [column for column in per_seed if column not in [*summary_keys, "seed"]]
    aggregates: list[dict[str, Any]] = []
    for identity, group in per_seed.groupby(summary_keys, sort=True):
        for column in numeric:
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            if len(values):
                aggregates.append(
                    {
                        **dict(zip(summary_keys, identity, strict=True)),
                        "metric": column,
                        "n_seeds": int(len(values)),
                        "mean": float(values.mean()),
                        "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                        "minimum": float(values.min()),
                        "maximum": float(values.max()),
                    }
                )
    five_seed = pd.DataFrame(aggregates)
    all_patients = pd.concat(patient_tables, ignore_index=True)
    segmentation = pd.concat(segmentation_tables, ignore_index=True)
    segmentation_summary = (
        segmentation.groupby(["model", "class_index", "class_name"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            roi_coverage=("roi_coverage", "mean"),
            roi_coverage_sd=("roi_coverage", "std"),
            dice_all_mean=("dice_all_mean", "mean"),
            dice_all_sd=("dice_all_mean", "std"),
            iou_all_mean=("iou_all_mean", "mean"),
            roi_hit_all_mean=("roi_hit_all_mean", "mean"),
        )
    )
    comparisons = _paired_primary_comparisons(cfg, all_patients)
    table_root = output_root(cfg) / "tables"
    artifacts: dict[str, Path] = {
        "classification_per_seed": _save_csv(table_root / "classification_per_seed.csv", per_seed),
        "classification_five_seed": _save_csv(table_root / "classification_five_seed_mean_sd.csv", five_seed),
        "patient_predictions_all_seeds": _save_csv(table_root / "patient_predictions_all_seeds.csv", all_patients),
        "segmentation_per_seed_class": _save_csv(table_root / "segmentation_per_seed_class.csv", segmentation),
        "segmentation_five_seed_class": _save_csv(table_root / "segmentation_five_seed_class.csv", segmentation_summary),
        "paired_primary_comparisons": _save_csv(table_root / "paired_primary_patient_cluster_bootstrap_holm.csv", comparisons),
    }
    figures = _publication_figures(cfg, per_seed, all_patients, segmentation)
    for index, path in enumerate(figures):
        artifacts[f"figure_{index:02d}"] = path
    disclosure = _save_json(
        output_root(cfg) / "summary" / "interpretation_and_claim_limits.json",
        {
            "analysis_status": cfg["analysis_status"],
            "confirmatory_claim_allowed": False,
            "prior_test_use_disclosure": cfg["prior_test_use"]["required_disclosure"],
            "external_validation_status": "not_performed",
            "required_next_validation": cfg["prior_test_use"]["required_confirmatory_evaluation"],
            "small_test_class_warning": cfg["statistics"]["small_test_class_warning"],
            "seed_summary_role": "descriptive_only_because_patient_holdouts_overlap_across_seeds",
            "primary_endpoint": cfg["evaluation"]["primary_estimand"],
        },
    )
    artifacts["claim_limits"] = disclosure
    publication_artifacts = generate_publication_outputs(cfg)
    for name, path in publication_artifacts.items():
        artifacts[f"publication_{Path(name).stem}"] = path
    q1_figure_artifacts = generate_q1_figures(cfg)
    for name, path in q1_figure_artifacts.items():
        artifacts[f"q1_figure_{Path(name).stem}"] = path
    receipt = _generic_receipt(
        cfg,
        "summarize",
        artifacts=artifacts,
        metadata={
            "units": 20,
            "classifier_fits": 40,
            "test_selection_or_refitting": False,
            "q1_outputs_generated": True,
        },
    )
    _write_progress(cfg, phase="summarize_complete", completed_units=40, target_units=40)
    return {"status": "complete", "receipt": str(receipt), "artifacts": [str(path) for path in artifacts.values()]}


def audit(cfg: Mapping[str, Any]) -> dict[str, Any]:
    imports = assert_all_upstream_imports(cfg)
    locks = assert_all_validation_locks(cfg)
    sentinel = assert_test_access_open(cfg)
    _verify_generic_receipt(cfg, "summarize")
    missing_evaluations: list[str] = []
    for model in cfg["models"]:
        for seed in cfg["split_seeds"]:
            verify_deferred_test_import_receipt(cfg, model, seed)
            _verify_generic_receipt(
                cfg, "audit-upstream-test", model=model, seed=int(seed)
            )
            for strategy in cfg["classifier"]["strategy_order"]:
                path = (
                    output_root(cfg)
                    / "state"
                    / "receipts"
                    / "evaluate"
                    / model
                    / f"seed_{seed}"
                    / f"{strategy}.json"
                )
                if not path.is_file():
                    missing_evaluations.append(str(path))
                    continue
                _verify_generic_receipt(
                    cfg,
                    "evaluate",
                    model=model,
                    seed=int(seed),
                    strategy=strategy,
                )
    if missing_evaluations:
        raise ProtocolGateError(f"Missing evaluation receipts: {missing_evaluations[:4]}")
    descriptive_tables = [
        "classification_per_seed.csv",
        "classification_five_seed_mean_sd.csv",
        "patient_predictions_all_seeds.csv",
        "segmentation_per_seed_class.csv",
        "segmentation_five_seed_class.csv",
        "paired_primary_patient_cluster_bootstrap_holm.csv",
    ]
    required_tables = [*descriptive_tables, *PUBLICATION_TABLE_NAMES]
    missing_tables = [name for name in required_tables if not (output_root(cfg) / "tables" / name).is_file()]
    if missing_tables:
        raise ProtocolGateError(f"Required Q1 tables are missing: {missing_tables}")
    publication_audit = validate_publication_outputs(cfg)
    figure_audit = validate_q1_figures(cfg)
    completed = _generic_receipt_path(cfg, "audit")
    if completed.is_file():
        _verify_generic_receipt(cfg, "audit")
        _write_or_verify_final_provenance(cfg)
        final_path = output_root(cfg) / "summary" / "final_audit.json"
        return read_json(final_path)
    report = {
        "status": "passed",
        "study_id": cfg["study_id"],
        "development_import_receipts": len(imports),
        "validation_locks": len(locks),
        "deferred_test_import_receipts": 20,
        "post_gate_test_import_audit_receipts": 20,
        "evaluation_receipts": 40,
        "required_tables": required_tables,
        "publication_contract_audit": publication_audit,
        "publication_figure_audit": figure_audit,
        "test_access_sentinel": sentinel,
        "analysis_status": cfg["analysis_status"],
        "confirmatory_claim_allowed": False,
        "prior_test_use_disclosure": cfg["prior_test_use"]["required_disclosure"],
        "final_provenance_snapshot": str(
            output_root(cfg) / "summary" / "final_provenance_snapshot.json"
        ),
        "audited_utc": utc_now(),
    }
    path = _save_json(output_root(cfg) / "summary" / "final_audit.json", report)
    _generic_receipt(cfg, "audit", artifacts={"final_audit": path}, metadata={"status": "passed"})
    _write_or_verify_final_provenance(cfg)
    _write_progress(cfg, phase="complete", completed_units=40, target_units=40)
    return report


def run_core(cfg: Mapping[str, Any]) -> dict[str, Any]:
    prepare(cfg)
    for model in cfg["models"]:
        for seed in cfg["split_seeds"]:
            import_upstream(cfg, model, int(seed))
    for model in cfg["models"]:
        preflight(cfg, model)
    for model in cfg["models"]:
        for seed in cfg["split_seeds"]:
            for strategy in cfg["classifier"]["strategy_order"]:
                train_classifier(cfg, model, int(seed), strategy)
                lock_validation(cfg, model, int(seed), strategy)
    open_test_access(cfg)
    for model in cfg["models"]:
        for seed in cfg["split_seeds"]:
            for strategy in cfg["classifier"]["strategy_order"]:
                evaluate(cfg, model, int(seed), strategy)
    summarize(cfg)
    return audit(cfg)


__all__ = [
    "LOGIT_COLUMNS",
    "PROBABILITY_COLUMNS",
    "audit",
    "evaluate",
    "import_upstream",
    "lock_validation",
    "prepare",
    "preflight",
    "run_core",
    "summarize",
    "train_classifier",
]
