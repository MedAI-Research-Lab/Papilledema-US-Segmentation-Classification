"""Predeclared ablation planning, inputs, caches, locks, and test gate.

This module deliberately separates two execution scopes:

* every learned arm is fitted per segmenter and seed with that segmenter's
  model-specific strict-ROI classifier family (including whole-image and the
  GT-ROI oracle, whose classifier families must not be pooled);
* aggregation and minimum-valid-frame variants reuse primary frame
  probabilities (no classifier refit), but receive their own validation-only
  calibration/threshold locks.

No function reads clinical files implicitly.  A test input cannot be built
without a verified global ablation-test gate.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from torch import Tensor
from torch.nn import functional as F

from .roi import (
    DEFAULT_NEUTRAL_RGB,
    EIGHT_CONNECTED,
    ROIPolicy,
    ROIStatus,
    extract_roi_tensor,
    hard_mask_from_probability,
    roi_geometry,
)


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

SHARED_SEED_ARMS: tuple[str, ...] = ()
MODEL_SEED_ARMS = (
    "whole_image_baseline",
    "gt_roi_oracle",
    "bbox_context",
    "largest_component_without_quality_gates",
    "geometry_only",
    "mask_only",
    "background_only_negative_control",
    "random_roi_negative_control",
    "appearance_plus_geometry",
)
PREDICTED_MASK_ARMS = (
    "bbox_context",
    "largest_component_without_quality_gates",
    "geometry_only",
    "mask_only",
    "background_only_negative_control",
    "random_roi_negative_control",
    "appearance_plus_geometry",
)
ANALYTICAL_ARMS = ("aggregation_rule", "minimum_valid_frames")
EXPECTED_ARMS = SHARED_SEED_ARMS + MODEL_SEED_ARMS + ANALYTICAL_ARMS

EXPECTED_INPUTS = {
    "whole_image_baseline": "whole_image",
    "gt_roi_oracle": "strict_ground_truth_roi",
    "bbox_context": "unmasked_predicted_bbox",
    "largest_component_without_quality_gates": "largest_predicted_component_only",
    "geometry_only": "predicted_mask_geometry_features_only",
    "mask_only": "binary_predicted_mask",
    "background_only_negative_control": "image_outside_predicted_roi_only",
    "random_roi_negative_control": "patient_independent_size_matched_random_roi",
    "appearance_plus_geometry": "strict_roi_appearance_plus_geometry",
}
EXPECTED_VARIANTS = {
    "aggregation_rule": ("mean", "median", "maximum", "predicted_mask_confidence_weighted_mean"),
    "minimum_valid_frames": (1, 4, 7),
}


class AblationScope(str, Enum):
    SEED_SHARED = "seed_shared"
    MODEL_SEED = "model_seed"
    ANALYTICAL_MODEL_SEED = "analytical_model_seed"


@dataclass(frozen=True)
class AblationSpec:
    name: str
    scope: AblationScope
    input_kind: str | None
    variants: tuple[str | int, ...] = ()
    requires_classifier_fit: bool = True

    @property
    def uses_predicted_mask(self) -> bool:
        return self.name in PREDICTED_MASK_ARMS

    @property
    def uses_ground_truth_input(self) -> bool:
        return self.name == "gt_roi_oracle"


@dataclass(frozen=True)
class AblationTask:
    arm: str
    scope: AblationScope
    seed: int
    model: str | None = None
    variant: str | int | None = None
    requires_classifier_fit: bool = True

    @property
    def key(self) -> str:
        if self.scope is AblationScope.SEED_SHARED:
            return f"seed_shared/seed_{self.seed}/{self.arm}"
        base = f"model_seed/{self.model}/seed_{self.seed}/{self.arm}"
        return f"{base}/{self.variant}" if self.variant is not None else base

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["scope"] = self.scope.value
        value["key"] = self.key
        return value


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class AblationPlan:
    specs: tuple[AblationSpec, ...]
    tasks: tuple[AblationTask, ...]
    models: tuple[str, ...]
    seeds: tuple[int, ...]
    random_seed_offset: int

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": 1,
            "models": list(self.models),
            "seeds": list(self.seeds),
            "random_seed_offset": self.random_seed_offset,
            "specs": [
                {
                    **asdict(spec),
                    "scope": spec.scope.value,
                    "variants": list(spec.variants),
                }
                for spec in self.specs
            ],
            "tasks": [task.to_dict() for task in self.tasks],
        }

    def task(self, arm: str, seed: int, *, model: str | None = None,
             variant: str | int | None = None) -> AblationTask:
        matches = [
            task for task in self.tasks
            if task.arm == arm and task.seed == int(seed) and task.model == model and task.variant == variant
        ]
        if len(matches) != 1:
            raise KeyError(f"Ablation task is absent or ambiguous: {arm}/{model}/{seed}/{variant}")
        return matches[0]

    def summary(self) -> dict[str, Any]:
        return {
            "arm_count": len(self.specs),
            "task_count": len(self.tasks),
            "classifier_fit_count": sum(task.requires_classifier_fit for task in self.tasks),
            "validation_lock_count": len(self.tasks),
            "shared_seed_fit_count": sum(
                task.requires_classifier_fit and task.scope is AblationScope.SEED_SHARED for task in self.tasks
            ),
            "model_seed_fit_count": sum(
                task.requires_classifier_fit and task.scope is AblationScope.MODEL_SEED for task in self.tasks
            ),
            "analytical_task_count": sum(
                task.scope is AblationScope.ANALYTICAL_MODEL_SEED for task in self.tasks
            ),
            "plan_sha256": self.sha256,
        }


@dataclass(frozen=True)
class AblationExecutionBatch:
    """A load-efficient group of tasks that never mixes validation and test."""

    phase: str
    seed: int
    tasks: tuple[AblationTask, ...]
    model: str | None = None

    @property
    def key(self) -> str:
        model = self.model if self.model is not None else "shared"
        return f"{self.phase}/{model}/seed_{self.seed}"


def build_ablation_validation_schedule(plan: AblationPlan) -> tuple[AblationExecutionBatch, ...]:
    """Return all validation work: 30 model-fit + 30 analytical batches."""

    batches: list[AblationExecutionBatch] = []
    for model in plan.models:
        for seed in plan.seeds:
            fit_tasks = tuple(
                task for task in plan.tasks
                if task.model == model and task.seed == seed
                and task.scope is AblationScope.MODEL_SEED
            )
            analytical_tasks = tuple(
                task for task in plan.tasks
                if task.model == model and task.seed == seed
                and task.scope is AblationScope.ANALYTICAL_MODEL_SEED
            )
            batches.append(AblationExecutionBatch("fit_and_lock_model", seed, fit_tasks, model))
            batches.append(AblationExecutionBatch("lock_analytical", seed, analytical_tasks, model))
    flattened = [task.key for batch in batches for task in batch.tasks]
    if len(flattened) != len(plan.tasks) or set(flattened) != {task.key for task in plan.tasks}:
        raise RuntimeError("Validation schedule does not cover every planned task exactly once")
    return tuple(batches)


def build_ablation_test_schedule(
    plan: AblationPlan, gate: VerifiedAblationTestGate
) -> tuple[AblationExecutionBatch, ...]:
    """Return test work only after a matching verified all-task gate exists."""

    if gate.plan_sha256 != plan.sha256:
        raise RuntimeError("Cannot schedule ablation test inference under a different plan gate")
    batches: list[AblationExecutionBatch] = []
    for model in plan.models:
        for seed in plan.seeds:
            tasks = tuple(
                task for task in plan.tasks
                if task.model == model and task.seed == seed
                and task.scope is not AblationScope.SEED_SHARED
            )
            batches.append(AblationExecutionBatch("evaluate_model_and_analytical", seed, tasks, model))
    flattened = [task.key for batch in batches for task in batch.tasks]
    if len(flattened) != len(plan.tasks) or set(flattened) != {task.key for task in plan.tasks}:
        raise RuntimeError("Test schedule does not cover every planned task exactly once")
    return tuple(batches)


def build_ablation_plan(cfg: Mapping[str, Any]) -> AblationPlan:
    """Validate all eleven configured arms and expand their nonduplicated work units."""

    experiments = cfg.get("ablations", {}).get("experiments")
    if not isinstance(experiments, list):
        raise ValueError("cfg.ablations.experiments must be a list")
    by_name: dict[str, Mapping[str, Any]] = {}
    for experiment in experiments:
        if not isinstance(experiment, Mapping) or not isinstance(experiment.get("name"), str):
            raise ValueError("Every ablation experiment must be a named mapping")
        name = str(experiment["name"])
        if name in by_name:
            raise ValueError(f"Duplicate ablation arm: {name}")
        by_name[name] = experiment
    if set(by_name) != set(EXPECTED_ARMS):
        raise ValueError(
            f"Ablation arms must be exactly {list(EXPECTED_ARMS)}; "
            f"missing={sorted(set(EXPECTED_ARMS)-set(by_name))}, extra={sorted(set(by_name)-set(EXPECTED_ARMS))}"
        )
    models = tuple(str(value) for value in cfg.get("models", ()))
    seeds = tuple(int(value) for value in cfg.get("split_seeds", ()))
    if not models or len(set(models)) != len(models) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Models and split seeds must be non-empty and unique")

    specs: list[AblationSpec] = []
    for name in EXPECTED_ARMS:
        configured = by_name[name]
        if name in EXPECTED_INPUTS:
            if configured.get("input") != EXPECTED_INPUTS[name]:
                raise ValueError(f"Unexpected input definition for {name}")
            scope = AblationScope.MODEL_SEED
            specs.append(AblationSpec(name, scope, EXPECTED_INPUTS[name]))
        else:
            variants = tuple(configured.get("variants", ()))
            if variants != EXPECTED_VARIANTS[name]:
                raise ValueError(f"Unexpected variants for {name}: {variants}")
            specs.append(
                AblationSpec(
                    name, AblationScope.ANALYTICAL_MODEL_SEED, None,
                    variants=variants, requires_classifier_fit=False,
                )
            )

    tasks: list[AblationTask] = []
    for spec in specs:
        if spec.scope is AblationScope.SEED_SHARED:
            tasks.extend(
                AblationTask(spec.name, spec.scope, seed, requires_classifier_fit=True)
                for seed in seeds
            )
        elif spec.scope is AblationScope.MODEL_SEED:
            tasks.extend(
                AblationTask(spec.name, spec.scope, seed, model=model, requires_classifier_fit=True)
                for model in models for seed in seeds
            )
        else:
            tasks.extend(
                AblationTask(
                    spec.name, spec.scope, seed, model=model, variant=variant,
                    requires_classifier_fit=False,
                )
                for model in models for seed in seeds for variant in spec.variants
            )
    keys = [task.key for task in tasks]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Expanded ablation task keys are not unique")
    offset = int(cfg.get("training", {}).get("seed_offsets", {}).get("classifier", 0))
    if offset <= 0:
        raise ValueError("A positive locked classifier seed offset is required")
    return AblationPlan(tuple(specs), tuple(tasks), models, seeds, offset)


@dataclass(frozen=True)
class VerifiedAblationTestGate:
    path: Path
    sha256: str
    plan_sha256: str
    config_sha256: str
    primary_gate_sha256: str
    validation_lock_sha256: Mapping[str, str]


def validation_lock_path(lock_root: str | Path, task: AblationTask) -> Path:
    return Path(lock_root).joinpath(*task.key.split("/"), "validation_lock.json")


def _validate_digest(value: str | None, name: str, *, optional: bool = False) -> None:
    if value is None and optional:
        return
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _immutable_json(path: Path, payload: Mapping[str, Any]) -> Path:
    content = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"Refusing to overwrite a different immutable artifact: {path}")
        return path
    _atomic_bytes(path, content)
    return path


def _validated_level_locks(level_locks: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    if not isinstance(level_locks, Mapping) or set(level_locks) != {"eye", "patient"}:
        raise ValueError("level_locks must contain exactly eye and patient")
    output: dict[str, dict[str, Any]] = {}
    for level in ("eye", "patient"):
        item = level_locks[level]
        if not isinstance(item, Mapping):
            raise ValueError(f"level_locks.{level} must be a mapping")
        status = item.get("status")
        threshold, temperature = item.get("threshold"), item.get("temperature")
        reason = item.get("unavailable_reason", item.get("reason"))
        if status in {"locked", "calibration_unavailable"}:
            if threshold is None or not np.isfinite(threshold) or not 0 <= float(threshold) <= 1:
                raise ValueError(f"Threshold-locked {level} level requires a finite threshold in [0,1]")
            calibration_status = item.get(
                "calibration_status", "locked" if status == "locked" else "unavailable"
            )
            if calibration_status == "locked":
                if temperature is None or not np.isfinite(temperature) or float(temperature) <= 0:
                    raise ValueError(f"Calibration-locked {level} level requires a positive temperature")
                if reason is not None:
                    raise ValueError(f"Fully locked {level} level cannot have an unavailable reason")
            elif calibration_status == "unavailable":
                if temperature is not None or not isinstance(reason, str) or not reason:
                    raise ValueError(
                        f"Calibration-unavailable {level} level requires no temperature and a reason"
                    )
                status = "calibration_unavailable"
            else:
                raise ValueError(f"Invalid {level} calibration_status")
            output[level] = {
                "status": status,
                "classification_threshold_status": "locked",
                "calibration_status": calibration_status,
                "threshold": float(threshold),
                "temperature": float(temperature) if temperature is not None else None,
                "threshold_probability_scale": item.get(
                    "threshold_probability_scale",
                    "temperature_scaled" if calibration_status == "locked"
                    else "raw_due_to_calibration_unavailable",
                ),
                "unavailable_reason": reason,
            }
        elif status == "unavailable":
            if threshold is not None or temperature is not None or not isinstance(reason, str) or not reason:
                raise ValueError(
                    f"Unavailable {level} level requires a reason and no threshold/temperature substitution"
                )
            output[level] = {
                "status": "unavailable",
                "classification_threshold_status": "unavailable",
                "calibration_status": "unavailable",
                "threshold": None,
                "temperature": None,
                "threshold_probability_scale": None,
                "unavailable_reason": reason,
            }
        else:
            raise ValueError(
                f"level_locks.{level}.status must be locked, calibration_unavailable, or unavailable"
            )
    return output


def write_ablation_validation_lock(
    lock_root: str | Path,
    task: AblationTask,
    *,
    plan_sha256: str,
    config_sha256: str,
    validation_source_sha256: str,
    level_locks: Mapping[str, Mapping[str, Any]],
    checkpoint_sha256: str | None = None,
    checkpoint_path: str | Path | None = None,
    validation_source_path: str | Path | None = None,
) -> Path:
    """Write one immutable validation decision; test information is not accepted."""

    for value, name in (
        (plan_sha256, "plan_sha256"),
        (config_sha256, "config_sha256"),
        (validation_source_sha256, "validation_source_sha256"),
    ):
        _validate_digest(value, name)
    _validate_digest(checkpoint_sha256, "checkpoint_sha256", optional=True)
    if (checkpoint_path is None) != (checkpoint_sha256 is None):
        # Legacy/test callers may attest a digest without a path; execution
        # callers always provide both so the gate can re-hash actual bytes.
        if checkpoint_path is not None:
            raise ValueError("checkpoint_path cannot be supplied without checkpoint_sha256")
    if checkpoint_path is not None:
        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.is_file() or _file_sha256(checkpoint_file) != checkpoint_sha256:
            raise ValueError("checkpoint_path bytes do not match checkpoint_sha256")
    if validation_source_path is not None:
        validation_file = Path(validation_source_path)
        if not validation_file.is_file() or _file_sha256(validation_file) != validation_source_sha256:
            raise ValueError("validation_source_path bytes do not match validation_source_sha256")
    levels = _validated_level_locks(level_locks)
    if (task.requires_classifier_fit and checkpoint_sha256 is None
            and any(item["classification_threshold_status"] == "locked" for item in levels.values())):
        raise ValueError("A fitted ablation with an evaluable level requires a checkpoint digest")
    if not task.requires_classifier_fit and checkpoint_sha256 is not None:
        raise ValueError("Analytical ablations reuse primary predictions and cannot have a checkpoint")
    payload = {
        "schema": 1,
        "partition_used_for_lock": "validation",
        "task": task.to_dict(),
        "plan_sha256": plan_sha256,
        "config_sha256": config_sha256,
        "validation_source_sha256": validation_source_sha256,
        "validation_source_path": str(validation_source_path) if validation_source_path is not None else None,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "level_locks": levels,
        "test_statistics_used": False,
    }
    return _immutable_json(validation_lock_path(lock_root, task), payload)


def verify_ablation_validation_lock(
    path: str | Path,
    task: AblationTask,
    *,
    plan_sha256: str,
    config_sha256: str,
) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise RuntimeError(f"Missing ablation validation lock: {source}")
    value = json.loads(source.read_text(encoding="utf-8"))
    if value.get("schema") != 1 or value.get("partition_used_for_lock") != "validation":
        raise RuntimeError(f"Invalid ablation validation lock schema/partition: {source}")
    if value.get("task") != task.to_dict() or value.get("plan_sha256") != plan_sha256:
        raise RuntimeError(f"Ablation task/plan changed after validation lock: {source}")
    if value.get("config_sha256") != config_sha256 or value.get("test_statistics_used") is not False:
        raise RuntimeError(f"Configuration or no-test lock invariant failed: {source}")
    _validate_digest(value.get("validation_source_sha256"), "validation_source_sha256")
    _validate_digest(value.get("checkpoint_sha256"), "checkpoint_sha256", optional=True)
    checkpoint_path = value.get("checkpoint_path")
    if checkpoint_path is not None:
        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.is_file() or _file_sha256(checkpoint_file) != value.get("checkpoint_sha256"):
            raise RuntimeError(f"Ablation classifier checkpoint changed: {checkpoint_file}")
    validation_source_path = value.get("validation_source_path")
    if validation_source_path is not None:
        validation_file = Path(validation_source_path)
        if not validation_file.is_file() or _file_sha256(validation_file) != value.get("validation_source_sha256"):
            raise RuntimeError(f"Ablation validation source changed: {validation_file}")
    if not task.requires_classifier_fit and value.get("checkpoint_sha256") is not None:
        raise RuntimeError(f"Analytical ablation lock unexpectedly names a checkpoint: {source}")
    try:
        levels = _validated_level_locks(value.get("level_locks"))
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"Malformed eye/patient locks: {source}: {error}") from error
    if (task.requires_classifier_fit and value.get("checkpoint_sha256") is None
            and any(item["classification_threshold_status"] == "locked" for item in levels.values())):
        raise RuntimeError(f"Fitted ablation lock lacks checkpoint: {source}")
    return value


def collect_ablation_validation_locks(
    plan: AblationPlan, lock_root: str | Path, *, config_sha256: str
) -> dict[str, str]:
    """Require all trainable and analytical validation locks before any test arm."""

    hashes: dict[str, str] = {}
    missing: list[str] = []
    invalid: list[str] = []
    for task in plan.tasks:
        path = validation_lock_path(lock_root, task)
        if not path.is_file():
            missing.append(task.key)
            continue
        try:
            verify_ablation_validation_lock(
                path, task, plan_sha256=plan.sha256, config_sha256=config_sha256
            )
        except Exception as error:  # aggregate a complete, actionable gate report
            invalid.append(f"{task.key}: {error}")
        else:
            hashes[task.key] = _file_sha256(path)
    if missing or invalid:
        details = []
        if missing:
            details.append("missing=" + ", ".join(missing))
        if invalid:
            details.append("invalid=" + " | ".join(invalid))
        raise RuntimeError("Ablation test access denied; " + "; ".join(details))
    return dict(sorted(hashes.items()))


def _gate_payload(
    plan: AblationPlan,
    *,
    config_sha256: str,
    primary_gate_sha256: str,
    lock_hashes: Mapping[str, str],
) -> dict[str, Any]:
    _validate_digest(config_sha256, "config_sha256")
    _validate_digest(primary_gate_sha256, "primary_gate_sha256")
    if set(lock_hashes) != {task.key for task in plan.tasks}:
        raise RuntimeError("Global ablation gate requires exactly one validation lock per expanded task")
    for value in lock_hashes.values():
        _validate_digest(value, "validation_lock_sha256")
    return {
        "schema": 1,
        "scope": "all_ablation_test_arms_global",
        "plan_sha256": plan.sha256,
        "config_sha256": config_sha256,
        "primary_test_gate_sha256": primary_gate_sha256,
        "validation_lock_sha256": dict(sorted(lock_hashes.items())),
        "test_tuning_allowed": False,
    }


def open_ablation_test_gate(
    cfg: Mapping[str, Any],
    *,
    plan: AblationPlan | None = None,
    lock_root: str | Path | None = None,
) -> VerifiedAblationTestGate:
    """Open one global gate only after primary evaluation and every ablation lock."""

    from .protocol import assert_all_primary_evaluations, open_ablation_test_access, output_root

    selected_plan = plan or build_ablation_plan(cfg)
    runtime = cfg.get("_runtime", {})
    config_sha256 = runtime.get("canonical_config_sha256")
    _validate_digest(config_sha256, "cfg._runtime.canonical_config_sha256")
    assert_all_primary_evaluations(cfg)
    # The protocol-level ablation gate freezes the 480-lock global receipt and
    # the already immutable primary summary.  It is distinct from the earlier
    # primary-test gate and is the only authority accepted here.
    primary_gate = open_ablation_test_access(cfg)
    primary_gate_sha256 = _file_sha256(primary_gate)
    root = Path(lock_root) if lock_root is not None else output_root(cfg) / "ablations" / "validation_locks"
    hashes = collect_ablation_validation_locks(selected_plan, root, config_sha256=config_sha256)
    destination = output_root(cfg) / "ablations" / "state" / "test_access_opened.json"
    expected = _gate_payload(
        selected_plan,
        config_sha256=config_sha256,
        primary_gate_sha256=primary_gate_sha256,
        lock_hashes=hashes,
    )
    if destination.exists():
        observed = json.loads(destination.read_text(encoding="utf-8"))
        if observed != expected:
            raise RuntimeError("Ablation test gate changed after it was opened")
    else:
        _immutable_json(destination, expected)
    return VerifiedAblationTestGate(
        destination,
        _file_sha256(destination),
        selected_plan.sha256,
        config_sha256,
        primary_gate_sha256,
        hashes,
    )


def verify_ablation_test_gate(
    gate_path: str | Path,
    plan: AblationPlan,
    lock_root: str | Path,
    *,
    config_sha256: str,
    primary_gate_path: str | Path,
) -> VerifiedAblationTestGate:
    source = Path(gate_path)
    if not source.is_file() or not Path(primary_gate_path).is_file():
        raise RuntimeError("Ablation or primary global test gate is missing")
    hashes = collect_ablation_validation_locks(plan, lock_root, config_sha256=config_sha256)
    primary_hash = _file_sha256(primary_gate_path)
    expected = _gate_payload(
        plan, config_sha256=config_sha256,
        primary_gate_sha256=primary_hash, lock_hashes=hashes,
    )
    observed = json.loads(source.read_text(encoding="utf-8"))
    if observed != expected:
        raise RuntimeError("Ablation test gate, plan, primary gate, or validation locks changed")
    return VerifiedAblationTestGate(
        source, _file_sha256(source), plan.sha256, config_sha256, primary_hash, hashes
    )


def _assert_partition_access(
    partition: str, plan: AblationPlan, gate: VerifiedAblationTestGate | None
) -> None:
    if partition not in {"train", "validation", "test"}:
        raise ValueError("partition must be train, validation, or test")
    if partition == "test":
        if gate is None or gate.plan_sha256 != plan.sha256:
            raise RuntimeError("Test ablation input requires the matching verified global test gate")


def _image_tensor(image: Tensor) -> Tensor:
    if not torch.is_tensor(image) or image.ndim != 3 or image.shape[0] != 3 or not image.is_floating_point():
        raise ValueError("image must be a floating-point RGB CxHxW tensor")
    if not bool(torch.isfinite(image).all()) or bool(((image < 0) | (image > 1)).any()):
        raise ValueError("raw ablation image must be finite and in [0,1]")
    return image


def _binary_mask(mask: Any, shape: tuple[int, int], *, name: str) -> np.ndarray:
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    array = np.asarray(mask)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.shape != shape or not np.isfinite(array).all() or not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{name} must be a hard binary raster with shape {shape}")
    return array.astype(bool, copy=False)


def _neutral(image: Tensor, values: Sequence[float]) -> Tensor:
    result = torch.as_tensor(values, dtype=image.dtype, device=image.device).flatten()
    if result.numel() != image.shape[0]:
        raise ValueError("neutral requires one value per channel")
    return result.view(-1, 1, 1)


def _resize_geometry(source_hw: tuple[int, int], target_hw: tuple[int, int]) -> tuple[int, int, int, int]:
    source_h, source_w = source_hw
    target_h, target_w = target_hw
    scale = min(target_h / source_h, target_w / source_w)
    resized_h = max(1, min(target_h, int(round(source_h * scale))))
    resized_w = max(1, min(target_w, int(round(source_w * scale))))
    top, left = (target_h - resized_h) // 2, (target_w - resized_w) // 2
    return resized_h, resized_w, top, left


def _letterbox_image(
    image: Tensor,
    target_hw: tuple[int, int],
    fill: Sequence[float],
    *,
    audit_mask: np.ndarray | None = None,
) -> tuple[Tensor, Tensor]:
    resized_h, resized_w, top, left = _resize_geometry(tuple(image.shape[-2:]), target_hw)
    resized = F.interpolate(
        image.unsqueeze(0), (resized_h, resized_w), mode="bilinear",
        align_corners=False, antialias=True,
    )[0]
    neutral_value = _neutral(image, fill)
    output = neutral_value.expand(-1, *target_hw).clone()
    output[:, top:top + resized_h, left:left + resized_w] = resized
    output_mask = torch.zeros(target_hw, dtype=torch.bool, device=image.device)
    if audit_mask is None:
        output_mask[top:top + resized_h, left:left + resized_w] = True
    else:
        mask_tensor = torch.as_tensor(audit_mask, dtype=torch.float32, device=image.device)[None, None]
        resized_mask = F.interpolate(mask_tensor, (resized_h, resized_w), mode="nearest")[0, 0].bool()
        output_mask[top:top + resized_h, left:left + resized_w] = resized_mask
    return output, output_mask


def _largest_component(mask: np.ndarray, score: np.ndarray | None = None) -> np.ndarray | None:
    labels, count = ndimage.label(mask, structure=EIGHT_CONNECTED)
    if count == 0:
        return None
    candidates = []
    for label in range(1, count + 1):
        component = labels == label
        mean = float(score[component].mean()) if score is not None else 1.0
        candidates.append((-int(component.sum()), -mean, label, component))
    return min(candidates, key=lambda item: item[:3])[3]


def _stable_random_mask(
    source_mask: np.ndarray,
    *,
    seed: int,
    sample_key: str,
) -> tuple[np.ndarray, dict[str, int]]:
    y, x = np.nonzero(source_mask)
    y0, y1, x0, x1 = int(y.min()), int(y.max()) + 1, int(x.min()), int(x.max()) + 1
    template = source_mask[y0:y1, x0:x1]
    h, w = source_mask.shape
    th, tw = template.shape
    positions = (h - th + 1) * (w - tw + 1)
    if positions < 1:
        raise ValueError("ROI template does not fit its source image")
    digest = hashlib.sha256(f"{int(seed)}|{sample_key}".encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little", signed=False))
    best: tuple[int, int, int] | None = None
    attempts = min(4096, max(64, positions))
    for _ in range(attempts):
        top = int(rng.integers(0, h - th + 1))
        left = int(rng.integers(0, w - tw + 1))
        overlap = int(np.logical_and(source_mask[top:top + th, left:left + tw], template).sum())
        candidate = (overlap, top, left)
        if best is None or candidate < best:
            best = candidate
        if overlap == 0:
            break
    assert best is not None
    overlap, top, left = best
    result = np.zeros_like(source_mask, dtype=bool)
    result[top:top + th, left:left + tw] = template
    return result, {"random_top": top, "random_left": left, "source_overlap_pixels": overlap}


@dataclass
class AblationInput:
    arm: str
    valid: bool
    status: str
    tensor: Tensor | None
    classifier_mask: Tensor | None
    source_mask: np.ndarray | None
    geometry: Tensor | None
    roi_confidence: float | None
    classifier_mode: str
    metadata: dict[str, Any]


class AblationInputBuilder:
    """Deterministic one-frame builder bound to one planned task and partition."""

    def __init__(
        self,
        plan: AblationPlan,
        task: AblationTask,
        partition: str,
        *,
        roi_policy: ROIPolicy | None = None,
        test_gate: VerifiedAblationTestGate | None = None,
        target_size: int | Sequence[int] = (224, 224),
        neutral: Sequence[float] = DEFAULT_NEUTRAL_RGB,
    ):
        if task not in plan.tasks:
            raise ValueError("task does not belong to the supplied ablation plan")
        _assert_partition_access(partition, plan, test_gate)
        if task.scope is AblationScope.ANALYTICAL_MODEL_SEED:
            raise ValueError("Analytical tasks reuse probabilities and do not build image caches")
        if task.scope is AblationScope.MODEL_SEED and roi_policy is None:
            raise ValueError("Predicted-mask-dependent arm requires a locked ROI policy")
        if isinstance(target_size, int):
            self.target_hw = (target_size, target_size)
        else:
            self.target_hw = (int(target_size[0]), int(target_size[1]))
        if min(self.target_hw) < 1:
            raise ValueError("target_size must be positive")
        self.plan, self.task, self.partition = plan, task, partition
        self.roi_policy, self.test_gate = roi_policy, test_gate
        self.neutral = tuple(float(value) for value in neutral)

    def _invalid(self, status: str, metadata: Mapping[str, Any] | None = None) -> AblationInput:
        return AblationInput(
            self.task.arm, False, status, None, None, None, None, None,
            "none_abstain", dict(metadata or {}),
        )

    def build(
        self,
        image: Tensor,
        *,
        sample_key: str,
        predicted_probability: Any | None = None,
        predicted_hard_mask: Any | None = None,
        selected_mask: Any | None = None,
        predicted_status: str | ROIStatus | None = None,
        roi_confidence: float | None = None,
        gt_mask: Any | None = None,
    ) -> AblationInput:
        image = _image_tensor(image)
        if not isinstance(sample_key, str) or not sample_key:
            raise ValueError("A non-empty stable sample_key is required")
        shape = tuple(int(value) for value in image.shape[-2:])
        arm = self.task.arm

        if arm == "whole_image_baseline":
            tensor, visible = _letterbox_image(image, self.target_hw, self.neutral)
            return AblationInput(
                arm, True, ROIStatus.VALID.value, tensor, visible, None,
                torch.empty(0, dtype=image.dtype, device=image.device), None,
                "model_specific_family_image", {"shared_across_models": False, "source": "whole_image"},
            )

        if arm == "gt_roi_oracle":
            if gt_mask is None:
                raise ValueError("GT-ROI oracle requires a GT mask")
            gt = _binary_mask(gt_mask, shape, name="gt_mask")
            selected = _largest_component(gt)
            if selected is None:
                raise ValueError("GT-ROI oracle encountered an empty annotation")
            extraction = extract_roi_tensor(
                image, selected, self.target_hw, self.neutral, return_metadata=True
            )
            return AblationInput(
                arm, True, ROIStatus.VALID.value, extraction.tensor, extraction.output_mask,
                selected, extraction.geometry, 1.0, "model_specific_family_image",
                {"shared_across_models": True, "non_deployable_oracle": True},
            )

        assert self.roi_policy is not None
        probability: np.ndarray | None = None
        hard: np.ndarray | None = None
        if predicted_probability is not None:
            probability = np.asarray(
                predicted_probability.detach().cpu().numpy()
                if torch.is_tensor(predicted_probability) else predicted_probability,
                dtype=np.float32,
            ).squeeze()
            if probability.shape != shape:
                raise ValueError("predicted_probability shape does not match image")
        elif predicted_hard_mask is not None:
            # Disk-bounded execution may retain only the locked hard/selected
            # rasters plus their scalar confidence.  This is sufficient for
            # every image ablation and avoids persisting one float probability
            # map per arm.  The strict status must come from the original
            # probability-map post-processing; it is never recomputed from a
            # lossy binary raster.
            hard = _binary_mask(predicted_hard_mask, shape, name="predicted_hard_mask")
        else:
            raise ValueError(f"{arm} requires a predicted probability map or locked hard mask")

        if arm == "largest_component_without_quality_gates":
            if hard is None:
                assert probability is not None
                hard = hard_mask_from_probability(probability, self.roi_policy)
            selected = _largest_component(hard, probability)
            if selected is None:
                return self._invalid(ROIStatus.EMPTY.value, {"quality_gates_applied": False})
            confidence = (
                float(probability[selected].mean()) if probability is not None
                else float(roi_confidence) if roi_confidence is not None else None
            )
            extraction = extract_roi_tensor(
                image, selected, self.target_hw, self.neutral, return_metadata=True
            )
            return AblationInput(
                arm, True, ROIStatus.VALID.value, extraction.tensor, extraction.output_mask,
                selected, extraction.geometry, confidence, "model_specific_family_image",
                {"quality_gates_applied": False, "selection": "largest_8_connected_component"},
            )

        selected_result = None
        if probability is not None:
            selected_result = self.roi_policy.postprocess_probability(probability)
            if not selected_result.valid:
                return self._invalid(
                    selected_result.status.value,
                    {"quality_gates_applied": True, **selected_result.to_record()},
                )
            selected = selected_result.mask
            confidence = float(selected_result.selected_component.mean_probability)
            bbox_xyxy = selected_result.bbox_xyxy
        else:
            if predicted_status is None:
                raise ValueError("Locked hard-mask input requires predicted_status")
            status = predicted_status if isinstance(predicted_status, ROIStatus) else ROIStatus(str(predicted_status))
            if status is not ROIStatus.VALID:
                return self._invalid(
                    status.value,
                    {"quality_gates_applied": True, "locked_postprocess_status_reused": True},
                )
            if selected_mask is None:
                raise ValueError("A locked valid ROI requires selected_mask")
            selected = _binary_mask(selected_mask, shape, name="selected_mask")
            labels, component_count = ndimage.label(selected, structure=EIGHT_CONNECTED)
            if component_count != 1 or not np.any(selected):
                raise ValueError("Locked valid selected_mask must contain exactly one 8-connected component")
            assert hard is not None
            if np.any(selected & ~hard):
                raise ValueError("selected_mask must be a subset of predicted_hard_mask")
            y, x = np.nonzero(selected)
            bbox_xyxy = (int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1)
            confidence = float(roi_confidence) if roi_confidence is not None else None
            if confidence is not None and (not np.isfinite(confidence) or not 0 <= confidence <= 1):
                raise ValueError("roi_confidence must be a finite probability")
        geometry_array = roi_geometry(selected)
        geometry = torch.as_tensor(geometry_array, dtype=image.dtype, device=image.device)

        if arm in {"appearance_plus_geometry", "geometry_only"}:
            extraction = extract_roi_tensor(
                image, selected, self.target_hw, self.neutral, return_metadata=True
            )
            tensor = extraction.tensor
            mode = "model_specific_family_image_plus_geometry"
            if arm == "geometry_only":
                fill = _neutral(image, self.neutral)
                tensor = fill.expand(-1, *self.target_hw).clone()
                mode = "geometry_only"
            return AblationInput(
                arm, True, ROIStatus.VALID.value, tensor, extraction.output_mask,
                selected, geometry, confidence, mode, {"quality_gates_applied": True},
            )

        if arm == "bbox_context":
            assert bbox_xyxy is not None
            x0, y0, x1, y1 = bbox_xyxy
            crop = image[:, y0:y1, x0:x1]
            # ``classifier_mask`` means pixels that the classifier is allowed
            # to retain during post-cache augmentation.  This arm deliberately
            # exposes the *entire* bounding-box context, not only the selected
            # component.  ``source_mask`` below remains the selected component
            # for audit/provenance.
            tensor, classifier_mask = _letterbox_image(
                crop, self.target_hw, self.neutral
            )
            return AblationInput(
                arm, True, ROIStatus.VALID.value, tensor, classifier_mask, selected,
                geometry, confidence, "model_specific_family_image",
                {"quality_gates_applied": True, "unmasked_bbox_context": True},
            )

        if arm == "mask_only":
            mask_tensor = torch.as_tensor(selected, dtype=image.dtype, device=image.device)
            binary_rgb = mask_tensor.unsqueeze(0).expand(3, -1, -1)
            tensor, classifier_mask = _letterbox_image(
                binary_rgb, self.target_hw, (0.0, 0.0, 0.0), audit_mask=selected
            )
            return AblationInput(
                arm, True, ROIStatus.VALID.value, tensor, classifier_mask, selected,
                geometry, confidence, "model_specific_family_image", {"binary_mask_only": True},
            )

        if arm == "background_only_negative_control":
            selected_tensor = torch.as_tensor(selected, dtype=torch.bool, device=image.device)
            background = torch.where(selected_tensor.unsqueeze(0), _neutral(image, self.neutral), image)
            tensor, classifier_mask = _letterbox_image(
                background, self.target_hw, self.neutral, audit_mask=~selected
            )
            # Re-neutralize the resized hard ROI so interpolation cannot leave a
            # classifier-visible trace inside the excluded region.
            tensor = torch.where(classifier_mask.unsqueeze(0), tensor, _neutral(tensor, self.neutral))
            return AblationInput(
                arm, True, ROIStatus.VALID.value, tensor, classifier_mask, selected,
                geometry, confidence, "model_specific_family_image", {"roi_pixels_removed": True},
            )

        if arm == "random_roi_negative_control":
            random_mask, random_metadata = _stable_random_mask(
                selected,
                seed=self.task.seed + self.plan.random_seed_offset,
                sample_key=f"{self.task.model}|{sample_key}",
            )
            extraction = extract_roi_tensor(
                image, random_mask, self.target_hw, self.neutral, return_metadata=True
            )
            return AblationInput(
                arm, True, ROIStatus.VALID.value, extraction.tensor, extraction.output_mask,
                random_mask, extraction.geometry, confidence, "model_specific_family_image",
                {
                    "patient_label_independent_rng": True,
                    "size_matched_area_pixels": int(random_mask.sum()),
                    **random_metadata,
                },
            )
        raise KeyError(f"No input builder is defined for arm {arm}")


def ablation_cache_path(
    cache_root: str | Path, task: AblationTask, partition: str, sample_key: str
) -> Path:
    if partition not in {"train", "validation", "test"}:
        raise ValueError("partition must be train, validation, or test")
    digest = hashlib.sha256(sample_key.encode("utf-8")).hexdigest()
    return Path(cache_root).joinpath(*task.key.split("/"), partition, f"{digest}.npz")


def _deterministic_npz(arrays: Mapping[str, np.ndarray]) -> bytes:
    """Create byte-stable NPZ (fixed member order, timestamp and attributes)."""

    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            member = io.BytesIO()
            np.lib.format.write_array(member, np.asanyarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, member.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return output.getvalue()


def write_ablation_cache(
    cache_root: str | Path,
    task: AblationTask,
    partition: str,
    sample_key: str,
    item: AblationInput,
) -> dict[str, Any]:
    """Write one valid deterministic cache or return an explicit abstention record."""

    if item.arm != task.arm:
        raise ValueError("Ablation input arm does not match task")
    frame_identity_sha256 = hashlib.sha256(sample_key.encode("utf-8")).hexdigest()
    base = {
        "arm": task.arm,
        "scope": task.scope.value,
        "model": task.model,
        "seed": task.seed,
        "variant": task.variant,
        "sample_key": sample_key,
        "frame_identity_sha256": frame_identity_sha256,
        "roi_valid": bool(item.valid),
        "abstention_reason": "" if item.valid else item.status,
        "classifier_mode": item.classifier_mode,
    }
    if not item.valid:
        if item.tensor is not None:
            raise ValueError("Invalid ablation input must not carry a classifier tensor")
        return {**base, "cache_path": None, "cache_sha256": None}
    if item.tensor is None or item.classifier_mask is None or item.geometry is None:
        raise ValueError("Valid ablation input is missing tensor/mask/geometry")
    destination = ablation_cache_path(cache_root, task, partition, sample_key)
    metadata = {**item.metadata, "status": item.status, "roi_confidence": item.roi_confidence}
    content = _deterministic_npz(
        {
            "frame_identity_sha256": np.frombuffer(
                bytes.fromhex(frame_identity_sha256), dtype=np.uint8
            ),
            "geometry": item.geometry.detach().cpu().numpy().astype(np.float32, copy=False),
            "image": item.tensor.detach().cpu().numpy().astype(np.float32, copy=False),
            "mask": item.classifier_mask.detach().cpu().numpy().astype(np.uint8, copy=False),
            "metadata_json": np.asarray(
                json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
            ),
        }
    )
    if destination.exists() and destination.read_bytes() != content:
        raise RuntimeError(f"Refusing to overwrite a different ablation cache: {destination}")
    if not destination.exists():
        _atomic_bytes(destination, content)
    return {**base, "cache_path": str(destination), "cache_sha256": _file_sha256(destination)}


def aggregate_frames_by_rule(
    frames: pd.DataFrame,
    *,
    rule: str,
    min_valid_frames: int,
    threshold: float = 0.5,
    frames_per_eye: int = 7,
    confidence_column: str = "roi_confidence",
    localisation_minimum_frames: int = 4,
) -> pd.DataFrame:
    """Analytically aggregate fixed frame probabilities without refitting a classifier."""

    allowed = set(EXPECTED_VARIANTS["aggregation_rule"])
    if rule not in allowed:
        raise ValueError(f"rule must be one of {sorted(allowed)}")
    if not 1 <= int(min_valid_frames) <= frames_per_eye:
        raise ValueError("min_valid_frames must lie within the fixed frames-per-eye count")
    if not 1 <= int(localisation_minimum_frames) <= frames_per_eye:
        raise ValueError("localisation_minimum_frames must lie within the fixed frames-per-eye count")
    required = {"patient_id", "case_id", "side", "frame_id", "label", "probability", "roi_valid"}
    if not required <= set(frames):
        raise ValueError(f"Missing frame columns: {sorted(required-set(frames))}")
    rows: list[dict[str, Any]] = []
    for (patient_key, case_id, side_key), group in frames.groupby(
        ["patient_id", "case_id", "side"], sort=True
    ):
        if len(group) != frames_per_eye or group.frame_id.nunique() != frames_per_eye:
            raise ValueError(f"Eye {case_id!r} must have exactly {frames_per_eye} unique frames")
        for column in ("patient_id", "side", "label"):
            if group[column].nunique(dropna=False) != 1:
                raise ValueError(f"Inconsistent {column} within eye {case_id!r}")
        probability = group.probability.to_numpy(float)
        observed_valid = group.roi_valid.to_numpy(dtype=object, copy=False)
        if any(not isinstance(value, (bool, np.bool_)) for value in observed_valid):
            raise ValueError("roi_valid must contain only real boolean values")
        valid = np.asarray([bool(value) for value in observed_valid], dtype=bool)
        if np.any(valid & ~np.isfinite(probability)) or np.any(~valid & np.isfinite(probability)):
            raise ValueError("Valid frames require probabilities and invalid frames must not carry fallback values")
        if np.any(valid & ((probability < 0) | (probability > 1))):
            raise ValueError("Frame probabilities must lie in [0,1]")
        n_valid = int(valid.sum())
        evaluable = n_valid >= int(min_valid_frames)
        value = np.nan
        if evaluable:
            selected = probability[valid]
            if rule == "mean":
                value = float(selected.mean())
            elif rule == "median":
                value = float(np.median(selected))
            elif rule == "maximum":
                value = float(selected.max())
            else:
                if confidence_column not in group:
                    raise ValueError(f"Weighted aggregation requires {confidence_column}")
                weights = group.loc[valid, confidence_column].to_numpy(float)
                if (not np.isfinite(weights).all() or np.any(weights < 0) or np.any(weights > 1)
                        or float(weights.sum()) <= 0):
                    raise ValueError("Valid ROI confidence weights must be finite probabilities with nonzero sum")
                value = float(np.average(selected, weights=weights))
        row = {
            "case_id": case_id,
            "patient_id": group.patient_id.iloc[0],
            "side": group.side.iloc[0],
            "label": int(group.label.iloc[0]),
            "n_frames": frames_per_eye,
            "n_valid_frames": n_valid,
            "evaluable": bool(evaluable),
            "probability": value,
            "prediction": int(value >= threshold) if evaluable else -1,
            "abstention_reason": "" if evaluable else "insufficient_valid_frames",
            "aggregation_rule": rule,
            "minimum_valid_frames": int(min_valid_frames),
        }
        if (
            str(row["patient_id"]) != str(patient_key)
            or str(row["side"]) != str(side_key)
        ):
            raise ValueError("Eye grouping identity changed during ablation aggregation")
        for column in ("label_3class", "split", "seed", "model"):
            if column in group:
                if group[column].nunique(dropna=False) != 1:
                    raise ValueError(f"Inconsistent {column} within eye {case_id!r}")
                row[column] = group[column].iloc[0]
        if "roi_hit" in group:
            hits = group.roi_hit.fillna(0).to_numpy(bool) & valid
            row["roi_hit_frames"] = int(hits.sum())
            # This retrospective endpoint stays at its preregistered 4/7
            # criterion even in the minimum-valid-frames sensitivity analysis.
            row["localized"] = bool(hits.sum() >= int(localisation_minimum_frames))
        rows.append(row)
    return pd.DataFrame(rows)


def build_analytical_eye_variants(
    frames: pd.DataFrame,
    cfg: Mapping[str, Any],
    *,
    threshold: float = 0.5,
    partition: str = "validation",
    plan: AblationPlan | None = None,
    test_gate: VerifiedAblationTestGate | None = None,
) -> dict[str, pd.DataFrame]:
    """Build all seven predeclared analytical variants from fixed frame outputs."""

    selected_plan = plan or build_ablation_plan(cfg)
    _assert_partition_access(partition, selected_plan, test_gate)
    experiments = {item["name"]: item for item in cfg["ablations"]["experiments"]}
    aggregation_rules = tuple(experiments["aggregation_rule"]["variants"])
    minimum_counts = tuple(int(value) for value in experiments["minimum_valid_frames"]["variants"])
    primary_minimum = int(cfg["aggregation"]["minimum_valid_frames"])
    localisation_minimum = int(cfg["evaluation"]["localized_success"]["minimum_hit_frames"])
    result = {
        f"aggregation_rule/{rule}": aggregate_frames_by_rule(
            frames, rule=rule, min_valid_frames=primary_minimum, threshold=threshold,
            localisation_minimum_frames=localisation_minimum,
        )
        for rule in aggregation_rules
    }
    result.update(
        {
            f"minimum_valid_frames/{count}": aggregate_frames_by_rule(
                frames, rule="mean", min_valid_frames=count, threshold=threshold,
                localisation_minimum_frames=localisation_minimum,
            )
            for count in minimum_counts
        }
    )
    return result


__all__ = [
    "ANALYTICAL_ARMS",
    "EXPECTED_ARMS",
    "MODEL_SEED_ARMS",
    "PREDICTED_MASK_ARMS",
    "SHARED_SEED_ARMS",
    "AblationInput",
    "AblationInputBuilder",
    "AblationExecutionBatch",
    "AblationPlan",
    "AblationScope",
    "AblationSpec",
    "AblationTask",
    "VerifiedAblationTestGate",
    "ablation_cache_path",
    "aggregate_frames_by_rule",
    "build_ablation_plan",
    "build_ablation_test_schedule",
    "build_ablation_validation_schedule",
    "build_analytical_eye_variants",
    "collect_ablation_validation_locks",
    "open_ablation_test_gate",
    "validation_lock_path",
    "verify_ablation_test_gate",
    "verify_ablation_validation_lock",
    "write_ablation_cache",
    "write_ablation_validation_lock",
]
