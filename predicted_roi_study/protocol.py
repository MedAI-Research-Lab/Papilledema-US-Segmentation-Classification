"""Immutable workflow receipts and test-access gates for the strict ROI study."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import shutil
import socket
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .config import (
    EXPECTED_MODELS,
    EXPECTED_SEEDS,
    PROJECT_ROOT,
    SHA256_RE,
    canonical_sha256,
    config_without_runtime,
    resolve_project_path,
    sha256_file,
    validate_config,
    verify_locked_sources,
)


PROTOCOL_PATH = Path(__file__).with_name("PROTOCOL.md")
STAGES = (
    "prepare",
    "preflight",
    "train-segmenters",
    "build-rois",
    "train-classifiers",
    "lock",
    "evaluate",
    "summarize",
    "lock-ablations",
    "evaluate-ablations",
)
PER_MODEL_STAGES = {"preflight"}
PER_UNIT_STAGES = {
    "train-segmenters",
    "build-rois",
    "train-classifiers",
    "lock",
    "evaluate",
}
GLOBAL_STAGES = {"prepare", "summarize", "lock-ablations", "evaluate-ablations"}


class ProtocolGateError(RuntimeError):
    """Raised when an immutable workflow prerequisite is absent or stale."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def save_json_atomic(path: str | Path, value: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def output_root(cfg: Mapping[str, Any]) -> Path:
    return resolve_project_path(str(cfg["output"]))


def unit_root(cfg: Mapping[str, Any], model: str, seed: int) -> Path:
    _validate_unit(cfg, model, seed)
    return output_root(cfg) / "runs" / model / f"seed_{seed}"


def split_path(cfg: Mapping[str, Any], seed: int) -> Path:
    _validate_seed(cfg, seed)
    return output_root(cfg) / "splits" / f"seed_{seed}_patients.csv"


def protocol_sha256() -> str:
    if not PROTOCOL_PATH.is_file():
        raise ProtocolGateError(f"Protocol document is missing: {PROTOCOL_PATH}")
    return sha256_file(PROTOCOL_PATH)


def code_fingerprint() -> dict[str, Any]:
    """Fingerprint new code plus every imported legacy segmenter source file."""

    package = Path(__file__).resolve().parent
    legacy_models = PROJECT_ROOT / "binary_study" / "models"
    discovered = {
        path
        for root, suffixes in (
            (package, {".py", ".json", ".md"}),
            (legacy_models, {".py", ".json", ".yaml", ".yml"}),
        )
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and ".pytest_cache" not in path.parts
        and path.suffix.lower() in suffixes
        and "weights" not in path.parts
    }
    clean_launcher = PROJECT_ROOT / "scripts" / "run_strict_roi_clean_seed.ps1"
    if not clean_launcher.is_file():
        raise ProtocolGateError(f"Clean-seed launcher is missing: {clean_launcher}")
    discovered.add(clean_launcher)
    files = sorted(discovered)
    records = {str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256_file(path) for path in files}
    return {"files": records, "sha256": canonical_sha256(records)}


def _save_code_snapshot_atomic(destination: Path) -> Path:
    """Save the exact fingerprinted sources in a deterministic ZIP archive."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        fingerprint = code_fingerprint()
        with zipfile.ZipFile(
            temporary, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for relative in sorted(fingerprint["files"]):
                source = resolve_project_path(relative, must_exist=True)
                info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, source.read_bytes(), compress_type=zipfile.ZIP_DEFLATED)
        if destination.exists():
            if destination.read_bytes() != temporary.read_bytes():
                raise ProtocolGateError("Refusing to replace a different code source snapshot")
            return destination
        os.replace(temporary, destination)
        return destination
    finally:
        if temporary.exists():
            temporary.unlink()


def stage_receipt_path(
    cfg: Mapping[str, Any], stage: str, *, model: str | None = None, seed: int | None = None
) -> Path:
    _validate_stage_scope(cfg, stage, model, seed)
    base = output_root(cfg) / "state" / "receipts"
    if stage in GLOBAL_STAGES:
        return base / f"{stage}.json"
    if stage in PER_MODEL_STAGES:
        return base / stage / f"{model}.json"
    return base / stage / str(model) / f"seed_{seed}.json"


def _validate_model(cfg: Mapping[str, Any], model: str) -> None:
    if model not in cfg["models"] or model not in EXPECTED_MODELS:
        raise ProtocolGateError(f"Model is not predeclared: {model}")


def _validate_seed(cfg: Mapping[str, Any], seed: int) -> None:
    if seed not in cfg["split_seeds"] or seed not in EXPECTED_SEEDS:
        raise ProtocolGateError(f"Seed is not predeclared: {seed}")


def _validate_unit(cfg: Mapping[str, Any], model: str, seed: int) -> None:
    _validate_model(cfg, model)
    _validate_seed(cfg, seed)


def _validate_stage_scope(
    cfg: Mapping[str, Any], stage: str, model: str | None, seed: int | None
) -> None:
    if stage not in STAGES:
        raise ProtocolGateError(f"Unknown workflow stage: {stage}")
    if stage in GLOBAL_STAGES:
        if model is not None or seed is not None:
            raise ProtocolGateError(f"{stage} is global and accepts neither model nor seed")
    elif stage in PER_MODEL_STAGES:
        if model is None or seed is not None:
            raise ProtocolGateError(f"{stage} requires model and forbids seed")
        _validate_model(cfg, model)
    else:
        if model is None or seed is None:
            raise ProtocolGateError(f"{stage} requires both model and seed")
        _validate_unit(cfg, model, seed)


def _runtime_context(cfg: Mapping[str, Any]) -> dict[str, Any]:
    runtime = cfg.get("_runtime")
    if not isinstance(runtime, Mapping) or "config_sha256" not in runtime:
        raise ProtocolGateError("Configuration must be loaded through predicted_roi_study.config.load_config")
    fingerprint = code_fingerprint()
    return {
        "study_id": cfg["study_id"],
        "protocol_version": cfg["protocol_version"],
        "config_sha256": runtime["config_sha256"],
        "canonical_config_sha256": runtime["canonical_config_sha256"],
        "protocol_sha256": protocol_sha256(),
        "code_sha256": fingerprint["sha256"],
        "manifest_sha256": cfg["dataset"]["manifest_sha256"],
    }


def _artifact_record(path: str | Path) -> dict[str, Any]:
    artifact = Path(path)
    if not artifact.is_absolute():
        artifact = resolve_project_path(artifact)
    artifact = artifact.resolve()
    try:
        relative = artifact.relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        raise ProtocolGateError(f"Receipt artifact is outside the workspace: {artifact}") from error
    if not artifact.is_file():
        raise ProtocolGateError(f"Receipt artifact does not exist or is not a file: {artifact}")
    return {
        "path": str(relative).replace("\\", "/"),
        "bytes": artifact.stat().st_size,
        "sha256": sha256_file(artifact),
    }


_CLASSIFIER_STRATEGIES = ("model_specific", "standardized_resnet18")


def _artifact_record_index(
    records: Sequence[Mapping[str, Any]]
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for record in records:
        path = record.get("path")
        if not isinstance(path, str) or path in result:
            raise ProtocolGateError("Receipt artifact paths must be unique strings")
        result[path] = record
    return result


def _require_attested_reference(
    index: Mapping[str, Mapping[str, Any]], path: Any, digest: Any, label: str,
    *, optional: bool = False,
) -> None:
    if path is None and optional:
        if digest is not None:
            raise ProtocolGateError(f"{label} has a digest without a path")
        return
    if not isinstance(path, str) or path not in index:
        raise ProtocolGateError(f"{label} is not an attested receipt artifact")
    if not isinstance(digest, str) or index[path].get("sha256") != digest:
        raise ProtocolGateError(f"{label} digest disagrees with the receipt artifact")


def _artifact_json_by_suffix(
    records: Sequence[Mapping[str, Any]], suffix: str, label: str
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    normalized = suffix.replace("\\", "/")
    matches = [record for record in records if str(record.get("path", "")).endswith(normalized)]
    if len(matches) != 1:
        raise ProtocolGateError(f"Receipt requires exactly one {label} artifact")
    path = resolve_project_path(str(matches[0]["path"]), must_exist=True)
    value = read_json(path)
    return value, matches[0]


def _validate_dual_training_receipt(
    cfg: Mapping[str, Any], metadata: Mapping[str, Any], records: Sequence[Mapping[str, Any]],
    *, model: str | None, seed: int | None,
) -> None:
    if metadata.get("primary_classifier_strategy") != _CLASSIFIER_STRATEGIES[0]:
        raise ProtocolGateError("train-classifiers primary must be model_specific")
    if metadata.get("classifier_strategies") != list(_CLASSIFIER_STRATEGIES):
        raise ProtocolGateError("train-classifiers requires the two locked strategies in order")
    if metadata.get("classifier_strategy_count") != 2:
        raise ProtocolGateError("train-classifiers requires classifier_strategy_count=2")
    if metadata.get("identical_oof_predicted_roi_inputs") is not True:
        raise ProtocolGateError("Both classifier strategies must use identical ROI inputs")
    if metadata.get("identical_training_optimization_index") is not True:
        raise ProtocolGateError("Both classifier strategies must use one optimization index")
    if metadata.get("identical_training_eye_eligibility_ledger") is not True:
        raise ProtocolGateError("Both classifier strategies must use one eye eligibility ledger")
    summaries = metadata.get("strategies")
    if not isinstance(summaries, Mapping) or set(summaries) != set(_CLASSIFIER_STRATEGIES):
        raise ProtocolGateError("train-classifiers metadata requires exactly both strategy summaries")
    manifest, _ = _artifact_json_by_suffix(
        records,
        f"runs/{model}/seed_{seed}/classifiers/training_manifest.json",
        "dual classifier training manifest",
    )
    if (
        manifest.get("schema") != 2
        or manifest.get("model") != model
        or manifest.get("seed") != seed
        or
        manifest.get("primary_classifier_strategy") != _CLASSIFIER_STRATEGIES[0]
        or manifest.get("secondary_classifier_strategy") != _CLASSIFIER_STRATEGIES[1]
        or manifest.get("identical_oof_predicted_roi_inputs") is not True
        or manifest.get("identical_training_optimization_index") is not True
        or manifest.get("identical_training_eye_eligibility_ledger") is not True
        or set(manifest.get("strategies", {})) != set(_CLASSIFIER_STRATEGIES)
    ):
        raise ProtocolGateError("Malformed dual classifier training manifest")
    index = _artifact_record_index(records)
    strategy_specs = {
        str(cfg["classifier"]["primary"]["name"]): cfg["classifier"]["primary"],
        str(cfg["classifier"]["secondary"]["name"]): cfg["classifier"]["secondary"],
    }
    common_train_hash = manifest.get("training_roi_index_sha256")
    common_validation_hash = manifest.get("validation_roi_index_sha256")
    common_optimization_hash = manifest.get("training_optimization_index_sha256")
    common_eligibility_hash = manifest.get("training_eye_eligibility_sha256")
    _require_attested_reference(
        index,
        manifest.get("training_roi_index"),
        common_train_hash,
        "common OOF training ROI index",
    )
    _require_attested_reference(
        index,
        manifest.get("validation_roi_index"),
        common_validation_hash,
        "common validation ROI index",
    )
    _require_attested_reference(
        index,
        manifest.get("training_optimization_index"),
        common_optimization_hash,
        "common classifier optimization index",
    )
    _require_attested_reference(
        index,
        manifest.get("training_eye_eligibility"),
        common_eligibility_hash,
        "common classifier eye eligibility ledger",
    )
    if (
        metadata.get("training_optimization_index_sha256") != common_optimization_hash
        or metadata.get("training_eye_eligibility_sha256") != common_eligibility_hash
    ):
        raise ProtocolGateError("Training receipt metadata disagrees with shared selection hashes")
    if (
        manifest.get("frames_per_eye") != int(cfg["dataset"]["frames_per_eye"])
        or manifest.get("minimum_valid_frames")
        != int(cfg["aggregation"]["minimum_valid_frames"])
    ):
        raise ProtocolGateError("Classifier training selection policy differs from config")
    for field in (
        "training_roi_dataframe_sha256",
        "training_optimization_dataframe_sha256",
        "training_eye_eligibility_dataframe_sha256",
    ):
        digest = manifest.get(field)
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            raise ProtocolGateError(f"Classifier training manifest lacks {field}")

    optimization_path = resolve_project_path(
        str(manifest.get("training_optimization_index")), must_exist=True
    )
    eligibility_path = resolve_project_path(
        str(manifest.get("training_eye_eligibility")), must_exist=True
    )
    optimization_fields, optimization_rows = _read_csv(optimization_path)
    eligibility_fields, eligibility_rows = _read_csv(eligibility_path)
    eye_columns = ("patient_id", "case_id", "side")
    if not {*eye_columns, "frame_id", "roi_valid", "cache_path"} <= set(
        optimization_fields
    ):
        raise ProtocolGateError("Classifier optimization index schema is incomplete")
    required_eligibility = {
        *eye_columns,
        "label",
        "total_frames",
        "valid_roi_frames",
        "minimum_required_frames",
        "eye_training_eligible",
        "optimization_frame_count",
        "exclusion_reason",
    }
    if not required_eligibility <= set(eligibility_fields):
        raise ProtocolGateError("Classifier eye eligibility ledger schema is incomplete")
    if any(
        row.get("roi_valid") != "True" or not row.get("cache_path")
        for row in optimization_rows
    ):
        raise ProtocolGateError("Optimization index contains an invalid or cache-free ROI")
    optimization_counts = Counter(
        tuple(row[column] for column in eye_columns) for row in optimization_rows
    )
    ledger_keys: set[tuple[str, str, str]] = set()
    eligible_eye_count = 0
    ineligible_eye_count = 0
    ledger_optimization_count = 0
    for row in eligibility_rows:
        key = tuple(row[column] for column in eye_columns)
        if key in ledger_keys:
            raise ProtocolGateError("Classifier eye eligibility ledger repeats an eye")
        ledger_keys.add(key)
        try:
            total_frames = int(row["total_frames"])
            valid_frames = int(row["valid_roi_frames"])
            minimum = int(row["minimum_required_frames"])
            selected_frames = int(row["optimization_frame_count"])
        except (TypeError, ValueError) as error:
            raise ProtocolGateError("Classifier eye eligibility counts are malformed") from error
        if (
            total_frames != int(cfg["dataset"]["frames_per_eye"])
            or minimum != int(cfg["aggregation"]["minimum_valid_frames"])
            or not 0 <= valid_frames <= total_frames
        ):
            raise ProtocolGateError("Classifier eye eligibility counts violate the 4/7 policy")
        flag = row["eye_training_eligible"]
        expected_eligible = valid_frames >= minimum
        if flag not in {"True", "False"} or (flag == "True") != expected_eligible:
            raise ProtocolGateError("Classifier eye eligibility flag is inconsistent")
        expected_selected = valid_frames if expected_eligible else 0
        if selected_frames != expected_selected or optimization_counts.get(key, 0) != expected_selected:
            raise ProtocolGateError("Eligibility ledger and optimization rows disagree")
        expected_reason = (
            f"eligible_ge_{minimum}_of_{total_frames}"
            if expected_eligible
            else "no_valid_predicted_roi"
            if valid_frames == 0
            else f"below_{minimum}_of_{total_frames}"
        )
        if row["exclusion_reason"] != expected_reason:
            raise ProtocolGateError("Classifier eye eligibility reason is inconsistent")
        eligible_eye_count += int(expected_eligible)
        ineligible_eye_count += int(not expected_eligible)
        ledger_optimization_count += selected_frames
    if set(optimization_counts) - ledger_keys:
        raise ProtocolGateError("Optimization index contains an eye absent from its ledger")
    if (
        len(optimization_rows) != ledger_optimization_count
        or manifest.get("optimization_frame_count") != ledger_optimization_count
        or manifest.get("eligible_training_eye_count") != eligible_eye_count
        or manifest.get("ineligible_training_eye_count") != ineligible_eye_count
    ):
        raise ProtocolGateError("Classifier training selection summary counts are inconsistent")
    for strategy in _CLASSIFIER_STRATEGIES:
        expected_prefix = f"runs/{model}/seed_{seed}/classifiers/{strategy}/"
        result, result_record = _artifact_json_by_suffix(
            records,
            f"runs/{model}/seed_{seed}/classifiers/{strategy}/training_result.json",
            f"{strategy} training result",
        )
        if (
            result.get("model") != model
            or result.get("seed") != seed
            or result.get("classifier_strategy") != strategy
        ):
            raise ProtocolGateError(f"{strategy} training result identity mismatch")
        if result.get("classifier_family") != model:
            raise ProtocolGateError(f"{strategy} training family/receipt scope mismatch")
        expected_spec = strategy_specs[strategy]
        if (
            result.get("classifier_role") != expected_spec.get("role")
            or result.get("classifier_estimand_id") != expected_spec.get("estimand_id")
        ):
            raise ProtocolGateError(f"{strategy} training estimand/role mismatch")
        available = result.get("available")
        if not isinstance(available, bool) or available != (result.get("checkpoint") is not None):
            raise ProtocolGateError(f"{strategy} availability/checkpoint mismatch")
        if (
            result.get("training_roi_index_sha256") != common_train_hash
            or result.get("validation_roi_index_sha256") != common_validation_hash
            or result.get("training_optimization_index_sha256")
            != common_optimization_hash
            or result.get("training_eye_eligibility_sha256")
            != common_eligibility_hash
            or result.get("training_optimization_index")
            != manifest.get("training_optimization_index")
            or result.get("training_eye_eligibility")
            != manifest.get("training_eye_eligibility")
            or result.get("frames_per_eye") != manifest.get("frames_per_eye")
            or result.get("minimum_valid_frames")
            != manifest.get("minimum_valid_frames")
            or result.get("eligible_training_eye_count")
            != manifest.get("eligible_training_eye_count")
            or result.get("ineligible_training_eye_count")
            != manifest.get("ineligible_training_eye_count")
            or result.get("optimization_frame_count")
            != manifest.get("optimization_frame_count")
            or result.get("training_roi_dataframe_sha256")
            != manifest.get("training_roi_dataframe_sha256")
            or result.get("training_optimization_dataframe_sha256")
            != manifest.get("training_optimization_dataframe_sha256")
            or result.get("training_eye_eligibility_dataframe_sha256")
            != manifest.get("training_eye_eligibility_dataframe_sha256")
        ):
            raise ProtocolGateError(
                "Classifier strategies did not use identical ROI/eligibility indexes"
            )
        _require_attested_reference(
            index, result.get("history"), result.get("history_sha256"), f"{strategy} history"
        )
        _require_attested_reference(
            index,
            result.get("validation_frames"),
            result.get("validation_frames_sha256"),
            f"{strategy} validation frames",
        )
        _require_attested_reference(
            index, result.get("model_info"), result.get("model_info_sha256"),
            f"{strategy} provenance",
        )
        model_info = read_json(resolve_project_path(result["model_info"], must_exist=True))
        if (
            model_info.get("model") != model
            or model_info.get("outer_seed") != seed
            or
            model_info.get("classifier_strategy") != strategy
            or model_info.get("classifier_family") != model
            or model_info.get("classifier_role") != expected_spec.get("role")
            or model_info.get("classifier_estimand_id") != expected_spec.get("estimand_id")
        ):
            raise ProtocolGateError(f"{strategy} provenance/receipt scope mismatch")
        model_selection = model_info.get("training_selection")
        if not isinstance(model_selection, Mapping) or any(
            model_selection.get(field) != result.get(field)
            for field in (
                "training_roi_index",
                "training_roi_index_sha256",
                "training_optimization_index",
                "training_optimization_index_sha256",
                "training_eye_eligibility",
                "training_eye_eligibility_sha256",
                "training_roi_dataframe_sha256",
                "training_optimization_dataframe_sha256",
                "training_eye_eligibility_dataframe_sha256",
                "frames_per_eye",
                "minimum_valid_frames",
                "eligible_training_eye_count",
                "ineligible_training_eye_count",
                "optimization_frame_count",
            )
        ):
            raise ProtocolGateError(f"{strategy} provenance lacks the shared selection contract")
        _require_attested_reference(
            index, result.get("checkpoint"), result.get("checkpoint_sha256"),
            f"{strategy} checkpoint", optional=True,
        )
        for field in ("history", "validation_frames", "model_info", "checkpoint"):
            path = result.get(field)
            if path is not None and expected_prefix not in str(path).replace("\\", "/"):
                raise ProtocolGateError(f"{strategy} {field} comes from another unit")
        for field, expected_suffix in (
            ("training_roi_index", f"runs/{model}/seed_{seed}/roi/train_oof_index.csv"),
            (
                "training_optimization_index",
                f"runs/{model}/seed_{seed}/classifiers/shared/training_optimization_index.csv",
            ),
            (
                "training_eye_eligibility",
                f"runs/{model}/seed_{seed}/classifiers/shared/training_eye_eligibility.csv",
            ),
            ("validation_roi_index", f"runs/{model}/seed_{seed}/roi/validation/index.csv"),
        ):
            if not str(result.get(field, "")).replace("\\", "/").endswith(expected_suffix):
                raise ProtocolGateError(f"{strategy} {field} comes from another unit")
        validation_path = resolve_project_path(result["validation_frames"], must_exist=True)
        with validation_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or "classifier_strategy" not in reader.fieldnames:
                raise ProtocolGateError(f"{strategy} validation frames lack classifier_strategy")
            if any(row.get("classifier_strategy") != strategy for row in reader):
                raise ProtocolGateError(f"{strategy} validation frames mix classifier strategies")
        branch = manifest["strategies"][strategy]
        if (
            branch.get("model") != model
            or branch.get("seed") != seed
            or branch.get("classifier_strategy") != strategy
            or branch.get("classifier_family") != model
            or branch.get("role") != expected_spec.get("role")
            or branch.get("estimand_id") != expected_spec.get("estimand_id")
            or
            branch.get("training_result_sha256") != result_record.get("sha256")
            or branch.get("training_result") != result_record.get("path")
            or branch.get("training_optimization_index")
            != manifest.get("training_optimization_index")
            or branch.get("training_optimization_index_sha256")
            != common_optimization_hash
            or branch.get("training_eye_eligibility")
            != manifest.get("training_eye_eligibility")
            or branch.get("training_eye_eligibility_sha256")
            != common_eligibility_hash
        ):
            raise ProtocolGateError(f"{strategy} manifest/result attestation mismatch")
        summary = summaries[strategy]
        if (
            not isinstance(summary, Mapping)
            or summary.get("model") != model
            or summary.get("seed") != seed
            or summary.get("classifier_strategy") != strategy
            or summary.get("classifier_family") != model
            or summary.get("role") != expected_spec.get("role")
            or summary.get("estimand_id") != expected_spec.get("estimand_id")
            or summary.get("available") != available
            or summary.get("monitor_status") != result.get("monitor_status")
        ):
            raise ProtocolGateError(f"{strategy} receipt summary disagrees with training result")


def _validate_branch_level_lock(value: Any, label: str, *, level: str) -> None:
    if not isinstance(value, Mapping):
        raise ProtocolGateError(f"{label} lock must be an object")
    if value.get("level") != level:
        raise ProtocolGateError(f"{label} level identity is invalid")
    if value.get("classification_threshold_status") not in {"locked", "unavailable"}:
        raise ProtocolGateError(f"{label} classification threshold status is invalid")
    if value.get("calibration_status") not in {"locked", "unavailable"}:
        raise ProtocolGateError(f"{label} calibration status is invalid")
    threshold_status = value.get("classification_threshold_status")
    calibration_status = value.get("calibration_status")
    threshold = value.get("threshold")
    temperature = value.get("temperature")
    scale = value.get("threshold_probability_scale")
    evaluable_units = value.get("evaluable_units")
    class_count = value.get("evaluable_class_count")
    if (
        isinstance(evaluable_units, bool)
        or not isinstance(evaluable_units, int)
        or evaluable_units < 0
        or isinstance(class_count, bool)
        or not isinstance(class_count, int)
        or class_count not in {0, 1, 2}
        or (evaluable_units == 0) != (class_count == 0)
    ):
        raise ProtocolGateError(f"{label} evaluable unit/class counts are invalid")
    if class_count < 2 and (
        threshold_status != "unavailable" or calibration_status != "unavailable"
    ):
        raise ProtocolGateError(f"{label} one/no-class decisions must be unavailable")
    if class_count == 2 and threshold_status != "locked":
        raise ProtocolGateError(f"{label} two-class threshold must be locked")
    if threshold_status == "unavailable" and calibration_status != "unavailable":
        raise ProtocolGateError(f"{label} cannot lock calibration without a decision threshold")
    if threshold_status == "locked":
        if (
            isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold)) or not 0 <= float(threshold) <= 1
        ):
            raise ProtocolGateError(f"{label} locked threshold is invalid")
        expected_scale = (
            "temperature_scaled" if calibration_status == "locked"
            else "raw_due_to_calibration_unavailable"
        )
        if scale != expected_scale:
            raise ProtocolGateError(f"{label} threshold probability scale is inconsistent")
    elif threshold is not None or scale is not None:
        raise ProtocolGateError(f"{label} unavailable threshold must remain null")
    if calibration_status == "locked":
        if (
            isinstance(temperature, bool) or not isinstance(temperature, (int, float))
            or not math.isfinite(float(temperature)) or float(temperature) <= 0
        ):
            raise ProtocolGateError(f"{label} locked temperature is invalid")
    elif temperature is not None:
        raise ProtocolGateError(f"{label} unavailable calibration must not substitute T=1")


def _validate_dual_lock_artifact(
    cfg: Mapping[str, Any], metadata: Mapping[str, Any], records: Sequence[Mapping[str, Any]],
    *, model: str | None, seed: int | None,
) -> None:
    if metadata.get("primary_classifier_strategy") != _CLASSIFIER_STRATEGIES[0]:
        raise ProtocolGateError("Validation lock primary must be model_specific")
    if metadata.get("secondary_classifier_strategy") != _CLASSIFIER_STRATEGIES[1]:
        raise ProtocolGateError("Validation lock secondary must be standardized_resnet18")
    if metadata.get("classifier_strategy_count") != 2:
        raise ProtocolGateError("Validation lock requires classifier_strategy_count=2")
    summaries = metadata.get("classifier_strategies")
    if not isinstance(summaries, Mapping) or set(summaries) != set(_CLASSIFIER_STRATEGIES):
        raise ProtocolGateError("Validation lock metadata requires exactly both strategies")
    lock, _ = _artifact_json_by_suffix(
        records, f"runs/{model}/seed_{seed}/lock/primary_lock.json", "dual primary lock"
    )
    branches = lock.get("classifier_strategies")
    if (
        lock.get("model") != model
        or lock.get("seed") != seed
        or lock.get("primary_classifier_strategy") != _CLASSIFIER_STRATEGIES[0]
        or lock.get("secondary_classifier_strategy") != _CLASSIFIER_STRATEGIES[1]
        or not isinstance(branches, Mapping)
        or set(branches) != set(_CLASSIFIER_STRATEGIES)
    ):
        raise ProtocolGateError("Primary lock does not contain the exact dual strategy set")
    index = _artifact_record_index(records)
    strategy_specs = {
        str(cfg["classifier"]["primary"]["name"]): cfg["classifier"]["primary"],
        str(cfg["classifier"]["secondary"]["name"]): cfg["classifier"]["secondary"],
    }
    for strategy in _CLASSIFIER_STRATEGIES:
        branch = branches[strategy]
        if not isinstance(branch, Mapping) or branch.get("classifier_strategy") != strategy:
            raise ProtocolGateError(f"Malformed {strategy} validation branch")
        if branch.get("classifier_family") != model:
            raise ProtocolGateError(f"{strategy} lock family/receipt scope mismatch")
        if branch.get("model") != model or branch.get("seed") != seed:
            raise ProtocolGateError(f"{strategy} lock model/seed scope mismatch")
        expected_spec = strategy_specs[strategy]
        if (
            branch.get("classifier_role") != expected_spec.get("role")
            or branch.get("classifier_estimand_id") != expected_spec.get("estimand_id")
        ):
            raise ProtocolGateError(f"{strategy} lock estimand/role mismatch")
        _validate_branch_level_lock(
            branch.get("eye"), f"{strategy} eye", level="eye"
        )
        _validate_branch_level_lock(
            branch.get("patient"), f"{strategy} patient", level="patient"
        )
        eye_count = branch.get("validation_evaluable_eye_count")
        class_count = branch.get("validation_evaluable_class_count")
        if (
            isinstance(eye_count, bool) or not isinstance(eye_count, int) or eye_count < 0
            or isinstance(class_count, bool) or not isinstance(class_count, int)
            or class_count not in {0, 1, 2}
            or (eye_count == 0) != (class_count == 0)
        ):
            raise ProtocolGateError(f"{strategy} validation eye/class counts are invalid")
        expected_monitor = {
            0: "non_evaluable", 1: "negative_nll_fallback", 2: "auroc"
        }[class_count]
        if branch.get("monitor_status") != expected_monitor:
            raise ProtocolGateError(f"{strategy} validation monitor/count semantics conflict")
        if class_count < 2 and (
            branch["eye"].get("classification_threshold_status") != "unavailable"
            or branch["eye"].get("calibration_status") != "unavailable"
        ):
            raise ProtocolGateError(f"{strategy} one/no-class eye lock must be unavailable")
        if (
            branch["eye"].get("evaluable_units") != eye_count
            or branch["eye"].get("evaluable_class_count") != class_count
        ):
            raise ProtocolGateError(f"{strategy} eye lock/count summary mismatch")
        for field, digest_field, label in (
            ("classifier_history", "classifier_history_sha256", "history"),
            ("classifier_provenance", "classifier_provenance_sha256", "provenance"),
            ("classifier_training_result", "classifier_training_result_sha256", "training result"),
            ("validation_frames", "validation_frames_sha256", "validation frames"),
            ("outside_roi_invariance_path", "outside_roi_invariance_sha256", "outside-ROI audit"),
        ):
            _require_attested_reference(
                index, branch.get(field), branch.get(digest_field),
                f"{strategy} {label}",
            )
            normalized_path = str(branch.get(field, "")).replace("\\", "/")
            expected_fragment = f"runs/{model}/seed_{seed}/"
            if expected_fragment not in normalized_path:
                raise ProtocolGateError(f"{strategy} {label} comes from another unit")
        for field, digest_field, suffix, label in (
            (
                "training_roi_index",
                "training_roi_index_sha256",
                f"runs/{model}/seed_{seed}/roi/train_oof_index.csv",
                "raw OOF training ROI index",
            ),
            (
                "training_optimization_index",
                "training_optimization_index_sha256",
                f"runs/{model}/seed_{seed}/classifiers/shared/training_optimization_index.csv",
                "classifier optimization index",
            ),
            (
                "training_eye_eligibility",
                "training_eye_eligibility_sha256",
                f"runs/{model}/seed_{seed}/classifiers/shared/training_eye_eligibility.csv",
                "classifier eye eligibility ledger",
            ),
        ):
            _require_attested_reference(
                index, branch.get(field), branch.get(digest_field),
                f"{strategy} {label}",
            )
            if not str(branch.get(field, "")).replace("\\", "/").endswith(suffix):
                raise ProtocolGateError(f"{strategy} {label} comes from another unit")
        if (
            branch.get("frames_per_eye") != int(cfg["dataset"]["frames_per_eye"])
            or branch.get("minimum_valid_frames")
            != int(cfg["aggregation"]["minimum_valid_frames"])
        ):
            raise ProtocolGateError(f"{strategy} training eligibility policy changed")
        _require_attested_reference(
            index,
            branch.get("classifier_checkpoint"),
            branch.get("classifier_checkpoint_sha256"),
            f"{strategy} checkpoint",
            optional=True,
        )
        if branch.get("classifier_checkpoint") is not None and (
            f"runs/{model}/seed_{seed}/classifiers/{strategy}/"
            not in str(branch["classifier_checkpoint"]).replace("\\", "/")
        ):
            raise ProtocolGateError(f"{strategy} checkpoint comes from another unit")
        checkpoint_present = branch.get("classifier_checkpoint") is not None
        expected_operational = bool(
            checkpoint_present
            and branch["eye"].get("classification_threshold_status") == "locked"
        )
        expected_calibrated = bool(
            expected_operational and branch["eye"].get("calibration_status") == "locked"
        )
        if (
            branch.get("operational_system_evaluable") is not expected_operational
            or branch.get("calibrated_system_evaluable") is not expected_calibrated
        ):
            raise ProtocolGateError(f"{strategy} checkpoint/evaluability semantics conflict")
        if not checkpoint_present:
            if (
                branch.get("operational_system_evaluable") is True
                or branch["eye"].get("classification_threshold_status") != "unavailable"
                or branch["patient"].get("classification_threshold_status") != "unavailable"
            ):
                raise ProtocolGateError(
                    f"{strategy} has decisions/evaluability without an attested checkpoint"
                )
        else:
            invariance = branch.get("outside_roi_invariance")
            if not isinstance(invariance, Mapping) or invariance.get("passed") is not True:
                raise ProtocolGateError(
                    f"{strategy} checkpoint requires a passed outside-ROI audit"
                )
        attested_invariance = read_json(
            resolve_project_path(branch["outside_roi_invariance_path"], must_exist=True)
        )
        if attested_invariance != branch.get("outside_roi_invariance"):
            raise ProtocolGateError(f"{strategy} inline outside-ROI audit is not attested")
        summary = summaries[strategy]
        if not isinstance(summary, Mapping):
            raise ProtocolGateError(f"Malformed {strategy} receipt summary")
        for field in (
            "monitor_status", "classification_threshold_status", "calibration_status",
            "operational_system_evaluable", "validation_evaluable_eye_count",
            "validation_evaluable_class_count",
        ):
            if summary.get(field) != branch.get(field):
                raise ProtocolGateError(f"{strategy} receipt summary disagrees on {field}")
        if (
            summary.get("role") != expected_spec.get("role")
            or summary.get("estimand_id") != expected_spec.get("estimand_id")
        ):
            raise ProtocolGateError(f"{strategy} receipt summary estimand/role mismatch")
    primary = branches[_CLASSIFIER_STRATEGIES[0]]
    secondary = branches[_CLASSIFIER_STRATEGIES[1]]
    for field in (
        "training_roi_index",
        "training_roi_index_sha256",
        "training_roi_dataframe_sha256",
        "training_optimization_index",
        "training_optimization_index_sha256",
        "training_optimization_dataframe_sha256",
        "training_eye_eligibility",
        "training_eye_eligibility_sha256",
        "training_eye_eligibility_dataframe_sha256",
        "frames_per_eye",
        "minimum_valid_frames",
        "eligible_training_eye_count",
        "ineligible_training_eye_count",
        "optimization_frame_count",
    ):
        if primary.get(field) != secondary.get(field):
            raise ProtocolGateError(
                f"Validation lock classifier branches disagree on shared {field}"
            )
    for alias, branch_field in (
        ("classifier_checkpoint", "classifier_checkpoint"),
        ("classifier_checkpoint_sha256", "classifier_checkpoint_sha256"),
        ("eye", "eye"),
        ("patient", "patient"),
        ("monitor_status", "monitor_status"),
        ("primary_system_evaluable", "operational_system_evaluable"),
    ):
        if lock.get(alias) != primary.get(branch_field):
            raise ProtocolGateError(f"Primary lock alias {alias} is not model_specific")


def _validate_preflight_report(
    cfg: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    model: str | None,
) -> None:
    matches = [
        record for record in records
        if "/verification/" in "/" + str(record.get("path", "")).replace("\\", "/")
        and str(record.get("path", "")).endswith(".json")
    ]
    if len(matches) != 1:
        raise ProtocolGateError("Preflight receipt requires one verification report")
    report = read_json(resolve_project_path(str(matches[0]["path"]), must_exist=True))
    if report.get("model") != model or not str(matches[0]["path"]).replace(
        "\\", "/"
    ).endswith(f"verification/{model}.json"):
        raise ProtocolGateError("Preflight report identity does not match receipt scope")
    if report.get("passed") is not True:
        raise ProtocolGateError("Preflight verification report did not pass")
    if (
        report.get("primary_classifier_strategy") != _CLASSIFIER_STRATEGIES[0]
        or report.get("classifier_strategies") != list(_CLASSIFIER_STRATEGIES)
        or set(report.get("classifier_audits", {})) != set(_CLASSIFIER_STRATEGIES)
    ):
        raise ProtocolGateError("Preflight report lacks the exact dual classifier audit")
    required = report.get("required_prelaunch_audits")
    expected = (
        (1, "dual_classifier_contract_and_backward"),
        (2, "synthetic_strict_roi_end_to_end_and_outside_roi_invariance"),
    )
    if not isinstance(required, list) or len(required) != 2:
        raise ProtocolGateError("Preflight report requires exactly two audit rounds")
    for audit, (round_number, name) in zip(required, expected, strict=True):
        if (
            not isinstance(audit, Mapping)
            or audit.get("round") != round_number
            or audit.get("name") != name
            or audit.get("status") != "passed"
            or set(audit.get("strategies", {})) != set(_CLASSIFIER_STRATEGIES)
            or audit.get("executed_scope") != f"{report.get('model')}_both_strategies"
            or audit.get("aggregate_required_scope") != "all_four_models_both_strategies"
        ):
            raise ProtocolGateError(f"Preflight audit round {round_number} is incomplete")
        if any(
            branch.get("strict_hard_mask_supplied") is not True
            for branch in audit.get("strategies", {}).values()
        ) and round_number == 1:
            raise ProtocolGateError("Both preflight strategies require an explicit hard mask")
    if required[1].get("strict_roi_tensor_shared_by_both_strategies") is not True:
        raise ProtocolGateError("Preflight round 2 must share one strict ROI tensor")
    if required[1].get("classifier_training_eligibility_contract_passed") is not True:
        raise ProtocolGateError("Preflight round 2 must pass the classifier 4/7 eligibility contract")
    eligibility = report.get("classifier_training_eligibility_audit")
    if (
        not isinstance(eligibility, Mapping)
        or eligibility.get("status") != "passed"
        or eligibility.get("canonical_roi_index_immutable") is not True
        or eligibility.get("three_of_seven_eye_excluded") is not True
        or eligibility.get("four_of_seven_eye_included") is not True
        or eligibility.get("seven_of_seven_eye_included") is not True
        or eligibility.get("valid_but_ineligible_cache_rows_never_relabelled") is not True
        or eligibility.get("valid_but_ineligible_cache_rows_never_enter_dataset") is not True
        or eligibility.get("identical_selection_for_both_classifier_strategies") is not True
        or eligibility.get("frames_per_eye") != int(cfg["dataset"]["frames_per_eye"])
        or eligibility.get("minimum_valid_frames")
        != int(cfg["aggregation"]["minimum_valid_frames"])
        or eligibility.get("optimization_frame_count") != 11
    ):
        raise ProtocolGateError("Preflight classifier training eligibility audit is incomplete")
    for field in ("optimization_index_sha256", "eligibility_ledger_sha256"):
        digest = eligibility.get(field)
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            raise ProtocolGateError(f"Preflight eligibility audit lacks {field}")
    primary_audit = report["classifier_audits"][_CLASSIFIER_STRATEGIES[0]]
    if primary_audit.get("trainability_policy_audit", {}).get("status") != "passed":
        raise ProtocolGateError("Model-specific classifier trainability audit did not pass")
    tolerance = report.get("outside_roi_invariance_tolerance")
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or tolerance < 0:
        raise ProtocolGateError("Preflight outside-ROI tolerance is invalid")
    for strategy in _CLASSIFIER_STRATEGIES:
        audit = report["classifier_audits"][strategy]
        difference = audit.get("outside_roi_invariance_maximum_logit_difference")
        if (
            not isinstance(audit, Mapping)
            or audit.get("classifier_strategy") != strategy
            or audit.get("finite_forward") is not True
            or audit.get("finite_backward") is not True
            or audit.get("classifier_logits_shape") != [1, 2]
            or audit.get("strict_hard_mask_supplied") is not True
            or audit.get("outside_roi_invariance_passed") is not True
            or isinstance(difference, bool)
            or not isinstance(difference, (int, float))
            or not math.isfinite(float(difference))
            or float(difference) < 0
            or float(difference) > float(tolerance)
        ):
            raise ProtocolGateError(f"Preflight classifier audit failed semantically: {strategy}")
        first = required[0]["strategies"][strategy]
        second = required[1]["strategies"][strategy]
        if (
            first.get("finite_forward") is not True
            or first.get("finite_backward") is not True
            or first.get("classifier_logits_shape") != [1, 2]
            or first.get("strict_hard_mask_supplied") is not True
            or second.get("outside_roi_invariance_passed") is not True
            or second.get("maximum_absolute_logit_difference") != difference
        ):
            raise ProtocolGateError(f"Preflight round summary disagrees for {strategy}")


def _validate_dual_evaluation_receipt(
    cfg: Mapping[str, Any], metadata: Mapping[str, Any], records: Sequence[Mapping[str, Any]],
    *, model: str | None, seed: int | None,
) -> None:
    if metadata.get("primary_classifier_strategy") != _CLASSIFIER_STRATEGIES[0]:
        raise ProtocolGateError("Evaluation primary must be model_specific")
    strategies = metadata.get("classifier_strategies")
    if not isinstance(strategies, Mapping) or set(strategies) != set(_CLASSIFIER_STRATEGIES):
        raise ProtocolGateError("Evaluation receipt requires exactly both classifier strategies")
    if metadata.get("test_selection_or_refitting") is not False:
        raise ProtocolGateError("Evaluation must attest test_selection_or_refitting=false")
    attempt, _ = _artifact_json_by_suffix(
        records, f"runs/{model}/seed_{seed}/evaluation/attempt.json", "evaluation attempt"
    )
    if (
        attempt.get("model") != model
        or attempt.get("seed") != seed
        or attempt.get("status") != "complete"
        or attempt.get("primary_classifier_strategy") != _CLASSIFIER_STRATEGIES[0]
        or set(attempt.get("classifier_strategies", {})) != set(_CLASSIFIER_STRATEGIES)
    ):
        raise ProtocolGateError("Evaluation attempt identity/dual strategy set is invalid")
    lock, lock_record = _artifact_json_by_suffix(
        records, f"runs/{model}/seed_{seed}/lock/primary_lock.json", "dual primary lock"
    )
    lock_branches = lock.get("classifier_strategies")
    if (
        lock.get("model") != model
        or lock.get("seed") != seed
        or lock.get("primary_classifier_strategy") != _CLASSIFIER_STRATEGIES[0]
        or not isinstance(lock_branches, Mapping)
        or set(lock_branches) != set(_CLASSIFIER_STRATEGIES)
        or attempt.get("primary_lock_sha256") != lock_record.get("sha256")
    ):
        raise ProtocolGateError("Evaluation attempt is not bound to the dual validation lock")
    index = _artifact_record_index(records)
    test_roi_matches = [
        record for record in records
        if str(record.get("path", "")).replace("\\", "/").endswith(
            f"runs/{model}/seed_{seed}/evaluation/test_rois/index.csv"
        )
    ]
    if len(test_roi_matches) != 1:
        raise ProtocolGateError("Evaluation receipt requires the unit test ROI index")
    access_matches = [
        record for record in records
        if str(record.get("path", "")).replace("\\", "/").endswith(
            "/protocol/test_access.json"
        )
    ]
    if (
        len(access_matches) != 1
        or attempt.get("test_access_sha256") != access_matches[0].get("sha256")
    ):
        raise ProtocolGateError("Evaluation attempt is not bound to the global test-access gate")
    required_names = (
        "frames.csv", "eyes.csv", "patients.csv", "metrics.json",
        "calibration_curves.csv", "calibration_reliability_bands.csv",
        "decision_curves.csv", "risk_coverage_curves.csv",
        "confusion_2x3.csv", "confusion_conditional_2x2.csv",
    )
    strategy_specs = {
        str(cfg["classifier"]["primary"]["name"]): cfg["classifier"]["primary"],
        str(cfg["classifier"]["secondary"]["name"]): cfg["classifier"]["secondary"],
    }
    for strategy in _CLASSIFIER_STRATEGIES:
        summary = strategies[strategy]
        attempt_summary = attempt["classifier_strategies"][strategy]
        expected_spec = strategy_specs[strategy]
        for value, label in ((summary, "receipt"), (attempt_summary, "attempt")):
            if (
                not isinstance(value, Mapping)
                or value.get("model") != model
                or value.get("seed") != seed
                or value.get("classifier_strategy") != strategy
                or value.get("classifier_family") != model
                or value.get("role") != expected_spec.get("role")
                or value.get("classifier_estimand_id") != expected_spec.get("estimand_id")
                or value.get("is_primary_estimand") is not (
                    strategy == _CLASSIFIER_STRATEGIES[0]
                )
            ):
                raise ProtocolGateError(f"Evaluation {label} identity mismatch for {strategy}")
        if summary != attempt_summary:
            raise ProtocolGateError(f"Evaluation receipt/attempt summary mismatch for {strategy}")
        locked_branch = lock_branches[strategy]
        if (
            not isinstance(locked_branch, Mapping)
            or locked_branch.get("classifier_strategy") != strategy
            or locked_branch.get("classifier_family") != model
            or locked_branch.get("model") != model
            or locked_branch.get("seed") != seed
            or attempt.get("classifier_strategy_checkpoint_sha256", {}).get(strategy)
            != locked_branch.get("classifier_checkpoint_sha256")
            or summary.get("operational_system_evaluable")
            != locked_branch.get("operational_system_evaluable")
        ):
            raise ProtocolGateError(f"Evaluation branch disagrees with lock: {strategy}")
        for field, digest_field, label in (
            ("classifier_history", "classifier_history_sha256", "history"),
            ("classifier_provenance", "classifier_provenance_sha256", "provenance"),
            (
                "classifier_training_result",
                "classifier_training_result_sha256",
                "training result",
            ),
            ("classifier_checkpoint", "classifier_checkpoint_sha256", "checkpoint"),
        ):
            _require_attested_reference(
                index,
                locked_branch.get(field),
                locked_branch.get(digest_field),
                f"evaluation {strategy} {label}",
                optional=field == "classifier_checkpoint",
            )
        for name in required_names:
            suffix = f"runs/{model}/seed_{seed}/evaluation/{strategy}/{name}"
            matches = [
                record for record in records
                if str(record.get("path", "")).replace("\\", "/").endswith(suffix)
            ]
            if len(matches) != 1:
                raise ProtocolGateError(f"Evaluation receipt lacks {strategy}/{name}")
            path = resolve_project_path(str(matches[0]["path"]), must_exist=True)
            if name == "metrics.json":
                report = read_json(path)
                if report.get("protocol", {}).get("classifier_strategy") != strategy:
                    raise ProtocolGateError(f"Evaluation metrics mislabeled for {strategy}")
            else:
                with path.open("r", encoding="utf-8-sig", newline="") as handle:
                    reader = csv.DictReader(handle)
                    if reader.fieldnames is None or "classifier_strategy" not in reader.fieldnames:
                        raise ProtocolGateError(f"Evaluation table lacks classifier_strategy: {path}")
                    if any(row.get("classifier_strategy") != strategy for row in reader):
                        raise ProtocolGateError(f"Evaluation table mixes classifier strategies: {path}")


def _validate_stage_metadata(
    cfg: Mapping[str, Any],
    stage: str,
    metadata_value: Mapping[str, Any],
    artifact_records: Sequence[Mapping[str, Any]] | None = None,
    *, model: str | None = None, seed: int | None = None,
) -> None:
    """Validate scientific lock status both when writing and re-reading receipts."""

    if stage == "preflight":
        if artifact_records is None:
            raise ProtocolGateError("Preflight receipt requires an artifact inventory")
        _validate_preflight_report(cfg, artifact_records, model=model)
    elif stage == "train-classifiers":
        if artifact_records is None:
            raise ProtocolGateError("train-classifiers requires an artifact inventory")
        _validate_dual_training_receipt(
            cfg, metadata_value, artifact_records, model=model, seed=seed
        )
    elif stage == "lock":
        monitor_status = metadata_value.get("monitor_status")
        allowed_statuses = {"auroc", "negative_nll_fallback", "non_evaluable"}
        if monitor_status not in allowed_statuses:
            raise ProtocolGateError(
                "A validation lock receipt requires monitor_status in " + str(sorted(allowed_statuses))
            )
        for required in (
            "classification_threshold_status",
            "calibration_status",
            "roi_grid_status",
            "validation_evaluable_eye_count",
            "validation_evaluable_class_count",
        ):
            if required not in metadata_value:
                raise ProtocolGateError(f"A validation lock receipt requires metadata.{required}")
        threshold_status = metadata_value["classification_threshold_status"]
        calibration_status = metadata_value["calibration_status"]
        roi_grid_status = metadata_value["roi_grid_status"]
        if threshold_status not in {"locked", "unavailable"}:
            raise ProtocolGateError("classification_threshold_status must be locked or unavailable")
        if calibration_status not in {"locked", "unavailable"}:
            raise ProtocolGateError("calibration_status must be locked or unavailable")
        if roi_grid_status not in {"feasible", "infeasible_fallback"}:
            raise ProtocolGateError("roi_grid_status must be feasible or infeasible_fallback")
        eye_count = metadata_value["validation_evaluable_eye_count"]
        class_count = metadata_value["validation_evaluable_class_count"]
        if isinstance(eye_count, bool) or not isinstance(eye_count, int) or eye_count < 0:
            raise ProtocolGateError("validation_evaluable_eye_count must be a non-negative integer")
        if isinstance(class_count, bool) or not isinstance(class_count, int) or class_count not in {0, 1, 2}:
            raise ProtocolGateError("validation_evaluable_class_count must be 0, 1, or 2")
        if (eye_count == 0) != (class_count == 0):
            raise ProtocolGateError("Evaluable eye and class counts are inconsistent")
        expected_monitor = {0: "non_evaluable", 1: "negative_nll_fallback", 2: "auroc"}[class_count]
        if monitor_status != expected_monitor:
            raise ProtocolGateError(
                f"monitor_status={monitor_status!r} conflicts with {class_count} evaluable validation classes"
            )
        if class_count < 2 and (threshold_status != "unavailable" or calibration_status != "unavailable"):
            raise ProtocolGateError(
                "Fewer than two evaluable validation classes require unavailable threshold and calibration"
            )
        if class_count == 2 and threshold_status != "locked":
            raise ProtocolGateError("Two evaluable validation classes require a locked classification threshold")
        reasons = metadata_value.get("status_reasons", {})
        if not isinstance(reasons, Mapping):
            raise ProtocolGateError("metadata.status_reasons must be an object")
        for status_name, status_value in (
            ("classification_threshold_status", threshold_status),
            ("calibration_status", calibration_status),
        ):
            if status_value == "unavailable" and not isinstance(reasons.get(status_name), str):
                raise ProtocolGateError(f"Unavailable {status_name} requires a text status_reasons entry")
        if roi_grid_status == "infeasible_fallback" and not isinstance(reasons.get("roi_grid_status"), str):
            raise ProtocolGateError("ROI-grid fallback requires status_reasons.roi_grid_status")
        if artifact_records is None:
            raise ProtocolGateError("Validation lock receipt requires an artifact inventory")
        _validate_dual_lock_artifact(
            cfg, metadata_value, artifact_records, model=model, seed=seed
        )
    elif stage == "evaluate":
        if artifact_records is None:
            raise ProtocolGateError("Evaluation receipt requires an artifact inventory")
        _validate_dual_evaluation_receipt(
            cfg, metadata_value, artifact_records, model=model, seed=seed
        )
    elif stage == "lock-ablations":
        expected = int(cfg["ablations"]["expected_validation_locks"])
        if metadata_value.get("ablation_validation_lock_count") != expected:
            raise ProtocolGateError(
                f"lock-ablations requires exactly {expected} completed validation locks"
            )
        if metadata_value.get("all_ablation_validation_locks_complete") is not True:
            raise ProtocolGateError("lock-ablations requires all_ablation_validation_locks_complete=true")
        if metadata_value.get("test_predictions_generated") is not False:
            raise ProtocolGateError("lock-ablations must attest test_predictions_generated=false")
        lock_manifest_sha256 = metadata_value.get("lock_manifest_sha256")
        if not isinstance(lock_manifest_sha256, str) or not SHA256_RE.fullmatch(lock_manifest_sha256):
            raise ProtocolGateError("lock-ablations requires a lowercase SHA-256 lock_manifest_sha256")
        if artifact_records is not None and not any(
            record.get("sha256") == lock_manifest_sha256 for record in artifact_records
        ):
            raise ProtocolGateError(
                "lock-ablations lock_manifest_sha256 must identify one attested artifact"
            )
    elif stage == "evaluate-ablations":
        expected = int(cfg["ablations"]["expected_test_evaluations"])
        if metadata_value.get("ablation_test_evaluation_count") != expected:
            raise ProtocolGateError(
                f"evaluate-ablations requires exactly {expected} completed test evaluations"
            )
        if metadata_value.get("all_ablation_test_evaluations_complete") is not True:
            raise ProtocolGateError(
                "evaluate-ablations requires all_ablation_test_evaluations_complete=true"
            )
        if metadata_value.get("separate_immutable_ablation_summary") is not True:
            raise ProtocolGateError(
                "evaluate-ablations requires separate_immutable_ablation_summary=true"
            )
        if metadata_value.get("primary_summary_modified") is not False:
            raise ProtocolGateError("evaluate-ablations must attest primary_summary_modified=false")
        locked = assert_all_ablation_locks(cfg)
        if metadata_value.get("ablation_lock_receipt_sha256") != locked["receipt_sha256"]:
            raise ProtocolGateError("evaluate-ablations does not attest the frozen ablation lock set")


def write_stage_receipt(
    cfg: Mapping[str, Any],
    stage: str,
    *,
    model: str | None = None,
    seed: int | None = None,
    artifacts: Iterable[str | Path] = (),
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically attest completion; existing valid receipts are immutable."""

    destination = stage_receipt_path(cfg, stage, model=model, seed=seed)
    if destination.exists():
        verify_stage_receipt(cfg, stage, model=model, seed=seed)
        return destination
    context = _runtime_context(cfg)
    records = [_artifact_record(path) for path in artifacts]
    if not records:
        raise ProtocolGateError(f"Cannot complete {stage} without at least one hashed artifact")
    metadata_value = dict(metadata or {})
    _validate_stage_metadata(
        cfg, stage, metadata_value, records, model=model, seed=seed
    )
    value: dict[str, Any] = {
        "receipt_schema": 1,
        "status": "complete",
        "stage": stage,
        "model": model,
        "seed": seed,
        "completed_utc": utc_now(),
        **context,
        "split_sha256": sha256_file(split_path(cfg, seed)) if seed is not None else None,
        "artifacts": records,
        "metadata": metadata_value,
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
        },
    }
    save_json_atomic(destination, value)
    return destination


def verify_stage_receipt(
    cfg: Mapping[str, Any], stage: str, *, model: str | None = None, seed: int | None = None
) -> dict[str, Any]:
    destination = stage_receipt_path(cfg, stage, model=model, seed=seed)
    if not destination.is_file():
        scope = f" {model}/seed {seed}" if model is not None else ""
        raise ProtocolGateError(f"Missing completed {stage} receipt{scope}: {destination}")
    receipt = read_json(destination)
    expected = _runtime_context(cfg)
    if receipt.get("receipt_schema") != 1 or receipt.get("status") != "complete":
        raise ProtocolGateError(f"Invalid or incomplete receipt: {destination}")
    if receipt.get("stage") != stage or receipt.get("model") != model or receipt.get("seed") != seed:
        raise ProtocolGateError(f"Receipt scope mismatch: {destination}")
    for key, expected_value in expected.items():
        if receipt.get(key) != expected_value:
            raise ProtocolGateError(f"{key} changed after {stage}: {destination}")
    if seed is not None:
        current_split = split_path(cfg, seed)
        if not current_split.is_file() or receipt.get("split_sha256") != sha256_file(current_split):
            raise ProtocolGateError(f"Patient split changed after {stage}: seed {seed}")
    records = receipt.get("artifacts")
    if not isinstance(records, list) or not records:
        raise ProtocolGateError(f"Receipt has no artifact inventory: {destination}")
    metadata = receipt.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ProtocolGateError(f"Receipt metadata is malformed: {destination}")
    _validate_stage_metadata(
        cfg, stage, metadata, records, model=model, seed=seed
    )
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
            raise ProtocolGateError(f"Malformed artifact record: {destination}")
        path = resolve_project_path(record["path"])
        if not path.is_file():
            raise ProtocolGateError(f"Attested artifact is missing: {path}")
        if path.stat().st_size != record.get("bytes") or sha256_file(path) != record.get("sha256"):
            raise ProtocolGateError(f"Attested artifact changed: {path}")
    return receipt


def assert_prerequisites(
    cfg: Mapping[str, Any], stage: str, *, model: str | None = None, seed: int | None = None
) -> None:
    """Enforce the protocol DAG before importing or invoking a stage runner."""

    _validate_stage_scope(cfg, stage, model, seed)
    if stage == "prepare":
        return
    verify_stage_receipt(cfg, "prepare")
    if stage == "preflight":
        return
    if stage == "train-segmenters" and cfg["preflight"].get(
        "train_segmenters_requires_all_four_preflight_receipts"
    ) is True:
        for required_model in cfg["models"]:
            verify_stage_receipt(cfg, "preflight", model=required_model)
    elif stage in PER_UNIT_STAGES:
        assert model is not None and seed is not None
        verify_stage_receipt(cfg, "preflight", model=model)
    predecessor = {
        "train-segmenters": None,
        "build-rois": "train-segmenters",
        "train-classifiers": "build-rois",
        "lock": "train-classifiers",
    }.get(stage)
    if predecessor is not None:
        verify_stage_receipt(cfg, predecessor, model=model, seed=seed)
    if stage == "evaluate":
        # No model may see the test set while another model/seed can still be
        # changed using validation results.
        assert_all_primary_locks(cfg)
    elif stage == "lock-ablations":
        # Freeze the primary-only summary before any secondary analysis and
        # before any ablation test prediction is generated.
        assert_all_primary_evaluations(cfg)
        verify_stage_receipt(cfg, "summarize")
    elif stage == "evaluate-ablations":
        # The global receipt can only exist after the engine has created and
        # audited every predeclared model/seed/arm validation lock.
        verify_stage_receipt(cfg, "summarize")
        assert_all_ablation_locks(cfg)
    elif stage == "summarize":
        assert_all_primary_evaluations(cfg)


def assert_all_primary_locks(cfg: Mapping[str, Any]) -> dict[str, str]:
    """Verify every lock and its complete model/seed predecessor receipt chain."""

    hashes: dict[str, str] = {}
    missing: list[str] = []
    invalid: list[str] = []
    for model in cfg["models"]:
        for seed in cfg["split_seeds"]:
            path = stage_receipt_path(cfg, "lock", model=model, seed=seed)
            if not path.is_file():
                missing.append(f"{model}/seed_{seed}")
                continue
            try:
                verify_stage_receipt(cfg, "preflight", model=model)
                chain_hashes: dict[str, str] = {
                    "preflight": sha256_file(stage_receipt_path(cfg, "preflight", model=model))
                }
                for chain_stage in (
                    "train-segmenters",
                    "build-rois",
                    "train-classifiers",
                    "lock",
                ):
                    verify_stage_receipt(cfg, chain_stage, model=model, seed=seed)
                    chain_hashes[chain_stage] = sha256_file(
                        stage_receipt_path(cfg, chain_stage, model=model, seed=seed)
                    )
            except ProtocolGateError as error:
                invalid.append(f"{model}/seed_{seed}: {error}")
            else:
                hashes[f"{model}/seed_{seed}"] = canonical_sha256(chain_hashes)
    if missing or invalid:
        details = []
        if missing:
            details.append("missing=" + ", ".join(missing))
        if invalid:
            details.append("invalid=" + " | ".join(invalid))
        expected = int(cfg["test_access"]["composite_model_seed_lock_count"])
        raise ProtocolGateError(
            f"Test access denied: all {expected} model/seed validation locks must be complete and valid; "
            + "; ".join(details)
        )
    if len(hashes) != len(EXPECTED_MODELS) * len(EXPECTED_SEEDS):
        expected = len(EXPECTED_MODELS) * len(EXPECTED_SEEDS)
        raise ProtocolGateError(f"Test access denied: lock cardinality is not {expected}")
    return hashes


def assert_all_primary_evaluations(cfg: Mapping[str, Any]) -> None:
    # Evaluation receipts are not sufficient on their own: their upstream
    # locks/checkpoints and the global test-access sentinel must still attest
    # the exact same configured lock chains when a summary or ablation phase is opened.
    lock_hashes = assert_all_primary_locks(cfg)
    test_access = output_root(cfg) / "state" / "test_access_opened.json"
    if not test_access.is_file():
        raise ProtocolGateError(
            "All primary evaluations require the immutable global test-access sentinel"
        )
    expected_access = {
        "schema": 1,
        "study_id": cfg["study_id"],
        **_runtime_context(cfg),
        "unit_receipt_chain_sha256": lock_hashes,
    }
    observed_access = read_json(test_access)
    if {key: observed_access.get(key) for key in expected_access} != expected_access:
        raise ProtocolGateError(
            "Primary lock chains or runtime context changed after test access was opened"
        )
    missing: list[str] = []
    for model in cfg["models"]:
        for seed in cfg["split_seeds"]:
            try:
                verify_stage_receipt(cfg, "evaluate", model=model, seed=seed)
            except ProtocolGateError:
                missing.append(f"{model}/seed_{seed}")
    if missing:
        expected = int(cfg["test_access"]["composite_model_seed_lock_count"])
        raise ProtocolGateError(
            f"All {expected} primary evaluations are required; missing/invalid: "
            + ", ".join(missing)
        )


def assert_all_ablation_locks(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the global configured ablation-lock receipt before any test."""

    receipt = verify_stage_receipt(cfg, "lock-ablations")
    metadata = receipt.get("metadata", {})
    expected = int(cfg["ablations"]["expected_validation_locks"])
    if metadata.get("ablation_validation_lock_count") != expected:
        raise ProtocolGateError(f"Ablation test access denied: expected {expected} validation locks")
    if metadata.get("all_ablation_validation_locks_complete") is not True:
        raise ProtocolGateError("Ablation test access denied: validation lock set is incomplete")
    return {
        "receipt_path": str(stage_receipt_path(cfg, "lock-ablations")),
        "receipt_sha256": sha256_file(stage_receipt_path(cfg, "lock-ablations")),
        "lock_manifest_sha256": metadata["lock_manifest_sha256"],
        "lock_count": expected,
    }


def open_ablation_test_access(cfg: Mapping[str, Any]) -> Path:
    """Freeze the global ablation lock set before the first ablation test read."""

    assert_all_primary_evaluations(cfg)
    locked = assert_all_ablation_locks(cfg)
    primary_summary = stage_receipt_path(cfg, "summarize")
    verify_stage_receipt(cfg, "summarize")
    destination = output_root(cfg) / "state" / "ablation_test_access_opened.json"
    expected = {
        "schema": 1,
        "study_id": cfg["study_id"],
        **_runtime_context(cfg),
        "primary_summary_receipt_sha256": sha256_file(primary_summary),
        "ablation_lock_receipt_sha256": locked["receipt_sha256"],
        "ablation_lock_manifest_sha256": locked["lock_manifest_sha256"],
        "ablation_validation_lock_count": locked["lock_count"],
    }
    if destination.exists():
        existing = read_json(destination)
        comparable = {key: existing.get(key) for key in expected}
        if comparable != expected:
            raise ProtocolGateError(
                "The code/config/protocol, primary summary, or ablation lock set changed after ablation test access"
            )
        return destination
    save_json_atomic(destination, {**expected, "opened_utc": utc_now()})
    return destination


def open_test_access(cfg: Mapping[str, Any]) -> Path:
    """Freeze one global lock-set before the first test inference."""

    lock_hashes = assert_all_primary_locks(cfg)
    destination = output_root(cfg) / "state" / "test_access_opened.json"
    expected = {
        "schema": 1,
        "study_id": cfg["study_id"],
        **_runtime_context(cfg),
        "unit_receipt_chain_sha256": lock_hashes,
    }
    if destination.exists():
        existing = read_json(destination)
        comparable = {key: existing.get(key) for key in expected}
        if comparable != expected:
            expected = int(cfg["test_access"]["composite_model_seed_lock_count"])
            raise ProtocolGateError(
                f"The code/config/protocol or {expected}-lock set changed after test access was opened"
            )
        return destination
    save_json_atomic(destination, {**expected, "opened_utc": utc_now()})
    return destination


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ProtocolGateError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def _audit_manifest(cfg: Mapping[str, Any]) -> dict[str, Any]:
    manifest = resolve_project_path(cfg["dataset"]["manifest"], must_exist=True)
    fields, rows = _read_csv(manifest)
    required = {"patient_id", "case_id", "side", "label_binary", "label_3class", "frame_id"}
    if not required <= set(fields):
        raise ProtocolGateError(f"Manifest is missing fields: {sorted(required - set(fields))}")
    if len(rows) != cfg["dataset"]["frames"]:
        raise ProtocolGateError("Manifest frame count differs from the protocol")
    cases: dict[str, list[dict[str, str]]] = defaultdict(list)
    patients: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        cases[row["case_id"]].append(row)
        patients[row["patient_id"]].append(row)
    if len(cases) != cfg["dataset"]["eyes"] or len(patients) != cfg["dataset"]["patients"]:
        raise ProtocolGateError("Manifest patient/eye cardinality differs from the protocol")
    for case_id, group in cases.items():
        if len(group) != 7 or len({row["frame_id"] for row in group}) != 7:
            raise ProtocolGateError(f"Eye {case_id} does not have seven unique frames")
        if len({row["patient_id"] for row in group}) != 1 or len({row["side"] for row in group}) != 1:
            raise ProtocolGateError(f"Eye {case_id} has inconsistent identity or side")
    patient_table: dict[str, dict[str, Any]] = {}
    for patient_id, group in patients.items():
        if len({row["case_id"] for row in group}) != 2 or {row["side"] for row in group} != {"SAG", "SOL"}:
            raise ProtocolGateError(f"Patient {patient_id} does not have exactly both eyes")
        labels3 = {int(row["label_3class"]) for row in group}
        labels2 = {int(row["label_binary"]) for row in group}
        if len(labels3) != 1 or len(labels2) != 1 or next(iter(labels2)) != int(next(iter(labels3)) > 0):
            raise ProtocolGateError(f"Patient {patient_id} has inconsistent labels")
        patient_table[patient_id] = {
            "label_3class": next(iter(labels3)),
            "label_binary": next(iter(labels2)),
        }
    return {
        "manifest_sha256": sha256_file(manifest),
        "frames": len(rows),
        "eyes": len(cases),
        "patients": len(patients),
        "patient_table": patient_table,
    }


def _audit_patient_split(
    cfg: Mapping[str, Any], seed: int, path: Path, manifest_audit: Mapping[str, Any]
) -> dict[str, Any]:
    fields, rows = _read_csv(path)
    required = {"patient_id", "label_3class", "label_binary", "split", "seed"}
    if not required <= set(fields):
        raise ProtocolGateError(f"Seed {seed} patient split is missing fields: {sorted(required - set(fields))}")
    if len(rows) != cfg["dataset"]["patients"] or len({row["patient_id"] for row in rows}) != len(rows):
        raise ProtocolGateError(f"Seed {seed} does not assign each patient exactly once")
    if {int(row["seed"]) for row in rows} != {seed}:
        raise ProtocolGateError(f"Seed column mismatch in {path}")
    if {row["split"] for row in rows} != {"train", "validation", "test"}:
        raise ProtocolGateError(f"Seed {seed} has invalid/missing partitions")
    manifest_patients = manifest_audit["patient_table"]
    if {row["patient_id"] for row in rows} != set(manifest_patients):
        raise ProtocolGateError(f"Seed {seed} patient identities differ from the manifest")
    counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        patient = manifest_patients[row["patient_id"]]
        if int(row["label_3class"]) != patient["label_3class"] or int(row["label_binary"]) != patient["label_binary"]:
            raise ProtocolGateError(f"Seed {seed} contains a label mismatch for {row['patient_id']}")
        counts[(row["split"], row["label_3class"])] += 1
    partitions = ["train", "validation", "test"]
    expected_quotas = cfg["split_policy"]["class_quotas_train_validation_test"]
    for label, quotas in expected_quotas.items():
        for partition, expected in zip(partitions, quotas):
            if counts[(partition, label)] != expected:
                raise ProtocolGateError(
                    f"Seed {seed} quota mismatch for {partition}/class {label}: "
                    f"{counts[(partition, label)]} != {expected}"
                )
    return {
        "sha256": sha256_file(path),
        "patients": len(rows),
        "counts": {
            partition: {label: counts[(partition, label)] for label in sorted(expected_quotas)}
            for partition in partitions
        },
    }


def prepare_reused_splits(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Verify and byte-copy all prior patient assignments; never resample them."""

    validate_config(config_without_runtime(cfg))
    source_hashes = verify_locked_sources(cfg)
    out = output_root(cfg)
    receipt = stage_receipt_path(cfg, "prepare")
    if receipt.exists():
        verified = verify_stage_receipt(cfg, "prepare")
        return {"status": "already_complete", "receipt": str(receipt), "metadata": verified["metadata"]}
    if out.exists():
        unexpected = sorted(
            path.name for path in out.iterdir() if path.name != "orchestration"
        )
        if unexpected:
            raise ProtocolGateError(
                "Clean output has content but no valid prepare receipt; refusing implicit reuse: "
                + ", ".join(unexpected[:10])
            )
    manifest_audit = _audit_manifest(cfg)
    split_audits: dict[str, Any] = {}
    copied: list[Path] = []
    for seed in cfg["split_seeds"]:
        source_spec = cfg["split_policy"]["sources"][str(seed)]
        source = resolve_project_path(source_spec["path"], must_exist=True)
        audit = _audit_patient_split(cfg, seed, source, manifest_audit)
        if audit["sha256"] != source_spec["sha256"]:
            raise ProtocolGateError(f"Seed {seed} source changed after audit")
        destination = split_path(cfg, seed)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if sha256_file(destination) != source_spec["sha256"]:
                raise ProtocolGateError(f"Refusing to replace a different prepared split: {destination}")
        else:
            shutil.copyfile(source, destination)
        if destination.read_bytes() != source.read_bytes():
            raise ProtocolGateError(f"Byte-for-byte split copy failed for seed {seed}")
        copied.append(destination)
        split_audits[str(seed)] = {
            **audit,
            "source": source_spec["path"],
            "destination": str(destination.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "byte_identical": True,
        }
    provenance = out / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    snapshot = provenance / "config_strict_roi.snapshot.json"
    source_config = Path(cfg["_runtime"]["config_path"])
    if snapshot.exists() and sha256_file(snapshot) != sha256_file(source_config):
        raise ProtocolGateError("Refusing to replace a different prepared config snapshot")
    if not snapshot.exists():
        shutil.copyfile(source_config, snapshot)
    protocol_snapshot = provenance / "PROTOCOL.snapshot.md"
    if protocol_snapshot.exists() and sha256_file(protocol_snapshot) != protocol_sha256():
        raise ProtocolGateError("Refusing to replace a different protocol snapshot")
    if not protocol_snapshot.exists():
        shutil.copyfile(PROTOCOL_PATH, protocol_snapshot)
    fingerprint_path = provenance / "code_fingerprint.json"
    save_json_atomic(fingerprint_path, code_fingerprint())
    source_snapshot = _save_code_snapshot_atomic(provenance / "fingerprinted_sources.snapshot.zip")
    audit_path = out / "splits" / "audit.json"
    public_manifest = {key: value for key, value in manifest_audit.items() if key != "patient_table"}
    save_json_atomic(
        audit_path,
        {
            "study_id": cfg["study_id"],
            "created_utc": utc_now(),
            "source_hashes": source_hashes,
            "manifest": public_manifest,
            "splits": split_audits,
            "policy": "immutable byte-for-byte reuse; no resampling",
            "clean_run": {
                "output_namespace": cfg["clean_run"]["output_namespace"],
                "prior_failed_output": cfg["clean_run"]["prior_failed_output"],
                "upstream_model_or_roi_artifacts_imported": False,
                "resume_policy": cfg["clean_run"]["resume_policy"],
            },
            "test_rasters_decoded_or_inferred": False,
            "test_label_metadata_read_for_split_integrity": True,
            "test_labels_used_for_training_tuning_or_model_selection": False,
        },
    )
    # Freeze the common qualitative examples before any model output exists.
    # This reads only locked CSV membership/labels and stable identifiers; it
    # does not decode a test raster or inspect a prediction.
    from .data import build_inner_fold_assignment, split_frames
    from .qualitative import build_common_selection_lock

    qualitative_sources = {
        int(seed): split_frames(cfg, int(seed), "test")[1]
        for seed in cfg["split_seeds"]
    }
    qualitative_lock_path = out / "qualitative" / "common_selection_lock.json"
    qualitative_lock = build_common_selection_lock(
        qualitative_sources,
        qualitative_lock_path,
        seeds=cfg["split_seeds"],
        labels=(0, 1, 2),
        examples_per_label=int(
            cfg["reporting"]["segmentation_qualitative_examples_per_class_per_seed"]
        ),
    )
    inner_fold_paths: list[Path] = []
    for seed in cfg["split_seeds"]:
        build_inner_fold_assignment(cfg, int(seed))
        inner_fold_paths.append(out / "splits" / f"seed_{seed}_inner_folds.csv")
    artifacts: list[Path] = [
        *copied,
        *inner_fold_paths,
        snapshot,
        protocol_snapshot,
        fingerprint_path,
        source_snapshot,
        audit_path,
        qualitative_lock_path,
    ]
    receipt_path = write_stage_receipt(
        cfg,
        "prepare",
        artifacts=artifacts,
        metadata={
            "split_count": len(copied),
            "patient_split_unit": True,
            "regenerated": False,
            "test_images_loaded": False,
            "test_label_metadata_read_for_split_integrity_and_model_blind_qualitative_lock": True,
            "test_labels_used_for_training_tuning_or_model_selection": False,
            "qualitative_common_selection_model_blind": True,
            "qualitative_common_selection_count": len(
                qualitative_lock["selections"]
            ),
            "inner_patient_grouped_fold_assignment_count": len(inner_fold_paths),
            "clean_output_namespace": cfg["clean_run"]["output_namespace"],
            "prior_failed_output_imported": False,
            "test_stage_in_seed_launcher": False,
            "fingerprinted_source_snapshot_sha256": sha256_file(source_snapshot),
        },
    )
    return {
        "status": "complete",
        "receipt": str(receipt_path),
        "manifest": public_manifest,
        "splits": split_audits,
    }


def audit_state(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Read-only workflow/provenance audit suitable before any expensive run."""

    source_status: dict[str, Any]
    try:
        source_status = {"valid": True, "sha256": verify_locked_sources(cfg)}
    except Exception as error:  # Report all state rather than stopping at source failure.
        source_status = {"valid": False, "error": str(error)}
    stages: dict[str, Any] = {}
    for stage in STAGES:
        scopes: list[tuple[str | None, int | None]]
        if stage in GLOBAL_STAGES:
            scopes = [(None, None)]
        elif stage in PER_MODEL_STAGES:
            scopes = [(model, None) for model in cfg["models"]]
        else:
            scopes = [(model, seed) for model in cfg["models"] for seed in cfg["split_seeds"]]
        complete = 0
        invalid: list[str] = []
        for model, seed in scopes:
            path = stage_receipt_path(cfg, stage, model=model, seed=seed)
            if not path.exists():
                continue
            try:
                verify_stage_receipt(cfg, stage, model=model, seed=seed)
            except Exception as error:
                invalid.append(f"{model or 'global'}/{seed if seed is not None else '-'}: {error}")
            else:
                complete += 1
        stages[stage] = {"complete": complete, "expected": len(scopes), "invalid": invalid}
    test_opened = output_root(cfg) / "state" / "test_access_opened.json"
    ablation_test_opened = output_root(cfg) / "state" / "ablation_test_access_opened.json"
    return {
        "study_id": cfg["study_id"],
        "audited_utc": utc_now(),
        "source_locks": source_status,
        "stages": stages,
        "test_access_opened": test_opened.exists(),
        "test_access_path": str(test_opened),
        "ablation_test_access_opened": ablation_test_opened.exists(),
        "ablation_test_access_path": str(ablation_test_opened),
    }


def select_models_and_seeds(
    cfg: Mapping[str, Any], model: str, seed: int | None
) -> tuple[list[str], list[int]]:
    models = list(cfg["models"]) if model == "all" else [model]
    seeds = list(cfg["split_seeds"]) if seed is None else [seed]
    for name in models:
        _validate_model(cfg, name)
    for value in seeds:
        _validate_seed(cfg, value)
    return models, seeds


__all__ = [
    "GLOBAL_STAGES",
    "PER_MODEL_STAGES",
    "PER_UNIT_STAGES",
    "PROTOCOL_PATH",
    "ProtocolGateError",
    "STAGES",
    "assert_all_ablation_locks",
    "assert_all_primary_evaluations",
    "assert_all_primary_locks",
    "assert_prerequisites",
    "audit_state",
    "code_fingerprint",
    "open_ablation_test_access",
    "open_test_access",
    "output_root",
    "prepare_reused_splits",
    "protocol_sha256",
    "read_json",
    "save_json_atomic",
    "select_models_and_seeds",
    "split_path",
    "stage_receipt_path",
    "unit_root",
    "utc_now",
    "verify_stage_receipt",
    "write_stage_receipt",
]
