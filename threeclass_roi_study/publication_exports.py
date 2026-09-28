"""Validated CLAIM/TRIPOD+AI publication exports for the three-class study.

This module is deliberately downstream-only.  It reads immutable evaluation
artifacts after the global test gate, converts them into the exact table
contracts declared in :mod:`threeclass_roi_study.reporting`, validates every
row before writing, and records a hash manifest.  It never trains, calibrates,
selects, or changes a prediction.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image

from . import metrics
from .config import (
    CLASS_NAMES,
    CLASS_ORDER,
    canonical_sha256,
    resolve_project_path,
    sha256_file,
    upstream_artifact_paths,
)
from .protocol import (
    ProtocolGateError,
    assert_test_access_open,
    output_root,
    read_json,
    save_json_atomic,
    unit_root,
    utc_now,
    verify_deferred_test_import_receipt,
)
from .reporting import (
    OUTPUT_SCHEMA_VERSION,
    OUTPUT_TABLE_SCHEMAS,
    build_q1_metadata,
    publication_output_schemas,
    validate_output_rows,
    validate_q1_metadata,
    write_q1_metadata,
)


PROBABILITY_COLUMNS = ("probability_0", "probability_1", "probability_2")
PUBLICATION_TABLE_NAMES = tuple(OUTPUT_TABLE_SCHEMAS)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """Validate and atomically write a publication-contract CSV."""

    clean_rows = [metrics.clean_json(dict(row)) for row in rows]
    validated = validate_output_rows(path.name, clean_rows)
    columns = list(OUTPUT_TABLE_SCHEMAS[path.name]["columns"])
    table = pd.DataFrame(validated, columns=columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        os.close(descriptor)
        table.to_csv(temporary, index=False, lineterminator="\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def write_publication_contracts(cfg: Mapping[str, Any]) -> dict[str, Path]:
    """Persist the locked metadata and machine-readable table contracts."""

    provenance = output_root(cfg) / "provenance"
    metadata = build_q1_metadata(cfg)
    metadata_path = write_q1_metadata(provenance / "q1_metadata.json", metadata)
    schema_path = save_json_atomic(
        provenance / "publication_output_schemas.json",
        {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "tables": publication_output_schemas(),
        },
    )
    return {"q1_metadata": metadata_path, "publication_output_schemas": schema_path}


def _unit_metrics(
    table: pd.DataFrame, *, localized: Sequence[bool] | np.ndarray | None = None
) -> dict[str, Any]:
    return metrics.selective_multiclass_metrics(
        table["label_3class"].astype(int),
        table.loc[:, PROBABILITY_COLUMNS].to_numpy(float),
        evaluable=table["evaluable"].astype(bool).to_numpy(),
        localized_success=localized,
    )


def _nullable_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _source_optional_number(value: Any, *, context: str) -> float | None:
    """Read an optional JSON number without coercing strings or booleans."""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolGateError(f"{context} must be a finite number or null")
    number = float(value)
    if not math.isfinite(number):
        raise ProtocolGateError(f"{context} must be finite")
    return number


def _source_number(value: Any, *, context: str) -> float:
    number = _source_optional_number(value, context=context)
    if number is None:
        raise ProtocolGateError(f"{context} cannot be null")
    return number


def _source_integer(value: Any, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolGateError(f"{context} must be a genuine JSON integer")
    return int(value)


def _source_boolean(value: Any, *, context: str) -> bool:
    if not isinstance(value, bool):
        raise ProtocolGateError(f"{context} must be a genuine JSON boolean")
    return value


def _source_string(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolGateError(f"{context} must be a non-empty JSON string")
    return value


_BOOTSTRAP_LEVELS = {
    "patient_calibrated": "patient",
    "eye_calibrated": "eye",
}


def _uncertainty_rows(
    cfg: Mapping[str, Any],
    model: str,
    seed: int,
    strategy: str,
    evaluation: Path,
) -> list[dict[str, Any]]:
    """Flatten the immutable calibrated eye/patient bootstrap artifact."""

    source = evaluation / "bootstrap_patient_cluster_ci.json"
    payload = read_json(source)
    if not isinstance(payload, Mapping):
        raise ProtocolGateError(f"Malformed bootstrap artifact: {source}")
    missing_levels = set(_BOOTSTRAP_LEVELS) - set(payload)
    if missing_levels:
        raise ProtocolGateError(
            f"Bootstrap artifact lacks calibrated levels: {sorted(missing_levels)}"
        )
    locked_draws = int(cfg["statistics"]["bootstrap_draws"])
    locked_confidence = float(cfg["statistics"]["confidence_level"])
    if locked_draws != 5000 or locked_confidence != 0.95:
        raise ProtocolGateError("Publication uncertainty requires 5000 draws at 95% confidence")
    bootstrap_offset = int(cfg["training"]["seed_offsets"]["bootstrap"])
    rows: list[dict[str, Any]] = []
    for artifact_level, level in _BOOTSTRAP_LEVELS.items():
        records = payload[artifact_level]
        if not isinstance(records, Mapping) or not records:
            raise ProtocolGateError(
                f"Bootstrap level {artifact_level!r} is empty or malformed"
            )
        for metric_path in sorted(records):
            record = records[metric_path]
            if not isinstance(record, Mapping):
                raise ProtocolGateError(
                    f"Bootstrap metric {artifact_level}.{metric_path} is malformed"
                )
            required = {
                "estimate",
                "low",
                "high",
                "bootstrap_se",
                "valid_draws",
                "requested_draws",
                "confidence",
                "requested_method",
                "method",
                "fallback_used",
                "fallback_reason",
                "resampling_unit",
                "stratified_by_label",
                "stratification",
                "bootstrap_seed",
            }
            missing = required - set(record)
            if missing:
                raise ProtocolGateError(
                    f"Bootstrap metric {artifact_level}.{metric_path} lacks {sorted(missing)}"
                )
            context = f"{artifact_level}.{metric_path}"
            requested_draws = _source_integer(
                record["requested_draws"], context=f"{context}.requested_draws"
            )
            valid_draws = _source_integer(
                record["valid_draws"], context=f"{context}.valid_draws"
            )
            confidence = _source_number(
                record["confidence"], context=f"{context}.confidence"
            )
            if requested_draws != locked_draws or confidence != locked_confidence:
                raise ProtocolGateError(
                    f"Bootstrap metric {artifact_level}.{metric_path} violates the locked design"
                )
            resampling_unit = _source_string(
                record["resampling_unit"], context=f"{context}.resampling_unit"
            )
            if resampling_unit != "whole_patient_cluster":
                raise ProtocolGateError("Bootstrap resampling unit is not the whole patient")
            stratified_by_label = _source_boolean(
                record["stratified_by_label"],
                context=f"{context}.stratified_by_label",
            )
            if not stratified_by_label:
                raise ProtocolGateError("Bootstrap was not stratified by diagnosis label")
            stratification = _source_string(
                record["stratification"], context=f"{context}.stratification"
            )
            if stratification != "three_class_patient_label":
                raise ProtocolGateError("Bootstrap stratification metadata changed")
            requested_method = _source_string(
                record["requested_method"], context=f"{context}.requested_method"
            )
            interval_method = _source_string(
                record["method"], context=f"{context}.method"
            )
            if requested_method != "bca" or interval_method not in {"bca", "percentile"}:
                raise ProtocolGateError(f"Bootstrap interval method is invalid for {context}")
            fallback_used = _source_boolean(
                record["fallback_used"], context=f"{context}.fallback_used"
            )
            if fallback_used != (interval_method != requested_method):
                raise ProtocolGateError(
                    f"Bootstrap fallback metadata is inconsistent for {metric_path}"
                )
            fallback_detail = record.get("fallback_reason")
            if fallback_detail is not None:
                fallback_detail = _source_string(
                    fallback_detail, context=f"{context}.fallback_reason"
                )
            if fallback_used and fallback_detail is None:
                raise ProtocolGateError(
                    f"Bootstrap metric {metric_path} does not explain its fallback"
                )
            if not fallback_used and fallback_detail:
                raise ProtocolGateError(
                    f"Bootstrap metric {metric_path} reports a spurious fallback reason"
                )
            bootstrap_seed = _source_integer(
                record["bootstrap_seed"], context=f"{context}.bootstrap_seed"
            )
            expected_seed = int(seed) + bootstrap_offset + int(level == "eye")
            if bootstrap_seed != expected_seed:
                raise ProtocolGateError(
                    f"Bootstrap RNG seed disagrees with the locked {level} evaluation seed"
                )
            rows.append(
                {
                    "model": model,
                    "seed": int(seed),
                    "classifier_strategy": strategy,
                    "level": level,
                    "probability_scale": "calibrated",
                    "metric_path": str(metric_path),
                    "estimate": _source_optional_number(
                        record["estimate"], context=f"{context}.estimate"
                    ),
                    "ci_95_low": _source_optional_number(
                        record["low"], context=f"{context}.low"
                    ),
                    "ci_95_high": _source_optional_number(
                        record["high"], context=f"{context}.high"
                    ),
                    "bootstrap_se": _source_optional_number(
                        record["bootstrap_se"], context=f"{context}.bootstrap_se"
                    ),
                    "confidence_level": confidence,
                    "requested_method": requested_method,
                    "interval_method": interval_method,
                    "fallback_used": fallback_used,
                    "fallback_detail": fallback_detail,
                    "valid_draws": valid_draws,
                    "requested_draws": requested_draws,
                    "resampling_unit": resampling_unit,
                    "stratified_by_label": stratified_by_label,
                    "stratification": stratification,
                    "bootstrap_seed": bootstrap_seed,
                }
            )
    return rows


def _patient_prediction_rows(
    cfg: Mapping[str, Any],
    model: str,
    seed: int,
    strategy: str,
    raw: pd.DataFrame,
    calibrated: pd.DataFrame,
    eyes: pd.DataFrame,
    calibration: Mapping[str, Any],
    localized_by_patient: Mapping[str, bool],
) -> list[dict[str, Any]]:
    raw = raw.sort_values("patient_id", kind="stable").reset_index(drop=True)
    calibrated = calibrated.sort_values("patient_id", kind="stable").reset_index(drop=True)
    if raw["patient_id"].astype(str).tolist() != calibrated["patient_id"].astype(str).tolist():
        raise ProtocolGateError("Raw/calibrated patient identities are not identical")
    if raw["label_3class"].astype(int).tolist() != calibrated["label_3class"].astype(int).tolist():
        raise ProtocolGateError("Raw/calibrated patient reference labels changed")
    eye_lookup: dict[tuple[str, str], pd.Series] = {}
    for _, eye in eyes.iterrows():
        key = (str(eye["patient_id"]), str(eye["side"]).upper())
        if key in eye_lookup:
            raise ProtocolGateError(f"Duplicate eye identity in publication export: {key}")
        eye_lookup[key] = eye

    rows: list[dict[str, Any]] = []
    status = str(calibration["patient"]["status"])
    for index in range(len(raw)):
        raw_row = raw.iloc[index]
        calibrated_row = calibrated.iloc[index]
        patient_id = str(raw_row["patient_id"])
        right = eye_lookup.get((patient_id, "SAG"))
        left = eye_lookup.get((patient_id, "SOL"))
        if right is None or left is None:
            raise ProtocolGateError(f"Both SAG/right and SOL/left eyes are required for {patient_id}")
        evaluable = bool(calibrated_row["evaluable"])
        localized = bool(localized_by_patient[patient_id])
        predicted_label = int(calibrated_row["prediction"])
        true_label = int(raw_row["label_3class"])
        rows.append(
            {
                "study_id": str(cfg["study_id"]),
                "model": model,
                "seed": int(seed),
                "classifier_strategy": strategy,
                "patient_id": patient_id,
                "true_label": true_label,
                "p_normal_raw": _nullable_number(raw_row[PROBABILITY_COLUMNS[0]]),
                "p_papilledema_raw": _nullable_number(raw_row[PROBABILITY_COLUMNS[1]]),
                "p_pseudopapilledema_raw": _nullable_number(raw_row[PROBABILITY_COLUMNS[2]]),
                "p_normal_calibrated": _nullable_number(calibrated_row[PROBABILITY_COLUMNS[0]]),
                "p_papilledema_calibrated": _nullable_number(calibrated_row[PROBABILITY_COLUMNS[1]]),
                "p_pseudopapilledema_calibrated": _nullable_number(calibrated_row[PROBABILITY_COLUMNS[2]]),
                "predicted_label": predicted_label,
                "abstained": not evaluable,
                "abstention_reason": (
                    None
                    if evaluable
                    else str(calibrated_row.get("abstention_reason") or "structural_roi_gate")
                ),
                "right_eye_evaluable": bool(right["evaluable"]),
                "left_eye_evaluable": bool(left["evaluable"]),
                "right_valid_frames": int(right["n_valid_frames"]),
                "left_valid_frames": int(left["n_valid_frames"]),
                "calibration_status": status,
                "localized": localized,
                "localized_diagnostic_success": bool(
                    evaluable and localized and predicted_label == true_label
                ),
                "prior_test_use_disclosed": True,
            }
        )
    return rows


def _patient_metric_row(
    cfg: Mapping[str, Any],
    model: str,
    seed: int,
    strategy: str,
    scale: str,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    conditional = result["conditional"]
    failure = result["failure_aware"]
    localized = result.get("localized_diagnostic_success")
    if not isinstance(localized, Mapping):
        raise ProtocolGateError("Patient metrics lack mandatory localized diagnostic success")
    return {
        "study_id": str(cfg["study_id"]),
        "model": model,
        "seed": int(seed),
        "classifier_strategy": strategy,
        "probability_scale": scale,
        "n_intended": int(result["n_total"]),
        "n_evaluable": int(result["n_covered"]),
        "n_abstained": int(result["n_abstain"]),
        "coverage": float(result["coverage"]),
        "failure_aware_balanced_accuracy": _nullable_number(result["failure_aware_balanced_accuracy"]),
        "conditional_balanced_accuracy": _nullable_number(conditional.get("balanced_accuracy")),
        "accuracy": _nullable_number(result["failure_aware_accuracy"]),
        "macro_f1": _nullable_number(failure.get("macro_f1")),
        "multiclass_mcc": _nullable_number(failure.get("multiclass_mcc")),
        "macro_ovr_auroc": _nullable_number(conditional.get("macro_auroc")),
        "macro_ovr_average_precision": _nullable_number(conditional.get("macro_average_precision")),
        "multiclass_nll": _nullable_number(conditional.get("multiclass_nll")),
        "multiclass_brier": _nullable_number(conditional.get("multiclass_brier")),
        "aurc": _nullable_number(result.get("failure_aware_aurc")),
        "localized_rate": float(localized["localized_rate"]),
        "localized_diagnostic_success_rate": float(localized["rate"]),
    }


def _confusion_rows(
    model: str, seed: int, strategy: str, result: Mapping[str, Any]
) -> list[dict[str, Any]]:
    matrix = np.asarray(result["confusion_matrix_3x4"], dtype=int)
    if matrix.shape != (3, 4):
        raise ProtocolGateError("Primary patient confusion matrix is not 3x4")
    return [
        {
            "model": model,
            "seed": int(seed),
            "classifier_strategy": strategy,
            "true_label": int(actual),
            "predicted_label": int(predicted),
            "count": int(matrix[actual, predicted]),
        }
        for actual in CLASS_ORDER
        for predicted in (0, 1, 2, 3)
    ]


def _classwise_rows(
    model: str, seed: int, strategy: str, result: Mapping[str, Any]
) -> list[dict[str, Any]]:
    conditional = result["conditional"].get("per_class", {})
    failure = result["failure_aware"].get("per_class", {})
    rows: list[dict[str, Any]] = []
    for class_label in CLASS_ORDER:
        class_name = CLASS_NAMES[class_label]
        intended = int(result["failure_aware"]["per_class"][class_name]["support"])
        coverage = float(result["class_conditional_coverage"][class_name])
        failure_values = failure[class_name]
        conditional_values = conditional.get(class_name, {})
        rows.append(
            {
                "model": model,
                "seed": int(seed),
                "classifier_strategy": strategy,
                "class_label": int(class_label),
                "class_name": class_name,
                "n_intended": intended,
                "n_evaluable": int(round(intended * coverage)),
                "coverage": coverage,
                "failure_aware_recall": _nullable_number(failure_values.get("recall")),
                "conditional_recall": _nullable_number(conditional_values.get("recall")),
                "precision": _nullable_number(failure_values.get("precision")),
                "f1": _nullable_number(failure_values.get("f1")),
                "ovr_auroc": _nullable_number(conditional_values.get("auroc_ovr")),
                "ovr_average_precision": _nullable_number(
                    conditional_values.get("average_precision_ovr")
                ),
            }
        )
    return rows


def _calibration_metric_rows(
    model: str,
    seed: int,
    strategy: str,
    level: str,
    scale: str,
    result: Mapping[str, Any],
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    conditional = result["conditional"]
    calibration_record = calibration[level]
    if scale == "raw":
        status = "not_applicable_raw"
        temperature: float | None = None
    else:
        status = str(calibration_record["status"])
        temperature = _nullable_number(calibration_record.get("temperature"))
    return {
        "model": model,
        "seed": int(seed),
        "classifier_strategy": strategy,
        "level": level,
        "probability_scale": scale,
        "status": status,
        "temperature": temperature,
        "multiclass_nll": _nullable_number(conditional.get("multiclass_nll")),
        "multiclass_brier": _nullable_number(conditional.get("multiclass_brier")),
        "top_label_ece": _nullable_number(conditional.get("top_label_ece")),
        "classwise_ece_macro": _nullable_number(conditional.get("macro_classwise_ece")),
    }


def _risk_rows(
    model: str, seed: int, strategy: str, patient_table: pd.DataFrame
) -> list[dict[str, Any]]:
    curve = metrics.multiclass_risk_coverage_curve(
        patient_table["label_3class"].astype(int),
        patient_table.loc[:, PROBABILITY_COLUMNS].to_numpy(float),
        evaluable=patient_table["evaluable"].astype(bool).to_numpy(),
        include_abstentions_as_failures=True,
    )
    return [
        {
            "model": model,
            "seed": int(seed),
            "classifier_strategy": strategy,
            "rank": int(row["rank"]),
            "coverage": float(row["coverage"]),
            "selective_risk": float(row["risk"]),
            "confidence_definition": "maximum_calibrated_patient_probability",
        }
        for _, row in curve.iterrows()
    ]


def _strict_bool_series(values: pd.Series, *, name: str) -> pd.Series:
    if values.dtype == bool:
        return values.astype(bool)
    mapped = values.astype(str).str.strip().str.lower().map({"true": True, "false": False})
    if mapped.isna().any():
        raise ProtocolGateError(f"Malformed boolean values in {name}")
    return mapped.astype(bool)


def _atomic_plain_csv(path: Path, table: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        os.close(descriptor)
        table.to_csv(temporary, index=False, lineterminator="\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def segmentation_only_audit(
    cfg: Mapping[str, Any], model: str, seed: int
) -> tuple[pd.DataFrame, Path]:
    """Build a clean post-gate segmentation audit without binary outputs.

    Only the allowlisted test ROI identity/audit columns are read.  The selected
    mask is loaded from each hash-verified upstream audit NPZ, while the
    reference mask is loaded from the hash-verified dataset manifest.  No
    binary logits, probabilities, thresholds, predictions, or calibrators are
    opened or propagated.
    """

    assert_test_access_open(cfg)
    deferred = verify_deferred_test_import_receipt(cfg, model, seed)
    test_index_path = upstream_artifact_paths(cfg, model, seed, phase="test")[
        "test_roi_index"
    ]
    manifest_path = resolve_project_path(str(cfg["dataset"]["manifest"]), must_exist=True)
    test_records = [
        item
        for item in deferred["artifacts"]
        if item.get("role") == "test_roi_index"
    ]
    if len(test_records) != 1 or sha256_file(test_index_path) != test_records[0].get("sha256"):
        raise ProtocolGateError("Deferred test ROI index receipt mismatch")

    destination = unit_root(cfg, model, seed) / "evaluation" / "segmentation_only_audit.csv"
    receipt_path = destination.with_suffix(".receipt.json")
    if receipt_path.is_file():
        receipt = read_json(receipt_path)
        if (
            receipt.get("study_id") != cfg["study_id"]
            or receipt.get("config_sha256") != cfg["config_sha256"]
            or receipt.get("model") != model
            or receipt.get("seed") != int(seed)
            or receipt.get("contains_binary_classifier_outputs") is not False
            or not destination.is_file()
            or receipt.get("audit_sha256") != sha256_file(destination)
            or receipt.get("test_roi_index_sha256") != sha256_file(test_index_path)
            or receipt.get("dataset_manifest_sha256") != sha256_file(manifest_path)
        ):
            raise ProtocolGateError("Segmentation-only audit receipt mismatch")
        referenced = receipt.get("referenced_artifacts")
        if not isinstance(referenced, list) or not referenced:
            raise ProtocolGateError("Segmentation-only audit lacks referenced-artifact receipts")
        for record in referenced:
            source = Path(str(record.get("path", "")))
            if not source.is_file() or sha256_file(source) != record.get("sha256"):
                raise ProtocolGateError(
                    f"Segmentation-only audit source artifact changed: {source}"
                )
        frame = pd.read_csv(
            destination,
            dtype={"patient_id": str, "case_id": str, "frame_id": str},
        )
        frame["segmentation_roi_valid"] = _strict_bool_series(
            frame["segmentation_roi_valid"], name="segmentation_roi_valid"
        )
        return frame, destination

    allowed_index_columns = [
        "patient_id",
        "case_id",
        "side",
        "frame_id",
        "label_3class",
        "frame_identity_sha256",
        "roi_valid",
        "abstention_reason",
        "audit_path",
        "audit_sha256",
    ]
    index = pd.read_csv(
        test_index_path,
        usecols=allowed_index_columns,
        dtype={"patient_id": str, "case_id": str, "frame_id": str},
    )
    index["roi_valid"] = _strict_bool_series(index["roi_valid"], name="roi_valid")
    identity = ["patient_id", "case_id", "side", "frame_id"]
    if index.duplicated(identity).any():
        raise ProtocolGateError("Test ROI index repeats a frame identity")

    if sha256_file(manifest_path) != cfg["dataset"]["manifest_sha256"]:
        raise ProtocolGateError("Dataset manifest hash mismatch during segmentation audit")
    manifest_columns = [
        *identity,
        "label_3class",
        "output_mask",
        "output_mask_sha256",
    ]
    manifest = pd.read_csv(
        manifest_path,
        usecols=manifest_columns,
        dtype={"patient_id": str, "case_id": str, "frame_id": str},
    )
    if manifest.duplicated(identity).any():
        raise ProtocolGateError("Dataset manifest repeats a frame identity")
    merged = index.merge(
        manifest,
        on=identity,
        how="left",
        validate="one_to_one",
        suffixes=("_roi", "_manifest"),
    )
    if merged[["output_mask", "output_mask_sha256"]].isna().any().any():
        raise ProtocolGateError("Test ROI index identity is missing from the dataset manifest")
    if not np.array_equal(
        merged["label_3class_roi"].astype(int),
        merged["label_3class_manifest"].astype(int),
    ):
        raise ProtocolGateError("Test ROI and manifest three-class labels disagree")

    threshold = float(cfg["evaluation"]["localized_success"]["frame_hit_iou_threshold"])
    dataset_root = str(cfg["dataset"]["root"])
    records: list[dict[str, Any]] = []
    referenced_inventory: list[dict[str, str]] = []
    for row in merged.itertuples(index=False):
        audit_path = resolve_project_path(str(row.audit_path), must_exist=True)
        if sha256_file(audit_path) != str(row.audit_sha256):
            raise ProtocolGateError(f"Segmentation audit NPZ hash mismatch: {audit_path}")
        reference_path = resolve_project_path(
            str(Path(dataset_root) / str(row.output_mask)), must_exist=True
        )
        if sha256_file(reference_path) != str(row.output_mask_sha256):
            raise ProtocolGateError(f"Reference mask hash mismatch: {reference_path}")
        with np.load(audit_path, allow_pickle=False) as payload:
            required_keys = {"selected_mask", "frame_identity_sha256"}
            if not required_keys <= set(payload.files):
                raise ProtocolGateError(f"Segmentation audit NPZ is incomplete: {audit_path}")
            predicted = payload["selected_mask"].astype(bool, copy=False)
            embedded_identity = bytes(payload["frame_identity_sha256"]).hex()
        if embedded_identity != str(row.frame_identity_sha256):
            raise ProtocolGateError(f"Segmentation audit identity mismatch: {audit_path}")
        with Image.open(reference_path) as image:
            reference = np.asarray(image.convert("L")) > 0
        if predicted.shape != reference.shape:
            raise ProtocolGateError("Predicted/reference segmentation mask shape mismatch")
        intersection = int(np.logical_and(predicted, reference).sum())
        predicted_area = int(predicted.sum())
        reference_area = int(reference.sum())
        union = predicted_area + reference_area - intersection
        dice = (
            float(2 * intersection / (predicted_area + reference_area))
            if predicted_area + reference_area
            else 1.0
        )
        iou = float(intersection / union) if union else 1.0
        records.append(
            {
                "patient_id": str(row.patient_id),
                "case_id": str(row.case_id),
                "side": str(row.side),
                "frame_id": str(row.frame_id),
                "label_3class": int(row.label_3class_roi),
                "frame_identity_sha256": str(row.frame_identity_sha256),
                "segmentation_roi_valid": bool(row.roi_valid),
                "segmentation_abstention_reason": (
                    "" if pd.isna(row.abstention_reason) else str(row.abstention_reason)
                ),
                "dice": dice,
                "iou": iou,
                "roi_hit": bool(iou >= threshold),
                "predicted_area_pixels": predicted_area,
                "reference_area_pixels": reference_area,
                "gt_used_for_inference_selection_or_abstention": False,
            }
        )
        referenced_inventory.extend(
            [
                {
                    "role": "selected_mask_audit_npz",
                    "path": str(audit_path.resolve()),
                    "sha256": str(row.audit_sha256),
                },
                {
                    "role": "reference_standard_mask",
                    "path": str(reference_path.resolve()),
                    "sha256": str(row.output_mask_sha256),
                },
            ]
        )
    frame = pd.DataFrame(records).sort_values(identity, kind="stable").reset_index(drop=True)
    _atomic_plain_csv(destination, frame)
    receipt = {
        "schema_version": 1,
        "stage": "post_gate_segmentation_only_audit",
        "study_id": cfg["study_id"],
        "config_sha256": cfg["config_sha256"],
        "model": model,
        "seed": int(seed),
        "test_roi_index": str(test_index_path.resolve()),
        "test_roi_index_sha256": sha256_file(test_index_path),
        "dataset_manifest": str(manifest_path.resolve()),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "referenced_artifact_inventory_sha256": canonical_sha256(referenced_inventory),
        "referenced_artifacts": referenced_inventory,
        "audit_path": str(destination.resolve()),
        "audit_sha256": sha256_file(destination),
        "rows": int(len(frame)),
        "contains_binary_classifier_outputs": False,
        "gt_used_for_inference_selection_or_abstention": False,
        "localized_success_rule": dict(cfg["evaluation"]["localized_success"]),
        "created_utc": utc_now(),
    }
    save_json_atomic(receipt_path, receipt)
    return frame, destination


def _source_segmentation_frames(cfg: Mapping[str, Any], model: str, seed: int) -> pd.DataFrame:
    """Compatibility wrapper returning only the clean segmentation audit table."""

    frame, _ = segmentation_only_audit(cfg, model, seed)
    return frame


def _localization_maps(
    cfg: Mapping[str, Any], model: str, seed: int
) -> tuple[dict[tuple[str, str, str], bool], dict[str, bool]]:
    """Derive audit-only eye/patient localization from immutable source hits."""

    specification = cfg["evaluation"]["localized_success"]
    if (
        specification["frame_hit_source"] != "immutable_source_evaluation_roi_hit"
        or float(specification["frame_hit_iou_threshold"]) != 0.5
        or specification["require_both_eyes"] is not True
        or specification["gt_used_for_retrospective_evaluation_only"] is not True
        or specification["gt_used_for_inference_selection_or_abstention"] is not False
    ):
        raise ProtocolGateError("Localized-success source, threshold, or audit-only role changed")
    minimum = int(specification["minimum_hit_frames_per_eye"])
    frame = _source_segmentation_frames(cfg, model, seed)
    roi_hit = pd.to_numeric(frame["roi_hit"], errors="coerce").fillna(0).astype(bool)
    valid_hit = roi_hit & frame["segmentation_roi_valid"].astype(bool)
    eye_map: dict[tuple[str, str, str], bool] = {}
    expected_frames = int(cfg["dataset"]["frames_per_eye"])
    for identity, index in frame.groupby(
        ["patient_id", "case_id", "side"], sort=True
    ).groups.items():
        if len(index) != expected_frames:
            raise ProtocolGateError(
                f"Localization eye does not have {expected_frames} frames: {identity}"
            )
        eye_map[tuple(map(str, identity))] = bool(valid_hit.loc[index].sum() >= minimum)
    by_patient: dict[str, list[tuple[str, bool]]] = {}
    for (patient_id, _case_id, side), localized in eye_map.items():
        by_patient.setdefault(patient_id, []).append((side.upper(), localized))
    patient_map: dict[str, bool] = {}
    for patient_id, eyes in by_patient.items():
        if len(eyes) != 2 or {side for side, _ in eyes} != {"SAG", "SOL"}:
            raise ProtocolGateError(f"Localization audit lacks both eyes for {patient_id}")
        patient_map[patient_id] = bool(all(localized for _, localized in eyes))
    return eye_map, patient_map


def localization_vectors(
    cfg: Mapping[str, Any],
    model: str,
    seed: int,
    eyes: pd.DataFrame,
    patients: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Align immutable retrospective localization to evaluation table order."""

    eye_map, patient_map = _localization_maps(cfg, model, seed)
    eye_values: list[bool] = []
    for row in eyes.itertuples(index=False):
        key = (str(row.patient_id), str(row.case_id), str(row.side))
        if key not in eye_map:
            raise ProtocolGateError(f"Missing localization eye identity: {key}")
        eye_values.append(eye_map[key])
    patient_values: list[bool] = []
    for patient_id in patients["patient_id"].astype(str):
        if patient_id not in patient_map:
            raise ProtocolGateError(f"Missing localization patient identity: {patient_id}")
        patient_values.append(patient_map[patient_id])
    return np.asarray(eye_values, dtype=bool), np.asarray(patient_values, dtype=bool)


_EMPTY_TINY = frozenset({"empty", "tiny"})
_OVERSEGMENTED_EDGE = frozenset({"border", "edge", "oversegmented", "oversegmented_or_edge"})
_AMBIGUOUS_MULTI = frozenset({"multi_ambiguous", "ambiguous_multi"})


def _reason_flags(frame: pd.DataFrame) -> pd.DataFrame:
    reason = frame["segmentation_abstention_reason"].fillna("").astype(str).str.strip().str.lower()
    return pd.DataFrame(
        {
            "empty_or_tiny": reason.isin(_EMPTY_TINY),
            "oversegmented_or_edge": reason.isin(_OVERSEGMENTED_EDGE),
            "ambiguous_multi": reason.isin(_AMBIGUOUS_MULTI),
        },
        index=frame.index,
    )


def _mean_or_none(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce")
    return float(numeric.mean()) if numeric.notna().any() else None


def _segmentation_rows(cfg: Mapping[str, Any], model: str, seed: int) -> list[dict[str, Any]]:
    frame = _source_segmentation_frames(cfg, model, seed)
    flags = _reason_flags(frame)
    for column in flags:
        frame[column] = flags[column]
    minimum = int(cfg["aggregation"]["minimum_valid_frames_per_eye"])
    rows: list[dict[str, Any]] = []
    for class_label in CLASS_ORDER:
        class_frames = frame.loc[frame["label_3class"].astype(int) == class_label].copy()
        if class_frames.empty:
            raise ProtocolGateError(f"Segmentation audit lacks class {class_label}")
        valid = class_frames["segmentation_roi_valid"].astype(bool)
        rows.append(
            {
                "model": model,
                "seed": int(seed),
                "class_label": int(class_label),
                "class_name": CLASS_NAMES[class_label],
                "level": "frame",
                "n_intended": int(len(class_frames)),
                "roi_coverage": float(valid.mean()),
                "dice": _mean_or_none(class_frames["dice"]),
                "iou": _mean_or_none(class_frames["iou"]),
                "empty_or_tiny": int(class_frames["empty_or_tiny"].sum()),
                "oversegmented_or_edge": int(class_frames["oversegmented_or_edge"].sum()),
                "ambiguous_multi": int(class_frames["ambiguous_multi"].sum()),
            }
        )

        eye_records: list[dict[str, Any]] = []
        for (patient_id, case_id, side), group in class_frames.groupby(
            ["patient_id", "case_id", "side"], sort=True
        ):
            eye_records.append(
                {
                    "patient_id": str(patient_id),
                    "case_id": str(case_id),
                    "side": str(side),
                    "evaluable": bool(group["segmentation_roi_valid"].astype(bool).sum() >= minimum),
                    "dice": _mean_or_none(group["dice"]),
                    "iou": _mean_or_none(group["iou"]),
                    "empty_or_tiny": bool(group["empty_or_tiny"].any()),
                    "oversegmented_or_edge": bool(group["oversegmented_or_edge"].any()),
                    "ambiguous_multi": bool(group["ambiguous_multi"].any()),
                }
            )
        eyes = pd.DataFrame(eye_records)
        rows.append(
            {
                "model": model,
                "seed": int(seed),
                "class_label": int(class_label),
                "class_name": CLASS_NAMES[class_label],
                "level": "eye",
                "n_intended": int(len(eyes)),
                "roi_coverage": float(eyes["evaluable"].mean()),
                "dice": _mean_or_none(eyes["dice"]),
                "iou": _mean_or_none(eyes["iou"]),
                "empty_or_tiny": int(eyes["empty_or_tiny"].sum()),
                "oversegmented_or_edge": int(eyes["oversegmented_or_edge"].sum()),
                "ambiguous_multi": int(eyes["ambiguous_multi"].sum()),
            }
        )

        patient_records: list[dict[str, Any]] = []
        for patient_id, group in eyes.groupby("patient_id", sort=True):
            if len(group) != 2 or set(group["side"].astype(str).str.upper()) != {"SAG", "SOL"}:
                raise ProtocolGateError(f"Patient {patient_id} lacks the locked two-eye structure")
            patient_records.append(
                {
                    "evaluable": bool(group["evaluable"].astype(bool).all()),
                    "dice": _mean_or_none(group["dice"]),
                    "iou": _mean_or_none(group["iou"]),
                    "empty_or_tiny": bool(group["empty_or_tiny"].any()),
                    "oversegmented_or_edge": bool(group["oversegmented_or_edge"].any()),
                    "ambiguous_multi": bool(group["ambiguous_multi"].any()),
                }
            )
        patients = pd.DataFrame(patient_records)
        rows.append(
            {
                "model": model,
                "seed": int(seed),
                "class_label": int(class_label),
                "class_name": CLASS_NAMES[class_label],
                "level": "patient",
                "n_intended": int(len(patients)),
                "roi_coverage": float(patients["evaluable"].mean()),
                "dice": _mean_or_none(patients["dice"]),
                "iou": _mean_or_none(patients["iou"]),
                "empty_or_tiny": int(patients["empty_or_tiny"].sum()),
                "oversegmented_or_edge": int(patients["oversegmented_or_edge"].sum()),
                "ambiguous_multi": int(patients["ambiguous_multi"].sum()),
            }
        )
    return rows


def _numeric_sum_count(values: pd.Series) -> tuple[float, float]:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric.to_numpy(dtype=float))]
    return float(finite.sum()), float(len(finite))


def _segmentation_patient_contributions(
    cfg: Mapping[str, Any], model: str, seed: int
) -> pd.DataFrame:
    """Aggregate nested segmentation observations to patient contributions.

    The resulting numerator/denominator pairs preserve each published point
    estimand while ensuring that every bootstrap draw samples patients, never
    individual frames or eyes.
    """

    frame = _source_segmentation_frames(cfg, model, seed).copy()
    required = {
        "patient_id",
        "case_id",
        "side",
        "label_3class",
        "segmentation_roi_valid",
        "dice",
        "iou",
    }
    missing = required - set(frame)
    if missing:
        raise ProtocolGateError(
            f"Segmentation audit lacks bootstrap columns: {sorted(missing)}"
        )
    frame["segmentation_roi_valid"] = _strict_bool_series(
        frame["segmentation_roi_valid"], name="segmentation_roi_valid"
    )
    minimum = int(cfg["aggregation"]["minimum_valid_frames_per_eye"])
    expected_frames = int(cfg["dataset"]["frames_per_eye"])
    records: list[dict[str, Any]] = []
    for patient_id, patient_frames in frame.groupby("patient_id", sort=True):
        labels = pd.to_numeric(patient_frames["label_3class"], errors="raise").astype(int)
        if labels.nunique() != 1 or int(labels.iloc[0]) not in CLASS_ORDER:
            raise ProtocolGateError(
                f"Patient {patient_id} has inconsistent segmentation class labels"
            )
        patient_valid = patient_frames["segmentation_roi_valid"].astype(bool)
        frame_dice_numerator, frame_dice_denominator = _numeric_sum_count(
            patient_frames["dice"]
        )
        frame_iou_numerator, frame_iou_denominator = _numeric_sum_count(
            patient_frames["iou"]
        )
        eye_records: list[dict[str, Any]] = []
        for (case_id, side), eye_frames in patient_frames.groupby(
            ["case_id", "side"], sort=True
        ):
            if len(eye_frames) != expected_frames:
                raise ProtocolGateError(
                    f"Segmentation bootstrap eye {(patient_id, case_id, side)} "
                    f"does not have {expected_frames} frames"
                )
            dice_numerator, dice_denominator = _numeric_sum_count(eye_frames["dice"])
            iou_numerator, iou_denominator = _numeric_sum_count(eye_frames["iou"])
            eye_records.append(
                {
                    "side": str(side).upper(),
                    "evaluable": bool(
                        eye_frames["segmentation_roi_valid"].astype(bool).sum()
                        >= minimum
                    ),
                    "dice": (
                        dice_numerator / dice_denominator
                        if dice_denominator
                        else np.nan
                    ),
                    "iou": (
                        iou_numerator / iou_denominator
                        if iou_denominator
                        else np.nan
                    ),
                }
            )
        eyes = pd.DataFrame(eye_records)
        if len(eyes) != 2 or set(eyes["side"]) != {"SAG", "SOL"}:
            raise ProtocolGateError(
                f"Patient {patient_id} lacks the locked two-eye structure"
            )
        eye_dice = pd.to_numeric(eyes["dice"], errors="coerce")
        eye_iou = pd.to_numeric(eyes["iou"], errors="coerce")
        finite_eye_dice = eye_dice[np.isfinite(eye_dice.to_numpy(dtype=float))]
        finite_eye_iou = eye_iou[np.isfinite(eye_iou.to_numpy(dtype=float))]
        patient_dice = (
            float(finite_eye_dice.mean()) if len(finite_eye_dice) else np.nan
        )
        patient_iou = float(finite_eye_iou.mean()) if len(finite_eye_iou) else np.nan
        record = {
            "patient_id": str(patient_id),
            "class_label": int(labels.iloc[0]),
            "frame_roi_coverage_numerator": float(patient_valid.sum()),
            "frame_roi_coverage_denominator": float(len(patient_frames)),
            "frame_dice_numerator": frame_dice_numerator,
            "frame_dice_denominator": frame_dice_denominator,
            "frame_iou_numerator": frame_iou_numerator,
            "frame_iou_denominator": frame_iou_denominator,
            "eye_roi_coverage_numerator": float(eyes["evaluable"].astype(bool).sum()),
            "eye_roi_coverage_denominator": float(len(eyes)),
            "eye_dice_numerator": float(finite_eye_dice.sum()),
            "eye_dice_denominator": float(len(finite_eye_dice)),
            "eye_iou_numerator": float(finite_eye_iou.sum()),
            "eye_iou_denominator": float(len(finite_eye_iou)),
            "patient_roi_coverage_numerator": float(
                eyes["evaluable"].astype(bool).all()
            ),
            "patient_roi_coverage_denominator": 1.0,
            "patient_dice_numerator": (
                float(patient_dice) if np.isfinite(patient_dice) else 0.0
            ),
            "patient_dice_denominator": float(np.isfinite(patient_dice)),
            "patient_iou_numerator": (
                float(patient_iou) if np.isfinite(patient_iou) else 0.0
            ),
            "patient_iou_denominator": float(np.isfinite(patient_iou)),
        }
        records.append(record)
    return pd.DataFrame(records).sort_values(
        ["class_label", "patient_id"], kind="stable"
    ).reset_index(drop=True)


def _segmentation_bootstrap_seed(model: str, seed: int) -> int:
    material = f"q1-segmentation-patient-bootstrap-v1|{model}|{int(seed)}".encode(
        "utf-8"
    )
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def _segmentation_uncertainty_rows(
    cfg: Mapping[str, Any], model: str, seed: int
) -> list[dict[str, Any]]:
    contributions = _segmentation_patient_contributions(cfg, model, seed)
    draws = int(cfg["statistics"]["bootstrap_draws"])
    confidence = float(cfg["statistics"]["confidence_level"])
    if draws != 5000 or confidence != 0.95:
        raise ProtocolGateError(
            "Segmentation uncertainty requires 5000 draws at 95% confidence"
        )
    bootstrap_seed = _segmentation_bootstrap_seed(model, seed)
    rows: list[dict[str, Any]] = []
    for class_label in CLASS_ORDER:
        class_contributions = contributions.loc[
            contributions["class_label"] == class_label
        ].reset_index(drop=True)
        if class_contributions.empty:
            raise ProtocolGateError(
                f"Segmentation bootstrap lacks diagnostic class {class_label}"
            )
        for level in ("frame", "eye", "patient"):
            for metric_name in ("roi_coverage", "dice", "iou"):
                prefix = f"{level}_{metric_name}"
                summary = metrics.patient_cluster_ratio_bootstrap_ci(
                    class_contributions[f"{prefix}_numerator"].to_numpy(float),
                    class_contributions[f"{prefix}_denominator"].to_numpy(float),
                    draws=draws,
                    seed=bootstrap_seed,
                    confidence=confidence,
                    ci_method="bca",
                    stratified_by_label=True,
                )
                rows.append(
                    {
                        "model": model,
                        "seed": int(seed),
                        "class_label": int(class_label),
                        "class_name": CLASS_NAMES[class_label],
                        "level": level,
                        "metric": metric_name,
                        "metric_role": (
                            "strict_predicted_roi_coverage_no_reference_standard"
                            if metric_name == "roi_coverage"
                            else "retrospective_reference_standard_overlap_only"
                        ),
                        "estimate": _nullable_number(summary["estimate"]),
                        "ci_95_low": _nullable_number(summary["low"]),
                        "ci_95_high": _nullable_number(summary["high"]),
                        "bootstrap_se": _nullable_number(summary["bootstrap_se"]),
                        "confidence_level": float(summary["confidence"]),
                        "requested_method": str(summary["requested_method"]),
                        "interval_method": str(summary["method"]),
                        "fallback_used": bool(summary["fallback_used"]),
                        "fallback_detail": (
                            None
                            if summary["fallback_reason"] is None
                            else str(summary["fallback_reason"])
                        ),
                        "valid_draws": int(summary["valid_draws"]),
                        "requested_draws": int(summary["requested_draws"]),
                        "resampling_unit": str(summary["resampling_unit"]),
                        "stratified_by_label": bool(
                            summary["stratified_by_label"]
                        ),
                        "stratification": "three_class_patient_label",
                        "bootstrap_seed": int(bootstrap_seed),
                        "gt_used_for_inference_selection_or_abstention": False,
                    }
                )
    return rows


def _resolve_artifact_path(raw: str) -> Path:
    path = Path(raw)
    return path.resolve() if path.is_absolute() else resolve_project_path(raw)


def _provenance_rows(
    cfg: Mapping[str, Any],
    *,
    additional: Sequence[tuple[str, Path, bool]] = (),
) -> list[dict[str, Any]]:
    receipt_root = output_root(cfg) / "state" / "receipts"
    rows: list[dict[str, Any]] = []
    keys: set[tuple[Any, ...]] = set()
    after_test_stages = {
        "import-upstream-test-after-global-lock",
        "audit-upstream-test",
        "evaluate",
        "summarize",
        "audit",
    }
    for receipt_path in sorted(receipt_root.rglob("*.json"), key=lambda item: item.as_posix()):
        payload = read_json(receipt_path)
        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, list):
            continue
        model = payload.get("model")
        seed = payload.get("seed")
        strategy = payload.get("classifier_strategy")
        stage = str(payload.get("stage") or receipt_path.parent.name)
        receipt_role = receipt_path.relative_to(receipt_root).with_suffix("").as_posix()
        for index, record in enumerate(artifacts):
            if not isinstance(record, Mapping) or not record.get("path"):
                raise ProtocolGateError(f"Malformed artifact record in {receipt_path}")
            path = _resolve_artifact_path(str(record["path"]))
            if not path.is_file():
                raise ProtocolGateError(f"Provenance artifact disappeared: {path}")
            digest = sha256_file(path)
            if digest != record.get("sha256"):
                raise ProtocolGateError(f"Provenance artifact hash mismatch: {path}")
            recorded_size = record.get("size_bytes", record.get("bytes"))
            if recorded_size is not None and int(recorded_size) != path.stat().st_size:
                raise ProtocolGateError(f"Provenance artifact size mismatch: {path}")
            role = f"{receipt_role}:{record.get('role') or f'artifact_{index:04d}'}"
            key = (model, seed, strategy, role)
            if key in keys:
                raise ProtocolGateError(f"Duplicate provenance primary key: {key}")
            keys.add(key)
            rows.append(
                {
                    "model": None if model is None else str(model),
                    "seed": None if seed is None else int(seed),
                    "classifier_strategy": None if strategy is None else str(strategy),
                    "artifact_role": role,
                    "path": str(record["path"]),
                    "sha256": digest,
                    "size_bytes": int(path.stat().st_size),
                    "created_before_test_open": stage not in after_test_stages,
                }
            )
    for role, path, created_before_test_open in additional:
        resolved = path.resolve()
        if not resolved.is_file():
            raise ProtocolGateError(f"Additional provenance artifact is missing: {resolved}")
        key = (None, None, None, role)
        if key in keys:
            raise ProtocolGateError(f"Duplicate additional provenance primary key: {key}")
        keys.add(key)
        rows.append(
            {
                "model": None,
                "seed": None,
                "classifier_strategy": None,
                "artifact_role": role,
                "path": str(resolved),
                "sha256": sha256_file(resolved),
                "size_bytes": int(resolved.stat().st_size),
                "created_before_test_open": bool(created_before_test_open),
            }
        )
    if not rows:
        raise ProtocolGateError("No immutable receipts were available for provenance export")
    return rows


def _assert_cross_table_consistency(
    predictions: Sequence[Mapping[str, Any]],
    metric_rows: Sequence[Mapping[str, Any]],
    confusion_rows: Sequence[Mapping[str, Any]],
    uncertainty_rows: Sequence[Mapping[str, Any]],
) -> None:
    pred = pd.DataFrame(predictions)
    metric = pd.DataFrame(metric_rows)
    confusion = pd.DataFrame(confusion_rows)
    uncertainty = pd.DataFrame(uncertainty_rows)
    uncertainty_to_metric = {
        "coverage": "coverage",
        "failure_aware.balanced_accuracy": "failure_aware_balanced_accuracy",
        "conditional.balanced_accuracy": "conditional_balanced_accuracy",
        "failure_aware.accuracy": "accuracy",
        "failure_aware.macro_f1": "macro_f1",
        "failure_aware.multiclass_mcc": "multiclass_mcc",
        "conditional.macro_auroc": "macro_ovr_auroc",
        "conditional.macro_average_precision": "macro_ovr_average_precision",
        "conditional.multiclass_nll": "multiclass_nll",
        "conditional.multiclass_brier": "multiclass_brier",
        "failure_aware_aurc": "aurc",
        "localized_diagnostic_success.rate": "localized_diagnostic_success_rate",
    }
    keys = ["model", "seed", "classifier_strategy"]
    for identity, group in pred.groupby(keys, sort=True):
        expected = len(group)
        calibrated = metric.loc[
            (metric["model"] == identity[0])
            & (metric["seed"] == identity[1])
            & (metric["classifier_strategy"] == identity[2])
            & (metric["probability_scale"] == "calibrated")
        ]
        if len(calibrated) != 1 or int(calibrated.iloc[0]["n_intended"]) != expected:
            raise ProtocolGateError(f"Patient metric denominator mismatch for {identity}")
        cells = confusion.loc[
            (confusion["model"] == identity[0])
            & (confusion["seed"] == identity[1])
            & (confusion["classifier_strategy"] == identity[2])
        ]
        if len(cells) != 12 or int(cells["count"].sum()) != expected:
            raise ProtocolGateError(f"Confusion denominator mismatch for {identity}")
        if int(group["abstained"].sum()) != int(calibrated.iloc[0]["n_abstained"]):
            raise ProtocolGateError(f"Abstention denominator mismatch for {identity}")
        patient_uncertainty = uncertainty.loc[
            (uncertainty["model"] == identity[0])
            & (uncertainty["seed"] == identity[1])
            & (uncertainty["classifier_strategy"] == identity[2])
            & (uncertainty["level"] == "patient")
        ]
        uncertainty_lookup = {
            str(row.metric_path): row.estimate
            for row in patient_uncertainty.itertuples(index=False)
        }
        missing_paths = set(uncertainty_to_metric) - set(uncertainty_lookup)
        if missing_paths:
            raise ProtocolGateError(
                f"Patient uncertainty lacks publication metrics for {identity}: "
                f"{sorted(missing_paths)}"
            )
        point_row = calibrated.iloc[0]
        for metric_path, column in uncertainty_to_metric.items():
            point = _nullable_number(point_row[column])
            estimate = _nullable_number(uncertainty_lookup[metric_path])
            if (point is None) != (estimate is None) or (
                point is not None
                and estimate is not None
                and not math.isclose(point, estimate, rel_tol=0.0, abs_tol=1e-12)
            ):
                raise ProtocolGateError(
                    "Patient uncertainty point estimate disagrees with "
                    f"patient_metrics.csv for {(*identity, metric_path)}"
                )


def _assert_segmentation_uncertainty_consistency(
    segmentation_rows: Sequence[Mapping[str, Any]],
    uncertainty_rows: Sequence[Mapping[str, Any]],
) -> None:
    point_lookup = {
        (
            row["model"],
            int(row["seed"]),
            int(row["class_label"]),
            row["level"],
        ): row
        for row in segmentation_rows
    }
    expected_keys = {
        (*identity, metric_name)
        for identity in point_lookup
        for metric_name in ("roi_coverage", "dice", "iou")
    }
    observed_keys: set[tuple[Any, ...]] = set()
    for row in uncertainty_rows:
        identity = (
            row["model"],
            int(row["seed"]),
            int(row["class_label"]),
            row["level"],
        )
        metric_name = str(row["metric"])
        observed_keys.add((*identity, metric_name))
        if identity not in point_lookup:
            raise ProtocolGateError(
                f"Segmentation uncertainty lacks a point-estimate row for {identity}"
            )
        point = _nullable_number(point_lookup[identity][metric_name])
        estimate = _nullable_number(row["estimate"])
        if (point is None) != (estimate is None) or (
            point is not None
            and estimate is not None
            and not math.isclose(point, estimate, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise ProtocolGateError(
                f"Segmentation uncertainty point estimate disagrees for {(*identity, metric_name)}"
            )
    if observed_keys != expected_keys:
        raise ProtocolGateError(
            "Segmentation uncertainty does not cover every class/level/metric point estimate"
        )


def generate_publication_outputs(cfg: Mapping[str, Any]) -> dict[str, Path]:
    """Create every locked publication table without changing model results."""

    prediction_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    classwise_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    uncertainty_rows: list[dict[str, Any]] = []
    segmentation_rows: list[dict[str, Any]] = []
    segmentation_uncertainty_rows: list[dict[str, Any]] = []

    for model in cfg["models"]:
        for seed_value in cfg["split_seeds"]:
            seed = int(seed_value)
            localized_eye_map, localized_patient_map = _localization_maps(
                cfg, model, seed
            )
            for strategy in cfg["classifier"]["strategy_order"]:
                evaluation = unit_root(cfg, model, seed) / "evaluation" / strategy
                raw_patients = pd.read_csv(evaluation / "patients_raw.csv", dtype={"patient_id": str})
                calibrated_patients = pd.read_csv(
                    evaluation / "patients_calibrated.csv", dtype={"patient_id": str}
                )
                raw_eyes = pd.read_csv(
                    evaluation / "eyes_raw.csv", dtype={"patient_id": str, "case_id": str}
                )
                calibration = read_json(
                    unit_root(cfg, model, seed)
                    / "classifiers"
                    / strategy
                    / "calibration_lock.json"
                )
                prediction_rows.extend(
                    _patient_prediction_rows(
                        cfg,
                        model,
                        seed,
                        strategy,
                        raw_patients,
                        calibrated_patients,
                        raw_eyes,
                        calibration,
                        localized_patient_map,
                    )
                )
                results: dict[tuple[str, str], dict[str, Any]] = {}
                for scale, patient_table in (
                    ("raw", raw_patients),
                    ("calibrated", calibrated_patients),
                ):
                    patient_localized = np.asarray(
                        [localized_patient_map[str(value)] for value in patient_table["patient_id"]],
                        dtype=bool,
                    )
                    result = _unit_metrics(
                        patient_table, localized=patient_localized
                    )
                    results[(scale, "patient")] = result
                    metric_rows.append(
                        _patient_metric_row(cfg, model, seed, strategy, scale, result)
                    )
                    if scale == "calibrated":
                        confusion_rows.extend(_confusion_rows(model, seed, strategy, result))
                        classwise_rows.extend(_classwise_rows(model, seed, strategy, result))
                        risk_rows.extend(
                            _risk_rows(model, seed, strategy, calibrated_patients)
                        )
                for level in ("eye", "patient"):
                    for scale in ("raw", "calibrated"):
                        if level == "patient":
                            result = results[(scale, "patient")]
                        else:
                            eye_table = pd.read_csv(
                                evaluation / f"eyes_{scale}.csv",
                                dtype={"patient_id": str, "case_id": str},
                            )
                            eye_localized = np.asarray(
                                [
                                    localized_eye_map[
                                        (
                                            str(row.patient_id),
                                            str(row.case_id),
                                            str(row.side),
                                        )
                                    ]
                                    for row in eye_table.itertuples(index=False)
                                ],
                                dtype=bool,
                            )
                            result = _unit_metrics(
                                eye_table, localized=eye_localized
                            )
                        calibration_rows.append(
                            _calibration_metric_rows(
                                model,
                                seed,
                                strategy,
                                level,
                                scale,
                                result,
                                calibration,
                            )
                        )
                uncertainty_rows.extend(
                    _uncertainty_rows(cfg, model, seed, strategy, evaluation)
                )
            segmentation_rows.extend(_segmentation_rows(cfg, model, seed))
            segmentation_uncertainty_rows.extend(
                _segmentation_uncertainty_rows(cfg, model, seed)
            )

    _assert_cross_table_consistency(
        prediction_rows, metric_rows, confusion_rows, uncertainty_rows
    )
    _assert_segmentation_uncertainty_consistency(
        segmentation_rows, segmentation_uncertainty_rows
    )
    rows_by_name: dict[str, list[dict[str, Any]]] = {
        "patient_predictions.csv": prediction_rows,
        "patient_metrics.csv": metric_rows,
        "patient_confusion_3x4.csv": confusion_rows,
        "classwise_metrics.csv": classwise_rows,
        "calibration_metrics.csv": calibration_rows,
        "risk_coverage.csv": risk_rows,
        "uncertainty_metrics.csv": uncertainty_rows,
        "segmentation_by_class.csv": segmentation_rows,
        "segmentation_uncertainty.csv": segmentation_uncertainty_rows,
    }
    table_root = output_root(cfg) / "tables"
    paths = {
        name: _atomic_csv(table_root / name, rows)
        for name, rows in rows_by_name.items()
    }
    provenance_additional = [
        (f"publication_output:{name}", path, False)
        for name, path in paths.items()
    ]
    provenance_additional.extend(
        [
            (
                "publication_contract:q1_metadata",
                output_root(cfg) / "provenance" / "q1_metadata.json",
                True,
            ),
            (
                "publication_contract:output_schemas",
                output_root(cfg) / "provenance" / "publication_output_schemas.json",
                True,
            ),
        ]
    )
    provenance_rows = _provenance_rows(cfg, additional=provenance_additional)
    rows_by_name["provenance_artifacts.csv"] = provenance_rows
    paths["provenance_artifacts.csv"] = _atomic_csv(
        table_root / "provenance_artifacts.csv", provenance_rows
    )
    if tuple(rows_by_name) != PUBLICATION_TABLE_NAMES:
        raise ProtocolGateError("Publication output table set changed")
    manifest = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "study_id": cfg["study_id"],
        "generated_utc": utc_now(),
        "generation_role": "downstream_reporting_only_no_training_selection_or_refitting",
        "primary_probability_scale": "calibrated",
        "classwise_probability_scale": "calibrated",
        "risk_coverage_probability_scale": "calibrated",
        "uncertainty_probability_scale": "calibrated",
        "uncertainty_confidence_level": 0.95,
        "uncertainty_requested_draws": 5000,
        "uncertainty_resampling_unit": "whole_patient_cluster",
        "uncertainty_stratification": "three_class_patient_label",
        "segmentation_overlap_role": "retrospective_reference_standard_overlap_only",
        "provenance_self_reference_policy": (
            "provenance_artifacts.csv lists all other publication tables and locked "
            "receipt artifacts; its own hash is recorded by this manifest"
        ),
        "prior_test_use_disclosure": cfg["prior_test_use"]["required_disclosure"],
        "tables": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "rows": len(rows_by_name[name]),
            }
            for name, path in paths.items()
        },
    }
    manifest_path = save_json_atomic(
        output_root(cfg) / "summary" / "publication_output_manifest.json", manifest
    )
    paths["publication_output_manifest.json"] = manifest_path
    return paths


def _coerce_csv_rows(table_name: str, table: pd.DataFrame) -> list[dict[str, Any]]:
    schema = OUTPUT_TABLE_SCHEMAS[table_name]
    rows: list[dict[str, Any]] = []
    for raw in table.to_dict(orient="records"):
        row: dict[str, Any] = {}
        for column, spec in schema["columns"].items():
            value = raw.get(column)
            if pd.isna(value):
                row[column] = None
            elif spec["type"] == "integer":
                row[column] = int(value)
            elif spec["type"] == "number":
                row[column] = float(value)
            elif spec["type"] == "boolean":
                if isinstance(value, (bool, np.bool_)):
                    row[column] = bool(value)
                elif str(value).strip().lower() in {"true", "1"}:
                    row[column] = True
                elif str(value).strip().lower() in {"false", "0"}:
                    row[column] = False
                else:
                    raise ProtocolGateError(
                        f"Malformed boolean {column!r} in {table_name}: {value!r}"
                    )
            else:
                row[column] = str(value)
        rows.append(row)
    return rows


def validate_publication_outputs(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Re-open and verify Q1 metadata, schemas, tables, row contracts and hashes."""

    root = output_root(cfg)
    metadata = read_json(root / "provenance" / "q1_metadata.json")
    validate_q1_metadata(metadata)
    schema_payload = read_json(root / "provenance" / "publication_output_schemas.json")
    if schema_payload != {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "tables": publication_output_schemas(),
    }:
        raise ProtocolGateError("Publication schema snapshot differs from locked schemas")
    manifest_path = root / "summary" / "publication_output_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != OUTPUT_SCHEMA_VERSION:
        raise ProtocolGateError("Publication manifest schema version mismatch")
    observed_counts: dict[str, int] = {}
    for table_name in PUBLICATION_TABLE_NAMES:
        path = root / "tables" / table_name
        if not path.is_file():
            raise ProtocolGateError(f"Missing publication table: {path}")
        record = manifest.get("tables", {}).get(table_name, {})
        if sha256_file(path) != record.get("sha256") or path.stat().st_size != record.get("size_bytes"):
            raise ProtocolGateError(f"Publication table hash/size mismatch: {path}")
        rows = _coerce_csv_rows(table_name, pd.read_csv(path))
        validate_output_rows(table_name, rows)
        if len(rows) != record.get("rows"):
            raise ProtocolGateError(f"Publication row-count mismatch: {table_name}")
        observed_counts[table_name] = len(rows)
    return {
        "status": "passed",
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "metadata_sha256": sha256_file(root / "provenance" / "q1_metadata.json"),
        "schema_sha256": sha256_file(
            root / "provenance" / "publication_output_schemas.json"
        ),
        "manifest_sha256": sha256_file(manifest_path),
        "row_counts": observed_counts,
    }


__all__ = [
    "PUBLICATION_TABLE_NAMES",
    "generate_publication_outputs",
    "localization_vectors",
    "segmentation_only_audit",
    "validate_publication_outputs",
    "write_publication_contracts",
]
