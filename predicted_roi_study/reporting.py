"""Reproducible exports for the strict predicted-ROI study.

This module writes only derived artefacts.  It never trains a model, chooses a
test threshold, fits a test calibrator, or substitutes another image source for
an abstained predicted ROI.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import special
from sklearn import metrics as skm

from .metrics import (
    ABSTAIN,
    CORE_ABLATION_ARMS,
    aggregate_eyes_to_patients,
    aggregate_frames_to_eyes,
    apply_temperature_scaling,
    calibration_curve_table,
    clean_json,
    decision_curve_net_benefit,
    evaluate_ablation_arms,
    flatten_metrics,
    paired_cluster_bootstrap_difference,
    patient_cluster_calibration_band,
    patient_cluster_bootstrap_ci,
    patient_cluster_continuous_ci,
    risk_coverage_curve,
    selective_binary_metrics,
    summarize_across_seeds,
    summarize_numeric_columns,
    validate_ablation_alignment,
    delong_auc_comparison,
    holm_adjust,
    mcnemar_exact_comparison,
)


ABLATION_ARM_METADATA = {
    "predicted_roi": {
        "role": "primary_deployable",
        "description": "Classifier receives only the quality-gated predicted ROI.",
    },
    "gt_oracle": {
        "role": "non_deployable_upper_bound",
        "description": "Classifier receives GT ROI; never used for deployable claims.",
    },
    "whole_image": {
        "role": "legacy_baseline",
        "description": "Whole-image classifier; tests the cost/benefit of anatomical restriction.",
    },
    "mask_only": {
        "role": "shortcut_control",
        "description": "Only predicted mask geometry is visible; no ROI texture.",
    },
    "background_only": {
        "role": "shortcut_control",
        "description": "Predicted ROI is removed; detects background/acquisition shortcuts.",
    },
    "roi_shuffle": {
        "role": "negative_control",
        "description": "ROI source is permuted within locked strata using a saved permutation.",
    },
    "whole_image_baseline": {
        "role": "legacy_baseline",
        "description": "Configured alias of the whole-image shortcut reference.",
    },
    "gt_roi_oracle": {
        "role": "non_deployable_upper_bound",
        "description": "Configured alias of the strict GT-ROI oracle.",
    },
    "bbox_context": {"role": "context_leakage_ablation"},
    "largest_component_without_quality_gates": {"role": "quality_gate_ablation"},
    "geometry_only": {"role": "shortcut_control"},
    "background_only_negative_control": {"role": "shortcut_control"},
    "random_roi_negative_control": {"role": "negative_control"},
    "appearance_plus_geometry": {"role": "feature_ablation"},
}

LABEL_3CLASS_NAMES = {0: "control", 1: "papilledema", 2: "pseudopapilledema"}


def _evaluation_threshold(threshold: float | None) -> float:
    """Numerical placeholder used only when every unit has already abstained."""
    return (
        float(threshold)
        if threshold is not None and np.isfinite(threshold) and 0 <= threshold <= 1
        else 0.5
    )


def _save_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(clean_json(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _normalise_probability_scale(
    scale: str | None, temperature: float | None
) -> str:
    if scale is None:
        return (
            "temperature_scaled"
            if temperature is not None and np.isfinite(temperature) and temperature > 0
            else "unavailable"
        )
    value = str(scale).strip().lower()
    if value in {"raw", "raw_due_to_calibration_unavailable", "uncalibrated"}:
        return "raw"
    if value in {"temperature_scaled", "calibrated", "temperature"}:
        return "temperature_scaled"
    if value in {"unavailable", "none"}:
        return "unavailable"
    raise ValueError(f"Unknown threshold probability scale: {scale!r}")


def _prepare_final_unit_table(
    table: pd.DataFrame,
    *,
    temperature: float | None,
    threshold: float | None,
    threshold_probability_scale: str | None,
) -> pd.DataFrame:
    """Attach raw/calibrated/operational scores without inventing calibration."""
    result = table.copy().rename(columns={"probability": "probability_raw"})
    roi_evaluable = result.evaluable.to_numpy(bool)
    result["roi_evaluable"] = roi_evaluable
    result["probability_calibrated"] = np.nan
    calibration_available = (
        temperature is not None and np.isfinite(temperature) and temperature > 0
    )
    if calibration_available:
        raw = result.loc[roi_evaluable, "probability_raw"].to_numpy(float)
        logits = special.logit(np.clip(raw, 1e-7, 1 - 1e-7))
        result.loc[roi_evaluable, "probability_calibrated"] = apply_temperature_scaling(
            logits, float(temperature)
        )
    result["calibrated_evaluable"] = roi_evaluable & calibration_available
    result["calibration_status"] = "locked" if calibration_available else "unavailable"

    scale = _normalise_probability_scale(threshold_probability_scale, temperature)
    threshold_available = threshold is not None and np.isfinite(threshold)
    scale_available = scale == "raw" or (scale == "temperature_scaled" and calibration_available)
    operational = roi_evaluable & threshold_available & scale_available
    result["probability"] = np.nan
    if scale == "raw":
        result.loc[operational, "probability"] = result.loc[
            operational, "probability_raw"
        ]
    elif scale == "temperature_scaled" and calibration_available:
        result.loc[operational, "probability"] = result.loc[
            operational, "probability_calibrated"
        ]
    result["probability_scale"] = scale
    result["evaluable"] = operational

    prediction = np.full(len(result), ABSTAIN, dtype=int)
    if operational.any():
        prediction[operational] = (
            result.loc[operational, "probability"].to_numpy(float) >= float(threshold)
        ).astype(int)
    result["prediction"] = prediction
    if not threshold_available or not scale_available:
        result["abstention_reason"] = "calibration_or_threshold_unavailable"
    if "localized" in result:
        result["localized_diagnostic_success"] = (
            operational
            & result.localized.to_numpy(bool)
            & (prediction == result.label.to_numpy(int))
        ).astype(int)
    return result


def _mark_threshold_outputs_unavailable(metrics: dict, reason: str) -> None:
    """Retain score/calibration metrics while removing every dummy threshold output."""
    metrics["threshold_metrics_status"] = reason
    metrics["conditional"]["threshold"] = np.nan
    for key in (
        "accuracy",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "f1",
        "mcc",
        "tn",
        "fp",
        "fn",
        "tp",
        "confusion_matrix",
        "confusion_matrix_row_proportions",
    ):
        metrics["conditional"][key] = None
    for key in tuple(metrics["failure_inclusive"]):
        metrics["failure_inclusive"][key] = None
    for key in (
        "selective_risk",
        "conditional_aurc",
        "failure_aware_aurc",
        "failure_aware_accuracy",
        "failure_aware_balanced_accuracy",
        "failure_aware_sensitivity",
        "failure_aware_specificity",
    ):
        metrics[key] = None
    metrics["prediction"] = None
    metrics["confusion_matrix_2x3"] = None
    metrics["confusion_matrix_2x3_row_proportions"] = None
    if "localized_diagnostic_success" in metrics:
        metrics["localized_diagnostic_success"] = {
            "status": "unavailable_without_locked_decision_threshold"
        }


def _scope_evaluation(
    table: pd.DataFrame,
    *,
    threshold: float | None,
    temperature: float | None,
    threshold_probability_scale: str | None,
    calibration_bins: int,
) -> dict:
    metric_threshold = _evaluation_threshold(threshold)
    scale = _normalise_probability_scale(threshold_probability_scale, temperature)
    calibration_available = (
        temperature is not None and np.isfinite(temperature) and temperature > 0
    )
    if scale == "temperature_scaled" and calibration_available:
        raw_equivalent_threshold = float(
            special.expit(
                float(temperature)
                * special.logit(np.clip(metric_threshold, 1e-7, 1 - 1e-7))
            )
        )
        calibrated_equivalent_threshold = metric_threshold
    elif scale == "raw":
        raw_equivalent_threshold = metric_threshold
        calibrated_equivalent_threshold = (
            float(
                special.expit(
                    special.logit(np.clip(metric_threshold, 1e-7, 1 - 1e-7))
                    / float(temperature)
                )
            )
            if calibration_available
            else np.nan
        )
    else:
        raw_equivalent_threshold = np.nan
        calibrated_equivalent_threshold = np.nan
    localized = table.localized if "localized" in table else None
    operational = selective_binary_metrics(
        table.label,
        table.probability,
        evaluable=table.evaluable,
        threshold=metric_threshold,
        n_calibration_bins=calibration_bins,
        localized_success=localized,
    )
    raw_threshold_available = np.isfinite(raw_equivalent_threshold)
    raw = selective_binary_metrics(
        table.label,
        table.probability_raw,
        evaluable=table.roi_evaluable,
        threshold=_evaluation_threshold(raw_equivalent_threshold),
        n_calibration_bins=calibration_bins,
        localized_success=localized,
    )
    if not raw_threshold_available:
        _mark_threshold_outputs_unavailable(
            raw, "unavailable_no_locked_raw_equivalent_threshold"
        )
    if calibration_available:
        calibrated = selective_binary_metrics(
            table.label,
            table.probability_calibrated,
            evaluable=table.calibrated_evaluable,
            threshold=_evaluation_threshold(calibrated_equivalent_threshold),
            n_calibration_bins=calibration_bins,
            localized_success=localized,
        )
        calibrated["status"] = "available"
        if not np.isfinite(calibrated_equivalent_threshold):
            calibrated["status"] = "available_score_only_threshold_unavailable"
            _mark_threshold_outputs_unavailable(
                calibrated, "unavailable_no_locked_calibrated_equivalent_threshold"
            )
    else:
        calibrated = selective_binary_metrics(
            table.label,
            np.full(len(table), np.nan),
            evaluable=np.zeros(len(table), dtype=bool),
            threshold=0.5,
            n_calibration_bins=calibration_bins,
            localized_success=localized,
        )
        calibrated.update(
            {
                "status": "unavailable",
                "reason": "temperature_scaling_not_available; no T=1 substitution",
                "operating_threshold": np.nan,
            }
        )
        calibrated["conditional"]["threshold"] = np.nan
    result = {
        "primary_operational": operational,
        # Backward-compatible key. When calibration is available it is the
        # operational result; otherwise it is explicitly unavailable.
        "primary_calibrated": calibrated,
        "raw": raw,
        "calibration_comparison": {
            "raw": raw["calibration"],
            "temperature_scaled": (
                calibrated["calibration"]
                if calibration_available
                else {"status": "unavailable", "reason": calibrated["reason"]}
            ),
            "note": "Both are evaluated only among quality-gate-evaluable units.",
        },
        "operating_thresholds": {
            "probability_scale": scale,
            "operational": metric_threshold if threshold is not None else np.nan,
            "temperature_scaled_equivalent": calibrated_equivalent_threshold,
            "raw_equivalent": raw_equivalent_threshold if threshold is not None else np.nan,
            "note": "Monotonic inverse-temperature mapping gives identical class decisions.",
        },
    }
    covered = table.loc[table.roi_evaluable.astype(bool), "label"].astype(int)
    events = int(covered.sum()) if len(covered) else 0
    nonevents = int(len(covered) - events)
    result["calibration_sample_size"] = {
        "roi_evaluable_units": int(len(covered)),
        "evaluable_units": int(len(covered)),
        "events": events,
        "nonevents": nonevents,
        "small_sample_caution": bool(min(events, nonevents) < 20),
        "note": (
            "Calibration slope/intercept and ECE are imprecise with few events or "
            "non-events; inspect patient-cluster intervals and reliability-bin support."
        ),
    }
    if threshold is None or not np.isfinite(threshold):
        result["decision_threshold_status"] = "unavailable_all_units_abstain"
        for version in (operational, raw, calibrated):
            version["conditional"]["threshold"] = np.nan
    return result


def _curves_for_scope(
    table: pd.DataFrame,
    *,
    level: str,
    scope: str,
    calibration_bins: int,
    decision_thresholds: Sequence[float] | None,
) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    curves = []
    # Each score scale has its own evaluability definition.  In particular,
    # raw score calibration remains reportable when no operating threshold or
    # temperature could be locked.
    for name, column, validity in (
        ("raw", "probability_raw", "roi_evaluable"),
        ("temperature_scaled", "probability_calibrated", "calibrated_evaluable"),
    ):
        valid = table[validity].to_numpy(bool)
        if not valid.any():
            continue
        for strategy in ("uniform", "quantile"):
            curve = calibration_curve_table(
                table.loc[valid, "label"],
                table.loc[valid, column],
                n_bins=calibration_bins,
                strategy=strategy,
            )
            curve.insert(0, "calibration", name)
            curve.insert(0, "scope", scope)
            curve.insert(0, "level", level)
            curves.append(curve)
    dca = decision_curve_net_benefit(
        table.label,
        table.probability,
        evaluable=table.evaluable,
        thresholds=decision_thresholds,
    )
    dca.insert(0, "scope", scope)
    dca.insert(0, "level", level)
    probability_scale = (
        str(table.probability_scale.iloc[0]) if len(table) else "unavailable"
    )
    dca.insert(2, "probability_scale", probability_scale)
    dca.insert(
        3,
        "decision_policy_status",
        "locked" if table.evaluable.to_numpy(bool).any() else "all_abstain_unavailable_lock",
    )
    return curves, dca


def _confusion_rows(metrics: Mapping, *, level: str, scope: str) -> tuple[list[dict], list[dict]]:
    primary = metrics["primary_operational"]
    matrix_2x3 = np.asarray(primary["confusion_matrix_2x3"], dtype=int)
    proportions_2x3 = np.asarray(
        primary["confusion_matrix_2x3_row_proportions"], dtype=float
    )
    rows_2x3 = []
    for true_label in (0, 1):
        for column_index, outcome in enumerate(("negative", "positive", "abstain")):
            rows_2x3.append(
                {
                    "level": level,
                    "scope": scope,
                    "true_label": true_label,
                    "outcome": outcome,
                    "n": int(matrix_2x3[true_label, column_index]),
                    "row_proportion": float(proportions_2x3[true_label, column_index]),
                }
            )
    conditional = primary["conditional"]
    matrix_2x2 = np.asarray(conditional.get("confusion_matrix", [[0, 0], [0, 0]]), dtype=int)
    row_sums_2x2 = matrix_2x2.sum(axis=1, keepdims=True)
    proportions_2x2 = np.divide(
        matrix_2x2,
        row_sums_2x2,
        out=np.full(matrix_2x2.shape, np.nan, dtype=float),
        where=row_sums_2x2 > 0,
    )
    rows_2x2 = []
    for true_label in (0, 1):
        for predicted_label in (0, 1):
            rows_2x2.append(
                {
                    "level": level,
                    "scope": scope,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "n": int(matrix_2x2[true_label, predicted_label]),
                    "row_proportion": float(proportions_2x2[true_label, predicted_label]),
                }
            )
    return rows_2x3, rows_2x2


def _abstention_table(table: pd.DataFrame, *, level: str, scope: str) -> pd.DataFrame:
    abstained = table.loc[~table.evaluable].copy()
    if abstained.empty:
        return pd.DataFrame(
            columns=["level", "scope", "label", "abstention_reason", "n"]
        )
    result = (
        abstained.groupby(["label", "abstention_reason"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )
    result.insert(0, "scope", scope)
    result.insert(0, "level", level)
    return result


def _score_distribution(values: pd.Series) -> dict[str, float | int]:
    scores = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    return {
        "n": int(len(scores)),
        "mean": float(scores.mean()) if len(scores) else np.nan,
        "sample_sd": float(scores.std(ddof=1)) if len(scores) > 1 else np.nan,
        "median": float(np.median(scores)) if len(scores) else np.nan,
        "q1": float(np.quantile(scores, 0.25)) if len(scores) else np.nan,
        "q3": float(np.quantile(scores, 0.75)) if len(scores) else np.nan,
        "minimum": float(scores.min()) if len(scores) else np.nan,
        "maximum": float(scores.max()) if len(scores) else np.nan,
    }


def label_3class_descriptive(
    table: pd.DataFrame, *, level: str, scope: str
) -> pd.DataFrame:
    """Describe binary-system behaviour within each original diagnostic class.

    Every subgroup contains a single binary truth label, so within-subgroup
    AUROC is intentionally undefined.  Control receives specificity and each
    disease subgroup receives sensitivity, both conditionally and with
    abstentions counted as failures.
    """
    if "label_3class" not in table:
        return pd.DataFrame()
    rows = []
    for original_label, group in table.groupby("label_3class", sort=True):
        if group.label.nunique() != 1:
            raise ValueError("Each label_3class subgroup must map to one binary label.")
        binary_label = int(group.label.iloc[0])
        valid = group.evaluable.to_numpy(bool)
        roi_valid = (
            group.roi_evaluable.to_numpy(bool)
            if "roi_evaluable" in group
            else valid
        )
        prediction = group.prediction.to_numpy(int)
        correct_outcome = 0 if binary_label == 0 else 1
        correct = int((prediction == correct_outcome).sum())
        conditional_denominator = int(valid.sum())
        metric_name = "specificity" if binary_label == 0 else "sensitivity"
        row = {
            "level": level,
            "scope": scope,
            "label_3class": int(original_label),
            "label_3class_name": LABEL_3CLASS_NAMES.get(
                int(original_label), f"class_{int(original_label)}"
            ),
            "binary_label": binary_label,
            "n_total": int(len(group)),
            "n_evaluable": conditional_denominator,
            "n_roi_evaluable": int(roi_valid.sum()),
            "n_abstain": int((~valid).sum()),
            "coverage": float(valid.mean()),
            "roi_coverage": float(roi_valid.mean()),
            "predicted_negative": int((prediction == 0).sum()),
            "predicted_positive": int((prediction == 1).sum()),
            "abstain": int((prediction == ABSTAIN).sum()),
            "class_appropriate_metric": metric_name,
            "conditional_class_appropriate_rate": (
                correct / conditional_denominator if conditional_denominator else np.nan
            ),
            "failure_aware_class_appropriate_rate": correct / len(group),
            "conditional_sensitivity": (
                correct / conditional_denominator
                if binary_label == 1 and conditional_denominator
                else np.nan
            ),
            "failure_aware_sensitivity": (
                correct / len(group) if binary_label == 1 else np.nan
            ),
            "conditional_specificity": (
                correct / conditional_denominator
                if binary_label == 0 and conditional_denominator
                else np.nan
            ),
            "failure_aware_specificity": (
                correct / len(group) if binary_label == 0 else np.nan
            ),
            "auroc": np.nan,
            "auroc_status": "undefined_single_binary_class_within_label_3class",
        }
        for name, column in (
            ("operational", "probability"),
            ("raw", "probability_raw"),
            ("temperature_scaled", "probability_calibrated"),
        ):
            distribution = _score_distribution(group[column])
            row.update({f"{name}_score_{key}": value for key, value in distribution.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def build_evaluation_bundle(
    frames: pd.DataFrame,
    *,
    eye_threshold: float | None,
    patient_threshold: float | None,
    eye_temperature: float | None = 1.0,
    patient_temperature: float | None = 1.0,
    eye_threshold_probability_scale: str | None = None,
    patient_threshold_probability_scale: str | None = None,
    eye_lock: Mapping | None = None,
    patient_lock: Mapping | None = None,
    probability_column: str = "probability",
    min_valid_frames: int = 4,
    frames_per_eye: int = 7,
    calibration_bins: int = 10,
    bootstrap_draws: int = 2000,
    bootstrap_seed: int = 1729,
    decision_thresholds: Sequence[float] | None = None,
    mean_segmentation_columns: Sequence[str] | None = None,
    classifier_strategy: str | None = None,
) -> dict:
    """Build eye/patient tables, metrics, CIs, calibration and DCA artefacts.

    Temperatures and thresholds must already have been locked on validation.
    Frame probabilities are averaged first; temperature scaling is applied at
    the final intended unit (separately for eye and patient).
    """
    frames = frames.copy()
    if classifier_strategy is not None:
        strategy = str(classifier_strategy)
        if not strategy:
            raise ValueError("classifier_strategy must be a non-empty string")
        if "classifier_strategy" in frames and not (
            frames["classifier_strategy"].astype(str) == strategy
        ).all():
            raise ValueError("Frame classifier_strategy conflicts with the requested export")
        frames["classifier_strategy"] = strategy
    if eye_lock is not None:
        eye_temperature, eye_threshold, eye_threshold_probability_scale = _lock_values(
            eye_lock
        )
    if patient_lock is not None:
        (
            patient_temperature,
            patient_threshold,
            patient_threshold_probability_scale,
        ) = _lock_values(patient_lock)
    eyes_raw = aggregate_frames_to_eyes(
        frames,
        threshold=0.5,
        frames_per_eye=frames_per_eye,
        min_valid_frames=min_valid_frames,
        probability_column=probability_column,
        mean_columns=mean_segmentation_columns,
    )
    # Segmentation/localisation coverage is an endpoint in its own right and
    # must not disappear when the downstream classifier has no selectable
    # checkpoint.  In that case diagnostic ``roi_valid`` is correctly false,
    # while ``segmentation_roi_valid`` still records whether the segmenter
    # supplied an anatomically admissible component.
    segmentation_validity_column = (
        "segmentation_roi_valid"
        if "segmentation_roi_valid" in frames
        else "roi_valid"
    )
    segmentation_eye_counts = (
        frames.assign(
            _segmentation_valid=frames[segmentation_validity_column].astype(bool)
        )
        .groupby("case_id", sort=False)["_segmentation_valid"]
        .sum()
    )
    eyes_raw["segmentation_n_valid_frames"] = (
        eyes_raw["case_id"].map(segmentation_eye_counts).astype(int)
    )
    eyes_raw["segmentation_valid_frame_fraction"] = (
        eyes_raw["segmentation_n_valid_frames"] / int(frames_per_eye)
    )
    eyes_raw["segmentation_roi_evaluable"] = (
        eyes_raw["segmentation_n_valid_frames"] >= int(min_valid_frames)
    )
    eyes = _prepare_final_unit_table(
        eyes_raw,
        temperature=eye_temperature,
        threshold=eye_threshold,
        threshold_probability_scale=eye_threshold_probability_scale,
    )

    # Patient raw score is the mean of the two raw eye scores.  Patient
    # calibration is fitted/applied independently from eye calibration.
    patient_input = eyes_raw.copy()
    patients_raw = aggregate_eyes_to_patients(
        patient_input,
        threshold=0.5,
        probability_column="probability",
        mean_columns=mean_segmentation_columns,
    )
    segmentation_patient_status = eyes_raw.groupby("patient_id", sort=False)[
        "segmentation_roi_evaluable"
    ].agg(["sum", "all"])
    patients_raw["segmentation_n_evaluable_eyes"] = (
        patients_raw["patient_id"].map(segmentation_patient_status["sum"]).astype(int)
    )
    patients_raw["segmentation_roi_evaluable"] = (
        patients_raw["patient_id"].map(segmentation_patient_status["all"]).astype(bool)
    )
    patients = _prepare_final_unit_table(
        patients_raw,
        temperature=patient_temperature,
        threshold=patient_threshold,
        threshold_probability_scale=patient_threshold_probability_scale,
    )

    report: dict = {
        "protocol": {
            "frames_per_eye": int(frames_per_eye),
            "min_valid_frames": int(min_valid_frames),
            "patient_requires_both_eyes": True,
            "invalid_roi_probability_policy": "must_be_missing; no fallback",
            "frame_aggregation": "arithmetic mean over valid predicted-ROI frame probabilities",
            "patient_aggregation": "arithmetic mean over two raw eye probabilities",
            "eye_temperature": float(eye_temperature) if eye_temperature is not None else np.nan,
            "patient_temperature": float(patient_temperature) if patient_temperature is not None else np.nan,
            "eye_threshold": float(eye_threshold) if eye_threshold is not None else np.nan,
            "patient_threshold": float(patient_threshold) if patient_threshold is not None else np.nan,
            "eye_threshold_probability_scale": _normalise_probability_scale(
                eye_threshold_probability_scale, eye_temperature
            ),
            "patient_threshold_probability_scale": _normalise_probability_scale(
                patient_threshold_probability_scale, patient_temperature
            ),
            "classifier_strategy": classifier_strategy,
            "calibration_fit_split": "validation only (caller responsibility; never test)",
            "unavailable_level_lock_policy": (
                "Use a validation-locked raw threshold only when the lock explicitly says "
                "raw_due_to_calibration_unavailable. Otherwise all units at that level "
                "abstain; never substitute temperature=1 or threshold=0.5."
            ),
        },
        "eye": {},
        "patient": {},
        "segmentation": {},
        "interpretation": (
            "Eyes contain seven repeated frames and are clustered within patients. "
            "CIs resample whole patients. AUROC/AP/calibration are conditional on "
            "evaluability; failure-inclusive metrics count abstention as failure."
        ),
    }
    calibration_curves: list[pd.DataFrame] = []
    calibration_bands: list[pd.DataFrame] = []
    decision_curves: list[pd.DataFrame] = []
    risk_coverage_curves: list[pd.DataFrame] = []
    ci_rows: list[dict] = []
    segmentation_ci_rows: list[dict] = []
    confusion_2x3_rows: list[dict] = []
    confusion_2x2_rows: list[dict] = []
    abstention_tables: list[pd.DataFrame] = []
    label_3class_tables: list[pd.DataFrame] = []

    segmentation_columns = [
        column for column in (mean_segmentation_columns or ()) if column in frames
    ]
    if segmentation_columns:
        segmentation_levels = {
            "frame": {"ALL": frames, "SAG": frames[frames.side == "SAG"], "SOL": frames[frames.side == "SOL"]},
            "eye": {"ALL": eyes, "SAG": eyes[eyes.side == "SAG"], "SOL": eyes[eyes.side == "SOL"]},
            "patient": {"ALL": patients},
        }
        for level, scopes in segmentation_levels.items():
            report["segmentation"][level] = {}
            for scope, segmentation_table in scopes.items():
                summary = summarize_numeric_columns(segmentation_table, segmentation_columns)
                if bootstrap_draws:
                    segmentation_ci = patient_cluster_continuous_ci(
                        segmentation_table,
                        [column for column in segmentation_columns if column in segmentation_table],
                        draws=bootstrap_draws,
                        seed=bootstrap_seed,
                    )
                    summary["ci95_patient_cluster"] = segmentation_ci
                    segmentation_ci_rows.extend(
                        {"level": level, "scope": scope, "metric": metric, **values}
                        for metric, values in segmentation_ci.items()
                    )
                report["segmentation"][level][scope] = summary

    eye_scopes = {"ALL": eyes, "SAG": eyes[eyes.side == "SAG"], "SOL": eyes[eyes.side == "SOL"]}
    for scope, table in eye_scopes.items():
        evaluation = _scope_evaluation(
            table,
            threshold=eye_threshold,
            temperature=eye_temperature,
            threshold_probability_scale=eye_threshold_probability_scale,
            calibration_bins=calibration_bins,
        )
        if bootstrap_draws:
            evaluation["ci95_patient_cluster"] = patient_cluster_bootstrap_ci(
                table,
                threshold=_evaluation_threshold(eye_threshold),
                draws=bootstrap_draws,
                seed=bootstrap_seed,
                n_calibration_bins=calibration_bins,
            )
            ci_rows.extend(
                {"level": "eye", "scope": scope, "metric": metric, **values}
                for metric, values in evaluation["ci95_patient_cluster"].items()
            )
        report["eye"][scope] = evaluation
        curves, dca = _curves_for_scope(
            table,
            level="eye",
            scope=scope,
            calibration_bins=calibration_bins,
            decision_thresholds=decision_thresholds,
        )
        calibration_curves.extend(curves)
        decision_curves.append(dca)
        for calibration_name, probability_name, validity_name in (
            ("raw", "probability_raw", "roi_evaluable"),
            ("temperature_scaled", "probability_calibrated", "calibrated_evaluable"),
        ):
            if bootstrap_draws:
                band = patient_cluster_calibration_band(
                    table,
                    probability_column=probability_name,
                    evaluable_column=validity_name,
                    n_bins=calibration_bins,
                    draws=bootstrap_draws,
                    seed=bootstrap_seed,
                )
                band.insert(0, "calibration", calibration_name)
                band.insert(0, "scope", scope)
                band.insert(0, "level", "eye")
                calibration_bands.append(band)
        risk_curve = risk_coverage_curve(
            table.label,
            table.probability,
            evaluable=table.evaluable,
            threshold=_evaluation_threshold(eye_threshold),
            include_abstentions_as_failures=True,
        )
        risk_curve.insert(0, "scope", scope)
        risk_curve.insert(0, "level", "eye")
        risk_coverage_curves.append(risk_curve)
        two_by_three, two_by_two = _confusion_rows(evaluation, level="eye", scope=scope)
        confusion_2x3_rows.extend(two_by_three)
        confusion_2x2_rows.extend(two_by_two)
        abstention_tables.append(_abstention_table(table, level="eye", scope=scope))
        subgroup = label_3class_descriptive(table, level="eye", scope=scope)
        if len(subgroup):
            label_3class_tables.append(subgroup)

    patient_evaluation = _scope_evaluation(
        patients,
        threshold=patient_threshold,
        temperature=patient_temperature,
        threshold_probability_scale=patient_threshold_probability_scale,
        calibration_bins=calibration_bins,
    )
    if bootstrap_draws:
        patient_evaluation["ci95_patient_cluster"] = patient_cluster_bootstrap_ci(
            patients,
            threshold=_evaluation_threshold(patient_threshold),
            draws=bootstrap_draws,
            seed=bootstrap_seed,
            n_calibration_bins=calibration_bins,
        )
        ci_rows.extend(
            {"level": "patient", "scope": "ALL", "metric": metric, **values}
            for metric, values in patient_evaluation["ci95_patient_cluster"].items()
        )
    report["patient"]["ALL"] = patient_evaluation
    curves, dca = _curves_for_scope(
        patients,
        level="patient",
        scope="ALL",
        calibration_bins=calibration_bins,
        decision_thresholds=decision_thresholds,
    )
    calibration_curves.extend(curves)
    decision_curves.append(dca)
    for calibration_name, probability_name, validity_name in (
        ("raw", "probability_raw", "roi_evaluable"),
        ("temperature_scaled", "probability_calibrated", "calibrated_evaluable"),
    ):
        if bootstrap_draws:
            band = patient_cluster_calibration_band(
                patients,
                probability_column=probability_name,
                evaluable_column=validity_name,
                n_bins=calibration_bins,
                draws=bootstrap_draws,
                seed=bootstrap_seed,
            )
            band.insert(0, "calibration", calibration_name)
            band.insert(0, "scope", "ALL")
            band.insert(0, "level", "patient")
            calibration_bands.append(band)
    risk_curve = risk_coverage_curve(
        patients.label,
        patients.probability,
        evaluable=patients.evaluable,
        threshold=_evaluation_threshold(patient_threshold),
        include_abstentions_as_failures=True,
    )
    risk_curve.insert(0, "scope", "ALL")
    risk_curve.insert(0, "level", "patient")
    risk_coverage_curves.append(risk_curve)
    two_by_three, two_by_two = _confusion_rows(
        patient_evaluation, level="patient", scope="ALL"
    )
    confusion_2x3_rows.extend(two_by_three)
    confusion_2x2_rows.extend(two_by_two)
    abstention_tables.append(_abstention_table(patients, level="patient", scope="ALL"))
    subgroup = label_3class_descriptive(patients, level="patient", scope="ALL")
    if len(subgroup):
        label_3class_tables.append(subgroup)

    label_3class_table = (
        pd.concat(label_3class_tables, ignore_index=True)
        if label_3class_tables
        else pd.DataFrame()
    )
    report["label_3class_descriptive"] = clean_json(
        label_3class_table.to_dict(orient="records")
    )

    derived_tables = {
        "calibration_curves": (
            pd.concat(calibration_curves, ignore_index=True) if calibration_curves else pd.DataFrame()
        ),
        "calibration_bands": (
            pd.concat(calibration_bands, ignore_index=True) if calibration_bands else pd.DataFrame()
        ),
        "decision_curves": pd.concat(decision_curves, ignore_index=True),
        "risk_coverage_curves": pd.concat(risk_coverage_curves, ignore_index=True),
        "bootstrap_ci": pd.DataFrame(ci_rows),
        "segmentation_bootstrap_ci": pd.DataFrame(segmentation_ci_rows),
        "confusion_2x3": pd.DataFrame(confusion_2x3_rows),
        "confusion_conditional_2x2": pd.DataFrame(confusion_2x2_rows),
        "abstention_reasons": pd.concat(abstention_tables, ignore_index=True),
        "label_3class_descriptive": label_3class_table,
    }
    if classifier_strategy is not None:
        for table in (eyes, patients, *derived_tables.values()):
            table["classifier_strategy"] = str(classifier_strategy)
    return {
        "frames": frames.copy(),
        "eyes": eyes,
        "patients": patients,
        "metrics": report,
        **derived_tables,
    }


def make_classification_plots(bundle: Mapping, output_dir: str | Path) -> None:
    """Create ROC, PR, calibration and decision-curve panels for eye ALL."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    eyes = bundle["eyes"]
    valid = eyes.evaluable.to_numpy(bool)
    probability_column = "probability"
    score_label = "operational"
    if not valid.any() and eyes.roi_evaluable.to_numpy(bool).any():
        valid = eyes.roi_evaluable.to_numpy(bool)
        probability_column = "probability_raw"
        score_label = "raw score only; operating lock unavailable"
    y = eyes.loc[valid, "label"].to_numpy(int)
    p = eyes.loc[valid, probability_column].to_numpy(float)
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    if len(y) and np.unique(y).size == 2:
        fpr, tpr, _ = skm.roc_curve(y, p)
        precision, recall, _ = skm.precision_recall_curve(y, p)
        axes[0, 0].plot(fpr, tpr, label=f"AUROC={skm.roc_auc_score(y, p):.3f}")
        axes[0, 1].plot(recall, precision, label=f"AP={skm.average_precision_score(y, p):.3f}")
        axes[0, 0].legend()
        axes[0, 1].legend()
    axes[0, 0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0, 0].set(title=f"Eye ROC ({score_label})", xlabel="False-positive rate", ylabel="Sensitivity")
    axes[0, 1].set(title=f"Eye precision-recall ({score_label})", xlabel="Recall", ylabel="Precision")

    calibration = bundle["calibration_curves"]
    subset = (
        calibration[
            (calibration.level == "eye")
            & (calibration.scope == "ALL")
            & (calibration.strategy == "uniform")
        ]
        if {"level", "scope", "strategy"}.issubset(calibration.columns)
        else pd.DataFrame()
    )
    if len(subset):
        for name, group in subset.groupby("calibration", sort=False):
            populated = group[group.n > 0]
            axes[1, 0].plot(
                populated.mean_probability, populated.observed_fraction, "o-", label=name
            )
    bands = bundle.get("calibration_bands", pd.DataFrame())
    if len(bands) and {"level", "scope", "calibration", "n"}.issubset(bands.columns):
        preferred_band = (
            "temperature_scaled"
            if (bands.calibration == "temperature_scaled").any()
            else "raw"
        )
        band = bands[
            (bands.level == "eye")
            & (bands.scope == "ALL")
            & (bands.calibration == preferred_band)
            & (bands.n > 0)
        ].sort_values("mean_probability")
        finite_band = band[
            np.isfinite(band.mean_probability)
            & np.isfinite(band.observed_low)
            & np.isfinite(band.observed_high)
        ]
        if len(finite_band):
            axes[1, 0].fill_between(
                finite_band.mean_probability.to_numpy(float),
                finite_band.observed_low.to_numpy(float),
                finite_band.observed_high.to_numpy(float),
                alpha=0.18,
            label=f"patient-cluster 95% band ({preferred_band})",
            )
    axes[1, 0].plot([0, 1], [0, 1], "--", color="gray")
    axes[1, 0].set(
        title="Eye calibration (evaluable)",
        xlabel="Mean predicted probability",
        ylabel="Observed event fraction",
    )
    if len(subset):
        axes[1, 0].legend()

    dca = bundle["decision_curves"]
    dca = dca[(dca.level == "eye") & (dca.scope == "ALL")]
    if len(dca):
        axes[1, 1].plot(dca.threshold, dca.net_benefit_model, label="model")
        axes[1, 1].plot(dca.threshold, dca.net_benefit_treat_all, label="treat all")
        axes[1, 1].plot(dca.threshold, dca.net_benefit_treat_none, label="treat none")
    axes[1, 1].set(title="Eye decision curve (full-cohort denominator)", xlabel="Threshold", ylabel="Net benefit")
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    for extension in ("png", "svg", "pdf"):
        fig.savefig(output / f"classification_calibration_decision_curves.{extension}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for label, name in ((0, "control"), (1, "disease")):
        group = eyes[valid & (eyes.label.to_numpy(int) == label)]
        axes[0].hist(
            group[probability_column],
            bins=np.linspace(0, 1, 21),
            alpha=0.45,
            label=f"{name} (n={len(group)})",
        )
    operational_scale = (
        str(eyes.probability_scale.iloc[0]) if len(eyes) else "unavailable"
    )
    axes[0].set(
        title=f"Eye probability density ({score_label})",
        xlabel=(
            f"Operational probability ({operational_scale})"
            if probability_column == "probability"
            else "Raw predicted probability"
        ),
        ylabel="Count",
    )
    axes[0].legend()
    risk = bundle["risk_coverage_curves"]
    risk = risk[(risk.level == "eye") & (risk.scope == "ALL")]
    if len(risk):
        axes[1].plot(risk.coverage, risk.risk)
        axes[1].set_title(f"Failure-aware risk-coverage (AURC={risk.aurc.iloc[0]:.3f})")
    axes[1].set(xlabel="Coverage", ylabel="Cumulative risk", xlim=(0, 1), ylim=(0, 1))
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    for extension in ("png", "svg", "pdf"):
        fig.savefig(output / f"calibration_density_and_risk_coverage.{extension}", dpi=180)
    plt.close(fig)


def export_evaluation(
    frames: pd.DataFrame,
    output_dir: str | Path,
    *,
    eye_threshold: float | None,
    patient_threshold: float | None,
    eye_temperature: float | None = 1.0,
    patient_temperature: float | None = 1.0,
    eye_threshold_probability_scale: str | None = None,
    patient_threshold_probability_scale: str | None = None,
    eye_lock: Mapping | None = None,
    patient_lock: Mapping | None = None,
    probability_column: str = "probability",
    min_valid_frames: int = 4,
    frames_per_eye: int = 7,
    calibration_bins: int = 10,
    bootstrap_draws: int = 2000,
    bootstrap_seed: int = 1729,
    decision_thresholds: Sequence[float] | None = None,
    mean_segmentation_columns: Sequence[str] | None = None,
    classifier_strategy: str | None = None,
    make_plots: bool = True,
) -> dict:
    """Build and export all primary Q1-ready diagnostic evaluation artefacts."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    bundle = build_evaluation_bundle(
        frames,
        eye_threshold=eye_threshold,
        patient_threshold=patient_threshold,
        eye_temperature=eye_temperature,
        patient_temperature=patient_temperature,
        eye_threshold_probability_scale=eye_threshold_probability_scale,
        patient_threshold_probability_scale=patient_threshold_probability_scale,
        eye_lock=eye_lock,
        patient_lock=patient_lock,
        probability_column=probability_column,
        min_valid_frames=min_valid_frames,
        frames_per_eye=frames_per_eye,
        calibration_bins=calibration_bins,
        bootstrap_draws=bootstrap_draws,
        bootstrap_seed=bootstrap_seed,
        decision_thresholds=decision_thresholds,
        mean_segmentation_columns=mean_segmentation_columns,
        classifier_strategy=classifier_strategy,
    )
    for key, filename in (
        ("frames", "frames.csv"),
        ("eyes", "eyes.csv"),
        ("patients", "patients.csv"),
        ("calibration_curves", "calibration_curves.csv"),
        ("calibration_bands", "calibration_reliability_bands.csv"),
        ("decision_curves", "decision_curves.csv"),
        ("risk_coverage_curves", "risk_coverage_curves.csv"),
        ("bootstrap_ci", "bootstrap_patient_cluster_ci.csv"),
        ("segmentation_bootstrap_ci", "segmentation_patient_cluster_ci.csv"),
        ("confusion_2x3", "confusion_2x3.csv"),
        ("confusion_conditional_2x2", "confusion_conditional_2x2.csv"),
        ("abstention_reasons", "abstention_reasons.csv"),
        ("label_3class_descriptive", "classification_by_label_3class.csv"),
    ):
        bundle[key].to_csv(output / filename, index=False, encoding="utf-8-sig")
    _save_json(output / "metrics.json", bundle["metrics"])
    if make_plots:
        make_classification_plots(bundle, output)
    return bundle


def _optional_float(value) -> float | None:
    return None if value is None else float(value)


def _lock_values(lock: Mapping) -> tuple[float | None, float | None, str | None]:
    """Read either nested metric-helper locks or the engine's flat level lock."""
    calibration = lock.get("calibration", lock)
    threshold_value = lock.get("threshold")
    probability_scale = lock.get("threshold_probability_scale")
    if isinstance(threshold_value, Mapping):
        probability_scale = threshold_value.get("probability_scale", probability_scale)
        threshold_value = threshold_value.get("threshold")
    return (
        _optional_float(calibration.get("temperature")),
        _optional_float(threshold_value),
        probability_scale,
    )


def build_evaluation_bundle_from_config(
    frames: pd.DataFrame,
    cfg: Mapping,
    *,
    eye_lock: Mapping,
    patient_lock: Mapping,
    probability_column: str = "probability",
    classifier_strategy: str | None = None,
) -> dict:
    """Config-driven wrapper that carries every locked reporting choice forward."""
    eye_temperature, eye_threshold, eye_scale = _lock_values(eye_lock)
    patient_temperature, patient_threshold, patient_scale = _lock_values(patient_lock)
    aggregation = cfg["aggregation"]
    calibration = cfg["calibration"]
    statistics = cfg["statistics"]
    evaluation = cfg["evaluation"]
    dca = evaluation["decision_curve"]
    decision_thresholds = np.arange(
        float(dca["threshold_start"]),
        float(dca["threshold_stop"]) + float(dca["threshold_step"]) / 2,
        float(dca["threshold_step"]),
    )
    bundle = build_evaluation_bundle(
        frames,
        eye_threshold=eye_threshold,
        patient_threshold=patient_threshold,
        eye_temperature=eye_temperature,
        patient_temperature=patient_temperature,
        eye_threshold_probability_scale=eye_scale,
        patient_threshold_probability_scale=patient_scale,
        probability_column=probability_column,
        min_valid_frames=int(aggregation["minimum_valid_frames"]),
        frames_per_eye=int(cfg["dataset"]["frames_per_eye"]),
        calibration_bins=int(calibration["ece_bins"]),
        bootstrap_draws=int(statistics["bootstrap_draws"]),
        bootstrap_seed=int(statistics["bootstrap_seed"]),
        decision_thresholds=decision_thresholds,
        mean_segmentation_columns=cfg["segmentation"]["metrics"],
        classifier_strategy=classifier_strategy,
    )
    bundle["metrics"]["locked_analysis_config"] = {
        "aggregation": aggregation,
        "calibration": calibration,
        "evaluation": evaluation,
        "statistics": statistics,
        "surface_dice_tolerance_pixels": cfg["segmentation"][
            "surface_dice_tolerance_pixels"
        ],
        "eye_lock": eye_lock,
        "patient_lock": patient_lock,
    }
    return bundle


def export_evaluation_from_config(
    frames: pd.DataFrame,
    output_dir: str | Path,
    cfg: Mapping,
    *,
    eye_lock: Mapping,
    patient_lock: Mapping,
    probability_column: str = "probability",
    classifier_strategy: str | None = None,
    make_plots: bool = True,
) -> dict:
    """Config-driven export; intended entry point for the training/evaluation engine."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    bundle = build_evaluation_bundle_from_config(
        frames,
        cfg,
        eye_lock=eye_lock,
        patient_lock=patient_lock,
        probability_column=probability_column,
        classifier_strategy=classifier_strategy,
    )
    for key, filename in (
        ("frames", "frames.csv"),
        ("eyes", "eyes.csv"),
        ("patients", "patients.csv"),
        ("calibration_curves", "calibration_curves.csv"),
        ("calibration_bands", "calibration_reliability_bands.csv"),
        ("decision_curves", "decision_curves.csv"),
        ("risk_coverage_curves", "risk_coverage_curves.csv"),
        ("bootstrap_ci", "bootstrap_patient_cluster_ci.csv"),
        ("segmentation_bootstrap_ci", "segmentation_patient_cluster_ci.csv"),
        ("confusion_2x3", "confusion_2x3.csv"),
        ("confusion_conditional_2x2", "confusion_conditional_2x2.csv"),
        ("abstention_reasons", "abstention_reasons.csv"),
        ("label_3class_descriptive", "classification_by_label_3class.csv"),
    ):
        bundle[key].to_csv(output / filename, index=False, encoding="utf-8-sig")
    _save_json(output / "metrics.json", bundle["metrics"])
    if make_plots:
        make_classification_plots(bundle, output)
    return bundle


def export_ablation_evaluation(
    table: pd.DataFrame,
    output_dir: str | Path,
    *,
    thresholds: float | Mapping[str, float],
    reference_arm: str = "predicted_roi",
    comparison_metric_paths: Sequence[str] = (
        "failure_inclusive.balanced_accuracy",
        "conditional.auroc",
        "conditional.brier",
    ),
    bootstrap_draws: int = 2000,
    bootstrap_seed: int = 2718,
) -> dict:
    """Export aligned ablation arms and paired cluster-bootstrap contrasts."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    arms = validate_ablation_alignment(table)
    if reference_arm not in arms:
        raise ValueError("reference_arm is absent from the aligned table.")
    metrics = evaluate_ablation_arms(table, thresholds=thresholds)
    metric_rows = []
    for arm, values in metrics.items():
        flat = flatten_metrics(values)
        metric_rows.append({"arm": arm, **flat})
    metric_table = pd.DataFrame(metric_rows)

    comparisons = []
    if bootstrap_draws:
        for arm in arms:
            if arm == reference_arm:
                continue
            for metric_path in comparison_metric_paths:
                comparisons.append(
                    paired_cluster_bootstrap_difference(
                        table,
                        reference_arm,
                        arm,
                        metric_path=metric_path,
                        thresholds=thresholds,
                        draws=bootstrap_draws,
                        seed=bootstrap_seed,
                    )
                )
    comparison_table = pd.DataFrame(comparisons)
    classical_rows = []
    indexed = {
        arm: table[table.arm.astype(str) == arm].set_index("unit_id").sort_index()
        for arm in arms
    }

    def threshold_for(arm: str) -> float:
        return float(thresholds[arm]) if isinstance(thresholds, Mapping) else float(thresholds)

    for arm in arms:
        if arm == reference_arm:
            continue
        reference = indexed[reference_arm]
        comparator = indexed[arm]
        delong = delong_auc_comparison(
            reference.label,
            reference.probability,
            comparator.probability,
            evaluable_a=reference.evaluable,
            evaluable_b=comparator.evaluable,
        )
        classical_rows.append(
            {
                "reference_arm": reference_arm,
                "comparator_arm": arm,
                "test": "delong_auroc_common_evaluable",
                "effect": delong["difference_a_minus_b"],
                "p_value": delong["p_value"],
                "n_common_evaluable": delong["n_common_evaluable"],
                "status_or_note": delong["status"],
            }
        )
        mcnemar = mcnemar_exact_comparison(
            reference.label,
            reference.probability,
            comparator.probability,
            threshold_a=threshold_for(reference_arm),
            threshold_b=threshold_for(arm),
            evaluable_a=reference.evaluable,
            evaluable_b=comparator.evaluable,
        )
        classical_rows.append(
            {
                "reference_arm": reference_arm,
                "comparator_arm": arm,
                "test": "mcnemar_exact_common_evaluable",
                "effect": mcnemar["conditional_accuracy_difference_a_minus_b"],
                "p_value": mcnemar["p_value_exact_two_sided"],
                "n_common_evaluable": mcnemar["n_common_evaluable"],
                "status_or_note": mcnemar["assumption_note"],
            }
        )
    classical_table = pd.DataFrame(classical_rows)
    if len(classical_table):
        classical_table["p_value_holm"] = np.nan
        classical_table["reject_holm_0_05"] = False
        for _, indices in classical_table.groupby("test").groups.items():
            adjusted, rejected = holm_adjust(classical_table.loc[indices, "p_value"])
            classical_table.loc[indices, "p_value_holm"] = adjusted
            classical_table.loc[indices, "reject_holm_0_05"] = rejected
    table.to_csv(output / "ablation_predictions.csv", index=False, encoding="utf-8-sig")
    metric_table.to_csv(output / "ablation_metrics.csv", index=False, encoding="utf-8-sig")
    comparison_table.to_csv(
        output / "ablation_paired_patient_cluster_differences.csv",
        index=False,
        encoding="utf-8-sig",
    )
    classical_table.to_csv(
        output / "ablation_classical_comparisons_holm.csv",
        index=False,
        encoding="utf-8-sig",
    )
    manifest = {
        "reference_arm": reference_arm,
        "arms": {arm: ABLATION_ARM_METADATA.get(arm, {"role": "custom"}) for arm in arms},
        "required_core_arms": list(CORE_ABLATION_ARMS),
        "present_core_arms": [arm for arm in CORE_ABLATION_ARMS if arm in arms],
        "inference_warning": (
            "Ablations are aligned paired analyses. GT oracle is non-deployable; "
            "whole-image, mask-only, background-only, and ROI-shuffle are controls."
        ),
    }
    _save_json(output / "ablation_manifest.json", manifest)
    return {
        "metrics": metrics,
        "metrics_table": metric_table,
        "comparisons": comparison_table,
        "classical_comparisons": classical_table,
        "manifest": manifest,
    }


def export_five_seed_summary(
    per_seed_metrics: pd.DataFrame,
    output_file: str | Path,
    *,
    metric_columns: Sequence[str],
    group_columns: Sequence[str] = ("model", "level", "scope", "arm"),
    seed_column: str = "seed",
) -> pd.DataFrame:
    """Write mean +/- sample SD over exactly five seed-level estimates."""
    summary = summarize_across_seeds(
        per_seed_metrics,
        metric_columns=metric_columns,
        group_columns=group_columns,
        seed_column=seed_column,
        expected_seeds=5,
    )
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False, encoding="utf-8-sig")
    return summary
