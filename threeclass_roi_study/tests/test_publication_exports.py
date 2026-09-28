from __future__ import annotations

import json

import pandas as pd
import pytest

from threeclass_roi_study.config import load_config
from threeclass_roi_study import publication_exports as publication_exports_module
from threeclass_roi_study.publication_exports import (
    _assert_cross_table_consistency,
    _patient_metric_row,
    _patient_prediction_rows,
    _segmentation_bootstrap_seed,
    _segmentation_rows,
    _segmentation_uncertainty_rows,
    _uncertainty_rows,
    _unit_metrics,
)
from threeclass_roi_study.reporting import ReportingSchemaError, validate_output_rows
from threeclass_roi_study.protocol import ProtocolGateError


def _patients() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "patient_id": "P0",
                "label_3class": 0,
                "evaluable": True,
                "prediction": 0,
                "abstention_reason": "",
                "probability_0": 0.8,
                "probability_1": 0.1,
                "probability_2": 0.1,
            },
            {
                "patient_id": "P1",
                "label_3class": 1,
                "evaluable": True,
                "prediction": 1,
                "abstention_reason": "",
                "probability_0": 0.1,
                "probability_1": 0.8,
                "probability_2": 0.1,
            },
            {
                "patient_id": "P2",
                "label_3class": 2,
                "evaluable": True,
                "prediction": 2,
                "abstention_reason": "",
                "probability_0": 0.1,
                "probability_1": 0.1,
                "probability_2": 0.8,
            },
        ]
    )


def _eyes() -> pd.DataFrame:
    rows = []
    for patient_id in ("P0", "P1", "P2"):
        for side in ("SAG", "SOL"):
            rows.append(
                {
                    "patient_id": patient_id,
                    "case_id": f"{patient_id}_{side}",
                    "side": side,
                    "evaluable": True,
                    "n_valid_frames": 7,
                }
            )
    return pd.DataFrame(rows)


def test_publication_patient_rows_include_auditable_localized_success() -> None:
    cfg = load_config()
    patients = _patients()
    rows = _patient_prediction_rows(
        cfg,
        "yolo26",
        17,
        "model_specific",
        patients,
        patients,
        _eyes(),
        {"patient": {"status": "available", "temperature": 1.2}},
        {"P0": True, "P1": False, "P2": True},
    )
    validated = validate_output_rows("patient_predictions.csv", rows)
    assert [row["localized_diagnostic_success"] for row in validated] == [True, False, True]
    assert all(row["prior_test_use_disclosed"] for row in validated)


def test_publication_patient_metric_reports_joint_localized_rate() -> None:
    cfg = load_config()
    result = _unit_metrics(_patients(), localized=[True, False, True])
    row = _patient_metric_row(
        cfg, "yolo26", 17, "model_specific", "calibrated", result
    )
    validated = validate_output_rows("patient_metrics.csv", [row])
    assert validated[0]["localized_rate"] == 2 / 3
    assert validated[0]["localized_diagnostic_success_rate"] == 2 / 3


def _bootstrap_record(
    *, method: str, fallback: bool, bootstrap_seed: int = 50017
) -> dict:
    return {
        "estimate": 0.75,
        "low": 0.5,
        "high": 0.9,
        "bootstrap_se": 0.1,
        "valid_draws": 5000,
        "requested_draws": 5000,
        "confidence": 0.95,
        "method": method,
        "requested_method": "bca",
        "fallback_used": fallback,
        "fallback_reason": (
            "bca_not_identified_percentile_used" if fallback else None
        ),
        "resampling_unit": "whole_patient_cluster",
        "stratified_by_label": True,
        "stratification": "three_class_patient_label",
        "bootstrap_seed": bootstrap_seed,
    }


def test_uncertainty_rows_flatten_stored_calibrated_bootstrap(
    tmp_path,
) -> None:
    cfg = load_config()
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    payload = {
        "patient_calibrated": {
            "coverage": _bootstrap_record(method="bca", fallback=False)
        },
        "eye_calibrated": {
            "coverage": _bootstrap_record(
                method="percentile", fallback=True, bootstrap_seed=50018
            )
        },
    }
    (evaluation / "bootstrap_patient_cluster_ci.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    rows = _uncertainty_rows(
        cfg, "yolo26", 17, "model_specific", evaluation
    )
    validated = validate_output_rows("uncertainty_metrics.csv", rows)
    assert len(validated) == 2
    patient = next(row for row in validated if row["level"] == "patient")
    eye = next(row for row in validated if row["level"] == "eye")
    assert patient["bootstrap_seed"] == 50017
    assert not patient["fallback_used"]
    assert eye["bootstrap_seed"] == 50018
    assert eye["interval_method"] == "percentile"
    assert eye["fallback_used"]
    assert eye["fallback_detail"] == "bca_not_identified_percentile_used"


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("requested_draws", 5000.0),
        ("valid_draws", "5000"),
        ("confidence", "0.95"),
        ("stratified_by_label", 1),
        ("fallback_used", "false"),
        ("bootstrap_seed", 50017.0),
        ("estimate", "0.75"),
    ],
)
def test_uncertainty_source_json_types_fail_closed(tmp_path, field, invalid) -> None:
    cfg = load_config()
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    patient = _bootstrap_record(method="bca", fallback=False)
    eye = _bootstrap_record(
        method="bca", fallback=False, bootstrap_seed=50018
    )
    patient[field] = invalid
    payload = {
        "patient_calibrated": {"coverage": patient},
        "eye_calibrated": {"coverage": eye},
    }
    (evaluation / "bootstrap_patient_cluster_ci.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    with pytest.raises(ProtocolGateError):
        _uncertainty_rows(cfg, "yolo26", 17, "model_specific", evaluation)


@pytest.mark.parametrize(
    "updates",
    [
        {"bootstrap_se": -0.01},
        {"bootstrap_se": None},
        {"valid_draws": 0},
        {
            "interval_method": "percentile",
            "fallback_used": True,
            "fallback_detail": "   ",
        },
    ],
)
def test_uncertainty_output_invariants_fail_closed(updates) -> None:
    row = {
        "model": "yolo26",
        "seed": 17,
        "classifier_strategy": "model_specific",
        "level": "patient",
        "probability_scale": "calibrated",
        "metric_path": "coverage",
        "estimate": 0.75,
        "ci_95_low": 0.5,
        "ci_95_high": 0.9,
        "bootstrap_se": 0.1,
        "confidence_level": 0.95,
        "requested_method": "bca",
        "interval_method": "bca",
        "fallback_used": False,
        "fallback_detail": None,
        "valid_draws": 5000,
        "requested_draws": 5000,
        "resampling_unit": "whole_patient_cluster",
        "stratified_by_label": True,
        "stratification": "three_class_patient_label",
        "bootstrap_seed": 50017,
    }
    row.update(updates)
    with pytest.raises(ReportingSchemaError):
        validate_output_rows("uncertainty_metrics.csv", [row])


def test_patient_uncertainty_estimates_must_match_patient_metrics() -> None:
    identity = {
        "model": "yolo26",
        "seed": 17,
        "classifier_strategy": "model_specific",
    }
    predictions = [{**identity, "abstained": False}]
    metrics_row = {
        **identity,
        "probability_scale": "calibrated",
        "n_intended": 1,
        "n_abstained": 0,
        "coverage": 1.0,
        "failure_aware_balanced_accuracy": 0.5,
        "conditional_balanced_accuracy": 0.6,
        "accuracy": 0.7,
        "macro_f1": 0.8,
        "multiclass_mcc": 0.2,
        "macro_ovr_auroc": 0.9,
        "macro_ovr_average_precision": 0.85,
        "multiclass_nll": 0.4,
        "multiclass_brier": 0.3,
        "aurc": 0.1,
        "localized_diagnostic_success_rate": 0.25,
    }
    confusion = [
        {
            **identity,
            "true_label": actual,
            "predicted_label": predicted,
            "count": int(actual == 0 and predicted == 0),
        }
        for actual in range(3)
        for predicted in range(4)
    ]
    mapping = {
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
    uncertainty = [
        {
            **identity,
            "level": "patient",
            "metric_path": metric_path,
            "estimate": metrics_row[column],
        }
        for metric_path, column in mapping.items()
    ]
    _assert_cross_table_consistency(
        predictions, [metrics_row], confusion, uncertainty
    )
    uncertainty[1]["estimate"] = 0.123
    with pytest.raises(ProtocolGateError, match="point estimate disagrees"):
        _assert_cross_table_consistency(
            predictions, [metrics_row], confusion, uncertainty
        )


def _segmentation_frames() -> pd.DataFrame:
    rows = []
    valid_counts = ((7, 7), (4, 3), (5, 6))
    for class_label in range(3):
        for patient_number, counts in enumerate(valid_counts):
            patient_id = f"C{class_label}P{patient_number}"
            for eye_index, (side, valid_count) in enumerate(
                zip(("SAG", "SOL"), counts)
            ):
                for frame_index in range(7):
                    valid = frame_index < valid_count
                    rows.append(
                        {
                            "patient_id": patient_id,
                            "case_id": f"{patient_id}_{side}",
                            "side": side,
                            "frame_id": str(frame_index),
                            "label_3class": class_label,
                            "segmentation_roi_valid": bool(valid),
                            "segmentation_abstention_reason": "" if valid else "empty",
                            "dice": 0.30
                            + class_label * 0.10
                            + patient_number * 0.03
                            + eye_index * 0.01
                            + frame_index * 0.001,
                            "iou": 0.20
                            + class_label * 0.08
                            + patient_number * 0.02
                            + eye_index * 0.01
                            + frame_index * 0.001,
                        }
                    )
    return pd.DataFrame(rows)


def test_segmentation_uncertainty_preserves_point_estimands_and_clusters_patients(
    monkeypatch,
) -> None:
    cfg = load_config()
    frame = _segmentation_frames()
    monkeypatch.setattr(
        publication_exports_module,
        "_source_segmentation_frames",
        lambda _cfg, _model, _seed: frame.copy(),
    )
    points = _segmentation_rows(cfg, "yolo26", 17)
    uncertainty = _segmentation_uncertainty_rows(cfg, "yolo26", 17)
    validated = validate_output_rows("segmentation_uncertainty.csv", uncertainty)
    assert len(validated) == 3 * 3 * 3
    point_lookup = {
        (row["class_label"], row["level"]): row for row in points
    }
    for row in validated:
        assert row["estimate"] == pytest.approx(
            point_lookup[(row["class_label"], row["level"])][row["metric"]]
        )
        assert row["requested_draws"] == 5000
        assert row["valid_draws"] == 5000
        assert row["resampling_unit"] == "whole_patient_cluster"
        assert row["stratified_by_label"]
        assert row["gt_used_for_inference_selection_or_abstention"] is False
        expected_role = (
            "strict_predicted_roi_coverage_no_reference_standard"
            if row["metric"] == "roi_coverage"
            else "retrospective_reference_standard_overlap_only"
        )
        assert row["metric_role"] == expected_role


def test_segmentation_bootstrap_seed_is_stable_and_derived_from_model_and_split() -> None:
    seed = _segmentation_bootstrap_seed("yolo26", 17)
    assert seed == _segmentation_bootstrap_seed("yolo26", 17)
    assert seed != _segmentation_bootstrap_seed("sam2_unet", 17)
    assert seed != _segmentation_bootstrap_seed("yolo26", 42)
