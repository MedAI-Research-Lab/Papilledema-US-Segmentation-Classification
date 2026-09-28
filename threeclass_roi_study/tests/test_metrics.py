from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from threeclass_roi_study.metrics import (
    ABSTAIN,
    CLASS_NAMES,
    DEFAULT_BOOTSTRAP_METRICS,
    aggregate_eyes_to_patients,
    aggregate_frames_to_eyes,
    apply_temperature_scaling,
    apply_temperature_to_probabilities,
    clean_json,
    conditional_multiclass_metrics,
    confusion_matrix_3x4,
    fit_temperature_scaling,
    multiclass_risk_coverage_curve,
    patient_cluster_bootstrap_ci,
    patient_cluster_ratio_bootstrap_ci,
    selective_multiclass_metrics,
)


def _perfect_labels_probabilities(repeats: int = 2) -> tuple[np.ndarray, np.ndarray]:
    labels = np.tile(np.arange(3), repeats)
    probability = np.full((len(labels), 3), 0.05)
    probability[np.arange(len(labels)), labels] = 0.90
    return labels, probability


def _frame_table() -> pd.DataFrame:
    rows = []
    specifications = [
        ("p0", "p0_R", "SAG", 0, 7),
        ("p0", "p0_L", "SOL", 0, 7),
        ("p1", "p1_R", "SAG", 1, 7),
        ("p1", "p1_L", "SOL", 1, 3),
    ]
    for patient, case, side, label, valid_count in specifications:
        for frame in range(7):
            valid = frame < valid_count
            vector = np.full(3, 0.05)
            vector[label] = 0.90
            rows.append(
                {
                    "patient_id": patient,
                    "case_id": case,
                    "side": side,
                    "frame_id": frame,
                    "label_3class": label,
                    "roi_valid": bool(valid),
                    "probability_0": vector[0] if valid else np.nan,
                    "probability_1": vector[1] if valid else np.nan,
                    "probability_2": vector[2] if valid else np.nan,
                    "abstention_reason": "" if valid else "empty_prediction",
                }
            )
    return pd.DataFrame(rows)


def test_temperature_scaling_is_multiclass_and_does_not_increase_fit_nll():
    y = np.array([0, 1, 2, 0, 1, 2])
    logits = np.array(
        [
            [8.0, 0.0, 0.0],
            [0.0, 8.0, 0.0],
            [0.0, 0.0, 8.0],
            [3.0, 2.0, 1.0],
            [2.0, 3.0, 1.0],
            [1.0, 2.0, 3.0],
        ]
    )
    fitted = fit_temperature_scaling(y, logits)
    probability = apply_temperature_scaling(logits, fitted["temperature"])
    probability_roundtrip = apply_temperature_to_probabilities(
        apply_temperature_scaling(logits, 1.0), fitted["temperature"]
    )
    assert fitted["status"].startswith("ok")
    assert fitted["nll_after"] <= fitted["nll_before"] + 1e-9
    assert probability.shape == (6, 3)
    np.testing.assert_allclose(probability.sum(axis=1), 1.0)
    np.testing.assert_allclose(probability, probability_roundtrip, atol=1e-6)


def test_temperature_scaling_missing_validation_class_has_no_identity_fallback():
    fitted = fit_temperature_scaling(
        [0, 1, 0, 1], np.array([[2, 0, -1], [0, 2, -1], [1, 0, -1], [0, 1, -1]])
    )
    assert fitted["status"] == "unavailable_missing_validation_class_no_default_temperature"
    assert np.isnan(fitted["temperature"])
    assert fitted["missing_classes"] == [2]


def test_conditional_metrics_report_every_class_macro_calibration_and_proper_scores():
    y, probability = _perfect_labels_probabilities()
    result = conditional_multiclass_metrics(y, probability, n_calibration_bins=3)
    assert result["accuracy"] == 1.0
    assert result["balanced_accuracy"] == 1.0
    assert result["macro_f1"] == 1.0
    assert result["multiclass_mcc"] == 1.0
    assert result["macro_auroc"] == 1.0
    assert result["macro_average_precision"] == 1.0
    assert 0 < result["multiclass_brier"] < 0.02
    assert 0 < result["multiclass_nll"] < 0.2
    assert set(result["per_class"]) == {"normal", "papilledema", "pseudopapilledema"}
    for metrics in result["per_class"].values():
        assert metrics["f1"] == 1.0
        assert metrics["auroc_ovr"] == 1.0
        assert metrics["average_precision_ovr"] == 1.0
        assert "brier_ovr" in metrics and "nll_ovr" in metrics and "ece_ovr" in metrics
    assert result["derived_discrimination"]["normal_vs_abnormal"]["auroc"] == 1.0
    assert (
        result["derived_discrimination"]["papilledema_vs_pseudopapilledema"]["auroc"]
        == 1.0
    )


def test_selective_metrics_expose_3x4_matrix_and_penalize_abstention():
    y, probability = _perfect_labels_probabilities()
    evaluable = np.array([True, True, True, True, True, False])
    probability[~evaluable] = np.nan
    result = selective_multiclass_metrics(
        y,
        probability,
        evaluable=evaluable,
        localized_success=[True, True, True, True, False, True],
    )
    assert ABSTAIN == 3
    assert result["coverage"] == pytest.approx(5 / 6)
    assert result["conditional"]["accuracy"] == 1.0
    assert result["failure_aware"]["accuracy"] == pytest.approx(5 / 6)
    assert result["failure_aware"]["balanced_accuracy"] == pytest.approx(5 / 6)
    assert result["prediction"][-1] == ABSTAIN
    assert result["confusion_matrix_3x4"] == [[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 1, 1]]
    assert result["class_conditional_coverage"]["pseudopapilledema"] == 0.5
    assert result["localized_diagnostic_success"]["rate"] == pytest.approx(4 / 6)


def test_confusion_matrix_rejects_unknown_outcomes_and_has_row_proportions():
    matrix, proportions = confusion_matrix_3x4([0, 1, 2], [0, ABSTAIN, 1])
    assert matrix.tolist() == [[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]]
    np.testing.assert_allclose(proportions.sum(axis=1), 1.0)
    with pytest.raises(ValueError):
        confusion_matrix_3x4([0], [4])


def test_risk_coverage_orders_confident_units_then_appends_abstention():
    y = np.array([0, 1, 2, 2])
    probability = np.array(
        [[0.95, 0.03, 0.02], [0.1, 0.8, 0.1], [0.2, 0.3, 0.5], [np.nan] * 3]
    )
    curve = multiclass_risk_coverage_curve(
        y, probability, evaluable=[True, True, True, False]
    )
    assert curve.unit_index.tolist() == [0, 1, 2, 3]
    assert np.isnan(curve.confidence.iloc[-1])
    assert curve.coverage.iloc[-1] == 1.0
    assert curve.risk.iloc[-1] == 0.25
    assert 0 <= curve.aurc.iloc[0] <= 1


def test_frame_and_patient_vector_aggregation_enforces_structural_abstention():
    eyes = aggregate_frames_to_eyes(_frame_table(), min_valid_frames=4)
    eyes = eyes.sort_values(["patient_id", "side"]).reset_index(drop=True)
    p0 = eyes[eyes.patient_id == "p0"]
    assert p0.evaluable.all()
    assert set(p0.prediction) == {0}
    assert set(p0.label_3class) == {0}
    p1 = eyes[eyes.patient_id == "p1"]
    assert p1.evaluable.sum() == 1
    assert p1.loc[~p1.evaluable, "prediction"].iloc[0] == ABSTAIN
    assert p1.loc[~p1.evaluable, "abstention_reason"].iloc[0] == "insufficient_valid_frames"

    patients = aggregate_eyes_to_patients(eyes).set_index("patient_id")
    assert patients.loc["p0", "evaluable"]
    assert patients.loc["p0", "prediction"] == 0
    assert patients.loc["p0", "label_3class"] == 0
    assert not patients.loc["p1", "evaluable"]
    assert patients.loc["p1", "prediction"] == ABSTAIN
    assert patients.loc["p1", "abstention_reason"] == "one_or_more_eyes_abstained"


def test_aggregation_rejects_probability_on_invalid_roi():
    frames = _frame_table()
    invalid = frames.index[~frames.roi_valid][0]
    frames.loc[invalid, ["probability_0", "probability_1", "probability_2"]] = [0.8, 0.1, 0.1]
    with pytest.raises(ValueError, match="hidden fallback"):
        aggregate_frames_to_eyes(frames)


def test_patient_cluster_bootstrap_is_reproducible_and_strict_json_serializable():
    rows = []
    for label in range(3):
        for patient_number in range(3):
            vector = np.full(3, 0.1)
            vector[label] = 0.8
            for side in ("SAG", "SOL"):
                rows.append(
                    {
                        "patient_id": f"c{label}p{patient_number}",
                        "side": side,
                        "label_3class": label,
                        "evaluable": not (label == 2 and patient_number == 2 and side == "SOL"),
                        "probability_0": vector[0],
                        "probability_1": vector[1],
                        "probability_2": vector[2],
                    }
                )
    table = pd.DataFrame(rows)
    invalid = ~table.evaluable
    table.loc[invalid, ["probability_0", "probability_1", "probability_2"]] = np.nan
    first = patient_cluster_bootstrap_ci(
        table,
        draws=40,
        seed=7,
        metric_paths=("coverage", "failure_aware.balanced_accuracy", "conditional.macro_auroc"),
    )
    second = patient_cluster_bootstrap_ci(
        table,
        draws=40,
        seed=7,
        metric_paths=("coverage", "failure_aware.balanced_accuracy", "conditional.macro_auroc"),
    )
    assert first == second
    assert first["coverage"]["valid_draws"] == 40
    assert first["coverage"]["resampling_unit"] == "whole_patient_cluster"
    # allow_nan=False proves clean_json removes every NumPy scalar and NaN.
    json.dumps(clean_json(first), allow_nan=False)


def test_default_bootstrap_metrics_include_every_required_classwise_endpoint():
    paths = set(DEFAULT_BOOTSTRAP_METRICS)
    for class_name in CLASS_NAMES:
        assert {
            f"class_conditional_coverage.{class_name}",
            f"failure_aware.per_class.{class_name}.recall",
            f"failure_aware.per_class.{class_name}.precision",
            f"failure_aware.per_class.{class_name}.f1",
            f"conditional.per_class.{class_name}.recall",
            f"conditional.per_class.{class_name}.precision",
            f"conditional.per_class.{class_name}.f1",
            f"conditional.per_class.{class_name}.auroc_ovr",
            f"conditional.per_class.{class_name}.average_precision_ovr",
        } <= paths


def test_patient_cluster_bootstrap_is_invariant_to_input_row_order():
    rows = []
    for label in range(3):
        for patient_number in range(4):
            predicted = label if patient_number != 0 else (label + 1) % 3
            vector = np.full(3, 0.05)
            vector[predicted] = 0.90
            for side in ("SAG", "SOL"):
                rows.append(
                    {
                        "patient_id": f"c{label}p{patient_number}",
                        "side": side,
                        "label_3class": label,
                        "evaluable": True,
                        "probability_0": vector[0],
                        "probability_1": vector[1],
                        "probability_2": vector[2],
                    }
                )
    table = pd.DataFrame(rows)
    paths = (
        "failure_aware.balanced_accuracy",
        "conditional.per_class.normal.auroc_ovr",
    )
    ordered = patient_cluster_bootstrap_ci(
        table, draws=100, seed=23, metric_paths=paths
    )
    shuffled = patient_cluster_bootstrap_ci(
        table.sample(frac=1, random_state=91),
        draws=100,
        seed=23,
        metric_paths=paths,
    )
    assert ordered == shuffled


def test_patient_cluster_ratio_bootstrap_reports_draws_seed_and_fallback():
    result = patient_cluster_ratio_bootstrap_ci(
        [7.0, 3.0, 5.0],
        [7.0, 7.0, 7.0],
        draws=50,
        seed=1701,
    )
    assert result["estimate"] == pytest.approx(15 / 21)
    assert result["requested_draws"] == 50
    assert result["valid_draws"] == 50
    assert result["bootstrap_seed"] == 1701
    assert result["resampling_unit"] == "whole_patient_cluster"
    assert result["stratification"] == "three_class_patient_label"
    assert result["fallback_used"] == (result["method"] != "bca")


def test_clean_json_converts_undefined_metrics_to_null():
    payload = clean_json({"x": np.nan, "i": np.int64(2), "a": np.array([1.0, np.inf])})
    assert payload == {"x": None, "i": 2, "a": [1.0, None]}
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload
