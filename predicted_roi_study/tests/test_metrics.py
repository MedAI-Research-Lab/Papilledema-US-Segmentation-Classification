from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from predicted_roi_study.metrics import (
    ABSTAIN,
    aggregate_eyes_to_patients,
    aggregate_frames_to_eyes,
    apply_temperature_scaling,
    calibration_curve_table,
    calibration_metrics,
    confusion_matrix_2x3,
    decision_curve_net_benefit,
    delong_auc_comparison,
    evaluate_ablation_arms,
    fit_temperature_scaling,
    holm_adjust,
    mcnemar_exact_comparison,
    paired_cluster_bootstrap_difference,
    patient_cluster_bootstrap_ci,
    segmentation_metrics,
    selective_binary_metrics,
    summarize_across_seeds,
    validate_ablation_alignment,
)
from predicted_roi_study.reporting import (
    build_evaluation_bundle,
    build_evaluation_bundle_from_config,
    export_evaluation,
    label_3class_descriptive,
)


def _frames() -> pd.DataFrame:
    rows = []
    specification = {
        ("p0", "SAG"): (0, 4, 0.10),
        ("p0", "SOL"): (0, 7, 0.20),
        ("p1", "SAG"): (1, 3, 0.80),
        ("p1", "SOL"): (1, 7, 0.90),
    }
    for (patient, side), (label, valid_count, probability) in specification.items():
        for frame in range(7):
            valid = frame < valid_count
            rows.append(
                {
                    "patient_id": patient,
                    "case_id": f"{patient}_{side}",
                    "side": side,
                    "frame_id": frame,
                    "label": label,
                    "label_3class": label,
                    "roi_valid": valid,
                    "probability": probability if valid else np.nan,
                    "abstention_reason": "" if valid else "tiny",
                    "roi_hit": int(valid),
                    "dice": 0.8 if valid else 0.0,
                }
            )
    return pd.DataFrame(rows)


def test_segmentation_metrics_keeps_empty_prediction_as_failure():
    gt = np.zeros((8, 8), dtype=bool)
    gt[2:5, 3:6] = True
    result = segmentation_metrics(np.zeros_like(gt), gt)
    assert result["dice"] == 0
    assert result["empty_prediction"] == 1
    assert result["target_fn"] == 1
    assert result["distance_valid"] == 0
    assert np.isnan(result["hd95_px"])


def test_eye_aggregation_requires_four_of_seven_without_fallback():
    eyes = aggregate_frames_to_eyes(_frames(), threshold=0.5)
    evaluable = eyes.set_index("case_id").evaluable.to_dict()
    assert evaluable["p0_SAG"] is True or bool(evaluable["p0_SAG"])
    assert not bool(evaluable["p1_SAG"])
    p1_sag = eyes.set_index("case_id").loc["p1_SAG"]
    assert np.isnan(p1_sag.probability)
    assert p1_sag.prediction == ABSTAIN
    assert p1_sag.abstention_reason == "insufficient_valid_frames"
    assert eyes.set_index("case_id").loc["p0_SAG", "probability"] == pytest.approx(0.1)


def test_invalid_roi_probability_is_rejected_as_possible_leakage():
    frames = _frames()
    index = frames.index[~frames.roi_valid][0]
    frames.loc[index, "probability"] = 0.42
    with pytest.raises(ValueError, match="could leak a fallback"):
        aggregate_frames_to_eyes(frames)


def test_patient_abstains_when_either_eye_abstains():
    eyes = aggregate_frames_to_eyes(_frames())
    patients = aggregate_eyes_to_patients(eyes).set_index("patient_id")
    assert bool(patients.loc["p0", "evaluable"])
    assert patients.loc["p0", "probability"] == pytest.approx(0.15)
    assert not bool(patients.loc["p1", "evaluable"])
    assert np.isnan(patients.loc["p1", "probability"])
    assert patients.loc["p1", "prediction"] == ABSTAIN


def test_selective_metrics_expose_2x3_and_failure_inclusive_results():
    y = np.array([0, 0, 1, 1])
    probability = np.array([0.1, np.nan, 0.9, np.nan])
    result = selective_binary_metrics(y, probability, threshold=0.5)
    assert result["confusion_matrix_2x3"] == [[1, 0, 1], [0, 1, 1]]
    assert result["conditional"]["accuracy"] == 1.0
    assert result["coverage"] == 0.5
    assert result["failure_inclusive"]["accuracy"] == 0.5
    assert result["failure_inclusive"]["balanced_accuracy"] == 0.5
    assert 0 <= result["failure_aware_aurc"] <= 1
    counts, proportions = confusion_matrix_2x3(y, [0, ABSTAIN, 1, ABSTAIN])
    assert counts.tolist() == [[1, 0, 1], [0, 1, 1]]
    assert proportions[:, 2].tolist() == pytest.approx([0.5, 0.5])


def test_calibration_metrics_and_temperature_are_auditable():
    y = np.array([0, 0, 0, 1, 1, 1])
    logits = np.array([-6.0, -2.0, 1.0, -1.0, 2.0, 6.0])
    fitted = fit_temperature_scaling(y, logits)
    calibrated = apply_temperature_scaling(logits, fitted["temperature"])
    assert fitted["temperature"] > 0
    assert fitted["nll_after"] <= fitted["nll_before"] + 1e-10
    assert np.all((0 <= calibrated) & (calibrated <= 1))
    metrics = calibration_metrics(y, calibrated, n_bins=3)
    assert {"brier", "nll", "ece", "adaptive_ece", "regression"} <= metrics.keys()
    assert metrics["regression"]["status"].startswith("ok")
    uniform = calibration_curve_table(y, calibrated, n_bins=3, strategy="uniform")
    adaptive = calibration_curve_table(y, calibrated, n_bins=3, strategy="quantile")
    assert uniform.n.sum() == len(y)
    assert adaptive.n.sum() == len(y)


def test_decision_curve_uses_full_cohort_denominator_for_abstentions():
    table = decision_curve_net_benefit(
        [1, 1, 0, 0],
        [0.9, np.nan, 0.8, np.nan],
        thresholds=[0.5],
    )
    row = table.iloc[0]
    assert row.n_total == 4
    assert row.n_evaluable == 2
    assert row.tp == 1 and row.fp == 1
    assert row.net_benefit_model == pytest.approx(0.0)


def test_cluster_bootstrap_is_reproducible_and_reports_valid_draws():
    rows = []
    for patient, label, probability in (
        ("a", 0, 0.1),
        ("b", 0, 0.3),
        ("c", 1, 0.7),
        ("d", 1, 0.9),
    ):
        for eye in ("SAG", "SOL"):
            rows.append(
                {
                    "patient_id": patient,
                    "case_id": f"{patient}_{eye}",
                    "label": label,
                    "probability": probability,
                    "evaluable": True,
                }
            )
    table = pd.DataFrame(rows)
    first = patient_cluster_bootstrap_ci(
        table, draws=40, seed=9, metric_paths=["conditional.auroc", "coverage"]
    )
    second = patient_cluster_bootstrap_ci(
        table, draws=40, seed=9, metric_paths=["conditional.auroc", "coverage"]
    )
    assert first == second
    assert first["conditional.auroc"]["valid_draws"] == 40
    assert first["coverage"]["estimate"] == 1.0


def _ablation_table() -> pd.DataFrame:
    base = pd.DataFrame(
        {
            "unit_id": ["a_R", "a_L", "b_R", "b_L", "c_R", "c_L", "d_R", "d_L"],
            "patient_id": np.repeat(["a", "b", "c", "d"], 2),
            "label": np.repeat([0, 0, 1, 1], 2),
        }
    )
    rows = []
    for arm, probabilities in (
        ("predicted_roi", [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]),
        ("whole_image", [0.2, 0.4, 0.6, 0.3, 0.4, 0.8, 0.7, 0.6]),
    ):
        frame = base.copy()
        frame["arm"] = arm
        frame["probability"] = probabilities
        frame["evaluable"] = True
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def test_ablation_hooks_require_alignment_and_compute_paired_difference():
    table = _ablation_table()
    assert validate_ablation_alignment(table) == ["predicted_roi", "whole_image"]
    evaluated = evaluate_ablation_arms(table, thresholds=0.5)
    assert evaluated["predicted_roi"]["conditional"]["auroc"] == 1.0
    difference = paired_cluster_bootstrap_difference(
        table,
        "predicted_roi",
        "whole_image",
        metric_path="failure_inclusive.balanced_accuracy",
        draws=30,
        seed=5,
    )
    assert difference["valid_draws"] == 30
    expected_p = min(
        1.0,
        2
        * min(
            (1 + difference["nonpositive_draws"]) / 31,
            (1 + difference["nonnegative_draws"]) / 31,
        ),
    )
    assert difference["p_value_two_sided_sign_tail"] == pytest.approx(expected_p)
    assert "paired patient-cluster bootstrap" in difference["p_value_method"]
    broken = table[~((table.arm == "whole_image") & (table.unit_id == "a_R"))]
    with pytest.raises(ValueError, match="identical units"):
        validate_ablation_alignment(broken)


def test_paired_classical_comparisons_and_holm_adjustment():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    strong = np.array([0.05, 0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 0.95])
    weak = np.array([0.1, 0.6, 0.4, 0.8, 0.2, 0.7, 0.55, 0.9])
    delong = delong_auc_comparison(y, strong, weak)
    assert delong["auc_a"] == pytest.approx(1.0)
    assert delong["difference_a_minus_b"] > 0
    assert 0 <= delong["p_value"] <= 1
    mcnemar = mcnemar_exact_comparison(y, strong, weak)
    assert mcnemar["n_common_evaluable"] == 8
    assert 0 <= mcnemar["p_value_exact_two_sided"] <= 1
    adjusted, rejected = holm_adjust([0.01, 0.03, 0.2, np.nan], alpha=0.05)
    assert adjusted[:3].tolist() == pytest.approx([0.03, 0.06, 0.2])
    assert rejected.tolist() == [True, False, False, False]


def test_five_seed_summary_uses_sample_standard_deviation():
    table = pd.DataFrame(
        {
            "model": ["m"] * 5,
            "seed": [1, 2, 3, 4, 5],
            "auroc": [1, 2, 3, 4, 5],
        }
    )
    summary = summarize_across_seeds(
        table, metric_columns=["auroc"], group_columns=["model"]
    ).iloc[0]
    assert summary.auroc_mean == 3
    assert summary.auroc_sd == pytest.approx(np.std([1, 2, 3, 4, 5], ddof=1))
    with pytest.raises(ValueError, match="Expected 5"):
        summarize_across_seeds(
            table.iloc[:4], metric_columns=["auroc"], group_columns=["model"]
        )


def test_reporting_bundle_and_exports_include_strict_primary_outputs(tmp_path):
    bundle = build_evaluation_bundle(
        _frames(),
        eye_threshold=0.5,
        patient_threshold=0.5,
        eye_temperature=1.2,
        patient_temperature=0.9,
        bootstrap_draws=10,
        bootstrap_seed=3,
        decision_thresholds=[0.25, 0.5, 0.75],
        mean_segmentation_columns=["dice"],
    )
    assert len(bundle["eyes"]) == 4
    assert len(bundle["patients"]) == 2
    assert bundle["metrics"]["eye"]["ALL"]["primary_calibrated"]["n_abstain"] == 1
    assert bundle["metrics"]["patient"]["ALL"]["primary_calibrated"]["n_abstain"] == 1
    assert bundle["metrics"]["segmentation"]["eye"]["ALL"]["dice"]["n_valid"] == 4
    assert "dice" in bundle["metrics"]["segmentation"]["patient"]["ALL"]
    assert set(bundle["confusion_2x3"].level) == {"eye", "patient"}

    output = tmp_path / "report"
    export_evaluation(
        _frames(),
        output,
        eye_threshold=0.5,
        patient_threshold=0.5,
        bootstrap_draws=0,
        decision_thresholds=[0.5],
        make_plots=False,
    )
    expected = {
        "frames.csv",
        "eyes.csv",
        "patients.csv",
        "metrics.json",
        "calibration_curves.csv",
        "calibration_reliability_bands.csv",
        "decision_curves.csv",
        "risk_coverage_curves.csv",
        "segmentation_patient_cluster_ci.csv",
        "confusion_2x3.csv",
        "confusion_conditional_2x2.csv",
        "abstention_reasons.csv",
        "classification_by_label_3class.csv",
    }
    assert expected <= {path.name for path in output.iterdir()}
    payload = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert payload["protocol"]["patient_requires_both_eyes"] is True


def test_level_specific_unavailable_locks_abstain_without_identity_defaults():
    eye_unavailable = build_evaluation_bundle(
        _frames(),
        eye_threshold=None,
        patient_threshold=0.5,
        eye_temperature=None,
        patient_temperature=1.0,
        bootstrap_draws=0,
        decision_thresholds=[0.5],
    )
    assert not eye_unavailable["eyes"].evaluable.any()
    originally_roi_valid = eye_unavailable["eyes"].roi_evaluable
    assert int(originally_roi_valid.sum()) == 3
    assert set(
        eye_unavailable["eyes"].loc[originally_roi_valid, "abstention_reason"]
    ) == {"calibration_or_threshold_unavailable"}
    assert set(eye_unavailable["eyes"].abstention_reason) == {
        "calibration_or_threshold_unavailable"
    }
    eye_unavailable_metrics = eye_unavailable["metrics"]["eye"]["ALL"]
    assert eye_unavailable_metrics["raw"]["n_covered"] == 3
    assert eye_unavailable_metrics["raw"]["conditional"]["auroc"] == 1.0
    assert eye_unavailable_metrics["raw"]["conditional"]["accuracy"] is None
    assert (
        eye_unavailable_metrics["raw"]["threshold_metrics_status"]
        == "unavailable_no_locked_raw_equivalent_threshold"
    )
    assert np.isnan(
        eye_unavailable_metrics["primary_calibrated"]["conditional"]["threshold"]
    )
    assert eye_unavailable_metrics["calibration_sample_size"]["roi_evaluable_units"] == 3
    raw_curves = eye_unavailable["calibration_curves"]
    assert "raw" in set(raw_curves.calibration)
    eye_dca = eye_unavailable["decision_curves"].query(
        "level == 'eye' and scope == 'ALL'"
    )
    assert set(eye_dca.probability_scale) == {"unavailable"}
    assert set(eye_dca.decision_policy_status) == {
        "all_abstain_unavailable_lock"
    }
    # Patient calibration is an independent final-unit lock and uses raw eye
    # scores, so the valid p0 patient remains evaluable here.
    assert bool(
        eye_unavailable["patients"].set_index("patient_id").loc["p0", "evaluable"]
    )

    patient_unavailable = build_evaluation_bundle(
        _frames(),
        eye_threshold=0.5,
        patient_threshold=None,
        eye_temperature=1.0,
        patient_temperature=None,
        bootstrap_draws=0,
        decision_thresholds=[0.5],
    )
    assert int(patient_unavailable["eyes"].evaluable.sum()) == 3
    assert not patient_unavailable["patients"].evaluable.any()
    patient_roi_valid = patient_unavailable["patients"].roi_evaluable
    assert int(patient_roi_valid.sum()) == 1
    assert set(
        patient_unavailable["patients"].loc[patient_roi_valid, "abstention_reason"]
    ) == {"calibration_or_threshold_unavailable"}
    assert (
        patient_unavailable["metrics"]["patient"]["ALL"]["decision_threshold_status"]
        == "unavailable_all_units_abstain"
    )


def test_segmentation_coverage_survives_unavailable_classifier():
    frames = _frames()
    frames["segmentation_roi_valid"] = frames["roi_valid"]
    frames["roi_valid"] = False
    frames["probability"] = np.nan
    frames["abstention_reason"] = "classifier_unavailable"

    bundle = build_evaluation_bundle(
        frames,
        eye_threshold=None,
        patient_threshold=None,
        eye_temperature=None,
        patient_temperature=None,
        bootstrap_draws=0,
        decision_thresholds=[0.5],
    )

    eyes = bundle["eyes"].set_index("case_id")
    patients = bundle["patients"].set_index("patient_id")
    assert not eyes["roi_evaluable"].any()
    assert int(eyes["segmentation_roi_evaluable"].sum()) == 3
    assert int(eyes.loc["p0_SAG", "segmentation_n_valid_frames"]) == 4
    assert not patients["roi_evaluable"].any()
    assert bool(patients.loc["p0", "segmentation_roi_evaluable"])
    assert not bool(patients.loc["p1", "segmentation_roi_evaluable"])


def test_config_wrapper_accepts_serialized_none_locks():
    cfg = {
        "dataset": {"frames_per_eye": 7},
        "aggregation": {"minimum_valid_frames": 4},
        "calibration": {"ece_bins": 4},
        "statistics": {"bootstrap_draws": 0, "bootstrap_seed": 1},
        "evaluation": {
            "decision_curve": {
                "threshold_start": 0.25,
                "threshold_stop": 0.75,
                "threshold_step": 0.25,
            }
        },
        "segmentation": {"metrics": [], "surface_dice_tolerance_pixels": 2.0},
    }
    unavailable = {
        "calibration": {"temperature": None},
        "threshold": {"threshold": None},
    }
    available = {
        "calibration": {"temperature": 1.0},
        "threshold": {"threshold": 0.5},
    }
    bundle = build_evaluation_bundle_from_config(
        _frames(), cfg, eye_lock=available, patient_lock=unavailable
    )
    assert int(bundle["eyes"].evaluable.sum()) == 3
    assert not bundle["patients"].evaluable.any()
    assert bundle["metrics"]["locked_analysis_config"]["patient_lock"] == unavailable
    flat_available = {"temperature": 1.0, "threshold": 0.5}
    flat_bundle = build_evaluation_bundle_from_config(
        _frames(), cfg, eye_lock=flat_available, patient_lock=flat_available
    )
    assert int(flat_bundle["eyes"].evaluable.sum()) == 3


def test_raw_equivalent_threshold_matches_temperature_scaled_decisions():
    frames = _frames()
    frames.loc[(frames.patient_id == "p0") & frames.roi_valid, "probability"] = 0.35
    bundle = build_evaluation_bundle(
        frames,
        eye_threshold=0.4,
        patient_threshold=0.4,
        eye_temperature=2.0,
        patient_temperature=2.0,
        bootstrap_draws=0,
        decision_thresholds=[0.4],
    )
    evaluation = bundle["metrics"]["eye"]["ALL"]
    assert (
        evaluation["primary_calibrated"]["confusion_matrix_2x3"]
        == evaluation["raw"]["confusion_matrix_2x3"]
    )
    assert evaluation["operating_thresholds"]["raw_equivalent"] < 0.4


def test_raw_scale_lock_preserves_predictions_when_temperature_is_unavailable():
    raw_lock = {
        "status": "calibration_unavailable",
        "classification_threshold_status": "locked",
        "calibration_status": "unavailable",
        "temperature": None,
        "threshold": 0.5,
        "threshold_probability_scale": "raw_due_to_calibration_unavailable",
    }
    bundle = build_evaluation_bundle(
        _frames(),
        eye_threshold=None,
        patient_threshold=None,
        eye_temperature=None,
        patient_temperature=None,
        eye_lock=raw_lock,
        patient_lock=raw_lock,
        bootstrap_draws=0,
        decision_thresholds=[0.5],
    )
    eyes = bundle["eyes"]
    assert int(eyes.evaluable.sum()) == 3
    assert np.allclose(
        eyes.loc[eyes.evaluable, "probability"],
        eyes.loc[eyes.evaluable, "probability_raw"],
    )
    assert eyes.probability_calibrated.isna().all()
    assert set(eyes.probability_scale) == {"raw"}
    assert (
        eyes.set_index("case_id").loc["p1_SAG", "abstention_reason"]
        == "insufficient_valid_frames"
    )
    evaluation = bundle["metrics"]["eye"]["ALL"]
    assert evaluation["primary_operational"]["n_covered"] == 3
    assert evaluation["raw"]["n_covered"] == 3
    assert evaluation["primary_calibrated"]["status"] == "unavailable"
    assert evaluation["primary_calibrated"]["n_covered"] == 0
    assert "no T=1 substitution" in evaluation["primary_calibrated"]["reason"]


def test_original_three_class_subgroups_are_descriptive_and_single_class_auc_is_undefined():
    table = pd.DataFrame(
        {
            "label_3class": [0, 0, 1, 1, 2, 2],
            "label": [0, 0, 1, 1, 1, 1],
            "evaluable": [True, False, True, True, True, False],
            "prediction": [0, ABSTAIN, 1, 0, 1, ABSTAIN],
            "probability": [0.1, np.nan, 0.8, 0.4, 0.9, np.nan],
            "probability_raw": [0.1, 0.2, 0.8, 0.4, 0.9, 0.7],
            "probability_calibrated": [0.1, np.nan, 0.8, 0.4, 0.9, np.nan],
        }
    )
    result = label_3class_descriptive(table, level="eye", scope="ALL").set_index(
        "label_3class"
    )
    assert result.loc[0, "class_appropriate_metric"] == "specificity"
    assert result.loc[1, "class_appropriate_metric"] == "sensitivity"
    assert result.loc[2, "class_appropriate_metric"] == "sensitivity"
    assert result.loc[0, "failure_aware_class_appropriate_rate"] == 0.5
    assert result.loc[1, "conditional_class_appropriate_rate"] == 0.5
    assert result.loc[2, "coverage"] == 0.5
    assert result.auroc.isna().all()
    assert set(result.auroc_status) == {
        "undefined_single_binary_class_within_label_3class"
    }


def test_holm_can_preserve_preregistered_family_size_with_undefined_tests():
    adjusted, rejected = holm_adjust(
        [0.01, np.nan, 0.04], planned_family_size=5, alpha=0.05
    )
    assert adjusted[0] == pytest.approx(0.05)
    assert adjusted[1] != adjusted[1]
    assert bool(rejected[0]) is True
    assert bool(rejected[1]) is False


def test_mcnemar_is_undefined_without_common_evaluable_units():
    result = mcnemar_exact_comparison(
        [0, 1],
        [np.nan, np.nan],
        [np.nan, np.nan],
        evaluable_a=[False, False],
        evaluable_b=[False, False],
    )
    assert result["status"] == "undefined_no_common_evaluable"
    assert np.isnan(result["p_value_exact_two_sided"])
