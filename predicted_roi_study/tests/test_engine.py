from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from predicted_roi_study.config import load_config
from predicted_roi_study.data import classifier_loss_weights
from predicted_roi_study.engine import (
    _aligned_unit_pair,
    _classifier_validation_monitor,
    _classifier_forward,
    _lock_calibration_and_threshold,
    _paired_locked_primary_effect,
    _paired_patient_continuous_effect,
    _verify_segmenter_initialization_provenance,
)
from predicted_roi_study.protocol import ProtocolGateError


def _eye_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": ["p0", "p1", "p2", "p3"],
            "case_id": ["e0", "e1", "e2", "e3"],
            "side": ["SAG", "SOL", "SAG", "SOL"],
            "label": [0, 0, 1, 1],
            "evaluable": [True, True, True, True],
            "probability": [0.1, 0.2, 0.8, 0.9],
        }
    )


def _frame_table(*, valid: bool = True) -> pd.DataFrame:
    rows = []
    for eye, patient, side, label, probability in (
        ("e0", "p0", "SAG", 0, 0.1),
        ("e1", "p0", "SOL", 0, 0.2),
        ("e2", "p1", "SAG", 1, 0.8),
        ("e3", "p1", "SOL", 1, 0.9),
    ):
        for frame in range(7):
            rows.append(
                {
                    "patient_id": patient,
                    "case_id": eye,
                    "side": side,
                    "frame_id": str(frame),
                    "label": label,
                    "roi_valid": valid,
                    "abstention_reason": "" if valid else "empty",
                    "probability": probability if valid else np.nan,
                }
            )
    return pd.DataFrame(rows)


def test_validation_temperature_and_threshold_lock_is_defined_for_two_classes():
    cfg = load_config()
    locked = _lock_calibration_and_threshold(cfg, _eye_table(), level="eye")
    assert locked["classification_threshold_status"] == "locked"
    assert locked["calibration_status"] == "locked"
    assert 0.05 <= locked["temperature"] <= 20.0
    assert locked["threshold"] in cfg["classifier"]["threshold_candidates"]


def test_validation_lock_never_invents_defaults_for_one_class():
    cfg = load_config()
    table = _eye_table().query("label == 0").copy()
    locked = _lock_calibration_and_threshold(cfg, table, level="eye")
    assert locked["classification_threshold_status"] == "unavailable"
    assert locked["calibration_status"] == "unavailable"
    assert locked["temperature"] is None
    assert locked["threshold"] is None


def test_classifier_monitor_uses_eye_auroc_and_handles_zero_coverage():
    cfg = load_config()
    usable = _classifier_validation_monitor(cfg, _frame_table(valid=True))
    assert usable["monitor_status"] == "auroc"
    assert usable["eye_auroc"] == 1.0
    unusable = _classifier_validation_monitor(cfg, _frame_table(valid=False))
    assert unusable["monitor_status"] == "non_evaluable"
    assert unusable["evaluable_eyes"] == 0


def test_classifier_boundary_requires_mask_and_remasks_standardized_outside_pixels():
    class MeanClassifier(torch.nn.Module):
        def forward(self, images, geometry=None):
            score = images.mean(dim=(1, 2, 3))
            return torch.stack((-score, score), dim=1)

    classifier = MeanClassifier()
    mask = torch.zeros((1, 8, 8), dtype=torch.bool)
    mask[:, 2:6, 2:6] = True
    first = torch.rand((1, 3, 8, 8))
    second = torch.where(mask.unsqueeze(1), first, torch.rand_like(first))
    first_logits = _classifier_forward(
        classifier,
        {"image": first, "roi_mask": mask},
        classifier_strategy="standardized_resnet18",
    )
    second_logits = _classifier_forward(
        classifier,
        {"image": second, "roi_mask": mask},
        classifier_strategy="standardized_resnet18",
    )
    assert torch.equal(first_logits, second_logits)
    with pytest.raises(ProtocolGateError, match="explicit cached hard ROI mask"):
        _classifier_forward(
            classifier,
            {"image": first},
            classifier_strategy="standardized_resnet18",
        )


def test_classifier_weights_equalize_eyes_and_binary_classes():
    rows = []
    specifications = (("e0", "p0", 0, 4), ("e1", "p0", 0, 7), ("e2", "p1", 1, 5), ("e3", "p1", 1, 6))
    for eye, patient, label, count in specifications:
        for frame in range(count):
            rows.append(
                {
                    "patient_id": patient,
                    "case_id": eye,
                    "side": "SAG" if eye in {"e0", "e2"} else "SOL",
                    "label": label,
                    "roi_valid": True,
                    "frame_id": str(frame),
                }
            )
    table = pd.DataFrame(rows)
    table["weight"] = classifier_loss_weights(table)
    eye_totals = table.groupby("case_id").weight.sum().to_numpy()
    class_totals = table.groupby("label").weight.sum().to_numpy()
    assert np.allclose(eye_totals, eye_totals[0])
    assert np.allclose(class_totals, class_totals[0])
    assert np.isclose(table.weight.mean(), 1.0)


def _locked_eye_predictions() -> pd.DataFrame:
    rows = []
    for patient_index, label in enumerate((0, 0, 1, 1)):
        for side in ("SAG", "SOL"):
            rows.append(
                {
                    "patient_id": f"p{patient_index}",
                    "case_id": f"p{patient_index}_{side}",
                    "side": side,
                    "label": label,
                    "label_3class": label,
                    "prediction": label,
                }
            )
    return pd.DataFrame(rows)


def test_pair_alignment_is_order_independent_and_identity_strict():
    left = _locked_eye_predictions()
    right = left.sample(frac=1, random_state=3)
    aligned_left, aligned_right = _aligned_unit_pair(left, right, level="eye")
    assert aligned_left.case_id.tolist() == aligned_right.case_id.tolist()
    broken = right.copy()
    broken.loc[broken.index[0], "side"] = "WRONG"
    with pytest.raises(ProtocolGateError):
        _aligned_unit_pair(left, broken, level="eye")


def test_primary_paired_effect_uses_locked_predictions_and_patient_clusters():
    left = _locked_eye_predictions()
    identical = _paired_locked_primary_effect(left, left.copy(), draws=199, seed=7)
    assert identical["estimate"] == 0.0
    assert identical["low"] == 0.0
    assert identical["high"] == 0.0
    worse = left.copy()
    worse.loc[worse.patient_id == "p2", "prediction"] = 0
    effect = _paired_locked_primary_effect(left, worse, draws=199, seed=7)
    assert effect["estimate"] == pytest.approx(0.25)
    assert 0 < effect["p_value_randomization_two_sided"] <= 1


def test_paired_continuous_effect_returns_exact_constant_difference():
    table = pd.DataFrame(
        {
            "patient_id": ["p0", "p1", "p2", "p3"],
            "label": [0, 0, 1, 1],
            "difference": [0.2, 0.2, 0.2, 0.2],
        }
    )
    result = _paired_patient_continuous_effect(table, draws=199, seed=11)
    assert result["estimate"] == pytest.approx(0.2)
    assert result["low"] == pytest.approx(0.2)
    assert result["high"] == pytest.approx(0.2)


def test_random_initialization_provenance_rejects_unexpected_loaded_weights():
    cfg = {
        "segmentation": {
            "initialization": {
                "toy": {
                    "mode": "random_segmentation_weights",
                    "checkpoint": None,
                    "sha256": None,
                }
            }
        }
    }
    result = _verify_segmenter_initialization_provenance(cfg, "toy", {})
    assert result["status"] == "verified_random_initialization"
    with pytest.raises(ProtocolGateError):
        _verify_segmenter_initialization_provenance(
            cfg, "toy", {"pretrained_provenance": {"pretrained": True}}
        )
