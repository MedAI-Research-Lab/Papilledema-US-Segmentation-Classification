from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from predicted_roi_study.models import PredictedROIPipeline, build_roi_classifier
from predicted_roi_study.roi import (
    ROIStatus,
    ROIPolicy,
    extract_roi_tensor,
    postprocess_probability,
)


def policy(shape=(12, 12), **overrides):
    values = {
        "min_area_pixels": 4,
        "max_area_pixels": 20,
        "image_shape": shape,
        "threshold": 0.5,
        "dominance_ratio": 1.5,
    }
    values.update(overrides)
    return ROIPolicy(**values)


def probability(shape=(12, 12), boxes=()):
    result = np.zeros(shape, dtype=np.float32)
    for y0, y1, x0, x1, score in boxes:
        result[y0:y1, x0:x1] = score
    return result


def test_fit_from_gt_uses_largest_8_connected_component_and_declared_formula():
    masks = np.zeros((2, 10, 10), dtype=np.uint8)
    masks[0, 2:4, 2:4] = 1  # area 4
    masks[0, 8, 8] = 1  # disconnected annotation speck must not count
    masks[1, 2:4, 2:5] = 1  # area 6
    fitted = ROIPolicy.fit_from_gt(masks, quantile_low=0.0, quantile_high=1.0)
    assert fitted.min_area_pixels == 2.0
    assert fitted.max_area_pixels == 9.0
    assert fitted.fitted_q_low_pixels == 4.0
    assert fitted.fitted_q_high_pixels == 6.0
    assert fitted.fitted_sample_count == 2
    assert ROIPolicy.from_dict(fitted.to_dict()) == fitted


def test_diagonal_pixels_are_one_8_connected_component():
    score = np.zeros((6, 6), dtype=np.float32)
    score[2, 2] = score[3, 3] = 0.9
    result = postprocess_probability(
        score,
        policy((6, 6), min_area_pixels=2, max_area_pixels=10, reject_border=False),
    )
    assert result.status is ROIStatus.VALID
    assert len(result.components) == 1
    assert result.area_pixels == 2


@pytest.mark.parametrize(
    ("score", "configured", "expected"),
    [
        (probability(), {}, ROIStatus.EMPTY),
        (probability(boxes=[(4, 5, 4, 5, 0.9)]), {}, ROIStatus.TINY),
        (probability(boxes=[(3, 8, 3, 8, 0.9)]), {}, ROIStatus.OVERSIZE),
        (probability(boxes=[(0, 2, 4, 6, 0.9)]), {}, ROIStatus.BORDER),
        (
            probability(boxes=[(2, 4, 2, 4, 0.9), (7, 9, 7, 9, 0.9)]),
            {},
            ROIStatus.MULTI_AMBIGUOUS,
        ),
    ],
)
def test_abstention_statuses_are_explicit(score, configured, expected):
    result = postprocess_probability(score, policy(**configured))
    assert result.status is expected
    assert not result.valid
    assert not result.mask.any()


def test_dominant_component_is_selected_and_tiny_specks_are_removed():
    score = probability(
        boxes=[
            (2, 4, 2, 5, 0.99),  # dominant mean probability
            (7, 9, 7, 9, 0.60),
            (5, 6, 10, 11, 0.99),  # tiny noise
        ]
    )
    result = postprocess_probability(score, policy())
    assert result.status is ROIStatus.VALID
    assert result.area_pixels == 6
    assert result.bbox_xyxy == (2, 2, 5, 4)
    assert result.mask.sum() == 6
    assert result.hard_mask.sum() == 11
    assert result.dominance_ratio_observed > 1.6


def test_oversize_component_invalidates_frame_even_with_plausible_smaller_component():
    score = probability(boxes=[(2, 8, 2, 8, 0.8), (9, 11, 9, 11, 0.99)])
    result = postprocess_probability(score, policy(max_area_pixels=30))
    assert result.status is ROIStatus.OVERSIZE
    assert not result.mask.any()


def test_only_small_enclosed_holes_are_filled():
    score = probability(boxes=[(2, 8, 2, 8, 0.9)])
    score[3, 3] = 0.0
    score[5:7, 5:7] = 0.0
    result = postprocess_probability(
        score,
        policy(max_area_pixels=40, max_hole_area_pixels=1),
    )
    assert result.valid
    assert result.hard_mask[3, 3]
    assert not result.hard_mask[5:7, 5:7].any()


def test_outside_roi_changes_cannot_change_classifier_tensor():
    torch.manual_seed(7)
    mask = np.zeros((12, 16), dtype=np.uint8)
    mask[3:9, 5:11] = 1
    first = torch.rand(3, 12, 16)
    second = torch.rand(3, 12, 16)
    # The two images agree only inside the selected predicted ROI.
    second[:, torch.as_tensor(mask, dtype=torch.bool)] = first[:, torch.as_tensor(mask, dtype=torch.bool)]
    first_roi = extract_roi_tensor(first, mask, target_size=(24, 28), neutral=(0.2, 0.3, 0.4))
    second_roi = extract_roi_tensor(second, mask, target_size=(24, 28), neutral=(0.2, 0.3, 0.4))
    assert torch.equal(first_roi, second_roi)


def test_roi_extraction_is_centered_neutral_letterbox_and_returns_geometry():
    image = torch.ones(3, 8, 12)
    mask = np.zeros((8, 12), dtype=np.uint8)
    mask[2:6, 4:6] = 1
    extraction = extract_roi_tensor(
        image, mask, target_size=12, neutral=(0.1, 0.2, 0.3), return_metadata=True
    )
    assert extraction.tensor.shape == (3, 12, 12)
    assert extraction.output_mask.shape == (12, 12)
    assert extraction.bbox_xyxy == (4, 2, 6, 6)
    assert extraction.resized_hw == (12, 6)
    assert extraction.placement_xyxy == (3, 0, 9, 12)
    assert torch.allclose(extraction.tensor[:, 0, 0], torch.tensor([0.1, 0.2, 0.3]))
    assert extraction.geometry.shape == (4,)


def test_extraction_rejects_multiple_components():
    image = torch.zeros(3, 8, 8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    mask[5:7, 5:7] = 1
    with pytest.raises(ValueError, match="exactly one"):
        extract_roi_tensor(image, mask)


def test_classifier_rejects_bad_checksum_before_deserialization(tmp_path, monkeypatch):
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"not a torch checkpoint")
    deserialized = False

    def forbidden_load(*args, **kwargs):
        nonlocal deserialized
        deserialized = True
        raise AssertionError("Unverified bytes must not be deserialized")

    monkeypatch.setattr(torch, "load", forbidden_load)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        build_roi_classifier(
            pretrained=True,
            weights_path=weights,
            expected_sha256="0" * 64,
        )
    assert not deserialized


def test_pipeline_never_calls_classifier_or_falls_back_when_roi_is_invalid():
    class EmptySegmenter(nn.Module):
        image_size = 12

        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.tensor(0.0))

        def forward(self, images):
            return {"seg_logits": images.new_full((len(images), 1, 12, 12), -20.0)}

    class SpyClassifier(nn.Module):
        input_size = (16, 16)
        neutral_rgb = (0.485, 0.456, 0.406)
        geometry_features = 0

        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.tensor(0.0))
            self.called = False

        def forward(self, images, geometry=None):
            self.called = True
            return images.new_zeros((len(images), 2)) + self.anchor

    classifier = SpyClassifier()
    pipeline = PredictedROIPipeline(EmptySegmenter(), classifier, policy())
    outputs = pipeline(torch.rand(2, 3, 12, 12))
    assert not classifier.called
    assert not outputs["valid_mask"].any()
    assert torch.isnan(outputs["cls_logits"]).all()
    assert [result.status for result in outputs["roi_results"]] == [ROIStatus.EMPTY, ROIStatus.EMPTY]
