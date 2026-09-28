from __future__ import annotations

from pathlib import Path

import pytest
import torch

from predicted_roi_study.config import EXPECTED_MODELS, load_config
from predicted_roi_study.models import (
    MODEL_SPECIFIC_CLASSIFIER_ARCHITECTURE_IDS,
    MODEL_SPECIFIC_CLASSIFIER_ARCHITECTURES,
    MODEL_SPECIFIC_CLASSIFIER_REGISTRY,
    PredictedROIPipeline,
    build_model_specific_roi_classifier,
    build_strict_roi_classifier,
)
from predicted_roi_study.roi import ROIPolicy


FAMILIES = (
    "yolo26",
    "vit_method2",
    "emcad",
    "sam2_unet",
)
NEUTRAL = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)


def _strict_batch(batch: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(7251)
    image = torch.rand((batch, 3, 224, 224), generator=generator)
    mask = torch.zeros((batch, 1, 224, 224), dtype=torch.bool)
    mask[:, :, 27:187, 43:181] = True
    return torch.where(mask, image, NEUTRAL), mask


def test_registry_has_exact_four_primary_families_and_alias() -> None:
    assert tuple(MODEL_SPECIFIC_CLASSIFIER_REGISTRY) == FAMILIES
    assert tuple(MODEL_SPECIFIC_CLASSIFIER_ARCHITECTURE_IDS) == FAMILIES
    assert tuple(MODEL_SPECIFIC_CLASSIFIER_ARCHITECTURES) == FAMILIES
    assert load_config()["classifier"]["primary"]["architecture_by_segmenter"] == {
        model: MODEL_SPECIFIC_CLASSIFIER_ARCHITECTURE_IDS[model]
        for model in EXPECTED_MODELS
    }
    first = build_strict_roi_classifier("emcad", lightweight=True)
    second = build_model_specific_roi_classifier("emcad", lightweight=True)
    assert type(first) is type(second)
    assert first.classifier_strategy == "model_specific"


@pytest.mark.parametrize("family", FAMILIES)
def test_lightweight_family_shape_contract_and_parameter_provenance(family: str) -> None:
    image, mask = _strict_batch()
    model = build_strict_roi_classifier(family, lightweight=True).eval()
    logits = model(image, hard_mask=mask)
    info = model.parameter_info()

    assert logits.shape == (2, 2)
    assert torch.isfinite(logits).all()
    assert model.input_size == (224, 224)
    assert model.model_family == family
    assert info["family"] == family
    assert info["architecture_id"] == MODEL_SPECIFIC_CLASSIFIER_ARCHITECTURE_IDS[family]
    assert info["architecture"] == MODEL_SPECIFIC_CLASSIFIER_ARCHITECTURES[family]
    assert info["registered_parameters"] > 0
    assert 0 < info["trainable_updateable_parameters"] <= info["registered_parameters"]
    assert info["frozen_parameters"] == (
        info["registered_parameters"] - info["trainable_updateable_parameters"]
    )
    assert info["full_image_pixels_consumed"] is False
    assert info["strict_hard_mask_reapplied"] is True
    assert info["pretrained_provenance"]["network_download_during_run"] is False


@pytest.mark.parametrize("family", FAMILIES)
def test_outside_roi_randomization_is_exactly_invariant(family: str) -> None:
    image, mask = _strict_batch()
    generator = torch.Generator().manual_seed(88291)
    mutated = torch.where(mask, image, torch.rand(image.shape, generator=generator))
    model = build_strict_roi_classifier(family, lightweight=True).eval()

    with torch.inference_mode():
        reference = model(image, hard_mask=mask)
        changed = model(mutated, hard_mask=mask)

    torch.testing.assert_close(reference, changed, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("family", FAMILIES)
def test_gradients_reach_encoder_and_head_but_not_outside_hard_roi(family: str) -> None:
    image, mask = _strict_batch()
    image = image.clone().requires_grad_(True)
    model = build_strict_roi_classifier(family, lightweight=True).eval()
    logits = model(image, hard_mask=mask)
    logits.square().sum().backward()

    assert image.grad is not None and torch.isfinite(image.grad).all()
    assert torch.count_nonzero(image.grad.masked_select(~mask.expand_as(image))) == 0
    assert torch.count_nonzero(image.grad.masked_select(mask.expand_as(image))) > 0
    encoder_gradients = [
        parameter.grad for parameter in model.encoder.parameters() if parameter.requires_grad
    ]
    head_gradients = [
        parameter.grad for parameter in model.classification_head.parameters()
        if parameter.requires_grad
    ]
    assert any(gradient is not None and torch.count_nonzero(gradient) for gradient in encoder_gradients)
    assert any(gradient is not None and torch.count_nonzero(gradient) for gradient in head_gradients)


@pytest.mark.parametrize("family", FAMILIES)
def test_empty_or_malformed_mask_is_rejected(family: str) -> None:
    image, mask = _strict_batch(batch=1)
    model = build_strict_roi_classifier(family, lightweight=True)
    with pytest.raises(ValueError, match="Empty ROI"):
        model(image, hard_mask=torch.zeros_like(mask))
    with pytest.raises(ValueError, match="hard_mask must have shape"):
        model(image, hard_mask=mask[:, :, :-1])
    with pytest.raises(ValueError, match="requires Bx3"):
        model(image[:, :, :-1], hard_mask=mask[:, :, :-1])


@pytest.mark.parametrize("family", FAMILIES)
def test_optional_geometry_has_locked_four_feature_contract(family: str) -> None:
    image, mask = _strict_batch()
    model = build_strict_roi_classifier(
        family, lightweight=True, geometry_features=4
    ).eval()
    geometry = torch.tensor([[0.2, 0.5, 0.4, 0.1], [0.3, 0.4, 0.6, -0.2]])
    logits = model(image, geometry, hard_mask=mask)
    assert logits.shape == (2, 2)
    with pytest.raises(ValueError, match="geometry must have shape"):
        model(image, geometry[:, :3], hard_mask=mask)




def test_pretrained_hash_mismatch_is_rejected_before_deserialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad = tmp_path / "yolo.pth"
    bad.write_bytes(b"this is not a torch checkpoint")
    called = False
    original = torch.load

    def tracked_load(*args, **kwargs):
        nonlocal called
        called = True
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "load", tracked_load)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        build_strict_roi_classifier(
            "yolo26",
            pretrained=True,
            weights_path=bad,
            expected_sha256="0" * 64,
        )
    assert called is False




def test_pretrained_false_never_requires_or_reads_a_weight_file() -> None:
    for family in FAMILIES:
        model = build_strict_roi_classifier(family, pretrained=False, lightweight=True)
        assert model.pretrained_provenance["pretrained"] is False
        assert model.pretrained_provenance["path"] is None


def test_sam2_production_topology_freezes_native_hiera_but_not_adapters_or_head() -> None:
    model = build_strict_roi_classifier("sam2_unet", pretrained=False)
    native = [
        parameter
        for name, parameter in model.feature_extractor.named_parameters()
        if ".block." in name
    ]
    adapters = [
        parameter
        for name, parameter in model.feature_extractor.named_parameters()
        if ".prompt_learn." in name
    ]
    head = list(model.classification_head.parameters())
    info = model.parameter_info()

    assert native and all(not parameter.requires_grad for parameter in native)
    assert adapters and all(parameter.requires_grad for parameter in adapters)
    assert head and all(parameter.requires_grad for parameter in head)
    assert info["original_hiera_trunk_frozen"] is True
    assert info["original_hiera_trunk_parameters"] > 0
    assert info["trainable_adapter_parameters"] == sum(p.numel() for p in adapters)
    assert info["classification_head_trainable_parameters"] == sum(p.numel() for p in head)
    assert model.pretrained_provenance["original_hiera_trunk_frozen"] is True
    assert model.pretrained_provenance["trainable_hiera_adapters"] is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Full Hiera gradient audit requires CUDA")
def test_sam2_production_gradient_reaches_only_adapters_and_head() -> None:
    image, mask = _strict_batch(batch=1)
    image = image.cuda().requires_grad_(True)
    mask = mask.cuda()
    model = build_strict_roi_classifier("sam2_unet", pretrained=False).cuda().train()
    logits = model(image, hard_mask=mask)
    logits.square().sum().backward()

    native = [
        parameter
        for name, parameter in model.feature_extractor.named_parameters()
        if ".block." in name
    ]
    adapters = [
        parameter
        for name, parameter in model.feature_extractor.named_parameters()
        if ".prompt_learn." in name
    ]
    head = list(model.classification_head.parameters())
    assert native and all(parameter.requires_grad is False and parameter.grad is None for parameter in native)
    assert adapters and any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) for parameter in adapters
    )
    assert head and any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) for parameter in head
    )
    assert image.grad is not None
    assert torch.count_nonzero(image.grad.masked_select(~mask.expand_as(image))) == 0


class _ConstantAuditSegmenter(torch.nn.Module):
    image_size = 32

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = images[:, :1] * 0.0 + self.anchor * 0.0 - 20.0
        logits[0, :, 8:24, 10:22] = 20.0
        return {"seg_logits": logits}


def _audit_pipeline() -> PredictedROIPipeline:
    return PredictedROIPipeline(
        _ConstantAuditSegmenter(),
        build_strict_roi_classifier("emcad", lightweight=True).eval(),
        ROIPolicy(
            min_area_pixels=10,
            max_area_pixels=300,
            image_shape=(32, 32),
            threshold=0.5,
            dominance_ratio=1.5,
            reject_border=True,
        ),
    ).eval()


def test_model_specific_deployable_pipeline_passes_extracted_hard_mask() -> None:
    pipeline = _audit_pipeline()
    source = torch.rand((2, 3, 32, 32), generator=torch.Generator().manual_seed(989))
    result = pipeline(source)

    assert result["valid_mask"].tolist() == [True, False]
    assert torch.isfinite(result["cls_logits"][0]).all()
    assert torch.isnan(result["cls_logits"][1]).all()
    assert result["roi_images"].shape == (1, 3, 224, 224)
    assert result["roi_masks"].shape == (1, 224, 224)
    assert result["roi_masks"].dtype == torch.bool
    assert result["roi_masks"].any()


def test_deployable_pipeline_is_invariant_to_original_pixels_outside_fixed_prediction() -> None:
    pipeline = _audit_pipeline()
    source = torch.rand((1, 3, 32, 32), generator=torch.Generator().manual_seed(1012))
    predicted = torch.zeros((1, 1, 32, 32), dtype=torch.bool)
    predicted[:, :, 8:24, 10:22] = True
    mutated = torch.where(
        predicted,
        source,
        torch.rand(source.shape, generator=torch.Generator().manual_seed(7741)),
    )

    with torch.inference_mode():
        first = pipeline(source)
        second = pipeline(mutated)

    assert first["valid_mask"].item() and second["valid_mask"].item()
    torch.testing.assert_close(first["roi_images"], second["roi_images"], rtol=0.0, atol=0.0)
    assert torch.equal(first["roi_masks"], second["roi_masks"])
    torch.testing.assert_close(first["cls_logits"], second["cls_logits"], rtol=0.0, atol=0.0)
