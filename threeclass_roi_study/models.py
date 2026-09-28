"""Three-logit ROI classifiers built from the locked study backbones.

The legacy binary package remains immutable.  These subclasses construct the
same feature extractors and then replace only the newly initialized diagnostic
head (and optional geometry residual) with three-logit layers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from predicted_roi_study import models as _binary


NUM_CLASSES = 3
THREE_CLASS_FAMILIES = ("yolo26", "vit_method2", "emcad", "sam2_unet")
RESNET18_WEIGHTS_URL = _binary.RESNET18_WEIGHTS_URL
RESNET18_WEIGHTS_SHA256 = _binary.RESNET18_WEIGHTS_SHA256

MODEL_SPECIFIC_CLASSIFIER_ARCHITECTURES: dict[str, str] = {
    family: _binary.MODEL_SPECIFIC_CLASSIFIER_ARCHITECTURES[family].replace(
        "binary head", "three-class head"
    )
    for family in THREE_CLASS_FAMILIES
}
MODEL_SPECIFIC_CLASSIFIER_ARCHITECTURE_IDS: dict[str, str] = {
    family: f"{_binary.MODEL_SPECIFIC_CLASSIFIER_ARCHITECTURE_IDS[family]}_3class"
    for family in THREE_CLASS_FAMILIES
}
MODEL_SPECIFIC_DEFAULT_DROPOUT: dict[str, float] = {
    family: _binary.MODEL_SPECIFIC_DEFAULT_DROPOUT[family]
    for family in THREE_CLASS_FAMILIES
}


def _replace_last_linear(module: nn.Module, *, out_features: int) -> nn.Linear:
    candidates = [
        (name, child)
        for name, child in module.named_modules()
        if name and isinstance(child, nn.Linear)
    ]
    if not candidates:
        raise RuntimeError("Classifier head has no terminal linear layer")
    name, previous = candidates[-1]
    parent_name, _, child_name = name.rpartition(".")
    parent = module.get_submodule(parent_name) if parent_name else module
    replacement = nn.Linear(
        previous.in_features,
        int(out_features),
        bias=previous.bias is not None,
        device=previous.weight.device,
        dtype=previous.weight.dtype,
    )
    if isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(child_name)] = replacement
    else:
        setattr(parent, child_name, replacement)
    return replacement


class _ThreeClassFamilyMixin:
    """Upgrade only a legacy family's random diagnostic head to three logits."""

    num_classes = NUM_CLASSES

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        terminal = _replace_last_linear(self.classification_head, out_features=NUM_CLASSES)
        if int(terminal.out_features) != NUM_CLASSES:
            raise RuntimeError("Failed to construct a three-logit diagnostic head")
        self.geometry_logit_residual = (
            nn.Linear(self.geometry_features, NUM_CLASSES, bias=False)
            if self.geometry_features
            else None
        )
        self.diagnostic_head_initialization = "new_random_three_class_head"

    def _finish_logits(self, logits: Tensor, geometry: Tensor | None) -> Tensor:
        expected = (len(logits), NUM_CLASSES)
        if tuple(logits.shape) != expected or not bool(torch.isfinite(logits).all()):
            raise RuntimeError(
                f"Three-class model-specific ROI classifier must return finite "
                f"Bx{NUM_CLASSES} logits"
            )
        if self.geometry_features:
            if geometry is None or tuple(geometry.shape) != (
                len(logits),
                self.geometry_features,
            ):
                raise ValueError(f"geometry must have shape Bx{self.geometry_features}")
            if not bool(torch.isfinite(geometry).all()):
                raise ValueError("geometry must be finite")
            assert self.geometry_logit_residual is not None
            logits = logits + self.geometry_logit_residual(
                geometry.to(device=logits.device, dtype=logits.dtype)
            )
        elif geometry is not None:
            raise ValueError("Primary appearance-only classifier must not receive geometry")
        return logits

    def parameter_info(self) -> dict[str, Any]:
        value = super().parameter_info()
        value.update(
            {
                "num_classes": NUM_CLASSES,
                "diagnostic_target": "control_vs_papilledema_vs_pseudopapilledema",
                "diagnostic_head_initialization": self.diagnostic_head_initialization,
                "architecture_id": MODEL_SPECIFIC_CLASSIFIER_ARCHITECTURE_IDS[
                    self.model_family
                ],
                "architecture": MODEL_SPECIFIC_CLASSIFIER_ARCHITECTURES[
                    self.model_family
                ],
            }
        )
        # Correct legacy prose without changing the old implementation.
        if "optimization_policy" in value:
            value["optimization_policy"] = str(value["optimization_policy"]).replace(
                "binary_head", "three_class_head"
            )
        return value


class ThreeClassYOLO26StrictROIClassifier(
    _ThreeClassFamilyMixin, _binary.YOLO26StrictROIClassifier
):
    pass


class ThreeClassViTMethod2StrictROIClassifier(
    _ThreeClassFamilyMixin, _binary.ViTMethod2StrictROIClassifier
):
    pass


class ThreeClassEMCADStrictROIClassifier(
    _ThreeClassFamilyMixin, _binary.EMCADStrictROIClassifier
):
    pass


class ThreeClassSAM2UNetStrictROIClassifier(
    _ThreeClassFamilyMixin, _binary.SAM2UNetStrictROIClassifier
):
    pass


MODEL_SPECIFIC_CLASSIFIER_REGISTRY: dict[
    str, type[_binary.StrictROIFamilyClassifier]
] = {
    "yolo26": ThreeClassYOLO26StrictROIClassifier,
    "vit_method2": ThreeClassViTMethod2StrictROIClassifier,
    "emcad": ThreeClassEMCADStrictROIClassifier,
    "sam2_unet": ThreeClassSAM2UNetStrictROIClassifier,
}


def build_threeclass_roi_classifier(
    family: str,
    *,
    pretrained: bool = False,
    weights_path: str | Path | None = None,
    expected_sha256: str | None = None,
    dropout: float | None = None,
    geometry_features: int = 0,
    lightweight: bool = False,
    num_classes: int = NUM_CLASSES,
) -> _binary.StrictROIFamilyClassifier:
    """Build one of the four study families with a fresh three-logit head."""

    if int(num_classes) != NUM_CLASSES:
        raise ValueError("The three-class study requires exactly three logits")
    if family not in MODEL_SPECIFIC_CLASSIFIER_REGISTRY:
        raise KeyError(
            f"Unknown three-class classifier family {family!r}; expected one of "
            f"{list(THREE_CLASS_FAMILIES)}"
        )
    selected_dropout = (
        MODEL_SPECIFIC_DEFAULT_DROPOUT[family]
        if dropout is None
        else float(dropout)
    )
    if not 0.0 <= selected_dropout < 1.0:
        raise ValueError("dropout must be in [0,1)")
    return MODEL_SPECIFIC_CLASSIFIER_REGISTRY[family](
        pretrained=bool(pretrained),
        weights_path=weights_path,
        expected_sha256=expected_sha256,
        dropout=selected_dropout,
        geometry_features=int(geometry_features),
        lightweight=bool(lightweight),
    )


# Names matching the existing engine's imports, scoped to this new package.
build_strict_roi_classifier = build_threeclass_roi_classifier
build_model_specific_roi_classifier = build_threeclass_roi_classifier


class ThreeClassResNet18ROIClassifier(_binary.ResNet18ROIClassifier):
    """Standardized ResNet-18 backbone with a fresh three-logit head."""

    classifier_strategy = "standardized_resnet18"
    num_classes = NUM_CLASSES

    def __init__(
        self,
        *,
        pretrained: bool = True,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.2,
        geometry_features: int = 0,
        weights_path: str | Path | None = None,
        expected_sha256: str | None = None,
    ) -> None:
        if int(num_classes) != NUM_CLASSES:
            raise ValueError("The three-class study requires exactly three logits")
        # The legacy constructor validates and loads only the ResNet backbone;
        # its random two-logit layer is immediately discarded below.
        super().__init__(
            pretrained=pretrained,
            num_classes=2,
            dropout=dropout,
            geometry_features=geometry_features,
            weights_path=weights_path,
            expected_sha256=expected_sha256,
        )
        _replace_last_linear(self.head, out_features=NUM_CLASSES)
        self.diagnostic_head_initialization = "new_random_three_class_head"

    def parameter_info(self) -> dict[str, Any]:
        registered = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        return {
            "strategy": self.classifier_strategy,
            "architecture_id": "torchvision_resnet18_roi_classifier_3class",
            "num_classes": NUM_CLASSES,
            "registered_parameters": int(registered),
            "trainable_updateable_parameters": int(trainable),
            "frozen_parameters": int(registered - trainable),
            "geometry_features": int(self.geometry_features),
            "diagnostic_head_initialization": self.diagnostic_head_initialization,
            "pretrained_provenance": dict(self.pretrained_provenance),
        }


ResNet18ROIClassifier = ThreeClassResNet18ROIClassifier


def build_roi_classifier(
    backbone: str = "resnet18",
    *,
    pretrained: bool = True,
    num_classes: int = NUM_CLASSES,
    dropout: float = 0.2,
    geometry_features: int = 0,
    weights_path: str | Path | None = None,
    expected_sha256: str | None = None,
) -> ThreeClassResNet18ROIClassifier:
    if backbone != "resnet18":
        raise KeyError("The standardized three-class ROI classifier is 'resnet18'")
    return ThreeClassResNet18ROIClassifier(
        pretrained=pretrained,
        num_classes=num_classes,
        dropout=dropout,
        geometry_features=geometry_features,
        weights_path=weights_path,
        expected_sha256=expected_sha256,
    )


build_standardized_roi_classifier = build_roi_classifier


__all__ = [
    "MODEL_SPECIFIC_CLASSIFIER_ARCHITECTURE_IDS",
    "MODEL_SPECIFIC_CLASSIFIER_ARCHITECTURES",
    "MODEL_SPECIFIC_CLASSIFIER_REGISTRY",
    "MODEL_SPECIFIC_DEFAULT_DROPOUT",
    "NUM_CLASSES",
    "RESNET18_WEIGHTS_SHA256",
    "RESNET18_WEIGHTS_URL",
    "ResNet18ROIClassifier",
    "THREE_CLASS_FAMILIES",
    "ThreeClassEMCADStrictROIClassifier",
    "ThreeClassResNet18ROIClassifier",
    "ThreeClassSAM2UNetStrictROIClassifier",
    "ThreeClassViTMethod2StrictROIClassifier",
    "ThreeClassYOLO26StrictROIClassifier",
    "build_model_specific_roi_classifier",
    "build_roi_classifier",
    "build_standardized_roi_classifier",
    "build_strict_roi_classifier",
    "build_threeclass_roi_classifier",
]
