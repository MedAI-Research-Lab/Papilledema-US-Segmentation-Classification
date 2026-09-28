"""Model adapters for the two-stage strict predicted-ROI experiment.

Stage 1 exposes only segmentation outputs from each of the four executed model
families.  Stage 2 has the model-specific strict-ROI classifiers used by the
primary estimand and one common ResNet-18 used by the standardized secondary
estimand.  Both consume inputs built exclusively by
:mod:`predicted_roi_study.roi`; legacy full-image classification heads are
frozen and are never returned by the segmenter adapter.
"""
from __future__ import annotations

import hashlib
import re
from importlib import import_module
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .roi import DEFAULT_NEUTRAL_RGB, ROIPolicy, ROIResult, extract_roi_tensor


SEGMENTER_REGISTRY: dict[str, tuple[str, str]] = {
    "yolo26": ("binary_study.models.yolo_joint", "YOLO26Joint"),
    "vit_method2": ("binary_study.models.vit_method2", "ViTMethod2Joint"),
    "emcad": ("binary_study.models.emcad_joint", "EMCADJoint"),
    "sam2_unet": ("binary_study.models.sam2_unet", "SAM2UNetJoint"),
}

RESNET18_WEIGHTS_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
RESNET18_WEIGHTS_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _freeze(module: nn.Module | None) -> int:
    if module is None:
        return 0
    count = 0
    for parameter in module.parameters():
        parameter.requires_grad_(False)
        count += parameter.numel()
    module.eval()
    return count


class SegmentationOnlyAdapter(nn.Module):
    """Suppress legacy full-image classifiers while preserving segmentation.

    The specialized forwards avoid even computing the legacy classifier for all
    four families.  The wrapped model remains checkpoint-compatible with the
    previous study.  YOLO's native tensors are retained so its native loss can
    still be computed by :func:`segmentation_loss`.
    """

    strict_predicted_roi_stage = "segmenter"

    def __init__(self, name: str, base_model: nn.Module):
        super().__init__()
        if name not in SEGMENTER_REGISTRY:
            raise KeyError(f"Unknown segmenter {name!r}")
        self.name = name
        self.base_model = base_model
        frozen = _freeze(getattr(base_model, "classifier", None))
        self.legacy_classification_parameters_frozen = frozen

    def __getattr__(self, name: str):
        """Delegate native-loss helpers such as YOLO ``set_epoch`` safely."""

        try:
            return super().__getattr__(name)
        except AttributeError:
            base = super().__getattr__("base_model")
            return getattr(base, name)

    def train(self, mode: bool = True):
        super().train(mode)
        # ``super().train`` visits all children; restore frozen legacy heads to
        # eval so dropout/batch-norm state cannot drift despite being unused.
        classifier = getattr(self.base_model, "classifier", None)
        if classifier is not None:
            classifier.eval()
        return self


    def _yolo(self, images: Tensor) -> dict[str, Any]:
        model = self.base_model
        if images.ndim != 4 or images.shape[1] != 3 or images.shape[-2:] != (model.image_size, model.image_size):
            raise ValueError(f"Expected Bx3x{model.image_size}x{model.image_size}")
        raw = model._raw_predictions(model.native_model(images))
        seg_logits, metadata = model._semantic_from_instances(raw, images.shape[-2:])
        return {"seg_logits": seg_logits, "native": raw, "native_metadata": metadata}

    def _vit_method2(self, images: Tensor) -> dict[str, Tensor]:
        model = self.base_model
        if images.ndim != 4 or images.shape[1] != 3 or images.shape[-2:] != (model.image_size, model.image_size):
            raise ValueError(f"Expected Bx3x{model.image_size}x{model.image_size}")
        embeddings = model.patch_embedding(model.patch_extractor(images)) + model.position_embedding
        tokens = embeddings
        for layer in model.encoder:
            if model.training and model.gradient_checkpointing and torch.is_grad_enabled():
                tokens = checkpoint(layer, tokens, use_reentrant=False, preserve_rng_state=True)
            else:
                tokens = layer(tokens)
        if model.training and model.gradient_checkpointing and torch.is_grad_enabled():
            fused = checkpoint(
                model.cross_attention, tokens, embeddings, use_reentrant=False, preserve_rng_state=True
            )
        else:
            fused = model.cross_attention(tokens, embeddings)
        grid = (model.image_size // model.patch_size, model.image_size // model.patch_size)
        return {"seg_logits": model.decoder(fused, grid)}


    def _emcad(self, images: Tensor) -> dict[str, Any]:
        model = self.base_model
        if images.ndim != 4 or images.shape[1] != 3 or images.shape[-2:] != (model.image_size, model.image_size):
            raise ValueError(f"Expected Bx3x{model.image_size}x{model.image_size}")
        x1, x2, x3, x4 = model._encoder((images - model.input_mean) / model.input_std)
        decoded = model.decoder(x4, [x3, x2, x1])
        logits = [
            F.interpolate(head(feature), size=images.shape[-2:], mode="bilinear", align_corners=False)
            for head, feature in zip(model.seg_heads, decoded)
        ]
        return {"seg_logits": logits[-1], "aux_seg_logits": logits[:-1]}

    def _sam2_unet(self, images: Tensor) -> dict[str, Any]:
        model = self.base_model
        if images.ndim != 4 or images.shape[1] != 3 or images.shape[-2:] != (model.image_size, model.image_size):
            raise ValueError(f"Expected Bx3x{model.image_size}x{model.image_size}")
        features = model._encoder((images - model.input_mean) / model.input_std)
        x1, x2, x3, x4 = [block(feature) for block, feature in zip(model.rfbs, features)]
        x = model.up1(x4, x3)
        aux1 = F.interpolate(model.side1(x), size=images.shape[-2:], mode="bilinear", align_corners=False)
        x = model.up2(x, x2)
        aux2 = F.interpolate(model.side2(x), size=images.shape[-2:], mode="bilinear", align_corners=False)
        x = model.up3(x, x1)
        segmentation = F.interpolate(model.head(x), size=images.shape[-2:], mode="bilinear", align_corners=False)
        return {"seg_logits": segmentation, "aux_seg_logits": [aux1, aux2]}

    def forward(self, images: Tensor) -> dict[str, Any]:
        method = {
            "yolo26": self._yolo,
            "vit_method2": self._vit_method2,
            "emcad": self._emcad,
            "sam2_unet": self._sam2_unet,
        }[self.name]
        outputs = method(images)
        if "seg_logits" not in outputs or "cls_logits" in outputs:
            raise RuntimeError("Segmentation-only adapter contract violated")
        return outputs


def build_segmenter(name: str, pretrained: bool = True, image_size: int = 768,
                    **model_kwargs: Any) -> SegmentationOnlyAdapter:
    """Build one of the four executed segmentation families with its old classifier disabled."""

    if name not in SEGMENTER_REGISTRY:
        raise KeyError(f"Unknown segmenter {name!r}; expected one of {sorted(SEGMENTER_REGISTRY)}")
    # The custom Method 2 model has no external segmentation pretraining.
    if name == "vit_method2" and "weights_path" not in model_kwargs:
        pretrained = False
    module_name, class_name = SEGMENTER_REGISTRY[name]
    model_class = getattr(import_module(module_name), class_name)
    model = model_class(pretrained=pretrained, image_size=image_size, **model_kwargs)
    return SegmentationOnlyAdapter(name, model)


def segmentation_loss(model: nn.Module, outputs: dict[str, Any], batch: dict[str, Tensor]) -> Tensor:
    """Preserve the previous study's segmentation objective without clinical CE."""

    base = model.base_model if isinstance(model, SegmentationOnlyAdapter) else model
    if hasattr(base, "compute_native_loss"):
        return base.compute_native_loss(outputs, batch)
    logits, target = outputs["seg_logits"].float(), batch["mask"].float()
    if logits.shape != target.shape:
        raise ValueError(f"Segmentation output {tuple(logits.shape)} != target {tuple(target.shape)}")
    probability = logits.sigmoid()
    dims = (1, 2, 3)
    dice = 1.0 - (
        (2.0 * (probability * target).sum(dims) + 1e-6)
        / (probability.sum(dims) + target.sum(dims) + 1e-6)
    ).mean()
    return F.binary_cross_entropy_with_logits(logits, target) + dice


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ResNet18ROIClassifier(nn.Module):
    """Common ImageNet ResNet-18 used identically in the secondary estimand.

    The standardized comparison uses ``geometry_features=0`` (appearance only).
    Setting it to four enables a predeclared mask-geometry ablation; those four
    values are area fraction, bounding-box width/height fractions and log aspect
    ratio.
    """

    strict_predicted_roi_stage = "classifier"
    input_size = (224, 224)

    def __init__(self, *, pretrained: bool = True, num_classes: int = 2,
                 dropout: float = 0.2, geometry_features: int = 0,
                 weights_path: str | Path | None = None,
                 expected_sha256: str | None = None):
        super().__init__()
        if num_classes != 2:
            raise ValueError("This study requires binary (two-logit) classification")
        if geometry_features not in {0, 4}:
            raise ValueError("geometry_features must be 0 (primary) or 4 (mask-geometry ablation)")
        location: Path | None = None
        actual_sha256: str | None = None
        # Validate bytes before importing torchvision as well as before torch
        # deserialization.  This keeps preflight useful in minimal environments
        # and guarantees an untrusted/mismatched file is rejected first.
        if pretrained:
            if weights_path is None or expected_sha256 is None:
                raise ValueError("pretrained=True requires local weights_path and expected_sha256; network download is disabled")
            if not isinstance(expected_sha256, str) or not SHA256_RE.fullmatch(expected_sha256):
                raise ValueError("expected_sha256 must contain exactly 64 hexadecimal characters")
            location = Path(weights_path)
            if not location.is_file():
                raise FileNotFoundError(f"Required local ResNet-18 weights not found: {location}")
            actual_sha256 = _sha256(location)
            if actual_sha256.lower() != expected_sha256.lower():
                raise ValueError(
                    f"ResNet-18 checkpoint SHA256 mismatch: expected {expected_sha256.lower()}, "
                    f"found {actual_sha256.lower()}. File was not deserialized."
                )
        try:
            from torchvision.models import resnet18
        except ImportError as exc:
            raise ImportError("The common ROI classifier requires torchvision") from exc
        # Construct without torchvision's weight enum: enums may implicitly
        # download when the cache is absent, which the locked protocol forbids.
        self.encoder = resnet18(weights=None)
        self.pretrained_provenance: dict[str, Any] = {"pretrained": False, "source": None}
        if pretrained:
            assert location is not None and actual_sha256 is not None
            state = torch.load(location, map_location="cpu", weights_only=True)
            if not isinstance(state, dict):
                raise ValueError("ResNet-18 checkpoint must contain a state dictionary")
            self.encoder.load_state_dict(state, strict=True)
            self.pretrained_provenance = {
                "pretrained": True,
                "library": "torchvision",
                "architecture": "resnet18",
                "weights_name": "ResNet18_Weights.IMAGENET1K_V1",
                "source": RESNET18_WEIGHTS_URL,
                "path": str(location.resolve()),
                "bytes": location.stat().st_size,
                "sha256": actual_sha256,
                "verified_before_deserialization": True,
                "network_download_during_run": False,
            }
        feature_width = int(self.encoder.fc.in_features)
        self.encoder.fc = nn.Identity()
        self.geometry_features = int(geometry_features)
        self.head = nn.Sequential(
            nn.LayerNorm(feature_width + self.geometry_features),
            nn.Dropout(float(dropout)),
            nn.Linear(feature_width + self.geometry_features, num_classes),
        )
        # Neutral raw RGB becomes exactly zero after this fixed normalization.
        self.register_buffer("input_mean", torch.tensor(DEFAULT_NEUTRAL_RGB).view(1, 3, 1, 1))
        self.register_buffer("input_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1))

    @property
    def neutral_rgb(self) -> tuple[float, float, float]:
        return tuple(float(value) for value in self.input_mean.flatten())

    def forward(self, roi_images: Tensor, geometry: Tensor | None = None) -> Tensor:
        if roi_images.ndim != 4 or roi_images.shape[1] != 3:
            raise ValueError("roi_images must have shape Bx3xHxW")
        if tuple(roi_images.shape[-2:]) != self.input_size:
            raise ValueError(f"ROI classifier requires {self.input_size}, got {tuple(roi_images.shape[-2:])}")
        features = self.encoder((roi_images - self.input_mean) / self.input_std)
        if self.geometry_features:
            if geometry is None or geometry.shape != (len(roi_images), self.geometry_features):
                raise ValueError(f"geometry must have shape Bx{self.geometry_features}")
            features = torch.cat((features, geometry.to(dtype=features.dtype)), dim=1)
        elif geometry is not None:
            raise ValueError("Primary appearance-only classifier must not receive geometry")
        return self.head(features)


def build_roi_classifier(backbone: str = "resnet18", *, pretrained: bool = True,
                         num_classes: int = 2, dropout: float = 0.2,
                         geometry_features: int = 0,
                         weights_path: str | Path | None = None,
                         expected_sha256: str | None = None) -> ResNet18ROIClassifier:
    """Build the common classifier; architecture variation is intentionally disallowed."""

    if backbone != "resnet18":
        raise KeyError("The locked common ROI classifier architecture is 'resnet18'")
    return ResNet18ROIClassifier(
        pretrained=pretrained,
        num_classes=num_classes,
        dropout=dropout,
        geometry_features=geometry_features,
        weights_path=weights_path,
        expected_sha256=expected_sha256,
    )


MODEL_SPECIFIC_CLASSIFIER_ARCHITECTURES: dict[str, str] = {
    "yolo26": "YOLO26s backbone/neck pooled features and original MLP",
    "vit_method2": "Method2 encoder/cross-attention global plus hard-ROI-weighted token and original MLP",
    "emcad": "PVTv2-B0 final feature GAP and original binary head",
    "sam2_unet": "SAM2 Hiera-Tiny final feature GAP and original binary head",
}

MODEL_SPECIFIC_CLASSIFIER_ARCHITECTURE_IDS: dict[str, str] = {
    "yolo26": "yolo26s_roi_classifier",
    "vit_method2": "vit_method2_roi_classifier",
    "emcad": "pvt_v2_b0_roi_classifier",
    "sam2_unet": "sam2_hiera_tiny_roi_classifier",
}

MODEL_SPECIFIC_DEFAULT_DROPOUT: dict[str, float] = {
    "yolo26": 0.10,
    "vit_method2": 0.10,
    "emcad": 0.10,
    "sam2_unet": 0.10,
}


def _verified_local_weight(
    family: str,
    weights_path: str | Path | None,
    expected_sha256: str | None,
) -> tuple[Path, str]:
    """Validate local bytes before deserialization; this function never downloads."""

    if weights_path is None or expected_sha256 is None:
        raise ValueError(
            f"pretrained=True for {family} requires a local weights_path and expected_sha256"
        )
    if not isinstance(expected_sha256, str) or not SHA256_RE.fullmatch(expected_sha256):
        raise ValueError("expected_sha256 must contain exactly 64 hexadecimal characters")
    location = Path(weights_path)
    if not location.is_file():
        raise FileNotFoundError(f"Required local {family} weights not found: {location}")
    actual = _sha256(location)
    if actual.lower() != expected_sha256.lower():
        raise ValueError(
            f"{family} checkpoint SHA256 mismatch: expected {expected_sha256.lower()}, "
            f"found {actual.lower()}. File was not deserialized."
        )
    return location, actual.lower()


def _unwrap_state_dict(payload: Any) -> Mapping[str, Tensor]:
    if not isinstance(payload, Mapping):
        raise ValueError("Checkpoint must contain a state dictionary")
    for key in ("model", "state_dict"):
        candidate = payload.get(key)
        if isinstance(candidate, Mapping):
            payload = candidate
            break
    if not isinstance(payload, Mapping) or not all(isinstance(key, str) for key in payload):
        raise ValueError("Checkpoint must contain a string-keyed state dictionary")
    return payload


class StrictROIFamilyClassifier(nn.Module):
    """Base contract for primary model-specific classifiers.

    Inputs are already tight-cropped and letterboxed by :mod:`roi`, but the
    supplied hard mask is deliberately re-applied here.  This makes outside-ROI
    invariance a property of the model boundary instead of a caller convention.
    If a mask is omitted, the exact neutral canvas is used to reconstruct it;
    production loaders should pass the cached hard mask explicitly.
    """

    strict_predicted_roi_stage = "classifier"
    classifier_strategy = "model_specific"
    input_size = (224, 224)
    model_family: str

    def __init__(self, family: str, *, geometry_features: int = 0, lightweight: bool = False):
        super().__init__()
        if family not in MODEL_SPECIFIC_CLASSIFIER_ARCHITECTURES:
            raise KeyError(f"Unknown model-specific classifier family: {family}")
        if int(geometry_features) not in {0, 4}:
            raise ValueError("geometry_features must be 0 (primary) or 4 (secondary ablation)")
        self.model_family = family
        self.geometry_features = int(geometry_features)
        self.lightweight = bool(lightweight)
        self.register_buffer(
            "strict_neutral_rgb",
            torch.tensor(DEFAULT_NEUTRAL_RGB, dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.geometry_logit_residual = (
            nn.Linear(self.geometry_features, 2, bias=False) if self.geometry_features else None
        )
        self.pretrained_provenance: dict[str, Any] = {
            "pretrained": False,
            "mode": "test_only_lightweight_random" if lightweight else "random_initialization",
            "source": None,
            "path": None,
            "sha256": None,
            "verified_before_deserialization": False,
            "network_download_during_run": False,
        }
        self.explicitly_disabled_parameters = 0

    @property
    def neutral_rgb(self) -> tuple[float, float, float]:
        return tuple(float(value) for value in self.strict_neutral_rgb.flatten())

    @property
    def encoder(self) -> nn.Module:
        """Feature module exposed for the predeclared frozen-encoder ablation."""

        return self.feature_extractor

    def _strict_input(
        self, roi_images: Tensor, hard_mask: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        if (
            roi_images.ndim != 4
            or roi_images.shape[1] != 3
            or tuple(roi_images.shape[-2:]) != self.input_size
        ):
            raise ValueError(
                f"Strict ROI classifier requires Bx3x{self.input_size[0]}x{self.input_size[1]}"
            )
        if not roi_images.is_floating_point() or not bool(torch.isfinite(roi_images).all()):
            raise ValueError("ROI images must be finite floating-point RGB tensors")
        neutral = self.strict_neutral_rgb.to(device=roi_images.device, dtype=roi_images.dtype)
        if hard_mask is None:
            mask = (roi_images - neutral).abs().amax(dim=1, keepdim=True) > 1e-7
        else:
            mask = hard_mask
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            if tuple(mask.shape) != (len(roi_images), 1, *self.input_size):
                raise ValueError(
                    f"hard_mask must have shape Bx1x{self.input_size[0]}x{self.input_size[1]}"
                )
            if mask.dtype != torch.bool:
                if not mask.is_floating_point() and mask.dtype not in (
                    torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64
                ):
                    raise ValueError("hard_mask must be boolean or binary numeric")
                if not bool(torch.isfinite(mask).all()) or not bool(((mask == 0) | (mask == 1)).all()):
                    raise ValueError("hard_mask must contain only zero and one")
                mask = mask.bool()
            mask = mask.to(device=roi_images.device)
        if not bool(mask.flatten(1).any(dim=1).all()):
            raise ValueError("Empty ROI masks must abstain upstream and cannot enter a classifier")
        return torch.where(mask, roi_images, neutral), mask

    def _finish_logits(self, logits: Tensor, geometry: Tensor | None) -> Tensor:
        if logits.shape != (len(logits), 2) or not bool(torch.isfinite(logits).all()):
            raise RuntimeError("Model-specific ROI classifier must return finite Bx2 logits")
        if self.geometry_features:
            if geometry is None or geometry.shape != (len(logits), self.geometry_features):
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
        registered = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return {
            "family": self.model_family,
            "strategy": self.classifier_strategy,
            "architecture_id": MODEL_SPECIFIC_CLASSIFIER_ARCHITECTURE_IDS[
                self.model_family
            ],
            "architecture": MODEL_SPECIFIC_CLASSIFIER_ARCHITECTURES[self.model_family],
            "input_size": list(self.input_size),
            "registered_parameters": int(registered),
            "trainable_updateable_parameters": int(trainable),
            "frozen_parameters": int(registered - trainable),
            "explicitly_disabled_parameters": int(self.explicitly_disabled_parameters),
            "geometry_features": self.geometry_features,
            "lightweight_test_variant": self.lightweight,
            "strict_hard_mask_reapplied": True,
            "full_image_pixels_consumed": False,
            "network_download_during_run": False,
            "pretrained_provenance": dict(self.pretrained_provenance),
        }






class _TinyYOLOFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = nn.Sequential(nn.Conv2d(3, 8, 3, 2, 1), nn.SiLU())
        self.stage2 = nn.Sequential(nn.Conv2d(8, 16, 3, 2, 1), nn.SiLU())
        self.stage3 = nn.Sequential(nn.Conv2d(16, 32, 3, 2, 1), nn.SiLU())

    def forward(self, images: Tensor) -> dict[str, Any]:
        one = self.stage1(images)
        two = self.stage2(one)
        three = self.stage3(two)
        return {"one2many": {"feats": [one, two, three]}}


class YOLO26StrictROIClassifier(StrictROIFamilyClassifier):
    def __init__(
        self, *, pretrained: bool, weights_path: str | Path | None,
        expected_sha256: str | None, dropout: float, geometry_features: int,
        lightweight: bool,
    ):
        super().__init__("yolo26", geometry_features=geometry_features, lightweight=lightweight)
        if lightweight:
            if pretrained:
                raise ValueError("The lightweight test variant cannot load pretrained weights")
            self.feature_extractor = _TinyYOLOFeatureExtractor()
            channels = (8, 16, 32)
        else:
            location: Path | None = None
            digest: str | None = None
            if pretrained:
                location, digest = _verified_local_weight(
                    self.model_family, weights_path, expected_sha256
                )
            from binary_study.models.yolo_joint import YOLO26Joint

            source = YOLO26Joint(
                image_size=224,
                pretrained=pretrained,
                weights_path=location if pretrained else None,
                dropout=dropout,
            )
            self.feature_extractor = source.native_model
            channels = tuple(int(value) for value in source.neck_channels)
            # The native detection/segmentation head must execute to expose its
            # neck tensor contract, but it is not optimized by diagnosis loss.
            native_head = self.feature_extractor.model[-1]
            native_head.requires_grad_(False)
            self.explicitly_disabled_parameters = sum(p.numel() for p in native_head.parameters())
            if pretrained:
                assert location is not None and digest is not None
                self.pretrained_provenance = {
                    **dict(source.pretraining_info),
                    "mode": "verified_local_yolo26s_backbone_neck",
                    "path": str(location.resolve()),
                    "sha256": digest,
                    "network_download_during_run": False,
                }
        self.neck_channels = channels
        self.classification_head = nn.Sequential(
            nn.LayerNorm(sum(channels)), nn.Linear(sum(channels), 256), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(256, 2),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.lightweight:
            self.feature_extractor.model[-1].eval()
        return self

    def forward(
        self, roi_images: Tensor, geometry: Tensor | None = None, *, hard_mask: Tensor | None = None
    ) -> Tensor:
        images, _ = self._strict_input(roi_images, hard_mask)
        native = self.feature_extractor(images)
        if self.lightweight:
            raw = native
        else:
            from binary_study.models.yolo_joint import YOLO26Joint

            raw = YOLO26Joint._raw_predictions(native)
        features = raw["one2many"]["feats"]
        if len(features) != len(self.neck_channels):
            raise RuntimeError("YOLO26 neck feature contract changed")
        pooled = torch.cat([feature.mean(dim=(-2, -1)) for feature in features], dim=1)
        return self._finish_logits(self.classification_head(pooled).float(), geometry)


class _Method2ROIEncoder(nn.Module):
    def __init__(
        self, *, embed_dim: int, depth: int, num_heads: int, dropout: float,
        gradient_checkpointing: bool,
    ):
        super().__init__()
        from binary_study.models.vit_method2 import (
            CrossAttention, PatchExtractor, TransformerEncoderBlock,
        )

        self.patch_extractor = PatchExtractor(16)
        self.patch_embedding = nn.Linear(3 * 16 * 16, embed_dim)
        self.position_embedding = nn.Parameter(torch.randn(1, 14 * 14, embed_dim))
        self.encoder = nn.ModuleList(
            [TransformerEncoderBlock(embed_dim, num_heads, dropout=dropout) for _ in range(depth)]
        )
        self.cross_attention = CrossAttention(embed_dim, num_heads, dropout)
        self.embed_dim = int(embed_dim)
        self.gradient_checkpointing = bool(gradient_checkpointing)

    def forward(self, images: Tensor) -> Tensor:
        embeddings = self.patch_embedding(self.patch_extractor(images)) + self.position_embedding
        tokens = embeddings
        for layer in self.encoder:
            if self.training and self.gradient_checkpointing and torch.is_grad_enabled():
                tokens = checkpoint(layer, tokens, use_reentrant=False, preserve_rng_state=True)
            else:
                tokens = layer(tokens)
        if self.training and self.gradient_checkpointing and torch.is_grad_enabled():
            return checkpoint(
                self.cross_attention, tokens, embeddings,
                use_reentrant=False, preserve_rng_state=True,
            )
        return self.cross_attention(tokens, embeddings)


def _adapt_method2_position_embedding(value: Tensor, *, width: int) -> Tensor:
    if value.ndim != 3 or value.shape[0] != 1 or value.shape[2] != width:
        raise ValueError("Method2 position embedding has an incompatible shape")
    source_grid = int(round(value.shape[1] ** 0.5))
    if source_grid * source_grid != value.shape[1]:
        raise ValueError("Method2 position embedding is not a square token grid")
    if source_grid == 14:
        return value
    return F.interpolate(
        value.reshape(1, source_grid, source_grid, width).permute(0, 3, 1, 2),
        size=(14, 14), mode="bicubic", align_corners=False,
    ).permute(0, 2, 3, 1).reshape(1, 14 * 14, width)


class ViTMethod2StrictROIClassifier(StrictROIFamilyClassifier):
    def __init__(
        self, *, pretrained: bool, weights_path: str | Path | None,
        expected_sha256: str | None, dropout: float, geometry_features: int,
        lightweight: bool,
    ):
        super().__init__("vit_method2", geometry_features=geometry_features, lightweight=lightweight)
        width, depth, heads = (32, 1, 4) if lightweight else (256, 6, 8)
        if lightweight and pretrained:
            raise ValueError("The lightweight test variant cannot load pretrained weights")
        self.feature_extractor = _Method2ROIEncoder(
            embed_dim=width, depth=depth, num_heads=heads, dropout=dropout,
            gradient_checkpointing=not lightweight,
        )
        self.classification_head = nn.Sequential(
            nn.LayerNorm(2 * width), nn.Linear(2 * width, width), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(width, 2),
        )
        self.position_adaptation = "square_grid_bicubic_to_14x14"
        if pretrained:
            location, digest = _verified_local_weight(
                self.model_family, weights_path, expected_sha256
            )
            source = _unwrap_state_dict(torch.load(location, map_location="cpu", weights_only=True))
            normalized: dict[str, Tensor] = {}
            for key, value in source.items():
                key = key.removeprefix("module.").removeprefix("base_model.")
                if key.startswith(("patch_extractor.", "patch_embedding.", "encoder.", "cross_attention.")):
                    normalized[f"feature_extractor.{key}"] = value
                elif key.startswith("classifier."):
                    normalized[f"classification_head.{key.removeprefix('classifier.')}"] = value
                elif key == "position_embedding":
                    normalized["feature_extractor.position_embedding"] = _adapt_method2_position_embedding(
                        value, width=width
                    )
            required = set(self.feature_extractor.state_dict())
            loaded_encoder = {
                key.removeprefix("feature_extractor.")
                for key in normalized if key.startswith("feature_extractor.")
            }
            missing = sorted(required - loaded_encoder)
            if missing:
                raise ValueError(f"Method2 checkpoint is missing encoder keys: {missing[:8]}")
            incompatible = self.load_state_dict(normalized, strict=False)
            unexpected = [key for key in incompatible.unexpected_keys if not key.startswith("decoder.")]
            if unexpected:
                raise ValueError(f"Unexpected Method2 checkpoint keys: {unexpected[:8]}")
            self.pretrained_provenance = {
                "pretrained": True,
                "mode": "verified_local_method2_encoder",
                "source": "explicit compatible local Method2 checkpoint",
                "path": str(location.resolve()),
                "bytes": location.stat().st_size,
                "sha256": digest,
                "position_adaptation": self.position_adaptation,
                "verified_before_deserialization": True,
                "network_download_during_run": False,
            }

    def forward(
        self, roi_images: Tensor, geometry: Tensor | None = None, *, hard_mask: Tensor | None = None
    ) -> Tensor:
        images, mask = self._strict_input(roi_images, hard_mask)
        fused = self.feature_extractor(images)
        weights = F.adaptive_avg_pool2d(mask.float(), (14, 14)).flatten(2).transpose(1, 2)
        global_feature = fused.float().mean(dim=1)
        local_feature = (fused.float() * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1e-6)
        logits = self.classification_head(
            torch.cat((global_feature, local_feature), dim=1).to(fused.dtype)
        )
        return self._finish_logits(logits.float(), geometry)






class _TinyFeatureMapEncoder(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.output_channels = int(width)
        self.layers = nn.Sequential(
            nn.Conv2d(3, width // 2, 3, 2, 1), nn.GELU(),
            nn.Conv2d(width // 2, width, 3, 2, 1), nn.GELU(),
        )

    def forward(self, images: Tensor) -> Tensor:
        return self.layers(images)


class EMCADStrictROIClassifier(StrictROIFamilyClassifier):
    def __init__(
        self, *, pretrained: bool, weights_path: str | Path | None,
        expected_sha256: str | None, dropout: float, geometry_features: int,
        lightweight: bool,
    ):
        super().__init__("emcad", geometry_features=geometry_features, lightweight=lightweight)
        if lightweight:
            if pretrained:
                raise ValueError("The lightweight test variant cannot load pretrained weights")
            self.feature_extractor = _TinyFeatureMapEncoder(32)
            width = 32
        else:
            from binary_study.models.emcad_external import pvt_v2_b0

            self.feature_extractor = pvt_v2_b0()
            self.feature_extractor.head = nn.Identity()
            width = 256
            if pretrained:
                location, digest = _verified_local_weight(
                    self.model_family, weights_path, expected_sha256
                )
                state = dict(_unwrap_state_dict(torch.load(location, map_location="cpu", weights_only=True)))
                state = {key.removeprefix("module."): value for key, value in state.items()}
                omitted = sorted(key for key in state if key.startswith("head."))
                trunk = {key: value for key, value in state.items() if not key.startswith("head.")}
                self.feature_extractor.load_state_dict(trunk, strict=True)
                self.pretrained_provenance = {
                    "pretrained": True,
                    "mode": "verified_local_pvtv2_b0_encoder",
                    "source": "https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b0.pth",
                    "path": str(location.resolve()),
                    "bytes": location.stat().st_size,
                    "sha256": digest,
                    "loaded": "PVTv2-B0 complete encoder",
                    "omitted_keys": omitted,
                    "verified_before_deserialization": True,
                    "network_download_during_run": False,
                }
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(width, 2)
        )

    def forward(
        self, roi_images: Tensor, geometry: Tensor | None = None, *, hard_mask: Tensor | None = None
    ) -> Tensor:
        images, _ = self._strict_input(roi_images, hard_mask)
        mean = self.strict_neutral_rgb.to(device=images.device, dtype=images.dtype)
        std = images.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        normalized = (images - mean) / std
        if self.lightweight:
            final_feature = self.feature_extractor(normalized)
        else:
            features = self.feature_extractor.forward_features(normalized)
            final_feature = features[-1]
        return self._finish_logits(self.classification_head(final_feature).float(), geometry)


class SAM2UNetStrictROIClassifier(StrictROIFamilyClassifier):
    def __init__(
        self, *, pretrained: bool, weights_path: str | Path | None,
        expected_sha256: str | None, dropout: float, geometry_features: int,
        lightweight: bool,
    ):
        super().__init__("sam2_unet", geometry_features=geometry_features, lightweight=lightweight)
        if lightweight:
            if pretrained:
                raise ValueError("The lightweight test variant cannot load pretrained weights")
            self.feature_extractor = _TinyFeatureMapEncoder(32)
            self.original_hiera_trunk_parameters = 0
            self.trainable_adapter_parameters = 0
            width = 32
        else:
            from binary_study.models.vendor.emcad_sam.hieradet import Hiera
            from binary_study.models.vendor.emcad_sam.sam2_unet_blocks import Adapter

            encoder = Hiera(
                embed_dim=96, num_heads=1, stages=(1, 2, 7, 2),
                global_att_blocks=(5, 7, 9), window_pos_embed_bkg_spatial_size=(7, 7),
            )
            if pretrained:
                location, digest = _verified_local_weight(
                    self.model_family, weights_path, expected_sha256
                )
                payload = torch.load(location, map_location="cpu", weights_only=True)
                if not isinstance(payload, Mapping) or not isinstance(payload.get("model"), Mapping):
                    raise ValueError("Official SAM2 checkpoint must contain a model state dictionary")
                prefix = "image_encoder.trunk."
                trunk = {
                    key[len(prefix):]: value
                    for key, value in payload["model"].items() if key.startswith(prefix)
                }
                if not trunk:
                    raise ValueError("Official SAM2 checkpoint contains no image_encoder.trunk weights")
                encoder.load_state_dict(trunk, strict=True)
                self.pretrained_provenance = {
                    "pretrained": True,
                    "mode": "verified_local_sam2_hiera_tiny_encoder",
                    "source": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt",
                    "path": str(location.resolve()),
                    "bytes": location.stat().st_size,
                    "sha256": digest,
                    "loaded": "complete original SAM2 Hiera-Tiny trunk",
                    "loaded_keys": len(trunk),
                    "verified_before_deserialization": True,
                    "network_download_during_run": False,
                }
            # Preserve the author SAM2-UNet optimization policy exactly: the
            # original Hiera trunk (including patch/position embeddings and all
            # native attention blocks) is frozen *before* trainable adapters are
            # introduced.  Freezing after wrapping would incorrectly freeze the
            # adapters too; omitting this step would fine-tune the full trunk.
            encoder.requires_grad_(False)
            self.original_hiera_trunk_parameters = sum(
                parameter.numel() for parameter in encoder.parameters()
            )
            encoder.blocks = nn.Sequential(*(Adapter(block) for block in encoder.blocks))
            self.trainable_adapter_parameters = sum(
                parameter.numel()
                for block in encoder.blocks
                for parameter in block.prompt_learn.parameters()
                if parameter.requires_grad
            )
            self.feature_extractor = encoder
            width = 768
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(width, 2)
        )
        self.pretrained_provenance.update(
            {
                "original_hiera_trunk_frozen": not lightweight,
                "trainable_hiera_adapters": not lightweight,
                "trainable_classification_head": True,
            }
        )

    def forward(
        self, roi_images: Tensor, geometry: Tensor | None = None, *, hard_mask: Tensor | None = None
    ) -> Tensor:
        images, _ = self._strict_input(roi_images, hard_mask)
        mean = self.strict_neutral_rgb.to(device=images.device, dtype=images.dtype)
        std = images.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        normalized = (images - mean) / std
        if self.lightweight:
            final_feature = self.feature_extractor(normalized)
        else:
            features = self.feature_extractor(normalized)
            final_feature = features[-1]
        return self._finish_logits(self.classification_head(final_feature).float(), geometry)

    def parameter_info(self) -> dict[str, Any]:
        value = super().parameter_info()
        value.update(
            {
                "original_hiera_trunk_frozen": not self.lightweight,
                "original_hiera_trunk_parameters": int(self.original_hiera_trunk_parameters),
                "trainable_adapter_parameters": int(self.trainable_adapter_parameters),
                "classification_head_trainable_parameters": int(
                    sum(
                        parameter.numel()
                        for parameter in self.classification_head.parameters()
                        if parameter.requires_grad
                    )
                ),
                "optimization_policy": (
                    "frozen_original_hiera_trunk_plus_trainable_adapters_and_binary_head"
                    if not self.lightweight else "test_only_tiny_encoder_and_binary_head"
                ),
            }
        )
        return value


MODEL_SPECIFIC_CLASSIFIER_REGISTRY: dict[str, type[StrictROIFamilyClassifier]] = {
    "yolo26": YOLO26StrictROIClassifier,
    "vit_method2": ViTMethod2StrictROIClassifier,
    "emcad": EMCADStrictROIClassifier,
    "sam2_unet": SAM2UNetStrictROIClassifier,
}


def build_strict_roi_classifier(
    family: str,
    *,
    pretrained: bool = False,
    weights_path: str | Path | None = None,
    expected_sha256: str | None = None,
    dropout: float | None = None,
    geometry_features: int = 0,
    lightweight: bool = False,
) -> StrictROIFamilyClassifier:
    """Build the primary family-specific classifier with no network fallback.

    ``pretrained=False`` always constructs the exact same state-dict topology
    without reading a weight file.  Evaluation code can therefore construct
    this shell and strictly restore the selected classifier checkpoint.
    ``lightweight`` exists solely for synthetic/unit audits and must never be
    used for a clinical run.
    """

    if family not in MODEL_SPECIFIC_CLASSIFIER_REGISTRY:
        raise KeyError(
            f"Unknown classifier family {family!r}; expected one of "
            f"{sorted(MODEL_SPECIFIC_CLASSIFIER_REGISTRY)}"
        )
    selected_dropout = (
        MODEL_SPECIFIC_DEFAULT_DROPOUT[family] if dropout is None else float(dropout)
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


# Explicit alias used by orchestration code and manuscript tables.
build_model_specific_roi_classifier = build_strict_roi_classifier


class PredictedROIPipeline(nn.Module):
    """Deployable segment -> quality gate -> hard ROI -> classifier pipeline.

    Invalid frames return NaN logits deliberately.  Downstream eye aggregation
    must use ``valid_mask`` and apply its locked minimum-valid-slice rule; it may
    never replace an abstention with full-image classification.
    """

    def __init__(self, segmenter: nn.Module, classifier: nn.Module,
                 roi_policy: ROIPolicy):
        super().__init__()
        self.segmenter = segmenter
        self.classifier = classifier
        self.roi_policy = roi_policy
        self.segmenter.requires_grad_(False)
        self.segmenter.eval()
        if tuple(roi_policy.image_shape) != (
            int(getattr(segmenter, "image_size", roi_policy.image_shape[0])),
            int(getattr(segmenter, "image_size", roi_policy.image_shape[1])),
        ):
            raise ValueError("ROI policy image shape and segmenter nominal size differ")

    def train(self, mode: bool = True):
        """Train/evaluate the classifier while the fitted segmenter stays frozen."""

        super().train(mode)
        self.segmenter.eval()
        self.classifier.train(mode)
        return self

    def forward(self, images: Tensor) -> dict[str, Any]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape Bx3xHxW")
        # The two-stage protocol freezes the segmenter; classifier gradients do
        # not cross hard thresholding or component selection.
        with torch.no_grad():
            outputs = self.segmenter(images)
            seg_logits = outputs["seg_logits"]
            probabilities = seg_logits.float().sigmoid().detach().cpu().numpy()[:, 0]
        results: list[ROIResult] = [self.roi_policy.postprocess_probability(item) for item in probabilities]
        valid_indices = [index for index, result in enumerate(results) if result.valid]
        valid_mask = torch.zeros(len(images), dtype=torch.bool, device=images.device)
        if valid_indices:
            valid_mask[torch.as_tensor(valid_indices, device=images.device)] = True
            extractions = [
                extract_roi_tensor(
                    images[index], results[index].mask,
                    target_size=self.classifier.input_size,
                    neutral=self.classifier.neutral_rgb,
                    return_metadata=True,
                )
                for index in valid_indices
            ]
            roi_images = torch.stack([item.tensor for item in extractions])
            roi_masks = torch.stack([item.output_mask for item in extractions]).bool()
            geometry = None
            if int(getattr(self.classifier, "geometry_features", 0)):
                geometry = torch.stack([item.geometry for item in extractions])
            if getattr(self.classifier, "classifier_strategy", None) == "model_specific":
                valid_logits = self.classifier(
                    roi_images, geometry, hard_mask=roi_masks
                )
            else:
                # Backward-compatible secondary standardized ResNet path.  Its
                # input is already re-masked by ``extract_roi_tensor``.
                valid_logits = self.classifier(roi_images, geometry)
            cls_logits = valid_logits.new_full((len(images), 2), torch.nan).index_copy(
                0, torch.as_tensor(valid_indices, device=valid_logits.device), valid_logits
            )
        else:
            roi_images = images.new_empty((0, 3, *self.classifier.input_size))
            roi_masks = torch.empty(
                (0, *self.classifier.input_size), dtype=torch.bool, device=images.device
            )
            cls_logits = images.new_full((len(images), 2), torch.nan)
        return {
            "seg_logits": seg_logits,
            "cls_logits": cls_logits,
            "valid_mask": valid_mask,
            "roi_results": results,
            "valid_indices": valid_indices,
            "roi_images": roi_images,
            "roi_masks": roi_masks,
        }


__all__ = [
    "EMCADStrictROIClassifier",
    "HybridTALONStrictROIClassifier",
    "MODEL_SPECIFIC_CLASSIFIER_ARCHITECTURE_IDS",
    "MODEL_SPECIFIC_CLASSIFIER_ARCHITECTURES",
    "MODEL_SPECIFIC_CLASSIFIER_REGISTRY",
    "MODEL_SPECIFIC_DEFAULT_DROPOUT",
    "PConvViTStrictROIClassifier",
    "PredictedROIPipeline",
    "ResNet18ROIClassifier",
    "SEGMENTER_REGISTRY",
    "RESNET18_WEIGHTS_SHA256",
    "RESNET18_WEIGHTS_URL",
    "SAM2UNetStrictROIClassifier",
    "SegmentationOnlyAdapter",
    "StrictROIFamilyClassifier",
    "ViTMethod2StrictROIClassifier",
    "YOLO26StrictROIClassifier",
    "build_model_specific_roi_classifier",
    "build_roi_classifier",
    "build_segmenter",
    "build_strict_roi_classifier",
    "segmentation_loss",
]
