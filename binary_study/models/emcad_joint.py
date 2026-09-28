"""Independent release adapter for externally supplied EMCAD/PVT components.

This adapter preserves the study's public tensor interface, registered module
names, four-stage encoder, normalization and segmentation-head ordering. It is
new release code, not the source file attested by the historical experiment.
The separately licensed upstream components are loaded by emcad_external.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .emcad_external import build_emcad_decoder, pvt_v2_b0
from .vendor.emcad_sam.weights import official_weight


class EMCADJoint(nn.Module):
    """Expose one full-resolution mask, three auxiliary masks and two logits."""

    def __init__(self, pretrained=True, image_size=768, gradient_checkpointing=True):
        super().__init__()
        self.image_size = int(image_size)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        for name, values in (
            ("input_mean", (0.485, 0.456, 0.406)),
            ("input_std", (0.229, 0.224, 0.225)),
        ):
            self.register_buffer(name, torch.tensor(values).reshape(1, 3, 1, 1))
        self.backbone = pvt_v2_b0()
        self.pretrained_provenance = {"pretrained": False}
        if pretrained:
            self._load_encoder_weights()
        widths = (256, 160, 64, 32)
        self.decoder = build_emcad_decoder(
            channels=list(widths), kernel_sizes=[1, 3, 5], expansion_factor=2,
            dw_parallel=True, add=True, lgag_ks=3, activation="relu",
        )
        self.seg_heads = nn.ModuleList(nn.Conv2d(width, 1, kernel_size=1) for width in widths)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(p=0.1), nn.Linear(256, 2)
        )

    def _load_encoder_weights(self):
        checkpoint_path, recorded = official_weight("emcad_joint")
        state = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
        state = state.get("state_dict", state)
        cleaned = {name.removeprefix("module."): value for name, value in state.items()}
        omitted = sorted(name for name in cleaned if name.startswith("head."))
        self.backbone.load_state_dict(
            {name: value for name, value in cleaned.items() if name not in omitted}, strict=True
        )
        self.pretrained_provenance = {
            **recorded, "pretrained": True, "loaded": "PVTv2-B0 complete encoder", "omitted_keys": omitted
        }

    def _encoder(self, image: Tensor) -> list[Tensor]:
        """Return ascending-stage feature maps, with training-only checkpointing."""
        outputs = []
        feature_map = image
        for index in (1, 2, 3, 4):
            embedding = getattr(self.backbone, f"patch_embed{index}")
            blocks = getattr(self.backbone, f"block{index}")
            normalization = getattr(self.backbone, f"norm{index}")
            sequence, rows, columns = embedding(feature_map)
            for block in blocks:
                recompute = self.training and self.gradient_checkpointing and torch.is_grad_enabled()
                sequence = (checkpoint(block, sequence, rows, columns, use_reentrant=False)
                            if recompute else block(sequence, rows, columns))
            sequence = normalization(sequence)
            feature_map = sequence.reshape(image.shape[0], rows, columns, -1)
            feature_map = feature_map.permute(0, 3, 1, 2).contiguous()
            outputs.append(feature_map)
        return outputs

    def forward(self, images: Tensor) -> dict[str, Tensor | list[Tensor]]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("EMCADJoint requires a Bx3xHxW RGB tensor")
        required_size = (self.image_size, self.image_size)
        if tuple(images.shape[-2:]) != required_size:
            raise ValueError(f"EMCADJoint requires native spatial size {required_size}; it never resizes input")
        stages = self._encoder((images - self.input_mean) / self.input_std)
        decoded = self.decoder(stages[-1], stages[-2::-1])
        masks = [
            F.interpolate(head(feature), required_size, mode="bilinear", align_corners=False)
            for head, feature in zip(self.seg_heads, decoded)
        ]
        return {
            "seg_logits": masks[-1],
            "aux_seg_logits": masks[:-1],
            "cls_logits": self.classifier(stages[-1]),
        }
