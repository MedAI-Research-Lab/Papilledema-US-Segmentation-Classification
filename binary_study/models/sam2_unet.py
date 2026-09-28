# Modified for the transorbital-ultrasound study and this source release.
# This is an adapted runtime subset, not an unmodified upstream archive.
# Upstream provenance and applicable licenses: THIRD_PARTY_NOTICES.md
# and licenses/ at the repository root. Original notices are retained.
"""Prompt-free author SAM2-UNet-Tiny adapter model with a binary class head."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .vendor.emcad_sam.hieradet import Hiera
from .vendor.emcad_sam.sam2_unet_blocks import Adapter, RFB_modified, Up
from .vendor.emcad_sam.weights import official_weight


class SAM2UNetJoint(nn.Module):
    """Uses Hiera-Tiny, not the full SAM prompt/video model.

    Frozen pretrained trunk plus trainable in-block adapters. Gradients flow
    through frozen blocks to adapters; the trunk is NOT wrapped in no_grad.
    pretrained controls weight acquisition only; freeze_encoder independently
    controls training so restoring a checkpoint does not change that policy.
    The loader's [0,1] inputs receive the author's documented ImageNet
    normalization inside this adapter, independently of weight loading.
    """
    def __init__(self, pretrained=True, image_size=768, gradient_checkpointing=True, freeze_encoder=True):
        super().__init__()
        self.image_size = int(image_size)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.freeze_encoder = bool(freeze_encoder)
        self.register_buffer("input_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("input_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        # Exact author sam2_hiera_t.yaml trunk settings, remaining Hiera
        # arguments retain the author's defaults.
        self.encoder = Hiera(embed_dim=96, num_heads=1, stages=(1, 2, 7, 2),
                             global_att_blocks=(5, 7, 9), window_pos_embed_bkg_spatial_size=(7, 7))
        self.pretrained_provenance = {"pretrained": False, "encoder_frozen": self.freeze_encoder}
        if pretrained:
            path, info = official_weight("sam2_unet")
            state = torch.load(path, map_location="cpu", weights_only=True)["model"]
            prefix = "image_encoder.trunk."
            trunk = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            if not trunk:
                raise RuntimeError("Official SAM2 checkpoint contains no image_encoder.trunk weights")
            self.encoder.load_state_dict(trunk, strict=True)
            self.pretrained_provenance = {**info, "pretrained": True, "encoder_frozen": self.freeze_encoder,
                                          "loaded": "complete original SAM2 Hiera-Tiny trunk", "loaded_keys": len(trunk)}
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(not self.freeze_encoder)
        # The author adapter wraps each original block; preserve its actual
        # pretrained qkv geometry, including downsampling transition blocks.
        self.encoder.blocks = nn.Sequential(*(Adapter(block) for block in self.encoder.blocks))
        self.rfbs = nn.ModuleList(RFB_modified(c, 64) for c in [96, 192, 384, 768])
        self.up1, self.up2, self.up3 = Up(128, 64), Up(128, 64), Up(128, 64)
        self.side1, self.side2, self.head = (nn.Conv2d(64, 1, 1) for _ in range(3))
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.1), nn.Linear(768, 2))

    def _encoder(self, images):
        # Author Hiera.forward with per-adapter-block checkpointing added.
        x = self.encoder.patch_embed(images)
        x = x + self.encoder._get_pos_embed(x.shape[1:3])
        features = []
        for index, block in enumerate(self.encoder.blocks):
            if self.training and self.gradient_checkpointing and torch.is_grad_enabled():
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
            if index in self.encoder.stage_ends:
                features.append(x.permute(0, 3, 1, 2))
        return features

    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("SAM2UNetJoint expects Bx3xHxW RGB, no prompts")
        if tuple(images.shape[-2:]) != (self.image_size, self.image_size):
            raise ValueError(f"Expected native {self.image_size}x{self.image_size}; implicit resizing is disabled")
        features = self._encoder((images - self.input_mean) / self.input_std)
        class_logits = self.classifier(features[-1])
        x1, x2, x3, x4 = [block(feature) for block, feature in zip(self.rfbs, features)]
        x = self.up1(x4, x3)
        aux1 = F.interpolate(self.side1(x), size=images.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up2(x, x2)
        aux2 = F.interpolate(self.side2(x), size=images.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up3(x, x1)
        segmentation = F.interpolate(self.head(x), size=images.shape[-2:], mode="bilinear", align_corners=False)
        return {"seg_logits": segmentation, "cls_logits": class_logits, "aux_seg_logits": [aux1, aux2]}
