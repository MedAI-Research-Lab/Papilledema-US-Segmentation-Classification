"""User Method2 ViT port: binary decoder + global/predicted-ROI image head.

Source provenance and design differences are described in THIRD_PARTY_NOTICES.md.
No pretrained backbone exists for that custom notebook architecture. ImageNet
normalization is therefore not silently added to the shared RGB [0,1] input.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class PatchExtractor(nn.Module):
    def __init__(self, patch_size: int = 16):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        p = self.patch_size
        if h % p or w % p:
            raise ValueError(f"Image dimensions {(h, w)} must divide patch_size={p}")
        patches = x.unfold(2, p, p).unfold(3, p, p).contiguous()
        return patches.view(b, c, -1, p, p).permute(0, 2, 1, 3, 4).reshape(b, -1, c * p * p)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, embed_dim), nn.Dropout(dropout))

    def forward(self, x):
        q = self.norm1(x)
        # This retains nn.MultiheadAttention parameters but permits PyTorch SDPA.
        x = x + self.attention(q, q, q, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class CrossAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value):
        q, kv = self.norm_q(query), self.norm_kv(key_value)
        return query + self.dropout(self.cross_attention(q, kv, kv, need_weights=False)[0])


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = (nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                       nn.BatchNorm2d(out_channels))
                         if in_channels != out_channels else nn.Identity())

    def forward(self, x):
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + self.shortcut(x))


class Decoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.blocks = nn.ModuleList([ResidualBlock(embed_dim, 128), ResidualBlock(128, 64),
                                     ResidualBlock(64, 32), ResidualBlock(32, 16)])
        self.final_conv = nn.Conv2d(16, 1, 1)

    def forward(self, tokens, grid):
        b, _, c = tokens.shape
        x = tokens.transpose(1, 2).reshape(b, c, *grid)
        for block in self.blocks:
            x = F.interpolate(block(x), scale_factor=2, mode="bilinear", align_corners=False)
        return self.final_conv(x)


class ViTMethod2Joint(nn.Module):
    """Original scratch ViT with a differentiable predicted-ROI classifier.

    ``pretrained=True`` requires an explicit compatible ``weights_path``: it never
    substitutes an unrelated pretrained ViT. The default preserves notebook
    width/depth/heads and patch16; small dimensions can be used for smoke tests.
    """
    native_objective = False

    def __init__(self, image_size: int = 768, pretrained: bool = False, *,
                 weights_path: str | Path | None = None, embed_dim: int = 256,
                 depth: int = 6, num_heads: int = 8, dropout: float = 0.1,
                 gradient_checkpointing: bool = True, detach_roi: bool = False):
        super().__init__()
        if image_size < 32 or image_size % 16:
            raise ValueError("ViT Method2 image_size must be >=32 and divisible by 16")
        if pretrained and weights_path is None:
            raise ValueError("The supplied custom ViT has no external pretrained weights; use pretrained=False "
                             "or provide an explicitly compatible weights_path.")
        self.image_size = image_size
        self.patch_size = 16
        self.gradient_checkpointing = gradient_checkpointing
        self.detach_roi = detach_roi
        self.patch_extractor = PatchExtractor(16)
        self.patch_embedding = nn.Linear(3 * 16 * 16, embed_dim)
        self.position_embedding = nn.Parameter(torch.randn(1, (image_size // 16) ** 2, embed_dim))
        self.encoder = nn.ModuleList([TransformerEncoderBlock(embed_dim, num_heads, dropout=dropout)
                                      for _ in range(depth)])
        self.cross_attention = CrossAttention(embed_dim, num_heads, dropout)
        self.decoder = Decoder(embed_dim)
        self.classifier = nn.Sequential(nn.LayerNorm(2 * embed_dim), nn.Linear(2 * embed_dim, embed_dim),
                                         nn.GELU(), nn.Dropout(dropout), nn.Linear(embed_dim, 2))
        self.pretrained = bool(pretrained)
        if weights_path is not None:
            state = torch.load(Path(weights_path), map_location="cpu", weights_only=True)
            self.load_state_dict(state.get("model", state), strict=True)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        if image.ndim != 4 or image.shape[1] != 3 or image.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(f"Expected Bx3x{self.image_size}x{self.image_size}, got {tuple(image.shape)}")
        embeddings = self.patch_embedding(self.patch_extractor(image)) + self.position_embedding
        tokens = embeddings
        for layer in self.encoder:
            if self.training and self.gradient_checkpointing and torch.is_grad_enabled():
                tokens = checkpoint(layer, tokens, use_reentrant=False, preserve_rng_state=True)
            else:
                tokens = layer(tokens)
        if self.training and self.gradient_checkpointing and torch.is_grad_enabled():
            fused = checkpoint(self.cross_attention, tokens, embeddings, use_reentrant=False, preserve_rng_state=True)
        else:
            fused = self.cross_attention(tokens, embeddings)
        grid = (self.image_size // 16, self.image_size // 16)
        seg_logits = self.decoder(fused, grid)
        # Loss and aggregation remain numerically stable under autocast.
        roi = seg_logits.float().sigmoid()
        if self.detach_roi:
            roi = roi.detach()
        weights = F.adaptive_avg_pool2d(roi, grid).flatten(2).transpose(1, 2)
        denominator = weights.sum(dim=1).clamp_min(1e-6)
        global_feature = fused.float().mean(dim=1)
        roi_feature = (fused.float() * weights).sum(dim=1) / denominator
        # In the limiting empty-prediction case use the global representation.
        roi_feature = torch.where(weights.sum(dim=1) > 1e-6, roi_feature, global_feature)
        cls_logits = self.classifier(torch.cat((global_feature, roi_feature), dim=1).to(fused.dtype))
        return {"seg_logits": seg_logits, "cls_logits": cls_logits}


__all__ = ["ViTMethod2Joint"]
