"""Patient-free checks of the external-source boundary and release adapter."""
from __future__ import annotations

import torch
from torch import nn
import pytest

from binary_study.models import emcad_external, emcad_joint


def test_external_source_is_not_downloaded_or_imported_without_opt_in(monkeypatch):
    monkeypatch.delenv("EMCAD_SOURCE_DIR", raising=False)
    with pytest.raises(RuntimeError, match="EMCAD_SOURCE_DIR"):
        emcad_external.pvt_v2_b0()


def test_incorrect_external_source_fails_before_execution(tmp_path, monkeypatch):
    library = tmp_path / "lib"
    library.mkdir()
    for filename in ("pvtv2.py", "decoders.py"):
        (library / filename).write_text("raise AssertionError('must never execute')\n")
    monkeypatch.setenv("EMCAD_SOURCE_DIR", str(tmp_path))
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        emcad_external.pvt_v2_b0()


class _Patch(nn.Module):
    def __init__(self, incoming, outgoing):
        super().__init__()
        self.proj = nn.Conv2d(incoming, outgoing, 3, stride=2, padding=1)

    def forward(self, image):
        feature = self.proj(image)
        return feature.flatten(2).transpose(1, 2), *feature.shape[-2:]


class _Block(nn.Module):
    def forward(self, sequence, rows, columns):
        assert sequence.shape[1] == rows * columns
        return sequence.tanh()


class _Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        previous = 3
        for index, width in enumerate((32, 64, 160, 256), start=1):
            setattr(self, f"patch_embed{index}", _Patch(previous, width))
            setattr(self, f"block{index}", nn.ModuleList([_Block()]))
            setattr(self, f"norm{index}", nn.LayerNorm(width))
            previous = width


class _Decoder(nn.Module):
    def forward(self, deepest, skips):
        return [deepest, *skips]


def test_release_adapter_shapes_checkpoint_gradients_and_registered_names(monkeypatch):
    monkeypatch.setattr(emcad_joint, "pvt_v2_b0", _Encoder)
    monkeypatch.setattr(emcad_joint, "build_emcad_decoder", lambda **kwargs: _Decoder())
    model = emcad_joint.EMCADJoint(pretrained=False, image_size=32).train()
    image = torch.rand(2, 3, 32, 32, requires_grad=True)
    result = model(image)
    assert result["seg_logits"].shape == (2, 1, 32, 32)
    assert result["cls_logits"].shape == (2, 2)
    assert len(result["aux_seg_logits"]) == 3
    assert all(mask.shape == (2, 1, 32, 32) for mask in result["aux_seg_logits"])
    assert all(key in model._modules for key in ("backbone", "decoder", "seg_heads", "classifier"))
    assert set(model._buffers) == {"input_mean", "input_std"}
    loss = result["seg_logits"].square().mean() + result["cls_logits"].square().mean()
    loss.backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()
    assert torch.count_nonzero(image.grad) > 0
    with pytest.raises(ValueError, match="spatial size"):
        model(image[..., :-1])
