"""Native YOLO26s instance segmentation + shared-neck clinical classifier.

The ROI detector has ONE object class for every patient, including controls.
Clinical normal/abnormal labels supervise only the separate two-logit head.
The segmentation loss is Ultralytics' native end-to-end instance objective with
raster PNG masks, not a replacement semantic decoder. See vendor provenance.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
from torch import nn
from torch.nn import functional as F


YOLO26S_WEIGHTS_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-seg.pt"
YOLO26S_WEIGHTS_SHA256 = "3da1d83e31caec96f9300eb4064f4f62882c133c7c264d63dfe61a7c197837a4"


def _verify_official_weights(path: Path) -> str:
    """Verify before any Ultralytics/PyTorch checkpoint deserialization."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != YOLO26S_WEIGHTS_SHA256:
        raise ValueError(f"YOLO26s-seg checkpoint SHA256 mismatch at {path}: expected "
                         f"{YOLO26S_WEIGHTS_SHA256}, found {actual}. The file was not deserialized.")
    return actual


def _get_verified_weights(path: Path) -> str:
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + "." + uuid4().hex + ".part")
        try:
            torch.hub.download_url_to_file(YOLO26S_WEIGHTS_URL, str(temporary), progress=True)
            _verify_official_weights(temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return _verify_official_weights(path)


class YOLO26Joint(nn.Module):
    """YOLO26s-seg native model plus global multi-scale neck classification.

    Forward receives RGB only. ``compute_native_loss`` receives targets after
    forward and returns a scalar native loss averaged over the batch. The common
    trainer MUST add clinical CE, and MUST NOT add its semantic BCE/Dice again.

    ``seg_logits`` is an inference adapter: max_i(object_score_i * mask_prob_i)
    inside each predicted box, with score floor0.001/max300 native one-to-one
    candidates. No GT, confidence tuning, mask threshold or morphology is used.
    It is deliberately detached: only the native objective trains the mask head.
    """
    native_objective = True
    required_ultralytics_version = "8.4.138"

    def __init__(self, image_size: int = 768, pretrained: bool = False, *,
                 weights_path: str | Path | None = None, confidence_floor: float = 0.001,
                 max_det: int = 300, mask_chunk_size: int = 8, dropout: float = 0.1,
                 total_epochs: int = 100):
        super().__init__()
        if image_size < 64 or image_size % 32:
            raise ValueError("YOLO26 image_size must be >=64 and divisible by 32")
        if not 0 <= confidence_floor < 1 or max_det < 1 or mask_chunk_size < 1:
            raise ValueError("Invalid instance retention/chunk configuration")
        try:
            import ultralytics
            from ultralytics.cfg import get_cfg
            from ultralytics.nn.tasks import SegmentationModel
        except ImportError as exc:
            raise ImportError("YOLO26Joint requires ultralytics==8.4.138 and its torch/torchvision dependencies") from exc
        if ultralytics.__version__ != self.required_ultralytics_version:
            raise RuntimeError(f"Native adapter audited for ultralytics=={self.required_ultralytics_version}; "
                               f"installed {ultralytics.__version__}. Re-audit before changing the pin.")
        self.image_size = image_size
        self.confidence_floor = float(confidence_floor)
        self.max_det = int(max_det)
        self.mask_chunk_size = int(mask_chunk_size)
        self._epoch = 0
        self._total_epochs = int(total_epochs)
        self.pretrained = bool(pretrained)
        self.pretraining_info = {"enabled": False, "source": None, "sha256": None}
        self.inference_policy = {"native_model": "yolo26s-seg", "native_branch": "one2one",
                                 "native_roi_class_count": 1, "clinical_class_count": 2,
                                 "confidence_floor": self.confidence_floor, "max_det": self.max_det,
                                 "semantic_adapter": "max_object_score_times_instance_mask_probability",
                                 "mask_interpolation": "bilinear_probability_then_predicted_box_crop",
                                 "nms": False, "morphology": False, "minimum_component_area": None}
        self.native_model = SegmentationModel("yolo26s-seg.yaml", ch=3, nc=1, verbose=False)
        # Defaults are explicit package-pinned native loss gains/assigner policy.
        self.native_model.args = get_cfg(overrides={"task": "segment", "imgsz": image_size,
                                                     "epochs": total_epochs, "overlap_mask": False})
        self.native_model.names = {0: "roi"}
        head = self.native_model.model[-1]
        if head.__class__.__name__ != "Segment26" or not head.end2end or head.nc != 1:
            raise RuntimeError("Expected native Segment26 end-to-end head with one ROI class")
        head.max_det = self.max_det
        if pretrained or weights_path is not None:
            from ultralytics import YOLO
            location = (Path(weights_path) if weights_path else
                        Path(__file__).parent / "vendor" / "yolo26_sources" / "weights" / "yolo26s-seg.pt")
            verified_sha256 = _get_verified_weights(location)
            self.pretraining_info = {"enabled": True, "source": YOLO26S_WEIGHTS_URL,
                                     "sha256": verified_sha256, "path": str(location.resolve()),
                                     "bytes": location.stat().st_size, "verified_before_deserialization": True}
            reference = YOLO(str(location), task="segment").model
            # Load only shape-compatible native weights; the nc=1 ROI output rows
            # are new. Do not reuse the COCO class score as a disease probability.
            self.native_model.load(reference, verbose=False)
            del reference
        channels = [branch[0].conv.in_channels for branch in head.cv2]
        self.neck_channels = tuple(channels)
        self.classifier = nn.Sequential(nn.LayerNorm(sum(channels)), nn.Linear(sum(channels), 256),
                                         nn.GELU(), nn.Dropout(dropout), nn.Linear(256, 2))
        self.last_native_loss_items: dict[str, float] = {}

    @property
    def head(self):
        return self.native_model.model[-1]

    def set_epoch(self, epoch: int, total_epochs: int | None = None):
        """Zero-based epoch, used to synchronize native progressive loss weights."""
        self._epoch = int(epoch)
        if total_epochs is not None:
            self._total_epochs = int(total_epochs)
        self.native_model.args.epochs = self._total_epochs

    @staticmethod
    def _raw_predictions(native_output):
        if isinstance(native_output, dict):
            raw = native_output
        elif isinstance(native_output, tuple) and len(native_output) == 2:
            raw = native_output[1]
        else:
            raise RuntimeError("Unexpected pinned Ultralytics segmentation forward result")
        if not isinstance(raw, dict) or not {"one2many", "one2one"} <= raw.keys():
            raise RuntimeError("Native end-to-end training branches are missing (do not fuse this model)")
        return raw

    @torch.no_grad()
    def _semantic_from_instances(self, raw: dict, spatial_size: tuple[int, int]):
        branch = raw["one2one"]
        # Decode the same native one-to-one box/class/mask predictions in both
        # train/eval mode. No NMS or alternate semantic-only branch replaces it.
        decoded = self.head._inference(branch).permute(0, 2, 1)
        candidates = self.head.postprocess(decoded)
        proto = branch["proto"]
        if isinstance(proto, tuple):
            proto = proto[0]  # auxiliary native semantic loss stays in raw.
        h, w = spatial_size
        output = []
        metadata = []
        for index, detections in enumerate(candidates):
            keep = detections[:, 4] >= self.confidence_floor
            detections = detections[keep]
            boxes = detections[:, :4].float().clone()
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h)
            scores = detections[:, 4].float()
            coefficients = detections[:, 6:].float()
            probability = proto.new_zeros((h, w), dtype=torch.float32)
            rows = torch.arange(h, device=proto.device, dtype=torch.float32)[None, :, None]
            cols = torch.arange(w, device=proto.device, dtype=torch.float32)[None, None, :]
            for start in range(0, len(detections), self.mask_chunk_size):
                stop = start + self.mask_chunk_size
                with torch.autocast(device_type=proto.device.type, enabled=False):
                    logits = coefficients[start:stop] @ proto[index].float().flatten(1)
                    logits = logits.reshape(-1, *proto.shape[-2:])
                    # Interpolate probabilities, then crop using predicted native
                    # boxes. Float32 reconstruction is kept until shared threshold.
                    masks = F.interpolate(logits.sigmoid().unsqueeze(1), (h, w), mode="bilinear",
                                          align_corners=False).squeeze(1)
                box = boxes[start:stop]
                inside = ((cols >= box[:, 0, None, None]) & (cols < box[:, 2, None, None]) &
                          (rows >= box[:, 1, None, None]) & (rows < box[:, 3, None, None]))
                weighted = masks * inside * scores[start:stop, None, None]
                probability = torch.maximum(probability, weighted.amax(dim=0))
            output.append(torch.logit(probability.clamp(1e-6, 1 - 1e-6)))
            # Compact reconstruction metadata; evaluator may persist these
            # detached tensors in NPZ rather than hundreds of full-size masks.
            metadata.append({"boxes_xyxy": boxes.detach(), "object_scores": scores.detach(),
                             "mask_coefficients": coefficients.detach(), "prototypes": proto[index].detach(),
                             "confidence_floor": self.confidence_floor, "max_det": self.max_det,
                             "native_branch": "one2one", "seg_adapter": "max_score_times_mask_probability"})
        return torch.stack(output).unsqueeze(1), metadata

    def forward(self, image: torch.Tensor) -> dict[str, Any]:
        if image.ndim != 4 or image.shape[1] != 3 or image.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(f"Expected Bx3x{self.image_size}x{self.image_size}, got {tuple(image.shape)}")
        raw = self._raw_predictions(self.native_model(image))
        # one2many neck features retain the backbone gradient. The native
        # one2one detached branch is not used for the clinical classifier.
        neck = raw["one2many"]["feats"]
        pooled = torch.cat([feature.mean(dim=(-2, -1)) for feature in neck], dim=1)
        cls_logits = self.classifier(pooled)
        seg_logits, metadata = self._semantic_from_instances(raw, image.shape[-2:])
        return {"seg_logits": seg_logits, "cls_logits": cls_logits,
                "native": raw, "native_metadata": metadata}

    @staticmethod
    def _raster_native_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        image, masks = batch["image"], batch["mask"]
        if masks.ndim != 4 or masks.shape[1] != 1 or masks.shape[0] != image.shape[0]:
            raise ValueError("Native ROI masks must have shape Bx1xHxW")
        if masks.shape[-2:] != image.shape[-2:]:
            raise ValueError("Image and raster supervision must share the same grid")
        if not bool(((masks == 0) | (masks == 1)).all()):
            raise ValueError("Native raster mask targets must be binary {0,1}")
        b, _, h, w = masks.shape
        indexes, boxes, raster = [], [], []
        for index in range(b):
            foreground = masks[index, 0] > 0
            y, x = foreground.nonzero(as_tuple=True)
            if x.numel() == 0:
                # Empty masks can occur in synthetic checks. In real data the
                # validated all-positive contract is enforced by the loader.
                continue
            x0, x1, y0, y1 = x.min().float(), x.max().float() + 1, y.min().float(), y.max().float() + 1
            boxes.append(torch.stack(((x0 + x1) / (2 * w), (y0 + y1) / (2 * h),
                                       (x1 - x0) / w, (y1 - y0) / h)))
            indexes.append(index)
            raster.append(masks[index, 0].float())
        device = image.device
        return {"img": image, "batch_idx": torch.tensor(indexes, device=device, dtype=torch.long),
                "cls": torch.zeros((len(indexes), 1), device=device),
                "bboxes": torch.stack(boxes) if boxes else torch.empty((0, 4), device=device),
                "masks": torch.stack(raster) if raster else torch.empty((0, h, w), device=device),
                # All foreground belongs to native object class0, irrespective
                # of clinical normal/abnormal label. Background is gated by masks.
                "sem_masks": torch.zeros((b, h, w), device=device, dtype=torch.long)}

    def compute_native_loss(self, outputs: dict, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Native instance loss scalar; clinical CE is the common trainer's job."""
        if getattr(self.native_model, "criterion", None) is None:
            self.native_model.criterion = self.native_model.init_criterion()
        criterion = self.native_model.criterion
        expected_device = batch["image"].device
        if criterion.one2many.device != expected_device:
            self.native_model.criterion = self.native_model.init_criterion()
            criterion = self.native_model.criterion
        criterion.updates = self._epoch
        criterion.o2m = criterion.decay(self._epoch)
        criterion.o2o = criterion.total - criterion.o2m
        native_batch = self._raster_native_batch(batch)
        vector, items = criterion(outputs["native"], native_batch)
        # Package loss is a vector scaled by batch size; normalize explicitly so
        # the clinical CE coefficient does not silently depend on physical batch.
        loss = vector.sum() / batch["image"].shape[0]
        self.last_native_loss_items = {key: float(value.detach()) for key, value in items.items()}
        self.last_native_loss_items.update({"o2many_weight": criterion.o2m, "o2one_weight": criterion.o2o})
        return loss


__all__ = ["YOLO26Joint"]
