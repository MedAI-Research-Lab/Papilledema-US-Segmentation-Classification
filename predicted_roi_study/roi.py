"""Strict predicted-ROI post-processing and leakage-free crop construction.

The classifier-facing mask in this module is *always* a hard, predicted mask.
Ground-truth masks are accepted only by :meth:`ROIPolicy.fit_from_gt`, which is
intended to be called on the training partition to establish anatomically
plausible area limits.  Validation/test ground truth must never enter
``postprocess_probability`` or ``extract_roi_tensor``.

All component analysis is 8-connected.  Bounding boxes use the half-open
``(x0, y0, x1, y1)`` convention.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from scipy import ndimage
from torch import Tensor
from torch.nn import functional as F


EIGHT_CONNECTED = np.ones((3, 3), dtype=np.uint8)
DEFAULT_NEUTRAL_RGB = (0.485, 0.456, 0.406)


class ROIStatus(str, Enum):
    """Mutually exclusive frame-level usability decisions."""

    VALID = "valid"
    EMPTY = "empty"
    TINY = "tiny"
    OVERSIZE = "oversize"
    BORDER = "border"
    MULTI_AMBIGUOUS = "multi_ambiguous"


@dataclass(frozen=True)
class ROIComponent:
    """Diagnostics for one 8-connected component."""

    label: int
    area_pixels: int
    probability_mass: float
    mean_probability: float
    bbox_xyxy: tuple[int, int, int, int]
    touches_border: bool


@dataclass(frozen=True)
class ROIResult:
    """Post-processing result for one predicted probability map.

    ``mask`` contains exactly one component only when ``valid`` is true.  It is
    all-false for every abstention status, which prevents accidental use of an
    invalid component by the classifier.  ``hard_mask`` retains the complete
    thresholded/morphologically processed raster for auditing.
    """

    status: ROIStatus
    mask: np.ndarray
    hard_mask: np.ndarray
    components: tuple[ROIComponent, ...]
    selected_component: ROIComponent | None
    dominance_ratio_observed: float | None = None

    @property
    def valid(self) -> bool:
        return self.status is ROIStatus.VALID

    @property
    def bbox_xyxy(self) -> tuple[int, int, int, int] | None:
        return self.selected_component.bbox_xyxy if self.selected_component else None

    @property
    def area_pixels(self) -> int:
        return self.selected_component.area_pixels if self.selected_component else 0

    def to_record(self) -> dict[str, Any]:
        """Return compact JSON-compatible diagnostics (raster data excluded)."""

        return {
            "status": self.status.value,
            "valid": self.valid,
            "component_count": len(self.components),
            "selected_area_pixels": self.area_pixels,
            "selected_bbox_xyxy": list(self.bbox_xyxy) if self.bbox_xyxy else None,
            "dominance_ratio_observed": self.dominance_ratio_observed,
            "components": [asdict(component) for component in self.components],
        }


def _as_2d_numpy(value: Any, *, name: str) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D raster (singleton leading axes are allowed)")
    return array


def _iter_masks(masks: Any) -> Iterable[np.ndarray]:
    if torch.is_tensor(masks):
        masks = masks.detach().cpu().numpy()
    if isinstance(masks, np.ndarray):
        if masks.ndim == 2:
            yield masks
            return
        if masks.ndim == 3:
            for mask in masks:
                yield _as_2d_numpy(mask, name="ground-truth mask")
            return
        if masks.ndim == 4 and masks.shape[1] == 1:
            for mask in masks:
                yield _as_2d_numpy(mask, name="ground-truth mask")
            return
        raise ValueError("Ground-truth masks must have shape HxW, NxHxW, or Nx1xHxW")
    for mask in masks:
        yield _as_2d_numpy(mask, name="ground-truth mask")


def _largest_component_area(mask: np.ndarray) -> int:
    labels, count = ndimage.label(mask, structure=EIGHT_CONNECTED)
    if count == 0:
        return 0
    return int(np.bincount(labels.ravel(), minlength=count + 1)[1:].max())


@dataclass(frozen=True)
class ROIPolicy:
    """Frozen single-ROI acceptance policy.

    Area bounds are derived from the largest 8-connected component of each
    non-empty *training* GT mask.  At inference, a component with
    ``area < min_area_pixels`` is noise and a component with
    ``area > max_area_pixels`` is over-segmentation.

    ``dominance_ratio`` compares mean predicted probability of the top two
    plausible components.  Probability mass is only a deterministic tie-break.
    Multiple plausible components are accepted only when the top component is
    sufficiently dominant.  This avoids silently joining spatially separate
    predictions into one classifier crop.
    """

    min_area_pixels: float
    max_area_pixels: float
    image_shape: tuple[int, int]
    threshold: float = 0.5
    dominance_ratio: float = 1.5
    dominance_margin: float = 0.0
    reject_border: bool = True
    border_margin_pixels: int = 0
    closing_iterations: int = 0
    max_hole_area_pixels: int = 0
    quantile_low: float = 0.01
    quantile_high: float = 0.99
    lower_scale: float = 0.5
    upper_scale: float = 1.5
    fitted_sample_count: int = 0
    fitted_q_low_pixels: float | None = None
    fitted_q_high_pixels: float | None = None
    provenance: str = "train_gt_largest_component_8_connected"

    def __post_init__(self) -> None:
        h, w = self.image_shape
        if h < 1 or w < 1:
            raise ValueError("image_shape must contain positive H,W")
        if not (0 < self.min_area_pixels <= self.max_area_pixels <= h * w):
            raise ValueError("Area bounds must satisfy 0 < minimum <= maximum <= image area")
        if not 0 < self.threshold < 1:
            raise ValueError("threshold must lie strictly between 0 and 1")
        if self.dominance_ratio < 1 or self.dominance_margin < 0:
            raise ValueError("dominance_ratio must be >=1 and dominance_margin >=0")
        if self.border_margin_pixels < 0 or self.closing_iterations < 0 or self.max_hole_area_pixels < 0:
            raise ValueError("Morphology and border parameters must be non-negative")
        if not 0 <= self.quantile_low < self.quantile_high <= 1:
            raise ValueError("Expected 0 <= quantile_low < quantile_high <= 1")
        if self.lower_scale <= 0 or self.upper_scale <= 0:
            raise ValueError("Area scale factors must be positive")

    @property
    def min_area_fraction(self) -> float:
        return float(self.min_area_pixels / np.prod(self.image_shape))

    @property
    def max_area_fraction(self) -> float:
        return float(self.max_area_pixels / np.prod(self.image_shape))

    @classmethod
    def fit_from_gt(
        cls,
        train_masks: Any,
        *,
        threshold: float = 0.5,
        quantile_low: float = 0.01,
        quantile_high: float = 0.99,
        lower_scale: float = 0.5,
        upper_scale: float = 1.5,
        **policy_kwargs: Any,
    ) -> "ROIPolicy":
        """Fit area bounds from training GT only.

        The caller is responsible for passing only the training partition.  An
        empty GT is rejected instead of being silently omitted, making split or
        annotation mistakes visible.  Tiny disconnected annotation specks do
        not inflate area because only the largest 8-connected GT component is
        measured.
        """

        prepared: list[np.ndarray] = []
        shape: tuple[int, int] | None = None
        for raw in _iter_masks(train_masks):
            if not np.isfinite(raw).all():
                raise ValueError("Ground-truth masks must be finite")
            binary = raw.astype(bool)
            if shape is None:
                shape = tuple(int(v) for v in binary.shape)
            elif binary.shape != shape:
                raise ValueError("All ground-truth masks must share one image shape")
            if not binary.any():
                raise ValueError("Training GT contains an empty ROI")
            prepared.append(binary)
        if not prepared or shape is None:
            raise ValueError("At least one training GT mask is required")
        areas = np.asarray([_largest_component_area(mask) for mask in prepared], dtype=np.float64)
        q_low = float(np.quantile(areas, quantile_low, method="linear"))
        q_high = float(np.quantile(areas, quantile_high, method="linear"))
        minimum = max(1.0, float(lower_scale * q_low))
        maximum = min(float(np.prod(shape)), float(upper_scale * q_high))
        if maximum < minimum:
            raise ValueError("Fitted area bounds are inconsistent")
        return cls(
            min_area_pixels=minimum,
            max_area_pixels=maximum,
            image_shape=shape,
            threshold=threshold,
            quantile_low=quantile_low,
            quantile_high=quantile_high,
            lower_scale=lower_scale,
            upper_scale=upper_scale,
            fitted_sample_count=len(prepared),
            fitted_q_low_pixels=q_low,
            fitted_q_high_pixels=q_high,
            **policy_kwargs,
        )

    def with_threshold(self, threshold: float) -> "ROIPolicy":
        """Return a copy after the segmentation threshold is locked on validation."""

        return replace(self, threshold=float(threshold))

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["image_shape"] = list(self.image_shape)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ROIPolicy":
        payload = dict(value)
        payload["image_shape"] = tuple(int(v) for v in payload["image_shape"])
        return cls(**payload)

    def postprocess_probability(self, probability: Any) -> ROIResult:
        return postprocess_probability(probability, self)


def _fill_small_holes(mask: np.ndarray, maximum_area: int) -> np.ndarray:
    if maximum_area <= 0 or not mask.any():
        return mask
    background_labels, count = ndimage.label(~mask, structure=EIGHT_CONNECTED)
    if count == 0:
        return mask
    border_ids = np.unique(
        np.concatenate(
            (background_labels[0], background_labels[-1], background_labels[:, 0], background_labels[:, -1])
        )
    )
    sizes = np.bincount(background_labels.ravel(), minlength=count + 1)
    fill = np.zeros(count + 1, dtype=bool)
    candidate_ids = np.arange(1, count + 1)
    fill[candidate_ids] = sizes[candidate_ids] <= maximum_area
    fill[border_ids] = False
    return mask | fill[background_labels]


def _component_touches_border(component: np.ndarray, margin: int) -> bool:
    h, w = component.shape
    y, x = np.nonzero(component)
    if not len(y):
        return False
    return bool(y.min() <= margin or x.min() <= margin or y.max() >= h - 1 - margin or x.max() >= w - 1 - margin)


def _empty_result(status: ROIStatus, hard_mask: np.ndarray, components: Sequence[ROIComponent],
                  dominance: float | None = None) -> ROIResult:
    return ROIResult(
        status=status,
        mask=np.zeros_like(hard_mask, dtype=bool),
        hard_mask=hard_mask,
        components=tuple(components),
        selected_component=None,
        dominance_ratio_observed=dominance,
    )


def hard_mask_from_probability(probability: Any, policy: ROIPolicy) -> np.ndarray:
    """Apply the policy's locked threshold and morphology without quality gates."""

    score_map = _as_2d_numpy(probability, name="probability").astype(np.float32, copy=False)
    if score_map.shape != policy.image_shape:
        raise ValueError(f"Probability shape {score_map.shape} does not match policy {policy.image_shape}")
    if not np.isfinite(score_map).all() or np.any(score_map < 0) or np.any(score_map > 1):
        raise ValueError("Probability map must be finite and bounded in [0,1]")
    hard = score_map >= policy.threshold
    if policy.closing_iterations:
        hard = ndimage.binary_closing(
            hard, structure=EIGHT_CONNECTED, iterations=policy.closing_iterations, border_value=0
        )
    return _fill_small_holes(hard, policy.max_hole_area_pixels).astype(bool, copy=False)


def postprocess_probability(probability: Any, policy: ROIPolicy) -> ROIResult:
    """Convert one predicted probability map to zero or one accepted ROI.

    Decision precedence is ``empty -> tiny -> oversize -> border ->
    multi_ambiguous/valid``.  Tiny specks may coexist with and are removed from
    one valid component.  In contrast, an oversized or plausible border-touching
    component invalidates the entire frame; it is never silently discarded to
    rescue a smaller component.
    """

    score_map = _as_2d_numpy(probability, name="probability").astype(np.float32, copy=False)
    hard = hard_mask_from_probability(score_map, policy)
    labels, count = ndimage.label(hard, structure=EIGHT_CONNECTED)
    if count == 0:
        return _empty_result(ROIStatus.EMPTY, hard, ())

    components: list[ROIComponent] = []
    component_masks: dict[int, np.ndarray] = {}
    for label_id in range(1, count + 1):
        component = labels == label_id
        component_masks[label_id] = component
        y, x = np.nonzero(component)
        area = int(len(y))
        values = score_map[component]
        components.append(
            ROIComponent(
                label=label_id,
                area_pixels=area,
                probability_mass=float(values.astype(np.float64).sum()),
                mean_probability=float(values.astype(np.float64).mean()),
                bbox_xyxy=(int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1),
                touches_border=_component_touches_border(component, policy.border_margin_pixels),
            )
        )

    non_tiny = [component for component in components if component.area_pixels >= policy.min_area_pixels]
    if not non_tiny:
        return _empty_result(ROIStatus.TINY, hard, components)
    if any(component.area_pixels > policy.max_area_pixels for component in non_tiny):
        return _empty_result(ROIStatus.OVERSIZE, hard, components)
    plausible = [component for component in non_tiny if component.area_pixels <= policy.max_area_pixels]
    if policy.reject_border and any(component.touches_border for component in plausible):
        return _empty_result(ROIStatus.BORDER, hard, components)

    ranked = sorted(
        plausible,
        key=lambda item: (-item.mean_probability, -item.probability_mass, -item.area_pixels, item.label),
    )
    observed: float | None = None
    if len(ranked) > 1:
        first, second = ranked[:2]
        observed = float(first.mean_probability / max(second.mean_probability, np.finfo(np.float64).eps))
        if observed < policy.dominance_ratio or first.mean_probability - second.mean_probability < policy.dominance_margin:
            return _empty_result(ROIStatus.MULTI_AMBIGUOUS, hard, components, observed)
    selected = ranked[0]
    selected_mask = component_masks[selected.label].astype(bool, copy=True)
    return ROIResult(
        status=ROIStatus.VALID,
        mask=selected_mask,
        hard_mask=hard,
        components=tuple(components),
        selected_component=selected,
        dominance_ratio_observed=observed,
    )


@dataclass(frozen=True)
class ROIExtraction:
    """Classifier tensor plus auditable crop/letterbox metadata."""

    tensor: Tensor
    output_mask: Tensor
    bbox_xyxy: tuple[int, int, int, int]
    resized_hw: tuple[int, int]
    placement_xyxy: tuple[int, int, int, int]
    geometry: Tensor


def _target_hw(target_size: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(target_size, int):
        target = (target_size, target_size)
    else:
        if len(target_size) != 2:
            raise ValueError("target_size must be an integer or (H,W)")
        target = (int(target_size[0]), int(target_size[1]))
    if min(target) < 1:
        raise ValueError("target_size values must be positive")
    return target


def _neutral_tensor(image: Tensor, neutral: float | Sequence[float] | Tensor) -> Tensor:
    value = torch.as_tensor(neutral, dtype=image.dtype, device=image.device).flatten()
    if value.numel() == 1:
        value = value.repeat(image.shape[0])
    if value.numel() != image.shape[0] or not bool(torch.isfinite(value).all()):
        raise ValueError("neutral must be one finite value or one value per image channel")
    return value.view(-1, 1, 1)


def roi_geometry(mask: Any) -> np.ndarray:
    """Return four mask-only features: area, bbox-W, bbox-H fractions, log aspect."""

    binary = _as_2d_numpy(mask, name="mask").astype(bool)
    if not binary.any():
        raise ValueError("Cannot compute geometry for an empty ROI")
    h, w = binary.shape
    y, x = np.nonzero(binary)
    box_w, box_h = int(x.max() - x.min() + 1), int(y.max() - y.min() + 1)
    return np.asarray(
        [binary.mean(), box_w / w, box_h / h, np.log(box_w / box_h)], dtype=np.float32
    )


def extract_roi_tensor(
    image: Tensor,
    mask: Any,
    target_size: int | Sequence[int] = 224,
    neutral: float | Sequence[float] | Tensor = DEFAULT_NEUTRAL_RGB,
    *,
    return_metadata: bool = False,
) -> Tensor | ROIExtraction:
    """Create a deterministic, strict ROI-only classifier input.

    The original-resolution image is neutralized *before* the tight crop is
    taken.  Consequently, neither letterbox resizing nor interpolation can
    sample an original pixel outside ``mask``.  The crop is aspect-preserving,
    centered on a fixed neutral canvas, and re-masked after interpolation.

    ``image`` must be floating point ``CxHxW``.  The mask must be one non-empty
    binary component; use :func:`postprocess_probability` first.
    """

    if not torch.is_tensor(image) or image.ndim != 3 or not image.is_floating_point():
        raise ValueError("image must be a floating-point CxHxW torch tensor")
    binary_np = _as_2d_numpy(mask, name="mask")
    if binary_np.shape != tuple(image.shape[-2:]):
        raise ValueError("Image and mask must share H,W")
    if not np.isfinite(binary_np).all() or not np.all((binary_np == 0) | (binary_np == 1)):
        raise ValueError("mask must be a finite hard binary raster")
    binary_np = binary_np.astype(bool, copy=False)
    if not binary_np.any():
        raise ValueError("Cannot extract an empty or abstained ROI")
    labels, count = ndimage.label(binary_np, structure=EIGHT_CONNECTED)
    if count != 1:
        raise ValueError("Classifier mask must contain exactly one 8-connected component")

    neutral_value = _neutral_tensor(image, neutral)
    mask_tensor = torch.as_tensor(binary_np, dtype=torch.bool, device=image.device)
    # Neutralization precedes cropping: this ordering is the core leakage guard.
    strict_image = torch.where(mask_tensor.unsqueeze(0), image, neutral_value)
    y, x = np.nonzero(binary_np)
    x0, x1 = int(x.min()), int(x.max()) + 1
    y0, y1 = int(y.min()), int(y.max()) + 1
    cropped = strict_image[:, y0:y1, x0:x1]
    cropped_mask = mask_tensor[y0:y1, x0:x1]

    target_h, target_w = _target_hw(target_size)
    crop_h, crop_w = cropped.shape[-2:]
    scale = min(target_h / crop_h, target_w / crop_w)
    resized_h = max(1, min(target_h, int(round(crop_h * scale))))
    resized_w = max(1, min(target_w, int(round(crop_w * scale))))
    resized = F.interpolate(
        cropped.unsqueeze(0), size=(resized_h, resized_w), mode="bilinear",
        align_corners=False, antialias=True,
    ).squeeze(0)
    resized_mask = F.interpolate(
        cropped_mask[None, None].to(dtype=torch.float32),
        size=(resized_h, resized_w), mode="nearest",
    )[0, 0].bool()
    # A second hard gate makes neutral padding/boundaries exact after resizing.
    resized = torch.where(resized_mask.unsqueeze(0), resized, neutral_value)

    canvas = neutral_value.expand(-1, target_h, target_w).clone()
    output_mask = torch.zeros((target_h, target_w), dtype=torch.bool, device=image.device)
    top = (target_h - resized_h) // 2
    left = (target_w - resized_w) // 2
    canvas[:, top:top + resized_h, left:left + resized_w] = resized
    output_mask[top:top + resized_h, left:left + resized_w] = resized_mask
    if not return_metadata:
        return canvas
    geometry = torch.as_tensor(roi_geometry(binary_np), dtype=image.dtype, device=image.device)
    return ROIExtraction(
        tensor=canvas,
        output_mask=output_mask,
        bbox_xyxy=(x0, y0, x1, y1),
        resized_hw=(resized_h, resized_w),
        placement_xyxy=(left, top, left + resized_w, top + resized_h),
        geometry=geometry,
    )


__all__ = [
    "DEFAULT_NEUTRAL_RGB",
    "EIGHT_CONNECTED",
    "ROIComponent",
    "ROIExtraction",
    "ROIPolicy",
    "ROIResult",
    "ROIStatus",
    "extract_roi_tensor",
    "hard_mask_from_probability",
    "postprocess_probability",
    "roi_geometry",
]
