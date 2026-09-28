"""Metrics for the strict predicted-ROI diagnostic pipeline.

The deployable unit is an eye (exactly seven frames).  A frame has a
classification probability only when its *predicted* ROI passes the locked ROI
quality gate.  An eye is evaluable when at least ``min_valid_frames`` frames are
valid; otherwise it abstains.  A patient is evaluable only when both eyes are
evaluable.  There is deliberately no whole-image or single-eye fallback.

Ground-truth ROI overlap columns may be supplied for retrospective localisation
audits, but they never participate in aggregation, probability calculation, or
the abstention decision.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import ndimage, optimize, special, stats
from sklearn import metrics as skm


ABSTAIN = -1
CORE_ABLATION_ARMS = (
    "predicted_roi",
    "gt_oracle",
    "whole_image",
    "mask_only",
    "background_only",
    "roi_shuffle",
)


def _strict_boolean_array(values: pd.Series, *, name: str) -> np.ndarray:
    """Reject text/numeric truthiness in provenance-bearing metric tables."""

    if values.isna().any():
        raise ValueError(f"{name} must not contain missing values")
    observed = values.to_numpy(dtype=object, copy=False)
    if any(not isinstance(value, (bool, np.bool_)) for value in observed):
        raise ValueError(f"{name} must contain only real boolean values")
    return np.asarray([bool(value) for value in observed], dtype=bool)


def ratio(numerator: float, denominator: float, empty: float = np.nan) -> float:
    """Safe ratio with an explicit value for a zero denominator."""
    return float(numerator / denominator) if denominator else float(empty)


def clean_json(value):
    """Convert NumPy/pandas values to strict-JSON-compatible Python values."""
    if isinstance(value, Mapping):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if pd.isna(value) if not isinstance(value, str) else False:
        return None
    return value


def _binary_labels(y_true: Sequence[int]) -> np.ndarray:
    y = np.asarray(y_true, dtype=int).reshape(-1)
    if y.size == 0:
        raise ValueError("At least one labelled observation is required.")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Binary labels must contain only 0 and 1.")
    return y


def _finite_probabilities(probability: Sequence[float]) -> np.ndarray:
    p = np.asarray(probability, dtype=float).reshape(-1)
    if not np.isfinite(p).all() or not ((0 <= p) & (p <= 1)).all():
        raise ValueError("Probabilities must be finite and in [0, 1].")
    return p


def segmentation_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    *,
    roi_iou: float = 0.5,
    distances: bool = True,
    surface_tolerance_px: float = 2.0,
) -> dict:
    """Evaluate one binary raster while retaining empty predictions as failures.

    Surface distances are pooled symmetric distances in processed-image pixels.
    They are undefined for an empty prediction and are therefore returned as
    ``NaN`` together with ``distance_valid=0``.
    """
    pred, gt = np.asarray(prediction, bool), np.asarray(ground_truth, bool)
    if pred.shape != gt.shape or pred.ndim != 2:
        raise ValueError("Prediction and ground truth must be matching 2-D rasters.")
    if not gt.any():
        raise ValueError("The agreed cohort contains one non-empty GT ROI per frame.")
    if not 0 <= roi_iou <= 1:
        raise ValueError("roi_iou must be in [0, 1].")

    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    tn = int((~pred & ~gt).sum())
    pred_area, gt_area = tp + fp, tp + fn
    components, n_components = ndimage.label(pred, structure=np.ones((3, 3)))
    sizes = np.bincount(components.ravel(), minlength=n_components + 1)[1:]
    intersections = np.bincount(components[gt].ravel(), minlength=n_components + 1)[1:]
    ious = intersections / np.maximum(sizes + gt_area - intersections, 1)
    roi_hit = int(bool(n_components) and float(ious.max()) >= roi_iou)

    result = {
        "dice": ratio(2 * tp, 2 * tp + fp + fn, 0),
        "iou": ratio(tp, tp + fp + fn, 0),
        "seg_precision": ratio(tp, pred_area),
        "seg_recall": ratio(tp, gt_area),
        "pixel_precision": ratio(tp, pred_area),
        "pixel_sensitivity": ratio(tp, gt_area),
        "pixel_specificity": ratio(tn, tn + fp),
        "pixel_tp": tp,
        "pixel_fp": fp,
        "pixel_fn": fn,
        "pixel_tn": tn,
        "pixel_fpr": ratio(fp, fp + tn),
        "pixel_fdr": ratio(fp, pred_area),
        "absolute_area_error_px": abs(pred_area - gt_area),
        "absolute_area_error": abs(pred_area - gt_area),
        "relative_area_error": ratio(abs(pred_area - gt_area), gt_area),
        "empty_prediction": int(pred_area == 0),
        "pred_components": int(n_components),
        "component_tp": roi_hit,
        "component_fp": int(n_components - roi_hit),
        "target_fn": 1 - roi_hit,
        "roi_hit": roi_hit,
        "best_component_iou": float(ious.max()) if n_components else 0.0,
        "hd95_px": np.nan,
        "assd_px": np.nan,
        "hausdorff95": np.nan,
        "average_symmetric_surface_distance": np.nan,
        "surface_dice": 0.0 if pred_area == 0 else np.nan,
        "surface_tolerance_px": float(surface_tolerance_px),
        "centroid_distance": np.nan,
        "distance_valid": int(pred_area > 0),
    }
    if distances and pred_area:
        edge_p = pred ^ ndimage.binary_erosion(
            pred, structure=np.ones((3, 3)), border_value=0
        )
        edge_g = gt ^ ndimage.binary_erosion(
            gt, structure=np.ones((3, 3)), border_value=0
        )
        d_gt_to_pred = ndimage.distance_transform_edt(~edge_p)[edge_g]
        d_pred_to_gt = ndimage.distance_transform_edt(~edge_g)[edge_p]
        pooled = np.concatenate([d_gt_to_pred, d_pred_to_gt])
        result["hd95_px"] = float(np.percentile(pooled, 95))
        result["assd_px"] = float(pooled.mean())
        result["hausdorff95"] = result["hd95_px"]
        result["average_symmetric_surface_distance"] = result["assd_px"]
        result["surface_dice"] = ratio(
            int((d_gt_to_pred <= surface_tolerance_px).sum())
            + int((d_pred_to_gt <= surface_tolerance_px).sum()),
            len(d_gt_to_pred) + len(d_pred_to_gt),
            0,
        )
        result["centroid_distance"] = float(
            np.linalg.norm(np.asarray(ndimage.center_of_mass(pred)) - np.asarray(ndimage.center_of_mass(gt)))
        )
    return result


def conditional_binary_metrics(
    y_true: Sequence[int], probability: Sequence[float], threshold: float = 0.5
) -> dict:
    """Binary discrimination and threshold metrics for evaluable units only."""
    y = _binary_labels(y_true)
    p = _finite_probabilities(probability)
    if len(y) != len(p):
        raise ValueError("Labels and probabilities must have equal length.")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1].")

    prediction = (p >= threshold).astype(int)
    tn, fp, fn, tp = skm.confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    tn, fp, fn, tp = map(int, (tn, fp, fn, tp))
    sensitivity = ratio(tp, tp + fn)
    specificity = ratio(tn, tn + fp)
    both_classes = np.unique(y).size == 2
    clipped = np.clip(p, 1e-7, 1 - 1e-7)
    return {
        "n": int(len(y)),
        "events": int(y.sum()),
        "nonevents": int((1 - y).sum()),
        "threshold": float(threshold),
        "auroc": float(skm.roc_auc_score(y, p)) if both_classes else np.nan,
        "average_precision": (
            float(skm.average_precision_score(y, p)) if np.any(y == 1) else np.nan
        ),
        "accuracy": ratio(tp + tn, len(y)),
        "balanced_accuracy": (
            float((sensitivity + specificity) / 2)
            if np.isfinite(sensitivity) and np.isfinite(specificity)
            else np.nan
        ),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ratio(tp, tp + fp),
        "npv": ratio(tn, tn + fn),
        "f1": ratio(2 * tp, 2 * tp + fp + fn, 0),
        "mcc": ratio(
            tp * tn - fp * fn,
            math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)),
        ),
        "brier": float(np.mean((p - y) ** 2)),
        "nll": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "confusion_matrix_row_proportions": [
            [ratio(tn, tn + fp), ratio(fp, tn + fp)],
            [ratio(fn, fn + tp), ratio(tp, fn + tp)],
        ],
    }


def confusion_matrix_2x3(
    y_true: Sequence[int], prediction: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Return counts and row proportions for columns [negative, positive, abstain]."""
    y = _binary_labels(y_true)
    pred = np.asarray(prediction, dtype=int).reshape(-1)
    if len(y) != len(pred):
        raise ValueError("Labels and predictions must have equal length.")
    if not np.isin(pred, [0, 1, ABSTAIN]).all():
        raise ValueError("Predictions must be 0, 1, or ABSTAIN (-1).")
    matrix = np.zeros((2, 3), dtype=int)
    column = {0: 0, 1: 1, ABSTAIN: 2}
    for label, value in zip(y, pred, strict=True):
        matrix[int(label), column[int(value)]] += 1
    row_sum = matrix.sum(axis=1, keepdims=True)
    proportions = np.divide(
        matrix,
        row_sum,
        out=np.full(matrix.shape, np.nan, dtype=float),
        where=row_sum > 0,
    )
    return matrix, proportions


def calibration_curve_table(
    y_true: Sequence[int],
    probability: Sequence[float],
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> pd.DataFrame:
    """Return auditable reliability bins for equal-width or equal-count binning."""
    y = _binary_labels(y_true)
    p = _finite_probabilities(probability)
    if len(y) != len(p):
        raise ValueError("Labels and probabilities must have equal length.")
    if n_bins < 1:
        raise ValueError("n_bins must be positive.")
    if strategy not in {"uniform", "quantile"}:
        raise ValueError("strategy must be 'uniform' or 'quantile'.")

    rows: list[dict] = []
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        assignment = np.minimum(np.searchsorted(edges, p, side="right") - 1, n_bins - 1)
        assignment = np.maximum(assignment, 0)
        groups = [(i, np.flatnonzero(assignment == i), edges[i], edges[i + 1]) for i in range(n_bins)]
    else:
        order = np.argsort(p, kind="mergesort")
        chunks = np.array_split(order, min(n_bins, len(order)))
        groups = []
        for i, idx in enumerate(chunks):
            lower = float(p[idx].min()) if len(idx) else np.nan
            upper = float(p[idx].max()) if len(idx) else np.nan
            groups.append((i, idx, lower, upper))

    for bin_index, idx, lower, upper in groups:
        count = int(len(idx))
        mean_probability = float(p[idx].mean()) if count else np.nan
        observed_fraction = float(y[idx].mean()) if count else np.nan
        rows.append(
            {
                "strategy": strategy,
                "bin": int(bin_index),
                "lower": float(lower),
                "upper": float(upper),
                "n": count,
                "mean_probability": mean_probability,
                "observed_fraction": observed_fraction,
                "absolute_gap": (
                    abs(mean_probability - observed_fraction) if count else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(curve: pd.DataFrame) -> float:
    """Weighted absolute calibration error from a calibration-curve table."""
    if not {"n", "absolute_gap"}.issubset(curve):
        raise ValueError("Calibration curve needs n and absolute_gap columns.")
    total = int(curve["n"].sum())
    if total == 0:
        return np.nan
    return float((curve["n"] * curve["absolute_gap"].fillna(0)).sum() / total)


def calibration_slope_intercept(
    y_true: Sequence[int], probability: Sequence[float], *, epsilon: float = 1e-7
) -> dict:
    """Fit ``logit(Y) = intercept + slope * logit(p)`` without regularisation.

    Undefined or weakly identified cases are returned with an explicit status
    instead of silently substituting a penalised estimate.
    """
    y = _binary_labels(y_true).astype(float)
    p = _finite_probabilities(probability)
    result = {
        "n": int(len(y)),
        "events": int(y.sum()),
        "nonevents": int((1 - y).sum()),
        "intercept": np.nan,
        "slope": np.nan,
        "intercept_se": np.nan,
        "slope_se": np.nan,
        "status": "ok",
    }
    if np.unique(y).size < 2:
        result["status"] = "undefined_single_class"
        return result
    x = special.logit(np.clip(p, epsilon, 1 - epsilon))
    if np.ptp(x) <= 1e-12:
        result["status"] = "undefined_constant_probability"
        return result

    design = np.column_stack([np.ones(len(x)), x])

    def objective(beta):
        eta = design @ beta
        return float(np.sum(np.logaddexp(0.0, eta) - y * eta))

    def gradient(beta):
        return design.T @ (special.expit(design @ beta) - y)

    fitted = optimize.minimize(
        objective,
        x0=np.array([0.0, 1.0]),
        jac=gradient,
        method="BFGS",
        options={"gtol": 1e-9, "maxiter": 1000},
    )
    beta = np.asarray(fitted.x, dtype=float)
    if not np.isfinite(beta).all():
        result["status"] = "fit_failed_nonfinite"
        return result
    fitted_probability = special.expit(design @ beta)
    information = design.T @ (design * (fitted_probability * (1 - fitted_probability))[:, None])
    try:
        covariance = np.linalg.inv(information)
        standard_error = np.sqrt(np.diag(covariance))
    except np.linalg.LinAlgError:
        standard_error = np.array([np.nan, np.nan])
        result["status"] = "ok_singular_information"
    if not fitted.success and result["status"] == "ok":
        # BFGS commonly reports precision loss for almost-perfect calibration;
        # the finite optimum is still useful, but the status remains visible.
        result["status"] = "ok_optimizer_warning"
    result.update(
        {
            "intercept": float(beta[0]),
            "slope": float(beta[1]),
            "intercept_se": float(standard_error[0]),
            "slope_se": float(standard_error[1]),
        }
    )
    return result


def calibration_metrics(
    y_true: Sequence[int], probability: Sequence[float], *, n_bins: int = 10
) -> dict:
    """Brier/NLL, fixed ECE, adaptive ECE, and calibration regression."""
    y = _binary_labels(y_true)
    p = _finite_probabilities(probability)
    if len(y) != len(p):
        raise ValueError("Labels and probabilities must have equal length.")
    clipped = np.clip(p, 1e-7, 1 - 1e-7)
    uniform = calibration_curve_table(y, p, n_bins=n_bins, strategy="uniform")
    adaptive = calibration_curve_table(y, p, n_bins=n_bins, strategy="quantile")
    ece = expected_calibration_error(uniform)
    adaptive_ece = expected_calibration_error(adaptive)
    regression = calibration_slope_intercept(y, p)
    return {
        "n": int(len(y)),
        "brier": float(np.mean((p - y) ** 2)),
        "nll": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))),
        "ece": ece,
        "adaptive_ece": adaptive_ece,
        "ece_equal_width": ece,
        "ece_equal_mass": adaptive_ece,
        "n_bins_requested": int(n_bins),
        "n_nonempty_uniform_bins": int((uniform.n > 0).sum()),
        "n_adaptive_bins": int((adaptive.n > 0).sum()),
        "regression": regression,
        "calibration_intercept": regression["intercept"],
        "calibration_slope": regression["slope"],
    }


def _binary_logit(raw_logits: Sequence[float] | np.ndarray) -> np.ndarray:
    """Normalise one-logit or two-logit binary outputs to log-odds for class 1."""
    logits = np.asarray(raw_logits, dtype=float)
    if logits.ndim == 1:
        result = logits
    elif logits.ndim == 2 and logits.shape[1] == 1:
        result = logits[:, 0]
    elif logits.ndim == 2 and logits.shape[1] == 2:
        result = logits[:, 1] - logits[:, 0]
    else:
        raise ValueError("Binary logits must have shape [N], [N,1], or [N,2].")
    if not np.isfinite(result).all():
        raise ValueError("Raw logits must be finite.")
    return result


def apply_temperature_scaling(
    raw_logits: Sequence[float] | np.ndarray, temperature: float
) -> np.ndarray:
    """Apply a locked positive temperature and return class-1 probabilities."""
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive.")
    return special.expit(_binary_logit(raw_logits) / float(temperature))


def fit_temperature_scaling(
    y_true: Sequence[int],
    raw_logits: Sequence[float] | np.ndarray,
    *,
    min_temperature: float = 0.05,
    max_temperature: float = 20.0,
) -> dict:
    """Fit scalar temperature on validation logits only by binary NLL.

    The returned dictionary is serialisable and records whether the optimum hit
    a predeclared bound.  Callers must persist it and apply it unchanged to test
    logits; this function must never be called on test labels.
    """
    y = _binary_labels(y_true).astype(float)
    logits = _binary_logit(raw_logits)
    if len(y) != len(logits):
        raise ValueError("Labels and logits must have equal length.")
    if not 0 < min_temperature < max_temperature:
        raise ValueError("Temperature bounds must satisfy 0 < min < max.")
    if np.unique(y).size < 2:
        return {
            "temperature": np.nan,
            "status": "unavailable_single_class_no_default_temperature",
            "n": int(len(y)),
            "events": int(y.sum()),
            "nll_before": float(np.mean(np.logaddexp(0.0, logits) - y * logits)),
            "nll_after": np.nan,
            "bounds": [float(min_temperature), float(max_temperature)],
        }

    log_bounds = (math.log(min_temperature), math.log(max_temperature))

    def nll(log_temperature: float) -> float:
        scaled = logits / math.exp(float(log_temperature))
        return float(np.mean(np.logaddexp(0.0, scaled) - y * scaled))

    fitted = optimize.minimize_scalar(nll, bounds=log_bounds, method="bounded")
    temperature = float(math.exp(fitted.x))
    if not fitted.success or not np.isfinite(temperature):
        return {
            "temperature": np.nan,
            "status": "unavailable_optimizer_failure_no_default_temperature",
            "n": int(len(y)),
            "events": int(y.sum()),
            "nll_before": nll(0.0),
            "nll_after": np.nan,
            "bounds": [float(min_temperature), float(max_temperature)],
            "optimizer_message": str(fitted.message),
        }
    tolerance = 1e-4
    hit_bound = (
        abs(temperature - min_temperature) <= tolerance * min_temperature
        or abs(temperature - max_temperature) <= tolerance * max_temperature
    )
    return {
        "temperature": temperature,
        "status": (
            "ok_bound_hit" if hit_bound else "ok"
        ),
        "n": int(len(y)),
        "events": int(y.sum()),
        "nll_before": nll(0.0),
        "nll_after": nll(float(fitted.x)),
        "bounds": [float(min_temperature), float(max_temperature)],
        "optimizer_message": str(fitted.message),
    }


def fit_temperature_on_probabilities(
    y_true: Sequence[int],
    raw_probability: Sequence[float],
    *,
    epsilon: float = 1e-7,
    min_temperature: float = 0.05,
    max_temperature: float = 20.0,
) -> dict:
    """Convenience wrapper for final-unit probabilities averaged from frames/eyes."""
    probability = _finite_probabilities(raw_probability)
    logits = special.logit(np.clip(probability, epsilon, 1 - epsilon))
    result = fit_temperature_scaling(
        y_true,
        logits,
        min_temperature=min_temperature,
        max_temperature=max_temperature,
    )
    result["input"] = "logit_of_clipped_final_unit_probability"
    result["epsilon"] = float(epsilon)
    return result


def risk_coverage_curve(
    y_true: Sequence[int],
    probability: Sequence[float],
    *,
    evaluable: Sequence[bool] | None = None,
    threshold: float = 0.5,
    include_abstentions_as_failures: bool = True,
) -> pd.DataFrame:
    """Confidence-ranked selective risk and coverage.

    Evaluable units are ranked by distance from the locked decision threshold.
    (At threshold 0.5 this is equivalent to ranking by ``max(p, 1-p)``.)  For the failure-aware
    curve, segmentation-gate abstentions are appended at lowest confidence and
    counted as errors.  The returned ``aurc`` convention is the mean cumulative
    risk across attainable coverage steps; it is repeated on every row for
    convenient export.
    """
    y = _binary_labels(y_true)
    p = np.asarray(probability, dtype=float).reshape(-1)
    if len(y) != len(p):
        raise ValueError("Labels and probabilities must have equal length.")
    valid = np.isfinite(p) if evaluable is None else np.asarray(evaluable, dtype=bool).reshape(-1)
    if len(valid) != len(y) or np.any(valid & ~np.isfinite(p)):
        raise ValueError("Evaluability/probability mismatch.")
    if np.any(valid & ((p < 0) | (p > 1))):
        raise ValueError("Evaluable probabilities must be in [0, 1].")

    valid_indices = np.flatnonzero(valid)
    confidence = np.abs(p[valid_indices] - threshold)
    # Stable ordering makes tied-confidence results reproducible.
    order = valid_indices[np.argsort(-confidence, kind="mergesort")]
    errors = ((p[order] >= threshold).astype(int) != y[order]).astype(int)
    confidences = np.abs(p[order] - threshold)
    if include_abstentions_as_failures:
        abstained = np.flatnonzero(~valid)
        order = np.concatenate([order, abstained])
        errors = np.concatenate([errors, np.ones(len(abstained), dtype=int)])
        confidences = np.concatenate([confidences, np.full(len(abstained), np.nan)])
        denominator = len(y)
    else:
        denominator = len(valid_indices)
    if len(order) == 0:
        return pd.DataFrame(
            columns=["rank", "unit_index", "confidence", "coverage", "risk", "aurc"]
        )
    cumulative_risk = np.cumsum(errors) / np.arange(1, len(errors) + 1)
    coverage = np.arange(1, len(errors) + 1) / denominator
    aurc = float(cumulative_risk.mean())
    return pd.DataFrame(
        {
            "rank": np.arange(1, len(order) + 1),
            "unit_index": order,
            "confidence": confidences,
            "coverage": coverage,
            "risk": cumulative_risk,
            "aurc": aurc,
            "failure_inclusive": bool(include_abstentions_as_failures),
        }
    )


def selective_binary_metrics(
    y_true: Sequence[int],
    probability: Sequence[float],
    *,
    evaluable: Sequence[bool] | None = None,
    threshold: float = 0.5,
    n_calibration_bins: int = 10,
    localized_success: Sequence[bool] | None = None,
) -> dict:
    """Evaluate a classifier with abstention as an explicit third outcome.

    AUROC, AP and calibration are conditional on evaluability because abstained
    units have no diagnostic probability.  Failure-inclusive sensitivity,
    specificity, balanced accuracy, and accuracy count abstentions as failures.
    """
    y = _binary_labels(y_true)
    p = np.asarray(probability, dtype=float).reshape(-1)
    if len(y) != len(p):
        raise ValueError("Labels and probabilities must have equal length.")
    if evaluable is None:
        valid = np.isfinite(p)
    else:
        valid = np.asarray(evaluable, dtype=bool).reshape(-1)
        if len(valid) != len(y):
            raise ValueError("evaluable must have one value per observation.")
    if np.any(valid & ~np.isfinite(p)):
        raise ValueError("Every evaluable unit must have a finite probability.")
    if np.any(valid & ((p < 0) | (p > 1))):
        raise ValueError("Every evaluable probability must be in [0, 1].")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1].")

    prediction = np.full(len(y), ABSTAIN, dtype=int)
    prediction[valid] = (p[valid] >= threshold).astype(int)
    matrix, row_proportions = confusion_matrix_2x3(y, prediction)
    tn, fp, abstain_negative = map(int, matrix[0])
    fn, tp, abstain_positive = map(int, matrix[1])
    n_covered = int(valid.sum())
    n_total = int(len(y))

    conditional = (
        conditional_binary_metrics(y[valid], p[valid], threshold)
        if n_covered
        else {
            key: np.nan
            for key in (
                "auroc",
                "average_precision",
                "accuracy",
                "balanced_accuracy",
                "sensitivity",
                "specificity",
                "ppv",
                "npv",
                "f1",
                "mcc",
                "brier",
                "nll",
            )
        }
    )
    if n_covered:
        conditional["n"] = n_covered
        calibration = calibration_metrics(y[valid], p[valid], n_bins=n_calibration_bins)
    else:
        conditional.update({"n": 0, "events": 0, "nonevents": 0, "threshold": threshold})
        calibration = {
            "n": 0,
            "brier": np.nan,
            "nll": np.nan,
            "ece": np.nan,
            "adaptive_ece": np.nan,
            "ece_equal_width": np.nan,
            "ece_equal_mass": np.nan,
            "calibration_intercept": np.nan,
            "calibration_slope": np.nan,
            "regression": {
                "n": 0,
                "events": 0,
                "nonevents": 0,
                "intercept": np.nan,
                "slope": np.nan,
                "intercept_se": np.nan,
                "slope_se": np.nan,
                "status": "undefined_no_evaluable_units",
            },
        }

    sensitivity_fi = ratio(tp, int((y == 1).sum()))
    specificity_fi = ratio(tn, int((y == 0).sum()))
    failure_inclusive = {
        "accuracy": ratio(tp + tn, n_total),
        "balanced_accuracy": (
            float((sensitivity_fi + specificity_fi) / 2)
            if np.isfinite(sensitivity_fi) and np.isfinite(specificity_fi)
            else np.nan
        ),
        "sensitivity": sensitivity_fi,
        "specificity": specificity_fi,
        "correct": int(tp + tn),
        "incorrect_or_abstained": int(n_total - tp - tn),
    }
    failure_curve = risk_coverage_curve(
        y,
        p,
        evaluable=valid,
        threshold=threshold,
        include_abstentions_as_failures=True,
    )
    conditional_curve = (
        risk_coverage_curve(
            y[valid],
            p[valid],
            threshold=threshold,
            include_abstentions_as_failures=False,
        )
        if n_covered
        else pd.DataFrame()
    )
    failure_inclusive["aurc"] = (
        float(failure_curve.aurc.iloc[0]) if len(failure_curve) else np.nan
    )
    result = {
        "n_total": n_total,
        "n_covered": n_covered,
        "n_abstain": int(n_total - n_covered),
        "coverage": ratio(n_covered, n_total),
        "coverage_negative": ratio(tn + fp, int((y == 0).sum())),
        "coverage_positive": ratio(fn + tp, int((y == 1).sum())),
        "class_conditional_coverage": {
            "negative": ratio(tn + fp, int((y == 0).sum())),
            "positive": ratio(fn + tp, int((y == 1).sum())),
        },
        "abstention_rate": ratio(n_total - n_covered, n_total),
        "selective_risk": ratio(fp + fn, n_covered),
        "conditional_aurc": (
            float(conditional_curve.aurc.iloc[0]) if len(conditional_curve) else np.nan
        ),
        "failure_aware_aurc": failure_inclusive["aurc"],
        "failure_aware_accuracy": failure_inclusive["accuracy"],
        "failure_aware_balanced_accuracy": failure_inclusive["balanced_accuracy"],
        "failure_aware_sensitivity": failure_inclusive["sensitivity"],
        "failure_aware_specificity": failure_inclusive["specificity"],
        "conditional": conditional,
        "failure_inclusive": failure_inclusive,
        "calibration": calibration,
        "confusion_matrix_2x3": matrix.tolist(),
        "confusion_matrix_2x3_row_proportions": row_proportions.tolist(),
        "confusion_columns": ["predicted_negative", "predicted_positive", "abstain"],
        "prediction": prediction.tolist(),
        "metric_scope_note": (
            "Discrimination and calibration are conditional on evaluability; "
            "failure-inclusive threshold metrics count abstention as failure."
        ),
    }
    if localized_success is not None:
        localized = np.asarray(localized_success, dtype=bool).reshape(-1)
        if len(localized) != n_total:
            raise ValueError("localized_success must have one value per observation.")
        correct = prediction == y
        joint = correct & localized & valid
        result["localized_diagnostic_success"] = {
            "n_success": int(joint.sum()),
            "rate": float(joint.mean()),
            "localized_rate": float(localized.mean()),
            "definition": "correct non-abstained diagnosis and locked GT-localisation criterion",
        }
    return result


def decision_curve_net_benefit(
    y_true: Sequence[int],
    probability: Sequence[float],
    *,
    evaluable: Sequence[bool] | None = None,
    thresholds: Iterable[float] | None = None,
) -> pd.DataFrame:
    """Decision-curve net benefit, with abstentions treated as no intervention.

    The denominator is the complete intended cohort, not only evaluable units.
    This prevents a low-coverage model from receiving an artificial advantage.
    """
    y = _binary_labels(y_true)
    p = np.asarray(probability, dtype=float).reshape(-1)
    if len(y) != len(p):
        raise ValueError("Labels and probabilities must have equal length.")
    valid = np.isfinite(p) if evaluable is None else np.asarray(evaluable, dtype=bool)
    if len(valid) != len(y) or np.any(valid & ~np.isfinite(p)):
        raise ValueError("Evaluability/probability mismatch.")
    grid = np.asarray(
        list(thresholds) if thresholds is not None else np.linspace(0.01, 0.99, 99),
        dtype=float,
    )
    if grid.ndim != 1 or len(grid) == 0 or not ((0 < grid) & (grid < 1)).all():
        raise ValueError("Decision thresholds must lie strictly between 0 and 1.")
    n = len(y)
    prevalence = float(y.mean())
    rows = []
    for threshold in grid:
        positive = valid & (p >= threshold)
        tp = int((positive & (y == 1)).sum())
        fp = int((positive & (y == 0)).sum())
        odds = float(threshold / (1 - threshold))
        model_nb = tp / n - fp / n * odds
        all_nb = prevalence - (1 - prevalence) * odds
        rows.append(
            {
                "threshold": float(threshold),
                "net_benefit_model": float(model_nb),
                "net_benefit_treat_all": float(all_nb),
                "net_benefit_treat_none": 0.0,
                "standardized_net_benefit_model": ratio(model_nb, prevalence),
                "n_total": int(n),
                "n_evaluable": int(valid.sum()),
                "n_abstain": int((~valid).sum()),
                "tp": tp,
                "fp": fp,
            }
        )
    return pd.DataFrame(rows)


def _identity_value(group: pd.DataFrame, column: str):
    if column not in group:
        raise ValueError(f"Required column is missing: {column}")
    if group[column].nunique(dropna=False) != 1:
        raise ValueError(f"Inconsistent {column} within aggregation unit.")
    return group[column].iloc[0]


def aggregate_frames_to_eyes(
    frames: pd.DataFrame,
    *,
    threshold: float = 0.5,
    frames_per_eye: int = 7,
    min_valid_frames: int = 4,
    validity_column: str = "roi_valid",
    probability_column: str = "probability",
    reason_column: str = "abstention_reason",
    strict_invalid_probability: bool = True,
    mean_columns: Sequence[str] | None = None,
    localisation_column: str = "roi_hit",
) -> pd.DataFrame:
    """Aggregate seven frame probabilities into an eye probability or abstention."""
    required = {"case_id", "patient_id", "side", "frame_id", "label", probability_column}
    missing = required - set(frames.columns)
    if missing:
        raise ValueError(f"Missing frame columns: {sorted(missing)}")
    if not 1 <= min_valid_frames <= frames_per_eye:
        raise ValueError("min_valid_frames must be between 1 and frames_per_eye.")

    rows: list[dict] = []
    for (patient_key, case_id, side_key), group in frames.groupby(
        ["patient_id", "case_id", "side"], sort=True
    ):
        if len(group) != frames_per_eye or group.frame_id.nunique() != frames_per_eye:
            raise ValueError(
                f"Eye {case_id!r} must contain exactly {frames_per_eye} unique frames."
            )
        patient_id = _identity_value(group, "patient_id")
        side = _identity_value(group, "side")
        if str(patient_id) != str(patient_key) or str(side) != str(side_key):
            raise ValueError("Eye grouping identity changed during aggregation")
        label = int(_identity_value(group, "label"))
        if label not in (0, 1):
            raise ValueError("Eye label must be binary.")

        probability = group[probability_column].to_numpy(float)
        valid = (
            _strict_boolean_array(group[validity_column], name=validity_column)
            if validity_column in group
            else np.isfinite(probability)
        )
        if np.any(valid & ~np.isfinite(probability)):
            raise ValueError(f"Eye {case_id!r}: valid ROI has no finite probability.")
        if np.any(valid & ((probability < 0) | (probability > 1))):
            raise ValueError(f"Eye {case_id!r}: valid ROI probability is outside [0, 1].")
        if strict_invalid_probability and np.any(~valid & np.isfinite(probability)):
            raise ValueError(
                f"Eye {case_id!r}: invalid ROI carries a probability; this could leak a fallback."
            )

        n_valid = int(valid.sum())
        evaluable = n_valid >= min_valid_frames
        eye_probability = float(probability[valid].mean()) if evaluable else np.nan
        prediction = int(eye_probability >= threshold) if evaluable else ABSTAIN
        row = {
            "case_id": case_id,
            "patient_id": patient_id,
            "side": side,
            "label": label,
            "n_frames": int(len(group)),
            "n_valid_frames": n_valid,
            "valid_frame_fraction": n_valid / frames_per_eye,
            "evaluable": bool(evaluable),
            "probability": eye_probability,
            "prediction": prediction,
            "abstention_reason": "" if evaluable else "insufficient_valid_frames",
        }
        for optional_identity in ("label_3class", "split", "seed", "model", "arm"):
            if optional_identity in group:
                row[optional_identity] = _identity_value(group, optional_identity)
        if reason_column in group:
            invalid_reasons = (
                group.loc[~valid, reason_column].fillna("unspecified").replace("", "unspecified")
            )
            counts = invalid_reasons.value_counts(sort=False).sort_index()
            row["invalid_frame_reasons"] = ";".join(
                f"{reason}:{int(count)}" for reason, count in counts.items()
            )
        selected_mean_columns = mean_columns or ()
        for column in selected_mean_columns:
            if column in group:
                row[column] = float(pd.to_numeric(group[column], errors="coerce").mean())

        # GT is an audit-only field.  It never affects evaluability or probability.
        if localisation_column in group:
            # Only a quality-gate-valid predicted ROI can support a localised
            # diagnosis; overlap from a rejected component is audit information,
            # not diagnostic evidence.
            hits = group[localisation_column].fillna(0).to_numpy(bool) & valid
            hit_count = int(hits.sum())
            row.update(
                {
                    "roi_hit_frames": hit_count,
                    "roi_hit_fraction": hit_count / frames_per_eye,
                    "localized": bool(hit_count >= min_valid_frames),
                    "localized_diagnostic_success": int(
                        evaluable and prediction == label and hit_count >= min_valid_frames
                    ),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_eyes_to_patients(
    eyes: pd.DataFrame,
    *,
    threshold: float = 0.5,
    required_sides: Sequence[str] = ("SAG", "SOL"),
    probability_column: str = "probability",
    evaluable_column: str = "evaluable",
    mean_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Aggregate two eyes; if either eye abstains, the patient abstains."""
    required = {"patient_id", "case_id", "side", "label", probability_column, evaluable_column}
    missing = required - set(eyes.columns)
    if missing:
        raise ValueError(f"Missing eye columns: {sorted(missing)}")
    expected_sides = set(required_sides)
    rows = []
    for patient_id, group in eyes.groupby("patient_id", sort=True):
        if len(group) != len(required_sides) or set(group.side) != expected_sides:
            raise ValueError(
                f"Patient {patient_id!r} must contain exactly the required two eyes."
            )
        label = int(_identity_value(group, "label"))
        valid = group[evaluable_column].to_numpy(bool)
        probability = group[probability_column].to_numpy(float)
        if np.any(valid & ~np.isfinite(probability)):
            raise ValueError("Evaluable eye lacks a finite probability.")
        patient_evaluable = bool(valid.all())
        patient_probability = float(probability.mean()) if patient_evaluable else np.nan
        prediction = int(patient_probability >= threshold) if patient_evaluable else ABSTAIN
        row = {
            "patient_id": patient_id,
            "label": label,
            "n_eyes": int(len(group)),
            "n_evaluable_eyes": int(valid.sum()),
            "evaluable": patient_evaluable,
            "probability": patient_probability,
            "prediction": prediction,
            "abstention_reason": "" if patient_evaluable else "one_or_more_eyes_abstained",
        }
        for optional_identity in ("label_3class", "split", "seed", "model", "arm"):
            if optional_identity in group:
                row[optional_identity] = _identity_value(group, optional_identity)
        for column in mean_columns or ():
            if column in group:
                row[column] = float(pd.to_numeric(group[column], errors="coerce").mean())
        if "localized" in group:
            localized = group.localized.to_numpy(bool)
            row["localized"] = bool(localized.all())
            row["localized_diagnostic_success"] = int(
                patient_evaluable and localized.all() and prediction == label
            )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_numeric_columns(
    table: pd.DataFrame, columns: Sequence[str]
) -> dict[str, dict]:
    """Descriptive summaries with explicit valid denominators for each metric."""
    result: dict[str, dict] = {}
    for column in columns:
        if column not in table:
            continue
        values = pd.to_numeric(table[column], errors="coerce").dropna().to_numpy(float)
        result[column] = {
            "n_valid": int(len(values)),
            "n_total": int(len(table)),
            "mean": float(values.mean()) if len(values) else np.nan,
            "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
            "median": float(np.median(values)) if len(values) else np.nan,
            "q1": float(np.quantile(values, 0.25)) if len(values) else np.nan,
            "q3": float(np.quantile(values, 0.75)) if len(values) else np.nan,
        }
    return result


def select_threshold(scores: Mapping[float, float]) -> float:
    """Select highest validation score, then closest to 0.5, then smaller cut-off."""
    if not scores or not all(np.isfinite(value) for value in scores.values()):
        raise ValueError("Threshold-selection scores must be non-empty and finite.")
    return float(min(scores, key=lambda t: (-scores[t], round(abs(t - 0.5), 12), t)))


def select_binary_threshold(
    y_true: Sequence[int],
    probability: Sequence[float],
    *,
    evaluable: Sequence[bool] | None = None,
    grid: Sequence[float] | None = None,
    objective: str = "balanced_accuracy",
) -> dict:
    """Lock a classification threshold from validation predictions.

    Only evaluable validation units enter the objective.  Coverage is fixed by
    the segmentation quality gate and is not tuned by this function.  The
    default grid is deliberately finite and must be recorded in the protocol.
    """
    y = _binary_labels(y_true)
    p = np.asarray(probability, dtype=float).reshape(-1)
    if len(y) != len(p):
        raise ValueError("Labels and probabilities must have equal length.")
    valid = np.isfinite(p) if evaluable is None else np.asarray(evaluable, dtype=bool)
    if len(valid) != len(y) or np.any(valid & ~np.isfinite(p)):
        raise ValueError("Evaluability/probability mismatch.")
    y_valid, p_valid = y[valid], p[valid]
    if not len(y_valid) or np.unique(y_valid).size < 2:
        return {
            "threshold": np.nan,
            "objective": objective,
            "objective_value": np.nan,
            "n_evaluable": int(len(y_valid)),
            "coverage": ratio(len(y_valid), len(y)),
            "grid": [float(value) for value in ([] if grid is None else grid)],
            "scores": {},
            "status": "unavailable_requires_evaluable_examples_from_both_classes",
            "test_policy": "all_test_decisions_abstain",
        }
    thresholds = np.asarray(
        grid if grid is not None else np.round(np.arange(0.05, 0.951, 0.05), 10),
        dtype=float,
    )
    if thresholds.ndim != 1 or len(thresholds) == 0 or not ((0 <= thresholds) & (thresholds <= 1)).all():
        raise ValueError("Threshold grid must be a non-empty 1-D sequence in [0, 1].")
    scores: dict[float, float] = {}
    for threshold in thresholds:
        metrics = conditional_binary_metrics(y_valid, p_valid, float(threshold))
        if objective not in metrics or not np.isscalar(metrics[objective]):
            raise ValueError(f"Unknown scalar threshold objective: {objective}")
        scores[float(threshold)] = float(metrics[objective])
    selected = select_threshold(scores)
    return {
        "threshold": selected,
        "objective": objective,
        "objective_value": scores[selected],
        "n_evaluable": int(len(y_valid)),
        "coverage": ratio(len(y_valid), len(y)),
        "grid": [float(value) for value in thresholds],
        "scores": {str(key): value for key, value in scores.items()},
        "tie_break": "maximum objective; closest to 0.5; smaller threshold",
        "status": "ok",
    }


def fit_final_unit_calibration_threshold_lock(
    y_true: Sequence[int],
    raw_probability: Sequence[float],
    *,
    evaluable: Sequence[bool] | None = None,
    threshold_grid: Sequence[float] | None = None,
    threshold_objective: str = "balanced_accuracy",
    probability_clip_epsilon: float = 1e-7,
    temperature_bounds: Sequence[float] = (0.05, 20.0),
) -> dict:
    """Fit validation-only temperature, then lock a calibrated threshold.

    This helper enforces the order temperature -> threshold and returns an
    unavailable lock (not an identity/default substitution) when calibration
    cannot be fitted.  The caller should then make every test decision at that
    level abstain, as recorded in ``test_policy``.
    """
    y = _binary_labels(y_true)
    raw = np.asarray(raw_probability, dtype=float).reshape(-1)
    if len(y) != len(raw):
        raise ValueError("Labels and probabilities must have equal length.")
    valid = np.isfinite(raw) if evaluable is None else np.asarray(evaluable, dtype=bool)
    if len(valid) != len(y) or np.any(valid & ~np.isfinite(raw)):
        raise ValueError("Evaluability/probability mismatch.")
    if np.any(valid & ((raw < 0) | (raw > 1))):
        raise ValueError("Evaluable probabilities must be in [0, 1].")
    if len(temperature_bounds) != 2:
        raise ValueError("temperature_bounds must contain [minimum, maximum].")
    if not valid.any():
        return {
            "status": "unavailable_no_evaluable_validation_units",
            "calibration": {
                "temperature": np.nan,
                "status": "unavailable_no_evaluable_validation_units",
                "n": 0,
            },
            "threshold": {
                "threshold": np.nan,
                "status": "unavailable_due_to_calibration",
                "test_policy": "all_test_decisions_abstain",
            },
            "test_policy": "all_test_decisions_abstain",
            "fit_partition": "validation",
        }
    calibration = fit_temperature_on_probabilities(
        y[valid],
        raw[valid],
        epsilon=probability_clip_epsilon,
        min_temperature=float(temperature_bounds[0]),
        max_temperature=float(temperature_bounds[1]),
    )
    if not str(calibration["status"]).startswith("ok"):
        return {
            "status": "unavailable_calibration_fit",
            "calibration": calibration,
            "threshold": {
                "threshold": np.nan,
                "status": "unavailable_due_to_calibration",
                "test_policy": "all_test_decisions_abstain",
            },
            "test_policy": "all_test_decisions_abstain",
            "fit_partition": "validation",
        }
    calibrated = np.full(len(raw), np.nan)
    raw_logits = special.logit(
        np.clip(raw[valid], probability_clip_epsilon, 1 - probability_clip_epsilon)
    )
    calibrated[valid] = apply_temperature_scaling(
        raw_logits, float(calibration["temperature"])
    )
    threshold = select_binary_threshold(
        y,
        calibrated,
        evaluable=valid,
        grid=threshold_grid,
        objective=threshold_objective,
    )
    status = "ok" if threshold.get("status") == "ok" else "unavailable_threshold_fit"
    return {
        "status": status,
        "calibration": calibration,
        "threshold": threshold,
        "test_policy": (
            "apply_locked_temperature_then_locked_threshold"
            if status == "ok"
            else "all_test_decisions_abstain"
        ),
        "fit_partition": "validation",
        "fit_order": ["temperature_scaling", "classification_threshold"],
    }


def flatten_metrics(value: Mapping, prefix: str = "") -> dict[str, object]:
    """Flatten nested metric dictionaries using dot-separated paths."""
    flat: dict[str, object] = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            flat.update(flatten_metrics(item, path))
        elif not isinstance(item, (list, tuple, np.ndarray)):
            flat[path] = item
    return flat


def _metric_at_path(metrics: Mapping, path: str) -> float:
    value = metrics
    for part in path.split("."):
        value = value[part]
    return float(value)


def _cluster_draw_indices(
    table: pd.DataFrame,
    rng: np.random.Generator,
    *,
    cluster_column: str,
    label_column: str,
    stratified: bool,
) -> np.ndarray:
    cluster_labels = table[[cluster_column, label_column]].drop_duplicates()
    if cluster_labels.groupby(cluster_column)[label_column].nunique().max() != 1:
        raise ValueError("Every bootstrap cluster must have one consistent label.")
    cluster_labels = cluster_labels.drop_duplicates(cluster_column)
    indices = {
        cluster: np.asarray(idx, dtype=int)
        for cluster, idx in table.reset_index(drop=True).groupby(cluster_column).indices.items()
    }
    sampled_parts: list[np.ndarray] = []
    strata = cluster_labels.groupby(label_column, sort=True) if stratified else [(None, cluster_labels)]
    for _, stratum in strata:
        clusters = stratum[cluster_column].to_numpy()
        draws = rng.choice(clusters, size=len(clusters), replace=True)
        sampled_parts.extend(indices[cluster] for cluster in draws)
    return np.concatenate(sampled_parts)


DEFAULT_BOOTSTRAP_METRICS = (
    "coverage",
    "coverage_negative",
    "coverage_positive",
    "selective_risk",
    "conditional_aurc",
    "failure_aware_aurc",
    "conditional.auroc",
    "conditional.average_precision",
    "conditional.accuracy",
    "conditional.balanced_accuracy",
    "conditional.sensitivity",
    "conditional.specificity",
    "conditional.f1",
    "conditional.ppv",
    "conditional.npv",
    "conditional.mcc",
    "conditional.brier",
    "conditional.nll",
    "failure_inclusive.accuracy",
    "failure_inclusive.balanced_accuracy",
    "failure_inclusive.sensitivity",
    "failure_inclusive.specificity",
    "failure_inclusive.aurc",
    "calibration.ece",
    "calibration.adaptive_ece",
    "calibration.regression.intercept",
    "calibration.regression.slope",
    "localized_diagnostic_success.rate",
)


def patient_cluster_bootstrap_ci(
    table: pd.DataFrame,
    *,
    threshold: float = 0.5,
    draws: int = 2000,
    seed: int = 1729,
    metric_paths: Sequence[str] = DEFAULT_BOOTSTRAP_METRICS,
    cluster_column: str = "patient_id",
    label_column: str = "label",
    probability_column: str = "probability",
    evaluable_column: str = "evaluable",
    stratified: bool = True,
    n_calibration_bins: int = 10,
    confidence: float = 0.95,
    ci_method: str = "bca",
) -> dict[str, dict]:
    """Whole-patient bootstrap CIs, BCa when defined and percentile otherwise."""
    if draws < 1:
        raise ValueError("draws must be positive.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between 0 and 1.")
    if ci_method not in {"bca", "percentile"}:
        raise ValueError("ci_method must be 'bca' or 'percentile'.")
    required = {cluster_column, label_column, probability_column, evaluable_column}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Missing bootstrap columns: {sorted(missing)}")
    table = table.reset_index(drop=True)

    def evaluate(sample: pd.DataFrame) -> dict:
        return selective_binary_metrics(
            sample[label_column],
            sample[probability_column],
            evaluable=sample[evaluable_column],
            threshold=threshold,
            n_calibration_bins=n_calibration_bins,
            localized_success=(
                sample["localized"].to_numpy(bool) if "localized" in sample else None
            ),
        )

    point = evaluate(table)
    values = {path: [] for path in metric_paths}
    rng = np.random.default_rng(seed)
    for _ in range(draws):
        indices = _cluster_draw_indices(
            table,
            rng,
            cluster_column=cluster_column,
            label_column=label_column,
            stratified=stratified,
        )
        metrics = evaluate(table.iloc[indices])
        for path in metric_paths:
            try:
                value = _metric_at_path(metrics, path)
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(value):
                values[path].append(value)

    clusters = table[cluster_column].drop_duplicates().to_numpy()
    jackknife: dict[str, list[float]] = {path: [] for path in metric_paths}
    if ci_method == "bca" and len(clusters) >= 3:
        for cluster in clusters:
            leave_one_out = table[table[cluster_column] != cluster]
            if leave_one_out.empty:
                continue
            metrics = evaluate(leave_one_out)
            for path in metric_paths:
                try:
                    value = _metric_at_path(metrics, path)
                except (KeyError, TypeError, ValueError):
                    value = np.nan
                jackknife[path].append(value)

    result = {}
    tail = (1 - confidence) / 2
    for path, observed in values.items():
        array = np.asarray(observed, dtype=float)
        try:
            estimate = _metric_at_path(point, path)
        except (KeyError, TypeError, ValueError):
            estimate = np.nan
        low = float(np.percentile(array, 100 * tail)) if len(array) else np.nan
        high = float(np.percentile(array, 100 * (1 - tail))) if len(array) else np.nan
        method_used = "percentile"
        if ci_method == "bca" and len(array) and np.isfinite(estimate):
            jack = np.asarray(jackknife[path], dtype=float)
            if len(jack) == len(clusters) and np.isfinite(jack).all() and len(jack) >= 3:
                proportion = (np.sum(array < estimate) + 0.5 * np.sum(array == estimate)) / len(array)
                proportion = float(np.clip(proportion, 1 / (2 * len(array)), 1 - 1 / (2 * len(array))))
                z0 = float(stats.norm.ppf(proportion))
                jack_mean = float(jack.mean())
                deviations = jack_mean - jack
                denominator = 6 * float(np.sum(deviations**2) ** 1.5)
                acceleration = ratio(float(np.sum(deviations**3)), denominator, 0.0)
                adjusted = []
                for alpha in (tail, 1 - tail):
                    z_alpha = float(stats.norm.ppf(alpha))
                    divisor = 1 - acceleration * (z0 + z_alpha)
                    if abs(divisor) <= 1e-12:
                        adjusted = []
                        break
                    adjusted.append(float(stats.norm.cdf(z0 + (z0 + z_alpha) / divisor)))
                if len(adjusted) == 2 and np.isfinite(adjusted).all() and adjusted[0] <= adjusted[1]:
                    adjusted = np.clip(adjusted, 0, 1)
                    low, high = map(float, np.quantile(array, adjusted))
                    method_used = "bca"
        result[path] = {
            "estimate": estimate,
            "low": low,
            "high": high,
            "bootstrap_se": float(array.std(ddof=1)) if len(array) > 1 else np.nan,
            "valid_draws": int(len(array)),
            "requested_draws": int(draws),
            "confidence": float(confidence),
            "method": method_used,
        }
    return result


def patient_cluster_calibration_band(
    table: pd.DataFrame,
    *,
    probability_column: str = "probability",
    evaluable_column: str = "evaluable",
    label_column: str = "label",
    cluster_column: str = "patient_id",
    n_bins: int = 10,
    draws: int = 2000,
    seed: int = 1729,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Patient-cluster percentile band for an equal-width reliability curve."""
    if draws < 1:
        raise ValueError("draws must be positive.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between 0 and 1.")
    required = {probability_column, evaluable_column, label_column, cluster_column}
    missing = required - set(table)
    if missing:
        raise ValueError(f"Missing calibration-band columns: {sorted(missing)}")
    table = table.reset_index(drop=True)
    valid = table[evaluable_column].to_numpy(bool)
    if not valid.any():
        return pd.DataFrame(
            columns=[
                "strategy",
                "bin",
                "lower",
                "upper",
                "n",
                "mean_probability",
                "observed_fraction",
                "observed_low",
                "observed_high",
                "valid_draws",
            ]
        )
    point = calibration_curve_table(
        table.loc[valid, label_column],
        table.loc[valid, probability_column],
        n_bins=n_bins,
        strategy="uniform",
    )
    observed = {index: [] for index in range(n_bins)}
    mean_probability = {index: [] for index in range(n_bins)}
    rng = np.random.default_rng(seed)
    for _ in range(draws):
        indices = _cluster_draw_indices(
            table,
            rng,
            cluster_column=cluster_column,
            label_column=label_column,
            stratified=True,
        )
        sample = table.iloc[indices]
        sample_valid = sample[evaluable_column].to_numpy(bool)
        if not sample_valid.any():
            continue
        curve = calibration_curve_table(
            sample.loc[sample_valid, label_column],
            sample.loc[sample_valid, probability_column],
            n_bins=n_bins,
            strategy="uniform",
        )
        for _, row in curve[curve.n > 0].iterrows():
            index = int(row.bin)
            observed[index].append(float(row.observed_fraction))
            mean_probability[index].append(float(row.mean_probability))
    tail = (1 - confidence) / 2
    point["observed_low"] = [
        float(np.quantile(observed[index], tail)) if observed[index] else np.nan
        for index in range(n_bins)
    ]
    point["observed_high"] = [
        float(np.quantile(observed[index], 1 - tail)) if observed[index] else np.nan
        for index in range(n_bins)
    ]
    point["mean_probability_low"] = [
        float(np.quantile(mean_probability[index], tail)) if mean_probability[index] else np.nan
        for index in range(n_bins)
    ]
    point["mean_probability_high"] = [
        float(np.quantile(mean_probability[index], 1 - tail)) if mean_probability[index] else np.nan
        for index in range(n_bins)
    ]
    point["valid_draws"] = [len(observed[index]) for index in range(n_bins)]
    point["requested_draws"] = int(draws)
    point["confidence"] = float(confidence)
    return point


def patient_cluster_continuous_ci(
    table: pd.DataFrame,
    columns: Sequence[str],
    *,
    cluster_column: str = "patient_id",
    label_column: str = "label",
    draws: int = 2000,
    seed: int = 1729,
    confidence: float = 0.95,
    ci_method: str = "bca",
) -> dict[str, dict]:
    """Patient-cluster CIs for mean segmentation/continuous outcomes."""
    if draws < 1:
        raise ValueError("draws must be positive.")
    if ci_method not in {"bca", "percentile"}:
        raise ValueError("ci_method must be 'bca' or 'percentile'.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between 0 and 1.")
    missing = {cluster_column, label_column, *columns} - set(table)
    if missing:
        raise ValueError(f"Missing continuous-CI columns: {sorted(missing)}")
    table = table.reset_index(drop=True)
    point = {
        column: float(pd.to_numeric(table[column], errors="coerce").mean())
        for column in columns
    }
    bootstrap = {column: [] for column in columns}
    rng = np.random.default_rng(seed)
    for _ in range(draws):
        indices = _cluster_draw_indices(
            table,
            rng,
            cluster_column=cluster_column,
            label_column=label_column,
            stratified=True,
        )
        sample = table.iloc[indices]
        for column in columns:
            value = float(pd.to_numeric(sample[column], errors="coerce").mean())
            if np.isfinite(value):
                bootstrap[column].append(value)
    clusters = table[cluster_column].drop_duplicates().to_numpy()
    jackknife = {column: [] for column in columns}
    if ci_method == "bca" and len(clusters) >= 3:
        for cluster in clusters:
            sample = table[table[cluster_column] != cluster]
            for column in columns:
                jackknife[column].append(
                    float(pd.to_numeric(sample[column], errors="coerce").mean())
                )
    tail = (1 - confidence) / 2
    result = {}
    for column in columns:
        array = np.asarray(bootstrap[column], dtype=float)
        low = float(np.quantile(array, tail)) if len(array) else np.nan
        high = float(np.quantile(array, 1 - tail)) if len(array) else np.nan
        method_used = "percentile"
        estimate = point[column]
        jack = np.asarray(jackknife[column], dtype=float)
        if (
            ci_method == "bca"
            and len(array)
            and np.isfinite(estimate)
            and len(jack) == len(clusters)
            and np.isfinite(jack).all()
            and len(jack) >= 3
        ):
            proportion = (np.sum(array < estimate) + 0.5 * np.sum(array == estimate)) / len(array)
            proportion = float(np.clip(proportion, 1 / (2 * len(array)), 1 - 1 / (2 * len(array))))
            z0 = float(stats.norm.ppf(proportion))
            deviations = float(jack.mean()) - jack
            denominator = 6 * float(np.sum(deviations**2) ** 1.5)
            acceleration = ratio(float(np.sum(deviations**3)), denominator, 0.0)
            adjusted = []
            for alpha in (tail, 1 - tail):
                z_alpha = float(stats.norm.ppf(alpha))
                divisor = 1 - acceleration * (z0 + z_alpha)
                if abs(divisor) <= 1e-12:
                    adjusted = []
                    break
                adjusted.append(float(stats.norm.cdf(z0 + (z0 + z_alpha) / divisor)))
            if len(adjusted) == 2 and np.isfinite(adjusted).all() and adjusted[0] <= adjusted[1]:
                low, high = map(float, np.quantile(array, np.clip(adjusted, 0, 1)))
                method_used = "bca"
        result[column] = {
            "estimate": estimate,
            "low": low,
            "high": high,
            "bootstrap_se": float(array.std(ddof=1)) if len(array) > 1 else np.nan,
            "valid_draws": int(len(array)),
            "requested_draws": int(draws),
            "confidence": float(confidence),
            "method": method_used,
        }
    return result


def validate_ablation_alignment(
    table: pd.DataFrame,
    *,
    arm_column: str = "arm",
    unit_column: str = "unit_id",
    patient_column: str = "patient_id",
    label_column: str = "label",
) -> list[str]:
    """Require every ablation arm to evaluate the same labelled units."""
    required = {arm_column, unit_column, patient_column, label_column}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Missing ablation columns: {sorted(missing)}")
    if table.duplicated([arm_column, unit_column]).any():
        raise ValueError("Each arm/unit pair must occur exactly once.")
    arms = sorted(table[arm_column].astype(str).unique())
    if len(arms) < 2:
        raise ValueError("At least two ablation arms are required.")
    reference = None
    for arm in arms:
        subset = table.loc[
            table[arm_column].astype(str) == arm,
            [unit_column, patient_column, label_column],
        ].sort_values(unit_column)
        records = list(subset.itertuples(index=False, name=None))
        if reference is None:
            reference = records
        elif records != reference:
            raise ValueError("Ablation arms must contain identical units, patients, and labels.")
    return arms


def evaluate_ablation_arms(
    table: pd.DataFrame,
    *,
    thresholds: float | Mapping[str, float] = 0.5,
    arm_column: str = "arm",
    probability_column: str = "probability",
    evaluable_column: str = "evaluable",
) -> dict[str, dict]:
    """Evaluate aligned predicted-ROI, oracle, and shortcut-control arms."""
    arms = validate_ablation_alignment(table, arm_column=arm_column)
    results = {}
    for arm in arms:
        group = table[table[arm_column].astype(str) == arm]
        threshold = float(thresholds[arm]) if isinstance(thresholds, Mapping) else float(thresholds)
        results[arm] = selective_binary_metrics(
            group.label,
            group[probability_column],
            evaluable=group[evaluable_column],
            threshold=threshold,
            localized_success=(group.localized if "localized" in group else None),
        )
    return results


def paired_cluster_bootstrap_difference(
    table: pd.DataFrame,
    arm_a: str,
    arm_b: str,
    *,
    metric_path: str = "failure_inclusive.balanced_accuracy",
    thresholds: float | Mapping[str, float] = 0.5,
    draws: int = 2000,
    seed: int = 2718,
    arm_column: str = "arm",
    unit_column: str = "unit_id",
) -> dict:
    """Paired whole-patient bootstrap difference ``arm_a - arm_b``."""
    arms = validate_ablation_alignment(table, arm_column=arm_column, unit_column=unit_column)
    if arm_a not in arms or arm_b not in arms:
        raise ValueError("Requested comparison arm is absent.")
    if draws < 1:
        raise ValueError("draws must be positive.")

    def threshold_for(arm: str) -> float:
        return float(thresholds[arm]) if isinstance(thresholds, Mapping) else float(thresholds)

    indexed = {
        arm: table[table[arm_column].astype(str) == arm].set_index(unit_column).sort_index()
        for arm in (arm_a, arm_b)
    }

    def arm_metric(frame: pd.DataFrame, arm: str) -> float:
        metrics = selective_binary_metrics(
            frame.label,
            frame.probability,
            evaluable=frame.evaluable,
            threshold=threshold_for(arm),
        )
        return _metric_at_path(metrics, metric_path)

    estimate = arm_metric(indexed[arm_a], arm_a) - arm_metric(indexed[arm_b], arm_b)
    base = indexed[arm_a].reset_index()
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        indices = _cluster_draw_indices(
            base,
            rng,
            cluster_column="patient_id",
            label_column="label",
            stratified=True,
        )
        unit_ids = base.iloc[indices][unit_column].to_numpy()
        sample_a = indexed[arm_a].loc[unit_ids]
        sample_b = indexed[arm_b].loc[unit_ids]
        difference = arm_metric(sample_a, arm_a) - arm_metric(sample_b, arm_b)
        if np.isfinite(difference):
            values.append(difference)
    array = np.asarray(values, dtype=float)
    low = float(np.percentile(array, 2.5)) if len(array) else np.nan
    high = float(np.percentile(array, 97.5)) if len(array) else np.nan
    ci_method = "percentile"
    # BCa acceleration is estimated by deleting one whole patient at a time.
    # If a leave-one-patient sample makes the metric undefined, retain the
    # explicitly declared percentile fallback.
    clusters = base.patient_id.drop_duplicates().to_numpy()
    jackknife: list[float] = []
    if len(clusters) >= 3 and len(array) and np.isfinite(estimate):
        for cluster in clusters:
            keep_units = base.loc[base.patient_id != cluster, unit_column].to_numpy()
            try:
                difference = (
                    arm_metric(indexed[arm_a].loc[keep_units], arm_a)
                    - arm_metric(indexed[arm_b].loc[keep_units], arm_b)
                )
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                difference = np.nan
            jackknife.append(float(difference))
        jack = np.asarray(jackknife, dtype=float)
        if len(jack) == len(clusters) and np.isfinite(jack).all():
            proportion = (
                np.sum(array < estimate) + 0.5 * np.sum(array == estimate)
            ) / len(array)
            proportion = float(
                np.clip(proportion, 1 / (2 * len(array)), 1 - 1 / (2 * len(array)))
            )
            z0 = float(stats.norm.ppf(proportion))
            jack_mean = float(jack.mean())
            deviations = jack_mean - jack
            denominator = 6 * float(np.sum(deviations**2) ** 1.5)
            acceleration = ratio(float(np.sum(deviations**3)), denominator, 0.0)
            adjusted: list[float] = []
            for alpha_value in (0.025, 0.975):
                z_alpha = float(stats.norm.ppf(alpha_value))
                divisor = 1 - acceleration * (z0 + z_alpha)
                if abs(divisor) <= 1e-12:
                    adjusted = []
                    break
                adjusted.append(
                    float(stats.norm.cdf(z0 + (z0 + z_alpha) / divisor))
                )
            if (
                len(adjusted) == 2
                and np.isfinite(adjusted).all()
                and adjusted[0] <= adjusted[1]
            ):
                low, high = map(float, np.quantile(array, np.clip(adjusted, 0, 1)))
                ci_method = "bca"
    nonpositive = int((array <= 0).sum())
    nonnegative = int((array >= 0).sum())
    # Plus-one correction prevents an impossible p=0 with finite Monte Carlo
    # draws.  This is an explicitly exploratory sign-tail bootstrap test; the
    # paired cluster effect and CI remain the primary outputs.
    lower_tail = (nonpositive + 1) / (len(array) + 1) if len(array) else np.nan
    upper_tail = (nonnegative + 1) / (len(array) + 1) if len(array) else np.nan
    p_value = min(1.0, 2 * min(lower_tail, upper_tail)) if len(array) else np.nan
    return {
        "arm_a": arm_a,
        "arm_b": arm_b,
        "contrast": f"{arm_a} - {arm_b}",
        "metric": metric_path,
        "estimate": float(estimate),
        "low": low,
        "high": high,
        "bootstrap_se": float(array.std(ddof=1)) if len(array) > 1 else np.nan,
        "valid_draws": int(len(array)),
        "requested_draws": int(draws),
        "confidence": 0.95,
        "ci_method": ci_method,
        "nonpositive_draws": nonpositive,
        "nonnegative_draws": nonnegative,
        "p_value_two_sided_sign_tail": float(p_value),
        "p_value_method": (
            "min(1, 2*min((1+n[delta<=0])/(B+1), "
            "(1+n[delta>=0])/(B+1))); paired patient-cluster bootstrap"
        ),
    }


def _midrank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[start:end] = 0.5 * (start + end - 1) + 1
        start = end
    result = np.empty(len(values), dtype=float)
    result[order] = ranks
    return result


def delong_auc_comparison(
    y_true: Sequence[int],
    probability_a: Sequence[float],
    probability_b: Sequence[float],
    *,
    evaluable_a: Sequence[bool] | None = None,
    evaluable_b: Sequence[bool] | None = None,
) -> dict:
    """Paired DeLong AUROC comparison on the common evaluable subset.

    DeLong assumes independent evaluation units.  For eye-level data, the
    patient-cluster bootstrap remains the primary inference; this result is a
    requested complementary analysis and its assumption is recorded.
    """
    y = _binary_labels(y_true)
    pa = np.asarray(probability_a, dtype=float).reshape(-1)
    pb = np.asarray(probability_b, dtype=float).reshape(-1)
    if len(pa) != len(y) or len(pb) != len(y):
        raise ValueError("Both probability vectors must align with labels.")
    va = np.isfinite(pa) if evaluable_a is None else np.asarray(evaluable_a, dtype=bool)
    vb = np.isfinite(pb) if evaluable_b is None else np.asarray(evaluable_b, dtype=bool)
    if len(va) != len(y) or len(vb) != len(y):
        raise ValueError("Evaluability vectors must align with labels.")
    common = va & vb
    y_common, pa_common, pb_common = y[common], pa[common], pb[common]
    base = {
        "n_common_evaluable": int(common.sum()),
        "n_total": int(len(y)),
        "common_coverage": float(common.mean()),
        "auc_a": np.nan,
        "auc_b": np.nan,
        "difference_a_minus_b": np.nan,
        "standard_error": np.nan,
        "z": np.nan,
        "p_value": np.nan,
        "status": "ok",
        "assumption_note": (
            "Classical DeLong treats rows as independent; use patient-level rows "
            "or regard eye-level p-values as complementary to cluster bootstrap."
        ),
    }
    positives = int(y_common.sum())
    negatives = int(len(y_common) - positives)
    if positives < 2 or negatives < 2:
        base["status"] = "undefined_requires_two_per_class"
        return base
    positive_first = np.argsort(-y_common, kind="mergesort")
    predictions = np.vstack([pa_common[positive_first], pb_common[positive_first]])
    m, n = positives, negatives
    k = predictions.shape[0]
    tx = np.empty_like(predictions, dtype=float)
    ty = np.empty((k, m), dtype=float)
    tz = np.empty((k, n), dtype=float)
    for classifier in range(k):
        tx[classifier] = _midrank(predictions[classifier])
        ty[classifier] = _midrank(predictions[classifier, :m])
        tz[classifier] = _midrank(predictions[classifier, m:])
    aucs = tx[:, :m].sum(axis=1) / (m * n) - (m + 1) / (2 * n)
    v01 = (tx[:, :m] - ty) / n
    v10 = 1 - (tx[:, m:] - tz) / m
    covariance = np.atleast_2d(np.cov(v01, bias=False)) / m + np.atleast_2d(
        np.cov(v10, bias=False)
    ) / n
    contrast = np.array([1.0, -1.0])
    variance = float(contrast @ covariance @ contrast)
    difference = float(aucs[0] - aucs[1])
    base.update(
        {
            "auc_a": float(aucs[0]),
            "auc_b": float(aucs[1]),
            "difference_a_minus_b": difference,
        }
    )
    if variance <= 0 or not np.isfinite(variance):
        base["status"] = "undefined_zero_or_nonfinite_variance"
        return base
    standard_error = math.sqrt(variance)
    z = difference / standard_error
    base.update(
        {
            "standard_error": standard_error,
            "z": float(z),
            "p_value": float(2 * stats.norm.sf(abs(z))),
        }
    )
    return base


def mcnemar_exact_comparison(
    y_true: Sequence[int],
    probability_a: Sequence[float],
    probability_b: Sequence[float],
    *,
    threshold_a: float = 0.5,
    threshold_b: float = 0.5,
    evaluable_a: Sequence[bool] | None = None,
    evaluable_b: Sequence[bool] | None = None,
) -> dict:
    """Exact paired McNemar test among units evaluable under both methods."""
    y = _binary_labels(y_true)
    pa = np.asarray(probability_a, dtype=float).reshape(-1)
    pb = np.asarray(probability_b, dtype=float).reshape(-1)
    if len(pa) != len(y) or len(pb) != len(y):
        raise ValueError("Both probability vectors must align with labels.")
    va = np.isfinite(pa) if evaluable_a is None else np.asarray(evaluable_a, dtype=bool)
    vb = np.isfinite(pb) if evaluable_b is None else np.asarray(evaluable_b, dtype=bool)
    common = va & vb
    if not int(common.sum()):
        return {
            "n_total": int(len(y)),
            "n_common_evaluable": 0,
            "common_coverage": 0.0,
            "a_correct_b_wrong": 0,
            "a_wrong_b_correct": 0,
            "discordant_pairs": 0,
            "conditional_accuracy_difference_a_minus_b": np.nan,
            "p_value_exact_two_sided": np.nan,
            "status": "undefined_no_common_evaluable",
            "assumption_note": (
                "No unit was evaluable under both methods; McNemar is undefined."
            ),
        }
    correct_a = (pa[common] >= threshold_a).astype(int) == y[common]
    correct_b = (pb[common] >= threshold_b).astype(int) == y[common]
    a_correct_b_wrong = int((correct_a & ~correct_b).sum())
    a_wrong_b_correct = int((~correct_a & correct_b).sum())
    discordant = a_correct_b_wrong + a_wrong_b_correct
    p_value = (
        float(stats.binomtest(min(a_correct_b_wrong, a_wrong_b_correct), discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    return {
        "n_total": int(len(y)),
        "n_common_evaluable": int(common.sum()),
        "common_coverage": float(common.mean()),
        "a_correct_b_wrong": a_correct_b_wrong,
        "a_wrong_b_correct": a_wrong_b_correct,
        "discordant_pairs": discordant,
        "conditional_accuracy_difference_a_minus_b": ratio(
            int(correct_a.sum()) - int(correct_b.sum()), int(common.sum())
        ),
        "p_value_exact_two_sided": p_value,
        "status": "ok",
        "assumption_note": (
            "Run at patient level for independent pairs; eye-level inference does "
            "not account for two-eye within-patient clustering."
        ),
    }


def holm_adjust(
    p_values: Sequence[float], *, alpha: float = 0.05,
    planned_family_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Holm adjustment, optionally preserving a preregistered family size.

    Undefined hypotheses remain NaN/non-rejected.  When ``planned_family_size``
    is supplied they still consume their preregistered place in the multiplicity
    factor rather than silently shrinking the family.
    """
    values = np.asarray(p_values, dtype=float).reshape(-1)
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1.")
    finite_indices = np.flatnonzero(np.isfinite(values))
    adjusted = np.full(len(values), np.nan)
    rejected = np.zeros(len(values), dtype=bool)
    if planned_family_size is not None:
        if isinstance(planned_family_size, bool) or int(planned_family_size) < len(values):
            raise ValueError("planned_family_size must be at least the number of hypotheses")
        family_size = int(planned_family_size)
    else:
        family_size = int(len(finite_indices))
    if not len(finite_indices):
        return adjusted, rejected
    order = finite_indices[np.argsort(values[finite_indices], kind="mergesort")]
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (family_size - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    rejected[finite_indices] = adjusted[finite_indices] <= alpha
    return adjusted, rejected


def summarize_across_seeds(
    results: pd.DataFrame,
    *,
    metric_columns: Sequence[str],
    group_columns: Sequence[str] = ("model", "level", "scope", "arm"),
    seed_column: str = "seed",
    expected_seeds: int | None = 5,
) -> pd.DataFrame:
    """Descriptive mean +/- sample SD across seeds; never pools test rows."""
    required = {seed_column, *metric_columns}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Missing seed-summary columns: {sorted(missing)}")
    groups = [column for column in group_columns if column in results]
    if not groups:
        iterator = [((), results)]
    else:
        iterator = results.groupby(groups, dropna=False, sort=True)
    rows = []
    for key, group in iterator:
        key = key if isinstance(key, tuple) else (key,)
        seeds = group[seed_column].nunique()
        if group.duplicated(seed_column).any():
            raise ValueError("Each summary group must have at most one row per seed.")
        if expected_seeds is not None and seeds != expected_seeds:
            raise ValueError(
                f"Expected {expected_seeds} unique seeds, observed {seeds}; do not hide missing runs."
            )
        row = {column: value for column, value in zip(groups, key, strict=True)}
        row["n_seeds"] = int(seeds)
        for metric in metric_columns:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
            row[f"{metric}_min"] = float(values.min()) if len(values) else np.nan
            row[f"{metric}_max"] = float(values.max()) if len(values) else np.nan
            row[f"{metric}_valid_seeds"] = int(len(values))
        rows.append(row)
    return pd.DataFrame(rows)
