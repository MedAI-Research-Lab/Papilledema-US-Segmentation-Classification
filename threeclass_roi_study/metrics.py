"""Auditable metrics for the strict three-class predicted-ROI pipeline.

The three diagnostic classes are ``normal``, ``papilledema``, and
``pseudopapilledema``.  Diagnostic probabilities exist only when the frozen
segmentation/ROI quality gate succeeds.  Discrimination and calibration are
therefore reported *conditional on evaluability*, while failure-aware
classification metrics retain the complete intended cohort and count an
abstention as an incorrect outcome.

No function in this module is permitted to invent a probability for an
abstained unit.  In particular, invalid frames/eyes must carry ``NaN`` in all
three probability columns.  This invariant prevents a hidden whole-image or
single-eye fallback.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import optimize, special, stats
from sklearn import metrics as skm


ABSTAIN = 3
N_CLASSES = 3
CLASS_LABELS = (0, 1, 2)
CLASS_NAMES = ("normal", "papilledema", "pseudopapilledema")
DEFAULT_PROBABILITY_COLUMNS = ("probability_0", "probability_1", "probability_2")


def _ratio(numerator: float, denominator: float, empty: float = np.nan) -> float:
    return float(numerator / denominator) if denominator else float(empty)


def _safe_mean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(array.mean()) if len(array) else np.nan


def clean_json(value):
    """Recursively convert metric output to strict-JSON-compatible values.

    Undefined metrics become ``None`` rather than non-standard ``NaN``.  The
    scientific functions retain ``NaN`` internally so bootstrap arithmetic can
    distinguish undefined estimates; exporters should call this function just
    before ``json.dump(..., allow_nan=False)``.
    """

    if isinstance(value, Mapping):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return [clean_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if value is pd.NA or value is None:
        return None
    return value


def _labels(y_true: Sequence[int], *, allow_empty: bool = False) -> np.ndarray:
    raw = np.asarray(y_true)
    if raw.ndim != 1:
        raw = raw.reshape(-1)
    if not allow_empty and raw.size == 0:
        raise ValueError("At least one labelled observation is required.")
    try:
        numeric = raw.astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("Labels must be the integers 0, 1, and 2.") from error
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError("Labels must be finite integers.")
    y = numeric.astype(int)
    if not np.isin(y, CLASS_LABELS).all():
        raise ValueError("Three-class labels must contain only 0, 1, and 2.")
    return y


def _predictions(prediction: Sequence[int]) -> np.ndarray:
    raw = np.asarray(prediction)
    if raw.ndim != 1:
        raw = raw.reshape(-1)
    try:
        numeric = raw.astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("Predictions must be 0, 1, 2, or ABSTAIN (3).") from error
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError("Predictions must be finite integers.")
    pred = numeric.astype(int)
    if not np.isin(pred, (*CLASS_LABELS, ABSTAIN)).all():
        raise ValueError("Predictions must be 0, 1, 2, or ABSTAIN (3).")
    return pred


def _probabilities(
    probability: Sequence[Sequence[float]] | np.ndarray,
    *,
    allow_empty: bool = False,
    atol: float = 1e-6,
) -> np.ndarray:
    p = np.asarray(probability, dtype=float)
    if p.ndim != 2 or p.shape[1] != N_CLASSES:
        raise ValueError(f"Probabilities must have shape [N,{N_CLASSES}].")
    if not allow_empty and p.shape[0] == 0:
        raise ValueError("At least one probability vector is required.")
    if not np.isfinite(p).all():
        raise ValueError("Probabilities must be finite.")
    if np.any((p < -atol) | (p > 1 + atol)):
        raise ValueError("Probabilities must lie in [0, 1].")
    if not np.allclose(p.sum(axis=1), 1.0, rtol=0, atol=atol):
        raise ValueError("Every probability vector must sum to one.")
    return np.clip(p, 0.0, 1.0)


def _logits(raw_logits: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    logits = np.asarray(raw_logits, dtype=float)
    if logits.ndim != 2 or logits.shape[1] != N_CLASSES:
        raise ValueError(f"Multiclass logits must have shape [N,{N_CLASSES}].")
    if logits.shape[0] == 0 or not np.isfinite(logits).all():
        raise ValueError("Raw logits must be non-empty and finite.")
    return logits


def _strict_boolean(values: pd.Series, *, name: str) -> np.ndarray:
    if values.isna().any():
        raise ValueError(f"{name} must not contain missing values.")
    raw = values.to_numpy(dtype=object, copy=False)
    if any(not isinstance(value, (bool, np.bool_)) for value in raw):
        raise ValueError(f"{name} must contain genuine boolean values.")
    return np.asarray(raw, dtype=bool)


def _identity_value(group: pd.DataFrame, column: str):
    if column not in group:
        raise ValueError(f"Required column is missing: {column}")
    if group[column].nunique(dropna=False) != 1:
        raise ValueError(f"Inconsistent {column} within an aggregation unit.")
    return group[column].iloc[0]


def _class_name(index: int, class_names: Sequence[str]) -> str:
    if len(class_names) != N_CLASSES or len(set(map(str, class_names))) != N_CLASSES:
        raise ValueError("class_names must contain three unique names.")
    return str(class_names[index])


def apply_temperature_scaling(
    raw_logits: Sequence[Sequence[float]] | np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Apply a previously locked scalar temperature and return softmax vectors."""

    logits = _logits(raw_logits)
    if not np.isfinite(temperature) or float(temperature) <= 0:
        raise ValueError("temperature must be finite and positive.")
    return special.softmax(logits / float(temperature), axis=1)


def apply_temperature_to_probabilities(
    raw_probability: Sequence[Sequence[float]] | np.ndarray,
    temperature: float,
    *,
    epsilon: float = 1e-7,
) -> np.ndarray:
    """Apply a locked temperature to already aggregated probability vectors."""

    if not 0 < epsilon < 1 / N_CLASSES:
        raise ValueError("epsilon must lie in (0, 1/3).")
    probability = _probabilities(raw_probability)
    return apply_temperature_scaling(
        np.log(np.clip(probability, epsilon, 1.0)), temperature
    )


def fit_temperature_scaling(
    y_true: Sequence[int],
    raw_logits: Sequence[Sequence[float]] | np.ndarray,
    *,
    min_temperature: float = 0.05,
    max_temperature: float = 20.0,
    require_all_classes: bool = True,
) -> dict:
    """Fit one scalar temperature by validation-only multinomial NLL.

    The caller must persist the returned lock and apply it unchanged to test
    logits.  Missing validation classes produce an explicit unavailable result;
    no identity-temperature fallback is silently substituted.
    """

    y = _labels(y_true)
    logits = _logits(raw_logits)
    if len(y) != len(logits):
        raise ValueError("Labels and logits must have equal length.")
    if not 0 < min_temperature < max_temperature:
        raise ValueError("Temperature bounds must satisfy 0 < min < max.")
    observed = set(np.unique(y).tolist())
    missing = sorted(set(CLASS_LABELS) - observed)

    def nll_for_temperature(temperature: float) -> float:
        log_probability = special.log_softmax(logits / float(temperature), axis=1)
        return float(-np.mean(log_probability[np.arange(len(y)), y]))

    base = {
        "n": int(len(y)),
        "class_counts": {str(c): int((y == c).sum()) for c in CLASS_LABELS},
        "missing_classes": missing,
        "nll_before": nll_for_temperature(1.0),
        "bounds": [float(min_temperature), float(max_temperature)],
        "fit_partition_required": "validation_only",
        "method": "scalar_temperature_multinomial_nll",
    }
    if require_all_classes and missing:
        return {
            **base,
            "temperature": np.nan,
            "nll_after": np.nan,
            "status": "unavailable_missing_validation_class_no_default_temperature",
            "test_policy": "halt_before_global_test_access_no_default_temperature",
        }

    log_bounds = (math.log(min_temperature), math.log(max_temperature))

    def objective(log_temperature: float) -> float:
        return nll_for_temperature(math.exp(float(log_temperature)))

    fitted = optimize.minimize_scalar(objective, bounds=log_bounds, method="bounded")
    temperature = float(math.exp(float(fitted.x)))
    if not fitted.success or not np.isfinite(temperature):
        return {
            **base,
            "temperature": np.nan,
            "nll_after": np.nan,
            "status": "unavailable_optimizer_failure_no_default_temperature",
            "optimizer_message": str(fitted.message),
            "test_policy": "halt_before_global_test_access_no_default_temperature",
        }
    tolerance = 1e-4
    hit_bound = (
        abs(temperature - min_temperature) <= tolerance * min_temperature
        or abs(temperature - max_temperature) <= tolerance * max_temperature
    )
    return {
        **base,
        "temperature": temperature,
        "nll_after": nll_for_temperature(temperature),
        "status": "ok_bound_hit" if hit_bound else "ok",
        "optimizer_message": str(fitted.message),
        "test_policy": "apply_locked_temperature_then_argmax",
    }


def fit_temperature_on_probabilities(
    y_true: Sequence[int],
    raw_probability: Sequence[Sequence[float]] | np.ndarray,
    *,
    epsilon: float = 1e-7,
    min_temperature: float = 0.05,
    max_temperature: float = 20.0,
    require_all_classes: bool = True,
) -> dict:
    """Fit temperature to final-unit vectors using log-probabilities as logits."""

    if not 0 < epsilon < 1 / N_CLASSES:
        raise ValueError("epsilon must lie in (0, 1/3).")
    p = _probabilities(raw_probability)
    logits = np.log(np.clip(p, epsilon, 1.0))
    result = fit_temperature_scaling(
        y_true,
        logits,
        min_temperature=min_temperature,
        max_temperature=max_temperature,
        require_all_classes=require_all_classes,
    )
    result["input"] = "log_of_clipped_final_unit_probability_vector"
    result["epsilon"] = float(epsilon)
    return result


def confusion_matrix_3x4(
    y_true: Sequence[int], prediction: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Return counts/proportions with columns class 0, class 1, class 2, abstain."""

    y = _labels(y_true)
    pred = _predictions(prediction)
    if len(y) != len(pred):
        raise ValueError("Labels and predictions must have equal length.")
    matrix = np.zeros((N_CLASSES, N_CLASSES + 1), dtype=int)
    column = {0: 0, 1: 1, 2: 2, ABSTAIN: 3}
    for actual, estimated in zip(y, pred, strict=True):
        matrix[int(actual), column[int(estimated)]] += 1
    row_sum = matrix.sum(axis=1, keepdims=True)
    row_proportions = np.divide(
        matrix,
        row_sum,
        out=np.full(matrix.shape, np.nan, dtype=float),
        where=row_sum > 0,
    )
    return matrix, row_proportions


def _binary_calibration_table(
    y_binary: np.ndarray,
    probability: np.ndarray,
    *,
    n_bins: int,
    strategy: str,
) -> pd.DataFrame:
    if n_bins < 1:
        raise ValueError("n_bins must be positive.")
    if strategy not in {"uniform", "quantile"}:
        raise ValueError("strategy must be 'uniform' or 'quantile'.")
    rows: list[dict] = []
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        assignment = np.clip(np.searchsorted(edges, probability, side="right") - 1, 0, n_bins - 1)
        groups = [
            (index, np.flatnonzero(assignment == index), edges[index], edges[index + 1])
            for index in range(n_bins)
        ]
    else:
        order = np.argsort(probability, kind="mergesort")
        chunks = np.array_split(order, min(n_bins, len(order)))
        groups = [
            (
                index,
                indices,
                float(probability[indices].min()) if len(indices) else np.nan,
                float(probability[indices].max()) if len(indices) else np.nan,
            )
            for index, indices in enumerate(chunks)
        ]
    for index, indices, lower, upper in groups:
        count = int(len(indices))
        confidence = float(probability[indices].mean()) if count else np.nan
        frequency = float(y_binary[indices].mean()) if count else np.nan
        rows.append(
            {
                "bin": int(index),
                "lower": float(lower),
                "upper": float(upper),
                "n": count,
                "mean_probability": confidence,
                "observed_fraction": frequency,
                "absolute_gap": abs(confidence - frequency) if count else np.nan,
                "strategy": strategy,
            }
        )
    return pd.DataFrame(rows)


def _ece(curve: pd.DataFrame) -> float:
    total = int(curve["n"].sum())
    if not total:
        return np.nan
    return float((curve["n"] * curve["absolute_gap"].fillna(0)).sum() / total)


def _classwise_statistics(
    y: np.ndarray,
    p: np.ndarray,
    prediction: np.ndarray,
    *,
    class_index: int,
    n_calibration_bins: int,
) -> dict:
    actual = y == class_index
    estimated = prediction == class_index
    tp = int((actual & estimated).sum())
    fn = int((actual & ~estimated).sum())
    fp = int((~actual & estimated).sum())
    tn = int((~actual & ~estimated).sum())
    recall = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    precision = _ratio(tp, tp + fp, 0.0)
    binary_target = actual.astype(int)
    clipped = np.clip(p[:, class_index], 1e-7, 1 - 1e-7)
    both_binary_classes = np.unique(binary_target).size == 2
    uniform = _binary_calibration_table(
        binary_target,
        p[:, class_index],
        n_bins=n_calibration_bins,
        strategy="uniform",
    )
    adaptive = _binary_calibration_table(
        binary_target,
        p[:, class_index],
        n_bins=n_calibration_bins,
        strategy="quantile",
    )
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "support": int(actual.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "recall": recall,
        "sensitivity": recall,
        "specificity": specificity,
        "precision": precision,
        "f1": _ratio(2 * tp, 2 * tp + fp + fn, 0.0),
        "balanced_accuracy_ovr": (
            float((recall + specificity) / 2)
            if np.isfinite(recall) and np.isfinite(specificity)
            else np.nan
        ),
        "mcc_ovr": _ratio(tp * tn - fp * fn, denominator, 0.0),
        "auroc_ovr": (
            float(skm.roc_auc_score(binary_target, p[:, class_index]))
            if both_binary_classes
            else np.nan
        ),
        "average_precision_ovr": (
            float(skm.average_precision_score(binary_target, p[:, class_index]))
            if actual.any()
            else np.nan
        ),
        "brier_ovr": float(np.mean((p[:, class_index] - binary_target) ** 2)),
        "nll_ovr": float(
            -np.mean(
                binary_target * np.log(clipped)
                + (1 - binary_target) * np.log(1 - clipped)
            )
        ),
        "nll_true_class": (
            float(-np.mean(np.log(clipped[actual]))) if actual.any() else np.nan
        ),
        "ece_ovr": _ece(uniform),
        "adaptive_ece_ovr": _ece(adaptive),
        "calibration_bins_uniform": uniform.to_dict(orient="records"),
        "calibration_bins_quantile": adaptive.to_dict(orient="records"),
    }


def _binary_discrimination(y_binary: np.ndarray, score: np.ndarray) -> dict:
    """Small auditable helper for predeclared clinically derived contrasts."""

    y_binary = np.asarray(y_binary, dtype=int).reshape(-1)
    score = np.asarray(score, dtype=float).reshape(-1)
    if len(y_binary) != len(score) or not np.isin(y_binary, (0, 1)).all():
        raise ValueError("Binary derived endpoint inputs are invalid.")
    if not np.isfinite(score).all():
        raise ValueError("Derived endpoint scores must be finite.")
    return {
        "n": int(len(y_binary)),
        "positive": int(y_binary.sum()),
        "negative": int((1 - y_binary).sum()),
        "auroc": (
            float(skm.roc_auc_score(y_binary, score))
            if np.unique(y_binary).size == 2
            else np.nan
        ),
        "average_precision": (
            float(skm.average_precision_score(y_binary, score))
            if np.any(y_binary == 1)
            else np.nan
        ),
    }


def conditional_multiclass_metrics(
    y_true: Sequence[int],
    probability: Sequence[Sequence[float]] | np.ndarray,
    *,
    class_names: Sequence[str] = CLASS_NAMES,
    n_calibration_bins: int = 10,
) -> dict:
    """Three-class metrics for evaluable units only.

    ``multiclass_brier`` uses the conventional mean sum of squared errors over
    all three class probabilities.  Classwise Brier/NLL/ECE are one-vs-rest;
    ``multiclass_nll`` is multinomial cross-entropy.  Macro values are unweighted
    means of the three classwise values.
    """

    y = _labels(y_true)
    p = _probabilities(probability)
    if len(y) != len(p):
        raise ValueError("Labels and probabilities must have equal length.")
    names = tuple(_class_name(index, class_names) for index in CLASS_LABELS)
    prediction = np.argmax(p, axis=1).astype(int)
    confusion = skm.confusion_matrix(y, prediction, labels=CLASS_LABELS)
    per_class = {
        names[index]: {
            "class_index": index,
            **_classwise_statistics(
                y,
                p,
                prediction,
                class_index=index,
                n_calibration_bins=n_calibration_bins,
            ),
        }
        for index in CLASS_LABELS
    }
    per_class_values = list(per_class.values())
    macro = {
        "recall": _safe_mean([item["recall"] for item in per_class_values]),
        "specificity": _safe_mean([item["specificity"] for item in per_class_values]),
        "precision": _safe_mean([item["precision"] for item in per_class_values]),
        "f1": _safe_mean([item["f1"] for item in per_class_values]),
        "balanced_accuracy_ovr": _safe_mean(
            [item["balanced_accuracy_ovr"] for item in per_class_values]
        ),
        "mcc_ovr": _safe_mean([item["mcc_ovr"] for item in per_class_values]),
        "auroc_ovr": _safe_mean([item["auroc_ovr"] for item in per_class_values]),
        "average_precision_ovr": _safe_mean(
            [item["average_precision_ovr"] for item in per_class_values]
        ),
        "brier_ovr": _safe_mean([item["brier_ovr"] for item in per_class_values]),
        "nll_ovr": _safe_mean([item["nll_ovr"] for item in per_class_values]),
        "nll_true_class": _safe_mean(
            [item["nll_true_class"] for item in per_class_values]
        ),
        "ece_ovr": _safe_mean([item["ece_ovr"] for item in per_class_values]),
        "adaptive_ece_ovr": _safe_mean(
            [item["adaptive_ece_ovr"] for item in per_class_values]
        ),
    }
    one_hot = np.eye(N_CLASSES, dtype=float)[y]
    clipped = np.clip(p, 1e-7, 1.0)
    confidence = p.max(axis=1)
    correct = (prediction == y).astype(int)
    top_uniform = _binary_calibration_table(
        correct,
        confidence,
        n_bins=n_calibration_bins,
        strategy="uniform",
    )
    top_adaptive = _binary_calibration_table(
        correct,
        confidence,
        n_bins=n_calibration_bins,
        strategy="quantile",
    )
    abnormal = (y != 0).astype(int)
    normal_vs_abnormal = _binary_discrimination(abnormal, p[:, 1] + p[:, 2])
    disease = y != 0
    disease_denominator = p[disease, 1] + p[disease, 2]
    disease_score = np.divide(
        p[disease, 2],
        disease_denominator,
        out=np.full(int(disease.sum()), 0.5, dtype=float),
        where=disease_denominator > 0,
    )
    papilledema_vs_pseudopapilledema = _binary_discrimination(
        (y[disease] == 2).astype(int), disease_score
    ) if disease.any() else {
        "n": 0,
        "positive": 0,
        "negative": 0,
        "auroc": np.nan,
        "average_precision": np.nan,
    }
    normal_vs_abnormal.update(
        {
            "positive_definition": "papilledema_or_pseudopapilledema",
            "score_definition": "p_papilledema_plus_p_pseudopapilledema",
        }
    )
    papilledema_vs_pseudopapilledema.update(
        {
            "positive_definition": "pseudopapilledema",
            "population": "reference_papilledema_or_pseudopapilledema_only",
            "score_definition": (
                "p_pseudopapilledema_divided_by_"
                "(p_papilledema_plus_p_pseudopapilledema)"
            ),
        }
    )
    return {
        "n": int(len(y)),
        "class_counts": {names[c]: int((y == c).sum()) for c in CLASS_LABELS},
        "accuracy": float(np.mean(prediction == y)),
        "balanced_accuracy": macro["recall"],
        "macro_f1": macro["f1"],
        "macro_precision": macro["precision"],
        "macro_recall": macro["recall"],
        "multiclass_mcc": float(skm.matthews_corrcoef(y, prediction)),
        "macro_auroc": macro["auroc_ovr"],
        "macro_average_precision": macro["average_precision_ovr"],
        "multiclass_brier": float(np.mean(np.sum((p - one_hot) ** 2, axis=1))),
        "multiclass_nll": float(-np.mean(np.log(clipped[np.arange(len(y)), y]))),
        "top_label_ece": _ece(top_uniform),
        "top_label_adaptive_ece": _ece(top_adaptive),
        "macro_classwise_ece": macro["ece_ovr"],
        "macro_classwise_adaptive_ece": macro["adaptive_ece_ovr"],
        "derived_discrimination": {
            "normal_vs_abnormal": normal_vs_abnormal,
            "papilledema_vs_pseudopapilledema": papilledema_vs_pseudopapilledema,
        },
        "macro": macro,
        "per_class": per_class,
        "prediction": prediction.tolist(),
        "confidence": confidence.tolist(),
        "confusion_matrix_3x3": confusion.tolist(),
        "confusion_matrix_3x3_row_proportions": np.divide(
            confusion,
            confusion.sum(axis=1, keepdims=True),
            out=np.full(confusion.shape, np.nan, dtype=float),
            where=confusion.sum(axis=1, keepdims=True) > 0,
        ).tolist(),
        "top_label_calibration_bins_uniform": top_uniform.to_dict(orient="records"),
        "top_label_calibration_bins_quantile": top_adaptive.to_dict(orient="records"),
        "brier_definition": "mean sum_c (p_c - 1[y=c])^2",
        "calibration_scope": "evaluable_units_only",
    }


def multiclass_risk_coverage_curve(
    y_true: Sequence[int],
    probability: Sequence[Sequence[float]] | np.ndarray,
    *,
    evaluable: Sequence[bool] | None = None,
    include_abstentions_as_failures: bool = True,
) -> pd.DataFrame:
    """Confidence-ranked selective risk and mean-step AURC.

    Evaluable units are ranked by maximum class probability using a stable sort.
    For the failure-aware curve, operational abstentions are appended at the
    lowest confidence and counted as errors.  The complete intended cohort is
    the coverage denominator.
    """

    y = _labels(y_true)
    raw = np.asarray(probability, dtype=float)
    if raw.ndim != 2 or raw.shape != (len(y), N_CLASSES):
        raise ValueError(f"Probabilities must have shape [{len(y)},{N_CLASSES}].")
    valid = np.isfinite(raw).all(axis=1) if evaluable is None else np.asarray(evaluable)
    if valid.ndim != 1 or len(valid) != len(y):
        raise ValueError("evaluable must have one value per observation.")
    if valid.dtype != bool:
        if not all(isinstance(value, (bool, np.bool_)) for value in valid.tolist()):
            raise ValueError("evaluable must contain genuine boolean values.")
        valid = valid.astype(bool)
    if np.any(valid & ~np.isfinite(raw).all(axis=1)):
        raise ValueError("Every evaluable unit must have a finite probability vector.")
    if valid.any():
        _probabilities(raw[valid])

    valid_indices = np.flatnonzero(valid)
    confidence = raw[valid_indices].max(axis=1) if len(valid_indices) else np.array([])
    order = valid_indices[np.argsort(-confidence, kind="mergesort")]
    errors = (
        np.argmax(raw[order], axis=1).astype(int) != y[order]
    ).astype(int) if len(order) else np.array([], dtype=int)
    ordered_confidence = raw[order].max(axis=1) if len(order) else np.array([])
    if include_abstentions_as_failures:
        abstained = np.flatnonzero(~valid)
        order = np.concatenate([order, abstained])
        errors = np.concatenate([errors, np.ones(len(abstained), dtype=int)])
        ordered_confidence = np.concatenate(
            [ordered_confidence, np.full(len(abstained), np.nan)]
        )
        denominator = len(y)
    else:
        denominator = len(valid_indices)
    if not len(order):
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
            "confidence": ordered_confidence,
            "coverage": coverage,
            "risk": cumulative_risk,
            "aurc": aurc,
            "failure_aware": bool(include_abstentions_as_failures),
        }
    )


def _failure_aware_from_matrix(
    matrix: np.ndarray,
    y: np.ndarray,
    prediction: np.ndarray,
    *,
    class_names: Sequence[str],
) -> dict:
    names = tuple(_class_name(index, class_names) for index in CLASS_LABELS)
    per_class: dict[str, dict] = {}
    for c in CLASS_LABELS:
        tp = int(matrix[c, c])
        fn = int(matrix[c, :].sum() - tp)
        fp = int(matrix[:, c].sum() - tp)
        actual_nonclass = int(matrix.sum() - matrix[c, :].sum())
        # Abstention is an operational failure rather than a true negative.
        nonclass_abstain = int(matrix[:, 3].sum() - matrix[c, 3])
        tn = int(actual_nonclass - fp - nonclass_abstain)
        recall = _ratio(tp, tp + fn)
        specificity = _ratio(tn, actual_nonclass)
        per_class[names[c]] = {
            "class_index": c,
            "support": int(matrix[c, :].sum()),
            "tp": tp,
            "fp": fp,
            "fn_including_abstain": fn,
            "tn_excluding_abstain": tn,
            "abstain": int(matrix[c, 3]),
            "recall": recall,
            "sensitivity": recall,
            "specificity": specificity,
            "precision": _ratio(tp, tp + fp, 0.0),
            "f1": _ratio(2 * tp, 2 * tp + fp + fn, 0.0),
            "balanced_accuracy_ovr": _safe_mean([recall, specificity]),
        }
    accuracy = _ratio(sum(matrix[c, c] for c in CLASS_LABELS), len(y))
    macro_recall = _safe_mean([value["recall"] for value in per_class.values()])
    return {
        "accuracy": accuracy,
        "balanced_accuracy": macro_recall,
        "macro_recall": macro_recall,
        "macro_precision": _safe_mean([value["precision"] for value in per_class.values()]),
        "macro_f1": _safe_mean([value["f1"] for value in per_class.values()]),
        "macro_specificity": _safe_mean(
            [value["specificity"] for value in per_class.values()]
        ),
        "multiclass_mcc": float(skm.matthews_corrcoef(y, prediction)),
        "correct": int(sum(matrix[c, c] for c in CLASS_LABELS)),
        "incorrect_or_abstained": int(len(y) - sum(matrix[c, c] for c in CLASS_LABELS)),
        "per_class": per_class,
        "mcc_note": "ABSTAIN=3 is retained as a fourth predicted category with no true row.",
    }


def selective_multiclass_metrics(
    y_true: Sequence[int],
    probability: Sequence[Sequence[float]] | np.ndarray,
    *,
    evaluable: Sequence[bool] | None = None,
    class_names: Sequence[str] = CLASS_NAMES,
    n_calibration_bins: int = 10,
    localized_success: Sequence[bool] | None = None,
) -> dict:
    """Evaluate the three-class selective classifier on the intended cohort."""

    y = _labels(y_true)
    raw = np.asarray(probability, dtype=float)
    if raw.ndim != 2 or raw.shape != (len(y), N_CLASSES):
        raise ValueError(f"Probabilities must have shape [{len(y)},{N_CLASSES}].")
    valid = np.isfinite(raw).all(axis=1) if evaluable is None else np.asarray(evaluable)
    if valid.ndim != 1 or len(valid) != len(y):
        raise ValueError("evaluable must have one value per observation.")
    if valid.dtype != bool:
        if not all(isinstance(value, (bool, np.bool_)) for value in valid.tolist()):
            raise ValueError("evaluable must contain genuine boolean values.")
        valid = valid.astype(bool)
    if np.any(valid & ~np.isfinite(raw).all(axis=1)):
        raise ValueError("Every evaluable unit must have a finite probability vector.")
    if valid.any():
        _probabilities(raw[valid])

    prediction = np.full(len(y), ABSTAIN, dtype=int)
    prediction[valid] = np.argmax(raw[valid], axis=1).astype(int)
    matrix, row_proportions = confusion_matrix_3x4(y, prediction)
    n_covered = int(valid.sum())
    conditional = (
        conditional_multiclass_metrics(
            y[valid],
            raw[valid],
            class_names=class_names,
            n_calibration_bins=n_calibration_bins,
        )
        if n_covered
        else {
            key: np.nan
            for key in (
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "multiclass_mcc",
                "macro_auroc",
                "macro_average_precision",
                "multiclass_brier",
                "multiclass_nll",
                "top_label_ece",
                "macro_classwise_ece",
            )
        }
    )
    if not n_covered:
        conditional.update({"n": 0, "per_class": {}, "macro": {}})
    failure_aware = _failure_aware_from_matrix(
        matrix, y, prediction, class_names=class_names
    )
    failure_curve = multiclass_risk_coverage_curve(
        y,
        raw,
        evaluable=valid,
        include_abstentions_as_failures=True,
    )
    conditional_curve = (
        multiclass_risk_coverage_curve(
            y[valid], raw[valid], include_abstentions_as_failures=False
        )
        if n_covered
        else pd.DataFrame()
    )
    failure_aurc = float(failure_curve.aurc.iloc[0]) if len(failure_curve) else np.nan
    conditional_aurc = (
        float(conditional_curve.aurc.iloc[0]) if len(conditional_curve) else np.nan
    )
    failure_aware["aurc"] = failure_aurc
    names = tuple(_class_name(index, class_names) for index in CLASS_LABELS)
    class_coverage = {
        names[c]: _ratio(int(valid[y == c].sum()), int((y == c).sum()))
        for c in CLASS_LABELS
    }
    result = {
        "n_total": int(len(y)),
        "n_covered": n_covered,
        "n_abstain": int(len(y) - n_covered),
        "coverage": _ratio(n_covered, len(y)),
        "class_conditional_coverage": class_coverage,
        "abstention_rate": _ratio(len(y) - n_covered, len(y)),
        "selective_risk": _ratio(
            int((prediction[valid] != y[valid]).sum()), n_covered
        ),
        "conditional_aurc": conditional_aurc,
        "failure_aware_aurc": failure_aurc,
        "failure_aware_accuracy": failure_aware["accuracy"],
        "failure_aware_balanced_accuracy": failure_aware["balanced_accuracy"],
        "conditional": conditional,
        "failure_aware": failure_aware,
        # Alias retained for consistency with the preceding binary report schema.
        "failure_inclusive": failure_aware,
        "prediction": prediction.tolist(),
        "confusion_matrix_3x4": matrix.tolist(),
        "confusion_matrix_3x4_row_proportions": row_proportions.tolist(),
        "confusion_columns": [
            f"predicted_{names[0]}",
            f"predicted_{names[1]}",
            f"predicted_{names[2]}",
            "abstain",
        ],
        "metric_scope_note": (
            "AUROC/AP/Brier/NLL/ECE are conditional on evaluability; "
            "failure-aware classification and risk-coverage retain abstentions as failures."
        ),
    }
    if localized_success is not None:
        localized = np.asarray(localized_success)
        if localized.ndim != 1 or len(localized) != len(y):
            raise ValueError("localized_success must have one value per observation.")
        if localized.dtype != bool:
            if not all(isinstance(value, (bool, np.bool_)) for value in localized.tolist()):
                raise ValueError("localized_success must contain genuine boolean values.")
            localized = localized.astype(bool)
        joint = valid & localized & (prediction == y)
        result["localized_diagnostic_success"] = {
            "n_success": int(joint.sum()),
            "rate": float(joint.mean()),
            "localized_rate": float(localized.mean()),
            "per_class_rate": {
                names[c]: _ratio(int(joint[y == c].sum()), int((y == c).sum()))
                for c in CLASS_LABELS
            },
            "definition": (
                "correct non-abstained three-class diagnosis and locked "
                "reference-localisation criterion"
            ),
        }
    return result


def _validated_probability_columns(probability_columns: Sequence[str]) -> tuple[str, ...]:
    columns = tuple(map(str, probability_columns))
    if len(columns) != N_CLASSES or len(set(columns)) != N_CLASSES:
        raise ValueError("probability_columns must contain exactly three unique columns.")
    return columns


def aggregate_frames_to_eyes(
    frames: pd.DataFrame,
    *,
    frames_per_eye: int = 7,
    min_valid_frames: int = 4,
    validity_column: str = "roi_valid",
    probability_columns: Sequence[str] = DEFAULT_PROBABILITY_COLUMNS,
    label_column: str = "label_3class",
    reason_column: str = "abstention_reason",
    strict_invalid_probability: bool = True,
) -> pd.DataFrame:
    """Average valid frame probability vectors into strict eye decisions."""

    probability_columns = _validated_probability_columns(probability_columns)
    required = {
        "case_id",
        "patient_id",
        "side",
        "frame_id",
        label_column,
        *probability_columns,
    }
    missing = required - set(frames)
    if missing:
        raise ValueError(f"Missing frame columns: {sorted(missing)}")
    if not 1 <= min_valid_frames <= frames_per_eye:
        raise ValueError("min_valid_frames must be between 1 and frames_per_eye.")

    rows: list[dict] = []
    for (patient_key, case_id, side_key), group in frames.groupby(
        ["patient_id", "case_id", "side"], sort=True
    ):
        if len(group) != frames_per_eye or group["frame_id"].nunique() != frames_per_eye:
            raise ValueError(
                f"Eye {case_id!r} must contain exactly {frames_per_eye} unique frames."
            )
        patient_id = _identity_value(group, "patient_id")
        side = _identity_value(group, "side")
        if str(patient_id) != str(patient_key) or str(side) != str(side_key):
            raise ValueError("Eye grouping identity changed during aggregation.")
        label = int(_identity_value(group, label_column))
        _labels([label])
        probability = group.loc[:, probability_columns].to_numpy(float)
        valid = (
            _strict_boolean(group[validity_column], name=validity_column)
            if validity_column in group
            else np.isfinite(probability).all(axis=1)
        )
        if np.any(valid & ~np.isfinite(probability).all(axis=1)):
            raise ValueError(f"Eye {case_id!r}: valid ROI has no complete probability vector.")
        if valid.any():
            _probabilities(probability[valid])
        if strict_invalid_probability and np.any(
            ~valid & np.isfinite(probability).any(axis=1)
        ):
            raise ValueError(
                f"Eye {case_id!r}: invalid ROI carries probabilities; hidden fallback rejected."
            )
        n_valid = int(valid.sum())
        evaluable = n_valid >= min_valid_frames
        eye_probability = (
            probability[valid].mean(axis=0) if evaluable else np.full(N_CLASSES, np.nan)
        )
        if evaluable:
            eye_probability = eye_probability / eye_probability.sum()
        prediction = int(np.argmax(eye_probability)) if evaluable else ABSTAIN
        row = {
            "case_id": case_id,
            "patient_id": patient_id,
            "side": side,
            label_column: label,
            "n_frames": int(len(group)),
            "n_valid_frames": n_valid,
            "valid_frame_fraction": n_valid / frames_per_eye,
            "evaluable": bool(evaluable),
            "prediction": prediction,
            "confidence": float(np.max(eye_probability)) if evaluable else np.nan,
            "abstention_reason": "" if evaluable else "insufficient_valid_frames",
        }
        row.update(
            {column: float(eye_probability[index]) for index, column in enumerate(probability_columns)}
        )
        for optional_identity in ("split", "seed", "model", "strategy", "arm"):
            if optional_identity in group:
                row[optional_identity] = _identity_value(group, optional_identity)
        if reason_column in group:
            invalid_reasons = (
                group.loc[~valid, reason_column]
                .fillna("unspecified")
                .replace("", "unspecified")
            )
            counts = invalid_reasons.value_counts(sort=False).sort_index()
            row["invalid_frame_reasons"] = ";".join(
                f"{reason}:{int(count)}" for reason, count in counts.items()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_eyes_to_patients(
    eyes: pd.DataFrame,
    *,
    required_sides: Sequence[str] = ("SAG", "SOL"),
    probability_columns: Sequence[str] = DEFAULT_PROBABILITY_COLUMNS,
    label_column: str = "label_3class",
    evaluable_column: str = "evaluable",
) -> pd.DataFrame:
    """Average two eye vectors; if either eye abstains, the patient abstains."""

    probability_columns = _validated_probability_columns(probability_columns)
    required = {
        "patient_id",
        "case_id",
        "side",
        label_column,
        evaluable_column,
        *probability_columns,
    }
    missing = required - set(eyes)
    if missing:
        raise ValueError(f"Missing eye columns: {sorted(missing)}")
    expected_sides = set(required_sides)
    if len(expected_sides) != len(required_sides):
        raise ValueError("required_sides must be unique.")
    rows: list[dict] = []
    for patient_id, group in eyes.groupby("patient_id", sort=True):
        if len(group) != len(required_sides) or set(group["side"]) != expected_sides:
            raise ValueError(
                f"Patient {patient_id!r} must contain exactly the required eyes."
            )
        label = int(_identity_value(group, label_column))
        _labels([label])
        valid = _strict_boolean(group[evaluable_column], name=evaluable_column)
        probability = group.loc[:, probability_columns].to_numpy(float)
        if np.any(valid & ~np.isfinite(probability).all(axis=1)):
            raise ValueError("Evaluable eye lacks a complete probability vector.")
        if valid.any():
            _probabilities(probability[valid])
        if np.any(~valid & np.isfinite(probability).any(axis=1)):
            raise ValueError("Abstained eye carries probabilities; hidden fallback rejected.")
        patient_evaluable = bool(valid.all())
        patient_probability = (
            probability.mean(axis=0)
            if patient_evaluable
            else np.full(N_CLASSES, np.nan)
        )
        if patient_evaluable:
            patient_probability = patient_probability / patient_probability.sum()
        prediction = int(np.argmax(patient_probability)) if patient_evaluable else ABSTAIN
        row = {
            "patient_id": patient_id,
            label_column: label,
            "n_eyes": int(len(group)),
            "n_evaluable_eyes": int(valid.sum()),
            "evaluable": patient_evaluable,
            "prediction": prediction,
            "confidence": float(np.max(patient_probability)) if patient_evaluable else np.nan,
            "abstention_reason": (
                "" if patient_evaluable else "one_or_more_eyes_abstained"
            ),
        }
        row.update(
            {
                column: float(patient_probability[index])
                for index, column in enumerate(probability_columns)
            }
        )
        for optional_identity in ("split", "seed", "model", "strategy", "arm"):
            if optional_identity in group:
                row[optional_identity] = _identity_value(group, optional_identity)
        rows.append(row)
    return pd.DataFrame(rows)


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
        raise ValueError("Every bootstrap patient cluster must have one consistent label.")
    cluster_labels = cluster_labels.drop_duplicates(cluster_column)
    cluster_labels = cluster_labels.sort_values(
        [label_column, cluster_column],
        kind="stable",
        key=lambda values: values.astype(str),
    )
    indices = {
        cluster: np.asarray(index, dtype=int)
        for cluster, index in table.groupby(cluster_column, sort=False).indices.items()
    }
    sampled: list[np.ndarray] = []
    strata = (
        cluster_labels.groupby(label_column, sort=True)
        if stratified
        else [(None, cluster_labels)]
    )
    for _, stratum in strata:
        clusters = stratum[cluster_column].to_numpy()
        draws = rng.choice(clusters, size=len(clusters), replace=True)
        sampled.extend(indices[cluster] for cluster in draws)
    return np.concatenate(sampled)


DEFAULT_BOOTSTRAP_METRICS = (
    "coverage",
    "selective_risk",
    "conditional_aurc",
    "failure_aware_aurc",
    "conditional.accuracy",
    "conditional.balanced_accuracy",
    "conditional.macro_f1",
    "conditional.multiclass_mcc",
    "conditional.macro_auroc",
    "conditional.macro_average_precision",
    "conditional.multiclass_brier",
    "conditional.multiclass_nll",
    "conditional.top_label_ece",
    "conditional.macro_classwise_ece",
    "conditional.derived_discrimination.normal_vs_abnormal.auroc",
    "conditional.derived_discrimination.normal_vs_abnormal.average_precision",
    "conditional.derived_discrimination.papilledema_vs_pseudopapilledema.auroc",
    "conditional.derived_discrimination.papilledema_vs_pseudopapilledema.average_precision",
    "failure_aware.accuracy",
    "failure_aware.balanced_accuracy",
    "failure_aware.macro_f1",
    "failure_aware.multiclass_mcc",
    "localized_diagnostic_success.rate",
) + tuple(
    path
    for class_name in CLASS_NAMES
    for path in (
        f"class_conditional_coverage.{class_name}",
        f"failure_aware.per_class.{class_name}.recall",
        f"failure_aware.per_class.{class_name}.precision",
        f"failure_aware.per_class.{class_name}.f1",
        f"conditional.per_class.{class_name}.recall",
        f"conditional.per_class.{class_name}.precision",
        f"conditional.per_class.{class_name}.f1",
        f"conditional.per_class.{class_name}.auroc_ovr",
        f"conditional.per_class.{class_name}.average_precision_ovr",
    )
)


def _bootstrap_interval_summary(
    estimate: float,
    distribution: Sequence[float] | np.ndarray,
    jackknife: Sequence[float] | np.ndarray,
    *,
    draws: int,
    seed: int,
    confidence: float,
    ci_method: str,
    resampling_unit: str = "whole_patient_cluster",
    stratified_by_label: bool = True,
) -> dict[str, float | int | bool | str | None]:
    """Summarise a bootstrap distribution with an auditable BCa fallback.

    Non-finite bootstrap replicates are excluded and reported through
    ``valid_draws``. BCa is used only when the point estimate and complete
    leave-one-cluster-out jackknife are identified; otherwise the interval
    visibly falls back to the percentile method.
    """

    if draws < 1:
        raise ValueError("draws must be positive.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between 0 and 1.")
    if ci_method not in {"bca", "percentile"}:
        raise ValueError("ci_method must be 'bca' or 'percentile'.")
    distribution_array = np.asarray(distribution, dtype=float).reshape(-1)
    distribution_array = distribution_array[np.isfinite(distribution_array)]
    jackknife_array = np.asarray(jackknife, dtype=float).reshape(-1)
    tail = (1 - confidence) / 2
    low = (
        float(np.quantile(distribution_array, tail))
        if len(distribution_array)
        else np.nan
    )
    high = (
        float(np.quantile(distribution_array, 1 - tail))
        if len(distribution_array)
        else np.nan
    )
    method_used = "percentile"
    fallback_reason: str | None = None
    if ci_method == "bca":
        bca_identified = (
            len(distribution_array) > 0
            and np.isfinite(estimate)
            and len(jackknife_array) >= 3
            and np.isfinite(jackknife_array).all()
        )
        if bca_identified:
            proportion = (
                np.sum(distribution_array < estimate)
                + 0.5 * np.sum(distribution_array == estimate)
            ) / len(distribution_array)
            proportion = float(
                np.clip(
                    proportion,
                    1 / (2 * len(distribution_array)),
                    1 - 1 / (2 * len(distribution_array)),
                )
            )
            z0 = float(stats.norm.ppf(proportion))
            deviations = float(jackknife_array.mean()) - jackknife_array
            denominator = 6 * float(np.sum(deviations**2) ** 1.5)
            acceleration = _ratio(
                float(np.sum(deviations**3)), denominator, 0.0
            )
            adjusted: list[float] = []
            for alpha in (tail, 1 - tail):
                z_alpha = float(stats.norm.ppf(alpha))
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
                low, high = map(
                    float,
                    np.quantile(distribution_array, np.clip(adjusted, 0, 1)),
                )
                method_used = "bca"
        if method_used != "bca":
            fallback_reason = "bca_not_identified_percentile_used"
    return {
        "estimate": float(estimate),
        "low": low,
        "high": high,
        "bootstrap_se": (
            float(distribution_array.std(ddof=1))
            if len(distribution_array) > 1
            else np.nan
        ),
        "valid_draws": int(len(distribution_array)),
        "requested_draws": int(draws),
        "confidence": float(confidence),
        "requested_method": ci_method,
        "method": method_used,
        "fallback_used": bool(method_used != ci_method),
        "fallback_reason": fallback_reason,
        "resampling_unit": str(resampling_unit),
        "stratified_by_label": bool(stratified_by_label),
        "stratification": (
            "three_class_patient_label" if stratified_by_label else "none"
        ),
        "bootstrap_seed": int(seed),
    }


def patient_cluster_ratio_bootstrap_ci(
    numerator_by_patient: Sequence[float] | np.ndarray,
    denominator_by_patient: Sequence[float] | np.ndarray,
    *,
    draws: int = 5000,
    seed: int = 19051,
    confidence: float = 0.95,
    ci_method: str = "bca",
    stratified_by_label: bool = True,
) -> dict[str, float | int | bool | str | None]:
    """Bootstrap a ratio estimand by resampling whole patient clusters.

    Each input position is one patient's contribution to a numerator and
    denominator. This supports frame- and eye-level estimands without ever
    treating their nested observations as independent sampling units. The
    caller supplies one diagnostic stratum at a time when
    ``stratified_by_label`` is true.
    """

    numerator = np.asarray(numerator_by_patient, dtype=float).reshape(-1)
    denominator = np.asarray(denominator_by_patient, dtype=float).reshape(-1)
    if len(numerator) != len(denominator) or not len(numerator):
        raise ValueError(
            "Patient numerator and denominator arrays must have equal positive length."
        )
    if not np.isfinite(numerator).all() or not np.isfinite(denominator).all():
        raise ValueError("Patient ratio contributions must be finite.")
    if np.any(denominator < 0):
        raise ValueError("Patient ratio denominators cannot be negative.")
    if draws < 1:
        raise ValueError("draws must be positive.")

    total_denominator = float(denominator.sum())
    estimate = _ratio(float(numerator.sum()), total_denominator)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(numerator), size=(draws, len(numerator)))
    sampled_numerator = numerator[indices].sum(axis=1)
    sampled_denominator = denominator[indices].sum(axis=1)
    distribution = np.divide(
        sampled_numerator,
        sampled_denominator,
        out=np.full(draws, np.nan, dtype=float),
        where=sampled_denominator > 0,
    )
    jackknife = np.asarray(
        [
            _ratio(
                float(numerator.sum() - numerator[index]),
                float(denominator.sum() - denominator[index]),
            )
            for index in range(len(numerator))
        ],
        dtype=float,
    )
    return _bootstrap_interval_summary(
        estimate,
        distribution,
        jackknife,
        draws=draws,
        seed=seed,
        confidence=confidence,
        ci_method=ci_method,
        stratified_by_label=stratified_by_label,
    )


def patient_cluster_bootstrap_ci(
    table: pd.DataFrame,
    *,
    draws: int = 5000,
    seed: int = 19051,
    metric_paths: Sequence[str] = DEFAULT_BOOTSTRAP_METRICS,
    cluster_column: str = "patient_id",
    label_column: str = "label_3class",
    probability_columns: Sequence[str] = DEFAULT_PROBABILITY_COLUMNS,
    evaluable_column: str = "evaluable",
    class_names: Sequence[str] = CLASS_NAMES,
    stratified: bool = True,
    n_calibration_bins: int = 10,
    confidence: float = 0.95,
    ci_method: str = "bca",
) -> dict[str, dict]:
    """Whole-patient cluster bootstrap CIs, BCa where identified.

    Patients, never individual eyes/frames, are resampled.  Stratification is by
    the three-class patient label.  If a BCa interval is not identified for a
    metric, the function visibly falls back to a percentile interval.
    """

    if draws < 1:
        raise ValueError("draws must be positive.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between 0 and 1.")
    if ci_method not in {"bca", "percentile"}:
        raise ValueError("ci_method must be 'bca' or 'percentile'.")
    probability_columns = _validated_probability_columns(probability_columns)
    required = {
        cluster_column,
        label_column,
        evaluable_column,
        *probability_columns,
    }
    missing = required - set(table)
    if missing:
        raise ValueError(f"Missing bootstrap columns: {sorted(missing)}")
    canonical_columns = [label_column, cluster_column]
    canonical_columns.extend(
        column
        for column in ("case_id", "side", "frame_id")
        if column in table and column not in canonical_columns
    )
    table = table.sort_values(
        canonical_columns,
        kind="stable",
        key=lambda values: values.astype(str),
    ).reset_index(drop=True)
    _labels(table[label_column])
    if not _strict_boolean(table[evaluable_column], name=evaluable_column).shape == (len(table),):
        raise AssertionError("unreachable")

    def evaluate(sample: pd.DataFrame) -> dict:
        return selective_multiclass_metrics(
            sample[label_column],
            sample.loc[:, probability_columns].to_numpy(float),
            evaluable=sample[evaluable_column].to_numpy(dtype=bool),
            class_names=class_names,
            n_calibration_bins=n_calibration_bins,
            localized_success=(
                sample["localized"].to_numpy(dtype=bool)
                if "localized" in sample
                else None
            ),
        )

    point = evaluate(table)
    values: dict[str, list[float]] = {path: [] for path in metric_paths}
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
            leave_one_patient_out = table[table[cluster_column] != cluster]
            if leave_one_patient_out.empty:
                continue
            metrics = evaluate(leave_one_patient_out)
            for path in metric_paths:
                try:
                    value = _metric_at_path(metrics, path)
                except (KeyError, TypeError, ValueError):
                    value = np.nan
                jackknife[path].append(value)

    result: dict[str, dict] = {}
    for path, observed in values.items():
        distribution = np.asarray(observed, dtype=float)
        try:
            estimate = _metric_at_path(point, path)
        except (KeyError, TypeError, ValueError):
            estimate = np.nan
        jack = np.asarray(jackknife[path], dtype=float)
        # BCa requires a complete leave-one-patient-out jackknife. Supplying
        # an empty jackknife makes any percentile fallback explicit.
        if len(jack) != len(clusters):
            jack = np.asarray([], dtype=float)
        result[path] = _bootstrap_interval_summary(
            estimate,
            distribution,
            jack,
            draws=draws,
            seed=seed,
            confidence=confidence,
            ci_method=ci_method,
            stratified_by_label=stratified,
        )
    return result


__all__ = [
    "ABSTAIN",
    "CLASS_LABELS",
    "CLASS_NAMES",
    "DEFAULT_BOOTSTRAP_METRICS",
    "DEFAULT_PROBABILITY_COLUMNS",
    "N_CLASSES",
    "aggregate_eyes_to_patients",
    "aggregate_frames_to_eyes",
    "apply_temperature_scaling",
    "apply_temperature_to_probabilities",
    "clean_json",
    "conditional_multiclass_metrics",
    "confusion_matrix_3x4",
    "fit_temperature_on_probabilities",
    "fit_temperature_scaling",
    "multiclass_risk_coverage_curve",
    "patient_cluster_bootstrap_ci",
    "patient_cluster_ratio_bootstrap_ci",
    "selective_multiclass_metrics",
]
