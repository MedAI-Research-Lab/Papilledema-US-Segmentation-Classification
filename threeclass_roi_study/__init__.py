"""Leakage-safe three-class predicted-ROI study utilities.

The package is intentionally separate from :mod:`predicted_roi_study`: the
published binary experiment remains immutable while the multiclass experiment
gets its own protocol, locks, receipts, and output namespace.
"""

from .metrics import (
    ABSTAIN,
    CLASS_NAMES,
    N_CLASSES,
    aggregate_eyes_to_patients,
    aggregate_frames_to_eyes,
    apply_temperature_scaling,
    apply_temperature_to_probabilities,
    clean_json,
    conditional_multiclass_metrics,
    confusion_matrix_3x4,
    fit_temperature_on_probabilities,
    fit_temperature_scaling,
    multiclass_risk_coverage_curve,
    patient_cluster_bootstrap_ci,
    selective_multiclass_metrics,
)

__all__ = [
    "ABSTAIN",
    "CLASS_NAMES",
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
    "selective_multiclass_metrics",
]
