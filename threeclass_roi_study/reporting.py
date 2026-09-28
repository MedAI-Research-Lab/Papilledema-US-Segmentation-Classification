"""Publication-facing metadata and output contracts for the three-class study.

The schemas are intentionally explicit and machine readable.  They do not
calculate study metrics; instead they prevent a reporting layer from silently
dropping failure-aware denominators, calibration status, provenance, or the
mandatory disclosure that the internal test patients were previously opened.
"""

from __future__ import annotations

import copy
import itertools
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .config import CLASS_NAMES, CLASS_ORDER, canonical_sha256, config_without_runtime


Q1_METADATA_SCHEMA_VERSION = "1.0.0"
OUTPUT_SCHEMA_VERSION = "1.0.0"
ABSTAIN_LABEL = 3
PREDICTION_LABELS = (0, 1, 2, ABSTAIN_LABEL)


class ReportingSchemaError(ValueError):
    """Raised when publication metadata or an output table violates its contract."""


def _field(
    dtype: str,
    *,
    nullable: bool = False,
    description: str,
    allowed: Sequence[Any] | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "type": dtype,
        "nullable": nullable,
        "description": description,
    }
    if allowed is not None:
        value["allowed"] = list(allowed)
    return value


Q1_METADATA_SCHEMA: dict[str, Any] = {
    "schema_version": Q1_METADATA_SCHEMA_VERSION,
    "title": "CLAIM 2024 and TRIPOD+AI aligned three-class study metadata",
    "required_sections": [
        "study_identity",
        "clinical_context",
        "data",
        "reference_standard",
        "partitioning",
        "frozen_upstream",
        "model_development",
        "aggregation",
        "calibration",
        "abstention",
        "evaluation",
        "statistics",
        "prior_test_use",
        "external_evaluation",
        "reproducibility",
        "limitations",
        "reporting_guidelines",
    ],
    "section_requirements": {
        "study_identity": [
            "study_id",
            "protocol_version",
            "study_design",
            "analysis_status",
            "intended_use",
            "primary_unit",
        ],
        "clinical_context": [
            "target_population",
            "input_modality",
            "care_pathway_role",
            "three_target_classes",
        ],
        "data": [
            "source",
            "patient_count",
            "eye_count",
            "frame_count",
            "patient_counts_by_class",
            "demographic_metadata_status",
        ],
        "reference_standard": [
            "diagnosis_reference_standard",
            "roi_reference_standard",
            "label_blinding_status",
        ],
        "partitioning": [
            "unit",
            "stratification",
            "seed_memberships",
            "within_patient_grouping",
            "duplicate_audit",
        ],
        "frozen_upstream": [
            "source_study_id",
            "source_output",
            "reuse_scope",
            "forbidden_imports",
            "development_receipt_count",
            "test_artifact_deferral",
        ],
        "model_development": [
            "classifier_formulation",
            "classifier_strategies",
            "class_count",
            "loss",
            "early_stopping_monitor",
            "binary_head_reused",
            "architectures",
            "initialization",
            "optimizer",
            "scheduler",
            "batch_size",
            "maximum_epochs",
            "patience",
            "augmentation",
            "mixed_precision",
            "determinism",
            "loss_weighting",
            "loss_weight_normalization",
        ],
        "aggregation": [
            "frame_to_eye",
            "minimum_valid_frames",
            "eye_to_patient",
            "both_eyes_required",
        ],
        "calibration": [
            "method",
            "fit_partition",
            "levels",
            "test_refitting",
        ],
        "abstention": [
            "primary_policy",
            "failure_aware_scoring",
            "uncertainty_policy_role",
        ],
        "evaluation": [
            "primary_estimand",
            "primary_endpoint",
            "primary_population",
            "primary_confusion_matrix",
            "localized_diagnostic_success_definition",
            "mandatory_secondary_endpoints",
        ],
        "statistics": [
            "confidence_level",
            "resampling_unit",
            "seed_summary",
            "overlapping_holdout_warning",
            "small_class_warning",
        ],
        "prior_test_use": [
            "source_test_previously_opened",
            "same_memberships_reused",
            "required_disclosure",
            "confirmatory_claim_allowed",
        ],
        "external_evaluation": [
            "status",
            "required_for_confirmatory_claim",
        ],
        "reproducibility": [
            "config_sha256",
            "manifest_sha256",
            "source_anchor_hashes",
            "artifact_receipts_required",
            "runtime_environment_artifact",
            "code_inventory_artifact",
        ],
        "limitations": [
            "post_hoc_internal_reuse",
            "small_class_counts",
            "seed_overlap",
            "missing_demographics",
        ],
        "reporting_guidelines": [
            "claim",
            "tripod",
            "terminology",
        ],
    },
}


OUTPUT_TABLE_SCHEMAS: dict[str, dict[str, Any]] = {
    "patient_predictions.csv": {
        "grain": "one row per model × seed × classifier strategy × intended test patient",
        "primary_key": ["model", "seed", "classifier_strategy", "patient_id"],
        "columns": {
            "study_id": _field("string", description="Locked three-class study identifier."),
            "model": _field("string", description="Frozen upstream segmenter condition."),
            "seed": _field("integer", description="Locked outer split seed."),
            "classifier_strategy": _field(
                "string",
                description="Three-class classifier strategy.",
                allowed=["model_specific", "standardized_resnet18"],
            ),
            "patient_id": _field("string", description="De-identified patient identifier."),
            "true_label": _field(
                "integer", description="Patient reference-standard class.", allowed=CLASS_ORDER
            ),
            "p_normal_raw": _field("number", nullable=True, description="Raw patient probability for normal."),
            "p_papilledema_raw": _field(
                "number", nullable=True, description="Raw patient probability for papilledema."
            ),
            "p_pseudopapilledema_raw": _field(
                "number", nullable=True, description="Raw patient probability for pseudopapilledema."
            ),
            "p_normal_calibrated": _field(
                "number", nullable=True, description="Calibrated patient probability for normal."
            ),
            "p_papilledema_calibrated": _field(
                "number", nullable=True, description="Calibrated patient probability for papilledema."
            ),
            "p_pseudopapilledema_calibrated": _field(
                "number", nullable=True, description="Calibrated patient probability for pseudopapilledema."
            ),
            "predicted_label": _field(
                "integer",
                description="Operational label; 3 denotes abstention.",
                allowed=PREDICTION_LABELS,
            ),
            "abstained": _field("boolean", description="Whether no three-class label was issued."),
            "abstention_reason": _field(
                "string", nullable=True, description="Locked structural or secondary uncertainty reason."
            ),
            "right_eye_evaluable": _field("boolean", description="Right-eye structural gate status."),
            "left_eye_evaluable": _field("boolean", description="Left-eye structural gate status."),
            "right_valid_frames": _field("integer", description="Valid predicted ROIs in the right eye."),
            "left_valid_frames": _field("integer", description="Valid predicted ROIs in the left eye."),
            "calibration_status": _field(
                "string",
                description="Patient-level scalar-temperature status.",
                allowed=["available", "unavailable"],
            ),
            "localized": _field(
                "boolean",
                description=(
                    "Both eyes have at least four of seven quality-gate-valid "
                    "predicted ROIs with retrospective reference IoU >= 0.5."
                ),
            ),
            "localized_diagnostic_success": _field(
                "boolean",
                description="Correct non-abstained diagnosis and locked localization success.",
            ),
            "prior_test_use_disclosed": _field(
                "boolean", description="Must remain true in every patient-level output."
            ),
        },
    },
    "patient_metrics.csv": {
        "grain": "one row per model × seed × classifier strategy × probability scale",
        "primary_key": ["model", "seed", "classifier_strategy", "probability_scale"],
        "columns": {
            "study_id": _field("string", description="Study identifier."),
            "model": _field("string", description="Segmenter condition."),
            "seed": _field("integer", description="Outer split seed."),
            "classifier_strategy": _field("string", description="Classifier strategy."),
            "probability_scale": _field(
                "string", description="Raw or calibrated.", allowed=["raw", "calibrated"]
            ),
            "n_intended": _field("integer", description="Complete intended patient denominator."),
            "n_evaluable": _field("integer", description="Patients receiving a score."),
            "n_abstained": _field("integer", description="Patients without an operational label."),
            "coverage": _field("number", description="n_evaluable / n_intended."),
            "failure_aware_balanced_accuracy": _field(
                "number", nullable=True, description="Primary endpoint; abstention is incorrect."
            ),
            "conditional_balanced_accuracy": _field(
                "number", nullable=True, description="Balanced accuracy among non-abstained patients."
            ),
            "accuracy": _field("number", nullable=True, description="Failure-aware patient accuracy."),
            "macro_f1": _field("number", nullable=True, description="Three-class macro F1."),
            "multiclass_mcc": _field("number", nullable=True, description="Generalised multiclass MCC."),
            "macro_ovr_auroc": _field("number", nullable=True, description="Macro one-vs-rest AUROC."),
            "macro_ovr_average_precision": _field(
                "number", nullable=True, description="Macro one-vs-rest average precision."
            ),
            "multiclass_nll": _field("number", nullable=True, description="Multiclass negative log loss."),
            "multiclass_brier": _field("number", nullable=True, description="Multiclass Brier score."),
            "aurc": _field("number", nullable=True, description="Area under the risk-coverage curve."),
            "localized_rate": _field(
                "number", description="Fraction meeting the locked retrospective localization criterion."
            ),
            "localized_diagnostic_success_rate": _field(
                "number",
                description="Fraction with both a correct issued diagnosis and locked localization success.",
            ),
        },
    },
    "patient_confusion_3x4.csv": {
        "grain": "one cell per true class × operational prediction",
        "primary_key": [
            "model",
            "seed",
            "classifier_strategy",
            "true_label",
            "predicted_label",
        ],
        "columns": {
            "model": _field("string", description="Segmenter condition."),
            "seed": _field("integer", description="Outer split seed."),
            "classifier_strategy": _field("string", description="Classifier strategy."),
            "true_label": _field("integer", description="Reference class.", allowed=CLASS_ORDER),
            "predicted_label": _field(
                "integer", description="Normal/papilledema/pseudopapilledema/abstain.", allowed=PREDICTION_LABELS
            ),
            "count": _field("integer", description="Cell count including zero cells."),
        },
        "required_complete_grid": {"true_label": list(CLASS_ORDER), "predicted_label": list(PREDICTION_LABELS)},
    },
    "classwise_metrics.csv": {
        "grain": "one row per model × seed × strategy × class",
        "primary_key": ["model", "seed", "classifier_strategy", "class_label"],
        "columns": {
            "model": _field("string", description="Segmenter condition."),
            "seed": _field("integer", description="Outer split seed."),
            "classifier_strategy": _field("string", description="Classifier strategy."),
            "class_label": _field("integer", description="Target class.", allowed=CLASS_ORDER),
            "class_name": _field("string", description="Human-readable class name."),
            "n_intended": _field("integer", description="Class denominator including abstentions."),
            "n_evaluable": _field("integer", description="Scored patients in class."),
            "coverage": _field("number", description="Class-conditional coverage."),
            "failure_aware_recall": _field("number", nullable=True, description="Recall with abstention as false negative."),
            "conditional_recall": _field("number", nullable=True, description="Recall among non-abstained patients."),
            "precision": _field("number", nullable=True, description="Class PPV."),
            "f1": _field("number", nullable=True, description="Class F1."),
            "ovr_auroc": _field("number", nullable=True, description="One-vs-rest AUROC."),
            "ovr_average_precision": _field("number", nullable=True, description="One-vs-rest average precision."),
        },
    },
    "calibration_metrics.csv": {
        "grain": "one row per model × seed × strategy × level × probability scale",
        "primary_key": ["model", "seed", "classifier_strategy", "level", "probability_scale"],
        "columns": {
            "model": _field("string", description="Segmenter condition."),
            "seed": _field("integer", description="Outer split seed."),
            "classifier_strategy": _field("string", description="Classifier strategy."),
            "level": _field("string", description="Eye or patient.", allowed=["eye", "patient"]),
            "probability_scale": _field("string", description="Raw or calibrated.", allowed=["raw", "calibrated"]),
            "status": _field(
                "string",
                description="Calibration availability; raw probabilities do not use a temperature.",
                allowed=["available", "unavailable", "not_applicable_raw"],
            ),
            "temperature": _field("number", nullable=True, description="Locked scalar temperature."),
            "multiclass_nll": _field("number", nullable=True, description="NLL."),
            "multiclass_brier": _field("number", nullable=True, description="Brier score."),
            "top_label_ece": _field("number", nullable=True, description="Top-label ECE."),
            "classwise_ece_macro": _field("number", nullable=True, description="Macro classwise ECE."),
        },
    },
    "risk_coverage.csv": {
        "grain": "one attainable coverage point per model × seed × strategy",
        "primary_key": ["model", "seed", "classifier_strategy", "rank"],
        "columns": {
            "model": _field("string", description="Segmenter condition."),
            "seed": _field("integer", description="Outer split seed."),
            "classifier_strategy": _field("string", description="Classifier strategy."),
            "rank": _field("integer", description="Stable confidence rank."),
            "coverage": _field("number", description="Fraction retained."),
            "selective_risk": _field("number", description="Error among retained patients."),
            "confidence_definition": _field(
                "string", description="Locked confidence definition.", allowed=["maximum_calibrated_patient_probability"]
            ),
        },
    },
    "uncertainty_metrics.csv": {
        "grain": (
            "one row per model × seed × strategy × calibrated level × metric path"
        ),
        "primary_key": [
            "model",
            "seed",
            "classifier_strategy",
            "level",
            "metric_path",
        ],
        "columns": {
            "model": _field("string", description="Segmenter condition."),
            "seed": _field("integer", description="Outer split seed."),
            "classifier_strategy": _field(
                "string", description="Classifier strategy."
            ),
            "level": _field(
                "string",
                description="Calibrated evaluation level.",
                allowed=["eye", "patient"],
            ),
            "probability_scale": _field(
                "string", description="Locked probability scale.", allowed=["calibrated"]
            ),
            "metric_path": _field(
                "string", description="Dot-delimited path in the evaluation metric object."
            ),
            "estimate": _field(
                "number", nullable=True, description="Observed point estimate."
            ),
            "ci_95_low": _field(
                "number", nullable=True, description="Lower 95% confidence limit."
            ),
            "ci_95_high": _field(
                "number", nullable=True, description="Upper 95% confidence limit."
            ),
            "bootstrap_se": _field(
                "number", nullable=True, description="Bootstrap standard error."
            ),
            "confidence_level": _field(
                "number", description="Locked confidence level.", allowed=[0.95]
            ),
            "requested_method": _field(
                "string", description="Pre-specified interval method.", allowed=["bca"]
            ),
            "interval_method": _field(
                "string",
                description="Method actually used after any fallback.",
                allowed=["bca", "percentile"],
            ),
            "fallback_used": _field(
                "boolean", description="Whether BCa fell back to a percentile interval."
            ),
            "fallback_detail": _field(
                "string",
                nullable=True,
                description="Machine-readable reason for an interval fallback.",
            ),
            "valid_draws": _field(
                "integer", description="Finite bootstrap replicates for this metric."
            ),
            "requested_draws": _field(
                "integer", description="Locked bootstrap replicate count.", allowed=[5000]
            ),
            "resampling_unit": _field(
                "string",
                description="Independent bootstrap sampling unit.",
                allowed=["whole_patient_cluster"],
            ),
            "stratified_by_label": _field(
                "boolean",
                description="Whether patients were sampled within diagnosis strata.",
                allowed=[True],
            ),
            "stratification": _field(
                "string",
                description="Explicit bootstrap stratification definition.",
                allowed=["three_class_patient_label"],
            ),
            "bootstrap_seed": _field(
                "integer",
                nullable=True,
                description=(
                    "Deterministic RNG seed; nullable only for legacy stored bootstrap JSON."
                ),
            ),
        },
    },
    "segmentation_by_class.csv": {
        "grain": "one row per model × seed × class × level",
        "primary_key": ["model", "seed", "class_label", "level"],
        "columns": {
            "model": _field("string", description="Segmenter condition."),
            "seed": _field("integer", description="Outer split seed."),
            "class_label": _field("integer", description="Diagnostic class.", allowed=CLASS_ORDER),
            "class_name": _field("string", description="Class name."),
            "level": _field("string", description="Frame, eye, or patient.", allowed=["frame", "eye", "patient"]),
            "n_intended": _field("integer", description="Complete class denominator."),
            "roi_coverage": _field("number", description="Strict predicted-ROI coverage."),
            "dice": _field("number", nullable=True, description="Reference-standard overlap."),
            "iou": _field("number", nullable=True, description="Reference-standard intersection over union."),
            "empty_or_tiny": _field("integer", description="Empty/tiny failures."),
            "oversegmented_or_edge": _field("integer", description="Oversegmentation/edge failures."),
            "ambiguous_multi": _field("integer", description="Ambiguous multi-component failures."),
        },
    },
    "segmentation_uncertainty.csv": {
        "grain": "one row per model × seed × class × level × segmentation metric",
        "primary_key": ["model", "seed", "class_label", "level", "metric"],
        "columns": {
            "model": _field("string", description="Segmenter condition."),
            "seed": _field("integer", description="Outer split seed."),
            "class_label": _field(
                "integer", description="Diagnostic class.", allowed=CLASS_ORDER
            ),
            "class_name": _field("string", description="Diagnostic class name."),
            "level": _field(
                "string",
                description="Nested segmentation summary level.",
                allowed=["frame", "eye", "patient"],
            ),
            "metric": _field(
                "string",
                description="Segmentation estimand.",
                allowed=["roi_coverage", "dice", "iou"],
            ),
            "metric_role": _field(
                "string",
                description=(
                    "Separates operational predicted-ROI coverage from retrospective "
                    "reference-standard overlap."
                ),
                allowed=[
                    "strict_predicted_roi_coverage_no_reference_standard",
                    "retrospective_reference_standard_overlap_only",
                ],
            ),
            "estimate": _field(
                "number", nullable=True, description="Observed hierarchical estimate."
            ),
            "ci_95_low": _field(
                "number", nullable=True, description="Lower 95% confidence limit."
            ),
            "ci_95_high": _field(
                "number", nullable=True, description="Upper 95% confidence limit."
            ),
            "bootstrap_se": _field(
                "number", nullable=True, description="Patient-cluster bootstrap SE."
            ),
            "confidence_level": _field(
                "number", description="Locked confidence level.", allowed=[0.95]
            ),
            "requested_method": _field(
                "string", description="Pre-specified interval method.", allowed=["bca"]
            ),
            "interval_method": _field(
                "string",
                description="Method actually used after any fallback.",
                allowed=["bca", "percentile"],
            ),
            "fallback_used": _field(
                "boolean", description="Whether BCa fell back to a percentile interval."
            ),
            "fallback_detail": _field(
                "string",
                nullable=True,
                description="Machine-readable reason for an interval fallback.",
            ),
            "valid_draws": _field(
                "integer", description="Finite patient-cluster bootstrap replicates."
            ),
            "requested_draws": _field(
                "integer", description="Locked bootstrap replicate count.", allowed=[5000]
            ),
            "resampling_unit": _field(
                "string",
                description="Independent bootstrap sampling unit.",
                allowed=["whole_patient_cluster"],
            ),
            "stratified_by_label": _field(
                "boolean",
                description="Patients are sampled within diagnosis strata.",
                allowed=[True],
            ),
            "stratification": _field(
                "string",
                description="Explicit bootstrap stratification definition.",
                allowed=["three_class_patient_label"],
            ),
            "bootstrap_seed": _field(
                "integer",
                description="Stable SHA-256-derived seed from model and split seed.",
            ),
            "gt_used_for_inference_selection_or_abstention": _field(
                "boolean",
                description="Must be false: overlap is a retrospective audit only.",
                allowed=[False],
            ),
        },
    },
    "provenance_artifacts.csv": {
        "grain": "one row per locked artifact",
        "primary_key": ["model", "seed", "classifier_strategy", "artifact_role"],
        "columns": {
            "model": _field("string", nullable=True, description="Segmenter condition or null for global artifacts."),
            "seed": _field("integer", nullable=True, description="Outer seed or null for global artifacts."),
            "classifier_strategy": _field("string", nullable=True, description="Classifier strategy if applicable."),
            "artifact_role": _field("string", description="Stable artifact role."),
            "path": _field("string", description="Project-relative or absolute artifact path."),
            "sha256": _field("string", description="Artifact SHA-256."),
            "size_bytes": _field("integer", description="Artifact size."),
            "created_before_test_open": _field("boolean", description="Whether artifact was locked before new test access."),
        },
    },
}


def publication_output_schemas() -> dict[str, dict[str, Any]]:
    return copy.deepcopy(OUTPUT_TABLE_SCHEMAS)


def build_q1_metadata(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Build the static, protocol-locked portion of the publication metadata."""

    metadata = {
        "schema_version": Q1_METADATA_SCHEMA_VERSION,
        "study_identity": {
            "study_id": cfg["study_id"],
            "protocol_version": cfg["protocol_version"],
            "study_design": cfg["study_design"],
            "analysis_status": cfg["analysis_status"],
            "intended_use": "research_only_three_class_patient_level_decision_support",
            "primary_unit": "patient",
        },
        "clinical_context": {
            "target_population": "patients undergoing the locked ocular ultrasound acquisition workflow",
            "input_modality": "ocular_ultrasound_frames",
            "care_pathway_role": "investigational differentiation of normal, papilledema, and pseudopapilledema",
            "three_target_classes": {
                str(label): CLASS_NAMES[label] for label in CLASS_ORDER
            },
        },
        "data": {
            "source": cfg["dataset"]["root"],
            "patient_count": cfg["dataset"]["patients"],
            "eye_count": cfg["dataset"]["eyes"],
            "frame_count": cfg["dataset"]["frames"],
            "patient_counts_by_class": cfg["dataset"]["patient_counts_by_class"],
            "demographic_metadata_status": cfg["reporting"]["demographic_metadata_status"],
        },
        "reference_standard": {
            "diagnosis_reference_standard": cfg["dataset"]["reference_standard"],
            "roi_reference_standard": "locked_anatomical_roi_masks_used_for_segmenter_development_and_retrospective_overlap_evaluation_only",
            "label_blinding_status": "diagnostic labels are not used by the frozen segmentation loss; classifier labels are available only in their assigned development partition",
        },
        "partitioning": {
            "unit": cfg["split_policy"]["unit"],
            "stratification": cfg["split_policy"]["stratify_by"],
            "seed_memberships": list(cfg["split_seeds"]),
            "within_patient_grouping": cfg["split_policy"]["both_eyes_and_all_frames_same_partition"],
            "duplicate_audit": cfg["split_policy"]["leakage_checks"],
        },
        "frozen_upstream": {
            "source_study_id": cfg["frozen_upstream"]["source_study_id"],
            "source_output": cfg["frozen_upstream"]["source_output"],
            "reuse_scope": cfg["frozen_upstream"]["development_import_roles"],
            "forbidden_imports": [
                "binary_classifier_checkpoints",
                "binary_logits_or_probabilities",
                "binary_thresholds",
                "binary_calibrators",
            ],
            "development_receipt_count": cfg["frozen_upstream"][
                "required_development_import_receipts"
            ],
            "test_artifact_deferral": cfg["frozen_upstream"][
                "test_artifacts_deferred_until_global_lock"
            ],
        },
        "model_development": {
            "classifier_formulation": cfg["classifier"]["formulation"],
            "classifier_strategies": cfg["classifier"]["strategy_order"],
            "class_count": cfg["classifier"]["classes"],
            "loss": cfg["classifier"]["loss"],
            "early_stopping_monitor": cfg["classifier"]["early_stopping_monitor"],
            "binary_head_reused": cfg["classifier"]["reuse_binary_head"],
            "architectures": {
                "model_specific": cfg["classifier"][
                    "model_specific_architecture_by_segmenter"
                ],
                "standardized": cfg["classifier"]["standardized_architecture"],
            },
            "initialization": {
                "model_specific": cfg["classifier"][
                    "initialization_by_segmenter"
                ],
                "standardized": cfg["classifier"]["standardized_initialization"],
                "network_download_during_run": False,
            },
            "optimizer": {
                "name": cfg["classifier"]["optimizer"],
                "learning_rate": cfg["training"]["learning_rate"],
                "weight_decay": cfg["training"]["weight_decay"],
                "gradient_clip": cfg["training"]["gradient_clip"],
            },
            "scheduler": cfg["classifier"]["scheduler"],
            "batch_size": cfg["training"]["batch_size"],
            "maximum_epochs": cfg["classifier"]["maximum_epochs"],
            "patience": cfg["classifier"]["patience"],
            "augmentation": cfg["training"]["augmentation"],
            "mixed_precision": {
                "enabled": cfg["training"]["amp"],
                "dtype": cfg["training"]["amp_dtype"],
            },
            "determinism": {
                "seed_offsets": cfg["training"]["seed_offsets"],
                "deterministic_algorithms_warn_only": cfg["training"][
                    "deterministic_algorithms_warn_only"
                ],
            },
            "loss_weighting": cfg["classifier"]["loss_weighting"],
            "loss_weight_normalization": (
                "minibatch_mean_divided_by_fold_fixed_global_weight_mean"
            ),
        },
        "aggregation": {
            "frame_to_eye": cfg["aggregation"]["frame_to_eye"],
            "minimum_valid_frames": cfg["aggregation"]["minimum_valid_frames_per_eye"],
            "eye_to_patient": cfg["aggregation"]["eye_to_patient"],
            "both_eyes_required": cfg["aggregation"]["require_both_eyes"],
        },
        "calibration": {
            "method": cfg["calibration"]["method"],
            "fit_partition": cfg["calibration"]["fit_partition"],
            "levels": cfg["calibration"]["levels"],
            "test_refitting": cfg["calibration"]["test_refitting"],
        },
        "abstention": {
            "primary_policy": cfg["abstention"]["primary_policy"],
            "failure_aware_scoring": cfg["abstention"][
                "failure_aware_metrics_count_abstention_as_incorrect"
            ],
            "uncertainty_policy_role": cfg["abstention"][
                "classification_uncertainty_policy"
            ],
        },
        "evaluation": {
            "primary_estimand": cfg["evaluation"]["primary_estimand"],
            "primary_endpoint": cfg["evaluation"]["primary_estimand"]["endpoint"],
            "primary_population": cfg["evaluation"]["primary_estimand"]["population"],
            "primary_confusion_matrix": cfg["evaluation"]["primary_estimand"][
                "confusion_matrix"
            ],
            "localized_diagnostic_success_definition": dict(
                cfg["evaluation"]["localized_success"]
            ),
            "mandatory_secondary_endpoints": cfg["evaluation"][
                "mandatory_secondary_endpoints"
            ],
        },
        "statistics": {
            "confidence_level": cfg["statistics"]["confidence_level"],
            "resampling_unit": cfg["statistics"]["bootstrap_unit"],
            "paired_comparison_unit": cfg["statistics"]["paired_comparison_unit"],
            "paired_comparison_count": cfg["statistics"]["paired_comparison_count"],
            "paired_interval_method": cfg["statistics"]["paired_interval_method"],
            "paired_test_method": cfg["statistics"]["paired_test_method"],
            "multiplicity_correction": cfg["statistics"]["multiplicity_correction"],
            "seed_summary": cfg["statistics"]["five_seed_summary"],
            "overlapping_holdout_warning": (
                "The five test memberships overlap and are not independent cohorts; "
                "seed estimates must not be pooled as independent observations."
            ),
            "small_class_warning": cfg["statistics"]["small_test_class_warning"],
        },
        "prior_test_use": {
            "source_test_previously_opened": cfg["prior_test_use"][
                "source_binary_test_was_opened"
            ],
            "same_memberships_reused": cfg["prior_test_use"][
                "same_patient_memberships_are_reused"
            ],
            "required_disclosure": cfg["prior_test_use"]["required_disclosure"],
            "confirmatory_claim_allowed": cfg["prior_test_use"][
                "confirmatory_claim_allowed"
            ],
        },
        "external_evaluation": {
            "status": "not_performed_in_this_internal_extension",
            "required_for_confirmatory_claim": cfg["prior_test_use"][
                "required_confirmatory_evaluation"
            ],
        },
        "reproducibility": {
            "config_sha256": cfg.get("config_sha256")
            or canonical_sha256(config_without_runtime(cfg)),
            "manifest_sha256": cfg["dataset"]["manifest_sha256"],
            "source_anchor_hashes": {
                name: spec["sha256"]
                for name, spec in cfg["frozen_upstream"]["source_anchors"].items()
            },
            "artifact_receipts_required": True,
            "runtime_environment_artifact": "provenance/runtime_environment.json",
            "code_inventory_artifact": "provenance/code_inventory.json",
        },
        "limitations": {
            "post_hoc_internal_reuse": True,
            "small_class_counts": True,
            "seed_overlap": True,
            "missing_demographics": True,
        },
        "reporting_guidelines": {
            "claim": cfg["reporting"]["frameworks"]["claim"],
            "tripod": cfg["reporting"]["frameworks"]["tripod"],
            "terminology": cfg["reporting"]["terminology"],
        },
    }
    validate_q1_metadata(metadata)
    return metadata


def validate_q1_metadata(metadata: Mapping[str, Any]) -> None:
    if not isinstance(metadata, Mapping):
        raise ReportingSchemaError("Q1 metadata must be an object")
    if metadata.get("schema_version") != Q1_METADATA_SCHEMA_VERSION:
        raise ReportingSchemaError("Q1 metadata schema version mismatch")
    for section in Q1_METADATA_SCHEMA["required_sections"]:
        value = metadata.get(section)
        if not isinstance(value, Mapping):
            raise ReportingSchemaError(f"Missing Q1 metadata section: {section}")
        missing = [
            field
            for field in Q1_METADATA_SCHEMA["section_requirements"][section]
            if field not in value
        ]
        if missing:
            raise ReportingSchemaError(
                f"Q1 metadata section {section!r} is missing: {', '.join(missing)}"
            )
    prior = metadata["prior_test_use"]
    disclosure = prior["required_disclosure"]
    if (
        prior["source_test_previously_opened"] is not True
        or prior["same_memberships_reused"] is not True
        or prior["confirmatory_claim_allowed"] is not False
        or not isinstance(disclosure, str)
        or "exploratory" not in disclosure.lower()
        or "post-hoc" not in disclosure.lower()
    ):
        raise ReportingSchemaError(
            "Q1 metadata must contain the explicit exploratory/post-hoc prior-test-use disclosure"
        )
    if metadata["evaluation"]["primary_endpoint"] != (
        "three_class_failure_aware_balanced_accuracy"
    ):
        raise ReportingSchemaError("Q1 metadata primary endpoint changed")
    if metadata["aggregation"]["both_eyes_required"] is not True:
        raise ReportingSchemaError("Q1 metadata must retain the both-eyes patient rule")
    if metadata["calibration"]["test_refitting"] is not False:
        raise ReportingSchemaError("Q1 metadata cannot permit test recalibration")
    localized = metadata["evaluation"]["localized_diagnostic_success_definition"]
    if (
        not isinstance(localized, Mapping)
        or localized.get("frame_hit_source") != "immutable_source_evaluation_roi_hit"
        or localized.get("frame_hit_iou_threshold") != 0.5
        or localized.get("minimum_hit_frames_per_eye") != 4
        or localized.get("require_both_eyes") is not True
        or localized.get("gt_used_for_retrospective_evaluation_only") is not True
        or localized.get("gt_used_for_inference_selection_or_abstention") is not False
    ):
        raise ReportingSchemaError("Q1 metadata localized diagnostic success rule changed")


def _matches_type(value: Any, dtype: str) -> bool:
    if dtype == "string":
        return isinstance(value, str)
    if dtype == "boolean":
        return isinstance(value, bool)
    if dtype == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if dtype == "number":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        )
    raise ReportingSchemaError(f"Unknown schema type: {dtype}")


def validate_output_rows(
    table_name: str,
    rows: Iterable[Mapping[str, Any]],
    *,
    strict_columns: bool = True,
) -> list[dict[str, Any]]:
    if table_name not in OUTPUT_TABLE_SCHEMAS:
        raise ReportingSchemaError(f"Unknown publication output table: {table_name}")
    schema = OUTPUT_TABLE_SCHEMAS[table_name]
    columns = schema["columns"]
    validated: list[dict[str, Any]] = []
    keys_seen: set[tuple[Any, ...]] = set()
    for index, raw_row in enumerate(rows):
        if not isinstance(raw_row, Mapping):
            raise ReportingSchemaError(f"{table_name} row {index} is not an object")
        row = dict(raw_row)
        missing = [column for column in columns if column not in row]
        if missing:
            raise ReportingSchemaError(
                f"{table_name} row {index} is missing columns: {', '.join(missing)}"
            )
        if strict_columns:
            extra = [column for column in row if column not in columns]
            if extra:
                raise ReportingSchemaError(
                    f"{table_name} row {index} has undeclared columns: {', '.join(extra)}"
                )
        for column, spec in columns.items():
            value = row[column]
            if value is None:
                if not spec["nullable"]:
                    raise ReportingSchemaError(
                        f"{table_name} row {index} column {column!r} cannot be null"
                    )
                continue
            if not _matches_type(value, spec["type"]):
                raise ReportingSchemaError(
                    f"{table_name} row {index} column {column!r} has the wrong type"
                )
            if "allowed" in spec and value not in spec["allowed"]:
                raise ReportingSchemaError(
                    f"{table_name} row {index} column {column!r} is outside its allowed values"
                )
        primary_key = tuple(row[column] for column in schema["primary_key"])
        if primary_key in keys_seen:
            raise ReportingSchemaError(
                f"{table_name} contains duplicate primary key {primary_key!r}"
            )
        keys_seen.add(primary_key)
        if table_name == "patient_predictions.csv":
            _validate_patient_prediction_row(row, index)
        if table_name in {"uncertainty_metrics.csv", "segmentation_uncertainty.csv"}:
            _validate_uncertainty_row(table_name, row, index)
        validated.append(row)
    complete_grid = schema.get("required_complete_grid")
    if complete_grid and validated:
        grid_columns = tuple(complete_grid)
        group_columns = tuple(
            column for column in schema["primary_key"] if column not in grid_columns
        )
        expected_cells = set(
            itertools.product(*(complete_grid[column] for column in grid_columns))
        )
        observed_by_group: dict[tuple[Any, ...], set[tuple[Any, ...]]] = {}
        for row in validated:
            group = tuple(row[column] for column in group_columns)
            cell = tuple(row[column] for column in grid_columns)
            observed_by_group.setdefault(group, set()).add(cell)
        for group, observed_cells in observed_by_group.items():
            if observed_cells != expected_cells:
                missing = sorted(expected_cells - observed_cells)
                raise ReportingSchemaError(
                    f"{table_name} group {group!r} does not contain the full grid; missing {missing!r}"
                )
    return validated


def _validate_patient_prediction_row(row: Mapping[str, Any], index: int) -> None:
    raw = (
        row["p_normal_raw"],
        row["p_papilledema_raw"],
        row["p_pseudopapilledema_raw"],
    )
    calibrated = (
        row["p_normal_calibrated"],
        row["p_papilledema_calibrated"],
        row["p_pseudopapilledema_calibrated"],
    )
    for name, vector in (("raw", raw), ("calibrated", calibrated)):
        if all(value is None for value in vector):
            continue
        if any(value is None for value in vector):
            raise ReportingSchemaError(
                f"patient_predictions.csv row {index} has a partial {name} probability vector"
            )
        if any(value < 0 or value > 1 for value in vector):
            raise ReportingSchemaError(
                f"patient_predictions.csv row {index} has invalid {name} probabilities"
            )
        if not math.isclose(sum(vector), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ReportingSchemaError(
                f"patient_predictions.csv row {index} {name} probabilities do not sum to one"
            )
    if row["abstained"] != (row["predicted_label"] == ABSTAIN_LABEL):
        raise ReportingSchemaError(
            f"patient_predictions.csv row {index} abstention flag and label disagree"
        )
    if row["abstained"] and any(value is not None for value in (*raw, *calibrated)):
        raise ReportingSchemaError(
            f"patient_predictions.csv row {index} gives probabilities to an abstained patient"
        )
    if not row["abstained"] and (
        any(value is None for value in raw) or any(value is None for value in calibrated)
    ):
        raise ReportingSchemaError(
            f"patient_predictions.csv row {index} lacks probabilities for an evaluable patient"
        )
    if row["abstained"] and not row["abstention_reason"]:
        raise ReportingSchemaError(
            f"patient_predictions.csv row {index} abstention requires a reason"
        )
    structural_evaluable = bool(
        row["right_eye_evaluable"] and row["left_eye_evaluable"]
    )
    if row["abstained"] == structural_evaluable:
        raise ReportingSchemaError(
            f"patient_predictions.csv row {index} violates the locked both-eyes gate"
        )
    for side in ("right", "left"):
        count = row[f"{side}_valid_frames"]
        eye_evaluable = row[f"{side}_eye_evaluable"]
        if not 0 <= count <= 7 or eye_evaluable != (count >= 4):
            raise ReportingSchemaError(
                f"patient_predictions.csv row {index} violates the locked 4-of-7 {side}-eye gate"
            )
    if row["prior_test_use_disclosed"] is not True:
        raise ReportingSchemaError(
            f"patient_predictions.csv row {index} suppresses the mandatory prior-use disclosure"
        )
    expected_localized_success = bool(
        (not row["abstained"])
        and row["localized"]
        and row["predicted_label"] == row["true_label"]
    )
    if row["localized_diagnostic_success"] is not expected_localized_success:
        raise ReportingSchemaError(
            f"patient_predictions.csv row {index} localized diagnostic success is inconsistent"
        )


def _validate_uncertainty_row(
    table_name: str, row: Mapping[str, Any], index: int
) -> None:
    if not 0 <= row["valid_draws"] <= row["requested_draws"]:
        raise ReportingSchemaError(
            f"{table_name} row {index} has invalid bootstrap draw counts"
        )
    bounds = (row["ci_95_low"], row["ci_95_high"])
    if (bounds[0] is None) != (bounds[1] is None):
        raise ReportingSchemaError(
            f"{table_name} row {index} has a partial confidence interval"
        )
    if bounds[0] is not None and bounds[0] > bounds[1]:
        raise ReportingSchemaError(
            f"{table_name} row {index} has reversed confidence limits"
        )
    if row["valid_draws"] == 0 and any(value is not None for value in bounds):
        raise ReportingSchemaError(
            f"{table_name} row {index} cannot report confidence limits without valid draws"
        )
    if row["valid_draws"] > 0 and any(value is None for value in bounds):
        raise ReportingSchemaError(
            f"{table_name} row {index} must report confidence limits when draws are valid"
        )
    bootstrap_se = row["bootstrap_se"]
    if bootstrap_se is not None and bootstrap_se < 0:
        raise ReportingSchemaError(
            f"{table_name} row {index} has a negative bootstrap standard error"
        )
    if row["valid_draws"] > 1 and bootstrap_se is None:
        raise ReportingSchemaError(
            f"{table_name} row {index} lacks bootstrap SE despite multiple valid draws"
        )
    if row["valid_draws"] <= 1 and bootstrap_se is not None:
        raise ReportingSchemaError(
            f"{table_name} row {index} reports bootstrap SE with fewer than two valid draws"
        )
    expected_fallback = row["interval_method"] != row["requested_method"]
    if row["fallback_used"] is not expected_fallback:
        raise ReportingSchemaError(
            f"{table_name} row {index} has inconsistent fallback metadata"
        )
    fallback_detail = row["fallback_detail"]
    has_fallback_detail = isinstance(fallback_detail, str) and bool(
        fallback_detail.strip()
    )
    if expected_fallback != has_fallback_detail:
        raise ReportingSchemaError(
            f"{table_name} row {index} must explain exactly the fallbacks it uses"
        )
    if row["bootstrap_seed"] is not None and row["bootstrap_seed"] < 0:
        raise ReportingSchemaError(
            f"{table_name} row {index} has a negative bootstrap seed"
        )
    if table_name == "segmentation_uncertainty.csv":
        if row["class_name"] != CLASS_NAMES[row["class_label"]]:
            raise ReportingSchemaError(
                f"{table_name} row {index} class label/name disagree"
            )
        expected_role = (
            "strict_predicted_roi_coverage_no_reference_standard"
            if row["metric"] == "roi_coverage"
            else "retrospective_reference_standard_overlap_only"
        )
        if row["metric_role"] != expected_role:
            raise ReportingSchemaError(
                f"{table_name} row {index} mislabels reference-standard use"
            )


def write_q1_metadata(path: str | Path, metadata: Mapping[str, Any]) -> Path:
    validate_q1_metadata(metadata)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        metadata, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


__all__ = [
    "ABSTAIN_LABEL",
    "OUTPUT_SCHEMA_VERSION",
    "OUTPUT_TABLE_SCHEMAS",
    "PREDICTION_LABELS",
    "Q1_METADATA_SCHEMA",
    "Q1_METADATA_SCHEMA_VERSION",
    "ReportingSchemaError",
    "build_q1_metadata",
    "publication_output_schemas",
    "validate_output_rows",
    "validate_q1_metadata",
    "write_q1_metadata",
]
