"""Locked configuration for the frozen-upstream three-class ROI study.

This module intentionally has no dependency on the binary study engine.  It
validates the scientific contract for the extension and resolves the immutable
artifacts that may be imported from the completed binary study.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
EXPECTED_MODELS = ("yolo26", "vit_method2", "emcad", "sam2_unet")
EXPECTED_SEEDS = (17, 42, 2026, 3407, 9103)
CLASS_ORDER = (0, 1, 2)
CLASS_NAMES = {
    0: "normal",
    1: "papilledema",
    2: "pseudopapilledema",
}
CLASSIFIER_STRATEGIES = ("model_specific", "standardized_resnet18")

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "config_threeclass_roi.json"


class ConfigError(ValueError):
    """Raised when the locked scientific configuration is malformed or unsafe."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_project_path(value: str | Path, *, must_exist: bool = False) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(path)
    return path


def _mapping(parent: Mapping[str, Any], key: str, where: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ConfigError(f"{where}.{key} must be an object")
    return value


def _sequence(parent: Mapping[str, Any], key: str, where: str) -> Sequence[Any]:
    value = parent.get(key)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigError(f"{where}.{key} must be an array")
    return value


def _exact(value: Any, expected: Any, where: str) -> None:
    if value != expected:
        raise ConfigError(f"{where} must equal {expected!r}; observed {value!r}")


def _sha256(value: Any, where: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value.lower())
    ):
        raise ConfigError(f"{where} must be a 64-character SHA-256 digest")


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{where} must be a positive integer")
    return value


def _validate_sources(cfg: Mapping[str, Any]) -> None:
    split = _mapping(cfg, "split_policy", "config")
    _exact(split.get("unit"), "patient", "split_policy.unit")
    _exact(split.get("stratify_by"), "label_3class", "split_policy.stratify_by")
    _exact(
        split.get("reuse_exact_binary_study_memberships"),
        True,
        "split_policy.reuse_exact_binary_study_memberships",
    )
    _exact(split.get("regenerate"), False, "split_policy.regenerate")
    _exact(
        split.get("both_eyes_and_all_frames_same_partition"),
        True,
        "split_policy.both_eyes_and_all_frames_same_partition",
    )
    _exact(
        split.get("class_quotas_train_validation_test"),
        {"0": [28, 10, 10], "1": [13, 4, 4], "2": [14, 4, 4]},
        "split_policy.class_quotas_train_validation_test",
    )
    sources = _mapping(split, "sources", "split_policy")
    if tuple(int(key) for key in sources) != EXPECTED_SEEDS:
        raise ConfigError("split_policy.sources must follow the five locked seeds")
    for seed in EXPECTED_SEEDS:
        spec = _mapping(sources, str(seed), "split_policy.sources")
        if not isinstance(spec.get("path"), str) or not spec["path"]:
            raise ConfigError(f"split_policy.sources.{seed}.path is required")
        _sha256(spec.get("sha256"), f"split_policy.sources.{seed}.sha256")


def _validate_upstream(cfg: Mapping[str, Any]) -> None:
    upstream = _mapping(cfg, "frozen_upstream", "config")
    _exact(
        upstream.get("source_study_id"),
        "strict_predicted_roi_binary_4model_v1_2_0_clean",
        "frozen_upstream.source_study_id",
    )
    _exact(
        upstream.get("source_output"),
        "strict_roi_results_4model_v1_2_0",
        "frozen_upstream.source_output",
    )
    _exact(
        upstream.get("mode"),
        "immutable_import_by_sha256_receipt",
        "frozen_upstream.mode",
    )
    for key, expected in {
        "segmenter_training_was_classification_label_blind": True,
        "classifier_loss_during_segmenter_training": 0.0,
        "segmenter_and_roi_parameters_frozen": True,
        "allow_retraining_or_retuning_upstream": False,
        "allow_binary_classifier_artifacts": False,
        "allow_binary_logits_probabilities_thresholds_or_calibrators": False,
        "test_artifacts_deferred_until_global_lock": True,
    }.items():
        _exact(upstream.get(key), expected, f"frozen_upstream.{key}")

    development_roles = tuple(
        _sequence(upstream, "development_import_roles", "frozen_upstream")
    )
    deferred_roles = tuple(
        _sequence(upstream, "deferred_test_import_roles", "frozen_upstream")
    )
    expected_development = (
        "source_build_roi_receipt",
        "source_segmenter_lock",
        "source_segmenter_checkpoint",
        "train_oof_roi_index",
        "validation_roi_index",
    )
    expected_deferred = ("source_evaluate_receipt", "test_roi_index")
    _exact(development_roles, expected_development, "frozen_upstream.development_import_roles")
    _exact(deferred_roles, expected_deferred, "frozen_upstream.deferred_test_import_roles")
    if set(development_roles) & set(deferred_roles):
        raise ConfigError("Development and deferred-test upstream roles must be disjoint")

    patterns = _mapping(
        upstream, "development_artifact_patterns", "frozen_upstream"
    )
    deferred_patterns = _mapping(
        upstream, "deferred_test_artifact_patterns", "frozen_upstream"
    )
    _exact(tuple(patterns), expected_development, "frozen_upstream.development_artifact_patterns")
    _exact(tuple(deferred_patterns), expected_deferred, "frozen_upstream.deferred_test_artifact_patterns")
    for role, pattern in {**patterns, **deferred_patterns}.items():
        if not isinstance(pattern, str) or "{model}" not in pattern or "{seed}" not in pattern:
            raise ConfigError(
                f"frozen_upstream artifact pattern {role!r} must contain model and seed placeholders"
            )
        if "classifiers/" in pattern:
            raise ConfigError("Binary classifier paths cannot be imported by the three-class study")

    anchors = _mapping(upstream, "source_anchors", "frozen_upstream")
    expected_anchors = (
        "prepare_receipt",
        "source_config_snapshot",
        "source_protocol_snapshot",
        "source_test_access_sentinel",
    )
    _exact(tuple(anchors), expected_anchors, "frozen_upstream.source_anchors")
    for name in expected_anchors:
        spec = _mapping(anchors, name, "frozen_upstream.source_anchors")
        if not isinstance(spec.get("path"), str) or not spec["path"]:
            raise ConfigError(f"frozen_upstream.source_anchors.{name}.path is required")
        _sha256(spec.get("sha256"), f"frozen_upstream.source_anchors.{name}.sha256")

    expected_units = len(EXPECTED_MODELS) * len(EXPECTED_SEEDS)
    _exact(
        upstream.get("required_development_import_receipts"),
        expected_units,
        "frozen_upstream.required_development_import_receipts",
    )
    _exact(
        upstream.get("required_deferred_test_import_receipts"),
        expected_units,
        "frozen_upstream.required_deferred_test_import_receipts",
    )


def _validate_classifier_and_endpoints(cfg: Mapping[str, Any]) -> None:
    classifier = _mapping(cfg, "classifier", "config")
    _exact(
        tuple(_sequence(classifier, "strategy_order", "classifier")),
        CLASSIFIER_STRATEGIES,
        "classifier.strategy_order",
    )
    _exact(classifier.get("primary_strategy"), CLASSIFIER_STRATEGIES[0], "classifier.primary_strategy")
    _exact(classifier.get("secondary_strategy"), CLASSIFIER_STRATEGIES[1], "classifier.secondary_strategy")
    _exact(classifier.get("formulation"), "direct_flat_three_class_softmax", "classifier.formulation")
    _exact(classifier.get("classes"), 3, "classifier.classes")
    _exact(classifier.get("replace_binary_head"), True, "classifier.replace_binary_head")
    _exact(classifier.get("reuse_binary_head"), False, "classifier.reuse_binary_head")
    _exact(
        classifier.get("reuse_binary_threshold_or_calibrator"),
        False,
        "classifier.reuse_binary_threshold_or_calibrator",
    )
    _exact(classifier.get("input"), "frozen_strict_predicted_roi", "classifier.input")
    _exact(
        classifier.get("loss_weighting"),
        "equal_total_weight_per_patient_within_each_class",
        "classifier.loss_weighting",
    )
    _exact(
        classifier.get("early_stopping_monitor"),
        "validation_patient_macro_nll",
        "classifier.early_stopping_monitor",
    )
    _exact(
        classifier.get("test_used_for_training_selection_or_stopping"),
        False,
        "classifier.test_used_for_training_selection_or_stopping",
    )
    _positive_int(classifier.get("maximum_epochs"), "classifier.maximum_epochs")
    _positive_int(classifier.get("patience"), "classifier.patience")
    fits = _mapping(classifier, "expected_fits", "classifier")
    _exact(
        dict(fits),
        {"model_specific": 20, "standardized_resnet18": 20, "total": 40},
        "classifier.expected_fits",
    )

    initialization = _mapping(
        classifier, "initialization_by_segmenter", "classifier"
    )
    _exact(tuple(initialization), EXPECTED_MODELS, "classifier.initialization_by_segmenter")
    for model in EXPECTED_MODELS:
        spec = _mapping(initialization, model, "classifier.initialization_by_segmenter")
        pretrained = spec.get("pretrained")
        if not isinstance(pretrained, bool):
            raise ConfigError(f"classifier.initialization_by_segmenter.{model}.pretrained must be boolean")
        _exact(
            spec.get("network_download_during_run"),
            False,
            f"classifier.initialization_by_segmenter.{model}.network_download_during_run",
        )
        if pretrained:
            if not isinstance(spec.get("checkpoint"), str) or not spec["checkpoint"]:
                raise ConfigError(f"classifier.initialization_by_segmenter.{model}.checkpoint is required")
            _sha256(spec.get("sha256"), f"classifier.initialization_by_segmenter.{model}.sha256")
        elif spec.get("checkpoint") is not None or spec.get("sha256") is not None:
            raise ConfigError(f"Random initialization for {model} cannot name a checkpoint")
    standardized = _mapping(
        classifier, "standardized_initialization", "classifier"
    )
    _exact(standardized.get("pretrained"), True, "classifier.standardized_initialization.pretrained")
    _exact(
        standardized.get("network_download_during_run"),
        False,
        "classifier.standardized_initialization.network_download_during_run",
    )
    if not isinstance(standardized.get("checkpoint"), str) or not standardized["checkpoint"]:
        raise ConfigError("classifier.standardized_initialization.checkpoint is required")
    _sha256(standardized.get("sha256"), "classifier.standardized_initialization.sha256")

    aggregation = _mapping(cfg, "aggregation", "config")
    _exact(
        aggregation.get("frame_to_eye"),
        "arithmetic_mean_of_valid_frame_probability_vectors",
        "aggregation.frame_to_eye",
    )
    _exact(
        aggregation.get("minimum_valid_frames_per_eye"),
        4,
        "aggregation.minimum_valid_frames_per_eye",
    )
    _exact(aggregation.get("require_both_eyes"), True, "aggregation.require_both_eyes")
    _exact(aggregation.get("primary_prediction_level"), "patient", "aggregation.primary_prediction_level")

    calibration = _mapping(cfg, "calibration", "config")
    _exact(
        calibration.get("method"),
        "single_scalar_temperature_scaling",
        "calibration.method",
    )
    _exact(calibration.get("fit_partition"), "validation", "calibration.fit_partition")
    _exact(calibration.get("test_refitting"), False, "calibration.test_refitting")
    _exact(
        calibration.get("vector_or_dirichlet_calibration_allowed"),
        False,
        "calibration.vector_or_dirichlet_calibration_allowed",
    )
    _exact(
        calibration.get("undefined_fit_policy"),
        "fail_closed_before_global_test_access_without_default_temperature",
        "calibration.undefined_fit_policy",
    )

    abstention = _mapping(cfg, "abstention", "config")
    _exact(
        abstention.get("primary_policy"),
        "structural_roi_gate_only",
        "abstention.primary_policy",
    )
    _exact(abstention.get("test_threshold_tuning"), False, "abstention.test_threshold_tuning")
    _exact(
        abstention.get("failure_aware_metrics_count_abstention_as_incorrect"),
        True,
        "abstention.failure_aware_metrics_count_abstention_as_incorrect",
    )

    evaluation = _mapping(cfg, "evaluation", "config")
    primary = _mapping(evaluation, "primary_estimand", "evaluation")
    _exact(primary.get("unit"), "patient", "evaluation.primary_estimand.unit")
    _exact(
        primary.get("endpoint"),
        "three_class_failure_aware_balanced_accuracy",
        "evaluation.primary_estimand.endpoint",
    )
    _exact(
        primary.get("confusion_matrix"),
        "three_true_classes_by_normal_papilledema_pseudopapilledema_abstain",
        "evaluation.primary_estimand.confusion_matrix",
    )
    _exact(
        evaluation.get("test_selection_or_refitting"),
        False,
        "evaluation.test_selection_or_refitting",
    )
    localized = _mapping(evaluation, "localized_success", "evaluation")
    _exact(localized.get("enabled"), True, "evaluation.localized_success.enabled")
    _exact(
        localized.get("frame_hit_source"),
        "immutable_source_evaluation_roi_hit",
        "evaluation.localized_success.frame_hit_source",
    )
    _exact(
        localized.get("frame_hit_iou_threshold"),
        0.5,
        "evaluation.localized_success.frame_hit_iou_threshold",
    )
    _exact(
        localized.get("minimum_hit_frames_per_eye"),
        4,
        "evaluation.localized_success.minimum_hit_frames_per_eye",
    )
    _exact(
        localized.get("require_both_eyes"),
        True,
        "evaluation.localized_success.require_both_eyes",
    )
    _exact(
        localized.get("gt_used_for_retrospective_evaluation_only"),
        True,
        "evaluation.localized_success.gt_used_for_retrospective_evaluation_only",
    )
    _exact(
        localized.get("gt_used_for_inference_selection_or_abstention"),
        False,
        "evaluation.localized_success.gt_used_for_inference_selection_or_abstention",
    )

    statistics = _mapping(cfg, "statistics", "config")
    _exact(statistics.get("confidence_level"), 0.95, "statistics.confidence_level")
    _exact(statistics.get("bootstrap_draws"), 5000, "statistics.bootstrap_draws")
    _exact(statistics.get("bootstrap_unit"), "patient", "statistics.bootstrap_unit")
    _exact(
        statistics.get("bootstrap_stratify_by_class"),
        True,
        "statistics.bootstrap_stratify_by_class",
    )
    _exact(
        statistics.get("paired_model_comparisons"),
        True,
        "statistics.paired_model_comparisons",
    )
    _exact(
        statistics.get("paired_comparison_unit"),
        "same_seed_same_patient",
        "statistics.paired_comparison_unit",
    )
    _exact(
        statistics.get("paired_comparison_count"),
        30,
        "statistics.paired_comparison_count",
    )
    _exact(
        statistics.get("paired_interval_method"),
        "class_stratified_patient_level_bca_bootstrap",
        "statistics.paired_interval_method",
    )
    _exact(
        statistics.get("paired_test_method"),
        "whole_patient_sign_flip_randomization_two_sided",
        "statistics.paired_test_method",
    )
    _exact(
        statistics.get("multiplicity_correction"),
        "holm_across_all_30_seed_specific_primary_model_pair_tests",
        "statistics.multiplicity_correction",
    )
    _exact(
        statistics.get("overlapping_seed_holdouts_treated_as_independent"),
        False,
        "statistics.overlapping_seed_holdouts_treated_as_independent",
    )


def _validate_test_and_reporting(cfg: Mapping[str, Any]) -> None:
    prior = _mapping(cfg, "prior_test_use", "config")
    for key in (
        "source_binary_test_was_opened",
        "source_binary_results_were_examined",
        "same_patient_memberships_are_reused",
    ):
        _exact(prior.get(key), True, f"prior_test_use.{key}")
    _exact(prior.get("confirmatory_claim_allowed"), False, "prior_test_use.confirmatory_claim_allowed")
    disclosure = prior.get("required_disclosure")
    if not isinstance(disclosure, str) or len(disclosure.split()) < 20:
        raise ConfigError("prior_test_use.required_disclosure must be a substantive disclosure")
    if "exploratory" not in disclosure.lower() or "post-hoc" not in disclosure.lower():
        raise ConfigError("Prior-use disclosure must explicitly say exploratory and post-hoc")

    access = _mapping(cfg, "test_access", "config")
    for key in (
        "global_gate",
        "require_all_locks_before_any_test_artifact_read",
        "deferred_test_artifacts_may_be_read_only_after_sentinel",
        "sentinel_requires_prior_use_disclosure",
    ):
        _exact(access.get(key), True, f"test_access.{key}")
    _exact(
        access.get("allow_test_hyperparameter_threshold_calibration_or_model_selection"),
        False,
        "test_access.allow_test_hyperparameter_threshold_calibration_or_model_selection",
    )
    _exact(access.get("required_development_import_receipts"), 20, "test_access.required_development_import_receipts")
    _exact(access.get("required_validation_locks"), 40, "test_access.required_validation_locks")
    _exact(access.get("validation_locks_per_model_seed"), 2, "test_access.validation_locks_per_model_seed")

    reporting = _mapping(cfg, "reporting", "config")
    frameworks = _mapping(reporting, "frameworks", "reporting")
    _exact(frameworks.get("claim"), "CLAIM_2024_Update", "reporting.frameworks.claim")
    _exact(frameworks.get("tripod"), "TRIPOD_plus_AI_2024", "reporting.frameworks.tripod")
    for key in (
        "patient_level_primary_table_required",
        "three_by_four_failure_aware_confusion_required",
        "class_conditional_segmentation_table_required",
        "calibration_and_selective_prediction_outputs_required",
        "prior_test_use_disclosure_required",
        "external_validation_status_required",
    ):
        _exact(reporting.get(key), True, f"reporting.{key}")


def validate_config(cfg: Mapping[str, Any]) -> None:
    if not isinstance(cfg, Mapping):
        raise ConfigError("Configuration root must be an object")
    _exact(cfg.get("schema_version"), SCHEMA_VERSION, "schema_version")
    _exact(cfg.get("protocol_version"), "1.0.0", "protocol_version")
    _exact(cfg.get("task"), "three_class_classification", "task")
    _exact(
        cfg.get("study_design"),
        "frozen_upstream_strict_predicted_roi_extension",
        "study_design",
    )
    _exact(cfg.get("analysis_status"), "exploratory_post_hoc_internal", "analysis_status")
    if not isinstance(cfg.get("study_id"), str) or "threeclass" not in cfg["study_id"]:
        raise ConfigError("study_id must identify the three-class study")
    if cfg.get("output") == _mapping(cfg, "frozen_upstream", "config").get("source_output"):
        raise ConfigError("The three-class study must use a new output namespace")
    if not isinstance(cfg.get("output"), str) or "threeclass" not in cfg["output"]:
        raise ConfigError("output must be a distinct three-class namespace")
    _exact(tuple(_sequence(cfg, "models", "config")), EXPECTED_MODELS, "models")
    _exact(tuple(_sequence(cfg, "split_seeds", "config")), EXPECTED_SEEDS, "split_seeds")

    classes = _mapping(cfg, "classes", "config")
    _exact(tuple(_sequence(classes, "order", "classes")), CLASS_ORDER, "classes.order")
    _exact(
        dict(_mapping(classes, "names", "classes")),
        {str(key): value for key, value in CLASS_NAMES.items()},
        "classes.names",
    )
    _exact(classes.get("classifier_outputs"), 3, "classes.classifier_outputs")

    dataset = _mapping(cfg, "dataset", "config")
    _exact(dataset.get("patients"), 91, "dataset.patients")
    _exact(dataset.get("eyes"), 182, "dataset.eyes")
    _exact(dataset.get("frames"), 1274, "dataset.frames")
    _exact(dataset.get("frames_per_eye"), 7, "dataset.frames_per_eye")
    _exact(dataset.get("eyes_per_patient"), 2, "dataset.eyes_per_patient")
    _exact(
        dict(_mapping(dataset, "patient_counts_by_class", "dataset")),
        {"0": 48, "1": 21, "2": 22},
        "dataset.patient_counts_by_class",
    )
    _sha256(dataset.get("manifest_sha256"), "dataset.manifest_sha256")

    _validate_sources(cfg)
    _validate_upstream(cfg)
    _validate_classifier_and_endpoints(cfg)
    _validate_test_and_reporting(cfg)

    training = _mapping(cfg, "training", "config")
    _exact(training.get("learning_rate"), 0.0001, "training.learning_rate")
    _exact(training.get("weight_decay"), 0.0001, "training.weight_decay")
    _exact(training.get("batch_size"), 16, "training.batch_size")
    _exact(training.get("gradient_accumulation"), 1, "training.gradient_accumulation")
    _exact(training.get("gradient_clip"), 1.0, "training.gradient_clip")
    _exact(training.get("amp"), True, "training.amp")
    _exact(training.get("amp_dtype"), "bfloat16", "training.amp_dtype")
    _exact(training.get("num_workers"), 0, "training.num_workers")
    augmentation = _mapping(training, "augmentation", "training")
    _exact(augmentation.get("rotation_degrees"), 7.0, "training.augmentation.rotation_degrees")
    _exact(augmentation.get("translation_fraction"), 0.03, "training.augmentation.translation_fraction")
    _exact(augmentation.get("scale"), [0.95, 1.05], "training.augmentation.scale")
    _exact(augmentation.get("brightness"), [0.9, 1.1], "training.augmentation.brightness")
    _exact(augmentation.get("flips"), False, "training.augmentation.flips")


def config_without_runtime(cfg: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(cfg))
    value.pop("_runtime", None)
    value.pop("config_sha256", None)
    return value


def load_config(
    path: str | Path | None = None,
    *,
    verify_frozen_sources: bool = False,
) -> dict[str, Any]:
    source = resolve_project_path(path or DEFAULT_CONFIG_PATH, must_exist=True)
    with source.open("r", encoding="utf-8") as stream:
        cfg = json.load(stream)
    validate_config(cfg)
    digest = canonical_sha256(config_without_runtime(cfg))
    cfg["config_sha256"] = digest
    cfg["_runtime"] = {
        "config_path": str(source),
        "config_sha256": digest,
        "project_root": str(PROJECT_ROOT),
    }
    if verify_frozen_sources:
        verify_frozen_source_anchors(cfg)
        verify_split_sources(cfg)
    return cfg


def _validate_unit(cfg: Mapping[str, Any], model: str, seed: int) -> None:
    if model not in tuple(cfg.get("models", ())):
        raise ConfigError(f"Unknown model: {model!r}")
    if seed not in tuple(cfg.get("split_seeds", ())):
        raise ConfigError(f"Unknown seed: {seed!r}")


def upstream_artifact_paths(
    cfg: Mapping[str, Any],
    model: str,
    seed: int,
    *,
    phase: str = "development",
) -> dict[str, Path]:
    """Resolve the locked upstream unit paths without reading their contents.

    ``phase='test'`` returns only the artifacts whose contents must remain
    unopened until the global validation-lock gate has produced its sentinel.
    """

    _validate_unit(cfg, model, seed)
    upstream = _mapping(cfg, "frozen_upstream", "config")
    if phase == "development":
        patterns = _mapping(
            upstream, "development_artifact_patterns", "frozen_upstream"
        )
    elif phase == "test":
        patterns = _mapping(
            upstream, "deferred_test_artifact_patterns", "frozen_upstream"
        )
    else:
        raise ConfigError("phase must be 'development' or 'test'")
    return {
        role: resolve_project_path(pattern.format(model=model, seed=seed))
        for role, pattern in patterns.items()
    }


def verify_frozen_source_anchors(cfg: Mapping[str, Any]) -> dict[str, str]:
    anchors = _mapping(
        _mapping(cfg, "frozen_upstream", "config"),
        "source_anchors",
        "frozen_upstream",
    )
    observed: dict[str, str] = {}
    for name, raw_spec in anchors.items():
        spec = dict(raw_spec)
        path = resolve_project_path(spec["path"], must_exist=True)
        digest = sha256_file(path)
        if digest.lower() != str(spec["sha256"]).lower():
            raise ConfigError(f"Frozen source anchor hash mismatch: {name}")
        observed[name] = digest
    return observed


def verify_split_sources(cfg: Mapping[str, Any]) -> dict[int, str]:
    sources = _mapping(
        _mapping(cfg, "split_policy", "config"), "sources", "split_policy"
    )
    observed: dict[int, str] = {}
    for seed in EXPECTED_SEEDS:
        spec = dict(sources[str(seed)])
        path = resolve_project_path(spec["path"], must_exist=True)
        digest = sha256_file(path)
        if digest.lower() != str(spec["sha256"]).lower():
            raise ConfigError(f"Locked split hash mismatch for seed {seed}")
        observed[seed] = digest
    return observed


__all__ = [
    "CLASSIFIER_STRATEGIES",
    "CLASS_NAMES",
    "CLASS_ORDER",
    "ConfigError",
    "DEFAULT_CONFIG_PATH",
    "EXPECTED_MODELS",
    "EXPECTED_SEEDS",
    "PACKAGE_ROOT",
    "PROJECT_ROOT",
    "SCHEMA_VERSION",
    "canonical_sha256",
    "config_without_runtime",
    "load_config",
    "resolve_project_path",
    "sha256_file",
    "upstream_artifact_paths",
    "validate_config",
    "verify_frozen_source_anchors",
    "verify_split_sources",
]
