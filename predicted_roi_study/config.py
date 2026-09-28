"""Strict configuration loading and validation for the predicted-ROI study.

The JSON file is the preregistered, machine-readable protocol.  This module is
deliberately dependency-free so every CLI command can validate the protocol
before importing PyTorch or touching clinical test data.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = Path(__file__).with_name("config_strict_roi.json")
SCHEMA_VERSION = 1
EXPECTED_PROTOCOL_VERSION = "1.2.0"
EXPECTED_STUDY_ID = "strict_predicted_roi_binary_4model_v1_2_0_clean"
EXPECTED_OUTPUT = "strict_roi_results_4model_v1_2_0"
EXPECTED_MODELS = (
    "yolo26",
    "vit_method2",
    "emcad",
    "sam2_unet",
)
EXPECTED_SEEDS = (17, 42, 2026, 3407, 9103)
PARTITIONS = ("train", "validation", "test")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ConfigError(ValueError):
    """Raised when a protocol configuration violates a locked invariant."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_project_path(value: str | Path, *, must_exist: bool = False) -> Path:
    """Resolve a protocol path and reject paths outside the study workspace."""

    raw = Path(value)
    if raw.is_absolute():
        raise ConfigError(f"Protocol paths must be project-relative, got: {value}")
    resolved = (PROJECT_ROOT / raw).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        raise ConfigError(f"Protocol path escapes the project root: {value}") from error
    if must_exist and not resolved.exists():
        raise ConfigError(f"Required protocol path does not exist: {resolved}")
    return resolved


def _mapping(parent: Mapping[str, Any], key: str, where: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ConfigError(f"{where}.{key} must be an object")
    return value


def _sequence(parent: Mapping[str, Any], key: str, where: str) -> Sequence[Any]:
    value = parent.get(key)
    if not isinstance(value, list):
        raise ConfigError(f"{where}.{key} must be an array")
    return value


def _exact(value: Any, expected: Any, where: str) -> None:
    if value != expected:
        raise ConfigError(f"{where} must be {expected!r}; got {value!r}")


def _positive_int(value: Any, where: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{where} must be an integer >= {minimum}")
    return value


def _probability(value: Any, where: str, *, open_interval: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where} must be numeric")
    number = float(value)
    valid = 0.0 < number < 1.0 if open_interval else 0.0 <= number <= 1.0
    if not valid:
        bracket = "(0, 1)" if open_interval else "[0, 1]"
        raise ConfigError(f"{where} must be in {bracket}")
    return number


def _number(value: Any, where: str, *, minimum: float | None = None, strict: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ConfigError(f"{where} must be a finite number")
    number = float(value)
    if minimum is not None and (number <= minimum if strict else number < minimum):
        operator = ">" if strict else ">="
        raise ConfigError(f"{where} must be {operator} {minimum}")
    return number


def _sha256(value: Any, where: str) -> None:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ConfigError(f"{where} must be a lowercase SHA-256 digest")


def _increasing_probabilities(values: Any, where: str) -> None:
    if not isinstance(values, list) or not values:
        raise ConfigError(f"{where} must be a non-empty array")
    numbers = [_probability(value, f"{where}[{index}]", open_interval=True) for index, value in enumerate(values)]
    if numbers != sorted(set(numbers)):
        raise ConfigError(f"{where} must contain unique, strictly increasing values")


def _increasing_positive_numbers(values: Any, where: str) -> None:
    if not isinstance(values, list) or not values:
        raise ConfigError(f"{where} must be a non-empty array")
    numbers: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0:
            raise ConfigError(f"{where}[{index}] must be a positive number")
        numbers.append(float(value))
    if numbers != sorted(set(numbers)):
        raise ConfigError(f"{where} must contain unique, strictly increasing values")


def validate_config(cfg: Mapping[str, Any]) -> None:
    """Validate scientific and operational invariants, not merely JSON types."""

    if not isinstance(cfg, Mapping):
        raise ConfigError("Top-level configuration must be an object")
    _exact(cfg.get("schema_version"), SCHEMA_VERSION, "schema_version")
    _exact(cfg.get("task"), "binary", "task")
    _exact(cfg.get("study_design"), "strict_predicted_roi", "study_design")
    _exact(cfg.get("analysis_status"), "exploratory_post_hoc_internal", "analysis_status")
    protocol_version = cfg.get("protocol_version")
    if not isinstance(protocol_version, str) or not re.fullmatch(r"\d+\.\d+\.\d+", protocol_version):
        raise ConfigError("protocol_version must use semantic x.y.z form")
    _exact(protocol_version, EXPECTED_PROTOCOL_VERSION, "protocol_version")

    study_id = cfg.get("study_id")
    if not isinstance(study_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{3,80}", study_id):
        raise ConfigError("study_id must be a stable lowercase identifier")
    _exact(study_id, EXPECTED_STUDY_ID, "study_id")

    models = tuple(_sequence(cfg, "models", "config"))
    seeds = tuple(_sequence(cfg, "split_seeds", "config"))
    if models != EXPECTED_MODELS:
        raise ConfigError(f"models must contain the four locked models in order: {EXPECTED_MODELS}")
    if seeds != EXPECTED_SEEDS:
        raise ConfigError(f"split_seeds must reuse the five locked seeds in order: {EXPECTED_SEEDS}")

    dataset = _mapping(cfg, "dataset", "config")
    dataset_root = dataset.get("root")
    if not isinstance(dataset_root, str):
        raise ConfigError("dataset.root must be a project-relative path")
    resolve_project_path(dataset_root)
    manifest_path = dataset.get("manifest")
    if not isinstance(manifest_path, str):
        raise ConfigError("dataset.manifest must be a project-relative path")
    resolve_project_path(manifest_path)
    _sha256(dataset.get("manifest_sha256"), "dataset.manifest_sha256")
    _exact(dataset.get("patients"), 91, "dataset.patients")
    _exact(dataset.get("eyes"), 182, "dataset.eyes")
    _exact(dataset.get("frames"), 1274, "dataset.frames")
    _exact(dataset.get("frames_per_eye"), 7, "dataset.frames_per_eye")
    _exact(dataset.get("eyes_per_patient"), 2, "dataset.eyes_per_patient")
    _exact(dataset.get("image_size"), [768, 768], "dataset.image_size")
    _exact(dataset.get("binary_mapping"), {"control": 0, "papilledema_or_pseudopapilledema": 1}, "dataset.binary_mapping")
    _exact(
        dataset.get("ground_truth_use"),
        "segmentation_training_validation_and_oracle_analysis_only",
        "dataset.ground_truth_use",
    )

    output = cfg.get("output")
    if not isinstance(output, str) or not output.strip():
        raise ConfigError("output must be a non-empty project-relative directory")
    _exact(output, EXPECTED_OUTPUT, "output")
    output_path = resolve_project_path(output)
    forbidden_roots = {
        resolve_project_path("çalışma_ds"),
        resolve_project_path("binary_results"),
        resolve_project_path("binary_results_seed17"),
        resolve_project_path("binary_results_seeds42_2026_fixed"),
        resolve_project_path("binary_results_seeds3407_9103_fixed"),
        resolve_project_path("predicted_roi_study"),
    }
    if output_path == PROJECT_ROOT.resolve() or any(
        output_path == forbidden or forbidden in output_path.parents for forbidden in forbidden_roots
    ):
        raise ConfigError("output must be a new directory and cannot overwrite prior studies or the dataset")
    clean_run = _mapping(cfg, "clean_run", "config")
    _exact(clean_run.get("output_namespace"), output, "clean_run.output_namespace")
    _exact(
        clean_run.get("prior_failed_output"),
        "strict_roi_results_clean_v1_1_1",
        "clean_run.prior_failed_output",
    )
    _exact(clean_run.get("import_upstream_artifacts"), False, "clean_run.import_upstream_artifacts")
    _exact(
        clean_run.get("resume_policy"),
        "explicit_switch_same_namespace_same_code_config_split_and_input_hashes_only",
        "clean_run.resume_policy",
    )
    _exact(
        clean_run.get("unreceipted_output_policy"),
        "reject_non_orchestration_content",
        "clean_run.unreceipted_output_policy",
    )
    _exact(
        clean_run.get("seed_scope"),
        "one_predeclared_seed_per_launcher_invocation",
        "clean_run.seed_scope",
    )
    _exact(clean_run.get("launcher_stops_after"), "validation_lock", "clean_run.launcher_stops_after")
    _exact(
        clean_run.get("test_evaluation_in_seed_launcher"),
        False,
        "clean_run.test_evaluation_in_seed_launcher",
    )
    prior_failed_output = resolve_project_path(clean_run["prior_failed_output"])
    if output_path == prior_failed_output or prior_failed_output in output_path.parents:
        raise ConfigError("clean_run output must be isolated from the prior failed output")

    split = _mapping(cfg, "split_policy", "config")
    _exact(split.get("unit"), "patient", "split_policy.unit")
    _exact(split.get("mode"), "immutable_copy_with_sha256", "split_policy.mode")
    _exact(split.get("regenerate"), False, "split_policy.regenerate")
    _exact(split.get("stratify_by"), "label_3class", "split_policy.stratify_by")
    _exact(split.get("partitions"), list(PARTITIONS), "split_policy.partitions")
    sources = _mapping(split, "sources", "split_policy")
    if set(sources) != {str(seed) for seed in EXPECTED_SEEDS}:
        raise ConfigError("split_policy.sources must contain exactly one source for every locked seed")
    for seed in EXPECTED_SEEDS:
        source = _mapping(sources, str(seed), "split_policy.sources")
        path = source.get("path")
        if not isinstance(path, str):
            raise ConfigError(f"split_policy.sources.{seed}.path must be a string")
        resolve_project_path(path)
        _sha256(source.get("sha256"), f"split_policy.sources.{seed}.sha256")
    quotas = _mapping(split, "class_quotas_train_validation_test", "split_policy")
    expected_quotas = {"0": [28, 10, 10], "1": [13, 4, 4], "2": [14, 4, 4]}
    _exact(dict(quotas), expected_quotas, "split_policy.class_quotas_train_validation_test")
    _exact(
        set(_sequence(split, "leakage_checks", "split_policy")),
        {
            "no_patient_overlap_within_seed",
            "both_eyes_in_same_partition",
            "exact_cross_patient_image_duplicates",
            "seven_unique_frames_per_eye",
            "consistent_patient_labels",
        },
        "split_policy.leakage_checks",
    )

    preflight = _mapping(cfg, "preflight", "config")
    _exact(preflight.get("require_cuda"), True, "preflight.require_cuda")
    _exact(preflight.get("minimum_cuda_device_count"), 1, "preflight.minimum_cuda_device_count")
    _exact(preflight.get("minimum_free_output_disk_gib"), 100, "preflight.minimum_free_output_disk_gib")
    _exact(preflight.get("verify_manifest_and_all_dataset_file_hashes"), True, "preflight.verify_manifest_and_all_dataset_file_hashes")
    _exact(preflight.get("verify_all_pretrained_weight_hashes"), True, "preflight.verify_all_pretrained_weight_hashes")
    _exact(preflight.get("network_access_during_run"), False, "preflight.network_access_during_run")
    _exact(preflight.get("synthetic_batch_size"), 1, "preflight.synthetic_batch_size")
    required_preflight = {
        "model_forward_and_output_shape",
        "segmentation_only_backward",
        "no_classifier_gradient_in_segmenter_fit",
        "both_classifier_strategies_forward_backward_and_output_shape",
        "both_classifier_strategies_strict_roi_input_contract",
        "classifier_training_eye_eligibility_contract",
        "model_specific_classifier_trainability_policy",
        "roi_component_edge_cases",
        "outside_roi_logit_invariance",
        "crossfit_patient_disjointness",
        "environment_and_package_provenance",
    }
    observed_preflight = set(_sequence(preflight, "required_checks", "preflight"))
    if observed_preflight != required_preflight:
        raise ConfigError(f"preflight.required_checks must be exactly {sorted(required_preflight)}")
    _exact(
        list(_sequence(preflight, "required_prelaunch_audits", "preflight")),
        [
            {
                "round": 1,
                "name": "dual_classifier_contract_and_backward",
                "scope": "all_four_models_both_strategies",
                "required_status": "passed",
            },
            {
                "round": 2,
                "name": "synthetic_strict_roi_end_to_end_and_outside_roi_invariance",
                "scope": "all_four_models_both_strategies",
                "required_status": "passed",
            },
        ],
        "preflight.required_prelaunch_audits",
    )
    _exact(preflight.get("verification_artifact"), "verification/{model}.json", "preflight.verification_artifact")
    _exact(
        preflight.get("receipt_must_hash_verification_artifact"),
        True,
        "preflight.receipt_must_hash_verification_artifact",
    )
    _exact(
        preflight.get("train_segmenters_requires_all_four_preflight_receipts"),
        True,
        "preflight.train_segmenters_requires_all_four_preflight_receipts",
    )

    training = _mapping(cfg, "training", "config")
    _exact(training.get("max_epochs"), 60, "training.max_epochs")
    _exact(training.get("patience"), 10, "training.patience")
    _exact(training.get("batch_size"), 1, "training.batch_size")
    _exact(training.get("gradient_accumulation"), 4, "training.gradient_accumulation")
    _exact(training.get("weight_decay"), 0.0001, "training.weight_decay")
    _exact(training.get("gradient_clip"), 1.0, "training.gradient_clip")
    _exact(training.get("amp"), True, "training.amp")
    _exact(training.get("amp_dtype"), "bfloat16", "training.amp_dtype")
    _exact(training.get("num_workers"), 0, "training.num_workers")
    _exact(training.get("deterministic_algorithms_warn_only"), True, "training.deterministic_algorithms_warn_only")
    _exact(training.get("test_used_for_selection"), False, "training.test_used_for_selection")
    augmentation = _mapping(training, "augmentation", "training")
    _exact(augmentation.get("rotation_degrees"), 7.0, "training.augmentation.rotation_degrees")
    _exact(augmentation.get("translation_fraction"), 0.03, "training.augmentation.translation_fraction")
    _exact(augmentation.get("scale"), [0.95, 1.05], "training.augmentation.scale")
    _exact(augmentation.get("brightness"), [0.9, 1.1], "training.augmentation.brightness")
    _exact(augmentation.get("flips"), False, "training.augmentation.flips")
    _exact(augmentation.get("apply_same_spatial_transform_to_image_and_mask"), True, "training.augmentation.apply_same_spatial_transform_to_image_and_mask")
    _exact(augmentation.get("classifier_augmentation_occurs_after_strict_masking"), True, "training.augmentation.classifier_augmentation_occurs_after_strict_masking")
    offsets = _mapping(training, "seed_offsets", "training")
    expected_offset_keys = {"segmentation", "augmentation", "cross_fit", "classifier", "bootstrap"}
    if set(offsets) != expected_offset_keys:
        raise ConfigError(f"training.seed_offsets must contain exactly {sorted(expected_offset_keys)}")
    offset_values = [_positive_int(offsets[key], f"training.seed_offsets.{key}") for key in sorted(offsets)]
    if len(set(offset_values)) != len(offset_values):
        raise ConfigError("training.seed_offsets values must be unique")
    _exact(
        dict(offsets),
        {
            "segmentation": 10000,
            "augmentation": 20000,
            "cross_fit": 30000,
            "classifier": 40000,
            "bootstrap": 50000,
        },
        "training.seed_offsets",
    )
    early = _mapping(training, "early_stopping", "training")
    _exact(early.get("segmentation_monitor"), "validation_eye_dice", "training.early_stopping.segmentation_monitor")
    _exact(early.get("classifier_monitor"), "validation_eye_auroc", "training.early_stopping.classifier_monitor")
    _exact(early.get("classifier_tie_breaker"), "validation_eye_nll", "training.early_stopping.classifier_tie_breaker")
    _exact(early.get("mode"), "max", "training.early_stopping.mode")
    _exact(early.get("restore_best_weights"), True, "training.early_stopping.restore_best_weights")
    _exact(
        early.get("undefined_classifier_auroc_policy"),
        "use_negative_validation_eye_nll_if_defined_else_stop_and_flag",
        "training.early_stopping.undefined_classifier_auroc_policy",
    )
    _exact(
        early.get("no_evaluable_validation_eyes_policy"),
        "stop_and_lock_non_evaluable_all_test_decisions_abstain",
        "training.early_stopping.no_evaluable_validation_eyes_policy",
    )
    _exact(early.get("lock_requires_monitor_status"), True, "training.early_stopping.lock_requires_monitor_status")
    _exact(early.get("evaluation_frequency_epochs"), 1, "training.early_stopping.evaluation_frequency_epochs")
    _exact(early.get("minimum_delta"), 0.0, "training.early_stopping.minimum_delta")

    crossfit = _mapping(cfg, "cross_fitting", "config")
    _exact(crossfit.get("enabled"), True, "cross_fitting.enabled")
    _exact(crossfit.get("group_unit"), "patient", "cross_fitting.group_unit")
    _exact(crossfit.get("folds"), 5, "cross_fitting.folds")
    _exact(crossfit.get("stratify_by"), "label_3class", "cross_fitting.stratify_by")
    _exact(crossfit.get("train_roi_source"), "out_of_fold_predicted_mask", "cross_fitting.train_roi_source")
    _exact(crossfit.get("validation_roi_source"), "outer_train_full_fit_predicted_mask", "cross_fitting.validation_roi_source")
    _exact(crossfit.get("test_roi_source"), "outer_train_full_fit_predicted_mask", "cross_fitting.test_roi_source")
    _exact(crossfit.get("inner_fit_epoch_policy"), "outer_full_fit_selected_epoch", "cross_fitting.inner_fit_epoch_policy")
    _exact(
        crossfit.get("inner_holdout_labels_used_for_training_or_selection"),
        False,
        "cross_fitting.inner_holdout_labels_used_for_training_or_selection",
    )
    _exact(crossfit.get("test_generated_before_lock"), False, "cross_fitting.test_generated_before_lock")
    _exact(
        dict(_mapping(crossfit, "expected_primary_segmenter_fits", "cross_fitting")),
        {"outer_full": 20, "inner_out_of_fold": 100, "total": 120},
        "cross_fitting.expected_primary_segmenter_fits",
    )
    _exact(
        dict(_mapping(crossfit, "expected_base_study_fits", "cross_fitting")),
        {
            "segmenters": 120,
            "model_specific_classifiers": 20,
            "standardized_resnet18_classifiers": 20,
            "classifiers_total": 40,
            "learned_fits_total": 160,
        },
        "cross_fitting.expected_base_study_fits",
    )

    segmentation = _mapping(cfg, "segmentation", "config")
    _exact(segmentation.get("training_mode"), "segmentation_only", "segmentation.training_mode")
    _exact(segmentation.get("classifier_loss_weight"), 0.0, "segmentation.classifier_loss_weight")
    _exact(
        dict(_mapping(segmentation, "loss", "segmentation")),
        {
            "name": "binary_cross_entropy_plus_soft_dice",
            "binary_cross_entropy_weight": 1.0,
            "soft_dice_weight": 1.0,
            "per_model": {
                "yolo26": "native_yolo_instance_segmentation_loss",
                "vit_method2": "binary_cross_entropy_plus_soft_dice",
                "emcad": "binary_cross_entropy_plus_soft_dice",
                "sam2_unet": "binary_cross_entropy_plus_soft_dice",
            },
        },
        "segmentation.loss",
    )
    _exact(segmentation.get("checkpoint_selection"), "validation_eye_dice", "segmentation.checkpoint_selection")
    _exact(
        segmentation.get("checkpoint_tie_breakers"),
        ["validation_eye_iou", "earliest_epoch"],
        "segmentation.checkpoint_tie_breakers",
    )
    _exact(segmentation.get("threshold_selection_partition"), "validation", "segmentation.threshold_selection_partition")
    _exact(segmentation.get("threshold_selection_metric"), "eye_dice", "segmentation.threshold_selection_metric")
    _exact(segmentation.get("surface_dice_tolerance_pixels"), 2.0, "segmentation.surface_dice_tolerance_pixels")
    _increasing_probabilities(segmentation.get("threshold_candidates"), "segmentation.threshold_candidates")
    _exact(
        segmentation.get("threshold_candidates"),
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "segmentation.threshold_candidates",
    )
    fixed_segmentation = _mapping(segmentation, "fixed_hyperparameters", "segmentation")
    if tuple(fixed_segmentation) != EXPECTED_MODELS:
        raise ConfigError("segmentation.fixed_hyperparameters must follow the four locked model order")
    for model in EXPECTED_MODELS:
        values = _mapping(fixed_segmentation, model, "segmentation.fixed_hyperparameters")
        if set(values) != {"learning_rate"}:
            raise ConfigError(f"segmentation.fixed_hyperparameters.{model} must contain only learning_rate")
        _number(values["learning_rate"], f"segmentation.fixed_hyperparameters.{model}.learning_rate", minimum=0.0, strict=True)
    _exact(
        {model: dict(fixed_segmentation[model]) for model in EXPECTED_MODELS},
        {
            "yolo26": {"learning_rate": 0.0003},
            "vit_method2": {"learning_rate": 0.0003},
            "emcad": {"learning_rate": 0.0001},
            "sam2_unet": {"learning_rate": 0.0003},
        },
        "segmentation.fixed_hyperparameters",
    )
    initialization = _mapping(segmentation, "initialization", "segmentation")
    if tuple(initialization) != EXPECTED_MODELS:
        raise ConfigError("segmentation.initialization must follow the four locked model order")
    random_models = {"vit_method2"}
    expected_pretrained_modes = {
        "yolo26": "verified_pretrained_segmentation_weights",
        "emcad": "verified_pretrained_encoder_weights",
        "sam2_unet": "verified_pretrained_encoder_weights",
    }
    for model in EXPECTED_MODELS:
        values = _mapping(initialization, model, "segmentation.initialization")
        if model in random_models:
            _exact(values.get("mode"), "random_segmentation_weights", f"segmentation.initialization.{model}.mode")
            _exact(values.get("checkpoint"), None, f"segmentation.initialization.{model}.checkpoint")
            _exact(values.get("sha256"), None, f"segmentation.initialization.{model}.sha256")
        else:
            _exact(values.get("mode"), expected_pretrained_modes[model], f"segmentation.initialization.{model}.mode")
            if not isinstance(values.get("checkpoint"), str):
                raise ConfigError(f"segmentation.initialization.{model}.checkpoint must be project-relative")
            resolve_project_path(values["checkpoint"])
            _sha256(values.get("sha256"), f"segmentation.initialization.{model}.sha256")
            if not isinstance(values.get("source"), str) or not values["source"].startswith("https://"):
                raise ConfigError(f"segmentation.initialization.{model}.source must be HTTPS")
    _exact(
        segmentation.get("metrics"),
        [
            "dice",
            "iou",
            "pixel_sensitivity",
            "pixel_specificity",
            "pixel_precision",
            "hausdorff95",
            "average_symmetric_surface_distance",
            "surface_dice",
            "absolute_area_error",
            "relative_area_error",
            "centroid_distance",
        ],
        "segmentation.metrics",
    )
    _exact(segmentation.get("aggregation_levels"), ["frame", "eye", "patient"], "segmentation.aggregation_levels")

    roi = _mapping(cfg, "roi", "config")
    strict_equals = {
        "mask_source": "predicted_probability_only",
        "hard_mask": True,
        "one_anatomical_roi_per_frame": True,
        "connectivity": 8,
        "allow_ground_truth_at_inference": False,
        "allow_full_image_fallback": False,
        "allow_global_feature_branch": False,
        "empty_policy": "abstain_frame",
        "ambiguous_multi_component_policy": "abstain_frame",
        "implausibly_large_policy": "abstain_frame",
        "edge_touch_policy": "abstain_frame",
        "outside_roi": "fixed_neutral_after_normalization",
        "crop": "tight_component_bbox_then_aspect_preserving_letterbox",
    }
    for key, expected in strict_equals.items():
        _exact(roi.get(key), expected, f"roi.{key}")
    _exact(roi.get("input_size"), [224, 224], "roi.input_size")
    _exact(roi.get("neutral_normalized_value"), [0.0, 0.0, 0.0], "roi.neutral_normalized_value")
    _exact(roi.get("include_unmasked_bbox_context"), False, "roi.include_unmasked_bbox_context")
    min_area = _mapping(roi, "minimum_area", "roi")
    _exact(min_area.get("reference_partition"), "train", "roi.minimum_area.reference_partition")
    _exact(min_area.get("reference_mask"), "ground_truth", "roi.minimum_area.reference_mask")
    _exact(min_area.get("quantile"), 0.01, "roi.minimum_area.quantile")
    _exact(min_area.get("multiplier"), 0.5, "roi.minimum_area.multiplier")
    _exact(
        min_area.get("formula"),
        "0.5 * Q01(train_ground_truth_roi_area_fraction)",
        "roi.minimum_area.formula",
    )
    maximum_area = _mapping(roi, "maximum_area", "roi")
    _exact(maximum_area.get("reference_partition"), "train", "roi.maximum_area.reference_partition")
    _exact(maximum_area.get("reference_mask"), "ground_truth", "roi.maximum_area.reference_mask")
    _exact(maximum_area.get("quantile"), 0.99, "roi.maximum_area.quantile")
    _exact(maximum_area.get("multiplier"), 1.5, "roi.maximum_area.multiplier")
    _exact(
        maximum_area.get("formula"),
        "1.5 * Q99(train_ground_truth_roi_area_fraction)",
        "roi.maximum_area.formula",
    )
    cleanup = _mapping(roi, "component_cleanup", "roi")
    _exact(cleanup.get("remove_components_below_minimum_area"), True, "roi.component_cleanup.remove_components_below_minimum_area")
    _exact(cleanup.get("fill_internal_holes"), True, "roi.component_cleanup.fill_internal_holes")
    _exact(cleanup.get("maximum_hole_area_pixels"), 64, "roi.component_cleanup.maximum_hole_area_pixels")
    _exact(cleanup.get("morphological_closing"), False, "roi.component_cleanup.morphological_closing")
    _exact(cleanup.get("morphological_closing_iterations"), 0, "roi.component_cleanup.morphological_closing_iterations")
    _exact(cleanup.get("never_merge_disconnected_anatomical_candidates"), True, "roi.component_cleanup.never_merge_disconnected_anatomical_candidates")
    _exact(cleanup.get("reject_frame_if_any_component_exceeds_maximum_area"), True, "roi.component_cleanup.reject_frame_if_any_component_exceeds_maximum_area")
    dominance = _mapping(roi, "dominance", "roi")
    _exact(dominance.get("metric"), "component_mean_probability", "roi.dominance.metric")
    _exact(
        dominance.get("ratio_definition"),
        "best_score_divided_by_second_best_score",
        "roi.dominance.ratio_definition",
    )
    _exact(dominance.get("threshold_selected_on"), "validation", "roi.dominance.threshold_selected_on")
    _increasing_positive_numbers(dominance.get("candidate_ratios"), "roi.dominance.candidate_ratios")
    _exact(dominance.get("candidate_ratios"), [1.1, 1.25, 1.5, 2.0], "roi.dominance.candidate_ratios")
    _exact(
        dominance.get("selection_objective"),
        "postprocessed_eye_dice_subject_to_eye_coverage",
        "roi.dominance.selection_objective",
    )
    _exact(
        dominance.get("postprocessed_eye_dice_definition"),
        "mean_of_seven_frame_dice_with_abstained_frame_prediction_empty_and_dice_zero",
        "roi.dominance.postprocessed_eye_dice_definition",
    )
    _exact(
        dominance.get("eye_coverage_definition"),
        "fraction_of_validation_eyes_with_at_least_four_valid_frames",
        "roi.dominance.eye_coverage_definition",
    )
    _exact(dominance.get("minimum_validation_coverage"), 0.8, "roi.dominance.minimum_validation_coverage")
    _exact(
        dominance.get("infeasible_fallback"),
        "highest_eye_coverage_then_postprocessed_eye_dice_and_flag_in_lock",
        "roi.dominance.infeasible_fallback",
    )
    _exact(dominance.get("ties_or_insufficient_dominance"), "abstain_frame", "roi.dominance.ties_or_insufficient_dominance")
    invariance = _mapping(roi, "anti_leakage_invariance_test", "roi")
    _exact(invariance.get("enabled"), True, "roi.anti_leakage_invariance_test.enabled")
    _exact(
        invariance.get("operation"),
        "randomize_every_pixel_outside_fixed_mask",
        "roi.anti_leakage_invariance_test.operation",
    )
    _exact(
        invariance.get("maximum_absolute_logit_difference"),
        0.000001,
        "roi.anti_leakage_invariance_test.maximum_absolute_logit_difference",
    )

    classifier = _mapping(cfg, "classifier", "config")
    _exact(
        classifier.get("strategy_order"),
        ["model_specific", "standardized_resnet18"],
        "classifier.strategy_order",
    )
    primary_classifier = _mapping(classifier, "primary", "classifier")
    _exact(primary_classifier.get("name"), "model_specific", "classifier.primary.name")
    _exact(primary_classifier.get("role"), "primary_estimand", "classifier.primary.role")
    _exact(
        primary_classifier.get("estimand_id"),
        "E1_model_specific_strict_roi_system",
        "classifier.primary.estimand_id",
    )
    _exact(
        dict(_mapping(primary_classifier, "architecture_by_segmenter", "classifier.primary")),
        {
            "yolo26": "yolo26s_roi_classifier",
            "vit_method2": "vit_method2_roi_classifier",
            "emcad": "pvt_v2_b0_roi_classifier",
            "sam2_unet": "sam2_hiera_tiny_roi_classifier",
        },
        "classifier.primary.architecture_by_segmenter",
    )
    _exact(
        primary_classifier.get("initialization_source"),
        "per_segmenter_locked_initialization_with_new_binary_head",
        "classifier.primary.initialization_source",
    )
    _exact(
        dict(_mapping(primary_classifier, "initialization_by_segmenter", "classifier.primary")),
        {
            "yolo26": {
                "pretrained": True,
                "checkpoint": "binary_study/models/vendor/yolo26_sources/weights/yolo26s-seg.pt",
                "sha256": "3da1d83e31caec96f9300eb4064f4f62882c133c7c264d63dfe61a7c197837a4",
                "source": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-seg.pt",
                "network_download_during_run": False,
            },
            "vit_method2": {
                "pretrained": False,
                "checkpoint": None,
                "sha256": None,
                "source": "random_initialization_no_compatible_locked_checkpoint",
                "network_download_during_run": False,
            },
            "emcad": {
                "pretrained": True,
                "checkpoint": "binary_study/weights/emcad_joint/pvt_v2_b0.pth",
                "sha256": "fbb931ad59ab4d64e3f4370991326803e3af882fcaa877b81550d368d1fbab1f",
                "source": "https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b0.pth",
                "network_download_during_run": False,
            },
            "sam2_unet": {
                "pretrained": True,
                "checkpoint": "binary_study/weights/sam2_unet/sam2_hiera_tiny.pt",
                "sha256": "65b50056e05bcb13694174f51bb6da89c894b57b75ccdf0ba6352c597c5d1125",
                "source": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt",
                "network_download_during_run": False,
            },
        },
        "classifier.primary.initialization_by_segmenter",
    )
    _exact(
        dict(_mapping(primary_classifier, "trainability_by_segmenter", "classifier.primary")),
        {
            "yolo26": {
                "policy_id": "full_yolo26_backbone_neck_finetune_plus_new_binary_head_native_segmentation_head_frozen",
                "trainable_components": ["yolo26_backbone", "yolo26_neck", "new_binary_head"],
                "frozen_components": ["native_yolo_detection_segmentation_head_forward_only"],
                "disabled_components": ["native_yolo_segmentation_loss", "legacy_joint_classifier"],
            },
            "vit_method2": {
                "policy_id": "full_method2_encoder_cross_attention_and_new_binary_head_training",
                "trainable_components": ["patch_embedding", "transformer_encoder", "cross_attention", "new_binary_head"],
                "frozen_components": [],
                "disabled_components": ["segmentation_decoder", "legacy_joint_classifier"],
            },
            "emcad": {
                "policy_id": "full_pvt_v2_b0_finetune_plus_new_binary_head",
                "trainable_components": ["pvt_v2_b0_full_backbone", "new_binary_head"],
                "frozen_components": [],
                "disabled_components": ["pvt_imagenet_head", "emcad_segmentation_decoder", "legacy_joint_classifier"],
            },
            "sam2_unet": {
                "policy_id": "frozen_original_hiera_trunk_plus_trainable_prompt_learn_adapters_and_new_binary_head",
                "trainable_components": ["adapter_prompt_learn", "new_binary_head"],
                "frozen_components": ["original_hiera_trunk_including_patch_position_and_native_attention_blocks"],
                "disabled_components": ["sam2_unet_segmentation_decoder", "legacy_joint_classifier"],
                "freeze_order": "freeze_original_hiera_before_wrapping_blocks_with_trainable_adapters",
                "gradient_through_frozen_trunk_to_adapters": True,
            },
        },
        "classifier.primary.trainability_by_segmenter",
    )
    _exact(
        primary_classifier.get("fitted_segmenter_frozen_during_classifier_training"),
        True,
        "classifier.primary.fitted_segmenter_frozen_during_classifier_training",
    )
    _exact(
        primary_classifier.get("require_parameter_audit_against_trainability_policy"),
        True,
        "classifier.primary.require_parameter_audit_against_trainability_policy",
    )
    _exact(
        primary_classifier.get("strict_roi_input"),
        "corresponding_segmenter_strict_predicted_roi_only",
        "classifier.primary.strict_roi_input",
    )
    _exact(
        primary_classifier.get("separate_weights_per_model_seed"),
        True,
        "classifier.primary.separate_weights_per_model_seed",
    )
    _exact(primary_classifier.get("seed_offset"), 0, "classifier.primary.seed_offset")
    _exact(primary_classifier.get("expected_fits"), 20, "classifier.primary.expected_fits")

    secondary_classifier = _mapping(classifier, "secondary", "classifier")
    _exact(secondary_classifier.get("name"), "standardized_resnet18", "classifier.secondary.name")
    _exact(
        secondary_classifier.get("role"),
        "prespecified_secondary_standardized_analysis",
        "classifier.secondary.role",
    )
    _exact(
        secondary_classifier.get("estimand_id"),
        "E2_standardized_resnet18_strict_roi_system",
        "classifier.secondary.estimand_id",
    )
    _exact(
        secondary_classifier.get("architecture"),
        "torchvision_resnet18",
        "classifier.secondary.architecture",
    )
    _exact(
        secondary_classifier.get("shared_architecture_across_segmenters"),
        True,
        "classifier.secondary.shared_architecture_across_segmenters",
    )
    _exact(
        secondary_classifier.get("strict_roi_input"),
        "corresponding_segmenter_strict_predicted_roi_only",
        "classifier.secondary.strict_roi_input",
    )
    _exact(
        secondary_classifier.get("separate_weights_per_model_seed"),
        True,
        "classifier.secondary.separate_weights_per_model_seed",
    )
    _exact(secondary_classifier.get("seed_offset"), 500009, "classifier.secondary.seed_offset")
    _exact(secondary_classifier.get("expected_fits"), 20, "classifier.secondary.expected_fits")
    _exact(
        classifier.get("shared_roi_artifact_contract"),
        "byte_identical_roi_tensor_mask_validity_and_abstention_status_for_both_strategies_within_model_seed_frame",
        "classifier.shared_roi_artifact_contract",
    )
    _exact(
        dict(_mapping(classifier, "expected_core_fits", "classifier")),
        {"model_specific": 20, "standardized_resnet18": 20, "total": 40},
        "classifier.expected_core_fits",
    )
    _exact(
        classifier.get("primary_features"),
        "strict_roi_masked_pixels_no_explicit_geometry_vector",
        "classifier.primary_features",
    )
    _exact(
        classifier.get("include_explicit_geometry_features_in_primary"),
        False,
        "classifier.include_explicit_geometry_features_in_primary",
    )
    _exact(classifier.get("classes"), 2, "classifier.classes")
    _exact(
        classifier.get("undefined_threshold_policy"),
        "mark_unavailable_and_all_test_decisions_abstain",
        "classifier.undefined_threshold_policy",
    )
    weights = _mapping(secondary_classifier, "pretrained_weights", "classifier.secondary")
    _exact(weights.get("required"), True, "classifier.secondary.pretrained_weights.required")
    _exact(weights.get("name"), "ResNet18_Weights.IMAGENET1K_V1", "classifier.secondary.pretrained_weights.name")
    _exact(
        weights.get("url"),
        "https://download.pytorch.org/models/resnet18-f37072fd.pth",
        "classifier.secondary.pretrained_weights.url",
    )
    _exact(
        weights.get("sha256"),
        "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
        "classifier.secondary.pretrained_weights.sha256",
    )
    _exact(
        weights.get("cache_path"),
        "predicted_roi_study/weights/resnet18-f37072fd.pth",
        "classifier.secondary.pretrained_weights.cache_path",
    )
    resolve_project_path(weights["cache_path"])
    _exact(weights.get("license"), "BSD-3-Clause", "classifier.secondary.pretrained_weights.license")
    _exact(weights.get("source"), "torchvision", "classifier.secondary.pretrained_weights.source")
    _exact(
        weights.get("network_download_during_run"),
        False,
        "classifier.secondary.pretrained_weights.network_download_during_run",
    )
    normalization = _mapping(classifier, "normalization", "classifier")
    _exact(normalization.get("mean"), [0.485, 0.456, 0.406], "classifier.normalization.mean")
    _exact(normalization.get("std"), [0.229, 0.224, 0.225], "classifier.normalization.std")
    _exact(classifier.get("optimizer"), "adamw", "classifier.optimizer")
    _exact(classifier.get("scheduler"), "cosine_annealing", "classifier.scheduler")
    _exact(classifier.get("learning_rate"), 0.0001, "classifier.learning_rate")
    _exact(classifier.get("weight_decay"), 0.0001, "classifier.weight_decay")
    _exact(classifier.get("batch_size"), 16, "classifier.batch_size")
    _exact(classifier.get("gradient_accumulation"), 1, "classifier.gradient_accumulation")
    _exact(classifier.get("loss"), "class_weighted_cross_entropy", "classifier.loss")
    _exact(
        classifier.get("loss_weighting"),
        "patient_class_balanced_and_equal_total_weight_per_eye",
        "classifier.loss_weighting",
    )
    _exact(
        classifier.get("loss_weight_formula"),
        "w_frame=1/(2*N_evaluable_training_eyes_in_class*N_valid_frames_in_eye)_then_normalize_mean_weight_to_one",
        "classifier.loss_weight_formula",
    )
    _exact(classifier.get("class_weight_source"), "outer_training_patients_only", "classifier.class_weight_source")
    _exact(classifier.get("threshold_selection_partition"), "validation", "classifier.threshold_selection_partition")
    _exact(classifier.get("threshold_selection_metric"), "balanced_accuracy", "classifier.threshold_selection_metric")
    _increasing_probabilities(classifier.get("threshold_candidates"), "classifier.threshold_candidates")
    _exact(
        classifier.get("threshold_candidates"),
        [
            0.05,
            0.1,
            0.15,
            0.2,
            0.25,
            0.3,
            0.35,
            0.4,
            0.45,
            0.5,
            0.55,
            0.6,
            0.65,
            0.7,
            0.75,
            0.8,
            0.85,
            0.9,
            0.95,
        ],
        "classifier.threshold_candidates",
    )
    aggregation = _mapping(cfg, "aggregation", "config")
    _exact(aggregation.get("frame_to_eye"), "arithmetic_mean_of_valid_frame_probabilities", "aggregation.frame_to_eye")
    _exact(aggregation.get("minimum_valid_frames"), 4, "aggregation.minimum_valid_frames")
    _exact(aggregation.get("insufficient_frames"), "abstain_eye", "aggregation.insufficient_frames")
    _exact(aggregation.get("eye_to_patient"), "arithmetic_mean_of_both_eye_probabilities", "aggregation.eye_to_patient")
    _exact(aggregation.get("require_both_eyes"), True, "aggregation.require_both_eyes")
    _exact(aggregation.get("insufficient_eyes"), "abstain_patient", "aggregation.insufficient_eyes")
    _exact(
        aggregation.get("ground_truth_roi_overlap_used_in_deployment_decision"),
        False,
        "aggregation.ground_truth_roi_overlap_used_in_deployment_decision",
    )

    calibration = _mapping(cfg, "calibration", "config")
    _exact(calibration.get("report_raw_probabilities"), True, "calibration.report_raw_probabilities")
    _exact(calibration.get("fit_partition"), "validation", "calibration.fit_partition")
    _exact(calibration.get("method"), "temperature_scaling", "calibration.method")
    _exact(
        calibration.get("fit_separately_per_classifier_strategy_model_seed_level"),
        True,
        "calibration.fit_separately_per_classifier_strategy_model_seed_level",
    )
    _exact(calibration.get("levels"), ["eye", "patient"], "calibration.levels")
    _exact(
        calibration.get("undefined_fit_policy"),
        "mark_unavailable_without_default_temperature",
        "calibration.undefined_fit_policy",
    )
    _exact(
        calibration.get("classification_threshold_if_temperature_unavailable"),
        "select_on_raw_validation_probability_and_flag",
        "calibration.classification_threshold_if_temperature_unavailable",
    )
    required_metrics = {"brier", "nll", "ece_equal_width", "ece_equal_mass", "calibration_intercept", "calibration_slope"}
    metrics = set(_sequence(calibration, "metrics", "calibration"))
    if not required_metrics <= metrics:
        raise ConfigError(f"calibration.metrics is missing {sorted(required_metrics - metrics)}")
    _positive_int(calibration.get("ece_bins"), "calibration.ece_bins", minimum=2)
    _exact(calibration.get("temperature_bounds"), [0.05, 20.0], "calibration.temperature_bounds")
    _exact(calibration.get("probability_clip_epsilon"), 1e-7, "calibration.probability_clip_epsilon")
    ece = _mapping(calibration, "ece_definition", "calibration")
    _exact(ece.get("equal_width"), "ten_[left_closed_right_open]_bins_with_probability_one_in_final_bin", "calibration.ece_definition.equal_width")
    _exact(ece.get("equal_mass"), "stable_probability_sort_then_numpy_array_split_into_min_ten_n_bins_ties_may_split", "calibration.ece_definition.equal_mass")
    _exact(ece.get("weighting"), "bin_count_divided_by_evaluable_count", "calibration.ece_definition.weighting")
    band = _mapping(calibration, "reliability_band", "calibration")
    _exact(band.get("resampling_unit"), "patient_cluster", "calibration.reliability_band.resampling_unit")
    _exact(band.get("interval"), "percentile_95", "calibration.reliability_band.interval")
    _positive_int(band.get("draws"), "calibration.reliability_band.draws", minimum=1000)
    _exact(calibration.get("test_refitting"), False, "calibration.test_refitting")
    _exact(
        calibration.get("plots"),
        ["reliability_diagram_with_bootstrap_band", "calibration_density"],
        "calibration.plots",
    )
    _exact(calibration.get("small_sample_caution"), True, "calibration.small_sample_caution")

    statistics = _mapping(cfg, "statistics", "config")
    _exact(statistics.get("confidence_level"), 0.95, "statistics.confidence_level")
    _exact(statistics.get("bootstrap_draws"), 5000, "statistics.bootstrap_draws")
    _exact(statistics.get("bootstrap_seed"), 19051, "statistics.bootstrap_seed")
    _exact(statistics.get("bootstrap_unit"), "patient_cluster", "statistics.bootstrap_unit")
    _exact(statistics.get("preserve_both_eyes_and_seven_frames"), True, "statistics.preserve_both_eyes_and_seven_frames")
    _exact(
        statistics.get("confidence_interval"),
        "bias_corrected_and_accelerated_when_defined_else_percentile",
        "statistics.confidence_interval",
    )
    _exact(statistics.get("multiple_comparison_correction"), "holm", "statistics.multiple_comparison_correction")
    _exact(statistics.get("seed_summary"), "mean_and_sample_std", "statistics.seed_summary")
    _exact(statistics.get("also_report_each_seed"), True, "statistics.also_report_each_seed")
    _exact(
        statistics.get("do_not_treat_overlapping_seed_holdouts_as_independent"),
        True,
        "statistics.do_not_treat_overlapping_seed_holdouts_as_independent",
    )
    _exact(
        statistics.get("paired_model_comparisons"),
        [
            "paired_patient_cluster_bootstrap_metric_difference",
            "delong_auroc_within_seed",
            "mcnemar_on_paired_non_abstained_predictions",
        ],
        "statistics.paired_model_comparisons",
    )
    _exact(statistics.get("paired_patient_cluster_bootstrap_is_primary_comparison"), True, "statistics.paired_patient_cluster_bootstrap_is_primary_comparison")
    _exact(statistics.get("primary_effect_interval"), "paired_patient_cluster_bca_else_percentile", "statistics.primary_effect_interval")
    _exact(statistics.get("primary_hypothesis_test"), "whole_patient_paired_model_label_swap_randomization_plus_one", "statistics.primary_hypothesis_test")
    _exact(statistics.get("paired_calibration_loss_tests"), ["brier", "nll"], "statistics.paired_calibration_loss_tests")
    _exact(statistics.get("paired_calibration_common_evaluable_only"), True, "statistics.paired_calibration_common_evaluable_only")
    _exact(
        statistics.get("paired_segmentation_metrics"),
        ["dice", "iou", "surface_dice", "hausdorff95", "average_symmetric_surface_distance", "centroid_distance"],
        "statistics.paired_segmentation_metrics",
    )
    _exact(statistics.get("paired_segmentation_representations"), ["postprocessed", "raw"], "statistics.paired_segmentation_representations")
    _exact(
        statistics.get("delong_scope"),
        "common_evaluable_subset_only_descriptive_sensitivity_analysis_due_to_two_eye_within_patient_dependence",
        "statistics.delong_scope",
    )
    holm = _mapping(statistics, "holm_families", "statistics")
    _exact(
        dict(holm),
        {
            "primary_model_specific": "E1_all_6_complete_system_pairs_by_5_seeds_for_eye_failure_aware_balanced_accuracy_30_hypotheses",
            "secondary_standardized_resnet18": "E2_all_6_segmenter_pairs_by_5_seeds_for_eye_failure_aware_balanced_accuracy_30_hypotheses",
            "within_segmenter_strategy": "E3_all_4_segmenters_by_5_seeds_for_model_specific_minus_standardized_eye_failure_aware_balanced_accuracy_20_hypotheses",
            "other_metrics_model_specific": "separate_family_for_each_metric_and_level_across_all_6_pairs_by_5_seeds",
            "other_metrics_standardized_resnet18": "separate_family_for_each_metric_and_level_across_all_6_pairs_by_5_seeds",
            "calibration_model_specific": "separate_family_for_each_calibration_metric_and_level_across_all_6_pairs_by_5_seeds",
            "calibration_standardized_resnet18": "separate_family_for_each_calibration_metric_and_level_across_all_6_pairs_by_5_seeds",
            "ablations_model_specific": "separate_family_for_each_test_and_level_within_each_segmenter_across_predeclared_arms_and_5_seeds",
        },
        "statistics.holm_families",
    )
    _exact(
        dict(_mapping(statistics, "comparison_sets", "statistics")),
        {
            "primary_model_specific": {
                "estimand_id": "E1_model_specific_strict_roi_system",
                "classifier_strategy": "model_specific",
                "contrasts_per_seed": 6,
                "seeds": 5,
                "planned_hypotheses": 30,
                "effect": "paired_difference_in_eye_failure_aware_balanced_accuracy",
                "interval": "paired_patient_cluster_bca_else_percentile",
                "test": "whole_patient_paired_model_label_swap_randomization_plus_one",
                "multiplicity_family": "primary_model_specific",
            },
            "secondary_standardized_resnet18": {
                "estimand_id": "E2_standardized_resnet18_strict_roi_system",
                "classifier_strategy": "standardized_resnet18",
                "contrasts_per_seed": 6,
                "seeds": 5,
                "planned_hypotheses": 30,
                "effect": "paired_difference_in_eye_failure_aware_balanced_accuracy",
                "interval": "paired_patient_cluster_bca_else_percentile",
                "test": "whole_patient_paired_model_label_swap_randomization_plus_one",
                "multiplicity_family": "secondary_standardized_resnet18",
            },
            "within_segmenter_strategy": {
                "estimand_id": "E3_model_specific_minus_standardized_within_segmenter",
                "classifier_strategies": ["model_specific", "standardized_resnet18"],
                "contrasts_per_seed": 4,
                "seeds": 5,
                "planned_hypotheses": 20,
                "effect": "paired_model_specific_minus_standardized_difference_in_eye_failure_aware_balanced_accuracy",
                "interval": "paired_patient_cluster_bca_else_percentile",
                "test": "whole_patient_paired_strategy_label_swap_randomization_plus_one",
                "multiplicity_family": "within_segmenter_strategy",
            },
        },
        "statistics.comparison_sets",
    )
    _exact(statistics.get("alpha"), 0.05, "statistics.alpha")
    _exact(statistics.get("effect_sizes_required"), True, "statistics.effect_sizes_required")

    evaluation = _mapping(cfg, "evaluation", "config")
    _exact(
        dict(_mapping(evaluation, "estimands", "evaluation")),
        {
            "primary": {
                "id": "E1_model_specific_strict_roi_system",
                "classifier_strategy": "model_specific",
                "unit": "eye",
                "population": "all_intended_held_out_test_eyes_including_abstentions",
                "endpoint": "failure_aware_balanced_accuracy",
                "contrast": "all_6_unordered_complete_system_pairs_within_each_seed",
                "interpretation": "joint_effect_of_segmenter_localization_and_its_model_specific_strict_roi_classifier",
            },
            "secondary_standardized": {
                "id": "E2_standardized_resnet18_strict_roi_system",
                "classifier_strategy": "standardized_resnet18",
                "unit": "eye",
                "population": "all_intended_held_out_test_eyes_including_abstentions",
                "endpoint": "failure_aware_balanced_accuracy",
                "contrast": "all_6_unordered_segmenter_pairs_within_each_seed_under_common_classifier_architecture",
                "interpretation": "segmenter_and_predicted_roi_effect_with_classifier_architecture_standardized",
            },
            "within_segmenter_strategy": {
                "id": "E3_model_specific_minus_standardized_within_segmenter",
                "classifier_strategies": ["model_specific", "standardized_resnet18"],
                "unit": "eye",
                "population": "all_intended_held_out_test_eyes_including_abstentions",
                "endpoint": "failure_aware_balanced_accuracy",
                "contrast": "model_specific_minus_standardized_resnet18_within_same_segmenter_seed_on_identical_strict_roi_artifacts",
                "interpretation": "incremental_classifier_strategy_effect_conditional_on_the_same_predicted_roi_stream",
            },
        },
        "evaluation.estimands",
    )
    _exact(evaluation.get("classification_levels"), ["eye", "patient"], "evaluation.classification_levels")
    _exact(evaluation.get("primary_level"), "eye", "evaluation.primary_level")
    _exact(
        evaluation.get("primary_endpoint"),
        "failure_aware_balanced_accuracy",
        "evaluation.primary_endpoint",
    )
    _exact(evaluation.get("discrimination_metrics"), ["auroc", "average_precision"], "evaluation.discrimination_metrics")
    _exact(
        evaluation.get("threshold_metrics"),
        ["sensitivity", "specificity", "ppv", "npv", "accuracy", "balanced_accuracy", "f1", "mcc"],
        "evaluation.threshold_metrics",
    )
    _exact(
        evaluation.get("selective_prediction_metrics"),
        [
            "coverage",
            "class_conditional_coverage",
            "selective_risk",
            "aurc",
            "failure_aware_accuracy",
            "failure_aware_sensitivity",
            "failure_aware_specificity",
        ],
        "evaluation.selective_prediction_metrics",
    )
    risk = _mapping(evaluation, "risk_coverage_definition", "evaluation")
    _exact(risk.get("confidence"), "absolute_probability_distance_from_locked_decision_threshold", "evaluation.risk_coverage_definition.confidence")
    _exact(risk.get("ranking"), "descending_confidence_stable_ties", "evaluation.risk_coverage_definition.ranking")
    _exact(risk.get("segmentation_gate_abstentions"), "append_at_lowest_confidence_and_count_as_errors", "evaluation.risk_coverage_definition.segmentation_gate_abstentions")
    _exact(risk.get("aurc"), "mean_cumulative_risk_over_all_attainable_coverage_steps", "evaluation.risk_coverage_definition.aurc")
    localized = _mapping(evaluation, "localized_success", "evaluation")
    _exact(localized.get("enabled"), True, "evaluation.localized_success.enabled")
    _exact(
        localized.get("definition"),
        "correct_classification_and_gt_roi_hit_in_at_least_four_of_seven_frames",
        "evaluation.localized_success.definition",
    )
    _exact(localized.get("gt_used_for_retrospective_evaluation_only"), True, "evaluation.localized_success.gt_used_for_retrospective_evaluation_only")
    _exact(localized.get("frame_hit_iou_threshold"), 0.5, "evaluation.localized_success.frame_hit_iou_threshold")
    _exact(localized.get("minimum_hit_frames"), 4, "evaluation.localized_success.minimum_hit_frames")
    confusion = _mapping(evaluation, "confusion_matrix", "evaluation")
    _exact(confusion.get("primary"), "two_true_classes_by_control_disease_abstain", "evaluation.confusion_matrix.primary")
    _exact(confusion.get("secondary"), "two_by_two_among_non_abstained_only", "evaluation.confusion_matrix.secondary")
    _exact(
        evaluation.get("clinical_utility"),
        ["decision_curve_net_benefit", "standardized_net_benefit"],
        "evaluation.clinical_utility",
    )
    dca = _mapping(evaluation, "decision_curve", "evaluation")
    _exact(dca.get("threshold_start"), 0.01, "evaluation.decision_curve.threshold_start")
    _exact(dca.get("threshold_stop"), 0.99, "evaluation.decision_curve.threshold_stop")
    _exact(dca.get("threshold_step"), 0.01, "evaluation.decision_curve.threshold_step")
    _exact(dca.get("positive_action"), "trigger_specialist_evaluation_for_papilledema_or_pseudopapilledema", "evaluation.decision_curve.positive_action")
    _exact(dca.get("abstention_action"), "no_model_triggered_intervention", "evaluation.decision_curve.abstention_action")
    _exact(dca.get("denominator"), "complete_intended_cohort_including_abstentions", "evaluation.decision_curve.denominator")

    ablations = _mapping(cfg, "ablations", "config")
    _exact(ablations.get("predeclared"), True, "ablations.predeclared")
    _exact(ablations.get("primary_must_be_evaluated_before_ablations"), True, "ablations.primary_must_be_evaluated_before_ablations")
    _exact(ablations.get("separate_checkpoints_and_locks"), True, "ablations.separate_checkpoints_and_locks")
    _exact(ablations.get("all_ablation_validation_locks_before_any_ablation_test"), True, "ablations.all_ablation_validation_locks_before_any_ablation_test")
    _exact(ablations.get("classifier_strategy"), "model_specific", "ablations.classifier_strategy")
    _exact(
        ablations.get("standardized_resnet18_is_secondary_comparator_not_ablation"),
        True,
        "ablations.standardized_resnet18_is_secondary_comparator_not_ablation",
    )
    _exact(ablations.get("expected_validation_locks"), 320, "ablations.expected_validation_locks")
    _exact(ablations.get("expected_test_evaluations"), 320, "ablations.expected_test_evaluations")
    _exact(ablations.get("expected_classifier_fits"), 180, "ablations.expected_classifier_fits")
    _exact(
        ablations.get("expected_no_refit_analytical_evaluations"),
        140,
        "ablations.expected_no_refit_analytical_evaluations",
    )
    _exact(
        dict(_mapping(ablations, "lock_breakdown", "ablations")),
        {
            "model_specific_learned": {
                "arms": [
                    "whole_image_baseline",
                    "gt_roi_oracle",
                    "bbox_context",
                    "largest_component_without_quality_gates",
                    "geometry_only",
                    "mask_only",
                    "background_only_negative_control",
                    "random_roi_negative_control",
                    "appearance_plus_geometry",
                ],
                "arm_count": 9,
                "seeds": 5,
                "models": 4,
                "locks": 180,
            },
            "analytical_no_refit": {
                "arms": ["aggregation_rule", "minimum_valid_frames"],
                "variant_count": 7,
                "seeds": 5,
                "models": 4,
                "locks": 140,
            },
        },
        "ablations.lock_breakdown",
    )
    _exact(ablations.get("summary"), "separate_immutable_ablation_summary", "ablations.summary")
    required_ablations = [
        "whole_image_baseline",
        "gt_roi_oracle",
        "bbox_context",
        "largest_component_without_quality_gates",
        "geometry_only",
        "mask_only",
        "background_only_negative_control",
        "random_roi_negative_control",
        "aggregation_rule",
        "minimum_valid_frames",
        "appearance_plus_geometry",
    ]
    experiments = _sequence(ablations, "experiments", "ablations")
    if any(not isinstance(item, Mapping) for item in experiments):
        raise ConfigError("ablations.experiments entries must be objects")
    names = [item.get("name") for item in experiments]
    _exact(names, required_ablations, "ablations.experiments names and order")
    _exact(experiments[8].get("variants"), ["mean", "median", "maximum", "predicted_mask_confidence_weighted_mean"], "ablations.aggregation_rule.variants")
    _exact(experiments[9].get("variants"), [1, 4, 7], "ablations.minimum_valid_frames.variants")

    access = _mapping(cfg, "test_access", "config")
    _exact(access.get("require_all_20_primary_locks_before_any_test"), True, "test_access.require_all_20_primary_locks_before_any_test")
    _exact(access.get("composite_model_seed_lock_count"), 20, "test_access.composite_model_seed_lock_count")
    _exact(
        access.get("classifier_strategies_per_composite_lock"),
        2,
        "test_access.classifier_strategies_per_composite_lock",
    )
    _exact(
        access.get("classification_levels_per_strategy"),
        2,
        "test_access.classification_levels_per_strategy",
    )
    _exact(access.get("expected_strategy_lock_records"), 40, "test_access.expected_strategy_lock_records")
    _exact(
        access.get("expected_level_specific_calibration_threshold_lock_records"),
        80,
        "test_access.expected_level_specific_calibration_threshold_lock_records",
    )
    _exact(
        access.get("expected_core_test_system_evaluations"),
        40,
        "test_access.expected_core_test_system_evaluations",
    )
    _exact(
        access.get("each_composite_lock_requires_both_classifier_strategies"),
        True,
        "test_access.each_composite_lock_requires_both_classifier_strategies",
    )
    _exact(access.get("immutable_evaluation_receipts"), True, "test_access.immutable_evaluation_receipts")
    _exact(access.get("allow_test_threshold_tuning"), False, "test_access.allow_test_threshold_tuning")
    _exact(access.get("allow_test_model_selection"), False, "test_access.allow_test_model_selection")
    _exact(access.get("allow_test_roi_quality_tuning"), False, "test_access.allow_test_roi_quality_tuning")
    _exact(access.get("test_predictions_written_once_per_model_seed"), True, "test_access.test_predictions_written_once_per_model_seed")
    _exact(
        access.get("failed_test_attempt_requires_same_code_config_and_locks"),
        True,
        "test_access.failed_test_attempt_requires_same_code_config_and_locks",
    )

    reporting = _mapping(cfg, "reporting", "config")
    _exact(
        reporting.get("primary_summarize_scope"),
        "primary_model_specific_20_model_seed_evaluations_only_immutable",
        "reporting.primary_summarize_scope",
    )
    _exact(
        reporting.get("secondary_standardized_summarize_scope"),
        "standardized_resnet18_20_model_seed_evaluations_prespecified_and_separate",
        "reporting.secondary_standardized_summarize_scope",
    )
    _exact(reporting.get("within_segmenter_strategy_contrasts"), True, "reporting.within_segmenter_strategy_contrasts")
    _exact(reporting.get("classifier_strategy_column_required"), True, "reporting.classifier_strategy_column_required")
    _exact(reporting.get("final_manuscript_synthesis"), "future_merge_of_primary_and_separate_ablation_summaries", "reporting.final_manuscript_synthesis")
    for key in (
        "per_seed_results",
        "five_seed_mean_plus_minus_sample_std",
        "eye_and_patient_tables",
        "calibration_tables_and_plots",
        "confusion_matrices_with_abstention",
        "report_abstention_reasons",
        "report_compute_and_parameter_counts",
        "report_environment_and_weight_provenance",
    ):
        _exact(reporting.get(key), True, f"reporting.{key}")
    _exact(
        reporting.get("segmentation_qualitative_examples"),
        "model_blind_sha256_min_globally_unique_per_seed_and_original_3class_label_before_model_output_access",
        "reporting.segmentation_qualitative_examples",
    )
    _exact(
        reporting.get("segmentation_qualitative_examples_per_class_per_seed"),
        1,
        "reporting.segmentation_qualitative_examples_per_class_per_seed",
    )
    _exact(
        reporting.get("failure_gallery_selection"),
        "outcome_conditioned_sha256_min_per_model_seed_status_illustrative_only",
        "reporting.failure_gallery_selection",
    )
    _exact(
        reporting.get("failure_gallery"),
        ["empty", "tiny", "oversegmentation", "ambiguous_multi_component", "edge_touch"],
        "reporting.failure_gallery",
    )
    _exact(
        reporting.get("flops_policy"),
        "not_reported_when_operator_coverage_and_end_to_end_path_are_not_comparable",
        "reporting.flops_policy",
    )


def load_config(path: str | Path | None = None, *, verify_source_files: bool = False) -> dict[str, Any]:
    """Load, validate and fingerprint the locked JSON protocol.

    Runtime metadata is added under ``_runtime`` and is not part of the JSON
    fingerprint.  ``verify_source_files`` is intentionally optional so unit
    tests and help commands do not hash the full dataset unnecessarily.
    """

    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    if not config_path.is_file():
        raise ConfigError(f"Configuration not found: {config_path}")
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ConfigError(f"Invalid JSON in {config_path}: {error}") from error
    validate_config(raw)
    cfg = copy.deepcopy(raw)
    cfg["_runtime"] = {
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "canonical_config_sha256": canonical_sha256(raw),
        "project_root": str(PROJECT_ROOT),
    }
    # Compatibility with the original binary study helpers.
    cfg["config_sha256"] = cfg["_runtime"]["config_sha256"]
    if verify_source_files:
        verify_locked_sources(cfg)
    return cfg


def config_without_runtime(cfg: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(cfg))
    value.pop("_runtime", None)
    value.pop("config_sha256", None)
    return value


def verify_locked_sources(cfg: Mapping[str, Any]) -> dict[str, str]:
    """Verify the immutable manifest and five reused patient membership files."""

    validate_config(config_without_runtime(cfg))
    observed: dict[str, str] = {}
    dataset = cfg["dataset"]
    manifest = resolve_project_path(dataset["manifest"], must_exist=True)
    observed["manifest"] = sha256_file(manifest)
    if observed["manifest"] != dataset["manifest_sha256"]:
        raise ConfigError("Dataset manifest differs from the protocol lock")
    for seed in EXPECTED_SEEDS:
        source = cfg["split_policy"]["sources"][str(seed)]
        path = resolve_project_path(source["path"], must_exist=True)
        observed[f"split_{seed}"] = sha256_file(path)
        if observed[f"split_{seed}"] != source["sha256"]:
            raise ConfigError(f"Patient split for seed {seed} differs from the protocol lock")
    return observed


def verify_pretrained_weights(cfg: Mapping[str, Any]) -> dict[str, str]:
    """Verify every locked segmenter and classifier initialization weight."""

    validate_config(config_without_runtime(cfg))
    observed: dict[str, str] = {}
    for model, spec in cfg["segmentation"]["initialization"].items():
        if spec["checkpoint"] is None:
            continue
        path = resolve_project_path(spec["checkpoint"], must_exist=True)
        digest = sha256_file(path)
        if digest != spec["sha256"]:
            raise ConfigError(f"Pretrained segmentation weight differs from lock: {model}")
        observed[f"segmenter_{model}"] = digest
    primary_initializations = cfg["classifier"]["primary"]["initialization_by_segmenter"]
    for model, spec in primary_initializations.items():
        if not spec["pretrained"]:
            continue
        path = resolve_project_path(spec["checkpoint"], must_exist=True)
        digest = sha256_file(path)
        if digest != spec["sha256"]:
            raise ConfigError(f"Pretrained model-specific classifier weight differs from lock: {model}")
        observed[f"classifier_model_specific_{model}"] = digest
    classifier = cfg["classifier"]["secondary"]["pretrained_weights"]
    path = resolve_project_path(classifier["cache_path"], must_exist=True)
    digest = sha256_file(path)
    if digest != classifier["sha256"]:
        raise ConfigError("Common ResNet-18 weight differs from the protocol lock")
    observed["classifier_standardized_resnet18"] = digest
    return observed


__all__ = [
    "ConfigError",
    "DEFAULT_CONFIG_PATH",
    "EXPECTED_MODELS",
    "EXPECTED_OUTPUT",
    "EXPECTED_PROTOCOL_VERSION",
    "EXPECTED_SEEDS",
    "EXPECTED_STUDY_ID",
    "PARTITIONS",
    "PROJECT_ROOT",
    "SCHEMA_VERSION",
    "canonical_sha256",
    "config_without_runtime",
    "load_config",
    "resolve_project_path",
    "sha256_file",
    "validate_config",
    "verify_locked_sources",
    "verify_pretrained_weights",
]
