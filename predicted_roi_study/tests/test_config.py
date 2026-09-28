from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import pytest

from predicted_roi_study.__main__ import COMMANDS, main
from predicted_roi_study.config import (
    ConfigError,
    EXPECTED_MODELS,
    EXPECTED_SEEDS,
    PROJECT_ROOT,
    config_without_runtime,
    load_config,
    resolve_project_path,
    sha256_file,
    validate_config,
    verify_locked_sources,
    verify_pretrained_weights,
)
from predicted_roi_study.protocol import (
    ProtocolGateError,
    assert_all_ablation_locks,
    assert_all_primary_locks,
    split_path,
    write_stage_receipt,
)


def pristine() -> dict:
    return config_without_runtime(load_config())


def assign(config: dict, dotted_path: str, value) -> None:
    keys = dotted_path.split(".")
    current = config
    for key in keys[:-1]:
        current = current[key]
    current[keys[-1]] = value


def test_default_config_has_exact_four_by_five_design() -> None:
    cfg = load_config()
    assert tuple(cfg["models"]) == EXPECTED_MODELS
    assert tuple(cfg["split_seeds"]) == EXPECTED_SEEDS
    assert len(cfg["models"]) * len(cfg["split_seeds"]) == 20
    assert cfg["cross_fitting"]["expected_primary_segmenter_fits"] == {
        "outer_full": 20,
        "inner_out_of_fold": 100,
        "total": 120,
    }
    assert cfg["classifier"]["strategy_order"] == ["model_specific", "standardized_resnet18"]
    assert cfg["classifier"]["expected_core_fits"] == {
        "model_specific": 20,
        "standardized_resnet18": 20,
        "total": 40,
    }
    assert cfg["cross_fitting"]["expected_base_study_fits"] == {
        "segmenters": 120,
        "model_specific_classifiers": 20,
        "standardized_resnet18_classifiers": 20,
        "classifiers_total": 40,
        "learned_fits_total": 160,
    }
    assert cfg["protocol_version"] == "1.2.0"
    assert cfg["output"] == "strict_roi_results_4model_v1_2_0"
    assert cfg["clean_run"]["import_upstream_artifacts"] is False
    assert cfg["clean_run"]["test_evaluation_in_seed_launcher"] is False
    assert cfg["_runtime"]["config_sha256"] == cfg["config_sha256"]


def test_protocol_document_version_matches_locked_config() -> None:
    cfg = load_config()
    protocol_text = (PROJECT_ROOT / "predicted_roi_study" / "PROTOCOL.md").read_text(
        encoding="utf-8"
    )
    assert f"Sürüm: {cfg['protocol_version']}" in {
        line.strip() for line in protocol_text.splitlines()[:6]
    }


def test_cli_exposes_only_explicit_staged_workflow() -> None:
    assert COMMANDS == (
        "prepare",
        "preflight",
        "train-segmenters",
        "build-rois",
        "train-classifiers",
        "lock",
        "evaluate",
        "summarize",
        "lock-ablations",
        "evaluate-ablations",
        "audit",
    )
    assert "all" not in COMMANDS


def test_clean_seed_launcher_is_parameterized_and_stops_before_test() -> None:
    launcher = PROJECT_ROOT / "scripts" / "run_strict_roi_clean_seed.ps1"
    text = launcher.read_text(encoding="utf-8")
    assert "[ValidateSet(17, 42, 2026, 3407, 9103)]" in text
    assert 'Invoke-StudyStage -Stage "train-segmenters"' in text
    assert 'Invoke-StudyStage -Stage "build-rois"' in text
    assert 'Invoke-StudyStage -Stage "train-classifiers"' in text
    assert 'Invoke-StudyStage -Stage "lock"' in text
    assert 'Invoke-StudyStage -Stage "evaluate"' not in text
    assert 'Invoke-StudyStage -Stage "audit"' not in text
    assert "-ResumeIncompleteSeed" in text
    assert "[System.IO.FileShare]::None" in text
    assert text.count("Test-Path -LiteralPath $testAccessPath -PathType Leaf") >= 3
    assert "[string]$ProjectRoot" not in text
    assert "[string]$ConfigPath" not in text
    assert "test_evaluation_started = $false" in text
    ordered_stages = [
        'Invoke-StudyStage -Stage "prepare"',
        'Invoke-StudyStage -Stage "preflight"',
        'Invoke-StudyStage -Stage "train-segmenters"',
        'Invoke-StudyStage -Stage "build-rois"',
        'Invoke-StudyStage -Stage "train-classifiers"',
        'Invoke-StudyStage -Stage "lock"',
    ]
    assert [text.index(stage) for stage in ordered_stages] == sorted(
        text.index(stage) for stage in ordered_stages
    )


def test_legacy_seed17_launchers_are_fail_closed() -> None:
    for name in (
        "run_strict_roi_seed17.ps1",
        "resume_strict_roi_seed17_when_gpu_free.ps1",
    ):
        text = (PROJECT_ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert text.lstrip().startswith("throw")
        assert "run_strict_roi_clean_seed.ps1" in text


@pytest.mark.parametrize(
    "command", ["train-segmenters", "build-rois", "train-classifiers", "lock"]
)
def test_clean_clinical_stages_require_one_explicit_seed(command: str) -> None:
    with pytest.raises(SystemExit):
        main([command, "--dry-run"])


def test_locked_manifest_and_reused_patient_splits_match_disk() -> None:
    observed = verify_locked_sources(load_config())
    assert set(observed) == {
        "manifest",
        "split_17",
        "split_42",
        "split_2026",
        "split_3407",
        "split_9103",
    }
    assert all(len(digest) == 64 for digest in observed.values())


def test_existing_pretrained_segmenter_weights_match_protocol_hashes() -> None:
    initialization = load_config()["segmentation"]["initialization"]
    assert {name for name, spec in initialization.items() if spec["checkpoint"] is None} == {
        "vit_method2",
    }
    for name in ("yolo26", "emcad", "sam2_unet"):
        path = resolve_project_path(initialization[name]["checkpoint"], must_exist=True)
        assert sha256_file(path) == initialization[name]["sha256"]


def test_all_pretrained_classifier_weights_match_protocol_hashes() -> None:
    observed = verify_pretrained_weights(load_config())
    assert set(observed) == {
        "segmenter_yolo26",
        "segmenter_emcad",
        "segmenter_sam2_unet",
        "classifier_model_specific_yolo26",
        "classifier_model_specific_emcad",
        "classifier_model_specific_sam2_unet",
        "classifier_standardized_resnet18",
    }
    assert all(len(digest) == 64 for digest in observed.values())


@pytest.mark.parametrize(
    ("path", "unsafe_value"),
    [
        ("split_policy.regenerate", True),
        ("clean_run.import_upstream_artifacts", True),
        ("clean_run.test_evaluation_in_seed_launcher", True),
        ("split_policy.unit", "frame"),
        ("roi.mask_source", "ground_truth"),
        ("roi.allow_ground_truth_at_inference", True),
        ("roi.allow_full_image_fallback", True),
        ("roi.allow_global_feature_branch", True),
        ("roi.outside_roi", "original_pixels"),
        ("roi.component_cleanup.maximum_hole_area_pixels", 128),
        ("roi.component_cleanup.morphological_closing_iterations", 1),
        ("aggregation.minimum_valid_frames", 1),
        ("aggregation.require_both_eyes", False),
        ("cross_fitting.enabled", False),
        ("cross_fitting.train_roi_source", "in_sample_predicted_mask"),
        ("cross_fitting.inner_holdout_labels_used_for_training_or_selection", True),
        ("training.test_used_for_selection", True),
        ("training.amp_dtype", "float16"),
        ("training.seed_offsets.cross_fit", 30001),
        ("classifier.strategy_order", ["standardized_resnet18", "model_specific"]),
        ("classifier.primary.name", "standardized_resnet18"),
        ("classifier.secondary.shared_architecture_across_segmenters", False),
        ("classifier.include_explicit_geometry_features_in_primary", True),
        ("classifier.primary.fitted_segmenter_frozen_during_classifier_training", False),
        ("classifier.primary.require_parameter_audit_against_trainability_policy", False),
        ("classifier.primary.trainability_by_segmenter.vit_method2.frozen_components", ["patch_embedding"]),
        ("classifier.primary.trainability_by_segmenter.yolo26.frozen_components", []),
        ("classifier.primary.trainability_by_segmenter.sam2_unet.trainable_components", ["new_binary_head"]),
        ("classifier.primary.trainability_by_segmenter.sam2_unet.gradient_through_frozen_trunk_to_adapters", False),
        ("classifier.scheduler", "reduce_on_plateau"),
        ("classifier.loss_weighting", "per_frame_class_balanced"),
        ("segmentation.surface_dice_tolerance_pixels", 1.0),
        ("segmentation.loss.per_model.yolo26", "binary_cross_entropy_plus_soft_dice"),
        ("roi.minimum_area.formula", "data_dependent_unlocked_formula"),
        ("calibration.temperature_bounds", [0.01, 100.0]),
        ("calibration.probability_clip_epsilon", 0.001),
        ("calibration.fit_separately_per_classifier_strategy_model_seed_level", False),
        ("evaluation.decision_curve.threshold_step", 0.05),
        ("test_access.require_all_20_primary_locks_before_any_test", False),
        ("test_access.allow_test_threshold_tuning", True),
    ],
)
def test_scientifically_unsafe_protocol_changes_are_rejected(path: str, unsafe_value) -> None:
    cfg = pristine()
    assign(cfg, path, unsafe_value)
    with pytest.raises(ConfigError):
        validate_config(cfg)


def test_model_and_seed_sets_are_ordered_and_immutable() -> None:
    cfg = pristine()
    cfg["models"] = list(reversed(cfg["models"]))
    with pytest.raises(ConfigError, match="four locked models"):
        validate_config(cfg)

    cfg = pristine()
    cfg["split_seeds"] = [17, 42, 2026, 3407]
    with pytest.raises(ConfigError, match="five locked seeds"):
        validate_config(cfg)


def test_roi_grid_is_segmentation_only_and_has_declared_fallback() -> None:
    dominance = load_config()["roi"]["dominance"]
    assert dominance["selection_objective"] == "postprocessed_eye_dice_subject_to_eye_coverage"
    assert dominance["minimum_validation_coverage"] == 0.8
    assert dominance["infeasible_fallback"] == "highest_eye_coverage_then_postprocessed_eye_dice_and_flag_in_lock"
    assert "class" not in dominance["selection_objective"]


def test_segmentation_losses_are_explicitly_locked_per_model() -> None:
    per_model = load_config()["segmentation"]["loss"]["per_model"]
    assert tuple(per_model) == EXPECTED_MODELS
    assert per_model["yolo26"] == "native_yolo_instance_segmentation_loss"
    assert {
        value for name, value in per_model.items() if name != "yolo26"
    } == {"binary_cross_entropy_plus_soft_dice"}


def test_ablation_count_is_predeclared_and_auditable() -> None:
    ablations = load_config()["ablations"]
    assert ablations["classifier_strategy"] == "model_specific"
    assert ablations["standardized_resnet18_is_secondary_comparator_not_ablation"] is True
    assert ablations["expected_validation_locks"] == 320
    assert ablations["expected_test_evaluations"] == 320
    assert ablations["expected_classifier_fits"] == 180
    assert ablations["expected_no_refit_analytical_evaluations"] == 140
    assert sum(part["locks"] for part in ablations["lock_breakdown"].values()) == 320


def test_dual_classifier_estimands_and_inference_families_are_predeclared() -> None:
    cfg = load_config()
    classifier = cfg["classifier"]
    assert classifier["primary"]["name"] == "model_specific"
    assert tuple(classifier["primary"]["architecture_by_segmenter"]) == EXPECTED_MODELS
    assert classifier["secondary"]["name"] == "standardized_resnet18"
    assert classifier["primary"]["seed_offset"] == 0
    assert classifier["secondary"]["seed_offset"] == 500009
    assert classifier["shared_roi_artifact_contract"].startswith("byte_identical_roi_tensor")

    estimands = cfg["evaluation"]["estimands"]
    assert estimands["primary"]["classifier_strategy"] == "model_specific"
    assert estimands["secondary_standardized"]["classifier_strategy"] == "standardized_resnet18"
    assert estimands["within_segmenter_strategy"]["classifier_strategies"] == [
        "model_specific",
        "standardized_resnet18",
    ]
    comparison_sets = cfg["statistics"]["comparison_sets"]
    assert comparison_sets["primary_model_specific"]["planned_hypotheses"] == 30
    assert comparison_sets["secondary_standardized_resnet18"]["planned_hypotheses"] == 30
    assert comparison_sets["within_segmenter_strategy"]["planned_hypotheses"] == 20


def test_model_specific_trainability_is_locked_per_family() -> None:
    primary = load_config()["classifier"]["primary"]
    policies = primary["trainability_by_segmenter"]
    assert tuple(policies) == EXPECTED_MODELS
    assert policies["yolo26"]["trainable_components"] == [
        "yolo26_backbone",
        "yolo26_neck",
        "new_binary_head",
    ]
    assert policies["yolo26"]["frozen_components"] == [
        "native_yolo_detection_segmentation_head_forward_only"
    ]
    assert policies["vit_method2"]["trainable_components"] == [
        "patch_embedding",
        "transformer_encoder",
        "cross_attention",
        "new_binary_head",
    ]
    assert policies["emcad"]["trainable_components"] == [
        "pvt_v2_b0_full_backbone",
        "new_binary_head",
    ]
    sam = policies["sam2_unet"]
    assert sam["trainable_components"] == ["adapter_prompt_learn", "new_binary_head"]
    assert sam["frozen_components"] == [
        "original_hiera_trunk_including_patch_position_and_native_attention_blocks"
    ]
    assert sam["freeze_order"] == "freeze_original_hiera_before_wrapping_blocks_with_trainable_adapters"
    assert sam["gradient_through_frozen_trunk_to_adapters"] is True
    assert primary["fitted_segmenter_frozen_during_classifier_training"] is True
    assert primary["require_parameter_audit_against_trainability_policy"] is True


def test_composite_gate_counts_both_classifier_strategies_across_20_units() -> None:
    access = load_config()["test_access"]
    assert access["composite_model_seed_lock_count"] == 20
    assert access["classifier_strategies_per_composite_lock"] == 2
    assert access["classification_levels_per_strategy"] == 2
    assert access["expected_strategy_lock_records"] == 40
    assert access["expected_level_specific_calibration_threshold_lock_records"] == 80
    assert access["expected_core_test_system_evaluations"] == 40
    assert access["each_composite_lock_requires_both_classifier_strategies"] is True


def test_two_prelaunch_audits_are_required_before_any_seed_launch() -> None:
    preflight = load_config()["preflight"]
    assert [item["round"] for item in preflight["required_prelaunch_audits"]] == [1, 2]
    assert all(item["required_status"] == "passed" for item in preflight["required_prelaunch_audits"])
    assert preflight["receipt_must_hash_verification_artifact"] is True
    assert preflight["train_segmenters_requires_all_four_preflight_receipts"] is True


def test_undefined_validation_policies_never_silently_choose_defaults() -> None:
    cfg = load_config()
    early = cfg["training"]["early_stopping"]
    assert early["undefined_classifier_auroc_policy"] == "use_negative_validation_eye_nll_if_defined_else_stop_and_flag"
    assert "non_evaluable" in early["no_evaluable_validation_eyes_policy"]
    assert cfg["classifier"]["undefined_threshold_policy"].startswith("mark_unavailable")
    assert cfg["calibration"]["undefined_fit_policy"] == "mark_unavailable_without_default_temperature"


def test_prior_output_directories_cannot_be_reused() -> None:
    cfg = pristine()
    cfg["output"] = "binary_results_seed17"
    with pytest.raises(ConfigError, match="output"):
        validate_config(cfg)


def test_project_paths_cannot_escape_workspace() -> None:
    with pytest.raises(ConfigError, match="escapes"):
        resolve_project_path("../outside")


def test_test_gate_requires_all_twenty_lock_receipts() -> None:
    cfg = load_config()
    cfg = copy.deepcopy(cfg)
    cfg["output"] = "strict_roi_results_test_config_missing"
    with pytest.raises(ProtocolGateError, match="all 20"):
        assert_all_primary_locks(cfg)


def test_lock_receipt_enforces_monitor_and_unavailable_status_consistency() -> None:
    with tempfile.TemporaryDirectory(prefix="strict_roi_protocol_test_", dir=PROJECT_ROOT) as temporary:
        root = Path(temporary)
        cfg = load_config()
        cfg["output"] = str(root.relative_to(PROJECT_ROOT)).replace("\\", "/")
        split = split_path(cfg, 17)
        split.parent.mkdir(parents=True)
        split.write_bytes(resolve_project_path(cfg["split_policy"]["sources"]["17"]["path"]).read_bytes())
        artifact = root / "dummy_lock.json"
        artifact.write_text("{}\n", encoding="utf-8")

        with pytest.raises(ProtocolGateError, match="conflicts"):
            write_stage_receipt(
                cfg,
                "lock",
                model="yolo26",
                seed=17,
                artifacts=[artifact],
                metadata={
                    "monitor_status": "auroc",
                    "classification_threshold_status": "locked",
                    "calibration_status": "locked",
                    "roi_grid_status": "feasible",
                    "validation_evaluable_eye_count": 4,
                    "validation_evaluable_class_count": 1,
                    "status_reasons": {},
                },
            )

        with pytest.raises(ProtocolGateError, match="require unavailable"):
            write_stage_receipt(
                cfg,
                "lock",
                model="yolo26",
                seed=17,
                artifacts=[artifact],
                metadata={
                    "monitor_status": "non_evaluable",
                    "classification_threshold_status": "locked",
                    "calibration_status": "unavailable",
                    "roi_grid_status": "feasible",
                    "validation_evaluable_eye_count": 0,
                    "validation_evaluable_class_count": 0,
                    "status_reasons": {"calibration_status": "no evaluable validation eyes"},
                },
            )


def test_global_ablation_gate_requires_all_320_locks_and_freezes_their_hash() -> None:
    with tempfile.TemporaryDirectory(prefix="strict_roi_ablation_gate_", dir=PROJECT_ROOT) as temporary:
        root = Path(temporary)
        cfg = load_config()
        cfg["output"] = str(root.relative_to(PROJECT_ROOT)).replace("\\", "/")
        manifest = root / "ablation_validation_locks.json"
        manifest.write_text('{"count":320}\n', encoding="utf-8")
        digest = sha256_file(manifest)

        with pytest.raises(ProtocolGateError, match="exactly 320"):
            write_stage_receipt(
                cfg,
                "lock-ablations",
                artifacts=[manifest],
                metadata={
                    "ablation_validation_lock_count": 319,
                    "all_ablation_validation_locks_complete": True,
                    "test_predictions_generated": False,
                    "lock_manifest_sha256": digest,
                },
            )

        with pytest.raises(ProtocolGateError, match="identify one attested artifact"):
            write_stage_receipt(
                cfg,
                "lock-ablations",
                artifacts=[manifest],
                metadata={
                    "ablation_validation_lock_count": 320,
                    "all_ablation_validation_locks_complete": True,
                    "test_predictions_generated": False,
                    "lock_manifest_sha256": "0" * 64,
                },
            )

        write_stage_receipt(
            cfg,
            "lock-ablations",
            artifacts=[manifest],
            metadata={
                "ablation_validation_lock_count": 320,
                "all_ablation_validation_locks_complete": True,
                "test_predictions_generated": False,
                "lock_manifest_sha256": digest,
            },
        )
        locked = assert_all_ablation_locks(cfg)
        assert locked["lock_count"] == 320
        assert locked["lock_manifest_sha256"] == digest

        with pytest.raises(ProtocolGateError, match="exactly 320"):
            write_stage_receipt(
                cfg,
                "evaluate-ablations",
                artifacts=[manifest],
                metadata={
                    "ablation_test_evaluation_count": 319,
                    "all_ablation_test_evaluations_complete": True,
                    "separate_immutable_ablation_summary": True,
                    "primary_summary_modified": False,
                    "ablation_lock_receipt_sha256": locked["receipt_sha256"],
                },
            )
