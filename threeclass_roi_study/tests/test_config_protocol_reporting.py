from __future__ import annotations

import copy
from pathlib import Path

import pytest

from threeclass_roi_study.config import (
    CLASSIFIER_STRATEGIES,
    CLASS_NAMES,
    CLASS_ORDER,
    ConfigError,
    EXPECTED_MODELS,
    EXPECTED_SEEDS,
    canonical_sha256,
    config_without_runtime,
    load_config,
    sha256_file,
    upstream_artifact_paths,
    validate_config,
    verify_frozen_source_anchors,
    verify_split_sources,
)
from threeclass_roi_study import protocol
from threeclass_roi_study.protocol import ProtocolGateError
from threeclass_roi_study.reporting import (
    OUTPUT_TABLE_SCHEMAS,
    ReportingSchemaError,
    build_q1_metadata,
    publication_output_schemas,
    validate_output_rows,
    validate_q1_metadata,
)


def test_default_config_locks_new_namespace_and_three_class_contract() -> None:
    cfg = load_config()
    assert cfg["schema_version"] == 1
    assert cfg["protocol_version"] == "1.0.0"
    assert cfg["task"] == "three_class_classification"
    assert cfg["analysis_status"] == "exploratory_post_hoc_internal"
    assert cfg["output"] == "threeclass_roi_results_4model_v1_0_0"
    assert cfg["output"] != cfg["frozen_upstream"]["source_output"]
    assert tuple(cfg["models"]) == EXPECTED_MODELS
    assert tuple(cfg["split_seeds"]) == EXPECTED_SEEDS
    assert tuple(cfg["classes"]["order"]) == CLASS_ORDER
    assert cfg["classes"]["names"] == {
        str(key): value for key, value in CLASS_NAMES.items()
    }
    assert tuple(cfg["classifier"]["strategy_order"]) == CLASSIFIER_STRATEGIES
    assert cfg["classifier"]["classes"] == 3
    assert cfg["classifier"]["reuse_binary_head"] is False
    assert cfg["classifier"]["reuse_binary_threshold_or_calibrator"] is False
    assert cfg["classifier"]["maximum_epochs"] == 60
    assert cfg["classifier"]["patience"] == 10


def test_frozen_source_anchors_and_split_hashes_match_disk() -> None:
    cfg = load_config()
    anchors = verify_frozen_source_anchors(cfg)
    splits = verify_split_sources(cfg)
    assert set(anchors) == {
        "prepare_receipt",
        "source_config_snapshot",
        "source_protocol_snapshot",
        "source_test_access_sentinel",
    }
    assert tuple(splits) == EXPECTED_SEEDS
    assert all(len(value) == 64 for value in [*anchors.values(), *splits.values()])


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("analysis_status", "confirmatory_external"),
        ("frozen_upstream.allow_binary_classifier_artifacts", True),
        ("frozen_upstream.allow_retraining_or_retuning_upstream", True),
        ("frozen_upstream.test_artifacts_deferred_until_global_lock", False),
        ("classifier.reuse_binary_head", True),
        ("classifier.classes", 2),
        ("classifier.test_used_for_training_selection_or_stopping", True),
        ("aggregation.minimum_valid_frames_per_eye", 1),
        ("aggregation.require_both_eyes", False),
        ("calibration.test_refitting", True),
        ("abstention.test_threshold_tuning", True),
        ("prior_test_use.confirmatory_claim_allowed", True),
        ("test_access.require_all_locks_before_any_test_artifact_read", False),
    ],
)
def test_scientifically_unsafe_config_changes_fail_closed(
    path: str, value: object
) -> None:
    cfg = config_without_runtime(load_config())
    cursor = cfg
    parts = path.split(".")
    for part in parts[:-1]:
        cursor = cursor[part]
    cursor[parts[-1]] = value
    with pytest.raises(ConfigError):
        validate_config(cfg)


def test_upstream_path_roles_separate_development_from_test() -> None:
    cfg = load_config()
    development = upstream_artifact_paths(cfg, "sam2_unet", 17, phase="development")
    deferred = upstream_artifact_paths(cfg, "sam2_unet", 17, phase="test")
    assert tuple(development) == tuple(cfg["frozen_upstream"]["development_import_roles"])
    assert tuple(deferred) == tuple(cfg["frozen_upstream"]["deferred_test_import_roles"])
    assert not set(development) & set(deferred)
    assert all("test_rois" not in path.as_posix() for path in development.values())
    assert any("test_rois" in path.as_posix() for path in deferred.values())


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _write_prepare_code_freeze(cfg: dict) -> None:
    package = Path(protocol.__file__).resolve().parent
    inventory = {
        path.name: sha256_file(path)
        for path in sorted(package.glob("*.py"))
    }
    root = Path(cfg["output"])
    inventory_path = protocol.save_json_atomic(
        root / "provenance" / "code_inventory.json", inventory
    )
    receipt = {
        "schema_version": protocol.RECEIPT_SCHEMA_VERSION,
        "stage": "prepare",
        "study_id": cfg["study_id"],
        "config_sha256": canonical_sha256(config_without_runtime(cfg)),
        "code_sha256": canonical_sha256(list(inventory.items())),
        "model": None,
        "seed": None,
        "classifier_strategy": None,
        "artifacts": [
            {
                "role": "code_inventory",
                "path": str(inventory_path.resolve()),
                "sha256": sha256_file(inventory_path),
                "size_bytes": inventory_path.stat().st_size,
            }
        ],
        "metadata": {"test_data_read": False},
        "completed_utc": protocol.utc_now(),
    }
    protocol.save_json_atomic(root / "state" / "receipts" / "prepare.json", receipt)


def _mini_protocol_fixture(tmp_path: Path) -> tuple[dict, str, int]:
    cfg = copy.deepcopy(load_config())
    model = "yolo26"
    seed = 17
    cfg["output"] = str(tmp_path / "threeclass-output")
    cfg["models"] = [model]
    cfg["split_seeds"] = [seed]
    cfg["frozen_upstream"]["required_development_import_receipts"] = 1
    cfg["frozen_upstream"]["required_deferred_test_import_receipts"] = 1
    cfg["test_access"]["required_development_import_receipts"] = 1
    cfg["test_access"]["required_validation_locks"] = 2
    cfg["test_access"]["validation_locks_per_model_seed"] = 2

    source = tmp_path / "source"
    development_patterns = {
        "source_build_roi_receipt": str(
            source / "{model}" / "seed_{seed}" / "build-rois.json"
        ),
        "source_segmenter_lock": str(
            source / "{model}" / "seed_{seed}" / "segmenter_lock.json"
        ),
        "source_segmenter_checkpoint": str(
            source / "{model}" / "seed_{seed}" / "segmenter.pt"
        ),
        "train_oof_roi_index": str(
            source / "{model}" / "seed_{seed}" / "train_oof.csv"
        ),
        "validation_roi_index": str(
            source / "{model}" / "seed_{seed}" / "validation.csv"
        ),
    }
    test_patterns = {
        "source_evaluate_receipt": str(
            source / "{model}" / "seed_{seed}" / "evaluate.json"
        ),
        "test_roi_index": str(source / "{model}" / "seed_{seed}" / "test.csv"),
    }
    cfg["frozen_upstream"]["development_artifact_patterns"] = development_patterns
    cfg["frozen_upstream"]["deferred_test_artifact_patterns"] = test_patterns
    for role, pattern in development_patterns.items():
        _write(Path(pattern.format(model=model, seed=seed)), f"{role}\n")
    for role, pattern in test_patterns.items():
        _write(Path(pattern.format(model=model, seed=seed)), f"{role}\n")

    source_sentinel = _write(tmp_path / "source-test-access.json", "{}\n")
    cfg["prior_test_use"]["test_open_sentinel"] = str(source_sentinel)
    cfg["frozen_upstream"]["source_anchors"]["source_test_access_sentinel"] = {
        "path": str(source_sentinel),
        "sha256": sha256_file(source_sentinel),
    }
    cfg.pop("config_sha256", None)
    cfg.pop("_runtime", None)
    _write_prepare_code_freeze(cfg)
    return cfg, model, seed


def _lock_metadata(strategy: str) -> dict:
    return {
        "classifier_strategy": strategy,
        "classifier_outputs": 3,
        "class_order": [0, 1, 2],
        "training_roi_source": "frozen_out_of_fold_predicted_roi",
        "selection_partition": "validation",
        "primary_prediction_level": "patient",
        "primary_endpoint": "three_class_failure_aware_balanced_accuracy",
        "decision_rule": "argmax",
        "primary_abstention_policy": "structural_roi_gate_only",
        "test_data_read": False,
        "test_selection_or_refitting": False,
        "binary_head_threshold_or_calibrator_reused": False,
        "monitor_status": "validation_patient_macro_nll",
        "calibration": {
            "eye": {"status": "available", "temperature": 1.2, "reason": None},
            "patient": {"status": "available", "temperature": 1.1, "reason": None},
        },
    }


def _write_all_validation_locks(
    tmp_path: Path, cfg: dict, model: str, seed: int
) -> None:
    for strategy in CLASSIFIER_STRATEGIES:
        checkpoint = _write(tmp_path / f"{strategy}.pt", f"{strategy}\n")
        protocol.write_validation_lock(
            cfg,
            model,
            seed,
            strategy,
            artifacts={"classifier_checkpoint": checkpoint},
            metadata=_lock_metadata(strategy),
        )


def test_test_artifact_import_is_impossible_before_all_validation_locks(
    tmp_path: Path,
) -> None:
    cfg, model, seed = _mini_protocol_fixture(tmp_path)
    protocol.write_upstream_import_receipt(cfg, model, seed)
    with pytest.raises(ProtocolGateError, match="Test access is closed"):
        protocol.write_deferred_test_import_receipt(cfg, model, seed)
    with pytest.raises(ProtocolGateError, match="Missing validation lock"):
        protocol.open_test_access(cfg)


def test_global_gate_records_prior_use_then_allows_deferred_test_import(
    tmp_path: Path,
) -> None:
    cfg, model, seed = _mini_protocol_fixture(tmp_path)
    receipt = protocol.write_upstream_import_receipt(cfg, model, seed)
    assert receipt.is_file()
    _write_all_validation_locks(tmp_path, cfg, model, seed)
    sentinel_path = protocol.open_test_access(cfg)
    sentinel = protocol.assert_test_access_open(cfg)
    root = Path(cfg["output"])
    assert sentinel_path.is_file()
    assert sentinel["development_import_receipt_count"] == 1
    assert sentinel["validation_lock_count"] == 2
    assert sentinel["code_inventory_sha256"] == sha256_file(
        root / "provenance" / "code_inventory.json"
    )
    assert sentinel["prepare_receipt_sha256"] == sha256_file(
        root / "state" / "receipts" / "prepare.json"
    )
    assert sentinel["code_sha256"] == protocol.package_code_sha256()
    assert sentinel["prior_test_use"]["confirmatory_claim_allowed"] is False
    assert "exploratory" in sentinel["prior_test_use"]["required_disclosure"].lower()
    deferred = protocol.write_deferred_test_import_receipt(cfg, model, seed)
    assert deferred.is_file()
    verified = protocol.verify_deferred_test_import_receipt(cfg, model, seed)
    assert verified["test_selection_or_refitting"] is False


@pytest.mark.parametrize(
    ("relative_path", "message"),
    [
        ("provenance/code_inventory.json", "prepare code inventory"),
        ("state/receipts/prepare.json", "prepare receipt"),
    ],
)
def test_global_gate_requires_prepare_code_freeze_artifacts(
    tmp_path: Path, relative_path: str, message: str
) -> None:
    cfg, model, seed = _mini_protocol_fixture(tmp_path)
    protocol.write_upstream_import_receipt(cfg, model, seed)
    _write_all_validation_locks(tmp_path, cfg, model, seed)
    (Path(cfg["output"]) / relative_path).unlink()

    with pytest.raises(ProtocolGateError, match=message):
        protocol.open_test_access(cfg)


@pytest.mark.parametrize(
    "relative_path",
    ["provenance/code_inventory.json", "state/receipts/prepare.json"],
)
def test_open_gate_rejects_prepare_code_freeze_artifact_mutation(
    tmp_path: Path, relative_path: str
) -> None:
    cfg, model, seed = _mini_protocol_fixture(tmp_path)
    protocol.write_upstream_import_receipt(cfg, model, seed)
    _write_all_validation_locks(tmp_path, cfg, model, seed)
    protocol.open_test_access(cfg)
    path = Path(cfg["output"]) / relative_path
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ProtocolGateError, match="hash mismatch"):
        protocol.assert_test_access_open(cfg)


def test_open_gate_rechecks_current_package_code_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, model, seed = _mini_protocol_fixture(tmp_path)
    protocol.write_upstream_import_receipt(cfg, model, seed)
    _write_all_validation_locks(tmp_path, cfg, model, seed)
    protocol.open_test_access(cfg)
    monkeypatch.setattr(protocol, "package_code_sha256", lambda: "0" * 64)

    with pytest.raises(ProtocolGateError, match="package code hash mismatch"):
        protocol.assert_test_access_open(cfg)


@pytest.mark.parametrize("level", ["eye", "patient"])
def test_validation_lock_requires_both_calibrations_available(level: str) -> None:
    cfg = load_config()
    metadata = _lock_metadata("model_specific")
    metadata["calibration"][level] = {
        "status": "unavailable",
        "temperature": None,
        "reason": "missing validation class",
    }

    with pytest.raises(ProtocolGateError, match=f"available {level} calibration"):
        protocol.validate_validation_lock_metadata(cfg, "model_specific", metadata)


@pytest.mark.parametrize("temperature", [float("nan"), float("inf"), float("-inf"), 0.0])
@pytest.mark.parametrize("level", ["eye", "patient"])
def test_validation_lock_requires_finite_positive_calibration_temperature(
    level: str, temperature: float
) -> None:
    cfg = load_config()
    metadata = _lock_metadata("model_specific")
    metadata["calibration"][level]["temperature"] = temperature

    with pytest.raises(ProtocolGateError, match="finite positive temperature"):
        protocol.validate_validation_lock_metadata(cfg, "model_specific", metadata)


def test_upstream_receipt_detects_artifact_mutation(tmp_path: Path) -> None:
    cfg, model, seed = _mini_protocol_fixture(tmp_path)
    protocol.write_upstream_import_receipt(cfg, model, seed)
    path = upstream_artifact_paths(cfg, model, seed)[
        "validation_roi_index"
    ]
    path.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ProtocolGateError, match="hash mismatch"):
        protocol.verify_upstream_import_receipt(cfg, model, seed)


def test_q1_metadata_has_claim_tripod_prior_use_and_primary_endpoint() -> None:
    cfg = load_config()
    metadata = build_q1_metadata(cfg)
    validate_q1_metadata(metadata)
    assert metadata["reporting_guidelines"]["claim"] == "CLAIM_2024_Update"
    assert metadata["reporting_guidelines"]["tripod"] == "TRIPOD_plus_AI_2024"
    assert metadata["evaluation"]["primary_endpoint"] == (
        "three_class_failure_aware_balanced_accuracy"
    )
    assert metadata["prior_test_use"]["confirmatory_claim_allowed"] is False
    assert metadata["external_evaluation"]["status"] == (
        "not_performed_in_this_internal_extension"
    )
    localized = metadata["evaluation"]["localized_diagnostic_success_definition"]
    assert localized["frame_hit_source"] == "immutable_source_evaluation_roi_hit"
    assert localized["frame_hit_iou_threshold"] == 0.5
    assert localized["minimum_hit_frames_per_eye"] == 4
    assert localized["require_both_eyes"] is True
    assert localized["gt_used_for_inference_selection_or_abstention"] is False


def test_q1_metadata_rejects_suppressed_prior_use_disclosure() -> None:
    metadata = build_q1_metadata(load_config())
    metadata["prior_test_use"]["required_disclosure"] = "Independent validation."
    with pytest.raises(ReportingSchemaError, match="prior-test-use"):
        validate_q1_metadata(metadata)


def _valid_patient_prediction() -> dict:
    return {
        "study_id": "strict_predicted_roi_threeclass_4model_v1_0_0",
        "model": "sam2_unet",
        "seed": 17,
        "classifier_strategy": "model_specific",
        "patient_id": "P_TEST",
        "true_label": 1,
        "p_normal_raw": 0.1,
        "p_papilledema_raw": 0.7,
        "p_pseudopapilledema_raw": 0.2,
        "p_normal_calibrated": 0.15,
        "p_papilledema_calibrated": 0.65,
        "p_pseudopapilledema_calibrated": 0.2,
        "predicted_label": 1,
        "abstained": False,
        "abstention_reason": None,
        "right_eye_evaluable": True,
        "left_eye_evaluable": True,
        "right_valid_frames": 7,
        "left_valid_frames": 7,
        "calibration_status": "available",
        "localized": True,
        "localized_diagnostic_success": True,
        "prior_test_use_disclosed": True,
    }


def test_patient_prediction_schema_enforces_probability_and_disclosure() -> None:
    row = _valid_patient_prediction()
    assert validate_output_rows("patient_predictions.csv", [row]) == [row]

    bad = dict(row)
    bad["p_papilledema_raw"] = 0.8
    with pytest.raises(ReportingSchemaError, match="do not sum to one"):
        validate_output_rows("patient_predictions.csv", [bad])

    bad = dict(row)
    bad["prior_test_use_disclosed"] = False
    with pytest.raises(ReportingSchemaError, match="mandatory prior-use"):
        validate_output_rows("patient_predictions.csv", [bad])


def test_primary_confusion_schema_requires_all_twelve_cells() -> None:
    rows = [
        {
            "model": "sam2_unet",
            "seed": 17,
            "classifier_strategy": "model_specific",
            "true_label": true_label,
            "predicted_label": predicted_label,
            "count": 0,
        }
        for true_label in (0, 1, 2)
        for predicted_label in (0, 1, 2, 3)
    ]
    assert len(validate_output_rows("patient_confusion_3x4.csv", rows)) == 12
    with pytest.raises(ReportingSchemaError, match="full grid"):
        validate_output_rows("patient_confusion_3x4.csv", rows[:-1])


def test_publication_schemas_are_defensive_copies_and_cover_required_outputs() -> None:
    schemas = publication_output_schemas()
    assert set(schemas) == {
        "patient_predictions.csv",
        "patient_metrics.csv",
        "patient_confusion_3x4.csv",
        "classwise_metrics.csv",
        "calibration_metrics.csv",
        "risk_coverage.csv",
        "uncertainty_metrics.csv",
        "segmentation_by_class.csv",
        "segmentation_uncertainty.csv",
        "provenance_artifacts.csv",
    }
    schemas["patient_metrics.csv"]["grain"] = "changed"
    assert OUTPUT_TABLE_SCHEMAS["patient_metrics.csv"]["grain"] != "changed"
