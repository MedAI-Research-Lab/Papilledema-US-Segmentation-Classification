from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from predicted_roi_study.config import PROJECT_ROOT, load_config, sha256_file
from predicted_roi_study.data import dataframe_sha256
from predicted_roi_study import protocol
from predicted_roi_study.protocol import ProtocolGateError


STRATEGIES = ("model_specific", "standardized_resnet18")
MODEL = "yolo26"
SEED = 17


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _records(paths: list[Path]) -> list[dict]:
    return [protocol._artifact_record(path) for path in paths]


def _dual_training_fixture(root: Path) -> tuple[dict, list[Path]]:
    cfg = load_config()
    run = root / "runs" / MODEL / f"seed_{SEED}"
    train_index = run / "roi" / "train_oof_index.csv"
    validation_index = run / "roi" / "validation" / "index.csv"
    train_index.parent.mkdir(parents=True, exist_ok=True)
    validation_index.parent.mkdir(parents=True, exist_ok=True)
    raw_training = pd.DataFrame(
        {
            "patient_id": ["p0"] * 7,
            "case_id": ["e0"] * 7,
            "side": ["SAG"] * 7,
            "frame_id": [str(value) for value in range(7)],
            "label": [0] * 7,
            "roi_valid": [True] * 4 + [False] * 3,
            "cache_path": ["synthetic.npz"] * 4 + [np.nan] * 3,
        }
    )
    raw_training.to_csv(train_index, index=False, encoding="utf-8-sig")
    validation_index.write_text("frame_id\n1\n", encoding="utf-8")
    shared = run / "classifiers" / "shared"
    shared.mkdir(parents=True, exist_ok=True)
    optimization = shared / "training_optimization_index.csv"
    optimization_frame = pd.DataFrame(
        {
            "patient_id": ["p0"] * 4,
            "case_id": ["e0"] * 4,
            "side": ["SAG"] * 4,
            "frame_id": [str(value) for value in range(4)],
            "roi_valid": [True] * 4,
            "cache_path": ["synthetic.npz"] * 4,
        }
    )
    optimization_frame.to_csv(optimization, index=False, encoding="utf-8-sig")
    eligibility = shared / "training_eye_eligibility.csv"
    eligibility_frame = pd.DataFrame(
        {
            "patient_id": ["p0"],
            "case_id": ["e0"],
            "side": ["SAG"],
            "label": [0],
            "total_frames": [7],
            "valid_roi_frames": [4],
            "minimum_required_frames": [4],
            "eye_training_eligible": [True],
            "optimization_frame_count": [4],
            "exclusion_reason": ["eligible_ge_4_of_7"],
        }
    )
    eligibility_frame.to_csv(eligibility, index=False, encoding="utf-8-sig")
    selection = {
        "training_roi_index": _relative(train_index),
        "training_roi_index_sha256": sha256_file(train_index),
        "training_roi_dataframe_sha256": dataframe_sha256(raw_training),
        "training_optimization_index": _relative(optimization),
        "training_optimization_index_sha256": sha256_file(optimization),
        "training_optimization_dataframe_sha256": dataframe_sha256(
            optimization_frame
        ),
        "training_eye_eligibility": _relative(eligibility),
        "training_eye_eligibility_sha256": sha256_file(eligibility),
        "training_eye_eligibility_dataframe_sha256": dataframe_sha256(
            eligibility_frame
        ),
        "frames_per_eye": 7,
        "minimum_valid_frames": 4,
        "eligible_training_eye_count": 1,
        "ineligible_training_eye_count": 0,
        "optimization_frame_count": 4,
    }
    artifacts = [train_index, validation_index, optimization, eligibility]
    branches: dict[str, dict] = {}
    summaries: dict[str, dict] = {}
    specs = {
        cfg["classifier"]["primary"]["name"]: cfg["classifier"]["primary"],
        cfg["classifier"]["secondary"]["name"]: cfg["classifier"]["secondary"],
    }
    for strategy in STRATEGIES:
        directory = run / "classifiers" / strategy
        directory.mkdir(parents=True, exist_ok=True)
        history = directory / "history.csv"
        history.write_text("status\nnon_evaluable\n", encoding="utf-8")
        validation = directory / "validation_frames_raw.csv"
        pd.DataFrame(
            {"frame_id": ["1"], "classifier_strategy": [strategy]}
        ).to_csv(validation, index=False, encoding="utf-8-sig")
        model_info = directory / "model_info.json"
        _json(
            model_info,
            {
                "model": MODEL,
                "outer_seed": SEED,
                "classifier_strategy": strategy,
                "classifier_family": MODEL,
                "classifier_role": specs[strategy]["role"],
                "classifier_estimand_id": specs[strategy]["estimand_id"],
                "training_selection": selection,
            },
        )
        result = directory / "training_result.json"
        result_value = {
            "available": False,
            "model": MODEL,
            "seed": SEED,
            "classifier_strategy": strategy,
            "classifier_family": MODEL,
            "classifier_role": specs[strategy]["role"],
            "classifier_estimand_id": specs[strategy]["estimand_id"],
            "monitor_status": "non_evaluable",
            "checkpoint": None,
            "checkpoint_sha256": None,
            "history": _relative(history),
            "history_sha256": sha256_file(history),
            "validation_frames": _relative(validation),
            "validation_frames_sha256": sha256_file(validation),
            "model_info": _relative(model_info),
            "model_info_sha256": sha256_file(model_info),
            **selection,
            "validation_roi_index": _relative(validation_index),
            "validation_roi_index_sha256": sha256_file(validation_index),
        }
        _json(result, result_value)
        artifacts.extend([history, validation, model_info, result])
        branches[strategy] = {
            "model": MODEL,
            "seed": SEED,
            "classifier_strategy": strategy,
            "classifier_family": MODEL,
            "role": specs[strategy]["role"],
            "estimand_id": specs[strategy]["estimand_id"],
            "training_result": _relative(result),
            "training_result_sha256": sha256_file(result),
            "training_optimization_index": _relative(optimization),
            "training_optimization_index_sha256": sha256_file(optimization),
            "training_eye_eligibility": _relative(eligibility),
            "training_eye_eligibility_sha256": sha256_file(eligibility),
        }
        summaries[strategy] = {
            "model": MODEL,
            "seed": SEED,
            "classifier_strategy": strategy,
            "classifier_family": MODEL,
            "role": specs[strategy]["role"],
            "estimand_id": specs[strategy]["estimand_id"],
            "available": False,
            "monitor_status": "non_evaluable",
        }
    manifest = run / "classifiers" / "training_manifest.json"
    _json(
        manifest,
        {
            "schema": 2,
            "model": MODEL,
            "seed": SEED,
            "primary_classifier_strategy": STRATEGIES[0],
            "secondary_classifier_strategy": STRATEGIES[1],
            "identical_oof_predicted_roi_inputs": True,
            "identical_training_optimization_index": True,
            "identical_training_eye_eligibility_ledger": True,
            **selection,
            "validation_roi_index": _relative(validation_index),
            "validation_roi_index_sha256": sha256_file(validation_index),
            "strategies": branches,
        },
    )
    artifacts.append(manifest)
    metadata = {
        "primary_classifier_strategy": STRATEGIES[0],
        "classifier_strategies": list(STRATEGIES),
        "classifier_strategy_count": 2,
        "identical_oof_predicted_roi_inputs": True,
        "identical_training_optimization_index": True,
        "identical_training_eye_eligibility_ledger": True,
        "training_optimization_index_sha256": sha256_file(optimization),
        "training_eye_eligibility_sha256": sha256_file(eligibility),
        "strategies": summaries,
    }
    return metadata, artifacts


def _unavailable_level(level: str) -> dict:
    return {
        "level": level,
        "status": "unavailable",
        "classification_threshold_status": "unavailable",
        "calibration_status": "unavailable",
        "evaluable_units": 0,
        "evaluable_class_count": 0,
        "temperature": None,
        "threshold": None,
        "threshold_probability_scale": None,
    }


def test_dual_training_receipt_binds_both_strategies_model_seed_and_estimand() -> None:
    with tempfile.TemporaryDirectory(prefix="dual_training_", dir=PROJECT_ROOT) as name:
        root = Path(name)
        cfg = load_config()
        metadata, artifacts = _dual_training_fixture(root)
        protocol._validate_dual_training_receipt(
            cfg, metadata, _records(artifacts), model=MODEL, seed=SEED
        )

        missing = copy.deepcopy(metadata)
        del missing["strategies"][STRATEGIES[1]]
        with pytest.raises(ProtocolGateError, match="exactly both"):
            protocol._validate_dual_training_receipt(
                cfg, missing, _records(artifacts), model=MODEL, seed=SEED
            )

        wrong_primary = copy.deepcopy(metadata)
        wrong_primary["primary_classifier_strategy"] = STRATEGIES[1]
        with pytest.raises(ProtocolGateError, match="primary must be model_specific"):
            protocol._validate_dual_training_receipt(
                cfg, wrong_primary, _records(artifacts), model=MODEL, seed=SEED
            )

        secondary_result = (
            root / "runs" / MODEL / f"seed_{SEED}" / "classifiers"
            / STRATEGIES[1] / "training_result.json"
        )
        value = json.loads(secondary_result.read_text(encoding="utf-8"))
        value["seed"] = 42
        _json(secondary_result, value)
        with pytest.raises(ProtocolGateError, match="identity mismatch"):
            protocol._validate_dual_training_receipt(
                cfg, metadata, _records(artifacts), model=MODEL, seed=SEED
            )


def test_dual_lock_rejects_missing_secondary_wrong_primary_and_cross_seed_branch() -> None:
    with tempfile.TemporaryDirectory(prefix="dual_lock_", dir=PROJECT_ROOT) as name:
        root = Path(name)
        cfg = load_config()
        _, training_artifacts = _dual_training_fixture(root)
        run = root / "runs" / MODEL / f"seed_{SEED}"
        specs = {
            cfg["classifier"]["primary"]["name"]: cfg["classifier"]["primary"],
            cfg["classifier"]["secondary"]["name"]: cfg["classifier"]["secondary"],
        }
        branches: dict[str, dict] = {}
        artifacts = list(training_artifacts)
        for strategy in STRATEGIES:
            directory = run / "classifiers" / strategy
            training_result_value = json.loads(
                (directory / "training_result.json").read_text(encoding="utf-8")
            )
            invariance_path = run / "lock" / f"outside_roi_invariance_{strategy}.json"
            invariance = {
                "status": "classifier_unavailable",
                "passed": None,
                "classifier_strategy": strategy,
            }
            _json(invariance_path, invariance)
            artifacts.append(invariance_path)
            branch = {
                "model": MODEL,
                "seed": SEED,
                "classifier_strategy": strategy,
                "classifier_family": MODEL,
                "classifier_role": specs[strategy]["role"],
                "classifier_estimand_id": specs[strategy]["estimand_id"],
                "classifier_checkpoint": None,
                "classifier_checkpoint_sha256": None,
                "classifier_history": _relative(directory / "history.csv"),
                "classifier_history_sha256": sha256_file(directory / "history.csv"),
                "classifier_provenance": _relative(directory / "model_info.json"),
                "classifier_provenance_sha256": sha256_file(directory / "model_info.json"),
                "classifier_training_result": _relative(directory / "training_result.json"),
                "classifier_training_result_sha256": sha256_file(directory / "training_result.json"),
                "training_roi_index": training_result_value["training_roi_index"],
                "training_roi_index_sha256": training_result_value[
                    "training_roi_index_sha256"
                ],
                "training_roi_dataframe_sha256": training_result_value[
                    "training_roi_dataframe_sha256"
                ],
                "training_optimization_index": training_result_value[
                    "training_optimization_index"
                ],
                "training_optimization_index_sha256": training_result_value[
                    "training_optimization_index_sha256"
                ],
                "training_optimization_dataframe_sha256": training_result_value[
                    "training_optimization_dataframe_sha256"
                ],
                "training_eye_eligibility": training_result_value[
                    "training_eye_eligibility"
                ],
                "training_eye_eligibility_sha256": training_result_value[
                    "training_eye_eligibility_sha256"
                ],
                "training_eye_eligibility_dataframe_sha256": training_result_value[
                    "training_eye_eligibility_dataframe_sha256"
                ],
                "frames_per_eye": 7,
                "minimum_valid_frames": 4,
                "eligible_training_eye_count": 1,
                "ineligible_training_eye_count": 0,
                "optimization_frame_count": 4,
                "validation_frames": _relative(directory / "validation_frames_raw.csv"),
                "validation_frames_sha256": sha256_file(directory / "validation_frames_raw.csv"),
                "outside_roi_invariance": invariance,
                "outside_roi_invariance_path": _relative(invariance_path),
                "outside_roi_invariance_sha256": sha256_file(invariance_path),
                "eye": _unavailable_level("eye"),
                "patient": _unavailable_level("patient"),
                "monitor_status": "non_evaluable",
                "classification_threshold_status": "unavailable",
                "calibration_status": "unavailable",
                "operational_system_evaluable": False,
                "calibrated_system_evaluable": False,
                "validation_evaluable_eye_count": 0,
                "validation_evaluable_class_count": 0,
            }
            branches[strategy] = branch
        primary = branches[STRATEGIES[0]]
        lock_path = run / "lock" / "primary_lock.json"
        lock = {
            "model": MODEL,
            "seed": SEED,
            "primary_classifier_strategy": STRATEGIES[0],
            "secondary_classifier_strategy": STRATEGIES[1],
            "classifier_strategies": branches,
            "classifier_checkpoint": None,
            "classifier_checkpoint_sha256": None,
            "eye": primary["eye"],
            "patient": primary["patient"],
            "monitor_status": "non_evaluable",
            "primary_system_evaluable": False,
        }
        _json(lock_path, lock)
        artifacts.append(lock_path)
        metadata = {
            "primary_classifier_strategy": STRATEGIES[0],
            "secondary_classifier_strategy": STRATEGIES[1],
            "classifier_strategy_count": 2,
            "classifier_strategies": {
                strategy: {
                    "role": specs[strategy]["role"],
                    "estimand_id": specs[strategy]["estimand_id"],
                    "monitor_status": "non_evaluable",
                    "classification_threshold_status": "unavailable",
                    "calibration_status": "unavailable",
                    "operational_system_evaluable": False,
                    "validation_evaluable_eye_count": 0,
                    "validation_evaluable_class_count": 0,
                }
                for strategy in STRATEGIES
            },
        }
        protocol._validate_dual_lock_artifact(
            cfg, metadata, _records(artifacts), model=MODEL, seed=SEED
        )

        wrong_primary = copy.deepcopy(metadata)
        wrong_primary["primary_classifier_strategy"] = STRATEGIES[1]
        with pytest.raises(ProtocolGateError, match="primary must be model_specific"):
            protocol._validate_dual_lock_artifact(
                cfg, wrong_primary, _records(artifacts), model=MODEL, seed=SEED
            )

        missing_secondary = copy.deepcopy(lock)
        del missing_secondary["classifier_strategies"][STRATEGIES[1]]
        _json(lock_path, missing_secondary)
        with pytest.raises(ProtocolGateError, match="exact dual strategy set"):
            protocol._validate_dual_lock_artifact(
                cfg, metadata, _records(artifacts), model=MODEL, seed=SEED
            )

        cross_seed = copy.deepcopy(lock)
        cross_seed["classifier_strategies"][STRATEGIES[1]]["seed"] = 42
        _json(lock_path, cross_seed)
        with pytest.raises(ProtocolGateError, match="model/seed scope mismatch"):
            protocol._validate_dual_lock_artifact(
                cfg, metadata, _records(artifacts), model=MODEL, seed=SEED
            )


def test_raw_probability_fallback_is_locked_without_fake_temperature() -> None:
    raw = {
        "level": "eye",
        "status": "calibration_unavailable",
        "classification_threshold_status": "locked",
        "calibration_status": "unavailable",
        "evaluable_units": 8,
        "evaluable_class_count": 2,
        "temperature": None,
        "threshold": 0.45,
        "threshold_probability_scale": "raw_due_to_calibration_unavailable",
    }
    protocol._validate_branch_level_lock(raw, "raw fallback", level="eye")
    substituted = dict(raw, temperature=1.0)
    with pytest.raises(ProtocolGateError, match="must not substitute T=1"):
        protocol._validate_branch_level_lock(
            substituted, "raw fallback", level="eye"
        )


def test_preflight_semantics_require_two_real_audit_rounds_for_both_strategies() -> None:
    with tempfile.TemporaryDirectory(prefix="dual_preflight_", dir=PROJECT_ROOT) as name:
        root = Path(name)
        audits = {
            strategy: {
                "classifier_strategy": strategy,
                "finite_forward": True,
                "finite_backward": True,
                "classifier_logits_shape": [1, 2],
                "strict_hard_mask_supplied": True,
                "outside_roi_invariance_passed": True,
                "outside_roi_invariance_maximum_logit_difference": 0.0,
                "trainability_policy_audit": {
                    "status": "passed" if strategy == STRATEGIES[0] else "not_applicable"
                },
            }
            for strategy in STRATEGIES
        }
        rounds = [
            {
                "round": 1,
                "name": "dual_classifier_contract_and_backward",
                "status": "passed",
                "executed_scope": f"{MODEL}_both_strategies",
                "aggregate_required_scope": "all_four_models_both_strategies",
                "strategies": {
                    strategy: {
                        "finite_forward": True,
                        "finite_backward": True,
                        "classifier_logits_shape": [1, 2],
                        "strict_hard_mask_supplied": True,
                    }
                    for strategy in STRATEGIES
                },
            },
            {
                "round": 2,
                "name": "synthetic_strict_roi_end_to_end_and_outside_roi_invariance",
                "status": "passed",
                "executed_scope": f"{MODEL}_both_strategies",
                "aggregate_required_scope": "all_four_models_both_strategies",
                "strict_roi_tensor_shared_by_both_strategies": True,
                "classifier_training_eligibility_contract_passed": True,
                "strategies": {
                    strategy: {
                        "outside_roi_invariance_passed": True,
                        "maximum_absolute_logit_difference": 0.0,
                    }
                    for strategy in STRATEGIES
                },
            },
        ]
        report = {
            "model": MODEL,
            "passed": True,
            "primary_classifier_strategy": STRATEGIES[0],
            "classifier_strategies": list(STRATEGIES),
            "classifier_audits": audits,
            "classifier_training_eligibility_audit": {
                "status": "passed",
                "canonical_roi_index_immutable": True,
                "three_of_seven_eye_excluded": True,
                "four_of_seven_eye_included": True,
                "seven_of_seven_eye_included": True,
                "valid_but_ineligible_cache_rows_never_relabelled": True,
                "valid_but_ineligible_cache_rows_never_enter_dataset": True,
                "frames_per_eye": 7,
                "minimum_valid_frames": 4,
                "optimization_frame_count": 11,
                "optimization_index_sha256": "1" * 64,
                "eligibility_ledger_sha256": "2" * 64,
                "identical_selection_for_both_classifier_strategies": True,
            },
            "required_prelaunch_audits": rounds,
            "outside_roi_invariance_tolerance": 1e-6,
        }
        path = root / "verification" / f"{MODEL}.json"
        _json(path, report)
        protocol._validate_preflight_report(load_config(), _records([path]), model=MODEL)

        report["required_prelaunch_audits"][1]["strategies"][STRATEGIES[1]][
            "outside_roi_invariance_passed"
        ] = False
        _json(path, report)
        with pytest.raises(ProtocolGateError, match="round summary disagrees"):
            protocol._validate_preflight_report(load_config(), _records([path]), model=MODEL)
