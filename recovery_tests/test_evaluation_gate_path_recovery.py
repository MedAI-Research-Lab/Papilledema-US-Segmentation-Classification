"""Regression and guardrail tests for the evaluation-only gate-path recovery.

This test module deliberately lives outside ``predicted_roi_study``.  The
locked study fingerprints every Python file below that package, including its
tests, so placing this regression there would invalidate the already completed
20 model/seed lock chains that the recovery is intended to preserve.
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECOVERY_SCRIPT = PROJECT_ROOT / "scripts" / "recover_strict_roi_evaluation_v1.py"
CONFIG_PATH = PROJECT_ROOT / "predicted_roi_study" / "config_strict_roi.json"
OUTPUT_ROOT = PROJECT_ROOT / "strict_roi_results_4model_v1_2_0"
CANONICAL_GATE = OUTPUT_ROOT / "state" / "test_access_opened.json"
PARTIAL_RUN = OUTPUT_ROOT / "runs" / "yolo26" / "seed_17"
EXPECTED_FROZEN_CODE_SHA256 = (
    "914ec1505a0d417b8b6766b3663160673b94d5ce107da68f81d3142bbf493776"
)


def _load_recovery_module() -> ModuleType:
    assert RECOVERY_SCRIPT.is_file(), f"Recovery runner is missing: {RECOVERY_SCRIPT}"
    spec = importlib.util.spec_from_file_location(
        "strict_roi_evaluation_gate_path_recovery", RECOVERY_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _partial_evaluation_artifacts(alias: Path) -> tuple[list[Path], dict]:
    """Recreate the artifact inventory built by ``engine.evaluate``.

    The failed run reached ``attempt.status=complete`` and failed only when the
    receipt validator searched for the obsolete gate path.  Reconstructing the
    inventory lets this regression exercise the real validator without reading
    test rasters or rerunning inference.
    """

    from predicted_roi_study.config import resolve_project_path

    evaluation = PARTIAL_RUN / "evaluation"
    attempt_path = evaluation / "attempt.json"
    lock_path = PARTIAL_RUN / "lock" / "primary_lock.json"
    attempt = _read_json(attempt_path)
    lock = _read_json(lock_path)
    artifacts: list[Path] = [
        attempt_path,
        evaluation / "test_rois" / "index.csv",
        lock_path,
        alias,
    ]
    report_names = (
        "frames.csv",
        "eyes.csv",
        "patients.csv",
        "metrics.json",
        "calibration_curves.csv",
        "calibration_reliability_bands.csv",
        "decision_curves.csv",
        "risk_coverage_curves.csv",
        "bootstrap_patient_cluster_ci.csv",
        "segmentation_patient_cluster_ci.csv",
        "confusion_2x3.csv",
        "confusion_conditional_2x2.csv",
        "abstention_reasons.csv",
        "classification_by_label_3class.csv",
    )
    for strategy in ("model_specific", "standardized_resnet18"):
        branch = lock["classifier_strategies"][strategy]
        destination = evaluation / strategy
        artifacts.extend(destination / name for name in report_names)
        artifacts.extend(
            sorted(destination.glob("classification_calibration_decision_curves.*"))
        )
        artifacts.extend(
            sorted(destination.glob("calibration_density_and_risk_coverage.*"))
        )
        if branch.get("classifier_checkpoint"):
            artifacts.append(
                resolve_project_path(branch["classifier_checkpoint"], must_exist=True)
            )
        for key in (
            "classifier_history",
            "classifier_provenance",
            "classifier_training_result",
        ):
            artifacts.append(resolve_project_path(branch[key], must_exist=True))
    artifacts = list(dict.fromkeys(path.resolve() for path in artifacts))
    missing = [str(path) for path in artifacts if not path.is_file()]
    assert not missing, f"The partial evaluation artifact set is incomplete: {missing}"
    metadata = {
        "test_frames": attempt["test_frames"],
        "primary_classifier_strategy": "model_specific",
        "classifier_strategies": attempt["classifier_strategies"],
        "primary_system_evaluable": bool(lock["primary_system_evaluable"]),
        "calibrated_system_evaluable": bool(
            lock.get("calibrated_system_evaluable", False)
        ),
        "test_selection_or_refitting": False,
    }
    return artifacts, metadata


def test_alias_creation_is_byte_identical_idempotent_and_fail_closed(
    tmp_path: Path,
) -> None:
    from predicted_roi_study.protocol import ProtocolGateError

    recovery = _load_recovery_module()
    helper = recovery.create_or_verify_byte_identical_alias
    source = tmp_path / "state" / "test_access_opened.json"
    alias = tmp_path / "amendment" / "protocol" / "test_access.json"
    source.parent.mkdir(parents=True)
    source.write_bytes(b'{\n  "schema": 1,\n  "opened_utc": "frozen"\n}\n')

    first = helper(source, alias)
    assert Path(first).resolve() == alias.resolve()
    assert alias.read_bytes() == source.read_bytes()
    first_stat = alias.stat()

    second = helper(source, alias)
    assert Path(second).resolve() == alias.resolve()
    assert alias.read_bytes() == source.read_bytes()
    second_stat = alias.stat()
    assert second_stat.st_ino == first_stat.st_ino
    assert second_stat.st_mtime_ns == first_stat.st_mtime_ns

    conflicting = tmp_path / "conflicting" / "protocol" / "test_access.json"
    conflicting.parent.mkdir(parents=True)
    conflicting.write_bytes(b"do-not-overwrite\n")
    before = conflicting.read_bytes()
    with pytest.raises(
        ProtocolGateError, match="(?i)different|mismatch|refus|overwrite"
    ):
        helper(source, conflicting)
    assert conflicting.read_bytes() == before


def test_real_gate_and_partial_attempt_remain_bound_to_frozen_core_hash() -> None:
    from predicted_roi_study.config import load_config, sha256_file
    from predicted_roi_study.protocol import code_fingerprint, protocol_sha256

    cfg = load_config(CONFIG_PATH)
    gate = _read_json(CANONICAL_GATE)
    attempt = _read_json(PARTIAL_RUN / "evaluation" / "attempt.json")
    provenance = _read_json(OUTPUT_ROOT / "provenance" / "code_fingerprint.json")
    current_code = code_fingerprint()

    assert current_code["sha256"] == EXPECTED_FROZEN_CODE_SHA256
    assert provenance["sha256"] == EXPECTED_FROZEN_CODE_SHA256
    assert gate["code_sha256"] == EXPECTED_FROZEN_CODE_SHA256
    assert attempt["code_sha256"] == EXPECTED_FROZEN_CODE_SHA256
    assert gate["config_sha256"] == cfg["_runtime"]["config_sha256"]
    assert gate["protocol_sha256"] == protocol_sha256()
    assert attempt["config_sha256"] == cfg["_runtime"]["config_sha256"]
    assert attempt["test_access_sha256"] == sha256_file(CANONICAL_GATE)
    assert (
        attempt["primary_lock_sha256"]
        == sha256_file(PARTIAL_RUN / "lock" / "primary_lock.json")
    )
    assert attempt["status"] == "complete"
    assert attempt["test_frames"] == 252
    assert attempt["segmentation_valid_frames"] == 238
    assert not any(
        path == "scripts/recover_strict_roi_evaluation_v1.py"
        for path in current_code["files"]
    )


def test_real_partial_yolo26_seed17_receipt_semantics_validate_through_alias() -> None:
    from predicted_roi_study import protocol
    from predicted_roi_study.config import PROJECT_ROOT, load_config

    recovery = _load_recovery_module()
    cfg = load_config(CONFIG_PATH)
    with tempfile.TemporaryDirectory(
        prefix="evaluation_gate_path_recovery_", dir=PROJECT_ROOT
    ) as temporary_name:
        alias = Path(temporary_name) / "protocol" / "test_access.json"
        recovery.create_or_verify_byte_identical_alias(CANONICAL_GATE, alias)
        artifacts, metadata = _partial_evaluation_artifacts(alias)
        records = [protocol._artifact_record(path) for path in artifacts]

        protocol._validate_dual_evaluation_receipt(
            cfg,
            metadata,
            records,
            model="yolo26",
            seed=17,
        )

        canonical_records = [
            protocol._artifact_record(CANONICAL_GATE)
            if Path(record["path"]).as_posix().endswith("/protocol/test_access.json")
            else record
            for record in records
        ]
        with pytest.raises(
            protocol.ProtocolGateError,
            match="not bound to the global test-access gate",
        ):
            protocol._validate_dual_evaluation_receipt(
                cfg,
                metadata,
                canonical_records,
                model="yolo26",
                seed=17,
            )


def test_frozen_inventory_requires_exact_paths_sizes_and_hashes() -> None:
    recovery = _load_recovery_module()
    with tempfile.TemporaryDirectory(
        prefix="evaluation_inventory_guard_", dir=PROJECT_ROOT
    ) as temporary_name:
        temporary = Path(temporary_name)
        evaluation = temporary / "evaluation"
        nested = evaluation / "model_specific"
        nested.mkdir(parents=True)
        first = evaluation / "attempt.json"
        second = nested / "metrics.json"
        first.write_text('{"status":"complete"}\n', encoding="utf-8")
        second.write_text('{"metric":1}\n', encoding="utf-8")

        from predicted_roi_study.config import sha256_file

        files = []
        for path in sorted(item for item in evaluation.rglob("*") if item.is_file()):
            files.append(
                {
                    "path": path.relative_to(PROJECT_ROOT).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        inventory = temporary / "inventory.json"
        inventory.write_text(
            json.dumps(
                {
                    "schema": 1,
                    "model": "yolo26",
                    "seed": 17,
                    "file_count": len(files),
                    "files": files,
                }
            ),
            encoding="utf-8",
        )

        verified = recovery.verify_frozen_file_inventory(
            inventory, evaluation, model="yolo26", seed=17
        )
        assert verified["file_count"] == 2

        second.write_text('{"metric":2}\n', encoding="utf-8")
        with pytest.raises(recovery.RecoveryError, match="differs from the frozen"):
            recovery.verify_frozen_file_inventory(
                inventory, evaluation, model="yolo26", seed=17
            )

        second.write_text('{"metric":1}\n', encoding="utf-8")
        extra = evaluation / "unexpected.txt"
        extra.write_text("unexpected\n", encoding="utf-8")
        with pytest.raises(recovery.RecoveryError, match="differs from the frozen"):
            recovery.verify_frozen_file_inventory(
                inventory, evaluation, model="yolo26", seed=17
            )

        extra.unlink()
        first.unlink()
        with pytest.raises(recovery.RecoveryError, match="differs from the frozen"):
            recovery.verify_frozen_file_inventory(
                inventory, evaluation, model="yolo26", seed=17
            )


def test_backfill_writes_only_receipt_and_never_calls_inference(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from predicted_roi_study import engine
    from predicted_roi_study.config import load_config

    recovery = _load_recovery_module()
    cfg = load_config(CONFIG_PATH)
    alias = tmp_path / "protocol" / "test_access.json"
    recovery.create_or_verify_byte_identical_alias(CANONICAL_GATE, alias)
    inventory = (
        OUTPUT_ROOT
        / "state"
        / "amendments"
        / "evaluation_gate_path_compatibility_v1"
        / "pre_recovery_yolo26_seed17_inventory.json"
    )
    receipt = tmp_path / "evaluate.json"
    events: list[str] = []
    writer_calls: list[dict] = []
    forbidden_calls: list[str] = []
    real_inventory_verifier = recovery.verify_frozen_file_inventory

    def traced_inventory(*args, **kwargs):
        events.append("inventory")
        return real_inventory_verifier(*args, **kwargs)

    def fake_write(*args, **kwargs):
        events.append("write")
        writer_calls.append(kwargs)
        return receipt

    def fake_verify_receipt(*args, **kwargs):
        events.append("verify_receipt")
        return {}

    def forbidden(name):
        def fail(*args, **kwargs):
            forbidden_calls.append(name)
            raise AssertionError(f"Backfill called forbidden recomputation function: {name}")

        return fail

    monkeypatch.setattr(recovery, "verify_frozen_file_inventory", traced_inventory)
    monkeypatch.setattr(engine, "write_stage_receipt", fake_write)
    monkeypatch.setattr(recovery.protocol, "verify_stage_receipt", fake_verify_receipt)
    for name in (
        "evaluate",
        "_infer_and_cache_rois",
        "_infer_classifier_frames",
        "_load_segmenter_from_lock",
        "_load_classifier_checkpoint",
        "export_evaluation",
    ):
        monkeypatch.setattr(engine, name, forbidden(name))

    result = recovery.backfill_completed_attempt(
        cfg,
        "yolo26",
        17,
        {"alias": alias, "partial_inventory": inventory},
    )
    assert result == receipt
    assert len(writer_calls) == 1
    assert events == ["inventory", "write", "inventory", "verify_receipt"]
    assert forbidden_calls == []
    assert writer_calls[0]["metadata"]["receipt_backfilled_without_repeating_test_inference"]
    assert writer_calls[0]["metadata"]["pre_recovery_inventory_verified"]


def test_v2_refuses_to_recapture_when_pinned_v1_evidence_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from predicted_roi_study.config import load_config

    recovery = _load_recovery_module()
    cfg = load_config(CONFIG_PATH)
    snapshot_calls: list[str] = []

    def forbidden_snapshot(*args, **kwargs):
        snapshot_calls.append("snapshot")
        raise AssertionError("Recovery attempted to capture a replacement baseline")

    monkeypatch.setattr(recovery, "SUPERSEDED_AMENDMENT_ID", "missing_pinned_v1")
    monkeypatch.setattr(recovery, "_snapshot_partial_evaluation_once", forbidden_snapshot)
    with pytest.raises(recovery.RecoveryError, match="pinned prepared v1 amendment"):
        recovery.create_or_verify_amendment(cfg, {}, CANONICAL_GATE)
    assert snapshot_calls == []


def test_recovery_acquires_master_then_seed_launcher_lock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recovery = _load_recovery_module()
    acquired: list[str] = []

    @contextmanager
    def fake_lock(path: Path):
        acquired.append(path.name)
        yield

    monkeypatch.setattr(recovery, "_exclusive_file_lock", fake_lock)
    with recovery._single_instance_lock(tmp_path):
        pass
    assert acquired == ["five_seed_core_exclusive.lock", "clean_study_exclusive.lock"]
