"""Audited evaluation-only recovery for the frozen strict-ROI v1.2.0 run.

This module deliberately lives outside ``predicted_roi_study`` so importing or
executing it does not alter the frozen study code fingerprint.  It repairs one
receipt-path compatibility defect without changing models, checkpoints,
thresholds, calibration, splits, locks, or test predictions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from predicted_roi_study import protocol  # noqa: E402
from predicted_roi_study.config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    load_config,
    resolve_project_path,
    sha256_file,
)


AMENDMENT_ID = "evaluation_gate_path_compatibility_v2"
SUPERSEDED_AMENDMENT_ID = "evaluation_gate_path_compatibility_v1"
EXPECTED_STUDY_ID = "strict_predicted_roi_binary_4model_v1_2_0_clean"
EXPECTED_PROTOCOL_VERSION = "1.2.0"
EXPECTED_CONFIG_SHA256 = "f9ad4cfd5e4222b50dfd91f9084c7ffe4c8ba60f12e8ff54d1d4c623c324c7ad"
EXPECTED_CANONICAL_CONFIG_SHA256 = (
    "f16bfcd4715c7a823dcef671f9b23681a62fd217029aaab260ffcab57806e7f0"
)
EXPECTED_PROTOCOL_SHA256 = "61f6f6b90d08ee673d590017ec158c6e7a34885fcc1cf954c882dd89318df74e"
EXPECTED_CODE_SHA256 = "914ec1505a0d417b8b6766b3663160673b94d5ce107da68f81d3142bbf493776"
EXPECTED_MANIFEST_SHA256 = "4efe349f7f3a5a7eda430251592526311df3b644a02139afb8c423b256668b48"
EXPECTED_GATE_SHA256 = "4fb2bb8f298e4fa131d848c72db8e3b6fc0e533f478e37e8d30bde3b33c18bdd"
EXPECTED_SUPERSEDED_AMENDMENT_SHA256 = (
    "09e22e62a8f63cbec1f3e78a04fd2c8a907a02e835f0e9a6a5852cfdee2fcc90"
)
EXPECTED_PARTIAL_INVENTORY_SHA256 = (
    "57ae65513380fa01d977044c256eb47ab65a679bba17565c40b17f741f0ebe2f"
)

CLASSIFIER_STRATEGIES = ("model_specific", "standardized_resnet18")
EVALUATION_ARTIFACT_NAMES = (
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


class RecoveryError(RuntimeError):
    """Raised when an evaluation-only recovery invariant is not satisfied."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")


def _json_print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


def _copy_bytes_create_once(source: Path, destination: Path) -> Path:
    """Create an immutable byte-identical copy, refusing divergent reuse."""

    source = source.resolve()
    destination = destination.resolve()
    if not source.is_file():
        raise RecoveryError(f"Source file is missing: {source}")
    payload = source.read_bytes()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or destination.read_bytes() != payload:
            raise RecoveryError(f"refusing to overwrite a divergent immutable file: {destination}")
        return destination
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    if destination.read_bytes() != payload:
        raise RecoveryError(f"Byte-identical copy verification failed: {destination}")
    return destination


def ensure_test_access_alias(canonical_gate: Path, alias_path: Path) -> Path:
    """Create the exact legacy receipt path as a byte-identical gate alias."""

    alias = _copy_bytes_create_once(canonical_gate, alias_path)
    if sha256_file(alias) != sha256_file(canonical_gate):
        raise RecoveryError("Canonical and compatibility test-access gate hashes differ")
    return alias


def create_or_verify_byte_identical_alias(source: Path, alias: Path) -> Path:
    """Public regression-test entry point for immutable alias creation."""

    try:
        return ensure_test_access_alias(Path(source), Path(alias))
    except RecoveryError as error:
        raise protocol.ProtocolGateError(str(error)) from error


def _receipt_hashes(cfg: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    prepare = protocol.stage_receipt_path(cfg, "prepare")
    protocol.verify_stage_receipt(cfg, "prepare")
    result["prepare"] = sha256_file(prepare)
    for model in cfg["models"]:
        preflight = protocol.stage_receipt_path(cfg, "preflight", model=model)
        protocol.verify_stage_receipt(cfg, "preflight", model=model)
        result[f"preflight/{model}"] = sha256_file(preflight)
        for seed in cfg["split_seeds"]:
            for stage in (
                "train-segmenters",
                "build-rois",
                "train-classifiers",
                "lock",
            ):
                receipt = protocol.stage_receipt_path(cfg, stage, model=model, seed=int(seed))
                protocol.verify_stage_receipt(cfg, stage, model=model, seed=int(seed))
                result[f"{stage}/{model}/seed_{seed}"] = sha256_file(receipt)
    return result


def verify_frozen_state(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Revalidate the exact pre-test state bound into the existing global gate."""

    runtime = cfg.get("_runtime", {})
    observed = {
        "study_id": cfg.get("study_id"),
        "protocol_version": cfg.get("protocol_version"),
        "config_sha256": runtime.get("config_sha256"),
        "canonical_config_sha256": runtime.get("canonical_config_sha256"),
        "protocol_sha256": protocol.protocol_sha256(),
        "code_sha256": protocol.code_fingerprint()["sha256"],
        "manifest_sha256": cfg.get("dataset", {}).get("manifest_sha256"),
    }
    expected = {
        "study_id": EXPECTED_STUDY_ID,
        "protocol_version": EXPECTED_PROTOCOL_VERSION,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "canonical_config_sha256": EXPECTED_CANONICAL_CONFIG_SHA256,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "code_sha256": EXPECTED_CODE_SHA256,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
    }
    if observed != expected:
        raise RecoveryError(
            "Frozen study identity changed; evaluation-only recovery is forbidden: "
            + json.dumps({"expected": expected, "observed": observed}, sort_keys=True)
        )

    receipt_hashes = _receipt_hashes(cfg)
    lock_chains = protocol.assert_all_primary_locks(cfg)
    canonical_gate = (
        protocol.output_root(cfg) / "state" / "test_access_opened.json"
    ).resolve()
    # ``open_test_access`` creates the gate when it is absent.  Recovery must
    # never recreate it, so prove that the already-opened sentinel is present
    # and frozen before calling the original routine for read-only revalidation.
    if not canonical_gate.is_file():
        raise RecoveryError("The immutable global test-access gate is missing")
    if sha256_file(canonical_gate) != EXPECTED_GATE_SHA256:
        raise RecoveryError("The immutable global test-access gate hash changed")
    opened_gate = protocol.open_test_access(cfg).resolve()
    if opened_gate != canonical_gate:
        raise RecoveryError("The protocol returned an unexpected global test-access gate")
    gate = protocol.read_json(canonical_gate)
    gate_expected = {
        **expected,
        "unit_receipt_chain_sha256": lock_chains,
    }
    if {key: gate.get(key) for key in gate_expected} != gate_expected:
        raise RecoveryError("The global test-access gate no longer binds the frozen 20-lock set")
    return {
        "identity": expected,
        "canonical_gate": canonical_gate,
        "gate": gate,
        "gate_sha256": sha256_file(canonical_gate),
        "lock_chains": lock_chains,
        "receipt_hashes": receipt_hashes,
    }


def _snapshot_failure_evidence_once(output_root: Path, amendment_root: Path) -> dict[str, Any]:
    orchestration = output_root / "orchestration"
    status_source = orchestration / "five_seed_core_status.json"
    stderr_source = orchestration / "five_seed_core_resume_20260918_151436_stderr.log"
    status_snapshot = amendment_root / "pre_recovery_failed_core_status.json"
    stderr_snapshot = amendment_root / "pre_recovery_failure_stderr.log"
    if not status_snapshot.exists():
        _copy_bytes_create_once(status_source, status_snapshot)
    if not stderr_snapshot.exists():
        _copy_bytes_create_once(stderr_source, stderr_snapshot)
    failed = protocol.read_json(status_snapshot)
    if failed.get("state") != "failed" or failed.get("stage") != "evaluate":
        raise RecoveryError("The immutable recovery status snapshot is not failed/evaluate")
    return {
        "status": {
            "path": _relative(status_snapshot),
            "sha256": sha256_file(status_snapshot),
        },
        "stderr": {
            "path": _relative(stderr_snapshot),
            "sha256": sha256_file(stderr_snapshot),
        },
    }


def _snapshot_partial_evaluation_once(output_root: Path, amendment_root: Path) -> Path:
    destination = amendment_root / "pre_recovery_yolo26_seed17_inventory.json"
    if destination.exists():
        return destination
    evaluation = output_root / "runs" / "yolo26" / "seed_17" / "evaluation"
    attempt_path = evaluation / "attempt.json"
    if not attempt_path.is_file() or protocol.read_json(attempt_path).get("status") != "complete":
        raise RecoveryError("Expected completed yolo26/seed_17 partial evaluation is missing")
    files = []
    for path in sorted(item for item in evaluation.rglob("*") if item.is_file()):
        files.append(
            {
                "path": _relative(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    protocol.save_json_atomic(
        destination,
        {
            "schema": 1,
            "captured_utc": utc_now(),
            "scope": "pre-recovery completed outputs without an evaluate receipt",
            "model": "yolo26",
            "seed": 17,
            "file_count": len(files),
            "files": files,
        },
    )
    return destination


def verify_frozen_file_inventory(
    inventory_path: Path,
    expected_root: Path,
    *,
    model: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Require an exact path/size/SHA match to an immutable file inventory."""

    inventory_path = Path(inventory_path).resolve()
    expected_root = Path(expected_root).resolve()
    if not inventory_path.is_file():
        raise RecoveryError(f"Frozen inventory is missing: {inventory_path}")
    if not expected_root.is_dir():
        raise RecoveryError(f"Inventoried directory is missing: {expected_root}")

    payload = protocol.read_json(inventory_path)
    if payload.get("schema") != 1 or not isinstance(payload.get("files"), list):
        raise RecoveryError("Frozen inventory has an unsupported or malformed schema")
    if model is not None and payload.get("model") != model:
        raise RecoveryError("Frozen inventory model identity changed")
    if seed is not None and payload.get("seed") != seed:
        raise RecoveryError("Frozen inventory seed identity changed")

    expected: dict[str, Mapping[str, Any]] = {}
    for record in payload["files"]:
        if not isinstance(record, Mapping):
            raise RecoveryError("Frozen inventory contains a malformed file record")
        path_value = record.get("path")
        if not isinstance(path_value, str) or not path_value:
            raise RecoveryError("Frozen inventory contains an invalid file path")
        relative = Path(path_value)
        if relative.is_absolute():
            raise RecoveryError("Frozen inventory may contain only project-relative paths")
        resolved = (PROJECT_ROOT / relative).resolve()
        try:
            resolved.relative_to(PROJECT_ROOT.resolve())
            resolved.relative_to(expected_root)
        except ValueError as error:
            raise RecoveryError(
                f"Frozen inventory path escapes the expected evaluation tree: {path_value}"
            ) from error
        key = _relative(resolved)
        if key in expected:
            raise RecoveryError(f"Frozen inventory contains a duplicate path: {key}")
        expected[key] = record

    declared_count = payload.get("file_count")
    if declared_count != len(expected):
        raise RecoveryError(
            f"Frozen inventory file_count mismatch: declared={declared_count}, records={len(expected)}"
        )

    observed_paths = {
        _relative(path.resolve()): path.resolve()
        for path in expected_root.rglob("*")
        if path.is_file()
    }
    expected_keys = set(expected)
    observed_keys = set(observed_paths)
    if expected_keys != observed_keys:
        missing = sorted(expected_keys - observed_keys)
        extra = sorted(observed_keys - expected_keys)
        raise RecoveryError(
            "Current evaluation tree differs from the frozen pre-recovery inventory: "
            + json.dumps({"missing": missing, "extra": extra}, sort_keys=True)
        )

    for key in sorted(expected):
        record = expected[key]
        path = observed_paths[key]
        observed_size = path.stat().st_size
        observed_sha256 = sha256_file(path)
        if record.get("bytes") != observed_size or record.get("sha256") != observed_sha256:
            raise RecoveryError(
                "Current evaluation artifact differs from the frozen pre-recovery inventory: "
                + json.dumps(
                    {
                        "path": key,
                        "expected_bytes": record.get("bytes"),
                        "observed_bytes": observed_size,
                        "expected_sha256": record.get("sha256"),
                        "observed_sha256": observed_sha256,
                    },
                    sort_keys=True,
                )
            )
    return {
        "file_count": len(expected),
        "inventory_sha256": sha256_file(inventory_path),
    }


def create_or_verify_amendment(
    cfg: Mapping[str, Any], frozen: Mapping[str, Any], alias_path: Path
) -> dict[str, Any]:
    output_root = protocol.output_root(cfg)
    amendment_root = output_root / "state" / "amendments" / AMENDMENT_ID
    manifest_path = amendment_root / "amendment.json"
    superseded_manifest = (
        output_root
        / "state"
        / "amendments"
        / SUPERSEDED_AMENDMENT_ID
        / "amendment.json"
    )
    if not superseded_manifest.is_file():
        raise RecoveryError(
            "The pinned prepared v1 amendment is required; refusing to recapture a new baseline"
        )
    supersedes: dict[str, Any] | None = None
    partial_inventory: Path
    if superseded_manifest.is_file():
        if sha256_file(superseded_manifest) != EXPECTED_SUPERSEDED_AMENDMENT_SHA256:
            raise RecoveryError("The prepared v1 amendment manifest changed")
        superseded_payload = protocol.read_json(superseded_manifest)
        frozen_inventory_record = (
            superseded_payload.get("pre_recovery_evidence", {})
            .get("partial_evaluation_inventory", {})
        )
        frozen_inventory_path = frozen_inventory_record.get("path")
        frozen_inventory_sha256 = frozen_inventory_record.get("sha256")
        if not isinstance(frozen_inventory_path, str) or not isinstance(
            frozen_inventory_sha256, str
        ):
            raise RecoveryError("The prepared v1 amendment lacks its frozen inventory binding")
        partial_inventory = resolve_project_path(frozen_inventory_path, must_exist=True)
        if frozen_inventory_sha256 != EXPECTED_PARTIAL_INVENTORY_SHA256:
            raise RecoveryError("The prepared v1 manifest binds an unexpected inventory hash")
        if sha256_file(partial_inventory) != EXPECTED_PARTIAL_INVENTORY_SHA256:
            raise RecoveryError("The prepared v1 frozen inventory artifact changed")
        verification = verify_frozen_file_inventory(
            partial_inventory,
            output_root / "runs" / "yolo26" / "seed_17" / "evaluation",
            model="yolo26",
            seed=17,
        )
        supersedes = {
            "amendment_id": SUPERSEDED_AMENDMENT_ID,
            "manifest_path": _relative(superseded_manifest),
            "manifest_sha256": sha256_file(superseded_manifest),
            "frozen_inventory_path": _relative(partial_inventory),
            "frozen_inventory_sha256": frozen_inventory_sha256,
            "frozen_inventory_file_count": verification["file_count"],
            "current_tree_verified_against_frozen_inventory": True,
            "disposition": "prepared_only_superseded_before_any_evaluate_receipt",
        }

    if not manifest_path.exists() and supersedes is not None:
        evaluate_receipt_root = output_root / "state" / "receipts" / "evaluate"
        existing_evaluate_receipts = (
            list(evaluate_receipt_root.rglob("*.json"))
            if evaluate_receipt_root.exists()
            else []
        )
        if existing_evaluate_receipts:
            raise RecoveryError(
                "Cannot supersede the prepared v1 amendment after evaluation receipts exist"
            )

    amendment_root.mkdir(parents=True, exist_ok=True)
    document_path = PROJECT_ROOT / "scripts" / "EVALUATION_ONLY_RECOVERY_AMENDMENT_v1_2_0.md"
    if not document_path.is_file():
        raise RecoveryError(f"Recovery amendment document is missing: {document_path}")
    failure_evidence = _snapshot_failure_evidence_once(output_root, amendment_root)
    if supersedes is None:
        raise RecoveryError("The pinned prepared v1 amendment could not be verified")
    provenance_root = output_root / "provenance"
    provenance = {}
    for name in (
        "code_fingerprint.json",
        "config_strict_roi.snapshot.json",
        "fingerprinted_sources.snapshot.zip",
        "PROTOCOL.snapshot.md",
    ):
        path = provenance_root / name
        if not path.is_file():
            raise RecoveryError(f"Frozen provenance artifact is missing: {path}")
        provenance[name] = {"path": _relative(path), "sha256": sha256_file(path)}

    stable = {
        "schema": 1,
        "amendment_id": AMENDMENT_ID,
        "study_id": cfg["study_id"],
        "protocol_version": cfg["protocol_version"],
        "scope": "evaluation receipt gate-path compatibility only",
        "reason": (
            "Frozen validator expects /protocol/test_access.json while the frozen gate writer "
            "created /state/test_access_opened.json"
        ),
        "supersedes_prepared_amendment": supersedes,
        "attestations": {
            "training_repeated": False,
            "model_or_hyperparameter_changed": False,
            "threshold_or_calibration_changed": False,
            "test_selection_or_refitting": False,
            "canonical_gate_modified": False,
            "fingerprinted_source_modified": False,
            "completed_partial_test_inference_repeated": False,
        },
        "frozen_identity": dict(frozen["identity"]),
        "canonical_gate": {
            "path": _relative(Path(frozen["canonical_gate"])),
            "sha256": frozen["gate_sha256"],
            "opened_utc": frozen["gate"].get("opened_utc"),
        },
        "compatibility_alias": {
            "path": _relative(alias_path),
            "sha256": sha256_file(alias_path),
            "byte_identical_to_canonical_gate": True,
        },
        "unit_receipt_chain_sha256": dict(frozen["lock_chains"]),
        "upstream_receipt_sha256": dict(frozen["receipt_hashes"]),
        "recovery_runner": {"path": _relative(SCRIPT_PATH), "sha256": sha256_file(SCRIPT_PATH)},
        "amendment_document": {
            "path": _relative(document_path),
            "sha256": sha256_file(document_path),
        },
        "pre_recovery_evidence": {
            **failure_evidence,
            "partial_evaluation_inventory": {
                "path": _relative(partial_inventory),
                "sha256": sha256_file(partial_inventory),
            },
        },
        "frozen_provenance": provenance,
    }
    if manifest_path.exists():
        existing = protocol.read_json(manifest_path)
        comparable = dict(existing)
        comparable.pop("created_utc", None)
        if comparable != stable:
            raise RecoveryError("The immutable evaluation recovery amendment changed")
    else:
        protocol.save_json_atomic(manifest_path, {**stable, "created_utc": utc_now()})
    return {
        "root": amendment_root,
        "manifest": manifest_path,
        "document": document_path,
        "partial_inventory": partial_inventory,
        "canonical_gate": Path(frozen["canonical_gate"]),
        "alias": alias_path,
    }


def _amendment_metadata(amendment: Mapping[str, Path]) -> dict[str, Any]:
    return {
        "amendment_id": AMENDMENT_ID,
        "manifest_sha256": sha256_file(amendment["manifest"]),
        "canonical_gate_sha256": sha256_file(amendment["canonical_gate"]),
        "compatibility_alias_sha256": sha256_file(amendment["alias"]),
        "evaluation_only": True,
        "training_repeated": False,
        "test_selection_or_refitting": False,
    }


def _support_artifacts(amendment: Mapping[str, Path]) -> list[Path]:
    return [
        amendment["manifest"],
        amendment["document"],
        amendment["partial_inventory"],
        amendment["canonical_gate"],
        amendment["alias"],
        SCRIPT_PATH,
    ]


def install_recovery_hooks(
    cfg: Mapping[str, Any], amendment: Mapping[str, Path]
) -> tuple[Any, Any]:
    """Patch only engine bindings; persisted receipts remain normally verifiable."""

    from predicted_roi_study import engine

    original_open = engine.open_test_access
    original_write = engine.write_stage_receipt

    def recovery_open_test_access(active_cfg: Mapping[str, Any]) -> Path:
        canonical = protocol.open_test_access(active_cfg).resolve()
        if canonical != amendment["canonical_gate"].resolve():
            raise RecoveryError("Recovery opened an unexpected canonical test gate")
        return ensure_test_access_alias(canonical, amendment["alias"])

    def amended_write_stage_receipt(
        active_cfg: Mapping[str, Any],
        stage: str,
        *,
        model: str | None = None,
        seed: int | None = None,
        artifacts: Iterable[str | Path] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        artifact_list = list(artifacts)
        metadata_value = dict(metadata or {})
        if stage in {"evaluate", "summarize"}:
            artifact_list.extend(_support_artifacts(amendment))
            metadata_value["evaluation_only_protocol_amendment"] = _amendment_metadata(
                amendment
            )
        unique: list[Path] = []
        seen: set[Path] = set()
        for item in artifact_list:
            path = Path(item).resolve()
            if path not in seen:
                seen.add(path)
                unique.append(path)
        return original_write(
            active_cfg,
            stage,
            model=model,
            seed=seed,
            artifacts=unique,
            metadata=metadata_value,
        )

    engine.open_test_access = recovery_open_test_access
    engine.write_stage_receipt = amended_write_stage_receipt
    return original_open, original_write


def _existing_evaluation_artifacts(
    cfg: Mapping[str, Any], model: str, seed: int, alias_path: Path
) -> tuple[list[Path], dict[str, Any], dict[str, Any]]:
    """Reconstruct the unchanged engine inventory for a completed partial attempt."""

    run = protocol.unit_root(cfg, model, seed)
    evaluation_root = run / "evaluation"
    attempt_path = evaluation_root / "attempt.json"
    lock_path = run / "lock" / "primary_lock.json"
    test_roi_index = evaluation_root / "test_rois" / "index.csv"
    for required in (attempt_path, lock_path, test_roi_index, alias_path):
        if not required.is_file():
            raise RecoveryError(f"Completed attempt artifact is missing: {required}")
    attempt = protocol.read_json(attempt_path)
    lock = protocol.read_json(lock_path)
    if (
        attempt.get("status") != "complete"
        or attempt.get("model") != model
        or attempt.get("seed") != seed
        or set(attempt.get("classifier_strategies", {})) != set(CLASSIFIER_STRATEGIES)
    ):
        raise RecoveryError(f"Cannot backfill an incomplete or malformed attempt: {model}/{seed}")
    if attempt.get("test_access_sha256") != sha256_file(alias_path):
        raise RecoveryError("Completed attempt is not bound to the byte-identical gate alias")

    artifacts: list[Path] = [attempt_path, test_roi_index, lock_path, alias_path]
    for strategy in CLASSIFIER_STRATEGIES:
        branch = lock["classifier_strategies"][strategy]
        checkpoint = branch.get("classifier_checkpoint")
        if checkpoint:
            artifacts.append(resolve_project_path(checkpoint, must_exist=True))
        destination = evaluation_root / strategy
        for name in EVALUATION_ARTIFACT_NAMES:
            path = destination / name
            if path.is_file():
                artifacts.append(path)
        artifacts.extend(sorted(destination.glob("classification_calibration_decision_curves.*")))
        artifacts.extend(sorted(destination.glob("calibration_density_and_risk_coverage.*")))
        for key in (
            "classifier_history",
            "classifier_provenance",
            "classifier_training_result",
        ):
            artifacts.append(resolve_project_path(branch[key], must_exist=True))
    unique = list(dict.fromkeys(path.resolve() for path in artifacts))
    return unique, attempt, lock


def backfill_completed_attempt(
    cfg: Mapping[str, Any], model: str, seed: int, amendment: Mapping[str, Path]
) -> Path:
    """Write only the missing receipt; do not repeat completed test inference."""

    from predicted_roi_study import engine

    inventory_verification: dict[str, Any] | None = None
    if model == "yolo26" and seed == 17:
        inventory_verification = verify_frozen_file_inventory(
            amendment["partial_inventory"],
            protocol.unit_root(cfg, model, seed) / "evaluation",
            model=model,
            seed=seed,
        )

    artifacts, attempt, lock = _existing_evaluation_artifacts(
        cfg, model, seed, amendment["alias"]
    )
    receipt = engine.write_stage_receipt(
        cfg,
        "evaluate",
        model=model,
        seed=seed,
        artifacts=artifacts,
        metadata={
            "test_frames": int(attempt["test_frames"]),
            "primary_classifier_strategy": CLASSIFIER_STRATEGIES[0],
            "classifier_strategies": attempt["classifier_strategies"],
            "primary_system_evaluable": bool(lock["primary_system_evaluable"]),
            "calibrated_system_evaluable": bool(
                lock.get("calibrated_system_evaluable", False)
            ),
            "test_selection_or_refitting": False,
            "receipt_backfilled_without_repeating_test_inference": True,
            "pre_recovery_inventory_verified": inventory_verification is not None,
            "pre_recovery_inventory_sha256": (
                inventory_verification["inventory_sha256"]
                if inventory_verification is not None
                else None
            ),
        },
    )
    if inventory_verification is not None:
        verify_frozen_file_inventory(
            amendment["partial_inventory"],
            protocol.unit_root(cfg, model, seed) / "evaluation",
            model=model,
            seed=seed,
        )
    protocol.verify_stage_receipt(cfg, "evaluate", model=model, seed=seed)
    return receipt


def _write_runtime_status(
    cfg: Mapping[str, Any],
    *,
    state: str,
    stage: str,
    message: str,
    model: str | None = None,
    seed: int | None = None,
    completed_units: int | None = None,
) -> None:
    output_root = protocol.output_root(cfg)
    orchestration = output_root / "orchestration"
    orchestration.mkdir(parents=True, exist_ok=True)
    record = {
        "study_id": cfg["study_id"],
        "protocol_version": cfg["protocol_version"],
        "state": state,
        "stage": stage,
        "model": model,
        "seed": seed,
        "message": message,
        "process_id": os.getpid(),
        "updated_at": datetime.now().astimezone().isoformat(),
        "output_root": str(output_root),
        "config": str(Path(cfg["_runtime"]["config_path"]).resolve()),
        "config_sha256": cfg["_runtime"]["config_sha256"],
        "test_access_open": True,
        "resume_requested": True,
        "recovery_amendment_id": AMENDMENT_ID,
    }
    protocol.save_json_atomic(orchestration / "five_seed_core_status.json", record)
    with (orchestration / "five_seed_core_events.jsonl").open(
        "a", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    progress = {
        "pid": os.getpid(),
        "updated_unix": datetime.now().timestamp(),
        "phase": stage,
        "model": model,
        "seed": seed,
        "completed_units": completed_units,
        "target_units": 20,
    }
    protocol.save_json_atomic(output_root / "progress.json", progress)


@contextmanager
def _exclusive_file_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = lock_path.open("a+b")
    except OSError as error:
        raise RecoveryError(
            f"Another launcher holds the exclusive study lock: {lock_path.name}"
        ) from error
    acquired = False
    try:
        if handle.seek(0, os.SEEK_END) == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                acquired = True
            except OSError as error:
                raise RecoveryError(
                    f"Another launcher holds the exclusive study lock: {lock_path.name}"
                ) from error
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except OSError as error:
                raise RecoveryError(
                    f"Another launcher holds the exclusive study lock: {lock_path.name}"
                ) from error
        yield
    finally:
        try:
            if acquired:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


@contextmanager
def _single_instance_lock(output_root: Path):
    orchestration = output_root / "orchestration"
    # Match the ordinary master -> seed-launcher acquisition order exactly.
    # Holding both handles for the entire recovery excludes both supported
    # mutation entry points as well as a second recovery process.
    with _exclusive_file_lock(orchestration / "five_seed_core_exclusive.lock"):
        with _exclusive_file_lock(orchestration / "clean_study_exclusive.lock"):
            yield


def run_recovery(cfg: Mapping[str, Any], amendment: Mapping[str, Path]) -> dict[str, Any]:
    from predicted_roi_study import engine

    install_recovery_hooks(cfg, amendment)
    completed = 0
    outputs: list[dict[str, Any]] = []
    for seed_value in cfg["split_seeds"]:
        seed = int(seed_value)
        for model_value in cfg["models"]:
            model = str(model_value)
            receipt_path = protocol.stage_receipt_path(
                cfg, "evaluate", model=model, seed=seed
            )
            if receipt_path.is_file():
                if model == "yolo26" and seed == 17:
                    verify_frozen_file_inventory(
                        amendment["partial_inventory"],
                        protocol.unit_root(cfg, model, seed) / "evaluation",
                        model=model,
                        seed=seed,
                    )
                protocol.verify_stage_receipt(cfg, "evaluate", model=model, seed=seed)
                action = "already_complete"
            else:
                attempt_path = protocol.unit_root(cfg, model, seed) / "evaluation" / "attempt.json"
                attempt = protocol.read_json(attempt_path) if attempt_path.is_file() else None
                if isinstance(attempt, Mapping) and attempt.get("status") == "complete":
                    if model != "yolo26" or seed != 17:
                        raise RecoveryError(
                            "A complete receipt-less attempt outside the frozen yolo26/seed_17 "
                            "inventory cannot be backfilled automatically"
                        )
                    _write_runtime_status(
                        cfg,
                        state="running",
                        stage="evaluate_recovery_backfill",
                        model=model,
                        seed=seed,
                        completed_units=completed,
                        message=(
                            "Backfilling the missing immutable receipt without repeating completed test inference"
                        ),
                    )
                    backfill_completed_attempt(cfg, model, seed, amendment)
                    action = "receipt_backfilled"
                else:
                    _write_runtime_status(
                        cfg,
                        state="running",
                        stage="evaluate_recovery",
                        model=model,
                        seed=seed,
                        completed_units=completed,
                        message="Evaluating the next frozen model/seed unit under the audited amendment",
                    )
                    protocol.assert_prerequisites(cfg, "evaluate", model=model, seed=seed)
                    engine.evaluate(cfg, model, seed)
                    protocol.verify_stage_receipt(cfg, "evaluate", model=model, seed=seed)
                    action = "evaluated"
            completed += 1
            outputs.append({"model": model, "seed": seed, "action": action})
            _write_runtime_status(
                cfg,
                state="running",
                stage="evaluate_recovery",
                model=model,
                seed=seed,
                completed_units=completed,
                message=f"Completed {completed}/20 frozen model/seed evaluations",
            )

    _write_runtime_status(
        cfg,
        state="running",
        stage="summarize",
        completed_units=20,
        message="Creating immutable five-seed primary and secondary summaries",
    )
    summarize_receipt = protocol.stage_receipt_path(cfg, "summarize")
    if summarize_receipt.is_file():
        protocol.verify_stage_receipt(cfg, "summarize")
        summary_action = "already_complete"
    else:
        protocol.assert_prerequisites(cfg, "summarize")
        engine.summarize(cfg)
        protocol.verify_stage_receipt(cfg, "summarize")
        summary_action = "summarized"

    audit = protocol.audit_state(cfg)
    audit_path = protocol.output_root(cfg) / "orchestration" / "evaluation_recovery_final_audit.json"
    protocol.save_json_atomic(audit_path, audit)
    _write_runtime_status(
        cfg,
        state="complete",
        stage="complete",
        completed_units=20,
        message="Five-seed evaluation and summaries completed under the audited path amendment",
    )
    return {
        "status": "complete",
        "amendment": str(amendment["manifest"]),
        "units": outputs,
        "summary_action": summary_action,
        "audit": str(audit_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Revalidate the frozen gate and 20 lock chains without writing or evaluating",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Create and verify the immutable amendment/alias, then stop before evaluation",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    output_root = protocol.output_root(cfg)
    with _single_instance_lock(output_root):
        frozen = verify_frozen_state(cfg)
        alias_path = output_root / "protocol" / "test_access.json"
        if args.dry_run:
            _json_print(
                {
                    "status": "ready",
                    "amendment_id": AMENDMENT_ID,
                    "code_sha256": frozen["identity"]["code_sha256"],
                    "gate_sha256": frozen["gate_sha256"],
                    "lock_count": len(frozen["lock_chains"]),
                    "planned_alias": str(alias_path),
                    "scientific_artifacts_written": False,
                }
            )
            return 0
        alias = ensure_test_access_alias(Path(frozen["canonical_gate"]), alias_path)
        amendment = create_or_verify_amendment(cfg, frozen, alias)
        if args.prepare_only:
            _json_print(
                {
                    "status": "prepared",
                    "amendment": str(amendment["manifest"]),
                    "alias": str(alias),
                    "alias_sha256": sha256_file(alias),
                }
            )
            return 0
        try:
            result = run_recovery(cfg, amendment)
        except Exception as error:
            _write_runtime_status(
                cfg,
                state="failed",
                stage="evaluation_recovery",
                message=f"{type(error).__name__}: {error}",
            )
            traceback.print_exc()
            raise
    _json_print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
