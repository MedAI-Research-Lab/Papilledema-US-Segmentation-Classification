"""Fail-closed provenance and test-access gates for the three-class study.

The module deliberately stops at protocol mechanics. It does not train a
model, calculate a metric, or expose a CLI. Development artifacts from the
completed binary study are imported by hash receipt; test ROI artifacts remain
unread until every three-class validation lock is present and a global sentinel
has been written.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import (
    CLASS_ORDER,
    EXPECTED_MODELS,
    EXPECTED_SEEDS,
    canonical_sha256,
    config_without_runtime,
    resolve_project_path,
    sha256_file,
    upstream_artifact_paths,
)


RECEIPT_SCHEMA_VERSION = 1
LOCK_SCHEMA_VERSION = 1
TEST_ACCESS_SCHEMA_VERSION = 1


class ProtocolGateError(RuntimeError):
    """Raised when provenance, validation locks, or test-access gates fail."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def save_json_atomic(path: str | Path, value: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
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


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def output_root(cfg: Mapping[str, Any]) -> Path:
    return resolve_project_path(str(cfg["output"]))


def unit_root(cfg: Mapping[str, Any], model: str, seed: int) -> Path:
    _validate_unit(cfg, model, seed)
    return output_root(cfg) / "runs" / model / f"seed_{seed}"


def protocol_sha256() -> str:
    return sha256_file(Path(__file__).resolve())


def _current_package_code_inventory() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    return {
        path.name: sha256_file(path)
        for path in sorted(package.glob("*.py"))
    }


def package_code_sha256() -> str:
    """Return the package-code digest used by ``engine._generic_receipt``."""

    inventory = _current_package_code_inventory()
    return canonical_sha256(list(inventory.items()))


def _config_sha256(cfg: Mapping[str, Any]) -> str:
    return canonical_sha256(config_without_runtime(cfg))


def _validate_unit(cfg: Mapping[str, Any], model: str, seed: int) -> None:
    if model not in tuple(cfg.get("models", ())):
        raise ProtocolGateError(f"Unknown model: {model!r}")
    if seed not in tuple(cfg.get("split_seeds", ())):
        raise ProtocolGateError(f"Unknown seed: {seed!r}")


def _validate_strategy(cfg: Mapping[str, Any], strategy: str) -> None:
    expected = tuple(cfg["classifier"]["strategy_order"])
    if strategy not in expected:
        raise ProtocolGateError(f"Unknown classifier strategy: {strategy!r}")


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    project = resolve_project_path(".")
    try:
        return resolved.relative_to(project).as_posix()
    except ValueError:
        return str(resolved)


def _artifact_record(path: str | Path, *, role: str) -> dict[str, Any]:
    source = Path(path).resolve()
    if not source.is_file():
        raise ProtocolGateError(f"Required artifact is missing: {source}")
    return {
        "role": role,
        "path": _portable_path(source),
        "sha256": sha256_file(source),
        "size_bytes": source.stat().st_size,
    }


def _resolve_record_path(record: Mapping[str, Any]) -> Path:
    raw = record.get("path")
    if not isinstance(raw, str) or not raw:
        raise ProtocolGateError("Artifact record path is malformed")
    return resolve_project_path(raw)


def _read_json_object(path: Path, *, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise ProtocolGateError(f"Required {label} is missing: {path}")
    try:
        value = read_json(path)
    except (OSError, ValueError) as error:
        raise ProtocolGateError(f"Required {label} is unreadable: {path}") from error
    if not isinstance(value, Mapping):
        raise ProtocolGateError(f"Required {label} must contain a JSON object")
    return value


def _verify_artifact_record(
    record: Mapping[str, Any],
    *,
    expected_role: str,
    expected_path: Path | None = None,
) -> str:
    if record.get("role") != expected_role:
        raise ProtocolGateError(
            f"Artifact role mismatch: expected {expected_role!r}, observed {record.get('role')!r}"
        )
    path = _resolve_record_path(record)
    if expected_path is not None and path.resolve() != expected_path.resolve():
        raise ProtocolGateError(f"Artifact path mismatch for role {expected_role}")
    if not path.is_file():
        raise ProtocolGateError(f"Artifact disappeared: {path}")
    digest = sha256_file(path)
    if digest != record.get("sha256"):
        raise ProtocolGateError(f"Artifact hash mismatch for role {expected_role}")
    if path.stat().st_size != record.get("size_bytes"):
        raise ProtocolGateError(f"Artifact size mismatch for role {expected_role}")
    return digest


def _prepared_code_freeze(cfg: Mapping[str, Any]) -> dict[str, str]:
    """Verify and describe the code state attested by ``engine.prepare``."""

    root = output_root(cfg)
    inventory_path = root / "provenance" / "code_inventory.json"
    prepare_receipt_path = root / "state" / "receipts" / "prepare.json"
    inventory = _read_json_object(inventory_path, label="prepare code inventory")
    current_inventory = _current_package_code_inventory()
    if dict(inventory) != current_inventory:
        raise ProtocolGateError(
            "Prepare code inventory does not match the current package code"
        )
    current_code_sha256 = canonical_sha256(list(current_inventory.items()))

    receipt = _read_json_object(prepare_receipt_path, label="prepare receipt")
    expected_identity = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "stage": "prepare",
        "study_id": cfg["study_id"],
        "config_sha256": _config_sha256(cfg),
        "model": None,
        "seed": None,
        "classifier_strategy": None,
    }
    for key, expected in expected_identity.items():
        if receipt.get(key) != expected:
            raise ProtocolGateError(f"Prepare receipt identity mismatch: {key}")
    if receipt.get("code_sha256") != current_code_sha256:
        raise ProtocolGateError(
            "Prepare receipt code hash does not match the current package code"
        )

    records = receipt.get("artifacts")
    if not isinstance(records, list):
        raise ProtocolGateError("Prepare receipt has no artifact inventory")
    inventory_records = [
        record
        for record in records
        if isinstance(record, Mapping) and record.get("role") == "code_inventory"
    ]
    if len(inventory_records) != 1:
        raise ProtocolGateError(
            "Prepare receipt must attest exactly one code_inventory artifact"
        )
    _verify_artifact_record(
        inventory_records[0],
        expected_role="code_inventory",
        expected_path=inventory_path,
    )

    return {
        "code_inventory": _portable_path(inventory_path),
        "code_inventory_sha256": sha256_file(inventory_path),
        "prepare_receipt": _portable_path(prepare_receipt_path),
        "prepare_receipt_sha256": sha256_file(prepare_receipt_path),
        "code_sha256": current_code_sha256,
    }


def _verify_test_access_code_freeze(
    cfg: Mapping[str, Any], sentinel: Mapping[str, Any]
) -> None:
    root = output_root(cfg)
    bound_paths = {
        "code_inventory": root / "provenance" / "code_inventory.json",
        "prepare_receipt": root / "state" / "receipts" / "prepare.json",
    }
    for field, expected_path in bound_paths.items():
        recorded_path = sentinel.get(field)
        if (
            not isinstance(recorded_path, str)
            or not recorded_path
            or resolve_project_path(recorded_path).resolve() != expected_path.resolve()
        ):
            raise ProtocolGateError(f"Test-access {field} path mismatch")
        if not expected_path.is_file():
            raise ProtocolGateError(f"Test-access {field} disappeared: {expected_path}")
        if sentinel.get(f"{field}_sha256") != sha256_file(expected_path):
            raise ProtocolGateError(f"Test-access {field} hash mismatch")

    if sentinel.get("code_sha256") != package_code_sha256():
        raise ProtocolGateError("Test-access package code hash mismatch")
    _prepared_code_freeze(cfg)


def upstream_import_receipt_path(
    cfg: Mapping[str, Any], model: str, seed: int
) -> Path:
    _validate_unit(cfg, model, seed)
    return (
        output_root(cfg)
        / "state"
        / "receipts"
        / "import-upstream-development"
        / model
        / f"seed_{seed}.json"
    )


def deferred_test_import_receipt_path(
    cfg: Mapping[str, Any], model: str, seed: int
) -> Path:
    _validate_unit(cfg, model, seed)
    return (
        output_root(cfg)
        / "state"
        / "receipts"
        / "import-upstream-test"
        / model
        / f"seed_{seed}.json"
    )


def validation_lock_path(
    cfg: Mapping[str, Any], model: str, seed: int, strategy: str
) -> Path:
    _validate_unit(cfg, model, seed)
    _validate_strategy(cfg, strategy)
    return unit_root(cfg, model, seed) / "locks" / f"{strategy}_validation_lock.json"


def test_access_path(cfg: Mapping[str, Any]) -> Path:
    return output_root(cfg) / str(cfg["test_access"]["sentinel"])


def _write_new_or_verify_identical(path: Path, payload: Mapping[str, Any]) -> Path:
    if path.exists():
        existing = read_json(path)
        left = dict(existing)
        right = dict(payload)
        left.pop("created_utc", None)
        right.pop("created_utc", None)
        if left != right:
            raise ProtocolGateError(f"Refusing to overwrite a non-identical receipt: {path}")
        return path
    return save_json_atomic(path, payload)


def write_upstream_import_receipt(
    cfg: Mapping[str, Any],
    model: str,
    seed: int,
    *,
    artifact_paths: Mapping[str, str | Path] | None = None,
) -> Path:
    """Hash the development-only upstream artifacts for one model×seed."""

    _validate_unit(cfg, model, seed)
    expected = upstream_artifact_paths(cfg, model, seed, phase="development")
    supplied = (
        {role: Path(path) for role, path in artifact_paths.items()}
        if artifact_paths is not None
        else expected
    )
    if tuple(supplied) != tuple(expected):
        raise ProtocolGateError(
            "Development import requires exactly the locked development artifact roles"
        )
    records: list[dict[str, Any]] = []
    for role in expected:
        path = supplied[role]
        if path.resolve() != expected[role].resolve():
            raise ProtocolGateError(f"Unexpected upstream path for role {role}")
        lowered = path.as_posix().lower()
        if "/classifiers/" in lowered or "/evaluation/test_rois/" in lowered:
            raise ProtocolGateError(
                "Binary classifier and deferred test artifacts cannot enter development imports"
            )
        records.append(_artifact_record(path, role=role))

    split_spec = cfg["split_policy"]["sources"][str(seed)]
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "stage": "import-upstream-development",
        "study_id": cfg["study_id"],
        "config_sha256": _config_sha256(cfg),
        "protocol_sha256": protocol_sha256(),
        "source_study_id": cfg["frozen_upstream"]["source_study_id"],
        "model": model,
        "seed": seed,
        "split_sha256": split_spec["sha256"],
        "artifacts": records,
        "segmenter_and_roi_parameters_frozen": True,
        "binary_classifier_artifacts_imported": False,
        "test_artifacts_read": False,
        "created_utc": utc_now(),
    }
    destination = upstream_import_receipt_path(cfg, model, seed)
    return _write_new_or_verify_identical(destination, receipt)


def verify_upstream_import_receipt(
    cfg: Mapping[str, Any], model: str, seed: int
) -> dict[str, Any]:
    destination = upstream_import_receipt_path(cfg, model, seed)
    if not destination.is_file():
        raise ProtocolGateError(f"Missing upstream import receipt: {destination}")
    receipt = read_json(destination)
    expected_identity = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "stage": "import-upstream-development",
        "study_id": cfg["study_id"],
        "config_sha256": _config_sha256(cfg),
        "protocol_sha256": protocol_sha256(),
        "source_study_id": cfg["frozen_upstream"]["source_study_id"],
        "model": model,
        "seed": seed,
        "split_sha256": cfg["split_policy"]["sources"][str(seed)]["sha256"],
        "segmenter_and_roi_parameters_frozen": True,
        "binary_classifier_artifacts_imported": False,
        "test_artifacts_read": False,
    }
    for key, value in expected_identity.items():
        if receipt.get(key) != value:
            raise ProtocolGateError(f"Upstream import receipt identity mismatch: {key}")
    records = receipt.get("artifacts")
    expected_paths = upstream_artifact_paths(cfg, model, seed, phase="development")
    if not isinstance(records, list) or [item.get("role") for item in records] != list(expected_paths):
        raise ProtocolGateError("Upstream import receipt has the wrong artifact role set")
    for record in records:
        role = record["role"]
        expected_path = expected_paths[role]
        recorded_path = _resolve_record_path(record)
        exact_path = expected_path if expected_path.exists() else recorded_path
        _verify_artifact_record(record, expected_role=role, expected_path=exact_path)
        lowered = recorded_path.as_posix().lower()
        if "/classifiers/" in lowered or "/evaluation/test_rois/" in lowered:
            raise ProtocolGateError("Development receipt contains a forbidden upstream artifact")
    return receipt


def assert_all_upstream_imports(cfg: Mapping[str, Any]) -> dict[str, str]:
    digests: dict[str, str] = {}
    for model in tuple(cfg["models"]):
        for seed in tuple(cfg["split_seeds"]):
            path = upstream_import_receipt_path(cfg, model, seed)
            verify_upstream_import_receipt(cfg, model, seed)
            digests[f"{model}:seed_{seed}"] = sha256_file(path)
    expected = cfg["test_access"]["required_development_import_receipts"]
    if len(digests) != expected:
        raise ProtocolGateError(
            f"Expected {expected} development import receipts; observed {len(digests)}"
        )
    return digests


def validate_validation_lock_metadata(
    cfg: Mapping[str, Any], strategy: str, metadata: Mapping[str, Any]
) -> None:
    _validate_strategy(cfg, strategy)
    if not isinstance(metadata, Mapping):
        raise ProtocolGateError("Validation-lock metadata must be an object")
    required_exact = {
        "classifier_strategy": strategy,
        "classifier_outputs": 3,
        "class_order": list(CLASS_ORDER),
        "training_roi_source": "frozen_out_of_fold_predicted_roi",
        "selection_partition": "validation",
        "primary_prediction_level": "patient",
        "primary_endpoint": "three_class_failure_aware_balanced_accuracy",
        "decision_rule": "argmax",
        "primary_abstention_policy": "structural_roi_gate_only",
        "test_data_read": False,
        "test_selection_or_refitting": False,
        "binary_head_threshold_or_calibrator_reused": False,
    }
    for key, expected in required_exact.items():
        if metadata.get(key) != expected:
            raise ProtocolGateError(
                f"Validation-lock metadata {key!r} must equal {expected!r}"
            )
    monitor_status = metadata.get("monitor_status")
    if not isinstance(monitor_status, str) or not monitor_status:
        raise ProtocolGateError("Validation lock requires monitor_status")
    calibration = metadata.get("calibration")
    if not isinstance(calibration, Mapping) or tuple(calibration) != ("eye", "patient"):
        raise ProtocolGateError("Validation lock requires eye and patient calibration records")
    for level in ("eye", "patient"):
        entry = calibration[level]
        if not isinstance(entry, Mapping):
            raise ProtocolGateError(f"Calibration record for {level} must be an object")
        status = entry.get("status")
        if status != "available":
            raise ProtocolGateError(
                f"Validation lock requires available {level} calibration"
            )
        temperature = entry.get("temperature")
        valid_temperature = False
        if not isinstance(temperature, bool) and isinstance(temperature, (int, float)):
            try:
                valid_temperature = math.isfinite(temperature) and temperature > 0
            except (OverflowError, TypeError, ValueError):
                valid_temperature = False
        if not valid_temperature:
            raise ProtocolGateError(
                f"Available {level} calibration requires finite positive temperature"
            )


def write_validation_lock(
    cfg: Mapping[str, Any],
    model: str,
    seed: int,
    strategy: str,
    *,
    artifacts: Mapping[str, str | Path],
    metadata: Mapping[str, Any],
) -> Path:
    _validate_unit(cfg, model, seed)
    _validate_strategy(cfg, strategy)
    validate_validation_lock_metadata(cfg, strategy, metadata)
    upstream_receipt = upstream_import_receipt_path(cfg, model, seed)
    verify_upstream_import_receipt(cfg, model, seed)
    if not artifacts:
        raise ProtocolGateError("A validation lock requires classifier artifacts")
    records = [_artifact_record(path, role=role) for role, path in artifacts.items()]
    if len({item["role"] for item in records}) != len(records):
        raise ProtocolGateError("Validation-lock artifact roles must be unique")
    lock = {
        "schema_version": LOCK_SCHEMA_VERSION,
        "stage": "threeclass-validation-lock",
        "study_id": cfg["study_id"],
        "config_sha256": _config_sha256(cfg),
        "protocol_sha256": protocol_sha256(),
        "model": model,
        "seed": seed,
        "classifier_strategy": strategy,
        "upstream_import_receipt": _portable_path(upstream_receipt),
        "upstream_import_receipt_sha256": sha256_file(upstream_receipt),
        "artifacts": records,
        "metadata": dict(metadata),
        "created_utc": utc_now(),
    }
    destination = validation_lock_path(cfg, model, seed, strategy)
    return _write_new_or_verify_identical(destination, lock)


def verify_validation_lock(
    cfg: Mapping[str, Any], model: str, seed: int, strategy: str
) -> dict[str, Any]:
    destination = validation_lock_path(cfg, model, seed, strategy)
    if not destination.is_file():
        raise ProtocolGateError(f"Missing validation lock: {destination}")
    lock = read_json(destination)
    expected = {
        "schema_version": LOCK_SCHEMA_VERSION,
        "stage": "threeclass-validation-lock",
        "study_id": cfg["study_id"],
        "config_sha256": _config_sha256(cfg),
        "protocol_sha256": protocol_sha256(),
        "model": model,
        "seed": seed,
        "classifier_strategy": strategy,
    }
    for key, value in expected.items():
        if lock.get(key) != value:
            raise ProtocolGateError(f"Validation lock identity mismatch: {key}")
    upstream_receipt = upstream_import_receipt_path(cfg, model, seed)
    verify_upstream_import_receipt(cfg, model, seed)
    if (
        resolve_project_path(lock.get("upstream_import_receipt", "")).resolve()
        != upstream_receipt.resolve()
    ):
        raise ProtocolGateError("Validation lock references the wrong upstream receipt")
    if lock.get("upstream_import_receipt_sha256") != sha256_file(upstream_receipt):
        raise ProtocolGateError("Validation lock upstream receipt hash mismatch")
    records = lock.get("artifacts")
    if not isinstance(records, list) or not records:
        raise ProtocolGateError("Validation lock has no classifier artifacts")
    for record in records:
        role = record.get("role")
        if not isinstance(role, str) or not role:
            raise ProtocolGateError("Validation-lock artifact role is malformed")
        _verify_artifact_record(record, expected_role=role)
    validate_validation_lock_metadata(cfg, strategy, lock.get("metadata", {}))
    return lock


def assert_all_validation_locks(cfg: Mapping[str, Any]) -> dict[str, str]:
    digests: dict[str, str] = {}
    for model in tuple(cfg["models"]):
        for seed in tuple(cfg["split_seeds"]):
            for strategy in tuple(cfg["classifier"]["strategy_order"]):
                path = validation_lock_path(cfg, model, seed, strategy)
                verify_validation_lock(cfg, model, seed, strategy)
                digests[f"{model}:seed_{seed}:{strategy}"] = sha256_file(path)
    expected = cfg["test_access"]["required_validation_locks"]
    if len(digests) != expected:
        raise ProtocolGateError(
            f"Expected {expected} validation locks; observed {len(digests)}"
        )
    return digests


def open_test_access(cfg: Mapping[str, Any]) -> Path:
    """Open the new study's test gate only after all imports and locks."""

    imports = assert_all_upstream_imports(cfg)
    locks = assert_all_validation_locks(cfg)
    code_freeze = _prepared_code_freeze(cfg)
    prior = cfg["prior_test_use"]
    disclosure = prior["required_disclosure"]
    if (
        prior.get("source_binary_test_was_opened") is not True
        or prior.get("confirmatory_claim_allowed") is not False
        or "exploratory" not in disclosure.lower()
        or "post-hoc" not in disclosure.lower()
    ):
        raise ProtocolGateError("Prior-test-use disclosure is absent or misleading")
    source_sentinel = resolve_project_path(
        prior["test_open_sentinel"], must_exist=True
    )
    source_anchor = cfg["frozen_upstream"]["source_anchors"][
        "source_test_access_sentinel"
    ]
    source_digest = sha256_file(source_sentinel)
    if source_digest != source_anchor["sha256"]:
        raise ProtocolGateError("Source binary test-access sentinel hash mismatch")

    payload = {
        "schema_version": TEST_ACCESS_SCHEMA_VERSION,
        "stage": "threeclass-test-access-opened",
        "study_id": cfg["study_id"],
        "config_sha256": _config_sha256(cfg),
        "protocol_sha256": protocol_sha256(),
        **code_freeze,
        "development_import_receipts": imports,
        "validation_locks": locks,
        "development_import_receipt_count": len(imports),
        "validation_lock_count": len(locks),
        "test_artifacts_read_before_open": False,
        "test_selection_or_refitting_allowed": False,
        "prior_test_use": {
            "source_binary_test_was_opened": True,
            "source_binary_results_were_examined": True,
            "same_patient_memberships_are_reused": True,
            "source_test_access_sentinel": _portable_path(source_sentinel),
            "source_test_access_sentinel_sha256": source_digest,
            "analysis_status": cfg["analysis_status"],
            "confirmatory_claim_allowed": False,
            "required_disclosure": disclosure,
        },
        "created_utc": utc_now(),
    }
    destination = test_access_path(cfg)
    return _write_new_or_verify_identical(destination, payload)


def assert_test_access_open(cfg: Mapping[str, Any]) -> dict[str, Any]:
    destination = test_access_path(cfg)
    if not destination.is_file():
        raise ProtocolGateError(
            "Test access is closed: all validation locks must exist before test artifacts are read"
        )
    sentinel = read_json(destination)
    if sentinel.get("study_id") != cfg["study_id"]:
        raise ProtocolGateError("Test-access sentinel study identity mismatch")
    if sentinel.get("config_sha256") != _config_sha256(cfg):
        raise ProtocolGateError("Test-access sentinel config hash mismatch")
    if sentinel.get("protocol_sha256") != protocol_sha256():
        raise ProtocolGateError("Test-access sentinel protocol hash mismatch")
    _verify_test_access_code_freeze(cfg, sentinel)
    if sentinel.get("test_artifacts_read_before_open") is not False:
        raise ProtocolGateError("Test-access sentinel reports premature test access")
    prior = sentinel.get("prior_test_use")
    if not isinstance(prior, Mapping):
        raise ProtocolGateError("Test-access sentinel lacks prior-use disclosure")
    if prior.get("confirmatory_claim_allowed") is not False:
        raise ProtocolGateError("Test-access sentinel permits a prohibited confirmatory claim")
    if prior.get("required_disclosure") != cfg["prior_test_use"]["required_disclosure"]:
        raise ProtocolGateError("Test-access prior-use disclosure changed")
    imports = assert_all_upstream_imports(cfg)
    locks = assert_all_validation_locks(cfg)
    if sentinel.get("development_import_receipts") != imports:
        raise ProtocolGateError("Test-access import receipt set changed")
    if sentinel.get("validation_locks") != locks:
        raise ProtocolGateError("Test-access validation lock set changed")
    return sentinel


def write_deferred_test_import_receipt(
    cfg: Mapping[str, Any],
    model: str,
    seed: int,
    *,
    artifact_paths: Mapping[str, str | Path] | None = None,
) -> Path:
    """Hash source test ROI artifacts, but only after the global gate is open."""

    _validate_unit(cfg, model, seed)
    sentinel = assert_test_access_open(cfg)
    expected = upstream_artifact_paths(cfg, model, seed, phase="test")
    supplied = (
        {role: Path(path) for role, path in artifact_paths.items()}
        if artifact_paths is not None
        else expected
    )
    if tuple(supplied) != tuple(expected):
        raise ProtocolGateError(
            "Deferred test import requires exactly the locked test artifact roles"
        )
    for role in expected:
        if supplied[role].resolve() != expected[role].resolve():
            raise ProtocolGateError(f"Unexpected deferred test path for role {role}")
    records = [_artifact_record(supplied[role], role=role) for role in expected]
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "stage": "import-upstream-test-after-global-lock",
        "study_id": cfg["study_id"],
        "config_sha256": _config_sha256(cfg),
        "protocol_sha256": protocol_sha256(),
        "model": model,
        "seed": seed,
        "test_access_sentinel": _portable_path(test_access_path(cfg)),
        "test_access_sentinel_sha256": sha256_file(test_access_path(cfg)),
        "prior_use_disclosure_sha256": canonical_sha256(
            sentinel["prior_test_use"]["required_disclosure"]
        ),
        "test_selection_or_refitting": False,
        "artifacts": records,
        "created_utc": utc_now(),
    }
    destination = deferred_test_import_receipt_path(cfg, model, seed)
    return _write_new_or_verify_identical(destination, receipt)


def verify_deferred_test_import_receipt(
    cfg: Mapping[str, Any], model: str, seed: int
) -> dict[str, Any]:
    assert_test_access_open(cfg)
    destination = deferred_test_import_receipt_path(cfg, model, seed)
    if not destination.is_file():
        raise ProtocolGateError(f"Missing deferred test import receipt: {destination}")
    receipt = read_json(destination)
    for key, expected in {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "stage": "import-upstream-test-after-global-lock",
        "study_id": cfg["study_id"],
        "config_sha256": _config_sha256(cfg),
        "protocol_sha256": protocol_sha256(),
        "model": model,
        "seed": seed,
        "test_selection_or_refitting": False,
    }.items():
        if receipt.get(key) != expected:
            raise ProtocolGateError(f"Deferred test import identity mismatch: {key}")
    if receipt.get("test_access_sentinel_sha256") != sha256_file(
        test_access_path(cfg)
    ):
        raise ProtocolGateError("Deferred test import sentinel hash mismatch")
    expected_paths = upstream_artifact_paths(cfg, model, seed, phase="test")
    records = receipt.get("artifacts")
    if not isinstance(records, list) or [
        record.get("role") for record in records
    ] != list(expected_paths):
        raise ProtocolGateError("Deferred test import role set is malformed")
    for record in records:
        role = record["role"]
        expected_path = expected_paths[role]
        recorded_path = _resolve_record_path(record)
        exact_path = expected_path if expected_path.exists() else recorded_path
        _verify_artifact_record(record, expected_role=role, expected_path=exact_path)
    return receipt


def iter_units(cfg: Mapping[str, Any]) -> Sequence[tuple[str, int]]:
    return tuple(
        (model, seed)
        for model in tuple(cfg.get("models", EXPECTED_MODELS))
        for seed in tuple(cfg.get("split_seeds", EXPECTED_SEEDS))
    )


__all__ = [
    "LOCK_SCHEMA_VERSION",
    "ProtocolGateError",
    "RECEIPT_SCHEMA_VERSION",
    "TEST_ACCESS_SCHEMA_VERSION",
    "assert_all_upstream_imports",
    "assert_all_validation_locks",
    "assert_test_access_open",
    "deferred_test_import_receipt_path",
    "iter_units",
    "open_test_access",
    "output_root",
    "package_code_sha256",
    "protocol_sha256",
    "read_json",
    "save_json_atomic",
    "test_access_path",
    "unit_root",
    "upstream_import_receipt_path",
    "utc_now",
    "validate_validation_lock_metadata",
    "validation_lock_path",
    "verify_deferred_test_import_receipt",
    "verify_upstream_import_receipt",
    "verify_validation_lock",
    "write_deferred_test_import_receipt",
    "write_upstream_import_receipt",
    "write_validation_lock",
]
