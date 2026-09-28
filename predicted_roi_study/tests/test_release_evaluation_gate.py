"""Synthetic regression coverage of the release-only gate path adaptation."""
import json
from pathlib import Path

import pytest

from predicted_roi_study import protocol, release_evaluate as release


@pytest.fixture
def environment(tmp_path, monkeypatch):
    root = tmp_path / "run"
    cfg = {"study_id": "synthetic", "output": str(root),
           "test_access": {"composite_model_seed_lock_count": 20}}
    monkeypatch.setattr(protocol, "output_root", lambda active: root)
    monkeypatch.setattr(protocol, "code_fingerprint", lambda: {"sha256": "a" * 64})
    monkeypatch.setattr(protocol, "_runtime_context", lambda active: {"code_sha256": "a" * 64, "config_sha256": "b" * 64})
    monkeypatch.setattr(protocol, "verify_stage_receipt", lambda active, stage: {"stage": stage})
    monkeypatch.setattr(protocol, "assert_all_primary_locks", lambda active: {f"unit_{n}": "c" * 64 for n in range(20)})
    return cfg, root


def test_alias_is_byte_identical_and_idempotent_with_original_gate_validation(environment):
    cfg, root = environment
    alias = release.create_gate_alias(cfg)
    canonical = root / "state/test_access_opened.json"
    assert alias == root / "protocol/test_access.json"
    assert alias.read_bytes() == canonical.read_bytes()
    assert len(json.loads(alias.read_text())["unit_receipt_chain_sha256"]) == 20
    first = alias.stat().st_mtime_ns
    assert release.create_gate_alias(cfg) == alias
    assert alias.stat().st_mtime_ns == first


def test_changed_code_or_config_is_rejected_without_rewriting_gate(environment, monkeypatch):
    cfg, root = environment
    alias = release.create_gate_alias(cfg)
    before = alias.read_bytes()
    monkeypatch.setattr(protocol, "_runtime_context", lambda active: {"code_sha256": "changed", "config_sha256": "b" * 64})
    with pytest.raises(protocol.ProtocolGateError, match="changed"):
        release.create_gate_alias(cfg)
    assert alias.read_bytes() == before
    assert (root / "state/test_access_opened.json").read_bytes() == before


def test_missing_locks_never_create_either_gate(environment, monkeypatch):
    cfg, root = environment
    def incomplete(active):
        raise protocol.ProtocolGateError("all 20 locks required")
    monkeypatch.setattr(protocol, "assert_all_primary_locks", incomplete)
    with pytest.raises(protocol.ProtocolGateError, match="20"):
        release.create_gate_alias(cfg)
    assert not (root / "state/test_access_opened.json").exists()
    assert not (root / "protocol/test_access.json").exists()


def test_changed_lock_chain_is_rejected(environment, monkeypatch):
    cfg, root = environment
    alias = release.create_gate_alias(cfg)
    before = alias.read_bytes()
    monkeypatch.setattr(protocol, "assert_all_primary_locks", lambda active: {"changed": "c" * 64})
    with pytest.raises(protocol.ProtocolGateError, match="changed"):
        release.create_gate_alias(cfg)
    assert alias.read_bytes() == before


def test_divergent_alias_is_never_overwritten(environment):
    cfg, root = environment
    alias = release.create_gate_alias(cfg)
    alias.write_text("divergent")
    canonical = (root / "state/test_access_opened.json").read_bytes()
    with pytest.raises(protocol.ProtocolGateError, match="divergent"):
        release.create_gate_alias(cfg)
    assert alias.read_text() == "divergent"
    assert (root / "state/test_access_opened.json").read_bytes() == canonical


def test_historical_archive_fingerprint_is_rejected(environment, monkeypatch):
    cfg, root = environment
    monkeypatch.setattr(protocol, "code_fingerprint", lambda: {"sha256": release.HISTORICAL_CODE_SHA256})
    with pytest.raises(protocol.ProtocolGateError, match="historical"):
        release.create_gate_alias(cfg)
    assert not root.exists()


def test_orphan_alias_cannot_open_gate(environment):
    cfg, root = environment
    alias = root / "protocol/test_access.json"
    alias.parent.mkdir(parents=True)
    alias.write_text("{}")
    with pytest.raises(protocol.ProtocolGateError, match="orphan"):
        release.create_gate_alias(cfg)
    assert not (root / "state/test_access_opened.json").exists()


def test_engine_binding_restored_after_error():
    from predicted_roi_study import engine
    original = engine.open_test_access
    with pytest.raises(RuntimeError, match="synthetic"):
        with release.gate_compatibility():
            assert engine.open_test_access is release.create_gate_alias
            raise RuntimeError("synthetic")
    assert engine.open_test_access is original


def test_execution_lock_rejects_concurrent_acquisition(tmp_path):
    path = tmp_path / "exclusive.lock"
    with release._exclusive_lock(path):
        with pytest.raises(protocol.ProtocolGateError, match="already owns"):
            with release._exclusive_lock(path):
                raise AssertionError("a second process-level lock must not be acquired")
