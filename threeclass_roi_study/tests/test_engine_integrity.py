from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from threeclass_roi_study.config import (
    canonical_sha256,
    config_without_runtime,
    load_config,
)
from threeclass_roi_study.engine import (
    _audit_locked_manifest,
    _calibrate_units,
    _read_roi_index,
    _write_or_verify_final_provenance,
    prepare,
)
from threeclass_roi_study.protocol import ProtocolGateError


def _temporary_output_config(tmp_path: Path) -> dict:
    cfg = copy.deepcopy(load_config())
    cfg["output"] = str(tmp_path / "threeclass_prepare_test")
    cfg["config_sha256"] = canonical_sha256(config_without_runtime(cfg))
    return cfg


def test_prepare_is_idempotent_and_attested_artifacts_are_not_overwritten(
    tmp_path: Path,
) -> None:
    cfg = _temporary_output_config(tmp_path)
    first = prepare(cfg)
    second = prepare(cfg)
    assert first["status"] == "complete"
    assert second["status"] == "already_complete"

    design = Path(cfg["output"]) / "provenance" / "study_design.json"
    design.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ProtocolGateError, match="artifact (size|hash) changed"):
        prepare(cfg)


def test_final_provenance_snapshot_includes_audit_and_is_self_attested(
    tmp_path: Path,
) -> None:
    cfg = _temporary_output_config(tmp_path)
    receipt_root = Path(cfg["output"]) / "state" / "receipts"
    receipt_root.mkdir(parents=True)
    audit_receipt = receipt_root / "audit.json"
    audit_receipt.write_text(
        '{"stage":"audit","model":null,"seed":null,'
        '"classifier_strategy":null}\n',
        encoding="utf-8",
    )
    snapshot = _write_or_verify_final_provenance(cfg)
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    assert payload["scope"] == "all_output_receipts_through_final_audit"
    assert any(row["stage"] == "audit" for row in payload["receipts"])
    final_receipt = receipt_root / "final-provenance.json"
    assert final_receipt.is_file()
    assert _write_or_verify_final_provenance(cfg) == snapshot

    audit_receipt.write_text('{"stage":"tampered"}\n', encoding="utf-8")
    with pytest.raises(ProtocolGateError, match="completed audit receipt"):
        _write_or_verify_final_provenance(cfg)


def test_locked_manifest_matches_patient_eye_frame_contract() -> None:
    audit = _audit_locked_manifest(load_config())
    assert audit["status"] == "passed"
    assert audit["patients"] == 91
    assert audit["eyes"] == 182
    assert audit["frames"] == 1274
    assert audit["patient_counts_by_class"] == {"0": 48, "1": 21, "2": 22}


def test_roi_index_reader_projects_out_legacy_binary_columns(tmp_path: Path) -> None:
    path = tmp_path / "index.csv"
    pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "case_id": "P1_SAG",
                "side": "SAG",
                "frame_id": "frame_0001",
                "label_3class": 0,
                "frame_identity_sha256": "a" * 64,
                "roi_valid": True,
                "abstention_reason": "",
                "cache_path": "cache.npz",
                "cache_sha256": "b" * 64,
                "binary_logit": 99.0,
                "probability_disease": 0.999,
                "binary_prediction": 1,
            }
        ]
    ).to_csv(path, index=False)
    table = _read_roi_index(path)
    assert "binary_logit" not in table
    assert "probability_disease" not in table
    assert "binary_prediction" not in table


def test_missing_temperature_fails_closed_instead_of_creating_abstentions() -> None:
    table = pd.DataFrame(
        {
            "evaluable": [True],
            "probability_0": [0.8],
            "probability_1": [0.1],
            "probability_2": [0.1],
        }
    )
    with pytest.raises(ProtocolGateError, match="Calibration is unavailable"):
        _calibrate_units(table, None)
