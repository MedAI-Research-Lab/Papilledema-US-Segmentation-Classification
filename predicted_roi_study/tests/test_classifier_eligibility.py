from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from predicted_roi_study.data import (
    ROICacheDataset,
    build_classifier_training_selection,
    dataframe_sha256,
)


def _eye(
    patient_id: str,
    case_id: str,
    side: str,
    label: int,
    valid_frames: int,
) -> list[dict]:
    rows: list[dict] = []
    for frame_index in range(7):
        valid = frame_index < valid_frames
        identity_sha256 = hashlib.sha256(
            f"{patient_id}|{case_id}|{side}|{frame_index}".encode("utf-8")
        ).hexdigest()
        rows.append(
            {
                "patient_id": patient_id,
                "case_id": case_id,
                "side": side,
                "frame_id": str(frame_index),
                "label": label,
                "label_3class": label,
                "frame_identity_sha256": identity_sha256,
                "roi_valid": bool(valid),
                "abstention_reason": "" if valid else "empty",
                "cache_path": f"{patient_id}_{case_id}_{frame_index}.npz" if valid else np.nan,
            }
        )
    return rows


def _selection(source: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return build_classifier_training_selection(
        source, frames_per_eye=7, minimum_valid_frames=4
    )


def test_three_of_seven_is_excluded_while_four_and_seven_are_included_without_mutation() -> None:
    source = pd.DataFrame(
        _eye("p0", "e0", "SAG", 0, 3)
        + _eye("p1", "e1", "SOL", 0, 4)
        + _eye("p2", "e2", "SAG", 1, 7)
    )
    before = source.copy(deep=True)
    before_sha = dataframe_sha256(source)

    optimization, ledger = _selection(source)

    pd.testing.assert_frame_equal(source, before)
    assert dataframe_sha256(source) == before_sha
    assert len(optimization) == 11
    assert set(optimization.case_id) == {"e1", "e2"}
    assert optimization.roi_valid.eq(True).all()
    assert len(ROICacheDataset(optimization)) == 11
    by_eye = ledger.set_index("case_id")
    assert not bool(by_eye.loc["e0", "eye_training_eligible"])
    assert bool(by_eye.loc["e1", "eye_training_eligible"])
    assert bool(by_eye.loc["e2", "eye_training_eligible"])
    assert by_eye.optimization_frame_count.to_dict() == {"e0": 0, "e1": 4, "e2": 7}
    assert by_eye.exclusion_reason.to_dict() == {
        "e0": "below_4_of_7",
        "e1": "eligible_ge_4_of_7",
        "e2": "eligible_ge_4_of_7",
    }
    assert source.loc[(source.case_id == "e0") & source.roi_valid, "cache_path"].notna().all()


def test_real_invalid_roi_with_cache_reference_remains_rejected() -> None:
    source = pd.DataFrame(_eye("p0", "e0", "SAG", 0, 7))
    source.loc[0, "roi_valid"] = False
    source.loc[0, "cache_path"] = "forbidden_invalid_cache.npz"
    with pytest.raises(ValueError, match="Only valid ROI rows"):
        ROICacheDataset(source)


@pytest.mark.parametrize("invalid_value", ["False", 0, 1, None])
def test_non_boolean_or_missing_roi_valid_is_rejected(invalid_value: object) -> None:
    source = pd.DataFrame(_eye("p0", "e0", "SAG", 0, 4))
    source["roi_valid"] = source["roi_valid"].astype(object)
    source.loc[0, "roi_valid"] = invalid_value
    with pytest.raises(ValueError, match="roi_valid"):
        _selection(source)


def test_exactly_seven_unique_frames_per_eye_is_required() -> None:
    missing = pd.DataFrame(_eye("p0", "e0", "SAG", 0, 4)[:-1])
    with pytest.raises(ValueError, match="exactly 7 unique frames"):
        _selection(missing)

    duplicate = pd.DataFrame(_eye("p0", "e0", "SAG", 0, 4))
    duplicate.loc[6, "frame_id"] = duplicate.loc[5, "frame_id"]
    with pytest.raises(ValueError, match="duplicate eye/frame"):
        _selection(duplicate)


def test_composite_eye_identity_prevents_case_id_collision() -> None:
    source = pd.DataFrame(
        _eye("p0", "shared_case", "SAG", 0, 3)
        + _eye("p1", "shared_case", "SOL", 1, 4)
    )
    optimization, ledger = _selection(source)
    assert len(ledger) == 2
    assert set(zip(optimization.patient_id, optimization.side, strict=True)) == {
        ("p1", "SOL")
    }
    assert len(optimization) == 4


def test_selection_hashes_are_independent_of_input_row_order() -> None:
    source = pd.DataFrame(
        _eye("p0", "e0", "SAG", 0, 4)
        + _eye("p1", "e1", "SOL", 1, 7)
    )
    first_optimization, first_ledger = _selection(source)
    second_optimization, second_ledger = _selection(
        source.sample(frac=1.0, random_state=31).reset_index(drop=True)
    )
    assert dataframe_sha256(first_optimization) == dataframe_sha256(second_optimization)
    assert dataframe_sha256(first_ledger) == dataframe_sha256(second_ledger)
