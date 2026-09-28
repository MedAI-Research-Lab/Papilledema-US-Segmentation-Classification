from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from predicted_roi_study.qualitative import (
    CANONICAL_FAILURE_STATUSES,
    OUTCOME_CONDITIONED_NOTE,
    build_common_selection_lock,
    overlay_masks,
    render_common_segmentation_gallery,
    render_failure_gallery,
    select_common_test_frames,
    select_failure_examples,
    verify_common_selection_lock,
    write_failure_selection_manifest,
)


MODELS = ("model_a", "model_b")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_rows(tmp_path: Path, *, frames_per_label: int = 3) -> pd.DataFrame:
    records = []
    index = 0
    for label in (0, 1, 2):
        for within in range(frames_per_label):
            image_path = tmp_path / f"image_{index}.png"
            mask_path = tmp_path / f"mask_{index}.png"
            image = np.zeros((12, 12, 3), dtype=np.uint8)
            image[..., 0] = 25 + 20 * label
            image[..., 1] = 10 * within
            mask = np.zeros((12, 12), dtype=np.uint8)
            mask[3:7, 4:8] = 255
            Image.fromarray(image).save(image_path)
            Image.fromarray(mask).save(mask_path)
            records.append(
                {
                    "patient_id": f"p{label}_{within}",
                    "case_id": f"c{label}_{within}",
                    "side": "SAG" if within % 2 == 0 else "SOL",
                    "frame_id": str(within),
                    "label_3class": label,
                    "outer_seed": 17,
                    "partition": "test",
                    "output_image": image_path.name,
                    "output_mask": mask_path.name,
                    "output_image_sha256": sha256(image_path),
                    "output_mask_sha256": sha256(mask_path),
                }
            )
            index += 1
    return pd.DataFrame(records)


def write_audits(tmp_path: Path, source: pd.DataFrame, model: str) -> pd.DataFrame:
    records = []
    statuses = list(CANONICAL_FAILURE_STATUSES)
    for index, row in source.reset_index(drop=True).iterrows():
        hard = np.zeros((12, 12), dtype=np.uint8)
        hard[2:6, 2 + index % 3:6 + index % 3] = 1
        selected = hard.copy()
        path = tmp_path / f"{model}_{index}.npz"
        np.savez_compressed(path, thresholded_mask=hard, selected_mask=selected)
        records.append(
            {
                "patient_id": row.patient_id,
                "case_id": row.case_id,
                "side": row.side,
                "frame_id": str(row.frame_id),
                "segmentation_abstention_reason": statuses[index % len(statuses)],
                "audit_path": str(path),
                "audit_sha256": sha256(path),
                "dice": 1.0 - index / 100.0,
            }
        )
    return pd.DataFrame(records)


def test_common_selection_is_model_blind_order_invariant_and_stratified(tmp_path: Path):
    source = source_rows(tmp_path)
    selected = select_common_test_frames(source, seed=17)
    shuffled = source.sample(frac=1.0, random_state=99).reset_index(drop=True)
    repeated = select_common_test_frames(shuffled, seed=17)

    pd.testing.assert_frame_equal(selected, repeated)
    assert selected.groupby("label_3class").size().to_dict() == {0: 1, 1: 1, 2: 1}
    assert selected.model_blind_selection.all()
    assert not any(
        forbidden in select_common_test_frames.__code__.co_varnames
        for forbidden in ("dice", "prediction", "confidence", "model_outputs")
    )


def test_common_selection_rejects_non_test_and_insufficient_class(tmp_path: Path):
    source = source_rows(tmp_path)
    source.loc[0, "partition"] = "validation"
    with pytest.raises(ValueError, match="non-test"):
        select_common_test_frames(source, seed=17)

    source = source_rows(tmp_path)
    with pytest.raises(ValueError, match="requested"):
        select_common_test_frames(source[source.label_3class != 2], seed=17)


def test_common_lock_is_immutable_and_detects_universe_or_payload_change(tmp_path: Path):
    source = source_rows(tmp_path)
    lock_path = tmp_path / "selection.json"
    lock = build_common_selection_lock({17: source}, lock_path, seeds=(17,))
    assert verify_common_selection_lock(lock_path, {17: source}) == lock
    assert lock["model_outputs_opened_during_selection"] is False
    assert "dice" in lock["forbidden_selection_inputs"]

    altered = source.copy()
    altered.loc[0, "label_3class"] = 1
    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        build_common_selection_lock({17: altered}, lock_path, seeds=(17,))
    with pytest.raises(RuntimeError, match="universe changed"):
        verify_common_selection_lock(lock_path, {17: altered})

    tampered = json.loads(lock_path.read_text(encoding="utf-8"))
    tampered["examples_per_label"] = 2
    with pytest.raises(RuntimeError, match="digest mismatch"):
        verify_common_selection_lock(tampered, {17: source})


def test_common_lock_enforces_different_frames_across_overlapping_seed_tests(tmp_path: Path):
    seed_17 = source_rows(tmp_path, frames_per_label=3)
    seed_42 = seed_17.copy()
    seed_42["outer_seed"] = 42
    lock = build_common_selection_lock(
        {17: seed_17, 42: seed_42},
        tmp_path / "two_seed_selection.json",
        seeds=(17, 42),
    )
    selected = pd.DataFrame(lock["selections"])
    keys = selected[["patient_id", "case_id", "side", "frame_id"]].astype(str)
    assert len(keys) == 6
    assert not keys.duplicated().any()
    assert selected.globally_unique_across_seeds.all()


def test_outer_seed_takes_precedence_over_legacy_unassigned_seed(tmp_path: Path):
    source = source_rows(tmp_path)
    source["seed"] = "unassigned"

    selected = select_common_test_frames(source, seed=17)

    assert len(selected) == 3
    assert set(selected["outer_seed"].astype(int)) == {17}


def test_failure_selection_is_hash_min_alias_aware_and_explicitly_illustrative(tmp_path: Path):
    source = source_rows(tmp_path, frames_per_label=2)
    frames = {model: write_audits(tmp_path, source, model) for model in MODELS}
    frames["model_a"].loc[frames["model_a"].segmentation_abstention_reason == "oversize", "segmentation_abstention_reason"] = "oversegmentation"
    frames["model_b"].loc[frames["model_b"].segmentation_abstention_reason == "border", "segmentation_abstention_reason"] = "edge_touch"
    first = select_failure_examples({17: source}, {17: frames}, models=MODELS)

    changed = {
        model: table.sample(frac=1.0, random_state=4).assign(dice=-100.0)
        for model, table in frames.items()
    }
    second = select_failure_examples({17: source}, {17: changed}, models=MODELS)
    columns = [
        "outer_seed", "model", "failure_status", "patient_id", "case_id",
        "side", "frame_id", "selection_digest",
    ]
    pd.testing.assert_frame_equal(first[columns], second[columns])
    assert len(first) == len(MODELS) * len(CANONICAL_FAILURE_STATUSES)
    assert first.outcome_conditioned_illustrative.all()
    assert set(first.interpretation_note) == {OUTCOME_CONDITIONED_NOTE}
    assert set(first.failure_status) == set(CANONICAL_FAILURE_STATUSES)

    manifest_path = tmp_path / "failures.json"
    manifest = write_failure_selection_manifest(
        manifest_path, first, source_rows_by_seed={17: source}
    )
    assert manifest["outcome_conditioned_illustrative"] is True
    assert "not a random" in manifest["interpretation_note"]


def test_failure_selection_records_unobserved_categories(tmp_path: Path):
    source = source_rows(tmp_path)
    frames = {model: write_audits(tmp_path, source, model) for model in MODELS}
    for table in frames.values():
        table["segmentation_abstention_reason"] = "empty"
    selection = select_failure_examples({17: source}, {17: frames}, models=MODELS)
    unavailable = selection[selection.failure_status != "empty"]
    assert not unavailable.available.any()
    assert unavailable.patient_id.isna().all()


def test_render_common_and_failure_galleries_from_audit_masks(tmp_path: Path):
    source = source_rows(tmp_path, frames_per_label=2)
    frames = {model: write_audits(tmp_path, source, model) for model in MODELS}
    lock_path = tmp_path / "common_selection.json"
    build_common_selection_lock({17: source}, lock_path, seeds=(17,))

    common = render_common_segmentation_gallery(
        lock_path,
        {17: source},
        {17: frames},
        dataset_root=tmp_path,
        output_dir=tmp_path / "common",
        models=MODELS,
        tile_size=(64, 64),
    )
    assert len(common.figures) == 3
    assert all(path.is_file() for path in common.figures)
    assert len(pd.read_csv(common.index_path)) == 3
    metadata = json.loads(common.metadata_path.read_text(encoding="utf-8"))
    assert metadata["model_outputs_used_for_selection"] is False
    with Image.open(common.figures[0]) as panel:
        assert panel.size == (4 * 64, 64 + 44)

    failures = select_failure_examples({17: source}, {17: frames}, models=MODELS)
    failure_gallery = render_failure_gallery(
        {17: source},
        {17: frames},
        failures,
        dataset_root=tmp_path,
        output_dir=tmp_path / "failures",
        models=MODELS,
        tile_size=(64, 64),
    )
    assert len(failure_gallery.figures) == 1
    with Image.open(failure_gallery.figures[0]) as panel:
        assert panel.size == (len(MODELS) * 64, len(CANONICAL_FAILURE_STATUSES) * (64 + 44))
    failure_metadata = json.loads(failure_gallery.metadata_path.read_text(encoding="utf-8"))
    assert failure_metadata["outcome_conditioned_illustrative"] is True


def test_overlay_uses_reference_green_and_prediction_orange_boundaries():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    reference = np.zeros((8, 8), dtype=bool)
    prediction = np.zeros((8, 8), dtype=bool)
    reference[1:4, 1:4] = True
    prediction[4:7, 4:7] = True
    result = np.asarray(
        overlay_masks(image, reference_mask=reference, predicted_mask=prediction)
    )
    assert tuple(result[1, 1]) == (40, 210, 70)
    assert tuple(result[4, 4]) == (255, 145, 30)
