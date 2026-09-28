from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from predicted_roi_study import protocol
from predicted_roi_study.ablations import (
    ANALYTICAL_ARMS,
    EXPECTED_ARMS,
    MODEL_SEED_ARMS,
    SHARED_SEED_ARMS,
    AblationInputBuilder,
    AblationPlan,
    AblationScope,
    AblationSpec,
    AblationTask,
    VerifiedAblationTestGate,
    aggregate_frames_by_rule,
    build_ablation_plan,
    build_ablation_test_schedule,
    build_ablation_validation_schedule,
    build_analytical_eye_variants,
    collect_ablation_validation_locks,
    open_ablation_test_gate,
    verify_ablation_test_gate,
    write_ablation_cache,
    write_ablation_validation_lock,
)
from predicted_roi_study.config import load_config
from predicted_roi_study.roi import DEFAULT_NEUTRAL_RGB, ROIPolicy


DIGEST = "1" * 64
OTHER_DIGEST = "2" * 64


def locked_levels(eye=(0.4, 1.1), patient=(0.45, 0.9)):
    return {
        "eye": {"status": "locked", "threshold": eye[0], "temperature": eye[1]},
        "patient": {"status": "locked", "threshold": patient[0], "temperature": patient[1]},
    }


def roi_policy(shape=(12, 12), *, maximum=20):
    return ROIPolicy(
        min_area_pixels=4,
        max_area_pixels=maximum,
        image_shape=shape,
        threshold=0.5,
        dominance_ratio=1.5,
        reject_border=True,
        max_hole_area_pixels=0,
    )


def probability(shape=(12, 12), mask=None, score=0.9):
    output = np.zeros(shape, dtype=np.float32)
    if mask is not None:
        output[np.asarray(mask, dtype=bool)] = score
    return output


def one_task_plan(arm, scope=AblationScope.MODEL_SEED, *, variant=None, fit=True):
    spec = AblationSpec(arm, scope, None, (variant,) if variant is not None else (), fit)
    task = AblationTask(
        arm, scope, 17, model=None if scope is AblationScope.SEED_SHARED else "model_a",
        variant=variant, requires_classifier_fit=fit,
    )
    return AblationPlan((spec,), (task,), ("model_a",), (17,), 40000), task


def valid_gate(plan):
    return VerifiedAblationTestGate(
        path=Path("synthetic_gate.json"),
        sha256=DIGEST,
        plan_sha256=plan.sha256,
        config_sha256=DIGEST,
        primary_gate_sha256=OTHER_DIGEST,
        validation_lock_sha256={},
    )


def test_plan_covers_all_eleven_arms_with_model_specific_learned_fits():
    plan = build_ablation_plan(load_config())
    assert tuple(spec.name for spec in plan.specs) == EXPECTED_ARMS
    assert set(SHARED_SEED_ARMS + MODEL_SEED_ARMS + ANALYTICAL_ARMS) == set(EXPECTED_ARMS)
    assert len({task.key for task in plan.tasks}) == len(plan.tasks)
    assert plan.summary() == {
        "arm_count": 11,
        "task_count": 320,
        "classifier_fit_count": 180,
        "validation_lock_count": 320,
        "shared_seed_fit_count": 0,
        "model_seed_fit_count": 180,
        "analytical_task_count": 140,
        "plan_sha256": plan.sha256,
    }
    for arm in ("whole_image_baseline", "gt_roi_oracle"):
        tasks = [task for task in plan.tasks if task.arm == arm]
        assert len(tasks) == 20
        assert all(task.model is not None for task in tasks)
    validation_schedule = build_ablation_validation_schedule(plan)
    test_schedule = build_ablation_test_schedule(plan, valid_gate(plan))
    assert len(validation_schedule) == 40
    assert len(test_schedule) == 20
    assert sum(len(batch.tasks) for batch in validation_schedule) == 320
    assert sum(len(batch.tasks) for batch in test_schedule) == 320


def test_test_partition_builder_requires_matching_global_gate():
    plan, task = one_task_plan("whole_image_baseline", AblationScope.SEED_SHARED)
    with pytest.raises(RuntimeError, match="global test gate"):
        AblationInputBuilder(plan, task, "test")
    wrong_plan, _ = one_task_plan("gt_roi_oracle", AblationScope.SEED_SHARED)
    with pytest.raises(RuntimeError, match="global test gate"):
        AblationInputBuilder(plan, task, "test", test_gate=valid_gate(wrong_plan))
    AblationInputBuilder(plan, task, "test", test_gate=valid_gate(plan))


def test_gt_oracle_has_no_outside_roi_leakage():
    plan, task = one_task_plan("gt_roi_oracle", AblationScope.SEED_SHARED)
    builder = AblationInputBuilder(plan, task, "validation", target_size=20)
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[3:7, 4:8] = 1
    first = torch.rand(3, 12, 12)
    second = torch.rand(3, 12, 12)
    selected = torch.as_tensor(mask, dtype=torch.bool)
    second[:, selected] = first[:, selected]
    a = builder.build(first, sample_key="case/frame", gt_mask=mask)
    b = builder.build(second, sample_key="case/frame", gt_mask=mask)
    assert a.valid and b.valid
    assert torch.equal(a.tensor, b.tensor)
    assert a.metadata["non_deployable_oracle"]


def test_bbox_context_exposes_context_but_strict_appearance_does_not():
    shape = (12, 12)
    mask = np.zeros(shape, dtype=bool)
    mask[3:7, 3] = True
    mask[6, 3:7] = True
    score = probability(shape, mask)
    first = torch.zeros(3, *shape)
    second = first.clone()
    second[:, 3, 6] = 1.0  # inside bbox, outside the L-shaped predicted ROI

    bbox_plan, bbox_task = one_task_plan("bbox_context")
    bbox_builder = AblationInputBuilder(
        bbox_plan, bbox_task, "validation", roi_policy=roi_policy(shape), target_size=24
    )
    strict_plan, strict_task = one_task_plan("appearance_plus_geometry")
    strict_builder = AblationInputBuilder(
        strict_plan, strict_task, "validation", roi_policy=roi_policy(shape), target_size=24
    )
    bbox_a = bbox_builder.build(first, sample_key="x", predicted_probability=score)
    bbox_b = bbox_builder.build(second, sample_key="x", predicted_probability=score)
    strict_a = strict_builder.build(first, sample_key="x", predicted_probability=score)
    strict_b = strict_builder.build(second, sample_key="x", predicted_probability=score)
    assert not torch.equal(bbox_a.tensor, bbox_b.tensor)
    assert torch.equal(strict_a.tensor, strict_b.tensor)


def test_context_and_background_classifier_masks_survive_post_cache_augmentation_gate():
    """The cache mask denotes visible pixels, not merely the selected ROI."""

    shape = (16, 16)
    selected = np.zeros(shape, dtype=bool)
    selected[5:9, 5] = True
    selected[8, 5:10] = True
    score = probability(shape, selected)
    image = torch.rand(3, *shape)
    neutral = torch.tensor(DEFAULT_NEUTRAL_RGB).view(3, 1, 1)

    bbox_plan, bbox_task = one_task_plan("bbox_context")
    bbox = AblationInputBuilder(
        bbox_plan, bbox_task, "train", roi_policy=roi_policy(shape, maximum=30), target_size=24
    ).build(image, sample_key="frame", predicted_probability=score)
    # The full rectangular crop must remain visible after the same hard gate
    # ROICacheDataset applies following affine/brightness augmentation.
    assert bbox.classifier_mask.sum() > int(selected.sum())
    augmented_bbox = torch.ones_like(bbox.tensor)
    gated_bbox = torch.where(bbox.classifier_mask.unsqueeze(0), augmented_bbox, neutral)
    assert bool((gated_bbox == 1).any())

    background_plan, background_task = one_task_plan("background_only_negative_control")
    background = AblationInputBuilder(
        background_plan, background_task, "train",
        roi_policy=roi_policy(shape, maximum=30), target_size=24,
    ).build(image, sample_key="frame", predicted_probability=score)
    assert background.classifier_mask.sum() > int(selected.sum())
    assert bool((~background.classifier_mask).any())
    gated_background = torch.where(
        background.classifier_mask.unsqueeze(0), background.tensor, neutral
    )
    assert torch.equal(gated_background, background.tensor)


def test_locked_hard_rasters_reproduce_probability_map_inputs_without_float_map_storage():
    shape = (16, 16)
    selected = np.zeros(shape, dtype=bool)
    selected[5:9, 6:11] = True
    score = probability(shape, selected, score=0.8)
    image = torch.rand(3, *shape)
    plan, task = one_task_plan("appearance_plus_geometry")
    builder = AblationInputBuilder(
        plan, task, "validation", roi_policy=roi_policy(shape, maximum=30), target_size=24
    )
    from_probability = builder.build(
        image, sample_key="frame", predicted_probability=score
    )
    from_locked_rasters = builder.build(
        image,
        sample_key="frame",
        predicted_hard_mask=selected,
        selected_mask=selected,
        predicted_status="valid",
        roi_confidence=0.8,
    )
    assert torch.equal(from_probability.tensor, from_locked_rasters.tensor)
    assert torch.equal(from_probability.classifier_mask, from_locked_rasters.classifier_mask)
    assert torch.equal(from_probability.geometry, from_locked_rasters.geometry)
    assert from_locked_rasters.roi_confidence == pytest.approx(0.8)


def test_largest_component_arm_accepts_prediction_rejected_by_quality_gate():
    shape = (12, 12)
    mask = np.zeros(shape, dtype=bool)
    mask[0:6, 0:6] = True  # border touching and over maximum
    score = probability(shape, mask)
    image = torch.rand(3, *shape)
    loose_plan, loose_task = one_task_plan("largest_component_without_quality_gates")
    loose = AblationInputBuilder(
        loose_plan, loose_task, "validation", roi_policy=roi_policy(shape), target_size=16
    ).build(image, sample_key="x", predicted_probability=score)
    strict_plan, strict_task = one_task_plan("appearance_plus_geometry")
    strict = AblationInputBuilder(
        strict_plan, strict_task, "validation", roi_policy=roi_policy(shape), target_size=16
    ).build(image, sample_key="x", predicted_probability=score)
    assert loose.valid
    assert not strict.valid
    assert strict.status == "oversize"


def test_mask_geometry_background_and_random_controls_are_deterministic_and_separated():
    shape = (20, 20)
    mask = np.zeros(shape, dtype=bool)
    mask[7:11, 8:13] = True
    score = probability(shape, mask)
    first = torch.rand(3, *shape)
    second = first.clone()
    second[:, torch.as_tensor(mask)] = 1.0 - first[:, torch.as_tensor(mask)]

    outputs = {}
    for arm in (
        "geometry_only", "mask_only", "background_only_negative_control",
        "random_roi_negative_control",
    ):
        plan, task = one_task_plan(arm)
        builder = AblationInputBuilder(
            plan, task, "validation", roi_policy=roi_policy(shape, maximum=30), target_size=24
        )
        outputs[arm] = (
            builder.build(first, sample_key="stable-key", predicted_probability=score),
            builder.build(second, sample_key="stable-key", predicted_probability=score),
        )
    assert torch.equal(outputs["geometry_only"][0].tensor, outputs["geometry_only"][1].tensor)
    neutral = torch.tensor(DEFAULT_NEUTRAL_RGB).view(3, 1, 1).expand(3, 24, 24)
    assert torch.allclose(outputs["geometry_only"][0].tensor, neutral)
    assert torch.equal(outputs["mask_only"][0].tensor, outputs["mask_only"][1].tensor)
    assert torch.equal(
        outputs["background_only_negative_control"][0].tensor,
        outputs["background_only_negative_control"][1].tensor,
    )
    random_a, random_b = outputs["random_roi_negative_control"]
    assert np.array_equal(random_a.source_mask, random_b.source_mask)
    assert int(random_a.source_mask.sum()) == int(mask.sum())
    assert random_a.metadata["source_overlap_pixels"] == 0


def test_cache_is_byte_deterministic_and_refuses_different_overwrite(tmp_path):
    plan, task = one_task_plan("whole_image_baseline", AblationScope.SEED_SHARED)
    item = AblationInputBuilder(plan, task, "train", target_size=16).build(
        torch.zeros(3, 12, 12), sample_key="frame-1"
    )
    first = write_ablation_cache(tmp_path, task, "train", "frame-1", item)
    second = write_ablation_cache(tmp_path, task, "train", "frame-1", item)
    assert first["cache_sha256"] == second["cache_sha256"]
    assert first["cache_path"] == second["cache_path"]
    changed = AblationInputBuilder(plan, task, "train", target_size=16).build(
        torch.ones(3, 12, 12), sample_key="frame-1"
    )
    with pytest.raises(RuntimeError, match="overwrite"):
        write_ablation_cache(tmp_path, task, "train", "frame-1", changed)
    with np.load(first["cache_path"], allow_pickle=False) as cached:
        assert cached["image"].shape == (3, 16, 16)
        assert cached["mask"].shape == (16, 16)
        assert json.loads(str(cached["metadata_json"]))["status"] == "valid"


def test_invalid_input_has_no_cache_path(tmp_path):
    plan, task = one_task_plan("appearance_plus_geometry")
    builder = AblationInputBuilder(
        plan, task, "train", roi_policy=roi_policy(), target_size=16
    )
    item = builder.build(
        torch.zeros(3, 12, 12), sample_key="empty", predicted_probability=np.zeros((12, 12))
    )
    record = write_ablation_cache(tmp_path, task, "train", "empty", item)
    assert not item.valid
    assert record["cache_path"] is None
    assert not list(tmp_path.rglob("*.npz"))


def test_analytical_aggregation_variants_and_minimum_valid_frames():
    frame = pd.DataFrame(
        {
            "patient_id": ["p"] * 7,
            "case_id": ["eye"] * 7,
            "side": ["SAG"] * 7,
            "frame_id": [str(index) for index in range(7)],
            "label": [1] * 7,
            "probability": [0.1, 0.2, 0.8, 0.9, np.nan, np.nan, np.nan],
            "roi_valid": [True, True, True, True, False, False, False],
            "roi_confidence": [0.1, 0.2, 0.3, 0.4, np.nan, np.nan, np.nan],
            "roi_hit": [1, 1, 1, 1, 0, 0, 0],
        }
    )
    expected = {
        "mean": 0.5,
        "median": 0.5,
        "maximum": 0.9,
        "predicted_mask_confidence_weighted_mean": 0.65,
    }
    for rule, value in expected.items():
        eye = aggregate_frames_by_rule(frame, rule=rule, min_valid_frames=4).iloc[0]
        assert eye.evaluable
        assert eye.probability == pytest.approx(value)
        assert eye.localized
    assert not aggregate_frames_by_rule(frame, rule="mean", min_valid_frames=7).iloc[0].evaluable

    variants = build_analytical_eye_variants(frame, load_config())
    assert set(variants) == {
        "aggregation_rule/mean",
        "aggregation_rule/median",
        "aggregation_rule/maximum",
        "aggregation_rule/predicted_mask_confidence_weighted_mean",
        "minimum_valid_frames/1",
        "minimum_valid_frames/4",
        "minimum_valid_frames/7",
    }
    # Localized-success remains the locked four-hit definition even when one
    # valid frame would make the diagnostic arm evaluable.
    assert variants["minimum_valid_frames/1"].iloc[0].localized


def test_every_task_must_lock_before_one_global_ablation_test_gate(tmp_path, monkeypatch):
    fitted_plan, fitted = one_task_plan("whole_image_baseline", AblationScope.SEED_SHARED)
    analytical_spec = AblationSpec(
        "minimum_valid_frames", AblationScope.ANALYTICAL_MODEL_SEED, None, (4,), False
    )
    analytical = AblationTask(
        "minimum_valid_frames", AblationScope.ANALYTICAL_MODEL_SEED, 17,
        model="model_a", variant=4, requires_classifier_fit=False,
    )
    plan = AblationPlan(
        fitted_plan.specs + (analytical_spec,), (fitted, analytical), ("model_a",), (17,), 40000
    )
    locks = tmp_path / "locks"
    with pytest.raises(RuntimeError, match="missing"):
        collect_ablation_validation_locks(plan, locks, config_sha256=DIGEST)
    write_ablation_validation_lock(
        locks, fitted, plan_sha256=plan.sha256, config_sha256=DIGEST,
        validation_source_sha256=DIGEST, checkpoint_sha256=OTHER_DIGEST,
        level_locks=locked_levels(),
    )
    write_ablation_validation_lock(
        locks, analytical, plan_sha256=plan.sha256, config_sha256=DIGEST,
        validation_source_sha256=OTHER_DIGEST, level_locks=locked_levels(),
    )
    hashes = collect_ablation_validation_locks(plan, locks, config_sha256=DIGEST)
    assert set(hashes) == {fitted.key, analytical.key}

    primary_gate = tmp_path / "primary_gate.json"
    primary_gate.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(protocol, "assert_all_primary_evaluations", lambda cfg: None)
    monkeypatch.setattr(protocol, "open_ablation_test_access", lambda cfg: primary_gate)
    monkeypatch.setattr(protocol, "output_root", lambda cfg: tmp_path)
    cfg = {"_runtime": {"canonical_config_sha256": DIGEST}}
    gate = open_ablation_test_gate(cfg, plan=plan, lock_root=locks)
    assert gate.plan_sha256 == plan.sha256
    verified = verify_ablation_test_gate(
        gate.path, plan, locks, config_sha256=DIGEST, primary_gate_path=primary_gate
    )
    assert verified.sha256 == gate.sha256


def test_analytical_lock_cannot_claim_a_classifier_checkpoint(tmp_path):
    plan, task = one_task_plan(
        "minimum_valid_frames", AblationScope.ANALYTICAL_MODEL_SEED, variant=4, fit=False
    )
    with pytest.raises(ValueError, match="cannot have a checkpoint"):
        write_ablation_validation_lock(
            tmp_path, task, plan_sha256=plan.sha256, config_sha256=DIGEST,
            validation_source_sha256=DIGEST, checkpoint_sha256=OTHER_DIGEST,
            level_locks=locked_levels(),
        )


def test_threshold_can_lock_when_temperature_calibration_is_unavailable(tmp_path):
    plan, task = one_task_plan("whole_image_baseline", AblationScope.SEED_SHARED)
    path = write_ablation_validation_lock(
        tmp_path,
        task,
        plan_sha256=plan.sha256,
        config_sha256=DIGEST,
        validation_source_sha256=DIGEST,
        checkpoint_sha256=OTHER_DIGEST,
        level_locks={
            "eye": {
                "status": "calibration_unavailable",
                "classification_threshold_status": "locked",
                "calibration_status": "unavailable",
                "threshold": 0.41,
                "temperature": None,
                "unavailable_reason": "optimizer_failed",
            },
            "patient": {"status": "locked", "threshold": 0.5, "temperature": 1.2},
        },
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["level_locks"]["eye"]["classification_threshold_status"] == "locked"
    assert value["level_locks"]["eye"]["temperature"] is None


def test_completed_ablation_result_rejects_mutated_summary_dependency(tmp_path):
    from predicted_roi_study import engine

    plan, task = one_task_plan("whole_image_baseline", AblationScope.SEED_SHARED)
    destination = tmp_path / "test" / "task"
    destination.mkdir(parents=True)
    for name in (*engine._ABLATION_EXPORT_FILENAMES, "test_index.csv"):
        (destination / name).write_text(f"original:{name}\n", encoding="utf-8")
    records = engine._ablation_test_artifact_records(destination, task)
    metrics_path = destination / "metrics.json"
    result_path = destination / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "schema": 1,
                "status": "complete",
                "task": task.to_dict(),
                "ablation_test_gate_sha256": DIGEST,
                "validation_lock_sha256": OTHER_DIGEST,
                "metrics_path": str(metrics_path.resolve()),
                "metrics_sha256": engine.sha256_file(metrics_path),
                "task_artifacts": records,
            }
        ),
        encoding="utf-8",
    )
    assert engine._completed_ablation_test_result(
        result_path, task, DIGEST, OTHER_DIGEST
    ) is not None

    # Paired contrasts read this table directly.  A partial-run resume must
    # not accept the task after those bytes change while metrics.json remains.
    (destination / "eyes.csv").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(engine.ProtocolGateError, match="artifact changed"):
        engine._completed_ablation_test_result(
            result_path, task, DIGEST, OTHER_DIGEST
        )


def test_lock_ablations_orchestrates_180_fits_and_140_analytical_locks_without_test(
    tmp_path, monkeypatch
):
    from predicted_roi_study import engine

    cfg = load_config()
    fitted, analytical = [], []
    receipt_metadata = {}

    monkeypatch.setattr(engine, "assert_prerequisites", lambda *args, **kwargs: None)

    def absent(*args, **kwargs):
        raise engine.ProtocolGateError("not complete")

    monkeypatch.setattr(engine, "verify_stage_receipt", absent)
    monkeypatch.setattr(engine, "output_root", lambda value: tmp_path)
    monkeypatch.setattr(engine, "_relative", lambda path: str(Path(path).resolve()))
    monkeypatch.setattr(
        engine,
        "split_frames",
        lambda *args, **kwargs: pytest.fail("mock orchestration must not decode any partition"),
    )

    def make_lock(plan, task, root):
        return write_ablation_validation_lock(
            root / "validation_locks",
            task,
            plan_sha256=plan.sha256,
            config_sha256=cfg["_runtime"]["canonical_config_sha256"],
            validation_source_sha256=DIGEST,
            checkpoint_sha256=OTHER_DIGEST if task.requires_classifier_fit else None,
            level_locks=locked_levels(),
        )

    def fake_fit(value, plan, task, root):
        fitted.append(task.key)
        return make_lock(plan, task, root)

    def fake_analytical(value, plan, tasks, root):
        paths = []
        for task in tasks:
            analytical.append(task.key)
            paths.append(make_lock(plan, task, root))
        return paths

    monkeypatch.setattr(engine, "_fit_and_lock_ablation_task", fake_fit)
    monkeypatch.setattr(engine, "_lock_analytical_ablation_batch", fake_analytical)

    def fake_verify_bundle(root, task, lock):
        path = tmp_path.joinpath("synthetic_bundles", *task.key.split("/"), "result.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
        return path.resolve(), [path.resolve()]

    monkeypatch.setattr(engine, "_verify_ablation_validation_bundle", fake_verify_bundle)

    def fake_receipt(value, stage, *, artifacts, metadata, **kwargs):
        assert stage == "lock-ablations"
        receipt_metadata.update(metadata)
        path = tmp_path / "lock-ablations-receipt.json"
        path.write_text("{}", encoding="utf-8")
        return path

    monkeypatch.setattr(engine, "write_stage_receipt", fake_receipt)
    result = engine.lock_ablations(cfg)
    assert result["status"] == "complete"
    assert len(fitted) == 180
    assert len(analytical) == 140
    assert len(set(fitted + analytical)) == 320
    assert receipt_metadata["ablation_validation_lock_count"] == 320
    assert receipt_metadata["test_predictions_generated"] is False


def test_evaluate_ablations_opens_global_gate_before_all_320_synthetic_results(
    tmp_path, monkeypatch
):
    from predicted_roi_study import engine

    cfg = load_config()
    state = {"gate_open": False, "fit": 0, "analytical": 0}
    receipt_metadata = {}
    monkeypatch.setattr(engine, "assert_prerequisites", lambda *args, **kwargs: None)

    def absent(*args, **kwargs):
        raise engine.ProtocolGateError("not complete")

    monkeypatch.setattr(engine, "verify_stage_receipt", absent)
    monkeypatch.setattr(engine, "output_root", lambda value: tmp_path)
    monkeypatch.setattr(engine, "_relative", lambda path: str(Path(path).resolve()))

    def open_gate(value, *, plan, lock_root):
        state["gate_open"] = True
        return valid_gate(plan)

    monkeypatch.setattr(engine, "open_ablation_test_gate", open_gate)

    def synthetic_result(task, root):
        assert state["gate_open"]
        directory = root / "synthetic" / task.key.replace("/", "_")
        directory.mkdir(parents=True, exist_ok=True)
        metrics = directory / "metrics.json"
        metrics.write_text("{}", encoding="utf-8")
        result = directory / "result.json"
        result.write_text(
            json.dumps({"task": task.to_dict(), "metrics_path": str(metrics.resolve())}),
            encoding="utf-8",
        )
        return result

    def fake_fit(value, plan, task, root, gate):
        state["fit"] += 1
        return synthetic_result(task, root)

    def fake_analytical(value, plan, task, root, gate, primary_frames, variants):
        state["analytical"] += 1
        return synthetic_result(task, root)

    monkeypatch.setattr(engine, "_evaluate_fitted_ablation_task", fake_fit)
    monkeypatch.setattr(engine, "_evaluate_analytical_ablation_task", fake_analytical)
    monkeypatch.setattr(
        engine,
        "_verified_ablation_test_artifacts",
        lambda path, task, value=None: [Path(json.loads(Path(path).read_text())["metrics_path"])],
    )

    def guarded_read_csv(*args, **kwargs):
        assert state["gate_open"]
        return pd.DataFrame({"frame_id": pd.Series(dtype=str)})

    monkeypatch.setattr(engine.pd, "read_csv", guarded_read_csv)
    monkeypatch.setattr(engine, "build_analytical_eye_variants", lambda *args, **kwargs: {})

    def fake_summary(value, plan, root, paths):
        outputs = []
        for name in ("per_seed.csv", "mean_sd.csv", "interpretation.json"):
            path = root / "summary" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")
            outputs.append(path)
        return outputs

    monkeypatch.setattr(engine, "_ablation_summary", fake_summary)
    lock_receipt = tmp_path / "lock-receipt.json"
    lock_receipt.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(protocol, "stage_receipt_path", lambda *args, **kwargs: lock_receipt)

    def fake_receipt(value, stage, *, artifacts, metadata, **kwargs):
        assert stage == "evaluate-ablations"
        receipt_metadata.update(metadata)
        path = tmp_path / "evaluate-receipt.json"
        path.write_text("{}", encoding="utf-8")
        return path

    monkeypatch.setattr(engine, "write_stage_receipt", fake_receipt)
    result = engine.evaluate_ablations(cfg)
    assert result["status"] == "complete"
    assert state["fit"] == 180
    assert state["analytical"] == 140
    assert receipt_metadata["ablation_test_evaluation_count"] == 320
    assert receipt_metadata["all_ablation_test_evaluations_complete"] is True
