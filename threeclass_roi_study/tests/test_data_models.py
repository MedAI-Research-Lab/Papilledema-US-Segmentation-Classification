from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from predicted_roi_study.models import (
    build_strict_roi_classifier as build_legacy_binary_classifier,
)
from threeclass_roi_study.data import (
    ThreeClassROICacheDataset,
    audit_frozen_upstream_artifacts,
    discover_frozen_upstream_artifacts,
    sha256_file,
    threeclass_classifier_loss_weights,
)
from threeclass_roi_study.models import (
    THREE_CLASS_FAMILIES,
    build_roi_classifier,
    build_threeclass_roi_classifier,
)


def _identity(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_roi_cache(path: Path, identity: str, *, size: int = 16) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((3, size, size), 0.485, dtype=np.float32)
    image[:, 3:-3, 4:-4] = 0.7
    mask = np.zeros((size, size), dtype=bool)
    mask[3:-3, 4:-4] = True
    np.savez_compressed(
        path,
        frame_identity_sha256=np.frombuffer(bytes.fromhex(identity), dtype=np.uint8),
        image=image,
        mask=mask,
        geometry=np.asarray([0.25, 0.5, 0.625, -0.1], dtype=np.float32),
    )


def test_roi_dataset_uses_threeclass_label_and_preserves_row_index(tmp_path: Path) -> None:
    valid_identity = _identity("valid")
    invalid_identity = _identity("invalid")
    cache = tmp_path / "roi.npz"
    _write_roi_cache(cache, valid_identity)
    frame = pd.DataFrame(
        [
            {
                "patient_id": "P2",
                "case_id": "P2_SAG",
                "side": "SAG",
                "frame_id": "frame_0001",
                "label": 1,
                "label_3class": 2,
                "frame_identity_sha256": valid_identity,
                "roi_valid": True,
                "abstention_reason": "",
                "cache_path": str(cache),
                "cache_sha256": sha256_file(cache),
            },
            {
                "patient_id": "P3",
                "case_id": "P3_SOL",
                "side": "SOL",
                "frame_id": "frame_0002",
                "label": 1,
                "label_3class": 1,
                "frame_identity_sha256": invalid_identity,
                "roi_valid": False,
                "abstention_reason": "empty",
                "cache_path": np.nan,
                "cache_sha256": np.nan,
            },
        ]
    )

    dataset = ThreeClassROICacheDataset(
        frame, project_root=tmp_path, include_geometry=True
    )
    sample = dataset[0]

    assert len(dataset) == 1
    assert sample["label"].item() == 2
    assert sample["label_3class"].item() == 2
    assert sample["index"].item() == 0
    assert sample["image"].shape == (3, 16, 16)
    assert sample["roi_mask"].dtype == torch.bool
    assert sample["geometry"].shape == (4,)


def test_roi_dataset_rejects_non_threeclass_target(tmp_path: Path) -> None:
    identity = _identity("bad-label")
    cache = tmp_path / "roi.npz"
    _write_roi_cache(cache, identity)
    frame = pd.DataFrame(
        [
            {
                "patient_id": "P",
                "case_id": "P_SAG",
                "side": "SAG",
                "frame_id": "frame_0001",
                "label_3class": 3,
                "frame_identity_sha256": identity,
                "roi_valid": True,
                "abstention_reason": "",
                "cache_path": str(cache),
                "cache_sha256": sha256_file(cache),
            }
        ]
    )
    with pytest.raises(ValueError, match="only"):
        ThreeClassROICacheDataset(frame, project_root=tmp_path)


def test_roi_dataset_rejects_cache_bytes_changed_after_index_lock(tmp_path: Path) -> None:
    identity = _identity("tampered-cache")
    cache = tmp_path / "roi.npz"
    _write_roi_cache(cache, identity)
    locked_digest = sha256_file(cache)
    frame = pd.DataFrame(
        [
            {
                "patient_id": "P",
                "case_id": "P_SAG",
                "side": "SAG",
                "frame_id": "frame_0001",
                "label_3class": 0,
                "frame_identity_sha256": identity,
                "roi_valid": True,
                "abstention_reason": "",
                "cache_path": str(cache),
                "cache_sha256": locked_digest,
            }
        ]
    )
    dataset = ThreeClassROICacheDataset(frame, project_root=tmp_path)
    cache.write_bytes(b"tampered-after-lock")
    with pytest.raises(ValueError, match="cache SHA-256 mismatch"):
        _ = dataset[0]


def test_loss_weights_balance_class_then_patient_then_eye() -> None:
    rows: list[dict[str, object]] = []

    def add(patient: str, label: int, side: str, frames: int) -> None:
        for frame in range(frames):
            rows.append(
                {
                    "patient_id": patient,
                    "case_id": f"{patient}_{side}",
                    "side": side,
                    "frame_id": f"frame_{frame:04d}",
                    "label_3class": label,
                    "roi_valid": True,
                }
            )

    add("C0", 0, "SAG", 1)
    add("P1", 1, "SAG", 1)
    add("P2", 1, "SAG", 4)
    add("D2", 2, "SAG", 2)
    add("D2", 2, "SOL", 5)
    frame = pd.DataFrame(rows).sort_values(
        ["patient_id", "case_id", "side", "frame_id"], kind="stable"
    ).reset_index(drop=True)
    weights = threeclass_classifier_loss_weights(frame)
    weighted = frame.assign(weight=weights)

    class_mass = weighted.groupby("label_3class").weight.sum().to_numpy()
    np.testing.assert_allclose(class_mass, np.repeat(class_mass[0], 3))
    class_one_patient_mass = (
        weighted.loc[weighted.label_3class == 1]
        .groupby("patient_id")
        .weight.sum()
        .to_numpy()
    )
    np.testing.assert_allclose(
        class_one_patient_mass, np.repeat(class_one_patient_mass[0], 2)
    )
    class_two_eye_mass = (
        weighted.loc[weighted.patient_id == "D2"]
        .groupby(["case_id", "side"])
        .weight.sum()
        .to_numpy()
    )
    np.testing.assert_allclose(class_two_eye_mass, np.repeat(class_two_eye_mass[0], 2))
    assert weights.mean() == pytest.approx(1.0)


def _write_upstream_unit(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    root = tmp_path / "strict_results"
    run = root / "runs" / "yolo26" / "seed_17"
    segmenter = run / "segmenter"
    segmenter.mkdir(parents=True)
    checkpoint = segmenter / "selected.pt"
    checkpoint.write_bytes(b"frozen-segmentation-only-checkpoint")

    partition_paths = {
        "train_oof": run / "roi" / "train_oof_index.csv",
        "validation": run / "roi" / "validation" / "index.csv",
        "test": run / "evaluation" / "test_rois" / "index.csv",
    }
    caches: dict[str, Path] = {}
    for offset, (name, index_path) in enumerate(partition_paths.items()):
        label = offset
        identity = _identity(name)
        cache = run / "frozen_tensors" / f"{name}.npz"
        audit = run / "frozen_audit" / f"{name}.npz"
        _write_roi_cache(cache, identity)
        audit.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(audit, marker=np.asarray([offset], dtype=np.int64))
        caches[name] = cache
        row = pd.DataFrame(
            [
                {
                    "patient_id": f"P_{name}",
                    "case_id": f"P_{name}_SAG",
                    "side": "SAG",
                    "frame_id": "frame_0001",
                    "label": int(label > 0),
                    "label_3class": label,
                    "frame_identity_sha256": identity,
                    "roi_valid": True,
                    "cache_path": str(cache.resolve()),
                    "cache_sha256": sha256_file(cache),
                    "audit_path": str(audit.resolve()),
                    "audit_sha256": sha256_file(audit),
                }
            ]
        )
        index_path.parent.mkdir(parents=True, exist_ok=True)
        row.to_csv(index_path, index=False)
    lock = {
        "model": "yolo26",
        "seed": 17,
        "stage": "segmentation_only",
        "classifier_loss_weight": 0.0,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
    }
    (segmenter / "segmenter_lock.json").write_text(
        json.dumps(lock), encoding="utf-8"
    )
    return root, caches


def test_frozen_upstream_discovery_and_hash_audit_fail_closed(tmp_path: Path) -> None:
    root, caches = _write_upstream_unit(tmp_path)
    artifacts = discover_frozen_upstream_artifacts(
        root,
        families=("yolo26",),
        seeds=(17,),
        project_root=tmp_path,
    )
    assert len(artifacts) == 1
    report = audit_frozen_upstream_artifacts(
        artifacts[0], project_root=tmp_path
    )
    assert report["family"] == "yolo26"
    assert report["seed"] == 17
    assert report["segmentation_only"] is True
    assert set(report["partitions"]) == {"train_oof", "validation", "test"}
    assert len(report["inventory_sha256"]) == 64

    caches["validation"].write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        audit_frozen_upstream_artifacts(artifacts[0], project_root=tmp_path)


@pytest.mark.parametrize("family", THREE_CLASS_FAMILIES)
def test_four_model_specific_families_emit_three_logits(family: str) -> None:
    image = torch.rand(
        (1, 3, 224, 224), generator=torch.Generator().manual_seed(20260921)
    )
    mask = torch.zeros((1, 1, 224, 224), dtype=torch.bool)
    mask[:, :, 24:196, 30:188] = True
    geometry = torch.tensor([[0.25, 0.5, 0.6, -0.1]])
    model = build_threeclass_roi_classifier(
        family, lightweight=True, geometry_features=4
    ).eval()

    with torch.inference_mode():
        logits = model(image, geometry=geometry, hard_mask=mask)

    assert logits.shape == (1, 3)
    assert torch.isfinite(logits).all()
    assert model.parameter_info()["num_classes"] == 3
    assert model.classification_head[-1].out_features == 3


def test_standardized_resnet_signature_and_legacy_binary_package_are_unchanged() -> None:
    image = torch.rand(
        (1, 3, 224, 224), generator=torch.Generator().manual_seed(20260922)
    )
    geometry = torch.tensor([[0.25, 0.5, 0.6, -0.1]])
    threeclass = build_roi_classifier(
        pretrained=False, geometry_features=4
    ).eval()
    legacy = build_legacy_binary_classifier("emcad", lightweight=True).eval()
    mask = torch.ones((1, 1, 224, 224), dtype=torch.bool)

    with torch.inference_mode():
        threeclass_logits = threeclass(image, geometry)
        binary_logits = legacy(image, hard_mask=mask)

    assert threeclass_logits.shape == (1, 3)
    assert threeclass.head[-1].out_features == 3
    assert binary_logits.shape == (1, 2)
    assert legacy.classification_head[-1].out_features == 2
