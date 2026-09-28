from __future__ import annotations

import json

import pandas as pd
import pytest
import torch

from predicted_roi_study.resource_reporting import (
    FLOPS_RATIONALE,
    collect_compute_and_provenance,
    environment_inventory,
    history_summary,
    parameter_inventory,
)


def test_parameter_inventory_labels_registered_trainable_frozen_and_disabled():
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    model[1].requires_grad_(False)
    inventory = parameter_inventory(
        model,
        explicitly_disabled_legacy_parameters=sum(p.numel() for p in model[1].parameters()),
    )
    assert inventory["registered_parameters"] == 26
    assert inventory["trainable_updateable_parameters"] == 16
    assert inventory["frozen_parameters"] == 10
    assert inventory["explicitly_disabled_legacy_parameters"] == 10
    with pytest.raises(ValueError):
        parameter_inventory(model, explicitly_disabled_legacy_parameters=11)


def test_environment_inventory_is_explicit_and_never_invents_flops():
    inventory = environment_inventory(packages=("definitely-not-a-package",))
    assert inventory["packages"]["definitely-not-a-package"] is None
    assert "determinism" in inventory
    assert "torch_cuda_runtime" in inventory["cuda"]
    assert "incomplete" in FLOPS_RATIONALE


def test_history_summary_records_epochs_and_wall_seconds(tmp_path):
    path = tmp_path / "history.csv"
    pd.DataFrame({"epoch": [1, 2], "seconds": [1.25, 2.75]}).to_csv(path, index=False)
    assert history_summary(path) == {"epochs_recorded": 2, "seconds": 4.0}
    assert history_summary(tmp_path / "missing.csv") == {
        "epochs_recorded": 0,
        "seconds": None,
    }


def test_collect_compute_and_provenance_aggregates_outer_oof_and_classifier(
    tmp_path, monkeypatch
):
    root = tmp_path / "strict"
    run = root / "runs" / "toy" / "seed_17"
    (run / "segmenter").mkdir(parents=True)
    checkpoint = tmp_path / "segmenter.pt"
    checkpoint.write_bytes(b"checkpoint")
    from predicted_roi_study.config import sha256_file

    (run / "segmenter" / "model_info.json").write_text(
        json.dumps(
            {
                "registered_parameters": 100,
                "trainable_updateable_parameters": 80,
                "frozen_parameters": 20,
                "explicitly_disabled_legacy_parameters": 10,
            }
        ),
        encoding="utf-8",
    )
    (run / "segmenter" / "segmenter_lock.json").write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame({"seconds": [2.0, 3.0]}).to_csv(
        run / "segmenter" / "history.csv", index=False
    )
    for fold, seconds in enumerate((5.0, 7.0)):
        folder = run / "crossfit" / f"fold_{fold}" / "segmenter"
        folder.mkdir(parents=True)
        pd.DataFrame({"seconds": [seconds]}).to_csv(folder / "history.csv", index=False)
    for strategy, seconds in (("model_specific", 11.0), ("standardized_resnet18", 13.0)):
        folder = run / "classifiers" / strategy
        folder.mkdir(parents=True)
        (folder / "model_info.json").write_text(
            json.dumps(
                {
                    "registered_parameters": 50,
                    "trainable_updateable_parameters": 50,
                    "frozen_parameters": 0,
                    "classifier_strategy": strategy,
                    "classifier_family": "toy",
                }
            ),
            encoding="utf-8",
        )
        pd.DataFrame({"seconds": [seconds]}).to_csv(
            folder / "history.csv", index=False
        )
        (folder / "training_result.json").write_text(
            json.dumps({"available": False, "checkpoint": None}), encoding="utf-8"
        )
    (root / "verification").mkdir()
    (root / "verification" / "toy.json").write_text(
        json.dumps({"passed": True}), encoding="utf-8"
    )
    cfg = {
        "models": ["toy"],
        "split_seeds": [17],
        "cross_fitting": {"folds": 2},
        "training": {"seed_offsets": {"segmentation": 10_000, "classifier": 40_000}},
        "segmentation": {
            "initialization": {
                "toy": {"mode": "random_segmentation_weights", "checkpoint": None}
            }
        },
        "classifier": {
            "strategy_order": ["model_specific", "standardized_resnet18"],
            "primary": {
                "name": "model_specific",
                "role": "primary_estimand",
                "estimand_id": "E1",
                "seed_offset": 0,
                "initialization_by_segmenter": {
                    "toy": {"pretrained": False, "checkpoint": None}
                },
            },
            "secondary": {
                "name": "standardized_resnet18",
                "role": "secondary",
                "estimand_id": "E2",
                "seed_offset": 500009,
                "pretrained_weights": {"sha256": "x" * 64},
            },
        },
    }
    monkeypatch.setattr(
        "predicted_roi_study.resource_reporting.resolve_project_path",
        lambda value, must_exist=False: checkpoint,
    )
    table, details = collect_compute_and_provenance(cfg, output_root=root)
    assert len(table) == 2
    row = table.loc[table.classifier_strategy == "model_specific"].iloc[0]
    assert row.outer_segmenter_seconds == 5.0
    assert row.oof_segmenter_seconds_total == 12.0
    assert row.classifier_seconds == 11.0
    assert set(table.classifier_strategy) == {"model_specific", "standardized_resnet18"}
    assert table.loc[
        table.classifier_strategy == "standardized_resnet18", "classifier_seconds"
    ].iloc[0] == 13.0
    assert row.flops_status == "not_reported"
    assert details["preflight_reports"]["toy"]["report"]["passed"] is True
