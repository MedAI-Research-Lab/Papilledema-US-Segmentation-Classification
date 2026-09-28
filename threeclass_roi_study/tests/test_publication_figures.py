from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from threeclass_roi_study.config import sha256_file
from threeclass_roi_study.metrics import multiclass_risk_coverage_curve
from threeclass_roi_study.publication_figures import (
    FIGURE_MANIFEST_NAME,
    PublicationFigureError,
    generate_q1_figures,
    validate_q1_figures,
)


MODELS = ("model_a", "model_b")
SEEDS = (17, 42)
STRATEGY = "model_specific"


def _prediction_rows() -> list[dict[str, object]]:
    examples = (
        ("P0", 0, (0.80, 0.10, 0.10)),
        ("P1", 0, (0.40, 0.50, 0.10)),
        ("P2", 1, (0.15, 0.70, 0.15)),
        ("P3", 1, (0.20, 0.25, 0.55)),
        ("P4", 2, (0.10, 0.20, 0.70)),
        ("P5", 2, (0.10, 0.60, 0.30)),
        ("P6", 2, None),
    )
    rows: list[dict[str, object]] = []
    for model_index, model in enumerate(MODELS):
        for seed_index, seed in enumerate(SEEDS):
            for patient_id, true_label, base_probability in examples:
                if base_probability is None:
                    probability = (np.nan, np.nan, np.nan)
                    prediction = 3
                    abstained = True
                else:
                    # A small deterministic perturbation exercises distinct traces while
                    # preserving the probability sum and operational argmax.
                    delta = 0.005 * (model_index + seed_index)
                    probability = (
                        base_probability[0] + delta,
                        base_probability[1],
                        base_probability[2] - delta,
                    )
                    prediction = int(np.argmax(probability))
                    abstained = False
                rows.append(
                    {
                        "model": model,
                        "seed": seed,
                        "classifier_strategy": STRATEGY,
                        "patient_id": patient_id,
                        "true_label": true_label,
                        "p_normal_calibrated": probability[0],
                        "p_papilledema_calibrated": probability[1],
                        "p_pseudopapilledema_calibrated": probability[2],
                        "predicted_label": prediction,
                        "abstained": abstained,
                    }
                )
    return rows


def _write_sources(root: Path, *, include_segmentation: bool = True) -> dict[str, object]:
    tables = root / "tables"
    summary = root / "summary"
    tables.mkdir(parents=True)
    summary.mkdir(parents=True)
    predictions = pd.DataFrame(_prediction_rows())

    metric_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    risk_rows: list[dict[str, object]] = []
    segmentation_rows: list[dict[str, object]] = []
    for model_index, model in enumerate(MODELS):
        for seed_index, seed in enumerate(SEEDS):
            group = predictions.loc[
                (predictions["model"] == model) & (predictions["seed"] == seed)
            ]
            metric_rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "classifier_strategy": STRATEGY,
                    "probability_scale": "calibrated",
                    "failure_aware_balanced_accuracy": 0.44 + 0.02 * model_index + 0.01 * seed_index,
                    "coverage": 6 / 7,
                }
            )
            for actual in range(3):
                for predicted in range(4):
                    count = int(
                        (
                            (group["true_label"] == actual)
                            & (group["predicted_label"] == predicted)
                        ).sum()
                    )
                    confusion_rows.append(
                        {
                            "model": model,
                            "seed": seed,
                            "classifier_strategy": STRATEGY,
                            "true_label": actual,
                            "predicted_label": predicted,
                            "count": count,
                        }
                    )
            calibration_rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "classifier_strategy": STRATEGY,
                    "level": "patient",
                    "probability_scale": "calibrated",
                    "status": "available",
                    "temperature": 1.1,
                }
            )
            probability = group[
                [
                    "p_normal_calibrated",
                    "p_papilledema_calibrated",
                    "p_pseudopapilledema_calibrated",
                ]
            ].to_numpy(float)
            curve = multiclass_risk_coverage_curve(
                group["true_label"].to_numpy(int),
                probability,
                evaluable=~group["abstained"].to_numpy(bool),
                include_abstentions_as_failures=True,
            )
            for row in curve.itertuples(index=False):
                risk_rows.append(
                    {
                        "model": model,
                        "seed": seed,
                        "classifier_strategy": STRATEGY,
                        "rank": int(row.rank),
                        "coverage": float(row.coverage),
                        "selective_risk": float(row.risk),
                        "confidence_definition": "maximum_calibrated_patient_probability",
                    }
                )
            for class_label, class_name in enumerate(
                ("normal", "papilledema", "pseudopapilledema")
            ):
                segmentation_rows.append(
                    {
                        "model": model,
                        "seed": seed,
                        "class_label": class_label,
                        "class_name": class_name,
                        "level": "frame",
                        "roi_coverage": 0.72 + 0.03 * class_label + 0.01 * seed_index,
                        "dice": 0.66 + 0.04 * class_label + 0.01 * model_index,
                    }
                )

    source_tables = {
        "patient_predictions.csv": predictions,
        "patient_metrics.csv": pd.DataFrame(metric_rows),
        "patient_confusion_3x4.csv": pd.DataFrame(confusion_rows),
        "calibration_metrics.csv": pd.DataFrame(calibration_rows),
        "risk_coverage.csv": pd.DataFrame(risk_rows),
    }
    if include_segmentation:
        source_tables["segmentation_by_class.csv"] = pd.DataFrame(segmentation_rows)

    records: dict[str, dict[str, object]] = {}
    for name, table in source_tables.items():
        path = tables / name
        table.to_csv(path, index=False, lineterminator="\n")
        records[name] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "rows": len(table),
        }
    publication_manifest = {
        "schema_version": 1,
        "study_id": "synthetic_q1_figure_study",
        "tables": records,
    }
    (summary / "publication_output_manifest.json").write_text(
        json.dumps(publication_manifest, sort_keys=True), encoding="utf-8"
    )
    return {
        "study_id": "synthetic_q1_figure_study",
        "config_sha256": "a" * 64,
        "output": str(root),
        "models": list(MODELS),
        "split_seeds": list(SEEDS),
        "classifier": {"primary_strategy": STRATEGY},
        "classes": {
            "names": {
                "0": "normal",
                "1": "papilledema",
                "2": "pseudopapilledema",
            }
        },
    }


def test_q1_figures_generate_validate_and_are_deterministic(tmp_path: Path) -> None:
    cfg = _write_sources(tmp_path / "study")
    first = generate_q1_figures(cfg)
    expected_stems = {
        "q1_patient_ovr_roc_pr",
        "q1_patient_reliability",
        "q1_patient_risk_coverage",
        "q1_failure_aware_ba_coverage",
        "q1_patient_confusion_3x4_by_seed",
        "q1_segmentation_class_conditional",
    }
    assert set(first) == {
        *(f"{stem}.{extension}" for stem in expected_stems for extension in ("png", "pdf", "svg")),
        FIGURE_MANIFEST_NAME,
    }
    audit = validate_q1_figures(cfg)
    assert audit["status"] == "passed"
    assert audit["figures"] == 6
    assert audit["outputs"] == 18

    manifest = json.loads(first[FIGURE_MANIFEST_NAME].read_text(encoding="utf-8"))
    assert "not independent replicates" in " ".join(manifest["disclosures"])
    assert manifest["figures"]["q1_patient_ovr_roc_pr"]["analytic_details"]["pooling"] == "none"
    assert manifest["figures"]["q1_patient_confusion_3x4_by_seed"]["analytic_details"][
        "matrix_shape"
    ] == [3, 4]
    assert manifest["sources"]["patient_predictions.csv"]["sha256"] == sha256_file(
        Path(cfg["output"]) / "tables" / "patient_predictions.csv"
    )

    first_hashes = {name: sha256_file(path) for name, path in first.items()}
    second = generate_q1_figures(cfg)
    second_hashes = {name: sha256_file(path) for name, path in second.items()}
    assert second_hashes == first_hashes

    # The output validator fails closed when a rendered artifact changes.
    png = second["q1_patient_reliability.png"]
    png.write_bytes(png.read_bytes() + b"tampered")
    with pytest.raises(PublicationFigureError, match="hash/size mismatch"):
        validate_q1_figures(cfg)


def test_q1_figure_generation_rejects_changed_finalized_source(tmp_path: Path) -> None:
    cfg = _write_sources(tmp_path / "study", include_segmentation=False)
    source = Path(cfg["output"]) / "tables" / "patient_metrics.csv"
    source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(PublicationFigureError, match="source hash/size mismatch"):
        generate_q1_figures(cfg)
