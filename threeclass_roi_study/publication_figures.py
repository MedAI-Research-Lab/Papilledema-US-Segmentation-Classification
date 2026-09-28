"""Deterministic, downstream-only Q1 publication figures.

The renderer consumes the validated publication tables written after
``summarize``.  It never opens checkpoints, fits a model, changes a prediction,
or treats the overlapping seed holdouts as independent samples.  Each seed is
therefore drawn as its own trace or point and all across-seed summaries are
explicitly descriptive.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from PIL import Image
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)

from .config import CLASS_NAMES, canonical_sha256, sha256_file
from .protocol import output_root, read_json, save_json_atomic


FIGURE_SCHEMA_VERSION = 1
FIGURE_DPI = 300
FIGURE_FORMATS = ("png", "pdf", "svg")
FIGURE_MANIFEST_NAME = "q1_figure_manifest.json"

_REQUIRED_SOURCE_TABLES = (
    "patient_predictions.csv",
    "patient_metrics.csv",
    "patient_confusion_3x4.csv",
    "calibration_metrics.csv",
    "risk_coverage.csv",
)
_OPTIONAL_SOURCE_TABLES = ("segmentation_by_class.csv",)
_REQUIRED_FIGURES = (
    "q1_patient_ovr_roc_pr",
    "q1_patient_reliability",
    "q1_patient_risk_coverage",
    "q1_failure_aware_ba_coverage",
    "q1_patient_confusion_3x4_by_seed",
)

_PROBABILITY_COLUMNS = (
    "p_normal_calibrated",
    "p_papilledema_calibrated",
    "p_pseudopapilledema_calibrated",
)
_CLASS_COLOURS = ("#0072B2", "#D55E00", "#009E73")
_MODEL_COLOURS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9")
_SEED_LINESTYLES = ("-", "--", "-.", ":", (0, (5, 2, 1, 2)))
_SEED_MARKERS = ("o", "s", "^", "D", "P", "X", "v")

_COMMON_DISCLOSURES = (
    "Seed holdouts reuse overlapping patient memberships and are not independent replicates; "
    "seed traces, ranges, and arithmetic means are descriptive only.",
    "All diagnostic figures use the locked primary classifier strategy and patient-level "
    "calibrated probabilities.",
    "ROC, precision-recall, and reliability traces condition on non-abstained patients; "
    "abstention is represented in the failure-aware performance, risk-coverage, and 3x4 "
    "confusion figures.",
)


class PublicationFigureError(RuntimeError):
    """Raised when a figure source or rendered output violates its contract."""


def _display(value: Any) -> str:
    return str(value).replace("_", " ").strip().title()


def _config_identity(cfg: Mapping[str, Any]) -> str:
    value = cfg.get("config_sha256")
    return str(value) if value else canonical_sha256(cfg)


def _class_names(cfg: Mapping[str, Any]) -> tuple[str, str, str]:
    configured = cfg.get("classes", {}).get("names", {})
    values = tuple(
        str(configured.get(str(index), configured.get(index, CLASS_NAMES[index])))
        for index in range(3)
    )
    if any(not value for value in values) or len(set(values)) != 3:
        raise PublicationFigureError("The three class display names must be non-empty and unique")
    return values  # type: ignore[return-value]


def _study_axes(cfg: Mapping[str, Any]) -> tuple[list[str], list[int], str]:
    models = [str(value) for value in cfg.get("models", ())]
    seeds = [int(value) for value in cfg.get("split_seeds", ())]
    try:
        primary_strategy = str(cfg["classifier"]["primary_strategy"])
    except (KeyError, TypeError) as exc:
        raise PublicationFigureError("classifier.primary_strategy is required") from exc
    if not models or len(models) != len(set(models)):
        raise PublicationFigureError("models must be a non-empty unique sequence")
    if not seeds or len(seeds) != len(set(seeds)):
        raise PublicationFigureError("split_seeds must be a non-empty unique sequence")
    if not primary_strategy:
        raise PublicationFigureError("classifier.primary_strategy cannot be empty")
    return models, seeds, primary_strategy


def _strict_bool(series: pd.Series, name: str) -> pd.Series:
    if series.dtype == bool:
        return series.astype(bool)
    mapped = series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if mapped.isna().any():
        raise PublicationFigureError(f"Malformed boolean values in {name}")
    return mapped.astype(bool)


def _require_columns(table: pd.DataFrame, columns: Iterable[str], table_name: str) -> None:
    missing = sorted(set(columns) - set(table.columns))
    if missing:
        raise PublicationFigureError(f"{table_name} is missing columns: {missing}")


def _numeric(
    table: pd.DataFrame,
    columns: Iterable[str],
    table_name: str,
    *,
    nullable: bool = False,
) -> None:
    for column in columns:
        converted = pd.to_numeric(table[column], errors="coerce")
        invalid = converted.isna() & table[column].notna()
        if invalid.any() or (not nullable and converted.isna().any()):
            raise PublicationFigureError(f"{table_name}.{column} contains non-numeric values")
        table[column] = converted


def _assert_unit_interval(
    table: pd.DataFrame, columns: Iterable[str], table_name: str, *, nullable: bool = False
) -> None:
    for column in columns:
        value = pd.to_numeric(table[column], errors="coerce")
        if not nullable and value.isna().any():
            raise PublicationFigureError(f"{table_name}.{column} contains missing values")
        finite = value.dropna()
        if ((finite < 0) | (finite > 1) | ~np.isfinite(finite)).any():
            raise PublicationFigureError(f"{table_name}.{column} must lie in [0, 1]")


def _expected_groups(models: Sequence[str], seeds: Sequence[int]) -> set[tuple[str, int]]:
    return {(model, int(seed)) for model in models for seed in seeds}


def _observed_groups(table: pd.DataFrame) -> set[tuple[str, int]]:
    return {
        (str(model), int(seed))
        for model, seed in table.loc[:, ["model", "seed"]].itertuples(index=False, name=None)
    }


def _filter_primary(
    table: pd.DataFrame,
    *,
    strategy: str,
    table_name: str,
    models: Sequence[str],
    seeds: Sequence[int],
) -> pd.DataFrame:
    _require_columns(table, ("model", "seed", "classifier_strategy"), table_name)
    table = table.loc[table["classifier_strategy"].astype(str) == strategy].copy()
    table["model"] = table["model"].astype(str)
    _numeric(table, ("seed",), table_name)
    table["seed"] = table["seed"].astype(int)
    expected = _expected_groups(models, seeds)
    observed = _observed_groups(table)
    if observed != expected:
        raise PublicationFigureError(
            f"{table_name} primary-strategy model/seed groups differ from the config; "
            f"missing={sorted(expected - observed)}, unexpected={sorted(observed - expected)}"
        )
    return table


def _read_publication_sources(
    cfg: Mapping[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    """Hash-check and semantically validate the figure source tables."""

    root = output_root(cfg)
    publication_manifest_path = root / "summary" / "publication_output_manifest.json"
    if not publication_manifest_path.is_file():
        raise PublicationFigureError(
            "Publication outputs are not finalized: publication_output_manifest.json is missing"
        )
    publication_manifest = read_json(publication_manifest_path)
    if publication_manifest.get("study_id") != cfg.get("study_id"):
        raise PublicationFigureError("Publication manifest study_id does not match the config")
    manifest_tables = publication_manifest.get("tables")
    if not isinstance(manifest_tables, Mapping):
        raise PublicationFigureError("Publication manifest lacks a table inventory")

    tables: dict[str, pd.DataFrame] = {}
    source_records: dict[str, dict[str, Any]] = {
        "publication_output_manifest.json": {
            "path": str(publication_manifest_path.resolve()),
            "sha256": sha256_file(publication_manifest_path),
            "size_bytes": int(publication_manifest_path.stat().st_size),
        }
    }
    for table_name in (*_REQUIRED_SOURCE_TABLES, *_OPTIONAL_SOURCE_TABLES):
        path = root / "tables" / table_name
        record = manifest_tables.get(table_name)
        if table_name in _OPTIONAL_SOURCE_TABLES and (not path.is_file() or not record):
            continue
        if not path.is_file() or not isinstance(record, Mapping):
            raise PublicationFigureError(f"Finalized publication source is missing: {table_name}")
        digest = sha256_file(path)
        if digest != record.get("sha256") or path.stat().st_size != record.get("size_bytes"):
            raise PublicationFigureError(f"Publication source hash/size mismatch: {table_name}")
        table = pd.read_csv(path, dtype={"patient_id": str})
        if int(record.get("rows", -1)) != len(table):
            raise PublicationFigureError(f"Publication source row-count mismatch: {table_name}")
        tables[table_name] = table
        source_records[table_name] = {
            "path": str(path.resolve()),
            "sha256": digest,
            "size_bytes": int(path.stat().st_size),
            "rows": int(len(table)),
        }

    models, seeds, strategy = _study_axes(cfg)
    predictions = _filter_primary(
        tables["patient_predictions.csv"],
        strategy=strategy,
        table_name="patient_predictions.csv",
        models=models,
        seeds=seeds,
    )
    _require_columns(
        predictions,
        (
            "patient_id",
            "true_label",
            *_PROBABILITY_COLUMNS,
            "predicted_label",
            "abstained",
        ),
        "patient_predictions.csv",
    )
    _numeric(
        predictions,
        ("true_label", "predicted_label", *_PROBABILITY_COLUMNS),
        "patient_predictions.csv",
        nullable=True,
    )
    predictions["true_label"] = predictions["true_label"].astype(int)
    predictions["predicted_label"] = predictions["predicted_label"].astype(int)
    predictions["abstained"] = _strict_bool(predictions["abstained"], "abstained")
    if not set(predictions["true_label"]).issubset({0, 1, 2}):
        raise PublicationFigureError("patient_predictions.csv has an invalid true label")
    if not set(predictions["predicted_label"]).issubset({0, 1, 2, 3}):
        raise PublicationFigureError("patient_predictions.csv has an invalid operational label")
    if predictions.duplicated(["model", "seed", "patient_id"]).any():
        raise PublicationFigureError("patient_predictions.csv repeats a patient within a model/seed")
    probability = predictions.loc[:, _PROBABILITY_COLUMNS].to_numpy(dtype=float)
    abstained = predictions["abstained"].to_numpy(dtype=bool)
    if np.isfinite(probability[abstained]).any():
        raise PublicationFigureError("Abstained patients must not have calibrated probabilities")
    if len(probability[~abstained]):
        if not np.isfinite(probability[~abstained]).all():
            raise PublicationFigureError("Non-abstained patients require complete probabilities")
        if np.any((probability[~abstained] < 0) | (probability[~abstained] > 1)):
            raise PublicationFigureError("Calibrated probabilities must lie in [0, 1]")
        if not np.allclose(probability[~abstained].sum(axis=1), 1.0, atol=1e-6):
            raise PublicationFigureError("Calibrated probability vectors must sum to one")
        issued = np.argmax(probability[~abstained], axis=1)
        if not np.array_equal(issued, predictions.loc[~predictions["abstained"], "predicted_label"]):
            raise PublicationFigureError("Operational predictions differ from calibrated argmax")
    if not (predictions.loc[predictions["abstained"], "predicted_label"] == 3).all():
        raise PublicationFigureError("Abstained patients must use operational label 3")
    tables["patient_predictions.csv"] = predictions.sort_values(
        ["model", "seed", "patient_id"], kind="stable"
    ).reset_index(drop=True)

    patient_metrics = _filter_primary(
        tables["patient_metrics.csv"],
        strategy=strategy,
        table_name="patient_metrics.csv",
        models=models,
        seeds=seeds,
    )
    _require_columns(
        patient_metrics,
        ("probability_scale", "failure_aware_balanced_accuracy", "coverage"),
        "patient_metrics.csv",
    )
    patient_metrics = patient_metrics.loc[
        patient_metrics["probability_scale"].astype(str) == "calibrated"
    ].copy()
    if patient_metrics.duplicated(["model", "seed"]).any() or _observed_groups(
        patient_metrics
    ) != _expected_groups(models, seeds):
        raise PublicationFigureError("patient_metrics.csv needs one calibrated row per model/seed")
    _numeric(
        patient_metrics,
        ("failure_aware_balanced_accuracy", "coverage"),
        "patient_metrics.csv",
    )
    _assert_unit_interval(
        patient_metrics,
        ("failure_aware_balanced_accuracy", "coverage"),
        "patient_metrics.csv",
    )
    tables["patient_metrics.csv"] = patient_metrics.sort_values(
        ["model", "seed"], kind="stable"
    ).reset_index(drop=True)

    confusion = _filter_primary(
        tables["patient_confusion_3x4.csv"],
        strategy=strategy,
        table_name="patient_confusion_3x4.csv",
        models=models,
        seeds=seeds,
    )
    _require_columns(confusion, ("true_label", "predicted_label", "count"), "patient_confusion_3x4.csv")
    _numeric(confusion, ("true_label", "predicted_label", "count"), "patient_confusion_3x4.csv")
    confusion[["true_label", "predicted_label", "count"]] = confusion[
        ["true_label", "predicted_label", "count"]
    ].astype(int)
    for identity, group in confusion.groupby(["model", "seed"], sort=False):
        cells = set(zip(group["true_label"], group["predicted_label"]))
        expected_cells = {(actual, predicted) for actual in range(3) for predicted in range(4)}
        if cells != expected_cells or len(group) != 12 or (group["count"] < 0).any():
            raise PublicationFigureError(f"Incomplete 3x4 confusion grid for {identity}")
        corresponding = predictions.loc[
            (predictions["model"] == identity[0]) & (predictions["seed"] == identity[1])
        ]
        observed = np.zeros((3, 4), dtype=int)
        for row in corresponding.itertuples(index=False):
            observed[int(row.true_label), int(row.predicted_label)] += 1
        published = (
            group.pivot(index="true_label", columns="predicted_label", values="count")
            .reindex(index=range(3), columns=range(4))
            .to_numpy(dtype=int)
        )
        if not np.array_equal(observed, published):
            raise PublicationFigureError(f"Confusion cells disagree with patient predictions for {identity}")
    tables["patient_confusion_3x4.csv"] = confusion.sort_values(
        ["model", "seed", "true_label", "predicted_label"], kind="stable"
    ).reset_index(drop=True)

    calibration = _filter_primary(
        tables["calibration_metrics.csv"],
        strategy=strategy,
        table_name="calibration_metrics.csv",
        models=models,
        seeds=seeds,
    )
    _require_columns(calibration, ("level", "probability_scale", "status"), "calibration_metrics.csv")
    calibration = calibration.loc[
        (calibration["level"].astype(str) == "patient")
        & (calibration["probability_scale"].astype(str) == "calibrated")
    ].copy()
    if calibration.duplicated(["model", "seed"]).any() or _observed_groups(
        calibration
    ) != _expected_groups(models, seeds):
        raise PublicationFigureError("calibration_metrics.csv needs one calibrated patient row per model/seed")
    if not set(calibration["status"].astype(str)).issubset({"available", "unavailable"}):
        raise PublicationFigureError("Unexpected calibrated patient status")
    tables["calibration_metrics.csv"] = calibration.sort_values(
        ["model", "seed"], kind="stable"
    ).reset_index(drop=True)

    risk = _filter_primary(
        tables["risk_coverage.csv"],
        strategy=strategy,
        table_name="risk_coverage.csv",
        models=models,
        seeds=seeds,
    )
    _require_columns(
        risk,
        ("rank", "coverage", "selective_risk", "confidence_definition"),
        "risk_coverage.csv",
    )
    _numeric(risk, ("rank", "coverage", "selective_risk"), "risk_coverage.csv")
    _assert_unit_interval(risk, ("coverage", "selective_risk"), "risk_coverage.csv")
    for identity, group in risk.groupby(["model", "seed"], sort=False):
        group = group.sort_values("rank", kind="stable")
        ranks = group["rank"].astype(int).to_numpy()
        if not np.array_equal(ranks, np.arange(1, len(group) + 1)):
            raise PublicationFigureError(f"Risk-coverage ranks are not consecutive for {identity}")
        coverage = group["coverage"].to_numpy(dtype=float)
        if np.any(np.diff(coverage) <= 0) or not math.isclose(float(coverage[-1]), 1.0, abs_tol=1e-9):
            raise PublicationFigureError(f"Risk-coverage must increase to 1 for {identity}")
        if set(group["confidence_definition"].astype(str)) != {
            "maximum_calibrated_patient_probability"
        }:
            raise PublicationFigureError(f"Risk-coverage confidence definition changed for {identity}")
    tables["risk_coverage.csv"] = risk.sort_values(
        ["model", "seed", "rank"], kind="stable"
    ).reset_index(drop=True)

    segmentation = tables.get("segmentation_by_class.csv")
    if segmentation is not None:
        _require_columns(
            segmentation,
            (
                "model",
                "seed",
                "class_label",
                "class_name",
                "level",
                "roi_coverage",
                "dice",
            ),
            "segmentation_by_class.csv",
        )
        segmentation["model"] = segmentation["model"].astype(str)
        _numeric(segmentation, ("seed", "class_label", "roi_coverage", "dice"), "segmentation_by_class.csv", nullable=True)
        segmentation["seed"] = segmentation["seed"].astype(int)
        segmentation["class_label"] = segmentation["class_label"].astype(int)
        segmentation = segmentation.loc[segmentation["level"].astype(str) == "frame"].copy()
        expected_segmentation = {
            (model, seed, class_label)
            for model in models
            for seed in seeds
            for class_label in range(3)
        }
        observed_segmentation = set(
            segmentation[["model", "seed", "class_label"]].itertuples(index=False, name=None)
        )
        if (
            observed_segmentation != expected_segmentation
            or segmentation.duplicated(["model", "seed", "class_label"]).any()
        ):
            raise PublicationFigureError("segmentation_by_class.csv lacks a complete frame-level grid")
        _assert_unit_interval(segmentation, ("roi_coverage",), "segmentation_by_class.csv")
        _assert_unit_interval(segmentation, ("dice",), "segmentation_by_class.csv", nullable=True)
        tables["segmentation_by_class.csv"] = segmentation.sort_values(
            ["model", "seed", "class_label"], kind="stable"
        ).reset_index(drop=True)

    return tables, source_records


def _style_axis(axis: plt.Axes, *, equal: bool = False) -> None:
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xticks(np.linspace(0, 1, 6))
    axis.set_yticks(np.linspace(0, 1, 6))
    axis.grid(True, color="#D8D8D8", linewidth=0.6, alpha=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    if equal:
        axis.set_aspect("equal", adjustable="box")


def _row_label(axis: plt.Axes, text: str) -> None:
    axis.text(
        -0.23,
        0.5,
        text,
        transform=axis.transAxes,
        ha="right",
        va="center",
        rotation=90,
        fontsize=10,
        fontweight="bold",
    )


def _legend_handles(
    class_names: Sequence[str], seeds: Sequence[int]
) -> tuple[list[Line2D], list[Line2D]]:
    class_handles = [
        Line2D([0], [0], color=_CLASS_COLOURS[index], lw=2.2, label=_display(name))
        for index, name in enumerate(class_names)
    ]
    seed_handles = [
        Line2D(
            [0],
            [0],
            color="#333333",
            linestyle=_SEED_LINESTYLES[index % len(_SEED_LINESTYLES)],
            lw=1.7,
            label=f"Seed {seed}",
        )
        for index, seed in enumerate(seeds)
    ]
    return class_handles, seed_handles


def _plot_roc_pr(
    predictions: pd.DataFrame,
    models: Sequence[str],
    seeds: Sequence[int],
    class_names: Sequence[str],
) -> tuple[Figure, dict[str, Any]]:
    figure, axes = plt.subplots(
        len(models), 2, squeeze=False, figsize=(12.0, 3.0 * len(models) + 1.65)
    )
    trace_statistics: list[dict[str, Any]] = []
    omitted: list[dict[str, Any]] = []
    for model_index, model in enumerate(models):
        roc_axis, pr_axis = axes[model_index]
        for seed_index, seed in enumerate(seeds):
            group = predictions.loc[
                (predictions["model"] == model)
                & (predictions["seed"] == seed)
                & ~predictions["abstained"]
            ]
            y = group["true_label"].to_numpy(dtype=int)
            probability = group.loc[:, _PROBABILITY_COLUMNS].to_numpy(dtype=float)
            for class_index, class_name in enumerate(class_names):
                binary = (y == class_index).astype(int)
                if len(binary) == 0 or len(np.unique(binary)) != 2:
                    omitted.append(
                        {
                            "model": model,
                            "seed": int(seed),
                            "class_label": class_index,
                            "reason": "one_vs_rest_requires_both_outcomes",
                        }
                    )
                    continue
                line_style = _SEED_LINESTYLES[seed_index % len(_SEED_LINESTYLES)]
                fpr, tpr, _ = roc_curve(binary, probability[:, class_index])
                precision, recall, _ = precision_recall_curve(
                    binary, probability[:, class_index]
                )
                roc_value = float(auc(fpr, tpr))
                ap_value = float(average_precision_score(binary, probability[:, class_index]))
                roc_axis.plot(
                    fpr,
                    tpr,
                    color=_CLASS_COLOURS[class_index],
                    linestyle=line_style,
                    linewidth=1.55,
                    alpha=0.9,
                )
                pr_axis.plot(
                    recall,
                    precision,
                    color=_CLASS_COLOURS[class_index],
                    linestyle=line_style,
                    linewidth=1.55,
                    alpha=0.9,
                )
                trace_statistics.append(
                    {
                        "model": model,
                        "seed": int(seed),
                        "class_label": class_index,
                        "class_name": class_name,
                        "n_evaluable": int(len(binary)),
                        "n_positive": int(binary.sum()),
                        "auroc": roc_value,
                        "average_precision": ap_value,
                    }
                )
        roc_axis.plot([0, 1], [0, 1], color="#777777", linestyle=(0, (2, 2)), linewidth=1)
        _style_axis(roc_axis)
        _style_axis(pr_axis)
        roc_axis.set_ylabel("Sensitivity")
        pr_axis.set_ylabel("Precision (positive predictive value)")
        roc_axis.set_xlabel("1 − specificity")
        pr_axis.set_xlabel("Recall (sensitivity)")
        _row_label(roc_axis, _display(model))
        if model_index == 0:
            roc_axis.set_title("One-vs-rest ROC", fontweight="bold")
            pr_axis.set_title("One-vs-rest precision–recall", fontweight="bold")

    class_handles, seed_handles = _legend_handles(class_names, seeds)
    figure.legend(
        handles=class_handles,
        loc="lower center",
        bbox_to_anchor=(0.32, 0.075),
        ncol=3,
        title="Target class (colour)",
        frameon=False,
    )
    figure.legend(
        handles=seed_handles,
        loc="lower center",
        bbox_to_anchor=(0.77, 0.075),
        ncol=min(len(seeds), 5),
        title="Held-out split (line style)",
        frameon=False,
    )
    figure.suptitle(
        "Patient-level calibrated discrimination — primary strategy",
        y=0.985,
        fontsize=14,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.012,
        "Each trace is one seed and target class among non-abstained patients; no patients or seeds are pooled. "
        "Seed holdouts overlap and are descriptive, not independent replications.",
        ha="center",
        va="bottom",
        fontsize=8.3,
    )
    figure.subplots_adjust(top=0.91, bottom=0.24, left=0.12, right=0.98, hspace=0.42, wspace=0.27)
    return figure, {
        "trace_statistics": trace_statistics,
        "omitted_traces": omitted,
        "conditional_on_evaluable": True,
        "pooling": "none",
    }


def _reliability_points(
    probability: np.ndarray, outcome: np.ndarray, edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bin_index = np.searchsorted(edges, probability, side="right") - 1
    bin_index = np.clip(bin_index, 0, len(edges) - 2)
    mean_probability: list[float] = []
    observed_frequency: list[float] = []
    counts: list[int] = []
    for index in range(len(edges) - 1):
        selected = bin_index == index
        if not selected.any():
            continue
        mean_probability.append(float(probability[selected].mean()))
        observed_frequency.append(float(outcome[selected].mean()))
        counts.append(int(selected.sum()))
    return (
        np.asarray(mean_probability),
        np.asarray(observed_frequency),
        np.asarray(counts, dtype=int),
    )


def _plot_reliability(
    predictions: pd.DataFrame,
    models: Sequence[str],
    seeds: Sequence[int],
    class_names: Sequence[str],
) -> tuple[Figure, dict[str, Any]]:
    edges = np.linspace(0, 1, 6)
    figure, axes = plt.subplots(
        len(models), 3, squeeze=False, figsize=(13.2, 3.05 * len(models) + 1.55)
    )
    bin_inventory: list[dict[str, Any]] = []
    for model_index, model in enumerate(models):
        for class_index, class_name in enumerate(class_names):
            axis = axes[model_index, class_index]
            axis.plot([0, 1], [0, 1], color="#666666", linestyle=(0, (2, 2)), linewidth=1)
            for seed_index, seed in enumerate(seeds):
                group = predictions.loc[
                    (predictions["model"] == model)
                    & (predictions["seed"] == seed)
                    & ~predictions["abstained"]
                ]
                probability = group[_PROBABILITY_COLUMNS[class_index]].to_numpy(dtype=float)
                outcome = (group["true_label"].to_numpy(dtype=int) == class_index).astype(float)
                mean_probability, observed, counts = _reliability_points(
                    probability, outcome, edges
                )
                axis.plot(
                    mean_probability,
                    observed,
                    color=_CLASS_COLOURS[class_index],
                    linestyle=_SEED_LINESTYLES[seed_index % len(_SEED_LINESTYLES)],
                    marker=_SEED_MARKERS[seed_index % len(_SEED_MARKERS)],
                    markersize=3.8,
                    linewidth=1.35,
                    alpha=0.9,
                )
                bin_inventory.append(
                    {
                        "model": model,
                        "seed": int(seed),
                        "class_label": class_index,
                        "nonempty_bins": int(len(counts)),
                        "bin_counts": counts.tolist(),
                    }
                )
            _style_axis(axis, equal=True)
            axis.set_xlabel("Mean predicted probability")
            if class_index == 0:
                axis.set_ylabel("Observed class frequency")
                _row_label(axis, _display(model))
            else:
                axis.set_ylabel("")
            if model_index == 0:
                axis.set_title(_display(class_name), color=_CLASS_COLOURS[class_index], fontweight="bold")

    seed_handles = [
        Line2D(
            [0],
            [0],
            color="#333333",
            linestyle=_SEED_LINESTYLES[index % len(_SEED_LINESTYLES)],
            marker=_SEED_MARKERS[index % len(_SEED_MARKERS)],
            markersize=4,
            label=f"Seed {seed}",
        )
        for index, seed in enumerate(seeds)
    ]
    figure.legend(
        handles=seed_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.068),
        ncol=min(len(seeds), 5),
        title="Held-out split",
        frameon=False,
    )
    figure.suptitle(
        "Classwise patient reliability — calibrated primary strategy",
        y=0.985,
        fontsize=14,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.012,
        "Five fixed-width probability bins per seed; empty bins are omitted. Diagonal denotes perfect calibration. "
        "Non-abstained patients only; seeds are not pooled.",
        ha="center",
        va="bottom",
        fontsize=8.3,
    )
    figure.subplots_adjust(top=0.91, bottom=0.21, left=0.105, right=0.985, hspace=0.45, wspace=0.26)
    return figure, {
        "binning": {
            "method": "fixed_width",
            "edges": edges.tolist(),
            "empty_bins": "omitted",
        },
        "bin_inventory": bin_inventory,
        "conditional_on_evaluable": True,
        "pooling": "none",
    }


def _plot_risk_coverage(
    risk: pd.DataFrame, models: Sequence[str], seeds: Sequence[int]
) -> tuple[Figure, dict[str, Any]]:
    columns = min(2, len(models))
    rows = int(math.ceil(len(models) / columns))
    figure, axes_raw = plt.subplots(
        rows, columns, squeeze=False, figsize=(6.2 * columns, 4.0 * rows + 1.45)
    )
    axes = list(axes_raw.flat)
    curve_inventory: list[dict[str, Any]] = []
    for model_index, model in enumerate(models):
        axis = axes[model_index]
        for seed_index, seed in enumerate(seeds):
            group = risk.loc[(risk["model"] == model) & (risk["seed"] == seed)].sort_values(
                "rank", kind="stable"
            )
            axis.step(
                group["coverage"],
                group["selective_risk"],
                where="post",
                color=_MODEL_COLOURS[model_index % len(_MODEL_COLOURS)],
                linestyle=_SEED_LINESTYLES[seed_index % len(_SEED_LINESTYLES)],
                linewidth=1.65,
                alpha=0.92,
                label=f"Seed {seed}",
            )
            curve_inventory.append(
                {
                    "model": model,
                    "seed": int(seed),
                    "points": int(len(group)),
                    "terminal_risk": float(group.iloc[-1]["selective_risk"]),
                }
            )
        _style_axis(axis)
        axis.set_xlabel("Coverage (fraction of intended patients retained)")
        axis.set_ylabel("Selective error rate")
        axis.set_title(_display(model), fontweight="bold")
    for axis in axes[len(models) :]:
        axis.set_visible(False)
    seed_handles = [
        Line2D(
            [0],
            [0],
            color="#333333",
            linestyle=_SEED_LINESTYLES[index % len(_SEED_LINESTYLES)],
            lw=1.8,
            label=f"Seed {seed}",
        )
        for index, seed in enumerate(seeds)
    ]
    figure.legend(
        handles=seed_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.07),
        ncol=min(len(seeds), 5),
        title="Held-out split",
        frameon=False,
    )
    figure.suptitle(
        "Failure-aware patient risk–coverage — calibrated primary strategy",
        y=0.985,
        fontsize=14,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.012,
        "Patients are ordered by maximum calibrated class probability within each seed. Structural abstentions are "
        "appended at lowest confidence and counted as errors; intended patients define coverage.",
        ha="center",
        va="bottom",
        fontsize=8.3,
    )
    figure.subplots_adjust(top=0.89, bottom=0.22, left=0.09, right=0.98, hspace=0.36, wspace=0.25)
    return figure, {
        "curve_inventory": curve_inventory,
        "confidence_definition": "maximum_calibrated_patient_probability",
        "abstentions": "appended_at_lowest_confidence_and_counted_as_errors",
        "pooling": "none",
    }


def _plot_per_seed_point_range(
    patient_metrics: pd.DataFrame,
    models: Sequence[str],
    seeds: Sequence[int],
) -> tuple[Figure, dict[str, Any]]:
    metrics = (
        ("failure_aware_balanced_accuracy", "Failure-aware balanced accuracy"),
        ("coverage", "Operational coverage"),
    )
    figure, axes = plt.subplots(1, 2, figsize=(12.4, max(4.4, 0.72 * len(models) + 2.75)))
    summary: list[dict[str, Any]] = []
    base_y = np.arange(len(models), dtype=float)
    offsets = np.linspace(-0.16, 0.16, len(seeds)) if len(seeds) > 1 else np.zeros(1)
    for axis, (column, label) in zip(axes, metrics, strict=True):
        for model_index, model in enumerate(models):
            group = patient_metrics.loc[patient_metrics["model"] == model].set_index("seed")
            values = np.asarray([float(group.loc[seed, column]) for seed in seeds])
            minimum, maximum, mean = float(values.min()), float(values.max()), float(values.mean())
            axis.hlines(
                base_y[model_index],
                minimum,
                maximum,
                color=_MODEL_COLOURS[model_index % len(_MODEL_COLOURS)],
                linewidth=3.2,
                alpha=0.55,
                zorder=1,
            )
            for seed_index, value in enumerate(values):
                axis.scatter(
                    value,
                    base_y[model_index] + offsets[seed_index],
                    marker=_SEED_MARKERS[seed_index % len(_SEED_MARKERS)],
                    s=34,
                    facecolor="white",
                    edgecolor=_MODEL_COLOURS[model_index % len(_MODEL_COLOURS)],
                    linewidth=1.35,
                    zorder=3,
                )
            axis.scatter(
                mean,
                base_y[model_index],
                marker="D",
                s=48,
                facecolor=_MODEL_COLOURS[model_index % len(_MODEL_COLOURS)],
                edgecolor="#222222",
                linewidth=0.7,
                zorder=4,
            )
            summary.append(
                {
                    "model": model,
                    "metric": column,
                    "seed_values": [float(value) for value in values],
                    "arithmetic_mean": mean,
                    "minimum": minimum,
                    "maximum": maximum,
                }
            )
        axis.set_xlim(0, 1)
        axis.set_xticks(np.linspace(0, 1, 6))
        axis.set_yticks(base_y, labels=[_display(model) for model in models])
        axis.invert_yaxis()
        axis.set_xlabel(label)
        axis.grid(True, axis="x", color="#D8D8D8", linewidth=0.7)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
    seed_handles = [
        Line2D(
            [0],
            [0],
            marker=_SEED_MARKERS[index % len(_SEED_MARKERS)],
            markerfacecolor="white",
            markeredgecolor="#333333",
            linestyle="None",
            label=f"Seed {seed}",
        )
        for index, seed in enumerate(seeds)
    ]
    seed_handles.append(
        Line2D(
            [0],
            [0],
            marker="D",
            markerfacecolor="#666666",
            markeredgecolor="#222222",
            linestyle="-",
            color="#999999",
            lw=3,
            label="Mean and observed range",
        )
    )
    figure.legend(
        handles=seed_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.08),
        ncol=min(len(seed_handles), 6),
        frameon=False,
    )
    figure.suptitle(
        "Patient failure-aware performance and coverage by seed",
        y=0.975,
        fontsize=14,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.018,
        "Dots are held-out seeds; diamonds are arithmetic means and bars span the observed minimum–maximum. "
        "These are descriptive summaries, not confidence intervals.",
        ha="center",
        va="bottom",
        fontsize=8.3,
    )
    figure.subplots_adjust(top=0.85, bottom=0.28, left=0.16, right=0.98, wspace=0.27)
    return figure, {
        "summary": summary,
        "point_unit": "held_out_seed",
        "range": "observed_minimum_to_maximum_not_confidence_interval",
        "centre": "arithmetic_mean_descriptive_only",
    }


def _plot_confusion(
    confusion: pd.DataFrame,
    models: Sequence[str],
    seeds: Sequence[int],
    class_names: Sequence[str],
) -> tuple[Figure, dict[str, Any]]:
    matrices: dict[tuple[str, int], np.ndarray] = {}
    maximum = 1
    for model in models:
        for seed in seeds:
            group = confusion.loc[
                (confusion["model"] == model) & (confusion["seed"] == seed)
            ]
            matrix = (
                group.pivot(index="true_label", columns="predicted_label", values="count")
                .reindex(index=range(3), columns=range(4))
                .to_numpy(dtype=int)
            )
            matrices[(model, seed)] = matrix
            maximum = max(maximum, int(matrix.max()))

    width = max(10.5, 2.55 * len(seeds) + 1.1)
    height = max(4.5, 2.35 * len(models) + 1.65)
    figure, axes = plt.subplots(
        len(models), len(seeds), squeeze=False, figsize=(width, height)
    )
    image = None
    for model_index, model in enumerate(models):
        for seed_index, seed in enumerate(seeds):
            axis = axes[model_index, seed_index]
            matrix = matrices[(model, seed)]
            image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=maximum, aspect="auto")
            for row in range(3):
                for column in range(4):
                    value = int(matrix[row, column])
                    axis.text(
                        column,
                        row,
                        str(value),
                        ha="center",
                        va="center",
                        fontsize=8.5,
                        color="white" if value > maximum * 0.55 else "#111111",
                        fontweight="bold" if value else "normal",
                    )
            axis.set_xticks(range(4))
            axis.set_yticks(range(3))
            if model_index == len(models) - 1:
                axis.set_xticklabels(
                    [_display(name) for name in class_names] + ["Abstain"],
                    rotation=42,
                    ha="right",
                    fontsize=7.5,
                )
            else:
                axis.set_xticklabels([])
            if seed_index == 0:
                axis.set_yticklabels([_display(name) for name in class_names], fontsize=7.8)
                axis.set_ylabel(_display(model), fontweight="bold", labelpad=9)
            else:
                axis.set_yticklabels([])
            axis.set_title(f"Seed {seed}", fontsize=9.5, fontweight="bold")
            axis.tick_params(length=0)
            for spine in axis.spines.values():
                spine.set_visible(False)
    if image is not None:
        colour_axis = figure.add_axes((0.93, 0.2, 0.012, 0.62))
        colourbar = figure.colorbar(image, cax=colour_axis)
        colourbar.set_label("Patients", fontsize=8.5)
        colourbar.ax.tick_params(labelsize=7.5)
    figure.suptitle(
        "Patient-level 3×4 operational confusion by held-out seed",
        y=0.975,
        fontsize=14,
        fontweight="bold",
    )
    figure.text(0.5, 0.078, "Predicted class", ha="center", fontsize=10, fontweight="bold")
    figure.text(
        0.012,
        0.52,
        "Reference-standard class",
        ha="left",
        va="center",
        rotation=90,
        fontsize=10,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.012,
        "Counts are shown separately for every model and seed; no confusion cells are pooled across overlapping holdouts.",
        ha="center",
        va="bottom",
        fontsize=8.3,
    )
    figure.subplots_adjust(top=0.91, bottom=0.16, left=0.105, right=0.91, hspace=0.32, wspace=0.12)
    return figure, {
        "matrix_shape": [3, 4],
        "rows": "reference_standard_class",
        "columns": [*class_names, "abstain"],
        "panel_unit": "model_x_seed",
        "pooling": "none",
        "matrices": {
            f"{model}|{seed}": matrices[(model, seed)].tolist()
            for model in models
            for seed in seeds
        },
    }


def _plot_segmentation(
    segmentation: pd.DataFrame,
    models: Sequence[str],
    seeds: Sequence[int],
    class_names: Sequence[str],
) -> tuple[Figure, dict[str, Any]]:
    groups = [(model, class_index) for model in models for class_index in range(3)]
    labels = [f"{_display(model)} — {_display(class_names[class_index])}" for model, class_index in groups]
    figure, axes = plt.subplots(1, 2, figsize=(13.0, max(6.0, 0.42 * len(groups) + 2.6)))
    metrics = (("dice", "Frame Dice"), ("roi_coverage", "Strict predicted-ROI coverage"))
    y = np.arange(len(groups), dtype=float)
    offsets = np.linspace(-0.13, 0.13, len(seeds)) if len(seeds) > 1 else np.zeros(1)
    summary: list[dict[str, Any]] = []
    for axis, (column, label) in zip(axes, metrics, strict=True):
        for group_index, (model, class_index) in enumerate(groups):
            subset = segmentation.loc[
                (segmentation["model"] == model)
                & (segmentation["class_label"] == class_index)
            ].set_index("seed")
            values = np.asarray([float(subset.loc[seed, column]) for seed in seeds])
            finite = values[np.isfinite(values)]
            if len(finite):
                minimum, maximum, mean = float(finite.min()), float(finite.max()), float(finite.mean())
                axis.hlines(
                    y[group_index],
                    minimum,
                    maximum,
                    color=_CLASS_COLOURS[class_index],
                    linewidth=3,
                    alpha=0.55,
                    zorder=1,
                )
                axis.scatter(
                    mean,
                    y[group_index],
                    marker="D",
                    s=42,
                    facecolor=_CLASS_COLOURS[class_index],
                    edgecolor="#222222",
                    linewidth=0.6,
                    zorder=4,
                )
            else:
                minimum = maximum = mean = None
                axis.text(0.02, y[group_index], "Not estimable", va="center", fontsize=7.5)
            for seed_index, value in enumerate(values):
                if not np.isfinite(value):
                    continue
                axis.scatter(
                    value,
                    y[group_index] + offsets[seed_index],
                    marker=_SEED_MARKERS[seed_index % len(_SEED_MARKERS)],
                    s=28,
                    facecolor="white",
                    edgecolor=_CLASS_COLOURS[class_index],
                    linewidth=1.2,
                    zorder=3,
                )
            summary.append(
                {
                    "model": model,
                    "class_label": class_index,
                    "metric": column,
                    "seed_values": [None if not np.isfinite(value) else float(value) for value in values],
                    "arithmetic_mean": mean,
                    "minimum": minimum,
                    "maximum": maximum,
                }
            )
        axis.set_xlim(0, 1)
        axis.set_xticks(np.linspace(0, 1, 6))
        axis.set_yticks(y, labels=labels)
        axis.invert_yaxis()
        axis.set_xlabel(label)
        axis.grid(True, axis="x", color="#D8D8D8", linewidth=0.7)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0, labelsize=8)
    class_handles = [
        Line2D(
            [0], [0], color=_CLASS_COLOURS[index], lw=3, label=_display(class_names[index])
        )
        for index in range(3)
    ]
    seed_handles = [
        Line2D(
            [0],
            [0],
            marker=_SEED_MARKERS[index % len(_SEED_MARKERS)],
            markerfacecolor="white",
            markeredgecolor="#333333",
            linestyle="None",
            label=f"Seed {seed}",
        )
        for index, seed in enumerate(seeds)
    ]
    figure.legend(
        handles=[*class_handles, *seed_handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.071),
        ncol=min(8, len(class_handles) + len(seed_handles)),
        frameon=False,
    )
    figure.suptitle(
        "Class-conditional frame segmentation by seed",
        y=0.978,
        fontsize=14,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.012,
        "Dots are held-out seeds; diamonds and bars show the descriptive arithmetic mean and observed range. "
        "Reference masks are used for retrospective evaluation only.",
        ha="center",
        va="bottom",
        fontsize=8.3,
    )
    figure.subplots_adjust(top=0.91, bottom=0.19, left=0.245, right=0.985, wspace=0.23)
    return figure, {
        "level": "frame",
        "summary": summary,
        "range": "observed_minimum_to_maximum_not_confidence_interval",
        "reference_masks": "retrospective_evaluation_only",
    }


def _render_metadata(extension: str, title: str, description: str) -> dict[str, Any]:
    if extension == "pdf":
        fixed_date = datetime(1970, 1, 1, tzinfo=timezone.utc)
        return {
            "Title": title,
            "Author": "threeclass_roi_study",
            "Subject": description,
            "Creator": "threeclass_roi_study.publication_figures",
            "Producer": "Matplotlib",
            "CreationDate": fixed_date,
            "ModDate": fixed_date,
        }
    if extension == "svg":
        return {
            "Title": title,
            "Description": description,
            "Creator": "threeclass_roi_study.publication_figures",
            "Date": "1970-01-01",
        }
    return {
        "Title": title,
        "Description": description,
        "Software": "threeclass_roi_study.publication_figures",
    }


def _save_figure(
    figure: Figure,
    figure_root: Path,
    slug: str,
    *,
    title: str,
    description: str,
) -> dict[str, dict[str, Any]]:
    figure_root.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, Any]] = {}
    for extension in FIGURE_FORMATS:
        destination = figure_root / f"{slug}.{extension}"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{slug}.", suffix=f".{extension}", dir=str(figure_root)
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            figure.savefig(
                temporary,
                format=extension,
                dpi=FIGURE_DPI,
                bbox_inches="tight",
                pad_inches=0.08,
                facecolor="white",
                metadata=_render_metadata(extension, title, description),
            )
            temporary.replace(destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        records[extension] = {
            "path": str(destination.resolve()),
            "sha256": sha256_file(destination),
            "size_bytes": int(destination.stat().st_size),
        }
    return records


def _renderer_rc() -> dict[str, Any]:
    return {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "legend.title_fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "threeclass-roi-q1-v1",
    }


def generate_q1_figures(cfg: Mapping[str, Any]) -> dict[str, Path]:
    """Generate all Q1 figures and their hash manifest.

    Returns a flat mapping from ``<figure>.<format>`` to the corresponding
    absolute path plus ``q1_figure_manifest.json``.  Source tables are verified
    against the finalized publication-output manifest before anything is drawn.
    """

    tables, sources = _read_publication_sources(cfg)
    models, seeds, primary_strategy = _study_axes(cfg)
    class_names = _class_names(cfg)
    figure_root = output_root(cfg) / "figures"
    figure_records: dict[str, dict[str, Any]] = {}
    returned: dict[str, Path] = {}

    specifications: list[
        tuple[
            str,
            str,
            str,
            Sequence[str],
            Any,
        ]
    ] = [
        (
            "q1_patient_ovr_roc_pr",
            "Patient-level calibrated one-vs-rest ROC and precision-recall",
            "Seed-specific one-vs-rest ROC and precision-recall traces among non-abstained patients.",
            ("patient_predictions.csv", "calibration_metrics.csv"),
            lambda: _plot_roc_pr(
                tables["patient_predictions.csv"], models, seeds, class_names
            ),
        ),
        (
            "q1_patient_reliability",
            "Patient-level classwise reliability",
            "Seed-specific fixed-width classwise reliability diagrams among non-abstained patients.",
            ("patient_predictions.csv", "calibration_metrics.csv"),
            lambda: _plot_reliability(
                tables["patient_predictions.csv"], models, seeds, class_names
            ),
        ),
        (
            "q1_patient_risk_coverage",
            "Patient failure-aware risk-coverage",
            "Seed-specific confidence-ranked error versus intended-cohort coverage, with abstentions as errors.",
            ("risk_coverage.csv",),
            lambda: _plot_risk_coverage(tables["risk_coverage.csv"], models, seeds),
        ),
        (
            "q1_failure_aware_ba_coverage",
            "Patient failure-aware balanced accuracy and coverage",
            "Per-seed points with descriptive arithmetic mean and observed range; bars are not confidence intervals.",
            ("patient_metrics.csv",),
            lambda: _plot_per_seed_point_range(tables["patient_metrics.csv"], models, seeds),
        ),
        (
            "q1_patient_confusion_3x4_by_seed",
            "Patient-level 3x4 operational confusion",
            "Separate 3x4 count matrix for every model and seed, including operational abstention.",
            ("patient_confusion_3x4.csv", "patient_predictions.csv"),
            lambda: _plot_confusion(
                tables["patient_confusion_3x4.csv"], models, seeds, class_names
            ),
        ),
    ]
    if "segmentation_by_class.csv" in tables:
        specifications.append(
            (
                "q1_segmentation_class_conditional",
                "Class-conditional frame segmentation",
                "Per-seed frame Dice and strict predicted-ROI coverage by model and diagnostic class.",
                ("segmentation_by_class.csv",),
                lambda: _plot_segmentation(
                    tables["segmentation_by_class.csv"], models, seeds, class_names
                ),
            )
        )

    with matplotlib.rc_context(_renderer_rc()):
        for slug, title, definition, source_tables, builder in specifications:
            figure, analytic_details = builder()
            try:
                outputs = _save_figure(
                    figure,
                    figure_root,
                    slug,
                    title=title,
                    description=definition,
                )
            finally:
                plt.close(figure)
            figure_records[slug] = {
                "title": title,
                "definition": definition,
                "source_tables": list(source_tables),
                "outputs": outputs,
                "analytic_details": analytic_details,
                "accessibility": {
                    "class_encoding": "colour_plus_direct_panel_or_legend_label",
                    "seed_encoding": "line_style_or_marker_plus_legend_label",
                    "model_encoding": "direct_panel_or_axis_label",
                    "font_family": "DejaVu Sans",
                },
            }
            for extension, record in outputs.items():
                returned[f"{slug}.{extension}"] = Path(record["path"])

    skipped_optional: dict[str, str] = {}
    if "segmentation_by_class.csv" not in tables:
        skipped_optional["q1_segmentation_class_conditional"] = (
            "segmentation_by_class.csv was not present in the finalized publication manifest"
        )
    manifest = {
        "schema_version": FIGURE_SCHEMA_VERSION,
        "study_id": cfg.get("study_id"),
        "config_sha256": _config_identity(cfg),
        "generation_role": "downstream_static_rendering_only_no_training_selection_refitting_or_test_pooling",
        "renderer_code_sha256": sha256_file(Path(__file__)),
        "rendering": {
            "formats": list(FIGURE_FORMATS),
            "raster_dpi": FIGURE_DPI,
            "font_family": "DejaVu Sans",
            "svg_text_preserved": True,
            "fixed_metadata_date": "1970-01-01",
            "randomness": "none",
        },
        "analysis_scope": {
            "level": "patient",
            "probability_scale": "calibrated",
            "classifier_strategy": primary_strategy,
            "models": list(models),
            "seeds": [int(seed) for seed in seeds],
            "classes": [
                {"label": index, "name": class_names[index]} for index in range(3)
            ],
        },
        "disclosures": list(_COMMON_DISCLOSURES),
        "sources": sources,
        "figures": figure_records,
        "skipped_optional_figures": skipped_optional,
    }
    manifest_path = save_json_atomic(figure_root / FIGURE_MANIFEST_NAME, manifest)
    returned[FIGURE_MANIFEST_NAME] = manifest_path.resolve()
    return returned


def _validate_png(path: Path) -> None:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            dpi = image.info.get("dpi")
            if not dpi or any(abs(float(value) - FIGURE_DPI) > 1 for value in dpi[:2]):
                raise PublicationFigureError(f"PNG is not tagged at {FIGURE_DPI} dpi: {path}")
            if image.width < 600 or image.height < 450:
                raise PublicationFigureError(f"PNG dimensions are unexpectedly small: {path}")
    except PublicationFigureError:
        raise
    except Exception as exc:
        raise PublicationFigureError(f"Unreadable PNG: {path}") from exc


def validate_q1_figures(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Verify source immutability, figure completeness, hashes, and PNG DPI."""

    _, current_sources = _read_publication_sources(cfg)
    models, seeds, primary_strategy = _study_axes(cfg)
    figure_root = output_root(cfg) / "figures"
    manifest_path = figure_root / FIGURE_MANIFEST_NAME
    if not manifest_path.is_file():
        raise PublicationFigureError(f"Missing Q1 figure manifest: {manifest_path}")
    try:
        manifest = read_json(manifest_path)
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicationFigureError("Q1 figure manifest is unreadable") from exc
    if manifest.get("schema_version") != FIGURE_SCHEMA_VERSION:
        raise PublicationFigureError("Q1 figure manifest schema version mismatch")
    if manifest.get("study_id") != cfg.get("study_id"):
        raise PublicationFigureError("Q1 figure manifest study_id mismatch")
    if manifest.get("config_sha256") != _config_identity(cfg):
        raise PublicationFigureError("Q1 figure manifest config hash mismatch")
    if manifest.get("renderer_code_sha256") != sha256_file(Path(__file__)):
        raise PublicationFigureError("Q1 figures were produced by a different renderer revision")
    scope = manifest.get("analysis_scope", {})
    if (
        scope.get("level") != "patient"
        or scope.get("probability_scale") != "calibrated"
        or scope.get("classifier_strategy") != primary_strategy
        or scope.get("models") != models
        or scope.get("seeds") != seeds
    ):
        raise PublicationFigureError("Q1 figure analysis scope differs from the config")
    if manifest.get("disclosures") != list(_COMMON_DISCLOSURES):
        raise PublicationFigureError("Q1 figure disclosures are incomplete or changed")

    recorded_sources = manifest.get("sources")
    if not isinstance(recorded_sources, Mapping) or set(recorded_sources) != set(current_sources):
        raise PublicationFigureError("Q1 figure source inventory differs from finalized sources")
    for name, current in current_sources.items():
        recorded = recorded_sources.get(name)
        if not isinstance(recorded, Mapping):
            raise PublicationFigureError(f"Missing figure source record: {name}")
        for field in ("path", "sha256", "size_bytes"):
            if recorded.get(field) != current.get(field):
                raise PublicationFigureError(f"Q1 figure source {field} mismatch: {name}")
        if "rows" in current and recorded.get("rows") != current.get("rows"):
            raise PublicationFigureError(f"Q1 figure source row-count mismatch: {name}")

    figures = manifest.get("figures")
    if not isinstance(figures, Mapping):
        raise PublicationFigureError("Q1 figure manifest lacks figure records")
    required = set(_REQUIRED_FIGURES)
    if "segmentation_by_class.csv" in current_sources:
        required.add("q1_segmentation_class_conditional")
    if set(figures) != required:
        raise PublicationFigureError(
            f"Q1 figure set mismatch; missing={sorted(required - set(figures))}, "
            f"unexpected={sorted(set(figures) - required)}"
        )
    output_count = 0
    paths: dict[str, str] = {}
    for slug in sorted(figures):
        figure = figures[slug]
        if not isinstance(figure, Mapping) or not figure.get("definition") or not figure.get("title"):
            raise PublicationFigureError(f"Figure definition is missing: {slug}")
        source_tables = figure.get("source_tables")
        if not isinstance(source_tables, list) or not source_tables:
            raise PublicationFigureError(f"Figure source-table mapping is missing: {slug}")
        if any(name not in current_sources for name in source_tables):
            raise PublicationFigureError(f"Figure references an untracked source: {slug}")
        outputs = figure.get("outputs")
        if not isinstance(outputs, Mapping) or set(outputs) != set(FIGURE_FORMATS):
            raise PublicationFigureError(f"Figure format set is incomplete: {slug}")
        for extension in FIGURE_FORMATS:
            expected_path = (figure_root / f"{slug}.{extension}").resolve()
            record = outputs[extension]
            if not isinstance(record, Mapping) or Path(str(record.get("path", ""))).resolve() != expected_path:
                raise PublicationFigureError(f"Figure path mismatch: {slug}.{extension}")
            if not expected_path.is_file():
                raise PublicationFigureError(f"Figure output is missing: {expected_path}")
            if (
                sha256_file(expected_path) != record.get("sha256")
                or expected_path.stat().st_size != record.get("size_bytes")
            ):
                raise PublicationFigureError(f"Figure hash/size mismatch: {expected_path}")
            if extension == "png":
                _validate_png(expected_path)
            elif extension == "pdf" and expected_path.read_bytes()[:5] != b"%PDF-":
                raise PublicationFigureError(f"Malformed PDF: {expected_path}")
            elif extension == "svg" and "<svg" not in expected_path.read_text(
                encoding="utf-8", errors="strict"
            )[:2000]:
                raise PublicationFigureError(f"Malformed SVG: {expected_path}")
            paths[f"{slug}.{extension}"] = str(expected_path)
            output_count += 1
    return {
        "status": "passed",
        "schema_version": FIGURE_SCHEMA_VERSION,
        "figures": len(figures),
        "outputs": output_count,
        "manifest_sha256": sha256_file(manifest_path),
        "paths": paths,
    }


__all__ = [
    "FIGURE_DPI",
    "FIGURE_FORMATS",
    "FIGURE_MANIFEST_NAME",
    "PublicationFigureError",
    "generate_q1_figures",
    "validate_q1_figures",
]
