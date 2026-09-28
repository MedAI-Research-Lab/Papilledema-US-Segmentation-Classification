"""Create descriptive segmentation supplements from the two locked run summaries."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path
import os
import shutil

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SEEDS = (17, 42, 2026, 3407, 9103)
COLORS = {
    17: "#0072B2",
    42: "#D55E00",
    2026: "#009E73",
    3407: "#CC79A7",
    9103: "#E69F00",
}
MODELS = {
    "yolo26": "YOLO26",
    "vit_method2": "ViT Method2",
    "emcad": "EMCAD",
    "sam2_unet": "SAM2-U-Net",
}
CLASS_NAMES = ("Normal", "Papilledema", "Pseudopapilledema")
CLASS_SOURCE = ROOT / "threeclass_roi_results_4model_v1_0_0/tables/segmentation_per_seed_class.csv"
POST_SOURCE = ROOT / "strict_roi_results_4model_v1_2_0/summary/segmentation_per_seed.csv"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "svg.fonttype": "none",
        "savefig.dpi": 300,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#777777",
        "axes.linewidth": 0.7,
        "grid.color": "#dddddd",
        "grid.linewidth": 0.6,
    }
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(values: list[float]) -> tuple[float, float]:
    assert len(values) == 5
    return statistics.mean(values), statistics.stdev(values)


def figure_shell(model: str, subtitle: str):
    fig, axes = plt.subplots(3, 1, figsize=(7.1, 9.0))
    fig.subplots_adjust(left=0.16, right=0.95, top=0.824, bottom=0.125, hspace=0.75)
    fig.text(0.5, 0.980, MODELS[model], ha="center", va="top", fontsize=11, fontweight="bold")
    fig.text(0.5, 0.957, subtitle, ha="center", va="top", fontsize=11)
    handles = [
        Line2D([], [], color=COLORS[seed], marker="o", markersize=5, linewidth=1, label=f"Seed {seed}")
        for seed in SEEDS
    ]
    handles.append(Line2D([], [], color="#1a1a1a", marker="D", markersize=5, linewidth=1.3, label="Mean ± sample SD"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.932), ncol=3, frameon=False, handlelength=1.9, columnspacing=1.6)
    return fig, axes


def paired_panel(ax, values: list[list[float]], labels: tuple[str, str], title: str,
                 ylabel: str, limits: tuple[float, float]):
    assert len(values) == 5 and all(len(row) == 2 for row in values)
    for offset, (seed, pair) in enumerate(zip(SEEDS, values)):
        jitter = (offset - 2) * 0.018
        x = (0 + jitter - 0.065, 1 + jitter - 0.065)
        ax.plot(x, pair, color=COLORS[seed], alpha=0.28, linewidth=1.15, zorder=2)
        ax.scatter(x, pair, color=COLORS[seed], s=28, alpha=0.86, edgecolors="white", linewidths=0.45, zorder=3)
    stats = [summary([row[index] for row in values]) for index in range(2)]
    ax.errorbar([0.115, 1.115], [item[0] for item in stats], yerr=[item[1] for item in stats],
                fmt="D", markersize=5, color="#191919", elinewidth=1.3, capsize=4, zorder=4)
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(*limits)
    ax.set_xticks([0, 1], [f"{label}\n{mean:.3f} ± {sd:.3f}" for label, (mean, sd) in zip(labels, stats)])
    ax.tick_params(axis="x", length=0, pad=6)
    ax.set_title(title, loc="left", fontweight="bold", pad=9)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", zorder=0)
    return {label: {"mean": mean, "sample_sd": sd, "seed_values": [row[index] for row in values]}
            for index, (label, (mean, sd)) in enumerate(zip(labels, stats))}


def save(fig, identifier: str) -> tuple[str, str]:
    png, svg = OUT / f"{identifier}.png", OUT / f"{identifier}.svg"
    fig.savefig(png, dpi=300)
    fig.savefig(svg)
    plt.close(fig)
    return str(png), str(svg)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = (CLASS_SOURCE, POST_SOURCE)
    hashes_before = {str(path): sha256(path) for path in sources}
    class_rows = read_csv(CLASS_SOURCE)
    post_rows = [row for row in read_csv(POST_SOURCE)
                 if row["classifier_strategy"] == "model_specific" and row["level"] == "frame" and row["scope"] == "ALL"]
    assert len(post_rows) == 20
    index, checks = [], {}
    for model in MODELS:
        identifier = f"S2_{model}_diagnosis_stratified_segmentation"
        fig, axes = figure_shell(model, "Segmentation by reference diagnosis")
        model_checks = {}
        for class_index, (class_name, ax) in enumerate(zip(CLASS_NAMES, axes)):
            selected = [row for row in class_rows if row["model"] == model and int(row["class_index"]) == class_index]
            assert len(selected) == 5 and {int(row["seed"]) for row in selected} == set(SEEDS)
            by_seed = {int(row["seed"]): row for row in selected}
            rows = [by_seed[seed] for seed in SEEDS]
            expected_frames = 140 if class_index == 0 else 56
            assert all(int(row["frames"]) == expected_frames for row in rows)
            for row in rows:
                assert abs(float(row["roi_coverage"]) - (int(row["frames"]) - int(row["roi_failures"])) / int(row["frames"])) < 1e-12
            values = [[float(row["dice_all_mean"]), float(row["roi_coverage"])] for row in rows]
            assert all(0 <= value <= 1 for pair in values for value in pair)
            panel = paired_panel(ax, values, ("Dice", "ROI coverage"),
                                 f"{chr(65 + class_index)}  {class_name} ({expected_frames} frames per split)",
                                 "Proportion", (-0.04, 1.08))
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            model_checks[class_name] = panel
        fig.text(0.16, 0.066, "All intended test frames; invalid masks contribute zero Dice.\nThe anatomical ROI is shared across diagnoses. Lines connect each seed.",
                 fontsize=10, va="top", linespacing=1.5)
        caption = (
            f"Exploratory diagnosis-stratified anatomical segmentation for {MODELS[model]}. "
            "Panels A–C show normal, papilledema and pseudopapilledema strata. Colored points show "
            "the five split-specific all-intended-frame Dice means and strict predicted-ROI coverage; "
            "transparent lines connect the two measurements from the same seed and do not denote "
            "a causal relationship. Black diamonds and whiskers show the arithmetic mean ± sample SD "
            "across five splits (not a confidence interval); values beneath each category repeat that summary. "
            "Each split includes 140 normal, 56 papilledema and 56 pseudopapilledema frames. Invalid "
            "predictions contribute zero Dice and remain in coverage denominators. One shared anatomical "
            "ROI was learned across diagnoses; these retrospective strata do not represent diagnosis-specific "
            "lesion segmentation. Test memberships overlap, so summaries are descriptive rather than "
            "independent-sample inference."
        )
        png, svg = save(fig, identifier)
        index.append({"id": identifier, "category": "segmentation_class", "model": model, "png": png, "svg": svg, "caption": caption})
        checks[identifier] = model_checks

        identifier = f"S3_{model}_raw_postprocessed_segmentation"
        fig, axes = figure_shell(model, "Raw and postprocessed segmentation")
        selected = [row for row in post_rows if row["model"] == model]
        assert len(selected) == 5 and {int(row["seed"]) for row in selected} == set(SEEDS)
        by_seed = {int(row["seed"]): row for row in selected}
        rows = [by_seed[seed] for seed in SEEDS]
        assert all(int(row["n_units"]) == 252 for row in rows)
        model_checks = {}
        for panel_index, (metric, title, ylabel, limits) in enumerate((
            ("dice", "Dice", "Dice", (-0.04, 1.08)),
            ("iou", "Intersection over union", "IoU", (-0.04, 1.08)),
            ("hausdorff95", "95th-percentile Hausdorff distance", "HD95 (pixels)", (0, 55)),
        )):
            ax = axes[panel_index]
            values = [[float(row[f"raw_{metric}"]), float(row[metric])] for row in rows]
            if metric in ("dice", "iou"):
                assert all(int(row[f"{key}.n_nonmissing"]) == 252 for row in rows for key in (f"raw_{metric}", metric))
            panel = paired_panel(ax, values, ("Raw", "Postprocessed"),
                                 f"{chr(65 + panel_index)}  {title}", ylabel, limits)
            if metric in ("dice", "iou"):
                ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            else:
                assert max(value for pair in values for value in pair) < 55
                ax.set_yticks([0, 10, 20, 30, 40, 50])
            panel["denominators"] = {"Raw": [int(row[f"raw_{metric}.n_nonmissing"]) for row in rows],
                                     "Postprocessed": [int(row[f"{metric}.n_nonmissing"]) for row in rows]}
            model_checks[metric] = panel
        fig.text(0.16, 0.069, "Dice / IoU include all 252 frames per split; HD95 uses nonempty masks.\nRaw / postprocessed HD95 populations may differ. Lines pair split summaries.\nDescriptive comparison; no raw-versus-postprocessing hypothesis test.",
                 fontsize=10, va="top", linespacing=1.5)
        raw_n = model_checks["hausdorff95"]["denominators"]["Raw"]
        post_n = model_checks["hausdorff95"]["denominators"]["Postprocessed"]
        caption = (
            f"Descriptive raw-versus-postprocessed anatomical segmentation for {MODELS[model]}. "
            "Panels A–C show Dice, IoU and HD95. Colored points are split-specific means, and transparent "
            "lines pair summaries from the same seed. Black diamonds and whiskers show mean ± sample SD "
            "across the five split means, not confidence intervals; values beneath each category repeat "
            "that summary. Dice and IoU retain all 252 intended frames per split, including invalid "
            "predictions scored as empty masks. HD95 is conditional on nonempty predictions and measured "
            "in pixels in the resized 768 × 768 analysis space. Raw and postprocessed HD95 may use different "
            "nonempty-mask subsets, so connected points do not imply paired valid-frame populations. "
            f"HD95 nonmissing counts for seeds 17, 42, 2026, 3407 and 9103 are {raw_n} (raw) and "
            f"{post_n} (postprocessed). Test memberships overlap; this is descriptive boundary/overlap "
            "assessment, and no raw-versus-postprocessed hypothesis test was prespecified."
        )
        png, svg = save(fig, identifier)
        index.append({"id": identifier, "category": "segmentation_postprocess", "model": model, "png": png, "svg": svg, "caption": caption})
        checks[identifier] = model_checks
    hashes_after = {str(path): sha256(path) for path in sources}
    assert hashes_before == hashes_after
    (OUT / "figure_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    audit = {"source_sha256": hashes_before, "source_unchanged_after_generation": True,
             "seeds": SEEDS, "seed_colors": COLORS, "model_order": list(MODELS),
             "figure_inches": [7.1, 9], "png_dpi": 300, "png_expected_pixels": [2130, 2700],
             "font_sizes_pt": [10, 11], "summary_definition": "arithmetic mean and sample SD (ddof=1) of five split-specific means",
             "primary_filters": {"segmentation_class": "all intended frames within diagnosis; dice_all_mean and roi_coverage",
                                 "segmentation_postprocess": "classifier_strategy=model_specific, level=frame, scope=ALL"},
             "plot_values_and_denominators": checks,
             "output_sha256": {str(path): sha256(path) for entry in index for path in (Path(entry["png"]), Path(entry["svg"]))},
             "visual_qa": "Pending inspection of all eight PNGs."}
    (OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"figures": len(index), "png": len(index), "svg": len(index), "index": str(OUT / "figure_index.json")}))


if __name__ == "__main__":
    main()
