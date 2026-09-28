"""Plot exact saved binary-eye aggregate coordinates; never run inference.

SVG coordinates use points, preserving 11-pt typography at 7.1-inch width.
Only files beneath work/new_figures are generated. Source CSVs are read-only.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import subprocess
from pathlib import Path
import os
import shutil


WORK = Path(__file__).resolve().parent
ROOT = WORK.parent.parent
OUT = WORK / "new_figures"
RUNS = ROOT / "strict_roi_results_4model_v1_2_0" / "runs"
NODE = Path(os.environ.get("PUBLICATION_NODE", shutil.which("node") or "node"))
SHARP = Path(os.environ.get("PUBLICATION_SHARP", "sharp"))
MODELS = [("yolo26", "YOLO26"), ("vit_method2", "ViT Method2"), ("emcad", "EMCAD"), ("sam2_unet", "SAM2-U-Net")]
SEEDS = [17, 42, 2026, 3407, 9103]
STYLES = [
    ("#1f77b4", "", "circle"),
    ("#ff7f0e", "5 3", "square"),
    ("#2ca02c", "5 2 1 2", "triangle"),
    ("#d62728", "1 2.5", "diamond"),
    ("#9467bd", "8 3", "plus"),
]
WIDTH = 511.2
RELIABILITY_HEIGHT = 540.0
RISK_HEIGHT = 410.4


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def save_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def text(x: float, y: float, value: str, anchor: str = "start", bold: bool = False, rotation: float | None = None) -> str:
    transform = f' transform="rotate({rotation:g} {x:g} {y:g})"' if rotation is not None else ""
    weight = ' font-weight="bold"' if bold else ""
    return f'<text x="{x:g}" y="{y:g}" text-anchor="{anchor}"{weight}{transform}>{html.escape(value)}</text>'


def line(x1: float, y1: float, x2: float, y2: float, color: str, width: float = 1.1, dash: str = "") -> str:
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" stroke="{color}" stroke-width="{width:g}"{d}/>'


def marker(x: float, y: float, shape: str, color: str, radius: float = 2.6) -> str:
    common = f' fill="white" stroke="{color}" stroke-width="1.1"'
    if shape == "circle":
        return f'<circle cx="{x:.6f}" cy="{y:.6f}" r="{radius:g}"{common}/>'
    if shape == "square":
        return f'<rect x="{x-radius:.6f}" y="{y-radius:.6f}" width="{2*radius:g}" height="{2*radius:g}"{common}/>'
    if shape == "triangle":
        coords = [(x, y-radius-0.5), (x+radius+0.4, y+radius), (x-radius-0.4, y+radius)]
    elif shape == "diamond":
        coords = [(x, y-radius-0.5), (x+radius+0.5, y), (x, y+radius+0.5), (x-radius-0.5, y)]
    elif shape == "plus":
        return line(x-radius-0.4, y, x+radius+0.4, y, color) + line(x, y-radius-0.4, x, y+radius+0.4, color)
    else:
        raise ValueError(shape)
    points = " ".join(f"{a:.6f},{b:.6f}" for a, b in coords)
    return f'<polygon points="{points}"{common}/>'


def header(height: float, title: str) -> list[str]:
    return [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH:g}pt" height="{height:g}pt" viewBox="0 0 {WIDTH:g} {height:g}">',
        '<style>text{font-family:Arial,Helvetica,sans-serif;font-size:11px;fill:#222}path,polyline,line{stroke-linejoin:round}</style>',
        f'<rect width="{WIDTH:g}" height="{height:g}" fill="white"/>',
        text(WIDTH / 2, 19, title, "middle", True),
    ]


def legend(y: float) -> list[str]:
    parts = []
    for index, seed in enumerate(SEEDS):
        x = 54 + index * 91
        color, dash, shape = STYLES[index]
        parts.extend([line(x, y-3, x+24, y-3, color, dash=dash), marker(x+12, y-3, shape, color), text(x+30, y, str(seed))])
    return parts


def axes(x: float, y: float, w: float, h: float, xlabel: str, ylabel: str, panel: str | None = None) -> list[str]:
    parts = []
    if panel:
        parts.append(text(x, y-13, panel, bold=True))
    for j in range(6):
        t = j / 5
        xx = x + t*w
        yy = y + (1-t)*h
        parts.extend([
            line(xx, y, xx, y+h, "#e5e5e5", 0.6),
            line(x, yy, x+w, yy, "#e5e5e5", 0.6),
            line(xx, y+h, xx, y+h+3.5, "#444", 0.7),
            line(x-3.5, yy, x, yy, "#444", 0.7),
            text(xx, y+h+17, f"{t:.1f}", "middle"),
            text(x-9, yy+3.5, f"{t:.1f}", "end"),
        ])
    parts.append(f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" fill="none" stroke="#555" stroke-width="0.7"/>')
    parts.append(text(x+w/2, y+h+37, xlabel, "middle"))
    parts.append(text(20, y+h/2, ylabel, "middle", rotation=-90))
    return parts


def trace(points: list[tuple[float, float]], box: tuple[float, float, float, float], style_index: int, every: int = 1) -> str:
    x, y, w, h = box
    color, dash, shape = STYLES[style_index]
    coords = [(x+px*w, y+(1-py)*h) for px, py in points]
    d = f' stroke-dasharray="{dash}"' if dash else ""
    polyline = f'<polyline points="{" ".join(f"{px:.6f},{py:.6f}" for px, py in coords)}" fill="none" stroke="{color}" stroke-width="1.1"{d}/>'
    marks = "".join(marker(px, py, shape, color) for i, (px, py) in enumerate(coords) if i % every == 0 or i == len(coords)-1)
    return polyline + marks


def num(row: dict[str, str], field: str) -> float:
    value = float(row[field])
    assert math.isfinite(value), (field, row)
    return value


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    coordinate_rows = []
    source_hashes = {}
    audit = {"source_filters": {"level": "eye", "scope": "ALL", "classifier_strategy": "model_specific", "reliability_strategy": "uniform", "reliability_n": ">0", "risk_failure_inclusive": "True"}, "models": {}, "dimensions_inches": {"S12": [7.1, 7.5], "S13": [7.1, 5.7]}, "font_pt": 11, "png_dpi": 300, "pooled_curves": False, "new_inference_or_statistics": False, "unit_identifiers_exported": False}
    figure_index = []
    for model, display in MODELS:
        reliability = {}
        risks = {}
        model_audit = {}
        for seed in SEEDS:
            folder = RUNS / model / f"seed_{seed}" / "evaluation" / "model_specific"
            cal_path = folder / "calibration_curves.csv"
            risk_path = folder / "risk_coverage_curves.csv"
            for source in (cal_path, risk_path):
                source_hashes[str(source.relative_to(ROOT)).replace("\\", "/")] = sha256(source)
            selected_cal = [r for r in csv_rows(cal_path) if r["level"] == "eye" and r["scope"] == "ALL" and r["classifier_strategy"] == "model_specific" and r["strategy"] == "uniform"]
            seed_audit = {"reliability": {}}
            for calibration in ("raw", "temperature_scaled"):
                all_bins = sorted([r for r in selected_cal if r["calibration"] == calibration], key=lambda r: int(r["bin"]))
                assert [int(r["bin"]) for r in all_bins] == list(range(10)), (model, seed, calibration)
                for j, row in enumerate(all_bins):
                    assert abs(num(row, "lower") - j/10) < 1e-12
                    assert abs(num(row, "upper") - (j+1)/10) < 1e-12
                    assert int(row["n"]) >= 0
                nonempty = [r for r in all_bins if int(r["n"]) > 0]
                assert nonempty
                for row in nonempty:
                    assert 0 <= num(row, "mean_probability") <= 1
                    assert 0 <= num(row, "observed_fraction") <= 1
                    coordinate_rows.append({"figure": "S12", "model": model, "seed": seed, "calibration": calibration, "level": "eye", "scope": "ALL", "classifier_strategy": "model_specific", "bin": row["bin"], "bin_lower": row["lower"], "bin_upper": row["upper"], "n": row["n"], "mean_probability": row["mean_probability"], "observed_fraction": row["observed_fraction"]})
                reliability[(seed, calibration)] = [(num(r, "mean_probability"), num(r, "observed_fraction")) for r in nonempty]
                n = sum(int(r["n"]) for r in nonempty)
                assert 0 < n <= 36
                seed_audit["reliability"][calibration] = {"n_evaluable": n, "nonempty_bins": len(nonempty)}
            assert seed_audit["reliability"]["raw"]["n_evaluable"] == seed_audit["reliability"]["temperature_scaled"]["n_evaluable"]
            curve = sorted([r for r in csv_rows(risk_path) if r["level"] == "eye" and r["scope"] == "ALL" and r["classifier_strategy"] == "model_specific" and r["failure_inclusive"].lower() == "true"], key=lambda r: int(r["rank"]))
            assert len(curve) == 36
            assert [int(r["rank"]) for r in curve] == list(range(1, 37))
            for rank, row in enumerate(curve, 1):
                assert abs(num(row, "coverage") - rank/36) < 1e-12
                assert 0 <= num(row, "risk") <= 1
                assert row["aurc"] == curve[0]["aurc"]
                coordinate_rows.append({"figure": "S13", "model": model, "seed": seed, "calibration": "operational", "level": "eye", "scope": "ALL", "classifier_strategy": "model_specific", "rank": row["rank"], "coverage": row["coverage"], "risk": row["risk"], "aurc": row["aurc"], "failure_inclusive": "True"})
            # Verification of the stored curve convention, not a new estimate.
            assert abs(sum(num(r, "risk") for r in curve)/36 - num(curve[0], "aurc")) < 1e-12
            assert num(curve[-1], "coverage") == 1
            risks[seed] = [(num(r, "coverage"), num(r, "risk")) for r in curve]
            seed_audit["risk"] = {"n_intended": 36, "n_curve_points": 36, "last_coverage": curve[-1]["coverage"], "last_risk": curve[-1]["risk"], "source_aurc": curve[0]["aurc"]}
            model_audit[str(seed)] = seed_audit

        cal_parts = header(RELIABILITY_HEIGHT, f"{display} — binary eye-level reliability")
        cal_parts.extend(legend(43))
        cal_parts.extend([line(199, 59, 226, 59, "#777", 1.0, "4 3"), text(233, 62, "Ideal calibration")])
        for calibration, panel, y in [("raw", "A  Raw probabilities", 87), ("temperature_scaled", "B  Temperature-scaled probabilities", 327)]:
            box = (74, y, 411, 158)
            cal_parts.extend(axes(*box, "Mean predicted probability of abnormal diagnosis", "Observed abnormal-class fraction", panel))
            cal_parts.append(line(box[0], y+158, box[0]+411, y, "#777", 1.0, "4 3"))
            for index, seed in enumerate(SEEDS):
                cal_parts.append(trace(reliability[(seed, calibration)], box, index))
        cal_parts.append("</svg>")
        cal_stem = f"S12_{model}_binary_eye_reliability"
        (OUT / f"{cal_stem}.svg").write_text("\n".join(cal_parts), encoding="utf-8")

        risk_parts = header(RISK_HEIGHT, f"{display} — binary eye-level risk–coverage")
        risk_parts.extend(legend(45))
        box = (74, 78, 411, 273)
        risk_parts.extend(axes(*box, "Coverage (all intended eyes)", "Failure-aware risk"))
        for index, seed in enumerate(SEEDS):
            risk_parts.append(trace(risks[seed], box, index, every=6))
        risk_parts.append("</svg>")
        risk_stem = f"S13_{model}_binary_eye_risk_coverage"
        (OUT / f"{risk_stem}.svg").write_text("\n".join(risk_parts), encoding="utf-8")
        figure_index.extend([
            {"number": "S12", "model": model, "name": display, "stem": cal_stem, "width_inches": 7.1, "height_inches": 7.5},
            {"number": "S13", "model": model, "name": display, "stem": risk_stem, "width_inches": 7.1, "height_inches": 5.7},
        ])
        audit["models"][model] = model_audit

    columns = ["figure", "model", "seed", "calibration", "level", "scope", "classifier_strategy", "bin", "bin_lower", "bin_upper", "n", "mean_probability", "observed_fraction", "rank", "coverage", "risk", "aurc", "failure_inclusive"]
    assert "unit_index" not in columns and "confidence" not in columns
    save_csv(OUT / "binary_curve_coordinates.csv", coordinate_rows, columns)
    # A round trip must preserve every retained aggregate numeric source string.
    back = csv_rows(OUT / "binary_curve_coordinates.csv")
    assert len(back) == len(coordinate_rows)
    for source, exported in zip(coordinate_rows, back):
        for column, value in source.items():
            assert str(value) == exported[column], (column, value, exported[column])
    for rel, digest in source_hashes.items():
        assert sha256(ROOT / rel) == digest, f"Source changed: {rel}"
    audit["source_sha256"] = source_hashes
    audit["coordinate_rows"] = len(coordinate_rows)
    audit["calibration_coordinate_rows"] = sum(r["figure"] == "S12" for r in coordinate_rows)
    audit["risk_coordinate_rows"] = sum(r["figure"] == "S13" for r in coordinate_rows)
    audit["all_source_and_coordinate_checks_passed"] = True
    # Render vector geometry at 300 dpi with the bundled sharp, not a desktop app.
    node_code = """const fs=require('fs'); const path=require('path'); const sharp=require(process.argv[1]); const folder=process.argv[2]; (async()=>{const info=[]; for(const name of fs.readdirSync(folder).filter(n=>/^S1[23]_.*[.]svg$/.test(n)).sort()){const dest=path.join(folder,name.slice(0,-4)+'.png'); await sharp(path.join(folder,name),{density:300}).resize({width:2130}).flatten({background:'#ffffff'}).withMetadata({density:300}).png().toFile(dest); const m=await sharp(dest).metadata(); if(m.width!==2130||m.density!==300)throw Error('Unexpected dimensions/density '+name); info.push({file:path.basename(dest),width:m.width,height:m.height,density:m.density});} console.log(JSON.stringify(info));})().catch(e=>{console.error(e);process.exit(1)});"""
    rendered = subprocess.run([str(NODE), "-e", node_code, str(SHARP), str(OUT)], capture_output=True, text=True)
    if rendered.returncode:
        raise RuntimeError(rendered.stderr)
    audit["png_metadata"] = json.loads(rendered.stdout)
    assert len(audit["png_metadata"]) == 8
    for info in audit["png_metadata"]:
        assert info["height"] == (2250 if "reliability" in info["file"] else 1710)
    (OUT / "figure_index.json").write_text(json.dumps(figure_index, indent=2), encoding="utf-8")
    (OUT / "plotting_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (OUT / "PLOTTING_AUDIT.md").write_text(
        "# Binary-eye supplementary plotting audit\n\n"
        "The eight figures reproduce saved CSV aggregate coordinates for all four model-specific pipelines and seeds 17, 42, 2026, 3407, and 9103. No model, calibrator, threshold, confidence interval, decision curve, or statistical comparison was fitted or recomputed.\n\n"
        "S12 selects eye / ALL / model_specific / uniform bins, separately for raw and temperature_scaled. There are ten fixed-width bins per source series; bins with n = 0 are omitted, never assigned a zero outcome. Points show the recorded mean abnormal-class probability and observed abnormal-class fraction. Raw and calibrated denominators reconcile within every model/seed. Lines connect nonempty-bin points as visual guides, not smoothed fits. The positive class combines papilledema and pseudopapilledema.\n\n"
        "S13 selects eye / ALL / model_specific / failure_inclusive = True. Every series has the 36 recorded attainable coverage steps, with no artificial zero-coverage point. Saved curves rank evaluable eyes by absolute probability distance from the locked decision threshold, append structural abstentions at lowest confidence, and count abstentions as errors. The CSV uses the operational prediction scale; the plotting script does not relabel it as a newly calculated scale. Terminal risk is incorrect-or-abstained / 36, not 1 minus balanced accuracy. Markers are drawn at every sixth source point and the endpoint for legibility; every coordinate remains in the connecting line and the aggregate export.\n\n"
        "All exported numeric strings were round-trip checked against selected source rows. Forty source files were SHA-256 checked before and after generation. The aggregate coordinate export excludes unit_index, confidence, participant IDs and eye identifiers. Separate traces retain dependent overlapping holdouts without pooled estimates.\n\n"
        "SVG typography is 11 pt at a physical width of 7.1 inches. S12 is 7.5 inches high; S13 is 5.7 inches high. PNGs are 2130 pixels wide at 300 dpi. Seed coding combines color, dash pattern and marker shape. All axes cover [0, 1]. Numerical checks passed; visual inspection is recorded separately.\n",
        encoding="utf-8",
    )
    print(json.dumps({"figures": len(figure_index), "coordinates": len(coordinate_rows), "output": str(OUT), "checks_passed": True}))


if __name__ == "__main__":
    main()
