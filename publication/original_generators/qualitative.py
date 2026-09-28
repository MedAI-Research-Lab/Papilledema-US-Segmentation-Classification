"""Render audited existing scientific masks; no fitting, inference, or mask editing.

Run with the bundled Codex Python. All writes are confined to this directory.
Detailed source paths contain personal names and are saved only under private/.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
import os
import shutil
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
RUN = ROOT / "strict_roi_results_4model_v1_2_0"
DATA = ROOT / "çalışma_ds"
SEED = 17
MODELS = ["yolo26", "vit_method2", "emcad", "sam2_unet"]
NAMES = {"yolo26": "YOLO26s-seg", "vit_method2": "ViT Method2", "emcad": "EMCAD", "sam2_unet": "SAM2-U-Net"}
KEYS = ["patient_id", "case_id", "side", "frame_id"]
CATEGORIES = ["best", "moderate", "worst"]
LABELS = {"best": "Best", "moderate": "Near median", "worst": "Worst"}
CROP = (0, 101, 768, 666)  # Remove only fixed letterbox padding, not anatomy.
CYAN = (0, 221, 242)
ORANGE = (255, 158, 56)
INK = (21, 34, 48)
MUTED = (80, 96, 112)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def pick_three(table: pd.DataFrame, value: str) -> dict[str, pd.Series]:
    """Choose endpoints first, then closest to full-cohort median; unique patients.

    Exact ties are resolved by patient, frame, eye, case, and fixed seed. In this
    dataset the patient-distinctness rule leaves both global extrema unchanged.
    """
    selected = {}
    used = set()
    median = float(table[value].median())
    tie = ["patient_id", "frame_id", "side", "case_id"]
    for category in ["best", "worst", "moderate"]:
        candidates = table.loc[~table.patient_id.isin(used)].copy()
        candidates["selection_distance"] = (
            -candidates[value] if category == "best"
            else candidates[value] if category == "worst"
            else (candidates[value] - median).abs()
        )
        row = candidates.sort_values(["selection_distance", *tie], kind="stable").iloc[0]
        selected[category] = row
        used.add(str(row.patient_id))
    assert len(used) == 3
    assert selected["best"][value] == table[value].max()
    assert selected["worst"][value] == table[value].min()
    assert abs(selected["moderate"][value] - median) == (table[value] - median).abs().min()
    return selected


def font(size: int, bold=False):
    return ImageFont.truetype(os.environ.get("PUBLICATION_ARIAL_BOLD", "Arial-Bold.ttf") if bold else os.environ.get("PUBLICATION_ARIAL_REGULAR", "Arial.ttf"), size)


class Canvas:
    """Identical PNG and embedded-raster SVG drawing, with vector SVG labels."""

    def __init__(self, width, height, title, description):
        self.width, self.height = width, height
        self.im = Image.new("RGB", (width, height), "white")
        self.draw = ImageDraw.Draw(self.im)
        self.svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width / 300:.6f}in" height="{height / 300:.6f}in" viewBox="0 0 {width} {height}" role="img">', f"<title>{escape(title)}</title><desc>{escape(description)}</desc>", f'<rect width="{width}" height="{height}" fill="white"/>']

    def text(self, x, y, text, size=36, color=INK, bold=False, anchor="la"):
        self.draw.text((x, y), text, font=font(size, bold), fill=color, anchor=anchor)
        css = f"rgb{color}"
        svg_anchor = "middle" if anchor.startswith("m") else "end" if anchor.startswith("r") else "start"
        # Pillow anchor 'a' begins at the font ascender; match SVG baseline.
        ascent, _ = font(size, bold).getmetrics()
        self.svg.append(f'<text x="{x}" y="{y + ascent}" font-family="Arial, sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" text-anchor="{svg_anchor}" fill="{css}">{escape(text)}</text>')

    def line(self, xy, color, width=3, dashed=False):
        x1, y1, x2, y2 = xy
        if dashed:
            for a in range(int(x1), int(x2), 16):
                self.draw.line((a, y1, min(a + 10, x2), y2), fill=color, width=width)
        else:
            self.draw.line(xy, fill=color, width=width)
        dash = ' stroke-dasharray="10 6"' if dashed else ""
        self.svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="rgb{color}" stroke-width="{width}"{dash}/>')

    def image(self, img, x, y, width, height):
        tile = img.resize((width, height), Image.Resampling.LANCZOS)
        self.im.paste(tile, (x, y))
        stream = io.BytesIO()
        tile.save(stream, format="PNG")
        encoded = base64.b64encode(stream.getvalue()).decode("ascii")
        self.svg.append(f'<image x="{x}" y="{y}" width="{width}" height="{height}" href="data:image/png;base64,{encoded}"/>')

    def save(self, stem):
        self.im.save(OUT / f"{stem}.png", dpi=(300, 300))
        (OUT / f"{stem}.svg").write_text("\n".join(self.svg + ["</svg>"]), encoding="utf-8")


def edge(mask):
    raster = Image.fromarray(mask.astype("uint8") * 255)
    return np.asarray(raster.filter(ImageFilter.MaxFilter(3))) != np.asarray(raster.filter(ImageFilter.MinFilter(3)))


def overlay(image_path: Path, gt: np.ndarray, prediction: np.ndarray):
    image = np.array(Image.open(image_path).convert("RGB"), copy=True)
    image[edge(gt)] = CYAN
    pred_edge = edge(prediction)
    yy, xx = np.indices(prediction.shape)
    pred_edge &= ((xx + yy) % 13) < 8
    image[pred_edge] = ORANGE
    return Image.fromarray(image).crop(CROP)


def legend(canvas, y):
    canvas.line((70, y + 21, 145, y + 21), CYAN, 7)
    canvas.text(160, y, "Reference", 34)
    canvas.line((375, y + 21, 450, y + 21), ORANGE, 7, dashed=True)
    canvas.text(465, y, "Prediction", 34)


def draw_matrix(stem, title, selections, tiles, shared):
    width, height = 2340, 2640
    canvas = Canvas(width, height, title, "Four models by three outcome-selected test frames; seed 17. Reference cyan, prediction orange dashed. Dice uses full 768 by 768 postprocessed masks.")
    canvas.text(65, 40, title, 56, bold=True)
    canvas.text(65, 114, "Test split: seed 17  |  Postprocessed masks  |  Outcome-selected examples", 34, color=MUTED)
    legend(canvas, 173)
    left, gap, tilew, tileh = 282, 24, 654, 481
    for j, category in enumerate(CATEGORIES):
        center = left + j * (tilew + gap) + tilew / 2
        canvas.text(center, 251, LABELS[category], 43, bold=True, anchor="ma")
        if shared:
            row = selections[(MODELS[0], category)]
            canvas.text(center, 307, f'{row["anonymous_patient"]}  |  mean Dice {row["cross_model_mean_dice"]:.3f}', 31, color=MUTED, anchor="ma")
    first_y = 368 if shared else 333
    row_height = 540
    for i, model in enumerate(MODELS):
        top = first_y + i * row_height
        canvas.text(45, top + 167, NAMES[model], 35, bold=True)
        for j, category in enumerate(CATEGORIES):
            row = selections[(model, category)]
            x = left + j * (tilew + gap)
            canvas.image(tiles[row["selection_id"]], x, top, tilew, tileh)
            # Metrics are placed inside a black header above image anatomy.
            label = f'Dice {row["dice"]:.3f}'
            if row["roi_invalid"]:
                label += "  |  Invalid ROI; empty prediction"
            elif not shared:
                label += f'  |  {row["anonymous_patient"]}'
            # Image display crop begins at anatomy, so labels use the interrow band.
            canvas.text(x + 2, top + tileh + 6, label, 34, color=INK)
    footer_y = 2573
    canvas.text(65, footer_y, "Descriptive examples selected by Dice; they do not estimate prevalence or comparative performance.", 29, color=MUTED)
    canvas.save(stem)


def draw_vertical(model, selections, tiles):
    width, height = 1450, 3630
    canvas = Canvas(width, height, f"{NAMES[model]}: best, near-median, and worst frames", "Three outcome-selected frames from distinct patients, test seed 17; saved postprocessed masks only.")
    canvas.text(70, 40, NAMES[model], 64, bold=True)
    canvas.text(70, 122, "Best, near-median and worst | Test seed 17", 39, color=MUTED)
    legend(canvas, 190)
    tilew, tileh = 1310, 964
    for j, category in enumerate(CATEGORIES):
        top = 285 + j * 1090
        row = selections[(model, category)]
        canvas.text(70, top, f'{LABELS[category]}  |  {row["anonymous_patient"]}  |  Dice {row["dice"]:.3f}', 45, bold=True)
        canvas.image(tiles[row["selection_id"]], 70, top + 70, tilew, tileh)
        if row["roi_invalid"]:
            canvas.text(70, top + 1038, "Invalid ROI; empty prediction", 32, color=MUTED)
    canvas.text(70, 3560, "Outcome-selected descriptive examples; not a representative sample.", 31, color=MUTED)
    canvas.save(f"supplement_{model}_best_median_worst_3x1")


def main():
    private = OUT / "private"
    private.mkdir(parents=True, exist_ok=True)
    cfg_path = RUN / "provenance/config_strict_roi.snapshot.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    manifest_path = DATA / "manifest.csv"
    assert sha(manifest_path) == cfg["dataset"]["manifest_sha256"]
    manifest = pd.read_csv(manifest_path, dtype={k: str for k in KEYS})
    source = manifest.set_index(KEYS)
    assert source.index.is_unique
    split_path = RUN / f"splits/seed_{SEED}_patients.csv"
    split = pd.read_csv(split_path)
    test_patients = set(split.loc[split.split == "test", "patient_id"])
    assert len(test_patients) == 18
    reference_cache, tables, sources = {}, {}, []
    verification = []
    summary_path = RUN / "summary/segmentation_per_seed.csv"
    summary = pd.read_csv(summary_path)
    for model in MODELS:
        frames_path = RUN / f"runs/{model}/seed_{SEED}/evaluation/model_specific/frames.csv"
        table = pd.read_csv(frames_path, dtype={k: str for k in KEYS})
        assert len(table) == 252 and set(table.patient_id) == test_patients
        assert not table.duplicated(KEYS).any()
        tables[model] = table
        checked_hashes, deltas = [], []
        for row in table.itertuples(index=False):
            key = tuple(getattr(row, k) for k in KEYS)
            original = source.loc[key]
            image_path, reference_path = DATA / original.output_image, DATA / original.output_mask
            if key not in reference_cache:
                assert sha(image_path) == original.output_image_sha256
                assert sha(reference_path) == original.output_mask_sha256
                reference_cache[key] = np.asarray(Image.open(reference_path)) > 0
            gt = reference_cache[key]
            audit_path = ROOT / row.audit_path
            assert RUN in audit_path.resolve().parents
            actual_hash = sha(audit_path)
            assert actual_hash == row.audit_sha256
            with np.load(audit_path, allow_pickle=False) as saved:
                pred = saved["selected_mask"].astype(bool)
            assert pred.shape == gt.shape == (768, 768)
            invalid = not bool(row.segmentation_roi_valid)
            assert not invalid or not pred.any()
            tp = int(np.count_nonzero(pred & gt))
            fp = int(np.count_nonzero(pred & ~gt))
            fn = int(np.count_nonzero(~pred & gt))
            recomputed = 2.0 * tp / (2 * tp + fp + fn)
            assert tp == int(row.pixel_tp) and fp == int(row.pixel_fp) and fn == int(row.pixel_fn)
            delta = abs(recomputed - float(row.dice))
            assert delta < 1e-12
            deltas.append(delta)
            checked_hashes.append(actual_hash)
        record = summary[(summary.model == model) & (summary.seed == SEED) & (summary.classifier_strategy == "model_specific") & (summary.level == "frame") & (summary.scope == "ALL")].iloc[0]
        assert abs(table.dice.mean() - record.dice) < 1e-12
        verification.append({"model": model, "seed": SEED, "n_frames": len(table), "n_patients": table.patient_id.nunique(), "n_invalid": int((~table.segmentation_roi_valid).sum()), "maximum_absolute_dice_difference": max(deltas), "source_summary_mean_dice": float(record.dice), "recomputed_mean_dice": float(table.dice.mean()), "frames_csv_path": rel(frames_path), "frames_csv_sha256": sha(frames_path), "audit_hashes_verified": len(checked_hashes)})
    wide = tables[MODELS[0]][KEYS].copy()
    for model, table in tables.items():
        wide = wide.merge(table[KEYS + ["dice"]].rename(columns={"dice": model}), on=KEYS, validate="one_to_one")
    wide["mean_dice"] = wide[MODELS].mean(axis=1)
    common = pick_three(wide, "mean_dice")
    extrema = {model: pick_three(table, "dice") for model, table in tables.items()}
    lookups = {m: t.set_index(KEYS) for m, t in tables.items()}
    anonymous, records, internal, tiles = {}, [], [], {}
    for panel, selection in [("shared_patients", common), ("per_model_extrema", extrema)]:
        for category in CATEGORIES:
            for model in MODELS:
                chosen = selection[category] if panel == "shared_patients" else selection[model][category]
                key = tuple(str(chosen[k]) for k in KEYS)
                row = lookups[model].loc[key]
                original = source.loc[key]
                if key[0] not in anonymous:
                    anonymous[key[0]] = f"Case {len(anonymous) + 1:02d}"
                median = float(wide.mean_dice.median()) if panel == "shared_patients" else float(tables[model].dice.median())
                selection_id = f"{panel}_{model}_{category}"
                record = {"selection_id": selection_id, "panel": panel, "model": model, "model_label": NAMES[model], "seed": SEED, "category": category, "anonymous_patient": anonymous[key[0]], "patient_id": key[0], "case_id": key[1], "side": key[2], "frame_id": key[3], "dice": float(row.dice), "cross_model_mean_dice": float(chosen.mean_dice) if panel == "shared_patients" else None, "selection_population_median": median, "roi_invalid": not bool(row.segmentation_roi_valid), "roi_invalid_reason": "" if pd.isna(row.segmentation_abstention_reason) else str(row.segmentation_abstention_reason), "predicted_mask_key": "selected_mask", "audit_path": str(row.audit_path), "audit_sha256": str(row.audit_sha256), "reference_mask_sha256": str(original.output_mask_sha256), "image_sha256": str(original.output_image_sha256), "outcome_selected": True}
                records.append(record)
                image_path, reference_path = DATA / original.output_image, DATA / original.output_mask
                internal.append({**record, "image_path": rel(image_path), "reference_mask_path": rel(reference_path)})
                with np.load(ROOT / row.audit_path, allow_pickle=False) as saved:
                    prediction = saved["selected_mask"].astype(bool)
                assert not prediction[:101].any() and not prediction[666:].any()
                assert not reference_cache[key][:101].any() and not reference_cache[key][666:].any()
                tiles[selection_id] = overlay(image_path, reference_cache[key], prediction)
    selections = {p: {(r["model"], r["category"]): r for r in records if r["panel"] == p} for p in ["shared_patients", "per_model_extrema"]}
    draw_matrix("main_shared_patients_4x3", "Same three patients across four models", selections["shared_patients"], tiles, True)
    draw_matrix("supplement_per_model_extrema_4x3", "Best, near-median and worst for each model", selections["per_model_extrema"], tiles, False)
    for model in MODELS:
        draw_vertical(model, selections["per_model_extrema"], tiles)
    rule = {"seed": SEED, "seed_choice": "Fixed seed 17 chosen before inspecting the ranking; no cross-seed search.", "candidate_population": "All 252 test frames from the 18 seed-17 test patients, common to all four models.", "shared_panel": "Rank frames by mean postprocessed Dice over the four models. Choose maximum, minimum, then closest to the median of all 252 means, with three distinct patients. Show columns best, near median, worst. All models use the identical frame within each column.", "extrema_panel": "Separately for each model, choose maximum Dice, minimum Dice, then closest to the median of its 252 test frames, requiring distinct patients within a row. Display best, near median, worst. No diagnosis quota.", "ties": "Ascending patient_id, frame_id, side, case_id; seed is fixed at 17. Endpoints are chosen before the near-median case. Assertions confirm the distinct-patient condition did not alter the global maximum, global minimum, or closest-median distance in this dataset.", "interpretation": "Retrospective, deliberately outcome-selected descriptive examples; not random, model blind, or representative. Ranking is frame based, not based on patient-average Dice.", "metric": "2*TP/(2*TP+FP+FN), on full 768x768 binary selected_mask and reference rasters; invalid ROI predictions are the stored empty selected_mask and receive Dice=0.", "display": {"crop_xyxy": CROP, "crop_purpose": "Remove only common fixed black letterbox padding (101 top, 102 bottom rows). No anatomical crop, enhancement, or intensity normalization.", "contrast": "Identical original RGB values across models", "reference": "cyan solid boundary", "prediction": "orange dashed boundary", "outline": "One-pixel-radius morphological boundary; dashes use fixed pixel phase ((x+y)%13)<8.", "dpi": 300}}
    write_json(OUT / "selection_manifest.json", {"rules": rule, "records": records})
    pd.DataFrame(records).to_csv(OUT / "selection_manifest.csv", index=False)
    write_json(private / "selection_manifest_private.json", {"privacy": "INTERNAL ONLY: original source paths contain personal names. Do not submit or publish this directory.", "rules": rule, "records": internal})
    pd.DataFrame(internal).to_csv(private / "selection_manifest_private.csv", index=False)
    audit = {"verification": verification, "source_manifest_sha256": sha(manifest_path), "locked_config_sha256": sha(cfg_path), "seed_split_sha256": sha(split_path), "segmentation_summary_sha256": sha(summary_path), "script_sha256": sha(Path(__file__)), "unique_reference_masks_verified": len(reference_cache), "saved_postprocessed_masks_verified": 1008, "source_run": rel(RUN), "new_model_inference": False, "mask_synthesis_or_editing": False, "all_metric_checks_passed": True}
    write_json(OUT / "source_audit.json", audit)
    common_caption = (
        "Qualitative comparison of saved postprocessed segmentations in three shared test patients. "
        "Rows identify the four models; each column uses an identical frame and reference mask across models. "
        "From all 252 frames of the 18 patients in fixed test split seed 17, frames were ranked by the mean Dice "
        "across the four models. The maximum, closest-to-median, and minimum means selected three distinct "
        "patients (0.943, 0.866, and 0.531, respectively); these are frame selections, not patient-average scores. "
        "Extrema were chosen first, then the closest-median frame; exact ties were resolved by ascending "
        "pseudonymous patient ID, frame ID, side, and case ID. No diagnosis quota was imposed. Cyan solid "
        "outlines denote reference masks and orange dashed outlines denote saved accepted-component masks. "
        "Dice is 2TP/(2TP+FP+FN) on the full 768-by-768 rasters; rejected ROIs use stored empty predictions "
        "and Dice=0. Only common black letterbox padding was removed, with identical contrast and scale. "
        "Displayed case labels are anonymous and no personal identifiers are rendered. These retrospective, "
        "outcome-selected examples are descriptive, not representative, and cannot estimate prevalence or "
        "comparative performance. A column's rank is across the four-model mean, not each model's own rank."
    )
    extrema_caption = (
        "Model-specific best, near-median, and worst saved postprocessed segmentations. "
        "For each model independently, all 252 frames from 18 patients in fixed test split seed 17 were ranked "
        "by full-frame Dice. The maximum and minimum were selected first, followed by the frame closest to "
        "the median; three distinct patients were used within each model. Exact ties were resolved by ascending "
        "pseudonymous patient ID, frame ID, side, and case ID, without diagnosis quotas. This distinct-patient "
        "condition did not change the extrema or closest-median distance. Cyan solid outlines denote reference "
        "masks and orange dashed outlines denote saved accepted-component predictions. Dice is "
        "2TP/(2TP+FP+FN) on full 768-by-768 rasters; invalid ROIs are stored empty predictions with Dice=0. "
        "Only fixed black letterbox padding was removed; contrast and scale are unchanged. Anonymous case "
        "labels are used and no personal identifiers are rendered. No inference was rerun. These retrospective, "
        "outcome-selected examples are descriptive, not representative, and cannot estimate prevalence or "
        "comparative performance. Frames may differ across model-specific selections."
    )
    figure_index = [
        {"id": "shared_patients_4x3", "category": "shared_cases", "png": str(OUT / "main_shared_patients_4x3.png"), "svg": str(OUT / "main_shared_patients_4x3.svg"), "caption": common_caption},
        {"id": "per_model_extrema_4x3", "category": "model_extrema", "png": str(OUT / "supplement_per_model_extrema_4x3.png"), "svg": str(OUT / "supplement_per_model_extrema_4x3.svg"), "caption": extrema_caption + " Rows identify models and columns show best, near-median, and worst frames."},
    ]
    for model in MODELS:
        stem = f"supplement_{model}_best_median_worst_3x1"
        figure_index.append({"id": f"{model}_extrema_3x1", "category": "model_extrema_vertical", "model": model, "png": str(OUT / f"{stem}.png"), "svg": str(OUT / f"{stem}.svg"), "caption": NAMES[model] + ": " + extrema_caption + " The three rows reproduce this model's best, near-median, and worst frames from the supplementary matrix at a larger display size."})
    write_json(OUT / "figure_index.json", figure_index)
    figures = sorted(p for p in [*OUT.glob("*.png"), *OUT.glob("*.svg")] if p.name != "QA_contact_sheet.png")
    write_json(OUT / "output_checksums.json", {p.name: sha(p) for p in figures})
    # Contact sheet is for inspection only, not a publication figure.
    previews = []
    for p in sorted(OUT.glob("*.png")):
        if p.name == "QA_contact_sheet.png":
            continue
        im = Image.open(p).convert("RGB")
        im.thumbnail((600, 720))
        tile = Image.new("RGB", (640, 790), "#eaf0f5")
        tile.paste(im, ((640 - im.width) // 2, 52))
        ImageDraw.Draw(tile).text((16, 14), p.stem.replace("supplement_", ""), font=font(18), fill=INK)
        previews.append(tile)
    contact = Image.new("RGB", (3 * 640, 2 * 790), "white")
    for i, tile in enumerate(previews):
        contact.paste(tile, ((i % 3) * 640, (i // 3) * 790))
    contact.save(OUT / "QA_contact_sheet.png")
    print(json.dumps({"figure_pairs": 6, "selection_records": len(records), "unique_selected_patients": len(anonymous), "verification": verification}, indent=2))


if __name__ == "__main__":
    main()
