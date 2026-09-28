from pathlib import Path
import os
import shutil
from PIL import Image, ImageDraw, ImageFont

def bv(model, strategy, level, field):
    r = BINARY_MEANS[(model, strategy, level, "ALL", "predicted_roi")]
    return float(r[field + "_mean"]), float(r[field + "_sd"])

def bms(model, strategy, level, field, digits=3):
    a, b = bv(model, strategy, level, field)
    return f"{a:.{digits}f} ± {b:.{digits}f}"

def bseg(model, field):
    r = BINARY_SEG[(model, "model_specific", "frame", "ALL")]
    return float(r[field + "_mean"]), float(r[field + "_sd"])

def bseg_level(model, level, field):
    r = BINARY_SEG[(model, "model_specific", level, "ALL")]
    return float(r[field + "_mean"]), float(r[field + "_sd"])

def v(model, strategy, level, metric):
    r = MEANS[(model, strategy, "calibrated", level, metric)]
    return float(r["mean"]), float(r["sample_sd"])

def ms(model, strategy, level, metric, digits=3):
    a, b = v(model, strategy, level, metric)
    return f"{a:.{digits}f} ± {b:.{digits}f}"

def binary_eye_figure():
    """Render an original English data plot, not an alteration of a source figure."""
    width, height = 2100, 790
    im = Image.new("RGB", (width, height), "white")
    dr = ImageDraw.Draw(im)
    font_file = Path(os.environ.get("PUBLICATION_ARIAL_REGULAR", "Arial.ttf"))
    bold_file = Path(os.environ.get("PUBLICATION_ARIAL_BOLD", "Arial-Bold.ttf"))
    regular = ImageFont.truetype(str(font_file), 30)
    small = ImageFont.truetype(str(font_file), 25)
    heading = ImageFont.truetype(str(bold_file), 38)
    dark = "#20242b"
    blue = "#1c6fb5"
    grid = "#d7dce2"
    names = [LABEL[m] for m in MODELS]
    ys = [205, 335, 465, 595]
    dr.text((60, 25), "Binary classification on strict predicted ROIs", font=heading, fill=dark)
    for k, (left, xmin, xmax, title, field) in enumerate([
        (410, .60, 1.00, "Failure aware eye level balanced accuracy", "failure_inclusive.balanced_accuracy"),
        (1250, .80, 1.00, "Eye level coverage", "coverage")]):
        right = left + 670
        dr.text((left, 100), title, font=regular, fill=dark)
        ticks = [xmin + (xmax-xmin)*j/4 for j in range(5)]
        for tick in ticks:
            x = int(left + (tick-xmin)/(xmax-xmin)*(right-left))
            dr.line([(x, 175), (x, 645)], fill=grid, width=2)
            dr.text((x-25, 663), f"{tick:.2f}", font=small, fill=dark)
        dr.line([(left, 645), (right, 645)], fill=dark, width=2)
        for j,m in enumerate(MODELS):
            y=ys[j]
            if k==0:
                dr.text((60,y-20),names[j],font=regular,fill=dark)
            mean,sd=bv(m,"model_specific","eye",field)
            x0=int(left+(max(xmin,mean-sd)-xmin)/(xmax-xmin)*(right-left))
            x1=int(left+(min(xmax,mean+sd)-xmin)/(xmax-xmin)*(right-left))
            xc=int(left+(mean-xmin)/(xmax-xmin)*(right-left))
            dr.line([(x0,y),(x1,y)],fill=blue,width=5)
            dr.line([(x0,y-13),(x0,y+13)],fill=blue,width=4)
            dr.line([(x1,y-13),(x1,y+13)],fill=blue,width=4)
            dr.ellipse((xc-9,y-9,xc+9,y+9),fill=blue)
    dr.text((60,735),"Points are five split means; whiskers are descriptive sample SD, not confidence intervals.",font=small,fill=dark)
    path=OUT/"binary_eye_ba_coverage.png"
    im.save(path)
    return path

def segmentation_overall_figure():
    """Original six-panel summary of overlap, boundary error, and ROI availability."""
    width, height = 2100, 1280
    im = Image.new("RGB", (width, height), "white")
    dr = ImageDraw.Draw(im)
    font_file = Path(os.environ.get("PUBLICATION_ARIAL_REGULAR", "Arial.ttf"))
    bold_file = Path(os.environ.get("PUBLICATION_ARIAL_BOLD", "Arial-Bold.ttf"))
    regular = ImageFont.truetype(str(font_file), 25)
    small = ImageFont.truetype(str(font_file), 21)
    heading = ImageFont.truetype(str(bold_file), 38)
    panel_heading = ImageFont.truetype(str(bold_file), 28)
    dark, grid = "#20242b", "#d7dce2"
    colors = {"yolo26": "#1c6fb5", "vit_method2": "#df7629",
              "emcad": "#138964", "sam2_unet": "#9459a3"}
    dr.text((45, 20), "Anatomical ROI segmentation performance across five held-out splits",
            font=heading, fill=dark)
    lx = 50
    for model in MODELS:
        color = colors[model]
        dr.ellipse((lx, 88, lx + 18, 106), fill=color)
        dr.text((lx + 28, 80), LABEL[model], font=regular, fill=dark)
        lx += 420 if model != "sam2_unet" else 0

    panels = [
        ("Dice", 0.60, 0.90, "dice", True),
        ("Intersection over union", 0.55, 0.80, "iou", True),
        ("Frame strict-ROI coverage", 0.75, 1.00, "roi_coverage", True),
        ("HD95 (pixels; lower is better)", 15.0, 36.0, "hausdorff95", False),
        ("ASSD (pixels; lower is better)", 5.0, 14.0, "average_symmetric_surface_distance", False),
        ("Relative area error (lower is better)", 0.10, 0.50, "relative_area_error", False),
    ]
    panel_positions = [(40, 150), (730, 150), (1420, 150),
                       (40, 690), (730, 690), (1420, 690)]
    for (title, xmin, xmax, field, _higher_better), (left, top) in zip(panels, panel_positions):
        axis_left, axis_right = left + 35, left + 610
        axis_top, axis_bottom = top + 92, top + 390
        dr.text((left + 20, top + 16), title, font=panel_heading, fill=dark)
        for j in range(5):
            tick = xmin + (xmax - xmin) * j / 4
            x = int(axis_left + (tick - xmin) / (xmax - xmin) * (axis_right - axis_left))
            dr.line([(x, axis_top), (x, axis_bottom)], fill=grid, width=2)
            label = f"{tick:.2f}" if xmax <= 1.0 else f"{tick:.1f}"
            dr.text((x - 25, axis_bottom + 15), label, font=small, fill=dark)
        for i, model in enumerate(MODELS):
            y = axis_top + 40 + i * 70
            mean, sd = bseg(model, field)
            lo, hi = max(xmin, mean - sd), min(xmax, mean + sd)
            x0 = int(axis_left + (lo - xmin) / (xmax - xmin) * (axis_right - axis_left))
            x1 = int(axis_left + (hi - xmin) / (xmax - xmin) * (axis_right - axis_left))
            xc = int(axis_left + (mean - xmin) / (xmax - xmin) * (axis_right - axis_left))
            color = colors[model]
            dr.line([(x0, y), (x1, y)], fill=color, width=5)
            dr.line([(x0, y - 9), (x0, y + 9)], fill=color, width=3)
            dr.line([(x1, y - 9), (x1, y + 9)], fill=color, width=3)
            dr.ellipse((xc - 8, y - 8, xc + 8, y + 8), fill=color)
        dr.rectangle((left + 5, top + 2, left + 645, top + 455), outline="#b9c0c8", width=2)
    dr.text((45, 1230),
            "Points are five-split means; whiskers are descriptive sample SD. Boundary distances are conditional on nonempty predictions.",
            font=small, fill=dark)
    path = OUT / "segmentation_overall_five_split.png"
    im.save(path)
    return path

def threeclass_patient_figure():
    """Original English primary patient BA/coverage plot from audited mean–SD rows."""
    width, height = 2100, 790
    im = Image.new("RGB", (width, height), "white")
    dr = ImageDraw.Draw(im)
    font_file = Path(os.environ.get("PUBLICATION_ARIAL_REGULAR", "Arial.ttf"))
    bold_file = Path(os.environ.get("PUBLICATION_ARIAL_BOLD", "Arial-Bold.ttf"))
    regular = ImageFont.truetype(str(font_file), 30)
    small = ImageFont.truetype(str(font_file), 25)
    heading = ImageFont.truetype(str(bold_file), 38)
    dark, purple, grid = "#20242b", "#9459a3", "#d7dce2"
    names = [LABEL[m] for m in MODELS]
    ys = [205, 335, 465, 595]
    dr.text((60, 25), "Direct three-class classification on strict predicted ROIs", font=heading, fill=dark)
    for k, (left, xmin, xmax, title, field) in enumerate([
        (410, .20, .80, "Failure-aware patient balanced accuracy", "failure_aware.balanced_accuracy"),
        (1250, .70, 1.00, "Patient coverage", "coverage")]):
        right = left + 670
        dr.text((left, 100), title, font=regular, fill=dark)
        for j in range(5):
            tick = xmin + (xmax-xmin)*j/4
            x = int(left + (tick-xmin)/(xmax-xmin)*(right-left))
            dr.line([(x, 175), (x, 645)], fill=grid, width=2)
            dr.text((x-25, 663), f"{tick:.2f}", font=small, fill=dark)
        dr.line([(left, 645), (right, 645)], fill=dark, width=2)
        for j,m in enumerate(MODELS):
            y=ys[j]
            if k==0:
                dr.text((60,y-20),names[j],font=regular,fill=dark)
            mean,sd=v(m,"model_specific","patient",field)
            x0=int(left+(max(xmin,mean-sd)-xmin)/(xmax-xmin)*(right-left))
            x1=int(left+(min(xmax,mean+sd)-xmin)/(xmax-xmin)*(right-left))
            xc=int(left+(mean-xmin)/(xmax-xmin)*(right-left))
            dr.line([(x0,y),(x1,y)],fill=purple,width=5)
            dr.line([(x0,y-13),(x0,y+13)],fill=purple,width=4)
            dr.line([(x1,y-13),(x1,y+13)],fill=purple,width=4)
            dr.ellipse((xc-9,y-9,xc+9,y+9),fill=purple)
    dr.text((60,735),"Points are five split means; whiskers are descriptive sample SD, not confidence intervals.",font=small,fill=dark)
    path=OUT/"threeclass_patient_ba_coverage.png"
    im.save(path)
    return path
