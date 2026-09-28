"""Create lossless, spatially aligned ultrasound image/mask masters.

Crop coordinates come ONLY from RGB screen furniture, never target masks.
Masks are read only for paired transformation and target-preservation QC.
No augmentation or data-fitted normalization is performed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

CLASSES = [("1KONTROL", "kontrol", 0, 0),
           ("USG PAPİLÖDEM_1", "papilodem", 1, 1),
           ("USG PSÖDOPAPİLÖDEM_1", "psodopapilodem", 2, 1)]
RGB_DIR = "07_roi_only_original_rgb_jpg"
MASK_DIR = "08_roi_only_original_mask_png"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_pair(image_path, mask_path, *, channels_first=False):
    """Return image float32 [0,1], mask uint8 {0,1}; no fitted statistics.

    Default image shape H,W,3; channels_first=True gives 3,H,W.
    Mask shape is always H,W. Works independently of ML frameworks.
    """
    with Image.open(image_path) as im:
        image = np.asarray(im.convert("RGB"), dtype=np.float32) / np.float32(255.0)
    with Image.open(mask_path) as im:
        raw_mask = np.asarray(im)
    if not np.isin(raw_mask, [0, 255]).all():
        raise ValueError("Expected binary master mask with values 0/255")
    mask = (raw_mask == 255).astype(np.uint8)
    if channels_first:
        image = np.ascontiguousarray(image.transpose(2, 0, 1))
    return image, mask


def bbox(mask):
    y, x = np.where(mask > 0)
    return [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1] if len(x) else None


def inventory(root):
    pairs = []
    for folder, label, label3, label2 in CLASSES:
        export = root / folder / "OUTPUT_EXPORT"
        if not export.is_dir():
            raise FileNotFoundError(export)
        for patient in sorted(p for p in export.iterdir() if p.is_dir()):
            patient_id = "P_" + hashlib.sha256(f"{folder}/{patient.name}".encode()).hexdigest()[:16]
            for side in ("SAG", "SOL"):
                case = patient / side
                images = {p.stem: p for p in (case / RGB_DIR).glob("*.jpg")}
                masks = {p.stem: p for p in (case / MASK_DIR).glob("*.png")}
                if not images or images.keys() != masks.keys():
                    raise ValueError(f"Missing/unpaired case: {case}")
                for stem in sorted(images):
                    pairs.append(dict(class_folder=folder, class_name=label, label_3class=label3,
                                      label_binary=label2, patient_name=patient.name, patient_id=patient_id,
                                      case_id=f"{patient_id}_{side}", side=side, frame_id=stem,
                                      source_image=images[stem], source_mask=masks[stem]))
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--crop", nargs=4, type=int, default=[80, 110, 920, 728], metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"))
    parser.add_argument("--size", type=int, default=768)
    parser.add_argument("--pilot", action="store_true", help="Process the first selected frame of every case")
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output or root / "çalışma_ds").resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    pairs = inventory(root)
    if args.pilot:
        seen = set()
        pairs = [p for p in pairs if p["case_id"] not in seen and not seen.add(p["case_id"])]
    left, top, right, bottom = args.crop
    if not (0 <= left < right <= 1024 and 0 <= top < bottom <= 768):
        raise ValueError("Invalid RGB-derived crop")
    width, height = right - left, bottom - top
    scale = min(1.0, args.size / width, args.size / height)
    resized = (round(width * scale), round(height * scale))
    pad_x, pad_y = (args.size - resized[0]) // 2, (args.size - resized[1]) // 2
    # Validate EVERY mask before creating outputs. Validation never changes the crop.
    for p in pairs:
        with Image.open(p["source_image"]) as im, Image.open(p["source_mask"]) as mask:
            if im.size != (1024, 768) or mask.size != im.size or im.mode != "RGB" or mask.mode != "L":
                raise ValueError(f"Unexpected mode/shape: {p['source_image']}")
            a = np.asarray(mask)
            if not np.isin(a, [0, 255]).all() or not np.any(a):
                raise ValueError(f"Invalid binary mask: {p['source_mask']}")
            if np.count_nonzero(a) != np.count_nonzero(a[top:bottom, left:right]):
                raise ValueError(f"RGB-derived crop would remove target; halted for review: {p['source_mask']}")
    output.mkdir(parents=True)
    records = []
    for index, p in enumerate(pairs, 1):
        with Image.open(p["source_image"]) as im, Image.open(p["source_mask"]) as mask:
            source_mask = np.asarray(mask)
            rgb = im.crop(args.crop)
            msk = mask.crop(args.crop)
            if resized != rgb.size:
                rgb = rgb.resize(resized, Image.Resampling.BILINEAR)
                msk = msk.resize(resized, Image.Resampling.NEAREST)
            rgb_out = Image.new("RGB", (args.size, args.size), 0)
            mask_out = Image.new("L", (args.size, args.size), 0)
            rgb_out.paste(rgb, (pad_x, pad_y))
            mask_out.paste(msk, (pad_x, pad_y))
            base = output / p["class_folder"] / p["patient_name"] / p["side"]
            rgb_path = base / "roi_only_original_rgb" / (p["frame_id"] + ".png")
            mask_path = base / "roi_only_original_mask" / (p["frame_id"] + ".png")
            rgb_path.parent.mkdir(parents=True, exist_ok=True)
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            rgb_out.save(rgb_path, compress_level=6)
            mask_out.save(mask_path, compress_level=6)
            normalized, binary = load_pair(rgb_path, mask_path)
            if normalized.shape != (args.size, args.size, 3) or binary.shape != (args.size, args.size):
                raise ValueError("Output dimensions failed verification")
            if not np.any(binary) or normalized.dtype != np.float32 or normalized.min() < 0 or normalized.max() > 1:
                raise ValueError("Output range/foreground failed verification")
            if not np.array_equal(np.asarray(Image.open(mask_path)), np.asarray(mask_out)):
                raise ValueError("Saved mask differs from intended transform")
            if not np.array_equal(np.asarray(Image.open(rgb_path)), np.asarray(rgb_out)):
                raise ValueError("Saved image differs from intended transform")
            record = {k: v for k, v in p.items() if k not in ("source_image", "source_mask")}
            record.update(source_image=p["source_image"].relative_to(root).as_posix(),
                          source_mask=p["source_mask"].relative_to(root).as_posix(),
                          output_image=rgb_path.relative_to(output).as_posix(),
                          output_mask=mask_path.relative_to(output).as_posix(),
                          source_image_sha256=digest(p["source_image"]), source_mask_sha256=digest(p["source_mask"]),
                          output_image_sha256=digest(rgb_path), output_mask_sha256=digest(mask_path),
                          crop_left=left, crop_top=top, crop_right=right, crop_bottom=bottom,
                          resized_width=resized[0], resized_height=resized[1],
                          pad_left=pad_x, pad_top=pad_y, pad_right=args.size-resized[0]-pad_x,
                          pad_bottom=args.size-resized[1]-pad_y, output_width=args.size, output_height=args.size,
                          scale_x=resized[0]/width, scale_y=resized[1]/height,
                          source_foreground_pixels=int(np.count_nonzero(source_mask)), crop_lost_foreground_pixels=0,
                          output_foreground_pixels=int(binary.sum()), source_bbox_xyxy=json.dumps(bbox(source_mask)),
                          output_bbox_xyxy=json.dumps(bbox(binary)), split="unassigned", seed="unassigned")
            records.append(record)
        if index % 100 == 0 or index == len(pairs):
            print(f"Processed and verified {index}/{len(pairs)} pairs", flush=True)
    with (output / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    report = dict(status="complete", created_utc=datetime.now(timezone.utc).isoformat(),
                  source_root=str(root), output_root=str(output), pilot=args.pilot,
                  patients=len({r["patient_id"] for r in records}), cases=len({r["case_id"] for r in records}),
                  pairs=len(records), image_files=len(records), mask_files=len(records),
                  class_pairs=dict(Counter(r["class_name"] for r in records)),
                  side_pairs=dict(Counter(r["side"] for r in records)),
                  crop_xyxy=args.crop, crop_origin="Fixed RGB screen-layout artifact boundaries; never mask-derived",
                  output_size=[args.size,args.size], resized_size=list(resized), padding_ltrb=[pad_x,pad_y,args.size-resized[0]-pad_x,args.size-resized[1]-pad_y],
                  image_interpolation="Pillow BILINEAR", mask_interpolation="Pillow NEAREST",
                  crop_lost_foreground_pixels=0, errors=0, all_outputs_readback_verified=True,
                  image_master="lossless PNG, RGB uint8 0..255", mask_master="lossless PNG, L uint8 {0,255}",
                  loader_image="float32 RGB / 255, [0,1]", loader_mask="uint8 {0,1}",
                  augmentation=False, clahe=False, denoise=False, sharpen=False,
                  split="unassigned: preserve patient grouping when subsequently assigned",
                  note="Foreground pixel count can change under nearest-neighbor downsampling; zero crop loss is checked BEFORE resizing.")
    (output / "qc_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "README.md").write_text(
        "# Prepared ultrasound dataset\n\n"
        "Only source DS 07/08 are used. Sources are unchanged. Each PNG image and mask use the same fixed RGB-layout crop, aspect-preserving downscale (pixel-rounded), and centered zero padding. No upscaling or augmentation.\n\n"
        f"Crop (0-based, exclusive right/bottom): {args.crop}. Resized content: {resized}; final: {args.size} x {args.size}.\n\n"
        "The crop removes screen furniture and some peripheral anatomy; it is not a target-specific anatomical crop. Ground-truth masks are only used to verify zero crop target loss, never to determine the crop.\n\n"
        "RGB masters are PNG uint8 0..255. Use scripts.preprocess_dataset.load_pair(image_path, mask_path) for float32 RGB [0,1] and uint8 mask {0,1}; channels_first=True returns CHW RGB. Do not normalize the saved PNG files again in place.\n\n"
        "manifest.csv links patient, eye/case, frame, original three-class label and derived binary label, source/output hashes and transform geometry. Binary: 0=control, 1=papilledema or pseudopapilledema. Three-class: 0=control, 1=papilledema, 2=pseudopapilledema. No split/seed has been chosen. All eyes and frames of each patient must share the later split.\n\n"
        "QC checks every paired source and reads back every output. Target pixels are fully retained before resize; nearest-neighbor downsampling changes raster pixel counts. The selected seven ROI-containing frames per case do not establish whole-video target detection.\n",
        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
