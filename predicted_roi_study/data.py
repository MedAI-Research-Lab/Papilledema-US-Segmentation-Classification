"""Leakage-safe data access for the strict predicted-ROI study.

The module deliberately never creates a new outer split.  It verifies and
reuses the five immutable patient-membership files from the earlier study.
Inner folds are generated only inside an outer-training partition and are used
to obtain out-of-fold segmentation predictions for classifier training.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .config import EXPECTED_SEEDS, PROJECT_ROOT, resolve_project_path, sha256_file


REQUIRED_MANIFEST_COLUMNS = {
    "patient_id",
    "case_id",
    "side",
    "frame_id",
    "label_binary",
    "label_3class",
    "output_image",
    "output_mask",
    "output_image_sha256",
    "output_mask_sha256",
}
PARTITIONS = ("train", "validation", "test")
IMAGE_NET_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
IMAGE_NET_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)
EYE_ID_COLUMNS = ("patient_id", "case_id", "side")
CLASSIFIER_ROW_ORDER = (*EYE_ID_COLUMNS, "frame_id")


def _atomic_text(path: Path, value: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding=encoding)
    temporary.replace(path)


def _canonical_csv(frame: pd.DataFrame) -> str:
    return frame.to_csv(index=False, lineterminator="\n")


def _frame_key(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["patient_id"].astype(str)
        + "|"
        + frame["case_id"].astype(str)
        + "|"
        + frame["side"].astype(str)
        + "|"
        + frame["frame_id"].astype(str)
    )


def strict_boolean_mask(values: pd.Series, *, name: str = "boolean field") -> pd.Series:
    """Return a real boolean mask and reject coercible/non-boolean values.

    ``astype(bool)`` is unsafe for provenance-bearing CSV data because strings
    such as ``"False"`` are truthy.  Study gates therefore accept only actual
    Python/NumPy boolean scalars and reject missing, numeric, or text values.
    """

    if values.isna().any():
        raise ValueError(f"{name} must not contain missing values")
    observed = values.to_numpy(dtype=object, copy=False)
    if any(not isinstance(value, (bool, np.bool_)) for value in observed):
        raise ValueError(f"{name} must contain only real boolean values")
    return pd.Series(
        np.asarray([bool(value) for value in observed], dtype=bool),
        index=values.index,
        name=values.name,
    )


def build_classifier_training_selection(
    index: pd.DataFrame,
    *,
    frames_per_eye: int,
    minimum_valid_frames: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Derive immutable classifier optimization rows and an eye ledger.

    ``roi_valid`` remains the frame-level predicted-ROI quality decision.  The
    eye-level ``minimum_valid_frames`` rule is represented separately and is
    used only to select classifier optimization rows.  Validation/test rows
    are not filtered here; their 4/7 decision is made during aggregation.
    """

    if isinstance(frames_per_eye, bool) or int(frames_per_eye) <= 0:
        raise ValueError("frames_per_eye must be a positive integer")
    if (
        isinstance(minimum_valid_frames, bool)
        or int(minimum_valid_frames) <= 0
        or int(minimum_valid_frames) > int(frames_per_eye)
    ):
        raise ValueError("minimum_valid_frames must be in [1, frames_per_eye]")
    required = {
        *EYE_ID_COLUMNS,
        "frame_id",
        "label",
        "roi_valid",
    }
    missing = required - set(index)
    if missing:
        raise ValueError(
            f"Classifier training ROI index is missing columns: {sorted(missing)}"
        )

    source = index.copy(deep=True)
    for column in (*EYE_ID_COLUMNS, "frame_id"):
        if source[column].isna().any() or source[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Classifier training ROI index has an empty {column}")
    valid = strict_boolean_mask(source["roi_valid"], name="roi_valid")
    labels = pd.to_numeric(source["label"], errors="raise")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("Classifier training labels must be binary integers 0/1")
    if source.duplicated(list(CLASSIFIER_ROW_ORDER)).any():
        raise ValueError("Classifier training ROI index contains duplicate eye/frame rows")

    source = source.assign(_roi_valid=valid.to_numpy(dtype=bool, copy=True))
    source = source.sort_values(list(CLASSIFIER_ROW_ORDER), kind="stable").reset_index(drop=True)
    ledger_rows: list[dict[str, Any]] = []
    for identity, group in source.groupby(list(EYE_ID_COLUMNS), sort=False, dropna=False):
        total_frames = int(len(group))
        unique_frames = int(group["frame_id"].astype(str).nunique())
        if total_frames != int(frames_per_eye) or unique_frames != int(frames_per_eye):
            raise ValueError(
                f"Eye {tuple(str(value) for value in identity)!r} must contain exactly "
                f"{int(frames_per_eye)} unique frames"
            )
        if group["label"].astype(int).nunique() != 1:
            raise ValueError(
                f"Eye {tuple(str(value) for value in identity)!r} has inconsistent labels"
            )
        valid_frames = int(group["_roi_valid"].sum())
        eligible = bool(valid_frames >= int(minimum_valid_frames))
        if eligible:
            reason = (
                f"eligible_ge_{int(minimum_valid_frames)}_of_{int(frames_per_eye)}"
            )
        elif valid_frames == 0:
            reason = "no_valid_predicted_roi"
        else:
            reason = f"below_{int(minimum_valid_frames)}_of_{int(frames_per_eye)}"
        ledger_rows.append(
            {
                **dict(zip(EYE_ID_COLUMNS, identity, strict=True)),
                "label": int(group["label"].iloc[0]),
                "total_frames": total_frames,
                "valid_roi_frames": valid_frames,
                "minimum_required_frames": int(minimum_valid_frames),
                "eye_training_eligible": eligible,
                "optimization_frame_count": valid_frames if eligible else 0,
                "exclusion_reason": reason,
            }
        )

    ledger = pd.DataFrame(ledger_rows).sort_values(
        list(EYE_ID_COLUMNS), kind="stable"
    ).reset_index(drop=True)
    eligible_keys = {
        tuple(row)
        for row in ledger.loc[
            strict_boolean_mask(
                ledger["eye_training_eligible"], name="eye_training_eligible"
            ),
            list(EYE_ID_COLUMNS),
        ].itertuples(index=False, name=None)
    }
    selected_eye = pd.Series(
        [
            tuple(getattr(row, column) for column in EYE_ID_COLUMNS) in eligible_keys
            for row in source.itertuples(index=False)
        ],
        index=source.index,
        dtype=bool,
    )
    optimization = source.loc[source["_roi_valid"] & selected_eye].drop(
        columns="_roi_valid"
    )
    optimization = optimization.sort_values(
        list(CLASSIFIER_ROW_ORDER), kind="stable"
    ).reset_index(drop=True)
    if len(optimization) != int(ledger["optimization_frame_count"].sum()):
        raise RuntimeError("Classifier training eligibility ledger does not match selected rows")
    return optimization, ledger


def read_manifest(cfg: Mapping[str, Any]) -> tuple[Path, pd.DataFrame]:
    """Read the locked manifest and return its containing dataset directory."""

    manifest_path = resolve_project_path(cfg["dataset"]["manifest"], must_exist=True)
    frame = pd.read_csv(manifest_path, encoding="utf-8-sig", dtype={"frame_id": str})
    return manifest_path.parent, frame


def validate_manifest(
    cfg: Mapping[str, Any], *, verify_pixels: bool = False, inspect_pixels: bool = False
) -> dict[str, Any]:
    """Validate hierarchy, paths and optionally hashes/pixel contents.

    ``verify_pixels`` retains the original API name but performs only byte-level
    SHA-256 checks.  ``inspect_pixels`` additionally decodes every image/mask;
    it must not be used before the global test-access lock.
    """

    dataset_root, frame = read_manifest(cfg)
    missing = REQUIRED_MANIFEST_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    expected = cfg["dataset"]
    if len(frame) != int(expected["frames"]):
        raise ValueError("Manifest frame count differs from the protocol lock")
    if frame["patient_id"].nunique() != int(expected["patients"]):
        raise ValueError("Manifest patient count differs from the protocol lock")
    if frame["case_id"].nunique() != int(expected["eyes"]):
        raise ValueError("Manifest eye count differs from the protocol lock")
    if frame.duplicated(["case_id", "frame_id"]).any() or _frame_key(frame).duplicated().any():
        raise ValueError("The manifest contains repeated eye/frame identifiers")
    if not (frame.groupby("case_id", sort=False).size() == int(expected["frames_per_eye"])).all():
        raise ValueError("Every eye must contain exactly seven frames")
    binary_expected = (frame["label_3class"].astype(int) > 0).astype(int)
    if not np.array_equal(frame["label_binary"].astype(int).to_numpy(), binary_expected.to_numpy()):
        raise ValueError("Binary labels must be control=0 versus papilledema/pseudopapilledema=1")
    for patient_id, group in frame.groupby("patient_id", sort=False):
        if group["case_id"].nunique() != int(expected["eyes_per_patient"]):
            raise ValueError(f"Patient {patient_id!r} does not have exactly two eyes")
        if set(group["side"].astype(str)) != {"SAG", "SOL"}:
            raise ValueError(f"Patient {patient_id!r} must have SAG and SOL eyes")
        if group["label_binary"].nunique() != 1 or group["label_3class"].nunique() != 1:
            raise ValueError(f"Patient {patient_id!r} has inconsistent labels")

    cross_patient_duplicates: list[dict[str, Any]] = []
    for digest, group in frame.groupby("output_image_sha256", sort=False):
        if group["patient_id"].nunique() > 1:
            cross_patient_duplicates.append({"sha256": digest, "patients": sorted(group.patient_id.astype(str).unique())})
    if cross_patient_duplicates:
        raise ValueError(f"Exact image duplicates cross patient boundaries: {cross_patient_duplicates[:5]}")

    checked_pairs = 0
    for row in frame.itertuples(index=False):
        image_path = (dataset_root / str(row.output_image)).resolve()
        mask_path = (dataset_root / str(row.output_mask)).resolve()
        for path in (image_path, mask_path):
            try:
                path.relative_to(dataset_root.resolve())
            except ValueError as error:
                raise ValueError(f"Manifest path escapes dataset root: {path}") from error
            if not path.is_file():
                raise FileNotFoundError(path)
        if verify_pixels:
            if sha256_file(image_path) != str(row.output_image_sha256):
                raise ValueError(f"Image hash mismatch: {image_path}")
            if sha256_file(mask_path) != str(row.output_mask_sha256):
                raise ValueError(f"Mask hash mismatch: {mask_path}")
            checked_pairs += 1
        if inspect_pixels:
            with Image.open(image_path) as image, Image.open(mask_path) as mask:
                mask_array = np.asarray(mask)
                if image.mode != "RGB" or image.size != mask.size:
                    raise ValueError(f"Image/mask geometry mismatch: {image_path}")
                if not set(np.unique(mask_array).tolist()) <= {0, 255} or not bool(mask_array.any()):
                    raise ValueError(f"Expected one non-empty binary annotation raster: {mask_path}")
    return {
        "manifest_sha256": sha256_file(resolve_project_path(expected["manifest"], must_exist=True)),
        "patients": int(frame.patient_id.nunique()),
        "eyes": int(frame.case_id.nunique()),
        "frames": int(len(frame)),
        "sha256_checked_file_pairs": checked_pairs if verify_pixels else 0,
        "decoded_pixel_pairs": checked_pairs if inspect_pixels else 0,
        "cross_patient_exact_duplicates": 0,
    }


def read_patient_split(cfg: Mapping[str, Any], seed: int) -> pd.DataFrame:
    """Load one original patient split only after verifying its locked digest."""

    if int(seed) not in EXPECTED_SEEDS:
        raise ValueError(f"Unsupported split seed: {seed}")
    source = cfg["split_policy"]["sources"][str(int(seed))]
    path = resolve_project_path(source["path"], must_exist=True)
    if sha256_file(path) != source["sha256"]:
        raise ValueError(f"Immutable patient split changed for seed {seed}")
    split = pd.read_csv(path, encoding="utf-8-sig")
    required = {"patient_id", "label_3class", "label_binary", "split"}
    if not required <= set(split):
        raise ValueError(f"Split {seed} is missing columns: {sorted(required - set(split))}")
    if split["patient_id"].duplicated().any() or len(split) != int(cfg["dataset"]["patients"]):
        raise ValueError(f"Split {seed} must assign every patient exactly once")
    split["patient_id"] = split["patient_id"].astype(str)
    if split["patient_id"].duplicated().any():
        raise ValueError(f"Split {seed} repeats a normalized patient identifier")
    split["split"] = split["split"].astype(str)
    if set(split["split"]) != set(PARTITIONS):
        raise ValueError(f"Split {seed} has invalid partition labels")
    if "seed" in split and not (split["seed"].astype(int) == int(seed)).all():
        raise ValueError(f"Split file seed column does not match {seed}")
    split["label_3class"] = split["label_3class"].astype(int)
    split["label_binary"] = split["label_binary"].astype(int)
    if not set(split["label_3class"]) <= {0, 1, 2}:
        raise ValueError(f"Split {seed} has an invalid three-class label")
    expected_binary = (split["label_3class"] > 0).astype(int)
    if not np.array_equal(split["label_binary"].to_numpy(), expected_binary.to_numpy()):
        raise ValueError(f"Split {seed} binary labels conflict with its three-class labels")

    _, manifest = read_manifest(cfg)
    manifest_required = {"patient_id", "label_3class", "label_binary"}
    if not manifest_required <= set(manifest):
        raise ValueError("Manifest lacks patient labels required to verify the split")
    manifest = manifest.copy()
    manifest["patient_id"] = manifest["patient_id"].astype(str)
    label_counts = manifest.groupby("patient_id", sort=False)[
        ["label_3class", "label_binary"]
    ].nunique(dropna=False)
    if not (label_counts == 1).all().all():
        raise ValueError("Manifest labels are inconsistent within a patient")
    manifest_labels = (
        manifest[["patient_id", "label_3class", "label_binary"]]
        .drop_duplicates("patient_id")
        .rename(
            columns={
                "label_3class": "manifest_label_3class",
                "label_binary": "manifest_label_binary",
            }
        )
    )
    aligned = split.merge(
        manifest_labels, on="patient_id", how="outer", validate="one_to_one",
        indicator=True,
    )
    if not (aligned["_merge"] == "both").all():
        raise ValueError("Manifest and patient split do not contain identical patient IDs")
    if not np.array_equal(
        aligned["label_3class"].astype(int).to_numpy(),
        aligned["manifest_label_3class"].astype(int).to_numpy(),
    ) or not np.array_equal(
        aligned["label_binary"].astype(int).to_numpy(),
        aligned["manifest_label_binary"].astype(int).to_numpy(),
    ):
        raise ValueError(f"Split {seed} labels do not match the locked manifest")
    expected_quotas = cfg["split_policy"]["class_quotas_train_validation_test"]
    for label, quotas in expected_quotas.items():
        observed = [
            int(((split.label_3class == int(label)) & (split.split == partition)).sum())
            for partition in PARTITIONS
        ]
        if observed != [int(value) for value in quotas]:
            raise ValueError(f"Split {seed} quota mismatch for label {label}: {observed}")
    return split.sort_values("patient_id", kind="stable").reset_index(drop=True)


def split_frames(cfg: Mapping[str, Any], seed: int, partition: str) -> tuple[Path, pd.DataFrame]:
    """Return frames for one immutable patient-level outer partition."""

    if partition not in PARTITIONS:
        raise ValueError(f"partition must be one of {PARTITIONS}")
    dataset_root, frame = read_manifest(cfg)
    split = read_patient_split(cfg, seed)
    manifest_patients = set(frame.patient_id.astype(str))
    split_patients = set(split.patient_id.astype(str))
    if manifest_patients != split_patients:
        raise ValueError("Manifest and patient split do not contain identical patient IDs")
    ids = set(split.loc[split.split == partition, "patient_id"].astype(str))
    selected = frame[frame.patient_id.astype(str).isin(ids)].copy()
    selected["frame_id"] = selected["frame_id"].astype(str)
    selected["outer_seed"] = int(seed)
    selected["partition"] = partition
    selected = selected.sort_values(
        ["patient_id", "case_id", "side", "frame_id"], kind="stable"
    ).reset_index(drop=True)
    expected_frames = len(ids) * int(cfg["dataset"]["eyes_per_patient"]) * int(cfg["dataset"]["frames_per_eye"])
    if len(selected) != expected_frames:
        raise ValueError(f"Partition {partition} for seed {seed} has an unexpected frame count")
    return dataset_root, selected


def copy_locked_splits(cfg: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Copy exact source memberships into the new study with provenance.

    Existing different content is never overwritten.  The copied files are a
    convenience; all subsequent reads still validate the original source lock.
    """

    output = resolve_project_path(cfg["output"]) / "splits"
    output.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, Any]] = {}
    for seed in cfg["split_seeds"]:
        source_cfg = cfg["split_policy"]["sources"][str(seed)]
        source = resolve_project_path(source_cfg["path"], must_exist=True)
        destination = output / f"seed_{seed}_patients.csv"
        content = source.read_bytes()
        if destination.exists() and destination.read_bytes() != content:
            raise RuntimeError(f"Refusing to overwrite a different copied split: {destination}")
        if not destination.exists():
            destination.write_bytes(content)
        records[str(seed)] = {
            "source": str(source.relative_to(PROJECT_ROOT)),
            "source_sha256": source_cfg["sha256"],
            "copy": str(destination.relative_to(PROJECT_ROOT)),
            "copy_sha256": sha256_file(destination),
        }
    return records


def build_inner_fold_assignment(cfg: Mapping[str, Any], seed: int) -> pd.DataFrame:
    """Create/verify deterministic patient-grouped 5-fold assignments in outer train."""

    split = read_patient_split(cfg, seed)
    patients = split.loc[split.split == "train", ["patient_id", "label_3class", "label_binary"]].copy()
    patients = patients.sort_values("patient_id", kind="stable").reset_index(drop=True)
    folds = int(cfg["cross_fitting"]["folds"])
    offsets = cfg["training"]["seed_offsets"]
    fold_seed = int(seed) + int(offsets["cross_fit"])
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=fold_seed)
    patients["inner_fold"] = -1
    dummy = np.zeros(len(patients), dtype=np.uint8)
    for fold, (_, heldout) in enumerate(splitter.split(dummy, patients.label_3class.astype(int))):
        patients.loc[heldout, "inner_fold"] = int(fold)
    if (patients.inner_fold < 0).any() or patients.patient_id.duplicated().any():
        raise RuntimeError("Incomplete cross-fitting assignment")

    output = resolve_project_path(cfg["output"]) / "splits" / f"seed_{seed}_inner_folds.csv"
    content = _canonical_csv(patients)
    if output.exists() and output.read_text(encoding="utf-8-sig") != content:
        raise RuntimeError(f"Refusing to replace a different inner-fold assignment: {output}")
    if not output.exists():
        _atomic_text(output, content, encoding="utf-8-sig")
    return patients


def crossfit_frames(
    cfg: Mapping[str, Any], seed: int, fold: int, role: str
) -> tuple[Path, pd.DataFrame]:
    """Return inner-train or inner-heldout frames; outer validation/test are impossible."""

    if role not in {"fit", "heldout"}:
        raise ValueError("role must be 'fit' or 'heldout'")
    assignments = build_inner_fold_assignment(cfg, seed)
    folds = int(cfg["cross_fitting"]["folds"])
    if not 0 <= int(fold) < folds:
        raise ValueError(f"fold must lie in [0,{folds - 1}]")
    selected_patients = set(
        assignments.loc[
            assignments.inner_fold.ne(int(fold)) if role == "fit" else assignments.inner_fold.eq(int(fold)),
            "patient_id",
        ].astype(str)
    )
    dataset_root, outer_train = split_frames(cfg, seed, "train")
    selected = outer_train[outer_train.patient_id.astype(str).isin(selected_patients)].copy()
    if set(selected.patient_id.astype(str)) != selected_patients:
        raise RuntimeError("Inner fold patients do not exactly match outer training frames")
    selected["inner_fold"] = int(fold)
    selected["inner_role"] = role
    return dataset_root, selected.reset_index(drop=True)


class SegmentationFrameDataset(Dataset):
    """Full-resolution image/mask pairs for segmentation-only optimization."""

    def __init__(
        self,
        dataset_root: str | Path,
        rows: pd.DataFrame,
        augmentation: Mapping[str, Any] | None = None,
        *,
        seed: int = 0,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.rows = rows.reset_index(drop=True).copy()
        self.augmentation = dict(augmentation) if augmentation else None
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows.iloc[int(index)]
        with Image.open(self.dataset_root / str(row.output_image)) as handle:
            image = handle.convert("RGB")
        with Image.open(self.dataset_root / str(row.output_mask)) as handle:
            mask = handle.convert("L")
        if self.augmentation:
            augmentation = self.augmentation
            rng = random.Random(self.seed + self.epoch * 1_000_003 + int(index) * 97)
            angle = rng.uniform(-float(augmentation["rotation_degrees"]), float(augmentation["rotation_degrees"]))
            limit = float(augmentation["translation_fraction"]) * image.width
            translation = [round(rng.uniform(-limit, limit)), round(rng.uniform(-limit, limit))]
            scale_values = augmentation["scale"]
            scale = rng.uniform(float(scale_values[0]), float(scale_values[1]))
            image = TF.affine(image, angle, translation, scale, [0.0, 0.0], InterpolationMode.BILINEAR, fill=0)
            mask = TF.affine(mask, angle, translation, scale, [0.0, 0.0], InterpolationMode.NEAREST, fill=0)
            brightness = augmentation["brightness"]
            image = TF.adjust_brightness(image, rng.uniform(float(brightness[0]), float(brightness[1])))
        image_tensor = torch.from_numpy(np.asarray(image, dtype=np.float32).transpose(2, 0, 1).copy() / 255.0)
        mask_tensor = torch.from_numpy((np.asarray(mask) > 0).astype(np.float32, copy=False)[None].copy())
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "label": torch.tensor(int(row.label_binary), dtype=torch.long),
            "index": torch.tensor(int(index), dtype=torch.long),
        }


class ImageFrameDataset(Dataset):
    """Image-only inference dataset.

    This class intentionally has no mask-loading branch.  It is used for OOF
    held-out, validation-cache and test inference so annotation rasters cannot
    accidentally influence the deployable ROI construction path.
    """

    def __init__(self, dataset_root: str | Path, rows: pd.DataFrame) -> None:
        self.dataset_root = Path(dataset_root)
        self.rows = rows.reset_index(drop=True).copy()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows.iloc[int(index)]
        with Image.open(self.dataset_root / str(row.output_image)) as handle:
            image = handle.convert("RGB")
        image_tensor = torch.from_numpy(
            np.asarray(image, dtype=np.float32).transpose(2, 0, 1).copy() / 255.0
        )
        return {
            "image": image_tensor,
            "index": torch.tensor(int(index), dtype=torch.long),
        }


def training_spatial_prior(dataset: SegmentationFrameDataset) -> torch.Tensor:
    """Compute Hybrid-TALON's prior from unaugmented training GT only."""

    if dataset.augmentation:
        raise ValueError("Spatial prior must be computed from unaugmented training masks")
    if not len(dataset):
        raise ValueError("Cannot build a spatial prior from an empty dataset")
    first = dataset[0]["mask"]
    prior = torch.zeros((1, 1, *first.shape[-2:]), dtype=torch.float32)
    for index in range(len(dataset)):
        prior[0] += dataset[index]["mask"]
    return prior / float(len(dataset))


class ROICacheDataset(Dataset):
    """Classifier inputs generated exclusively from accepted predicted ROIs.

    Invalid frames must stay in the cache index for coverage accounting, but
    they are rejected here and cannot enter classifier optimization/inference.
    """

    def __init__(
        self,
        index: pd.DataFrame | str | Path,
        *,
        normalize: bool = False,
        include_geometry: bool = False,
        augmentation: Mapping[str, Any] | None = None,
        seed: int = 0,
    ) -> None:
        source = pd.read_csv(index, encoding="utf-8-sig", dtype={"frame_id": str}) if isinstance(index, (str, Path)) else index.copy()
        required = {
            "patient_id", "case_id", "side", "frame_id", "label", "label_3class",
            "frame_identity_sha256", "roi_valid", "abstention_reason", "cache_path",
        }
        if not required <= set(source):
            raise ValueError(f"ROI cache index is missing columns: {sorted(required - set(source))}")
        valid = strict_boolean_mask(source["roi_valid"], name="roi_valid")
        if source.loc[valid, "cache_path"].isna().any() or source.loc[~valid, "cache_path"].notna().any():
            raise ValueError("Only valid ROI rows may reference classifier tensors")
        self.rows = source.loc[valid].sort_values(
            list(CLASSIFIER_ROW_ORDER), kind="stable"
        ).reset_index(drop=True)
        # ResNet18ROIClassifier owns the locked normalization.  This optional
        # branch is retained only for non-model audit consumers.
        self.normalize = bool(normalize)
        self.include_geometry = bool(include_geometry)
        self.augmentation = dict(augmentation) if augmentation else None
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows.iloc[int(index)]
        path = resolve_project_path(str(row.cache_path), must_exist=True)
        with np.load(path, allow_pickle=False) as cached:
            if "frame_identity_sha256" not in cached.files:
                raise ValueError(f"ROI tensor cache lacks embedded frame identity: {path}")
            encoded_identity = np.asarray(cached["frame_identity_sha256"])
            image = torch.from_numpy(cached["image"].astype(np.float32, copy=False).copy())
            roi_mask = torch.from_numpy(cached["mask"].astype(bool, copy=False).copy())
            geometry = torch.from_numpy(cached["geometry"].astype(np.float32, copy=False).copy())
        expected_identity = str(row.frame_identity_sha256)
        if (
            encoded_identity.dtype != np.uint8
            or encoded_identity.shape != (32,)
            or encoded_identity.tobytes().hex() != expected_identity
        ):
            raise ValueError(f"ROI tensor cache belongs to a different frame: {path}")
        if image.ndim != 3 or image.shape[0] != 3 or roi_mask.shape != image.shape[-2:]:
            raise ValueError(f"Invalid ROI tensor cache: {path}")
        if self.augmentation:
            augmentation = self.augmentation
            rng = random.Random(self.seed + self.epoch * 1_000_003 + int(index) * 97)
            angle = rng.uniform(
                -float(augmentation["rotation_degrees"]),
                float(augmentation["rotation_degrees"]),
            )
            limit = float(augmentation["translation_fraction"]) * image.shape[-1]
            translation = [
                round(rng.uniform(-limit, limit)),
                round(rng.uniform(-limit, limit)),
            ]
            scale = rng.uniform(float(augmentation["scale"][0]), float(augmentation["scale"][1]))
            image = TF.affine(
                image,
                angle,
                translation,
                scale,
                [0.0, 0.0],
                InterpolationMode.BILINEAR,
                fill=IMAGE_NET_MEAN.flatten().tolist(),
            )
            roi_mask = TF.affine(
                roi_mask[None].float(),
                angle,
                translation,
                scale,
                [0.0, 0.0],
                InterpolationMode.NEAREST,
                fill=0.0,
            )[0].bool()
            brightness = rng.uniform(
                float(augmentation["brightness"][0]),
                float(augmentation["brightness"][1]),
            )
            bright = TF.adjust_brightness(image, brightness)
            image = torch.where(roi_mask.unsqueeze(0), bright, IMAGE_NET_MEAN)
        if self.normalize:
            image = (image - IMAGE_NET_MEAN) / IMAGE_NET_STD
        sample: dict[str, Any] = {
            "image": image,
            "roi_mask": roi_mask,
            "label": torch.tensor(int(row.label), dtype=torch.long),
            "index": torch.tensor(int(index), dtype=torch.long),
        }
        if self.include_geometry:
            sample["geometry"] = geometry
        return sample


def classifier_loss_weights(index: pd.DataFrame) -> np.ndarray:
    """Give each ROI-bearing eye equal influence and balance patient-level labels."""

    required = {*EYE_ID_COLUMNS, "label", "roi_valid"}
    if not required <= set(index):
        raise ValueError(f"ROI index is missing columns: {sorted(required - set(index))}")
    rows = index.loc[
        strict_boolean_mask(index["roi_valid"], name="roi_valid")
    ].copy()
    if rows.empty:
        raise ValueError("No valid predicted ROIs are available for classifier training")
    patient_labels = rows[["patient_id", "label"]].drop_duplicates()
    if patient_labels.patient_id.duplicated().any():
        raise ValueError("Patient labels are inconsistent")
    eye_labels = rows[[*EYE_ID_COLUMNS, "label"]].drop_duplicates()
    if eye_labels.duplicated(list(EYE_ID_COLUMNS)).any():
        raise ValueError("Eye labels are inconsistent")
    counts = eye_labels.label.astype(int).value_counts().reindex([0, 1], fill_value=0)
    if (counts == 0).any():
        raise ValueError("Both binary classes must occur among valid training ROIs")
    frames_per_eye = rows.groupby(list(EYE_ID_COLUMNS), sort=False).size()
    weights = np.asarray(
        [
            1.0
            / (
                2.0
                * float(counts.loc[int(row.label)])
                * float(
                    frames_per_eye.loc[
                        tuple(getattr(row, column) for column in EYE_ID_COLUMNS)
                    ]
                )
            )
            for row in rows.itertuples()
        ],
        dtype=np.float64,
    )
    return weights / weights.mean()


def dataframe_sha256(frame: pd.DataFrame, columns: Sequence[str] | None = None) -> str:
    """Stable digest used in cache and stage receipts."""

    selected = frame.loc[:, list(columns)].copy() if columns is not None else frame.copy()
    payload = _canonical_csv(selected).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "IMAGE_NET_MEAN",
    "IMAGE_NET_STD",
    "CLASSIFIER_ROW_ORDER",
    "EYE_ID_COLUMNS",
    "ImageFrameDataset",
    "ROICacheDataset",
    "SegmentationFrameDataset",
    "build_classifier_training_selection",
    "build_inner_fold_assignment",
    "classifier_loss_weights",
    "copy_locked_splits",
    "crossfit_frames",
    "dataframe_sha256",
    "read_manifest",
    "read_patient_split",
    "split_frames",
    "strict_boolean_mask",
    "training_spatial_prior",
    "validate_manifest",
]
