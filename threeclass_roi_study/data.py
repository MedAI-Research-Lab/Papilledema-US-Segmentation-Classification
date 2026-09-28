"""Three-class classifier data and frozen upstream-artifact auditing.

This module intentionally consumes the predicted-ROI tensors produced by the
locked segmentation-only study.  It never changes those artifacts and never
uses the legacy merged binary label as the classifier target.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


PROJECT_ROOT = Path(__file__).resolve().parents[1]
THREE_CLASS_LABELS = (0, 1, 2)
THREE_CLASS_FAMILIES = ("yolo26", "vit_method2", "emcad", "sam2_unet")
EYE_ID_COLUMNS = ("patient_id", "case_id", "side")
CLASSIFIER_ROW_ORDER = (*EYE_ID_COLUMNS, "frame_id")
IMAGE_NET_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
IMAGE_NET_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def sha256_file(path: str | Path) -> str:
    """Return a lowercase SHA-256 digest without deserializing the file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve(path: str | Path, *, project_root: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _portable_path(path: Path, *, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def strict_boolean_mask(values: pd.Series, *, name: str = "boolean field") -> pd.Series:
    """Reject truthy strings/numbers instead of silently coercing them."""

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


def _validated_threeclass_labels(values: pd.Series, *, name: str = "label_3class") -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise")
    array = numeric.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(array).all() or not np.equal(array, np.floor(array)).all():
        raise ValueError(f"{name} must contain integer labels")
    labels = pd.Series(array.astype(np.int64), index=values.index, name=values.name)
    observed = set(labels.unique().tolist())
    if not observed <= set(THREE_CLASS_LABELS):
        raise ValueError(
            f"{name} must contain only {THREE_CLASS_LABELS}; observed {sorted(observed)}"
        )
    return labels


def _validate_label_consistency(rows: pd.DataFrame) -> None:
    patient_labels = rows[["patient_id", "label_3class"]].drop_duplicates()
    if patient_labels["patient_id"].duplicated().any():
        raise ValueError("Patient three-class labels are inconsistent")
    eye_labels = rows[[*EYE_ID_COLUMNS, "label_3class"]].drop_duplicates()
    if eye_labels.duplicated(list(EYE_ID_COLUMNS)).any():
        raise ValueError("Eye three-class labels are inconsistent")


def _missing_path_mask(values: pd.Series) -> pd.Series:
    return values.isna() | values.astype("string").str.strip().eq("")


class ThreeClassROICacheDataset(Dataset):
    """Load valid predicted-ROI tensors and target ``label_3class``.

    Invalid ROI rows remain in the immutable index for coverage accounting but
    never enter classifier optimization or inference.  The embedded frame
    identity is checked on every load so a tensor cannot be silently paired
    with a different manifest row.
    """

    def __init__(
        self,
        index: pd.DataFrame | str | Path,
        *,
        project_root: str | Path | None = None,
        normalize: bool = False,
        include_geometry: bool = False,
        augmentation: Mapping[str, Any] | None = None,
        seed: int = 0,
    ) -> None:
        self.project_root = Path(project_root or PROJECT_ROOT).resolve()
        required = {
            "patient_id",
            "case_id",
            "side",
            "frame_id",
            "label_3class",
            "frame_identity_sha256",
            "roi_valid",
            "abstention_reason",
            "cache_path",
            "cache_sha256",
        }
        if isinstance(index, (str, Path)):
            header = pd.read_csv(index, encoding="utf-8-sig", nrows=0)
            missing = required - set(header)
            if missing:
                raise ValueError(
                    f"ROI cache index is missing columns: {sorted(missing)}"
                )
            # Import only the three-class identity/ROI contract.  Legacy binary
            # logits, probabilities, thresholds and predictions are forbidden.
            source = pd.read_csv(
                index,
                encoding="utf-8-sig",
                dtype={"frame_id": str},
                usecols=sorted(required),
            )
        else:
            source = index.loc[:, [column for column in index if column in required]].copy()
        missing = required - set(source)
        if missing:
            raise ValueError(f"ROI cache index is missing columns: {sorted(missing)}")
        source["label_3class"] = _validated_threeclass_labels(source["label_3class"])
        _validate_label_consistency(source)
        identities = source["frame_identity_sha256"].astype(str)
        if not identities.map(lambda value: bool(SHA256_RE.fullmatch(value))).all():
            raise ValueError("frame_identity_sha256 must contain 64 hexadecimal characters")
        if identities.duplicated().any():
            raise ValueError("ROI cache index contains duplicate frame identities")
        valid = strict_boolean_mask(source["roi_valid"], name="roi_valid")
        path_missing = _missing_path_mask(source["cache_path"])
        if path_missing.loc[valid].any() or (~path_missing.loc[~valid]).any():
            raise ValueError("Only valid ROI rows may reference classifier tensors")
        self.rows = source.loc[valid].sort_values(
            list(CLASSIFIER_ROW_ORDER), kind="stable"
        ).reset_index(drop=True)
        self.normalize = bool(normalize)
        self.include_geometry = bool(include_geometry)
        self.augmentation = dict(augmentation) if augmentation else None
        self.seed = int(seed)
        self.epoch = 0
        self._verified_cache_sha256: set[str] = set()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows.iloc[int(index)]
        path = _resolve(str(row.cache_path), project_root=self.project_root)
        if not path.is_file():
            raise FileNotFoundError(f"ROI tensor cache does not exist: {path}")
        expected_cache_sha256 = _validate_declared_hash(
            row.cache_sha256, label=f"ROI cache digest for {row.frame_identity_sha256}"
        )
        cache_token = str(path.resolve())
        if cache_token not in self._verified_cache_sha256:
            observed_cache_sha256 = sha256_file(path)
            if observed_cache_sha256 != expected_cache_sha256:
                raise ValueError(
                    f"ROI tensor cache SHA-256 mismatch for {path}: expected "
                    f"{expected_cache_sha256}, found {observed_cache_sha256}"
                )
            self._verified_cache_sha256.add(cache_token)
        with np.load(path, allow_pickle=False) as cached:
            required = {"frame_identity_sha256", "image", "mask", "geometry"}
            if not required <= set(cached.files):
                raise ValueError(
                    f"ROI tensor cache is missing arrays: {sorted(required - set(cached.files))}"
                )
            encoded_identity = np.asarray(cached["frame_identity_sha256"])
            image = torch.from_numpy(
                cached["image"].astype(np.float32, copy=False).copy()
            )
            roi_mask = torch.from_numpy(
                cached["mask"].astype(bool, copy=False).copy()
            )
            geometry = torch.from_numpy(
                cached["geometry"].astype(np.float32, copy=False).copy()
            )
        expected_identity = str(row.frame_identity_sha256).lower()
        if (
            encoded_identity.dtype != np.uint8
            or encoded_identity.shape != (32,)
            or encoded_identity.tobytes().hex() != expected_identity
        ):
            raise ValueError(f"ROI tensor cache belongs to a different frame: {path}")
        if (
            image.ndim != 3
            or image.shape[0] != 3
            or roi_mask.ndim != 2
            or tuple(roi_mask.shape) != tuple(image.shape[-2:])
            or not bool(torch.isfinite(image).all())
        ):
            raise ValueError(f"Invalid ROI tensor cache: {path}")
        if geometry.shape != (4,) or not bool(torch.isfinite(geometry).all()):
            raise ValueError(f"ROI geometry must contain four finite values: {path}")

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
            scale = rng.uniform(
                float(augmentation["scale"][0]), float(augmentation["scale"][1])
            )
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
            image = torch.where(
                roi_mask.unsqueeze(0), TF.adjust_brightness(image, brightness), IMAGE_NET_MEAN
            )
        if self.normalize:
            image = (image - IMAGE_NET_MEAN) / IMAGE_NET_STD
        sample: dict[str, Any] = {
            "image": image,
            "roi_mask": roi_mask,
            "label": torch.tensor(int(row.label_3class), dtype=torch.long),
            "label_3class": torch.tensor(int(row.label_3class), dtype=torch.long),
            "index": torch.tensor(int(index), dtype=torch.long),
        }
        if self.include_geometry:
            sample["geometry"] = geometry
        return sample


# Short compatibility name for new orchestration code.  It is deliberately a
# local alias; the binary package remains untouched.
ROICacheDataset = ThreeClassROICacheDataset


def threeclass_classifier_loss_weights(index: pd.DataFrame) -> np.ndarray:
    """Return frame weights balanced first by class, then patient and eye.

    Each class receives equal total mass.  Inside a class, every patient
    receives equal mass; inside a patient, every ROI-bearing eye receives equal
    mass; and an eye's mass is divided equally among its valid frames.  The
    returned vector is ordered exactly like :class:`ThreeClassROICacheDataset`.
    """

    required = {*EYE_ID_COLUMNS, "frame_id", "label_3class", "roi_valid"}
    missing = required - set(index)
    if missing:
        raise ValueError(f"ROI index is missing columns: {sorted(missing)}")
    source = index.copy()
    source["label_3class"] = _validated_threeclass_labels(source["label_3class"])
    _validate_label_consistency(source)
    rows = source.loc[
        strict_boolean_mask(source["roi_valid"], name="roi_valid")
    ].sort_values(list(CLASSIFIER_ROW_ORDER), kind="stable")
    if rows.empty:
        raise ValueError("No valid predicted ROIs are available for classifier training")

    patient_labels = rows[["patient_id", "label_3class"]].drop_duplicates()
    patient_counts = (
        patient_labels["label_3class"]
        .astype(int)
        .value_counts()
        .reindex(THREE_CLASS_LABELS, fill_value=0)
    )
    if (patient_counts == 0).any():
        raise ValueError("All three classes must occur among ROI-bearing training patients")
    eyes_per_patient = (
        rows[[*EYE_ID_COLUMNS]]
        .drop_duplicates()
        .groupby("patient_id", sort=False)
        .size()
    )
    frames_per_eye = rows.groupby(list(EYE_ID_COLUMNS), sort=False).size()
    weights = np.asarray(
        [
            1.0
            / (
                float(len(THREE_CLASS_LABELS))
                * float(patient_counts.loc[int(row.label_3class)])
                * float(eyes_per_patient.loc[row.patient_id])
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


classifier_loss_weights = threeclass_classifier_loss_weights


@dataclass(frozen=True)
class FrozenUpstreamArtifacts:
    """Paths for one immutable model/seed predicted-ROI run."""

    family: str
    seed: int
    results_root: Path
    run_directory: Path
    segmenter_checkpoint: Path
    segmenter_lock: Path
    training_oof_index: Path
    validation_index: Path
    test_index: Path | None

    @property
    def index_paths(self) -> dict[str, Path]:
        paths = {
            "train_oof": self.training_oof_index,
            "validation": self.validation_index,
        }
        if self.test_index is not None:
            paths["test"] = self.test_index
        return paths


def discover_frozen_upstream_artifacts(
    results_root: str | Path,
    *,
    families: Sequence[str] | None = THREE_CLASS_FAMILIES,
    seeds: Sequence[int] | None = None,
    project_root: str | Path | None = None,
    require_test: bool = True,
) -> list[FrozenUpstreamArtifacts]:
    """Discover complete upstream model/seed runs and fail closed on gaps."""

    workspace = Path(project_root or PROJECT_ROOT).resolve()
    root = _resolve(results_root, project_root=workspace)
    runs = root / "runs" if (root / "runs").is_dir() else root
    if not runs.is_dir():
        raise FileNotFoundError(f"Upstream runs directory does not exist: {runs}")
    selected_families = tuple(families) if families is not None else tuple(
        sorted(path.name for path in runs.iterdir() if path.is_dir())
    )
    unknown = set(selected_families) - set(THREE_CLASS_FAMILIES)
    if unknown:
        raise ValueError(f"Unsupported three-class upstream families: {sorted(unknown)}")
    selected_seeds = None if seeds is None else tuple(sorted({int(seed) for seed in seeds}))
    discovered: list[FrozenUpstreamArtifacts] = []
    observed: set[tuple[str, int]] = set()
    for family in selected_families:
        family_dir = runs / family
        if not family_dir.is_dir():
            raise FileNotFoundError(f"Upstream family directory is missing: {family_dir}")
        seed_dirs = sorted(family_dir.glob("seed_*"), key=lambda path: path.name)
        for run_directory in seed_dirs:
            try:
                seed = int(run_directory.name.removeprefix("seed_"))
            except ValueError as error:
                raise ValueError(f"Malformed upstream seed directory: {run_directory}") from error
            if selected_seeds is not None and seed not in selected_seeds:
                continue
            artifact = FrozenUpstreamArtifacts(
                family=family,
                seed=seed,
                results_root=root,
                run_directory=run_directory.resolve(),
                segmenter_checkpoint=(run_directory / "segmenter" / "selected.pt").resolve(),
                segmenter_lock=(run_directory / "segmenter" / "segmenter_lock.json").resolve(),
                training_oof_index=(run_directory / "roi" / "train_oof_index.csv").resolve(),
                validation_index=(run_directory / "roi" / "validation" / "index.csv").resolve(),
                test_index=(
                    run_directory / "evaluation" / "test_rois" / "index.csv"
                ).resolve()
                if require_test
                else None,
            )
            required_paths = [
                artifact.segmenter_checkpoint,
                artifact.segmenter_lock,
                artifact.training_oof_index,
                artifact.validation_index,
            ]
            if artifact.test_index is not None:
                required_paths.append(artifact.test_index)
            missing = [path for path in required_paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    "Incomplete frozen upstream run; missing: "
                    + ", ".join(str(path) for path in missing)
                )
            if any(not _is_within(path, root) for path in required_paths):
                raise ValueError("Frozen upstream artifacts must remain inside results_root")
            discovered.append(artifact)
            observed.add((family, seed))
    if selected_seeds is not None:
        expected = {(family, seed) for family in selected_families for seed in selected_seeds}
        missing_units = sorted(expected - observed)
        if missing_units:
            raise FileNotFoundError(f"Missing upstream model/seed runs: {missing_units}")
    if not discovered:
        raise FileNotFoundError("No complete frozen upstream runs were discovered")
    return sorted(discovered, key=lambda value: (value.family, value.seed))


def _validate_declared_hash(value: Any, *, label: str) -> str:
    digest = str(value)
    if not SHA256_RE.fullmatch(digest):
        raise ValueError(f"{label} must be a 64-character SHA-256 digest")
    return digest.lower()


def _audit_index(
    path: Path,
    *,
    artifact: FrozenUpstreamArtifacts,
    project_root: Path,
    verify_referenced_files: bool,
    verify_payload_identity: bool,
) -> tuple[dict[str, Any], set[str], set[str]]:
    required = {
        *EYE_ID_COLUMNS,
        "frame_id",
        "label",
        "label_3class",
        "frame_identity_sha256",
        "roi_valid",
        "cache_path",
        "cache_sha256",
        "audit_path",
        "audit_sha256",
    }
    header = pd.read_csv(path, encoding="utf-8-sig", nrows=0)
    missing = required - set(header)
    if missing:
        raise ValueError(f"Frozen ROI index {path} is missing columns: {sorted(missing)}")
    # Explicit projection prevents legacy binary classifier outputs in the
    # source indices from entering this three-class process.
    frame = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={"frame_id": str},
        usecols=sorted(required),
    )
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Frozen ROI index {path} is missing columns: {sorted(missing)}")
    labels = _validated_threeclass_labels(frame["label_3class"])
    frame["label_3class"] = labels
    _validate_label_consistency(frame)
    binary = pd.to_numeric(frame["label"], errors="raise").astype(int)
    if not np.array_equal(binary.to_numpy(), (labels.to_numpy() > 0).astype(int)):
        raise ValueError(f"Legacy binary labels disagree with label_3class in {path}")
    identities = frame["frame_identity_sha256"].astype(str).str.lower()
    if not identities.map(lambda value: bool(SHA256_RE.fullmatch(value))).all():
        raise ValueError(f"Malformed frame identity in frozen ROI index: {path}")
    if identities.duplicated().any():
        raise ValueError(f"Duplicate frame identity in frozen ROI index: {path}")
    valid = strict_boolean_mask(frame["roi_valid"], name="roi_valid")
    cache_missing = _missing_path_mask(frame["cache_path"])
    if cache_missing.loc[valid].any() or (~cache_missing.loc[~valid]).any():
        raise ValueError(f"Frozen ROI cache-path validity invariant failed: {path}")
    if _missing_path_mask(frame["audit_path"]).any():
        raise ValueError(f"Every frozen ROI row must reference an audit tensor: {path}")

    inventory: list[dict[str, str]] = []
    for row in frame.itertuples(index=False):
        identity = str(row.frame_identity_sha256).lower()
        references = [("audit", row.audit_path, row.audit_sha256)]
        if bool(row.roi_valid):
            references.append(("cache", row.cache_path, row.cache_sha256))
        for kind, reference, declared in references:
            digest = _validate_declared_hash(
                declared, label=f"{kind} digest for frame {identity}"
            )
            resolved = _resolve(str(reference), project_root=project_root)
            if not _is_within(resolved, artifact.results_root):
                raise ValueError(f"Frozen {kind} path escapes results_root: {resolved}")
            if not resolved.is_file():
                raise FileNotFoundError(f"Frozen {kind} artifact is missing: {resolved}")
            if verify_referenced_files:
                actual = sha256_file(resolved)
                if actual != digest:
                    raise ValueError(
                        f"Frozen {kind} SHA-256 mismatch for {resolved}: "
                        f"expected {digest}, found {actual}"
                    )
            if kind == "cache" and verify_payload_identity:
                with np.load(resolved, allow_pickle=False) as payload:
                    if "frame_identity_sha256" not in payload.files:
                        raise ValueError(f"Frozen cache lacks embedded identity: {resolved}")
                    encoded = np.asarray(payload["frame_identity_sha256"])
                if (
                    encoded.dtype != np.uint8
                    or encoded.shape != (32,)
                    or encoded.tobytes().hex() != identity
                ):
                    raise ValueError(f"Frozen cache identity mismatch: {resolved}")
            inventory.append(
                {
                    "kind": kind,
                    "path": _portable_path(resolved, project_root=project_root),
                    "sha256": digest,
                }
            )
    label_counts = labels.value_counts().reindex(THREE_CLASS_LABELS, fill_value=0)
    patients = set(frame["patient_id"].astype(str))
    summary = {
        "index_path": _portable_path(path, project_root=project_root),
        "index_sha256": sha256_file(path),
        "rows": int(len(frame)),
        "valid_rows": int(valid.sum()),
        "invalid_rows": int((~valid).sum()),
        "patients": int(len(patients)),
        "label_counts": {str(label): int(label_counts.loc[label]) for label in THREE_CLASS_LABELS},
        "referenced_artifacts": int(len(inventory)),
        "referenced_inventory_sha256": _canonical_sha256(
            sorted(inventory, key=lambda value: (value["kind"], value["path"]))
        ),
    }
    return summary, set(identities), patients


def audit_frozen_upstream_artifacts(
    artifact: FrozenUpstreamArtifacts,
    *,
    project_root: str | Path | None = None,
    verify_referenced_files: bool = True,
    verify_payload_identity: bool = True,
) -> dict[str, Any]:
    """Hash-audit one frozen segmenter and its OOF/validation/test ROI caches."""

    workspace = Path(project_root or PROJECT_ROOT).resolve()
    for path in (
        artifact.segmenter_checkpoint,
        artifact.segmenter_lock,
        *artifact.index_paths.values(),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Frozen upstream artifact is missing: {path}")
        if not _is_within(path, artifact.results_root):
            raise ValueError(f"Frozen upstream artifact escapes results_root: {path}")

    checkpoint_sha256 = sha256_file(artifact.segmenter_checkpoint)
    lock = json.loads(artifact.segmenter_lock.read_text(encoding="utf-8"))
    if (
        lock.get("model") != artifact.family
        or int(lock.get("seed", -1)) != artifact.seed
        or lock.get("stage") != "segmentation_only"
        or float(lock.get("classifier_loss_weight", float("nan"))) != 0.0
    ):
        raise ValueError("Segmenter lock does not attest this segmentation-only model/seed")
    declared_checkpoint_sha = _validate_declared_hash(
        lock.get("checkpoint_sha256"), label="segmenter checkpoint digest"
    )
    if checkpoint_sha256 != declared_checkpoint_sha:
        raise ValueError(
            "Frozen segmenter checkpoint SHA-256 disagrees with segmenter_lock.json"
        )
    locked_checkpoint = _resolve(str(lock.get("checkpoint")), project_root=workspace)
    if locked_checkpoint != artifact.segmenter_checkpoint.resolve():
        raise ValueError("Segmenter lock points at a different checkpoint")

    partitions: dict[str, Any] = {}
    identities_by_partition: dict[str, set[str]] = {}
    patients_by_partition: dict[str, set[str]] = {}
    for name, path in artifact.index_paths.items():
        summary, identities, patients = _audit_index(
            path,
            artifact=artifact,
            project_root=workspace,
            verify_referenced_files=bool(verify_referenced_files),
            verify_payload_identity=bool(verify_payload_identity),
        )
        partitions[name] = summary
        identities_by_partition[name] = identities
        patients_by_partition[name] = patients
    names = tuple(partitions)
    for offset, left in enumerate(names):
        for right in names[offset + 1 :]:
            if identities_by_partition[left] & identities_by_partition[right]:
                raise ValueError(f"Frame leakage between frozen {left} and {right} indices")
            if patients_by_partition[left] & patients_by_partition[right]:
                raise ValueError(f"Patient leakage between frozen {left} and {right} indices")

    report: dict[str, Any] = {
        "schema": 1,
        "family": artifact.family,
        "seed": artifact.seed,
        "source_results_root": _portable_path(
            artifact.results_root, project_root=workspace
        ),
        "segmentation_only": True,
        "segmenter_checkpoint": {
            "path": _portable_path(artifact.segmenter_checkpoint, project_root=workspace),
            "bytes": artifact.segmenter_checkpoint.stat().st_size,
            "sha256": checkpoint_sha256,
        },
        "segmenter_lock": {
            "path": _portable_path(artifact.segmenter_lock, project_root=workspace),
            "bytes": artifact.segmenter_lock.stat().st_size,
            "sha256": sha256_file(artifact.segmenter_lock),
        },
        "partitions": partitions,
        "referenced_files_verified": bool(verify_referenced_files),
        "payload_identities_verified": bool(verify_payload_identity),
    }
    report["inventory_sha256"] = _canonical_sha256(report)
    return report


def audit_frozen_upstream_collection(
    artifacts: Iterable[FrozenUpstreamArtifacts],
    **kwargs: Any,
) -> dict[str, Any]:
    """Audit multiple model/seed units and return one canonical inventory hash."""

    reports = [audit_frozen_upstream_artifacts(value, **kwargs) for value in artifacts]
    identities = [(report["family"], int(report["seed"])) for report in reports]
    if len(identities) != len(set(identities)):
        raise ValueError("Frozen upstream collection contains duplicate model/seed units")
    reports.sort(key=lambda value: (value["family"], int(value["seed"])))
    return {
        "schema": 1,
        "units": reports,
        "unit_count": len(reports),
        "inventory_sha256": _canonical_sha256(
            [report["inventory_sha256"] for report in reports]
        ),
    }


__all__ = [
    "CLASSIFIER_ROW_ORDER",
    "EYE_ID_COLUMNS",
    "FrozenUpstreamArtifacts",
    "IMAGE_NET_MEAN",
    "IMAGE_NET_STD",
    "PROJECT_ROOT",
    "ROICacheDataset",
    "THREE_CLASS_FAMILIES",
    "THREE_CLASS_LABELS",
    "ThreeClassROICacheDataset",
    "audit_frozen_upstream_artifacts",
    "audit_frozen_upstream_collection",
    "classifier_loss_weights",
    "discover_frozen_upstream_artifacts",
    "sha256_file",
    "strict_boolean_mask",
    "threeclass_classifier_loss_weights",
]
