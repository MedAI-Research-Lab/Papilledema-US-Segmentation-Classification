"""Deterministic, auditable qualitative segmentation galleries.

The common comparison gallery is deliberately selected *before* model outputs
are supplied.  Its sampling universe is defined only by the locked outer seed,
test membership, original three-class label, and stable frame identifiers.  A
SHA-256 rank then chooses the same frame(s) for every model.  This prevents a
visually attractive model output, Dice score, confidence, or error type from
influencing the main qualitative examples.

The failure gallery is different by design: it is outcome-conditioned.  It
selects one deterministic SHA-256-minimum example for each model, seed, and
predeclared failure category.  Every record and rendered sidecar labels these
examples as illustrative and unsuitable for prevalence or performance claims.

Rendering consumes already-written test audit rasters.  It never runs a model,
changes a threshold, or feeds a ground-truth mask into an inference decision.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from .config import EXPECTED_MODELS, PROJECT_ROOT


SCHEMA_VERSION = 1
FRAME_KEY_COLUMNS = ("patient_id", "case_id", "side", "frame_id")
COMMON_SELECTION_RULE = (
    "sha256_min_per_outer_seed_and_original_3class_label_global_unique_v2"
)
COMMON_SELECTION_SALT = "strict-predicted-roi-common-gallery-v2"
FAILURE_SELECTION_RULE = (
    "outcome_conditioned_sha256_min_per_model_seed_failure_type_v1"
)
FAILURE_SELECTION_SALT = "strict-predicted-roi-failure-gallery-v1"
CANONICAL_FAILURE_STATUSES = (
    "empty",
    "tiny",
    "oversize",
    "border",
    "multi_ambiguous",
)
FAILURE_STATUS_ALIASES = {
    "empty": "empty",
    "tiny": "tiny",
    "oversize": "oversize",
    "oversegmentation": "oversize",
    "border": "border",
    "edge_touch": "border",
    "multi_ambiguous": "multi_ambiguous",
    "ambiguous_multi_component": "multi_ambiguous",
}
OUTCOME_CONDITIONED_NOTE = (
    "Outcome-conditioned illustrative failure example; it is not a random or "
    "representative sample and must not be used to estimate prevalence or performance."
)


@dataclass(frozen=True)
class RenderedGallery:
    """Files and machine-readable index produced by one gallery render."""

    figures: tuple[Path, ...]
    index_path: Path
    metadata_path: Path


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _json_compatible(value: Any) -> Any:
    """Convert pandas/NumPy scalars and missing values to strict JSON values."""

    if value is None:
        return None
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _sha256_value(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ) + "\n"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != value:
            raise RuntimeError(f"Refusing to overwrite a different qualitative lock: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
    temporary.replace(path)


def _partition_column(frame: pd.DataFrame) -> str:
    for candidate in ("partition", "split"):
        if candidate in frame:
            return candidate
    raise ValueError(
        "Source rows must include a partition or split column proving test membership"
    )


def _normalise_sources(
    source_rows_by_seed: Mapping[int | str, pd.DataFrame] | pd.DataFrame,
) -> dict[int, pd.DataFrame]:
    """Validate and canonicalise the model-blind test sampling universe."""

    if isinstance(source_rows_by_seed, pd.DataFrame):
        source = source_rows_by_seed.copy()
        seed_column = "outer_seed" if "outer_seed" in source else "seed"
        if seed_column not in source:
            raise ValueError("A combined source table must include outer_seed or seed")
        items = [(int(seed), group.copy()) for seed, group in source.groupby(seed_column)]
    elif isinstance(source_rows_by_seed, Mapping):
        items = [(int(seed), frame.copy()) for seed, frame in source_rows_by_seed.items()]
    else:
        raise TypeError("source_rows_by_seed must be a DataFrame or seed-to-DataFrame mapping")

    normalised: dict[int, pd.DataFrame] = {}
    required = {*FRAME_KEY_COLUMNS, "label_3class"}
    for seed, frame in items:
        missing = required - set(frame)
        if missing:
            raise ValueError(f"Seed {seed} source rows lack columns: {sorted(missing)}")
        partition_column = _partition_column(frame)
        if not (frame[partition_column].astype(str).str.lower() == "test").all():
            raise ValueError(f"Seed {seed} qualitative universe contains non-test rows")
        # ``outer_seed`` is the authoritative split identifier. The manifest
        # can retain a legacy provenance column named ``seed`` whose value is
        # the literal string ``unassigned``; that column is not a split key
        # when ``outer_seed`` is available.
        authoritative_seed_column = "outer_seed" if "outer_seed" in frame else "seed"
        if authoritative_seed_column in frame and not (
            frame[authoritative_seed_column].astype(int) == seed
        ).all():
            raise ValueError(
                f"Source {authoritative_seed_column} disagrees with mapping key {seed}"
            )
        frame = frame.copy()
        for column in FRAME_KEY_COLUMNS:
            if frame[column].isna().any():
                raise ValueError(f"Seed {seed} contains a missing {column}")
            frame[column] = frame[column].astype(str)
        if frame.label_3class.isna().any():
            raise ValueError(f"Seed {seed} contains a missing label_3class")
        frame["label_3class"] = frame.label_3class.astype(int)
        frame["outer_seed"] = int(seed)
        frame["partition"] = "test"
        if frame.duplicated(list(FRAME_KEY_COLUMNS)).any():
            raise ValueError(f"Seed {seed} contains duplicate stable frame keys")
        normalised[seed] = frame.sort_values(
            ["label_3class", *FRAME_KEY_COLUMNS], kind="stable"
        ).reset_index(drop=True)
    if not normalised:
        raise ValueError("The qualitative source universe is empty")
    return dict(sorted(normalised.items()))


def _source_universe_records(sources: Mapping[int, pd.DataFrame]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for seed, frame in sorted(sources.items()):
        for row in frame.itertuples(index=False):
            records.append(
                {
                    "outer_seed": int(seed),
                    "partition": "test",
                    "label_3class": int(row.label_3class),
                    **{column: str(getattr(row, column)) for column in FRAME_KEY_COLUMNS},
                }
            )
    return records


def _rank_digest(
    *, salt: str, seed: int, row: Any, label: int, model: str | None = None,
    failure_status: str | None = None,
) -> str:
    value = {
        "salt": salt,
        "outer_seed": int(seed),
        "partition": "test",
        "label_3class": int(label),
        **{column: str(getattr(row, column)) for column in FRAME_KEY_COLUMNS},
    }
    if model is not None:
        value["model"] = str(model)
    if failure_status is not None:
        value["failure_status"] = str(failure_status)
    return _sha256_value(value)


def select_common_test_frames(
    source_rows: pd.DataFrame,
    *,
    seed: int,
    labels: Sequence[int] = (0, 1, 2),
    examples_per_label: int = 1,
) -> pd.DataFrame:
    """Select model-blind common frames by deterministic SHA-256 ranking.

    No model output is accepted by this API.  Selection therefore cannot be
    influenced by Dice, ROI validity, probability, or visual inspection.
    """

    if isinstance(examples_per_label, bool) or int(examples_per_label) < 1:
        raise ValueError("examples_per_label must be a positive integer")
    sources = _normalise_sources({int(seed): source_rows})
    frame = sources[int(seed)]
    records: list[dict[str, Any]] = []
    for label in tuple(int(value) for value in labels):
        candidates = frame.loc[frame.label_3class == label].copy()
        if len(candidates) < int(examples_per_label):
            raise ValueError(
                f"Seed {seed}, label {label} has {len(candidates)} test frames; "
                f"{examples_per_label} requested"
            )
        candidates["selection_digest"] = [
            _rank_digest(
                salt=COMMON_SELECTION_SALT, seed=int(seed), row=row, label=label
            )
            for row in candidates.itertuples(index=False)
        ]
        candidates = candidates.sort_values(
            ["selection_digest", *FRAME_KEY_COLUMNS], kind="stable"
        ).head(int(examples_per_label))
        for rank, row in enumerate(candidates.itertuples(index=False), 1):
            records.append(
                {
                    "outer_seed": int(seed),
                    "partition": "test",
                    "label_3class": label,
                    **{column: str(getattr(row, column)) for column in FRAME_KEY_COLUMNS},
                    "selection_rank_within_label": rank,
                    "selection_digest": str(row.selection_digest),
                    "selection_rule": COMMON_SELECTION_RULE,
                    "model_blind_selection": True,
                }
            )
    return pd.DataFrame(records).sort_values(
        ["outer_seed", "label_3class", "selection_rank_within_label"], kind="stable"
    ).reset_index(drop=True)


def build_common_selection_lock(
    source_rows_by_seed: Mapping[int | str, pd.DataFrame] | pd.DataFrame,
    path: str | Path,
    *,
    seeds: Sequence[int] | None = None,
    labels: Sequence[int] = (0, 1, 2),
    examples_per_label: int = 1,
) -> dict[str, Any]:
    """Create an immutable, model-blind qualitative selection lock.

    This function should be called before opening any model prediction/audit
    file.  Re-running it with different content cannot overwrite the lock.
    """

    sources = _normalise_sources(source_rows_by_seed)
    selected_seeds = tuple(sorted(sources)) if seeds is None else tuple(int(s) for s in seeds)
    if len(set(selected_seeds)) != len(selected_seeds):
        raise ValueError("Qualitative seed list contains duplicates")
    if set(selected_seeds) != set(sources):
        raise ValueError("The requested seeds must exactly match the supplied source universes")
    selection_records: list[dict[str, Any]] = []
    used_frame_keys: set[tuple[str, str, str, str]] = set()
    for seed in selected_seeds:
        for label in (int(value) for value in labels):
            chosen = 0
            label_candidates = sources[seed].loc[
                sources[seed].label_3class == label
            ].copy()
            label_candidates["selection_digest"] = [
                _rank_digest(
                    salt=COMMON_SELECTION_SALT,
                    seed=int(seed),
                    row=row,
                    label=label,
                )
                for row in label_candidates.itertuples(index=False)
            ]
            label_candidates = label_candidates.sort_values(
                ["selection_digest", *FRAME_KEY_COLUMNS], kind="stable"
            )
            for row in label_candidates.itertuples(index=False):
                key = tuple(str(getattr(row, column)) for column in FRAME_KEY_COLUMNS)
                if key in used_frame_keys:
                    continue
                used_frame_keys.add(key)
                record = {
                    "outer_seed": int(seed),
                    "partition": "test",
                    "label_3class": label,
                    **{
                        column: str(getattr(row, column))
                        for column in FRAME_KEY_COLUMNS
                    },
                    "selection_rank_within_label": chosen + 1,
                    "selection_digest": str(row.selection_digest),
                    "selection_rule": COMMON_SELECTION_RULE,
                    "model_blind_selection": True,
                    "globally_unique_across_seeds": True,
                }
                selection_records.append(record)
                chosen += 1
                if chosen == int(examples_per_label):
                    break
            if chosen != int(examples_per_label):
                raise ValueError(
                    f"Seed {seed}, label {label} lacks enough globally unique test frames"
                )
    selections = pd.DataFrame(selection_records).sort_values(
        ["outer_seed", "label_3class", "selection_rank_within_label"], kind="stable"
    ).reset_index(drop=True)
    universe = _source_universe_records(sources)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "selection_type": "model_blind_common_segmentation_gallery",
        "selection_rule": COMMON_SELECTION_RULE,
        "selection_salt": COMMON_SELECTION_SALT,
        "selection_inputs": [
            "outer_seed",
            "test_membership",
            "label_3class",
            *FRAME_KEY_COLUMNS,
        ],
        "forbidden_selection_inputs": [
            "model_identity",
            "predicted_mask",
            "probability",
            "confidence",
            "dice",
            "iou",
            "roi_status",
            "visual_judgment",
        ],
        "model_outputs_opened_during_selection": False,
        "seeds": list(selected_seeds),
        "labels": [int(value) for value in labels],
        "examples_per_label": int(examples_per_label),
        "globally_unique_frame_keys_across_seeds": True,
        "source_universe_sha256": _sha256_value(universe),
        "source_universe_size": len(universe),
        "selections": selections.to_dict(orient="records"),
    }
    payload["payload_sha256"] = _sha256_value(payload)
    _atomic_json(Path(path), payload)
    return payload


def _read_json_mapping(value: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return json.loads(Path(value).read_text(encoding="utf-8"))


def verify_common_selection_lock(
    lock: str | Path | Mapping[str, Any],
    source_rows_by_seed: Mapping[int | str, pd.DataFrame] | pd.DataFrame,
) -> dict[str, Any]:
    """Verify payload integrity, source universe, and deterministic selections."""

    value = _read_json_mapping(lock)
    if value.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("Unsupported qualitative selection lock schema")
    expected_digest = value.get("payload_sha256")
    unsigned = {key: item for key, item in value.items() if key != "payload_sha256"}
    if expected_digest != _sha256_value(unsigned):
        raise RuntimeError("Qualitative selection lock payload digest mismatch")
    if value.get("model_outputs_opened_during_selection") is not False:
        raise RuntimeError("Common qualitative selection was not model blind")
    sources = _normalise_sources(source_rows_by_seed)
    if tuple(sorted(sources)) != tuple(sorted(int(seed) for seed in value.get("seeds", []))):
        raise RuntimeError("Qualitative lock seeds differ from the supplied source universe")
    universe = _source_universe_records(sources)
    if _sha256_value(universe) != value.get("source_universe_sha256"):
        raise RuntimeError("Qualitative test sampling universe changed after selection lock")
    # Recompute through the exact global-uniqueness algorithm without writing
    # another lock.
    used: set[tuple[str, str, str, str]] = set()
    expected_records: list[dict[str, Any]] = []
    for seed in value["seeds"]:
        frame = sources[int(seed)]
        for label in (int(item) for item in value["labels"]):
            chosen = 0
            candidates = frame.loc[frame.label_3class == label].copy()
            candidates["selection_digest"] = [
                _rank_digest(
                    salt=COMMON_SELECTION_SALT,
                    seed=int(seed),
                    row=row,
                    label=label,
                )
                for row in candidates.itertuples(index=False)
            ]
            candidates = candidates.sort_values(
                ["selection_digest", *FRAME_KEY_COLUMNS], kind="stable"
            )
            for row in candidates.itertuples(index=False):
                key = tuple(str(getattr(row, column)) for column in FRAME_KEY_COLUMNS)
                if key in used:
                    continue
                used.add(key)
                record = {
                    "outer_seed": int(seed),
                    "partition": "test",
                    "label_3class": label,
                    **{
                        column: str(getattr(row, column))
                        for column in FRAME_KEY_COLUMNS
                    },
                    "selection_rank_within_label": chosen + 1,
                    "selection_digest": str(row.selection_digest),
                    "selection_rule": COMMON_SELECTION_RULE,
                    "model_blind_selection": True,
                    "globally_unique_across_seeds": True,
                }
                expected_records.append(record)
                chosen += 1
                if chosen == int(value["examples_per_label"]):
                    break
    expected = pd.DataFrame(expected_records).sort_values(
        ["outer_seed", "label_3class", "selection_rank_within_label"], kind="stable"
    ).reset_index(drop=True).to_dict(orient="records")
    if expected != value.get("selections"):
        raise RuntimeError("Locked common examples are not the deterministic model-blind selection")
    return value


def _normalise_model_frames(
    model_frames_by_seed: Mapping[int | str, Mapping[str, pd.DataFrame]],
    sources: Mapping[int, pd.DataFrame],
    models: Sequence[str],
) -> dict[int, dict[str, pd.DataFrame]]:
    observed_seeds = {int(seed) for seed in model_frames_by_seed}
    if observed_seeds != set(sources):
        raise ValueError("Model-frame seed keys must exactly match source seed keys")
    output: dict[int, dict[str, pd.DataFrame]] = {}
    for raw_seed, by_model in model_frames_by_seed.items():
        seed = int(raw_seed)
        if set(by_model) != set(models):
            raise ValueError(f"Seed {seed} must provide exactly the requested model tables")
        valid_keys = set(
            map(tuple, sources[seed].loc[:, FRAME_KEY_COLUMNS].astype(str).to_numpy())
        )
        output[seed] = {}
        for model in models:
            frame = by_model[model].copy()
            missing = set(FRAME_KEY_COLUMNS) - set(frame)
            if missing:
                raise ValueError(f"{model}/{seed} frame table lacks: {sorted(missing)}")
            for column in FRAME_KEY_COLUMNS:
                frame[column] = frame[column].astype(str)
            if frame.duplicated(list(FRAME_KEY_COLUMNS)).any():
                raise ValueError(f"{model}/{seed} has duplicate frame keys")
            keys = set(map(tuple, frame.loc[:, FRAME_KEY_COLUMNS].to_numpy()))
            if not keys <= valid_keys:
                raise ValueError(f"{model}/{seed} includes rows outside the locked test membership")
            output[seed][model] = frame.sort_values(
                list(FRAME_KEY_COLUMNS), kind="stable"
            ).reset_index(drop=True)
    return output


def _canonical_failure(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().lower()
    return FAILURE_STATUS_ALIASES.get(text)


def select_failure_examples(
    source_rows_by_seed: Mapping[int | str, pd.DataFrame] | pd.DataFrame,
    model_frames_by_seed: Mapping[int | str, Mapping[str, pd.DataFrame]],
    *,
    models: Sequence[str] = EXPECTED_MODELS,
    statuses: Sequence[str] = CANONICAL_FAILURE_STATUSES,
    status_column: str = "segmentation_abstention_reason",
    include_unobserved: bool = True,
) -> pd.DataFrame:
    """Select deterministic outcome-conditioned failure examples.

    The candidate universe is first restricted to the supplied test membership.
    Quantitative quality values are never used for ordering; only failure type
    and a SHA-256 rank are used.  Missing categories are explicitly recorded.
    """

    sources = _normalise_sources(source_rows_by_seed)
    model_frames = _normalise_model_frames(model_frames_by_seed, sources, models)
    canonical_statuses: list[str] = []
    for status in statuses:
        canonical = _canonical_failure(status)
        if canonical is None:
            raise ValueError(f"Unknown failure status: {status}")
        if canonical not in canonical_statuses:
            canonical_statuses.append(canonical)
    records: list[dict[str, Any]] = []
    for seed, source in sources.items():
        labels = source.set_index(list(FRAME_KEY_COLUMNS))["label_3class"]
        for model in models:
            frames = model_frames[seed][model].copy()
            effective_status_column = status_column
            if effective_status_column not in frames:
                if "abstention_reason" not in frames:
                    raise ValueError(
                        f"{model}/{seed} lacks {status_column} and abstention_reason"
                    )
                effective_status_column = "abstention_reason"
            frames["failure_status"] = frames[effective_status_column].map(_canonical_failure)
            for status in canonical_statuses:
                candidates = frames.loc[frames.failure_status == status].copy()
                if candidates.empty:
                    if include_unobserved:
                        records.append(
                            {
                                "outer_seed": seed,
                                "model": model,
                                "failure_status": status,
                                "observed_status": None,
                                **{column: None for column in FRAME_KEY_COLUMNS},
                                "label_3class": None,
                                "available": False,
                                "selection_digest": None,
                                "selection_rule": FAILURE_SELECTION_RULE,
                                "outcome_conditioned_illustrative": True,
                                "interpretation_note": OUTCOME_CONDITIONED_NOTE,
                            }
                        )
                    continue
                digests: list[str] = []
                for row in candidates.itertuples(index=False):
                    key = tuple(str(getattr(row, column)) for column in FRAME_KEY_COLUMNS)
                    label = int(labels.loc[key])
                    digests.append(
                        _rank_digest(
                            salt=FAILURE_SELECTION_SALT,
                            seed=seed,
                            row=row,
                            label=label,
                            model=model,
                            failure_status=status,
                        )
                    )
                candidates["selection_digest"] = digests
                chosen = candidates.sort_values(
                    ["selection_digest", *FRAME_KEY_COLUMNS], kind="stable"
                ).iloc[0]
                key = tuple(str(chosen[column]) for column in FRAME_KEY_COLUMNS)
                records.append(
                    {
                        "outer_seed": seed,
                        "model": model,
                        "failure_status": status,
                        "observed_status": str(chosen[effective_status_column]),
                        **{column: str(chosen[column]) for column in FRAME_KEY_COLUMNS},
                        "label_3class": int(labels.loc[key]),
                        "available": True,
                        "selection_digest": str(chosen.selection_digest),
                        "selection_rule": FAILURE_SELECTION_RULE,
                        "outcome_conditioned_illustrative": True,
                        "interpretation_note": OUTCOME_CONDITIONED_NOTE,
                    }
                )
    return pd.DataFrame(records).sort_values(
        ["outer_seed", "model", "failure_status"], kind="stable"
    ).reset_index(drop=True)


def write_failure_selection_manifest(
    path: str | Path,
    selection: pd.DataFrame,
    *,
    source_rows_by_seed: Mapping[int | str, pd.DataFrame] | pd.DataFrame,
) -> dict[str, Any]:
    """Write an auditable manifest that explicitly declares outcome conditioning."""

    sources = _normalise_sources(source_rows_by_seed)
    records = _json_compatible(selection.to_dict(orient="records"))
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "selection_type": "outcome_conditioned_failure_gallery",
        "selection_rule": FAILURE_SELECTION_RULE,
        "selection_salt": FAILURE_SELECTION_SALT,
        "outcome_conditioned_illustrative": True,
        "interpretation_note": OUTCOME_CONDITIONED_NOTE,
        "source_universe_sha256": _sha256_value(_source_universe_records(sources)),
        "selections": records,
    }
    payload["payload_sha256"] = _sha256_value(payload)
    _atomic_json(Path(path), payload)
    return payload


def _resolve_path(value: Any, *, root: Path | None = None) -> Path:
    if value is None or pd.isna(value):
        raise ValueError("A required qualitative artifact path is missing")
    path = Path(str(value))
    if not path.is_absolute():
        path = (root if root is not None else PROJECT_ROOT) / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _verify_optional_hash(path: Path, expected: Any, *, label: str) -> str:
    observed = _sha256_file(path)
    if expected is not None and not pd.isna(expected) and str(expected):
        if observed != str(expected):
            raise RuntimeError(f"{label} SHA-256 mismatch: {path}")
    return observed


def _load_source_rasters(
    row: pd.Series, dataset_root: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    required = {"output_image", "output_mask"}
    missing = required - set(row.index)
    if missing:
        raise ValueError(f"Source row lacks raster columns: {sorted(missing)}")
    image_path = _resolve_path(row.output_image, root=dataset_root)
    mask_path = _resolve_path(row.output_mask, root=dataset_root)
    image_sha = _verify_optional_hash(
        image_path, row.get("output_image_sha256"), label="Source image"
    )
    mask_sha = _verify_optional_hash(
        mask_path, row.get("output_mask_sha256"), label="Reference mask"
    )
    with Image.open(image_path) as handle:
        image = np.asarray(handle.convert("RGB")).copy()
    with Image.open(mask_path) as handle:
        reference = np.asarray(handle).copy()
    while reference.ndim > 2 and reference.shape[-1] == 1:
        reference = reference[..., 0]
    if reference.ndim == 3:
        reference = np.any(reference != 0, axis=-1)
    elif reference.ndim == 2:
        reference = reference != 0
    else:
        raise ValueError("Reference mask must be a 2D raster")
    if reference.shape != image.shape[:2]:
        raise ValueError("Reference mask and source image shapes differ")
    return image, reference.astype(bool), {
        "image_path": str(image_path),
        "image_sha256": image_sha,
        "reference_mask_path": str(mask_path),
        "reference_mask_sha256": mask_sha,
    }


def _load_audit_mask(
    row: pd.Series,
    *,
    mask_key: str,
    expected_shape: tuple[int, int],
    audit_root: Path | None,
) -> tuple[np.ndarray, dict[str, str]]:
    if "audit_path" not in row:
        raise ValueError("Model frame table lacks audit_path")
    path = _resolve_path(row.audit_path, root=audit_root)
    digest = _verify_optional_hash(path, row.get("audit_sha256"), label="Audit raster")
    with np.load(path, allow_pickle=False) as saved:
        if mask_key not in saved.files:
            raise ValueError(f"Audit {path} lacks requested mask key {mask_key!r}")
        mask = np.asarray(saved[mask_key])
    while mask.ndim > 2 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim != 2 or mask.shape != expected_shape:
        raise ValueError(f"Audit mask shape {mask.shape} differs from {expected_shape}")
    if not np.isfinite(mask).all():
        raise ValueError("Audit mask contains non-finite values")
    return mask.astype(bool), {
        "audit_path": str(path),
        "audit_sha256": digest,
        "audit_mask_key": mask_key,
    }


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return mask.copy()
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    interior = mask.copy()
    for y_offset in range(3):
        for x_offset in range(3):
            interior &= padded[
                y_offset:y_offset + mask.shape[0],
                x_offset:x_offset + mask.shape[1],
            ]
    return mask & ~interior


def overlay_masks(
    image: np.ndarray | Image.Image,
    *,
    reference_mask: np.ndarray | None = None,
    predicted_mask: np.ndarray | None = None,
    reference_color: tuple[int, int, int] = (40, 210, 70),
    predicted_color: tuple[int, int, int] = (255, 145, 30),
    fill_alpha: float = 0.25,
) -> Image.Image:
    """Overlay green reference and orange prediction with opaque boundaries."""

    rgb = np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("image must have shape HxWx3")
    output = rgb.astype(np.float32, copy=True)
    for mask, color in (
        (reference_mask, reference_color),
        (predicted_mask, predicted_color),
    ):
        if mask is None:
            continue
        binary = np.asarray(mask, dtype=bool)
        if binary.shape != output.shape[:2]:
            raise ValueError("Overlay mask shape differs from image")
        output[binary] = (
            (1.0 - float(fill_alpha)) * output[binary]
            + float(fill_alpha) * np.asarray(color, dtype=np.float32)
        )
        output[_mask_boundary(binary)] = np.asarray(color, dtype=np.float32)
    return Image.fromarray(np.clip(np.rint(output), 0, 255).astype(np.uint8), mode="RGB")


def _titled_tile(image: Image.Image, title: str, tile_size: tuple[int, int]) -> Image.Image:
    width, height = (int(tile_size[0]), int(tile_size[1]))
    if width < 64 or height < 64:
        raise ValueError("tile_size must be at least 64x64")
    title_height = 44
    canvas = Image.new("RGB", (width, height + title_height), (250, 250, 250))
    contained = image.copy()
    contained.thumbnail((width, height), Image.Resampling.LANCZOS)
    left = (width - contained.width) // 2
    top = title_height + (height - contained.height) // 2
    canvas.paste(contained, (left, top))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    lines = str(title).split("\n")[:3]
    for line_index, line in enumerate(lines):
        draw.text((5, 4 + line_index * 13), line[:52], fill=(20, 20, 20), font=font)
    return canvas


def _compose_tiles(
    tiles: Sequence[Image.Image], *, columns: int, background: tuple[int, int, int] = (230, 230, 230),
) -> Image.Image:
    if not tiles or columns < 1:
        raise ValueError("At least one tile and one column are required")
    width = max(tile.width for tile in tiles)
    height = max(tile.height for tile in tiles)
    rows = (len(tiles) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * width, rows * height), background)
    for index, tile in enumerate(tiles):
        canvas.paste(tile, ((index % columns) * width, (index // columns) * height))
    return canvas


def _save_png_atomic(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.png")
    image.save(temporary, format="PNG", optimize=False, compress_level=6)
    temporary.replace(path)


def _source_lookup(sources: Mapping[int, pd.DataFrame]) -> dict[int, pd.DataFrame]:
    return {
        seed: frame.set_index(list(FRAME_KEY_COLUMNS), drop=False)
        for seed, frame in sources.items()
    }


def _model_lookup(
    model_frames: Mapping[int, Mapping[str, pd.DataFrame]],
) -> dict[int, dict[str, pd.DataFrame]]:
    return {
        seed: {
            model: frame.set_index(list(FRAME_KEY_COLUMNS), drop=False)
            for model, frame in by_model.items()
        }
        for seed, by_model in model_frames.items()
    }


def render_common_segmentation_gallery(
    selection_lock: str | Path | Mapping[str, Any],
    source_rows_by_seed: Mapping[int | str, pd.DataFrame] | pd.DataFrame,
    model_frames_by_seed: Mapping[int | str, Mapping[str, pd.DataFrame]],
    *,
    dataset_root: str | Path,
    output_dir: str | Path,
    models: Sequence[str] = EXPECTED_MODELS,
    audit_root: str | Path | None = None,
    predicted_mask_key: str = "selected_mask",
    tile_size: tuple[int, int] = (320, 320),
) -> RenderedGallery:
    """Render original/reference/four-model panels from a verified common lock."""

    sources = _normalise_sources(source_rows_by_seed)
    locked = verify_common_selection_lock(selection_lock, sources)
    frames = _normalise_model_frames(model_frames_by_seed, sources, models)
    source_lookup = _source_lookup(sources)
    model_lookup = _model_lookup(frames)
    dataset = Path(dataset_root).resolve()
    audit_base = Path(audit_root).resolve() if audit_root is not None else None
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output_records: list[dict[str, Any]] = []
    figure_paths: list[Path] = []
    for selected in locked["selections"]:
        seed = int(selected["outer_seed"])
        key = tuple(str(selected[column]) for column in FRAME_KEY_COLUMNS)
        if key not in source_lookup[seed].index:
            raise RuntimeError(f"Locked qualitative frame is absent: {seed}/{key}")
        source = source_lookup[seed].loc[key]
        if isinstance(source, pd.DataFrame):
            raise RuntimeError(f"Non-unique source frame during rendering: {seed}/{key}")
        image, reference, source_meta = _load_source_rasters(source, dataset)
        tiles = [
            _titled_tile(Image.fromarray(image), "Original", tile_size),
            _titled_tile(
                overlay_masks(image, reference_mask=reference),
                "Reference ROI (green)",
                tile_size,
            ),
        ]
        audit_records: list[dict[str, Any]] = []
        for model in models:
            if key not in model_lookup[seed][model].index:
                raise RuntimeError(f"{model}/{seed} lacks locked common frame {key}")
            model_row = model_lookup[seed][model].loc[key]
            if isinstance(model_row, pd.DataFrame):
                raise RuntimeError(f"Non-unique model frame during rendering: {model}/{seed}/{key}")
            prediction, audit_meta = _load_audit_mask(
                model_row,
                mask_key=predicted_mask_key,
                expected_shape=reference.shape,
                audit_root=audit_base,
            )
            status = None
            for column in ("segmentation_abstention_reason", "abstention_reason"):
                if column in model_row and pd.notna(model_row[column]) and str(model_row[column]):
                    status = str(model_row[column])
                    break
            title = f"{model}\nref green | pred orange"
            if status:
                title += f"\nstatus: {status}"
            tiles.append(
                _titled_tile(
                    overlay_masks(
                        image, reference_mask=reference, predicted_mask=prediction
                    ),
                    title,
                    tile_size,
                )
            )
            audit_records.append({"model": model, "status": status, **audit_meta})
        panel = _compose_tiles(tiles, columns=4)
        digest = str(selected["selection_digest"])
        filename = (
            f"common_seed_{seed}_label_{int(selected['label_3class'])}_"
            f"rank_{int(selected['selection_rank_within_label'])}_{digest[:10]}.png"
        )
        figure_path = destination / filename
        _save_png_atomic(panel, figure_path)
        sidecar = {
            "schema_version": SCHEMA_VERSION,
            "selection_type": "model_blind_common_segmentation_gallery",
            "selection_payload_sha256": locked["payload_sha256"],
            "selection": selected,
            "model_blind_selection": True,
            "colors": {"reference": "green", "prediction": "orange"},
            "predicted_mask_key": predicted_mask_key,
            "source": source_meta,
            "model_audits": audit_records,
            "figure_sha256": _sha256_file(figure_path),
        }
        sidecar_path = figure_path.with_suffix(".json")
        _atomic_json(sidecar_path, sidecar)
        figure_paths.append(figure_path)
        output_records.append(
            {
                **selected,
                "figure_path": str(figure_path),
                "figure_sha256": sidecar["figure_sha256"],
                "sidecar_path": str(sidecar_path),
            }
        )
    index_path = destination / "common_gallery_index.csv"
    _atomic_csv(index_path, pd.DataFrame(output_records))
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "gallery": "common_model_blind",
        "selection_payload_sha256": locked["payload_sha256"],
        "models": list(models),
        "figure_count": len(figure_paths),
        "model_outputs_used_for_selection": False,
        "predicted_mask_key": predicted_mask_key,
        "index_sha256": _sha256_file(index_path),
    }
    metadata_path = destination / "common_gallery_metadata.json"
    _atomic_json(metadata_path, metadata)
    return RenderedGallery(tuple(figure_paths), index_path, metadata_path)


def render_failure_gallery(
    source_rows_by_seed: Mapping[int | str, pd.DataFrame] | pd.DataFrame,
    model_frames_by_seed: Mapping[int | str, Mapping[str, pd.DataFrame]],
    failure_selection: pd.DataFrame,
    *,
    dataset_root: str | Path,
    output_dir: str | Path,
    models: Sequence[str] = EXPECTED_MODELS,
    audit_root: str | Path | None = None,
    predicted_mask_key: str = "thresholded_mask",
    tile_size: tuple[int, int] = (280, 280),
) -> RenderedGallery:
    """Render outcome-conditioned failure tiles, grouped into one panel per seed."""

    sources = _normalise_sources(source_rows_by_seed)
    frames = _normalise_model_frames(model_frames_by_seed, sources, models)
    source_lookup = _source_lookup(sources)
    model_lookup = _model_lookup(frames)
    required = {
        "outer_seed", "model", "failure_status", "available",
        "outcome_conditioned_illustrative", *FRAME_KEY_COLUMNS,
    }
    if not required <= set(failure_selection):
        raise ValueError(f"Failure selection lacks: {sorted(required - set(failure_selection))}")
    if not failure_selection.outcome_conditioned_illustrative.astype(bool).all():
        raise ValueError("Every failure example must be marked outcome-conditioned")
    expected_selection = select_failure_examples(
        sources,
        frames,
        models=models,
        statuses=CANONICAL_FAILURE_STATUSES,
        include_unobserved=True,
    )
    comparison_columns = [
        "outer_seed",
        "model",
        "failure_status",
        "observed_status",
        *FRAME_KEY_COLUMNS,
        "label_3class",
        "available",
        "selection_digest",
        "selection_rule",
        "outcome_conditioned_illustrative",
    ]
    supplied_records = _json_compatible(
        failure_selection.loc[:, comparison_columns]
        .sort_values(["outer_seed", "model", "failure_status"], kind="stable")
        .to_dict(orient="records")
    )
    expected_records = _json_compatible(
        expected_selection.loc[:, comparison_columns].to_dict(orient="records")
    )
    if supplied_records != expected_records:
        raise ValueError(
            "Failure selection is not the deterministic outcome-conditioned hash-min result"
        )
    dataset = Path(dataset_root).resolve()
    audit_base = Path(audit_root).resolve() if audit_root is not None else None
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output_records: list[dict[str, Any]] = []
    figure_paths: list[Path] = []
    for seed in sorted(sources):
        subset = failure_selection.loc[failure_selection.outer_seed.astype(int) == seed]
        tiles: list[Image.Image] = []
        first_output_record = len(output_records)
        for status in CANONICAL_FAILURE_STATUSES:
            for model in models:
                matches = subset.loc[
                    (subset.model.astype(str) == str(model))
                    & (subset.failure_status.astype(str) == status)
                ]
                if len(matches) != 1:
                    raise ValueError(f"Expected one failure record for {model}/{seed}/{status}")
                selected = matches.iloc[0]
                if not bool(selected.available):
                    blank = Image.new("RGB", tile_size, (238, 238, 238))
                    draw = ImageDraw.Draw(blank)
                    draw.text((10, 10), "Not observed", fill=(70, 70, 70), font=ImageFont.load_default())
                    tiles.append(_titled_tile(blank, f"{model}\n{status}", tile_size))
                    output_records.append(
                        {
                            "outer_seed": seed,
                            "model": model,
                            "failure_status": status,
                            "available": False,
                            "figure_path": None,
                            "outcome_conditioned_illustrative": True,
                            "interpretation_note": OUTCOME_CONDITIONED_NOTE,
                        }
                    )
                    continue
                key = tuple(str(selected[column]) for column in FRAME_KEY_COLUMNS)
                source = source_lookup[seed].loc[key]
                model_row = model_lookup[seed][model].loc[key]
                if isinstance(source, pd.DataFrame) or isinstance(model_row, pd.DataFrame):
                    raise RuntimeError(f"Non-unique failure frame: {model}/{seed}/{key}")
                image, reference, source_meta = _load_source_rasters(source, dataset)
                prediction, audit_meta = _load_audit_mask(
                    model_row,
                    mask_key=predicted_mask_key,
                    expected_shape=reference.shape,
                    audit_root=audit_base,
                )
                tile = _titled_tile(
                    overlay_masks(image, reference_mask=reference, predicted_mask=prediction),
                    f"{model}\n{status}\nillustrative only",
                    tile_size,
                )
                tiles.append(tile)
                output_records.append(
                    {
                        **_json_compatible(selected.to_dict()),
                        "source_image_sha256": source_meta["image_sha256"],
                        "reference_mask_sha256": source_meta["reference_mask_sha256"],
                        **audit_meta,
                    }
                )
        panel = _compose_tiles(tiles, columns=len(models))
        figure_path = destination / f"failure_gallery_seed_{seed}.png"
        _save_png_atomic(panel, figure_path)
        figure_digest = _sha256_file(figure_path)
        for record in output_records[first_output_record:]:
            record["overview_figure_path"] = str(figure_path)
            record["overview_figure_sha256"] = figure_digest
        figure_paths.append(figure_path)
    index = pd.DataFrame(output_records)
    index_path = destination / "failure_gallery_index.csv"
    _atomic_csv(index_path, index)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "gallery": "outcome_conditioned_failures",
        "selection_rule": FAILURE_SELECTION_RULE,
        "outcome_conditioned_illustrative": True,
        "interpretation_note": OUTCOME_CONDITIONED_NOTE,
        "models": list(models),
        "failure_statuses": list(CANONICAL_FAILURE_STATUSES),
        "predicted_mask_key": predicted_mask_key,
        "figure_count": len(figure_paths),
        "index_sha256": _sha256_file(index_path),
    }
    metadata_path = destination / "failure_gallery_metadata.json"
    _atomic_json(metadata_path, metadata)
    return RenderedGallery(tuple(figure_paths), index_path, metadata_path)


__all__ = [
    "CANONICAL_FAILURE_STATUSES",
    "COMMON_SELECTION_RULE",
    "FAILURE_SELECTION_RULE",
    "FAILURE_STATUS_ALIASES",
    "FRAME_KEY_COLUMNS",
    "OUTCOME_CONDITIONED_NOTE",
    "RenderedGallery",
    "build_common_selection_lock",
    "overlay_masks",
    "render_common_segmentation_gallery",
    "render_failure_gallery",
    "select_common_test_frames",
    "select_failure_examples",
    "verify_common_selection_lock",
    "write_failure_selection_manifest",
]
