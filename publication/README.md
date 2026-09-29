# Publication reproduction

This directory reproduces the final publication tables and figures from saved aggregate results. It does not train models, load model checkpoints, select a new operating point, refit calibration, or rerun statistical inference.

## Public figures

From the repository root, install `publication/requirements-publication.txt` in a separate Python environment. The complete figure command also uses Node.js with Sharp and Poppler's `pdftoppm`. The tested native versions are Sharp 0.35.4 and Poppler 26.07.0. Arial regular and bold must be supplied from a locally licensed installation; font files are not distributed. Some plots use Matplotlib's bundled DejaVu Sans. No font enlargement or restyling is applied.

```sh
python -m publication.reproduce --edition 3000 --output outputs/publication \
  --font-regular /local/fonts/arial.ttf --font-bold /local/fonts/arialbd.ttf \
  --node node --sharp /local/node_modules/sharp --pdftoppm pdftoppm
python -m publication.verify --replay-output outputs/publication
```

Use `--edition 5000` for the other edition's caption and numbering inventory. The underlying 62 public figure assets are identical between editions. Original asset filenames remain stable; their final manuscript or supplementary figure numbers are recorded in `provenance/figures_3000.json` and `figures_5000.json`. Historical `S12_...`/`S13_...` filenames therefore do not define the final figure numbering. Each inventory record includes the final caption and note.

The five rendering groups are independently selectable with `--groups classification segmentation legacy binary methods`. They produce 42 classification figures, 8 segmentation figures, 3 saved-summary figures, 8 binary calibration/risk figures, and 1 methods schematic. The methods group also exports editable Draw.io, SVG and PDF versions. The seed confusion panels retain their original 3-by-1 layout (three seeds in part A and two seeds in part B); ROC/PR panels retain their 2-by-1 layout. The output `replay_audit.json` records original and regenerated SHA-256 hashes.

`replay_figures.py` is a portable aggregate-data adapter. It calls the archived plotting functions, replacing only their private prediction-file reads with saved aggregate coordinates/counts. `original_generators/` contains the executed generator dependency chain with runtime paths parameterized. The legacy manuscript generator is represented by exact figure/helper function extracts, not historical manuscript prose. Original and packaged source hashes and adaptation descriptions are recorded in `provenance/generator_sources.json`. Archived modules retain historical interfaces; use the public CLI above for aggregate replay.

## Tables

```sh
python -m publication.tables --edition 3000 --output outputs/tables_3000
python -m publication.tables --edition 5000 --output outputs/tables_5000
```

Each public edition exports 30 exact table records: 5 main tables and 25 supplementary tables. CSV and Markdown exports preserve cell text, captions, notes and numbering, but do not reproduce Word typography. The clinical-provenance table is rendered from an authorized local supplementary document:

```sh
python -m publication.tables --edition 3000 --output /private/table_export \
  --supplement-docx /private/Supplementary_3000.docx
```

This local route restores the complete 31-table set after verifying the document checksum. Use the corresponding 5000-edition document for `--edition 5000`. Clinical-source extraction requires an output outside the repository. Public table numbering deliberately retains the clinical table's position: S1 in the 3000 edition and S2 in the 5000 edition. `tables/numbering_*.json` maps historical supplementary labels to final edition labels.

## Qualitative clinical panels: local inputs only

The six qualitative illustration assets are generated from authorized local study data, not from public aggregates. No ultrasound pixels, masks, selected-case manifests, participant identifiers, or prediction-level files are distributed here.

```sh
python -m publication.qualitative_local --study-root /private/original_study \
  --output /private/qualitative_figures \
  --font-regular /local/fonts/arial.ttf --font-bold /local/fonts/arialbd.ttf
```

The private study root must contain `çalışma_ds/manifest.csv` with its referenced images and reference masks, and `strict_roi_results_4model_v1_2_0/` with the frozen configuration, seed split, segmentation summary, per-model frame evaluations, and saved mask audit files. The generator verifies saved masks and metrics before selecting outcome-selected illustrative examples; it performs no model inference or mask editing. Its original four-model by three-case panel and individual three-case panels are retained. The entire output is private: it includes clinical pixels in PNG/SVG and selection provenance with identifiers. The wrapper refuses any output inside this repository. Source paths are resolved under the supplied study root; the output itself may be anywhere outside the repository.

## Aggregate interpretation

- Seeds are 17, 42, 2026, 3407 and 9103. The repeated patient holdouts overlap; they are not independent cohorts and are not pooled. Five-seed whiskers/cells are descriptive sample SD (`ddof=1`), not confidence intervals.
- Each test split comprises 18 patients, 36 eyes and 252 frames. Eye confusion matrices retain all intended eyes, including abstentions: normal/abnormal denominators are 20/16; normal/papilledema/pseudopapilledema denominators are 20/8/8. Eye labels inherit the patient diagnosis. Three-class patient confusion counts are supplied separately.
- The primary binary endpoint is eye-level; the primary three-class endpoint is patient-level. Three-class eye confusion displays are secondary. Binary abnormal includes papilledema and pseudopapilledema, not papilledema alone.
- ROC/PR and reliability plots describe evaluable observations. Risk–coverage retains intended-unit denominators and saved ranking, appending structural abstentions as errors. Public bin exports contain only nonempty bins; replay does not synthesize observations in empty bins.
- Segmentation overlap and boundary metrics have different validity rules. Dice/IoU include failed postprocessed masks; boundary distances require valid nonempty masks. Raw/postprocessed boundary populations can differ, so boundary improvement is not attributable solely to contour regularization. Distances are in processed-image pixels.
- `segmentation_all_360_paired_comparisons.csv` preserves 12 separate families of 30 contrasts (6 model pairs × 5 seeds) with the recorded common-frame/patient denominators, bootstrap intervals and Holm-adjusted tests. It is not a single 360-test family. Binary and three-class inferential exports likewise retain their saved family structure.

The 24 aggregate CSVs are byte-exact source copies. Row counts, columns, study-relative source paths and checksums are in `provenance/aggregate_sources.json`. These are research outputs from exploratory internal repeated-holdout analyses.

## Maintainer export and verification

`prepare_bundle.py --study-root LOCAL_ARCHIVE` rebuilds the public archive from authorized local sources. Its explicit table allowlist excludes the clinical-provenance record, and its CSV checks reject participant/path fields. It exports neither DOCX/PDF manuscripts nor clinical images. This command is not needed by users of the public bundle.

The release test reproduced all 62 public PNGs byte-for-byte against the final document assets. Both public 30-table exports and local complete 31-table exports passed CSV round-trip checks. Representative mean/SD and seed confusion matrices, ROC/PR, raw/postprocessed segmentation, reliability, risk–coverage and methods-flow panels were visually checked at full size. Font rasterization and renderer versions can affect byte identity on another platform without altering the plotted values; the recorded hashes expose such differences.
