# Papilledema US Segmentation and Classification

Analysis and publication-generation code for **From anatomical segmentation to diagnostic classification in transorbital ultrasound: an exploratory study of papilledema and pseudopapilledema**.

The workflow evaluates anatomical segmentation separately from diagnostic classification. A predicted anatomical region of interest is passed to independently trained binary and direct three-class classifiers. Patient-grouped development, out-of-fold training ROIs, validation-only model selection, and a global test-access gate organize the analysis.

Repository: <https://github.com/MedAI-Research-Lab/Papilledema-US-Segmentation-Classification>

## Study implementation

| Component | Implementation |
|---|---|
| Anatomical segmentation | YOLO26, custom ViT Method2, EMCAD with PVT-v2-B0, and prompt-free SAM2-U-Net with Hiera-Tiny |
| Binary classification | Normal versus papilledema/pseudopapilledema combined |
| Three-class classification | Normal, papilledema, and pseudopapilledema |
| Classifier strategies | Model-specific classifier and separately fitted ResNet-18 comparator |
| Repeated partitions | Seeds 17, 42, 2026, 3407, and 9103; patient-grouped train/validation/test membership |
| ROI policy | Predicted ROIs; localization failures retained as structural abstentions in failure-aware evaluation |
| Primary diagnostic endpoints | Eye-level binary and patient-level three-class balanced accuracy |
| Main confusion figures | Eye-level confusion matrices summarized across five seeds |
| Statistical outputs | Seed-level and mean/SD summaries, patient-cluster uncertainty, paired comparisons and multiplicity-adjusted results |

The cohort configuration specifies 91 participants, 182 eyes and 1,274 frames. The binary and three-class protocols retain their respective frozen configuration and source-identity records.

## Repository guide

| Directory | Contents |
|---|---|
| `predicted_roi_study/` | Segmentation, out-of-fold ROI creation, binary classification, selection locks, evaluation and reporting |
| `threeclass_roi_study/` | Frozen-development import, direct three-class fitting, 40 validation locks, global test gate and reporting |
| `binary_study/models/` | Four model-family adapters and supporting model utilities |
| `scripts/` | Dataset preprocessing and executed orchestration/evaluation scripts |
| `publication/` | Figure generators, final manuscript/supplementary table records, aggregate source data and reproduction commands |
| `provenance/` | Experimental source identities, release-file identities and pinned upstream dependency records |
| `docs/` | Controlled-experiment and data/dependency instructions |
| `licenses/` | Applicable upstream license texts |

## Environment

The experimental environment used Python 3.12.7, PyTorch 2.8.0 with CUDA 12.6, torchvision 0.23.0, timm 1.0.19 and Ultralytics 8.4.138. Exact package records accompany the source. Use a separate environment for publication rendering and follow [publication instructions](publication/README.md) for its dependencies and commands.

The four-model experimental workflow and its commands are documented in [EXPERIMENTS.md](docs/EXPERIMENTS.md). Configurations retain the study's manifest, split, source and weight hashes. Model fitting uses controlled local study inputs and the prescribed pretrained caches. The three-class test gate opens only after all 40 classifier validation locks pass.

EMCAD uses a locally acquired, pinned upstream checkout selected through `EMCAD_SOURCE_DIR`. See [third-party notices](THIRD_PARTY_NOTICES.md) for source acquisition, attribution and component-specific terms. The release keeps original experimental provenance separate from packaging adaptations and does not modify historical receipts.

## Tables and figures

The publication bundle links each final figure group to its generation code and retains separate numbering maps for the 3000- and 5000-word editions. Each edition has five main tables, five main figures, 26 supplementary tables and 13 supplementary figure groups. Thirty tables per edition have public presentation records with captions, cells and notes. The clinical-source table is extracted from an explicitly supplied local supplementary document when exporting the complete 31-table set.

Aggregate numerical exports support reproduction of segmentation summaries, binary and three-class confusion matrices, ROC/precision–recall curves, calibration, risk–coverage plots and statistical tables. Qualitative segmentation-panel code takes an explicitly selected private study directory. Clinical pixels and participant-level records stay in the controlled study archive; public aggregate files are supplied independently of those inputs.

See [publication/README.md](publication/README.md) for the complete commands, figure map and table export instructions. Generated outputs are written under an ignored output directory. The original executed generators and the portable replay adapters have separate provenance records.

Quick table export from the repository root:

```bash
python -m publication.tables --edition 3000 --output outputs/tables_3000
python -m publication.tables --edition 5000 --output outputs/tables_5000
```

Numerical figure replay uses `python -m publication.reproduce` with the locally installed fonts and renderer paths specified in the publication guide. This command reproduces 62 nonclinical figure panels; the six qualitative ultrasound panels use the separate private-input command.


## Citation

Use the author metadata in [CITATION.cff](CITATION.cff) when citing this software. The manuscript's code-availability statement identifies this repository:

> The analysis code and scripts used to generate the tables and figures are be available at https://github.com/MedAI-Research-Lab/Papilledema-US-Segmentation-Classification.

## Data and component terms

Access to clinical data and study-trained checkpoints is governed by the corresponding institutional arrangements. The public package contains analysis code, aggregate publication inputs and dependency provenance. See [data and licensing](docs/DATA_AND_LICENSING.md) and [third-party notices](THIRD_PARTY_NOTICES.md). Component-specific licenses and copyright notices are retained.
