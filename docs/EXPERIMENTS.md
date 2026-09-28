# Experiments and execution

This source release contains the executed four-model anatomical-segmentation and strict predicted-ROI classification workflow: YOLO26, custom ViT Method 2, EMCAD/PVT-v2-B0 and prompt-free SAM2-UNet/Hiera-Tiny. Binary classification (BCC) separates normal from the combined papilledema/pseudopapilledema group. Direct three-class classification uses normal, papilledema and pseudopapilledema. The classifiers are fitted separately for the two tasks; no binary classifier is relabelled as a three-class classifier.

## Environment

The recorded environment is Python 3.12.7, PyTorch 2.8.0+cu126, torchvision 0.23.0+cu126, timm 1.0.19 and Ultralytics 8.4.138. `requirements.txt` lists direct dependencies; `requirements-lock.txt` is the captured environment lock. For the CUDA build represented by that lock, configure the PyTorch CUDA 12.6 package index when installing:

```powershell
python -m venv .venv-binary
.\.venv-binary\Scripts\python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cu126 -r requirements-lock.txt
```

PowerShell launchers resolve `.venv-binary/Scripts/python.exe` relative to this repository. Python module entry points can also be called directly from a configured environment.

EMCAD's separately licensed source is supplied by the user through `EMCAD_SOURCE_DIR`; see [third-party acquisition and licensing](../THIRD_PARTY_NOTICES.md). The directory must contain `lib/pvtv2.py` and `lib/decoders.py` at the pinned revision. Both files are hash-checked before either is imported. No restricted EMCAD source is bundled. The release adapter preserves registered state names, four feature stages, normalization, final segmentation head and auxiliary-output ordering. It is independently authored release code, not the original source attested in historical receipts.

## Data and frozen study configuration

The two `config_*.json` files are byte-identical copies of the executed scientific protocols. They retain all numerical parameters, seeds, output namespaces, input hashes and source anchors. They describe the specific study cohort, not an unrestricted dataset-independent template. Authorized use requires the corresponding private image/mask masters, manifest, patient-level split files and pretrained initialization files at the configured paths. Neither patient identifiers nor split membership, images, masks, clinical checkpoints or results are distributed here. [Data and licensing](DATA_AND_LICENSING.md) describes the boundary between source and controlled inputs.

The cohort contract is 91 patients, 182 eyes and 1,274 frames, with seven frames per eye and two eyes per patient. The five fixed patient-level holdouts use seeds 17, 42, 2026, 3407 and 9103. These are repeated holdouts, not independent cross-validation folds. BCC uses failure-aware eye-level balanced accuracy as its primary endpoint; three-class classification uses failure-aware patient-level balanced accuracy. The analyses account for patient clustering and overlapping holdouts. Previously inspected test data make the study exploratory and post hoc.

The preprocessing entry point is:

```powershell
python scripts/preprocess_dataset.py --root PRIVATE_SOURCE_ROOT --output PRIVATE_OUTPUT_ROOT --crop 80 110 920 728 --size 768
```

`PRIVATE_SOURCE_ROOT` and `PRIVATE_OUTPUT_ROOT` are placeholders for authorized local paths. This script produces aligned RGB/mask masters and a private manifest; it must not be pointed at a public repository intended for redistribution. Its generated manifest can contain patient names and original paths. Existing frozen input hashes must not be bypassed by regenerating a different manifest or split under the same study identity.

## Segmentation and binary classification

The staged CLI exposes `prepare`, `preflight`, `train-segmenters`, `build-rois`, `train-classifiers`, `lock`, `evaluate`, `summarize`, `audit`, and separately gated ablation stages. `--help` only describes commands and starts no fit.

```powershell
python -m predicted_roi_study --help
python -m predicted_roi_study prepare --config predicted_roi_study/config_strict_roi.json
python -m predicted_roi_study preflight --model all
```

For each of the five declared seeds, complete the development stages, for example:

```powershell
python -m predicted_roi_study train-segmenters --model all --seed 17
python -m predicted_roi_study build-rois --model all --seed 17
python -m predicted_roi_study train-classifiers --model all --seed 17
python -m predicted_roi_study lock --model all --seed 17
```

The corresponding supervised development launcher is `scripts/run_strict_roi_clean_seed.ps1 -Seed 17`. It deliberately stops at validation lock. Repeat for all five declared seeds. Resume requires the explicit `-ResumeIncompleteSeed` switch and unchanged namespace, source, configuration, input and split hashes.

Segmenters learn anatomical masks, not three disease-specific mask classes. Patient-level cross-fitting generates training ROI predictions out of fold. The final segmenter supplies validation/test predictions. Classifiers consume only accepted predicted-ROI pixels; the hard mask is reapplied after resizing and outside-ROI pixels are neutralized. Neither ground-truth masks nor full-frame pixels rescue failed localization at inference. An eye requires at least four valid frames; a patient requires both evaluable eyes.

Each model/seed unit has two separately fitted classifier strategies on the identical ROI stream: `model_specific` (primary) and `standardized_resnet18` (secondary). Calibration and decision thresholds are selected on validation only. The global binary gate requires all 20 composite model/seed locks, representing 40 classifier strategies and their level-specific validation settings, before any test inference.

For a new run prepared and trained with **this release**, use the separate compatibility entry point for evaluation:

```powershell
python -m predicted_roi_study.release_evaluate evaluate --model all
python -m predicted_roi_study.release_evaluate summarize
python -m predicted_roi_study.release_evaluate audit
```

This entry point retains the original gate, receipt and artifact checks. After validating current source/configuration and all 20 lock chains, it supplies a byte-identical `protocol/test_access.json` alias for the canonical `state/test_access_opened.json` sentinel expected by the historical receipt validator. It refuses divergent aliases, changed locks and historical archive identity. It does not change inference, calibration, ROI selection, statistics or source receipt content. It owns the same exclusive orchestration locks as the launchers. `--dry-run` performs the original prerequisite checks without creating the alias or running inference.

The unmodified historical master launcher remains in `scripts/run_strict_roi_4model_five_seeds.ps1` as an execution record. For the release-specific sequence, use the development launcher followed by `release_evaluate` as shown above. Do not use the master launcher to bypass this documented evaluation entry point.

## Historical binary evaluation-only recovery

`scripts/recover_strict_roi_evaluation_v1.py` and its amendment document preserve the actual archive-bound recovery implementation. That script requires the original source/configuration/gate hashes and the frozen partial-evaluation inventory. The yolo26/seed17 evaluation receipt was an inventory-verified backfill; later units were unchanged evaluations. This historical replay tool is not a generic resume script for the source release and is expected to reject release fingerprints. Historical receipts are never rewritten to match this distribution.

`provenance/core_source_manifest.json` records original versus released source hashes, the original binary aggregate fingerprint and release adaptations. The four-family registry, attribution notices, release evaluation entry point and independently authored EMCAD adapter make this release intentionally distinct from the historical fingerprint. A byte-identical configuration does not imply byte-identical source provenance.

## Three-class classification with frozen upstream localization

The three-class protocol imports and audits development-only segmentation/ROI artifacts from the completed binary study. It does not retrain segmentation. The frozen upstream source anchors in its configuration are part of the executed study identity, and authorized replay verifies them exactly. A newly generated upstream run has a different provenance identity and must not be presented as satisfying the published archive's fixed anchors merely because its model settings match.

The staged interface is:

```powershell
python -m threeclass_roi_study validate-config
python -m threeclass_roi_study plan
python -m threeclass_roi_study prepare
python -m threeclass_roi_study import-upstream --model all --seed all
python -m threeclass_roi_study preflight --model all
python -m threeclass_roi_study train --model all --seed all --strategy all
python -m threeclass_roi_study lock --model all --seed all --strategy all
python -m threeclass_roi_study open-test
python -m threeclass_roi_study evaluate --model all --seed all --strategy all
python -m threeclass_roi_study summarize
python -m threeclass_roi_study audit
```

`python -m threeclass_roi_study run-core` and `scripts/run_threeclass_roi_4model_five_seeds.ps1` execute the same gated sequence. Forty model/seed/strategy validation locks must pass before test access. Binary and three-class outcomes are reported separately, with raw/calibrated discrimination, failure-aware and conditional classification, coverage, calibration, paired comparisons and patient-cluster uncertainty. “MCC” in metric columns means Matthews correlation coefficient; it must not be confused with the three-class task abbreviation.

## Reporting and verification

`predicted_roi_study/reporting.py`, `qualitative.py` and `resource_reporting.py` provide binary/segmentation reporting. `threeclass_roi_study/reporting.py`, `publication_exports.py` and `publication_figures.py` implement three-class and publication-oriented exports. Publication assembly scripts are documented separately in this repository. Output inventories and receipts record the executed analyses.

Patient-free tests can be run from the repository root without images, splits, checkpoints or external EMCAD source:

```powershell
python -B -m pytest --import-mode=importlib -p no:cacheprovider predicted_roi_study/tests/test_external_emcad_release.py predicted_roi_study/tests/test_release_evaluation_gate.py predicted_roi_study/tests/test_model_specific_classifiers.py predicted_roi_study/tests/test_roi.py predicted_roi_study/tests/test_metrics.py threeclass_roi_study/tests/test_data_models.py threeclass_roi_study/tests/test_metrics.py threeclass_roi_study/tests/test_cli.py -k "not production_gradient" -q
```

`--import-mode=importlib` keeps same-named test modules in the two packages distinct. Other retained tests include private-input and historical-archive integration checks, which require their attested inputs. Release verification uses synthetic tensors and temporary files to check shape/gradient/ROI-invariance contracts, classifier registries, calibration/aggregation metrics and fail-closed gate behavior.
