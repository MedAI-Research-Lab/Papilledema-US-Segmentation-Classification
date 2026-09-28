# Data and licensing

## Research workflow and data boundary

The software supports anatomical segmentation, binary normal-versus-combined-abnormal classification, and direct normal/papilledema/pseudopapilledema classification from predicted regions of interest. Patient-grouped partitioning keeps a patient's eyes and frames together. The three-class workflow reuses locked anatomical segmenters and trains separate diagnostic classifiers. Local manifests, ROI caches, calibration records and evaluation receipts record each run's inputs and decisions.

Clinical inputs remain under the data controller's access and ethics arrangements. This repository distributes implementation code and public dependency provenance, not the clinical dataset or a right to use it. Real ultrasound frames, annotation masks, videos, patient-to-image linkage, split memberships, per-patient predictions and trained study checkpoints are kept outside the release. Any demonstrative identifiers or generated test fixtures are for software testing and must not be substituted for clinical records.

For a new dataset, use study-specific pseudonymous identifiers and keep the re-identification key in a separate controlled location. Pseudonymization alone does not make linked clinical records anonymous. Inspect burned-in image text and image/video metadata before sharing selected illustrations. De-identification and permission to publish an illustration are separate requirements.

Run outputs may contain pseudonymous patient IDs, frame paths, dates, hardware information and absolute local directories. Keep output roots, weight caches and original provenance sidecars outside version control. Publish aggregate exports only after checking their contents and the applicable disclosure permissions; a filename such as `summary` or a `.gitignore` rule is not itself evidence that a file is safe to share. Source-snapshot ZIPs should receive the same review as the files they contain.

## Software and dependency rights

The [third-party notices](../THIRD_PARTY_NOTICES.md) identify the executed model families, source revisions and applicable licenses. This release does not apply a new blanket license to the authors' study-specific code or alter the rights attached to third-party components, pretrained weights, clinical inputs or figures.

EMCAD is supplied from the user's own licensed checkout through `EMCAD_SOURCE_DIR`. The adapter verifies the pinned `lib/decoders.py` and `lib/pvtv2.py` files; EMCAD source is not bundled or automatically downloaded. SAM/Hiera attribution, inherited U-Net notices and Ultralytics' AGPL license are retained explicitly. Authors and downstream distributors remain responsible for obtaining permissions applicable to their intended distribution and use.

## Reproducing an analysis

Prepare the prescribed environment and obtain the required official initialization files before a controlled run. Retain source and weight hashes, configuration, patient-grouped splits and the generated locks with the private research record. Model fitting, ROI selection, calibration and test evaluation use the separate stages described in the workflow documentation. The source-only release excludes private and redistribution-restricted material, so its package fingerprint is distinct from an original experimental snapshot; original run records remain unchanged.

These are research workflows. Their study reports describe exploratory internal repeated-holdout evaluations, including structural abstentions and patient-level dependence. Running the software does not establish clinical diagnostic validity, regulatory clearance or permission to replace clinical assessment.
