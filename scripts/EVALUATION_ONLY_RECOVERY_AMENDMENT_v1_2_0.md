# Evaluation-only recovery amendment — strict ROI protocol v1.2.0

## Status and purpose

This document records a post-test-access, implementation-only amendment for
`strict_predicted_roi_binary_4model_v1_2_0_clean`. It is not a retrospective
change to the scientific design, model set, data split, training procedure,
validation policy, estimands, thresholds, calibration, abstention policy, or
analysis plan.

The global test gate opened only after all 20 predeclared model/seed composite
locks had been completed. The first evaluation unit (`yolo26`, seed `17`)
then completed both classifier branches and wrote its outputs, but the immutable
evaluation receipt could not be written because the receipt validator searched
for an artifact ending in `protocol/test_access.json`, whereas the canonical
gate created and returned by the protocol is
`state/test_access_opened.json`. The attempt and canonical gate already carry
the same SHA-256 digest; the failure is therefore a pathname-contract mismatch,
not a model, data, prediction, metric, or gate-content failure.

This amendment permits only the minimum compatibility mechanism required to
bind evaluation receipts to the already-opened canonical gate and to continue
the predeclared exhaustive evaluation. It does not authorize retraining or any
other scientific change.

The executable implementation record is
`evaluation_gate_path_compatibility_v2`. It supersedes the prepared-only v1
manifest (`09e22e62a8f63cbec1f3e78a04fd2c8a907a02e835f0e9a6a5852cfdee2fcc90`)
before any evaluation receipt was created. V2 does not recapture the first
unit as a new baseline: it reuses and verifies the v1-frozen 532-file inventory
(`57ae65513380fa01d977044c256eb47ab65a679bba17565c40b17f741f0ebe2f`)
against the current evaluation tree before it creates its own manifest.

## Frozen incident identity

Recovery is valid only for the following exact state:

- Study ID: `strict_predicted_roi_binary_4model_v1_2_0_clean`
- Protocol version: `1.2.0`
- Configuration SHA-256:
  `f9ad4cfd5e4222b50dfd91f9084c7ffe4c8ba60f12e8ff54d1d4c623c324c7ad`
- Canonical configuration SHA-256:
  `f16bfcd4715c7a823dcef671f9b23681a62fd217029aaab260ffcab57806e7f0`
- Protocol SHA-256:
  `61f6f6b90d08ee673d590017ec158c6e7a34885fcc1cf954c882dd89318df74e`
- Original locked code SHA-256:
  `914ec1505a0d417b8b6766b3663160673b94d5ce107da68f81d3142bbf493776`
- Canonical test-access gate:
  `strict_roi_results_4model_v1_2_0/state/test_access_opened.json`
- Canonical gate SHA-256:
  `4fb2bb8f298e4fa131d848c72db8e3b6fc0e533f478e37e8d30bde3b33c18bdd`
- Fingerprinted source snapshot SHA-256:
  `b8c2b31af782c7873198b03e8d0c24c4e10888c562f98cacc9aa9c13cb9e7b6e`
- Failed-run stderr log SHA-256:
  `94e42406edcc16b2c7cb43f2991ae804009085ef34972ca5fceb032c82b7d9e8`

Any disagreement with one of these identities is terminal and requires a new
study namespace rather than this recovery.

## Permitted scope

The recovery may perform only the following actions, in this order:

1. Revalidate the original prepare receipt, all four preflight receipts, all
   20 complete per-model/per-seed predecessor chains, every artifact named by
   those receipts, and the 20 composite locks using the unmodified v1.2.0
   verifier.
2. Revalidate the canonical test-access sentinel against the current locked
   runtime context and the exact 20-chain hash map stored when the gate opened.
3. Create one byte-identical compatibility copy of the canonical gate at a
   pathname ending in `protocol/test_access.json` within a versioned amendment
   directory.
4. Freeze an immutable amendment manifest that binds the canonical gate, the
   compatibility copy, the original receipt chains, the recovery runner, the
   original failure evidence, and the pre-recovery first-unit output inventory.
5. Create the missing evaluation receipt for the already-complete first unit
   from its existing artifacts without executing segmentation or classification
   inference again.
6. Evaluate every remaining predeclared model/seed unit, in the locked order,
   with the unchanged v1.2.0 evaluation implementation.
7. Produce the originally predeclared primary and standardized summaries only
   after all 20 dual-branch evaluation receipts validate.
8. Run the ordinary, unmodified final audit.

No training, refitting, threshold selection, calibration fitting, ROI-policy
selection, model selection, hyperparameter selection, seed exclusion, system
exclusion, or analysis-plan change is permitted.

## Invariants

The recovery runner must fail closed unless all of the following remain true:

1. **No core-code mutation.** Files included by the original code fingerprint,
   including every file under `predicted_roi_study`, the imported legacy model
   sources, and `scripts/run_strict_roi_clean_seed.ps1`, remain byte-identical.
   The live code fingerprint must equal the original locked code SHA-256 above.
2. **No protocol/config mutation.** The live protocol and configuration hashes
   must equal the frozen values above.
3. **No upstream mutation.** Every prepare, preflight, segmenter, OOF ROI,
   classifier, and composite-lock receipt must validate with its original
   artifact inventory. All checkpoint, split, OOF index, optimization index,
   eligibility ledger, calibration, threshold, and ROI-policy hashes remain
   unchanged.
4. **Canonical gate immutability.** The canonical gate is never replaced,
   edited, renamed, or regenerated. Its bytes and digest must remain unchanged.
5. **Complete lock-set identity.** The canonical gate's
   `unit_receipt_chain_sha256` map must exactly equal a fresh verification of all
   20 model/seed chains.
6. **Alias content identity.** The compatibility gate must be a regular,
   immutable, byte-for-byte copy of the canonical gate. Its digest must equal
   the canonical gate digest before every receipt operation.
7. **Original attempt identity.** Every evaluation attempt must remain bound to
   the same configuration, original code fingerprint, model/seed lock, two
   classifier checkpoints, and canonical gate digest.
8. **Exhaustive fixed plan.** Recovery must evaluate all four locked models
   (`yolo26`, `vit_method2`, `emcad`, `sam2_unet`) for all five locked seeds
   (`17`, `42`, `2026`, `3407`, `9103`) and both predeclared classifier
   strategies. No result-dependent branching or early model removal is allowed.
9. **Immutable receipts.** Existing valid receipts are verified and skipped,
   never overwritten. A present invalid receipt is a terminal error, not an
   incomplete unit.
10. **No hidden fallback.** Classification remains strict predicted-ROI only;
    no full-image or ground-truth-ROI fallback is introduced.
11. **Concurrency exclusion.** Recovery must acquire and hold both ordinary
    launcher locks for its entire lifetime, in the ordinary master-to-seed
    order: `five_seed_core_exclusive.lock`, then
    `clean_study_exclusive.lock`. This excludes the five-seed master, direct
    seed launcher, and a second recovery runner from concurrent mutation.
12. **Amendment provenance.** The recovery runner's own digest and the
    amendment manifest are recorded. Evaluation and summary receipts must
    attest the amendment manifest in addition to their ordinary artifacts.

## Compatibility alias mechanism

The canonical gate remains:

```text
strict_roi_results_4model_v1_2_0/state/test_access_opened.json
```

The compatibility copy is created at the exact legacy path expected by the
unchanged validator:

```text
strict_roi_results_4model_v1_2_0/protocol/test_access.json
```

Before returning the compatibility path to the unchanged evaluation engine, the
recovery binding must call the original `open_test_access` implementation. That
call revalidates the original code/config/protocol context and all 20 lock
chains against the already-opened canonical sentinel. The recovery binding then
requires byte equality and equal SHA-256 values between canonical and
compatibility files and returns only the compatibility pathname.

This works without weakening or replacing the original validator: evaluation
attempts bind the gate by SHA-256, while the validator additionally requires an
artifact pathname ending in `protocol/test_access.json`. The compatibility file
supplies that pathname while preserving the exact canonical content. Every
evaluation receipt should also include the canonical gate and amendment
manifest as additional hashed artifacts. The standard unmodified receipt
verifier must be able to verify each resulting receipt in a clean process.

The compatibility mechanism is additive and evaluation-only. It must not be
placed inside the fingerprinted core package and must not monkey-patch the
protocol receipt validator. A proper source-level pathname correction belongs
to a future protocol/code version and a new output namespace.

## First-unit receipt backfill

The first unit, `yolo26` with seed `17`, is a special recovery case:

- `evaluation/attempt.json` has `status: complete`.
- Both locked classifier strategies completed.
- The test ROI index and both complete output bundles exist.
- The attempt's `test_access_sha256` equals the canonical gate digest.
- No evaluation receipt exists because failure occurred during receipt semantic
  validation after the attempt and result files had been finalized.

The recovery must inventory and hash these files before taking any action. It
must then reconstruct the same artifact list and metadata that the unchanged
evaluation engine would pass to `write_stage_receipt`, substituting only the
byte-identical compatibility gate pathname and adding the canonical gate and
amendment provenance. The unchanged receipt writer and unchanged semantic
validator must create and verify the receipt.

Segmentation inference, classifier inference, calibration, aggregation,
bootstrapping, plotting, and metric calculation must **not** be rerun for this
unit. If the attempt is not complete, if any expected output is absent, if an
artifact hash changes, or if any attempt/lock/gate identity differs, backfill is
forbidden and recovery must stop.

Before and after receipt creation, the live first-unit tree must have exactly
the same path set, byte sizes, and SHA-256 values as the v1-frozen inventory.
No newly captured v2 inventory may replace that baseline. Receipt-only
backfill is limited to `yolo26` seed `17`; any other complete attempt lacking a
receipt fails closed and requires a separately audited recovery decision.

For a later crash during one of the remaining units, continuation is allowed
only under the pre-existing failed-attempt rule: the attempt must carry the
same original code, configuration, gate, and lock/checkpoint identities. A
complete attempt without a receipt may be backfilled under the same strict
checks; an existing valid receipt is only verified and skipped.

## Threat model and controls

| Threat | Required control |
|---|---|
| A core source edit silently changes inference or metrics | Preserve and compare the original code fingerprint; do not edit fingerprinted files |
| A config, protocol, split, checkpoint, ROI cache index, classifier, threshold, or lock is changed after test access | Revalidate the entire original receipt DAG and every attested artifact before recovery and before each evaluation receipt |
| A different file is substituted as the compatibility gate | Require canonical open-gate validation, byte equality, and the frozen canonical SHA-256 on every use |
| The canonical gate is rewritten to match a new state | Treat the frozen gate digest and 20-chain map as constants; never regenerate or replace the sentinel |
| The recovery is used to select models after observing the first result | Freeze and execute the full 20-unit, two-strategy plan with no conditional exclusions |
| The first unit is recomputed and overwritten | Hash its pre-recovery tree and perform receipt-only backfill; never invoke inference or export functions for that unit |
| A valid evaluation receipt is overwritten | Use immutable receipt semantics: verify-and-skip existing receipts; fail on invalid receipts |
| A concurrent launcher corrupts or races the namespace | Hold the existing exclusive orchestration lock for the whole recovery |
| The external recovery runner changes between restarts | Record its SHA-256 in the amendment and require the live runner hash to match |
| The amendment record is modified | Include it as a hashed artifact in each recovered evaluation receipt and in the summary receipt |
| Test results influence refitting, calibration, thresholds, ROI policy, or eligibility | Call only the unchanged evaluation/summarization code and prohibit all training/selection entry points |
| A partial recovery is presented as a completed study | Final summary and audit require all 20 valid dual-strategy evaluation receipts (40 system evaluations) |

## Verification requirements

Before starting GPU evaluation, recovery must pass at least these checks:

1. A read-only preflight reports 20/20 valid composite chains, the frozen live
   code/config/protocol hashes, the unchanged canonical gate, exactly one
   complete unreceipted attempt (`yolo26`, seed `17`), and the remaining fixed
   evaluation plan.
2. A synthetic pathname regression demonstrates that the unchanged validator
   rejects the canonical pathname as currently implemented, accepts only a
   byte-identical artifact at the compatibility suffix, and rejects a
   one-byte-different alias.
3. A backfill test proves that inference functions are not called for a complete
   attempt and that missing or altered result artifacts are rejected.
4. After first-unit backfill, an ordinary unpatched process successfully runs
   `verify_stage_receipt` for `yolo26`, seed `17`.
5. The live core code fingerprint remains the original value after the external
   recovery files and amendment are created.

After completion:

- all 20 evaluation receipts must verify in an ordinary unpatched process;
- they must contain exactly both classifier branches, giving 40 predeclared
  system evaluations;
- the canonical gate and all pre-evaluation receipt/artifact hashes must still
  match the amendment baseline;
- the normal summary receipt and normal final audit must pass; and
- no training or validation-lock receipt may have a new timestamp or digest.

## Reporting disclosure

The manuscript and reproducibility supplement must disclose this event as a
post-test-access software-path amendment. A suitable disclosure is:

> After all 20 model-seed validation locks had been frozen and the global test
> gate had opened, evaluation stopped after the first model-seed output bundle
> because the receipt validator expected a legacy gate pathname, while the gate
> creator used the canonical v1.2.0 pathname. Gate content, lock hashes,
> checkpoints, predictions, and analysis parameters were unchanged. We used a
> versioned evaluation-only amendment that supplied a byte-identical,
> hash-verified compatibility alias, backfilled the already-complete first
> receipt without recomputation, and exhaustively evaluated all prespecified
> systems. No retraining, refitting, threshold/calibration/ROI-policy tuning,
> model selection, or test-dependent exclusion was performed.

The amendment identifier, amendment manifest hash, recovery-runner hash,
canonical gate hash, first-unit pre-recovery inventory hash, and final audit
result should be retained with the study artifacts and made available with the
reproducibility materials.
