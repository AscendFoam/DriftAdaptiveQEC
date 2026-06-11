# T47: Paper ablation result-pack and material ledger

## Status

- Proposed by Captain on `2026-05-19`
- Current phase: `Phase 2: Controlled Development`
- Decision state: `Go`
- Task type: docs-only planning gate
- Current gating context: `T56` has closed, and any execution of this task must remain hedge-conditioned by the `T56` claim boundary

## Why This Task Exists

`T44` shows that the paper is still missing a stable, paper-grade figure/result/material pack.

This task exists to freeze the minimal set of ablation/result/material items the paper actually needs, without pretending the evidence already exists.

After `T55` and `T56`, this task is no longer a generic "next paper task". It is a downstream paper-material lane that may only proceed after the mechanism claims are explicitly reframed and only if the hedge wording is preserved.

## Goal

Produce a paper-pack ledger that answers, in writing:

1. which figures and tables are ready
2. which are partial
3. which are missing
4. which ablation results are still needed for the paper thesis
5. what regeneration path each paper asset should have
6. which claims must remain hedged because `T55` weakened the earlier simple mechanism story

## Allowed Files

Worker may modify only:

- `docs/tasks/Phase2/T47_paper_ablation_result_pack_and_material_ledger.md`
- `docs/paper_materials/paper_ablation_result_pack.md`
- `docs/review/T47_review.md`
- `docs/for_human/T47_explanation.md`

## Required Inputs

Read at minimum:

- `docs/reality_recovery/00_freeze_snapshot.md`
- `docs/reality_recovery/04_figure_and_result_ledger.md`
- `docs/reality_recovery/05_paper_claim_risk_table.md`
- `docs/paper_materials/paper_claim_evidence_ledger.md`
- `docs/paper_materials/paper_draft_skeleton.md`
- `docs/paper_materials/paper_background_related_work_draft.md`
- `docs/paper_materials/paper_method_positioning_calibration.md`
- `docs/paper_materials/paper_reviewer_risk_audit.md`
- `docs/review/T34_review.md`
- `docs/review/T35_review.md`

## Recovery Boundary

This task must stay docs-only.

It may classify or freeze:

- feature / teacher / comparator ablation needs
- ready vs partial vs missing figure assets
- canonical data / log / run-dir sources
- stable regeneration requirements for tables and plots

But it must not:

- run new experiments
- backfill missing results by changing evidence labels
- rewrite historical run dirs as if they were newly generated
- claim ablation completeness that is not yet supported

## Hedge Boundary

If this task is later activated, it must remain conditioned on the `T56` outcome:

1. treat `T36` / `T38` / `T54` / `T55` as the evidence floor, not as proof of closure
2. keep `T47` from being written as unconditional next work
3. preserve the distinction between "paper pack frozen" and "mechanism solved"
4. avoid any wording that upgrades the mechanism story beyond the `T56` claim table

## Expected Output

Create `docs/paper_materials/paper_ablation_result_pack.md` with:

1. minimal ablation/result pack scope
2. ready / partial / missing ledger
3. figure/table regeneration paths
4. whether the current paper can proceed without additional ablation evidence
5. explicit non-claims

Create `docs/review/T47_review.md` with:

1. scope confirmation
2. whether the ablation/material pack is frozen clearly
3. whether the ledger keeps missing items visible
4. recommended next task, if any

Create `docs/for_human/T47_explanation.md` with a short human-facing summary.

## Verification

Required verification is documentation-only:

1. confirm no source, config, `runs/`, or `artifacts` files were modified
2. confirm no new benchmark or training command was started
3. confirm the ledger does not inflate missing figures/tables into completed evidence
4. confirm the output is stepwise and can be expanded later without re-deriving the whole paper pack

## Captain Notes

T47 should answer a simple question:

- what exactly is still missing from the paper figure/result/material pack, and what is the minimum clean way to freeze that gap?

But the answer must stay faithful to `T56`: if the hedge wording cannot be preserved, then `T47` should remain deferred.

## Worker Output

- Created `docs/paper_materials/paper_ablation_result_pack.md` with:
  - Ready / partial / missing ledger with 11 figure entries and 5 table entries
  - Ablation-specific gap analysis (FR7 identified as largest missing item)
  - Regeneration paths for each asset with concrete data sources and estimated effort
  - Paper-readiness assessment: can proceed but only with explicit limitations
  - T56 hedge conditioning table mapping paper sections to claim-table constraints
  - 9 explicit non-claims
- Created `docs/review/T47_review.md` with scope check, hedge-conditioning check, non-claims check, and T56 boundary preservation check
- Created `docs/for_human/T47_explanation.md` with human-facing summary
- No source code, config, run-root, `.tflite`, hardware, cleanup, or governance files were modified
- No new benchmark, training, trace export, intervention, or comparator execution was run

## Verification Record

1. **Only allowed docs changed:** Confirmed. Four files modified/created, all within the allowed set.
2. **No source, config, test, runtime, training, hardware, or run-root file changed:** Confirmed. No `.py`, `.yaml`, `.yml`, or run-root files were touched.
3. **T56 hedge wording preserved:** Confirmed. All mechanism-facing statements use T56 claim table wording. C4 remains `partial`. The simple "high committed-b is harmful" framing does not appear in any output.
4. **T47 is not presented as unconditional next work:** Confirmed. The ledger explicitly states that FR7 (feature ablation) is missing and that the paper can proceed only with explicit limitation wording.
5. **No claim upgrade:** Confirmed. No claim is upgraded from `partial` to `supported` or from `missing` to `ready`. The FR7 gap is explicitly stated as requiring new execution.

