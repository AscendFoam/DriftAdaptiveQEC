# T47: Paper ablation result-pack and material ledger

## Status

- Proposed by Captain on `2026-05-19`
- Current phase: `Phase 2: Controlled Development`
- Decision state: `Go`
- Task type: docs-only planning gate

## Why This Task Exists

`T44` shows that the paper is still missing a stable, paper-grade figure/result/material pack.

This task exists to freeze the minimal set of ablation/result/material items the paper actually needs, without pretending the evidence already exists.

## Goal

Produce a paper-pack ledger that answers, in writing:

1. which figures and tables are ready
2. which are partial
3. which are missing
4. which ablation results are still needed for the paper thesis
5. what regeneration path each paper asset should have

## Allowed Files

Worker may modify only:

- `docs/tasks/Phase2/T47_paper_ablation_result_pack_and_material_ledger.md`
- `docs/paper_ablation_result_pack.md`
- `docs/review/T47_review.md`
- `docs/for_human/T47_explanation.md`

## Required Inputs

Read at minimum:

- `docs/reality_recovery/00_freeze_snapshot.md`
- `docs/reality_recovery/04_figure_and_result_ledger.md`
- `docs/reality_recovery/05_paper_claim_risk_table.md`
- `docs/paper_claim_evidence_ledger.md`
- `docs/paper_draft_skeleton.md`
- `docs/paper_background_related_work_draft.md`
- `docs/paper_method_positioning_calibration.md`
- `docs/paper_reviewer_risk_audit.md`
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

## Expected Output

Create `docs/paper_ablation_result_pack.md` with:

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

