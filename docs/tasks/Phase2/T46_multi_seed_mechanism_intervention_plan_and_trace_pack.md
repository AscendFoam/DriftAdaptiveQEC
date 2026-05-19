# T46: Multi-seed mechanism/intervention plan and trace pack

## Status

- Proposed by Captain on `2026-05-19`
- Current phase: `Phase 2: Controlled Development`
- Decision state: `Go`
- Task type: docs-only planning gate

## Why This Task Exists

`T44` and the existing trace work show that `seed=20260429` is still only a single-seed diagnosis lane.

That is useful, but it is not enough for a strong mechanism claim.

This task exists to define the smallest credible next step for mechanism evidence: a multi-seed or intervention-oriented trace plan that could eventually close the gap without overbuilding the task.

## Goal

Produce a mechanism plan that answers, in writing:

1. what exact mechanism claim is currently too weak
2. what minimal additional trace or intervention evidence would strengthen it
3. which seeds, traces, and comparison fields would be needed
4. what would count as diagnostic evidence versus causal evidence
5. how to keep the scope small enough that the task remains bounded

## Allowed Files

Worker may modify only:

- `docs/tasks/Phase2/T46_multi_seed_mechanism_intervention_plan_and_trace_pack.md`
- `docs/seed_mechanism_multi_seed_plan.md`
- `docs/review/T46_review.md`
- `docs/for_human/T46_explanation.md`

## Required Inputs

Read at minimum:

- `docs/reality_recovery/00_freeze_snapshot.md`
- `docs/reality_recovery/01_claim_evidence_table.md`
- `docs/reality_recovery/04_figure_and_result_ledger.md`
- `docs/reality_recovery/05_paper_claim_risk_table.md`
- `docs/seed20260429_failure_diagnosis.md`
- `docs/seed20260429_trace_export_diagnosis.md`
- `docs/review/T36_review.md`
- `docs/review/T38_review.md`
- `docs/08_risks_and_open_questions.md`
- `docs/paper_reviewer_risk_audit.md`

## Recovery Boundary

This task must stay docs-only.

It may define:

- minimal seed selection logic
- trace-field requirements
- intervention/counterfactual ideas
- acceptance thresholds for diagnostic vs causal language
- what should remain out of scope until later

But it must not:

- run a new benchmark
- extend the existing frozen benchmark set
- train a new branch
- add a new model family
- make causal claims that are not yet supported

## Expected Output

Create `docs/seed_mechanism_multi_seed_plan.md` with:

1. current mechanism boundary
2. minimal multi-seed / intervention design
3. required trace fields and comparison rows
4. diagnostic vs causal evidence separation
5. a go / no-go recommendation for later execution
6. explicit non-claims

Create `docs/review/T46_review.md` with:

1. boundary confirmation
2. whether the mechanism gap is still single-seed only
3. whether the plan stays small and executable
4. recommended next task, if any

Create `docs/for_human/T46_explanation.md` with a short human-facing summary.

## Verification

Required verification is documentation-only:

1. confirm no source, config, `runs/`, or `artifacts` files were modified
2. confirm no new training or benchmark command was started
3. confirm the plan does not claim causal proof
4. confirm the task remains a stepwise plan, not a broad mechanism project

## Captain Notes

T46 should answer a simple question:

- what is the smallest believable next step that can move `seed=20260429` from a single-seed diagnosis toward a real mechanism story?

