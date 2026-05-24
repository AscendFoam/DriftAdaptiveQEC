# T56: Post-I1 mechanism claim reframing gate

## Status

- Proposed by Captain on `2026-05-24`
- Current phase: `Phase 2: Controlled Development`
- Decision state: `Go`
- Task type: docs-only evidence interpretation and next-lane gate task

## Why This Task Exists

`T55` has now answered the first bounded intervention question:

- the pure I1 lower-clip intervention does not cleanly help across the locked 6-seed pack
- the effect is mixed: harmful on most seeds, helpful on a minority
- the earlier simple narrative "high committed-`b` is the problem" is not supported as a general explanation

This means the project should not jump directly to `T47` as if the mechanism story were already closed. Before any paper-material packaging or any second intervention design, the repository needs one bounded gate task that freezes what claims remain valid, which claims need reframing, and what next lane is still justified.

## Goal

Produce a bounded post-`T55` mechanism-claim gate that answers:

1. which mechanism statements from `T36` / `T38` / `T54` remain valid after `T55`
2. which statements must be weakened, retired, or reframed
3. whether a second intervention lane is justified at all, and if so under what stricter question framing
4. whether `T47` can proceed, and if yes, under what explicit mechanism-hedge boundary

## Allowed Files

Worker may modify only:

- `docs/tasks/Phase2/T56_post_i1_mechanism_claim_reframing_gate.md`
- `docs/post_t55_mechanism_claim_reframing_gate.md`
- `docs/paper_claim_evidence_ledger.md`
- `docs/review/T56_review.md`
- `docs/for_human/T56_explanation.md`

## Docs To Update

This task should update only:

1. `docs/post_t55_mechanism_claim_reframing_gate.md`
2. `docs/paper_claim_evidence_ledger.md`
3. `docs/review/T56_review.md`
4. `docs/for_human/T56_explanation.md`
5. this task package itself, only to append Worker output and verification notes after completion

## Forbidden Scope

This task must not:

1. run any new benchmark, training, trace export, `.tflite`, hardware, cleanup, or comparator execution
2. edit any source code, benchmark code, config, test, runtime, hardware, training, or run-root file
3. edit `docs/00_project_snapshot.md`, `docs/01_legacy_audit.md`, `docs/02_experiment_plan.md`, `docs/03_hil_p4_boundary_audit.md`, `docs/04_task_board.md`, `docs/05_decision_log.md`, `docs/06_repo_noise_governance.md`, `docs/07_handoff.md`, or `docs/08_risks_and_open_questions.md`
4. rewrite historical task reports to make earlier claims look stronger than they were at that time
5. silently reopen `T47`, benchmark expansion, second intervention, `.tflite`, or real-board scope as if already approved
6. upgrade any result into causal proof, mechanism closure, paper-grade benchmark evidence, `.tflite` validation, or real-board validation

## Required Inputs

Read at minimum:

- `docs/04_task_board.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/paper_claim_evidence_ledger.md`
- `docs/seed_mechanism_multi_seed_plan.md`
- `docs/seed20260429_failure_diagnosis.md`
- `docs/seed20260429_trace_export_diagnosis.md`
- `docs/multi_seed_trace_generalization_probe.md`
- `docs/multi_seed_i1_intervention_probe.md`
- `docs/review/T36_review.md`
- `docs/review/T38_review.md`
- `docs/review/T46_review.md`
- `docs/review/T54_review.md`
- `docs/review/T55_review.md`

## Required Output Structure

Create `docs/post_t55_mechanism_claim_reframing_gate.md` with:

1. a short status recap from `T36` through `T55`
2. a claim table with at least these columns:
   - claim or hypothesis
   - strongest supporting task(s)
   - strongest contradicting task(s)
   - current status (`retain`, `weaken`, `retire`, `reframe`, `still-open`)
   - exact wording boundary
3. a section that explicitly states what `T55` changed in the mechanism story
4. a section that answers whether a second intervention lane is:
   - `no-go for now`
   - `deferred pending better question`
   - `conditionally justified`
5. a section that answers whether `T47` can proceed, and under what mechanism-hedge wording
6. explicit non-claims

Update `docs/paper_claim_evidence_ledger.md` only where `T55` changes the status or wording boundary of mechanism-related rows.

Create `docs/review/T56_review.md` with:

1. scope and boundary confirmation
2. whether the claim table reflects `T36/T38/T54/T55` honestly
3. whether any claim was silently upgraded beyond evidence
4. whether the `T47` recommendation is properly conditioned
5. whether any second-intervention recommendation is bounded and justified

Create `docs/for_human/T56_explanation.md` with a short human-facing summary.

## Verification

Required verification:

1. confirm only the allowed docs changed
2. confirm no source, config, test, runtime, training, hardware, or run-root file changed
3. confirm the gate does not upgrade mechanism evidence into causal proof or closure
4. confirm the `T47` recommendation, if any, is conditional and hedge-aware
5. confirm any follow-up intervention recommendation is explicitly bounded and not auto-approved

## Review No-Go Triggers

Review should be treated as `BLOCK` if any of the following happen:

1. the worker runs any new execution or edits any run-root file
2. the worker upgrades the mechanism story to "solved", "supported", or causal proof without new evidence
3. the worker recommends `T47` as unconditional next task
4. the worker proposes a second intervention as automatic next execution without a tighter question framing
5. the worker edits governance docs or `docs/02_experiment_plan.md`

## Captain Notes

This task is intentionally between `T55` and any later paper-material or second-intervention decision.

The point is not to write prettier prose. The point is to make the repository honest about what `T55` actually proved and what it did not.

## Worker Output

- Created `docs/post_t55_mechanism_claim_reframing_gate.md` with:
  - Status recap from T36 through T55
  - 10-row claim table (3 retain, 4 weaken, 2 retire, 1 reframe, 2 still-open) with supporting/contradicting tasks and exact wording boundaries
  - Section on what T55 changed in the mechanism story
  - Second intervention lane verdict: `deferred pending better question`
  - T47 recommendation: can proceed only under conditioned mechanism-hedge wording
  - Explicit non-claims (9 items)
- Updated `docs/paper_claim_evidence_ledger.md`:
  - C4 claim wording updated to reflect T54/T55 multi-seed evidence and I1 mixed results
  - C4 evidence paths expanded to include T54/T55 probe documents and reviews
  - F1 figure boundary updated to note instability mostly helps
  - Wording guardrail #4 expanded to prohibit "high committed-b is harmful" and "instability = defect"
- Created `docs/review/T56_review.md` with scope check, claim-table honesty check, no-upgrade check, T47 conditioning check, and second-intervention check
- Created `docs/for_human/T56_explanation.md` with human-facing summary
- No source code, config, test, runtime, hardware, training, run-root, or governance files were modified
- No new benchmark, training, trace export, `.tflite`, hardware, cleanup, or comparator execution was run

## Verification Record

1. **Only allowed docs changed:** Confirmed. Five files modified/created, all within the allowed set.
2. **No source, config, test, runtime, training, hardware, or run-root file changed:** Confirmed. No `.py`, `.yaml`, `.yml`, or run-root files were touched.
3. **Gate does not upgrade mechanism evidence into causal proof or closure:** Confirmed. All new text uses diagnostic hedging language. "Causal proof," "mechanism proven," and "root cause identified" appear only in non-claim or negative context.
4. **T47 recommendation is conditional and hedge-aware:** Confirmed. Gate Section 5 states T47 "can proceed, but only under explicit mechanism-hedge wording" with 5 specific hedge boundaries. T47 is explicitly not recommended as unconditional next work.
5. **Any follow-up intervention recommendation is explicitly bounded and not auto-approved:** Confirmed. Gate Section 4 verdict is `deferred pending better question` with explicit prerequisites before any I2/I3 execution would be justified.
