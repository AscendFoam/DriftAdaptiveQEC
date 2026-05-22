# T54: Phase A multi-seed trace-only generalization probe

## Status

- Proposed by Captain on `2026-05-22`
- Current phase: `Phase 2: Controlled Development`
- Decision state: `Go`
- Task type: bounded execution task on an already validated path

## Why This Task Exists

`T46` passed and froze the mechanism-evidence plan. That plan makes one point clear:

- current mechanism evidence is still only single-seed diagnostic evidence anchored on `seed=20260429`
- before any intervention or paper-material freezing, the project needs a bounded check of whether the committed-`b` instability pattern generalizes beyond one seed

This task is therefore the smallest execution follow-up after `T46`: a Phase A trace-only probe that stays inside the existing `T38` path and does not yet claim any causal result.

## Goal

Run the smallest believable multi-seed trace probe that can answer:

1. whether the committed combined-`b` instability pattern appears outside `seed=20260429`
2. whether the currently observed pattern is isolated, repeated, or mixed across a very small seed pack
3. whether a later intervention task is justified at all

This task must remain trace-only. It must not test interventions yet.

## Allowed Files

Worker may modify only:

- `docs/tasks/Phase2/T54_multi_seed_trace_only_generalization_probe.md`
- `docs/multi_seed_trace_generalization_probe.md`
- `docs/review/T54_review.md`
- `docs/for_human/T54_explanation.md`

Worker may create:

- one T54-scoped run root under `runs/`, for example `runs/T54_multi_seed_trace_phase_a_*`
- derived CSV/JSON summaries only inside that T54-scoped run root

## Docs To Update

This task should update only:

1. `docs/multi_seed_trace_generalization_probe.md`
2. `docs/review/T54_review.md`
3. `docs/for_human/T54_explanation.md`
4. this task package itself, only to append Worker output and verification notes after completion

## Forbidden Scope

This task must not:

1. edit any source code, benchmark code, config, test, runtime, hardware, or training file
2. edit `docs/00_project_snapshot.md`, `docs/01_legacy_audit.md`, `docs/03_hil_p4_boundary_audit.md`, `docs/04_task_board.md`, `docs/05_decision_log.md`, `docs/06_repo_noise_governance.md`, `docs/07_handoff.md`, `docs/08_risks_and_open_questions.md`, or `docs/02_experiment_plan.md`
3. modify or overwrite any historical `runs/` or `artifacts/` path outside the single T54-scoped run root
4. add new baselines, new scenario families, new drift families, new training runs, `.tflite` runtime work, hardware work, cleanup, or benchmark-scope expansion
5. run any intervention variant in this task
6. treat trace-only evidence as causal proof or as paper-grade benchmark evidence
7. exceed 6 total seeds without a new Captain-approved task package

## Required Inputs

Read at minimum:

- `docs/04_task_board.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/seed_mechanism_multi_seed_plan.md`
- `docs/review/T46_review.md`
- `docs/seed20260429_failure_diagnosis.md`
- `docs/seed20260429_trace_export_diagnosis.md`
- `docs/review/T36_review.md`
- `docs/review/T38_review.md`
- `docs/paper_claim_evidence_ledger.md`
- `cnn_fpga/benchmark/run_p4_teacher_representation_paired.py`
- `cnn_fpga/benchmark/analyze_seed20260429_trace.py`
- `cnn_fpga/config/experiment_runtime_b_residual.yaml`
- `cnn_fpga/config/experiment_runtime_b_residual_norm_gated_teacher_v5.yaml`

## Fixed Execution Boundary

This task is locked to the following execution boundary:

1. modes:
   - `Full`
   - `Gated v5`
2. scenarios:
   - `static_bias_theta`
   - `linear_ramp`
   - `step_sigma_theta`
   - `periodic_drift`
3. repeats:
   - `2`
4. total seed ceiling:
   - `6`
5. evidence type:
   - trace-only diagnostic evidence

## Seed Policy

Use the `T46` seed pack exactly:

1. existing seeds:
   - `20260427`
   - `20260428`
   - `20260429`
2. new seeds:
   - `20260425`
   - `20260430`
   - `20260510`

Execution rules:

1. Reuse existing `20260429` T38 trace outputs; do not rerun `20260429` unless the existing T38 trace export is proven unusable.
2. Preflight existing artifacts for `20260427` and `20260428` first. If they already contain the required fields and can be exported by the existing analysis path, reuse them instead of rerunning.
3. Only rerun `20260427` and/or `20260428` if the preflight proves the required trace fields are unavailable from existing artifacts.
4. Always keep any new rerun inside the single T54-scoped run root.

## Required Output Artifacts

Inside the T54-scoped run root, produce at minimum:

1. per-seed trace export outputs in the same schema family as T38
2. `cross_seed_comparison.csv`
3. `delta_b_amplitude_by_seed.csv`
4. `mechanism_summary.csv`
5. one manifest or note that records which seeds were reused from existing artifacts and which required rerun

## Expected Report Output

Create `docs/multi_seed_trace_generalization_probe.md` with:

1. exact command list and run-root structure
2. artifact-reuse versus rerun matrix by seed
3. field-availability summary
4. per-seed and cross-seed comparison summary
5. mechanism generalization verdict:
   - isolated to `20260429`
   - partially repeated
   - broadly repeated
6. recommendation on whether a later intervention task is justified
7. explicit non-claims

The report must contain at least these concrete tables:

1. seed execution matrix
2. field-availability table
3. cross-seed outcome summary table
4. mechanism classification table

Create `docs/review/T54_review.md` with:

1. scope and boundary confirmation
2. whether the worker stayed inside the frozen scenarios / Full vs Gated v5 boundary
3. whether reuse versus rerun decisions were honest and justified
4. whether the final wording stays diagnostic rather than causal
5. recommended next bounded task, if any

Create `docs/for_human/T54_explanation.md` with a short human-facing summary.

## Verification

Required verification:

1. confirm only allowed docs changed, plus the single T54-scoped run root
2. confirm no source, config, test, runtime, hardware, or training file was modified
3. confirm no historical `runs/` or `artifacts/` path was overwritten
4. confirm all new execution stayed inside the fixed four scenarios and two modes
5. confirm the report does not claim causal proof, intervention success, `.tflite` runtime validation, or real-board validation

## Review No-Go Triggers

Review should be treated as `BLOCK` if any of the following happen:

1. the worker adds new baselines, new scenarios, or new seeds outside the locked pack
2. the worker modifies code or config to widen semantics instead of reusing the T38 path
3. the worker runs intervention variants inside this trace-only task
4. the worker rewrites or overwrites historical run paths
5. the report upgrades trace-only evidence into causal proof or paper-grade benchmark evidence

## Captain Notes

This task is intentionally before the old `T47` paper-material lane.

If `T54` does not show a repeated pattern beyond `20260429`, then the project should not rush into intervention or paper-ablation packaging as if the mechanism were already stable. In that case, C4 remains `partial`, and the paper story must keep diagnostic hedging.

## Worker Output

- Pending.

## Verification Record

- Pending.
