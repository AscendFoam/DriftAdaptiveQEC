# T36: seed=20260429 failure-mechanism diagnosis, bounded no-new-branch scope

## Status

- Created by Captain on `2026-05-13`
- Current phase: `Phase 2: Controlled Development`
- Decision state: `Go`
- Task type: bounded mechanism diagnosis / existing-artifact analysis

## Why This Task Exists

`docs/02_experiment_plan.md` lists `seed=20260429` failure-mechanism diagnosis as the first-priority mechanism task: Gated v5 is strong on most reviewed seeds, but its advantage shrinks or reverses around `seed=20260429`.

T27/T28/T29 repaired diagnostics observability and report formatting, and T30 completed the statcalib interface contract. The project can now investigate this seed-specific behavior without inheriting known report/diagnostic ambiguity.

## Goal

Use existing artifacts to explain why `Gated v5` is less stable on `seed=20260429`.

This task should produce a bounded, evidence-labeled diagnosis. It is not a new benchmark, not a new model branch, and not a paper-claim task.

## Allowed Files

Worker may modify:

- `docs/tasks/Phase2/T36_seed20260429_failure_mechanism_diagnosis.md`
- `docs/seed20260429_failure_diagnosis.md`
- `docs/review/T36_review.md`
- `docs/for_human/T36_explanation.md`
- `cnn_fpga/benchmark/analyze_seed20260429_failure.py`

If the analysis can be done entirely in docs, Worker does not need to add the Python script. If adding the script, it must only read existing artifacts and print/write compact diagnostic summaries.

## Required Inputs

Read at minimum:

- `docs/02_experiment_plan.md`
- `docs/02_experiment_plan_simplified.md`
- `docs/04_task_board.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/CNN_FPGA_GKP_阶段结论.md`
- `docs/legacy_context/2026-05-06_CNN_FPGA_GKP_legacy_handoff.md`
- `runs/teachrepr_v5_chunked_pair/paired_20260427_220702/summary.csv`
- `runs/teachrepr_v5_chunked_pair/paired_20260427_220702/summary.json`
- `runs/teachrepr_v5_chunked/p4_benchmark/trp60429_resume/summary.json`
- `runs/teachrepr/p4_benchmark/trp60429_20260427_142013_2a59bc_24060/comparison.csv`

Worker may read other existing `runs/teachrepr*` CSV/JSON files only to compare `20260429` against `20260427` and `20260428`.

## Forbidden Scope

Do not:

- modify `docs/02_experiment_plan.md`
- run any benchmark, training, `.tflite`, hardware, or cleanup command
- create or rewrite `runs/` or `artifacts/`
- modify model code, configs, benchmark runner semantics, formal protocol, baseline/scenario set, seed/repeat policy, or result boundary
- add a new teacher-representation branch
- add statcalib, soft-information comparator, new drift family, or CI-driven stopping
- claim the diagnosis is causal if the existing artifacts only support a hypothesis

## Expected Output

Create `docs/seed20260429_failure_diagnosis.md` with:

1. Evidence inventory: exact artifact paths used.
2. `seed=20260429` Full vs Gated v5 summary by scenario.
3. Cross-seed comparison against `20260427` and `20260428` if available from existing summaries.
4. Mechanism matrix with evidence labels:
   - sign offset
   - magnitude overshoot
   - response lag
   - teacher instability
   - gated branch too conservative
5. Clear conclusion split into:
   - supported observations
   - plausible hypotheses
   - not answerable from current artifacts
6. Recommended next bounded task, but do not execute it.

If adding `cnn_fpga/benchmark/analyze_seed20260429_failure.py`, keep it small and deterministic. It must not import project runtime paths that execute simulations.

## Verification

Required verification:

1. If script is added, run it against existing artifacts and record the command/output path.
2. If no script is added, document the manual extraction method and exact source artifact lines/fields used.
3. Confirm no benchmark run directory was created.
4. Confirm no files outside the allowed set were changed.
5. Confirm the report does not claim new benchmark evidence.

## Docs To Update

- `docs/tasks/Phase2/T36_seed20260429_failure_mechanism_diagnosis.md`
- `docs/seed20260429_failure_diagnosis.md`
- `docs/review/T36_review.md`
- `docs/for_human/T36_explanation.md`

## Reviewer Type

Adversarial review.

Focus areas:

- no hidden benchmark rerun
- no new branch or config mutation
- artifact paths are real and sufficient
- causal claims are not stronger than evidence
- diagnosis does not rewrite formal benchmark or statcalib boundaries

## Captain Notes

If existing artifacts do not contain per-window or per-commit time series, Worker must say so explicitly and downgrade the conclusion to scenario/summary-level diagnosis. Do not generate new evidence by running the benchmark.
