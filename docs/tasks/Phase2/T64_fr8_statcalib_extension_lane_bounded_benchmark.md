# T64: FR8 statcalib extension-lane bounded benchmark

## Status

- Proposed by Captain on `2026-05-27`
- Current phase: `Phase 2: Controlled Development`
- Decision state: `Go`
- Task type: bounded execution + audit task on the mainline experiment branch

## Why This Task Exists

`T59` integrated `statcalib` as a separate comparator lane.

`T60` then closed the cross-mode semantics and regression-hardening blocker.

`T61` failed the clean-provenance goal and was judged `BLOCK`.

`T62` repaired that exact provenance blocker on clean `main`.

`T63` then completed the pre-FR8 gate review and answered the remaining captain question: the repository no longer needs another smaller prerequisite before one bounded FR8 execution task.

That does **not** mean the repository may silently rewrite the historical frozen benchmark set.

The next smallest honest step is one bounded `FR8` extension-lane benchmark that:

1. reuses the locked four-scenario formal software benchmark boundary
2. preserves the historical frozen five-mode table
3. adds `statcalib` only as a separately labeled extension lane
4. stays fully inside mock-backed software-HIL scope

## Goal

Produce one bounded FR8 extension-lane result pack that answers:

1. how `statcalib` behaves on the locked four-scenario software benchmark boundary
2. whether that behavior remains visible when compared against the frozen five-mode set under the same paired-seed / `repeats=2` protocol
3. whether the extension lane can be reported honestly without rewriting the historical `T24` frozen ranked table
4. what boundary wording must remain explicit after the run

## Allowed Files

Worker may modify only:

- `docs/tasks/Phase2/T64_fr8_statcalib_extension_lane_bounded_benchmark.md`
- `docs/evidence_packs/statcalib_fr8/fr8_statcalib_extension_lane_benchmark.md`
- `docs/review/T64_review.md`
- `docs/for_human/T64_explanation.md`
- `docs/worker_summary/T64_worker_summary.md`
- `cnn_fpga/config/p4_multiscenario_statcalib_extension_lane.yaml`

Worker may create only:

- one task-scoped run root under `runs/p4_benchmark/T64_fr8_statcalib_extension_lane_*`

## Docs To Update

This task should update only:

1. `docs/evidence_packs/statcalib_fr8/fr8_statcalib_extension_lane_benchmark.md`
2. `docs/review/T64_review.md`
3. `docs/for_human/T64_explanation.md`
4. `docs/worker_summary/T64_worker_summary.md`
5. `cnn_fpga/config/p4_multiscenario_statcalib_extension_lane.yaml`, only if a task-scoped derived config is strictly needed
6. this task package itself, only to append Worker output and verification notes after completion

## Forbidden Scope

This task must not:

1. edit `docs/00_project_snapshot.md`, `docs/01_legacy_audit.md`, `docs/02_experiment_plan.md`, `docs/03_hil_p4_boundary_audit.md`, `docs/04_task_board.md`, `docs/05_decision_log.md`, `docs/06_repo_noise_governance.md`, `docs/07_handoff.md`, or `docs/08_risks_and_open_questions.md`
2. edit theory-only materials or files under `docs/follow-up_plan/`
3. modify any source file or test file
4. modify `cnn_fpga/config/p4_multiscenario_strong_baselines.yaml` or any other historical source-tree benchmark config
5. widen the benchmark beyond the locked four scenarios, the frozen five-mode set plus one `statcalib` extension lane, paired seeds, and `repeats=2`
6. add any extra comparator beyond `statcalib`
7. change `statcalib` estimator semantics relative to the already integrated lane used in `T59/T62`
8. change seed policy, repeat policy, CI stopping policy, metric definitions, or reporting semantics outside the bounded extension-lane visibility problem
9. touch `.tflite`, real-board, cleanup, training, intervention, paper-prose expansion, or theory-branch work
10. rewrite, resume into, or relabel the historical `T24`, `T59`, `T61`, or `T62` run roots
11. write any claim that `T64` validates `.tflite`, validates real-board behavior, or replaces the historical `T24` frozen-set evidence

## Required Inputs

Read at minimum:

- `docs/04_task_board.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/evidence_packs/statcalib_fr8/fr8_statcalib_comparator_gate_review.md`
- `docs/review/T63_review.md`
- `docs/review/T59_review.md`
- `docs/review/T60_review.md`
- `docs/review/T62_review.md`
- `docs/evidence_packs/statcalib_fr8/statcalib_comparator_lane_smoke.md`
- `docs/evidence_packs/statcalib_fr8/statcalib_lane_isolation_and_regression_hardening.md`
- `docs/evidence_packs/statcalib_fr8/statcalib_provenance_isolated_fairness_rerun.md`
- `docs/protocols/benchmark/P4_benchmark_formal_protocol.md`
- `cnn_fpga/config/p4_multiscenario_strong_baselines.yaml`
- `cnn_fpga/config/p4_multiscenario_statcalib_smoke.yaml`
- `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/summary.json`
- `runs/p4_benchmark/T62_statcalib_provenance_isolated_20260527_122943/summary.json`

## Fixed Boundary

This task is locked to the following boundary:

1. current evidence remains mock-backed software-HIL only
2. the historical `T24` frozen five-mode ranked table remains authoritative and must not be silently rewritten
3. `statcalib` enters only as a separately labeled extension lane
4. the locked four-scenario protocol remains:
   - `static_bias_theta`
   - `linear_ramp`
   - `step_sigma_theta`
   - `periodic_drift`
5. the frozen five-mode set remains:
   - `ekf`
   - `ukf`
   - `constant_residual_mu`
   - `rls_residual_b`
   - `hybrid_residual_b`
6. the only extra mode allowed is:
   - `statcalib`
7. paired seeds and `repeats=2` remain fixed
8. if execution is chunked, chunk only by repeat range under one fixed run root

## Implementation Requirements

### 1. Clean-provenance preflight

Before creating docs or launching the run:

- confirm `git branch --show-current` is exactly `main`
- confirm `git status --short` is empty
- record `git rev-parse --short HEAD`
- record a human-readable launch timestamp

If the worktree is dirty or the branch is not `main`, stop and report `BLOCKED_BY_PRECHECK`.

### 2. Preserve frozen semantics first

The frozen benchmark semantics must be preserved before the extension lane is added:

1. keep the four scenarios in the same order used by `cnn_fpga/config/p4_multiscenario_strong_baselines.yaml`
2. keep the five frozen modes in the same order used by `cnn_fpga/config/p4_multiscenario_strong_baselines.yaml`
3. append `statcalib` only after the five frozen modes so the frozen mode ordering is not shifted
4. keep the protocol anchored to the same paired-seed and `repeats=2` settings used by `T24`

Do not insert `statcalib` in the middle of the frozen mode list.

### 3. Task-scoped config rule

Preferred path:

- use the existing runner with `cnn_fpga/config/p4_multiscenario_strong_baselines.yaml` semantics unchanged

If the extension lane cannot be expressed honestly without a task-scoped config file, create:

- `cnn_fpga/config/p4_multiscenario_statcalib_extension_lane.yaml`

That derived config must:

1. preserve the frozen four scenarios
2. preserve the frozen five modes in the same order
3. append `statcalib` as a separately labeled sixth mode
4. reuse the same minimal `statcalib` parameter block already used in `cnn_fpga/config/p4_multiscenario_statcalib_smoke.yaml`
5. avoid any other semantic change

Allowed `statcalib` parameter block:

- `teacher_mode: ukf`
- `residual_scale_b: 0.50`
- `residual_clip_b: 0.08`
- `signal_threshold: 0.001`
- `postprocess.ema_alpha: 1.0`
- `postprocess.ema_apply_to: b`

### 4. Bounded execution shape

Run the locked four scenarios with the frozen five modes plus `statcalib`, keeping:

- `--paired-seeds`
- `--repeats 2`
- one fixed T64-scoped run root

Accepted execution shapes:

1. one foreground invocation across the full matrix, or
2. repeat-range chunking only:
   - `--repeat-start 0 --repeat-stop 1`
   - `--repeat-start 1 --repeat-stop 2`
   - optional final `--resume-only`

Do not split by scenario or by mode, because that changes local runner indexing and seed semantics.

### 5. Required result reporting

The output report must keep the frozen subset and the extension lane separated.

At minimum report:

1. exact preflight and post-run `branch` / `HEAD`
2. exact config path used
3. exact run root
4. whether execution was one-shot or repeat-chunked
5. `summary.json["git_commit"]`
6. `missing_runs`, coverage, and raw-row counts
7. a frozen five-mode subset table for the T64 run
8. a separate `statcalib` extension-lane comparison against the frozen winner and runner-up per scenario
9. `final_ler_mean` and `final_ler_std`
10. `statcalib_status`, `statcalib_reason`, and generated-window counts
11. any difference between the T64 frozen subset and the historical T24 frozen-set summary

If the frozen subset differs from T24, describe the difference honestly instead of hiding it.

### 6. Boundary wording that must remain explicit

The final docs must say explicitly:

- this remains mock-backed software-HIL evidence
- this is a bounded FR8 extension-lane task, not a rewrite of `T24`
- this does not validate `.tflite` runtime
- this does not validate real-board behavior
- this does not upgrade the lane into paper-grade expanded benchmark evidence by itself

## Expected Output Artifacts

Create `docs/evidence_packs/statcalib_fr8/fr8_statcalib_extension_lane_benchmark.md` with:

1. exact preflight result (`branch`, `git status`, launch `HEAD`)
2. exact config path used and whether a derived config was needed
3. exact benchmark command(s)
4. exact run root
5. exact post-run `branch`, `HEAD`, and `summary.json git_commit`
6. one table for the frozen five-mode subset
7. one separate table for the `statcalib` extension lane
8. per-scenario winner / runner-up / extension-lane gap summary
9. coverage, missing-run, and raw-row evidence
10. explicit boundary statement and residual risk

Create `docs/review/T64_review.md` with:

1. scope and boundary check
2. confirmation that the run started from a clean committed `main` worktree
3. confirmation that the frozen five-mode ordering was preserved
4. confirmation that `statcalib` stayed a separately labeled extension lane
5. confirmation that no source/test file was modified
6. confirmation that no historical run root was rewritten
7. whether the result pack is appropriately bounded and honest

Create `docs/for_human/T64_explanation.md` with a short human-facing summary.

Create `docs/worker_summary/T64_worker_summary.md` with a concise worker-facing summary of changes, verification, and residual risk.

## Verification

Required verification:

1. `git branch --show-current` is `main` before the run starts
2. `git status --short` is empty before the run starts
3. the run uses the locked four scenarios and the frozen five modes plus `statcalib`
4. `--paired-seeds` and `--repeats 2` are preserved
5. if chunked, chunking happens only by repeat range with one fixed run root
6. launch / finish / `summary.json` commit identity all match
7. the frozen five-mode ordering is preserved with `statcalib` appended last
8. no source/test file was modified
9. no historical run root was resumed, regenerated, or rewritten
10. every output doc keeps the mock-backed software-HIL boundary explicit

Suggested one-shot command shape:

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config cnn_fpga/config/p4_multiscenario_statcalib_extension_lane.yaml --scenario static_bias_theta --scenario linear_ramp --scenario step_sigma_theta --scenario periodic_drift --mode ekf --mode ukf --mode constant_residual_mu --mode rls_residual_b --mode hybrid_residual_b --mode statcalib --paired-seeds --repeats 2 --run-dir runs/p4_benchmark/T64_fr8_statcalib_extension_lane_<timestamp>
```

Accepted chunked command shapes if needed:

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config cnn_fpga/config/p4_multiscenario_statcalib_extension_lane.yaml --scenario static_bias_theta --scenario linear_ramp --scenario step_sigma_theta --scenario periodic_drift --mode ekf --mode ukf --mode constant_residual_mu --mode rls_residual_b --mode hybrid_residual_b --mode statcalib --paired-seeds --repeats 2 --run-dir runs/p4_benchmark/T64_fr8_statcalib_extension_lane_<timestamp> --repeat-start 0 --repeat-stop 1
```

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config cnn_fpga/config/p4_multiscenario_statcalib_extension_lane.yaml --scenario static_bias_theta --scenario linear_ramp --scenario step_sigma_theta --scenario periodic_drift --mode ekf --mode ukf --mode constant_residual_mu --mode rls_residual_b --mode hybrid_residual_b --mode statcalib --paired-seeds --repeats 2 --run-dir runs/p4_benchmark/T64_fr8_statcalib_extension_lane_<timestamp> --repeat-start 1 --repeat-stop 2
```

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config cnn_fpga/config/p4_multiscenario_statcalib_extension_lane.yaml --scenario static_bias_theta --scenario linear_ramp --scenario step_sigma_theta --scenario periodic_drift --mode ekf --mode ukf --mode constant_residual_mu --mode rls_residual_b --mode hybrid_residual_b --mode statcalib --paired-seeds --repeats 2 --run-dir runs/p4_benchmark/T64_fr8_statcalib_extension_lane_<timestamp> --resume-only
```

If the final command differs, document exactly why and keep it no broader than the same boundary.

## Review No-Go Triggers

Review should be treated as `BLOCK` if any of the following happen:

1. the run starts from a dirty worktree or a non-`main` branch
2. the worker modifies any source file or test file
3. the worker modifies `cnn_fpga/config/p4_multiscenario_strong_baselines.yaml`
4. the worker changes the frozen five-mode ordering or inserts `statcalib` into the middle of that set
5. the worker widens scenarios, repeats, comparator scope, or deployment scope
6. the worker rewrites, resumes into, or relabels historical `T24`, `T59`, `T61`, or `T62` outputs
7. launch / finish / `summary.json` commit identity do not all match
8. the report silently rewrites the historical `T24` frozen-set evidence as if it had always included `statcalib`
9. the report claims `.tflite`, real-board, or paper-grade expanded benchmark validation

## Captain Notes

`T63` is complete and accepted as `PASS`.

The next step is now execution, not another gate. But it remains bounded execution only.

This task must stay on `main`-branch experiment work and must remain isolated from the user's separate theory branch.

## Worker Output

### Status

- Completed within allowed scope
- Task end-state not marked here; waiting for external review as requested

### What Changed

1. Created the task-scoped derived config:
   - `cnn_fpga/config/p4_multiscenario_statcalib_extension_lane.yaml`
2. Executed the bounded extension-lane benchmark in:
   - `runs/p4_benchmark/T64_fr8_statcalib_extension_lane_20260527_221658`
3. Wrote the result pack:
   - `docs/evidence_packs/statcalib_fr8/fr8_statcalib_extension_lane_benchmark.md`
   - `docs/review/T64_review.md`
   - `docs/for_human/T64_explanation.md`
   - `docs/worker_summary/T64_worker_summary.md`

### Verification Notes

1. Preflight passed on clean `main`:
   - launch timestamp: `2026-05-27 22:11:07 +08:00`
   - launch `HEAD`: `1e59f24`
2. The run preserved the locked benchmark boundary:
   - scenarios: `static_bias_theta`, `linear_ramp`, `step_sigma_theta`, `periodic_drift`
   - modes: `ekf`, `ukf`, `constant_residual_mu`, `rls_residual_b`, `hybrid_residual_b`, `statcalib`
   - `statcalib` appended last
   - `--paired-seeds`
   - `--repeats 2`
3. Provenance passed:
   - finish branch: `main`
   - finish `HEAD`: `1e59f24`
   - `summary.json git_commit`: `1e59f24`
4. Output integrity passed:
   - `comparison_rows_count=24`
   - `raw_rows_count=48`
   - `missing_runs_count=0`
   - all rows have `coverage=1.0`
   - all rows have `completed_repeats=2`
   - `progress.jsonl` has no duplicate `running` key
5. Frozen-table preservation passed:
   - T64 frozen subset matches historical `T24` exactly across all 20 frozen comparison rows
6. Extension-lane result:
   - `statcalib` won all four scenarios as a separate lane
   - `statcalib_status=generated`
   - `statcalib_reason=statcalib_params_emitted`

### Remaining Risk

- Evidence remains mock-backed software-HIL only.
- `T64` does not validate `.tflite` or real-board behavior.
- `T64` does not rewrite or replace the historical `T24` frozen benchmark evidence.
