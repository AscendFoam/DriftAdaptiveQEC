# T62: Statcalib provenance-isolated fairness rerun

## Status

- Proposed by Captain on `2026-05-27`
- Current phase: `Phase 2: Controlled Development`
- Decision state: `Go`
- Task type: bounded execution + audit task on the mainline experiment branch

## Why This Task Exists

`T61` preserved the bounded fairness signal, but it did **not** close the blocker it was created to repair.

What `T61` did show:

1. `statcalib` still won both locked scenarios
2. `statcalib_status=generated` remained stable
3. the strong result did not collapse

What `T61` failed to show:

1. a single defensible clean-provenance artifact anchor
2. uninterrupted mainline execution without branch/worktree movement
3. a summary artifact whose `git_commit` matches the clean launch commit

So `T61_review.md` is correctly judged as `BLOCK`.

This follow-up is the **single blocking-only automatic retry** for the same underlying issue. It must repair provenance isolation only. It must not expand comparator scope, rewrite benchmark semantics, or jump ahead to `FR8`.

## Goal

Produce the smallest believable provenance repair by answering:

1. can the exact `T61` matrix be rerun from a clean committed `main` worktree
2. can that rerun finish without any branch/worktree movement or same-run resume
3. does the finished artifact preserve one single commit identity from launch through summary generation
4. after that rerun, is the next honest step an `FR8` gate discussion, or does the provenance blocker remain open

## Allowed Files

Worker may modify only:

- `docs/tasks/Phase2/T62_statcalib_provenance_isolated_fairness_rerun.md`
- `docs/statcalib_provenance_isolated_fairness_rerun.md`
- `docs/review/T62_review.md`
- `docs/for_human/T62_explanation.md`
- `docs/worker_summary/T62_worker_summary.md`

Worker may create only:

- one task-scoped run root under `runs/p4_benchmark/T62_statcalib_provenance_isolated_*`

## Docs To Update

This task should update only:

1. `docs/statcalib_provenance_isolated_fairness_rerun.md`
2. `docs/review/T62_review.md`
3. `docs/for_human/T62_explanation.md`
4. `docs/worker_summary/T62_worker_summary.md`
5. this task package itself, only to append Worker output and verification notes after completion

## Forbidden Scope

This task must not:

1. edit `docs/00_project_snapshot.md`, `docs/01_legacy_audit.md`, `docs/02_experiment_plan.md`, `docs/03_hil_p4_boundary_audit.md`, `docs/04_task_board.md`, `docs/05_decision_log.md`, `docs/06_repo_noise_governance.md`, `docs/07_handoff.md`, or `docs/08_risks_and_open_questions.md`
2. edit theory-only materials or files under `docs/follow-up_plan/`
3. modify any source file, test file, or source-tree config file under `cnn_fpga/config/`
4. widen the matrix beyond `static_bias_theta` / `linear_ramp`, `ukf` / `hybrid_residual_b` / `statcalib`, `--paired-seeds`, and `--repeats 2`
5. reuse, resume into, or rewrite the historical `T59` or `T61` run roots
6. create more than one T62-scoped run root
7. launch a second invocation against the same T62 run root after interruption, timeout, or partial progress
8. perform or trigger any `git checkout`, `git commit`, `git merge`, `git pull`, `git rebase`, or branch switch between preflight and finished summary generation
9. run `.tflite`, real-board, cleanup, training, intervention, benchmark-expansion, or theory-branch work
10. write any `FR8` formal result-table claim

## Required Inputs

Read at minimum:

- `docs/04_task_board.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/review/T59_review.md`
- `docs/review/T60_review.md`
- `docs/review/T61_review.md`
- `docs/statcalib_comparator_lane_smoke.md`
- `docs/statcalib_lane_isolation_and_regression_hardening.md`
- `docs/statcalib_fairness_sanity.md`
- `docs/worker_summary/T59_worker_summary.md`
- `docs/worker_summary/T60_worker_summary.md`
- `docs/worker_summary/T61_worker_summary.md`
- `cnn_fpga/config/p4_multiscenario_statcalib_smoke.yaml`
- `runs/p4_benchmark/t59statc_20260526_211532_3a3d00_23740/summary.json`
- `runs/p4_benchmark/T61_statcalib_fairness_sanity_20260527_015239/summary.json`

## Fixed Boundary

This task is locked to the following boundary:

1. `T61` is blocked for provenance reasons only
2. this task is execution/audit only
3. exactly one new T62-scoped run root is allowed
4. the run must reuse the existing T59/T61 smoke config and mode/scenario set
5. the command must run in one uninterrupted invocation
6. this task is still **not** `FR8`

## Implementation Requirements

### 1. Preflight provenance check

Before creating docs or launching the rerun:

- confirm `git branch --show-current` is exactly `main`
- confirm `git status --short` is empty
- record `git rev-parse --short HEAD`
- record a human-readable launch timestamp
- if the worktree is dirty or the branch is not `main`, stop and report `BLOCKED_BY_PRECHECK`

This task exists to repair provenance weakness, so it must start from a clean committed `main` state.

### 2. Run one bounded matrix in one uninterrupted invocation

Reuse the existing config:

- `cnn_fpga/config/p4_multiscenario_statcalib_smoke.yaml`

Run only:

- scenarios: `static_bias_theta`, `linear_ramp`
- modes: `ukf`, `hybrid_residual_b`, `statcalib`
- `--paired-seeds`
- `--repeats 2`

Use a fixed T62-scoped run directory under:

- `runs/p4_benchmark/T62_statcalib_provenance_isolated_*`

Do not modify the source-tree config to do this; use CLI filtering/override only.

Important execution rule:

- use **one** foreground invocation only
- if the host wrapper times out or the run is interrupted before `summary.json` is complete, stop and report `BLOCKED_BY_TIMEOUT_OR_INTERRUPT`
- do **not** resume the same run root
- do **not** start a second T62 run root inside the same task

### 3. Post-run provenance validation

After the run finishes:

- record post-run `git branch --show-current`
- record post-run `git rev-parse --short HEAD`
- confirm both match the launch values
- confirm `summary.json["git_commit"]` matches the same value
- inspect `progress.jsonl` and confirm there is no duplicate `running` entry for the same `(scenario, mode, repeat)` key

If any of the above checks fail, this task remains blocked.

### 4. Compare T62 against T59 and T61

The output report must compare:

- per-scenario ranking
- `final_ler_mean`
- runner-up gap
- `statcalib_status` / `statcalib_reason`
- `statcalib_generated_windows_mean`
- whether the strong result persists, weakens, or collapses
- whether provenance is now actually closed, not only improved

This comparison is still a sanity/provenance audit, not a formal comparator claim.

### 5. Keep evidence boundary explicit

The final docs must say explicitly:

- this remains mock-backed software-HIL evidence
- this is not `FR8`
- this does not validate `.tflite` runtime
- this does not validate real-board behavior
- if provenance is still not clean after T62, the automatic retry budget for this blocker is exhausted and the issue must return to the user for arbitration

## Expected Output Artifacts

Create `docs/statcalib_provenance_isolated_fairness_rerun.md` with:

1. exact preflight result (`branch`, `git status`, launch `HEAD`)
2. exact rerun command
3. exact run root
4. exact post-run `branch` and `HEAD`
5. exact `git_commit` anchoring seen in the new run
6. per-scenario comparison table for `ukf` / `hybrid_residual_b` / `statcalib`
7. direct comparison against both the T59 smoke and the blocked T61 rerun
8. explicit statement of whether provenance is now closed
9. explicit statement of what still remains before any later `FR8` task

Create `docs/review/T62_review.md` with:

1. scope and boundary check
2. confirmation that the rerun started from a clean committed `main` worktree
3. confirmation that only one T62-scoped run root was created
4. confirmation that the run was executed in one uninterrupted invocation with no resume
5. confirmation that launch / finish / summary commit identity all match
6. whether the next step should be an `FR8` gate discussion or a user-arbitrated stop

Create `docs/for_human/T62_explanation.md` with a short human-facing summary.

Create `docs/worker_summary/T62_worker_summary.md` with a concise worker-facing summary of changes, verification, and residual risk.

## Verification

Required verification:

1. `git branch --show-current` is `main` before the run starts
2. `git status --short` is empty before the run starts
3. run the bounded benchmark command with:
   - existing T59 config
   - `--paired-seeds`
   - `--repeats 2`
   - fixed T62-scoped `--run-dir`
4. confirm the new `summary.json` records the same clean `git_commit` seen at launch and finish
5. confirm post-run branch and `HEAD` did not drift
6. confirm `progress.jsonl` contains no duplicate `running` entry for the same repeat key
7. confirm no source, test, config, governance, or theory-only files were modified
8. confirm no historical `T59` or blocked `T61` output was rewritten

Suggested command shape:

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config cnn_fpga/config/p4_multiscenario_statcalib_smoke.yaml --scenario static_bias_theta --scenario linear_ramp --mode ukf --mode hybrid_residual_b --mode statcalib --paired-seeds --repeats 2 --run-dir runs/p4_benchmark/T62_statcalib_provenance_isolated_<timestamp>
```

If the final command differs, document exactly why.

## Review No-Go Triggers

Review should be treated as `BLOCK` if any of the following happen:

1. the rerun starts from a dirty worktree or a non-`main` branch
2. the worker modifies source code, tests, or source-tree config
3. the worker widens the matrix beyond the locked two scenarios / three modes / paired-seeds / repeats=2 boundary
4. the worker reuses or resumes into the historical `T59` or blocked `T61` run roots
5. the worker launches more than one T62 run root
6. the worker resumes or re-invokes the same T62 run root after interruption
7. launch / finish / summary commit identity do not all match
8. the worker writes any `FR8` or formal comparator-ranking claim

## Captain Notes

`T61` is blocked, not complete.

This task is the single allowed automatic retry for the same provenance blocker. If `T62` still returns `BLOCK`, Captain should stop automatic progression and return the issue to the user instead of opening another auto-retry task.

## Worker Output

### What changed

1. Executed the bounded T62 sanity matrix in the single allowed run root:
   - `runs/p4_benchmark/T62_statcalib_provenance_isolated_20260527_122943`
2. Created:
   - `docs/statcalib_provenance_isolated_fairness_rerun.md`
   - `docs/review/T62_review.md`
   - `docs/for_human/T62_explanation.md`
   - `docs/worker_summary/T62_worker_summary.md`
3. Appended the worker output into this task package.

### Verification record

1. Preflight passed:
   - launch branch = `main`
   - `git status --short` had no repo status entries
   - launch `HEAD = e2773d3`
2. Matrix boundary stayed locked:
   - config: `cnn_fpga/config/p4_multiscenario_statcalib_smoke.yaml`
   - scenarios: `static_bias_theta`, `linear_ramp`
   - modes: `ukf`, `hybrid_residual_b`, `statcalib`
   - `--paired-seeds`
   - `--repeats 2`
3. Execution stayed within the T62 one-shot rule:
   - one foreground invocation only
   - no same-run resume
   - one T62 run root only
4. Output integrity:
   - `summary.json` exists
   - `missing_runs_count=0`
   - all rows have `coverage=1.0`
   - all rows have `completed_repeats=2`
5. Provenance closure:
   - finish branch = `main`
   - finish `HEAD = e2773d3`
   - `summary.json git_commit = e2773d3`
   - `progress.jsonl` duplicate `running` entries for the same repeat key = none
6. T59/T61/T62 comparison outcome:
   - `statcalib` remained the winner in both scenarios
   - `statcalib_status=generated`
   - `statcalib_reason=statcalib_params_emitted`
   - `statcalib_generated_windows_mean=600.0`
   - T62 aggregated comparison rows match T61 numerically

### Residual risk

- T62 closes the T61 provenance blocker, but it is still not `FR8`.
- The evidence remains mock-backed software HIL only.
- Any later `FR8` task still requires a separate gate decision rather than automatic promotion.
