# T61: Statcalib clean-provenance fairness sanity rerun

## Status

- Proposed by Captain on `2026-05-27`
- Current phase: `Phase 2: Controlled Development`
- Decision state: `Go`
- Task type: bounded execution + audit task on the mainline experiment branch

## Why This Task Exists

`T59` integrated the separate `statcalib` comparator lane and proved one bounded smoke could run end-to-end. `T60` then closed the cross-mode semantics blocker and the regression-coverage blocker honestly.

So the remaining pre-`FR8` blocker is no longer code semantics. The remaining blocker is evidence quality:

1. the only integrated statcalib smoke (`T59`) was produced from a dirty worktree
2. that smoke showed an unexpectedly strong `statcalib` win in both scenarios
3. before any `FR8` result-table task, the project needs one clean-provenance bounded sanity rerun to determine whether this result is stable enough to keep taking seriously

This task must stay on the mainline experiment branch and must not interfere with theory-only branch materials.

## Goal

Produce the smallest believable sanity pass by answering:

1. can the T59 matrix be rerun from a clean committed worktree
2. does `statcalib` still emit `generated` consistently under that rerun
3. does the strong advantage over `ukf` and `hybrid_residual_b` persist, weaken, or collapse
4. after that rerun, is the next honest step `FR8`, or is another blocker still open

## Allowed Files

Worker may modify only:

- `docs/tasks/Phase2/T61_statcalib_clean_provenance_fairness_sanity.md`
- `docs/statcalib_fairness_sanity.md`
- `docs/review/T61_review.md`
- `docs/for_human/T61_explanation.md`
- `docs/worker_summary/T61_worker_summary.md`

Worker may create only:

- one task-scoped run root under `runs/p4_benchmark/T61_statcalib_fairness_sanity_*`

## Docs To Update

This task should update only:

1. `docs/statcalib_fairness_sanity.md`
2. `docs/review/T61_review.md`
3. `docs/for_human/T61_explanation.md`
4. `docs/worker_summary/T61_worker_summary.md`
5. this task package itself, only to append Worker output and verification notes after completion

## Forbidden Scope

This task must not:

1. edit `docs/00_project_snapshot.md`, `docs/01_legacy_audit.md`, `docs/02_experiment_plan.md`, `docs/03_hil_p4_boundary_audit.md`, `docs/04_task_board.md`, `docs/05_decision_log.md`, `docs/06_repo_noise_governance.md`, `docs/07_handoff.md`, or `docs/08_risks_and_open_questions.md`
2. edit theory-only materials such as `docs/reference/寤剁敵鐞嗚.md`, `docs/reference/寤朵几鏀硅繘鎬濊矾.md`, or files under `docs/follow-up_plan/`
3. modify any source file, test file, or source-tree config file under `cnn_fpga/config/`
4. widen the matrix beyond `static_bias_theta` / `linear_ramp`, `ukf` / `hybrid_residual_b` / `statcalib`, `--paired-seeds`, and `repeats=2`
5. rerun or rewrite the historical `T59` run root
6. run `.tflite`, real-board, cleanup, training, intervention, benchmark-expansion, or theory-branch work
7. write any `FR8` formal result-table claim

## Required Inputs

Read at minimum:

- `docs/04_task_board.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/review/T59_review.md`
- `docs/review/T60_review.md`
- `docs/statcalib_comparator_lane_smoke.md`
- `docs/statcalib_lane_isolation_and_regression_hardening.md`
- `docs/worker_summary/T59_worker_summary.md`
- `docs/worker_summary/T60_worker_summary.md`
- `cnn_fpga/config/p4_multiscenario_statcalib_smoke.yaml`
- `runs/p4_benchmark/t59statc_20260526_211532_3a3d00_23740/summary.json`

## Fixed Boundary

This task is locked to the following boundary:

1. `T60` is already complete and must not be reopened into new code changes
2. this task is execution/audit only
3. exactly one new T61-scoped run root is allowed
4. the run must reuse the existing T59 smoke config and mode/scenario set
5. the only allowed CLI strengthening is `--repeats 2`
6. this task is still **not** `FR8`

## Implementation Requirements

### 1. Preflight provenance check

Before creating docs or launching the rerun:

- confirm the worktree is clean with `git status --short`
- record the exact `git rev-parse --short HEAD` value
- if the worktree is not clean, stop and report `BLOCKED_BY_DIRTY_WORKTREE`

This task exists partly to repair provenance weakness, so the rerun must not start from another dirty state.

### 2. Run one bounded sanity matrix

Reuse the existing config:

- `cnn_fpga/config/p4_multiscenario_statcalib_smoke.yaml`

Run only:

- scenarios: `static_bias_theta`, `linear_ramp`
- modes: `ukf`, `hybrid_residual_b`, `statcalib`
- `--paired-seeds`
- `--repeats 2`

Use a fixed T61-scoped run directory under:

- `runs/p4_benchmark/T61_statcalib_fairness_sanity_*`

Do not modify the source-tree config to do this; use CLI filtering/override only.

### 3. Compare T61 against T59

The output report must compare:

- per-scenario ranking
- `final_ler_mean`
- runner-up gap
- `statcalib_status` / `statcalib_reason`
- `statcalib_generated_windows_mean`
- whether the strong T59 result persists, weakens, or collapses

This comparison is a sanity audit only, not a formal comparator claim.

### 4. Keep evidence boundary explicit

The final docs must say explicitly:

- this remains mock-backed software-HIL evidence
- this is not `FR8`
- this does not validate `.tflite` runtime
- this does not validate real-board behavior
- a positive T61 result only means the lane deserves the next gate discussion

## Expected Output Artifacts

Create `docs/statcalib_fairness_sanity.md` with:

1. exact clean-worktree preflight result
2. exact rerun command
3. exact run root
4. exact `git_commit` anchoring seen in the new run
5. per-scenario comparison table for `ukf` / `hybrid_residual_b` / `statcalib`
6. direct comparison against the T59 smoke outcome
7. explicit statement of whether the strong `statcalib` result persisted, weakened, or collapsed
8. explicit statement of what still remains before any later `FR8` task

Create `docs/review/T61_review.md` with:

1. scope and boundary check
2. confirmation that the rerun started from a clean committed worktree
3. confirmation that only one T61-scoped run root was created
4. confirmation that the matrix stayed bounded
5. whether the next step should be `FR8` or another prerequisite

Create `docs/for_human/T61_explanation.md` with a short human-facing summary.

Create `docs/worker_summary/T61_worker_summary.md` with a concise worker-facing summary of changes, verification, and residual risk.

## Verification

Required verification:

1. `git status --short` is empty before the run starts
2. run the bounded benchmark command with:
   - existing T59 config
   - `--paired-seeds`
   - `--repeats 2`
   - fixed T61-scoped `--run-dir`
3. confirm the new `summary.json` records the clean `git_commit`
4. confirm no source, test, config, governance, or theory-only files were modified
5. confirm no historical `T59` output was rewritten

Suggested command shape:

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config cnn_fpga/config/p4_multiscenario_statcalib_smoke.yaml --scenario static_bias_theta --scenario linear_ramp --mode ukf --mode hybrid_residual_b --mode statcalib --paired-seeds --repeats 2 --run-dir runs/p4_benchmark/T61_statcalib_fairness_sanity_<timestamp>
```

If the final command differs, document exactly why.

## Review No-Go Triggers

Review should be treated as `BLOCK` if any of the following happen:

1. the rerun starts from a dirty worktree
2. the worker modifies source code, tests, or source-tree config
3. the worker widens the matrix beyond the locked two scenarios / three modes / paired-seeds / repeats=2 boundary
4. the worker rewrites or resumes into the historical T59 run root
5. the worker writes any `FR8` or formal comparator-ranking claim

## Captain Notes

`T60` is complete and accepted as `PASS`.

This task exists to repair the remaining provenance/fairness blocker before any later `FR8` discussion. It must not be merged with source changes or theory-branch work.

## Worker Output

### What changed

1. Executed the bounded T61 sanity matrix in the single allowed run root:
   - `runs/p4_benchmark/T61_statcalib_fairness_sanity_20260527_015239`
2. Created:
   - `docs/statcalib_fairness_sanity.md`
   - `docs/review/T61_review.md`
   - `docs/for_human/T61_explanation.md`
   - `docs/worker_summary/T61_worker_summary.md`
3. Appended the worker output back into this task package on `main`.

### Verification record

1. Preflight cleanliness passed:
   - `git status --short` had no repo status entries before the run
   - preflight `git rev-parse --short HEAD` was `9174065`
2. Matrix boundary stayed locked:
   - config: `cnn_fpga/config/p4_multiscenario_statcalib_smoke.yaml`
   - scenarios: `static_bias_theta`, `linear_ramp`
   - modes: `ukf`, `hybrid_residual_b`, `statcalib`
   - `--paired-seeds`
   - `--repeats 2`
3. Only one T61 run root exists:
   - `runs/p4_benchmark/T61_statcalib_fairness_sanity_20260527_015239`
4. T61 output integrity:
   - `summary.json` exists
   - `missing_runs_count=0`
   - all rows have `coverage=1.0`
   - all rows have `completed_repeats=2`
5. T59 comparison outcome:
   - `statcalib` remained the winner in both scenarios
   - `statcalib_status=generated`
   - `statcalib_reason=statcalib_params_emitted`
   - `statcalib_generated_windows_mean=600.0`
6. Remaining blocker:
   - final `summary.json` anchor is `git_commit=6058f42`
   - clean-start anchor was `9174065`
   - `git reflog` shows an in-flight checkout during execution
   - therefore the strong result persisted, but provenance was not fully repaired

### Residual risk

- This remains mock-backed software HIL evidence only.
- This should not be promoted into `FR8`.
- The next honest prerequisite is a provenance-isolated rerun path, not a formal comparator claim.
