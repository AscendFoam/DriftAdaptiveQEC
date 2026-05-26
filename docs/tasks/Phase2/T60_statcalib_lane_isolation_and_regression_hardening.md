# T60: Statcalib lane isolation and regression hardening

## Status

- Proposed by Captain on `2026-05-26`
- Current phase: `Phase 2: Controlled Development`
- Decision state: `Go`
- Task type: bounded code + test hardening task on the mainline experiment branch

## Why This Task Exists

`T59` has now closed the first integrated `statcalib` lane gap honestly: the repository can run a separately labeled `statcalib` slow-loop mode and carry `statcalib_status` / `statcalib_reason` through the bounded smoke outputs.

However, `T59` review was accepted only as `PASS_WITH_WARNINGS`, not as a clean handoff to `FR8`.

The main deferred issues are:

1. `slow_loop.statcalib.teacher_mode` is now present in the generic `teacher_mode` fallback chain, which creates a cross-mode coupling risk
2. direct estimator-branch and aggregation regression coverage is still incomplete
3. the current smoke remains integration evidence only, not formal comparator evidence

So the next smallest honest step is **not** an `FR8` result-table task yet. The next step is to harden lane isolation and regression coverage without widening benchmark scope.

This task must stay on the mainline experiment branch and must not interfere with theory-only branch materials.

## Goal

Produce the smallest believable hardening pass by answering:

1. does `statcalib` configuration stay isolated to `mode=statcalib`
2. are estimator negative/clip/error branches covered directly
3. are aggregation/report semantics for `statcalib_*` fields covered directly
4. can all of that be done without creating any new run root or smoke artifact

## Allowed Files

Worker may modify only:

- `docs/tasks/Phase2/T60_statcalib_lane_isolation_and_regression_hardening.md`
- `docs/statcalib_lane_isolation_and_regression_hardening.md`
- `docs/review/T60_review.md`
- `docs/for_human/T60_explanation.md`
- `docs/worker_summary/T60_worker_summary.md`
- `cnn_fpga/decoder/statcalib.py`
- `cnn_fpga/runtime/slow_loop_runtime.py`
- `cnn_fpga/benchmark/run_hil_suite.py`
- `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py`
- `tests/test_statcalib_interface.py`
- `tests/test_statcalib_runtime_smoke.py`

Worker may create:

- `tests/test_statcalib_estimator.py`
- `tests/test_statcalib_aggregation.py`

## Docs To Update

This task should update only:

1. `docs/statcalib_lane_isolation_and_regression_hardening.md`
2. `docs/review/T60_review.md`
3. `docs/for_human/T60_explanation.md`
4. `docs/worker_summary/T60_worker_summary.md`
5. this task package itself, only to append Worker output and verification notes after completion

## Forbidden Scope

This task must not:

1. edit `docs/00_project_snapshot.md`, `docs/01_legacy_audit.md`, `docs/02_experiment_plan.md`, `docs/03_hil_p4_boundary_audit.md`, `docs/04_task_board.md`, `docs/05_decision_log.md`, `docs/06_repo_noise_governance.md`, `docs/07_handoff.md`, or `docs/08_risks_and_open_questions.md`
2. edit theory-only materials such as `docs/reference/延申理论.md`, `docs/reference/延伸改进思路.md`, or files under `docs/follow-up_plan/`
3. create or modify any benchmark config file, smoke config file, or run root
4. run any new bounded smoke, benchmark rerun, fairness rerun, `.tflite` task, real-board task, cleanup task, training task, or intervention task
5. modify `cnn_fpga/decoder/param_mapper.py` or silently change existing `ukf`, `window_variance`, `cnn_fpga`, `hybrid_residual_b`, or frozen `T24` semantics beyond isolating the statcalib fallback lookup
6. rewrite or delete historical `runs/` or `artifacts/`
7. write any `FR8` formal result-table claim

## Required Inputs

Read at minimum:

- `docs/04_task_board.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/review/T59_review.md`
- `docs/statcalib_comparator_lane_smoke.md`
- `docs/worker_summary/T59_worker_summary.md`
- `cnn_fpga/decoder/statcalib.py`
- `cnn_fpga/runtime/slow_loop_runtime.py`
- `cnn_fpga/benchmark/run_hil_suite.py`
- `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py`
- `tests/test_statcalib_interface.py`
- `tests/test_statcalib_runtime_smoke.py`

## Fixed Boundary

This task is locked to the following boundary:

1. `T59` is already complete and must not be re-opened into a new smoke lane
2. this task is code/test hardening only
3. no new run root is allowed
4. no fairness or ranking conclusion is allowed
5. this task is still **not** `FR8`

## Implementation Requirements

### 1. Isolate `teacher_mode` lookup

Harden `SlowLoopRuntimeConfig.from_config()` so that:

- `slow_loop.statcalib.teacher_mode` is consulted only when `mode=statcalib`
- non-statcalib modes cannot inherit `statcalib.teacher_mode` implicitly through the generic fallback chain
- existing frozen mainline semantics remain unchanged for historical configs

### 2. Add direct estimator tests

Add focused tests for `run_statcalib_estimator()` coverage, including at minimum:

- invalid window path
- zero histogram mass path
- signal-below-threshold path
- clip-boundary behavior
- diagnostic-error fallback path

These should be direct contract-level tests, not benchmark reruns.

### 3. Add aggregation/report regression tests

Add focused tests for the new `statcalib_*` aggregation path, including at minimum:

- non-statcalib modes remain `not_applicable` / `mode_does_not_emit_statcalib`
- generated status aggregates correctly
- mixed or missing status behavior stays explicit and deterministic

If a tiny helper extraction is needed to make this testable, keep it narrowly scoped to aggregation semantics only.

### 4. Keep artifact history frozen

Do not regenerate `T59` smoke outputs.

This task must leave:

- `runs/p4_benchmark/t59statc_20260526_211532_3a3d00_23740`
- historical benchmark outputs
- historical `summary.json` / `comparison.csv` facts

fully unchanged.

## Expected Output Artifacts

Create `docs/statcalib_lane_isolation_and_regression_hardening.md` with:

1. exact code/test changes made
2. exact semantics of the `teacher_mode` isolation fix
3. exact new tests added
4. exact verification commands run
5. explicit statement that no new smoke or run root was created
6. explicit statement of what still remains before any later fairness or `FR8` task

Create `docs/review/T60_review.md` with:

1. scope and boundary check
2. confirmation that `statcalib.teacher_mode` no longer leaks across modes
3. confirmation that regression coverage now covers estimator and aggregation branches
4. confirmation that no new run root or smoke artifact was created
5. whether the next step should be a fairness sanity task or another prerequisite

Create `docs/for_human/T60_explanation.md` with a short human-facing summary.

Create `docs/worker_summary/T60_worker_summary.md` with a concise worker-facing summary of changes, verification, and residual risk.

## Verification

Required verification:

1. run focused unit tests covering:
   - `tests.test_statcalib_interface`
   - `tests.test_statcalib_runtime_smoke`
   - any new estimator/aggregation test modules created in this task
2. run `py_compile` or equivalent minimal static verification on changed Python files
3. confirm no new run root was created under `runs/`
4. confirm no config file was modified
5. confirm no theory-only branch materials were touched

Suggested command shape:

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m unittest tests.test_statcalib_interface tests.test_statcalib_runtime_smoke tests.test_statcalib_estimator tests.test_statcalib_aggregation
```

If the final command differs, document exactly why.

## Review No-Go Triggers

Review should be treated as `BLOCK` if any of the following happen:

1. `statcalib.teacher_mode` can still affect non-statcalib modes implicitly
2. the worker creates or modifies any new run root
3. the worker reruns or rewrites the historical `T59` smoke artifact
4. the worker widens scope into fairness ranking, `FR8`, `.tflite`, real-board, cleanup, training, or theory-only work
5. direct estimator/aggregation regression coverage is still absent after the supposed hardening pass

## Captain Notes

`T59` is complete and accepted as `PASS_WITH_WARNINGS`.

This task exists to harden semantics and regression coverage before any later fairness sanity check or `FR8` task. It must not reopen benchmark execution.

## Worker Output

### What changed

- Hardened `cnn_fpga/runtime/slow_loop_runtime.py` so `slow_loop.statcalib.teacher_mode` is only consulted when `mode=statcalib`.
- Extended `tests/test_statcalib_runtime_smoke.py` with direct teacher-mode isolation regression coverage.
- Added `tests/test_statcalib_estimator.py` covering invalid-window, zero-mass, below-threshold, clip-boundary, and diagnostic-error estimator branches.
- Added `tests/test_statcalib_aggregation.py` covering statcalib aggregation defaults, generated/mixed semantics, benchmark status-field behavior, and report-column visibility.
- Added task-scoped closeout docs:
  - `docs/statcalib_lane_isolation_and_regression_hardening.md`
  - `docs/review/T60_review.md`
  - `docs/for_human/T60_explanation.md`
  - `docs/worker_summary/T60_worker_summary.md`

### Verification Notes

- Focused unit tests passed:
  - `C:\ProgramData\anaconda3\python.exe -m unittest tests.test_statcalib_interface tests.test_statcalib_runtime_smoke tests.test_statcalib_estimator tests.test_statcalib_aggregation`
- Static verification passed:
  - `C:\ProgramData\anaconda3\python.exe -m py_compile cnn_fpga/decoder/statcalib.py cnn_fpga/runtime/slow_loop_runtime.py cnn_fpga/benchmark/run_hil_suite.py cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py tests/test_statcalib_interface.py tests/test_statcalib_runtime_smoke.py tests/test_statcalib_estimator.py tests/test_statcalib_aggregation.py`
- Range checks passed:
  - `git diff --name-only -- runs cnn_fpga/config docs/reference docs/follow-up_plan docs/02_experiment_plan.md docs/04_task_board.md docs/07_handoff.md docs/08_risks_and_open_questions.md cnn_fpga/decoder/param_mapper.py`
  - `git status --short -- runs cnn_fpga/config`
- No new smoke was run, no new run root was created, and no config file was modified in this task.
