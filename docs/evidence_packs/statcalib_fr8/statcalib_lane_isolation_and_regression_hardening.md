# T60 Statcalib Lane Isolation And Regression Hardening

## Scope

This task was code/test hardening only. It did not rerun `T59`, did not create a new run root, did not modify any config file, and did not make any fairness or `FR8` claim.

## Code And Test Changes

- Hardened `SlowLoopRuntimeConfig.from_config()` in `cnn_fpga/runtime/slow_loop_runtime.py`.
- Extended `tests/test_statcalib_runtime_smoke.py` with direct config-isolation coverage.
- Added `tests/test_statcalib_estimator.py` for direct `run_statcalib_estimator()` branch coverage.
- Added `tests/test_statcalib_aggregation.py` for `statcalib_*` aggregation and report-column coverage.

## Teacher-Mode Isolation Fix

The `teacher_mode` resolution logic now has two paths:

- If `mode=statcalib`, it reads `slow_loop.statcalib.teacher_mode`, with fallback to the pre-existing shared teacher default chain.
- If `mode!=statcalib`, it reads the active mode subtree plus the pre-existing shared teacher default chain, and it no longer consults `slow_loop.statcalib.teacher_mode`.

This removes the `T59` cross-mode coupling risk while preserving historical non-statcalib fallback semantics.

## New Regression Coverage

### Runtime/config isolation

- `test_statcalib_teacher_mode_does_not_leak_into_other_modes`
- `test_statcalib_mode_uses_statcalib_teacher_mode`

### Direct estimator coverage

- invalid-window path
- zero-histogram-mass path
- signal-below-threshold path
- clip-boundary behavior
- diagnostic-error fallback path

### Aggregation/report coverage

- missing statcalib metadata stays `not_applicable` / `mode_does_not_emit_statcalib`
- generated status/count aggregation
- mixed status/reason aggregation
- benchmark status-field default/mixed semantics
- report output keeps the `Statcalib` column visible

## Verification Commands

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m unittest tests.test_statcalib_interface tests.test_statcalib_runtime_smoke tests.test_statcalib_estimator tests.test_statcalib_aggregation
```

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m py_compile cnn_fpga/decoder/statcalib.py cnn_fpga/runtime/slow_loop_runtime.py cnn_fpga/benchmark/run_hil_suite.py cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py tests/test_statcalib_interface.py tests/test_statcalib_runtime_smoke.py tests/test_statcalib_estimator.py tests/test_statcalib_aggregation.py
```

```powershell
git diff --name-only -- runs cnn_fpga/config docs/reference docs/follow-up_plan docs/02_experiment_plan.md docs/04_task_board.md docs/07_handoff.md docs/08_risks_and_open_questions.md cnn_fpga/decoder/param_mapper.py
```

```powershell
git status --short -- runs cnn_fpga/config
```

## Boundary Confirmation

- No new smoke was run.
- No new run root was created under `runs/`.
- No config file was modified under `cnn_fpga/config/`.
- No theory-only materials were touched.

## What Still Remains Before FR8

This task does not change the evidence level of `T59`.

Still open before any later fairness or `FR8` task:

- a bounded fairness/robustness sanity check on the unexpectedly strong `T59` smoke result
- a separate decision on whether the current statcalib lane definition is the comparator worth defending formally
- any broader comparator ranking or expanded matrix execution under a new task package only
