# T60 Worker Summary

## What Changed

- Isolated `statcalib.teacher_mode` lookup in `cnn_fpga/runtime/slow_loop_runtime.py` so only `mode=statcalib` can consult the `slow_loop.statcalib` subtree.
- Extended `tests/test_statcalib_runtime_smoke.py` with direct teacher-mode isolation regression tests.
- Added `tests/test_statcalib_estimator.py` for direct estimator branch coverage.
- Added `tests/test_statcalib_aggregation.py` for HIL aggregation, benchmark status-field, and report-column coverage.

## Verification

- `C:\ProgramData\anaconda3\python.exe -m unittest tests.test_statcalib_interface tests.test_statcalib_runtime_smoke tests.test_statcalib_estimator tests.test_statcalib_aggregation`
- `C:\ProgramData\anaconda3\python.exe -m py_compile cnn_fpga/decoder/statcalib.py cnn_fpga/runtime/slow_loop_runtime.py cnn_fpga/benchmark/run_hil_suite.py cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py tests/test_statcalib_interface.py tests/test_statcalib_runtime_smoke.py tests/test_statcalib_estimator.py tests/test_statcalib_aggregation.py`
- `git diff --name-only -- runs cnn_fpga/config docs/reference docs/follow-up_plan docs/02_experiment_plan.md docs/04_task_board.md docs/07_handoff.md docs/08_risks_and_open_questions.md cnn_fpga/decoder/param_mapper.py`
- `git status --short -- runs cnn_fpga/config`

## Residual Risk

- This task still does not upgrade `T59` to formal comparator evidence.
- The next meaningful blocker is fairness/robustness, not code semantics: `T59` still showed an unexpectedly strong generated-only smoke result.
- No conclusion about `FR8` or comparator ranking should be drawn from this hardening pass alone.
