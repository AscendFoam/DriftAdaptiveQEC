# T60 Review

- Verdict: `PASS`
- Review basis: I rechecked the live diff, reran the focused `unittest` suite, reran `py_compile`, and checked that `runs/` and `cnn_fpga/config/` stayed unchanged.

## Blocking issues

- None.
- `slow_loop.statcalib.teacher_mode` no longer leaks into non-`statcalib` modes. The isolation is now explicit in `cnn_fpga/runtime/slow_loop_runtime.py:115-127`, and the new regression tests in `tests/test_statcalib_runtime_smoke.py:93-120` cover both the non-leak case and the `mode=statcalib` case.

## Non-blocking issues

- None within the T60 task boundary.
- Residual project-level caution only: T59's dirty-worktree smoke provenance weakness and the later fairness/robustness question are still open before any future `FR8` task, but T60 was not supposed to solve either one.

## Missing tests

- None relative to T60's stated goal.
- Direct regression coverage now exists for:
  - runtime teacher-mode isolation: `tests/test_statcalib_runtime_smoke.py:93-120`
  - estimator negative / clip / error branches: `tests/test_statcalib_estimator.py:49-86`
  - HIL aggregation, benchmark status-field semantics, and report-column visibility: `tests/test_statcalib_aggregation.py:12-114`
- A future comparator-evidence task may still want an end-to-end harness case for `not_generated` or `diagnostic_error`, but that is beyond T60's required scope.

## Suspicious implementation details

- `tests/test_statcalib_estimator.py:81-86` reaches the diagnostic-error branch with a deliberately invalid argument type. That is acceptable as branch hardening, but it is synthetic coverage, not evidence that a real runtime path currently reaches the same failure mode.
- The `statcalib` lane itself is still the same intentionally minimal heuristic comparator introduced in T59. T60 hardens its semantics and regression net; it does not and should not be read as formal comparator evidence.

## Recommended next action

- Accept T60 as complete.
- Treat T59 warning `W1` as closed by this task.
- If `statcalib` is still being considered for later `FR8` work, open a separate bounded fairness/robustness sanity task plus a provenance-clean rerun policy, instead of widening T60.
