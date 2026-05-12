# T29: P4 markdown report header cleanup after T28

## Status

- Created by Captain on `2026-05-12`
- Current phase: `Phase 2: Controlled Development`
- Decision state: `Go`
- Task type: bounded report-format cleanup

## Why This Task Exists

`T28` repaired teacher diagnostics missing-vs-zero semantics and passed independent review with warnings. The only concrete code bug left by the review is local to `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py::_write_report()`:

- the old markdown table header row was left in place
- a new header row with `Teacher Diag` was added
- the report therefore has two header rows with different column counts

The machine-readable `comparison.csv` is correct, and T28 remains complete. This task fixes the human-readable markdown report formatting before moving on to statcalib or failure-mechanism work.

## Goal

Remove the duplicate old markdown report header and verify the `_write_report()` markdown table header / separator / data rows are structurally consistent.

## Allowed Files

Worker may modify:

- `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py`
- `docs/review/T29_review.md`
- `docs/for_human/T29_explanation.md`
- `docs/tasks/Phase2/T29_p4_report_header_cleanup.md`

If a focused existing test already covers report formatting, Worker may modify that test file. Do not create broad test scaffolding.

## Required Inputs

Read at minimum:

- `docs/review/T28_review.md`
- `docs/tasks/Phase2/T28_teacher_diagnostics_semantics_repair.md`
- `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py`

## Forbidden Scope

Do not:

- modify `docs/02_experiment_plan.md`
- run benchmark, training, `.tflite`, hardware, or cleanup
- create a new run directory
- change teacher diagnostics semantics, comparison CSV columns, aggregation behavior, baseline/scenario set, seed policy, ParamMapper semantics, or formal benchmark protocol
- modify historical `runs/` or `artifacts/`
- touch tracked `.pyc` files intentionally
- claim new benchmark or mechanism evidence

## Expected Repair Shape

Expected code change should be minimal:

- remove the old 11-column markdown report header row in `_write_report()`
- keep the new 12-column header with `Teacher Diag`
- keep separator and data rows aligned

## Verification

Worker must perform lightweight verification only:

1. Static syntax check:
   - `python -m py_compile cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py`
2. Inspect or script-check `_write_report()` output shape without running benchmark:
   - exactly one markdown table header row for the P4 comparison table
   - header includes `Teacher Diag`
   - header, separator, and representative data row have matching column counts

Use the project-recommended interpreter where practical:

- `C:\ProgramData\anaconda3\python.exe`

## Expected Output

Create `docs/review/T29_review.md` with:

- changed files
- exact fix
- verification commands and outputs
- remaining risks

Create `docs/for_human/T29_explanation.md` with a concise explanation.

Update this task package with Worker Output and Verification Record.

## Captain Notes

After T29, reasonable next candidates are:

- `T26`: Calibration/statcalib baseline feasibility gate and minimal design plan
- `T36`: `seed=20260429` failure-mechanism diagnosis

Do not start either inside T29.

## Worker Output

- Updated `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py`.
- Created:
  - `docs/review/T29_review.md`
  - `docs/for_human/T29_explanation.md`

Exact fix:

- removed the old 11-column markdown comparison-table header row from `_write_report()`
- kept the new 12-column header row that includes `Teacher Diag`
- left separator and data-row rendering unchanged

No benchmark/report semantics were changed beyond this local markdown cleanup.

## Verification Record

Static verification:

- Command:
  - `& 'C:\ProgramData\anaconda3\python.exe' -m py_compile cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py`
- Result:
  - passed

Format verification without running benchmark:

- Method:
  - invoked `_write_report()` in a temporary directory with one representative comparison row
  - verified the generated markdown in-memory / temp-file scope only
  - did not create a benchmark run dir
- Observed output:
  - `header_rows=1`
  - `teacher_diag_header=yes`
  - `column_counts=[12, 12, 12]`

Boundary confirmation:

- no benchmark, training, `.tflite`, hardware, or cleanup command was run
- no new run directory was created
- no historical `runs/` or `artifacts/` output was modified
