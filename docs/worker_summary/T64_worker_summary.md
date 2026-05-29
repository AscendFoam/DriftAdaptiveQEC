# T64 Worker Summary: FR8 Statcalib Extension-Lane Bounded Benchmark

## What Changed

1. Added the task-scoped derived config:
   - `cnn_fpga/config/p4_multiscenario_statcalib_extension_lane.yaml`
2. Executed the bounded T64 benchmark in the single allowed run root:
   - `runs/p4_benchmark/T64_fr8_statcalib_extension_lane_20260527_221658`
3. Created the T64 result doc:
   - `docs/fr8_statcalib_extension_lane_benchmark.md`
4. Created review and human-summary docs:
   - `docs/review/T64_review.md`
   - `docs/for_human/T64_explanation.md`
5. Updated the task package with worker output:
   - `docs/tasks/Phase2/T64_fr8_statcalib_extension_lane_bounded_benchmark.md`

## Verification

1. Preflight passed:
   - branch = `main`
   - launch `HEAD = 1e59f24`
   - `git status --short` had no repo status entries before launch
2. The derived config preserved the locked benchmark boundary:
   - scenarios: `static_bias_theta`, `linear_ramp`, `step_sigma_theta`, `periodic_drift`
   - frozen modes in order: `ekf`, `ukf`, `constant_residual_mu`, `rls_residual_b`, `hybrid_residual_b`
   - `statcalib` appended last
   - `--paired-seeds`
   - `--repeats 2`
3. Execution integrity passed:
   - one-shot full-matrix run only
   - one T64 run root only
   - no same-run resume
   - `summary.json git_commit = 1e59f24`
   - finish `HEAD = 1e59f24`
4. Output integrity passed:
   - `comparison_rows_count=24`
   - `raw_rows_count=48`
   - `missing_runs_count=0`
   - all rows have `coverage=1.0`
   - all rows have `completed_repeats=2`
   - `progress.jsonl` has no duplicate `running` entry for the same repeat key
5. Frozen-table preservation passed:
   - T64 frozen subset matches `T24` exactly across all 20 frozen comparison rows
   - max absolute delta in `final_ler_mean` = `0`
   - max absolute delta in `overflow_rate_mean` = `0`
6. Extension-lane result:
   - `statcalib` won all four scenarios
   - `statcalib_status=generated`
   - `statcalib_reason=statcalib_params_emitted`
   - `statcalib_generated_windows_mean` stayed at `899.5` or `900.0`
7. Historical-root isolation passed:
   - no historical benchmark config was modified
   - inspected last-write timestamps for `T24`, `T59`, `T61`, and `T62` remained unchanged
   - no source or test file was modified

## Remaining Risks

- T64 is still mock-backed software-HIL evidence only.
- T64 does not validate `.tflite` runtime behavior.
- T64 does not validate real-board behavior.
- T64 does not replace the historical `T24` frozen ranked table.
- T64 strengthens the bounded comparator story, but it still does not by itself make the lane paper-grade expanded benchmark evidence.
