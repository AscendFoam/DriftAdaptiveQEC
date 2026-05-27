# T62 Worker Summary: Statcalib Provenance-Isolated Fairness Rerun

## What Changed

1. Executed the bounded T62 sanity matrix in the single allowed run root:
   - `runs/p4_benchmark/T62_statcalib_provenance_isolated_20260527_122943`
2. Created the T62 result doc:
   - `docs/statcalib_provenance_isolated_fairness_rerun.md`
3. Created review and human-summary docs:
   - `docs/review/T62_review.md`
   - `docs/for_human/T62_explanation.md`
4. Updated the task package with worker output:
   - `docs/tasks/Phase2/T62_statcalib_provenance_isolated_fairness_rerun.md`

## Verification

1. Preflight:
   - branch = `main`
   - `git status --short` had no repo status entries
   - launch `HEAD = e2773d3`
2. Execution:
   - one uninterrupted foreground invocation only
   - no same-run resume
   - config: `cnn_fpga/config/p4_multiscenario_statcalib_smoke.yaml`
   - scenarios: `static_bias_theta`, `linear_ramp`
   - modes: `ukf`, `hybrid_residual_b`, `statcalib`
   - `--paired-seeds`
   - `--repeats 2`
3. Output integrity:
   - one T62 run root only
   - `summary.json` exists
   - `missing_runs_count=0`
   - all rows have `coverage=1.0`
   - all rows have `completed_repeats=2`
4. Provenance closure:
   - finish branch = `main`
   - finish `HEAD = e2773d3`
   - `summary.json git_commit = e2773d3`
   - `progress.jsonl` has no duplicate `running` entry for the same repeat key
5. Bounded fairness sanity:
   - `statcalib` remained the winner in both scenarios
   - `statcalib_status=generated`
   - `statcalib_reason=statcalib_params_emitted`
   - `statcalib_generated_windows_mean=600.0`
   - T62 aggregated comparison rows match T61 numerically

## Remaining Risks

- T62 closes the T61 provenance blocker, but it still does not upgrade the evidence into `FR8`.
- The evidence remains mock-backed software HIL only. No `.tflite` or real-board claim is supported.
- Any later `FR8` task would still need a separate gate decision on whether the current `statcalib` lane definition is defendable as a formal comparator.
