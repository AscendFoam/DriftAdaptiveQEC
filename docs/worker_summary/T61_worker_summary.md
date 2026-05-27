# T61 Worker Summary: Statcalib Clean-Provenance Fairness Sanity

## What Changed

1. Executed the bounded T61 sanity matrix in the single allowed run root:
   - `runs/p4_benchmark/T61_statcalib_fairness_sanity_20260527_015239`
2. Created the T61 result doc:
   - `docs/statcalib_fairness_sanity.md`
3. Created review and human-summary docs:
   - `docs/review/T61_review.md`
   - `docs/for_human/T61_explanation.md`
4. Updated the task package with worker output:
   - `docs/tasks/Phase2/T61_statcalib_clean_provenance_fairness_sanity.md`

## Verification

1. Preflight cleanliness:
   - `git status --short` had no repo status entries before the run
   - preflight `git rev-parse --short HEAD` was `9174065`
2. Bounded matrix execution:
   - config: `cnn_fpga/config/p4_multiscenario_statcalib_smoke.yaml`
   - scenarios: `static_bias_theta`, `linear_ramp`
   - modes: `ukf`, `hybrid_residual_b`, `statcalib`
   - `--paired-seeds`
   - `--repeats 2`
3. Output integrity:
   - `summary.json` exists in the T61 run root
   - `missing_runs_count=0`
   - all rows have `coverage=1.0` and `completed_repeats=2`
4. Fairness sanity:
   - `statcalib` remained the winner in both scenarios
   - `statcalib_status=generated`
   - `statcalib_reason=statcalib_params_emitted`
   - `statcalib_generated_windows_mean=600.0`
5. Provenance issue explicitly captured:
   - final `summary.json` anchor is `git_commit=6058f42`
   - clean-start anchor was `9174065`
   - `git reflog` shows an in-flight checkout during execution

## Remaining Risks

- The strong `statcalib` result persisted, but T61 did not fully close the provenance blocker because the run artifact anchor drifted during execution.
- Since `9174065` and `6058f42` differ in source/config/test paths, this run should not be promoted into `FR8`.
- The evidence remains mock-backed software HIL only. No `.tflite` or real-board claim is supported.
