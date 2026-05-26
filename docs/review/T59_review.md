# T59 Review

- Verdict: `PASS_WITH_WARNINGS`

## Blocking issues

- None.

## Non-blocking issues

- `cnn_fpga/runtime/slow_loop_runtime.py` now lets `slow_loop.statcalib.teacher_mode` participate in the generic `teacher_mode` fallback chain for other modes. T59's own smoke config is safe because `hybrid_residual_b` declares its own teacher mode explicitly, but this is still a cross-mode coupling risk in future mixed configs.
- `docs/statcalib_comparator_lane_smoke.md` says `hil_summary.json` exposes `statcalib_diagnostics.status` / `reason`. The actual keys are `statcalib_diagnostics.statcalib_status` / `statcalib_diagnostics.statcalib_reason`. This is a small documentation accuracy issue, not a code failure.
- The bounded smoke artifact was generated from a dirty worktree. `runs/p4_benchmark/t59statc_20260526_211532_3a3d00_23740/summary.json` records `git_commit: a40adca`, but that commit hash does not by itself identify the uncommitted T59 diff. The run is still useful review evidence, but provenance is weaker until the patch is landed.

## Missing tests

- No direct unit test covers `run_statcalib_estimator()` branches for `valid_window=false`, zero histogram mass, or explicit clip-boundary behavior.
- No regression test covers benchmark/report aggregation for the new `statcalib_*` columns, especially the expectation that non-statcalib modes stay `not_applicable`.
- No end-to-end artifact exercises the `not_generated` or `diagnostic_error` statcalib path inside the benchmark harness; current smoke only shows the fully generated path.

## Suspicious implementation details

- The estimator is intentionally minimal and not a stub, but it is still a hard-coded controller-like rule: `delta_b = clip(residual_scale_b * [mean_syndrome_q, mean_syndrome_p])`, then apply that delta on top of teacher-mapped params. This is acceptable for T59 integration, but it is not yet strong evidence for any formal comparator claim.
- In both bounded smoke repeats, `statcalib` emitted `generated` on all 600 observed windows. In the final `static_bias_theta` snapshot, the `p` component is already clip-bound (`delta_b_pre_clip = -0.0800499767...` and `applied_delta_b = -0.08`). Combined with the unexpectedly large LER gap, this warrants a dedicated fairness/robustness audit before any FR8-style ranking task.

## Recommended next action

- Accept T59 as an integration-complete comparator-lane task.
- Before any FR8 or result-table task, open a bounded follow-up that removes the cross-mode `teacher_mode` fallback coupling.
- Before any FR8 or result-table task, add direct estimator and aggregation regression tests.
- Before any FR8 or result-table task, run a fairness sanity check on why the minimal statcalib lane is outperforming `ukf` and `hybrid_residual_b` so strongly in this tiny smoke.
