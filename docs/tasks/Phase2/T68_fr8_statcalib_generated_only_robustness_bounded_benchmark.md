# T68: FR8 statcalib generated-only robustness bounded benchmark

## Status

- Proposed by Captain on `2026-06-05`
- Current phase: `Phase 2: Controlled Development`
- Decision state: `Go`
- Task type: bounded benchmark execution + grouped robustness summary task on the mainline experiment branch

## Why This Task Exists

`T66` already answered the local-parameter fragility question under the `ukf` teacher anchor.

`T67` then answered the gross teacher-anchor dependence question honestly:

1. the bounded statcalib gain is not narrowly tied to `teacher_mode=ukf`
2. non-`ukf` teachers remain strong
3. all six T67 teacher-anchor variants still beat both frozen anchors in all four locked scenarios

That means the next unresolved mainline question is narrower than before:

1. the strongest aggregate statcalib lane still carries `mixed` provenance on some rows
2. the repository still does not know whether a predeclared non-`ukf` candidate can keep all-four-scenario wins while avoiding `mixed` rows

The next smallest honest step is therefore not more prose, not `.tflite`, not real-board work, and not a broader benchmark expansion. It is one bounded generated-only robustness benchmark that:

1. keeps the historical `T24` frozen ranked table authoritative
2. keeps `statcalib` as a separately labeled extension lane only
3. reuses the strongest non-`ukf` teacher anchors from `T67`
4. varies only `signal_threshold` across a small predeclared interpolation grid
5. preserves clean provenance and paired-seed fairness
6. produces one grouped summary pack aimed directly at the remaining `R24` question

## Goal

Produce one bounded generated-only robustness package that answers:

1. whether any predeclared non-`ukf` statcalib candidate beats both frozen anchors in all four locked scenarios while keeping `statcalib_status = generated` in all four scenario rows
2. if no such candidate exists, whether the remaining `mixed` provenance looks intrinsic to the current bounded lane rather than like a missing threshold interpolation
3. whether `R24` can be narrowed from "extension lane still mixed" to a more precise generated-only limitation statement

## Allowed Files

Worker may modify only:

- `docs/tasks/Phase2/T68_fr8_statcalib_generated_only_robustness_bounded_benchmark.md`
- `docs/statcalib_generated_only_robustness_bounded_benchmark.md`
- `docs/review/T68_review.md`
- `docs/for_human/T68_explanation.md`
- `docs/worker_summary/T68_worker_summary.md`
- `cnn_fpga/config/p4_multiscenario_statcalib_generated_only.yaml`
- `cnn_fpga/benchmark/summarize_statcalib_generated_only.py`
- `tests/test_statcalib_generated_only_summary.py`
- exactly one new run root under `runs/p4_benchmark/T68_statcalib_generated_only_*`

## Docs To Update

Worker must update:

- `docs/statcalib_generated_only_robustness_bounded_benchmark.md`
- `docs/review/T68_review.md`
- `docs/for_human/T68_explanation.md`
- `docs/worker_summary/T68_worker_summary.md`

Worker must not update governance docs. Captain will do that after review.

## Forbidden Scope

Worker must not:

- modify `docs/02_experiment_plan.md`
- modify any governance doc under `docs/00_*` to `docs/08_*`
- modify `cnn_fpga/decoder/statcalib.py`
- modify `cnn_fpga/runtime/slow_loop_runtime.py`
- modify `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py`
- modify historical config files such as `cnn_fpga/config/p4_multiscenario_strong_baselines.yaml`, `cnn_fpga/config/p4_multiscenario_statcalib_extension_lane.yaml`, `cnn_fpga/config/p4_multiscenario_statcalib_sensitivity.yaml`, or `cnn_fpga/config/p4_multiscenario_statcalib_teacher_anchor.yaml`
- modify any historical file under `runs/`
- create more than one `T68` run root
- change benchmark semantics, comparator semantics, or runtime semantics
- widen the candidate search into scale / clip / EMA / scenario / baseline expansion beyond the matrix explicitly declared below
- widen into `.tflite`, real-board, training, cleanup, or theory-only branch work
- rewrite `T24`, `T64`, `T65`, `T66`, or `T67` into mature calibration-comparator, deployment, or paper-grade evidence

## Required Inputs

Worker must reuse:

- `docs/P4_benchmark_formal_protocol.md`
- `docs/fr8_statcalib_extension_lane_benchmark.md`
- `docs/fr8_statcalib_extension_lane_consistency_audit.md`
- `docs/statcalib_sensitivity_bounded_benchmark.md`
- `docs/statcalib_teacher_anchor_bounded_benchmark.md`
- `docs/review/T64_review.md`
- `docs/review/T65_review.md`
- `docs/review/T66_review.md`
- `docs/review/T67_review.md`
- `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743`
- `runs/p4_benchmark/T64_fr8_statcalib_extension_lane_20260527_221658`
- `runs/p4_benchmark/T66_statcalib_sensitivity_20260529_210906`
- `runs/p4_benchmark/T67_statcalib_teacher_anchor_20260601_225718`
- current mainline benchmark runner behavior without semantic edits

## Fixed Boundary

- Branch: clean committed `main`
- Evidence scope: mock-backed software-HIL only
- Historical anchor: `T24` remains the authoritative frozen ranked table
- Extension-lane rule: `statcalib` remains separately labeled and appended outside the historical frozen ranked set
- Fairness rule: `--paired-seeds` must remain enabled
- Repeat rule: `repeats=2`

## Locked Generated-Only Robustness Matrix

### Frozen scenarios

- `static_bias_theta`
- `linear_ramp`
- `step_sigma_theta`
- `periodic_drift`

### Frozen anchor modes

- `ukf`
- `hybrid_residual_b`

### Predeclared statcalib candidate variants

Use exactly these eight statcalib lanes:

1. `statcalib_window_variance_t001`
   - `teacher_mode = window_variance`
   - `residual_scale_b = 0.50`
   - `residual_clip_b = 0.08`
   - `signal_threshold = 0.001`
2. `statcalib_window_variance_t003`
   - `teacher_mode = window_variance`
   - `residual_scale_b = 0.50`
   - `residual_clip_b = 0.08`
   - `signal_threshold = 0.003`
3. `statcalib_window_variance_t005`
   - `teacher_mode = window_variance`
   - `residual_scale_b = 0.50`
   - `residual_clip_b = 0.08`
   - `signal_threshold = 0.005`
4. `statcalib_window_variance_t010`
   - `teacher_mode = window_variance`
   - `residual_scale_b = 0.50`
   - `residual_clip_b = 0.08`
   - `signal_threshold = 0.010`
5. `statcalib_ekf_t001`
   - `teacher_mode = ekf`
   - `residual_scale_b = 0.50`
   - `residual_clip_b = 0.08`
   - `signal_threshold = 0.001`
6. `statcalib_ekf_t003`
   - `teacher_mode = ekf`
   - `residual_scale_b = 0.50`
   - `residual_clip_b = 0.08`
   - `signal_threshold = 0.003`
7. `statcalib_ekf_t005`
   - `teacher_mode = ekf`
   - `residual_scale_b = 0.50`
   - `residual_clip_b = 0.08`
   - `signal_threshold = 0.005`
8. `statcalib_ekf_t010`
   - `teacher_mode = ekf`
   - `residual_scale_b = 0.50`
   - `residual_clip_b = 0.08`
   - `signal_threshold = 0.010`

### Total matrix

- `4 scenarios x 10 modes x 2 repeats = 80 repeat-runs`

## Implementation Requirements

1. Add one task-scoped derived config that expresses the full T68 matrix without changing historical configs.
2. Launch from a clean committed `main` worktree and record that provenance clearly in task-local docs.
3. Preferred execution shape:
   - one full-matrix invocation under one fixed T68 run root
4. Use a host-launch shape that is expected to outlive the local foreground shell timeout. Do not intentionally rely on killing and relaunching the identical full matrix against the same run root.
5. If an interruption still occurs, the only allowed continuation shape is:
   - same run root
   - same frozen mode/scenario matrix
   - split by repeat range only via `--repeat-start` / `--repeat-stop` / `--resume-only`
6. Mode-chunking is forbidden.
7. Scenario-chunking is forbidden.
8. The summary helper must compute at least:
   - per-scenario best statcalib candidate
   - per-candidate mean LER
   - per-candidate worst-scenario LER and best-scenario LER
   - per-candidate `generated` row count and `mixed` row count
   - per-candidate count of scenarios where the candidate beats both frozen anchors
   - whether any candidate is generated-only across all four scenarios and also beats both frozen anchors across all four scenarios
   - grouped comparison of `window_variance` vs `ekf` at each threshold
   - grouped comparison of threshold ranking within each teacher anchor
   - grouped comparison of mean-best candidate versus worst-case-best candidate
   - a generated-only Pareto-style summary over mean LER and generated-row count
   - threshold-to-threshold monotonicity notes within each teacher anchor, if no monotonic ranking exists
   - an explicit tie representation if mean LER equality occurs
   - the closest near-miss candidate if no full generated-only winner exists
9. The summary helper and tests must stay task-scoped. Do not refactor mainline runner semantics inside T68.

## Expected Output Artifacts

Worker must produce:

- one run root:
  - `runs/p4_benchmark/T68_statcalib_generated_only_*`
- one benchmark report:
  - `docs/statcalib_generated_only_robustness_bounded_benchmark.md`
- one review file:
  - `docs/review/T68_review.md`
- one human explanation:
  - `docs/for_human/T68_explanation.md`
- one worker summary:
  - `docs/worker_summary/T68_worker_summary.md`

## Verification

Worker must run and report:

1. benchmark execution command(s) used for the T68 matrix
2. `python -m py_compile cnn_fpga/benchmark/summarize_statcalib_generated_only.py`
3. `python -m unittest tests.test_statcalib_generated_only_summary`
4. one invocation of the new summary helper against the final T68 run root
5. explicit confirmation that:
   - launch commit == finish commit == `summary.json git_commit`
   - exactly one T68 run root was created
   - no historical `runs/` file was modified
   - `missing_runs == []`
   - every comparison row has `coverage = 1.0` and `completed_repeats = 2`
6. explicit reporting of:
   - generated row count per candidate
   - mixed row count per candidate
   - whether any candidate is full generated-only and still beats both frozen anchors in all four scenarios
   - whether the mean-best candidate and the worst-case-best candidate are the same or different

## Review No-Go Triggers

Reviewer should return `BLOCK` if any of the following happen:

1. more than one T68 run root is created
2. worker changes statcalib/runtime/main runner semantics
3. worker chunks by mode or scenario instead of repeat range
4. worker relaunches the identical full matrix against the same run root after interruption instead of using repeat-range continuation
5. historical `T24`, `T64`, `T65`, `T66`, or `T67` artifacts are modified or rewritten
6. report language upgrades the evidence to `.tflite`, real-board, mature calibration comparator, or paper-grade expanded benchmark
7. provenance is not clean and single-commit anchored
8. task output silently rewrites the historical frozen ranked table rather than treating statcalib as a separate extension lane
9. worker widens the candidate set beyond the predeclared two teachers and four thresholds

## Captain Notes

- This task is intentionally stronger than a simple follow-up, but it remains bounded.
- The point is to answer the generated-only robustness question honestly, not to build a generic statcalib optimization framework by stealth.
- If T68 shows that no generated-only candidate exists inside this bounded grid, that is still a successful task outcome as long as the result is reported honestly.
- Keep the theory-only branch completely separate.

## Worker Output

Worker must report:

1. what files were changed
2. exact benchmark command(s) executed
3. run root path
4. verification command outputs
5. explicit provenance summary
6. scenario-by-scenario outcome summary
7. grouped teacher/threshold outcome summary
8. whether any full generated-only winner exists
9. whether the mean-best candidate and the worst-case-best candidate are the same or different
10. remaining risks or interpretation limits
