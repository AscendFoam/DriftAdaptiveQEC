# T66: FR8 statcalib sensitivity bounded benchmark

## Status

- Proposed by Captain on `2026-05-29`
- Current phase: `Phase 2: Controlled Development`
- Decision state: `Go`
- Task type: bounded benchmark execution + summary task on the mainline experiment branch

## Why This Task Exists

`T65` closed the `T64` report/artifact consistency gap and made the existing bounded FR8 extension-lane result pack self-audited.

That means the next unresolved mainline question is no longer editorial. It is substantive:

1. does the `T64` statcalib win survive a small predeclared local sensitivity grid
2. or is it only a narrow heuristic point created by one very specific parameter setting

The next smallest honest step is therefore not more prose and not a deployment claim. It is one bounded sensitivity benchmark that:

1. keeps the historical `T24` frozen ranked table authoritative
2. keeps `statcalib` as a separately labeled extension lane only
3. probes only a small predeclared grid around the current statcalib heuristic
4. preserves clean provenance and paired-seed fairness
5. produces one reusable task-scoped summary pack

## Goal

Produce one bounded sensitivity package that answers:

1. whether the `T64` statcalib advantage persists across a small predeclared heuristic grid
2. whether the best-performing statcalib point still clears the frozen mainline anchors under the locked four-scenario protocol
3. whether the result can be summarized honestly without rewriting `T24` or upgrading evidence beyond mock-backed software-HIL scope

## Allowed Files

Worker may modify only:

- `docs/tasks/Phase2/T66_fr8_statcalib_sensitivity_bounded_benchmark.md`
- `docs/statcalib_sensitivity_bounded_benchmark.md`
- `docs/review/T66_review.md`
- `docs/for_human/T66_explanation.md`
- `docs/worker_summary/T66_worker_summary.md`
- `cnn_fpga/config/p4_multiscenario_statcalib_sensitivity.yaml`
- `cnn_fpga/benchmark/summarize_statcalib_sensitivity.py`
- `tests/test_statcalib_sensitivity_summary.py`
- exactly one new run root under `runs/p4_benchmark/T66_statcalib_sensitivity_*`

## Docs To Update

Worker must update:

- `docs/statcalib_sensitivity_bounded_benchmark.md`
- `docs/review/T66_review.md`
- `docs/for_human/T66_explanation.md`
- `docs/worker_summary/T66_worker_summary.md`

Worker must not update governance docs. Captain will do that after review.

## Forbidden Scope

Worker must not:

- modify `docs/02_experiment_plan.md`
- modify any governance doc under `docs/00_*` to `docs/08_*`
- modify `cnn_fpga/decoder/statcalib.py`
- modify `cnn_fpga/runtime/slow_loop_runtime.py`
- modify `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py`
- modify historical config files such as `cnn_fpga/config/p4_multiscenario_strong_baselines.yaml` or `cnn_fpga/config/p4_multiscenario_statcalib_extension_lane.yaml`
- modify any historical file under `runs/`
- create more than one `T66` run root
- change benchmark semantics, comparator semantics, or runtime semantics
- widen into `.tflite`, real-board, training, cleanup, or theory-only branch work
- rewrite `T24`, `T64`, or `T65` into mature calibration-comparator, deployment, or paper-grade evidence

## Required Inputs

Worker must reuse:

- `docs/P4_benchmark_formal_protocol.md`
- `docs/fr8_statcalib_extension_lane_benchmark.md`
- `docs/fr8_statcalib_extension_lane_consistency_audit.md`
- `docs/review/T64_review.md`
- `docs/review/T65_review.md`
- `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743`
- `runs/p4_benchmark/T64_fr8_statcalib_extension_lane_20260527_221658`
- current mainline benchmark runner behavior without semantic edits

## Fixed Boundary

- Branch: clean committed `main`
- Evidence scope: mock-backed software-HIL only
- Historical anchor: `T24` remains the authoritative frozen ranked table
- Extension-lane rule: `statcalib` remains separately labeled and appended outside the historical frozen ranked set
- Fairness rule: `--paired-seeds` must remain enabled
- Repeat rule: `repeats=2`

## Locked Sensitivity Matrix

### Frozen scenarios

- `static_bias_theta`
- `linear_ramp`
- `step_sigma_theta`
- `periodic_drift`

### Frozen anchor modes

- `ukf`
- `hybrid_residual_b`

### Predeclared statcalib variants

Use exactly these five statcalib lanes:

1. `statcalib_default`
   - `residual_scale_b = 0.50`
   - `residual_clip_b = 0.08`
   - `signal_threshold = 0.001`
2. `statcalib_low_scale`
   - `residual_scale_b = 0.25`
   - `residual_clip_b = 0.08`
   - `signal_threshold = 0.001`
3. `statcalib_high_scale`
   - `residual_scale_b = 1.00`
   - `residual_clip_b = 0.08`
   - `signal_threshold = 0.001`
4. `statcalib_low_clip`
   - `residual_scale_b = 0.50`
   - `residual_clip_b = 0.04`
   - `signal_threshold = 0.001`
5. `statcalib_high_threshold`
   - `residual_scale_b = 0.50`
   - `residual_clip_b = 0.08`
   - `signal_threshold = 0.010`

### Total matrix

- `4 scenarios x 7 modes x 2 repeats = 56 repeat-runs`

## Implementation Requirements

1. Add one task-scoped derived config that expresses the full T66 matrix without changing historical configs.
2. Launch from a clean committed `main` worktree and record that provenance clearly in task-local docs.
3. Preferred execution shape:
   - one full-matrix invocation under one fixed T66 run root
4. If chunking is necessary, the only allowed chunking shape is:
   - same run root
   - same frozen mode/scenario matrix
   - split by repeat range only via `--repeat-start` / `--repeat-stop` / `--resume-only`
5. Mode-chunking is forbidden.
6. Scenario-chunking is forbidden.
7. The summary helper must compute at least:
   - per-scenario winner
   - per-mode logical error rates
   - gap versus `ukf`
   - gap versus `hybrid_residual_b`
   - ranking among statcalib variants
   - any generated-window or signal-norm columns already emitted by the current runner, if present
8. The summary helper and tests must stay task-scoped. Do not refactor mainline runner semantics inside T66.

## Expected Output Artifacts

Worker must produce:

- one run root:
  - `runs/p4_benchmark/T66_statcalib_sensitivity_*`
- one benchmark report:
  - `docs/statcalib_sensitivity_bounded_benchmark.md`
- one review file:
  - `docs/review/T66_review.md`
- one human explanation:
  - `docs/for_human/T66_explanation.md`
- one worker summary:
  - `docs/worker_summary/T66_worker_summary.md`

## Verification

Worker must run and report:

1. benchmark execution command(s) used for the T66 matrix
2. `python -m py_compile cnn_fpga/benchmark/summarize_statcalib_sensitivity.py`
3. `python -m unittest tests.test_statcalib_sensitivity_summary`
4. one invocation of the new summary helper against the final T66 run root
5. explicit confirmation that:
   - launch commit == finish commit == `summary.json git_commit`
   - exactly one T66 run root was created
   - no historical `runs/` file was modified

## Review No-Go Triggers

Reviewer should return `BLOCK` if any of the following happen:

1. more than one T66 run root is created
2. worker changes statcalib/runtime/main runner semantics
3. worker chunks by mode or scenario instead of repeat range
4. historical `T24`, `T64`, or `T65` artifacts are modified or rewritten
5. report language upgrades the evidence to `.tflite`, real-board, mature calibration comparator, or paper-grade expanded benchmark
6. provenance is not clean and single-commit anchored
7. task output silently rewrites the historical frozen ranked table rather than treating statcalib as a separate extension lane

## Captain Notes

- This task is intentionally stronger than a trivial docs follow-up, but it remains bounded.
- The point is to probe robustness honestly, not to win by moving semantics.
- If T66 shows that the T64 win is fragile, that is still a successful task outcome as long as the result is reported honestly.
- Keep the theory-only branch completely separate.

## Worker Output

Worker must report:

1. what files were changed
2. exact benchmark command(s) executed
3. run root path
4. verification command outputs
5. scenario-by-scenario outcome summary
6. whether the best statcalib variant beats `ukf`
7. whether the best statcalib variant beats `hybrid_residual_b`
8. remaining risks or interpretation limits
