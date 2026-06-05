# T67: FR8 statcalib teacher-anchor dependence bounded benchmark

## Status

- Proposed by Captain on `2026-06-01`
- Current phase: `Phase 2: Controlled Development`
- Decision state: `Go`
- Task type: bounded benchmark execution + grouped summary task on the mainline experiment branch

## Why This Task Exists

`T66` already answered the local-parameter question honestly: the `T64` statcalib win survives one small predeclared five-point sensitivity grid under clean provenance.

That means the next unresolved mainline question is no longer local heuristic fragility. It is teacher-anchor dependence:

1. does the extension-lane gain depend critically on `teacher_mode=ukf`
2. or do the strongest T66 statcalib points remain competitive when the teacher anchor changes

The next smallest honest step is therefore not more prose and not deployment work. It is one bounded teacher-anchor dependence benchmark that:

1. keeps the historical `T24` frozen ranked table authoritative
2. keeps `statcalib` as a separately labeled extension lane only
3. reuses only the two strongest T66 parameter points
4. changes only the teacher anchor inside predeclared statcalib lanes
5. preserves clean provenance and paired-seed fairness
6. produces one reusable grouped summary pack

## Goal

Produce one bounded teacher-anchor package that answers:

1. whether the strongest T66 statcalib points remain competitive when `teacher_mode` changes
2. whether any non-`ukf` teacher-anchor statcalib lane still beats both frozen anchors under the locked four-scenario protocol
3. whether the remaining `R24` concern is mostly teacher-anchor dependence or something narrower

## Allowed Files

Worker may modify only:

- `docs/tasks/Phase2/T67_fr8_statcalib_teacher_anchor_dependence_bounded_benchmark.md`
- `docs/statcalib_teacher_anchor_bounded_benchmark.md`
- `docs/review/T67_review.md`
- `docs/for_human/T67_explanation.md`
- `docs/worker_summary/T67_worker_summary.md`
- `cnn_fpga/config/p4_multiscenario_statcalib_teacher_anchor.yaml`
- `cnn_fpga/benchmark/summarize_statcalib_teacher_anchor.py`
- `tests/test_statcalib_teacher_anchor_summary.py`
- exactly one new run root under `runs/p4_benchmark/T67_statcalib_teacher_anchor_*`

## Docs To Update

Worker must update:

- `docs/statcalib_teacher_anchor_bounded_benchmark.md`
- `docs/review/T67_review.md`
- `docs/for_human/T67_explanation.md`
- `docs/worker_summary/T67_worker_summary.md`

Worker must not update governance docs. Captain will do that after review.

## Forbidden Scope

Worker must not:

- modify `docs/02_experiment_plan.md`
- modify any governance doc under `docs/00_*` to `docs/08_*`
- modify `cnn_fpga/decoder/statcalib.py`
- modify `cnn_fpga/runtime/slow_loop_runtime.py`
- modify `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py`
- modify historical config files such as `cnn_fpga/config/p4_multiscenario_strong_baselines.yaml`, `cnn_fpga/config/p4_multiscenario_statcalib_extension_lane.yaml`, or `cnn_fpga/config/p4_multiscenario_statcalib_sensitivity.yaml`
- modify any historical file under `runs/`
- create more than one `T67` run root
- change benchmark semantics, comparator semantics, or runtime semantics
- widen into `.tflite`, real-board, training, cleanup, or theory-only branch work
- rewrite `T24`, `T64`, `T65`, or `T66` into mature calibration-comparator, deployment, or paper-grade evidence

## Required Inputs

Worker must reuse:

- `docs/P4_benchmark_formal_protocol.md`
- `docs/fr8_statcalib_extension_lane_benchmark.md`
- `docs/fr8_statcalib_extension_lane_consistency_audit.md`
- `docs/statcalib_sensitivity_bounded_benchmark.md`
- `docs/review/T64_review.md`
- `docs/review/T65_review.md`
- `docs/review/T66_review.md`
- `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743`
- `runs/p4_benchmark/T64_fr8_statcalib_extension_lane_20260527_221658`
- `runs/p4_benchmark/T66_statcalib_sensitivity_20260529_210906`
- current mainline benchmark runner behavior without semantic edits

## Fixed Boundary

- Branch: clean committed `main`
- Evidence scope: mock-backed software-HIL only
- Historical anchor: `T24` remains the authoritative frozen ranked table
- Extension-lane rule: `statcalib` remains separately labeled and appended outside the historical frozen ranked set
- Fairness rule: `--paired-seeds` must remain enabled
- Repeat rule: `repeats=2`

## Locked Teacher-Anchor Matrix

### Frozen scenarios

- `static_bias_theta`
- `linear_ramp`
- `step_sigma_theta`
- `periodic_drift`

### Frozen anchor modes

- `ukf`
- `hybrid_residual_b`

### Predeclared statcalib teacher-anchor variants

Use exactly these six statcalib lanes:

1. `statcalib_default_teacher_ukf`
   - `teacher_mode = ukf`
   - `residual_scale_b = 0.50`
   - `residual_clip_b = 0.08`
   - `signal_threshold = 0.001`
2. `statcalib_default_teacher_window_variance`
   - `teacher_mode = window_variance`
   - `residual_scale_b = 0.50`
   - `residual_clip_b = 0.08`
   - `signal_threshold = 0.001`
3. `statcalib_default_teacher_ekf`
   - `teacher_mode = ekf`
   - `residual_scale_b = 0.50`
   - `residual_clip_b = 0.08`
   - `signal_threshold = 0.001`
4. `statcalib_high_threshold_teacher_ukf`
   - `teacher_mode = ukf`
   - `residual_scale_b = 0.50`
   - `residual_clip_b = 0.08`
   - `signal_threshold = 0.010`
5. `statcalib_high_threshold_teacher_window_variance`
   - `teacher_mode = window_variance`
   - `residual_scale_b = 0.50`
   - `residual_clip_b = 0.08`
   - `signal_threshold = 0.010`
6. `statcalib_high_threshold_teacher_ekf`
   - `teacher_mode = ekf`
   - `residual_scale_b = 0.50`
   - `residual_clip_b = 0.08`
   - `signal_threshold = 0.010`

### Total matrix

- `4 scenarios x 8 modes x 2 repeats = 64 repeat-runs`

## Implementation Requirements

1. Add one task-scoped derived config that expresses the full T67 matrix without changing historical configs.
2. Launch from a clean committed `main` worktree and record that provenance clearly in task-local docs.
3. Preferred execution shape:
   - one full-matrix invocation under one fixed T67 run root
4. Use a host-launch shape that is expected to outlive the local foreground shell timeout. Do not intentionally rely on killing and relaunching the identical full matrix against the same run root.
5. If an interruption still occurs, the only allowed continuation shape is:
   - same run root
   - same frozen mode/scenario matrix
   - split by repeat range only via `--repeat-start` / `--repeat-stop` / `--resume-only`
6. Mode-chunking is forbidden.
7. Scenario-chunking is forbidden.
8. The summary helper must compute at least:
   - per-scenario best statcalib variant
   - per-anchor-mode gaps versus `ukf`
   - per-anchor-mode gaps versus `hybrid_residual_b`
   - grouped comparison of `default` vs `high_threshold` within each teacher anchor
   - grouped comparison of `ukf` vs `window_variance` vs `ekf` within each parameter point
   - whether any non-`ukf` teacher variant still beats both frozen anchors in all four scenarios
   - any generated-window or signal-norm columns already emitted by the current runner, if present
9. The summary helper and tests must stay task-scoped. Do not refactor mainline runner semantics inside T67.

## Expected Output Artifacts

Worker must produce:

- one run root:
  - `runs/p4_benchmark/T67_statcalib_teacher_anchor_*`
- one benchmark report:
  - `docs/statcalib_teacher_anchor_bounded_benchmark.md`
- one review file:
  - `docs/review/T67_review.md`
- one human explanation:
  - `docs/for_human/T67_explanation.md`
- one worker summary:
  - `docs/worker_summary/T67_worker_summary.md`

## Verification

Worker must run and report:

1. benchmark execution command(s) used for the T67 matrix
2. `python -m py_compile cnn_fpga/benchmark/summarize_statcalib_teacher_anchor.py`
3. `python -m unittest tests.test_statcalib_teacher_anchor_summary`
4. one invocation of the new summary helper against the final T67 run root
5. explicit confirmation that:
   - launch commit == finish commit == `summary.json git_commit`
   - exactly one T67 run root was created
   - no historical `runs/` file was modified

## Review No-Go Triggers

Reviewer should return `BLOCK` if any of the following happen:

1. more than one T67 run root is created
2. worker changes statcalib/runtime/main runner semantics
3. worker chunks by mode or scenario instead of repeat range
4. worker relaunches the identical full matrix against the same run root after interruption instead of using repeat-range continuation
5. historical `T24`, `T64`, `T65`, or `T66` artifacts are modified or rewritten
6. report language upgrades the evidence to `.tflite`, real-board, mature calibration comparator, or paper-grade expanded benchmark
7. provenance is not clean and single-commit anchored
8. task output silently rewrites the historical frozen ranked table rather than treating statcalib as a separate extension lane

## Captain Notes

- This task is intentionally stronger than a simple follow-up, but it remains bounded.
- The point is to probe teacher-anchor dependence honestly, not to build a generic calibration framework by stealth.
- If T67 shows that the current win is strongly `ukf`-teacher-dependent, that is still a successful task outcome as long as the result is reported honestly.
- Keep the theory-only branch completely separate.

## Worker Output

Worker must report:

1. what files were changed
2. exact benchmark command(s) executed
3. run root path
4. verification command outputs
5. scenario-by-scenario outcome summary
6. grouped teacher-anchor outcome summary
7. whether any non-`ukf` teacher variant beats `ukf`
8. whether any non-`ukf` teacher variant beats `hybrid_residual_b`
9. remaining risks or interpretation limits

### Worker Output

- Changed files:
  - `cnn_fpga/config/p4_multiscenario_statcalib_teacher_anchor.yaml`
  - `cnn_fpga/benchmark/summarize_statcalib_teacher_anchor.py`
  - `tests/test_statcalib_teacher_anchor_summary.py`
  - `docs/statcalib_teacher_anchor_bounded_benchmark.md`
  - `docs/review/T67_review.md`
  - `docs/for_human/T67_explanation.md`
  - `docs/worker_summary/T67_worker_summary.md`

- Exact benchmark command executed inside the detached host launch:

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config C:\t67cfg_20260601_225718.yaml --scenario static_bias_theta --scenario linear_ramp --scenario step_sigma_theta --scenario periodic_drift --mode ukf --mode hybrid_residual_b --mode statcalib_default_teacher_ukf --mode statcalib_default_teacher_window_variance --mode statcalib_default_teacher_ekf --mode statcalib_high_threshold_teacher_ukf --mode statcalib_high_threshold_teacher_window_variance --mode statcalib_high_threshold_teacher_ekf --paired-seeds --repeats 2 --run-dir D:\Codes\Quantum\DriftAdaptiveQEC\runs\p4_benchmark\T67_statcalib_teacher_anchor_20260601_225718
```

- Run root:
  - `runs/p4_benchmark/T67_statcalib_teacher_anchor_20260601_225718`

- Verification outputs:
  - `python -m py_compile cnn_fpga/benchmark/summarize_statcalib_teacher_anchor.py`: pass
  - `python -m unittest tests.test_statcalib_teacher_anchor_summary`: `Ran 6 tests`, `OK`
  - `python -m cnn_fpga.benchmark.summarize_statcalib_teacher_anchor --run-dir runs/p4_benchmark/T67_statcalib_teacher_anchor_20260601_225718`: pass
  - provenance closure:
    - launch `HEAD = 84f4468`
    - finish `HEAD = 84f4468`
    - `summary.json git_commit = 84f4468`
  - integrity:
    - `comparison.csv` rows = `32`
    - `missing_runs = []`
    - all comparison rows have `coverage=1.0`, `completed_repeats=2`
    - `progress.jsonl`: `running=64`, `completed=64`, duplicate `running=0`
  - preservation:
    - exactly one `T67` run root exists
    - `T24/T64/T66` historical run-root last-write times did not change

- Scenario-by-scenario outcome summary:
  - `static_bias_theta`: best = `statcalib_high_threshold_teacher_window_variance`, LER `0.430249`, gap vs `ukf` `0.396432`, gap vs `hybrid_residual_b` `0.381506`, status `mixed`
  - `linear_ramp`: best = `statcalib_default_teacher_window_variance`, LER `0.466071`, gap vs `ukf` `0.347155`, gap vs `hybrid_residual_b` `0.322450`, status `generated`
  - `step_sigma_theta`: best = `statcalib_default_teacher_ekf`, LER `0.458070`, gap vs `ukf` `0.354527`, gap vs `hybrid_residual_b` `0.331012`, status `generated`
  - `periodic_drift`: best = `statcalib_default_teacher_window_variance`, LER `0.437656`, gap vs `ukf` `0.384544`, gap vs `hybrid_residual_b` `0.370817`, status `generated`

- Grouped teacher-anchor outcome summary:
  - parameter-point ranking:
    - `default`: `window_variance > ekf > ukf`
    - `high_threshold`: `window_variance > ekf > ukf`
  - teacher-anchor best parameter point:
    - `ukf`: `default` slightly better than `high_threshold`
    - `window_variance`: `high_threshold` slightly better than `default`
    - `ekf`: `default` and `high_threshold` tie on mean LER
  - aggregate variant ranking:
    1. `statcalib_high_threshold_teacher_window_variance`
    2. `statcalib_default_teacher_window_variance`
    3. `statcalib_default_teacher_ekf`
    4. `statcalib_high_threshold_teacher_ekf`
    5. `statcalib_default_teacher_ukf`
    6. `statcalib_high_threshold_teacher_ukf`

- Whether any non-`ukf` teacher variant beats `ukf`:
  - Yes.
  - Within both parameter points, `window_variance` and `ekf` rank ahead of `ukf`.

- Whether any non-`ukf` teacher variant beats `hybrid_residual_b`:
  - Yes.
  - `statcalib_default_teacher_window_variance`
  - `statcalib_default_teacher_ekf`
  - `statcalib_high_threshold_teacher_window_variance`
  - `statcalib_high_threshold_teacher_ekf`
  all beat `hybrid_residual_b` in all four locked scenarios.

- Remaining risks / interpretation limits:
  - evidence remains mock-backed software-HIL only
  - not `.tflite`
  - not real-board
  - does not rewrite `T24`
  - two comparison rows remain `mixed`:
    - `static_bias_theta / statcalib_high_threshold_teacher_window_variance`
    - `step_sigma_theta / statcalib_high_threshold_teacher_ukf`
