# T67 Statcalib Teacher-Anchor Bounded Benchmark

## Verdict

`T67` completed one bounded teacher-anchor dependence benchmark under a single T67 run root:

- run root: `runs/p4_benchmark/T67_statcalib_teacher_anchor_20260601_225718`
- evidence scope: mock-backed software-HIL only
- historical anchor: `T24` remains the authoritative frozen ranked table
- comparator boundary: `statcalib` remains a separately labeled extension lane only
- launch branch: `main`
- launch `HEAD`: `84f4468`
- finish branch: `main`
- finish `HEAD`: `84f4468`
- `summary.json git_commit`: `84f4468`

The bounded answer is clear:

1. the T64/T66 statcalib advantage is not critically dependent on `teacher_mode=ukf`
2. within both predeclared parameter points, non-`ukf` teacher anchors outrank the `ukf` teacher anchor
3. all six statcalib teacher-anchor variants still beat both frozen anchors in all four locked scenarios

This report does not upgrade the evidence into `.tflite`, real-board, mature calibration-comparator, or paper-grade expanded benchmark claims.

## Preflight And Acceleration Handling

The source workspace was not clean because it contained one unrelated user PDF modification outside T67 scope. To satisfy the task-package requirement of launching from clean committed `main` without touching that user change, T67 was launched from a clean short-path clone:

- clean launch clone: `C:\t67c`
- launch timestamp: `2026-06-01 22:57:18 +08:00`
- `git -C C:\t67c branch --show-current`: `main`
- `git -C C:\t67c rev-parse --short HEAD`: `84f4468`
- `git -C C:\t67c status --short`: clean

Acceleration note:

1. true mode/scenario parallelization was not safe inside T67 because the task package forbids mode-chunking, scenario-chunking, and runner-semantic edits
2. the safe acceleration choice was a one-shot detached host launch from the clean clone, so the full matrix could outlive interactive shell timeout without relaunching the same matrix into the same run root

## Config Lineage And Exact Benchmark Command

Repo-preserved task config:

- `cnn_fpga/config/p4_multiscenario_statcalib_teacher_anchor.yaml`

Launch-time temp config:

- `C:\t67cfg_20260601_225718.yaml`

Lineage note:

1. the temp launch config only replaced `base_config` with an absolute clone-local path
2. after normalizing that `base_config` line and trimming trailing blank lines, the temp launch config and the repo-preserved task config are text-equivalent

Exact benchmark command executed inside the detached host launch:

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config C:\t67cfg_20260601_225718.yaml --scenario static_bias_theta --scenario linear_ramp --scenario step_sigma_theta --scenario periodic_drift --mode ukf --mode hybrid_residual_b --mode statcalib_default_teacher_ukf --mode statcalib_default_teacher_window_variance --mode statcalib_default_teacher_ekf --mode statcalib_high_threshold_teacher_ukf --mode statcalib_high_threshold_teacher_window_variance --mode statcalib_high_threshold_teacher_ekf --paired-seeds --repeats 2 --run-dir D:\Codes\Quantum\DriftAdaptiveQEC\runs\p4_benchmark\T67_statcalib_teacher_anchor_20260601_225718
```

Execution shape:

1. one full-matrix invocation
2. one fixed T67 run root
3. no mode-chunking
4. no scenario-chunking
5. no relaunch of the identical full matrix
6. no repeat-range continuation was needed

## Coverage And Provenance Checks

Artifact checks from the final run root:

- `comparison.csv` rows: `32`
- `progress.jsonl` `running` records: `64`
- `progress.jsonl` `completed` records: `64`
- duplicate `running` records for the same `(scenario, mode, repeat)`: none
- `summary.json["missing_runs"] = []`
- all comparison rows have `coverage=1.0`
- all comparison rows have `completed_repeats=2`
- `summary.json["git_commit"] = 84f4468`

Provenance closure:

1. launch `HEAD` from clean clone = `84f4468`
2. finish `HEAD` in the same clean clone = `84f4468`
3. `summary.json["git_commit"]` = `84f4468`

Historical run-root preservation:

- exactly one T67-scoped run root exists under `runs/p4_benchmark/`
- `T24` last write time remains `2026-05-11 15:51:14 +08:00`
- `T64` last write time remains `2026-05-29 12:01:16 +08:00`
- `T66` last write time remains `2026-06-01 01:48:11 +08:00`

## Scenario-by-Scenario Outcome

| Scenario | Best StatCalib Variant | Teacher Anchor | Parameter Point | Best LER Mean | Gap vs UKF | Gap vs Hybrid Residual-B | Status |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| `static_bias_theta` | `statcalib_high_threshold_teacher_window_variance` | `window_variance` | `high_threshold` | `0.430249` | `0.396432` | `0.381506` | `mixed` |
| `linear_ramp` | `statcalib_default_teacher_window_variance` | `window_variance` | `default` | `0.466071` | `0.347155` | `0.322450` | `generated` |
| `step_sigma_theta` | `statcalib_default_teacher_ekf` | `ekf` | `default` | `0.458070` | `0.354527` | `0.331012` | `generated` |
| `periodic_drift` | `statcalib_default_teacher_window_variance` | `window_variance` | `default` | `0.437656` | `0.384544` | `0.370817` | `generated` |

Scenario interpretation:

1. `window_variance` teacher wins `3/4` scenarios
2. `ekf` teacher wins the remaining `step_sigma_theta` scenario
3. `ukf` teacher wins `0/4` scenarios

## Grouped Teacher-Anchor Outcomes

### Parameter-point comparison: `default` vs `high_threshold`

| Parameter Point | Teacher-Anchor Ranking | Best Mode | Best Mean LER | Non-UKF Teacher Best | Non-UKF Variants That Beat Both Frozen Anchors In All 4 Scenarios |
| --- | --- | --- | ---: | --- | --- |
| `default` | `window_variance > ekf > ukf` | `statcalib_default_teacher_window_variance` | `0.448243` | `True` | `statcalib_default_teacher_window_variance`, `statcalib_default_teacher_ekf` |
| `high_threshold` | `window_variance > ekf > ukf` | `statcalib_high_threshold_teacher_window_variance` | `0.448191` | `True` | `statcalib_high_threshold_teacher_window_variance`, `statcalib_high_threshold_teacher_ekf` |

### Teacher-anchor comparison: `default` vs `high_threshold`

| Teacher Anchor | Better Parameter Point By Mean LER | Default Mean LER | High-Threshold Mean LER | Both Variants Beat Both Frozen Anchors In All 4 Scenarios |
| --- | --- | ---: | ---: | --- |
| `ukf` | `default` | `0.449355` | `0.449663` | `True` |
| `window_variance` | `high_threshold` | `0.448243` | `0.448191` | `True` |
| `ekf` | `high_threshold` | `0.448706` | `0.448706` | `True` |

Aggregate ranking across all six teacher-anchor variants:

1. `statcalib_high_threshold_teacher_window_variance` = `0.448191`
2. `statcalib_default_teacher_window_variance` = `0.448243`
3. `statcalib_default_teacher_ekf` = `0.448706`
4. `statcalib_high_threshold_teacher_ekf` = `0.448706`
5. `statcalib_default_teacher_ukf` = `0.449355`
6. `statcalib_high_threshold_teacher_ukf` = `0.449663`

## Bounded Interpretation

What T67 supports:

1. the current bounded statcalib win is not narrowly tied to `teacher_mode=ukf`
2. `window_variance` and `ekf` teacher anchors remain competitive and, within this matrix, both outrank the `ukf` teacher anchor
3. the remaining `R24` concern is therefore narrower than gross teacher-anchor dependence

What T67 does not support:

1. `.tflite` runtime validation
2. real-board validation
3. a rewrite of the historical `T24` frozen ranked table
4. a mature calibration-comparator claim
5. paper-grade expanded benchmark evidence by itself

## Residual Risk And Caveats

Two comparison rows still carry `statcalib_status = mixed`:

1. `static_bias_theta / statcalib_high_threshold_teacher_window_variance`
2. `step_sigma_theta / statcalib_high_threshold_teacher_ukf`

So the strongest aggregate variant is not a fully generated-only result pack. That caveat should travel with any reuse of T67.

The narrow honest conclusion is:

1. teacher-anchor dependence is not the main explanation for the bounded T64/T66 win
2. non-`ukf` teachers remain strong, and in this matrix they are stronger than the `ukf` teacher anchor
3. the evidence still stops at mock-backed software-HIL extension-lane scope
