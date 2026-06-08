# T68 StatCalib Generated-Only Robustness Bounded Benchmark

## Verdict

`T68` completed one bounded generated-only robustness benchmark under a single T68 run root:

- run root: `runs/p4_benchmark/T68_statcalib_generated_only_20260605_205723`
- evidence scope: mock-backed software-HIL only
- historical anchor: `T24` remains the authoritative frozen ranked table
- comparator boundary: `statcalib` remains a separately labeled extension lane only
- launch branch: `main`
- launch `HEAD`: `bda8f2b`
- finish branch: `main`
- finish `HEAD`: `bda8f2b`
- `summary.json git_commit`: `bda8f2b`

The bounded answer is positive and more specific than `T67`:

1. at least one predeclared non-`ukf` statcalib candidate is fully `generated` across all four locked scenarios and still beats both frozen anchors in all four scenarios
2. the strongest clean result comes from the `window_variance` teacher anchor, not the `ukf` teacher anchor
3. `R24` is therefore no longer blocked by "maybe no generated-only winner exists in this bounded grid"

This report does not upgrade the evidence into `.tflite`, real-board, mature calibration-comparator, or paper-grade expanded benchmark claims.

## Preflight And Acceleration Handling

The active workspace was intentionally left untouched after the T68 task-scoped files were created. To preserve the task-package requirement of launching from clean committed `main`, T68 was executed from a clean short-path clone:

- clean launch clone: `C:\t68cf2b`
- launch timestamp: `2026-06-05 20:57:25 +08:00`
- `git -C C:\t68cf2b branch --show-current`: `main`
- `git -C C:\t68cf2b rev-parse --short HEAD`: `bda8f2b`
- `git -C C:\t68cf2b status --short`: clean

Safe acceleration choice:

1. no mode-chunking
2. no scenario-chunking
3. no runner-semantic edits
4. one detached host launch for the full matrix so the benchmark could outlive foreground shell timeouts without relaunching the same matrix

## Config Lineage And Exact Benchmark Command

Repo-preserved task config:

- `cnn_fpga/config/p4_multiscenario_statcalib_generated_only.yaml`

Launch-time config:

- `runs/p4_benchmark/T68_statcalib_generated_only_20260605_205723/launch_config_from_clean_clone.yaml`

Lineage note:

1. the launch-time config only replaced `base_config` with an absolute clone-local path
2. all task semantics stayed on the repo-preserved T68 matrix

Exact benchmark command executed inside the detached host launch:

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config D:\Codes\Quantum\DriftAdaptiveQEC\runs\p4_benchmark\T68_statcalib_generated_only_20260605_205723\launch_config_from_clean_clone.yaml --scenario static_bias_theta --scenario linear_ramp --scenario step_sigma_theta --scenario periodic_drift --mode ukf --mode hybrid_residual_b --mode statcalib_window_variance_t001 --mode statcalib_window_variance_t003 --mode statcalib_window_variance_t005 --mode statcalib_window_variance_t010 --mode statcalib_ekf_t001 --mode statcalib_ekf_t003 --mode statcalib_ekf_t005 --mode statcalib_ekf_t010 --paired-seeds --repeats 2 --run-dir D:\Codes\Quantum\DriftAdaptiveQEC\runs\p4_benchmark\T68_statcalib_generated_only_20260605_205723
```

Execution shape:

1. one full-matrix invocation
2. one fixed T68 run root
3. no mode-chunking
4. no scenario-chunking
5. no repeat-range continuation
6. no relaunch of the identical full matrix

## Coverage And Provenance Checks

Artifact checks from the final run root:

- `summary.json["missing_runs"] = []`
- comparison rows: `40`
- all comparison rows have `coverage=1.0`
- all comparison rows have `completed_repeats=2`
- `progress.jsonl`: `running=80`, `completed=80`
- duplicate `running` keys: none
- duplicate `completed` keys: none
- exactly one T68-scoped run root exists under `runs/p4_benchmark/`

Provenance closure:

1. launch `HEAD` from clean clone = `bda8f2b`
2. finish `HEAD` in the same clean clone = `bda8f2b`
3. `summary.json["git_commit"]` = `bda8f2b`

Observed historical run-root write times remained on their pre-existing roots:

- `T24`: `2026-05-11 15:51:14 +08:00`
- `T64`: `2026-05-29 12:01:16 +08:00`
- `T66`: `2026-06-01 01:48:11 +08:00`
- `T67`: `2026-06-05 17:06:40 +08:00`

## Scenario-By-Scenario Outcome

| Scenario | Best StatCalib Candidate Set | Teacher Anchor | Best LER Mean | Gap vs UKF | Gap vs Hybrid Residual-B | Best Status |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `static_bias_theta` | `window_variance_t001 = t003 = t005 = t010` | `window_variance` | `0.428758` | `0.398396` | `0.382062` | `generated` |
| `linear_ramp` | `window_variance_t001 = t003 = t005 = t010` | `window_variance` | `0.466279` | `0.347593` | `0.322628` | `generated` |
| `step_sigma_theta` | `window_variance_t001 = t003 = t005` | `window_variance` | `0.456712` | `0.355252` | `0.330789` | `generated` |
| `periodic_drift` | `window_variance_t001 = t003 = t005` | `window_variance` | `0.438623` | `0.386225` | `0.368738` | `generated` |

Scenario interpretation:

1. every scenario-best statcalib candidate beats both frozen anchors
2. every scenario-best set comes from the `window_variance` teacher anchor
3. the winning threshold set is not unique, but the clean winners are concentrated in `t001/t003/t005`

## Candidate Aggregate Summary

| Candidate | Teacher | Threshold | Mean LER Mean | Worst-Scenario LER | Generated Rows | Mixed Rows | Beats Both Frozen Anchors | Full Generated-Only Winner |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `statcalib_window_variance_t001` | `window_variance` | `0.001` | `0.447593` | `0.466279` | `4` | `0` | `4/4` | `True` |
| `statcalib_window_variance_t003` | `window_variance` | `0.003` | `0.447593` | `0.466279` | `4` | `0` | `4/4` | `True` |
| `statcalib_window_variance_t005` | `window_variance` | `0.005` | `0.447593` | `0.466279` | `4` | `0` | `4/4` | `True` |
| `statcalib_window_variance_t010` | `window_variance` | `0.010` | `0.447677` | `0.466279` | `2` | `2` | `4/4` | `False` |
| `statcalib_ekf_t003` | `ekf` | `0.003` | `0.448519` | `0.466760` | `3` | `1` | `4/4` | `False` |
| `statcalib_ekf_t005` | `ekf` | `0.005` | `0.448519` | `0.466760` | `3` | `1` | `4/4` | `False` |
| `statcalib_ekf_t010` | `ekf` | `0.010` | `0.448519` | `0.466760` | `3` | `1` | `4/4` | `False` |
| `statcalib_ekf_t001` | `ekf` | `0.001` | `0.448552` | `0.466760` | `4` | `0` | `4/4` | `True` |

Generated-only conclusion:

1. `statcalib_window_variance_t001`
2. `statcalib_window_variance_t003`
3. `statcalib_window_variance_t005`
4. `statcalib_ekf_t001`

all remain fully `generated` and beat both frozen anchors in all four scenarios.

## Grouped Teacher And Threshold Outcomes

### Threshold-by-threshold teacher comparison

At every threshold in the predeclared grid, `window_variance` beats `ekf` by mean LER:

| Threshold | Window-Variance Mean LER | EKF Mean LER | Winner |
| --- | ---: | ---: | --- |
| `0.001` | `0.447593` | `0.448552` | `window_variance` |
| `0.003` | `0.447593` | `0.448519` | `window_variance` |
| `0.005` | `0.447593` | `0.448519` | `window_variance` |
| `0.010` | `0.447677` | `0.448519` | `window_variance` |

### Threshold ranking within each teacher anchor

- `window_variance`: `t001 = t003 = t005 > t010`
- monotonicity: `monotonic_non_decreasing`

- `ekf`: `t003 = t005 = t010 > t001`
- monotonicity: `monotonic_non_increasing`

### Mean-best vs worst-case-best

- mean-best candidates: `statcalib_window_variance_t001 = statcalib_window_variance_t003 = statcalib_window_variance_t005`
- worst-case-best candidates: `statcalib_window_variance_t001 = statcalib_window_variance_t003 = statcalib_window_variance_t005 = statcalib_window_variance_t010`
- relation: `different`

Interpretation:

1. `window_variance_t010` matches the same worst-case LER as the three clean mean-best candidates
2. but it loses on average LER and carries two `mixed` rows
3. so the cleanest portable answer remains the `window_variance t001/t003/t005` tie set

### Pareto summary

The generated-only Pareto front contains only:

1. `statcalib_window_variance_t001`
2. `statcalib_window_variance_t003`
3. `statcalib_window_variance_t005`

`statcalib_ekf_t001` is also fully generated-only, but it is dominated by the three `window_variance` winners on mean LER at the same generated-row count.

## Residual Mixed Rows

The bounded grid still contains some mixed-provenance candidates:

1. `step_sigma_theta / statcalib_window_variance_t010`
2. `periodic_drift / statcalib_window_variance_t010`
3. `periodic_drift / statcalib_ekf_t003`
4. `periodic_drift / statcalib_ekf_t005`
5. `periodic_drift / statcalib_ekf_t010`

This does not block T68, because the task asked whether any fully generated-only winner exists in the predeclared grid, and the answer is now yes.

## Bounded Interpretation

What T68 supports:

1. the generated-only existence question is answered positively inside the locked grid
2. a non-`ukf` teacher anchor can stay fully generated across all four locked scenarios and still beat both frozen anchors in all four scenarios
3. the strongest clean answer belongs to the `window_variance` teacher anchor, with a three-way tie at `t001/t003/t005`

What T68 does not support:

1. a rewrite of the frozen `T24` ranked table
2. `.tflite` runtime validation
3. real-board validation
4. a mature calibration-comparator claim
5. paper-grade expanded benchmark evidence by itself

The narrow honest conclusion is:

`T68` closes the bounded "maybe no generated-only winner exists" concern. It does not turn the extension lane into a deployment-ready or paper-grade claim, and it still should be reported separately from the frozen `T24` main table.
