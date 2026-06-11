# T66 Statcalib Sensitivity Bounded Benchmark

## Verdict

`T66` completed one bounded `FR8` sensitivity package under a single T66 run root:

- run root: `runs/p4_benchmark/T66_statcalib_sensitivity_20260529_210906`
- evidence scope: mock-backed software-HIL only
- historical anchor: `T24` remains the authoritative frozen ranked table
- comparator boundary: `statcalib` remains a separately labeled extension lane only
- launch branch: `main`
- launch `HEAD`: `ad981bb`
- finish branch: `main`
- finish `HEAD`: `ad981bb`
- `summary.json git_commit`: `ad981bb`

The bounded answer is positive but narrow:

1. the `T64` statcalib advantage survives this predeclared five-point local grid
2. the best per-scenario statcalib variant beats both `ukf` and `hybrid_residual_b` in all four locked scenarios
3. the best aggregate statcalib variant also still beats both frozen anchors in all four locked scenarios

This report does not upgrade the evidence into `.tflite`, real-board, mature calibration-comparator, or paper-grade expanded benchmark claims.

## Preflight And Config Lineage

Clean-main preflight was observed before any T66 repo edits:

- launch timestamp: `2026-05-29 21:09:06 +08:00`
- `git branch --show-current`: `main`
- `git rev-parse --short HEAD`: `ad981bb`
- `git status --short`: no repo status entries; only global ignore warnings for `C:\Users\26410/.config/git/ignore`

Execution used a task-local launch config:

- launch-plan config path: `C:\Users\26410\AppData\Local\Temp\t66_statcalib_sensitivity_20260529_210906.yaml`
- repo-preserved task config: `cnn_fpga/config/p4_multiscenario_statcalib_sensitivity.yaml`
- text-equivalence check: `same_text=True` after normalizing only the `base_config` path from repo-relative to absolute

This split was intentional. The benchmark was launched from a clean committed `main` worktree first, then the identical task-scoped config was written into the repo so the T66 matrix is preserved under the allowed path.

## Exact Benchmark Commands

Initial foreground launch:

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config C:\Users\26410\AppData\Local\Temp\t66_statcalib_sensitivity_20260529_210906.yaml --scenario static_bias_theta --scenario linear_ramp --scenario step_sigma_theta --scenario periodic_drift --mode ukf --mode hybrid_residual_b --mode statcalib_default --mode statcalib_low_scale --mode statcalib_high_scale --mode statcalib_low_clip --mode statcalib_high_threshold --paired-seeds --repeats 2 --run-dir runs/p4_benchmark/T66_statcalib_sensitivity_20260529_210906
```

Foreground shell timeout occurred after two hours, while the run was still inside the same full matrix. To finish the same matrix under the same run root, the identical command was relaunched in the background with `Start-Process`, still without mode-chunking or scenario-chunking and still without changing code or semantics.

Background continuation launcher:

```powershell
Start-Process -FilePath 'C:\ProgramData\anaconda3\python.exe' -ArgumentList '-m','cnn_fpga.benchmark.run_p4_multiscenario_benchmark','--config','C:\Users\26410\AppData\Local\Temp\t66_statcalib_sensitivity_20260529_210906.yaml','--scenario','static_bias_theta','--scenario','linear_ramp','--scenario','step_sigma_theta','--scenario','periodic_drift','--mode','ukf','--mode','hybrid_residual_b','--mode','statcalib_default','--mode','statcalib_low_scale','--mode','statcalib_high_scale','--mode','statcalib_low_clip','--mode','statcalib_high_threshold','--paired-seeds','--repeats','2','--run-dir','runs/p4_benchmark/T66_statcalib_sensitivity_20260529_210906'
```

## Coverage And Integrity

Artifact checks from the final run root:

- `summary.json["missing_runs"] = []`
- comparison rows: `28`
- raw rows: `56`
- all comparison rows have `coverage=1.0`
- all comparison rows have `completed_repeats=2`
- exactly one T66 run root exists under `runs/p4_benchmark/`

Summary helper output was written to:

- `runs/p4_benchmark/T66_statcalib_sensitivity_20260529_210906/statcalib_sensitivity_summary/summary.json`
- `runs/p4_benchmark/T66_statcalib_sensitivity_20260529_210906/statcalib_sensitivity_summary/scenario_summary.csv`
- `runs/p4_benchmark/T66_statcalib_sensitivity_20260529_210906/statcalib_sensitivity_summary/mode_summary.csv`

Historical run roots were not modified during T66. Their latest observed file write times remain earlier than this T66 execution:

- `T24`: `2026-05-11 15:51:14 +0800`
- `T64`: `2026-05-29 12:01:16 +0800`
- `T61`: `2026-05-27 10:22:12 +0800`
- `T62`: `2026-05-27 16:18:55 +0800`

## Execution-Shape Warning

`progress.jsonl` is not perfectly clean:

- `running` records: `57`
- `completed` records: `56`
- duplicate `running` key: `static_bias_theta/statcalib_default/repeat_01`

This happened because the foreground shell timed out before that repeat had written its completed payload, and the identical full-matrix command was relaunched against the same run root. The final benchmark outputs are complete and single-commit anchored, but the progress log retains this one duplicate `running` marker. That warning should travel with any reuse of the T66 provenance story.

## Scenario Summary

| Scenario | Best Statcalib Variant | Best LER Mean | Gap vs UKF | Gap vs Hybrid Residual-B | Best Status | Ranking |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `static_bias_theta` | `statcalib_high_threshold` | `0.431684` | `0.394934` | `0.379889` | `mixed` | `high_threshold > default > high_scale > low_scale > low_clip` |
| `linear_ramp` | `statcalib_default` | `0.467094` | `0.340788` | `0.321178` | `generated` | `default > high_threshold > high_scale > low_clip > low_scale` |
| `step_sigma_theta` | `statcalib_default` | `0.458834` | `0.356466` | `0.328883` | `generated` | `default > high_threshold > high_scale > low_clip > low_scale` |
| `periodic_drift` | `statcalib_default` | `0.439352` | `0.383509` | `0.367232` | `generated` | `default > high_threshold > high_scale > low_scale > low_clip` |

Interpretation:

1. `statcalib_default` wins three of four scenarios
2. `statcalib_high_threshold` wins one of four scenarios
3. every scenario-best statcalib variant still beats both frozen anchors

## Variant Aggregate Ranking

Across the four locked scenarios, the five statcalib variants rank as follows:

| Variant | Mean LER Mean | Mean Rank Within Statcalib | Scenario Wins |
| --- | ---: | ---: | ---: |
| `statcalib_high_threshold` | `0.449241` | `1.75` | `1` |
| `statcalib_default` | `0.449254` | `1.25` | `3` |
| `statcalib_high_scale` | `0.456477` | `3.00` | `0` |
| `statcalib_low_clip` | `0.481484` | `4.50` | `0` |
| `statcalib_low_scale` | `0.485129` | `4.50` | `0` |

Two different notions of "best" both matter:

1. by average LER, `statcalib_high_threshold` is numerically first by a very small margin
2. by scenario wins and mean within-statcalib rank, `statcalib_default` is the more stable point inside this bounded grid

The globally best aggregate variant, `statcalib_high_threshold`, still beats both `ukf` and `hybrid_residual_b` in all four locked scenarios:

- `static_bias_theta`: `0.431684` vs `ukf 0.826618`, `hybrid 0.811573`
- `linear_ramp`: `0.467094` vs `ukf 0.807882`, `hybrid 0.788272`
- `step_sigma_theta`: `0.458834` vs `ukf 0.815300`, `hybrid 0.787717`
- `periodic_drift`: `0.439352` vs `ukf 0.822861`, `hybrid 0.806584`

## Interpretation Limits

What T66 supports:

1. the bounded T64 win is not a single-point fluke inside this five-point local grid
2. the extension-lane result is robust enough to survive small residual-scale / clip / threshold perturbations
3. the frozen T24 table still stays separate and untouched

What T66 does not support:

1. `.tflite` runtime validation
2. real-board validation
3. a rewrite of the historical frozen ranked table
4. a mature calibration-comparator claim
5. paper-grade expanded benchmark evidence by itself

The narrow honest conclusion is: within this locked four-scenario, two-anchor, five-variant statcalib sensitivity grid, the extension-lane advantage persists, but provenance reuse should keep the duplicate-running warning and the mock-backed software-HIL boundary attached.
