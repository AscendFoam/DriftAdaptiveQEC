# FR8 Statcalib Extension-Lane Benchmark

## Verdict

`T64` completed one bounded four-scenario extension-lane benchmark without changing the historical `T24` frozen-set evidence.

- launch branch: `main`
- launch `HEAD`: `1e59f24`
- finish branch: `main`
- finish `HEAD`: `1e59f24`
- `summary.json git_commit`: `1e59f24`
- execution shape: one-shot full-matrix run under one fixed T64 run root
- frozen five-mode subset vs `T24`: exact match across all 20 frozen comparison rows

This result pack stays inside the required boundary:

- mock-backed software-HIL only
- bounded `FR8` extension-lane evidence only
- not a rewrite of `T24`
- not `.tflite` validation
- not real-board validation
- not paper-grade expanded benchmark evidence by itself

## Preflight Result

Preflight before launch:

- launch timestamp: `2026-05-27 22:11:07 +08:00`
- `git branch --show-current`: `main`
- `git status --short`: no repo status entries; only global ignore warnings for `C:\Users\26410/.config/git/ignore`
- `git rev-parse --short HEAD`: `1e59f24`

This satisfied the T64 clean committed `main` requirement.

## Config And Command

Derived config used:

- `cnn_fpga/config/p4_multiscenario_statcalib_extension_lane.yaml`

Why a derived config was needed:

- the runner deep-merges lists by replacement, so appending `statcalib` honestly required a task-scoped config that preserved the frozen five-mode order and added `statcalib` only as the sixth mode
- no historical config was modified

The derived config preserves:

- scenarios: `static_bias_theta`, `linear_ramp`, `step_sigma_theta`, `periodic_drift`
- frozen mode order: `ekf`, `ukf`, `constant_residual_mu`, `rls_residual_b`, `hybrid_residual_b`
- extension lane appended last: `statcalib`
- paired seeds and `repeats=2`
- the minimal `statcalib` block already used in `cnn_fpga/config/p4_multiscenario_statcalib_smoke.yaml`

Exact benchmark command:

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config cnn_fpga/config/p4_multiscenario_statcalib_extension_lane.yaml --scenario static_bias_theta --scenario linear_ramp --scenario step_sigma_theta --scenario periodic_drift --mode ekf --mode ukf --mode constant_residual_mu --mode rls_residual_b --mode hybrid_residual_b --mode statcalib --paired-seeds --repeats 2 --run-dir runs/p4_benchmark/T64_fr8_statcalib_extension_lane_20260527_221658
```

Execution note:

- one detached one-shot invocation only
- no repeat-range chunking
- no resume against the same run root

## Run Root And Post-Run Provenance

- run root: `runs/p4_benchmark/T64_fr8_statcalib_extension_lane_20260527_221658`
- finish timestamp from `summary.json`: `2026-05-29 12:01:16 +08:00`
- finish branch: `main`
- finish `HEAD`: `1e59f24`
- `summary.json["git_commit"]`: `1e59f24`
- T64-scoped run-root count under `runs/p4_benchmark/`: `1`

All three provenance anchors match:

1. launch `HEAD`
2. finish `HEAD`
3. `summary.json git_commit`

## Coverage And Integrity

`summary.json` / `progress.jsonl` checks:

- `comparison_rows_count=24`
- `raw_rows_count=48`
- `missing_runs_count=0`
- `paired_seeds=true`
- `repeats=2`
- all 24 comparison rows have `coverage=1.0`
- all 24 comparison rows have `completed_repeats=2`
- `progress.jsonl` has `48` `running` records and `48` `completed` records
- duplicate `running` record for the same `(scenario, mode, repeat)` key: none

Historical run-root rewrite check:

- only one new T64-scoped run root exists
- `T24`, `T59`, `T61`, and `T62` run-root last-write timestamps remained unchanged during this task audit

## Frozen Five-Mode Subset

The frozen subset in `T64` matches `T24` exactly for every frozen `(scenario, mode)` row.

- compared rows: `20`
- max absolute delta in `final_ler_mean`: `0`
- max absolute delta in `overflow_rate_mean`: `0`
- max absolute delta in `coverage`: `0`

| Scenario | EKF | UKF | Constant Residual-Mu | RLS Residual-B | Hybrid Residual-B | Frozen Winner |
| --- | --- | --- | --- | --- | --- | --- |
| `static_bias_theta` | `0.838110 +- 0.000833` | `0.825370 +- 0.000673` | `0.836658 +- 0.000129` | `0.837577 +- 0.000734` | `0.810902 +- 0.001188` | `hybrid_residual_b` |
| `linear_ramp` | `0.819200 +- 0.000170` | `0.811201 +- 0.000868` | `0.816911 +- 0.000124` | `0.819373 +- 0.000092` | `0.787755 +- 0.000439` | `hybrid_residual_b` |
| `step_sigma_theta` | `0.822365 +- 0.000170` | `0.811548 +- 0.000761` | `0.819784 +- 0.000276` | `0.821493 +- 0.000420` | `0.788800 +- 0.001069` | `hybrid_residual_b` |
| `periodic_drift` | `0.832192 +- 0.000558` | `0.821558 +- 0.001885` | `0.829670 +- 0.000345` | `0.832334 +- 0.000040` | `0.806392 +- 0.000289` | `hybrid_residual_b` |

## Statcalib Extension Lane

`statcalib` is reported only as a separately labeled sixth lane.

| Scenario | StatCalib `final_ler_mean +- std` | Status | Reason | Generated Windows Mean | Signal Norm Mean |
| --- | --- | --- | --- | ---: | ---: |
| `static_bias_theta` | `0.431708 +- 0.000412` | `generated` | `statcalib_params_emitted` | `899.5` | `0.186567` |
| `linear_ramp` | `0.467083 +- 0.000123` | `generated` | `statcalib_params_emitted` | `900.0` | `0.166400` |
| `step_sigma_theta` | `0.460016 +- 0.000152` | `generated` | `statcalib_params_emitted` | `900.0` | `0.170711` |
| `periodic_drift` | `0.438751 +- 0.000183` | `generated` | `statcalib_params_emitted` | `899.5` | `0.181411` |

## Extension-Lane Gap Summary

| Scenario | Frozen Winner | Frozen Winner LER | Frozen Runner-Up | Frozen Runner-Up LER | StatCalib LER | Gap vs Frozen Winner | Gap vs Frozen Runner-Up |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| `static_bias_theta` | `hybrid_residual_b` | `0.810902` | `ukf` | `0.825370` | `0.431708` | `0.379193` | `0.393662` |
| `linear_ramp` | `hybrid_residual_b` | `0.787755` | `ukf` | `0.811201` | `0.467083` | `0.320672` | `0.344117` |
| `step_sigma_theta` | `hybrid_residual_b` | `0.788800` | `ukf` | `0.811548` | `0.460016` | `0.328783` | `0.351531` |
| `periodic_drift` | `hybrid_residual_b` | `0.806392` | `ukf` | `0.821558` | `0.438751` | `0.367641` | `0.382807` |

So the bounded answer to the T64 question is:

1. the frozen five-mode table remains unchanged and reproducible
2. `statcalib` can be added as a separate extension lane without rewriting that frozen table
3. inside this bounded software-HIL matrix, `statcalib` wins all four scenarios by a large margin over both the frozen winner and the frozen runner-up

## Residual Risk

What T64 closes:

1. a clean-provenance bounded extension-lane benchmark exists on the locked four-scenario protocol
2. the extension lane can be reported honestly without silently rewriting `T24`

What T64 does not close:

1. `.tflite` runtime validation
2. real-board validation
3. any claim beyond mock-backed software-HIL
4. paper-grade expanded benchmark evidence by itself
5. any claim that the historical `T24` frozen ranked table has been replaced

The practical interpretation is narrow: `statcalib` now has a clean, bounded, separately labeled four-scenario extension-lane result pack. That is stronger than the earlier smoke evidence, but it is still not deployment-boundary validation.
