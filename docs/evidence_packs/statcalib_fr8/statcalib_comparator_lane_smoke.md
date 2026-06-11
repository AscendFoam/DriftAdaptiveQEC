# T59 Statcalib Comparator Lane Smoke

## Scope

This task integrated `statcalib` as a separate slow-loop comparator lane and ran one bounded smoke only. It did not rewrite frozen `T24` semantics, did not touch theory-only materials, and did not claim `FR8` formal evidence.

## Code And Config Changes

- Added a minimal teacher-anchored statcalib estimator in `cnn_fpga/decoder/statcalib.py`.
- Added a distinct `slow_loop.mode=statcalib` path in `cnn_fpga/runtime/slow_loop_runtime.py`.
- Propagated `statcalib_status` / `statcalib_reason` into `hil_summary.json`, `comparison.csv`, `summary.json`, and `report.md`.
- Added focused runtime tests in `tests/test_statcalib_runtime_smoke.py`.
- Added task-scoped smoke config `cnn_fpga/config/p4_multiscenario_statcalib_smoke.yaml`.

## Interpreter

- `C:\ProgramData\anaconda3\python.exe`

## Smoke Commands

Initial bounded smoke command:

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config cnn_fpga/config/p4_multiscenario_statcalib_smoke.yaml --scenario static_bias_theta --scenario linear_ramp --mode ukf --mode hybrid_residual_b --mode statcalib --paired-seeds --repeats 1
```

The first foreground run hit the Codex command timeout before the whole smoke matrix finished. To avoid widening scope or discarding completed repeats, the successful completion used the same bounded matrix with one extra resume argument:

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config cnn_fpga/config/p4_multiscenario_statcalib_smoke.yaml --scenario static_bias_theta --scenario linear_ramp --mode ukf --mode hybrid_residual_b --mode statcalib --paired-seeds --repeats 1 --run-dir runs/p4_benchmark/t59statc_20260526_211532_3a3d00_23740
```

## Run Root

- `runs/p4_benchmark/t59statc_20260526_211532_3a3d00_23740`

## Smoke Matrix

- Scenarios: `static_bias_theta`, `linear_ramp`
- Modes: `ukf`, `hybrid_residual_b`, `statcalib`
- Repeats: `1`
- Seed policy: `--paired-seeds`

## Per-Mode Summary

| scenario | mode | final_ler_mean | overflow_rate_mean | statcalib_status | statcalib_reason |
| --- | --- | ---: | ---: | --- | --- |
| static_bias_theta | ukf | 0.8250075 | 0.0026541667 | not_applicable | mode_does_not_emit_statcalib |
| static_bias_theta | hybrid_residual_b | 0.8087958333 | 0.0026337500 | not_applicable | mode_does_not_emit_statcalib |
| static_bias_theta | statcalib | 0.4315304167 | 0.0016541667 | generated | statcalib_params_emitted |
| linear_ramp | ukf | 0.8179529167 | 0.0025500000 | not_applicable | mode_does_not_emit_statcalib |
| linear_ramp | hybrid_residual_b | 0.8031875000 | 0.0026116667 | not_applicable | mode_does_not_emit_statcalib |
| linear_ramp | statcalib | 0.4450845833 | 0.0016429167 | generated | statcalib_params_emitted |

## Statcalib Lane Behavior

- `comparison.csv` contains a separate `mode=statcalib` row for both scenarios.
- `hil_summary.json` for `statcalib` exposes `statcalib_diagnostics.status=generated`, `reason=statcalib_params_emitted`, and `statcalib_generated_windows=600`.
- Runtime metadata for the final committed params includes `statcalib_status`, `statcalib_reason`, `statcalib_provenance`, `applied_delta_b`, and `statcalib_metadata`.
- In this smoke, `statcalib` emitted params on all observed windows for both scenarios.

## What Still Remains Before FR8

This is still not `FR8`.

What remains before any formal `FR8` result-table task:

- Freeze whether this statcalib lane definition is the comparator we actually want to defend.
- Run a dedicated fairness sanity check on the unexpectedly strong smoke outcome before treating it as benchmark evidence.
- Re-run on the full intended `FR8` matrix only under a new task package with explicit boundary and reporting rules.

The current smoke only proves lane integration, status propagation, and bounded end-to-end execution.
