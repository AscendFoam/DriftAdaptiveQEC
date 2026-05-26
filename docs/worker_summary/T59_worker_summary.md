# T59 Worker Summary

## What Changed

- Added a separate `statcalib` slow-loop mode in `cnn_fpga/runtime/slow_loop_runtime.py`.
- Added a minimal teacher-anchored statcalib estimator and histogram-summary helpers in `cnn_fpga/decoder/statcalib.py`.
- Added statcalib status propagation to `cnn_fpga/benchmark/run_hil_suite.py` and `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py`.
- Added task-scoped smoke config `cnn_fpga/config/p4_multiscenario_statcalib_smoke.yaml`.
- Added focused runtime smoke tests in `tests/test_statcalib_runtime_smoke.py`.
- Generated the bounded smoke run at `runs/p4_benchmark/t59statc_20260526_211532_3a3d00_23740`.

## Verification

- `C:\ProgramData\anaconda3\python.exe -m unittest tests.test_statcalib_interface tests.test_statcalib_runtime_smoke`
- `C:\ProgramData\anaconda3\python.exe -m py_compile cnn_fpga/decoder/statcalib.py cnn_fpga/runtime/slow_loop_runtime.py cnn_fpga/benchmark/run_hil_suite.py cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py tests/test_statcalib_interface.py tests/test_statcalib_runtime_smoke.py`
- Bounded smoke:
  - initial command: `C:\ProgramData\anaconda3\python.exe -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config cnn_fpga/config/p4_multiscenario_statcalib_smoke.yaml --scenario static_bias_theta --scenario linear_ramp --mode ukf --mode hybrid_residual_b --mode statcalib --paired-seeds --repeats 1`
  - resumed completion with fixed run root after command timeout: `C:\ProgramData\anaconda3\python.exe -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config cnn_fpga/config/p4_multiscenario_statcalib_smoke.yaml --scenario static_bias_theta --scenario linear_ramp --mode ukf --mode hybrid_residual_b --mode statcalib --paired-seeds --repeats 1 --run-dir runs/p4_benchmark/t59statc_20260526_211532_3a3d00_23740`
- Confirmed `comparison.csv` contains separate `mode=statcalib` rows with `coverage=1.0`.
- Confirmed `hil_summary.json` and final runtime metadata expose `statcalib_status` and `statcalib_reason` end-to-end.
- Confirmed no theory-only materials or forbidden governance docs were modified.

## Residual Risk

- This is integration evidence only, not `FR8` formal comparator evidence.
- The statcalib smoke result is unexpectedly strong, so it needs a bounded fairness/robustness follow-up before any broader paper claim.
- The smoke remains limited to `2 scenarios x 3 modes x 1 repeat`; no ranked-set conclusion should be drawn from it.
