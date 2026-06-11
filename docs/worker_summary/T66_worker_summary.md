# T66 Worker Summary

Changed files:

- `cnn_fpga/config/p4_multiscenario_statcalib_sensitivity.yaml`
- `cnn_fpga/benchmark/summarize_statcalib_sensitivity.py`
- `tests/test_statcalib_sensitivity_summary.py`
- `docs/evidence_packs/statcalib_fr8/statcalib_sensitivity_bounded_benchmark.md`
- `docs/review/T66_review.md`
- `docs/for_human/T66_explanation.md`
- `docs/worker_summary/T66_worker_summary.md`
- `docs/tasks/Phase2/T66_fr8_statcalib_sensitivity_bounded_benchmark.md`

Benchmark commands executed:

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config C:\Users\26410\AppData\Local\Temp\t66_statcalib_sensitivity_20260529_210906.yaml --scenario static_bias_theta --scenario linear_ramp --scenario step_sigma_theta --scenario periodic_drift --mode ukf --mode hybrid_residual_b --mode statcalib_default --mode statcalib_low_scale --mode statcalib_high_scale --mode statcalib_low_clip --mode statcalib_high_threshold --paired-seeds --repeats 2 --run-dir runs/p4_benchmark/T66_statcalib_sensitivity_20260529_210906
```

```powershell
Start-Process -FilePath 'C:\ProgramData\anaconda3\python.exe' -ArgumentList '-m','cnn_fpga.benchmark.run_p4_multiscenario_benchmark','--config','C:\Users\26410\AppData\Local\Temp\t66_statcalib_sensitivity_20260529_210906.yaml','--scenario','static_bias_theta','--scenario','linear_ramp','--scenario','step_sigma_theta','--scenario','periodic_drift','--mode','ukf','--mode','hybrid_residual_b','--mode','statcalib_default','--mode','statcalib_low_scale','--mode','statcalib_high_scale','--mode','statcalib_low_clip','--mode','statcalib_high_threshold','--paired-seeds','--repeats','2','--run-dir','runs/p4_benchmark/T66_statcalib_sensitivity_20260529_210906'
```

Run root:

- `runs/p4_benchmark/T66_statcalib_sensitivity_20260529_210906`

Verification:

1. `C:\ProgramData\anaconda3\python.exe -m py_compile cnn_fpga/benchmark/summarize_statcalib_sensitivity.py`
   - passed
2. `C:\ProgramData\anaconda3\python.exe -m unittest tests.test_statcalib_sensitivity_summary`
   - `Ran 5 tests`, `OK`
3. `C:\ProgramData\anaconda3\python.exe -m cnn_fpga.benchmark.summarize_statcalib_sensitivity --run-dir runs/p4_benchmark/T66_statcalib_sensitivity_20260529_210906`
   - wrote `statcalib_sensitivity_summary/summary.json`, `scenario_summary.csv`, `mode_summary.csv`
4. provenance and integrity checks
   - launch commit = finish commit = `summary.json git_commit` = `ad981bb`
   - exactly one T66 run root exists
   - `missing_runs=[]`
   - `comparison_rows=28`
   - `raw_rows=56`
   - all comparison rows have `coverage=1.0` and `completed_repeats=2`
   - historical `T24/T64/T61/T62` run roots retain pre-T66 last-write times

Scenario-by-scenario outcome summary:

1. `static_bias_theta`
   - best statcalib variant: `statcalib_high_threshold`
   - LER: `0.431684`
   - gap vs `ukf`: `0.394934`
   - gap vs `hybrid_residual_b`: `0.379889`
   - status: `mixed`
2. `linear_ramp`
   - best statcalib variant: `statcalib_default`
   - LER: `0.467094`
   - gap vs `ukf`: `0.340788`
   - gap vs `hybrid_residual_b`: `0.321178`
   - status: `generated`
3. `step_sigma_theta`
   - best statcalib variant: `statcalib_default`
   - LER: `0.458834`
   - gap vs `ukf`: `0.356466`
   - gap vs `hybrid_residual_b`: `0.328883`
   - status: `generated`
4. `periodic_drift`
   - best statcalib variant: `statcalib_default`
   - LER: `0.439352`
   - gap vs `ukf`: `0.383509`
   - gap vs `hybrid_residual_b`: `0.367232`
   - status: `generated`

Best statcalib variant checks:

1. best aggregate variant by mean LER: `statcalib_high_threshold`
2. it still beats `ukf` in all four scenarios
3. it still beats `hybrid_residual_b` in all four scenarios
4. more stable local pattern: `statcalib_default` wins `3/4` scenarios while `statcalib_high_threshold` wins `1/4`

Remaining risks:

1. evidence is still mock-backed software-HIL only
2. this does not rewrite `T24`
3. this does not validate `.tflite` or real-board behavior
4. the first foreground launch hit shell timeout, so the same full-matrix command was relaunched in the background under the same run root
5. because of that relaunch, `progress.jsonl` contains one duplicate `running` record for `static_bias_theta/statcalib_default/repeat_01`
6. `static_bias_theta / statcalib_high_threshold` is the best row in that scenario but its aggregated statcalib status is `mixed`, so the bounded win should not be paraphrased as uniformly clean generated-only behavior
