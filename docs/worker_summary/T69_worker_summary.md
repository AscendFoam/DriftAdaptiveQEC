# T69 Worker Summary

## 改了什么

新增了这 3 个 task-scoped 文件：

1. `cnn_fpga/config/p4_multiscenario_statcalib_clean_winner_tiebreak.yaml`
2. `cnn_fpga/benchmark/summarize_statcalib_clean_winner_tiebreak.py`
3. `tests/test_statcalib_clean_winner_tiebreak_summary.py`

并完成了唯一允许的 run root：

- `runs/p4_benchmark/T69_statcalib_clean_winner_tiebreak_20260608_160358`

随后补齐了这 4 份 T69 文档：

1. `docs/statcalib_clean_winner_tiebreak_bounded_benchmark.md`
2. `docs/review/T69_review.md`
3. `docs/for_human/T69_explanation.md`
4. `docs/worker_summary/T69_worker_summary.md`

同时把同样结论追加回任务包：

- `docs/tasks/Phase2/T69_fr8_statcalib_clean_winner_tiebreak_bounded_benchmark.md`

## 怎么验证

本地验证：

1. `C:\ProgramData\anaconda3\python.exe -m py_compile cnn_fpga/benchmark/summarize_statcalib_clean_winner_tiebreak.py`
2. `C:\ProgramData\anaconda3\python.exe -m unittest tests.test_statcalib_clean_winner_tiebreak_summary`
   - 结果：`Ran 5 tests`, `OK`

benchmark 执行命令：

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config C:\t69cfg_20260608_160358.yaml --scenario static_bias_theta --scenario linear_ramp --scenario step_sigma_theta --scenario periodic_drift --mode ukf --mode hybrid_residual_b --mode statcalib_window_variance_t001 --mode statcalib_window_variance_t003 --mode statcalib_window_variance_t005 --mode statcalib_ekf_t001 --paired-seeds --repeats 4 --run-dir D:\Codes\Quantum\DriftAdaptiveQEC\runs\p4_benchmark\T69_statcalib_clean_winner_tiebreak_20260608_160358
```

summary helper：

1. `C:\ProgramData\anaconda3\python.exe -m cnn_fpga.benchmark.summarize_statcalib_clean_winner_tiebreak --run-dir runs/p4_benchmark/T69_statcalib_clean_winner_tiebreak_20260608_160358`

provenance 与完整性：

1. launch branch = `main`
2. launch `HEAD = 1dbfbc3`
3. finish branch = `main`
4. finish `HEAD = 1dbfbc3`
5. `summary.json["git_commit"] = 1dbfbc3`
6. T69 run root 数量 = `1`
7. historical `T24/T64/T66/T67/T68` run roots 没有 diff
8. `missing_runs = []`
9. comparison rows = `24`
10. raw rows = `96`
11. 所有 comparison rows 都是 `coverage = 1.0`
12. 所有 comparison rows 都是 `completed_repeats = 4`
13. `progress.jsonl` 中 `running = 96`、`completed = 96`
14. duplicate `running = 0`
15. duplicate `completed = 0`

## 结果结论

四个候选的结果是：

1. `statcalib_window_variance_t001`
   - generated rows = `4`
   - mixed rows = `0`
   - full generated-only = `True`
   - mean LER = `0.448167534722222`
   - worst-case LER = `0.466226875000000`
2. `statcalib_window_variance_t003`
   - generated rows = `4`
   - mixed rows = `0`
   - full generated-only = `True`
   - mean LER = `0.448167534722222`
   - worst-case LER = `0.466226875000000`
3. `statcalib_window_variance_t005`
   - generated rows = `4`
   - mixed rows = `0`
   - full generated-only = `True`
   - mean LER = `0.448167534722222`
   - worst-case LER = `0.466226875000000`
4. `statcalib_ekf_t001`
   - generated rows = `4`
   - mixed rows = `0`
   - full generated-only = `True`
   - mean LER = `0.449340763888889`
   - worst-case LER = `0.466866458333333`

四个场景的最佳 statcalib 结果都相同：

1. `static_bias_theta`：`window_variance_t001 = t003 = t005`
2. `linear_ramp`：`window_variance_t001 = t003 = t005`
3. `step_sigma_theta`：`window_variance_t001 = t003 = t005`
4. `periodic_drift`：`window_variance_t001 = t003 = t005`

pairwise 结果：

1. 三个 `window_variance` 候选彼此在 `4/4` 场景里全部打平
2. 这三个候选都在 `4/4` 场景里全部胜过 `statcalib_ekf_t001`

grouped tie-break 结论：

1. `T68` clean-winner tie set = `window_variance_t001 = t003 = t005`
2. `T69` clean answer set 仍然是 `window_variance_t001 = t003 = t005`
3. `T68` tie-set relation = `persists`
4. mean-best candidate set 和 worst-case-best candidate set = `same`
5. final classification = `persistent_clean_tie_set`
6. unique clean reference point = `False`

## 剩余风险

1. `T69` 成功回答了 tie-break 问题，但答案不是唯一点，而是 persistent tie set。后续不能把它写成“唯一最佳阈值已确定”。
2. 这仍然只是 mock-backed software-HIL extension-lane evidence，不是 `.tflite`，不是真板，也不是 mature calibration comparator。
3. `T24` 仍然是 authoritative frozen ranked table，`T69` 不能拿来重写主表，只能作为单独 extension-lane 结果报告。
