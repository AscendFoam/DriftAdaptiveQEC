# T68 Worker Summary

## 改了什么

- 新增 task-scoped config：`cnn_fpga/config/p4_multiscenario_statcalib_generated_only.yaml`
- 新增 task-scoped summary helper：`cnn_fpga/benchmark/summarize_statcalib_generated_only.py`
- 新增 focused unit test：`tests/test_statcalib_generated_only_summary.py`
- 完成了唯一允许的 T68 run root：`runs/p4_benchmark/T68_statcalib_generated_only_20260605_205723`
- 补齐了 T68 文档：
  - `docs/statcalib_generated_only_robustness_bounded_benchmark.md`
  - `docs/review/T68_review.md`
  - `docs/for_human/T68_explanation.md`

## 运行命令

- benchmark launch:

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config D:\Codes\Quantum\DriftAdaptiveQEC\runs\p4_benchmark\T68_statcalib_generated_only_20260605_205723\launch_config_from_clean_clone.yaml --scenario static_bias_theta --scenario linear_ramp --scenario step_sigma_theta --scenario periodic_drift --mode ukf --mode hybrid_residual_b --mode statcalib_window_variance_t001 --mode statcalib_window_variance_t003 --mode statcalib_window_variance_t005 --mode statcalib_window_variance_t010 --mode statcalib_ekf_t001 --mode statcalib_ekf_t003 --mode statcalib_ekf_t005 --mode statcalib_ekf_t010 --paired-seeds --repeats 2 --run-dir D:\Codes\Quantum\DriftAdaptiveQEC\runs\p4_benchmark\T68_statcalib_generated_only_20260605_205723
```

- helper / tests:
  - `C:\ProgramData\anaconda3\python.exe -m py_compile cnn_fpga/benchmark/summarize_statcalib_generated_only.py`
  - `C:\ProgramData\anaconda3\python.exe -m unittest tests.test_statcalib_generated_only_summary`
  - `C:\ProgramData\anaconda3\python.exe -m cnn_fpga.benchmark.summarize_statcalib_generated_only --run-dir runs/p4_benchmark/T68_statcalib_generated_only_20260605_205723`

## 如何验证

- `py_compile` 通过
- `unittest` 通过：
  - `Ran 7 tests`
  - `OK`
- summary helper 通过，并生成：
  - `runs/p4_benchmark/T68_statcalib_generated_only_20260605_205723/statcalib_generated_only_summary/summary.json`
  - `.../scenario_summary.csv`
  - `.../candidate_summary.csv`
  - `.../threshold_comparison.csv`
  - `.../teacher_anchor_summary.csv`
  - `.../pareto_summary.csv`

- provenance 闭合：
  - launch `HEAD = bda8f2b`
  - finish `HEAD = bda8f2b`
  - `summary.json git_commit = bda8f2b`

- run 完整性：
  - `missing_runs = []`
  - `comparison.csv` 共 `40` 行
  - 所有 comparison rows 都是 `coverage = 1.0`
  - 所有 comparison rows 都是 `completed_repeats = 2`
  - `progress.jsonl` 里：
    - `running = 80`
    - `completed = 80`
    - duplicate `running = 0`
    - duplicate `completed = 0`

- 范围保护：
  - 只有一个 T68 run root
  - forbidden scope 定向 `git diff --name-only -- ...` 为空
  - 历史 `T24/T64/T66/T67` 路径没有新的 `git status` 项

## 结果摘要

### 1. 是否存在 full generated-only winner？

存在，而且有 4 个：

1. `statcalib_window_variance_t001`
2. `statcalib_window_variance_t003`
3. `statcalib_window_variance_t005`
4. `statcalib_ekf_t001`

它们都满足：

1. 四个场景全是 `generated`
2. 四个场景都赢过 `ukf`
3. 四个场景都赢过 `hybrid_residual_b`

### 2. 哪组是最强的 clean winner？

按平均 LER 看，最强的是三路并列：

1. `statcalib_window_variance_t001`
2. `statcalib_window_variance_t003`
3. `statcalib_window_variance_t005`

它们三者：

1. `mean_final_ler_mean = 0.4475929166666667`
2. `generated_row_count = 4`
3. `mixed_row_count = 0`
4. `beats_both_frozen_anchors_count = 4`

`statcalib_ekf_t001` 也是 full generated-only winner，但平均 LER 稍差：

- `mean_final_ler_mean = 0.44855232638888887`

### 3. mean-best 和 worst-case-best 是否相同？

不同。

- mean-best candidates:
  - `statcalib_window_variance_t001`
  - `statcalib_window_variance_t003`
  - `statcalib_window_variance_t005`

- worst-case-best candidates:
  - `statcalib_window_variance_t001`
  - `statcalib_window_variance_t003`
  - `statcalib_window_variance_t005`
  - `statcalib_window_variance_t010`

差异原因：

1. `window_variance_t010` 的 worst-case LER 没有更差
2. 但它平均 LER 更差一些
3. 而且它还有 `mixed` 行，所以不属于最干净的 mean-best 集合

### 4. teacher / threshold 分组结果

- 每个 threshold 上，`window_variance` 都优于 `ekf`
- `window_variance` 阈值排序：
  - `t001 = t003 = t005 > t010`
- `ekf` 阈值排序：
  - `t003 = t005 = t010 > t001`

### 5. 仍然 mixed 的候选

- `statcalib_window_variance_t010`
  - `generated_row_count = 2`
  - `mixed_row_count = 2`
- `statcalib_ekf_t003`
  - `generated_row_count = 3`
  - `mixed_row_count = 1`
- `statcalib_ekf_t005`
  - `generated_row_count = 3`
  - `mixed_row_count = 1`
- `statcalib_ekf_t010`
  - `generated_row_count = 3`
  - `mixed_row_count = 1`

## 剩余风险

- `T68` 正面回答了 generated-only existence question，但它没有把 `statcalib` 升级成主表 comparator。
- `T24` 仍然是 authoritative frozen ranked table，`statcalib` 仍然只能单独作为 extension lane 报告。
- 证据边界仍然只是 mock-backed software-HIL，不是 `.tflite`，也不是真板。
- 这次存在多个 clean winner，并不是唯一单一最优点；后续引用必须保留 tie 结构，不能写成“已经找到唯一最佳 threshold”。
