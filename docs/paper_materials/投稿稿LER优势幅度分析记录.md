# 投稿稿 LER 优势幅度分析记录

日期：2026-07-03

本记录从 Fig. 2 的 `source_data_fig02_main_results.csv` 与 `source_data_fig02_paired_deltas.csv` 派生 UKF-minus-Hybrid LER 优势幅度。它不重新运行 benchmark，不新增实验，也不提供统计显著性结论。

## 生成文件

- `docs\paper_materials\submission_draft_ler_advantage_margin_analysis.csv`
- `docs\paper_materials\submission_draft_ler_advantage_margin_analysis.json`

## 结果摘要

| Scenario | UKF mean | Hybrid mean | Mean delta | Rel. reduction | Direction | Delta/max SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `static_bias_theta` | 0.825370 | 0.810902 | 0.014469 | 1.75% | 2/2 | 12.17 |
| `linear_ramp` | 0.811201 | 0.787755 | 0.023446 | 2.89% | 2/2 | 27.00 |
| `step_sigma_theta` | 0.811547 | 0.788800 | 0.022748 | 2.80% | 2/2 | 21.28 |
| `periodic_drift` | 0.821558 | 0.806392 | 0.015166 | 1.85% | 2/2 | 8.05 |

## 可写边界

- 可以写：四个预声明场景中，现有 paired repeats 的 UKF-minus-Hybrid delta 均为正。
- 可以写：mean delta、relative reduction 和 delta/max reported SD 是描述性优势幅度读数。
- 不可以写：这些数字构成置信区间、标准误、p 值、显著性检验、expanded benchmark、holdout robustness 或硬件测量。
