# 投稿稿 paired uncertainty 分析记录

日期：2026-07-03

本记录从 `source_data_fig02_paired_deltas.csv` 派生 paired repeat uncertainty 摘要。它不重跑 benchmark，不新增实验，不提供显著性检验。

## 生成文件

- `docs\paper_materials\submission_draft_paired_uncertainty_analysis.csv`
- `docs\paper_materials\submission_draft_paired_uncertainty_analysis.json`

## 协议

- random seed: `20260776`
- bootstrap resamples: `20000`
- source rows: UKF 与 hybrid residual branch 的 paired final_ler deltas
- bootstrap span: paired deltas 的 repeat-level mean bootstrap percentile span

## 结果摘要

| Scenario | n | Mean delta | Bootstrap span | Direction | Mean rel. reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| static_bias_theta | 2 | 0.014469 | [0.013953, 0.014984] | 2/2 | 1.75% |
| linear_ramp | 2 | 0.023446 | [0.023017, 0.023875] | 2/2 | 2.89% |
| step_sigma_theta | 2 | 0.022748 | [0.022440, 0.023056] | 2/2 | 2.80% |
| periodic_drift | 2 | 0.015166 | [0.013571, 0.016762] | 2/2 | 1.85% |
| all_scenarios | 8 | 0.018957 | [0.016044, 0.021892] | 8/8 | 2.32% |

## 可写边界

- 可以写：所有现有 paired repeats 的 UKF-minus-hybrid delta 为正，方向一致。
- 可以写：bootstrap span 是 repeat-level descriptive uncertainty marker，用于透明展示 n=2 的不确定性。
- 不能写：该 span 是 inferential confidence interval、standard error、p-value、significance test 或 distribution-level robustness proof。
