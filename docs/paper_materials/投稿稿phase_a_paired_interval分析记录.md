# 投稿稿 Phase A paired interval 分析记录

日期：2026-07-06

本文档从 `submission_draft_phase_a_repeat_summary.csv` 中读取已经完成的 formal-length Phase A 场景行，计算 UKF-minus-Hybrid paired delta 的小样本 paired-t 区间和 paired bootstrap percentile 区间。它不运行 benchmark，不计算 p-value，也不补硬件证据。

## 生成文件

- `docs\paper_materials\submission_draft_phase_a_paired_interval_analysis.csv`
- `docs\paper_materials\submission_draft_phase_a_paired_interval_analysis.json`

## 结果

| Scenario | n | Mean delta | Paired-t 95% interval | Bootstrap 95% interval | Positive pairs | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `linear_ramp` | 12 | 0.022417 | [0.021215, 0.023619] | [0.021405, 0.023440] | 12/12 | completed_scenario_interval_positive_but_all_scenario_gate_blocked |
| `static_bias_theta` | 12 | 0.015563 | [0.014778, 0.016348] | [0.014933, 0.016256] | 12/12 | completed_scenario_interval_positive_but_all_scenario_gate_blocked |

## 可写边界

- 可以写：已完成且 lower bounds 为正的 formal-length Phase A 场景包括：`linear_ramp`, `static_bias_theta`。
- 可以写：这些结果只覆盖已完成场景的 formal interval check；四场景 repeat-expanded gate、pooled analysis、holdout drift 和硬件测量仍未完成，除非全部预声明场景均补齐并重新汇总。
- 不能写：已经证明全场景 repeat-expanded advantage、p-value 显著性、holdout robustness、FPGA latency/resource/source-vs-board agreement 或 deployment readiness。
