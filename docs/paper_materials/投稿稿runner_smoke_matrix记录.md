# 投稿稿 runner smoke matrix 记录

日期：2026-07-03

本文档记录一次用于投稿稿 repeat-expansion 路线的 all-scenario runner pilot：四个预声明 scenario、UKF 与 Hybrid Residual-B 两个 anchor mode、两个 paired repeats、smoke-length 配置。它证明全场景 runner 路径可执行并能产出完整 comparison/summary 文件，但不进入论文主结果层。

## Source run

- run dir: `runs\paper_submission_pilot\smoke_all_scenarios_ukf_hybrid_r2_20260703_093537`
- summary: `runs\paper_submission_pilot\smoke_all_scenarios_ukf_hybrid_r2_20260703_093537\summary.json`
- comparison: `runs\paper_submission_pilot\smoke_all_scenarios_ukf_hybrid_r2_20260703_093537\comparison.csv`

## Pilot matrix

| Scenario | UKF final LER | Hybrid final LER | UKF-minus-Hybrid delta | Relative reduction | Positive pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| `static_bias_theta` | 0.817940 | 0.801012 | 0.016927 | 2.07% | 2/2 |
| `linear_ramp` | 0.840509 | 0.813909 | 0.026600 | 3.16% | 2/2 |
| `step_sigma_theta` | 0.828792 | 0.816023 | 0.012769 | 1.54% | 2/2 |
| `periodic_drift` | 0.810660 | 0.781844 | 0.028817 | 3.55% | 2/2 |

## 可写边界

- 可以写入投稿材料：repeat-expansion 的全场景 UKF/Hybrid runner pilot 已可执行，并且四个 smoke scenario 都有完整 paired rows。
- 可以用于规划：正式 Phase A repeat-expanded benchmark 的运行成本、source-data 字段和缺失行检查。
- 不能写入主文性能 claim：它使用 smoke-length timing，不是当前主结果 benchmark，不是 expanded benchmark，不是 CI/p-value，不是 holdout robustness，也不是硬件证据。
