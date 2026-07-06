# 投稿稿 runner smoke pair 记录

日期：2026-07-03

本文档记录一次用于投稿稿规划的 runner smoke pair：单个 scenario、单个 paired repeat、只比较 UKF 与 Hybrid Residual-B。它证明当前 runner 路径可执行并产出可解析的 comparison/summary 文件，但不进入论文主结果层。

## Source run

- run dir: `runs\paper_submission_pilot\smoke_static_ukf_hybrid_r1`
- summary: `runs\paper_submission_pilot\smoke_static_ukf_hybrid_r1\summary.json`
- comparison: `runs\paper_submission_pilot\smoke_static_ukf_hybrid_r1\comparison.csv`

## Pilot result

| Scenario | UKF final LER | Hybrid final LER | UKF-minus-Hybrid delta | Relative reduction |
| --- | ---: | ---: | ---: | ---: |
| `static_bias_theta` | 0.823552 | 0.802429 | 0.021123 | 2.56% |

## 可写边界

- 可以写入内部材料：正式 runner 的 UKF/Hybrid smoke pair 已可执行并可被机器解析。
- 可以用于估算：repeat-expanded benchmark 的运行成本和后续任务规模。
- 不能写入主文性能 claim：它不是 expanded benchmark，不是 CI/p-value，不是 holdout robustness，不是硬件证据。
