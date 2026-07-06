# 投稿稿 benchmark expansion protocol 补强记录

日期：2026-07-03

本文档服务 `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex`。本轮只把当前 `n=2` paired descriptive 主结果转换为下一步可执行的 repeat-expanded benchmark protocol；不运行 benchmark，不新增实验，不报告 CI / p-value，也不升级 hardware、`.tflite`、real-board 或 deployment 证据等级。

## 生成文件

- `docs\paper_materials\submission_draft_benchmark_expansion_protocol.csv`
- `docs\paper_materials\submission_draft_benchmark_expansion_protocol.json`

## Phase A：repeat-expanded anchor comparison

Phase A 只比较当前主结果最关键的 `ukf` 与 `hybrid_residual_b`，保留现有五模式主表作为 anchor，不把 repeat expansion 静默写成新的五模式 full matrix。

| Scenario | Current pairs | Mean delta | Sample SD | Min delta | Positive pairs | Planning min pairs | Target pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `static_bias_theta` | 2 | 0.014469 | 0.000729 | 0.013953 | 2/2 | 12 | 16 |
| `linear_ramp` | 2 | 0.023446 | 0.000607 | 0.023017 | 2/2 | 12 | 16 |
| `step_sigma_theta` | 2 | 0.022748 | 0.000436 | 0.022440 | 2/2 | 12 | 16 |
| `periodic_drift` | 2 | 0.015166 | 0.002256 | 0.013571 | 2/2 | 12 | 16 |

## 可写边界

- 可以写：当前稿件已把下一步强统计 benchmark 的 repeat unit、scenario unit、mode subset、minimum repeat budget 和 upgrade gate 机器可读化。
- 可以写：Phase A 的目标是把 UKF-vs-Hybrid 从 descriptive paired deltas 升级为 repeat-expanded paired interval analysis。
- 可以写：Phase B 才会处理 random-walk、burst/reset、faster-than-window 等 holdout drift family。
- 不能写：当前 `n=2` 数据已有 confidence interval、p-value、significance 或 robustness proof。
- 不能写：controlled holdout stress diagnostics 已经等价于正式 software-HIL holdout benchmark。
- 不能写：该 protocol 证明了 FPGA latency/resource、source-vs-board agreement 或硬件有效性。
