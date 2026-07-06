# 投稿稿 Phase A repeat plan 记录

日期：2026-07-03

本文档服务 `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex`。本轮只把 UKF-vs-Hybrid repeat-expanded anchor comparison 的执行命令、分块、输出需求和 claim boundary 机器可读化；它不运行 benchmark，不新增统计结论，也不补硬件证据。

## 生成文件

- `docs\paper_materials\submission_draft_phase_a_repeat_plan.csv`
- `docs\paper_materials\submission_draft_phase_a_repeat_plan.json`

## Plan shape

- Formal-length Phase A rows: `12`
- Smoke-length feasibility rows: `4`
- Scenario unit: four predeclared scenario families reported separately.
- Comparison unit: paired seed/repeat within scenario.
- Mode subset: `ukf` versus `hybrid_residual_b` only.

## Formal-length chunks

| Scenario | Repeats | Chunks | Config | Boundary |
| --- | ---: | --- | --- | --- |
| `static_bias_theta` | 12 | `0-4, 4-8, 8-12` | `cnn_fpga/config/p4_multiscenario_strong_baselines.yaml` | planned only until completed and audited |
| `linear_ramp` | 12 | `0-4, 4-8, 8-12` | `cnn_fpga/config/p4_multiscenario_strong_baselines.yaml` | planned only until completed and audited |
| `step_sigma_theta` | 12 | `0-4, 4-8, 8-12` | `cnn_fpga/config/p4_multiscenario_strong_baselines.yaml` | planned only until completed and audited |
| `periodic_drift` | 12 | `0-4, 4-8, 8-12` | `cnn_fpga/config/p4_multiscenario_strong_baselines.yaml` | planned only until completed and audited |

## Smoke feasibility rows

Smoke-length rows are allowed only to test command shape, missing-row accounting and collector logic. They must not be copied into the main performance claim.

## 可写边界

- 可以写：Phase A 的 repeat-expanded execution plan 已经有机器可读 command rows、scenario/mode/repeat units 和 post-run artifact requirements。
- 可以写：正式统计升级必须等待 formal-length run 完成并进行 paired interval analysis。
- 不能写：该 plan 证明了 robustness、statistical significance、hardware latency/resource、source-vs-board agreement 或 deployment readiness。
