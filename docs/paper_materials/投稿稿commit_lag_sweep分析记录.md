# 投稿稿 commit-lag sweep 分析记录

日期：2026-07-03

本记录服务于 `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex`。它在三类 holdout drift stress family 上扫描 lagged affine 的 commit lag，用于把 stale-parameter/commit-latency 风险从叙述性缺口转化为可复核的仿真诊断。

## 生成文件

- `docs\paper_materials\submission_draft_commit_lag_sweep_analysis.csv`
- `docs\paper_materials\submission_draft_commit_lag_sweep_analysis.json`

## 协议

- random seed: `20260776`
- sequences per scenario: `384`
- steps per sequence: `512`
- fixed commit interval: `64` simulation steps
- swept commit lag: `0, 8, 16, 32, 64, 128` simulation steps
- baseline references inside each scenario: oracle affine and fixed nominal affine
- metric: residual MSE and rare half-lattice residual-boundary crossing proxy

## 结果摘要

| Scenario | Lag 0 | Lag 8 | Lag 16 | Lag 32 | Lag 64 | Lag 128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random_walk_drift | 0.073831 | 0.073976 | 0.074130 | 0.074410 | 0.075173 | 0.076093 |
| burst_reset_drift | 0.067454 | 0.068193 | 0.068657 | 0.068694 | 0.068583 | 0.066913 |
| faster_than_window_oscillation | 0.069251 | 0.069406 | 0.068989 | 0.069039 | 0.068842 | 0.068249 |

## 可写边界

- 可以写：该 sweep 将 slow-loop stale parameter 风险变成了 simulation-step 级可审计变量。
- 可以写：短 lag 在 random-walk stress 中仍接近 oracle，但 burst/reset 与 faster-than-window family 对 lag 更敏感。
- 不能写：该 sweep 测得了 FPGA/board latency、timing closure、source-vs-board agreement、trained CNN branch holdout generalization、CI/p-value 或 logical-channel fidelity。
