# Sidecar Worktree 计划与退役说明

## 1. 状态

- 原始计划日期：`2026-06-08`
- PSE1 更新日期：`2026-06-12`
- 当前状态：旧 Wave A 长期 worktree 计划退役为 historical reference。

## 2. 退役原因

main 分支在 `T70` 之后继续推进了大量主线治理、paper-material 和 evidence-pack 整理。旧 `.wt/*` 分支相对 main 的文档差异持续扩大，而每条分支实际保留的 sidecar 独有资产主要只是 S0 设计文档。继续逐分支同步会消耗大量治理成本，不利于实验推进。

因此 PSE1 之后：

1. 不要求维护 `.wt/tcn`、`.wt/teach`、`.wt/bank`、`.wt/ctrl` 的长期同步。
2. 不要求基于这些旧 worktree 创建 Codex 会话。
3. 不从这些旧分支直接合并主线事实。
4. 有价值的 S0 思路收编到 `docs/sidecar/lane_plans/`。

## 3. 旧 Wave A 映射

| Lane ID | 旧分支 | 旧 worktree | PSE1 状态 |
| --- | --- | --- | --- |
| `sidecar_slowloop_temporal_tcn` | `codex/sidecar-temporal-tcn-residual` | `.wt/tcn` | retired/read-only |
| `sidecar_teacher_adaptive_confidence` | `codex/sidecar-adaptive-teacher-replay` | `.wt/teach` | retired/read-only |
| `sidecar_fastloop_gain_bank` | `codex/sidecar-gain-scheduled-bank-sim` | `.wt/bank` | retired/read-only |
| `sidecar_control_commit_rollback` | `codex/sidecar-atomic-commit-rollback` | `.wt/ctrl` | retired/read-only |

## 4. 新执行面策略

后续 sidecar 不再默认创建长期分支。执行面按任务强度选择：

1. `S0_design` / `S1_toy_or_replay`：优先在 main checkout 下新增-only 文件推进。
2. 需要并行 Codex 会话：从当前 main 稳定 commit 新开短生命周期 worktree。
3. 需要多日长跑：使用 clean short-path clone，例如 `C:/daqec_<lane>/`，并记录 launch provenance。

任何执行面都必须遵守：

- 结果写入 `runs/sidecar/<lane_id>/<run_id>/`。
- 不写入主线历史 run root。
- 不改写 `T24`、`T64-T70`、`.tflite` 或 real-board 事实口径。
- 不把 sidecar 输出直接写入当前唯一主线任务。

## 5. 清理建议

旧 `.wt/*` worktree 暂时可以保留作只读参考。若后续需要清理，必须单独确认：

1. S0 设计是否已完整收编到 `docs/sidecar/lane_plans/`。
2. 旧 worktree 中是否还有未保存的人工编辑。
3. 清理命令是否只移除 linked worktree，不误删主仓库。

本文件不授权执行清理命令。
