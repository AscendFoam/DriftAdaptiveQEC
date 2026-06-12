# 00 Sidecar 快照

## 1. 目的

Sidecar 是主线之外的扩展实验控制台，用来探索慢回路时间模型、自适应 teacher、快回路参数 bank、commit / rollback 控制等候选路线。它服务于论文和后续主线筛选，但不是主线事实来源。

## 2. 当前状态

- 日期：`2026-06-12`
- 阶段：`PSE1`
- 决策：允许在 main 代码基础上推进 sidecar，但必须遵守新增-only和结果隔离规则。
- 旧 Wave A `.wt/*`：不再强制维护，视为 retired/read-only。
- 当前主线任务：仍以 `docs/04_task_board.md` 和 `docs/07_handoff.md` 为准。

## 3. 核心原则

1. main 是治理控制台，不是 sidecar 结果自动晋升入口。
2. sidecar 可以新增代码、配置、测试、helper 和文档，但不得破坏原有逻辑。
3. sidecar 结果只能写入 `runs/sidecar/<lane_id>/<run_id>/`。
4. sidecar 不得改写历史主线结果或论文 claim。
5. sidecar 若要进入主线，必须先通过 Captain promotion gate。

## 4. 当前允许的 sidecar 类型

| 类型 | 例子 | 第一安全动作 |
| --- | --- | --- |
| slow-loop temporal model | temporal histogram + tiny TCN residual head | `S0_design` 或 cached replay |
| teacher policy | adaptive teacher / confidence fallback | offline replay |
| fast-loop structure | gain-scheduled bank / LUT-assisted affine | toy replay or contract tests |
| control safety | atomic commit / rollback | deterministic mock contract tests |
| feature diagnostics | moments / EWMA / entropy | cached feature probe |

## 5. 当前不允许的外推

- 不把 sidecar 写成 `T24` 替代表。
- 不把 sidecar 写成 `statcalib` 主线 promotion。
- 不把 sidecar 写成 `.tflite` deployment。
- 不把 sidecar 写成 real-board validation。
- 不把 toy/replay 结果写成 formal benchmark。

