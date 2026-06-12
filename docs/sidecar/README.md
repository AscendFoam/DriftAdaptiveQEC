# Sidecar 扩展实验治理

本目录是并行 sidecar 扩展实验的统一治理入口。它承担的是“main 控制台”职责：登记路线、冻结边界、规定结果目录和晋升门槛。它不是主线 `00~08` 治理文档的替代品，也不改变当前唯一主线任务。

## 当前决策

- 日期：`2026-06-12`
- 状态：`PSE1 sidecar main-controlled governance refresh`
- 结论：旧 `.wt/*` 长期分支不再作为必须维护对象；后续 sidecar 默认由 main 中的精简治理文档管理。
- 执行面：优先使用 main 当前代码基础上的新增-only helper / standalone module / task-scoped config；如需长跑或多会话隔离，可以临时开短生命周期 worktree 或 clean clone。
- 结果目录：所有 sidecar 结果必须写入 `runs/sidecar/<lane_id>/<run_id>/`。

## 文件清单

| 文件 | 角色 |
| --- | --- |
| `00_sidecar_snapshot.md` | sidecar 当前阶段快照和核心原则 |
| `01_sidecar_lane_registry.md` | lane 注册表、旧 Wave A 路线收编状态和新增路线登记规则 |
| `02_sidecar_execution_protocol.md` | 执行协议、代码新增规则、run dir 规则和 provenance 要求 |
| `03_sidecar_artifact_schema.md` | manifest、summary、metrics、provenance 文件 schema |
| `04_sidecar_promotion_gate.md` | sidecar 结果进入主线前的 promotion gate |
| `lane_plans/` | 每条路线的 S0/S1 设计草案，作为 main 控制台下的候选材料 |
| `parallel_sidecar_extension_governance.md` | PSE1 后的总治理入口，兼容旧 PSE0 路径引用 |
| `parallel_sidecar_worktree_plan.md` | 旧 Wave A worktree 计划的退役说明和可选执行面规则 |

## 红线

Sidecar 输出在 Captain promotion gate 前不能进入主线事实。任何 sidecar lane 都不得改写 `T24`、`T64-T70`、`.tflite`、real-board 或当前唯一任务状态。
