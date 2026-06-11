# Sidecar Governance

本目录保存并行 sidecar 扩展实验的治理和 worktree 规划文档。

## 文件清单

| 文件 | 内容 |
| --- | --- |
| `parallel_sidecar_extension_governance.md` | sidecar lane 定义、artifact schema、promotion gate 和红线 |
| `parallel_sidecar_worktree_plan.md` | sidecar worktree root、分支、候选 lane 和执行边界 |

## 边界

Sidecar 输出在 Captain promotion gate 前不能进入主线事实。任何 sidecar lane 都不得改写 `T24`、`T64-T70`、`.tflite`、real-board 或当前唯一任务状态。
