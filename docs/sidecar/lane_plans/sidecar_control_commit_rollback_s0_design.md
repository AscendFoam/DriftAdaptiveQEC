# sidecar_control_commit_rollback：S0 设计草案

## 状态

- 来源：旧 `.wt/ctrl` worktree 的 S0_design，PSE1 收编到 main 控制台。
- 当前级别：`S0_design`
- 当前状态：`candidate`
- 旧分支：`codex/sidecar-atomic-commit-rollback`，已退役为 read-only reference。
- 结果目录：`runs/sidecar/sidecar_control_commit_rollback/<run_id>/`

## 目标

定义参数提交、ack、rollback、freeze、version mismatch、timeout 和 safety fallback 的 control-safety sidecar。第一阶段只允许 deterministic mock contract tests，不允许 real-board write-side 行为。

## 边界

- 不把 `board_backend.py` placeholder 写成真实板级完成。
- 不执行 MMIO/DMA/register 写入。
- 不改变主线 commit 默认语义。
- 不把控制安全测试写成 hardware validation。

## S1 最小问题

1. pending / committed / acked / failed / rolled back 状态如何定义？
2. 哪些事件触发 rollback 或 freeze？
3. mock contract tests 能覆盖哪些真实风险，不能覆盖哪些？
4. event log 和 manifest 应记录哪些字段？

## 建议新增文件

- `cnn_fpga/sidecar/control_commit_rollback/...`
- `tests/test_sidecar_control_commit_rollback_*.py`

这些文件只能在后续 S1 任务包授权后新增。

