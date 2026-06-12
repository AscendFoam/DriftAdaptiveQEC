# sidecar_slowloop_temporal_tcn：S0 设计草案

## 状态

- 来源：旧 `.wt/tcn` worktree 的 S0_design，PSE1 收编到 main 控制台。
- 当前级别：`S0_design`
- 当前状态：`candidate`
- 旧分支：`codex/sidecar-temporal-tcn-residual`，已退役为 read-only reference。
- 结果目录：`runs/sidecar/sidecar_slowloop_temporal_tcn/<run_id>/`

## 目标

探索 slow-loop temporal modeling：把最近多个 slow-loop window 的 syndrome histogram、histogram delta、窗口均值/方差、EWMA 或 entropy 等统计量组织为时间序列，用一个 tiny TCN 或等价轻量模型输出 residual-`b` 候选。

## 边界

- 不改变 fast-loop input/output contract。
- 不修改主线 `ParamMapper` 或 `SlowLoopRuntime` 默认行为。
- 第一阶段只做 cached replay 或 toy probe。
- 不写成 formal benchmark。
- 不写入主线 `runs/p4_benchmark/*`。

## S1 最小问题

1. 当前 5-window histogram stack 相比真正 temporal model 缺少什么？
2. 使用更长历史或 causal convolution 是否能改善 drift regime 识别？
3. 输出 residual-`b` 时，如何限制幅度、平滑度和 fallback？
4. negative result 是否说明当前短历史特征已经足够？

## 建议新增文件

- `cnn_fpga/sidecar/slowloop_temporal_tcn/...`
- `cnn_fpga/benchmark/sidecar_slowloop_temporal_tcn_replay.py`
- `tests/test_sidecar_slowloop_temporal_tcn_*.py`

这些文件只能在后续 S1 任务包授权后新增。

