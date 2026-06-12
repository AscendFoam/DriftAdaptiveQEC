# sidecar_fastloop_gain_bank：S0 设计草案

## 状态

- 来源：旧 `.wt/bank` worktree 的 S0_design，PSE1 收编到 main 控制台。
- 当前级别：`S0_design`
- 当前状态：`candidate`
- 旧分支：`codex/sidecar-gain-scheduled-bank-sim`，已退役为 read-only reference。
- 结果目录：`runs/sidecar/sidecar_fastloop_gain_bank/<run_id>/`

## 目标

探索 gain-scheduled / piecewise-affine parameter bank：不改变 fast-loop correction contract，只让 slow-loop 选择有限 bank 或 piecewise-affine 参数候选，用于后续 bank-selection replay。

## 边界

- 不实现真实 FPGA bank。
- 不改变 `param_bank` 主线默认行为。
- 不改变 `Delta = K @ s + b` contract。
- 第一阶段只做 toy replay 或 contract-level safety check。

## S1 最小问题

1. bank 数量和 region 切分是否能减少参数抖动？
2. bank thrashing、range clipping、rollback 如何检测？
3. 与当前 single committed `(K,b)` 路线相比，收益是否只是复杂度转移？
4. 是否需要被标为 `contract_or_registry_touch`？

## 建议新增文件

- `cnn_fpga/sidecar/fastloop_gain_bank/...`
- `cnn_fpga/benchmark/sidecar_gain_bank_replay.py`
- `tests/test_sidecar_fastloop_gain_bank_*.py`

这些文件只能在后续 S1 任务包授权后新增。

