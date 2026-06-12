# sidecar_teacher_adaptive_confidence：S0 设计草案

## 状态

- 来源：旧 `.wt/teach` worktree 的 S0_design，PSE1 收编到 main 控制台。
- 当前级别：`S0_design`
- 当前状态：`candidate`
- 旧分支：`codex/sidecar-adaptive-teacher-replay`，已退役为 read-only reference。
- 结果目录：`runs/sidecar/sidecar_teacher_adaptive_confidence/<run_id>/`

## 目标

设计 adaptive syndrome-only teacher / confidence-aware teacher replay：根据窗口统计、teacher drift、variance spike、staleness 或 mixed-row 风险，离线评估 maintain、freeze、fallback、defer 等 teacher policy。

## 边界

- 第一阶段只做 offline replay。
- 不改变 runtime teacher 默认语义。
- 不把 replay policy 写成正式 comparator。
- 不触碰 `.tflite` 或 real-board 路线。

## S1 最小问题

1. confidence 字段如何定义，才能避免只做事后解释？
2. policy action 是否能在不降低安全性的前提下减少 bad update？
3. teacher freeze / fallback 触发是否可复现？
4. 失败结果如何反过来约束论文叙事？

## 建议新增文件

- `cnn_fpga/sidecar/teacher_adaptive_confidence/...`
- `cnn_fpga/benchmark/sidecar_teacher_confidence_replay.py`
- `tests/test_sidecar_teacher_adaptive_confidence_*.py`

这些文件只能在后续 S1 任务包授权后新增。

