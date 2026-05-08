# Task Board

## Phase 0: Stabilization

- [x] T0: 冻结 legacy 状态并完成只读审计
- [x] T1: 确认依赖矩阵与最小入口
- [x] T2: 跑通最小 P0 smoke benchmark，或把阻塞固定为可执行修复项
- [x] T3: 审计 HIL / P4 链路中的 mock、stub、placeholder 边界
- [x] T4: 补软件 HIL 最小 bootstrap / smoke test
- [x] T5: 清点并处理仓库中的缓存/生成物噪声治理策略

## Phase 1: Recovery

- [x] T6: 重新验收一个软件 HIL 最小路径
- [x] T7: 重新验收一个 P4 benchmark 最小路径
- [x] T8: 决定是否进入 `Go` 或继续 `Repair`
- [x] T9: 重新验收一个 P4 frozen baseline 单场景全模式 smoke path
- [x] T10: 基于 `T8 + T9` 重新做一次 `Go / Repair` gate review
- [x] T11: 补一份恢复期最小依赖 manifest（优先覆盖 P0/P3/P4 recovery smoke）
- [x] T12: 收敛 software HIL recovery smoke 的随机源与确定性表述
- [x] T13: 做 recovery exit review 并完成阶段收尾

## Phase 2: Controlled Development

- [ ] 待定义下一张 bounded 开发任务包

## Current Unique Task

`待定义（Phase 1 已收尾，等待下一张继续开发任务包）`

已完成的收口结果：

- 连跑复验命令：
  - `& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_hil_suite --config cnn_fpga/config/hardware_hil_recovery_smoke.yaml`
  - `& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_hil_suite --config cnn_fpga/config/hardware_hil_recovery_smoke.yaml`
- 新运行目录：
  - `runs/hil_suite/hardware_hil_recovery_smoke_20260508_172221_3ae9f9176104`
  - `runs/hil_suite/hardware_hil_recovery_smoke_20260508_172232_3ae9f9176104`
- 对比结论：
  - `hil_summary.json` 的 SHA256 完全一致
  - `hil_events.json` 的 SHA256 完全一致
  - `final_ler = 0.454375`
  - `overflow_rate = 0.002`
- 当前结论：
  - 这条 bounded software HIL recovery smoke 在固定 seed 链路下已经做到逐字一致复验
  - 仍不推广到真板、`.tflite` 真实部署或正式多场景 benchmark
- `T13` 收尾结论：
  - `docs/review/T13_recovery_exit_review.md` verdict = `Allow`
  - 项目退出 `Phase 1: Recovery`
  - 项目进入 `Phase 2: Controlled Development`
  - 决策状态切换为 `Go`
- 下一唯一任务：
  - 待定义

本任务完成标准：

1. 明确当前 software HIL recovery smoke 的主要随机源或不确定性来源
2. 如果可以在不改 benchmark 语义的前提下做最小修复，就限定在恢复期边界内完成
3. 至少基于两次 recovery smoke 证据，重新判断该路径是否仍只能表述为“可复验”，还是可升级为更强的确定性结论
4. 同步更新治理文档与 bootstrap 文档，但不得把 `P3-软件 HIL` 写成 `P3-真板 HIL 已完成`

注意：

- `T12` 不得静默改动正式 benchmark 口径、baseline 集合或 ParamMapper 主线语义
- `T12` 不得把 `board_backend.py` 的 placeholder 语义改写成真板完成
- `T12` 不得借机扩写 `.tflite` runtime、teacher-representation 或真板任务
- `T12` 不得顺手做 `runs/`、`artifacts/`、`__pycache__/` 的大规模清理
