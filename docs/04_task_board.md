# Task Board

本文件是当前仓库的任务主状态。后续 Worker 只能领取 `Current Unique Task` 指向的单个任务包；Captain 完成整合前，不自动领取下一项。

## Workflow State

- 当前阶段：`Phase 2: Controlled Development`
- 决策状态：`Go`
- 状态来源：
  - `docs/review/T13_recovery_exit_review.md` verdict = `Allow`
  - `docs/02_experiment_plan.md`
  - `docs/reference/AI_coding_workflow.md`
- 当前任务原则：
  - 每轮只推进一个 bounded task
  - 每个任务包必须有 Allowed files / Forbidden scope / Verification / Docs to update
  - 不把 `mock`、`stub`、`placeholder`、计划项或未来能力写成完成事实

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

### Milestone 2A: Benchmark Evidence Hardening

- [ ] T14: P4 frozen benchmark protocol audit and bounded run plan
  - Task package: `docs/tasks/Phase2/T14_p4_frozen_benchmark_protocol_audit.md`
- [ ] T15: P4 multi-scenario frozen baseline bounded smoke
  - Task package: `docs/tasks/Phase2/T15_p4_multiscenario_frozen_smoke.md`
- [ ] T16: P4 benchmark evidence review and next-gate decision
  - Task package: `docs/tasks/Phase2/T16_p4_evidence_gate_review.md`

### Milestone 2B: Environment Manifests

- [ ] T17: Training-chain independent manifest and bootstrap
  - Task package: `docs/tasks/Phase2/T17_training_manifest_bootstrap.md`
- [ ] T18: TFLite export/runtime manifest and boundary smoke plan
  - Task package: `docs/tasks/Phase2/T18_tflite_manifest_and_smoke_plan.md`

### Milestone 2C: Repository Hygiene

- [ ] T19: Bounded cleanup manifest for tracked cache files
  - Task package: `docs/tasks/Phase2/T19_tracked_cache_cleanup_manifest.md`

### Milestone 2D: Hardware Boundary Readiness

- [ ] T20: Real-board HIL readiness checklist without implementation claims
  - Task package: `docs/tasks/Phase2/T20_real_board_readiness_checklist.md`

## Current Unique Task

`T14: P4 frozen benchmark protocol audit and bounded run plan`

为什么现在做它：

1. `T9` 已完成 `single-scenario + four-mode + repeats=1` 的 P4 recovery smoke。
2. `T13` 已允许项目进入 `Go`，但 `Go` 只代表可以继续做 bounded 开发任务。
3. `docs/02_experiment_plan.md` 明确禁止无准备地启动长时间正式多场景 benchmark。
4. 因此下一步应先审计正式/开发级 P4 benchmark 口径、确认最小 bounded 扩展方案，再决定是否运行更强证据。

## Captain Output For Current Task

1. 当前唯一任务：`T14`
2. Worker 任务包：`docs/tasks/Phase2/T14_p4_frozen_benchmark_protocol_audit.md`
3. Allowed files：
   - `docs/tasks/Phase2/T14_p4_frozen_benchmark_protocol_audit.md`
   - `docs/P4_benchmark_development_protocol.md`
   - `docs/04_task_board.md`
   - `docs/07_handoff.md`
   - `docs/08_risks_and_open_questions.md`
4. Forbidden scope：
   - 不改 `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py`
   - 不改 `cnn_fpga/decoder/param_mapper.py`
   - 不改正式 benchmark baseline 集合或场景定义
   - 不启动长跑 benchmark
   - 不把 `mock-backed` 结果写成 `real_board` 或 `.tflite` 验收
5. Verification：
   - 只读检查相关 P4 config、runner 参数与既有 run evidence
   - 可运行轻量只读命令，例如 `Select-String` / `Get-Content`
   - 不要求产生新 `runs/` 结果
6. Docs to update：
   - 新增或更新 `docs/P4_benchmark_development_protocol.md`
   - 根据结论更新 `docs/07_handoff.md`
   - 必要时更新 `docs/08_risks_and_open_questions.md`

## Done Criteria For T14

1. 明确正式 P4 frozen benchmark 与 recovery smoke 的区别。
2. 明确下一步可运行的 bounded smoke 参数，包括 scenario、mode、repeat、seed pairing、解释器与配置。
3. 明确不允许本任务直接改变 benchmark 口径。
4. 输出能被 `T15` 直接复用的 Worker 运行计划。
5. 未修改代码、未产生新事实性 benchmark 结论。

