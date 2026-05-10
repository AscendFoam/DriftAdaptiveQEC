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

- [x] T14: P4 frozen benchmark protocol audit and bounded run plan
  - Task package: `docs/tasks/Phase2/T14_p4_frozen_benchmark_protocol_audit.md`
- [x] T15: P4 multi-scenario frozen baseline bounded smoke
  - Task package: `docs/tasks/Phase2/T15_p4_multiscenario_frozen_smoke.md`
- [x] T16: P4 benchmark evidence review and next-gate decision
  - Task package: `docs/tasks/Phase2/T16_p4_evidence_gate_review.md`

### Milestone 2B: Environment Manifests

- [x] T17: Training-chain independent manifest and bootstrap
  - Task package: `docs/tasks/Phase2/T17_training_manifest_bootstrap.md`
- [x] T18: TFLite export/runtime manifest and boundary smoke plan
  - Task package: `docs/tasks/Phase2/T18_tflite_manifest_and_smoke_plan.md`

### Milestone 2C: Repository Hygiene

- [x] T19: Bounded cleanup manifest for tracked cache files
  - Task package: `docs/tasks/Phase2/T19_tracked_cache_cleanup_manifest.md`

### Milestone 2D: Hardware Boundary Readiness

- [ ] T20: Real-board HIL readiness checklist without implementation claims
  - Task package: `docs/tasks/Phase2/T20_real_board_readiness_checklist.md`

## Current Unique Task

`T20: Real-board HIL readiness checklist without implementation claims`

为什么现在做它：

1. `T19` review verdict = `PASS`，没有 blocking issue。
2. `T19` 已完成 tracked cache cleanup manifest，并把已跟踪 `.pyc` / `__pycache__/` 的边界和回滚方案固定下来。
3. `T19` 的 non-blocking issue 仅是命令写法与统计口径说明，均为 `accepted`，不影响 cleanup 边界。
4. 物理 cleanup 仍未执行，且必须作为后续独立任务处理，不能借当前轮次顺手落地。
5. 真板路径仍是 placeholder 语义，当前更适合进入 real-board readiness checklist，先固定设备、地址、权限、日志与禁止表述。
6. `T20` 是当前最小的下一步有界任务，可继续保持边界诚实而不扩写真板完成事实。

## Captain Output For Current Task

1. 当前唯一任务：`T20`
2. Worker 任务包：`docs/tasks/Phase2/T20_real_board_readiness_checklist.md`
3. Allowed files：
   - `docs/tasks/Phase2/T20_real_board_readiness_checklist.md`
   - `docs/real_board_hil_readiness.md`
   - `docs/03_hil_p4_boundary_audit.md`
   - `docs/04_task_board.md`
   - `docs/07_handoff.md`
   - `docs/08_risks_and_open_questions.md`
4. Forbidden scope：
   - 不改 `cnn_fpga/hwio/board_backend.py`
   - 不改 `cnn_fpga/hwio/fpga_driver.py`
   - 不写真实板级完成
   - 不运行需要硬件设备的命令
   - 不把 mock backend 证据外推到 real board
5. Verification：
   - 只读审计
   - 不调用硬件
   - 核对 placeholder / mock / real-board 语义边界
6. Docs to update：
   - 更新 `docs/03_hil_p4_boundary_audit.md`
   - 更新 `docs/04_task_board.md`
   - 更新 `docs/07_handoff.md`
   - 更新 `docs/08_risks_and_open_questions.md`

预期 Worker 只读产出：

- `docs/real_board_hil_readiness.md`
- placeholder 证据清点与禁止表述清单
- 真板 readiness 前置条件与最小 smoke 验收标准
- 未修改 `board_backend.py`

## Done Criteria For T20

1. 产出 real-board HIL readiness checklist。
2. 明确当前 `board_backend.py` / `fpga_driver.py` 的 placeholder 证据。
3. 明确真板任务前置条件、设备/地址/权限/日志需求与最小 smoke 验收标准。
4. 不修改真板代码，不调用硬件命令。
5. 不把 `mock` backend、软件 HIL 或计划项写成 real-board HIL 已完成。
