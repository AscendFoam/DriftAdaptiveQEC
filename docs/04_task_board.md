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

- [x] T20: Real-board HIL readiness checklist without implementation claims
  - Task package: `docs/tasks/Phase2/T20_real_board_readiness_checklist.md`

### Milestone 2E: Phase 2 Gate

- [ ] T21: Phase 2 milestone review and next-phase decision
  - Task package: `docs/tasks/Phase2/T21_phase2_milestone_review.md`

## Current Unique Task

`T21: Phase 2 milestone review and next-phase decision`

为什么现在做它：

1. `T20` adversarial review verdict = `PASS`，没有 blocking issue。
2. `T20` 已完成 real-board readiness checklist，但该产物仍只是 readiness / acceptance criteria，不是真板验证。
3. `T20` 的 non-blocking issues 影响后续真板执行任务设计，分类为 `deferred` 并写入 risks/open questions。
4. `T14` 至 `T20` 已覆盖 Phase 2 原计划的 benchmark、manifest、repo hygiene 与 hardware readiness 任务。
5. 直接进入真板 smoke、物理 cleanup 或新的 benchmark 前，需要先做一次 Phase 2 milestone gate。
6. `T21` 只做只读 milestone review 和下一阶段决策，不运行 benchmark、不执行 cleanup、不调用硬件。

## Captain Output For Current Task

1. 当前唯一任务：`T21`
2. Worker 任务包：`docs/tasks/Phase2/T21_phase2_milestone_review.md`
3. Allowed files：
   - `docs/tasks/Phase2/T21_phase2_milestone_review.md`
   - `docs/review/T21_phase2_milestone_review.md`
   - `docs/04_task_board.md`
   - `docs/05_decision_log.md`
   - `docs/07_handoff.md`
   - `docs/08_risks_and_open_questions.md`
4. Forbidden scope：
   - 不运行新的 benchmark
   - 不执行物理 cleanup
   - 不调用硬件或真板命令
   - 不改源码、配置或 benchmark 口径
   - 不把 `T15` development run 写成 formal benchmark
   - 不把 `T20` readiness checklist 写成 real-board validation
5. Verification：
   - 只读 milestone review
   - 核对 `T14`-`T20` 证据等级与剩余风险
   - 输出 `Allow` / `Conditional` / `Block` gate decision
6. Docs to update：
   - 更新 `docs/04_task_board.md`
   - 更新 `docs/05_decision_log.md`
   - 更新 `docs/07_handoff.md`
   - 更新 `docs/08_risks_and_open_questions.md`

预期 Worker 只读产出：

- `docs/review/T21_phase2_milestone_review.md`
- Phase 2 completed task summary (`T14`-`T20`)
- evidence level assessment
- remaining blockers / warnings
- gate decision 与推荐下一唯一任务

## Done Criteria For T21

1. 产出 Phase 2 milestone review。
2. 覆盖 `T14` 至 `T20` 的完成状态、证据等级与边界。
3. 明确 remaining blockers / warnings，尤其是 P4 formal benchmark、`.tflite` runtime、physical cleanup、real-board readiness。
4. 给出 `Allow` / `Conditional` / `Block` gate decision。
5. 推荐下一唯一任务，但不执行下一任务。
