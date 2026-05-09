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

`T16: P4 benchmark evidence review and next-gate decision`

为什么现在做它：

1. `T15` review verdict = `PASS_WITH_WARNINGS`，且无 blocking issue。
2. `T15` 已按 `T14` matrix 完成双场景、五模式、`repeats=2` 的 development bounded run。
3. Review warning 已分类：N1 accepted 并由 Captain 修正文档状态；N2 deferred 给 `T16`；N3 accepted 作为 strong-baseline config 下的预期设计后果。
4. 下一步应先做 gate review，判断是否继续扩大 P4 benchmark、转向环境 manifest，或暂停 P4 扩展。

## Captain Output For Current Task

1. 当前唯一任务：`T16`
2. Worker 任务包：`docs/tasks/Phase2/T16_p4_evidence_gate_review.md`
3. Allowed files：
   - `docs/tasks/Phase2/T16_p4_evidence_gate_review.md`
   - `docs/review/T16_p4_evidence_gate_review.md`
   - `docs/04_task_board.md`
   - `docs/07_handoff.md`
   - `docs/08_risks_and_open_questions.md`
   - `docs/05_decision_log.md`
4. Forbidden scope：
   - 不运行新的 benchmark
   - 不修改代码或 config
   - 不把 `T15` bounded run 升级为正式四场景 formal benchmark 结论
   - 不把 `mock-backed` 结果写成 `real_board` 或 `.tflite` 验收
   - 不自动领取 `T17` 或扩大 P4 run
5. Verification：
   - 只读审查 `T14 + T15` 的 protocol、run evidence、review warning 与风险记录
   - 输出 `docs/review/T16_p4_evidence_gate_review.md`
   - 结论只能是 `Allow` / `Conditional` / `Block`
6. Docs to update：
   - 新增 `docs/review/T16_p4_evidence_gate_review.md`
   - 更新 `docs/04_task_board.md`
   - 更新 `docs/07_handoff.md`
   - 更新 `docs/08_risks_and_open_questions.md`
   - 若改变决策状态则更新 `docs/05_decision_log.md`

## Done Criteria For T16

1. 读取 `docs/review/T15_frozen_smoke_review.md` 并处理 warning：
   - N2 teacher diagnostics 全零必须进入 gate 判断或 deferred risk
   - N3 delta rows 为 null 应作为 config 设计后果解释清楚
2. 判断当前双场景 bounded evidence 是否支持继续扩大 P4 benchmark。
3. 明确下一步建议：扩大到剩余场景、转向 manifest、或暂停 P4。
4. 不产生新的 benchmark run。
5. 不把 `T15` 写成正式四场景 frozen benchmark 已恢复。
