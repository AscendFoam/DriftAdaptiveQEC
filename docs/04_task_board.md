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

- [ ] T19: Bounded cleanup manifest for tracked cache files
  - Task package: `docs/tasks/Phase2/T19_tracked_cache_cleanup_manifest.md`

### Milestone 2D: Hardware Boundary Readiness

- [ ] T20: Real-board HIL readiness checklist without implementation claims
  - Task package: `docs/tasks/Phase2/T20_real_board_readiness_checklist.md`

## Current Unique Task

`T19: Bounded cleanup manifest for tracked cache files`

为什么现在做它：

1. `T18` review verdict = `PASS`，没有 blocking issue。
2. `T18` 已完成 `.tflite` export/runtime manifest 与 boundary smoke plan。
3. `T18` 的 non-blocking issue 只是推荐表述的 Markdown 格式问题，分类为 `accepted`，不影响交接。
4. 真实 `.tflite` runtime 仍未恢复，已作为 R12 保留；不应借当前轮次继续扩写 `.tflite` 实体能力。
5. 当前更适合进入 repo hygiene 的只读 cleanup manifest，先把已跟踪缓存/字节码文件清点与执行方案写清楚。

## Captain Output For Current Task

1. 当前唯一任务：`T19`
2. Worker 任务包：`docs/tasks/Phase2/T19_tracked_cache_cleanup_manifest.md`
3. Allowed files：
   - `docs/tasks/Phase2/T19_tracked_cache_cleanup_manifest.md`
   - `docs/cleanup_tracked_cache_manifest.md`
   - `docs/06_repo_noise_governance.md`
   - `docs/04_task_board.md`
   - `docs/07_handoff.md`
   - `docs/08_risks_and_open_questions.md`
4. Forbidden scope：
   - 不删除 `runs/`
   - 不删除 `artifacts/`
   - 不执行批量破坏性命令，除非 Captain 后续明确批准 cleanup 执行任务
   - 不改源码
5. Verification：
   - 只读清点
   - 可使用 `git ls-files` / `rg --files` 统计目标文件
   - 不执行删除、不执行 `git rm`
6. Docs to update：
   - 新增 `docs/cleanup_tracked_cache_manifest.md`
   - 更新 `docs/06_repo_noise_governance.md`
   - 更新 `docs/04_task_board.md`
   - 更新 `docs/07_handoff.md`
   - 更新 `docs/08_risks_and_open_questions.md`

## Done Criteria For T19

1. 产出 tracked cache cleanup manifest。
2. 明确目标文件类别、只读清点结果、cleanup 命令草案、回滚方式与验收标准。
3. 不执行删除，不执行 `git rm`，不改源码。
4. 不触碰 `runs/` 和 `artifacts/`。
5. 为后续 Captain 决定是否执行物理 cleanup 留出清晰边界。
