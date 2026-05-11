# Task Board

本文件是当前仓库的任务主状态。后续 Worker 只能领取 `Current Unique Task` 指向的单个任务包；Captain 完成整合前，不自动领取下一项。

全局建议：运行代码可以使用conda的DLEnv环境(重环境)，也可以直接使用conda的默认python环境(轻环境)。

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

- [x] T21: Phase 2 milestone review and next-phase decision
  - Task package: `docs/tasks/Phase2/T21_phase2_milestone_review.md`

### Milestone 2F: Real-Board Planning

- [x] T22: Real-board smoke execution plan with platform / AXI-map audit and quantitative acceptance thresholds
  - Task package: `docs/tasks/Phase2/T22_real_board_smoke_execution_plan.md`

### Milestone 2G: Formal Benchmark Readiness

- [x] T23: P4 formal benchmark protocol lock and evidence gap audit
  - Task package: `docs/tasks/Phase2/T23_p4_formal_benchmark_protocol_lock.md`

### Milestone 2H: Formal Benchmark Execution And Gate

- [x] T24: P4 bounded formal software revalidation execution
  - Task package: `docs/tasks/Phase2/T24_p4_formal_software_revalidation.md`
  - Run dir: `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743`
  - `missing_runs = []`, all 20 scenario/mode pairs `coverage = 1.0`, 40 repeat-runs completed
  - All four scenarios won by `hybrid_residual_b`; runner-up = `ukf` in all four
  - Teacher diagnostics still all-zero (deferred mechanism-analysis gap)
  - Captain verdict on `docs/review/T24_review.md`: `PASS_WITH_WARNINGS`
  - Mock-backed software HIL only
- [x] T25: P4 formal evidence gate review and result-boundary update
  - Task package: `docs/tasks/Phase2/T25_p4_formal_evidence_gate_review.md`
  - Review output: `docs/review/T25_p4_formal_evidence_gate_review.md`
  - Captain verdict: `PASS_WITH_WARNINGS`
  - Boundary: T24 may be treated as completed frozen-set formal software revalidation, but still mock-backed software HIL only
  - Warning classification: N1 correction saturation structural zero = `deferred` / R20; N2 task-board environment note = `accepted`; N3 teacher diagnostics header-only = `deferred` / R10
  - Next-task recommendation accepted: `T27`

### Milestone 2I: Mechanism Evidence Hardening

- [ ] T26: Calibration/statcalib baseline feasibility gate and minimal design plan
  - Task package: pending
- [ ] T27: Teacher diagnostics path audit and mechanism-evidence repair plan
  - Task package: `docs/tasks/Phase2/T27_teacher_diagnostics_path_audit.md`
- [ ] T28: `seed=20260429` failure-mechanism diagnosis, bounded no-new-branch scope
  - Task package: pending
- [ ] T29: Paper-inspired statcalib branch design gate, no long run until approved
  - Task package: pending

### Milestone 2J: Reproducibility And Deployment Boundary

- [ ] T30: Training-chain portable dependency lock plan
  - Task package: pending
- [ ] T31: True `.tflite` runtime smoke, only if environment is available
  - Task package: pending
- [ ] T32: Tracked cache physical cleanup execution, only within T19 manifest
  - Task package: pending
- [ ] T33: Real-board smoke execution gate, only if hardware host and bitstream evidence are ready
  - Task package: pending

### Milestone 2K: Paper Assembly Readiness

- [ ] T34: Paper claim/evidence ledger and figure-table outline
  - Task package: pending
- [ ] T35: Paper draft skeleton and reviewer-risk audit
  - Task package: pending

Long-term objective:

以发表质量为最终目标，但最近任务仍按“证据口径锁定 -> 有界执行 -> gate review -> 机制解释 -> 复现/部署边界 -> 论文收口”的顺序推进。除 `Current Unique Task` 外，上述 pending 项只是路线图，不可直接执行。

## Current Unique Task

`T27: Teacher diagnostics path audit and mechanism-evidence repair plan`

状态说明：

- `T25` 已完成 gate review，Captain verdict = `PASS_WITH_WARNINGS`
- `T24` 可作为 completed frozen-set formal software revalidation，但仅限 `mock-backed` software HIL
- `T24` 不得外推为 `.tflite` runtime、`real_board` HIL 或 paper-grade expanded benchmark
- `T25` warning 分类已收口：N1 correction saturation structural zero = `deferred` / R20；N2 task-board environment note = `accepted`；N3 teacher diagnostics header-only = `deferred` / R10
- `T25` 建议下一唯一任务为 `T27`，Captain 接受该优先级
- `T27` 只做 teacher diagnostics 路径审计和机制证据修复计划，不运行新 benchmark、不改代码、不补新 baseline

为什么现在做它：

1. `T15` 到 `T24` 多次出现 teacher diagnostics 全零 / header-only 现象。
2. `T25` 已确认该问题不阻塞 T24 LER ranking，但会限制 `hybrid_residual_b` 的机制解释可信度。
3. `R10` 是当前最长 deferred 链；`R20` correction saturation structural zero 可在同一只读审计中作为相邻 zero-metric 问题检查。
4. 在机制证据路径没有厘清前，不宜先推进 statcalib 设计、论文 claim ledger、`.tflite` runtime 或真板执行。

## Captain Output For Current Task

1. 当前唯一任务：`T27`
2. `T25` 已按 `PASS_WITH_WARNINGS` 收口；本轮不需要重复 Claude review，因为 T25 本身就是 review 任务。
3. T25 accepted warning：
   - N2 `docs/04_task_board.md` environment-note warning = `accepted`，归为 Captain 治理同步提示。
4. T25 deferred warnings：
   - N1 `correction_saturation_rate_mean` structural zero = `deferred`，继续挂 R20。
   - N3 `teacher_scalar_diagnostics.csv` header-only / teacher diagnostics 全零 = `deferred`，继续挂 R10。
5. T27 任务包：`docs/tasks/Phase2/T27_teacher_diagnostics_path_audit.md`

## Done Criteria For T27

1. 只读追踪 teacher diagnostics 的生成、聚合、写出路径，并定位 `teacher_scalar_diagnostics.csv` header-only 的直接原因或最小可验证假设。
2. 明确分类问题性质：数据未生成、聚合/写出 bug、当前模式 not applicable、或需要后续源码修复任务。
3. 相邻检查 `correction_saturation_rate_mean` structural zero 的指标路径，判断是否与 teacher diagnostics 共享 dead path 或属于独立问题。
4. 输出最小机制证据修复计划与下一任务建议，但不执行修复、不运行 benchmark。
5. 不运行 benchmark、不改代码、不改 config、不执行 cleanup、不调用硬件、不新增 run dir。
