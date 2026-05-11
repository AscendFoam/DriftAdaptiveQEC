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
- [x] T27: Teacher diagnostics path audit and mechanism-evidence repair plan
  - Task package: `docs/tasks/Phase2/T27_teacher_diagnostics_path_audit.md`
  - Review output: `docs/review/T27_teacher_diagnostics_path_audit.md`
  - Captain verdict: `PASS_WITH_WARNINGS`
  - R10 narrowed: hybrid path uses broadcast teacher features while scalar explain diagnostics require `scalar_feature_dim > 0`; data is not generated for current hybrid path, and downstream CSV coercion masks absence as `0.0`
  - R20 narrowed: independent fast-loop correction saturation path; current T24 `0.0` is not caused by teacher diagnostics dead path
- [ ] T28: Teacher diagnostics missing-vs-zero semantics repair and minimal smoke
  - Task package: `docs/tasks/Phase2/T28_teacher_diagnostics_semantics_repair.md`
- [ ] T29: Paper-inspired statcalib branch design gate, no long run until approved
  - Task package: pending
- [ ] T36: `seed=20260429` failure-mechanism diagnosis, bounded no-new-branch scope
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

`T28: Teacher diagnostics missing-vs-zero semantics repair and minimal smoke`

状态说明：

- `T27` 已完成只读 path audit，Captain verdict = `PASS_WITH_WARNINGS`
- `R10` 已缩窄：当前 T24 `hybrid_residual_b` 使用 broadcast teacher 特征；`tiny_cnn.py` explain 只在 `scalar_feature_dim > 0` 时产出 teacher scalar diagnostics，因此当前 hybrid path 的 teacher diagnostics 是 `data not generated`
- `R10` 仍未修复：下游 aggregation / CSV 会把部分缺失值压成 `0.0`，掩盖 `not generated` 与 `true zero`
- `R20` 已缩窄：`correction_saturation_rate_mean` 走独立 fast-loop saturation counter 路径，不与 teacher diagnostics 共享死路径；当前 T24 更像现参数区间下未触发 saturation
- `T28` 只修 teacher diagnostics missing-vs-zero 语义，并做最小 smoke 验证；不扩 formal benchmark、baseline/scenario、`.tflite` 或真板范围

为什么现在做它：

1. `T27` 已经把 teacher diagnostics 缺口定位到当前 teacher feature layout 与 explain 机制之间的语义不匹配。
2. 如果不先修复 `not generated` 与 `true zero` 的输出语义，后续机制分析、statcalib 设计或论文 claim ledger 都会继续继承模糊指标。
3. T28 可以用最小代码/报告语义修复和最小 smoke 收口 R10 的可观察性问题，而不扩大 benchmark 口径。
4. `R20` 不需要在 T28 中扩大 stress run；只需保持 T27 的缩窄结论，不把当前零值当作全局机制结论。

## Captain Output For Current Task

1. 当前唯一任务：`T28`
2. `T27` 已按 `PASS_WITH_WARNINGS` 收口。
3. T27 accepted / narrowed warning：
   - R20 correction saturation structural zero = 独立 fast-loop path；当前 T24 零值不再按 teacher diagnostics dead path 处理，但不关闭 R20 的全局触发性问题。
4. T27 deferred warnings：
   - R10 hybrid teacher diagnostics = `data not generated`，需要后续 repair。
   - downstream CSV `0.0` coercion masks missing-vs-zero semantics，需要后续 repair。
5. T28 任务包：`docs/tasks/Phase2/T28_teacher_diagnostics_semantics_repair.md`

## Done Criteria For T28

1. 最小修复 teacher diagnostics 输出语义，使 downstream 报告能区分 `not generated` 与 `true zero`。
2. 明确当前 broadcast teacher path 与 scalar teacher diagnostics 的支持边界，避免把未生成诊断写成已解释机制。
3. 用最小 smoke 验证修复后的字段语义；可新增 T28 专属 run dir，但不得扩展 formal benchmark 口径。
4. 更新必要的 review/for-human 文档和任务包 Worker output。
5. 不新增 baseline/scenario，不实现 statcalib，不触碰 `.tflite` runtime 或真板路径，不改写 T15/T24 历史 outputs。
