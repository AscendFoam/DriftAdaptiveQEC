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
  - Task package: `docs/tasks/Phase2/T26_statcalib_feasibility_gate.md`
- [x] T27: Teacher diagnostics path audit and mechanism-evidence repair plan
  - Task package: `docs/tasks/Phase2/T27_teacher_diagnostics_path_audit.md`
  - Review output: `docs/review/T27_teacher_diagnostics_path_audit.md`
  - Captain verdict: `PASS_WITH_WARNINGS`
  - R10 narrowed: hybrid path uses broadcast teacher features while scalar explain diagnostics require `scalar_feature_dim > 0`; data is not generated for current hybrid path, and downstream CSV coercion masks absence as `0.0`
  - R20 narrowed: independent fast-loop correction saturation path; current T24 `0.0` is not caused by teacher diagnostics dead path
- [x] T28: Teacher diagnostics missing-vs-zero semantics repair and minimal smoke
  - Task package: `docs/tasks/Phase2/T28_teacher_diagnostics_semantics_repair.md`
  - Review output: `docs/review/T28_review.md`
  - Captain verdict: `PASS_WITH_WARNINGS`
  - R10 further narrowed: current outputs explicitly distinguish `not_applicable` and `not_generated`; mechanism evidence still not fully repaired
  - R21 closed for current writer semantics: missing teacher diagnostics are no longer silently coerced to `0.0`
  - Deferred follow-up: duplicate markdown report header row in `_write_report()`
- [x] T29: P4 markdown report header cleanup after T28
  - Task package: `docs/tasks/Phase2/T29_p4_report_header_cleanup.md`
  - Review output: `docs/review/T29_review.md`
  - Captain verdict: `PASS`
  - Fixed duplicate old markdown report header in `_write_report()`
  - Verification: `py_compile` passed; `_write_report()` static shape check showed `header_rows=1`, `column_counts=[12, 12, 12]`
  - Non-blocking `.pyc` side-effect is not a technical change and must not be committed as task output
- [ ] T30: Paper-inspired statcalib branch design gate, no long run until approved
  - Task package: pending
- [ ] T36: `seed=20260429` failure-mechanism diagnosis, bounded no-new-branch scope
  - Task package: pending

### Milestone 2J: Reproducibility And Deployment Boundary

- [ ] T31: Training-chain portable dependency lock plan
  - Task package: pending
- [ ] T32: True `.tflite` runtime smoke, only if environment is available
  - Task package: pending
- [ ] T33: Tracked cache physical cleanup execution, only within T19 manifest
  - Task package: pending
- [ ] T37: Real-board smoke execution gate, only if hardware host and bitstream evidence are ready
  - Task package: pending

### Milestone 2K: Paper Assembly Readiness

- [ ] T34: Paper claim/evidence ledger and figure-table outline
  - Task package: pending
- [ ] T35: Paper draft skeleton and reviewer-risk audit
  - Task package: pending

Long-term objective:

以发表质量为最终目标，但最近任务仍按“证据口径锁定 -> 有界执行 -> gate review -> 机制解释 -> 复现/部署边界 -> 论文收口”的顺序推进。除 `Current Unique Task` 外，上述 pending 项只是路线图，不可直接执行。

## Current Unique Task

`T26: Calibration/statcalib baseline feasibility gate and minimal design plan`

状态说明：

- `T29` 已完成并通过 independent review，Captain verdict = `PASS`
- `T29` 修复了 `_write_report()` 中重复旧 markdown header 的人读 report 格式问题
- `R22` 可收口：当前 P4 markdown report header / separator / data row 列数一致
- `R10` 仍未关闭：teacher diagnostics 可观察性已改善，但机制证据仍不完整
- `R20` 仍未关闭：correction saturation structural zero 仍需后续独立 edge/stress 判断
- `R23` 仍未关闭：aggregation/report writer 缺少 focused tests 的风险仍存在
- `T26` 只做 calibration/statcalib baseline feasibility gate 与最小设计计划，不实现 comparator、不运行 benchmark、不新增长跑

为什么现在做它：

1. T24/T25 已完成 frozen-set formal software revalidation 与 result-boundary gate，但没有纳入 calibration/statcalib comparator。
2. T27/T28/T29 已先完成 teacher diagnostics 路径审计、missing-vs-zero 语义修复和人读 report 格式修复，避免 statcalib 任务继承破损指标输出。
3. 直接实现 statcalib 或启动长跑会扩大 benchmark 范围；当前更稳妥的是先做 feasibility gate、输入/输出 contract、最小设计与 go/no-go。
4. T36 seed failure diagnosis 仍重要，但它更适合在 comparator feasibility 边界明确后再排期。

## Captain Output For Current Task

1. 当前唯一任务：`T26`
2. `T29` 已按 `PASS` 收口。
3. T29 accepted / non-blocking warning：
   - N1 tracked `.pyc` side-effect = `accepted as known repo-noise side effect / rejected as technical signal`；不写入新风险，不作为有意义改动提交，继续按 T19/T28 tracked-cache governance 处理。
4. T29 deferred warnings：
   - none
5. T29 rejected warnings：
   - none beyond the `.pyc` technical-signal rejection above
6. T26 任务包：`docs/tasks/Phase2/T26_statcalib_feasibility_gate.md`

## Done Criteria For T26

1. 只读审计现有 benchmark / baseline / ParamMapper / teacher diagnostics / protocol 文档，判断 calibration/statcalib comparator 是否具备最小可实现前提。
2. 输出一份 feasibility gate 文档，明确 adopted / deferred / rejected 的 statcalib 设计项。
3. 给出最小设计计划：输入、输出、配置开关、指标、验证方式、与现有 frozen-set benchmark 的隔离方式。
4. 不实现 statcalib comparator，不运行 benchmark，不新增 run dir，不改 baseline 集合、scenario 集合、seed/repeat policy 或 formal protocol。
5. 更新 T26 review/for-human 文档与任务包 Worker output；若发现 blocker，明确交回 Captain 判断。
