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
- [ ] T25: P4 formal evidence gate review and result-boundary update
  - Task package: `docs/tasks/Phase2/T25_p4_formal_evidence_gate_review.md`

### Milestone 2I: Mechanism Evidence Hardening

- [ ] T26: Calibration/statcalib baseline feasibility gate and minimal design plan
  - Task package: pending
- [ ] T27: Teacher diagnostics path audit and mechanism-evidence repair plan
  - Task package: pending
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

`T25: P4 formal evidence gate review and result-boundary update`

状态说明：

- `T24` Worker 已完成：四场景、五模式、`repeats=2` formal software revalidation 执行成功
- `docs/review/T24_review.md` verdict = `PASS_WITH_WARNINGS`，blocking issues = none
- `T24` run dir: `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743`
- `T24` verification: `missing_runs = []`, 20/20 scenario/mode pairs `coverage = 1.0`, 40 repeat-runs
- `T24` 仍为 `mock-backed` software HIL，不是 `.tflite` runtime、`real_board` HIL 或 paper-grade expanded benchmark
- Captain 已接受 T24 为 `PASS_WITH_WARNINGS`；N2 accepted，N1/N3 deferred 到 risks / T25-T27 后续收口

为什么现在做它：

1. `T24` Worker 已完成 formal software revalidation 执行。
2. `T24` verification 全部通过：`missing_runs = []`、`coverage = 1.0`、`raw_rows = 40`。
3. `T24` adversarial review 无 blocking issue，但留下 correction saturation structural zero 与 teacher diagnostics zero-row 机制缺口。
4. T24 仍不得写成 `.tflite` runtime、`real_board` HIL 或 paper-grade expanded benchmark。

## Captain Output For Current Task

1. 当前唯一任务：`T25`
2. `T24` Worker 已完成，reviewer verdict = `PASS_WITH_WARNINGS`，Captain 已收口
3. Worker 产出：
   - Run dir: `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743`
   - `docs/P4_benchmark_formal_protocol.md` 已更新 T24 execution record (Section 15)
   - `docs/tasks/Phase2/T24_p4_formal_software_revalidation.md` 已更新 Worker output
   - Verification 全部通过：`missing_runs = []`, 20/20 rows `coverage = 1.0`, 40 repeat-runs
4. T25 任务包：`docs/tasks/Phase2/T25_p4_formal_evidence_gate_review.md`
5. 下一步：交给 Worker 执行只读 adversarial gate review，不启动新 benchmark

## Done Criteria For T25

1. 只读审查 T24 evidence pack、T24 review warning 与治理文档一致性。
2. 明确判断 T24 证据等级：能否作为 frozen-set formal software revalidation；不能升级到 `.tflite` runtime、`real_board` 或 paper-grade expanded benchmark。
3. 将 T24 warnings 分类为 accepted / deferred / rejected，并检查 deferred 项是否已进入 risks。
4. 推荐下一唯一任务，但不执行；优先考虑机制证据缺口（teacher diagnostics / correction saturation）或必要的边界任务。
5. 不运行 benchmark、不改代码、不改 config、不执行 cleanup、不调用硬件。
