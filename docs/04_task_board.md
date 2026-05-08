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

`T15: P4 multi-scenario frozen baseline bounded smoke`

为什么现在做它：

1. `T14` review verdict = `PASS`，且无 blocking issue。
2. `docs/P4_benchmark_development_protocol.md` 已固定 `T15` 的 bounded matrix。
3. `T15` 只允许执行双场景、五模式、`repeats=2` 的 development bounded run，不恢复完整四场景 formal benchmark。
4. 该任务直接承接 `T9` 的单场景四模式 recovery smoke，用于给 `T16` gate review 提供更强但仍有边界的 P4 evidence。

## Captain Output For Current Task

1. 当前唯一任务：`T15`
2. Worker 任务包：`docs/tasks/Phase2/T15_p4_multiscenario_frozen_smoke.md`
3. Allowed files：
   - `docs/tasks/Phase2/T15_p4_multiscenario_frozen_smoke.md`
   - `docs/P4_benchmark_development_protocol.md`
   - `docs/P4_benchmark_recovery_bootstrap.md`
   - `docs/04_task_board.md`
   - `docs/07_handoff.md`
   - `docs/08_risks_and_open_questions.md`
   - 新产生的 `runs/p4_benchmark/...` 输出目录
4. Forbidden scope：
   - 不改 `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py`
   - 不改 `cnn_fpga/decoder/param_mapper.py`
   - 不改正式 benchmark baseline 集合或场景定义
   - 不改 `cnn_fpga/config/p4_multiscenario_strong_baselines.yaml`
   - 不运行超出 `docs/P4_benchmark_development_protocol.md` Section 6 的 matrix
   - 不把 bounded development run 写成正式四场景 formal benchmark
   - 不把 `mock-backed` 结果写成 `real_board` 或 `.tflite` 验收
5. Verification：
   - 按 `docs/P4_benchmark_development_protocol.md` Section 7 的命令运行
   - 检查新 run 的 `summary.json`、`comparison.csv`、`delta.csv`、`report.md`、`progress.jsonl`
   - 检查各 repeat `hil_summary.json` 中 backend / artifact / inference mode 标签
6. Docs to update：
   - 更新 `docs/P4_benchmark_development_protocol.md`
   - 更新 `docs/P4_benchmark_recovery_bootstrap.md`
   - 更新 `docs/07_handoff.md`
   - 必要时更新 `docs/08_risks_and_open_questions.md`

## Done Criteria For T15

1. 运行范围严格等于：
   - scenarios: `static_bias_theta`, `linear_ramp`
   - modes: `ekf`, `ukf`, `constant_residual_mu`, `rls_residual_b`, `hybrid_residual_b`
   - repeats: `2`
   - seed policy: `--paired-seeds`
   - config: `cnn_fpga/config/p4_multiscenario_strong_baselines.yaml`
2. 未修改 benchmark runner、config、baseline 集合、场景定义或 ParamMapper 语义。
3. 记录新 run dir 与关键 summary/comparison 字段。
4. 明确写清该结果是 `development bounded run`，不是正式四场景 formal benchmark。
5. 为 `T16` gate review 留下足够证据。
