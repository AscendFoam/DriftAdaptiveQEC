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
- [ ] T18: TFLite export/runtime manifest and boundary smoke plan
  - Task package: `docs/tasks/Phase2/T18_tflite_manifest_and_smoke_plan.md`

### Milestone 2C: Repository Hygiene

- [ ] T19: Bounded cleanup manifest for tracked cache files
  - Task package: `docs/tasks/Phase2/T19_tracked_cache_cleanup_manifest.md`

### Milestone 2D: Hardware Boundary Readiness

- [ ] T20: Real-board HIL readiness checklist without implementation claims
  - Task package: `docs/tasks/Phase2/T20_real_board_readiness_checklist.md`

## Current Unique Task

`T18: TFLite export/runtime manifest and boundary smoke plan`

为什么现在做它：

1. `T17` review verdict = `PASS`，没有 blocking issue。
2. `T17` 已完成训练链独立 bootstrap，并明确没有把 `DLEnv` 写成跨机器保证。
3. `T17` reviewer 的 `torch` dev build 与未产出 `requirements-train.txt` 观察均为非阻塞；训练链可移植性后续再单开任务，不影响进入 `T18`。
4. `.tflite` 路径已有真实导出/runtime 与 `tflite_stub_v1` 两类代码路径，但当前 Phase 2 尚未重新复验真实 runtime。
5. 当前更适合先补 `.tflite` manifest / boundary smoke plan，再决定是否执行实际 smoke 或转入 cleanup / real-board readiness。

## Captain Output For Current Task

1. 当前唯一任务：`T18`
2. Worker 任务包：`docs/tasks/Phase2/T18_tflite_manifest_and_smoke_plan.md`
3. Allowed files：
   - `docs/tasks/Phase2/T18_tflite_manifest_and_smoke_plan.md`
   - `docs/TFLite_runtime_bootstrap.md`
   - `docs/04_task_board.md`
   - `docs/07_handoff.md`
   - `docs/08_risks_and_open_questions.md`
4. Forbidden scope：
   - 不改 `cnn_fpga/model/export.py`
   - 不改 `cnn_fpga/runtime/inference_service.py`
   - 不把 `.tflite.json` stub manifest 写成真实 `.tflite` runtime
   - 不改 HIL benchmark 口径
5. Verification：
   - 只读代码审计加环境探测
   - 如环境具备，可运行最小 `--help` 或 import smoke
   - 不得强行改代码绕过依赖
6. Docs to update：
   - 新增 `docs/TFLite_runtime_bootstrap.md`
   - 更新 `docs/04_task_board.md`
   - 更新 `docs/07_handoff.md`
   - 更新 `docs/08_risks_and_open_questions.md`

## Done Criteria For T18

1. 产出 `.tflite` export/runtime 独立 bootstrap 文档。
2. 明确真实 `.tflite` 与 `tflite_stub_v1` 的边界、依赖和 smoke 命令。
3. 不改导出/runtime 代码，不改 benchmark 口径。
4. 若环境无法验证真实 `.tflite`，把阻塞项写清楚，不伪造成通过。
5. 为后续实际 `.tflite` smoke、repo cleanup 或 real-board readiness 留出清晰边界。
