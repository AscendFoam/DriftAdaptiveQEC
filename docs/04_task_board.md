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

- [x] T21: Phase 2 milestone review and next-phase decision
  - Task package: `docs/tasks/Phase2/T21_phase2_milestone_review.md`

### Milestone 2F: Real-Board Planning

- [x] T22: Real-board smoke execution plan with platform / AXI-map audit and quantitative acceptance thresholds
  - Task package: `docs/tasks/Phase2/T22_real_board_smoke_execution_plan.md`

### Milestone 2G: Formal Benchmark Readiness

- [x] T23: P4 formal benchmark protocol lock and evidence gap audit
  - Task package: `docs/tasks/Phase2/T23_p4_formal_benchmark_protocol_lock.md`

### Milestone 2H: Formal Benchmark Execution And Gate

- [ ] T24: P4 bounded formal software revalidation execution
  - Task package: `docs/tasks/Phase2/T24_p4_formal_software_revalidation.md`
- [ ] T25: P4 formal evidence gate review and result-boundary update
  - Task package: pending

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

`T24: P4 bounded formal software revalidation execution`

状态说明：

- `T23` review verdict = `PASS_WITH_WARNINGS`，blocking issues 为无，Captain 接受并标记完成
- `T24` 是执行任务，但只允许执行 `docs/P4_benchmark_formal_protocol.md` 锁定的 frozen-set software revalidation
- `T24` 仍不得写成 `.tflite` runtime、`real_board` HIL 或 paper-grade expanded benchmark

为什么现在做它：

1. `T23` 已锁定 formal / development / recovery 证据边界、formal matrix、baseline 集合、统计输出、compute budget 与 evidence pack。
2. `T23` gate = `GO_FOR_BOUNDED_FORMAL_SOFTWARE_REVALIDATION` + `NO_GO_FOR_SCOPE_EXPANSION_INSIDE_T24`。
3. `statcalib`、soft-information comparator、额外 drift family、CI-driven stopping、真实 `.tflite` runtime 与真板 smoke 都不是 `T24` 范围。
4. `T24` 应补齐当前最大软件证据缺口：历史 frozen-set 的四场景、五模式、`repeats=2` formal software revalidation。
5. `T24` 必须使用同一固定 `run_dir` 和完整 scenario/mode selection；如需切块，只允许按 repeat range 切块以保持 seed 语义。

## Captain Output For Current Task

1. 当前唯一任务：`T24`
2. Worker 任务包：`docs/tasks/Phase2/T24_p4_formal_software_revalidation.md`
3. Allowed files：
   - `docs/tasks/Phase2/T24_p4_formal_software_revalidation.md`
   - `docs/P4_benchmark_formal_protocol.md`
   - `docs/04_task_board.md`
   - `docs/07_handoff.md`
   - `docs/08_risks_and_open_questions.md`
   - `runs/p4_benchmark/T24_formal_software_revalidation_*`
4. Forbidden scope：
   - 不改源码
   - 不改 benchmark 口径、baseline 集合或 ParamMapper 语义
   - 不改 config 文件语义
   - 不运行训练、`.tflite` runtime、cleanup 或硬件命令
   - 不新增 `statcalib`、soft-information comparator、额外 scenario family 或 teacher-representation 长跑
   - 不把 mock-backed software formal revalidation 写成 `.tflite` runtime、`real_board` 或 paper-grade final benchmark
5. Verification：
   - 使用 `C:\ProgramData\anaconda3\python.exe`
   - 使用 `cnn_fpga/config/p4_multiscenario_strong_baselines.yaml`
   - 锁定 `4 scenarios x 5 modes x repeats=2`
   - 使用 `--paired-seeds`
   - 如需切块，只用 `--repeat-start/--repeat-stop`，不得按单场景切块改变 seed 语义
   - final evidence pack 必须包含 `launch_plan.json`、`progress.jsonl`、`summary.json`、`comparison.csv`、`delta.csv`、`teacher_scalar_diagnostics.csv`、`report.md` 和每个 repeat 的 `hil_summary.json` / `repeat_status.json`
   - 明确报告 `missing_runs`、coverage、实际可用统计字段与缺失字段
6. Docs to update：
   - 更新 `docs/04_task_board.md`
   - 更新 `docs/07_handoff.md`
   - 更新 `docs/08_risks_and_open_questions.md`

预期 Worker 产出：

- 一个固定 T24 run dir
- 完整或明确缺口化的 formal software revalidation evidence pack
- `docs/P4_benchmark_formal_protocol.md` 中的 T24 execution record
- `docs/tasks/Phase2/T24_p4_formal_software_revalidation.md` 中的 Worker output
- 明确声明本轮仍是 `mock-backed` software HIL formal revalidation，不是 `.tflite` runtime 或 `real_board` validation

## Done Criteria For T24

1. 使用 T24 任务包固定的 CLI shape 或等价 repeat-chunked shape 执行。
2. 不改源码、不改 config、不改 benchmark 语义。
3. `summary.json` 中 `missing_runs = []` 且所有 scenario/mode coverage 为 `1.0`，否则不得写成 completed formal revalidation。
4. `comparison.csv` 覆盖四场景、五模式、`repeats=2`。
5. 明确记录实际可用统计字段；若 `histogram_input_saturation_rate_mean`、`correction_saturation_rate_mean` 或 `fast_cycle_violation_rate_mean` 缺失，必须报告为缺口，不得静默省略。
6. 不把 `statcalib`、soft-information、额外 drift families、CI stopping、`.tflite` runtime 或真板 smoke 塞进本任务。
7. 完成后进入 reviewer；Captain 收口前不启动 `T25`。
