# Task Board

本文件是当前仓库的任务主状态。后续 Worker 只能领取 `Current Unique Task` 指向的单个任务包；Captain 完成整合前，不自动领取下一项。

全局建议：运行代码可以使用conda的DLEnv环境(重环境)，也可以直接使用conda的默认python环境(轻环境)。

## Workflow State

- 当前阶段：`Phase 2: Controlled Development`
- 决策状态：`Go`
- 当前子模式：`Research Reality Recovery Mode`
- 状态来源：
  - `docs/review/T13_recovery_exit_review.md` verdict = `Allow`
  - `docs/02_experiment_plan.md`
  - `docs/reference/AI_coding_workflow.md`
- 子模式触发：
  - `2026-05-18` 用户明确要求按 `docs/reference/科研纠偏意见.md` 进入 recovery-first 推进
  - 当前优先级从“继续扩 prose”切换为“先冻结 claim/evidence/material truth，再补证据、图表、复现与边界缺口”
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

- [x] T26: Calibration/statcalib baseline feasibility gate and minimal design plan
  - Task package: `docs/tasks/Phase2/T26_statcalib_feasibility_gate.md`
  - Gate output: `docs/statcalib_feasibility_gate.md`
  - Review output: `docs/review/T26_review.md`
  - Captain verdict: `PASS`
  - Gate verdict: `CONDITIONAL_GO`
  - Boundary: statcalib is feasible only as a separate comparator lane; no silent insertion into the T24 frozen benchmark set
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
- [x] T30: Statcalib comparator interface contract and bounded implementation package
  - Task package: `docs/tasks/Phase2/T30_statcalib_interface_contract.md`
  - Review output: `docs/review/T30_review.md`
  - Captain verdict: `PASS`
  - Added interface-only `cnn_fpga/decoder/statcalib.py` with typed `StatCalibInput` / `StatCalibOutput` and focused tests
  - Verification: `unittest` passed (`Ran 6 tests`, `OK`); `py_compile` passed; no diff in `ParamMapper`, `SlowLoopRuntime`, P4 benchmark runner, or config
  - Boundary: statcalib is not integrated into slow-loop runtime or frozen benchmark evidence
- [x] T36: `seed=20260429` failure-mechanism diagnosis, bounded no-new-branch scope
  - Task package: `docs/tasks/Phase2/T36_seed20260429_failure_mechanism_diagnosis.md`
  - Diagnosis report: `docs/seed20260429_failure_diagnosis.md`
  - Review output: `docs/review/T36_review.md`
  - Captain verdict: `PASS`
  - Conclusion: existing artifacts narrow `seed=20260429` to a residual-amplitude / teacher-delta regime instability hypothesis, but do not expose per-window committed-parameter traces; no benchmark rerun or branch expansion occurred
- [x] T38: `seed=20260429` single-seed trace-export probe, bounded unchanged-semantics rerun
  - Task package: `docs/tasks/Phase2/T38_seed20260429_trace_export_probe.md`
  - Trace diagnosis: `docs/seed20260429_trace_export_diagnosis.md`
  - Review output: `docs/review/T38_review.md`
  - Captain verdict: `PASS`
  - Run root: `runs/T38_seed20260429_trace_probe_20260513`
  - Result: `4798` trace rows with required fields present; combined committed-`b` instability is trace-supported for `seed=20260429`, but still seed-bounded diagnostic evidence
  - Milestone gate: `docs/review/Milestone2I_review.md` verdict = `Conditional Allow`

### Milestone 2J: Reproducibility And Deployment Boundary

- [x] T31: Training-chain portable dependency lock plan
  - Task package: `docs/tasks/Phase2/T31_training_chain_portable_dependency_lock_plan.md`
  - Plan output: `docs/training_chain_portable_dependency_lock_plan.md`
  - Review output: `docs/review/T31_review.md`
  - Captain verdict: `PASS`
  - Result: training-chain local interpreter/package facts and CPU-vs-GPU lock strategy are documented; clean-environment rebuild remains unverified
- [x] T39: Training-chain CPU-only clean-environment draft lock and dry-run bootstrap
  - Task package: `docs/tasks/Phase2/T39_training_chain_cpu_cleanenv_draft_lock.md`
  - Bootstrap output: `docs/training_chain_cpu_cleanenv_bootstrap.md`
  - Review output: `docs/review/T39_review.md`
  - Captain verdict: `PASS`
  - Result: clean CPU-only environment, draft dependency lock, and dry-run/import-level bootstrap are verified; real clean-environment training execution remains unverified
- [x] T40: Training-chain CPU-only clean-environment minimal real-training smoke
  - Task package: `docs/tasks/Phase2/T40_training_chain_cpu_cleanenv_minimal_train_smoke.md`
  - Smoke output: `docs/training_chain_cpu_cleanenv_train_smoke.md`
  - Review output: `docs/review/T40_review.md`
  - Captain verdict: `PASS`
  - Result: clean CPU-only environment completed one real training smoke with isolated outputs; full training reproducibility and broader portability remain unverified
- [ ] T32: True `.tflite` runtime smoke, only if environment is available
  - Task package: pending
- [x] T33: Tracked cache physical cleanup execution, only within T19 manifest
  - Task package: `docs/tasks/Phase2/T33_tracked_cache_physical_cleanup_execution.md`
  - Review output: `docs/review/T33_review.md`
  - Captain verdict: `PASS`
  - Result: 116 tracked `.pyc` files across 9 manifest-listed `__pycache__` directories were removed from the Git index; `runs/`, `artifacts`, source, config, benchmark, `.tflite`, and hardware scope remained untouched
- [ ] T37: Real-board smoke execution gate, only if hardware host and bitstream evidence are ready
  - Task package: pending

### Milestone 2K: Paper Assembly Readiness

- [x] T34: Paper claim/evidence ledger and figure-table outline
  - Task package: `docs/tasks/Phase2/T34_paper_claim_evidence_ledger.md`
  - Output: `docs/paper_claim_evidence_ledger.md`
  - Review output: `docs/review/T34_review.md`
  - Captain verdict: `PASS`
  - Result: bounded claim/evidence ledger and figure-table outline are in place; paper assembly can proceed without silently upgrading mock/stub/smoke/readiness evidence
- [x] T35: Paper draft skeleton and reviewer-risk audit
  - Task package: `docs/tasks/Phase2/T35_paper_draft_skeleton_and_reviewer_risk_audit.md`
  - Output: `docs/paper_draft_skeleton.md`
  - Output: `docs/paper_reviewer_risk_audit.md`
  - Review output: `docs/review/T35_review.md`
  - Captain verdict: `PASS`
  - Result: bounded manuscript skeleton and reviewer-risk audit are in place; Milestone 2K paper-assembly readiness is complete without upgrading blocked evidence

### Milestone 2L: Paper Positioning Gate

- [x] T41: Milestone 2K paper-assembly gate review and next-phase decision
  - Task package: `docs/tasks/Phase2/T41_paper_assembly_milestone_review.md`
  - Output: `docs/review/Milestone2K_review.md`
  - Captain verdict: `PASS`
  - Result: Milestone 2K is formally closable with verdict `Allow`; minimum safe paper positioning and the need for Background / Related Work before prose expansion are now explicit

### Milestone 2M: Paper Framing And Scaffold Extension

- [x] T42: Paper Background / Related Work scaffold and method-positioning calibration
  - Task package: `docs/tasks/Phase2/T42_paper_background_related_work_and_positioning.md`
  - Output: `docs/paper_method_positioning_calibration.md`
  - Review output: `docs/review/T42_review.md`
  - Captain verdict: `PASS`
  - Result: Background / Related Work scaffold and method-positioning calibration are now in place; the working paper framing is method-forward title plus evidence-bounded body text, without upgrading blocked claims

### Milestone 2N: Paper Background Prose Draft

- [x] T43: Paper Background / Related Work bounded prose draft
  - Task package: `docs/tasks/Phase2/T43_paper_background_related_work_prose_draft.md`
  - Output: `docs/paper_background_related_work_draft.md`
  - Review output: `docs/review/T43_review.md`
  - Captain verdict: `PASS`
  - Warning classification:
    - N1 subsection-6 neutrality = `accepted`
    - N2 placeholder citation markers = `accepted`
    - N3 internal drafting annotations = `accepted`
    - N4 inline claim-reference formatting inconsistency = `accepted`
  - Result: bounded Background / Related Work prose draft exists, but it does not authorize continued paper expansion ahead of evidence/material recovery

### Milestone 2O: Research Reality Recovery Mode

- [ ] T44: Research Reality Recovery Mode setup and evidence-gap ledger
  - Task package: `docs/tasks/Phase2/T44_research_reality_recovery_mode_setup_and_evidence_gap_ledger.md`

Long-term objective:

以论文级质量为最终目标，但当前先进入 `Research Reality Recovery Mode`。后续任务顺序改为“真实性冻结 -> claim/evidence/material 台账 -> 复现/图表/结果缺口审计 -> 风险收口 -> 再决定是否恢复论文扩写”。除 `Current Unique Task` 外，其他 pending 项只代表路线图，不可直接执行。

## Current Unique Task

`T44: Research Reality Recovery Mode setup and evidence-gap ledger`

状态说明：

- `T43` 已完成并通过 review，Captain verdict = `PASS`
- T43 review blocking issues：
  - none
- T43 review non-blocking issues：
  - N1 subsection-6 neutrality：`accepted`
  - N2 placeholder citation markers：`accepted`
  - N3 internal drafting annotations：`accepted`
  - N4 inline claim-reference formatting inconsistency：`accepted`
- T43 没有 `deferred` warning，因此本轮不新增风险条目
- 当前项目保持 `Phase 2: Controlled Development / Go`，但子模式切换为 `Research Reality Recovery Mode`
- 论文 prose 扩写在 T43 后暂停；当前优先级改为补 claim/evidence/material/figure/reproducibility truth baseline
- `R10`、`R11`、`R12`、`R13`、`R14`、`R20`、`R23`、`R24` 仍未收口，且目前缺少面向论文成稿的统一 figure/result/material ledger
- `T32` 仍被当前机器缺少真实 `.tflite` runtime 依赖阻塞，`T37` 仍被硬件/bitstream 前提阻塞

为什么现在做它：

1. 用户已明确要求按 `docs/reference/科研纠偏意见.md` 先做 recovery，而不是继续把 paper prose 往前推。
2. `T34` 到 `T43` 已经产出 skeleton、ledger、定位校准和 bounded prose draft，但这些产物不能替代证据充分性、图表充分性和复现充分性审计。
3. 当前最缺的不是再多一段 prose，而是一份把“哪些能写、哪些不能写、哪些还缺材料”冻结清楚的 recovery baseline。
4. `T44` 是 docs-only bounded task，不依赖新环境、新硬件或新运行结果，且能为后续 recovery 任务建立统一入口。

## Captain Output For Current Task

1. 当前唯一任务：`T44`
2. `T43` 已按 `PASS` 收口。
3. T43 review blocking issues：
   - none
4. T43 non-blocking comments：
   - accepted: subsection-6 neutrality needs to remain modest
   - accepted: placeholder citation markers must later be resolved via shared bibliography
   - accepted: internal drafting annotations must be cleaned during manuscript assembly, not now
   - accepted: inline claim-reference formatting should be normalized later
5. T43 review output：`docs/review/T43_review.md`
6. T44 任务包：`docs/tasks/Phase2/T44_research_reality_recovery_mode_setup_and_evidence_gap_ledger.md`

## Done Criteria For T44

1. Freeze the current truth/evidence boundary in `docs/reality_recovery/00_freeze_snapshot.md`.
2. Build a `supported / partial / blocked` claim-evidence table for current paper-facing claims.
3. Audit code truth, experiment reproducibility, figure/result readiness, and paper-claim risk in separate recovery docs.
4. Produce a short human-facing brief that says whether paper writing should continue right now.
5. Stay docs-only: no code, config, benchmark, training, `.tflite`, hardware, cleanup, `runs/`, or `artifacts` changes.
6. Do not modify `docs/02_experiment_plan.md` or silently upgrade any evidence level.
