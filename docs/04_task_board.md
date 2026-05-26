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

- [x] T44: Research Reality Recovery Mode setup and evidence-gap ledger
  - Task package: `docs/tasks/Phase2/T44_research_reality_recovery_mode_setup_and_evidence_gap_ledger.md`
  - Review output: `docs/review/T44_review.md`
  - Captain verdict: `PASS`
  - Result: recovery baseline, claim/evidence freeze, reproducibility audit, figure/result ledger, and paper-risk table are now in place; paper prose remains paused

### Milestone 2O.5: Theory Analysis For Mainline Loop

- [x] T53: Mainline theory analysis document for the full GKP correction loop
  - Task package: `docs/tasks/Phase2/T53_mainline_theory_analysis_document.md`
  - Output: `docs/mainline_theory_analysis.md`
  - Review output: `docs/review/T53_review.md`
  - Captain verdict: `PASS`
  - Result: a bounded mainline theory walkthrough now exists for personal understanding and later paper support; it explains the full GKP correction loop from approximate-code definition to runtime `(K, b)` execution without upgrading any blocked `.tflite`, real-board, or paper-grade evidence

### Milestone 2P: Mainline Evidence Hardening (proposed)

- [x] T45: Paper-grade benchmark expansion protocol lock and gap audit
  - Task package: `docs/tasks/Phase2/T45_paper_grade_benchmark_expansion_protocol_lock_and_gap_audit.md`
  - Review output: `docs/review/T45_review.md`
  - Captain verdict: `PASS`
  - Result: benchmark-expansion protocol is now frozen at the policy level; frozen-set evidence stays separate from any future expansion lane, and `docs/reference/延伸改进思路.md` remains reference-only
- [x] T46: Multi-seed mechanism/intervention plan and trace pack
  - Task package: `docs/tasks/Phase2/T46_multi_seed_mechanism_intervention_plan_and_trace_pack.md`
  - Output: `docs/seed_mechanism_multi_seed_plan.md`
  - Review output: `docs/review/T46_review.md`
  - Captain verdict: `PASS`
  - Warning handling: all non-blocking comments `accepted`; no `deferred` or `rejected` items
  - Result: the project now has a bounded multi-seed / intervention evidence plan, but current mechanism evidence is still only single-seed diagnostic and has not been upgraded to multi-seed confirmation or causal proof
- [x] T54: Phase A multi-seed trace-only generalization probe
  - Task package: `docs/tasks/Phase2/T54_multi_seed_trace_only_generalization_probe.md`
  - Output: `docs/multi_seed_trace_generalization_probe.md`
  - Review output: `docs/review/T54_review.md`
  - Captain verdict: `PASS`
  - Warning handling: all non-blocking comments `accepted`; no `deferred` or `rejected` items
  - Result: the committed-`b` instability pattern is broadly repeated with qualifications across the locked 6-seed pack, but this remains bounded diagnostic evidence and `C4` stays `partial`
- [x] T55: Phase B multi-seed I1 residual-clip intervention probe
  - Task package: `docs/tasks/Phase2/T55_multi_seed_i1_residual_clip_intervention_probe.md`
  - Output: `docs/multi_seed_i1_intervention_probe.md`
  - Review output: `docs/review/T55_review.md`
  - Captain verdict: `PASS`
  - Warning handling: all non-blocking comments `accepted`; no `deferred` or `rejected` items
  - Result: pure I1 lower-clip intervention is mixed and mostly harmful (harms 4/6, helps 2/6); the simple “high committed-b is harmful” mechanism framing is not supported as a general explanation, and `C4` remains `partial`
- [x] T56: Post-I1 mechanism claim reframing gate
  - Task package: `docs/tasks/Phase2/T56_post_i1_mechanism_claim_reframing_gate.md`
  - Output: `docs/post_t55_mechanism_claim_reframing_gate.md`
  - Review output: `docs/review/T56_review.md`
  - Captain verdict: `PASS`
  - Warning handling: all non-blocking comments `accepted`; no `deferred` or `rejected` items
  - Result: mechanism claims are now explicitly retain / weaken / retire / reframe / still-open; `T47` may proceed only under conditioned mechanism-hedge wording
- [x] T47: Paper ablation result-pack and material ledger
  - Task package: `docs/tasks/Phase2/T47_paper_ablation_result_pack_and_material_ledger.md`
  - Output: `docs/paper_ablation_result_pack.md`
  - Review output: `docs/review/T47_review.md`
  - Captain verdict: `PASS`
  - Warning handling:
    - N1 figure-entry miscount in Worker Output = `accepted`
    - N2 worker summary outside strict Allowed Files boundary = `accepted`
    - N3 F2 ready-without-drawn-figure classification = `accepted`
    - N4 F3 blocked carry-forward note = `accepted`
    - N5 conceptual regeneration paths instead of executable scripts = `accepted`
  - Result: the paper ablation/material ledger is now frozen honestly; FR7 remains missing and is now the next bounded execution gap
- [x] T57: FR7 feature/teacher ablation re-execution under locked T24 protocol
  - Task package: `docs/tasks/Phase2/T57_fr7_feature_teacher_ablation_reexecution.md`
  - Output: `docs/fr7_feature_teacher_ablation_reexecution.md`
  - Review output: `docs/review/T57_review.md`
  - Captain verdict: `PASS`
  - Warning handling: no blocking issues and no new `deferred` / `rejected` warning items
  - Result: FR7 is now a ready frozen-set result table, but it does not close causal interpretation and it weakens any simple "teacher params are necessary for the win" story
- [x] T58: FR6 multi-seed mechanism/intervention figure pack
  - Task package: `docs/tasks/Phase2/T58_fr6_multi_seed_mechanism_intervention_figure_pack.md`
  - Output: `docs/fr6_multi_seed_mechanism_intervention_figure_pack.md`
  - Review output: `docs/review/T58_review.md`
  - Captain verdict: `PASS_WITH_WARNINGS`
  - Warning handling:
    - N1 mixed-governance diff provenance = `accepted`
    - N2 FR6 deliverable completeness note = `accepted`
    - N3 task-local seed-category derivation = `accepted`
    - N4 worker self-review is not final acceptance review = `accepted`
  - Result: FR6 is now a ready bounded descriptive figure pack; it does not close `R10` and does not upgrade `C4` beyond `partial`
- [ ] T59: Statcalib separate comparator lane integration and bounded smoke
  - Task package: `docs/tasks/Phase2/T59_statcalib_comparator_lane_integration_and_smoke.md`
  - Status: current mainline integration/smoke lane to unlock FR8 honestly without changing frozen T24 semantics

### Milestone 2Q: Deployment Boundary Boosters (proposed)

- [ ] T48: True `.tflite` runtime smoke gate
  - Task package: pending
- [ ] T49: Real-board smoke execution gate
  - Task package: pending

### Milestone 2R: Reproducibility And Material Pack (proposed)

- [ ] T50: Training reproducibility and material-regeneration pack
  - Task package: pending

### Milestone 2S: Paper Re-open Gate (proposed)

- [ ] T51: Paper positioning re-gate after evidence hardening
  - Task package: pending
- [ ] T52: Manuscript expansion gate for the next bounded prose wave
  - Task package: pending

Long-term objective:

以论文级质量为最终目标，但当前先进入 `Research Reality Recovery Mode`。后续任务顺序改为“真实性冻结 -> claim/evidence/material 台账 -> 复现/图表/结果缺口审计 -> 风险收口 -> 再决定是否恢复论文扩写”。除 `Current Unique Task` 外，其他 pending 项只代表路线图，不可直接执行。

## Current Unique Task

`T59: Statcalib separate comparator lane integration and bounded smoke`

状态说明：

- `T58` 已完成并通过 review，Captain verdict = `PASS_WITH_WARNINGS`
- `docs/review/T58_review.md` blocking issues：none
- `T58` warning classification：`N1 accepted`、`N2 accepted`、`N3 accepted`、`N4 accepted`
- `T58` 没有新的 `deferred` / `rejected` warning，因此没有新的 warning-derived risk
- `T58` 已把 `FR6` 收口为 bounded descriptive figure pack，但这不是 `R10` 的 causal closure，也不会把 `C4` 升级为 `supported`
- `FR8` 现在是最大的 mainline paper-material gap，但当前仓库仍缺 integrated `statcalib` comparator lane
- 当前项目保持 `Phase 2: Controlled Development / Go`，但子模式仍是 `Research Reality Recovery Mode`
- `T59` 是 mainline experiment-evidence lane 上的 bounded integration/smoke 任务，且必须与 theory-only branch materials 保持隔离

为什么现在做它：

1. `T57` 与 `T58` 已分别收口 `FR7` 和 `FR6`，当前最小且合理的 mainline gap 只剩 `FR8`。
2. `FR8` 不能直接写成正式结果表，因为当前仓库只有 `T26` feasibility gate 和 `T30` interface contract，还没有 slow-loop / benchmark integration lane。
3. `T59` 把下一步压缩为“separate statcalib comparator lane integration + bounded smoke”，先证明该 lane 能否在不改写 frozen `T24` semantics 的前提下 end-to-end 运行。
4. `T59` 必须保持 mainline experiment branch 与 theory-only branch 的隔离；如果做不到，就不能把 `FR8` 推向 paper-facing evidence。

## Captain Output For Current Task

- Current unique task: `T59`
- Latest completed review: `docs/review/T58_review.md` with verdict `PASS_WITH_WARNINGS`
- Warning classification for T58: `N1 accepted`, `N2 accepted`, `N3 accepted`, `N4 accepted`; no `deferred` / `rejected`
- Next worker-facing task package: `docs/tasks/Phase2/T59_statcalib_comparator_lane_integration_and_smoke.md`
- `T59` is allowed to proceed only as a separate statcalib comparator lane integration + bounded smoke task; no frozen-set rewrite and no theory-branch edits

1. 当前唯一任务：`T59`
2. `T58` 已按 `PASS_WITH_WARNINGS` 收口。
3. T58 review blocking issues：
   - none
4. T58 warning handling：
   - N1 accepted
   - N2 accepted
   - N3 accepted
   - N4 accepted
   - no new `deferred` / `rejected` warning
   - no new risk opened by warning classification
5. T58 review output：`docs/review/T58_review.md`
6. T59 任务包：`docs/tasks/Phase2/T59_statcalib_comparator_lane_integration_and_smoke.md`

## Done Criteria For T59

1. Add a separate `statcalib` slow-loop comparator lane without rewriting frozen `T24` ranked-set semantics.
2. Complete one bounded task-scoped smoke run and record the exact command, config, interpreter, and run root.
3. Show that `statcalib` appears as a separately labeled mode in the smoke outputs and that its status/reason semantics propagate end-to-end.
4. Keep all changes inside the T59 allowed-file set, plus one task-scoped run root only.
5. Do not touch `.tflite`, real-board, training, cleanup, benchmark expansion, historical `runs/` / `artifacts/`, theory-only branch materials, or `docs/02_experiment_plan.md`.
6. If the bounded smoke cannot produce an executable comparator lane honestly, report that boundary explicitly instead of upgrading `FR8`.

## 2026-05-24 Captain Update (T47 closeout)
 
- `T47` review is accepted as `PASS`.
- Blocking issues: none.
- Warning classification:
  - N1 figure-entry miscount in Worker Output = `accepted`
  - N2 worker summary outside strict Allowed Files boundary = `accepted`
  - N3 F2 ready-without-drawn-figure classification = `accepted`
  - N4 F3 blocked carry-forward note = `accepted`
  - N5 conceptual regeneration paths instead of executable scripts = `accepted`
- No `deferred` or `rejected` warning remains from this review, so no new risk item is opened by warning classification alone.
- `T47` is now complete. Its paper-facing ledger is frozen honestly, and `FR7` is explicitly confirmed as the largest remaining ablation-pack gap.
- Current unique task is now `T57: FR7 feature/teacher ablation re-execution under locked T24 protocol`.
- `T57` must stay inside the frozen four scenarios, the fixed six-mode feature-ablation set, and `repeats=2`. It must not retrain, touch source-tree code/config, or reopen `.tflite`, real-board, cleanup, benchmark expansion, comparator expansion, or intervention scope.
- Worker-facing task package: `docs/tasks/Phase2/T57_fr7_feature_teacher_ablation_reexecution.md`.

## 2026-05-26 Captain Update (T57 closeout)

- `T57` review is accepted as `PASS`.
- Blocking issues: none.
- `T57` review does not introduce any new non-blocking item that needs `accepted / deferred / rejected` classification beyond the verdict itself.
- Therefore no new risk item is opened by warning classification for `T57`.
- `T57` closes `FR7` as a bounded frozen-set result-table gap, but it does not close `R10` and does not justify causal or architectural-attribution upgrades.
- The strongest bounded caution from `T57` is that `hybrid_no_teacher_params` becomes best in all 4 scenarios, so the paper must not claim that teacher params are a necessary positive contributor to the win.
- Current unique task is now `T58: FR6 multi-seed mechanism/intervention figure pack`.
- `T58` is docs-only. It must reuse existing `T54/T55/T56` evidence, must not run new benchmark/trace/intervention work, and must not touch theory-only branch materials.
- Worker-facing task package: `docs/tasks/Phase2/T58_fr6_multi_seed_mechanism_intervention_figure_pack.md`.
