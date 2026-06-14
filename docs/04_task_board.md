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
  - `2026-05-18` 用户明确要求按 `docs/legacy_context/reference_retired_2026-06-11/科研纠偏意见.md` 进入 recovery-first 推进
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
  - Gate output: `docs/evidence_packs/statcalib_fr8/statcalib_feasibility_gate.md`
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
  - Diagnosis report: `docs/evidence_packs/mechanism_ablation/seed20260429_failure_diagnosis.md`
  - Review output: `docs/review/T36_review.md`
  - Captain verdict: `PASS`
  - Conclusion: existing artifacts narrow `seed=20260429` to a residual-amplitude / teacher-delta regime instability hypothesis, but do not expose per-window committed-parameter traces; no benchmark rerun or branch expansion occurred
- [x] T38: `seed=20260429` single-seed trace-export probe, bounded unchanged-semantics rerun
  - Task package: `docs/tasks/Phase2/T38_seed20260429_trace_export_probe.md`
  - Trace diagnosis: `docs/evidence_packs/mechanism_ablation/seed20260429_trace_export_diagnosis.md`
  - Review output: `docs/review/T38_review.md`
  - Captain verdict: `PASS`
  - Run root: `runs/T38_seed20260429_trace_probe_20260513`
  - Result: `4798` trace rows with required fields present; combined committed-`b` instability is trace-supported for `seed=20260429`, but still seed-bounded diagnostic evidence
  - Milestone gate: `docs/review/Milestone2I_review.md` verdict = `Conditional Allow`

### Milestone 2J: Reproducibility And Deployment Boundary

- [x] T31: Training-chain portable dependency lock plan
  - Task package: `docs/tasks/Phase2/T31_training_chain_portable_dependency_lock_plan.md`
  - Plan output: `docs/evidence_packs/training_reproducibility/training_chain_portable_dependency_lock_plan.md`
  - Review output: `docs/review/T31_review.md`
  - Captain verdict: `PASS`
  - Result: training-chain local interpreter/package facts and CPU-vs-GPU lock strategy are documented; clean-environment rebuild remains unverified
- [x] T39: Training-chain CPU-only clean-environment draft lock and dry-run bootstrap
  - Task package: `docs/tasks/Phase2/T39_training_chain_cpu_cleanenv_draft_lock.md`
  - Bootstrap output: `docs/evidence_packs/training_reproducibility/training_chain_cpu_cleanenv_bootstrap.md`
  - Review output: `docs/review/T39_review.md`
  - Captain verdict: `PASS`
  - Result: clean CPU-only environment, draft dependency lock, and dry-run/import-level bootstrap are verified; real clean-environment training execution remains unverified
- [x] T40: Training-chain CPU-only clean-environment minimal real-training smoke
  - Task package: `docs/tasks/Phase2/T40_training_chain_cpu_cleanenv_minimal_train_smoke.md`
  - Smoke output: `docs/evidence_packs/training_reproducibility/training_chain_cpu_cleanenv_train_smoke.md`
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
- [ ] T37: Real-board smoke execution gate, only if `Linux + FPGA` hardware host, device path, and bitstream evidence are ready
  - Status note: `resource-blocked / lowest-priority backlog`
  - Task package: pending

### Milestone 2K: Paper Assembly Readiness

- [x] T34: Paper claim/evidence ledger and figure-table outline
  - Task package: `docs/tasks/Phase2/T34_paper_claim_evidence_ledger.md`
  - Output: `docs/paper_materials/paper_claim_evidence_ledger.md`
  - Review output: `docs/review/T34_review.md`
  - Captain verdict: `PASS`
  - Result: bounded claim/evidence ledger and figure-table outline are in place; paper assembly can proceed without silently upgrading mock/stub/smoke/readiness evidence
- [x] T35: Paper draft skeleton and reviewer-risk audit
  - Task package: `docs/tasks/Phase2/T35_paper_draft_skeleton_and_reviewer_risk_audit.md`
  - Output: `docs/paper_materials/paper_draft_skeleton.md`
  - Output: `docs/paper_materials/paper_reviewer_risk_audit.md`
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
  - Output: `docs/paper_materials/paper_method_positioning_calibration.md`
  - Review output: `docs/review/T42_review.md`
  - Captain verdict: `PASS`
  - Result: Background / Related Work scaffold and method-positioning calibration are now in place; the working paper framing is method-forward title plus evidence-bounded body text, without upgrading blocked claims

### Milestone 2N: Paper Background Prose Draft

- [x] T43: Paper Background / Related Work bounded prose draft
  - Task package: `docs/tasks/Phase2/T43_paper_background_related_work_prose_draft.md`
  - Output: `docs/paper_materials/paper_background_related_work_draft.md`
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
  - Output: `docs/paper_materials/mainline_theory_analysis.md`
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
  - Output: `docs/evidence_packs/mechanism_ablation/seed_mechanism_multi_seed_plan.md`
  - Review output: `docs/review/T46_review.md`
  - Captain verdict: `PASS`
  - Warning handling: all non-blocking comments `accepted`; no `deferred` or `rejected` items
  - Result: the project now has a bounded multi-seed / intervention evidence plan, but current mechanism evidence is still only single-seed diagnostic and has not been upgraded to multi-seed confirmation or causal proof
- [x] T54: Phase A multi-seed trace-only generalization probe
  - Task package: `docs/tasks/Phase2/T54_multi_seed_trace_only_generalization_probe.md`
  - Output: `docs/evidence_packs/mechanism_ablation/multi_seed_trace_generalization_probe.md`
  - Review output: `docs/review/T54_review.md`
  - Captain verdict: `PASS`
  - Warning handling: all non-blocking comments `accepted`; no `deferred` or `rejected` items
  - Result: the committed-`b` instability pattern is broadly repeated with qualifications across the locked 6-seed pack, but this remains bounded diagnostic evidence and `C4` stays `partial`
- [x] T55: Phase B multi-seed I1 residual-clip intervention probe
  - Task package: `docs/tasks/Phase2/T55_multi_seed_i1_residual_clip_intervention_probe.md`
  - Output: `docs/evidence_packs/mechanism_ablation/multi_seed_i1_intervention_probe.md`
  - Review output: `docs/review/T55_review.md`
  - Captain verdict: `PASS`
  - Warning handling: all non-blocking comments `accepted`; no `deferred` or `rejected` items
  - Result: pure I1 lower-clip intervention is mixed and mostly harmful (harms 4/6, helps 2/6); the simple “high committed-b is harmful” mechanism framing is not supported as a general explanation, and `C4` remains `partial`
- [x] T56: Post-I1 mechanism claim reframing gate
  - Task package: `docs/tasks/Phase2/T56_post_i1_mechanism_claim_reframing_gate.md`
  - Output: `docs/evidence_packs/mechanism_ablation/post_t55_mechanism_claim_reframing_gate.md`
  - Review output: `docs/review/T56_review.md`
  - Captain verdict: `PASS`
  - Warning handling: all non-blocking comments `accepted`; no `deferred` or `rejected` items
  - Result: mechanism claims are now explicitly retain / weaken / retire / reframe / still-open; `T47` may proceed only under conditioned mechanism-hedge wording
- [x] T47: Paper ablation result-pack and material ledger
  - Task package: `docs/tasks/Phase2/T47_paper_ablation_result_pack_and_material_ledger.md`
  - Output: `docs/paper_materials/paper_ablation_result_pack.md`
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
  - Output: `docs/evidence_packs/mechanism_ablation/fr7_feature_teacher_ablation_reexecution.md`
  - Review output: `docs/review/T57_review.md`
  - Captain verdict: `PASS`
  - Warning handling: no blocking issues and no new `deferred` / `rejected` warning items
  - Result: FR7 is now a ready frozen-set result table, but it does not close causal interpretation and it weakens any simple "teacher params are necessary for the win" story
- [x] T58: FR6 multi-seed mechanism/intervention figure pack
  - Task package: `docs/tasks/Phase2/T58_fr6_multi_seed_mechanism_intervention_figure_pack.md`
  - Output: `docs/evidence_packs/mechanism_ablation/fr6_multi_seed_mechanism_intervention_figure_pack.md`
  - Review output: `docs/review/T58_review.md`
  - Captain verdict: `PASS_WITH_WARNINGS`
  - Warning handling:
    - N1 mixed-governance diff provenance = `accepted`
    - N2 FR6 deliverable completeness note = `accepted`
    - N3 task-local seed-category derivation = `accepted`
    - N4 worker self-review is not final acceptance review = `accepted`
  - Result: FR6 is now a ready bounded descriptive figure pack; it does not close `R10` and does not upgrade `C4` beyond `partial`
- [x] T59: Statcalib separate comparator lane integration and bounded smoke
  - Task package: `docs/tasks/Phase2/T59_statcalib_comparator_lane_integration_and_smoke.md`
  - Output: `docs/evidence_packs/statcalib_fr8/statcalib_comparator_lane_smoke.md`
  - Review output: `docs/review/T59_review.md`
  - Captain verdict: `PASS_WITH_WARNINGS`
  - Warning handling:
    - W1 cross-mode `teacher_mode` fallback coupling = `deferred`
    - W2 smoke-doc key-name mismatch = `accepted`
    - W3 dirty-worktree smoke provenance weakness = `deferred`
  - Result: T59 closes the first integrated statcalib lane smoke gap, but it does not open FR8 and does not constitute formal comparator evidence
- [x] T60: Statcalib lane isolation and regression hardening
  - Task package: `docs/tasks/Phase2/T60_statcalib_lane_isolation_and_regression_hardening.md`
  - Output: `docs/evidence_packs/statcalib_fr8/statcalib_lane_isolation_and_regression_hardening.md`
  - Review output: `docs/review/T60_review.md`
  - Captain verdict: `PASS`
  - Warning handling: no blocking issues and no new `accepted / deferred / rejected` warning item
  - Result: T60 closes the cross-mode semantics and regression-coverage blocker from T59; it closes `W1`/`R26`, but it does not close `R27` and does not open FR8
- [ ] T61: Statcalib clean-provenance fairness sanity rerun
  - Task package: `docs/tasks/Phase2/T61_statcalib_clean_provenance_fairness_sanity.md`
  - Output: `docs/evidence_packs/statcalib_fr8/statcalib_fairness_sanity.md`
  - Review output: `docs/review/T61_review.md`
  - Captain verdict: `BLOCK`
  - Blocking issue: launch clean `HEAD=9174065`, but final `summary.json git_commit=6058f42`; mid-run branch movement means the task did not close the clean-provenance blocker it was created to repair
  - Result: the bounded fairness signal persisted, but `R27` remains open and `T61` is not complete
- [x] T62: Statcalib provenance-isolated fairness rerun
  - Task package: `docs/tasks/Phase2/T62_statcalib_provenance_isolated_fairness_rerun.md`
  - Output: `docs/evidence_packs/statcalib_fr8/statcalib_provenance_isolated_fairness_rerun.md`
  - Review output: `docs/review/T62_review.md`
  - Captain verdict: `PASS`
  - Warning handling: no blocking issues and no new `accepted / deferred / rejected` warning item
  - Result: T62 closes the T61 provenance blocker and provides one provenance-clean bounded fairness sanity rerun, but it still does not open FR8 and does not upgrade the evidence beyond mock-backed software-HIL scope
- [x] T63: FR8 statcalib comparator gate review
  - Task package: `docs/tasks/Phase2/T63_fr8_statcalib_comparator_gate_review.md`
  - Output: `docs/evidence_packs/statcalib_fr8/fr8_statcalib_comparator_gate_review.md`
  - Review output: `docs/review/T63_review.md`
  - Captain verdict: `PASS`
  - Result: T63 closes the pre-FR8 gate-discussion lane honestly; the repository may now open exactly one bounded FR8 extension-lane task, but T63 is not itself FR8 evidence and does not close `R24`
- [x] T64: FR8 statcalib extension-lane bounded benchmark
  - Task package: `docs/tasks/Phase2/T64_fr8_statcalib_extension_lane_bounded_benchmark.md`
  - Output: `docs/evidence_packs/statcalib_fr8/fr8_statcalib_extension_lane_benchmark.md`
  - Review output: `docs/review/T64_review.md`
  - Captain verdict: `PASS_WITH_WARNINGS`
  - Warning handling:
    - N1 execution-shape wording drift in the result doc = `deferred`
    - N2 finish-timestamp provenance wording drift = `deferred`
    - N3 extension-lane over-interpretation boundary = `deferred`
  - Result: T64 closes one clean-provenance bounded FR8 extension-lane benchmark on the locked four-scenario protocol without rewriting `T24`, but the evidence remains mock-backed software-HIL only
- [x] T65: FR8 extension-lane consistency guard and report closeout
  - Task package: `docs/tasks/Phase2/T65_fr8_extension_lane_consistency_guard_and_closeout.md`
  - Output: `docs/evidence_packs/statcalib_fr8/fr8_statcalib_extension_lane_consistency_audit.md`
  - Review output: `docs/review/T65_review.md`
  - Captain verdict: `PASS_WITH_WARNINGS`
  - Warning handling:
    - N1 mixed-diff scope acceptance depends on explicit user/captain clarification = `accepted`
    - N2 T64-specific audit helper is intentionally narrow, not generic FR8 framework = `accepted`
    - N3 review wording should have stated the clarification dependency more explicitly = `accepted`
  - Result: T65 closes the T64 report/artifact consistency gap and makes the T64 result pack self-audited and safer to reuse, but it does not close `R24`
- [x] T66: FR8 statcalib sensitivity bounded benchmark
  - Task package: `docs/tasks/Phase2/T66_fr8_statcalib_sensitivity_bounded_benchmark.md`
  - Output: `docs/evidence_packs/statcalib_fr8/statcalib_sensitivity_bounded_benchmark.md`
  - Review output: `docs/review/T66_review.md`
  - Captain verdict: `PASS_WITH_WARNINGS`
  - Warning handling:
    - `N1` duplicate-running progress-log artifact after same-run-root timeout relaunch = `accepted`
    - `N2` aggregate-best vs stability-best split = `deferred`
    - `N3` `static_bias_theta / statcalib_high_threshold` best row still carries aggregate `statcalib_status = mixed` = `deferred`
  - Result: T66 closes one bounded local-grid robustness gap under clean provenance, but it does not close `R24` and does not upgrade statcalib into a mature calibration comparator
- [x] T67: FR8 statcalib teacher-anchor dependence bounded benchmark
  - Task package: `docs/tasks/Phase2/T67_fr8_statcalib_teacher_anchor_dependence_bounded_benchmark.md`
  - Output: `docs/evidence_packs/statcalib_fr8/statcalib_teacher_anchor_bounded_benchmark.md`
  - Review output: `docs/review/T67_review.md`
  - Captain verdict: `PASS_WITH_WARNINGS`
  - Warning handling:
    - `N1` source-worktree scope-external PDF diff but clean-clone launch preserved provenance = `accepted`
    - `N2` equal-mean tie is not represented explicitly in `better_parameter_point_by_mean_ler` = `accepted`
    - `N3` two comparison rows remain `mixed` = `deferred`
  - Result: T67 closes the gross teacher-anchor dependence question honestly, but it does not close `R24`; the strongest aggregate statcalib lane still is not a clean generated-only result pack
- [x] T68: FR8 statcalib generated-only robustness bounded benchmark
  - Task package: `docs/tasks/Phase2/T68_fr8_statcalib_generated_only_robustness_bounded_benchmark.md`
  - Output: `docs/evidence_packs/statcalib_fr8/statcalib_generated_only_robustness_bounded_benchmark.md`
  - Review output: `docs/review/T68_review.md`
  - Captain verdict: `PASS_WITH_WARNINGS`
  - Warning handling:
    - `N1` full generated-only winner set remains a tie, not a unique final threshold = `deferred -> R24`
    - `N2` some predeclared candidates remain `mixed` even though the bounded existence question is closed = `deferred -> R24`
    - `N3` clean short-path clone launch boundary must remain visible in downstream retellings = `accepted`
  - Result: T68 closes the bounded generated-only existence question honestly; full generated-only winners now exist inside the predeclared grid, but the strongest clean answer is still a tied `window_variance t001/t003/t005` set and `R24` remains open
- [x] T69: FR8 statcalib clean-winner tie-break bounded benchmark
  - Task package: `docs/tasks/Phase2/T69_fr8_statcalib_clean_winner_tiebreak_bounded_benchmark.md`
  - Output: `docs/evidence_packs/statcalib_fr8/statcalib_clean_winner_tiebreak_bounded_benchmark.md`
  - Review output: `docs/review/T69_review.md`
  - Captain verdict: `PASS_WITH_WARNINGS`
  - Warning handling:
    - `N1` persistent clean tie set is the honest bounded answer, not a unique final threshold = `accepted`
    - `N2` bounded-matrix-only conclusion must remain explicit in downstream retellings = `accepted`
  - Result: T69 closes the bounded clean-winner tie-break question honestly; no unique clean reference point emerges, and the strongest clean answer remains the persistent `window_variance_t001 = t003 = t005` tie set. `R24` remains open as a reporting/promotion boundary only, not as an unresolved tie-break execution question
- [x] T70: FR8 statcalib bounded closure pack and promotion gate
  - Task package: `docs/tasks/Phase2/T70_fr8_statcalib_bounded_closure_pack_and_promotion_gate.md`
  - Output: `docs/evidence_packs/statcalib_fr8/fr8_statcalib_bounded_closure_pack.md`
  - Review output: `docs/review/T70_review.md`
  - Captain verdict: `PASS`
  - Result: T70 closes the FR8 mainline closure-pack gap honestly; the repository now has one code-backed closure artifact that preserves `T24` as the authoritative frozen ranked table, preserves `statcalib` as a separately labeled extension lane, gives an explicit `no_promotion_keep_extension_lane_only` gate, and gives an explicit `future_selection_task_required` gate for any later single-threshold choice

### 并行 Sidecar 扩展实验治理

- [x] PSE0：并行 sidecar 扩展实验治理设置
  - 任务包：`docs/tasks/Phase2/PSE0_parallel_sidecar_extension_governance_setup.md`
  - 治理输出：`docs/sidecar/parallel_sidecar_extension_governance.md`
  - worktree 规划输出：`docs/sidecar/parallel_sidecar_worktree_plan.md`
  - Captain 状态：docs-only 设置任务已完成并通过验证
  - 边界：本任务不执行 `T69`，不创建 sidecar worktree，不启动实验，也不改变任何主线 benchmark 语义
  - 结果：后续 sidecar lane 可在 frozen-anchor、artifact-schema、promotion-gate、run-dir 和红线规则下规划；主线当前唯一任务仍以 `Current Unique Task` 区块为准
- [x] PSE1：sidecar main-controlled governance refresh
  - 任务包：`docs/tasks/Phase2/PSE1_sidecar_main_controlled_governance_refresh.md`
  - 治理入口：`docs/sidecar/README.md`
  - 精简治理：`docs/sidecar/00_sidecar_snapshot.md` 至 `docs/sidecar/04_sidecar_promotion_gate.md`
  - Captain 状态：docs-only 设置任务；不运行实验，不创建 `runs/sidecar`，不改变当前唯一主线任务
  - 结果：旧 `.wt/*` 长期分支退役为 read-only reference，后续 sidecar 默认由 main 控制台管理，允许新增-only代码/配置/helper，但不得破坏主线逻辑；结果仍必须写入 `runs/sidecar/<lane_id>/<run_id>/`

### Milestone 2Q: Deployment Boundary Boosters (proposed)

- [x] T48: True `.tflite` runtime smoke gate
  - Task package: `docs/tasks/Phase2/T48_true_tflite_runtime_smoke_gate.md`
  - Output: `docs/evidence_packs/deployment_boundary/t48_true_tflite_runtime_gate.md`
  - Review output: `docs/review/T48_review.md`
  - Captain verdict: `PASS`
  - Result: T48 closes one narrow current-host true `.tflite` runtime truth gap honestly; the repository now has one isolated `tensorflow==2.21.0` environment on this machine that can real-load and real-execute preserved `static_theta_v2` float / int8 `.tflite` artifacts and can run bounded source-vs-`.tflite` consistency checks, but this does not restore default-environment compatibility and does not upgrade the evidence to HIL, real-board, or deployment closure
- [x] T49: Real-board smoke execution gate
  - Task package: `docs/tasks/Phase2/T49_real_board_smoke_execution_gate.md`
  - Output: `docs/evidence_packs/deployment_boundary/t49_real_board_smoke_execution_gate.md`
  - Review output: `docs/review/T49_review.md`
  - Captain verdict: `PASS_WITH_WARNINGS`
  - Warning handling:
    - `W1` device-path readiness counts openable paths without enforcing `mmio + dma` role split = `deferred -> R30`
    - `W2` role-aware regression and checked-in-artifact replay regression are still missing = `deferred -> R30`
    - `W3` checked-in read-only regeneration entrypoint for the full gate artifact pack is still missing = `deferred -> R30`
  - Result: T49 closes one honest current-host real-board gate pack with verdict `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`; no real-board smoke was executed, and `R13/R14` remain open but narrower
- [x] T71: Real-board gate regeneration and host-transfer pack
  - Task package: `docs/tasks/Phase2/T71_real_board_gate_regeneration_and_host_transfer_pack.md`
  - Output: `docs/evidence_packs/deployment_boundary/t71_real_board_gate_regeneration_pack.md`
  - Review output: `docs/review/T71_review.md`
  - Captain verdict: `PASS_WITH_WARNINGS`
  - Warning handling:
    - `W1` `probe_limitations` 将未实际执行的限制写成既成事实 = `deferred -> R31`
    - `W2` `source_records` / `expected_byte_count_basis` 仍写死默认 config 口径 = `deferred -> R31`
    - `W3` `--config` / `--mmio-path` / `--dma-path` 的 provenance/override 回归不足 = `deferred -> R31`
    - `W4` collector 继续 import `BoardFPGAConfig` 作为 repo 内 config 读取入口 = `accepted`
  - Result: T71 closes the R30 gap honestly by hardening role-aware gate logic, adding a checked-in read-only collector, and proving replay/regeneration consistency; it still does not unlock `T37` or validate any real-board execution
- [x] T72: Real-board transfer-pack provenance hardening
  - Task package: `docs/tasks/Phase2/T72_real_board_transfer_pack_provenance_hardening.md`
  - Output: `docs/evidence_packs/deployment_boundary/t72_real_board_transfer_pack_provenance_hardening.md`
  - Review output: `docs/review/T72_review.md`
  - Captain verdict: `PASS_WITH_WARNINGS`
  - Warning handling:
    - `N1` 最小 config 场景下 path provenance 仍会把代码默认值写成 `source_kind=config_field` = `deferred -> R32`
    - `N2` Worker 原始主报告路径曾短暂落在精确 allowed files 之外，但当前 `HEAD` 已整理回允许目录 = `accepted`
    - `N3` 缺少覆盖 path 字段缺省回退标签的 focused regression = `deferred -> R32`
  - Result: T72 closes `R31` honestly by hardening probe provenance、default/override-aware source records、expected-byte-count derivation 和 focused override regressions；current-host verdict 仍是 `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`，`T37` 继续 blocked

### Milestone 2T: Mainline Paper-Facing Ledger Refresh (proposed)

- [x] T73: Mainline claim/evidence and result/figure/risk ledger refresh
  - Task package: `docs/tasks/Phase2/T73_mainline_claim_evidence_and_result_figure_ledger_refresh.md`

### Milestone 2U: Paper-Ready Simulation Material Pack (proposed)

- [x] T74: Paper-ready simulation result and figure pack
  - Task package: `docs/tasks/Phase2/T74_paper_ready_simulation_result_and_figure_pack.md`
  - Review output: `docs/review/T74_review.md`
  - Captain verdict: `PASS`
  - Result: T74 closes the paper-ready simulation/material packaging gap by producing stable-ID table/figure/caption/insertion/gap packs and task-scoped traceability assets without upgrading any deployment, real-board, or `statcalib` evidence level

### Milestone 2V: Main-Text Result Authoring Pack (proposed)

- [x] T75: Main-text results prose and final figure authoring pack
  - Task package: `docs/tasks/Phase2/T75_maintext_results_prose_and_final_figure_authoring_pack.md`
  - Output: `docs/paper_materials/paper_maintext_results_authoring_pack.md` plus task-scoped authoring assets under `docs/figure_assets/T75_maintext_results_authoring_pack/`
  - Review output: `docs/review/T75_review.md`
  - Captain verdict: `PASS`
  - Result: T75 closes the bounded main-text Results authoring gap by producing stable-ID-linked prose, caption/placement lock, appendix bridge, do-not-write guardrails, and publication-facing SVG assets without upgrading any benchmark, `.tflite`, real-board, or `statcalib` evidence level

### Milestone 2W: Rendered Figure QA And Results Assembly Pack (proposed)

- [x] T76: Rendered figure QA and results-section assembly pack
  - Task package: `docs/tasks/Phase2/T76_rendered_figure_qa_and_results_section_assembly_pack.md`
  - Output: `docs/paper_materials/paper_rendered_figure_qa.md`、`docs/paper_materials/paper_results_section_assembly_pack.md`、`docs/paper_materials/paper_results_callout_sheet.md` 与 `docs/figure_assets/T76_rendered_figure_qa_pack/*`
  - Review output: `docs/review/T76_review.md`
  - Captain verdict: `PASS_WITH_WARNINGS`
  - Warning handling:
    - `N1` preview-source 聚合行字段语义复用 = `deferred -> R34`
    - `N2` `.tmp_t76_*` 探针/缓存残留 = `accepted`
    - `N3` 逐图 QA 结论未内联完整上游 `T74-*` stable ID = `deferred -> R34`
  - Result: T76 closes the rendered-preview / legibility-QA / Results-section assembly gap honestly, but leaves one paper-facing traceability/schema hardening gap before the preview pack should be reused as a cleaner note-draft source layer

### Milestone 2X: Paper Note Results Sync And Traceability Hardening (proposed)

- [x] T77: 论文 note-draft 结果层同步与 T76 traceability hardening
  - Task package: `docs/tasks/Phase2/T77_paper_note_results_sync_and_traceability_hardening.md`
  - Output: `docs/paper_materials/paper_note_results_sync_manifest.md`、更新后的 `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex` 与 `docs/figure_assets/T76_rendered_figure_qa_pack/*`
  - Review output: `docs/review/T77_review.md`
  - Captain verdict: `PASS_WITH_WARNINGS`
  - Warning classification:
    - `N1` 整份 `.tex` 仍含未校准的非结果层历史段落 = `deferred -> R35`
    - `N2` `statcalib` 在 `Numerical Results` 中的视觉层级仍偏高 = `deferred -> R35`
    - `N3` `.log` 仍有 `Underfull \hbox` 排版 warning = `deferred -> R35`
    - `N4` section-scope audit 仍依赖 manifest / `% T77-SOURCE` 注释而非机械 guard = `deferred -> R35`
  - Result: T77 closes the note results-layer sync and T76 traceability hardening gap honestly, closes `R34`, but leaves one bounded note-calibration / hierarchy / layout closeout gap before any paper reopen gate

### Milestone 2Y: Paper Note Calibration And Layout Closeout (proposed)

- [x] T78: 论文 note-draft 非结果层校准、statcalib 层级降权与排版 warning 收口
  - Task package: `docs/tasks/Phase2/T78_paper_note_alignment_statcalib_hierarchy_and_layout_closeout.md`
  - Output: `docs/paper_materials/paper_note_alignment_and_layout_closeout.md`
  - Review output: `docs/review/T78_review.md`
  - Captain verdict: `PASS`
  - Result: T78 closes the bounded note non-results calibration / hierarchy / layout gap honestly; the note is no longer blocked on `R35`, but the mainline still needs one explicit reopen gate before any bounded prose expansion

### Milestone 2Z: Paper Reopen Gate And Prose Readiness Review (proposed)

- [x] T79: 论文材料 reopen gate 与 bounded prose 扩写就绪性评审
  - Task package: `docs/tasks/Phase2/T79_paper_reopen_gate_and_prose_readiness_review.md`
  - Output: `docs/paper_materials/paper_reopen_gate_and_prose_readiness_review.md`
  - Output: `docs/paper_materials/paper_reopen_gap_matrix.md`
  - Review output: `docs/review/T79_review.md`
  - Captain verdict: `PASS`
  - Result: `T79` 以 gate 形式确认当前材料栈已达到 `GO_FOR_BOUNDED_PROSE_REOPEN`，但这个结论只授权下一轮有界 prose reopen，不授权 full-manuscript reopen、方法章扩写或任何证据等级升级

### Milestone 2AA: Mainline Calibrated Section Bounded Prose Reopen (proposed)

- [x] T80: 主线校准段落的 bounded prose reopen
  - Task package: `docs/tasks/Phase2/T80_mainline_calibrated_sections_bounded_prose_reopen.md`
  - Output: `docs/paper_materials/paper_bounded_prose_reopen_manifest.md`
  - Review output: `docs/review/T80_review.md`
  - Captain verdict: `PASS`
  - Result: `T80` 真实完成了一轮 section-bounded prose reopen；当前 note 的 8 个 ready narrative / result-facing sections 已与主线 evidence stack 对齐，但 `Summary of Contributions` 与三章 methods 仍刻意保持 untouched，不得被回述成 full-manuscript reopen

### Milestone 2AB: Contribution And Methods Calibration Pack (proposed)

- [x] T81: Summary of Contributions 与 methods-only calibration pack
  - Task package: `docs/tasks/Phase2/T81_summary_and_methods_calibration_pack.md`
  - Output: `docs/paper_materials/paper_methods_and_contribution_calibration_manifest.md`
  - Review output: `docs/review/T81_review.md`
  - Captain verdict: `PASS`
  - Result: `T81` 真实完成了 `Summary of Contributions` 与三章 methods 的受控校准；当前 note 的主线正文与方法叙事已压回到同一 evidence stack，但这仍不等于 full-manuscript closeout

### Milestone 2AC: Supporting Material Closeout And Boundary Integration (proposed)

- [x] T82: supporting-material 收口与 appendix/supplement 边界整合包
  - Task package: `docs/tasks/Phase2/T82_supporting_material_closeout_and_boundary_integration_pack.md`
  - Output: `docs/paper_materials/paper_supporting_material_closeout_pack.md`
  - Output: `docs/paper_materials/paper_manuscript_closeout_readiness_matrix.md`
  - Review output: `docs/review/T82_review.md`
  - Captain verdict: `PASS`
  - Result: `T82` 真实完成了 supporting-boundary 的 `main text / appendix / supplement / blocked` 四层收口，并把 note 中 4 处 supporting 段落压回到当前 evidence stack；但这仍不等于 full-manuscript closeout

### Milestone 2AD: Mainline Note Full Consistency And Closeout Gate (proposed)

- [x] T83: 主线 note 全文一致性收口与 manuscript closeout gate
  - Task package: `docs/tasks/Phase2/T83_mainline_note_full_consistency_sweep_and_closeout_gate.md`
  - Output: `docs/paper_materials/paper_fullnote_consistency_crosswalk.md`
  - Output: `docs/paper_materials/paper_closeout_gate_and_blocker_register.md`
  - Review output: `docs/review/T83_review.md`
  - Captain verdict: `PASS`
  - Result: `T83` 真实完成了全文一致性 sweep、受控 wording 收口与唯一 closeout gate，并给出 `GO_FOR_BOUNDED_FINAL_POLISH_ONLY`；但这仍不等于 submission-ready pack、deployment closure 或 real-board success

### Milestone 2AE: Bounded Final Polish And Reader-Facing Assembly (proposed)

- [ ] T84: 主线 note 有界 final polish 与读者化装配包
  - Task package: `docs/tasks/Phase2/T84_mainline_bounded_final_polish_and_reader_facing_assembly.md`

### Milestone 2R: Reproducibility And Material Pack (proposed)

- [x] T50: Training reproducibility and material-regeneration pack
  - Task package: `docs/tasks/Phase2/T50_training_reproducibility_and_material_regeneration_pack.md`
  - Output: `docs/evidence_packs/training_reproducibility/training_reproducibility_and_material_regeneration_pack.md`
  - Review output: `docs/review/T50_review.md`
  - Captain verdict: `PASS`
  - Result: T50 closes one missing mainline training reproducibility/material-regeneration gap honestly; the repository now has one code-backed pack that enumerates canonical training materials, audits preserved mainline model references, and adds one clean CPU-only bounded train+eval rerun without upgrading the claims to full reproducibility, `.tflite`, real-board, benchmark, or deployment closure

### Milestone 2S: Paper Re-open Gate (proposed)

- [ ] T51: Paper positioning re-gate after evidence hardening
  - Task package: pending
- [ ] T52: Manuscript expansion gate for the next bounded prose wave
  - Task package: pending

Long-term objective:

以论文级质量为最终目标，但当前先进入 `Research Reality Recovery Mode`。后续任务顺序改为“真实性冻结 -> claim/evidence/material 台账 -> 复现/图表/结果缺口审计 -> 风险收口 -> 再决定是否恢复论文扩写”。除 `Current Unique Task` 外，其他 pending 项只代表路线图，不可直接执行。

## 2026-06-14 Captain Final Supersession (T83 closeout)

- Current unique task: `T84: 主线 note 有界 final polish 与读者化装配包`
- Task package: `docs/tasks/Phase2/T84_mainline_bounded_final_polish_and_reader_facing_assembly.md`
- `T83` has been judged `PASS`.
- `T83` has honestly completed one docs-only full-note consistency sweep plus one explicit closeout gate, and it adds no new warning-derived risk.
- `R13/R14/R32/R33` remain open, and `T37` remains `blocked + lowest-priority backlog`.
- `T84` is next because the mainline bottleneck is no longer “whether the note is self-consistent”, but “whether the already consistent note can be translated into reader-facing language and assembled into a tighter main-text/appendix/supplement route without promoting any blocked surface”.
- `T84` must remain docs-only, mainline-only, and final-polish-bounded: it may do reader-facing terminology translation, structure condensation, appendix/supplement assembly, one final-polish change map, one term-translation table, one assembly map, README registration, and compile-aware refresh; it must not widen into benchmark/HIL reruns, `.tflite` portability, real-board execution, theory-branch large-scale rewriting, sidecar promotion, or direct submission-pack completion.

## 2026-06-13 Captain Final Supersession (T82 closeout)

- Current unique task: `T83: 主线 note 全文一致性收口与 manuscript closeout gate`
- Task package: `docs/tasks/Phase2/T83_mainline_note_full_consistency_sweep_and_closeout_gate.md`
- `T82` has been judged `PASS`.
- `T82` has honestly completed one docs-only supporting-material closeout route across `main text / appendix / supplement / blocked`, and it adds no new warning-derived risk.
- `R13/R14/R32/R33` remain open, and `T37` remains `blocked + lowest-priority backlog`.
- `T83` is next because the mainline bottleneck is no longer “whether supporting-boundary materials are scattered”, but “whether the current full note is already section-by-section self-consistent and whether any remaining blocker should still block full-manuscript closeout”.
- `T83` must remain docs-only, mainline-only, and closeout-gate-bounded: it may do a full-note consistency sweep, bounded wording cleanup, one section-to-evidence crosswalk, one closeout gate / blocker register, and compile-aware refresh; it must not widen into benchmark/HIL reruns, `.tflite` portability, real-board execution, theory-branch large-scale rewriting, sidecar promotion, or direct full submission-pack assembly.

## 2026-06-12 Captain Final Supersession (T80 closeout)

- Current unique task: `T81: Summary of Contributions 与 methods-only calibration pack`
- Task package: `docs/tasks/Phase2/T81_summary_and_methods_calibration_pack.md`
- `T80` has been judged `PASS`.
- `T80` has honestly completed one docs-only section-bounded prose reopen on the 8 already-ready sections while keeping `Summary of Contributions` and the three methods chapters untouched by design.
- `T80` introduces no deferred/rejected warning and opens no new risk.
- `R13/R14/R32/R33` remain open, and `T37` remains `blocked + lowest-priority backlog`.
- `T81` is next because the dominant mainline bottleneck is no longer “whether the ready narrative/result-facing sections can be rewritten”, but “whether the still-untouched contribution and methods sections can be calibrated to the same evidence stack without silently widening to full-manuscript reopen”.
- `T81` must remain docs-only, mainline-only, and section-bounded: it may rewrite only `Summary of Contributions`、`Brief Review of the GKP Code`、`Noise and Drift Model`、`Model Architecture`; it must not widen into benchmark reruns, `.tflite`, real-board execution, theory-branch large-scale rewriting, sidecar promotion, or direct full-manuscript expansion.

## 2026-06-12 Captain Final Supersession (T79 closeout)

- Current unique task: `T80: 主线校准段落的 bounded prose reopen`
- Task package: `docs/tasks/Phase2/T80_mainline_calibrated_sections_bounded_prose_reopen.md`
- `T79` has been judged `PASS`.
- `T79` has honestly completed one docs-only reopen gate: the repo now has one explicit gate verdict, one section-level readiness matrix, one gap-to-action matrix, and one single recommended next task.
- `T79` introduces no deferred/rejected warning and opens no new risk.
- `R13/R14/R32/R33` remain open, and `T37` remains `blocked + lowest-priority backlog`.
- `T80` is next because the dominant mainline bottleneck is no longer “whether prose reopen is allowed at all”, but “whether the ready sections can be rewritten cleanly without touching methods chapters or upgrading any evidence boundary”.
- `T80` must remain docs-only, mainline-only, and section-bounded: it may rewrite only `Title`、`Abstract`、`Introduction`、`Related Work / positioning`、`Experimental Setup`、`Numerical Results`、`Discussion`、`Conclusion`; it must not widen into methods chapters, benchmark reruns, `.tflite`, real-board execution, theory-branch large-scale rewriting, sidecar promotion, or direct full-manuscript expansion.

## 2026-06-12 Captain Final Supersession (T78 closeout)

- Current unique task: `T79: 论文材料 reopen gate 与 bounded prose 扩写就绪性评审`
- Task package: `docs/tasks/Phase2/T79_paper_reopen_gate_and_prose_readiness_review.md`
- `T78` has been judged `PASS`.
- `T78` has honestly completed the bounded note non-results alignment, `statcalib` hierarchy de-emphasis, layout warning closeout, and scope-bounded note calibration record without touching code, tests, `runs/`, `artifacts/`, or governance docs.
- `T78` introduces no deferred/rejected warning and opens no new risk.
- `R35` is closed by `T78`; `R13/R14/R32/R33` remain open.
- `T79` is next because the mainline bottleneck is no longer note calibration itself, but whether the current note/results/claim-evidence/risk stack is already sufficient to support one bounded prose reopen on `main`.
- `T79` must remain docs-only, mainline-only, and must not widen into benchmark reruns, `.tflite`, real-board execution, theory-branch large-scale rewriting, sidecar promotion, or direct full-manuscript expansion.

## 2026-06-12 Captain Final Supersession (T77 closeout)

- Current unique task: `T78: 论文 note-draft 非结果层校准、statcalib 层级降权与排版 warning 收口`
- Task package: `docs/tasks/Phase2/T78_paper_note_alignment_statcalib_hierarchy_and_layout_closeout.md`
- `T77` has been judged `PASS_WITH_WARNINGS`.
- `T77` has honestly completed the bounded note results-layer synchronization, T76 preview-source / stable-ID traceability hardening, local note compile refresh, and exact-path cleanup of temporary render residue without touching code, tests, `runs/`, `artifacts/`, or governance docs.
- `T77` warning classification:
  - `N1` whole-file `.tex` still contains unsynchronized non-results legacy hunks = `deferred -> R35`
  - `N2` `statcalib` still sits visually too close to the main results layer inside `Numerical Results` = `deferred -> R35`
  - `N3` note `.log` still contains `Underfull \hbox` layout warnings = `deferred -> R35`
  - `N4` section-scope proof still relies on manifest / `% T77-SOURCE` comments rather than a more mechanical audit = `deferred -> R35`
- `R34` is closed by `T77`; `R35` is new; `R13/R14/R32/R33` remain open.
- `T78` is next because the mainline bottleneck has shifted again: the repo no longer lacks note results-layer sync, but it still lacks one bounded note-calibration pass that de-emphasizes `statcalib`, aligns remaining non-results wording to the current evidence stack, and closes the visible LaTeX/layout loose ends before any paper reopen gate.
- `T78` must remain docs-only, mainline-only, and must not widen into benchmark reruns, `.tflite`, real-board execution, theory-branch large-scale rewriting, sidecar promotion, or full-manuscript reopen.

## 2026-06-12 Captain Final Supersession (T76 closeout)

- Current unique task: `T77: 论文 note-draft 结果层同步与 T76 traceability hardening`
- Task package: `docs/tasks/Phase2/T77_paper_note_results_sync_and_traceability_hardening.md`
- `T76` has been judged `PASS_WITH_WARNINGS`.
- `T76` has honestly completed the bounded rendered-QA / Results-assembly step: the repo now has real preview PNGs, contact sheet, PDF bundle, paper-facing QA notes, callout sheet, and Results-section assembly materials under the already locked `T75` asset boundary.
- `T76` warning classification:
  - `N1` preview-source 聚合行字段语义复用 = `deferred -> R34`
  - `N2` `.tmp_t76_*` 探针/缓存残留 = `accepted`
  - `N3` 逐图 QA 结论未内联完整上游 `T74-*` stable ID = `deferred -> R34`
- `R34` is new; `R13/R14/R32/R33` remain open.
- `T77` is next because the mainline bottleneck has shifted from “whether the figures are readable” to “whether the rendered-QA pack is traceability-clean enough and synchronized into the current paper note without reopening the whole manuscript”.
- `T77` must remain docs-only, mainline-only, and must not widen into benchmark reruns, `.tflite`, real-board execution, theory-branch large-scale rewriting, sidecar promotion, or full-manuscript reopen.

## 2026-06-12 Captain Final Supersession (T75 closeout)

- Current unique task: `T76: Rendered figure QA and results-section assembly pack`
- Task package: `docs/tasks/Phase2/T76_rendered_figure_qa_and_results_section_assembly_pack.md`
- `T75` has been judged `PASS`.
- `T75` has honestly completed the bounded authoring step: main-text Results prose, caption/placement lock, appendix bridge, do-not-write guardrails, and three publication-facing `T75-FIG-*` assets are now present and explicitly linked back to `T74` stable IDs.
- `T75` has no blocking issue and no deferred/rejected warning; this closeout opens no new risk item.
- The carry-forward notes are operational rather than blocking: future commits still require precise staging because the worktree contains coexisting governance diffs, and rendered preview QA should now be handled by one new bounded task rather than silently widening `T75`.
- `T76` is next because the remaining paper-material gap is no longer authoring structure, but rendered figure QA plus manuscript-facing Results-section assembly under the already locked `T75` asset and wording boundary.
- `T76` must remain docs-only, mainline-only, and must not widen into benchmark reruns, `.tflite`, real-board execution, theory-branch work, sidecar promotion, or full-manuscript reopen.

## 2026-06-12 Captain Final Supersession (T74 closeout)

- Current unique task: `T75: Main-text results prose and final figure authoring pack`
- Task package: `docs/tasks/Phase2/T75_maintext_results_prose_and_final_figure_authoring_pack.md`
- `T74` has been judged `PASS`.
- `T74` has honestly completed the paper-ready simulation/material packaging step: stable IDs、result tables、caption pack、insertion map、traceability assets and submission-material gap checklist are now present without touching code、tests、`runs/`、`artifacts/` 或治理边界。
- `T74` has no blocking issue and no deferred/rejected warning; this closeout opens no new risk item.
- The only non-blocking carry-forward note is operational: the current worktree contains coexisting captain-side governance diffs, so future commits should use precise staging rather than blanket add-all.
- `T75` is next because the mainline gap has shifted again: the repo now has a strong material pack, but it still lacks one bounded authoring task that converts the `T74` stable-ID route into main-text Results prose, caption lock, and actual publication-facing figure assets.
- `T75` remains docs-only, mainline-only, and must not widen into benchmark reruns, `.tflite`, real-board execution, theory-branch work, sidecar promotion, or full-manuscript reopen.

## 2026-06-12 Captain Final Supersession (T73 closeout)

- Current unique task: `T74: Paper-ready simulation result and figure pack`
- Task package: `docs/tasks/Phase2/T74_paper_ready_simulation_result_and_figure_pack.md`
- `T73` has been judged `PASS`.
- `T73` has honestly completed the post-`T72` mainline ledger refresh across claim/evidence、result/figure、risk and README entry layers without touching code、tests、`runs/`、`artifacts/` 或治理文档。
- `T73` has no blocking issue, no deferred warning, and no rejected warning; this closeout opens no new risk item.
- `T74` is next because the mainline gap has shifted from “whether recent bounded evidence is written back consistently” to “whether the repo has one paper-ready simulation material pack with result tables、figure/caption packs、traceability assets and explicit submission-material gap accounting”.
- The strengthened `T74` package now requires a stricter paper-material bundle: stable IDs、main-text/appendix/supplement placement、insertion map、submission-bundle manifest and task-scoped traceability assets.
- `T74` remains docs-only, mainline-only, and must not widen into benchmark reruns, `.tflite`, real-board execution, theory-branch work, sidecar promotion, or paper prose reopen.

## 2026-06-11 Captain Final Supersession (T72 closeout)

- Current unique task: `T73: Mainline claim/evidence and result/figure/risk ledger refresh`
- Task package: `docs/tasks/Phase2/T73_mainline_claim_evidence_and_result_figure_ledger_refresh.md`
- `T72` has been judged `PASS_WITH_WARNINGS`.
- `T72` closes `R31` honestly: the checked-in read-only real-board gate / transfer-pack now has execution-derived probe limitations、default/override-aware provenance 和 focused override regressions, while the current-host regenerated verdict remains `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`.
- `T72` warning classification:
  - `N1` 最小 config 场景下 path provenance 仍会把代码默认值写成 `source_kind=config_field` = `deferred -> R32`
  - `N2` Worker 原始主报告路径曾短暂落在精确 allowed files 之外，但当前 `HEAD` 已整理回允许目录 = `accepted`
  - `N3` 缺少覆盖 path 字段缺省回退标签的 focused regression = `deferred -> R32`
- `R13/R14` remain open, `T37` remains blocked, `R31` is closed by `T72`, and `R32` is the new narrowed deployment-boundary carry-forward risk.
- `T73` is next because the repo now needs one post-`T72` 主线 paper-facing 三台账刷新入口，把 `T48/T50/T57/T58/T70/T72` 的现状统一回写，而不是继续开启新的执行长跑。
- `T73` is docs-only, mainline-only, and must not widen into benchmark, `.tflite`, real-board execution, theory-branch work, or paper prose reopen.

## 2026-06-11 Captain 优先级调整（paper-first / board-lowest）

- 用户已明确当前暂无可用的 `Linux + FPGA` 硬件宿主。
- 因此 main 分支当前主线顺序调整为：`T73` 台账刷新 -> `T74` 论文可直接复用的仿真结果/图表打包 -> 其余 supporting-material tasks。
- `T37` 继续保持 `blocked + lowest-priority backlog`，不是 `T73` 之后的下一任务。
- 在硬件条件变化前，real-board 方向只保留 truth / gate / provenance 维护，不重新打开真板 execution 任务。

## Current Unique Task

`T84: 主线 note 有界 final polish 与读者化装配包`

Status:

- `T83` has been reviewed as `PASS`.
- `T83` completed the full-note consistency sweep honestly and removed the last “is the note even self-consistent?” blocker before reader-facing final polish.
- `R13/R14/R32/R33` remain open, and `T37` remains blocked.
- 当前暂无 `Linux + FPGA` 硬件宿主，因此 `T37` 同时也是 `resource-blocked / lowest-priority backlog`。
- `T24` remains the authoritative historical frozen ranked table and must continue to be preserved as the anchor.
- `T64/T65/T66/T67/T68/T69/T70` remain bounded mock-backed software-HIL extension-lane evidence only; they are still not `.tflite`, real-board, or mature calibration-comparator validation.
- The current project state remains `Phase 2: Controlled Development / Go` under `Research Reality Recovery Mode`.
- `T84` must remain docs-only and must not rewrite any deployment-boundary, training, FR7/FR8, or mechanism evidence into stronger completed claims.

Why this task is next:

1. `T80`、`T81`、`T82`、`T83` 已分别完成 ready sections、contribution/methods、supporting-boundary 与全文一致性四层收口，当前主线已不再缺“局部 section 的单点补丁”或“全文是否自洽”的 gate。
2. The next bottleneck is one bounded final-polish pass that translates internal task/provenance language into reader-facing wording and assembles the already-accepted note into a tighter main-text/appendix/supplement route.
3. 当前暂无 `Linux + FPGA` 硬件宿主，因此 `T37` 不仅证据未满足，而且属于资源受限 backlog，不应早于 paper-material 主线任务。
4. `T51/T52` full paper re-open tasks 仍然过早；即使 `T83` 已完成，当前也只允许进入一张 bounded final-polish 任务，而不是恢复无界 full-manuscript 扩写或 submission-pack 总装。
5. `T84` is intentionally stronger than a 简单润色任务，因为它必须同时做 reader-facing 术语翻译、Results/appendix/supplement 结构压缩、reader-facing 装配台账与 compile-aware verification。

## Captain Output For Current Task

- Current unique task: `T84`
- Latest reviewed task: `docs/review/T83_review.md` with verdict `PASS`
- T83 closeout:
  - blocking issues = none
  - warning-derived risk changes = none
  - new risk opened = none
  - carry-forward notes = `T83` only authorizes bounded final polish; it does not authorize submission-ready pack, deployment upgrade, or any real-board / `statcalib` boundary promotion
- Next worker-facing task package: `docs/tasks/Phase2/T84_mainline_bounded_final_polish_and_reader_facing_assembly.md`
- `T84` may perform only one bounded reader-facing final-polish pass, one final-polish change map, one terminology translation table, one appendix/supplement assembly map, README registration, and compile-aware refresh if local toolchain is available; it must not touch governance docs, source code, tests, `runs/`, `artifacts/`, stable-ID result assets, or theory-branch large-scale content
- `T84` is the only recommended next mainline task after `T83`

## 并行 Sidecar 扩展实验治理

- Captain-only 设置任务：`PSE0`
- 任务包：`docs/tasks/Phase2/PSE0_parallel_sidecar_extension_governance_setup.md`
- 治理规则：`docs/sidecar/parallel_sidecar_extension_governance.md`
- worktree 计划：`docs/sidecar/parallel_sidecar_worktree_plan.md`
- `PSE0` 不改变当前唯一主线任务；当前唯一任务以 `Current Unique Task` 区块为准。
- `PSE1` 已将 sidecar 改为 main-controlled governance：旧 `.wt/*` 长期分支不再强制同步，后续默认在 main 当前代码基础上做新增-only sidecar helper / module / config；需要并行隔离时再新开短生命周期 worktree 或 clean clone。
- sidecar lane 结果必须使用 `runs/sidecar/<lane_id>/...` run root。
- sidecar 输出不是主线事实，不能改写 `T24`、`T64`、`T65`、`T66`、`T67`、`T68` 或 `T69`。
- sidecar 晋升必须经过后续 Captain gate；`PSE0` 不晋升也不执行任何 sidecar lane。
- Post-PSE0 Wave A setup：已创建 `.wt/tcn`、`.wt/teach`、`.wt/bank`、`.wt/ctrl` 四个隔离 worktree，并分别写入 `S0_design` 任务包；PSE1 后这些旧 worktree 退役为 read-only reference，S0 思路收编到 `docs/sidecar/lane_plans/`；未运行实验，未创建 `runs/sidecar`，未改变主线当前唯一任务机制。

Older numbered lines below this point are historical carry-forward text and are superseded by the current `T75/T76` block above.

1. Current unique task: `T68`
2. `T67` is complete and accepted as `PASS_WITH_WARNINGS`.
3. T67 warning handling:
   - `N1` source-worktree scope-external PDF diff but clean-clone launch preserved provenance = `accepted`
   - `N2` equal-mean tie is not represented explicitly in `better_parameter_point_by_mean_ler` = `accepted`
   - `N3` two comparison rows remain `mixed` = `deferred -> R24`
4. T67 review output: `docs/review/T67_review.md`
5. T68 task package: `docs/tasks/Phase2/T68_fr8_statcalib_generated_only_robustness_bounded_benchmark.md`

## Done Criteria For T65

1. Correct the T64 result-doc wording so execution shape and finish-timestamp provenance match the actual artifacts and accepted T64 task semantics.
2. Add one lightweight audit helper that checks T64 report consistency against `summary.json`, `launch_plan.json`, `progress.jsonl`, and the frozen-subset anchor from `T24`.
3. Add focused regression coverage for the new audit logic.
4. Produce one explicit consistency-audit doc for T64.
5. Create no new run root and do not modify any historical file under `runs/`.
6. Keep the T64 boundary explicit: mock-backed software-HIL only, separate extension lane only, not `.tflite`, not real-board, not a rewrite of `T24`.
7. Keep all changes inside the T65 allowed-file set only, and do not touch `docs/02_experiment_plan.md`.

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
