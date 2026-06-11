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
- [x] T59: Statcalib separate comparator lane integration and bounded smoke
  - Task package: `docs/tasks/Phase2/T59_statcalib_comparator_lane_integration_and_smoke.md`
  - Output: `docs/statcalib_comparator_lane_smoke.md`
  - Review output: `docs/review/T59_review.md`
  - Captain verdict: `PASS_WITH_WARNINGS`
  - Warning handling:
    - W1 cross-mode `teacher_mode` fallback coupling = `deferred`
    - W2 smoke-doc key-name mismatch = `accepted`
    - W3 dirty-worktree smoke provenance weakness = `deferred`
  - Result: T59 closes the first integrated statcalib lane smoke gap, but it does not open FR8 and does not constitute formal comparator evidence
- [x] T60: Statcalib lane isolation and regression hardening
  - Task package: `docs/tasks/Phase2/T60_statcalib_lane_isolation_and_regression_hardening.md`
  - Output: `docs/statcalib_lane_isolation_and_regression_hardening.md`
  - Review output: `docs/review/T60_review.md`
  - Captain verdict: `PASS`
  - Warning handling: no blocking issues and no new `accepted / deferred / rejected` warning item
  - Result: T60 closes the cross-mode semantics and regression-coverage blocker from T59; it closes `W1`/`R26`, but it does not close `R27` and does not open FR8
- [ ] T61: Statcalib clean-provenance fairness sanity rerun
  - Task package: `docs/tasks/Phase2/T61_statcalib_clean_provenance_fairness_sanity.md`
  - Output: `docs/statcalib_fairness_sanity.md`
  - Review output: `docs/review/T61_review.md`
  - Captain verdict: `BLOCK`
  - Blocking issue: launch clean `HEAD=9174065`, but final `summary.json git_commit=6058f42`; mid-run branch movement means the task did not close the clean-provenance blocker it was created to repair
  - Result: the bounded fairness signal persisted, but `R27` remains open and `T61` is not complete
- [x] T62: Statcalib provenance-isolated fairness rerun
  - Task package: `docs/tasks/Phase2/T62_statcalib_provenance_isolated_fairness_rerun.md`
  - Output: `docs/statcalib_provenance_isolated_fairness_rerun.md`
  - Review output: `docs/review/T62_review.md`
  - Captain verdict: `PASS`
  - Warning handling: no blocking issues and no new `accepted / deferred / rejected` warning item
  - Result: T62 closes the T61 provenance blocker and provides one provenance-clean bounded fairness sanity rerun, but it still does not open FR8 and does not upgrade the evidence beyond mock-backed software-HIL scope
- [x] T63: FR8 statcalib comparator gate review
  - Task package: `docs/tasks/Phase2/T63_fr8_statcalib_comparator_gate_review.md`
  - Output: `docs/fr8_statcalib_comparator_gate_review.md`
  - Review output: `docs/review/T63_review.md`
  - Captain verdict: `PASS`
  - Result: T63 closes the pre-FR8 gate-discussion lane honestly; the repository may now open exactly one bounded FR8 extension-lane task, but T63 is not itself FR8 evidence and does not close `R24`
- [x] T64: FR8 statcalib extension-lane bounded benchmark
  - Task package: `docs/tasks/Phase2/T64_fr8_statcalib_extension_lane_bounded_benchmark.md`
  - Output: `docs/fr8_statcalib_extension_lane_benchmark.md`
  - Review output: `docs/review/T64_review.md`
  - Captain verdict: `PASS_WITH_WARNINGS`
  - Warning handling:
    - N1 execution-shape wording drift in the result doc = `deferred`
    - N2 finish-timestamp provenance wording drift = `deferred`
    - N3 extension-lane over-interpretation boundary = `deferred`
  - Result: T64 closes one clean-provenance bounded FR8 extension-lane benchmark on the locked four-scenario protocol without rewriting `T24`, but the evidence remains mock-backed software-HIL only
- [x] T65: FR8 extension-lane consistency guard and report closeout
  - Task package: `docs/tasks/Phase2/T65_fr8_extension_lane_consistency_guard_and_closeout.md`
  - Output: `docs/fr8_statcalib_extension_lane_consistency_audit.md`
  - Review output: `docs/review/T65_review.md`
  - Captain verdict: `PASS_WITH_WARNINGS`
  - Warning handling:
    - N1 mixed-diff scope acceptance depends on explicit user/captain clarification = `accepted`
    - N2 T64-specific audit helper is intentionally narrow, not generic FR8 framework = `accepted`
    - N3 review wording should have stated the clarification dependency more explicitly = `accepted`
  - Result: T65 closes the T64 report/artifact consistency gap and makes the T64 result pack self-audited and safer to reuse, but it does not close `R24`
- [x] T66: FR8 statcalib sensitivity bounded benchmark
  - Task package: `docs/tasks/Phase2/T66_fr8_statcalib_sensitivity_bounded_benchmark.md`
  - Output: `docs/statcalib_sensitivity_bounded_benchmark.md`
  - Review output: `docs/review/T66_review.md`
  - Captain verdict: `PASS_WITH_WARNINGS`
  - Warning handling:
    - `N1` duplicate-running progress-log artifact after same-run-root timeout relaunch = `accepted`
    - `N2` aggregate-best vs stability-best split = `deferred`
    - `N3` `static_bias_theta / statcalib_high_threshold` best row still carries aggregate `statcalib_status = mixed` = `deferred`
  - Result: T66 closes one bounded local-grid robustness gap under clean provenance, but it does not close `R24` and does not upgrade statcalib into a mature calibration comparator
- [x] T67: FR8 statcalib teacher-anchor dependence bounded benchmark
  - Task package: `docs/tasks/Phase2/T67_fr8_statcalib_teacher_anchor_dependence_bounded_benchmark.md`
  - Output: `docs/statcalib_teacher_anchor_bounded_benchmark.md`
  - Review output: `docs/review/T67_review.md`
  - Captain verdict: `PASS_WITH_WARNINGS`
  - Warning handling:
    - `N1` source-worktree scope-external PDF diff but clean-clone launch preserved provenance = `accepted`
    - `N2` equal-mean tie is not represented explicitly in `better_parameter_point_by_mean_ler` = `accepted`
    - `N3` two comparison rows remain `mixed` = `deferred`
  - Result: T67 closes the gross teacher-anchor dependence question honestly, but it does not close `R24`; the strongest aggregate statcalib lane still is not a clean generated-only result pack
- [x] T68: FR8 statcalib generated-only robustness bounded benchmark
  - Task package: `docs/tasks/Phase2/T68_fr8_statcalib_generated_only_robustness_bounded_benchmark.md`
  - Output: `docs/statcalib_generated_only_robustness_bounded_benchmark.md`
  - Review output: `docs/review/T68_review.md`
  - Captain verdict: `PASS_WITH_WARNINGS`
  - Warning handling:
    - `N1` full generated-only winner set remains a tie, not a unique final threshold = `deferred -> R24`
    - `N2` some predeclared candidates remain `mixed` even though the bounded existence question is closed = `deferred -> R24`
    - `N3` clean short-path clone launch boundary must remain visible in downstream retellings = `accepted`
  - Result: T68 closes the bounded generated-only existence question honestly; full generated-only winners now exist inside the predeclared grid, but the strongest clean answer is still a tied `window_variance t001/t003/t005` set and `R24` remains open
- [x] T69: FR8 statcalib clean-winner tie-break bounded benchmark
  - Task package: `docs/tasks/Phase2/T69_fr8_statcalib_clean_winner_tiebreak_bounded_benchmark.md`
  - Output: `docs/statcalib_clean_winner_tiebreak_bounded_benchmark.md`
  - Review output: `docs/review/T69_review.md`
  - Captain verdict: `PASS_WITH_WARNINGS`
  - Warning handling:
    - `N1` persistent clean tie set is the honest bounded answer, not a unique final threshold = `accepted`
    - `N2` bounded-matrix-only conclusion must remain explicit in downstream retellings = `accepted`
  - Result: T69 closes the bounded clean-winner tie-break question honestly; no unique clean reference point emerges, and the strongest clean answer remains the persistent `window_variance_t001 = t003 = t005` tie set. `R24` remains open as a reporting/promotion boundary only, not as an unresolved tie-break execution question
- [x] T70: FR8 statcalib bounded closure pack and promotion gate
  - Task package: `docs/tasks/Phase2/T70_fr8_statcalib_bounded_closure_pack_and_promotion_gate.md`
  - Output: `docs/fr8_statcalib_bounded_closure_pack.md`
  - Review output: `docs/review/T70_review.md`
  - Captain verdict: `PASS`
  - Result: T70 closes the FR8 mainline closure-pack gap honestly; the repository now has one code-backed closure artifact that preserves `T24` as the authoritative frozen ranked table, preserves `statcalib` as a separately labeled extension lane, gives an explicit `no_promotion_keep_extension_lane_only` gate, and gives an explicit `future_selection_task_required` gate for any later single-threshold choice

### 并行 Sidecar 扩展实验治理

- [x] PSE0：并行 sidecar 扩展实验治理设置
  - 任务包：`docs/tasks/Phase2/PSE0_parallel_sidecar_extension_governance_setup.md`
  - 治理输出：`docs/parallel_sidecar_extension_governance.md`
  - worktree 规划输出：`docs/parallel_sidecar_worktree_plan.md`
  - Captain 状态：docs-only 设置任务已完成并通过验证
  - 边界：本任务不执行 `T69`，不创建 sidecar worktree，不启动实验，也不改变任何主线 benchmark 语义
  - 结果：后续 sidecar lane 可在 frozen-anchor、artifact-schema、promotion-gate、run-dir 和红线规则下规划；主线当前唯一任务仍以 `Current Unique Task` 区块为准

### Milestone 2Q: Deployment Boundary Boosters (proposed)

- [x] T48: True `.tflite` runtime smoke gate
  - Task package: `docs/tasks/Phase2/T48_true_tflite_runtime_smoke_gate.md`
  - Output: `docs/t48_true_tflite_runtime_gate.md`
  - Review output: `docs/review/T48_review.md`
  - Captain verdict: `PASS`
  - Result: T48 closes one narrow current-host true `.tflite` runtime truth gap honestly; the repository now has one isolated `tensorflow==2.21.0` environment on this machine that can real-load and real-execute preserved `static_theta_v2` float / int8 `.tflite` artifacts and can run bounded source-vs-`.tflite` consistency checks, but this does not restore default-environment compatibility and does not upgrade the evidence to HIL, real-board, or deployment closure
- [x] T49: Real-board smoke execution gate
  - Task package: `docs/tasks/Phase2/T49_real_board_smoke_execution_gate.md`
  - Output: `docs/t49_real_board_smoke_execution_gate.md`
  - Review output: `docs/review/T49_review.md`
  - Captain verdict: `PASS_WITH_WARNINGS`
  - Warning handling:
    - `W1` device-path readiness counts openable paths without enforcing `mmio + dma` role split = `deferred -> R30`
    - `W2` role-aware regression and checked-in-artifact replay regression are still missing = `deferred -> R30`
    - `W3` checked-in read-only regeneration entrypoint for the full gate artifact pack is still missing = `deferred -> R30`
  - Result: T49 closes one honest current-host real-board gate pack with verdict `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`; no real-board smoke was executed, and `R13/R14` remain open but narrower
- [x] T71: Real-board gate regeneration and host-transfer pack
  - Task package: `docs/tasks/Phase2/T71_real_board_gate_regeneration_and_host_transfer_pack.md`
  - Output: `docs/t71_real_board_gate_regeneration_pack.md`
  - Review output: `docs/review/T71_review.md`
  - Captain verdict: `PASS_WITH_WARNINGS`
  - Warning handling:
    - `W1` `probe_limitations` 将未实际执行的限制写成既成事实 = `deferred -> R31`
    - `W2` `source_records` / `expected_byte_count_basis` 仍写死默认 config 口径 = `deferred -> R31`
    - `W3` `--config` / `--mmio-path` / `--dma-path` 的 provenance/override 回归不足 = `deferred -> R31`
    - `W4` collector 继续 import `BoardFPGAConfig` 作为 repo 内 config 读取入口 = `accepted`
  - Result: T71 closes the R30 gap honestly by hardening role-aware gate logic, adding a checked-in read-only collector, and proving replay/regeneration consistency; it still does not unlock `T37` or validate any real-board execution
- [ ] T72: Real-board transfer-pack provenance hardening
  - Task package: `docs/tasks/Phase2/T72_real_board_transfer_pack_provenance_hardening.md`

### Milestone 2R: Reproducibility And Material Pack (proposed)

- [x] T50: Training reproducibility and material-regeneration pack
  - Task package: `docs/tasks/Phase2/T50_training_reproducibility_and_material_regeneration_pack.md`
  - Output: `docs/training_reproducibility_and_material_regeneration_pack.md`
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

## 2026-06-11 Captain Final Supersession (T71 closeout)

- Current unique task: `T72: Real-board transfer-pack provenance hardening`
- Task package: `docs/tasks/Phase2/T72_real_board_transfer_pack_provenance_hardening.md`
- `T71` has been judged `PASS_WITH_WARNINGS`.
- `T71` closes `R30` honestly: the repository now has one checked-in、read-only、role-aware、可 replay / regeneration 的 real-board gate pack，且 current-host regenerated verdict 仍是 `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`。
- `T71` warning classification:
  - `W1` fixed `probe_limitations` text not derived from actual probing = `deferred -> R31`
  - `W2` fixed default-config `source_records` / `expected_byte_count_basis` text = `deferred -> R31`
  - `W3` missing focused regression for `--config` / `--mmio-path` / `--dma-path` provenance behavior = `deferred -> R31`
  - `W4` collector keeps repo-internal `BoardFPGAConfig` import = `accepted`
- `R13/R14` remain open, `T37` remains blocked, and `R31` is now the active deployment-boundary carry-forward risk.
- `T72` is the next bounded mainline task because the remaining question is no longer whether the gate exists, but whether the transfer-pack provenance is execution-derived and override-safe enough for future-host reuse.
- `T72` must stay on main only, remain isolated from theory-branch work, and must not widen into benchmark, `.tflite`, real-board execution, or paper reopen.

## Current Unique Task

`T72: Real-board transfer-pack provenance hardening`

Status:

- `T71` has been reviewed as `PASS_WITH_WARNINGS`.
- `T71` closes `R30` honestly: the checked-in regeneration path now exists and the current-host regenerated verdict remains `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`.
- `T71` warning handling is deferred rather than blocking:
  - `W1` fixed `probe_limitations` text not derived from actual probing = `deferred -> R31`
  - `W2` fixed default-config `source_records` / `expected_byte_count_basis` text = `deferred -> R31`
  - `W3` missing focused regression for `--config` / `--mmio-path` / `--dma-path` provenance behavior = `deferred -> R31`
  - `W4` collector keeps repo-internal `BoardFPGAConfig` import = `accepted`
- `R13/R14` remain open but narrower: the current-host truth is no longer unknown; it is explicitly blocked by missing openable device paths, missing bound bitstream/RTL/DMA contract evidence, and a placeholder repo execution path.
- `T24` remains the authoritative historical frozen ranked table and must continue to be preserved as the anchor.
- `T64/T65/T66/T67/T68/T69/T70` remain bounded mock-backed software-HIL extension-lane evidence only; they are still not `.tflite`, real-board, or mature calibration-comparator validation.
- The current project state remains `Phase 2: Controlled Development / Go` under `Research Reality Recovery Mode`.
- `T72` must remain isolated from benchmark/HIL/sidecar/paper-reopen/theory-branch outputs, and it must not rewrite provenance hardening as real-board validation or perform write-side MMIO/DMA/register actions.

Why this task is next:

1. `T49` has already answered the first honest current-host real-board question: this machine is presently `NO_GO`, not “unknown”.
2. The next unresolved deployment-boundary question is now reproducibility and future-host portability of that gate, not immediate board execution on the current host.
3. `T37` remains blocked, and `T51/T52` paper re-open tasks are still premature before the repo has a checked-in, role-aware, read-only gate regeneration path.
4. `T71` is materially stronger than a docs-only task because it requires gate-logic hardening, a checked-in read-only artifact collector, replay/regression coverage, and one current-host regeneration pack that agrees with `T49`.

## Captain Output For Current Task

- Current unique task: `T72`
- Latest reviewed task: `docs/review/T71_review.md` with verdict `PASS_WITH_WARNINGS`
- T71 closeout:
  - `W1/W2/W3` = `deferred -> R31`
  - `W4` = `accepted`
  - `R30` = closed by T71
- Next worker-facing task package: `docs/tasks/Phase2/T72_real_board_transfer_pack_provenance_hardening.md`
- `T72` may harden only `cnn_fpga/hwio/collect_t71_real_board_gate_artifacts.py` plus focused tests/docs/task-scoped artifacts; it must not touch `board_backend.py` / `axi_map.py` / `dma_client.py`, must not modify governance docs, must not modify canonical configs, must not create `runs/`, and must not perform write-side MMIO/DMA/register actions or any board benchmark / board execution

## 并行 Sidecar 扩展实验治理

- Captain-only 设置任务：`PSE0`
- 任务包：`docs/tasks/Phase2/PSE0_parallel_sidecar_extension_governance_setup.md`
- 治理规则：`docs/parallel_sidecar_extension_governance.md`
- worktree 计划：`docs/parallel_sidecar_worktree_plan.md`
- `PSE0` 不改变当前唯一主线任务；当前唯一任务以 `Current Unique Task` 区块为准。
- sidecar lane 必须使用 `codex/sidecar-*` 分支、隔离 worktree 和 `runs/sidecar/<lane_id>/...` run root。
- sidecar 输出不是主线事实，不能改写 `T24`、`T64`、`T65`、`T66`、`T67`、`T68` 或 `T69`。
- sidecar 晋升必须经过后续 Captain gate；`PSE0` 不晋升也不执行任何 sidecar lane。
- Post-PSE0 Wave A setup：已创建 `.wt/tcn`、`.wt/teach`、`.wt/bank`、`.wt/ctrl` 四个隔离 worktree，并分别写入 `S0_design` 任务包；未运行实验，未创建 `runs/sidecar`，未改变主线当前唯一任务机制。

Older numbered lines below this point are historical carry-forward text and are superseded by the current `T71` block above.

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
