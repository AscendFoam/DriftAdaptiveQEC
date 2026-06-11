# T23 Review: P4 Formal Benchmark Protocol Lock and Evidence Gap Audit

**Reviewer**: Claude Code Reviewer (adversarial)
**Date**: 2026-05-10
**Task package**: `docs/tasks/Phase2/T23_p4_formal_benchmark_protocol_lock.md`

---

## Verdict: PASS_WITH_WARNINGS

---

## 1. Task Completion Check

### 1.1 Primary deliverable

`docs/protocols/benchmark/P4_benchmark_formal_protocol.md` has been created and covers all required sections:

- [x] Evidence levels: recovery / development / formal boundary table (Section 2)
- [x] Frozen software boundary: runner entry, config family, inheritance, real_board_policy (Section 3)
- [x] Locked formal matrix: 4 scenarios x 5 modes x paired_seeds x repeats=2 (Section 4)
- [x] Baseline inclusion and exclusion rules (Section 5)
- [x] Paired-seed and statistical reporting rules (Section 6)
- [x] Deep-research recommendation audit table with adopted/deferred/rejected (Section 7)
- [x] Compute budget and execution risk (Section 8)
- [x] Required evidence pack (Section 9)
- [x] Evidence gaps after T23 (Section 10)
- [x] T24 gate: GO_FOR_BOUNDED_FORMAL_SOFTWARE_REVALIDATION + NO_GO_FOR_SCOPE_EXPANSION_INSIDE_T24 (Section 11)
- [x] Explicit non-claims (Section 12)
- [x] `T23 did not run benchmark` stated explicitly (Section 1)

### 1.2 Governance sync

- [x] `docs/04_task_board.md` updated: T23 as current unique task, T21/T22 marked complete, T24-T35 roadmap laid out
- [x] `docs/07_handoff.md` updated: items 32-36 added, section 4 items 21-25, section 6 task summary rewritten for T23, section 7 rewritten
- [x] `docs/08_risks_and_open_questions.md` updated: R15-R18 added, open questions 19-28 added, status note updated
- [x] `docs/protocols/benchmark/P4_benchmark_development_protocol.md` updated: Section 12 added linking to formal protocol

### 1.3 Verification of protocol claims against codebase

| Claim in formal protocol | Verified against codebase | Result |
| --- | --- | --- |
| Config inherits `p4_multiscenario_hybrid_b_long.yaml` | `base_config: p4_multiscenario_hybrid_b_long.yaml` in config | Match |
| Scenarios: `static_bias_theta`, `linear_ramp`, `step_sigma_theta`, `periodic_drift` | Inherited from `p4_multiscenario.yaml` | Match |
| Modes: `ekf`, `ukf`, `constant_residual_mu`, `rls_residual_b`, `hybrid_residual_b` | Defined in `p4_multiscenario_strong_baselines.yaml` | Match |
| `real_board_policy = conditional_extension` | Inherited from base config | Match |
| `experiment.seed = 20260403` | In config | Match |
| `scenario_seed_stride = 1000` | Inherited from base config | Match |
| Runner supports `--run-dir`, `--repeat-start`, `--repeat-stop`, `--resume-only` | `argparse` definitions in runner | Match |
| Runner supports `--paired-seeds` | `argparse` definition and logic in runner | Match |
| Runner produces `launch_plan.json`, `progress.jsonl`, `summary.json`, `comparison.csv`, `delta.csv`, `teacher_scalar_diagnostics.csv`, `report.md` | Writer functions in runner | Match |
| `teacher_scalar_diagnostics.csv` is actually produced | Line 668/699 in runner write this file | Match |
| `aggressive_param_rate`, `slow_update_violation_rate` are real metrics | Extracted and aggregated in runner | Match |
| Runner supports `--repeats` CLI override | Line 42 in runner argparse | Match |

### 1.4 Repeats reconciliation

The config's `p4_multiscenario_hybrid_b_long.yaml` defaults to `repeats: 4`, but the formal protocol locks `repeats = 2`. The runner supports `--repeats` as a CLI override (line 42), so T24 can achieve `repeats=2` without editing the config. This is consistent with Section 3's prohibition on config edits.

### 1.5 Deep-research recommendation audit

The table in Section 7 classifies 12 recommendations across adopted / deferred / deferred_as_followup / partially_adopted / rejected. The classifications are reasonable:

- **adopted** items (strong classical baselines, historical four-scenario set, training/eval seed separation, `.tflite` before real-board): correctly in scope for T24
- **deferred** items (soft-information, extra scenario families, CI stopping, learned variants beyond hybrid_residual_b, rollback/fallback): correctly outside frozen-set revalidation
- **rejected** item (merging T23 into a mega-task): correct, consistent with task package scope

---

## 2. Blocking Issues

None.

---

## 3. Non-blocking Issues

### N1: 7 files modified outside the allowed list

The worker modified the following files that are NOT in the task package's allowed list:

1. `docs/00_project_snapshot.md`
2. `docs/01_legacy_audit.md`
3. `docs/03_hil_p4_boundary_audit.md`
4. `docs/05_decision_log.md`
5. `docs/06_repo_noise_governance.md`
6. `docs/evidence_packs/deployment_boundary/real_board_hil_readiness.md`
7. `docs/tasks/Phase2/T21_phase2_milestone_review.md`

All modifications are governance synchronization (updating task references, adding decision log entries, adding risk entries, backfilling T21 worker output summary). This is the same pattern observed in T22 and accepted by Captain as governance sync rather than Worker scope violation.

**Recommendation**: Captain should decide whether to accept this as governance sync or flag for future task package tightening.

### N2: Protocol does not specify exact CLI shape for T24

Section 9 item 10 requires "the exact CLI shape" in the evidence pack, and Section 3 point 6 lists allowed chunking controls. However, the formal protocol does not itself state the exact CLI command that T24 should use (e.g., `--repeats 2 --paired-seeds --scenario ...`). This is acceptable at the protocol-lock stage but should be specified in the T24 task package.

### N3: `histogram_input_saturation_rate_mean` and `correction_saturation_rate_mean` listed in Section 6.3

These metrics are listed as required statistical outputs but were not individually verified against the runner code in this review. If they exist in the HIL summary output, they will be captured by the comparison pipeline. If not, T24 should report which metrics are actually available versus which were requested but absent.

### N4: `fast_cycle_violation_rate_mean` in Section 6.3

Same as N3 — listed as required but not individually verified against runner output in this review pass.

---

## 4. Missing Tests

Not applicable. T23 is a documentation-only task with read-only verification. No code was changed, no benchmarks were run.

---

## 5. Suspicious Implementation Details

None. The formal protocol does not contain pseudo-completion, mock results, or plan-written-as-fact:

- `T23 did not run benchmark` is stated in Section 1 and reinforced in Section 12
- Evidence levels clearly distinguish recovery / development / formal
- Non-claims section explicitly lists what the document does NOT claim
- T24 gate is conservatively scoped to frozen-set revalidation only

---

## 6. Recommended Next Action

1. Captain should accept T23 as `PASS_WITH_WARNINGS`.
2. N1 (out-of-scope governance files): apply the same judgment as T22 — accept as governance sync.
3. N2/N3/N4: carry forward into T24 task package requirements — T24 should specify the exact CLI shape and should verify which statistical outputs are actually available from the runner.
4. T24 task package should explicitly state `--repeats 2` in the CLI shape to override the config's `repeats: 4` default.
5. After T23 is accepted and committed, create T24 task package scoped to bounded formal software revalidation of the frozen four-scenario, five-mode, repeats=2 matrix.
