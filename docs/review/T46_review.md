# Review: T46 — Adversarial Review

Verdict: **PASS**

Reviewer: independent adversarial reviewer (not the worker self-review that previously occupied this file)

## Blocking Issues

None.

## Non-Blocking Issues

### N1 Seed 20260430 is numerically adjacent to 20260429, limiting RNG-trajectory separation

Section 4.4 proposes 20260430 as a "new test seed" with the rationale "First seed after the existing block; tests whether the 20260429 pattern continues." This is a defensible design choice — it explicitly tests whether the instability extends to a neighboring seed. However, the seed-selection criteria in Section 4.3 also state "Spaced from existing seeds: avoid exact neighbors of 20260427/20260428/20260429 to reduce the chance of near-identical RNG trajectories." The word "exact neighbors" could be read to exclude 20260430, or to only exclude duplicates. The plan's text is internally consistent if "exact neighbors" means "the same seed value," but the wording creates tension.

Classification: `accepted` — the rationale for including 20260430 is explicitly stated and scientifically valid (testing whether the pattern continues to an adjacent seed). The tension in the criteria wording is minor and should be resolved at execution time.

### N2 Governance file changes appear in the working tree alongside T46 worker output

`docs/05_decision_log.md` and `docs/07_handoff.md` show unstaged changes in the same working tree snapshot as T46 output. The worker correctly states these changes were pre-existing (D-2026-05-22-01 decision entry and handoff sync). The task's Forbidden Scope explicitly forbids editing `docs/05_decision_log.md` and `docs/07_handoff.md`. However, the Captain-level governance sync that produced these changes is clearly distinguishable from worker output by content and date.

Classification: `accepted` — the governance changes are Captain-level, not worker-level. The worker did not modify forbidden governance files. Captain should acknowledge these changes separately during integration.

### N3 No execution time or resource estimate for Phase A

Section 6.1 estimates 80 runs for Phase A (trace export for 5 seeds) and 128 total for Phase A + B. While the run count is bounded, the plan does not provide a time estimate. T38's trace export for a single seed with 4 scenarios × 2 modes × 2 repeats took non-trivial wall-clock time. A rough time estimate would help the future execution task's scope definition.

Classification: `accepted` — time estimation is the responsibility of the future execution task, not this planning gate. The run count ceiling (80 for Phase A) is sufficient for scope control.

### N4 The plan does not directly reference `docs/paper_materials/paper_claim_evidence_ledger.md` for C4 status tracking

The claim-boundary table (Section 3.3) correctly labels the current evidence as `trace-supported diagnostic (C4 partial)` and the worker verification record correctly states "C4 保持 partial，无 evidence level 升级." However, the plan does not provide a back-reference to the paper claim evidence ledger. A reader who wants to trace C4's full status history must separately consult `docs/paper_materials/paper_claim_evidence_ledger.md` and `docs/reality_recovery/01_claim_evidence_table.md`.

Classification: `accepted` — the plan's audience is the mechanism execution task, not the paper ledger. The C4 label is correct and does not overclaim. A cross-reference would be nice but is not required.

### N5 Worker self-review (previously in this file) was accurate and thorough

The worker's self-review identified 4 non-blocking issues (N1: 3-seed sample size, N2: clip factor, N3: existing seed reuse, N4: I3 implementation feasibility). All 4 are valid observations. The self-review verdict of PASS is consistent with the evidence. No errors or overclaims were found in the worker's self-review.

Classification: `informational` — the worker's self-assessment was honest.

## Scope and Boundary Confirmation

### Allowed files check

| File | Status | Allowed? |
| --- | --- | --- |
| `docs/evidence_packs/mechanism_ablation/seed_mechanism_multi_seed_plan.md` | New | Yes |
| `docs/review/T46_review.md` | New (worker self-review, now overwritten by adversarial review) | Yes |
| `docs/for_human/T46_explanation.md` | New | Yes |
| `docs/tasks/Phase2/T46_multi_seed_mechanism_intervention_plan_and_trace_pack.md` | Modified (Worker Output + Verification Record appended) | Yes |

No other files created or modified by the worker.

### Forbidden scope check

| Forbidden action | Verified |
| --- | --- |
| No source code, benchmark code, config, test, runtime, hardware, or training file edited | `git diff --name-only -- runs/ artifacts/ cnn_fpga/ physics/ benchmark/ tests/` returns empty |
| No `runs/`, `artifacts/`, or new run/result directories | Confirmed |
| No governance docs other than the task package itself | Worker did not edit `05_decision_log.md` or `07_handoff.md`; those changes are pre-existing Captain sync |
| No benchmark, training, `.tflite`, hardware, or cleanup command started | Confirmed by diff and worker output |
| No rewrite of T36/T38 single-seed evidence as multi-seed confirmation | Grep confirms all mentions of "multi-seed confirmation" / "causal proof" / "mechanism proven" / "root cause identified" appear only in negation/forbidden context |
| No widening into T47/T48/T49 | Plan explicitly scopes out benchmark expansion, deployment, and model design |

### Evidence level check

| Claim | Plan's label | Honest? |
| --- | --- | --- |
| seed=20260429 committed-b instability | `trace-supported diagnostic` (C4 partial) | Yes |
| Instability generalizes to other seeds | `unsupported` | Yes |
| Intervention improves outcome | `unsupported` | Yes |
| Mechanism is causal | `unsupported` | Yes |

### Frozen-set boundary check

The plan references T45's locked benchmark expansion protocol, uses only the existing frozen four scenarios, and does not propose new baselines or scenario families. T45 is not reopened.

## Missing Tests

Not applicable — this is a docs-only planning gate. No code was modified, so no tests are required.

## Suspicious Implementation Details

None. The plan is well-structured, honestly scoped, and internally consistent.

## Structural Completeness

| Required section (from task package) | Present? |
| --- | --- |
| Status and scope | Yes (Section 1) |
| Current supported mechanism statement | Yes (Section 2) |
| Stronger unsupported claims and evidence gap | Yes (Section 3) |
| Minimal seed-selection logic | Yes (Section 4) |
| Minimal trace schema and field inventory | Yes (Section 5) |
| Minimal future comparison pack | Yes (Section 6) |
| Intervention or counterfactual matrix | Yes (Section 7) |
| Diagnostic vs causal evidence boundary | Yes (Section 8) |
| Go / no-go recommendation | Yes (Section 9) |
| Explicit non-claims | Yes (Section 10) |

| Required table | Present? |
| --- | --- |
| Claim-boundary table | Yes (Section 3.3) |
| Seed-selection table | Yes (Section 4.5) |
| Trace-field inventory table | Yes (Section 5.1) |
| Future execution-pack table | Yes (Section 6.1) |

All required structural elements are present.

## Recommended Next Action

1. Captain should accept T46 as **PASS**.
2. The recommended next bounded task is a Phase A trace-only probe:
   - Export trace data for seeds 20260427, 20260428 (from existing artifacts, if field-complete) and 20260425, 20260430, 20260510 (from new bounded reruns using the same T38 trace-export path).
   - Produce cross-seed comparison summaries.
   - Determine whether committed-b instability generalizes beyond seed=20260429.
3. Phase B (intervention I1) should only be proposed after Phase A results are available and show a positive generalization signal.
4. If Phase A shows the pattern does not generalize, C4 remains `partial` and the paper should use appropriate diagnostic hedging rather than causal language.
5. Captain should separately acknowledge the pre-existing governance changes in `docs/05_decision_log.md` (D-2026-05-22-01) and `docs/07_handoff.md` during integration.
