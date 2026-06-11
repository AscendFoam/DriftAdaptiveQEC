# Multi-Seed Mechanism / Intervention Plan and Trace Pack

## 1. Status and Scope

This document is a docs-only planning gate. It does not:

- run benchmark, training, `.tflite`, hardware, or cleanup
- modify source code, config, `runs/`, or `artifacts`
- claim multi-seed confirmation or causal proof already exists
- reopen the frozen-set benchmark boundary (T45)
- upgrade existing single-seed evidence into multi-seed confirmation

Its purpose is to define the smallest credible next step for mechanism evidence: a bounded multi-seed and intervention-oriented trace plan that could later test the current hypothesis without overbuilding the task.

## 2. Current Supported Mechanism Statement

### 2.1 What current artifacts can support

The following statement is defensible today:

> For `seed=20260429`, the Gated v5 teacher-guided residual-b path exhibits combined committed-`b` instability: large teacher-`b` amplitude plus large residual `delta_b` amplitude produce committed `b` values much larger than the Full variant, leading to scenario/repeat-sensitive LER degradation. This is trace-supported diagnostic evidence from 4798 per-window trace rows (T38), consistent with the residual-amplitude / teacher-delta regime instability hypothesis narrowed by T36.

Evidence anchors:
- T36: `docs/evidence_packs/mechanism_ablation/seed20260429_failure_diagnosis.md` — summary-level diagnosis
- T38: `docs/evidence_packs/mechanism_ablation/seed20260429_trace_export_diagnosis.md` — per-window trace diagnosis
- T38 run: `runs/T38_seed20260429_trace_probe_20260513/`
- T38 trace: 4798 rows across 4 scenarios × 2 modes × 2 repeats

### 2.2 What makes this statement narrow

1. It applies to one seed only (`20260429`).
2. It is diagnostic, not causal — no intervention was tested.
3. It does not prove whether the teacher amplitude shift or the CNN residual amplitude shift is the first upstream cause.
4. It does not test whether reducing residual amplitude (e.g., lower clip) would fix the instability without harming other seeds.

### 2.3 Mechanism update from T36 + T38

| Candidate mechanism | T36 label | T38 label | Current standing |
| --- | --- | --- | --- |
| Sign offset | `not answerable` | `partially observed but not leading` | Sign flips are real but not the primary differentiator |
| Magnitude overshoot | `plausible / partially supported` | `supported` | Strongest trace-level separation is persistent high-amplitude delta_b |
| Response lag | `not supported` | (unchanged) | Ruled out by matching scheduler statistics |
| Teacher prediction instability | `partially supported` | `partially supported` | Teacher-b amplitude is much larger in Gated v5, but not sole root cause |
| CNN residual output instability | (not separately labeled) | `supported` | Raw delta_b is much larger, frequently clipped, flips sign |
| Committed combined-b instability | (not separately labeled) | `strongly supported` | Clearest trace-level explanation: large teacher_b + large delta_b → unstable committed b |

## 3. Stronger Unsupported Claims and Remaining Evidence Gap

### 3.1 What remains unsupported today

The following statements cannot be made with current evidence:

1. "Combined committed-`b` instability is the root cause of Gated v5 degradation across all seeds." — only tested on one seed.
2. "Lowering the residual clip from 0.12 to a smaller value would stably improve Gated v5 across all seeds." — no intervention has been tested.
3. "The teacher-delta amplitude regime shift is the upstream trigger that causes the instability." — teacher amplitude and CNN residual amplitude are confounded in current trace data.
4. "Gated v5 is fundamentally flawed; no parameter tuning can make it stable." — v8/v9 showed over-conservative variants also fail, but no targeted amplitude intervention has been tested.

### 3.2 The evidence gap in one sentence

Current evidence narrows the mechanism to combined committed-`b` instability on one seed, but does not establish whether this mechanism generalizes across seeds or whether a targeted amplitude intervention would stably close the gap.

### 3.3 Claim-boundary table

| Statement | Evidence level | What would upgrade it |
| --- | --- | --- |
| "seed=20260429 shows combined committed-b instability" | `trace-supported diagnostic` (C4 partial) | Already at this level from T38 |
| "The instability generalizes to other seeds" | `unsupported` | Multi-seed trace evidence showing same pattern |
| "Reducing residual amplitude stably improves outcomes" | `unsupported` | Intervention experiment showing improvement across seeds |
| "The mechanism is causal" | `unsupported` | Intervention that reliably changes outcome across seeds |

## 4. Minimal Seed-Selection Logic

### 4.1 Existing seeds

Three seeds have paired/chunked Gated v5 vs Full evidence from the v5 chunked paired benchmark:

| Seed | Full LER | Gated v5 LER | Gap | Observation |
| --- | ---: | ---: | ---: | --- |
| 20260427 | 0.8066 | 0.6205 | -0.1861 | Gated v5 clearly better |
| 20260428 | 0.8326 | 0.5944 | -0.2382 | Gated v5 clearly better |
| 20260429 | 0.6374 | 0.6397 | +0.0024 | Near tie / Gated v5 slightly worse |

Source: `runs/teachrepr_v5_chunked_pair/paired_20260427_220702/summary.csv`

### 4.2 Why additional seeds are needed

1. 20260429 is the only seed where Gated v5 fails to clearly beat Full. A mechanism story anchored on one outlier seed is not paper-grade evidence.
2. It is unknown whether 20260429's pattern (committed-b instability, high-amplitude delta regime) replicates on other seeds, or whether 20260429 is genuinely unique.
3. The intervention hypothesis (lower residual amplitude → more stable outcome) requires testing on both "good" seeds (to confirm no regression) and "bad" or "borderline" seeds (to confirm improvement).

### 4.3 Seed-selection criteria

New seeds should be selected to satisfy:

1. **Spaced from existing seeds**: avoid exact neighbors of 20260427/20260428/20260429 to reduce the chance of near-identical RNG trajectories.
2. **Predeclared**: all seeds must be listed before execution begins.
3. **Small upper bound**: no more than 3 new seeds, for a total of 6 including the existing 3.
4. **Trace-export compatible**: must use the same T38 trace-export path.

### 4.4 Proposed seed pack

| Seed | Role | Why included |
| --- | --- | --- |
| 20260430 | New test seed | First seed after the existing block; tests whether the 20260429 pattern continues |
| 20260425 | New test seed | Seed before the existing block; provides temporal separation |
| 20260510 | New test seed | Seed from a different date range; provides calendar separation |

Upper-bound philosophy: the total multi-seed pack should not exceed 6 seeds. If the pattern is not visible within 6 seeds, adding more seeds is unlikely to produce a clean mechanism story without first addressing the underlying intervention question.

### 4.5 Seed-selection table

| Category | Seeds | Count | Purpose |
| --- | --- | ---: | --- |
| Existing with Full vs Gated v5 paired evidence | 20260427, 20260428, 20260429 | 3 | Already have summary-level evidence; may need trace export for new seeds |
| New test seeds | 20260430, 20260425, 20260510 | 3 | Test generalization of committed-b instability pattern |
| **Total** | | **6** | Upper bound; do not exceed without separate task |

## 5. Minimal Trace Schema and Required File/Field Inventory

### 5.1 Trace-field inventory

The following fields are required per window. All were already verified as present by T38's `field_availability.json`.

| # | Field | Source | Required for mechanism test |
| --- | --- | --- | --- |
| 1 | `scenario` | `summary.raw_rows[].scenario` | Yes — group by scenario |
| 2 | `mode` | `summary.raw_rows[].mode` | Yes — Full vs Gated v5 |
| 3 | `repeat` | `summary.raw_rows[].repeat` | Yes — within-seed variance |
| 4 | `seed` | `summary.raw_rows[].seed` | Yes — cross-seed comparison |
| 5 | `window_id` | `host_events[].readout.window.window_id` | Yes — per-window chronology |
| 6 | `teacher_b_q` | `host_events[].proposed_params.metadata.teacher_params.b[0]` | Yes — teacher amplitude regime |
| 7 | `teacher_b_p` | `host_events[].proposed_params.metadata.teacher_params.b[1]` | Yes — teacher amplitude regime |
| 8 | `raw_delta_b_q` | `host_events[].proposed_params.metadata.raw_delta_b[0]` | Yes — CNN output before clip |
| 9 | `raw_delta_b_p` | `host_events[].proposed_params.metadata.raw_delta_b[1]` | Yes — CNN output before clip |
| 10 | `applied_delta_b_q` | `host_events[].proposed_params.metadata.applied_delta_b[0]` | Yes — residual after clip |
| 11 | `applied_delta_b_p` | `host_events[].proposed_params.metadata.applied_delta_b[1]` | Yes — residual after clip |
| 12 | `committed_b_q` | `host_events[].proposed_params.b[0]` | Yes — committed-b instability |
| 13 | `committed_b_p` | `host_events[].proposed_params.b[1]` | Yes — committed-b instability |
| 14 | `window_ler` | window diagnostics | Yes — outcome per window |
| 15 | `overflow_ratio` | window diagnostics | Yes — overflow regime |
| 16 | `correction_saturation_ratio` | window diagnostics | Yes — saturation regime |
| 17 | `mean_correction_utilization` | window diagnostics | Optional — secondary signal |

Fields 6-13 are the core mechanism fields. Fields 14-16 are outcome/diagnostic fields. Field 17 is a secondary signal that may help interpretation but is not strictly required.

### 5.2 Summary-row requirements

The following summary rows are minimally required for cross-seed comparison:

1. **Per-scenario, per-mode, per-repeat**: already produced by T38
2. **Per-scenario, per-mode, cross-repeat mean/std**: already produced by T38
3. **Per-scenario, per-mode, per-seed, cross-repeat mean**: new requirement for multi-seed
4. **Cross-seed, cross-scenario, per-mode mean**: new requirement for paper-facing comparison
5. **Per-seed delta-b amplitude regime classification**: new diagnostic summary

### 5.3 Comparison rows

| Comparison type | Required | Purpose |
| --- | --- | --- |
| Full vs Gated v5 per scenario/seed/repeat | Yes | Primary mechanism comparison |
| Full vs Intervention variant per scenario/seed/repeat | Yes (if intervention executed) | Causal test |
| Cross-seed Full vs Gated v5 gap distribution | Yes | Generalization test |
| Delta-b amplitude regime: Full vs Gated v5 per seed | Yes | Mechanism narrowing |
| Teacher-b amplitude regime: Full vs Gated v5 per seed | Yes | Upstream attribution |

## 6. Minimal Future Comparison Pack

### 6.1 Execution-pack table

| # | Item | Seeds | Scenarios | Modes | Repeats | Estimated runs |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 1 | Trace export for existing seeds (20260427, 20260428) | 2 | 4 | 2 (Full, Gated v5) | 2 | 32 |
| 2 | Trace export for new seeds (20260425, 20260430, 20260510) | 3 | 4 | 2 (Full, Gated v5) | 2 | 48 |
| 3 | Intervention variant: lower residual clip (e.g., 0.06) | 6 | 4 | 1 (Gated v5 intervention) | 2 | 48 |
| **Total** | | **6** | **4** | | | **128** |

This is the upper bound. The execution task may start with items 1-2 (trace export only) and defer item 3 (intervention) to a separate bounded task.

### 6.2 File output requirements

For each seed/scenario/mode/repeat combination, the execution task should produce:

1. `hil_events.json` (existing format, contains all trace fields)
2. `hil_summary.json` (existing format, contains aggregate metrics)
3. Trace CSV export (same format as T38 `trace_rows.csv`)

Cross-seed summary outputs:

4. `cross_seed_comparison.csv` — Full vs Gated v5 gap per seed/scenario
5. `delta_b_amplitude_by_seed.csv` — delta-b regime classification per seed
6. `mechanism_summary.csv` — committed-b instability indicator per seed/scenario

## 7. Intervention or Counterfactual Matrix

### 7.1 Classification of candidate interventions

| # | Intervention idea | Type | Why it is a mechanism test | Risk |
| --- | --- | --- | --- | --- |
| I1 | Lower `residual_clip_b` from 0.12 to 0.06 for Gated v5 | **Mechanism test** | Directly tests "does reducing delta-b amplitude stabilize committed b?" If LER improves on 20260429 and does not regress on 20260427/20260428, the amplitude hypothesis is strengthened. | May over-constrain; needs both good and bad seeds |
| I2 | Lower `residual_scale_b` from 1.0 to 0.5 for Gated v5 | **Mechanism test** | Tests "does scaling down residual output amplitude (pre-clip) produce the same effect as lower clip?" Helps distinguish clip-boundary artifact from amplitude-regime effect. | Partially confounded with I1; may be redundant |
| I3 | Teacher-delta attenuation: scale `teacher_delta_b_q/p` inputs by 0.5 | **Mechanism test** | Tests "does attenuating the teacher-delta channel (which shifts to high-activity regime on 20260429) reduce instability?" | Affects CNN input, not just output; broader perturbation |
| I4 | New gated architecture (v10+) with different gate/activation structure | **Not a mechanism test** | This is model-design work, not a mechanism test. It would blur the intervention with architecture changes. | Rejected for this task |
| I5 | Different loss function (e.g., L1 instead of MSE) | **Not a mechanism test** | Training-time change that would require retraining and blur mechanism testing with model improvement. | Rejected for this task |
| I6 | Remove teacher-delta channels entirely | **Not a mechanism test** | Equivalent to a new variant (removing features), not a targeted intervention. Already partially covered by No TeacherParams evidence. | Rejected for this task |

### 7.2 Recommended intervention pack

If a future execution task includes intervention testing, the recommended minimal pack is:

1. **I1** (lower residual clip) — highest priority, directly targets the amplitude hypothesis
2. **I3** (teacher-delta attenuation) — secondary priority, tests upstream channel contribution

I2 can be deferred because it is partially confounded with I1. I4/I5/I6 are out of scope for mechanism testing.

### 7.3 What counts as diagnostic vs causal evidence

| Evidence type | Definition | Current status | What would produce it |
| --- | --- | --- | --- |
| Diagnostic | "Pattern X is observed on seed Y" | Available for seed=20260429 from T36+T38 | Multi-seed trace export (items 1-2 from execution pack) |
| Generalized diagnostic | "Pattern X is observed across N seeds" | Not yet available | Multi-seed trace export showing same committed-b instability pattern |
| Causal (mechanism test) | "Intervention Z reliably changes outcome Y" | Not yet available | Intervention experiment (I1 or I3) showing improvement on 20260429 without regression on good seeds |
| Causal (upstream attribution) | "Changing parameter A changes intermediate B which changes outcome Y" | Not yet available | Would require upstream-downstream trace decomposition, which is beyond minimal scope |

## 8. Diagnostic Versus Causal Evidence Boundary

### 8.1 The boundary in one statement

- **Diagnostic evidence** answers: "What pattern do we observe?"
- **Causal evidence** answers: "Does changing X reliably change Y?"

T36 + T38 produced diagnostic evidence. No causal evidence exists yet.

### 8.2 Language rules for the boundary

| Current evidence level | Safe wording | Unsafe wording |
| --- | --- | --- |
| Single-seed diagnostic | "trace-supported diagnosis suggests" / "single-seed trace evidence supports" | "mechanism proven" / "root cause identified" / "causal evidence" |
| Multi-seed diagnostic | "pattern observed across N seeds" / "multi-seed trace evidence consistent with" | "confirmed across seeds" / "robust mechanism evidence" |
| Single-intervention causal | "intervention X changed outcome Y on seed Z" / "intervention test suggests" | "causal proof" / "mechanism validated" / "confirmed by intervention" |
| Multi-seed multi-intervention | "causal test results consistent across seeds" | Still not "causal proof" without controlled experimental design |

### 8.3 What would constitute paper-grade causal evidence

Paper-grade causal evidence would require:

1. The committed-b instability pattern is observed across ≥3 seeds (including 20260429).
2. A targeted amplitude intervention (I1 or I3) improves outcome on the unstable seed(s) without degrading outcome on the stable seed(s).
3. The improvement is consistent across ≥3 scenarios.
4. The intervention does not introduce new instability modes (e.g., over-constraining the residual).

Until all four conditions are met, the mechanism claim should remain at the diagnostic level.

## 9. Go / No-Go Recommendation for a Later Execution Task

### 9.1 Go condition

A later bounded execution task should proceed if:

1. The task proposes ≤3 new seeds (total ≤6 including existing).
2. The task uses only the existing frozen four scenarios (no scenario expansion).
3. The task uses only Full and Gated v5 modes (plus at most one intervention variant).
4. The task uses the existing T38 trace-export path without modifying runtime semantics.
5. The task produces trace evidence and cross-seed summaries only.
6. The task does not modify source code semantics (config-only intervention variants are acceptable).
7. The task does not claim causal proof from trace evidence alone.

### 9.2 No-Go triggers

A proposed execution task should be blocked if:

1. It proposes more than 6 total seeds.
2. It adds new scenario families beyond the frozen four.
3. It adds new baselines or modes beyond Full, Gated v5, and one intervention variant.
4. It modifies benchmark runner semantics, formal protocol, or frozen-set anchor.
5. It claims causal proof from diagnostic trace evidence.
6. It mixes mechanism testing with benchmark expansion, deployment validation, or new model training.
7. It reopens T45 frozen-set separation.

### 9.3 Recommended phased approach

**Phase A (trace-only, minimal):**
- Run trace export for 3 new seeds + 2 existing seeds (20260427, 20260428) using the same T38 path
- Produce cross-seed comparison summaries
- Determine whether committed-b instability generalizes

**Phase B (intervention, conditional):**
- Only if Phase A shows the pattern generalizes
- Run one intervention variant (I1: lower residual clip) on all 6 seeds
- Compare outcomes: does the intervention help without harming?

Phase A is the recommended first execution task. Phase B should only proceed if Phase A produces a positive signal.

## 10. Explicit Non-Claims

This plan does not claim:

1. that multi-seed confirmation already exists
2. that the committed-b instability hypothesis is proven
3. that any intervention will succeed
4. that benchmark expansion or new model architectures are part of this plan
5. that the frozen-set benchmark boundary is being reopened
6. that `.tflite` runtime, real-board validation, or training reproducibility are affected
7. that this plan constitutes executed evidence rather than a planning gate
8. that any reference document (e.g., `docs/reference/延伸改进思路.md`) defines current mainline truth
