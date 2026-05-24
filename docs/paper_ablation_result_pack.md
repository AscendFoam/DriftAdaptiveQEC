# Paper Ablation Result-Pack and Material Ledger

## Scope Note

This document is a hedge-conditioned paper-material ledger, not a paper draft and not a claim-upgrade vehicle.

It inherits the T56 claim table boundaries:

- mechanism claims are explicitly classified as retain / weaken / retire / reframe / still-open
- C4 remains `partial`; no claim is upgraded to causal proof or mechanism closure
- the simple "high committed-b is harmful" framing is not supported as a general explanation
- any second intervention lane remains `deferred pending better question`
- all mechanism-facing statements below are bounded by the T56 wording guardrails

This ledger builds on the existing `docs/reality_recovery/04_figure_and_result_ledger.md` (frozen 2026-05-19) and extends it with regeneration paths, ablation-specific gaps, and a paper-readiness assessment conditioned on T56.

## 1. Ablation/Result Pack Scope

### What is in scope

1. Frozen-set benchmark ranking (C2, C3): the core empirical result
2. Seed=20260429 mechanism-diagnosis figure (C4 partial): bounded single-seed diagnostic
3. System architecture figure: dual-loop / param-bank / HIL boundary schematic
4. Scenario-wise benchmark summary: bar charts or LER curves per scenario
5. Feature/teacher ablation status: what ablation evidence exists and what is missing
6. Latency/commit/violation summary table
7. Benchmark boundary / evidence-level table
8. Training reproducibility boundary table
9. Deployment / readiness boundary table
10. Statcalib comparator status table (interface-only)

### What is out of scope (not in this pack)

- Multi-seed mechanism/intervention figure (deferred by T56)
- True `.tflite` runtime latency/accuracy figure (blocked by R12)
- Real-board smoke evidence figure/table (blocked by R13, R14)
- Statcalib integrated comparator result table (blocked by R24)
- Cross-platform training reproducibility figure (blocked by R11)
- Any expanded benchmark beyond the frozen set (blocked by R5, R9)

## 2. Ready / Partial / Missing Ledger

### 2.1 Figures

| ID | Item | Status | Source data path(s) | Regeneration path | T56 hedge note |
| --- | --- | --- | --- | --- | --- |
| F1 | Seed=20260429 mechanism-diagnosis figure: per-window `teacher_b`, `delta_b`, committed `b`, window outcome | `partial` | `runs/T38_seed20260429_trace_probe_20260513/trace_export/trace_rows.csv`<br>`runs/T38_seed20260429_trace_probe_20260513/trace_export/paired_repeat_comparison.csv` | Manual script from `trace_rows.csv` columns: `teacher_b_q/p`, `predicted_delta_b_q/p`, `committed_b_q/p`, window outcome fields | Must be captioned as single-seed diagnostic evidence, not causal proof. T56 M4 retired "high committed-b is harmful" — do not use this figure to imply the instability is harmful. |
| F2 | Benchmark evidence-boundary diagram: P3 software HIL, T24 revalidation, TFLite boundary, real-board readiness | `ready` | `docs/03_hil_p4_boundary_audit.md`<br>`docs/P4_benchmark_formal_protocol.md`<br>`docs/TFLite_runtime_bootstrap.md`<br>`docs/real_board_hil_readiness.md` | Schematic figure; draw.io or equivalent based on boundary descriptions in source docs | No direct mechanism claim. Safe as boundary diagram. |
| F3 | Training portability / reproducibility figure | `blocked` | `docs/training_chain_portable_dependency_lock_plan.md`<br>`docs/training_chain_cpu_cleanenv_train_smoke.md` | Not producible. Replace with boundary table (T3). | N/A — blocked by R11. |
| FR1 | System architecture figure: fast loop / slow loop / param bank / HIL boundary | `partial` | `physics/` runtime files<br>`cnn_fpga/runtime/`<br>`cnn_fpga/hwio/` | Schematic figure based on `docs/02_experiment_plan.md` Section 2 and code structure. Generation script not yet frozen. | Architecture is mechanism-neutral. No T56 hedge needed. |
| FR4 | Scenario-wise benchmark summary figure: LER bar or line per scenario | `partial` | `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/comparison.csv` | Python script reading `comparison.csv`, grouping by scenario and mode, plotting `final_ler_mean` with error bars from `final_ler_std`. Script does not exist as a reusable asset. | Not mechanism-adjacent. Safe if focused on frozen-set ranking. |
| FR6 | Multi-seed mechanism/intervention figure | `missing` | No data. T55 intervention results exist but T56 defers second intervention and reframes the mechanism story. | Cannot produce. T56 section 4 verdict: `deferred pending better question`. Any such figure would require a reframed intervention question and new execution task. | T56 M10: second intervention is `still-open` and deferred. Do not claim interventional causal evidence. |
| FR7 | Feature/teacher ablation result table | `missing` | Historical P4 features ablation results exist but predate T24 frozen-set protocol. No formal revalidation under current locked protocol. | Requires a bounded re-execution of feature ablation under the T24 protocol (Full vs No HistDelta vs No TeacherPred vs No TeacherParams vs No TeacherDelta). This is the single largest evidence gap in the current ablation pack. | Ablation table would interact with T56 M9 (weakened: seed categories do not predict intervention outcomes). The ablation must not be used to support a "which feature causes failure" causal claim. |
| FR8 | Statcalib comparator result table | `missing` | `cnn_fpga/decoder/statcalib.py` (interface only) | No integrated slow-loop run exists. Requires statcalib integration task beyond current roadmap. | N/A — blocked by R24. |
| FR10 | True `.tflite` runtime latency/accuracy figure | `blocked` | `docs/TFLite_runtime_bootstrap.md` | Not producible. Requires TFLite runtime environment. | N/A — blocked by R12. |
| FR11 | Real-board smoke evidence figure/table | `blocked` | `docs/real_board_hil_readiness.md` | Not producible. Requires hardware and bitstream. | N/A — blocked by R13, R14. |
| FR12 | Latency / commit / violation summary table | `partial` | T24 `comparison.csv` and `summary.json` fields: `fast_cycle_violation_rate_mean`, `slow_update_violation_rate_mean`, `n_commits_applied`, `correction_saturation_rate_mean`, `aggressive_param_rate_mean` | Python script extracting these fields from T24 summary files. Shape and format not frozen. Correction saturation is structural zero (R20), requiring a footnote. | Not mechanism-adjacent. Safe if bounded to T24 observations. |

### 2.2 Tables

| ID | Item | Status | Source data path(s) | Regeneration path | T56 hedge note |
| --- | --- | --- | --- | --- | --- |
| T1 | Frozen-set benchmark ranking summary | `ready` | `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/comparison.csv` | `comparison.csv` contains all rows: scenario, mode, `final_ler_mean`, `final_ler_std`, `overflow_rate_mean`, etc. Generate LaTeX/markdown table by grouping scenario and ranking by `final_ler_mean`. | Must be labeled `mock-backed software HIL formal software revalidation`. Not "comprehensive benchmark." |
| T2 | Benchmark boundary / evidence-level table | `ready` | `docs/03_hil_p4_boundary_audit.md`<br>`docs/P4_benchmark_formal_protocol.md` | Manual table from protocol docs; no data generation needed. | Safe because it is a non-claim boundary table. |
| T3 | Training reproducibility boundary table | `ready` | `docs/training_chain_portable_dependency_lock_plan.md`<br>`docs/training_chain_cpu_cleanenv_train_smoke.md` | Manual table from bootstrap docs. | No mechanism claim. Safe as boundary table. |
| T4 | Deployment / readiness boundary table | `ready` | `docs/03_hil_p4_boundary_audit.md`<br>`docs/TFLite_runtime_bootstrap.md`<br>`docs/real_board_hil_readiness.md` | Manual table from boundary docs. | No mechanism claim. Safe as non-claim table. |
| T5 | Statcalib comparator status table | `partial` | `cnn_fpga/decoder/statcalib.py`<br>`tests/test_statcalib_interface.py` | Manual table from interface code and tests. Benchmark-evidence cells must remain `blocked`. | N/A — statcalib is orthogonal to mechanism. |

### 2.3 Ablation-Specific Items

The paper thesis requires feature/teacher ablation evidence to support the claim that the teacher-guided residual-b design is the key contributor to the frozen-set win. The current ablation evidence situation:

| Ablation question | Existing evidence | Status in this pack | Gap |
| --- | --- | --- | --- |
| Does removing histogram delta degrade LER? | Historical (pre-T24): Yes, LER degrades below UKF | `missing` from T24 formal protocol | No formal re-execution under locked protocol and 6-seed pack |
| Does removing teacher prediction degrade LER? | Historical (pre-T24): Yes, but still above UKF | `missing` from T24 formal protocol | Same — historical only |
| Does removing teacher params improve LER? | Historical (pre-T24): Apparent advantage, but flips with seed | `missing` from T24 formal protocol | Known to be seed-dependent; T56 M2 weakened the "primary cause" framing |
| Does removing teacher deltas degrade LER? | Historical (pre-T24): Minimal effect | `missing` from T24 formal protocol | Marginal channel; lowest priority for revalidation |
| Is the Gated v5 advantage reproducible under T24 protocol? | Historical (pre-T24): Yes, 3 seed × 4 scenario | `partial` | Not re-executed under T24 locked protocol with 6-seed pack |
| Is the I1 intervention harmful or helpful? | T55: Mostly harmful (4/6 harmed) | `present` under T55 protocol | Not an ablation; it is an intervention. T56 M4 retired "high committed-b is harmful" |

## 3. Regeneration Paths Summary

| Asset | Regeneration type | Concrete steps | Estimated effort |
| --- | --- | --- | --- |
| F1 | Custom script | Read `trace_rows.csv`, plot per-window committed-b / teacher-b / delta-b with window-outcome overlay | Low (data exists, one-seed) |
| F2 | Manual draw | Schematic from boundary descriptions in source docs | Low |
| FR1 | Manual draw | System architecture from code structure and experiment plan Section 2 | Medium |
| FR4 | Python script | Read `comparison.csv`, group by scenario, bar chart with error bars | Low |
| FR12 | Python script | Extract timing/commit/violation fields from T24 summary files | Low |
| T1 | Python script | Read `comparison.csv`, group by scenario, rank by `final_ler_mean` | Low |
| T2-T5 | Manual | From source docs | Low |
| FR7 | **New execution needed** | Re-run feature ablation under T24 protocol: 5 variants × 4 scenarios × 2 repeats × paired seeds | High (~40 runs) |

## 4. Paper-Readiness Assessment

### Can the current paper proceed without additional ablation evidence?

**Yes, but with explicit limitations.**

The current evidence pack supports a bounded paper thesis:

1. A working dual-loop teacher-guided residual-b decoding framework exists and is operational under mock-backed software HIL (C1).
2. Under the frozen-set formal protocol, hybrid_residual_b wins all four drift scenarios against five classical baselines (C2, C3).
3. One clean-environment CPU-only training smoke has been executed (C5).
4. A statcalib interface contract and focused tests exist (C9).
5. A bounded single-seed trace-supported mechanism diagnosis is available (C4 partial).
6. The system boundary, deployment gap, and evidence limitations are well-documented.

**What the paper cannot claim without FR7 (feature ablation re-execution):**

1. **"Teacher-guided residual-b is the key design choice that causes the benchmark win"** — without formal ablation evidence under the locked T24 protocol, the paper cannot attribute the win to specific architectural features. The historical ablation evidence is pre-T24 and was not re-executed under the frozen protocol.
2. **"Histogram delta is the critical input channel supporting the win"** — same reason. The historical ablation conclusion (experiment plan stable conclusion 9.1 item 8) is not backed by T24-grade evidence.
3. **"Removing teacher params harms performance"** — the historical ablation evidence showed seed-dependent flip, and T56 further weakened the causal interpretation.

**What the paper can claim without FR7:**

1. The frozen-set ranking result (who won, by how much) — this is fully supported by T24.
2. The single-seed trace diagnosis — this is supported by T38/T54/T56, with explicit C4 partial wording.
3. The system architecture and boundary — this is documentation, not experimental evidence.

**Recommended stance:**

- If the paper can be positioned as an **evidence-bounded methods description** with the frozen-set ranking as the core empirical result and explicit disclosure of the ablation gap, then FR7 is a quality booster rather than a hard blocker.
- If the paper needs to make **strong architectural attribution claims** ("the teacher-guided residual design explains the win"), then FR7 becomes a hard blocker and must be re-executed before submission.

### T56 Hedge Conditioning for Paper Claims

Every paper-facing claim that touches the mechanism story must respect these boundaries from the T56 claim table:

| Paper section | T56 constraint | Required wording guardrail |
| --- | --- | --- |
| Abstract | M4 retired, M8 reframed | Do not write "instability is harmful" or "instability must be fixed." If mentioned, write "committed-b instability is broadly present and mostly correlates with Gv5 advantage." |
| Introduction / contribution bullets | C4 partial, M2 weakened | Contribution must not claim mechanism understanding beyond single-seed diagnostic evidence. |
| Mechanism subsection | M1 retain, M3 retain, M7 still-open | May state that seed=20260429 shows committed-b instability (M1) and the pattern generalizes to 5/6 seeds (M3). Must not claim teacher-delta causation (M7 still-open). |
| Results / F1 caption | M4 retired, M8 reframed | Figure caption must state "single-seed diagnostic trace, not causal proof." Must not imply the instability is harmful. |
| Limitations section | All T56 non-claims | Must surface C4 partial status, M7/M10 still-open, and the deferred second-intervention lane. |

## 5. Explicit Non-Claims

The following statements must not appear in the paper as completed evidence-backed claims, even if they could be supported by future ablation re-execution:

1. **"Feature/teacher ablation is complete under the formal protocol."** — FR7 is missing. Only historical pre-T24 ablation evidence exists.
2. **"Teacher-guided residual-b is proven to be the optimal design choice."** — Requires ablation evidence under formal protocol that does not exist yet.
3. **"The committed-b instability is harmful and should be reduced."** — T56 M4 retired this claim. M8 reframes it as a feature, not a defect.
4. **"The multi-seed mechanism story is understood."** — T56 M7 (teacher-delta causation) is `still-open`. M10 (second intervention) is `still-open` and deferred.
5. **"The second intervention lane is justified and will be executed."** — T56 Section 4 verdict: `deferred pending better question`.
6. **"The paper makes comprehensive empirical claims."** — The ablation result pack is intentionally bounded. FR7 is missing. FR6 is deferred.
7. **"The ablation result pack is complete and frozen."** — Two of the three missing items (FR7, FR6) from the original ledger remain unresolved. Only FR8 (statcalib) is intentionally deferred by roadmap.
8. **"The mechanism-evidence gap is closed."** — C4 remains `partial`. M2, M4, M6, M9 are weakened or retired. M7 and M10 remain `still-open`.
9. **"T47 closes the material gap for paper submission."** — This ledger identifies the gaps. It does not create new evidence. The paper may proceed only with explicit disclosure of the remaining ablation and mechanism gaps.
