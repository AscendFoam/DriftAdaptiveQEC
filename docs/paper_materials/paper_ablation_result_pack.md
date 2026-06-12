# Paper Ablation Result-Pack and Material Ledger

## Scope Note

This document is a hedge-conditioned paper-material ledger, not a paper draft and not a claim-upgrade vehicle.

It inherits the T56 claim table boundaries:

- C4 remains `partial`; no claim is upgraded to causal proof or mechanism closure
- the simple "high committed-b is harmful" framing is not supported as a general explanation
- any second intervention lane remains `deferred pending better question`
- all mechanism-facing statements below stay inside the T56 wording guardrails

It is updated by T57, T58, and T70 to reflect that FR7 and FR6 are no longer historical-only gaps, and that FR8 is no longer simply "missing". The new evidence is still bounded to the frozen software-HIL, six-seed mechanism, and separately labeled `statcalib` extension lanes and must not be over-read as broader benchmark expansion, comparator promotion, or causal closure.

## 1. Ready / Partial / Missing Ledger

| ID | Item | Status | Source data path(s) | Regeneration path | T56/T57 hedge note |
| --- | --- | --- | --- | --- | --- |
| F1 | Seed=20260429 mechanism-diagnosis figure: per-window `teacher_b`, `delta_b`, committed `b`, window outcome | `partial` | `runs/T38_seed20260429_trace_probe_20260513/trace_export/trace_rows.csv`<br>`runs/T38_seed20260429_trace_probe_20260513/trace_export/paired_repeat_comparison.csv` | Manual script from `trace_rows.csv` columns: `teacher_b_q/p`, `predicted_delta_b_q/p`, `committed_b_q/p`, window outcome fields | Single-seed diagnostic only; not causal proof. |
| F2 | Benchmark evidence-boundary diagram: P3 software HIL, T24 revalidation, TFLite boundary, real-board gate/provenance | `ready` | `docs/03_hil_p4_boundary_audit.md`<br>`docs/protocols/benchmark/P4_benchmark_formal_protocol.md`<br>`docs/evidence_packs/deployment_boundary/t48_true_tflite_runtime_gate.md`<br>`docs/evidence_packs/deployment_boundary/t72_real_board_transfer_pack_provenance_hardening.md` | Schematic figure from boundary docs | Safe as boundary diagram. |
| FR1 | System architecture figure: fast loop / slow loop / param bank / HIL boundary | `partial` | `physics/` runtime files<br>`cnn_fpga/runtime/`<br>`cnn_fpga/hwio/` | Schematic figure based on code structure and experiment plan | Architecture-neutral; no mechanism claim. |
| FR4 | Scenario-wise benchmark summary figure | `partial` | `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/comparison.csv` | Plot from `comparison.csv`; regeneration script still not frozen in governance docs | Safe if kept to frozen-set ranking. |
| FR6 | Multi-seed mechanism/intervention figure | `ready` | `docs/evidence_packs/mechanism_ablation/fr6_multi_seed_mechanism_intervention_figure_pack.md`<br>`docs/figure_assets/T58_fr6_multi_seed_mechanism_intervention/fr6_multi_seed_mechanism_intervention.svg`<br>`docs/figure_assets/T58_fr6_multi_seed_mechanism_intervention/figure_data.csv`<br>`docs/figure_assets/T58_fr6_multi_seed_mechanism_intervention/figure_manifest.json`<br>`runs/T54_multi_seed_trace_phase_a_20260522/cross_seed_comparison.csv`<br>`runs/T55_multi_seed_i1_probe_20260523/analysis/intervention_summary.csv` | Regenerate with `docs/figure_assets/T58_fr6_multi_seed_mechanism_intervention/build_figure.py` | Ready as a descriptive figure pack only. Do not claim multi-seed causal closure. |
| FR7 | Feature/teacher ablation result table | `ready` | `runs/p4_benchmark/T57_fr7_feature_teacher_ablation_20260524_000000/summary.json`<br>`runs/p4_benchmark/T57_fr7_feature_teacher_ablation_20260524_000000/comparison.csv`<br>`runs/p4_benchmark/T57_fr7_feature_teacher_ablation_20260524_000000/summary_pack/table.csv`<br>`runs/p4_benchmark/T57_fr7_feature_teacher_ablation_20260524_000000/provenance_manifest.json` | Already regenerated under the locked T24 protocol by T57 | Ready as a bounded frozen-set table only. Not causal attribution and not proof that every teacher channel is necessary. |
| FR8 | Statcalib extension-lane closure / no-promotion summary table | `partial` | `docs/evidence_packs/statcalib_fr8/fr8_statcalib_bounded_closure_pack.md`<br>`docs/review/T70_review.md`<br>`runs/p4_benchmark/T64_fr8_statcalib_extension_lane_20260527_221658/summary.json`<br>`runs/p4_benchmark/T66_statcalib_sensitivity_20260529_210906/statcalib_sensitivity_summary/summary.json`<br>`runs/p4_benchmark/T67_statcalib_teacher_anchor_20260601_225718/statcalib_teacher_anchor_summary/summary.json`<br>`runs/p4_benchmark/T68_statcalib_generated_only_20260605_205723/statcalib_generated_only_summary/summary.json`<br>`runs/p4_benchmark/T69_statcalib_clean_winner_tiebreak_20260608_160358/statcalib_clean_winner_tiebreak_summary/summary.json` | Reassemble from the T70 closure pack and preserved T64/T66/T67/T68/T69 summaries; no new benchmark required | Bound to extension-lane closure only; not an integrated promoted comparator table. |
| FR12 | Latency / commit / violation summary table | `partial` | T24/T57 `comparison.csv` and `summary.json` timing and commit fields | Scriptable from summary files; shape still not frozen in governance docs | Safe as bounded software-HIL observation table. |

## 2. FR6 Outcome Summary

T58 closes the paper-material FR6 gap by turning the existing T54/T55/T56 evidence chain into a reproducible figure pack without running new experiments.

Key bounded readings from `docs/figure_assets/T58_fr6_multi_seed_mechanism_intervention/figure_data.csv`:

| Question | T58 bounded answer | Evidence status | Caution |
| --- | --- | --- | --- |
| Does the instability pattern appear outside the original borderline seed? | Yes. The six-seed figure reproduces the `quiet`, `classic`, and `universal` cross-seed picture already established by T54. | `present` | Descriptive category summary only; not causal proof. |
| Does Gated v5 usually beat Full in the six-seed pack? | Mostly yes. Panel A shows four clear negative-gap classic seeds, one near-tie quiet seed, and one near-tie universal seed. | `present` | Bound to the six-seed T54 evidence pack only. |
| Does the tested I1 clip-reduction intervention reliably help? | No. Panel B shows it is harmful in four seeds, mixed/no-clear-effect in one, and helpful in one. | `present` | One bounded intervention lane only; not mechanism closure. |
| Can the paper now cite a bounded multi-seed mechanism/intervention figure? | Yes. T58 provides a final figure, a companion export, a figure-data snapshot, a provenance manifest, and a caption. | `present` | The figure must stay descriptive and non-causal. |

## 3. FR7 Outcome Summary

T57 closes the historical-only FR7 gap by re-running the full 4-scenario x 6-mode x 2-repeat matrix under the locked T24 feature-ablation protocol.

Key bounded readings from `runs/p4_benchmark/T57_fr7_feature_teacher_ablation_20260524_000000/summary_pack/table.csv`:

| Question | T57 bounded answer | Evidence status | Caution |
| --- | --- | --- | --- |
| Does `hybrid_full` still beat `ukf` on average? | Yes. Average LER improves from `0.817382` to `0.798545` (`dLER=-0.018837`). | `present` | Frozen-set software-HIL only. |
| Does removing histogram delta hurt vs `hybrid_full`? | Yes, in all 4 scenarios and on average (`dLER=+0.028178`). It is also slightly worse than `ukf` on average. | `present` | Supports usefulness of this channel under the frozen lane, not causal proof. |
| Does removing teacher prediction hurt vs `hybrid_full`? | Yes, in all 4 scenarios and on average (`dLER=+0.008706`), though it still remains slightly better than `ukf` on average. | `present` | Bounded evidence only. |
| Does removing teacher params hurt vs `hybrid_full`? | No. It improves in all 4 scenarios and is the best mode in every scenario (`dLER=-0.048924`). | `present` | This weakens any simple "teacher params are a necessary positive contributor" story. |
| Does removing teacher deltas materially hurt vs `hybrid_full`? | Only marginally overall (`dLER=+0.001784`) with mixed per-scenario sign and `aggressive_param` becoming the dominant overflow source. | `present` | Treat as near-neutral/mixed, not a strong necessity signal. |

## 4. Paper-Readiness Assessment After T57/T58/T70

### What is now available

1. A formal FR7 ablation table exists under the locked T24 feature-ablation lane.
2. A formal FR6 figure pack now exists under the locked T54/T55/T56 six-seed evidence lane.
3. A bounded FR8 extension-lane closure input now exists under the explicit T70 no-promotion gate.
4. The paper can now cite feature/teacher ablation evidence without relying on historical pre-T24 runs.
5. The paper can now cite a bounded multi-seed mechanism/intervention figure without treating old narrative text as the figure itself.
6. The result pack now includes a frozen-set ablation provenance manifest, an FR6 figure-pack provenance manifest, and a separate FR8 closure/gate source chain.

### What this does not justify

1. It does not prove a causal mechanism for the benchmark win.
2. It does not justify the claim that the complete teacher-guided residual design is uniformly optimal.
3. It does not justify the claim that teacher params are a necessary positive contributor, because the bounded FR7 table shows the opposite pattern under this reused ablation lane.
4. It does not promote `statcalib` into a mature mainline comparator or rewrite the T24 frozen table.
5. It does not close the broader mechanism-claim gap, TFLite, real-board, training portability, or expanded-benchmark gaps.

### Updated paper stance

- FR6 is no longer a blocker for citing a bounded multi-seed mechanism/intervention figure.
- FR7 is no longer a blocker for citing a bounded ablation table.
- FR8 is no longer simply `missing`; it is now a bounded extension-lane closure input that must carry the `no_promotion_keep_extension_lane_only` boundary.
- FR6 and FR7 do remain blockers for any strong causal or architectural attribution sentence that assumes the mechanism story is closed or that more teacher channels explain the win.
- FR8 remains a blocker for any sentence that treats `statcalib` as a promoted comparator, a T24 replacement, or a unique clean calibration winner.
- The safest paper reading is:
  `Under the frozen T24 software-HIL lane, histogram delta removal clearly hurts, teacher prediction removal mildly hurts, teacher delta removal is near-neutral/mixed, and the reused no-teacher-params variant unexpectedly performs best.`
  For the six-seed mechanism lane, the safest reading is:
  `The instability pattern is broadly present across the locked six-seed pack, and the tested clip-reduction intervention is mixed and mostly harmful; this is descriptive evidence, not causal proof.`

## 5. Regeneration Paths Summary

| Asset | Regeneration type | Concrete steps | Current status |
| --- | --- | --- | --- |
| F1 | Custom script | Read `trace_rows.csv`, plot per-window committed-b / teacher-b / delta-b with window-outcome overlay | Data ready; figure script not frozen |
| F2 | Manual draw | Draw schematic from boundary docs | Ready |
| FR1 | Manual draw | Schematic from code structure and experiment plan | Partial |
| FR4 | Python script | Read T24 `comparison.csv`, plot grouped scenario bars/lines | Partial |
| FR6 | Existing bounded execution + task-scoped figure script | Use `docs/figure_assets/T58_fr6_multi_seed_mechanism_intervention/build_figure.py` to regenerate the figure assets from frozen T54/T55 summaries | Ready |
| FR7 | Existing bounded execution | Use T57 run root plus `summary_pack/table.csv` and `summary_pack/report.md` | Ready |
| FR8 | Existing bounded closure pack + preserved historical run summaries | Reassemble from `docs/evidence_packs/statcalib_fr8/fr8_statcalib_bounded_closure_pack.md` plus the preserved `T64/T66/T67/T68/T69` summary packs; do not create a new run root | Partial |
| FR12 | Python script | Extract timing/commit/violation fields from T24/T57 summary files | Partial |

## 6. Explicit Non-Claims

The following statements must not appear in the paper as completed evidence-backed claims:

1. `Teacher-guided residual-b is proven to be the optimal design choice.`  
   T57 shows the bounded no-teacher-params variant performs best in all 4 scenarios.
2. `Teacher params are a necessary positive contributor to the frozen-set win.`  
   T57 contradicts that strong reading under the reused ablation lane.
3. `FR7 closes the mechanism story.`  
   FR7 closes the result-table gap, not the causal-mechanism gap.
4. `Histogram delta is the single cause of the win.`  
   T57 only shows bounded degradation when that channel is removed.
5. `Teacher-delta removal proves this channel does not matter.`  
   The signal is mixed and near-neutral, not a universal zero-effect proof.
6. `The FR6 figure proves the mechanism story.`  
   T58 only provides a descriptive figure pack built from existing T54/T55/T56 evidence.
7. `FR8 proves statcalib is now a promoted comparator.`  
   T70 explicitly keeps `statcalib` in a separately labeled extension lane with a no-promotion gate.
8. `The paper now has comprehensive empirical coverage.`  
   TFLite portability/default-env, real-board execution, training portability, and expanded-benchmark gaps remain.

## 7. Verdict

After T57/T58/T70, FR6 and FR7 should be treated as `ready` for bounded paper-material use, while FR8 should be treated as `partial` and explicitly tagged as extension-lane closure/no-promotion material.

The new limitation is no longer "FR6 or FR7 is missing." The new limitation is interpretive and boundary-driven: the completed FR6 figure and FR7 table still point to a more complicated non-causal story than the historical narrative, and the new FR8 closure input still does not justify comparator promotion. Paper wording must therefore stay descriptive, separately labeled, and bounded.
