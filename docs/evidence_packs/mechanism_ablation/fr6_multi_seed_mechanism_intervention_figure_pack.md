# FR6 Multi-Seed Mechanism/Intervention Figure Pack

## 1. Scope

This document packages a bounded paper-facing figure asset for `FR6` using existing evidence only.

- No new benchmark, trace export, intervention, retraining, `.tflite`, or hardware task was run for T58.
- The figure pack is derived only from existing `T54`, `T55`, and `T56` evidence.
- The resulting figure is descriptive and paper-facing, but it does not upgrade `C4` into causal proof or mechanism closure.

## 2. Exact Input-Source Matrix

| Plotted quantity | Source path | Source columns | Aggregation used in T58 |
| --- | --- | --- | --- |
| Panel A baseline gap by seed | `runs/T54_multi_seed_trace_phase_a_20260522/cross_seed_comparison.csv` | `seed_source`, `mode`, `mean_window_ler` | For each `seed_source` and mode, take the simple mean of `mean_window_ler` across all available rows; plot `mean(Gated v5) - mean(Full)` |
| Panel A instability category | `runs/T54_multi_seed_trace_phase_a_20260522/cross_seed_comparison.csv` | `seed_source`, `mode`, `max_delta_b_norm`, `max_committed_b_norm` | Mark a mode as unstable if `max_delta_b_norm > 0.08` and `max_committed_b_norm > 0.5`; classify each seed as `quiet`, `classic`, or `universal` |
| Panel B I1 intervention delta by seed | `runs/T55_multi_seed_i1_probe_20260523/analysis/intervention_summary.csv` | `seed`, `mean_gap_i1_minus_bl`, `verdict`, `n_scenarios_with_data` | Use per-seed summary rows directly; plot `mean(I1) - mean(Gated v5 baseline)` |
| Caption boundary | `docs/evidence_packs/mechanism_ablation/post_t55_mechanism_claim_reframing_gate.md` | T56 wording boundary | Keep the figure descriptive; no causal proof, mitigation-success claim, or mechanism closure wording |

## 3. Figure Specification

### 3.1 Panel definition

- Panel A: seed-wise `T54` baseline gap with category labels
  - negative values mean Gated v5 performs better than Full
  - colors encode `quiet`, `classic`, and `universal`
- Panel B: seed-wise `T55` I1 intervention delta with verdict labels
  - positive values mean the lower-clip intervention is worse than the original Gated v5 baseline
  - colors encode `harmful`, `mixed_or_no_clear_effect`, and `helpful`

### 3.2 Aggregation note

T58 plots the exact quantities stored in `docs/figure_assets/T58_fr6_multi_seed_mechanism_intervention/figure_data.csv`.

For T54-derived baseline gaps, T58 uses one bounded aggregation rule that is directly traceable to `cross_seed_comparison.csv`: simple means over the available per-seed rows in that file. Rounded values can therefore differ slightly from rounded prose summaries in `docs/evidence_packs/mechanism_ablation/multi_seed_trace_generalization_probe.md`, but every plotted value is directly reproducible from the frozen CSV.

## 4. Figure Asset Directory Contents

Directory: `docs/figure_assets/T58_fr6_multi_seed_mechanism_intervention/`

- `build_figure.py`
- `fr6_multi_seed_mechanism_intervention.svg`
- `fr6_multi_seed_mechanism_intervention.png`
- `figure_data.csv`
- `figure_manifest.json`
- `caption.md`

## 5. Paper-Ready Caption

The exact caption used by the figure pack is stored in:

- `docs/figure_assets/T58_fr6_multi_seed_mechanism_intervention/caption.md`

Its bounded reading is:

`Panel A shows that the committed-b instability pattern is broadly present across the locked six-seed pack, while Panel B shows that the tested lower-clip intervention has mixed and mostly harmful outcomes. The figure is descriptive only and must not be read as causal proof or mechanism closure.`

## 6. FR6 Classification

`FR6` should now be marked `ready` as a bounded paper-facing figure pack.

Why `ready` is justified:

1. The repository now contains a final figure asset, a companion export, a figure-data snapshot, a provenance manifest, and a caption.
2. Every plotted value is traceable to frozen `T54/T55` CSV evidence.
3. The caption and report remain inside the `T56` hedge boundary.

What `ready` does not mean:

1. `C4` remains `partial`.
2. The figure does not prove that committed-`b` instability is the root cause.
3. The figure does not show that lowering the clip solves the mechanism story.
4. The figure does not reopen benchmark scope beyond the locked 6-seed, 4-scenario evidence pack.

## 7. Explicit Non-Claims and Residual Limitations

T58 does not support any of the following statements:

1. `The mechanism story is closed.`
2. `The I1 intervention validates a causal fix.`
3. `High committed-b is generally harmful and should be reduced.`
4. `The figure upgrades the project to expanded benchmark evidence.`
5. `The figure says anything about TFLite validation, real-board validation, or broader deployment readiness.`

Residual limitations:

1. The evidence is still confined to the locked six-seed pack and frozen four scenarios.
2. The intervention evidence is one bounded intervention lane (`I1`) only.
3. The remaining risk is interpretive overreach, not missing figure assets.

## 8. Ledger Decision

The paper-facing ledgers can now mark `FR6` as `ready`, but only in this bounded sense:

- `FR6 ready as a descriptive multi-seed mechanism/intervention figure pack`
- not `FR6 closes the mechanism story`
- not `FR6 upgrades C4 to supported`
