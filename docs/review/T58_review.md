# Review: T58 FR6 Multi-Seed Mechanism/Intervention Figure Pack

Verdict: **PASS_WITH_WARNINGS**

## Blocking Issues

None.

## Non-Blocking Issues

### N1. Scope judgment depends on clarified provenance of mixed governance edits

The workspace diff includes governance-file updates that are outside the T58 Worker allowed-file list:

- `docs/00_project_snapshot.md`
- `docs/01_legacy_audit.md`
- `docs/04_task_board.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`

After the user's clarification, this review treats those edits as **pre-existing Captain-owned changes that were not made by the Worker during T58**.

Under that clarified provenance assumption, T58 itself stays within scope.

This is no longer a blocking issue, but the handoff discipline is still worth tightening: Captain governance sync should ideally be committed or otherwise isolated before Worker review tasks begin, so the review boundary is mechanically obvious from git state.

### N2. The FR6 deliverable itself is materially complete and bounded

The core T58 output exists and is coherent:

- `docs/figure_assets/T58_fr6_multi_seed_mechanism_intervention/`
  - `build_figure.py`
  - `fr6_multi_seed_mechanism_intervention.svg`
  - `fr6_multi_seed_mechanism_intervention.png`
  - `figure_data.csv`
  - `figure_manifest.json`
  - `caption.md`
- `docs/fr6_multi_seed_mechanism_intervention_figure_pack.md`
- paper-facing ledgers updated to mark `FR6` as `ready` in the bounded figure-pack sense

The figure, caption, and ledgers remain descriptive. I did not find wording that upgrades T54/T55/T56 into causal proof, mechanism closure, `.tflite` validation, real-board validation, or expanded benchmark evidence.

### N3. The seed-category logic is re-derived inside the helper script

`build_figure.py` re-derives the `quiet / classic / universal` labels using hard-coded thresholds:

- `max_delta_b_norm > 0.08`
- `max_committed_b_norm > 0.5`

This matches the current T54/T55 evidence picture, including the final six-seed categories used in `docs/multi_seed_trace_generalization_probe.md`. So this is not a fake result.

However, the provenance is weaker than it could be because the category labels are not read from a frozen T54 category table; they are reconstructed by T58 task-local logic. That makes the figure-pack slightly more fragile to future reinterpretation.

### N4. The worker-written T58 review is a self-check, not an adversarial review

The existing `docs/review/T58_review.md` written by the worker reads as a delivery checklist, not an independent review, and it misses the scope problem above.

That is not itself a scope breach, because the task package explicitly asked the worker to create a review file. But it should not be treated as the final acceptance review.

## Missing Tests

### M1. No recorded clean regeneration check for the figure pack

For a figure-pack task with a generator script, the review would be stronger if the worker had recorded a no-ambiguity regeneration check such as:

1. regenerate `figure_data.csv`, `figure_manifest.json`, `caption.md`, `svg`, `png`
2. confirm the regenerated outputs match the checked-in outputs

The current materials are internally consistent, and I spot-checked the plotted values against:

- `runs/T54_multi_seed_trace_phase_a_20260522/cross_seed_comparison.csv`
- `runs/T55_multi_seed_i1_probe_20260523/analysis/intervention_summary.csv`

But that verification is not documented by the worker as a formal regeneration step.

## Suspicious Implementation Details

### S1. Category provenance is implicit rather than frozen as a first-class source table

The figure pack uses real T54/T55 evidence, but the `quiet / classic / universal` regime labels are not sourced from an explicit frozen CSV column. Instead, T58 reconstructs them from raw summary fields and hard-coded thresholds.

That is acceptable for a bounded figure-pack helper script, but it is the main place where T58 introduces new derivation logic rather than only repackaging already-labeled evidence.

### S2. Worker verification text depends on diff provenance, not raw workspace state

Without provenance context, the raw workspace state makes that statement look too strong. With the user's clarification that the mixed governance edits are Captain-owned and predate Worker execution, the statement becomes substantively acceptable, but it is not self-evident from git alone.

## Recommended Next Action

1. Accept T58 as `PASS_WITH_WARNINGS` under the clarified assumption that the mixed governance edits were Captain-owned pre-existing changes, not Worker changes.
2. Keep the FR6 figure-pack assets and the paper-facing ledger updates.
3. Optionally strengthen provenance in a later bounded cleanup:
   - either cite the T54 category derivation more explicitly in `figure_manifest.json`
   - or add a frozen seed-category table to the figure-pack inputs
4. Tighten future collaboration hygiene:
   - Captain governance sync should be committed or isolated before Worker execution
   - Worker-facing reviews should start from a mechanically clean task boundary when possible
