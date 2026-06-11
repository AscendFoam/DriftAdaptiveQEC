# T58: FR6 multi-seed mechanism/intervention figure pack

## Status

- Proposed by Captain on `2026-05-26`
- Current phase: `Phase 2: Controlled Development`
- Decision state: `Go`
- Task type: docs-only paper-material task using existing bounded evidence only

## Why This Task Exists

`T57` has now closed `FR7` honestly. The reviewer explicitly identifies the next bounded paper-material gap as `FR6`, not another FR7 rerun.

The repository already has the bounded evidence needed for `FR6`:

- `T54` provides the multi-seed trace-only generalization pack
- `T55` provides the bounded I1 intervention comparison pack
- `T56` freezes the post-I1 mechanism wording boundary

What is still missing is a paper-facing figure pack that turns those existing results into a reproducible, bounded, non-causal figure artifact and updates the ledgers accordingly.

This task is not a new experiment lane.

## Goal

Produce a bounded FR6 figure pack that answers:

1. whether the current repository can now support a paper-facing multi-seed mechanism/intervention figure
2. exactly which existing T54/T55 data tables feed that figure
3. whether `FR6` should become `ready`, remain `partial`, or stay `missing`
4. what caption and interpretation boundaries must remain explicit after `T56`

## Allowed Files

Worker may modify only:

- `docs/tasks/Phase2/T58_fr6_multi_seed_mechanism_intervention_figure_pack.md`
- `docs/evidence_packs/mechanism_ablation/fr6_multi_seed_mechanism_intervention_figure_pack.md`
- `docs/paper_materials/paper_ablation_result_pack.md`
- `docs/paper_materials/paper_claim_evidence_ledger.md`
- `docs/reality_recovery/04_figure_and_result_ledger.md`
- `docs/reality_recovery/05_paper_claim_risk_table.md`
- `docs/review/T58_review.md`
- `docs/for_human/T58_explanation.md`
- `docs/worker_summary/T58_worker_summary.md`

Worker may create:

- one task-scoped figure asset directory under `docs/figure_assets/T58_fr6_multi_seed_mechanism_intervention/`
- only derived figure assets, figure-data snapshots, figure manifests, captions, and helper scripts inside that one task-scoped directory

## Docs To Update

This task should update only:

1. `docs/evidence_packs/mechanism_ablation/fr6_multi_seed_mechanism_intervention_figure_pack.md`
2. `docs/paper_materials/paper_ablation_result_pack.md`
3. `docs/paper_materials/paper_claim_evidence_ledger.md`
4. `docs/reality_recovery/04_figure_and_result_ledger.md`
5. `docs/reality_recovery/05_paper_claim_risk_table.md`
6. `docs/review/T58_review.md`
7. `docs/for_human/T58_explanation.md`
8. `docs/worker_summary/T58_worker_summary.md`
9. this task package itself, only to append Worker output and verification notes after completion

## Forbidden Scope

This task must not:

1. edit any source code, source-tree config, test, runtime, hardware, or training file
2. edit `docs/00_project_snapshot.md`, `docs/01_legacy_audit.md`, `docs/02_experiment_plan.md`, `docs/03_hil_p4_boundary_audit.md`, `docs/04_task_board.md`, `docs/05_decision_log.md`, `docs/06_repo_noise_governance.md`, `docs/07_handoff.md`, or `docs/08_risks_and_open_questions.md`
3. run any new benchmark, trace export, intervention, retraining, `.tflite` runtime, real-board, cleanup, or statcalib integration task
4. create or modify any `runs/` or `artifacts/` directory
5. overwrite or rewrite historical `runs/` / `artifacts/` evidence
6. upgrade any wording into causal proof, mechanism closure, expanded benchmark evidence, `.tflite` validation, or real-board validation
7. edit theory-only or paper-idea branch materials such as `docs/reference/延申理论.md`, `docs/reference/延伸改进思路.md`, or files under `docs/follow-up_plan/`

## Required Inputs

Read at minimum:

- `docs/04_task_board.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/evidence_packs/mechanism_ablation/multi_seed_trace_generalization_probe.md`
- `docs/evidence_packs/mechanism_ablation/multi_seed_i1_intervention_probe.md`
- `docs/evidence_packs/mechanism_ablation/post_t55_mechanism_claim_reframing_gate.md`
- `docs/paper_materials/paper_ablation_result_pack.md`
- `docs/paper_materials/paper_claim_evidence_ledger.md`
- `docs/reality_recovery/04_figure_and_result_ledger.md`
- `docs/reality_recovery/05_paper_claim_risk_table.md`
- `docs/review/T54_review.md`
- `docs/review/T55_review.md`
- `docs/review/T56_review.md`
- `docs/review/T57_review.md`
- `runs/T54_multi_seed_trace_phase_a_20260522/cross_seed_comparison.csv`
- `runs/T54_multi_seed_trace_phase_a_20260522/delta_b_amplitude_by_seed.csv`
- `runs/T54_multi_seed_trace_phase_a_20260522/mechanism_summary.csv`
- `runs/T55_multi_seed_i1_probe_20260523/analysis/intervention_comparison.csv`
- `runs/T55_multi_seed_i1_probe_20260523/analysis/intervention_summary.csv`
- `runs/T55_multi_seed_i1_probe_20260523/analysis/intervention_summary.json`

## Fixed Evidence Boundary

This task is locked to the following evidence boundary:

1. data sources:
   - only existing `T54`, `T55`, and `T56` evidence
2. seeds:
   - only the locked 6-seed pack already present in `T54/T55`
3. scenarios:
   - only the frozen four scenarios already present in `T54/T55`
4. repeats:
   - no new repeats; reuse existing summarized evidence only
5. evidence type:
   - figure-pack assembly and ledger update
   - not new execution evidence

## Required Output Artifacts

Inside `docs/figure_assets/T58_fr6_multi_seed_mechanism_intervention/`, produce at minimum:

1. one final figure in vector or raster form:
   - `fr6_multi_seed_mechanism_intervention.svg` or `fr6_multi_seed_mechanism_intervention.png`
2. one companion export in the other format if practical:
   - `...png` or `...svg`
3. `figure_data.csv`
4. `figure_manifest.json`
5. `caption.md`
6. any helper script(s) used to build the figure, but only inside this task-scoped asset directory

## Figure Content Requirement

The figure must remain bounded and descriptive. At minimum it should capture both:

1. the `T54` cross-seed mechanism picture:
   - seed category or instability regime
   - Full vs Gated v5 gap
2. the `T55` intervention picture:
   - per-seed I1 effect relative to the Gated v5 baseline

A safe default is a two-panel figure:

- Panel A: seed-wise baseline gap with quiet / classic / universal labeling
- Panel B: seed-wise I1 intervention delta showing harmful / mixed / helpful outcomes

The figure must not imply that the intervention proves causality.

## Expected Report Output

Create `docs/evidence_packs/mechanism_ablation/fr6_multi_seed_mechanism_intervention_figure_pack.md` with:

1. exact input-source matrix for every plotted quantity
2. figure specification and panel definition
3. figure-asset directory contents
4. caption text in paper-ready but bounded form
5. FR6 classification:
   - ready
   - partial
   - missing
6. explicit non-claims and residual limitations
7. whether `docs/paper_materials/paper_ablation_result_pack.md` and `docs/paper_materials/paper_claim_evidence_ledger.md` can now mark FR6 as ready

Update `docs/paper_materials/paper_ablation_result_pack.md` and `docs/paper_materials/paper_claim_evidence_ledger.md` only if the FR6 status honestly changes.

Update `docs/reality_recovery/04_figure_and_result_ledger.md` and `docs/reality_recovery/05_paper_claim_risk_table.md` only if FR6 status or related paper-risk wording actually changes.

Create `docs/review/T58_review.md` with:

1. scope and boundary confirmation
2. confirmation that no new execution or source-tree edits occurred
3. figure-data provenance check
4. wording-boundary check against `T56`
5. recommended next bounded task, if any

Create `docs/for_human/T58_explanation.md` with a short human-facing summary.

Create `docs/worker_summary/T58_worker_summary.md` with a concise worker-facing summary of changes, verification, and residual risk.

## Verification

Required verification:

1. confirm only allowed docs changed, plus the single task-scoped figure asset directory
2. confirm no source, source-tree config, test, runtime, hardware, training, `runs/`, or `artifacts/` path was modified
3. confirm every plotted value is traceable to existing `T54/T55` CSV/JSON evidence
4. confirm the figure and caption do not claim causal proof, mechanism closure, expanded benchmark evidence, `.tflite` validation, or real-board validation
5. confirm the paper-facing docs preserve the `T56` hedge boundary and do not revive the retired "high committed-b is harmful" story
6. confirm no theory-only branch materials were touched

## Review No-Go Triggers

Review should be treated as `BLOCK` if any of the following happen:

1. the worker runs any new benchmark, trace export, intervention, retraining, `.tflite`, or hardware task
2. the worker edits source-tree code/config instead of staying inside docs and the task-scoped figure asset directory
3. the worker creates or modifies any `runs/` or `artifacts/` path
4. the worker upgrades FR6 into causal proof or mechanism closure
5. the worker uses theory-only planning materials as if they were current experiment evidence
6. the worker changes governance docs or `docs/02_experiment_plan.md`

## Captain Notes

`T57` is complete and accepted as `PASS`.

The next smallest honest paper-material gap is `FR6`, not another FR7 pass and not `FR8` yet.

This task stays on the mainline experiment-evidence lane. It must not interfere with the user's separate theory branch.

## Worker Output

### What changed

1. Created `docs/figure_assets/T58_fr6_multi_seed_mechanism_intervention/` with:
   - `build_figure.py`
   - `fr6_multi_seed_mechanism_intervention.svg`
   - `fr6_multi_seed_mechanism_intervention.png`
   - `figure_data.csv`
   - `figure_manifest.json`
   - `caption.md`
2. Added `docs/evidence_packs/mechanism_ablation/fr6_multi_seed_mechanism_intervention_figure_pack.md` to document:
   - exact input-source matrix
   - aggregation rules
   - figure-pack contents
   - FR6 classification and non-claims
3. Updated:
   - `docs/paper_materials/paper_ablation_result_pack.md`
   - `docs/paper_materials/paper_claim_evidence_ledger.md`
   - `docs/reality_recovery/04_figure_and_result_ledger.md`
   - `docs/reality_recovery/05_paper_claim_risk_table.md`
   so that `FR6` is now tracked as `ready` in the bounded figure-pack sense
4. Added:
   - `docs/review/T58_review.md`
   - `docs/for_human/T58_explanation.md`
   - `docs/worker_summary/T58_worker_summary.md`

### Verification Notes

1. Confirmed only allowed docs changed, plus the single task-scoped figure asset directory.
2. Confirmed no source, source-tree config, test, runtime, hardware, training, `runs/`, or `artifacts/` path was modified.
3. Confirmed every plotted value is traceable to existing `T54/T55` CSV evidence and recorded in `figure_manifest.json`.
4. Confirmed the figure and caption do not claim causal proof, mechanism closure, expanded benchmark evidence, `.tflite` validation, or real-board validation.
5. Confirmed the updated paper-facing docs preserve the `T56` hedge boundary and do not revive the retired `high committed-b is harmful` story.
6. Confirmed no theory-only branch materials were touched.
