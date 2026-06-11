# T57: FR7 feature/teacher ablation re-execution under locked T24 protocol

## Status

- Proposed by Captain on `2026-05-24`
- Current phase: `Phase 2: Controlled Development`
- Decision state: `Go`
- Task type: bounded execution task on the frozen T24 benchmark lane

## Why This Task Exists

`T47` has now closed cleanly as a paper-material ledger task, but it also made the gap explicit: `FR7` is still the largest missing piece in the ablation pack.

This task exists to re-execute `FR7` under the locked `T24` protocol, so the project can honestly answer whether the feature/teacher ablation table is still `missing`, has become `partial`, or can now be treated as `ready` without widening the benchmark boundary.

The task is not a new benchmark family. It is a bounded re-execution of the already frozen feature-ablation lane.

## Goal

Re-run the locked feature/teacher ablation matrix and produce a paper-grade but still bounded FR7 evidence pack that answers:

1. whether the ablation table can be regenerated under the locked `T24` protocol
2. whether the paper can now truthfully cite feature/teacher ablation evidence without historical-only caveats
3. whether any remaining limitations must stay explicit in the paper and evidence ledgers

## Allowed Files

Worker may modify only:

- `docs/tasks/Phase2/T57_fr7_feature_teacher_ablation_reexecution.md`
- `docs/evidence_packs/mechanism_ablation/fr7_feature_teacher_ablation_reexecution.md`
- `docs/paper_materials/paper_ablation_result_pack.md`
- `docs/paper_materials/paper_claim_evidence_ledger.md`
- `docs/reality_recovery/04_figure_and_result_ledger.md`
- `docs/reality_recovery/05_paper_claim_risk_table.md`
- `docs/review/T57_review.md`
- `docs/for_human/T57_explanation.md`
- `docs/worker_summary/T57_worker_summary.md`

Worker may create:

- one T57-scoped run root under `runs/p4_benchmark/T57_fr7_feature_teacher_ablation_*`
- derived benchmark outputs and summaries only inside that T57-scoped run root

## Docs To Update

This task should update only:

1. `docs/evidence_packs/mechanism_ablation/fr7_feature_teacher_ablation_reexecution.md`
2. `docs/paper_materials/paper_ablation_result_pack.md`
3. `docs/paper_materials/paper_claim_evidence_ledger.md`
4. `docs/reality_recovery/04_figure_and_result_ledger.md`
5. `docs/reality_recovery/05_paper_claim_risk_table.md`
6. `docs/review/T57_review.md`
7. `docs/for_human/T57_explanation.md`
8. `docs/worker_summary/T57_worker_summary.md`
9. this task package itself, only to append Worker output and verification notes after completion

## Forbidden Scope

This task must not:

1. edit any source code, source-tree config, test, runtime, hardware, or training file
2. edit `docs/00_project_snapshot.md`, `docs/01_legacy_audit.md`, `docs/02_experiment_plan.md`, `docs/03_hil_p4_boundary_audit.md`, `docs/04_task_board.md`, `docs/05_decision_log.md`, `docs/06_repo_noise_governance.md`, `docs/07_handoff.md`, or `docs/08_risks_and_open_questions.md`
3. retrain any model, create a new learned variant, or change the model architecture / feature semantics
4. expand beyond the locked `T24` frozen scenarios, mode set, seed semantics, or repeat count
5. add any new baseline family, new comparator lane, new intervention lane, or new drift family
6. run `.tflite` runtime work, real-board work, cleanup, benchmark expansion, or statcalib integration
7. overwrite or rewrite historical `runs/` or `artifacts/` paths
8. upgrade results into causal proof, full mechanism closure, paper-grade benchmark expansion, `.tflite` validation, or real-board validation

## Required Inputs

Read at minimum:

- `docs/04_task_board.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/protocols/benchmark/P4_benchmark_formal_protocol.md`
- `docs/reality_recovery/00_freeze_snapshot.md`
- `docs/reality_recovery/04_figure_and_result_ledger.md`
- `docs/reality_recovery/05_paper_claim_risk_table.md`
- `docs/paper_materials/paper_ablation_result_pack.md`
- `docs/paper_materials/paper_claim_evidence_ledger.md`
- `docs/review/T24_review.md`
- `docs/review/T25_p4_formal_evidence_gate_review.md`
- `docs/review/T47_review.md`
- `docs/review/T56_review.md`
- `cnn_fpga/config/p4_hybrid_vs_ukf_ablation_features.yaml`
- `cnn_fpga/benchmark/run_p4_hybrid_vs_ukf_ablation.py`
- `cnn_fpga/benchmark/summarize_p4_features_ablation.py`

## Fixed Execution Boundary

This task is locked to the following execution boundary:

1. protocol:
   - `T24` frozen-set formal software revalidation boundary
2. scenarios:
   - `static_bias_theta`
   - `linear_ramp`
   - `step_sigma_theta`
   - `periodic_drift`
3. modes:
   - `ukf`
   - `hybrid_full`
   - `hybrid_no_hist_deltas`
   - `hybrid_no_teacher_prediction`
   - `hybrid_no_teacher_params`
   - `hybrid_no_teacher_deltas`
4. repeats:
   - `2`
5. evidence type:
   - FR7 re-execution evidence, not causal proof and not benchmark expansion

## Model And Config Reuse Rules

1. Reuse the existing feature-ablation config `cnn_fpga/config/p4_hybrid_vs_ukf_ablation_features.yaml`; do not edit it.
2. Reuse existing model artifacts if they are already present; do not retrain or create any new learned variant.
3. Use the locked T24 protocol semantics and paired-seed ordering as implemented by the runner; do not change seed policy or scenario order.
4. Keep every generated benchmark artifact, summary, helper file, and report inside one T57-scoped run root.
5. If a required artifact is missing and cannot be reused, stop and report the blocker instead of opening a training lane.

## Expected Run Matrix

The benchmark matrix is:

- `4 scenarios x 6 modes x repeats=2 = 48 repeat-runs`

## Required Output Artifacts

Inside the T57-scoped run root, produce at minimum:

1. `summary.json`
2. `comparison.csv`
3. any per-repeat `hil_summary.json` / `repeat_status.json` files produced by the runner
4. a reuse / provenance manifest that records which artifacts were reused and which were regenerated
5. a summary artifact produced by `summarize_p4_features_ablation.py` or an equivalent bounded summarizer, kept inside the same run root

## Expected Report Output

Create `docs/evidence_packs/mechanism_ablation/fr7_feature_teacher_ablation_reexecution.md` with:

1. exact command list and run-root structure
2. provenance matrix for reused artifacts versus any regenerated artifacts
3. protocol conformance table against the locked `T24` boundary
4. scenario-wise and mode-wise outcome summary
5. FR7 classification:
   - ready
   - partial
   - missing
6. explicit non-claims and residual limitations
7. whether `docs/paper_materials/paper_ablation_result_pack.md` and `docs/paper_materials/paper_claim_evidence_ledger.md` can now be updated without overstating closure

Update `docs/paper_materials/paper_ablation_result_pack.md` and `docs/paper_materials/paper_claim_evidence_ledger.md` so the paper-facing evidence pack reflects the FR7 result honestly and keeps all hedge wording intact.

Update `docs/reality_recovery/04_figure_and_result_ledger.md` and `docs/reality_recovery/05_paper_claim_risk_table.md` only if the FR7 status or risk annotations actually change.

Create `docs/review/T57_review.md` with:

1. scope and boundary confirmation
2. whether the worker stayed inside the frozen scenarios, modes, and repeat count
3. whether any reuse / regeneration decisions were honest and justified
4. whether the final wording stays bounded and non-causal
5. recommended next bounded task, if any

Create `docs/for_human/T57_explanation.md` with a short human-facing summary.

Create `docs/worker_summary/T57_worker_summary.md` with a concise worker-facing summary of changes, verification, and residual risk.

## Verification

Required verification:

1. confirm only allowed docs changed, plus the single T57-scoped run root
2. confirm no source, source-tree config, test, runtime, hardware, or training file was modified
3. confirm no historical `runs/` or `artifacts/` path was overwritten
4. confirm all execution stayed inside the locked `T24` frozen four scenarios, six modes, and repeats=`2`
5. confirm the report does not claim causal proof, new benchmark expansion, `.tflite` runtime validation, or real-board validation
6. confirm the paper-facing docs preserve the T56 hedge boundary and do not upgrade mechanism claims beyond evidence

## Review No-Go Triggers

Review should be treated as `BLOCK` if any of the following happen:

1. the worker retrains any model or creates a new learned variant
2. the worker changes source-tree code or config instead of reusing the frozen feature-ablation lane
3. the worker expands beyond the four frozen scenarios, six modes, or repeat count `2`
4. the worker changes seed policy or scenario order
5. the worker runs any intervention variant or comparator lane outside FR7
6. the worker writes new outputs outside the single T57-scoped run root
7. the report upgrades FR7 evidence into causal proof or paper-grade benchmark expansion

## Captain Notes

`T47` is complete and accepted as `PASS`.

`T57` is the smallest remaining execution lane that can close FR7 honestly without reopening `.tflite`, real-board, cleanup, or benchmark-expansion scope.

If `T57` still leaves FR7 incomplete, the paper must keep the gap explicit rather than silently strengthening the claim.

## Worker Output

Worker execution completed on `2026-05-26`.

Summary:

1. Re-executed the full FR7 matrix inside `runs/p4_benchmark/T57_fr7_feature_teacher_ablation_20260524_000000`.
2. Completed both repeat chunks and regenerated the full `4 scenarios x 6 modes x 2 repeats = 48` bounded run set.
3. Generated `summary.json`, `comparison.csv`, `delta.csv`, `report.md`, `summary_pack/*`, and `provenance_manifest.json` inside the single allowed T57 run root.
4. Created `docs/evidence_packs/mechanism_ablation/fr7_feature_teacher_ablation_reexecution.md` and updated the allowed paper-facing ledgers so FR7 is no longer treated as missing.
5. Kept all wording bounded: FR7 is now a ready frozen-set result table, but not causal proof and not expanded benchmark evidence.

Key bounded result:

- `hybrid_no_teacher_params` is the best mode in all 4 scenarios under this reused ablation lane.
- `hybrid_no_hist_deltas` degrades against `hybrid_full` in all 4 scenarios.
- `hybrid_no_teacher_prediction` also degrades against `hybrid_full` in all 4 scenarios.
- `hybrid_no_teacher_deltas` is near-neutral/mixed overall.

## Verification Notes

Verification performed:

1. Confirmed `summary.json` reports `comparison_row_count=24`, `missing_runs_count=0`, and `bad_rows_count=0`; every comparison row has `completed_repeats=2` and `coverage=1.0`.
2. Confirmed execution stayed inside the locked 4 scenarios, 6 modes, paired-seed policy, and `repeats=2`.
3. Confirmed no source-tree code/config/test paths under `cnn_fpga`, `physics`, or `tests` were modified by this task.
4. Confirmed no historical `runs/` or `artifacts/` paths outside the single T57 run root were modified by this task.
5. Confirmed updated docs keep explicit non-claim language around causal proof, expanded benchmark evidence, `.tflite`, and real-board validation.

Residual risk:

- FR7 closes the missing-result-table gap, but it does not close the T56 mechanism hedge.
- The bounded result weakens the simple historical architectural-attribution story, so paper wording must stay descriptive and non-causal.
