# T55: Phase B multi-seed I1 residual-clip intervention probe

## Status

- Proposed by Captain on `2026-05-23`
- Current phase: `Phase 2: Controlled Development`
- Decision state: `Go`
- Task type: bounded execution task on an already validated path

## Why This Task Exists

`T54` passed and closed the Phase A question:

- the committed-`b` instability pattern is no longer a single-seed-only observation
- the pattern is broadly repeated with qualifications across the locked 6-seed pack
- the mechanism story is still not closed, because `C4` remains `partial`

The next smallest unresolved gap is therefore not paper packaging. It is whether a pure lower-clip intervention helps, harms, or does not materially change outcomes on the same bounded seed/scenario pack.

This task is the smallest Phase B follow-up after `T54`. It must stay on the same mock-backed P4 wrapper over software HIL lane and must not reopen training, `.tflite`, hardware, cleanup, or benchmark-expansion scope.

## Goal

Run the smallest believable intervention probe that can answer:

1. whether lowering Gated v5 `residual_clip_b` from `0.12` to `0.06` changes the observed mechanism signals on the same 6-seed pack
2. whether that intervention is helpful, harmful, mixed, or no-clear-effect at the bounded evidence level
3. whether the project has enough mechanism evidence to justify any later paper-material lane

## Allowed Files

Worker may modify only:

- `docs/tasks/Phase2/T55_multi_seed_i1_residual_clip_intervention_probe.md`
- `docs/multi_seed_i1_intervention_probe.md`
- `docs/review/T55_review.md`
- `docs/for_human/T55_explanation.md`

Worker may create:

- one T55-scoped run root under `runs/`, for example `runs/T55_multi_seed_i1_probe_*`
- generated benchmark YAML config(s) only inside that T55-scoped run root
- helper analysis script(s) only inside that T55-scoped run root
- derived CSV/JSON summaries only inside that T55-scoped run root

## Docs To Update

This task should update only:

1. `docs/multi_seed_i1_intervention_probe.md`
2. `docs/review/T55_review.md`
3. `docs/for_human/T55_explanation.md`
4. this task package itself, only to append Worker output and verification notes after completion

## Forbidden Scope

This task must not:

1. edit any source code, benchmark code, source-tree config, test, runtime, hardware, or training file
2. edit `docs/00_project_snapshot.md`, `docs/01_legacy_audit.md`, `docs/03_hil_p4_boundary_audit.md`, `docs/04_task_board.md`, `docs/05_decision_log.md`, `docs/06_repo_noise_governance.md`, `docs/07_handoff.md`, `docs/08_risks_and_open_questions.md`, or `docs/02_experiment_plan.md`
3. retrain any model, rebuild any dataset, create any new learned variant, or start any new teacher-representation long run outside this task's bounded execution
4. use any seed outside the locked 6-seed pack
5. use any scenario outside the frozen four scenarios
6. add any new baseline family, any comparator lane, or any second intervention variant
7. treat existing `v6` / `v7` / `v8` / `v9` configs as proxies for the pure I1 intervention
8. run `.tflite` runtime work, real-board work, cleanup, benchmark expansion, `statcalib` integration, or comparator expansion
9. overwrite or rewrite historical `runs/` or `artifacts/`
10. upgrade results into causal proof, full mechanism closure, paper-grade benchmark evidence, `.tflite` validation, or real-board validation

## Required Inputs

Read at minimum:

- `docs/04_task_board.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/seed_mechanism_multi_seed_plan.md`
- `docs/review/T46_review.md`
- `docs/multi_seed_trace_generalization_probe.md`
- `docs/review/T54_review.md`
- `docs/seed20260429_trace_export_diagnosis.md`
- `docs/paper_claim_evidence_ledger.md`
- `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py`
- `cnn_fpga/config/p4_hybrid_vs_ukf_ablation_features.yaml`
- `cnn_fpga/config/experiment_runtime_b_residual.yaml`
- `cnn_fpga/config/experiment_runtime_b_residual_norm_gated_teacher_v5.yaml`

## Fixed Execution Boundary

This task is locked to the following execution boundary:

1. seed pack:
   - `20260425`
   - `20260427`
   - `20260428`
   - `20260429`
   - `20260430`
   - `20260510`
2. scenarios:
   - `static_bias_theta`
   - `linear_ramp`
   - `step_sigma_theta`
   - `periodic_drift`
3. repeats:
   - `2`
4. evidence type:
   - bounded intervention evidence, not causal proof
5. new execution ceiling:
   - exactly one intervention variant

## Model And Config Reuse Rules

1. Reuse existing model artifacts from the same seed/model lane documented by `T54`; do not retrain.
2. Reuse `T54` baseline references for `Full` and `Gated v5`; do not rerun baseline modes unless blocked by a missing comparable field that must be regenerated for a fair same-format comparison.
3. Generate benchmark config(s) only inside the T55 run root; do not edit source-tree config files.
4. Pure I1 means changing only `slow_loop.hybrid_residual_b.residual_clip_b` from `0.12` to `0.06`.
5. Do not change `residual_scale_b`, teacher features, gate bias, architecture, model artifact path, baseline set, or scenario set.
6. Keep benchmark output roots inside the same T55 run root.

## Required Output Artifacts

Inside the T55-scoped run root, produce at minimum:

1. `seed_model_reuse_manifest.json`
2. generated benchmark config YAML file(s)
3. intervention benchmark outputs
4. intervention trace export(s) in the same schema family as T38/T54
5. `intervention_comparison.csv`
6. `intervention_trace_summary.csv`
7. `intervention_summary.json`

## Expected Report Output

Create `docs/multi_seed_i1_intervention_probe.md` with:

1. exact command list and run-root structure
2. seed/model reuse matrix
3. config delta table for baseline Gated v5 versus I1 clip-0.06
4. per-seed/per-scenario outcome comparison versus reused `T54` baselines
5. per-seed trace-effect summary covering delta-`b` amplitude, clip ratio, committed-`b`, and LER
6. intervention verdict by seed:
   - helpful
   - harmful
   - mixed
   - no-clear-effect
7. a recommendation on whether `T47` can proceed or whether more mechanism work is still required
8. explicit non-claims

Create `docs/review/T55_review.md` with:

1. scope and boundary confirmation
2. whether the worker stayed inside the same 6 seeds, frozen four scenarios, and repeats=`2`
3. whether the intervention remained a pure I1 lower-clip change rather than a proxy variant
4. whether source-tree code/config remained untouched
5. whether the wording stays at bounded intervention evidence rather than causal proof
6. recommended next bounded task, if any

Create `docs/for_human/T55_explanation.md` with a short human-facing summary.

## Verification

Required verification:

1. confirm only allowed docs changed, plus the single T55-scoped run root
2. confirm no source, source-tree config, test, runtime, hardware, or training file was modified
3. confirm no historical `runs/` or `artifacts/` path was overwritten
4. confirm only one intervention variant was executed
5. confirm all new execution stayed inside the fixed 6 seeds, the frozen four scenarios, and repeats=`2`
6. confirm the report does not claim causal proof, full mechanism closure, `.tflite` runtime validation, or real-board validation

## Review No-Go Triggers

Review should be treated as `BLOCK` if any of the following happen:

1. the worker retrains any model or creates a new learned variant
2. the worker uses `v6` / `v7` / `v8` / `v9` as a proxy for pure I1
3. the worker edits any source code or source-tree config
4. the worker runs more than one new intervention variant
5. the worker writes new outputs outside the single T55-scoped run root
6. the report upgrades bounded intervention evidence into causal proof or paper-grade benchmark evidence

## Captain Notes

This task is intentionally before the old `T47` paper-material lane.

`T54` answered the generalization question. `T55` is the smallest execution lane that can answer the intervention question without reopening wider scope.

If `T55` shows no useful intervention signal, then the project should not rush into `T47` as if the mechanism story were already closed.

## Worker Output

- Pending.

## Verification Record

- Pending.
