# T27: Teacher diagnostics path audit and mechanism-evidence repair plan

## Status

- Created by Captain on `2026-05-11`
- Current phase: `Phase 2: Controlled Development`
- Decision state: `Go`
- Task type: read-only mechanism-evidence audit

## Why This Task Exists

`T25` accepted `T24` as completed frozen-set formal software revalidation, but only within the `mock-backed software HIL` boundary.

Two mechanism-evidence gaps remain deferred:

1. `teacher_scalar_diagnostics.csv` is header-only and teacher diagnostics remain all-zero from `T15` through `T24` (`R10`).
2. `correction_saturation_rate_mean` is structurally `0.0` across all 20 T24 scenario/mode rows (`R20`).

This task exists to audit the existing code/data path and produce a minimal repair plan. It does not repair the path, rerun benchmark, add a new baseline, or expand the frozen set.

## Goal

Determine, using read-only inspection, why teacher diagnostics are absent/all-zero in the existing T15/T24 evidence path, and whether `correction_saturation_rate_mean` shares the same missing/dead metric path or is an independent issue.

The output should classify the root cause as precisely as current evidence allows:

- data not generated
- data generated but not aggregated
- data aggregated but not written
- mode/scenario not applicable
- genuine zero under the current parameter regime
- source change required to determine the answer

## Allowed Files

Worker may read any repository file needed for this audit.

Worker may modify only:

- `docs/review/T27_teacher_diagnostics_path_audit.md`
- `docs/for_human/T27_explanation.md`
- `docs/tasks/Phase2/T27_teacher_diagnostics_path_audit.md`

Captain may later update governance files after review:

- `docs/04_task_board.md`
- `docs/05_decision_log.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`

## Required Inputs

Read at minimum:

- `docs/review/T25_p4_formal_evidence_gate_review.md`
- `docs/review/T24_review.md`
- `docs/review/T15_frozen_smoke_review.md`
- `docs/P4_benchmark_formal_protocol.md`
- `docs/tasks/Phase2/T24_p4_formal_software_revalidation.md`
- `docs/tasks/Phase2/T25_p4_formal_evidence_gate_review.md`
- `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/comparison.csv`
- `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/teacher_scalar_diagnostics.csv`
- `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/report.md`

Suggested source-path search:

- `rg "teacher_scalar|teacher_gate|teacher_contribution|diagnostic|diagnostics|correction_saturation|saturation_rate" cnn_fpga physics benchmark docs`
- Inspect the relevant runner / runtime files found by that search, especially benchmark aggregation and report-writing paths.

## Forbidden Scope

Do not:

- run benchmark, training, `.tflite`, hardware, cleanup, or any command that creates a new run directory
- modify source code, configs, benchmark protocol, baseline set, scenario set, or ParamMapper semantics
- add `statcalib`, soft-information comparator, new drift family, CI-driven stopping, `.tflite` runtime, or real-board scope
- rewrite `runs/` or `artifacts/` content as a new fact source
- mark teacher diagnostics or correction saturation as fixed unless the existing evidence directly proves it
- claim T24 is paper-grade expanded benchmark, true `.tflite` runtime, or real-board validation

## Expected Output

Create `docs/review/T27_teacher_diagnostics_path_audit.md` with:

1. Verdict:
   - `PASS`: path cause is identified and a bounded next task is recommended.
   - `PASS_WITH_WARNINGS`: path cause is partly identified, with deferred uncertainties explicitly mapped to risks.
   - `BLOCK`: current evidence is internally inconsistent or the audit cannot be trusted without a blocking correction.
2. Exact files and functions inspected.
3. Teacher diagnostics data-flow map:
   - generation source
   - aggregation path
   - CSV/report writer
   - where the path becomes empty/all-zero
4. `correction_saturation_rate_mean` path note:
   - shared issue with teacher diagnostics, independent dead path, genuine zero, or not enough evidence
5. Risk mapping:
   - whether R10 remains open, can be narrowed, or needs a new follow-up task
   - whether R20 remains open, can be narrowed, or needs a new follow-up task
6. Minimal next task recommendation, without executing it.

Create `docs/for_human/T27_explanation.md` with a concise human-readable summary.

## Verification

Before handoff, Worker must report:

- no new benchmark/training/runtime/hardware/cleanup command was run
- no source/config/run/artifact file was modified
- exact inspected files and any commands used for read-only search
- whether `docs/review/T27_teacher_diagnostics_path_audit.md` and `docs/for_human/T27_explanation.md` were created

## Captain Notes

T26 remains a pending roadmap item, but it is not the current unique task. T25 recommended T27 first because R10 is the longest deferred mechanism-evidence chain and directly affects interpretation of `hybrid_residual_b`.

## Worker Output

- Created `docs/review/T27_teacher_diagnostics_path_audit.md`.
- Created `docs/for_human/T27_explanation.md`.
- Audited the existing teacher-diagnostics path as read-only and concluded:
  - non-hybrid modes are `not applicable` for teacher diagnostics
  - current T24 `hybrid_residual_b` path does reach artifact inference, but teacher diagnostics are not generated because the frozen strong-baseline config uses broadcast teacher features while `tiny_cnn.py::explain_from_loaded_artifact()` only emits teacher diagnostics when `scalar_feature_dim > 0`
  - downstream aggregation/CSV writing preserves emptiness for `per_scalar`, but also coerces several missing values to `0.0`, which hides the distinction between `not generated` and `true zero`
  - `correction_saturation_rate_mean` is generated from an independent fast-loop saturation counter path and is not on the same dead metric path as teacher diagnostics

## Verification Record

- No new benchmark, training, runtime, hardware, or cleanup command was run.
- No source, config, run, or artifact file was modified.
- Read-only search/inspection commands used:
  - `rg "teacher_scalar|teacher_gate|teacher_contribution|diagnostic|diagnostics|correction_saturation|saturation_rate" cnn_fpga physics benchmark docs`
  - `rg -n "explain_from_loaded_artifact|per_scalar_contribution|per_scalar_gate_effect|teacher_contribution|gate_mean|gate_std|scalar_features_raw" cnn_fpga/model/tiny_cnn.py`
  - `Get-ChildItem runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743 -Recurse -Filter hil_summary.json`
  - `Get-Content` on the required docs, config, source files, `comparison.csv`, `teacher_scalar_diagnostics.csv`, and representative `hil_summary.json`
- Created files confirmed:
  - `docs/review/T27_teacher_diagnostics_path_audit.md`
  - `docs/for_human/T27_explanation.md`

## Captain Acceptance

- Verdict accepted: `PASS_WITH_WARNINGS`
- Blocking issues: none
- Warning classification:
  - R10 root cause found but not repaired: `deferred`
  - downstream missing-vs-zero `0.0` coercion: `deferred`, tracked as R21
  - R20 independent fast-loop saturation path: `accepted` as narrowed, but R20 remains open for future stress/edge evidence
- Next unique task: `T28: Teacher diagnostics missing-vs-zero semantics repair and minimal smoke`
