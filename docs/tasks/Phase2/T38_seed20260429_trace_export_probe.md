# T38: seed=20260429 single-seed trace-export probe, bounded unchanged-semantics rerun

## Status

- Created by Captain on `2026-05-13`
- Current phase: `Phase 2: Controlled Development`
- Decision state: `Go`
- Task type: bounded instrumentation / single-seed diagnostic rerun

## Why This Task Exists

`T36` passed review and narrowed the `seed=20260429` Gated-v5 shrinkage to a residual-amplitude / teacher-delta regime instability hypothesis. However, `T36` also found that existing artifacts only contain scenario summaries and final snapshots. They do not expose per-window or per-commit traces for `teacher_b`, predicted `delta_b`, or committed `b`.

This task fills that specific evidence gap without widening the benchmark or changing model semantics.

## Goal

Add and run the smallest trace-export probe needed to decide whether the `seed=20260429` behavior is driven by sign offset, magnitude overshoot chronology, teacher prediction instability, CNN residual output, or the combined committed `teacher_b + delta_b`.

The output must be a diagnostic trace, not a new formal benchmark result.

## Allowed Files

Worker may modify:

- `docs/tasks/Phase2/T38_seed20260429_trace_export_probe.md`
- `docs/seed20260429_trace_export_diagnosis.md`
- `docs/review/T38_review.md`
- `docs/for_human/T38_explanation.md`
- `cnn_fpga/runtime/slow_loop_runtime.py`
- `cnn_fpga/benchmark/run_hil_suite.py`
- `cnn_fpga/benchmark/run_p4_teacher_representation_paired.py`
- `cnn_fpga/benchmark/analyze_seed20260429_trace.py`
- one T38-scoped config file if required: `cnn_fpga/config/p4_teacher_repr_seed20260429_trace.yaml`

If the trace can be exported by existing metadata hooks without editing every allowed code path, prefer the smaller edit.

Worker may create one T38-scoped run directory under `runs/` only for this probe.

## Required Inputs

Read at minimum:

- `docs/seed20260429_failure_diagnosis.md`
- `docs/review/T36_review.md`
- `docs/tasks/Phase2/T36_seed20260429_failure_mechanism_diagnosis.md`
- `docs/04_task_board.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `cnn_fpga/runtime/slow_loop_runtime.py`
- `cnn_fpga/benchmark/run_hil_suite.py`
- `cnn_fpga/benchmark/run_p4_teacher_representation_paired.py`
- `cnn_fpga/config/experiment_runtime_b_residual.yaml`
- `cnn_fpga/config/experiment_runtime_b_residual_norm_gated_teacher_v5.yaml`

Use the same existing Gated-v5 / Full artifacts and seed context referenced by T36 as the baseline for interpretation.

## Forbidden Scope

Do not:

- modify `docs/02_experiment_plan.md`
- train models or add a new teacher-representation branch
- add statcalib, soft-information comparator, new drift family, new baseline, or CI-driven stopping
- change existing formal benchmark protocol, frozen ranked set, scenario set, seed policy, or result boundary
- touch `.tflite`, true TFLite runtime, real-board backend, hardware commands, or cleanup
- rewrite historical `runs/` or `artifacts/`
- present the single-seed trace probe as paper-grade benchmark evidence

## Required Trace Fields

For each slow update / window where available, export:

- scenario, mode, repeat, seed, window index or epoch id
- `teacher_b_q`, `teacher_b_p`
- predicted `delta_b_q`, `delta_b_p`
- committed `b_q`, `b_p`
- active bank / staged bank or equivalent commit identifier
- window-level LER or nearest available per-window outcome proxy
- correction-utilization / overflow / saturation proxy if already available from the same path

If any field is unavailable without broad refactor, document the gap explicitly instead of inventing it.

## Expected Output

Create `docs/seed20260429_trace_export_diagnosis.md` with:

1. Exact command and run directory.
2. Trace schema and field availability table.
3. Scenario-level and repeat-level findings for `Full` vs `Gated v5`.
4. Mechanism update matrix for:
   - sign offset
   - magnitude overshoot chronology
   - teacher prediction instability
   - CNN residual output instability
   - committed combined-b instability
5. Clear split between supported conclusions and remaining hypotheses.
6. Recommendation for the next bounded task, if any, but do not execute it.

## Verification

Required verification:

1. Run the bounded trace-export probe exactly once as specified by this task package.
2. Confirm the run creates only T38-scoped outputs and does not overwrite historical artifacts.
3. Run a compile or targeted smoke check for any changed Python files.
4. Confirm no files outside the allowed set were changed, except the single T38-scoped run directory.
5. Confirm the report does not claim a new formal benchmark, new model result, `.tflite` runtime, or real-board validation.

## Docs To Update

- `docs/tasks/Phase2/T38_seed20260429_trace_export_probe.md`
- `docs/seed20260429_trace_export_diagnosis.md`
- `docs/review/T38_review.md`
- `docs/for_human/T38_explanation.md`

## Reviewer Type

Adversarial review.

Focus areas:

- trace export did not alter benchmark semantics
- single-seed rerun stayed inside the T38 scope
- no new branch/baseline/scenario/protocol was silently added
- trace fields are real and not fabricated from final snapshots
- conclusions do not overclaim causality beyond one seed

## Captain Notes

This task is allowed to run a bounded diagnostic rerun because T36 explicitly proved existing artifacts cannot answer the per-window question. Keep the rerun narrow: `seed=20260429`, existing Full vs Gated v5 path, same four scenarios, no benchmark expansion.
