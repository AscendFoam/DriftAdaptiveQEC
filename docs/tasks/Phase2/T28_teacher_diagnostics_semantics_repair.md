# T28: Teacher diagnostics missing-vs-zero semantics repair and minimal smoke

## Status

- Created by Captain on `2026-05-11`
- Current phase: `Phase 2: Controlled Development`
- Decision state: `Go`
- Task type: bounded repair + minimal smoke

## Why This Task Exists

`T27` narrowed `R10` to a concrete path issue:

- current T24 `hybrid_residual_b` uses broadcast teacher features
- `tiny_cnn.py::explain_from_loaded_artifact()` only emits teacher scalar diagnostics when `scalar_feature_dim > 0`
- downstream aggregation / CSV writing coerces missing teacher diagnostics to `0.0`, masking `not generated` as if it were `true zero`

`T27` also narrowed `R20`:

- `correction_saturation_rate_mean` comes from an independent fast-loop saturation counter
- current T24 `0.0` is not caused by the teacher diagnostics dead path

This task repairs the teacher diagnostics observability/semantics issue only. It does not change the formal benchmark matrix or claim new mechanism evidence quality beyond the minimal smoke.

## Goal

Make teacher diagnostics outputs distinguishable between:

- `not generated`
- `not applicable`
- `true zero`

and document the current support boundary between broadcast teacher features and scalar-branch teacher diagnostics.

## Allowed Files

Worker may modify:

- `cnn_fpga/runtime/slow_loop_runtime.py`
- `cnn_fpga/benchmark/run_hil_suite.py`
- `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py`
- `docs/review/T28_teacher_diagnostics_semantics_repair.md`
- `docs/for_human/T28_explanation.md`
- `docs/tasks/Phase2/T28_teacher_diagnostics_semantics_repair.md`

If tests already exist and need minimal updates, Worker may modify:

- files under `tests/` that directly cover the changed paths

Worker may create one T28-specific run directory only if required for minimal smoke verification.

## Required Inputs

Read at minimum:

- `docs/review/T27_teacher_diagnostics_path_audit.md`
- `docs/for_human/T27_explanation.md`
- `docs/tasks/Phase2/T27_teacher_diagnostics_path_audit.md`
- `docs/P4_benchmark_formal_protocol.md`
- `cnn_fpga/config/p4_multiscenario_strong_baselines.yaml`
- `cnn_fpga/runtime/feature_builder.py`
- `cnn_fpga/runtime/slow_loop_runtime.py`
- `cnn_fpga/model/tiny_cnn.py`
- `cnn_fpga/benchmark/run_hil_suite.py`
- `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py`

## Forbidden Scope

Do not:

- modify `docs/02_experiment_plan.md`
- change ParamMapper mainline semantics
- change formal benchmark scenario set, baseline set, seed policy, or T24 historical result files
- rewrite any existing `runs/` or `artifacts/` outputs
- implement `statcalib`, soft-information comparator, new drift family, CI-driven stopping, true `.tflite` runtime, or real-board path
- claim teacher mechanism evidence is fully repaired unless minimal smoke directly proves the repaired output semantics
- start a long benchmark, training run, cleanup task, hardware call, or `.tflite` runtime task

## Expected Repair Shape

Prefer the smallest change that preserves current behavior while adding explicit semantics.

At minimum, the repaired output should make it clear when teacher diagnostics are absent because they were not generated. Acceptable approaches include:

- adding an explicit status/source field such as `teacher_diagnostics_status`
- preserving `null`/empty values instead of coercing missing diagnostics to `0.0`
- documenting that broadcast teacher features do not currently produce scalar-branch diagnostics
- ensuring per-scalar CSV remains empty only with an explicit upstream status in the main comparison/report output

Do not redesign the model, retrain, or switch the benchmark to scalar-branch teacher features unless that can be done as a tiny smoke-only config path within the allowed scope and without changing formal benchmark claims.

## Verification

Worker must perform bounded verification only:

1. Static check or focused unit test for missing-vs-zero serialization if available.
2. Minimal smoke that exercises:
   - one broadcast `hybrid_residual_b` path, or an equivalent minimal invocation that reaches the same summary/writer code
   - optional scalar-branch path only if already supported by existing config/code and cheap to run
3. Confirm the resulting report/CSV/summary distinguishes `not generated` from `true zero`.

The smoke must not be treated as a new formal benchmark.

## Expected Output

Create `docs/review/T28_teacher_diagnostics_semantics_repair.md` with:

- files changed
- exact repair semantics
- verification commands and outputs
- whether R10 can be narrowed further, remains open, or can be partly closed
- whether R21 remains open or is closed
- explicit statement that T24 historical evidence was not rewritten

Create `docs/for_human/T28_explanation.md` with a concise human-readable explanation.

Update this task package with Worker Output and Verification Record.

## Captain Notes

T26 remains pending but is not the current task. Statcalib or paper-claim work should wait until the teacher diagnostics output semantics are no longer ambiguous.
