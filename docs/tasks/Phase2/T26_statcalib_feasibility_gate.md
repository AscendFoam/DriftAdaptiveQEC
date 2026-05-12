# T26: Calibration/statcalib baseline feasibility gate and minimal design plan

## Status

- Created by Captain on `2026-05-12`
- Current phase: `Phase 2: Controlled Development`
- Decision state: `Go`
- Task type: read-only feasibility gate / design plan

## Why This Task Exists

`T24` completed a frozen-set formal software revalidation under the existing mock-backed software HIL boundary. `T25` accepted that result boundary, while `T27` and `T28` narrowed teacher diagnostics risks and repaired missing-vs-zero output semantics. `T29` then fixed the human-readable P4 markdown report header.

The next controlled step is not to implement a new comparator or launch another benchmark. The next step is to decide whether a calibration/statcalib baseline can be added cleanly, what prerequisites are missing, and what minimal design would avoid silently changing the existing formal benchmark semantics.

## Goal

Produce a feasibility gate and minimal design plan for a future calibration/statcalib baseline/comparator, without implementing it and without running any benchmark.

## Allowed Files

Worker may modify:

- `docs/tasks/Phase2/T26_statcalib_feasibility_gate.md`
- `docs/review/T26_statcalib_feasibility_gate.md`
- `docs/for_human/T26_explanation.md`
- `docs/statcalib_feasibility_gate.md`

If strictly necessary for documenting the design, Worker may add one small docs-only appendix under `docs/`.

## Required Inputs

Read at minimum:

- `docs/02_experiment_plan.md`
- `docs/reference/AI_coding_workflow.md`
- `docs/P4_benchmark_formal_protocol.md`
- `docs/review/T24_review.md`
- `docs/review/T25_p4_formal_evidence_gate_review.md`
- `docs/review/T27_teacher_diagnostics_path_audit.md`
- `docs/review/T28_review.md`
- `docs/review/T29_review.md`
- `docs/08_risks_and_open_questions.md`
- `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py`
- `cnn_fpga/decoder/param_mapper.py`

If useful, read the paper-inspired/statcalib background draft, but do not treat it as implementation authority.

## Forbidden Scope

Do not:

- modify source code, configs, benchmark runners, ParamMapper, model code, runtime code, `runs/`, or `artifacts/`
- modify `docs/02_experiment_plan.md`
- implement `statcalib`, soft-information comparator, new baseline mode, or new teacher branch
- run benchmark, training, `.tflite`, hardware, or cleanup
- create a new run directory
- change formal benchmark protocol, baseline/scenario set, seed/repeat policy, metric definitions, or result boundary
- claim calibration/statcalib evidence exists
- write `P3 real-board HIL` or `.tflite` runtime as completed

## Expected Output

Create `docs/statcalib_feasibility_gate.md` with:

- current evidence boundary
- candidate statcalib/calibration objective
- prerequisite checklist
- adopted / deferred / rejected design items
- minimal comparator interface proposal
- metrics and validation plan
- go/no-go recommendation for a future implementation task
- explicit non-claims

Create `docs/review/T26_statcalib_feasibility_gate.md` with:

- read-only scope confirmation
- files inspected
- feasibility verdict
- blockers / warnings
- recommended next task, if any

Create `docs/for_human/T26_explanation.md` with a concise non-technical explanation.

Update this task package with Worker Output and Verification Record.

## Verification

Required verification is documentation-only:

1. Confirm no source/config/run/artifact changes were made.
2. Confirm no benchmark/run directory was created.
3. Confirm `docs/statcalib_feasibility_gate.md` explicitly labels future statcalib work as unimplemented and unvalidated.
4. Confirm any future implementation recommendation includes Allowed files, Forbidden scope, Verification, and Docs to update.

## Captain Notes

T26 is a gate, not an implementation task. If the feasibility gate is positive, the next task may be a minimal statcalib implementation package. If it is negative or conditional, the next task should target the smallest missing prerequisite instead.

Do not start `T30` paper-inspired branch work or `T36` seed failure diagnosis inside T26.
