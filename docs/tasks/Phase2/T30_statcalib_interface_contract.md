# T30: Statcalib comparator interface contract and bounded implementation package

## Status

- Created by Captain on `2026-05-12`
- Current phase: `Phase 2: Controlled Development`
- Decision state: `Go`
- Task type: bounded interface contract / minimal implementation package

## Why This Task Exists

`T26` completed the calibration/statcalib feasibility gate with `CONDITIONAL_GO`. The gate allows statcalib only as a separately labeled comparator lane. It does not allow silent insertion into the frozen T24 benchmark set, broad benchmark expansion, `.tflite` work, or real-board work.

Reviewer non-blocking feedback on T26 also identified the next concrete gap: `StatCalibInput` and `StatCalibOutput` were still conceptual. T30 must make that contract exact before any broader validation is considered.

## Goal

Define and, if feasible within this task, minimally implement a separate statcalib comparator interface with exact field names, types, status semantics, and an interface-level verification path.

This is not a benchmark task. It is not a paper-claim task.

## Allowed Files

Worker may modify:

- `docs/tasks/Phase2/T30_statcalib_interface_contract.md`
- `docs/evidence_packs/statcalib_fr8/statcalib_feasibility_gate.md`
- `docs/review/T30_review.md`
- `docs/for_human/T30_explanation.md`
- `cnn_fpga/decoder/statcalib.py`
- `tests/test_statcalib_interface.py`

If the existing test layout requires a different focused test path, Worker may add one small equivalent test file under `tests/`, but must explain why.

## Required Inputs

Read at minimum:

- `docs/02_experiment_plan.md`
- `docs/reference/AI_coding_workflow.md`
- `docs/evidence_packs/statcalib_fr8/statcalib_feasibility_gate.md`
- `docs/review/T26_review.md`
- `docs/08_risks_and_open_questions.md`
- `docs/protocols/benchmark/P4_benchmark_formal_protocol.md`
- `cnn_fpga/decoder/param_mapper.py`
- `cnn_fpga/runtime/slow_loop_runtime.py`
- `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py`

## Forbidden Scope

Do not:

- modify `docs/02_experiment_plan.md`
- modify existing `ParamMapper.map_prediction()` semantics for current modes
- modify benchmark runner semantics, formal protocol, baseline/scenario set, seed/repeat policy, or result boundary
- run formal benchmark, long benchmark, training, `.tflite`, hardware, or cleanup
- add statcalib to the frozen ranked benchmark set
- create or rewrite `runs/` or `artifacts/` evidence
- claim statcalib has been validated beyond interface-level verification
- touch real-board, `.tflite`, teacher-representation long-run, or T36 seed diagnosis scope

## Expected Output

Produce a concrete contract for:

- `StatCalibInput`
- `StatCalibOutput`
- status values and reason strings
- provenance / source fields
- conversion boundary to `DecoderRuntimeParams`

If implementing code, keep it minimal and deterministic:

- a separate `cnn_fpga/decoder/statcalib.py` module
- no changes to current mapper behavior
- no benchmark integration unless it is purely an interface-level import/static smoke

Create or update:

- `docs/review/T30_review.md`
- `docs/for_human/T30_explanation.md`
- this task package Worker Output and Verification Record

## Verification

Required verification is interface-level only:

1. Run a focused test or static smoke for `StatCalibInput` / `StatCalibOutput`.
2. Confirm existing `ParamMapper` behavior is not modified.
3. Confirm no benchmark run directory was created.
4. Confirm no source/config files outside the allowed set were changed.
5. Confirm output docs state that statcalib is not yet formal-benchmark validated.

## Docs To Update

- `docs/tasks/Phase2/T30_statcalib_interface_contract.md`
- `docs/review/T30_review.md`
- `docs/for_human/T30_explanation.md`
- `docs/evidence_packs/statcalib_fr8/statcalib_feasibility_gate.md` only if clarifying the interface contract, not rewriting the gate conclusion

## Reviewer Type

Adversarial review.

Focus areas:

- interface honesty
- no silent benchmark expansion
- no placeholder marketed as validated statcalib
- `DecoderRuntimeParams` compatibility
- status/null semantics preservation
- no `ParamMapper` mainline behavior drift

## Captain Notes

T30 may produce a minimal interface implementation, but only if it remains a separate comparator lane and is verified by focused tests/static smoke. If Worker finds implementation requires touching benchmark integration, config matrices, or `ParamMapper` mainline semantics, stop and report a blocker instead of expanding scope.

## Worker Output

- Read the required T30 inputs, including the T26 gate, current risks, `ParamMapper`, `SlowLoopRuntime`, `DecoderRuntimeParams`, and the current benchmark writer boundary.
- Implemented a separate contract module at `cnn_fpga/decoder/statcalib.py`.
- Added one focused interface test file at `tests/test_statcalib_interface.py` because the repo did not already have a `tests/` layout to extend.
- Tightened `docs/evidence_packs/statcalib_fr8/statcalib_feasibility_gate.md` from conceptual I/O language to an exact field-level contract and explicit status/reason sets.
- Added `docs/review/T30_review.md` and `docs/for_human/T30_explanation.md`.

## Verification Record

1. Focused interface test:
   - `C:\ProgramData\anaconda3\python.exe -m unittest tests.test_statcalib_interface`
   - Result: passed (`Ran 6 tests`, `OK`)
2. Static compile smoke:
   - `C:\ProgramData\anaconda3\python.exe -m py_compile cnn_fpga\decoder\statcalib.py tests\test_statcalib_interface.py`
   - Result: passed
3. Existing `ParamMapper` behavior not modified:
   - `git diff --name-only -- cnn_fpga/decoder/param_mapper.py`
   - Result: no diff
4. No benchmark run directory created:
   - confirmed during task closeout; T30 did not run benchmark commands
5. Scope check:
   - no source/config files outside the allowed set were changed
6. Documentation honesty:
   - updated docs explicitly state that statcalib is not yet formal-benchmark validated
