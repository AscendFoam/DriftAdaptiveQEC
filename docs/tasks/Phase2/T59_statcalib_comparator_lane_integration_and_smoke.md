# T59: Statcalib separate comparator lane integration and bounded smoke

## Status

- Proposed by Captain on `2026-05-26`
- Current phase: `Phase 2: Controlled Development`
- Decision state: `Go`
- Task type: bounded code + smoke integration task on the mainline experiment branch

## Why This Task Exists

`T57` and `T58` have now closed `FR7` and `FR6` honestly. The largest remaining mainline paper-material gap is `FR8`: a `statcalib` comparator result table.

However, the repository still does **not** have an integrated `statcalib` comparator lane:

- `T26` only gave a `CONDITIONAL_GO` feasibility gate
- `T30` only added an interface-only contract and focused tests
- there is still no slow-loop runtime path, no benchmark mode, and no bounded run evidence for `statcalib`

So the next smallest honest step is **not** to write an `FR8` result table yet. The next step is to integrate `statcalib` as a **separate comparator lane** and prove it runs end-to-end in one bounded smoke.

This task must stay on the mainline experiment-evidence branch and must not interfere with theory-only branch materials.

## Goal

Produce the smallest believable end-to-end `statcalib` lane by answering:

1. can the repository run a separately labeled `statcalib` slow-loop mode without changing frozen `T24` semantics
2. can that mode emit explicit status/reason semantics through the runtime and benchmark outputs
3. can one bounded task-scoped smoke run complete successfully
4. what still remains before any later `FR8` formal result-table task

## Allowed Files

Worker may modify only:

- `docs/tasks/Phase2/T59_statcalib_comparator_lane_integration_and_smoke.md`
- `docs/statcalib_comparator_lane_smoke.md`
- `docs/review/T59_review.md`
- `docs/for_human/T59_explanation.md`
- `docs/worker_summary/T59_worker_summary.md`
- `cnn_fpga/decoder/statcalib.py`
- `cnn_fpga/runtime/slow_loop_runtime.py`
- `cnn_fpga/benchmark/run_hil_suite.py`
- `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py`
- `tests/test_statcalib_interface.py`

Worker may create:

- `tests/test_statcalib_runtime_smoke.py`
- `cnn_fpga/config/p4_multiscenario_statcalib_smoke.yaml`
- at most one additional statcalib-specific helper config if strictly necessary under `cnn_fpga/config/`
- one task-scoped run root under `runs/p4_benchmark/T59_statcalib_lane_smoke_*`

## Docs To Update

This task should update only:

1. `docs/statcalib_comparator_lane_smoke.md`
2. `docs/review/T59_review.md`
3. `docs/for_human/T59_explanation.md`
4. `docs/worker_summary/T59_worker_summary.md`
5. this task package itself, only to append Worker output and verification notes after completion

## Forbidden Scope

This task must not:

1. edit `docs/00_project_snapshot.md`, `docs/01_legacy_audit.md`, `docs/02_experiment_plan.md`, `docs/03_hil_p4_boundary_audit.md`, `docs/04_task_board.md`, `docs/05_decision_log.md`, `docs/06_repo_noise_governance.md`, `docs/07_handoff.md`, or `docs/08_risks_and_open_questions.md`
2. edit theory-only materials such as `docs/reference/延申理论.md`, `docs/reference/延伸改进思路.md`, or files under `docs/follow-up_plan/`
3. modify `cnn_fpga/decoder/param_mapper.py` or silently change existing `ukf`, `window_variance`, `cnn_fpga`, `hybrid_residual_b`, or other frozen mainline mode semantics
4. insert `statcalib` into the frozen `T24` ranked set as if it had always been there
5. widen baseline families, scenario families, seed policy, repeat policy, CI stopping, or metric definitions beyond the bounded smoke below
6. touch `.tflite` runtime, real-board, cleanup, training, benchmark expansion, soft-information comparator, or new intervention scope
7. rewrite or delete historical `runs/` or `artifacts/`

## Required Inputs

Read at minimum:

- `docs/04_task_board.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/statcalib_feasibility_gate.md`
- `docs/review/T25_p4_formal_evidence_gate_review.md`
- `docs/review/T26_review.md`
- `docs/review/T30_review.md`
- `docs/paper_claim_evidence_ledger.md`
- `docs/reality_recovery/04_figure_and_result_ledger.md`
- `docs/reality_recovery/05_paper_claim_risk_table.md`
- `cnn_fpga/decoder/statcalib.py`
- `cnn_fpga/runtime/slow_loop_runtime.py`
- `cnn_fpga/benchmark/run_hil_suite.py`
- `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py`
- `tests/test_statcalib_interface.py`
- `cnn_fpga/config/p4_multiscenario_hybrid_b.yaml`

## Fixed Boundary

This task is locked to the following boundary:

1. `statcalib` must remain a **separate comparator lane**
2. frozen `T24` semantics must remain unchanged for historical evidence
3. this task is allowed to add a new bounded smoke config and one new task-scoped run root only
4. this task is **not** `FR8` formal evidence yet
5. this task must stay fully separate from theory-only branch materials

## Bounded Smoke Matrix

Use exactly this smoke boundary unless a smaller subset is required to fix a concrete blocker:

1. scenarios:
   - `static_bias_theta`
   - `linear_ramp`
2. modes:
   - `ukf`
   - `hybrid_residual_b`
   - `statcalib`
3. repeats:
   - `1`
4. seed policy:
   - `--paired-seeds`
5. run root:
   - `runs/p4_benchmark/T59_statcalib_lane_smoke_*`

This is a smoke only. Do not widen to four scenarios, more modes, more repeats, or formal ranking claims.

## Implementation Requirements

### 1. Separate runtime mode

Add a distinct slow-loop mode named `statcalib`.

It must:

- remain separate from existing frozen modes
- use the `StatCalibInput` / `StatCalibOutput` contract from `cnn_fpga/decoder/statcalib.py`
- emit explicit `status`, `reason`, and provenance metadata
- convert to `DecoderRuntimeParams` only through the statcalib contract path

### 2. Minimal estimator semantics

Use the smallest transparent estimator that keeps the lane honest.

Safe default:

- anchor on a teacher mode such as `window_variance`
- derive a bounded residual-`b` correction from a small set of current-window histogram/calibration features
- clip the emitted `delta_b`
- emit `generated` only when the signal is valid
- otherwise emit `not_generated` or `diagnostic_error` explicitly

Do not over-design this. The purpose is lane integration and status propagation, not a full calibration research result.

### 3. Status propagation

Plumb `statcalib` status/reason into runtime metadata and benchmark outputs so the bounded smoke can answer:

- did the lane execute
- did it emit params
- was the result `generated`, `not_generated`, or `diagnostic_error`

If a small summary/report field addition is needed in `run_hil_suite.py` or `run_p4_multiscenario_benchmark.py`, keep it narrowly scoped to this comparator-lane visibility problem.

### 4. Dedicated smoke config

Create one dedicated smoke config such as:

- `cnn_fpga/config/p4_multiscenario_statcalib_smoke.yaml`

It must:

- keep `statcalib` explicitly labeled
- avoid modifying frozen historical configs
- stay bounded to the smoke matrix above

## Expected Output Artifacts

Create `docs/statcalib_comparator_lane_smoke.md` with:

1. exact code/config changes made
2. exact smoke command
3. interpreter used
4. run root path
5. smoke matrix
6. per-mode completion/status summary
7. whether `statcalib` emitted params in the smoke
8. explicit statement of what remains before any later `FR8` formal result-table task

Create `docs/review/T59_review.md` with:

1. scope and boundary check
2. confirmation that `statcalib` stayed a separate lane
3. confirmation that frozen `T24` semantics were not rewritten
4. smoke verification result
5. whether the next step should be a formal `FR8` result-table task or another prerequisite

Create `docs/for_human/T59_explanation.md` with a short human-facing summary.

Create `docs/worker_summary/T59_worker_summary.md` with a concise worker-facing summary of changes, verification, and residual risk.

## Verification

Required verification:

1. run focused tests for the statcalib contract/runtime path
2. run `py_compile` or equivalent minimal static verification on changed Python files
3. run the bounded smoke command and record the exact command/output run root
4. confirm `comparison.csv` contains a separately labeled `statcalib` mode
5. confirm `statcalib` status/reason is visible end-to-end in the smoke outputs
6. confirm existing frozen configs and historical run roots were not modified
7. confirm no theory-only branch materials were touched

Required command shape:

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config cnn_fpga/config/p4_multiscenario_statcalib_smoke.yaml --scenario static_bias_theta --scenario linear_ramp --mode ukf --mode hybrid_residual_b --mode statcalib --paired-seeds --repeats 1
```

If the final command must differ, document exactly why and keep it no broader than the same smoke boundary.

## Review No-Go Triggers

Review should be treated as `BLOCK` if any of the following happen:

1. `statcalib` is inserted into the frozen `T24` ranked set as if historical evidence had changed
2. existing mainline mode semantics are silently rewritten
3. the worker widens the smoke into a formal benchmark expansion
4. the worker touches `.tflite`, real-board, cleanup, training, or theory-only branch materials
5. historical `runs/` or `artifacts/` are rewritten
6. no bounded smoke is actually run

## Captain Notes

`T58` is complete and accepted as `PASS_WITH_WARNINGS`.

The largest remaining mainline paper-material gap is `FR8`, but there is still no integrated `statcalib` lane. Therefore the next step is integration + smoke, not a direct result-table claim.

This task must stay on `main`-branch experiment work and must remain isolated from the user's separate theory branch.

## Worker Output

### What changed

- Pending.

### Verification Notes

- Pending.
