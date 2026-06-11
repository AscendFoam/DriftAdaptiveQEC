# T65: FR8 extension-lane consistency guard and report closeout

## Status

- Proposed by Captain on `2026-05-29`
- Current phase: `Phase 2: Controlled Development`
- Decision state: `Go`
- Task type: code/test/docs hardening task on the mainline experiment branch

## Why This Task Exists

`T64` already produced one clean-provenance bounded FR8 extension-lane benchmark on the locked four-scenario protocol.

That execution result is acceptable, but `docs/review/T64_review.md` also surfaced three carry-forward issues:

1. the T64 result doc used execution-shape wording that is looser than the task package's accepted wording
2. the T64 result doc attributed a finish timestamp to `summary.json` even though that value actually came from filesystem metadata
3. the repository still lacks a lightweight automated guard that future FR8 reuse is report-to-artifact consistent and that the frozen five-mode subset is still preserved against `T24`

The next smallest honest step is therefore not another benchmark run. It is one bounded hardening task that:

1. corrects the T64 report wording
2. adds a reusable consistency audit helper
3. adds focused regression coverage for that helper
4. produces one explicit T64 consistency-audit doc before any later FR8 figure/table/gate reuse

## Goal

Produce one bounded hardening package that answers:

1. whether the T64 report wording now matches the actual task-package boundary and run artifacts
2. whether the repository has a lightweight automated guard for FR8 extension-lane report/artifact consistency
3. whether T64 still proves frozen-subset preservation against historical `T24` without silently rewriting that table
4. whether the T64 result pack can now be reused as a self-audited bounded extension-lane artifact while keeping the same mock-backed software-HIL boundary

## Allowed Files

Worker may modify only:

- `docs/tasks/Phase2/T65_fr8_extension_lane_consistency_guard_and_closeout.md`
- `docs/evidence_packs/statcalib_fr8/fr8_statcalib_extension_lane_benchmark.md`
- `docs/evidence_packs/statcalib_fr8/fr8_statcalib_extension_lane_consistency_audit.md`
- `docs/review/T65_review.md`
- `docs/for_human/T65_explanation.md`
- `docs/worker_summary/T65_worker_summary.md`
- `cnn_fpga/benchmark/audit_fr8_extension_lane_consistency.py`
- `tests/test_fr8_extension_lane_consistency.py`

Worker may create:

- no new run root
- no new artifact directory outside the allowed-file set

## Docs To Update

This task should update only:

1. `docs/evidence_packs/statcalib_fr8/fr8_statcalib_extension_lane_benchmark.md`
2. `docs/evidence_packs/statcalib_fr8/fr8_statcalib_extension_lane_consistency_audit.md`
3. `docs/review/T65_review.md`
4. `docs/for_human/T65_explanation.md`
5. `docs/worker_summary/T65_worker_summary.md`
6. `cnn_fpga/benchmark/audit_fr8_extension_lane_consistency.py`
7. `tests/test_fr8_extension_lane_consistency.py`
8. this task package itself, only to append Worker output and verification notes after completion

## Forbidden Scope

This task must not:

1. edit `docs/00_project_snapshot.md`, `docs/01_legacy_audit.md`, `docs/02_experiment_plan.md`, `docs/03_hil_p4_boundary_audit.md`, `docs/04_task_board.md`, `docs/05_decision_log.md`, `docs/06_repo_noise_governance.md`, `docs/07_handoff.md`, or `docs/08_risks_and_open_questions.md`
2. edit theory-only materials or files under `docs/follow-up_plan/`
3. modify any historical run artifact under `runs/`
4. rerun benchmark execution, create a new run root, or resume into any historical run root
5. modify benchmark runner semantics, estimator semantics, decoder/runtime semantics, or any source-tree benchmark config
6. change the frozen four scenarios, frozen five-mode set, `statcalib` comparator semantics, paired-seed policy, repeat policy, or reporting boundary of `T24` / `T64`
7. touch `.tflite`, real-board, training, cleanup, paper-prose expansion, comparator expansion, or theory-branch work
8. rewrite `T64` as `.tflite` validation, real-board validation, paper-grade expanded benchmark evidence, or a historical `T24` replacement

## Required Inputs

Read at minimum:

- `docs/04_task_board.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/tasks/Phase2/T64_fr8_statcalib_extension_lane_bounded_benchmark.md`
- `docs/evidence_packs/statcalib_fr8/fr8_statcalib_extension_lane_benchmark.md`
- `docs/review/T64_review.md`
- `runs/p4_benchmark/T64_fr8_statcalib_extension_lane_20260527_221658/summary.json`
- `runs/p4_benchmark/T64_fr8_statcalib_extension_lane_20260527_221658/launch_plan.json`
- `runs/p4_benchmark/T64_fr8_statcalib_extension_lane_20260527_221658/progress.jsonl`
- `runs/p4_benchmark/T64_fr8_statcalib_extension_lane_20260527_221658/comparison.csv`
- `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/comparison.csv`
- `docs/protocols/benchmark/P4_benchmark_formal_protocol.md`

## Fixed Boundary

This task is locked to the following boundary:

1. `T64` experimental evidence is frozen; this task hardens report/audit quality only
2. current evidence remains mock-backed software-HIL only
3. the historical `T24` frozen five-mode ranked table remains authoritative and must not be silently rewritten
4. `statcalib` remains only a separately labeled extension lane
5. no new benchmark evidence may be created by this task
6. no new deployment-boundary claim may be introduced by this task

## Implementation Requirements

### 1. T64 report wording closeout

Update `docs/evidence_packs/statcalib_fr8/fr8_statcalib_extension_lane_benchmark.md` so that it:

1. uses execution-shape wording that is faithful to both:
   - the accepted `T64` task-package execution shapes
   - the actual T64 run artifact pattern
2. does not attribute any timestamp to `summary.json` unless that field actually exists inside the JSON
3. explicitly distinguishes:
   - JSON-recorded fields
   - filesystem metadata used only as auxiliary provenance
4. does not alter any numeric table unless a direct artifact-to-doc mismatch is proven

### 2. Add one reusable consistency audit helper

Create one lightweight audit helper:

- `cnn_fpga/benchmark/audit_fr8_extension_lane_consistency.py`

It must read the existing T64/T24 artifacts and the T64 report, and check at minimum:

1. launch/run/report provenance anchors are derived from real artifact fields
2. the report execution-shape wording does not contradict the accepted task-package shapes
3. the locked four scenarios and frozen five-mode ordering plus `statcalib` extension lane are preserved
4. `paired_seeds=true` and `repeats=2` are preserved
5. `progress.jsonl` has no duplicate `running` record for the same `(scenario, mode, repeat)` key
6. the T64 frozen five-mode subset still matches `T24` on all 20 frozen comparison rows for:
   - `final_ler_mean`
   - `overflow_rate_mean`
7. the report keeps the required boundary statements:
   - mock-backed software-HIL only
   - separate extension lane only
   - not a rewrite of `T24`
   - not `.tflite`
   - not real-board

The helper must fail loudly if a required check fails. It must not hardcode a pass result.

### 3. Add focused regression coverage

Add focused regression coverage in:

- `tests/test_fr8_extension_lane_consistency.py`

The test coverage must exercise the new audit logic on the current T64/T24 artifact set or its core pure helper functions. The goal is not broad test volume; it is to prevent the specific T64 report/artifact consistency drift from silently reappearing.

### 4. Produce one explicit consistency-audit doc

Create:

- `docs/evidence_packs/statcalib_fr8/fr8_statcalib_extension_lane_consistency_audit.md`

This doc must summarize:

1. exact inputs audited
2. exact checks run
3. pass/fail result for each check
4. whether the T64 report is now artifact-consistent
5. whether frozen-subset preservation against `T24` still holds
6. what boundary wording remains mandatory after the audit

## Expected Output Artifacts

Create or update:

1. `docs/evidence_packs/statcalib_fr8/fr8_statcalib_extension_lane_benchmark.md`
2. `docs/evidence_packs/statcalib_fr8/fr8_statcalib_extension_lane_consistency_audit.md`
3. `docs/review/T65_review.md`
4. `docs/for_human/T65_explanation.md`
5. `docs/worker_summary/T65_worker_summary.md`
6. `cnn_fpga/benchmark/audit_fr8_extension_lane_consistency.py`
7. `tests/test_fr8_extension_lane_consistency.py`

## Verification

Required verification:

1. `python -m unittest tests/test_fr8_extension_lane_consistency.py`
2. `python -m py_compile cnn_fpga/benchmark/audit_fr8_extension_lane_consistency.py`
3. run the new audit helper against:
   - `docs/evidence_packs/statcalib_fr8/fr8_statcalib_extension_lane_benchmark.md`
   - `docs/tasks/Phase2/T64_fr8_statcalib_extension_lane_bounded_benchmark.md`
   - `runs/p4_benchmark/T64_fr8_statcalib_extension_lane_20260527_221658`
   - `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743`
4. confirm no new run root was created
5. confirm no file under `runs/` was modified

Suggested command shape:

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.audit_fr8_extension_lane_consistency --task-package docs/tasks/Phase2/T64_fr8_statcalib_extension_lane_bounded_benchmark.md --report docs/evidence_packs/statcalib_fr8/fr8_statcalib_extension_lane_benchmark.md --run-dir runs/p4_benchmark/T64_fr8_statcalib_extension_lane_20260527_221658 --frozen-baseline-run-dir runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743
```

If the final command differs, document exactly why and keep it no broader than the same boundary.

## Review No-Go Triggers

Review should be treated as `BLOCK` if any of the following happen:

1. the worker creates a new run root or reruns benchmark execution
2. the worker modifies any historical artifact under `runs/`
3. the worker changes benchmark, decoder, runtime, or `statcalib` semantics rather than adding bounded audit logic
4. the worker widens the task into `.tflite`, real-board, cleanup, training, or theory-branch scope
5. the worker edits governance docs or `docs/02_experiment_plan.md`
6. the audit helper hardcodes expected outcomes instead of deriving them from repository artifacts
7. the final docs silently upgrade `T64` into paper-grade expanded benchmark evidence, `.tflite` validation, or real-board validation

## Captain Notes

`T64` is complete and accepted only as `PASS_WITH_WARNINGS`.

`T65` exists to close `R28` and tighten `R24` reuse discipline. It is not permission to open a new FR8 benchmark lane or to widen the deployment boundary.

## Worker Output

### Status

- Completed within allowed scope
- Task end-state not marked here; waiting for Worker completion and external review

### What Changed

1. Updated `docs/evidence_packs/statcalib_fr8/fr8_statcalib_extension_lane_benchmark.md` to:
   - replace loose execution-shape wording with artifact-visible wording
   - remove the false `finish timestamp from summary.json` attribution
   - distinguish artifact-recorded fields, live repo observations, and auxiliary filesystem metadata
2. Added `cnn_fpga/benchmark/audit_fr8_extension_lane_consistency.py`
3. Added `tests/test_fr8_extension_lane_consistency.py`
4. Added `docs/evidence_packs/statcalib_fr8/fr8_statcalib_extension_lane_consistency_audit.md`
5. Added `docs/review/T65_review.md`
6. Added `docs/for_human/T65_explanation.md`
7. Added `docs/worker_summary/T65_worker_summary.md`

### Verification Notes

1. `C:\ProgramData\anaconda3\python.exe -m unittest tests.test_fr8_extension_lane_consistency`
   - passed (`Ran 5 tests`, `OK`)
2. `C:\ProgramData\anaconda3\python.exe -m py_compile cnn_fpga/benchmark/audit_fr8_extension_lane_consistency.py`
   - passed
3. `C:\ProgramData\anaconda3\python.exe -m cnn_fpga.benchmark.audit_fr8_extension_lane_consistency --task-package docs/tasks/Phase2/T64_fr8_statcalib_extension_lane_bounded_benchmark.md --report docs/evidence_packs/statcalib_fr8/fr8_statcalib_extension_lane_benchmark.md --run-dir runs/p4_benchmark/T64_fr8_statcalib_extension_lane_20260527_221658 --frozen-baseline-run-dir runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743`
   - passed (`8/8` audit checks)
4. No new run root was created
5. No file under `runs/` was modified

### Remaining Risk

- `T65` closes the report/artifact consistency hardening gap, but it does not close the substantive comparator-scope caution behind `R24`.
- `T64` remains mock-backed software-HIL evidence only and is still not `.tflite`, real-board, or mature calibration-comparator validation.
