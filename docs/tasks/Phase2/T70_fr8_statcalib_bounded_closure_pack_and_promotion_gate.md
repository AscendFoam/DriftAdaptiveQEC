# T70: FR8 statcalib bounded closure pack and promotion gate

## Status

- Proposed by Captain on `2026-06-10`
- Current phase: `Phase 2: Controlled Development`
- Decision state: `Go`
- Task type: bounded read-only consolidation + task-scoped helper/test + promotion/no-promotion gate on the mainline experiment branch

## Why This Task Exists

`T64`, `T66`, `T67`, `T68`, and `T69` have now answered the bounded mainline FR8 questions in sequence:

1. `T64` proved one clean-provenance statcalib extension-lane win under the locked four-scenario protocol.
2. `T66` showed that the win is not just one accidental local sensitivity point.
3. `T67` showed that the bounded win is not grossly tied to `teacher_mode=ukf`.
4. `T68` showed that full generated-only winners do exist inside the predeclared grid.
5. `T69` showed that the strongest clean answer still does not collapse to one unique threshold and remains the persistent `window_variance_t001 = t003 = t005` tie set.

So the mainline repository no longer lacks another threshold rerun.
What it lacks now is one authoritative, code-backed closure pack that downstream governance, paper-material, and branch-integration work can cite without:

1. flattening the extension-lane boundary
2. silently rewriting `T24`
3. inventing a unique-threshold story
4. mixing mainline evidence with sidecar outputs

The next smallest honest step is therefore not another benchmark execution and not a purely manual prose note.
It is one bounded closure/gate task that consolidates the already accepted FR8 artifacts and states clearly:

1. what the current statcalib lane does prove
2. what it still does not prove
3. whether any promotion into the frozen ranked table is allowed
4. whether any unique-threshold claim is allowed
5. what kind of future task would be required before changing either answer

## Goal

Produce one bounded FR8 closure pack that:

1. reconstructs the accepted evidence chain across `T24`, `T64`, `T66`, `T67`, `T68`, and `T69` from read-only historical artifacts
2. summarizes the strongest bounded answer for each sub-question:
   - extension-lane win existence
   - local sensitivity robustness
   - teacher-anchor robustness
   - generated-only closure
   - clean-winner tie-break outcome
3. states explicitly that the strongest clean answer is still the persistent `window_variance_t001 = t003 = t005` tie set
4. states explicitly that `T24` remains the authoritative frozen ranked table and must not be rewritten by FR8 artifacts
5. gives one explicit `promotion gate` verdict for the current FR8 lane as exactly one of:
   - `no_promotion_keep_extension_lane_only`
   - `promotion_conditionally_possible_after_new_gate`
6. gives one explicit `unique-threshold gate` verdict for the current FR8 lane as exactly one of:
   - `no_unique_threshold_supported`
   - `future_selection_task_required`

## Allowed Files

Worker may modify only:

- `docs/tasks/Phase2/T70_fr8_statcalib_bounded_closure_pack_and_promotion_gate.md`
- `docs/evidence_packs/statcalib_fr8/fr8_statcalib_bounded_closure_pack.md`
- `docs/review/T70_review.md`
- `docs/for_human/T70_explanation.md`
- `docs/worker_summary/T70_worker_summary.md`
- `cnn_fpga/benchmark/build_fr8_statcalib_bounded_closure_pack.py`
- `tests/test_fr8_statcalib_bounded_closure_pack.py`

## Docs To Update

Worker must update:

- `docs/evidence_packs/statcalib_fr8/fr8_statcalib_bounded_closure_pack.md`
- `docs/review/T70_review.md`
- `docs/for_human/T70_explanation.md`
- `docs/worker_summary/T70_worker_summary.md`

Worker must not update governance docs. Captain will do that after review.

## Forbidden Scope

Worker must not:

- modify `docs/02_experiment_plan.md`
- modify any governance doc under `docs/00_*` to `docs/08_*`
- modify any file under `runs/`
- create any new run root
- rerun any benchmark
- modify `cnn_fpga/decoder/statcalib.py`
- modify `cnn_fpga/runtime/slow_loop_runtime.py`
- modify `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py`
- modify historical config files used by `T24/T64/T66/T67/T68/T69`
- rewrite `T24`, `T64`, `T66`, `T67`, `T68`, or `T69` into mature calibration-comparator, `.tflite`, real-board, or paper-grade expanded benchmark evidence
- mix sidecar outputs, sidecar worktrees, or theory-only branch materials into the mainline FR8 closure pack
- force or imply a unique threshold choice not supported by accepted artifacts

## Required Inputs

Worker must reuse:

- `docs/protocols/benchmark/P4_benchmark_formal_protocol.md`
- `docs/evidence_packs/statcalib_fr8/fr8_statcalib_extension_lane_benchmark.md`
- `docs/evidence_packs/statcalib_fr8/fr8_statcalib_extension_lane_consistency_audit.md`
- `docs/evidence_packs/statcalib_fr8/statcalib_sensitivity_bounded_benchmark.md`
- `docs/evidence_packs/statcalib_fr8/statcalib_teacher_anchor_bounded_benchmark.md`
- `docs/evidence_packs/statcalib_fr8/statcalib_generated_only_robustness_bounded_benchmark.md`
- `docs/evidence_packs/statcalib_fr8/statcalib_clean_winner_tiebreak_bounded_benchmark.md`
- `docs/review/T64_review.md`
- `docs/review/T65_review.md`
- `docs/review/T66_review.md`
- `docs/review/T67_review.md`
- `docs/review/T68_review.md`
- `docs/review/T69_review.md`
- `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743`
- `runs/p4_benchmark/T64_fr8_statcalib_extension_lane_20260527_221658`
- `runs/p4_benchmark/T66_statcalib_sensitivity_20260529_210906`
- `runs/p4_benchmark/T67_statcalib_teacher_anchor_20260601_225718`
- `runs/p4_benchmark/T68_statcalib_generated_only_20260605_205723`
- `runs/p4_benchmark/T69_statcalib_clean_winner_tiebreak_20260608_160358`

## Fixed Boundary

- Branch: current mainline experiment branch only
- Evidence scope: mock-backed software-HIL only
- Historical anchor: `T24` remains the authoritative frozen ranked table
- Extension-lane rule: `statcalib` remains separately labeled outside the frozen ranked set
- Read-only artifact rule: all historical run roots are read-only inputs only
- No-execution rule: `T70` creates no new run root and performs no new benchmark execution

## Implementation Requirements

1. Add one task-scoped closure helper that reads the accepted FR8 historical artifacts and produces one consolidated closure pack.
2. Add focused regression tests for the closure helper.
3. The closure helper must compute or restate at least:
   - the frozen `T24` anchor verdict
   - the T64 bounded extension-lane win fact
   - the T66 local sensitivity robustness fact
   - the T67 teacher-anchor robustness fact
   - the T68 full generated-only existence fact
   - the T69 persistent clean tie-set fact
   - the strongest clean answer set after T69
   - whether a unique clean reference point exists
   - whether the broader FR8 lane should be promoted into the frozen ranked table
   - whether a future unique-threshold selection task is required before choosing one threshold
4. The closure helper must represent the final promotion outcome explicitly and must not leave it implicit in prose only.
5. The closure helper must not modify historical artifacts and must not depend on sidecar outputs.
6. The final report must include one compact table that distinguishes:
   - `frozen anchor evidence`
   - `extension-lane evidence`
   - `supported claims`
   - `unsupported claims`
7. The final report must include one explicit section called `No-Promotion Gate`.
8. The final report must include one explicit section called `Unique-Threshold Gate`.
9. The final report must include one explicit section called `What A Future Task Would Need`.
10. The helper and tests must stay task-scoped. Do not refactor mainline benchmark runner semantics inside T70.

## Expected Output Artifacts

Worker must produce:

- one closure-pack report:
  - `docs/evidence_packs/statcalib_fr8/fr8_statcalib_bounded_closure_pack.md`
- one review file:
  - `docs/review/T70_review.md`
- one human explanation:
  - `docs/for_human/T70_explanation.md`
- one worker summary:
  - `docs/worker_summary/T70_worker_summary.md`

## Verification

Worker must run and report:

1. `python -m py_compile cnn_fpga/benchmark/build_fr8_statcalib_bounded_closure_pack.py`
2. `python -m unittest tests.test_fr8_statcalib_bounded_closure_pack`
3. one invocation of the new closure helper
4. explicit confirmation that:
   - no new run root was created
   - the helper reads historical artifacts read-only
   - no historical `runs/` file was modified
   - no sidecar output was used
5. explicit reporting of:
   - the final strongest clean answer set after T69
   - whether a unique clean reference point exists
   - the final promotion gate verdict
   - the final unique-threshold gate verdict
   - the minimal prerequisites for any later single-threshold-selection task

## Review No-Go Triggers

Reviewer should return `BLOCK` if any of the following happen:

1. any new run root is created
2. worker reruns any benchmark instead of consolidating historical artifacts
3. worker rewrites or relabels `T24` as if FR8 extension-lane evidence replaced it
4. worker upgrades current FR8 evidence into mature calibration comparator, `.tflite`, real-board, or paper-grade expanded benchmark evidence
5. worker silently chooses one unique threshold from the surviving T69 tie set
6. worker mixes sidecar outputs or theory-only branch materials into the mainline closure pack
7. helper output omits the explicit promotion/no-promotion verdict
8. helper output omits the explicit unique-threshold gate verdict

## Captain Notes

- This task is intentionally more substantial than a simple narrative follow-up, but it remains bounded and read-only with respect to historical experimental evidence.
- The point is to create one reusable, self-consistent FR8 closure artifact that later docs can cite without overstating what the run roots show.
- If the final gate verdict is still `no_promotion_keep_extension_lane_only`, that is a successful and honest outcome.
- Keep theory-only branch materials completely separate.

## Worker Output

Worker must report:

1. what files were changed
2. exact verification commands executed
3. whether any new run root was created
4. the final strongest clean answer set after T69
5. whether a unique clean reference point exists
6. the final promotion gate verdict
7. the final unique-threshold gate verdict
8. the minimal prerequisites for any later threshold-selection task
9. remaining risks or interpretation limits

## Worker Output

### Files changed

- `cnn_fpga/benchmark/build_fr8_statcalib_bounded_closure_pack.py`
- `tests/test_fr8_statcalib_bounded_closure_pack.py`
- `docs/evidence_packs/statcalib_fr8/fr8_statcalib_bounded_closure_pack.md`
- `docs/review/T70_review.md`
- `docs/for_human/T70_explanation.md`
- `docs/worker_summary/T70_worker_summary.md`
- `docs/tasks/Phase2/T70_fr8_statcalib_bounded_closure_pack_and_promotion_gate.md`

### Exact verification commands executed

1. `C:\ProgramData\anaconda3\python.exe -m unittest tests.test_fr8_statcalib_bounded_closure_pack`
2. `C:\ProgramData\anaconda3\python.exe -m py_compile cnn_fpga\benchmark\build_fr8_statcalib_bounded_closure_pack.py`
3. `C:\ProgramData\anaconda3\python.exe -m cnn_fpga.benchmark.build_fr8_statcalib_bounded_closure_pack`
4. `@(Get-ChildItem -LiteralPath 'runs\\p4_benchmark' -Directory | Where-Object { $_.Name -like 'T70*' }).Count`
5. `git diff --name-only -- runs`

### Run-root / read-only confirmation

- new run root created: `No`
- helper output field `no_new_run_root_created = true`
- helper output field `historical_runs_modified = false`
- helper output field `sidecar_outputs_used = false`
- `git diff --name-only -- runs` returned empty

### Final strongest clean answer set after T69

- `statcalib_window_variance_t001`
- `statcalib_window_variance_t003`
- `statcalib_window_variance_t005`

### Unique clean reference point

- exists: `False`

### Final gate verdicts

- promotion gate verdict: `no_promotion_keep_extension_lane_only`
- unique-threshold gate verdict: `future_selection_task_required`

### Minimal prerequisites for any later threshold-selection task

1. Predeclare a selection criterion that is not already tied by the current `T69` mean/worst-case LER evidence.
2. Lock the candidate set and decision rule before any new execution or downstream retelling.
3. Keep `T24` frozen and keep `statcalib` labeled as an extension lane unless a later promotion gate explicitly changes that status.
4. If a stronger-than-mock-backed claim is desired, open a separate bounded validation task for that target surface instead of reusing `T64-T69` as deployment proof.

### Remaining risks / interpretation limits

1. `R24` remains open as an overclaim/promotion boundary, not as an unresolved tie-break execution question.
2. The broader predeclared statcalib grid is still not uniformly clean.
3. Current FR8 evidence remains mock-backed software-HIL extension-lane evidence only.
4. Current accepted artifacts still do not support a unique final threshold claim.
