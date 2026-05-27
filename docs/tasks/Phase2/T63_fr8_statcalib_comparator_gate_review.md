# T63: FR8 statcalib comparator gate review

## Status

- Proposed by Captain on `2026-05-27`
- Current phase: `Phase 2: Controlled Development`
- Decision state: `Go`
- Task type: docs-only gate review on the mainline experiment branch

## Why This Task Exists

`T59` integrated `statcalib` as a separate comparator lane.

`T60` then closed the cross-mode semantics and regression-hardening blocker.

`T61` tried to close the provenance blocker but failed, so it was judged `BLOCK`.

`T62` repaired that exact blocker and is now reviewable as a clean-provenance bounded rerun:

1. clean `main` preflight
2. one uninterrupted invocation
3. one T62-scoped run root
4. launch / finish / `summary.json` commit identity match
5. the bounded `statcalib` advantage persisted

That still does **not** mean the repository should silently jump to `FR8`.

The next smallest honest step is a docs-only gate review that decides whether current evidence is strong enough to justify opening one bounded `FR8` formal comparator-result task, or whether one more prerequisite is still required first.

## Goal

Produce a bounded gate-review answer to these questions:

1. what exactly is already validated for the `statcalib` comparator lane after `T59` through `T62`
2. what is still unvalidated and therefore cannot be claimed yet
3. whether `R27` should now be treated as closed by `T62`
4. whether the next honest task is:
   - a bounded `FR8` formal comparator-result-table task, or
   - one more prerequisite before any `FR8` task is opened

## Allowed Files

Worker may modify only:

- `docs/tasks/Phase2/T63_fr8_statcalib_comparator_gate_review.md`
- `docs/fr8_statcalib_comparator_gate_review.md`
- `docs/review/T63_review.md`
- `docs/for_human/T63_explanation.md`
- `docs/worker_summary/T63_worker_summary.md`

## Docs To Update

This task should update only:

1. `docs/fr8_statcalib_comparator_gate_review.md`
2. `docs/review/T63_review.md`
3. `docs/for_human/T63_explanation.md`
4. `docs/worker_summary/T63_worker_summary.md`
5. this task package itself, only to append Worker output and verification notes after completion

## Forbidden Scope

This task must not:

1. edit `docs/00_project_snapshot.md`, `docs/01_legacy_audit.md`, `docs/02_experiment_plan.md`, `docs/03_hil_p4_boundary_audit.md`, `docs/04_task_board.md`, `docs/05_decision_log.md`, `docs/06_repo_noise_governance.md`, `docs/07_handoff.md`, or `docs/08_risks_and_open_questions.md`
2. edit theory-only materials or files under `docs/follow-up_plan/`
3. modify any source file, test file, or source-tree config file
4. run any benchmark, smoke, training, `.tflite`, hardware, cleanup, or new analysis job
5. create or rewrite any `runs/` or `artifacts/` directory
6. silently promote `T59/T60/T61/T62` into completed `FR8` evidence
7. write any `.tflite` validation, real-board validation, expanded benchmark, or paper-grade comparator claim as already complete

## Required Inputs

Read at minimum:

- `docs/04_task_board.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/review/T26_review.md`
- `docs/review/T30_review.md`
- `docs/review/T59_review.md`
- `docs/review/T60_review.md`
- `docs/review/T61_review.md`
- `docs/review/T62_review.md`
- `docs/statcalib_comparator_lane_smoke.md`
- `docs/statcalib_lane_isolation_and_regression_hardening.md`
- `docs/statcalib_fairness_sanity.md`
- `docs/statcalib_provenance_isolated_fairness_rerun.md`
- `docs/paper_claim_evidence_ledger.md`
- `docs/paper_ablation_result_pack.md`
- `docs/worker_summary/T59_worker_summary.md`
- `docs/worker_summary/T60_worker_summary.md`
- `docs/worker_summary/T61_worker_summary.md`
- `docs/worker_summary/T62_worker_summary.md`
- `docs/P4_benchmark_formal_protocol.md`
- `runs/p4_benchmark/t59statc_20260526_211532_3a3d00_23740/summary.json`
- `runs/p4_benchmark/T61_statcalib_fairness_sanity_20260527_015239/summary.json`
- `runs/p4_benchmark/T62_statcalib_provenance_isolated_20260527_122943/summary.json`

## Fixed Boundary

This task is locked to the following boundary:

1. this is a gate-review task, not an execution task
2. it must reuse only already existing repository evidence
3. it must not re-litigate the truth boundary that `T62` remains mock-backed software-HIL evidence only
4. it must not decide more than one next bounded task
5. it must keep mainline experiment evidence separate from theory-only branch materials

## Required Analysis

The gate review must explicitly answer:

### 1. What is already closed

At minimum, evaluate whether the repository can now honestly say:

1. `statcalib` separate-lane integration exists
2. cross-mode fallback leakage is closed
3. direct regression hardening exists
4. provenance-clean bounded fairness sanity evidence exists

### 2. What is still open

At minimum, evaluate whether current evidence still lacks:

1. broader fairness/robustness evidence outside the tiny two-scenario smoke matrix
2. stronger justification that the minimal `statcalib` heuristic is a defendable formal comparator rather than only a bounded smoke lane
3. any comparator evidence beyond mock-backed software-HIL scope

### 3. Whether `R27` is now closed

The report must explicitly decide whether:

1. `R27` should now be treated as closed by `T62`, or
2. some part of `R27` must remain open

If it remains open, the report must identify the exact remaining blocker and cite local evidence.

### 4. The next honest task

The report must end with exactly one of the following recommendations:

1. `GO_FOR_BOUNDED_FR8_TASK`
2. `NO_GO_NEEDS_ONE_MORE_PREREQUISITE`

If the answer is `GO_FOR_BOUNDED_FR8_TASK`, the report must define the smallest safe FR8 scope in concrete terms.

If the answer is `NO_GO_NEEDS_ONE_MORE_PREREQUISITE`, the report must name exactly one next bounded prerequisite and explain why it is smaller and safer than opening FR8 directly.

## Expected Output Artifacts

Create `docs/fr8_statcalib_comparator_gate_review.md` with:

1. a short verdict section
2. a table of already-closed vs still-open evidence items
3. an explicit `R27` closure decision
4. an explicit statement that current evidence remains mock-backed software-HIL only
5. a final recommendation of either `GO_FOR_BOUNDED_FR8_TASK` or `NO_GO_NEEDS_ONE_MORE_PREREQUISITE`
6. if `GO`, a concrete smallest FR8 scope proposal
7. if `NO_GO`, a concrete smallest prerequisite proposal

Create `docs/review/T63_review.md` with:

1. scope/boundary check
2. verification that no code/config/run changes were made
3. whether the gate reasoning is consistent with the cited repository evidence
4. whether the next-step recommendation is appropriately bounded

Create `docs/for_human/T63_explanation.md` with a short human-facing summary.

Create `docs/worker_summary/T63_worker_summary.md` with a concise worker-facing summary of changes, verification, and residual risk.

## Verification

Required verification:

1. `git diff --name-only` stays inside the T63 allowed-file set
2. no source, test, config, `runs/`, or `artifacts/` path is modified or created
3. all claims in `docs/fr8_statcalib_comparator_gate_review.md` cite concrete existing local evidence
4. the report states explicitly that:
   - current evidence remains mock-backed software-HIL only
   - this task is not `FR8`
   - `.tflite` and real-board validation remain outside current evidence

## Review No-Go Triggers

Review should be treated as `BLOCK` if any of the following happen:

1. the worker runs new experiments or creates a new run root
2. the worker modifies source, tests, or source-tree config
3. the worker touches theory-only materials
4. the worker silently upgrades T62 into completed formal comparator evidence
5. the worker recommends multiple simultaneous next tasks instead of one bounded next step

## Captain Notes

`T62` should be treated as the closure of the T61 provenance-repair loop unless contradictory repository evidence is found.

This task is not permission to start `FR8`. It is only permission to decide, in a bounded and reviewable way, whether a future FR8 task should exist at all and what its smallest honest scope would be.
