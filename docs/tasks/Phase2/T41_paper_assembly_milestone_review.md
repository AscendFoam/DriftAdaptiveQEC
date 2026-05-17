# T41: Milestone 2K paper-assembly gate review and next-phase decision

## Status

- Created by Captain on `2026-05-17`
- Current phase: `Phase 2: Controlled Development`
- Decision state: `Go`
- Task type: bounded read-only milestone review

## Why This Task Exists

`T34` and `T35` have now completed the paper-assembly readiness lane:

1. `T34` established the claim/evidence ledger and figure-table outline.
2. `T35` established the manuscript skeleton and reviewer-risk audit.

Before any prose expansion, milestone switch, or renewed push toward blocked deployment/hardware lanes, the project needs one explicit gate review to decide:

- whether Milestone 2K may close,
- what the minimum safe paper positioning is,
- whether a Background / Related Work scaffold must be added before drafting,
- and what the next unique task should be.

## Goal

Produce a read-only milestone review that:

1. reviews `T34 + T35` together as Milestone 2K,
2. assigns a gate verdict: `Allow` / `Conditional` / `Block`,
3. states the minimum safe paper positioning supported by current evidence,
4. decides whether future prose expansion must first add a Background / Related Work scaffold,
5. recommends the next unique task, but does not execute it.

## Allowed Files

Worker may modify only:

- `docs/tasks/Phase2/T41_paper_assembly_milestone_review.md`
- `docs/review/Milestone2K_review.md`
- `docs/for_human/T41_explanation.md`

## Required Inputs

Read at minimum:

- `README.md`
- `AGENTS.md`
- `docs/00_project_snapshot.md`
- `docs/01_legacy_audit.md`
- `docs/02_experiment_plan.md`
- `docs/03_hil_p4_boundary_audit.md`
- `docs/04_task_board.md`
- `docs/05_decision_log.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/paper_claim_evidence_ledger.md`
- `docs/paper_draft_skeleton.md`
- `docs/paper_reviewer_risk_audit.md`
- `docs/review/T34_review.md`
- `docs/review/T35_review.md`
- `docs/review/Milestone2I_review.md`
- `docs/TFLite_runtime_bootstrap.md`
- `docs/training_chain_cpu_cleanenv_train_smoke.md`
- `docs/real_board_hil_readiness.md`
- `docs/real_board_smoke_execution_plan.md`

## Required Output Shape

### A. `docs/review/Milestone2K_review.md`

Must include at least:

1. review metadata
2. milestone scope reviewed (`T34`, `T35`)
3. verdict: `Allow` / `Conditional` / `Block`
4. whether Milestone 2K may close
5. minimum safe paper positioning supported today
6. blocked claims that still prevent stronger positioning
7. decision on whether Background / Related Work must be added before prose expansion
8. recommended next unique task

### B. `docs/for_human/T41_explanation.md`

Short Chinese explanation for humans covering:

1. what Milestone 2K now proves,
2. what it still does not prove,
3. why the next step is a gate review rather than immediate drafting or deployment work.

## Required Boundary Rules

T41 is read-only. It must not silently upgrade any current paper claim or engineering boundary.

In particular, do not upgrade:

- mock-backed software HIL into real-board validation
- `.tflite` entrypoints or stub path into true runtime validation
- one clean CPU-only training smoke into full reproducibility or portability
- frozen-set formal software revalidation into paper-grade expanded benchmark
- statcalib interface contract into integrated comparator evidence
- single-seed trace diagnosis into causal proof

## Forbidden Scope

Do not:

- modify `docs/02_experiment_plan.md`
- modify source code, configs, tests, benchmark protocol, `runs/`, or `artifacts`
- run benchmark, training, `.tflite`, hardware, or cleanup commands
- rewrite stage-conclusion docs to claim stronger evidence
- start full paper prose drafting inside this task
- silently choose a new paper title/venue claim that depends on blocked evidence

## Required Verification

Verification is review-consistency based:

1. `T34` and `T35` outputs are cross-checked against the current claim/risk ledger
2. blocked claims remain blocked in the milestone review
3. the paper-positioning recommendation is consistent with `docs/paper_reviewer_risk_audit.md`
4. no code, config, `runs/`, or `artifacts` changes are introduced

## Docs To Update

- `docs/tasks/Phase2/T41_paper_assembly_milestone_review.md`
- `docs/review/Milestone2K_review.md`
- `docs/for_human/T41_explanation.md`

## Reviewer Type

Milestone review.

Focus areas:

- paper positioning does not drift beyond current evidence
- T35 non-blocking notes are handled as gate-level decisions, not as hidden claim upgrades
- the next unique task recommendation is concrete and bounded
- no deployment/runtime/hardware blocker is quietly bypassed

## Captain Notes

This task exists to decide what we are actually allowed to say next, and what we should do next. It is not the full paper-writing task, and it is not a backdoor to reopen experiments or deployment work.

## Verification Record

- Worker completed T41 on `2026-05-17`.
- All required inputs read and cross-checked.
- `docs/review/Milestone2K_review.md` produced with verdict = `Allow`.
- `docs/for_human/T41_explanation.md` produced with Chinese human-facing explanation.
- Blocked claims (C6, C7, C8, C10, C11) remain blocked in the milestone review.
- Paper positioning recommendation is consistent with `docs/paper_reviewer_risk_audit.md` "Minimum Safe Paper Positioning" and "Do-Not-Publish-As-Claimed List."
- Decision on Background / Related Work: yes, must be added before prose expansion.
- Recommended next unique task: T42 (Background / Related Work scaffold and method-positioning calibration).
- No code, config, `runs/`, or `artifacts` changes were introduced.
