# T42: Paper Background / Related Work scaffold and method-positioning calibration

## Status

- Created by Captain on `2026-05-17`
- Current phase: `Phase 2: Controlled Development`
- Decision state: `Go`
- Task type: bounded docs-only paper-structure task

## Why This Task Exists

`T41` closed Milestone 2K with verdict `Allow`, but also made two explicit gate decisions:

1. Background / Related Work must be added before prose expansion begins.
2. Title and method positioning need calibration so the paper does not read as either:
   - an overclaimed method paper, or
   - an unnecessarily narrow recovery report.

The next controlled step is therefore not full paper drafting. It is to extend the existing manuscript scaffold with a bounded Background / Related Work section and to produce a method-positioning calibration note that locks the safe framing before prose expansion starts.

## Goal

Produce:

1. an updated `docs/paper_draft_skeleton.md` that includes a bounded Background / Related Work section,
2. a method-positioning calibration note that compares conservative vs method-forward framing and recommends the safe choice,
3. calibrated title candidates and introduction contribution bullets that remain aligned with the T34 ledger and T41 milestone review.

## Allowed Files

Worker may modify only:

- `docs/tasks/Phase2/T42_paper_background_related_work_and_positioning.md`
- `docs/paper_draft_skeleton.md`
- `docs/paper_method_positioning_calibration.md`
- `docs/review/T42_review.md`
- `docs/for_human/T42_explanation.md`

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
- `docs/review/T41_review.md`
- `docs/review/Milestone2K_review.md`
- `docs/CNN_FPGA_GKP_工程化实验方案.md`
- `docs/CNN_FPGA_GKP_阶段结论.md`
- `docs/CNN_FPGA_GKP_后续仿真与工程补强计划.md`
- `docs/CNN_FPGA_GKP_paper_inspired分支实验设计草案.md`

## Required Output Shape

### A. `docs/paper_draft_skeleton.md`

Must be updated to include:

1. one explicit `Background / Related Work` major section,
2. intended subsection headings,
3. which claim IDs / figure IDs / table IDs it may cite where applicable,
4. what must stay out because it depends on blocked claims,
5. calibrated title candidates and introduction contribution bullets if they are updated.

The section should create structural space for at least:

- GKP QEC problem framing
- fast-loop / slow-loop separation
- residual / teacher-guided decoder positioning
- benchmark / deployment / hardware evidence boundary context

### B. `docs/paper_method_positioning_calibration.md`

Must include at least:

1. conservative title framing option
2. method-forward title framing option
3. recommended safe framing and why
4. contribution-bullet calibration against `C1`-`C11`
5. phrases that remain forbidden because they would upgrade blocked claims

### C. `docs/review/T42_review.md`

Adversarial review output covering:

1. whether Background / Related Work was added without upgrading evidence
2. whether title / contribution calibration stays inside Milestone2K limits
3. whether any blocked claim was silently promoted

## Required Boundary Rules

T42 must preserve all current boundaries from:

- `docs/paper_claim_evidence_ledger.md`
- `docs/paper_reviewer_risk_audit.md`
- `docs/review/Milestone2K_review.md`

In particular, do not silently upgrade:

- mock-backed software HIL into real-board validation
- `.tflite` entrypoints or stub path into true runtime validation
- one clean CPU-only smoke into full reproducibility or portability
- frozen-set formal software revalidation into paper-grade expanded benchmark
- statcalib interface contract into integrated comparator evidence
- single-seed diagnosis into causal proof

## Forbidden Scope

Do not:

- modify `docs/02_experiment_plan.md`
- modify source code, configs, tests, benchmark protocol, `runs/`, or `artifacts`
- run benchmark, training, `.tflite`, hardware, or cleanup commands
- write full paper prose paragraphs beyond bounded scaffold bullets/notes
- rewrite stage-conclusion docs to claim stronger evidence
- silently add new paper claims not grounded in the T34 ledger

## Required Verification

Verification is structure-and-boundary based:

1. the new Background / Related Work section exists and is structurally integrated into the skeleton
2. updated title candidates and contribution bullets remain consistent with `C1`-`C11`
3. blocked claims remain blocked
4. the calibration note clearly distinguishes safe vs unsafe framing
5. no code, config, `runs/`, or `artifacts` changes are introduced

## Docs To Update

- `docs/tasks/Phase2/T42_paper_background_related_work_and_positioning.md`
- `docs/paper_draft_skeleton.md`
- `docs/paper_method_positioning_calibration.md`
- `docs/review/T42_review.md`
- `docs/for_human/T42_explanation.md`

## Reviewer Type

Adversarial review.

Focus areas:

- Background / Related Work is present and useful rather than generic filler
- method-forward framing does not drift beyond evidence
- contribution bullets stay aligned with the ledger
- no hidden switch from scaffolding to full prose drafting

## Captain Notes

This task is still docs-only. It is not the full drafting task. It exists to make sure that when prose expansion starts later, the paper has the right structural home for prior work and the right calibrated framing for the current evidence level.
