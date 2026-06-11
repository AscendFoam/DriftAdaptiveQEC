# T43: Paper Background / Related Work bounded prose draft

## Status

- Created by Captain on `2026-05-17`
- Current phase: `Phase 2: Controlled Development`
- Decision state: `Go`
- Task type: bounded docs-only prose drafting task

## Why This Task Exists

`T42` completed two prerequisite steps:

1. it added the Background / Related Work scaffold to the manuscript structure, and
2. it calibrated the safe paper framing to a method-forward title direction with evidence-bounded body text.

The next controlled step is not full-manuscript drafting. It is to draft only the Background / Related Work prose so the paper has a real narrative foundation before any wider prose expansion begins.

## Goal

Produce a bounded prose draft for the Background / Related Work section only, consistent with:

- `docs/paper_materials/paper_draft_skeleton.md`
- `docs/paper_materials/paper_method_positioning_calibration.md`
- `docs/paper_materials/paper_claim_evidence_ledger.md`
- `docs/paper_materials/paper_reviewer_risk_audit.md`
- `docs/review/Milestone2K_review.md`
- `docs/review/T42_review.md`

## Allowed Files

Worker may modify only:

- `docs/tasks/Phase2/T43_paper_background_related_work_prose_draft.md`
- `docs/paper_materials/paper_background_related_work_draft.md`
- `docs/review/T43_review.md`
- `docs/for_human/T43_explanation.md`

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
- `docs/paper_materials/paper_claim_evidence_ledger.md`
- `docs/paper_materials/paper_draft_skeleton.md`
- `docs/paper_materials/paper_method_positioning_calibration.md`
- `docs/paper_materials/paper_reviewer_risk_audit.md`
- `docs/review/T34_review.md`
- `docs/review/T35_review.md`
- `docs/review/T41_review.md`
- `docs/review/T42_review.md`
- `docs/review/Milestone2K_review.md`

## Captain Framing Lock

For this task, the working framing is locked as:

- title direction: method-forward
- body text: evidence-bounded

Concretely, the Worker should draft under the assumption that option 3 remains the current working title direction:

`A Dual-Loop Teacher-Guided Residual Decoding Framework for Real-Time GKP Error Correction`

This is a drafting lock for T43 only. It is not an evidence upgrade, and later human override remains allowed.

## Required Output Shape

### A. `docs/paper_materials/paper_background_related_work_draft.md`

Must contain bounded prose for Background / Related Work only.

Recommended structure:

1. GKP QEC problem framing
2. Fast-loop / slow-loop time-scale separation
3. Prior CNN-assisted / ML-assisted QEC decoding work
4. Classical adaptive estimators for drift tracking
5. Residual / teacher-guided correction positioning
6. Optional short neutral evidence-boundary paragraph only if it reads like survey material, not self-justification

Requirements:

- write real prose paragraphs, not just bullets
- keep subsection 6 short and neutral; if it feels self-serving, omit it and leave that material for Limitations in a later task
- keep citations, claim references, and comparison language consistent with the ledger and risk audit

### B. `docs/review/T43_review.md`

Adversarial review output covering at least:

1. whether the draft stays inside the evidence boundary
2. whether blocked claims remain blocked
3. whether subsection 6 stays neutral or should be removed
4. whether the method-forward framing drifts into novelty overclaim

## Required Boundary Rules

T43 must preserve all current boundaries from:

- `docs/paper_materials/paper_claim_evidence_ledger.md`
- `docs/paper_materials/paper_reviewer_risk_audit.md`
- `docs/review/Milestone2K_review.md`
- `docs/review/T42_review.md`

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
- draft Abstract, Results, Conclusion, or full-manuscript sections
- rewrite stage-conclusion docs to claim stronger evidence
- silently add new paper claims not grounded in the T34 ledger

## Required Verification

Verification is wording-and-boundary based:

1. only Background / Related Work prose is drafted
2. the draft stays consistent with the T42 skeleton and framing lock
3. blocked claims remain blocked
4. subsection 6, if kept, reads as neutral survey material rather than self-justifying novelty
5. no code, config, `runs/`, or `artifacts` changes are introduced

## Docs To Update

- `docs/tasks/Phase2/T43_paper_background_related_work_prose_draft.md`
- `docs/paper_materials/paper_background_related_work_draft.md`
- `docs/review/T43_review.md`
- `docs/for_human/T43_explanation.md`

## Reviewer Type

Adversarial review.

Focus areas:

- prose quality without evidence drift
- neutral handling of evidence-boundary language
- method-forward framing without novelty inflation
- strict containment to Background / Related Work only

## Captain Notes

This task is still docs-only. It is not full-paper drafting. It exists to convert the T42 scaffold into bounded prose while the evidence boundary is still fresh and explicit.

## Verification Record

- Worker completed T43 on `2026-05-18`.
- All required inputs read and cross-checked: README.md, docs/legacy_context/plan_variants_2026-06-11/02_experiment_plan_simplified.md, docs/04_task_board.md, docs/07_handoff.md, docs/00_project_snapshot.md, docs/01_legacy_audit.md, docs/03_hil_p4_boundary_audit.md, docs/05_decision_log.md, docs/08_risks_and_open_questions.md, docs/paper_materials/paper_claim_evidence_ledger.md, docs/paper_materials/paper_draft_skeleton.md, docs/paper_materials/paper_method_positioning_calibration.md, docs/paper_materials/paper_reviewer_risk_audit.md, docs/review/T34_review.md, docs/review/T35_review.md, docs/review/T41_review.md, docs/review/T42_review.md, docs/review/Milestone2K_review.md.
- `docs/paper_materials/paper_background_related_work_draft.md` produced: bounded prose for Background / Related Work with 6 subsections (GKP QEC problem framing, dual-loop time-scale separation, ML-assisted QEC decoding, classical adaptive estimators, teacher-guided residual correction positioning, evidence boundaries in quantum system validation).
- `docs/review/T43_review.md` produced with verdict = `PASS`, no blocking issues, 4 non-blocking issues (subsection 6 neutrality, placeholder citations, internal cross-references, inline claim reference formatting).
- `docs/for_human/T43_explanation.md` produced with Chinese human-facing explanation.
- Blocked claims (C6, C7, C8, C10, C11) remain blocked in all drafted prose.
- All supported claims cited with correct wording: C2, C3 (frozen-set formal software revalidation), C5 (one clean CPU-only smoke), C9 (statcalib interface contract, not cited in Background/Related Work as it is scoped to Method section).
- Stable conclusion 9.1 item 7 ("offline training improvement ≠ formal HIL improvement") referenced correctly in subsection 5.
- No forbidden phrases from `docs/paper_materials/paper_method_positioning_calibration.md` Section 6 appear in the draft.
- Subsection 6 kept short and neutral; can be folded into Limitations if needed during later review.
- Method-forward framing lock maintained throughout: the draft positions teacher-guided residual-b as a method contribution with frozen-set evidence support, without claiming broad superiority or hardware validation.
- No code, config, `runs/`, or `artifacts` changes were introduced.
- No forbidden scope violations detected.
- Drafting scope correctly bounded to Background / Related Work only: no Abstract, Introduction, Method, Experiment, Results, Conclusion, or Appendix prose was written.
