# T35: Paper draft skeleton and reviewer-risk audit

## Status

- Created by Captain on `2026-05-17`
- Current phase: `Phase 2: Controlled Development`
- Decision state: `Go`
- Task type: bounded docs-only paper-assembly task

## Why This Task Exists

`T34` has already produced a bounded claim/evidence ledger and figure-table outline. The next controlled step is not new evidence generation. It is to turn that ledger into a paper draft skeleton that stays aligned with current evidence boundaries, and to audit the draft structure for likely reviewer objections before any prose expansion starts.

## Goal

Produce:

1. a section-level paper draft skeleton that maps each section to current supported/partial/blocked evidence
2. a reviewer-risk audit that anticipates likely objections, overclaim traps, and evidence-grade weaknesses in the current paper state

## Allowed Files

Worker may modify only:

- `docs/tasks/Phase2/T35_paper_draft_skeleton_and_reviewer_risk_audit.md`
- `docs/paper_materials/paper_draft_skeleton.md`
- `docs/paper_materials/paper_reviewer_risk_audit.md`
- `docs/review/T35_review.md`
- `docs/for_human/T35_explanation.md`

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
- `docs/legacy_context/reference_retired_2026-06-11/CNN_FPGA_GKP_工程化实验方案.md`
- `docs/CNN_FPGA_GKP_阶段结论.md`
- `docs/CNN_FPGA_GKP_后续仿真与工程补强计划.md`
- `docs/CNN_FPGA_GKP_paper_inspired分支实验设计草案.md`
- `docs/protocols/benchmark/P4_benchmark_formal_protocol.md`
- `docs/evidence_packs/mechanism_ablation/seed20260429_failure_diagnosis.md`
- `docs/evidence_packs/mechanism_ablation/seed20260429_trace_export_diagnosis.md`
- `docs/evidence_packs/training_reproducibility/training_chain_cpu_cleanenv_train_smoke.md`
- `docs/evidence_packs/deployment_boundary/TFLite_runtime_bootstrap.md`
- `docs/evidence_packs/deployment_boundary/real_board_hil_readiness.md`
- `docs/evidence_packs/deployment_boundary/real_board_smoke_execution_plan.md`
- relevant review files for `T24`, `T25`, `T30`, `T34`, `T38`, `T39`, `T40`

## Required Output Shape

### A. `docs/paper_materials/paper_draft_skeleton.md`

Must include at least:

1. title candidates
2. abstract skeleton
3. introduction skeleton
4. method/system skeleton
5. experiment/evidence skeleton
6. results skeleton
7. limitations/boundary skeleton
8. conclusion skeleton

For each section, list:

- intended subsection headings
- which ledger claims or figure/table IDs it may cite
- which claims are still blocked and must not be written as completed prose

### B. `docs/paper_materials/paper_reviewer_risk_audit.md`

Must include at least:

1. likely novelty challenge points
2. likely evidence-grade challenge points
3. likely overclaim wording traps
4. likely reproducibility/deployment challenge points
5. likely ablation/mechanism challenge points
6. concrete mitigation options:
   - wording-only mitigation
   - evidence-upgrade-needed mitigation

## Required Boundary Rules

T35 must preserve all current boundaries from `docs/paper_materials/paper_claim_evidence_ledger.md`.

In particular, do not silently upgrade:

- mock-backed software HIL into real-board validation
- `.tflite` entrypoints or stub path into true runtime validation
- clean CPU-only one-run smoke into full reproducibility or portability
- frozen-set formal software revalidation into paper-grade expanded benchmark
- statcalib interface contract into integrated comparator evidence

## Forbidden Scope

Do not:

- modify `docs/02_experiment_plan.md`
- modify source code, configs, tests, benchmark protocol, `runs/`, or `artifacts`
- run benchmark, training, `.tflite`, hardware, or cleanup commands
- rewrite stage-conclusion docs to claim stronger evidence
- reopen `T34` by silently changing the ledger instead of explicitly working from it

## Required Verification

Verification is document-structure and boundary-consistency based:

1. every section in the draft skeleton must point to specific claim IDs and figure/table IDs where applicable
2. the reviewer-risk audit must tie each major objection to a concrete current blocker, risk, or wording hazard
3. blocked claims must stay blocked in the skeleton
4. no code, config, `runs/`, or `artifacts` changes are introduced

## Docs To Update

- `docs/tasks/Phase2/T35_paper_draft_skeleton_and_reviewer_risk_audit.md`
- `docs/paper_materials/paper_draft_skeleton.md`
- `docs/paper_materials/paper_reviewer_risk_audit.md`
- `docs/review/T35_review.md`
- `docs/for_human/T35_explanation.md`

## Reviewer Type

Adversarial review.

Focus areas:

- draft skeleton does not overclaim blocked evidence
- reviewer-risk audit is concrete rather than generic
- current ledger boundaries are preserved section-by-section
- no hidden reopening of experiment scope or governance scope

## Captain Notes

This task is still docs-only. The worker is not writing the full paper and is not filling sections with persuasive prose. The worker is building a bounded scaffold and a risk audit so later writing cannot drift past the current evidence boundary.

## Worker Output

- Added `docs/paper_materials/paper_draft_skeleton.md` as a bounded manuscript scaffold.
- Added `docs/paper_materials/paper_reviewer_risk_audit.md` as an adversarial reviewer-risk and wording-trap audit.
- Added `docs/review/T35_review.md` with worker-side pre-review notes and residual-risk checks.
- Added `docs/for_human/T35_explanation.md` with a short Chinese explanation of scope and boundaries.
- Preserved T34 ledger boundaries explicitly; no attempt was made to upgrade blocked evidence or expand experiment scope.

## Verification Record

- Checked that every major section in `docs/paper_materials/paper_draft_skeleton.md` includes explicit claim IDs and figure/table IDs where applicable.
- Checked that `docs/paper_materials/paper_reviewer_risk_audit.md` ties objections to concrete claim states, blocker categories, or open-risk IDs such as `R5`, `R9`, `R10`, `R11`, `R12`, `R13`, `R14`, `R20`, and `R24`.
- Checked that blocked claims `C6`, `C7`, `C8`, `C10`, and `C11` remain blocked and are called out as non-completed evidence.
- Verified separately that no code, config, `runs/`, or `artifacts` paths were modified by this task.
