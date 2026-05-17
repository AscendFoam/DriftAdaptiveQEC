# T34: Paper claim/evidence ledger and figure-table outline

## Status

- Created by Captain on `2026-05-17`
- Current phase: `Phase 2: Controlled Development`
- Decision state: `Go`
- Task type: bounded docs-only paper-assembly-readiness task

## Why This Task Exists

`T33` has completed the bounded tracked-cache cleanup lane. `T32` remains blocked by missing true `.tflite` runtime dependencies on the current machine, and `T37` remains blocked by hardware/bitstream readiness. That leaves `T34` as the next bounded task that can move the project forward without inventing new evidence.

The point of `T34` is not to draft the paper itself and not to upgrade any evidence level. The point is to build a disciplined claim/evidence ledger so later paper-writing work cannot silently overclaim mock-backed software HIL, stub `.tflite`, clean-environment smoke, or real-board readiness artifacts.

## Goal

Produce one docs-only ledger that answers:

1. which claims are currently supported by concrete evidence
2. which claims are only partially supported and require explicit caveats
3. which claims are currently blocked and must not appear as completed statements
4. which figures/tables can already be outlined from existing evidence, and which remain blocked

## Allowed Files

Worker may modify only:

- `docs/tasks/Phase2/T34_paper_claim_evidence_ledger.md`
- `docs/paper_claim_evidence_ledger.md`
- `docs/review/T34_review.md`
- `docs/for_human/T34_explanation.md`

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
- `docs/CNN_FPGA_GKP_工程化实验方案.md`
- `docs/CNN_FPGA_GKP_阶段结论.md`
- `docs/CNN_FPGA_GKP_后续仿真与工程补强计划.md`
- `docs/CNN_FPGA_GKP_paper_inspired分支实验设计草案.md`
- `docs/P4_benchmark_formal_protocol.md`
- `docs/training_chain_portable_dependency_lock_plan.md`
- `docs/training_chain_cpu_cleanenv_bootstrap.md`
- `docs/training_chain_cpu_cleanenv_train_smoke.md`
- `docs/TFLite_runtime_bootstrap.md`
- `docs/real_board_hil_readiness.md`
- `docs/real_board_smoke_execution_plan.md`
- relevant review files for `T24`, `T25`, `T27`, `T28`, `T29`, `T30`, `T31`, `T33`, `T38`, `T39`, `T40`

## Required Output Shape

`docs/paper_claim_evidence_ledger.md` must include at least these sections:

1. scope and non-claims
2. claim ledger
3. figure outline
4. table outline
5. blocked claims and blockers
6. wording guardrails for future paper drafting

The claim ledger should use stable IDs such as `C1`, `C2`, `F1`, `T1` so later drafting can cite them directly.

## Required Claim-Ledger Rules

For each claim row, include:

- claim ID
- short claim text
- current status: `supported`, `partial`, or `blocked`
- exact evidence paths
- explicit boundary wording
- linked open risk / blocker if not fully supported

At minimum, the ledger must explicitly distinguish:

- mock-backed software HIL vs real-board validation
- true `.tflite` runtime vs stub/fallback path
- frozen-set formal software revalidation vs paper-grade expanded benchmark
- clean CPU-only one-run training smoke vs full training reproducibility / portability
- statcalib interface contract vs integrated comparator evidence

## Figure/Table Outline Requirements

The figure/table outline must not invent results. For each planned figure/table:

- give it an ID
- state what evidence it would draw from
- mark it as `supported`, `partial`, or `blocked`
- state the blocker if it is not fully supported

At minimum, evaluate whether the current repo evidence can support:

1. benchmark ranking summary table
2. benchmark boundary / evidence-level table
3. mechanism-diagnosis figure for `seed=20260429`
4. training reproducibility boundary table
5. deployment/readiness boundary table

## Forbidden Scope

Do not:

- modify `docs/02_experiment_plan.md`
- modify source code, configs, tests, or benchmark protocol
- create or modify `runs/` or `artifacts`
- run benchmark, training, `.tflite`, hardware, or cleanup commands
- rewrite phase conclusion docs to claim new completed evidence
- turn readiness, placeholder, stub, or smoke evidence into stronger completion claims
- silently change benchmark semantics, baselines, ParamMapper meaning, or evidence boundaries

## Required Verification

Verification is document-structure and evidence-traceability based:

1. every supported or partial claim must cite concrete existing evidence paths
2. every blocked claim must cite a concrete blocker or open risk
3. the document must explicitly preserve hard boundaries around:
   - mock-backed software HIL
   - true `.tflite` runtime
   - real-board validation
   - training reproducibility
4. no code, config, `runs/`, or `artifacts` changes are introduced

## Docs To Update

- `docs/tasks/Phase2/T34_paper_claim_evidence_ledger.md`
- `docs/paper_claim_evidence_ledger.md`
- `docs/review/T34_review.md`
- `docs/for_human/T34_explanation.md`

## Reviewer Type

Adversarial review.

Focus areas:

- claim/evidence linkage is concrete rather than hand-wavy
- no placeholder, stub, mock, or readiness artifact is overclaimed
- blocked claims are honestly marked as blocked
- figure/table outline does not invent unavailable evidence

## Captain Notes

This task is intentionally documentation-only. Do not turn it into paper drafting, new experiments, code cleanup, or benchmark reruns. If a claim cannot be supported by a concrete evidence path already in the repo, mark it `blocked` instead of stretching the wording.

## Worker Output

- Pending.

## Verification Record

- Pending.
