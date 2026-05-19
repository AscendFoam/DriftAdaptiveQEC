# T44: Research Reality Recovery Mode setup and evidence-gap ledger

## Status

- Created by Captain on `2026-05-18`
- Current phase: `Phase 2: Controlled Development`
- Decision state: `Go`
- Task type: bounded docs-only recovery task

## Verification Record

Worker completed on `2026-05-19`.

### Output shape check

1. `docs/reality_recovery/00_freeze_snapshot.md` — exists; contains current posture, verified items, unverified items, must-not-claim list, mainline/booster/extension classification, roadmap sufficiency judgment.
2. `docs/reality_recovery/01_claim_evidence_table.md` — exists; classifies 15 claims (C1-C11, RRC12-RRC15) as supported/partial/blocked with concrete evidence paths and risk cross-references.
3. `docs/reality_recovery/02_code_truth_audit.md` — exists; audits 7 code-doc consistency areas; identifies `board_backend.py` stale label; confirms zero hidden marker comments.
4. `docs/reality_recovery/03_experiment_reproducibility_audit.md` — exists; documents 4 reproducible paths, 3 partially reproducible areas, and 4 non-reproducible areas with specific commands and evidence.
5. `docs/reality_recovery/04_figure_and_result_ledger.md` — exists; catalogs 16 figures/tables: 5 ready, 5 partial, 3 missing, 3 blocked.
6. `docs/reality_recovery/05_paper_claim_risk_table.md` — exists; maps all claim areas to risks, T44 closure assessment, task coverage assessment, mainline/booster/extension classification.
7. `docs/reality_recovery/06_human_brief.md` — exists; YELLOW project state; says paper writing cannot continue; recommends T45-T47.
8. `docs/review/T44_review.md` — exists; verdict PASS; blocking issues none.
9. `docs/for_human/T44_explanation.md` — exists.

### Boundary check

1. Recovery baseline is explicit about verified and unverified items: yes.
2. Every claim tagged supported/partial/blocked: yes (5 supported, 1 partial, 9 blocked).
3. Human brief does not say project is ready to resume paper expansion: yes (explicitly says "No").
4. No code, config, runs, or artifacts changes introduced: yes (verified by file scope check).

### Additional recovery judgments

1. T44 alone cannot raise project to strong-submission standard: explicitly stated in 00, 05, 06.
2. Currently visible pending tasks not sufficient by themselves: explicitly stated in 00, 01 (RRC13).
3. Missing items classified as:
   - Mainline paper-readiness blockers: benchmark broadening (T45), multi-seed mechanism (T46), ablation result-pack (T47)
   - Strong-quality boosters: true `.tflite` runtime (T48), real-board smoke (T49), training reproducibility (T50)
   - Future extension lanes: items from 延伸改进思路.md

### Code truth audit findings

- `board_backend.py`: docstring says "Placeholder" but code is 308-line structurally complete implementation (stale label).
- No TODO/FIXME/HACK/PLACEHOLDER marker comments in `cnn_fpga/` or `physics/`.
- All other code-doc consistency areas: consistent.

## Why This Task Exists

`T43` was accepted as `PASS`, but the user explicitly requested that the project enter `Research Reality Recovery Mode` because paper drafting has moved ahead of some preparation and evidence repair.

This task exists to freeze the current truth/evidence boundary before any further prose expansion. It is not a prose task and it is not an experiment task.

## Goal

Produce a recovery baseline that makes the next stage about evidence repair, not about narrative expansion.

The recovery baseline should answer, in writing:

- what is actually verified
- what is only partially supported
- what remains blocked
- what material is still missing for paper-quality figures, tables, and claims
- whether the current visible roadmap is sufficient for a strong paper package
- which missing items are mainline blockers versus later extension research

## Allowed Files

Worker may modify only:

- `docs/tasks/Phase2/T44_research_reality_recovery_mode_setup_and_evidence_gap_ledger.md`
- `docs/reality_recovery/00_freeze_snapshot.md`
- `docs/reality_recovery/01_claim_evidence_table.md`
- `docs/reality_recovery/02_code_truth_audit.md`
- `docs/reality_recovery/03_experiment_reproducibility_audit.md`
- `docs/reality_recovery/04_figure_and_result_ledger.md`
- `docs/reality_recovery/05_paper_claim_risk_table.md`
- `docs/reality_recovery/06_human_brief.md`
- `docs/review/T44_review.md`
- `docs/for_human/T44_explanation.md`

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
- `docs/06_repo_noise_governance.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/paper_claim_evidence_ledger.md`
- `docs/paper_draft_skeleton.md`
- `docs/paper_method_positioning_calibration.md`
- `docs/paper_reviewer_risk_audit.md`
- `docs/paper_background_related_work_draft.md`
- `docs/review/T43_review.md`
- `docs/reference/AI_coding_workflow.md`
- `docs/reference/科研纠偏意见.md`
- `relative_papers/Fast and accurate AI-based pre-decoders for surface codes.md`

## Recovery Mode Rules

- Freeze claim/evidence status before writing any new prose.
- Do not run experiments, training, `.tflite`, hardware, or cleanup commands.
- Do not modify source code, configs, tests, benchmark protocols, `runs/`, or `artifacts/`.
- Do not upgrade mock / stub / placeholder / smoke / readiness evidence into stronger claims.
- Use `supported / partial / blocked` language consistently.
- Treat historical artifacts as frozen evidence, not as something to rewrite.

## Required Output Shape

### A. `docs/reality_recovery/00_freeze_snapshot.md`

Capture the current project posture in one place:

- current phase / decision state
- current unique task
- what has been verified
- what remains unverified
- what must not be claimed

### B. `docs/reality_recovery/01_claim_evidence_table.md`

Create a claim-by-claim ledger that classifies each claim as:

- `supported`
- `partial`
- `blocked`

Include the concrete evidence path and any missing link.

### C. `docs/reality_recovery/02_code_truth_audit.md`

Audit code-path truth versus document truth:

- where the code really does what the docs say
- where the docs are ahead of the code
- where a placeholder or fallback still exists
- where a historical result is being reused correctly

### D. `docs/reality_recovery/03_experiment_reproducibility_audit.md`

Document what is reproducible now and what is not:

- clean environment facts
- seed / run / config facts
- benchmark / smoke / runtime boundaries
- what would still be needed for stronger reproducibility claims

### E. `docs/reality_recovery/04_figure_and_result_ledger.md`

Ledger every figure/table/result intended for the paper:

- source script or generation path
- data or log path
- seed or scenario linkage
- status: ready / partial / missing / blocked

### F. `docs/reality_recovery/05_paper_claim_risk_table.md`

Map paper claims to risks:

- claim
- supporting evidence
- residual risk
- whether the claim is blocked, partial, or supported

### G. `docs/reality_recovery/06_human_brief.md`

Write a short human-facing brief that answers:

- what was verified
- what is still unverified
- what looks suspicious
- can paper writing continue right now
- what human decision is needed

### Additional required recovery judgment

The recovery baseline must also state explicitly:

1. whether `T44` alone can raise the project to strong-submission standard
2. whether the currently visible pending tasks are enough by themselves
3. which missing items belong to:
   - mainline paper-readiness blockers
   - strong-quality boosters
   - future extension lanes

## Suggested Working Order

1. Freeze the top-level posture first in `00_freeze_snapshot.md`.
2. Build the claim/evidence ledger before any deeper audits.
3. Audit code truth and reproducibility next so later tables can reuse the same boundary language.
4. Finish with the figure/result ledger and paper-claim risk table.
5. Write the human brief last, after the other five recovery docs are stable.

## Forbidden Scope

Do not:

- modify `docs/02_experiment_plan.md`
- modify source code, configs, tests, benchmark protocol, `runs/`, or `artifacts/`
- run benchmark, training, `.tflite`, hardware, or cleanup commands
- write Abstract, Results, Conclusion, or full-manuscript prose
- silently upgrade evidence levels
- treat recovery documentation as evidence completion

## Required Verification

Verification is wording-and-boundary based:

1. the recovery baseline is explicit about what is verified and unverified
2. every claim is tagged `supported`, `partial`, or `blocked`
3. the human brief does not say the project is ready to resume paper expansion unless evidence actually supports that
4. no code, config, `runs/`, or `artifacts` changes are introduced

## Docs To Update

- `docs/tasks/Phase2/T44_research_reality_recovery_mode_setup_and_evidence_gap_ledger.md`
- `docs/reality_recovery/00_freeze_snapshot.md`
- `docs/reality_recovery/01_claim_evidence_table.md`
- `docs/reality_recovery/02_code_truth_audit.md`
- `docs/reality_recovery/03_experiment_reproducibility_audit.md`
- `docs/reality_recovery/04_figure_and_result_ledger.md`
- `docs/reality_recovery/05_paper_claim_risk_table.md`
- `docs/reality_recovery/06_human_brief.md`
- `docs/review/T44_review.md`
- `docs/for_human/T44_explanation.md`

## Reviewer Type

Adversarial.

Focus areas:

- evidence freeze discipline
- truth-vs-doc consistency
- reproducibility boundary honesty
- avoiding premature paper-expansion claims

## Captain Notes

This task is the first explicit Recovery Mode task after T43. It is meant to make the rest of Phase 2 evidence-led again.

The recovery output should not stop at "what is true now."  
It should also answer the planning question the user now cares about:

- is the current T44-era roadmap enough for the target paper?
- if not, what kinds of bounded next tasks are still missing?
- which reference ideas should remain extension-only for now?
