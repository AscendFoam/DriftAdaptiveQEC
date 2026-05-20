# Review: T45

Verdict: PASS

Reviewer type: adversarial

## Blocking Issues

None.

## Non-Blocking Issues

### N1 Worker self-review overwritten by adversarial review

The worker wrote a self-review into `docs/review/T45_review.md` (verdict: PASS). This adversarial review replaces it. The worker's self-review was honest in its scope confirmation and non-blocking classification, but a self-review cannot serve as the final review artifact.

Classification: `accepted` — standard project practice is for the reviewer to overwrite the worker's self-review.

### N2 `sinusoidal` rejection rationale could be stronger

Section 4 of `docs/paper_benchmark_expansion_protocol.md` rejects `sinusoidal` as a required new family, reasoning that `periodic_drift` already exists in the frozen set. However, `periodic_drift` may differ from `sinusoidal` in amplitude envelope, frequency content, or drift-law structure. The rejection is defensible at protocol-lock level but may invite reviewer questions if the paper later uses `periodic_drift` as a proxy without explicitly comparing against a pure sinusoidal baseline.

Classification: `accepted` — T45 is a protocol-lock task, not an execution task; the exact scenario-coverage argument belongs in a later expansion-lane task.

### N3 Exact drift parameter grid remains intentionally unlocked

The protocol adopts `random_walk` and `burst_reset` as future candidates but explicitly defers exact parameter grids. This is correct for a gap-audit task, but any future expansion task must lock these grids before execution begins, not during.

Classification: `accepted` — documented in protocol Section 5 rule 4 ("Any new scenario family must be predeclared before execution").

### N4 Worker explanation file naming convention mismatch

The task package lists `docs/for_human/T45_explanation.md` as an allowed file, and the worker wrote to that path. However, `CLAUDE.md` Section 4 prescribes `docs/for_human/<TaskID>_review_explanation.md` for reviewer explanations. The worker file at `T45_explanation.md` is the worker's own human-facing summary, which is fine. The reviewer explanation will be written separately as `T45_reviewer_explanation.md`.

Classification: `accepted` — no conflict; two separate files with distinct purposes.

## Missing Tests

Not applicable. T45 is a docs-only task. No code, config, benchmark, training, `.tflite`, or hardware changes were made. Verification is documentation-level only.

## Suspicious Implementation Details

None found. The following checks passed:

1. **Allowed files check**: `git diff --stat HEAD` shows only `docs/tasks/Phase2/T45_...md` was modified. Three new files (`docs/paper_benchmark_expansion_protocol.md`, `docs/review/T45_review.md`, `docs/for_human/T45_explanation.md`) are untracked additions. All four are listed in the task's `Allowed Files`. No source, config, `runs/`, `artifacts/`, benchmark code, or governance files (`04_task_board.md`, `07_handoff.md`) were touched.

2. **No benchmark execution**: The protocol document explicitly states it does not run benchmark, training, `.tflite`, or hardware. No run directories were created. No CLI invocation evidence exists in the diff.

3. **No evidence upgrade**: The frozen T24 set is described as "anchor evidence" that "remains unchanged." No claim is made that expansion evidence already exists.

4. **Reference-only discipline**: Section 9 (Explicit Non-Claims) item 2 explicitly states that `docs/reference/延伸改进思路.md` is not current mainline truth. Item 3 makes the same statement about the deep-research report.

5. **Cross-reference accuracy**: Verified that C2, C3, C11 exist in `docs/paper_claim_evidence_ledger.md` with statuses matching the protocol's claims (C2 supported, C3 supported, C11 blocked). Verified that E3 exists in `docs/paper_reviewer_risk_audit.md` and describes exactly what the protocol references.

6. **Adopted / deferred / rejected classification**: The candidate expansion ledger (Section 4) covers all expansion items mentioned in the required inputs — extra drift families, soft-information comparators, statcalib, CI-driven stopping, learned branches, `.tflite`/real_board mixing, latency/commit/saturation metrics, rollback/fallback metrics, training-seed separation. Classification logic is internally consistent.

7. **No over-engineering**: The document is a protocol specification with clear sections (boundary, decision frame, ledger, locked rules, gap audit, go/no-go, non-claims). It does not contain implementation code, configuration changes, or execution instructions beyond classification rules.

8. **Frozen-set separation**: Section 5 rule 1 ("The T24 frozen set remains unchanged and remains separately reported") and rule 2 ("Expanded results must be labeled as expansion lane") explicitly prevent silent redefinition.

## Scope Confirmation

1. Allowed files respected: all four modified/created files are in the task's `Allowed Files` list.
2. No source, config, `runs/`, `artifacts/`, benchmark code, or governance files were modified.
3. No benchmark, training, `.tflite`, or hardware execution was started.
4. The protocol keeps frozen-set evidence separate from any future expansion lane.
5. `docs/reference/延伸改进思路.md` and the deep-research report are treated as reference-only inputs, not current truth.
6. The worker did not mark the task as complete in `docs/04_task_board.md`.
7. The worker did not modify `docs/07_handoff.md` or `docs/08_risks_and_open_questions.md`.

## Recommended Next Action

1. Captain should accept T45 as `PASS` and update the task board, handoff, and risk register.
2. The recommended next task is `T46: Multi-seed mechanism/intervention plan and trace pack`, which would address the mechanism-evidence gap identified in Section 7.1 item 5 ("mechanism evidence is still not multi-seed closed").
3. If the project instead wants to pursue a stronger benchmark story first, a new bounded expansion task should be created that follows the locked rules in `docs/paper_benchmark_expansion_protocol.md` Section 5.
