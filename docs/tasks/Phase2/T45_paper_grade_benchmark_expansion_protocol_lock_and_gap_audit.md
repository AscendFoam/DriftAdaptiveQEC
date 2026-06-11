# T45: Paper-grade benchmark expansion protocol lock and gap audit

## Status

- Proposed by Captain on `2026-05-19`
- Current phase: `Phase 2: Controlled Development`
- Decision state: `Go`
- Task type: docs-only planning gate

## Why This Task Exists

`T44` makes the current paper boundary honest: the frozen-set P4 result is real, but it is still not broad enough for a strong submission package.

This task exists to decide whether the paper should:

- stay strictly frozen-set and bounded, or
- admit a controlled benchmark expansion lane with an explicit protocol

It is not a benchmark execution task. It is a protocol-lock and gap-audit task.

## Goal

Produce a benchmark-expansion protocol note that answers, in writing:

1. what benchmark breadth is still missing
2. which expansion ideas are worth adopting
3. which expansion ideas should remain deferred
4. which expansion ideas should be rejected for the current mainline
5. whether the paper can remain frozen-set only without silently weakening the submission story

## Allowed Files

Worker may modify only:

- `docs/tasks/Phase2/T45_paper_grade_benchmark_expansion_protocol_lock_and_gap_audit.md`
- `docs/protocols/benchmark/paper_benchmark_expansion_protocol.md`
- `docs/review/T45_review.md`
- `docs/for_human/T45_explanation.md`

## Required Inputs

Read at minimum:

- `docs/04_task_board.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/protocols/benchmark/P4_benchmark_formal_protocol.md`
- `docs/paper_materials/paper_claim_evidence_ledger.md`
- `docs/paper_materials/paper_reviewer_risk_audit.md`
- `docs/reality_recovery/00_freeze_snapshot.md`
- `docs/reality_recovery/01_claim_evidence_table.md`
- `docs/reference/延伸改进思路.md`
- `docs/deep_research_reports/进一步的深度研究结果.md`

## Recovery Boundary

This task must stay docs-only.

It may classify candidate benchmark-expansion ideas such as:

- extra drift families
- soft-information or correlation-aware comparator lanes
- calibration / statcalib comparator inclusion rules
- CI-driven stopping or broader repeat policy
- latency / commit / fallback / saturation metric requirements

But it must not:

- run benchmark
- run training
- run `.tflite`
- call hardware
- modify benchmark code or config
- silently expand the frozen ranked set
- claim that expansion evidence already exists

## Expected Output

Create `docs/protocols/benchmark/paper_benchmark_expansion_protocol.md` with:

1. current benchmark boundary
2. candidate expansion items
3. adopted / deferred / rejected classification
4. what metrics and evidence would be required for a future expansion task
5. a go / no-go recommendation for widening the paper benchmark story
6. explicit non-claims

Create `docs/review/T45_review.md` with:

1. scope confirmation
2. adopted / deferred / rejected clarity
3. whether the protocol stays separate from execution
4. recommended next task, if any

Create `docs/for_human/T45_explanation.md` with a short human-facing summary.

## Verification

Required verification is documentation-only:

1. confirm no source, config, `runs/`, or `artifacts` files were modified
2. confirm no benchmark execution was started
3. confirm the protocol keeps frozen-set evidence separate from any future expansion lane
4. confirm `reference/延伸改进思路.md` is treated as reference-only, not as current mainline truth

## Captain Notes

T45 should answer a simple but important question:

- can the paper stay frozen-set only and still be honest enough to submit, or do we need a bounded benchmark expansion lane before claiming stronger method value?

## Verification Record

- Worker completed T45 on `2026-05-19`.
- Required inputs read and cross-checked:
  - `docs/04_task_board.md`
  - `docs/07_handoff.md`
  - `docs/08_risks_and_open_questions.md`
  - `docs/protocols/benchmark/P4_benchmark_formal_protocol.md`
  - `docs/paper_materials/paper_claim_evidence_ledger.md`
  - `docs/paper_materials/paper_reviewer_risk_audit.md`
  - `docs/reality_recovery/00_freeze_snapshot.md`
  - `docs/reality_recovery/01_claim_evidence_table.md`
  - `docs/reference/延伸改进思路.md`
  - `docs/deep_research_reports/进一步的深度研究结果.md`
- Produced outputs:
  - `docs/protocols/benchmark/paper_benchmark_expansion_protocol.md`
  - `docs/review/T45_review.md`
  - `docs/for_human/T45_explanation.md`
- Protocol decisions captured:
  1. current frozen-set software benchmark remains the anchor evidence and is not rewritten
  2. paper-grade broadening, if pursued, must use a separate bounded expansion lane
  3. `statcalib` comparator inclusion is recommended only as a separately labeled future lane
  4. extra drift-family ideas were classified into adopted / deferred / rejected buckets rather than treated as current truth
  5. deployment-boundary items (`.tflite`, `real_board`) remain outside this benchmark-expansion protocol
- Verification checks completed:
  1. no source, config, `runs/`, or `artifacts` files were modified
  2. no benchmark execution, training, `.tflite`, or hardware command was started
  3. the protocol explicitly keeps frozen-set evidence separate from any future expansion lane
  4. `docs/reference/延伸改进思路.md` and the deep-research report are treated as reference-only inputs, not as current mainline truth
- Scope discipline:
  - no benchmark code or config edits
  - no silent expansion of the frozen ranked set
  - no task-board or handoff status change
