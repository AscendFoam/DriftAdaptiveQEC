# Review: T34

Verdict: PASS

## Blocking Issues

None.

## Non-Blocking Issues

### N1 C9 evidence paths are indirect

`C9` claims `statcalib` has "a separate interface contract and focused tests" with status `supported`, but cites only `docs/review/T26_review.md` and `docs/review/T30_review.md`. The direct evidence — `cnn_fpga/decoder/statcalib.py` and `tests/test_statcalib_interface.py` — is not listed in the evidence paths column. The reviews do reference these files, so the claim is accurate, but a future paper drafter tracing the ledger would need an extra hop. Not blocking because the claim is correct and the reviews are concrete.

### N2 No explicit claim for float/int8 quantization gap

The experiment plan (`docs/02_experiment_plan_simplified.md` Section 4.2–4.3) documents that float/int8 R² degradation is < 1%, and this is listed as stable conclusion 9.1 item 6. The ledger does not include a claim row for this. The omission is defensible because those results predate the current Phase 2 recovery cycle and were not revalidated under the T24 frozen-set protocol. However, future paper drafting will likely need this claim, and it would be useful to have it in the ledger with an explicit caveat about pre-recovery historical evidence.

### N3 No explicit claim for ablation conclusions

Stable conclusions 9.1 items 8–10 (`histogram delta` is key channel, `teacher params` encoding is the core issue, `Gated v5` direction correct) are not represented in the claim ledger. These come from historical teacher-representation runs that also predate Phase 2 revalidation. Same justification as N2, but a future paper drafter will need to decide whether to cite these as historical evidence or revalidate them. Having them in the ledger with `partial` status and a note about pre-recovery evidence would reduce the risk of silent over/under-claiming later.

### N4 Worker pre-review overwritten by adversarial review

The worker's pre-review content in this file has been replaced by this adversarial review. The worker's verification notes are preserved in the task package's Verification Record section, so no information is lost. This follows the established pattern from T36 and other tasks.

## Missing Tests

Not applicable — T34 is a docs-only task with no code or configuration changes. Verification is document-structure and evidence-traceability based, and the worker correctly performed it.

## Suspicious Implementation Details

None found. Specific checks:

1. **No pseudo-implementation**: All 11 claims, 3 figures, and 5 tables are document artifacts with no code.
2. **No mock/stub/hardcode**: No code was changed.
3. **No overclaiming**: Every `supported` claim cites concrete existing evidence paths (all verified to exist on disk). Every `blocked` claim cites a specific risk ID (all verified in `docs/08_risks_and_open_questions.md`). Every `partial` claim includes explicit boundary wording.
4. **Hard boundaries preserved**:
   - C1 vs C8: mock-backed software HIL separated from real-board ✅
   - C7: true `.tflite` runtime vs stub/fallback ✅
   - C2/C3/C11: frozen-set revalidation vs paper-grade expanded benchmark ✅
   - C5 vs C6: clean CPU-only one-run smoke vs full reproducibility ✅
   - C9 vs C10: statcalib interface contract vs integrated comparator evidence ✅
5. **No forbidden scope violations**: `git diff --name-only HEAD -- cnn_fpga benchmark physics tests runs artifacts requirements-recovery.txt docs/02_experiment_plan.md docs/04_task_board.md docs/07_handoff.md` returned empty.
6. **All evidence paths verified**: All 23 doc paths and all 9 run/artifact paths cited in the ledger exist on disk.
7. **Risk IDs verified**: R5, R8, R9, R10, R11, R12, R13, R14, R24 all exist in `docs/08_risks_and_open_questions.md`.

## Recommended Next Action

1. Captain should accept T34 as `PASS`.
2. The ledger should be treated as a living document — when T35 (paper draft skeleton) or any subsequent evidence-upgrading task completes, the corresponding claim rows should be updated.
3. Before T35 starts, the Captain should decide whether N2 (float/int8 gap) and N3 (ablation conclusions) should be added to the ledger as `partial` claims with pre-recovery caveats, or deferred to a separate evidence-revalidation task.
4. `T35: Paper draft skeleton and reviewer-risk audit` is the natural next task in Milestone 2K, subject to Captain's priority judgment.
