# Review: T35

Verdict: PASS

## Blocking Issues

None.

## Non-Blocking Issues

### N1 Title candidates are unusually conservative

The four title candidates all center on "recovery," "revalidation," "boundary audit," and "evidence-bounded." While these are defensible as the safest framing given current evidence, they may be too narrow for the target venues listed in `docs/legacy_context/plan_variants_2026-06-11/02_experiment_plan_simplified.md` Section 10.4 (QCE, TQE, EPJ Quantum Technology). The experiment plan's recommended title — "A Dual-Loop Teacher-Guided Residual Decoding Framework for Real-Time GKP Error Correction" — is more method-forward while still being honest about the evidence state. The skeleton's titles would position the paper as a software-engineering-recovery paper rather than a quantum-methods paper, which may not match any of the listed venues.

This is not blocking because: (a) the titles are technically safe and do not overclaim, and (b) title selection is a Captain/human decision, not a Worker deliverable requirement. Flagging it because the risk-audit correctly identifies N1 ("this reads like a recovery report, not a novel method paper") but the skeleton's titles lean directly into that framing without offering a middle-ground option.

### N2 Skeleton omits Related Work / Background section

The task required at minimum: title candidates, abstract skeleton, introduction skeleton, method/system skeleton, experiment/evidence skeleton, results skeleton, limitations/boundary skeleton, conclusion skeleton. The skeleton delivers all eight. However, the experiment plan (Section 10.5) calls for a Background section covering GKP syndrome, fast/slow loop time scales, and `(K, b)` as runtime targets. This section is absent from the skeleton. Without it, later drafting has no scaffold for situating the work in the GKP QEC landscape — which is important for the novelty defense the risk audit itself identifies as a challenge (N1, N2).

Not blocking because the task's minimum required sections are met. But a later drafting task should add Background/Related Work before prose expansion starts.

### N3 Risk-audit section-by-section table uses generic hotspot labels

The "Section-by-Section Reviewer Hotspots" table uses high-level descriptions like "Evidence inflation," "Contribution inflation," "Benchmark-scope drift." These are useful but less actionable than the preceding tables (N1-N3, E1-E5, W1-W6, R1-R3, A1-A3) which tie each objection to specific claim IDs and evidence paths. The section-by-section table could have cross-referenced the preceding table IDs (e.g., "Abstract: watch for W1, W5, E1") for stronger traceability.

Not blocking because the preceding tables already provide the detailed mapping. The section-by-section table serves a different purpose (quick-reference checklist) and is adequate for that role.

### N4 Worker pre-review overwritten by adversarial review

Same pattern as T34/T36. Worker pre-review content replaced by this adversarial review. Verification notes preserved in the task package's Verification Record. No information lost.

## Missing Tests

Not applicable — T35 is a docs-only task with no code or configuration changes.

## Suspicious Implementation Details

None found. Specific checks:

1. **No pseudo-implementation**: All outputs are document artifacts with no code.
2. **No mock/stub/hardcode**: No code was changed.
3. **No overclaiming**: Each skeleton section explicitly lists blocked claims that must not appear as completed prose. The risk audit's "Do-Not-Publish-As-Claimed List" is unusually direct and useful.
4. **Hard boundaries preserved**: Cross-checked the skeleton against the T34 ledger:
   - C1/C8: mock-backed software HIL vs real-board ✅
   - C7: true `.tflite` vs stub ✅
   - C2/C3/C11: frozen-set vs paper-grade expanded benchmark ✅
   - C5/C6: clean CPU-only smoke vs full reproducibility ✅
   - C9/C10: statcalib interface vs integrated comparator ✅
5. **No forbidden scope violations**: `git diff --name-only HEAD -- cnn_fpga benchmark physics tests runs artifacts requirements-recovery.txt docs/02_experiment_plan.md docs/04_task_board.md docs/07_handoff.md` returned empty. The T34 ledger was not modified. No governance docs were touched.
6. **Cross-references verified**: The skeleton contains 46 C/F/T/R references, the risk audit contains 23. Both consistently use the same ID scheme established in T34.
7. **Risk IDs match**: R5, R9, R10, R11, R12, R13, R14, R20, R24 all verified as current in `docs/08_risks_and_open_questions.md`.

## Recommended Next Action

1. Captain should accept T35 as `PASS`.
2. Before prose expansion begins, the Captain should decide:
   - Whether the title candidates should include a method-forward option (e.g., the experiment plan's recommended title) alongside the current conservative options.
   - Whether a Background/Related Work section should be added to the skeleton.
3. Milestone 2K is now complete (T34 + T35 both passed). Captain should decide whether to proceed with a milestone gate review or continue to the next development phase.
4. The "Minimum Safe Paper Positioning" paragraph in the risk audit is the single most useful artifact for any future drafter — it should be treated as the canonical starting point for prose expansion.
