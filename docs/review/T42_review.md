# Review: T42

Verdict: PASS

## Blocking Issues

None.

## Non-Blocking Issues

### N1 Background subsection 6 may read as self-serving novelty framing

The Background / Related Work skeleton includes subsection 6: "Benchmark and deployment evidence boundaries in quantum system papers." The drafting notes describe this as "a contribution to reproducible quantum engineering practice."

This framing risks reading to a reviewer as: "our transparency about weaknesses is itself a novelty." While the observation that quantum systems papers face software-vs-hardware evidence gaps is valid, positioning it as a contribution rather than a good-practice survey section could trigger the exact novelty challenge (N1) the paper is trying to avoid.

Not blocking because: (a) the skeleton only proposes headings and drafting notes, not prose; (b) the subsection can be drafted neutrally as a literature survey of evidence-grade norms without claiming novelty; (c) if awkward during prose expansion, it can be folded into the Limitations section without affecting the other five subsections.

### N2 Method-forward title recommendation is a human decision, not a Worker commitment

The calibration note recommends "A Dual-Loop Teacher-Guided Residual Decoding Framework for Real-Time GKP Error Correction" as the primary working title. This is well-supported (experiment plan Section 10.1, T35 review N1, Milestone 2K review all suggest method-forward is viable), but title selection is ultimately a Captain/human decision that should be explicitly confirmed before prose expansion begins.

Not blocking because: the skeleton preserves all four title candidates (2 conservative + 2 method-forward) without deleting any option, and the calibration note clearly labels this as "recommended."

### N3 Worker pre-review overwritten by adversarial review

Standard pattern from T34/T35/T36/T38/T41. The Worker's self-review was replaced by this adversarial review. Verification notes are preserved in the task package's Verification Record.

Not blocking because: this is the expected project workflow.

## Missing Tests

Not applicable. T42 is a docs-only task with no code, configuration, or executable changes. Verification is structure-and-boundary consistency based.

## Suspicious Implementation Details

None found. Specific checks:

1. **Allowed files respected**: `git status` confirms exactly 5 T42-scoped files changed/created:
   - `docs/paper_materials/paper_draft_skeleton.md` (modified)
   - `docs/tasks/Phase2/T42_paper_background_related_work_and_positioning.md` (modified — Verification Record added)
   - `docs/paper_materials/paper_method_positioning_calibration.md` (new)
   - `docs/review/T42_review.md` (new)
   - `docs/for_human/T42_explanation.md` (new)
   No source code, config, `runs/`, `artifacts/`, governance docs, or stage-conclusion documents were touched.

2. **Background / Related Work added without upgrading evidence**: The new section's "Allowed evidence map" restricts to C1, C2, C3, C9 for situating the method — no new claims introduced. The "Blocked claims not allowed in completed prose" block correctly lists C6, C7, C8, C10, C11 with specific instructions not to cite them as established results.

3. **Title calibration stays inside Milestone 2K limits**: The method-forward title was already recommended by the experiment plan (Section 10.1) and identified as a reasonable compromise by T35 review N1 and the Milestone 2K review. The calibration note does not upgrade evidence or bypass the milestone review's positioning decision. Conservative title options are preserved, not deleted.

4. **Contribution bullets aligned with C1–C11**: Cross-checked each bullet against the claim ledger:
   - C1 (supported) — "bounded software-HIL revalidation" ✅ correct status
   - C2, C3 (supported) — "frozen-set formal benchmark showing hybrid_residual_b wins all four" ✅ correct status
   - C4 (partial) — "single-seed trace-supported mechanism diagnosis...as a hypothesis" ✅ includes limitation wording
   - C5 (supported) — "one clean-environment CPU-only training smoke" ✅ correct status
   - C9 (supported) — "separate statcalib interface contract" ✅ includes boundary wording
   - C6, C7, C8, C10, C11 (blocked) — explicitly excluded from contribution bullets ✅

5. **No blocked claim silently promoted**: The calibration note's Section 6 ("Forbidden Phrases") lists 8 phrase categories that would upgrade blocked claims. Each category maps correctly to a specific upgrade path (e.g., "hardware validated" → upgrades C1 to C8, "reproducible training pipeline" → upgrades C5 to C6).

6. **Calibration note consistent with risk audit**: The "Minimum Safe Paper Positioning" and "Do-Not-Publish-As-Claimed List" from `docs/paper_materials/paper_reviewer_risk_audit.md` are preserved without drift. The conservative framing option quotes the exact same positioning language from the risk audit.

7. **Background section drafting notes reference correct sources**: Subsection 5 references "offline training improvement ≠ formal HIL improvement" which matches stable conclusion 9.1 item 7 from the experiment plan. Subsection drafting notes cite risk-audit N1 and N2 correctly.

8. **Verification Record is complete**: All 11 required checks documented in the task package's Verification Record.

## Recommended Next Action

1. Captain should accept T42 as `PASS`.
2. The method-forward title recommendation should be confirmed or overridden by Captain/human before prose expansion begins — this is a framing decision, not a technical one.
3. The Background / Related Work scaffold is ready for prose expansion. The next unique task should be a bounded prose-drafting task starting with Background / Related Work subsections 1–4 (standard background material), followed by subsection 5 (the key novelty framing section).
4. No code, config, `runs/`, or `artifacts` changes are needed before prose expansion can start.
