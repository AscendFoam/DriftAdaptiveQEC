# Review: T43

Verdict: PASS

## Blocking Issues

None.

## Non-Blocking Issues

### N1 Subsection 6 transitions from survey material to self-referencing

The first paragraph of subsection 6 ("Evidence Boundaries in Quantum System Validation") is neutral survey material about the simulation-vs-hardware evidence gap in quantum systems research. The second paragraph shifts to "The present work encounters this boundary directly" and proceeds to enumerate the project's own evidence boundaries. This self-referencing echoes the concern raised in T42 review N1: the subsection risks being read as "our transparency about evidence gaps is itself a novelty."

The subsection avoids actual overclaim---every boundary statement is factually correct and correctly references the claim ledger (C5 supported, C6/C7/C8 blocked). It does not claim novelty for the transparency itself. However, the closing sentence ("Being explicit about these evidence boundaries is important for two reasons") nudges toward self-justification.

Not blocking because: (a) the subsection is short (two paragraphs), (b) all factual content is correct, (c) blocked claims are marked as blocked rather than disguised, (d) the T43 task package explicitly allowed omitting this subsection if it felt self-serving, and (e) if awkward during later prose expansion, it can be folded into the Limitations section without affecting the other five subsections.

### N2 Placeholder citation markers lack a bibliography

The draft uses numeric citation markers [1]--[7] for key references (GKP encoding, surface code neural decoders, circuit-level noise decoders, etc.) but provides no corresponding bibliography or mapping to specific papers. The markers are contextually appropriate (e.g., "[1, 2]" for GKP codes is standard), but they cannot be resolved without a shared bibliography file.

Not blocking because: the T43 scope is bounded Background/Related Work prose only, not full manuscript assembly. The citation markers are correctly placed in context and can be resolved in a later task. Recommendation: establish a shared bibliography file before the next prose drafting task to prevent citation drift across sections.

### N3 Internal drafting annotations in prose require later cleanup

The draft contains internal cross-references that are not standard academic prose: "[stable conclusion 9.1 item 7]" in subsection 5, "[supported claims C2, C3]" in subsection 4, "[supported claim C3]" and "[stable conclusion 9.1 item 3]" in subsection 5, and "[supported claim C5]", "[blocked claim C6]", "[blocked claims C7, C8]" in subsection 6. These are useful drafting-stage annotations that keep the evidence boundary visible during prose expansion, but they must be removed or converted to natural language before manuscript assembly.

Not blocking because: these annotations serve their purpose during the bounded drafting process and the T43 task package explicitly allows them. They do not affect the correctness of the prose content.

### N4 Inline claim reference formatting inconsistency

The draft uses both plural and singular forms: "[supported claims C2, C3]" vs "[supported claim C3]" vs "[blocked claim C6]" vs "[blocked claims C7, C8]". This is purely cosmetic but should be normalized during manuscript assembly.

Not blocking because: the formatting variation does not affect meaning or evidence correctness.

## Missing Tests

Not applicable. T43 is a docs-only prose drafting task with no code, configuration, or executable changes. Verification is wording-and-boundary based.

## Suspicious Implementation Details

None found. Specific checks:

1. **Allowed files respected**: `git status` confirms exactly 4 T43-scoped files:
   - `docs/paper_materials/paper_background_related_work_draft.md` (new)
   - `docs/review/T43_review.md` (new)
   - `docs/for_human/T43_explanation.md` (new)
   - `docs/tasks/Phase2/T43_paper_background_related_work_prose_draft.md` (new, with Verification Record)
   No source code, config, `runs/`, `artifacts/`, or governance docs were touched.

2. **Evidence boundary preserved throughout**: Cross-checked every factual statement against the claim ledger:
   - Subsection 1: Pure background (GKP encoding, syndrome measurement, linear decoding). No project-specific claims.
   - Subsection 2: Architecture description (fast loop ~5us, slow loop 10--100ms, 32x32 histogram, parameter bank). Consistent with experiment plan Section 1.1.
   - Subsection 3: Literature survey of ML-based QEC decoding. Project positioning uses neutral language: "the present work follows this direction."
   - Subsection 4: Describes five classical baselines (EKF, UKF, WV, RLS, Constant Residual-Mu). References C2 and C3 as supported. Uses correct terminology: "mock-backed software hardware-in-the-loop (HIL) revalidation."
   - Subsection 5: Key method positioning. References stable conclusion 9.1 item 7 (offline training improvement does not reliably predict formal HIL improvement). References C3 for frozen-set results. UKF correctly identified as strongest classical baseline. Correction saturation and aggressive parameter rates correctly stated as zero (consistent with experiment plan Section 9.1 item 3).
   - Subsection 6: Evidence boundary discussion. Correctly marks C5 as supported, C6/C7/C8 as blocked.

3. **No blocked claim silently promoted**: Each blocked claim in the prose is explicitly marked as blocked:
   - C6: "full cross-platform training reproducibility remains unverified [blocked claim C6]"
   - C7/C8: "True TFLite runtime validation and real-board HIL execution are not yet available on the current machine [blocked claims C7, C8]"
   - C10/C11: Not mentioned in Background/Related Work (correctly scoped out).

4. **No forbidden phrases used**: Checked against all 8 forbidden phrase categories from the calibration note Section 6:
   - "hardware validated" / "real-board validated" absent
   - "deployment-ready" / "TFLite deployed" absent
   - "reproducible training pipeline" / "training reproducibility established" absent
   - "mechanism explained" / "root cause proven" absent
   - "comprehensive benchmark" / "broad evaluation" absent
   - "integrated calibrated comparator" / "statcalib benchmark advantage" absent
   - "generally superior decoder" / "state-of-the-art" absent
   - "nearly deployment-ready" / "effectively reproducible" absent

5. **Method-forward framing stays within evidence**: The draft positions teacher-guided residual correction as a method contribution with frozen-set benchmark support. The novelty claim in subsection 5 ("to the best of our knowledge, novel") is qualified and anchored to a specific architectural distinction (teacher + residual vs. CNN-replaces-everything). This directly addresses risk-audit novelty challenges N1 and N2 without overclaiming. The draft does not claim broad superiority, state-of-the-art status, or hardware validation.

6. **Subsection 5 benchmark result citation is defensible**: The frozen-set benchmark result appears in subsection 5 as method justification rather than results reporting. The T42 skeleton's drafting notes explicitly frame subsection 5 as the "key novelty framing section," so referencing benchmark evidence to justify the method choice is within scope.

7. **Drafting scope correctly bounded**: Only Background/Related Work prose is produced. No Abstract, Introduction, Method, Experiment, Results, Conclusion, or Appendix prose appears. The draft does not drift into full manuscript assembly.

8. **Verification Record is complete**: All required verification checks documented in the task package's Verification Record.

## Recommended Next Action

1. Captain should accept T43 as `PASS`.
2. Before the next prose drafting task, consider establishing a shared bibliography file so citation markers [1]--[7] can be resolved consistently across sections.
3. Subsection 6 can remain in the draft for now. If it continues to feel self-serving during later review rounds, fold the content into the Limitations section as recommended by T42 review N1.
4. The next prose drafting task should target either: (a) Introduction prose (calibrated against the T42 contribution bullets), or (b) Method/System prose (the system description that follows naturally from the Background subsections 1--2).
5. Inline drafting annotations ([supported claim C3], [stable conclusion 9.1 item 7], etc.) should be preserved through the drafting phase and cleaned up during manuscript assembly.
