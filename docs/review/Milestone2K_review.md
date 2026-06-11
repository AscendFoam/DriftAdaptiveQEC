# Milestone 2K Review: Paper-Assembly Readiness

## Review Metadata

- Reviewer: Worker (T41)
- Date: `2026-05-17`
- Milestone: `2K: Paper Assembly Readiness`
- Scope reviewed:
  - `T34`: Paper claim/evidence ledger and figure-table outline
  - `T35`: Paper draft skeleton and reviewer-risk audit

## Verdict

`Allow`

Milestone 2K may close. Both T34 and T35 passed adversarial review with no blocking issues. The paper-assembly readiness artifacts — claim/evidence ledger, manuscript skeleton, figure-table outline, and reviewer-risk audit — are in place and internally consistent. No evidence boundary was silently upgraded during either task.

## 1. Milestone Scope Reviewed

### T34: Paper Claim/Evidence Ledger and Figure-Table Outline

- Output: `docs/paper_materials/paper_claim_evidence_ledger.md`
- Review: `docs/review/T34_review.md` (adversarial), verdict = `PASS`
- The ledger records 11 claims (C1–C11), 3 figures (F1–F3), and 5 tables (T1–T5).
- Status distribution: 5 `supported`, 1 `partial`, 5 `blocked`.
- Every `supported` claim cites concrete evidence paths verified to exist on disk.
- Every `blocked` claim cites a specific risk ID verified in `docs/08_risks_and_open_questions.md`.
- Boundary checks confirmed: C1/C8 (mock vs real-board), C7 (true `.tflite` vs stub), C2/C3/C11 (frozen-set vs expanded), C5/C6 (CPU smoke vs full reproducibility), C9/C10 (statcalib interface vs integrated comparator).
- Non-blocking issues from review: N1 (C9 indirect paths), N2 (no float/int8 gap claim), N3 (no ablation claims), N4 (worker pre-review overwrite). All accepted.

### T35: Paper Draft Skeleton and Reviewer-Risk Audit

- Output: `docs/paper_materials/paper_draft_skeleton.md`, `docs/paper_materials/paper_reviewer_risk_audit.md`
- Review: `docs/review/T35_review.md` (adversarial), verdict = `PASS`
- The skeleton provides 8 required sections: title candidates, abstract, introduction, method/system, experiment/evidence, results, limitations/boundary, conclusion.
- The risk audit provides 5 challenge-point categories (novelty, evidence-grade, overclaim wording, reproducibility/deployment, ablation/mechanism), 6 overclaim wording traps, a section-by-section hotspot table, and a minimum safe paper positioning statement.
- Cross-references verified: 46 C/F/T/R references in skeleton, 23 in risk audit, all consistent with T34 ID scheme.
- Non-blocking issues from review: N1 (conservative titles), N2 (no Background/Related Work), N3 (generic hotspot labels), N4 (worker pre-review overwrite). All accepted.

## 2. Whether Milestone 2K May Close

Yes. Milestone 2K may close.

Both tasks completed their goals without violating allowed-file or forbidden-scope constraints. The ledger, skeleton, and risk audit form a coherent paper-assembly toolkit that is ready for prose expansion — subject to the positioning decisions below.

## 3. Minimum Safe Paper Positioning Supported Today

The strongest defensible positioning, derived from the claim ledger and risk audit, is:

> A bounded recovery and revalidation manuscript for a CNN-assisted dual-loop GKP decoding pipeline, validated at the mock-backed software-HIL and frozen-set benchmark level, with one clean-environment CPU-only training smoke, and with explicit disclosure that deployment/runtime, real-board, broader benchmark, and integrated statcalib evidence are not yet complete.

This positioning:

- Is safe for all five `supported` claims: C1, C2, C3, C5, C9.
- Accommodates C4 (`partial`) with "single-seed trace-supported diagnosis" wording.
- Does not depend on any `blocked` claim: C6, C7, C8, C10, C11.
- Is consistent with the risk audit's "Minimum Safe Paper Positioning" and "Do-Not-Publish-As-Claimed List."

If a method-forward title is desired alongside the conservative options, the experiment plan's recommended title — "A Dual-Loop Teacher-Guided Residual Decoding Framework for Real-Time GKP Error Correction" — is defensible provided the abstract, introduction, and limitations sections stay within the evidence boundaries above. The T35 review N1 noted that the current title candidates lean too far toward "recovery report" framing for the target venues (QCE, TQE, EPJ Quantum Technology). A method-forward title with evidence-bounded body text is a reasonable compromise.

## 4. Blocked Claims That Still Prevent Stronger Positioning

The following blocked claims prevent upgrading to a stronger paper positioning:

| Blocked claim | What it blocks | Current blocker | Risk ID |
| --- | --- | --- | --- |
| C6 | "Reproducible training pipeline" | Only one clean CPU-only smoke; no cross-platform, GPU, or multi-run reproducibility | R11 |
| C7 | "TFLite deployment-ready" | No `tensorflow` / `tflite_runtime` on current machine | R12 |
| C8 | "Real-board HIL validated" | `board_backend.py` still placeholder; no device, permissions, or register evidence | R13, R14 |
| C10 | "Integrated statcalib comparator evidence" | statcalib is interface-only; not integrated into slow-loop or benchmark | R24 |
| C11 | "Paper-grade expanded benchmark" | Current benchmark is frozen-set only; no expanded scenario, comparator, or CI-driven stopping | R5, R9 |

These blockers remain active. No T34 or T35 output silently upgraded any of them.

Additionally, the following partial/open risks constrain even the supported claims:

- R10 (mechanism evidence): trace-supported for one seed only; no mitigation or multi-seed confirmation.
- R20 (correction saturation structural zero): may be genuine zero in current parameter range, but triggerability not proven.
- R23 (aggregation/report writer lacks focused tests): regression risk remains.

## 5. Decision: Background / Related Work Before Prose Expansion

**Yes. A Background / Related Work scaffold must be added before prose expansion begins.**

Rationale:

1. The T35 review (N2) identified that the skeleton omits this section. The experiment plan (Section 10.5) calls for it. Target venues (QCE, TQE, EPJ Quantum Technology) require situating the work in the GKP QEC landscape.
2. Without a Background / Related Work scaffold, later drafting has no structural home for: (a) GKP syndrome basics and fast/slow loop time-scale separation, (b) prior work on CNN-assisted or adaptive QEC decoding, (c) why the evidence-boundary approach matters relative to standard system papers.
3. The risk audit's novelty challenge N1 ("this reads like a recovery report, not a novel method paper") is partially addressable by a well-structured Background section that frames the method contribution before the evidence-boundary contribution.
4. Adding this scaffold is a bounded docs-only task that does not require new experiments, benchmark runs, or evidence upgrades.

This should be the first prose-expansion task, before writing any body text.

## 6. Recommended Next Unique Task

The recommended next unique task is:

> **T42: Paper Background / Related Work scaffold and method-positioning calibration**

Scope:

- Add a Background / Related Work section to the existing skeleton in `docs/paper_materials/paper_draft_skeleton.md`.
- Decide whether to adopt a method-forward title (e.g., the experiment plan's recommended title) alongside or replacing the current conservative candidates.
- Calibrate the introduction's contribution bullets against the claim ledger: ensure C1–C5, C9 appear in contribution bullets with correct status wording, and blocked claims do not.
- Output remains bounded: no full prose drafting, no new experiments, no code changes.

This task does not execute prose expansion beyond the Background scaffold. Full prose expansion should follow after T42, section by section, as separate bounded tasks.

Alternative candidates considered but deferred:

- Full paper prose drafting: premature without Background scaffold and title calibration.
- Statcalib integration or new benchmark: blocked by R24, R5, R9; does not directly advance paper readiness.
- Real-board execution: blocked by R13, R14; requires hardware prerequisites.
- `.tflite` runtime restoration: blocked by R12; requires environment setup.

## Evidence-Level Impact

Milestone 2K improves paper-assembly readiness from "no structured drafting toolkit" to "ledger, skeleton, and risk audit in place."

It does not upgrade the project to:

- paper-grade expanded benchmark evidence
- deployment or runtime evidence
- hardware evidence
- causal mechanism proof

## Residual Risks

The following risks remain unchanged from pre-Milestone 2K state:

- R5, R9: frozen-set benchmark scope
- R10: mechanism evidence limited to single-seed diagnosis
- R11: training reproducibility not proven beyond one CPU-only smoke
- R12: true `.tflite` runtime unavailable
- R13, R14: real-board path still placeholder
- R20: correction saturation triggerability not proven
- R23: aggregation/report writer test gap
- R24: statcalib not integrated into comparator lane

No new risks were introduced by T34 or T35.
