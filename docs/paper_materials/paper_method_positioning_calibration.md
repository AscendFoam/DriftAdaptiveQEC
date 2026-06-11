# Paper Method-Positioning Calibration

## 1. Purpose

This document calibrates the paper's framing against the claim/evidence ledger (`docs/paper_materials/paper_claim_evidence_ledger.md`, C1–C11) and the Milestone 2K review (`docs/review/Milestone2K_review.md`). It compares two framing options, recommends the safe choice, and lists forbidden phrases that would upgrade blocked claims.

## 2. Conservative Framing Option

**Title direction**: "Evidence-Bounded Recovery..." or "Controlled Revalidation of..."

**Positioning**: A bounded recovery and revalidation manuscript for a CNN-assisted dual-loop GKP decoding pipeline, validated at mock-backed software-HIL and frozen-set benchmark level, with one clean-environment CPU-only training smoke, and explicit disclosure of deployment/runtime, real-board, broader benchmark, and integrated statcalib gaps.

**Strengths**:
- Maximally safe: no claim can be challenged as overclaimed.
- Honest about all evidence boundaries.
- Consistent with the "Minimum Safe Paper Positioning" from the reviewer risk audit.

**Weaknesses**:
- The T35 review (N1) noted this framing reads like "a recovery report, not a novel method paper."
- Conservative titles do not match the expected framing at target venues (QCE, TQE, EPJ Quantum Technology).
- The method contribution (teacher-guided residual-b) is buried under recovery/revalidation language.

**Safe claims under this framing**: C1, C2, C3, C5, C9, C4 (with limitation wording).

## 3. Method-Forward Framing Option

**Title direction**: "A Dual-Loop Teacher-Guided Residual Decoding Framework for Real-Time GKP Error Correction" (experiment plan recommended title) or "Teacher-Guided Residual Adaptive Decoding for GKP Error Correction Under Real-Time Hardware Constraints"

**Positioning**: A method paper introducing the teacher-guided residual-b decoding framework for real-time GKP error correction, supported by frozen-set benchmark evidence showing the method wins all four drift scenarios against five classical baselines under mock-backed software HIL, with one clean-environment CPU-only training smoke and explicit disclosure of evidence boundaries.

**Strengths**:
- Correctly centers the method contribution (dual-loop teacher-guided residual-b) rather than the recovery process.
- Matches the expected framing at target venues.
- The experiment plan (Section 10.1) already recommended this title direction.
- The T35 review N1 and Milestone 2K review both suggested that a method-forward title with evidence-bounded body text is a reasonable compromise.

**Weaknesses**:
- Requires disciplined boundary language in the abstract, introduction, and limitations to avoid drifting into overclaim.
- Reviewers may challenge novelty if the body text does not clearly distinguish the method contribution from the evidence-boundary contribution.

**Safe claims under this framing**: Same as conservative — C1, C2, C3, C5, C9, C4 (with limitation wording). The framing change does not upgrade any claim.

## 4. Recommended Safe Framing

**Recommendation**: Method-forward title, evidence-bounded body.

The method-forward title "A Dual-Loop Teacher-Guided Residual Decoding Framework for Real-Time GKP Error Correction" is recommended as the primary working title, pending any later Captain/human override before full-paper submission, for the following reasons:

1. **Venue match**: QCE, TQE, and EPJ Quantum Technology expect method contributions, not recovery reports. The method-forward title signals the right category.
2. **Evidence-safe**: The title describes a framework, not a completed deployment. The word "framework" is honest: the dual-loop architecture, teacher-guided residual-b method, frozen-set benchmark evidence, training smoke, and statcalib interface contract are all real contributions that exist today.
3. **Risk-audit alignment**: The reviewer risk audit's novelty challenges (N1, N2) are best addressed by centering the method contribution. A well-structured Background / Related Work section that frames "teacher + residual" as distinct from "CNN replaces everything" provides the novelty defense.
4. **No evidence upgrade**: The method-forward title does not depend on any blocked claim. The framework is real; the evidence supporting it is bounded but genuine.

**Critical constraint**: The abstract, introduction contribution bullets, and limitations section must stay strictly within the evidence boundaries. The title signals method ambition; the body delivers evidence-bounded results.

## 5. Contribution-Bullet Calibration Against C1–C11

The introduction contribution bullets should be calibrated as follows:

| Contribution | Source claim | Status | Required wording constraint |
| --- | --- | --- | --- |
| Dual-loop CNN-FPGA GKP decoding framework with deterministic software-HIL replay | C1 | supported | "mock-backed software HIL," not "hardware validated" |
| Frozen-set formal benchmark showing hybrid_residual_b wins all four drift scenarios | C2, C3 | supported | "frozen-set formal software revalidation," not "comprehensive benchmark" |
| Single-seed trace-supported mechanism diagnosis of seed instability | C4 | partial | "single-seed trace-supported diagnosis suggests," not "mechanism proven" or "root cause found" |
| Clean-environment CPU-only training smoke | C5 | supported | "one clean-environment CPU-only smoke," not "reproducible training pipeline" |
| Statcalib interface contract and focused tests | C9 | supported | "interface contract only," not "integrated calibrated comparator" |

Claims that must NOT appear in contribution bullets:

| Blocked claim | Why blocked | Forbidden contribution wording |
| --- | --- | --- |
| C6 | Only one CPU-only smoke | "reproducible training pipeline," "cross-platform training validated" |
| C7 | No tensorflow/tflite_runtime | "TFLite deployment-ready," "runtime validation complete" |
| C8 | board_backend.py is placeholder | "real-board HIL validated," "hardware deployment demonstrated" |
| C10 | Statcalib is interface-only | "integrated statcalib comparator evidence," "calibrated benchmark advantage" |
| C11 | Frozen-set only | "comprehensive benchmark," "broad evaluation," "expanded comparator coverage" |

## 6. Forbidden Phrases

The following phrases are forbidden in any paper section because they would silently upgrade blocked claims:

1. "hardware validated" or "real-board validated" — upgrades C1 to C8
2. "deployment-ready" or "TFLite deployed" — upgrades C7
3. "reproducible training pipeline" or "training reproducibility established" — upgrades C5 to C6
4. "mechanism explained" or "root cause proven" — upgrades C4 from partial to supported
5. "comprehensive benchmark" or "broad evaluation" — upgrades C2/C3 to C11
6. "integrated calibrated comparator" or "statcalib benchmark advantage" — upgrades C9 to C10
7. "generally superior decoder" or "state-of-the-art" — implies C11-level evidence
8. "nearly deployment-ready" or "effectively reproducible" — future-tense disguised as completion

Safe replacements are documented in the claim ledger's "Wording Guardrails" section and the risk audit's "Overclaim Wording Traps" table.
