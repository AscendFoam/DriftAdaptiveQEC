# Paper Draft Skeleton

## Scope Note

This file is a bounded manuscript scaffold derived from `docs/paper_claim_evidence_ledger.md`. It is not full paper prose. Any later expansion must preserve the current claim states:

- `supported`: may be written as completed evidence-backed prose
- `partial`: must be written with explicit limitation language
- `blocked`: must not be written as completed evidence-backed prose

## Title Candidates

1. Evidence-Bounded Recovery of a CNN-Assisted QEC Decoding Pipeline Under Mock-Backed Software HIL
2. Controlled Revalidation of a CNN-Based QEC Decoder: Frozen-Set Benchmark Evidence, Training-Chain Smoke, and Deployment Boundaries
3. Recovering a Residual CNN Decoder Stack With Explicit Evidence Boundaries for Software HIL and Training Reproducibility
4. Benchmark Revalidation and Deployment-Boundary Audit for a CNN-Guided QEC Decoding Pipeline

## Global Guardrails

- Allowed completed-claim pool: `C1`, `C2`, `C3`, `C5`, `C9`
- Allowed partial-claim pool with explicit caveats: `C4`
- Blocked claims that must not be upgraded into completed prose: `C6`, `C7`, `C8`, `C10`, `C11`
- Reusable figures/tables:
  - Figures: `F1` partial, `F2` supported, `F3` blocked
  - Tables: `T1` supported, `T2` supported, `T3` supported, `T4` supported, `T5` partial

## Abstract Skeleton

### Intended subsection headings

1. Problem framing
2. What was recovered and revalidated
3. What current evidence supports
4. What remains explicitly out of scope

### Allowed evidence map

- Claims: `C1`, `C2`, `C3`, `C5`, `C9`, `C4` with limitation wording only
- Figures/Tables: none required in abstract prose

### Blocked claims not allowed in completed prose

- `C6`: full training reproducibility or portability
- `C7`: true `.tflite` runtime restored
- `C8`: real-board HIL validated
- `C10`: statcalib integrated comparator evidence complete
- `C11`: expanded paper-grade benchmark evidence complete

### Drafting notes

- Position the paper as a bounded recovery and evidence-audit manuscript, not as a deployment-complete system paper.
- If mentioning mechanism insight, phrase it as a single-seed trace-supported diagnosis under `C4`, not proof.

## Introduction Skeleton

### Intended subsection headings

1. QEC decoding motivation and engineering gap
2. Why evidence boundaries matter for this repo state
3. Narrow paper thesis
4. Evidence-bounded contribution bullets

### Allowed evidence map

- Claims: `C1`, `C2`, `C3`, `C5`, `C9`, `C4` partial
- Figures/Tables: `T2`, `T3`, `T4`, `T5`, optionally `F2`

### Blocked claims not allowed in completed prose

- `C8` and `C7` must not appear in any contribution bullet
- `C11` must not be implied by phrases like "comprehensive benchmark" or "broad evaluation"
- `C10` must not be implied by phrases like "integrated calibrated comparator"

### Drafting notes

- Contribution bullets should read like:
  - bounded software-HIL revalidation
  - frozen-set benchmark ranking evidence
  - one clean-environment training smoke
  - interface-level statcalib comparator preparation
- Avoid novelty framing that depends on real-board deployment or broad benchmark coverage.

## Method / System Skeleton

### Intended subsection headings

1. Repository recovery state and evidence model
2. Decoder variants and scenario family
3. Mock-backed software HIL execution boundary
4. Training-chain boundary and clean-environment smoke
5. Statcalib interface contract status

### Allowed evidence map

- Claims: `C1`, `C2`, `C5`, `C9`
- Figures/Tables: `F2`, `T2`, `T3`, `T4`, `T5`

### Blocked claims not allowed in completed prose

- `C7`: no true runtime `.tflite` method subsection
- `C8`: no real-board hardware execution subsection
- `C10`: no integrated statcalib comparator pipeline subsection

### Drafting notes

- The system diagram must stay software-side unless new evidence closes `C8`.
- Training subsection should document environment and isolation boundaries rather than claim reproducibility.
- If deployment path is mentioned, route it to `T4` as readiness-only status.

## Experiment / Evidence Skeleton

### Intended subsection headings

1. Frozen-set benchmark protocol and gate conditions
2. Scenario and seed coverage actually exercised
3. Training smoke protocol
4. Mechanism-diagnosis trace protocol
5. Boundary-only deployment/readiness checks

### Allowed evidence map

- Claims: `C2`, `C3`, `C4` partial, `C5`, `C9`
- Figures/Tables: `F1`, `F2`, `T1`, `T2`, `T3`, `T4`, `T5`

### Blocked claims not allowed in completed prose

- `C11`: do not describe the benchmark as broad or exhaustive
- `C6`: do not describe the training protocol as reproducibility validation
- `C7` and `C8`: do not present runtime or board execution as experiment sections with completed results

### Drafting notes

- This section should separate "formal revalidation evidence" from "readiness-only evidence."
- `F1` must be explicitly captioned as single-seed diagnosis evidence.
- `T4` should be presented as deployment-boundary status, not deployment success.

## Results Skeleton

### Intended subsection headings

1. Frozen-set ranking result
2. Scenario-wise consistency and limitations
3. Seed `20260429` diagnosis signal
4. Clean CPU-only training smoke outcome
5. Statcalib comparator status summary

### Allowed evidence map

- Claims: `C3`, `C4` partial, `C5`, `C9`
- Figures/Tables: `F1`, `T1`, `T3`, `T5`

### Blocked claims not allowed in completed prose

- `C11`: no generalization from frozen-set wins to expanded benchmark superiority
- `C6`: no claim that the train smoke proves stable reproducibility
- `C10`: no claim that statcalib has integrated benchmark advantage

### Drafting notes

- Keep the strongest empirical statement at the frozen-set level: `hybrid_residual_b` wins all four frozen scenarios under current protocol.
- Mechanism narrative must stay diagnostic, not causal-proof language.
- Training result framing should be "one real clean-environment run executed successfully on CPU-only Windows/Python 3.12."

## Limitations / Boundary Skeleton

### Intended subsection headings

1. Benchmark coverage boundary
2. Mechanism-evidence boundary
3. Training reproducibility boundary
4. Deployment/runtime boundary
5. Real-board boundary
6. Statcalib integration boundary

### Allowed evidence map

- Claims: `C4` partial, `C6`, `C7`, `C8`, `C10`, `C11`
- Figures/Tables: `T2`, `T3`, `T4`, `T5`
- Risk references to surface explicitly: `R5`, `R9`, `R10`, `R11`, `R12`, `R13`, `R14`, `R20`, `R24`

### Blocked claims not allowed in completed prose

- None of the blocked claims may be converted into future-tense disguised completion statements such as "effectively solved" or "nearly validated"

### Drafting notes

- This section should be unusually explicit. It is the main protection against reviewer overclaim concerns.
- If the paper includes a threat-to-validity subsection, it can be merged here but must preserve the same blocked statuses.

## Conclusion Skeleton

### Intended subsection headings

1. Safe summary of supported outcomes
2. What the present evidence does not establish
3. Evidence-upgrade path for a stronger paper

### Allowed evidence map

- Claims: `C1`, `C2`, `C3`, `C5`, `C9`, `C4` partial
- Figures/Tables: `T1`, `T2`, `T3`, `T4`, `T5`

### Blocked claims not allowed in completed prose

- `C6`, `C7`, `C8`, `C10`, `C11`

### Drafting notes

- Close on controlled recovery, bounded revalidation, and evidence transparency.
- Do not conclude with deployment readiness, full reproducibility, or hardware completion language.

## Appendix Planning Notes

### Safe appendix candidates

- Benchmark protocol details anchored to `C2`, `T1`, `T2`
- Training smoke environment details anchored to `C5`, `T3`
- Deployment/readiness boundary note anchored to `T4`
- Statcalib interface-only status anchored to `C9`, `T5`

### Appendix topics that must stay absent unless evidence changes

- Real-board execution logs as validation evidence
- True `.tflite` runtime latency/accuracy results
- Cross-platform or GPU reproducibility matrices
- Integrated statcalib comparator win/loss analysis
