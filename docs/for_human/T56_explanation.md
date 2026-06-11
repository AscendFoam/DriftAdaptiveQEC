# T56: Post-I1 Mechanism Claim Reframing Gate — Human Summary

## What this task is trying to accomplish

After running a multi-seed intervention experiment (T55) that tested whether reducing the "committed-b instability" in Gated v5 would improve results, the project discovered something surprising: the intervention **hurt** performance on 4 out of 6 seeds and only helped 2. This means the earlier narrative — "high committed-b is the problem, reducing it will help" — is not correct as a general explanation.

T56 is a gate task that pauses before any next step to honestly reassess what claims remain valid and which ones need to be retired or reframed.

## What changed

The main gate document (`docs/evidence_packs/mechanism_ablation/post_t55_mechanism_claim_reframing_gate.md`) contains a claim table with 10 mechanism claims, each classified as:

- **Retain (3 claims):** The core observations are still valid — seed=20260429 shows instability, the pattern generalizes to 5/6 seeds, and teacher-delta channels remain an open hypothesis.
- **Weaken (4 claims):** Some claims need narrower wording — the instability is not the primary cause of degradation (it mostly helps), it is not exclusive to Gv5, the seed categories do not predict intervention outcomes, and committed-b is not "the cause of Gv5 degradation."
- **Retire (2 claims):** Two claims are directly contradicted by T55 — "high committed-b is harmful" and "reducing the residual clip will stably improve outcomes."
- **Reframe (1 claim):** "The instability needs to be fixed" should be reframed — the instability appears to be Gv5's performance mechanism, not a defect.
- **Still-open (2 claims):** Teacher-delta causation and second intervention justification remain unresolved.

## What this means for the project

1. **T47 (paper ablation packaging) can proceed**, but only with explicit hedging around the mechanism story. The paper should not claim the instability is "solved" or "proven harmful."
2. **A second intervention is deferred**, not approved. The question needs reframing before another execution is justified.
3. **The mechanism story is more nuanced than expected.** Gated v5's committed-b instability is broadly present but mostly correlates with better outcomes. It may be a feature, not a bug.
4. **No code, benchmark, or execution was changed.** This task only produced documentation.

## Key files

- Main gate document: `docs/evidence_packs/mechanism_ablation/post_t55_mechanism_claim_reframing_gate.md`
- Updated claim ledger: `docs/paper_materials/paper_claim_evidence_ledger.md` (C4 wording updated)
- Review: `docs/review/T56_review.md`
