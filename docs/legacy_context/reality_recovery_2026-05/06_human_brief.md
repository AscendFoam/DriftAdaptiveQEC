# Human Brief

Freeze date: `2026-05-19`

## Project State

YELLOW

## What Was Verified

1. A bounded `mock-backed software HIL` path has been recovered, revalidated, and confirmed deterministic.
2. `hybrid_residual_b` wins all four frozen scenarios under the locked formal protocol (4 scenarios x 5 modes x 2 repeats).
3. One clean Windows/Python 3.12 CPU-only training smoke completed successfully.
4. `statcalib` has an interface contract and focused tests (interface-only, not integrated).
5. `seed=20260429` has single-seed trace-level evidence supporting a committed-`b` instability hypothesis.
6. Code truth audit confirms docs are mostly honest relative to code. One stale label found: `board_backend.py` says "placeholder" but is structurally complete (308 lines of AXI/DMA code). No real hardware connected though, so evidence boundary remains correct.
7. No TODO/FIXME/HACK/PLACEHOLDER markers hidden in code.

## What Is Still Unverified

1. True `.tflite` export/runtime (tensorflow not installed).
2. Real-board HIL validation (no device connected, no board logs).
3. Multi-seed mechanism closure (only `seed=20260429` has trace evidence).
4. Broader benchmark expansion beyond the frozen set.
5. Training reproducibility beyond one CPU-only smoke.
6. Correction saturation triggerability (metric is identically 0.0 across all T24 runs).
7. Integrated statcalib comparator evidence.
8. Paper-ready figure/material pack (5 items `ready`, 5 `partial`, 3 `missing`, 3 `blocked`).

## What Looks Suspicious

1. `board_backend.py` is labeled "placeholder" but is actually a 308-line structurally complete implementation. The label should be updated but the evidence boundary is correctly honest.
2. The paper Background/Related Work prose draft exists and is evidence-bounded, but has internal drafting annotations that need cleanup. The prose does not silently upgrade any blocked claims.
3. 9 out of 15 claims are still `blocked`. The ratio means the paper can proceed with framing and bounded prose drafting, but cannot make strong empirical claims yet.
4. 3 key figures are `missing` and 3 are `blocked`. The paper does not have a complete figure/material pack.

## Can Paper Writing Continue Right Now?

No. Not as full-paper results expansion.

Paper prose is paused by user instruction (`docs/legacy_context/reference_retired_2026-06-11/科研纠偏意见.md`). The correct next step is evidence repair, not prose expansion.

What may continue:
- Governance and framing calibration (already done in T42/T43)
- Evidence gap filling (T45-T47)

What must not continue until mainline blockers show progress:
- Abstract / Results / Conclusion prose
- Any claim that upgrades evidence levels

## Is T44 Enough for a Strong Paper?

No. T44 freezes truth but does not create missing evidence.

Even with all currently proposed tasks (T45-T52), the paper would still need:
- T45-T47 execution (mainline evidence hardening)
- At least T48 or T49 (deployment boosters)
- T50 (reproducibility pack)
- T51-T52 (paper re-open gate)

## What Human Decision Is Needed

Choose one:

1. **Continue recovery**: Execute T45-T47 as bounded evidence-hardening tasks. This is the recommended path.
2. **Narrow paper scope**: Accept the frozen-set-only evidence as the final paper scope and write a narrower workshop/tech-report-style paper.
3. **Pause project**: Wait for hardware availability (real board, TensorFlow environment) before continuing.
4. **Salvage code only**: Abandon paper submission for now and focus on code cleanup and reproducibility.

Recommendation: Option 1 (continue recovery with T45-T47).

## How to Treat 延伸改进思路

Treat `docs/reference/延伸改进思路.md` as a future extension lane. Do not treat it as:
- current mainline evidence
- a hidden requirement for the current paper thesis
- mandatory reading for the next bounded task
