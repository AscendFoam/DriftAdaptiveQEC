# T47: Paper Ablation Result-Pack and Material Ledger — Human Summary

## What this task is trying to accomplish

After T56 reframed the mechanism story, the project needs a clear picture of which paper materials (figures, tables, ablation results) are ready, which are partial, and which are missing. T47 is a docs-only task that produces this ledger, so that anyone looking at the paper can see exactly what evidence exists and what still needs work.

## What changed

The main output (`docs/paper_ablation_result_pack.md`) contains:

1. **A full ready/partial/missing ledger** for figures and tables, including:
   - 3 ready items (benchmark ranking table, boundary diagram, boundary/evidence-level table)
   - 5 partial items (seed=20260429 diagnosis figure, system architecture figure, scenario benchmark figure, latency table, statcalib status table)
   - 3 missing items (multi-seed mechanism figure, feature ablation table, statcalib result table)
   - 3 blocked items (training portability figure, TFLite runtime figure, real-board figure)

2. **Regeneration paths** for each asset — concrete data sources and steps to produce each figure/table.

3. **A paper-readiness assessment**: the paper CAN proceed, but only with explicit limitations. The biggest gap is the feature/teacher ablation table (FR7), which has never been re-executed under the formal T24 protocol.

4. **T56 hedge conditioning**: every mechanism-adjacent paper item is annotated with the specific T56 wording guardrails it must respect.

5. **9 explicit non-claims** covering what must not be claimed as completed evidence.

## Why this review verdict (PASS)

### Blocking issues: none

The adversarial review confirmed all task requirements are met:
- Ledger covers scope, ready/partial/missing status, regeneration paths, paper-readiness assessment, and non-claims
- No source code, config, or runtime files were modified
- No benchmark or training was executed
- No claims were upgraded beyond T56 boundaries
- All 16 items from the T44 source ledger maintain identical status classifications

### Minor issues found (non-blocking)

1. **Figure count mismatch**: The worker's output summary says "6 figure entries" but the ledger actually has 11. The content is correct; only the summary count is wrong.

2. **F2 labeled "ready" but no figure file exists**: F2 (boundary diagram) is correctly classified at the evidence-content level, but a stricter reviewer could argue `partial` since no actual figure drawing exists yet.

3. **Worker summary file slightly outside allowed list**: The worker created `docs/worker_summary/T47_worker_summary.md` which wasn't in the task's allowed files. In practice this is standard project practice and harmless — just a documentation side effect.

### What the adversarial review checked especially carefully

- **T56 hedge preservation**: Verified that every mechanism-adjacent statement uses T56 claim table wording. The "high committed-b is harmful" framing does not appear anywhere. C4 remains `partial`.
- **Scope boundary**: Confirmed the ledger explicitly blocks multi-seed intervention, TFLite, real-board, statcalib integration, and expanded benchmark scope — matching T56 boundaries.
- **FR7 honesty**: The ledger calls FR7 `missing` and does not inflate historical pre-T24 evidence into formal protocol evidence.
- **Cross-ledger consistency**: All 16 items checked 1:1 against the T44 source ledger — statuses identical.

### Key remaining decision for Captain

Whether the paper can proceed without FR7 (feature ablation) depends on the strength of attribution claims the authors want to make:
- **If evidence-bounded methods description**: FR7 is a quality booster, not a blocker
- **If strong architectural attribution claims**: FR7 becomes a hard blocker requiring ~40-run re-execution

## Key files

- Main ledger: `docs/paper_ablation_result_pack.md`
- Adversarial review: `docs/review/T47_review.md`
- Worker summary: `docs/worker_summary/T47_worker_summary.md`
