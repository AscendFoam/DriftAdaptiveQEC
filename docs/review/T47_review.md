# Review: T47 — Paper Ablation Result-Pack and Material Ledger

Verdict: **PASS**

Reviewer: adversarial review (replaces worker self-review per CLAUDE.md §5.3)

## Blocking Issues

None.

## Non-Blocking Issues

### N1: Worker Output miscounts figure entries

The Worker Output section in the task package states "6 figure entries" but the ledger ([docs/paper_materials/paper_ablation_result_pack.md](../paper_materials/paper_ablation_result_pack.md) §2.1) actually contains **11 figure entries** (F1, F2, F3, FR1, FR4, FR6, FR7, FR8, FR10, FR11, FR12). Table count of 5 is correct (T1–T5).

**Impact:** The ledger content itself is correct and complete; only the summary count in the Worker Output block is wrong. Does not affect any downstream workflow. **Accepted** — cosmetic.

### N2: Worker summary created outside strict Allowed Files boundary

Task package `Allowed files` lists only 4 files:
- `docs/tasks/Phase2/T47_paper_ablation_result_pack_and_material_ledger.md`
- `docs/paper_materials/paper_ablation_result_pack.md`
- `docs/review/T47_review.md`
- `docs/for_human/T47_explanation.md`

The worker additionally created `docs/worker_summary/T47_worker_summary.md`, which is not in the allowed set.

**Impact:** This is consistent with the project's de facto practice (every prior task has a worker summary). No source, config, runtime, or governance file was touched. The worker summary is a docs-only side effect that aids handoff continuity. **Accepted** — consider adding `docs/worker_summary/` to future task package allowed files if worker summaries are expected.

### N3: F2 (boundary diagram) labeled `ready` despite no actual figure file

F2 is classified `ready`, but the regeneration path says "Schematic figure; draw.io or equivalent based on boundary descriptions in source docs." No figure file or generation script exists.

**Strict reading:** `ready` should mean "the figure can be produced from the evidence pack right now, no new data needed." The content description is fully determined, so the classification is defensible — the *evidence content* is ready even if the *figure file* isn't drawn yet. However, a stricter reviewer could argue `partial`.

**Impact:** Minimal. The regeneration path honestly describes the remaining manual step. If the paper pack later requires actual figure files, this item will need a drafting step. **Accepted** — content-level readiness is the correct granularity for this ledger.

### N4: F3 labeled `blocked` — consistent with source ledger, but worth noting

The T44 source ledger also labels F3 as `blocked`. The T47 ledger preserves this classification. A future decision point: if the paper needs any training portability visualization, F3 would require an unblocking task first. Correctly flagged as blocked.

**Accepted** — faithful carry-forward from source ledger.

### N5: Regeneration paths are conceptual rather than executable

The "Concrete steps" column in §3 describes regeneration at the concept level ("Read comparison.csv, group by scenario, rank by final_ler_mean") without producing executable scripts. This is appropriate for a docs-only ledger task, but the regeneration paths cannot be executed as-is.

**Impact:** A future material-pack task will need to convert these descriptions into actual scripts or manual procedures. The ledger correctly identifies the data sources and steps, so the conversion is mechanical. **Accepted** — the task boundary explicitly forbids code/script creation.

## Missing Tests

Not applicable — T47 is a docs-only ledger task with zero code changes.

## Suspicious Implementation Details

### 1. Claim integrity verified

All mechanism-facing statements are bounded by T56 claim table wording:
- C4 remains `partial` throughout
- "high committed-b is harmful" framing does not appear in any output
- Every mechanism-adjacent ledger row includes a T56 hedge note column
- Section 4 maps each paper section to specific T56 constraints
- 9 explicit non-claims cover ablation completeness, mechanism closure, second intervention, and paper-readiness overclaim

### 2. No scope escalation

The ledger's "What is out of scope" list explicitly blocks:
- Multi-seed intervention figure (deferred by T56)
- True `.tflite` runtime figure (blocked by R12)
- Real-board evidence (blocked by R13, R14)
- Statcalib integrated comparator (blocked by R24)
- Cross-platform reproducibility (blocked by R11)
- Expanded benchmark beyond frozen set (blocked by R5, R9)

This preserves all existing evidence boundaries.

### 3. FR7 gap not hidden

FR7 (feature/teacher ablation) is correctly classified as `missing`. The regeneration path explicitly states "New execution needed: ~40 runs" and notes that historical pre-T24 evidence cannot substitute. No attempt to inflate historical evidence into completed formal protocol evidence.

### 4. Cross-ledger consistency check

Compared against `docs/reality_recovery/04_figure_and_result_ledger.md` (frozen 2026-05-19):

| Item | Source status | T47 status | Delta |
|------|-------------|-----------|-------|
| F1 | partial | partial | ✓ Preserved |
| F2 | ready | ready | ✓ Preserved |
| F3 | blocked | blocked | ✓ Preserved |
| FR1 | partial | partial | ✓ Preserved |
| FR4 | partial | partial | ✓ Preserved |
| FR6 | missing | missing | ✓ Preserved |
| FR7 | missing | missing | ✓ Preserved |
| FR8 | missing | missing | ✓ Preserved |
| FR10 | blocked | blocked | ✓ Preserved |
| FR11 | blocked | blocked | ✓ Preserved |
| FR12 | partial | partial | ✓ Preserved |
| T1 | ready | ready | ✓ Preserved |
| T2 | ready | ready | ✓ Preserved |
| T3 | ready | ready | ✓ Preserved |
| T4 | ready | ready | ✓ Preserved |
| T5 | partial | partial | ✓ Preserved |

All 16 items maintain identical status. T47 correctly adds regeneration paths and T56 hedge annotations without changing any classification.

### 5. Git diff verification

Only the task package file was modified (22 lines added for Worker Output and Verification Record). Three new files were created:
- `docs/paper_materials/paper_ablation_result_pack.md`
- `docs/review/T47_review.md`
- `docs/for_human/T47_explanation.md`

No `.py`, `.yaml`, `.yml`, run-root, or governance files were touched. Zero executable changes.

## Recommended Next Action

1. **Accept T47 as PASS.** The task goal is met, all expected outputs are produced, and no evidence boundaries are violated.

2. **Correct the figure count** in the Worker Output block (6 → 11) for accuracy, though this is cosmetic.

3. **Decide on FR7 disposition.** The ledger correctly identifies FR7 as the largest gap. Captain should decide whether to:
   - Spawn a bounded FR7 re-execution task (5 variants × 4 scenarios × 2 repeats under T24 protocol), or
   - Accept the gap and proceed with the paper in evidence-bounded form (explicit limitation wording on architectural attribution claims).

4. **Set next unique task.** FR7 disposition decision drives the next task. Candidates:
   - If FR7 is needed: bounded feature ablation re-execution task (new T#)
   - If FR7 is accepted as gap: next paper-assembly or reproducibility task
   - Current roadmap candidates include T50 (Training reproducibility / material-regeneration pack) or paper re-open gate tasks (T51/T52)
