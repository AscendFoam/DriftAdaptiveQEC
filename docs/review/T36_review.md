# T36 Adversarial Review: `seed=20260429` Failure-Mechanism Diagnosis

## Review Metadata

- **Reviewer**: Claude Code adversarial review session
- **Date**: 2026-05-13
- **Task package**: `docs/tasks/Phase2/T36_seed20260429_failure_mechanism_diagnosis.md`
- **Worker files reviewed**:
  - `docs/evidence_packs/mechanism_ablation/seed20260429_failure_diagnosis.md` (new)
  - `cnn_fpga/benchmark/analyze_seed20260429_failure.py` (new)
  - `docs/review/T36_review.md` (worker pre-review input, new)
  - `docs/for_human/T36_explanation.md` (new)
  - `docs/tasks/Phase2/T36_seed20260429_failure_mechanism_diagnosis.md` (updated Worker Output / Verification Record sections)

## Verdict

**PASS**

## Blocking Issues

None.

## Non-Blocking Issues

### N1: Unused import `Iterable` in analysis script

- **Location**: `cnn_fpga/benchmark/analyze_seed20260429_failure.py:7`
- `from typing import Dict, Iterable, List, Mapping` — `Iterable` is imported but never used.
- **Impact**: Cosmetic only. No functional or correctness impact.
- **Classification**: `accepted` as cosmetic. Does not warrant rework.

### N2: Hardcoded scenario/mode folder name mappings in script

- **Location**: `cnn_fpga/benchmark/analyze_seed20260429_failure.py:32-43`
- `SCENARIO_FOLDERS` and `MODE_FOLDERS` are hardcoded dictionaries mapping canonical names to directory names. These would break silently if the run directory structure changed.
- **Impact**: Low. This is a bounded diagnostic script targeting existing frozen artifacts, not a production tool. The mappings are correct for the current artifact set (verified: all 16 repeat `hil_summary.json` paths resolve).
- **Classification**: `accepted` for a bounded diagnostic script. A future reusable analysis tool would need dynamic discovery.

### N3: Worker pre-review file shares the same filename as this adversarial review

- The worker wrote `docs/review/T36_review.md` as a pre-review input, which this adversarial review overwrites.
- **Impact**: Minor. The worker's verification record and scope confirmation are fully preserved in the task package's Worker Output / Verification Record section.
- **Classification**: `accepted`. The task package is the primary record; the review file is the reviewer's output.

## Task Completion Check

### Required outputs (from task package Section "Expected Output")

| # | Required output | Status | Notes |
|---|----------------|--------|-------|
| 1 | Evidence inventory with exact artifact paths | Present | Section 2 of diagnosis report; all 11+ paths verified to exist on disk |
| 2 | `seed=20260429` Full vs Gated v5 summary by scenario | Present | Section 3.2 of diagnosis report; numbers verified against `comparison.csv` |
| 3 | Cross-seed comparison against 20260427 and 20260428 | Present | Sections 3.1 and 3.3; verified against `summary.csv` |
| 4 | Mechanism matrix with evidence labels | Present | Section 5; five candidate mechanisms labeled with evidence levels |
| 5 | Clear conclusion split (supported / hypotheses / not answerable) | Present | Section 6; three-part split |
| 6 | Recommended next bounded task | Present | Section 7; trace-export follow-up for single-seed per-window analysis |

All six required outputs are present and substantively complete.

## Scope Compliance

### Allowed files check

`git status --short --untracked-files=all` shows exactly 5 files, all within the allowed set:

1. `docs/tasks/Phase2/T36_seed20260429_failure_mechanism_diagnosis.md` — allowed
2. `cnn_fpga/benchmark/analyze_seed20260429_failure.py` — allowed
3. `docs/for_human/T36_explanation.md` — allowed
4. `docs/review/T36_review.md` — allowed
5. `docs/evidence_packs/mechanism_ablation/seed20260429_failure_diagnosis.md` — allowed

No files outside the allowed set were modified or created.

### Forbidden scope check

| Forbidden action | Verified |
|------------------|----------|
| Modify `docs/02_experiment_plan.md` | No change detected |
| Run any benchmark, training, `.tflite`, hardware, or cleanup command | Script is read-only; `git diff --name-only -- runs artifacts` is empty |
| Create or rewrite `runs/` or `artifacts/` | No change |
| Modify model code, configs, benchmark runner, protocol, baseline/scenario set | No change |
| Add a new teacher-representation branch | No change |
| Add statcalib, soft-information comparator, new drift family, or CI-driven stopping | No change |
| Claim the diagnosis is causal if evidence only supports hypothesis | Report explicitly labels hypotheses and not-answerable items |

All forbidden scope boundaries respected.

## Evidence Verification

### Artifact path existence

All referenced artifact paths were verified to exist:

- `runs/teachrepr_v5_chunked_pair/paired_20260427_220702/summary.csv` — EXISTS
- `runs/teachrepr_v5_chunked_pair/paired_20260427_220702/summary.json` — EXISTS
- `runs/teachrepr_v5_chunked/p4_benchmark/trp60429_resume/summary.json` — EXISTS
- Cross-seed `comparison.csv` for all 3 seeds — EXISTS
- Cross-seed `teacher_scalar_diagnostics.csv` for all 3 seeds — EXISTS
- Legacy `trp60429_20260427_142013_2a59bc_24060/comparison.csv` — EXISTS
- All 16 repeat-level `hil_summary.json` files for 20260429 — EXISTS

### Numerical spot-check

Paired summary average gap (`summary.csv`):

- 20260427: gap = -0.186055 — matches report Section 3.1
- 20260428: gap = -0.238206 — matches report Section 3.1
- 20260429: gap = +0.002358 — matches report Section 3.1

20260429 per-scenario LER (`comparison.csv`):

- `static_bias_theta`: Full = 0.633164, Gated v5 = 0.645085 — matches report Section 3.2
- `linear_ramp`: Full = 0.637529, Gated v5 = 0.631815 — matches report Section 3.2
- `step_sigma_theta`: Full = 0.636237, Gated v5 = 0.639388 — matches report Section 3.2
- `periodic_drift`: Full = 0.642520, Gated v5 = 0.642593 — matches report Section 3.2

All spot-checked numbers match the underlying artifacts.

### Analysis script

- **Imports**: `csv`, `json`, `math`, `pathlib.Path`, `typing.{Dict, Iterable, List, Mapping}` — all standard library. No project runtime imports.
- **Determinism**: Script reads files and prints JSON to stdout. No file writes, no RNG, no side effects.
- **Execution**: Ran successfully, producing valid JSON output.

## Causal Claim Discipline

The report correctly:

1. Labels `sign offset` as `not answerable` — appropriate, because no per-window b trace exists
2. Labels `magnitude overshoot` as `plausible / partially supported` — appropriate, based on final-snapshot evidence only
3. Labels `response lag` as `not supported` — appropriate, scheduler stats match between modes
4. Labels `teacher instability` as `partially supported` — appropriate, with the correct caveat that 20260427 also has strong teacher activity
5. Labels `gated branch too conservative` as `not supported` — appropriate, final |b| is larger not smaller

The report does not claim:
- A full causal proof
- That formal benchmark boundaries should change
- That statcalib, `.tflite`, or real-board scope is affected

This discipline is correct and consistent with the task package's instructions.

## Missing Validation

No additional validation is required beyond what the worker already performed. The task is a read-only analysis, not a code change. The verification record in the task package documents:
1. Script execution — passed
2. Compile check — passed
3. No benchmark run directory — confirmed
4. Allowed-file scope — confirmed
5. Documentation honesty — confirmed

## Suspicious Implementation Details

None found. The script is straightforward CSV/JSON reading with arithmetic aggregation. No mock data, no stubs, no hardcoded results, no hidden simulation.

## Remaining Open Risks

These risks remain after T36, as the worker correctly noted:

- **R10**: Teacher mechanism evidence is still incomplete at the full time-series level. T36 narrows the hypothesis but does not close R10.
- **R20**: Correction saturation remains structurally zero. This is a separate concern and was correctly excluded from the 20260429 diagnosis.
- **Trace gap**: No per-window committed-parameter trace exists, so sign offset and overshoot chronology remain unanswerable from current artifacts.

## Recommended Next Action

The worker's recommended next bounded task (Section 7 of the diagnosis report) is sound:

> Add a trace-export path for one bounded rerun of `seed=20260429` only, with unchanged benchmark semantics, exporting per-window `teacher_b`, predicted `delta_b`, committed `b`, and window-level LER.

This is the correct follow-up because:
- Current artifacts are sufficient to narrow the problem to residual-amplitude / teacher-delta regime instability
- They are not sufficient to distinguish sign offset, overshoot chronology, and exact source attribution
- A single-seed trace export would directly answer the missing question without widening the benchmark

Captain should consider this as the next bounded task after T36 closeout, unless higher-priority project needs intervene.
