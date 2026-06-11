# T38 Adversarial Review: `seed=20260429` Single-Seed Trace-Export Probe

## Review Metadata

- **Reviewer**: Claude Code adversarial review session
- **Date**: 2026-05-16
- **Task package**: `docs/tasks/Phase2/T38_seed20260429_trace_export_probe.md`
- **Worker files reviewed**:
  - `cnn_fpga/benchmark/analyze_seed20260429_trace.py` (new)
  - `docs/evidence_packs/mechanism_ablation/seed20260429_trace_export_diagnosis.md` (new)
  - `docs/review/T38_review.md` (worker pre-review notes, new)
  - `docs/for_human/T38_explanation.md` (new)
  - `docs/tasks/Phase2/T38_seed20260429_trace_export_probe.md` (updated Worker Output / Verification Record)
  - One T38-scoped run directory: `runs/T38_seed20260429_trace_probe_20260513`

## Verdict

**PASS**

## Blocking Issues

None.

## Non-Blocking Issues

### N1: Unused imports in trace analysis script

- **Location**: `cnn_fpga/benchmark/analyze_seed20260429_trace.py:12`
- `Iterable`, `Mapping` are imported from `typing` but never used.
- **Impact**: Cosmetic only.
- **Classification**: `accepted` as cosmetic.

### N2: Verification record minor imprecision on `missing_runs` format

- **Location**: Task package Verification Record section 2
- Worker writes `summary.json shows missing_runs = 0`; actual JSON value is `missing_runs: []` (empty array, not integer 0).
- **Impact**: Semantically equivalent (both mean "no missing runs"), but the documentation should match the actual artifact format.
- **Classification**: `accepted`. Not a correctness issue.

### N3: Constant `max_abs_delta_b` geometric explanation not explicit in report

- **Location**: `docs/evidence_packs/mechanism_ablation/seed20260429_trace_export_diagnosis.md` Section 5.2
- The report notes `Gated v5 max_abs_delta_b = 0.169705627` is constant across all 8 scenario/repeat pairs but does not explicitly explain that this equals `sqrt(2) * 0.12`, i.e., the L2 norm when both delta_b components simultaneously hit the per-component clip boundary of 0.12.
- **Impact**: A careful reader might find the constant value suspicious (fabricated?) without the geometric explanation. The explanation confirms it is expected: the clip rectangle diagonal.
- **Verification**: `sqrt(2) * 0.12 = 0.169705627...` matches exactly.
- **Classification**: `accepted`. Section 5.3 already discusses clipping counts, so the reader can infer this. But a one-line note would improve clarity.

### N4: Initial timeout and resume execution pattern

- **Location**: Task package Verification Record section 2
- The first worker invocation hit the tool's 1-hour wall-clock timeout. All follow-up commands resumed the same `t3860429_resume` directory rather than starting a second independent probe.
- **Impact**: This is the correct behavior for a resumable benchmark runner. The final state (`missing_runs = []`, 16 raw rows, 8 comparison rows) confirms a single complete probe.
- **Classification**: `accepted`. The worker documented this transparently. The reviewer confirms this is one resumable probe, not multiple independent runs.

## Task Completion Check

### Required trace fields (from task package Section "Required Trace Fields")

| # | Required field | Status | Notes |
|---|---------------|--------|-------|
| 1 | scenario, mode, repeat, seed, window index | `present` | 4798/4798 rows |
| 2 | `teacher_b_q`, `teacher_b_p` | `present` | 4798/4798 rows |
| 3 | predicted `delta_b_q`, `delta_b_p` | `present` | 4798/4798 rows (both raw and applied) |
| 4 | committed `b_q`, `b_p` | `present` | 4798/4798 rows |
| 5 | active bank / staged bank / commit identifier | `present` | commit_target_bank, commit_epoch, commit_version |
| 6 | window-level LER | `present` | 4798/4798 rows |
| 7 | correction-utilization / overflow / saturation proxy | `present` | All present: mean_correction_utilization, overflow_ratio, correction_saturation_ratio, etc. |

All 7 required trace field groups are present at 100% availability (4798/4798 for all 19 checked fields).

### Required output sections (from task package Section "Expected Output")

| # | Required output | Status | Notes |
|---|----------------|--------|-------|
| 1 | Exact command and run directory | Present | Section 2: initial command, resume command, and trace export command |
| 2 | Trace schema and field availability table | Present | Section 3: 19-field availability table, all `present` |
| 3 | Scenario/repeat-level findings | Present | Sections 4-5: per-scenario LER table, per-window trace analysis |
| 4 | Mechanism update matrix for 5 candidates | Present | Section 6: sign offset, magnitude overshoot, teacher instability, CNN residual, committed combined-b |
| 5 | Clear split: supported vs hypotheses | Present | Section 7: 5 supported conclusions, 3 remaining limits |
| 6 | Recommended next bounded task | Present | Section 8: mitigation probe, not further observability |

All 6 required outputs present and substantively complete.

## Scope Compliance

### Allowed files check

`git status --short --untracked-files=all` shows exactly 5 files plus the T38 run root:

1. `docs/tasks/Phase2/T38_seed20260429_trace_export_probe.md` — allowed
2. `cnn_fpga/benchmark/analyze_seed20260429_trace.py` — allowed
3. `docs/for_human/T38_explanation.md` — allowed
4. `docs/review/T38_review.md` — allowed
5. `docs/evidence_packs/mechanism_ablation/seed20260429_trace_export_diagnosis.md` — allowed

Run directory: `runs/T38_seed20260429_trace_probe_20260513` — allowed (one T38-scoped run root per task package).

No files outside the allowed set were modified or created.

### Worker chose the minimal edit path

The task package allowed modifying `slow_loop_runtime.py`, `run_hil_suite.py`, `run_p4_teacher_representation_paired.py`, and creating a new config file. The worker correctly determined that the needed trace fields were already present in `hil_events.json` and did **not** modify any of these files. This is the better choice: no runtime instrumentation was needed.

### Forbidden scope check

| Forbidden action | Verified |
|------------------|----------|
| Modify `docs/02_experiment_plan.md` | No change |
| Train models or add new branch | No change |
| Add statcalib / soft-info / new drift / new baseline / CI stopping | No change |
| Change formal benchmark protocol / frozen set / scenario / seed policy | No change |
| Touch `.tflite` / real TFLite / real-board / hardware / cleanup | No change |
| Rewrite historical `runs/` or `artifacts/` | No change (`git diff --name-only -- runs artifacts` is empty) |
| Present as paper-grade benchmark evidence | Report explicitly states it is not |

All forbidden scope boundaries respected.

## Trace Data Authenticity

### Source verification

The trace script reads from `hil_events.json` (the per-repeat HIL event log), not from `hil_summary.json` (the aggregate summary). It filters events with `kind == "slow_update_finished"` and extracts fields from the event payload's `proposed_params.metadata` subtree.

This is genuine per-window event data recorded during the benchmark run, not reconstructed from final snapshots.

### Numerical spot-check

**Paired repeat comparison** (`paired_repeat_comparison.csv`):

| Scenario | Rep | Full max_delta | Gated max_delta |
|----------|-----|----------------|-----------------|
| linear_ramp | 0 | 0.0254 | 0.1697 |
| linear_ramp | 1 | 0.0265 | 0.1697 |
| step_sigma_theta | 0 | 0.0269 | 0.1697 |
| step_sigma_theta | 1 | 0.0283 | 0.1697 |
| ... | ... | ... | ... |

The constant Gated v5 value of 0.169705627 is geometrically expected: it equals `sqrt(2) * 0.12`, the L2 norm at the clip rectangle diagonal when both components saturate. Confirmed: `sqrt(2) * 0.12 = 0.169705627...`.

**Trace row count**: 4798 total, split 2399/2399 between `hybrid_full` and `hybrid_gated_teacher_v5`. This is consistent with 4 scenarios x 2 repeats x ~300 windows per repeat.

**Rerun completion**: `missing_runs = []` (empty), `raw_rows = 16` (4 scenarios x 2 modes x 2 repeats), `comparison_rows = 8` (4 scenarios x 2 modes).

All spot-checked numbers match underlying artifacts and are internally consistent.

## Causal Claim Discipline

The report correctly:

1. Labels the result as "single-seed diagnostic trace probe" — not formal benchmark
2. Labels `sign offset` as "partially observed but not leading explanation" — sign flips are real but not the sole cause
3. Labels `magnitude overshoot chronology` as "supported" — backed by trace-level delta_b amplitude data
4. Labels `CNN residual output instability` as "supported" — backed by raw vs applied delta comparison
5. Labels `committed combined-b instability` as "strongly supported" — the clearest trace-level explanation
6. Explicitly states limits: does not test mitigation, does not isolate upstream cause, one seed only

The report does **not** claim:
- A paper-grade causal proof
- That formal benchmark boundaries should change
- That statcalib, `.tflite`, or real-board scope is affected

This discipline is correct and consistent with the task package's instructions.

## Missing Validation

No additional validation is required beyond what the worker performed:

1. Static compile check: passed
2. Bounded rerun: completed (`missing_runs = []`, 16 raw rows)
3. Trace export: 4798 rows with 100% field availability
4. Run-dir isolation: confirmed (`git diff --name-only -- runs artifacts` empty)
5. Allowed-file scope: confirmed
6. Report boundary honesty: confirmed

## Suspicious Implementation Details

None found.

- No hardcoded results or fabricated data
- No mock or stub in trace extraction
- The constant `max_abs_delta_b = 0.169705627` is geometrically explained by the clip boundary
- No project runtime imports in the analysis script (standard library only)
- The script writes deterministic CSV/JSON outputs from read-only event data

## Remaining Open Risks

After T38:

- **R10**: Teacher mechanism evidence is significantly narrowed by T38 trace data, but the upstream cause (teacher amplitude vs CNN residual amplitude) is not fully isolated. R10 should be considered substantially narrowed but not fully closed.
- **R20**: Correction saturation remains structurally zero. Not addressed by T38 and not in scope.
- **Single-seed limitation**: T38 evidence is bounded to `seed=20260429`. The mechanism claim should remain seed-bounded.

## Recommended Next Action

The worker's recommended next bounded task (Section 8 of the diagnosis report) is sound:

> Test one minimal mitigation against the same T38 path, for example: lower residual clip / residual scale for Gated v5, or a bounded teacher-delta attenuation variant.

This is the correct next step because:

- T36 narrowed the issue to residual-amplitude / teacher-delta instability
- T38 upgraded that to trace-supported combined committed-b instability
- The next question is now mitigation, not further observability
- A mitigation probe would stay within bounded scope (same seed, same scenarios, same two modes)

Captain should consider this as the next bounded task after T38 closeout, unless higher-priority project needs intervene.
