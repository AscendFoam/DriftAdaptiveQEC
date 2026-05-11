# T28 Independent Review

## Reviewer

Independent Claude Code reviewer (separate from Worker self-review).

## Verdict

`PASS_WITH_WARNINGS`

## Task Summary

T28 aimed to repair teacher diagnostics output semantics so that `not generated`, `not applicable`, and `true zero` are distinguishable in downstream reports and CSVs. The task was bounded: no benchmark expansion, no baseline/scenario changes, no historical evidence rewrite.

The task is substantively complete. The core semantic repair is correct and verified by smoke output. There is one code bug (duplicate markdown report header row) and one minor behavioral subtlety worth noting.

## Blocking Issues

None.

## Non-blocking Issues

### N1: Duplicate markdown report header row in `_write_report()`

**File:** `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py`, lines 318-320

The old header row was not removed when the new header (with `Teacher Diag` column) was added. Both are present:

```
"| Scenario | Mode | ... | Dominant Source | Artifact |",          ← old (11 cols)
"| Scenario | Mode | ... | Dominant Source | Teacher Diag | Artifact |",  ← new (12 cols)
"| --- | --- | ... | --- | --- | --- |",                         ← separator (12 cols)
```

The resulting markdown table will be malformed — two header rows with different column counts.

**Impact:** Non-blocking. The `comparison.csv` is the primary machine-readable output and is correct. The markdown report is secondary. But any future `_write_report()` invocation will produce a broken table.

**Recommendation:** Remove line 318 (the old 11-column header).

### N2: `__pycache__` .pyc files modified as side-effect

Three tracked `.pyc` files changed as a result of code execution:
- `cnn_fpga/benchmark/__pycache__/run_hil_suite.cpython-312.pyc`
- `cnn_fpga/benchmark/__pycache__/run_p4_multiscenario_benchmark.cpython-312.pyc`
- `cnn_fpga/runtime/__pycache__/slow_loop_runtime.cpython-312.pyc`

These are tracked in git history (per T5 governance) and will appear in the diff. This is expected behavior — code changes trigger recompilation — but should not be committed as meaningful changes.

**Recommendation:** Exclude from commit, or accept as incidental per T19 tracked-cache governance.

### N3: `comparison.csv` column order changed

New columns added: `teacher_diagnostics_status`, `teacher_diagnostics_status_reason`, `teacher_diagnostics_support_boundary`, `teacher_diagnostics_generated_repeats`, `teacher_scalar_feature_dim_mean`.

Any downstream consumer expecting the old column order/position will break. This is expected for a semantics repair task but is worth noting for anyone parsing T24-vs-T28 CSVs side-by-side.

**Recommendation:** Accepted. This is an intentional interface change.

## Missing Tests

No tests exist for the changed paths (`test_*slow_loop*`, `test_*hil*` returned empty glob results). The task package acknowledged this with "if available" language. The Worker performed:

1. Static check (`py_compile`) — passed
2. Bounded smoke run (`static_bias_theta` × `ukf` + `hybrid_residual_b`, `repeats=1`, `n_slow_updates=2`) — verified

The smoke output is present and verifiable:
- `runs/p4_benchmark/T28_teacher_diag_semantics_smoke_manual_20260511/comparison.csv`
- `runs/p4_benchmark/T28_teacher_diag_semantics_smoke_manual_20260511/comparison_rows.json`
- `runs/p4_benchmark/T28_teacher_diag_semantics_smoke_manual_20260511/static/hybrid/repeat_00/hil_summary.json`
- `runs/p4_benchmark/T28_teacher_diag_semantics_smoke_manual_20260511/static/ukf/repeat_00/hil_summary.json`

I confirmed the smoke outputs match the Worker's claims:
- `ukf` row: `teacher_diagnostics_status = not_applicable`, numeric teacher fields all `null`/empty
- `hybrid_residual_b` row: `teacher_diagnostics_status = not_generated`, `teacher_scalar_feature_dim_mean = 0.0`, numeric teacher fields all `null`/empty
- `correction_saturation_rate_mean = 0.0` preserved as true zero in both rows
- `teacher_scalar_diagnostics.csv` is header-only

**Recommendation:** Accepted for this bounded repair. Future tasks that modify aggregation logic should add focused unit tests.

## Suspicious Implementation Details

### S1: `_teacher_branch_input_summary` hardcodes `teacher_diagnostics_support_boundary: "scalar_branch_only"`

This is a hardcoded string, not derived from config. However, it accurately documents the current system state — scalar-branch explain diagnostics are the only path that produces teacher diagnostics. This is acceptable as documentation-in-code.

**Verdict:** Not suspicious. This is intentional boundary documentation.

### S2: `generated` path removes `predicted_vector` fallback

The old code computed a `predicted_vector` from the prediction metadata and used it as a fallback for `teacher_contribution_vector`. The new code only uses `explanation.get("teacher_contribution")`. For the `generated` path, if the explanation doesn't contain `teacher_contribution`, the value will be `None` instead of the predicted vector.

**Impact:** Currently zero — no existing code path reaches the `generated` branch (all current paths are `not_generated` or `not_applicable`). If a future scalar-branch teacher path is added, the `teacher_contribution_vector` field behavior will differ from the old code. This is acceptable since the old fallback was arguably incorrect (it conflated prediction output with diagnostic contribution).

**Verdict:** Acceptable behavioral change for a future path that doesn't exist yet.

### S3: `_aggregate_metric` error handling contract change

Old: `float(item[key])` — raises `KeyError` if key missing.
New: `item.get(key)` — returns `None` if key missing, then filtered out.

This is a deliberate change to support null-safe aggregation, consistent with the repair goal. But it silently swallows missing keys instead of failing fast.

**Verdict:** Acceptable for the repair context. The old behavior was equally problematic (it would crash on missing keys rather than reporting them as absent).

## Scope Compliance

### Allowed files — all within scope:

- `cnn_fpga/runtime/slow_loop_runtime.py` ✓
- `cnn_fpga/benchmark/run_hil_suite.py` ✓
- `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py` ✓
- `docs/review/T28_teacher_diagnostics_semantics_repair.md` ✓
- `docs/for_human/T28_explanation.md` ✓
- `docs/tasks/Phase2/T28_teacher_diagnostics_semantics_repair.md` ✓

### Forbidden scope — no violations detected:

- `docs/02_experiment_plan.md` — not modified ✓
- ParamMapper mainline semantics — not modified ✓
- Formal benchmark scenario/baseline/seed — not modified ✓
- Historical `runs/`/`artifacts/` — not rewritten ✓
- No statcalib, soft-information comparator, new drift family, CI-driven stopping, `.tflite` runtime, or real-board path ✓
- No claim that teacher mechanism evidence is fully repaired ✓

### One T28-specific run directory — confirmed:

- `runs/p4_benchmark/T28_teacher_diag_semantics_smoke_manual_20260511`

A second directory `T28_teacher_diag_semantics_smoke_20260511` was started but abandoned due to timeout. The Worker explicitly stated this was not used as final evidence.

## Risk Mapping After T28

| Risk | Before T28 | After T28 |
|------|-----------|-----------|
| R10 | narrowed: broadcast hybrid path doesn't generate scalar diagnostics | further narrowed: now explicitly labeled `not_generated` in output; still open |
| R20 | narrowed: independent fast-loop path | unchanged |
| R21 | open: downstream `0.0` coercion masks missing | closed for current writer semantics |

## Historical Evidence Boundary

- T24 historical run directory `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743` was not modified.
- No existing `runs/` or `artifacts/` outputs were rewritten.
- The new smoke results are isolated to `runs/p4_benchmark/T28_teacher_diag_semantics_smoke_manual_20260511`.

## Recommended Next Action

1. Fix the duplicate markdown header row (N1) — one-line removal.
2. Captain integration: mark T28 complete, update task board, handoff, and risks.
3. Next task priority: T26 (statcalib) or T36 (seed=20260429 failure diagnosis) — now that diagnostics observability is no longer ambiguous, mechanism analysis can proceed without inheriting misleading zero-valued metrics.
