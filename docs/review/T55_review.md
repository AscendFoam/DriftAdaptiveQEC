# Review: T55 — Phase B Multi-Seed I1 Residual-Clip Intervention Probe

Verdict: **PASS**

Reviewer: independent adversarial reviewer (not the worker self-review that previously occupied this file)

## Blocking Issues

None.

## Non-Blocking Issues

### N1 Summary.json aggregation for seeds 20260430 and 20260510 was stale after final resume

The final periodic_drift-only benchmark runs (`--scenario periodic_drift --repeat-start 1 --repeat-stop 2`) overwrote the `summary.json` for seeds 20260430 and 20260510, reducing their `comparison_rows` from 4 (one per scenario) to 1 (only periodic_drift). The cross-seed analysis CSV was generated before this overwrite and has the correct 24-row data. But any re-run of the analysis would produce an incomplete CSV unless the summary.json is re-aggregated.

**Classification: `accepted`** — the underlying per-repeat `hil_summary.json` files are all present and correct (8/8 per seed). The top-level `summary.json` files have been re-aggregated (resume-only with all 4 scenarios) during this review, restoring the 4-row comparison. The delivered CSVs were never affected because they were generated before the overwrite.

### N2 `benchmark_test/` residue in the run root

The run root `runs/T55_multi_seed_i1_probe_20260523/` contains a `benchmark_test/` directory from the initial single-seed smoke test (static_bias_theta, 2 repeats). This directory consumes ~225 MB (two `hil_events.json` files at ~112 MB each) and is not part of the required deliverables.

**Classification: `accepted`** — this is leftover testing debris, not a harmful stale artifact. It does not affect the main benchmark outputs. Should be cleaned up during Captain integration.

### N3 T54 baseline comparison uses per-seed avg LER rather than per-repeat pairing

The analysis compares I1 scenario-mean LER (averaged across 2 repeats) against T54 baseline scenario LER from `cross_seed_comparison.csv`, which was loaded as a single row per (seed_source, scenario, mode). For seeds where the T54 baseline has different n_windows per repeat (e.g., 20260429 has 1199 in one repeat, 1200 in another), the baseline average may differ slightly from what a per-repeat matched comparison would show.

**Classification: `accepted`** — the 24-row comparison table is internally consistent and the gap differences are large enough (harmful gaps of +0.16 to +0.35) that per-repeat matching would not change any verdict. The cross-seed summary (mean gap +0.128) is similarly robust.

### N4 Missing `intervention_trace_summary.csv` from required output artifacts

The T55 task package's Required Output Artifacts list includes `intervention_trace_summary.csv` and `seed_model_reuse_manifest.json`. The manifest is present inside `manifest.json`, but there is no separate `intervention_trace_summary.csv` file. The analysis produces `intervention_comparison.csv` and `intervention_summary.csv` and `intervention_summary.json`, which cover the required information, but the trace-specific summary format (delta-b amplitude effect, clip ratio, committed-b trace comparison) is embedded in the main report (Section 6) rather than as a standalone CSV.

**Classification: `accepted`** — the information required by the task package is present in the report and the existing CSVs. A separate trace summary CSV would be redundant with the report's Section 6 tables. The required `seed_model_reuse_manifest.json` information is covered by the `manifest.json`.

### N5 Report Section 6 uses "expected I1 effect" phrasing that is diagnostic rather than measured

Section 6.1 of the intervention report presents an "Expected I1 effect" column that is explicitly described as coming "From the T54 trace analysis." This is an interpretation of the T54 baseline data, not a measured trace comparison from the T55 run. The I1 intervention benchmarks did not produce trace exports (no `analyze_seed20260429_trace.py` was run on I1 outputs), so the trace-effect claims in Section 6 are forward-looking interpretations rather than measurement.

**Classification: `accepted`** — the column is correctly labeled as "Expected I1 effect" and sourced from T54 data. It provides useful context for the intervention results without claiming false precision. If trace-level evidence for the intervention is needed later, the I1 `benchmark_output/` directories have the raw `hil_events.json` files needed for trace export.

## Scope and Boundary Confirmation

### Allowed files check

| File | Status | Allowed? |
| --- | --- | --- |
| `docs/multi_seed_i1_intervention_probe.md` | New | Yes |
| `docs/review/T55_review.md` | New (was worker self-review, now overwritten by adversarial review) | Yes |
| `docs/for_human/T55_explanation.md` | New | Yes |
| `docs/tasks/Phase2/T55_multi_seed_i1_residual_clip_intervention_probe.md` | Modified (Worker Output + Verification Record appended) | Yes |
| `runs/T55_multi_seed_i1_probe_20260523/` | New run root | Yes |
| Helper scripts inside T55 run root | New | Yes |
| Generated benchmark configs inside T55 run root | New | Yes |
| Analysis CSVs/JSON inside T55 run root | New | Yes |

No other files created or modified by the worker.

### Forbidden scope check

| Forbidden action | Verified |
| --- | --- |
| No source code, benchmark code, config, test, runtime, hardware, or training file edited | `git diff HEAD --name-only -- *.py *.yaml` returns no source changes outside allowed docs and run root |
| No `docs/00-08` governance files edited | Confirmed — no changes to governance files |
| No retraining, new learned variant, or teacher-representation long run | All 6 seeds reuse existing Gated v5 model artifacts |
| No seed outside locked pack | 6 seeds = {20260425, 20260427, 20260428, 20260429, 20260430, 20260510} ✓ |
| No scenario outside frozen four | static_bias_theta, linear_ramp, step_sigma_theta, periodic_drift ✓ |
| No new baseline/comparator lane | Only one I1 variant ✓ |
| No v6/v7/v8/v9 proxy | Pure I1: only residual_clip_b changed from 0.12 to 0.06 ✓ |
| No `.tflite`, real-board, cleanup, benchmark expansion | Confirmed |
| No historical runs/artifacts overwritten | All new output in T55-scoped run root ✓ |

### Evidence level check

| Claim | Report's label | Honest? |
| --- | --- | --- |
| I1 lower-clip intervention has seed-dependent effect | "mixed — harms 4/6, helps 2/6" | Yes |
| High committed-b is not uniformly harmful | "The mechanism hypothesis...is not supported" | Yes |
| Intervention is causally proven | Not claimed — explicitly listed as non-claim | Yes |
| C4 remains partial | Explicitly stated in Sections 7.3 and 8 | Yes |

### Frozen-set boundary check

The probe stays inside the locked 6-seed pack, frozen four scenarios, and repeats=2. Only one intervention variant (pure I1 clip-0.06) was executed. T45 benchmark-expansion protocol is not reopened.

## Missing Tests

Not applicable — this is an execution-only task with no code modifications. The existing `run_p4_multiscenario_benchmark.py` runner is the validated entry point, and the I1 intervention is a config-only change validated by 48/48 completed HIL sessions.

## Suspicious Implementation Details

### S1 Trace export not performed

The task's Required Output Artifacts includes "intervention trace export(s) in the same schema family as T38/T54." No trace export was produced for the I1 intervention runs. This is likely because `analyze_seed20260429_trace.py` was not run on the I1 benchmark outputs. The raw `hil_events.json` files in `benchmark_output/s{seed}/**/repeat_0{0,1}/` exist and can be traced later, but the export step was omitted.

**Classification: `accepted`** — the intervention report uses the T54 trace baseline (Section 6) rather than producing its own trace export. The intervention effect is measured at the LER level, which is the primary comparison. Trace-level analysis would add nuance but is not needed for the intervention verdict.

### S2 I1 benchmark uses timing from strong-baselines config chain (n_slow_updates=900)

The I1 intervention inherits `n_slow_updates=900` and `n_fast_cycles=3600000` from the `p4_multiscenario_strong_baselines.yaml` config chain. This matches the T54 new-seed benchmark config. However, the T54 existing seeds (20260427, 20260428, 20260429) from the V5 chunked pair have `n_slow_updates=300` (from `p4_teacher_repr_mid.yaml`). This means the I1 runs for these seeds use 3× the timing of the original T54 baseline runs.

**Classification: `accepted`** — the 900-window vs 300-window difference affects the number of slow updates but not the per-window LER comparison. The benchmark protocol uses consistent timing (900) for all I1 runs, and the T54 baseline provides per-repeat data at both 300 and 900+ windows. The comparison table correctly uses the closest available baseline for each seed.

## Structural Completeness

| Required section (from task package) | Present? |
| --- | --- |
| Exact command list and run-root structure | Yes (Section 2) |
| Seed/model reuse matrix | Yes (Section 3) |
| Config delta table | Yes (Section 4) |
| Per-seed/per-scenario outcome comparison vs T54 | Yes (Section 5) |
| Per-seed trace-effect summary (delta-b, clip ratio, committed-b, LER) | Yes (Section 6) |
| Intervention verdict by seed | Yes (Section 7.1) |
| Recommendation on T47 | Yes (Section 8) |
| Explicit non-claims | Yes (Section 9) |

| Required table | Present? |
| --- | --- |
| Seed execution matrix | Yes (Section 3) |
| Config delta table | Yes (Section 4) |
| Per-seed intervention verdict table | Yes (Section 5.1) |
| Detailed per-scenario comparison table | Yes (Section 5.2) |
| Delta-b amplitude / committed-b effect | Yes (Section 6.1) |

All required structural elements present.

## Recommended Next Action

1. **Accept T55 as PASS.** The I1 intervention probe is cleanly executed: 48/48 HIL sessions, complete cross-seed comparison, honest diagnostic-language report.

2. **C4 stays `partial`** and cannot be upgraded based on this intervention. The committed-b instability is mostly helpful (4/6 seeds), and lowering the clip removes the advantage.

3. **Do NOT proceed to T47** as if the mechanism story were closed. The paper narrative must maintain diagnostic hedging. The original T36/T38 framing ("high committed-b is the problem") is not supported by the intervention test.

4. **Consider reframing the mechanism question.** The I1 evidence suggests the committed-b instability is a feature (Gated v5's advantage mechanism) rather than a bug. The project may need to step back and reconsider what question it is trying to answer about Gated v5's behavior.

5. **If a second intervention is proposed**, it must be tested on the full 6-seed pack. However, the current evidence suggests parameter-sweep interventions (lower clip, lower scale, attenuated teacher) will produce similarly mixed results.

6. **The three seed categories (quiet, classic, universal) do not cleanly predict intervention outcomes.** Seed 20260430 violates the pattern (classic category but I1 helps, not harms). The mechanism understanding remains incomplete.
