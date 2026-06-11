# Review: T54 — Phase A Multi-Seed Trace-Only Generalization Probe

Verdict: **PASS**

Reviewer: independent adversarial reviewer (not the worker self-review that previously occupied this file)

## Blocking Issues

None.

## Non-Blocking Issues

### N1 New benchmark run dirs created outside the T54-scoped run root

The paired runner's default `--benchmark-output-root` creates benchmark output dirs (`trp60425_resume`, `trp60430_resume`, `trp60510_resume`) under `runs/teachrepr/p4_benchmark/` rather than inside `runs/T54_multi_seed_trace_phase_a_20260522/`. This means the new seed benchmark artifacts are outside the single T54-scoped run root.

Classification: `accepted` — these are new directories, not modifications to historical runs. The trace exports, analysis CSVs, and cross-seed summaries are all inside the T54 root. No historical data was overwritten. The runner's output convention is an upstream design choice, not a Worker scope violation.

### N2 Seed reuse manifest has incomplete metadata for new seeds

`seed_reuse_manifest.json` records `trace_row_count: null` and `field_availability: null` for seeds 20260425, 20260430, and 20260510 (the three new rerun seeds), suggesting the manifest was created before the trace export completed. The final report in `docs/evidence_packs/mechanism_ablation/multi_seed_trace_generalization_probe.md` Section 4 correctly reports all fields present for all seeds, and Section 3 reports the exact row counts.

Classification: `accepted` — the manifest documents the pre-rerun decision state honestly (rerun_required: true). The final report has the complete metadata. No reader could be misled.

### N3 Cross-seed analysis uses data from two different paired-seed conventions

Existing seeds (20260427, 20260428) were run with the V5 chunked paired benchmark's paired-seed convention (repeat 0 = base seed, repeat 1 = base seed + 1). New seeds (20260425, 20260430, 20260510) were run with `--paired-seeds` using a different pairing convention. The cross-seed analysis correctly groups by the actual seed value per trace row, but the underlying paired-seed designs differ across sources.

Classification: `accepted` — the analysis correctly uses actual seed values, not pair groupings. The mechanism generalization conclusion is not affected by the different pairing conventions. This is a methodological note for anyone extending the analysis.

### N4 `cross_seed_analysis.py` is a helper script, not a CSV/JSON summary

The task package's Required Output Artifacts specify "derived CSV/JSON summaries only inside that T54-scoped run root." The Worker additionally created `cross_seed_analysis.py` inside the run root. While this is a Python script rather than a data file, it reads existing trace CSVs and produces summary CSVs — it does not modify any source code path or benchmark runner.

Classification: `accepted` — the script is correctly placed inside the run root (not in `cnn_fpga/` or the source tree). It does not modify any code path. It is a run-root-local analysis helper.

### N5 Mechanism verdict refined rather than simply confirmed

The initial 3-seed analysis (based on existing seeds only) produced a verdict of "broadly repeated." After all 6 seeds were included, the verdict was refined to "broadly repeated with qualifications," acknowledging (a) one quiet seed with no instability, (b) one seed where Full is also unstable, and (c) that the instability is not always harmful.

Classification: `accepted` — this refinement is honest and increases the scientific value of the report. The original question ("does the pattern appear beyond seed=20260429?") is answered affirmatively with appropriate nuance. The refinement shows the Worker did not cherry-pick the initial verdict.

## Scope and Boundary Confirmation

### Allowed files check

| File | Status | Allowed? |
| --- | --- | --- |
| `docs/evidence_packs/mechanism_ablation/multi_seed_trace_generalization_probe.md` | New | Yes |
| `docs/review/T54_review.md` | New (worker self-review, now overwritten by adversarial review) | Yes |
| `docs/for_human/T54_explanation.md` | New | Yes |
| `docs/tasks/Phase2/T54_multi_seed_trace_only_generalization_probe.md` | Modified (Worker Output + Verification Record appended) | Yes |
| `runs/T54_multi_seed_trace_phase_a_20260522/` | New run root | Yes |

### Forbidden scope check

| Forbidden action | Verified |
| --- | --- |
| No source code, benchmark code, config, test, runtime, hardware, or training file edited | Confirmed — `git diff HEAD --name-only -- *.py *.yaml *.yml` returns empty (`.claude/settings.json` change is pre-existing and unrelated to T54) |
| No `docs/00-08` governance files edited | Confirmed — `git diff HEAD -- docs/00* docs/01* docs/03* docs/04* docs/05* docs/06* docs/07* docs/08* docs/02_experiment_plan.md` returns empty |
| No historical `runs/` or `artifacts/` path overwritten | Confirmed — new benchmark dirs under `runs/teachrepr/p4_benchmark/` are new directories |
| No new baselines, new scenarios, or seeds outside locked pack | Confirmed — only Full and Gated v5, 4 frozen scenarios, 6 seeds from T46 locked pack |
| No intervention variants run | Confirmed — trace-only probe |
| No upgrade of trace evidence to causal proof | Grep confirms "causal proof" / "mechanism proven" / "root cause identified" / "multi-seed confirmation" appear only in non-claims sections or negative context |
| No `.tflite` runtime, real-board validation, or training reproducibility affected | Confirmed — scope stays within trace-only diagnostic |

### Evidence level check

| Claim | Report's label | Honest? |
| --- | --- | --- |
| Committed-b instability on seed=20260429 | `trace-supported diagnostic` | Yes |
| Instability generalizes to other seeds | `broadly repeated with qualifications` (diagnostic) | Yes — 5/6 seeds show Gv5 instability, 1 quiet, 1 universal |
| Instability is the root cause of worse outcomes | `unsupported — mostly helps, only 20260429 static_bias_theta worse` | Yes |
| Mechanism is causal | `unsupported` | Yes |
| Intervention would help | `justified but tempered expectations` | Yes — correctly tempered |

### Frozen-set boundary check

The probe uses only Full and Gated v5 modes with the existing frozen four scenarios (static_bias_theta, linear_ramp, step_sigma_theta, periodic_drift). No new baselines, scenarios, or benchmark code changes. T45 is not reopened.

## Missing Tests

Not applicable — this is an execution task with trace-only diagnostic output. No code was modified, so no code tests are required. The Worker correctly used the existing validated T38 trace-export path.

## Suspicious Implementation Details

None. The analysis correctly uses the existing T38 trace-export path for existing seeds and runs the full paired pipeline for new seeds. The seed reuse decisions are honest and documented in `seed_reuse_manifest.json`. The cross-seed analysis correctly handles the paired-seed convention by using actual seed values per trace row.

## Structural Completeness

| Required section (from task package) | Present? |
| --- | --- |
| Exact command list and run-root structure | Yes (Section 2) |
| Artifact-reuse versus rerun matrix | Yes (Section 3) |
| Field-availability summary | Yes (Section 4) |
| Per-seed and cross-seed comparison summary | Yes (Section 5) |
| Mechanism generalization verdict | Yes (Section 6) |
| Recommendation on later intervention task | Yes (Section 7) |
| Explicit non-claims | Yes (Section 8) |

| Required table | Present? |
| --- | --- |
| Seed execution matrix | Yes (Section 3) |
| Field-availability table | Yes (Section 4) |
| Cross-seed outcome summary table | Yes (Section 5.3) |
| Mechanism classification table | Yes (Section 5.4) |

All required structural elements are present.

## Recommended Next Action

1. **Accept T54 as PASS.** This is a clean execution of the T46 Phase A plan. The committed-b instability pattern is confirmed to generalize beyond seed=20260429, and the nuanced classification (quiet / classic / universal) is a genuine scientific finding.

2. **If Phase B intervention (I1: lower residual clip) proceeds**, test on all 6 seeds with the same frozen scenarios and modes. Pay particular attention to seed 20260425 (quiet — intervention may be irrelevant) and 20260510 (universal — intervention may affect both modes).

3. **Keep C4 at `partial`.** The mechanism story is supported but more complex than T36/T38's single-seed picture suggested:
   - The instability helps Gv5 in most classic seeds, contradicting a simple "high committed-b = bad" narrative
   - The universal-instability seed (20260510) shows that committed-b amplitude isn't Gv5-exclusive
   - The quiet seed (20260425) shows the mechanism depends on the trained model's teacher_b trajectory

4. **Do not treat the three seed categories (quiet, classic, universal) as fixed regimes.** They reflect the 6-seed sample and may shift with additional seeds or different training conditions.

5. **Consider whether the 20260510 universal-instability pattern could be a separate issue** — it may be a Full-mode problem (or training artifact) that coincidentally overlaps with the Gv5 committed-b instability.
