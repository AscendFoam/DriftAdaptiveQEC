# T54: Multi-Seed Trace-Only Generalization Probe

## 1. Status and Scope

This document reports the results of a Phase A multi-seed trace-only generalization probe, following the mechanism-evidence plan defined in `docs/evidence_packs/mechanism_ablation/seed_mechanism_multi_seed_plan.md` (T46).

**Execution date:** `2026-05-22` (started), `2026-05-23` (completed)

**Run root:** `runs/T54_multi_seed_trace_phase_a_20260522/`

**Total trace rows:** 57,586 across 6 seed sources

This task does not:

- run any intervention variant
- modify source code, config, `runs/` outside the T54 root, `artifacts/`, or governance docs
- claim multi-seed confirmation or causal proof from trace evidence alone
- reopen the frozen-set benchmark boundary (T45)

## 2. Exact Command List and Run-Root Structure

### 2.1 Run-root directory

```
runs/T54_multi_seed_trace_phase_a_20260522/
├── seed_reuse_manifest.json
├── cross_seed_analysis.py
├── cross_seed_comparison.csv        (72 rows, 6 seed sources)
├── delta_b_amplitude_by_seed.csv    (72 rows, 6 seed sources)
├── mechanism_summary.csv            (72 rows, 6 seed sources)
├── cross_seed_analysis_summary.json
├── trace_export_s20260425/   (14396 rows, new seed)
├── trace_export_s20260427/   (4800 rows, reused)
├── trace_export_s20260428/   (4798 rows, reused)
├── trace_export_s20260429/   (4798 rows, reused from T38)
├── trace_export_s20260430/   (14396 rows, new seed)
├── trace_export_s20260510/   (14398 rows, new seed)
└── paired_new_seeds/         (prepare + benchmark outputs for new seeds)
```

### 2.2 Commands executed

**Preflight (existing seeds 20260427, 20260428):**

```powershell
# Preflight: verify field availability in existing artifacts
C:\ProgramData\anaconda3\python.exe -c "import json; ..."
```

**Trace export for existing seeds (no rerun):**

```powershell
C:\ProgramData\anaconda3\python.exe -m cnn_fpga.benchmark.analyze_seed20260429_trace \
  --run-dir runs/teachrepr_v5_chunked/p4_benchmark/trp60427_resume \
  --output-dir runs/T54_multi_seed_trace_phase_a_20260522/trace_export_s20260427

C:\ProgramData\anaconda3\python.exe -m cnn_fpga.benchmark.analyze_seed20260429_trace \
  --run-dir runs/teachrepr_v5_chunked/p4_benchmark/trp60428_resume \
  --output-dir runs/T54_multi_seed_trace_phase_a_20260522/trace_export_s20260428
```

**T38 trace reuse (seed 20260429):**

```powershell
cp -r runs/T38_seed20260429_trace_probe_20260513/trace_export \
  runs/T54_multi_seed_trace_phase_a_20260522/trace_export_s20260429
```

**New seeds (20260425, 20260430, 20260510) — full pipeline:**

```powershell
C:\ProgramData\anaconda3\envs\DLEnv\python.exe -m cnn_fpga.benchmark.run_p4_teacher_representation_paired \
  --seed 20260425 --seed 20260430 --seed 20260510 \
  --variant full --variant gated_v5 \
  --benchmark-scenario static_bias_theta \
  --benchmark-scenario linear_ramp \
  --benchmark-scenario step_sigma_theta \
  --benchmark-scenario periodic_drift \
  --repeats 2 \
  --session-root runs/T54_multi_seed_trace_phase_a_20260522 \
  --output-session-dir runs/T54_multi_seed_trace_phase_a_20260522/paired_new_seeds \
  --chunk-repeat-size 2 \
  --skip-existing
```

**Trace export for new seeds:**

```powershell
C:\ProgramData\anaconda3\python.exe -m cnn_fpga.benchmark.analyze_seed20260429_trace \
  --run-dir runs/teachrepr/p4_benchmark/trp60425_resume \
  --output-dir runs/T54_multi_seed_trace_phase_a_20260522/trace_export_s20260425

C:\ProgramData\anaconda3\python.exe -m cnn_fpga.benchmark.analyze_seed20260429_trace \
  --run-dir runs/teachrepr/p4_benchmark/trp60430_resume \
  --output-dir runs/T54_multi_seed_trace_phase_a_20260522/trace_export_s20260430

C:\ProgramData\anaconda3\python.exe -m cnn_fpga.benchmark.analyze_seed20260429_trace \
  --run-dir runs/teachrepr/p4_benchmark/trp60510_resume \
  --output-dir runs/T54_multi_seed_trace_phase_a_20260522/trace_export_s20260510
```

**Cross-seed analysis (all 6 seeds):**

```powershell
C:\ProgramData\anaconda3\python.exe runs/T54_multi_seed_trace_phase_a_20260522/cross_seed_analysis.py
```

## 3. Artifact-Reuse Versus Rerun Matrix

| Seed | Source | Original run dir | Trace rows | Rerun required? | Reason |
| --- | --- | --- | ---: | --- | --- |
| 20260427 | Reused from existing V5 chunked pair | `runs/teachrepr_v5_chunked/p4_benchmark/trp60427_resume` | 4,800 | No | All 19 required trace fields present in existing hil_events.json |
| 20260428 | Reused from existing V5 chunked pair | `runs/teachrepr_v5_chunked/p4_benchmark/trp60428_resume` | 4,798 | No | All 19 required trace fields present in existing hil_events.json |
| 20260429 | Reused from T38 trace probe | `runs/T38_seed20260429_trace_probe_20260513` | 4,798 | No | T38 already validated all required fields; direct copy |
| 20260425 | New rerun | `runs/teachrepr/p4_benchmark/trp60425_resume` | 14,396 | Yes | New seed per T46 seed pack; no existing artifacts |
| 20260430 | New rerun | `runs/teachrepr/p4_benchmark/trp60430_resume` | 14,396 | Yes | New seed per T46 seed pack; no existing artifacts |
| 20260510 | New rerun | `runs/teachrepr/p4_benchmark/trp60510_resume` | 14,398 | Yes | New seed per T46 seed pack; no existing artifacts |

## 4. Field-Availability Summary

All 19 required trace fields are present across all 6 seeds.

| Field | s20260425 | s20260427 | s20260428 | s20260429 | s20260430 | s20260510 |
| --- | --- | --- | --- | --- | --- | --- |
| scenario | present | present | present | present | present | present |
| mode | present | present | present | present | present | present |
| repeat | present | present | present | present | present | present |
| seed | present | present | present | present | present | present |
| window_id | present | present | present | present | present | present |
| teacher_b_q | present | present | present | present | present | present |
| teacher_b_p | present | present | present | present | present | present |
| delta_b_q | present | present | present | present | present | present |
| delta_b_p | present | present | present | present | present | present |
| committed_b_q | present | present | present | present | present | present |
| committed_b_p | present | present | present | present | present | present |
| window_ler | present | present | present | present | present | present |
| overflow_ratio | present | present | present | present | present | present |
| correction_saturation_ratio | present | present | present | present | present | present |

## 5. Per-Seed and Cross-Seed Comparison Summary

### 5.1 Seed-source level outcome summary

Averaged across 4 scenarios and 2 repeats:

| Seed source | Mode | Avg mean LER | Max delta-b | Max committed-b | Gap (Gv5 − Full) |
| --- | --- | ---: | ---: | ---: | ---: |
| 20260425 | hybrid_full | 0.6626 | 0.0088 | 0.0212 | — |
| 20260425 | hybrid_gated_teacher_v5 | 0.6635 | 0.0339 | 0.0165 | +0.0004 |
| 20260427 | hybrid_full | 0.6540 | 0.0162 | 0.0878 | — |
| 20260427 | hybrid_gated_teacher_v5 | 0.5086 | 0.1475 | 0.7390 | −0.1495 |
| 20260428 | hybrid_full | 0.6005 | 0.0270 | 0.2006 | — |
| 20260428 | hybrid_gated_teacher_v5 | 0.5215 | 0.1697 | 0.6348 | −0.0790 |
| 20260429 | hybrid_full | 0.6315 | 0.0284 | 0.2009 | — |
| 20260429 | hybrid_gated_teacher_v5 | 0.5035 | 0.1697 | 0.8727 | −0.1279 |
| 20260430 | hybrid_full | 0.6629 | 0.0088 | 0.0220 | — |
| 20260430 | hybrid_gated_teacher_v5 | 0.4921 | 0.1697 | 0.8729 | −0.1708 |
| 20260510 | hybrid_full | 0.4925 | 0.1697 | 0.8758 | — |
| 20260510 | hybrid_gated_teacher_v5 | 0.4885 | 0.1697 | 0.8758 | −0.0037 |

### 5.2 Delta-b amplitude regime by seed source

| Seed source | Full regime | Full max delta-b | Full clip ratio | Gv5 regime | Gv5 max delta-b | Gv5 clip ratio |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| 20260425 | low | 0.009 | 0.000 | low | 0.034 | 0.000 |
| 20260427 | low | 0.016 | 0.000 | high | 0.148 | 0.937 |
| 20260428 | low | 0.027 | 0.000 | high | 0.170 | 0.425 |
| 20260429 | low | 0.028 | 0.000 | high | 0.170 | 0.742 |
| 20260430 | low | 0.009 | 0.000 | high | 0.170 | 0.900 |
| 20260510 | **high** | **0.170** | **0.963** | high | 0.170 | 0.988 |

### 5.3 Cross-seed outcome summary table

| Seed source | Full avg LER | Gv5 avg LER | Gap | Full regime | Gv5 regime | Category |
| --- | ---: | ---: | ---: | --- | --- | --- |
| 20260425 | 0.6626 | 0.6635 | +0.0004 | low | low | quiet |
| 20260427 | 0.6540 | 0.5086 | −0.1495 | low | high | classic |
| 20260428 | 0.6005 | 0.5215 | −0.0790 | low | high | classic |
| 20260429 | 0.6315 | 0.5035 | −0.1279 | low | high | classic |
| 20260430 | 0.6629 | 0.4921 | −0.1708 | low | high | classic |
| 20260510 | 0.4925 | 0.4885 | −0.0037 | high | high | universal |

### 5.4 Mechanism classification table

| Seed source | Gv5 committed-b instability? | Full committed-b instability? | Outcome | Instability → harm? |
| --- | --- | --- | --- | --- |
| 20260425 | No | No | near-tie | N/A (no instability) |
| 20260427 | Yes | No | Gv5 wins big | No (instability helps) |
| 20260428 | Yes (3/4 scenarios) | No | Gv5 wins | No (instability helps) |
| 20260429 | Yes | No | Gv5 wins (static_bias_theta: Gv5 slightly worse) | Mixed (mostly helps) |
| 20260430 | Yes | No | Gv5 wins big | No (instability helps) |
| 20260510 | Yes | Yes | near-tie | N/A (both unstable) |

## 6. Mechanism Generalization Verdict

### 6.1 Refined finding

The committed-b instability pattern is **broadly repeated with important seed-level variation** across the 6-seed pack.

The 3-seed evidence (20260427, 20260428, 20260429) suggested the instability was "broadly repeated" in a uniform way: Full always stable, Gated v5 always unstable, Gated v5 usually better. The 6-seed evidence reveals a more nuanced picture with three distinct categories:

1. **"Quiet" seed (20260425, 1/6):** No instability in either mode. Both Full and Gated v5 show low delta-b (< 0.04) and low committed-b (< 0.02). Outcome is a near-tie. The Gated v5 gating mechanism barely activates for this seed because teacher_b stays very small throughout all scenarios.

2. **"Classic" seeds (20260427, 20260428, 20260429, 20260430, 4/6):** Full is stable (low delta-b), Gated v5 is unstable (high delta-b > 0.12, high committed-b > 0.63). This is the pattern originally observed in T36/T38 for seed 20260429. Gated v5 wins in 3/4 of these seeds by clear margins. Seed 20260429 is the borderline case where Gv5 is slightly worse in static_bias_theta.

3. **"Universal instability" seed (20260510, 1/6):** Both Full and Gated v5 show high delta-b (0.17) and high committed-b (0.87). The Full variant — previously always stable across other seeds — is itself unstable for this seed. Outcome is a near-tie because both modes perform similarly at high committed-b.

### 6.2 What this means for the original question

T54's original question was: "Does the committed combined-b instability pattern appear outside seed=20260429?"

**Answer: Yes.** The instability pattern appears in 5 of 6 seeds (20260427, 20260428, 20260430, 20260510 definitely; 20260429 is the original). Only seed 20260425 shows no instability.

However, the pattern is not uniform:
- The instability is a property of Gated v5 in 4/6 seeds (classic pattern)
- In 1/6 seeds, both modes are unstable (seed 20260510)
- In 1/6 seeds, neither mode shows instability (seed 20260425)

### 6.3 What this means for the mechanism hypothesis

The committed-b instability hypothesis posits that Gated v5's residual-update path creates high-amplitude delta-b, which leads to large committed-b norms via the teacher_b + delta_b composition, which can degrade performance.

The 6-seed evidence supports this as a partial, not universal, mechanism:

1. **The instability IS systematic in Gated v5** for most seeds (5/6). It is not a seed=20260429 anomaly.

2. **But the instability is NOT always harmful.** In 4/5 seeds with Gv5 instability, Gated v5 outperforms Full. Only in seed 20260429 does the instability correlate with slightly worse performance in one scenario (static_bias_theta).

3. **The instability is NOT exclusive to Gated v5.** Seed 20260510 shows that Full can also develop high committed-b under certain training conditions. The committed-b amplitude depends on the trained model's teacher_b trajectory, not solely on the Gated v5 residual-update mechanism.

4. **One seed is entirely quiet.** Seed 20260425 trains a model with very low teacher_b norms (< 0.02), so the residual-update path never accumulates large committed-b. This shows the instability is conditional on the teacher_b trajectory, not inevitable.

### 6.4 Classification

**Verdict: broadly repeated with qualifications**

The committed-b instability pattern observed in T36/T38 for seed 20260429 generalizes beyond that seed. It is a systematic property of Gated v5 that appears in 5 of 6 tested seeds. However:

- It is not the only instability mode (Full can also be unstable)
- It does not always correlate with worse outcomes
- It is not present in all seeds (1/6 is quiet)

## 7. Recommendation on Later Intervention Task

Based on the 6-seed trace evidence:

1. **Phase B intervention (I1: lower residual clip) remains justified but with tempered expectations.** The high-amplitude delta-b pattern is systematic across most seeds, so a targeted amplitude intervention would test a real mechanism. However, the intervention should not be expected to universally improve outcomes, because the instability already helps in most seeds.

2. **The intervention should account for seed 20260425.** A clip reduction that does nothing on a "quiet" seed is acceptable; a clip reduction that hurts a "classic" seed where Gv5 already wins is the main risk to watch.

3. **Seed 20260510 deserves attention in any intervention study.** If lowering the clip also constrains Full's committed-b (which the raw delta-b data suggests could happen), the intervention might shift the universal-instability regime. Whether this helps or hurts is an empirical question.

4. **The intervention should be tested on all 6 seeds** with the same frozen scenarios and modes.

5. **C4 should remain at `partial`.** The instability is broadly present but not uniformly causal. The mechanism story is more complex than a simple "high committed-b = bad" narrative.

## 8. Explicit Non-Claims

This report does not claim:

1. that multi-seed confirmation of causal mechanism already exists
2. that the committed-b instability hypothesis is proven as root cause
3. that any intervention will succeed
4. that the frozen-set benchmark boundary is being reopened
5. that `.tflite` runtime, real-board validation, or training reproducibility are affected
6. that this probe constitutes causal evidence — it is diagnostic trace evidence only
7. that the three seed categories (quiet, classic, universal) are exhaustive or the only possible regimes
8. that the mechanism generalization verdict is final — it reflects the 6-seed trace evidence available at this time
