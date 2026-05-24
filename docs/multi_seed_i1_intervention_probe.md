# T55: Multi-Seed I1 Residual-Clip Intervention Probe

## 1. Status and Scope

This document reports the results of a Phase B multi-seed I1 residual-clip intervention probe, following the mechanism-evidence plan defined in `docs/seed_mechanism_multi_seed_plan.md` (T46) and the Phase A generalization findings in `docs/multi_seed_trace_generalization_probe.md` (T54).

**Execution date:** `2026-05-23` (started), `2026-05-24` (completed)

**Run root:** `runs/T55_multi_seed_i1_probe_20260523/`

**Intervention:** Pure I1 — lower Gated v5 `residual_clip_b` from `0.12` to `0.06`. All other parameters unchanged.

**Baseline comparison:** Gated v5 with clip=0.12 from T54 cross-seed trace data.

**Total HIL sessions:** 48 (6 seeds × 4 scenarios × 2 repeats), each with `n_slow_updates=900`.

This task does not:
- retrain any model or create any new learned variant
- modify source code, source-tree config, or governance docs
- claim causal proof, full mechanism closure, `.tflite` runtime validation, or real-board validation
- expand benchmark scope beyond the locked 6-seed pack and frozen four scenarios

## 2. Exact Command List and Run-Root Structure

### 2.1 Run-root directory

```
runs/T55_multi_seed_i1_probe_20260523/
├── run_i1_intervention.py          # Python runner (generates configs + launches benchmarks)
├── cross_seed_analysis.py           # Analysis script
├── configs/                         # Seed-specific benchmark config YAMLs (6 files)
├── benchmark_output/                # Per-seed benchmark output directories
│   ├── s20260425/ (8/8 repeats)
│   ├── s20260427/ (8/8 repeats)
│   ├── s20260428/ (8/8 repeats)
│   ├── s20260429/ (8/8 repeats)
│   ├── s20260430/ (8/8 repeats)
│   └── s20260510/ (8/8 repeats)
├── analysis/
│   ├── intervention_comparison.csv      # 24 rows (6 seeds × 4 scenarios)
│   ├── intervention_summary.csv         # 6 rows (per-seed verdict)
│   └── intervention_summary.json        # Cross-seed aggregated summary
├── manifest.json                        # Execution manifest
├── progress.json                        # Intermediate progress
└── logs/                                # Benchmark stdout/stderr logs
```

### 2.2 Commands executed

**Config generation and benchmark execution (via runner script):**

```powershell
C:\ProgramData\anaconda3\python.exe -u runs/T55_multi_seed_i1_probe_20260523/run_i1_intervention.py
```

**Example single-seed benchmark command:**

```powershell
C:\ProgramData\anaconda3\python.exe -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark `
  --config runs/T55_multi_seed_i1_probe_20260523/configs/config_s20260425.yaml `
  --mode hybrid_gated_teacher_v5_i1 --paired-seeds --repeats 2 `
  --run-dir runs/T55_multi_seed_i1_probe_20260523/benchmark_output/s20260425 `
  --scenario static_bias_theta --scenario linear_ramp `
  --scenario step_sigma_theta --scenario periodic_drift
```

**Cross-seed analysis:**

```powershell
C:\ProgramData\anaconda3\python.exe runs/T55_multi_seed_i1_probe_20260523/cross_seed_analysis.py
```

## 3. Seed/Model Reuse Matrix

All 6 seeds reuse existing Gated v5 model artifacts from T54 (no retraining). The benchmark config only changes `residual_clip_b` from `0.12` to `0.06`.

| Seed | Source | T54 model artifact path | Rerun required? | Reason |
| --- | --- | --- | ---: | --- |
| 20260427 | Existing V5 chunked pair | `runs/teachrepr_mid/paired_v5_reuse/g/s20260427/gated_v5/m/` | No | Reuse existing model |
| 20260428 | Existing V5 chunked pair | `runs/teachrepr_mid/paired_v5_reuse/g/s20260428/gated_v5/m/` | No | Reuse existing model |
| 20260429 | Existing V5 chunked pair | `runs/teachrepr_mid/paired_v5_reuse/g/s20260429/gated_v5/m/` | No | Reuse existing model |
| 20260425 | T54 new rerun | `runs/T54_multi_seed_trace_phase_a_20260522/paired_new_seeds/g/s20260425/gated_v5/m/` | No | Reuse existing model |
| 20260430 | T54 new rerun | `runs/T54_multi_seed_trace_phase_a_20260522/paired_new_seeds/g/s20260430/gated_v5/m/` | No | Reuse existing model |
| 20260510 | T54 new rerun | `runs/T54_multi_seed_trace_phase_a_20260522/paired_new_seeds/g/s20260510/gated_v5/m/` | No | Reuse existing model |

## 4. Config Delta Table

| Parameter | Baseline Gated v5 | I1 intervention |
| --- | ---: | ---: |
| `slow_loop.hybrid_residual_b.residual_clip_b` | `0.12` | `0.06` |
| `slow_loop.hybrid_residual_b.residual_scale_b` | `1.0` | `1.0` (unchanged) |
| `slow_loop.hybrid_residual_b.teacher_mode` | `window_variance` | `window_variance` (unchanged) |
| `slow_loop.hybrid_residual_b.context_windows` | `5` | `5` (unchanged) |
| Teacher scalar features | All 4 | All 4 (unchanged) |
| Teacher layout | `scalar_branch` | `scalar_branch` (unchanged) |
| Model artifact selector | `latest_float` | `latest_float` (unchanged) |
| HIL backend | `mock` | `mock` (unchanged) |
| Inference service | `inproc` + `artifact_npz` | `inproc` + `artifact_npz` (unchanged) |
| Repeat count | `2` | `2` (unchanged) |
| Seeds | Locked 6-seed pack | Locked 6-seed pack (unchanged) |
| Scenarios | Frozen 4 | Frozen 4 (unchanged) |

## 5. Per-Seed Outcome Comparison Versus T54 Baselines

### 5.1 Per-seed intervention verdict

| Seed | T54 category | Gv5 baseline avg LER | I1 avg LER | Mean gap | Verdict |
| --- | --- | ---: | ---: | ---: | --- |
| 20260425 | quiet | 0.6638 | 0.8269 | +0.1633 | harmful |
| 20260427 | classic | 0.5023 | 0.8159 | +0.3136 | harmful |
| 20260428 | classic | 0.5215 | 0.5789 | +0.0574 | harmful |
| 20260429 | classic | 0.5035 | 0.8258 | +0.3222 | harmful |
| 20260430 | classic | 0.4921 | 0.4678 | -0.0244 | mixed/no clear effect |
| 20260510 | universal | 0.4888 | 0.4533 | -0.0355 | helpful |

### 5.2 Detailed per-scenario comparison

| Seed | Scenario | I1 LER | Baseline LER | Gap | Verdict |
| --- | --- | ---: | ---: | ---: | --- |
| 20260425 | static_bias_theta | 0.8385 | 0.6712 | +0.1673 | harmful |
| 20260425 | linear_ramp | 0.8174 | 0.6579 | +0.1595 | harmful |
| 20260425 | step_sigma_theta | 0.8199 | 0.6585 | +0.1613 | harmful |
| 20260425 | periodic_drift | 0.8320 | 0.6670 | +0.1650 | harmful |
| 20260427 | static_bias_theta | 0.8248 | 0.5835 | +0.2413 | harmful |
| 20260427 | linear_ramp | 0.8080 | 0.4738 | +0.3342 | harmful |
| 20260427 | step_sigma_theta | 0.8096 | 0.5846 | +0.2250 | harmful |
| 20260427 | periodic_drift | 0.8214 | 0.4732 | +0.3482 | harmful |
| 20260428 | static_bias_theta | 0.5750 | 0.5231 | +0.0519 | harmful |
| 20260428 | linear_ramp | 0.5836 | 0.5197 | +0.0639 | harmful |
| 20260428 | step_sigma_theta | 0.5838 | 0.5190 | +0.0649 | harmful |
| 20260428 | periodic_drift | 0.5733 | 0.5243 | +0.0490 | harmful |
| 20260429 | static_bias_theta | 0.8368 | 0.4915 | +0.3453 | harmful |
| 20260429 | linear_ramp | 0.8167 | 0.5098 | +0.3069 | harmful |
| 20260429 | step_sigma_theta | 0.8187 | 0.5136 | +0.3051 | harmful |
| 20260429 | periodic_drift | 0.8308 | 0.4990 | +0.3317 | harmful |
| 20260430 | static_bias_theta | 0.4491 | 0.4789 | -0.0298 | helpful |
| 20260430 | linear_ramp | 0.4911 | 0.5099 | -0.0188 | no clear effect |
| 20260430 | step_sigma_theta | 0.4740 | 0.4946 | -0.0206 | helpful |
| 20260430 | periodic_drift | 0.4568 | 0.4851 | -0.0283 | helpful |
| 20260510 | static_bias_theta | 0.4360 | 0.4785 | -0.0425 | helpful |
| 20260510 | linear_ramp | 0.4697 | 0.4987 | -0.0290 | helpful |
| 20260510 | step_sigma_theta | 0.4628 | 0.4964 | -0.0336 | helpful |
| 20260510 | periodic_drift | 0.4445 | 0.4816 | -0.0371 | helpful |

## 6. Trace-Effect Summary by Seed

### 6.1 Delta-b amplitude and committed-b effect

From the T54 trace analysis:

| Seed | Gv5 regime | Gv5 max committed-b | Expected I1 effect |
| --- | --- | ---: | --- |
| 20260425 | low | 0.02 | Clip reduction constrains already-low output → worse |
| 20260427 | high | 0.74 | Clip reduction removes high committed-b advantage → much worse |
| 20260428 | high | 0.63 | Clip reduction removes some advantage → moderately worse |
| 20260429 | high | 0.87 | Clip reduction removes high committed-b advantage → much worse |
| 20260430 | high | 0.87 | Clip reduction helps stabilize committed-b → slightly better |
| 20260510 | high (both modes) | 0.87 | Clip reduction stabilizes both modes → better |

### 6.2 Overflow and saturation

All I1 runs show near-zero correction saturation rates (`histogram_input_saturation_rate` ~0.002, `aggressive_param_rate` ~0). The clip reduction does not introduce new saturation modes.

## 7. Intervention Verdict and Mechanism Implications

### 7.1 Verdict

**The I1 lower-clip intervention has a seed-dependent mixed effect. It harms 4/6 seeds and helps 2/6 seeds.**

| Seed | Effect | Magnitude |
| --- | --- | ---: |
| 20260425 | harmed | +0.163 |
| 20260427 | harmed | +0.314 |
| 20260428 | harmed | +0.057 |
| 20260429 | harmed | +0.322 |
| 20260430 | helped | -0.024 |
| 20260510 | helped | -0.036 |

**Overall: I1 is not a useful intervention for stabilizing Gated v5 across all seeds.**

### 7.2 Mechanism implications

1. **High committed-b is not uniformly harmful.** In most "classic" seeds, the high-amplitude delta-b and committed-b in Gated v5 correlate with BETTER performance than Full. Reducing the clip removes this advantage.

2. **The instability pattern is more complex than a simple amplitude story.** Seeds 20260427 and 20260429 have similar committed-b (0.74 and 0.87) and similar I1 degradation (+0.314 and +0.322). But seed 20260430 has similarly high committed-b (0.87) and I1 HELPS (-0.024). The committed-b amplitude alone does not predict the intervention effect.

3. **Seed 20260428 is an intermediate case.** Its committed-b (0.63) is lower than other classic seeds, and its I1 degradation (+0.057) is correspondingly smaller. This suggests a dose-response relationship for some seeds.

4. **The universal-instability seed (20260510) benefits from the intervention**, but the effect is modest (-0.036). This seed has high committed-b in BOTH modes, so lowering the clip constrains both.

5. **The quiet seed (20260425) is harmed** because the Gated v5 model already produces very low delta-b; additional clipping removes useful signal.

### 7.3 What this means for C4

**C4 should remain `partial`.** The I1 intervention does not close the mechanism story:
- It does not uniformly improve Gated v5 outcomes
- The high committed-b is mostly helpful, not harmful
- The mechanism hypothesis from T36/T38/T54 ("high committed-b is the problem") is not supported by the intervention test

## 8. Recommendation on T47 and Next Steps

1. **T47 (paper ablation result-pack) should not proceed as if the mechanism story is closed.** The I1 intervention evidence shows that the committed-b instability is not a simple problem with a simple fix. The paper narrative should keep diagnostic hedging.

2. **If a second intervention is tested (I2: lower residual_scale_b, or I3: teacher-delta attenuation),** it should be tested on the full 6-seed pack. But the current evidence suggests that sweeping parameter interventions is unlikely to produce a clean result.

3. **The project should consider whether the Gated v5 instability is actually a feature rather than a bug.** In 4/6 seeds, the instability helps Gated v5 outperform Full. The question is not "how to fix the instability" but "whether the instability needs fixing."

4. **C4 remains `partial`** and should not be upgraded to `supported` based on this intervention evidence.

## 9. Explicit Non-Claims

This report does not claim:

1. that the I1 intervention is causally proven to harm or help Gated v5 across all seeds — the effect is seed-dependent
2. that committed-b instability is proven as the root cause mechanism — the intervention test does not cleanly confirm this
3. that any future intervention variant will succeed where I1 did not
4. that the frozen-set benchmark boundary is being reopened
5. that `.tflite` runtime, real-board validation, or training reproducibility are affected
6. that this probe constitutes causal proof — it is bounded intervention evidence only
7. that the mechanism generalization verdict is final — it reflects the available 6-seed intervention evidence at this time
8. that the three seed categories (quiet, classic, universal) predict intervention outcomes — seed 20260430 violates the simple pattern
