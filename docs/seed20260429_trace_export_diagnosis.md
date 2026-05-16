# `seed=20260429` Trace Export Diagnosis

## 1. Scope And Boundary

This is a bounded T38 diagnostic rerun for `seed=20260429`.

- It keeps the existing `Full` vs `Gated v5` benchmark semantics.
- It does not add a new branch, new baseline, new scenario family, `.tflite`, or real-board scope.
- It uses one T38-scoped run root only:
  - `runs/T38_seed20260429_trace_probe_20260513`
- The output is trace evidence, not a new formal benchmark claim.

Important execution note:

- The first worker invocation hit the tool's 1-hour timeout while writing into the same resumable run dir.
- I continued only by resuming the same `t3860429_resume` directory until `missing_runs = 0`.
- This stayed inside one T38-scoped probe rather than creating a second independent rerun.

## 2. Exact Command And Run Directory

### 2.1 Bounded T38 rerun

Initial bounded rerun command:

```powershell
C:\ProgramData\anaconda3\python.exe -m cnn_fpga.benchmark.run_p4_teacher_representation_paired `
  --stage benchmark `
  --seed 20260429 `
  --variant full `
  --variant gated_v5 `
  --benchmark-scenario linear_ramp `
  --benchmark-scenario periodic_drift `
  --benchmark-scenario static_bias_theta `
  --benchmark-scenario step_sigma_theta `
  --repeats 2 `
  --output-session-dir runs/T38_seed20260429_trace_probe_20260513/paired_session `
  --session-root runs/T38_seed20260429_trace_probe_20260513 `
  --benchmark-output-root runs/T38_seed20260429_trace_probe_20260513 `
  --benchmark-base-config cnn_fpga/config/p4_teacher_repr_mid.yaml `
  --model-session-dir runs/teachrepr_mid/paired_v5_reuse `
  --experiment-prefix t38 `
  --chunk-repeat-size 1
```

Same-run-dir resume-only aggregation command:

```powershell
C:\ProgramData\anaconda3\python.exe -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark `
  --config runs/T38_seed20260429_trace_probe_20260513/paired_session/cfg/s20260429/benchmark.yaml `
  --run-dir runs/T38_seed20260429_trace_probe_20260513/p4_benchmark/t3860429_resume `
  --repeats 2 `
  --paired-seeds `
  --scenario linear_ramp `
  --scenario periodic_drift `
  --scenario static_bias_theta `
  --scenario step_sigma_theta `
  --resume-only
```

### 2.2 Trace export command

```powershell
C:\ProgramData\anaconda3\python.exe cnn_fpga/benchmark/analyze_seed20260429_trace.py `
  --run-dir runs/T38_seed20260429_trace_probe_20260513/p4_benchmark/t3860429_resume `
  --output-dir runs/T38_seed20260429_trace_probe_20260513/trace_export
```

### 2.3 Primary T38 artifact paths

- Benchmark run dir:
  - `runs/T38_seed20260429_trace_probe_20260513/p4_benchmark/t3860429_resume`
- Benchmark summary:
  - `runs/T38_seed20260429_trace_probe_20260513/p4_benchmark/t3860429_resume/summary.json`
- Trace export root:
  - `runs/T38_seed20260429_trace_probe_20260513/trace_export`
- Trace rows:
  - `runs/T38_seed20260429_trace_probe_20260513/trace_export/trace_rows.csv`
- Repeat summary:
  - `runs/T38_seed20260429_trace_probe_20260513/trace_export/repeat_summary.csv`
- Scenario/mode summary:
  - `runs/T38_seed20260429_trace_probe_20260513/trace_export/scenario_mode_summary.csv`
- Paired repeat comparison:
  - `runs/T38_seed20260429_trace_probe_20260513/trace_export/paired_repeat_comparison.csv`
- Field availability:
  - `runs/T38_seed20260429_trace_probe_20260513/trace_export/field_availability.json`

## 3. Trace Schema And Field Availability

### 3.1 Exported per-window fields

- scenario, mode, repeat, seed, window id
- `teacher_b_q`, `teacher_b_p`
- raw predicted `delta_b_q`, `delta_b_p`
- applied `delta_b_q`, `delta_b_p`
- committed `b_q`, `b_p`
- commit bank / commit epoch / commit version
- `window_ler`
- `mean_correction_utilization`
- overflow / saturation diagnostics
- teacher explain status and teacher contribution L2 when present

### 3.2 Availability table

All required T38 trace fields are present in the exported event stream.

| Field | Status | Source |
| --- | --- | --- |
| `scenario` | `present` | `summary.raw_rows[].scenario` |
| `mode` | `present` | `summary.raw_rows[].mode` |
| `repeat` | `present` | `summary.raw_rows[].repeat` |
| `seed` | `present` | `summary.raw_rows[].seed` |
| `window_id` | `present` | `host_events[].readout.window.window_id` |
| `teacher_b_q / teacher_b_p` | `present` | `host_events[].proposed_params.metadata.teacher_params.b[]` |
| `delta_b_q / delta_b_p` | `present` | `host_events[].proposed_params.metadata.applied_delta_b[]` |
| `committed_b_q / committed_b_p` | `present` | `host_events[].proposed_params.b[]` |
| `commit_target_bank` | `present` | `host_events[].commit.target_bank` |
| `commit_epoch` | `present` | `host_events[].commit.commit_epoch` |
| `commit_version` | `present` | `host_events[].commit.version` |
| `window_ler` | `present` | `host_events[].proposed_params.metadata.window_diagnostics.window_ler` |
| `mean_correction_utilization` | `present` | `host_events[].proposed_params.metadata.window_diagnostics.mean_correction_utilization` |
| `overflow_ratio` | `present` | `host_events[].proposed_params.metadata.window_diagnostics.overflow_ratio` |
| `correction_saturation_ratio` | `present` | `host_events[].proposed_params.metadata.window_diagnostics.correction_saturation_ratio` |
| `dominant_overflow_source` | `present` | `host_events[].proposed_params.metadata.window_diagnostics.dominant_overflow_source` |

Boundary:

- T38 did not need new runtime instrumentation.
- The needed fields were already present in `hil_events.json`; T38 adds export and bounded rerun evidence around them.

## 4. Rerun Summary

### 4.1 Completed bounded rerun status

From `runs/T38_seed20260429_trace_probe_20260513/p4_benchmark/t3860429_resume/summary.json`:

- `missing_runs = 0`
- `raw_rows = 16`
- `comparison_rows = 8`
- one T38 run dir only

### 4.2 Scenario-level final LER means

| Scenario | Full | Gated v5 | Gap (`gated - full`) | Winner |
| --- | ---: | ---: | ---: | --- |
| `static_bias_theta` | `0.634351250` | `0.632402083` | `-0.001949167` | `Gated v5` |
| `linear_ramp` | `0.635961250` | `0.632686250` | `-0.003275000` | `Gated v5` |
| `step_sigma_theta` | `0.636448750` | `0.658752083` | `+0.022303333` | `Full` |
| `periodic_drift` | `0.641247083` | `0.642612083` | `+0.001365000` | `Full` |

Interpretation:

- The bounded T38 rerun stays close to the T36 picture in shape, not in exact decimals.
- `step_sigma_theta` remains the strongest negative case for `Gated v5`.
- `static_bias_theta` and `linear_ramp` are favorable or near-favorable to `Gated v5`.
- `periodic_drift` is close, but `Full` is slightly better.

## 5. Trace-Level Findings

### 5.1 Teacher branch is not dead, and the issue is not teacher inactivity

Across all four scenarios, `Gated v5` shows:

- non-null `teacher_contribution_l2_mean`
- much larger `teacher_b` magnitudes than `Full`
- frequent `teacher_b` / `delta_b` / committed-`b` sign flips

Examples from `scenario_mode_summary.csv`:

- `linear_ramp`, `Gated v5`:
  - `teacher_contribution_l2_mean_mean = 0.149861081`
  - `max_abs_teacher_b_mean = 0.358406947`
  - `delta_b_q_sign_flips_total = 107`
  - `committed_b_q_sign_flips_total = 103`
- `static_bias_theta`, `Gated v5`:
  - `teacher_contribution_l2_mean_mean = 0.150993424`
  - `max_abs_teacher_b_mean = 0.347697673`
- `step_sigma_theta`, `Gated v5`:
  - `teacher_contribution_l2_mean_mean = 0.132083790`
  - `max_abs_teacher_b_mean = 0.350995762`

This supports:

- the gated teacher path is active
- the issue is not "teacher branch missing" or "gated branch too weak"

### 5.2 The main separation is amplitude regime, not scheduler lag

From `paired_repeat_comparison.csv`, every scenario/repeat pair shows a much larger `Gated v5` residual amplitude than `Full`.

Typical pattern:

- `Full max_abs_delta_b` is about `0.025` to `0.028`
- `Gated v5 max_abs_delta_b` is always `0.169705627`
- the gap is consistently about `+0.141` to `+0.145`

Scenario/repeat examples:

- `linear_ramp`, repeat 0:
  - Full `max_abs_delta_b = 0.025480987`
  - Gated `max_abs_delta_b = 0.169705627`
- `static_bias_theta`, repeat 1:
  - Full `max_abs_delta_b = 0.024931869`
  - Gated `max_abs_delta_b = 0.169705627`
- `step_sigma_theta`, repeat 1:
  - Full `max_abs_delta_b = 0.028354392`
  - Gated `max_abs_delta_b = 0.169705627`

This is the strongest trace-level separation in the run.

### 5.3 Gated raw deltas are frequently clipped, but clipping alone does not decide win/loss

From `trace_rows.csv`:

- Full:
  - `raw_delta_b` and applied `delta_b` are identical in all checked windows
  - no component hits the `0.12` clip boundary
- Gated v5:
  - `raw_delta_b` exceeds applied `delta_b` in many windows
  - windows with at least one component at clip:
    - `linear_ramp`: `325 / 600`
    - `periodic_drift`: `295 / 599`
    - `static_bias_theta`: `325 / 600`
    - `step_sigma_theta`: `288 / 600`

Mean vector norm comparison:

| Scenario | Mode | mean abs raw delta | mean abs applied delta |
| --- | --- | ---: | ---: |
| `linear_ramp` | `Full` | `0.019780` | `0.019780` |
| `linear_ramp` | `Gated v5` | `0.150377` | `0.118190` |
| `periodic_drift` | `Full` | `0.019191` | `0.019191` |
| `periodic_drift` | `Gated v5` | `0.143212` | `0.113359` |
| `static_bias_theta` | `Full` | `0.019422` | `0.019422` |
| `static_bias_theta` | `Gated v5` | `0.151527` | `0.119405` |
| `step_sigma_theta` | `Full` | `0.020045` | `0.020045` |
| `step_sigma_theta` | `Gated v5` | `0.132609` | `0.104294` |

Interpretation:

- `Gated v5` is operating in a much higher-amplitude residual regime.
- clipping is common and real
- but clipping is not by itself the full failure explanation, because:
  - `static_bias_theta` still wins for `Gated v5`
  - `periodic_drift` has similar clipping behavior but only a small mean loss

### 5.4 Trace evidence does not support a simple sign-offset story

Why not:

- `Gated v5` sometimes wins and sometimes loses under the same sign-flip-heavy regime
- sign flips are abundant in every `Gated v5` scenario, not just the failing one
- `Full` remains near-zero on sign flips for almost all channels

So the trace narrows the problem to:

- sign instability is present
- but the more decisive signal is high-amplitude, oscillatory `delta_b` / committed-`b` dynamics
- not a single clean constant sign bias

### 5.5 Per-window outcome chronology points to committed combined-`b` instability

The strongest evidence is that the loser cases are the ones where:

- `teacher_b` is already much larger than `Full`
- `delta_b` remains much larger than `Full`
- committed `b` therefore becomes much larger than `Full`
- final or peak `window_ler` rises with that combined amplitude

Examples:

- `step_sigma_theta`, repeat 1:
  - Full final window LER = `0.509765625`
  - Gated final window LER = `0.589355469`
  - Full max abs committed `b` = `0.200945555`
  - Gated max abs committed `b` = `0.423980861`
- `linear_ramp`, repeat 1:
  - Full final window LER = `0.520019531`
  - Gated final window LER = `0.589843750`
  - Full max abs committed `b` = `0.192759904`
  - Gated max abs committed `b` = `0.443791672`

This is more consistent with:

- combined committed `teacher_b + delta_b` instability

than with:

- teacher-only failure
- CNN-only silent failure
- scheduler/commit lag

## 6. Mechanism Update Matrix

| Candidate mechanism | T38 label | T38 update |
| --- | --- | --- |
| `sign offset` | `partially observed but not leading explanation` | Sign changes are real in `Gated v5`, but they occur across both winning and losing scenarios. T38 does not support a simple fixed sign-bias story. |
| `magnitude overshoot chronology` | `supported` | The strongest repeat-level separation is persistent high-amplitude `delta_b` and committed `b` in `Gated v5`, with frequent clipping and large amplitude gaps versus `Full`. |
| `teacher prediction instability` | `partially supported` | `teacher_b` amplitude is much larger in `Gated v5`, but T38 does not isolate teacher prediction as the sole root cause. |
| `CNN residual output instability` | `supported` | `raw_delta_b` is much larger in `Gated v5`, often needs clipping, and flips sign frequently; this strongly supports residual-output instability. |
| `committed combined-b instability` | `strongly supported` | The clearest trace-level explanation is that large teacher-b plus large residual delta produce a much larger committed `b`, and loser repeats track that combined amplitude regime. |

## 7. Supported Conclusions Vs Remaining Hypotheses

### 7.1 Supported conclusions

1. T38 did not require new runtime semantics; the needed trace fields were already present in `hil_events.json`.
2. The bounded T38 rerun stayed inside one new T38-scoped run root and completed with `missing_runs = 0`.
3. `Gated v5` operates in a far larger `delta_b` regime than `Full` on `seed=20260429`.
4. That regime frequently triggers delta clipping and frequent sign reversals.
5. The most convincing explanation from trace evidence is combined committed-`b` instability, not response lag, not a dead teacher branch, and not a simple constant sign offset.

### 7.2 Remaining hypotheses / limits

1. T38 still does not prove whether the teacher amplitude shift or the CNN residual amplitude shift is the first upstream cause.
2. T38 does not test a mitigation such as lower residual clip, lower residual scale, or changed teacher encoding.
3. Because this is still one seed only, the mechanism claim should remain seed-bounded and diagnostic rather than paper-grade causal evidence.

## 8. Recommended Next Bounded Task

Recommended next bounded task, not executed here:

- keep the same seed-bounded diagnostic scope
- do not widen benchmark protocol or scenario family
- test one minimal mitigation against the same T38 path, for example:
  - lower residual clip / residual scale for `Gated v5`, or
  - a bounded teacher-delta attenuation variant

Why this is the right next step:

- T36 narrowed the issue to residual-amplitude / teacher-delta instability.
- T38 upgrades that to trace-supported combined committed-`b` instability.
- The next bounded question is now mitigation, not further observability.
