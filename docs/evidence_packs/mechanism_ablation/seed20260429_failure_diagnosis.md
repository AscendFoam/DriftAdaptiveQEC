# `seed=20260429` Failure Diagnosis

## 1. Scope And Evidence Boundary

This report is a bounded, read-only diagnosis for `seed=20260429`.

- It uses existing artifacts only.
- It does not rerun benchmark, training, `.tflite`, hardware, or cleanup.
- It does not rewrite the formal benchmark boundary, frozen baseline set, or statcalib scope.
- The older non-chunked `trp60429` run is used as context only, not mixed into the main paired/chunked conclusion.

Important boundary:

- The current artifacts do **not** expose a full per-window or per-commit time series for committed `b_q / b_p`, CNN `delta_b`, or `teacher_b + delta_b`.
- Repeat-level `hil_summary.json` preserves aggregate metrics and one final snapshot only.
- Therefore the diagnosis below is a **summary-level / final-snapshot-level diagnosis**, not a causal proof.

## 2. Evidence Inventory

### 2.1 Primary paired/chunked summary

- `runs/teachrepr_v5_chunked_pair/paired_20260427_220702/summary.csv`
- `runs/teachrepr_v5_chunked_pair/paired_20260427_220702/summary.json`
- `runs/teachrepr_v5_chunked/p4_benchmark/trp60429_resume/summary.json`

### 2.2 Cross-seed scenario comparison

- `runs/teachrepr_v5_chunked/p4_benchmark/trp60427_resume/comparison.csv`
- `runs/teachrepr_v5_chunked/p4_benchmark/trp60428_resume/comparison.csv`
- `runs/teachrepr_v5_chunked/p4_benchmark/trp60429_resume/comparison.csv`

### 2.3 Cross-seed teacher diagnostics

- `runs/teachrepr_v5_chunked/p4_benchmark/trp60427_resume/teacher_scalar_diagnostics.csv`
- `runs/teachrepr_v5_chunked/p4_benchmark/trp60428_resume/teacher_scalar_diagnostics.csv`
- `runs/teachrepr_v5_chunked/p4_benchmark/trp60429_resume/teacher_scalar_diagnostics.csv`

### 2.4 `seed=20260429` repeat-level final-snapshot evidence

- `runs/teachrepr_v5_chunked/p4_benchmark/trp60429_resume/static/hybrid/repeat_00/hil_summary.json`
- `runs/teachrepr_v5_chunked/p4_benchmark/trp60429_resume/static/hybrid/repeat_01/hil_summary.json`
- `runs/teachrepr_v5_chunked/p4_benchmark/trp60429_resume/static/hybri1/repeat_00/hil_summary.json`
- `runs/teachrepr_v5_chunked/p4_benchmark/trp60429_resume/static/hybri1/repeat_01/hil_summary.json`
- `runs/teachrepr_v5_chunked/p4_benchmark/trp60429_resume/linear/hybrid/repeat_00/hil_summary.json`
- `runs/teachrepr_v5_chunked/p4_benchmark/trp60429_resume/linear/hybrid/repeat_01/hil_summary.json`
- `runs/teachrepr_v5_chunked/p4_benchmark/trp60429_resume/linear/hybri1/repeat_00/hil_summary.json`
- `runs/teachrepr_v5_chunked/p4_benchmark/trp60429_resume/linear/hybri1/repeat_01/hil_summary.json`
- `runs/teachrepr_v5_chunked/p4_benchmark/trp60429_resume/stepsi/hybrid/repeat_00/hil_summary.json`
- `runs/teachrepr_v5_chunked/p4_benchmark/trp60429_resume/stepsi/hybrid/repeat_01/hil_summary.json`
- `runs/teachrepr_v5_chunked/p4_benchmark/trp60429_resume/stepsi/hybri1/repeat_00/hil_summary.json`
- `runs/teachrepr_v5_chunked/p4_benchmark/trp60429_resume/stepsi/hybri1/repeat_01/hil_summary.json`
- `runs/teachrepr_v5_chunked/p4_benchmark/trp60429_resume/period/hybrid/repeat_00/hil_summary.json`
- `runs/teachrepr_v5_chunked/p4_benchmark/trp60429_resume/period/hybrid/repeat_01/hil_summary.json`
- `runs/teachrepr_v5_chunked/p4_benchmark/trp60429_resume/period/hybri1/repeat_00/hil_summary.json`
- `runs/teachrepr_v5_chunked/p4_benchmark/trp60429_resume/period/hybri1/repeat_01/hil_summary.json`

### 2.5 Historical context only

- `runs/teachrepr/p4_benchmark/trp60429_20260427_142013_2a59bc_24060/comparison.csv`

I treat this older run as context only because its run shape differs from the paired/chunked rerun. In particular, its `n_commits_applied_mean` is `60`, while the paired/chunked rerun uses `300` commits per scenario.

## 3. Primary Result Summary

### 3.1 Cross-seed average result from the paired/chunked rerun

| Seed | Full LER | Gated v5 LER | Gap (`gated - full`) |
| --- | ---: | ---: | ---: |
| `20260427` | `0.806566458` | `0.620511354` | `-0.186055104` |
| `20260428` | `0.832559167` | `0.594353542` | `-0.238205625` |
| `20260429` | `0.637362500` | `0.639720208` | `+0.002357708` |

Primary observation:

- `20260429` is the only reviewed seed where the average gap flips from "Gated clearly better" to "Gated slightly worse".

### 3.2 `seed=20260429` summary by scenario

| Scenario | Full | Gated v5 | Gap (`gated - full`) | Winner |
| --- | ---: | ---: | ---: | --- |
| `static_bias_theta` | `0.633163750` | `0.645085000` | `+0.011921250` | `Full` |
| `linear_ramp` | `0.637528750` | `0.631814583` | `-0.005714167` | `Gated v5` |
| `step_sigma_theta` | `0.636237083` | `0.639388333` | `+0.003151250` | `Full` |
| `periodic_drift` | `0.642520417` | `0.642592917` | `+0.000072500` | `Near tie / Full by 7.25e-05` |

Interpretation:

- The reversal is not uniform across all scenarios.
- The main losses are `static_bias_theta` and `step_sigma_theta`.
- `linear_ramp` still slightly favors `Gated v5`.
- `periodic_drift` is effectively a tie at this artifact resolution.

### 3.3 Cross-seed scenario comparison

| Scenario | Gap @ `20260427` | Gap @ `20260428` | Gap @ `20260429` |
| --- | ---: | ---: | ---: |
| `static_bias_theta` | `-0.139174167` | `-0.232110833` | `+0.011921250` |
| `linear_ramp` | `-0.305685833` | `-0.236957500` | `-0.005714167` |
| `step_sigma_theta` | `-0.162734167` | `-0.255184583` | `+0.003151250` |
| `periodic_drift` | `-0.136626250` | `-0.228569583` | `+0.000072500` |

Interpretation:

- The collapse on `20260429` is broad, not limited to one scenario.
- But it is also not a clean all-scenario failure: one scenario still wins and one is effectively tied.

### 3.4 Historical context only: the older pre-chunked `trp60429`

From `runs/teachrepr/p4_benchmark/trp60429_20260427_142013_2a59bc_24060/comparison.csv`:

- `Full ≈ 0.688990`
- `Gated v5 ≈ 0.674559`
- gap `≈ -0.014432`

I do **not** combine these numbers with the paired/chunked rerun, because the run shape differs. But this older run still matters qualitatively:

- `20260429` was already a **marginal / sensitive** seed, not a seed with a large stable Gated-v5 advantage.
- The later paired/chunked rerun tightening that margin into a slight reversal is consistent with "seed-specific instability", not with "the Gated branch is always bad on 20260429".

## 4. Additional Diagnostic Signals

### 4.1 Response lag / scheduler path

Observed facts:

- In `comparison.csv`, `n_commits_applied_mean` is `299.5` to `300.0` for both modes in all four scenarios.
- `slow_update_violation_rate_mean = 0.0` for both modes in all four scenarios.
- `fast_cycle_violation_rate_mean` is identical within each scenario.
- Repeat-level `hil_summary.json` shows matching scheduler statistics within each scenario between `Full` and `Gated v5`.

Read:

- The slow-update scheduler and commit path are shared and numerically matched between the two modes inside `20260429`.
- Current artifacts do **not** support "response lag" as the differentiating mechanism for the seed flip.

### 4.2 Overflow / saturation path

Observed facts:

- `correction_saturation_rate_mean = 0.0` for all eight `20260429` comparison rows.
- `overflow_rate_mean` is slightly **lower** for `Gated v5` than for `Full` in all four scenarios:
  - static: `0.002257083 < 0.002267917`
  - linear: `0.002213750 < 0.002318750`
  - step: `0.002342500 < 0.002379167`
  - periodic: `0.002194167 < 0.002292500`

Read:

- The `20260429` reversal is not explained by more overflow or correction saturation in `Gated v5`.

### 4.3 Teacher branch is active on `20260429`

Observed facts from `20260429` `comparison.csv`:

- `teacher_contribution_l2_mean_mean` remains nonzero in all four Gated-v5 rows: `0.140855` to `0.150536`
- `teacher_scalar_abs_mean_mean` remains nonzero: `0.091115` to `0.097452`
- `teacher_gate_mean_mean` stays near `0.507` to `0.512`
- `teacher_gate_std_mean` stays near `0.486` to `0.490`

Read:

- The teacher branch is active.
- The seed flip is **not** explained by a dead teacher branch or a missing diagnostics path.

### 4.4 Teacher-scalar regime shift across seeds

Average `ablation_l2_mean` by scalar, averaged over four scenarios:

| Scalar | `20260427` | `20260428` | `20260429` |
| --- | ---: | ---: | ---: |
| `teacher_b_q` | `0.118719381` | `0.002776038` | `0.080193639` |
| `teacher_b_p` | `0.177749249` | `0.156019549` | `0.078633967` |
| `teacher_delta_b_q` | `0.001308029` | `0.001057334` | `0.066147847` |
| `teacher_delta_b_p` | `0.002375738` | `0.001696365` | `0.041109218` |

Average `gate_delta_l2_mean` by scalar, averaged over four scenarios:

| Scalar | `20260427` | `20260428` | `20260429` |
| --- | ---: | ---: | ---: |
| `teacher_b_q` | `5.379024642` | `0.241867920` | `3.647329419` |
| `teacher_b_p` | `3.063790922` | `5.809171469` | `3.278943223` |
| `teacher_delta_b_q` | `0.470669655` | `0.245284155` | `3.667584672` |
| `teacher_delta_b_p` | `0.520626261` | `0.334783509` | `3.084537134` |

Read:

- `20260429` is not simply "a slightly worse copy of 20260427/20260428".
- The biggest regime shift is on the `teacher_delta_b_*` channels:
  - versus `20260428`, `teacher_delta_b_q` ablation rises from `0.001057` to `0.066148`
  - versus `20260428`, `teacher_delta_b_p` ablation rises from `0.001696` to `0.041109`
- This strongly suggests that `20260429` pushes the Gated-v5 path into a different teacher-feature regime, especially on delta-b channels.

### 4.5 Repeat instability and final-snapshot evidence

`20260429` repeat-level final bias-norm snapshot (`|b|` from the final committed active bank in `hil_summary.json`):

| Scenario | Full `repeat_00 / repeat_01` | Gated v5 `repeat_00 / repeat_01` |
| --- | --- | --- |
| `static_bias_theta` | `0.177998 / 0.182373` | `0.140143 / 0.419183` |
| `linear_ramp` | `0.171417 / 0.177412` | `0.266493 / 0.331711` |
| `step_sigma_theta` | `0.184769 / 0.184655` | `0.294980 / 0.368709` |
| `periodic_drift` | `0.165429 / 0.163295` | `0.305748 / 0.322092` |

Scenario-level `final_ler_std`:

| Scenario | Full Std | Gated v5 Std |
| --- | ---: | ---: |
| `static_bias_theta` | `0.000562917` | `0.017364167` |
| `linear_ramp` | `0.001098750` | `0.000124583` |
| `step_sigma_theta` | `0.000735417` | `0.001532500` |
| `periodic_drift` | `0.000589583` | `0.009128750` |

Read:

- In `static_bias_theta`, `step_sigma_theta`, and `periodic_drift`, the losing or near-losing Gated-v5 repeats tend to end with much larger final `|b|` than `Full`.
- `linear_ramp` is the exception: Gated v5 still wins there, even with larger final `|b|`.
- This pattern is consistent with **scenario/repeat-sensitive residual amplitude instability**, not with a uniformly dead or uniformly weak branch.

Boundary reminder:

- These are final snapshots only.
- They support a hypothesis about residual magnitude behavior, but they do **not** prove the whole per-window trajectory.

## 5. Mechanism Matrix

| Candidate mechanism | Evidence label | Current evidence | Why not stronger |
| --- | --- | --- | --- |
| `sign offset` | `not answerable` | Final snapshots sometimes differ in sign between repeats/modes, but current artifacts do not contain full committed-parameter time series. | No per-window/per-commit `b_q / b_p` trace or target `b` trace. |
| `magnitude overshoot` | `plausible / partially supported` | Losing `20260429` Gated-v5 repeats often end with much larger final `|b|` than `Full`, and scenario-level `final_ler_std` is much larger in static/periodic. | Only final snapshots are available; overshoot chronology cannot be proven. |
| `response lag` | `not supported` | Commit counts, scheduler-violation rates, and repeat-level slow-update stats match between modes. | Current artifacts already rule this out at the summary level. |
| `teacher instability` | `partially supported` | `teacher_delta_b_*` channels move into a much stronger and more variable regime on `20260429`. | `20260427` also has strong teacher activity and still wins strongly; teacher instability alone is not sufficient. |
| `gated branch too conservative` | `not supported` | Gated-v5 final `|b|` is usually larger, not smaller, than `Full`; teacher branch diagnostics remain active. | Current evidence points away from under-activation. |

## 6. Conclusion

### 6.1 Supported observations

1. `20260429` is the only reviewed seed where the paired/chunked average gap flips from negative to slightly positive (`+0.002357708`).
2. The reversal is concentrated in `static_bias_theta` and `step_sigma_theta`, while `linear_ramp` still slightly favors `Gated v5`.
3. The teacher branch is active on `20260429`; this is not a missing-diagnostics or dead-branch problem.
4. `response lag`, commit-path failure, overflow growth, and correction saturation are **not** supported as the differentiating mechanism.
5. `20260429` pushes the Gated-v5 path into a different teacher-feature regime, especially on `teacher_delta_b_q` and `teacher_delta_b_p`.

### 6.2 Plausible hypotheses

1. The leading hypothesis is **scenario/repeat-sensitive magnitude instability** in the residual-b correction path.
2. A likely contributing factor is that the `teacher_delta_b_*` channels become much more active/variable on `20260429`, which makes the gated residual path less stable than on `20260427/20260428`.
3. This is **not** the same as saying "teacher features are bad". The evidence is narrower: the current Gated-v5 encoding looks fragile on this seed-specific delta-b regime.

### 6.3 Not answerable from current artifacts

1. Whether the first error is a true sign mistake, a late correction, or a late-window overshoot.
2. Whether the instability originates first in the teacher prediction itself, in the residual CNN output, or in the combined committed `teacher_b + delta_b`.
3. Whether a different loss, stronger clipping, or a different teacher encoding would fix `20260429` without harming the good seeds.

## 7. Recommended Next Bounded Task

Recommended next bounded task, not executed here:

- add a trace-export path for one bounded rerun of `seed=20260429` only, with unchanged benchmark semantics, exporting per-window:
  - `teacher_b_q / teacher_b_p`
  - predicted `delta_b_q / delta_b_p`
  - committed `b_q / b_p`
  - window-level LER / correction-utilization
- keep the scope at:
  - same two modes (`Full`, `Gated v5`)
  - same four scenarios
  - no new baseline, no branch expansion, no formal benchmark rewrite

Why this next task is the right follow-up:

- The current artifacts are sufficient to narrow the failure to a residual-amplitude / teacher-delta regime problem.
- They are **not** sufficient to decide between sign offset, overshoot chronology, and exact source attribution.
- A single-seed trace-export follow-up would answer that missing question directly, without widening the benchmark or changing project boundaries.
