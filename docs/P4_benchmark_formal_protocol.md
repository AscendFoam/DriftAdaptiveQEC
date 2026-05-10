# P4 Benchmark Formal Protocol

## 1. Status

This document locks the next-step P4 formal benchmark protocol at the documentation level only.

- `T23 did not run benchmark`.
- `T23` only audited protocol, evidence gaps, compute budget, and later execution gates.
- The target here is a formal **software benchmark revalidation** of the historical frozen P4 comparison set.
- It is not a `.tflite` runtime validation.
- It is not a `real_board` validation.

## 2. Evidence Levels

| Level | Config family | Scope | Allowed claim |
| --- | --- | --- | --- |
| Recovery smoke | `cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml` | single scenario, bounded mode subset, `repeats=1` | wrapper and recovery path are runnable on the current machine |
| Development bounded run | `cnn_fpga/config/p4_multiscenario_strong_baselines.yaml` with explicit filters | two scenarios, five frozen modes, `repeats=2` | bounded multi-scenario development evidence only |
| Formal software benchmark revalidation | `cnn_fpga/config/p4_multiscenario_strong_baselines.yaml` without semantic edits | four historical formal scenarios, five frozen modes, paired seeds, fixed report pack | historical frozen software benchmark has been re-run on the recovered path |

Boundary reminder:

- `formal` here still means `mock-backed P4 wrapper over software HIL`.
- `formal` does not mean true `.tflite` runtime recovered.
- `formal` does not mean `real_board` HIL validated.

## 3. Frozen Software Boundary

The following protocol items are locked for later formal execution:

1. Runner entry stays `python -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark`.
2. Config family stays `cnn_fpga/config/p4_multiscenario_strong_baselines.yaml`.
3. The strong-baseline config continues to inherit `p4_multiscenario_hybrid_b_long.yaml`.
4. The benchmark remains a software HIL path with `real_board_policy = conditional_extension`.
5. No code edits, no benchmark semantic edits, no baseline renaming, and no ParamMapper changes are allowed inside the later execution task.
6. Chunking and resume are allowed only through existing runner controls:
   - `--run-dir`
   - `--repeat-start`
   - `--repeat-stop`
   - `--resume-only`

## 4. Locked Formal Matrix

### 4.1 Scenarios

The historical frozen formal scenario set remains:

- `static_bias_theta`
- `linear_ramp`
- `step_sigma_theta`
- `periodic_drift`

### 4.2 Modes

The frozen comparison set remains exactly:

- `ekf`
- `ukf`
- `constant_residual_mu`
- `rls_residual_b`
- `hybrid_residual_b`

### 4.3 Repeat and seed policy

Locked rules for the revalidation pass:

- `repeats = 2`
- `paired_seeds = true`
- seed base follows the benchmark config family:
  - `experiment.seed = 20260403`
  - `scenario_seed_stride = 1000`
- all modes within the same scenario/repeat must share the same evaluation seed stream

### 4.4 Formal execution shape

The bounded formal revalidation matrix is therefore:

- `4 scenarios x 5 modes x 2 repeats = 40 repeat-runs`

This is the exact next-step execution scope. Any larger scope is out of protocol unless a later task package says otherwise.

## 5. Baseline Inclusion And Exclusion Rules

### 5.1 Included in the ranked formal table

Only these modes enter the ranked formal comparison:

- `ekf`
- `ukf`
- `constant_residual_mu`
- `rls_residual_b`
- `hybrid_residual_b`

Reason:

- they are the frozen set already declared by `p4_multiscenario_strong_baselines.yaml`
- changing this set inside the recovery/revalidation lane would silently redefine the benchmark

### 5.2 Explicitly excluded from the ranked formal table

The following remain out of the formal ranked set for `T24`:

- `static_linear`
- `window_variance`
- `cnn_fpga`
- teacher-representation branches such as `Gated v5`
- `paper_inspired_statcalib_v1`
- any `.tflite` service or stub path as a separate benchmark mode
- any `board` / `real_board` backend path

Reason:

- `static_linear / window_variance / cnn_fpga` belong to smoke/development anchoring, not the frozen strong-baseline ranking set
- teacher-representation and paper-inspired variants are different research lanes, not part of the frozen mainline set
- `.tflite` and `real_board` are deployment-boundary questions, not part of the current formal software benchmark claim

### 5.3 Calibration / statcalib rule

`statcalib` is not added into the frozen `T24` ranking set by default.

However:

- `statcalib` remains a recommended later comparator for paper-grade evidence
- if approved later, it must be added as a separately labeled extension lane, not by silently rewriting the frozen set

## 6. Statistical And Reporting Rules

### 6.1 Fairness rules

Later formal execution must keep:

1. paired seeds across modes within each scenario/repeat
2. unchanged scenario definitions
3. unchanged baseline semantics
4. unchanged artifact selection semantics for learned modes

### 6.2 Training seed vs evaluation seed separation

This recommendation is adopted as a reporting rule now.

For each learned mode in the formal pack:

1. report artifact path
2. report training config or provenance pointer
3. report training seed information if known
4. report evaluation seed policy separately from training seed provenance

Current note:

- in the frozen set, the main learned mode is `hybrid_residual_b`
- its evaluation seeds must stay on the benchmark-side paired seed chain

### 6.3 Required statistical outputs for the next formal run

The later execution task must report at least:

1. per-scenario winners and runner-up gaps
2. `final_ler_mean` and `final_ler_std`
3. `overflow_rate_mean`
4. `histogram_input_saturation_rate_mean`
5. `correction_saturation_rate_mean`
6. `aggressive_param_rate_mean`
7. `n_commits_applied_mean`
8. `slow_update_violation_rate_mean`
9. `fast_cycle_violation_rate_mean`
10. raw per-repeat rows
11. `missing_runs`
12. coverage for each scenario/mode pair

### 6.4 Confidence-interval / stopping-rule decision

Deep-research advice recommends `95%` confidence intervals or larger trace counts. For this repository's next step:

- `mean/std + raw_rows + paired seeds` are adopted for the immediate revalidation pass
- CI-driven stopping is deferred from `T24`
- any upgrade from fixed `repeats=2` to a CI-driven stopping rule requires a separate later task, because it changes compute scope materially

This keeps `T24` comparable to the recovered historical frozen protocol, while leaving a later paper-grade expansion path open.

## 7. Deep-Research Recommendation Audit

| Recommendation | Decision in T23 | Handling |
| --- | --- | --- |
| Strong classical baseline classes | `adopted` | frozen set keeps `ekf / ukf / constant_residual_mu / rls_residual_b` as the classical comparison core |
| Soft-information / correlation-aware baselines | `deferred` | not in current code/config family; adding them now would redefine the frozen set |
| Calibration / statcalib baseline | `deferred_as_followup` | should become a later comparator lane, but not silently inserted into `T24` frozen-set revalidation |
| Learned baseline classes beyond `hybrid_residual_b` | `deferred` | teacher-representation and FiLM-like variants need their own task packages and should not enter `T24` |
| Historical four-scenario set | `adopted` | `static_bias_theta / linear_ramp / step_sigma_theta / periodic_drift` are the locked formal revalidation set |
| Extra scenario families: `random-walk / sinusoidal / burst-reset` | `deferred` | recommended for later robustness extension, but not part of the recovered frozen set |
| Training/evaluation seed separation | `adopted` | must be reported explicitly for learned modes |
| `95%` CI or stopping rule | `deferred` | useful later, but not required for the immediate fixed-scope revalidation |
| Latency / commit metrics | `partially_adopted` | commit counts and violation rates are already in runner outputs and must be reported |
| Rollback / fallback metrics | `deferred` | current runner pack does not expose them as first-class fields |
| True `.tflite` runtime before real-board smoke for deployment claims | `adopted` | deployment claims should prioritize true runtime restoration before board-level smoke |
| Rewriting T23 into a benchmark + mechanism + deployment mega-task | `rejected` | T23 stays protocol-lock only |

## 8. Compute Budget And Execution Risk

### 8.1 Budget in repeat-run units

Reference points:

- `T15` executed `2 scenarios x 5 modes x 2 repeats = 20 repeat-runs`
- locked `T24` revalidation would require `40 repeat-runs`
- this is exactly `2x` the `T15` repeat-run count

Increment rules:

- add one extra mode across the frozen four-scenario set: `+8 repeat-runs`
- add one extra repeat across the frozen set: `+20 repeat-runs`
- add one extra scenario across the frozen five-mode, two-repeat set: `+10 repeat-runs`

### 8.2 Wall-clock risk signal already observed

`T15` already showed that a full command can exceed the interactive shell timeout and may need resume on the same `run_dir`.

Therefore the later formal execution task should assume:

1. one-shot execution is operationally risky
2. chunked execution is acceptable if it preserves the same fixed `run_dir`
3. resume/re-aggregate on the same `run_dir` is acceptable and does not change semantics

### 8.3 Recommended chunking

Preferred execution pattern for the later run:

1. one fixed `run_dir`
2. scenario-wise chunking, or equivalent bounded repeat-split chunking
3. final aggregation through the existing runner outputs

## 9. Required Evidence Pack

The later formal execution task must preserve and report:

1. `launch_plan.json`
2. `progress.jsonl`
3. `summary.json`
4. `comparison.csv`
5. `delta.csv`
6. `teacher_scalar_diagnostics.csv`
7. `report.md`
8. each repeat directory's:
   - `hil_summary.json`
   - `repeat_status.json`
9. the exact config path
10. the exact CLI shape
11. `config_hash`
12. `git_commit`

Additional reporting requirements:

1. explicitly state whether execution was chunked
2. explicitly state whether `resume-only` was used
3. explicitly state that the result is still `mock-backed` software HIL evidence
4. explicitly state that the result is not `.tflite` runtime evidence
5. explicitly state that the result is not `real_board` evidence

## 10. Evidence Gaps After T23

`T23` locks protocol, but does not erase the following evidence gaps:

1. the two historical formal scenarios not yet re-run on the recovered path:
   - `step_sigma_theta`
   - `periodic_drift`
2. no calibration/statcalib comparator in the current frozen set
3. no soft-information / correlation-aware comparator in the current code path
4. `hybrid_residual_b` teacher diagnostics were all zero in `T15`, which is non-blocking for LER ranking but still a mechanism-analysis gap
5. no CI-driven stopping rule yet
6. no true `.tflite` runtime evidence yet
7. no `real_board` validation yet
8. rollback/fallback metrics are not yet first-class benchmark outputs

## 11. T24 Gate

### 11.1 Gate result

`T23` sets the following gate:

- `GO_FOR_BOUNDED_FORMAL_SOFTWARE_REVALIDATION`
- `NO_GO_FOR_SCOPE_EXPANSION_INSIDE_T24`

### 11.2 What T24 may do

`T24` may execute the formal revalidation only if it stays within all of the following:

1. use `cnn_fpga/config/p4_multiscenario_strong_baselines.yaml`
2. use the locked four-scenario set
3. use the locked five-mode frozen set
4. keep `repeats=2`
5. keep paired seeds
6. keep the existing software HIL boundary
7. keep the existing artifact semantics
8. preserve the evidence pack in Section 9

### 11.3 What T24 may not absorb

The following are outside the allowed execution scope for `T24` and must be handled by later prerequisite or extension tasks:

1. adding `statcalib` into the frozen ranking set
2. adding soft-information / closest-lattice / correlation-aware comparators
3. adding `random-walk / sinusoidal / burst-reset` scenario families
4. replacing fixed repeats with a CI-driven stop rule
5. mixing `.tflite` runtime recovery into the same task
6. mixing `real_board` smoke or validation into the same task
7. reopening teacher-representation long runs

## 12. Explicit Non-Claims

This document does not claim:

1. `T23` executed any benchmark
2. formal P4 results have already been restored
3. `.tflite` runtime has been restored
4. `real_board` HIL has been restored
5. the deep-research recommendation set has been fully implemented

## 13. Captain Closeout For T23

`docs/review/T23_review.md` verdict is `PASS_WITH_WARNINGS` with no blocking issues.

Captain handling:

1. N1 out-of-scope governance sync: `accepted`
2. N2 exact CLI shape: `deferred` to T24 task package and R19
3. N3 requested saturation metrics not individually verified: `deferred` to T24 metric availability check and R19
4. N4 requested fast-cycle violation metric not individually verified: `deferred` to T24 metric availability check and R19

T23 is complete as a protocol-lock task. It still did not run a benchmark.

## 14. T24 Execution Shape

The next task may execute bounded formal software revalidation with one fixed run directory under:

- `runs/p4_benchmark/T24_formal_software_revalidation_*`

The exact task package is:

- `docs/tasks/Phase2/T24_p4_formal_software_revalidation.md`

T24 must keep the full scenario and mode selection in every runner invocation. If chunking is needed, chunk only by repeat range:

1. `--repeat-start 0 --repeat-stop 1`
2. `--repeat-start 1 --repeat-stop 2`
3. final `--resume-only`

Do not split by a single scenario at a time, because scenario filtering changes the local `scenario_idx` used by the runner seed construction.
