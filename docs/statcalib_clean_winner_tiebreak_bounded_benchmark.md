# T69 StatCalib Clean-Winner Tie-Break Bounded Benchmark

## Verdict

`T69` completed one bounded clean-winner tie-break benchmark under a single T69 run root:

- run root: `runs/p4_benchmark/T69_statcalib_clean_winner_tiebreak_20260608_160358`
- evidence scope: mock-backed software-HIL only
- historical anchor: `T24` remains the authoritative frozen ranked table
- comparator boundary: `statcalib` remains a separately labeled extension lane only
- launch branch: `main`
- launch `HEAD`: `1dbfbc3`
- finish branch: `main`
- finish `HEAD`: `1dbfbc3`
- `summary.json git_commit`: `1dbfbc3`

The bounded T69 answer is narrower and cleaner than `T68`:

1. all four frozen T68 clean-winner candidates remain full `generated` winners under `repeats=4`
2. the old `window_variance_t001 = t003 = t005` clean tie set does **not** collapse
3. within the four-candidate T69 tie-break matrix, the mean-best set and the worst-case-best set are now the same
4. no unique clean reference point emerges

So the honest final T69 classification is:

- `persistent_clean_tie_set`

This report does not upgrade the evidence into `.tflite`, real-board, mature calibration-comparator, or paper-grade expanded benchmark claims.

## Preflight And Safe Acceleration Choice

The active workspace was intentionally left untouched after the T69 task-scoped files were created. The source workspace had unrelated governance edits and untracked T69 task-scoped files, so the benchmark was launched from a clean short-path clone instead of the live working tree:

- clean launch clone: `C:\t69c_1dbfbc3`
- launch timestamp: `2026-06-08 16:03:58 +08:00`
- `git -C C:\t69c_1dbfbc3 branch --show-current`: `main`
- `git -C C:\t69c_1dbfbc3 rev-parse --short HEAD`: `1dbfbc3`
- `git -C C:\t69c_1dbfbc3 status --short`: clean

Safe acceleration choice:

1. no mode-chunking
2. no scenario-chunking
3. no runner-semantic edits
4. one detached host launch for the full matrix so the benchmark could outlive foreground shell timeouts
5. no same-run-root repeat-range continuation was needed

## Config Lineage And Exact Benchmark Command

Repo-preserved task config:

- `cnn_fpga/config/p4_multiscenario_statcalib_clean_winner_tiebreak.yaml`

Launch-time config:

- `C:\t69cfg_20260608_160358.yaml`

Lineage note:

1. the launch-time config only replaced `base_config` with an absolute clone-local path
2. all task semantics stayed on the repo-preserved T69 matrix

Exact benchmark command executed inside the detached host launch:

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config C:\t69cfg_20260608_160358.yaml --scenario static_bias_theta --scenario linear_ramp --scenario step_sigma_theta --scenario periodic_drift --mode ukf --mode hybrid_residual_b --mode statcalib_window_variance_t001 --mode statcalib_window_variance_t003 --mode statcalib_window_variance_t005 --mode statcalib_ekf_t001 --paired-seeds --repeats 4 --run-dir D:\Codes\Quantum\DriftAdaptiveQEC\runs\p4_benchmark\T69_statcalib_clean_winner_tiebreak_20260608_160358
```

Execution shape:

1. one full-matrix invocation
2. one fixed T69 run root
3. no mode-chunking
4. no scenario-chunking
5. no repeat-range continuation
6. no relaunch of the identical full matrix

## Coverage And Provenance Checks

Artifact checks from the final run root:

- `summary.json["missing_runs"] = []`
- comparison rows: `24`
- raw rows: `96`
- all comparison rows have `coverage=1.0`
- all comparison rows have `completed_repeats=4`
- `progress.jsonl`: `running=96`, `completed=96`
- duplicate `running` keys: none
- duplicate `completed` keys: none
- exactly one T69-scoped run root exists under `runs/p4_benchmark/`

Provenance closure:

1. launch `HEAD` from clean clone = `1dbfbc3`
2. finish `HEAD` in the active workspace = `1dbfbc3`
3. `summary.json["git_commit"]` = `1dbfbc3`

Historical run protection:

1. `git diff --name-only -- runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743 runs/p4_benchmark/T64_fr8_statcalib_extension_lane_20260527_221658 runs/p4_benchmark/T66_statcalib_sensitivity_20260529_210906 runs/p4_benchmark/T67_statcalib_teacher_anchor_20260601_225718 runs/p4_benchmark/T68_statcalib_generated_only_20260605_205723` returned empty
2. no historical `runs/` artifact was modified or rewritten

## Scenario-By-Scenario Outcome

| Scenario | Best StatCalib Candidate Set | Best LER Mean | Candidate Ranking |
| --- | --- | ---: | --- |
| `static_bias_theta` | `window_variance_t001 = t003 = t005` | `0.430274` | `window_variance_t001 = t003 = t005 > ekf_t001` |
| `linear_ramp` | `window_variance_t001 = t003 = t005` | `0.466227` | `window_variance_t001 = t003 = t005 > ekf_t001` |
| `step_sigma_theta` | `window_variance_t001 = t003 = t005` | `0.458064` | `window_variance_t001 = t003 = t005 > ekf_t001` |
| `periodic_drift` | `window_variance_t001 = t003 = t005` | `0.438105` | `window_variance_t001 = t003 = t005 > ekf_t001` |

Scenario interpretation:

1. every scenario-best statcalib candidate beats both frozen anchors
2. every scenario-best set is exactly the same three-way tie
3. `statcalib_ekf_t001` remains competitive and clean, but it loses to the three `window_variance` candidates in all four scenarios

## Candidate Aggregate Summary

| Candidate | Teacher | Threshold | Mean LER Mean | Worst-Scenario LER | Generated Rows | Mixed Rows | Beats Both Frozen Anchors | Full Generated-Only |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `statcalib_window_variance_t001` | `window_variance` | `0.001` | `0.448168` | `0.466227` | `4` | `0` | `4/4` | `True` |
| `statcalib_window_variance_t003` | `window_variance` | `0.003` | `0.448168` | `0.466227` | `4` | `0` | `4/4` | `True` |
| `statcalib_window_variance_t005` | `window_variance` | `0.005` | `0.448168` | `0.466227` | `4` | `0` | `4/4` | `True` |
| `statcalib_ekf_t001` | `ekf` | `0.001` | `0.449341` | `0.466866` | `4` | `0` | `4/4` | `True` |

Generated-only conclusion:

1. all four T69 candidates remain full `generated`
2. all four still beat both frozen anchors in all four scenarios
3. the clean tie-break question is therefore no longer about generated-vs-mixed status
4. it is now purely about whether the `window_variance` triplet collapses internally

## Pairwise Head-To-Head Outcome

The pairwise table is fully decisive:

1. `statcalib_window_variance_t001` vs `statcalib_window_variance_t003`: `4` scenario ties
2. `statcalib_window_variance_t001` vs `statcalib_window_variance_t005`: `4` scenario ties
3. `statcalib_window_variance_t003` vs `statcalib_window_variance_t005`: `4` scenario ties
4. each of the three `window_variance` candidates beats `statcalib_ekf_t001` in all `4/4` scenarios

That means the T69 tie-break did not narrow the `window_variance` triplet at all.

## Grouped Clean-Winner Tie-Break Summary

### T68 tie-set comparison

Preserved T68 clean tie set:

- `statcalib_window_variance_t001`
- `statcalib_window_variance_t003`
- `statcalib_window_variance_t005`

T69 current clean answer set:

- `statcalib_window_variance_t001`
- `statcalib_window_variance_t003`
- `statcalib_window_variance_t005`

Relation relative to T68:

- `persists`

### Mean-best vs worst-case-best

- mean-best candidates: `statcalib_window_variance_t001 = statcalib_window_variance_t003 = statcalib_window_variance_t005`
- worst-case-best candidates: `statcalib_window_variance_t001 = statcalib_window_variance_t003 = statcalib_window_variance_t005`
- relation: `same`

This is stronger than T68 in one narrow sense:

1. in T68, worst-case-best was wider than mean-best because `window_variance_t010` still matched the worst-case LER
2. in T69, the four-candidate clean tie-break matrix closes that gap
3. but it still does **not** produce one unique clean reference point

### Final classification

T69 final clean-winner classification:

- `persistent_clean_tie_set`

Unique clean reference point:

- does **not** exist after T69

The bounded honest final answer is therefore:

1. the strongest clean answer is still the `window_variance_t001 = t003 = t005` tie set
2. the tie is now better defended because it survives the stronger `repeats=4` budget
3. forcing one unique threshold choice would overstate what the artifacts show

## Bounded Interpretation

What T69 supports:

1. the old T68 clean tie set survives a stronger bounded repeat budget
2. all four tested candidates remain full generated-only winners against both frozen anchors
3. the `window_variance` triplet stays strictly better than the `ekf_t001` clean winner
4. the mean-best and worst-case-best candidate sets align within the T69 tie-break matrix

What T69 does not support:

1. a rewrite of the frozen `T24` ranked table
2. `.tflite` runtime validation
3. real-board validation
4. a mature calibration-comparator claim
5. paper-grade expanded benchmark evidence
6. a unique clean threshold claim

The narrow honest conclusion is:

`T69` successfully answers the bounded tie-break question, and the answer is that the clean winner set remains a persistent three-way tie. This is a valid positive outcome, but it stays extension-lane, mock-backed software-HIL evidence only.
