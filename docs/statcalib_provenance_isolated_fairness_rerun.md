# T62 Statcalib Provenance-Isolated Fairness Rerun

## Verdict

The bounded `statcalib` fairness signal persisted, and this time the provenance blocker was actually closed at the task boundary.

- launch branch: `main`
- launch `HEAD`: `e2773d3`
- finish branch: `main`
- finish `HEAD`: `e2773d3`
- `summary.json git_commit`: `e2773d3`
- duplicate `running` entries in `progress.jsonl`: none

So `T62` repaired the specific blocker that caused `T61` to fail.

This is still:

- mock-backed software HIL only
- not `FR8`
- not `.tflite` validation
- not real-board validation

## Preflight Result

Preflight before the rerun:

- launch timestamp: `2026-05-27 12:29:18 +08:00`
- `git branch --show-current`: `main`
- `git status --short`: no repo status entries; only two warnings about inaccessible global ignore config at `C:\Users\26410/.config/git/ignore`
- `git rev-parse --short HEAD`: `e2773d3`

This satisfied the T62 `clean committed main` requirement.

## Exact Rerun Command

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config cnn_fpga/config/p4_multiscenario_statcalib_smoke.yaml --scenario static_bias_theta --scenario linear_ramp --mode ukf --mode hybrid_residual_b --mode statcalib --paired-seeds --repeats 2 --run-dir runs/p4_benchmark/T62_statcalib_provenance_isolated_20260527_122943
```

Execution notes:

- one foreground invocation only
- no same-run resume
- no second invocation against the same T62 run root
- no branch/worktree movement during execution

## Exact Run Root

`runs/p4_benchmark/T62_statcalib_provenance_isolated_20260527_122943`

Only one T62-scoped run root exists under `runs/p4_benchmark/`.

## Post-Run Provenance Validation

Post-run state:

- finish timestamp: `2026-05-27 16:19:26 +08:00`
- `git branch --show-current`: `main`
- `git rev-parse --short HEAD`: `e2773d3`
- `summary.json["git_commit"]`: `e2773d3`

All three commit anchors match:

1. launch `HEAD`
2. finish `HEAD`
3. `summary.json git_commit`

`progress.jsonl` check:

- duplicate `running` entry for the same `(scenario, mode, repeat)` key: `[]`

So the specific T61 failure mode is absent in T62.

## T59 vs T61 vs T62 Comparison

### Per-Mode Table

| Scenario | Mode | T59 `final_ler_mean` | T61 `final_ler_mean` | T62 `final_ler_mean` | T62 `final_ler_std` | T62 status | T62 reason | T62 generated windows |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| `static_bias_theta` | `ukf` | 0.825007500 | 0.824683958 | 0.824683958 | 0.000323542 | `not_applicable` | `mode_does_not_emit_statcalib` | 0.0 |
| `static_bias_theta` | `hybrid_residual_b` | 0.808795833 | 0.810047083 | 0.810047083 | 0.001251250 | `not_applicable` | `mode_does_not_emit_statcalib` | 0.0 |
| `static_bias_theta` | `statcalib` | 0.431530417 | 0.430785417 | 0.430785417 | 0.000745000 | `generated` | `statcalib_params_emitted` | 600.0 |
| `linear_ramp` | `ukf` | 0.817952917 | 0.819855000 | 0.819855000 | 0.001902083 | `not_applicable` | `mode_does_not_emit_statcalib` | 0.0 |
| `linear_ramp` | `hybrid_residual_b` | 0.803187500 | 0.804060833 | 0.804060833 | 0.000873333 | `not_applicable` | `mode_does_not_emit_statcalib` | 0.0 |
| `linear_ramp` | `statcalib` | 0.445084583 | 0.445927917 | 0.445927917 | 0.000843333 | `generated` | `statcalib_params_emitted` | 600.0 |

T62 output integrity:

- `missing_runs_count=0`
- all six rows have `coverage=1.0`
- all six rows have `completed_repeats=2`

### Per-Scenario Ranking

| Scenario | T59 winner | T59 gap | T61 winner | T61 gap | T62 winner | T62 gap | Interpretation |
| --- | --- | ---: | --- | ---: | --- | ---: | --- |
| `static_bias_theta` | `statcalib` | 0.377265417 | `statcalib` | 0.379261667 | `statcalib` | 0.379261667 | persisted; same as T61 |
| `linear_ramp` | `statcalib` | 0.358102917 | `statcalib` | 0.358132917 | `statcalib` | 0.358132917 | persisted; same as T61 |

## What Persisted, Weakened, Or Collapsed

The strong `statcalib` signal persisted again.

- It did not collapse in either scenario.
- It did not lose `generated` status.
- It did not lose emitted-window coverage.
- The winner ranking stayed unchanged.
- The runner-up gap stayed large.
- Numerically, T62 matched T61 exactly at the aggregated comparison-row level.

So the story after T62 is:

- the bounded comparator signal still looks strong
- the provenance blocker from T61 is now closed for this task

## What Is Closed And What Is Not

Closed by T62:

1. clean-start `main` preflight
2. uninterrupted one-shot execution
3. launch / finish / summary commit identity match
4. no same-run resume noise in `progress.jsonl`

Not closed by T62:

1. `FR8` formal comparator evidence
2. `.tflite` runtime validation
3. real-board validation
4. any claim above mock-backed software-HIL bounded sanity evidence

## Next Honest Step Before Any Later FR8 Work

The provenance blocker targeted by `T62` is now closed. So the next honest step is no longer another automatic provenance retry.

The next step should be a user/Captain decision on whether to open a bounded `FR8` gate discussion.

That is still only a gate discussion, not an automatic promotion to a formal result table. Any later `FR8` task would still need to decide whether the current `statcalib` lane definition and bounded evidence are sufficient for a defendable comparator claim.
