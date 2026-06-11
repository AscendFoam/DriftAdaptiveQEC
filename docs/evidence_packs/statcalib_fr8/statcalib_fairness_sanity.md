# T61 Statcalib Fairness Sanity

## Verdict

The bounded fairness sanity signal persisted, but the clean-provenance goal did not fully close.

- `statcalib` remained the winner in both locked scenarios.
- `statcalib_status=generated`, `statcalib_reason=statcalib_params_emitted`, and `statcalib_generated_windows_mean=600.0` remained stable.
- The next honest step is **not** `FR8`.
- The remaining blocker is provenance isolation, not comparator collapse.

This document remains within the software-only evidence boundary:

- mock-backed software HIL only
- not `FR8`
- not `.tflite` validation
- not real-board validation

## Preflight Provenance Check

Preflight was executed before the rerun started.

- `git status --short`: no repo status entries; only two warnings about inaccessible global ignore config at `C:\Users\26410/.config/git/ignore`
- preflight `git rev-parse --short HEAD`: `9174065`
- clean-start branch at preflight: `main`

So the rerun did start from a clean committed worktree state.

## Exact Rerun Command

The bounded matrix was executed with the existing T59 config, paired seeds, and `repeats=2`.

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config cnn_fpga/config/p4_multiscenario_statcalib_smoke.yaml --scenario static_bias_theta --scenario linear_ramp --mode ukf --mode hybrid_residual_b --mode statcalib --paired-seeds --repeats 2 --run-dir runs/p4_benchmark/T61_statcalib_fairness_sanity_20260527_015239
```

Operational note:

- the first foreground invocation created the single T61 run root and advanced partway through the matrix
- the host command timed out before `summary.json` was written
- the same CLI was then invoked again against the same `--run-dir`
- the benchmark resumed completed repeats in place and finished without creating a second T61 run root

## Exact Run Root

`runs/p4_benchmark/T61_statcalib_fairness_sanity_20260527_015239`

Only one T61-scoped run root exists under `runs/p4_benchmark/`.

## Git Anchoring Seen In The New Run

The completed `summary.json` records:

- `git_commit=6058f42`

This does **not** match the clean-start preflight anchor (`9174065`).

`git reflog` shows why:

- `9174065 HEAD@{2026-05-27 01:50:45 +0800}: commit: docs, test, refactor: 完成T60收口并推进T61任务`
- `6058f42 HEAD@{2026-05-27 09:43:26 +0800}: checkout: moving from main to codex-pro-research-governance-plan`

So the run started from clean `HEAD=9174065`, but a branch checkout happened while the long benchmark was still running. The benchmark writes `summary.json` at the end of execution, and `run_p4_multiscenario_benchmark.py` records `git_commit` at that end-of-run stage rather than at launch time.

This matters because `git diff --name-only 9174065 6058f42 -- cnn_fpga tests` is not empty. The two commits differ in benchmark/runtime/config/test paths, so this T61 run does **not** fully repair the provenance weakness even though the fairness signal itself stayed stable.

## T59 vs T61 Comparison

### Per-Mode Table

| Scenario | Mode | T59 `final_ler_mean` | T61 `final_ler_mean` | T61 `final_ler_std` | Delta (`T61-T59`) | T61 status | T61 reason | T61 generated windows |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| `static_bias_theta` | `ukf` | 0.825007500 | 0.824683958 | 0.000323542 | -0.000323542 | `not_applicable` | `mode_does_not_emit_statcalib` | 0.0 |
| `static_bias_theta` | `hybrid_residual_b` | 0.808795833 | 0.810047083 | 0.001251250 | 0.001251250 | `not_applicable` | `mode_does_not_emit_statcalib` | 0.0 |
| `static_bias_theta` | `statcalib` | 0.431530417 | 0.430785417 | 0.000745000 | -0.000745000 | `generated` | `statcalib_params_emitted` | 600.0 |
| `linear_ramp` | `ukf` | 0.817952917 | 0.819855000 | 0.001902083 | 0.001902083 | `not_applicable` | `mode_does_not_emit_statcalib` | 0.0 |
| `linear_ramp` | `hybrid_residual_b` | 0.803187500 | 0.804060833 | 0.000873333 | 0.000873333 | `not_applicable` | `mode_does_not_emit_statcalib` | 0.0 |
| `linear_ramp` | `statcalib` | 0.445084583 | 0.445927917 | 0.000843333 | 0.000843333 | `generated` | `statcalib_params_emitted` | 600.0 |

All six T61 rows recorded:

- `coverage=1.0`
- `completed_repeats=2`
- `missing_runs_count=0`

### Per-Scenario Ranking

| Scenario | T59 winner | T59 runner-up gap | T61 winner | T61 runner-up gap | Interpretation |
| --- | --- | ---: | --- | ---: | --- |
| `static_bias_theta` | `statcalib` | 0.377265417 | `statcalib` | 0.379261667 | persisted, slightly stronger |
| `linear_ramp` | `statcalib` | 0.358102917 | `statcalib` | 0.358132917 | persisted, effectively unchanged |

## What Persisted, Weakened, Or Collapsed

The strong T59 `statcalib` result **persisted** under the T61 bounded rerun.

- It did not collapse in either scenario.
- It did not lose `generated` status.
- It did not lose emitted-window coverage.
- The ranking stayed the same in both scenarios.
- The runner-up gap stayed large and nearly unchanged.

So the remaining issue after T61 is not fairness collapse. The remaining issue is clean provenance anchoring for a long-running execution.

## What Still Remains Before Any Later FR8 Task

Before any future `FR8`-style result-table discussion, the project still needs another bounded prerequisite:

1. isolate execution from branch/worktree movement during the full run
2. capture launch-time commit identity in the run artifacts, not only end-of-run `git_commit`
3. rerun the same bounded matrix under that stricter provenance setup if the project wants a clean comparator gate

Until that is done, this T61 result can honestly support only the following statement:

`statcalib` still looks unusually strong in this bounded mock-backed software-HIL sanity matrix, so the lane still deserves further gated discussion.

It cannot yet support:

- a formal comparator-ranking claim
- an `FR8` result-table claim
- any `.tflite` deployment claim
- any real-board behavior claim
