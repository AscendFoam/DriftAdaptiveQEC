# T69: FR8 statcalib clean-winner tie-break bounded benchmark

## Status

- Proposed by Captain on `2026-06-08`
- Current phase: `Phase 2: Controlled Development`
- Decision state: `Go`
- Task type: bounded benchmark execution + tie-break/stability summary task on the mainline experiment branch

## Why This Task Exists

`T64` established one clean-provenance bounded statcalib extension-lane win under the locked four-scenario protocol.

`T66` then showed that the win is not just one accidental local threshold point under the `ukf` teacher anchor.

`T67` then showed that the bounded gain is not grossly dependent on `teacher_mode=ukf`.

`T68` then answered the next narrower question honestly:

1. full generated-only statcalib winners do exist inside the predeclared grid
2. the strongest clean answer is the `window_variance_t001 = t003 = t005` tie set
3. `statcalib_ekf_t001` is also a full generated-only winner, but weaker on mean LER

That means the next unresolved mainline question is narrower again:

1. the repository no longer needs to ask whether any clean winner exists
2. the repository still does not know whether the clean-winner tie set collapses under a slightly stronger bounded repeat budget
3. if it does not collapse, the repository should stop pretending a unique final threshold exists and report the tie set honestly

The next smallest honest step is therefore not more prose, not `.tflite`, not real-board work, and not a broader benchmark expansion. It is one bounded clean-winner tie-break benchmark that:

1. keeps the historical `T24` frozen ranked table authoritative
2. keeps `statcalib` as a separately labeled extension lane only
3. reuses only the four full generated-only winners discovered by `T68`
4. slightly strengthens the repeat budget without widening scenarios, teachers, or runtime semantics
5. preserves clean provenance and paired-seed fairness
6. produces one grouped summary pack aimed directly at the remaining `R24` tie-set question

## Goal

Produce one bounded clean-winner tie-break package that answers:

1. whether the four `T68` full generated-only winners remain full generated-only under a slightly stronger bounded repeat budget
2. whether the `window_variance_t001 = t003 = t005` tie set persists, collapses to a smaller tie set, or collapses to one unique clean reference point
3. whether any single candidate becomes both mean-best and worst-case-best while remaining full generated-only across all four locked scenarios
4. if no single candidate emerges, whether the honest final answer should remain an explicit clean tie set rather than a forced single-threshold claim

## Allowed Files

Worker may modify only:

- `docs/tasks/Phase2/T69_fr8_statcalib_clean_winner_tiebreak_bounded_benchmark.md`
- `docs/evidence_packs/statcalib_fr8/statcalib_clean_winner_tiebreak_bounded_benchmark.md`
- `docs/review/T69_review.md`
- `docs/for_human/T69_explanation.md`
- `docs/worker_summary/T69_worker_summary.md`
- `cnn_fpga/config/p4_multiscenario_statcalib_clean_winner_tiebreak.yaml`
- `cnn_fpga/benchmark/summarize_statcalib_clean_winner_tiebreak.py`
- `tests/test_statcalib_clean_winner_tiebreak_summary.py`
- exactly one new run root under `runs/p4_benchmark/T69_statcalib_clean_winner_tiebreak_*`

## Docs To Update

Worker must update:

- `docs/evidence_packs/statcalib_fr8/statcalib_clean_winner_tiebreak_bounded_benchmark.md`
- `docs/review/T69_review.md`
- `docs/for_human/T69_explanation.md`
- `docs/worker_summary/T69_worker_summary.md`

Worker must not update governance docs. Captain will do that after review.

## Forbidden Scope

Worker must not:

- modify `docs/02_experiment_plan.md`
- modify any governance doc under `docs/00_*` to `docs/08_*`
- modify `cnn_fpga/decoder/statcalib.py`
- modify `cnn_fpga/runtime/slow_loop_runtime.py`
- modify `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py`
- modify historical config files such as `cnn_fpga/config/p4_multiscenario_strong_baselines.yaml`, `cnn_fpga/config/p4_multiscenario_statcalib_extension_lane.yaml`, `cnn_fpga/config/p4_multiscenario_statcalib_sensitivity.yaml`, `cnn_fpga/config/p4_multiscenario_statcalib_teacher_anchor.yaml`, or `cnn_fpga/config/p4_multiscenario_statcalib_generated_only.yaml`
- modify any historical file under `runs/`
- create more than one `T69` run root
- change benchmark semantics, comparator semantics, or runtime semantics
- widen the candidate search beyond the four `T68` full generated-only winners explicitly declared below
- widen into `.tflite`, real-board, training, cleanup, or theory-only branch work
- rewrite `T24`, `T64`, `T65`, `T66`, `T67`, or `T68` into mature calibration-comparator, deployment, or paper-grade evidence

## Required Inputs

Worker must reuse:

- `docs/protocols/benchmark/P4_benchmark_formal_protocol.md`
- `docs/evidence_packs/statcalib_fr8/fr8_statcalib_extension_lane_benchmark.md`
- `docs/evidence_packs/statcalib_fr8/fr8_statcalib_extension_lane_consistency_audit.md`
- `docs/evidence_packs/statcalib_fr8/statcalib_sensitivity_bounded_benchmark.md`
- `docs/evidence_packs/statcalib_fr8/statcalib_teacher_anchor_bounded_benchmark.md`
- `docs/evidence_packs/statcalib_fr8/statcalib_generated_only_robustness_bounded_benchmark.md`
- `docs/review/T64_review.md`
- `docs/review/T65_review.md`
- `docs/review/T66_review.md`
- `docs/review/T67_review.md`
- `docs/review/T68_review.md`
- `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743`
- `runs/p4_benchmark/T64_fr8_statcalib_extension_lane_20260527_221658`
- `runs/p4_benchmark/T66_statcalib_sensitivity_20260529_210906`
- `runs/p4_benchmark/T67_statcalib_teacher_anchor_20260601_225718`
- `runs/p4_benchmark/T68_statcalib_generated_only_20260605_205723`
- current mainline benchmark runner behavior without semantic edits

## Fixed Boundary

- Branch: clean committed `main`
- Evidence scope: mock-backed software-HIL only
- Historical anchor: `T24` remains the authoritative frozen ranked table
- Extension-lane rule: `statcalib` remains separately labeled and appended outside the historical frozen ranked set
- Fairness rule: `--paired-seeds` must remain enabled
- Repeat rule: `repeats=4`

## Locked Tie-Break Matrix

### Frozen scenarios

- `static_bias_theta`
- `linear_ramp`
- `step_sigma_theta`
- `periodic_drift`

### Frozen anchor modes

- `ukf`
- `hybrid_residual_b`

### Frozen statcalib clean-winner modes

Use exactly these four statcalib lanes:

1. `statcalib_window_variance_t001`
   - `teacher_mode = window_variance`
   - `residual_scale_b = 0.50`
   - `residual_clip_b = 0.08`
   - `signal_threshold = 0.001`
2. `statcalib_window_variance_t003`
   - `teacher_mode = window_variance`
   - `residual_scale_b = 0.50`
   - `residual_clip_b = 0.08`
   - `signal_threshold = 0.003`
3. `statcalib_window_variance_t005`
   - `teacher_mode = window_variance`
   - `residual_scale_b = 0.50`
   - `residual_clip_b = 0.08`
   - `signal_threshold = 0.005`
4. `statcalib_ekf_t001`
   - `teacher_mode = ekf`
   - `residual_scale_b = 0.50`
   - `residual_clip_b = 0.08`
   - `signal_threshold = 0.001`

### Total matrix

- `4 scenarios x 6 modes x 4 repeats = 96 repeat-runs`

## Implementation Requirements

1. Add one task-scoped derived config that expresses the full T69 matrix without changing historical configs.
2. Launch from a clean committed `main` worktree and record that provenance clearly in task-local docs.
3. Preferred execution shape:
   - one full-matrix invocation under one fixed T69 run root
4. Use a host-launch shape that is expected to outlive the local foreground shell timeout. Do not intentionally rely on killing and relaunching the identical full matrix against the same run root.
5. If an interruption still occurs, the only allowed continuation shape is:
   - same run root
   - same frozen mode/scenario matrix
   - split by repeat range only via `--repeat-start` / `--repeat-stop` / `--resume-only`
6. Mode-chunking is forbidden.
7. Scenario-chunking is forbidden.
8. The summary helper must compute at least:
   - per-scenario best statcalib candidate within the clean-winner set
   - per-candidate mean LER
   - per-candidate worst-scenario LER and best-scenario LER
   - per-candidate `generated` row count and `mixed` row count
   - per-candidate count of scenarios where the candidate beats both frozen anchors
   - whether each of the four candidates remains full generated-only across all four scenarios
   - grouped comparison of `window_variance_t001` vs `t003` vs `t005`
   - grouped comparison of `window_variance` clean-winner set versus `ekf_t001`
   - grouped comparison of mean-best candidate set versus worst-case-best candidate set
   - explicit tie persistence / tie collapse reporting relative to `T68`
   - pairwise head-to-head win table across the four statcalib candidates by scenario
   - a final classification of the clean-winner answer as exactly one of:
     - `unique_clean_reference_point`
     - `reduced_clean_tie_set`
     - `persistent_clean_tie_set`
9. The summary helper should read the preserved `T68` summary pack only for tie-set comparison; it must not rewrite historical `T68` outputs.
10. The summary helper and tests must stay task-scoped. Do not refactor mainline runner semantics inside T69.

## Expected Output Artifacts

Worker must produce:

- one run root:
  - `runs/p4_benchmark/T69_statcalib_clean_winner_tiebreak_*`
- one benchmark report:
  - `docs/evidence_packs/statcalib_fr8/statcalib_clean_winner_tiebreak_bounded_benchmark.md`
- one review file:
  - `docs/review/T69_review.md`
- one human explanation:
  - `docs/for_human/T69_explanation.md`
- one worker summary:
  - `docs/worker_summary/T69_worker_summary.md`

## Verification

Worker must run and report:

1. benchmark execution command(s) used for the T69 matrix
2. `python -m py_compile cnn_fpga/benchmark/summarize_statcalib_clean_winner_tiebreak.py`
3. `python -m unittest tests.test_statcalib_clean_winner_tiebreak_summary`
4. one invocation of the new summary helper against the final T69 run root
5. explicit confirmation that:
   - launch commit == finish commit == `summary.json git_commit`
   - exactly one T69 run root was created
   - no historical `runs/` file was modified
   - `missing_runs == []`
   - every comparison row has `coverage = 1.0` and `completed_repeats = 4`
6. explicit reporting of:
   - generated row count per candidate
   - mixed row count per candidate
   - whether each candidate remains full generated-only
   - whether the mean-best candidate set and the worst-case-best candidate set are the same or different
   - whether the `T68` clean-winner tie set persists, collapses, or reduces
   - whether a unique clean reference point exists after T69

## Review No-Go Triggers

Reviewer should return `BLOCK` if any of the following happen:

1. more than one T69 run root is created
2. worker changes statcalib/runtime/main runner semantics
3. worker chunks by mode or scenario instead of repeat range
4. worker relaunches the identical full matrix against the same run root after interruption instead of using repeat-range continuation
5. historical `T24`, `T64`, `T65`, `T66`, `T67`, or `T68` artifacts are modified or rewritten
6. report language upgrades the evidence to `.tflite`, real-board, mature calibration comparator, or paper-grade expanded benchmark
7. provenance is not clean and single-commit anchored
8. task output silently rewrites the historical frozen ranked table rather than treating statcalib as a separate extension lane
9. worker widens the candidate set beyond the two frozen anchors plus the four frozen T68 clean winners
10. helper output flattens a surviving tie without representing it explicitly

## Captain Notes

- This task is intentionally stronger than a simple follow-up, but it remains bounded.
- The point is to answer the clean-winner tie-break question honestly, not to build a generic statcalib optimizer or a paper-grade search framework by stealth.
- If T69 still ends with a persistent tie set, that is a successful task outcome as long as the result is reported honestly.
- Keep the theory-only branch completely separate.

## Worker Output

Worker must report:

1. what files were changed
2. exact benchmark command(s) executed
3. run root path
4. verification command outputs
5. explicit provenance summary
6. scenario-by-scenario outcome summary
7. grouped clean-winner tie-break summary
8. whether each candidate remains full generated-only
9. whether a unique clean reference point exists
10. remaining risks or interpretation limits

## Worker Output

### What changed

Added these 3 task-scoped files:

1. `cnn_fpga/config/p4_multiscenario_statcalib_clean_winner_tiebreak.yaml`
2. `cnn_fpga/benchmark/summarize_statcalib_clean_winner_tiebreak.py`
3. `tests/test_statcalib_clean_winner_tiebreak_summary.py`

Completed the only allowed run root:

- `runs/p4_benchmark/T69_statcalib_clean_winner_tiebreak_20260608_160358`

Wrote these 4 T69 documents:

1. `docs/evidence_packs/statcalib_fr8/statcalib_clean_winner_tiebreak_bounded_benchmark.md`
2. `docs/review/T69_review.md`
3. `docs/for_human/T69_explanation.md`
4. `docs/worker_summary/T69_worker_summary.md`

### Exact benchmark command executed

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config C:\t69cfg_20260608_160358.yaml --scenario static_bias_theta --scenario linear_ramp --scenario step_sigma_theta --scenario periodic_drift --mode ukf --mode hybrid_residual_b --mode statcalib_window_variance_t001 --mode statcalib_window_variance_t003 --mode statcalib_window_variance_t005 --mode statcalib_ekf_t001 --paired-seeds --repeats 4 --run-dir D:\Codes\Quantum\DriftAdaptiveQEC\runs\p4_benchmark\T69_statcalib_clean_winner_tiebreak_20260608_160358
```

### Verification

Ran:

1. `C:\ProgramData\anaconda3\python.exe -m py_compile cnn_fpga/benchmark/summarize_statcalib_clean_winner_tiebreak.py`
2. `C:\ProgramData\anaconda3\python.exe -m unittest tests.test_statcalib_clean_winner_tiebreak_summary`
   - output: `Ran 5 tests`, `OK`
3. `C:\ProgramData\anaconda3\python.exe -m cnn_fpga.benchmark.summarize_statcalib_clean_winner_tiebreak --run-dir runs/p4_benchmark/T69_statcalib_clean_winner_tiebreak_20260608_160358`
   - output:
     - `final_clean_winner_classification=persistent_clean_tie_set`
     - `t68_clean_tie_set_relation=persists`
     - `current_clean_answer_modes=statcalib_window_variance_t001,statcalib_window_variance_t003,statcalib_window_variance_t005`
     - `mean_best_vs_worst_case_best=same`

### Explicit provenance summary

1. launch branch = `main`
2. launch `HEAD = 1dbfbc3`
3. finish branch = `main`
4. finish `HEAD = 1dbfbc3`
5. `summary.json["git_commit"] = 1dbfbc3`
6. exactly one T69 run root exists
7. `missing_runs = []`
8. comparison rows = `24`
9. raw rows = `96`
10. every comparison row has `coverage = 1.0`
11. every comparison row has `completed_repeats = 4`
12. historical `T24/T64/T66/T67/T68` run roots were not modified
13. `progress.jsonl` counts:
    - `running = 96`
    - `completed = 96`
    - duplicate `running = 0`
    - duplicate `completed = 0`

### Scenario-by-scenario outcome summary

1. `static_bias_theta`: `window_variance_t001 = t003 = t005`, best LER `0.43027388888888884`
2. `linear_ramp`: `window_variance_t001 = t003 = t005`, best LER `0.46622687500000004`
3. `step_sigma_theta`: `window_variance_t001 = t003 = t005`, best LER `0.45806395833333335`
4. `periodic_drift`: `window_variance_t001 = t003 = t005`, best LER `0.4381054166666667`

### Grouped clean-winner tie-break summary

1. `T68` clean-winner tie set was:
   - `statcalib_window_variance_t001`
   - `statcalib_window_variance_t003`
   - `statcalib_window_variance_t005`
2. `T69` current clean answer set is still:
   - `statcalib_window_variance_t001`
   - `statcalib_window_variance_t003`
   - `statcalib_window_variance_t005`
3. relation relative to `T68` = `persists`
4. mean-best candidate set = worst-case-best candidate set
5. final classification = `persistent_clean_tie_set`

### Candidate generated/mixed status

1. `statcalib_window_variance_t001`
   - generated rows = `4`
   - mixed rows = `0`
   - full generated-only = `True`
2. `statcalib_window_variance_t003`
   - generated rows = `4`
   - mixed rows = `0`
   - full generated-only = `True`
3. `statcalib_window_variance_t005`
   - generated rows = `4`
   - mixed rows = `0`
   - full generated-only = `True`
4. `statcalib_ekf_t001`
   - generated rows = `4`
   - mixed rows = `0`
   - full generated-only = `True`

### Unique clean reference point

- Does a unique clean reference point exist after T69? `No`

### Remaining risks

1. `T69` resolves the tie-break question honestly, but the answer is still a persistent tie, not a unique threshold.
2. Evidence remains mock-backed software-HIL extension-lane evidence only.
3. `T24` remains the authoritative frozen ranked table and must not be rewritten using T69.
