# T68 Review

- Verdict: `PASS_WITH_WARNINGS`

I inspected the T68 task package, the task-scoped config/helper/test changes, the preserved run root at `runs/p4_benchmark/T68_statcalib_generated_only_20260605_205723`, the generated T68 summary pack, and the lightweight verification outputs. The bounded task goal was completed. I did not find edits to `cnn_fpga/decoder/statcalib.py`, `cnn_fpga/runtime/slow_loop_runtime.py`, `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py`, or historical `T24/T64/T66/T67` artifacts.

## Blocking issues

- None.

## Non-blocking issues

1. T68 closes the generated-only existence question, but it does not identify one unique final threshold.
   - By mean LER, the strongest clean set is a three-way tie:
     - `statcalib_window_variance_t001`
     - `statcalib_window_variance_t003`
     - `statcalib_window_variance_t005`
   - By worst-case LER, that set expands to a four-way tie that also includes `statcalib_window_variance_t010`.
   - Downstream summaries should preserve this tie structure instead of collapsing T68 into one uniquely tuned threshold.

2. Some predeclared candidates remain mixed even though the task itself succeeded.
   - `statcalib_window_variance_t010` is mixed on:
     - `step_sigma_theta`
     - `periodic_drift`
   - `statcalib_ekf_t003`
   - `statcalib_ekf_t005`
   - `statcalib_ekf_t010`
     are mixed on:
     - `periodic_drift`
   - This does not weaken the core T68 answer, because the task asked whether any full generated-only winner exists, and the answer is yes.
   - It does mean the whole bounded grid is not uniformly clean.

3. The benchmark was launched from a clean short-path clone at `C:\t68cf2b`, not from the active workspace.
   - That is the correct provenance-preserving move for this repo and this task package.
   - Later retellings should keep this launch boundary visible so the clean-commit claim remains auditable.

## Missing tests

- No blocking test gap found for T68.

Recommended follow-up tests:

1. add a negative test for duplicate comparison rows
2. add a negative test for `coverage != 1.0`
3. add a negative test for `completed_repeats != 2`
4. add a negative test for `summary.json["missing_runs"] != []`

## Suspicious implementation details

1. `cnn_fpga/benchmark/summarize_statcalib_generated_only.py` is intentionally task-scoped and hardcodes:
   - the four T68 scenarios
   - the two frozen anchors
   - the two teacher anchors
   - the four thresholds
   This is correct for T68 and should not be mistaken for generic benchmark infrastructure.

2. The helper now represents equal mean-LER winners explicitly with tie strings.
   - That is a real improvement over the T67 helper nuance.
   - It is also why the T68 report can honestly preserve multi-winner structure instead of flattening it.

3. The evidence remains bounded mock-backed software-HIL only.
   - It does not upgrade the result into `.tflite`, real-board, or mature comparator claims.
   - It does not rewrite the frozen `T24` five-mode main table.

4. The preserved run evidence is real, not a hand-written summary.
   - `python -m py_compile cnn_fpga/benchmark/summarize_statcalib_generated_only.py` passed.
   - `python -m unittest tests.test_statcalib_generated_only_summary` passed with `Ran 7 tests`, `OK`.
   - `python -m cnn_fpga.benchmark.summarize_statcalib_generated_only --run-dir runs/p4_benchmark/T68_statcalib_generated_only_20260605_205723` passed.
   - `progress.jsonl` shows `running=80`, `completed=80`, duplicate `running=0`, duplicate `completed=0`.

## Recommended next action

Accept T68 as `PASS_WITH_WARNINGS`.

The strongest bounded reading is now:

1. fully generated-only statcalib winners do exist inside the predeclared T68 grid
2. the strongest clean winners come from the `window_variance` teacher anchor at `t001/t003/t005`
3. the result remains extension-lane, mock-backed software-HIL evidence only

If T68 is cited downstream, keep these boundaries attached:

1. `T24` remains the authoritative frozen ranked table
2. `statcalib` remains a separately labeled extension lane
3. the result is not `.tflite` evidence
4. the result is not real-board evidence
5. the winner set is tied, not unique
