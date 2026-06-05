# T67 Review

- Verdict: `PASS_WITH_WARNINGS`

I inspected the T67 task package, the task-local config/helper/test changes, the preserved run root at `runs/p4_benchmark/T67_statcalib_teacher_anchor_20260601_225718`, the generated T67 summary pack, and the lightweight verification outputs. The bounded task goal was completed. I did not find edits to `cnn_fpga/decoder/statcalib.py`, `cnn_fpga/runtime/slow_loop_runtime.py`, `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py`, or historical `T24/T64/T66` artifacts.

## Blocking issues

- None.

## Non-blocking issues

1. The active worktree still contains one scope-external modified file: `docs/汇报用/5月汇报材料/note-draft逐段口头汇报解释.pdf`.
   - I did not treat this as a blocker because the T67 report and `host_launch_meta.json` consistently show that Worker isolated the benchmark launch in a clean short-path clone at `C:\t67c` instead of touching the unrelated file.
   - It still weakens pure diff attribution. Future review passes are cleaner if unrelated edits are separated before task review.

2. `cnn_fpga/benchmark/summarize_statcalib_teacher_anchor.py` does not represent exact ties explicitly for `better_parameter_point_by_mean_ler`.
   - The helper chooses `"high_threshold"` whenever `default_mean_ler == high_threshold_mean_ler`.
   - In the preserved T67 summary pack, the `ekf` teacher-anchor row is a tie on mean LER, while the machine-readable field still says `"high_threshold"`.
   - The prose report correctly describes this as a tie, so the issue is small, but the helper output and the prose are not perfectly aligned.

3. Two comparison rows remain `statcalib_status = mixed`:
   - `static_bias_theta / statcalib_high_threshold_teacher_window_variance`
   - `step_sigma_theta / statcalib_high_threshold_teacher_ukf`
   This does not block T67, but downstream summaries should not flatten T67 into a fully clean generated-only claim.

## Missing tests

- No blocking test gap found for T67.

Recommended follow-up test additions:

1. Add a unit test for the equal-mean tie case in `better_parameter_point_by_mean_ler`, so the helper either emits `tie` explicitly or the report wording stays consistent with the helper output.
2. Add a negative test for `summary.json["missing_runs"] != []`.
3. Add negative tests for duplicate comparison rows and for unexpected `coverage` or `completed_repeats` values.

## Suspicious implementation details

1. The summary helper is intentionally task-scoped and hardcodes the T67 scenario set, frozen anchors, parameter points, and teacher anchors.
   - That is correct for this task.
   - It should not be mistaken for generic benchmark infrastructure.

2. The benchmark evidence is real, not a mock summary or handwritten table.
   - `python -m py_compile cnn_fpga/benchmark/summarize_statcalib_teacher_anchor.py` passed.
   - `python -m unittest tests.test_statcalib_teacher_anchor_summary` passed with `Ran 6 tests`, `OK`.
   - `python -m cnn_fpga.benchmark.summarize_statcalib_teacher_anchor --run-dir runs/p4_benchmark/T67_statcalib_teacher_anchor_20260601_225718` passed.
   - `progress.jsonl` shows `running=64`, `completed=64`, duplicate `running=0`.

3. The preserved result is still bounded mock-backed software-HIL evidence only.
   - It does not upgrade the claim to `.tflite`, real-board, or a mature calibration-comparator conclusion.
   - It does not rewrite the frozen `T24` five-mode main table.

## Recommended next action

Accept T67 as `PASS_WITH_WARNINGS`.

If T67 is cited downstream, keep these boundaries attached:

1. `T24` remains the authoritative frozen ranked table.
2. `statcalib` remains a separately labeled extension lane.
3. The evidence remains mock-backed software-HIL only.
4. The two `mixed` rows and the helper tie-label nuance remain part of the provenance story.
