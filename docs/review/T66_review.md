# T66 Review

- Verdict: `PASS_WITH_WARNINGS`

I inspected the T66 task package, the current diff, the preserved run root at `runs/p4_benchmark/T66_statcalib_sensitivity_20260529_210906`, the new summary helper, the new tests, and the lightweight verification outputs. The bounded goal was completed. I did not find changes to `statcalib`, runtime, or benchmark-runner semantics, and I did not find rewrites of historical `T24` or `T64` artifacts. `T24` remains the frozen main table, and `statcalib` remains a separately labeled extension lane.

## Blocking issues

- None.

## Non-blocking issues

- After the first foreground benchmark command hit a shell timeout, Worker did not continue with `--repeat-start` / `--repeat-stop`. Instead, Worker relaunched the same full-matrix command against the same run root.
  - This did not trigger the task-package no-go for mode-chunking or scenario-chunking.
  - It is still less clean than the preferred repeat-range continuation shape, so I am keeping a warning.
  - The visible artifact is one duplicate `running` record in `progress.jsonl` for `static_bias_theta/statcalib_default/repeat_01`.

- The bounded sensitivity result has two different notions of "best":
  - By aggregate mean LER, `statcalib_high_threshold` is first.
  - By scenario wins and mean within-statcalib rank, `statcalib_default` is the more stable point.
  - The current report preserves that distinction; downstream summaries should not collapse it into one "single best parameter" claim.

- `static_bias_theta / statcalib_high_threshold` is the best row for that scenario, but its aggregate `statcalib_status` is `mixed`, not fully clean `generated`.
  - This does not invalidate T66.
  - It does mean the bounded conclusion still carries a local provenance caveat.

## Missing tests

- I did not find a test gap that should block T66. The current `tests/test_statcalib_sensitivity_summary.py` already covers:
  - aggregate ranking generation
  - per-scenario statcalib rank columns
  - the global-best-variant all-scenario guard
  - incomplete-matrix rejection
  - requested-mode mismatch rejection

- Optional follow-up tests, but not blockers:
  - `_validate_run()` rejecting `missing_runs != []`
  - `_validate_run()` rejecting duplicate comparison rows
  - `_validate_run()` rejecting `coverage != 1.0` or `completed_repeats != 2`

## Suspicious implementation details

- I did not find pseudo-implementation, fake completion, or hidden mock/stub inflation.
  - The evidence is still explicitly bounded to mock-backed software-HIL.
  - Worker did not rewrite it into `.tflite`, real-board, or mature calibration-comparator claims.

- `cnn_fpga/benchmark/summarize_statcalib_sensitivity.py` hardcodes the exact T66 scenarios, modes, and repeat count.
  - That is acceptable here because the task package explicitly asked for a task-scoped helper.
  - It is intentionally narrow helper code, not reusable generic benchmark infrastructure.

- I compared the temp launch config with the repo-preserved `cnn_fpga/config/p4_multiscenario_statcalib_sensitivity.yaml`.
  - I did not find parameter drift.
  - The observed differences are the absolute `base_config` path in the temp file and a UTF-8 BOM.
  - After normalizing those two details, the contents match.

## Recommended next action

- Accept T66 as `PASS_WITH_WARNINGS`.
  - It answered the bounded question it was supposed to answer: the T64 statcalib advantage is not just a single-point fluke inside this predeclared local grid.

- If Captain cites T66 downstream, keep three boundaries attached:
  - `T24` remains the authoritative frozen ranked table.
  - `statcalib` remains a separately labeled extension lane.
  - The evidence remains mock-backed software-HIL only; it should not be upgraded into `.tflite`, real-board, mature calibration-comparator, or paper-grade expanded benchmark claims.

- If comparator work continues, the next step should be a new bounded task, not a silent assumption that T66 already closes `R24`.
