# T66 Explanation

## 1. Plain-language summary

T66 asks one narrow question: if we nudge the `statcalib` heuristic a little, does the T64 win disappear?

The answer is no, it does not disappear inside this bounded local grid.

This does not mean FR8 is now fully proven. It means the extension-lane result looks less like a one-point accident than it did before.

## 2. What the task was trying to do

After T64 and T65, the next unresolved question was no longer editorial. The open question was whether the `statcalib` extension-lane result was robust to a small predeclared parameter neighborhood.

T66 was designed to answer that question without changing the verified mainline:

- keep `T24` frozen and authoritative
- keep `statcalib` as a separately labeled extension lane
- keep the evidence scope at mock-backed software-HIL
- avoid any runtime or comparator semantic changes
- run only one bounded sensitivity matrix under one T66 run root

This fits the current `Phase 2: Controlled Development` rule set: move forward on validated paths, but do it in a bounded and auditable way.

## 3. What changed in the repo

The main changes are four task-local pieces.

First, a new task-local config was added:

- `cnn_fpga/config/p4_multiscenario_statcalib_sensitivity.yaml`

This config does not edit historical configs. It defines the fixed T66 matrix:

- the same four locked scenarios from T64
- the same two frozen anchors: `ukf` and `hybrid_residual_b`
- five predeclared statcalib variants:
  - `statcalib_default`
  - `statcalib_low_scale`
  - `statcalib_high_scale`
  - `statcalib_low_clip`
  - `statcalib_high_threshold`

Second, a new task-scoped summary helper was added:

- `cnn_fpga/benchmark/summarize_statcalib_sensitivity.py`

This helper reads the finished artifacts and computes:

- per-scenario winners
- per-mode logical error rates
- gaps versus `ukf`
- gaps versus `hybrid_residual_b`
- rankings among statcalib variants
- generated-window and signal-norm columns when present

Third, focused tests were added:

- `tests/test_statcalib_sensitivity_summary.py`

These tests do not try to be a universal framework. They protect the exact logic T66 depends on, such as ranking, matrix completeness, and mode validation.

Fourth, the task report was added:

- `docs/statcalib_sensitivity_bounded_benchmark.md`

That report records the boundary conditions, the execution shape, the outcome tables, and the interpretation limits.

## 4. What actually ran

The preserved run root is:

- `runs/p4_benchmark/T66_statcalib_sensitivity_20260529_210906`

The matrix size is:

- `4 scenarios x 7 modes x 2 repeats = 56 repeat-runs`

The key integrity facts I checked are:

- exactly one T66 run root exists
- `launch commit == finish commit == summary.json["git_commit"] == ad981bb`
- `comparison.csv` has 28 rows
- raw rows total 56
- `missing_runs = []`
- all comparison rows have `coverage = 1.0`
- all comparison rows have `completed_repeats = 2`

So the bounded matrix really was completed.

## 5. What the result means

The most important conclusion is not "one parameter is universally best." The important conclusion is that the extension-lane advantage survives a small local perturbation grid.

The result has two different "best" views:

- `statcalib_high_threshold` is first by aggregate mean LER
- `statcalib_default` wins 3 of the 4 scenarios and looks more stable inside the grid

That distinction matters. T66 strengthens the bounded case for statcalib, but it does not justify oversimplifying the story into "we found the one true best setting."

It also does not justify upgrading the evidence level. T66 is still:

- not `.tflite`
- not real-board
- not a rewrite of the frozen T24 main table
- not a mature calibration-comparator conclusion

## 6. Why the review result is PASS_WITH_WARNINGS

The result is not `BLOCK` because the task was genuinely completed:

- no out-of-scope source-semantic edits
- no historical artifact rewrite
- no fake implementation
- lightweight verification passed:
  - `python -m py_compile cnn_fpga/benchmark/summarize_statcalib_sensitivity.py`
  - `python -m unittest tests.test_statcalib_sensitivity_summary`
  - `python -m cnn_fpga.benchmark.summarize_statcalib_sensitivity --run-dir ...`

The result is not a clean `PASS` because one provenance caveat remains:

- the first foreground benchmark command timed out at the shell layer
- Worker then relaunched the same full-matrix command against the same run root
- this left one duplicate `running` marker in `progress.jsonl`

That does not break the final matrix, but it is less clean than the preferred repeat-range continuation shape. That is why the correct review outcome is `PASS_WITH_WARNINGS`.

There is a second interpretation warning too:

- the aggregate best variant and the most stable variant are not the same

That should stay visible in any downstream write-up.

## 7. Was Worker's own draft documentation correct

Mostly yes, but it was incomplete.

The draft review had the right direction:

- it did not overclaim the evidence
- it did not incorrectly block the task

But it was too light on two points:

- why the same-run-root full-command relaunch is a warning rather than a blocker
- why the hardcoded helper is acceptable here as a task-scoped tool rather than suspicious fake logic

The draft explanation was also too short. It said what happened, but it did not explain enough of:

- where T66 sits in the `T64 -> T65 -> T66` chain
- why the task matters for later development
- why the verdict is `PASS_WITH_WARNINGS` rather than plain `PASS`

## 8. The safest final takeaway

The plainest honest summary is:

T66 does not prove FR8 in a stronger deployment sense. What it does prove is narrower and still useful: inside this locked four-scenario, five-variant local sensitivity grid, the statcalib extension-lane advantage does not collapse immediately.

That is exactly the kind of result this stage needed: stronger than a single-point success, but still bounded and honestly labeled.
