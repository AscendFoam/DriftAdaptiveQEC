# T26 Statcalib Feasibility Gate

## Verdict

`CONDITIONAL_GO`

Statcalib is feasible only as a separately labeled comparator lane. It is not feasible as a silent insertion into the frozen T24 P4 set.

## Current Evidence Boundary

- T24 completed the frozen four-scenario, five-mode, paired-seed, `repeats=2` formal software revalidation.
- T25 accepted the result boundary as mock-backed software HIL only.
- T27-T29 improved teacher diagnostics observability and report formatting, but they did not create a statcalib baseline.
- `cnn_fpga/decoder/param_mapper.py` still defines the runtime contract as a mapping from predicted noise parameters to `DecoderRuntimeParams` (`K`, `b`).
- `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py` now preserves teacher-diagnostics status fields, but the frozen benchmark protocol still does not include statcalib in the ranked set.

## Candidate Objective

Build a compact calibration/statcalib comparator that estimates a small runtime correction from slow-loop statistics and emits the same decoder runtime contract as the existing pipeline.

The safest target is a bounded calibration lane that updates a small prior/correction representation, not a rewrite of the frozen benchmark semantics.

## Prerequisite Checklist

- Separate comparator lane with explicit label.
- Clear input contract for calibration features.
- Clear output contract compatible with `DecoderRuntimeParams`.
- Status semantics that preserve `generated`, `not_generated`, `not_applicable`, and `true_zero`.
- Focused interface smoke before any longer benchmark.
- No change to frozen scenarios, frozen baselines, seed policy, repeat policy, or metric definitions.

## Design Items

### Adopted

- Keep statcalib outside the frozen P4 ranked table.
- Reuse the existing runtime `K`/`b` contract.
- Keep null-safe reporting and explicit status fields.
- Require paired-seed reporting if statcalib is benchmarked later.

### Deferred

- Soft-information / correlation-aware extras.
- Extra drift families.
- CI-driven stopping.
- True `.tflite` runtime and real-board validation.

### Rejected

- Silent insertion into the frozen T24 comparison set.
- Rewriting `ParamMapper` semantics in place for existing modes.
- Claiming statcalib evidence from current T24-T29 runs.

## Minimal Comparator Interface

- `StatCalibInput` exact fields:
  - `window_id: int`
  - `slow_update_index: int`
  - `prior_decoder_params: DecoderRuntimeParams`
  - `histogram_summary: Dict[str, float]`
  - `calibration_features: Dict[str, float]`
  - `source: str`
  - `teacher_prediction: Dict[str, float] | None`
  - `teacher_decoder_params: DecoderRuntimeParams | None`
  - `provenance: Dict[str, Any]`
  - `metadata: Dict[str, Any]`
- `StatCalibOutput` exact fields:
  - `status: str`
  - `reason: str`
  - `source: str`
  - `K: np.ndarray | None`
  - `b: np.ndarray | None`
  - `delta_b: np.ndarray | None`
  - `provenance: Dict[str, Any]`
  - `metadata: Dict[str, Any]`
- Required status set:
  - `generated`
  - `not_generated`
  - `not_applicable`
  - `diagnostic_error`
- Required reason set:
  - `statcalib_params_emitted`
  - `insufficient_calibration_signal`
  - `mode_does_not_emit_statcalib`
  - `interface_validation_failed`
  - `statcalib_diagnostic_error`
- Implementation boundary: a separate adapter/module, not a change to `ParamMapper.map_prediction()` for the existing benchmark lanes.
- Conversion boundary: only `generated` output may convert to `DecoderRuntimeParams`.

## Metrics And Validation Plan

Future implementation should validate:

- `final_ler_mean` / `final_ler_std`
- `overflow_rate_mean`
- `histogram_input_saturation_rate_mean`
- `correction_saturation_rate_mean`
- `aggressive_param_rate_mean`
- `n_commits_applied_mean`
- `slow_update_violation_rate_mean`
- `fast_cycle_violation_rate_mean`
- teacher-diagnostics status fields

Validation order:

1. interface-shape test
2. bounded smoke for the new comparator lane
3. only then any broader benchmark extension

## Recommendation

`GO` for a future implementation task only if it keeps statcalib as a separate comparator lane with its own boundary.

`NO-GO` for any task that folds statcalib into the frozen benchmark set or changes benchmark semantics.

## Required Shape Of A Future Implementation Task

Any follow-up implementation package must explicitly include:

- `Allowed files`: new statcalib-specific source/tests/docs only
- `Forbidden scope`: no frozen-set rewrite, no scenario/baseline/seed/repeat changes, no `.tflite`, no real-board expansion
- `Verification`: interface smoke plus bounded comparator-lane validation
- `Docs to update`: task package, review doc, human explanation, and one statcalib-specific design/result doc

## Explicit Non-Claims

- T26 did not claim statcalib existed at that time. T30 later added an interface-only contract module, but this gate document still does not claim an integrated or benchmark-validated statcalib comparator exists.
- This document does not claim statcalib has been validated.
- No benchmark was run for this gate.
- At T26 gate time no source code, config, run, or artifact was changed. T30 later added `cnn_fpga/decoder/statcalib.py` and interface tests only; no benchmark run, config change, run artifact, frozen-set expansion, `.tflite` work, or real-board work followed from that.
