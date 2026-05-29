# FR8 Statcalib Extension-Lane Consistency Audit

## Verdict

The T64 report is now artifact-consistent for the bounded checks required by `T65`.

- audit result: `PASS`
- checks passed: `8/8`
- audited report: `docs/fr8_statcalib_extension_lane_benchmark.md`
- audited T64 run root: `runs/p4_benchmark/T64_fr8_statcalib_extension_lane_20260527_221658`
- frozen baseline anchor: `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743`

This audit does not create new benchmark evidence. It hardens reuse discipline around the already frozen T64 result pack.

## Exact Inputs Audited

Primary inputs:

1. `docs/tasks/Phase2/T64_fr8_statcalib_extension_lane_bounded_benchmark.md`
2. `docs/fr8_statcalib_extension_lane_benchmark.md`
3. `runs/p4_benchmark/T64_fr8_statcalib_extension_lane_20260527_221658/summary.json`
4. `runs/p4_benchmark/T64_fr8_statcalib_extension_lane_20260527_221658/launch_plan.json`
5. `runs/p4_benchmark/T64_fr8_statcalib_extension_lane_20260527_221658/progress.jsonl`
6. `runs/p4_benchmark/T64_fr8_statcalib_extension_lane_20260527_221658/comparison.csv`
7. `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/comparison.csv`

Verification command:

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.audit_fr8_extension_lane_consistency --task-package docs/tasks/Phase2/T64_fr8_statcalib_extension_lane_bounded_benchmark.md --report docs/fr8_statcalib_extension_lane_benchmark.md --run-dir runs/p4_benchmark/T64_fr8_statcalib_extension_lane_20260527_221658 --frozen-baseline-run-dir runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743
```

## Checks Run

The audit helper executed these checks:

1. `task_package_execution_shapes_present`
2. `report_provenance_wording`
3. `report_execution_shape_wording`
4. `locked_boundary_preserved`
5. `paired_seed_and_repeat_policy`
6. `progress_log_duplicate_running_guard`
7. `frozen_subset_matches_t24`
8. `required_boundary_statements_present`

## Check Results

| Check | Result | Summary |
| --- | --- | --- |
| `task_package_execution_shapes_present` | `PASS` | The T64 task package still exposes the accepted execution-shape wording for full-matrix invocation and repeat-range chunking. |
| `report_provenance_wording` | `PASS` | The T64 report now distinguishes artifact-recorded fields, live repo observations, and auxiliary filesystem metadata, and no longer attributes a finish timestamp to `summary.json`. |
| `report_execution_shape_wording` | `PASS` | The T64 report now describes only the artifact-visible shape: one full-matrix invocation, not repeat-range chunked, not resumed, and without inferring foreground vs detached transport. |
| `locked_boundary_preserved` | `PASS` | The four locked scenarios, the frozen five-mode order, and the appended `statcalib` sixth lane all still match the preserved T64 artifacts. |
| `paired_seed_and_repeat_policy` | `PASS` | `paired_seeds=true`, `repeats=2`, `resume_only=false`, and repeat range `0..2` remain preserved. |
| `progress_log_duplicate_running_guard` | `PASS` | `progress.jsonl` still has no duplicate `running` record for the same `(scenario, mode, repeat)` key. |
| `frozen_subset_matches_t24` | `PASS` | The T64 frozen five-mode subset still matches T24 on all 20 frozen rows for `final_ler_mean` and `overflow_rate_mean`. |
| `required_boundary_statements_present` | `PASS` | The report still states mock-backed software-HIL only, separate extension lane only, not a rewrite of `T24`, not `.tflite`, and not real-board. |

## Outcome Interpretation

Yes, the T64 result pack can now be reused as a self-audited bounded extension-lane artifact in the narrow sense required by `T65`:

1. the report wording matches the preserved artifact set
2. frozen-subset preservation against `T24` is now checked by code, not only by manual review
3. the reuse boundary remains explicit and narrow

No, this audit does not promote T64 into:

1. a rewrite of the historical `T24` frozen ranked table
2. `.tflite` validation
3. real-board validation
4. paper-grade expanded benchmark evidence
5. mature calibration-comparator validation

## Mandatory Boundary Wording After Audit

These boundary statements remain mandatory after T65:

1. mock-backed software-HIL only
2. separate extension lane only
3. not a rewrite of `T24`
4. not `.tflite`
5. not real-board
6. not a mature calibration comparator claim
