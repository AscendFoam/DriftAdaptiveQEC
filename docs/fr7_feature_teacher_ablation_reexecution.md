# FR7 Feature/Teacher Ablation Re-Execution

## Scope

This report covers only T57: the bounded FR7 feature/teacher ablation re-execution under the locked T24 protocol.

It does not claim:

- causal proof
- expanded benchmark evidence
- `.tflite` validation
- real-board validation
- retraining or new model creation

## Run Commands And Run Root

Run root:

- `runs/p4_benchmark/T57_fr7_feature_teacher_ablation_20260524_000000`

Execution commands:

1. `C:\ProgramData\anaconda3\python.exe -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config cnn_fpga/config/p4_hybrid_vs_ukf_ablation_features.yaml --scenario static_bias_theta --scenario linear_ramp --scenario step_sigma_theta --scenario periodic_drift --mode ukf --mode hybrid_full --mode hybrid_no_hist_deltas --mode hybrid_no_teacher_prediction --mode hybrid_no_teacher_params --mode hybrid_no_teacher_deltas --paired-seeds --repeats 2 --run-dir runs/p4_benchmark/T57_fr7_feature_teacher_ablation_20260524_000000 --repeat-start 0 --repeat-stop 1`
2. `C:\ProgramData\anaconda3\python.exe -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config cnn_fpga/config/p4_hybrid_vs_ukf_ablation_features.yaml --scenario static_bias_theta --scenario linear_ramp --scenario step_sigma_theta --scenario periodic_drift --mode ukf --mode hybrid_full --mode hybrid_no_hist_deltas --mode hybrid_no_teacher_prediction --mode hybrid_no_teacher_params --mode hybrid_no_teacher_deltas --paired-seeds --repeats 2 --run-dir runs/p4_benchmark/T57_fr7_feature_teacher_ablation_20260524_000000 --repeat-start 1 --repeat-stop 2`
3. `C:\ProgramData\anaconda3\python.exe -m cnn_fpga.benchmark.summarize_p4_features_ablation --run-dir runs/p4_benchmark/T57_fr7_feature_teacher_ablation_20260524_000000 --output-dir runs/p4_benchmark/T57_fr7_feature_teacher_ablation_20260524_000000/summary_pack`

Key outputs:

- `summary.json`
- `comparison.csv`
- `delta.csv`
- `teacher_scalar_diagnostics.csv`
- `report.md`
- `summary_pack/summary.json`
- `summary_pack/table.csv`
- `summary_pack/scenario_rows.csv`
- `provenance_manifest.json`

Note:

- `launch_plan.json` reflects the second execution chunk because the runner rewrites it on each launch.
- The full two-chunk execution history is preserved in `provenance_manifest.json`, `progress.jsonl`, `chunk0_python.out.log`, and `chunk1_python.out.log`.

## Provenance Matrix

| Category | Path(s) | T57 treatment |
| --- | --- | --- |
| Frozen config | `cnn_fpga/config/p4_hybrid_vs_ukf_ablation_features.yaml` | Reused unchanged |
| Full hybrid artifact | `artifacts/models/runtime_b_residual_v1/tiny_cnn_20260401_083648_2fc740424c0d.npz` | Reused |
| No-hist artifact | `artifacts/models/runtime_b_residual_no_hist_deltas_v1/tiny_cnn_20260404_194053_ecb0245eb9c1.npz` | Reused |
| No-teacher-pred artifact | `artifacts/models/runtime_b_residual_no_teacher_prediction_v1/tiny_cnn_20260404_201238_b8929e477b83.npz` | Reused |
| No-teacher-params artifact | `artifacts/models/runtime_b_residual_no_teacher_params_v1/tiny_cnn_20260404_205257_51db70f8c56a.npz` | Reused |
| No-teacher-deltas artifact | `artifacts/models/runtime_b_residual_no_teacher_deltas_v1/tiny_cnn_20260404_211944_25f3b7ddfa8d.npz` | Reused |
| UKF baseline | No model artifact | Executed as baseline mode |
| Historical pre-T24 FR7 run | `runs/p4_benchmark/p4_hybrid_vs_ukf_ablation_features_v1_20260404_211945_a7d2984ad50e_23637` | Reference only; not used as current evidence |
| T57 outputs | `runs/p4_benchmark/T57_fr7_feature_teacher_ablation_20260524_000000/*` | Newly generated inside the single allowed run root |

## Protocol Conformance

| Boundary item | Required | Observed |
| --- | --- | --- |
| Protocol family | T24 frozen-set feature ablation | `p4_hybrid_vs_ukf_ablation_features_v1` |
| Scenarios | 4 fixed scenarios | `static_bias_theta`, `linear_ramp`, `step_sigma_theta`, `periodic_drift` |
| Modes | 6 fixed modes | `ukf`, `hybrid_full`, `hybrid_no_hist_deltas`, `hybrid_no_teacher_prediction`, `hybrid_no_teacher_params`, `hybrid_no_teacher_deltas` |
| Repeats | `2` | `2` |
| Seed policy | paired seeds | preserved |
| Model handling | reuse only | preserved |
| Output location | one T57-scoped run root | preserved |

## Outcome Summary

Mode-level summary from `summary_pack/table.csv`:

| Mode | Avg LER | dLER vs UKF | dLER vs Hybrid Full | Avg overflow |
| --- | ---: | ---: | ---: | ---: |
| `ukf` | 0.817382 | 0.000000 | +0.018837 | 0.002564 |
| `hybrid_full` | 0.798545 | -0.018837 | 0.000000 | 0.002539 |
| `hybrid_no_hist_deltas` | 0.826723 | +0.009341 | +0.028178 | 0.002593 |
| `hybrid_no_teacher_prediction` | 0.807251 | -0.010131 | +0.008706 | 0.002558 |
| `hybrid_no_teacher_params` | 0.749621 | -0.067761 | -0.048924 | 0.002439 |
| `hybrid_no_teacher_deltas` | 0.800329 | -0.017053 | +0.001784 | 0.002250 |

Scenario winners:

- `static_bias_theta`: `hybrid_no_teacher_params`
- `linear_ramp`: `hybrid_no_teacher_params`
- `step_sigma_theta`: `hybrid_no_teacher_params`
- `periodic_drift`: `hybrid_no_teacher_params`

Per-channel bounded reading:

1. Removing histogram delta worsens LER against `hybrid_full` in all 4 scenarios.
2. Removing teacher prediction also worsens LER against `hybrid_full` in all 4 scenarios, but remains better than `ukf` on average.
3. Removing teacher params improves LER in all 4 scenarios and becomes the best mode throughout this bounded re-execution.
4. Removing teacher deltas is near-neutral overall, with mixed scenario sign and `aggressive_param` as the dominant overflow source instead of `histogram_input`.

## FR7 Classification

FR7 should now be classified as `ready` for a bounded frozen-set feature/teacher ablation table.

What changed:

- historical-only caveat removed
- formal T24-lane re-execution now exists
- paper-facing FR7 evidence can cite current run artifacts rather than pre-T24 history

What did not change:

- FR7 is still not causal proof
- FR7 does not close the T56 mechanism hedge
- FR7 does not justify a simple `teacher-guided residual design explains the win` sentence

## Explicit Non-Claims And Residual Limits

Do not upgrade T57 into any of the following:

1. `teacher params are proven necessary`
2. `hybrid_full is proven architecturally optimal`
3. `teacher channels causally explain the win`
4. `FR7 closes the mechanism gap`
5. `T57 upgrades the repository to expanded benchmark evidence`

Residual limitations:

- the no-teacher-params result directly weakens the simple historical attribution story
- teacher scalar diagnostics remain `not_generated` for the broadcast teacher-feature variants, so FR7 is a result-table lane, not a scalar-mechanism lane
- FR6, TFLite, real-board, and broader benchmark gaps remain open

## Paper-Facing Implication

`docs/paper_ablation_result_pack.md` and `docs/paper_claim_evidence_ledger.md` can now truthfully cite FR7 as a present bounded result table.

The safe paper wording is descriptive:

`Under the frozen T24 feature-ablation lane, histogram delta removal degrades performance, teacher prediction removal mildly degrades performance, teacher delta removal is near-neutral/mixed, and the reused no-teacher-params variant performs best across all four scenarios.`

That wording is evidence-backed. Any stronger architectural attribution is not.
