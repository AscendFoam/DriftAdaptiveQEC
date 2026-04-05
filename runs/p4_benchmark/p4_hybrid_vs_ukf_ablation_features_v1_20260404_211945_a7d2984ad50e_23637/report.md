# P4 Multi-Scenario Benchmark

## Protocol

- protocol_id: `p4_hybrid_vs_ukf_ablation_features_v1`
- report_template_version: `p4_hil_report_v1`
- repeats: `4`
- real_board_policy: `conditional_extension`

## Scenario Comparison

| Scenario | Mode | LER Mean | LER Std | Overflow Mean | Hist Sat Mean | Commit Mean | Slow Viol Mean | Fast Viol Mean | Dominant Source | Artifact |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| static_bias_theta | UKF | 0.826590 | 0.000808 | 0.002677 | 0.002677 | 900.000000 | 0.000000 | 0.000016 | histogram_input |  |
| static_bias_theta | Hybrid Full | 0.810757 | 0.000922 | 0.002638 | 0.002638 | 899.750000 | 0.000000 | 0.000016 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/runtime_b_residual_v1/tiny_cnn_20260401_083648_2fc740424c0d.npz |
| static_bias_theta | Hybrid No HistDelta | 0.837235 | 0.000727 | 0.002692 | 0.002692 | 900.000000 | 0.000000 | 0.000016 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/runtime_b_residual_no_hist_deltas_v1/tiny_cnn_20260404_194053_ecb0245eb9c1.npz |
| static_bias_theta | Hybrid No TeacherPred | 0.814637 | 0.000923 | 0.002660 | 0.002660 | 900.000000 | 0.000000 | 0.000014 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/runtime_b_residual_no_teacher_prediction_v1/tiny_cnn_20260404_201238_b8929e477b83.npz |
| static_bias_theta | Hybrid No TeacherParams | 0.734263 | 0.001585 | 0.002505 | 0.002505 | 899.500000 | 0.000000 | 0.000016 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/runtime_b_residual_no_teacher_params_v1/tiny_cnn_20260404_205257_51db70f8c56a.npz |
| static_bias_theta | Hybrid No TeacherDelta | 0.801496 | 0.000385 | 0.002297 | 0.002297 | 899.750000 | 0.000000 | 0.000017 | aggressive_param | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/runtime_b_residual_no_teacher_deltas_v1/tiny_cnn_20260404_211944_25f3b7ddfa8d.npz |
| linear_ramp | UKF | 0.809234 | 0.000661 | 0.002439 | 0.002439 | 900.000000 | 0.000000 | 0.000016 | histogram_input |  |
| linear_ramp | Hybrid Full | 0.788336 | 0.000899 | 0.002449 | 0.002449 | 900.000000 | 0.000000 | 0.000015 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/runtime_b_residual_v1/tiny_cnn_20260401_083648_2fc740424c0d.npz |
| linear_ramp | Hybrid No HistDelta | 0.817267 | 0.000332 | 0.002504 | 0.002504 | 899.250000 | 0.000000 | 0.000017 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/runtime_b_residual_no_hist_deltas_v1/tiny_cnn_20260404_194053_ecb0245eb9c1.npz |
| linear_ramp | Hybrid No TeacherPred | 0.801789 | 0.000232 | 0.002461 | 0.002461 | 899.750000 | 0.000000 | 0.000016 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/runtime_b_residual_no_teacher_prediction_v1/tiny_cnn_20260404_201238_b8929e477b83.npz |
| linear_ramp | Hybrid No TeacherParams | 0.760859 | 0.001109 | 0.002388 | 0.002388 | 900.000000 | 0.000000 | 0.000015 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/runtime_b_residual_no_teacher_params_v1/tiny_cnn_20260404_205257_51db70f8c56a.npz |
| linear_ramp | Hybrid No TeacherDelta | 0.800800 | 0.000538 | 0.002212 | 0.002212 | 899.750000 | 0.000000 | 0.000017 | aggressive_param | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/runtime_b_residual_no_teacher_deltas_v1/tiny_cnn_20260404_211944_25f3b7ddfa8d.npz |
| step_sigma_theta | UKF | 0.815113 | 0.002540 | 0.002482 | 0.002482 | 900.000000 | 0.000000 | 0.000016 | histogram_input |  |
| step_sigma_theta | Hybrid Full | 0.787033 | 0.001108 | 0.002483 | 0.002483 | 899.750000 | 0.000000 | 0.000014 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/runtime_b_residual_v1/tiny_cnn_20260401_083648_2fc740424c0d.npz |
| step_sigma_theta | Hybrid No HistDelta | 0.819309 | 0.001415 | 0.002547 | 0.002547 | 900.000000 | 0.000000 | 0.000016 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/runtime_b_residual_no_hist_deltas_v1/tiny_cnn_20260404_194053_ecb0245eb9c1.npz |
| step_sigma_theta | Hybrid No TeacherPred | 0.802357 | 0.001017 | 0.002508 | 0.002508 | 900.000000 | 0.000000 | 0.000015 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/runtime_b_residual_no_teacher_prediction_v1/tiny_cnn_20260404_201238_b8929e477b83.npz |
| step_sigma_theta | Hybrid No TeacherParams | 0.756925 | 0.002327 | 0.002426 | 0.002426 | 900.000000 | 0.000000 | 0.000015 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/runtime_b_residual_no_teacher_params_v1/tiny_cnn_20260404_205257_51db70f8c56a.npz |
| step_sigma_theta | Hybrid No TeacherDelta | 0.801696 | 0.000430 | 0.002251 | 0.002251 | 899.250000 | 0.000000 | 0.000013 | aggressive_param | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/runtime_b_residual_no_teacher_deltas_v1/tiny_cnn_20260404_211944_25f3b7ddfa8d.npz |
| periodic_drift | UKF | 0.821388 | 0.000413 | 0.002585 | 0.002585 | 899.250000 | 0.000000 | 0.000013 | histogram_input |  |
| periodic_drift | Hybrid Full | 0.807292 | 0.000499 | 0.002586 | 0.002586 | 900.000000 | 0.000000 | 0.000016 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/runtime_b_residual_v1/tiny_cnn_20260401_083648_2fc740424c0d.npz |
| periodic_drift | Hybrid No HistDelta | 0.831878 | 0.000281 | 0.002630 | 0.002630 | 899.750000 | 0.000000 | 0.000015 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/runtime_b_residual_no_hist_deltas_v1/tiny_cnn_20260404_194053_ecb0245eb9c1.npz |
| periodic_drift | Hybrid No TeacherPred | 0.811439 | 0.000829 | 0.002611 | 0.002611 | 900.000000 | 0.000000 | 0.000016 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/runtime_b_residual_no_teacher_prediction_v1/tiny_cnn_20260404_201238_b8929e477b83.npz |
| periodic_drift | Hybrid No TeacherParams | 0.745696 | 0.001556 | 0.002469 | 0.002469 | 900.000000 | 0.000000 | 0.000016 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/runtime_b_residual_no_teacher_params_v1/tiny_cnn_20260404_205257_51db70f8c56a.npz |
| periodic_drift | Hybrid No TeacherDelta | 0.797900 | 0.000344 | 0.002252 | 0.002252 | 899.500000 | 0.000000 | 0.000015 | aggressive_param | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/runtime_b_residual_no_teacher_deltas_v1/tiny_cnn_20260404_211944_25f3b7ddfa8d.npz |

## Delta Summary

| Scenario | Mode | dLER vs Static | dLER vs CNN | dOverflow vs Static | dOverflow vs CNN |
| --- | --- | ---: | ---: | ---: | ---: |
| static_bias_theta | UKF |  |  |  |  |
| static_bias_theta | Hybrid Full |  |  |  |  |
| static_bias_theta | Hybrid No HistDelta |  |  |  |  |
| static_bias_theta | Hybrid No TeacherPred |  |  |  |  |
| static_bias_theta | Hybrid No TeacherParams |  |  |  |  |
| static_bias_theta | Hybrid No TeacherDelta |  |  |  |  |
| linear_ramp | UKF |  |  |  |  |
| linear_ramp | Hybrid Full |  |  |  |  |
| linear_ramp | Hybrid No HistDelta |  |  |  |  |
| linear_ramp | Hybrid No TeacherPred |  |  |  |  |
| linear_ramp | Hybrid No TeacherParams |  |  |  |  |
| linear_ramp | Hybrid No TeacherDelta |  |  |  |  |
| step_sigma_theta | UKF |  |  |  |  |
| step_sigma_theta | Hybrid Full |  |  |  |  |
| step_sigma_theta | Hybrid No HistDelta |  |  |  |  |
| step_sigma_theta | Hybrid No TeacherPred |  |  |  |  |
| step_sigma_theta | Hybrid No TeacherParams |  |  |  |  |
| step_sigma_theta | Hybrid No TeacherDelta |  |  |  |  |
| periodic_drift | UKF |  |  |  |  |
| periodic_drift | Hybrid Full |  |  |  |  |
| periodic_drift | Hybrid No HistDelta |  |  |  |  |
| periodic_drift | Hybrid No TeacherPred |  |  |  |  |
| periodic_drift | Hybrid No TeacherParams |  |  |  |  |
| periodic_drift | Hybrid No TeacherDelta |  |  |  |  |

## Scenario Winners

| Scenario | Best Mode | Best LER Mean | Runner-Up Gap |
| --- | --- | ---: | ---: |
| static_bias_theta | Hybrid No TeacherParams | 0.734263 | 0.067233 |
| linear_ramp | Hybrid No TeacherParams | 0.760859 | 0.027477 |
| step_sigma_theta | Hybrid No TeacherParams | 0.756925 | 0.030107 |
| periodic_drift | Hybrid No TeacherParams | 0.745696 | 0.052205 |
