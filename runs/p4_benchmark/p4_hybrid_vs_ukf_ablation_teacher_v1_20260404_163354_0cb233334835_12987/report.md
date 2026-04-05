# P4 Multi-Scenario Benchmark

## Protocol

- protocol_id: `p4_hybrid_vs_ukf_ablation_teacher_v1`
- report_template_version: `p4_hil_report_v1`
- repeats: `1`
- real_board_policy: `conditional_extension`

## Scenario Comparison

| Scenario | Mode | LER Mean | LER Std | Overflow Mean | Hist Sat Mean | Commit Mean | Slow Viol Mean | Fast Viol Mean | Dominant Source | Artifact |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| static_bias_theta | UKF | 0.833766 | 0.000000 | 0.002658 | 0.002658 | 900.000000 | 0.000000 | 0.000014 | histogram_input |  |
| static_bias_theta | Hybrid Teacher=WV | 0.812919 | 0.000000 | 0.002669 | 0.002669 | 900.000000 | 0.000000 | 0.000015 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/runtime_b_residual_v1/tiny_cnn_20260401_083648_2fc740424c0d.npz |
| static_bias_theta | Hybrid Teacher=UKF | 0.809151 | 0.000000 | 0.002670 | 0.002670 | 900.000000 | 0.000000 | 0.000017 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/runtime_b_residual_teacher_ukf_v1/tiny_cnn_20260404_163353_9e2cdba7f6f3.npz |
| linear_ramp | UKF | 0.808759 | 0.000000 | 0.002448 | 0.002448 | 900.000000 | 0.000000 | 0.000014 | histogram_input |  |
| linear_ramp | Hybrid Teacher=WV | 0.788112 | 0.000000 | 0.002461 | 0.002461 | 900.000000 | 0.000000 | 0.000012 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/runtime_b_residual_v1/tiny_cnn_20260401_083648_2fc740424c0d.npz |
| linear_ramp | Hybrid Teacher=UKF | 0.796830 | 0.000000 | 0.002419 | 0.002419 | 900.000000 | 0.000000 | 0.000015 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/runtime_b_residual_teacher_ukf_v1/tiny_cnn_20260404_163353_9e2cdba7f6f3.npz |
| step_sigma_theta | UKF | 0.811714 | 0.000000 | 0.002485 | 0.002485 | 900.000000 | 0.000000 | 0.000016 | histogram_input |  |
| step_sigma_theta | Hybrid Teacher=WV | 0.787780 | 0.000000 | 0.002512 | 0.002512 | 900.000000 | 0.000000 | 0.000012 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/runtime_b_residual_v1/tiny_cnn_20260401_083648_2fc740424c0d.npz |
| step_sigma_theta | Hybrid Teacher=UKF | 0.800324 | 0.000000 | 0.002488 | 0.002488 | 900.000000 | 0.000000 | 0.000012 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/runtime_b_residual_teacher_ukf_v1/tiny_cnn_20260404_163353_9e2cdba7f6f3.npz |
| periodic_drift | UKF | 0.821809 | 0.000000 | 0.002609 | 0.002609 | 899.000000 | 0.000000 | 0.000014 | histogram_input |  |
| periodic_drift | Hybrid Teacher=WV | 0.807609 | 0.000000 | 0.002587 | 0.002587 | 900.000000 | 0.000000 | 0.000017 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/runtime_b_residual_v1/tiny_cnn_20260401_083648_2fc740424c0d.npz |
| periodic_drift | Hybrid Teacher=UKF | 0.807863 | 0.000000 | 0.002586 | 0.002586 | 900.000000 | 0.000000 | 0.000013 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/runtime_b_residual_teacher_ukf_v1/tiny_cnn_20260404_163353_9e2cdba7f6f3.npz |

## Delta Summary

| Scenario | Mode | dLER vs Static | dLER vs CNN | dOverflow vs Static | dOverflow vs CNN |
| --- | --- | ---: | ---: | ---: | ---: |
| static_bias_theta | UKF |  |  |  |  |
| static_bias_theta | Hybrid Teacher=WV |  |  |  |  |
| static_bias_theta | Hybrid Teacher=UKF |  |  |  |  |
| linear_ramp | UKF |  |  |  |  |
| linear_ramp | Hybrid Teacher=WV |  |  |  |  |
| linear_ramp | Hybrid Teacher=UKF |  |  |  |  |
| step_sigma_theta | UKF |  |  |  |  |
| step_sigma_theta | Hybrid Teacher=WV |  |  |  |  |
| step_sigma_theta | Hybrid Teacher=UKF |  |  |  |  |
| periodic_drift | UKF |  |  |  |  |
| periodic_drift | Hybrid Teacher=WV |  |  |  |  |
| periodic_drift | Hybrid Teacher=UKF |  |  |  |  |

## Scenario Winners

| Scenario | Best Mode | Best LER Mean | Runner-Up Gap |
| --- | --- | ---: | ---: |
| static_bias_theta | Hybrid Teacher=UKF | 0.809151 | 0.003769 |
| linear_ramp | Hybrid Teacher=WV | 0.788112 | 0.008718 |
| step_sigma_theta | Hybrid Teacher=WV | 0.787780 | 0.012544 |
| periodic_drift | Hybrid Teacher=WV | 0.807609 | 0.000254 |
