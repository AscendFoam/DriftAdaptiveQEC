# P4 Multi-Scenario Benchmark

## Protocol

- protocol_id: `p4_hil_hybrid_b_long_v1`
- report_template_version: `p4_hil_report_v1`
- repeats: `4`
- real_board_policy: `conditional_extension`

## Scenario Comparison

| Scenario | Mode | LER Mean | LER Std | Overflow Mean | Hist Sat Mean | Commit Mean | Slow Viol Mean | Fast Viol Mean | Dominant Source | Artifact |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| static_bias_theta | EKF | 0.838382 | 0.000336 | 0.002685 | 0.002685 | 899.750000 | 0.000000 | 0.000015 | histogram_input |  |
| static_bias_theta | Constant Residual-Mu | 0.836621 | 0.000845 | 0.002689 | 0.002689 | 899.750000 | 0.000000 | 0.000015 | histogram_input |  |
| static_bias_theta | Hybrid Residual-B | 0.812559 | 0.000629 | 0.002640 | 0.002640 | 900.000000 | 0.000000 | 0.000015 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/runtime_b_residual_v1/tiny_cnn_20260401_083648_2fc740424c0d.npz |
| linear_ramp | EKF | 0.820122 | 0.000502 | 0.002467 | 0.002467 | 900.000000 | 0.000000 | 0.000016 | histogram_input |  |
| linear_ramp | Constant Residual-Mu | 0.817717 | 0.000681 | 0.002478 | 0.002478 | 900.000000 | 0.000000 | 0.000015 | histogram_input |  |
| linear_ramp | Hybrid Residual-B | 0.787913 | 0.000499 | 0.002437 | 0.002437 | 899.250000 | 0.000000 | 0.000014 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/runtime_b_residual_v1/tiny_cnn_20260401_083648_2fc740424c0d.npz |
| step_sigma_theta | EKF | 0.821444 | 0.000364 | 0.002499 | 0.002499 | 900.000000 | 0.000000 | 0.000016 | histogram_input |  |
| step_sigma_theta | Constant Residual-Mu | 0.819799 | 0.000664 | 0.002513 | 0.002513 | 900.000000 | 0.000000 | 0.000014 | histogram_input |  |
| step_sigma_theta | Hybrid Residual-B | 0.787615 | 0.000504 | 0.002481 | 0.002481 | 900.000000 | 0.000000 | 0.000015 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/runtime_b_residual_v1/tiny_cnn_20260401_083648_2fc740424c0d.npz |
| periodic_drift | EKF | 0.832484 | 0.000991 | 0.002606 | 0.002606 | 899.500000 | 0.000000 | 0.000016 | histogram_input |  |
| periodic_drift | Constant Residual-Mu | 0.830635 | 0.000626 | 0.002623 | 0.002623 | 900.000000 | 0.000000 | 0.000015 | histogram_input |  |
| periodic_drift | Hybrid Residual-B | 0.807143 | 0.001580 | 0.002608 | 0.002608 | 899.750000 | 0.000000 | 0.000015 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/runtime_b_residual_v1/tiny_cnn_20260401_083648_2fc740424c0d.npz |

## Delta Summary

| Scenario | Mode | dLER vs Static | dLER vs CNN | dOverflow vs Static | dOverflow vs CNN |
| --- | --- | ---: | ---: | ---: | ---: |
| static_bias_theta | EKF |  |  |  |  |
| static_bias_theta | Constant Residual-Mu |  |  |  |  |
| static_bias_theta | Hybrid Residual-B |  |  |  |  |
| linear_ramp | EKF |  |  |  |  |
| linear_ramp | Constant Residual-Mu |  |  |  |  |
| linear_ramp | Hybrid Residual-B |  |  |  |  |
| step_sigma_theta | EKF |  |  |  |  |
| step_sigma_theta | Constant Residual-Mu |  |  |  |  |
| step_sigma_theta | Hybrid Residual-B |  |  |  |  |
| periodic_drift | EKF |  |  |  |  |
| periodic_drift | Constant Residual-Mu |  |  |  |  |
| periodic_drift | Hybrid Residual-B |  |  |  |  |

## Scenario Winners

| Scenario | Best Mode | Best LER Mean | Runner-Up Gap |
| --- | --- | ---: | ---: |
| static_bias_theta | Hybrid Residual-B | 0.812559 | 0.024062 |
| linear_ramp | Hybrid Residual-B | 0.787913 | 0.029804 |
| step_sigma_theta | Hybrid Residual-B | 0.787615 | 0.032185 |
| periodic_drift | Hybrid Residual-B | 0.807143 | 0.023492 |
