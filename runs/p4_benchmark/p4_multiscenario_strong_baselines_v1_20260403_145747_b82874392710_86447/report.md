# P4 Multi-Scenario Benchmark

## Protocol

- protocol_id: `p4_strong_baselines_v1`
- report_template_version: `p4_hil_report_v1`
- repeats: `4`
- real_board_policy: `conditional_extension`

## Scenario Comparison

| Scenario | Mode | LER Mean | LER Std | Overflow Mean | Hist Sat Mean | Commit Mean | Slow Viol Mean | Fast Viol Mean | Dominant Source | Artifact |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| static_bias_theta | EKF | 0.838399 | 0.000762 | 0.002670 | 0.002670 | 899.750000 | 0.000000 | 0.000015 | histogram_input |  |
| static_bias_theta | UKF | 0.826767 | 0.000558 | 0.002655 | 0.002655 | 899.750000 | 0.000000 | 0.000016 | histogram_input |  |
| static_bias_theta | Constant Residual-Mu | 0.835738 | 0.000840 | 0.002705 | 0.002705 | 900.000000 | 0.000000 | 0.000016 | histogram_input |  |
| static_bias_theta | RLS Residual-B | 0.838028 | 0.000382 | 0.002672 | 0.002672 | 900.000000 | 0.000000 | 0.000015 | histogram_input |  |
| static_bias_theta | Hybrid Residual-B | 0.811530 | 0.002574 | 0.002658 | 0.002658 | 899.500000 | 0.000000 | 0.000016 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/runtime_b_residual_v1/tiny_cnn_20260401_083648_2fc740424c0d.npz |
| linear_ramp | EKF | 0.820161 | 0.000456 | 0.002466 | 0.002466 | 900.000000 | 0.000000 | 0.000016 | histogram_input |  |
| linear_ramp | UKF | 0.810263 | 0.002583 | 0.002459 | 0.002459 | 900.000000 | 0.000000 | 0.000016 | histogram_input |  |
| linear_ramp | Constant Residual-Mu | 0.817122 | 0.000852 | 0.002496 | 0.002496 | 899.500000 | 0.000000 | 0.000015 | histogram_input |  |
| linear_ramp | RLS Residual-B | 0.819421 | 0.000670 | 0.002484 | 0.002484 | 900.000000 | 0.000000 | 0.000016 | histogram_input |  |
| linear_ramp | Hybrid Residual-B | 0.787960 | 0.001258 | 0.002414 | 0.002414 | 900.000000 | 0.000000 | 0.000016 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/runtime_b_residual_v1/tiny_cnn_20260401_083648_2fc740424c0d.npz |
| step_sigma_theta | EKF | 0.821733 | 0.000313 | 0.002519 | 0.002519 | 900.000000 | 0.000000 | 0.000016 | histogram_input |  |
| step_sigma_theta | UKF | 0.812898 | 0.001769 | 0.002510 | 0.002510 | 900.000000 | 0.000000 | 0.000014 | histogram_input |  |
| step_sigma_theta | Constant Residual-Mu | 0.819342 | 0.000577 | 0.002537 | 0.002537 | 900.000000 | 0.000000 | 0.000015 | histogram_input |  |
| step_sigma_theta | RLS Residual-B | 0.821399 | 0.000112 | 0.002506 | 0.002506 | 900.000000 | 0.000000 | 0.000016 | histogram_input |  |
| step_sigma_theta | Hybrid Residual-B | 0.787161 | 0.001492 | 0.002486 | 0.002486 | 900.000000 | 0.000000 | 0.000014 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/runtime_b_residual_v1/tiny_cnn_20260401_083648_2fc740424c0d.npz |
| periodic_drift | EKF | 0.833184 | 0.000220 | 0.002613 | 0.002613 | 899.250000 | 0.000000 | 0.000015 | histogram_input |  |
| periodic_drift | UKF | 0.821967 | 0.000317 | 0.002605 | 0.002605 | 900.000000 | 0.000000 | 0.000015 | histogram_input |  |
| periodic_drift | Constant Residual-Mu | 0.830672 | 0.000433 | 0.002661 | 0.002661 | 899.750000 | 0.000000 | 0.000015 | histogram_input |  |
| periodic_drift | RLS Residual-B | 0.832786 | 0.000938 | 0.002618 | 0.002618 | 900.000000 | 0.000000 | 0.000017 | histogram_input |  |
| periodic_drift | Hybrid Residual-B | 0.806678 | 0.000410 | 0.002584 | 0.002584 | 900.000000 | 0.000000 | 0.000016 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/runtime_b_residual_v1/tiny_cnn_20260401_083648_2fc740424c0d.npz |

## Delta Summary

| Scenario | Mode | dLER vs Static | dLER vs CNN | dOverflow vs Static | dOverflow vs CNN |
| --- | --- | ---: | ---: | ---: | ---: |
| static_bias_theta | EKF |  |  |  |  |
| static_bias_theta | UKF |  |  |  |  |
| static_bias_theta | Constant Residual-Mu |  |  |  |  |
| static_bias_theta | RLS Residual-B |  |  |  |  |
| static_bias_theta | Hybrid Residual-B |  |  |  |  |
| linear_ramp | EKF |  |  |  |  |
| linear_ramp | UKF |  |  |  |  |
| linear_ramp | Constant Residual-Mu |  |  |  |  |
| linear_ramp | RLS Residual-B |  |  |  |  |
| linear_ramp | Hybrid Residual-B |  |  |  |  |
| step_sigma_theta | EKF |  |  |  |  |
| step_sigma_theta | UKF |  |  |  |  |
| step_sigma_theta | Constant Residual-Mu |  |  |  |  |
| step_sigma_theta | RLS Residual-B |  |  |  |  |
| step_sigma_theta | Hybrid Residual-B |  |  |  |  |
| periodic_drift | EKF |  |  |  |  |
| periodic_drift | UKF |  |  |  |  |
| periodic_drift | Constant Residual-Mu |  |  |  |  |
| periodic_drift | RLS Residual-B |  |  |  |  |
| periodic_drift | Hybrid Residual-B |  |  |  |  |

## Scenario Winners

| Scenario | Best Mode | Best LER Mean | Runner-Up Gap |
| --- | --- | ---: | ---: |
| static_bias_theta | Hybrid Residual-B | 0.811530 | 0.015237 |
| linear_ramp | Hybrid Residual-B | 0.787960 | 0.022304 |
| step_sigma_theta | Hybrid Residual-B | 0.787161 | 0.025737 |
| periodic_drift | Hybrid Residual-B | 0.806678 | 0.015289 |
