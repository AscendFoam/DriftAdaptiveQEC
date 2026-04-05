# P4 Multi-Scenario Benchmark

## Protocol

- protocol_id: `p4_strong_baselines_v1`
- report_template_version: `p4_hil_report_v1`
- repeats: `4`
- real_board_policy: `conditional_extension`

## Scenario Comparison

| Scenario | Mode | LER Mean | LER Std | Overflow Mean | Hist Sat Mean | Commit Mean | Slow Viol Mean | Fast Viol Mean | Dominant Source | Artifact |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| static_bias_theta | EKF | 0.838285 | 0.000696 | 0.002689 | 0.002689 | 899.750000 | 0.000000 | 0.000015 | histogram_input |  |
| static_bias_theta | Constant Residual-Mu | 0.836932 | 0.000473 | 0.002688 | 0.002688 | 899.750000 | 0.000000 | 0.000016 | histogram_input |  |
| static_bias_theta | RLS Residual-B | 0.837807 | 0.000353 | 0.002656 | 0.002656 | 900.000000 | 0.000000 | 0.000016 | histogram_input |  |
| static_bias_theta | Hybrid Residual-B | 0.811007 | 0.001789 | 0.002660 | 0.002660 | 900.000000 | 0.000000 | 0.000015 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/runtime_b_residual_v1/tiny_cnn_20260401_083648_2fc740424c0d.npz |
| linear_ramp | EKF | 0.819411 | 0.000417 | 0.002474 | 0.002474 | 900.000000 | 0.000000 | 0.000016 | histogram_input |  |
| linear_ramp | Constant Residual-Mu | 0.817105 | 0.000891 | 0.002492 | 0.002492 | 900.000000 | 0.000000 | 0.000016 | histogram_input |  |
| linear_ramp | RLS Residual-B | 0.819814 | 0.000186 | 0.002449 | 0.002449 | 899.500000 | 0.000000 | 0.000015 | histogram_input |  |
| linear_ramp | Hybrid Residual-B | 0.788664 | 0.000869 | 0.002420 | 0.002420 | 899.750000 | 0.000000 | 0.000016 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/runtime_b_residual_v1/tiny_cnn_20260401_083648_2fc740424c0d.npz |
| step_sigma_theta | EKF | 0.821874 | 0.000762 | 0.002503 | 0.002503 | 900.000000 | 0.000000 | 0.000016 | histogram_input |  |
| step_sigma_theta | Constant Residual-Mu | 0.818792 | 0.001049 | 0.002553 | 0.002553 | 900.000000 | 0.000000 | 0.000014 | histogram_input |  |
| step_sigma_theta | RLS Residual-B | 0.821393 | 0.000936 | 0.002505 | 0.002505 | 900.000000 | 0.000000 | 0.000015 | histogram_input |  |
| step_sigma_theta | Hybrid Residual-B | 0.787850 | 0.001627 | 0.002476 | 0.002476 | 900.000000 | 0.000000 | 0.000016 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/runtime_b_residual_v1/tiny_cnn_20260401_083648_2fc740424c0d.npz |
| periodic_drift | EKF | 0.832557 | 0.000525 | 0.002621 | 0.002621 | 899.250000 | 0.000000 | 0.000015 | histogram_input |  |
| periodic_drift | Constant Residual-Mu | 0.830753 | 0.000463 | 0.002608 | 0.002608 | 900.000000 | 0.000000 | 0.000015 | histogram_input |  |
| periodic_drift | RLS Residual-B | 0.832319 | 0.000657 | 0.002614 | 0.002614 | 899.750000 | 0.000000 | 0.000015 | histogram_input |  |
| periodic_drift | Hybrid Residual-B | 0.807282 | 0.000598 | 0.002589 | 0.002589 | 900.000000 | 0.000000 | 0.000017 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/runtime_b_residual_v1/tiny_cnn_20260401_083648_2fc740424c0d.npz |

## Delta Summary

| Scenario | Mode | dLER vs Static | dLER vs CNN | dOverflow vs Static | dOverflow vs CNN |
| --- | --- | ---: | ---: | ---: | ---: |
| static_bias_theta | EKF |  |  |  |  |
| static_bias_theta | Constant Residual-Mu |  |  |  |  |
| static_bias_theta | RLS Residual-B |  |  |  |  |
| static_bias_theta | Hybrid Residual-B |  |  |  |  |
| linear_ramp | EKF |  |  |  |  |
| linear_ramp | Constant Residual-Mu |  |  |  |  |
| linear_ramp | RLS Residual-B |  |  |  |  |
| linear_ramp | Hybrid Residual-B |  |  |  |  |
| step_sigma_theta | EKF |  |  |  |  |
| step_sigma_theta | Constant Residual-Mu |  |  |  |  |
| step_sigma_theta | RLS Residual-B |  |  |  |  |
| step_sigma_theta | Hybrid Residual-B |  |  |  |  |
| periodic_drift | EKF |  |  |  |  |
| periodic_drift | Constant Residual-Mu |  |  |  |  |
| periodic_drift | RLS Residual-B |  |  |  |  |
| periodic_drift | Hybrid Residual-B |  |  |  |  |

## Scenario Winners

| Scenario | Best Mode | Best LER Mean | Runner-Up Gap |
| --- | --- | ---: | ---: |
| static_bias_theta | Hybrid Residual-B | 0.811007 | 0.025925 |
| linear_ramp | Hybrid Residual-B | 0.788664 | 0.028440 |
| step_sigma_theta | Hybrid Residual-B | 0.787850 | 0.030942 |
| periodic_drift | Hybrid Residual-B | 0.807282 | 0.023472 |
