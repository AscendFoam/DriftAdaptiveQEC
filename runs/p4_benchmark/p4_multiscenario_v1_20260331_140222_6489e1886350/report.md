# P4 Multi-Scenario Benchmark

## Protocol

- protocol_id: `p4_hil_v1`
- report_template_version: `p4_hil_report_v1`
- repeats: `2`
- real_board_policy: `conditional_extension`

## Scenario Comparison

| Scenario | Mode | LER Mean | LER Std | Overflow Mean | Hist Sat Mean | Commit Mean | Slow Viol Mean | Fast Viol Mean | Dominant Source | Artifact |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| static_bias_theta | Static Linear | 1.129327 | 0.000017 | 0.015015 | 0.015015 | 600.000000 | 0.000000 | 0.000015 | histogram_input |  |
| static_bias_theta | Window Variance | 0.866035 | 0.000670 | 0.020518 | 0.020518 | 600.000000 | 0.000000 | 0.000012 | histogram_input |  |
| static_bias_theta | EKF | 0.863364 | 0.000281 | 0.020202 | 0.020202 | 600.000000 | 0.000000 | 0.000016 | histogram_input |  |
| static_bias_theta | CNN-FPGA | 0.965762 | 0.000044 | 0.020139 | 0.020139 | 599.500000 | 0.000000 | 0.000014 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz |
| linear_ramp | Static Linear | 1.103325 | 0.000369 | 0.014659 | 0.014659 | 599.500000 | 0.000000 | 0.000014 | histogram_input |  |
| linear_ramp | Window Variance | 0.853791 | 0.000776 | 0.019587 | 0.019587 | 599.500000 | 0.000000 | 0.000015 | histogram_input |  |
| linear_ramp | EKF | 0.852963 | 0.000079 | 0.019449 | 0.019449 | 599.500000 | 0.000000 | 0.000018 | histogram_input |  |
| linear_ramp | CNN-FPGA | 0.949238 | 0.000105 | 0.019348 | 0.019348 | 599.500000 | 0.000000 | 0.000015 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz |
| step_sigma_theta | Static Linear | 1.105959 | 0.000020 | 0.014718 | 0.014718 | 600.000000 | 0.000000 | 0.000017 | histogram_input |  |
| step_sigma_theta | Window Variance | 0.852163 | 0.000406 | 0.019775 | 0.019775 | 600.000000 | 0.000000 | 0.000014 | histogram_input |  |
| step_sigma_theta | EKF | 0.850495 | 0.000235 | 0.019464 | 0.019464 | 599.500000 | 0.000000 | 0.000014 | histogram_input |  |
| step_sigma_theta | CNN-FPGA | 0.947990 | 0.000157 | 0.019539 | 0.019539 | 600.000000 | 0.000000 | 0.000013 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz |
| periodic_drift | Static Linear | 1.109243 | 0.000181 | 0.014820 | 0.014820 | 600.000000 | 0.000000 | 0.000017 | histogram_input |  |
| periodic_drift | Window Variance | 0.857088 | 0.000314 | 0.019894 | 0.019894 | 600.000000 | 0.000000 | 0.000016 | histogram_input |  |
| periodic_drift | EKF | 0.855610 | 0.000028 | 0.019547 | 0.019547 | 600.000000 | 0.000000 | 0.000013 | histogram_input |  |
| periodic_drift | CNN-FPGA | 0.954215 | 0.000361 | 0.019617 | 0.019617 | 600.000000 | 0.000000 | 0.000017 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz |

## Delta Summary

| Scenario | Mode | dLER vs Static | dLER vs CNN | dOverflow vs Static | dOverflow vs CNN |
| --- | --- | ---: | ---: | ---: | ---: |
| static_bias_theta | Static Linear | 0.000000 | 0.163566 | 0.000000 | -0.005123 |
| static_bias_theta | Window Variance | -0.263292 | -0.099726 | 0.005503 | 0.000379 |
| static_bias_theta | EKF | -0.265963 | -0.102398 | 0.005187 | 0.000063 |
| static_bias_theta | CNN-FPGA | -0.163566 | 0.000000 | 0.005123 | 0.000000 |
| linear_ramp | Static Linear | 0.000000 | 0.154086 | 0.000000 | -0.004689 |
| linear_ramp | Window Variance | -0.249534 | -0.095448 | 0.004929 | 0.000240 |
| linear_ramp | EKF | -0.250362 | -0.096276 | 0.004790 | 0.000101 |
| linear_ramp | CNN-FPGA | -0.154086 | 0.000000 | 0.004689 | 0.000000 |
| step_sigma_theta | Static Linear | 0.000000 | 0.157970 | 0.000000 | -0.004822 |
| step_sigma_theta | Window Variance | -0.253796 | -0.095827 | 0.005058 | 0.000236 |
| step_sigma_theta | EKF | -0.255465 | -0.097495 | 0.004746 | -0.000076 |
| step_sigma_theta | CNN-FPGA | -0.157970 | 0.000000 | 0.004822 | 0.000000 |
| periodic_drift | Static Linear | 0.000000 | 0.155028 | 0.000000 | -0.004796 |
| periodic_drift | Window Variance | -0.252154 | -0.097126 | 0.005074 | 0.000278 |
| periodic_drift | EKF | -0.253633 | -0.098605 | 0.004726 | -0.000070 |
| periodic_drift | CNN-FPGA | -0.155028 | 0.000000 | 0.004796 | 0.000000 |

## Scenario Winners

| Scenario | Best Mode | Best LER Mean | Runner-Up Gap |
| --- | --- | ---: | ---: |
| static_bias_theta | EKF | 0.863364 | 0.002671 |
| linear_ramp | EKF | 0.852963 | 0.000828 |
| step_sigma_theta | EKF | 0.850495 | 0.001668 |
| periodic_drift | EKF | 0.855610 | 0.001478 |
