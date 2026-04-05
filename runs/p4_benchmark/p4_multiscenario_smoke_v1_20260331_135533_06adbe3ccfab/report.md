# P4 Multi-Scenario Benchmark

## Protocol

- protocol_id: `p4_hil_smoke_v1`
- report_template_version: `p4_hil_report_v1`
- repeats: `1`
- real_board_policy: `conditional_extension`

## Scenario Comparison

| Scenario | Mode | LER Mean | LER Std | Overflow Mean | Hist Sat Mean | Commit Mean | Slow Viol Mean | Fast Viol Mean | Dominant Source | Artifact |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| static_bias_theta | Static Linear | 1.120604 | 0.000000 | 0.015625 | 0.015625 | 60.000000 | 0.000000 | 0.000008 | histogram_input |  |
| static_bias_theta | Window Variance | 0.827900 | 0.000000 | 0.019813 | 0.019813 | 60.000000 | 0.000000 | 0.000017 | histogram_input |  |
| static_bias_theta | EKF | 0.844800 | 0.000000 | 0.019958 | 0.019958 | 60.000000 | 0.000000 | 0.000008 | histogram_input |  |
| static_bias_theta | CNN-FPGA | 0.932896 | 0.000000 | 0.020454 | 0.020454 | 60.000000 | 0.000000 | 0.000021 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz |

## Delta Summary

| Scenario | Mode | dLER vs Static | dLER vs CNN | dOverflow vs Static | dOverflow vs CNN |
| --- | --- | ---: | ---: | ---: | ---: |
| static_bias_theta | Static Linear | 0.000000 | 0.187708 | 0.000000 | -0.004829 |
| static_bias_theta | Window Variance | -0.292704 | -0.104996 | 0.004188 | -0.000642 |
| static_bias_theta | EKF | -0.275804 | -0.088096 | 0.004333 | -0.000496 |
| static_bias_theta | CNN-FPGA | -0.187708 | 0.000000 | 0.004829 | 0.000000 |

## Scenario Winners

| Scenario | Best Mode | Best LER Mean | Runner-Up Gap |
| --- | --- | ---: | ---: |
| static_bias_theta | Window Variance | 0.827900 | 0.016900 |
