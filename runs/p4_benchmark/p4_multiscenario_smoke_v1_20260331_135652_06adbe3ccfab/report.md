# P4 Multi-Scenario Benchmark

## Protocol

- protocol_id: `p4_hil_smoke_v1`
- report_template_version: `p4_hil_report_v1`
- repeats: `1`
- real_board_policy: `conditional_extension`

## Scenario Comparison

| Scenario | Mode | LER Mean | LER Std | Overflow Mean | Hist Sat Mean | Commit Mean | Slow Viol Mean | Fast Viol Mean | Dominant Source | Artifact |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| static_bias_theta | Static Linear | 1.119642 | 0.000000 | 0.015850 | 0.015850 | 60.000000 | 0.000000 | 0.000008 | histogram_input |  |
| static_bias_theta | Window Variance | 0.830296 | 0.000000 | 0.020692 | 0.020692 | 60.000000 | 0.000000 | 0.000017 | histogram_input |  |
| static_bias_theta | EKF | 0.849237 | 0.000000 | 0.019775 | 0.019775 | 60.000000 | 0.000000 | 0.000008 | histogram_input |  |
| static_bias_theta | CNN-FPGA | 0.928879 | 0.000000 | 0.020137 | 0.020137 | 60.000000 | 0.000000 | 0.000021 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz |
| linear_ramp | Static Linear | 1.167067 | 0.000000 | 0.016879 | 0.016879 | 60.000000 | 0.000000 | 0.000008 | histogram_input |  |
| linear_ramp | Window Variance | 0.835942 | 0.000000 | 0.021500 | 0.021500 | 60.000000 | 0.000000 | 0.000008 | histogram_input |  |
| linear_ramp | EKF | 0.863296 | 0.000000 | 0.021725 | 0.021725 | 60.000000 | 0.000000 | 0.000017 | histogram_input |  |
| linear_ramp | CNN-FPGA | 0.945079 | 0.000000 | 0.021321 | 0.021321 | 60.000000 | 0.000000 | 0.000013 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz |
| step_sigma_theta | Static Linear | 1.160862 | 0.000000 | 0.016746 | 0.016746 | 60.000000 | 0.000000 | 0.000013 | histogram_input |  |
| step_sigma_theta | Window Variance | 0.839617 | 0.000000 | 0.021425 | 0.021425 | 60.000000 | 0.000000 | 0.000021 | histogram_input |  |
| step_sigma_theta | EKF | 0.858838 | 0.000000 | 0.021450 | 0.021450 | 60.000000 | 0.000000 | 0.000013 | histogram_input |  |
| step_sigma_theta | CNN-FPGA | 0.941379 | 0.000000 | 0.021212 | 0.021212 | 60.000000 | 0.000000 | 0.000017 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz |
| periodic_drift | Static Linear | 1.075000 | 0.000000 | 0.014992 | 0.014992 | 60.000000 | 0.000000 | 0.000000 | histogram_input |  |
| periodic_drift | Window Variance | 0.811300 | 0.000000 | 0.019104 | 0.019104 | 60.000000 | 0.000000 | 0.000021 | histogram_input |  |
| periodic_drift | EKF | 0.829633 | 0.000000 | 0.018417 | 0.018417 | 60.000000 | 0.000000 | 0.000004 | histogram_input |  |
| periodic_drift | CNN-FPGA | 0.907937 | 0.000000 | 0.018937 | 0.018937 | 60.000000 | 0.000000 | 0.000029 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz |

## Delta Summary

| Scenario | Mode | dLER vs Static | dLER vs CNN | dOverflow vs Static | dOverflow vs CNN |
| --- | --- | ---: | ---: | ---: | ---: |
| static_bias_theta | Static Linear | 0.000000 | 0.190762 | 0.000000 | -0.004287 |
| static_bias_theta | Window Variance | -0.289346 | -0.098583 | 0.004842 | 0.000554 |
| static_bias_theta | EKF | -0.270404 | -0.079642 | 0.003925 | -0.000362 |
| static_bias_theta | CNN-FPGA | -0.190762 | 0.000000 | 0.004287 | 0.000000 |
| linear_ramp | Static Linear | 0.000000 | 0.221988 | 0.000000 | -0.004442 |
| linear_ramp | Window Variance | -0.331125 | -0.109137 | 0.004621 | 0.000179 |
| linear_ramp | EKF | -0.303771 | -0.081783 | 0.004846 | 0.000404 |
| linear_ramp | CNN-FPGA | -0.221988 | 0.000000 | 0.004442 | 0.000000 |
| step_sigma_theta | Static Linear | 0.000000 | 0.219483 | 0.000000 | -0.004467 |
| step_sigma_theta | Window Variance | -0.321246 | -0.101762 | 0.004679 | 0.000213 |
| step_sigma_theta | EKF | -0.302025 | -0.082542 | 0.004704 | 0.000238 |
| step_sigma_theta | CNN-FPGA | -0.219483 | 0.000000 | 0.004467 | 0.000000 |
| periodic_drift | Static Linear | 0.000000 | 0.167063 | 0.000000 | -0.003946 |
| periodic_drift | Window Variance | -0.263700 | -0.096637 | 0.004112 | 0.000167 |
| periodic_drift | EKF | -0.245367 | -0.078304 | 0.003425 | -0.000521 |
| periodic_drift | CNN-FPGA | -0.167063 | 0.000000 | 0.003946 | 0.000000 |

## Scenario Winners

| Scenario | Best Mode | Best LER Mean | Runner-Up Gap |
| --- | --- | ---: | ---: |
| static_bias_theta | Window Variance | 0.830296 | 0.018942 |
| linear_ramp | Window Variance | 0.835942 | 0.027354 |
| step_sigma_theta | Window Variance | 0.839617 | 0.019221 |
| periodic_drift | Window Variance | 0.811300 | 0.018333 |
