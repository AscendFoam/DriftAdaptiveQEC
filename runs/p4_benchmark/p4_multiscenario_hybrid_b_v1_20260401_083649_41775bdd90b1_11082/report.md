# P4 Multi-Scenario Benchmark

## Protocol

- protocol_id: `p4_hil_hybrid_b_v1`
- report_template_version: `p4_hil_report_v1`
- repeats: `2`
- real_board_policy: `conditional_extension`

## Scenario Comparison

| Scenario | Mode | LER Mean | LER Std | Overflow Mean | Hist Sat Mean | Commit Mean | Slow Viol Mean | Fast Viol Mean | Dominant Source | Artifact |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| static_bias_theta | Window Variance | 0.865047 | 0.000553 | 0.020449 | 0.020449 | 599.500000 | 0.000000 | 0.000017 | histogram_input |  |
| static_bias_theta | EKF | 0.864070 | 0.000557 | 0.020077 | 0.020077 | 600.000000 | 0.000000 | 0.000013 | histogram_input |  |
| static_bias_theta | Constant Residual-Mu | 0.862517 | 0.000333 | 0.020308 | 0.020308 | 600.000000 | 0.000000 | 0.000014 | histogram_input |  |
| static_bias_theta | CNN-FPGA | 0.965526 | 0.000420 | 0.020299 | 0.020299 | 600.000000 | 0.000000 | 0.000018 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz |
| static_bias_theta | Hybrid Residual-B | 0.859707 | 0.005309 | 0.020285 | 0.020285 | 600.000000 | 0.000000 | 0.000015 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/runtime_b_residual_v1/tiny_cnn_20260401_083648_2fc740424c0d.npz |
| linear_ramp | Window Variance | 0.853825 | 0.000247 | 0.019600 | 0.019600 | 600.000000 | 0.000000 | 0.000016 | histogram_input |  |
| linear_ramp | EKF | 0.852645 | 0.000523 | 0.019477 | 0.019477 | 600.000000 | 0.000000 | 0.000016 | histogram_input |  |
| linear_ramp | Constant Residual-Mu | 0.851993 | 0.001561 | 0.019670 | 0.019670 | 600.000000 | 0.000000 | 0.000012 | histogram_input |  |
| linear_ramp | CNN-FPGA | 0.949278 | 0.000132 | 0.019470 | 0.019470 | 600.000000 | 0.000000 | 0.000016 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz |
| linear_ramp | Hybrid Residual-B | 0.851224 | 0.001179 | 0.019713 | 0.019713 | 600.000000 | 0.000000 | 0.000017 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/runtime_b_residual_v1/tiny_cnn_20260401_083648_2fc740424c0d.npz |
| step_sigma_theta | Window Variance | 0.852199 | 0.000270 | 0.019711 | 0.019711 | 600.000000 | 0.000000 | 0.000016 | histogram_input |  |
| step_sigma_theta | EKF | 0.851172 | 0.000393 | 0.019426 | 0.019426 | 600.000000 | 0.000000 | 0.000014 | histogram_input |  |
| step_sigma_theta | Constant Residual-Mu | 0.851684 | 0.000276 | 0.019748 | 0.019748 | 600.000000 | 0.000000 | 0.000015 | histogram_input |  |
| step_sigma_theta | CNN-FPGA | 0.948073 | 0.000240 | 0.019599 | 0.019599 | 600.000000 | 0.000000 | 0.000015 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz |
| step_sigma_theta | Hybrid Residual-B | 0.844036 | 0.000289 | 0.019680 | 0.019680 | 600.000000 | 0.000000 | 0.000014 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/runtime_b_residual_v1/tiny_cnn_20260401_083648_2fc740424c0d.npz |
| periodic_drift | Window Variance | 0.856992 | 0.001016 | 0.019929 | 0.019929 | 600.000000 | 0.000000 | 0.000019 | histogram_input |  |
| periodic_drift | EKF | 0.855229 | 0.000271 | 0.019585 | 0.019585 | 600.000000 | 0.000000 | 0.000013 | histogram_input |  |
| periodic_drift | Constant Residual-Mu | 0.856003 | 0.001030 | 0.019870 | 0.019870 | 600.000000 | 0.000000 | 0.000017 | histogram_input |  |
| periodic_drift | CNN-FPGA | 0.954384 | 0.000394 | 0.019629 | 0.019629 | 600.000000 | 0.000000 | 0.000016 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz |
| periodic_drift | Hybrid Residual-B | 0.848230 | 0.000025 | 0.019704 | 0.019704 | 600.000000 | 0.000000 | 0.000016 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/runtime_b_residual_v1/tiny_cnn_20260401_083648_2fc740424c0d.npz |

## Delta Summary

| Scenario | Mode | dLER vs Static | dLER vs CNN | dOverflow vs Static | dOverflow vs CNN |
| --- | --- | ---: | ---: | ---: | ---: |
| static_bias_theta | Window Variance |  | -0.100479 |  | 0.000150 |
| static_bias_theta | EKF |  | -0.101456 |  | -0.000222 |
| static_bias_theta | Constant Residual-Mu |  | -0.103009 |  | 0.000009 |
| static_bias_theta | CNN-FPGA |  | 0.000000 |  | 0.000000 |
| static_bias_theta | Hybrid Residual-B |  | -0.105819 |  | -0.000015 |
| linear_ramp | Window Variance |  | -0.095453 |  | 0.000130 |
| linear_ramp | EKF |  | -0.096634 |  | 0.000007 |
| linear_ramp | Constant Residual-Mu |  | -0.097285 |  | 0.000200 |
| linear_ramp | CNN-FPGA |  | 0.000000 |  | 0.000000 |
| linear_ramp | Hybrid Residual-B |  | -0.098054 |  | 0.000242 |
| step_sigma_theta | Window Variance |  | -0.095874 |  | 0.000112 |
| step_sigma_theta | EKF |  | -0.096901 |  | -0.000173 |
| step_sigma_theta | Constant Residual-Mu |  | -0.096389 |  | 0.000148 |
| step_sigma_theta | CNN-FPGA |  | 0.000000 |  | 0.000000 |
| step_sigma_theta | Hybrid Residual-B |  | -0.104037 |  | 0.000081 |
| periodic_drift | Window Variance |  | -0.097392 |  | 0.000301 |
| periodic_drift | EKF |  | -0.099155 |  | -0.000044 |
| periodic_drift | Constant Residual-Mu |  | -0.098382 |  | 0.000241 |
| periodic_drift | CNN-FPGA |  | 0.000000 |  | 0.000000 |
| periodic_drift | Hybrid Residual-B |  | -0.106155 |  | 0.000075 |

## Scenario Winners

| Scenario | Best Mode | Best LER Mean | Runner-Up Gap |
| --- | --- | ---: | ---: |
| static_bias_theta | Hybrid Residual-B | 0.859707 | 0.002810 |
| linear_ramp | Hybrid Residual-B | 0.851224 | 0.000769 |
| step_sigma_theta | Hybrid Residual-B | 0.844036 | 0.007136 |
| periodic_drift | Hybrid Residual-B | 0.848230 | 0.007000 |
