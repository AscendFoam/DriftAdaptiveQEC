# P4 Multi-Scenario Benchmark

## Protocol

- protocol_id: `p4_strong_baselines_smoke_v1`
- report_template_version: `p4_hil_report_v1`
- repeats: `1`
- real_board_policy: `conditional_extension`

## Scenario Comparison

| Scenario | Mode | LER Mean | LER Std | Overflow Mean | Hist Sat Mean | Commit Mean | Slow Viol Mean | Fast Viol Mean | Dominant Source | Artifact |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| static_bias_theta | EKF | 0.830612 | 0.000000 | 0.002660 | 0.002660 | 119.000000 | 0.000000 | 0.000013 | histogram_input |  |
| static_bias_theta | UKF | 0.832771 | 0.000000 | 0.002610 | 0.002610 | 120.000000 | 0.000000 | 0.000015 | histogram_input |  |
| static_bias_theta | Constant Residual-Mu | 0.823610 | 0.000000 | 0.002725 | 0.002725 | 120.000000 | 0.000000 | 0.000017 | histogram_input |  |
| static_bias_theta | RLS Residual-B | 0.822369 | 0.000000 | 0.002602 | 0.002602 | 120.000000 | 0.000000 | 0.000015 | histogram_input |  |
| static_bias_theta | Hybrid Residual-B | 0.799548 | 0.000000 | 0.002537 | 0.002537 | 120.000000 | 0.000000 | 0.000021 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/runtime_b_residual_v1/tiny_cnn_20260401_083648_2fc740424c0d.npz |
| linear_ramp | EKF | 0.838054 | 0.000000 | 0.002767 | 0.002767 | 120.000000 | 0.000000 | 0.000013 | histogram_input |  |
| linear_ramp | UKF | 0.826362 | 0.000000 | 0.002812 | 0.002812 | 120.000000 | 0.000000 | 0.000015 | histogram_input |  |
| linear_ramp | Constant Residual-Mu | 0.827292 | 0.000000 | 0.002802 | 0.002802 | 120.000000 | 0.000000 | 0.000010 | histogram_input |  |
| linear_ramp | RLS Residual-B | 0.833321 | 0.000000 | 0.002817 | 0.002817 | 120.000000 | 0.000000 | 0.000013 | histogram_input |  |
| linear_ramp | Hybrid Residual-B | 0.814408 | 0.000000 | 0.002667 | 0.002667 | 120.000000 | 0.000000 | 0.000023 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/runtime_b_residual_v1/tiny_cnn_20260401_083648_2fc740424c0d.npz |
| step_sigma_theta | EKF | 0.838360 | 0.000000 | 0.002704 | 0.002704 | 120.000000 | 0.000000 | 0.000017 | histogram_input |  |
| step_sigma_theta | UKF | 0.839929 | 0.000000 | 0.002783 | 0.002783 | 120.000000 | 0.000000 | 0.000015 | histogram_input |  |
| step_sigma_theta | Constant Residual-Mu | 0.831079 | 0.000000 | 0.002929 | 0.002929 | 120.000000 | 0.000000 | 0.000004 | histogram_input |  |
| step_sigma_theta | RLS Residual-B | 0.840142 | 0.000000 | 0.002742 | 0.002742 | 120.000000 | 0.000000 | 0.000019 | histogram_input |  |
| step_sigma_theta | Hybrid Residual-B | 0.817094 | 0.000000 | 0.002625 | 0.002625 | 120.000000 | 0.000000 | 0.000008 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/runtime_b_residual_v1/tiny_cnn_20260401_083648_2fc740424c0d.npz |
| periodic_drift | EKF | 0.818231 | 0.000000 | 0.002523 | 0.002523 | 120.000000 | 0.000000 | 0.000017 | histogram_input |  |
| periodic_drift | UKF | 0.820440 | 0.000000 | 0.002502 | 0.002502 | 120.000000 | 0.000000 | 0.000019 | histogram_input |  |
| periodic_drift | Constant Residual-Mu | 0.806019 | 0.000000 | 0.002527 | 0.002527 | 120.000000 | 0.000000 | 0.000013 | histogram_input |  |
| periodic_drift | RLS Residual-B | 0.817287 | 0.000000 | 0.002402 | 0.002402 | 120.000000 | 0.000000 | 0.000015 | histogram_input |  |
| periodic_drift | Hybrid Residual-B | 0.783665 | 0.000000 | 0.002475 | 0.002475 | 120.000000 | 0.000000 | 0.000027 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/runtime_b_residual_v1/tiny_cnn_20260401_083648_2fc740424c0d.npz |

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
| static_bias_theta | Hybrid Residual-B | 0.799548 | 0.022821 |
| linear_ramp | Hybrid Residual-B | 0.814408 | 0.011954 |
| step_sigma_theta | Hybrid Residual-B | 0.817094 | 0.013985 |
| periodic_drift | Hybrid Residual-B | 0.783665 | 0.022354 |
