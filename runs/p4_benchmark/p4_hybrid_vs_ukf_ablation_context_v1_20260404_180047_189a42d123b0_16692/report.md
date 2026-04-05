# P4 Multi-Scenario Benchmark

## Protocol

- protocol_id: `p4_hybrid_vs_ukf_ablation_context_v1`
- report_template_version: `p4_hil_report_v1`
- repeats: `1`
- real_board_policy: `conditional_extension`

## Scenario Comparison

| Scenario | Mode | LER Mean | LER Std | Overflow Mean | Hist Sat Mean | Commit Mean | Slow Viol Mean | Fast Viol Mean | Dominant Source | Artifact |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| static_bias_theta | UKF | 0.826406 | 0.000000 | 0.002583 | 0.002583 | 900.000000 | 0.000000 | 0.000014 | histogram_input |  |
| static_bias_theta | Hybrid Ctx1 | 0.824150 | 0.000000 | 0.002709 | 0.002709 | 900.000000 | 0.000000 | 0.000015 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/runtime_b_residual_ctx1_v1/tiny_cnn_20260404_173830_e0c385cfbf20.npz |
| static_bias_theta | Hybrid Ctx3 | 0.838387 | 0.000000 | 0.002689 | 0.002689 | 900.000000 | 0.000000 | 0.000017 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/runtime_b_residual_ctx3_v1/tiny_cnn_20260404_180046_57bcac7985c6.npz |
| static_bias_theta | Hybrid Ctx5 | 0.813885 | 0.000000 | 0.002626 | 0.002626 | 900.000000 | 0.000000 | 0.000015 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/runtime_b_residual_v1/tiny_cnn_20260401_083648_2fc740424c0d.npz |

## Delta Summary

| Scenario | Mode | dLER vs Static | dLER vs CNN | dOverflow vs Static | dOverflow vs CNN |
| --- | --- | ---: | ---: | ---: | ---: |
| static_bias_theta | UKF |  |  |  |  |
| static_bias_theta | Hybrid Ctx1 |  |  |  |  |
| static_bias_theta | Hybrid Ctx3 |  |  |  |  |
| static_bias_theta | Hybrid Ctx5 |  |  |  |  |

## Scenario Winners

| Scenario | Best Mode | Best LER Mean | Runner-Up Gap |
| --- | --- | ---: | ---: |
| static_bias_theta | Hybrid Ctx5 | 0.813885 | 0.010265 |
