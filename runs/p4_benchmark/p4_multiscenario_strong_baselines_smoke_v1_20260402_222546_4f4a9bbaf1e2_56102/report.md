# P4 Multi-Scenario Benchmark

## Protocol

- protocol_id: `p4_strong_baselines_smoke_v1`
- report_template_version: `p4_hil_report_v1`
- repeats: `1`
- real_board_policy: `conditional_extension`

## Scenario Comparison

| Scenario | Mode | LER Mean | LER Std | Overflow Mean | Hist Sat Mean | Commit Mean | Slow Viol Mean | Fast Viol Mean | Dominant Source | Artifact |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| static_bias_theta | EKF | 0.831983 | 0.000000 | 0.002692 | 0.002692 | 119.000000 | 0.000000 | 0.000013 | histogram_input |  |
| static_bias_theta | RLS Residual-B | 0.832421 | 0.000000 | 0.002690 | 0.002690 | 120.000000 | 0.000000 | 0.000015 | histogram_input |  |
| static_bias_theta | Hybrid Residual-B | 0.797921 | 0.000000 | 0.002519 | 0.002519 | 120.000000 | 0.000000 | 0.000017 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/runtime_b_residual_v1/tiny_cnn_20260401_083648_2fc740424c0d.npz |

## Delta Summary

| Scenario | Mode | dLER vs Static | dLER vs CNN | dOverflow vs Static | dOverflow vs CNN |
| --- | --- | ---: | ---: | ---: | ---: |
| static_bias_theta | EKF |  |  |  |  |
| static_bias_theta | RLS Residual-B |  |  |  |  |
| static_bias_theta | Hybrid Residual-B |  |  |  |  |

## Scenario Winners

| Scenario | Best Mode | Best LER Mean | Runner-Up Gap |
| --- | --- | ---: | ---: |
| static_bias_theta | Hybrid Residual-B | 0.797921 | 0.034062 |
