# P4 Multi-Scenario Benchmark

## Protocol

- protocol_id: `p4_hil_hybrid_mu_v1`
- report_template_version: `p4_hil_report_v1`
- repeats: `2`
- real_board_policy: `conditional_extension`

## Scenario Comparison

| Scenario | Mode | LER Mean | LER Std | Overflow Mean | Hist Sat Mean | Commit Mean | Slow Viol Mean | Fast Viol Mean | Dominant Source | Artifact |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| periodic_drift | Window Variance | 0.857409 | 0.000631 | 0.019878 | 0.019878 | 600.000000 | 0.000000 | 0.000014 | histogram_input |  |
| periodic_drift | EKF | 0.855032 | 0.000491 | 0.019554 | 0.019554 | 600.000000 | 0.000000 | 0.000016 | histogram_input |  |
| periodic_drift | CNN-FPGA | 0.954013 | 0.000091 | 0.019609 | 0.019609 | 600.000000 | 0.000000 | 0.000013 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz |
| periodic_drift | Hybrid Residual-Mu | 0.854607 | 0.000966 | 0.019818 | 0.019818 | 600.000000 | 0.000000 | 0.000017 | histogram_input | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/runtime_mu_residual_v1/tiny_cnn_20260401_000659_bd7d177e4ffa.npz |

## Delta Summary

| Scenario | Mode | dLER vs Static | dLER vs CNN | dOverflow vs Static | dOverflow vs CNN |
| --- | --- | ---: | ---: | ---: | ---: |
| periodic_drift | Window Variance |  | -0.096605 |  | 0.000269 |
| periodic_drift | EKF |  | -0.098981 |  | -0.000055 |
| periodic_drift | CNN-FPGA |  | 0.000000 |  | 0.000000 |
| periodic_drift | Hybrid Residual-Mu |  | -0.099406 |  | 0.000209 |

## Scenario Winners

| Scenario | Best Mode | Best LER Mean | Runner-Up Gap |
| --- | --- | ---: | ---: |
| periodic_drift | Hybrid Residual-Mu | 0.854607 | 0.000424 |
