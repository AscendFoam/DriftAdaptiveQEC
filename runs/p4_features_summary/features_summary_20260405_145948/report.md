# P4 Features Ablation Summary

## Protocol

- run_dir: `/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/runs/p4_benchmark/p4_hybrid_vs_ukf_ablation_features_v1_20260404_211945_a7d2984ad50e_23637`
- best_mode: `Hybrid No TeacherParams`

## Aggregated Table

| Mode | Description | Avg LER | dLER vs UKF | dLER vs Hybrid Full | Avg Overflow | Avg Hist Sat | Avg Commits | Overflow Sources |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| UKF | 最强经典 baseline | 0.818081 | +0.000000 | +0.019726 | 0.002546 | 0.002546 | 899.81 | histogram_input |
| Hybrid Full | 完整 teacher-guided residual-b 主模型 | 0.798355 | -0.019726 | +0.000000 | 0.002539 | 0.002539 | 899.88 | histogram_input |
| Hybrid No HistDelta | 去掉 histogram delta 通道 | 0.826422 | +0.008341 | +0.028067 | 0.002593 | 0.002593 | 899.75 | histogram_input |
| Hybrid No TeacherPred | 去掉 teacher prediction 通道 | 0.807556 | -0.010525 | +0.009201 | 0.002560 | 0.002560 | 899.94 | histogram_input |
| Hybrid No TeacherParams | 去掉 teacher params 通道 | 0.749436 | -0.068645 | -0.048919 | 0.002447 | 0.002447 | 899.88 | histogram_input |
| Hybrid No TeacherDelta | 去掉 teacher delta 通道 | 0.800473 | -0.017608 | +0.002118 | 0.002253 | 0.002253 | 899.56 | aggressive_param |

## Scenario Rows

| Scenario | Mode | LER Mean | Overflow | Hist Sat | Aggressive Source |
| --- | --- | ---: | ---: | ---: | --- |
| static_bias_theta | UKF | 0.826590 | 0.002677 | 0.002677 | histogram_input |
| static_bias_theta | Hybrid Full | 0.810757 | 0.002638 | 0.002638 | histogram_input |
| static_bias_theta | Hybrid No HistDelta | 0.837235 | 0.002692 | 0.002692 | histogram_input |
| static_bias_theta | Hybrid No TeacherPred | 0.814637 | 0.002660 | 0.002660 | histogram_input |
| static_bias_theta | Hybrid No TeacherParams | 0.734263 | 0.002505 | 0.002505 | histogram_input |
| static_bias_theta | Hybrid No TeacherDelta | 0.801496 | 0.002297 | 0.002297 | aggressive_param |
| linear_ramp | UKF | 0.809234 | 0.002439 | 0.002439 | histogram_input |
| linear_ramp | Hybrid Full | 0.788336 | 0.002449 | 0.002449 | histogram_input |
| linear_ramp | Hybrid No HistDelta | 0.817267 | 0.002504 | 0.002504 | histogram_input |
| linear_ramp | Hybrid No TeacherPred | 0.801789 | 0.002461 | 0.002461 | histogram_input |
| linear_ramp | Hybrid No TeacherParams | 0.760859 | 0.002388 | 0.002388 | histogram_input |
| linear_ramp | Hybrid No TeacherDelta | 0.800800 | 0.002212 | 0.002212 | aggressive_param |
| step_sigma_theta | UKF | 0.815113 | 0.002482 | 0.002482 | histogram_input |
| step_sigma_theta | Hybrid Full | 0.787033 | 0.002483 | 0.002483 | histogram_input |
| step_sigma_theta | Hybrid No HistDelta | 0.819309 | 0.002547 | 0.002547 | histogram_input |
| step_sigma_theta | Hybrid No TeacherPred | 0.802357 | 0.002508 | 0.002508 | histogram_input |
| step_sigma_theta | Hybrid No TeacherParams | 0.756925 | 0.002426 | 0.002426 | histogram_input |
| step_sigma_theta | Hybrid No TeacherDelta | 0.801696 | 0.002251 | 0.002251 | aggressive_param |
| periodic_drift | UKF | 0.821388 | 0.002585 | 0.002585 | histogram_input |
| periodic_drift | Hybrid Full | 0.807292 | 0.002586 | 0.002586 | histogram_input |
| periodic_drift | Hybrid No HistDelta | 0.831878 | 0.002630 | 0.002630 | histogram_input |
| periodic_drift | Hybrid No TeacherPred | 0.811439 | 0.002611 | 0.002611 | histogram_input |
| periodic_drift | Hybrid No TeacherParams | 0.745696 | 0.002469 | 0.002469 | histogram_input |
| periodic_drift | Hybrid No TeacherDelta | 0.797900 | 0.002252 | 0.002252 | aggressive_param |

## Reading Guide

- `dLER vs UKF < 0` 说明该模式优于 `UKF`。
- `dLER vs Hybrid Full > 0` 说明去掉该类输入后性能变差，意味着该输入通道是有价值的。
- 若 overflow 接近但 LER 明显变差，更支持“性能差异来自估计质量而非饱和差异”。
