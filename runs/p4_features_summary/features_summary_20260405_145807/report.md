# P4 Features Ablation Summary

## Protocol

- run_dir: `/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/runs/p4_benchmark/p4_hybrid_vs_ukf_ablation_features_v1_20260404_211945_a7d2984ad50e_23637`
- best_mode: `Hybrid No TeacherParams`

## Scenario Table

| Mode | Description | Scenario | LER Mean | dLER vs UKF | dLER vs Hybrid Full | Overflow | Hist Sat | Commits | Dominant Overflow |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| UKF | 最强经典 baseline | periodic_drift | 0.821388 | +0.000000 | +0.014096 | 0.002585 | 0.002585 | 899.2 | histogram_input |
| Hybrid Full | 完整 teacher-guided residual-b 主模型 | periodic_drift | 0.807292 | -0.014096 | +0.000000 | 0.002586 | 0.002586 | 900.0 | histogram_input |
| Hybrid No HistDelta | 去掉 histogram delta 通道 | periodic_drift | 0.831878 | +0.010490 | +0.024586 | 0.002630 | 0.002630 | 899.8 | histogram_input |
| Hybrid No TeacherPred | 去掉 teacher prediction 通道 | periodic_drift | 0.811439 | -0.009949 | +0.004147 | 0.002611 | 0.002611 | 900.0 | histogram_input |
| Hybrid No TeacherParams | 去掉 teacher params 通道 | periodic_drift | 0.745696 | -0.075692 | -0.061597 | 0.002469 | 0.002469 | 900.0 | histogram_input |
| Hybrid No TeacherDelta | 去掉 teacher delta 通道 | periodic_drift | 0.797900 | -0.023488 | -0.009392 | 0.002252 | 0.002252 | 899.5 | aggressive_param |

## Reading Guide

- `dLER vs UKF < 0` 说明该模式优于 `UKF`。
- `dLER vs Hybrid Full > 0` 说明去掉该类输入后性能变差，意味着该输入通道是有价值的。
- 若 overflow 接近但 LER 明显变差，更支持“性能差异来自估计质量而非饱和差异”。
