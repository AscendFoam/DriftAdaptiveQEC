# T6.18.3 multimode posterior-weighted CPD 漂移扩展

- verdict：`GO_POSTERIOR_WEIGHTED_CPD_DRIFT_GAIN`
- formal cycles：9,600,000
- comparator decodes：38,400,000
- gates / mutations：21/21 / 21/21

## Aggregate

| method | p_L | worst 512-window | CVaR95 | seconds/decode |
| --- | ---: | ---: | ---: | ---: |
| `static_euclidean` | 0.23692937 | 0.320312 | 0.277271 | 0.000199721 |
| `weighted_static` | 0.26167854 | 0.460938 | 0.404130 | 0.000193983 |
| `observed_only_posterior_predictive_weighted` | 0.17226094 | 0.291016 | 0.240627 | 0.000193104 |
| `oracle_metric_upper_bound` | 0.17192948 | 0.292969 | 0.240211 | 0.000192343 |

## Paired p_L contrasts

| contrast | mean [95% CI] | Holm p |
| --- | ---: | ---: |
| `adaptive_vs_static_euclidean` | 0.06466844 [0.06441281, 0.06492636] | 2.99997e-05 |
| `adaptive_vs_weighted_static` | 0.08941760 [0.08718633, 0.09165222] | 2.99997e-05 |
| `weighted_static_vs_static_euclidean` | -0.02474917 [-0.02688389, -0.02261969] | 2.99997e-05 |

## 边界

该实验是 project-native、observed-only、strict-causal 的 d=3 heteroscedastic drift 扩展。oracle 使用当前真实 metric，只作上界参考；结果不回写 Phase 6B，不代表 official Lin et al. drift experiment、stationary threshold、一般 multimode SOTA、FPGA 或物理装置优势。

## Artifacts

- `docs/t6_18_3_multimode_posterior_weighted_cpd.json`
- `docs/t6_18_3_multimode_posterior_weighted_cpd_source_data.csv`
- `scripts/run_multimode_posterior_weighted_cpd.jl`
- `cnn_fpga/benchmark/multimode_posterior_weighted_cpd.py`
