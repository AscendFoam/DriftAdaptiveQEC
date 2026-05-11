# T27 人话版说明

这次审计的核心结论很直接：

- `teacher_scalar_diagnostics.csv` 不是“写文件时漏了”，而是更早的 explain 路径就没产出这类诊断数据。
- 当前 T24 的 `hybrid_residual_b` 用的是 broadcast teacher 特征，不是 scalar side-branch。
- 但 `tiny_cnn.py` 里的 explain 逻辑，只有在 `scalar_feature_dim > 0` 时才会生成 `teacher_contribution`、`per_scalar`、`teacher_gate_*` 这些诊断。
- 所以现在看到的 `0.0` 和空表，主要是在“没生成数据”的前提下，被后续聚合和 CSV 默认值写得像“真的是零”。

`correction_saturation_rate_mean` 是另一条独立路径：

- 它来自 fast-loop saturation counter，不依赖 teacher diagnostics。
- 现有 T24 证据更支持“当前参数区间下确实没打到 correction saturation”，而不是“指标链路坏了”。

因此：

- `R10` 还开着，但原因已经缩窄清楚了。
- `R20` 也还开着，不过现在更像“边界没打到”，不是 teacher diagnostics 那条空路径的连带问题。

下一步如果要修，不该直接改 benchmark 结论；应该先做一个很小的后续任务，把 “broadcast teacher” 和 “scalar teacher diagnostics” 的语义关系补清楚，并让 CSV 能区分“未生成”和“真实为零”。
