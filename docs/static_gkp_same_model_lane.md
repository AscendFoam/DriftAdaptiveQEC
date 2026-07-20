# T6.8.1 相对静态 GKP 解码的同模型比较 lane

## 结论

机器判定为：

`PASS_STATIC_GKP_SAME_MODEL_LANE_ROUTE_A_SUPERIORITY_FALSIFIED`

lane 本身完整通过，但“Route-A 优于 static GKP decoding”的主张被证否。24 个 formal seed 上，paired `static full MAP LER - Route-A LER` 为

```text
-2.454828e-5, 95% CI [-3.959514e-5, -9.112888e-6]
```

CI 全为负，说明在冻结的 T6.7.1 smooth simulator/protocol 下，Route-A 的平均 LER 显著高于 frozen full static MAP。该结论不能被 Route-A 相对 locked EWMA 的正结果抵消。

## 一手文献与本项目 adapter 映射

| 角色 | 一手来源 | 本项目映射 | 禁止外推 |
| --- | --- | --- | --- |
| standard binning origin | Gottesman, Kitaev & Preskill, *Encoding a qubit in an oscillator*, [PRA 64, 012310](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.64.012310) | centered-cell / parity hard action | 不把原论文不同 noise/state 的数字与本项目 raw LER 相减 |
| analog likelihood precedent | Fukui et al., *Analog quantum error correction with encoding a qubit into an oscillator*, [PRL 119, 180507](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.119.180507) | 保留连续 residual，构造周期 likelihood | 不导入外层码 threshold/squeezing 数值 |
| finite-energy decoding boundary | Jafarzadeh et al., *Logical channels in approximate GKP error correction*, [arXiv:2504.13383](https://arxiv.org/abs/2504.13383) | 说明 finite-energy 下 optimized decoding 与 SB 可不同 | 该文 logical-channel 数字不是本项目 syndrome simulator |
| prior calibration cross-code precedent | Sivak, Newman & Klimov, *Optimization of Decoder Priors for Accurate QEC*, [PRL 133, 150603](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.133.150603) | prior calibration 是已有方向 | 其 repetition/surface-code `16%/3.3%` 不能当 GKP baseline |

本任务用 `nature-academic-search` 的 multi-source workflow 核对标识；academic-search MCP 未挂载后按技能规则转用本地全文、APS/arXiv 一手页面。搜索结果只进入文献身份和口径边界，不进入数值排名。

## same-trace 比较

所有方法消费相同的 576 条 smooth formal trajectory、24 seeds、28,311,552 scored decisions；hidden-state oracle 独立分栏。

| 方法 | average LER | p95-window | worst-window | `p_X` | `p_Y` | `p_Z` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| standard binning | `9.78611e-4` | `0.005859` | `0.033203` | `6.57859e-4` | `2.47249e-6` | `3.18280e-4` |
| static full joint MAP | `9.68191e-4` | `0.005859` | `0.031250` | `6.40269e-4` | `2.68442e-6` | `3.25238e-4` |
| top-K=4 static MAP | `9.68191e-4` | `0.005859` | `0.031250` | `6.40269e-4` | `2.68442e-6` | `3.25238e-4` |
| proposed Route-A | `9.92740e-4` | `0.003906` | `0.115234` | `7.41005e-4` | `1.02432e-6` | `2.50710e-4` |
| hidden-state oracle | `1.62337e-4` | `0.001953` | `0.013672` | `8.70316e-5` | `6.00462e-7` | `7.47045e-5` |

Route-A 的 p95 较低，但 average 和 worst 均更差；这种分位权衡不支持总体优势。static-to-oracle gap closure 仍为负，只有 T6.7.1 的 locked-EWMA aggregate 与 periodic Holm discovery 可保留为正结果。

## top-K / full MAP 完整域证明

T6.7 deployable residual 在生成时已经量化到 10-bit ADC bin centre，因此输入域恰为 `1024 × 1024 = 1,048,576` 个 syndrome pair。runner 对完整域逐点计算：

- K=4 lattice-coset truncated MAP hard action；
- 相同 mean/covariance/prior/tail radius 下的 full periodic Gaussian MAP；
- hard disagreement：`0/1,048,576`；
- action SHA-256 完全相同；
- q/p LLR 最大绝对差分别 `7.11e-15 / 8.88e-15`，仅数值归约量级。

因此 K=4 可以在该冻结 static model 的整个 deployable domain 上安全继承 full static 的 Pauli/error trace。这不是抽样，也不是把少量 smoke 当 formal；同时不能外推成 K=4 对任意 covariance/noise/soft posterior 都严格相同。

## 确定性成本代理

| static 实现 | aliases | retained bits | serial upper proxy | pipeline lower proxy | top-K comparisons upper |
| --- | ---: | ---: | ---: | ---: | ---: |
| full candidate sum | 81 | 3200 | 2209 | 137 | 2025 |
| K=4 | 81 | 512 | 424 | 95 | 324 |

K=4 在当前模型上将 retained state 降至 `16%`，serial proxy 降至约 `19.2%`，同时保持完整部署域 hard-action 等价。这是 operation/storage proxy，不是 synthesis、P&R 或板测；实际 LUT/FF/BRAM/DSP/Fmax/功耗仍为空。

## 证据、验证与 claim 边界

- runner：`cnn_fpga/benchmark/static_gkp_same_model_lane.py`
- report：`docs/t6_8_1_static_gkp_same_model_lane.json`
- Source Data：`docs/t6_8_1_static_gkp_same_model_lane_source_data.csv`
- tests：`tests/test_static_gkp_same_model_lane.py`
- machine gates：`11/11`
- semantic mutations：`8/8`
- focused tests：`8 passed`

允许：same-model 下 Route-A 平均 LER 劣于 static；当前 frozen static model 下 K=4/full hard-action 完整域等价，并有较低确定性 operation/storage proxy。

禁止：Route-A 优于 static GKP、global GKP SOTA、真实 finite-energy break-even、K=4 universally exact、已经获得 FPGA 资源或时延。

