# T3.1.5 single-mode top-K lattice-coset truncated MAP

## 结论

本任务实现的是 single-mode GKP 周期高斯 MAP 的可控近似，不是 surface-code
matching，也不是 K minimum-weight matchings。对每个二维逻辑陪集分别保留联合高斯似然最大的
`K` 个 lattice-pair aliases，再在陪集内求和。生产验证覆盖 6 类各向同性、各向异性、正/负相关、
宽噪声和大偏置场景，4 个独立 evaluation seeds、每 seed 12,000 个样本，共 288,000 个样本。

所有 6 个场景的 `K=1` 软 LLR 都不是 full periodic MAP；按预注册的 hard-decision、LER 和
LLR 联合门，场景相关收敛点为 `K=2--4`。`K=128` 覆盖每个场景的全部候选陪集项，与同一
`tail_sigma=8` 候选矩形的 full reference 在浮点误差内一致。该结果只说明当前冻结 Gaussian
scenario family 中的近似误差，不能外推为设备通用最优 K。

## 数学与实现合同

对 centered syndrome `s=(s_q,s_p)`、冻结均值 `mu`、协方差 `Sigma` 和 lattice spacing
`L=sqrt(2*pi)`，候选 lattice pair `n=(n_q,n_p)` 的 log weight 为

```text
ell_n(s) = -1/2 (s + L n - mu)^T Sigma^{-1} (s + L n - mu)
           - log(2*pi) - 1/2 log(det Sigma).
```

逻辑陪集按 `c(n)=(n_q mod 2,n_p mod 2)` 划分。对每个陪集 `c` 独立降序排列后，

```text
log Lambda_c^(K)(s) = log prior_c + logsumexp(top-K {ell_n(s): c(n)=c}).
```

`K` 增大到该陪集候选数时得到同一有限候选矩形内的 full periodic Gaussian MAP。二维相关项
始终以 joint lattice pair 排名，没有把 q/p 独立截断。full reference 直接复用
`physics.ideal_gkp_decoder.map_decode_2d`，候选半径规则完全相同；因此比较测的是实现截断误差，
不是两个不同 decoder/model 的差异。

在线输入仅为当前 `q/p` centered modular syndrome；均值、协方差、四陪集 prior、`K` 和
`tail_sigma` 均为冻结离线参数。API 对半开区间、正定协方差、正 prior、整数 K、int64 index 和
候选工作量 fail closed。

## K 扫描结果

| 场景 | aliases / 最大陪集数 | K=1 hard disagreement | K=1 LLR p99 误差 | 收敛 K | 收敛点 hard disagreement | 收敛点 LLR p99 误差 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| narrow isotropic | 81 / 25 | 0 | 0.616795 | 2 | 0 | `1.41e-9` |
| anisotropic | 117 / 35 | 0 | 0.641542 | 4 | 0 | `1.67e-5` |
| positive correlation | 143 / 42 | 0 | 0.644816 | 2 | 0 | `1.89e-6` |
| negative correlation | 143 / 42 | 0 | 0.647223 | 2 | 0 | `1.82e-6` |
| broad correlated | 255 / 72 | 0.002479 | 0.533093 | 4 | `8.33e-5` | `3.88e-4` |
| large bias | 169 / 49 | 0.000250 | 0.504444 | 4 | 0 | `7.70e-6` |

收敛标准为 hard disagreement `<=1e-4`、axis LLR p99 error `<=1e-3`、绝对 paired LER
difference `<=1e-4`。LER difference 保留带符号值：例如 broad-correlated 的 `K=1` 为
`+2.5e-4`，`K=4` 为 `-6.25e-5`。有限样本上的 LER 不要求随 K 单调，不能用偶然负差声称
truncated decoder 优于 full MAP。

## 确定性成本口径

成本画像对应一个保守的 probability-domain serial-streaming mapping：每个 alias 计算二维高斯
quadratic form，执行一次 exponential-LUT query，按等价 log-weight 顺序维护每陪集长度 K 的
有序列表，再累加保留权重。报告 Gaussian 乘加、exp 查询、top-K comparison 上界、accumulation、
alias/state bits 和串行/流水 cycle proxy。它不是 NumPy `logaddexp` 的 RTL 计时。

以最大候选的 broad-correlated 场景为例，`K=1/4/128` 的 retained-state bits 分别为
`136/544/9792`，comparison 上界为 `255/1020/18360`；`K=128` 在最大陪集 72 项处饱和。
所有 `target_lut/ff/bram/dsp/fmax` 字段保持 `null` 且 `target_measured=false`。真实资源与频率只可
在 T5.5 的具名器件、RTL、工具版本和 synthesis/P&R 证据后填写。

## 证据与复现

- 实现：`cnn_fpga/benchmark/topk_lattice_coset_map.py`；
- 独立测试：`tests/test_topk_lattice_coset_map.py`；
- 机器报告：`docs/t3_1_5_topk_map_validation.json`；
- Source Data：`docs/t3_1_5_topk_map_source_data.csv`，192 行 scenario/seed/K 数据；
- 复现命令：`python -m cnn_fpga.benchmark.topk_lattice_coset_map`。

测试使用独立嵌套枚举核对 top-K aliases，验证 K 饱和时与 full MAP 一致、likelihood 对 K 单调、
一次排序扫描与逐 K 调用相同、偶数 lattice-mean 平移不变、异常输入 fail closed、CSV 可重算、
源码 hash 绑定和硬件字段为空。最终 8/8 machine gates、133/133 focused+adjacent tests 通过；
显式全量 `tests/` 为 895 passed、4 skipped、4 failed，失败仍仅来自 R-N012 缺少两份历史
FR8/P4 文档。

## Claim 边界

允许写：当前冻结的 single-mode correlated-Gaussian periodic-MAP 场景中，top-K likelihood
accumulation 在 `K=2--4` 达到预设近似门，并给出可复算的确定性 operation/storage proxy。

禁止写：实现了 surface-code K-MWM、`K=4` 对所有装置/噪声最优、有限 K 的 LER 单调、已获得
FPGA LUT/BRAM/DSP/Fmax，或本任务证明了真实 finite-energy/sBs logical recovery 性能。
