# T2.3.8 Heisenberg-inspired noise-transfer 中保真代理

**日期：** 2026-07-14  
**状态：** Implemented and validated  
**实现：** `physics/noise_transfer_surrogate.py`  
**机器证据：** `docs/t2_3_8_noise_transfer_validation.json`

## 1. 模型定位

该层受 Ralph 等 2024 年 noise-transfer 分析启发，把 GKP correction 的状态拆成三类：

1. 离散 lattice signal：整数格点 `n=(n_q,n_p)`；
2. 连续 fluctuation：offset `mu` 与 `2x2` covariance `V`；
3. 离散 logical jump：nearest-domain alias 的 q/p 奇偶。

一手来源为 *Noise Transfer in Fault-Tolerant Quantum Error Correction*（Entropy 26,
874；arXiv:2411.05262）。实现采用两个独立 decoder-standardized axis：vacuum variance
为 `1`，correction-cell spacing 为 `sqrt(2*pi)`。这是一种 classical normalization，
不是一对 joint canonical operators；Fock/Fourier 对照显式转到 `[x,p]=i`。它不是论文
teleportation circuit 的逐元件复现，也不是 T2.3.2 SBS Kraus、pulse/transmon 或
device-calibrated model。

## 2. 传播方程

令 `s=nL+mu`、loss transmissivity 为 `eta`、measurement efficiency 为 `xi`、
feedforward gain 为矩阵 `G`，则：

```text
b_loss = sqrt(eta) (n L + mu) - n L
V_loss = eta V_in + (1-eta) V_vac I
V_meas = V_resource + (1-xi)/xi V_vac I
V_dec  = V_loss + V_meas
mu_out = (I-G) b_loss
V_out  = (I-G) V_loss (I-G)^T + G V_meas G^T
```

对每一轴，决策变量在 cell `[(k-1/2)L,(k+1/2)L)` 上的 Gaussian integral 给出
`P_k`；奇数 `k` 是该轴的 logical jump。若 `V_dec` 为 diagonal，q/p 独立，因此可给出
完整 I/X/Z/Y product distribution；若存在相关项，只报告边缘概率和安全 Fréchet bounds，
不伪造 joint Pauli rate。

## 3. clipping 与有效性门

代理同时计算：

- central-cell probability；
- 对每个已判定 domain 分别去条件均值后的 weighted within-domain variance；
- `clipping_ratio = within-domain variance / unfurled variance`。

只有两轴 central probability 不低于 `0.95` 且 clipping ratio 不低于 `0.90` 时标记
`localized`。低于门槛标记 `clipping_dominated`，仍输出统计但禁止作为线性
noise-transfer 有效证据。门槛是本项目保守 validity gate，不宣称来自装置校准。

## 4. squeezing dB 与 state/Fock 对齐

canonical 与 decoder-standardized 单峰 probability variance 分别为：

```text
V_peak_canonical(dB) = 0.5 * 10^(-dB/10)
V_peak_decoder(dB)   =       10^(-dB/10)
Delta(dB)  = sqrt(atanh(10^(-dB/10)))
```

对 logical `0/1/+/-`，验证同时计算 canonical 解析 proxy、decoder state density 经方差
换算后的 q-domain conditional variance，以及 registered `sqrt(2)` bridge 下 cutoff-48
Hermite/Fock 重构。T-RISK-20260714-01 后，坐标、Jacobian 与 canonical folding 均显式。

| dB | validity | central P | odd alias P | clipping ratio | proxy→state max rel. error | Fock→state max rel. error |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 3 | clipping dominated | 0.781298 | 0.218478 | 0.357689 | 56.04% | 0.136% |
| 5 | clipping dominated | 0.874209 | 0.125786 | 0.514255 | — | — |
| 8 | clipping dominated | 0.964237 | 0.035763 | 0.783272 | — | — |
| 10 | localized validated | 0.989494 | 0.010506 | 0.913080 | 0.748% | 0.171% |
| 12 | localized validated | 0.997871 | 0.002129 | 0.976111 | 0.0135% | 4.39% |

3 dB 的四逻辑态 direct variance spread 为目标峰方差的 `37.33%`，因此低 squeezing
证否不是“数值不够精确”，而是 isolated localized-peak proxy 本身失效。12 dB 的 Fock
误差增至 `4.39%`，同时最小 cutoff capture 为 `0.99766`，该截断敏感度保留到 T2.3.3
归因，未被包装为无限维收敛。

## 5. 非 demo 验证

- 45 项 direct tests；高斯 cell 概率与 domain-conditioned moments 由独立 quadrature
  重算，不调用 production alias helper；
- 40 万样本单轴 alias 检验，以及 production 20 万样本二维 MC；最大 z-score `0.6202`；
- production decision-covariance relative error `0.002934`；
- `G=I` 时 `V_out=V_meas`，证明 continuous input-noise refresh；`G=0` 时保留
  post-loss input covariance；
- measurement efficiency 降低严格增加 equivalent noise；loss bias 对 lattice index
  的 1:3 比例得到解析复核；
- diagonal covariance 才给 exact Pauli distribution，correlated covariance fail closed；
- 3/10/12 dB 的 state/Fock q-domain 对照和低 squeezing state-dependence 证否门全部通过。

## 6. Claim 边界与后续

允许写：decoder-standardized-coordinate、Heisenberg-inspired signal/noise/jump surrogate；在本任务
门控下，约 10 dB 及以上与 state/Fock q-domain moments 对齐。

禁止写：paper-exact teleportation correction、SBS-exact dynamics、完整 q/p coherent
state fidelity、device calibration、pulse-level dynamics、真实逻辑 lifetime 或硬件结果。

T2.3.3 已在统一 sweep 上把该代理与 Fock、effective finite-squeezing 和 syndrome lane
分列，并保留低 squeezing/clipping、Fock cutoff 与坐标/指标口径的误差归因。
