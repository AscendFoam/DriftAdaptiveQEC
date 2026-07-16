# T2.3.3 四 lane 跨保真度交叉验证

**日期：** 2026-07-14  
**状态：** Done；T-RISK-20260714-01 coordinate correction 已回灌  
**实现：** `physics/cross_fidelity_validation.py`  
**机器证据：** `docs/t2_3_3_cross_fidelity_validation.json`

## 1. 比较契约

四条 lane 是：

1. finite-cutoff Fock quadrature response + completed analytic SBS native metrics；
2. decomposed finite-squeezing stochastic effective model；
3. Heisenberg-inspired noise-transfer surrogate；
4. normalized finite-energy state-density syndrome MAP model。

共同输入固定为 `3/5/8/10/12 dB`，并用

```text
V_peak_canonical = 0.5 * 10^(-dB/10)
V_peak_decoder   =       10^(-dB/10)
Delta  = sqrt(atanh(10^(-dB/10)))
sigma_external_decoder^2 = 0.18^2 + 0.06^2 + V_peak_decoder
```

将 channel、measurement 和 ancilla peak 口径统一。Fock q/p 都先逐轴从 classical
decoder chart 转入 canonical `[x,p]=i` chart，再做 Hermite/Fourier folding；在明确的
independent-axis projection 下生成 Pauli-twirled `LER`、correct-coset occupancy 和
`F_avg`。逐轴转换不是把 decoder 两轴冒充为 joint canonical operator pair，也不冒充
coherent process tomography。

模型原生层保持分离：Fock SBS code survival、effective central-domain occupancy、
noise-transfer central-cell mass 具有不同分母，只比较方向，不做绝对排名。

## 2. 生产结果

| dB | Fock q-axis LER | Fock p-axis LER | Fock two-axis LER | effective LER | noise-transfer LER | direct-syndrome LER | Fock SBS survival | clipping ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 0.233904 | 0.203707 | 0.389963 | 0.358892 | 0.389252 | 0.413098 | 0.588863 | 0.357665 |
| 5 | 0.128041 | 0.122682 | 0.235015 | 0.224843 | 0.234878 | 0.239687 | 0.819566 | 0.515333 |
| 8 | 0.0351584 | 0.0351122 | 0.0690361 | 0.0673077 | 0.0685798 | 0.0690822 | 0.954687 | 0.786986 |
| 10 | 0.00992772 | 0.00992757 | 0.0197567 | 0.0191128 | 0.0196678 | 0.0197455 | 0.985044 | 0.917134 |
| 12 | 0.00232313 | 0.00232313 | 0.00464087 | 0.00381635 | 0.00371247 | 0.00371999 | 0.986324 | 0.978680 |

四条共同 lane 的 LER 均随 squeezing 严格下降，correct-coset occupancy 和 Pauli-twirled
`F_avg` 均严格上升。Fock SBS code survival/code-weighted fidelity 与两个 central-domain
occupancy 也严格上升。

高压缩区：

- noise-transfer 与 direct-syndrome q-LER 最大绝对差 `3.93e-5`；
- effective 200k Monte Carlo 与 noise-transfer 最大 z-score `1.708`；
- canonical Fock q/p 最大轴差 `1.51e-7`；
- Fock-q 与 direct-syndrome 最大绝对差 `4.61e-4`，来自 12 dB cutoff tail sensitivity。

低压缩区：3 dB noise-transfer 与 direct-syndrome q-LER 差 `0.015408`，同时 clipping
ratio 仅 `0.358`，因此作为明确证否，不用于 surrogate calibration。

## 3. Fock cutoff 归因

12 dB 的 alias rate 对很小的尾概率敏感：

| cutoff | q-axis LER | minimum captured probability |
| ---: | ---: | ---: |
| 24 | 0.0110094 | 0.930910 |
| 30 | 0.00600592 | 0.976409 |
| 36 | 0.00393826 | 0.989168 |
| 42 | 0.00319232 | 0.991136 |
| 48 | 0.00232313 | 0.997664 |

LER 随 cutoff 严格下降、capture 严格上升，但 `N=48` 尚未等于无限维结果。因此 12 dB
使用绝对差门，不使用会被 rare-tail 分母放大的相对误差门。

## 4. 不一致与误差归因

| ID | 区域 | 观测 | 归因 | 报告规则 |
| --- | --- | --- | --- | --- |
| `XA-LOW-CLIPPING` | 3/5 dB | Gaussian proxy 与 coherent state-density 分离 | localized-peak assumption 在 clipping/state-envelope overlap 下失效 | 低 dB 只作 falsification |
| `XA-HIGH-CUTOFF` | 12 dB | Fock alias tail 随 `N=24→48` 明显变化 | narrow peaks 需要更高 photon cutoff | 保留全 cutoff sweep，不称无限维收敛 |
| `XA-P-COORDINATE` | 全域，10/12 dB 最明显 | canonical q/p 轴差 `<1.51e-7`；legacy audit 仍差 `>0.418` | 旧路径混用 decoder cell、canonical Fourier domain，并漏掉 width/envelope/Jacobian dilation | 允许 axis-resolved q/p；independent-axis projection 不升级 coherent joint-axis claim |
| `XA-OCCUPANCY-SEMANTICS` | 全域 | code survival、central-domain mass、correct-coset probability 数值不同 | protocol leakage、domain localization 与 logical correctness 是不同事件 | native occupancy 只比较方向 |

第三项已由 `T-RISK-20260714-01` 解释并修复：canonical 路径通过 q/p Fourier 对齐，旧路径
没有删除，而是以 `legacy_ambiguous_operational_fourier` 保留为负证据。PC-N01 的坐标/解析
子门已通过；source-device envelope/`nbar` 校准及 coherent joint-axis correlation 仍 fail closed。

## 5. 非 demo 验证

- Fock folded density 与独立 direct state-density syndrome response 在 3/10 dB 对照；
- Fock probability mass、Hermite reconstruction、`N=24/30/36/42/48` cutoff 与四逻辑态
  q/p negative audit；
- effective lane 每点 200k 样本，noise-transfer exact Gaussian alias；
- Fock SBS 每点六 Pauli eigenstates、perfect-readout completed one-round；
- 15 production gates 和四条结构化 error attribution 全通过；
- scope validator、demo-sized sample rejection、metric/input negative paths 均有测试。

## 6. Claim 边界

允许：四 lane 在 chart-qualified independent-axis Pauli projection 下的方向一致；10/12 dB
noise-transfer/effective/syndrome 高压缩对齐；canonical Fock q/p 轴对齐；Fock-q 的
cutoff-aware 绝对差；低压缩 clipping 失效区。

禁止：四模型绝对等价、occupancy 数值可直接排名、Fock 无限维收敛、coherent
joint-axis correlation/process `F_avg`、device-calibrated squeezing、pulse/transmon 或硬件结论。
