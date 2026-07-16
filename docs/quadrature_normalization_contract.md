# GKP quadrature normalization contract

**Task：** T-RISK-20260714-01  
**日期：** 2026-07-14  
**状态：** Verified contract  
**机器结果：** `docs/t_risk_20260714_01_quadrature_validation.json`

## 1. 结论与边界

仓库旧常数 `LATTICE_CONST=sqrt(2*pi)` 继续保留，但其唯一语义是：两个独立
classical decoder-standardized axis 上，相邻 logical coset 的 correction-cell spacing。
它不是 canonical oscillator 的 logical spacing，也不能把两个 axis 同时解释为一对
quantum operators。

canonical Fock convention 冻结为

\[
x=\frac{a+a^\dagger}{\sqrt 2},\qquad
p=\frac{i(a^\dagger-a)}{\sqrt 2},\qquad [x,p]=i.
\]

square qubit GKP 在此 convention 下的相邻 logical-coset spacing 为 `sqrt(pi)`，
stabilizer translation 为 `2*sqrt(pi)`。所有 Fock operator、Hermite wavefunction、
Fourier transform 和 physical covariance 必须使用这一 canonical chart。

## 2. 一手来源核验

| 来源 | 已核验事实 | 本地锚点 | 判定 |
| --- | --- | --- | --- |
| Gottesman--Kitaev--Preskill, *Encoding a qubit in an oscillator* | 代码保护 canonical `q/p` shift，square-grid reciprocal structure | [官方 arXiv](https://arxiv.org/abs/quant-ph/0008040) | Verified |
| Campagne-Ibarcq et al., *Quantum error correction of a qubit encoded in grid states of an oscillator* | `[q,p]=i`；square-code stabilizer displacement 写作 `2*sqrt(pi)`，logical Pauli 是一半；syndrome 模 stabilizer reciprocal period | `relative_papers/Quantum_error_correction_of_a_qubit_encoded_in_grid_states_of_an_oscillator/Quantum_error_correction_of_a_qubit_encoded_in_grid_states_of_an_oscillator.md:13`；[官方 arXiv](https://arxiv.org/abs/1907.12487) | Verified |
| Sivak et al., *Real-time quantum error correction beyond break-even* | complex displacement-amplitude convention 中 `l_S=sqrt(2*pi)`，logical displacement 为 `l_S/2`；square matrix `det M=1`；`x,p` 定义及 `(x,p)->(-p,x)` Fourier rotation | `relative_papers/Real-time_quantum_error_correction_beyond_break-even/Real-time_quantum_error_correction_beyond-break-even.md:795`、`:821`、`:823`、`:849`；[官方 arXiv](https://arxiv.org/abs/2211.09116) | Verified |

注意：Campagne 文中的 canonical displacement vector 与 Sivak 文中的 complex
annihilation-operator amplitude 不是同一个坐标参数。`sqrt(2*pi)` 在本仓库 decoder cell
与 Sivak `l_S` 中数值相同，只是数值巧合，不能共用语义。

本次按 `nature-academic-search` 的 citation-verification + minimal multi-source-search
流程执行。学术 MCP 在当前会话不可用，因此按该流程降级为本地一手全文与官方 arXiv
交叉核验；未使用二手网页填公式。

## 3. 四种 chart

令 `s=sqrt(2)`，canonical phase vector 为 `z_c=(x,p)^T`。

| chart | 从 canonical 的 map | logical-cell spacing `(q,p)` | commutator multiplier / determinant | 允许用途 |
| --- | --- | --- | --- | --- |
| `canonical_fock` | `diag(1,1)` | `(sqrt(pi), sqrt(pi))` | `1` | Fock、Fourier、operator、physical covariance |
| `decoder_standardized` | `diag(s,s)` | `(sqrt(2*pi), sqrt(2*pi))` | `2` | 两个独立 classical syndrome axis；禁止解释为 joint operator pair |
| `symplectic_bridge` | `diag(s,1/s)` | `(sqrt(2*pi), sqrt(pi/2))` | `1` | 需要保持 `[q,p]=i` 的 anisotropic quantum bridge |
| `displacement_amplitude` | `diag(1/s,1/s)` | `(sqrt(pi/2), sqrt(pi/2))` | `1/2` | complex `alpha` 的实部/虚部与 Weyl phase；不是 canonical pair |

因此：

\[
q_d=sx,\ p_d=sp
\]

可用于两个独立 decoder feature，但 `det=2`，不是 symplectic map。若要求保持
commutator 且让 q 轴具有相同尺度，只能使用

\[
q_s=sx,\qquad p_s=p/s.
\]

## 4. wavefunction、covariance 与 dB 映射

对一维 coordinate dilation `q_d=sx`，归一化 wavefunction 必须满足

\[
\psi_d(q_d)=\frac{1}{\sqrt{s}}\psi_c(q_d/s),\qquad
\psi_c(x)=\sqrt{s}\,\psi_d(sx).
\]

方差和 covariance 使用线性 map `V_t=M V_s M^T`。canonical vacuum variance 为
`1/2`；decoder-standardized 单轴 vacuum variance 为 `1`。因此同一 squeezing dB 的
isolated probability-peak variance 为

\[
V_{\rm peak}^{(c)}=\frac12 10^{-dB/10},\qquad
V_{\rm peak}^{(d)}=10^{-dB/10}.
\]

对 `N_Delta=exp(-Delta^2 n)` damped-projector family：

\[
V_{\rm peak}^{(c)}=\frac12\tanh(\Delta^2),\qquad
\Delta(dB)=\sqrt{\operatorname{atanh}(10^{-dB/10})}.
\]

decoder chart 中 amplitude variance 必须乘 `s^2`，envelope inverse width 必须除以
`s`。旧实现只把 comb centers 乘尺度、没有同步 width/envelope，这也是旧 Fourier-p
audit 失真的组成部分。

## 5. Fourier 与 reciprocal lattice

采用

\[
\tilde\psi(p)=\frac{1}{\sqrt{2\pi}}\int e^{-ipx}\psi(x)\,dx,
\]

Hermite/Fock coefficient 的 Fourier phase 为 `(-i)^n`。canonical logical spacing 与
stabilizer spacing 满足

\[
\sqrt\pi\,(2\sqrt\pi)=2\pi,
\]

即 logical comb 的 reciprocal spacing 正好是 stabilizer spacing。Fourier 只能在
canonical chart 或明确的 symplectic chart 中执行，禁止直接把 decoder
`sqrt(2*pi)` cell 当作 canonical Fourier domain spacing。

## 6. 根因与修复

旧 audit 同时做了三件不相容的事：

1. 将 decoder logical-cell `sqrt(2*pi)` 当作 canonical Fock folding period；
2. 将 decoder 两轴 isotropic scaling 当成一对保持 commutator 的 operator；
3. q-wavefunction projection 只缩 centers，漏掉 width、envelope、Jacobian 与 noise
   variance 的完整 dilation。

现在：

- `physics/quadrature_conventions.py` 集中注册 chart、辛性、vector/covariance/sigma、
  wavefunction Jacobian 与机器验证；
- `physics/finite_energy_gkp.py` 对 damped-projector 做完整 coordinate dilation；
- `physics/fock_density_model.py` 的标准 GKP preparation 只接受注册的
  decoder-q → canonical-x `sqrt(2)` bridge；任意探索性 dilation 只能走低层 API；
- `physics/cross_fidelity_validation.py` 的 q/p Fock response 都在 canonical chart 中
  folding；旧含混路径以 `legacy_ambiguous_operational_fourier` 单独保留为负证据；
- `physics/noise_transfer_surrogate.py` 与 `physics/finite_squeezing_noise.py` 的
  dB/peak variance 变为 chart-qualified；
- `physics/constants.py` 和 `docs/paper_parameter_registry.json` 不再把同值常数混为同义。

## 7. 验证结果

机器 contract 的 15/15 gates 通过，包括：

- canonical reciprocal lattice、paper `l_S/2` 与 symplectic cell-area identities；
- decoder `det=2` 的主动拒绝与 anisotropic bridge `det=1`；
- phase vector/covariance/peak position roundtrip；
- wavefunction norm/Jacobian、canonical q/p Gaussian moment 和 FFT roundtrip；
- canonical↔decoder Gaussian parity-alias probability invariance。

finite-state/Fock 交叉验证进一步得到：

- 10/12 dB 最大 canonical Fock `|q-LER-p-LER| = 1.51e-7`；
- legacy ambiguous path 在高 squeezing 的最小 `p-LER-q-LER = 0.4182`，负证据未删除；
- high-squeezing noise-transfer ↔ direct syndrome 最大 q-LER gap `3.93e-5`；
- 12 dB Fock ↔ direct syndrome 最大 q-LER gap `4.61e-4`，由 cutoff tail 主导；
- 3 dB 仍有 `0.01541` 的 clipping/model-form gap，不用于代理校准。

`tests/test_quadrature_conventions.py` 还覆盖四逻辑态 wavefunction dilation、canonical
source 与 registered Fock preparation 等价、mixed-contract negative paths、10/12 dB q/p
alias 以及 legacy mismatch。

## 8. Claim 规则

允许：

- 写 canonical/decoder/displacement/symplectic normalization 已冻结并验证；
- 写 axis-resolved canonical Fock q/p response 在 10/12 dB 对齐；
- 对两个独立 classical decoder axes 做明确标注的 independent-axis Pauli projection。

禁止：

- 把 decoder-standardized `(q_d,p_d)` 称为 canonical quantum pair；
- 因 axis-resolved q/p 对齐就声称已验证 coherent joint-axis correlation/process fidelity；
- 把 `LATTICE_CONST` 与 paper `l_S` 因数值相同而直接互换；
- 把 analytic dB/variance map 当作 source-device envelope、`nbar` 或实验 calibration。
