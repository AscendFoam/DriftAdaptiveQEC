# T2.3.1 finite-cutoff GKP 密度矩阵模型

**实现：** `physics/fock_density_model.py`  
**机器验证：** `docs/t2_3_1_fock_density_validation.json`  
**边界：** 单腔模有限 Fock 截断 reference；不是 transmon、多能级器件、脉冲 Hamiltonian 或实板模型。

## 1. 为什么独立实现

仓库旧 `gkp_state.py` 在 Strawberry Fields 不可用时会退化为启发式 signed-grid
可视化，不能生成可审计的密度矩阵。T2.3.1 因此不复用该 fallback，而是用本地
NumPy/SciPy 构造有限维谐振子算符、密度矩阵和 CPTP 通道。它与 T2.1/T2.2 的高速
syndrome-level 路径相互独立，后续 T2.3.3 才比较两种 fidelity。

## 2. 态制备与截断语义

使用仓库已经归一化的 damped-projector position wavefunction
`psi_Delta(q)`，在同一固定 q-window 上投影到

```text
phi_n(q) = pi^(-1/4) H_n(q) exp(-q^2/2) / sqrt(2^n n!)
c_n      = integral phi_n(q) psi_Delta(q) dq .
```

T2.3.2 增加了显式 `source_coordinate_scale=s` 桥：投影前执行
`psi_Fock(q)=sqrt(s) psi_source(s q)`。T-RISK-20260714-01 进一步发现，只缩 centers
而不缩 width/envelope 仍是伪映射；现已把 decoder damped-projector 整个 wavefunction
按 `s=sqrt(2)` dilation，并让标准 `prepare_damped_projector_gkp` 只接受这一注册桥。
任意 `s` 的探索性投影仍可走低层 `project_finite_energy_gkp`，但不能冒充标准 GKP
canonical preparation。

Hermite functions 用三项递推生成，积分采用 Simpson quadrature。返回值同时保存：

- 截断前投影系数 `c_n`；
- `sum_{n<N}|c_n|^2` captured probability；
- 在 N 维内重新归一化的纯态密度矩阵；
- q-window、grid points、source model 与 logical state。

修正后的 production cutoff `18/24/30/36` captured probability 为
`0.999729/0.999896/0.9999984/0.9999996`，相邻嵌入 fidelity 为
`0.999833/0.999897/0.9999988`。因此默认最高 `N=36, Delta=0.45` 的 capture 与最后
相邻 embedding 都超过 `99.999%`；这仍不是所有 Delta、所有通道强度可复用的统一 cutoff。

## 3. 通道

### 3.1 Displacement

显式计算 `D(alpha)=exp(alpha a^dagger-alpha* a)` 并执行 `D rho D^dagger`。
有限矩阵指数严格酉；production 正反位移 Frobenius error 为 `2.02e-16`。

### 3.2 Pure loss

使用完整有限截断 Kraus 和：

```text
K_l = sum_{n=l}^{N-1} sqrt(C(n,l))
      (1-eta)^(l/2) eta^((n-l)/2) |n-l><n| .
```

它保留全部截断内 coherence，不是只更新 mean photon number。`|5>` 基准满足
`<n>_out=eta<n>_in`，误差 `4.44e-16`；测试还逐项检查 binomial population 和
`eta=0/1` 端点。

### 3.3 Thermal excitation

构造截断算符的 sparse Lindblad superoperator，直接用 `expm_multiply` 作用于
column-major vectorized density matrix：

```text
L = (n_th+1) D[a] + n_th D[a^dagger].
```

真空解析基准 `<n>(t)=n_th(1-exp(-t))` 的误差为 `4.16e-17`；`n_th=0` 又与
`eta=exp(-t)` 的 pure-loss Kraus map 做了矩阵级交叉检查。为阻止无界内存误用，
该 sparse path 对 `N>48` fail closed。

### 3.4 Phase diffusion 与 Kerr

相位扩散直接执行
`rho_mn -> exp[-variance (m-n)^2/2] rho_mn`，保留 population；Kerr/anharmonicity
执行 `U=exp[-i chi n(n-1)/2]`，production population error 为 `3.47e-18`。

### 3.5 Modular measurement backaction

由截断内酉位移构造 Hermitian observable
`O=contrast Re[exp(-i phase)D(beta)]` 和 `E_±=(I±O)/2`。采用正半定 effect 的
Lüders square root 更新条件态，而不是只抽取一个 classical bit。测试检查 effect
正定、`E_++E_-=I`、条件概率和为一、后验态正定、nonselective/conditional mixture
相等，以及 zero-contrast 不扰动态端点。

### 3.6 High-Fock leakage proxy

该接口是一个显式 CPTP 的“向上移动一个腔体光子”压力项，最高截断边界保持不变。
它只用于 high-occupation sensitivity，不含 transmon `|f>`、readout classifier、reset
或 cavity--transmon exchange。任何文档都不得把它写成真实 ancillary leakage。

## 4. 反简化与验证

- 原 T2.3.1 的 38 项 direct tests 加上 T-RISK 的 chart/dilation/Fourier tests，覆盖算符、态合法性、cutoff/grid/coordinate-scale 收敛、解析通道、端点、组合通道、
  POVM/backaction、top-cutoff 边界、非法参数和 claim scope；
- finite-energy/finite-squeezing 相邻测试合计 81 项通过；
- production 10 项 gate 全 PASS；所有受测输出最小 eigenvalue 为 `-1.33e-16`，属于
  浮点误差；
- 不安装 qutip/Strawberry Fields，也不把缺依赖时的 heuristic fallback 当物理证据。

## 5. 当前允许与禁止的 claim

允许：finite-cutoff single-oscillator density-matrix reference、解析通道回归、cutoff
convergence、CPTP/measurement-backaction 验证。

禁止：完整一轮 sBs/sharpen--trim 已在 Fock space 实现、真实 cavity--transmon master
equation、transmon leakage、装置 squeezing-dB 映射、无限维收敛、脉冲或硬件 fidelity。
这些分别由 T2.3.2/T2.3.3、后续校准和硬件 evidence gate 承接。
