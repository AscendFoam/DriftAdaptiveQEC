下面是一套**从零启动到形成高质量论文**的科研路线的规划。目标限定在以下范围：**近似 GKP 码本身的解码，不引入外层表面码/QLDPC 等纠错码；CNN 慢回路做噪声漂移估计或解码器重标定，FPGA 快回路做低延迟恢复/Pauli frame 更新；先完成理论与高保真仿真，论文初稿先成形，再补真实 FPGA 结果**。

我建议这篇论文的核心叙事不要写成“CNN 在所有情况下打败最大似然解码”，而应写成：

> **在有限能量近似 GKP 态、非理想 syndrome extraction、损耗/测量噪声/辅助态噪声以及时间漂移共同存在时，静态解码器会发生模型失配；双回路架构用 CNN 慢回路在线估计噪声状态，用 FPGA 快回路执行硬件可部署的 finite-energy-aware MAP/LUT 解码，从而降低漂移平均 LER、tail LER 和逻辑通道保真度损失，同时保持确定性低延迟。**

这个叙事与文献更稳：GKP 原始码本身就是为校正 (q,p) 小位移误差而设计；有限能量近似 GKP 会使 standard binning 不总是最优；已有 memory-assisted decoder 用多轮 syndrome 做 Bayesian 更新；已有 FPGA 神经网络实时 QEC 主要是在 surface code 上，而不是单模近似 GKP 本体解码。([arXiv][1])

---

# 0. 项目总目标与论文主张

## 0.1 建议论文题目方向

可以考虑以下几类题目：

**题目 A，偏物理与纠错：**
**Drift-Adaptive Real-Time Decoding of Approximate GKP Qubits with a CNN–FPGA Dual-Loop Architecture**

**题目 B，偏硬件协同设计：**
**Hardware-Aware Adaptive Decoding for Finite-Energy GKP Codes under Nonstationary Bosonic Noise**

**题目 C，偏算法与仿真：**
**Finite-Energy-Aware Adaptive Decoding of GKP Qubits from Continuous Syndrome Streams**

我最推荐 B 或 C，因为在真实 FPGA 结果还没补齐之前，B/C 比 A 更稳。

---

## 0.2 最终论文应证明的 5 个主张

你的论文至少应证明以下 5 个 claim：

**Claim 1：模型层面**
建立一个足够真实的近似 GKP 解码仿真框架，覆盖有限 squeezing、辅助态噪声、syndrome 测量噪声、损耗、非等方高斯位移、均值漂移、相关噪声、burst/outlier 噪声和硬件量化延迟。

**Claim 2：算法层面**
提出 CNN 慢回路 + FPGA 快回路的双时间尺度解码架构。CNN 不一定直接输出纠错动作，而是更稳妥地输出当前噪声状态或 likelihood-table 参数；FPGA 快回路执行低延迟 MAP/LUT/阈值解码。

**Claim 3：性能层面**
在非平稳噪声下，相比 standard binning、静态 MAP、静态 Bayesian、多轮 memory-assisted 但参数固定的 decoder，你的双回路 decoder 降低 drift-averaged LER、tail-window LER，并缩小与 oracle MAP 的差距。

**Claim 4：逻辑通道层面**
不仅报告分类准确率或单轮 LER，还要报告 (p_X,p_Z,p_Y)、average logical fidelity、entanglement fidelity、Pauli transfer matrix、effective squeezing、pseudo-threshold 或 break-even boundary。有限能量 GKP 的逻辑通道不能粗暴等同为理想 Pauli channel，这一点应在论文中明确。([arXiv][2])

**Claim 5：硬件可部署层面**
即使真实 FPGA 实验放在最后，也要在初稿前完成 fixed-point、LUT、pipeline、ADC quantization、closed-loop latency、resource estimate 的硬件感知仿真。真实 FPGA 结果补上后，论文从“理论+仿真”升级为“算法—物理—硬件协同设计”。

---

# 1. Phase 0：问题冻结与文献地图

## Milestone 0.1：冻结研究对象

**目标：** 避免项目一开始就扩散到 surface-GKP、QLDPC-GKP、光学 MBQC 等大系统。

### Task 0.1.1：冻结主对象

主对象建议设为：

[
\text{single-mode square approximate GKP qubit}
]

运行模式设为 repeated quantum memory error correction：

[
\rho_0
\rightarrow
\mathcal{N}_1
\rightarrow
\text{syndrome extraction}
\rightarrow
\text{decoder}
\rightarrow
\mathcal{R}_1
\rightarrow
\rho_1
\rightarrow \cdots
]

输出逻辑错误率和逻辑通道。

**完成标准：** 写出一页 scope note，明确“不研究外层码、不研究多模格码作为主线、不研究 GKP 态制备优化作为主线”。

### Task 0.1.2：冻结平台抽象

建议主仿真采用**超导 cavity + transmon-like syndrome extraction 抽象**，因为 FPGA 快回路在这个平台叙事最强；但数学模型要保持平台无关。超导腔已有 GKP 制备、实时 QEC、autonomous QEC、beyond-break-even 等实验支撑；困离子和光学平台可作为讨论或参数迁移。([arXiv][3])

**完成标准：** 画出一张 cycle diagram：idle noise → syndrome extraction → ADC/measurement → decoder → frame/correction → next cycle。

### Task 0.1.3：冻结双回路接口

定义慢回路和快回路：

[
\hat{\theta}_t
==============

g_{\phi}^{\rm CNN}
(s_{t-W:t},a_{t-W:t},c_{t-W:t})
]

[
\hat{a}_t
=========

f_{\hat{\theta}_t}^{\rm FPGA}
(s_t,h_t)
]

其中 (s_t=(s_{q,t},s_{p,t}))，(\hat{\theta}_t) 可以包含：

[
\hat{\theta}_t=
(\hat{\mu}_q,\hat{\mu}*p,
\hat{\sigma}*q,\hat{\sigma}*p,
\hat{\rho}*{qp},
\hat{\sigma}*{\rm meas},
\hat{\eta},
\hat{p}*{\rm outlier})
]

**完成标准：** 给出输入、输出、更新频率、位宽、fallback 策略的接口表。

---

## Milestone 0.2：文献矩阵

### Task 0.2.1：按四条线建 Zotero/文献表

四条线：

1. GKP 基础、有限能量、实验；
2. GKP 本体解码、standard binning、MAP、Bayesian、loss logical channel；
3. 自适应噪声估计、noise-aware decoding；
4. NN/QEC/FPGA 低延迟解码。

**完成标准：** 每篇文献提取 5 个字段：噪声模型、解码器、指标、是否有限能量、是否硬件实时。

### Task 0.2.2：形成 “gap statement”

你的 gap 可以写成：

> 已有 GKP 本体解码主要分析静态或已知噪声；已有 neural decoders 多集中在 surface code 或 GKP+外层码；已有 FPGA NN QEC 主要服务离散稳定子码；尚缺少针对单模近似 GKP 码、面向非平稳连续 syndrome 流、同时满足 finite-energy-aware 和 low-latency closed-loop 部署的双回路解码框架。

已有 GKP analog-information、finite-energy optimized decoding、memory-assisted Bayesian decoder、FPGA NN surface-code decoder 都是你的直接对照，而不是被忽略的竞争者。([arXiv][4])

**完成标准：** 写成论文 Introduction 的 3–4 段草稿。

---

# 2. Phase 1：理论模型与指标体系

## Milestone 1.1：理想 GKP syndrome-level 解码模型

### Task 1.1.1：推导 standard binning

对单个 quadrature，测得 syndrome (s\in[-\sqrt{\pi}/2,\sqrt{\pi}/2))，standard binning 选择最近晶格点。逻辑错误对应残余位移跨过半格边界。

对高斯随机位移 (u\sim \mathcal{N}(0,\sigma^2))，单方向逻辑翻转概率可写为：

[
p_{\rm flip}(\sigma)=
\sum_{k\in\mathbb{Z}}
\int_{(2k+1/2)\sqrt{\pi}}^{(2k+3/2)\sqrt{\pi}}
\frac{1}{\sqrt{2\pi}\sigma}
e^{-u^2/(2\sigma^2)}
du .
]

**完成标准：** Monte Carlo 与解析积分误差小于预设容差。

### Task 1.1.2：推导 MAP / soft-output likelihood

对给定 syndrome (s)，定义偶/奇逻辑陪集似然：

[
P_{\rm even}(s;\theta)
======================

\sum_{m\in 2\mathbb{Z}}
p_\theta(s+m\sqrt{\pi})
]

[
P_{\rm odd}(s;\theta)
=====================

\sum_{m\in 2\mathbb{Z}+1}
p_\theta(s+m\sqrt{\pi})
]

LLR：

[
\Lambda(s;\theta)
=================

\log\frac{P_{\rm even}(s;\theta)}
{P_{\rm odd}(s;\theta)} .
]

快回路最终可以只查 (\Lambda(s;\hat{\theta})) 的 LUT。

**完成标准：** 实现 standard binning、MAP hard decision、MAP soft-output 三种模式。

### Task 1.1.3：二维 (q,p) 相关噪声扩展

真实噪声可能有：

[
\Sigma_t=
\begin{pmatrix}
\sigma_q^2 & \rho\sigma_q\sigma_p\
\rho\sigma_q\sigma_p & \sigma_p^2
\end{pmatrix}
]

此时不能总是把 (q,p) 独立解码。需要实现二维 MAP：

[
P_{\ell_q,\ell_p}(s_q,s_p;\theta)
=================================

\sum_{m,n\in \mathcal{C}*{\ell_q,\ell_p}}
p*\theta(s_q+m\sqrt{\pi},s_p+n\sqrt{\pi}) .
]

**完成标准：** 展示 (\rho_{qp}=0) 时退化为两个一维解码器，(\rho_{qp}\neq0) 时二维 MAP 优于独立解码。

---

## Milestone 1.2：近似 GKP 有限能量模型

### Task 1.2.1：定义近似 GKP 态族

至少支持两种近似：

1. **高斯峰 + 高斯包络模型**；
2. **damped GKP state / finite-energy projector model**。

有限能量态的峰宽可用 (\Delta) 或 effective squeezing dB 表征。综述文献强调 finite squeezing 是 GKP 态质量和纠错性能的核心限制；Jafarzadeh 等还指出 finite-energy code states 会泄漏出理想 GKP code space，使 logical channel 分析更微妙。([arXiv][5])

**完成标准：** 生成 (|0_L^\Delta\rangle, |1_L^\Delta\rangle, |+_L^\Delta\rangle, |-^\Delta_L\rangle) 的 wavefunction/Wigner/syndrome distribution。

### Task 1.2.2：定义 finite-energy-aware decoding

不要只比较“是否跨过 (\sqrt{\pi}/2)”。需要将解码后态投影或嵌入到逻辑子空间，得到逻辑通道：

[
\mathcal{L}_{\rm dec}
=====================

\mathcal{D}*{\rm log}
\circ
\mathcal{R}*{\rm dec}
\circ
\mathcal{N}
\circ
\mathcal{E}_{\rm GKP}^{\Delta}.
]

**完成标准：** 对每个 decoder 输出 (p_X,p_Z,p_Y)、(F_{\rm avg})、(F_e)、Pauli transfer matrix。

### Task 1.2.3：复现一组文献趋势

复现至少一条已知趋势：

* finite-energy standard binning 在某些 regime 下次优；
* optimized decoder 在有限能量下优于 standard binning，但优势随能量升高收缩；
* memory-assisted Bayesian 多轮 syndrome 优于 single-round memoryless decoder。

这些趋势分别对应 Jafarzadeh 等和 Wan 等的主线。([arXiv][2])

**若证否：** 先不要继续 CNN。优先检查单位约定、(\sqrt{\pi}) 缩放、finite-energy state normalization、loss/displacement 顺序、syndrome wrap 区间、随机数采样和投影定义。

---

## Milestone 1.3：非平稳噪声漂移模型

### Task 1.3.1：定义 drift process

至少支持：

[
\mu_t = \mu_{t-1} + \epsilon_t
]

[
\sigma_t = \sigma_0[1+A\sin(2\pi f t+\phi)] + \xi_t
]

[
\eta_t = \eta_0 + \delta\eta_t
]

[
p_{\rm outlier}(t)
==================

p_0 + \delta p_t
]

还要支持 step drift、telegraph drift、burst drift。

### Task 1.3.2：定义 oracle MAP

oracle MAP 知道真实 (\theta_t)，是不可部署上界：

[
\hat{a}_t^{\rm oracle}
======================

\arg\max_a P(a|s_t,\theta_t)
]

你的双回路目标不是超过 oracle，而是缩小与 oracle 的 gap。

### Task 1.3.3：定义 regret / oracle gap

建议用：

[
G_{\rm oracle}
==============

\frac{
\overline{P_L^{\rm dual}}-\overline{P_L^{\rm oracle}}
}{
\overline{P_L^{\rm static}}-\overline{P_L^{\rm oracle}}
}.
]

越接近 0 越好。若 (G_{\rm oracle}<0.2)，叙事很强：说明双回路吃掉了大部分静态模型失配损失。

### Task 1.3.4：与已有自适应噪声估计对齐

已有 syndrome-only drifting noise estimation 工作说明：传统 stationary 噪声假设会导致次优解码，滑动窗口 syndrome statistics 可用于恢复 drift frequency 并改善 logical error rate。你的 CNN 慢回路应把这类方法作为 baseline，而不是忽略。([arXiv][6])

**完成标准：** 在 synthetic drift 下，static MAP 与 oracle MAP 之间出现足够大的可利用差距；否则该项目的“自适应优势”没有空间。

**若证否：**
如果 static MAP 与 oracle MAP 几乎重合，说明噪声漂移设得太弱、指标不敏感，或单轮 GKP 对该漂移不敏感。替代方向：

1. 加入 measurement noise drift；
2. 加入 (q/p) bias drift；
3. 加入 offset drift (\mu_q,\mu_p)；
4. 从单轮 LER 换成多轮 logical lifetime；
5. 将 claim 改为“置信度校准与 tail-risk 降低”，而非平均 LER 大幅降低。

---

# 3. Phase 2：高保真仿真平台

这是整个项目能否投好期刊的关键。你的论文初稿在真实 FPGA 实验前必须让审稿人相信：**仿真不是玩具模型，而是真实 GKP 纠错流程的可信近似。**

---

## Milestone 2.1：syndrome-level fast simulator

### Task 2.1.1：实现 syndrome stream 生成器

输入：

[
\theta_t =
(\mu_q,\mu_p,\sigma_q,\sigma_p,\rho,\eta,\sigma_{\rm meas},p_{\rm outlier})
]

输出：

[
s_t=(s_{q,t},s_{p,t}),\quad y_t=(\text{true logical } X/Z/Y)
]

### Task 2.1.2：实现多轮 memory

状态变量至少包括：

* accumulated residual shift；
* previous correction；
* decoder confidence；
* Pauli frame；
* optional active displacement error。

### Task 2.1.3：实现高速 Monte Carlo

目标是能跑到低 LER 区间。若目标 LER 是 (10^{-4})，每个点至少需要 (10^6) 量级 shots 才能稳定估计；更低 LER 需要 importance sampling 或 rare-event sampling。

**完成标准：** 输出 (P_L(\sigma))、confidence interval、seed reproducibility。

---

## Milestone 2.2：finite-energy effective simulator

### Task 2.2.1：加入有限 squeezing 等效噪声

初始近似：

[
\sigma_{\rm eff}^2
==================

\sigma_{\rm channel}^2
+
\sigma_{\rm data,GKP}^2
+
\sigma_{\rm ancilla,GKP}^2
+
\sigma_{\rm meas}^2 .
]

但不要长期停留在这个等效模型，后面必须用 Fock-space 或文献数据验证。

### Task 2.2.2：加入辅助态与测量噪声

Steane-type GKP correction 和 teleportation/Knill-type correction 的信息流不同。Marqversen 等 2025 年给出了 Knill 与 Steane 型 GKP error correction 的性能分析，可作为你选择 syndrome extraction model 的理论对照。([arXiv][7])

### Task 2.2.3：加入 active correction imperfection

实际恢复位移不应假设完美。加入：

[
D(-\hat{u}) \rightarrow D(-\hat{u}+\epsilon_{\rm ctrl})
]

并加入 DAC/AWG 量化误差、pulse amplitude miscalibration、latency-induced extra noise。

**完成标准：** finite-energy effective simulator 与 ideal syndrome-level simulator 在 (\Delta\rightarrow0) 或 high squeezing 极限下收敛。

---

## Milestone 2.3：Fock-space / density-matrix 验证器

### Task 2.3.1：构造有限 cutoff 模型

选择 cutoff (N_{\rm cut})，构造 approximate GKP density matrix。支持：

* displacement noise；
* photon loss；
* thermal excitation；
* phase diffusion；
* small Kerr / anharmonicity；
* measurement backaction。

### Task 2.3.2：实现一轮完整纠错

一轮流程：

1. 初始化 (\rho_L^\Delta)；
2. idle noise (\mathcal{N}_{\rm idle})；
3. syndrome extraction circuit；
4. noisy measurement；
5. classical decoder；
6. active displacement 或 Pauli frame；
7. logical projection / tomography。

### Task 2.3.3：对照 loss logical channel

Photon loss 是光学和 bosonic 平台的重要噪声，且 pure loss 诱导的 GKP 逻辑通道不应简单当成 stochastic Pauli channel。Hastrup–Andersen 和 Harris 等关于 loss correction / loss-induced logical channel 的分析可以作为你的验证参考。([Welcome to DTU Research Database][8])

**完成标准：** 在 small cutoff 可承受范围内，Fock 模型与 effective simulator 的 LER 和 (F_{\rm avg}) 趋势一致。

**若证否：**
如果 syndrome-level 优势在 Fock-space 中消失，需要定位原因：

1. finite-energy envelope 改变最优决策边界；
2. loss 产生非 Pauli / non-Gaussian 逻辑通道；
3. active correction 引入额外能量或 distortion；
4. CNN 学到的是 syndrome-level artifact。

替代方向：

* 改用 finite-energy-aware loss function；
* 从 hard correction 改成 soft correction / confidence flag；
* 让 CNN 输出 decoder calibration 而不是直接输出 action；
* 用 Fock-space 生成少量高保真校准数据，fine-tune 慢回路。

---

## Milestone 2.4：硬件时序仿真器

### Task 2.4.1：建立 closed-loop timing model

[
L_{\rm closed}
==============

L_{\rm meas}
+
L_{\rm ADC}
+
L_{\rm transfer}
+
L_{\rm decode}
+
L_{\rm command}
+
L_{\rm AWG}
+
L_{\rm pulse}.
]

### Task 2.4.2：加入 decoder backlog

如果 decoder 不能在每个 QEC cycle 内完成，会发生 backlog。实时 QEC 文献强调 decoder 必须同时满足高吞吐和低延迟；2026 年 FPGA NN surface-code 实验报告了 550 ns deterministic closed-loop latency，其中 NN decoding 124 ns，说明“硬件解码延迟”本身已经成为 QEC 论文的重要指标。([arXiv][9])

### Task 2.4.3：加入 fixed-point 和 LUT 误差

模拟：

* syndrome ADC bit width；
* LUT address bit width；
* LLR quantization；
* threshold quantization；
* CNN output quantization；
* parameter update granularity。

**完成标准：** 得到 latency–resource–LER 三维 tradeoff 曲线。

---

# 4. Phase 3：强 baseline 体系

如果 baseline 弱，项目叙事会被审稿人直接击穿。你至少需要以下 baseline。

---

## Milestone 3.1：基础 baseline

### Task 3.1.1：standard binning

固定半格边界，最近格点恢复。

### Task 3.1.2：static MAP

用训练集平均噪声参数 (\bar{\theta})，不随时间更新：

[
\hat{a}_t=\arg\max_a P(a|s_t,\bar{\theta}).
]

### Task 3.1.3：oracle MAP

知道真实 (\theta_t)，作为上界。

### Task 3.1.4：static finite-energy optimized decoder

参考 Jafarzadeh 等的 finite-energy decoder 思路，至少实现一个“不是 simple binning”的 optimized static decoder。([arXiv][2])

**完成标准：** 所有主要结果图都同时出现 standard binning、static MAP、oracle MAP、你的 decoder。

---

## Milestone 3.2：多轮和自适应 baseline

### Task 3.2.1：memory-assisted Bayesian decoder

Wan 等已经说明多轮 syndrome + Bayesian estimation 可提升近似 GKP 保护效果。你的双回路必须与它比较，否则容易被质疑“只是重新发明 Bayesian memory”。([arXiv][10])

### Task 3.2.2：EWMA / Kalman adaptive MAP

这是最强传统自适应 baseline：

[
\hat{\theta}_t
==============

\alpha \hat{\theta}_{t-1}
+
(1-\alpha)\tilde{\theta}_t.
]

如果 CNN 不能超过 Kalman/EWMA，你仍可转向“硬件低延迟 + 可解释 adaptive MAP”，但神经网络贡献要降调。

### Task 3.2.3：sliding-window syndrome estimator

参考 drifting-noise syndrome statistics 文献，实现 sliding-window / overlapping-window estimator。([arXiv][6])

### Task 3.2.4：postselection / repeat-syndrome policy

GKP syndrome 具有 analog confidence。Fukui 等早已强调 analog information 不应浪费；你的系统可以比较“低置信度时重复 syndrome extraction”与“直接恢复”的资源—性能折中。([arXiv][4])

**完成标准：** 形成 baseline ranking，而不是只拿 standard binning 做对照。

---

# 5. Phase 4：CNN + FPGA 双回路算法设计

## Milestone 4.1：CNN 慢回路

### Task 4.1.1：选择模型结构

优先顺序：

1. causal 1D-CNN；
2. temporal convolutional network；
3. lightweight GRU；
4. transformer 只作为后备，不建议第一版使用。

理由：你要能映射到硬件，且慢回路只需估计 drift，不需要庞大模型。

### Task 4.1.2：定义输入

输入窗口：

[
X_t=
[
s_{q,t-W:t},
s_{p,t-W:t},
\hat{a}*{t-W:t},
\Lambda*{t-W:t},
r_{t-W:t}
]
]

其中 (r_t) 是 correction residual 或 confidence。

### Task 4.1.3：定义输出

推荐输出噪声参数，而不是直接输出纠错动作：

[
\hat{\theta}_t=
(\hat{\mu}_q,\hat{\mu}_p,
\hat{\sigma}*q,\hat{\sigma}*p,
\hat{\rho},
\hat{\sigma}*{\rm meas},
\hat{\eta},
\hat{p}*{\rm outlier})
]

外加 uncertainty：

[
u_t=\text{epistemic/aleatoric uncertainty}.
]

### Task 4.1.4：定义损失函数

组合损失：

[
\mathcal{L}
===========

\lambda_1
|\hat{\theta}*t-\theta_t|^2
+
\lambda_2
\mathrm{CE}(\hat{y}*t,y_t)
+
\lambda_3
\mathrm{NLL}*{\rm syndrome}
+
\lambda_4
\mathrm{ECE}*{\rm calibration}.
]

不要只训练 classification accuracy。最终目标是降低 LER 和逻辑通道损失。

**完成标准：** CNN 在 held-out drift families 上估计 (\theta_t) 的误差低于 EWMA/Kalman，并且 uncertainty 与实际失误率校准。

**若证否：**
如果 CNN 不如 Kalman/EWMA，替代路线：

1. 改成 hybrid：Kalman 给初值，CNN 学 residual；
2. 用 CNN 做 OOD/burst detection，而非连续估计；
3. 用 CNN 只输出 LUT correction factor；
4. 改用 TCN/GRU；
5. 保留 FPGA adaptive MAP，降低神经网络 claim。

---

## Milestone 4.2：FPGA 快回路解码器

### Task 4.2.1：设计 parametric MAP-LUT

快回路执行：

[
s_t
\rightarrow
\text{address}
\rightarrow
\Lambda(s_t;\hat{\theta}_t)
\rightarrow
\text{decision / confidence}.
]

建议用 LUT 或 piecewise-linear approximation，而不是把大 CNN 放入快回路。

### Task 4.2.2：设计硬件输出动作

快回路输出：

1. active displacement command；
2. Pauli frame update；
3. confidence；
4. repeat-syndrome flag；
5. fallback flag。

### Task 4.2.3：设计 fallback policy

当 CNN uncertainty 高、输入 OOD、LUT 参数超界时，回退到 conservative static MAP 或 standard binning。

### Task 4.2.4：定点化

仿真：

* 8-bit syndrome；
* 10-bit syndrome；
* 12-bit syndrome；
* 8/12/16-bit LLR；
* LUT size vs LER。

**完成标准：** fixed-point decoder 的 LER 与 float decoder 相差在可接受范围内，并给出资源估计。

**若证否：**
如果 fixed-point 损失太大：

1. 增加 syndrome address bits；
2. 对边界附近使用高精度子 LUT；
3. 对 (|\Lambda|) 小的 ambiguous region 使用 repeat-syndrome；
4. 慢回路只更新少量阈值，避免大 LUT；
5. 改为 piecewise-quadratic likelihood approximation。

---

## Milestone 4.3：双回路闭环

### Task 4.3.1：定义更新频率

慢回路不需要每轮更新。定义：

[
\hat{\theta}_{t+K}
==================

g_{\phi}(s_{t-W:t})
]

每 (K) 轮更新一次 FPGA LUT 或阈值。

### Task 4.3.2：定义 hysteresis

避免参数抖动：

[
\theta^{\rm FPGA}_{t+1}
=======================

(1-\beta)\theta^{\rm FPGA}_t+\beta\hat{\theta}_t
]

并设置 update threshold。

### Task 4.3.3：闭环稳定性测试

测试 rapid drift、burst drift、OOD drift 下是否因错误更新导致性能恶化。

**完成标准：** 双回路在 drift 场景下优于 static MAP，并且在 iid Gaussian 场景下不明显劣于 static/oracle MAP。

**若证否：**
如果双回路偶发灾难性失效，主张转为“安全自适应解码”：只在 confidence 足够高时更新；否则退回静态 MAP。tail risk 比平均 LER 更重要。

---

# 6. Phase 5：核心优势验证

这一阶段决定论文能投到什么档次。

---

## Milestone 5.1：LER 优势验证

### 任务设计

比较：

1. no correction；
2. standard binning；
3. static MAP；
4. memory-assisted Bayesian；
5. Kalman/EWMA adaptive MAP；
6. sliding-window adaptive MAP；
7. CNN slow-loop + FPGA fast-loop；
8. oracle MAP。

噪声场景至少包括：

| 场景                        | 目的                       |
| ------------------------- | ------------------------ |
| iid isotropic Gaussian    | sanity check，证明你的方法不退化   |
| anisotropic Gaussian      | 证明 (q/p) bias adaptation |
| variance drift            | 核心漂移场景                   |
| mean drift                | 校准 offset 适应             |
| correlated (q,p) noise    | 证明二维 likelihood 的必要性     |
| burst/outlier             | tail-risk 场景             |
| pure loss + displacement  | 真实 bosonic 场景            |
| measurement/ancilla drift | 实验相关场景                   |

### 建议主指标

[
\overline{P_L}
==============

\frac{1}{T}\sum_t P_L(t)
]

[
P_L^{95%}
=========

\text{windowed LER 的 95 分位}
]

[
G_{\rm oracle}
==============

\frac{
\overline{P_L^{\rm dual}}-\overline{P_L^{\rm oracle}}
}{
\overline{P_L^{\rm static}}-\overline{P_L^{\rm oracle}}
}
]

### 成功标准

强结果：

* drift regime 下 (\overline{P_L}) 相比 static MAP 明显降低；
* (P_L^{95%}) 或 worst-window LER 明显降低；
* oracle gap closure 明显；
* iid Gaussian 下不明显输给 MAP。

### 若证否

如果 LER 没优势：

1. 检查 oracle gap 是否足够大；
2. 换成更真实的 measurement drift / offset drift；
3. 改用 confidence-weighted repeat syndrome；
4. 用 CNN 做 noise-state classifier，而非连续参数估计；
5. 将主指标从 average LER 改成 tail LER 或 adaptation lag；
6. 如果所有场景都无优势，项目应转型为“硬件低延迟 finite-energy MAP-LUT decoder”，不再主打 CNN。

---

## Milestone 5.2：逻辑保真度与逻辑通道优势

### Task 5.2.1：逻辑通道重构

对输入 (|0_L\rangle,|1_L\rangle,|+_L\rangle,|-_L\rangle,|+i_L\rangle,|-i_L\rangle) 运行纠错，估计 PTM：

[
R_{ij}
======

\frac{1}{2}\mathrm{Tr}[
P_i\mathcal{L}(P_j)
].
]

### Task 5.2.2：报告 fidelity

[
F_{\rm avg}
===========

\frac{dF_e+1}{d+1}
\quad d=2.
]

### Task 5.2.3：检查 non-Pauli 成分

尤其在 pure loss、finite-energy envelope、active correction imperfection 下，逻辑通道可能不是简单 Pauli channel。Harris 等关于 heralded/pure loss 的 GKP logical channel 分析非常适合作为对照。([arXiv][11])

### 成功标准

双回路不仅降低 (p_X,p_Z)，也改善 (F_{\rm avg}) 或减少 PTM 中不希望的偏置/非对角成分。

### 若证否

如果 LER 改善但 fidelity 不改善，说明 decoder 可能只是改变 Pauli 错误分布，未改善完整逻辑通道。替代方向：

1. 损失函数直接优化 (F_{\rm avg})；
2. 输出 soft correction 而非 hard binning；
3. 把 active displacement 成本纳入 objective；
4. 报告“LER 优化 decoder”和“fidelity 优化 decoder”两类模式。

---

## Milestone 5.3：复杂噪声适应性优势

### Task 5.3.1：OOD 测试

训练集不包含某些 drift family，测试集加入：

* abrupt jump；
* burst outlier；
* heavy-tailed displacement；
* correlated measurement error；
* slow phase rotation。

### Task 5.3.2：uncertainty-gated fallback

当 CNN uncertainty 高时，自动 fallback。

### Task 5.3.3：tail-risk 图

画：

[
\Pr(P_L^{\rm window}<x)
]

或者 rolling LER CDF。

### 成功标准

即使平均 LER 改善有限，tail-window LER 和 catastrophic failure probability 下降，也足以形成强叙事。

### 若证否

如果 OOD 下 CNN 失效：

1. 使用 domain randomization；
2. 加入 ensemble CNN；
3. 加入 conformal prediction；
4. fallback 更保守；
5. 论文中诚实限定适用范围：slow drift / bounded drift。

---

## Milestone 5.4：延迟与硬件效率优势

### Task 5.4.1：软件浮点 vs fixed-point

报告 float decoder、quantized decoder、LUT decoder 的性能差距。

### Task 5.4.2：资源模型

报告：

* LUT；
* BRAM；
* DSP；
* FF；
* maximum clock；
* pipeline depth；
* worst-case latency；
* throughput；
* update bandwidth。

### Task 5.4.3：与 NN/FPGA QEC 文献对齐

2026 年 FPGA-based NN surface-code decoder 的 550 ns closed-loop latency 和 124 ns NN decoding 是很好的参照点；你不需要在单模 GKP 上照抄其结构，但需要说明你的 latency budget 为什么足够进入 GKP correction cycle。([arXiv][9])

### 成功标准

快回路 latency 小于目标 QEC cycle 的安全余量；固定点性能损失小；LUT/BRAM 使用可控。

### 若证否

如果 latency 太高：

1. CNN 不进快回路，只低频更新 LUT；
2. 快回路只做 threshold comparator；
3. LLR LUT 分区；
4. 多周期 pipeline；
5. active correction 改为 Pauli frame update；
6. 只对 ambiguous syndrome 区域调用复杂逻辑。

---

## Milestone 5.5：纠错成本与 pseudo-threshold

### Task 5.5.1：定义 operational pseudo-threshold

单模 GKP 不应声称 surface-code 式 threshold。建议定义：

[
P_L^{\rm QEC}(\sigma^\star)

===========================

P_L^{\rm noQEC}(\sigma^\star)
]

或：

[
F_{\rm avg}^{\rm QEC}(\sigma^\star)
===================================

F_{\rm avg}^{\rm noQEC}(\sigma^\star).
]

### Task 5.5.2：定义 break-even boundary

如果模拟 quantum memory，则定义 logical lifetime：

[
T_L^{\rm corrected}

>

T_L^{\rm uncorrected}.
]

已有 GKP 相关实验已经用 beyond-break-even gain 作为重要指标，例如实时 QEC 和 GKP qudit QEC。([Inspire][12])

### Task 5.5.3：定义真实纠错成本

建议指标：

[
C_{\rm logical}
===============

\frac{
N_{\rm syndrome}
+
\alpha N_{\rm ancilla}
+
\beta N_{\rm active\ displacement}
+
\gamma E_{\rm classical}
}{
-\log_{10} P_L
}.
]

### 成功标准

在相同目标 LER 或 fidelity 下，双回路减少 syndrome repetition、active correction 次数、postselection 或 squeezing requirement。

### 若证否

如果成本没有降低，但 LER 降低，可以主打性能；如果性能和成本都无优势，只能保留“硬件低延迟可部署”贡献，期刊定位要下降。

---

# 7. Phase 6：论文初稿先行

你希望先完成论文初稿，再补真实 FPGA 结果。这个顺序是合理的，但初稿必须像一篇完整论文，而不是“等 FPGA 后再填坑”。

---

## Milestone 6.1：锁定主图

建议初稿至少有 8 张主图：

1. **系统架构图**：GKP syndrome → FPGA fast decoder；syndrome history → CNN slow estimator → LUT/threshold update。
2. **GKP 解码几何图**：standard binning vs finite-energy-aware MAP boundary。
3. **仿真流程图**：syndrome-level、finite-energy effective、Fock-space validation 三层。
4. **静态噪声 sanity check**：dual-loop 不输 static MAP。
5. **漂移噪声主结果**：rolling LER、oracle gap。
6. **复杂噪声 robustness**：tail LER CDF 或 boxplot。
7. **逻辑通道结果**：PTM / (F_{\rm avg}) / (F_e)。
8. **硬件感知结果**：latency–resource–LER tradeoff。

---

## Milestone 6.2：写作结构

建议论文结构：

**I. Introduction**
GKP 的价值、有限能量与漂移问题、现有 decoder 与硬件实时 QEC 的 gap、你的贡献。

**II. Approximate GKP decoding model**
理想 GKP、有限能量、syndrome extraction、噪声模型、logical metrics。

**III. Dual-loop adaptive decoder**
CNN 慢回路、FPGA 快回路、LUT/MAP、fallback、fixed-point。

**IV. Realistic simulation framework**
三层仿真：syndrome-level、effective finite-energy、Fock-space validation、hardware timing。

**V. Results**
LER、fidelity、oracle gap、complex noise、latency/cost。

**VI. Discussion**
适用范围、与 MAP/Bayesian/analog-info/FPGA NN 的关系、限制、真实实验路径。

**VII. Conclusion**

**Supplementary**
公式推导、baseline details、hyperparameters、statistical confidence、additional ablation、full noise tables、hardware estimates。

---

## Milestone 6.3：审稿风险预处理

### 可能被问 1：为什么不用精确 MAP？

回答：精确 MAP 在噪声参数已知时是 oracle；真实系统中参数漂移，static MAP 失配。你的贡献是缩小 static MAP 与 oracle MAP 的 gap，而不是声称超越 oracle。

### 可能被问 2：CNN 是否只是过拟合模拟器？

应对：

* held-out drift families；
* OOD tests；
* Fock-space validation；
* real experimental parameter ranges；
* uncertainty-gated fallback；
* baseline includes Kalman/EWMA/sliding window。

### 可能被问 3：单模 GKP 哪来的 threshold？

应对：不用 surface-code threshold 语言，改用 operational pseudo-threshold、break-even boundary、logical lifetime gain。

### 可能被问 4：FPGA 结果还没做，为什么谈硬件？

初稿中只谈 hardware-aware simulation 和 fixed-point synthesis estimate；真实 FPGA 作为最终补充实验，不把未完成内容写成已完成结果。

---

# 8. Phase 7：真实 FPGA 实验补充

这个阶段放在论文初稿之后。

## Milestone 7.1：FPGA fast-loop 原型

### Task 7.1.1：实现 fixed-point LUT-MAP

用 HDL/HLS 实现：

* syndrome input；
* address mapping；
* LLR LUT；
* correction decision；
* confidence output；
* parameter update interface。

### Task 7.1.2：testbench 对齐 Python

逐 shot 比较 Python float、Python fixed-point、FPGA output。

### Task 7.1.3：资源与时延测量

报告真实：

* clock frequency；
* pipeline cycles；
* worst-case latency；
* BRAM/DSP/LUT/FF；
* update latency；
* throughput。

---

## Milestone 7.2：CNN 慢回路部署路径

两种路线：

**路线 A：CNN 在 host/GPU/CPU 上慢速运行，更新 FPGA LUT。**
最稳，适合第一篇论文。

**路线 B：量化 CNN 也部署在 FPGA 上，但不进入每轮 critical path。**
更完整，但工作量和风险更大。

**建议：** 第一篇主打路线 A。这样“FPGA 快回路”真实可测，“CNN 慢回路”仍可作为边缘端/主机端在线估计器存在。

---

## Milestone 7.3：hardware-in-the-loop

### Task 7.3.1：PC 生成 synthetic syndrome stream

PC/GPU 按真实 drift model 生成 syndrome，送入 FPGA。

### Task 7.3.2：FPGA 实时输出 correction

记录每 shot：

[
(s_t,\hat{\theta}_t,\Lambda_t,\hat{a}*t,L*{\rm decode})
]

### Task 7.3.3：与 oracle 离线结果对齐

比较 FPGA 输出与 Python reference。

**完成标准：** FPGA-in-the-loop 的 LER 与 Python fixed-point 仿真一致，latency 满足目标。

---

## Milestone 7.4：真实 GKP 数据或量子硬件接入

如果能接真实实验，优先做**离线 re-decoding**，再做闭环：

1. 收集 GKP syndrome history；
2. 估计真实 drift；
3. 离线比较 existing decoder 与你的 decoder；
4. 再把 FPGA 接入控制链路；
5. 先做 Pauli frame update；
6. 最后做 active displacement feedback。

若没有真实量子硬件，FPGA-in-the-loop + 公开/合作数据 re-decoding 仍然可作为强工程补充。

---

# 9. 项目推进时的“单 task 规则”

你可以按下面方式管理每次工作：

每个 task 必须有：

1. **输入**：使用哪份数据/公式/代码；
2. **输出**：一个图、一个表、一个测试、一个模块或一段文字；
3. **通过标准**：数值误差、性能提升、CI 覆盖、baseline 对齐；
4. **失败分支**：若不通过，下一步改什么；
5. **记录**：写入 lab notebook。

推荐 task 粒度：

* 一个 task 只改一个模块；
* 一个 task 只回答一个科学问题；
* 一个 task 只生成一张核心图或一个验证表；
* 不把“训练 CNN + 改仿真器 + 调 baseline + 写论文”混成一个 task。

---

# 10. 总里程碑—优势—证否分支总表

| 优势主张           | 对应 milestone | 证实标准                                                           | 若证否，替代方向                                                |
| -------------- | ------------ | -------------------------------------------------------------- | ------------------------------------------------------- |
| Drift 下 LER 降低 | M5.1         | dual-loop 显著低于 static MAP / Bayesian，接近 oracle                 | 改用 Kalman+CNN residual；聚焦 tail LER；加入 measurement drift |
| Fidelity 改善    | M5.2         | (F_{\rm avg})、(F_e)、PTM 优于 baseline                            | 用 fidelity-aware loss；改 soft correction；主打 LER 而非完整通道   |
| 复杂噪声适应         | M5.3         | OOD/burst 下 fallback 降低 catastrophic failure                   | 加 domain randomization；ensemble uncertainty；限定适用范围      |
| 低延迟            | M5.4 / M7.1  | fixed-point LUT/FPGA 满足 QEC cycle                              | CNN 不进快回路；LUT 分层；frame update 代替 active correction      |
| 成本下降           | M5.5         | 同等 LER 下减少 syndrome repeats / active pulses / squeezing demand | 保留性能提升 claim；成本只作为不显著增加                                 |
| 论文可投性          | M6           | 初稿已有完整主图、强 baseline、统计置信区间                                     | 降目标期刊；补 Fock-space 或 FPGA-in-loop 后再投                   |

---

# 11. 推荐参考文献池

下面按用途分组。不是“所有 GKP 文献”的绝对全集，但覆盖你这个项目需要的主要参考链；后续可用这些文献做 forward/backward citation expansion。

## A. GKP 基础、综述、有限能量理论

1. Gottesman, Kitaev, Preskill, **Encoding a qubit in an oscillator**, 2001. GKP 码原始论文。([arXiv][1])
2. Glancy, Knill, **Error Analysis for Encoding a Qubit in an Oscillator**, 2006. 早期误差分析。([arXiv][13])
3. Terhal, **Quantum error correction for quantum memories**, Rev. Mod. Phys. 2015.
4. Grimsmo, Puri, **Quantum Error Correction with the Gottesman-Kitaev-Preskill Code**, PRX Quantum 2021.([APS Link][14])
5. Brady, Eickbusch, Singh, Wu, Zhuang, **Advances in bosonic quantum error correction with GKP codes**, 2024 review.([arXiv][5])
6. Conrad, Eisert, Arzani, **Gottesman-Kitaev-Preskill codes: A lattice perspective**, Quantum 2022.([Quantum][15])
7. Royer, Singh, Girvin, **Stabilization of Finite-Energy Gottesman-Kitaev-Preskill States**, PRL 2020.([arXiv][16])
8. Walshe, Baragiola, Alexander, Menicucci, **Continuous-variable gate teleportation and bosonic-code error correction**, 2020.
9. Error Correction Zoo, **Square-lattice GKP code / multimode GKP code**，用于快速查稳定子、相关码族和参考链。([错误纠正动物园][17])

## B. GKP 本体解码、finite-energy、loss、MAP/Bayesian

10. Fukui, Tomita, Okamoto, **Analog quantum error correction with encoding a qubit into an oscillator**, 2017.([arXiv][4])
11. Wan, Neville, Kolthammer, **Memory-assisted decoder for approximate GKP codes**, PRR 2020.([arXiv][10])
12. Jafarzadeh, Conrad, Alexander, Baragiola, **Logical channels in approximate GKP error correction**, 2025.([arXiv][2])
13. Marqversen, Wesenberg, Zinner, Andersen, **Performance analysis of GKP error correction**, 2025.([arXiv][7])
14. Hastrup, Andersen, **Analysis of loss correction with the GKP code**, PRA 2023.([Welcome to DTU Research Database][8])
15. Harris et al., **Logical channel for heralded and pure loss with the GKP code**, 2025.([arXiv][11])
16. Zheng, He, Lee, Noh, Jiang, **Performance and achievable rates of the GKP code for pure-loss and amplification channels**, PRX Quantum 2025.
17. Lin, Chamberland, Noh, **Closest lattice point decoding for multimode GKP codes**, 2023.
18. Lin, Noh, **Exploring the quantum capacity of a Gaussian random-displacement channel using GKP codes and maximum-likelihood decoding**, 2025.([Amazon Science][18])
19. Xu, Wang, Kuo, Albert, **Qubit-Oscillator Concatenated Codes: Decoding Formalism & Code Comparison**, 2023.
20. Ralph, **Noise Transfer Approach to GKP Quantum Circuits**, 2024.([MDPI][19])
21. Wakaura, Tanimae, **QIFE: a calibration-free decoder for finite-energy GKP qubits**, 2026 preprint. 这篇只建议作为近期方向参考，正式使用前要独立核查。([Research Square][20])

## C. GKP 实验与平台参数来源

22. Flühmann et al., **Encoding a qubit in a trapped-ion mechanical oscillator**, Nature 2019.([arXiv][21])
23. Campagne-Ibarcq et al., **Quantum error correction of a qubit encoded in grid states of an oscillator**, Nature 2020.([arXiv][3])
24. de Neeve et al., **Error correction of a logical grid state qubit by dissipative pumping**, Nature Physics 2022.([Nature][22])
25. Sivak et al., **Real-time quantum error correction beyond break-even**, Nature 2023.([Inspire][12])
26. Lachance-Quirion et al., **Autonomous quantum error correction of GKP states**, PRL 2024.([APS Link][23])
27. Konno et al., **Logical states for fault-tolerant quantum computation with propagating light**, Science 2024.([科学组织][24])
28. Matsos et al., **Robust and Deterministic Preparation of Bosonic Logical States in a Trapped Ion**, PRL 2024.([arXiv][25])
29. Matsos et al., **Universal quantum gate set for GKP logical qubits**, Nature Physics 2025.([Nature][26])
30. Brock et al., **Quantum error correction of qudits beyond break-even**, Nature 2025.([Nature][27])
31. Larsen et al., **Integrated photonic source of GKP qubits**, Nature 2025.([Nature][28])
32. Dahan et al., **Creation of Optical Cat and GKP States Using Shaped Free Electrons**, PRX 2023.([天文数据系统][29])
33. Hastrup et al., **Protocol for generating optical GKP states**, PRL 2022.([Welcome to DTU Research Database][30])

## D. GKP 仿真与 bosonic simulation

34. Hopfmueller et al., **Bosonic Pauli+: Efficient Simulation of Concatenated GKP Codes**, Quantum 2024.([arXiv][31])
35. The Walrus / Strawberry Fields GKP state tutorials，用于光学近似 GKP 态数值生成参考。([Walrus 文档][32])
36. QuTiP, QuantumOptics.jl 等 Fock-space simulation 工具文献。
37. Stim / PyMatching 主要用于离散码，不是本项目主线，但可借鉴大规模 Monte Carlo 和 decoder benchmarking 的工程方式。

## E. 神经网络解码与机器学习 QEC

38. Breuckmann, Ni, **Scalable Neural Network Decoders for Higher Dimensional Quantum Codes**, Quantum 2018.([Quantum][33])
39. Fitzek et al., **Deep Q-learning decoder for depolarizing noise on the toric code**, 2019.([arXiv][34])
40. Gicev, Hollenberg, Usman, **A scalable and fast artificial neural network syndrome decoder for surface codes**, Quantum 2023.([Quantum][35])
41. Bausch et al., **Learning high-accuracy error decoding for quantum processors / AlphaQubit**, Nature 2024.([Nature][36])
42. Hall et al., **Artificial neural network syndrome decoding on IBM quantum processors**, PRR 2024.([APS Link][37])
43. Varbanov et al., **Neural network decoder for near-term surface-code experiments**, PRR 2025.([APS Link][38])
44. Lange et al., **Data-driven decoding of quantum error correcting codes using graph neural networks**, PRR 2025.([APS Link][39])
45. Wang et al., **Multidimensional Bose quantum error correction based on GKP code**, npj Quantum Information 2022. 这篇涉及 surface-GKP 与神经网络解码，但不是单模 GKP 本体。([Nature][40])
46. Zeng et al., **Neural-Network-Based Design of Approximate GKP Code**, PRL 2025. 这篇是神经网络设计 GKP 态，不是实时解码，但与近似 GKP 和 NN 很相关。([APS Link][41])
47. Biamonte et al. / recent reviews on AI for QEC，可用于综述背景。([arXiv][42])

## F. 自适应噪声估计、noise-aware decoding、漂移建模

48. Fujiwara, **Instantaneous Quantum Channel Estimation during Quantum Information Processing**, 2014.([arXiv][43])
49. Nickerson et al., **Analysing correlated noise on the surface code using adaptive decoding**, Quantum 2019.([Quantum][44])
50. Wagner et al., **Optimal noise estimation from syndrome statistics of quantum codes**, 2020.([arXiv][45])
51. Sivak et al., **Optimization of Decoder Priors for Accurate Quantum Error Correction**, PRL 2024.([APS Link][46])
52. Bhardwaj, Takou, Lin, Brown, **Adaptive Estimation of Drifting Noise in Quantum Error Correction**, 2025.([arXiv][6])
53. **Improving error suppression with noise-aware decoding**, 2025.([arXiv][47])
54. **Differentiable Maximum Likelihood Noise Estimation for Quantum Error Correction**, 2026 preprint.([arXiv][48])

## G. FPGA / 实时 QEC / 硬件解码

55. Yang et al., **Real-time Surface-Code Error Correction Using an FPGA-based Neural-Network Decoder**, 2026.([arXiv][9])
56. Huo, **FPGA-Accelerated Early-Exit Neural Decoder for Quantum Error Correction**, 2025.([IEEE Xplore][49])
57. **QUEKUF: An FPGA Union Find Decoder for Quantum Error Correction**, 2025.([ACM Digital Library][50])
58. Riverlane local clustering / FPGA decoder reports，偏工程参考，正式论文中应谨慎引用。([Riverlane][51])
59. Skoric et al., **Parallel window decoding enables scalable fault-tolerant quantum computation**, Nat. Commun. 2023.
60. Tan et al., **Scalable surface-code decoders with parallelization in time**, PRX Quantum 2022.
61. Fowler et al., surface-code MWPM / real-time decoding 系列文献，用于硬件解码背景。

## H. 相关但非主线：GKP 与外层码

这些不是你论文主体，但 Introduction 和 Related Work 应简短覆盖，避免审稿人认为你忽略 GKP decoding 大背景：

62. Vuillot et al., **Quantum Error Correction with the Toric-GKP Code**, 2019.
63. Noh, Chamberland, **Fault-tolerant bosonic QEC with the surface-GKP code**, 2020.([IBM Research][52])
64. Noh, Chamberland, Brandão, **Low-overhead fault-tolerant QEC with the surface-GKP code**, 2022.([天文数据系统][53])
65. Fukui et al., **High-threshold fault-tolerant quantum computation with GKP qubit and analog information**, 2018.
66. Zhang et al., **Color-GKP code**, 2021.([APS Link][54])
67. Raveendran et al., **Finite Rate QLDPC-GKP Coding Scheme that Surpasses the CSS Hamming Bound**, Quantum 2022.([Quantum][55])
68. Berent et al., **Analog Information Decoding of Bosonic QLDPC Codes**, PRX Quantum 2024.
69. Borah et al., **Fault Tolerant Decoding of QLDPC-GKP Codes with Circuit Level Soft Information**, 2025.([arXiv][56])

---

# 12. 最小可发表版本与增强版本

## 最小可发表版本

只做：

1. single-mode square approximate GKP；
2. drifted Gaussian displacement + measurement noise；
3. standard binning / static MAP / oracle MAP / Bayesian / Kalman baseline；
4. CNN 输出噪声参数；
5. FPGA 快回路用 LUT-MAP；
6. 报 LER、oracle gap、tail LER、fixed-point latency/resource estimate；
7. Fock-space 验证一小组参数。

这可以形成一篇严肃的理论+仿真+硬件感知论文。

## 高质量增强版本

再加：

1. pure loss / thermal loss；
2. finite-energy logical channel；
3. (F_{\rm avg})、(F_e)、PTM；
4. burst/OOD noise；
5. hardware-in-the-loop；
6. 真实 FPGA latency；
7. 离线真实 GKP syndrome 数据 re-decoding。

这个版本才更接近 PRX Quantum / Quantum / npj Quantum Information 级别的叙事强度。

---

# 13. 最终建议

第一篇论文的核心贡献应压在：

> **drift-aware finite-energy GKP decoding + hardware-realistic dual-loop implementation**

而不是压在：

> **CNN 比所有 GKP 解码器都更强**

后者风险太高，因为静态已知噪声下 MAP/oracle 本来就接近最优。前者更符合真实实验痛点，也更容易通过扎实仿真和 FPGA 补实验建立可信度。

---

# 14. 2026-07-12 文献核查后的执行修订

本节是对原始实验计划的低频修订，不删除前述历史规划。触发原因是对以下三份本地材料的全文、补充材料和实验图片核查表明，原计划的协议参考、非平稳噪声叙事、FPGA 阶段顺序和论文证据门槛需要实质调整：

- `docs/relative_papers/Quantum_error_correction_of_a_qubit_encoded_in_grid_states_of_an_oscillator.md`；
- `docs/relative_papers/Advances_in_Bosonic_Quantum_Error_Correction_with_Gottesman–Kitaev–Preskill_Codes_Theory_Engineering_and_Applications.md`；
- `docs/relative_papers/Real-time_quantum_error_correction_beyond_break-even.md`。

## 14.1 修订后的研究边界

- 主物理对象仍是 repeated quantum-memory QEC 下的 single-mode square approximate GKP qubit；已完成的 Phase 0–1 结论继续有效。
- 实验式数字孪生以 sBs 低秩耗散协议为主参考，以 2020 年 sharpen–trim/measurement-feedback 协议为交叉验证参考；不再只用 Steane/Knill 信息流代表真实超导 GKP 实验。
- 系统采用“两个计算域、三个时间尺度”：FPGA 执行逐周期确定性路径和事件计数，主机执行窗口级状态估计及更慢的参数优化/重标定。
- 非平稳噪声采用连续漂移与离散故障状态的混合模型，至少区分 normal、large-error recovery、burst、leakage 和 calibration-shift regime。
- 约 300 元纯数字 FPGA 开发板只用于验证经典控制平面、定点解码、事件监测和 HIL；不得声称生成微波、采集真实量子读出或完成真实 GKP 量子纠错。

## 14.2 修订后的阶段顺序

1. 保留并完成 Phase 1 当前理论任务；
2. 增加 claim/board/protocol contract gate；
3. 建立 sBs-first、sharpen–trim-secondary 的实验式数字孪生；
4. 完成多保真仿真、强 baseline 和三时间尺度控制器；
5. 通过协议复现、因果故障注入、算法、逻辑通道和鲁棒性证据门；
6. 在冻结论文主图前完成低成本 FPGA 实测和 HIL；
7. 最后完成论文、审稿风险预处理和可复现发布；
8. 真实 GKP syndrome 数据或量子硬件接入保留为可选增强阶段。

`docs/new_task_board.md` 是本修订后的唯一当前执行顺序和状态源；本文件前述 Phase 6/7 顺序保留为历史规划，不再作为当前排程依据。

## 14.3 必须满足的证据门

- **协议可信度**：复现 sBs/sharpen–trim 的方向性 syndrome、恢复和时序趋势，不强行拟合实验论文的峰值增益。
- **因果可信度**：分别注入 displacement、ancilla bit/phase flip、readout error 和 leakage，并验证预期 syndrome/recovery signature。
- **算法可信度**：与 static/oracle MAP、memory-assisted Bayesian、EWMA/Kalman、sliding-window、run-length/HMM/change-point baseline 比较。
- **逻辑通道可信度**：报告 Pauli lifetimes、PTM、`F_avg`、`F_e` 和短时有效退极化率；仅将模型结果称为 simulation-derived coherence gain。
- **硬件可信度**：报告真实开发板的 bit-accurate equivalence、worst-case latency、jitter、deadline miss、Fmax 和 LUT/FF/BRAM/DSP。
- **长序列可信度**：使用至少 `1e5` cycles、目标 `1e6` cycles 的 HIL replay 验证稀有 leakage、计数器饱和、参数更新和通信扰动。

## 14.4 论文主张边界

第一篇论文的可接受主张是：实验协议启发的 syndrome-history-aware GKP 经典控制架构，在主机侧估计连续漂移和离散健康状态，在低成本 FPGA 上执行确定性、定点化、run-length-aware 的 MAP-LUT 控制，并通过多保真仿真、强 baseline、因果故障注入、真实板卡测量和 HIL 建立证据。

禁止将以下内容写成已完成结果：真实 beyond-break-even、真实微波/ADC 控制、真实 cavity/transmon 接入、板上 PPO/CNN 训练、CNN 超越 oracle MAP、以 post-selection 提升冒充在线纠错增益。

---

# 15. 2026-07-13 Non-Markovian feedback 文献补强

本节记录对 `docs/relative_papers/Non-Markovian_feedback_for_optimized_quantum_error_correction.md` 正文、Supplemental Material 和 23 张图片核查后，对第 14 节执行修订的定向补强。该文是 model-based numerical study，不是真实 cavity/transmon 实验；其主要价值是为“measurement-history belief state -> adaptive sBs control”提供直接近邻和可复现 benchmark。

## 15.1 新增术语与范围边界

- **decoder oracle**：知道真实 noise/drift state 的不可部署 MAP 上界，继续沿用 T1.3.2 语义。
- **control oracle**：在有限 measurement trajectory horizon 上，为每条 history 独立优化 sBs 控制参数的 lookup-table 上界；不得与 decoder oracle 混称。
- **teacher**：在可微物理模型上用 Feedback-GRAPE 或等价方法训练的 RNN/GRU 控制策略。
- **student**：由 teacher 蒸馏得到的低维指数递推、有限状态或 parameter-bank 控制器，目标是定点化并映射到低成本 FPGA。
- 主机慢回路继续负责 drift/regime 估计；逐周期 student 只保留短时 recovery/history memory。项目由此保持两个计算域、三个时间尺度，而不是把完整 RNN 直接放入逐周期关键路径。

## 15.2 Feedback-GRAPE 可行性门

进入 teacher 训练前必须满足：

1. 有限 cutoff 的 cavity-ancilla trajectory simulator 可微；
2. 梯度同时包含 reward path 和 trajectory-probability log-likelihood path；
3. 自动微分与 finite difference 在小模型上对齐；
4. standard sBs、memoryless Markovian feedback 和 non-Markovian teacher 的方向性 ranking 可复现；
5. cutoff、batch、trajectory horizon 的运行时间和显存成本已量化。

若该门未通过，保留现有 v2 drift-aware MAP/事件控制主线；将 PRL 的指数递推策略作为强 baseline，不声称完成 teacher distillation。

## 15.3 新增公平 baseline 和证据要求

- 增加 latest-outcome FNN/Markovian feedback，以隔离 memory 的贡献。
- 增加 autonomous sBs，并按物理时间、measurement/reset 次数和控制成本公平比较，不只按 cycle 数。
- 增加有限时域 control oracle，以及 handcrafted exponential recurrence。
- 训练不得只报告 best-of-N agent；必须报告所有 seeds、validation-based selection、独立 test 和 median/IQR/worst-quartile。
- 必须检查短训练 horizon 到 `1e3`、`1e5` 及目标 `1e6` cycles 的 hidden-state 有界性和性能外推。
- 六个 Pauli eigenstates、average channel fidelity、logical lifetime、`p(g)`、e/leakage burden、parameter slew 和 fallback cost 均进入评价。
- gate bias 不得只测单一固定向量；需覆盖随机 bias family、readout confusion、leakage/reset failure、dephasing、drift 和未见 dynamics。

## 15.4 Teacher-to-student 硬件路线

完整 PRL-style RNN 仅作为 teacher 或 optional deployment candidate。第一篇论文的默认硬件主线是蒸馏递推 student，例如：

$$
\pi_{t+1}=a_{m_t}\pi_t+(1-a_{m_t})\pi_{m_t}^{\infty},
\qquad m_t\in\{g,e,\mathrm{leak}\}.
$$

student 必须报告：teacher gain retention、状态维数、固定点位宽、乘加次数、worst-case latency、LUT/FF/BRAM/DSP、long-horizon stability 和 bit-accurate HIL。若完整量化 GRU 经 synthesis 证明可装入目标板且满足 deadline，可作为增强对照，但不取代 student 主线。

## 15.5 补强后的条件性论文主张

强版本主张：仿真中的 model-aware recurrent teacher 学到可解释的 history-dependent sBs feedback；蒸馏 student 在 leakage/drift/model mismatch 下保留大部分 simulated lifetime/channel-fidelity gain，并在低成本 FPGA 上达到 bit-accurate、确定性 deadline。

证否版本主张：若 teacher 不可复现、蒸馏损失过大或完整闭环不稳定，则删除 NMF 性能主张，保留 PRL-inspired recurrence 作为强 baseline，回到 experiment-informed drift/regime-aware MAP-LUT + FPGA co-design 主线。

---

# 16. 2026-07-13 六篇补充论文后的 v2.2 小幅补强

本节记录对六篇补充论文的标题、摘要、实验/数值部分、结论和关键结果图核查后，对第 14—15 节执行路线的定向补强：

- [`Error Correction of Beamsplitter-Generated Entangled GKP States`][57]（arXiv:2605.08009）；
- [`Optimized Gottesman-Kitaev-Preskill Error Correction via Tunable Preprocessing`][58]（arXiv:2604.08247）；
- [`Noise Transfer Approach to GKP Quantum Circuits`][19]（arXiv:2411.05262）；
- [`Performance analysis of GKP error correction`][7]（arXiv:2505.14775）；
- [`The Near-optimal Performance of Quantum Error Correction Codes`][59]（arXiv:2401.02022）；
- [`Approximate maximum-likelihood decoding with K minimum weight matchings`][60]（arXiv:2510.06531）。

v2.2 不改变 phase 顺序，也不扩大 single-mode 主范围；只补充三个能直接增强本项目证据链的执行对象，并将其余结果限制为 secondary evidence。

## 16.1 三项主线补强

### 16.1.1 Heisenberg noise-transfer 中保真度代理

在 syndrome-level 与 Fock/density-matrix model 之间增加解析/半解析代理，将每个 quadrature 分解为 lattice signal 与 fluctuation：

$$
\hat q=\hat q_c+\delta\hat q,
\qquad
\hat p=\hat p_c+\delta\hat p.
$$

该代理传播 covariance、loss、measurement efficiency、feedforward gain 和离散 domain-misidentification/logical jump，用于组件级 noise budget、快速参数扫描和解析回归。它不是高保真物理模型：约 `10 dB` 及以上 squeezing 的 state-independent 区域应与 Fock/effective model 对齐；低 squeezing、domain clipping 或明显 state-dependent fluctuation 必须形成显式失败案例。

### 16.1.2 QEC-matrix/Petz channel-recovery bound

在 decoder oracle 和 finite-horizon control oracle 之外，增加第三类不可部署参考：基于 QEC matrix、transpose/Petz recovery 得到的 near-optimal channel fidelity 双边界。其作用是估计给定 encoding + noise 在允许任意 recovery 时的潜在性能，而不是给出本项目实际 sBs decoder/controller。

执行时先在小 cutoff 上与 SDP optimal recovery 校验，再评估是否能扩展到更高能量或更大 cutoff。论文报告实际 sBs、teacher/student 与该 bound 的 gap；不得把 bound 写成板上 decoder、oracle MAP 或真实可实现控制器的性能。

### 16.1.3 top-K lattice-coset truncated MAP

只借鉴 K-MWM 的“以 K 调节精度—计算量”思想，不实现 surface-code matching graph。对 single-mode periodic MAP 的每个逻辑陪集保留最可能的 K 个 lattice aliases：

$$
\log P_K(\ell\mid s)
=\operatorname{logsumexp}_{n\in\mathcal C_\ell^{(K)}}
\log p(n,s),
\qquad \ell\in\{0,1\}.
$$

扫描 `K=1` 到收敛区，比较 full periodic MAP 的 LLR/LER 误差以及 fixed-point latency、存储、乘加和并行资源。该方法命名为 top-K lattice-coset MAP，不称为 K-MWM，也不引入外层 surface-GKP。

## 16.2 Secondary protocol 与实验叙事证据

- Knill/qunaught、standard Bell-resource Knill、Steane/ME-Steane 和 P-Steane 进入 secondary reproduction table。重点核查协议等价、post-correction squeezing/displacement 趋势，以及 data/ancilla noise ratio 改变时的 noise-shaping 方向；不进入 sBs 主排名。
- P-Steane 的 `(a,b)` 是量子辅助态/逻辑门 preprocessing 参数。低成本 FPGA 最多选择离线生成的 parameter bank，不能宣称开发板实现了物理 squeezing 或量子 preprocessing。
- 2026 trapped-ion 双模实验仅用于强化报告结构：逐 Pauli observable、QEC on/off、per-cycle 与 wall-clock lifetime、reset/recoil、并行控制和故障来源必须分开。其约 `500 us` QEC round 是平台事实，不是本项目 FPGA deadline。
- 多模 Bell-state、surface-GKP 和外层码不进入第一篇论文的主实验范围；若未来采用，只能走 Phase 8 或独立后续工作。

## 16.3 v2.2 证据门与降级路径

1. noise-transfer 代理未通过有效域交叉验证时，仅保留为解析诊断，不用于生成主结果；
2. Petz/QEC-matrix bound 不能在可控 cutoff 上复现双边界时，不进入论文定量 gap；
3. top-K 在目标板上没有形成有意义的精度—资源折中，则部署点回退现有 MAP-LUT；
4. Knill/P-Steane reproduction 失败不阻塞 sBs 主线，只在任务记录中保留负结果；
5. 主文贡献仍是 single-mode、experiment-informed classical control/decoding 与低成本 FPGA HIL，不升级为多模、surface-GKP 或真实量子控制实验。

---

# 17. 2026-07-17 Phase 6 真板依赖拆分

本节记录 Phase 6 执行依赖的低频修订。触发原因不是硬件结论已经升级，而是 T5.5 已经建立 bit-accurate Python contract、可综合 fast-path RTL、CXXRTL 逐周期对拍和固定目标器件的开源 P&R estimate；这些产物足以继续做板卡无关的 production RTL 与软件资格验证。若仍把 `T6.1.1` 设为整个 Phase 6 的全局前置条件，会无必要地推迟可复现的长序列和故障路径验证，并把“缺少实物板卡”与“缺少合法 RTL 被测对象”混为一谈。

## 17.1 拆分原则

- 外部依赖只阻塞确实需要实物的工作：板卡型号/版本/接口确认、bitstream 下载、真实 transport adapter、logic analyzer/GPIO 测量、板上功耗和真实 HIL。
- 不需要实物的工作继续推进：production RTL requirement mapping、状态机补强、独立 golden/CXXRTL 对拍、长序列递推、抽象 transport 故障模型、结构/功能覆盖和 mutation audit。
- 两类证据必须分栏保存。真板前资格验证不得写成 vendor signoff、bitstream、board measurement、实际 transport throughput/power 或真实 HIL；目标器件 P&R 仍只称 open-source target-device estimate。

## 17.2 两条依赖轨与汇合点

1. **板卡无关资格验证轨**：`T6.2.1 -> T6.2.2`。直接依赖 T5.5.1—T5.5.4，不依赖 T6.1.1；当前优先执行。
2. **实物板卡验证轨**：`T6.1.1 -> T6.1.2 -> T6.1.3`。T6.1.1 在实物信息缺失时保持 Blocked，阻塞只沿本轨传播。
3. **汇合点**：T6.2.3 同时要求 T6.2.2 和 T6.1.1—T6.1.3 完成，才允许形成 actual-board correctness smoke。T6.2.4 可在 T6.2.2 后按板卡无关路线继续，但不替代本轮 T6.2.1/T6.2.2 的短期主线。
4. **板上重复验证**：T6.4.1 与 T6.4.3 在 T6.2.3 后重跑长序列和 negative path；T6.2.2 的结果只作为可复现的预期基线，不计入板上样本量或板测通过率。

## 17.3 T6.2.1—T6.2.2 真板前验收

T6.2.1 不从 demo 重新起步，而是对 T5.5 的 `gkp_fast_path_core` 做 requirement-to-RTL 审计并补齐 production 缺口。至少覆盖 syndrome classification、MAP-LUT、run-length、Pauli/phase frame、trusted A/B bank、version/CRC、饱和、leakage、deadline/fallback 和 action output；硬编码 trace、删除状态路径、仅组合 LUT 或只观察 activity 的 harness 均不得作为完成证据。

T6.2.2 使用独立 Python golden 和 RTL/CXXRTL 对全部可见 output/state word 逐周期 bit-for-bit 对拍。除 normal/boundary 外，必须覆盖 saturation、leakage、CRC/version、stale/rollback/untrusted bank、reset、deadline、commit race，以及 FIFO overflow/backpressure、通信 pause/drop/duplicate/reorder 的抽象行为模型。每个 nominal/fault family 或已登记 stream 至少运行 `1e5` cycles，聚合不少于 `1e6` cycles；要求零未定义 action、零 silent overflow、状态有界，并交付功能分支覆盖矩阵、mutation/fault audit、Source Data 和可复现 runner。

## 17.4 证据升级条件

T6.2.1/T6.2.2 只能升级 software/RTL qualification 字段。只有 T6.1.1—T6.1.3 完成、同一 source-bound 设计生成并下载 bitstream，且实际板上回放与测量通过后，T6.2.3/T6.4 才能分别升级 board correctness、真实 transport、core/transport/end-to-end latency、长时 HIL、negative path 和可测 power 字段。软件仿真用例必须在板上重复，但不得把重复用例的预板结果复制成板测结果。

---

# 18. 2026-07-17 Route-A contract-centric 安全自适应双回路

本节记录用户确认的主叙事重构。T5.1.4 已证否 CNN/learned decoder 的强主分支，T5.4.2 又保留 calibration shift、compound 和 nominal fallback 的反例；因此第一篇高水平论文不再依赖“CNN 在所有场景胜出”，而改为证明一个职责与失败语义明确的系统合同：static/adaptive MAP 负责 LER，HMM/event/fallback 负责 tail-safe 切换，FPGA fast path 负责 deterministic source-to-action latency，CNN 与 Feedback-GRAPE teacher/student 只是可替换扩展。

## 18.1 可主张贡献与不可混排 lane

1. **GKP decoder lane**：在同一 syndrome/protocol 模型内比较 per-round `p_L,p_X,p_Y,p_Z`、average/p95/worst-window LER、static-to-oracle gap、adaptation lag 与 compute/update cost。
2. **Puviani/GQF controller lane**：只在官方 GQF 环境内比较 standard/MF/NMF 与项目扩展的 six-state logical-channel lifetime、gain retention 和计算成本；不得把本项目 syndrome decoder 的 raw LER 与 GQF lifetime 相减。
3. **FPGA lane**：只在 code family、problem size、precision 与 latency boundary 可对齐时比较 core、source-to-action、II、deadline、资源和功耗；surface-code raw decoder latency 不自动构成 single-mode GKP 胜负。
4. hidden-state oracle、finite-horizon control reference 和 channel-recovery bound 都是 privileged reference，不得加入 deployable aggregate 或硬件性能表。

## 18.2 Unified execution contract 与预注册

所有 deployable 方法共享 syndrome schema、MAP-LUT、定点精度/舍入/饱和、6-cycle event/action path、versioned A/B bank、update cadence、observed-only 输入、wall-clock/MAC/memory budget 和 deadline accounting。正式比较包含 standard binning、static joint MAP、Window MAP、EWMA、Kalman、legacy CNN residual、proposed Route-A 与独立分栏的 hidden-state oracle。

calibration/pilot/formal evaluation 的 seeds、transition rates、amplitudes 和 durations 必须互斥。smooth 组覆盖 mean、variance、correlation、periodic；abrupt/OOD 组覆盖 step、telegraph、burst、readout/reset、leakage、compound。所有场景共用 validation-only 冻结阈值，禁止逐场景调参；正式评估前冻结 paired cluster CI、multiplicity、tail-window、catastrophic degradation 和 nominal fallback non-inferiority 定义。

## 18.3 Regime-aware 安全策略

- normal/smooth posterior：允许 EWMA/Kalman 形成 candidate update，但仍需 version/CRC/age/ack 与 envelope gate；
- calibration shift/burst posterior：冻结当前在线更新，切换 trusted bank；
- leakage posterior：进入 leakage/reset FSM，不把 leakage 当 Pauli bin；
- posterior 不确定或 CRC/version/age 异常：回滚 last-known-good；
- 只有 hysteresis、健康状态和 commit acknowledgement 同时恢复，才重新开放参数提交。

每轮必须记录 posterior、reason code、bank/version、commit/freeze/rollback/fallback、action 和 deadline。prefix causality、truth denylist、policy on/off same-trace replay、fault mutation 与 independent recomputation 是完成条件，不允许只实现名义状态图。

## 18.4 决定性通过门

主指标为 per-round `p_L,p_X,p_Y,p_Z`、average/p95/worst-window LER、static-to-oracle gap closure、adaptation lag、false update、unnecessary fallback、avoided/induced errors、deadline miss、source-to-action latency 和 LUT/FF/BRAM/DSP/power。

Route A 的 promotion 必须同时满足：

1. 相对最强 deployable baseline，aggregate paired LER improvement 的 95% 下界 `>0`；
2. smooth drift 保持既有 average 优势；
3. calibration-shift 不再出现 proposed worst `55/512` 高于 static `37/512` 的反例，并通过冻结的 tail non-inferiority gate；
4. abrupt/compound 不越过预注册 catastrophic-degradation 门；nominal fallback cost 在 non-inferiority margin 内；
5. integrated fixed-point/RTL 每 family 至少 `1e5` cycles、聚合不少于 `1e6` cycles，零 bit mismatch、零 undefined action、零 silent overflow；
6. CNN matched comparison 失败时只保留消融，不得由其他模块的收益替代 CNN gate。

失败分支预先冻结：tail 失败则降为 smooth-only；average 失败则回退 static MAP-LUT + deterministic FPGA；CNN 失败进入消融；真板未到/未过时只写 hardware-aware/synthesis estimate。

## 18.5 四类外部比较

- **静态 GKP**：同一 trace 比 standard binning、static joint/top-K/full MAP、Route-A 与 model oracle；只在 paired CI 支持时声称 static-to-adaptive 优势。
- **一般漂移自适应**：刷新 primary literature 与公开实现；至少选择两个最接近方法，其中至少一个外部可复现方法，并匹配 history、update cadence、训练信息、wall-clock 与 compute budget。没有公开/可复现实现时，措辞降为“相对已实现强 baseline”。
- **Puviani NMF**：先将官方 MIT 仓库 `https://github.com/Matteo-Puviani/GQF` 固定 commit 导入 `third_party/GQF`，保留 license、environment lock、upstream source hash 和 patch series；再按论文/补充/runner 冻结的 cutoff、timing、noise、training/selection、agents/seeds 和 1000-cycle six-state logical-channel protocol做 exact reproduction。只有复现通过后才运行同 GQF simulator/budget 下的 Route-A student/contract extension；只有 paired lifetime improvement 95% LCB `>0` 才写“超过 NMF”，否则只写 retention/compression/safety extension 或负结果。
- **FPGA decoder**：按 code/problem/precision/core-vs-source-to-action/average-vs-worst/II/clock/resource/power/evidence level 规范化；不可比字段为 `null`。只有真实板上同任务可比子集支持时才写 speed advantage。

## 18.6 执行阶段与论文 GO/NO-GO

`docs/new_task_board.md` 新增 Phase 6A、Milestone 6.5—6.9。板卡无关路径从 `T6.2.2` 后继续：claim/contract/pre-registration -> unified adapter/policy/audit -> smooth/OOD/long-sequence gates -> static/drift/GQF/FPGA external lanes -> integrated P&R。实际板测只在 T6.2.3/T6.4 后执行，并要求至少 `1e6` cycles 零 bit mismatch/undefined/silent overflow/deadline miss及 core/transport/source-to-action/end-to-end 分层测量。

Phase 7 只有在 Route-A evidence gate 为 GO 后才能冻结主图和正文。若 official GQF、外部 drift baseline 或真板证据未完成，相应主张必须删除或降级；不能用同 simulator 的方向性结果替代 official reproduction，也不能用 P&R estimate 替代 measured latency。

# 19. 2026-07-21 双证据 lane 与 multimode LER SOTA 路线

本节记录 v2.3 的实质修订。此前 single-mode Route-A/V5 在 strongest Window、static 与 tail 上没有建立主算法优势，而 Phase 6C 的 multimode posterior-weighted CPD 在冻结 `d=3` balanced heteroscedastic family 上出现约 27.3% 的 project-native LER 改善。该结果既不能追认为 SOTA，也不能迁移到现有 single-mode RTL。后续论文因此改为两个并列、但不共享性能分母的证据 lane：

1. **multimode software algorithm lane**：在 surface-square GKP、全新未见漂移和 strongest eligible baselines 下争取 LER SOTA；
2. **single-mode deterministic RTL lane**：证明 6-cycle、II=1、atomic A/B、CRC/version、LKG rollback 与 fail-closed；
3. **CNN/student extension**：只近似 posterior、LLR、coset probability 或 action，不作为独立主线。

`docs/new_task_board.md` 是具体顺序与状态源；本节只冻结科学问题、基线和证据门。T6.18.3 与 T7.1—T7.2 现有产物保持历史快照，不作新 formal。

## 19.1 两条 lane 的 task signature 与禁止迁移

Multimode 算法 lane 的主对象是 surface-square multimode GKP、Gaussian displacement 及其非平稳扩展；observable 为当前 analog syndrome 和历史 syndrome，action 为 logical coset correction，主指标为 per-round `p_L`。Single-mode RTL lane 的对象是现有 production MAP/event fast path，指标为 cycles、II、atomicity、fault response、bit agreement、资源与 pre-board timing。

必须机器阻断：

- multimode `p_L` 改善被写成现有 FPGA 实现的 LER；
- single-mode 6-cycle 被写成 multimode MLD latency；
- CNN/student agreement 被写成 classical algorithm 或 RTL safety gate；
- 两 lane 的指标形成加权总分、胜场或相互补门。

最终只允许 `GO_TWO_LANE`、`GO_MULTIMODE_ONLY`、`GO_RTL_ONLY` 或 `NO_GO`。真板未到时 measured latency/jitter/deadline/power 继续为 null。

## 19.2 Multimode strong-baseline denominator

最低可信 direct decoder backend 为：

- Euclidean CPD/MWPM 与 nominal/estimated-metric weighted CPD；
- folded-Gaussian periodic analog-likelihood MWPM；
- surface-square exact logical-coset MLD；
- K-MWM 的 `K`—accuracy—cost Pareto；
- syndrome-estimated frozen marginal/mixture exact MLD。

最低可信 causal frontend 为 delayed sliding/overlapping Window、EWMA、Kalman、adapted SMC-EAP、causal GP，以及 BOCPD 或 IMM/HMM。所有 frontend 必须接相同 decoder backend，分别报告 estimator 与 backend 的贡献。Lin–Chamberland–Noh CPD、Lin–Noh exact MLD 与 Lin K-MWM 使用固定的官方 `LatticeAlgorithms.jl` source；Fukui analog information 按 folded likelihood 转录。Roy–Pousset–Royer noisy-auxiliary、Noh full-history circuit-level、Borah QLDPC-GKP、Sivak prior optimization、Puviani NMF 和外部 FPGA 不共享 task signature，只进入独立协议或边界表。

完整来源、资格标签与失败分支见 `docs/multimode_strong_baseline_registry.md`。任何 eligible strong baseline 缺失时，只能写“相对已实现 baselines”，不能写 SOTA。

## 19.3 Prequential posterior-predictive coset MLD

所有 deployable 方法使用同一因果顺序：

\[
q_t(\theta)=p(\theta_t\mid s_{<t}),\qquad
\hat L_t=\arg\max_L\int P(L,s_t\mid\theta)q_t(\theta)\,d\theta,
\]

随后才用 `s_t` 更新 posterior，candidate bank 最早在 `t+1` 生效。`true theta`、scenario ID、future suffix、formal logical label、RTS/forward-backward/Viterbi、retrospective BOCPD 和 full-record GP/FFT 禁止进入 deployable 表。

共享隐藏参数经边缘化后可能诱导 mode 间相关性，因此不能用 `E[precision] -> weighted CPD/MWPM` 冒充 posterior-predictive MLD。proposed 方法必须在 logical-coset probability 层积分；posterior delta 极限退化到 plug-in exact MLD，`d=3` 与显式 coset sum/高精度 quadrature 对拍。旧 `oracle_metric_upper_bound` 改名为 `true_metric_CPD_reference`；真正不可部署上界是 `true_theta_exact_MLD_oracle`。

正式实现前先在 development-only split 分解 estimator、likelihood metric、coset sum、posterior marginalization 和 action headroom。相对 strongest eligible deployable 的可用 headroom point 少于 15% 或 paired 95% LCB 少于 12% 时直接 NO-GO，不运行大规模 formal。

## 19.4 全新 split、formal benchmark 与 SOTA 门

T6.18.3 的 seeds、spatial pattern、balanced variance law 和 transition 参数全部视为 opened。新 train/calibration/pilot/formal 四分割必须在任何结果访问前冻结，并覆盖至少：

- `d=3,5` 与多个 below/near/above operating `sigma`；
- stationary、mean/variance/correlation/periodic、OU/random walk；
- step、telegraph、burst、heavy-tail、compound；
- 未见 spatial sign/permutation、variance law、off-diagonal covariance、transition rate/amplitude/duration 和 likelihood mismatch；
- noisy auxiliary 仅在 COR-MED source/protocol 资格通过后进入独立表。

同一 physical trace 使用 paired seeds；统计单位是 trajectory/transition block cluster，不把 round 当独立样本。pilot 只能一次性选择一个 candidate；只有 pilot point gain `>=10%` 且 95% LCB `>0` 才进入 untouched formal。

“frozen-benchmark SOTA”必须同时满足：

1. 相对每个 eligible strongest deployable baseline，aggregate relative LER improvement 的 simultaneous paired 95% LCB 均 `>10%`，absolute LCB `>0`；
2. calibration/telegraph 预注册 worst-window/CVaR improvement LCB `>0`；
3. stationary degradation 的 95% 上界 `<=2%`，任一 OOD family degradation 上界 `<=5%`；
4. `d=3/5`、多个 `sigma` 方向一致；
5. source、adapter、precision、compute、memory、deadline 和 missingness 资格完整；
6. 第二实现只从 raw data/hash 独立重算全部结论。

任一门失败即降为 best-among-implemented、non-inferior 或 negative。禁止 universal multimode/device SOTA、删除不利 baseline、按 family 重调或复用 T6.18.3 恢复主张。

## 19.5 Single-mode deterministic RTL 独立门

RTL lane 只消费 actual parameterized production top。首先检查 T6.2.2/T6.7.3/T6.9.1/T6.19.1 的 live source/config/hash；若任何实现变化，必须真实重跑而非复制旧 PASS。

完成条件为：

- formal/property 证明 A/B old-or-new、读写隔离、CRC/version/age/ack、LKG rollback、commit/cancel/drain/reset/deadline、FIFO/backpressure 和 near-wrap fail-closed；
- 每项 property 有 reachable cover 和可杀死的 targeted mutation；
- 每 family 至少 `1e5`、aggregate 至少 `1e6` CXXRTL cycles，6-cycle、II=1、零 mismatch、零 undefined、零 silent overflow/version wrap；
- actual top 至少 3 seed synthesis/P&R，报告 Fmax/resource/critical path/clock-model 与 analytic power sensitivity；
- 全部 measured 字段保持 null，真实板测继续由 T6.9.2 独立升级。

该 lane 可以支持 deterministic、atomic、fail-closed 的预板系统贡献，但不能支持 multimode decoder latency、真实 source-to-action 或 fastest FPGA claim。

## 19.6 CNN/student 与论文汇合

CNN/student 的 teacher target、权限、split、参数量、精度、training compute、latency/memory 和选择规则必须预先冻结。报告 posterior calibration、action agreement、LER retention、worst-family retention、量化误差与成本。只有在 matched budget 下达到冻结 retention margin 且有明确压缩/成本收益，才作为 optional approximation；否则进入消融，不影响 classical algorithm 或 RTL verdict。

论文主图和正文必须分成：

- multimode LER/tail/scaling/compute panel；
- single-mode 6-cycle/II=1、atomic/fail-closed、CXXRTL/formal/P&R/board-null panel；
- learning approximation 只作 inset/ablation。

Phase 7 保留 T7.1.1—T7.2.5 历史 restricted snapshot，待 T6.26.4 后通过 T7.1.5/T7.2.6 生成 delta。投稿前审计同时消费 T6.15.5、T6.19.3 与 T6.26.4，防止旧 V5/Phase 6C 证据被重新包装。

[1]: https://arxiv.org/abs/quant-ph/0008040?utm_source=chatgpt.com "Encoding a qubit in an oscillator"
[2]: https://arxiv.org/abs/2504.13383?utm_source=chatgpt.com "Logical channels in approximate Gottesman-Kitaev-Preskill error correction"
[3]: https://arxiv.org/abs/1907.12487?utm_source=chatgpt.com "Quantum error correction of a qubit encoded in grid states ..."
[4]: https://arxiv.org/abs/1706.03011?utm_source=chatgpt.com "Analog quantum error correction with encoding a qubit into an oscillator"
[5]: https://arxiv.org/abs/2308.02913?utm_source=chatgpt.com "[2308.02913] Advances in Bosonic Quantum Error ..."
[6]: https://arxiv.org/abs/2511.09491?utm_source=chatgpt.com "Adaptive Estimation of Drifting Noise in Quantum Error Correction"
[7]: https://arxiv.org/abs/2505.14775?utm_source=chatgpt.com "Performance analysis of GKP error correction"
[8]: https://orbit.dtu.dk/en/publications/analysis-of-loss-correction-with-the-gottesman-kitaev-preskill-co/?utm_source=chatgpt.com "Analysis of loss correction with the Gottesman-Kitaev ..."
[9]: https://arxiv.org/abs/2605.04892?utm_source=chatgpt.com "Real-time Surface-Code Error Correction Using an FPGA-based Neural-Network Decoder"
[10]: https://arxiv.org/abs/1912.00829?utm_source=chatgpt.com "Memory-assisted decoder for approximate Gottesman-Kitaev-Preskill codes"
[11]: https://arxiv.org/html/2504.13497v3?utm_source=chatgpt.com "Logical channel for heralded and pure loss with ..."
[12]: https://inspirehep.net/literature/2182897?utm_source=chatgpt.com "Real-time quantum error correction beyond break-even"
[13]: https://arxiv.org/abs/quant-ph/0510107?utm_source=chatgpt.com "Error Analysis For Encoding A Qubit In An Oscillator"
[14]: https://link.aps.org/doi/10.1103/PRXQuantum.2.020101?utm_source=chatgpt.com "Quantum Error Correction with the Gottesman-Kitaev-Preskill ..."
[15]: https://quantum-journal.org/papers/q-2022-02-10-648/?utm_source=chatgpt.com "Gottesman-Kitaev-Preskill codes: A lattice perspective"
[16]: https://arxiv.org/abs/2009.07941?utm_source=chatgpt.com "Stabilization of Finite-Energy Gottesman-Kitaev-Preskill States"
[17]: https://errorcorrectionzoo.org/c/gkp?utm_source=chatgpt.com "Square-lattice GKP code"
[18]: https://www.amazon.science/publications/exploring-the-quantum-capacity-of-a-gaussian-random-displacement-channel-using-gottesman-kitaev-preskill-codes-and-maximum-likelihood-decoding?utm_source=chatgpt.com "Exploring the quantum capacity of a Gaussian random ..."
[19]: https://www.mdpi.com/1099-4300/26/10/874?utm_source=chatgpt.com "Noise Transfer Approach to GKP Quantum Circuits"
[20]: https://www.researchsquare.com/article/rs-9755708/v1.pdf?c=1779267828000&utm_source=chatgpt.com "QIFE: a calibration-free decoder for finite-energy GKP qubits"
[21]: https://arxiv.org/abs/1807.01033?utm_source=chatgpt.com "Encoding a qubit in a trapped-ion mechanical oscillator"
[22]: https://www.nature.com/articles/s41567-021-01487-7?utm_source=chatgpt.com "Error correction of a logical grid state qubit by dissipative ..."
[23]: https://link.aps.org/doi/10.1103/PhysRevLett.132.150607?utm_source=chatgpt.com "Autonomous Quantum Error Correction of Gottesman-Kitaev ..."
[24]: https://www.science.org/doi/10.1126/science.adk7560?utm_source=chatgpt.com "Logical states for fault-tolerant quantum computation with ..."
[25]: https://arxiv.org/abs/2310.15546?utm_source=chatgpt.com "Robust and Deterministic Preparation of Bosonic Logical States in a Trapped Ion"
[26]: https://www.nature.com/articles/s41567-025-03002-8?utm_source=chatgpt.com "Universal quantum gate set for Gottesman–Kitaev–Preskill ..."
[27]: https://www.nature.com/articles/s41586-025-08899-y?utm_source=chatgpt.com "Quantum error correction of qudits beyond break-even"
[28]: https://www.nature.com/articles/s41586-025-09044-5?utm_source=chatgpt.com "Integrated photonic source of Gottesman–Kitaev–Preskill ..."
[29]: https://ui.adsabs.harvard.edu/abs/2023PhRvX..13c1001D/abstract?utm_source=chatgpt.com "Creation of Optical Cat and GKP States Using Shaped Free ..."
[30]: https://orbit.dtu.dk/files/294118505/PhysRevLett.128.170503.pdf?utm_source=chatgpt.com "Protocol for Generating Optical Gottesman-Kitaev-Preskill ..."
[31]: https://arxiv.org/abs/2402.09333?utm_source=chatgpt.com "Bosonic Pauli+: Efficient Simulation of Concatenated Gottesman-Kitaev-Preskill Codes"
[32]: https://the-walrus.readthedocs.io/en/latest/gallery/gkp.html?utm_source=chatgpt.com "GKP states — The Walrus 0.23.0-dev documentation"
[33]: https://quantum-journal.org/papers/q-2018-05-24-68/?utm_source=chatgpt.com "Scalable Neural Network Decoders for Higher Dimensional ..."
[34]: https://arxiv.org/abs/1912.12919?utm_source=chatgpt.com "Deep Q-learning decoder for depolarizing noise on the toric code"
[35]: https://quantum-journal.org/papers/q-2023-07-12-1058/?utm_source=chatgpt.com "A scalable and fast artificial neural network syndrome ..."
[36]: https://www.nature.com/articles/s41586-024-08148-8?utm_source=chatgpt.com "Learning high-accuracy error decoding for quantum ..."
[37]: https://link.aps.org/doi/10.1103/PhysRevResearch.6.L032004?utm_source=chatgpt.com "Artificial neural network syndrome decoding on IBM quantum ..."
[38]: https://link.aps.org/doi/10.1103/PhysRevResearch.7.013029?utm_source=chatgpt.com "Neural network decoder for near-term surface-code experiments"
[39]: https://link.aps.org/doi/10.1103/PhysRevResearch.7.023181?utm_source=chatgpt.com "Data-driven decoding of quantum error correcting codes using ..."
[40]: https://www.nature.com/articles/s41534-022-00650-z?utm_source=chatgpt.com "Multidimensional Bose quantum error correction based on ..."
[41]: https://link.aps.org/doi/10.1103/PhysRevLett.134.060601?utm_source=chatgpt.com "Neural-Network-Based Design of Approximate Gottesman ..."
[42]: https://arxiv.org/html/2412.20380v1?utm_source=chatgpt.com "Artificial Intelligence for Quantum Error Correction"
[43]: https://arxiv.org/abs/1405.6267?utm_source=chatgpt.com "Instantaneous Quantum Channel Estimation during Quantum Information Processing"
[44]: https://quantum-journal.org/papers/q-2019-04-08-131/?utm_source=chatgpt.com "Analysing correlated noise on the surface code using ..."
[45]: https://arxiv.org/abs/2010.02243?utm_source=chatgpt.com "Optimal noise estimation from syndrome statistics of quantum codes"
[46]: https://link.aps.org/doi/10.1103/PhysRevLett.133.150603?utm_source=chatgpt.com "Optimization of Decoder Priors for Accurate Quantum Error ..."
[47]: https://arxiv.org/html/2502.21044v1?utm_source=chatgpt.com "Improving error suppression with noise-aware decoding"
[48]: https://arxiv.org/html/2602.19722v1?utm_source=chatgpt.com "Differentiable Maximum Likelihood Noise Estimation for ..."
[49]: https://ieeexplore.ieee.org/document/11272758/?utm_source=chatgpt.com "FPGA-Accelerated Early-Exit Neural Decoder for Quantum ..."
[50]: https://dl.acm.org/doi/10.1145/3733239?utm_source=chatgpt.com "QUEKUF: An FPGA Union Find Decoder for Quantum Error ..."
[51]: https://www.riverlane.com/news/riverlane-unveils-first-hardware-decoder-to-deliver-real-time-scalable-quantum-error-correction?utm_source=chatgpt.com "Riverlane unveils first hardware decoder to deliver real- ..."
[52]: https://research.ibm.com/publications/fault-tolerant-bosonic-quantum-error-correction-with-the-surface-gottesman-kitaev-preskill-code?utm_source=chatgpt.com "Fault-tolerant bosonic quantum error correction with the ..."
[53]: https://ui.adsabs.harvard.edu/abs/2022PRXQ....3a0315N/abstract?utm_source=chatgpt.com "Low-Overhead Fault-Tolerant Quantum Error Correction ..."
[54]: https://link.aps.org/doi/10.1103/PhysRevA.104.062434?utm_source=chatgpt.com "Quantum error correction with the color-Gottesman-Kitaev ..."
[55]: https://quantum-journal.org/papers/q-2022-07-20-767/?utm_source=chatgpt.com "Finite Rate QLDPC-GKP Coding Scheme that Surpasses ..."
[56]: https://arxiv.org/html/2505.06385v1?utm_source=chatgpt.com "Fault Tolerant Decoding of QLDPC-GKP Codes with Circuit ..."
[57]: https://arxiv.org/abs/2605.08009 "Error Correction of Beamsplitter-Generated Entangled GKP States"
[58]: https://arxiv.org/abs/2604.08247 "Optimized Gottesman-Kitaev-Preskill Error Correction via Tunable Preprocessing"
[59]: https://arxiv.org/abs/2401.02022 "The Near-optimal Performance of Quantum Error Correction Codes"
[60]: https://arxiv.org/abs/2510.06531 "Approximate maximum-likelihood decoding with K minimum weight matchings"
