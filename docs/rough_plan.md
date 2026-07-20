下面我按你的项目设定来分析：**CNN + FPGA 双回路，用于近似 GKP 码本身的噪声漂移适应性解码；FPGA 做快回路，CNN 做慢回路；先理论和仿真，真实量子实验最后做**。

我的核心判断是：这个项目最强的叙事不是“CNN 在所有情况下打败最大似然解码”，而是：

> **在近似 GKP 态、有限能量、测量链路非理想、噪声参数随时间漂移的真实条件下，用慢回路学习噪声状态，用快回路执行确定性低延迟解码，从而把实际解码性能从“静态失配解码器”推向“在线 oracle 解码器”。**

也就是说，项目优势应当围绕 **model mismatch、drift robustness、low-latency implementation、finite-energy-aware decoding** 来讲，而不是简单声称“CNN 比 ML/MAP 更优”。

---

# 1. 你的方案应当如何定义

建议把系统定义成一个双时间尺度解码器：

[
\text{syndrome stream } {s_t}*{t=1}^{T},\quad s_t=(s*{q,t},s_{p,t})
]

其中 (s_{q,t},s_{p,t}) 是每轮 GKP syndrome extraction 得到的连续余量。

**快回路 FPGA** 每一轮做：

[
\hat{c}*t=f*{\theta_t}^{\mathrm{fast}}(s_t,h_t)
]

输出本轮恢复位移、Pauli frame 更新、是否触发额外 syndrome 测量、置信度或 soft likelihood。它必须是确定性、低延迟、资源可控的，可以是查表、分段线性 MAP、小型量化 CNN、TCN、LUT + 逻辑比较器，或者这些的组合。

**慢回路 CNN** 每隔 (W) 轮或每个滑动窗口做：

[
\hat{\theta}*t=g*{\phi}^{\mathrm{slow}}(s_{t-W:t})
]

估计当前噪声状态，例如：

[
\hat{\theta}*t=
(\sigma_q,\sigma_p,\rho*{qp},\mu_q,\mu_p,\eta,\Delta_{\rm meas},p_{\rm outlier},\text{drift mode})
]

然后更新 FPGA 中的阈值、LUT、先验概率、soft-output 置信度标定或小网络权重。

这个定义非常重要，因为它把 CNN 从“每轮必须在纳秒级直接输出纠错”的压力中解放出来。慢回路负责**学模型**，快回路负责**保实时性**。这比“把一个大 CNN 直接塞到 FPGA 里逐 shot 解码”更容易做成严肃项目。

---

# 2. 与已有工作的定位差异

已有 GKP 本体解码大致有几类。

第一类是 **standard binning / nearest-lattice / closest-integer 解码**。它把 syndrome 规约到 ([-\sqrt{\pi}/2,\sqrt{\pi}/2)) 区间并做最近格点恢复。GKP 原始论文的基本思想就是保护 (q,p) 小位移误差；Glancy 和 Knill 后续分析了数据态与辅助态都有误差时可纠正位移的边界。([arXiv][1])

第二类是 **最大似然 / MAP / 有限能量优化解码**。这类方法在噪声模型已知时原则上更接近最优。多模 GKP 的最大似然解码可以表述为格 theta 函数或 closest-vector/closest-lattice-point 问题；一般多模情形复杂度可能很高，但有结构码可以降低复杂度。([错误修正动物园][2]) 近似 GKP 态方面，Jafarzadeh 等 2025 年指出 finite-energy GKP 态会泄漏出理想码空间，标准分箱对有限能量态一般是次优的，优化解码可以针对电路噪声调整；但随着能量升高，优化解码相对 standard binning 的优势会收缩。([arXiv][3])

第三类是 **多轮 Bayesian / memory-assisted 解码**。Wan、Neville、Kolthammer 的 memory-assisted decoder 用多轮 syndrome extraction 和 Bayesian estimation 改善有限能量近似 GKP 态的保护，相比单轮 memoryless 方法更好。([arXiv][4])

第四类是 **analog-information / soft-information 解码**。Fukui、Tomita、Okamoto 强调 GKP syndrome 的连续测量结果包含有用的 analog information，传统把它压成离散二进制会浪费信息；他们用最大似然方法结合 analog 与 digital 信息提升纠错性能。([arXiv][5])

第五类是 **神经网络相关工作**。目前更常见的是把神经网络用于近似 GKP 态设计，或者用于 surface-GKP 等外层码解码。例如 Zeng 等用神经网络设计近似 GKP 态，目标是降低制备复杂度而不是做实时解码。([arXiv][6]) Wang 等在 surface-GKP 结构中引入神经网络解码，并报告在其模型中阈值从 (\sigma\approx0.50) 提升到 (\sigma\approx0.78)，但那是**表面码 + GKP** 的组合解码，不是单模近似 GKP 本体解码。([Nature][7])

第六类是 **硬件实时神经网络解码**，但主要在 surface code 上。2026 年 Yang 等展示了 FPGA-based NN decoder 的实时 surface-code QEC，闭环延迟 550 ns，其中 NN 解码 124 ns，QEC 周期 1.25 μs；他们也强调了吞吐量和闭环延迟是实时纠错的关键指标。([arXiv][8]) 这说明 FPGA + NN 在 QEC 控制链路中有现实可行性，但这不是 GKP 本体解码。

所以你的项目差异可以概括成：

> **已有 GKP 解码多假设噪声模型已知或准静态；已有神经网络 QEC 多集中在外层离散码；已有 FPGA NN 实验多是 surface code。你的项目把“近似 GKP 本体解码 + 噪声漂移估计 + FPGA 低延迟闭环”放在一起，这是一个明确的空档。**

---

# 3. 指标级优势分析

## 3.1 LER：逻辑错误率

### 你能合理主张的优势

在**静态、已知、独立高斯位移噪声**下，单模方形 GKP 的 closest-integer / MAP 解码已经非常强。此时 CNN 没有理论理由必然超过精确 MAP。项目叙事不要写成“CNN 击败 ML decoder”。

你真正能打的点是：

[
P_L^{\mathrm{static\ mismatch}}

>

P_L^{\mathrm{dual-loop}}
\approx
P_L^{\mathrm{oracle}}
]

其中 oracle 是知道当前真实噪声参数的最优或近最优 MAP decoder。

也就是说，你要证明的是：**当噪声漂移、辅助态质量变化、测量方差变化、(q/p) 噪声偏置变化、纯损耗与高斯位移混合变化时，静态解码器会失配，而你的慢回路能跟踪漂移，快回路能低延迟执行更新后的解码。**

建议报告以下 LER 指标：

[
\overline{P_L}
==============

\frac{1}{T}\sum_{t=1}^T P_L(t)
]

[
P_L^{95%}
=========

\text{time-window logical error rate 的 95 分位}
]

[
G_{\rm LER}
===========

\frac{\overline{P_L^{\rm baseline}}}
{\overline{P_L^{\rm dual}}}
]

[
R_{\rm oracle}
==============

\frac{\overline{P_L^{\rm dual}}-\overline{P_L^{\rm oracle}}}
{\overline{P_L^{\rm baseline}}-\overline{P_L^{\rm oracle}}}
]

其中 (R_{\rm oracle}) 越小越好。如果你能做到 (R_{\rm oracle}<0.2)，叙事上就可以说“消除了大部分由噪声漂移导致的解码失配损失”。

### 和已有工作怎么比

与 standard binning 比：你的优势是 finite-energy-aware 和 drift-aware。文献已经说明 finite-energy GKP 的标准分箱并非总是最优。([arXiv][3])

与 memory-assisted Bayesian 比：你的优势不是“多轮记忆”本身，因为 Wan 等已经做了 Bayesian memory-assisted decoder；你的优势应当是 **Bayesian 解码通常需要一个明确噪声模型，而 CNN 慢回路可以从 syndrome 流中学习模型漂移，并把模型参数下发给硬件快回路**。([arXiv][4])

与 exact MAP 比：你的优势是实际部署和失配鲁棒性。exact MAP 是上界或 oracle baseline，不应当被描述成你一定能超过的对象。

---

## 3.2 保真度：平均逻辑保真度、entanglement fidelity、逻辑通道

LER 只看逻辑翻转概率，不足以描述近似 GKP 的全部问题。有限能量 GKP 态不是严格正交态，也会泄漏出理想码空间；文献已经指出 finite-energy code state 的处理需要更谨慎的逻辑通道定义。([arXiv][3])

因此你的仿真不应只报 (P_X,P_Z)，还应报告逻辑通道：

[
\mathcal{L}_{\rm dec}
=====================

\Pi_{\rm log}
\circ
\mathcal{R}*{\rm dec}
\circ
\mathcal{N}
\circ
\mathcal{E}*{\rm GKP}
]

并计算：

[
F_{\rm avg}(\mathcal{L}_{\rm dec}, I)
]

或 entanglement fidelity：

[
F_e(\mathcal{L}_{\rm dec})
]

如果使用 Pauli transfer matrix，可以报告 (X,Y,Z) 方向衰减常数和偏置：

[
\lambda_X,\lambda_Y,\lambda_Z
]

你的优势叙事是：

> 静态硬判决解码可能只降低某一类 Pauli-like 错误，但在损耗、有限能量 envelope、测量漂移下，逻辑通道可能变成非 Pauli、偏置或有相干成分；慢回路可以估计当前通道形状，快回路更新判决边界或恢复位移，从而提高平均逻辑保真度并减少通道偏置。

这点尤其适合结合 loss 场景。Harris 等指出 pure loss 诱导的 GKP 逻辑通道并不是简单 stochastic Pauli channel。([arXiv][9]) Hastrup 和 Andersen 也分析过 GKP loss correction，并指出把 loss 先转成随机高斯位移再纠错并不总是最佳，实际参数下额外放大可能恶化性能。([arXiv][10])

所以你可以把保真度优势说成：

> **不是只优化“高斯位移下是否跨过 (\sqrt{\pi}/2) 边界”，而是优化真实近似 GKP 逻辑通道。**

---

## 3.3 纠错延迟与效率

你的双回路架构在这个指标上很有优势。

已有实时 QEC 硬件工作已经证明：解码不只是准确率问题，还要满足确定性低延迟和高吞吐。Yang 等的 FPGA NN surface-code 实验把 NN decoder 做到 124 ns，整个闭环 550 ns，并在 1.25 μs QEC cycle 内反馈。([arXiv][8]) 这给你的项目提供了很强的硬件叙事依据。

你的 GKP 本体解码甚至可能比 surface-code NN 更适合 FPGA：

1. 单模 GKP 每轮输入维度小，核心是连续 syndrome、历史窗口和噪声参数；
2. 快回路可以用 LUT / piecewise-linear likelihood / fixed-point arithmetic，而不一定要完整 CNN；
3. 慢回路 CNN 可以低频运行，不进入每轮反馈关键路径；
4. FPGA 输出可以是 frame update，不一定每次都施加物理位移，从而减少控制脉冲开销。

建议报告这些延迟指标：

[
L_{\rm decode}^{\rm worst}
]

[
L_{\rm closed-loop}
===================

L_{\rm ADC}
+
L_{\rm syndrome}
+
L_{\rm decode}
+
L_{\rm command}
+
L_{\rm AWG}
]

[
T_{\rm throughput}
==================

\text{两次连续解码输出之间的最小间隔}
]

[
\text{jitter}
=============

L_{99.9%}-L_{50%}
]

你的叙事重点应当是：

> **CNN 慢回路提升适应性，FPGA 快回路保证实时性；准确性和低延迟不是二选一。**

但要注意：如果你把完整 CNN 放在每轮快回路里，反而会削弱这个优势。更推荐“CNN 生成/更新 FPGA 解码表或低维参数”的架构。

---

## 3.4 复杂噪声场景适应性

这是你项目最强的方向。

真实 GKP 平台会面对：

* (q/p) 非等方高斯位移噪声；
* 噪声方差缓慢漂移；
* syndrome 测量噪声漂移；
* 辅助态 squeezing 漂移；
* 损耗率 (\eta(t)) 漂移；
* 位移均值偏移 (\mu_q(t),\mu_p(t))，即 calibration offset；
* 相空间旋转或相位参考漂移；
* telegraph noise / burst noise / heavy-tail outlier；
* 辅助 qubit bit flip、phase flip、reset error；
* 真实 loss-induced 非 Pauli 逻辑通道。

GKP 综述指出 finite squeezing 会表现为内禀 GKP noise，并且这种噪声会进入编码态和使用辅助 GKP 态的稳定子测量中；高质量 GKP 态对高保真处理很关键。([arXiv][11]) 综述还指出 GKP 对高斯噪声、additive noise、thermal loss 很有优势，但非高斯噪声如 dephasing 会改变情况。([arXiv][11])

已有 QEC 漂移估计工作也支持你的方向。Bhardwaj 等 2025 年提出利用 syndrome statistics 的滑动窗口方法估计 drifting noise，并显示相比静态模型可降低逻辑错误率；他们明确指出传统静态噪声假设会导致次优解码。([arXiv][12]) Huo 和 Li 更早提出不打断 QEC 的实时错误率估计，用历史纠错数据估计和预测 time-dependent noise，并降低纠错失败概率。([arXiv][13])

你的优势可以写成：

> **现有 GKP 本体解码多在固定噪声模型下比较性能；本项目直接把噪声漂移建模为 syndrome stream 中的可学习动态变量，并以硬件闭环形式实时更新解码器。**

---

## 3.5 纠错阈值：这里必须谨慎

如果你只研究**单模近似 GKP 码本身**，严格意义上通常没有 surface code 那种“随着码距增加逻辑错误率任意压低”的阈值。对于连续变量 GKP-O2O 一类结构，综述还特别提到 finite squeezing 下存在 no-threshold 行为：在有限 squeezing 资源下，连续误差不能任意压到零。([arXiv][11])

所以项目里不要把“阈值”写成离散表面码那种 threshold theorem。

更合适的指标是：

### 1. operational pseudo-threshold

定义为：

[
P_L^{\rm QEC}(\sigma^\star)
===========================

P_L^{\rm no\ QEC}(\sigma^\star)
]

或：

[
F_{\rm avg}^{\rm QEC}(\sigma^\star)
===================================

F_{\rm avg}^{\rm no\ QEC}(\sigma^\star)
]

### 2. break-even point

定义为纠错后逻辑寿命超过未纠错基准寿命。GKP 实验文献常用 QEC gain 衡量 break-even；综述指出 Sivak 等 2023 年 GKP 实验达到 beyond break-even，并且该实验的重要改进包括更好的辅助 transmon、small-Big-small 协议和在线强化学习优化。([arXiv][11])

### 3. drift-robust threshold-like boundary

你可以定义：

[
\sigma^\star_{\rm drift}
========================

\max \sigma_0
\quad
\text{s.t.}
\quad
\overline{P_L(t)}<P_{\rm target}
]

并比较静态解码、慢回路自适应、oracle MAP 三者。

这样叙事更严谨：

> **本项目不声称突破 GKP 有限 squeezing 的根本阈值限制，而是在给定 squeezing 和硬件资源下，提升 operational break-even boundary 和 drift-robust pseudo-threshold。**

---

## 3.6 真实纠错成本

你的方案在“真实纠错成本”上可以讲得很有说服力，因为它主要增加的是 classical control 复杂度，而不是 quantum hardware 复杂度。

### 可能降低的量子成本

如果慢回路能降低失配错误，则可能带来：

1. 更少的 syndrome repetition；
2. 更低的 postselection 需求；
3. 对 GKP squeezing 的要求稍微降低；
4. 较少的主动恢复位移脉冲；
5. 较少的辅助态重新制备或辅助 qubit 测量轮数；
6. 在相同目标 LER 下更长的纠错周期容忍度。

但这些必须用仿真证明，不能空口声称。

### 增加的 classical 成本

你需要诚实报告：

* FPGA LUT / BRAM / DSP / FF / LUTRAM 使用率；
* 定点位宽；
* 网络量化误差；
* 单次解码能耗；
* 参数更新频率；
* CNN 慢回路训练或推理成本；
* FPGA 参数热更新安全性；
* 在线更新失败时的 fallback 策略。

建议定义一个综合成本指标：

[
C_{\rm logical}
===============

\frac{
\text{quantum cycles}
+
\alpha \cdot \text{ancilla uses}
+
\beta \cdot \text{feedback pulses}
+
\gamma \cdot \text{classical energy}
}{
-\log_{10} P_L
}
]

这样可以把“真实纠错成本”从口号变成可比较的图。

---

# 4. 你的项目相对已有工作的优势总结表

| 指标   | 已有工作常见状态                                                               | 你的可叙事优势                                                                   | 必须避免的过度表述               |
| ---- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------- | ----------------------- |
| LER  | standard binning 简单；MAP 在已知模型下强；Bayesian memory-assisted 可用历史 syndrome | 漂移噪声下低于静态失配解码，接近 oracle MAP                                               | 不要说静态高斯下 CNN 必然超过精确 MAP |
| 保真度  | 很多工作报逻辑错误率或有效 squeezing；finite-energy 逻辑通道更复杂                          | 报告完整逻辑通道、平均保真度、Pauli transfer matrix，体现非 Pauli / loss / finite-energy 适应性 | 不要只报分类 accuracy         |
| 延迟   | 软件 MAP/Bayesian 可能难以闭环；FPGA NN 已在 surface code 中证明可低延迟                 | 慢回路不进关键路径，快回路 FPGA 固定延迟                                                   | 不要把大 CNN 放进每轮快回路导致不可部署  |
| 复杂噪声 | 多数 GKP 本体解码假设固定高斯或固定 loss                                              | syndrome-only drift tracking；适配偏置、漂移、非高斯、辅助噪声                             | 不要只在 iid Gaussian 上验证   |
| 阈值   | 单模 GKP 没有 surface-code 式真阈值；finite squeezing 有根本限制                     | 报 operational pseudo-threshold、break-even gain、drift-robust boundary      | 不要宣称突破 no-threshold 限制  |
| 真实成本 | 许多理论解码忽略控制链路；实验受辅助 qubit、测量、反馈限制                                       | 不增加量子硬件，主要增加可工程化 classical control                                        | 不要忽略 FPGA 资源、功耗、量化误差    |
| 实验落地 | GKP 实验已有 break-even、autonomous QEC、反馈制备                                | 先做仿真和硬件在环，再接真实 GKP 平台                                                     | 不要一开始就承诺真实量子实验提升        |

---

# 5. 建议的仿真实验设计

## 5.1 三层仿真框架

### 第一层：快速 syndrome-level Monte Carlo

用于跑大量 shot，估计 LER、漂移鲁棒性、置信区间。

模型：

[
u_t \sim \mathcal{N}(\mu_t,\Sigma_t)
]

[
s_t = u_t \bmod \sqrt{\pi} + n_t
]

逻辑错误由：

[
u_t-\hat{u}_t
]

是否跨越奇数个 (\sqrt{\pi}) 判定。

这层适合训练 CNN 慢回路和做大规模 ablation。

### 第二层：finite-energy effective noise model

把近似 GKP 的峰宽、包络、辅助态噪声、测量噪声加入：

[
\sigma_{\rm eff}^2
==================

\sigma_{\rm channel}^2
+
\sigma_{\rm GKP,data}^2
+
\sigma_{\rm GKP,ancilla}^2
+
\sigma_{\rm meas}^2
]

但要允许 (\sigma_q\neq\sigma_p)、相关性 (\rho_{qp}\neq0)、均值漂移 (\mu\neq0)。

这一层适合比较 standard binning、MAP、Bayesian、CNN-adaptive。

### 第三层：Fock-space / master-equation 小规模验证

用有限 Fock cutoff 模拟真实近似 GKP 态、loss、dephasing、辅助测量。它很慢，但可以用于验证 syndrome-level 近似是否可靠。finite-energy GKP 态本身会带来内禀 GKP noise，这一点在综述中被明确强调。([arXiv][11])

---

## 5.2 必须设置的 baseline

至少比较：

1. **No QEC / no correction**
2. **standard binning / nearest-integer**
3. **static MAP**：知道训练集平均噪声，但不知道实时漂移
4. **oracle MAP**：知道每一时刻真实 (\theta_t)
5. **memory-assisted Bayesian decoder**
6. **Kalman / EWMA drift estimator + MAP**
7. **CNN slow-loop + FPGA fast-loop**
8. **ablation：只有 CNN 无 FPGA约束、只有 FPGA 无慢回路、慢回路更新频率变化**

这样你才能证明：提升来自“漂移估计 + 快速部署”，不是来自数据泄漏或 baseline 太弱。

---

## 5.3 噪声场景矩阵

建议至少做 8 组噪声：

| 场景                        | 数学形式                                   | 目的                            |
| ------------------------- | -------------------------------------- | ----------------------------- |
| iid Gaussian              | (\sigma_q=\sigma_p=\sigma_0)           | sanity check；此时应接近 MAP，不应夸大优势 |
| anisotropic Gaussian      | (\sigma_q\neq\sigma_p)                 | 测试偏置适应                        |
| drift variance            | (\sigma_q(t),\sigma_p(t)) 慢变           | 核心场景                          |
| mean drift                | (\mu_q(t),\mu_p(t)) 漂移                 | 测试校准 offset                   |
| correlated Gaussian       | (\rho_{qp}(t)\neq0)                    | 测试二维联合判决                      |
| telegraph / burst         | 偶发大位移或方差跳变                             | 测试 OOD 和 tail LER             |
| pure loss + displacement  | (\eta(t)) 漂移                           | 测试非 Pauli / 非纯位移近似            |
| measurement/ancilla noise | (\sigma_{\rm meas}(t))、辅助态 squeezing 变 | 接近实验                          |

尤其要强调 drift，因为已有自适应 QEC 文献已指出静态噪声假设会导致次优解码，而 syndrome-only drift tracking 可以降低逻辑错误率。([arXiv][12])

---

# 6. CNN + FPGA 架构建议

## 6.1 CNN 慢回路不要直接输出纠错，优先输出“噪声参数”

最稳妥的设计是让 CNN 输出可解释参数：

[
\hat{\theta}_t
==============

(\hat{\mu}_q,\hat{\mu}_p,\hat{\sigma}*q,\hat{\sigma}*p,\hat{\rho},
\hat{p}*{\rm outlier},\hat{\sigma}*{\rm meas},\hat{\eta})
]

然后 FPGA 根据这些参数执行 parametric MAP：

[
\Lambda_q(s;\hat{\theta})
=========================

\log
\frac{
\sum_{k\in 2\mathbb{Z}} p_{\hat{\theta}}(s+k\sqrt{\pi})
}{
\sum_{k\in 2\mathbb{Z}+1} p_{\hat{\theta}}(s+k\sqrt{\pi})
}
]

[
\Lambda_p(s;\hat{\theta})
=========================

\log
\frac{
\sum_{k\in 2\mathbb{Z}} p_{\hat{\theta}}(s+k\sqrt{\pi})
}{
\sum_{k\in 2\mathbb{Z}+1} p_{\hat{\theta}}(s+k\sqrt{\pi})
}
]

这样有三个好处：

1. 可解释；
2. 易于 FPGA 查表；
3. 容易和 oracle MAP 比较。

## 6.2 快回路建议从 LUT + piecewise-linear MAP 开始

FPGA 快回路可以实现：

* syndrome offset correction；
* log-likelihood LUT；
* confidence LUT；
* threshold adaptation；
* Pauli frame update；
* optional displacement command；
* optional repeat-syndrome flag。

不要一开始就在 FPGA 上实现复杂 CNN。先做：

[
s_t \rightarrow \text{address}
\rightarrow \mathrm{LLR}(s_t;\hat{\theta}_t)
\rightarrow \text{correction}
]

如果后续需要，再把 LUT 换成小型量化 TCN/CNN。

## 6.3 CNN 慢回路建议用 causal 1D-CNN / TCN

输入：

[
X_t =
[s_{q,t-W:t},s_{p,t-W:t},c_{t-W:t},\text{previous confidence},\text{previous corrections}]
]

输出：

[
\hat{\theta}_t,\quad \text{uncertainty},\quad \text{OOD flag}
]

慢回路 CNN 应当带不确定性估计。若 CNN 对当前噪声状态不确定，则 FPGA 回退到 conservative MAP 或 standard binning。

---

# 7. 建议重点做的图

如果你要增强项目叙事，建议最终论文或报告至少有这些图。

## 图 1：架构图

显示：

[
\text{GKP syndrome}
\rightarrow
\text{FPGA fast decoder}
\rightarrow
\text{correction / frame update}
]

同时：

[
\text{syndrome history}
\rightarrow
\text{CNN slow estimator}
\rightarrow
\text{decoder priors / LUT update}
]

## 图 2：静态噪声下不输 baseline

在 iid Gaussian 下画：

[
P_L(\sigma)
]

对比 standard binning、static MAP、your dual-loop、oracle MAP。

目标不是大幅超过，而是证明没有明显损失。

## 图 3：漂移噪声下明显优势

让 (\sigma_q(t),\sigma_p(t)) 缓慢变化，画：

[
P_L(t)
]

比较 static MAP、Kalman-MAP、CNN-dual、oracle MAP。

这是项目最关键的图。

## 图 4：tail risk

画 windowed LER 的箱线图或 CDF：

[
\Pr(P_L^{\rm window}<x)
]

强调 CNN-dual 降低最坏窗口错误率。

## 图 5：保真度 / 逻辑通道

画 Pauli transfer matrix 或：

[
F_{\rm avg}(t)
]

证明不是只降低某一种逻辑翻转，而是改善完整逻辑通道。

## 图 6：延迟和 FPGA 资源

报告：

* decoder latency；
* closed-loop latency；
* throughput；
* LUT/BRAM/DSP/FF；
* fixed-point 位宽；
* update frequency；
* energy per decode。

## 图 7：oracle gap

画：

[
\frac{P_L^{\rm dual}-P_L^{\rm oracle}}
{P_L^{\rm static}-P_L^{\rm oracle}}
]

这是最能体现“接近 oracle”的图。

---

# 8. 实验路线建议

## 阶段 A：纯理论与快速仿真

目标：证明算法方向成立。

任务：

1. 推导 standard binning、MAP、Bayesian、adaptive MAP 的统一形式；
2. 定义近似 GKP 的 effective noise model；
3. 做 syndrome-level Monte Carlo；
4. 加入 drift；
5. 训练 CNN slow estimator；
6. 对比 oracle MAP。

这一阶段可以完全不碰真实硬件。

## 阶段 B：有限能量密度矩阵验证

目标：避免审稿人质疑“你只是模拟理想 GKP syndrome”。

任务：

1. 用 Fock cutoff 或 Zak/modular 表示模拟有限能量 GKP；
2. 加入 loss、finite squeezing、measurement noise；
3. 提取真实 syndrome distribution；
4. 验证 syndrome-level 模型是否近似可靠；
5. 计算 (F_{\rm avg})、(F_e)、logical channel。

## 阶段 C：FPGA-in-the-loop

目标：证明低延迟可部署。

任务：

1. PC/GPU 生成 syndrome stream；
2. FPGA 接收 stream；
3. FPGA 输出 correction/frame update；
4. CNN 慢回路周期性更新 FPGA LUT 或参数；
5. 记录真实 latency、jitter、throughput；
6. 与 software oracle 逐 shot 对齐比较。

这一步非常关键，因为它把项目从“机器学习仿真”提升到“实时纠错控制系统”。

## 阶段 D：离线真实数据 re-decoding

目标：先不用闭环，降低实验难度。

如果能获得 GKP 实验数据，先做：

1. 固定制备 (|0_L\rangle,|1_L\rangle,|+_L\rangle,|-_L\rangle)；
2. 收集多轮 syndrome；
3. 用已有方法离线解码；
4. 用你的 CNN-dual 模拟 re-decoding；
5. 证明 drift 期间 LER 或保真度提升。

这比直接闭环实验容易很多。

## 阶段 E：真实闭环实验

最后再做：

1. FPGA 接入控制链路；
2. 先只做 Pauli frame update；
3. 再做条件位移；
4. 最后做多轮 real-time QEC；
5. 报告 logical lifetime、QEC gain、closed-loop latency。

如果平台可选，**超导微波腔 GKP**最适合展示 FPGA 实时反馈价值，因为已有 GKP break-even 和实时反馈实验基础；**困离子平台**周期较慢，算法验证容易但 FPGA 低延迟优势不明显；**光学平台**长期价值大，但真实 feed-forward 和 GKP 态制备难度更高。

---

# 9. 项目最有力的叙事版本

建议你把项目叙事压缩成下面这句话：

> **已有 GKP 解码器在已知、静态、理想化噪声下已经很强；真正实验瓶颈是近似态有限能量、测量链路非理想和噪声漂移造成的模型失配。我们提出 CNN + FPGA 双回路：CNN 慢回路从 syndrome history 中估计漂移噪声状态，FPGA 快回路以确定性低延迟执行 finite-energy-aware MAP/LUT 解码。目标不是在静态高斯模型下超过 oracle，而是在真实漂移噪声下显著降低 drift-averaged LER、tail LER 和保真度损失，同时保持实时闭环可部署。**

这套叙事很稳，因为它与文献事实一致：GKP 对位移噪声天然有效，finite-energy 会带来额外复杂性，analog syndrome 很有价值，漂移噪声会使静态先验次优，FPGA NN 已经在 QEC 控制链中展示低延迟可行性。([arXiv][1])

---

# 10. 最后给你的执行建议

优先完成一个“最小可发表版本”：

1. **只做单模方形近似 GKP qubit**；
2. 噪声先用 drifted Gaussian displacement + measurement noise；
3. baseline 包括 standard binning、static MAP、oracle MAP、Bayesian memory-assisted；
4. CNN 只输出噪声参数，不直接输出纠错；
5. FPGA 快回路用 LUT-MAP，不急着做大 CNN；
6. 核心结果报：drift-averaged LER、tail LER、oracle gap、(F_{\rm avg})、latency、FPGA resources；
7. 之后再加 loss、non-Gaussian outlier、finite-energy Fock simulation；
8. 最后再接真实实验。

这样项目叙事会非常清楚：**先建立理论和仿真可信度，再证明硬件实时性，最后进入真实 GKP 实验。**

[1]: https://arxiv.org/abs/quant-ph/0008040?utm_source=chatgpt.com "[quant-ph/0008040] Encoding a qubit in an oscillator"
[2]: https://errorcorrectionzoo.org/c/multimodegkp "Gottesman-Kitaev-Preskill (GKP) code | Error Correction Zoo"
[3]: https://arxiv.org/abs/2504.13383?utm_source=chatgpt.com "Logical channels in approximate Gottesman-Kitaev-Preskill error correction"
[4]: https://arxiv.org/abs/1912.00829?utm_source=chatgpt.com "Memory-assisted decoder for approximate Gottesman-Kitaev-Preskill codes"
[5]: https://arxiv.org/abs/1706.03011?utm_source=chatgpt.com "Analog quantum error correction with encoding a qubit into an oscillator"
[6]: https://arxiv.org/abs/2411.01265?utm_source=chatgpt.com "Neural Network-Based Design of Approximate Gottesman-Kitaev-Preskill Code"
[7]: https://www.nature.com/articles/s41534-022-00650-z "Multidimensional Bose quantum error correction based on neural network decoder | npj Quantum Information"
[8]: https://arxiv.org/html/2605.04892v1 "Real-time Surface-Code Error Correction Using an FPGA-based Neural-Network Decoder"
[9]: https://arxiv.org/abs/2504.13497?utm_source=chatgpt.com "Logical channel for heralded and pure loss with the Gottesman-Kitaev-Preskill code"
[10]: https://arxiv.org/abs/2112.01425?utm_source=chatgpt.com "Analysis of loss correction with the Gottesman-Kitaev-Preskill code"
[11]: https://arxiv.org/html/2308.02913v3 "Advances in Bosonic Quantum Error Correction with Gottesman–Kitaev–Preskill Codes: Theory, Engineering and Applications"
[12]: https://arxiv.org/html/2511.09491v1 "Adaptive Estimation of Drifting Noise in Quantum Error Correction"
[13]: https://arxiv.org/abs/1710.03636?utm_source=chatgpt.com "Learning time-dependent noise to reduce logical errors: Real time error rate estimation in quantum error correction"
