# 结论先行

**这个方向值得投入，但需要重新定义创新点。**

单纯把原论文的 **RNN 换成 Transformer，再部署到 FPGA**，现在已经不足以构成强创新：

* 2026 年已经出现专门的 **Transformer 量子反馈控制**论文，而且同样强调长程时间关联和非马尔可夫控制；它还是 Marquardt 团队这条研究线的直接延伸。([APS Links][1])
* 2023 年已经有 **FPGA 上亚微秒神经网络量子反馈**；2026 年又有 FPGA 神经网络 surface-code 解码器实现 **550 ns 确定性闭环延迟，其中网络推理 124 ns**。([Nature][2])
* “快慢两时间尺度、自适应非平稳噪声”也已有相邻工作：一篇工作把 QEC 分成离线代码发现与在线轻量适配，2026 年 Nature 工作则利用 QEC 检错事件持续在线校准漂移。([arXiv][3])

但截至 **2026 年 7 月 21 日**，在我核验的文献中，我没有发现一项工作同时具备以下四点：

1. 面向物理 GKP/sBs 纠错；
2. 基于完整历史的非马尔可夫闭环控制；
3. Transformer 或长上下文模型；
4. FPGA 硬实时、明确的快慢双回路协同。

因此，真正值得做的课题应重新表述为：

> **面向 GKP 纠错的、硬件感知的多时间尺度非马尔可夫反馈控制：慢回路进行长程隐变量与漂移估计，快回路在 FPGA 上完成确定性、量化、安全的实时控制。**

这里的首要创新点应当是**多速率 belief-state control、GKP 软信息、硬实时闭环和逻辑寿命提升**，而不是 Transformer 这个模型名称。

---

# 一、原论文实际上做了什么

《Non-Markovian Feedback for Optimized Quantum Error Correction》于 2025 年发表于 PRL，预印本最初在 2023 年发布。它利用 Feedback-GRAPE，通过可微分的器件模型训练一个 RNN。RNN 每次接收最新的二值 ancilla 测量结果，但其隐藏状态保留完整历史，然后输出后续 sBs 纠错操作的参数。论文报告其策略相对标准 sBs 将逻辑寿命提高了大约 100%。([arXiv][4])

一个重要的术语修正是：

> 这项工作严格来说不是普通意义上的“神经网络解码器”，而是一个**测量历史驱动的反馈控制策略**。

它并非只判断“发生了哪种错误”，而是根据历史记录调整后续的 cavity displacement、qubit rotation、virtual rotation 等操作参数。原文示例中，RNN 控制的参数共有约 15 个。([arXiv][4])

所以你的课题若继续输出门参数或脉冲参数，论文标题中最好使用：

* feedback controller；
* adaptive recovery controller；
* belief-state controller；
* non-Markovian GKP control；

而不只是 decoder。

原作者已经公开了 GQF 代码，其中包含 GKP 环境、Lindblad 模拟、Feedback-GRAPE 和 RNN/前馈网络训练实现，这为复现与扩展提供了较好的起点。([GitHub][5])

---

# 二、被引文献检索口径

不同索引的计数并不一致。截至 2026 年 7 月 21 日：

* APS 页面显示约 **26** 次引用；
* Google Scholar 作者页面显示约 **30** 次；
* ResearchGate 显示 **22** 次。([APS Links][6])

这些数字的差异主要来自：

* arXiv 与期刊版被重复统计；
* 学位论文、网页、代码库和数据库条目；
* 索引延迟；
* 某些页面把“被引文献”“相关论文”和原论文自己的参考文献混在一起。

例如 ResearchGate 在 “Show more” 之后开始展示原论文的参考文献和推荐文献，其中包括 2024 年以前的论文；这些论文显然不可能反向引用 2025 年的 PRL，因此不能直接计入。Goldblatt 等人的 2024 年 cavity-qubit feedback 工作就是原论文参考文献 [53]，而不是可靠的反向引用。([ResearchGate][7])

经过以下处理：

* 合并 arXiv—期刊版本；
* 排除学位论文、网页和 Error Correction Zoo 条目；
* 在引用者的参考文献或引用上下文中核验题目、DOI 或 arXiv 编号；
* 排除数据库错误关联；

本次得到 **22 篇可核验的独立论文或预印本**。下面列出完整快照。由于引用数据库持续更新，这应理解为“截至该日期的可审计全集”，而不是永久不变的绝对全集。

---

# 三、核验到的 22 篇引用论文

## A. 技术上直接相关或方法上有明显继承关系：9 篇

### 1. *Quantum feedback control with a transformer neural network architecture*，2026

这是与你构想重合度最高的工作。它直接用 Transformer 做量子反馈控制，通过监督学习和强化学习处理长程相关及非马尔可夫系统。实验对象仍是二能级系统状态稳定和多体能量最小化，没有实现 GKP QEC，也没有 FPGA 部署。([APS Links][1])

**影响：**“Transformer 比 RNN 更适合长历史量子反馈”已经不能作为你的核心创新。

### 2. *Learning Feedback Mechanisms for Measurement-Based Variational Quantum State Preparation*，Quantum 2025

使用测量和反馈学习变分态制备策略，并在较大系统中引入 RNN 反馈。任务不是 QEC，但说明“RNN 作为测量历史压缩器”的范式正在向其他量子控制问题扩展。([arXiv][8])

### 3. *Real-time adaptive quantum error correction by model-free multi-agent learning*，2025，2026 年更新

提出两个学习层次：离线 MARL 发现完整 QEC 电路，在线 BRAVE 轻量层持续调整低维参数，以跟踪非平稳噪声。它不是 GKP，也没有 FPGA 实现，但与“快慢双回路”的概念相当接近。([arXiv][3])

### 4. *Precision Quantum Parameter Inference with Continuous Observation*，2024

利用连续量子观测轨迹进行参数估计。它不是 QEC 工作，但非常适合作为你慢回路的理论参考：慢回路不一定直接输出纠错动作，而可以估计器件隐参数和漂移。([arXiv][9])

### 5. *Preparing Schrödinger Cat States in a Microwave Cavity Using a Neural Network*，PRX Quantum 2025

实验上使用神经网络为 bosonic cavity 生成控制脉冲，展示了模拟训练、实验迁移以及 bosonic-control 的现实可行性。它不是闭环 QEC，但对你的 sim-to-real 路径很重要。([APS Links][10])

### 6. *Quantum Circuit Discovery for Fault-Tolerant Logical State Preparation with Reinforcement Learning*，PRX 2025

利用强化学习发现容错逻辑态制备电路，关注 flag、连接约束和门数，不是实时历史解码或反馈控制。([APS Links][11])

### 7. *Scaling the Automated Discovery of Quantum Circuits via Reinforcement Learning with Gadgets*，2025

通过 composite-gate “gadgets” 加速自动量子电路发现，并应用于 QEC encoder 搜索。其价值在于离线策略发现和动作空间压缩，而非在线反馈。([arXiv][12])

### 8. *Reinforcement Learning for Quantum Technology*，2026

覆盖量子反馈控制、QEC、量子电路发现和实验部署的综述，明确把实时性、可解释性、可扩展性和实验集成列为关键开放问题。([arXiv][13])

### 9. *On the Interpretability of Neural Network Decoders*，2025

研究神经网络 QEC 解码器的可解释性。虽然其主要场景不是 GKP，但它说明仅报告网络性能已经不够，论文还应解释隐藏状态或注意力究竟学习了哪类错误关联。([Wiley Online Library][14])

---

## B. 非马尔可夫动力学、连续监测或 bosonic control 邻近工作：8 篇

### 10. *CDJ-Pontryagin Optimal Control for General Continuously Monitored Quantum Systems*，Quantum 2026

把连续监测量子系统的随机路径积分推广到一般 Pontryagin 最优控制，并包含多个 bosonic-control 示例。它可作为神经网络之外的理论基线。([ResearchGate][7])

### 11. *Deterministic quantum master equation for non-Markovian signal processing*，2026

构造能够描述任意记忆结构反馈信号的确定性主方程。其意义是提供更明确的“反馈记忆维数”和非马尔可夫结构，而不是只把 RNN 隐状态当黑箱。([ResearchGate][7])

### 12. *Experimental realization of deterministic and selective photon addition in a bosonic mode assisted by an ancillary qubit*，2025

属于 ancilla-assisted bosonic 操作与实验控制路线，对 GKP recovery primitive 有间接参考价值，但没有提出历史神经反馈解码。([arXiv][15])

### 13. *Learning to Restore Heisenberg Limit in Noisy Quantum Sensing via Quantum Digital Twin*，2025

利用数字孪生或学习模型跟踪噪声并优化控制。它提供了慢回路“器件数字孪生/在线系统辨识”的邻近范式。([arXiv][16])

### 14. *Verifying Quantum Memory in the Dynamics of Spin-Boson Models*，2025–2026

研究 process tensor 和 dynamical map 意义下的量子记忆。它不做 QEC，但可用于更严格地区分“环境本身的非马尔可夫性”和“控制器使用历史信息”。([ResearchGate][7])

### 15. *On the emergence of quantum memory in non-Markovian dynamics*，2025

属于非马尔可夫动力学与量子记忆的基础研究，主要是概念性引用。([arXiv][17])

### 16. *Optimal Distillation of Non-Markovianity: Bounds, Multi-Copy Gain, and the Weak-to-Essential Transition*，2026

研究如何增强或“蒸馏”非马尔可夫性，和 GKP 实时控制距离较远。([ResearchGate][7])

### 17. *Non-Markovianity in Quantum Information Processing: Interplay with Quantum Error Mitigation*，2025

讨论非马尔可夫性在 QEC、传态与误差缓解中的作用，属于理论综述或概念层引用。([arXiv][18])

---

## C. 主要是背景式或旁支引用：5 篇

### 18. *Controllable non-Hermiticity in continuous-variable qubits*，PRA 2026

连续变量量子比特与非厄米控制工作，对你的项目没有直接算法继承。([ResearchGate][7])

### 19. *Fast arbitrary manipulation of qubit-boson states utilizing gear-inspired mechanism*，PRA 2026

关注快速 qubit–boson 操作机制，可作为脉冲控制背景，但不涉及 GKP 历史反馈或 FPGA 解码。([ResearchGate][7])

### 20. *Machine learning approach to tomographic pattern generation and classification of quantum states of light*，2026

在机器学习量子光学应用背景中引用原工作，与 QEC 反馈的直接联系较弱。([ResearchGate][7])

### 21. *Development of hybrid quantum classifiers for realistic classification tasks*，2026

把原论文作为 AI 辅助 QEC 的案例引用，技术路线本身是混合量子分类器。([ResearchGate][7])

### 22. *Generative adversarial networks in the quantum realm: Computational insights, implementation difficulties, and analytical benchmarks*，2025

属于量子生成模型综述式引用，与实时 GKP 控制没有实质继承关系。([ResearchGate][7])

---

# 四、被引网络真正说明了什么

这 22 篇中，真正直接推进原工作核心问题的论文并不多。最重要的三条后续路线是：

1. **RNN → Transformer 的长程量子反馈；**
2. **静态策略 → 在线跟踪漂移的自适应 QEC；**
3. **纯模拟控制 → 实验或硬件实时控制。**

这三条路线现在都已经各自有人占位。因此你的项目不能把三项技术简单并列为：

> Transformer + FPGA + GKP。

更有竞争力的科学问题应当是：

> **在部分可观测、非平稳且具有严格实时约束的 GKP 纠错中，如何把短时间尺度误差恢复和长时间尺度器件漂移估计分离，同时保证闭环逻辑寿命、确定性延迟和安全性？**

这是一个 POMDP、量子控制、QEC 和实时系统交叉的问题。

---

# 五、你的构想与现有 SOTA 的重合度

## 1. Transformer：已经有明确先例

现有 Transformer 量子反馈工作已经证明 Transformer 可以捕获长程相关，并在非马尔可夫控制中优于若干 RNN 或策略梯度基线；论文还直接将 QEC 列为潜在应用。([arXiv][19])

但它仍留下三个明显空位：

* 没有应用于 GKP/sBs；
* 没有硬件实时实现；
* 没有输出受约束的 GKP recovery 操作并评估逻辑寿命。

因此：

* “Transformer 用于量子反馈”不是新贡献；
* “Transformer 用于真实约束下的 GKP 非马尔可夫 feedback”仍可能是新贡献。

## 2. FPGA：单独部署已经不是创新

Reuer 等人在 2023 年已经把神经网络强化学习策略部署到 FPGA，实现单个超导量子比特的亚微秒反馈。([Nature][2])

2026 年的 FPGA surface-code 工作更进一步：

* distance-3 实验 surface code；
* 124 ns 神经网络推理；
* 550 ns 确定性闭环；
* 在 1.25 μs QEC cycle 内完成反馈；
* 实时性能接近离线解码。([arXiv][20])

因此只报告“模型成功在 FPGA 上运行，延迟若干纳秒”已经不够。你需要同时报告：

* ADC/readout integration；
* IQ discrimination；
* 数据传输；
* 网络推理；
* 动作投影或 waveform lookup；
* sequencer trigger；
* 最坏情况延迟 WCET；
* p99 延迟和是否发生 backlog。

**NN kernel latency 不等于闭环反馈 latency。**

## 3. 快慢双回路：概念也已有邻近工作

Guatto 等人的工作已经明确提出两个学习时间尺度：离线发现和在线低维适配。([arXiv][3])

2026 年 Nature 工作则把 QEC error-detection event 同时作为纠错信号和在线学习信号，在 Willow 超导处理器上针对注入漂移将 surface-code 逻辑稳定性提高 3.5 倍。([Nature][21])

你的差异必须是：

* 双回路都在运行中的 GKP QEC 内；
* 快回路负责每个 sBs 半周期或周期的 recovery；
* 慢回路从长时间轨迹估计 cavity/ancilla/readout 漂移；
* 慢回路只通过受控接口更新快回路，不能破坏实时性或稳定性。

## 4. GKP 软信息和自适应调度也开始拥挤

2026 年的 Surface-GKP 工作已经利用 inner GKP correction 产生的模拟可靠度，决定 outer surface-code 稳定子是测两个、只测一个，还是全部跳过，从而降低测量开销和测量注入噪声。([arXiv][22])

此外，GKP-concatenated codes 中利用模拟 syndrome 或 soft information 的解码路线已经较为明确。([arXiv][23])

所以“使用 GKP analog information”也不应单独作为核心创新，但它应当成为你的必要组成部分。

---

# 六、原工作基础上最值得做的改进

| 原工作的局限或未覆盖问题                                  | 推荐改进                                |
| --------------------------------------------- | ----------------------------------- |
| 主要使用二值 ancilla outcome                        | 输入原始或压缩后的 IQ、LLR、readout confidence |
| RNN 隐状态同时承担短期误差记忆和长期漂移记忆                      | 显式拆分快、慢 belief state                |
| 基于已知可微分器件模型训练                                 | 域随机化、在线 system identification、数字孪生  |
| 没有 FPGA 硬实时闭环                                 | 量化感知训练、确定性 pipeline、HIL/实验          |
| 网络直接输出多个连续门参数                                 | 安全动作 codebook + 有界 residual         |
| 延迟和 jitter 未进入训练目标                            | 在仿真中注入真实延迟分布、丢帧和排队                  |
| 训练轨迹仅约 10 个完整周期，随后评估到约 1000 周期                | 长时 curriculum、截断反传与稳定性正则化           |
| 没有把 leakage、reset failure、readout bias 作为统一输入 | 增加 leakage/reset flags 和不确定度        |
| 模型性能与硬件成本未做等预算比较                              | 在同等 LUT/DSP/BRAM、延迟和参数量下比较          |
| 策略安全性主要依赖训练                                   | OOD 检测、动作投影、标准 sBs fallback         |

原论文确实已经测试了不同噪声水平和不完美门，因此不能简单说它“完全没有鲁棒性”。更准确的批评是：它仍主要依赖预先给定的可微分动力学模型，离线泛化测试并不等价于实际运行中的在线漂移辨识。论文还使用约 10 个完整 QEC cycles 的轨迹训练，再在长至约 1000 cycles 的轨迹上评估，这为长时间尺度建模留下了明显空间。([arXiv][4])

---

# 七、推荐的快慢双回路架构

## 1. 系统结构

```text
      ADC / IQ / ancilla readout / reset & leakage flags
                           │
                           ▼
                ┌─────────────────────┐
每半周期/周期 → │ FPGA 快回路 F_fast  │ → 安全动作投影
                │ 固定点、固定延迟      │        │
                └─────────────────────┘        ▼
                           ▲              pulse/codebook/
                           │              virtual rotation
                  参数双缓冲更新
                           │
                ┌─────────────────────┐
每 K 周期或事件 →│ 慢回路 Transformer   │
触发            │ 长历史、漂移估计      │
                └─────────────────────┘
                           ▲
                 trajectory ring buffer
```

可写为：

[
h_t^{f}
=======

F_{\theta_k}^{(q)}
\left(h_{t-1}^{f},x_t\right),
]

[
a_t
===

\Pi_{\mathcal A}
\left[
a_{\mathrm{sBs}}+
G_{\theta_k}^{(q)}(h_t^f)
\right],
]

[
(z_{k+1},\theta_{k+1})
======================

T_\phi
\left(
x_{t-W:t},
a_{t-W:t},
z_k
\right).
]

其中：

* (F^{(q)})：FPGA 上量化的快策略；
* (h_t^f)：短时间尺度 belief state；
* (T_\phi)：慢回路因果 Transformer；
* (z_k)：器件漂移或隐噪声状态；
* (\theta_k)：快策略参数、阈值或 codebook；
* (\Pi_{\mathcal A})：安全动作集合投影；
* (a_{\mathrm{sBs}})：标准 sBs 基准动作。

## 2. 快回路应该做什么

快回路每个半周期或完整周期执行，输入可包括：

* ancilla 二值 outcome；
* IQ 或 LLR；
* readout confidence；
* ancilla reset 成功标志；
* leakage indicator；
* 最近的 pulse amplitude、phase 和 timing；
* 当前慢回路输出的漂移估计。

快回路输出最好不是任意连续脉冲，而是：

[
\text{action}=
\text{codebook index}
+
\text{bounded residual}.
]

例如：

* 选择一个已经校准好的 sBs 参数模板；
* 加一个有限范围内的 cavity displacement 修正；
* virtual rotation；
* Pauli-frame 更新；
* 必要时触发 reset 或 fallback。

这样做有三个优势：

1. FPGA 实现简单；
2. 避免网络产生危险或不可实现的脉冲；
3. 更容易给出稳定性和 WCET 保证。

## 3. 慢回路应该做什么

慢回路不必直接参与每次实时决策。它可以估计：

* cavity loss rate (\kappa)；
* ancilla (T_1/T_2)；
* readout bias 和 readout fidelity；
* Stark shift 或频率漂移；
* ancilla leakage；
* reset error；
* pulse-amplitude scale drift；
* 当前噪声是否超出训练分布。

随后慢回路更新：

* 快回路的少量 adapter 参数；
* normalization 或 threshold；
* 动作 codebook；
* belief-state prior；
* measurement schedule；
* fallback 灵敏度。

更新应使用双缓冲，在 QEC cycle 边界原子切换，不能在快回路执行过程中改变权重。

---

# 八、Transformer 是否应该放在快回路中

我的判断是：**不应强制把完整 Transformer 放进最内层快回路。**

原因并不是 FPGA 做不了 Transformer，而是对于原论文的输入——每半周期只有一个二值 outcome——完整 self-attention 很可能是过度设计：

* 每个 token 的信息量极低；
* 固定历史窗口的 full attention 成本随窗口长度平方增长；
* softmax、KV memory 和归一化增加 BRAM/DSP 压力；
* RNN、TCN 或状态空间模型可能在相同硬件预算下更快；
* Reviewer 会要求解释 Transformer 为什么是必要的。

Transformer 的合理性主要来自：

* 长历史；
* 多模态 telemetry；
* 稀疏但关键的历史事件；
* 长时间尺度漂移；
* 跨多个变量的相关性。

因此推荐两个方案。

## 方案 A：最稳妥

* 慢回路：因果 Transformer；
* 快回路：量化 GRU/TCN/状态空间模型；
* 训练时由慢 Transformer 作为 teacher；
* 部署时快回路使用 distilled student。

论文仍可称为 Transformer-guided hierarchical feedback，但快路径不会被大模型拖累。

## 方案 B：保留“Transformer on FPGA”

* 快回路使用 tiny causal Transformer；
* 固定短窗口；
* local/sliding-window attention；
* 固定 token 数和维数；
* 完全展开或深度流水；
* 慢回路使用更长窗口和更大模型。

即使采用方案 B，也必须与 GRU、TCN 和状态空间模型做**同资源、同延迟、同输入**比较。不能只证明 Transformer 比原作者未经硬件优化的 RNN 精度高。

---

# 九、建议的训练方法

## 阶段 1：严格复现原论文

首先直接复现 GQF：

* 相同噪声参数；
* 相同 binary measurement input；
* 相同 sBs action space；
* 相同 lifetime metric；
* 相同训练和评估协议。

若无法稳定复现接近论文所报告的约 100% lifetime improvement，不应立即增加 Transformer 和 FPGA 复杂度。([arXiv][4])

## 阶段 2：架构公平比较

在完全相同的信息和动作空间下比较：

* 标准 sBs；
* 只看最新结果的前馈网络；
* 原始 RNN；
* GRU/LSTM；
* TCN；
* HMM/Bayesian filter/particle filter；
* causal Transformer；
* selective state-space model。

必须匹配：

* 参数量；
* FPGA latency；
* LUT/FF/BRAM/DSP；
* 数值精度；
* 训练样本量。

这是判断 Transformer 是否真的有科学必要性的关键实验。

## 阶段 3：增加模拟软信息

再加入：

* IQ；
* LLR；
* measurement confidence；
* leakage/reset flags。

这里要特别做消融实验，因为模型性能提升可能来自“输入信息变多”，而不是 Transformer 架构本身：

[
\text{binary RNN}
\rightarrow
\text{binary Transformer}
\rightarrow
\text{analog RNN}
\rightarrow
\text{analog Transformer}.
]

## 阶段 4：非平稳噪声和慢回路

训练时随机化：

* (\kappa)；
* (T_1/T_2)；
* readout fidelity；
* detuning；
* reset error；
* pulse scale；
* drift bandwidth；
* 突变和缓慢漂移；
* 网络延迟和 jitter。

慢回路的目标不应只是在训练集上提高平均 fidelity，而应减少：

* 漂移后的逻辑寿命下降；
* 恢复时间；
* OOD 状态下的灾难性动作；
* 重校准停机频率。

## 阶段 5：硬件感知训练

推荐在目标函数中加入：

[
J =
\mathbb E[T_L]
-\lambda_{\mathrm{lat}}L_{\mathrm{WCET}}
-\lambda_{\mathrm{res}}R_{\mathrm{FPGA}}
-\lambda_{\mathrm{safe}}P_{\mathrm{unsafe}} .
]

执行：

* quantization-aware training；
* 8–12 bit fixed point 起步；
* structured pruning；
* teacher–student distillation；
* latency-in-the-loop；
* OOD uncertainty calibration；
* 标准 sBs 安全回退。

---

# 十、必须报告的 SOTA 指标

## 量子纠错指标

* logical lifetime (T_L)；
* logical error per cycle；
* break-even gain；
* 各逻辑 Pauli 本征态性能；
* recovery from injected displacement；
* 长轨迹稳定性；
* 不同漂移速率下的 lifetime；
* leakage/reset failure 下的性能。

## 实时系统指标

* 推理 latency；
* end-to-end closed-loop latency；
* p50、p99 和 WCET；
* throughput；
* 是否发生 cycle backlog；
* LUT、FF、BRAM、DSP；
* 时钟频率；
* 功耗；
* 量化前后性能差；
* 模型更新时延；
* 慢回路更新是否干扰快回路。

## 必要消融

* 快回路 only；
* 慢回路 only；
* 快慢双回路；
* binary vs analog；
* RNN vs Transformer vs SSM；
* continuous action vs codebook；
* fixed parameters vs adaptive parameters；
* 无延迟训练 vs latency-aware training；
* 无安全投影 vs 安全投影。

---

# 十一、建议的研究止损标准

下面是我建议的工程和论文门槛，不是现有文献统一规定的标准。

## 第一阶段：算法门槛

在与原 RNN **相同输入、相同动作空间、相同计算预算**时，至少满足一个：

* logical lifetime 再提高约 15%–20%；
* 达到相同 lifetime，但延迟或资源减少约 2 倍；
* 在模型失配和漂移下显著优于 RNN，而静态条件下不退化。

若 binary-input Transformer 在公平比较中不能超过 RNN/SSM，应立即把主要贡献转向多速率控制和软信息，而不是继续强行证明 Transformer。

## 第二阶段：量化门槛

* FPGA 量化后 lifetime 或 logical error 性能下降小于约 2%–5%；
* 不出现特定历史模式下的极端输出；
* 量化模型和浮点模型的 action distribution 一致性可解释。

## 第三阶段：实时门槛

* end-to-end latency 最好不超过 cycle budget 的约 10%；
* WCET 有确定上界；
* 连续运行不积压；
* 慢回路更新不暂停快回路；
* OOD 时能够在一个或有限个 cycle 内切回标准 sBs。

## 第四阶段：SOTA 门槛

至少完成硬件在环：

* 真实或录制的 IQ/readout 数据；
* 实际 FPGA bitstream；
* 实际 sequencer/trigger；
* 真实数据传输链；
* 闭环时序测量，而不是只给 HLS synthesis report。

仅有“数值仿真 + HLS 资源估计”的工作，面对当前 FPGA QEC 和 Transformer quantum-feedback 文献，通常只能算增量型结果。真正有竞争力的是：

* hardware-in-the-loop；
* 或实际 GKP 装置闭环；
* 或至少与实验团队提供的真实轨迹和控制栈结合。

---

# 十二、创新性和可行性评分

| 维度           | 你目前的表述：Transformer+FPGA | 推荐重构后的课题 |
| ------------ | ----------------------: | -------: |
| 模型创新         |                     2/5 |    3.5/5 |
| QEC 科学问题     |                     3/5 |    4.5/5 |
| 与现有 SOTA 区分度 |                   2.5/5 |      4/5 |
| 算法可行性        |                     4/5 |      4/5 |
| FPGA 实现可行性   |                   3.5/5 |    3.5/5 |
| 纯模拟论文潜力      |                   2.5/5 |    3.5/5 |
| HIL 论文潜力     |                     3/5 |      4/5 |
| 实际 GKP 闭环潜力  |                   3.5/5 |    4.5/5 |

这里最大的风险不是网络训练，而是：

1. 是否有真实 GKP 硬件或实验轨迹；
2. sim-to-real 模型失配；
3. 从 IQ 到 pulse trigger 的完整实时链；
4. 能否证明双回路收益来自时间尺度分离，而非简单增加参数；
5. Transformer 在同 FPGA 预算下是否真的胜过 GRU/TCN/SSM。

---

# 最终判断

**值得投入，但不值得按“RNN 换 Transformer，然后放到 FPGA”这一原始版本投入。**

最推荐的课题定义是：

> **Hardware-aware hierarchical non-Markovian feedback for real-time GKP quantum error correction**

或者：

> **Multi-rate belief-state control for FPGA-accelerated GKP quantum error correction**

最有说服力的贡献组合应当是：

1. **快慢双 belief-state 回路的明确理论定义；**
2. **慢回路 Transformer 估计器件隐状态和漂移；**
3. **快回路 FPGA 量化策略提供确定性 recovery；**
4. **使用 IQ/LLR、leakage、reset 等软信息；**
5. **codebook + bounded residual + OOD fallback 的安全控制；**
6. **端到端闭环延迟，而非仅网络推理延迟；**
7. **在非平稳噪声下证明逻辑寿命和恢复速度的提升；**
8. **与 RNN、TCN、Bayesian filter 和状态空间模型进行等资源比较。**

从研究策略看，最合理的实施顺序是：

> **先复现原始 GQF → 做等预算架构比较 → 引入漂移与双回路 → 加软信息 → 量化与 FPGA → HIL/实验。**

只要能够完成硬件在环，最好进一步接入实际 GKP 装置，这个重新定义后的方向在创新性和 SOTA 可行性上是有研究价值的；它的论文核心将是**“多时间尺度、硬实时、漂移鲁棒的 GKP 闭环控制”**，Transformer 和 FPGA则分别是实现长程估计与确定性执行的技术手段。

[1]: https://link.aps.org/doi/10.1103/m429-jy1j?utm_source=chatgpt.com "Quantum feedback control with a transformer neural network ..."
[2]: https://www.nature.com/articles/s41467-023-42901-3 "Realizing a deep reinforcement learning agent for real-time quantum feedback | Nature Communications"
[3]: https://arxiv.org/abs/2509.03974 "[2509.03974] Real-time adaptive quantum error correction by model-free multi-agent learning"
[4]: https://arxiv.org/html/2312.07391v2 "Non-Markovian feedback for optimized quantum error correction"
[5]: https://github.com/Matteo-Puviani/GQF?utm_source=chatgpt.com "GQF - Non-Markovian feedback for quantum error ..."
[6]: https://link.aps.org/doi/10.1103/PhysRevLett.134.020601?utm_source=chatgpt.com "Non-Markovian Feedback for Optimized Quantum Error ..."
[7]: https://www.researchgate.net/publication/388078852_Non-Markovian_Feedback_for_Optimized_Quantum_Error_Correction "(PDF) Non-Markovian Feedback for Optimized Quantum Error Correction"
[8]: https://arxiv.org/html/2411.19914v2?utm_source=chatgpt.com "Learning Feedback Mechanisms for Measurement-Based ..."
[9]: https://arxiv.org/abs/2407.12650?utm_source=chatgpt.com "Precision Quantum Parameter Inference with Continuous Observation"
[10]: https://link.aps.org/doi/10.1103/PRXQuantum.6.010321?utm_source=chatgpt.com "Preparing Schr\\\"odinger Cat States in a Microwave Cavity ..."
[11]: https://link.aps.org/doi/10.1103/gqpr-dgz7?utm_source=chatgpt.com "Quantum Circuit Discovery for Fault-Tolerant Logical State ..."
[12]: https://arxiv.org/abs/2503.11638?utm_source=chatgpt.com "Scaling the Automated Discovery of Quantum Circuits via Reinforcement Learning with Gadgets"
[13]: https://arxiv.org/abs/2601.18953?utm_source=chatgpt.com "Reinforcement Learning for Quantum Technology"
[14]: https://advanced.onlinelibrary.wiley.com/doi/10.1002/qute.202500158?utm_source=chatgpt.com "On the Interpretability of Neural Network Decoders"
[15]: https://arxiv.org/abs/2212.12079 "https://arxiv.org/abs/2212.12079"
[16]: https://arxiv.org/abs/2508.11198 "https://arxiv.org/abs/2508.11198"
[17]: https://arxiv.org/abs/2507.21907 "https://arxiv.org/abs/2507.21907"
[18]: https://arxiv.org/abs/2510.20224 "https://arxiv.org/abs/2510.20224"
[19]: https://arxiv.org/abs/2411.19253?utm_source=chatgpt.com "Quantum feedback control with a transformer neural network architecture"
[20]: https://arxiv.org/abs/2605.04892 "[2605.04892] Real-time Surface-Code Error Correction Using an FPGA-based Neural-Network Decoder"
[21]: https://www.nature.com/articles/s41586-026-10759-2 "Reinforcement learning control of quantum error correction | Nature"
[22]: https://arxiv.org/abs/2606.24469 "[2606.24469] When to Skip Syndrome Extraction in Surface-GKP Codes"
[23]: https://arxiv.org/abs/2505.06385 "https://arxiv.org/abs/2505.06385"
