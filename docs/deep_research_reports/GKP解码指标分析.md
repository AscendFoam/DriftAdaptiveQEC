## 结论

对于GKP码，**最重要的并不是逻辑寿命、LER或保真度中的某一个孤立数字，而是完整纠错流程所实现的“逻辑量子信道”**：

[
\mathcal N_{\mathrm L}
======================

\mathcal D\circ\mathcal R\circ
\mathcal N_{\mathrm{phys}}\circ\mathcal E ,
]

其中 (\mathcal E) 是编码或态制备，(\mathcal N_{\mathrm{phys}}) 是一个纠错周期内的损耗、退相位、有限压缩、门和测量噪声，(\mathcal R) 是GKP恢复，(\mathcal D) 是逻辑解码。

**逻辑寿命、LER和各种保真度都只是这个逻辑信道的不同投影。**

如果必须为**单模GKP量子存储器**选一个最有代表性的标量，我会选择：

[
\boxed{\text{单位时间逻辑衰减率 }\Gamma_{\mathrm L}
\quad+\quad
\text{相对于最佳物理基准的QEC增益 }G}
]

而不是单独选择裸态保真度。

---

## 不同任务下，最重要的指标并不相同

| 评价对象             | 首要指标                                         | 必须同时报告                       |
| ---------------- | -------------------------------------------- | ---------------------------- |
| GKP态制备           | 无条件逻辑SPAM错误或有限能量目标态质量                        | 有效压缩、稳定子期望值、平均光子数、成功概率       |
| 单轮GKP纠错/decoder  | 每轮逻辑错误概率 (p_{\mathrm L}^{\rm round}) 或完整逻辑信道 | (p_X,p_Y,p_Z)、泄漏、周期时间、噪声模型   |
| 重复纠错的量子存储        | 逻辑衰减率 (\Gamma_{\mathrm L})、(T_X,T_Y,T_Z)     | QEC增益、break-even、非指数衰减和时间相关性 |
| GKP逻辑门           | 可组合逻辑门错误或diamond-like距离                      | 泄漏、门时长、纠错前后误差、相关错误           |
| Surface–GKP等容错架构 | 最终逻辑失败率及其随码距的标度                              | 阈值、模式数、辅助量子比特数、时间和资源开销       |
| 通信或传感            | 可达率/容量或Fisher信息、灵敏度                          | 损耗、成功率和资源消耗                  |

---

# 1. 对量子存储而言：逻辑衰减率比单一逻辑寿命更基本

近期GKP实验常通过平均信道保真度随存储时间的衰减，定义有效逻辑衰减率 (\Gamma_{\mathrm L})。例如，GKP qudit越过break-even的实验采用平均信道保真度，并比较逻辑GKP qudit与同一系统中最佳物理Fock qudit的衰减率：

[
G_{\mathrm{QEC}}
================

\frac{\Gamma_{\mathrm{physical}}}
{\Gamma_{\mathrm{logical}}}.
]

当 (G_{\mathrm{QEC}}>1) 时，逻辑编码才真正越过break-even。该实验得到qutrit和ququart约1.82和1.87的增益。更重要的是，论文明确指出平均信道保真度一般不必严格指数衰减，因此短时间有效衰减率比强行拟合一个统一寿命更稳健。([Nature][1])

逻辑寿命当然非常直观，但它至少存在三个问题：

1. **不同逻辑方向寿命不同。**
   应分别测量 (T_X,T_Y,T_Z)，而不是只给一个 (T_{\mathrm L})。GKP qudit实验中，不同Pauli本征态的寿命确实不同。([Nature][1])

2. **寿命可能不是指数型的。**
   辅助量子比特错误、非马尔可夫反馈、残余位移的积累以及decoder历史信息都会导致非指数行为。

3. **寿命依赖基准定义。**
   仅说“寿命提高了两倍”不够；必须说明是相对于自由演化GKP态、腔体 (T_1)、裸Fock qubit，还是该硬件中最佳无编码qubit。自主GKP纠错实验通过观察纠错后逻辑寿命增加来证明纠错净收益，但跨平台比较时仍需要统一基准。([arXiv][2])

因此，对存储器我建议同时给出：

[
\boxed{
\Gamma_{\mathrm L},
\quad
T_X,T_Y,T_Z,
\quad
G_{\mathrm{QEC}},
\quad
\tau_{\mathrm{cycle}}
}
]

其中 (\tau_{\mathrm{cycle}}) 是一个纠错周期的时长。

---

# 2. 对decoder而言：LER通常是最直接的主指标

如果研究问题是：

> “这个GKP decoder或恢复方案是否优于另一个decoder？”

那么首要指标通常是：

[
p_{\mathrm L}^{\mathrm{round}}
==============================

\Pr(\text{一次完整纠错后发生不可恢复的逻辑错误}).
]

对于方形GKP码，在理想位移噪声模型下，这基本对应于恢复后的位移落入错误逻辑晶格陪集的概率。GKP码最初就是为纠正相空间中的小位移而设计的。([arXiv][3])

但必须写清楚LER的分母：

* 每次GKP综合征提取；
* 每个完整 (q+p) 纠错周期；
* 每微秒；
* 每个逻辑门；
* 每一轮外层码综合征测量。

**“每轮LER”不能脱离周期时长比较。** 一个周期很长的decoder可能每轮错误较低，却有更高的单位时间错误率。弱错误、独立周期近似下，

[
\Gamma_{\mathrm L}
\approx
\frac{p_{\mathrm L}^{\mathrm{round}}}
{\tau_{\mathrm{cycle}}}.
]

因此，比较两个单模GKP decoder时，最有用的组合是：

[
\boxed{
p_{\mathrm L}^{\mathrm{round}},
\quad
p_X,p_Y,p_Z,
\quad
\tau_{\mathrm{cycle}},
\quad
p_{\mathrm{leak}}
}
]

---

## LER与逻辑寿命什么时候等价？

如果每轮逻辑信道近似为时间平稳、无泄漏的Pauli信道，设逻辑Pauli错误概率为 (p_X,p_Y,p_Z)，则逻辑Pauli期望值的单轮衰减因子满足

[
\lambda_X=1-2(p_Y+p_Z),
]

[
\lambda_Y=1-2(p_X+p_Z),
]

[
\lambda_Z=1-2(p_X+p_Y).
]

相应的方向性逻辑寿命为

[
T_\alpha
========

-\frac{\tau_{\mathrm{cycle}}}
{\ln |\lambda_\alpha|}.
]

所以在这个理想条件下，LER和逻辑寿命基本上只是同一逻辑信道的不同参数化。

但现实有限能量GKP码并不总是Pauli信道。有限能量码字不严格正交，并会泄漏出理想GKP码空间；标准binning也可能不是有限能量条件下的最优译码。因此，一个总LER可能遗漏泄漏、相干误差和非Pauli结构。([arXiv][4])

---

# 3. “保真度”必须区分是哪一种保真度

“保真度”这个词本身过于模糊，至少要区分四种情况。

## 3.1 GKP物理态制备保真度

这是制备态 (\rho) 与某个指定目标GKP态之间的重叠。

它适合评价态制备，但不是最终纠错性能指标。原因包括：

* 理想GKP态具有无限能量，是非物理态，直接以它作为目标时保真度并不是良定义的实验指标；
* 对有限能量GKP态，结果依赖选择哪一种包络、阻尼和参数约定；
* 高整体保真度不保证位移分布在逻辑判决边界附近的尾部足够小；
* 两个具有相似保真度或相似方差的状态，可能有很不一样的LER。

近期GKP表征工作也明确指出，对非物理的理想GKP目标直接使用态保真度存在问题，因此常使用稳定子期望值或GKP nonlinear squeezing作为态质量指标。([arXiv][5])

所以态制备阶段应报告：

[
\langle S_q\rangle,\quad
\langle S_p\rangle,\quad
\Delta_q^{\mathrm{eff}},
\quad
\Delta_p^{\mathrm{eff}},
\quad
\bar n,
]

同时还要给出实际decoder下的逻辑SPAM错误率。

**有效压缩dB是重要的硬件诊断量，但不是最终QEC性能。**

---

## 3.2 单输入态保真度

例如只制备 (|0_{\mathrm L}\rangle)，纠错后测量

[
F_0=\langle0_{\mathrm L}|\rho_{\mathrm{out}}|0_{\mathrm L}\rangle.
]

这个指标非常不充分，因为它可能只测到某一种逻辑错误。例如：

* (|0_{\mathrm L}\rangle) 对逻辑 (Z) 相位错误不敏感；
* (|+!_{\mathrm L}\rangle) 对逻辑 (X) 错误不敏感；
* 某种偏置噪声可能让一个基底的保真度很好，另一个基底很差。

至少要覆盖 (X,Y,Z) 三组逻辑Pauli本征态，或者直接做逻辑过程层析。

---

## 3.3 平均逻辑信道保真度

这是量子存储实验中更有意义的保真度：

[
F_{\mathrm{avg}}
\left(\mathcal N_{\mathrm L},\mathcal I\right).
]

它评价整个逻辑Bloch球上的平均表现，而不是某一个输入态。GKP qudit越过break-even实验就是通过平均信道保真度定义有效衰减率和QEC gain。([Nature][1])

若逻辑信道确实是无泄漏的qubit Pauli信道，则

[
1-F_{\mathrm{avg}}
==================

\frac{2}{3}p_{\mathrm L},
\qquad
p_{\mathrm L}=p_X+p_Y+p_Z.
]

这时平均保真度与LER可以相互换算。

但只要存在相干误差、泄漏或强时间相关性，两者之间就不再有这种简单关系。

---

## 3.4 逻辑门保真度

平均门保真度适合实验测量，却不是长电路中最严格的指标，因为它可能低估：

* 相干误差的累积；
* 最坏输入态错误；
* 码空间泄漏；
* 连续逻辑门组合后的误差。

针对近似GKP码，近期工作提出了**可组合逻辑门错误**，同时计入目标逻辑门偏差和码空间泄漏，并且在连续门组合下具有次可加性。这比单独报告平均门保真度更接近容错计算真正需要的指标。([arXiv][6])

---

# 4. GKP码特别需要关注的附加指标

除LER、寿命和信道保真度外，GKP码还有几个不能忽略的特有指标。

### 逻辑错误偏置

应分别报告 (p_X,p_Z,p_Y)，以及 (q) 和 (p) 正交分量的残余噪声。矩形晶格、偏置压缩或非对称损耗可能使不同逻辑错误差别很大。

### 码空间泄漏

有限能量GKP态并不严格处于理想二维码空间中。一个恢复方案可能看起来逻辑测量正确，但不断增加包络能量、gauge漂移或码空间外成分。有限能量GKP逻辑信道研究专门强调了这一问题。([arXiv][4])

### 模拟综合征信息

GKP综合征不是只有0/1，而是包含“离逻辑判决边界有多远”的连续模拟信息。这些软信息会显著改变后续外码decoder得到的逻辑失败率。因此，同样的GKP压缩水平，在不同decoder和软信息使用策略下可能得到很不同的系统级LER。([arXiv][7])

### 时间和跨模式相关错误

辅助量子比特衰减、错误传播和重复综合征提取会产生时间相关性。在多模或外码级联时，还应报告两体相关逻辑错误，而不只是独立单模LER。

### 资源成本

至少应同时报告：

* 有效压缩dB；
* 平均光子数 (\bar n)；
* 辅助GKP态或辅助qubit数量；
* 每轮门和测量次数；
* 周期时间；
* 后选择或heralding成功概率；
* 经典译码延迟。

一个LER较低但需要极高能量、极慢周期或极低成功率的方案，未必是更好的工程方案。

---

# 5. 我建议采用的GKP性能报告标准

对于你刚才关注的**单模GKP纠错/解码论文**，最合理的优先级是：

### 第一层：真正决定纠错能力的指标

[
\boxed{
\mathcal N_{\mathrm L},
\quad
p_X,p_Y,p_Z,
\quad
p_{\mathrm{leak}}
}
]

即完整逻辑信道、方向性LER和泄漏。

### 第二层：量子存储的长期性能

[
\boxed{
\Gamma_{\mathrm L},
\quad
T_X,T_Y,T_Z,
\quad
G_{\mathrm{QEC}}
}
]

同时给出与最佳物理基准的break-even比较。

### 第三层：工程可比性

[
\boxed{
\tau_{\mathrm{cycle}},
\quad
\bar n,
\quad
\Delta_{\mathrm{eff}},
\quad
\text{ancilla/SPAM错误},
\quad
\text{成功率}
}
]

### 第四层：面向容错计算的最终指标

[
\boxed{
p_{\mathrm{fail}}(d)
\quad\text{及其资源开销}
}
]

也就是把单模GKP输出交给Surface code、QLDPC或其他外码以后，最终逻辑错误率是否随码距增加而下降，以及达到目标错误率需要多少模式、辅助量子比特和时间。Surface–GKP研究通常以逻辑失败率、阈值和资源开销作为最终判据。([arXiv][8])

---

## 最直接的回答

* **研究单模GKP decoder：LER最重要，但必须是完整周期的、方向分辨的LER，并同时给出周期时间和泄漏。**
* **研究GKP量子存储实验：逻辑衰减率和break-even增益比单一LER或单一寿命更有代表性。**
* **研究GKP逻辑门：可组合逻辑门错误或diamond-like误差比平均门保真度更重要。**
* **研究态制备：有效压缩、稳定子和有限能量态保真度有用，但都只是中间指标。**
* **研究可扩展容错计算：最终最重要的是固定资源下的系统级逻辑失败概率，而不是单模态本身有多“漂亮”。**

因此，从一般性和可操作性上排序，可以概括为：

[
\boxed{
\text{完整逻辑信道}

>

\text{方向性LER/逻辑衰减率}

>

\text{break-even增益}

>

\text{泄漏与相关性}

>

\text{平均信道保真度}

>

\text{有效压缩}

>

\text{裸态保真度}
}
]

[1]: https://www.nature.com/articles/s41586-025-08899-y "Quantum error correction of qudits beyond break-even | Nature"
[2]: https://arxiv.org/abs/2310.11400?utm_source=chatgpt.com "Autonomous quantum error correction of Gottesman-Kitaev-Preskill states"
[3]: https://arxiv.org/abs/quant-ph/0008040?utm_source=chatgpt.com "Encoding a qubit in an oscillator"
[4]: https://arxiv.org/abs/2504.13383?utm_source=chatgpt.com "Logical channels in approximate Gottesman-Kitaev-Preskill error correction"
[5]: https://arxiv.org/html/2604.17303v1?utm_source=chatgpt.com "Efficient characterization of general Gottesman-Kitaev ..."
[6]: https://arxiv.org/abs/2509.14658?utm_source=chatgpt.com "Composable logical gate error in approximate quantum error correction: reexamining gate implementations in Gottesman-Kitaev-Preskill codes"
[7]: https://arxiv.org/html/2606.24469v1?utm_source=chatgpt.com "When to Skip Syndrome Extraction in Surface-GKP Codes"
[8]: https://arxiv.org/abs/2103.06994?utm_source=chatgpt.com "Low overhead fault-tolerant quantum error correction with ..."
