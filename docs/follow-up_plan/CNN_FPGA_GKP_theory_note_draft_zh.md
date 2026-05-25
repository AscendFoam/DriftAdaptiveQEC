# 面向漂移自适应 GKP 解码的教师锚定残差校准框架


## 目录

1. [证据边界](#证据边界)
2. [摘要](#摘要)
3. [引言](#引言)
4. [贡献摘要](#贡献摘要)
5. [GKP 码简述](#gkp-码简述)
6. [噪声与漂移设定](#噪声与漂移设定)
7. [模型架构](#模型架构)
8. [当前证据状态与近期论文计划](#当前证据状态与近期论文计划)
9. [参考文献线索](#参考文献线索)


## 摘要

近似 Gottesman--Kitaev--Preskill（GKP）量子纠错依赖有限能量振子码态和连续值 syndrome 信息来推断位移误差。但在非平稳硬件环境里，决定最佳 syndrome-to-correction 映射的有效噪声统计会随时间漂移，使固定解码器逐渐失配。本文草拟一种双时间尺度解码框架：快回路用 FPGA 友好的仿射校正执行低时延推理，慢回路则根据最近的 syndrome 统计更新仿射参数。快回路执行

$$
\Delta_t = K_t s_t + b_t,
$$

其中 $s_t$ 是当前 GKP syndrome，$(K_t,b_t)$ 是量化后的运行时参数。慢回路以经典 teacher 估计器作为稳定锚点，只让轻量 CNN 学习残差校准项，当前主线聚焦于对 $b_t$ 的残差修正。这个设计的动机来自物理 GKP 态的有限压缩结构、GKP 位移估计的局部线性 MMSE 近似、近期 QEC 文献中的模块化 learned-decoder 趋势，以及实时路径必须保持确定性与低时延的工程需求。本文的目标不是做一个通用神经 GKP 替代器，而是提出一个受部署约束的、teacher 锚定的残差校准层，用于漂移自适应 GKP 纠错。

## 引言

玻色码通过把逻辑比特编码到谐振子的 Hilbert 空间中，为量子信息保护提供了一条路径。其中 GKP 码尤其有吸引力，因为相空间中的小位移误差可以通过连续值的模综合征测量来诊断 [gkp2001]。物理 GKP 码字必然是近似的有限能量态，而不是理想的不可归一化 comb；Grimsmo 和 Puri 的综述强调，这种近似结构本身就是 GKP 码实现和容错扩展的核心 [grimsmo2021]。在理想化设置下，解码器可以把每个 syndrome 映射为一个校正位移；但在真实器件里，有限能量码态、测量效率不足、ancilla 噪声、漂移和标定误差都会让 syndrome 分布随时间变化。

这种时间变化带来一个实际张力：高质量解码器既需要利用模拟软信息和非平稳噪声统计，也要满足实时量子纠错对低时延和确定性的要求。近期 GKP 和 concatenated-GKP 解码工作已经强调了 analog soft information 的价值 [noh2022, borah2025, roy2025]；与此同时，surface-code 方向的实时 QEC 研究表明，learned module 只有在作为局部、模块化、低时延组件插入时才更有工程价值，而不应成为整个解码栈的无约束替代 [chamberland2026]。硬件条件化 decoder 和 decoder prior 优化进一步提示：慢速校准信号和硬件统计本身就是可利用的信息源 [stein2026, sivak2024]。

本项目的核心想法是把这种系统原则迁移到漂移自适应 GKP 解码里。 proposed decoder 把问题分成两个时间尺度。快回路刻意保持简单：它做一个仿射的 syndrome-informed 位移估计

$$
\Delta_t = K_t s_t + b_t,
$$

并使用固定点运算和双缓冲参数 bank。慢回路观察最近的 syndrome histogram，估计当前有效噪声状态，并以更低频率更新 $(K_t,b_t)$。与其让 CNN 从头学完整 decoder，不如保留一个经典 teacher 作为基线估计器，只学习一个残差校准项。

这个表述刻意比“神经 GKP 解码器”更窄。更准确地说，它应该被理解为：一个运行时约束下的 affine GKP fast path 的 teacher 锚定残差校准层。这个更窄的主张既更忠实于当前实现，也更符合附近最强文献的叙事边界。

## 贡献摘要

目标论文可以围绕四项贡献来写。

### 1. 面向有限能量 GKP 与有效噪声漂移的双时间尺度仿射形式

我们把实时纠错路径形式化为一个仿射估计器

$$
\Delta_t = K_t s_t + b_t,
$$

其参数由最近的 syndrome 统计驱动更新。这把 GKP 连续 syndrome 图景和硬件兼容的 runtime contract 联系起来：昂贵的推理被移出每拍路径，而 fast loop 只保留量化矩阵-向量运算。这个形式也把物理噪声要素显式化：有限压缩、位移噪声、测量噪声、偏置漂移和协方差旋转被压缩成慢回路可估计的有效状态。

### 2. 教师锚定的残差校准策略

主线不是让 CNN 直接输出全部噪声参数或全部解码参数，而是让经典 teacher 先给出一个稳定基线 $(K_t^{\rm teacher}, b_t^{\rm teacher})$。学习模块只预测一个小残差，目前是

$$
\delta b_t = f_\phi(H_{t-c+1:t}, \Delta H_t, z_t^{\rm teacher}),
$$

并在运行时使用

$$
K_t = K_t^{\rm teacher}, \qquad
b_t = {\rm EMA}\!\left(b_t^{\rm teacher}+\delta b_t\right).
$$

这个设计把 learned module 当成校准层，而不是完整 decoder 的替代物。

### 3. 面向部署的运行时架构

该架构天然对应一个快/慢实现。快回路使用 fixed-point 运算、clip、饱和诊断和 staged parameter bank。慢回路消费 32×32 的 syndrome histogram、teacher 特征和 histogram delta，然后通过 stage-and-commit 接口写入新参数。这样 latency、stale-parameter 影响、overflow 和 commit 行为就都成了一级评价对象。

### 4. 软件 HIL 评估的有界证据协议

当前仓库支持一条恢复后的 mock-backed software-HIL 路径，以及 frozen-set formal software revalidation。这些结果必须按其准确边界来报告：它们支持 frozen scenario/mode 集合上的软件级比较，但还不构成真实 `.tflite` runtime 验证、真板验证或扩展后的 paper-grade benchmark。完整论文在做更宽主张之前，还需要更强的 comparator lane、未见 drift family、runtime deployment 检查，以及带机制 hedge 的 ablation 表。

## GKP 码简述

### 理想与近似码态

GKP 码通过在相空间里构造周期性的 comb 状态，把一个 qubit 编码到振子里 [gkp2001]。在本项目的约定下，晶格常数为

$$
\lambda = \sqrt{2\pi}.
$$

理想逻辑态可以启发式写成无限 comb，例如

$$
|\bar 0\rangle \propto \sum_{n\in\mathbb{Z}} |n\lambda\rangle_q .
$$

物理 GKP 态必须是近似的有限能量态，因此 comb 峰会展宽并被包络包起来。这个有限能量结构不是细节，它决定了内在噪声底，也影响解码器看到的 syndrome 统计。

按照 Grimsmo 和 Puri 总结的有限能量图景 [grimsmo2021]，近似 GKP 码字可以理解为理想 comb 态再经过一个能量阻尼包络。常见的数学写法是

$$
|\widetilde{\mu}_L\rangle
\propto
e^{-\Delta^2 \hat a^\dagger \hat a}|\mu_L\rangle ,
$$

其中 $\Delta$ 控制包络和峰宽；只有在不物理的 $\Delta\rightarrow 0$ 极限下，才会回到理想 GKP 码。等价地说，有限压缩会把精确的晶格尖峰变成带全局包络的高斯峰。对本项目而言，这有两个直接后果。第一，syndrome 在额外硬件噪声进入前就已经是模拟的、带噪的量。第二，解码器看到的 histogram 应被视为有限能量、有限分辨率的统计对象，而不是理想码下的精确 syndrome 分布。

### syndrome 测量与模位移信息

设校正前累计的相空间位移误差为

$$
e_t = \begin{bmatrix}e_{q,t}\\ e_{p,t}\end{bmatrix}.
$$

理想 GKP syndrome 测量的是这个位移对晶格的模值：

$$
s_t = e_t \bmod \lambda,
\qquad
s_{q,t}, s_{p,t} \in [-\lambda/2, \lambda/2).
$$

真实测量还会叠加 finite-squeezing、inefficiency、shot 和 ancilla 等项，因此更适合写成

$$
\tilde{s}_t = {\rm mod}(e_t,\lambda)+\eta_t^{\rm meas}.
$$

所以 decoder 看到的不是绝对位移，而是这个位移的带噪模代表。

### 局部仿射解码及其局限

模结构让精确 GKP 解码变成非线性且分支相关的问题。但在局部分支内、且采用高斯近似时，误差 $e$ 与 syndrome $s$ 可以看成联合高斯：

$$
\begin{bmatrix}e\\s\end{bmatrix}
\sim
\mathcal{N}
\left(
\begin{bmatrix}\mu_e\\\mu_s\end{bmatrix},
\begin{bmatrix}
\Sigma_{ee} & \Sigma_{es}\\
\Sigma_{se} & \Sigma_{ss}
\end{bmatrix}
\right).
$$

对应的 linear-MMSE 估计是

$$
\hat e = \mu_e+\Sigma_{es}\Sigma_{ss}^{-1}(s-\mu_s)=Ks+b,
$$

其中

$$
K=\Sigma_{es}\Sigma_{ss}^{-1},
\qquad
b=\mu_e-K\mu_s.
$$

这解释了为什么仿射形式是一个合理的 fast-path 近似；同时也解释了它的局限：在格点判决边界附近，后验可能是多峰的，单个全局仿射估计器会把多个分支平均掉。所提出的架构因此并不声称自己是 Bayes-optimal GKP 解码，而是追求一种低时延的自适应近似。

### 逻辑失败判据

施加校正后，残余位移会被折回 GKP 的 fundamental cell。逻辑错误发生在残差跨过逻辑判决边界时，例如

$$
|r_{q,t}|>\lambda/2 \Rightarrow X_L \ {\rm error},
$$

对 $p$ 四极的情况同理。因此 closed-loop logical error rate 比离线参数回归误差更适合作为评价目标。

## 噪声与漂移设定

### 物理噪声要素

本项目的理论噪声模型把物理噪声来源和运行时 decoder 实际使用的低维有效状态分开。物理层面主要包括：

1. 有限 GKP 压缩，对应近似码态中展宽的 comb 峰和能量包络；
2. $q$、$p$ 两个四极上的加性位移噪声；
3. 测量噪声，包括有限读出精度、测量效率不足、shot noise 和辅助态不完美；
4. 相干或缓慢变化的位移偏置，用非零均值 $(\mu_q,\mu_p)$ 表示；
5. 各向异性或旋转协方差，用不同主轴宽度和旋转角 $\vartheta$ 表示。

光子损失、热激发、退相干和 ancilla fault 等底层物理通道，都可以在 decoder 更新的时间尺度上投影到这种“有效位移 + 有效测量噪声”图景中。这个投影不是要抹掉底层物理，而是给低时延 affine fast path 一个可执行接口，使它不需要在实时路径里携带完整器件级仿真器，也能响应硬件漂移。

### 有效运行时状态

慢回路把当前噪声条件概括为

$$
\theta_t^{\rm noise}
=
(\sigma_t,\mu_{q,t},\mu_{p,t},\vartheta_t),
$$

在各向异性版本中也可以写成

$$
\theta_t^{\rm noise}
=
(\sigma_{q,t},\sigma_{p,t},\mu_{q,t},\mu_{p,t},\vartheta_t).
$$

这里 $\sigma$ 控制有效位移尺度，$\mu_q$ 和 $\mu_p$ 表示 syndrome 中心偏置，$\vartheta$ 表示协方差主轴旋转。teacher 和 CNN 不需要分别识别每一个微观噪声源；它们需要估计的是决定 affine 校正参数的有效统计量。

### 漂移场景

本项目把漂移理解为 $\theta_t^{\rm noise}$ 随时间缓慢变化。当前主线漂移词汇可以由四类典型场景定义：

1. **static bias**：稳定的非零位移均值或固定协方差方向；
2. **linear ramp**：噪声尺度、偏置或方向的渐变；
3. **step change**：突然的标定跳变，例如 $\sigma_t$ 或 $\vartheta_t$ 突变；
4. **periodic drift**：由可重复环境或标定周期引起的振荡变化。

这些场景足以定义一个清晰的自适应解码问题：固定 affine decoder 是按某个分布标定的，而实际 syndrome 分布正在移动。慢回路估计器应跟踪变化中的 histogram，并在 fast path 明显失配前 stage 新的 $(K,b)$ 参数。

### 与仿射解码的关系

在局部高斯近似下，有效状态直接决定 affine map。位移尺度增大时，最优增益 $K$ 会变化；均值偏置改变 offset $b$；协方差旋转会引入跨四极耦合；测量噪声越大，syndrome 中可用信息越少，增益也应相应收缩。因此运行时接口写成 $(K,b)$，而不是自由形式的神经校正。神经分支只有在能为当前有效噪声与漂移状态产生更好的运行时参数时，才真正改善校准。

## 模型架构

### 快回路：仿射 fixed-point 解码器

快回路接收当前测得的 syndrome $s_t$ 并读取 active parameter bank。运行时计算为

$$
s_t^{\rm clip}={\rm clip}(s_t,-s_{\max},s_{\max}),
$$

$$
\Delta_t^{\rm raw}=K_t s_t^{\rm clip}+b_t,
$$

$$
\Delta_t=Q\!\left({\rm clip}(\Delta_t^{\rm raw},-\Delta_{\max},\Delta_{\max})\right),
$$

其中 $Q(\cdot)$ 表示 fixed-point 量化。项目约定采用面向 FPGA 的 Q4.20 表示法来存放 syndrome 值和运行时参数。快回路还会累积 syndrome histogram，并记录 histogram-input saturation、correction saturation 和 aggressive-parameter 事件等诊断量。

### 有效噪声状态与参数映射

慢回路用一个低维有效状态来描述当前噪声条件：

$$
\theta_t^{\rm noise} = (\sigma_t,\mu_{q,t},\mu_{p,t},\vartheta_t).
$$

给定这一状态的估计，parameter mapper 会构造误差协方差

$$
C = R(\vartheta)
\begin{bmatrix}
\sigma_q^2 & 0\\
0 & \sigma_p^2
\end{bmatrix}
R(\vartheta)^\top,
$$

以及测量协方差

$$
R_{\rm meas}=(\sigma_{\rm meas}^2+\Delta_{\rm eff}^2)I.
$$

原始增益是

$$
K_{\rm raw}=C(C+R_{\rm meas})^{-1},
$$

然后再做特征值裁剪和可选增益缩放。偏置目标是

$$
b_{\rm target}=\alpha(I-K_{\rm target})\mu,
\qquad
\mu=\begin{bmatrix}\mu_q\\\mu_p\end{bmatrix}.
$$

最后用指数平滑得到 staged runtime parameters：

$$
K_t=(1-\beta)K_{t-1}+\beta K_{\rm target},
\qquad
b_t=(1-\beta)b_{t-1}+\beta b_{\rm target}.
$$

### Teacher 估计器

teacher 家族负责从最近的 syndrome 历史里给出稳定、可解释的噪声状态估计 $\theta_t^{\rm noise}$。最简单的 teacher 会直接从 histogram window 里计算一阶和二阶矩；更强的 teacher，如 EKF、UKF、RLS 或 particle filter 变体，会再加一层时序状态先验。抽象地写就是

$$
\hat{\theta}_t^{\rm teacher} = {\rm Teacher}(H_{1:t}),
$$

以及

$$
(K_t^{\rm teacher},b_t^{\rm teacher}) = {\rm ParamMapper}(\hat{\theta}_t^{\rm teacher}).
$$

teacher 不是修辞，而是防止 learned module 变成无约束 decoder 的稳定锚点。

### CNN 残差分支

CNN 的输入是短上下文的归一化 syndrome histogram 和 histogram delta：

$$
X_t^{\rm hist}=[H_{t-c+1},\ldots,H_t,\Delta H_{t-c+2},\ldots,\Delta H_t].
$$

teacher 侧特征可以写成

$$
z_t^{\rm teacher} = (\hat{\theta}_t^{\rm teacher}, K_t^{\rm teacher}, b_t^{\rm teacher}, \Delta b_t^{\rm teacher},\ldots),
$$

不过当前 gated 分支会把它收窄成少量 teacher-$b$ 和 teacher-$\Delta b$ 标量。学习分支预测

$$
\widehat{\delta b}_t = f_\phi(X_t^{\rm hist},z_t^{\rm teacher}),
$$

然后再裁剪：

$$
\delta b_t = {\rm clip}(s_b\widehat{\delta b}_t,-b_{\max},b_{\max}).
$$

最终提交的参数是

$$
K_t=K_t^{\rm teacher},
\qquad
b_t={\rm EMA}(b_t^{\rm teacher}+\delta b_t).
$$

这个残差形式很重要：它让 CNN 只承担一个小的、与部署相关的修正，而不是去承担整个 GKP 解码问题。

### stage-and-commit 运行时契约

慢回路不会直接修改 active fast-loop 参数，而是先把候选 $(K,b)$ 写入一个非活动 bank，并在安全的 epoch 边界提交：

$$
(K_t,b_t)=
\begin{cases}
(K^A,b^A), & t<t_{\rm commit},\\
(K^B,b^B), & t\ge t_{\rm commit}.
\end{cases}
$$

这个契约是部署叙事的核心。它使论文可以评价的不只是 logical error rate，还包括 update latency、stale-parameter 效应、commit 成功率、rollback/fallback 行为，以及 fixed-point 稳定性。

## 当前证据状态与近期论文计划

当前能安全写的结果是有边界的。仓库支持一条恢复后的 mock-backed software-HIL 路径，以及一组 frozen 四场景、五模式的 software revalidation，其中 `hybrid_residual_b` 在 frozen set 内排第一。这个结论必须限定在 frozen set 和 software HIL 边界内，不能泛化到真实 `.tflite`、真板 HIL 或更宽的 SOTA 比较。

机制故事也需要谨慎。早期 trace analysis 曾暗示 committed-$b$ 幅度和 residual clipping 参与了 Gated-v5 的 seed-dependent 行为；但 T55 的 lower-clip intervention 是 mixed 且整体偏 harmful 的。因此，不能写成“较大的 committed-$b$ 就是根因”。更安全的说法是：learned residual branch 会进入某些 seed 和场景依赖的高幅度 regime，而它的效应不能被一个简单的单调幅度解释完全描述。

在完整投稿前，最重要的补充包括：

1. 保留 frozen-set 锚点，同时加入未见 drift family 和更强的 calibration/statistical baselines；
2. 给出一张机制表，区分 T55 之后哪些假说是 retained、weakened、retired 和 still-open；
3. 如果环境允许，补真实 `.tflite` 的 runtime boundary 证据；
4. 补 fixed-point 与 latency 证据，把部署约束当成可测量量，而不是 prose 假设。

## 参考文献线索

- `gkp2001`: D. Gottesman, A. Kitaev, J. Preskill, *Encoding a qubit in an oscillator*, PRA, 2001.
- `grimsmo2021`: A. L. Grimsmo, S. Puri, *Quantum Error Correction with the Gottesman-Kitaev-Preskill Code*, PRX Quantum 2, 020101, 2021. DOI: 10.1103/PRXQuantum.2.020101.
- `noh2022`: K. Noh, C. Chamberland, F. G. S. L. Brandao, *Low overhead fault-tolerant quantum error correction with the surface-GKP code*, npj Quantum Information, 2022.
- `borah2025`: S. K. Borah et al., *Fault Tolerant Decoding of QLDPC-GKP Codes with Circuit Level Soft Information*, preprint, 2025.
- `roy2025`: M.-A. Roy et al., *Decoding Multimode Gottesman-Kitaev-Preskill Codes with Noisy Auxiliary States*, preprint, 2025.
- `chamberland2026`: *Fast and accurate AI-based pre-decoders for surface codes*, arXiv:2604.12841, 2026.
- `stein2026`: *Calibration-Conditioned FiLM Decoders for Low-Latency Decoding of Quantum Error Correction Evaluated on IBM Repetition-Code Experiments*, preprint, 2026.
- `sivak2024`: V. Sivak, M. Newman, P. Klimov, *Optimization of decoder priors for accurate quantum error correction*, Nature, 2024.
- `yang2026`: *Real-time Surface-Code Error Correction Using an FPGA-based Neural-Network Decoder*, preprint, 2026.
