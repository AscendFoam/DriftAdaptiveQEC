# `CNN_FPGA_GKP_theory_note_draft.tex` 逐段口头汇报解释稿

本文用于口头汇报解释 `docs/follow-up_plan/CNN_FPGA_GKP_theory_note_draft.tex`。目标听众假设为：理解一般科研问题和工程系统，但不熟悉量子纠错、GKP 码或 FPGA/HIL 术语。

本文和英文 note 保持同一证据边界：

- 当前可信边界是 `mock-backed software HIL` 与 frozen-set software revalidation。
- 不声称真板 HIL 已完成。
- 不声称真实 `.tflite` runtime 已恢复。
- 不声称 T64 的 `statcalib` extension lane 已经是成熟 SOTA comparator。
- T64 只能作为单独标注的第六 lane，不改写历史 T24 冻结五模式表。
- T65 是一致性 guard 和报告收口任务，不新增实验结果。
- 机制故事仍然要谨慎，不能说 residual-b 幅度已经给出因果闭环。

---

## 0. 汇报总叙事

### 一句话版本

这个项目研究的是：在 GKP 量子纠错中，把每一拍必须快速完成的纠错操作压缩成 FPGA 友好的仿射公式 `Delta = K s + b`，同时让较慢的 teacher/CNN 校准回路根据最近的 syndrome histogram 更新 `K,b`，从而适应噪声漂移。

### 面向不懂量子纠错老师的 2 分钟版本

可以这样开场：

> GKP 码把一个量子比特编码到振子的连续变量相空间里。错误可以理解成相空间里的小位移。每一轮纠错会测到一个连续值 syndrome，它告诉我们误差在 GKP 晶格周期内的大概位置。问题是，真实硬件的噪声大小、偏置和相关方向会随时间漂移，所以固定 decoder 会逐渐失配。我的方案是一个快慢回路：快回路只执行 `Delta = K s + b`，适合低时延 FPGA；慢回路用最近一段 syndrome histogram 和经典 teacher 估计当前噪声状态，再让轻量 CNN 只学习一个残差校准项，而不是从头替代完整 GKP decoder。

### 这份 note 的核心判断

汇报时要抓住四点：

1. **理论来源**：`K s + b` 可以看成局部高斯近似下的 linear-MMSE GKP 位移估计。
2. **工程约束**：复杂估计放到慢回路，每拍执行路径保持固定点矩阵乘加。
3. **文献定位**：已有工作已经覆盖 analog soft information、adaptive prior、calibration-conditioned neural decoder 和 FPGA QEC decoder，所以本文贡献必须收窄到 teacher-anchored residual calibration for affine GKP fast path。
4. **证据边界**：T24/T57/T64 都是软件 HIL 层面的有界证据；T64 很有价值，但仍是 extension lane，不是真板或 `.tflite`。

---

## 1. 标题、摘要和目录

### 标题

原文标题：

> A Teacher-Anchored Residual Calibration Framework for Drift-Adaptive GKP Decoding

口头解释：

> 这个标题里有三个关键词。`Teacher-Anchored` 表示 CNN 不自由接管 decoder，而是被一个经典估计器锚定。`Residual Calibration` 表示 CNN 只学基线之外的小修正。`Drift-Adaptive GKP Decoding` 表示目标不是静态噪声，而是噪声参数会随时间漂移的 GKP 解码。

术语解释：

- **Teacher**：经典统计估计器，比如 window moment、EKF、UKF、RLS。它给出稳定、可解释的基线。
- **Anchored**：被锚定，表示神经网络输出围绕 teacher 结果做小修正，不完全自由。
- **Residual calibration**：残差校准，先有基础估计，再学习“还差多少”。
- **Drift-adaptive**：适应漂移。漂移可以是噪声均值、方差、协方差方向随时间变化。
- **GKP decoding**：根据 GKP syndrome 判断应施加什么相空间位移校正。

### Abstract 第一段：近似 GKP 和连续 syndrome

口头解释：

> 摘要开头先讲问题对象。GKP 码把量子信息编码在振子中，纠错时依赖连续值 syndrome 来推断位移误差。现实中不是理想 GKP，而是有限能量近似 GKP，因此 syndrome 的统计分布会受到 GKP 峰宽、测量噪声、电路误差和慢漂移共同影响。

术语解释：

- **Approximate GKP**：近似 GKP。理想 GKP 态需要无限能量，现实中只能制备有限能量近似态。
- **Finite-energy**：有限能量，意味着 GKP 梳状峰不是无限尖锐，而是有宽度和包络。
- **Continuous-valued syndrome**：连续值综合征，不是普通稳定子码里常见的 0/1 syndrome。
- **Displacement error**：相空间中的位移误差，可以分成 q 和 p 两个方向。
- **Calibration drift**：硬件标定参数随时间变化，导致同一个 syndrome 对应的最佳校正也变。

### Abstract 第二段：双时间尺度和 teacher residual

口头解释：

> 这一段给出方法。快回路做低时延仿射校正 `Delta_t = K_t s_t + b_t`。慢回路根据最近的 syndrome histogram 更新参数。CNN 的角色很窄：它被 teacher 锚定，只预测有界 residual，当前主要修正 `b_t`。

术语解释：

- **Two-timescale**：双时间尺度。快回路每拍执行，慢回路按窗口更新。
- **Fast loop**：快回路，直接输出校正，不能太复杂。
- **Slow loop**：慢回路，汇总统计、估计噪声、更新参数。
- **Affine correction**：仿射校正，形式是 `K s + b`。
- **Bounded residual**：有边界的残差，输出会被 clip，避免闭环失稳。

### Abstract 第三段：文献定位和贡献边界

口头解释：

> 摘要最后把本文放进文献版图里。已有工作已经做了 GKP analog soft information、decoder prior adaptation、calibration-conditioned neural decoder 和 real-time FPGA decoder。我们不能把这些宽方向当作原创。本文真正的定位是：在部署约束下，为 GKP affine fast path 做 teacher-anchored residual calibration。

术语解释：

- **Analog soft information**：模拟软信息，保留连续测量值或置信度，而不是硬判决。
- **Decoder prior**：decoder 使用的先验噪声概率或权重。
- **Calibration-conditioned**：把硬件标定信息作为条件输入或调制信号。
- **Board-level FPGA demonstration**：真实 FPGA 板级验证。当前项目还没有。

### 目录

口头解释：

> 新版 note 加了目录，目的是让导师能看到论文结构已经开始成型。它不是只写 abstract 和 Introduction，而是把理论背景、噪声模型、模型结构、相关工作、结果框架和近期计划都放进一条主线。

---

## 2. Scope and Evidence Boundary

### 段落 1：这是一份 theory-facing draft

口头解释：

> 这一段先说明文档性质。它是未来论文前半部分和方法部分的理论草稿，用来明确科学 claim、数学模型、架构和 benchmark 结构，不是把当前项目证据等级往上拔。

术语解释：

- **Theory-facing draft**：偏理论和写作定位的草稿。
- **Evidence level**：证据等级，指结果能支持到什么强度。
- **Benchmark structure**：实验对比结构，包括场景、baseline、指标、统计方法。

### 段落 2：当前验证边界

口头解释：

> 当前可信边界还是 mock-backed software HIL。T64 新增了 clean-provenance 的四场景 statcalib extension lane，但它必须单独标注，不能改写 T24 历史冻结表。T65 只是 consistency guard，不会新增实验结论。

术语解释：

- **Mock-backed software HIL**：保留硬件接口语义，但后端是 mock 软件，不是真板。
- **T24 frozen table**：历史冻结的四场景五模式正式软件复验表。
- **T64 statcalib extension lane**：在 T24 五模式之外追加的第六条 statcalib 对比路线。
- **Clean provenance**：运行来源清楚，包括 branch、commit、run root、summary 对得上。
- **Consistency guard**：一致性守卫，通常是检查报告与 artifact 是否一致。

### 段落 3：为什么要前置边界

口头解释：

> 这个边界不是自降身价，而是避免被审稿人抓住过度 claim。我们可以说架构 FPGA-friendly，但不能说已经 FPGA-validated。可以说 statcalib 在 T64 软件 HIL 中结果强，但不能说已经是成熟 SOTA comparator。

术语解释：

- **FPGA-friendly**：结构适合 FPGA，例如 fixed-point、矩阵乘加、时序清楚。
- **FPGA-validated**：已在真实 FPGA 上验证。当前不能这么说。
- **SOTA comparator**：领域最强或成熟对比方法。T64 的 statcalib 还不能这样写。
- **Causal closure**：因果闭环，指机制已经通过干预证明。当前也不能这么说。

---

## 3. Introduction

### 段落 1：玻色码和 GKP 基本背景

口头解释：

> 这一段介绍 GKP 码。它属于 bosonic code，也就是用振子这样的连续变量系统编码逻辑量子比特。理想 GKP 态在相空间里是周期晶格，小位移错误可以通过模 syndrome 测量诊断出来。

术语解释：

- **Bosonic code**：玻色码，用振子模式编码量子信息。
- **Oscillator**：振子，可用 q/p 两个连续变量描述。
- **Hilbert space**：量子态所在的数学空间。
- **Phase space**：相空间，由 q 和 p 两个正交坐标组成。
- **Modular syndrome**：取模后的 syndrome，只知道误差在一个晶格周期内的位置。

### 段落 2：有限能量和真实噪声

口头解释：

> 这一段说明现实 GKP 不理想。有限能量会让梳状峰变宽，测量、辅助态、电路和振子损耗也会引入噪声。这些因素让 syndrome 分布随硬件状态变化。

术语解释：

- **Comb peak**：GKP 态在相空间中周期出现的峰。
- **Envelope**：包络，用来限制无限能量。
- **Measurement inefficiency**：测量效率不足带来的额外噪声。
- **Ancilla state**：辅助态，用于 syndrome 测量或纠错电路。
- **Effective noise model**：对 decoder 有用的简化噪声模型。

### 段落 3：相关工作启发

口头解释：

> 这一段把项目接到已有文献。GKP 和 bosonic decoding 文献已经说明 analog information 有用。surface-code AI predecoder、FiLM decoder 和 prior optimization 文献说明，学习模块如果要进入 QEC 系统，最好是低时延、模块化、边界明确。

术语解释：

- **Surface-GKP**：把 GKP 作为内层物理编码，再与 surface code 结合。
- **Bosonic-QLDPC**：把 bosonic code 与量子 LDPC 外码结合。
- **FiLM**：Feature-wise Linear Modulation，用外部条件对 CNN 中间特征做缩放和平移。
- **Prior optimization**：优化 decoder 使用的噪声先验或权重。

### 段落 4：本项目如何应用这些原则

口头解释：

> 本项目把这些系统原则放到 GKP physical correction layer。每拍只执行 `Delta = K s + b`。慢回路从 histogram 估计噪声状态，再更新 `K,b`。CNN 不是从零学习完整纠错，而是被 teacher 限制在 residual calibration 上。

术语解释：

- **Physical correction layer**：物理层纠错，直接作用于 GKP 位移校正。
- **Histogram**：把一段时间内 syndrome 分布统计成二维网格。
- **Runtime parameter**：实际被快回路读取和执行的参数。
- **Residual calibration layer**：只校准基线之外残差的模块。

### 段落 5：为什么要收窄 claim

口头解释：

> 这一段很关键。我们不能说自己是第一个 analog GKP decoder、第一个 adaptive decoder、第一个 calibration-conditioned neural decoder 或第一个 FPGA QEC decoder。最安全也最准确的说法是：这是 teacher-anchored residual calibration for a deployment-constrained affine GKP fast path。

术语解释：

- **Claim scope**：论文主张范围。
- **Deployment-constrained**：受部署约束，包括时延、fixed-point、clip、commit、fallback。
- **Affine GKP fast path**：每拍执行 `K s + b` 的 GKP 快速路径。

---

## 4. Summary of Contributions

### 总体说明

口头解释：

> 贡献点现在分成五个，不只是“用了 CNN”。分别是：双时间尺度仿射 GKP 解码、teacher residual 校准、部署感知 runtime、paper-facing benchmark/ablation protocol，以及有界软件 HIL 证据协议。

### 贡献 1：双时间尺度自适应仿射形式

口头解释：

> 第一项贡献是把实时 GKP correction 写成 `Delta_t = K_t s_t + b_t`。快回路只做固定点矩阵乘加，慢回路负责根据 syndrome 统计更新参数。这把 GKP 连续 syndrome 理论和硬件运行契约连接起来。

术语解释：

- **Per-shot path**：每次 syndrome 到来时必须立即执行的路径。
- **Matrix-vector operation**：矩阵向量乘法，硬件实现相对简单。
- **Runtime contract**：运行时契约，明确输入、输出、更新时刻和安全约束。

### 贡献 2：teacher-anchored residual calibration

口头解释：

> 第二项贡献是学习策略。teacher 先给 `K_teacher,b_teacher`，CNN 只预测 `delta_b`。最后 `K` 保持 teacher，`b` 等于 teacher 的 b 加 residual 后再做 EMA。

术语解释：

- **`K_t^{teacher}`**：teacher 给出的线性增益矩阵。
- **`b_t^{teacher}`**：teacher 给出的偏置项。
- **`delta_b`**：CNN 对偏置项的残差修正。
- **EMA**：指数滑动平均，避免参数跳变。

### 贡献 3：deployment-aware runtime

口头解释：

> 第三项贡献是把 fixed-point、clip、saturation、stage-and-commit 等工程约束放进方法本身。这样论文不是先离线训练一个模型，再事后说可以部署，而是一开始就把部署约束作为评价对象。

术语解释：

- **Fixed-point**：定点数，用固定小数位表示实数，适合 FPGA。
- **Clipping**：裁剪，把数值限制在安全范围内。
- **Saturation**：饱和，输出撞到上下限。
- **Stage-and-commit**：先暂存参数，再在安全时刻提交。

### 贡献 4：benchmark 和 ablation protocol

口头解释：

> 第四项贡献是实验结构。参考 NVIDIA 那类 predecoder 论文，结果不应只报 LER，还要报 adaptation lag、saturation、overflow、fixed-point degradation、runtime boundary 等指标。T57 和 T64 可以成为这部分的种子证据。

术语解释：

- **Ablation**：消融实验，移除某个输入或模块，看性能变化。
- **Adaptation lag**：漂移发生后系统跟上的延迟。
- **Fixed-point degradation**：从浮点到定点后性能下降多少。
- **Runtime boundary**：运行路径真实到了哪一层，比如 Python、mock HIL、`.tflite`、真板。

### 贡献 5：bounded evidence protocol

口头解释：

> 第五项贡献是证据组织方式。T24 是历史冻结五模式表，T57 是 feature/teacher ablation，T64 是 statcalib extension lane。它们都可以服务论文，但都必须保留 mock-backed software-HIL 边界。

---

## 5. Brief Review of the GKP Code

### 理想和近似 GKP 态

口头解释：

> GKP 码把 qubit 编码成振子相空间里的周期结构。理想情况下逻辑态像无限梳子，但这需要无限能量。现实只能制备有限能量近似态，所以峰有宽度、有包络。这直接影响 syndrome 噪声，而不是一个可以忽略的实验细节。

术语解释：

- **Square-lattice GKP**：方格晶格 GKP 编码。
- **Lattice constant `lambda`**：晶格周期，本项目取 `sqrt(2*pi)`。
- **Logical state**：编码后的逻辑量子态。
- **Comb**：梳状结构，表示相空间中周期重复的峰。
- **Intrinsic syndrome uncertainty**：有限压缩本身造成的 syndrome 不确定性。

### 公式 `lambda = sqrt(2*pi)`

口头解释：

> 这个公式只是固定项目使用的尺度约定。后面取模区间、逻辑边界和 residual wrapping 都用这个尺度。

### 理想逻辑态公式

原文公式：

```tex
|\bar 0\rangle \propto \sum_{n\in\mathbb{Z}} |n\lambda\rangle_q
```

口头解释：

> 这表示逻辑 0 态在 q 方向上由很多等间距峰叠加而成。`n` 是整数，`n lambda` 是峰的位置。`propto` 表示正比，因为这里省略了归一化。

### Syndrome 是模位移信息

口头解释：

> 假设真实位移误差是 `e_t = [e_q,e_p]`。GKP syndrome 不是告诉我们绝对误差，而是告诉我们这个误差对晶格周期取模后的代表值。也就是说，我们只知道它落在基本区间的哪个位置。

术语解释：

- **q/p quadrature**：振子的两个正交连续变量。
- **Modulo**：取模，只保留周期内的代表值。
- **Fundamental cell**：基本晶胞，一个周期内的代表区域。
- **Noisy modular representative**：带噪声的取模代表值。

### 有限能量和测量噪声公式

口头解释：

> 新版 note 把测量 syndrome 写成 `mod(e_t,lambda) + eta_meas + eta_GKP`。这比理想公式更贴近现实：`eta_meas` 是测量和辅助态噪声，`eta_GKP` 是有限压缩 GKP 态本身带来的不确定性。

术语解释：

- **`eta_meas`**：测量噪声项。
- **`eta_GKP`**：有限能量 GKP 造成的等效噪声项。
- **Finite squeezing**：有限压缩，峰不够尖锐。

### 局部仿射解码和 linear-MMSE

口头解释：

> 精确 GKP 解码是非线性的，因为 syndrome 取模后存在多个 lattice branch。但如果只看一个局部分支，并且误差和 syndrome 可以近似成联合高斯分布，那么最优线性估计就是 `K s + b`。这就是 fast path 的理论来源。

术语解释：

- **Local branch**：某一个晶格分支附近的局部区域。
- **Jointly Gaussian**：联合高斯分布。
- **Linear-MMSE**：线性最小均方误差估计。
- **Posterior**：后验分布，观察到 syndrome 后误差可能是什么。

### 仿射近似的限制

口头解释：

> 在接近晶格判决边界时，后验可能有多个峰，一个全局 `K s + b` 会把多种可能平均掉，因此不是全局 Bayes 最优。论文中必须明确：我们追求的是部署可行的低时延近似，不是完整 ML/CVP GKP decoder。

术语解释：

- **Decision boundary**：判决边界，过界后会变成逻辑错误。
- **Multimodal posterior**：多峰后验。
- **Maximum-likelihood decoder**：最大似然 decoder。
- **Closest lattice point decoder**：最近晶格点 decoder。

### Logical failure criterion

口头解释：

> 纠错后还剩一个 residual displacement。如果这个 residual 超出基本区间的一半，就会导致逻辑错误。所以最终指标应该看 closed-loop logical error rate，而不是只看参数预测误差。

术语解释：

- **Residual displacement**：校正后剩下的位移。
- **Logical error**：编码后的逻辑信息发生错误。
- **LER**：Logical Error Rate，逻辑错误率。
- **`final_ler_mean`**：仓库当前固定协议下使用的 LER 类指标，越低越好。

---

## 6. Noise and Drift Model

### Effective noise state

口头解释：

> 慢回路不会试图恢复所有底层硬件参数，而是估计一个低维有效噪声状态：噪声尺度 `sigma`，q/p 偏置 `mu_q, mu_p`，以及协方差方向 `vartheta`。这四类量足以映射到 `K,b`，也比较容易从 histogram 估计。

术语解释：

- **Effective state**：有效状态，只保留对 decoder 有用的低维参数。
- **`sigma`**：噪声标准差或尺度。
- **`mu_q, mu_p`**：q/p 方向的均值偏置。
- **`vartheta`**：协方差主轴旋转角。

### Noise sources

口头解释：

> 当前模型可以吸收五类噪声：有限能量 GKP 宽度、Gaussian random displacement、均值偏置、各向异性或旋转协方差、测量与辅助态噪声。这不是完整 circuit-level GKP noise model，而是和当前 affine fast path 对齐的有效模型。

术语解释：

- **Gaussian random displacement**：高斯随机位移噪声。
- **Mean bias**：均值偏移，噪声分布中心不在零。
- **Anisotropic covariance**：各向异性协方差，q/p 方向噪声大小不同。
- **Circuit-level model**：显式建模门、测量、辅助态、传播错误的电路级模型。

### Drift scenarios

口头解释：

> 目前冻结协议有四个场景：`static_bias_theta`、`linear_ramp`、`step_sigma_theta`、`periodic_drift`。它们分别测试稳态偏置、缓慢跟踪、突变响应和周期跟踪。论文后续还要补随机游走、burst noise、低 shot histogram 等更难场景。

术语解释：

- **Static bias**：静态偏置。
- **Linear ramp**：线性缓慢变化。
- **Step drift**：阶跃突变。
- **Periodic drift**：周期性漂移。
- **Unseen drift family**：训练或调参时没见过的漂移类型。

---

## 7. Model Architecture

### Fast loop

口头解释：

> 快回路只做三件事：先 clip syndrome，再做 `K s + b`，最后 clip 和 quantize 输出。这样保证每拍执行路径非常简单，适合固定点硬件。

术语解释：

- **`s_max`**：syndrome 输入最大允许幅度。
- **`Delta_max`**：校正输出最大允许幅度。
- **`Q()`**：量化函数。
- **Q4.20**：项目使用的定点格式约定。
- **Overflow**：数值溢出。
- **Aggressive parameter event**：参数过激事件。

### Parameter mapping

口头解释：

> 参数映射器把噪声状态转成 `K,b`。先根据噪声尺度和旋转角构造协方差矩阵 `C`，再加上测量协方差 `R_meas`，得到类似 Kalman gain 的 `K_raw = C(C+R_meas)^{-1}`。如果 syndrome 可靠，K 更大；如果测量噪声大，K 更保守。

术语解释：

- **Covariance matrix**：协方差矩阵，描述误差椭圆的大小和方向。
- **Rotation matrix**：旋转矩阵，把主轴坐标转回 q/p 坐标。
- **Measurement covariance**：测量协方差。
- **Kalman gain**：决定多相信测量的增益。
- **Eigenvalue clipping**：限制矩阵增益的特征值范围。

### Bias target

口头解释：

> `b_target` 负责处理均值偏置。公式 `alpha(I-K)mu` 是当前项目 ParamMapper 的语义：如果噪声分布中心偏离零，仅靠 `K s` 不够，需要 `b` 纠正系统性偏移。

术语解释：

- **Bias term**：偏置项。
- **`alpha`**：偏置强度系数。
- **ParamMapper**：把噪声参数映射成运行时参数的模块。

### Teacher estimators

口头解释：

> teacher 是经典估计器家族。最简单的是窗口矩估计，复杂一些可以是 EKF、UKF、RLS 或粒子滤波。它先从 histogram 历史估计噪声状态，再通过 ParamMapper 得到 `K_teacher,b_teacher`。

术语解释：

- **Window moment**：窗口均值、方差、协方差等统计量。
- **EKF**：Extended Kalman Filter，扩展 Kalman 滤波。
- **UKF**：Unscented Kalman Filter，无迹 Kalman 滤波。
- **RLS**：Recursive Least Squares，递归最小二乘。
- **Particle filter**：粒子滤波，用样本近似后验。

### CNN residual branch

口头解释：

> CNN 输入最近几个 histogram 和 histogram delta。histogram 告诉它当前分布是什么样，delta 告诉它分布如何变化。它还可以看到少量 teacher-side scalar features。输出不是完整 `K,b`，而是 `delta_b`，并且会被 clip。

术语解释：

- **Context window**：最近多个 histogram 组成的上下文。
- **Histogram delta**：相邻 histogram 的差分。
- **Teacher-side features**：来自 teacher 的辅助特征。
- **Scalar feature**：标量特征。
- **Residual clip**：残差裁剪阈值。

### Stage-and-commit

口头解释：

> 慢回路不会直接改 active 参数，而是先写入 inactive bank，等安全 epoch 到来再切换。这避免快回路读到半更新参数。这个设计让论文可以报告 stale-parameter、commit latency、fallback 等系统指标。

术语解释：

- **Active bank**：当前快回路正在使用的参数 bank。
- **Inactive bank**：当前不被读取、可写入新参数的 bank。
- **Commit**：把 inactive bank 切换成 active bank。
- **Stale parameter**：参数更新滞后。
- **Fallback**：异常时退回安全模式。

---

## 8. Relationship to Existing Work

### GKP / bosonic analog soft information

口头解释：

> 这一节说明已有工作已经证明 analog syndrome 有用，例如 surface-GKP、bosonic-QLDPC、analog information decoding。因此本文不能说“首次使用 analog GKP 信息”。差异点是：我们把 analog histogram 压缩成物理层 affine fast path 的参数更新。

术语解释：

- **Outer-code prior**：外层码 decoder 使用的先验权重。
- **Matching weight**：MWPM 图边权重。
- **Belief propagation message**：BP 中的概率消息。

### Adaptive priors 和 syndrome-statistics estimation

口头解释：

> Spitz、Wagner、Chen、Sivak 等工作都说明 decoder prior 可以从 measured data 或 syndrome statistics 更新。因此本文也不能说“首次从 syndrome statistics 做适应”。我们的区别是更新目标是 GKP affine 参数 `K,b`，并且带 stage-and-commit runtime contract。

术语解释：

- **Adaptive weight estimator**：自适应权重估计器。
- **Syndrome-only noise estimation**：只从 syndrome 估计噪声。
- **Decoder prior optimization**：优化 decoder 先验，使 logical performance 更好。

### Learned low-latency QEC modules

口头解释：

> Chamberland 的 AI predecoder 和 Stein 的 FiLM decoder 是很近的学习型 QEC 系统工作。它们的启发是：学习模块要有清楚的系统角色和时延边界。我们的区别是 CNN 不在 per-shot path 里做完整 decoding，而是在慢回路更新 affine 参数。

术语解释：

- **Predecoder**：在全局 decoder 前先做局部预处理或降密度的模块。
- **Conditioned decoder**：被硬件标定或上下文条件调制的 decoder。
- **Per-shot compute**：每一拍必须执行的计算量。

### Real-time and FPGA QEC decoders

口头解释：

> 已有 surface-code 或 superconducting 系统中的 real-time FPGA decoder。因此本文不能说自己完成了 FPGA QEC decoder。当前只能说架构 FPGA-friendly，验证仍在 software-HIL 层。

术语解释：

- **Closed-loop timing**：从测量到反馈的闭环时延。
- **Worst-case latency**：最坏情况延迟，实时系统比平均延迟更关心它。
- **Resource use**：硬件资源，如 LUT、DSP、BRAM 等。

---

## 9. Numerical Results and Benchmark Plan

### Current bounded evidence

口头解释：

> 结果现在要分层讲。T24 是冻结五模式软件 HIL，`hybrid_residual_b` 四场景第一。T57 是 feature/teacher ablation，说明 histogram delta 很重要，但 teacher params 不是简单正贡献。T64 是新的 statcalib extension lane，结果很强，但必须单独标注。T65 只做一致性 guard。

术语解释：

- **Frozen five-mode matrix**：冻结的五个 mode 对比矩阵。
- **Extension lane**：额外补充的对比路线，不改写原冻结表。
- **`hybrid_residual_b`**：teacher + CNN residual-b 主线方法。
- **`statcalib`**：统计校准 comparator lane。

### T64 表格

口头解释：

> T64 的四个场景里，历史 frozen winner 都是 `hybrid_residual_b`，它们的 `final_ler_mean` 大约在 0.79 到 0.81。新增的 `statcalib` lane 在四个场景里分别是 0.431708、0.467083、0.460016、0.438751，明显更低。这个结果很值得重视，但当前解释只能是“有界 software-HIL extension lane 显示 statcalib 很强”，不能外推成真实部署或最终 SOTA。

术语解释：

- **Lower is better**：该指标越低越好。
- **Gap vs frozen winner**：相对历史冻结 winner 的差距。
- **Provenance-clean**：运行来源、commit、run root、summary 都能对上。
- **Anomalously strong**：结果强得需要进一步核查和机制解释，不能直接过度宣传。

### Recommended benchmark structure

口头解释：

> 正式论文不能只放一个表。应该像低时延 predecoder 论文一样，把 accuracy 和系统指标放在一起：LER、oracle regret、adaptation lag、saturation、overflow、commit latency、fixed-point vs float、fallback count 等。

术语解释：

- **Regret to oracle**：相对于知道真实噪声参数的 oracle decoder 损失多少。
- **Oracle affine**：用真实噪声状态构造的理想仿射 baseline。
- **Wrapped-Gaussian baseline**：考虑取模结构的高斯 decoder baseline。
- **Fallback count**：触发安全回退的次数。

### Comparator lanes

口头解释：

> 后续对比至少要包括 nearest-lattice/hard GKP、static affine、moment teacher、EKF/UKF/RLS、statcalib、direct CNN-to-parameters、teacher residual-b、FiLM-style conditioned affine head。这样才能说明本项目不是自娱自乐，而是在相关方法谱系里有位置。

术语解释：

- **Nearest-lattice decoder**：把 syndrome 映射到最近晶格点的硬判决 decoder。
- **Static affine**：固定 `K,b` 的仿射 decoder。
- **Direct CNN-to-parameters**：CNN 直接输出 `K,b`，不走 teacher residual。
- **FiLM-style affine head**：借鉴 calibration-conditioned FiLM，把条件信息用于调制 affine 参数头。

### Figure and table plan

口头解释：

> note 现在给出了论文图表路线：架构图、局部仿射理论图、T24 frozen benchmark、T57 ablation、T64 extension lane、runtime-contract table。后续填图时要保证每张图都对应一个明确 claim。

---

## 10. Writing Position and Near-Term Plan

### 安全论文 thesis

口头解释：

> 最安全的论文主线是：我们研究的是 deployment-constrained teacher-anchored residual calibration for drift-adaptive affine GKP decoding。快路径是 fixed-point affine correction，慢路径用 syndrome histogram 和 teacher/CNN 更新参数。

### 需要避免的说法

不要说：

- “第一个 adaptive neural QEC decoder。”
- “第一个 calibration-conditioned decoder。”
- “第一个使用 GKP analog soft information。”
- “已经 FPGA validated。”
- “residual-b amplitude 已经证明机制。”
- “T64 说明 statcalib 已经是最终 SOTA。”

推荐说：

- “当前结果支持 mock-backed software-HIL 下的 bounded revalidation。”
- “T64 是 separately labeled statcalib extension lane。”
- “fast path 是 FPGA-friendly，但真板验证仍是未来任务。”
- “机制解释仍为 partial，需要更多 ablation 和 unseen drift 测试。”

### T65 后的写法

口头解释：

> T65 完成后，note 应该把三层结果分开：T24 是历史冻结五模式表，T64 是第六条 statcalib extension lane，剩余 open items 是 `.tflite`、real-board、expanded benchmark、runtime evidence。这样论文不会混淆证据等级。

---

## 11. 术语速查表

| 术语 | 汇报时的简明解释 |
| --- | --- |
| GKP code | 把 qubit 编到振子的连续变量相空间里的量子纠错码 |
| Approximate GKP | 现实可制备的有限能量 GKP 态 |
| Bosonic code | 使用振子等玻色模式编码量子信息的纠错码 |
| Oscillator | 连续变量量子系统，可用 q/p 相空间坐标描述 |
| Phase space | 由 q 和 p 两个正交变量组成的空间 |
| Syndrome | 纠错测量得到的错误诊断信息 |
| Continuous syndrome | 连续值 syndrome，不是 0/1 |
| Modular measurement | 对晶格周期取模后的测量 |
| Displacement error | 相空间中的位移误差 |
| Finite energy | 有限能量，导致 GKP 峰有宽度 |
| Decoder | 根据 syndrome 决定如何校正的算法或系统 |
| Fast loop | 每个 syndrome 到来时快速执行校正 |
| Slow loop | 周期性估计噪声状态并更新参数 |
| FPGA | 适合低时延固定逻辑的可编程硬件 |
| Affine decoder | 形式为 `K s + b` 的解码器 |
| K | 线性增益矩阵 |
| b | 偏置项，用于处理均值漂移 |
| Teacher | 经典估计器，提供稳定 baseline |
| Residual | 基线之外的修正量 |
| Calibration | 根据当前硬件或噪声状态调整参数 |
| Histogram | syndrome 在一个窗口内的二维统计图 |
| LER | Logical Error Rate，逻辑错误率 |
| HIL | Hardware-in-the-loop；当前可信路径是 software HIL |
| Mock-backed | 由 mock 后端支撑，不是真实硬件 |
| `.tflite` | TensorFlow Lite 模型格式或 runtime 路径 |
| Fixed-point | 定点数，硬件友好的数值格式 |
| Q4.20 | 项目采用的一类定点格式约定 |
| Stage-and-commit | 先暂存参数，再在安全时刻提交 |
| Frozen set | 冻结的 benchmark 场景与 baseline 集合 |
| Extension lane | 单独追加的对比路线，不改写原冻结表 |
| Evidence boundary | 证据能支持到哪里，不能外推到哪里 |

---

## 12. 汇报顺序建议

建议按以下顺序讲：

1. **背景**：GKP 纠错根据连续 syndrome 校正位移误差。
2. **现实困难**：有限能量、测量噪声和硬件漂移会改变 syndrome 分布。
3. **核心设计**：快回路 `K s + b`，慢回路更新 `K,b`。
4. **理论来源**：局部 Gaussian / linear-MMSE 给出仿射形式。
5. **文献定位**：已有工作覆盖 analog soft information、adaptive prior、FiLM decoder、FPGA decoder；本项目定位在 GKP affine residual calibration。
6. **模型结构**：teacher、CNN residual、ParamMapper、stage-and-commit。
7. **当前结果**：T24、T57、T64 分层讲，T64 强但不越界。
8. **下一步**：T65 consistency guard、expanded benchmark、runtime evidence、真实 `.tflite` 和真板验证。

---

## 13. 可以直接使用的开场白

> 这份 note 的目标是把论文的理论和方法主线补起来。项目研究的是漂移环境下的 GKP 解码。GKP syndrome 是连续值，现实 GKP 态又是有限能量近似态，所以 syndrome 统计会受到噪声和硬件漂移影响。我的方法是快慢回路：快回路保持为 FPGA 友好的 `Delta = K s + b`，慢回路用 syndrome histogram、经典 teacher 和轻量 CNN 来更新参数。CNN 不是替代 decoder，而是学习 teacher 基线上的 bounded residual calibration。目前证据仍然限定在 mock-backed software HIL。T64 新增的 `statcalib` extension lane 结果很强，但仍要作为单独 lane 报告，不是真板或 `.tflite` 结果。

---

## 14. 老师可能会问的问题与建议回答

### Q1. 为什么一定要用 CNN？

建议回答：

> 不是一定要 CNN 替代经典方法。这里 CNN 的角色很窄：从 histogram 这种二维统计对象中学习 teacher 难以捕捉的残差结构。teacher 仍然提供稳定基线，所以 CNN 是校准层，不是完整 decoder。

### Q2. 为什么不用完整最优 GKP decoder？

建议回答：

> 完整最优 decoder 要处理 modulo 结构下的多分支后验，计算复杂，不适合每拍低时延路径。本文目标不是离线 Bayes 最优，而是在实时约束下做可部署的 adaptive approximation。

### Q3. 为什么当前主线只修正 b，不修正 K？

建议回答：

> `K` 控制整体增益和方向，改动风险更大；`b` 更直接对应均值漂移和偏置失配。当前主线先让 CNN 修正 `b`，是保守设计。是否扩展到 `K` 需要后续 ablation 证明。

### Q4. T64 的 statcalib 结果这么好，是否说明原模型不重要了？

建议回答：

> 不能这么直接判断。T64 说明在当前 mock-backed software-HIL 四场景 extension lane 中，`statcalib` 表现很强，值得重点跟进。但它不是成熟部署 comparator，也没有改写 T24 历史冻结表。更合理的结论是：统计校准 baseline 必须进入正式论文对比，并需要进一步核查泛化、机制和 runtime 边界。

### Q5. 当前结果能否说明已经可以上 FPGA？

建议回答：

> 不能。当前只能说 fast path 设计是 FPGA-friendly，并且 software-HIL 路径可跑。真实 `.tflite` runtime 和真板 HIL 仍然是后续任务。

### Q6. 这项工作和 FiLM decoder 或 AI predecoder 有什么区别？

建议回答：

> 它们证明了 learned module 可以在 QEC 中低时延工作，但本文不是把 CNN 放进每拍 decoder。我们的 CNN 在慢回路里更新低维 affine 参数，快回路仍是固定点 `K s + b`。同时应用对象是 GKP physical-layer displacement correction，而不是 surface-code 全局 decoding。

### Q7. 机制现在怎么讲？

建议回答：

> 机制只能谨慎讲。T55 以后不能说“高 committed-b 一定有害”。T57 也显示 teacher channels 的作用不是简单单调的。当前最安全说法是：residual branch 在某些 seed 和 scenario 下进入高幅度 regime，其效果依赖场景，需要进一步机制 ablation。

