# `CNN_FPGA_GKP_theory_note_draft.tex` 逐段口头汇报解释稿

本文用于口头汇报解释 `docs/follow-up_plan/CNN_FPGA_GKP_theory_note_draft.tex`。目标听众假设为：理解一般科研问题和工程系统，但不熟悉量子纠错、GKP 码或 FPGA/HIL 术语。

本文不是新的论文正文，不新增实验结论，不升级项目证据等级。所有涉及项目状态的表述都保持当前边界：

- 当前可信边界是 `mock-backed software HIL` 与 frozen-set software revalidation。
- 不声称真板 HIL 已完成。
- 不声称真实 `.tflite` runtime 已恢复。
- 不声称机制已经因果闭环。
- T55 之后，不能再把“高 committed-b 一般有害”写成稳定机制结论。

---

## 0. 汇报时的总叙事

### 一句话版本

这个项目研究的是：在 GKP 量子纠错中，把每一拍必须快速完成的纠错操作压缩成 FPGA 友好的线性公式，同时让一个较慢的 CNN/teacher 模块根据最近的 syndrome 统计来更新这个线性公式的参数，从而适应噪声漂移。

### 面向不懂量子纠错老师的 2 分钟版本

可以这样开场：

> 这个项目可以先不从“神经网络”讲起，而是从“实时纠错控制”讲起。GKP 码会把量子信息编码在一个振子的连续变量相空间里。错误表现为相空间中的小位移。我们每一轮能测到一个连续值 syndrome，它告诉我们误差大概落在晶格基本区间里的什么位置。问题是，硬件噪声会随时间漂移，所以固定的 syndrome-to-correction 规则会逐渐失配。我的方案是用一个快慢回路：快回路只执行非常简单的仿射校正 `Delta = K s + b`，适合 FPGA；慢回路用最近一段 syndrome histogram 和经典 teacher 来估计当前噪声状态，并让轻量 CNN 只学习一个 residual calibration，而不是替代整个 decoder。

### 这份 note 的核心判断

汇报时要强调三点：

1. **理论合理性**：`K s + b` 不是拍脑袋来的，它可以看成 GKP 位移估计在局部高斯近似下的 linear-MMSE 解码形式。
2. **工程合理性**：把复杂推理放到慢回路，把每拍执行路径压缩成固定点矩阵乘加，符合实时 QEC 的部署约束。
3. **证据边界**：当前只能说软件 HIL 和 frozen-set revalidation 支持这条路线，不能说真板完成、`.tflite` 完成或机制完全证明。

---

## 1. 逐段解释：标题与证据边界

### P0. 标题

原文标题：

> A Teacher-Anchored Residual Calibration Framework for Drift-Adaptive GKP Decoding

口头解释：

> 这个标题有三个关键词。第一是 teacher-anchored，意思是我们不是让 CNN 自己从头做解码，而是让一个经典估计器先给稳定基线。第二是 residual calibration，意思是 CNN 只学一个小的修正项。第三是 drift-adaptive GKP decoding，意思是目标场景不是静态噪声，而是噪声参数随时间漂移的 GKP 解码。

术语解释：

- **Teacher**：这里不是教师模型蒸馏里的泛泛 teacher，而是一个经典统计估计器，如 window variance、EKF、UKF。它提供稳定、可解释的 baseline。
- **Anchored**：表示 CNN 的输出被锚定在 teacher 结果附近，不允许完全自由地替代解码器。
- **Residual calibration**：残差校准。先有一个基础估计，再学习“还差多少”。
- **Drift-adaptive**：能适应漂移。漂移指噪声均值、方差、方向等随时间变化。
- **GKP decoding**：根据 GKP syndrome 判断应该施加什么相空间位移校正。

### P1. Evidence Boundary

原文含义：

这份 note 只是未来论文前半部分的理论草稿，不声称真板、真实 `.tflite` 或 paper-grade expanded benchmark 已完成。当前验证边界仍是 mock-backed software HIL。T55 后机制故事仍是 partial，不能写成因果闭环。

口头解释：

> 我先说明证据边界，避免把理论构想讲成已经完成的硬件系统。目前项目确实有恢复后的软件 HIL 和 frozen benchmark 证据，但它们是 mock-backed，也就是软件模拟后端，不是真板。`.tflite` 路径也还不能写成真实 runtime 已恢复。机制方面，T55 的干预结果显示简单的“高 committed-b 有害”解释不成立，所以机制还需要更谨慎表述。

术语解释：

- **Evidence boundary**：证据边界。哪些结论已经有证据支持，哪些还只是计划或理论解释。
- **HIL**：Hardware-in-the-loop。正常意思是硬件在环测试；当前项目里的可信路径是 software HIL，也就是保留硬件接口语义，但后端仍是 mock。
- **Mock-backed**：由 mock 后端支撑，不是真实硬件。
- **`.tflite` runtime**：TensorFlow Lite 推理运行时。当前有代码入口和 stub 路径，但当前机器没有完成真实 runtime 验证。
- **Paper-grade benchmark**：足以支撑投稿主张的扩展 benchmark，通常需要更强 baseline、更宽场景、多 seed/CI、可复现图表。
- **Causal closure**：机制因果闭环。不是只看到相关性，而是通过干预证明某机制导致某结果。

---

## 2. 逐段解释：Abstract

### P2. “GKP 纠错依赖连续 syndrome 来推断位移误差”

口头解释：

> GKP 码和普通离散 qubit 稳定子码不太一样。普通码里 syndrome 往往是 0/1，而 GKP syndrome 是连续值。它告诉我们振子在相空间里发生了多大的位移误差。纠错的目标就是根据这个 syndrome 估计误差，然后施加反向校正。

术语解释：

- **Approximate GKP**：近似 GKP。理想 GKP 态需要无限能量，现实中只能用有限能量近似态。
- **Syndrome**：综合征。纠错中用来诊断错误的信息。
- **Continuous-valued syndrome**：连续值综合征，不是二进制 0/1。
- **Oscillator**：振子，可以理解为一个连续变量量子系统。
- **Displacement error**：位移误差，相空间中 q/p 坐标发生偏移。

### P3. “非平稳硬件中噪声统计会漂移”

口头解释：

> 如果噪声分布永远不变，一个固定 decoder 也许够用。但真实硬件的噪声大小、偏置、相关方向都可能随时间变化。这样同一个 syndrome 在不同时间对应的最优校正可能不同，所以固定参数会失配。

术语解释：

- **Nonstationary**：非平稳，统计性质随时间变化。
- **Effective noise statistics**：有效噪声统计。这里不是底层所有物理噪声细节，而是对解码有用的低维量，比如方差、均值偏置、协方差方向。
- **Syndrome-to-correction map**：从 syndrome 到校正位移的映射。
- **Fixed decoder**：固定解码器，参数不随时间更新。

### P4. “提出双时间尺度框架”

口头解释：

> 我们把系统分成快慢两个回路。快回路每一拍都要低时延执行，所以只做简单公式；慢回路可以较慢地看最近一段 syndrome 统计，再更新快回路参数。这样兼顾了实时性和自适应性。

术语解释：

- **Two-timescale**：双时间尺度。快回路按微秒级运行，慢回路按毫秒级或窗口级运行。
- **Fast loop**：快回路，每个 syndrome 到来时立即给校正。
- **Slow loop**：慢回路，周期性更新 decoder 参数。
- **Affine correction**：仿射校正，形式是 `K s + b`，比纯线性多一个偏置项 `b`。
- **FPGA-friendly**：适合 FPGA 部署，通常意味着固定点、矩阵乘加、控制逻辑明确、时延可预算。

### P5. “快回路公式与慢回路 teacher”

口头解释：

> 快回路执行的是 `Delta_t = K_t s_t + b_t`。这里 `s_t` 是当前测到的 syndrome，`K_t` 和 `b_t` 是当前有效的 decoder 参数。慢回路并不让 CNN 完全自由输出 decoder，而是先用经典 teacher 得到稳定估计，再让 CNN 只预测残差校准项。

术语解释：

- **Runtime parameters**：运行时参数。实际被快回路读入并执行的参数。
- **Quantized**：量化。把浮点数变成固定点或整数格式，便于硬件执行。
- **Residual term**：残差项。基础估计之外的修正。
- **Calibration**：校准。根据当前系统状态修正参数。

### P6. “设计动机与贡献边界”

口头解释：

> 这个设计来自三方面动机：第一，GKP 位移估计在局部可以近似成 linear-MMSE，因此 `K s + b` 有理论来源；第二，近期 QEC 学习型 decoder 越来越倾向于模块化，而不是端到端替代全部结构；第三，实时路径必须低时延、确定。最后强调：本文不是做通用神经 GKP decoder，而是做一个部署约束下的 residual calibration layer。

术语解释：

- **Linear-MMSE**：线性最小均方误差估计。在线性估计器中，使均方误差最小的估计形式。
- **Modular learned decoder**：模块化 learned decoder。神经网络只做局部辅助，而不是替代整个解码链。
- **Deterministic low latency**：确定低时延。每次执行时间可控，不依赖不确定的复杂搜索。

---

## 3. 逐段解释：Introduction

### P7. “玻色码和 GKP 码的基本背景”

口头解释：

> 玻色码是把量子信息编码在振子这种连续变量系统里，而不是只用有限维 qubit。GKP 码是其中很重要的一类，它把逻辑信息放在相空间的周期结构中。小位移误差可以通过模 syndrome 测量诊断出来。

术语解释：

- **Bosonic code**：玻色码，利用振子等玻色模式编码量子信息。
- **Logical qubit**：逻辑比特，经过编码保护后的 qubit。
- **Hilbert space**：量子态所在的数学空间。
- **Phase space**：相空间，用 q/p 两个连续变量描述振子状态。
- **Modular syndrome measurement**：模 syndrome 测量，只测误差相对于晶格周期的位置。

### P8. “真实器件中 syndrome 分布会随时间变化”

口头解释：

> 理想条件下，syndrome 到 correction 的映射可以固定。但真实器件里，GKP 态不是无限精确的，测量也有噪声，ancilla 也会出错，硬件还会漂移。这些都会改变 syndrome 分布，所以 decoder 需要适应。

术语解释：

- **Finite-energy code states**：有限能量码态，现实可制备的 GKP 近似态。
- **Measurement inefficiency**：测量效率不足，会引入等效噪声。
- **Ancilla noise**：辅助系统噪声。纠错测量常需要辅助模式或辅助 qubit。
- **Calibration error**：标定误差，硬件参数与实际值不一致。

### P9. “高质量 decoder 与实时约束之间的张力”

口头解释：

> 解码器越想利用完整软信息和复杂噪声统计，计算就越重。但实时 QEC 要求很快，不能每一拍都跑复杂算法。这就是本文要解决的张力：如何把复杂适应性放到慢回路，而让快回路保持简单。

术语解释：

- **Analog soft information**：模拟软信息。不是只保留硬判决结果，而是保留连续值或置信度。
- **Concatenated-GKP**：把 GKP 码和外层稳定子码等结构级联。
- **Real-time QEC**：实时量子纠错，要求测量、解码、反馈都在硬件时间预算内完成。
- **Latency**：延迟，从输入到输出所需时间。
- **Determinism**：确定性，执行路径和时间可预测。

### P10. “相关工作给出的系统启发”

口头解释：

> 最近一些 surface-code AI predecoder、hardware-conditioned decoder、decoder prior optimization 的工作有一个共同启发：神经网络最好作为受约束的模块来帮助解码，而不是完全替代整个解码器。这和我们当前 teacher residual 的定位一致。

术语解释：

- **Surface code**：表面码，主流二维 qubit 量子纠错码。
- **AI pre-decoder**：预解码器，在完整 decoder 前先做局部提示或快速筛选。
- **Hardware-conditioned decoder**：把硬件标定信息作为 decoder 输入或条件。
- **Decoder prior optimization**：优化 decoder 的先验参数，使其更匹配硬件噪声。

### P11. “本项目的中心思想”

口头解释：

> 本项目就是把上述系统原则用于 GKP。快回路只执行仿射估计 `Delta = K s + b`。慢回路看最近 histogram，估计有效噪声状态，再更新 `K,b`。这不是让 CNN 从零学完整量子纠错规则，而是让它修正 teacher 的残差。

术语解释：

- **Histogram**：直方图。这里是把一段时间内的 syndrome 分布累积成 32×32 网格。
- **Effective noise state**：有效噪声状态，如 `sigma, mu_q, mu_p, theta`。
- **Double-buffered parameter bank**：双缓冲参数 bank，一边运行，一边写入新参数，到安全边界再切换。

### P12. “为什么主张要收窄”

口头解释：

> 我这里不想把工作写成“一个神经 GKP decoder”，因为这个说法太宽，也容易撞已有工作。更准确的说法是：这是一个 teacher-anchored residual calibration layer，用来更新一个受 runtime 约束的 affine GKP fast path。

术语解释：

- **Claim scope**：主张范围。论文声称解决了什么问题。
- **Runtime-constrained**：受运行时约束，包括时延、量化、参数提交、异常回退等约束。
- **Affine GKP fast path**：每一拍执行 `K s + b` 的 GKP 快速解码路径。

---

## 4. 逐段解释：Summary of Contributions

### P13. “贡献总体说明”

口头解释：

> 这一节不是列实验结果，而是列未来论文可以主打的理论和系统贡献。当前可以写四类：双时间尺度仿射形式、teacher residual 策略、部署感知 runtime、以及有边界的软件 HIL 证据协议。

术语解释：

- **Contribution**：论文贡献。不是单个代码功能，而是审稿人能识别的科学或工程增量。
- **Evidence protocol**：证据协议。说明结果如何得到、边界是什么、哪些不能外推。

### P14. 贡献 1：双时间尺度自适应仿射形式

口头解释：

> 第一项贡献是形式化。我们把实时纠错路径写成 `Delta = K s + b`，并允许 `K,b` 根据最近 syndrome 统计更新。这样理论上连到 GKP 连续 syndrome，工程上连到硬件里的矩阵乘加。

术语解释：

- **Affine estimator**：仿射估计器，输出等于矩阵乘输入再加偏置。
- **Per-shot path**：每一次 syndrome 到来时必须执行的路径。
- **Matrix-vector operation**：矩阵-向量运算，硬件上比较容易实现。

### P15. 贡献 2：teacher-anchored residual calibration

口头解释：

> 第二项贡献是学习策略。我们不让 CNN 直接输出所有参数，而是先由 teacher 输出稳定的 `K_teacher,b_teacher`，CNN 只预测 `delta_b`。最终执行时保留 teacher 的 `K`，只对 `b` 做 EMA 后的残差修正。

术语解释：

- **`K_t^{teacher}`**：teacher 给出的线性增益矩阵。
- **`b_t^{teacher}`**：teacher 给出的偏置项。
- **`delta_b`**：CNN 学到的偏置残差。
- **EMA**：指数滑动平均，用于防止参数跳变。
- **Full decoder replacement**：完整替代 decoder。本文明确不这么做。

### P16. 贡献 3：deployment-aware runtime

口头解释：

> 第三项贡献是工程结构。这个系统不是离线跑个模型分数，而是考虑 fixed-point、clip、saturation、parameter bank 和 stage-and-commit。这些都是未来部署时审稿人会关心的点。

术语解释：

- **Fixed-point**：定点数，用固定的小数位表示实数，适合 FPGA。
- **Clipping**：裁剪，把值限制在安全范围内。
- **Saturation diagnostics**：饱和诊断，记录数值是否撞到上下限。
- **Stale-parameter effect**：参数更新滞后造成的性能影响。
- **Commit behavior**：参数提交行为，新参数何时真正进入快回路。

### P17. 贡献 4：有界 software-HIL 评估协议

口头解释：

> 第四项贡献是证据组织方式。当前可以说我们有恢复后的 software-HIL 路径和 frozen-set revalidation。但必须明确它们只是软件 HIL，不是真板，不是真 `.tflite`，也不是最终 paper-grade expanded benchmark。完整论文还需要更强 comparator、未见 drift family 和 runtime evidence。

术语解释：

- **Frozen-set**：冻结的场景和 baseline 集合，用于可比复验。
- **Comparator lane**：对比方法路线，例如 statcalib、UKF、RLS、CNN-only 等。
- **Unseen drift family**：训练/调参时没见过的漂移类型，用来测试泛化。
- **Mechanism-hedged ablation**：带机制边界的消融。承认机制未闭环，不把 ablation 写成因果证明。

---

## 5. 逐段解释：GKP 码简述

### P18. “理想与近似码态”

口头解释：

> GKP 码把一个 qubit 编到振子的连续变量里。理想情况下，逻辑态是相空间中无限周期的梳状结构。但无限梳状态需要无限能量，现实中只能做有限能量近似，所以峰会变宽，有包络。这些近似会引入噪声底。

术语解释：

- **Ideal GKP state**：理想 GKP 态，数学上有无限尖锐的周期峰。
- **Approximate GKP state**：近似 GKP 态，有限能量、峰有宽度。
- **Comb**：梳状周期结构，表示相空间中周期重复的峰。
- **Envelope**：包络，限制高能成分，使状态可物理实现。
- **Noise floor**：噪声底，即使没有额外噪声也存在的基础不确定性。

### P19. “晶格常数 lambda”

口头解释：

> 项目采用的晶格常数是 `lambda = sqrt(2*pi)`。可以把它理解为 GKP 相空间格点之间的基本间距。后面的 syndrome 取模、逻辑边界和 residual wrap 都用这个尺度。

术语解释：

- **Lattice constant**：晶格常数，周期结构的间距。
- **`sqrt(2*pi)`**：本项目采用的相空间尺度约定。
- **Fundamental cell**：基本晶胞，一个周期内的代表区域。

### P20. “理想逻辑态公式”

原文公式：

$$
|\bar 0\rangle \propto \sum_{n\in\mathbb{Z}} |n\lambda\rangle_q .
$$

口头解释：

> 这个公式只是说明理想逻辑零态在 q 方向上是很多等间距峰的叠加。`n` 是整数，`n lambda` 是每个峰的位置。`propto` 表示正比，因为这里省略了归一化。

术语解释：

- **`|\bar 0\rangle`**：编码后的逻辑 0 态。
- **`|n\lambda\rangle_q`**：q 位置本征态，位置在 `n lambda`。
- **`propto`**：正比。
- **`n in Z`**：n 是整数。

### P21. “为什么有限能量结构重要”

口头解释：

> 有限能量不是一个实验小瑕疵，而是直接影响 decoder 的输入分布。因为峰变宽后，syndrome 本身就有不确定性。后面我们用 histogram 和噪声参数来建模，就是在处理这种现实条件下的统计变化。

术语解释：

- **Decoder input distribution**：解码器看到的输入统计分布。
- **Syndrome statistics**：syndrome 在窗口内的统计形状，比如均值、方差、相关方向。

### P22. “syndrome 测量是模位移信息”

口头解释：

> 假设纠错前的位移误差是 `e_t = [e_q, e_p]`。GKP syndrome 不告诉我们绝对误差在哪，而是告诉我们误差对晶格周期取模后的位置。这和“只知道它在基本区间里的代表值”类似。

术语解释：

- **q/p quadrature**：振子的两个正交连续变量，可理解为相空间横纵坐标。
- **Modulo**：取模，只保留相对于周期的位置。
- **Representative**：代表元，同一等价类中选出的基本区间值。

### P23. “理想 syndrome 公式”

原文公式：

$$
s_t = e_t \bmod \lambda,
\qquad
s_{q,t},s_{p,t}\in[-\lambda/2,\lambda/2).
$$

口头解释：

> 这个公式说 syndrome 是误差对 `lambda` 取模后的值，并被映射到 `[-lambda/2, lambda/2)` 这个基本区间里。也就是说，decoder 每一拍看到的是折回来的误差，而不是原始累计误差。

术语解释：

- **`s_t`**：当前 syndrome。
- **`e_t`**：当前累计位移误差。
- **`[-lambda/2, lambda/2)`**：对称基本区间。

### P24. “真实测量带噪”

原文公式：

$$
\tilde{s}_t = {\rm mod}(e_t,\lambda)+\eta_t^{\rm meas}.
$$

口头解释：

> 真实 syndrome 等于理想取模结果加上测量噪声。这里的测量噪声包括有限压缩、探测效率不足、shot noise 和 ancilla noise 等。这个公式提醒我们：decoder 面对的是一个带噪的模代表。

术语解释：

- **`tilde{s}`**：带噪测量值。
- **`eta^{meas}`**：测量噪声项。
- **Shot noise**：测量中的随机涨落。
- **Finite squeezing**：有限压缩，GKP 峰无法无限尖锐。

### P25. “局部仿射解码”

口头解释：

> 精确 GKP 解码是非线性的，因为取模会带来多个可能分支。但如果误差比较小、暂时只看某个局部分支，就可以把误差和 syndrome 近似成联合高斯。在线性估计器里，最优形式就是 `K s + b`。

术语解释：

- **Branch**：分支。由于取模，同一个 syndrome 可能对应多个原始误差位置。
- **Jointly Gaussian**：联合高斯，多个变量整体服从高斯分布。
- **Posterior**：后验分布，看到 syndrome 后误差可能值的概率分布。
- **Multimodal**：多峰分布，表示有多个可能分支。

### P26. “linear-MMSE 公式”

原文公式：

$$
\hat e = \mu_e+\Sigma_{es}\Sigma_{ss}^{-1}(s-\mu_s)=Ks+b.
$$

口头解释：

> 这个公式是理论核心。它说明在局部高斯近似下，最优线性误差估计可以写成 `K s + b`。`K` 来自误差和 syndrome 的协方差关系，`b` 来自均值偏移。因此项目里的快回路公式不是任意设计，而是有统计估计来源。

术语解释：

- **`hat e`**：估计出的误差。
- **`mu_e, mu_s`**：误差和 syndrome 的均值。
- **`Sigma_es`**：误差与 syndrome 的交叉协方差。
- **`Sigma_ss`**：syndrome 自身协方差。
- **Covariance**：协方差，描述变量如何共同变化。

### P27. “仿射近似的局限”

口头解释：

> 这个 affine decoder 的局限也必须讲清楚。在格点边界附近，真实后验可能有多个峰，单个 `K s + b` 会把多分支平均掉，所以它不是 Bayes 最优 decoder。我们的目标是低时延自适应近似，而不是离线最优解码。

术语解释：

- **Bayes-optimal**：贝叶斯最优，完整利用后验概率的理论最优。
- **Decision boundary**：判决边界，超过后可能导致逻辑错误。
- **Low-latency adaptive approximation**：低时延自适应近似，用工程可执行的简化形式接近较优解。

### P28. “逻辑失败判据”

口头解释：

> 纠错不是看单次估计误差是不是小，而是看多轮后残差有没有跨过逻辑边界。如果 q 方向残差超过 `lambda/2`，就会出现逻辑 X 错误；p 方向类似对应 Z 错误。所以闭环 LER 比离线 MSE 更重要。

术语解释：

- **Residual displacement**：残余位移，校正后还剩下的误差。
- **Logical error**：逻辑错误，编码后的量子信息发生不可恢复错误。
- **LER**：Logical Error Rate，逻辑错误率。
- **Offline regression error**：离线回归误差，比如参数预测 MSE，不一定等价于闭环纠错性能。

---

## 6. 逐段解释：Model Architecture

### P29. “快回路接收 syndrome 并读 active bank”

口头解释：

> 快回路的任务很窄：每来一个 syndrome，就从当前 active bank 读取 `K,b`，做一次固定点仿射计算，输出校正。它不运行 CNN，也不做复杂搜索。

术语解释：

- **Active bank**：当前正在被快回路读取的参数 bank。
- **Parameter bank**：参数存储区，保存 `K,b`。
- **Fast path**：实时执行路径。

### P30. “clip + quantize 的快回路公式”

原文公式：

$$
s_t^{\rm clip}={\rm clip}(s_t,-s_{\max},s_{\max}),
$$

$$
\Delta_t^{\rm raw}=K_t s_t^{\rm clip}+b_t,
$$

$$
\Delta_t=Q({\rm clip}(\Delta_t^{\rm raw},-\Delta_{\max},\Delta_{\max})).
$$

口头解释：

> 这三步对应真实硬件会做的事情。先把 syndrome 限制在安全范围，再做矩阵乘加得到原始校正，再把校正值限制范围并量化成 fixed-point。这样避免数值爆掉，也让硬件实现可控。

术语解释：

- **`s_max`**：syndrome 输入允许的最大幅度。
- **`Delta_max`**：校正输出允许的最大幅度。
- **`Q()`**：量化函数。
- **Raw correction**：裁剪和量化前的校正值。

### P31. “Q4.20 与诊断量”

口头解释：

> Q4.20 是 fixed-point 格式，表示总共给整数和小数分配固定比特位。它比浮点更适合 FPGA。快回路还会记录输入 histogram 是否饱和、校正是否饱和、参数是否过激，这些诊断量用于判断系统是否稳定。

术语解释：

- **Q4.20**：定点格式，通常表示 4 位整数相关范围和 20 位小数精度的约定。
- **Histogram-input saturation**：syndrome 输入超出 histogram 表示范围。
- **Correction saturation**：校正输出撞到上下限。
- **Aggressive-parameter event**：参数过激事件。

### P32. “有效噪声状态”

口头解释：

> 慢回路不用建模所有底层物理噪声，而是用一个低维有效状态描述当前噪声：总噪声尺度 sigma、q/p 方向均值偏置 mu_q 和 mu_p、协方差方向 theta。这样既有物理含义，也方便映射成 `K,b`。

术语解释：

- **`sigma`**：噪声尺度或标准差。
- **`mu_q, mu_p`**：q/p 方向的均值偏置。
- **`theta` 或 `vartheta`**：协方差主轴旋转角。
- **Low-dimensional effective state**：低维有效状态，用少数参数描述对解码最重要的噪声特征。

### P33. “协方差构造”

口头解释：

> 根据 sigma 和 theta，可以构造一个误差协方差矩阵 C。先在主轴坐标下写成对角矩阵，再旋转回实验室坐标。这个 C 描述误差在 q/p 空间中大小和方向如何分布。

术语解释：

- **Covariance matrix**：协方差矩阵，描述二维误差分布的形状和方向。
- **Principal axes**：主轴方向，即误差椭圆的长短轴方向。
- **Rotation matrix `R(theta)`**：旋转矩阵，把主轴系转到实验室坐标系。
- **Lab frame**：实验室坐标系，代码中 q/p 坐标所在的固定坐标。

### P34. “测量协方差”

原文公式：

$$
R_{\rm meas}=(\sigma_{\rm meas}^2+\Delta_{\rm eff}^2)I.
$$

口头解释：

> `R_meas` 表示测量本身的不确定性。它由测量噪声和有效 GKP 宽度共同决定。直观上，如果测量很不准，decoder 就不能完全相信 syndrome，因此 `K` 应该更保守。

术语解释：

- **Measurement covariance**：测量协方差。
- **`I`**：单位矩阵，表示 q/p 两个方向暂时按相同测量噪声处理。
- **`Delta_eff`**：有效 GKP 宽度或有限能量带来的额外不确定性。

### P35. “K_raw 公式”

原文公式：

$$
K_{\rm raw}=C(C+R_{\rm meas})^{-1}.
$$

口头解释：

> 这个公式和 Wiener filter / Kalman gain 的直觉一致：如果真实误差方差 C 大，而测量噪声小，就更相信 syndrome，K 更大；如果测量噪声大，就更保守。之后还要做裁剪，避免硬件控制过激。

术语解释：

- **Wiener filter**：经典线性最优滤波器。
- **Kalman gain**：Kalman 滤波中的增益，决定多相信测量。
- **Eigenvalue clipping**：特征值裁剪，限制矩阵增益范围。
- **Gain scaling**：整体增益缩放。

### P36. “b_target 公式”

原文公式：

$$
b_{\rm target}=\alpha(I-K_{\rm target})\mu.
$$

口头解释：

> `b` 处理的是均值偏置。如果噪声分布中心不在零点，仅靠 `K s` 不够，还要加一个偏置项。这里用 `(I-K) mu` 是当前主线的参数映射语义，不能随意改成简单的 `-mu`。

术语解释：

- **Bias term `b`**：偏置项，处理系统性均值偏移。
- **`alpha`**：偏置强度系数。
- **`I-K`**：单位映射和增益矩阵之间的差。
- **ParamMapper**：把噪声参数映射成运行时 `K,b` 的模块。

### P37. “指数平滑”

口头解释：

> 即使慢回路每次估计出新参数，也不能让快回路参数剧烈跳变，所以要用指数平滑。新参数只占一部分权重，旧参数保留一部分。这样能降低抖动。

术语解释：

- **Exponential smoothing**：指数平滑。
- **`beta`**：更新权重，越大表示越快跟随新估计。
- **Staged runtime parameters**：已经准备提交、但不一定立刻成为 active 的运行时参数。

### P38. “Teacher 估计器”

口头解释：

> teacher 是一类经典估计器，从 syndrome 历史里估计当前噪声状态。最简单的是从 histogram 计算均值和协方差；更强的如 EKF/UKF/RLS/PF，会考虑时间上的状态演化。

术语解释：

- **Window variance**：窗口方差法，从当前窗口估计统计量。
- **EKF**：Extended Kalman Filter，扩展 Kalman 滤波。
- **UKF**：Unscented Kalman Filter，无迹 Kalman 滤波。
- **RLS**：Recursive Least Squares，递归最小二乘。
- **Particle filter**：粒子滤波，用样本表示后验分布。

### P39. “Teacher 抽象公式”

原文公式：

$$
\hat{\theta}_t^{\rm teacher} = {\rm Teacher}(H_{1:t}),
$$

$$
(K_t^{\rm teacher},b_t^{\rm teacher}) = {\rm ParamMapper}(\hat{\theta}_t^{\rm teacher}).
$$

口头解释：

> teacher 先从历史 histogram 得到噪声参数估计，再通过 ParamMapper 转成快回路真正执行的 `K,b`。这一步保证了 teacher 输出和 runtime 控制语义是一致的。

术语解释：

- **`H_{1:t}`**：从第一个窗口到当前窗口的 histogram 历史。
- **`theta_hat`**：估计出的噪声状态。
- **Runtime control semantics**：运行时控制语义，也就是最终快回路实际执行的参数含义。

### P40. “teacher 不是修辞”

口头解释：

> 这里特别强调 teacher 不是写论文时的漂亮词，而是系统里的实际稳定锚点。没有 teacher，CNN 容易变成无约束回归器；离线指标好，不一定闭环好。

术语解释：

- **Stability anchor**：稳定锚点，保证模型输出有可靠基线。
- **Unconstrained decoder**：无约束 decoder，输出不受物理结构或运行时边界约束。
- **Offline metric vs closed-loop metric**：离线指标和闭环指标不等价。

### P41. “CNN residual branch 的输入”

口头解释：

> CNN 看的是最近几个窗口的 histogram 以及窗口之间的变化量。当前窗口告诉它噪声分布是什么样，histogram delta 告诉它分布如何变化。

术语解释：

- **Context window**：上下文窗口，最近多个 histogram。
- **Histogram delta**：相邻 histogram 的差分。
- **`X_t^{hist}`**：CNN 的 histogram 输入特征。

### P42. “teacher-side features”

口头解释：

> 除了 histogram，CNN 还可以看到 teacher 侧特征，比如 teacher 估计出的噪声参数、teacher 的 `K,b`、以及 teacher 参数变化量。当前 gated 分支把这些信息收窄成少量 `teacher_b` 和 `teacher_delta_b` 标量，避免把太多冗余 teacher 信息广播进 CNN。

术语解释：

- **Teacher-side features**：来自 teacher 的辅助输入。
- **Scalar feature**：标量特征，不是图像通道。
- **Broadcast feature**：把标量铺成整张平面作为 CNN 通道。
- **Gated branch**：门控分支，用少量标量调制 CNN hidden state。

### P43. “CNN 输出 delta_b”

口头解释：

> CNN 预测的不是完整 `K,b`，而是 `delta_b`，也就是对 teacher 偏置项的修正。然后这个修正还会被 clip 到安全范围内。这样 CNN 的自由度被限制住，更符合部署稳定性。

术语解释：

- **`delta_b`**：偏置残差。
- **`b_max`**：残差允许的最大幅度。
- **`residual_scale_b`**：残差缩放系数。
- **`residual_clip_b`**：残差裁剪阈值。

### P44. “最终组合公式”

原文公式：

$$
K_t=K_t^{\rm teacher},
\qquad
b_t={\rm EMA}(b_t^{\rm teacher}+\delta b_t).
$$

口头解释：

> 最终组合很简单：K 完全沿用 teacher，b 则等于 teacher 的 b 加上 CNN 残差，再做平滑。这就是主线 `Hybrid Residual-B` 的真正数学语义。

术语解释：

- **Hybrid Residual-B**：混合 residual-b 方案。hybrid 指 teacher + CNN，residual-b 指只修正 b。
- **Committed parameter**：实际提交到快回路的参数。

### P45. “为什么 residual formulation 重要”

口头解释：

> 这个残差设计让 CNN 负责一个小而明确的部署相关修正，而不是承担整个 GKP 解码问题。这样更容易解释，也更容易控制风险。

术语解释：

- **Deployment-relevant correction**：与部署实际效果相关的修正。
- **Risk control**：风险控制，这里指限制模型输出自由度，避免闭环失稳。

### P46. “stage-and-commit 契约”

口头解释：

> 慢回路不会直接改 active 参数，而是先写到 inactive bank，等到安全 epoch 再切换。这避免了快回路执行到一半时参数被改掉。

术语解释：

- **Stage**：暂存新参数。
- **Commit**：在安全时刻切换到新参数。
- **Inactive bank**：当前不被快回路读取的参数 bank。
- **Epoch boundary**：预定义的周期边界。

### P47. “分段公式”

原文公式：

$$
(K_t,b_t)=
\begin{cases}
(K^A,b^A), & t<t_{\rm commit},\\
(K^B,b^B), & t\ge t_{\rm commit}.
\end{cases}
$$

口头解释：

> 这个公式表示提交前快回路用 A bank，提交后用 B bank。它不是连续混合，而是一次原子切换。这个细节对硬件系统很重要。

术语解释：

- **Atomic switch**：原子切换，切换过程不暴露中间状态。
- **No-glitch update**：无毛刺更新，不让快回路看到半写入参数。

### P48. “为什么 stage-and-commit 支撑部署叙事”

口头解释：

> 有了这个契约，论文就不仅能评价 LER，还能评价 update latency、stale parameter、commit 成功率和 fallback 行为。这些都是系统型论文的重要指标。

术语解释：

- **Rollback**：回滚到旧参数。
- **Fallback**：失败时退回安全模式。
- **System metric**：系统指标，不只是算法精度，还包括时延、稳定性、资源、异常处理。

---

## 7. 逐段解释：当前证据状态与近期计划

### P49. “当前安全结果”

口头解释：

> 当前可以安全写的是：仓库支持恢复后的 mock-backed software-HIL 路径，并且在 frozen 四场景、五模式的 software revalidation 中，`hybrid_residual_b` 在这个 frozen set 内排第一。但这个结论不能外推到真 `.tflite`、真板或更宽 SOTA。

术语解释：

- **Frozen four-scenario, five-mode**：已经冻结的四个场景和五个 mode 对比，不是任意扩展 benchmark。
- **SOTA**：state of the art，领域最优水平。当前不能这么写。
- **Generalization beyond frozen set**：超出冻结集合的泛化，当前仍需要额外证据。

### P50. “机制故事要谨慎”

口头解释：

> 早期 trace 说明 committed-b 幅度和 residual clip 可能参与了 seed-dependent 行为。但 T55 把 clip 从 0.12 降到 0.06 后，结果是 4/6 seed 变差、2/6 变好，所以不能说“大的 committed-b 就是坏的”。更准确的是：learned residual branch 会进入某些高幅度 regime，但这种 regime 的效果是 seed 和场景相关的。

术语解释：

- **Committed-b**：已经提交并被快回路实际使用的 b 参数。
- **Seed-dependent**：依赖随机种子，不同 seed 下现象不同。
- **High-amplitude regime**：高幅度输出区间。
- **Monotonic explanation**：单调解释，如“越大越坏”或“越小越好”。T55 表明这种解释不成立。

### P51. “完整投稿前的补强”

口头解释：

> 如果目标是正式投稿，还需要四类补强：一是 benchmark 扩展但保留 frozen anchor；二是机制表，把哪些假说保留、削弱或废弃讲清楚；三是真 `.tflite` runtime；四是 fixed-point 和 latency 证据。

术语解释：

- **Frozen-set anchor**：保留现有 frozen benchmark 作为可比锚点。
- **Mechanism table**：机制假说状态表。
- **Runtime-boundary evidence**：运行时边界证据，说明真实部署路径到哪一步。
- **Measured constraints**：测量到的约束，而不是只在文字中假设。

---

## 8. 术语速查表

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
| Decoder | 根据 syndrome 决定如何校正的算法或系统 |
| Fast loop | 每个 syndrome 到来时快速执行校正 |
| Slow loop | 周期性估计噪声状态并更新参数 |
| FPGA | 适合低时延固定逻辑的可编程硬件 |
| Affine decoder | 形式为 `K s + b` 的解码器 |
| K | 线性增益矩阵 |
| b | 偏置项，用于处理均值漂移 |
| Teacher | 经典估计器，提供稳定 baseline |
| Residual | 基线之外的修正量 |
| Calibration | 根据当前硬件/噪声状态调整参数 |
| Histogram | syndrome 在一个窗口内的二维统计图 |
| LER | Logical Error Rate，逻辑错误率 |
| HIL | Hardware-in-the-loop；当前可信路径是 software HIL |
| Mock-backed | 由 mock 后端支撑，不是真实硬件 |
| `.tflite` | TensorFlow Lite 模型格式或 runtime 路径 |
| Fixed-point | 定点数，硬件友好的数值格式 |
| Q4.20 | 项目采用的一类定点格式约定 |
| Stage-and-commit | 先暂存参数，再在安全时刻提交 |
| Frozen set | 冻结的 benchmark 场景与 baseline 集合 |
| Evidence boundary | 证据能支持到哪里，不能外推到哪里 |

---

## 9. 汇报时建议采用的顺序

建议按以下顺序讲，不要直接从公式开始：

1. **问题背景**：GKP 纠错要根据连续 syndrome 校正位移误差。
2. **现实困难**：硬件噪声会漂移，固定 decoder 会失配。
3. **核心设计**：快回路 `K s + b`，慢回路更新 `K,b`。
4. **为什么合理**：局部 Gaussian / linear-MMSE 给出 `K s + b` 的理论来源。
5. **为什么不是普通 CNN decoder**：CNN 只学 teacher residual，不替代完整解码器。
6. **为什么适合工程部署**：fixed-point、stage-and-commit、diagnostics、software HIL。
7. **证据边界**：当前是 mock-backed software HIL，不是真板，不是真 `.tflite`。
8. **下一步**：更强 benchmark、机制表、runtime evidence、fixed-point/latency 证据。

---

## 10. 可以直接使用的开场白

> 我这份 note 想先把论文的理论部分补起来。它的重点不是宣称实验已经完全结束，而是把主线方法讲清楚：我们研究的是漂移环境下的 GKP 解码。GKP syndrome 是连续值，硬件噪声会变，所以固定 decoder 会失配。我的方案是一个快慢回路系统：快回路保持为 FPGA 友好的 `Delta = K s + b`，慢回路通过 syndrome histogram、经典 teacher 和轻量 CNN 来更新参数。CNN 不是替代 decoder，而是只学习 teacher 基线上的 residual calibration。当前证据仍然限定在 mock-backed software HIL 和 frozen-set revalidation，后续还要补更强 benchmark、机制解释和 runtime 边界。

---

## 11. 老师可能会问的问题与建议回答

### Q1. 为什么一定要用 CNN？

建议回答：

> 不是一定要 CNN 替代经典方法。这里 CNN 的角色很窄：从 histogram 这种二维统计对象中学习 teacher 难以捕捉的残差结构。我们仍保留 classical teacher 作为稳定基线，所以 CNN 是校准层，不是完整 decoder。

### Q2. 为什么不用完整最优 GKP decoder？

建议回答：

> 完整最优 decoder 需要处理 modulo 后的多分支后验，计算更复杂，不适合每拍低时延硬件路径。本文目标不是离线 Bayes 最优，而是在实时约束下做可部署的 adaptive approximation。

### Q3. 为什么只修正 b，不修正 K？

建议回答：

> 这是当前主线的保守设计。`K` 控制整体增益和方向，改动风险更大；`b` 更直接对应均值漂移和偏置失配。teacher 负责稳定的 `K,b` 基线，CNN 先只修正 `b`，更容易控制闭环稳定性。后续是否扩展到 `K` 需要单独 ablation 支持。

### Q4. 当前结果能否说明已经可以上 FPGA？

建议回答：

> 不能这样说。当前只能说 fast path 设计是 FPGA-friendly，并且 software HIL 路径已恢复。真板 HIL 和真实 `.tflite` runtime 都还没有完成。论文中必须把这三层分开写。

### Q5. T55 之后机制怎么讲？

建议回答：

> T55 表明简单的“高 committed-b 有害”不成立。更准确的说法是，learned residual branch 会出现 seed/scenario-dependent 的高幅度 regime，它有时带来收益，有时带来不稳定。机制现在是 partial，需要用 hedge wording。

### Q6. 这项工作和已有 neural decoder 有什么区别？

建议回答：

> 区别不在于“用了神经网络”，而在于系统定位。这里不是端到端神经 GKP decoder，而是 teacher-anchored residual calibration for an affine fast path。也就是说，神经网络只更新低维运行时参数，快回路保持硬件友好的 `K s + b`。

---

## 12. 汇报时需要避免的表述

不要说：

- “我们已经完成 FPGA 真板验证。”
- “`.tflite` 部署已经恢复。”
- “这个方法已经是 GKP 解码 SOTA。”
- “机制已经证明了。”
- “高 committed-b 一定有害。”
- “CNN 取代了 GKP decoder。”

推荐说：

- “当前证据支持 mock-backed software HIL 下的 bounded revalidation。”
- “该方法是面向部署约束的 teacher residual calibration。”
- “当前机制证据仍为 partial，T55 后需要更谨慎的机制表述。”
- “fast loop 是 FPGA-friendly，但 real-board validation 仍是后续任务。”
- “后续要补 expanded benchmark、runtime evidence 和机制 ablation。”

