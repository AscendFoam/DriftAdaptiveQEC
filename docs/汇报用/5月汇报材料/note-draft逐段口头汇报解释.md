# `CNN_FPGA_GKP_theory_note_draft.tex` 逐段口头汇报解释

本文用于口头解释 `docs/follow-up_plan/CNN_FPGA_GKP_theory_note_draft.tex`。目标听众可以假设为：了解科研问题和工程系统，但不熟悉量子纠错、GKP 码、bosonic code、FPGA 解码器或 HIL 验证。

当前 note 的题目是：

> Teacher-Anchored and Statistical Calibration for Drift-Adaptive Affine GKP Decoding

中文可以先解释成：

> 面向漂移噪声的 GKP 仿射解码：用经典 teacher 和统计校准来更新一个适合硬件执行的快速纠错规则。

汇报时要把主线讲清楚：这不是“直接用 CNN 替代量子纠错 decoder”，而是把每一拍必须快速执行的 GKP 校正压缩成一个简单的仿射公式 `Delta = K s + b`，再用较慢的校准回路根据最近的 syndrome 分布更新 `K` 和 `b`。这样既保留 GKP 连续 syndrome 的物理信息，又让实时路径保持可预测、可量化、适合硬件实现。

---

## 0. 汇报总叙事

### 一句话版本

这个项目研究的是：在近似 GKP 码纠错中，如何用 recent syndrome histogram 估计慢变噪声状态，并把这种估计转化为硬件友好的仿射校正规则 `Delta_t = K_t s_t + b_t`，从而让低延迟 GKP 解码器能够适应噪声漂移。

### 2 分钟开场版本

可以这样开场：

> GKP 码是一类 bosonic quantum error-correcting code。它把一个逻辑 qubit 编码到振子的连续变量相空间里。错误可以直观理解成相空间中的小位移。每轮纠错会测到一个连续值 syndrome，它告诉我们位移误差落在 GKP 晶格周期内的大概位置。困难在于，真实 GKP 态不是理想无限能量态，而是有限能量的近似态；再加上测量噪声、辅助态噪声、电路误差和硬件标定漂移，syndrome 的统计分布会随时间变化。固定 decoder 在一个工作点调得很好，但漂移后就会失配。
>
> 我的思路是把 decoder 分成两个时间尺度：快回路每一拍只执行 `Delta_t = K_t s_t + b_t`，这个公式是固定点硬件友好的；慢回路每隔一个窗口统计最近 syndrome histogram，估计当前有效噪声状态，并更新 `K_t,b_t`。CNN 如果使用，也不是端到端替代 decoder，而是在经典 teacher 给出的参数附近学习一个有界残差。最近的数值结果还显示，一个非神经网络的 statistical calibration 分支在同样运行契约下非常强，因此论文故事应该从“CNN decoder”调整成“histogram-driven adaptive affine calibration for GKP fast path”。

### 这版 note 的核心判断

口头汇报时要抓住五点：

1. **理论来源**：`Delta = K s + b` 可以看作局部高斯近似下的 linear-MMSE 位移估计。
2. **物理对象**：GKP syndrome 是连续值和模格点信息，不是普通 stabilizer code 里的 0/1 syndrome。
3. **工程约束**：复杂估计放在慢回路；每拍执行路径保持固定点矩阵乘加、裁剪和参数 bank commit。
4. **文献定位**：已有工作覆盖了 analog soft information、adaptive priors、calibration-conditioned neural decoder 和 real-time FPGA decoder。本文的差异是把这些思想收窄到 GKP physical-layer affine fast path 的漂移校准。
5. **结果叙事**：现有结果不应该只讲“CNN 有效”。更强的叙事是：在同一个仿射 runtime contract 内，teacher residual 和 statistical calibration 都支持“低维 histogram-driven calibration”这一主线。

---

## 1. Title、Abstract 和 Table of Contents

### 标题：Teacher-Anchored and Statistical Calibration for Drift-Adaptive Affine GKP Decoding

口头解释：

> 标题里有四个关键词。`Teacher-Anchored` 表示学习模块不自由接管 decoder，而是围绕一个经典 teacher 估计器给出的小修正。`Statistical Calibration` 表示我们也研究不用神经网络、直接从 syndrome 统计量估计偏置修正的路径。`Drift-Adaptive` 表示目标噪声不是固定的，而是会随时间漂移。`Affine GKP Decoding` 表示实时校正规则被限制为 `K s + b` 这样的仿射形式。

专业词解释：

- **Teacher**：经典估计器，例如 moment estimator、EKF、UKF、RLS 或其它状态估计方法。它给出稳定、可解释的参数基线。
- **Anchored**：锚定。神经网络输出围绕 teacher 结果做小幅修正，而不是完全自由输出校正。
- **Statistical calibration**：统计校准。根据最近 syndrome 的均值、分布形状或 histogram 特征直接修正参数。
- **Drift-adaptive**：适应漂移。噪声的均值、方差或相关方向会随时间变化，decoder 需要跟踪。
- **Affine decoding**：仿射解码。输出形式是线性项 `K s` 加偏置项 `b`。

### Abstract 第一段：问题对象和近似 GKP 背景

原文第一段说：近似 GKP 量子纠错把 qubit 编码进振子，使用连续值 modular syndrome 推断 displacement error；真实有限能量 GKP 态的 syndrome 统计受制备宽度、测量噪声、电路误差和慢标定漂移影响。

口头解释：

> 摘要开头先交代为什么这个问题不是普通分类任务。GKP syndrome 是连续值，所以它携带比二值 syndrome 更多的信息；但真实 GKP 态是有限能量近似态，峰有宽度，测量也有噪声。因此 syndrome 分布本身会反映硬件状态。如果硬件状态漂移，那么 decoder 应该跟着更新，而不是长期使用固定映射。

专业词解释：

- **Approximate GKP**：近似 GKP。理想 GKP 态需要无限能量，真实实验只能制备有限能量近似态。
- **Finite-energy**：有限能量。对应相空间中峰不是无穷尖，而是有宽度和 envelope。
- **Continuous-valued modular syndrome**：连续值模 syndrome。它不是 0/1，而是落在一个晶格周期内的连续坐标。
- **Displacement error**：位移误差。振子相空间中沿 `q` 或 `p` 方向发生的小偏移。
- **Calibration drift**：标定漂移。硬件参数或有效噪声统计随时间缓慢变化。

### Abstract 第二段：双时间尺度架构

原文第二段说：快回路执行 `Delta_t = K_t s_t + b_t`，慢回路从 recent syndrome histograms 估计当前有效噪声状态并更新 runtime parameters。

口头解释：

> 这段给出本文方法的系统结构。关键是分离“每一拍必须快”和“估计噪声可以慢”这两个需求。快回路只做固定点仿射计算，延迟可控；慢回路看一段时间的 syndrome histogram，估计当前噪声状态，再把新的 `K,b` 写入运行参数。

专业词解释：

- **Two-timescale**：双时间尺度。快回路每拍执行；慢回路按窗口或周期更新参数。
- **Fast loop**：快回路。直接根据当前 syndrome 输出校正量。
- **Slow calibration loop**：慢校准回路。统计历史 syndrome，估计噪声，更新参数。
- **Runtime parameters**：运行时参数。实际被快回路读取和执行的 `K,b`。
- **Fixed-point hardware execution**：固定点硬件执行。使用固定小数位整数表示实数，适合 FPGA。

### Abstract 第三段：learning 的窄角色

原文第三段强调：classical teacher 提供可解释基线，lightweight CNN 只预测有界 residual calibration，当前聚焦 bias `b_t`；这不同于端到端 neural decoder。

口头解释：

> 这段要避免老师误解为“直接把神经网络放进实时量子纠错”。我们的学习模块很窄：teacher 给出 `K,b` 的主估计，CNN 只修正偏置 `b` 的小残差，而且要 clip 和 EMA。换句话说，实时 decoder 仍然是确定性仿射规则，学习只负责慢速校准。

专业词解释：

- **Residual calibration**：残差校准。先有基线估计，再学习“还差多少”。
- **Bias `b_t`**：仿射公式中的偏置项，表示 syndrome 为零时仍需要的校正偏移。
- **End-to-end neural decoder**：端到端神经 decoder，从 syndrome 直接输出最终纠错决策。
- **Outer-code soft-information decoder**：外层码软信息 decoder，通常把 GKP 连续信息转换成 surface code 或 LDPC decoder 的权重/消息。

### Abstract 第四段：实验结论的正确讲法

原文最后说：四个漂移场景的软件 HIL 实验显示 adaptive affine calibration 优于 filtering baselines，histogram-delta features 对 learned residual branch 重要，statistical calibration 在同一漂移 suite 上显著降低 logical-error proxy，local sensitivity 说明优势不依赖单个窄参数点。

口头解释：

> 这里的重点是结果叙事的升级。不是简单说 CNN 赢了，而是说 adaptive affine calibration 这个 runtime contract 是有价值的。CNN residual branch 在五种 affine 模式里表现最好；但 statistical calibration 在同样的四个场景中更强。这说明项目的核心贡献应该放在“用 syndrome 统计驱动低维 affine 参数校准”，CNN 是其中一种实现路径。

专业词解释：

- **Software-HIL**：软件 HIL。保留硬件接口、参数 bank 和 commit 语义，但底层用软件模拟执行。
- **Logical-error proxy**：逻辑错误代理指标。这里主要是 `final_ler_mean`，越低表示逻辑失败越少。
- **Histogram-delta feature**：相邻 syndrome histogram 的差分特征，用来捕捉分布随时间变化。
- **Local sensitivity**：局部敏感性。围绕默认参数小范围改变 scale、clip、threshold，检查结论是否稳定。

### Table of Contents

口头解释：

> 目录说明这份 note 已经接近正式论文结构。它不是只写 abstract 和 introduction，而是把 GKP 背景、噪声漂移模型、模型结构、相关工作、实验设置、数值结果、未来 benchmark 缺口和 conclusion 放进同一条逻辑链里。

---

## 2. Introduction

### 第一段：Bosonic code 和 GKP code

原文说：bosonic codes 通过把 logical qubit 编码到 oscillator 的 Hilbert space 保护量子信息；GKP code 是核心例子，理想态在 phase space 中形成周期 lattice，小位移错误可通过 modular syndrome measurement 诊断并用 displacement operation 校正。

口头解释：

> 先从量子纠错的直觉讲。普通 qubit 错误常被看作离散错误，而 GKP 用的是振子的连续变量。理想 GKP 态像相空间中的周期性晶格，如果错误只是小位移，就可以通过测量它在一个晶格周期内的位置来判断需要推回多少。

专业词解释：

- **Bosonic code**：玻色码，用振子模式承载逻辑量子信息。
- **Oscillator**：振子，例如微波腔或机械/光学模式，可用连续变量描述。
- **Hilbert space**：量子态所在的数学空间。
- **Phase space**：相空间，由 `q` 和 `p` 两个正交连续变量组成。
- **Lattice**：晶格，GKP 态在相空间中周期重复的结构。
- **Modular syndrome measurement**：模 syndrome 测量，只保留误差在一个周期内的位置。

### 第二段：近似 GKP 的实际困难

原文说：physical GKP states 是 approximate finite-energy states；comb peaks 有有限宽度和 envelope；syndrome 还受测量低效、noisy auxiliary states、电路误差、oscillator loss 和 calibration error 影响；因此 decoder 看到的 effective noise model 会随时间变化。

口头解释：

> 这段是问题动机。理想 GKP 是数学理想化，真实状态的峰不可能无限尖，所以 syndrome 天生带不确定性。硬件还会引入额外噪声和漂移。于是同一个 syndrome 值，在不同时间可能对应不同的最优校正，这就要求 decoder 能适应漂移。

专业词解释：

- **Comb peak**：梳状峰，GKP 态在相空间中周期性出现的峰。
- **Envelope**：包络，用来限制无限 comb 的能量。
- **Measurement inefficiency**：测量效率不足导致的额外噪声。
- **Auxiliary state / ancilla**：辅助态，用于 syndrome 测量或纠错电路。
- **Oscillator loss**：振子能量损耗，会造成相空间位移和退相干。
- **Effective noise model**：有效噪声模型，不追踪所有硬件细节，只保留 decoder 需要的统计特征。

### 第三段：已有工作给出的启发

原文说：GKP 和 bosonic decoding 文献说明 analog soft information 有价值；surface-GKP 和 bosonic-QLDPC decoders 会用连续 syndrome 信息设置 matching weights、belief-propagation messages 或 outer-code priors；现代 QEC decoder 系统说明 learned modules 最有价值时往往有明确 latency role 和 integration boundary。

口头解释：

> 这段把本文放到已有研究版图里。别人已经证明连续 syndrome 不能简单硬判决掉，它对外层码解码很有用。同时，像 AI pre-decoder、FiLM decoder 这样的工作说明，学习模块进入量子纠错系统时，必须说明它在哪个环节运行、延迟是多少、和经典 decoder 怎么接起来。本文继承这个系统思想，但把目标放在 GKP 物理层校正，而不是外层 surface code 或 LDPC 消息。

专业词解释：

- **Analog soft information**：模拟软信息。保留连续测量值或置信度，而不是只转成硬判决。
- **Surface-GKP**：把 GKP 作为物理层或内层编码，再与 surface code 组合。
- **Bosonic-QLDPC**：把 bosonic code 与量子 LDPC 外层码组合。
- **Matching weights**：匹配 decoder 中边的权重，通常由错误概率决定。
- **Belief propagation**：置信传播，一类基于概率消息传递的解码方法。
- **Integration boundary**：系统集成边界，即一个模块的输入、输出、延迟和职责。

### 第四段：本文的系统原则

原文说：本文把系统原则应用到 physical GKP correction layer；快回路是 `Delta_t = K_t s_t + b_t`；慢回路观察 syndrome histograms，估计 effective noise state，并低频更新 `K,b`。

口头解释：

> 这里给出本文区别于外层软信息 decoder 的地方。我们不把连续 syndrome 只用于外层码的权重，而是在 GKP 物理层直接更新位移校正规则。公式里的 `s_t` 是当前 syndrome，`K_t,b_t` 是当前运行参数。快回路每次只做一个小矩阵乘加；慢回路异步估计噪声并更新参数。

专业词解释：

- **Physical GKP correction layer**：GKP 物理层校正，直接决定施加多少相空间位移。
- **Syndrome histogram**：把一段时间的 syndrome 样本统计成二维分布图。
- **Asynchronous update**：异步更新。参数估计不阻塞每一拍实时校正。
- **Calibrated parameter update**：经过校准约束后的参数更新，通常包括 clip、smooth 和 commit。

### 第五段：项目优势的叙事

原文说：main advantage 是 physical structure、adaptation 和 deployment discipline 的组合；相比 fixed affine decoder 可以跟踪漂移；相比 full neural decoder 在线计算更小、更可预测；相比 outer-code soft-information methods，适应目标是 physical-layer GKP displacement rule；相比 calibration-conditioned neural decoders，slow context 不扩大 per-shot path。

口头解释：

> 这一段是“为什么有价值”的核心。它不是宣称比所有 decoder 都强，而是指出一个中间位置：固定仿射 decoder 太僵硬，端到端神经 decoder 太难放进确定性低延迟路径；本文保留简单硬件路径，同时让参数能随噪声漂移调整。这个定位对审稿人更可信。

建议口头强调四个对比：

- **对比固定 decoder**：固定 `K,b` 不能追踪 bias、variance 或 covariance rotation 漂移。
- **对比完整神经 decoder**：本文实时路径更小、更确定，方便 fixed-point 和 FPGA。
- **对比外层软信息方法**：本文校准的是 GKP 位移校正参数，不是 matching graph 或 LDPC message。
- **对比 calibration-conditioned neural decoder**：本文 slow context 只更新参数 bank，不增加每拍计算图。

---

## 3. Summary of Contributions

### 总体讲法

原文把贡献组织成四点。汇报时不要只说“用了 CNN”，而要把贡献讲成“理论形式 + 校准策略 + runtime contract + benchmark suite”。

### 贡献 1：双时间尺度自适应仿射 GKP 解码

原文说：per-shot correction path 是 bounded affine estimator `Delta_t = K_t s_t + b_t`；昂贵的 adaptation problem 移到 slow loop；这把 continuous-syndrome GKP picture 和 quantized matrix-vector runtime contract 连接起来。

口头解释：

> 第一项贡献是把问题形式化。每拍只做 `K s + b`，这个规则来自 GKP 局部高斯近似，也适合固定点矩阵乘加。复杂的噪声估计不放在实时路径里，而是放进慢回路。这是理论和工程之间的接口。

专业词解释：

- **Per-shot correction path**：每次 syndrome 到来后必须立即执行的路径。
- **Bounded estimator**：有界估计器，输出被 clip 到安全范围。
- **Quantized matrix-vector operation**：量化矩阵向量运算，使用固定点数实现。

### 贡献 2：Teacher-anchored residual calibration

原文说：classical teacher 产生稳定 baseline `(K_teacher,b_teacher)`；CNN 预测小 residual `delta b_t`；最终 `K_t=K_teacher`，`b_t=EMA(b_teacher+delta b_t)`；learning 用来校准低维控制面，而不是端到端替代 decoder。

口头解释：

> 第二项贡献是学习模块的约束方式。teacher 先给出可解释基线，CNN 只学习偏置项还差的一小段，并且经过 clip 和 EMA。这样神经网络不会直接输出任意校正量，降低失控风险，也方便做 ablation。

专业词解释：

- **`K_t^{teacher}`**：teacher 给出的线性增益矩阵。
- **`b_t^{teacher}`**：teacher 给出的偏置向量。
- **`delta b_t`**：CNN 对偏置的残差修正。
- **EMA**：指数滑动平均，用来平滑参数更新。
- **Control surface**：控制面，可以理解为低维参数空间中的校准曲面。

### 贡献 3：Deployment-aware runtime architecture

原文说：架构暴露 fixed-point quantization、clipping、saturation diagnostics、parameter smoothing、stale-parameter behavior 和 double-buffered stage-and-commit updates。这些约束成为 benchmark 可测对象，而不是算法完成后再临时加入。

口头解释：

> 第三项贡献是工程约束前置。我们不只是离线训练一个模型，而是从一开始就问：这个参数怎么量化？输出如果太大怎么办？参数什么时候 commit？旧参数延迟更新会损失多少？这些都是未来投稿时能增强说服力的工程指标。

专业词解释：

- **Fixed-point quantization**：固定点量化，把实数映射到有限小数位表示。
- **Clipping**：裁剪，把值限制在上下界内。
- **Saturation**：饱和，运算结果撞到上限或下限。
- **Stale parameter**：滞后参数，噪声已变但快回路还在使用旧参数。
- **Double-buffered stage-and-commit**：双缓冲暂存与提交。一个 bank 正在执行，另一个 bank 准备新参数，到安全边界再切换。

### 贡献 4：Adaptive affine calibration benchmark suite

原文说：数值研究比较 filtering baselines、learned residual variants 和 statistical calibration variants，分离三个问题：adaptive affine 是否优于 filtering baseline，learned residual 哪些特征重要，简单 statistical calibration 是否能作为强非神经 comparator。

口头解释：

> 第四项贡献是实验组织方式。它不是只给一个最终分数，而是用同一 drift suite 分别比较过滤器、CNN residual 和 statistical calibration。这样可以回答“是神经网络本身有效，还是 histogram-driven calibration 这个结构有效”。现有结果倾向于后者更重要。

专业词解释：

- **Filtering baseline**：滤波基线，例如 EKF、UKF、RLS，用状态空间方式估计噪声。
- **Ablation**：消融实验，去掉某个输入或模块看性能变化。
- **Comparator**：对比方法，用来判断主方法是否真的有优势。

---

## 4. Brief Review of the GKP Code

### 4.1 Ideal and approximate code states

原文说：square-lattice GKP code 通过 `q,p` quadratures 的周期结构编码 qubit；晶格常数 `lambda = sqrt(2 pi)`；理想 logical state 可写作无限 comb；理想态不物理，因为需要无限能量；approximate GKP 用 finitely squeezed peaks 和 envelope 替代理想尖峰。

口头解释：

> GKP 码的直觉是把逻辑 0 和 1 编码成相空间中的周期性峰列。理想情况下，峰无限尖、周期无限延伸，所以可以精确知道错误相对于晶格的位置。但这需要无限能量，不可能实现。真实 approximate GKP 的峰有宽度，这个宽度本身就是 syndrome 不确定性的来源。

公式解释：

- `lambda = sqrt(2*pi)`：本文采用的晶格周期约定。
- `|bar 0> proportional sum_n |n lambda>_q`：逻辑 0 可以直观看成 q 方向上间隔为 `lambda` 的无限峰列。

专业词解释：

- **Square-lattice GKP**：方格晶格 GKP 编码，q 和 p 方向都有周期结构。
- **Quadrature**：正交连续变量，通常记为 q 和 p。
- **Logical state**：编码后的逻辑量子态。
- **Finite squeezing**：有限压缩，峰不能无限尖。
- **Intrinsic syndrome uncertainty**：由近似 GKP 态本身带来的 syndrome 不确定性。

### 4.2 Syndrome measurement as modular displacement information

原文说：累积位移误差是 `e_t=[e_q,t,e_p,t]^T`；理想 syndrome 观测 `e_t mod lambda`；真实测量可写成 `mod(e_t,lambda)+eta_meas+eta_GKP`。

口头解释：

> GKP syndrome 不告诉我们绝对位移是多少，只告诉我们位移相对于晶格周期的余数。可以类比钟表：如果只看时针位置，只知道在 12 小时周期内的位置，不知道已经转了几圈。真实 syndrome 还叠加了测量噪声和近似 GKP 峰宽带来的噪声。

专业词解释：

- **Modulo / mod**：取模，只保留周期内代表值。
- **Fundamental cell**：基本晶胞，一个周期内的代表区域。
- **`eta_meas`**：测量和辅助态引入的噪声项。
- **`eta_GKP`**：有限能量 GKP 态自身引入的等效噪声项。
- **Noisy modular representative**：带噪声的模代表值。

### 4.3 Local affine decoding and its limits

原文说：GKP modulo structure 让精确解码非线性且依赖 lattice branch；但在单个 branch 附近，如果 `e` 和 `s` 联合高斯，linear-MMSE 给出 `hat e = K s + b`，其中 `K=Sigma_es Sigma_ss^{-1}`，`b=mu_e-K mu_s`。靠近 decision boundaries 时 posterior 可能多峰，ML、closest-lattice-point 或 wrapped-Gaussian decoder 会优于单个全局仿射规则。

口头解释：

> 这段是 fast path 的理论依据。严格说 GKP 解码是非线性的，因为取模后可能对应多个晶格分支。但如果误差主要落在某个局部分支附近，而且可以近似为高斯分布，那么最优线性估计正好是 `K s + b`。所以仿射解码不是随便选的工程公式，而是一个局部统计近似。
>
> 同时要承认它的限制：当 syndrome 接近晶格决策边界时，后验分布可能有多个峰。一个全局仿射公式可能会把多个候选平均掉，不可能总是 Bayes 最优。

专业词解释：

- **Linear-MMSE**：线性最小均方误差估计，在高斯近似下给出最优线性估计。
- **Covariance**：协方差，描述不同变量如何一起变化。
- **Posterior**：后验分布，观察 syndrome 后位移误差的可能分布。
- **Multimodal posterior**：多峰后验，表示存在多个可能晶格分支。
- **Maximum-likelihood decoder**：最大似然 decoder，选择概率最大的候选。
- **Closest-lattice-point decoder**：最近晶格点 decoder，选择最近的晶格分支。
- **Wrapped-Gaussian decoder**：考虑取模周期性的高斯 decoder。

### 4.4 Logical failure criterion

原文说：校正后 residual displacement 会 wrap 到 GKP fundamental cell；如果 residual 越过逻辑决策边界，例如 `|r_q,t| > lambda/2`，则出现 `X_L` 逻辑错误；因此 closed-loop logical error probability 是中心指标。

口头解释：

> 纠错最终关心的不是 `K,b` 是否预测得像 teacher，而是校正后逻辑信息是否出错。只要 residual displacement 还在安全区间内，逻辑态就没有被翻转；一旦越过半个晶格周期，就会落到错误的逻辑分支。因此结果部分使用 logical-error proxy，而不是单纯参数回归误差。

专业词解释：

- **Residual displacement**：校正后剩余位移。
- **Logical error**：编码后的逻辑 qubit 发生错误，例如逻辑 X 或逻辑 Z 错误。
- **Closed-loop**：闭环。decoder 的输出会影响下一步系统状态或最终 logical failure。
- **LER**：Logical Error Rate，逻辑错误率，越低越好。

---

## 5. Noise and Drift Model

### 5.1 Effective noise state

原文说：slow loop 不估计所有微观硬件参数，而使用低维 effective state `theta_noise=(sigma,mu_q,mu_p,vartheta)`，其中 `sigma` 控制 displacement scale，`mu_q,mu_p` 表示 mean bias，`vartheta` 捕捉 covariance-axis rotation。

口头解释：

> 这里要说明为什么不用完整硬件模型。真实硬件噪声来源很多，如果全部建模会过于复杂，也不一定能从有限 syndrome 窗口中稳定估计。因此本文只保留对仿射校正最有用的低维状态：噪声大小、q/p 均值偏置、协方差主轴方向。

专业词解释：

- **Effective state**：有效状态，只保留与 decoder 相关的压缩变量。
- **`sigma`**：噪声尺度或标准差。
- **`mu_q, mu_p`**：q、p 方向的均值偏置。
- **`vartheta`**：协方差主轴旋转角。
- **Covariance-axis rotation**：噪声椭圆方向发生旋转，表示 q/p 不再完全独立。

### 5.2 Noise sources represented by the model

原文列出五类噪声：finite-energy GKP peak width and envelope、Gaussian random displacement、biased displacement means、anisotropic or rotated covariance、syndrome measurement noise and noisy auxiliary-state contributions。

口头解释：

> 这个模型不是完整 circuit-level GKP noise model，而是 effective displacement-noise model。它把多个物理来源压缩成位移噪声、均值偏置、各向异性和测量不确定性。这样做的原因是 runtime fast path 只需要知道如何设置 `K,b`，不需要恢复每个底层硬件参数。

专业词解释：

- **Gaussian random displacement**：高斯随机位移噪声。
- **Mean bias**：均值偏移，噪声中心不在零点。
- **Anisotropic covariance**：各向异性协方差，q/p 或旋转方向上的噪声大小不同。
- **Circuit-level noise model**：电路级噪声模型，显式建模门、测量、辅助态和传播错误。

### 5.3 Drift scenarios

原文使用四个漂移场景：`static_bias_theta`、`linear_ramp`、`step_sigma_theta`、`periodic_drift`。它们分别测试 steady calibration、tracking、shock response 和 periodic following。

口头解释：

> 四个场景的作用是让 benchmark 不只覆盖一种漂移。静态偏置测试校准偏置工作点；线性 ramp 测试能否缓慢跟踪；step 测试突变后的响应；periodic drift 测试周期性非平稳噪声。这些场景还不等于完整部署环境，但可以构成第一组受控对比。

专业词解释：

- **Static bias**：静态偏置，噪声中心和方向固定但不理想。
- **Linear ramp**：线性漂移，噪声参数缓慢连续变化。
- **Step drift**：阶跃漂移，噪声大小或方向突然改变。
- **Periodic drift**：周期漂移，噪声参数按周期变化。
- **Nonstationary noise**：非平稳噪声，统计分布随时间变化。

---

## 6. Model Architecture

### 6.1 Fast loop: affine fixed-point decoder

原文说：fast loop 接收 syndrome `s_t`，读取 active parameter bank，先 clip syndrome，再计算 `Delta_raw=K_t s_clip + b_t`，再对输出 clip 并量化 `Q(.)`。实现使用 FPGA-oriented Q4.20 表示，并记录 saturation、overflow、commit、fallback 等诊断计数。

口头解释：

> 快回路就是实时 decoder。它不运行 CNN，也不做复杂优化，只做三步：输入 syndrome 裁剪、矩阵乘加得到 raw correction、输出裁剪并量化成固定点。这样每一拍的计算量和最坏情况延迟都比较清楚。

专业词解释：

- **Active parameter bank**：当前正在被快回路使用的参数存储区。
- **Q4.20**：一种固定点格式，通常表示若干整数位加 20 个小数位。
- **Overflow**：溢出，计算超出表示范围。
- **Diagnostic counter**：诊断计数器，用来记录饱和、溢出、commit 等事件。
- **Fallback signal**：回退信号，例如参数异常时使用保守参数。

### 6.2 Parameter mapping

原文说：根据估计 noise state 构造 error covariance `C` 和 measurement covariance `R_meas`；raw gain 是 `K_raw=C(C+R_meas)^{-1}`；bias target 是 `b_target=alpha(I-K_target)mu`；最后用 exponential smoothing 产生 staged parameters。

口头解释：

> 这部分解释 teacher 或 statistical calibration 估计出的噪声状态如何变成 `K,b`。如果误差协方差大、测量比较可信，那么 `K` 会更相信 syndrome；如果测量噪声大，`K` 会更保守。`b` 则主要补偿均值偏置。最后用 smoothing 防止参数跳变。

专业词解释：

- **Error covariance `C`**：位移误差的协方差矩阵。
- **Measurement covariance `R_meas`**：测量不确定性的协方差。
- **Gain `K`**：增益矩阵，决定 syndrome 被转换成校正量的比例和方向。
- **Bias target `b_target`**：偏置项目标，用于补偿均值误差。
- **Exponential smoothing**：指数平滑，新旧参数按比例融合。
- **Staged parameters**：暂存参数，先准备好但不一定立即生效。

### 6.3 Teacher estimators

原文说：teacher family 从 recent syndrome history 估计 `theta_noise`。简单 teacher 计算 histogram window moments；更强 teacher 如 EKF、UKF、RLS 或 particle-filter variants 加入时序状态假设。teacher 让 learned branch 接近可解释 calibration surface，也产生有意义的 ablation baselines。

口头解释：

> Teacher 可以理解为“经典控制/统计估计器”。它不需要训练，或者训练很少，主要根据最近观测估计当前噪声状态。它的价值有两个：一是给 CNN 一个合理基线，避免神经网络从零学；二是作为对比基线，检查 CNN 或 statistical calibration 到底比经典估计好在哪里。

专业词解释：

- **EKF**：Extended Kalman Filter，扩展卡尔曼滤波，用线性化处理非线性系统。
- **UKF**：Unscented Kalman Filter，无迹卡尔曼滤波，用 sigma points 传播均值和协方差。
- **RLS**：Recursive Least Squares，递归最小二乘，常用于在线估计参数。
- **Particle filter**：粒子滤波，用一组采样粒子表示状态分布。
- **Moment estimator**：矩估计器，用均值、方差等统计矩估计参数。

### 6.4 CNN residual branch

原文说：CNN 输入短上下文的 normalized syndrome histograms 和 histogram deltas；teacher-side features 可包含 `theta_teacher,K_teacher,b_teacher,Delta b_teacher` 等；gated branch 缩窄到 teacher-b 和 teacher-Delta-b 标量；learned branch 输出 `delta b`，再 clip；最终 `K=K_teacher`，`b=EMA(b_teacher+delta b)`。

口头解释：

> CNN 分支看的不是单个 syndrome，而是一段时间的 histogram 和 histogram 的变化。这样它可以捕捉漂移趋势。它输出的是偏置残差 `delta b`，不是完整校正量。这个设计让学习模块职责很窄：只在 teacher 的基础上微调 bias。

专业词解释：

- **Histogram context**：多个时间窗口的 syndrome 统计图。
- **Histogram delta**：相邻 histogram 的差，表示分布如何变化。
- **Teacher-side feature**：teacher 估计出的参数或参数变化，作为网络输入。
- **Gated branch**：门控分支，只允许一部分 teacher 特征进入模型。
- **Residual scale `s_b`**：残差缩放因子，控制 CNN 修正幅度。
- **`b_max`**：bias residual 的裁剪上限。

### 6.5 Statistical calibration branch

原文说：同一个 affine runtime contract 也支持非神经 statistical calibration branch。该分支直接从 recent syndrome statistics 估计 bounded bias correction，并通过同样的 clipping、smoothing、stage-and-commit 路径输出 runtime parameters。它的科学作用是测试收益是否来自一般 histogram-driven calibration 原则，而不是 CNN 架构本身。

口头解释：

> 这一段非常重要，因为它改变论文故事。如果 statistical calibration 在同一 runtime contract 下更强，就说明贡献不能只包装成 CNN。更稳妥的讲法是：我们发现 syndrome histogram 驱动的低维仿射校准很有效，CNN residual 和 statistical calibration 是两条实现路径。

专业词解释：

- **Non-neural comparator**：非神经网络对比方法。
- **Bounded bias correction**：有界偏置修正，避免参数过大导致不稳定。
- **Calibration principle**：校准原则，即从统计量更新运行参数的通用思想。

### 6.6 Stage-and-commit runtime contract

原文说：slow loop 不直接改变 active fast-loop parameters，而是把候选 `(K,b)` staged 到 inactive bank，并在 safe epoch boundary commit。这样可以分别测量 stale-parameter effects、update latency、commit success、rollback/fallback 和 fixed-point stability。

口头解释：

> 这就是工程系统里的安全更新机制。实时路径不能在计算中途被改参数，所以新参数先写到 inactive bank，等到安全边界再切换。这样可以把“算法效果”和“运行时更新机制”分开测量。

专业词解释：

- **Inactive bank**：暂未被实时路径使用的参数 bank。
- **Commit boundary**：安全提交边界，通常是窗口或 epoch 结束处。
- **Rollback**：回滚，如果新参数异常则退回旧参数。
- **Update latency**：参数估计完成到实际生效之间的延迟。
- **Fixed-point stability**：固定点实现中没有数值发散、溢出或异常饱和。

---

## 7. Relationship to Existing Work

### 7.1 GKP and bosonic analog soft information

原文说：已有 analog QEC、surface-GKP、concatenated bosonic-QLDPC 工作使用 continuous measurement information 设置 decoder weights 或 soft messages；本文区别是把 analog statistics 压缩进 physical-layer affine GKP fast-path parameters，而不是 outer-code matching weights 或 LDPC messages。

口头解释：

> 这段的作用是避免夸大原创性。连续 syndrome 有价值不是本文首次发现。本文的不同点是使用位置：别人很多是在外层 decoder 里用 soft information，而本文直接把这些统计压缩成 GKP 物理层的 `K,b` 参数。

可用对比句：

> Existing work asks how analog GKP information should inform an outer decoder; here we ask how recent analog GKP statistics should recalibrate the physical affine correction itself.

专业词解释：

- **Concatenated code**：级联码，把一个编码作为内层，另一个编码作为外层。
- **Outer-code decoder**：外层码 decoder，例如 surface code 或 LDPC decoder。
- **Soft message**：概率型消息，包含置信度而非硬判决。

### 7.2 Adaptive priors and syndrome-statistics estimation

原文说：adaptive-weight 和 syndrome-statistics 方法会从数据估计 decoder weights 或 noise parameters；calibrated decoders 和 decoder-prior optimization 说明 prior mismatch 会影响 logical performance；本文借鉴这个 lesson，但把 adaptation target 映射到 GKP affine parameters `(K,b)`，并用 stage-and-commit 限制更新。

口头解释：

> 这段说明“从 syndrome 统计估计噪声”也不是凭空来的。已有 stabilizer code 和 decoder calibration 文献说明，如果 decoder 使用的噪声先验和真实硬件不一致，逻辑错误率会变差。本文把这个思想落到 GKP 的仿射参数上。

专业词解释：

- **Prior mismatch**：先验失配，decoder 假设的噪声分布和真实分布不一致。
- **Adaptive weight**：自适应权重，根据观测数据更新 decoder 中的概率权重。
- **Syndrome statistics**：syndrome 的经验统计分布。

### 7.3 Learned low-latency QEC modules

原文说：AI pre-decoder、高精度神经 decoder 和 calibration-conditioned FiLM decoder 说明，学习模块最有用时需要明确 input/output contract、latency role 和 classical decoder integration。本文 learned module 不是 per-shot decoder，而是慢校准组件。

口头解释：

> 这里可以类比 NVIDIA 那类论文结构：它不是单纯展示一个网络，而是强调网络怎么嵌入低延迟解码系统。本文学习模块也不是每拍跑一次，而是慢速更新低维参数。这样能避免“神经网络延迟太高、不适合实时纠错”的常见质疑。

专业词解释：

- **AI pre-decoder**：AI 预解码器，在完整 decoder 前先做简化或筛选。
- **FiLM**：Feature-wise Linear Modulation，用标定信息对网络中间特征做缩放和平移。
- **Latency role**：延迟角色，即模块是在每拍路径、慢速后台还是离线阶段运行。

### 7.4 Real-time and FPGA QEC decoders

原文说：实时 QEC decoder 和 FPGA-integrated systems 已经在 stabilizer-code settings 报告低延迟硬件解码；real-time bosonic QEC 也有 experimental feedback systems。这些工作设定了完整 deployment claim 的标准：closed-loop timing、worst-case latency、resource use 和 hardware integration。本文贡献是 hardware-compatible GKP affine calibration contract 和 software-HIL evaluation。

口头解释：

> 这一段要非常谨慎。我们可以说结构是 hardware-compatible 或 FPGA-oriented，但不能把软件 HIL 说成真板验证。已有 FPGA QEC 文献的标准很高，包括真实闭环时序、资源占用、最坏情况延迟。本文当前更准确的定位是：提出并验证一个适合硬件实现的 GKP 仿射校准契约。

专业词解释：

- **Hardware-compatible**：结构适合硬件实现，但不等于已经完成真板实验。
- **Closed-loop timing**：闭环时序，从测量到反馈校正的总时间。
- **Worst-case latency**：最坏情况延迟，实时系统必须关注。
- **Resource use**：FPGA LUT、DSP、BRAM 等资源占用。

---

## 8. Experimental Setup

### 8.1 Software-HIL protocol

原文说：numerical experiments 使用 software-HIL protocol，保留 fast loop、slow loop、parameter-bank 和 staged-commit interfaces，同时底层实验在软件中执行；这足以在 identical drift trajectories 和 paired seeds 下比较 adaptive calibration laws，但不测量 board-level latency 或 hardware resource use。

口头解释：

> Software-HIL 的意义是保留未来硬件接口语义，但先在软件中做受控比较。这样可以公平比较不同校准律，因为它们面对同样漂移轨迹和随机种子。不过这仍然不是板级时延实验，所以汇报中要把“算法/接口验证”和“硬件实测”分开。

专业词解释：

- **HIL**：Hardware-in-the-loop，硬件在环。这里是 software-HIL，即保留接口但由软件后端执行。
- **Paired seeds**：配对随机种子，不同方法使用相同随机轨迹，提高比较公平性。
- **Drift trajectory**：噪声参数随时间变化的轨迹。

### 8.2 Scenarios and modes

原文列出四个 drift scenario，并说明 main comparison 包含 filtering baselines、residual affine baselines、learned hybrid residual branch 和 statistical calibration variants；指标是 `final_ler_mean`，越低越好。

口头解释：

> 这段说明实验怎么比较。每个方法都跑同样四个漂移场景，比较最终 logical-error proxy。方法类别包括传统滤波器、teacher/residual affine、CNN hybrid 和 statistical calibration。这样能把“跟踪漂移的能力”放到同一评价框架里。

专业词解释：

- **`final_ler_mean`**：固定协议下的平均逻辑错误率代理指标，越低越好。
- **Hybrid residual branch**：teacher + CNN residual 的混合分支。
- **Residual affine baseline**：仍是仿射快路径，但参数更新方式不同的基线。

---

## 9. Numerical Results and Benchmark Plan

这一节最接近正式论文结果部分。汇报时建议先说“已经有三类结果”和“还需要补三类 benchmark”：

已可汇报的结果：

1. 五种 affine 模式在四个漂移场景上的比较。
2. CNN residual branch 的 feature / teacher ablation。
3. statistical calibration 的强 comparator 结果和局部敏感性。

仍应补强的 benchmark：

1. unseen drift generalization。
2. oracle / wrapped-Gaussian / nearest-lattice 等理论 baseline。
3. runtime、fixed-point、embedded 和 board-level 指标。

### 9.1 Four-scenario affine benchmark

原文表格比较 EKF、UKF、Const.-mu、RLS-b、Hybrid-b 五种 affine decoding modes。Hybrid-b 在四个场景中都是五者最好，UKF 是最强 filtering baseline。

口头解释：

> 这张表先回答：teacher-anchored learned residual 是否比常见滤波类 baseline 更好。在四个漂移场景中，Hybrid-b 都是五种 affine 模式中 LER 最低的。这说明用 histogram 和 teacher 信息学习 bias residual 能提升 adaptive affine decoder。

表格读法：

- `static_bias_theta`：Hybrid-b 从 UKF 的 `0.825370` 降到 `0.810902`。
- `linear_ramp`：Hybrid-b 从 UKF 的 `0.811201` 降到 `0.787755`。
- `step_sigma_theta`：Hybrid-b 从 UKF 的 `0.811548` 降到 `0.788800`。
- `periodic_drift`：Hybrid-b 从 UKF 的 `0.821558` 降到 `0.806392`。

专业词解释：

- **EKF / UKF**：两类卡尔曼滤波器，用来估计随时间变化的噪声状态。
- **Const.-mu**：使用固定均值偏置假设的模式。
- **RLS-b**：用递归最小二乘更新 bias 的模式。
- **Hybrid-b**：teacher + CNN residual 主要修正 bias 的模式。

### 9.2 Feature and teacher ablations

原文表格比较 `ukf`、`hybrid_full`、`hybrid_no_hist_deltas`、`hybrid_no_teacher_prediction`、`hybrid_no_teacher_params`、`hybrid_no_teacher_deltas`。去掉 histogram deltas 会明显变差；去掉 teacher prediction 也比 full hybrid 差；但 no-teacher-params 在这组 ablation 中最好，因此不能简单说所有 teacher channel 都有益。

口头解释：

> 这张表要讲得谨慎。它支持两个结论：第一，histogram delta 很重要，因为去掉后 LER 从 `0.798545` 变差到 `0.826723`；第二，teacher 特征不是越多越好，因为 `hybrid_no_teacher_params` 反而达到 `0.749621`，是这组里最低的。这说明当前 teacher feature 设计可能有冗余或干扰，后续写论文时不能把 teacher 所有通道都包装成必然有效。

建议汇报措辞：

> The ablation suggests that temporal histogram changes are robustly useful, while the role of teacher-side channels is more nuanced.

专业词解释：

- **Ablation table**：消融表，通过删除特征或模块判断其贡献。
- **`Delta vs UKF`**：相对 UKF 的变化，负数表示比 UKF 更好。
- **`Delta vs Hybrid Full`**：相对完整 hybrid 的变化，负数表示比完整 hybrid 更好。
- **Nuanced**：有细微差别，不能用单一句子概括。

### 9.3 Statistical calibration as a strong comparator

原文表格显示：StatCalib 在四个场景中都明显优于 UKF 和 Hybrid-b，例如 `static_bias_theta` 中从 Hybrid-b 的 `0.810902` 降到 `0.431708`，`linear_ramp` 中从 `0.787755` 降到 `0.467083`。

口头解释：

> 这是目前结果部分最关键的转折。StatCalib 不是 CNN，但在同一四场景 suite 中显著更好。这说明核心科学问题不是“CNN 是否比滤波器强”，而是“是否可以用 recent syndrome statistics 对仿射 GKP fast path 做有效校准”。这个结论对论文更有价值，也更稳。

表格读法：

- `static_bias_theta`：StatCalib `0.431708`，比 Hybrid-b 低约 `0.379`。
- `linear_ramp`：StatCalib `0.467083`，比 Hybrid-b 低约 `0.321`。
- `step_sigma_theta`：StatCalib `0.460016`，比 Hybrid-b 低约 `0.329`。
- `periodic_drift`：StatCalib `0.438751`，比 Hybrid-b 低约 `0.368`。

建议汇报措辞：

> This result reframes the paper: the primary object is an adaptive affine calibration contract, with CNN residual learning and statistical calibration as two realizations.

专业词解释：

- **StatCalib**：Statistical Calibration，直接从 syndrome 统计量估计偏置修正的非神经方法。
- **Runtime contract**：运行契约，所有方法都必须通过同样的 clip、smooth、commit 和固定点接口。
- **Reframe**：重塑叙事重点。

### 9.4 Local sensitivity of statistical calibration

原文说：StatCalib 在默认参数附近做五点局部网格：default、low residual scale、high residual scale、low residual clip、high signal threshold。high-threshold 平均 LER 最低但优势极小；default 赢 3/4 场景并有最好 mean rank。

口头解释：

> 这张表回答“StatCalib 是否只是某个参数点碰巧好”。结果显示不是。high-threshold 的平均 LER `0.449241` 略低于 default 的 `0.449254`，差距非常小；default 在四个场景里赢了三个，mean rank 最好。因此更稳妥的结论是：StatCalib 的优势在局部参数变化下比较稳定，而不是依赖某一个特别调出来的点。

专业词解释：

- **Residual scale**：残差缩放，控制修正幅度。
- **Residual clip**：残差裁剪上限。
- **Signal threshold**：信号阈值，决定统计校准何时认为偏置信号足够可靠。
- **Mean rank**：每个场景内排序后再求平均，越低表示越稳定靠前。
- **Scenario wins**：在多少个场景中排名第一。

### 9.5 Per-scenario best statistical calibration variant

原文表格列出每个场景最好的 StatCalib variant。每个场景的最好 variant 都大幅超过 UKF 和 Hybrid-b；`static_bias_theta` 的 high-threshold 行带有局部 caveat，因此最好作为有条件胜利解释。

口头解释：

> 这张表展示了 StatCalib 的上界潜力。每个场景选择最好的 StatCalib variant，都比 Hybrid-b 低很多。但口头汇报时不要过度强调 scenario-wise best，因为这有一点“按场景挑参数”的味道。更适合作为补充：说明不同场景下 StatCalib 有很大的可调空间。

专业词解释：

- **Per-scenario best**：每个场景单独选择最优参数变体。
- **Gap vs Hybrid-b**：Hybrid-b 的 LER 减去最好 StatCalib LER，数值越大表示 StatCalib 优势越大。
- **Caveat**：保留说明或限制条件。

### 9.6 Mechanism probe for residual-b behavior

原文说：已有 multi-seed mechanism figure 比较 seed-wise hybrid-model gaps 和 lower-clip intervention；描述性结论是 residual-b high-amplitude behavior 在多个 seed 中出现，但降低 residual clip 的干预结果混合且常常有害。因此 residual amplitude 参与某些 failure modes，但不是单调因果解释。

口头解释：

> 这一小节用于避免过度解释 CNN 的行为。我们观察到某些 seed 中 residual-b 幅度偏高和性能 gap 有关，但直接降低 clip 不一定改善，甚至可能变差。所以现在只能说 residual amplitude 可能参与 failure mode，不能说已经证明“降低 residual 就能解决问题”。

专业词解释：

- **Mechanism probe**：机制探针，用实验或图表探索性能差异背后的原因。
- **Seed-wise gap**：按随机种子分别计算的性能差距。
- **Intervention**：干预实验，人为改变某个因素看结果是否改变。
- **Monotonic causal explanation**：单调因果解释，例如“clip 越低越好”。当前结果不支持这种简单说法。

### 9.7 Unseen drift generalization

原文说：四个场景覆盖 steady bias、ramp tracking、step response 和 periodic drift；更强 benchmark 需要 random-walk drift、`1/f`-like drift、burst measurement noise、coupled bias-variance drift 和 faster-than-window drift。

口头解释：

> 这部分是正式论文应该补的泛化实验。四个手工场景只能证明在受控漂移下有效，不能证明模型对未见漂移也稳健。随机游走、低频噪声、突发测量噪声和快于窗口的漂移可以测试方法是否真正适应 nonstationary noise，而不是只适应这四个固定轨迹。

专业词解释：

- **Random-walk drift**：随机游走漂移，参数每步随机小变化。
- **`1/f`-like drift**：低频占主导的漂移，常见于硬件慢噪声。
- **Burst measurement noise**：突发测量噪声，短时间内噪声突然变大。
- **Faster-than-window drift**：漂移速度快于 slow loop 统计窗口，容易造成估计滞后。

### 9.8 Oracle and wrapped-Gaussian baselines

原文说：affine methods 需要与更强理论 baselines 比较，包括 nearest-lattice hard decoding、static affine decoding、oracle affine decoding with true noise state、wrapped-Gaussian 或 maximum-likelihood decoding。

口头解释：

> 这部分是为了让论文对比不“自娱自乐”。只和 EKF、UKF 比还不够，还要知道仿射快路径本身损失多少，以及噪声估计不完美损失多少。Oracle affine 使用真实噪声状态，可以给出慢回路估计的上界；wrapped-Gaussian 或 ML decoder 可以给出非仿射更强 decoder 的参考。

专业词解释：

- **Oracle affine decoder**：知道真实噪声参数的理想仿射 decoder，用来估计可达到上界。
- **Static affine decoder**：固定 `K,b` 不自适应的 decoder。
- **Nearest-lattice hard decoding**：最近晶格点硬判决。
- **Theoretical baseline**：理论基线，用来衡量方法距离理想或强模型还有多远。

### 9.9 Runtime, quantization, and fixed-point degradation

原文说：deployment claim 不只依赖 logical performance，还需要 fixed-point vs floating-point、saturation/overflow rates、commit latency、stale-parameter penalty、fallback frequency 和 slow-loop cost。

口头解释：

> 这部分是工程论文说服力的关键。哪怕 LER 很好，如果固定点量化后性能明显下降，或者 saturation 很多，或者 commit 延迟太大，就不能说适合低延迟硬件。因此要把数值性能和 runtime diagnostics 一起报告。

专业词解释：

- **Floating-point**：浮点数，软件中常用，精度高但硬件成本更高。
- **Fixed-point degradation**：从浮点变固定点后的性能下降。
- **Commit latency**：新参数从生成到生效的延迟。
- **Fallback frequency**：触发回退保守参数的频率。
- **Slow-loop cost**：慢回路每次更新的计算成本。

### 9.10 Embedded runtime and board-level validation

原文说：embedded inference 和 board-level validation 与 software-HIL 不同；相关指标包括 exported-model equivalence、embedded runtime latency、host-to-device update cost、hardware resource use 和 closed-loop timing。这些实验完成后才能把 “hardware-compatible” 升级成更强的 hardware-demonstrated claim。

口头解释：

> 这段要讲清楚边界。当前可说的是结构和接口面向硬件，不能说已经完成真实 FPGA 闭环验证。后续如果要冲更好的期刊，就需要导出模型等价性、嵌入式推理延迟、主机到设备参数更新时间、FPGA 资源和闭环时序这些数据。

专业词解释：

- **Exported-model equivalence**：导出模型与原始软件模型输出是否一致。
- **Embedded runtime latency**：嵌入式平台上的实际运行延迟。
- **Host-to-device update cost**：主机把新参数写到设备的时间成本。
- **Board-level validation**：真板级验证。
- **Hardware-demonstrated claim**：经过真实硬件实验支撑的主张。

---

## 10. Discussion

当前 note 中 Discussion 章节保留为空。口头汇报时可以简单说明：

> Discussion 将来主要用于综合解释三件事：第一，为什么 statistical calibration 强并不削弱本文，反而说明 adaptive affine calibration 这个结构是核心；第二，CNN residual 在什么情况下有意义，例如更复杂 drift 或统计规则难以手写时；第三，软件 HIL 与真实硬件部署之间还差哪些证据。

建议将来 Discussion 可写的主题：

- **CNN 与 StatCalib 的关系**：CNN 不是唯一贡献，StatCalib 提供强非神经 comparator。
- **仿射 fast path 的边界**：它低延迟、可部署，但在多峰后验和 lattice boundary 附近可能不是最优。
- **结果可信度边界**：software-HIL 适合比较校准律，但不能替代板级时序和资源测量。
- **未来扩展**：更强 drift family、wrapped-Gaussian baseline、embedded runtime、real hardware closed-loop。

---

## 11. Conclusion

原文 conclusion 有三段：第一段总结 two-timescale affine calibration；第二段总结数值结果；第三段指出剩余技术 gap。

### 第一段：方法总结

口头解释：

> 结论第一段重新强调本文的设计哲学：fast loop 是固定点仿射规则，slow loop 从 syndrome statistics 更新参数。这个结构介于固定 decoder 和完整 neural decoder 之间，既能适应漂移，又不把实时路径变复杂。

### 第二段：结果总结

口头解释：

> 结论第二段把现有结果浓缩为两个结论。第一，teacher-anchored residual branch 在四个漂移场景中优于 filtering baselines，histogram delta 有用。第二，statistical calibration 是更强 comparator，且在局部参数网格中稳定。因此最稳的科学故事是低维 histogram-driven calibration，而不是单纯 CNN 优于经典滤波器。

### 第三段：剩余 gap

口头解释：

> 最后一段说明要形成更强投稿，还需要三类证据：更广漂移族、oracle/wrapped-Gaussian 理论 baseline、真实 runtime 约束测量。这样写的好处是既展示已有结果，也不把当前证据夸大成完整部署验证。

---

## 12. 术语总表

### 量子纠错与 GKP

- **QEC**：Quantum Error Correction，量子纠错。
- **GKP code**：Gottesman-Kitaev-Preskill code，用振子的周期相空间结构编码 qubit。
- **Qubit**：量子比特。
- **Bosonic mode**：玻色模式，例如腔场模式，具有连续变量自由度。
- **Syndrome**：纠错测量得到的信息，用来推断错误。
- **Analog syndrome**：连续值 syndrome。
- **Logical qubit**：经过编码保护的 qubit。
- **Logical error**：编码层面发生的错误。
- **Displacement correction**：相空间位移校正。

### 统计与估计

- **Gaussian approximation**：高斯近似，把局部分布近似为高斯。
- **Linear-MMSE**：线性最小均方误差估计。
- **Covariance matrix**：协方差矩阵。
- **Bias**：均值偏置。
- **Drift**：噪声统计随时间变化。
- **Histogram**：直方图，用来表示样本分布。
- **Ablation**：消融实验。
- **Oracle baseline**：使用真实隐藏信息的理想基线。

### 工程与硬件

- **FPGA**：Field-Programmable Gate Array，现场可编程门阵列。
- **HIL**：Hardware-in-the-loop，硬件在环。
- **Software-HIL**：接口保持硬件在环风格，但实际后端由软件执行。
- **Fixed-point**：固定点数值格式。
- **Quantization**：量化，把连续或高精度值映射到有限精度。
- **Saturation**：饱和，数值达到上下限。
- **Stage-and-commit**：先暂存后提交的安全参数更新机制。
- **Parameter bank**：参数存储区。

---

## 13. 可能被问到的问题与回答

### Q1：为什么不用完整神经网络直接输出校正？

回答：

> 量子纠错实时路径对延迟、确定性和可验证性要求很高。完整神经网络直接放在每拍路径中会增加最坏情况延迟和硬件实现复杂度。本文选择把学习放在慢校准回路里，每拍只执行 `K s + b`，更符合低延迟硬件约束。

### Q2：为什么 `K s + b` 有理论依据？

回答：

> 在单个 GKP 晶格分支附近，如果位移误差和 syndrome 可以近似为联合高斯分布，那么 linear-MMSE 估计正好给出 `hat e = K s + b`。所以仿射规则是局部高斯近似下的最优线性估计，不只是工程简化。

### Q3：StatCalib 比 CNN 强，会不会削弱论文？

回答：

> 不一定。它反而说明本文更核心的对象不是 CNN 架构，而是 adaptive affine calibration contract。CNN residual 是一种 learned realization，StatCalib 是一种 non-neural realization。二者共同支持“recent syndrome statistics 可以有效校准 GKP 仿射快路径”这个主张。

### Q4：现在的结果是否足够支撑 FPGA 论文？

回答：

> 现有结果支撑 hardware-compatible architecture 和 software-HIL 条件下的漂移自适应效果，但还不足以支撑完整板级部署主张。若要更强投稿，需要补固定点退化、嵌入式延迟、host-to-device 更新成本、资源占用和 closed-loop timing。

### Q5：为什么还需要 oracle 和 wrapped-Gaussian baseline？

回答：

> 它们能分解性能损失来源。Oracle affine baseline 说明如果噪声状态估计完美，仿射路径最多能做到什么；wrapped-Gaussian 或 ML decoder 说明如果不限制为仿射路径，理论上还能提升多少。这样对比会让论文更公平。

### Q6：四个漂移场景是否足够？

回答：

> 四个场景覆盖 steady bias、slow ramp、step response 和 periodic drift，是受控 benchmark 的起点。但投稿时最好加入 unseen drift family，例如 random walk、`1/f` drift、burst noise 和 faster-than-window drift，用来验证泛化能力。

### Q7：为什么 logical error rate 比参数误差更重要？

回答：

> 参数估计只是中间量。量子纠错最终关心的是校正后逻辑 qubit 有没有出错。一个参数估计看似更接近 teacher，但如果没有降低 logical failure，就不是更好的 decoder。因此主要指标应该是 closed-loop LER。

---

## 14. 建议汇报节奏

15 分钟版本可以这样安排：

1. **2 分钟**：GKP 码、近似态、连续 syndrome 和漂移问题。
2. **3 分钟**：双时间尺度架构，解释 `Delta = K s + b` 和 slow calibration loop。
3. **3 分钟**：teacher residual、statistical calibration、stage-and-commit runtime contract。
4. **4 分钟**：结果表格，先讲 Hybrid-b 优于 filtering baseline，再讲 ablation，最后讲 StatCalib 重新定义故事线。
5. **2 分钟**：相关工作定位和项目优势。
6. **1 分钟**：后续 benchmark：unseen drift、oracle/wrapped-Gaussian、fixed-point/runtime/board-level。

最重要的一句话收尾：

> 这个项目最值得讲的不是“把 CNN 用到 GKP 解码”，而是把漂移自适应 GKP 纠错组织成一个低维、可校准、可量化、适合硬件执行的 affine fast-path framework。
