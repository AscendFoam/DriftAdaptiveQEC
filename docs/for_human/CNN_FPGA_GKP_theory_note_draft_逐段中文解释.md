# CNN_FPGA_GKP_theory_note_draft 逐段中文解释

源文件：`docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`

本文档用途：按源稿顺序解释 theory note 的内容，帮助读者从 GKP 近似码原理、漂移自适应动机、模型结构、实验表格和证据边界出发，理解每一段在讲什么。本文档不是新实验记录，不改变源稿结论，也不把软件仿真、`.tflite` 本地运行或硬件验证要求升级为硬件结果。

## 0. 先给读者的总览

这份 note 的核心思想可以用一句话概括：不要把 GKP 解码器做成一个难以验证的端到端神经网络，而是把实时校正路径压缩成一个很小的仿射公式 `Delta = K s + b`，再用较慢的校准模块根据最近的 syndrome 统计量更新 `K` 和 `b`。

其中 `s` 是 GKP syndrome，也就是测量出来的相空间位移余量；`Delta` 是要施加的校正位移；`K` 是一个 2x2 增益矩阵；`b` 是偏置项。实时路径只做矩阵乘法、加法、裁剪和定点量化，所以比较适合未来 FPGA 或嵌入式硬件。复杂的统计估计、CNN residual 或 statistical calibration 都被放到慢环路，不直接进入每次 shot 的实时关键路径。

整篇 note 的证据层次很重要：

- 主证据是四个漂移场景下的软件 HIL benchmark。这里 Hybrid-b 在四个场景都优于 EKF、UKF、Const-mu、RLS-b 等参考方法。
- 较强但仍有限的补充证据是两个场景完成了 12 对 paired interval 检查，方向都是 UKF-minus-Hybrid 为正。
- 机制、ablation、statcalib、oracle、wrapped-Gaussian、holdout、lag、channel surrogate、Q4.20 和 runtime counter 表格主要用于解释和边界审计，不等同于新的完整 benchmark。
- 硬件相关表格目前只是要求清单或软件可观测性，不是 FPGA 实测。

## 1. 关键术语通俗表

| 术语 | 通俗解释 |
| --- | --- |
| GKP code | 把一个 qubit 编码到谐振子相空间中的一种玻色量子纠错码。可以想象为在 `q` 和 `p` 坐标上放了一排周期性“梳齿”。 |
| approximate GKP | 真实设备不能产生无限尖、无限长的理想梳齿，只能产生有限宽度、有限能量的近似梳齿。有限宽度会带来不确定性。 |
| syndrome | 测到的位移误差信息。GKP 的 syndrome 是连续值，不只是 0/1。 |
| modular syndrome | syndrome 只在一个晶格周期内给出余量，所以它是“取模以后”的位移信息。 |
| half-lattice boundary | 半个晶格周期的位置。残余位移超过这个边界时，会被判到错误逻辑分支，可能产生逻辑错误。 |
| affine decoder | 用 `Delta = K s + b` 这种线性加偏置的规则做校正。 |
| fast loop | 每次 shot 都要快速执行的实时校正路径。 |
| slow loop | 较慢更新一次的校准路径，根据历史 syndrome 统计量估计 `K` 和 `b`。 |
| teacher | 传统统计或滤波估计器，例如 UKF/EKF/RLS，用来给 CNN 一个稳定锚点。 |
| residual-b | CNN 不直接输出完整解码结果，只预测对偏置 `b` 的小修正。 |
| software-HIL | 保留硬件接口和运行时约束，但实际在软件中执行的硬件在环风格仿真。 |
| final_ler_mean | 表格里的主要性能指标，越低代表逻辑错误代理指标越低。 |

## 2. 标题、摘要和全文主线

### 内容块 A01：标题

源稿标题是“A Dual-Loop Teacher-Anchored Residual Calibration Framework for Drift-Adaptive GKP Decoding”。标题中的几个关键词已经把论文定位说清楚了：

- Dual-Loop：有快慢两个环路。快环路负责每次 shot 的实时校正，慢环路负责根据统计信息更新参数。
- Teacher-Anchored：CNN 不是自由发挥，而是围绕一个 classical teacher 的估计做小修正。
- Residual Calibration：学习模块预测的是 residual，也就是对已有校准结果的增量修正。
- Drift-Adaptive GKP Decoding：目标不是一般意义上的神经解码，而是 GKP 解码在噪声漂移下的自适应校准。

### 内容块 A02：摘要第 1 段

摘要开头说明 practical GKP 的困难：GKP syndrome 是连续的、取模的，理论上能告诉我们位移误差，但真实近似 GKP 态有有限能量，设备状态也会漂移。如果校正规则固定不变，一旦偏置、方差或协方差方向变化，原来的校正就会失配。

这一段提出的解决方案是 dual-loop：快环路使用 `Delta_t = K_t s_t + b_t`，慢环路根据最近的 syndrome histogram 更新运行时参数。直观地说，实时执行时仍然只算一个小公式；复杂的“现在噪声状态变了吗”这件事放到慢速校准系统里。

### 内容块 A03：摘要第 2 段

这一段解释 teacher 和 CNN 的关系。classical teacher 给出可解释的基准参数，CNN 只预测一个 bounded residual calibration term，目前主要集中在 `b_t`。这和端到端 neural decoder 的区别非常关键：这里不是让网络直接决定每次如何纠错，而是让网络给一个低维物理校准面做小幅修正。

通俗理解：teacher 像一个保守但稳定的老师傅，CNN 像一个观察近期统计变化的助手。助手不能直接接管机器，只能建议把偏置旋钮微调一点。

### 内容块 A04：摘要第 3 段

这一段给出主结果：在四个受控漂移场景中，teacher-anchored hybrid residual branch 在四个场景都排第一，UKF 都排第二。相对 UKF 的平均下降幅度分别是 1.75%、2.89%、2.80%、1.85%，总体平均约 2.32%。

这一段还强调新加入的 paired-interval 数据只加强了两个场景：`static_bias_theta` 和 `linear_ramp`。其他 all-scenario、pooled、holdout、hardware 验证还没有完成。因此它支持“两个场景方向更可信”，但不支持“所有场景都完成统计闭环”。

### 内容块 A05：摘要第 4 段

摘要最后给出谨慎结论：ablation 和多 seed 材料只能支持 descriptive reading。也就是说，它们能说明 histogram-delta 特征有用、某些不稳定模式在不同 seed 下重复出现、lower-clip intervention 多数有害，但不能证明一个完整因果机制。

statistical-calibration 分支也很重要：它说明强结果可能来自“仿射校准契约”这个思想，而不一定只来自 CNN 架构。但是当前证据没有选出唯一应该推广的阈值。因此最强结论不是“某个 CNN 已经解决 GKP decoding”，而是“漂移自适应的低维仿射校准是可信、硬件边界清晰的组织原则”。

## 3. Introduction 逐段解释

### 内容块 I01：GKP syndrome 为什么特别

引言第一段介绍 bosonic code 和 GKP code。GKP 码把逻辑 qubit 编码进一个 oscillator 的 Hilbert space。理想 GKP 态在相空间里形成周期晶格，小位移错误可以通过 modular syndrome 测量出来，再用 displacement 操作纠正。

这里最重要的是 analog syndrome。普通 stabilizer syndrome 往往是二值或离散的，而 GKP syndrome 是连续值。连续值不仅告诉你“往哪边修”，还告诉你“离逻辑边界有多近”。离边界越近，越容易因为一点测量噪声或校准误差跨过边界。

### 内容块 I02：近似 GKP 态带来的实际困难

第二段说明现实中没有理想无限能量 GKP 态。真实态的 comb peak 有有限宽度，还有包络。测量还会受到 measurement inefficiency、noisy auxiliary states、circuit imperfections、oscillator loss 和 calibration error 的影响。

通俗地说，理想 GKP 像无限细、无限准的刻度线；真实 GKP 像每条刻度线都变粗了，还带有背景噪声。decoder 看到的 syndrome 不再只是干净位移，而是混入了状态制备、测量和设备漂移后的有效噪声。如果设备状态随时间变，固定校正规则就会越来越不准。

### 内容块 I03：已有工作给出的两条启发

第三段总结相关领域的启发。GKP 和 bosonic decoding 领域已经证明 analog soft information 有价值，例如 surface-GKP 或 bosonic-QLDPC decoder 可以把连续 syndrome 用来设置 matching weights、belief propagation messages 或 outer-code priors。

另一条启发来自硬件 QEC：学习模块只有在 latency role 和 integration boundary 明确时才有工程意义。也就是说，不能只说“用了神经网络更强”，还要说明网络运行在哪里、多久运行一次、失败时如何退回、硬件怎样验证。

### 内容块 I04：本文的出发点

第四段提出本文核心：漂移适应应该通过 bounded runtime contract 进入 GKP 层，而不是通过自由形式的 per-shot decoder。公式 `Delta_t = K_t s_t + b_t` 是这个 contract 的中心。

这里 `s_t` 是当前两象限 syndrome，`K_t` 和 `b_t` 是量化后的运行时参数。慢环路观察最近的 syndrome histogram，估计有效噪声状态，然后把更新后的参数写入 runtime bank。这样每次 shot 的在线路径仍是小型仿射运算。

### 内容块 I05：为什么这种分离有系统优势

第五段把方法与三类替代方案比较：

- 相比固定 affine decoder，它能跟踪 drift。
- 相比 full neural decoder，它的每次 shot 计算更小、更可解释、更容易做 fixed-point 验证。
- 相比 outer-code soft-information decoder，它直接校准 physical-layer GKP displacement rule，而不是只调整外层图模型或 LDPC message。

这段是整篇论文的系统定位：学习分支不是实时 decoder，而是慢环路校准模块。

### 内容块 I06：与 FPGA-QEC 文献的关系

第六段说明很多实时硬件 QEC 研究主要处理 binary stabilizer syndrome，例如 surface code、repetition code、qLDPC memory 等。这些研究建立了 closed-loop latency、resource reporting 的标准，但它们不直接处理 GKP 物理层这种 analog、modular、finite-energy syndrome。

本文架构因此是 complementary。它可以放在更大 stabilizer 或 LDPC decoding stack 前面、里面或旁边，作用是提供一个低维物理层校准接口。

### 内容块 I07：证据层次声明

引言最后一段非常重要。它明确说：controlled software-HIL benchmark 是 main result layer；feature/teacher ablation、多 seed mechanism、statistical-calibration extension lane 是 supporting layers；training/material regeneration、isolated true `.tflite`、hardware readiness artifacts 只是 boundary evidence。

这段相当于提前告诉审稿人：本文不会把辅助材料包装成同等级主结果，也不会把硬件准备材料说成真实板级结果。

## 4. Summary of Contributions 逐段解释

### 内容块 C01：贡献总述

贡献部分把工作整理成六个 bounded contributions。这里的 bounded 很关键，意思是每个贡献都有限定条件，不是泛泛地宣称“解决了 GKP 解码”。

### 内容块 C02：贡献 1，双环路仿射校准

第一项贡献是把每次 shot 的 correction path 写成 `Delta_t = K_t s_t + b_t`。自适应问题被移动到慢环路，慢环路根据近期 syndrome statistics 估计 effective noise state，再更新快环路要用的参数。

这项贡献的本质不是一个新神经网络，而是一个运行时接口设计：把连续 syndrome、近似 GKP 物理图像和固定点矩阵向量执行连接起来。

### 内容块 C03：贡献 2，teacher-anchored residual architecture

第二项贡献描述 teacher 和 residual CNN。teacher 先生成 `K_teacher` 和 `b_teacher`，CNN 只输出 `delta b_t`，最终 `K` 仍取 teacher 的 `K`，`b` 则是 teacher bias 加 residual 后再做 EMA 平滑。

通俗解释：网络不是决定整个方向盘怎么打，而是在老师傅给出的基础方向上微调偏置。这样模型的行为更容易被解释，也更容易限制到硬件可验证范围。

### 内容块 C04：贡献 3，受控参考证据层

第三项贡献说明最强性能证据仍是四场景软件 HIL benchmark。Hybrid-b 四个场景全部第一，UKF 全部第二。这里特别强调这不是 expanded benchmark、不是 deployment closure、也不是 real-board result。

这段在写作上很像审稿预防针：它主动压住 overclaim，避免读者误以为已有真实硬件或大规模 benchmark。

### 内容块 C05：贡献 4，保守机制解释层

第四项贡献说明 ablation 和 mechanism 只能给 descriptive support。histogram-delta features 确实重要；六 seed pack 中不稳定模式重复出现；但 lower-clip intervention 多数有害，所以不能说机制已经被因果证明。

这意味着论文目前能讲“我们观察到这些特征与性能变化相关”，不能讲“我们已经证明 residual amplitude 是失败因果原因，并且 clip 可以修复”。

### 内容块 C06：贡献 5，statistical-calibration extension lane

第五项贡献说同一个 affine runtime contract 也能容纳非神经的 histogram-driven calibration rule。这个结果对论文主线很有帮助，因为它说明真正关键的可能是“低维校准接口”，而不是“CNN 本身”。

但是这条线被明确标成 supplement-side calibration-extension analysis，不是主 comparator，也没有选出唯一 promoted threshold。

### 内容块 C07：贡献 6，runtime 和部署边界证据链

第六项贡献把 runtime、`.tflite`、硬件准备等材料分层。当前有训练/材料再生包、CPU-only bounded rerun、selected `.tflite` isolated local runtime confirmation、read-only hardware-readiness pack。但它们不是 board execution，不是 timing closure，也不是 deployment claim。

通俗说：这些材料证明“走向硬件验证的路有部分台阶已经搭好”，但还没有证明“已经走到硬件上并跑通”。

## 5. Metric-level advantages 表格解释

### 内容块 M01：表格功能

Metric-level advantages 表不是简单宣传“我们的架构更好”，而是把优势拆成 metric、advantage、architectural reason 和 supported scope 四列。这样能防止把不同证据等级混在一起。

### 内容块 M02：Logical-error proxy 行

这一行说，受控软件 HIL benchmark 中 Hybrid-b 比五个参考 baseline 更低；statcalib extension lane 在它自己的 bounded protocol 下也可强于参考锚点。

理解重点：这是真正有数值支持的性能优势，但只支持 controlled reference ranking 和 supplementary statcalib analysis。不能写成 paper-grade expanded benchmark 或 promoted comparator。

### 内容块 M03：Per-shot latency 行

这一行说实时路径低延迟，这是架构设计上的优势。原因是快环路只算 2x2 affine map、bias addition、clipping、quantization 和 parameter-bank selection。

证据边界：这是 architectural claim，不是板上 timing measurement。未来需要真实硬件 p50/p95/p99/worst-case latency。

### 内容块 M04：FPGA resource and cost 行

这一行说明快路径资源预计小，因为只需要 fixed-point multiply-add、saturation logic 和小 parameter bank。慢估计器可以跑在 host、embedded processor 或离线更新服务上。

证据边界：resource、energy、cost 目前没有 real device measurement。

### 内容块 M05：Engineering implementation 行

这一行强调 simple control boundary：量化的 `K,b`、clipping limits、stale-parameter handling、saturation counters、double-buffered stage-and-commit。它们让系统更容易集成、测试和 fail safely。

这里的重点不是性能，而是工程可审计性。软件 HIL 只验证 interface shape，board execution 和 source-vs-board agreement 还没有。

### 内容块 M06：Drift robustness 行

这一行说自适应机制可以跟踪 mean、variance、orientation changes。原因是 histogram windows 和 teacher/statistical summaries 会更新 affine parameters。

证据边界：只支持当前四个 effective drift scenarios。更广泛 drift family 还是未来 benchmark-expansion target。

### 内容块 M07：Modularity 行

这一行解释 modularity：慢环路估计器可以换，快环路不变。teacher residual CNN、statistical calibration、temporal model 或 gain-scheduled bank 都可以映射到同一个 committed `(K,b)` 接口。

这对论文主线很重要，因为它让 CNN 结果不必承载全部科学价值。即使将来换掉 CNN，仿射校准契约仍然成立。

### 内容块 M08：Compatibility with larger QEC stacks 行

这一行说明方法可以作为 GKP physical-layer calibration module，放在 outer stabilizer、surface-GKP 或 LDPC-GKP decoder 前面。当前还没有 system-level concatenated-code gains。

因此不能说它已经提升了完整量子计算架构的逻辑错误率，只能说它可能改善进入外层 decoder 之前的物理位移校正。

## 6. Brief Review of the GKP Code 逐段解释

### 内容块 G01：本节目的

本节不是提出新 GKP 理论，而是固定 notation，并解释为什么在局部 branch 内用 affine fast path 是合理近似。它也说明这个近似哪里会失效。

### 内容块 G02：理想 GKP 晶格常数

源稿使用 square-lattice GKP code，晶格常数 `lambda = sqrt(2*pi)`。这里的 `lambda` 是软件 residual coordinate，用于 syndrome wrapping、clipping 和 half-lattice boundary tests。

重要边界：它不是新的物理 GKP convention，也不是 calibrated finite-energy device parameter。

### 内容块 G03：理想 comb state

理想 `|0bar>` 被写成无限 comb，即在 `q` 坐标上每隔 `lambda` 一个尖峰。这个公式是启发式表达，帮助读者想象 GKP 态是周期性峰列。

但理想 comb 需要无限能量，因此不能真实制备。它只是理论参照。

### 内容块 G04：approximate GKP 态公式

近似 GKP 态把无限尖峰替换成有限宽的 Gaussian peaks，并加上一个宽 envelope。公式里的 `Delta` 表示 peak width，也就是 finite squeezing；`kappa` 表示 broad envelope scale，让整个态可归一化。

通俗理解：理想 comb 是无穷细、无穷长的梳子；近似 GKP 是每个齿有宽度，远处齿还逐渐变小的梳子。有限宽度导致相邻峰尾部重叠，所以 syndrome 不能百分百确定真实位移属于哪个 lattice branch。

### 内容块 G05：边界附近为什么最危险

本段说明 syndrome 值给出的是 possible lattice shifts 的 posterior。靠近 half-lattice boundary 时，两个 branch 的可能性接近，少量测量噪声或校准偏差就可能让 decoder 选错 branch。

这正是 analog syndrome 比 binary flag 更有价值的原因：它不仅告诉你估计值，还告诉你离危险边界有多近。

### 内容块 G06：syndrome 是 modular displacement information

本节定义累积位移误差 `e_t = [e_q, e_p]`，理想 syndrome 是 `e_t mod lambda`，范围在 `[-lambda/2, lambda/2)`。

这意味着 decoder 看到的是“余数”，不是绝对位移。如果真实位移超过一个周期，syndrome 只告诉你折回基本胞元后的值。

### 内容块 G07：有限能量和测量噪声下的 syndrome

源稿把测得 syndrome 写成 `mod(e_t, lambda) + measurement noise + GKP finite-squeezing noise`。这里的两个噪声项分别代表测量/辅助态贡献，以及近似 GKP 态本身的不确定性。

所以 decoder 实际观察的是 noisy modular representative。它既有位移信息，也有噪声、有限压缩和设备误差。

### 内容块 G08：boundary distance 的含义

公式 `d_bdry = lambda/2 - |s|` 表示 syndrome 到 half-lattice boundary 的距离。`d_bdry` 大，说明离边界远，branch decision 稳；`d_bdry` 小，说明接近边界，容易逻辑翻转。

这也是为什么只用 hard decision 会丢信息。连续 syndrome 的大小本身就是一种 confidence-like signal。

### 内容块 G09：local affine decoding 的 Gaussian 推导

源稿假设固定一个 branch `m`，把局部未包装 residual 写成 `r = e - lambda m`。如果 `r` 和 `s` 在该 branch 内近似 jointly Gaussian，那么线性 MMSE 估计就是 `r_hat = mu_r + Sigma_rs Sigma_ss^{-1}(s - mu_s)`，可写成 `K s + b`。

这段是 affine decoder 的理论理由：在局部 Gaussian、branch 已固定的条件下，仿射估计是合理的低延迟近似。

### 内容块 G10：affine approximation 的边界

源稿随后立刻说明限制：modulo structure 使 exact GKP decoding 是 nonlinear、branch dependent 的。接近 lattice decision boundary 时 posterior 可能是 multimodal，单一 affine rule 可能不如 ML、closest-lattice 或 wrapped-Gaussian decoder。

因此本文没有声称 affine decoder 全局最优。它只把 affine rule 放在 fast path，并通过慢环路自适应 `K,b`。

### 内容块 G11：moment-matched correction 的控制观点

源稿进一步解释：affine rule 可以看作对已 committed branch 的 moment-matched correction。bias、variance、covariance 改变时，局部条件矩也改变，所以最佳 `K,b` 也会变。

这就是 dual-loop 的控制原则：fast loop 继续做一个矩阵向量校正；slow loop 用 teacher、histogram features 或 calibration summaries 隐式更新这些矩。

### 内容块 G12：logical failure criterion

校正后 residual 被 wrap 到 GKP fundamental cell。如果 `|r_q| > lambda/2`，就发生对应逻辑错误，`p` 象限同理。

因此核心 metric 应该是 closed-loop logical error probability 或其代理，而不是单纯的参数回归误差。参数估计只有在能减少最终逻辑错误时才有意义。

## 7. Noise and Drift Model 逐段解释

### 内容块 N01：effective calibration model

本节说明噪声模型是 effective calibration model，不是完整 circuit-level 或 hardware-validated noise closure。它的作用是把慢环路可以从 syndrome 统计中估计的部分压缩成低维状态。

### 内容块 N02：effective noise state

状态写成 `(sigma_t, mu_q,t, mu_p,t, vartheta_t)`。其中 `sigma` 控制位移尺度，`mu_q/mu_p` 表示两个象限的平均偏置，`vartheta` 表示协方差轴旋转。

直观说，这四类量回答：“噪声有多大？中心偏到哪里？椭圆方向转了多少？”

### 内容块 N03：模型吸收的物理来源

源稿列出 finite-energy peak width/envelope、Gaussian displacement noise、biased means、anisotropic/rotated covariance、measurement noise 和 noisy auxiliary contributions。

这些都被压缩成 effective displacement-noise model。它比完整物理模型窄，但正好服务于 affine fast-path contract。

### 内容块 N04：四个 drift scenarios

四个场景分别是：

- `static_bias_theta`：有偏置、有旋转，但状态相对稳定，测试稳态校准。
- `linear_ramp`：噪声参数缓慢变化，测试跟踪能力。
- `step_sigma_theta`：方差或方向突然跳变，测试 shock response。
- `periodic_drift`：周期性非平稳，测试能否跟随周期变化。

这四类场景覆盖了几种典型适应需求，但不是 exhaustive drift coverage。

## 8. Model Architecture 逐段解释

### 内容块 R01：架构总述

架构围绕 teacher-anchored residual path，同时还有若干 supporting 或 boundary layers。只有 teacher-anchored hybrid path 参加 primary reference benchmark ranking；statistical-calibration branch 是 extension lane；deployment-facing paths 是 boundary evidence。

### 内容块 R02：fast loop 输入和 clipping

fast loop 收到 syndrome `s_t`，读取当前 active parameter bank，先把 syndrome 裁剪到 `[-s_max, s_max]`。这一步是为了防止异常输入让后面的固定点运算出界。

### 内容块 R03：raw affine correction 和 fixed-point quantization

随后计算 `Delta_raw = K s_clip + b`，再把输出裁剪到允许校正范围并做 fixed-point quantization。源稿使用 Q4.20 表示 syndrome 和 runtime parameters。

Q4.20 的含义是用固定点数近似浮点值，硬件友好，但需要检查量化误差、溢出和饱和。

### 内容块 R04：fast loop diagnostic counters

fast loop 还记录 histogram-input saturation、correction saturation、overflow、aggressive parameter events、commit counts、fallback-related signals 等计数器。

这些计数器的意义是：即使性能表看起来好，也要知道运行时是否频繁溢出、饱和、fallback 或 stale。它们是未来硬件验证前必须保留的可观测信号。

### 内容块 R05：fast path 的硬件边界

源稿同步了 submission draft 中的诊断：affine fast path 分析上每次 shot 只需 4 次乘法和 4 次加法；Q4.20 emulation 在受控样本上最大 correction 差异约 `1.6e-6`，没有改变 residual-boundary crossings，也没有 quantization saturation。

这些支持 fixed-point motivation，但不支持 FPGA synthesis、timing closure、resource、power 或 source-vs-board agreement。

### 内容块 R06：parameter mapping 的协方差构造

给定估计的 noise state，mapper 构造 error covariance `C`。`vartheta` 通过旋转矩阵决定协方差椭圆方向，`sigma_q/sigma_p` 决定两个轴上的噪声大小。

这一步把“噪声现在大概是什么形状”转成后续可计算的矩阵。

### 内容块 R07：measurement covariance 和 raw gain

measurement covariance 写成 `(sigma_meas^2 + Delta_eff^2) I`，其中包含测量噪声和有限 GKP squeezing 的有效贡献。raw gain 是 `C(C + R_meas)^{-1}`。

直觉上，如果真实位移噪声很大而测量可靠，gain 会更信任 syndrome；如果测量噪声很大，gain 会更保守。

### 内容块 R08：bias target 和 smoothing

bias target 是 `alpha(I-K)mu`，再通过 EMA 更新 `K_t` 和 `b_t`。EMA 平滑可以防止参数因为短窗口统计波动而剧烈跳变。

这一步体现了 slow loop 的工程特性：更新不是瞬间硬切换，而是平滑、可 staged、可 commit。

### 内容块 R09：teacher estimators

teacher family 从近期 syndrome history 估计 noise state。简单 teacher 可以用 histogram window moments；强一些的 teacher 如 EKF、UKF、RLS、particle filter 会加入时间状态空间假设。

teacher 的作用是给 learned branch 一个稳定、可解释的 baseline，也为 ablation 提供有意义的 classical reference。

### 内容块 R10：CNN residual branch 的输入

CNN 输入包括短上下文内的 normalized syndrome histograms 和 histogram deltas。histogram deltas 表示近期 histogram 如何变化，特别适合捕捉漂移趋势。

teacher-side features 可能包括 teacher 的 noise-state estimate、`K_teacher`、`b_teacher`、`Delta b_teacher` 等。

### 内容块 R11：CNN residual branch 的输出和限制

CNN 输出 `delta b_hat`，经过 scale 和 clipping 得到 `delta b_t`。最终 `K_t = K_teacher`，`b_t = EMA(b_teacher + delta b)`。

这里有两个限制很关键：CNN 只修正 bias，不改整个 decoder；修正量 bounded，不允许无限制输出。这使它更像校准器，而不是黑箱实时 decoder。

### 内容块 R12：statistical calibration branch

statistical calibration branch 不用神经网络，而是直接从近期 syndrome statistics 估计 bounded bias correction，再经过同样的 clipping、smoothing、stage-and-commit。

它的科学作用是验证收益是否来自 histogram-driven calibration principle。如果非神经方法也强，说明 affine runtime contract 本身很重要。

### 内容块 R13：stage-and-commit runtime contract

慢环路不直接修改 active 参数，而是先写入 inactive bank，在 safe epoch boundary commit。公式中 `A/B` bank 表示双缓冲切换。

这能让 stale-parameter、update latency、commit success、rollback/fallback 和 fixed-point stability 单独被测量。软件 HIL 验证这个 contract 可运行可审计，但还没有硬件 timing 和 source-vs-board agreement。

## 9. Relationship to Existing Work 逐段解释

### 内容块 W01：GKP analog soft information

已有 GKP 相关工作已经使用 analog syndrome 改善 matching weights 或 soft messages。本文区别在于使用层级不同：不是把 analog 信息只交给 outer-code decoder，而是把近期 syndrome statistics 压缩到 physical-layer affine fast-path parameters。

### 内容块 W02：adaptive priors 和 syndrome-statistics estimation

adaptive-weight 和 syndrome-statistics 工作表明 decoder priors 如果和设备统计不匹配，会影响 logical performance。本文接受这个教训，但把适应目标落到 GKP 的 `K,b`，并通过 stage-and-commit 限制更新。

### 内容块 W03：learned low-latency QEC modules

已有 AI pre-decoder、neural decoder、FiLM decoder 等说明 learned module 必须有清楚的输入输出契约和 latency role。本文 learned module 更窄：不是 per-shot decoder，而是 slow calibration component。

### 内容块 W04：real-time and FPGA QEC decoders

相关硬件 QEC 文献覆盖 lookup table、union-find、matching、greedy、clustering、qLDPC message passing 等。这些文献设定了完整 deployment claim 的标准：closed-loop timing、worst-case latency、resource、hardware integration。

本文贡献更窄：针对 GKP physical-layer correction，问的是 analog displacement rule 在 drift 下怎么更新。

### 内容块 W05：preserving analog information at the right layer

这一优势段说明本文把 analog histogram 用在更早的 physical GKP displacement rule 层，而不是只在外层 decoder 调权重。这可能让进入 surface-code、LDPC 或其他外层 decoder 的 residual 信息更好。

### 内容块 W06：drift adaptation without neural real-time critical path

端到端 neural decoder 表达力强，但 worst-case latency、fixed-point validation、fallback 行为更难证明。本文把 CNN 或 statcalib 放在慢环路，实时路径仍是 deterministic affine logic。

### 内容块 W07：replaceable calibration module

这一段强调同一个 runtime contract 可以容纳 teacher、learned residual 或 non-neural statistical calibration。重要对象是 histogram-driven affine calibration contract，不是某个固定 CNN。

### 内容块 W08：explicit deployment diagnostics

framework 把 quantization、clipping、saturation、stale-parameter、commit counts、fallback、update latency 变成 benchmark objects。这比只报 offline regression error 更接近 FPGA-QEC 的工程要求。

### 内容块 W09：compatibility with larger QEC stacks

因为 fast path 输出 physical displacement correction，而不是完整 outer-code decision，所以它原则上可以和 surface-GKP、bosonic-QLDPC 或 conventional stabilizer decoder 结合。当前只是兼容性论证，不是系统级 concatenated code 实测。

## 10. Experimental Setup 逐段解释

### 内容块 E01：software-HIL protocol

主实验使用 controlled software-HIL。它保留 fast loop、slow loop、parameter bank、stage-and-commit 接口，但实际实验在软件里执行。

这类实验的价值是：在没有真实 FPGA 板验证前，先检查运行时结构、接口、参数切换和逻辑错误代理指标。但它不测 board latency、hardware resource 或 deployment readiness。

### 内容块 E02：authoritative reference benchmark

权威参考 benchmark 包含四个 drift scenario、五个 comparison mode、paired seeds 和 two repeats。它足以在同一漂移轨迹、共享随机结构下比较 adaptive calibration laws。

但 later ablation、multi-seed、calibration-extension materials 要相对于这个 anchor 解释，不能把它们拿来重写主 benchmark。

### 内容块 E03：四个场景和五种模式

四个场景前面已经解释。五种 primary mode 是 EKF、UKF、Const-mu affine baseline、RLS residual-b baseline、learned hybrid residual branch。

`final_ler_mean` 越低越好。读表时不要把大数看成准确率；这里是错误率或错误代理，低才是优势。

## 11. Numerical Results 总体解释

### 内容块 Z01：结果证据阶梯

源稿把结果分成 evidence ladder：

1. 第一层：四场景 reference ranking。
2. 第二层：descriptive margins 和两个 formal paired-interval scenario rows。
3. 第三层：oracle、wrapped-Gaussian、residual-boundary、finite-squeezing、holdout、commit-lag 等诊断。
4. 第四层：operation-count、Q4.20 fixed-point、runtime-counter、hardware-measurement rows。

理解重点：越往后越偏解释、边界和未来验证，不要把所有表都当成同等级主性能证据。

## 12. Four-scenario affine benchmark 表格解释

### 内容块 T01：表格读法

表格列出四个场景和五种方法的 `final_ler_mean`。每行越低越好。加粗的是 Hybrid-b：

- `static_bias_theta`：Hybrid-b = 0.810902，UKF = 0.825370。
- `linear_ramp`：Hybrid-b = 0.787755，UKF = 0.811201。
- `step_sigma_theta`：Hybrid-b = 0.788800，UKF = 0.811548。
- `periodic_drift`：Hybrid-b = 0.806392，UKF = 0.821558。

### 内容块 T02：这张表能说明什么

它说明在当前受控四场景软件 HIL benchmark 中，Hybrid-b 四场景都排第一，UKF 四场景都排第二。这是 note 的 primary ranking result。

### 内容块 T03：这张表不能说明什么

它不能说明 Hybrid-b 在所有漂移类型、所有 seed、大规模 repeat、真实 `.tflite` runtime 或真实 FPGA 板上也一定第一。它也不是硬件 benchmark。

## 13. Descriptive UKF-versus-hybrid margins 表格解释

### 内容块 T04：为什么要看 margin

只看“第一名”可能不够，因为差距可能很小。UKF-minus-Hybrid margin 表告诉读者 Hybrid 比 UKF 好多少。`Delta = UKF final_ler - Hybrid final_ler`，正数表示 Hybrid 更低、更好。

### 内容块 T05：四个 margin 的含义

四个场景的 mean Delta 分别是：

- static：0.014469，相对下降 1.75%。
- linear：0.023446，相对下降 2.89%。
- step：0.022748，相对下降 2.80%。
- periodic：0.015166，相对下降 1.85%。
- 全场景平均 Delta：0.018957，平均相对下降约 2.32%。

### 内容块 T06：envelope 和 Delta/max SD 的边界

表中 envelope low/high 和 `Delta/max SD` 是 auditability diagnostics。它们帮助检查差距是否稳定，但不是 confidence interval、p-value、standard error，也不是 robustness proof。

## 14. Completed paired-interval scenario checks 表格解释

### 内容块 T07：为什么 paired interval 更强

paired repeat 的意思是同一组随机结构下比较 UKF 和 Hybrid，减少随机差异干扰。如果 12 对里全部 UKF-minus-Hybrid 为正，而且置信区间下界也为正，就比两 repeat 的描述性表更有说服力。

### 内容块 T08：static_bias_theta 行

`static_bias_theta` 有 12 对，mean Delta = 0.015563，SD = 0.001236，min-max 是 [0.013953, 0.018355]。paired-t 95% interval 是 [0.014778, 0.016348]，bootstrap interval 是 [0.014933, 0.016256]。

通俗解释：在这个场景中，12 次 paired 比较都显示 Hybrid 比 UKF 低，而且 interval 下界仍为正，所以这个场景的方向性比较可信。

### 内容块 T09：linear_ramp 行

`linear_ramp` 有 12 对，mean Delta = 0.022417，SD = 0.001892，min-max 是 [0.019951, 0.024946]。paired-t interval 是 [0.021215, 0.023619]，bootstrap interval 是 [0.021405, 0.023440]。

这说明在缓慢漂移场景下，Hybrid 相对 UKF 的优势也有更强的 paired interval 支持。

### 内容块 T10：这张表的限制

这张表只完成两个场景。它不支持 all-scenario repeat-expanded advantage，不支持 pooled p-value，不支持 holdout robustness，也不支持 hardware claim。

## 15. Repeat-expansion protocol checks 表格解释

### 内容块 T11：short-run protocol check 的目的

short-run runner matrix 使用同样四个预设场景和 UKF-vs-Hybrid 比较，但运行时长缩短，只有 two paired repeats。它的目的不是提供正式性能结果，而是检查命令形状、行计数和字段是否齐全。

### 内容块 T12：short-run 表格读法

四个场景都保留正 Delta：

- static：Delta 0.016927，relative reduction 2.07%。
- linear：Delta 0.026600，relative reduction 3.16%。
- step：Delta 0.012769，relative reduction 1.54%。
- periodic：Delta 0.028817，relative reduction 3.55%。

这些数说明扩展 runner 路线能跑通并保持正方向，但它不是正式 benchmark row。

### 内容块 T13：upgrade-threshold 表格读法

这个表告诉读者：哪些层现在能怎么写，哪些还需要升级。

- Reference ranking：能写 descriptive software-HIL ranking。
- Paired-repeat evidence：只能对 static 和 linear 写 scenario-level positive interval。
- Mechanism and estimator checks：只能写 feature sensitivity 和 affine-interface evidence。
- Controlled diagnostics：只能写 local-validity 和 stale-commit diagnostics。
- Datapath/runtime：只能写 fixed-point/software observability。

这张表是防 overclaim 的核心工具。

### 内容块 T14：figure-result sync 表格读法

这个表把 submission draft 中更多结果层同步到 note：主软件 HIL、ablation/mechanism、statcalib、controlled decoder/stress、channel bridge、datapath/runtime。

它的作用是让 note 不只是几句总结，而是保留 submission draft 中关键数值锚点。但它仍然保持边界：不是硬件结果，不是完整 expanded benchmark。

## 16. Feature and teacher ablations 表格解释

### 内容块 ABL01：ablation 表的目的

ablation 表不是新主榜单，而是用来理解哪些输入或 teacher 组件影响 Hybrid。读法是 Avg. LER 越低越好，Delta vs UKF 负数表示优于 UKF，Delta vs Hybrid Full 正数表示比完整 Hybrid 差。

### 内容块 ABL02：hybrid_full 行

`hybrid_full` Avg. LER = 0.798545，比 UKF 低 0.018837。这与主 benchmark 的方向一致，说明完整 Hybrid 有优势。

### 内容块 ABL03：no_hist_deltas 行

`hybrid_no_hist_deltas` Avg. LER = 0.826723，比 UKF 还差 0.009341，比 full hybrid 差 0.028178。

这说明 histogram deltas 很重要。没有“变化趋势”特征后，模型无法很好捕捉漂移。

### 内容块 ABL04：no_teacher_prediction 行

`hybrid_no_teacher_prediction` Avg. LER = 0.807251，仍优于 UKF，但比 full hybrid 差。这说明 teacher prediction 有帮助，但去掉后不是彻底崩溃。

### 内容块 ABL05：no_teacher_params 行

`hybrid_no_teacher_params` Avg. LER = 0.749621，是这张 ablation 表里最低的。这个结果很敏感：它反而说明不能简单宣称 teacher 参数“必不可少”。

因此源稿保持谨慎：teacher-side design matters，但目前不能讲 teacher necessity。

### 内容块 ABL06：no_teacher_deltas 行

`hybrid_no_teacher_deltas` Avg. LER = 0.800329，接近 full hybrid。说明某些 teacher delta 特征可能帮助有限，至少在这组 bounded ablation 中不是唯一关键因素。

### 内容块 ABL07：ablation 的总体结论

最稳妥结论是：histogram-delta 特征很重要；teacher-side 输入影响性能；但机制没有闭合，不能讲成简单因果故事。

## 17. Statistical calibration extension lane 表格解释

### 内容块 SC01：为什么 statcalib 重要

statistical calibration 是非神经方法，它也走同一个 affine runtime contract。如果它很强，就说明“从 histogram 估计 affine calibration”这个思想本身有价值。

### 内容块 SC02：scenario rows 的读法

表中 statcalib value 明显低于 UKF 和 Hybrid-b：

- static：best statcalib = 0.431708。
- linear：best statcalib = 0.467083。
- step：best statcalib = 0.460016。
- periodic：best statcalib = 0.438751。

这些数值比主表 Hybrid-b 低很多，看起来很强。

### 内容块 SC03：为什么不能直接提升为主 comparator

源稿明确说这个 protocol 不匹配主 ranking，而且 default 和 high-threshold variants 近乎并列，没有选出唯一 promoted setting。default variant 平均约 0.449254，high-threshold variant 平均约 0.449241，差距极小。

因此它只能支持 supplement-side affine-calibration evidence，不能替代主五模式 benchmark。

### 内容块 SC04：statcalib 对论文叙事的真正价值

它帮助把论文从“某个 CNN 获胜”转为“低维、histogram-driven、仿射校准 contract 是可行的”。这反而让论文更稳，因为主张不依赖单一 CNN 架构。

## 18. Mechanism probe for residual-b behavior 表格解释

### 内容块 ME01：机制探针想回答什么

这一节想知道 residual-b branch 为什么有效，是否存在某种 residual amplitude 或 clipping 机制。但源稿保持谨慎：当前只能描述现象，不能给因果闭环。

### 内容块 ME02：Gated-v5-minus-full 列

这个列如果为负，表示 Gated v5 比 full branch 更低。多 seed 中经常出现负值，说明 gated residual branch 在一些 seed 下确实改善 full branch。

### 内容块 ME03：I1-minus-Gated-v5 列

这个列衡量 lower-clip intervention 相对 Gated v5 的变化。正数表示 intervention 更差，负数表示更好。

六个 seed 中四个 harmful、一个 mixed/no clear effect、一个 helpful。也就是说 lower-clip intervention 大多数情况下有害。

### 内容块 ME04：机制结论

这张表支持“residual amplitude 参与某些失败模式”这种谨慎说法，但不支持“lower clip 是有效 mitigation”，也不支持“我们已经证明了因果机制”。

## 19. Unseen drift generalization 和 holdout stress 表格解释

### 内容块 H01：为什么需要 unseen drift

四个主场景是手工设计的。要证明真正 drift adaptation，需要看未见过的 drift family，例如 random walk、1/f-like drift、burst noise、coupled bias-variance drift、faster-than-window drift。

源稿说这些仍是 future benchmark lane，不是当前完成声明。

### 内容块 H02：holdout stress diagnostic 的性质

submission draft 加入了 random-walk、burst/reset、faster-than-window 三类 controlled non-hardware stress diagnostic。这里的 oracle affine 使用 known state，所以它是上界或诊断参考，不是实际 trained branch。

### 内容块 H03：holdout stress 表格读法

residual MSE 越低越好。三类 stress 中 Oracle 都最低：

- random_walk：Fixed 0.078869，Lagged 0.073867，Oracle 0.072685。
- burst_reset：Fixed 0.066865，Lagged 0.068622，Oracle 0.062144。
- faster_window：Fixed 0.068354，Lagged 0.068890，Oracle 0.063745。

这说明如果知道真实状态，affine 校准有 headroom。但 lagged affine 在 burst/reset 和 faster_window 中可能比 fixed 更差，说明 stale 参数会造成风险。

### 内容块 H04：surrogate F_avg 的边界

表中的 Oracle `F_avg^surr` 接近 1，不代表真实 finite-energy logical-channel fidelity。它只是由 residual-boundary identity rate 诱导的 surrogate，一种一致性桥接指标。

### 内容块 H05：lag-validity 表格读法

lag 表说明 stale commit 的影响：

- random_walk：lag 增大时 residual MSE 平滑变差。
- burst_reset：非单调，因为 reset timing 可能让旧参数偶然重新对上。
- faster_window：也非单调，因为相位对齐会影响结果。

这张表的价值是把 commit lag 从“工程细节”变成可测量变量。

## 20. Oracle and wrapped-Gaussian baselines 表格解释

### 内容块 O01：为什么需要 oracle 和 wrapped baseline

affine 方法需要和理论上更强的 baseline 比，例如 nearest-lattice hard decoding、static affine、oracle affine、wrapped-Gaussian 或 ML decoding。这样才能分清两种损失：

- fast path 被限制为 affine 带来的损失。
- slow loop 没有完美估计 noise state 带来的损失。

### 内容块 O02：one-step local-Gaussian 表格读法

residual MSE 越低越好。direct nearest-syndrome row 很差，因为它把 noisy wrapped syndrome 当成 hard correction。Oracle affine 在四个状态中都优于 fixed affine，step 后提升最大。

wrapped mean 只在 static state 略好，wrapped MAP 在 ramp、step、periodic 中更弱。因此当前不能说 wrapped posterior naive baseline 更强。

### 内容块 O03：这张表的正确结论

正确结论不是“affine 全局最优”，而是“需要 tuned sequence-level nearest-lattice 或 wrapped-posterior baseline 才能作为更强比较器”。当前 direct nearest 和 naive wrapped reference 只是 sanity check。

### 内容块 O04：sequence-level controlled baseline 表格读法

表中是 half-lattice residual-boundary crossing proxy，越低越好。Fixed 和 Oracle 在几个 sequence 状态下很低，而 wrapped mean/MAP 在 ramp、step、periodic 下 crossing proxy 更高。

这进一步说明 naive wrapped posterior branching 有实际风险，但仍不能证明 affine 全局最优。

## 21. Channel bridge 和 finite-squeezing toy-channel 表格解释

### 内容块 CH01：为什么要做 channel bridge

`final_ler`、Pauli-event language 和 fidelity language 容易被混淆。channel bridge 用 `p_any` 和 surrogate `F_avg` 把 residual-boundary crossing 转成一个可读的 Pauli-style surrogate，提醒读者这不是物理过程层析或真实 fidelity。

### 内容块 CH02：boundary surrogate block

固定 affine 和 oracle affine 的 mean `p_any` 都是 0.000127，worst 0.000467，surrogate fidelity 0.999915。Wrapped mean 和 wrapped MAP 的 crossing 更高，尤其 wrapped MAP mean 0.002129、worst 0.006450。

这说明在这些 controlled local states 中，naive wrapped MAP 反而更容易产生 boundary crossing。

### 内容块 CH03：toy-channel Delta sweep

toy channel 扫 `Delta = 0.18, 0.26, 0.34`。在较强 toy stress `Delta=0.34` 下，Hard nearest mean `p_any` = 0.001002，而 fixed/oracle affine 约 `1.6e-4`。

这支持一个谨慎说法：在该 toy diagnostic 下，硬 nearest-syndrome correction 不一定好。但这不是 finite-energy GKP logical-channel fidelity。

### 内容块 CH04：per-state residual-boundary rows

per-state 表显示 static 和 ramp 中 affine rows 没有 observed crossings；step 和 periodic 中 affine rows 仍低于 wrapped-posterior rows。

这张表帮助读者检查 aggregate 表不是隐藏了某个状态的大失败。

### 内容块 CH05：Q4.20 fixed-point rows

Q4.20 block 中 max diff 和 p99 diff 都在 `1e-6` 到 `2e-6` 量级，MSE delta 和 crossing delta 为 0，quantization saturation 为 0。

这说明在这些软件样本上，把 affine path 做成 Q4.20 固定点没有改变边界 crossing。但这仍只是软件 fixed-point emulation，不是板上 source-vs-board agreement。

## 22. Runtime, quantization, and fixed-point degradation 表格解释

### 内容块 RT01：runtime 边界总述

这一节强调 deployment story 不只看 logical performance。还要看 saturation、overflow、commit latency、stale-parameter penalty、fallback frequency、host-side update cost、source-vs-embedded drift。

当前最强 runtime-facing fact 很窄：selected preserved float/int8 `.tflite` artifacts 在一个 isolated local runtime environment 中能被加载和执行。

### 内容块 RT02：operation-count 表格读法

Affine fast path：4 次乘法、4 次加法、0 nonlinear ops、6 个 state scalars。

Wrapped MAP：49 次乘法、40 次加法。

Wrapped mean：99 次乘法、98 次加法、18 次 nonlinear ops。

这说明 affine path 作为实时路径显著更小，更适合固定点硬件。

### 内容块 RT03：Q4.20 rows 的含义

static、ramp、step、periodic 四行 max diff 分别约 `1e-6, 1e-6, 2e-6, 1e-6`，没有 crossing change。

这支持 fixed-point feasibility argument，但不是 synthesis/timing/resource/power evidence。

### 内容块 RT04：runtime counters 表格读法

五个 mode 的 commit 数约 899.8 到 899.9，slow violation 为 0，fast violation 约 0.0000158，overflow 约 0.0025，correction saturation 为 0。

这些计数器说明 software protocol 中 stage-and-commit contract 可观测、没有 slow violation、没有 correction saturation。但它不测 board commit latency 或硬件可靠性。

## 23. Embedded runtime and board-level validation 逐段解释

### 内容块 HW01：本节的核心边界

这一节非常明确：embedded runtime 和 board-level validation 与 simulation results 分开。local-runtime layer 只说明 selected `.tflite` artifacts 在 isolated software environment 执行。hardware layer 仍是 measurement target。

### 内容块 HW02：validation-status matrix 的读法

这张表把不同证据层同步到一个矩阵：

- Main ranking：四场景五模式 two repeats，Hybrid-b 四行最低。
- Completed interval rows：static 和 linear 完成 12/12 positive pairs。
- Runner check：短跑用于 row accounting，不是性能证据。
- Diagnostic layers：oracle/wrapped、holdout、lag、surrogate channel，用于暴露限制。
- Implementation feasibility：operation count、Q4.20、runtime counters。
- Runtime/supporting records：selected `.tflite` isolated runtime、source tables、metadata、checksums。
- Hardware-validation surface：列出未来板级证据需求。

### 内容块 HW03：analysis-file coverage 表的意义

coverage-treatment 表说明哪些数来自 checked analysis files，哪些是 protocol 或 requirement statements。它把 main performance、completed interval、controlled diagnostics、implementation feasibility、analysis/reporting maps、validation protocols 分开。

这对写论文很重要，因为审稿人会问：表格数字来自哪里？哪些是 measured result？哪些只是计划？

### 内容块 HW04：hardware measurement requirements 表格读法

hardware plan 表列出未来要做的真实硬件测量：

- board platform and host：Linux + FPGA host、device path、permissions、board model、driver version。
- bitstream/RTL/DMA：bitstream hash、AXI map、DMA histogram shape、element width、timeout policy。
- fast-path latency：p50/p95/p99/worst-case cycles。
- commit latency and reliability：stage、commit、ack、rollback、stale counters。
- resource and power：LUT/FF/DSP/BRAM、clock、timing closure、power。
- source-vs-board agreement：软件参考和板输出在共享 test vectors 上一致。

这张表不是结果表，而是未来硬件 claim 的最低证据清单。

## 24. Discussion 逐段解释

### 内容块 D01：主张不是“CNN universal decoder”

讨论第一段主动缩小主张：本文不是说 CNN 是万能 GKP decoder，而是说 drift-adaptive GKP correction 可以被组织成 low-dimensional affine calibration。teacher-CNN、statcalib 和 filters 都是同一个 runtime question 的不同答案。

### 内容块 D02：优势的两个层级

第二段区分 primary benchmark level 和 model-comparison level。primary benchmark 中 Hybrid-b 优于 filtering baselines；但 model-comparison 层面不能说 CNN 永远最好，因为 statcalib extension 也很强。

因此 learned branch 的价值是作为 replaceable slow-loop module，而不是唯一被验证的最终 comparator。

### 内容块 D03：系统级优势

第三段说系统优势来自 modularity under real-time constraints。未来设备可以换 slow estimator、teacher policy 或 fallback logic，同时保留 affine fast path。

这就是为什么本文强调 contract 而不只是模型精度：contract 让估计器复杂度和更新时间可以独立调整。

### 内容块 D04：结果层要按 tier 阅读

第四段重申 evidence tier：reference ranking 和两个 completed paired-interval scenarios 是最强性能证据；ablation、statcalib、oracle/wrapped、channel surrogate、holdout、fixed-point、runtime-counter 是解释和边界。

任何 board latency、resource、power、deployment portability、source-vs-board agreement 都需要未来 Linux + FPGA measurement path。

## 25. Conclusion 逐段解释

### 内容块 K01：结论第 1 段

结论重新陈述方法：把 drift-adaptive GKP correction 写成 dual-loop affine calibration problem。快环路是 bounded fixed-point rule `Delta = K s + b`，慢环路根据 recent syndrome statistics 更新参数。

这提供了 fixed decoder 和 full neural decoder 之间的折中：保留自适应能力，又不把黑箱网络放到每次 shot 的实时路径。

### 内容块 K02：结论第 2 段

结论总结两个主要实验结论：

第一，受控四场景 software-HIL 中 teacher-anchored residual branch 相对 filtering baselines 改善，UKF-relative mean reductions 为 1.75% 到 2.89%，且两个场景有 12/12 positive interval checks。

第二，supplementary statcalib 说明简单非神经估计器也可以在同一 runtime contract 下很强，但当前证据没有选择唯一 promoted threshold。

因此最强科学故事是“低维 histogram-driven affine calibration 是有希望的结构”，而不是“某个 CNN 已经完成全部比较”。

### 内容块 K03：结论第 3 段

最后一段列出 note 已同步的实验层：四场景 ranking、两个 paired-interval rows、oracle/wrapped diagnostics、residual-boundary surrogates、holdout/lag stress、operation counts、fixed-point emulation、runtime counters。

剩余缺口也明确：all-scenario repeat-expanded intervals、stronger sequence-level baselines、default-environment `.tflite` portability、board-level timing/resource claims。

最终结论是：这是一份 evidence-synchronized theory note，有清楚硬件和 portability 限制，不是 deployment validation。

## 26. 如何向读者解释这些实验结果

### 26.1 最重要的主结果怎么说

可以说：在一个固定、受控、软件 HIL 的四场景 benchmark 中，Hybrid-b 在四个漂移场景都获得最低 `final_ler_mean`，UKF 是稳定 runner-up。相对 UKF 的平均下降大约 2.32%。

不应说：Hybrid-b 已经在所有 GKP 漂移条件、真实硬件或完整外层 QEC stack 中验证优于所有方法。

### 26.2 两个 paired interval 怎么说

可以说：`static_bias_theta` 和 `linear_ramp` 两个场景完成了 12 对 paired repeat，所有 paired deltas 都为正，paired-t 和 bootstrap interval 下界为正，因此这两个场景的 UKF-vs-Hybrid 方向更有统计支撑。

不应说：所有四个场景都已经完成同等级 paired interval validation，或者已有 pooled all-scenario inference。

### 26.3 ablation 怎么说

可以说：去掉 histogram deltas 后性能明显变差，说明时间变化特征对漂移适应很重要。teacher-side features 影响性能，但 no-teacher-params 行表现异常强，说明不能简单宣称 teacher 必不可少。

不应说：ablation 已证明完整因果机制，或证明 teacher 是唯一必要组件。

### 26.4 statcalib 怎么说

可以说：一个非神经 statistical calibration 分支在单独 extension lane 中很强，支持“histogram-driven affine calibration contract”这个更宽的系统观点。

不应说：statcalib 已经成为主 comparator，或已经选出唯一最佳阈值。

### 26.5 holdout 和 oracle 怎么说

可以说：known-state oracle affine 在 controlled holdout stress 中显示仿射校准有 headroom；lagged affine 显示 stale 参数会在 burst/reset 和 faster-than-window 情况下造成风险。

不应说：trained branch 已经完成 unseen drift robustness proof。

### 26.6 Q4.20 和 runtime counters 怎么说

可以说：软件固定点仿真显示 Q4.20 对受控样本的 correction 差异在 `1e-6` 量级，没有 crossing change；runtime counters 显示软件 stage-and-commit 可观测。

不应说：这等同于 FPGA timing closure、resource/power measurement 或 source-vs-board agreement。

## 27. 对作者后续整理稿件的建议

1. 主文结果应围绕四场景 benchmark 和两个 completed paired-interval rows 展开，不要让过多诊断表淹没主线。
2. 把 statcalib 明确放在 supplement-side 或 appendix-side，强调它支持 affine calibration contract，而不是推翻主 comparator。
3. 把 hardware plan 表保留为 requirements，不要写成 results。
4. 对 every table 都在 caption 或正文里写清楚“lower is better/positive Delta means Hybrid better/diagnostic only/not hardware evidence”。
5. 如果未来要投稿完整论文，最需要补齐的是另外两个场景的 formal paired intervals、predeclared holdout drift benchmark、tuned sequence-level nearest/wrapped baseline、真实板级 source-vs-board agreement 和 latency/resource/power 数据。

