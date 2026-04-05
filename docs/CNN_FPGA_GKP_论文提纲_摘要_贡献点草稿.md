# CNN-FPGA-GKP 论文提纲、摘要与贡献点草稿

## 1. 文档目的

本文档用于在当前项目进度基础上，提前形成一版可继续迭代的论文写作草稿。  
它不是最终论文，也不是阶段结论文档的重复，而是为后续论文撰写提供以下三类可直接复用材料：

1. 论文提纲
2. 摘要草稿
3. 贡献点草稿

当前草稿基于已经拿到的正式结果撰写，特别参考了：

- [CNN_FPGA_GKP_阶段结论.md](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/docs/CNN_FPGA_GKP_阶段结论.md)
- [CNN_FPGA_GKP_项目完成目标与投稿路线报告.md](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/docs/CNN_FPGA_GKP_项目完成目标与投稿路线报告.md)
- [P4_UKF正式结果与Hybrid机制分析.md](/Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/docs/P4_UKF正式结果与Hybrid机制分析.md)

需要注意：

- `teacher` 组正式 ablation 已有结果；
- `context` 组已有单场景 smoke 结果；
- `features` 组正式版仍在后台运行，因此本草稿对 `features` 的描述仍保留为“待最终回填”的形式。

---

## 2. 建议论文标题方向

下面给出几组风格不同、但都和当前项目状态相符的标题方向。

### 2.1 工程系统型标题

1. `A Runtime-Consistent CNN-FPGA Adaptive Decoder for Drift-Aware GKP Error Correction`
2. `A Dual-Loop CNN-FPGA Framework for Drift-Adaptive GKP Error Correction`
3. `Deployment-Aligned Adaptive Decoding for GKP Codes via a Dual-Loop CNN-FPGA Architecture`

适用场景：

- 投稿偏工程、系统、量子工程；
- 强调 runtime、HIL、部署一致性；
- 适合 `QCE / TQE / EPJ Quantum Technology` 风格。

### 2.2 方法机制型标题

1. `Teacher-Guided Residual Adaptive Decoding for GKP Error Correction Under Real-Time Hardware Constraints`
2. `Residual-B Adaptive Decoding Improves Classical Drift Tracking for GKP Error Correction`
3. `Teacher-Guided Residual Correction for Runtime-Consistent Adaptive GKP Decoding`

适用场景：

- 强调 `Hybrid Residual-B` 的方法论价值；
- 突出 `Hybrid > UKF` 的正式结论；
- 适合把“为什么 hybrid 有效”写成主线。

### 2.3 系统+方法折中标题

1. `A Dual-Loop Teacher-Guided Residual Decoding Framework for Real-Time GKP Error Correction`
2. `Runtime-Consistent Teacher-Guided Residual Decoding with CNN-FPGA Co-Design for GKP Codes`

如果要在当前阶段给出一个最稳妥的工作标题，我建议优先考虑：

> `A Dual-Loop Teacher-Guided Residual Decoding Framework for Real-Time GKP Error Correction`

原因：

- 不把贡献写窄到“只是 CNN”；
- 不把结论写满到“优于所有经典解码器”；
- 同时保留系统和方法两个层面的亮点。

---

## 3. 论文核心主张

当前最稳的论文主张不应是：

- “CNN 全面优于所有经典解码器”

而应是：

- 在实时硬件约束下，单纯做绝对参数回归的 CNN 并不是最有效的在线解码路线；
- 更合理的做法是保留稳定的经典 teacher，并让轻量 CNN 仅学习对实际控制参数有用的残差修正；
- 在运行时一致的数据构造、双回路调度和软件 HIL 约束下，这种 `teacher-guided residual-b` 方案能够稳定超过当前最强经典自适应 baseline `UKF`。

这条主张的好处是：

1. 与已有正式结果一致
2. 不夸大当前结论
3. 更容易经受审稿时对公平性和可部署性的追问

---

## 4. 论文摘要草稿

下面给出三版摘要草稿，分别对应短摘要、标准摘要和偏工程风格摘要。

### 4.1 标准摘要草稿

> Continuous-variable GKP error correction under realistic hardware constraints requires adaptive decoding that can track drift while respecting strict latency budgets. Directly deploying a learned model as a full replacement for classical online estimation is often brittle, because the offline prediction target may not align with the runtime control semantics used by the decoder. In this work, we present a runtime-consistent dual-loop CNN-FPGA decoding framework for drift-adaptive GKP error correction. The fast loop executes low-latency linear decoding with staged parameter-bank switching, while the slow loop uses histogram windows to update decoder parameters under realistic scheduler, latency, and commit constraints. Rather than regressing absolute physical parameters end-to-end, our final method adopts a teacher-guided residual strategy, where a lightweight CNN predicts a residual correction to the runtime control bias on top of a stable classical teacher. Across formal multi-scenario benchmarks, the resulting Hybrid Residual-B decoder achieves lower logical error rate than a set of classical adaptive baselines, including EKF, RLS residual decoding, constant residual compensation, and a corrected UKF baseline. In the current formal setting, Hybrid Residual-B reaches an average logical error rate of 0.798332, compared with 0.817974 for UKF, the strongest classical baseline. We further show that this gain is not explained by more aggressive control or correction saturation, but is consistent with improved residual correction under runtime-consistent deployment semantics. These results suggest that lightweight teacher-guided residual learning is a practical route for real-time adaptive quantum error correction systems.

### 4.2 偏工程投稿风格摘要

> Real-time quantum error correction requires decoders that are not only accurate in offline evaluation, but also consistent with deployment-time latency, buffering, parameter switching, and hardware update constraints. We develop a dual-loop CNN-FPGA framework for drift-adaptive GKP error correction, in which a fast FPGA-side loop performs low-latency linear decoding and histogram accumulation, while a slow host-side loop estimates decoder updates from windowed statistics. We show that direct absolute-parameter regression is not the most reliable runtime strategy. Instead, a teacher-guided residual-b formulation, trained on runtime-consistent data and deployed through staged parameter-bank commits, yields the strongest formal performance among the tested approaches. In multi-scenario long-run benchmarks, the proposed Hybrid Residual-B decoder outperforms EKF, RLS residual decoding, constant residual compensation, and a corrected UKF baseline, while preserving zero correction saturation and zero aggressive-parameter events. The results support a deployment-aligned view of learning-assisted decoding: the most useful learned component is not necessarily a full estimator, but a lightweight residual corrector that complements a stable classical teacher.

### 4.3 中文摘要草稿

> 在具有实时硬件约束的 GKP 量子纠错中，解码器不仅要具备离线预测精度，还必须满足在线调度、窗口缓冲、参数切换和时延预算等部署约束。本文提出一套面向漂移自适应 GKP 解码的双回路 CNN-FPGA 框架：快回路在 FPGA 侧执行低时延线性解码与直方图累积，慢回路在窗口级别利用统计信息更新解码参数。我们发现，直接让 CNN 回归完整绝对物理参数并不是最稳妥的在线方案；更有效的做法是保留稳定的经典 teacher，并让轻量 CNN 仅学习对运行时控制偏置有用的残差修正。基于运行时一致数据集、参数双缓冲提交和软件 HIL 约束构建的正式多场景 benchmark 表明，最终的 Hybrid Residual-B 方案稳定优于 EKF、RLS residual、常数残差补偿以及修正后的 UKF 等经典自适应基线。在当前正式设置下，Hybrid Residual-B 的平均逻辑错误率为 0.798332，而最强经典基线 UKF 为 0.817974。进一步分析表明，该优势并非来自更激进的控制参数或校正饱和，而更符合“在部署语义一致条件下对 teacher 残余偏差进行有效修正”的机制图景。结果说明，teacher-guided residual learning 是面向实时量子纠错系统的一条可行工程路线。

### 4.4 摘要中后续需要替换的部分

在 `features` 正式结果出来后，摘要中还可以补一句更强的机制性表述，例如：

- 哪一类输入通道对性能最关键；
- 去掉哪一类 teacher 条件后退化最明显；
- 这是否支持“residual learner 主要在修正 teacher 的系统偏差”。

目前这一句先不写死，等正式结果出来后再补最稳妥。

---

## 5. 论文贡献点草稿

下面给出一版当前最稳的贡献点写法。

### 5.1 三点式贡献写法

1. 我们提出了一套面向实时硬件约束的双回路 CNN-FPGA 自适应 GKP 解码框架。

- 该框架将低时延快回路解码、窗口级统计累积、参数 bank 切换、慢回路推理和 HIL 验证组织为同一条可复现实验链路；
- 使研究问题从“离线模型精度”提升为“部署语义一致的在线解码系统”。

2. 我们提出并验证了 teacher-guided residual-b 的运行时一致学习方案。

- 相比直接回归绝对物理参数，该方案更贴近在线解码真正使用的控制语义；
- 在当前正式多场景 benchmark 中，它稳定优于一组经典自适应基线；
- 在现有正式结果下，`Hybrid Residual-B` 的平均 `LER` 为 `0.798332`，优于当前最强经典 baseline `UKF = 0.817974`。

3. 我们给出了从物理仿真、数据构建、量化 artifact 到软件 HIL 的系统化工程验证。

- 包括 float / int8 / TFLite artifact 验证；
- 包括 overflow 来源定位与输入统计范围修正；
- 包括 teacher、context 以及 features 等 ablation 路线；
- 说明该方法并非单一模型技巧，而是一套可落地的工程方法。

### 5.2 四点式贡献写法

如果投稿目标更偏系统/工程，也可以拆成四点：

1. 提出双回路实时解码架构  
2. 提出 runtime-consistent residual-b 学习目标  
3. 证明 `Hybrid Residual-B` 稳定超过强经典 baseline `UKF`  
4. 建立从 artifact 到 HIL 的工程验证闭环

### 5.3 不建议写成的贡献点

当前不建议把贡献写成：

1. “提出首个优于所有经典解码器的量子纠错 CNN”
2. “提出通用于所有量子纠错码的统一解码器”
3. “完成了完整真实 FPGA 部署并实现工业级可用”

这些写法都超出了当前证据边界。

---

## 6. 论文结构提纲

下面是一版更贴近最终成稿的论文结构。

## 6.1 Introduction

建议重点回答四个问题：

1. GKP 在线纠错为什么需要漂移自适应？
2. 为什么单纯离线回归精度不足以说明在线可用性？
3. 实时硬件约束下，什么才是合理的解码问题定义？
4. 本文的核心贡献是什么？

建议在引言结尾明确写出：

- 我们不把学习模型当成 classical teacher 的完全替代；
- 而是将其设计为满足运行时语义的 residual corrector。

## 6.2 Background and Problem Formulation

可包含：

1. GKP syndrome 与漂移参数背景
2. 快回路与慢回路的时间尺度差异
3. 在线解码的部署约束
4. 为什么 `(K, b)` 才是运行时真正执行的目标

这一节的核心作用是把“预测物理参数”和“驱动实时解码控制”的区别讲清楚。

## 6.3 Dual-Loop Runtime-Consistent Decoding Framework

建议拆成以下小节：

1. Fast loop
- 线性解码
- 直方图累积
- fixed-point / cycle budget

2. Slow loop
- histogram window
- teacher estimation
- residual model inference
- parameter mapping

3. Parameter update protocol
- param bank
- stage / commit
- atomic switch
- latency and scheduler

4. HIL-consistent execution
- mock FPGA
- software HIL
- artifact runtime

这节是系统论文的核心。

## 6.4 Runtime-Consistent Learning Formulation

这是方法部分的核心。

建议说明：

1. 为什么 absolute parameter regression 不够理想
2. 为什么目标应从 `delta_mu` 进一步切到 `delta_b`
3. 为什么 teacher-guided residual 更符合部署语义
4. 输入由哪些部分构成：
- 当前窗口 histogram
- 历史窗口
- histogram delta
- teacher prediction
- teacher params
- teacher deltas

这一节最好配一张结构图。

## 6.5 Experimental Protocol

建议写清：

1. 正式场景集
- `static_bias_theta`
- `linear_ramp`
- `step_sigma_theta`
- `periodic_drift`

2. repeats、seed、固定协议
3. baseline 集合
- `EKF`
- `UKF`
- `Constant Residual-Mu`
- `RLS Residual-B`
- `Hybrid Residual-B`

4. 评价指标
- `LER`
- `overflow_rate`
- `histogram_input_saturation_rate`
- `commit`
- `slow_update_violation_rate`
- `fast_cycle_violation_rate`

## 6.6 Main Results

这一节建议先写正式强 baseline 结果：

1. `Hybrid Residual-B`
2. `UKF`
3. `Constant Residual-Mu`
4. `RLS Residual-B`
5. `EKF`

重点突出：

- `Hybrid > UKF`
- 优势不是 saturation 换来的
- overflow 差异很小，但 LER 仍明显更低

## 6.7 Mechanism Analysis and Ablation

建议按如下顺序：

1. `teacher` ablation
- `UKF`
- `Hybrid Teacher=WV`
- `Hybrid Teacher=UKF`

2. `context` ablation
- `Ctx1`
- `Ctx3`
- `Ctx5`

3. `features` ablation
- `Hybrid Full`
- `No HistDelta`
- `No TeacherPred`
- `No TeacherParams`
- `No TeacherDelta`

其中 `features` 这一块在当前仍待结果回填，但结构可以先固定。

## 6.8 Engineering Considerations

建议纳入：

1. float / int8 / TFLite artifact
2. latency budget
3. stage/commit behavior
4. overflow source analysis
5. board backend 现状与限制

## 6.9 Discussion

建议讨论：

1. 当前优势为什么更像“对 teacher 偏差的残差修正”
2. 为什么不应把工作表述成“全面替代经典解码器”
3. 未来如何扩展到 concatenated GKP-surface
4. 与 `MWPM / Blossom` 的合理关系是什么

## 6.10 Conclusion

建议结论聚焦：

1. 运行时一致性是关键
2. teacher-guided residual 是当前最有效路线
3. 当前工作是一套可部署的自适应量子纠错工程框架

---

## 7. 每节可直接展开的写作要点

为了方便后续正式写作，下面给出每节的可直接展开内容。

### 7.1 Introduction 可直接写的要点

1. 量子纠错在线部署和离线模型评估是两个不同问题
2. 对 GKP 这类连续变量系统，漂移和偏置会导致固定参数解码逐渐失配
3. 仅依靠经典 teacher 有上限，仅依靠端到端 CNN 又容易和运行时语义错位
4. 因此需要一种既能保持稳定 teacher，又能利用学习模块补残差的方案

### 7.2 Method 可直接写的要点

1. residual-b 的目标比绝对参数更直接服务于快回路
2. teacher 输入使模型不必重新学习全部物理结构
3. context 和 histogram delta 让模型能看到短时变化趋势
4. param mapper 把高层预测严格映射到 `(K, b)`，避免训练部署漂移

### 7.3 Results 可直接写的要点

1. 正式结果中 `UKF` 已成为最强经典 baseline
2. 在此基础上 `Hybrid Residual-B` 仍保持稳定领先
3. 该优势并非来自更激进参数
4. overflow 主导来源仍是 histogram input，说明提升主要来自估计质量而非控制强度

### 7.4 Discussion 可直接写的要点

1. 当前工作说明“学习型慢回路”最合理的角色是 residual corrector
2. 真正重要的是 runtime-consistent data + deployment semantics
3. 未来扩展到其他码时，迁移的是框架，不是现成模型

---

## 8. 当前论文中最需要的图和表

为了后续写稿高效，建议尽早冻结以下图表结构。

### 8.1 主结果表

列建议：

- Method
- Avg. LER
- Avg. Overflow
- Avg. Histogram Saturation
- Notes

行建议：

- `EKF`
- `UKF`
- `Constant Residual-Mu`
- `RLS Residual-B`
- `Hybrid Residual-B`

### 8.2 分场景结果表

列建议：

- Scenario
- `EKF`
- `UKF`
- `Constant Residual-Mu`
- `RLS Residual-B`
- `Hybrid Residual-B`

### 8.3 teacher ablation 表

列建议：

- Method
- Avg. LER
- Delta vs UKF
- Interpretation

### 8.4 context ablation 表

列建议：

- Method
- Scenario
- LER
- Delta vs Hybrid Full

### 8.5 features ablation 表

列建议：

- Variant
- Removed Channel
- Avg. LER
- Delta vs Hybrid Full
- Delta vs UKF
- Likely Meaning

### 8.6 系统图

至少建议三张图：

1. 双回路架构图
2. runtime-consistent learning pipeline 图
3. baseline / ablation 结果柱状图

---

## 9. 当前最推荐的论文叙事版本

如果只选一个当前最稳的叙事，我建议如下：

> 本文研究在实时硬件约束下如何实现漂移自适应的 GKP 在线解码。我们发现，直接用 CNN 回归完整物理参数并不是最稳妥的部署方案；更有效的是在稳定 classical teacher 的基础上，用轻量学习模块仅修正运行时真正使用的控制偏置。为此，我们构建了一个运行时一致的双回路 CNN-FPGA 解码框架，并在正式多场景 benchmark 中证明：该 teacher-guided residual-b 方法不仅优于 EKF、RLS residual 和常数残差补偿，而且在当前修正后的最强经典 baseline UKF 之上仍保持稳定增益。

这段几乎可以直接变成：

- 摘要最后两句
- 引言末尾一段
- 结论第一段

---

## 10. 后续待回填项目

在 `features` 正式结果出来后，本文档建议补以下内容：

1. `features` ablation 的正式数值表
2. 对每类输入通道作用的机制解释
3. 最终摘要中的一句机制性总结
4. 论文贡献点是否需要把“特征消融结果”单列出来

在真实板卡或更多系统指标补齐后，建议再补：

1. 资源/时延表
2. artifact / backend 比较表
3. HIL 与真实 board 的对接限制说明

---

## 11. 当前一句话结论

如果现在要用一句话概括整篇论文的核心贡献，最稳妥的版本是：

> 我们证明了：在部署语义一致的双回路实时解码框架下，保留经典 teacher 并只让轻量 CNN 学习 residual-b 修正，比直接做绝对参数回归更有效，也比当前最强经典自适应 baseline `UKF` 更优。
