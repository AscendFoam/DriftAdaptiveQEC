# CNN-FPGA-GKP 项目完成目标、论文定位与投稿路线报告

## 1. 文档目的

本文档用于把此前关于“项目做到什么程度算完成、是否适合写论文、适合投哪里、还应补哪些 baseline、能否迁移到其他量子纠错码”的讨论，整理成一份更完整、更接近当前实际进度的书面报告。

这份报告不是单纯重复旧对话，而是在旧判断基础上，结合当前已经拿到的正式实验结果，给出一个更稳妥的工程与论文判断。

本文档主要回答七个问题：

1. 这个项目做到什么程度可以认为“主线完成”？
2. 以目前进度，项目是否已经具有论文价值？
3. 如果写论文，论文应该怎么定位，而不该怎么定位？
4. 目前更适合投哪些期刊或会议？
5. 在投稿前还缺哪些关键证据？
6. 还应增加哪些 baseline 才更有说服力？
7. 当前 CNN-FPGA / hybrid 解码框架能否迁移到其他量子纠错码？

---

## 2. 当前项目实际位置

### 2.1 已经完成的核心工作

从“只有 CNN 训练脚手架”的早期状态，到当前代码与实验状态，项目已经完成了以下关键跨越：

1. 物理仿真与工程链路已打通

- 已从单纯的离线回归训练，推进到“物理仿真 -> 运行时一致数据集 -> 双回路调度 -> 软件 HIL -> 报告导出”的完整链路。
- 这意味着当前项目已经不再只是一个“训练一个 CNN 看误差”的小实验，而是一套面向在线解码部署的工程原型。

2. 静态参数回归模型已完成验收

- 在静态参数辨识任务上，`Tiny-CNN` 浮点与 `int8` 量化模型都已达到较高精度。
- 对 `sigma / mu_q / mu_p / theta_deg` 的回归已经不是当前主瓶颈。
- 这部分更像是“可用的慢回路感知器件”已经建立完成。

3. 运行时主线已从“绝对参数回归 CNN”切换到“teacher + residual-b”

- 当前主线不再是让 CNN 直接输出全部绝对物理参数。
- 现有结果已经表明，更有效的路线是：
  - 先用稳定、可解释的经典估计器给出 teacher；
  - 再由轻量 CNN 学习小幅、在线、短时一致的残差修正；
  - 最终映射到快回路真正使用的 `(K, b)`。

4. 软件侧 HIL 链路已具备正式实验能力

- 当前已经具备：
  - 双回路 scheduler
  - param bank
  - latency injection
  - inference service
  - mock FPGA / board backend 骨架
  - `.tflite` 导出与独立评测
- 也就是说，论文中“工程系统”这一块已经有了实质内容，不再只是概念图。

5. 已完成正式强 baseline 对比

- 当前已经不只是和 `Window Variance / EKF` 这样较弱基线比较。
- 项目已经补入并跑通了更强经典对照，包括：
  - `UKF`
  - `RLS Residual-B`
  - `Constant Residual-Mu`
- 在这组更强、更公平的正式比较中，当前主线 `Hybrid Residual-B` 仍保持最优。

### 2.2 当前最重要的正式结果

截至目前，最值得用于论文主结论的结果不是早期的小样本 smoke，也不是单次 seed 的局部现象，而是正式长配置、多场景 benchmark。

当前正式强 baseline 结果为：

1. `Hybrid Residual-B = 0.798332`
2. `UKF = 0.817974`
3. `Constant Residual-Mu = 0.825719`
4. `RLS Residual-B = 0.827908`
5. `EKF = 0.828369`

这组结果说明：

- 修正后的 `UKF` 已经成为当前最强经典 baseline；
- 但 `Hybrid Residual-B` 仍然稳定领先 `UKF`；
- 平均 `LER` 优势约为 `0.019642`。

更重要的是，这个优势不是靠更激进的控制换来的：

- `correction_saturation_rate = 0`
- `aggressive_param_rate = 0`
- overflow 主导来源仍然是 `histogram_input`

因此，当前最合理的解释不是“CNN 把系统推得更狠”，而是：

- `Hybrid` 在 teacher 已经不差的前提下，确实学到了对运行时有用的残差修正。

### 2.3 当前最新 ablation 进度

除正式强 baseline 外，当前还新增了 `Hybrid vs UKF` 的机制分析与 ablation 路线。

已拿到的阶段性结果包括：

1. `teacher` 组已完成正式运行

跨四个场景的平均 `LER`：

- `UKF = 0.819012`
- `Hybrid Teacher=WV = 0.799105`
- `Hybrid Teacher=UKF = 0.803542`

这说明：

- 即使把 teacher 升级为 `UKF`，residual learner 仍然有额外价值；
- 但当前 `teacher = window_variance` 的主模型仍略强于 `teacher = ukf` 版本。

2. `context` 组已完成单场景 smoke

在 `static_bias_theta` 上：

- `UKF = 0.826406`
- `Hybrid Ctx1 = 0.824150`
- `Hybrid Ctx3 = 0.838388`
- `Hybrid Ctx5 = 0.813885`

这说明：

- 当前不是“上下文越长越好”这种简单关系；
- `Ctx5` 目前最好；
- `Ctx3` 的退化更像训练或特征组合问题，而不是框架逻辑错误。

3. `features` 组正式版仍在后台运行

因此，对“到底是 histogram delta、teacher prediction、teacher params 还是 teacher deltas 更关键”这个问题，最终结论还需要等待正式结果。

### 2.4 还没有完成的部分

虽然项目已经具有论文价值，但严格说并不能称为“全部完成”。还未完成或尚未完全收口的部分主要包括：

1. `features` 组正式 ablation 尚未跑完
2. 真板卡语义尚未与真实地址表/寄存器表完成联调
3. 更强离散码 baseline 尚未接入
4. 跨码迁移验证尚未开展
5. 论文级资源、时延、吞吐、复杂度分析还需系统整理

因此，当前更准确的说法是：

- 主线工程已经基本成形；
- 主线论文叙事已经成立；
- 但距离“非常稳的投稿版本”还差若干补强工作。

---

## 3. 项目做到什么程度算“完成”

我不建议把项目完成标准定义成：

- “CNN 全面打败所有经典解码器”
- “必须在所有码、所有场景、所有 baseline 上第一”
- “必须等真板卡全部落地后才算完成”

这种定义不现实，也不适合作为工程项目的验收目标。

更合理的完成标准应分三档。

### 3.1 第一档：可交付工程版

满足以下条件，就可以认为项目主线已经完成，具备一个可以交付、可以讲清楚、可以复现实验的工程版本：

1. 正式协议冻结

- 场景集固定
- seed 固定
- repeats 固定
- 报告模板固定
- baseline 集合固定

2. 主方案稳定成立

- `Hybrid Residual-B` 在正式口径下稳定优于 `EKF / Window Variance / Constant Residual-Mu`
- 并且在加入 `UKF` 之后仍然不劣

3. 运行时链路闭环完整

- runtime-consistent dataset
- slow loop inference
- param mapper
- param bank stage/commit
- fast loop emulator
- software HIL
- float / int8 / TFLite artifact

4. 主瓶颈已定位

- overflow 来源已明确
- 输入范围调优有效
- 能解释为什么之前表现不稳，以及为什么现在更稳

5. 关键 ablation 至少完成一轮

- teacher
- context
- features
- EMA 或后处理
- 输入统计范围

按这个标准看，项目已经非常接近这档，甚至主线部分已经基本达标。

### 3.2 第二档：可稳妥投稿版

如果目标是“可以认真写论文并投出去”，我建议至少再补齐以下内容：

1. 正式 ablation 全部跑完并整理

- 当前 `teacher` 已有正式结果
- `context` 已有 smoke 结果
- `features` 正式结果还需要补齐

2. 更完整的稳定性复验

- 多 seed
- 更长配置
- 对关键场景如 `static_bias_theta` 做单独复查

3. 复杂度和工程代价分析

- 推理时延
- stage/commit 更新频率
- `int8` 与 float 的差异
- 软件 HIL 开销
- 若可能，FPGA 侧资源预算或估算

4. 机制解释要写扎实

- 为什么纯 CNN 绝对参数回归不够好
- 为什么 teacher-guided residual 更有效
- 为什么优势不是 overflow 偶然差异导致
- 为什么优势不是更激进控制导致

按当前状态看，项目已经进入这一档的前半段，但还没完全收口。

### 3.3 第三档：更强期刊版

如果目标是冲更强、更挑剔的期刊或更高层次的系统/量子期刊，还需要再补几项：

1. 真实板卡联调结果

- 至少有一套真实寄存器表
- 至少完成一轮真实 board backend 的语义对齐
- 最好有真实板卡上的时延/吞吐/稳定性指标

2. 跨码验证

- 例如扩展到 concatenated GKP-surface
- 或至少证明该框架可以迁移到离散 syndrome 图码族

3. 更强、更多样的 baseline

- 对当前连续变量漂移问题：`UKF / RLS / PF` 已不错
- 若扩展到 concatenated 或离散图码：应加入 `MWPM / Blossom / Union-Find / BP+OSD`

4. 真正完整的系统论文要素

- 资源
- 延迟
- 精度
- 稳定性
- 可部署性
- 失败模式与边界条件

---

## 4. 这个项目现在适不适合写论文

### 4.1 结论：适合

当前项目已经具备论文价值，而且不是勉强凑一篇短文的程度。

理由很明确：

1. 你已经不是只有模型精度

很多项目最后只有一张“训练误差下降”的图，这种很难形成有说服力的论文。  
当前项目已经有：

- 物理仿真基础
- 运行时一致数据集
- 双回路调度
- 软件 HIL
- float/int8/TFLite 链路
- 正式 benchmark
- 强经典 baseline

这使它具有系统型和工程型工作的完整骨架。

2. 你已经有“比谁强”之外的内容

审稿人真正关心的通常不是“你比某个 baseline 低 2%”，而是：

- 你解决了什么真实问题？
- 你的工程约束是什么？
- 你为什么要这样设计？
- 你的方法为什么有效？
- 它和已有方法相比多了什么能力？

当前项目已经能回答这些问题的一大半。

3. 你已经有一个比较稳的核心叙事

当前最稳的论文主叙事不是：

- “CNN 比所有经典解码器更强”

而是：

- “在实时硬件约束下，teacher-guided residual 学习型慢回路，可以稳定超过轻量经典自适应基线，且与双回路 FPGA/HIL 工程链路一致”

这个叙事比单纯比大小更稳，也更像可以投稿的研究问题。

### 4.2 但不建议怎样写

当前阶段不建议把论文写成以下风格：

1. “深度学习全面替代经典量子纠错解码器”

原因：

- 现有结果并不支持这种极强说法；
- 当前最强经典 baseline `UKF` 仍然很强；
- 未来如果接入更重的离散码 baseline，直接宣称“全面替代”风险很高。

2. “我们提出了一个通用于所有量子纠错码的最优方法”

原因：

- 目前结果主要建立在 GKP 连续变量漂移跟踪场景；
- 框架可迁移，不等于模型和结论可直接迁移。

3. “本文主要贡献是 CNN 的预测精度很高”

原因：

- 纯回归精度并不是这个项目现在最有价值的部分；
- 真正有价值的是“与运行时语义一致的在线自适应解码框架”。

---

## 5. 更适合的论文定位

### 5.1 最稳的定位

如果以当前已有结果为基础，我认为最稳的定位是以下三类之一。

1. 工程系统型定位

推荐表述：

- 面向实时硬件约束的漂移自适应量子纠错解码框架
- 一种双回路的 CNN-FPGA 协同自适应解码系统

这类定位最适合当前已有的软件 HIL、参数 bank、延迟预算、artifact 导出链路。

2. 方法机制型定位

推荐表述：

- teacher-guided residual correction for runtime-consistent adaptive decoding
- 在经典 teacher 基础上进行轻量残差修正的学习型慢回路

这类定位最适合当前 `Hybrid Residual-B > UKF` 的正式结果。

3. 数据与部署一致性定位

推荐表述：

- runtime-consistent dataset and deployment-aligned adaptive quantum decoding
- 从离线回归转向运行时一致训练与部署闭环

这类定位最适合强调：

- 为什么早期 CNN 不稳定；
- 为什么需要 runtime-consistent 数据；
- 为什么纯绝对参数回归不如 residual 语义。

### 5.2 如果要压缩成一句话

目前最适合的单句定位是：

> 本项目不是在证明“CNN 全面替代经典解码器”，而是在证明“在实时硬件约束下，teacher-guided residual 学习模块能够稳定提升经典自适应解码，并且该增益可以在运行时一致的双回路工程链路中落地”。

---

## 6. 现在适合投哪些期刊或会议

下面的判断分为“当前最匹配”“补强后可冲”“硬件向补强后再考虑”三档。

### 6.1 当前最匹配的投稿目标

#### 6.1.1 IEEE Transactions on Quantum Engineering (TQE)

这是当前最匹配的期刊之一。

原因：

- 官方 scope 明确强调 quantum engineering 的工程应用；
- 覆盖 quantum computation、software、hardware、devices、metrology；
- 非常适合“系统 + 运行时 + 工程实现 + 量子应用”这种论文。

对你当前项目尤其匹配的点在于：

- 有软件与硬件协同叙事；
- 有部署链路；
- 有工程约束；
- 有量化、artifact、HIL；
- 不是纯理论推导。

结论：

- 如果你想投一个“工程味强、系统味强、对量子工程友好”的期刊，`TQE` 是非常自然的第一梯队选择。

#### 6.1.2 IEEE Quantum Week (QCE)

这是当前最稳的会议选择之一。

原因：

- 官方定位就是 quantum computing and engineering；
- 明确强调 multidisciplinary；
- 对系统、工具链、平台、实践型论文接受度高。

如果你希望：

- 更快形成公开结果；
- 先拿会议验证方向；
- 论文强调系统整合与工程实现；

那 `QCE` 很适合。

结论：

- 如果优先级是“尽快形成可信发表成果”，`QCE` 很值得优先考虑。

#### 6.1.3 EPJ Quantum Technology

这本期刊也很适合当前项目。

原因：

- 官方 scope 覆盖 quantum engineering、quantum control、quantum information、hybrid quantum systems；
- 对工程实现和技术路径比较友好；
- 相比极高选择性的顶刊，叙事空间更适合系统型工作。

结论：

- 如果你更希望把论文写成“量子工程应用论文”，`EPJ Quantum Technology` 是一个很稳妥的目标。

### 6.2 补强后可以考虑的更高目标

#### 6.2.1 Quantum Science and Technology (QST)

`QST` 是可以考虑的，但它比上面三者更挑。

原因：

- 官方说明其选择性较高；
- 要求论文不仅对局部问题有效，还要对更广泛的量子科技社区有持续影响；
- 更适合“结果更完整、机制更清晰、系统更扎实”的版本。

你当前项目若想更稳地投 `QST`，建议至少补齐：

- 完整 ablation
- 更强 baseline
- 更充分的机制分析
- 更完整的工程开销分析
- 最好再补一点真实硬件信息

结论：

- 可以作为中高目标；
- 但不建议把它作为当前唯一投稿目标。

#### 6.2.2 npj Quantum Information

如果论文写法更偏“量子信息/量子纠错/量子控制系统方法”，`npj Quantum Information` 也可以考虑。

但要注意：

- 它会更看重量子信息层面的新意与广泛兴趣；
- 如果论文写得太像“工程流水线搭建”，未必是最佳落点。

因此：

- 若后续把跨码、强 baseline、机制解释补强，`npj Quantum Information` 会变得更有吸引力；
- 以当前状态，它更像备选的中高目标，而不是最稳目标。

#### 6.2.3 ACM Transactions on Quantum Computing (TQC)

如果你把稿件写得更偏计算机系统/算法/架构/编译与运行时协同，而不是偏量子实验工程，`ACM TQC` 也是可以考虑的目标。

但它更适合：

- 计算机科学表达更强；
- 框架抽象和一般性更强；
- 对 fault-tolerant quantum computation、architecture、design automation 的表述更明确。

### 6.3 现在不建议作为主目标的高位期刊

#### 6.3.1 PRX Quantum

不是说完全不可能，而是以当前状态不建议把它设成主目标。

原因非常直接：

- 官方明确强调高选择性；
- 接收标准要求“exceptional advance / exceptional capabilities / exceptional insight”；
- 这类刊物通常要求论文在影响力、完整性、普适性上都更强。

当前项目虽然已经很有价值，但还没有强到可以稳冲这一档。

更现实的策略是：

- 先把系统论文写扎实；
- 若后续补上真板、跨码、强离散码 baseline，再重新评估。

### 6.4 如果以后把硬件做得更扎实，可考虑的硬件向 venue

如果后续真实 FPGA 综合、资源报告、真板卡时延都补齐，那么也可以考虑更偏 FPGA/EDA 的 venue。

#### 6.4.1 FCCM

`FCCM` 是 FPGA 与可重构计算的核心会议之一。

适合条件：

- 你不仅有算法；
- 还要有比较硬的 FPGA 架构、实现、工具流、资源/性能分析。

#### 6.4.2 ACM FPGA / ISFPGA

这个 venue 对 FPGA 技术本身的推进更敏感。

如果你能证明：

- 双回路调度在 FPGA/边缘部署上有明确系统贡献；
- 量化、buffer、commit、时延设计具有可重用性；

那么它会更合适。

#### 6.4.3 ICCAD / DATE

若论文进一步转向：

- runtime / architecture / CAD co-design
- compiler / toolflow / hardware-software co-optimization

那么 `ICCAD` 或 `DATE` 也可以考虑。

但以当前状态看，还更像未来扩展方向，而不是最优先目标。

### 6.5 当前建议排序

综合当前结果与工作风格，我建议的投稿优先顺序是：

1. `QCE`
2. `TQE`
3. `EPJ Quantum Technology`
4. `QST`
5. `npj Quantum Information / ACM TQC`
6. `FCCM / ACM FPGA / ICCAD / DATE`（前提是硬件实现显著补强）

---

## 7. 投稿前还差什么

### 7.1 最关键缺口

如果你现在就开始写论文，最需要补齐的不是再去反复调一个小超参，而是以下几类证据。

1. `features` 组正式 ablation

因为这组实验将直接回答：

- residual learner 的收益到底来自哪些输入通道；
- 是 teacher prediction 更关键，还是 teacher params 更关键；
- histogram delta 是否真有必要。

它对论文中的“为什么 hybrid 有效”非常关键。

2. 机制解释要再收口

目前已经有很强的工程推断，但写论文时最好整理成明确论证链：

- 纯 absolute regression 为什么不够好
- runtime-consistent 数据为什么更合理
- teacher-guided residual 为什么与部署语义更一致
- 为什么 `Hybrid > UKF` 不是 overflow 偶然差异

3. 更完整的成本分析

论文不是只看 `LER`。

还应整理：

- float / int8 / TFLite 精度差异
- slow loop latency 分布
- fast loop violation rate
- commit 成功率
- 参数更新频率
- 软件 HIL 的实际运行开销

4. 真实板卡结果不是硬性必须，但会加分

如果你投的是 `QCE / TQE / EPJQT`，没有真板卡结果未必致命。  
但如果有真实板卡结果，论文说服力会明显上一个台阶。

### 7.2 不一定必须补的部分

以下内容有加分，但不应阻塞主线：

1. 完整真实板卡联调
2. 所有码族全面迁移
3. 一开始就把所有超强离散码 baseline 全补齐

否则项目会很容易被拖入无止境扩张，迟迟写不出论文。

---

## 8. 还可以添加什么 baseline

### 8.1 当前 GKP 连续变量主线最该补的 baseline

对你当前这个“连续变量 + 漂移 + 在线估计 + 运行时参数更新”的问题，最自然、最公平的 baseline 主要有以下几类。

1. `Window Variance`

- 作为最轻量统计 teacher；
- 它的作用不是争第一，而是提供一个简单、可解释、可复现的统计基准。

2. `EKF`

- 已有；
- 作用是提供经典递推滤波基线。

3. `UKF`

- 已有；
- 目前已成为最强经典 baseline；
- 是论文中必须保留的关键对照。

4. `Constant Residual-Mu`

- 已有；
- 它很重要，因为它说明“只加一个常数补偿”能到什么程度；
- 这样才能证明 CNN 的收益不是 trivial residual。

5. `RLS Residual-B`

- 已有；
- 它很好，因为它和当前 `residual-b` 语义接近；
- 这是比纯 Kalman 更贴近你部署语义的一类线性自适应基线。

6. `Teacher-only + EMA`

- 如果还没有做成严格正式基线，建议补齐；
- 因为它可以回答：“到底是 teacher + 平滑已经够了，还是 residual learner 真有独立增益？”

7. `Teacher-guided residual PF`

- 可以做，但优先级不一定高于 `features` ablation；
- 因为 PF 容易更复杂，但不一定比当前 `UKF` 更公平、更稳定。

8. 小型时序学习基线

- 例如 `GRU / TCN`；
- 用来回答“是不是 CNN 结构本身不对，而不是学习思想不对”。

### 8.2 哪些 baseline 暂时不该作为主线阻塞项

像下面这些方法，不是不重要，而是它们更适合在“跨码或离散 syndrome 图”情境下加入：

1. `MWPM / Blossom`
2. `Union-Find`
3. `Belief Propagation / OSD`

原因很简单：

- 这些方法最自然的应用对象是 surface code、XZZX、repetition code 这类离散 syndrome 图码；
- 对当前单独 GKP 连续变量漂移跟踪问题，它们并不是最自然、最公平的第一批 baseline。

所以，当前阶段如果没有接入它们，并不削弱论文价值。  
但如果未来扩展到 concatenated GKP-surface，那么它们就会变成非常关键的强基线。

---

## 9. 如果以后补 MWPM / Blossom，这个项目还有超过它们的可能吗

这是一个非常重要的问题，而且必须诚实回答。

### 9.1 对当前单独 GKP 漂移跟踪问题

`MWPM / Blossom` 不是最自然对象，因此“超过它们”并不是当前项目最关键的胜负标准。

也就是说：

- 当前项目的价值，不建立在“打败 MWPM”上；
- 当前真正的对手，是在线漂移估计与参数自适应类基线；
- 因此 `UKF / EKF / RLS / statistics-based teacher` 才是更合理的第一层对照。

### 9.2 如果以后扩展到 concatenated GKP-surface

那情况会改变。

一旦你把框架扩展到：

- GKP 先做连续变量误差跟踪；
- 上层再进入 surface code / graph decoder；

那么 `MWPM / Blossom` 会成为非常关键的正式 baseline。

此时更合理的比较方式不是：

- “CNN-FPGA 直接替代 MWPM”

而是：

- “学习型慢回路是否能为上层图解码器提供更好的 bias / edge weights / error model”
- “在相同硬件时延预算下，adaptive front-end 能否提升整体逻辑错误率”

也就是说，未来真正强的叙事可能是：

- `Hybrid front-end + MWPM back-end`

而不是：

- `CNN` 单独击败 `MWPM`。

### 9.3 所以项目会不会因此失去创新性

不会。

当前项目的创新性不应理解成：

- “提出一个通吃所有码族、所有 baseline 的万能神经网络”

而应理解成：

- 提出一套面向实时硬件约束的学习型自适应解码框架；
- 在 teacher 已有稳定估计的前提下，用 residual learner 取得稳定增益；
- 把这个思想落到运行时一致的数据、双回路调度、参数 bank、量化 artifact 和软件 HIL 链路中。

这套价值即使未来与 `MWPM` 结合，也仍然成立。

---

## 10. 这个框架能不能迁移到其他量子纠错码

### 10.1 结论：能迁移，但迁移的是框架，不是现成模型

更准确地说：

- 当前方法不是“一个只能用于 GKP 的 CNN 模型”
- 它更像“一套面向实时量子纠错的自适应解码框架”

可迁移的是框架层：

1. 双回路架构

- 快回路做超低时延本地解码
- 慢回路做较低频率的统计估计与参数更新

2. 运行时机制

- param bank
- stage/commit
- latency budget
- HIL 记录与复验

3. 学习思想

- teacher 提供稳定一阶估计
- learner 学残差，而不是生硬替代全部物理建模

### 10.2 更适合迁移到哪些码

1. `surface code`
2. `XZZX code`
3. `repetition code`
4. `concatenated GKP-surface`
5. 其他 bosonic code，例如 `cat` 或 `binomial`

### 10.3 迁移时需要改什么

迁移时必须修改的部分通常包括：

1. 输入特征

- 对 GKP 是直方图窗口
- 对 surface / XZZX 可能是 syndrome patch、time stack、detector events、graph features

2. 输出目标

- 当前是 `(K, b)` 或 `delta_b`
- 对其他码可能变成：
  - edge weights
  - bias terms
  - measurement error rates
  - decoder hyperparameters

3. 快回路解码器

- 当前是线性解码
- 未来可能是：
  - MWPM
  - Union-Find
  - BP
  - OSD

因此最准确的长期定位是：

> 当前工作是 GKP 上验证的一套学习型自适应解码工程框架，未来可以迁移到更广泛的量子纠错体系，但需要更换特征、标签和快回路解码器。

---

## 11. 论文撰写时最推荐的主结论表述

如果现在开始准备论文，我建议主结论不要写得过满。

### 11.1 推荐表述

可以写成：

1. 在运行时一致的双回路自适应解码框架下，`Hybrid Residual-B` 在正式多场景 benchmark 中稳定优于经典自适应基线。

2. 当前最强经典 baseline 为 `UKF`，但 `Hybrid Residual-B` 仍保持稳定更低的 `LER`，说明 residual learner 的价值不等同于单纯使用更强的经典滤波器。

3. 项目价值不仅在于模型精度，更在于把数据构建、推理、参数映射、量化 artifact 和 HIL 验证组织成了可部署、可复验的工程闭环。

### 11.2 不推荐表述

不建议直接写成：

1. CNN 全面优于所有经典解码器
2. 本文提出了通用于所有量子纠错码的统一最优方法
3. 本文实现了完整真实 FPGA 系统并证明其工业可用

这些表述要么超出当前证据，要么容易被审稿人抓住边界条件。

---

## 12. 最终判断

### 12.1 关于“项目做到什么程度算完成”

如果以工程主线是否成立为标准，那么当前项目已经接近完成。

如果以“可稳妥投稿”为标准，那么当前已经进入可以认真写论文的阶段，但还建议把：

- `features` 正式 ablation
- 更完整机制分析
- 更完整工程代价分析

补齐后再定稿。

### 12.2 关于“现在适不适合写论文”

适合。

而且不是只能写成一篇弱论文；只要后续收口得当，完全可以写成一篇有明确系统贡献和方法贡献的量子工程论文。

### 12.3 关于“最适合投哪里”

当前最稳妥的选择是：

1. `QCE`
2. `TQE`
3. `EPJ Quantum Technology`

更高目标可考虑：

4. `QST`
5. `npj Quantum Information / ACM TQC`

硬件强补强后可再考虑：

6. `FCCM / ACM FPGA / ICCAD / DATE`

### 12.4 关于“这个项目还有没有创新性”

有，而且当前创新性已经比较清楚：

- 不是“深度学习替代一切”
- 而是“在实时硬件约束下，把 teacher-guided residual adaptive decoding 做成可运行、可验证、可部署的工程系统”

这件事本身就有研究价值。

### 12.5 我对当前项目的直接建议

接下来最值得做的，不是再大面积扩线，而是把已经形成优势的主线收紧：

1. 跑完 `features` 正式 ablation
2. 把 `Hybrid vs UKF` 机制分析彻底写清
3. 补充复杂度、时延、量化和部署成本分析
4. 再决定走 `QCE/TQE` 还是继续补强后冲更高目标

---

## 13. 参考的官方投稿定位来源

以下链接用于支持本文中对 venue 风格与适配度的判断：

1. IEEE TQE 官方 scope  
   https://tqe.ieee.org/

2. IEEE Quantum Week 官方主页  
   https://qce.quantum.ieee.org/

3. IEEE Quantum Week 官方 about 页面  
   https://qce.quantum.ieee.org/2024/about/about-ieee-quantum-week/

4. EPJ Quantum Technology aims and scope  
   https://epjquantumtechnology.springeropen.com/submission-guidelines/aims-and-scope

5. Quantum Science and Technology official about/scope  
   https://publishingsupport.iopscience.iop.org/journals/quantum-science-technology/about-quantum-science-technology/

6. npj Quantum Information aims and scope  
   https://www.nature.com/npjqi/aims

7. ACM Transactions on Quantum Computing launch/introduction  
   https://www.acm.org/media-center/2020/december/tqc-launches

8. FCCM 官方主页  
   https://www.fccm.org/

9. ACM/SIGDA FPGA 官方主页  
   https://www.isfpga.org/

10. ICCAD 官方页面  
    https://iccad.com/2026

11. PRX Quantum about/scope  
    https://journals.aps.org/prxquantum/about  
    https://journals.aps.org/prxquantum/scope
