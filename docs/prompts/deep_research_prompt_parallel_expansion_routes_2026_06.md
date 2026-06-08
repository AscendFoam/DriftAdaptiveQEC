# Deep Research Prompt: Parallel Expansion Routes For Teacher-Anchored Fast/Slow-Loop QEC

请作为一名同时理解量子纠错、机器学习解码器、硬件感知推理、FPGA 实时系统和研究路线规划的深度研究助手，进行一次面向“后续并行扩展实验路线”的全网调研。

本次调研的目的不是直接写论文，也不是泛泛列举点子，而是要回答一个非常具体的项目管理问题：

> 在当前主线实验单次运行周期很长的情况下，是否可以保持“快慢双回路 + FPGA 快回路”为架构不变前提，同时并行开出若干彼此相对独立、可边跑主线边推进的扩展路线？

如果可以，请调研并筛选出值得后续试验的方向，并给出优先级、实现难度、与现有主线的兼容性以及潜在论文价值。

---

## 1. 项目背景

我正在推进一个研究型工程项目：`DriftAdaptiveQEC`。

当前主线可以概括为：

1. **快回路**由 FPGA / FPGA-like runtime 执行低延迟纠错：
   - 输入：syndrome `(s_q, s_p)`
   - 输出：低延迟 correction
   - 形式近似为：`Delta = K @ s + b`
   - 目标：低延迟、确定性、固定点友好、可参数 bank 切换

2. **慢回路**周期性利用统计特征进行参数更新：
   - 当前典型输入：syndrome histogram，例如 `32 x 32`
   - 当前主线偏向：teacher anchored / teacher guided residual correction
   - CNN 不是完全替代 classical estimator，而是学习 residual / calibration，尤其偏向对 `b` 的修正

3. **当前问题**
   - 主线实验单次 task 的运行时间很长，经常需要 2 到 4 天
   - 我不希望所有后续探索都堵在这条主线上排队等待
   - 我希望在主线 benchmark 运行期间，并行地研究并准备新的扩展方向

---

## 2. 当前已确认的真实边界

这些边界非常重要，请严格遵守，不要误判当前项目状态：

1. 当前项目处于 `Phase 2: Controlled Development`
2. 当前已有真实代码与实验路径，不是空想项目
3. 当前已验证的是 **mock-backed software HIL** 边界，不是 real-board FPGA 验证
4. 不要假设 `.tflite` true runtime 已恢复完成
5. 不要假设 real-board HIL 已完成
6. 历史正式 frozen table 的权威锚点来自既有主线 formal revalidation，不允许被随意重写
7. `statcalib` 当前仍只是单独标注的 extension lane，不应被自动升级为成熟 comparator

也就是说，请把这个项目理解为：

> 一个已经有真实代码、真实 benchmark、真实 software-HIL 证据，但仍处在受控扩展阶段的研究工程项目。

---

## 3. 这次调研要回答的核心问题

请重点回答：

### A. 并行扩展路线是否总体可行？

这里的“可行”不是问“理论上能不能想出别的方法”，而是问：

1. 是否能在保持主线不被破坏的情况下，拆出若干相对独立的扩展路线
2. 这些路线是否能在项目主线长时间 benchmark 运行期间并行推进
3. 它们是否能够共享一部分架构前提、评测协议或硬件约束，而不至于完全碎片化

### B. 哪些维度适合作为“并行路线”？

我当前直觉是：

1. **快慢双回路** 继续保留
2. **FPGA 快回路** 作为核心细节和主约束保留
3. 允许变动：
   - 慢回路模型
   - teacher 形式
   - 特征表示
   - 具体纠错码 / 外码结构
   - FPGA 快回路中的参数化逻辑

请你判断这个拆法是否合理，并给出更好的拆分方式。

### C. 哪些方向值得真正进入后续实验池？

我希望你不是简单列出一堆“可以试试”的方向，而是要筛掉不值得做、风险过高或和当前项目不兼容的路线。

---

## 4. 请重点调研的扩展路线维度

请围绕下面几类路线做全网调研，并判断各自是否适合作为后续并行扩展实验方向。

### Route Family 1: 慢回路模型替代

请调研：

1. 当前 teacher anchored CNN 思路是否已经明显过时
2. 是否有更新、更强、又仍适合硬件感知闭环的慢回路方案
3. 特别比较：
   - CNN
   - TCN / temporal convolution
   - RNN / GRU / LSTM
   - state-space models，例如 S4 / Mamba 类
   - lightweight transformer / recurrent-transformer
   - classical teacher + tiny residual head
   - calibration-conditioned decoder / FiLM-like conditioning

请重点分析：

1. 哪些模型更适合时间序列 drift
2. 哪些模型在低延迟、低参数量、易量化、易部署方面更现实
3. 哪些模型只是“学术上新”，但实际上会破坏当前工程约束

### Route Family 2: teacher 机制替代

请调研：

1. 除了 UKF / EKF / window statistics / current teacher anchors 之外，是否存在更新、更合理的 teacher 方案
2. 是否存在：
   - adaptive prior / decoder prior optimization
   - classical estimator + learned calibration
   - uncertainty-aware teacher
   - confidence-gated teacher / fallback teacher
   - model-based Bayesian teacher + neural residual
   - syndrome-only online noise estimation teacher

请分析：

1. 哪些 teacher 更适合作为“稳定底座”
2. 哪些 teacher 更适合与神经 residual 结合
3. 哪些 teacher 会显著增加实验复杂度或运行时间

### Route Family 3: 特征表示替代

请调研当前 `32 x 32 histogram` 类表示是否已经足够，或者是否存在更优路线：

1. raw time-series sequence
2. histogram + temporal stacking
3. low-order statistical moments
4. graph / spatiotemporal representation
5. event-sequence / detector-history style representation
6. compressed features for FPGA-aware calibration

请重点回答：

1. 哪些表示最适合 drift 估计
2. 哪些表示最适合与 teacher anchored residual 路线结合
3. 哪些表示最可能在实验成本和信息量之间取得更好平衡

### Route Family 4: 纠错码 / 任务定义扩展

请调研以下问题：

1. 维持物理层 GKP fast-path 是否最合理
2. 是否值得后续拓展到：
   - surface-GKP
   - concatenated outer code
   - QLDPC-GKP
   - bosonic soft-information outer decoder setting

请重点判断：

1. 这些方向哪些适合“后续并行分支”
2. 哪些方向虽然学术价值高，但会把问题定义彻底改写，不适合当前阶段
3. 哪些方向可以只作为 paper positioning / future work，而不应变成短期实验任务

### Route Family 5: FPGA 快回路逻辑替代

请调研在保持“快回路硬件主约束”的前提下，还有哪些值得探索的快回路形式：

1. affine `K @ s + b`
2. piecewise affine / regime switching
3. gain scheduling
4. LUT-assisted correction
5. low-bit quantized learned micro-head
6. safety-bounded neural correction on top of classical fast path
7. staged parameter bank / atomic commit / rollback-aware control

请重点判断：

1. 哪些仍然是 FPGA-friendly 的
2. 哪些会破坏确定性和可验证性
3. 哪些适合作为“快回路结构扩展”而不是纯慢回路扩展

---

## 5. 对每条路线要做的评价维度

请不要只做文献总结。请对每条候选路线给出结构化评价，至少包含：

1. **方向名称**
2. **核心思想**
3. **代表性论文 / arXiv / GitHub / 技术报告**
4. **与当前项目的相容性**
5. **是否保留快慢双回路基本思想**
6. **是否保留 FPGA 快回路核心约束**
7. **是否可以与当前主线并行推进**
8. **实现成本**
   - docs-only / design-only
   - 小规模 toy simulation
   - bounded benchmark
   - 大规模 rerun
   - 需要重新训练
   - 需要改 FPGA contract
9. **潜在实验周期**
   - 很短
   - 中等
   - 很长
10. **潜在论文价值**
11. **与当前项目 narrative 的一致性**
12. **主要风险**
13. **是否建议进入后续任务池**
   - yes / maybe / no

---

## 6. 我最关心的额外判断

请你务必明确回答下面这些问题：

1. **CNN 换成 RNN/GRU/LSTM 是否值得？**
   - 是短期高性价比方向，还是低优先级方向？
   - 它是更好地匹配时间序列，还是会在训练稳定性/并行性/部署性上吃亏？

2. **比 RNN 更现代的慢回路模型是否更值得优先调研？**
   - 比如 TCN、S4、Mamba、轻量 transformer、recurrent-transformer

3. **teacher anchored 路线是否已显老旧？**
   - 是“还能用但不新”
   - 还是“依然有价值，因为它与硬件约束和安全边界更匹配”

4. **哪些方向适合作为主线运行期间的并行路线？**
   - 我希望这些路线尽量不和当前耗时长跑互相阻塞

5. **哪些方向不适合现在开？**
   - 即便它们学术上有趣，也请明确指出为什么现在不值得

---

## 7. 结果输出格式要求

请输出一份结构化调研报告，建议至少包含：

1. Executive summary
2. Search scope and search keywords
3. Candidate route families
4. Comparison table of candidate routes
5. Most relevant recent works from the last 1-3 years
6. CNN vs RNN vs TCN vs SSM vs transformer judgment
7. Teacher alternatives judgment
8. Feature representation alternatives judgment
9. Code-family expansion judgment
10. Fast-loop FPGA logic expansion judgment
11. Which routes are truly parallelizable with the current mainline
12. Which routes should remain future-work only
13. Ranked shortlist:
    - Top 3 routes worth trying next
    - Top 3 routes worth researching but not implementing yet
14. Suggested next-task roadmap

---

## 8. 输出时的强约束

请严格遵守：

1. 不要误写当前项目已完成 real-board FPGA validation
2. 不要误写当前项目已完成 true `.tflite` runtime recovery
3. 不要把当前 software-HIL 证据升级成硬件完成态
4. 不要把当前某个 extension lane 写成已被正式确立的成熟主线
5. 不要只给“想法列表”，必须给出优先级和可行性判断
6. 不要只给旧文献，优先覆盖最近 1-3 年的新工作，同时保留必要的经典文献
7. 如果发现某条扩展路线会和当前项目叙事严重冲突，请明确写出冲突原因
8. 如果你认为“可以并行开路线，但必须满足若干治理前提”，请把这些前提明确列成 checklist

---

## 9. 额外要求：请给出面向项目管理的最终结论

最后请给出一句非常明确的结论，必须是下面三种之一：

1. **Recommended now**  
   适合现在进入后续扩展路线池

2. **Research only for now**  
   适合先做调研/设计，不适合立刻进实验池

3. **Not recommended in current phase**  
   当前阶段不建议开

并分别给出原因。
