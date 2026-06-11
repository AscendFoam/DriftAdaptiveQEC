先给结论：**论文里的 2D CNN 噪声学习模型，角色上更像你项目早期的“绝对参数回归 CNN-FPGA”，而不是你现在的主线 `hybrid_residual_b`。**  
它们都属于“慢回路统计估计器”，不是直接做逐次纠错的快回路解码器；但你当前主线已经进一步演化成了 **teacher-guided residual controller**，这一点和论文有明显区别。

**对比**
| 维度 | 论文 2D CNN noise-learning | 你项目早期 `CNN-FPGA` | 你项目当前主线 `hybrid_residual_b` |
|---|---|---|---|
| 输入 | 两个连续 bulk syndrome rounds 的 2D 编码，带几何通道 | 单个 `32x32` syndrome 直方图窗口 | 多窗口直方图历史 + histogram delta + teacher prediction/params/deltas |
| 输出 | `25` 个电路级噪声参数 | `(σ, μ_q, μ_p, θ)` | `(b_q, b_p)` 残差补偿 |
| 目标 | 估计有效噪声，从而重建 PyMatching 的 edge / hyperedge 权重 | 回归物理噪声摘要参数，再映射到 `(K, b)` | 在 teacher 基础上直接修正运行时偏置 `b` |
| 损失 | 不是直接回归参数，而是通过可微公式拟合 `18` 类 edge + `43` 类 hyperedge 概率 | 直接监督回归，核心是参数 MSE | 直接监督回归 residual-b，强调 runtime-consistent 闭环标签 |
| 与下游解码器关系 | 松耦合但强语义对齐：输出专门服务 PyMatching | 中间耦合：输出先经 `ParamMapper` 再服务线性解码器 | 强耦合：就是 teacher 的闭环补偿器 |
| 推理方式 | 多 shot 批量统计后再做全局估计，模型内含 batch logit averaging | 单窗口预测，外部再做 EMA/慢回路更新 | 单窗口或短上下文预测，外部闭环平滑与限幅 |
| 设计重点 | 距离泛化、统计稳健、权重重建 | FPGA 可部署、参数可解释 | 闭环一致性、teacher 协同、formal HIL 有效 |

**相同点**
- 两者都不是“直接输出每轮纠错动作”的网络，而是**先估计有效噪声/状态，再去调下游解码器**。
- 两者都把二维统计结构当作主要输入。论文用 syndrome 的空间图样，你项目用 `q-p` 平面的 syndrome 直方图。
- 两者都很强调**模型要嵌入物理结构**，而不是做纯黑箱回归。论文把 edge/hyperedge 公式写进训练目标，你项目把 `ParamMapper`、协方差一致映射、teacher/runtime 语义写进系统。
- 两者都服务实时系统，只是工程约束不同。论文偏 GPU 高吞吐解码，你项目偏 `FPGA 快回路 + ARM/CNN 慢回路` 的双回路控制。

**关键区别**
- 论文模型的本质是**“校准 PyMatching 权重的统计估计器”**；你项目当前主线的本质是**“修 teacher 偏差的闭环补偿器”**。这两个角色不一样。
- 论文训练目标更“下游一致”。它不在乎 25 个参数本身误差最小，而在乎这些参数推出来的 edge/hyperedge 概率对解码最有用。你项目早期 CNN 更像“先把 `(σ, μ_q, μ_p, θ)` 拟合准”，这一步和最终 LER 之间还有一层 `ParamMapper` 与闭环动态。
- 论文模型是**全局统计估计**。它用 `GAP + batch averaging` 明确地表达“噪声参数是统计量，不是单样本局部标签”。你项目目前更多是**单窗口预测 + 外部 EMA**。
- 论文没有 teacher；你当前主线已经明确走向 **teacher-guided residual**。这意味着你的方法更像“learned correction layer”，而不是独立 noise learner。
- 论文的输出更偏**可迁移的噪声表征**，而你当前 `residual_b` 输出更偏**任务特化的控制量**。后者工程上更强，前者学术上更容易讲“可解释”和“可迁移”。

**对你项目最有价值的启发**
- 最重要的一点是：**训练目标要尽量贴近下游真正关心的量**。  
  论文不是直接最小化“噪声参数误差”，而是最小化“由这些参数生成的解码图权重误差”。  
  对你来说，这意味着可以继续从“参数回归”往“闭环有效量回归”推进。你已经从绝对参数回归走到 `residual_b`，这条路其实和论文的思想是一致的，而且方向是对的。

- 论文很支持你当前正在做的一个判断：**全局统计量不应该粗暴广播成整幅图像通道**。  
  它把全局噪声信息放在 `GAP -> MLP` 这条低维路径里。你项目在 P4 里已经发现 `teacher params` 的 broadcast 表示会带来耦合问题，而低维 gated scalar branch 更合理。这个判断是有外部论文支撑的。

- 你可以考虑引入一个**两阶段慢回路**：  
  第一阶段做“effective noise state estimation”，第二阶段做“teacher residual correction”。  
  也就是把当前主线再拆清楚一些：  
  `统计识别器` 负责估计慢漂移/有效噪声状态，`残差控制器` 负责把它转成对 `(K,b)` 或 `b` 的修正。这样论文味道会更强，结构也更清楚。

- 你可以借鉴论文的**模型内统计聚合**思路。  
  现在你更多靠窗口和 EMA 稳定输出；未来可以试：
  `多窗口输入 -> CNN 提特征 -> pooling/attention 聚合 -> 输出状态`
  而不是完全把稳定性交给后处理。这样可能比“单窗口预测 + EMA”更稳。

- 你可以把 loss 再往“闭环语义”推进一步。  
  比如不只监督 `b_q, b_p`，还监督：
  `映射后 K,b 的误差`、`一步校正后的残余能量`、`短 horizon 的 surrogate LER / overflow / saturation 风险`。  
  这会更像论文那种“通过结构化可微中间层训练最终有用的参数”。

- 论文也提醒了一点：**可识别性比模型容量更重要**。  
  你项目早期 `theta` 学不稳，本质上就是输入分布对该量不敏感。论文之所以能做 25 参数学习，是因为它的目标函数和 detector statistics 紧耦合。  
  对你来说，今后应优先问“这个量能不能从当前窗口统计里被识别”，再决定要不要让 CNN 去学它。

- 如果你未来想写论文，论文里的 framing 很值得借鉴：  
  不要把你的 CNN 讲成“替代解码器”，而要讲成  
  **“a hardware-aware learned calibration / residual correction module for real-time decoder adaptation under drift”**。  
  这和你现在的双回路、teacher-guided、FPGA/HIL 主线是很对得上的。

一句话概括对你的启发：**这篇论文最值得你借的不是具体结构，而是方法论：让 CNN 学“对下游解码真正有用的有效统计量或残差”，并且把统计聚合、物理先验和部署语义一起写进模型。**

参考我主要对照了这些文件：[CNN_FPGA_GKP_工程化实验方案.md](</d:/Codes/Quantum/DriftAdaptiveQEC/docs/legacy_context/reference_retired_2026-06-11/CNN_FPGA_GKP_工程化实验方案.md>)、[CNN_FPGA_GKP_阶段结论.md](</d:/Codes/Quantum/DriftAdaptiveQEC/docs/CNN_FPGA_GKP_阶段结论.md>)、[dataset_builder.py](</d:/Codes/Quantum/DriftAdaptiveQEC/cnn_fpga/data/dataset_builder.py>)、[runtime_dataset_builder.py](</d:/Codes/Quantum/DriftAdaptiveQEC/cnn_fpga/data/runtime_dataset_builder.py>)、[feature_builder.py](</d:/Codes/Quantum/DriftAdaptiveQEC/cnn_fpga/runtime/feature_builder.py>)、[tiny_cnn.py](</d:/Codes/Quantum/DriftAdaptiveQEC/cnn_fpga/model/tiny_cnn.py>)、[param_mapper.py](</d:/Codes/Quantum/DriftAdaptiveQEC/cnn_fpga/decoder/param_mapper.py>)。

如果你愿意，我下一步可以继续帮你做一件更具体的事：  
把“论文 2D CNN 噪声学习模型”改写成一个**适配你 GKP 项目的候选新分支设计草案**，包括输入张量、输出定义、loss、以及它如何接入你现有 `hybrid_residual_b` 主线。
