结论：这篇 PRL 是目前与本项目最接近的方法论文之一，应该成为核心对照文献。但它不应把项目带向“照搬 RNN 做 GKP 纠错”，更好的方向是：

> 用 Feedback-GRAPE/RNN 作为仿真中的 non-Markovian teacher，再把学习到的历史依赖策略蒸馏成可解释、定点化、可在廉价 FPGA 上逐周期运行的递推控制器，并进一步验证其在 drift、leakage、SPAM 和模型失配下是否仍然有效。

这会比单纯复现 PRL 更有新意，也与刚完成的 v2 任务板高度兼容。

## 一、必须先澄清：这篇 PRL 是数值论文，不是真实量子实验

论文模拟 cavity–transmon 联合密度矩阵，用 Feedback-GRAPE 通过整个多步随机测量过程反向传播。RNN 根据每个 half-cycle 的 g/e outcome 输出下一周期的 15 个 sBs 门参数。[方法与POMDP解释](</D:/Codes/Quantum/CNN_FPGA_GKP/docs/relative_papers/Non-Markovian_feedback_for_optimized_quantum_error_correction.md:49>) [Feedback-GRAPE梯度](</D:/Codes/Quantum/CNN_FPGA_GKP/docs/relative_papers/Non-Markovian_feedback_for_optimized_quantum_error_correction.md:465>)

但其“realistic simulation”仍使用：

- 完美瞬时门；
- 门后等效 idle dissipation；
- 数值 reset；
- 忽略真实脉冲和 Hamiltonian 演化；
- 忽略 leakage 和 SPAM；
- 只将 ancilla 观测简化为 g/e。[模型假设](</D:/Codes/Quantum/CNN_FPGA_GKP/docs/relative_papers/Non-Markovian_feedback_for_optimized_quantum_error_correction.md:363>) [sBs时序模型](</D:/Codes/Quantum/CNN_FPGA_GKP/docs/relative_papers/Non-Markovian_feedback_for_optimized_quantum_error_correction.md:439>)

这恰好留下了本项目可以补强的空间。尤其论文认为 leakage 对各 measurement-based protocol 影响相近，可以忽略；但前一篇 beyond-break-even 实验表明，多周期 leakage 会产生长相关尾和明显非平稳性，[实验leakage分析](</D:/Codes/Quantum/CNN_FPGA_GKP/docs/relative_papers/Real-time_quantum_error_correction_beyond_break-even.md:1029>)。因此我们的任务板不应撤掉 leakage/regime 建模，反而应把它作为超越 PRL 数值模型的重要贡献。

## 二、它和本项目“相似但不同”的地方

| 维度 | PRL NMF | 当前项目 | 建议融合方式 |
| --- | --- | --- | --- |
| 历史信息 | RNN hidden state 保存全部 g/e 历史 | CNN/TCN 估计较慢的 drift/regime | 增加逐周期短记忆与窗口级慢漂移两类 memory |
| 输出 | 每 half-cycle 直接输出 15 个 sBs 门参数 | 慢回路输出状态，FPGA 执行 MAP-LUT | RNN teacher 输出门参数，蒸馏后由 FPGA 递推更新 |
| 训练 | 可微物理模型上的 Feedback-GRAPE | 计划中的 PPO/BO/CMA-ES teacher | 将 Feedback-GRAPE 加入 teacher 候选，并验证梯度 |
| 硬件 | 仅讨论可实验部署，没有 FPGA 实测 | 约 300 元 FPGA + HIL | 把蒸馏后的控制律作为实际板级贡献 |
| 非平稳性 | 固定噪声、固定 bias、跨噪声测试 | drift、burst、leakage、regime | 验证 NMF 是否在真实非平稳性下仍有优势 |

所以，当前任务板的 sBs 数字孪生、hidden regime、offline teacher、FPGA FSM 和 HIL 方向是正确的，[现有M2.0](</D:/Codes/Quantum/CNN_FPGA_GKP/docs/task_board.md:113>) [现有三时间尺度控制](</D:/Codes/Quantum/CNN_FPGA_GKP/docs/task_board.md:184>)；现在需要做的是增加一条明确的 teacher–distillation 路线。

## 三、建议升级后的中心科学问题

建议把中心问题改为：

> Can a model-aware recurrent teacher discover useful non-Markovian sBs feedback, and can this strategy be compressed into a fixed-point controller that preserves most of the simulated lifetime gain under drift, leakage and model mismatch while meeting a low-cost FPGA deadline?

中文即：

> 可微物理模型训练出的 RNN 是否真正学到了可泛化的 GKP 历史反馈规律？该规律能否压缩为廉价 FPGA 可执行的递推控制器，并在更真实的漂移、泄漏和模型失配下保留大部分仿真增益？

这比“CNN 估计 drift”更尖锐，也比“复现 PRL 的 RNN”更有原创空间。

## 四、建议对任务板做 v2.1 定向增补

### 1. 在 M2.3 后增加 Feedback-GRAPE 可行性门

建议新增：

- `T2.3.4`：构造短时域可微 sBs trajectory simulator。
- `T2.3.5`：验证 Feedback-GRAPE 的两项梯度：
  \[
  \partial_\theta \mathcal R,\qquad
  \mathcal R\,\partial_\theta\log P_\theta(\mathbf m)
  \]
  并与 finite difference 对齐。
- `T2.3.6`：检查 cutoff、trajectory batch、10-cycle horizon 的显存和运行时间。
- `T2.3.7`：复现 standard、MF、NMF 的方向性 lifetime ranking。

这里应设置失败分支：若无法在合理 cutoff 下复现趋势，就把 PRL 的解析递推策略作为 baseline，不能声称“RNN teacher distillation”。

### 2. 扩充 Phase 3 的 memory-specific baseline

现有 Bayesian、Kalman、run-length 和 HMM baseline 很好，[当前M3.2](</D:/Codes/Quantum/CNN_FPGA_GKP/docs/task_board.md:169>)，但还缺少证明“memory 本身有用”的直接对照：

- `T3.2.7`：memoryless FNN / latest-outcome policy；
- `T3.2.8`：autonomous sBs，按真实物理时间而非 cycle 数归一化；
- `T3.2.9`：有限时域 trajectory lookup control oracle；
- `T3.2.10`：PRL 式指数递推手工策略；
- `T3.2.11`：history shuffle、history truncation、hidden-state reset 消融。

必须区分两个 oracle：

- `decoder oracle`：知道真实 noise state 的 MAP；
- `control oracle`：短时域内为每条 measurement trajectory 独立优化控制参数的 lookup table。

PRL 本身把有限时域 lookup table 作为 control-policy 上界，但它随时间指数增长。[lookup oracle与泛化](</D:/Codes/Quantum/CNN_FPGA_GKP/docs/relative_papers/Non-Markovian_feedback_for_optimized_quantum_error_correction.md:565>)

### 3. 修改 T4.1.1：模型结构不再预设 CNN 优先

当前任务板写的是 causal CNN/TCN 优先、GRU 备选，[T4.1.1](</D:/Codes/Quantum/CNN_FPGA_GKP/docs/task_board.md:184>)。读完这篇论文后，应该改为证据选择：

- causal CNN/TCN；
- small GRU；
- HMM/Kalman；
- PRL-inspired exponential recurrence；
- run-length FSM。

要求在相同输入、参数量、history budget 和 latency budget 下比较。最终模型不一定是 CNN。

### 4. 新增 M4.4：RNN teacher 到 FPGA student 的策略蒸馏

这是最值得新增的里程碑：

- `T4.4.1`：训练 residual RNN teacher。必须从 nominal sBs 参数附近开始，因为论文明确说明从零训练失败。[初始化和动作空间](</D:/Codes/Quantum/CNN_FPGA_GKP/docs/relative_papers/Non-Markovian_feedback_for_optimized_quantum_error_correction.md:501>)
- `T4.4.2`：提取 RNN hidden state、门参数和 g/e/leakage history 的关系。
- `T4.4.3`：拟合指数递推：
  \[
  \pi_{t+1}=a_m\pi_t+(1-a_m)\pi_m^\infty ,
  \quad m\in\{g,e,\mathrm{leak}\}.
  \]
- `T4.4.4`：比较 teacher、蒸馏 student、run-length FSM 和 static sBs。
- `T4.4.5`：要求 student 保留预设比例的 teacher gain，同时显著降低参数量、MAC 和 worst-case latency。

论文的 Fig. 4 和 Supplement 明确发现门参数在连续 g/e 段中呈指数饱和，并提出可转成解析表达式。[指数策略分析](</D:/Codes/Quantum/CNN_FPGA_GKP/docs/relative_papers/Non-Markovian_feedback_for_optimized_quantum_error_correction.md:729>) 这为廉价 FPGA 提供了非常自然的切入点。

### 5. 不建议直接把完整 PRL RNN 放到廉价 FPGA

其结构是 10 个 GRU 单元，加两层 256 neurons，再输出 15 个参数。[Table S2](</D:/Codes/Quantum/CNN_FPGA_GKP/docs/relative_papers/Non-Markovian_feedback_for_optimized_quantum_error_correction.md:501>)

粗略估计约有 7.3 万个参数：

- 16-bit 权重约 1.17 Mbit；
- 每 half-cycle 还需要数万次 MAC；
- PRL 的 half-cycle 参考 deadline 约 \(5\,\mu s\)。

这对约 300 元小型 FPGA 的片上 BRAM、DSP 和时序都很紧张。因此 Phase 5.5 应新增：

- 完整 GRU、量化 GRU、指数递推 student 三者的资源估计；
- 只有 synthesis 证明完整 GRU 可行时才考虑上板；
- 默认实际硬件主线应是蒸馏递推器，而不是完整 RNN。

现有 FPGA fast path 和 hardware freeze 已提供合适承载位置，[M5.5](</D:/Codes/Quantum/CNN_FPGA_GKP/docs/task_board.md:258>) [Phase 6](</D:/Codes/Quantum/CNN_FPGA_GKP/docs/task_board.md:268>)。

## 五、证据门还应增加五项

### 1. 多 agent 选择偏差

论文训练 20 个 RNN，只报告最好 agent。[agent post-selection](</D:/Codes/Quantum/CNN_FPGA_GKP/docs/relative_papers/Non-Markovian_feedback_for_optimized_quantum_error_correction.md:667>)

本项目必须报告：

- 所有 seeds 的分布；
- median、IQR、worst quartile；
- 选择 agent 的 validation set；
- 独立 test set；
- 不能在 test lifetime 上选“最好模型”。

### 2. 训练时域到部署时域的稳定性

论文训练 10 cycles，却测试到 1000 cycles。[训练与评估时域](</D:/Codes/Quantum/CNN_FPGA_GKP/docs/relative_papers/Non-Markovian_feedback_for_optimized_quantum_error_correction.md:95>)

需要新增：

- 训练 horizon sweep；
- hidden-state boundedness；
- 1000-cycle extrapolation；
- \(10^5\)–\(10^6\) cycle student/HIL 稳定性；
- hidden-state reset/bit-flip/upset 故障测试。

### 3. 实验可行性约束 \(p(g)\)

论文发现某些策略峰值寿命更高，但 \(p(g)\) 较低、泛化更差，也更不利于实验；最终主 agent 保持 \(p(g)\approx0.9\)。[实验可行性约束](</D:/Codes/Quantum/CNN_FPGA_GKP/docs/relative_papers/Non-Markovian_feedback_for_optimized_quantum_error_correction.md:103>)

所以 loss/selection 应加入：

- \(p(g)\) 或 ancilla-excitation burden；
- e/leakage occupancy；
- reset burden；
- parameter slew；
- action safety envelope。

不能只最大化 lifetime。

### 4. 公平的物理时间比较

autonomous sBs 周期约为 measurement-feedback 周期的 0.7。[autonomous时序](</D:/Codes/Quantum/CNN_FPGA_GKP/docs/relative_papers/Non-Markovian_feedback_for_optimized_quantum_error_correction.md:595>)

因此所有结果必须同时报告：

- per cycle；
- per microsecond；
- measurement/reset 次数；
- active gate/control cost。

否则容易把周期定义差异误写成算法优势。

### 5. 更真实的模型失配

PRL 只测试了一组固定 gate bias，并补充了 cavity dephasing而不重训。[bias和dephasing测试](</D:/Codes/Quantum/CNN_FPGA_GKP/docs/relative_papers/Non-Markovian_feedback_for_optimized_quantum_error_correction.md:785>)

本项目应升级为：

- 随机 gate-bias family，而不是单一 bias vector；
- readout confusion；
- leakage/reset failure；
- continuous drift；
- unseen dynamics/timing；
- train-on-simplified、test-on-higher-fidelity；
- teacher 与 student 分别做 OOD 测试。

## 六、可以借鉴的 PRL 写作结构

这篇文章主文只有四张核心图，论证非常紧：

1. Fig. 1：一句话架构——measurement history → memory agent → control。
2. Fig. 2：把 RNN 放进真实 sBs cycle。
3. Fig. 3：一次性给出 lifetime、强 baseline/noise levels、故障注入。
4. Fig. 4：解释 RNN 到底学到了什么。

本项目也可以把当前六张主图压缩成：

1. 实验边界、三时间尺度和 POMDP/belief-state 架构；
2. sBs digital twin + Feedback-GRAPE teacher + distilled FPGA student；
3. standard/MF/NMF/control-oracle/student 在 lifetime、drift、leakage 和 fault injection 下的决定性结果；
4. 指数递推解释、teacher–student gain retention、真实 FPGA latency/resource/HIL。

其余 cutoff、训练曲线、六 Pauli eigenstates、所有 seeds、OOD、定点、失败模式全部放 Supplement。这个写法比把每个指标各做一张主图更接近 PRL 的“一个问题、一条主线、四张决定性图”。

## 七、建议的最终论文定位

建议主线升级为：

> **Hardware-realizable distilled non-Markovian feedback for experiment-informed GKP control under drift and leakage**

但正文中的主张必须分层：

- 仿真证明：RNN teacher 和 distilled student 的 logical lifetime / channel fidelity 改善；
- HIL 证明：FPGA 输出与 fixed-point reference 一致；
- 板卡证明：latency、jitter、Fmax 和资源真实测得；
- 不得写成：低成本 FPGA 已在真实 cavity/transmon 上延长逻辑寿命。

如果这条路线成功，它会比当前单纯的“CNN 慢速漂移估计 + MAP-LUT”更有论文辨识度；如果 Feedback-GRAPE 复现或蒸馏失败，则保留现有 v2 主线，把 PRL 的 NMF、MF 和指数递推仅作为强 baseline。这就是最安全的双分支设计。
