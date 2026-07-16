结论：上一版重构方向基本正确，但这篇论文提示还需要一次关键升级——项目不应只讲“CNN 估计连续漂移 + FPGA 执行 MAP-LUT”，而应改成：

> 面向实验式 GKP syndrome history 的“两个计算域、三个时间尺度”控制架构：FPGA 完成确定性逐周期控制和稀有事件监测，主机完成较慢的噪声状态估计与参数优化；在 sBs/sharpen–trim 数字孪生、故障注入、强 baseline、低成本 FPGA 实测和 HIL 中形成闭合证据链。

这比单纯做一个“漂移自适应解码器”更贴近真实实验，也更适合只有约 300 元 FPGA 开发板的条件。

## 一、这篇论文带来的关键修正

### 1. FPGA 快回路与学习优化必须严格分开

实验中一个完整 sBs QEC cycle 是两个 \(4.924\,\mu s\) 的正交象限子周期；单个子周期包含测量、判决、反馈和虚拟旋转。[论文时序说明](</D:/Codes/Quantum/CNN_FPGA_GKP/docs/relative_papers/Real-time_quantum_error_correction_beyond_break-even.md:45>) FPGA 上的数字信号处理约 332 ns、结果分发 100 ns、分支反馈 200 ns、虚拟旋转更新 448 ns。[读出与反馈细节](</D:/Codes/Quantum/CNN_FPGA_GKP/docs/relative_papers/Real-time_quantum_error_correction_beyond_break-even.md:481>) [Table S3](</D:/Codes/Quantum/CNN_FPGA_GKP/docs/relative_papers/Real-time_quantum_error_correction_beyond_break-even.md:901>)

但 PPO 并不在逐周期关键路径：每个 epoch 评估 10 个候选、共 3000 个实验 shot，每个候选运行 160 cycles，整个 epoch 约 15.6 s，主要瓶颈是重新编译并装载 FPGA 指令和波形。[RL训练流程](</D:/Codes/Quantum/CNN_FPGA_GKP/docs/relative_papers/Real-time_quantum_error_correction_beyond_break-even.md:771>)

因此建议把架构明确成：

| 时间尺度 | 功能 | 部署位置 |
| --- | --- | --- |
| \(<4.924\,\mu s\) | syndrome 判决、MAP-LUT、Pauli-frame/相位更新、reset/fallback 动作 | FPGA |
| 数个至数百 cycles | e-run、leakage streak、相关尾、regime change 检测 | FPGA 为主，主机汇总 |
| 秒至周 | 参数估计、CNN/优化器、参数库更新、重新标定 | PC/CPU/GPU |

也就是说，仍然是“主机 + FPGA”两个计算域，但实验叙事上应是三个时间尺度。板上 CNN 继续降级为可选增强项，不进入第一篇论文主线。

### 2. 漂移模型必须从“连续参数漂移”升级为“混合状态非平稳性”

现有任务板已经覆盖 mean、variance、telegraph、burst 等漂移，[T1.3.1](</D:/Codes/Quantum/CNN_FPGA_GKP/docs/task_board.md:67>)，但这篇实验说明，真实非平稳性不一定只是 \(\mu_t,\sigma_t,\rho_t\) 平滑变化。

论文的 syndrome 图显示：

- 孤立 e 通常对应小误差；
- `eg/eg/...` 表示大误差沿 error hierarchy 逐步“trickle down”；
- 多周期 leakage streak 对应 transmon 逃逸到未被 reset 覆盖的高能级；
- 去除持续两周期以上的 leakage 事件后，原来延伸数百周期的非平稳相关尾消失；剩余典型大误差记忆长度约 \(3.9\pm0.1\) cycles。[Fig. S17及分析](</D:/Codes/Quantum/CNN_FPGA_GKP/docs/relative_papers/Real-time_quantum_error_correction_beyond_break-even.md:1003>)

所以建议将隐状态改为：

\[
\theta_t=
(\mu_t,\Sigma_t,p_{\rm loss},p_{\rm outlier},
z_t^{\rm regime},d_t^{\rm recovery},l_t^{\rm leakage}),
\]

其中：

- \(z_t^{\rm regime}\)：正常、标定偏移、burst、泄漏、恢复等离散状态；
- \(d_t^{\rm recovery}\)：离开 code space 的估计层级或恢复深度；
- \(l_t^{\rm leakage}\)：泄漏 run length 和风险概率。

这会让 CNN 的输出从单纯的连续噪声参数，升级为“连续参数 + 离散健康状态 + 不确定度”。

### 3. sBs 应从可选参考协议提升为主要实验数字孪生

现有 T2.2.2 主要要求选择 Steane-type 或 teleportation/Knill-type syndrome extraction，[当前定义](</D:/Codes/Quantum/CNN_FPGA_GKP/docs/task_board.md:88>)，与两篇主要超导 GKP 实验的实际控制叙事仍有距离。

建议形成两个协议模式：

- Protocol A：2020 实验的 sharpen–trim/measurement-feedback 模式，用于复现早期 FPGA 实时控制逻辑；
- Protocol B：本文 sBs 低秩耗散模式，作为主要实验式数字孪生。

sBs 模型至少要表示：

- \(X/Z\) 交替的 rank-2 channel；
- 一完整 cycle 的 \(K_{gg},K_{ge},K_{eg},K_{ee}\)；
- error subspace \(C_i\) 的逐级转移；
- g/e/leakage 三类观测；
- readout misclassification、ancilla reset 和 Pauli-frame tracking；
- 大误差需要多个 cycle 恢复的“trickle-down”过程。

尤其不能把单独的 g/e 简单等同于“无误差/有误差”。补充材料明确指出，严格解释应基于 `gg/ge/eg/ee` 成对结果，`gg` 也不严格证明状态已经位于 code space。[sBs Kraus与syndrome语义](</D:/Codes/Quantum/CNN_FPGA_GKP/docs/relative_papers/Real-time_quantum_error_correction_beyond_break-even.md:837>)

## 二、建议新增和重写的任务

不建议推翻已经完成的 Phase 0–1。先完成当前的 T1.3.3、T1.3.4，[当前推荐任务](</D:/Codes/Quantum/CNN_FPGA_GKP/docs/task_board.md:67>)，随后在进入原 Phase 2 前插入以下里程碑。

### M1.4：实验式 claim 与接口冻结

- `T1.4.1`：建立 claim ladder，严格区分 simulation、board measurement、HIL、quantum experiment。
- `T1.4.2`：冻结低成本开发板边界：只实现数字控制平面，不声称产生微波、采集真实量子读出或实现真实 GKP QEC。
- `T1.4.3`：冻结两个计算域、三个时间尺度的接口。
- `T1.4.4`：确定实验参考参数表，包括 \(4.924\,\mu s\) 子周期、读出分类、reset、leakage 和参数漂移范围。

### M2.0：实验对齐的 sBs 数字孪生

- `T2.0.1`：实现 sBs Kraus/error-space transition model。
- `T2.0.2`：实现 g/e/leakage 观测、误分类和 reset 模型。
- `T2.0.3`：实现与 Table S3 对齐的 cycle state machine。
- `T2.0.4`：复现注入位移误差后 e-run 长度随位移幅度变化的趋势。
- `T2.0.5`：复现 leakage 引起的长相关尾，以及去除 leakage 后近似平稳的趋势。
- `T2.0.6`：用 hidden-state truth 和 syndrome-only estimator 两种独立方法估计 code-space occupancy，作为交叉验证。

最后一项尤其有论文价值：原实验分别从 Wigner tomography 和 syndrome string 得到约 0.82 的 code-space occupancy。[双重估计结果](</D:/Codes/Quantum/CNN_FPGA_GKP/docs/relative_papers/Real-time_quantum_error_correction_beyond_break-even.md:181>) 本项目可以用“模拟器隐藏真值 vs 仅 syndrome 估计”复现这种证据三角验证，而不是只报告分类准确率。

### M3.3：实验式事件 baseline

保留现有 Bayesian、Kalman/EWMA 和 sliding-window baseline，[现有强baseline](</D:/Codes/Quantum/CNN_FPGA_GKP/docs/task_board.md:125>)，并新增：

- run-length FSM；
- leakage-aware HMM；
- change-point detector；
- syndrome-correlation monitor；
- static parameter-bank switching；
- oracle regime detector。

现有 `T3.2.4 postselection` 应改成“诊断性上界”，不能作为主要实时纠错结果。论文使用 post-selection 是为了证明 syndrome 与误差相关，但真正的 beyond-break-even 结果本身并不依赖 post-selection；同时严格 post-selection 会付出明显的 trajectory rejection 代价。[post-selection代价](</D:/Codes/Quantum/CNN_FPGA_GKP/docs/relative_papers/Real-time_quantum_error_correction_beyond_break-even.md:139>)

### M4.2：增强 FPGA 快路径

在现有 MAP-LUT、frame update、fallback 和定点化任务基础上，[当前FPGA设计](</D:/Codes/Quantum/CNN_FPGA_GKP/docs/task_board.md:147>)，增加：

- 2-bit g/e/leakage 输入协议；
- X/Z quadrature phase bit；
- e-run 和 leakage-run 饱和计数器；
- recovery-depth 状态机；
- phase/frame accumulator；
- normal/recovery/hold/reset-request/fallback 动作；
- 双参数库与原子切换；
- CRC/version/timestamp，防止更新过程中使用半套参数；
- deadline miss 和 overflow 计数器。

对约 300 元纯数字 FPGA 开发板，可以由 PC 经 UART/USB 重放定点 I/Q 或分类后的 syndrome；FPGA 完成阈值分类、状态机、MAP-LUT 和动作输出。论文必须明确这是“digitized-readout replay/HIL”，不是板卡直接连接微波 ADC。

## 三、重新定义证据门槛

建议把 Phase 5 从普通结果列表改为以下六道 evidence gate：

1. **协议可信度**：sBs/sharpen–trim 模型能复现论文中的方向性趋势，而不是强行拟合 \(G=2.27\)。
2. **因果可信度**：位移、bit-flip、phase-flip、readout error、leakage 必须分别注入，并产生预期 syndrome/recovery signature。
3. **算法可信度**：与 static MAP、oracle MAP、Bayesian、Kalman/EWMA、sliding window、HMM/run-length FSM 比较。
4. **消融可信度**：分别关闭 CNN、history、leakage state、parameter update、fallback；证明收益来自哪里。
5. **硬件可信度**：报告真实板卡上的最大/平均/尾延迟、jitter、deadline miss、LUT/FF/BRAM/DSP、Fmax、定点误差和 bit-accurate equivalence。
6. **长序列可信度**：至少 \(10^5\)–\(10^6\) cycles 的 HIL replay，覆盖稀有 leakage、计数器饱和、通信抖动和参数更新竞态。

论文指标建议分为三层：

- 物理/控制：code-space occupancy、recovery depth、recovery time、leakage burden、syndrome correlation length；
- 解码：average/windowed/tail LER、oracle-gap closure、calibration error、false fallback rate；
- 硬件：deadline miss、worst-case latency、throughput、资源、功耗或板级能耗代理。

`G` 或 break-even 只能写成 `simulation-derived coherence gain`。其计算应遵循论文用六个 Pauli eigenstates 和短时平均通道保真度斜率定义有效退极化率的方法，[平均通道保真度定义](</D:/Codes/Quantum/CNN_FPGA_GKP/docs/relative_papers/Real-time_quantum_error_correction_beyond_break-even.md:1199>)，不能把单一逻辑态的寿命随意称为 beyond break-even。

## 四、Phase 顺序还应继续调整

推荐最终顺序为：

1. 完成 T1.3.3–T1.3.4；
2. M1.4 claim/接口冻结；
3. M2.0 sBs 实验数字孪生；
4. 原 Phase 2 的多保真仿真；
5. 强 baseline 和事件 baseline；
6. 三时间尺度控制器；
7. 核心仿真与因果故障注入；
8. **低成本 FPGA 实现和 HIL**；
9. 主图冻结、论文撰写和开源归档；
10. 可选真实 GKP 数据或量子硬件接入。

也就是说，现有“Phase 6 先写论文、Phase 7 再做 FPGA”[当前顺序](</D:/Codes/Quantum/CNN_FPGA_GKP/docs/task_board.md:213>)应当交换。既然最终论文的主要物理证据来自仿真，而主要现实证据来自这块低成本开发板，就不能在拿到板级证据之前冻结主图和核心结论。

## 五、最终论文最可信的主张

建议将核心主张收敛为：

> We demonstrate an experiment-informed, syndrome-history-aware GKP control architecture in which a host estimator tracks continuous drift and discrete controller-health regimes, while a resource-constrained FPGA executes deterministic run-length-aware MAP-LUT control. Evidence is provided by calibrated sBs/sharpen–trim digital twins, strong adaptive baselines, causal fault injection, long-duration hardware-in-the-loop replay, and measured low-cost FPGA timing and resource utilization.

中文直译就是：

> 本文不是展示真实量子纠错，而是展示一个以真实 GKP 实验协议、时序和 syndrome 现象为依据的经典实时控制平面，并证明它能在廉价 FPGA 上确定性执行。

这种定位不会达到真实 beyond-break-even 量子实验论文的物理说服力，但可以形成一篇可信、证据充分的“量子控制/解码—FPGA 协同设计”论文。它的说服力主要来自：实验协议对齐、因果故障注入、强 baseline、独立交叉验证、真实板卡测量和长序列 HIL，而不是来自昂贵的微波谐振腔和 transmon。

本轮只完成了论文、补充材料和全部图片的精读与重构分析，没有修改仓库文件；`docs/relative_papers/` 的未跟踪状态也保持不变。