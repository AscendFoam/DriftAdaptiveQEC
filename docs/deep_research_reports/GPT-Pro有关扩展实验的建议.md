# 1. Executive summary

**总体判断：可以并行开扩展路线，但必须把它们定义为 sidecar extension lanes，而不是重写主线。** 在你给定的边界下，`DriftAdaptiveQEC` 当前应被视为：已有真实代码、真实 benchmark、mock-backed software-HIL 证据，但尚未完成 real-board FPGA validation，也不能假设 true `.tflite` runtime 已恢复。基于这个状态，最合理的扩展策略不是“换掉主线”，而是在保持 **快慢双回路 + FPGA 快回路 contract** 不变的前提下，开出若干低耦合、可用缓存数据或小规模仿真推进的路线。

外部研究趋势支持这个拆法：实时 QEC 解码的关键约束仍然是**确定性低延迟、与 QEC cycle 匹配、闭环控制可集成**，而不是单纯追求神经网络规模或 decoder accuracy。实时解码综述强调 decoder 必须足够快以跟上微秒级 QEC cycle，并且必须成为控制系统的一部分；后续系统需求研究也把闭环延迟放在数十微秒量级的系统约束内。([arXiv][1]) 近年的 real-time FPGA neural decoder 工作已经展示了 FPGA 上 NN decoder 可以进入亚微秒级闭环，但那是外部成果，不能映射为你当前项目已完成 real-board HIL。([arXiv][2])

**最值得现在进入实验池的三条路线是：**

1. **Histogram temporal stacking + tiny TCN / temporal-CNN residual `b` head**
   保留当前 histogram 路线，但给慢回路加入时间维度；比直接换 RNN 更高性价比。

2. **Syndrome-only adaptive teacher / sliding-window noise-estimation teacher + confidence-gated fallback**
   不推翻 teacher anchored 叙事，而是把 teacher 从固定 anchor 升级为带不确定性和 fallback 的稳定底座。

3. **FPGA fast-loop gain scheduling / piecewise-affine / LUT-assisted parameter bank + atomic commit / rollback**
   不把神经网络塞进 fast loop，而是扩展 `K @ s + b` 的参数化形态，保持确定性、fixed-point 友好和 bank switching 叙事。

**最不建议现在做的是：** 大型 recurrent transformer / full transformer 替代慢回路、raw time-series 全量输入、大规模 surface-GKP / QLDPC-GKP 重定义主任务、diffusion/autoregressive decoder、以及把 real-board HIL 或 `.tflite` runtime recovery 当作普通 extension lane。它们要么会改写问题定义，要么会破坏当前 Phase 2 的工程边界。

---

# 2. Search scope and search keywords

本次调研覆盖五个方向：实时 QEC 解码、机器学习 decoder、漂移噪声估计、GKP/outer-code soft information、以及时序模型与 FPGA-aware inference。

使用的核心检索关键词包括：

| 主题               | 关键词                                                                                                                                                                                             |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 实时 QEC / FPGA    | `real-time quantum error correction decoder FPGA`, `controller decoder system requirements surface code`, `FPGA neural network decoder quantum error correction`                                |
| 神经 QEC decoder   | `recurrent transformer surface code decoder`, `neural network decoder surface code 2024 2025`, `graph neural network detector graph quantum decoder`                                            |
| 漂移与 teacher      | `adaptive estimation drifting noise quantum error correction`, `syndrome statistics noise estimation QEC`, `noise-aware decoding ACES`, `decoder switching confidence quantum error correction` |
| GKP / outer code | `GKP surface code concatenation`, `QLDPC GKP soft information decoder`, `bosonic soft information outer decoder`                                                                                |
| 时序模型             | `TCN vs LSTM sequence modeling`, `S4 state space model long sequences`, `Mamba time series forecasting selective state space model`                                                             |
| 量化与部署            | `low-bit quantized neural network FPGA inference`, `fixed point neural decoder FPGA`                                                                                                            |

---

# 3. 并行扩展路线是否总体可行？

**结论：可行，但只在“受控并行”的定义下可行。**

可行的原因有三点。

第一，你的主线已经有稳定的架构分层：fast loop 接收 syndrome `(s_q, s_p)` 并输出低延迟 correction；slow loop 周期性更新参数。只要 extension lane 不改变 fast-loop ABI，不修改 frozen baseline table，就可以作为 sidecar 运行。

第二，多数候选路线可以共享同一套评测协议：syndrome 数据、histogram 窗口、teacher 输出、`b` residual target、fixed-point envelope、bank-switch safety check。这意味着扩展路线不会完全碎片化。

第三，外部研究趋势也支持“decoder / calibration / controller”分层。实时 QEC 研究反复强调 decoder latency、硬件集成和 closed-loop control，而不是只看离线 accuracy。([arXiv][1]) Google 2025 surface-code below-threshold 实验也把 real-time decoder latency 作为系统性指标报告；这说明你的 FPGA fast loop 约束是合理的核心叙事。([arXiv][3])

**但必须满足以下治理 checklist：**

| Checklist            | 要求                                                                             |
| -------------------- | ------------------------------------------------------------------------------ |
| Frozen anchor        | 历史 formal revalidation table 只作为权威锚点，不被 extension lane 重写                      |
| Sidecar artifacts    | 新路线只产出 sidecar metrics、sidecar configs、sidecar candidates                      |
| Fast-loop ABI freeze | 默认不改变 fast-loop 输入输出 contract；若必须改，单独标为 contract-change lane                   |
| Mock-backed boundary | 所有结果明确标注为 software-HIL / mock-backed，不能升级为 real-board FPGA evidence            |
| `.tflite` boundary   | 不假设 true `.tflite` runtime 已恢复；若涉及部署，只能写成 future/gated integration             |
| Small-first policy   | 先 docs-only / toy simulation / bounded benchmark，再决定是否大规模 rerun                |
| Promotion rule       | 只有在 frozen-anchor A/B comparison、deterministic replay、rollback test 后才允许进入主线候选 |
| statcalib boundary   | `statcalib` 仍保持 extension lane，不自动升级为成熟 comparator                             |

---

# 4. 更好的拆分方式

你提出的拆法是合理的：保留快慢双回路和 FPGA fast loop，允许慢回路模型、teacher、特征、码族、fast-loop 参数逻辑变化。

我建议进一步拆成六个相互正交的 axis：

| Axis                      | 保留不变                                               | 可变内容                                                                       | 为什么这样拆               |
| ------------------------- | -------------------------------------------------- | -------------------------------------------------------------------------- | -------------------- |
| A. Fast-loop contract     | syndrome in, correction out, deterministic latency | `K,b` 参数 bank、piecewise bank、LUT assist                                    | 防止 extension 侵入实时路径  |
| B. Slow-loop estimator    | 慢周期更新，不直接抢 fast-loop latency budget                | CNN、TCN、SSM、teacher-conditioned residual head                              | 主要并行实验区              |
| C. Teacher / prior        | teacher anchored 安全底座                              | sliding-window noise teacher、uncertainty teacher、confidence-gated fallback | 保留 narrative，同时提高新颖性 |
| D. Feature window         | syndrome statistics                                | histogram stack、moments、detector history、compressed features               | 控制信息量和实验成本           |
| E. Task / code family     | 物理层 GKP fast-path 主线不变                             | surface-GKP、outer code、QLDPC-GKP soft info                                 | 多数应先 research-only   |
| F. Governance / promotion | frozen anchor + software-HIL boundary              | artifact registry、rollback、canary metrics                                  | 保证并行不污染主线            |

这比单纯“模型替换”更稳，因为它把研究问题表述为：**在不破坏 hard real-time fast loop 的前提下，slow loop 如何更好地估计 drift 并安全更新 fast-loop parameters。**

---

# 5. Candidate route comparison table

下面是筛选后的候选路线。表中“建议”含义为：
**yes** = 建议进入后续任务池；**maybe** = 先研究或 toy，不进主实验池；**no** = 当前阶段不建议开。

|  # | 方向名称                                                          | 核心思想                                                                                                   | 代表工作 / 外部依据                                                                                                                                                                                                | 相容性 | 保留快慢双回路 | 保留 FPGA fast-loop 约束 | 可否并行       | 成本 / 周期                           | 论文价值                                    | 主要风险                                                  | 建议                           |
| -: | ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --- | ------- | -------------------- | ---------- | --------------------------------- | --------------------------------------- | ----------------------------------------------------- | ---------------------------- |
|  1 | **Histogram temporal stacking + tiny TCN residual `b` head**  | 把单帧 `32x32` histogram 扩展为短窗口序列，用 TCN/temporal CNN 学 `b` residual 或 drift velocity                      | TCN 经典研究显示卷积式时序模型在多类序列任务上可超过 canonical RNN，并有更长有效记忆；时序卷积比 RNN 更并行。([arXiv][4])                                                                                                                             | 很高  | 是       | 是                    | 是          | 小规模 retrain；bounded benchmark；短-中 | 高：直接服务 drift-adaptive hardware-safe QEC | 窗口长度和 drift 频谱不匹配；overfit histogram artifacts         | **yes**                      |
|  2 | **Teacher-conditioned CNN / FiLM-like residual head**         | CNN 不替代 classical estimator，而是把 teacher state、confidence、estimated noise prior 作为 conditioning 输入      | 校准条件化 / residual learning 与 noise-aware decoding 叙事兼容；noise-aware decoding 用 characterization 结果提升解码性能。([arXiv][5])                                                                                        | 很高  | 是       | 是                    | 是          | 小规模 retrain；短                     | 中-高：保守但可信                               | 新颖性不如 SSM/transformer；需要清楚写成安全残差                      | **yes**                      |
|  3 | **GRU / LSTM slow loop**                                      | 用 hidden state 建模 drift temporal dependency                                                            | RNN 适合序列直觉，但 TCN 文献表明卷积式序列模型常是更自然起点。([arXiv][4])                                                                                                                                                           | 中   | 是       | 是                    | 是          | 需要 retrain；中                      | 中                                       | 串行 hidden state、rollback 难、并行性差、部署叙事弱                 | **maybe / low priority**     |
|  4 | **S4 / Mamba slow-loop estimator**                            | 用 state-space model 捕捉长时间 drift；适合长序列和线性扩展                                                             | S4 提供结构化 state-space 长序列建模；Mamba 提出 selective state spaces 和 hardware-aware parallel algorithm，线性扩展到长序列。([arXiv][6])                                                                                       | 中   | 是       | 大体是                  | 可做 toy     | toy/design；中-长                    | 高：现代模型叙事强                               | runtime/量化/可验证性不清；对当前 `b` residual 可能过重               | **maybe / research first**   |
|  5 | **Lightweight transformer / recurrent transformer slow loop** | 用 attention 或 recurrent transformer 捕捉长时程 syndrome pattern                                             | Recurrent transformer decoder 在 Sycamore surface-code 数据和模拟距离扩展上表现强。([Nature][7])                                                                                                                          | 中-低 | 可保留     | 易偏离                  | 可离线研究      | 训练成本高；中-长                         | 高，但更像 full decoder 论文                   | 容易把项目从 calibration residual 改成 generic neural decoder | **maybe / research only**    |
|  6 | **Syndrome-only adaptive drifting-noise teacher**             | 用 sliding window / overlapping window 从 syndrome statistics 估计 time-dependent noise，再作为 teacher anchor | 2025 drifting-noise 工作显示仅用 syndrome statistics 可估计 time-dependent Pauli noise，adaptive decoding 可降低 logical error。([arXiv][8])                                                                             | 很高  | 是       | 是                    | 是          | toy + bounded benchmark；短-中       | 高：与 drift-adaptive narrative 完全一致       | teacher estimator 可能慢；窗口参数敏感                          | **yes**                      |
|  7 | **Noise-aware teacher / adaptive prior optimization**         | 使用 calibrated/noise-aware prior 指导 slow-loop residual；不让 NN 全权解码                                       | ACES/noise-aware decoding 表明 Pauli characterization 可校准 correlated matching decoder，并改善 error suppression。([arXiv][5]) Belief-matching 也强调利用 full noise information 可提升 surface-code decoding。([arXiv][9]) | 高   | 是       | 是                    | 是          | 设计 + bounded；短-中                  | 高                                       | 需要定义 teacher confidence 与 residual target             | **yes**                      |
|  8 | **Confidence-gated teacher / fallback teacher**               | teacher 输出 confidence；低置信度时回退 classical anchor 或冻结 bank                                                | Decoder-switching 方向把 weak/strong decoder 选择作为性能-成本 tradeoff；可借鉴为 teacher fallback gating。([arXiv][10])                                                                                                    | 很高  | 是       | 是                    | 是          | docs + bounded replay；很短-短        | 中-高：工程安全性强                              | 论文新颖性需与 drift/bank safety 绑定                          | **yes**                      |
|  9 | **Histogram + moments + drift features**                      | 在 `32x32` hist 外加 EWMA、entropy、marginals、delta moments、teacher residual statistics                     | 与 syndrome-statistics teacher 和 compressed calibration 兼容；漂移估计文献支持从 syndrome statistics 提取 time-dependent noise 信息。([arXiv][8])                                                                            | 很高  | 是       | 是                    | 是          | docs/toy；很短-短                     | 中                                       | 信息量可能不足；需 ablation                                    | **yes**                      |
| 10 | **Raw time-series sequence input**                            | 直接喂 syndrome/event 序列而非 histogram                                                                      | 神经 decoder 近期工作利用 spatiotemporal syndrome、soft readout、leakage 等 richer features。([Nature][7])                                                                                                             | 中-低 | 可保留     | 可保留但 I/O 重           | 可离线        | 新数据管线；中-长                         | 中-高                                     | 存储/训练成本大；偏离 histogram 主线                              | **maybe / not first**        |
| 11 | **Detector-history / graph representation / GNN**             | 把 syndrome history 转为 detector graph，使用 GNN 或 message passing                                          | GNN data-driven decoding 把 stabilizer measurements 转为 annotated detector graph，推理近似线性，但训练计算量大。([arXiv][11])                                                                                                | 中   | 不一定     | 不适合 fast loop        | 可 research | 大量新 pipeline；长                    | 高                                       | 会转向 full decoder；训练重                                  | **maybe / research only**    |
| 12 | **surface-GKP / concatenated outer code**                     | 保持 GKP fast-path，但外接 surface/XZZX code soft-info decoder                                               | XZZX-surface GKP 研究显示 concatenated GKP + surface-family code 有阈值和 overhead 优势。([arXiv][12])                                                                                                                | 中   | 可保留     | 部分保留                 | 只适合小 toy   | 新任务定义；中-长                         | 高                                       | 主问题被改写；benchmark 不可直接比较                               | **maybe / research only**    |
| 13 | **QLDPC-GKP soft-information outer decoder**                  | 用 GKP analog/soft info 辅助 QLDPC outer decoder                                                          | 2025 QLDPC-GKP 工作强调 inner bosonic syndrome 的 analog information 可显著帮助 outer code，QLDPC 有更高编码率潜力。([arXiv][13])                                                                                              | 低-中 | 可概念保留   | fast path 需重定义       | 不适合短期      | 新 simulator/decoder；很长            | 很高                                      | 完全改写任务；主线阻塞风险高                                        | **maybe / future-work only** |
| 14 | **Gain scheduling / piecewise affine fast loop**              | 从单一 `K,b` 扩展到 regime-conditioned `K_i,b_i`，slow loop 选择 bank                                           | 与 FPGA deterministic bank switching 高度一致；外部 real-time FPGA decoder 研究也强调 deterministic closed-loop latency。([arXiv][2])                                                                                    | 很高  | 是       | 是                    | 是          | software-HIL contract sim；短-中     | 高                                       | regime 切换抖动；需 atomic commit/rollback                  | **yes**                      |
| 15 | **LUT-assisted correction**                                   | 对局部 syndrome regime 用 small LUT 修正 affine residual                                                     | FPGA 友好；可与 reduced-precision / LUT inference 叙事结合                                                                                                                                                          | 高   | 是       | 是                    | 是          | contract sim；短                    | 中                                       | LUT 爆炸；覆盖率有限                                          | **yes**                      |
| 16 | **Low-bit learned micro-head in fast loop**                   | 在 affine fast path 后加 very tiny quantized NN residual，并用 safety bounds 限幅                              | 外部 FPGA NN decoder 已展示 NN 可在 FPGA low-latency 场景中运行，但那是完整硬件系统成果。([arXiv][2])                                                                                                                               | 中   | 是       | 有压力                  | 可 toy      | 需改 fast-loop contract；中-长         | 高                                       | 验证难、确定性和 safety proof 更复杂                             | **maybe / later**            |
| 17 | **Staged bank commit / rollback-aware control**               | 参数更新先 shadow bank，再 canary replay，最后 atomic commit；异常 rollback                                         | 与 fast-loop safety governance 强一致                                                                                                                                                                          | 很高  | 是       | 是                    | 是          | docs + mock HIL；很短                | 中-高：工程可靠性卖点                             | 不是单独算法，需绑定其他路线                                        | **yes**                      |
| 18 | **Diffusion / autoregressive decoder**                        | 用生成模型解决复杂 decoding                                                                                     | 2025 QLDPC diffusion decoder 展示了 masked diffusion 在 QLDPC decoding 上的潜力。([arXiv][14])                                                                                                                      | 低   | 不自然     | 不适合                  | 可读文献       | 长                                 | 高但偏题                                    | 多步随机推理、latency 和 determinism 冲突                       | **no**                       |

---

# 6. Most relevant recent works from the last 1–3 years

## 6.1 实时 QEC / FPGA 方向

实时 QEC 的核心约束是 decoder 必须跟上 QEC cycle，并能集成到 hard real-time controller 中；综述类研究明确把“快速、准确、可实时集成”作为 decoder 的系统要求。([arXiv][1]) 后续 controller-decoder 系统需求研究进一步把 closed-loop latency 放在 tens of microseconds 级别的系统约束内。([arXiv][15])

2025 年 Google below-threshold surface-code 实验报告了距离 5 与距离 7 的逻辑性能，并在距离 5 系统中集成 real-time decoder，报告平均 decoder latency 为 63 μs；这是实时解码已进入系统实验的强信号。([arXiv][3])

2026 年 FPGA-based NN decoder 工作展示了外部系统中 deterministic closed-loop latency 550 ns，其中 NN decoding 124 ns，处在 1.25 μs QEC cycle 内。这个结果说明 FPGA 上的 learned decoder/NN inference 在原则上可行，但它不能被解释为 `DriftAdaptiveQEC` 已完成 real-board FPGA validation。([arXiv][2])

## 6.2 神经 decoder 与时序 syndrome

Recurrent transformer decoder 在近年 surface-code decoding 中表现突出，尤其是在 Google Sycamore 数据和更大距离模拟上展示了高精度潜力；这说明 transformer/recurrent-transformer 是强 decoder 方向。([Nature][7]) 但它的目标通常更接近 full decoder，而不是你的当前慢回路 residual calibration，因此短期不应直接替代 teacher-anchored CNN 主线。

GNN detector-graph decoder 把 syndrome / stabilizer measurement 映射为 detector graph，推理可近似线性，但训练计算量较大；它适合作为研究方向，不适合作为当前 Phase 2 的低成本并行实验。([arXiv][11])

## 6.3 漂移噪声、teacher 与 noise-aware decoding

2025 年 adaptive drifting-noise estimation 工作非常贴近你的项目：它研究仅从 syndrome statistics 估计 time-dependent Pauli noise，并用 sliding/overlapping window 捕捉 drift frequency，进而通过 adaptive decoding 降低 logical error。([arXiv][8]) 这强烈支持“syndrome-only adaptive teacher + neural residual”路线。

Noise-aware decoding 方向也很相关。ACES/noise-aware decoding 工作显示，通过 Pauli noise characterization 校准 correlated matching decoder 可以改善 error suppression，并且把 calibration 与 decoder prior 连接起来。([arXiv][5]) Belief-matching / belief-find 进一步说明利用完整噪声信息可以提升 decoding 性能。([arXiv][9])

## 6.4 GKP / outer-code soft information

GKP 与 outer code 的结合仍是高价值方向。XZZX-surface GKP 工作显示，GKP 与 surface-family code concatenation 可改善 threshold / overhead。([arXiv][12]) 2025 QLDPC-GKP soft-information 工作进一步强调 inner bosonic syndrome 的 analog information 可以帮助 outer QLDPC decoder，并指出 QLDPC outer codes 有更高 encoding rate 潜力。([arXiv][13]) 但这些方向会显著改写任务定义，短期应保持 research-only。

## 6.5 TCN / SSM / Mamba

TCN 是最适合近期实验的现代化慢回路替代。经典 TCN 研究显示，卷积式序列模型在多类 sequence modeling benchmark 上可超过 canonical recurrent networks，并且具有更长有效记忆。([arXiv][4])

S4 和 Mamba 是更现代的 long-sequence 路线。S4 提供结构化 state-space 长序列建模；Mamba 引入 selective state spaces 和 hardware-aware parallel algorithm，并以线性复杂度处理长序列。([arXiv][6]) 它们值得调研和 toy simulation，但当前不应优先进入主实验池。

---

# 7. CNN vs RNN vs TCN vs SSM vs Transformer judgment

## 7.1 CNN 是否过时？

**不建议把当前 teacher-anchored CNN 判定为过时。**

更准确的判断是：**CNN 作为单帧 histogram residual learner 不够新，但作为 hardware-safe residual calibration head 仍然合理。** 它的优点是结构简单、可量化、推理稳定、容易做 bounded residual、不会破坏 FPGA fast-loop narrative。问题在于单帧 `32x32 histogram` 缺少 drift temporal direction，因此应扩展输入表示，而不是先激进换模型。

**结论：CNN 不应被淘汰；应升级为 temporal / teacher-conditioned residual head。**

## 7.2 CNN 换成 RNN / GRU / LSTM 是否值得？

**短期不值得作为高优先级方向。**

RNN/GRU/LSTM 的直觉优势是能建模 drift history，但它们有四个工程问题：

1. hidden state 串行依赖强，不如 TCN 易并行；
2. rollback / deterministic replay 更麻烦；
3. hidden state reset、warm-up、bank switching 边界更复杂；
4. 量化与 fixed-point verification 叙事不如小型卷积清晰。

TCN 文献已经给出一个重要经验：对于很多序列任务，卷积式序列模型是比 canonical RNN 更自然的起点，并且表现可以更好。([arXiv][4])

**判断：GRU/LSTM 可以做一个 very small baseline，但不应进入 Top-3 实验池。**

## 7.3 TCN 是否更值得？

**是。TCN / temporal CNN 是短期最高性价比的模型替代。**

原因：

| 维度                 | TCN 判断                                          |
| ------------------ | ----------------------------------------------- |
| drift 建模           | 固定 receptive field 适合窗口化 drift                  |
| 并行性                | 优于 RNN                                          |
| 部署                 | 卷积结构更容易量化和静态 shape 编译                           |
| 与 histogram        | 可以直接用 `T x 32 x 32` 或 compressed temporal stack |
| 与 teacher residual | 很自然：输出 `Δb`、confidence、bank score               |
| 与 FPGA fast loop   | 不改变 fast loop，只影响 slow-loop parameter update    |

**结论：TCN 是最值得现在试的慢回路模型扩展。**

## 7.4 S4 / Mamba 是否比 RNN 更值得？

**比 RNN 更值得调研，但不比 TCN 更适合马上进入主实验池。**

S4/Mamba 的优势是长序列、线性扩展、现代时序建模能力；Mamba 还强调 hardware-aware parallel algorithm。([arXiv][6]) 但你的当前任务不是通用 long-context forecasting，而是安全地估计 drift 并更新 FPGA fast-loop parameters。S4/Mamba 的工程风险包括：

* 需要新的训练与 runtime 路径；
* fixed-point / quantization envelope 不如 CNN/TCN 清楚；
* 模型复杂度可能超过 `b` residual 的必要性；
* paper narrative 可能变成“用了新模型”，而不是“硬件安全闭环 QEC”。

**结论：S4/Mamba 适合 research-only + toy simulation，不应先于 TCN。**

## 7.5 Transformer / recurrent-transformer 是否值得？

**作为文献定位很重要，作为当前慢回路替代不优先。**

Recurrent transformer decoder 在 surface-code decoding 上非常强，尤其是 full syndrome decoder 场景。([Nature][7]) 但你的当前主线不是 full neural decoder，而是 teacher-guided residual calibration。直接引入 transformer 可能导致：

* 模型语义从 calibration 变成 decoding；
* 训练成本上升；
* 与 FPGA fast loop 的确定性叙事冲突；
* 很难在主线 benchmark 空档中低成本推进。

**结论：transformer 应用于 paper positioning / offline teacher 研究，不建议现在作为实验池主路线。**

---

# 8. Teacher alternatives judgment

## 8.1 Teacher anchored 是否显老旧？

**不显老旧。它不是最“潮”的 ML 叙事，但非常适合当前项目。**

teacher anchored 的价值在于：

| 价值           | 对当前项目的意义                                               |
| ------------ | ------------------------------------------------------ |
| 稳定性          | teacher 可作为 fallback，避免 NN residual 失控                 |
| 可解释性         | 输出可被解释为 calibration / residual，不是 black-box correction |
| 硬件安全         | NN 不直接进入 fast loop critical path                       |
| 可验证性         | 可以对 `Δb`、bank switch、confidence 做边界检查                  |
| narrative 一致 | 与 Phase 2 controlled development 高度匹配                  |

近年的 noise-aware decoding 和 adaptive noise estimation 工作并没有否定 teacher/prior，反而说明 decoder prior、noise estimation、syndrome statistics 对性能有实际价值。([arXiv][8])

**结论：teacher anchored 不是老旧路线，而是当前项目的安全底座。创新点应从 fixed teacher 升级为 adaptive / uncertainty-aware teacher。**

## 8.2 最值得试的 teacher 方向

| Teacher 方向                                         | 推荐度 | 理由                                                                        |
| -------------------------------------------------- | --: | ------------------------------------------------------------------------- |
| **Syndrome-only sliding-window drift teacher**     |  很高 | 与 drift-adaptive QEC 叙事完全一致；不需要 real board；可用 cached syndrome / hist logs |
| **Overlapping-window / multi-timescale teacher**   |   高 | 可捕捉快慢 drift；适合输出 teacher confidence                                       |
| **Noise-aware prior teacher**                      |   高 | 与 ACES、belief-matching 等方向一致；适合作为 stable anchor                           |
| **Confidence-gated fallback teacher**              |  很高 | 工程价值大；能定义 rollback / freeze / fallback                                    |
| **Model-based Bayesian teacher + neural residual** |   中 | 论文价值高，但可能增加复杂度                                                            |
| **Full learned teacher / neural-only teacher**     |   低 | 与安全边界冲突，不适合当前阶段                                                           |

## 8.3 推荐 teacher 组合

最稳组合是：

[
\text{teacher output} = (\hat{b}*{teacher}, \Sigma*{teacher}, c_{teacher}, regime_id)
]

然后 slow-loop residual head 只学习：

[
\Delta b_{residual} = f_\theta(H_{t-k:t}, \hat{b}*{teacher}, c*{teacher})
]

最后进入 fast-loop 的更新受限为：

[
b_{new} = b_{teacher} + \mathrm{clip}(\Delta b_{residual}, \epsilon)
]

并配合 confidence gate：

[
\text{commit} =
\begin{cases}
\text{yes}, & c_{teacher} > \tau_c \land |\Delta b| < \tau_b \
\text{hold/fallback}, & \text{otherwise}
\end{cases}
]

这条路线非常适合当前 Phase 2，因为它把神经网络限制在 residual calibration，而不是替代 fast-loop correction。

---

# 9. Feature representation alternatives judgment

## 9.1 当前 `32 x 32 histogram` 是否足够？

**作为稳定 baseline 足够；作为 drift estimator 单独使用不够。**

`32x32 histogram` 的优势是压缩、稳定、易训练、易缓存、适合 slow-loop；问题是它丢失了时间方向和 drift frequency 信息。近期 drifting-noise estimation 工作强调从 syndrome statistics 的时间窗口中估计 time-dependent noise，因此“单帧 histogram”应升级为“短时序 histogram / 多尺度统计”。([arXiv][8])

## 9.2 推荐特征优先级

| 特征表示                                           |           优先级 | 判断                                                            |
| ---------------------------------------------- | ------------: | ------------------------------------------------------------- |
| **Histogram temporal stacking**                |            最高 | 最小改动获得 temporal signal；适合 TCN                                 |
| **Histogram + moments / EWMA / deltas**        |            最高 | 低成本、高可解释；适合 teacher confidence                                |
| **Compressed FPGA-aware calibration features** |             高 | 适合 slow-to-fast contract，例如输出 `Δb`, `regime_id`, `confidence` |
| **Raw time-series syndrome**                   |           中-低 | 信息量高但成本高；不适合先做                                                |
| **Event-sequence / detector-history**          |             中 | 对 full decoder 有价值；对 `b` residual 可能过重                        |
| **Graph / spatiotemporal GNN**                 | research-only | 论文价值高，但会把项目推向 generic decoder                                 |

## 9.3 最推荐的特征设计

建议下一步先试三种轻量表示：

1. **Temporal histogram stack**
   [
   X_t = [H_t, H_{t-1}, ..., H_{t-k}]
   ]

2. **Multi-timescale EWMA features**
   [
   \mathrm{EWMA}*{short}, \mathrm{EWMA}*{mid}, \mathrm{EWMA}_{long}
   ]

3. **Teacher residual statistics**
   [
   r_t = b_{observed/estimated} - b_{teacher}
   ]

这些特征能在实验成本和信息量之间取得最好平衡。

---

# 10. Code-family expansion judgment

## 10.1 维持物理层 GKP fast-path 是否最合理？

**是。短期应维持物理层 GKP fast-path 为主任务。**

原因是当前项目的主贡献集中在 drift-aware calibration、slow-loop teacher residual、FPGA-friendly fast loop。如果现在把任务扩展到 surface-GKP 或 QLDPC-GKP，问题会从“如何安全更新 fast-loop correction parameters”变成“如何设计 outer-code decoder”。这会稀释主线。

## 10.2 surface-GKP / concatenated outer code

**建议：Research only for now。**

surface-GKP 和 XZZX-surface-GKP 有明确论文价值。相关工作显示 GKP 与 XZZX surface code concatenation 可带来 threshold / overhead 改善。([arXiv][12]) 但它会引入新的 outer-code syndrome、decoder、logical error metric、benchmark protocol，不适合在当前主线长跑期间作为短期工程实验。

适合做的程度：

| 层级                           | 是否建议    |
| ---------------------------- | ------- |
| paper positioning            | 是       |
| small toy interface          | 可以      |
| bounded benchmark            | 谨慎      |
| full rerun / main experiment | 不建议当前阶段 |

## 10.3 QLDPC-GKP

**建议：future-work only 或 research-only。**

QLDPC-GKP soft-information 方向非常有前沿价值，2025 工作明确强调 analog bosonic syndrome 信息可帮助 QLDPC outer decoder，且 QLDPC outer codes 有高编码率潜力。([arXiv][13]) 但这几乎是新项目：新的码族、新的 decoder、可能新的 hardware mapping、新的 benchmark，不适合作为当前 Phase 2 并行实验池。

## 10.4 Bosonic soft-information outer decoder setting

**建议：作为 narrative 和 future-work 写入，但不要短期大规模实现。**

它与当前项目有概念连接：你的 slow loop 已经在利用 syndrome statistics 和 calibration residual，而 bosonic soft-information outer decoder 也强调 soft information 的价值。但当前主线的 fast-loop FPGA contract 不是 outer-code soft decoder，因此短期实现会重写问题。

---

# 11. Fast-loop FPGA logic expansion judgment

## 11.1 `K @ s + b` 是否仍应保留？

**应保留为 golden fast path。**

它是当前架构最重要的工程资产：低延迟、确定性、fixed-point 友好、bank switching 友好、容易 mock-HIL 验证。外部实时 QEC 研究也说明 hard real-time decoder integration 是核心系统约束。([arXiv][1])

## 11.2 最值得扩展的 fast-loop 形式

| 快回路扩展                           | 推荐度 | 判断                                    |
| ------------------------------- | --: | ------------------------------------- |
| **Gain scheduling**             |  很高 | 最自然；slow loop 选择 regime-specific gain |
| **Piecewise affine**            |  很高 | 保持 deterministic；可解释；适合 bank          |
| **LUT-assisted correction**     |   高 | FPGA-friendly；适合小区域 residual          |
| **Staged parameter bank**       |  很高 | 是安全治理必须项                              |
| **Atomic commit / rollback**    |  很高 | 是从 mock-HIL 走向更强 HIL 的必要控制机制          |
| **Low-bit neural micro-head**   | 中-低 | 可 toy；不应先进入 fast-loop contract        |
| **Unbounded neural correction** |   低 | 破坏确定性和验证性                             |

## 11.3 推荐 fast-loop extension contract

保持原始 fast-loop 形式：

[
\Delta = K_i s + b_i + \delta_{LUT}(s, i)
]

其中：

* `i = regime_id` 由 slow loop 选择；
* `K_i, b_i` 来自 staged parameter bank；
* `δ_LUT` 可选，必须 bounded；
* commit 必须 atomic；
* rollback 必须可 replay；
* 所有 fixed-point range 必须在 mock-backed software-HIL 中先验证。

这条路线不要求 real-board FPGA，也不要求 true `.tflite` runtime。

---

# 12. Which routes are truly parallelizable with the current mainline

以下路线可以在主线 2–4 天 benchmark 运行期间并行推进，因为它们可以用 cached data、toy simulation、bounded replay 或 contract-level mock-HIL 完成。

| 并行路线                                             | 为什么不阻塞主线                                            | 推荐动作                     |
| ------------------------------------------------ | --------------------------------------------------- | ------------------------ |
| **Temporal histogram stack + tiny TCN residual** | 可用已生成 histogram / syndrome logs；不改主线代码路径            | 先做 bounded benchmark     |
| **Histogram + moments / EWMA feature ablation**  | 主要是特征工程和小模型训练                                       | 先做 docs + toy            |
| **Adaptive syndrome-only teacher**               | 可独立读取 syndrome statistics；teacher 输出 sidecar target | 先做 teacher replay        |
| **Confidence-gated fallback teacher**            | 不需大训练；可基于已有 teacher 输出模拟                            | 先做 safety policy         |
| **Gain scheduling / piecewise affine bank sim**  | fast-loop contract 可 mock；不需 real board             | 先做 bank-selection replay |
| **Atomic commit / rollback mechanism**           | 工程治理路线；可 mock-HIL 验证                                | 先做 contract tests        |
| **FiLM / teacher-conditioned residual head**     | 可复用当前 CNN pipeline，只加 conditioning inputs           | 小规模 retrain              |

---

# 13. Which routes should remain future-work only

| 路线                                                    | 当前不适合原因                                                        |
| ----------------------------------------------------- | -------------------------------------------------------------- |
| **Full recurrent transformer decoder**                | 强研究方向，但更像 full neural decoder，会偏离 teacher residual calibration |
| **Large transformer slow loop**                       | 参数量、训练成本、验证复杂度偏高，不适合 Phase 2                                   |
| **S4/Mamba full slow-loop replacement**               | 值得研究，但 runtime、量化和 safety envelope 尚不清楚                        |
| **Raw time-series full input**                        | 数据管线和存储成本高，可能与 histogram 主线割裂                                  |
| **Detector-graph / GNN full decoder**                 | 高论文价值，但训练重、任务语义变化大                                             |
| **surface-GKP full benchmark**                        | 会引入 outer-code decoder 和新 metric，不适合短期并行                       |
| **QLDPC-GKP full implementation**                     | 几乎是新项目；应作为 future-work                                         |
| **Diffusion / autoregressive decoder**                | 多步生成推理和 hard real-time deterministic fast loop 冲突              |
| **Adaptive syndrome extraction**                      | 会改变量子测量 schedule，不是当前 software-HIL 边界内的自然 extension            |
| **Real-board HIL validation as extension lane**       | 这是集成里程碑，不是可随主线并跑的算法扩展                                          |
| **`.tflite` true runtime recovery as extension lane** | 这是 runtime recovery / deployment gate，不应混入研究路线池                |

---

# 14. Ranked shortlist

## 14.1 Top 3 routes worth trying next

### Rank 1 — Temporal histogram stack + tiny TCN residual `b` head

**推荐等级：Recommended now**

这是最符合当前项目状态的模型扩展路线。它保留 histogram 主线，又补上 temporal drift 信息；保留 teacher anchored residual，又能显著增强慢回路表达力；保留 FPGA fast loop，因为输出仍然只是 `b` residual 或 bank score。

**建议实验：**

| 项目       | 设置                                                             |
| -------- | -------------------------------------------------------------- |
| 输入       | `T x 32 x 32` histogram stack + optional moments               |
| 模型       | tiny TCN / temporal CNN                                        |
| 输出       | `Δb`, confidence, optional `regime_id`                         |
| baseline | current CNN residual head                                      |
| 指标       | residual MSE、logical proxy、bank switch frequency、rollback rate |
| 周期       | 短-中                                                            |
| 风险       | overfit；窗口长度敏感                                                 |

### Rank 2 — Adaptive syndrome-only teacher + confidence-gated fallback

**推荐等级：Recommended now**

这是最能增强 teacher anchored 路线新颖性的方向。它不问“teacher 是否过时”，而是把 teacher 升级为 drift-aware、uncertainty-aware、fallback-aware。2025 drifting-noise estimation 工作直接支持 syndrome statistics 可用于 time-dependent noise estimation。([arXiv][8])

**建议实验：**

| 项目       | 设置                                                    |
| -------- | ----------------------------------------------------- |
| teacher  | sliding window / overlapping window estimator         |
| 输出       | `b_teacher`, `noise_prior`, `confidence`, `regime_id` |
| residual | CNN/TCN 学 bounded `Δb`                                |
| fallback | low confidence 时 freeze 或回退 anchor                    |
| 周期       | 短-中                                                   |
| 风险       | teacher confidence calibration                        |

### Rank 3 — Piecewise-affine / gain-scheduled FPGA fast-loop bank

**推荐等级：Recommended now**

这是 fast-loop 结构扩展中最稳的路线。它不是把复杂 ML 放进 FPGA critical path，而是让 FPGA 仍执行 deterministic affine correction，只是允许 slow loop 选择不同 bank。

**建议实验：**

| 项目          | 设置                                 |
| ----------- | ---------------------------------- |
| fast path   | `Δ = K_i s + b_i`                  |
| bank select | teacher / TCN 输出 `regime_id`       |
| safety      | atomic commit、rollback、range check |
| 验证          | mock-backed software-HIL replay    |
| 周期          | 短-中                                |
| 风险          | regime oscillation；bank thrashing  |

---

## 14.2 Top 3 routes worth researching but not implementing yet

### Research 1 — S4 / Mamba slow-loop estimator

S4/Mamba 比 RNN 更值得调研，因为它们是更现代的 long-sequence modeling 路线，并且 Mamba 强调线性扩展和 hardware-aware algorithm。([arXiv][6]) 但当前应限于 toy simulation：输入 compressed temporal features，输出 `Δb` 或 teacher confidence，不要替换主线。

### Research 2 — surface-GKP / bosonic soft-information outer decoder positioning

这个方向论文价值高，能连接 GKP physical layer 与 outer-code soft information；但短期应作为 paper positioning 和 small toy，不应改主线 benchmark。([arXiv][12])

### Research 3 — QLDPC-GKP / Relay-BP-style future architecture

QLDPC-GKP soft-information 是非常前沿的 future-work 方向。([arXiv][13]) Relay-BP / FPGA qLDPC 解码方向也显示 qLDPC real-time decoding 正在形成新路线。([arXiv][16]) 但它会改写 code family 和 decoder protocol，不适合当前 Phase 2 实验池。

---

# 15. Suggested next-task roadmap

## Stage 0 — Governance freeze

先固定 extension-lane 规则：

| 任务                        | 输出                                                     |
| ------------------------- | ------------------------------------------------------ |
| Frozen anchor manifest    | 明确哪张 table / 哪些 metrics 是正式主线锚点                        |
| Extension artifact schema | 所有新路线输出 sidecar JSON / CSV / config                    |
| Fast-loop ABI document    | 明确 syndrome input、correction output、fixed-point ranges |
| Mock-HIL label            | 所有结果标注 mock-backed software-HIL                        |
| Promotion gate            | 定义何时从 extension candidate 升级为 mainline candidate       |

## Stage 1 — 低成本并行路线

优先做三个小实验：

| 实验                                | 目标                                | 成本 |
| --------------------------------- | --------------------------------- | -- |
| Histogram temporal stack ablation | 比较 `H_t` vs `[H_{t-k:t}]`         | 很短 |
| Moments / EWMA features           | 测试是否提升 teacher residual stability | 很短 |
| Teacher confidence replay         | 测试 fallback / freeze policy       | 短  |

## Stage 2 — Bounded benchmark

进入 bounded benchmark 的三条：

| 实验                               | 目标                            |
| -------------------------------- | ----------------------------- |
| Tiny TCN residual head           | 替代/增强当前 CNN residual          |
| Adaptive syndrome-only teacher   | 输出 drift-aware teacher target |
| Piecewise affine bank simulation | 验证 bank switching 是否稳定        |

## Stage 3 — Integration candidate

只有当 Stage 2 同时满足以下条件时才进入 integration candidate：

| Gate          | 条件                                               |
| ------------- | ------------------------------------------------ |
| Accuracy      | 不低于 frozen anchor 或在 drift 场景显著改善                |
| Safety        | residual bounded；无异常 bank oscillation            |
| Determinism   | replay deterministic                             |
| Latency model | fast loop 不增加 critical-path latency              |
| Contract      | 不要求 real-board FPGA、不依赖 true `.tflite` recovery  |
| Narrative     | 仍是 drift-adaptive QEC，不变成 generic neural decoder |

## Stage 4 — Research-only backlog

并行做文献和 toy，不进入主实验：

| 方向                    | 输出                              |
| --------------------- | ------------------------------- |
| S4/Mamba              | 设计 memo + tiny toy              |
| Recurrent transformer | paper positioning + 不做大训练       |
| surface-GKP           | task interface memo             |
| QLDPC-GKP             | future-work architecture sketch |
| GNN detector graph    | compare-only literature note    |

---

# 16. 对你最关心问题的直接回答

## 16.1 CNN 换成 RNN / GRU / LSTM 是否值得？

**不值得作为短期高性价比方向。**

RNN/GRU/LSTM 的确更像时间序列模型，但它们在训练并行性、hidden-state replay、rollback、fixed-point verification 和工程叙事上都不如 TCN。最合理做法是：保留 CNN baseline，加一个 very small GRU 作为低优先级 ablation，但不要把它作为主扩展路线。

## 16.2 比 RNN 更现代的慢回路模型是否更值得优先调研？

**是，但优先级顺序应是：TCN > S4/Mamba > lightweight transformer > GRU/LSTM。**

TCN 最适合当前实验池；S4/Mamba 最适合 research-only；transformer 适合定位 full decoder literature，不适合作为当前 slow-loop residual head 的第一选择。

## 16.3 teacher anchored 路线是否已显老旧？

**不是。**

teacher anchored 的学术新颖性不如“纯神经 decoder”，但它和当前硬件约束、安全边界、Phase 2 controlled development 更匹配。真正应该升级的是 teacher 的形式：从固定 anchor 升级为 adaptive prior、uncertainty-aware teacher、confidence-gated fallback teacher。

## 16.4 哪些方向适合主线运行期间并行推进？

适合现在并行推进的是：

1. temporal histogram + TCN residual；
2. histogram moments / EWMA features；
3. syndrome-only adaptive teacher；
4. confidence-gated fallback；
5. piecewise-affine / gain-scheduled bank simulation；
6. atomic commit / rollback control；
7. teacher-conditioned CNN residual head。

这些路线都不要求 real-board FPGA，不要求 true `.tflite` runtime，也不需要重写 frozen anchor。

## 16.5 哪些方向不适合现在开？

不适合现在开的是：

1. full recurrent transformer decoder；
2. large transformer / full neural decoder 替代；
3. raw time-series full input 大管线；
4. detector-graph GNN full decoder；
5. surface-GKP full benchmark；
6. QLDPC-GKP full implementation；
7. diffusion / autoregressive decoder；
8. adaptive syndrome extraction；
9. real-board HIL 作为“扩展实验”；
10. `.tflite` runtime recovery 作为“研究路线”。

---

# 17. Final project-management conclusion

## Recommended now

**Recommended now — 建议现在进入后续扩展路线池，但必须作为 sidecar extension lanes，并满足 governance checklist。**

具体推荐进入实验池的是：

1. **Histogram temporal stacking + tiny TCN residual `b` head**
   原因：最高兼容性、低成本、能直接解决 drift temporal information 不足。

2. **Adaptive syndrome-only teacher + confidence-gated fallback**
   原因：强化 teacher anchored 路线，而不是推翻它；与漂移噪声估计和硬件安全叙事一致。

3. **Piecewise-affine / gain-scheduled FPGA parameter bank + atomic commit / rollback**
   原因：保留 FPGA fast loop 的确定性与 fixed-point 友好，同时提供真正的 fast-loop structure extension。

## Research only for now

**Research only for now — 适合先做调研、设计 memo 或 toy simulation，不适合立刻进主实验池。**

包括：

1. **S4 / Mamba slow-loop estimator**
   前沿、有潜力，但工程和验证成本尚不适合优先落地。

2. **Lightweight / recurrent transformer**
   文献价值高，但更像 full decoder，不适合当前 residual calibration 主线。

3. **surface-GKP / QLDPC-GKP soft-information outer decoder**
   论文价值高，但会改变任务定义，应作为 positioning / future-work。

4. **GNN detector-history decoder**
   可作为未来 decoder family 对照，但当前训练和 pipeline 成本偏高。

## Not recommended in current phase

**Not recommended in current phase — 当前阶段不建议开。**

包括：

1. **full neural fast-loop correction without safety bounds**
   原因：破坏 deterministic FPGA fast-loop 和可验证性。

2. **diffusion / autoregressive decoder**
   原因：多步随机推理与 hard real-time fast loop 冲突。

3. **large raw time-series pipeline**
   原因：数据和训练成本高，容易阻塞主线。

4. **real-board FPGA validation / true `.tflite` recovery 作为普通 extension lane**
   原因：它们是集成里程碑或 runtime gate，不是可与主线 benchmark 并跑的研究扩展。

**最终一句话结论：Recommended now — 可以现在并行开扩展路线池，但只推荐开启“temporal TCN residual、adaptive teacher、piecewise-affine FPGA bank”这类保留快慢双回路和 FPGA fast-loop contract 的受控 sidecar 路线；其余高风险模型和码族扩展应先保持 research-only 或 future-work。**

[1]: https://arxiv.org/abs/2303.00054 "[2303.00054] Real-Time Decoding for Fault-Tolerant Quantum Computing: Progress, Challenges and Outlook"
[2]: https://arxiv.org/html/2605.04892v1 "Real-time Surface-Code Error Correction Using an FPGA-based Neural-Network Decoder"
[3]: https://arxiv.org/abs/2408.13687?utm_source=chatgpt.com "Quantum error correction below the surface code threshold"
[4]: https://arxiv.org/abs/1803.01271 "[1803.01271] An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling"
[5]: https://arxiv.org/abs/2502.21044?utm_source=chatgpt.com "Improving error suppression with noise-aware decoding"
[6]: https://arxiv.org/abs/2111.00396?utm_source=chatgpt.com "Efficiently Modeling Long Sequences with Structured State ..."
[7]: https://www.nature.com/articles/s41586-024-08148-8?utm_source=chatgpt.com "Learning high-accuracy error decoding for quantum ..."
[8]: https://arxiv.org/abs/2511.09491?utm_source=chatgpt.com "Adaptive Estimation of Drifting Noise in Quantum Error Correction"
[9]: https://arxiv.org/abs/2203.04948?utm_source=chatgpt.com "Improved decoding of circuit noise and fragile boundaries of tailored surface codes"
[10]: https://arxiv.org/html/2510.25222v1?utm_source=chatgpt.com "Decoder Switching: Breaking the Speed-Accuracy Tradeoff ..."
[11]: https://arxiv.org/abs/2307.01241?utm_source=chatgpt.com "Data-driven decoding of quantum error correcting codes using graph neural networks"
[12]: https://arxiv.org/abs/2207.04383?utm_source=chatgpt.com "Concatenation of the Gottesman-Kitaev-Preskill code with the XZZX surface code"
[13]: https://arxiv.org/abs/2505.06385?utm_source=chatgpt.com "Fault Tolerant Decoding of QLDPC-GKP Codes with Circuit Level Soft Information"
[14]: https://arxiv.org/abs/2509.22347?utm_source=chatgpt.com "Decoding quantum low density parity check codes with diffusion"
[15]: https://arxiv.org/abs/2412.00289 "[2412.00289] Controller-decoder system requirements derived by implementing Shor's algorithm with surface code"
[16]: https://arxiv.org/html/2510.21600v1?utm_source=chatgpt.com "Real-time decoding of the gross code memory with FPGAs."
