# 漂移自适应 GKP 经典控制：研究缺口与 Introduction 草稿

**任务：** T0.2.2  
**日期：** 2026-07-14  
**适用阶段：** 研究设计与后续论文 Introduction/Related Work 的 claim contract；不是结果声明。  
**证据底座：** `docs/literature_matrix.md`

## 1. 一句话论点

在 repeated quantum-memory error correction 的 single-mode square approximate GKP setting 中，本项目要检验的是：**syndrome-history-aware 的主机慢回路能否估计连续漂移与离散健康状态，并把 history-dependent controller/decoder teacher 蒸馏为 FPGA 可执行的确定性低维定点 student，从而在未访问真实噪声状态的条件下，稳定缩小 static MAP 与 time-dependent oracle MAP 之间的 logical-performance gap；结论只在 finite-energy-aware 多保真仿真、强 baseline、因果故障注入、板级数字控制平面测量和 HIL 实际支持的证据层级内成立。**

这句话是研究假设与验收方向，不是当前仓库已经证明的结果。

## 2. 术语表

| Canonical term | 定义 | 禁止混用 |
| --- | --- | --- |
| approximate GKP qubit | Gaussian peaks + finite-energy envelope，或等效 damped/projector family 的 single-mode square GKP qubit | ideal comb state、surface-GKP 外码、多模 GKP |
| dual-loop classical-control architecture | host slow loop 更新低维参数/策略；FPGA fast domain 执行确定性 student，并包含 window/event health timescale | per-shot CNN decoder、主机模型进入 critical path |
| three timescales | subcycle/cycle fast path、window/event health update、host slow estimation/optimization | 三个独立 decoder |
| decoder oracle MAP | 每时刻读取真实 `DriftState theta_t` 的不可部署 MAP 上界 | control oracle、teacher、实际 decoder |
| finite-horizon control oracle | 在短时域数字孪生中访问模型/状态并优化 future action 的不可部署控制上界 | decoder oracle MAP |
| channel-recovery bound | QEC-matrix/Petz/transpose-channel 类编码—噪声恢复性能界 | 可执行 decoder、oracle MAP |
| recurrent teacher | model-aware、history-dependent 的训练/分析用 controller；可来自 Feedback-GRAPE/NMF 路线 | FPGA student、每周期实机推理 |
| deterministic student | 从 teacher 或统计策略蒸馏的 bounded fixed-point recurrence/MAP-LUT/affine rule | full GRU、CNN teacher |
| hardware-aware simulation | 显式包含 fixed-point、LUT/bank、FIFO、backlog、jitter、deadline 和 fallback 的模型 | measured FPGA/QPU result |
| logical metric | `P_L`/logical-channel metrics、operational pseudo-threshold、lifetime gain、oracle-gap closure | surface-code threshold 语言用于 single-mode GKP |

## 3. 已知进展：四条文献线分别解决了什么

### 3.1 GKP finite-energy physics 与真实纠错已经成立

GKP 编码、有限包络近似态、ancilla/data shift-error propagation、loss 下的有限能量恢复和 approximate-GKP logical channel 已有直接理论基础 [@gottesman_encoding_2001; @glancy_error_2006; @grimsmo_quantum_2021; @hastrup_analysis_2023; @jafarzadeh_logical_2025]。Cavity–transmon grid-state stabilization、autonomous GKP QEC 与 beyond-break-even real-time GKP QEC 也给出了真实实验标准 [Campagne-Ibarcq 2020, DOI:10.1038/s41586-020-2603-3; @lachance-quirion_autonomous_2024; @sivak_real-time_2023]。因此，本项目不能把 finite-energy GKP、measurement feedback 或实时 GKP QEC 本身写成新贡献。

### 3.2 Analog、soft 和 history-aware GKP 解码已经有直接先例

Analog GKP residual 已被用于 surface-GKP/QLDPC-GKP matching 或 message passing [@fukui_high-threshold_2018; @noh_fault-tolerant_2020; @noh_low_2022; @raveendran_finite_2022; @berent_analog_2024; @borah_fault_2025]；approximate-GKP 多轮 syndrome history 也已有 Bayesian memory-assisted decoder [@wan_memory-assisted_2020]。因此，“保留 analog syndrome”“利用历史”或“神经网络用于 bosonic/GKP decoding”都不能作为 novelty headline [@wang_multidimensional_2022]。

### 3.3 非平稳噪声估计与 decoder calibration 已形成成熟邻近线

Gaussian-process prediction、adaptive-window weight estimation、syndrome-only EM/HEM、decoding-graph reweighting、experimental decoder calibration、direct prior optimization 和 overlapping-window multi-frequency tracking，分别表明 noise state、weights 和 priors 可以从 error-correction data 中更新 [Huo 2017, arXiv:1710.03636; @spitz_adaptive_2018; @wagner_optimal_2021; @wang_dgr_2023; @chen_calibrated_2022; @sivak_optimization_2024; Bhardwaj 2025, arXiv:2511.09491]。Calibration-conditioned FiLM 和 RL control 进一步展示了跨 calibration state 的 learned adaptation 与利用 detection events 稳定物理控制 [@stein_calibration-conditioned_2026; @sivak_reinforcement_2026]。因此，本项目不能声称首次进行 syndrome-driven adaptation、decoder calibration 或 dual-timescale learning。

### 3.4 Neural/FPGA decoder 已达到真实低延迟甚至 QPU feedback

LUT、distributed Union-Find、collision/local clustering 与 fixed-point belief propagation 已给出 FPGA/ASIC latency、throughput、memory 和 resource 标准 [@das_lilliput_2021; @liyanage_scalable_2023; @barber_real-time_2023; @ziad_local_2024; @maurer_real-time_2025]。AlphaQubit 给出高质量 learned-decoder accuracy 标准 [@bausch_learning_2024]；Caune 2024 与 Yang 2026 已分别展示控制系统集成 FPGA decoder 和 FPGA neural decoder 的真实 QPU feedback [@caune_demonstrating_2024; @yang_real-time_2026]。因此，本项目不能用 software latency、synthesis estimate 或 HIL 替代 measured real-time QEC，也不能把 teacher 的 host/GPU inference 写进 FPGA critical path。

## 4. 精确缺口：不是单点算法空白，而是联合证据链空白

| 已有路线 | 已解决问题 | 与本项目最相关的未解决联合条件 |
| --- | --- | --- |
| finite-energy/experimental GKP | 真实 approximate-GKP physics、loss/ancilla/control effects、实验 feedback/lifetime | 通常不以非平稳 noise-state estimation、可部署 learned-to-deterministic student 和廉价 FPGA 数字控制平面为共同变量 |
| GKP analog/history decoder | soft likelihood、full-history Bayesian、outer-code dynamic weights | 多数使用 stationary/known noise 或特定外码；没有同时给出 drift/regime tracking、deadline/fallback 与板级 parameter-bank 证据 |
| adaptive/calibrated QEC | 从 syndrome/error events 更新 error rates、weights、priors 或 physical controls | 多数针对 binary stabilizer/surface/repetition codes；不直接处理 finite-energy GKP continuous residual、auxiliary/reset/leakage regime 与 GKP action contract |
| learned QEC | learned history、calibration conditioning、强 decoding accuracy | full neural inference 常承担 per-shot decoding；teacher、student、oracle 和 control policy 的角色/部署边界不总是分离 |
| FPGA/real-time QEC | 低 latency、high throughput、resource/closed-loop evidence | 多数为 binary syndrome decoder；不回答 slow drift estimation 如何原子更新 continuous-GKP fast-path 参数，或错误更新如何安全 fallback |

据此，本项目的研究缺口应写为：

> **现有工作尚未在同一、可审计的实验设计中联合回答：对于受 finite-energy、measurement/reset/leakage 和 time-varying displacement/loss statistics 影响的 repeated single-mode GKP memory，是否能把 syndrome history 中的慢变量与事件状态压缩成一个低维、原子更新、具 fallback 的 deterministic fixed-point control surface；该 surface 是否在 held-out drift/regime 下可重复缩小 static-to-oracle logical gap；以及这一收益在 simulator fidelity、quantization、deadline、fault injection、真实 FPGA resource/latency 和 HIL 约束下能保留多少。**

这是一个架构—算法—证据联合缺口，不是“首次使用 CNN”“首次 analog GKP”“首次 FPGA QEC”的单点 novelty。

## 5. 可证伪研究问题与通过/降级条件

| RQ | 可证伪问题 | 必需证据 | 失败后的结论降级 |
| --- | --- | --- | --- |
| RQ1 | 非平稳 noise state 是否在 syndrome-only observation 下可识别到足以改善 logical action？ | identifiability、state-estimation error、held-out drift、truth-vs-estimate、logical metric | 若不可识别，只保留 static/protocol-aware MAP；不训练 CNN |
| RQ2 | history 是否提供超过 memoryless MF/static MAP 的因果收益？ | memoryless MF、Bayesian/history、NMF teacher、history-off ablation、paired seeds/CI | 若 history-off 无损，删除 NMF/history novelty |
| RQ3 | 低维 deterministic student 能否保留 teacher/control-oracle 的有用部分？ | teacher-to-student gain retention、state/action fidelity、OOD、safety envelope | 若蒸馏失败，回退 run-length-aware MAP-LUT，不把 teacher 写成 deployable |
| RQ4 | adaptive method 是否缩小 static MAP 到 decoder oracle MAP 的 gap？ | `P_L^static`, `P_L^adaptive`, `P_L^oracle` 与 `G_oracle`，同 seed/同 trace | 若无正向且稳定 gap closure，取消 drift-adaptive performance claim |
| RQ5 | finite-energy/protocol effects 是否改变算法排序？ | ideal、effective finite-energy、noise-transfer、Fock-space/sBs fidelity cross-check | 若只在 ideal syndrome model 有效，限定为 idealized study |
| RQ6 | quantization/deadline/fallback 后是否仍保留收益？ | float/fixed bit parity、precision-resource-performance Pareto、fault injection、HIL | 若硬件约束吞噬收益，只保留 software method，不写 FPGA-ready |
| RQ7 | 廉价 FPGA 上是否有真实数字控制平面证据？ | board model/provenance、post-route resource/Fmax、core/transport/end-to-end latency、HIL logs | 若缺测，只能写 hardware-aware simulation/synthesis estimate |

## 6. Direct baseline contract

主要比较不能只放 standard binning，也不能让 proposed method 与弱 baseline 比较。

| 层级 | 必须进入的 baseline / bound | 作用 |
| --- | --- | --- |
| GKP hard/soft | standard binning、static periodic MAP、finite-energy/protocol-aware static optimized decoder、top-K lattice-coset MAP | 分离 hard-decision 损失、known-noise soft gain、近似 posterior 成本 |
| 不可部署 decoder bound | time-dependent oracle MAP（读取真实 `DriftState`） | 定义 static-to-oracle gap；不作为实际算法 |
| history/adaptive | memoryless MF、Wan-style Bayesian/history baseline、GP/sliding-window、EWMA、Kalman/UKF、HMM/change-point、run-length MAP | 覆盖慢漂移、频率变化和离散事件；Kalman/EWMA 标为工程 baseline |
| learned/control | CNN-only/full learned comparator、Feedback-GRAPE/NMF recurrent teacher、finite-horizon control oracle、distilled student、run-length MAP-LUT fallback | 分离表示能力、teacher upper target、压缩损失与可部署主线 |
| recovery bound | QEC-matrix/Petz/transpose-channel bound | 衡量编码—噪声可恢复空间；不混为 decoder oracle |
| hardware | float、fixed-point software、RTL simulation、post-route、board core、transport、HIL | 分离算法、量化、实现和系统集成损失 |

公平性要求：相同 trace、paired seeds、相同 observation/action availability、相同 causal information、相同 stopping rule；oracle/bound 可使用额外信息，但必须单列为不可部署上界。

## 7. Claim ladder 与允许措辞

| 证据层级 | 允许表述 | 禁止表述 |
| --- | --- | --- |
| ideal/effective simulation | `simulation-derived`, `under the specified syndrome/effective model` | realistic physical GKP experiment |
| finite-energy 多保真交叉验证 | `finite-energy-aware`, `protocol-aligned`, `trend-consistent across fidelities` | exact cavity–transmon dynamics, experimentally validated |
| fixed-point/synthesis/post-route | `hardware-aware`, `fixed-point/post-route estimate` | measured FPGA latency/resource（若未上板） |
| 廉价板卡数字控制平面 | `measured board-level digital control-plane latency/resource`, `replay/HIL` | microwave/ADC/cavity/transmon closed loop |
| 真实量子硬件（可选） | 只报告实际接入、标定和测得指标 | 用计划、论文参数或 HIL 替代真实实验 |

论文贡献只允许表述为在相应证据层内缩小 static MAP 与 oracle MAP 的 gap、保留 deterministic fast path、并量化部署损失；不得声称超过 oracle MAP，不使用 single-mode surface-code threshold 语言。

## 8. English Introduction draft

### Paragraph 1 — field stake and physical setting

The Gottesman–Kitaev–Preskill (GKP) code protects oscillator-encoded quantum information against small phase-space displacements by converting continuous errors into modular syndromes [@gottesman_encoding_2001]. Practical GKP states, however, have finite energy and therefore inherit non-ideal peak widths, envelopes, loss sensitivity and ancilla-induced error propagation [@glancy_error_2006; @grimsmo_quantum_2021; @hastrup_analysis_2023; @jafarzadeh_logical_2025]. Experiments in superconducting cavities have progressed from repeated stabilization of grid states to autonomous and real-time error correction beyond break-even [Campagne-Ibarcq 2020, DOI:10.1038/s41586-020-2603-3; @lachance-quirion_autonomous_2024; @sivak_real-time_2023]. These advances make the classical processing of repeated GKP syndrome records a systems problem as well as a decoding problem: a useful controller must respond to imperfect measurements and evolving device conditions without violating the timing and safety constraints of the feedback loop.

### Paragraph 2 — GKP decoding prior art

Continuous GKP outcomes already provide information beyond hard nearest-cell decisions. Analog likelihoods have been incorporated into surface-GKP and QLDPC-GKP decoders, yielding dynamic matching weights or message-passing inputs [@fukui_high-threshold_2018; @noh_fault-tolerant_2020; @noh_low_2022; @raveendran_finite_2022; @berent_analog_2024; @borah_fault_2025]. Multi-round Bayesian inference can further exploit syndrome history for approximate GKP states [@wan_memory-assisted_2020], and neural decoders have also been studied for bosonic or GKP-concatenated codes [@wang_multidimensional_2022]. These methods establish the value of soft and temporal information, but they mainly target known or stationary noise models, specific outer-code decoders, or computational procedures whose update and failure semantics are not defined as a low-dimensional real-time control interface.

### Paragraph 3 — adaptation and deployment prior art

In parallel, quantum-error-correction studies have shown that syndrome data can support time-dependent error-rate prediction, adaptive decoder weights, noise-parameter estimation, decoding-graph reweighting and direct optimization of decoder priors [Huo 2017, arXiv:1710.03636; @spitz_adaptive_2018; @wagner_optimal_2021; @wang_dgr_2023; @chen_calibrated_2022; @sivak_optimization_2024; Bhardwaj 2025, arXiv:2511.09491]. Learned decoders conditioned on calibration records and reinforcement-learning controllers extend this idea to unseen calibration states and continuous control adaptation [@stein_calibration-conditioned_2026; @sivak_reinforcement_2026]. Meanwhile, FPGA decoders based on lookup tables, clustering, belief propagation and quantized neural networks have reached sub-microsecond processing and, in recent demonstrations, real-time feedback on superconducting processors [@das_lilliput_2021; @barber_real-time_2023; @maurer_real-time_2025; @caune_demonstrating_2024; @yang_real-time_2026]. The adaptation and hardware literatures nevertheless remain largely centred on discrete stabilizer syndromes, while the finite-energy GKP literature rarely couples non-stationary estimation to an atomic, bounded and fail-safe parameter-update contract.

### Paragraph 4 — exact gap

The unresolved question is therefore not whether GKP syndromes can be decoded softly, whether noise can drift, or whether neural and FPGA decoders can operate at low latency. It is whether repeated finite-energy GKP error correction under time-varying displacement, loss, readout, reset and leakage conditions can be organized into a causal dual-loop architecture in which a slow estimator or model-aware teacher compresses syndrome history into bounded state and parameter updates, while a deterministic fixed-point student executes the cycle-critical action. A convincing answer requires more than an average accuracy gain: it must quantify the gap between static and time-dependent oracle decoding, separate decoder and control oracles from channel-recovery bounds, test held-out drift and regime changes, and trace any gain through model fidelity, quantization, deadline violations, fallbacks and board-level execution.

### Paragraph 5 — present study and boundary

Here we develop an experiment-informed framework to test this hypothesis for repeated single-mode square approximate-GKP memory correction. The framework separates two computational domains and three timescales: a cycle-level deterministic fast path, a window-level event and health monitor, and a host-level estimator or recurrent teacher. The teacher is not placed in the real-time critical path; instead, it provides a target for a bounded fixed-point student or, if distillation fails, a run-length-aware MAP–LUT fallback. We evaluate the resulting policies against standard binning, static and finite-energy-aware MAP decoders, Bayesian and window-based adaptive baselines, a time-dependent decoder oracle, a finite-horizon control oracle and a channel-recovery bound. Claims are tied to an explicit evidence ladder spanning multi-fidelity simulation, fixed-point and RTL equivalence, fault injection, post-route estimates, low-cost-board measurements and hardware-in-the-loop replay. Until physical GKP hardware is connected, the scope remains protocol-aligned and hardware-aware rather than a claim of real-time quantum-hardware error correction.

## 9. Claim–evidence 对应与引用支撑等级

| Segment | Claim | 主要证据 | 支撑等级 | 写作限制 |
| --- | --- | --- | --- | --- |
| S001 | practical GKP requires finite-energy/noise-aware treatment | GKP/Glancy/Hastrup/Jafarzadeh | strong | 不能由理想 syndrome model 自动继承 |
| S002 | real GKP QEC and feedback exist | Campagne-Ibarcq/Lachance-Quirion/Sivak | strong | 外部实验，不是本项目结果 |
| S003 | analog and history-aware GKP decoding exist | Fukui/Noh/Wan/Berent/Borah | strong | 禁止 novelty overclaim |
| S004 | syndrome-driven adaptation/calibration exists | Huo/Spitz/Wagner/DGR/Chen/Sivak/Bhardwaj | strong/partial by code family | 多数为 stabilizer/surface/repetition code |
| S005 | FPGA and real-time QPU feedback decoders exist | LILLIPUT/Collision/Caune/Yang | strong | 必须区分 `I` 与 `E` 证据层 |
| S006 | combined finite-energy GKP + drift + distilled deterministic student + full deployment gate is unresolved | 四线 collision analysis | inferred gap | 只能写“尚未在本矩阵发现/联合评估”，不写“从未有人做过” |
| S007 | proposed architecture can close static-to-oracle gap | 后续 Phase 1–6 结果 | needs evidence | 当前只能作为 RQ，不写成已证明 |
| S008 | FPGA-ready/real-time performance | 后续 post-route/board/HIL | needs evidence | 未实测前只写 hardware-aware estimate |

## 10. 结构选择与缺失输入

- Introduction 采用 `general-to-specific + open-with-challenge`：先建立 finite-energy GKP/实验背景，再交汇 decoding、adaptation、hardware 三条线，最后提出联合缺口。
- Related Work 折入 Introduction 作为四个 mechanism groups；后续若投 CS/ML venue，可把第 3 节扩成独立 Related Work。
- 英文稿不写结果数字，因为 T0.2.2 只冻结 gap，Phase 1–6 尚未提供新任务序列的实验证据。
- 当前未指定目标期刊、字数和 reference style；T7.2.1 起草正式 Introduction 时需按 venue 重排和压缩。
- `MISS` 文献使用作者+arXiv ID，不伪造 citation key；正式投稿前需补 Zotero 和稳定参考文献条目。

