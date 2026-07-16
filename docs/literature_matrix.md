# GKP 漂移自适应解码四线文献矩阵

**任务：** T0.2.1  
**冻结日期：** 2026-07-14  
**用途：** 为后续 gap statement、baseline 选择、有限能量模型、控制器/decoder 术语和 FPGA 证据门提供可追溯文献底座。

## 1. 范围、证据等级与判定规则

本矩阵覆盖四条直接服务当前主线的文献线：

1. GKP 基础、有限能量与实验；
2. GKP 本体、analog/soft-information 与 history-aware 解码；
3. 自适应/非平稳噪声估计与 decoder calibration；
4. 神经网络 QEC、FPGA 和真实实时解码。

本文件不把综述中的二手数字当作原始实验结果，也不把“算法可在线运行”写成“已在 FPGA/量子闭环实测”。字段约定如下。

| 字段 | 等级 | 含义 |
| --- | --- | --- |
| 证据 | `A` | 本地 PDF/全文卡已核到图、表、公式或正文数值；正式投稿仍需固定页码 |
| 证据 | `B` | 本轮由 arXiv/出版社一级来源核到摘要、正文或公开全文 |
| 证据 | `C` | 仅元数据/摘要索引，不用于强数值 claim |
| finite-energy | `D` | 直接建模 approximate/finite-energy GKP、finite squeezing/envelope 或其逻辑通道 |
| finite-energy | `P` | 与实验 GKP、loss、辅助态或 finite squeezing 有关，但不是完整有限能量 decoder |
| finite-energy | `N` | 不涉及；不能据此支撑本项目有限能量 claim |
| 实时硬件 | `E` | 在真实量子处理器/闭环中执行并给出端到端或 feedback 时序 |
| 实时硬件 | `I` | FPGA/ASIC/kernel 或板上 decoder 实测，但没有同等级量子闭环证据 |
| 实时硬件 | `L` | 软件/GPU/综合/理论 latency，或只说明可在线，不是 FPGA/闭环实测 |
| 实时硬件 | `N` | 没有可引用的实时硬件证据 |
| Zotero | `Z38` | 已在 `zotero_export_for_literature_review.bib` 的 38 条快照中，括号内为 canonical BibTeX key |
| Zotero | `ZAPI` | 本轮由 Zotero local API 查到，但不在 38 条导出快照中；括号内为 item key |
| Zotero | `MISS` | 本轮 Zotero 标题检索和 38 条导出均未覆盖；不伪造 item key |

去重主键按 `DOI > arXiv ID > 规范化题名+首作者+年份`。submission 中的短 key 只是 alias，不重复计数；附件目录的 8 位 storage key 不当作 Zotero item key。

## 2. 线 A：GKP 基础、有限能量与实验

| ID | 文献与主标识 | 噪声模型 / 方法 | 主要指标或证据入口 | finite-energy | 实时硬件 | Zotero | 证据 | 对本项目的作用与边界 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A01 | Gottesman, Kitaev & Preskill, *Encoding a qubit in an oscillator* (2001), DOI `10.1103/PhysRevA.64.012310` | 连续位移误差；相空间格点 stabilizer、modular syndrome 与 nearest-lattice recovery | Fig. 1/2；correctable shift 与 approximate codeword 方程 | D | N | Z38 (`gottesman_encoding_2001`), API `HAE5MSI3` | A | 定义 square-GKP 格距、逻辑陪集和 wrapping；不支持 drift、CNN 或硬件 claim |
| A02 | Glancy & Knill, *Error Analysis For Encoding A Qubit In An Oscillator* (2006), DOI `10.1103/PhysRevA.73.012325` | data/ancilla shift errors；近似 GKP 纠错电路与误差传播 | Fig. 2/3/6/7；no-error probability、photon-number/large-shift trade-off | D | N | Z38 (`glancy_error_2006`), API `WBJPZ396` | A | 约束 finite squeezing、ancilla error 和 correctable region；不能被 syndrome-only demo 替代 |
| A03 | Campagne-Ibarcq et al., *Quantum error correction of a qubit encoded in grid states of an oscillator* (2020), DOI `10.1038/s41586-020-2603-3` | superconducting cavity 中有限能量 grid states；measurement-feedback、sharpen/trim | 实验 logical lifetime、syndrome/feedback cycle 与 cavity–transmon 链路 | P | E | ZAPI (`SKTLAMK7`) | A | 平台与协议交叉验证锚点；本项目无同等级量子硬件，不得写成实验复现 |
| A04 | Grimsmo & Puri, *Quantum Error Correction with the GKP Code* (2021), DOI `10.1103/PRXQuantum.2.020101` | GKP lattice、approximate states、loss/dephasing 与实验路线综述 | Fig. 1–3；modular squeezing 定义 | D | N | Z38 (`grimsmo_quantum_2021`) | A | 统一术语与有限能量背景；数值性能应回原始论文，不从综述二次外推 |
| A05 | Hastrup et al., *Analysis of loss correction with the GKP code* (2023), DOI `10.1103/PhysRevA.108.052413` | pure loss、pre-amplification、finite-energy GKP recovery | Fig. 2/3；logical-channel infidelity 随 loss 和 finite-energy 参数变化 | D | N | Z38 (`hastrup_analysis_2023`) | A | 区分真实 loss channel 与本项目 effective variance/loss proxy；禁止 capacity 外推 |
| A06 | Jafarzadeh et al., *Logical channels in approximate GKP error correction* (2025), DOI `10.1103/c8hk-v1qf` | approximate GKP teleportation/recovery 的 logical channel | logical Pauli probabilities、Wigner/marginal、logical maps | D | N | Z38 (`jafarzadeh_logical_2025`) | A | 支撑 `p_X/p_Z/p_Y`、PTM、`F_avg/F_e` 指标；不证明当前 `physics/` 已实现完整 logical channel |
| A07 | Lachance-Quirion et al., *Autonomous quantum error correction of Gottesman-Kitaev-Preskill states* (2024), DOI `10.1103/PhysRevLett.132.150607`, arXiv `2310.11400` | cavity+transmon+reservoir；sBs/autonomous correction、reset | Fig. 1–3；with/without optimized QEC logical lifetime | P | E | Z38 (`lachance-quirion_autonomous_2024`), API `MLF7ITGK` | A | sBs-first 数字孪生的实验结构锚点；本项目当前只可写 protocol-aligned simulation |
| A08 | Sivak et al., *Real-time quantum error correction beyond break-even* (2023), DOI `10.1038/s41586-023-05782-6` | superconducting bosonic logical qubit；实时 syndrome feedback | coherence gain `2.27±0.07`；system coherence、outcome characterization、timing table | P | E | Z38 (`sivak_real-time_2023`), API `W4DI4CSK` | A | 真实闭环/break-even 最高证据参照；`4.924 us` 是其装置时序，不是本项目板卡实测 |

## 3. 线 B：GKP 本体、analog/soft-information 与 history-aware 解码

| ID | 文献与主标识 | 噪声模型 / decoder 或方法 | 主要指标或证据入口 | finite-energy | 实时硬件 | Zotero | 证据 | 对本项目的作用与边界 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B01 | Fukui et al., *High-threshold fault-tolerant quantum computation with analog QEC* (2018), DOI `10.1103/PhysRevX.8.021054` | Gaussian shift；保留 GKP residual 构造 analog likelihood | threshold/squeezing 曲线；required squeezing 降至约 `9.8 dB` 的结果入口 | P | N | Z38 (`fukui_high-threshold_2018`), API `FTXC3FWQ` | A | 证明 analog GKP syndrome 不是本项目首创；外层码收益不能外推到 affine fast path |
| B02 | Noh & Chamberland, *Fault-tolerant bosonic QEC with the surface-GKP code* (2020), DOI `10.1103/PhysRevA.101.012316` | circuit/GKP noise；analog outcome-dependent matching weights | Fig. 6/Table 1 的 squeezing 和 component thresholds | D | N | Z38 (`noh_fault-tolerant_2020`), API `VJATVHYU` | A | soft GKP 到外码 decoder 的直接先例；本项目当前不实现 surface-GKP |
| B03 | Noh, Chamberland & Brandão, *Low overhead fault-tolerant QEC with the surface-GKP code* (2022), DOI `10.1103/PRXQuantum.3.010315` | teleportation GKP QEC、ML decoding、space-time correlated edges | `12 dB` CNOT failure 与 `9.9 dB` threshold；Table III resource comparison | D | N | Z38 (`noh_low_2022`), API `546HDPNZ` | A | 强 ML/soft baseline 和资源参照；不得把外码 low-overhead 结果写成本项目结果 |
| B04 | Raveendran et al., *Finite Rate QLDPC-GKP Coding Scheme that Surpasses the CSS Hamming Bound* (2022), DOI `10.22331/q-2022-02-10-642` | Gaussian shift；analog LLR + min-sum/message passing | Fig. 3/4 与 LLR 方程；阈值数值仍需固定 PDF 页码 | P | N | Z38 (`raveendran_finite_2022`), API `GHXLICF9` | A | 说明 soft GKP 也服务 QLDPC；本项目短期不声称 circuit-level QLDPC benchmark |
| B05 | Wan et al., *Memory-assisted decoder for approximate GKP codes* (2020), DOI `10.1103/PhysRevResearch.2.043280` | approximate GKP + Gaussian displacement；多轮 syndrome-history Bayesian estimation | memoryless vs assisted fidelity；`Delta≈0.22`, mean photon number约10 | D | N | Z38 (`wan_memory-assisted_2020`), API `T2PCPHZT` | A | 必须进入 history-aware GKP direct baseline；histogram window 不等价于完整 Bayesian decoder |
| B06 | Berent et al., *Analog information decoding of bosonic quantum LDPC codes* (2024), DOI `10.1103/PRXQuantum.5.020349` | bosonic readout analog values；outcome-dependent probabilities + Tanner/message passing | Fig. 1 与 full decoding curves | P | N | Z38 (`berent_analog_2024`), API `AAIKJVZS` | A | 强化 analog-information 成熟度；外码 message-passing 收益不能替代本项目实验 |
| B07 | Borah et al., *Fault Tolerant Decoding of QLDPC-GKP Codes with Circuit Level Soft Information* (2025), arXiv `2505.06385` | circuit-level noisy GKP measurements；precomputed/real-time soft information | no-soft、precomputed 和 real-time soft 三路线比较 | P | L | Z38 (`borah_fault_2025`) | A | 约束 circuit-level soft-info 层级；`real-time` 算法语义不是硬件闭环实测 |
| B08 | Roy et al., *Decoding Multimode GKP Codes with Noisy Auxiliary States* (2025), arXiv `2510.12677` | noisy auxiliary states 引入跨模相关；显式相关 decoder | logical-error improvement 至少一个数量级的摘要入口；Fig. 1/2 结构 | D | N | Z38 (`roy_decoding_2025`) | A | 要求模型区分 auxiliary/measurement noise；该多模收益不作为当前 single-mode 结果 |
| B09 | Wang et al., *Multidimensional Bose QEC based on neural network decoder* (2022), DOI `10.1088/2058-9565/ac82f9` | multidimensional GKP-like Gaussian/noisy measurement；NN decoder | NN/MWPM threshold curves，区分 clean/noisy measurements | P | L | Z38 (`wang_multidimensional_2022`) | A | 神经 GKP/bosonic decoding 已存在；不支持“首次 neural GKP decoder” |
| B10 | Zheng et al., *Performance and achievable rates of the GKP code for pure-loss and amplification channels* (2024), arXiv `2412.06715` | loss/amplification；transpose/Petz-like near-optimal recovery | Fig. 2 recovery comparison、rate/channel conditions | D | N | Z38 (`zheng_performance_2024`) | A | 为 T5.2 recovery bound 提供理论参照；bound/oracle 不能混称为可部署 decoder |

## 4. 线 C：自适应/非平稳噪声估计与 decoder calibration

| ID | 文献与主标识 | 噪声模型 / estimator、decoder 或 controller | 主要指标或证据入口 | finite-energy | 实时硬件 | Zotero | 证据 | 对本项目的作用与边界 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C01 | Huo & Li, *Learning time-dependent noise to reduce logical errors* (2017), DOI `10.1088/1367-2630/aa916e`, arXiv `1710.03636` | time-dependent error rates；Gaussian-process tracking/prediction 后更新 decoder | error-rate tracking 和 logical-failure reduction；收益随 code distance 增强 | N | L | MISS | B | 预测型自适应强邻近 baseline；“real time”是方法语义，无 FPGA/QPU latency 证据 |
| C02 | Spitz et al., *Adaptive Weight Estimator for QEC in a Time-Dependent Environment* (2018), DOI `10.1002/qute.201800012` | 时变独立误差；历史 detection events + 有限窗口更新 MWPM weights | `Delta∝N^-1.2`；周期漂移 `N_opt≈1265` | N | L | Z38 (`spitz_adaptive_2018`), API `XC5AHR6P` | A | T1.3.4/T3.2.3 直接 adaptive-window baseline；不是 GKP physical-layer correction |
| C03 | Wagner et al., *Optimal noise estimation from syndrome statistics of quantum codes* (2021), DOI `10.1103/PhysRevResearch.3.013292` | Pauli-rate identifiability；EM/HEM syndrome-only estimation | `10^5–10^6` samples、约5次迭代、MSE 达 Cramér–Rao bound | N | N | Z38 (`wagner_optimal_2021`), API `TDUYKZBX` | A | 为 identifiability/sample efficiency 提供基准；本身不是 drift tracker |
| C04 | Chen et al., *Calibrated decoders for experimental QEC* (2022), DOI `10.1103/PhysRevLett.128.110504` | 真实 correlated events 至 size-4；calibrated graph/hyperedge decoder | 0–10 rounds；post-selection retained fraction 与 LER/round | N | L | Z38 (`chen_calibrated_2022`), API `VTD58CZ5` | A | 支撑 calibration 必要性；不等于本项目已有真实实验 calibration |
| C05 | Wang et al., *DGR: Tackling Drifted and Correlated Noise via Decoding Graph Re-weighting* (2023), arXiv `2311.16214` | surface/honeycomb drift + correlated 2Q noise；edge/edge-pair reweighting | average mismatch LER 改善 `3.6x`；极端收益只作该设定内证据 | N | N | Z38 (`wang_dgr_2023`), API `JL2PSXWM` | A | correlation-aware graph 强 baseline；巨大 graph mismatch 收益不得外推到 GKP affine fast path |
| C06 | Sivak, Newman & Klimov, *Optimization of decoder priors for accurate QEC* (2024), DOI `10.1103/PhysRevLett.133.150603` | 实验 prior mismatch；直接按 LER 优化 decoder prior | repetition 平均 `48%`/再改善 `16%`；surface-code memory约 `3.3%` | N | L | Z38 (`sivak_optimization_2024`), API `EUXMSLYN` | A | 最直接 prior-calibration 对照；物理参数 MSE 最优不保证 LER 最优 |
| C07 | Puviani et al., *Non-Markovian Feedback for Optimized Quantum Error Correction* (2025), DOI `10.1103/PhysRevLett.134.020601`, arXiv `2312.07391` | GKP measurement history；Feedback-GRAPE + recurrent controller 联合优化后续 unitary | standard、memoryless Markov feedback 与 full-history NMF 的性能排序 | P | L | ZAPI (`IJ5EDGZF`; BibTeX `puviani_non-markovian_2025`) | A | T2.3/T3.2 的 memory-specific teacher/baseline；RNN control oracle/teacher 不是实际 decoder 或 FPGA student |
| C08 | Bhardwaj et al., *Adaptive Estimation of Drifting Noise in QEC* (2025), arXiv `2511.09491` | time-dependent Pauli noise；optimal/overlapping sliding windows、多频 drift recovery | phenomenological+circuit-level tracking；estimated LER 对齐 truth，并优于 static model | N | N | MISS | B | 当前最直接 sliding-window drift estimator；需进入 T1.3.4，且不等同 Kalman |
| C09 | Stein et al., *Calibration-Conditioned FiLM Decoders...* (2026), arXiv `2601.16123` | IBM calibration graph；GCN/FiLM 调制 CNN，跨日期/链 generalization | 2.76M shots、`d<=11`；最高 `11.11x` LER reduction；folded GPU `85–95 us` | N | L | Z38 (`stein_calibration-conditioned_2026`) | A | 与慢校准+快推理结构最邻近；GPU/repetition-code 结果不能写成 FPGA-GKP 证据 |
| C10 | Sivak et al., *Reinforcement Learning Control of QEC* (2025 preprint), arXiv `2511.08493` | injected drift；error-detection events 同时驱动 QEC 与 RL physical-control steering | 真实处理器 stability `3.5x`；仿真到 `d=15` | N | E | Z38 (`sivak_reinforcement_2026`) | A | slow-loop controller/teacher 的实验上限参照；仓库年份按首投2025规范化，且不是 FPGA fast decoder |

## 5. 线 D：神经网络 QEC、FPGA 与真实实时解码

| ID | 文献与主标识 | 噪声模型 / decoder 或硬件方法 | 主要指标或证据入口 | finite-energy | 实时硬件 | Zotero | 证据 | 对本项目的作用与边界 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D01 | Das et al., *LILLIPUT* (2022；preprint 2021), DOI `10.1145/3503222.3507707`, arXiv `2108.06569` | surface-code syndrome；offline decoder 生成压缩 LUT，在线查表 | FPGA logic `<7%`、`28–42 ns`；CLUT `148 MB -> 1.38 MB` | N | I | Z38 (`das_lilliput_2021`), API `L9TW3WJZ` | A | parameter/LUT bank 的硬件强参照；不能借用其资源/latency 代替本项目测量 |
| D02 | Liyanage et al., *Scalable QEC for Surface Codes using FPGA* / Helios (2023), DOI `10.1109/FCCM57271.2023.00045`, arXiv `2301.08419` | phenomenological surface-code noise；distributed Union-Find/tree-grid | VCU129、`d=21`、`11.5 ns/measurement round` 的公开结果入口 | N | I | Z38 (`liyanage_scalable_2023`) | A | decoder throughput/backlog 标准；仍需固定论文表页后才写强 comparison 数字 |
| D03 | Barber et al., *A real-time, scalable, fast and resource-efficient decoder* / Collision Clustering (2025；preprint 2023), DOI `10.1038/s41928-024-01319-5`, arXiv `2309.05558` | surface-code defects；collision clustering hardware decoder | FPGA `881` qubits/`810 ns`/`10 KB`；ASIC `240 ns`/area/power | N | I | Z38 (`barber_real-time_2023`) | A | 强制报告 scale、latency、memory、area/power；不是 GKP analog decoder |
| D04 | Ziad et al., *Local Clustering Decoder* (2025；preprint 2024), DOI `10.1038/s41467-025-66773-x`, arXiv `2411.10343` | surface-code + leakage-dominant model；local/adaptive clustering | `<1 us/round`；VU19P `d=17` 约6% LUT/3% FF | N | I | Z38 (`ziad_local_2024`) | A | local fast path + adaptive policy 的系统参照；binary syndrome 与 GKP analog 不可混写，adaptivity-engine latency 尚未闭合 |
| D05 | Caune et al., *Demonstrating real-time and low-latency QEC with superconducting qubits* (2024), arXiv `2410.05202` | 真实 superconducting stability experiment；FPGA Collision decoder | 5–25 rounds约 `0.44–0.79 us/round`；9-round response `9.6 us` | N | E | Z38 (`caune_demonstrating_2024`), API `VESKCXZI` | A | 真实 decoder+control 通信闭环标准；software-HIL 不能称同等级 demo |
| D06 | Bausch et al., *Learning high-accuracy error decoding for quantum processors* (2024), DOI `10.1038/s41586-024-08148-8` | real+synthetic surface-code data；AlphaQubit recurrent/attention decoder | 325k experimental shots；Sycamore `d=3/5` LER 与 MWPM-Corr 比较 | N | L | Z38 (`bausch_learning_2024`), API `8WHC6P46` | A | learned decoder 的数据规模、split、强 baseline 标准；没有本项目同等级实机证据 |
| D07 | Maurer et al., *Real-time decoding of the gross code memory with FPGAs* (2025), arXiv `2510.21600` | gross/bivariate-bicycle code；FPGA belief propagation | BP iteration `24 ns`、window `<240 ns`、cycle `<1 us`；resource breakdown | N | I | Z38 (`maurer_real-time_2025`), API `74BVAIMU` | A | fixed-point/resource/LER 共同报告的强标准；不是 GKP/surface decoder |
| D08 | Chamberland et al., *Fast and accurate AI-based pre-decoders for surface codes* (2026), arXiv `2604.12841` | surface-code；AI pre-decoder + global decoder | `d=31,p=0.006` 子步骤 `270.83 -> 38.78 us`，pipeline `3.54x` | N | L | Z38 (`chamberland_fast_2026`) | A | accuracy-runtime-deployment role 写作范式；软件/GPU数值不可替代 FPGA core timing |
| D09 | Peled et al., *Neural Minimum Weight Perfect Matching for Quantum Error Codes* (2026), arXiv `2601.00242` | surface/toric noise；GNN/Transformer 学 MWPM weights | independent/depolarizing thresholds `10.95%/17.9%` | N | L | Z38 (`peled_neural_2026`) | A | structured hybrid decoder 先例；不是 GKP affine correction |
| D10 | Hanisch et al., *Soft information decoding with superconducting qubits* (2026), DOI `10.1103/y9fh-4x6n`, arXiv `2411.16228` | 真实 superconducting repetition-code soft readout；soft decoder | threshold 平均改善 `24.4%`；最高 `30x` error reduction；1 byte/measurement | N | L | Z38 (`hanisch_soft_2026`) | A | 低比特 soft feature/ADC 设计参照；不是 GKP drift benchmark 或 FPGA闭环 |
| D11 | Yang et al., *Real-time Surface-Code Error Correction Using an FPGA-based Neural-Network Decoder* (2026), arXiv `2605.04892` | `d=3` 真实 superconducting surface code；6-bit recurrent NN on FPGA | deterministic closed-loop `550 ns`，NN `124 ns`，QEC cycle `1.25 us` | N | E | Z38 (`yang_real-time_2026`), API `B235MJY8` | A | neural+FPGA+QPU 的最近邻强证据；本项目 fast path 是 deterministic student，当前无此实测层级 |
| D12 | Yan, Li & Du, *Rethink the Role of Neural Decoders in QEC* (2026), arXiv `2605.12046` | surface-code neural decoder 五类架构；compression/INT4 FPGA deployability study | 到 `d=9`；accuracy-latency sweep；INT4 达微秒级所需条件 | N | L | MISS | B | 要求把 quantization、architecture bias、data scale 与部署一起评估；不是实机闭环结果 |

## 6. 覆盖统计、去重与直接结论

| 项目 | 结果 |
| --- | --- |
| 纳入条目 | 40 篇：A 线 8、B 线 10、C 线 10、D 线 12 |
| finite-energy | `D=10`、`P=9`、`N=21`；仅 `D/P` 可用于有限能量相关陈述 |
| 实时硬件 | `E=6`、`I=5`、`L=13`、`N=16`；只有 `E` 可写真实量子闭环 |
| Zotero | `Z38=35`、`ZAPI-only=1`、`MISS=4`；`MISS` 为 Huo 2017、Puviani 2023、Bhardwaj 2025、Yan 2026 |
| 已识别 alias | submission 的 `spitz2018/wagner2021/dgr2023/chen2022/sivak2024/stein2026/sivak2026/wan2020` 均映射到长 canonical key，不重复计数 |
| 不存在的直接先例 | 本轮未找到“Kalman/EWMA 直接作为 GKP syndrome drift decoder 且有真实 FPGA闭环”的一级文献；它们应标成经典工程 baseline，而不是伪装成 QEC 文献先例 |

四线交叉后的保守定位是：analog GKP syndrome、finite-energy GKP、history-aware Bayesian recovery、decoder prior calibration、learned QEC decoder 和 FPGA real-time decoder 都已有直接先例。当前项目可争取的不是这些概念的“首次”，而是 **experiment-informed、finite-energy-aware、drift-adaptive 的 GKP classical-control architecture：慢回路从 syndrome history 估计连续漂移/离散健康状态，快回路使用可审计的确定性低维定点 student，并用 static MAP、memory-assisted/Bayesian、sliding-window/GP、oracle MAP、learned decoder 与真实硬件证据门共同约束 claim**。

## 7. Zotero 缺口与后续动作

T0.2.1 初始只做只读覆盖审计；后续任务在其明确证据门内补录时，须保存 DOI/arXiv、
Zotero item key 与稳定 BibTeX key。下列状态不会阻塞矩阵完成，但会影响投稿引用完整性：

| 优先级 | 缺口 | 后续动作 | 阻塞性 |
| --- | --- | --- | --- |
| Resolved 2026-07-14 | Puviani `2312.07391` 已入 Zotero `IJ5EDGZF` | 已固定 PRL 134, 020601 (2025)、DOI 与 BibTeX key `puviani_non-markovian_2025` | T2.3.7 文献前置门已关闭；强数值仍必须由本项目独立实验支撑 |
| High | Bhardwaj `2511.09491` 未入 Zotero | 在 T1.3.4 前补录，作为 sliding-window direct baseline | 不阻塞矩阵；会削弱漂移 baseline 完整性 |
| Medium | Huo `1710.03636` 未入 Zotero | 补录 DOI/arXiv，作为 GP prediction baseline | 不阻塞；由 T1.3.4 覆盖 |
| Medium | Yan `2605.12046` 未入 Zotero | 在 T5.5/T6 quantization 对比前补录 | 不阻塞；属于2026部署补充线 |
| Low | 38 条导出与 live API 覆盖不完全同构 | 后续生成带 item key 的稳定快照，避免只凭 attachment storage key | 不阻塞当前只读审计 |
