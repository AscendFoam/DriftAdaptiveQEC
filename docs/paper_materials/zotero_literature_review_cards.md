# Zotero 文献阅读摘要卡片：图表指标补强版

**修订日期**：2026-07-02  
**输入快照**：`docs/paper_materials/zotero_export_for_literature_review.bib`，共 38 条 Zotero BibTeX 条目。  
**用途**：为 `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex` 后续改写、benchmark 设计、baseline 选择和 claim/evidence 审计提供文献索引。  
**本轮修订重点**：修正旧版“摘要式总结过多、逐篇图表/数值证据不足”的问题；每篇文献都补上图、表、公式或正文数值层面的证据入口，并显式标明核验等级。

## 0. 证据等级与边界

本文件不是全文翻译，也不是最终 Related Work。它是投稿准备阶段的“文献证据卡片”。本轮按以下等级标注每篇文献：

| 等级 | 含义 | 可否直接写入论文强 claim |
| --- | --- | --- |
| `PDF图表/表格已核验` | 本地 Zotero PDF 或本地已有论文渲染页中核到图、表、公式或正文数值 | 可作为论文草稿候选，但正式投稿前仍建议回原 PDF 页码复核 |
| `公开PDF/HTML核验` | 通过 arXiv PDF、ar5iv/arXiv HTML、publisher HTML 或其他公开全文页面核到图、表、公式或正文数值 | 可作为相关工作候选，正式引用前需固定 PDF/DOI 页码 |
| `摘要/元数据级` | 仅从 arXiv/DOI 摘要、题名页、Zotero 元数据或公开 abstract 得到指标 | 只能作索引和待复核提示，不应直接写入强 claim |

本轮确实做了本地 PDF 批量抽取：对 BibTeX 中带 PDF 附件的 22 篇论文运行 `pdftotext/pdfinfo`，抽取 figure/table caption 和含指标文本；并对 surface-GKP 与 FPGA NN decoder 的关键结果页做了图像级抽查。对于 BibTeX 没有本地 PDF 的条目，则优先使用 arXiv/ar5iv/HTML 全文入口；只有仍未定位全文图表或正文证据的条目才保留为摘要/元数据级。

**覆盖检查**：38/38 条目均已覆盖；没有保留“本轮未核验”的空卡片。

## 1. 分类总览

| 类别 | 文献 | 对本项目的主要作用 |
| --- | --- | --- |
| GKP 基础与近似 GKP | GKP 2001；Glancy--Knill 2006；Grimsmo--Puri 2021；Jafarzadeh 2025 | 定义 GKP code、finite-energy approximation、correctable shift、logical channel 与 finite squeezing 边界 |
| GKP analog / soft information | Fukui 2018；Noh 2020/2022；Raveendran 2022；Berent 2024；Borah 2025；Roy 2025 | 说明 analog/soft information 已是成熟路线，本项目不能宣称“首次使用 analog GKP syndrome” |
| loss / memory / teleportation / performance analysis | Hastrup 2023；Wan 2020；Walshe 2020；Marqversen 2025；Zheng 2024；Lachance-Quirion 2024 | 补充 approximate GKP、loss、memory、teleportation 和实验 GKP QEC 背景 |
| adaptive / calibration decoders | Spitz 2018；Wagner 2021；DGR 2023；Chen 2022；Sivak 2024 | 支撑 syndrome statistics、decoder prior、drifted/correlated-noise calibration 已有强相关工作 |
| learned / AI QEC decoders | Bausch 2024；Chamberland 2026；Stein 2026；Peled 2026；Wang 2022；Sivak 2026；Hanisch 2026 | 提供 learned decoder、calibration-conditioned decoder、soft-information decoder 的强对照 |
| FPGA / real-time QEC | LILLIPUT；Helios；Collision decoder；qLDPC FPGA；Local Clustering；gross-code FPGA；Caune 2024；Yang 2026；Sivak 2023 | 规定硬件论文必须报告 latency、resource、throughput、closed-loop 与 hardware provenance；本项目当前不能外推为 real-board result |

## 2. 逐篇覆盖矩阵

| BibTeX key | 图表/数值证据入口 | 核验等级 |
| --- | --- | --- |
| `gottesman_encoding_2001` | Fig. 1/2：格点 codeword 与 finite-squeezed approximate codeword；方程链给出 correctable shift / lattice syndrome 基础 | `PDF图表/表格已核验` |
| `glancy_error_2006` | Fig. 1/2/3/6/7：approximate qubit、shift-correction circuit、误差概率与 photon number trade-off | `PDF图表/表格已核验` |
| `grimsmo_quantum_2021` | Fig. 1/2/3：square/hexagonal GKP lattices、loss/dephasing、measurement robustness；式 (13) 给 modular squeezing dB | `PDF图表/表格已核验` |
| `jafarzadeh_logical_2025` | Fig. 1/2/3/4：teleportation logical maps、Wigner functions、twirl-aware recovery；正文给出 GRN logical-channel probabilities | `PDF图表/表格已核验` |
| `fukui_high-threshold_2018` | Fig. 1-3 与 Eq. (7)-(9)：analog likelihood；Fig. 2/3 surface-code logical error curves；正文给出 16.0 dB -> 9.8 dB | `公开PDF/HTML核验` |
| `noh_fault-tolerant_2020` | Fig. 6 / Table 1 相关阈值：约 11.2 dB、18.6 dB、0.69%、0.81% | `公开PDF/HTML核验` |
| `noh_low_2022` | Fig. 6/7 + Table III：0.87% -> 0.36% CNOT failure；9.9 dB threshold；291 modes / 97 qubits vs 1457 qubits | `PDF图表/表格已核验` |
| `raveendran_finite_2022` | Fig. 3/4 与 LLR/min-sum decoder 方程；阈值数字仍需 PDF 固定 | `公开PDF/HTML核验` |
| `berent_analog_2024` | Fig. 1 展示 analog-valued syndrome / Tanner graph decoding；30 页 15 图 | `PDF图表/表格已核验` |
| `borah_fault_2025` | 5 图；比较 no soft info / precomputed probabilities / real-time soft information | `PDF图表/表格已核验` |
| `roy_decoding_2025` | Fig. 1/2：multimode GKP Voronoi cell 与 Steane-type QEC；摘要报告 logical error probability 至少降一个数量级 | `PDF图表/表格已核验` |
| `hastrup_analysis_2023` | Fig. 2/3：loss、pre-amplification 与 logical channel infidelity | `PDF图表/表格已核验` |
| `wan_memory-assisted_2020` | Fig. 1/2/3 + Eq. (15)：multi-round Bayesian memory、\(\Delta\approx0.22\)、\(\bar n\approx10\)、expected displacement bound | `PDF图表/表格已核验` |
| `walshe_continuous-variable_2020` | Fig. 1/2：CV macronode wire、gate teleportation；Kraus-operator 公式 | `PDF图表/表格已核验` |
| `marqversen_performance_2025` | Fig. 1-5：Knill/Steane GKP correction、Bell-state approximation、maximum error \(10^{-8}\) comparison | `PDF图表/表格已核验` |
| `zheng_performance_2024` | Fig. 2：loss/amplification 下 near-optimal recovery 与既有 decoder 对比；\(|\tau/(1-\tau)|\) 条件 | `PDF图表/表格已核验` |
| `lachance-quirion_autonomous_2024` | Fig. 1-3：hardware architecture、reset、sBs autonomous QEC；Fig. 3(c,d) logical lifetime with/without optimized QEC | `公开PDF/HTML核验` |
| `spitz_adaptive_2018` | Fig. 3/4：\(\Delta\propto N^{-1.2}\)，漂移窗口 \(N_{\rm opt}\approx1265\) | `公开PDF/HTML核验` |
| `wagner_optimal_2021` | Fig. 2/3：EM/HEM，\(10^5-10^6\) samples，\(10^8\) perfect-knowledge baseline，Cramer-Rao bound | `公开PDF/HTML核验` |
| `wang_dgr_2023` | Fig. 3：硬件噪声 drift；DGR average-case 3.6x、worst-case 最高 7360x | `公开PDF/HTML核验` |
| `chen_calibrated_2022` | Fig. 2/3：calibrated hyperedge probabilities、0-10 stabilizer rounds；14 figures / 5 tables；正文给出 size-4 correlated events 与 LER | `公开PDF/HTML核验` |
| `sivak_optimization_2024` | Fig. 3/4 + Table I：48%、16%、3.3% prior-calibration gains；53 surface-code experiments | `PDF图表/表格已核验` |
| `bausch_learning_2024` | Fig. 3/4：Sycamore d=3/d=5 LER；Pauli+ d=11 scaling；325k experimental samples | `PDF图表/表格已核验` |
| `chamberland_fast_2026` | Table IX/X：d=31、p=0.006 下 270.83 -> 38.78 us，7.0x sub-step speedup，3.54x total speedup | `公开PDF/HTML核验` |
| `stein_calibration-conditioned_2026` | Fig. 6/7 + Table I/V/VI：2,760,704 shots、400 calibration snapshots、11.11x LER reduction、~85-95 us folded FiLM latency | `PDF图表/表格已核验` |
| `peled_neural_2026` | Fig. 2-5：NMWPM threshold 10.95% / 17.9%，对照 MWPM/BPOSD/QECCT | `PDF图表/表格已核验` |
| `wang_multidimensional_2022` | Fig. 6/7：MWPM threshold \(\sigma\approx0.50/0.25\)，NN threshold \(\sigma\approx0.78/0.34\) | `PDF图表/表格已核验` |
| `sivak_reinforcement_2026` | Fig. 3-5：20% LER suppression、31% LER reduction、3.5x stability、d=15 / 40,000 parameters | `PDF图表/表格已核验` |
| `hanisch_soft_2026` | Fig. 7/8 + Table I：soft decoding threshold +24.4%，最高 30x lower error，1 byte/measurement | `PDF图表/表格已核验` |
| `das_lilliput_2021` | Fig. 1/3/13/16/17 + Table 1-6：d=3/4/5 layouts、<7% logic、28-42 ns latency、148 MB -> 1.38 MB CLUT | `公开PDF/HTML核验` |
| `liyanage_scalable_2023` | Helios VCU129；d=21；0.1% phenomenological noise；11.5 ns/measurement round；PDF/HTML入口已定位 | `公开PDF/HTML核验` |
| `barber_real-time_2023` | Collision decoder：881/1057 qubits，810 ns / 240 ns，0.06 mm²，8 mW | `公开PDF/HTML核验` |
| `maurya_fpga-tailored_2025` | Fig. 2：logical error rate vs FPGA cycle budget；Relay 约 1-2 orders faster | `公开PDF/HTML核验` |
| `ziad_local_2024` | Fig. 3：under 1 us/round；d=17 VU19P 约 6% LUT / 3% FF；4x fewer physical qubits | `公开PDF/HTML核验` |
| `maurer_real-time_2025` | Table 1：BP iteration 24 ns；<240 ns window；<1 us/cycle；97% LUT breakdown | `公开PDF/HTML核验` |
| `caune_demonstrating_2024` | Fig. 3/4：8-qubit stability；25 rounds；0.44-0.79 us/round；9-round response 9.6 us | `PDF图表/表格已核验` |
| `yang_real-time_2026` | Fig. 1/2/3：17 qubits、550 ns closed-loop、124 ns NN decoding、1.25 us QEC cycle、6-bit weights | `PDF图表/表格已核验` |
| `sivak_real-time_2023` | Fig. 3 system coherence；Fig. 4 syndrome/outcome characterization；Supp. Table S3 timing；coherence gain \(G=2.27\pm0.07\) | `公开PDF/HTML核验` |

## 3. GKP 基础与近似 GKP

### `gottesman_encoding_2001` - Encoding a qubit in an oscillator

- 原理：把 qubit 编码到振子的连续变量相空间格点中，用周期性 position/momentum codeword 抵抗小位移误差。
- 图表/数值证据：Fig. 1 展示 \(n=2\) code 的 codewords；Fig. 2 展示 approximate codeword 的 position-space probability distribution，例子中 \(\Delta=\kappa=.25\)。方程链给出格点位移、stabilizer 与 correctable shift 条件。
- 对本项目的意义：`physics/` 中 syndrome wrapping、nearest-lattice / modular residual、logical boundary 的理论根基必须回到这里。
- 写作边界：可作为 GKP definition 的基础文献；不要用它证明本项目的漂移模型或 FPGA runtime。

### `glancy_error_2006` - Error Analysis For Encoding A Qubit In An Oscillator

- 原理：分析 approximate GKP qubit 与 ancilla 都含 shift error 时，error-correction circuit 如何传播错误。
- 图表/数值证据：Fig. 1 给出 approximate qubit states，示例参数 \(\Delta=k=1/4\)；Fig. 2/3 是 x/p shift-correction circuits；Fig. 6 绘制 approximate qubit 在 x/p shifts 均小于 \(\sqrt{\pi}/6\) 时的 no-error probability；Fig. 7 给出 mean photon number 与 large-shift probability 的关系。
- 对本项目的意义：有限压缩、ancilla error、correctable shift bound 都不是可忽略实现细节；它们决定 simulation noise 是否能支持论文 claim。
- 写作边界：可用于质疑/审计当前 `physics/` 的 effective noise 是否过度简化。

### `grimsmo_quantum_2021` - Quantum Error Correction with the GKP Code

- 原理：综述 GKP 码的 lattice、approximate codewords、finite-energy realization、loss/dephasing 和实验路线。
- 图表/数值证据：Fig. 1 展示 square/hexagonal GKP lattices；Fig. 2 比较 loss/dephasing noise 下 GKP correction potential；Fig. 3 展示 logical measurement 的 position wavefunctions；式 (13) 定义 modular squeezing \(S_{X,Z}=-10\log_{10}(\Delta^2_{X,Z})\)。
- 对本项目的意义：note 中 approximate GKP 背景应以该文为综述锚点，说明 finite-energy envelope 和 squeezing 会进入 syndrome noise 与 logical-channel 表述。
- 写作边界：综述文献中的数值来自被综述工作，正文引用具体性能时应回到原始论文。

### `jafarzadeh_logical_2025` - Logical channels in approximate GKP error correction

- 原理：把 approximate GKP error correction 分析到 logical channel 层，而不是只看物理位移。
- 图表/数值证据：Fig. 1 给出 state preparation、N rounds teleportation、auxiliary readout 与 logical maps；Fig. 2 展示不同 \(\beta=\{0.4,0.2,0.1\}\) 下 Wigner functions 和 marginal distributions；正文给出 GRN channel Pauli probabilities，如 \(p_I=0.9900,p_X=p_Z=0.0050,p_Y=0\)。
- 对本项目的意义：结果指标最好不止 parameter MSE 或 residual displacement，还应有 logical-error proxy 或 logical-channel 解释。
- 写作边界：它支持“为什么 logical-level 指标重要”，不直接支持本项目当前 simulation 的 logical-channel 完整性。

## 4. GKP analog / soft-information decoding

### `fukui_high-threshold_2018` - High-threshold fault-tolerant quantum computation with analog QEC

- 原理：把 GKP syndrome 的连续 residual 保留下来，作为上层 decoder 的 analog information。
- 图表/数值证据：公开 ar5iv/论文 HTML 中 Fig. 1 给出 analog likelihood 与 syndrome residual 的使用方式；Fig. 2/3 给出 surface-code logical error probability curves，用于比较 conventional QEC 和 analog QEC；正文在阈值讨论中给出 required squeezing 从约 16.0 dB 降到 9.8 dB，并在摘要层面概括为从约 14.8 dB 降到低于 10 dB。
- 对本项目的意义：analog GKP information 已是强已有工作；本项目不能写成“首次利用 analog syndrome”。
- 写作边界：可作为 analog GKP soft-information 背景；正式投稿前应把 16.0 dB / 9.8 dB 对应的 PDF 页码和图号固定到 BibTeX note 或文献索引中。

### `noh_fault-tolerant_2020` - Fault-tolerant bosonic QEC with the surface-GKP code

- 原理：把 GKP qubit 与 surface code 拼接，利用 GKP stabilizer measurement 的 analog information 动态更新 matching graph weights。
- 图表/数值证据：Fig. 6 / Table 1 相关结果给出多个阈值：GKP noisy only 约 11.2 dB；GKP 与 circuit elements 同噪声时约 18.6 dB；component failure threshold 约 0.69%；noiseless GKP 下 circuit-element threshold 约 0.81%。
- 对本项目的意义：本项目如果讲 “GKP physical-layer calibration feeding outer decoder”，应把该文作为 surface-GKP soft-information 先例。
- 写作边界：它是外码 matching-weight route；本项目目前是 physical-layer affine calibration，不是 surface-GKP outer decoder。

### `noh_low_2022` - Low overhead fault-tolerant QEC with the surface-GKP code

- 原理：通过 teleportation-based GKP error correction、maximum-likelihood decoding 和 space-time correlated edges 降低 surface-GKP 开销。
- 图表/数值证据：摘要与结果页显示 12 dB 下 GKP CNOT failure 从 0.87% 降到 0.36%；Fig. 6b threshold 约 9.9 dB；Table III 明确在 \(\sigma_{\rm gkp}^{(dB)}=12\) dB 时，surface code 达到 \(p_L<10^{-7}\) 需要 1457 qubits，而 surface-GKP 需要 291 modes & 97 qubits。Fig. 7 比较不同 \(\lambda\) 下 logical Z failure rate。
- 对本项目的意义：这是 related work 中最重要的 GKP+surface 低开销数值锚点之一；本项目的 GKP front-end calibration 应被写成服务这类低开销链路的前端模块。
- 写作边界：不能把 surface-GKP 的 low-overhead 结果写成本项目结果。

### `raveendran_finite_2022` - Finite Rate QLDPC-GKP Coding Scheme

- 原理：把 QLDPC 与 GKP 结合，用 analog log-likelihood ratio 输入硬件友好的 min-sum / message-passing decoder。
- 图表/数值证据：Fig. 3/4 与相关方程展示 LP-QLDPC-GKP、analog LLR 和 decoder construction；具体阈值曲线数值需 PDF 定点复核。
- 对本项目的意义：说明 GKP analog information 不只服务 surface code，也服务 QLDPC-GKP。项目未来可把 affine calibration 作为 QLDPC-GKP soft front-end。
- 写作边界：短期不应写成已经覆盖 QLDPC-GKP circuit-level benchmark。

### `berent_analog_2024` - Analog information decoding of bosonic quantum LDPC codes

- 原理：在 bosonic QLDPC decoding 中把 analog-valued readout 信息转成 outcome-dependent error probabilities，服务 Tanner graph / message-passing 解码。
- 图表/数值证据：Fig. 1 展示 cat/bosonic qubit 的 analog-valued syndrome、outcome-dependent probabilities 和 analog Tanner graph decoding；论文为 30 页、15 图级别系统研究。
- 对本项目的意义：本项目应承认 analog information decoding 已经成熟，创新点应转向 drift-adaptive, low-dimensional, FPGA-oriented affine calibration。
- 写作边界：不要把外码 message-passing 的收益直接外推到本项目 fast-loop affine rule。

### `borah_fault_2025` - Fault Tolerant Decoding of QLDPC-GKP Codes with Circuit Level Soft Information

- 原理：在 QLDPC-GKP fault-tolerant decoding 中加入 circuit-level soft information。
- 图表/数值证据：论文 10 页、5 图；摘要/结构比较 no soft information、precomputed probabilities、real-time soft information 三种场景，强调 minimum-depth schedule 对 reliable soft information 的重要性。
- 对本项目的意义：如果本项目后续扩展到 circuit-level / QLDPC-GKP，应把 soft information 的层级讲清楚。
- 写作边界：当前项目 effective-noise/drift setting 还不是 circuit-level soft information benchmark。

### `roy_decoding_2025` - Decoding Multimode GKP Codes with Noisy Auxiliary States

- 原理：研究 multimode GKP code 中 noisy auxiliary states 引入的跨模相关性，并把辅助态噪声纳入 decoder。
- 图表/数值证据：Fig. 1 展示 logical Voronoi cell；Fig. 2 展示 multimode Steane-type QEC circuit。公开 abstract 报告 explicit tracking auxiliary-state noise 可使 logical error probability 至少降低一个数量级；正式写入强 claim 前仍应回到结果曲线固定读数。
- 对本项目的意义：当前 `physics/` noise model 应明确是否模拟 auxiliary-state / measurement noise；如果没有，论文需写成 effective-noise setting。
- 写作边界：该文的数量级改善来自 multimode/noisy auxiliary setting，不能直接作为本项目 benchmark baseline。

## 5. loss / memory / teleportation / performance analysis

### `hastrup_analysis_2023` - Analysis of loss correction with the GKP code

- 原理：分析 pure-loss channel 下 GKP QEC，以及 pre-amplification 是否有利。
- 图表/数值证据：Fig. 2 展示 loss 后 marginal distribution 和 pre-amplification effects；Fig. 3 展示 logical channel infidelity 随 finite-energy parameter 与 loss 变化。
- 对本项目的意义：如果当前 drift scenarios 只是 effective mean/variance/covariance drift，就不能写成真实 loss-channel correction。
- 写作边界：可用于噪声模型边界说明，而非当前结果对标。

### `wan_memory-assisted_2020` - Memory-assisted decoder for approximate GKP codes

- 原理：多轮 syndrome extraction 中保留历史测量信息，用 Bayesian estimation 辅助 approximate GKP decoding。
- 图表/数值证据：Fig. 1 是 q-quadrature syndrome extraction；Fig. 2 对比 memoryless vs memory-assisted；Fig. 3 在 \(\Delta\approx0.22\)、\(\bar n\approx10\)、\(\sigma_0^2=0.0005\) 下比较 fidelity；正文/摘要给出 expected total displacement error 可 bounded by \(2\sqrt{\pi}\)。
- 对本项目的意义：本项目 histogram window / temporal statistics 与该文 memory-assisted 思想相邻，但输出是 low-dimensional runtime affine parameters。
- 写作边界：不能把本项目 histogram window 写成等价于 Bayesian full decoder。

### `walshe_continuous-variable_2020` - Continuous-variable gate teleportation and bosonic-code error correction

- 原理：用 CV gate teleportation 的 Kraus-operator 视角连接非 Gaussian 操作、GKP/bosonic-code QEC 和 state injection。
- 图表/数值证据：Fig. 1 展示 macronode wire；Fig. 2 对应 gate teleportation；正文给出 teleportation operator 与 measurement-dependent gate。
- 对本项目的意义：理论背景中可解释 GKP error correction 与 bosonic-code operations 的关系。
- 写作边界：不宜把该文展开过多，否则会偏离本项目的 decoder/calibration 主线。

### `marqversen_performance_2025` - Performance analysis of GKP error correction

- 原理：比较 Knill-type 与 Steane-type GKP error correction，并分析 post-correction squeezing / displacement errors。
- 图表/数值证据：Fig. 1 展示 logical/approximate GKP wavefunctions；Fig. 2/3 对比 Knill/Steane circuits；Fig. 5 用 Strawberry Fields simulation 比较输出 Wigner functions，报告 maximum error \(10^{-8}\)。
- 对本项目的意义：有助于把 GKP correction circuit 与 simulation-level approximation 区分开。
- 写作边界：本项目当前不是 optical / Strawberry Fields circuit simulation。

### `zheng_performance_2024` - Performance and achievable rates of the GKP code for pure-loss and amplification channels

- 原理：用 transpose-channel / lattice geometry 分析 GKP 在 pure-loss 与 amplification channels 下的 performance 与 achievable rates。
- 图表/数值证据：Fig. 2 比较 near-optimal recovery 与 AD、conventional decoder、sBs circuit，在 loss \(\gamma=10\%\) 与 amplification \(G=1.1\) 下展示 infidelity；公开 abstract / 正文入口给出的 capacity-achieving 条件为 \(|\tau/(1-\tau)|\) 为整数，正式引用时应固定对应 theorem、lemma 或正文页码。
- 对本项目的意义：提醒论文区分 physical channel capacity 与 effective decoder noise benchmark。
- 写作边界：本项目不能写 capacity claim。

### `lachance-quirion_autonomous_2024` - Autonomous QEC of GKP states

- 原理：实验方向的 autonomous GKP QEC，用 reservoir engineering 和 auxiliary transmon reset 实现自动纠错。
- 图表/数值证据：公开 ar5iv/论文 HTML 中 Fig. 1 给出 superconducting cavity + transmon + reservoir 的硬件结构；Fig. 2 给出 reset protocol 与 average reset error；Fig. 3 给出 sBs autonomous QEC protocol 以及 with/without optimized QEC 的 logical lifetime 对比，正文说明 optimized protocol/QEC 增加 logical lifetime，并达到 corrected errors 多于 generated errors 的操作点。
- 对本项目的意义：提供真实 GKP 实验闭环背景。
- 写作边界：本项目当前无真实 GKP 实验系统或 real-board execution，引用时只能作为目标标准/背景。

## 6. adaptive / calibration decoders

### `spitz_adaptive_2018` - Adaptive weight estimator for QEC

- 原理：从 syndrome history 中估计 decoder weights，使 decoder 随噪声漂移调整。
- 图表/数值证据：Fig. 3 在 d=3 repetition code、\(\gamma_i=5\times10^{-3}\) 下显示 adaptive decoder 对 ideal blossom decoder 的收敛，\(\Delta\propto N^{-\alpha}\)、\(\alpha\approx1.2\)；Fig. 4 的周期漂移例子中 \(N_{\rm opt}\approx1265\)。
- 对本项目的意义：直接支撑“syndrome statistics -> decoder parameter”不是新概念；本项目要强调映射目标是 GKP affine fast-path。
- 写作边界：该文是 repetition-code adaptive weights，不是 GKP physical-layer affine correction。

### `wagner_optimal_2021` - Optimal noise estimation from syndrome statistics

- 原理：从 syndrome statistics 中做 noise parameter estimation，并分析 optimality。
- 图表/数值证据：Fig. 2 对比 maximum-likelihood decoder LER；estimation 使用 \(10^5-10^6\) random errors，perfect-knowledge decoder 使用 \(10^8\) random errors；Fig. 3 显示 EM estimator MSE 达到 Cramer-Rao bound；HEM/EM 约 5 iterations。
- 对本项目的意义：statcalib / histogram-moment route 应与 noise-estimation 文献对齐。
- 写作边界：本项目当前 statcalib 仍是 bounded extension lane，不是 mature optimal estimator。

### `wang_dgr_2023` - DGR: Tackling Drifted and Correlated Noise via Decoding Graph Re-weighting

- 原理：面对 drifted/correlated noise，动态重加权 decoding graph。
- 图表/数值证据：Fig. 3 量化短期/长期硬件 drift，100 秒内 gate-error drift 可超过 15x，1Q/2Q drift 可超过 10x，1Q gate drift 可超过 1000x；DGR average-case mismatch LER 平均降低 3.6x，surface-code worst-case 平均 695x、最高 7360x。
- 对本项目的意义：漂移适应已有 graph-level 强对照；本项目应定位为 GKP physical-layer affine calibration。
- 写作边界：不要把 graph re-weighting 的巨大收益外推成本项目性能。

### `chen_calibrated_2022` - Calibrated decoders for experimental QEC

- 原理：实验 QEC 中用 correlated event calibration 训练 matching decoder。
- 图表/数值证据：公开 ar5iv/论文 HTML 中 Fig. 2(b) 给出 calibrated graphlike / hyperedge error probabilities；Fig. 3 上半部分比较不同 post-selection 条件下 retained fraction，下半部分给出 0-10 stabilizer rounds 的 logical error 结果；正文/摘要报告使用 up to size-4 correlated events，partial post-selection 保留数据量约为 full post-selection 的 10x，logical errors per round 为 \(2.2\pm0.1\times10^{-2}\)（decoded without post-selection）和 \(5.1\pm0.7\times10^{-4}\)（full post-selection），低于 physical measurement error \(7\times10^{-3}\)。
- 对本项目的意义：calibrated decoder 是实验 QEC 的真实需求，本项目可以引用它支撑 decoder calibration 重要性。
- 写作边界：可作为 calibration decoder 重要性的实验背景；本项目当前没有同等级 experimental-QEC calibration evidence。

### `sivak_optimization_2024` - Optimization of decoder priors for accurate QEC

- 原理：直接优化 decoder prior，而不是改 decoder 主体。
- 图表/数值证据：Fig. 3 的 repetition-code 数据显示 calibrated prior 相比 uninformative prior 平均 LER 降低 48%，RL prior 相比 correlation-based method 再改善 16%；Fig. 4 / Table I 的 surface-code memory 实验中，相对 decoder-agnostic prior 改善约 3.3%，数据集包含 53 experiments、每个 \((d,r)\) 约 \(7.5\times10^4\) shots。
- 对本项目的意义：支持把 slow-loop teacher/statistics 模块写成 prior/control-surface calibration，而不是纯 CNN 准确率竞赛。
- 写作边界：该文是真实实验 prior optimization，本项目当前是 software-HIL / simulation-boundary。

## 7. learned / AI QEC decoders

### `bausch_learning_2024` - Learning high-accuracy error decoding for quantum processors

- 原理：AlphaQubit 用 recurrent/transformer-style learned decoder 学真实 surface-code data。
- 图表/数值证据：Fig. 3 报告 Sycamore d=3/d=5 finetuned LER：\((2.901\pm0.023)\times10^{-2}\) 与 \((2.748\pm0.015)\times10^{-2}\)，对照 MWPM-Corr 约 3.498% / 3.597%；Fig. 4 扩展到 Pauli+ d=11；训练/微调样本包括 325,000 experimental samples 和大规模 synthetic samples。
- 对本项目的意义：高质量 learned-decoder paper 必须有真实 train/test split、强 baseline、大规模数据和明确实验系统。
- 写作边界：本项目当前不是真实量子处理器实验验证。

### `chamberland_fast_2026` - Fast and accurate AI-based pre-decoders for surface codes

- 原理：AI pre-decoder 在 global decoder 前预处理 syndrome，目标是同时降低 LER 与 runtime。
- 图表/数值证据：Table IX/X 给出 runtime / speedup；例如 d=31、p=0.006 下 correlated PyMatching 子步骤从 270.83 us 降到 38.78 us，sub-step speedup 7.0x，总 pipeline speedup 3.54x；摘要称 large-distance runtime 达 \(O(1\mu s)\)/round。
- 对本项目的意义：它的写作范式最值得模仿：系统约束、模块责任、accuracy-runtime trade-off、deployment role。
- 写作边界：不要借用其 GPU/runtime 指标；本项目需要自己的 latency/resource 证据。

### `stein_calibration-conditioned_2026` - Calibration-Conditioned FiLM Decoders

- 原理：用 calibration graph embedding 通过 FiLM 调制 CNN decoder，使 decoder 适配不同 calibration states。
- 图表/数值证据：Fig. 1 是 hardware graph encoder + FiLM generator + CNN backbone；Fig. 6/7 与 Table V/VI 显示 unseen recalibrated chains 上 Z-basis 最高 11.11x LER reduction；实验覆盖 IBM Fez/Kingston/Pittsburgh，2,760,704 shots、400 calibration snapshots、distance up to d=11；Table I 显示 folded FiLM latency 约 85-95 us，Dynamic FiLM 约 1.4 ms。
- 对本项目的意义：这是最邻近 “calibration-conditioned learned decoder” 工作。本项目必须强调差异：我们更新的是 low-dimensional affine runtime surface，fast path 不是 per-shot neural decoder。
- 写作边界：不能把本项目当前 CNN/teacher route 写成 FiLM-style full neural decoder。

### `peled_neural_2026` - Neural Minimum Weight Perfect Matching

- 原理：用 learned component 预测 MWPM 权重，把神经网络嵌入经典结构 decoder。
- 图表/数值证据：Fig. 4 报告 NMWPM threshold：independent noise 下 10.95%，对照 MWPM 10.3%、BPOSD-2 10.8%、QECCT 10.7%；depolarizing noise 下 NMWPM 17.9%，对照 MWPM/BPOSD-2 16.0%、QECCT 17.8%。模型参数包括 hidden dim 128、4-layer GNN、4 attention heads、2-layer Transformer encoder。
- 对本项目的意义：支持“structured hybrid decoder”写法：神经模块不必替代整个 decoder，也可校准/调制经典结构。
- 写作边界：该文是 surface/toric code MWPM route，不是 GKP affine correction。

### `wang_multidimensional_2022` - Multidimensional Bose QEC based on neural network decoder

- 原理：神经网络用于 multidimensional bosonic / GKP-like QEC decoding。
- 图表/数值证据：Fig. 6/7 报告 threshold：MWPM 在 data GKP noisy only 时 \(\sigma\approx0.50\)，measurement also noisy 时 \(\sigma\approx0.25\)；neural decoder 在 noise-free measurements 时 \(\sigma\approx0.78\)，all measurements noisy 时 \(\sigma\approx0.34\)。对应 \(\bar p\) 约 12.3%/10.02% 与 15.12%/11.37%。
- 对本项目的意义：神经网络用于 GKP/bosonic decoding 已存在；本项目新意应写成 runtime-bounded drift calibration。
- 写作边界：不能写“neural GKP decoding 首次提出”。

### `sivak_reinforcement_2026` - Reinforcement Learning Control of QEC

- 原理：把 QEC error-detection events 作为 RL control signal，在线 steering physical control parameters / decoder。
- 图表/数值证据：Fig. 3-5 报告 RL fine-tuning 约 20% LER suppression；injected drift 下 LER reduction 24%、stability 2.4x，加入 decoder steering 后 LER reduction 31%、stability 3.5x；distance-7 surface code 平均 logical error per cycle \(\epsilon_L=7.72(9)\times10^{-4}\)，distance-5 color code 为 \(8.19(14)\times10^{-3}\)；d=15 simulation 约 40,000 control parameters。
- 对本项目的意义：说明 learned control/adaptation 已进入真实 QEC 控制层；本项目应强调 simpler dual-loop affine calibration。
- 写作边界：不要把本项目写成 RL control 系统。

### `hanisch_soft_2026` - Soft information decoding with superconducting qubits

- 原理：在 superconducting-qubit repetition-code experiments 中保留 soft measurement information。
- 图表/数值证据：Fig. 7 simulation 中 soft threshold 比 hard 高 11.5%（Z basis）和 15.7%（X basis）；Fig. 8 / Table I 在 IBM Sherbrooke T=50 数据中给出平均 threshold improvement 24.4%；摘要称 error rates 可最高降低 30x，每次测量 1 byte 信息足以接近最优。
- 对本项目的意义：soft information 不只属于 GKP；低比特 soft features 也很适合 fixed-point / FPGA feature design。
- 写作边界：该文是 superconducting repetition-code soft decoding，不是 GKP drift benchmark。

## 8. FPGA / real-time / hardware decoder

### `das_lilliput_2021` - LILLIPUT

- 原理：lookup-table based low-latency decoder，用 offline software decoder 预填 LUT，在线实时查询。
- 图表/数值证据：公开 ar5iv/论文 HTML 中 Fig. 1 给出 LILLIPUT 流程；Fig. 3 展示 distance-5 layout；Table 1 列出 distance 3/4/5 的 lattice 配置；Table 3/4 给出 FPGA logic utilization、frequency 和 latency，在线部分逻辑 <7%、latency 约 28-42 ns；Fig. 16 与 Table 6 给出 CLUT compression，从 148 MB 压到 1.38 MB，约 107x reduction。
- 对本项目的意义：支持 compact fast path / table-bank / affine parameter bank 的硬件友好叙事。
- 写作边界：可作为 low-latency LUT decoder 的硬件对比；但本项目不能用该文数据替代自己的 FPGA resource / timing 证据。

### `liyanage_scalable_2023` - Helios FPGA surface-code decoder

- 原理：FPGA-based distributed Union-Find decoder，利用 tree-grid parallel resources 降低 decoding backlog。
- 图表/数值证据：公开 arXiv/PDF 入口给出 Xilinx VCU129 上实现到 distance-21 surface code，在 0.1% phenomenological noise 下平均 11.5 ns / measurement round；摘要/正文入口还指出 decoding time per round 随 code distance 增大而下降，这是 parallel resource scaling 的核心指标。
- 对本项目的意义：给出真实 FPGA decoder latency 标准；本项目如果讲 FPGA，必须有 latency/resource/provenance 表。
- 写作边界：该条已定位公开全文入口，但目前卡片仍主要依赖公开入口中的核心数值；若作为正文硬件表的强对标，后续应补原 PDF 的 table/figure 页码。

### `barber_real-time_2023` - Collision Clustering decoder

- 原理：面向 real-time surface-code decoding 的 collision clustering hardware decoder。
- 图表/数值证据：公开 HTML/摘要给出 FPGA 可覆盖 881-qubit surface code、810 ns、10 KB memory；ASIC 到 1057 qubits、240 ns、0.06 mm²、8 mW。
- 对本项目的意义：硬件论文的指标必须包括 scale、latency、memory、area/power。
- 写作边界：本项目当前没有 ASIC/FPGA 实测闭环，不能并列表述为同等成熟度。

### `maurya_fpga-tailored_2025` - FPGA-tailored algorithms for real-time qLDPC decoding

- 原理：比较 message passing、filtered-OSD、clustering 等 qLDPC FPGA-tailored decoder。
- 图表/数值证据：Fig. 2 以 logical error rate vs FPGA cycle budget 比较不同 decoder；Relay 在 gross code 上约一个数量级优势，在 two-gross 上约两个数量级优势。
- 对本项目的意义：强调 algorithm-hardware co-design，不只是“模型可部署”。
- 写作边界：该文是 qLDPC decoder，不是 GKP affine fast-loop。

### `ziad_local_2024` - Local Clustering Decoder

- 原理：local/adaptive hardware decoder，尤其考虑 leakage-dominant model。
- 图表/数值证据：Fig. 3 展示 under 1 us/round；Xilinx VU19P 上 d=17 约 6% LUT、3% FF；leakage-aware adaptivity 可用约 4x fewer physical qubits。
- 对本项目的意义：可借鉴 local/fast path + adaptive policy update 的系统表达。
- 写作边界：surface-code binary syndrome 与 GKP analog syndrome 不能混写。

### `maurer_real-time_2025` - Real-time decoding of the gross code memory with FPGAs

- 原理：gross-code / bivariate bicycle code 的 real-time FPGA BP decoder。
- 图表/数值证据：Table 1 给出 resource breakdown；BP iteration 24 ns，12-cycle window 平均 <240 ns；circuit-model error probability <3e-3 时 per-cycle decoding <1 us；BP decoder 消耗 97% LUT，其中 VNU 67.2%、CNU 22.6%。
- 对本项目的意义：如果继续讲 fixed-point/低精度，应报告 precision、resource 和 logical performance 接近浮点的证据。
- 写作边界：本项目当前无对应 FPGA synthesis/resource table。

### `caune_demonstrating_2024` - Real-time and low-latency QEC with superconducting qubits

- 原理：真实 superconducting system 上用 FPGA Collision Clustering decoder 做 real-time / low-latency QEC。
- 图表/数值证据：Fig. 3/4 与正文报告 8-qubit stability experiment、最多 25 decoding rounds，mean decoding time <1 us/round；5 到 25 rounds 平均约 0.44-0.79 us；9-round fast-feedback response 9.6 us，其中 decoding 6.5 us、communication/control latency 3.1 us。
- 对本项目的意义：这是硬件闭环标准参照；本项目当前只能写 software-HIL / read-only gate。
- 写作边界：不能把 current-host gate 或 software-HIL 写成 real-time QEC demo。

### `yang_real-time_2026` - Real-time Surface-Code Error Correction Using an FPGA-based Neural-Network Decoder

- 原理：FPGA-integrated recurrent neural-network decoder，用于 real-time surface-code QEC。
- 图表/数值证据：Fig. 1/2/3 给出系统架构、FPGA NN decoder 与 real-time QEC；distance-3，17 physical qubits = 9 data + 8 ancilla；closed-loop latency 550 ns，其中 NN decoding 124 ns，syndrome compute 20 ns，PFU 4 ns；QEC cycle 1.25 us；NN throughput period 184 ns；6-bit quantized weights；real-time logical error rates \(0.072(1)\) / \(0.097(2)\)，接近 offline MWPM \(0.073(2)\) / \(0.095(2)\)。
- 对本项目的意义：这是 neural+FPGA+real-time QEC 的强邻近工作。本项目必须说明差异：我们的 CNN/teacher 在 slow-loop calibration side，fast path 是 deterministic affine rule。
- 写作边界：本项目当前没有达到 550 ns closed-loop / real QPU feedback 证据。

### `sivak_real-time_2023` - Real-time QEC beyond break-even

- 原理：bosonic code / superconducting platform 上的 real-time QEC beyond break-even。
- 图表/数值证据：公开 ar5iv/论文 HTML 中 Fig. 3 给出 system coherence 对比；Fig. 4 给出 syndrome 与 QEC outcome characterization；Supplementary Table S3 给出 QEC timing breakdown；摘要/正文报告 fully stabilized error-corrected logical qubit 的 coherence gain \(G=2.27\pm0.07\)，超过最佳物理组件。
- 对本项目的意义：硬件闭环 QEC 的高标准实验参照。
- 写作边界：本项目不能借此写成 break-even 或真实闭环；只能作为 real-time experimental QEC 标准参照。

## 9. 对本项目的写作与实验启发

1. **理论定位**：GKP analog syndrome、finite-energy GKP、surface-GKP / QLDPC-GKP soft information 都已有强文献。本项目新意应写成 drift-adaptive physical-layer affine calibration under dual-loop runtime constraints。
2. **方法定位**：Stein FiLM、Sivak prior optimization、DGR、Spitz/Wagner 都说明 calibration/adaptation 已存在。本项目要强调 fast path 是 deterministic affine rule，learned/statistical module 只更新 low-dimensional runtime surface。
3. **benchmark 定位**：高质量论文至少需要 fixed prior / adaptive prior / oracle prior / UKF-EKF / window variance / CNN-only / teacher-residual / statcalib extension 的清晰分层，且要有 held-out drift 或 unseen drift。
4. **硬件定位**：LILLIPUT、Helios、Collision、Caune、Yang 等文献的核心指标是 latency、resource、throughput、closed-loop、真实硬件证据。本项目当前只能写 software-HIL、isolated `.tflite` runtime 和 read-only real-board gate，不能写 real-time FPGA result。
5. **图表定位**：论文至少应有 related-work comparison table，显式列出 “GKP analog / calibration decoder / learned decoder / FPGA real-time decoder / 本项目证据等级”。

## 10. 正式投稿前仍需补强的页码/数值队列

以下不是缺失条目；38 篇都已经有独立证据入口。这里列的是如果要把对应数字写进正式论文正文或 comparison table，仍建议回到 PDF 固定页码、图号或曲线读数的条目：

| key | 当前等级 | 需要补的内容 |
| --- | --- | --- |
| `raveendran_finite_2022` | `公开PDF/HTML核验` | 补具体 threshold curve 数字 |
| `borah_fault_2025` | `PDF图表/表格已核验` | 后续若用于正文强 claim，需从 Fig. 结果曲线读出具体 LER/threshold 数值 |
| `roy_decoding_2025` | `PDF图表/表格已核验` | Fig. 1/2 的结构证据已核；若写“至少一个数量级”改善，需补对应结果图或表的具体读数 |
| `zheng_performance_2024` | `PDF图表/表格已核验` | Fig. 2 的 performance comparison 已核；若写 capacity-achieving 条件，需固定原文 theorem/lemma 或正文页码 |
| `liyanage_scalable_2023` | `公开PDF/HTML核验` | 若列入硬件主 comparison table，需补 evaluation table 或 PDF 页码来支撑 d=21 / 11.5 ns |

## 11. note 中仍未稳定同步的条目

`CNN_FPGA_GKP_theory_note_draft.tex` 中此前提到但 Zotero 侧仍未稳定核验的硬件 decoder 条目：

| note key | note 中题名 | 当前状态 | 后续动作 |
| --- | --- | --- | --- |
| `afs2022` | `AFS: Accurate, fast, and scalable error-decoding for fault-tolerant quantum computers` | 仍缺稳定 DOI/arXiv/ACM/IEEE 来源 | 需要补元数据；未补前不建议作为强引用 |
| `astrea2023` | `Astrea: Accurate quantum error-decoding via practical minimum-weight perfect-matching acceleration` | 仍缺稳定 DOI/arXiv/ACM/IEEE 来源 | 需要补元数据；未补前不建议作为强引用 |
