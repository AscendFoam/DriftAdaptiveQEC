# 针对 `CNN_FPGA_GKP_theory_note_draft.tex` 的结构化审阅报告

我直接读取了上传的 `.tex` 草稿；下面对草稿内容用“草稿行号”定位，对外部文献用可点击引用定位。核心判断是：**没有发现一个已发表/预印本工作完全等同于“从 GKP syndrome histogram 估计漂移，并在线更新仿射 GKP decoder 参数 `(K,b)`，且用 teacher-anchored neural residual 做部署约束校准”**。但如果论文把贡献写宽，会立刻撞上四个很近的方向：GKP analog soft-information decoding、syndrome-statistics adaptive priors、calibration-conditioned neural decoders、FPGA/real-time QEC decoders。

---

## 1. Executive summary

### 1.1 总体判断

**[已验证：草稿事实]** 草稿当前最稳妥的贡献边界是：一个面向 GKP decoding 的 two-timescale adaptive affine decoder 设计；fast loop 是 FPGA-friendly 的
[
\Delta_t = K_t s_t + b_t
]
slow loop 用 syndrome histogram、teacher estimator 和轻量 CNN residual 更新主要偏置项 (b_t)。草稿自己已经明确写出：当前证据只是 **mock-backed software HIL + frozen-set revalidation**，不是真实 FPGA board，不是真实 `.tflite` runtime，也不是 paper-grade expanded benchmark；机制故事也不能写成单调因果闭环。这一点草稿第 18–26、135–143、466–493 行写得很清楚，应该保留并前置。

**[文献事实]** “GKP analog information 有用”已经不是新贡献。GKP 基础编码来自 Gottesman–Kitaev–Preskill；近年的 GKP review、surface-GKP、toric/surface-GKP、QLDPC-GKP、bosonic soft-information decoding 已经系统使用连续 syndrome / analog readout / soft information 来改进 decoding。([arXiv][1])

**[文献事实]** “从 syndrome statistics / calibration data 更新 decoder priors”也不是新贡献。Spitz 等的 adaptive weight estimator、DGR、Wagner 等的 syndrome-only noise estimation、Chen 等 calibrated decoders、Sivak 等 decoder-prior optimization 都已经把 decoder prior / graph weights / hypergraph priors 与 syndrome statistics、hardware calibration 或 logical error rate 直接连接起来。([arXiv][2])

**[文献事实]** “calibration-conditioned neural decoder / low-latency AI decoder”也已经很接近。Stein 等 2026 的 calibration-conditioned FiLM decoder 明确利用 calibration drift 比 decoding 慢的 timescale separation，把 calibration 信息编码后调制低延迟 convolutional decoder；Chamberland 等 2026 的 AI pre-decoder 也强调模块化、低延迟、可组合，并包含从 syndrome statistics 估计未知/时变噪声的方向。([arXiv][3])

**[文献事实]** “FPGA / real-time QEC decoder”更不能泛泛声称为新。已有 real-time surface-code FPGA neural decoder、Rigetti/Riverlane 低延迟 FPGA decoder、trapped-ion real-time feedback、local clustering FPGA decoder、qLDPC/Gross-code FPGA decoder 等；此外 real-time bosonic QEC beyond break-even 也已经存在。([arXiv][4])

### 1.2 最重要结论

**没有发现“正面完全撞车”。** 我没有找到一个工作同时具备以下四点：

1. GKP syndrome histogram 作为 slow-loop drift estimator；
2. fast loop 保持为硬件友好的 affine displacement rule ((K,b))；
3. teacher estimator 作为稳定锚点；
4. neural residual 只做 deployment-constrained calibration，尤其是 residual (b) 或小范围 correction，而不是完整 learned decoder。

**但有明显“宽口径撞车风险”。** 如果论文写成“第一个 adaptive neural QEC decoder”“第一个 calibration-conditioned decoder”“第一个 syndrome-statistics drift-adaptive decoder”“第一个 FPGA QEC decoder”“第一个 GKP soft-information decoder”，都会不成立或非常危险。最安全、最有差异化的定位是：

> **A deployment-constrained, teacher-anchored residual calibration layer for drift-adaptive affine GKP decoding, evaluated so far only in mock-backed software HIL and frozen-set revalidation.**

### 1.3 对草稿的最高优先级修改

1. **把理论 claim 降到 local linear-MMSE / affine approximation。** 草稿第 202–237 行的 Gaussian local-branch derivation 是站得住的，但必须明确：这不是全局 Bayes-optimal GKP decoder；在 modular lattice boundary 附近 posterior 是多峰的，ML/CVP/lattice decoders 会更强。
2. **明确 (\Delta_t) 的符号约定。** 草稿现在同时使用“estimated displacement”和“correction displacement”的叙事；如果 (\Delta_t) 是实际施加的 corrective displacement，它通常应为误差估计的负号。第 349–388 行的 (b_{\mathrm{target}}=\alpha(I-K)\mu) 需要严格解释。
3. **把 benchmark 从 parameter regression 改为 logical error / regret / adaptation-lag / hardware-contract。** 草稿第 241–248 行已经承认 closed-loop LER 更强，应把它变成主指标。
4. **related work 要重构，不要只在引言中点名。** 现在 bibliography 太薄；至少要新增 adaptive priors、soft information、FiLM-conditioned decoder、AI predecoder、FPGA decoder、syndrome noise estimation、QLDPC-GKP 几组文献。
5. **持续 hedge 硬件证据。** 不要把 mock software HIL 写成 true board；不要暗示 `.tflite` runtime 已恢复；不要把 frozen-set ranking 写成一般化性能结论。

---

## 2. Most similar works matrix

| 相似工作                                                                                                                                       | 像在哪里                                                                                                                        | 不同在哪里                                                                                                         | 对本项目该借什么                                                                                        | 该避免什么                                                         | 撞车风险 |
| ------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------- | ---- |
| **Noh–Chamberland–Brandão, surface-GKP, PRX Quantum 2022, arXiv:2103.06994, DOI 10.1103/PRXQuantum.3.010315**                              | 使用 GKP analog information / full syndrome history 动态计算 matching edge weights；属于 GKP + soft information 的核心近邻。([arXiv][5])   | 目标是 surface-GKP outer-code decoder；更新的是 matching/graph weights，不是 GKP physical affine correction ((K,b))。     | 借它的 analog soft-information baseline、dynamic-weight narrative、logical-error evaluation。         | 不要声称“首次利用 GKP analog syndrome”或“首次 dynamic GKP soft decoder”。 | 高    |
| **Noh–Chamberland, fault-tolerant bosonic QEC with surface-GKP, PRA 2020, arXiv:1908.03579**                                               | 把 GKP stabilizer information 与 surface-code decoding 结合，讨论 noisy circuits 与 threshold。([arXiv][6])                          | 不是 drift-adaptive affine correction，也不是 hardware residual calibration。                                        | 借 circuit-level noise framing、noisy GKP ancilla/measurement 设置。                                 | 不要只用 ideal displacement noise。                                | 中高   |
| **Borah et al., QLDPC-GKP with circuit-level soft information, arXiv:2505.06385**                                                          | 强调 real-time soft information 对 QLDPC-GKP decoding 很关键；与“实时 soft priors”高度相关。([arXiv][7])                                   | 关注 QLDPC-GKP concatenation 与 circuit-level soft info，不是 histogram drift estimator 更新 ((K,b))。                 | 借 “no soft / precomputed soft / real-time soft” 三档 baseline。                                    | 不要把 soft-info advantage 当成本项目独有。                              | 高    |
| **Berent et al., analog information decoding of bosonic QLDPC codes, PRX Quantum 2024, arXiv:2311.01328, DOI 10.1103/PRXQuantum.5.020349** | 使用 bosonic readout analog information 改进 concatenated QLDPC decoding。([arXiv][8])                                           | 不是漂移校准，不是 affine fast path，不是 FPGA residual update。                                                           | 借 analog-information integration vocabulary。                                                    | 不要写成“bosonic analog decoding 的新范式”。                           | 中    |
| **Raveendran et al., QLDPC-GKP, Quantum 2022, arXiv:2111.07029, DOI 10.22331/q-2022-07-20-767**                                            | 用 GKP analog information 与 min-sum / QLDPC decoding；有 hardware-friendly flavor。([arXiv][9])                                 | 重点是 finite-rate QLDPC-GKP code 与 decoder，不是在线 drift calibration。                                              | 借 min-sum / trapping-set / error-floor 讨论。                                                      | 不要忽略 QLDPC-GKP soft decoder baseline。                         | 中    |
| **Stein et al., calibration-conditioned FiLM decoders, arXiv:2601.16123**                                                                  | 与本草稿最像的 learned-adaptation 近邻：slow calibration context 条件化 fast neural decoder；明确利用 drift timescale separation。([arXiv][3]) | repetition-code experiments；FiLM 条件化完整 neural decoder；不是 GKP histogram → affine ((K,b))，也不是 teacher residual。 | 借 calibration-conditioned framing、generalization-to-new-days/chains、latency-overhead reporting。 | 不要声称“首次 calibration-conditioned neural QEC decoder”。          | 极高   |
| **Chamberland et al., AI pre-decoders, arXiv:2604.12841**                                                                                  | modular AI layer、低延迟、可与 global decoder 组合；还提到从 syndrome statistics 学 unknown/time-varying noise。([arXiv][10])               | surface-code pre-decoder / GPU runtime；不是 GKP affine correction，也不是 teacher-anchored residual (b)。            | 借 modular predecoder vs global decoder 的 positioning。                                           | 不要把 CNN slow-loop 写成一般 AI predecoder 首创。                      | 极高   |
| **DGR, decoding graph re-weighting, arXiv:2311.16214**                                                                                     | 从 decoding-iteration statistics 更新 edge probabilities / correlations，应对 drifted and correlated noise。([arXiv][11])          | surface-code graph reweighting，不是 GKP syndrome histogram 更新 affine displacement rule。                         | 借 drifted/correlated noise baseline 和 “mismatched prior regret” 指标。                             | 不要声称“首次 drift-adaptive decoder prior”。                        | 高    |
| **Spitz et al., adaptive weight estimator, Adv. Quantum Technol. 2018, DOI 10.1002/qute.201800012**                                        | decoder weights 来自 measured data，面向 time-dependent environment。([arXiv][2])                                                 | stabilizer-code/MWPM 权重更新；不是 GKP affine fast loop。                                                            | 借 AWE 作为 adaptive baseline。                                                                     | 不要把 syndrome-driven weight adaptation 写成新。                    | 高    |
| **Wagner et al., syndrome-only noise estimation, PRR 2021 / Quantum 2022**                                                                 | 研究从 syndrome measurements 估计噪声，是 histogram-based online calibration 的理论近邻。([arXiv][12])                                     | Pauli/stabilizer channel estimation，不是 modular continuous GKP histogram → ((K,b))。                            | 借 identifiability、Fisher information、CRB、online estimation 语言。                                  | 不要忽略 histogram identifiability 问题。                            | 高    |
| **Sivak–Newman–Klimov, decoder-prior optimization, PRL 2024, arXiv:2406.02700, DOI 10.1103/PhysRevLett.133.150603**                        | 直接优化 decoder priors 以最小化 logical error rate；强调 proxy metrics 不足。([APS Link][13])                                            | RL-inspired offline/experimental prior optimization；不是 lightweight GKP affine residual runtime loop。          | 借“以 LER 而非参数误差为目标”的叙事。                                                                          | 草稿 bib 把它写成 Nature 不对，应改为 PRL 133, 150603。                    | 高    |
| **Yang et al., FPGA neural-network surface-code decoder, arXiv:2605.04892**                                                                | 真 hardware-integrated FPGA neural decoder，报告 deterministic closed-loop latency。([arXiv][4])                                 | surface code d=3；不是 GKP/bosonic；本项目目前没有真实 board 结果。                                                           | 借 p50/p99/worst latency、closed-loop timing、resource contract 写法。                                | 不要把 mock HIL 与该类真实 FPGA 结果并列。                                 | 高    |
| **Rigetti/Riverlane real-time FPGA decoder, arXiv:2410.05202**                                                                             | 集成到 control system，报告 per-round decode latency 和 feedback response。([arXiv][14])                                            | superconducting stabilizer experiment；不是 GKP affine residual。                                                 | 借 HIL/real-time reporting checklist。                                                            | 不要写“real-time FPGA validated”除非真的上板。                          | 中高   |
| **Sivak et al., real-time bosonic QEC beyond break-even, Nature 2023, DOI 10.1038/s41586-023-05782-6**                                     | real-time bosonic QEC 的重要硬件参照。([Nature][15])                                                                                | 不是 FPGA histogram-to-affine decoder；也不是本项目的软件 HIL 设置。                                                         | 借 bosonic QEC experimental context。                                                             | 不要借此暗示本项目已经有 bosonic hardware loop。                           | 中    |

---

## 3. Literature map by theme

### 3.1 GKP / bosonic soft-information decoding

**[文献事实]** GKP code 的基本机制是把 qubit 编码到 oscillator phase space 的 lattice states 中，并通过 modular displacement syndrome 来纠正 small shift errors；近年 review 已经把 finite-energy GKP、displacement noise、bosonic QEC 与 fault tolerance 系统化。([arXiv][1])

**[文献事实]** Fukui 等的 analog QEC 工作、Noh 等的 surface-GKP 工作、Vuillot/Terhal 等 toric-GKP 工作都已经利用连续 syndrome / analog information 改善 decoding threshold 或 logical error performance。([arXiv][16])

**[对草稿影响]** 草稿不能把 “GKP syndrome histogram / analog information 进入 decoder” 当作贡献本身。贡献要落在：**在 deployment-constrained fast path 中，把 analog statistics 压缩为 affine parameters ((K,b))，并用 teacher-anchored residual 做 slow calibration**。

### 3.2 Surface-GKP / concatenated GKP / QLDPC-GKP

**[文献事实]** Surface-GKP 与 QLDPC-GKP 文献已经形成一条强线：GKP analog readout 不只是单模校正信息，也可以转化为 outer-code soft weights、min-sum messages、matching weights 或 real-time soft information。Borah 等 2025 明确比较 circuit-level soft information 的使用；Berent 等 2024 讨论 bosonic QLDPC analog-information decoding；Raveendran 等 2022 使用 QLDPC-GKP 与 min-sum 思路突破 CSS Hamming bound。([arXiv][7])

**[对草稿影响]** 如果草稿只在 single-mode / few-mode GKP 上做漂移校准，就应诚实说是 **physical-layer GKP affine calibration layer**，不是 surface-GKP 或 QLDPC-GKP 的完整 decoder。若未来想接 surface/QLDPC outer code，应把本方法描述为给 outer decoder 提供更稳定、更 calibrated 的 physical soft information，而不是替代 outer decoder。

### 3.3 Soft information 与 decoder priors

**[文献事实]** Soft-information QEC 已经覆盖 measurement readout、detector likelihood、MWPM/UF 权重、calibrated decoder priors 等方向。Pattison 等展示 soft measurement information 可以改善 QEC decoder；Hesner 等用 detector likelihood 做 QEC benchmarking；Chen 等在 experimental QEC 中校准 decoder；Sivak 等直接优化 decoder priors 并强调 proxy metric 不一定等于 logical performance。([arXiv][17])

**[对草稿影响]** 草稿目前第 241–248 行已经意识到 logical failure criterion 比 parameter regression 更强；建议把这一点升级为论文主线：**本方法不是为了更准地估计 drift 参数，而是为了在 latency/fixed-point/commit constraints 下最小化 closed-loop LER 或 regret-to-oracle**。

### 3.4 Teacher-guided / residual / FiLM / calibration-conditioned decoder

**[文献事实]** Stein 等 2026 的 calibration-conditioned FiLM decoder 是本项目 learned-calibration 方向的最近邻：它把 calibration data 编码成 conditioning signal，用 FiLM 调制低延迟 convolutional decoder，并强调 calibration drift 与 decoding latency 的 timescale separation。([arXiv][3])

**[文献事实]** Chamberland 等 2026 的 AI pre-decoder 也很近：它是模块化、低延迟、可与传统 decoder 组合的 learned layer，并讨论 unknown/time-varying noise 的 syndrome-statistics estimation。([arXiv][10])

**[文献推断]** 我没有找到一个 QEC 工作明确采用“classical teacher estimator + neural residual only for runtime adaptation”的完整组合，并且把 residual 限制到 GKP affine ((K,b)) 或主要 (b) correction。最接近的是 calibration-conditioned neural decoder、AI predecoder、decoder-prior optimization、DGR-style reweighting，但它们不是 teacher-anchored residual affine GKP decoder。这个差异化空间存在，但需要用 ablation 证明 teacher anchor 真有用。

### 3.5 Drift-adaptive / prior optimization / online calibration

**[文献事实]** Syndrome statistics 用于估计噪声或更新 decoder weights 已经很成熟。Wagner 等研究从 syndrome-only measurements 估计 noise；Spitz 等用 measured data 自适应估计 decoder weights；DGR 用 decoding statistics 更新 drifted/correlated noise 下的 graph probabilities；CaliScalpel 等把 in-situ calibration 与 surface-code QEC 集成。([arXiv][12])

**[对草稿影响]** 草稿第 275–314 行定义 (\theta=(\sigma,\mu_q,\mu_p,\vartheta)) 与四类 drift scenario 是合理的，但还不够。必须加入 **identifiability / aliasing / stale update** 分析：modular GKP syndrome histogram 对 mean drift、variance drift、measurement noise、rotation drift 可能存在不可辨识区域，尤其在 wrapping、低 shot budget、drift faster than window 的情况下。

### 3.6 Real-time / FPGA / hardware-in-the-loop QEC decoder

**[文献事实]** 真实 real-time / FPGA QEC decoder 已经有多个强参照：Yang 等 2026 在 superconducting surface code 上做 FPGA neural-network decoder；Rigetti/Riverlane 报告 integrated FPGA decoder 与 feedback latency；Ziad 等 local clustering decoder 强调 FPGA、低延迟、leakage-adaptive；Maurer 等把 Gross-code/qLDPC decoder 映射到 FPGA；QUEKUF 做 FPGA union-find decoder。([arXiv][4])

**[文献事实]** Bosonic/GKP hardware 侧也有 real-time bosonic QEC beyond break-even 与 autonomous GKP QEC 等实验背景，但这些不是本草稿所声称的 FPGA affine decoder。([Nature][15])

**[对草稿影响]** 草稿可以说“hardware-aware / FPGA-friendly / mock-backed software HIL”，不能说“FPGA demonstrated / embedded runtime validated / true real-time board result”。如果要和这些工作对话，必须报告 latency distribution、fixed-point precision、resource estimate、commit/fallback behavior，而不是只报 logical error rate。

---

## 4. Direct implications for this project

### 4.1 理论是否站得住

**结论：局部站得住，全局不能过度声称。**

草稿第 202–237 行把 affine rule 解释为 local Gaussian / linear-MMSE approximation，这是正确方向。对 Gaussian displacement error (e) 与 noisy syndrome (s)，
[
\hat e = K s + b,\quad K=\Sigma_{es}\Sigma_{ss}^{-1},\quad b=\mu_e-K\mu_s
]
在单一 branch、unwrapped、近似 Gaussian 的条件下是合理的。问题是 GKP syndrome 是 modular variable，posterior 一般是 wrapped / multimodal；在 lattice boundary 附近，linear-MMSE 不会是 Bayes-optimal。ML、closest-lattice-point、multimode GKP decoding 等文献已经给出更强的全局目标。([arXiv][18])

**必须修改的理论点：**

1. **定义 (\Delta_t) 到底是 error estimate 还是 corrective displacement。**
   如果 (\Delta_t) 是 decoder 输出的“估计误差”，则 (+\hat e) 合理；如果它是实际施加到 oscillator 的 correction displacement，则通常应为 (-\hat e)。草稿第 349–388 行的 (b_{\mathrm{target}}=\alpha(I-K)\mu) 需要与第 202–237 行的 (b=\mu_e-K\mu_s) 对齐。建议在 notation table 中写：
   [
   \hat e_t = K_t s_t + b_t,\qquad \Delta_t=-\hat e_t
   ]
   或反过来，但全文只能选一种。

2. **把 affine decoder 明确写成“fast-path approximation”。**
   推荐句式：

   > The affine rule is not intended to replace exact GKP maximum-likelihood decoding. It is a deterministic low-latency fast path whose parameters are adapted by a slower calibration loop.

3. **增加 branch / wrapping condition。**
   至少写出：当 (|e|\ll \sqrt{\pi}/2) 且 measurement noise 使 posterior 保持局部单峰时，affine approximation 合理；靠近 decision boundary 或高噪声时，teacher/ML baseline 应显著更强。

4. **解释为什么 residual 只修 (b) 而不是 (K)。**
   当前草稿第 111–125、411–448 行把 runtime mainline 设为 (K=K_{\rm teacher})，CNN 主要输出 (\delta b)。这在安全上很合理，但理论上要说明：mean drift / coherent bias 主要进入 intercept；gain/covariance drift 由 teacher 负责；residual (b) 是 bounded correction，降低 instability risk。

### 4.2 实验与噪声/漂移设定是否合理

**结论：四类 drift scenario 合理，但还不够 paper-grade。**

草稿第 252–314 行包括 finite squeezing、measurement noise、coherent/slow bias、anisotropic/rotated covariance，并设置 static bias、linear ramp、step、periodic drift。作为 first-pass benchmark 合理；但若投稿，需要更接近真实 bosonic/GKP 系统和 decoder mismatch 的设置。

建议新增：

1. **有限 squeezing 以 dB 报告。**
   用 8–14 dB 或更宽范围，不要只用抽象 (\sigma)。Surface-GKP 文献常以 squeezing dB 报告 threshold / overhead，便于对比。([arXiv][5])

2. **measurement noise 与 ancilla noise 分开扫。**
   草稿现在把 measurement noise 放入 (R_{\rm meas}) 很自然，但 benchmark 应分别扫 data displacement、measurement imprecision、ancilla preparation noise，否则 (K) 的 shrinkage 解释会混在一起。

3. **加入 temporal correlation / 1/f-like drift。**
   只用 step/ramp/periodic 容易过于干净。DGR 和 adaptive-weight literature 的重点就是 time-dependent / correlated noise mismatch；本项目若主打 drift-adaptive，也应包含 correlated drift。([arXiv][11])

4. **加入 histogram shot-budget stress test。**
   Slow loop 用 (32\times 32) histogram 时，窗口长度、update cadence、shot noise 会决定 estimator 是否稳定。建议报告 (W\in{128,512,2048,8192}) 或类似窗口，画 LER vs window size / update delay。

5. **加入 modulo aliasing / identifiability test。**
   对 mean drift 接近半格点、variance 较大或 branch 混合时，histogram 可能误导 teacher。这里可以借 Wagner syndrome-noise-estimation 的 identifiability 叙事。([arXiv][12])

### 4.3 Benchmark 和 baseline 是否足够强且公平

**结论：草稿当前 baseline 不够强。**

草稿第 466–493 行说明当前 frozen four-scenario five-mode software revalidation 中 `hybrid_residual_b` 排第一，但也承认不能泛化。这个边界必须保留。要变成 paper-grade，需要补齐以下 baseline：

**GKP physical-layer baselines：**

1. **Closest-integer / nearest-lattice baseline。**
   这是 GKP decoding 的最低基线；Noh 等 surface-GKP 工作也与 closest-integer style decoding 对比。([arXiv][5])

2. **Known-noise Gaussian Bayes / ML baseline。**
   作为 oracle upper bound：知道真实 (\theta_t) 时的 best local affine 或 exact wrapped-Gaussian posterior decoder。

3. **Static calibrated affine baseline。**
   初始 ((K_0,b_0)) 固定不更新。这个 baseline 会回答 drift-adaptation 的必要性。

4. **Periodic oracle recalibration baseline。**
   每隔 (M) epoch 用真实 noise labels 或大样本重估 ((K,b))，作为 slow-loop oracle。

5. **Teacher-only baselines。**
   moment teacher、EKF/UKF/RLS、particle filter；草稿第 390–409 行已经列出，但实验必须作为正式 comparators，而不是 architecture description。

6. **Residual variants。**
   no residual、(\delta b)-only、(\delta K+\delta b)、direct CNN-to-((K,b))、FiLM-conditioned affine head、teacher features removed、histogram removed、EMA removed、clipping removed。

**Outer-code / soft-information baselines：**

如果论文涉及 surface-GKP 或 concatenated code，必须和 dynamic soft weights、QLDPC-GKP analog information、real-time soft information baselines 对齐。Borah、Berent、Raveendran、Noh 这些工作会成为审稿人自然想到的参照。([arXiv][7])

**Hardware / runtime baselines：**

当前只能写 software HIL。若要讲 FPGA-friendly，需要至少给出 fixed-point simulation、latency model、resource estimate、commit/fallback statistics。真实 FPGA 文献已经会报告 deterministic closed-loop latency、per-round decode time、feedback latency、resource/precision tradeoff；本项目不能只报 Python timing。([arXiv][4])

---

## 5. Recommended new papers to add

下面这些建议加入 bibliography，并在 related work 中分主题组织。

| 主题                                     | 推荐文献                                                                                                                                                                   | 为什么必须加                                                                            |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| GKP 基础                                 | Gottesman, Kitaev, Preskill, “Encoding a qubit in an oscillator,” Phys. Rev. A 64, 012310, arXiv:quant-ph/0008040, DOI 10.1103/PhysRevA.64.012310                      | 所有 GKP 工作基础引用。([arXiv][1])                                                        |
| GKP review                             | Grimsmo & Puri, “Quantum Error Correction with the GKP Code,” PRX Quantum 2, 020101, arXiv:2106.12989, DOI 10.1103/PRXQuantum.2.020101                                 | 有助于引入 finite-energy GKP、noise model、实验背景。([arXiv][19])                            |
| Analog GKP decoding                    | Fukui et al., “High-threshold fault-tolerant quantum computation with analog quantum error correction,” PRX 8, 021054, arXiv:1712.00294, DOI 10.1103/PhysRevX.8.021054 | 说明 analog information 早已用于 fault-tolerant GKP decoding。([arXiv][16])              |
| Surface-GKP                            | Noh, Chamberland, Brandão, PRX Quantum 3, 010315, arXiv:2103.06994, DOI 10.1103/PRXQuantum.3.010315                                                                    | 最近的 GKP soft-information 强近邻。([arXiv][5])                                         |
| Surface-GKP circuit-level              | Noh & Chamberland, Phys. Rev. A 101, 012316, arXiv:1908.03579, DOI 10.1103/PhysRevA.101.012316                                                                         | 噪声模型与 circuit-level baseline 相关。([arXiv][6])                                      |
| Multimode GKP decoding                 | Lin & Noh, “Closest Lattice Point Decoding for Multimode GKP Codes,” PRX Quantum 4, 040334                                                                             | 给 affine decoder 的“不是全局最优”提供参照。([APS Link][20])                                   |
| GKP capacity / ML decoding             | Lin & Noh, “Exploring quantum capacity … GKP and maximum likelihood decoding,” Phys. Rev. A 111, 052445, arXiv:2411.04277                                              | 用作 exact/ML decoding baseline 或 upper bound 参照。([arXiv][18])                      |
| QLDPC-GKP                              | Raveendran et al., Quantum 6, 767, arXiv:2111.07029, DOI 10.22331/q-2022-07-20-767                                                                                     | 说明 GKP analog info + QLDPC/min-sum 已有。([arXiv][9])                                |
| Bosonic QLDPC analog info              | Berent et al., PRX Quantum 5, 020349, arXiv:2311.01328, DOI 10.1103/PRXQuantum.5.020349                                                                                | 与 bosonic soft-information decoding 高度相关。([arXiv][8])                             |
| Circuit-level QLDPC-GKP soft info      | Borah et al., arXiv:2505.06385                                                                                                                                         | 与 real-time soft information / circuit-level benchmark 直接相关。([arXiv][7])          |
| Syndrome noise estimation              | Wagner et al., Phys. Rev. Research 3, 013292, arXiv:2010.02243                                                                                                         | 支撑 syndrome histogram identifiability / estimation discussion。([arXiv][12])       |
| Pauli-channel estimation from syndrome | Wagner et al., Quantum 6, 809, arXiv:2107.14252, DOI 10.22331/q-2022-09-19-809                                                                                         | online adaptation from syndrome statistics 的理论近邻。([arXiv][21])                    |
| Adaptive decoder weights               | Spitz et al., Adv. Quantum Technol. 1, 1800012, arXiv:1712.02360, DOI 10.1002/qute.201800012                                                                           | drift-adaptive decoder weight baseline。([arXiv][2])                               |
| Drifted/correlated noise               | DGR, arXiv:2311.16214                                                                                                                                                  | 直接撞车于 “drift-adaptive decoder priors”。([arXiv][11])                               |
| Calibrated decoder                     | Chen et al., PRL 128, 110504, DOI 10.1103/PhysRevLett.128.110504                                                                                                       | experimental calibrated decoder prior 参照。([APS Link][22])                         |
| Prior optimization                     | Sivak, Newman, Klimov, PRL 133, 150603, arXiv:2406.02700, DOI 10.1103/PhysRevLett.133.150603                                                                           | 草稿已引但 bibliographic venue 应改；也应借 LER-targeted prior optimization。([APS Link][13]) |
| Calibration-conditioned neural decoder | Stein et al., arXiv:2601.16123                                                                                                                                         | learned slow-context / fast-decoder 最近邻。([arXiv][3])                              |
| AI predecoder                          | Chamberland et al., arXiv:2604.12841                                                                                                                                   | modular learned low-latency QEC layer 最近邻。([arXiv][10])                           |
| FPGA neural decoder                    | Yang et al., arXiv:2605.04892                                                                                                                                          | 真实 FPGA neural QEC decoder，硬件 claim 必须对照。([arXiv][4])                             |
| Real-time FPGA QEC                     | Caune et al., arXiv:2410.05202                                                                                                                                         | HIL / latency / feedback reporting 参照。([arXiv][14])                               |
| FPGA local clustering                  | Ziad et al., arXiv:2411.10343                                                                                                                                          | adaptive hardware decoder 参照，尤其 leakage/control-signal adaptation。([arXiv][23])   |
| qLDPC FPGA decoder                     | Maurer et al., arXiv:2510.21600                                                                                                                                        | fixed/reduced-precision FPGA qLDPC decoder 参照。([arXiv][24])                       |
| Real-time bosonic QEC                  | Sivak et al., Nature 616, 50–55, DOI 10.1038/s41586-023-05782-6                                                                                                        | bosonic hardware context，但不能混同为本项目证据。([Nature][15])                               |
| Autonomous GKP QEC                     | Lachance-Quirion et al., PRL 132, 150607, DOI 10.1103/PhysRevLett.132.150607                                                                                           | GKP hardware/control context。([APS Link][25])                                     |

---

## 6. Benchmark and experiment recommendations

### 6.1 主实验应改成 “closed-loop logical performance under drift”

建议主表不再以 (K,b) regression error 为中心，而是：

| 指标                                               | 为什么必要                                                                     |
| ------------------------------------------------ | ------------------------------------------------------------------------- |
| Logical error rate / logical failure probability | QEC 最终目标；Sivak prior-optimization 工作也强调 proxy metrics 不足。([APS Link][13]) |
| Regret to oracle                                 | 衡量相对 known-(\theta_t) Bayes/ML 或 oracle affine decoder 的损失。               |
| Adaptation lag after step drift                  | 直接检验 slow loop 是否跟得上。                                                     |
| Overshoot / oscillation / rollback count         | 防止 residual 被写成稳定闭环但实际不稳定。                                                |
| Stale-parameter penalty                          | 对应草稿第 450–464 行的 staged commit / delayed update。                          |
| Saturation / clipping rate                       | 对应 Q4.20 fixed-point 与安全边界。                                               |
| Commit success / fallback frequency              | 体现 deployment contract，而不是只报模型准确率。                                        |
| Latency p50 / p95 / p99 / worst-case             | 真实-time decoder 文献都会关心 worst-case，不只是 average。([arXiv][4])                |
| Fixed-point vs float degradation                 | FPGA-friendly claim 的最低证据。                                                |

### 6.2 数据集设计

建议把 benchmark 拆成三层：

**Layer A：clean synthetic drift。**
保留草稿现有 static bias、linear ramp、step、periodic drift，但加入 seed sweep 与 confidence intervals。

**Layer B：unseen drift / distribution shift。**
训练不用的 drift：piecewise-smooth random walk、1/f-like drift、sudden covariance rotation、measurement-noise burst、bias plus variance coupling、hidden mode-dependent drift。

**Layer C：adversarial / safety stress。**
低 histogram shot budget、delayed updates、stale teacher state、teacher miscalibration、branch aliasing、saturation-heavy regime、drift faster than slow window。

### 6.3 Baseline 套件

最低建议 baseline：

1. nearest-lattice / closest-integer GKP decoder；
2. static calibrated affine decoder；
3. oracle affine with true (\theta_t)；
4. exact wrapped-Gaussian Bayes / ML decoder，至少单模或小规模；
5. moment teacher only；
6. EKF/UKF/RLS teacher only；
7. particle teacher only，如果计算允许；
8. CNN direct-to-((K,b))；
9. teacher + residual (\delta b)；
10. teacher + residual (\delta K,\delta b)；
11. FiLM-conditioned affine head，作为 Stein-style inspired baseline；
12. no histogram / no teacher / no EMA / no clipping ablations。

### 6.4 统计方法

建议所有核心结果使用 paired seeds：同一 noise trajectory 同时喂给所有 decoders，报告 paired bootstrap CI 或 Wilson/Beta-binomial CI。不要只报告 frozen-set best rank。草稿已经写了 “frozen-set evidence supports bounded software revalidation”，这应该变成 methodology section 的 guardrail，而不是隐藏在 appendix。

### 6.5 Runtime / HIL 证据分层

建议清楚分四档，论文里只 claim 已经达到的档：

| 证据等级                                   | 可以声称                                                      | 不能声称                                |
| -------------------------------------- | --------------------------------------------------------- | ----------------------------------- |
| Python / NumPy simulation              | algorithmic proof-of-concept                              | real-time, embedded, FPGA           |
| mock-backed software HIL               | interface-level HIL, staged commit logic under mock       | real board, hardware timing closure |
| `.tflite` or embedded runtime restored | actual runtime equivalence / quantized inference evidence | FPGA board result                   |
| FPGA board                             | real hardware latency/resource/closed-loop timing         | 仍需避免超出实验设置                          |

当前草稿只能处在前两档。不要默认 `.tflite` runtime 已恢复，也不要把 “FPGA-friendly” 写成 “FPGA-validated”。

---

## 7. Paper positioning and claims

### 7.1 推荐定位

建议标题/摘要围绕：

> **Teacher-anchored residual calibration for deployment-constrained drift-adaptive affine GKP decoding**

或更保守：

> **A software-HIL study of drift-adaptive affine GKP decoding with teacher-anchored residual calibration**

这个定位把 novelty 放在“组合与约束”上，而不是放在任何一个已经拥挤的单点上。

### 7.2 可以安全声称的内容

**[在当前证据边界下可声称]**

1. 提出一个 two-timescale architecture：fast deterministic affine GKP correction，slow histogram/teacher/residual calibration。
2. 把 adaptive decoding 参数限制为 hardware-friendly ((K,b)) fast path，并加入 quantization、clipping、double-buffer stage-and-commit、fallback diagnostics。
3. 在 mock-backed software HIL 与 frozen-set revalidation 中，`hybrid_residual_b` 在当前四类 drift / 五模式设置下优于若干内部 baseline。
4. 该结果支持进一步真实 runtime / board validation，但尚不构成真实 FPGA 或 `.tflite` runtime 证明。

### 7.3 应避免的 claim

这些 claim 风险很高，建议删掉或改写：

| 高风险 claim                                       | 原因                                                                                       |
| ----------------------------------------------- | ---------------------------------------------------------------------------------------- |
| “first adaptive neural QEC decoder”             | Stein、Chamberland、Bausch、Varbanov 等已经有 learned/adaptive neural decoder 相关工作。([arXiv][3]) |
| “first calibration-conditioned decoder”         | Stein 2026 已经非常明确。([arXiv][3])                                                           |
| “first syndrome-statistics drift adaptation”    | Wagner、Spitz、DGR、Sivak 等已经覆盖 syndrome statistics / adaptive priors。([arXiv][12])         |
| “first FPGA QEC decoder”                        | 已有多个 real-time FPGA QEC decoders。([arXiv][4])                                            |
| “real-time FPGA demonstrated”                   | 当前草稿证据不是 true board。                                                                     |
| “true `.tflite` runtime validated”              | 草稿自己说未恢复 true `.tflite` runtime validation。                                              |
| “mechanism proves monotonic causal closed loop” | 草稿第 466–493 行已承认 post-T55 mechanism story 需要 hedge。                                      |

### 7.4 建议的 abstract 改写方向

现在 abstract 的方向是对的，但还可以更硬地限定边界。建议加入类似句子：

> We do not claim a new maximum-likelihood GKP decoder or a board-level FPGA demonstration. Instead, we study a deployment-constrained calibration layer whose fast path remains an affine fixed-point rule, while a slower teacher-anchored residual loop updates bounded calibration parameters from syndrome histograms.

再加一句：

> Current evidence is limited to mock-backed software HIL and frozen-set revalidation; true embedded runtime and board-level timing closure are left as validation targets.

### 7.5 Related work 重构建议

建议 related work 不按时间写，而按“撞车风险”写成五段：

1. **GKP analog / bosonic soft information.**
   GKP、Fukui、Noh surface-GKP、QLDPC-GKP、Berent、Borah、Raveendran。
2. **Noise estimation and adaptive decoder priors.**
   Wagner、Spitz、DGR、Chen、Sivak、CaliScalpel。
3. **Learned and calibration-conditioned QEC decoders.**
   Stein FiLM、Chamberland AI predecoder、Bausch、Varbanov/Gicev 等。
4. **Real-time and FPGA QEC decoders.**
   Yang、Caune/Riverlane/Rigetti、Ryan-Anderson、Ziad、Maurer、QUEKUF。
5. **This work’s niche.**
   “Unlike these works, we keep the per-shot path as a bounded affine GKP correction and restrict learning to a teacher-anchored residual calibration layer evaluated under explicit software-HIL/fixed-point/stale-update constraints.”

---

## 8. 直接回答用户指定问题

### Q1. 是否已有工作非常接近“从 GKP syndrome histogram 估计漂移并更新仿射 decoder 参数 `(K,b)`”？

**结论：没有找到完全相同的工作，但有三组非常接近。**

1. **GKP analog soft-information / surface-GKP** 很接近，因为它们利用 GKP analog information 和 syndrome history 动态设置 decoder weights；但它们更新的是 surface-code matching/graph weights 或 outer decoder priors，不是 physical GKP affine parameters ((K,b))。([arXiv][5])
2. **Syndrome-statistics noise estimation / adaptive priors** 很接近，因为它们从 syndrome measurements 或 decoding statistics 估计 noise / weights；但它们主要在 Pauli/stabilizer/MWPM graph setting 中，不是 modular continuous GKP histogram → affine displacement rule。([arXiv][12])
3. **Calibration-conditioned neural decoders** 很接近，因为它们使用慢变化 calibration context 条件化低延迟 decoder；但不是 GKP histogram 更新 ((K,b))。([arXiv][3])

所以，窄 claim 可以成立：**“to our knowledge, no prior work has specifically studied histogram-driven teacher-residual adaptation of affine GKP fast-path parameters under deployment constraints.”** 但必须加 “to our knowledge” 和边界条件。

### Q2. 是否已有工作提出 teacher + neural residual 用于 QEC runtime adaptation？

**结论：我没有找到 exact match。**

最接近的是 Stein 的 calibration-conditioned FiLM decoder、Chamberland 的 AI predecoder、Sivak 的 prior optimization、DGR 的 adaptive reweighting，以及更广泛的 neural QEC decoder 文献。它们分别覆盖 calibration conditioning、modular learned decoder、LER-targeted prior optimization、drifted noise reweighting，但没有看到明确的 “classical teacher estimator anchors runtime parameters, neural net predicts bounded residual correction to ((K,b))” 这一完整组合。([arXiv][3])

这意味着本项目可以把 **teacher-anchored residual calibration** 作为主要差异化点，但必须通过 ablation 证明：teacher-only 不够，direct neural 不够稳定，bounded residual (b) 在 drift 与 fixed-point constraints 下更稳。

### Q3. 是否已有工作在 FPGA / embedded / real-time 约束下做过类似 GKP 或 bosonic decoder？

**结论：真实-time bosonic QEC 有，FPGA/embedded QEC decoder 有，但我没有找到与本项目相同的 FPGA/embedded GKP histogram-to-affine decoder。**

已有 real-time bosonic QEC beyond break-even 与 autonomous GKP QEC 等 bosonic/GKP hardware control 工作；另一方面，surface-code/qLDPC real-time FPGA decoders 已经很强，包括 FPGA neural decoder、Rigetti/Riverlane decoder、local clustering decoder、Gross-code FPGA decoder、FPGA union-find decoder。([Nature][15])

因此本项目可以说：**hardware-aware GKP affine residual calibration appears underexplored**。但不能说已经完成 FPGA demonstration，也不能把 mock HIL 与真实 FPGA decoder 工作并列比较。

---

## 9. 最可能撞车风险与最有希望的差异化空间

### 最可能撞车风险

1. **Stein 2026 FiLM decoder：** 如果你写 “slow calibration context + neural decoder for low latency”，会被认为已被覆盖。
2. **Chamberland 2026 AI predecoder：** 如果你写 “modular AI predecoder using syndrome statistics for time-varying noise”，会非常接近。
3. **DGR / Spitz / Wagner / Sivak：** 如果你写 “syndrome statistics estimate drift and adapt decoder priors”，会撞 adaptive prior literature。
4. **Noh / Borah / Berent / Raveendran：** 如果你写 “GKP analog soft information improves decoding”，会撞 GKP soft-decoding literature。
5. **Yang / Rigetti-Riverlane / Ziad / Maurer：** 如果你写 “FPGA real-time QEC decoder”，当前证据会被真实硬件工作压住。

### 最有希望的差异化空间

1. **Affine GKP fast-path contract：** 每个 shot 只执行 (\Delta=Ks+b)，可 fixed-point、clipping、double-buffer；这比 generic neural decoder 更可部署。
2. **Teacher-anchored residual，而不是 full neural decoder：** 让 classical teacher 保持 stability，CNN 只修 residual，尤其是 (b)。
3. **Histogram slow loop 专门服务 physical GKP calibration：** 与 outer-code graph reweighting 区分。
4. **Explicit stale-update / commit / fallback / fixed-point analysis：** 这能让论文从“又一个 neural decoder”变成“deployment-constrained calibration architecture”。
5. **诚实的证据边界：** 把 mock-backed software HIL 写清楚，反而比过度声称更可信。

---

## 10. 最短行动清单

1. **修 bib：** `sivak2024` venue 改成 PRL 133, 150603；加入上表文献。
2. **改 abstract：** 第一段保留 GKP drift；第二段强调 affine fast path；最后一句明确 software-HIL-only。
3. **补 theory subsection：** local Gaussian LMMSE、modular posterior failure cases、(\Delta) 符号、(b)-residual rationale。
4. **补 related work 五段式。** 不要把 Stein/Chamberland/Yang 只放 bibliography，要正面比较。
5. **补 benchmark 表：** static affine、oracle affine、ML/Bayes、teacher-only、FiLM-style、direct CNN、residual variants、fixed-point variants。
6. **补 safety/runtime table：** stale updates、commit/fallback、saturation、latency p99/worst、fixed-point degradation。
7. **把机制叙事 hedge：** 写成 “consistent with residual bias calibration under these scenarios”，不要写成 “proves monotonic causal closed-loop adaptation”。

[1]: https://arxiv.org/abs/quant-ph/0008040?utm_source=chatgpt.com "[quant-ph/0008040] Encoding a qubit in an oscillator"
[2]: https://arxiv.org/abs/1712.02360?utm_source=chatgpt.com "Adaptive weight estimator for quantum error correction"
[3]: https://arxiv.org/abs/2601.16123?utm_source=chatgpt.com "Calibration-Conditioned FiLM Decoders for Low-Latency Decoding of Quantum Error Correction Evaluated on IBM Repetition-Code Experiments"
[4]: https://arxiv.org/abs/2605.04892?utm_source=chatgpt.com "Real-time Surface-Code Error Correction Using an FPGA-based Neural-Network Decoder"
[5]: https://arxiv.org/abs/2103.06994?utm_source=chatgpt.com "Low overhead fault-tolerant quantum error correction with the surface-GKP code"
[6]: https://arxiv.org/abs/1908.03579?utm_source=chatgpt.com "Fault-tolerant bosonic quantum error correction with the surface-GKP code"
[7]: https://arxiv.org/abs/2505.06385?utm_source=chatgpt.com "Fault Tolerant Decoding of QLDPC-GKP Codes with Circuit Level Soft Information"
[8]: https://arxiv.org/abs/2311.01328?utm_source=chatgpt.com "Analog information decoding of bosonic quantum LDPC ..."
[9]: https://arxiv.org/abs/2111.07029?utm_source=chatgpt.com "Finite Rate QLDPC-GKP Coding Scheme that Surpasses the CSS Hamming Bound"
[10]: https://arxiv.org/abs/2604.12841?utm_source=chatgpt.com "Fast and accurate AI-based pre-decoders for surface codes"
[11]: https://arxiv.org/abs/2311.16214?utm_source=chatgpt.com "DGR: Tackling Drifted and Correlated Noise in Quantum Error Correction via Decoding Graph Re-weighting"
[12]: https://arxiv.org/abs/2010.02243?utm_source=chatgpt.com "Optimal noise estimation from syndrome statistics of quantum codes"
[13]: https://link.aps.org/doi/10.1103/PhysRevLett.133.150603?utm_source=chatgpt.com "Optimization of Decoder Priors for Accurate Quantum Error ..."
[14]: https://arxiv.org/abs/2410.05202?utm_source=chatgpt.com "Demonstrating real-time and low-latency quantum error correction with superconducting qubits"
[15]: https://www.nature.com/articles/s41586-023-05782-6?utm_source=chatgpt.com "Real-time quantum error correction beyond break-even"
[16]: https://arxiv.org/abs/1712.00294?utm_source=chatgpt.com "High-threshold fault-tolerant quantum computation with analog quantum error correction"
[17]: https://arxiv.org/abs/2107.13589?utm_source=chatgpt.com "Improved quantum error correction using soft information"
[18]: https://arxiv.org/abs/2411.04277?utm_source=chatgpt.com "Exploring the quantum capacity of a Gaussian random displacement channel using Gottesman-Kitaev-Preskill codes and maximum likelihood decoding"
[19]: https://arxiv.org/abs/2106.12989?utm_source=chatgpt.com "Quantum Error Correction with the Gottesman-Kitaev-Preskill Code"
[20]: https://link.aps.org/doi/10.1103/PRXQuantum.4.040334?utm_source=chatgpt.com "Closest Lattice Point Decoding for Multimode Gottesman ..."
[21]: https://arxiv.org/abs/2107.14252?utm_source=chatgpt.com "Pauli channels can be estimated from syndrome measurements in quantum error correction"
[22]: https://link.aps.org/doi/10.1103/PhysRevLett.128.110504?utm_source=chatgpt.com "Calibrated Decoders for Experimental Quantum Error Correction"
[23]: https://arxiv.org/abs/2411.10343?utm_source=chatgpt.com "Local Clustering Decoder: a fast and adaptive hardware decoder for the surface code"
[24]: https://arxiv.org/abs/2510.21600?utm_source=chatgpt.com "Real-time decoding of the gross code memory with FPGAs"
[25]: https://link.aps.org/doi/10.1103/PhysRevLett.132.150607?utm_source=chatgpt.com "Autonomous Quantum Error Correction of Gottesman-Kitaev ..."
