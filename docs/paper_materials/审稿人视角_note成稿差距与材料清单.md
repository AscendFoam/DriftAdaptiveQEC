# 审稿人视角 note 成稿差距与材料清单

## 0. 评估范围与边界

本文从预投稿审稿人的角度评估 `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex` 距离一篇完整论文还差什么。评估使用三类本地规则：

- `nature-reader`：把当前 `.tex` 当作已有手稿文本读取，先抽取结构、主张、证据锚点和缺失块，而不是只做摘要。
- `nature-writing`：按研究/算法系统论文的 claim-evidence chain 审查题名、摘要、引言、方法、实验、讨论与结论。
- `nature-reviewer`：按 Nature-style 审稿口径，从 originality、scientific importance、interdisciplinary readership、technical soundness、nonspecialist readability 五个轴给出三份不同权重的审稿意见。

本文不新增实验，不升级任何证据等级，不改变当前 frozen mainline。当前仓库事实仍以 `docs/04_task_board.md`、`docs/07_handoff.md`、`docs/08_risks_and_open_questions.md` 和 `docs/paper_materials/paper_frozen_mainline_handoff_packet.md` 为准。

## 1. 共享事实基线

当前 note 已经不是空白草稿。它有完整论文骨架：标题、摘要、Introduction、Summary of Contributions、GKP 背景、Noise and Drift Model、Model Architecture、Related Work、Experimental Setup、Numerical Results、Discussion、Conclusion 和 BibTeX 风格参考文献。

当前 note 最强、最安全的中心命题是：

> Drift-adaptive GKP correction can be organized as a dual-loop affine calibration problem: a deterministic fast path executes \(\Delta=Ks+b\), while a slower teacher/statistical/learned calibration loop updates the low-dimensional runtime surface from syndrome statistics.

当前最强证据层是：

- `T24` frozen-set mock-backed software-HIL 四场景五模式结果：`hybrid_residual_b` 在四个 frozen scenarios 中均为 winner，UKF 为 runner-up。
- `FR6/FR7` 类机制与 ablation 材料：支持 descriptive mechanism reading，但不支持 causal closure，也不支持 teacher necessity。
- `FR8/statcalib`：支持 supplement-side extension lane 和 no-promotion/no-unique-threshold 结论，不支持 promoted mature comparator。
- `T50`：支持 canonical training/material chain + one clean CPU-only bounded train/eval rerun，不支持 full reproducibility。
- `T48`：支持 selected preserved `.tflite` artifacts 的 isolated current-host true runtime，不支持 default environment、HIL closure 或 deployment closure。
- `T49/T71/T72`：支持 read-only real-board gate/regeneration/provenance with current-host `NO_GO`，不支持 real-board execution success。

因此，当前实验是“有边界可信”的，不是“全面可信”。它足以支撑一篇有诚实边界的 simulation/material-first 系统方法稿雏形；不足以支撑真实硬件论文、deployment-ready 论文、paper-grade expanded benchmark 论文或广义 SOTA 解码器论文。

## 2. 总体审稿判断

### 当前状态

这份 note 已经完成了内部治理意义上的 frozen mainline handoff，但还没有达到外部论文意义上的 submission-ready manuscript。它的优点是边界非常诚实，缺点是审稿人会明显感觉它仍保留大量内部任务台账语言，并且核心实验矩阵偏窄。

### 是否像一篇完整论文

结构上接近完整论文；证据上仍像一份“项目主线材料包 + 论文草稿”。审稿人最可能的判断是：

- 方法思想清楚，有潜在价值。
- 当前主结果范围太窄，只能证明 frozen software-HIL setting 下的 bounded result。
- 论文写作过度依赖内部 task/provenance 语言，外部读者不应被迫理解 `T24`、`T70`、`FR8`、`NO_GO`、`extension lane` 这些治理术语。
- 如果目标是一般量子工程/系统方法期刊，需要补强 benchmark、baseline、统计、图表和 reproducibility。
- 如果目标是 Nature-style broad readership，当前还缺少“outstanding importance / arresting result / far-reaching implication”的实验证据。

### 核心结论

当前 note 距离“可继续人工精修的内部主线稿”已经很近；距离“完整可投稿论文”还差一个 evidence-hardening wave；距离“高质量/高影响论文”还差更强 benchmark、理论 baseline、统计协议、bit-accurate/runtime 证据或真实硬件证据。

## 3. Reviewer 1：技术可信度与实验严谨性优先

### Overall assessment

稿件提出的 dual-loop affine calibration framework 是合理的，且比“CNN decoder”叙事更可信。作者已经主动限制硬件、`.tflite`、training reproducibility 和 `statcalib` 的证据边界，这是很大的优点。

但当前 case 还没有被充分建立。主结果依赖四个 hand-designed drift scenarios、五个 mainline modes、mock-backed software-HIL 和有限 repeats。审稿人会认为这是一个 promising prototype，而不是已经完成验证的 adaptive GKP decoding framework。

### Who would be interested

感兴趣的读者包括：

- 研究 GKP analog syndrome / soft information 的量子纠错读者；
- 研究 decoder calibration、noise estimation、drift adaptation 的读者；
- 研究 hardware-aware QEC runtime 的工程读者；
- 希望把 neural module 限制在 slow loop、保留 deterministic fast path 的系统设计读者。

但若稿件继续使用 FPGA / runtime / deployment 语言，就必须面对真实 hardware QEC 文献的 latency、resource、closed-loop timing 标准。

### Major strengths

- 把 per-shot path 限制为 \(\Delta=Ks+b\)，把学习和统计估计放在 slow loop，系统边界清楚。
- 主文主动承认 affine fast path 是 local approximation，而不是 full wrapped posterior decoder。
- `statcalib` 结果没有被强行 promoted，这避免了过度包装。
- real-board 与 `.tflite` 边界写得诚实，没有把 gate/provenance 写成 execution。

### Major concerns

1. **主 benchmark 太窄**：四个 frozen scenarios 不足以支撑“drift-adaptive GKP decoding”这个宽泛命题。
2. **缺少强理论 baseline**：nearest-lattice、wrapped-Gaussian、oracle affine、known-noise affine 等 baseline 没有实际结果。
3. **缺少统计可信度**：主表只有 mean LER，没有 confidence interval、effect size、paired-seed test、bootstrap 或 per-seed variability。
4. **机制解释仍弱**：FR6/FR7 只支持 descriptive evidence，不能解释为什么 hybrid residual branch 在四场景都赢。
5. **runtime 证据不足**：fast path 低延迟是架构推断，不是 measurement；board/resource/timing 均缺。
6. **training reproducibility 不完整**：T50 只是一条 clean CPU-only bounded rerun；T90 尚未完成时，不应写 repeated-run stability。

### Technical failings to address

- 给出 theory-to-config bridge：每个 drift scenario 如何对应 \(\theta_t^{noise}\)、dataset builder、config 和 physical interpretation。
- 对 `physics/` 中 syndrome wrapping、logical error boundary、noise streams 和 finite-energy approximation 做独立审计。
- 对 runner fallback、artifact type、stub/runtime path 做审计，证明结果没有静默 fallback。
- 对主结果添加 CI/seed variability。
- 增加 oracle/nearest-lattice/wrapped-Gaussian 或至少小规模 sanity baseline。
- 增加 unseen drift holdout，否则“drift adaptation”容易被看成对四个 fixed trajectories 的调参。
- 若保留 FPGA/low-latency 语言，至少给出 bit-accurate fixed-point simulation、op count、latency proxy 或 synthesizable fast-path resource estimate。

### Nature-style criteria

- Originality：有系统组织上的新意，但必须更清楚地区分它与 adaptive decoder weights、calibrated decoders、FiLM/AI pre-decoders、GKP analog information 的差别。
- Scientific importance：目前是 field-local useful，不足以证明 outstanding importance。
- Interdisciplinary readership：技术主题有交叉潜力，但当前 prose 仍太内部化。
- Technical soundness：bounded software-HIL 层可信；广义 claim 未建立。
- Readability：对内部读者清楚，对外部读者仍有治理术语负担。

### Recommendation posture

Promising but not yet established. 建议在补完 expanded benchmark/provenance/statistics 之前，不按高影响完整论文投出。

## 4. Reviewer 2：创新性、重要性与论文定位优先

### Overall assessment

稿件最好的创新点不是 CNN，而是把 GKP drift adaptation 写成 low-dimensional affine calibration contract。这个定位比“neural GKP decoder”更稳，也更容易与 hardware-aware QEC 联系起来。

但是，稿件当前的 novelty case 仍偏防守。作者反复说明“不是什么”，但还没有充分说明“它改变了什么”。审稿人会问：这个 framework 给 GKP decoding 社区带来的新知识是一个架构哲学，还是一个经验证明有效的算法？

### Major strengths

- 题名和摘要已经从 “CNN + FPGA” 收窄为 dual-loop teacher-anchored residual calibration，这很正确。
- Introduction 能把 analog GKP soft information、adaptive priors、learned low-latency QEC modules 和 FPGA decoder 文献连接起来。
- Discussion 清楚解释了为什么 `statcalib` 强不推翻项目，而是支持 calibration-contract story。

### Major concerns

1. **novelty 仍靠 framing 多于证据**：如果最强结果只是四场景 software-HIL ranking，审稿人可能认为这是一个工程组合，而非新科学发现。
2. **Related Work 仍需更硬的 collision analysis**：目前引用了很多方向，但需要明确每组文献“解决了什么、本项目没有解决什么、本项目真正补了什么”。
3. **缺少外部 benchmark 标准**：没有与 GKP analog decoder、adaptive weight estimator、calibrated decoder 或 learned decoder 的可比任务连接。
4. **statcalib 的强结果削弱 CNN headline**：这已经被 note 诚实处理，但论文中心必须从“CNN wins”彻底转向“runtime affine calibration contract works”。
5. **Nature-style significance 不够**：如果没有硬件证据、broader drift、oracle baseline 或 theoretical insight，影响范围更像量子工程/应用方法，而不是广义高影响发现。

### Required positioning changes

- 把中心贡献写成“physical-layer GKP affine calibration under drift”，不要写成“CNN-assisted decoder superiority”。
- 把 `hybrid_residual_b` 的四场景 winner 写成 one evidence layer，不要让它独占论文意义。
- 把 `statcalib` 作为 support for contract, not promoted comparator。
- 把 hardware / FPGA 改为 “runtime-bounded design target and boundary evidence”，除非未来补真实 timing/resource。
- Related Work 必须加入一张 topic-based comparison table，列出 analog GKP soft information、adaptive prior/noise estimation、learned decoder、real-time FPGA decoder 与本项目的差异。

### Nature-style criteria

- Originality：中等，有组合式新意；需要更强 proof of non-obviousness。
- Scientific importance：目前不够强；需要展示该 contract 对 GKP experiments 或 QEC runtime 有明确改变。
- Interdisciplinary readership：量子纠错 + 硬件系统读者可能感兴趣，但非专业读者需要更清晰的 schematic 和 summary paragraph。
- Technical soundness：主张边界诚实，但证据不够广。
- Readability：论文故事需要减少内部 task-history 语言。

### Recommendation posture

Potentially publishable after substantial reframing and evidence strengthening. 若保持当前证据规模，更适合量子工程/系统方法 venue，而不是广义 high-impact venue。

## 5. Reviewer 3：跨学科可读性与成稿质量优先

### Overall assessment

这份 note 的问题不是“不会写”，而是“太像内部治理后的 frozen note”。外部审稿人不应看到太多 `T24`、`T70`、`NO_GO`、`extension lane`、`frozen mainline`、`bounded manual finish` 等内部术语。它们对仓库治理必要，但正式论文需要转译成读者语言。

### Major strengths

- 摘要已经清楚交代 finite-energy GKP、drift、dual-loop affine fast path、teacher residual、bounded result 和 no-overclaim。
- Background 和 Method 的逻辑比较完整。
- Discussion 有明确边界意识，避免硬件过度宣称。

### Major concerns

1. **读者入口仍不够顺**：Introduction 第一屏还需要更强的 broad problem statement 和 one-sentence contribution。
2. **内部证据术语太多**：`frozen`, `T24`, `FR8`, `NO_GO`, `extension lane`, `no-promotion gate` 应转为论文术语，如 `predeclared four-scenario software-HIL benchmark`, `separate supplementary calibration analysis`, `hardware-readiness gate`.
3. **Results 过长且层级混杂**：主结果、ablation、statcalib supplement、future lanes、runtime boundary 都放在 Numerical Results，读者容易误读为同一证据等级。
4. **follow-up wave tables 不像 Results**：Wave A/B 更像 future work or roadmap，不应在主 Results 中占据太多篇幅。
5. **参考文献格式未投稿化**：thebibliography 中多篇 preprint/未来年份/题名级条目需要 DOI、arXiv、完整作者、版本核查。
6. **缺少 figure-first story**：目前主要靠表格和解释，缺少 Fig. 1 架构图、Fig. 2 GKP/noise/drift schematic、Fig. 3 主结果图、Fig. 4 adaptation trace。

### Required readability changes

- 在摘要或 Introduction 加一个非内部读者能理解的 summary paragraph：问题、方法、结果、边界。
- 把 `Summary of Contributions` 压缩为 3-4 个贡献，而不是 6 个内部层级贡献。
- 把 supporting/boundary/future material 移到 Appendix/Supplement/Future Work，不挤在 main Results。
- 每个表标题都说明 sample/protocol/metric direction/证据等级。
- 建一张 terminology ledger：GKP, software-HIL, fast path, slow loop, teacher, residual calibration, statistical calibration, runtime gate, real-board gate。

### Nature-style criteria

- Originality：读者需要更快看到“new move”。
- Scientific importance：当前 significance 被边界说明稀释。
- Interdisciplinary readership：需要图和术语转译。
- Technical soundness：诚实但过于内部化。
- Readability：需要 substantial reader-facing rewrite。

### Recommendation posture

Readable draft foundation, not yet a polished manuscript. 建议先做“读者化重构”，但不要在证据未补强前把措辞改得更强。

## 6. Cross-review synthesis

### Consensus strengths

- 中心 framing 从 CNN 转向 dual-loop affine calibration 是正确方向。
- 证据边界非常诚实，减少了硬件、`.tflite`、training reproducibility、`statcalib` 的 overclaim 风险。
- 现有主线结果可作为 simulation/material-first paper 的基础。
- 机制、ablation、runtime 和 real-board gate 已有材料入口，后续可逐步补强。

### Consensus technical risks

- frozen benchmark 太窄。
- 缺少 strong theoretical baselines。
- 缺少 unseen drift holdout。
- 缺少 statistical confidence。
- 缺少 runner/fallback audit。
- 缺少 bit-accurate/runtime/hardware measurement。
- training reproducibility 和 `.tflite` portability 仍未闭合。
- `statcalib` 太强但不能 promoted，必须解释得非常清楚。

### Broad-interest readout

当前 manuscript 的 broad-interest case 还没有建立。它对 GKP/QEC systems 社区有明确价值，但要达到高影响论文，需要至少满足以下之一：

- 更强理论 insight：证明 low-dimensional affine calibration 在某类 drift/noise 下有可解释优势；
- 更强实验 breadth：expanded benchmark + strong baselines + holdout drift + CI；
- 更强 systems proof：bit-accurate fixed-point / true runtime / board timing/resource；
- 更强 reproducibility：clean-provenance reruns, artifact manifests, audit helper, cross-host or repeated-run evidence。

## 7. 当前实验可信度判断

### 可信的部分

1. **T24 frozen software-HIL ranking 可信**  
   作为 mock-backed software-HIL frozen set，四场景结果可引用。它是当前主文最稳的结果表。

2. **FR6/FR7 作为 descriptive support 可信**  
   可以支持“histogram-delta features matter”“teacher-side inputs do not support a simple necessity story”“mechanism remains descriptive”。

3. **FR8/statcalib 作为 supplement-side extension lane 可信**  
   可以支持“same affine contract can host non-neural calibration”，但必须保留 no-promotion/no-unique-threshold。

4. **T48/T50/T72 作为 boundary/support material 可信**  
   它们可以进入 appendix/supplement 的 evidence boundary table，但不能进入主结果 claim。

### 不可信或尚未足够可信的部分

1. **不可信为 broad benchmark**  
   四场景 frozen set 不能代表 drift adaptation 全空间。

2. **不可信为 causal mechanism proof**  
   干预 mixed and mostly harmful，不能写成 root cause solved。

3. **不可信为 hardware/FPGA validation**  
   没有 real-board execution、latency、resource、throughput、bitstream/DMA/AXI contract evidence。

4. **不可信为 deployment closure**  
   isolated `.tflite` runtime 不是 default-env portability、HIL closure 或 embedded deployment。

5. **不可信为 full reproducibility**  
   目前只有 one bounded rerun。T90 完成前 repeated-run consistency 也不能写成事实。

6. **不可信为 promoted statcalib comparator**  
   statcalib 的强数值只能作为 extension evidence；当前 no-promotion gate 是必须保留的结论。

## 8. 距离完整论文的主要缺口

### A. 论文主张缺口

- 需要把核心 claim 固定成一句可检验命题：dual-loop affine calibration improves drift-adaptive GKP correction under a predeclared software-HIL protocol while preserving deterministic fast-path constraints。
- 需要删除或迁移内部 governance language。
- 需要把 “software-HIL result / appendix support / supplement gate / blocked future surface” 用论文读者语言重写。
- 需要明确 target venue。如果目标是 QST/TQE/EPJ Quantum Technology，simulation/material-first route 可能可行；如果目标是 FPGA/hardware venue，当前证据不足。

### B. 理论缺口

- GKP finite-energy/noisy auxiliary state/measurement noise 的模型还需要更严谨地映射到代码。
- Local affine LMMSE 推导需要说明在哪些 branch/variance/measurement assumptions 下成立。
- Wrapped posterior/multimodal posterior 的 failure regime 需要更明确。
- `theta_noise` 与 benchmark config 的 bridge 需要表格化。
- logical-error proxy 与真实 logical channel 的差别需要写清。
- adaptation lag、commit cadence、stale parameter penalty、fallback frequency 应有数学定义。

### C. 实验缺口

- expanded benchmark protocol：更多 drift families、holdout drift、train/eval seed split、predeclared stopping。
- stronger baselines：nearest-lattice, wrapped-Gaussian, oracle affine, known-noise affine, window variance, EKF/UKF/RLS, CNN-only, teacher-only, statcalib。
- statistics：mean/std/CI/effect size/paired tests/bootstrap。
- ablation：histogram windows, deltas, teacher state, residual branch, context length, update cadence, clipping, EMA, parameter bank。
- mechanism：trace figures plus counterfactual/intervention designs that do not rely on one mixed lower-clip test。
- runtime：fixed-point degradation, saturation/overflow, source-vs-tflite drift, slow-loop update cost, latency proxy。
- reproducibility：T90 same-host repeated-run consistency，后续 cross-host / GPU / Linux portability 才能更强。

### D. 图表缺口

至少需要以下正式图表：

- Fig. 1：架构图，GKP syndrome -> affine fast path -> slow calibration -> parameter bank；硬件 blocked surface 用虚线。
- Fig. 2：GKP/noise/drift schematic，说明 modulo syndrome、finite-energy uncertainty、effective drift。
- Fig. 3：主 benchmark 图，四场景 LER + CI。
- Fig. 4：adaptation trace，展示 drift trajectory、\(K,b\)、commit points、residual/failure relation。
- Fig. 5：ablation/mechanism，展示 feature/teacher/residual branch 的 bounded support。
- Fig. 6：runtime/evidence boundary，显示 software-HIL、isolated `.tflite`、real-board gate 的证据层级。
- Table 1：related work comparison。
- Table 2：claim-evidence map。
- Table 3：benchmark protocol and provenance。
- Table 4：hardware/deployment boundary table。

### E. 文献缺口

- `zotero_literature_review_cards.md` 已经提供 38 篇文献卡片，但 note 中参考文献仍需投稿化。
- 所有 `preprint, 2025/2026` 条目需要核对 arXiv/DOI/版本。
- 真实硬件 decoder 文献必须记录 latency/resource/closed-loop metrics，不能只列标题。
- GKP analog soft information 文献要防止“本项目首次使用 analog syndrome”的误写。
- adaptive/calibrated decoder 文献要用于说明本项目的真实差异：physical-layer affine fast-path calibration。
- 参考文献应从 `thebibliography` 迁到 BibTeX 或至少补全 key、作者、venue、year、DOI/arXiv。

### F. 可复现与材料缺口

- 每个结果行需要 machine-readable provenance：task id、git commit、config、seed、artifact hash、env、runner version。
- 需要一个 audit helper 检查 paper table 与 result artifact 一致。
- 需要保存图表生成脚本、figure data、manifest。
- 需要给出 data/code availability statement 草案。
- 需要说明哪些 artifacts 是 canonical，哪些是 historical，哪些是 excluded。

## 9. 建议的重构路线

### 最小可行投稿稿路线

1. 保留当前中心命题：dual-loop affine calibration for drift-adaptive GKP decoding。
2. 主文只放 `T24` frozen benchmark、FR6/FR7 descriptive support 和 reader-facing 方法解释。
3. 把 `statcalib`、training/material、isolated `.tflite`、real-board gate 移到 appendix/supplement/boundary table。
4. 去掉 Results 中 Wave A/B 大表，改为 Discussion/Future Work 的短段。
5. 做 citation cleanup 和 figure-first rewrite。
6. 明确写成 software-HIL/system-method paper，不写成 hardware/deployment paper。

### 更稳健论文路线

1. 先完成 `T90` repeated-run consistency pack。
2. 做 physics/noise/logical-error audit。
3. 做 runner/provenance/fallback audit。
4. 写 expanded benchmark protocol。
5. 运行一个 clean-provenance expanded benchmark，至少加入 oracle/nearest-lattice/wrapped-Gaussian 小规模 baseline、unseen drift holdout 和 CI。
6. 再重写摘要、引言、结果和讨论。

### 高质量投稿路线

1. 除 expanded benchmark 外，补 bit-accurate fixed-point simulation 或真实 hardware timing/resource。
2. 对 real-time FPGA/QEC decoder 文献做 metric-level comparison。
3. 对 GKP analog/soft-information baselines 做更接近任务定义的对照。
4. 提供 artifact reproducibility pack 和 audit helper。
5. 将论文从“项目材料说明”压缩成“一个可检验科学命题 + 一组强实验 + 清晰边界”。

## 10. 建议下一批任务

优先级从高到低：

1. **citation sync gate**：把 Zotero 文献卡片与 note 引用逐条对齐，补 DOI/arXiv/版本和文献分组表。
2. **physics/noise/logical-error audit**：审计 syndrome wrapping、lattice constant、noise stream、logical error criterion。
3. **theory-to-config bridge audit**：把 note 的 \(\theta_t^{noise}\) 与 configs/scenarios/dataset builder 逐项映射。
4. **runtime/fallback semantic audit**：确认 software-HIL、mock backend、`.tflite`、stub、artifact_npz、inproc path 没有混写或静默 fallback。
5. **runner/provenance audit**：检查 result table、summary、config、commit、artifact hash、env 的一致性。
6. **claim-to-evidence/literature crosswalk**：每个摘要/引言/结论 claim 必须回指 experiment evidence 和 literature evidence。
7. **paper-grade benchmark protocol refresh**：先锁协议，再决定是否 rerun。
8. **figure pack task**：生成 reader-facing architecture/noise/main-result/adaptation trace/ablation/runtime boundary 图。
9. **reader-facing rewrite task**：在证据不升级的前提下，把内部治理语言转为论文语言。

## 11. Risk / unsupported claims

以下 claim 目前不能在论文中作为完成事实出现：

- real-board HIL validated。
- hardware-ready or deployment-ready system。
- default-environment `.tflite` portability closed。
- full training reproducibility or cross-host portability。
- mature promoted `statcalib` comparator。
- unique clean `statcalib` threshold。
- broad superiority over GKP decoders。
- causal mechanism closure。
- paper-grade expanded benchmark completed。
- Nature-level far-reaching implication established。

## 12. 复核补充：整改优先级与材料清单

本节把上面的审稿意见压成可执行的修改顺序。它不是新的任务包，也不授权修改主线 note；如果后续由 Captain 拆包，应继续按 `Allowed files / Forbidden scope / Verification / Docs to update` 写清边界。

### P0：不补实验也必须先改的成稿问题

| 问题 | 审稿风险 | 建议处理 |
| --- | --- | --- |
| 内部治理语言过多 | 外部审稿人会把稿件读成项目 closeout 报告 | 把 `T24`、`FR8`、`NO_GO`、`extension lane` 等在正文中翻译成 protocol、supplementary analysis、hardware-readiness gate 等论文语言 |
| 主张句不够单一 | 摘要、引言、贡献段容易各讲一个中心 | 固定一句中心命题：dual-loop affine calibration 在预声明 software-HIL 协议下改善 drift-adaptive GKP correction，同时保留 deterministic fast-path boundary |
| 证据层级混排 | 主结果、supporting boundary、future lane 被误读成同级结果 | 主文只放核心 software-HIL 结果和必要 ablation；`statcalib`、`.tflite`、real-board gate、training reproducibility 放 appendix/supplement/boundary table |
| 贡献点过多 | 六个贡献像内部材料清单，不像论文贡献 | 压成 3 个：affine runtime contract、teacher/residual calibration evidence、bounded deployment/readiness boundary |
| 参考文献未投稿化 | credibility 下降，Related Work 难以支撑 novelty | 从 Zotero cards 建 BibTeX，逐条核对 DOI/arXiv/venue/version，删除或降级无法核验条目 |

### P1：决定论文能否成立的证据材料

| 材料 | 为什么必要 | 当前状态 | 最小补法 |
| --- | --- | --- | --- |
| theory-to-config bridge | 审稿人需要知道数学 drift/noise 符号如何落到四个 scenario | 当前散落在 note、protocol、config 与治理文档中 | 做一张表：symbol / code field / config / scenario / physical interpretation / non-claim |
| physics/noise/logical-error audit | 防止 `final_ler` 被误解为完整 finite-energy logical-channel fidelity | 当前已有代码和边界说明，但不是投稿式审计 | 审计 lattice constant、wrapping、noise stream、logical failure criterion、proxy boundary |
| runner/fallback audit | 防止结果被质疑为 stub、fallback 或 artifact path 混用 | 当前治理上有边界，投稿稿仍需机器化证明 | 检查 software-HIL、artifact_npz、inproc、`.tflite`、stub fallback 的 source 字段和 manifest |
| statistical treatment | `repeats=2` 均值不足以支撑强显著性结论 | 当前只能 descriptive | 至少给 per-scenario paired deltas、SD、bootstrap envelope；若要强 claim，需要新增 protocol rerun |
| source-data manifest | 结果表必须可追溯 | 当前材料已有多份 ledger，但投稿式 source data 仍需集中 | 每个 figure/table 绑定 CSV/JSON、hash、config、run root、review、boundary |

### P2：让论文从“雏形”变成“较稳投稿稿”的实验包

| 实验/分析 | 审稿人会问什么 | 需要补到什么程度 |
| --- | --- | --- |
| strong baseline ladder | affine fast path 是否只是在弱 baseline 上赢 | 至少加入 fixed affine、known-noise/oracle affine、wrapped-Gaussian/nearest-lattice sanity、EKF/UKF/RLS、CNN-only/teacher-only 的可比口径 |
| unseen drift holdout | drift-adaptive 是否泛化到未见轨迹 | random-walk、burst/reset、faster-than-window 或 sinusoidal drift 的 predeclared holdout；如果只做 diagnostic，必须降级表述 |
| ablation matrix | 到底是 histogram、teacher、CNN residual 还是 update cadence 起作用 | feature removal、teacher removal、residual-b/residual-K、context length、commit cadence、clip/EMA 的分层 ablation |
| mechanism trace | 为什么 winner 会赢 | 展示 drift trajectory、estimated parameter、committed \(K,b\)、residual、failure events 的同图 trace；不能把 mixed intervention 写成因果证明 |
| fixed-point/runtime proxy | FPGA-facing 语言是否有任何工程支撑 | bit-accurate fixed-point software emulation、operation-count、saturation/overflow、latency/resource proxy；没有真板前不得写 measured board result |
| reproducibility pack | 结果是否可重复 | T90 same-host repeated-run consistency 是近端补强；cross-host/GPU/Linux 是后续更强层级，不可提前写成完成 |

### P3：高影响或硬件 venue 需要的额外材料

- 真实硬件或等价硬件路径：bitstream/RTL/DMA/AXI/register contract、source-vs-board vector、commit latency、throughput、resource、power。
- 更广泛的 benchmark：更多 drift families、predeclared stopping、更多 seeds、CI 或严格 descriptive uncertainty。
- 外部文献 metric 对齐：把 GKP analog information、adaptive decoder、real-time FPGA decoder 文献按 metric 而不是按标题比较。
- Data/code availability：说明代码、configs、source data、artifact hashes、excluded surfaces 和 historical artifacts 的释放边界。
- 投稿格式材料：目标期刊模板、figure source data、supplementary information、methods detail、bibliography style、author contribution / competing interests / data availability。

### 当前可投稿性分级

| 目标 | 当前判断 | 必须满足 |
| --- | --- | --- |
| 内部冻结主线 note | 基本已达成 | 继续保持 frozen handoff 和 post-freeze change-control |
| simulation/material-first 系统方法稿 | 接近，但仍需 reader-facing rewrite 和 source-data/provenance pack | P0 + P1 |
| 稳健量子工程/系统论文 | 还差一轮 evidence-hardening | P0 + P1 + 至少部分 P2 |
| FPGA/hardware venue | 当前不够 | P3 的真实 timing/resource/board 或等价硬件证据 |
| Nature-style high-impact Article | 当前远未建立 | outstanding broad significance、强 baseline、强统计、硬件或理论突破至少满足一条 |

## 13. 最终评语

这份 note 最有价值的地方是：它已经从容易 overclaim 的 “CNN + FPGA GKP decoder” 收敛成了更可信的 “deployment-bounded affine calibration framework”。这是正确方向。

但从审稿人视角看，它还需要从“内部可信材料包”转成“外部可审论文”。关键不是继续润色句子，而是补三类东西：第一，证明核心命题的更强实验；第二，证明结果没有 fallback/provenance 问题的审计材料；第三，让外部读者不需要理解仓库治理史也能读懂的图表和叙事。

在当前证据下，最诚实的判断是：可以形成一篇 simulation/material-first 的系统方法论文雏形；尚不能形成真实硬件、完整 deployment、广义 SOTA 或 high-impact broad-claim 论文。

## 14. 2026-07-03 复核：投稿稿补强后的差距更新

本节基于后续形成的 `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex`、source-data audit、symbol-boundary audit、literature metric crosswalk、source-data manifest、row-level provenance manifest、benchmark-expansion protocol、runner smoke pair、commit-lag sweep 和 source-data coverage matrix 进行复核。它不替代上文的原始审稿意见，而是说明哪些问题已经被压低，哪些仍然会被审稿人追问。

### 已明显改善的部分

1. **论文语体已经基本脱离内部治理语言。** 投稿稿主文已经把 `T24`、`FR8`、`NO_GO`、`extension lane`、`evidence boundary` 等内部项目词汇压缩或转写为 `predeclared software-HIL protocol`、`supplementary calibration analysis`、`planned hardware-validation measurements`、`source-data manifest` 等外部读者可理解的论文语言。最新禁词扫描未发现主要内部任务编号或项目状态词残留在投稿稿主文中。

2. **主张-证据层级比原 note 清楚。** 投稿稿新增了 metric-readiness、claim-scope/source-data、validation-scope、source-data coverage matrix 和 statistical-treatment 表格，把主结果、诊断分析、实现可行性、文献定位和计划性边界分开。审稿人仍可质疑证据是否足够强，但不太容易再误读为作者把计划项写成已完成结果。

3. **理论桥接比原 note 更完整。** 投稿稿已经补入 approximate-GKP syndrome model、local Gaussian/LMMSE affine fast-path derivation、residual-boundary channel surrogate、GKP boundary sensitivity 和 physics-metric boundary 表述。当前仍不是 finite-energy logical-channel paper，但至少给出了从 half-lattice crossing proxy 到 channel-language surrogate 的明确降级路径。

4. **非硬件实现可行性证据增加。** 投稿稿现在包含 fast-path analytical operation-count、Q4.20 fixed-point software parity、software runtime-discipline counters、validation-contract figure/source data 和 commit-lag sweep。它们不能替代 FPGA synthesis 或 board latency，但足以支持“affine fast path 是一个合理的 hardware-facing target”这一较弱主张。

5. **外部文献比较变得可审计。** literature metric crosswalk 已把 GKP analog information、logical-channel/fidelity metrics、learned QEC modules 和 real-time FPGA decoders 的指标按 citation key、reported metric、manuscript use、non-claim boundary、anchor strength 和 final pinning follow-up 记录。当前稿件不再只是堆引用，而是能解释哪些外部指标只是定位参照，哪些需要最终 page/table/figure pinning。

6. **source-data 机械一致性显著增强。** 投稿稿已有 source-data audit、symbol-boundary audit、source-data manifest 和 row-level provenance manifest。最新 source-data audit 对主结果、paired deltas、paired uncertainty、GKP boundary sensitivity、oracle-affine sanity check、sequence baselines、holdout stress、commit-lag sweep、fast-path cost、fixed-point parity、runtime discipline、literature crosswalk、coverage matrix、benchmark-expansion protocol、runner smoke pair 和 row-level provenance 进行机械检查，并返回 `PASS_WITH_LIMITATIONS`。

### 仍然会构成审稿硬问题的部分

1. **主 benchmark 仍然是小样本 descriptive evidence。** 目前主结果仍来自四个预声明场景、五个模式、paired seeds、每场景每模式两次 repeat。paired-bootstrap envelope 只能作为 source-data transparency，不能写成 confidence interval、standard error、p-value、significance test 或 robustness proof。若目标是高质量期刊，至少需要 repeat-expanded paired analysis；若要更强，还需要 holdout drift families 和 missing-run accounting。

2. **holdout drift 仍是 controlled diagnostic，不是 trained-branch generalization proof。** random-walk、burst/reset、faster-than-window 和 commit-lag sweep 已经降低了“只在四个固定场景上调参”的风险，但它们当前是非硬件、known-state 或 controlled diagnostic。审稿人仍会要求正式 predeclared holdout benchmark、trained-branch holdout validation 和 repeat-level uncertainty。

3. **logical-channel fidelity 仍未建立。** 当前 residual-boundary surrogate 对读者有帮助，但它不是 finite-energy GKP logical-channel simulation、process fidelity、logical-channel tomography 或 outer-code LER。若正文继续使用 fidelity/infidelity 语言，必须保持 surrogate 标签，并把真正的 logical-channel simulation 放入 future validation。

4. **文献数字仍需最终原文 pinning。** crosswalk 已经记录 anchor policy，但若投稿稿在正文中使用外部硬件 latency、resource、memory、closed-loop timing 或 logical-channel probability 的具体数字，作者仍需逐项回到 PDF/HTML 的 page、figure、table 或 caption 做最终核对。card-level anchor 不足以支撑强 per-value comparison。

5. **hardware claim 仍然完全未闭合。** 投稿稿现在把硬件写成 planned measurement surface，这是正确的。但如果目标 venue 或摘要继续突出 FPGA，审稿人会要求 board logs、bitstream/RTL hash、DMA/MMIO evidence、source-vs-board vectors、latency distribution、resource/power 和 timing closure。没有这些材料时，只能写 hardware-facing design target。

6. **图件和排版仍需最终视觉 QA。** 当前有 Python draft figures、source data 和 manifest，但投稿级 figure package 仍需要最终 caption polish、PDF visual QA、source-data freeze、字体/版式检查和目标期刊模板适配。长表较多，正式投稿前可能需要把 coverage/provenance/boundary tables 下沉到 Supplement。

7. **复现性仍不是 full reproducibility。** source-data manifest 是 file-level manuscript-facing hash manifest；row-level provenance 覆盖当前软件 HIL 主结果行；这不等于 historical `runs/` 递归 hash closure、cross-host reproducibility、training repeated-run closure、container lock 或 hardware provenance。

### 当前稿件可信度的新判断

当前投稿稿已经比原 note 更接近一篇正式论文。若目标是 bounded systems-method 或 quantum engineering manuscript，它已经具备一个相当完整的 manuscript skeleton、source-data guardrail 和 honest limitation stack。若目标是 high-quality/high-impact journal，非硬件部分仍至少缺一轮强证据：repeat-expanded anchor benchmark、formal holdout drift validation、final literature value pinning、figure-source freeze 和更完整 reproducibility pack。

因此，最新判断应从“还只是内部主线 note”更新为：

> 当前稿件可以作为一篇诚实的 bounded software-HIL systems-method manuscript draft 继续人工终修；它尚不能被宣布为 high-impact complete paper，因为关键统计、holdout、logical-channel fidelity、final literature pinning 和硬件实测仍未闭合。

### 下一步最有价值的非硬件补强

1. **Phase A repeat-expanded UKF-vs-Hybrid anchor comparison。** 按 `submission_draft_benchmark_expansion_protocol.csv/json` 执行至少 12、目标 16 paired repeats per scenario，并预声明 interval/reporting rule。不要在执行前把当前 n=2 envelope 写成 CI。

2. **Phase B formal holdout drift benchmark。** 将当前 controlled holdout diagnostics 转化为正式 predeclared holdout family run，至少覆盖 random-walk、burst/reset 和 faster-than-window families，并把 trained branch、fixed affine、UKF、oracle affine 和 wrapped-Gaussian references 区分清楚。

3. **final literature pinning pass。** 对所有正文强数字，特别是 FPGA latency/resource/power/memory/closed-loop timing 和 finite-energy logical-channel probability，补 page/table/figure/caption anchors。若不能 pin，就把数字降级为 categorical positioning。

4. **source-data freeze and visual QA。** 冻结 manuscript-facing CSV/JSON、figure source data、figure PDFs、source manifest 和 audit reports；对 PDF 页面做最终视觉检查，尤其是长表、caption、figure placement 和 supplement split。

5. **finite-energy logical-channel follow-up。** 若要保留 fidelity/infidelity 作为重要卖点，必须新增真正的 finite-energy GKP channel simulation 或等价 logical-channel estimate；否则继续把当前项限定为 residual-boundary surrogate。

## 15. 2026-07-06 复核：基于当前 `CNN_FPGA_GKP_theory_note_draft.tex` 的源文件审稿标注

本节重新回到 `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex` 本身，而不是只看后续投稿化稿件。结论是：当前 note 已经具备完整论文骨架和相当成熟的边界意识，但它在源文件层面仍保留了明显的内部治理痕迹、过多层级化自证材料，以及尚未由正式实验闭合的硬主张入口。若要把这份 note 直接整理为外部论文，应先把“内部冻结说明”转译为“读者可审的 scientific manuscript”。

### 15.1 Nature-reader 源文结构图

| 源文件区域 | 当前功能 | 审稿阅读判断 |
| --- | --- | --- |
| line 22-57 Abstract | 概括 approximate GKP、dual-loop affine fast path、teacher residual、software-HIL ranking、statcalib no-promotion | 信息完整，但太多边界和内部证据层同时进入摘要；正式投稿应压缩为 context/gap/approach/key result/boundary 六句以内 |
| line 63-139 Introduction | 建立 GKP analog syndrome、drift、learned/hardware QEC 背景和证据层级 | 逻辑方向正确；line 132-139 的 “current note / evidence hierarchy” 更像内部说明，投稿时应改成普通论文的 validation-scope paragraph |
| line 144-207 Summary of Contributions | 六个贡献点，覆盖 formulation、teacher residual、evidence layer、mechanism、statcalib、runtime boundary | 贡献太多且层级不齐；建议合并为三条核心贡献，supporting/runtime/boundary 放 Methods、Results 或 Supplement |
| line 209-297 Metric-level advantages | 按 LER、latency、FPGA resource、implementation、drift robustness、modularity、outer-code compatibility 分层 | 审稿人会喜欢分层诚实性，但主文表太重；建议转为 Supplement，并在主文只保留 2-3 个最硬优势 |
| line 300-399 Brief Review of the GKP Code | 给出 approximate GKP、modular syndrome、local affine LMMSE、logical failure criterion | 是当前 note 的理论亮点；仍需把 finite-energy surrogate 与 true logical-channel/fidelity 边界写得更严格 |
| line 401-451 Noise and Drift Model | 定义 \(\theta_t^{noise}\) 与四个 drift scenario | 需要补 theory-to-config bridge：每个符号如何对应 config 字段、runner、dataset builder 和物理解释 |
| line 454-606 Model Architecture | 描述 fast loop、ParamMapper、teacher、CNN residual、statcalib、stage-and-commit | 方法结构完整；但 statcalib 的强结果会持续削弱 “CNN residual” headline，应把中心彻底转到 affine calibration contract |
| line 610-713 Relationship to Existing Work | 对 analog GKP、adaptive priors、learned QEC、FPGA decoder 做定位 | 已有正确分层；还需要一张更投稿化的 related-work comparison table，减少未来年份或未核验条目的强依赖 |
| line 717-758 Experimental Setup | 说明 mock-backed software-HIL、四场景、五模式、`final_ler_mean` | 是主结果可信度的入口；需要补 repeat/statistical protocol，否则只能 descriptive |
| line 766-837 Main results and ablations | 给出四场景主表与 feature/teacher ablation | 主结果可作为 bounded anchor；ablation 中 `hybrid_no_teacher_params` 最优说明不能宣称 teacher-parameter necessity |
| line 843-937 Statcalib extension | 展示 statcalib 很强但 no-promotion | 边界写得诚实；正式论文中应下沉 supplement，避免读者认为主线 CNN 被非神经 baseline 推翻 |
| line 939-1004 Mechanism/generalization/runtime/hardware gaps | 明确机制非因果、unseen drift 未完成、runtime/hardware 未闭合 | 这些是非常重要的限制；投稿时应改成 validation ladder，而不是让 Results 像风险台账 |
| line 1008-1090 Follow-up routes | Wave A/B sidecar 候选 | 不应留在主文 Results；可转为 Outlook 或 Supplement，且只保留与论文中心命题直接相关的少数路线 |
| line 1100-1224 Discussion and Conclusion | 重新定位为 low-dimensional affine calibration contract，并强调 frozen route/non-closure | 方向正确；但 “frozen mainline note / manual-finish / handoff” 等内部语义不适合正式论文，应全部转译或删除 |
| line 1226-1381 Bibliography | 手写 `thebibliography` | 需要改成 BibTeX 或至少逐条补全 DOI/arXiv/version/page anchors；未来年份/预印本条目必须最终核对 |

### 15.2 术语账本与转译建议

| 当前术语 | 投稿稿建议术语 | 原因 |
| --- | --- | --- |
| `T24` frozen benchmark | predeclared four-scenario software-HIL benchmark | 外部读者不应理解任务编号 |
| mock-backed software-HIL | software-in-the-loop simulation with mocked board backend | 保留证据等级，但减少内部口径 |
| FR8 / statcalib extension lane | supplementary statistical-calibration analysis | 避免把治理 lane 写进主文 |
| no-promotion gate | not promoted to the main comparator set | 转为普通论文判断 |
| real-board `NO_GO` | hardware validation not executed because device/host prerequisites are absent | 避免像内部 gate log |
| isolated true `.tflite` runtime | selected artifact runtime check in an isolated host environment | 防止误读为 deployment |
| frozen mainline handoff | current manuscript scope | 删除或转为 scope statement |
| blocked surface | validation not yet performed | 删除内部治理感 |
| `final_ler_mean` | protocol-defined logical-error proxy | 必须注明不是 finite-energy logical-channel fidelity |
| sidecar Wave A/B | future validation directions | 只保留最相关内容 |

### 15.3 当前 note 的成稿距离判断

1. **作为内部研究 note：已基本完整。** 它清楚记录了架构、主结果、支持材料、边界和后续路线，且没有把硬件、`.tflite` 或 statcalib 过度升级。
2. **作为 simulation/material-first 论文初稿：接近，但仍需重构。** 需要删去内部任务语言，压缩贡献和结果层级，把 audit/gate/sidecar 表格下沉 Supplement。
3. **作为稳健量子工程论文：还缺一轮证据硬化。** 最关键是全场景 repeat-expanded interval、formal holdout drift、strong comparator ladder、source-data freeze 和 reproducibility pack。
4. **作为 FPGA/hardware 论文：当前证据不足。** 没有 board timing/resource/source-vs-board/bitstream/DMA/MMIO 证据。
5. **作为 Nature-style high-impact Article：当前远未建立。** 目前更像一个诚实的 systems-method manuscript，而不是 broad-interest field-changing result。

### 15.4 实验可信度的最终审稿判断

| 实验层 | 可信度 | 可写边界 | 不可写边界 |
| --- | --- | --- | --- |
| 四场景 software-HIL 主表 | 中等可信 | bounded descriptive ranking under a predeclared software-HIL protocol | paper-grade expanded benchmark, broad robustness, hardware result |
| feature/teacher ablation | 有用但有限 | descriptive support for feature relevance and non-necessity of teacher params | causal mechanism closure |
| statcalib extension | 数值强但只能 supplement | same affine contract can host strong non-neural calibration | promoted main comparator, unique threshold, mature statcalib validation |
| unseen drift paragraph | 只是缺口说明 | future benchmark need | completed generalization proof |
| oracle/wrapped-Gaussian paragraph | 只是缺口说明 | needed comparator ladder | already compared against theoretical optimum |
| `.tflite` runtime | 边界事实可信 | selected artifacts run in isolated host environment | default-env portability, HIL closure, deployment closure |
| real-board gate | 只可信为未执行边界 | hardware validation is blocked by missing host/device prerequisites | real-board HIL success |
| fast-path latency/resource | 架构推断可信 | low arithmetic footprint by design | measured FPGA latency/resource/power |

### 15.5 优先修改清单

**第一优先级：不新增实验也必须整理**

1. 删除或转译所有 `% Txx-*` 任务注释、正文中的 `T24`、`FR8`、`NO_GO`、`frozen mainline`、`handoff`、`extension lane` 等内部词。
2. 把六个贡献压成三条：affine runtime contract；teacher/residual or calibration evidence；validation ladder and hardware-facing boundary。
3. 把 `Metric-level advantages` 和大部分 `statcalib`/sidecar/coverage 类表格移到 Supplement。
4. 把 Results 改成“主结果 -> uncertainty -> ablation -> bounded diagnostics”，不要把 future lane 混在 Results 主层。
5. 把结论中 “frozen route / manual finish / blocked surface disclaimer” 改成普通论文的 limitations and validation outlook。

**第二优先级：决定能否成为稳健论文的材料**

1. 完成全四场景 repeat-expanded Phase A，至少 scenario-wise paired intervals。
2. 增加 formal holdout drift benchmark，不能只保留 future paragraph。
3. 增加 formal comparator ladder：nearest-lattice / wrapped-Gaussian / oracle or known-noise affine / fixed affine / UKF/EKF/RLS / CNN-only or teacher-only。
4. 增加 finite-energy logical-channel 或明确删除 fidelity-level 语言。
5. 冻结 source-data bundle：每个 figure/table 的 CSV/JSON/hash/config/run root/script/review boundary。

**第三优先级：若目标包含 FPGA 或硬件 venue**

1. 提供 bit-accurate fixed-point simulation 与 source-vector parity。
2. 提供 synthesis/timing/resource/power 或真实 board logs。
3. 提供 DMA/MMIO/AXI/register contract、bitstream/RTL hash、source-vs-board vectors。
4. 给出 latency p50/p95/p99/worst-case，而不是只写 low-latency by design。

### 15.6 最终审稿式结论

当前 note 的真正价值是把早期容易过度宣传的 “CNN + FPGA GKP decoder” 收缩成了更可信的 “drift-adaptive affine calibration contract”。这个方向是对的，也有论文潜力。但审稿人会要求作者证明三件事：第一，这个 contract 在更强、更宽的 drift/comparator/statistical protocol 下仍有优势；第二，所有结果确实来自可审计 source data 而不是 fallback 或内部材料混写；第三，硬件和 `.tflite` 只是 future validation surface，而不是当前完成事实。

因此，当前 note 距离完整论文的主要差距不是“还缺一轮润色”，而是“还缺一轮外部审稿可接受的证据硬化和主文压缩”。在不补硬件实验的前提下，最现实目标是一篇 bounded software-HIL / systems-method paper；若要冲击更高影响或硬件方向 venue，必须补 repeat-expanded benchmark、formal holdout、formal comparator ladder、logical-channel/fidelity evidence 或真实硬件测量中的至少一组强证据。
