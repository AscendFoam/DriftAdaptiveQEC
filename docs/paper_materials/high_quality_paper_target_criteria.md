# 高质量投稿论文目标清单

**生成日期**：2026-07-02  
**适用对象**：`CNN_FPGA_GKP_theory_note_draft.tex` 及后续 paper-grade 实验、图表、审计任务。  
**核心判断**：当前 note 已有论文骨架，但距离高质量投稿的主要差距不在 prose，而在理论严谨度、代码证据可信度、benchmark 公平性、硬件边界和文献对齐。

## 1. 目标论文应讲清的核心命题

更稳妥的中心命题是：

> Drift-adaptive GKP correction can be organized as a dual-loop affine calibration problem: a deterministic low-latency fast path executes \(\Delta=Ks+b\), while a slower teacher/statistical/learned calibration loop updates the low-dimensional runtime surface from syndrome statistics.

这句话比“CNN 解码器优于已有方法”更稳，因为：

- GKP analog soft-information decoding 已有强文献基础；
- ML / calibration-conditioned decoder 已有邻近工作；
- FPGA real-time decoder 文献对硬件证据要求很高；
- 本项目当前真实强项是 runtime contract 和 evidence-bounded drift adaptation，而不是完整真实 FPGA 系统。

## 2. 理论目标

### 必须达到

1. **近似 GKP 背景完整**  
   明确 ideal comb state、finite-energy envelope、finite squeezing、auxiliary-state / measurement noise，以及这些因素如何进入 effective displacement noise。

2. **syndrome 与 logical error 定义严谨**  
   说明 syndrome 是 modulo-lattice observation，不是普通线性观测；逻辑错误来自 residual displacement 跨越 decision boundary。

3. **affine fast path 有推导而非拍脑袋**  
   用 local Gaussian / LMMSE 解释 \(K=\Sigma_{es}\Sigma_{ss}^{-1}\)、\(b=\mu_e-K\mu_s\) 的来源，同时明确 wrapped posterior / multi-branch 情况下 affine 只是局部近似。

4. **noise/drift model 与代码配置一致**  
   理论中列出的 static bias、linear ramp、step variance、periodic drift、covariance / anisotropy 等，必须能映射到具体 config、dataset builder 和 benchmark scenario。

5. **teacher/statcalib/CNN 的角色分离**  
   teacher 是 slow-loop anchor，CNN 是 residual calibrator，statcalib 是 separate extension/comparator lane；三者都不应被写成 per-shot neural decoder。

### 加分目标

1. 给出 oracle affine / wrapped-Gaussian / nearest-lattice 小规模理论 baseline，用来分解“affine 限制损失”和“slow-loop 估计误差”。
2. 给出 adaptation lag、commit cadence、stale parameter penalty 的数学定义。
3. 把 GKP physical-layer calibration 与 surface-GKP / QLDPC-GKP outer decoder 的接口写成 future-compatible formulation。

## 3. 代码与物理实现审计目标

高质量论文前，必须先做 evidence audit，尤其是用户担心的 demo/fallback 问题。

### `physics/` 审计目标

- syndrome wrapping 区间、符号约定、\(\sqrt{2\pi}\) 或相关 lattice constant 一致；
- finite-energy / squeezing / measurement noise 的采样与文献模型不冲突；
- random seed、noise stream、measurement noise stream 可追踪；
- logical error 判定与 GKP decision boundary 一致；
- loss channel、Gaussian displacement channel、effective drift scenario 不混写。

### FPGA / runtime / HIL 审计目标

- 明确 fast-loop 是否只是 software emulator、mock-HIL、`.tflite` runtime 或 real-board path；
- fixed-point Q format、saturation、overflow、clip、rounding 行为可复查；
- parameter-bank stage / commit / rollback / stale parameter 语义有测试；
- runner 不会静默 fallback 到 stub / placeholder / artifact-only path；
- 所有 `.tflite` 结果区分 true runtime 与 `.json` stub；
- real-board 相关只写 gate/provenance，除非未来真的有 device path、bitstream、DMA/AXI contract 和 measured timing。

### 产物目标

每个审计任务应输出一张 ledger：

| 类型 | 说明 |
| --- | --- |
| bug | 代码或数学实现明显错误 |
| demo/fallback risk | 可能把 demo/stub/path fallback 当成正式证据 |
| unsupported assumption | 论文 claim 强于代码证据 |
| acceptable simplification | 可接受，但需要在论文中写清边界 |
| docs wording issue | 主要是措辞或引用边界问题 |
| needs rerun | 修正后必须重跑才能恢复 claim |

## 4. Benchmark 与实验目标

### 最低可投稿目标

1. 保留 `T24` frozen-set ranking 作为历史 anchor。
2. 对当前四个 drift scenarios 报告 LER / logical-error proxy、mean/std 或 confidence interval。
3. 至少包含 fixed affine、EKF、UKF、window variance、teacher residual、CNN residual、statcalib extension lane。
4. 报告 feature/teacher ablation，且不能把 teacher params 写成必要 positive contributor。
5. 报告 seed split、train/eval split、config hash、commit hash、artifact hash。
6. 把 drift timescale 与 update cadence 写清楚，避免只有“adaptive”字样而没有可跟踪条件。

### 更强目标

1. 增加 oracle affine、wrapped-Gaussian / nearest-lattice 小规模 baseline。
2. 增加 unseen drift holdout：random walk、burst/reset、sinusoidal、mixed covariance。
3. 增加 adaptation metrics：update lag、commit failure、rollback/freeze frequency、stale penalty。
4. 增加 runtime metrics：p50/p95/p99 latency proxy、fast-loop op count、slow-loop update frequency、saturation/overflow。
5. 增加 statistical protocol：paired seeds、bootstrap CI、effect size、predeclared stopping。
6. 增加 soft-information / analog-information 对照：hard-binning、standard GKP binning、history-aware Bayesian / memory-assisted baseline、runtime soft-info baseline。

### 最强目标

1. 重跑一个 clean-provenance expanded benchmark，而不是只复用历史 runs。
2. 对每个 result row 生成 machine-readable provenance：task id、git commit、config、seed、artifact、env、runner version。
3. 用一个 audit helper 自动检查 result table 与 artifact 一致。
4. 如果继续讲 FPGA，至少给出 synthesizable fast-path 或 bit-accurate fixed-point simulation；如果没有，就明确为 software-HIL / deployment-bounded。
5. 如果继续讲 neural+FPGA，必须正面对比已有真实闭环文献中 ns/us 量级 latency、资源、throughput 和 closed-loop 证据；不能只用“可部署”措辞替代测量。

## 5. 图表目标

正式论文至少应有以下图表：

| 图/表 | 内容 | 证据要求 |
| --- | --- | --- |
| Fig. 1 架构图 | GKP syndrome -> fast affine path -> slow calibration loop -> parameter bank | 只画真实存在的模块；blocked hardware 用虚线 |
| Fig. 2 GKP/noise model | approximate GKP、modular syndrome、drifted effective noise | 与 `physics/` 审计一致 |
| Fig. 3 主结果 | 四场景 LER / logical-error proxy 比较 | 回链 run root 和 result ledger |
| Fig. 4 drift/adaptation trace | drift trajectory、teacher/statcalib/CNN 更新、commit 参数 | 需要 trace-level provenance |
| Fig. 5 ablation | feature、teacher、residual branch、statcalib lane | 不升级机制因果 |
| Fig. 6 runtime boundary | fast-loop op count、latency proxy、`.tflite` current-host table、real-board NO_GO | 明确 evidence layer |
| Table 1 related work | GKP analog / adaptive calibration / ML decoder / FPGA decoder 对比 | 文献卡片回链 |
| Table 2 claim-evidence map | 每个 claim 对应 task/run/artifact/literature | 论文投稿前必须更新 |
| Table 3 hardware boundary | LILLIPUT / Helios / Collision / Yang 等真实硬件指标 vs 本项目当前证据 | 明确本项目没有 real-board execution success |

## 6. 文献与 Related Work 目标

Related Work 应按“问题维度”组织，而不是堆作者：

1. **GKP and approximate GKP QEC**：GKP 2001、Glancy 2006、Grimsmo/Puri 2021、finite-energy/logical channel/loss 文献。
2. **Analog and soft-information decoding**：Fukui、Noh、Berent、Borah、Hanisch 等。
3. **Adaptive calibration and noise estimation**：Spitz、Wagner、Chen、DGR、Sivak prior optimization。
4. **Learned low-latency decoders**：Google/Bausch、Chamberland AI pre-decoder、Stein FiLM、neural MWPM。
5. **FPGA and real-time QEC decoders**：LILLIPUT、Helios、Collision、Local Clustering、qLDPC FPGA、Yang、Caune、Sivak break-even。

每一类都要写清：

- 已有工作解决了什么；
- 本项目不 claim 的部分；
- 本项目真正补的缺口：GKP physical-layer drift-adaptive affine runtime calibration under deployment-bounded constraints。

注意两类强邻近工作：

- FiLM / AI pre-decoder 类工作已经明确使用 calibration-conditioned 或 low-latency learned module；本项目必须强调 slow-loop calibration surface 和 deterministic affine fast path。
- FPGA neural decoder / real-time QEC 已有真实闭环 ns/us 指标；本项目当前只能把 FPGA 写作目标或 deployment boundary，不能写作已实现硬件结果。

## 7. 写作目标

### 摘要

摘要必须同时包含：

- 问题：approximate GKP under drift；
- 方法：dual-loop affine calibration；
- 结果：四场景 frozen software-HIL 中的 strongest accepted ranking；
- 边界：software-HIL / current-host runtime / real-board gate 分层；
- 结论：不是 CNN closed decoder，而是 calibration contract。

### Introduction

Introduction 应避免一上来讲实现细节。推荐逻辑：

1. GKP analog syndrome 有价值；
2. approximate GKP 和 drift 使 fixed decoder 脆弱；
3. 现有 analog soft-information 多面向外码或 decoder weights；
4. learned decoder 和 FPGA decoder 都要求明确 latency role；
5. 本项目提出 physical-layer affine calibration contract。

### Results

Results 不应只给平均 LER。需要：

- 主结果 ranking；
- scenario-wise robustness；
- ablation；
- mechanism trace；
- runtime/evidence boundary；
- failed/negative result 的 honest interpretation。

### Discussion

Discussion 应明确：

- 为什么 statcalib 强不推翻项目，而是支持 calibration-contract story；
- 为什么当前不是 real-board FPGA 论文；
- 哪些结果可写入 main text，哪些只能 appendix/supplement，哪些必须 blocked。

## 8. 投稿前 gate

建议设置三个等级：

### Gate A: 可形成内部论文草稿

- 文献卡片完整；
- claim-evidence map 完整；
- physics/runtime demo-fallback 审计完成；
- note 中所有引用可追溯到 Zotero/BibTeX。

### Gate B: 可投一般量子工程/交叉期刊

- expanded benchmark 或至少 clean-provenance rerun 完成；
- baseline/ablation/CI 完整；
- `.tflite` 与 real-board 边界完全诚实；
- 图表可直接复用。

### Gate C: 冲更高质量投稿

- 有强 oracle/soft-information/ML/hardware 文献对标；
- 有 unseen drift holdout 和统计置信区间；
- 有 bit-accurate fixed-point 或真实硬件 timing/resource 证据；
- 有 artifact reproducibility pack；
- 论文叙事从“项目说明”压缩成“可检验科学命题 + 系统证据”。
- 每个写入正文的外部文献数值都已回到原文图表、caption 或摘要页核验，并在本地文献卡片中记录来源。

## 9. 推荐下一批任务

1. `T91`：Zotero 文献卡片与 note citation sync gate。
2. `T92`：`physics/` GKP/noise/logical-error 机制审计。
3. `T93`：noise/drift theory-to-config bridge audit。
4. `T94`：FPGA/mock-HIL/runtime fallback semantic audit。
5. `T95`：runner/provenance/fallback audit。
6. `T96`：note claim-to-evidence/literature crosswalk。
7. `T97`：paper-grade benchmark protocol refresh。

这些任务应先审计，再修复，再重跑。不要在未完成审计前继续堆大实验或直接润色成投稿稿。
