# Deep Research Prompt: CNN-FPGA Assisted GKP Error Correction

请作为量子纠错、机器学习解码器和硬件感知系统研究方向的深度研究助手，进行一次全网文献与项目查重调研。重点检索最近半年到一年内的新论文、预印本、开源项目、技术报告和相关基准工作。

## 项目背景

我正在推进一个研究型工程项目：`DriftAdaptiveQEC`。项目目标是面向连续变量 GKP 量子纠错中的漂移噪声，构建一种 CNN + FPGA 快慢回路协同的自适应解码系统。

核心设想：

1. 快回路由 FPGA / FPGA-like runtime 执行低延迟线性解码：
   - 输入：GKP syndrome `(s_q, s_p)`
   - 输出：线性 correction
   - 形式：`Delta = K @ s + b`
   - 目标：低延迟、确定性、固定点友好
2. 慢回路由 CPU / ARM / CNN 侧周期性处理 syndrome histogram：
   - 输入：多窗口 `32 x 32` syndrome histogram
   - 输出：漂移相关的控制更新
   - 目标：估计或修正噪声漂移下的解码参数
3. 当前主线倾向 teacher-guided residual 思路：
   - 不让 CNN 完全替代经典估计器
   - 使用稳定 classical teacher 作为底座
   - CNN 学习 residual / calibration，尤其是 runtime control bias `b` 的修正
4. 项目重视工程约束：
   - latency budget
   - staged parameter bank / atomic commit
   - fixed-point representation
   - HIL / software HIL
   - `.tflite` deployment boundary
   - future real-board FPGA validation

## 当前项目状态

项目已完成第一轮恢复期治理，目前处于 `Phase 2: Controlled Development`。

已确认：

- 代码仓库不是空壳，已有 `physics/`、`cnn_fpga/`、`benchmark/` 等模块。
- 已恢复 P0 / P3 / P4 的最小可复验路径。
- P3 software HIL bounded path 已做到逐字一致复验。
- P4 已有 development bounded run，但还不是正式 formal benchmark。
- 训练链、`.tflite`、real-board HIL 都已有代码或计划边界，但不能夸大为完整恢复。

重要边界：

- 当前 P3 可复验证据是 `mock + artifact_npz + inproc`，不是 real-board。
- `board_backend.py` 仍是 placeholder，不能写成真实板级 HIL 已完成。
- `.tflite` 路径需要区分真实 runtime 和 stub manifest。
- 当前正式论文结论还需要更严格的 formal benchmark 和机制证据。

当前近期规划：

- 先锁定 P4 formal benchmark protocol。
- 再决定是否执行 bounded formal benchmark。
- 后续再推进机制诊断、训练链可复现、真实 `.tflite` runtime、真板 smoke gate。
- 论文发表是最终目标，但不是一步到位直接写论文。

## 希望你重点调研的问题

### 1. 查重与撞车风险

请检索是否已有工作与以下组合高度重合：

- GKP quantum error correction + CNN / neural decoder
- GKP drift-adaptive decoding
- histogram-based syndrome processing for GKP
- teacher-guided / residual learning for quantum error correction decoder
- hardware-aware neural decoder for GKP or continuous-variable QEC
- FPGA / real-time / HIL quantum error correction decoder
- learned calibration or parameter update for GKP decoders
- hybrid classical estimator + neural residual correction

请特别判断：

- 是否已有工作提出几乎相同的“CNN 从 histogram 估计漂移并更新 GKP 解码参数”。
- 是否已有工作提出“teacher + neural residual”用于 GKP 或类似 QEC。
- 是否已有工作在 FPGA / embedded / real-time 约束下实现类似解码器。
- 如果有相似工作，本项目还能差异化在哪里。

### 2. 最近半年到一年的新增工作

请重点搜索最近半年到一年内的：

- arXiv 论文
- Nature / PRX / PRL / PRA / Quantum / QST / npj Quantum Information 等期刊论文
- 会议论文或 workshop
- GitHub / Zenodo / institutional technical reports
- IBM / Google / AWS / Xanadu / PsiQuantum / academic group 的相关 technical blog 或 repo

旧工作也可以提，但请标注时间，优先突出新进展。

### 3. 可借鉴方法

请找出可借鉴到本项目的技术点，例如：

- 更合理的 GKP noise drift model
- 更强 classical baseline
- 更公平的 benchmark protocol
- 更稳健的 learned decoder 训练目标
- teacher / residual / calibration 的替代设计
- uncertainty / confidence / fail-safe mechanism
- latency-aware or hardware-aware neural inference method
- FPGA-friendly decoder architecture
- histogram feature engineering
- deployment validation / HIL protocol

### 4. 对当前技术路线的挑战

请批判性分析：

- 当前 CNN + FPGA 快慢回路设想是否有明显理论或工程弱点。
- teacher-guided residual-b 是否可能只是工程 trick，而非有足够学术贡献。
- 当前 emphasis on histogram windows 是否可能被已有工作覆盖。
- P4 formal benchmark 应该包含哪些 baseline 才有说服力。
- 如果没有 real-board result，这个工作适合投什么类型 venue。
- 如果有 real-board result，投稿定位是否会明显提升。

### 5. 对后续任务规划的建议

请结合调研结果，给出对后续任务的建议：

- T23 P4 formal benchmark protocol 应该重点锁定哪些 baseline / scenarios / seeds / repeats。
- 机制诊断应该优先解释什么现象。
- 是否值得继续做 paper-inspired statcalib branch。
- 是否应该优先恢复 true `.tflite` runtime。
- real-board smoke 对论文价值有多大，是否必须。
- 如果资源有限，最小可投稿证据集是什么。

## 输出格式要求

请输出一份结构化报告，建议包含：

1. Executive summary
2. Search scope and keywords
3. Most similar works table
   - title
   - year / date
   - link
   - method summary
   - overlap with this project
   - collision risk: high / medium / low
   - useful idea to borrow
4. Recent works from last 6-12 months
5. Differentiation analysis
6. Benchmark and baseline recommendations
7. Hardware / deployment / FPGA relevance
8. Recommended next experiments
9. Risks to the current project narrative
10. Suggested revised task roadmap

请尽量提供可点击链接、DOI/arXiv ID、代码仓库链接。不要只给泛泛摘要；要明确哪些工作最像、哪里相同、哪里不同、是否会影响本项目的论文定位。

## 重要约束

- 不要默认本项目已经完成 real-board validation。
- 不要默认本项目已经恢复真实 `.tflite` runtime。
- 不要把 development smoke 当成 formal benchmark。
- 请把“已验证事实”和“未来计划”严格区分。
- 如果发现直接撞车工作，请明确说明，并建议如何调整项目主张。
