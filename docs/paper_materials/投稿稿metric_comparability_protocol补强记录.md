# 投稿稿 metric comparability protocol 补强记录

日期：2026-07-03

本文档记录 `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex` 中外部指标比较段落的一次投稿语体补强。补强目的不是新增实验结果，而是把 LER、logical-channel fidelity、FPGA latency/resource 等外部指标的可比性规则写成正式论文中的 comparison protocol。

## 修改内容

- 在 Related Work / comparison tables 之前加入三条读表规则：
  1. endpoint metric 不能跨层互换；
  2. 必须说明 adaptive object 是 outer-code graph、decoder prior、neural decoder state、message-passing likelihood，还是本文的 physical-layer affine surface \((K,b)\)；
  3. 必须把 runtime surface 与 estimator 分开，本文 fast path 是 deterministic clipped affine rule。
- 在外部比较表后补一段正式论文式解释，说明最有说服力的比较不是 leaderboard，而是：
  - analog-GKP / calibration-aware QEC 证明 continuous syndrome information 与 drift calibration 是重要问题；
  - learned-decoder / real-time-FPGA 工作定义 LER 与 latency/resource 的证据标准；
  - 本文的独特点是把 drift adaptation 暴露为低维 \((K,b)\) surface，并使 online arithmetic、state 和 future board check 可审计。
- 摘要中同步加入 nearest-syndrome 硬参照，避免与正文 controlled baseline 不一致。

## 可写边界

- 可以写：本文的优势是 interface-level / architecture-level advantage，具体表现为低维 runtime surface、fast-path arithmetic 可数、source-vs-board target 清楚。
- 可以写：外部 LER、fidelity、latency/resource 数值作为 comparison standards，而不是本文结果。
- 可以写：当前 manuscript 解释了为什么本文不能和 surface-GKP overhead、finite-energy logical-channel fidelity、real-time FPGA latency 做 normalized leaderboard。

## 不可外推边界

- 不得写成本文已达到外部文献中的 LER、logical-channel fidelity、closed-loop latency、resource/power 或 hardware scale。
- 不得把 literature metric crosswalk 写成实验 source data。
- 不得把 analytical cost / Q4.20 software parity 写成 FPGA synthesis、timing closure 或 real-board result。

## 验证口径

本记录绑定正文叙事补强与已有 `submission_draft_literature_metric_crosswalk.csv/json`。最终可信度以后续 LaTeX 编译、source-data audit 和内部项目语汇扫描为准。
