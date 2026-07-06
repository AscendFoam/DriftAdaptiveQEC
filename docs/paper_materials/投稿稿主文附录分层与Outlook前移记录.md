# 投稿稿主文附录分层与 Outlook 前移记录

日期：2026-07-03

## 本次修改

本次只修改投稿稿的论文叙事结构，不新增实验、不升级证据等级。

改动集中在 `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex`：

1. 将 `Outlook` 从附录区移到 `Conclusion` 之后、`\appendix` 之前，使统计扩展、物理信道验证和硬件验证三条后续路径成为正式主文收束，而不是附录台账的一部分。
2. 保留 `Data, Code, and Reproducibility Availability` 及后续 source-data / validation tables 在 appendix 层，避免主文后半部分继续呈现为项目管理式清单。
3. 将附录中 “Before journal submission...” 的流程化表达改为 “For a stronger empirical version...” 的论文式表达，降低本地交付口吻。

## 审稿人口吻二次修订

随后对主文做了一轮更偏审稿人阅读体验的表述降噪：

1. 将 Introduction 中的防御性边界段落改为正面主张：检验 slow calibration module 是否能在 controlled drift 下改进 deterministic affine GKP fast path，同时明确当前证据仍是 simulation-based。
2. 将第四条 contribution 从 source-data package 改为 validation roadmap，避免把论文贡献写成材料清单。
3. 将 `Physics model and metric scope` 改名为 `Metric interpretation`，降低治理口吻。
4. 将 `Stage-and-commit runtime contract` 中的 “should record” 改为当前合同暴露的 observable quantities，并继续明确 board-level commit latency、source-vs-board agreement 和 resource use 未测。
5. 将 Discussion 和 Outlook 中的计划式段落压缩为 interface-level advantage、future validation axes 和 hardware-measurement requirements，避免把未完成硬件验证写成当前结果。

## 证据边界

- 本次没有引用 timed-out partial run，也没有新增 benchmark、HIL、`.tflite`、real-board 或 FPGA 结果。
- `Outlook` 中的 hardware validation 仍是 future measurement track，不是当前硬件完成事实。
- 附录中的 source-data / reproducibility tables 仍是支撑材料路线图，不把 `final_ler` proxy 改写为 finite-energy logical-channel fidelity，也不把 software fixed-point parity 改写为 FPGA timing closure。

## 预期作用

这次修改面向审稿人的阅读路径：主文先给出问题、方法、结果、比较优势、限制和前景；附录再承接 source-data、scope、artifact 和 reproducibility 清单。这样稿件更像论文，而不是把治理/审计清单直接铺在主文叙事中。
