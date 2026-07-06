# 投稿稿 appendix scope 主文支撑层级补强记录

日期：2026-07-06

## 对象

- `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex`

## 修改内容

- 在 `\appendix` 后新增 `Appendix overview` 非编号段落。
- 明确主文到 `Outlook` 结束，后续 appendix 只承担 source-data routing、validation-scope tables、reproducibility limits 和 terminology controls。
- 明确 appendix 不引入强于 Results 与 Discussion 的新经验主张，只登记 evidence class、source artifact 与剩余 validation requirement。

## 证据边界

- 本次修改只调整稿件层级与读者导航。
- 不新增实验、不改数值、不升级统计推断、finite-energy fidelity、hardware、real-board、`.tflite`、benchmark 或 deployment 证据。
- 硬件相关内容仍是 future board-level measurement / planned validation，不是实板完成证据。

## 验证

- 预期验证：重新编译投稿稿，扫描 LaTeX warning、内部项目语体、审稿元语言残留、硬件/统计过强主张和行尾空白。
