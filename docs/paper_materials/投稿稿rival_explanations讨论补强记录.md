# 投稿稿 rival explanations 讨论补强记录

日期：2026-07-03

本记录服务 `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex` 的 Discussion 补强。修改目标是让正文主动回答审稿人可能提出的替代解释，而不是只重复主结果。

## 修改位置

- `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex`
- `\section{Discussion}` 中新增一段 alternative explanations / controls 讨论。

## 新增论点

新增段落把当前证据约束为五个层次：

1. paired rows 与 LER advantage-margin readout 支持现有 repeats 下方向一致，但 `n=2` 仍不能写成 inferential statistics。
2. ablation 与 statistical-calibration 结果说明优势不能简单写成 “CNN because neural”，更稳妥的科学对象是 affine commit contract。
3. residual-boundary channel surrogate 明确 `final_ler` 与 q/p half-lattice crossing events 的关系，避免把 proxy 写成 finite-energy channel fidelity。
4. oracle / wrapped-Gaussian checks 说明 posterior-style rule 不自动替代 affine contract，但也要求后续补 stronger known-noise / sequence-level baselines。
5. 硬件相关优势仍只停留在接口和验证目标层面，不写成 board result。

## 可写边界

- 可以写：正文 Discussion 主动处理 statistical fluke、CNN-specific explanation、metric artifact、posterior-baseline replacement 与硬件推断等反解释。
- 可以写：这些反解释被当前 source-data 和 controlled diagnostics 部分约束，但没有被完全消除。
- 不可以写：这些段落新增了实验、显著性、holdout robustness、logical-channel fidelity、real-board latency 或 deployment evidence。

## 验证

- 需要重新运行 source-data audit，确认新增 discussion prose 不破坏已审计表格。
- 需要重新编译投稿稿并扫描 LaTeX log。
- 需要继续扫描内部项目词，避免把任务编号或治理语气带入正式稿件。
