# 投稿稿 nearest-syndrome 基线补强记录

日期：2026-07-03

本文档记录 `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex` 中 controlled baseline 的一次有界补强。补强目的不是新增正式 benchmark，而是把已经生成在 one-step source CSV 中、并新补入 short-sequence source CSV 的 `nearest_syndrome` 硬校正参照显式写入正文表格。

## 修改范围

- `docs/paper_materials/run_sequence_controlled_baseline_analysis.py` 新增 `nearest_syndrome` 分支。
- `docs/paper_materials/submission_draft_sequence_controlled_baseline_analysis.csv/json` 由脚本重新生成，新增四个 scenario 的 `nearest_syndrome` 行。
- `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex` 的 `tab:controlled-oracle-affine` 与 `tab:sequence-controlled-baselines` 新增 nearest-syndrome 列。
- `docs/figure_assets/submission_draft_python_figures/audit_submission_draft_source_data.py` 同步校验新增列，确保 TeX 表格仍由 source CSV 约束。

## 可写结论

- 可以写：在 one-step 和 short-sequence controlled local-Gaussian setting 中，direct nearest-syndrome 硬校正参照弱于 affine fast path，说明 noisy wrapped syndrome 直接作为 correction 会暴露测量噪声和 branch ambiguity 风险。
- 可以写：该参照帮助区分 hard-correction risk、oracle affine local-model benefit 与 wrapped-posterior branch design risk。
- 可以写：该结果加强了 calibrated affine fast path 作为工程路径的合理性。

## 不可外推边界

- 不得写成 tuned finite-energy nearest-lattice decoder。
- 不得写成正式 known-noise affine / nearest-lattice / wrapped-Gaussian benchmark 已完成。
- 不得写成 P4 software-HIL benchmark、holdout drift benchmark、CI/p-value、logical-channel fidelity、hardware timing/resource/source-vs-board evidence 或 `.tflite` 证据。
- 不得用该受控 CSV 替代 repeat-expanded benchmark 或硬件验证。

## 验证口径

本记录只绑定当前投稿稿 controlled-baseline 源数据、TeX 表格和 source-data audit helper。最终可信度仍以后续 mechanical audit、LaTeX 编译和 source-data manifest 重新生成为准。
