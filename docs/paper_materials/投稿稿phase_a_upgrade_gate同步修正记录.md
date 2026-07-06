# 投稿稿 Phase A upgrade gate 同步修正记录

日期：2026-07-06

## 修正对象

- `docs/paper_materials/build_submission_draft_phase_a_upgrade_gate.py`
- `docs/paper_materials/submission_draft_phase_a_upgrade_gate.csv`
- `docs/paper_materials/submission_draft_phase_a_upgrade_gate.json`
- `docs/paper_materials/投稿稿phase_a_upgrade_gate记录.md`
- `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex`

## 修正原因

`submission_draft_phase_a_paired_interval_analysis.csv` 的 claim status 已改为 completed-scenario 泛化口径，但 `build_submission_draft_phase_a_upgrade_gate.py` 仍按旧的 one-scenario status 字符串识别 positive interval row。与此同时，投稿稿正文的 `tab:phase-a-upgrade-gate` 把 short-run rehearsal 写成“一个 smoke scenario 已完成 12 paired repeats”，这会把 short-run rehearsal 与 formal `static_bias_theta` one-scenario interval 混在一起。

## 修正内容

- upgrade-gate helper 改为用 `interval_lower_bounds_positive=true` 识别 positive formal interval row，不再依赖旧的 claim-status 字符串。
- 重新生成 Phase A upgrade gate CSV、JSON 与中文记录。
- 投稿稿正文 `tab:phase-a-upgrade-gate` 的 Short-run repeat rehearsal 行改为：当前 Phase A collector output 中没有 complete short-run scenario row；该层只检查 command shape、row accounting 与 collector logic。
- formal repeat expansion 行继续保留当前 narrow truth：仅一个 formal-length scenario row 完成并具有 positive paired-interval lower bounds，all-scenario gate 仍未完成。

## 不可外推边界

- 本修正不运行 benchmark。
- 本修正不新增 LER、confidence interval、p-value、holdout robustness、training reproducibility、latency/resource、source-vs-board 或硬件测量。
- 本修正不把 `static_bias_theta` one-scenario formal interval 改写成 all-scenario repeat-expanded advantage。

## 验证

- 重新运行 `python docs\paper_materials\build_submission_draft_phase_a_upgrade_gate.py`。
- 检查 `submission_draft_phase_a_upgrade_gate.csv` 中 short-run status 为 `0 short-run scenario(s) complete; feasibility only`。
- 检查投稿稿正文不再出现旧的 short-run 误句。
