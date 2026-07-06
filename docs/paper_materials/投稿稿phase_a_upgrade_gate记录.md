# 投稿稿 Phase A validation-threshold gate 记录

日期：2026-07-06

本文档服务 `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex`。它把当前描述性结果、short-run rehearsal、formal repeat expansion、holdout drift 与 hardware measurements 分成独立验证门槛，防止把计划或短运行覆盖演练写成主结果。

## 生成文件

- `docs\paper_materials\submission_draft_phase_a_upgrade_gate.csv`
- `docs\paper_materials\submission_draft_phase_a_upgrade_gate.json`

## Gate matrix

| Evidence class | Current status | Upgrade condition | Forbidden inference |
| --- | --- | --- | --- |
| Current descriptive benchmark | satisfied for descriptive ranking | None; this is the current claim ceiling. | Do not state statistical significance, broad robustness, hardware latency or deployment readiness. |
| Short-run repeat rehearsal | 0 short-run scenario(s) complete; feasibility only | No stronger statement is permitted from short-run rows. | Do not use short-run rows as manuscript performance evidence or inferential uncertainty. |
| Formal Phase A repeat expansion | 2 formal scenario row(s) complete with positive paired interval; all-scenario gate still incomplete | Complete all four scenarios with at least 12 and preferably 16 paired repeats, no missing paired rows, and positive predeclared paired-interval lower bounds per scenario plus pooled analysis. | Do not claim repeat-expanded advantage, confidence intervals, p-values or robustness before this gate passes. |
| Formal holdout drift expansion | planned only | Run the predeclared random-walk, burst/reset and faster-than-window families with missing-run accounting. | Do not treat controlled stress diagnostics or anchor scenarios as holdout generalization proof. |
| Hardware-facing measurements | not measured | Provide board logs, bitstream or RTL hash, source vectors, measured latency/resource/power and source-vs-board agreement. | Do not claim FPGA timing closure, resource efficiency, source-vs-board agreement or board-level correction success. |

## 可写边界

- 可以写：稿件已有明确的 validation-threshold gate，区分 descriptive ranking、short-run rehearsal、formal repeat expansion、holdout expansion 和 hardware measurement。
- 可以写：short-run rehearsal 只验证执行路径和 collector，不改变主文性能结论。
- 不能写：当前材料已经提供 CI、p-value、repeat-expanded advantage、holdout robustness、hardware latency/resource 或 source-vs-board agreement。
