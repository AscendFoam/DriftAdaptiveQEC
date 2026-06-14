# T87 Worker Summary

## 改了什么

- 新增 4 份作者终检 / 投稿前 QA 文档：
  - `docs/paper_materials/paper_author_final_qa_checklist.md`
  - `docs/paper_materials/paper_presubmission_regression_gate.md`
  - `docs/paper_materials/paper_submission_wording_redflag_register.md`
  - `docs/paper_materials/paper_manual_finish_queue.md`
- 更新 `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`，只对 `Numerical Results`、`Discussion`、`Conclusion` 做最小 QA 导向刷新，并新增：
  - `% T87-QA: Numerical Results`
  - `% T87-QA: Discussion`
  - `% T87-QA: Conclusion`
- 更新 `docs/paper_notes/README.md` 与 `docs/paper_materials/README.md`，登记 `T87` 的 checklist / gate / red-flag / manual-finish 入口和 `% T87-QA` 使用规则。
- 新增 `docs/review/T87_review.md` 与 `docs/for_human/T87_explanation.md`，分别沉淀正式 review 结论和作者向解释材料。
- 刷新当前主机上的 note 编译产物：
  - `CNN_FPGA_GKP_theory_note_draft.fdb_latexmk`
  - `CNN_FPGA_GKP_theory_note_draft.fls`
  - `CNN_FPGA_GKP_theory_note_draft.log`
  - `CNN_FPGA_GKP_theory_note_draft.pdf`
  - `CNN_FPGA_GKP_theory_note_draft.synctex.gz`

## 如何验证

- 使用 allowlist-scoped `git status --short --untracked-files=all -- ...` 核对 `T87` 允许路径内的真实改动。
- 使用 allowlist-scoped `git diff --check -- ...` 验证本轮修改无内容级格式错误；当前仅剩 Windows 工作副本的 `LF -> CRLF` 提示，不构成内容缺陷。
- grep 主 note，确认：
  - `% T80-REOPEN`
  - `% T81-CALIBRATION`
  - `% T82-SUPPORT`
  - `% T83-CLOSEOUT`
  - `% T84-POLISH`
  - `% T85-PREFLIGHT`
  - `% T86-ASSEMBLY`
  仍然保留，且 `% T87-QA` 只出现在 `Numerical Results`、`Discussion`、`Conclusion`。
- 检查 4 份新台账列头，确认分别包含任务包要求字段：
  - `qa_id / surface_or_section / check_type / pass_condition / evidence_anchor / status / manual_note`
  - `redflag_id / forbidden_wording / why_wrong / allowed_replacement / evidence_anchor / scan_result`
  - `queue_id / allowed_manual_action / why_manual / depends_on / must_not_upgrade / owner`
  - `GO_FOR_BOUNDED_AUTHOR_MANUAL_FINISH_ONLY` 唯一 gate verdict
- 对当前 note/material 执行 red-flag 扫描，覆盖至少四类风险：
  - submission-ready completed
  - real-board execution success / hardware-ready
  - default-env / cross-host `.tflite` portability closed
  - full training reproducibility / mature `statcalib` comparator
  扫描结果已写入 `paper_submission_wording_redflag_register.md`。
- 在 `docs/paper_notes/` 下执行：
  - `latexmk -g -pdf -synctex=1 -interaction=nonstopmode -halt-on-error CNN_FPGA_GKP_theory_note_draft.tex`
- 编译记录：
  - 工具链：`TeX Live 2024 + latexmk`
  - 目标：`CNN_FPGA_GKP_theory_note_draft.tex`
  - 刷新产物：`.fdb_latexmk/.fls/.log/.pdf/.synctex.gz`
  - `.log` 关键字扫描：未检出 `Underfull`、`Overfull`、`LaTeX Warning`、`undefined`、`Citation`

## 剩余风险

- `T87` 的 gate verdict 只是 `GO_FOR_BOUNDED_AUTHOR_MANUAL_FINISH_ONLY`，不等于 submission-ready completed；后续人工终修仍必须服从 `paper_manual_finish_queue.md` 与 `paper_submission_wording_redflag_register.md`。
- 当前 note 中仍会以“负向边界提示”的形式出现 `submission-ready completion`、`hardware-ready finalization`、`default-environment portability`、`unique clean threshold` 等词串；这些在 `T87` 中被保留为 guardrail 语句，不是完成态 claim，但后续人工改稿时最容易被误删或误写强。
- 当前主机的 Git 噪声仍存在：
  - `C:\\Users\\26410/.config/git/ignore` 访问告警
  - `LF will be replaced by CRLF` 提示
  它们不构成内容错误，但人工审查时需要和真正的文档问题区分开。
- note 编译成功只代表这台主机上的 `TeX Live 2024 + latexmk` 链路可用，不外推到其他宿主、期刊模板或未来新的 manuscript 分支。
