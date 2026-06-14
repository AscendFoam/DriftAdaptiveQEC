# T88 Worker Summary

## 改了什么

- 新增 5 份 `T88` manual-finish / surface-freeze 文档：
  - `docs/paper_materials/paper_manual_finish_execution_log.md`
  - `docs/paper_materials/paper_mainline_surface_freeze_manifest.md`
  - `docs/paper_materials/paper_author_edit_decision_register.md`
  - `docs/paper_materials/paper_blocked_surface_disclaimer_table.md`
  - `docs/paper_materials/paper_frozen_mainline_handoff_gate.md`
- 更新 `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`，只对以下 section 做最小 manual-finish 落地，并补上：
  - `% T88-MANUAL: Numerical Results`
  - `% T88-MANUAL: Mechanism probe for residual-b behavior`
  - `% T88-MANUAL: Discussion`
  - `% T88-MANUAL: Conclusion`
- 更新 `docs/paper_notes/README.md` 与 `docs/paper_materials/README.md`，登记 `T88` 的 execution log / freeze manifest / decision register / disclaimer table / handoff gate 入口。
- 新增 `docs/review/T88_review.md` 与 `docs/for_human/T88_explanation.md`。
- 刷新当前主机上的 note 编译产物：
  - `CNN_FPGA_GKP_theory_note_draft.fdb_latexmk`
  - `CNN_FPGA_GKP_theory_note_draft.fls`
  - `CNN_FPGA_GKP_theory_note_draft.log`
  - `CNN_FPGA_GKP_theory_note_draft.pdf`
  - `CNN_FPGA_GKP_theory_note_draft.synctex.gz`

## 如何验证

- 使用 allowlist-scoped `git status --short --untracked-files=all -- ...` 核对 `T88` 允许路径内的真实改动。
- 使用 allowlist-scoped `git diff --check -- ...` 验证本轮修改无内容级格式错误；当前仅剩 Windows 工作副本的 `LF -> CRLF` 提示，不构成内容缺陷。
- grep 主 note，确认：
  - `% T80-REOPEN`
  - `% T81-CALIBRATION`
  - `% T82-SUPPORT`
  - `% T83-CLOSEOUT`
  - `% T84-POLISH`
  - `% T85-PREFLIGHT`
  - `% T86-ASSEMBLY`
  - `% T87-QA`
  仍然保留，且 `% T88-MANUAL` 只覆盖本轮实际修改的 4 个 section / subsection。
- 检查 5 份新台账列头与 gate verdict，确认包含任务包要求字段，并且唯一 verdict 为 `GO_FOR_FROZEN_MAINLINE_HANDOFF_ONLY`。
- 检查 `paper_manual_finish_execution_log.md`，确认 `MF01-MF05` 全部被标记为 `executed` 或 `left_as_is`。
- 对当前 note/material 执行 red-flag 复扫，覆盖至少以下风险：
  - submission-ready completed
  - real-board execution success / hardware-ready
  - default-env / cross-host `.tflite` portability closed
  - full training reproducibility / mature `statcalib` comparator
  复扫结果已写入 `paper_frozen_mainline_handoff_gate.md`。
- 在 `docs/paper_notes/` 下执行：
  - `latexmk -g -pdf -synctex=1 -interaction=nonstopmode -halt-on-error CNN_FPGA_GKP_theory_note_draft.tex`
- 编译记录：
  - 工具链：`TeX Live 2024 + latexmk`
  - 目标：`CNN_FPGA_GKP_theory_note_draft.tex`
  - 刷新产物：`.fdb_latexmk/.fls/.log/.pdf/.synctex.gz`
  - `.log` 关键字扫描：未检出 `Underfull`、`Overfull`、`LaTeX Warning`、`undefined`、`Citation`

## 剩余风险

- `T88` 的 gate verdict 只是 `GO_FOR_FROZEN_MAINLINE_HANDOFF_ONLY`，不等于 submission-ready completed；后续任何人工维护都必须继续服从当前 frozen surface 与 disclaimer table。
- 当前 note 中仍保留一些负向 guardrail 词串，例如 `submission-ready completion`、`hardware-ready finalization`、`default-environment compatibility`；它们现在是边界提示，不是完成态 claim，但后续人工终修时仍最容易被误删或误写强。
- `MF04` 被明确保留为 `left_as_is`：当前 note 没有内嵌 boundary schematic 的独立 figure/caption，因此 caption 仍依赖 `T75/T74` 外部锁定稿；后续如果有人把 schematic 真的插回 note，就必须重新开 bounded 任务而不是直接手改。
- 当前宿主机的 Git 噪声仍存在：
  - `C:\\Users\\26410/.config/git/ignore` 访问告警
  - `LF will be replaced by CRLF` 提示
  它们不构成内容错误，但人工审查时需要和真正问题区分。
- note 编译成功只代表这台主机上的 `TeX Live 2024 + latexmk` 链路可用，不外推到其他宿主、期刊模板或未来新的 manuscript 分支。
