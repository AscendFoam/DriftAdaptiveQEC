# T78 Review

Verdict: `PASS`

## Blocking issues

- 无。

## Non-blocking issues

- `T78` 只校准了任务包允许的 note 范围，未触碰 `Brief Review of the GKP Code`、`Noise and Drift Model`、`Model Architecture` 主体和 `Experimental Setup`。这不是遗漏，但也不应被转述成“整份 note 已全稿重校准”；这一点已在 `docs/paper_materials/paper_note_alignment_and_layout_closeout.md:65` 明确写出。
- `statcalib` 的层级降权这次主要发生在 `Numerical Results` 内部：三段结果说明已从同级 `\subsection` 降为更低层级的 `\subsubsection`，并增加了 supplement-side bridge 句；但方法章节中原有的 `Statistical calibration branch` 结构仍保留，后续如果真的进入 full-manuscript reopen，最好再做一次全文层级一致性检查。见 `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex:501` 与 `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex:735`。

## Missing tests

- 无阻塞性测试缺口。`T78` 是 docs-only 任务，关键验证已经覆盖到：
  - note 文本 diff 是否仍落在允许 section；
  - `% T78-SCOPE` 是否覆盖本轮非结果层改动；
  - `statcalib` 是否保留 `extension lane / no-promotion / persistent tie / no unique clean threshold`；
  - LaTeX 编译产物是否刷新，以及 `Underfull \hbox` 是否收敛；
  - `runs/`、`artifacts/`、源码目录和 `docs/00-08` 治理文档是否保持零 diff。
- 如果后续还会频繁做 note/manuscript 层同步，可以考虑补一个脚本化的 section-range guard；但这不构成 `T78` 的缺测阻塞。

## Suspicious implementation details

- 未发现伪实现、mock、stub、hardcode 冒充完成态的问题。`T78` 的核心改动确实是：
  - 非结果层 wording 校准；
  - `statcalib` 的 note 内部层级降权；
  - LaTeX layout warning 收口；
  - README / manifest / closeout 文档同步。
- 当前工作树没有显示任何 `cnn_fpga/`、`physics/`、`benchmark/`、`tests/`、`runs/`、`artifacts/` 或 `docs/00-08` 的改动，说明这次收口没有顺手越界。
- Worker 预先写入的 `docs/review/T78_review.md` 只是自检占位，不是正式 reviewer verdict；正式结论应以本次覆盖后的 review 为准。

## Recommended next action

- 可按 `PASS` 接受 `T78`，并在 Captain closeout 时把 `T77` 遗留的 note-calibration / hierarchy / layout 类 warning 视为已完成收口或显著缩窄。
- 下一步如果要继续推进论文材料，建议先走一张很窄的 paper reopen gate，而不是直接恢复 full-manuscript 扩写；重点判断当前 note、results pack、claim/evidence ledger 和风险表是否已经足以支撑下一轮 prose 扩展。

## Reviewer verification notes

- 已核对 `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex` 的本轮 diff 只落在任务包允许的区域：
  - 标题与摘要：`docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex:11`
  - 引言 / Summary of Contributions：`docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex:126`
  - `Relationship to Existing Work`：`docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex:583`
  - `Numerical Results` 中 `statcalib` bridge 与三段小节：`docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex:735`
  - `Discussion` / `Conclusion`：`docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex:1024`
- 已核对 `% T78-SCOPE` 注释覆盖了本轮非结果层 section，closeout 文档也显式区分了“已校准”和“未校准”范围：`docs/paper_materials/paper_note_alignment_and_layout_closeout.md:13`、`docs/paper_materials/paper_note_alignment_and_layout_closeout.md:65`、`docs/paper_materials/paper_note_alignment_and_layout_closeout.md:106`。
- 已核对 `statcalib` 的结果层视觉降权是结构性的，不只是文字提醒：三段标题已经从 `\subsection` 降为 `\subsubsection`，并增加了 supplement-side bridge 句，见 `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex:735`。
- 已核对 `HEAD` 基线中的 `Underfull \hbox = 32`，当前工作树 `Underfull \hbox = 0`；当前 `.log` 里没有残余 `Underfull \hbox`、`Overfull \hbox` 或真实 `pdfTeX warning`。
- 已核对：
  - `git diff --name-only -- runs` 为空；
  - `git diff --name-only -- artifacts` 为空；
  - `git diff --name-only -- cnn_fpga physics benchmark tests` 为空；
  - `git diff --name-only -- docs/00_project_snapshot.md docs/01_legacy_audit.md docs/02_experiment_plan.md docs/03_hil_p4_boundary_audit.md docs/04_task_board.md docs/05_decision_log.md docs/06_repo_noise_governance.md docs/07_handoff.md docs/08_risks_and_open_questions.md` 为空。
