# T85 Worker Summary

## 改了什么

- 对 `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex` 做了最小的 residual wording-lag 清扫，只实际修改了：
  - `Discussion`
  - `Conclusion`
- 在这两个 section 中补入 `% T85-PREFLIGHT: ...` 注释，并把 `T84` 已完成的 reader-facing polish 不再写成未来待办，而是收紧为“下一步若继续推进，只是 bounded submission-facing assembly”。
- 新增 3 份 `paper_materials` 台账：
  - `paper_submission_readiness_preflight_gate.md`
  - `paper_submission_blocker_matrix.md`
  - `paper_residual_state_lag_sweep.md`
- 更新 `docs/paper_notes/README.md` 与 `docs/paper_materials/README.md`，登记 `T85` 入口，并把写作链路标题从 `T74-T84` 修正为 `T74-T85`。
- 刷新了 note 编译产物：
  - `CNN_FPGA_GKP_theory_note_draft.aux`
  - `CNN_FPGA_GKP_theory_note_draft.fdb_latexmk`
  - `CNN_FPGA_GKP_theory_note_draft.fls`
  - `CNN_FPGA_GKP_theory_note_draft.log`
  - `CNN_FPGA_GKP_theory_note_draft.out`
  - `CNN_FPGA_GKP_theory_note_draft.pdf`
  - `CNN_FPGA_GKP_theory_note_draft.synctex.gz`
  - `CNN_FPGA_GKP_theory_note_draft.toc`
- 新增本轮 `docs/review/T85_review.md` 与 `docs/for_human/T85_explanation.md`。

## 如何验证

- 使用 allowlist-scoped status，只核对 `T85` 允许路径内的改动，而不把整仓库脏状态误记成 `T85` 输出。
- 检查 note 源文件中的标记链，确认：
  - `% T80-REOPEN` 仍保留；
  - `% T81-CALIBRATION` 仍保留；
  - `% T82-SUPPORT` 仍保留；
  - `% T83-CLOSEOUT` 仍保留；
  - `% T84-POLISH` 仍保留；
  - `% T85-PREFLIGHT` 已覆盖本轮实际修改的 `Discussion` 与 `Conclusion`。
- 检查 note 源文件，确认已无法 grep 到：
  - `The remaining writing work is to translate these internal layers into a final reader-facing polish pass`
  - `remaining manuscript-side work is reader-facing condensation and route cleanup`
- 检查 3 份新增台账的必需表头，确认分别包含：
  - `blocker_id / blocker_type / affected_surface / why_not_ready / next_bounded_task`
  - `location / stale_wording_summary / action_taken / boundary_preserved`
  - `paper_submission_readiness_preflight_gate.md` 只保留一个 verdict：`GO_FOR_BOUNDED_SUBMISSION_PACK_ASSEMBLY`
- 在 `docs/paper_notes/` 下执行：
  - `latexmk -g -pdf -synctex=1 -interaction=nonstopmode -halt-on-error CNN_FPGA_GKP_theory_note_draft.tex`
- 编译环境与结果：
  - 工具链：`TeX Live 2024 + latexmk`
  - 目标：`CNN_FPGA_GKP_theory_note_draft.tex`
  - 产物：`.aux/.fdb_latexmk/.fls/.log/.out/.pdf/.synctex.gz/.toc`
  - `.log` 关键字扫描：未检出 `Underfull`、`Overfull`、`LaTeX Warning`、`undefined`、`Citation`

## 剩余风险

- 当前 worktree 在进入 `T85` 前已经存在与本任务无关的脏状态，因此后续审查仍应继续使用 allowlist-scoped diff/status，而不是把全仓库状态直接当成 `T85` 变更清单。
- `paper_submission_readiness_preflight_gate.md` 给出的 `GO_FOR_BOUNDED_SUBMISSION_PACK_ASSEMBLY` 只表示“允许打开下一张受边界约束的 submission-facing assembly 任务”，不等于：
  - submission-ready pack 已完成；
  - deployment closure 已完成；
  - real-board execution 已完成；
  - default-env `.tflite` portability 已收口；
  - full training reproducibility 已收口；
  - expanded benchmark 已补齐。
- 以下 blocker / guardrail 仍必须原样保留：
  - board-level execution / timing / resource surface 仍受 `Linux + FPGA` host 缺失约束；
  - `.tflite` 仍只是 isolated current-host true runtime；
  - training/material 仍只是 canonical chain intact + one clean CPU-only bounded rerun；
  - `FR8/statcalib` 仍只是 extension lane，保持 no-promotion / no unique clean threshold；
  - real-board 仍只是 read-only gate / regeneration / provenance，当前 host 仍不能进入 board execution。
