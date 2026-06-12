# Paper Notes

本目录专门存放论文 note、LaTeX 草稿及其已保留的编译产物。它从 `docs/follow-up_plan/` 中拆出，避免后续计划目录同时承担“路线维护”和“论文 note 存档”两种职责。

当前文件组：

- `CNN_FPGA_GKP_theory_note_draft.tex`：论文/theory note 源文件。
- `CNN_FPGA_GKP_theory_note_draft.pdf`：对应编译产物。
- `CNN_FPGA_GKP_theory_note_draft.*`：保留下来的 LaTeX 辅助/日志/同步文件。
- `../paper_materials/paper_note_results_sync_manifest.md`：`T77` 结果层同步 manifest；记录 note 的结果层 section 与 `T74/T75/T76` trace。
- `../paper_materials/paper_note_alignment_and_layout_closeout.md`：`T78` 非结果层校准、`statcalib` 层级降权与 LaTeX warning 收口记录。

注意：`.log`、`.fls`、`.fdb_latexmk` 等 LaTeX 辅助文件可能仍包含迁移前的历史编译路径 `docs/follow-up_plan/...`。这些是编译器记录的历史元数据，不是当前维护入口；重新编译后会按新目录刷新。

使用规则：

1. 本目录中的 note 可以作为论文写作素材，不作为当前项目完成态证据。
2. 若 note 中的表述涉及 benchmark、`.tflite`、HIL、real-board、statcalib 或投稿结论，必须先与 `docs/04_task_board.md`、`docs/07_handoff.md`、`docs/08_risks_and_open_questions.md` 对齐。
3. 重新编译后产生的辅助文件如果仍需保留，应继续放在本目录，不再放回 `docs/follow-up_plan/`。
4. `T77` 之后，结果层同步优先通过 `paper_note_results_sync_manifest.md` 与源码内的 `% T77-SOURCE: ...` 注释回链到 `T74/T75/T76` stable ID；摘要、结果段、讨论和结论之外的章节若未被该 manifest 覆盖，不得默认视为已同步。
5. `T78` 之后，标题、引言、`Relationship to Existing Work`、讨论、结论和 note 内部 `statcalib` 层级若做过进一步校准，应优先查看 `paper_note_alignment_and_layout_closeout.md` 与源码内的 `% T78-SCOPE: ...` 注释；这仍然只是 note 质量收口，不是证据升级。
