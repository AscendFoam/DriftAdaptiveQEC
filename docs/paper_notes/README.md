# Paper Notes

本目录保存主线论文 `note`、LaTeX 草稿及其保留的编译产物。它是写作素材入口，不是项目完成态证据入口。

## 当前文件组

- `CNN_FPGA_GKP_theory_note_draft.tex`：当前主线 note 源文件。
- `CNN_FPGA_GKP_theory_note_draft.pdf`：对应编译产物。
- `CNN_FPGA_GKP_theory_note_draft.*`：保留的 LaTeX 辅助文件、日志与同步文件。
- `../paper_materials/paper_note_results_sync_manifest.md`：`T77` 结果层同步 manifest。
- `../paper_materials/paper_note_alignment_and_layout_closeout.md`：`T78` 非结果层校准、`statcalib` 层级降权与排版收口记录。
- `../paper_materials/paper_bounded_prose_reopen_manifest.md`：`T80` 的 ready-section bounded prose reopen manifest。
- `../paper_materials/paper_methods_and_contribution_calibration_manifest.md`：`T81` 的 contribution/methods calibration manifest。
- `../paper_materials/paper_supporting_material_closeout_pack.md`：`T82` 的 supporting-material closeout pack。
- `../paper_materials/paper_manuscript_closeout_readiness_matrix.md`：`T82` 的 manuscript-facing readiness matrix。
- `../paper_materials/paper_fullnote_consistency_crosswalk.md`：`T83` 的全文 section-to-evidence consistency crosswalk。
- `../paper_materials/paper_closeout_gate_and_blocker_register.md`：`T83` 的 closeout gate 与 blocker register。

## 使用规则

1. 本目录中的 note 可以作为论文写作素材，不作为当前项目完成态证据。
2. 若 note 文本涉及 benchmark、`.tflite`、HIL、real-board、`statcalib` 或投稿完成态，必须先与 `docs/04_task_board.md`、`docs/07_handoff.md`、`docs/08_risks_and_open_questions.md` 的当前边界对齐。
3. 重新编译后产生的 `.aux/.log/.fls/.fdb_latexmk/.synctex.gz/.toc/.out/.pdf` 若需保留，应继续放在本目录。
4. `T77` 之后，结果层同步优先查看 `paper_note_results_sync_manifest.md` 与源码中的 `% T77-SOURCE: ...` 注释；未被该链路覆盖的 section，不应默认视为“结果层已同步”。
5. `T78` 之后，标题、引言、`Relationship to Existing Work`、讨论、结论以及 note 内部 `statcalib` 层级的进一步校准，优先查看 `paper_note_alignment_and_layout_closeout.md` 与 `% T78-SCOPE: ...` 注释；这仍然只是 note 质量收口，不是证据升级。
6. `T80` 之后，若需判断当前 note 的 mainline prose 是否已经过 bounded reopen，应优先查看 `paper_bounded_prose_reopen_manifest.md` 与 `% T80-REOPEN: ...` 注释；该链路只覆盖 `Title`、`Abstract`、`Introduction`、`Relationship to Existing Work`、`Experimental Setup`、`Numerical Results`、`Discussion`、`Conclusion` 八个 ready sections。
7. `T81` 之后，若需判断 `Summary of Contributions` 与三章 methods 是否已经校准到当前 strongest supported truth，应优先查看 `paper_methods_and_contribution_calibration_manifest.md` 与 `% T81-CALIBRATION: ...` 注释；该链路只覆盖 `Summary of Contributions`、`Brief Review of the GKP Code`、`Noise and Drift Model`、`Model Architecture` 四个 target sections，不代表 full-manuscript reopen。
8. `T82` 之后，若需判断 supporting-boundary 段落是否已经按 `main text / appendix / supplement / blocked` 四层收口，应优先查看 `paper_supporting_material_closeout_pack.md`、`paper_manuscript_closeout_readiness_matrix.md` 与 `% T82-SUPPORT: ...` 注释；该链路只覆盖 `Runtime, quantization, and fixed-point degradation`、`Embedded runtime and board-level validation`、`Discussion` 中的 deployment/support boundary 段落、`Conclusion` 中的 remaining technical gap 段落，不代表 full-manuscript closeout。
9. `T83` 之后，如需判断当前 note 是否已经完成全文级 consistency sweep，以及后续是否只能进入 bounded final polish，应优先查看 `paper_fullnote_consistency_crosswalk.md`、`paper_closeout_gate_and_blocker_register.md` 与 `% T83-CLOSEOUT: ...` 注释；该链路只证明“当前主线 note 已形成可审计的一致性版本”，不等于 submission-ready pack、deployment closure 或 real-board success。
