# T84 Worker Summary

## 改了什么

- 对 `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex` 做了受边界约束的 reader-facing final polish，实际覆盖 6 个 section：
  - `Summary of Contributions`
  - `Experimental Setup`
  - `Numerical Results`
  - `Follow-up routes that remain outside the accepted result layer`
  - `Discussion`
  - `Conclusion`
- 在上述 section 中补入 `% T84-POLISH: ...` 注释，并把偏内部的 `T24`、`FR8/statcalib`、`current-host NO_GO`、follow-up register 等说法，压成更接近读者稿的 frozen reference benchmark / supplement / blocked 语言。
- 新增 3 份 `paper_materials` 台账：
  - `paper_bounded_final_polish_change_map.md`
  - `paper_reader_facing_term_translation_table.md`
  - `paper_appendix_supplement_reader_assembly_map.md`
- 更新 `docs/paper_notes/README.md` 与 `docs/paper_materials/README.md`，登记 `T84` 入口，并把 `paper_materials/README.md` 的链路标题从 `T74-T82` 修正为 `T74-T84`。
- 刷新了 note 编译产物：
  - `CNN_FPGA_GKP_theory_note_draft.aux`
  - `CNN_FPGA_GKP_theory_note_draft.fdb_latexmk`
  - `CNN_FPGA_GKP_theory_note_draft.fls`
  - `CNN_FPGA_GKP_theory_note_draft.log`
  - `CNN_FPGA_GKP_theory_note_draft.out`
  - `CNN_FPGA_GKP_theory_note_draft.pdf`
  - `CNN_FPGA_GKP_theory_note_draft.synctex.gz`
  - `CNN_FPGA_GKP_theory_note_draft.toc`
- 新增本轮 `docs/review/T84_review.md` 与 `docs/for_human/T84_explanation.md`。

## 如何验证

- 使用 allowlist-scoped 状态检查，只核对 `T84` 允许路径内的改动，而不把全仓库脏状态误记成 `T84` 结果。
- 检查 note 源文件中的标记链，确认：
  - `% T80-REOPEN` 仍保留；
  - `% T81-CALIBRATION` 仍保留；
  - `% T82-SUPPORT` 仍保留；
  - `% T83-CLOSEOUT` 仍保留；
  - `% T84-POLISH` 已覆盖本轮实际修改的 6 个 section。
- 检查 3 份新增台账的必需表头，确认分别包含：
  - `section / touched_in_t84 / polish_goal / strongest_supported_truth_retained / untouched_boundary`
  - `internal_term / allowed_reader_facing_phrasing / forbidden_retelling / anchor`
  - `surface / recommended_destination / reader_facing_status / boundary_to_keep / next_bounded_action`
- 在 `docs/paper_notes/` 下执行：
  - `latexmk -g -pdf -synctex=1 -interaction=nonstopmode -halt-on-error CNN_FPGA_GKP_theory_note_draft.tex`
- 编译环境与结果：
  - 工具链：`TeX Live 2024 + latexmk`
  - 目标：`CNN_FPGA_GKP_theory_note_draft.tex`
  - 产物：`.aux/.fdb_latexmk/.fls/.log/.out/.pdf/.synctex.gz/.toc`
  - `.log` 关键字扫描：未检出 `Underfull`、`Overfull`、`LaTeX Warning`、`undefined`、`Citation`

## 剩余风险

- 当前 worktree 在进入 `T84` 前已经存在与本任务无关的脏状态，因此后续审查仍应继续使用 allowlist-scoped diff/status，而不是把全仓库状态直接当成 `T84` 变更清单。
- `T84` 只完成了 reader-facing translation / condensation / assembly，不等于：
  - submission-ready pack 已完成；
  - expanded benchmark 已补齐；
  - default-env `.tflite` portability 已收口；
  - real-board execution 已完成；
  - hardware-dependent surface 已解锁。
- 以下 guardrail 仍必须原样保留：
  - 主结果仍只是锁定四场景的 mock-backed software-HIL 排名；
  - `FR6/FR7` 仍只是 descriptive support；
  - `FR8/statcalib` 仍只是 supplement-side extension lane，且保持 no-promotion / no unique clean threshold；
  - `.tflite` 仍只是 isolated current-host true runtime；
  - real-board 仍只是 read-only gate / regeneration / provenance，当前 host 仍不能进入板级执行；
  - 缺失 `Linux + FPGA` host 的硬件依赖面仍是 blocked。
