# T77 Worker Summary

## 改了什么

1. 修补了 `T76` 预览 trace schema：
   - `preview_source_map.csv` 新增 `source_preview_ids`
   - contact sheet / PDF 聚合行不再把 `T76-PREVIEW-*` 混写进 `upstream_t74_ids`
   - `render_manifest.json`、`docs/figure_assets/T76_rendered_figure_qa_pack/README.md`、`paper_rendered_figure_qa.md` 同步解释这条 schema
2. 强化了逐图 QA traceability：
   - `paper_rendered_figure_qa.md` 为 `T75-FIG-M01`、`T75-FIG-M02`、`T75-FIG-A01` 逐条补写 `T76-PREVIEW-*` 与全部上游 `T74-*`
3. 同步了 note 的结果层：
   - 更新 `Abstract`
   - 更新 `Summary of Contributions`
   - 更新 `Experimental Setup`
   - 更新 `Numerical Results and Benchmark Plan`
   - 更新 `Discussion`
   - 更新 `Conclusion`
   - 在相关段落旁加入 `% T77-SOURCE: ...` 注释
4. 新增 `docs/paper_materials/paper_note_results_sync_manifest.md`，把 section-level source chain、允许/禁止表述和未同步内容集中记录下来。
5. 更新了 `paper_results_section_assembly_pack.md` 与 `paper_results_callout_sheet.md`，把 note-sync 的主图顺序、appendix bridge 和 `statcalib` supplement-side 边界补齐。
6. 更新了 `docs/paper_notes/README.md` 与 `docs/paper_materials/README.md`，把 `T77` 的 note-sync 入口和使用规则登记进去。

## 如何验证

- `preview_source_map.csv`
  - `Import-Csv` 解析成功
  - `source_preview_ids` 列已存在
  - `upstream_t74_ids` 现只包含 `T74-*`
- `paper_rendered_figure_qa.md`
  - 已逐图显式列出 `T75-FIG-*`、`T76-PREVIEW-*` 与上游 `T74-*`
  - 已写明 `T76-PREVIEW-CS01` / `T76-PREVIEW-PDF01` 的聚合 trace
- `paper_note_results_sync_manifest.md`
  - 已覆盖本轮同步的 `Abstract`、`Summary of Contributions`、`Experimental Setup`、`Numerical Results and Benchmark Plan`、`Discussion`、`Conclusion`
- note 边界检查
  - `locked four-scenario`
  - `isolated current-host`
  - `NO_GO`
  - `no-promotion`
  - `unique clean threshold`
  这些关键边界仍保留在 note 中
- note 章节范围检查
  - 本轮新增的 `% T77-SOURCE: ...` 注释全部位于允许章节
  - 需要单独记录：该 `.tex` 文件在本轮开始前就已有既存 diff，所以“仅限 T77 新改动”的核验以 `T77-SOURCE` 标记为准，而不能把整文件历史 diff 全部归因于本轮
- 本地 LaTeX 工具链诊断与编译
  - `latex_doctor.py --json` 在设置 `PYTHONUTF8=1` 后确认 `TeX Live 2024` 可用
  - `compile_latex.py ... --json` 编译成功，已刷新 `pdf/aux/fdb_latexmk/fls/log/out/synctex.gz/toc`
- `.tmp_t76_*` / `.tmp_t76_fontcache/`
  - 已全部删除
- `git diff --name-only -- runs`
  - 为空
- `git diff --name-only -- artifacts`
  - 为空
- `git diff --name-only -- cnn_fpga physics benchmark tests`
  - 为空
- `git diff --name-only -- docs/00_project_snapshot.md docs/01_legacy_audit.md docs/02_experiment_plan.md docs/03_hil_p4_boundary_audit.md docs/04_task_board.md docs/05_decision_log.md docs/06_repo_noise_governance.md docs/07_handoff.md docs/08_risks_and_open_questions.md`
  - 为空

## 剩余风险

1. `Title` 仍未同步，因为 `T77` 不允许扩大到标题级重写。
2. `statcalib` 原始表格仍保留在 note 结果章节中，所以后续 review 仍需重点检查“extension-lane / no-promotion”口径是否足够醒目。
3. note 编译虽已成功，但 `.log` 仍有 underfull hbox 与未解引用/引文 warning；这说明稿件整体排版与参考文献链路还没有完全收口。
4. `CNN_FPGA_GKP_theory_note_draft.tex` 在本轮开始前就已有既存 diff；后续 reviewer 若只看整文件 `git diff`，需要避免把这些既有历史差异误判成 `T77` 越界修改。
