# T77 Review

Verdict: `PASS_WITH_WARNINGS`

## Blocking issues

- 无。

## Non-blocking issues

- `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex` 的当前 `git diff` 仍混有不在 `T77` 允许同步范围内的既存 note hunks；最明显的是 `Relationship to Existing Work` 下的 `Advantages relative to existing QEC decoder approaches` 一段（现文件约 [docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex](D:/Codes/Quantum/DriftAdaptiveQEC/docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex:575)）。Worker 已在 [paper_note_results_sync_manifest.md](D:/Codes/Quantum/DriftAdaptiveQEC/docs/paper_materials/paper_note_results_sync_manifest.md:49) 明确声明这些章节“未同步”，并用 `% T77-SOURCE` 标记结果层新同步内容；因此这不是 `T77` 的 blocking failure，但当前整份 `.tex` 的 whole-file diff 不能直接当成“全部已按 T77 校准”的结果。
- `statcalib` 的边界口径已经在结果层被压回 `extension-lane / no-promotion / no unique clean threshold`，例如 [paper_note_results_sync_manifest.md](D:/Codes/Quantum/DriftAdaptiveQEC/docs/paper_materials/paper_note_results_sync_manifest.md:28) 和 `.tex` 中 [726-799](D:/Codes/Quantum/DriftAdaptiveQEC/docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex:726) 一带都写得比较谨慎；但这些表格和小节仍然留在 `Numerical Results` 主体内部，视觉上仍接近主结果层。对内部 note 这可以接受，但若后续要继续面向作者或外部读者传播，最好再做一次版面/层级降权，而不是只依赖文字提醒。
- LaTeX 编译已成功，PDF 和辅助文件均刷新，但当前 `.log` 仍有多处 `Underfull \hbox`（例如 [CNN_FPGA_GKP_theory_note_draft.log](D:/Codes/Quantum/DriftAdaptiveQEC/docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.log:405) 起）。这不影响 `T77` 的结果层同步与 traceability hardening 结论，但说明稿件排版仍不是最终收口状态。

## Missing tests

- 无新增代码测试缺口。`T77` 是 docs-only 任务，关键验证点是 source-map schema、note section trace、compile 结果与 git scope。
- 仍缺一个机械化的 section-scope 审计：当前对 `.tex` “只同步允许章节”的证明主要依赖 `% T77-SOURCE` 标记和 `paper_note_results_sync_manifest.md`，而不是一个自动化 diff guard。鉴于该文件本轮开始前就已有既存 diff，这个缺口暂时可接受，但如果后续继续做 note/manuscript 同步任务，最好补一个更确定的 section-range 检查。

## Suspicious implementation details

- `T76` traceability/schema 硬化本身是干净完成的：
  - [preview_source_map.csv](D:/Codes/Quantum/DriftAdaptiveQEC/docs/figure_assets/T76_rendered_figure_qa_pack/preview_source_map.csv:1) 已新增 `source_preview_ids`；
  - 聚合行 [5-6](D:/Codes/Quantum/DriftAdaptiveQEC/docs/figure_assets/T76_rendered_figure_qa_pack/preview_source_map.csv:5) 不再把 `T76-PREVIEW-*` 混写进 `upstream_t74_ids`；
  - [render_manifest.json](D:/Codes/Quantum/DriftAdaptiveQEC/docs/figure_assets/T76_rendered_figure_qa_pack/render_manifest.json) 与 [paper_rendered_figure_qa.md](D:/Codes/Quantum/DriftAdaptiveQEC/docs/paper_materials/paper_rendered_figure_qa.md:29) 已同步解释这条 schema。
- `paper_rendered_figure_qa.md` 现在确实逐图内联了 `T75-FIG-*`、`T76-PREVIEW-*` 和上游 `T74-*` stable IDs；这正对上了 `T76` 留下的 `R34` 型 warning。
- 当前 [docs/review/T77_review.md](D:/Codes/Quantum/DriftAdaptiveQEC/docs/review/T77_review.md) 在我覆盖之前是一个 Worker 自检包，而不是 adversarial review。它更像交付物内的 self-check placeholder，不是伪实现，但也说明这一路径需要 reviewer 最终重写，而不能直接沿用。
- `.tmp_t76_*` 和 `.tmp_t76_fontcache/*` 目前表现为受控删除 diff，且这些路径确实在 git index 中存在；本轮清理是任务包允许且应该做的 exact-path cleanup，不属于越界删除。

## Recommended next action

- 接受 `T77` 为完成态，但按 `PASS_WITH_WARNINGS` 收口。
- Captain closeout 时可以考虑把 `R34` 中属于 preview-source schema 与逐图 stable-ID 粒度的那部分风险关闭或显著缩窄；但要保留一条新的操作性提醒：
  - 当前 note 只有 `% T77-SOURCE` 覆盖到的结果层章节被明确同步；
  - 不要把整份 `CNN_FPGA_GKP_theory_note_draft.tex` 当作“已全稿校准”的材料。
- 如果下一步是 paper reopen gate，建议优先新开一张非常窄的 note/manuscript 校准任务，专门处理：
  - 非结果层章节与当前 evidence stack 的对齐；
  - `statcalib` 在 note 里的视觉层级降权；
  - LaTeX 排版 warning 的进一步收口。

## Reviewer verification notes

- 已核对 `git diff --name-only -- runs` 为空。
- 已核对 `git diff --name-only -- artifacts` 为空。
- 已核对 `git diff --name-only -- cnn_fpga physics benchmark tests` 为空。
- 已核对 `git diff --name-only -- docs/00_project_snapshot.md docs/01_legacy_audit.md docs/02_experiment_plan.md docs/03_hil_p4_boundary_audit.md docs/04_task_board.md docs/05_decision_log.md docs/06_repo_noise_governance.md docs/07_handoff.md docs/08_risks_and_open_questions.md` 为空。
- 已核对：
  - [preview_source_map.csv](D:/Codes/Quantum/DriftAdaptiveQEC/docs/figure_assets/T76_rendered_figure_qa_pack/preview_source_map.csv:1) 可解析，且 `preview_id` / `upstream_t74_ids` / `source_preview_ids` 语义已分开；
  - [paper_rendered_figure_qa.md](D:/Codes/Quantum/DriftAdaptiveQEC/docs/paper_materials/paper_rendered_figure_qa.md:29) 逐图显式列出 `T75-FIG-*`、`T76-PREVIEW-*` 和全部上游 `T74-*`；
  - [paper_note_results_sync_manifest.md](D:/Codes/Quantum/DriftAdaptiveQEC/docs/paper_materials/paper_note_results_sync_manifest.md:18) 覆盖本轮同步的结果层 section；
  - note 结果层里的 `T24` / `T48` / `T49/T71/T72` / `T64-T70` 边界写法仍然保守；
  - note 编译产物 [PDF](D:/Codes/Quantum/DriftAdaptiveQEC/docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.pdf) 与辅助文件已刷新，日志显示编译完成但仍有排版 warning。
