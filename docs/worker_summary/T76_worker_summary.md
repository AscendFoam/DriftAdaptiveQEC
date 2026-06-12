# T76 Worker Summary

## 改了什么

本轮只修改了 `T76` 任务包允许的 paper-facing 路径，并完成了四组交付：

1. 真实渲染预览产物：
   - `docs/figure_assets/T76_rendered_figure_qa_pack/t75_fig_m01_preview.png`
   - `docs/figure_assets/T76_rendered_figure_qa_pack/t75_fig_m02_preview.png`
   - `docs/figure_assets/T76_rendered_figure_qa_pack/t75_fig_a01_preview.png`
   - `docs/figure_assets/T76_rendered_figure_qa_pack/t75_preview_contact_sheet.png`
   - `docs/figure_assets/T76_rendered_figure_qa_pack/t75_preview_bundle.pdf`
2. 预览 trace / QA 文档：
   - `docs/figure_assets/T76_rendered_figure_qa_pack/README.md`
   - `docs/figure_assets/T76_rendered_figure_qa_pack/render_manifest.json`
   - `docs/figure_assets/T76_rendered_figure_qa_pack/preview_source_map.csv`
   - `docs/figure_assets/T76_rendered_figure_qa_pack/visual_qa_checklist.md`
3. `paper_materials` 侧作者文档：
   - 新建 `paper_rendered_figure_qa.md`
   - 新建 `paper_results_section_assembly_pack.md`
   - 新建 `paper_results_callout_sheet.md`
   - 更新 `docs/paper_materials/README.md`
4. 为真实渲染修正做的 presentation-only 同步：
   - 更新 `T75` 三张 SVG 与其 `README.md` / `authoring_manifest.json` / `asset_source_map.csv`
   - 更新 `paper_maintext_results_authoring_pack.md`
   - 更新 `paper_caption_lock_and_placement_notes.md`
   - 更新 `paper_appendix_bridge_pack.md`
   - 更新 `paper_authoring_do_not_write_list.md`
   - 新建 `docs/review/T76_review.md`
   - 新建 `docs/for_human/T76_explanation.md`
   - 新建本文件

## 如何验证

我实际执行并确认了以下事项：

1. 使用 bundled Node + `sharp 0.34.5` 对三张 `T75-FIG-*` SVG 做真实栅格化，输出三张 preview PNG。
2. 使用 bundled Python + `Pillow` + `reportlab` 生成 contact sheet PNG 与四页 PDF bundle。
3. 人工查看真实渲染结果，确认：
   - `T75-FIG-M01` 的图例说明与 footer 已完整显示；
   - `T75-FIG-M02` 的 footer 已完整显示；
   - `T75-FIG-A01` 的三层长句裁切已消失，blocked slot / appendix role / footer 可读。
4. `render_manifest.json` 解析成功，并包含：
   - `T76-PREVIEW-M01`
   - `T76-PREVIEW-M02`
   - `T76-PREVIEW-A01`
   - `T76-PREVIEW-CS01`
   - `T76-PREVIEW-PDF01`
5. `preview_source_map.csv` 的 preview ID 集合与 manifest 一致。
6. `paper_rendered_figure_qa.md` 已显式覆盖 `T75-FIG-M01`、`T75-FIG-M02`、`T75-FIG-A01`。
7. `paper_results_section_assembly_pack.md` 已给出主文顺序、图表放置、fallback 与 boundary notes。
8. `paper_results_callout_sheet.md` 已给出 callout、asset ID、allowed wording、forbidden wording。
9. `T75` 相关回链仍然成立：
   - `authoring_manifest.json` 可解析；
   - `asset_source_map.csv` 继续把 `T75-FIG-*` 映射回 `T74-*`；
   - 本轮只做 presentation-only 修正，没有改 stable ID 或上游证据角色。

## 剩余风险

1. 这些 preview 解决的是当前 host 下的真实可读性，不等于最终出版社模板导出的版面完全锁定。
2. `T75-FIG-M01` 即使现在已可稳定渲染，仍然只是 `T74-TBL-01` 的 visual compression；若期刊更偏表格，应直接退回 `T74-TBL-01`。
3. `T75-FIG-M02` 仍只支持 descriptive mechanism/intervention reading，不能被作者借“图更清楚了”写成 causal closure。
4. `T75-FIG-A01` 仍然只是 appendix-only boundary schematic；`.tflite`、real-board、`FR8`、`T74-FIG-04` 的边界没有因为 rendered QA 而变强。
5. 本轮尝试清理 `.tmp_t76_*` 临时探针文件时，桌面端权限审批两次超时；若这些文件仍在工作区，它们属于过程噪声，需要在后续提交前单独确认是否删除。
