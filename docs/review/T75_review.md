# T75 Review

Verdict: `PASS`

## Blocking issues

- 无。

## Non-blocking issues

- 当前工作区仍有 `T75` 之外的既有治理文档 diff。它们不是本次 `T75` 交付造成的，也不构成阻塞项，但后续提交时应精确暂存，避免把无关治理变更一并带入。
- `T75` 的三张最终图目前完成了文件存在、`<svg` 结构、XML 解析、manifest/source-map 回链等只读验证，但没有看到一份明确的“已渲染人工预览”证据。考虑到任务包本身只要求结构级验证，这一点不阻塞通过；但如果后续真要进入投稿排版，仍建议补一次真实渲染预览，确认文字不重叠、颜色/图例可读、版式符合期刊模板。

## Missing tests

- 无新增代码测试缺口。`T75` 是 docs-only 的 paper-facing authoring 任务，关键验证点是：
  - `authoring_manifest.json` 与 `asset_source_map.csv` 的 ID 和路径一致性；
  - 三张 `T75-FIG-*` SVG 真实存在且结构可解析；
  - `paper_maintext_results_authoring_pack.md` 是否显式绑定上游 `T74-*` stable IDs，并给出 safe / forbidden wording；
  - `paper_authoring_do_not_write_list.md` 是否覆盖 `.tflite`、real-board、`FR8`、blocked slot 等主要 overclaim 风险。
- 可补强但非必需的验证是：对三张 SVG 做一次真实渲染预览或导出预览，作为投稿前的视觉 QA。

## Suspicious implementation details

- 未发现把 `T75` 写成新实验、full-manuscript reopen、或 evidence-level upgrade 的问题。
- `T75-FIG-M01` 被正确写成 `T74-TBL-01` 的 publication-facing visual compression，并保留了 “若版面不适合则退回 `T74-TBL-01`” 的 authoritative substitute 口径。
- `T75-FIG-M02` 继续保持 `FR6` 的 descriptive mechanism/intervention 语义，没有把 mixed / mostly harmful 的 lower-clip intervention 误写成修复证据或机制闭环。
- `T75-FIG-A01` 明确保留了 `T74-FIG-04` 的 blocked slot，没有把分层边界证据压扁成统一 portability/deployment closure 叙事。
- `paper_authoring_do_not_write_list.md` 对 `T48`、`T49/T71/T72`、`FR8`、`FR6/FR7`、`T24` 的主要 overclaim 口径做了显式冻结，方向正确。

## Recommended next action

- 后续若继续写论文主文，可直接以 `paper_maintext_results_authoring_pack.md` 和 `paper_caption_lock_and_placement_notes.md` 为主入口，主文优先使用 `T75-FIG-M01`、`T75-FIG-M02`。
- Appendix / supplement 继续按 `paper_appendix_bridge_pack.md` 分层放置，不要把 `T74-TBL-06`、`T74-TBL-07` 或 `T74-SUP-*` 抬升成主结果。
- 若下一步要进入真正的稿件排版或投稿包整合，建议新开一个只处理 “rendered figure QA + manuscript assembly” 的有界任务，而不是回到 `T75` 里继续扩 scope。

## Reviewer verification notes

- 任务要求的四份 authoring 文档、三张 task-local SVG、`authoring_manifest.json`、`asset_source_map.csv`、目录 `README`、worker summary、review、for-human 文档均已落地。
- 机器核查结果为：
  - `authoring_manifest.json` 可解析，且只包含 `T75-FIG-M01`、`T75-FIG-M02`、`T75-FIG-A01` 三个资产 ID；
  - 每个 `T75` 资产都映射到至少一个上游 `T74-*` stable ID；
  - `asset_source_map.csv` 中的 `t75_asset_id` 唯一集合与 manifest 一致；
  - 三张 SVG 文件均存在、均包含 `<svg`，且 XML 解析成功；
  - `paper_maintext_results_authoring_pack.md`、`paper_caption_lock_and_placement_notes.md`、`paper_appendix_bridge_pack.md` 都显式引用了同一套 `T75-FIG-*` 资产 ID。
- diff 边界核查结果为：
  - `git diff --name-only -- runs` 为空；
  - `git diff --name-only -- artifacts` 为空；
  - `git diff --name-only -- cnn_fpga physics benchmark tests` 为空；
  - `docs/00_*` 到 `docs/08_*` 的 diff 依旧非空，但属于本轮开始前已存在的治理工作区变更，未见 `T75` 越权去修改这些文件。
