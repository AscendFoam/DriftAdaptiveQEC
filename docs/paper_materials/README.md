# Paper Materials

本目录保存主线任务产出的论文材料、claim/evidence ledger、result/figure ledger、paper-facing 风险台账、草稿骨架、reviewer risk audit 和理论说明。

## 文件清单

| 文件 | 来源/用途 |
| --- | --- |
| `mainline_theory_analysis.md` | `T53` 主线理论 walkthrough |
| `paper_claim_evidence_ledger.md` | `T34` 初版、`T73` 刷新后的主线 claim/evidence 台账；吸收 `T48/T50/T70/T72` 后的当前可支持口径 |
| `paper_result_figure_ledger.md` | `T73` 新增的主线 result/figure/material ledger；把图、表、result-pack 与对应 run/review/source 绑到一起 |
| `paper_claim_risk_table.md` | `T73` 新增的主线 paper-facing 风险台账；把 claim area 与 `R*` / review warning 对齐 |
| `paper_draft_skeleton.md` | `T35` paper skeleton |
| `paper_reviewer_risk_audit.md` | `T35` reviewer risk audit |
| `paper_method_positioning_calibration.md` | `T42` method-positioning calibration |
| `paper_background_related_work_draft.md` | `T43` bounded Background / Related Work prose draft |
| `paper_ablation_result_pack.md` | `T47/T57/T58/T70` paper ablation/result-pack ledger；其中 `FR8` 现在只允许写成 bounded extension-lane closure/no-promotion 素材 |
| `paper_simulation_result_table_pack.md` | `T74` 新增的 paper-ready simulation table pack；把 `T24/FR6/FR7/FR8/T50/T48/T72` 的表格候选、放置层级和安全口径绑到 stable ID 上 |
| `paper_figure_caption_pack.md` | `T74` 新增的 figure/table/supplement caption pack；每个 `T74-*` stable ID 都有推荐标题、caption 草案、状态和证据回指 |
| `paper_maintext_insertion_map.md` | `T74` 新增的主文/附录/补充材料插入图；说明每个 stable ID 应放在哪一层，以及为什么不能升格 |
| `paper_submission_material_gap_checklist.md` | `T74` 新增的 submission-side gap checklist；区分 ready、partial、blocked 和必须等待硬件条件的项 |
| `paper_maintext_results_authoring_pack.md` | `T75` 新增的主文 Results authoring pack；把 `T74` stable IDs 压缩成可直接落笔的 bounded Results 段落句胚 |
| `paper_caption_lock_and_placement_notes.md` | `T75` 新增的最终成图标题/caption/placement 锁定说明；服务于 `T75-FIG-*` authoring 资产 |
| `paper_appendix_bridge_pack.md` | `T75` 新增的主文到附录/补充材料桥接包；说明哪些表图必须留在 appendix / supplement |
| `paper_authoring_do_not_write_list.md` | `T75` 新增的作者禁写清单；集中冻结 `.tflite`、real-board、`FR8` 与 blocked 图位的 overclaim 边界 |
| `paper_rendered_figure_qa.md` | `T76` 新增的真实渲染图形 QA 记录；把 `T75-FIG-*` 的 preview 问题、修正和当前可读性状态绑定到 `T76-PREVIEW-*` |
| `paper_results_section_assembly_pack.md` | `T76` 新增的 Results section 装配包；锁定主文段落顺序、图表放置、fallback 路线与边界说明 |
| `paper_results_callout_sheet.md` | `T76` 新增的作者 callout 清单；给出 paragraph-level 可写/不可写表述，并绑定 `T75-FIG-*` / `T74-*` 来源 |

## 推荐阅读顺序

1. `paper_claim_evidence_ledger.md`
2. `paper_result_figure_ledger.md`
3. `paper_ablation_result_pack.md`
4. `paper_simulation_result_table_pack.md`
5. `paper_figure_caption_pack.md`
6. `paper_maintext_insertion_map.md`
7. `paper_submission_material_gap_checklist.md`
8. `paper_maintext_results_authoring_pack.md`
9. `paper_caption_lock_and_placement_notes.md`
10. `paper_appendix_bridge_pack.md`
11. `paper_authoring_do_not_write_list.md`
12. `paper_rendered_figure_qa.md`
13. `paper_results_section_assembly_pack.md`
14. `paper_results_callout_sheet.md`
15. `paper_claim_risk_table.md`

## 边界

本目录是论文材料入口，不是当前计划入口。任何 manuscript claim 必须回到 task/review/run/artifact 证据；不得把 draft prose、risk audit 或 theory walkthrough 写成实验完成事实。

当前必须随引用一起保留的 post-`T72` 边界：

- `T48` 只支持 isolated current-host true `.tflite` runtime；不支持 default-env / HIL / deployment closure。
- `T49/T71/T72` 只支持 read-only real-board gate / regeneration / provenance 边界；不支持 real-board execution success。
- `T64`-`T70` 只支持 `statcalib` extension lane 与 `no_promotion_keep_extension_lane_only`；不支持 mature comparator retelling。

`T74` 之后新增的一条使用规则是：

- 所有 paper-ready simulation material 优先通过 `T74-TBL-*` / `T74-FIG-*` / `T74-SUP-*` stable ID 引用，再回指对应 task/review/run/artifact；不要在 paper 草稿中直接发明新的临时命名。

`T75` 之后新增的一条 authoring 规则是：

- 主文 Results prose 与最终成图优先通过 `T75-FIG-*` 资产 ID 引用，并在 `authoring_manifest.json` 中显式映射回上游 `T74-*` stable IDs；`T75` 只能压缩和锁定表述，不能提升证据等级。

`T76` 之后新增的一条 rendered-QA / assembly 规则是：

- 真实渲染预览、contact sheet、PDF bundle 与 paragraph callout 优先通过 `T76-PREVIEW-*` 和 `paper_results_callout_sheet.md` 引用，再回链到 `T75-FIG-*` 与上游 `T74-*`；`T76` 只验证 paper-facing 可读性与装配顺序，不升级任何实验或部署证据。
