# T76：Rendered Figure QA 与 Results Section Assembly Pack

## 状态
- 由 Captain 在 `2026-06-12` 基于 `T75` 完成后的主线需要提出
- 当前阶段：`Phase 2: Controlled Development`
- 决策状态：`Go`
- 任务类型：高强度 docs-only 论文材料质量控制 / figure QA / manuscript-facing assembly 打包任务

## 为什么现在做这个任务

`T75` 已经把主线 paper-facing 结果材料压缩成了可以直接被作者引用的 prose、caption lock、appendix bridge 和三张 publication-facing `T75-FIG-*` 成图资产，但 `T75` 的通过标准仍主要停留在：

- 资产真实存在；
- `<svg` 结构可解析；
- manifest/source-map 回链一致；
- safe / forbidden wording 已经锁定。

这足以支撑 `T75 = PASS`，但还不足以支撑“可以放心进入真实稿件排版/投稿整合”。当前仍缺少的一层是：

1. 对三张 `T75-FIG-*` 做真实 rendered preview，而不是只做 XML/结构检查；
2. 对文字重叠、字号可读性、图例拥挤、灰度可读性、双栏/单栏适配、callout 对齐等做一次人工可读性 QA；
3. 若 QA 发现 honest presentation 问题，在不升级任何证据等级的前提下，对 `T75` 资产和对应 authoring 文档做极小范围修正；
4. 把当前锁定的结果材料进一步装配成一份 manuscript-facing 的 Results-section assembly pack，供后续真正写作或排版直接使用。

因此，`T76` 不是新实验，也不是 full-manuscript reopen，而是一张比“再写一份说明”更强、但仍然严格 docs-only 的质量控制与装配任务。

## 前置条件

只有在以下条件都满足时，`T76` 才可执行：

- `T75` 已完成并通过 Captain `PASS`
- `docs/paper_materials/paper_maintext_results_authoring_pack.md` 已存在
- `docs/paper_materials/paper_caption_lock_and_placement_notes.md` 已存在
- `docs/paper_materials/paper_appendix_bridge_pack.md` 已存在
- `docs/paper_materials/paper_authoring_do_not_write_list.md` 已存在
- `docs/figure_assets/T75_maintext_results_authoring_pack/` 下三张 `T75-FIG-*` SVG、`authoring_manifest.json`、`asset_source_map.csv` 已存在

若这些条件不满足，Worker 不得在 `T76` 中重建 `T75`，而必须如实报告 blocker。

## 目标

在不运行任何新实验、不修改任何源码/配置/历史结果、不碰治理文档的前提下，完成以下六件事：

1. 对三张 `T75-FIG-*` SVG 产出真实 rendered preview，并记录实际使用的渲染路径与命令/方法；
2. 形成一份逐图人工可读性 QA 文档，明确哪些项通过、哪些项需修、哪些项只能保留为当前局限；
3. 如果 rendered QA 暴露出 honest presentation 问题，允许在严格任务边界内对现有 `T75` SVG 或其对应 authoring 文档做极小范围修正；
4. 形成一份 manuscript-facing 的 Results section assembly pack，明确正文图表顺序、callout、段落插入位置、替代表述和保留边界；
5. 形成一份结果段 callout 对照表，把主文中的 callout、对应图表资产 ID、上游 stable ID 和允许表达边界统一锁定；
6. 如果 `T75` 资产被修正，必须把 `T75` 的 manifest/source-map/README 与对应 authoring 文档同步更新，保持 traceability 一致。

## Allowed Files

Worker 只可修改或新增以下路径：

- `docs/tasks/Phase2/T76_rendered_figure_qa_and_results_section_assembly_pack.md`
- `docs/paper_materials/README.md`
- `docs/paper_materials/paper_rendered_figure_qa.md`
- `docs/paper_materials/paper_results_section_assembly_pack.md`
- `docs/paper_materials/paper_results_callout_sheet.md`
- `docs/paper_materials/paper_maintext_results_authoring_pack.md`
- `docs/paper_materials/paper_caption_lock_and_placement_notes.md`
- `docs/paper_materials/paper_appendix_bridge_pack.md`
- `docs/paper_materials/paper_authoring_do_not_write_list.md`
- `docs/figure_assets/T76_rendered_figure_qa_pack/README.md`
- `docs/figure_assets/T76_rendered_figure_qa_pack/render_manifest.json`
- `docs/figure_assets/T76_rendered_figure_qa_pack/preview_source_map.csv`
- `docs/figure_assets/T76_rendered_figure_qa_pack/visual_qa_checklist.md`
- `docs/figure_assets/T76_rendered_figure_qa_pack/t75_fig_m01_preview.png`
- `docs/figure_assets/T76_rendered_figure_qa_pack/t75_fig_m02_preview.png`
- `docs/figure_assets/T76_rendered_figure_qa_pack/t75_fig_a01_preview.png`
- `docs/figure_assets/T76_rendered_figure_qa_pack/t75_preview_contact_sheet.png`
- `docs/figure_assets/T76_rendered_figure_qa_pack/t75_preview_bundle.pdf`
- `docs/figure_assets/T75_maintext_results_authoring_pack/README.md`
- `docs/figure_assets/T75_maintext_results_authoring_pack/authoring_manifest.json`
- `docs/figure_assets/T75_maintext_results_authoring_pack/asset_source_map.csv`
- `docs/figure_assets/T75_maintext_results_authoring_pack/t75_fig_m01_t24_frozen_summary.svg`
- `docs/figure_assets/T75_maintext_results_authoring_pack/t75_fig_m02_fr6_multi_seed_mechanism.svg`
- `docs/figure_assets/T75_maintext_results_authoring_pack/t75_fig_a01_boundary_schematic.svg`
- `docs/review/T76_review.md`
- `docs/for_human/T76_explanation.md`
- `docs/worker_summary/T76_worker_summary.md`

## Docs To Update

Worker 必须更新：

- `docs/paper_materials/README.md`
- `docs/paper_materials/paper_rendered_figure_qa.md`
- `docs/paper_materials/paper_results_section_assembly_pack.md`
- `docs/paper_materials/paper_results_callout_sheet.md`
- `docs/review/T76_review.md`
- `docs/for_human/T76_explanation.md`
- `docs/worker_summary/T76_worker_summary.md`

如果 `T76` 中对任何 `T75` authoring 文档或图资产做了修正，还必须同步更新：

- `docs/paper_materials/paper_maintext_results_authoring_pack.md`
- `docs/paper_materials/paper_caption_lock_and_placement_notes.md`
- `docs/paper_materials/paper_appendix_bridge_pack.md`
- `docs/paper_materials/paper_authoring_do_not_write_list.md`
- `docs/figure_assets/T75_maintext_results_authoring_pack/README.md`
- `docs/figure_assets/T75_maintext_results_authoring_pack/authoring_manifest.json`
- `docs/figure_assets/T75_maintext_results_authoring_pack/asset_source_map.csv`

## Forbidden Scope

Worker 不得：

- 修改 `docs/02_experiment_plan.md`
- 修改任何治理文档 `docs/00_*` 到 `docs/08_*`
- 修改任何 `cnn_fpga/`、`physics/`、`benchmark/`、`tests/` 源码或测试
- 修改任何 `runs/`、`artifacts/`、`docs/evidence_packs/` 历史事实文件
- 运行任何 benchmark、训练、`.tflite` smoke、real-board gate/regeneration/smoke
- 安装新的系统级依赖、下载大型外部工具，或因为渲染需求去改变主机环境
- 修改任何 theory branch 材料、`docs/paper_notes/` 中的 full-manuscript 草稿或 sidecar lane 文档
- 借“Results assembly”之名恢复 full-manuscript 扩写、摘要/引言/related work 全文撰写或投稿包总装
- 静默提升任何证据等级，尤其不得把 `T24` 写成 paper-grade expanded benchmark、把 `T48` 写成 deployment closure、把 `T72` 写成 real-board execution success、把 `T70` 写成 mature statcalib comparator
- 如果当前主机没有可用的真实渲染路径，伪造“已完成 rendered QA”；必须明确报告 blocker

## 必须复用的输入

Worker 必须复用以下输入，而不是重写历史事实：

- 治理入口：
  - `README.md`
  - `docs/00_project_snapshot.md`
  - `docs/02_experiment_plan.md`
  - `docs/03_hil_p4_boundary_audit.md`
  - `docs/04_task_board.md`
  - `docs/07_handoff.md`
  - `docs/08_risks_and_open_questions.md`
- `T75` 的主 authoring 入口：
  - `docs/paper_materials/paper_maintext_results_authoring_pack.md`
  - `docs/paper_materials/paper_caption_lock_and_placement_notes.md`
  - `docs/paper_materials/paper_appendix_bridge_pack.md`
  - `docs/paper_materials/paper_authoring_do_not_write_list.md`
  - `docs/figure_assets/T75_maintext_results_authoring_pack/authoring_manifest.json`
  - `docs/figure_assets/T75_maintext_results_authoring_pack/asset_source_map.csv`
  - `docs/figure_assets/T75_maintext_results_authoring_pack/t75_fig_m01_t24_frozen_summary.svg`
  - `docs/figure_assets/T75_maintext_results_authoring_pack/t75_fig_m02_fr6_multi_seed_mechanism.svg`
  - `docs/figure_assets/T75_maintext_results_authoring_pack/t75_fig_a01_boundary_schematic.svg`
- 上游 `T74` stable-ID 与主线证据：
  - `docs/paper_materials/paper_simulation_result_table_pack.md`
  - `docs/paper_materials/paper_figure_caption_pack.md`
  - `docs/paper_materials/paper_maintext_insertion_map.md`
  - `docs/paper_materials/paper_submission_material_gap_checklist.md`
  - `docs/figure_assets/T74_paper_ready_simulation_result_pack/figure_manifest.json`
  - `docs/figure_assets/T74_paper_ready_simulation_result_pack/result_source_map.csv`
  - `docs/figure_assets/T74_paper_ready_simulation_result_pack/caption_source_map.csv`
  - `docs/figure_assets/T74_paper_ready_simulation_result_pack/table_snapshot.csv`
  - `docs/figure_assets/T74_paper_ready_simulation_result_pack/submission_bundle_manifest.json`
  - `docs/review/T74_review.md`
  - `docs/review/T75_review.md`

## 固定边界

- 这是主线 paper-facing QA/assembly 任务，不是实验任务
- 只允许“真实 rendered preview”“人工可读性 QA”“必要时极小范围修图修文”“主文 Results 装配”
- 不允许“新建 run”“重算结果”“替换主线结论”“补新的 paper claim”
- 任何对 `T75` 资产的修改都必须是 presentation-level 的 honest fix，而不是 evidence-level 的 upgrade
- `T74-FIG-04` 的 blocked 状态必须保留；`T76` 不得把它修成“统一 portability / deployment closure 图”
- 如果某图在真实预览后仍无法诚实地达到主文质量，必须在 assembly/callout 文档中明确退回 `T74` 表格替代方案，而不是静默保留有缺陷图

## ID 与一致性约束

- `T76` 必须显式回链到 `T75-FIG-*`，并通过 `T75` 再回链到 `T74-*` stable IDs
- 建议使用以下 preview asset ID：
  - `T76-PREVIEW-M01`
  - `T76-PREVIEW-M02`
  - `T76-PREVIEW-A01`
  - `T76-PREVIEW-CS01`
- 每个 `T76` preview asset ID 必须在 `render_manifest.json` 中明确映射到源 `T75-FIG-*` 资产
- 同一预览资产的 ID、源文件、输出文件、render method 和 QA 结论必须在以下文件中保持一致：
  - `paper_rendered_figure_qa.md`
  - `paper_results_section_assembly_pack.md`
  - `paper_results_callout_sheet.md`
  - `render_manifest.json`
  - `preview_source_map.csv`
  - `visual_qa_checklist.md`
- 如果 `T75` 资产被修改，更新后的 `T75-FIG-*` 文件名和 ID 不得随意改名；必须保持与 `T75` authoring 文档一致

## 任务要求

### A. 产出真实 rendered figure QA 文档

`docs/paper_materials/paper_rendered_figure_qa.md` 至少要包含：

1. 实际使用的渲染路径说明
   - 例如浏览器导出、已有 Python SVG 渲染路径、现有本机工具
   - 必须写明未新增系统级依赖
2. 三张 `T75-FIG-*` 的逐图 QA 记录
   - 文本是否重叠
   - 字号是否可读
   - 图例是否拥挤
   - 灰度打印是否可读
   - 双栏/单栏适配建议
   - 是否需要退回表格替代
3. 若有问题，必须区分：
   - 已在 `T76` 内修正
   - 当前可接受保留
   - 无法在当前边界内修正
4. 每条 QA 结论都必须显式绑定对应 `T75-FIG-*` 和上游 `T74-*` stable IDs

### B. 产出 Results section assembly pack

`docs/paper_materials/paper_results_section_assembly_pack.md` 至少要包含：

1. 主文 Results 的推荐图表顺序
2. 每个图/表的推荐插入位置
3. 推荐的一句话 callout
4. 主结果段、机制段、边界段的图文配对建议
5. 若 `T75-FIG-M01` 不适合最终版面，必须明确何时退回 `T74-TBL-01`
6. 哪些 appendix/supplement 内容继续留在附录，不得抬升为主结果

### C. 产出 callout 对照表

`docs/paper_materials/paper_results_callout_sheet.md` 至少要包含：

1. 每个正文 callout 的短标签
2. 对应 `T75-FIG-*` 或保留的 `T74-*` stable ID
3. 允许表达
4. 禁止表达
5. 对应段落角色
   - 主结果
   - 机制描述
   - 边界/限制

### D. 生成 preview 资产目录

`docs/figure_assets/T76_rendered_figure_qa_pack/` 下必须生成：

1. `README.md`
   - 说明该目录是什么、不是什麽、如何回链到 `T75` 和 `T74`
2. `render_manifest.json`
   - 至少记录 `t76_preview_id, source_t75_asset_id, upstream_t74_ids, render_method, output_file, qa_status, notes`
3. `preview_source_map.csv`
   - `t76_preview_id,source_t75_asset_id,upstream_t74_id,source_path,output_path,role`
4. `visual_qa_checklist.md`
   - 逐图 checklist，至少覆盖 legibility / overlap / legend / grayscale / placement readiness
5. `t75_fig_m01_preview.png`
6. `t75_fig_m02_preview.png`
7. `t75_fig_a01_preview.png`
8. `t75_preview_contact_sheet.png`
9. `t75_preview_bundle.pdf`
   - 若本机无合适零安装路径可省略，但必须在 `README.md` 与 `paper_rendered_figure_qa.md` 中明确说明原因

### E. 如有必要，对 T75 资产做极小范围修正

只有在真实 rendered QA 已证明存在 honest presentation 问题时，才允许修改：

- `paper_maintext_results_authoring_pack.md`
- `paper_caption_lock_and_placement_notes.md`
- `paper_appendix_bridge_pack.md`
- `paper_authoring_do_not_write_list.md`
- `authoring_manifest.json`
- `asset_source_map.csv`
- 三张 `T75` SVG

修正约束：

- 只能修 presentation-level 问题
- 不得新增新的结果主张
- 不得更改 stable-ID 语义
- 不得把 blocked/partial/no-promotion 证据修成 completed claim
- 必须在 `paper_rendered_figure_qa.md` 中写明修正前后差异

## 预期输出

Worker 必须产出：

- `docs/paper_materials/paper_rendered_figure_qa.md`
- `docs/paper_materials/paper_results_section_assembly_pack.md`
- `docs/paper_materials/paper_results_callout_sheet.md`
- `docs/figure_assets/T76_rendered_figure_qa_pack/README.md`
- `docs/figure_assets/T76_rendered_figure_qa_pack/render_manifest.json`
- `docs/figure_assets/T76_rendered_figure_qa_pack/preview_source_map.csv`
- `docs/figure_assets/T76_rendered_figure_qa_pack/visual_qa_checklist.md`
- `docs/figure_assets/T76_rendered_figure_qa_pack/t75_fig_m01_preview.png`
- `docs/figure_assets/T76_rendered_figure_qa_pack/t75_fig_m02_preview.png`
- `docs/figure_assets/T76_rendered_figure_qa_pack/t75_fig_a01_preview.png`
- `docs/figure_assets/T76_rendered_figure_qa_pack/t75_preview_contact_sheet.png`
- 如可行则补 `docs/figure_assets/T76_rendered_figure_qa_pack/t75_preview_bundle.pdf`
- 更新后的 `docs/paper_materials/README.md`
- 如有必要，更新后的 `T75` authoring 文档/资产
- `docs/review/T76_review.md`
- `docs/for_human/T76_explanation.md`
- `docs/worker_summary/T76_worker_summary.md`

## 验证

Worker 必须实际执行并报告：

1. `render_manifest.json` 是否可解析，且至少包含 `T76-PREVIEW-M01/M02/A01/CS01`
2. `preview_source_map.csv` 中的 preview ID 是否与 manifest 一致
3. 三张 preview PNG 和 contact sheet 是否真实存在且非空
4. `paper_rendered_figure_qa.md` 是否逐图覆盖全部 `T75-FIG-*`
5. `paper_results_section_assembly_pack.md` 是否给出主文图表顺序、插入位置、退回替代条件与边界说明
6. `paper_results_callout_sheet.md` 是否逐项写清 callout、资产 ID、允许表达与禁止表达
7. 如果 `T75` 资产被修改，`authoring_manifest.json` 和 `asset_source_map.csv` 是否仍能正确回链
8. `git diff --name-only -- runs`
9. `git diff --name-only -- artifacts`
10. `git diff --name-only -- cnn_fpga physics benchmark tests`
11. `git diff --name-only -- docs/00_project_snapshot.md docs/01_legacy_audit.md docs/03_hil_p4_boundary_audit.md docs/04_task_board.md docs/05_decision_log.md docs/06_repo_noise_governance.md docs/07_handoff.md docs/08_risks_and_open_questions.md`

如果没有真实渲染路径可用，还必须额外报告：

- 已尝试的本机可用方法
- 为什么当前方法不可用
- 哪些输出因此无法诚实完成
- 为什么这应作为 blocker，而不是“结构检查替代 rendered QA”

## 完成标准

只有同时满足以下条件，`T76` 才可视为完成：

1. 已形成一套真实 rendered preview + 人工 QA + Results-section assembly 的统一入口
2. 三张 `T75` 成图资产都已获得真实预览或被如实标记为当前不可预览的 blocker
3. 若存在 presentation-level 缺陷，已在当前边界内完成最小修正，或明确保留/退回方案
4. 所有 `T76` 预览资产都能 trace back 到 `T75`，并通过 `T75` trace back 到 `T74` stable IDs
5. 没有把任何 blocked / partial / extension-lane / no-promotion / gate-only 证据静默升级
6. 没有改动治理文档、源码、测试、`runs/`、`artifacts/`
