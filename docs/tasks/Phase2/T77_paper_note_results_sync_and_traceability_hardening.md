# T77：论文 note-draft 结果层同步与 T76 traceability hardening

## 状态

- 由 Captain 在 `2026-06-12` 基于 `T76` 的 `PASS_WITH_WARNINGS` closeout 提出
- 当前阶段：`Phase 2: Controlled Development`
- 决策状态：`Go`
- 任务类型：高强度 docs-only 论文结果层同步 / traceability hardening / note 编译检查任务

## 为什么现在做这个任务

`T76` 已经把 `T75-FIG-*` 的真实 rendered preview、人工可读性 QA、callout sheet 和 Results-section assembly 收口出来，因此“图是否可读、结果段如何装配”这个问题已经有了 paper-facing 的直接答案。

但 `T76_review.md` 也明确留下了两条需要认真处理的 paper-facing warning：

1. `docs/figure_assets/T76_rendered_figure_qa_pack/preview_source_map.csv` 的聚合预览行复用了 `upstream_t74_ids` 这一列语义，导致 preview 自身 ID 与真正的上游 stable ID 混在一起；
2. `docs/paper_materials/paper_rendered_figure_qa.md` 的逐图 QA 结论虽然总体 traceability 成立，但没有把每张图的上游 `T74-*` stable ID 全部内联写全。

与此同时，仓库里已经有 `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`，并且它已经包含：

- `Experimental Setup`
- `Numerical Results and Benchmark Plan`
- `Discussion`
- `Conclusion`

这些章节已经到了应该与主线当前 evidence stack 对齐的时候。当前更合理的下一步不是直接恢复 full-manuscript 扩写，而是先做一张更强但仍受控的任务：

1. 把 `T76` preview pack 的 traceability 粒度修干净；
2. 把已经被 `T74/T75/T76` 收口的结果层材料同步进现有 note-draft；
3. 在本地工具可用时，对更新后的 note 做一次受控编译检查；
4. 保持这仍然只是“结果层同步”，而不是 full-manuscript reopen。

## 前置条件

只有在以下条件都满足时，`T77` 才可执行：

- `T76` 已完成并通过 Captain `PASS_WITH_WARNINGS`
- 以下文件已存在：
  - `docs/paper_materials/paper_rendered_figure_qa.md`
  - `docs/paper_materials/paper_results_section_assembly_pack.md`
  - `docs/paper_materials/paper_results_callout_sheet.md`
  - `docs/figure_assets/T76_rendered_figure_qa_pack/render_manifest.json`
  - `docs/figure_assets/T76_rendered_figure_qa_pack/preview_source_map.csv`
  - `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`

如果这些前提不满足，Worker 不得在 `T77` 中重建 `T76` 或另起草稿，而必须如实报告 blocker。

## 目标

在不运行任何新实验、不修改任何源码/配置/历史结果、不碰治理文档的前提下，完成以下五件事：

1. 修补 `T76` preview pack 的 traceability/schema 粒度：
   - 聚合预览行不得再用 `upstream_t74_ids` 混写 preview 自身 ID；
   - 逐图 QA 结论必须显式写出每张图对应的上游 `T74-*` stable IDs；
   - `README.md`、`render_manifest.json`、`preview_source_map.csv`、`paper_rendered_figure_qa.md` 四者保持一致。
2. 把已经收口的结果层材料同步到 `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`：
   - 仅限结果相关章节；
   - 保持 `T24/T48/T70/T72` 的边界诚实；
   - 不借同步之名恢复全文扩写。
3. 形成一份 note 层同步 manifest：
   - 记录每个被修改的 section / table / paragraph 对应哪些 `T74-*` / `T75-FIG-*` / `T76-PREVIEW-*`；
   - 记录允许表达与禁止表达；
   - 记录哪些内容仍保留为 future work / appendix / blocked。
4. 在本地工具可用时，对 note-draft 做一次受控编译检查：
   - 允许更新 PDF 与 LaTeX 辅助文件；
   - 如果本机没有可用工具链，必须如实记录 `NO_LOCAL_LATEX_TOOLCHAIN`，不得伪造成功编译。
5. 清理 `T76` 留下的本地探针/缓存残留：
   - `.tmp_t76_render_a01.png`
   - `.tmp_t76_render_m02.png`
   - `.tmp_t76_render_probe.png`
   - `.tmp_t76_fontcache/`

## Allowed Files

Worker 只可修改或新增以下路径：

- `docs/tasks/Phase2/T77_paper_note_results_sync_and_traceability_hardening.md`
- `docs/paper_notes/README.md`
- `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`
- `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.pdf`
- `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.aux`
- `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.fdb_latexmk`
- `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.fls`
- `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.log`
- `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.out`
- `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.synctex.gz`
- `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.toc`
- `docs/paper_materials/README.md`
- `docs/paper_materials/paper_rendered_figure_qa.md`
- `docs/paper_materials/paper_results_section_assembly_pack.md`
- `docs/paper_materials/paper_results_callout_sheet.md`
- `docs/paper_materials/paper_note_results_sync_manifest.md`
- `docs/figure_assets/T76_rendered_figure_qa_pack/README.md`
- `docs/figure_assets/T76_rendered_figure_qa_pack/render_manifest.json`
- `docs/figure_assets/T76_rendered_figure_qa_pack/preview_source_map.csv`
- `docs/review/T77_review.md`
- `docs/for_human/T77_explanation.md`
- `docs/worker_summary/T77_worker_summary.md`
- `.tmp_t76_render_a01.png`
- `.tmp_t76_render_m02.png`
- `.tmp_t76_render_probe.png`
- `.tmp_t76_fontcache/`

## Docs To Update

Worker 必须更新：

- `docs/paper_notes/README.md`
- `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`
- `docs/paper_materials/README.md`
- `docs/paper_materials/paper_rendered_figure_qa.md`
- `docs/paper_materials/paper_results_section_assembly_pack.md`
- `docs/paper_materials/paper_results_callout_sheet.md`
- `docs/paper_materials/paper_note_results_sync_manifest.md`
- `docs/review/T77_review.md`
- `docs/for_human/T77_explanation.md`
- `docs/worker_summary/T77_worker_summary.md`

如果本地有可用 LaTeX 工具链并完成了编译检查，还应同步更新：

- `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.pdf`
- `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.aux`
- `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.fdb_latexmk`
- `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.fls`
- `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.log`
- `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.out`
- `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.synctex.gz`
- `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.toc`

## Forbidden Scope

Worker 不得：

- 修改任何治理文档 `docs/00_*` 到 `docs/08_*`
- 修改任何 `cnn_fpga/`、`physics/`、`benchmark/`、`tests/` 源码或测试
- 修改任何 `runs/`、`artifacts/`、`docs/evidence_packs/` 历史事实文件
- 运行任何 benchmark、训练、`.tflite` smoke、real-board gate/regeneration/smoke
- 安装新的系统级依赖或为了编译/渲染去改变主机环境
- 新建第二个 manuscript 草稿、第二个 note 主文件，或把任务扩成 full-manuscript reopen
- 大范围改写 `Introduction`、`Relationship to Existing Work`、`Model Architecture` 等非结果层章节
- 修改 theory 分支其他材料、sidecar lane 文档或 `docs/sidecar/*`
- 静默提升任何证据等级，尤其不得把：
  - `T24` 写成 paper-grade expanded benchmark
  - `T48` 写成 default-env / deployment closure
  - `T49/T71/T72` 写成 real-board execution success
  - `T64`-`T70` 写成 mature `statcalib` comparator promotion
- 如果当前主机没有可用的本地 LaTeX 工具链，伪造“已成功编译 note”；必须明确报告 blocker

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
- 结果层纸面材料：
  - `docs/paper_materials/paper_claim_evidence_ledger.md`
  - `docs/paper_materials/paper_claim_risk_table.md`
  - `docs/paper_materials/paper_ablation_result_pack.md`
  - `docs/paper_materials/paper_simulation_result_table_pack.md`
  - `docs/paper_materials/paper_figure_caption_pack.md`
  - `docs/paper_materials/paper_maintext_insertion_map.md`
  - `docs/paper_materials/paper_maintext_results_authoring_pack.md`
  - `docs/paper_materials/paper_caption_lock_and_placement_notes.md`
  - `docs/paper_materials/paper_appendix_bridge_pack.md`
  - `docs/paper_materials/paper_authoring_do_not_write_list.md`
  - `docs/paper_materials/paper_rendered_figure_qa.md`
  - `docs/paper_materials/paper_results_section_assembly_pack.md`
  - `docs/paper_materials/paper_results_callout_sheet.md`
- 结果层图资产与 preview 包：
  - `docs/figure_assets/T74_paper_ready_simulation_result_pack/figure_manifest.json`
  - `docs/figure_assets/T74_paper_ready_simulation_result_pack/result_source_map.csv`
  - `docs/figure_assets/T74_paper_ready_simulation_result_pack/caption_source_map.csv`
  - `docs/figure_assets/T75_maintext_results_authoring_pack/authoring_manifest.json`
  - `docs/figure_assets/T75_maintext_results_authoring_pack/asset_source_map.csv`
  - `docs/figure_assets/T76_rendered_figure_qa_pack/render_manifest.json`
  - `docs/figure_assets/T76_rendered_figure_qa_pack/preview_source_map.csv`
- review 边界：
  - `docs/review/T74_review.md`
  - `docs/review/T75_review.md`
  - `docs/review/T76_review.md`
- 当前 note 草稿：
  - `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`
  - `docs/paper_notes/README.md`

## 固定边界

- 这是主线 paper-facing note-sync / traceability 任务，不是实验任务
- 只允许“结果层同步”“source-ID 补强”“受控编译检查”“临时探针清理”
- 不允许“新增结果”“改写主线结论”“补新的 benchmark / runtime / board 事实”
- `statcalib` 在 note 中仍必须保留 extension-lane / no-promotion 边界
- `T48` 在 note 中仍必须保留 isolated current-host true runtime only 边界
- `T49/T71/T72` 在 note 中仍必须保留 read-only gate / provenance / `NO_GO` 边界
- `T74-FIG-04` 的 blocked 状态必须保留；不得在 note 中补写成统一 deployment closure 图
- 如果 note 当前某段无法在现有证据边界内诚实保留，必须降级为 future-work / appendix-boundary / blocked wording，而不是静默保留过强说法

## note-draft 允许修改的章节范围

本任务只允许修改 `CNN_FPGA_GKP_theory_note_draft.tex` 中与结果层直接相关的部分：

1. `Abstract`
2. `Summary of Contributions`
3. `Experimental Setup`
4. `Numerical Results and Benchmark Plan`
5. `Discussion`
6. `Conclusion`

不允许把任务扩展到：

- `Introduction`
- `Brief Review of the GKP Code`
- `Noise and Drift Model`
- `Model Architecture`
- `Relationship to Existing Work`

除非某句与当前受控结果层发生直接冲突，且最小修正是删除或弱化该冲突句。

## 任务要求

### A. 修补 T76 traceability 粒度

至少完成以下事项：

1. `preview_source_map.csv` 必须把 preview 自身 ID 与真正的上游 `T74-*` stable IDs 分开表达；
2. 聚合预览产物（contact sheet / PDF bundle）不得再把 preview ID 填进 `upstream_t74_ids` 语义位；
3. `paper_rendered_figure_qa.md` 每个逐图小节都必须显式列出：
   - 对应 `T75-FIG-*`
   - 对应 `T76-PREVIEW-*`
   - 对应全部上游 `T74-*` stable IDs
4. `README.md`、`render_manifest.json`、`preview_source_map.csv`、`paper_rendered_figure_qa.md` 四者对同一 preview 的 source chain 必须一致。

### B. 同步 note-draft 的结果层

至少完成以下事项：

1. 将 `Experimental Setup` 与 `Numerical Results and Benchmark Plan` 对齐到当前 reviewed 口径；
2. 把 `T76` 的主图顺序 / fallback 路线 / callout 边界同步进 note；
3. 对 `statcalib` 相关段落补齐 extension-lane / no-promotion / persistent-tie 的边界诚实性；
4. 对 runtime / board-level validation 段落保持 “future work / boundary layer”，不得写成已验证；
5. 如有必要，可在相关段落前增加 LaTeX 注释，格式建议为：

```tex
% T77-SOURCE: T74-TBL-01; T75-FIG-M01; T76-PREVIEW-M01; review=T76
```

### C. 形成 note 层同步 manifest

`docs/paper_materials/paper_note_results_sync_manifest.md` 至少要包含：

1. 被修改的 note section / subsection 列表；
2. 每个 section 对应的 source stable IDs / task IDs；
3. 每个 section 的允许表达；
4. 每个 section 的禁止表达；
5. 若本轮没有同步的段落，为什么没同步：
   - 证据不足
   - 属于 appendix / supplement
   - 属于 blocked / future work

### D. 受控编译检查

若本地已有可用工具链：

1. 对更新后的 note 执行一次受控编译；
2. 更新 PDF 与辅助文件；
3. 在 `worker_summary` 与 `T77_review.md` 中记录使用的工具链与结果。

若本地没有可用工具链：

1. 不得安装新工具；
2. 在 `paper_note_results_sync_manifest.md`、`worker_summary` 与 `T77_review.md` 中明确记录：
   - 已尝试的本地方法
   - 为什么不可用
   - 哪些编译产物因此未更新

### E. 清理 T76 本地探针/缓存残留

如果以下路径仍存在：

- `.tmp_t76_render_a01.png`
- `.tmp_t76_render_m02.png`
- `.tmp_t76_render_probe.png`
- `.tmp_t76_fontcache/`

则应在本任务中删除，并在 `worker_summary` 中说明这些是过程性残留而非正式 deliverable。

## 预期输出

Worker 必须产出：

- 更新后的 `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`
- 更新后的 `docs/paper_notes/README.md`
- 更新后的 `docs/paper_materials/paper_rendered_figure_qa.md`
- 更新后的 `docs/paper_materials/paper_results_section_assembly_pack.md`
- 更新后的 `docs/paper_materials/paper_results_callout_sheet.md`
- 新增 `docs/paper_materials/paper_note_results_sync_manifest.md`
- 更新后的 `docs/figure_assets/T76_rendered_figure_qa_pack/README.md`
- 更新后的 `docs/figure_assets/T76_rendered_figure_qa_pack/render_manifest.json`
- 更新后的 `docs/figure_assets/T76_rendered_figure_qa_pack/preview_source_map.csv`
- `docs/review/T77_review.md`
- `docs/for_human/T77_explanation.md`
- `docs/worker_summary/T77_worker_summary.md`

若本地编译成功，还应产出：

- 更新后的 `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.pdf`
- 对应 LaTeX 辅助文件

## 验证

Worker 必须实际执行并报告：

1. `preview_source_map.csv` 是否可解析，且 preview 自身 ID 与上游 `T74-*` stable ID 字段已分离
2. `paper_rendered_figure_qa.md` 是否逐图显式列出：
   - `T75-FIG-*`
   - `T76-PREVIEW-*`
   - 上游 `T74-*`
3. `paper_note_results_sync_manifest.md` 是否覆盖本轮所有被修改的 note section
4. `CNN_FPGA_GKP_theory_note_draft.tex` 是否只修改了允许的结果层章节
5. note 中是否仍然保持以下边界：
   - `T24` = frozen-set mock-backed software-HIL anchor
   - `T48` = isolated current-host true runtime only
   - `T49/T71/T72` = read-only real-board gate / provenance / `NO_GO`
   - `T64`-`T70` = `statcalib` extension lane / no-promotion
6. 如有本地 LaTeX 工具链，编译是否成功；若失败，错误是什么
7. `.tmp_t76_*` / `.tmp_t76_fontcache/` 是否已清理
8. `git diff --name-only -- runs`
9. `git diff --name-only -- artifacts`
10. `git diff --name-only -- cnn_fpga physics benchmark tests`
11. `git diff --name-only -- docs/00_project_snapshot.md docs/01_legacy_audit.md docs/02_experiment_plan.md docs/03_hil_p4_boundary_audit.md docs/04_task_board.md docs/05_decision_log.md docs/06_repo_noise_governance.md docs/07_handoff.md docs/08_risks_and_open_questions.md`

## 完成标准

只有同时满足以下条件，`T77` 才可视为完成：

1. `T76` preview-source / stable-ID traceability 粒度已经补干净；
2. note-draft 的结果层章节已与当前 reviewed evidence stack 对齐；
3. 没有把任何 blocked / partial / extension-lane / no-promotion / gate-only 证据静默升级；
4. 如有可用本地工具链，已完成受控编译检查；如无，则已诚实记录 blocker；
5. `.tmp_t76_*` / `.tmp_t76_fontcache/` 过程性残留已处理；
6. 没有改动治理文档、源码、测试、`runs/`、`artifacts/`。
