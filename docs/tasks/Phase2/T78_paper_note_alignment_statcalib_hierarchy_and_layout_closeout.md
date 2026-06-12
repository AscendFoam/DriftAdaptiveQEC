# T78：论文 note-draft 非结果层校准、statcalib 层级降权与排版 warning 收口

## 状态

- 由 Captain 在 `2026-06-12` 基于 `T77` 的 `PASS_WITH_WARNINGS` closeout 提出
- 当前阶段：`Phase 2: Controlled Development`
- 决策状态：`Go`
- 任务类型：高强度 docs-only note 校准 / hierarchy / layout closeout 任务

## 为什么现在做这个任务

`T77` 已经完成了两件关键事情：

1. 把 `T74/T75/T76` 锁定的结果层材料同步进 `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`
2. 把 `T76` 的 preview-source / stable-ID traceability 粒度补干净

这意味着“结果层有没有同步进去、preview/source chain 是否够干净”这个问题已经有了明确答案。

但 `T77_review.md` 同时留下了四个不能忽略的 note 质量问题：

1. 当前整份 `.tex` 仍含未校准的非结果层历史段落，尤其是 `Relationship to Existing Work` 一类段落不能被误当成已经按当前 evidence stack 全量校准；
2. `statcalib` 虽然 wording 已压回 `extension-lane / no-promotion / no unique clean threshold`，但在 `Numerical Results` 里的视觉层级仍偏高；
3. note 已编译成功，但 `.log` 仍有多处 `Underfull \hbox`；
4. 当前“只同步了允许章节”的证明仍主要依赖 `paper_note_results_sync_manifest.md` 与 `% T77-SOURCE` 注释，缺少更机械的 section-scope 审计。

因此，当前更合理的下一步不是直接恢复 full-manuscript 扩写，而是先做一张更强但仍受控的 note 校准任务：

1. 补齐 note 的非结果层 evidence-facing wording 校准；
2. 把 `statcalib` 从视觉上进一步降回“补充/扩展 lane”，而不是并列主结果；
3. 收一轮 LaTeX 版面 warning；
4. 增加更机械的 section-scope 审计；
5. 保持这仍然只是 note 质量收口，而不是新的实验、不是新的 claim 升级、也不是 full-manuscript reopen。

## 前置条件

只有在以下条件都满足时，`T78` 才可执行：

- `T77` 已完成并通过 Captain `PASS_WITH_WARNINGS`
- 以下文件已存在：
  - `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`
  - `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.log`
  - `docs/paper_materials/paper_note_results_sync_manifest.md`
  - `docs/review/T77_review.md`
  - `docs/paper_notes/README.md`
  - `docs/paper_materials/README.md`

如果这些前提不满足，Worker 不得在 `T78` 中另起草稿或重建 `T77`，而必须如实报告 blocker。

## 目标

在不运行任何新实验、不修改任何源码/配置/历史结果、不碰治理文档的前提下，完成以下五件事：

1. 对现有 note 做一轮**非结果层 evidence-facing 校准**：
   - 只处理当前会影响 paper-facing 真实性的 wording；
   - 不把任务扩成理论分支大范围重写；
   - 不把局部校准误写成 full-manuscript 完成。
2. 对 `statcalib` 做一轮**视觉层级降权**：
   - 保持 `extension-lane / no-promotion / persistent-tie`；
   - 避免在版面结构上看起来像并列主结果支柱。
3. 对 note 做一轮**LaTeX 排版 warning 收口**：
   - 目标是减少或解释 `Underfull \hbox`；
   - 不是追求排版完美，而是收掉明显可修的问题。
4. 增加一份**更机械的 section-scope 审计**：
   - 清楚说明本轮到底改了哪些 section；
   - 明确哪些段落仍未校准；
   - 使后续 reviewer 不必只依赖口头说明。
5. 产出一份**author-facing closeout 记录**：
   - 说明哪些 note 层问题已收口；
   - 说明哪些问题仍保留到后续 paper reopen gate。

## Allowed Files

Worker 只可修改或新增以下路径：

- `docs/tasks/Phase2/T78_paper_note_alignment_statcalib_hierarchy_and_layout_closeout.md`
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
- `docs/paper_materials/paper_note_results_sync_manifest.md`
- `docs/paper_materials/paper_results_section_assembly_pack.md`
- `docs/paper_materials/paper_results_callout_sheet.md`
- `docs/paper_materials/paper_note_alignment_and_layout_closeout.md`
- `docs/review/T78_review.md`
- `docs/for_human/T78_explanation.md`
- `docs/worker_summary/T78_worker_summary.md`

## Docs To Update

Worker 必须更新：

- `docs/paper_notes/README.md`
- `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`
- `docs/paper_materials/README.md`
- `docs/paper_materials/paper_note_alignment_and_layout_closeout.md`
- `docs/review/T78_review.md`
- `docs/for_human/T78_explanation.md`
- `docs/worker_summary/T78_worker_summary.md`

如为完成层级降权与 scope 审计有必要，也可同步更新：

- `docs/paper_materials/paper_note_results_sync_manifest.md`
- `docs/paper_materials/paper_results_section_assembly_pack.md`
- `docs/paper_materials/paper_results_callout_sheet.md`

如果本地有可用 LaTeX 工具链并完成编译检查，还应同步更新：

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
- 安装新的系统级依赖或为了编译去改变主机环境
- 新建第二个 manuscript 草稿、第二个 note 主文件，或把任务扩成 full-manuscript reopen
- 大范围重写 `Noise and Drift Model`、`Model Architecture` 等理论章节；只能做最小 evidence-facing 校准
- 修改 theory 分支其他材料、sidecar lane 文档或 `docs/sidecar/*`
- 静默提升任何证据等级，尤其不得把：
  - `T24` 写成 paper-grade expanded benchmark
  - `T48` 写成 default-env / deployment closure
  - `T49/T71/T72` 写成 real-board execution success
  - `T64`-`T70` 写成 mature `statcalib` comparator promotion
- 以“校准 note”为名重新设计全文结构、增加新实验结果、引入新 figure/table 资产或创建新结果结论

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
- note / paper-facing 材料：
  - `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`
  - `docs/paper_notes/README.md`
  - `docs/paper_materials/paper_note_results_sync_manifest.md`
  - `docs/paper_materials/paper_results_section_assembly_pack.md`
  - `docs/paper_materials/paper_results_callout_sheet.md`
  - `docs/paper_materials/paper_authoring_do_not_write_list.md`
  - `docs/paper_materials/paper_claim_evidence_ledger.md`
  - `docs/paper_materials/paper_claim_risk_table.md`
  - `docs/paper_materials/paper_ablation_result_pack.md`
- review 边界：
  - `docs/review/T70_review.md`
  - `docs/review/T72_review.md`
  - `docs/review/T76_review.md`
  - `docs/review/T77_review.md`

## 固定边界

- 这是主线 note 校准 / hierarchy / layout 任务，不是实验任务
- 只允许“非结果层校准”“层级降权”“排版收口”“section-scope 审计”
- 不允许“新增结果”“改写主线结论”“补新的 benchmark / runtime / board 事实”
- `statcalib` 在 note 中仍必须保留 extension-lane / no-promotion / persistent-tie 边界
- `T48` 在 note 中仍必须保留 isolated current-host true runtime only 边界
- `T49/T71/T72` 在 note 中仍必须保留 read-only gate / provenance / `NO_GO` 边界
- 如果某句在现有证据边界内无法诚实保留，必须降级为 future-work / appendix-boundary / blocked wording，而不是静默保留过强说法

## note 允许修改的章节范围

本任务只允许修改 `CNN_FPGA_GKP_theory_note_draft.tex` 中以下与 paper-facing 校准直接相关的部分：

1. `Title`
2. `Abstract`
3. `Summary of Contributions`
4. `Introduction` 中直接涉及 evidence framing / contribution positioning 的段落
5. `Relationship to Existing Work`
6. `Numerical Results and Benchmark Plan` 中仅与 `statcalib` hierarchy / bridge wording / layout 直接相关的部分
7. `Discussion`
8. `Conclusion`

不允许把任务扩展到：

- `Brief Review of the GKP Code`
- `Noise and Drift Model`
- `Model Architecture`
- 任何新的 appendix / supplement 主体扩写

除非某句与当前受控 evidence stack 发生直接冲突，且最小修正是删除、弱化或重排该句。

## 任务要求

### A. 非结果层校准

至少完成以下事项：

1. 审核 `Title`、`Abstract`、`Summary of Contributions`、`Introduction` evidence-facing 段落与 `Relationship to Existing Work`；
2. 找出其中所有会让读者误以为“整份 note 已 fully calibrated”或“证据等级比当前更强”的 wording；
3. 用最小改动把它们压回当前 evidence stack；
4. 对本轮改动过的非结果层 section 增加源码注释，格式建议为：

```tex
% T78-SCOPE: evidence-alignment; source=T77-manifest,T70,T72
```

### B. `statcalib` 层级降权

至少完成以下事项：

1. 让 `statcalib` 在 `Numerical Results` 中的视觉层级明显低于 `T24` 主结果层与 `FR7/FR6` 支撑层；
2. 继续保留 `extension-lane / no-promotion / persistent-tie / no unique clean threshold`；
3. 不允许通过新的段落顺序、标题级别或表格位置，让 `statcalib` 看起来像并列主结果；
4. 如有必要，可同步更新 `paper_results_section_assembly_pack.md` 或 `paper_results_callout_sheet.md`，让 note 与 upstream assembly guidance 一致。

### C. 排版 warning 收口

至少完成以下事项：

1. 若本地已有可用工具链，对更新后的 note 执行一次受控编译；
2. 统计编译前后的 `Underfull \hbox` 数量或至少列出主要 warning 所在行段；
3. 修掉明显可修的排版问题；
4. 若仍有残余 warning，必须在 closeout 文档中说明：
   - 剩余数量或代表性位置
   - 为什么暂不继续消除
   - 它们是否影响当前 note 的 paper-facing 使用

### D. 机械化 section-scope 审计

`docs/paper_materials/paper_note_alignment_and_layout_closeout.md` 至少要包含：

1. 本轮实际改动的 section / subsection 列表；
2. 每个 section 的改动目标：
   - evidence alignment
   - hierarchy demotion
   - layout cleanup
3. 每个 section 对应的 source task / review；
4. 哪些 section 仍未被本轮校准，以及为什么未校准；
5. 编译 warning 的 before / after 摘要；
6. `T78-SCOPE` 注释覆盖情况。

### E. README / 入口同步

至少完成以下事项：

1. `docs/paper_notes/README.md` 明确区分：
   - `T77` = 结果层同步
   - `T78` = 非结果层校准 / hierarchy / layout closeout
2. `docs/paper_materials/README.md` 新增或更新对 `paper_note_alignment_and_layout_closeout.md` 的索引说明；
3. 不把任何 README 写成 evidence 升级文案。

## 预期输出

Worker 必须产出：

- 更新后的 `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`
- 更新后的 `docs/paper_notes/README.md`
- 更新后的 `docs/paper_materials/README.md`
- 新增 `docs/paper_materials/paper_note_alignment_and_layout_closeout.md`
- `docs/review/T78_review.md`
- `docs/for_human/T78_explanation.md`
- `docs/worker_summary/T78_worker_summary.md`

如确有必要，也可同步产出：

- 更新后的 `docs/paper_materials/paper_note_results_sync_manifest.md`
- 更新后的 `docs/paper_materials/paper_results_section_assembly_pack.md`
- 更新后的 `docs/paper_materials/paper_results_callout_sheet.md`

若本地编译成功，还应产出：

- 更新后的 `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.pdf`
- 对应 LaTeX 辅助文件

## 验证

Worker 必须实际执行并报告：

1. `CNN_FPGA_GKP_theory_note_draft.tex` 是否只修改了允许章节范围
2. `rg -n "T78-SCOPE"` 是否能覆盖本轮改动过的非结果层 section
3. `statcalib` 相关段落是否仍保留：
   - extension lane
   - no promotion
   - persistent tie / no unique threshold
4. `T24` / `T48` / `T49/T71/T72` / `T64-T70` 的边界 wording 是否仍然保守
5. `paper_note_alignment_and_layout_closeout.md` 是否覆盖：
   - 改动 section
   - source 绑定
   - 未校准 section
   - warning before / after
6. 如有本地 LaTeX 工具链，编译是否成功；`Underfull \hbox` 是否减少或被明确解释
7. `git diff --name-only -- runs`
8. `git diff --name-only -- artifacts`
9. `git diff --name-only -- cnn_fpga physics benchmark tests`
10. `git diff --name-only -- docs/00_project_snapshot.md docs/01_legacy_audit.md docs/02_experiment_plan.md docs/03_hil_p4_boundary_audit.md docs/04_task_board.md docs/05_decision_log.md docs/06_repo_noise_governance.md docs/07_handoff.md docs/08_risks_and_open_questions.md`

## 完成标准

只有同时满足以下条件，`T78` 才可视为完成：

1. note 的非结果层 evidence-facing wording 已完成一轮有界校准；
2. `statcalib` 的视觉层级已被明确降权，不再容易被看成并列主结果；
3. 已形成一份更机械的 section-scope 审计记录；
4. 如有本地工具链，已完成受控编译与 warning 收口；如仍有残余 warning，已清楚记录；
5. 没有把任何 blocked / partial / extension-lane / no-promotion / gate-only 证据静默升级；
6. 没有改动治理文档、源码、测试、`runs/`、`artifacts/`。
