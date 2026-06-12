# T79：论文材料 reopen gate 与 bounded prose 扩写就绪性评审

## 状态

- 由 Captain 在 `2026-06-12` 基于 `T78` 的 `PASS` closeout 提出
- 当前阶段：`Phase 2: Controlled Development`
- 决策状态：`Go`
- 任务类型：高强度 docs-only gate / readiness review 任务

## 为什么现在做这个任务

`T74`、`T75`、`T76`、`T77`、`T78` 已经把主线 paper-facing 材料推进到一个新的状态：

1. `T74` 已收口 stable-ID 结果表、caption、insertion map 与 traceability；
2. `T75` 已收口 bounded Results authoring；
3. `T76` 已收口真实 rendered preview、visual QA 与 Results-section assembly；
4. `T77` 已把结果层材料同步进现有 note-draft，并收口 `T76` traceability；
5. `T78` 已收口 note 的非结果层校准、`statcalib` hierarchy 降权、LaTeX layout warning 与 section-scope 审计。

这意味着当前主线缺的已经不再是“再补一轮 note 校准”，而是一个更明确的问题：

> 现在这套 note / results pack / claim-evidence ledger / risk table / supporting manifests，是否已经足够支撑下一轮受控 prose reopen？

如果没有这一层 gate，主线很容易再次出现两种错误：

1. 过早把“材料比以前更整齐了”误写成“已经可以直接恢复 full-manuscript 扩写”；
2. 反过来继续机械补文档，却没有明确指出真正阻塞 prose reopen 的缺口到底是什么。

因此，`T79` 的目标不是写 prose，而是先做一张更强的 gate：

1. 给出唯一的 reopen/readiness verdict；
2. 把当前材料栈拆成章节级 readiness matrix；
3. 把缺口整理成 gap-to-action matrix；
4. 最后只推荐一张后续任务，而不是同时打开多条线。

## 前置条件

只有在以下条件都满足时，`T79` 才可执行：

- `T78` 已完成并通过 Captain `PASS`
- 以下文件已存在：
  - `docs/review/T78_review.md`
  - `docs/paper_materials/paper_note_alignment_and_layout_closeout.md`
  - `docs/paper_materials/paper_note_results_sync_manifest.md`
  - `docs/paper_materials/paper_results_section_assembly_pack.md`
  - `docs/paper_materials/paper_claim_evidence_ledger.md`
  - `docs/paper_materials/paper_claim_risk_table.md`
  - `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`

如果这些前提不满足，Worker 不得在 `T79` 中重建上游材料，而必须如实报告 blocker。

## 目标

在不运行任何新实验、不修改任何源码/配置/历史结果、不触碰治理文档、不直接恢复 prose 扩写的前提下，完成以下五件事：

1. 产出一份**论文材料 reopen gate 报告**：
   - 明确当前材料栈的 strongest supported truth；
   - 明确哪些部分已经可以支撑 bounded prose reopen；
   - 明确哪些部分仍然不足。
2. 产出一份**章节级 readiness matrix**：
   - 至少覆盖标题/摘要、引言、相关工作、方法相关章节、实验设置、结果、讨论、结论、图表材料、claim/evidence ledger、risk table、training/runtime/board supporting boundary；
   - 每项必须标出当前状态和证据来源。
3. 产出一份**gap-to-action matrix**：
   - 把 prose reopen 之前的真实阻塞拆成有限缺口；
   - 每个缺口都要绑定现有 evidence / review / material；
   - 不得虚构不存在的实验或资源条件。
4. 给出**唯一 gate verdict**：
   - `GO_FOR_BOUNDED_PROSE_REOPEN`
   - `CONDITIONAL_GO_WITH_PRE_REOPEN_FIXES`
   - `NO_GO_NEED_MORE_MATERIALS`
5. 只推荐**一张**后续任务：
   - 若 verdict 为 `GO`，则定义 bounded prose reopen 任务；
   - 若 verdict 为 `CONDITIONAL_GO` 或 `NO_GO`，则定义唯一的材料补缺任务；
   - 不得同时推荐多个并行主线任务。

## Allowed Files

Worker 只可修改或新增以下路径：

- `docs/tasks/Phase2/T79_paper_reopen_gate_and_prose_readiness_review.md`
- `docs/paper_materials/README.md`
- `docs/paper_materials/paper_reopen_gate_and_prose_readiness_review.md`
- `docs/paper_materials/paper_reopen_gap_matrix.md`
- `docs/review/T79_review.md`
- `docs/for_human/T79_explanation.md`
- `docs/worker_summary/T79_worker_summary.md`

## Docs To Update

Worker 必须更新：

- `docs/paper_materials/README.md`
- `docs/paper_materials/paper_reopen_gate_and_prose_readiness_review.md`
- `docs/paper_materials/paper_reopen_gap_matrix.md`
- `docs/review/T79_review.md`
- `docs/for_human/T79_explanation.md`
- `docs/worker_summary/T79_worker_summary.md`

## Forbidden Scope

Worker 不得：

- 修改任何治理文档 `docs/00_*` 到 `docs/08_*`
- 修改任何 `cnn_fpga/`、`physics/`、`benchmark/`、`tests/` 源码或测试
- 修改任何 `runs/`、`artifacts/`、`docs/evidence_packs/` 历史事实文件
- 修改 `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex` 或任何 note 编译产物
- 直接撰写或扩写 manuscript / note 正文
- 运行任何 benchmark、训练、`.tflite` smoke、real-board gate/regeneration/smoke
- 新增 figure/table 资产或改写任何历史结果表
- 修改 theory 分支材料、`docs/sidecar/*` 或 sidecar 输出
- 静默提升任何证据等级，尤其不得把：
  - `T24` 写成 paper-grade expanded benchmark
  - `T48` 写成 default-env / deployment closure
  - `T49/T71/T72` 写成 real-board execution success
  - `T64`-`T70` 写成 mature `statcalib` comparator promotion
- 以“gate review”为名直接批准 full-manuscript reopen、主文定稿或对外宣称论文已 ready

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
- paper-facing 材料：
  - `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`
  - `docs/paper_notes/README.md`
  - `docs/paper_materials/README.md`
  - `docs/paper_materials/paper_note_alignment_and_layout_closeout.md`
  - `docs/paper_materials/paper_note_results_sync_manifest.md`
  - `docs/paper_materials/paper_results_section_assembly_pack.md`
  - `docs/paper_materials/paper_results_callout_sheet.md`
  - `docs/paper_materials/paper_maintext_results_authoring_pack.md`
  - `docs/paper_materials/paper_claim_evidence_ledger.md`
  - `docs/paper_materials/paper_claim_risk_table.md`
  - `docs/paper_materials/paper_ablation_result_pack.md`
- review 边界：
  - `docs/review/T74_review.md`
  - `docs/review/T75_review.md`
  - `docs/review/T76_review.md`
  - `docs/review/T77_review.md`
  - `docs/review/T78_review.md`

## 固定边界

- 这是主线 reopen/readiness gate 任务，不是实验任务，也不是 prose drafting 任务
- 只允许“状态评审”“readiness 分层”“gap-to-action 拆解”“唯一 verdict”“唯一下一任务推荐”
- 不允许“新增结果”“补新实验”“改写主线结论”“先写 prose 再倒推 justify”
- `statcalib` 在 gate 报告中仍必须保留 extension-lane / no-promotion / no unique clean threshold 边界
- `T48` 在 gate 报告中仍必须保留 isolated current-host true runtime only 边界
- `T49/T71/T72` 在 gate 报告中仍必须保留 read-only gate / provenance / `NO_GO` 边界
- `T37` 仍必须保留 `blocked + lowest-priority backlog`
- 若某个章节或材料当前无法诚实支撑 prose reopen，必须写成 `needs_fix` / `blocked_by_evidence` / `defer_out_of_scope`，而不是弱化成模糊通过

## Gate 输出格式要求

### A. `paper_reopen_gate_and_prose_readiness_review.md`

至少应包含：

1. `Gate Verdict`
   - 只允许三种固定值之一：
     - `GO_FOR_BOUNDED_PROSE_REOPEN`
     - `CONDITIONAL_GO_WITH_PRE_REOPEN_FIXES`
     - `NO_GO_NEED_MORE_MATERIALS`
2. `Strongest Supported Truth`
   - 说明当前主线在 note、results、claims、风险、runtime、board 边界上分别能诚实说到哪里
3. `Section-Level Readiness Matrix`
   - 每项至少包含：
     - `area`
     - `status`
     - `why`
     - `evidence anchors`
4. `What Is Already Sufficient`
5. `What Still Blocks Reopen`
6. `Single Recommended Next Task`
   - 只允许一张

### B. `paper_reopen_gap_matrix.md`

至少应包含一个结构化表格，列建议至少包括：

- `gap_id`
- `gap_area`
- `current_symptom`
- `why_it_blocks_or_limits_reopen`
- `existing_evidence`
- `required_action`
- `can_be_solved_in_one_bounded_task`
- `priority`

### C. README 索引同步

`docs/paper_materials/README.md` 必须新增或更新：

1. `T79` 入口说明
2. `paper_reopen_gate_and_prose_readiness_review.md` 的用途
3. `paper_reopen_gap_matrix.md` 的用途
4. 明确写出：`T79` 是 gate，不是 prose reopen 本身

## Readiness Matrix 的最小覆盖范围

至少要覆盖以下 area：

1. 标题 / 摘要
2. 引言
3. Related Work / positioning
4. 方法相关章节
5. Experimental Setup
6. Numerical Results
7. Discussion
8. Conclusion
9. 主图 / 主表 / caption / insertion 路由
10. claim/evidence ledger
11. risk table
12. training/material supporting boundary
13. `.tflite` supporting boundary
14. real-board supporting boundary

每个 area 的 `status` 只允许使用以下值：

- `ready_for_bounded_reopen`
- `needs_fix_before_reopen`
- `blocked_by_evidence`
- `defer_out_of_scope`

## 唯一 verdict 的判定规则

### 可以给 `GO_FOR_BOUNDED_PROSE_REOPEN` 的前提

只有当以下条件同时成立时才可给出：

1. 主要章节的 paper-facing wording 已经与当前 evidence stack 对齐；
2. 主结果表/图/claim/evidence/risk 路由已足够支撑下一轮 bounded prose；
3. 当前缺口只剩 prose 组织与表达，不再是证据边界或材料真实性问题；
4. 没有任何一条会迫使 prose reopen 时再次误写 `.tflite` / real-board / statcalib / benchmark 边界。

### 可以给 `CONDITIONAL_GO_WITH_PRE_REOPEN_FIXES` 的前提

当材料总体接近可用，但仍有一张有限补缺任务能显著降低 reopen 风险时使用。

### 给 `NO_GO_NEED_MORE_MATERIALS` 的前提

当当前阻塞仍主要是材料/边界/证据充分性，而不是 prose 组织时使用。

## Verification

Worker 至少要完成以下验证：

1. `git diff --name-only` 范围核查：确认变更只落在 Allowed Files。
2. 对 gate 报告执行一次人工一致性自检：
   - 是否只有一个 verdict
   - 是否只有一个推荐后续任务
   - readiness matrix 是否覆盖要求的最小 area
3. 对 gap matrix 做一次结构检查：
   - 每个 gap 是否都绑定了现有 evidence
   - 是否没有把未来实验写成已存在事实
4. review 文件中明确写出：
   - 是否存在 blocker
   - verdict 是什么
   - 为什么不是更强或更弱的 verdict

## 完成标准

只有同时满足以下条件，`T79` 才算完成：

1. `paper_reopen_gate_and_prose_readiness_review.md` 已生成，且包含唯一 gate verdict
2. `paper_reopen_gap_matrix.md` 已生成，且缺口都绑定现有 evidence
3. `docs/paper_materials/README.md` 已同步 `T79` 入口
4. `docs/review/T79_review.md` 已写出正式 review 结论
5. `docs/for_human/T79_explanation.md` 已向作者说明“为什么现在是这个 gate verdict”
6. `docs/worker_summary/T79_worker_summary.md` 已总结：
   - 改了什么
   - 怎么验证
   - 剩余风险
7. 全程未越界到 prose drafting、实验执行、note 正文改写或治理文档修改

## 交付提醒

本任务的产出必须优先使用中文。

如果 Worker 判断当前材料已经可以进入下一轮 prose reopen，也只能在 `T79` 中给出 gate verdict 和下一张任务包建议，不得顺手开始写 prose。
