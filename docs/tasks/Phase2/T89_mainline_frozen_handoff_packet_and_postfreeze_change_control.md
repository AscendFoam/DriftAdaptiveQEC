# T89：主线 frozen-mainline handoff 包与 post-freeze change-control 收口

## 状态

- 由 Captain 于 `2026-06-15` 基于 `T88` 的 `PASS` closeout 提出
- 当前阶段：`Phase 2: Controlled Development`
- 决策状态：`Go`
- 任务类型：高强度 docs-only 主线 handoff / freeze-preserving / change-control 任务

## 为什么现在做这个任务

`T88` 已经完成了当前主线 note/material 的 bounded manual finish 执行，并把以下内容固定下来：

- `paper_manual_finish_execution_log.md`
- `paper_mainline_surface_freeze_manifest.md`
- `paper_author_edit_decision_register.md`
- `paper_blocked_surface_disclaimer_table.md`
- `paper_frozen_mainline_handoff_gate.md`

而且 `T88` 的唯一 gate verdict 已经收口为：

- `GO_FOR_FROZEN_MAINLINE_HANDOFF_ONLY`

这意味着当前主线的下一步，不再是继续改 note，也不是继续做 QA / assembly / manual finish，而是把已经冻结的主线答案整理成一套：

1. 可直接移交给后续作者/协作者/未来 Captain 的单一 handoff 包；
2. 可约束后续 main 分支人工修改的 post-freeze change-control 规则；
3. 可明确说明“哪些 surface 仍 blocked，未来必须靠什么新证据才允许重开”的 re-entry 条件表。

因此，`T89` 不是投稿完成任务，不是 full-manuscript reopen，不是 theory 分支并回 main，也不是任何实验/部署/真板补强任务。它只是在 `T88` 之后，把 frozen-mainline 的“可交接、可维护、不可误写”的边界再收紧一层。

## 前置条件

只有以下条件全部满足时，`T89` 才可执行：

- `T88` 已完成并通过 Captain `PASS`
- 以下文件已存在：
  - `docs/review/T88_review.md`
  - `docs/paper_materials/paper_manual_finish_execution_log.md`
  - `docs/paper_materials/paper_mainline_surface_freeze_manifest.md`
  - `docs/paper_materials/paper_author_edit_decision_register.md`
  - `docs/paper_materials/paper_blocked_surface_disclaimer_table.md`
  - `docs/paper_materials/paper_frozen_mainline_handoff_gate.md`
  - `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`

如果这些前提不满足，Worker 不得在 `T89` 中补造上游事实，而必须如实汇报 blocker。

## 目标

在不改动主线 note 正文、不新增实验、不修改任何源码/配置/历史结果、不触碰治理文档、也不把 frozen mainline 写成 submission-ready completed 的前提下，完成以下工作：

1. 新增 `paper_frozen_mainline_handoff_packet.md`，形成单一 handoff 入口，至少包含：
   - 当前 frozen-mainline 的唯一推荐入口
   - 当前主线允许被引用的核心材料
   - 明确仍 blocked / excluded 的 surface
   - main 分支后续维护时必须保留的 boundary wording
   - 不得外推的 non-claims
2. 新增 `paper_frozen_mainline_source_of_truth_map.md`，把当前 frozen-mainline 的主文/附录/补充/blocked surface 回链到已有 authoritative material，至少登记：
   - `surface_id`
   - `surface_name`
   - `current_primary_reader_entry`
   - `authoritative_source`
   - `must_not_imply`
   - `status`
   - `note`
3. 新增 `paper_postfreeze_change_control.md`，把 post-freeze 修改分层，至少区分：
   - 无需 reopen 的微小改动
   - 需要新的 bounded docs-only task 的改动
   - 需要新的 evidence task 才允许的改动
   - 当前 main 分支直接禁止的改动
4. 新增 `paper_blocked_surface_reentry_conditions.md`，把 blocked surface 的未来重开条件明确下来，至少登记：
   - `reentry_id`
   - `blocked_surface`
   - `current_block_reason`
   - `minimum_new_evidence_needed`
   - `candidate_future_task_type`
   - `must_not_shortcut`
5. 更新 `docs/paper_materials/README.md` 与 `docs/paper_notes/README.md`，把 `T89` 的 handoff / source-of-truth / change-control / re-entry 入口登记清楚。

## Allowed Files

Worker 只可修改或新增以下路径：

- `docs/tasks/Phase2/T89_mainline_frozen_handoff_packet_and_postfreeze_change_control.md`
- `docs/paper_materials/README.md`
- `docs/paper_notes/README.md`
- `docs/paper_materials/paper_frozen_mainline_handoff_packet.md`
- `docs/paper_materials/paper_frozen_mainline_source_of_truth_map.md`
- `docs/paper_materials/paper_postfreeze_change_control.md`
- `docs/paper_materials/paper_blocked_surface_reentry_conditions.md`
- `docs/review/T89_review.md`
- `docs/for_human/T89_explanation.md`
- `docs/worker_summary/T89_worker_summary.md`

## Docs To Update

Worker 必须更新：

- `docs/paper_materials/README.md`
- `docs/paper_notes/README.md`
- `docs/paper_materials/paper_frozen_mainline_handoff_packet.md`
- `docs/paper_materials/paper_frozen_mainline_source_of_truth_map.md`
- `docs/paper_materials/paper_postfreeze_change_control.md`
- `docs/paper_materials/paper_blocked_surface_reentry_conditions.md`
- `docs/review/T89_review.md`
- `docs/for_human/T89_explanation.md`
- `docs/worker_summary/T89_worker_summary.md`

## Forbidden Scope

Worker 不得：

- 修改任何治理文档 `docs/00_*` 到 `docs/08_*`
- 修改 `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex` 或任何 note 编译产物
- 修改任何 `cnn_fpga/`、`physics/`、`benchmark/`、`tests/` 源码或测试
- 修改任何 `runs/`、`artifacts/`、`docs/evidence_packs/` 历史事实文件
- 运行任何 benchmark、训练、`.tflite` smoke、real-board gate/regeneration/smoke
- 新增或改写任何 stable-ID 结果资产、figure/table 文件本体、caption pack、insertion map
- 新建第二份主 note、第二份 manuscript 草稿，或把独立 theory 分支内容拉回 main
- 以“handoff / change-control”之名顺手扩成 venue-template 适配、cover letter、投稿元数据、正式 submission package、claim promotion、deployment story 升级或 hardware-ready retelling
- 把 `T89` 直接写成 “submission-ready completed”

## 强制 guardrails

以下口径在 `T89` 中必须继续保留：

1. `T24` 仍是 mainline frozen-set formal software-HIL 历史主锚点。
2. `FR6/FR7` 仍只可写成 descriptive support，而不是 causal closure。
3. `FR8/statcalib` 仍只可写成 extension lane / no-promotion / no unique clean threshold。
4. training/material 仍只可写成 canonical chain intact + one clean CPU-only bounded rerun。
5. `.tflite` 仍只可写成 isolated current-host true runtime for selected preserved artifacts。
6. real-board 仍只可写成 read-only gate / regeneration / provenance with current-host `NO_GO`。
7. 当前暂无 `Linux + FPGA` 硬件宿主，因此任何 hardware-dependent surface 都只可保留为 blocked / future-host requirement。
8. `T89` 的目标是 handoff consolidation + source-of-truth map + post-freeze change-control + blocked-surface re-entry rule，不是 claim promotion。
9. 独立 theory 分支仍与本任务隔离；`T89` 不得把 theory branch 的内容回写到 main，也不得把 main 的 frozen-mainline handoff 写成 theory/main 已融合。

## 推荐执行顺序

1. 先阅读 `T88_review.md`、`paper_frozen_mainline_handoff_gate.md`、`paper_mainline_surface_freeze_manifest.md` 与 `paper_blocked_surface_disclaimer_table.md`，确认当前 frozen-mainline 的唯一结论。
2. 先写 `paper_frozen_mainline_handoff_packet.md` 的初版结构，把“当前可交接答案是什么”固定下来。
3. 再写 `paper_frozen_mainline_source_of_truth_map.md`，把主文 / 附录 / 补充 / blocked surface 的 authoritative source 一一回链。
4. 再写 `paper_postfreeze_change_control.md`，把后续修改分成无需 reopen / 需 bounded docs-only task / 需新 evidence task / main 直接禁止四层。
5. 再写 `paper_blocked_surface_reentry_conditions.md`，把 blocked surface 的未来重开条件与不得走捷径的规则写清楚。
6. 最后更新两个 README、`review`、`for_human` 与 `worker_summary`。

## Verification

至少完成以下验证：

1. 必须使用 allowlist-scoped diff 验证，而不是把全仓 `git diff --name-only` 直接当作 `T89` 改动清单。
2. 必须确认 `T89` 没有修改：
   - `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`
   - 任一 note 编译产物
   - 任一 `runs/` / `artifacts/` / `docs/evidence_packs/` 文件
3. `paper_frozen_mainline_handoff_packet.md` 必须明确保留 `GO_FOR_FROZEN_MAINLINE_HANDOFF_ONLY`，且不得把它重写成 submission-ready completion。
4. `paper_frozen_mainline_source_of_truth_map.md` 必须至少覆盖 `T88` freeze manifest 中已经冻结的全部 mainline surfaces。
5. `paper_postfreeze_change_control.md` 必须至少给出 8 条具体 change-control 规则，且不能只写空泛原则。
6. `paper_blocked_surface_reentry_conditions.md` 必须至少覆盖以下 blocked surface：
   - real-board execution / timing / resource
   - default-env / cross-host `.tflite` portability
   - full training reproducibility
   - `FR8/statcalib` mature comparator / unique clean threshold
   - expanded benchmark / stronger oracle baseline
   - theory-branch content mergeback into main
7. 两个 README 必须把 `T89` 新增文档登记进去，并明确其“不升级证据等级”的边界。

## 交付物要求

Worker 完成后必须交付：

1. `docs/review/T89_review.md`
2. `docs/for_human/T89_explanation.md`
3. `docs/worker_summary/T89_worker_summary.md`

其中 `worker_summary` 必须明确写出：

- 新增了哪些 frozen-mainline handoff / change-control 文档
- 哪些 source-of-truth surface 已被回链
- 哪些 blocked surface 仍保持 blocked
- 哪些改动未来仍必须单开新 evidence task，不能在 main 上手工推进
