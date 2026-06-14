# T88：主线 bounded manual finish 执行与 surface freeze 收口包

## 状态

- 由 Captain 于 `2026-06-14` 基于 `T87` 的 `PASS` closeout 提出
- 当前阶段：`Phase 2: Controlled Development`
- 决策状态：`Go`
- 任务类型：高强度 docs-only 主线 manual-finish execution / surface-freeze / blocked-disclaimer 固化任务

## 为什么现在做这个任务

`T87` 已经把当前主线 note/material 压到一个更严格的作者终检结论：

- `paper_presubmission_regression_gate.md` 给出唯一 gate verdict：`GO_FOR_BOUNDED_AUTHOR_MANUAL_FINISH_ONLY`
- `paper_manual_finish_queue.md` 已把作者还可以继续做的动作压成 `MF01-MF05`
- `paper_submission_wording_redflag_register.md` 已固定危险表述与允许替代表述

所以当前主线真正缺的，不再是再做一轮 QA，也不是重新展开 prose/实验，而是：

1. 把 `MF01-MF05` 这批 bounded manual finish 动作真实执行掉；
2. 把执行后的主线 surface 选择、桥接、caption/wording 收束和 blocked disclaimer 固定成一套可审计答案；
3. 给出一个更窄的 freeze/handoff 结论，防止后续人工修改再次漂移。

因此，`T88` 不是投稿完成任务，不是 venue-template 适配，也不是 claim promotion。它只是把 `T87` 已允许的 manual finish 真正落完，并把落地后的主线写作 surface 冻住。

## 前置条件

只有以下条件全部满足时，`T88` 才可执行：

- `T87` 已完成并通过 Captain `PASS`
- 以下文件已存在：
  - `docs/review/T87_review.md`
  - `docs/paper_materials/paper_author_final_qa_checklist.md`
  - `docs/paper_materials/paper_presubmission_regression_gate.md`
  - `docs/paper_materials/paper_submission_wording_redflag_register.md`
  - `docs/paper_materials/paper_manual_finish_queue.md`
  - `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`
  - `docs/paper_materials/paper_submission_pack_assembly_manifest.md`
  - `docs/paper_materials/paper_submission_surface_route_map.md`
  - `docs/paper_materials/paper_submission_exclusion_register.md`
  - `docs/paper_materials/paper_submission_author_handoff.md`

如果这些前提不满足，Worker 不得在 `T88` 中补造上游事实，而必须如实汇报 blocker。

## 目标

在不新增实验、不修改任何源码/配置/历史结果、不触碰治理文档、不回收独立 theory 分支内容、也不把当前主线写成 submission-ready completed 的前提下，完成以下工作：

1. 新增 `paper_manual_finish_execution_log.md`，至少登记：
   - `mf_id`
   - `planned_action`
   - `executed_change`
   - `touched_surface`
   - `status`
   - `evidence_anchor`
   - `boundary_check`
   - `note`
2. 新增 `paper_mainline_surface_freeze_manifest.md`，至少登记：
   - `freeze_id`
   - `surface`
   - `selected_primary_representation`
   - `route_location`
   - `supporting_anchor`
   - `blocked_surface_preserved`
   - `note`
3. 新增 `paper_author_edit_decision_register.md`，至少登记：
   - `decision_id`
   - `decision_topic`
   - `options_considered`
   - `selected_option`
   - `reason`
   - `must_not_imply`
   - `evidence_anchor`
4. 新增 `paper_blocked_surface_disclaimer_table.md`，至少登记：
   - `disclaimer_id`
   - `blocked_surface`
   - `mandatory_boundary_wording`
   - `expected_location`
   - `status`
   - `note`
5. 新增 `paper_frozen_mainline_handoff_gate.md`，至少给出：
   - 当前 gate question
   - 已执行 manual-finish 动作
   - 已冻结的 mainline surfaces
   - 仍 blocked / excluded 的 surface
   - 唯一 gate verdict
   - gate verdict 只能在以下二者中选择其一：
     - `GO_FOR_FROZEN_MAINLINE_HANDOFF_ONLY`
     - `HOLD_FOR_MANUAL_FINISH_CLEANUP`
6. 对 `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex` 做最小必要 manual-finish 执行，只允许：
   - 执行 `MF01-MF05` 内已经允许的句法润色、桥接句压缩、caption 读者化、主呈现选择落地、排版/断句收束
   - 加强 blocked / excluded surface 的 reader-facing disclaimer 保留
   - 清除与 `T87` gate / queue / red-flag register 冲突的残余 wording drift
7. 更新 `docs/paper_notes/README.md` 与 `docs/paper_materials/README.md`，把 `T88` 的 execution log / freeze manifest / decision register / disclaimer table / handoff gate 入口登记清楚
8. 如本地 LaTeX 工具链可用，则完成一次受控编译刷新；如不可用，必须如实记录，不得伪造 compile 结论

## Allowed Files

Worker 只可修改或新增以下路径：

- `docs/tasks/Phase2/T88_mainline_bounded_manual_finish_and_surface_freeze.md`
- `docs/paper_notes/README.md`
- `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`
- `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.aux`
- `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.fdb_latexmk`
- `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.fls`
- `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.log`
- `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.out`
- `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.pdf`
- `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.synctex.gz`
- `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.toc`
- `docs/paper_materials/README.md`
- `docs/paper_materials/paper_manual_finish_execution_log.md`
- `docs/paper_materials/paper_mainline_surface_freeze_manifest.md`
- `docs/paper_materials/paper_author_edit_decision_register.md`
- `docs/paper_materials/paper_blocked_surface_disclaimer_table.md`
- `docs/paper_materials/paper_frozen_mainline_handoff_gate.md`
- `docs/review/T88_review.md`
- `docs/for_human/T88_explanation.md`
- `docs/worker_summary/T88_worker_summary.md`

## Docs To Update

Worker 必须更新：

- `docs/paper_notes/README.md`
- `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`
- `docs/paper_materials/README.md`
- `docs/paper_materials/paper_manual_finish_execution_log.md`
- `docs/paper_materials/paper_mainline_surface_freeze_manifest.md`
- `docs/paper_materials/paper_author_edit_decision_register.md`
- `docs/paper_materials/paper_blocked_surface_disclaimer_table.md`
- `docs/paper_materials/paper_frozen_mainline_handoff_gate.md`
- `docs/review/T88_review.md`
- `docs/for_human/T88_explanation.md`
- `docs/worker_summary/T88_worker_summary.md`

如执行了本地编译，还必须同步刷新对应的 note 编译产物。

## Forbidden Scope

Worker 不得：

- 修改任何治理文档 `docs/00_*` 到 `docs/08_*`
- 修改任何 `cnn_fpga/`、`physics/`、`benchmark/`、`tests/` 源码或测试
- 修改任何 `runs/`、`artifacts/`、`docs/evidence_packs/` 历史事实文件
- 运行任何 benchmark、训练、`.tflite` smoke、real-board gate/regeneration/smoke
- 新增或改写任何 stable-ID 结果资产、figure/table 文件本体、caption pack、insertion map
- 新建第二份主 note、第二份 manuscript 草稿，或把独立 theory 分支内容拉回 main
- 以“manual finish / surface freeze”之名顺手扩成 venue-template 适配、cover letter、投稿元数据、正式 submission package、claim promotion、deployment story 升级或 hardware-ready retelling
- 把 `T88` 直接写成 “submission-ready completed”

## 强制 guardrails

以下口径在 `T88` 中必须继续保留：

1. `T24` 仍是 mainline frozen-set formal software-HIL 历史主锚点。
2. `FR6/FR7` 仍只可写成 descriptive support，而不是 causal closure。
3. `FR8/statcalib` 仍只可写成 extension lane / no-promotion / no unique clean threshold。
4. training/material 仍只可写成 canonical chain intact + one clean CPU-only bounded rerun。
5. `.tflite` 仍只可写成 isolated current-host true runtime for selected preserved artifacts。
6. real-board 仍只可写成 read-only gate / regeneration / provenance with current-host `NO_GO`。
7. 当前暂无 `Linux + FPGA` 硬件宿主，因此任何 hardware-dependent surface 都只可保留为 blocked / future-host requirement。
8. `T88` 的目标是 manual-finish execution + surface freeze + blocked-disclaimer 固化，不是 claim promotion；任何 blocked surface 仍必须保持 blocked。
9. 独立 theory 分支仍与本任务隔离；`T88` 只允许处理 main 分支当前 note/material 的 manual finish，不得引入 branch-specific 理论扩写。

## Section 注释要求

如果 `T88` 修改了某个 section，必须在相邻位置至少保留一条 `T88` 注释，例如：

```tex
% T88-MANUAL: Conclusion
```

最低要求：

- 所有被 `T88` 实际修改的 section，都必须有一条 `% T88-MANUAL: ...` 注释；
- `paper_manual_finish_execution_log.md` 与 `paper_mainline_surface_freeze_manifest.md` 中列出的 touched sections，必须能回链到源码中的 `% T88-MANUAL: ...` 注释。

## 推荐执行顺序

1. 先阅读 `T87_review`、`paper_presubmission_regression_gate.md`、`paper_manual_finish_queue.md` 与 `paper_submission_wording_redflag_register.md`，确认 `T87` 允许做什么、不允许做什么。
2. 先写 `paper_manual_finish_execution_log.md` 的初版框架，把 `MF01-MF05` 全部列进去。
3. 再写 `paper_author_edit_decision_register.md`，先把 figure/table 主呈现、bridge 句、caption 简化、页数压缩等人工决策框架固定下来。
4. 然后回到 note，执行真正必要的 bounded manual finish，并给所有 touched sections 加 `% T88-MANUAL: ...` 注释。
5. 再写 `paper_mainline_surface_freeze_manifest.md`，把执行后的主 surface 选择和 route 固定下来。
6. 再写 `paper_blocked_surface_disclaimer_table.md`，确保 blocked/excluded surface 的 disclaimer 没被手工终修稀释掉。
7. 最后写 `paper_frozen_mainline_handoff_gate.md`，给出唯一 freeze/handoff verdict。
8. 更新两个 README。
9. 如本地工具链可用，执行一次 note 编译并记录 log scan 结果。
10. 最后写 `review`、`for_human` 与 `worker_summary`。

## Verification

至少完成以下验证：

1. 必须使用 allowlist-scoped diff 验证，而不是把全仓 `git diff --name-only` 直接当作 `T88` 改动清单。
2. 必须确认：
   - `T80` 的 `% T80-REOPEN` 标记仍保留；
   - `T81` 的 `% T81-CALIBRATION` 标记仍保留；
   - `T82` 的 `% T82-SUPPORT` 标记仍保留；
   - `T83` 的 `% T83-CLOSEOUT` 标记仍保留；
   - `T84` 的 `% T84-POLISH` 标记仍保留；
   - `T85` 的 `% T85-PREFLIGHT` 标记仍保留；
   - `T86` 的 `% T86-ASSEMBLY` 标记仍保留；
   - `T87` 的 `% T87-QA` 标记仍保留。
3. `paper_manual_finish_execution_log.md` 必须完整覆盖 `MF01-MF05`，并明确每一项是 `executed`、`left_as_is` 还是 `not_applied_with_reason`。
4. `paper_mainline_surface_freeze_manifest.md` 必须至少覆盖：
   - main text primary result presentation
   - appendix / supplement bridge route
   - boundary schematic presentation choice
   - blocked surface preservation
5. `paper_author_edit_decision_register.md` 必须至少记录 4 条真实编辑决策，而不是空泛原则。
6. `paper_blocked_surface_disclaimer_table.md` 必须至少列出以下五类 blocked/excluded surface：
   - real-board execution / timing / resource
   - default-env / cross-host `.tflite` portability
   - full training reproducibility
   - `FR8/statcalib` mature comparator / unique clean threshold
   - expanded benchmark / stronger oracle baseline
7. `paper_frozen_mainline_handoff_gate.md` 必须明确写出唯一 gate verdict，且 verdict 只能是：
   - `GO_FOR_FROZEN_MAINLINE_HANDOFF_ONLY`
   - `HOLD_FOR_MANUAL_FINISH_CLEANUP`
8. 如果 note 某个 section 被 `T88` 修改，源码中必须能 grep 到对应的 `% T88-MANUAL: ...` 注释。
9. 必须再次执行 red-flag 扫描，确认 manual finish 没有把禁写表述误写回主叙述句。
10. 如本地工具链可用并执行编译，需要记录：
   - 使用的工具链
   - 编译目标
   - 产物集合
   - `.log` 关键字扫描结果
11. 如工具链不可用，必须在 `paper_frozen_mainline_handoff_gate.md` 或 `worker_summary` 中明确写出未编译原因。

## 完成标准

只有同时满足以下条件，`T88` 才算完成：

1. `paper_manual_finish_execution_log.md` 已完成，并覆盖 `MF01-MF05`。
2. `paper_mainline_surface_freeze_manifest.md` 已完成。
3. `paper_author_edit_decision_register.md` 已完成。
4. `paper_blocked_surface_disclaimer_table.md` 已完成。
5. `paper_frozen_mainline_handoff_gate.md` 已完成，并给出唯一 gate verdict。
6. 被 `T88` 实际修改的 section 已全部加上 `% T88-MANUAL: ...` 注释。
7. `docs/paper_notes/README.md` 与 `docs/paper_materials/README.md` 已登记 `T88` 入口。
8. `docs/review/T88_review.md` 已给出正式 review 结论。
9. `docs/for_human/T88_explanation.md` 已向作者说明：
   - 本轮为什么是 manual-finish execution + surface freeze，而不是投稿完成；
   - 哪些手工动作已执行；
   - 哪些 blocked surface 仍必须被 disclaimer 锁住；
   - 为什么即使 `T88` 成功，也不自动等于 submission-ready completed。
10. `docs/worker_summary/T88_worker_summary.md` 已总结：
   - 改了什么
   - 怎么验证
   - 剩余风险

## 交付提醒

- 这是 `T87` 之后更偏“执行 manual finish + 冻结写作 surface”的 docs-only 强任务，不是再做一轮空泛 gate。
- 但它仍不是“恢复无界 full-manuscript 扩写”，更不是“直接投稿”。
- `T88` 的成功标准不是“论文现在已经完成投稿包”，而是“`T87` 允许的 bounded manual finish 已被执行、对应 surface 与 blocked disclaimer 已被固化、后续主线 handoff 不再容易漂移”。
