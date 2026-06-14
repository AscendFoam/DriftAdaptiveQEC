# T87：主线作者终检与 pre-submission QA 收口包

## 状态

- 由 Captain 于 `2026-06-14` 基于 `T86` 的 `PASS` closeout 提出
- 当前阶段：`Phase 2: Controlled Development`
- 决策状态：`Go`
- 任务类型：高强度 docs-only 主线作者终检 / pre-submission regression gate / wording red-flag 收口任务

## 为什么现在做这个任务

`T86` 已经把当前主线的 submission-facing 装配问题收到了一个可审计答案里：

1. `paper_submission_pack_assembly_manifest.md` 已明确哪些 surface 进入当前 package；
2. `paper_submission_surface_route_map.md` 已明确 main text / appendix / supplement / exclusion route；
3. `paper_submission_exclusion_register.md` 已明确当前不能进入 submission-facing package 的 blocked/excluded surface；
4. `paper_submission_author_handoff.md` 已明确当前作者还能继续做什么、绝不能写强什么。

因此当前主线真正缺的已经不再是再来一轮 assembly，而是更严格的作者侧 QA：

- 当前 note 与 `T74-T86` 材料链之间是否仍有 wording regression、route drift、边界 retelling 漂移；
- 是否还存在会把当前主线误写成 `submission-ready completed`、`real-board success`、`default-env .tflite portability closed`、`full training reproducibility closed`、`statcalib mature comparator` 的红旗表述；
- 作者后续如果继续做人工润色、手工排版、投稿前整理，哪些属于允许的 bounded manual finish，哪些会越界。

所以，`T87` 的目标不是宣布可以正式投稿，更不是恢复无界 full-manuscript 扩写；它只负责给出一份更严格的 author-final QA / pre-submission gate 答案。

## 前置条件

只有以下条件全部满足时，`T87` 才可执行：

- `T86` 已完成并通过 Captain `PASS`
- 以下文件已存在：
  - `docs/review/T86_review.md`
  - `docs/paper_materials/paper_submission_pack_assembly_manifest.md`
  - `docs/paper_materials/paper_submission_surface_route_map.md`
  - `docs/paper_materials/paper_submission_exclusion_register.md`
  - `docs/paper_materials/paper_submission_author_handoff.md`
  - `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`
  - `docs/paper_materials/paper_claim_evidence_ledger.md`
  - `docs/paper_materials/paper_result_figure_ledger.md`
  - `docs/paper_materials/paper_claim_risk_table.md`

如果这些前提不满足，Worker 不得在 `T87` 中补造上游材料，而必须如实汇报 blocker。

## 目标

在不运行任何新实验、不修改任何源码/配置/历史结果、不触碰治理文档、不回收独立 theory 分支内容、也不把当前主线写成 submission-ready 完成态的前提下，完成以下工作：

1. 新增 `paper_author_final_qa_checklist.md`，至少登记：
   - `qa_id`
   - `surface_or_section`
   - `check_type`
   - `pass_condition`
   - `evidence_anchor`
   - `status`
   - `manual_note`
2. 新增 `paper_presubmission_regression_gate.md`，至少给出：
   - 当前 gate question
   - 已通过检查
   - 仍 blocked / 不在当前 scope 的表面
   - 唯一 gate verdict
   - gate verdict 只能在以下二者中选择其一：
     - `GO_FOR_BOUNDED_AUTHOR_MANUAL_FINISH_ONLY`
     - `HOLD_FOR_MAINLINE_QA_FIXES`
3. 新增 `paper_submission_wording_redflag_register.md`，至少登记：
   - `redflag_id`
   - `forbidden_wording`
   - `why_wrong`
   - `allowed_replacement`
   - `evidence_anchor`
   - `scan_result`
4. 新增 `paper_manual_finish_queue.md`，至少登记：
   - `queue_id`
   - `allowed_manual_action`
   - `why_manual`
   - `depends_on`
   - `must_not_upgrade`
   - `owner`
5. 对 `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex` 做最小必要 QA 导向 refresh，只允许：
   - 修正与 `T86` 装配答案冲突的极小 wording drift
   - 修正会误导为 submission-ready / deployment-ready / hardware-ready 的极小表述
   - 加强对 blocked / excluded surface 的 reader-facing 边界提示
6. 更新 `docs/paper_notes/README.md` 与 `docs/paper_materials/README.md`，把 `T87` 的 QA checklist / gate / red-flag / manual-finish 入口登记清楚
7. 如本地 LaTeX 工具链可用，则完成一次受控编译刷新；如不可用，必须如实记录，不得伪造 compile 结论

## Allowed Files

Worker 只可修改或新增以下路径：

- `docs/tasks/Phase2/T87_mainline_author_final_qa_and_presubmission_gate.md`
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
- `docs/paper_materials/paper_author_final_qa_checklist.md`
- `docs/paper_materials/paper_presubmission_regression_gate.md`
- `docs/paper_materials/paper_submission_wording_redflag_register.md`
- `docs/paper_materials/paper_manual_finish_queue.md`
- `docs/review/T87_review.md`
- `docs/for_human/T87_explanation.md`
- `docs/worker_summary/T87_worker_summary.md`

## Docs To Update

Worker 必须更新：

- `docs/paper_notes/README.md`
- `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`
- `docs/paper_materials/README.md`
- `docs/paper_materials/paper_author_final_qa_checklist.md`
- `docs/paper_materials/paper_presubmission_regression_gate.md`
- `docs/paper_materials/paper_submission_wording_redflag_register.md`
- `docs/paper_materials/paper_manual_finish_queue.md`
- `docs/review/T87_review.md`
- `docs/for_human/T87_explanation.md`
- `docs/worker_summary/T87_worker_summary.md`

如执行了本地编译，还必须同步刷新对应的 note 编译产物。

## Forbidden Scope

Worker 不得：

- 修改任何治理文档 `docs/00_*` 到 `docs/08_*`
- 修改任何 `cnn_fpga/`、`physics/`、`benchmark/`、`tests/` 源码或测试
- 修改任何 `runs/`、`artifacts/`、`docs/evidence_packs/` 历史事实文件
- 运行任何 benchmark、训练、`.tflite` smoke、real-board gate/regeneration/smoke
- 新增或改写任何 stable-ID 结果资产、figure/table、caption、insertion map
- 新建第二份主 note、第二份 manuscript 草稿，或把独立 theory 分支内容拉回 main
- 以“作者终检 / pre-submission QA”之名顺手扩成 cover letter、投稿元数据、期刊模板适配、正式 submission package、claim promotion、deployment story 升级或 hardware-ready retelling
- 把 `T87` 直接写成 “submission-ready completed”

## 强制 guardrails

以下口径在 `T87` 中必须继续保留：

1. `T24` 仍是 mainline frozen-set formal software-HIL 历史主锚点。
2. `FR6/FR7` 仍只可写成 descriptive support，而不是 causal closure。
3. `FR8/statcalib` 仍只可写成 extension lane / no-promotion / no unique clean threshold。
4. training/material 仍只可写成 canonical chain intact + one clean CPU-only bounded rerun。
5. `.tflite` 仍只可写成 isolated current-host true runtime for selected preserved artifacts。
6. real-board 仍只可写成 read-only gate / regeneration / provenance with current-host `NO_GO`。
7. 当前暂无 `Linux + FPGA` 硬件宿主，因此任何 hardware-dependent surface 都只可保留为 blocked / future-host requirement。
8. `T87` 的目标是 QA / red-flag / manual-finish 边界固化，不是 claim promotion；任何 blocked surface 仍必须保持 blocked。
9. 独立 theory 分支仍与本任务隔离；`T87` 只允许处理 main 分支当前 note/material 的 QA，不得引入 branch-specific 理论扩写。

## Section 注释要求

如果 `T87` 修改了某个 section，必须在相邻位置至少保留一条 `T87` 注释，例如：

```tex
% T87-QA: Conclusion
```

最低要求：

- 所有被 `T87` 实际修改的 section，都必须有一条 `% T87-QA: ...` 注释；
- `paper_author_final_qa_checklist.md` 与 `paper_presubmission_regression_gate.md` 中列出的 touched sections，必须能回链到源码中的 `% T87-QA: ...` 注释。

## 推荐执行顺序

1. 先阅读 `T86_review`、`paper_submission_pack_assembly_manifest.md`、`paper_submission_surface_route_map.md`、`paper_submission_exclusion_register.md` 与 `paper_submission_author_handoff.md`，确认 `T86` 已回答什么、没回答什么。
2. 先写 `paper_submission_wording_redflag_register.md`，把当前最危险的 overclaim / wording drift 风险固定下来。
3. 再写 `paper_author_final_qa_checklist.md`，把 note、surface route、claim/evidence、blocked surface、compile 状态等检查项列成单一 QA 列表。
4. 再写 `paper_manual_finish_queue.md`，把作者还可以继续人工完善的动作限制在 bounded manual finish 范围内。
5. 然后回看 `note`，只处理真正被 QA 结论要求修的最小 wording / boundary 提示，并为 touched sections 加 `% T87-QA: ...` 注释。
6. 再写 `paper_presubmission_regression_gate.md`，给出唯一 gate verdict。
7. 更新两个 README。
8. 如本地工具链可用，执行一次 note 编译并记录 log scan 结果。
9. 最后写 `review`、`for_human` 与 `worker_summary`。

## Verification

至少完成以下验证：

1. 必须使用 allowlist-scoped diff 验证，而不是把全仓 `git diff --name-only` 直接当作 `T87` 改动清单。
2. 必须确认：
   - `T80` 的 `% T80-REOPEN` 标记仍保留；
   - `T81` 的 `% T81-CALIBRATION` 标记仍保留；
   - `T82` 的 `% T82-SUPPORT` 标记仍保留；
   - `T83` 的 `% T83-CLOSEOUT` 标记仍保留；
   - `T84` 的 `% T84-POLISH` 标记仍保留；
   - `T85` 的 `% T85-PREFLIGHT` 标记仍保留；
   - `T86` 的 `% T86-ASSEMBLY` 标记仍保留。
3. `paper_author_final_qa_checklist.md` 必须至少列出：
   - `qa_id`
   - `surface_or_section`
   - `check_type`
   - `pass_condition`
   - `evidence_anchor`
   - `status`
   - `manual_note`
4. `paper_presubmission_regression_gate.md` 必须明确写出唯一 gate verdict，且 verdict 只能是：
   - `GO_FOR_BOUNDED_AUTHOR_MANUAL_FINISH_ONLY`
   - `HOLD_FOR_MAINLINE_QA_FIXES`
5. `paper_submission_wording_redflag_register.md` 必须至少列出：
   - `redflag_id`
   - `forbidden_wording`
   - `why_wrong`
   - `allowed_replacement`
   - `evidence_anchor`
   - `scan_result`
6. `paper_manual_finish_queue.md` 必须至少列出：
   - `queue_id`
   - `allowed_manual_action`
   - `why_manual`
   - `depends_on`
   - `must_not_upgrade`
   - `owner`
7. 必须对当前 note/material 执行一轮 red-flag 扫描，并在 `paper_submission_wording_redflag_register.md` 或 `worker_summary` 中记录至少以下四类是否命中：
   - submission-ready completed
   - real-board execution success / hardware-ready
   - default-env / cross-host `.tflite` portability closed
   - full training reproducibility / mature `statcalib` comparator
8. 如果 note 某个 section 被 `T87` 修改，源码中必须能 grep 到对应的 `% T87-QA: ...` 注释。
9. 如本地工具链可用并执行编译，需要记录：
   - 使用的工具链
   - 编译目标
   - 产物集合
   - `.log` 关键字扫描结果
10. 如工具链不可用，必须在 `paper_presubmission_regression_gate.md` 或 `worker_summary` 中明确写出未编译原因。

## 完成标准

只有同时满足以下条件，`T87` 才算完成：

1. `paper_author_final_qa_checklist.md` 已完成。
2. `paper_presubmission_regression_gate.md` 已完成，并给出唯一 gate verdict。
3. `paper_submission_wording_redflag_register.md` 已完成。
4. `paper_manual_finish_queue.md` 已完成。
5. 被 `T87` 实际修改的 section 已全部加上 `% T87-QA: ...` 注释。
6. `docs/paper_notes/README.md` 与 `docs/paper_materials/README.md` 已登记 `T87` 入口。
7. `docs/review/T87_review.md` 已给出正式 review 结论。
8. `docs/for_human/T87_explanation.md` 已向作者说明：
   - 本轮为什么不是新实验而是更强的 QA / gate 收口；
   - 哪些内容可以继续做 bounded manual finish；
   - 哪些内容仍必须 blocked / excluded；
   - 为什么即使 `T87` 成功，也不自动等于 submission-ready completed。
9. `docs/worker_summary/T87_worker_summary.md` 已总结：
   - 改了什么
   - 怎么验证
   - 剩余风险

## 交付提醒

- 这是比 `T86` 更偏“作者终检 / 投稿前 QA 纪律”的 docs-only 任务，因为它不再只是装配，而是要把 red-flag、manual-finish 与 regression gate 一并收清。
- 但它仍不是“恢复无界 full-manuscript 扩写”，更不是“直接投稿”。
- `T87` 的成功标准不是“把论文做成最终成稿”，而是“把当前 mainline note/material 是否已经足够支撑 bounded manual finish、哪些地方仍绝不能写强、哪些事项必须继续手工完成”全部变成可审计事实。
