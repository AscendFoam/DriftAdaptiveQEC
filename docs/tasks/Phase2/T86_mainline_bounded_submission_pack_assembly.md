# T86：主线 bounded submission-pack assembly 与显式 exclusion route 收口

## 状态

- 由 Captain 于 `2026-06-14` 基于 `T85` 的 `PASS` closeout 提出
- 当前阶段：`Phase 2: Controlled Development`
- 决策状态：`Go`
- 任务类型：高强度 docs-only 主线 assembly / exclusion-route / author-handoff 任务

## 为什么现在做这个任务

`T85` 已经完成了三件关键事情：

1. 清掉了 `T84` 留下的唯一 residual wording-lag。
2. 建立了 `paper_submission_readiness_preflight_gate.md`。
3. 建立了 `paper_submission_blocker_matrix.md`，并给出唯一结论 `GO_FOR_BOUNDED_SUBMISSION_PACK_ASSEMBLY`。

这意味着当前主线已经不再缺：

- ready sections 的 prose reopen；
- contribution / methods calibration；
- supporting-boundary closeout；
- full-note consistency gate；
- reader-facing final polish；
- submission-readiness preflight。

当前缺的也不是新实验，不是新图，不是新 benchmark，更不是投稿完成态本身。当前真正缺的是一层更严格的 **submission-facing assembly discipline**：

1. 现有 mainline note / paper-material 中，哪些内容应该进入 submission-facing package，必须有显式 route。
2. 哪些 surface 仍只能保留为 blocked / excluded / appendix-only / supplement-only，必须有显式 exclusion register。
3. 作者在后续继续手工完善时，需要一份单点 handoff，明确“现在能继续写什么，不能继续写什么，为什么”。

因此，`T86` 的目标不是直接宣布 submission-ready pack completed，也不是恢复无界 full-manuscript 扩写，而是：

- 产出一份 **submission-pack assembly manifest**
- 产出一份 **surface route map**
- 产出一份 **explicit exclusion register**
- 产出一份 **author handoff**
- 对当前主线 note 做极小范围、装配导向、边界保持型 refresh

## 前置条件

只有以下条件全部满足时，`T86` 才可执行：

- `T85` 已完成并通过 Captain `PASS`
- 以下文件已存在：
  - `docs/review/T85_review.md`
  - `docs/paper_materials/paper_submission_readiness_preflight_gate.md`
  - `docs/paper_materials/paper_submission_blocker_matrix.md`
  - `docs/paper_materials/paper_residual_state_lag_sweep.md`
  - `docs/paper_materials/paper_claim_evidence_ledger.md`
  - `docs/paper_materials/paper_result_figure_ledger.md`
  - `docs/paper_materials/paper_claim_risk_table.md`
  - `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`

如果这些前提不满足，Worker 不得在 `T86` 中补造上游材料，而必须如实汇报 blocker。

## 目标

在不运行任何新实验、不修改任何源码/配置/历史结果、不触碰治理文档、不切入独立 theory 分支大范围改写、也不把当前主线写成 submission-ready 完成态的前提下，完成以下工作：

1. 新增 `paper_submission_pack_assembly_manifest.md`，至少登记：
   - `surface_id`
   - `surface_role`
   - `included_source`
   - `evidence_anchor`
   - `author_action`
2. 新增 `paper_submission_surface_route_map.md`，至少登记：
   - `claim_or_section`
   - `main_text_route`
   - `appendix_route`
   - `supplement_route`
   - `exclusion_note`
   - `source_anchor`
3. 新增 `paper_submission_exclusion_register.md`，至少登记：
   - `exclusion_id`
   - `blocked_surface`
   - `why_excluded_now`
   - `do_not_claim_wording`
   - `future_unblock_task`
4. 新增 `paper_submission_author_handoff.md`，至少明确：
   - 当前 submission-facing package 已经具备的内容
   - 仍未完成的 blocked surface
   - 作者后续可以继续做的 bounded polishing / manual editorial action
   - 绝对不能写强的 claim / boundary
5. 对 `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex` 做**最小必要**装配导向 refresh，只允许：
   - 对已有 mainline section 加强 route / exclusion 过渡句
   - 对 appendix / supplement / blocked surface 的 reader-facing 指向做更明确的界定
   - 删除或压缩会误导为“submission-ready 已完成”的句子
6. 更新 `docs/paper_notes/README.md` 与 `docs/paper_materials/README.md`，把 `T86` 四份新台账与 note 对应注释链登记清楚。
7. 如本地 LaTeX 工具链可用，则完成一次受控编译刷新；如不可用，必须如实记录，不得伪造 compile 结论。

## Allowed Files

Worker 只可修改或新增以下路径：

- `docs/tasks/Phase2/T86_mainline_bounded_submission_pack_assembly.md`
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
- `docs/paper_materials/paper_submission_pack_assembly_manifest.md`
- `docs/paper_materials/paper_submission_surface_route_map.md`
- `docs/paper_materials/paper_submission_exclusion_register.md`
- `docs/paper_materials/paper_submission_author_handoff.md`
- `docs/review/T86_review.md`
- `docs/for_human/T86_explanation.md`
- `docs/worker_summary/T86_worker_summary.md`

## Docs To Update

Worker 必须更新：

- `docs/paper_notes/README.md`
- `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`
- `docs/paper_materials/README.md`
- `docs/paper_materials/paper_submission_pack_assembly_manifest.md`
- `docs/paper_materials/paper_submission_surface_route_map.md`
- `docs/paper_materials/paper_submission_exclusion_register.md`
- `docs/paper_materials/paper_submission_author_handoff.md`
- `docs/review/T86_review.md`
- `docs/for_human/T86_explanation.md`
- `docs/worker_summary/T86_worker_summary.md`

如执行了本地编译，还必须同步更新对应的 note 编译产物。

## Forbidden Scope

Worker 不得：

- 修改任何治理文档 `docs/00_*` 到 `docs/08_*`
- 修改任何 `cnn_fpga/`、`physics/`、`benchmark/`、`tests/` 源码或测试
- 修改任何 `runs/`、`artifacts/`、`docs/evidence_packs/` 历史事实文件
- 运行任何 benchmark、训练、`.tflite` smoke、real-board gate/regeneration/smoke
- 新增或改写任何 stable-ID 结果资产、figure/table、caption、insertion map
- 新建第二份主 note、第二份 manuscript 草稿，或把独立 theory 分支内容拉回 main
- 以 “submission-pack assembly” 之名顺手扩成 cover letter、投稿元数据、期刊模板适配、正式投稿包、claim promotion、deployment story 升级或 hardware-ready retelling
- 把 `T86` 直接写成 “submission-ready pack 已完成”

## 强制 guardrails

以下口径在 `T86` 中必须继续保留：

1. `T24` 仍是 mainline frozen-set formal software-HIL 历史主锚点。
2. `FR6/FR7` 仍只可写成 descriptive support，而不是 causal closure。
3. `FR8/statcalib` 仍只可写成 extension lane / no-promotion / no unique clean threshold。
4. training/material 仍只可写成 canonical chain intact + one clean CPU-only bounded rerun。
5. `.tflite` 仍只可写成 isolated current-host true runtime for selected preserved artifacts。
6. real-board 仍只可写成 read-only gate / regeneration / provenance with current-host `NO_GO`。
7. 当前暂无 `Linux + FPGA` 硬件宿主，因此任何 hardware-dependent surface 都只可保留为 blocked / future-host requirement。
8. `T86` 的目标是 assembly 与 exclusion 明确化，不是 claim promotion；任何 blocked surface 仍必须保持 blocked。
9. 独立 theory 分支仍与本任务隔离；`T86` 只能处理 main 分支当前已存在 note/material 的装配问题，不得引入 branch-specific 理论扩写。

## Section 注释要求

如果 `T86` 修改了某个 section，必须在相邻位置至少保留一条 `T86` 注释，例如：

```tex
% T86-ASSEMBLY: Discussion
```

最低要求：

- 所有被 `T86` 实际修改的 section，都必须有一条 `% T86-ASSEMBLY: ...` 注释。
- `paper_submission_surface_route_map.md` 与 `paper_submission_author_handoff.md` 中列出的 touched section，必须能回链到源码中的 `% T86-ASSEMBLY: ...` 注释。

## 推荐执行顺序

1. 先阅读 `T85_review`、`paper_submission_readiness_preflight_gate.md`、`paper_submission_blocker_matrix.md` 与 `paper_residual_state_lag_sweep.md`，确认 `T85` 已经回答了什么、还没回答什么。
2. 先写 `paper_submission_exclusion_register.md`，明确哪些 surface 现在不能进 submission-facing package。
3. 再写 `paper_submission_surface_route_map.md`，把 main text / appendix / supplement / exclusion 路由固定下来。
4. 然后写 `paper_submission_pack_assembly_manifest.md`，把每个会进入 package 的 surface 绑定到具体 evidence anchor。
5. 再回写 note，只处理 route / exclusion 需要的最小装配性措辞，并为 touched sections 加 `% T86-ASSEMBLY: ...` 注释。
6. 然后写 `paper_submission_author_handoff.md`，把当前 package 的可继续编辑边界、blocked surface 与禁写口径集中给作者。
7. 更新两个 README。
8. 如本地工具链可用，执行一次 note 编译并记录 log scan 结果。
9. 最后写 `review`、`for_human` 与 `worker_summary`。

## Verification

至少完成以下验证：

1. 必须使用 allowlist-scoped diff 验证，而不是把全仓 `git diff --name-only` 直接当作 `T86` 改动清单。
2. 必须确认：
   - `T80` 的 `% T80-REOPEN` 标记仍保留；
   - `T81` 的 `% T81-CALIBRATION` 标记仍保留；
   - `T82` 的 `% T82-SUPPORT` 标记仍保留；
   - `T83` 的 `% T83-CLOSEOUT` 标记仍保留；
   - `T84` 的 `% T84-POLISH` 标记仍保留；
   - `T85` 的 `% T85-PREFLIGHT` 标记仍保留。
3. `paper_submission_pack_assembly_manifest.md` 必须至少列出：
   - `surface_id`
   - `surface_role`
   - `included_source`
   - `evidence_anchor`
   - `author_action`
4. `paper_submission_surface_route_map.md` 必须至少列出：
   - `claim_or_section`
   - `main_text_route`
   - `appendix_route`
   - `supplement_route`
   - `exclusion_note`
   - `source_anchor`
5. `paper_submission_exclusion_register.md` 必须至少列出：
   - `exclusion_id`
   - `blocked_surface`
   - `why_excluded_now`
   - `do_not_claim_wording`
   - `future_unblock_task`
6. `paper_submission_author_handoff.md` 必须明确写出仍不可升级的四类边界：
   - real-board execution / timing / resource
   - default-env / cross-host `.tflite` portability
   - full training reproducibility
   - `statcalib` mature comparator promotion
7. 如果 note 某个 section 被 `T86` 修改，源码中必须能 grep 到对应的 `% T86-ASSEMBLY: ...` 注释。
8. 如本地工具链可用并执行编译，需要记录：
   - 使用的工具链
   - 编译目标
   - 产物集合
   - `.log` 关键字扫描结果
9. 如工具链不可用，必须在 author handoff 或 worker summary 中明确写出未编译原因。

## 完成标准

只有同时满足以下条件，`T86` 才算完成：

1. `paper_submission_pack_assembly_manifest.md` 已完成。
2. `paper_submission_surface_route_map.md` 已完成。
3. `paper_submission_exclusion_register.md` 已完成。
4. `paper_submission_author_handoff.md` 已完成。
5. 被 `T86` 实际修改的 section 已全部加上 `% T86-ASSEMBLY: ...` 注释。
6. `docs/paper_notes/README.md` 与 `docs/paper_materials/README.md` 已登记 `T86` 入口。
7. `docs/review/T86_review.md` 已给出正式 review 结论。
8. `docs/for_human/T86_explanation.md` 已向作者说明：
   - 本轮为什么不是新实验而是更强的 assembly / exclusion 收口；
   - 哪些内容可以继续进入 submission-facing package；
   - 哪些内容仍必须排除；
   - 为什么即便 `T86` 成功，也不自动等于 submission-ready pack 已完成。
9. `docs/worker_summary/T86_worker_summary.md` 已总结：
   - 改了什么
   - 怎么验证
   - 剩余风险

## 交付提醒

- 这是一张比 `T85` 更偏“提交前材料装配纪律”的 docs-only 任务，因为它不仅要做整理，还要把 inclusion / exclusion route 固定下来。
- 但它仍然不是“恢复无界 full-manuscript 扩写”，更不是“直接投稿”。
- `T86` 的成功标准不是“把论文做成最终成稿”，而是“把当前 mainline note/material 能如何被组装成 submission-facing package、哪些仍必须显式排除、作者下一步还能怎么做”全部变成可审计事实。
