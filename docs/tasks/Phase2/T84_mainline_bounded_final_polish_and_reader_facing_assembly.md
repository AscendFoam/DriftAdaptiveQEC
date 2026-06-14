# T84：主线 note 有界 final polish 与读者化装配包

## 状态

- 由 Captain 在 `2026-06-14` 基于 `T83` 的 `PASS` closeout 提出
- 当前阶段：`Phase 2: Controlled Development`
- 决策状态：`Go`
- 任务类型：高强度 docs-only 主线 final-polish / reader-facing assembly 任务

## 为什么现在做这个任务

`T83` 已经完成全文一致性 sweep，并且给出了唯一 gate 结论：

- `GO_FOR_BOUNDED_FINAL_POLISH_ONLY`

这意味着当前主线材料的主要缺口已不再是：

1. 哪些 section 还互相打架；
2. 哪些 supporting surface 还没有 route；
3. 是否还需要再开一轮 closeout gate。

更真实的下一瓶颈已经变成：

1. 当前 note 中仍保留较多内部 task / provenance / governance 术语，最终读者稿需要一次**受控翻译**；
2. `Numerical Results` 与 follow-up / supporting surfaces 的组织方式仍偏内部 closeout register，需要一次**结构压缩与装配**；
3. appendix / supplement / blocked surfaces 的边界虽然已经清楚，但仍需要一次**读者化 assembly map**，避免作者在后续手工润色时把内部边界说法写乱。

因此，`T84` 的目标不是恢复无界 manuscript 扩写，也不是直接宣布 submission-ready，而是做一张更强的主线任务：

- 对当前 note 做**reader-facing final polish**
- 对内部 task / provenance 语言做**受控读者化翻译**
- 对 `Results / appendix / supplement / blocked` 路由做**装配压缩**
- 产出一套**终稿级但仍受边界约束**的 reader-facing assembly 台账

## 前置条件

只有以下条件全部满足时，`T84` 才可执行：

- `T83` 已完成并通过 Captain `PASS`
- 以下文件已存在：
  - `docs/review/T83_review.md`
  - `docs/paper_materials/paper_fullnote_consistency_crosswalk.md`
  - `docs/paper_materials/paper_closeout_gate_and_blocker_register.md`
  - `docs/paper_materials/paper_supporting_material_closeout_pack.md`
  - `docs/paper_materials/paper_manuscript_closeout_readiness_matrix.md`
  - `docs/paper_materials/paper_claim_evidence_ledger.md`
  - `docs/paper_materials/paper_claim_risk_table.md`
  - `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`

如果这些前提不满足，Worker 不得在 `T84` 中补造上游材料，而必须如实汇报 blocker。

## 目标

在不运行任何新实验、不修改任何源码/配置/历史结果、不触碰治理文档、不直接宣布 submission-ready pack 的前提下，完成以下工作：

1. 对当前 `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex` 做一次**reader-facing bounded final polish**，重点覆盖：
   - `Summary of Contributions`
   - `Experimental Setup`
   - `Numerical Results`
   - `Bounded follow-up lanes outside the accepted result layer`
   - `Discussion`
   - `Conclusion`
2. 在不引入新 claim 的前提下，只对发现存在内部 task/provenance 语汇过重、读者不友好、结构过散、层级过硬的段落做受控修订。
3. 新增一份 `paper_bounded_final_polish_change_map.md`，逐 section 记录：
   - section 名称
   - 本轮是否修改
   - 修改目标
   - 保留的 strongest supported truth
   - 明确未触碰的 boundary
4. 新增一份 `paper_reader_facing_term_translation_table.md`，至少记录：
   - internal term
   - allowed reader-facing phrasing
   - forbidden retelling
   - evidence / boundary anchor
5. 新增一份 `paper_appendix_supplement_reader_assembly_map.md`，至少记录：
   - surface / material item
   - 推荐落点（main text / appendix / supplement / blocked）
   - 当前 reader-facing 状态
   - 不可升级的 boundary
   - 若后续继续推进，应开哪类 bounded task
6. 更新 `docs/paper_notes/README.md` 与 `docs/paper_materials/README.md`，把 `T84` 的 final-polish / reader-facing assembly 入口登记清楚，并顺手修正 `paper_materials/README.md` 中仍停留在 `T74-T82` 的链路标题不一致。
7. 如果本地 LaTeX 工具链可用，则完成一次受控编译刷新；如果不可用，必须如实记录，不能伪造 compile 结论。

## Allowed Files

Worker 只可修改或新增以下路径：

- `docs/tasks/Phase2/T84_mainline_bounded_final_polish_and_reader_facing_assembly.md`
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
- `docs/paper_materials/paper_bounded_final_polish_change_map.md`
- `docs/paper_materials/paper_reader_facing_term_translation_table.md`
- `docs/paper_materials/paper_appendix_supplement_reader_assembly_map.md`
- `docs/review/T84_review.md`
- `docs/for_human/T84_explanation.md`
- `docs/worker_summary/T84_worker_summary.md`

## Docs To Update

Worker 必须更新：

- `docs/paper_notes/README.md`
- `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`
- `docs/paper_materials/README.md`
- `docs/paper_materials/paper_bounded_final_polish_change_map.md`
- `docs/paper_materials/paper_reader_facing_term_translation_table.md`
- `docs/paper_materials/paper_appendix_supplement_reader_assembly_map.md`
- `docs/review/T84_review.md`
- `docs/for_human/T84_explanation.md`
- `docs/worker_summary/T84_worker_summary.md`

如果执行了本地编译，还必须同步更新对应的 note 编译产物。

## Forbidden Scope

Worker 不得：

- 修改任何治理文档 `docs/00_*` 到 `docs/08_*`
- 修改任何 `cnn_fpga/`、`physics/`、`benchmark/`、`tests/` 源码或测试
- 修改任何 `runs/`、`artifacts/`、`docs/evidence_packs/` 历史事实文件
- 运行任何 benchmark、训练、`.tflite` smoke、real-board gate/regeneration/smoke
- 新增或改写任何 figure/table/stable-ID/caption/insertion map
- 新建第二个 note 主文件、第二份 manuscript 草稿或 theory-branch 大范围重写
- 以“final polish”之名扩成 submission pack、投稿信、封面页、参考文献大规模重构、真实硬件结论升级或 deployment 故事升级
- 把 `T84` 直接写成“submission-ready pack 已完成”

## 强制 guardrails

以下口径在 `T84` 中必须继续保留：

1. `T24` 仍是 mainline frozen-set formal software-HIL 历史主锚点。
2. `FR6/FR7` 仍只能写成 descriptive support，而不是 causal closure。
3. `FR8/statcalib` 仍只能写成 extension lane / no-promotion / no unique clean threshold。
4. training/material 仍只能写成 canonical chain intact + one clean CPU-only bounded rerun。
5. `.tflite` 仍只能写成 isolated current-host true runtime for selected preserved artifacts。
6. real-board 仍只能写成 read-only gate / regeneration / provenance with current-host `NO_GO`。
7. 当前暂无 `Linux + FPGA` 硬件宿主，因此任何 hardware-dependent surface 都只能保留为 blocked / future-host 需求，不能回述成已有 execution path。
8. `T84` 的目标是 reader-facing translation / condensation / assembly，不是 claim promotion；任何 blocked surface 仍必须保持 blocked。

## Section 注释要求

如果 `T84` 修改了某个 section，必须在相邻位置至少保留一条 `T84` 注释，例如：

```tex
% T84-POLISH: Numerical Results
```

最低要求：

- 所有被 `T84` 实际修改的 section，都必须有一条 `% T84-POLISH: ...` 注释。
- `paper_bounded_final_polish_change_map.md` 中列出的 `touched_sections` 必须与源码中的 `% T84-POLISH: ...` 注释一致。

## 推荐执行顺序

1. 先阅读 `T83` 的 crosswalk 与 gate/register，明确哪些 section 已经自洽、哪些问题只剩 reader-facing / assembly 层。
2. 先建立 `paper_reader_facing_term_translation_table.md`，明确哪些内部术语要翻、哪些 guardrail 不能翻没。
3. 再建立 `paper_appendix_supplement_reader_assembly_map.md`，把 current route 从内部 closeout 语言压成读者化装配表。
4. 然后对 note 做受控修订，并为被修改的 section 加 `% T84-POLISH: ...` 注释。
5. 再写 `paper_bounded_final_polish_change_map.md`。
6. 更新两个 README，并修正 `paper_materials/README.md` 的链路标题不一致。
7. 如本地工具链可用，执行一次 note 编译并记录 log scan 结果。
8. 最后写 `review`、`for_human` 与 `worker_summary`。

## Verification

至少完成以下验证：

1. 必须使用 allowlist-scoped diff 验证，而不是把全仓 `git diff --name-only` 直接当作 `T84` 改动清单。
2. 必须确认：
   - `T80` 的 `% T80-REOPEN` 标记仍保留；
   - `T81` 的 `% T81-CALIBRATION` 标记仍保留；
   - `T82` 的 `% T82-SUPPORT` 标记仍保留；
   - `T83` 的 `% T83-CLOSEOUT` 标记仍保留。
3. `paper_bounded_final_polish_change_map.md` 必须至少列出：
   - section
   - touched_in_t84
   - polish_goal
   - strongest_supported_truth_retained
   - untouched_boundary
4. `paper_reader_facing_term_translation_table.md` 必须至少列出：
   - internal_term
   - allowed_reader_facing_phrasing
   - forbidden_retelling
   - anchor
5. `paper_appendix_supplement_reader_assembly_map.md` 必须至少列出：
   - surface
   - recommended_destination
   - reader_facing_status
   - boundary_to_keep
   - next_bounded_action
6. 若 note 有 section 被 `T84` 修改，源码中必须能 grep 到对应的 `% T84-POLISH: ...` 注释。
7. 如果本地工具链可用并执行编译，需要记录：
   - 使用的工具链
   - 编译目标
   - 产物集合
   - `.log` 关键字扫描结果
8. 如果工具链不可用，必须在 `paper_bounded_final_polish_change_map.md` 或 `worker_summary` 中明确写出未编译原因。

## 完成标准

只有同时满足以下条件，`T84` 才算完成：

1. `paper_bounded_final_polish_change_map.md` 已完成。
2. `paper_reader_facing_term_translation_table.md` 已完成。
3. `paper_appendix_supplement_reader_assembly_map.md` 已完成。
4. 被 `T84` 实际修改的 section 已全部加上 `% T84-POLISH: ...` 注释。
5. `docs/paper_notes/README.md` 与 `docs/paper_materials/README.md` 已登记 `T84` 入口，且 `paper_materials/README.md` 的链路标题不一致已修正。
6. `docs/review/T84_review.md` 已给出正式 review 结论。
7. `docs/for_human/T84_explanation.md` 已向作者说明：
   - 本轮为什么比 `T83` 更接近读者稿；
   - 哪些 section 真正被 final polish 覆盖；
   - 为什么这仍然不自动等于 submission-ready pack。
8. `docs/worker_summary/T84_worker_summary.md` 已总结：
   - 改了什么
   - 怎么验证
   - 还剩哪些不可升级的 boundary

## 交付提醒

- 这是一张比 `T83` 更偏作者终稿层的主线任务，因为它允许做 reader-facing 语言翻译与结构装配。
- 但它仍然不是“恢复无界 manuscript 扩写”或“直接进入投稿包总装”。
- `T84` 的成功标准不是“把论文写成最终成稿”，而是“把当前可写事实翻译成更像读者稿的形式，同时不把 blocked surface 偷偷写强”。
