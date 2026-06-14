# T83：主线 note 全文一致性收口与 manuscript closeout gate

## 状态

- 由 Captain 在 `2026-06-13` 基于 `T82` 的 `PASS` closeout 提出
- 当前阶段：`Phase 2: Controlled Development`
- 决策状态：`Go`
- 任务类型：高强度 docs-only 主线全文一致性收口 + closeout gate 任务

## 为什么现在做这个任务

`T80` 已完成 ready narrative / result-facing sections 的有界 prose reopen，`T81` 已完成 `Summary of Contributions` 与三章 methods 的受控校准，`T82` 又把 supporting-boundary 路由压成了：

- `main text`
- `appendix`
- `supplement`
- `blocked`

四层 manuscript-facing closeout 结构。

到这一刻为止，主线材料的主要缺口已不再是“缺少某一块 prose”或“缺少某一份 supporting material”。更真实的下一瓶颈变成：

1. 当前 note 是否已经形成一份**全文范围内自洽**、不会互相打架的 manuscript-facing 草稿；
2. 各 section 是否都能回链到当前 strongest supported truth，而不是仍保留零散的历史措辞漂移；
3. 如果现在仍然不能写成 full-manuscript closeout，那么剩下的 blocker 到底是什么，应该压成哪一张后续 bounded task，而不是继续模糊推进。

因此，`T83` 的目标不是直接宣布论文已 ready，也不是恢复无界 full-manuscript 扩写，而是做一张更强的主线任务：

- 对当前 note 做**全文一致性 sweep**
- 对必要的 section 做**受控 wording 收口**
- 产出一份**section-to-evidence crosswalk**
- 再给出一份**明确的 closeout gate / blocker register**

这样后续 Captain 才能基于一份更硬的全文证据对齐结果，决定是否只开一张 final-polish 任务，还是继续保持 `NO_GO_FULL_MANUSCRIPT_CLOSEOUT`。

## 前置条件

只有以下条件全部满足时，`T83` 才可执行：

- `T82` 已完成并通过 Captain `PASS`
- 以下文件已存在：
  - `docs/review/T82_review.md`
  - `docs/paper_materials/paper_supporting_material_closeout_pack.md`
  - `docs/paper_materials/paper_manuscript_closeout_readiness_matrix.md`
  - `docs/paper_materials/paper_methods_and_contribution_calibration_manifest.md`
  - `docs/paper_materials/paper_bounded_prose_reopen_manifest.md`
  - `docs/paper_materials/paper_note_alignment_and_layout_closeout.md`
  - `docs/paper_materials/paper_claim_evidence_ledger.md`
  - `docs/paper_materials/paper_claim_risk_table.md`
  - `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`

如果这些前提不满足，Worker 不得在 `T83` 中补造上游材料，而必须如实汇报 blocker。

## 目标

在不运行任何新实验、不修改任何源码/配置/历史结果、不触碰治理文档、不直接宣布 full-manuscript closeout 的前提下，完成以下工作：

1. 对当前 `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex` 做一次**全文 section-by-section consistency sweep**，至少覆盖：
   - `Title`
   - `Abstract`
   - `Summary of Contributions`
   - `Introduction`
   - `Relationship to Existing Work`
   - `Brief Review of the GKP Code`
   - `Noise and Drift Model`
   - `Model Architecture`
   - `Experimental Setup`
   - `Numerical Results`
   - `Discussion`
   - `Conclusion`
2. 在不引入新 claim 的前提下，只对发现存在 wording drift、层级冲突、unsupported intensifier、blocked surface 模糊化的 section 做受控修订。
3. 新增一份 `paper_fullnote_consistency_crosswalk.md`，逐 section 记录：
   - 当前 section 名称
   - strongest supported truth
   - primary evidence anchors
   - forbidden retelling
   - 若仍需补写，允许的最小后续动作
4. 新增一份 `paper_closeout_gate_and_blocker_register.md`，给出本轮唯一 gate verdict，且 verdict 只能在下列集合中选择其一：
   - `GO_FOR_BOUNDED_FINAL_POLISH_ONLY`
   - `NO_GO_FULL_MANUSCRIPT_CLOSEOUT_BLOCKERS_EXPLICIT`
5. 在 `paper_closeout_gate_and_blocker_register.md` 中列出所有剩余 blocker，并至少标注：
   - blocker_id
   - blocker_type
   - 受影响 section / surface
   - 为什么当前证据还不够
   - 如果要继续推进，应开哪一类 bounded task
6. 更新 `docs/paper_notes/README.md` 与 `docs/paper_materials/README.md`，把 `T83` 的全文一致性收口与 closeout gate 入口登记清楚。
7. 如果本地 LaTeX 工具链可用，则完成一次受控编译刷新；如果不可用，必须如实记录，不能伪造 compile 结论。

## Allowed Files

Worker 只可修改或新增以下路径：

- `docs/tasks/Phase2/T83_mainline_note_full_consistency_sweep_and_closeout_gate.md`
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
- `docs/paper_materials/paper_fullnote_consistency_crosswalk.md`
- `docs/paper_materials/paper_closeout_gate_and_blocker_register.md`
- `docs/review/T83_review.md`
- `docs/for_human/T83_explanation.md`
- `docs/worker_summary/T83_worker_summary.md`

## Docs To Update

Worker 必须更新：

- `docs/paper_notes/README.md`
- `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`
- `docs/paper_materials/README.md`
- `docs/paper_materials/paper_fullnote_consistency_crosswalk.md`
- `docs/paper_materials/paper_closeout_gate_and_blocker_register.md`
- `docs/review/T83_review.md`
- `docs/for_human/T83_explanation.md`
- `docs/worker_summary/T83_worker_summary.md`

如果执行了本地编译，还必须同步更新对应的 note 编译产物。

## Forbidden Scope

Worker 不得：

- 修改任何治理文档 `docs/00_*` 到 `docs/08_*`
- 修改任何 `cnn_fpga/`、`physics/`、`benchmark/`、`tests/` 源码或测试
- 修改任何 `runs/`、`artifacts/`、`docs/evidence_packs/` 历史事实文件
- 运行任何 benchmark、训练、`.tflite` smoke、real-board gate/regeneration/smoke
- 新增或改写任何 figure/table/stable-ID/caption/insertion map
- 新建第二个 note 主文件、第二份 manuscript 草稿或 theory-branch 大范围重写
- 以“全文一致性收口”之名扩成 full submission pack、投稿信、封面页、bib 大规模重构、真实硬件结论升级或 deployment 故事升级
- 把 `T83` 直接写成“full-manuscript closeout 已完成”

## 强制 guardrails

以下口径在 `T83` 中必须继续保留：

1. `T24` 仍是 mainline frozen-set formal software-HIL 历史主锚点。
2. `FR6/FR7` 仍只能写成 descriptive support，而不是 causal closure。
3. `FR8/statcalib` 仍只能写成 extension lane / no-promotion / no unique clean threshold。
4. training/material 仍只能写成 canonical chain intact + one clean CPU-only bounded rerun。
5. `.tflite` 仍只能写成 isolated current-host true runtime for selected preserved artifacts。
6. real-board 仍只能写成 read-only gate / regeneration / provenance with current-host `NO_GO`。
7. 当前暂无 `Linux + FPGA` 硬件宿主，因此任何 hardware-dependent surface 都只能保留为 blocked / future-host 需求，不能回述成已有 execution path。
8. 如果本轮 gate 最终不是 `GO_FOR_BOUNDED_FINAL_POLISH_ONLY`，则必须把 `NO_GO_FULL_MANUSCRIPT_CLOSEOUT_BLOCKERS_EXPLICIT` 写清，而不能用模糊措辞代替。

## Section 注释要求

如果 `T83` 修改了某个 section，必须在相邻位置至少保留一条 `T83` 注释，例如：

```tex
% T83-CLOSEOUT: Abstract
```

最低要求：

- 所有被 `T83` 实际修改的 section，都必须有一条 `% T83-CLOSEOUT: ...` 注释。
- `paper_fullnote_consistency_crosswalk.md` 中列出的 `touched_sections` 必须与源码中的 `% T83-CLOSEOUT: ...` 注释一致。

## 推荐执行顺序

1. 先阅读 `T80`、`T81`、`T82` 的 manifest / closeout pack / readiness matrix / README 入口，明确哪些 section 已经收口、哪些 surface 仍 blocked。
2. 做全文 section-by-section 审计，先列出：
   - 已自洽 section
   - 仍有 wording drift 的 section
   - 仍有 blocked surface 模糊化风险的 section
3. 先写 `paper_fullnote_consistency_crosswalk.md`。
4. 再对 note 做受控修订，并为被修改的 section 加 `% T83-CLOSEOUT: ...` 注释。
5. 然后写 `paper_closeout_gate_and_blocker_register.md`，给出唯一 gate verdict。
6. 更新两个 README。
7. 如本地工具链可用，执行一次 note 编译并记录 log scan 结果。
8. 最后写 `review`、`for_human` 与 `worker_summary`。

## Verification

至少完成以下验证：

1. `git diff --name-only` 中属于 `T83` 的路径必须全部落在 `Allowed Files` 内。
2. 必须确认：
   - `T80` 的 `% T80-REOPEN` 标记仍保留；
   - `T81` 的 `% T81-CALIBRATION` 标记仍保留；
   - `T82` 的 `% T82-SUPPORT` 标记仍保留。
3. `paper_fullnote_consistency_crosswalk.md` 必须至少列出：
   - section
   - strongest supported truth
   - primary evidence anchors
   - forbidden retelling
   - next bounded action
4. `paper_closeout_gate_and_blocker_register.md` 必须至少列出：
   - gate verdict
   - blocker_id
   - blocker_type
   - affected section / surface
   - next bounded task type
5. 若 note 有 section 被 `T83` 修改，源码中必须能 grep 到对应的 `% T83-CLOSEOUT: ...` 注释。
6. 如果本地工具链可用并执行编译，需要记录：
   - 使用的工具链
   - 编译目标
   - 产物集合
   - `.log` 关键字扫描结果
7. 如果工具链不可用，必须在 `paper_closeout_gate_and_blocker_register.md` 或 `worker_summary` 中明确写出未编译原因。

## 完成标准

只有同时满足以下条件，`T83` 才算完成：

1. `paper_fullnote_consistency_crosswalk.md` 已完成。
2. `paper_closeout_gate_and_blocker_register.md` 已完成，且只给出一个 gate verdict。
3. 被 `T83` 实际修改的 section 已全部加上 `% T83-CLOSEOUT: ...` 注释。
4. `docs/paper_notes/README.md` 与 `docs/paper_materials/README.md` 已登记 `T83` 入口。
5. `docs/review/T83_review.md` 已给出正式 review 结论。
6. `docs/for_human/T83_explanation.md` 已向作者说明：
   - 本轮为什么比 `T82` 更强；
   - 哪些 section 真正被全文一致性 sweep 覆盖；
   - 为什么这仍然不自动等于 full-manuscript closeout。
7. `docs/worker_summary/T83_worker_summary.md` 已总结：
   - 改了什么
   - 怎么验证
   - 还剩哪些 blocker

## 交付提醒

- 这是一张比 `T82` 更强的主线任务，因为它允许做**全文级别**的一致性 sweep，并要求给出一个真正的 closeout gate。
- 但它仍然不是“恢复无界 full-manuscript 扩写”。
- `T83` 的成功标准不是“把论文写得更像成稿”，而是“让全文当前能写到哪、不能写到哪、为什么不能写、下一步该开什么 bounded task”全部变成可审计事实。
