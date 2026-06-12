# T82：论文 supporting-material 收口与 appendix/supplement 边界整合包

## 状态
- 由 Captain 在 `2026-06-12` 基于 `T81` 的 `PASS` closeout 提出
- 当前阶段：`Phase 2: Controlled Development`
- 决策状态：`Go`
- 任务类型：高强度 docs-only 主线 supporting-material 收口任务

## 为什么现在做这个任务

`T80` 已完成 ready narrative / result-facing sections 的 bounded prose reopen，`T81` 已完成 `Summary of Contributions` 与三章 methods 的受控校准。到这一刻为止，当前 mainline note 的主叙事与方法叙事已经基本压回到了现有 evidence stack 上。

但论文要达到更稳的可提交材料状态，仍然缺少一层单独的 supporting-material 收口：

1. `FR8/statcalib` extension lane、training/material、isolated true `.tflite` runtime、real-board `NO_GO` gate 这几条 supporting / boundary surface 目前分散在 `T50/T48/T72/T70/T74/T79/T80/T81` 等文档中。
2. 当前 note 虽然已经能安全书写这些 supporting boundary，但还没有一份“主文 / 附录 / supplement / blocked hardware surface”的统一 closeout 包。
3. 在当前暂无 `Linux + FPGA` 硬件宿主的前提下，继续优先补齐 hardware-independent supporting materials，比过早重开真板 execution 或宣称 full-manuscript closeout 更符合主线。

因此，`T82` 的目标不是恢复 full-manuscript 扩写，而是把当前已存在的 supporting-boundary 材料整合成一份更强的 manuscript-facing closeout 包，并只对 note 中少数 supporting-boundary 段落做受控收口。

## 前置条件

只有在以下条件都满足时，`T82` 才可执行：

- `T81` 已完成并通过 Captain `PASS`
- 以下文件已存在：
  - `docs/review/T81_review.md`
  - `docs/paper_materials/paper_methods_and_contribution_calibration_manifest.md`
  - `docs/paper_materials/paper_submission_material_gap_checklist.md`
  - `docs/paper_materials/paper_reopen_gap_matrix.md`
  - `docs/paper_materials/paper_claim_evidence_ledger.md`
  - `docs/paper_materials/paper_claim_risk_table.md`
  - `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`

如果这些前提不满足，Worker 不得在 `T82` 中补造上游材料，而必须如实汇报 blocker。

## 目标

在不运行任何新实验、不修改任何源码/配置/历史结果、不触碰治理文档、不恢复 full-manuscript reopen 的前提下，完成以下工作：

1. 只对当前 note 中与 supporting-boundary 直接相关的少数段落做受控校准，重点限于：
   - `Runtime, quantization, and fixed-point degradation`
   - `Embedded runtime and board-level validation`
   - `Discussion` 中 deployment/supporting-boundary 相关段落
   - `Conclusion` 中 remaining technical gap / supporting-boundary 相关段落
2. 为上述被重写的 note 段落加入局部 `T82` 注释标记，便于后续 scope 审计。
3. 新增一份 `paper_supporting_material_closeout_pack.md`，至少要把以下 surface 做统一收口：
   - `FR8/statcalib` extension lane supplement-side boundary
   - training/material reproducibility supporting boundary
   - isolated current-host true `.tflite` supporting boundary
   - real-board read-only gate / provenance with current-host `NO_GO`
   - 当前仍 blocked 的 hardware-dependent surface
4. 新增一份 `paper_manuscript_closeout_readiness_matrix.md`，把当前 manuscript-facing surface 按 `ready / support-only / blocked` 分类，并明确：
   - 对应 evidence anchors
   - 不得外推的 forbidden claims
   - 若要继续推进，下一步应开哪类 bounded task，而不是直接扩写
5. 更新 `docs/paper_notes/README.md` 与 `docs/paper_materials/README.md`，把 `T82` 的 supporting-material closeout 入口登记清楚。
6. 如果本地 LaTeX 工具链可用，则完成一次受控编译刷新；如果不可用，必须如实记录，不能伪造 compile 结论。

## Allowed Files

Worker 只可修改或新增以下路径：

- `docs/tasks/Phase2/T82_supporting_material_closeout_and_boundary_integration_pack.md`
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
- `docs/paper_materials/paper_supporting_material_closeout_pack.md`
- `docs/paper_materials/paper_manuscript_closeout_readiness_matrix.md`
- `docs/review/T82_review.md`
- `docs/for_human/T82_explanation.md`
- `docs/worker_summary/T82_worker_summary.md`

## Docs To Update

Worker 必须更新：

- `docs/paper_notes/README.md`
- `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`
- `docs/paper_materials/README.md`
- `docs/paper_materials/paper_supporting_material_closeout_pack.md`
- `docs/paper_materials/paper_manuscript_closeout_readiness_matrix.md`
- `docs/review/T82_review.md`
- `docs/for_human/T82_explanation.md`
- `docs/worker_summary/T82_worker_summary.md`

如果执行了本地编译，还必须同步更新对应的 note 编译产物。

## Forbidden Scope

Worker 不得：

- 修改任何治理文档 `docs/00_*` 到 `docs/08_*`
- 修改任何 `cnn_fpga/`、`physics/`、`benchmark/`、`tests/` 源码或测试
- 修改任何 `runs/`、`artifacts/`、`docs/evidence_packs/` 历史事实文件
- 运行任何 benchmark、训练、`.tflite` smoke、real-board gate/regeneration/smoke
- 新增任何 figure/table 资产，或重写既有 stable-ID / caption / insertion map
- 以 supporting-material closeout 之名重写 `Title`、`Abstract`、`Introduction`、三章 methods、`Experimental Setup`、主结果段落
- 把 `T82` 扩成 full-manuscript reopen、投稿包总装、deployment story 升级、real-board 成功叙事、`statcalib` promotion 或 theory 分支大范围改写

## 强制 guardrails

以下口径在 `T82` 中必须继续保留：

1. `T24` 仍是 mainline frozen-set formal software-HIL 历史主锚点。
2. `FR6/FR7` 仍只能写成 descriptive support，而不是 causal closure。
3. `FR8/statcalib` 仍只能写成 extension lane / no-promotion / no unique clean threshold。
4. training/material 仍只能写成 canonical chain intact + one clean CPU-only bounded rerun。
5. `.tflite` 仍只能写成 isolated current-host true runtime for selected preserved artifacts。
6. real-board 仍只能写成 read-only gate / regeneration / provenance with current-host `NO_GO`。
7. 当前暂无 `Linux + FPGA` 硬件宿主，因此任何 hardware-dependent surface 都只能保留为 blocked / future-host 需求，不能回述成已有 execution path。

## Section 注释要求

每个被 `T82` 收口的 note 段落，至少要在相邻位置保留一条 `T82` 注释，例如：

```tex
% T82-SUPPORT: Embedded runtime and board-level validation
```

最低要求是以下 4 条标记全部出现：

1. `% T82-SUPPORT: Runtime, quantization, and fixed-point degradation`
2. `% T82-SUPPORT: Embedded runtime and board-level validation`
3. `% T82-SUPPORT: Discussion deployment/support boundary`
4. `% T82-SUPPORT: Conclusion remaining technical gap`

## 推荐执行顺序

1. 先阅读 `T81` manifest、`paper_submission_material_gap_checklist.md`、`paper_reopen_gap_matrix.md`、claim/evidence ledger 与 risk table，明确当前 strongest supported truth。
2. 先写 `paper_supporting_material_closeout_pack.md`，把 supporting surfaces 的 main-text / appendix / supplement / blocked 路由整理清楚。
3. 再写 `paper_manuscript_closeout_readiness_matrix.md`，把 manuscript-facing closeout 状态做成一张统一矩阵。
4. 之后只回写 note 中 4 处 supporting-boundary 段落，并加上 `% T82-SUPPORT` 注释。
5. 更新两个 README。
6. 如本地工具链可用，执行一次 note 编译并记录 log scan 结果。
7. 最后写 `review`、`for_human` 与 `worker_summary`。

## Verification

至少完成以下验证：

1. `git diff --name-only` 中属于 `T82` 的路径必须全部落在 `Allowed Files` 内。
2. `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex` 中必须能 grep 到 4 条 `% T82-SUPPORT` 标记。
3. 必须确认 `T80` 的 8 条 `% T80-REOPEN` 与 `T81` 的 4 条 `% T81-CALIBRATION` 标记仍然保留。
4. `paper_supporting_material_closeout_pack.md` 必须至少列出：
   - supporting surface
   - main text / appendix / supplement / blocked placement
   - evidence anchors
   - forbidden claims
5. `paper_manuscript_closeout_readiness_matrix.md` 必须至少列出：
   - surface 或 section
   - readiness status
   - blocker type
   - next bounded action
6. 如果本地工具链可用并执行编译，需要记录：
   - 使用的工具链
   - 编译目标
   - 产物集合
   - `.log` 关键字扫描结果
7. 如果工具链不可用，必须在 closeout pack 或 worker summary 中明确写出未编译原因。

## 完成标准

只有同时满足以下条件，`T82` 才算完成：

1. supporting-material closeout pack 已完成。
2. manuscript closeout readiness matrix 已完成。
3. note 中已出现 4 条 `% T82-SUPPORT` 标记。
4. `docs/paper_notes/README.md` 与 `docs/paper_materials/README.md` 已登记 `T82` 入口。
5. `docs/review/T82_review.md` 已给出正式 review 结论。
6. `docs/for_human/T82_explanation.md` 已向作者说明本轮补了哪些 supporting materials、没补什么、为什么这仍不是 full-manuscript reopen。
7. `docs/worker_summary/T82_worker_summary.md` 已总结：
   - 改了什么
   - 怎么验证
   - 剩余风险

## 交付提醒

- 这是一张比 `T81` 更偏“材料整合与边界收口”的 docs-only 主线任务，但它仍然不是 full-manuscript closeout。
- 如果某段 supporting prose 只有在新增 benchmark、默认环境 `.tflite` portability、真板 execution 或硬件性能数据出现后才能成立，那么该段就不应写进 `T82`。
- `T82` 的成功标准不是“看起来更像最终论文”，而是“把当前已经存在的 supporting-boundary 材料压成一份可审计、可回链、可继续推进的 manuscript-facing closeout 包，同时继续诚实保留 blocked surface”。
