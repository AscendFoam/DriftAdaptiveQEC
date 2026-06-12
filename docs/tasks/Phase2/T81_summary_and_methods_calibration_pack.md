# T81：Summary of Contributions 与 methods-only calibration pack

## 状态

- 由 Captain 在 `2026-06-12` 基于 `T80` 的 `PASS` closeout 提出
- 当前阶段：`Phase 2: Controlled Development`
- 决策状态：`Go`
- 任务类型：高强度 docs-only 主线方法章/贡献段校准任务

## 为什么现在做这个任务

`T80` 已经完成并通过 review `PASS`。它把当前 note 中最需要先行收口的 8 个 ready narrative / result-facing sections 重写到了与当前 evidence stack 一致的口径，但也明确保留了 4 块未动区域：

1. `Summary of Contributions`
2. `Brief Review of the GKP Code`
3. `Noise and Drift Model`
4. `Model Architecture`

这些区域继续保持 untouched 是 `T80` 的正确边界，而不是缺陷。但在 `T80` 已完成之后，它们也成为当前 mainline note 距离“更完整的 manuscript-facing 校准状态”之间最明显、最集中的剩余缺口。

因此，`T81` 的目标不是恢复 full-manuscript reopen，而是在仍然严格 docs-only、mainline-only 的条件下，把：

1. `Summary of Contributions` 的贡献口径与 `T80` 后的正文、主结果锚点和 supporting-boundary 口径对齐；
2. 三章 methods 的理论/工程叙述校准到当前最强可支持事实；
3. `statcalib`、`.tflite`、real-board、training/material、mechanism evidence 的边界继续锁在当前 strongest supported truth 上；
4. 为后续是否进入更大范围 manuscript closeout 提供一份更强的 methods/contribution calibration manifest。

## 前置条件

只有在以下条件都满足时，`T81` 才可执行：

- `T80` 已完成并通过 Captain `PASS`
- 以下文件已存在：
  - `docs/review/T80_review.md`
  - `docs/paper_materials/paper_bounded_prose_reopen_manifest.md`
  - `docs/paper_materials/paper_reopen_gate_and_prose_readiness_review.md`
  - `docs/paper_materials/paper_reopen_gap_matrix.md`
  - `docs/paper_materials/paper_claim_evidence_ledger.md`
  - `docs/paper_materials/paper_claim_risk_table.md`
  - `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`

如果这些前提不满足，Worker 不得在 `T81` 中补造上游材料，而必须如实报告 blocker。

## 目标

在不运行任何新实验、不修改任何源码/配置/历史结果、不触碰治理文档、不恢复 full-manuscript reopen 的前提下，完成以下七件事：

1. 只重写 note 中以下 4 个 target sections：
   - `Summary of Contributions`
   - `Brief Review of the GKP Code`
   - `Noise and Drift Model`
   - `Model Architecture`
2. 让 `Summary of Contributions` 与当前正文口径一致：
   - 不把 `T80` 写成 full-manuscript reopen
   - 不把 `FR8/statcalib` 写成 promoted comparator
   - 不把 `.tflite` 写成 default-env / deployment closure
   - 不把 real-board 写成 execution success
3. 让三章 methods 与当前 strongest supported truth 一致：
   - `Brief Review of the GKP Code` 只承担物理/解码背景，不偷带新 claim
   - `Noise and Drift Model` 明确是 effective model，而不是 full circuit-level closure
   - `Model Architecture` 明确区分 mainline teacher-anchored path、extension-lane `statcalib`、以及 deployment-boundary 仍未闭环
4. 在 note 源码中对每个被重写的 target section 增加局部 `T81` 注释标记，便于后续 scope 审计。
5. 产出一份 methods/contribution calibration manifest，逐节记录：
   - 改了哪些 section
   - 每节绑定了哪些 evidence anchors
   - 每节必须保留哪些 non-claims / guardrails
   - 哪些已由 `T80` 重写的 section 在本轮保持 untouched
6. 更新 `docs/paper_notes/README.md` 与 `docs/paper_materials/README.md`，把 `T81` 的方法章/贡献段校准入口登记清楚。
7. 在本地 LaTeX 工具链可用时完成一次受控编译刷新；若不可用，必须如实记录缺失，而不是伪造 compile 结论。

## Allowed Files

Worker 只可修改或新增以下路径：

- `docs/tasks/Phase2/T81_summary_and_methods_calibration_pack.md`
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
- `docs/paper_materials/paper_methods_and_contribution_calibration_manifest.md`
- `docs/review/T81_review.md`
- `docs/for_human/T81_explanation.md`
- `docs/worker_summary/T81_worker_summary.md`

## Docs To Update

Worker 必须更新：

- `docs/paper_notes/README.md`
- `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`
- `docs/paper_materials/README.md`
- `docs/paper_materials/paper_methods_and_contribution_calibration_manifest.md`
- `docs/review/T81_review.md`
- `docs/for_human/T81_explanation.md`
- `docs/worker_summary/T81_worker_summary.md`

如执行了本地编译，还必须同步更新对应的 note 编译产物。

## Forbidden Scope

Worker 不得：

- 修改任何治理文档 `docs/00_*` 到 `docs/08_*`
- 修改任何 `cnn_fpga/`、`physics/`、`benchmark/`、`tests/` 源码或测试
- 修改任何 `runs/`、`artifacts/`、`docs/evidence_packs/` 历史事实文件
- 新增任何 figure/table 资产，或改写既有 stable-ID / caption / insertion map
- 运行任何 benchmark、训练、`.tflite` smoke、real-board gate/regeneration/smoke
- 修改 theory 分支的大范围材料，或把 theory-only rewrite 混进 main 分支当前 note
- 借 methods calibration 之名重写 `Title`、`Abstract`、`Introduction`、`Relationship to Existing Work`、`Experimental Setup`、`Numerical Results`、`Discussion`、`Conclusion`
- 把 `T81` 扩成 full-manuscript reopen、投稿包总装、benchmark/claim 升级或 deployment 故事补写

## 强制 guardrails

以下口径在 `T81` 中必须继续保留，不能被弱化或覆盖：

1. `T24` 仍是 mainline frozen-set formal software-HIL 主锚点。
2. `FR6/FR7` 仍然只能写成 descriptive support，而不是 causal closure。
3. `FR8/statcalib` 仍然只能写成 extension lane / no-promotion / no unique clean threshold。
4. training/material 仍然只能写成 canonical chain intact + one clean CPU-only bounded rerun。
5. `.tflite` 仍然只能写成 isolated current-host true runtime for selected preserved artifacts。
6. real-board 仍然只能写成 read-only gate / regeneration / provenance with current-host `NO_GO`。
7. methods 章节可以更清晰，但不能把理论背景、effective model 或 runtime contract 偷写成“已完成硬件验证”。

## Section 注释要求

每个被 `T81` 重写的 target section，至少要在 section 入口附近保留一条 `T81` 注释，例如：

```tex
% T81-CALIBRATION: Summary of Contributions
```

最低要求是以下 4 条 section-level 标记：

1. `% T81-CALIBRATION: Summary of Contributions`
2. `% T81-CALIBRATION: Brief Review of the GKP Code`
3. `% T81-CALIBRATION: Noise and Drift Model`
4. `% T81-CALIBRATION: Model Architecture`

## 推荐执行顺序

1. 先阅读 `T80` manifest、`T79` gate/gap 文档、claim/evidence ledger 与 risk table，明确当前 strongest supported truth。
2. 先重写 `Summary of Contributions`，确保贡献点与 `T80` 后正文口径一致。
3. 再依次重写三章 methods：
   - `Brief Review of the GKP Code`
   - `Noise and Drift Model`
   - `Model Architecture`
4. 写出 methods/contribution calibration manifest。
5. 更新两个 README。
6. 如本地工具链可用，执行一次 note 编译并记录 compile/log scan 结果。
7. 写 `review`、`for_human` 与 `worker_summary`。

## Verification

至少完成以下验证：

1. `git diff --name-only` 中属于 `T81` 的路径必须全部落在 `Allowed Files` 内。
2. `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex` 中必须能 grep 到 4 条 `% T81-CALIBRATION` section-level 标记。
3. 必须确认 `T80` 的 8 个 `% T80-REOPEN` section 标记仍在，且本轮没有把 `T81` 扩成对这些 section 的再次大改。
4. `paper_methods_and_contribution_calibration_manifest.md` 必须列出：
   - 4 个 changed sections
   - 每节 evidence anchors
   - 每节 non-claims / guardrails
   - compile 状态
5. 若本地工具链可用并执行编译，需至少记录：
   - 使用的工具链
   - 编译目标
   - 产物集合
   - `.log` 关键字扫描结果
6. 若本地工具链不可用，必须在 manifest 和 worker summary 中明确写出未编译原因。

## 完成标准

只有同时满足以下条件，`T81` 才算完成：

1. `Summary of Contributions` 与三章 methods 已完成受控校准。
2. note 源码中已出现 4 条 `% T81-CALIBRATION` 标记。
3. `docs/paper_notes/README.md` 与 `docs/paper_materials/README.md` 已登记 `T81` 入口。
4. `docs/paper_materials/paper_methods_and_contribution_calibration_manifest.md` 已完整记录本轮 scope、evidence anchors、guardrails 与 compile 状态。
5. `docs/review/T81_review.md` 已写出正式 review 结论。
6. `docs/for_human/T81_explanation.md` 已向作者说明本轮 methods/contribution calibration 改了什么、没改什么、为什么仍不是 full-manuscript reopen。
7. `docs/worker_summary/T81_worker_summary.md` 已总结：
   - 改了什么
   - 怎么验证
   - 剩余风险

## 交付提醒

- 这是一张比 `T80` 更强的 docs-only mainline 任务，但它仍然不是 full-manuscript reopen。
- 如果某段 methods prose 只有在升级 benchmark / deployment / board / comparator 证据后才能成立，那么该段就不应写进 `T81`。
- `T81` 的成功标准不是“写得更像论文”，而是“把剩余 4 个核心 untouched sections 校准到当前 strongest supported truth，并保持可审计、可回链、可编译”。
