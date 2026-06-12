# T80：主线校准段落的 bounded prose reopen

## 状态

- 由 Captain 在 `2026-06-12` 基于 `T79` 的 `PASS` closeout 提出
- 当前阶段：`Phase 2: Controlled Development`
- 决策状态：`Go`
- 任务类型：高强度 docs-only 主线 prose reopen 任务

## 为什么现在做这个任务

`T79` 已经完成 reopen gate，并给出唯一 gate verdict：

> `GO_FOR_BOUNDED_PROSE_REOPEN`

这代表当前主线材料栈已经足够支持一轮**有界** prose reopen，但并不代表：

1. full-manuscript reopen 已获批准；
2. 方法章已经 ready；
3. deployment / `.tflite` / real-board / `statcalib` 边界可以升级；
4. 现在可以跳过 evidence guardrail 直接自由扩写。

因此，`T80` 的目标不是“把整篇稿子都重写一遍”，而是在 `T79` 已判定 ready 的区域内，做一轮强约束、可编译、可追溯、可审计的主线 prose reopen：

1. 只重写当前已经 ready 的 narrative / result-facing sections；
2. 把 `T24` 主结果、`FR6/FR7` 描述性支撑、`FR8` extension-lane no-promotion、training/material、`.tflite`、real-board 的安全口径统一落到 note；
3. 保留 methods chapters 不动；
4. 产出一份 section-level prose reopen manifest，明确每个被改 section 的证据锚点与禁止外推边界；
5. 在本地工具可用时完成一次受控 LaTeX 编译刷新。

## 前置条件

只有在以下条件都满足时，`T80` 才可执行：

- `T79` 已完成并通过 Captain `PASS`
- 以下文件已存在：
  - `docs/review/T79_review.md`
  - `docs/paper_materials/paper_reopen_gate_and_prose_readiness_review.md`
  - `docs/paper_materials/paper_reopen_gap_matrix.md`
  - `docs/paper_materials/paper_claim_evidence_ledger.md`
  - `docs/paper_materials/paper_claim_risk_table.md`
  - `docs/paper_materials/paper_results_section_assembly_pack.md`
  - `docs/paper_materials/paper_results_callout_sheet.md`
  - `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`

如果这些前提不满足，Worker 不得在 `T80` 中重建上游材料，而必须如实报告 blocker。

## 目标

在不运行任何新实验、不修改任何源码/配置/历史结果、不触碰治理文档、不扩写方法章的前提下，完成以下六件事：

1. 在现有 note 中只重写以下 8 个 ready sections：
   - `Title`
   - `Abstract`
   - `Introduction`
   - `Related Work / positioning`
   - `Experimental Setup`
   - `Numerical Results`
   - `Discussion`
   - `Conclusion`
2. 保持所有主线边界口径一致：
   - `T24` = frozen-set formal software revalidation main anchor
   - `FR6/FR7` = descriptive support only
   - `FR8/statcalib` = extension lane / no-promotion / no unique clean threshold
   - training/material = canonical chain + one clean CPU-only bounded rerun
   - `.tflite` = isolated current-host true runtime only
   - real-board = read-only gate / regeneration / provenance with current-host `NO_GO`
3. 在 note 中对每个被重写 section 增加局部 `T80` 注释标记，便于后续 scope 审计。
4. 产出一份 prose reopen manifest，逐节记录：
   - 改了哪些 section
   - 每节绑定了哪些 evidence anchors
   - 每节必须保留哪些 guardrail
5. 更新 `docs/paper_notes/README.md` 与 `docs/paper_materials/README.md`，把 `T80` 的 prose reopen 入口登记清楚。
6. 在本地 LaTeX 工具链可用时完成一次受控编译刷新；若不可用，必须如实记录缺失而不是伪造编译结论。

## Allowed Files

Worker 只可修改或新增以下路径：

- `docs/tasks/Phase2/T80_mainline_calibrated_sections_bounded_prose_reopen.md`
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
- `docs/paper_materials/paper_bounded_prose_reopen_manifest.md`
- `docs/review/T80_review.md`
- `docs/for_human/T80_explanation.md`
- `docs/worker_summary/T80_worker_summary.md`

## Docs To Update

Worker 必须更新：

- `docs/paper_notes/README.md`
- `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`
- `docs/paper_materials/README.md`
- `docs/paper_materials/paper_bounded_prose_reopen_manifest.md`
- `docs/review/T80_review.md`
- `docs/for_human/T80_explanation.md`
- `docs/worker_summary/T80_worker_summary.md`

如执行了本地编译，还必须同步更新对应的 note 编译产物。

## Forbidden Scope

Worker 不得：

- 修改任何治理文档 `docs/00_*` 到 `docs/08_*`
- 修改任何 `cnn_fpga/`、`physics/`、`benchmark/`、`tests/` 源码或测试
- 修改任何 `runs/`、`artifacts/`、`docs/evidence_packs/` 历史事实文件
- 新增任何 figure/table 资产，或改写既有 stable-ID / caption / insertion map
- 运行任何 benchmark、训练、`.tflite` smoke、real-board gate/regeneration/smoke
- 修改 theory 分支的大范围材料，或把 theory-only rewrite 混进 main 分支当前 note
- 修改以下方法章正文：
  - `Brief Review of the GKP Code`
  - `Noise and Drift Model`
  - `Model Architecture`
- 把 bounded prose reopen 扩成 full-manuscript reopen
- 静默提升任何证据等级，尤其不得把：
  - `T24` 写成 paper-grade expanded benchmark
  - `T48` 写成 default-env / deployment closure
  - `T49/T71/T72` 写成 real-board execution success
  - `T64`-`T70` 写成 mature `statcalib` comparator promotion
- 借 prose reopen 之名新增新结论、新实验、新机制解释或新部署叙事

## 必须复用的输入

Worker 必须复用以下输入，而不是重写历史事实：

- gate / 边界输入：
  - `docs/review/T79_review.md`
  - `docs/paper_materials/paper_reopen_gate_and_prose_readiness_review.md`
  - `docs/paper_materials/paper_reopen_gap_matrix.md`
  - `docs/08_risks_and_open_questions.md`
- note / paper-facing 输入：
  - `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`
  - `docs/paper_notes/README.md`
  - `docs/paper_materials/README.md`
  - `docs/paper_materials/paper_claim_evidence_ledger.md`
  - `docs/paper_materials/paper_claim_risk_table.md`
  - `docs/paper_materials/paper_maintext_results_authoring_pack.md`
  - `docs/paper_materials/paper_results_section_assembly_pack.md`
  - `docs/paper_materials/paper_results_callout_sheet.md`
  - `docs/paper_materials/paper_note_results_sync_manifest.md`
  - `docs/paper_materials/paper_note_alignment_and_layout_closeout.md`
- review 输入：
  - `docs/review/T74_review.md`
  - `docs/review/T75_review.md`
  - `docs/review/T76_review.md`
  - `docs/review/T77_review.md`
  - `docs/review/T78_review.md`
  - `docs/review/T79_review.md`

## 固定 section 边界

### 允许重写的 section

1. `Title`
2. `Abstract`
3. `Introduction`
4. `Related Work / positioning`
5. `Experimental Setup`
6. `Numerical Results`
7. `Discussion`
8. `Conclusion`

### 明确排除的 section

1. `Brief Review of the GKP Code`
2. `Noise and Drift Model`
3. `Model Architecture`

如果某个段落横跨允许区和排除区，Worker 必须保守处理：宁可少改，也不要把 methods calibration 偷偷带进来。

## prose reopen guardrail

1. 只允许做“组织、压缩、校准、澄清、统一口径”，不允许写出新实验故事。
2. 结果叙事必须继续锚定 `T24` 为主表主锚点。
3. `FR6/FR7` 只能写成支持性、描述性材料，不能写成 causal closure 或 teacher necessity。
4. `FR8/statcalib` 只能写成 extension lane / no-promotion / no unique clean threshold，不得提升为主线 comparator。
5. `.tflite` 只能写成 isolated current-host true runtime supporting boundary。
6. real-board 只能写成 read-only gate / regeneration / provenance with current-host `NO_GO`。
7. training/material 只能写成 canonical chain intact + one clean CPU-only bounded rerun。
8. 若某段 prose 需要越过现有 claim/evidence ledger 才成立，该段就不应该被写进 `T80`。

## 注释与 manifest 要求

### A. note 内部注释

每个被 `T80` 重写的 section，至少要在 section 入口附近保留一条 `T80` 注释，例如：

```tex
% T80-REOPEN: Abstract
```

注释只用于 scope 审计，不应变成大段解释。

### B. `paper_bounded_prose_reopen_manifest.md`

至少应包含：

1. `Scope Verdict`
   - 明确本轮只覆盖哪 8 个 section
2. `Section Change Ledger`
   - `section`
   - `changed_or_not`
   - `evidence_anchors`
   - `guardrails_preserved`
3. `Boundary Checklist`
   - `T24` main anchor 是否保留
   - `FR6/FR7` descriptive-only 是否保留
   - `FR8` extension-lane no-promotion 是否保留
   - training/material、`.tflite`、real-board supporting boundary 是否保留
4. `Compile Status`
   - `compiled`
   - `not_compiled_toolchain_unavailable`
   - `compiled_with_nonblocking_warnings`
5. `Out-of-Scope Sections Left Untouched`

## Verification

Worker 至少要完成以下验证：

1. `git diff --name-only` 范围核查：确认变更只落在 Allowed Files。
2. section-scope 核查：
   - `paper_bounded_prose_reopen_manifest.md` 中记录的 changed sections 必须全部属于允许的 8 个 section
   - methods chapters 必须保持 untouched
3. 边界口径核查：
   - 不得把 `.tflite` 写成 default-env / deployment closure
   - 不得把 real-board 写成 execution success
   - 不得把 `statcalib` 写成 promoted comparator
   - 不得把 `FR6/FR7` 写成 causal closure
4. LaTeX 编译核查：
   - 如果本地工具链可用，至少编译一次并记录结果
   - 如果不可用，必须在 manifest / worker summary / review 中明确写出原因
5. review 文件中必须明确写出：
   - 是否有 blocker
   - section scope 是否守住
   - compile 状态是什么
   - verdict 是什么

## 完成标准

只有同时满足以下条件，`T80` 才算完成：

1. `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex` 已完成有界 prose reopen，且只覆盖允许的 8 个 section
2. `docs/paper_materials/paper_bounded_prose_reopen_manifest.md` 已生成，并逐节回指 evidence anchors 与 guardrails
3. `docs/paper_notes/README.md` 与 `docs/paper_materials/README.md` 已登记 `T80` 入口
4. 若本地工具链可用，note 已完成一次受控编译刷新；若不可用，缺口已如实记录
5. `docs/review/T80_review.md` 已写出正式 review 结论
6. `docs/for_human/T80_explanation.md` 已向作者说明本轮 prose reopen 改了什么、没改什么、为什么仍不是 full-manuscript reopen
7. `docs/worker_summary/T80_worker_summary.md` 已总结：
   - 改了什么
   - 怎么验证
   - 剩余风险
8. 全程未越界到方法章扩写、实验执行、治理文档修改或证据等级升级

## 交付提醒

本任务产出必须优先使用中文。

如果 Worker 发现 current note 的某个 ready section 实际上仍无法在不越界的前提下重写，也必须如实写进 manifest / review，而不是偷偷扩大 scope 去修方法章或补实验。
