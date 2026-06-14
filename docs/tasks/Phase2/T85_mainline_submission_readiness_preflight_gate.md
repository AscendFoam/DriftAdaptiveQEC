# T85：主线 submission-readiness preflight gate 与残余状态滞后清扫

## 状态

- 由 Captain 于 `2026-06-14` 基于 `T84` 的 `PASS_WITH_WARNINGS` closeout 提出
- 当前阶段：`Phase 2: Controlled Development`
- 决策状态：`Go`
- 任务类型：高强度 docs-only 主线 preflight / blocker-gate / wording-sweep 任务

## 为什么现在做这个任务

`T84` 已经完成了：

- 主线 note 的 bounded reader-facing final polish
- 内部术语到 reader-facing wording 的第一轮受控翻译
- `Results / appendix / supplement` 路由的 reader-facing assembly 台账

但 `T84_review` 明确指出，当前主线 still not ready to jump directly into any submission-pack retelling，原因不是又缺实验，也不是又缺 figure 资产，而是还差一层更严格的主线收口：

1. 当前 note 是否已经达到“可以被当作 submission-facing 草稿继续推进”的最小诚实状态，需要一张单独的 preflight gate。
2. `Conclusion` 仍残留 1 处状态滞后句，把本轮已经完成的 reader-facing polish 写成“后续仍待执行”的工作；这不构成 blocker，但会轻微模糊后续对外叙事与提交判断。
3. 当前暂无 `Linux + FPGA` 硬件宿主，因此主线继续优先推进 paper-material / note / blocker-matrix，而不是重开 real-board execution。

因此，`T85` 的目标不是恢复无界 full-manuscript 扩写，也不是直接宣布 submission-ready pack，而是：

- 对当前主线 note 做一轮 **residual wording-lag sweep**
- 产出一份 **submission-readiness preflight gate**
- 产出一份 **submission blocker matrix**
- 把是否允许进入下一步 bounded submission-pack assembly 压成一张唯一 verdict

## 前置条件

只有以下条件全部满足时，`T85` 才可执行：

- `T84` 已完成并通过 Captain `PASS_WITH_WARNINGS`
- 以下文件已存在：
  - `docs/review/T84_review.md`
  - `docs/paper_materials/paper_bounded_final_polish_change_map.md`
  - `docs/paper_materials/paper_reader_facing_term_translation_table.md`
  - `docs/paper_materials/paper_appendix_supplement_reader_assembly_map.md`
  - `docs/paper_materials/paper_claim_evidence_ledger.md`
  - `docs/paper_materials/paper_claim_risk_table.md`
  - `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`

如果这些前提不满足，Worker 不得在 `T85` 中补造上游材料，而必须如实汇报 blocker。

## 目标

在不运行任何新实验、不修改任何源码/配置/历史结果、不触碰治理文档、不切入 theory 分支大范围改写、也不直接宣布 submission-ready pack 的前提下，完成以下工作：

1. 对当前 `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex` 做一轮 **residual wording-lag sweep**，重点覆盖：
   - `Summary of Contributions`
   - `Experimental Setup`
   - `Numerical Results`
   - `Discussion`
   - `Conclusion`
2. 必须处理 `T84_review` 指出的 `Conclusion` 状态滞后句：
   - `The remaining writing work is to translate these internal layers into a final reader-facing polish pass...`
   - 可以改写、压缩或删除，但不得把 blocked surface 写强。
3. 新增一份 `paper_submission_readiness_preflight_gate.md`，并且只允许给出以下二选一 verdict：
   - `GO_FOR_BOUNDED_SUBMISSION_PACK_ASSEMBLY`
   - `NO_GO_SUBMISSION_PACK_BLOCKERS_EXPLICIT`
4. 新增一份 `paper_submission_blocker_matrix.md`，至少登记：
   - blocker_id
   - blocker_type
   - affected_surface
   - why_not_ready
   - next_bounded_task
5. 新增一份 `paper_residual_state_lag_sweep.md`，至少登记：
   - location
   - stale_wording_summary
   - action_taken
   - boundary_preserved
6. 更新 `docs/paper_notes/README.md` 与 `docs/paper_materials/README.md`，把 `T85` 的 preflight / blocker-matrix / residual-sweep 入口登记清楚。
7. 如本地 LaTeX 工具链可用，则完成一次受控编译刷新；如不可用，必须如实记录，不得伪造 compile 结论。

## Allowed Files

Worker 只可修改或新增以下路径：

- `docs/tasks/Phase2/T85_mainline_submission_readiness_preflight_gate.md`
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
- `docs/paper_materials/paper_submission_readiness_preflight_gate.md`
- `docs/paper_materials/paper_submission_blocker_matrix.md`
- `docs/paper_materials/paper_residual_state_lag_sweep.md`
- `docs/review/T85_review.md`
- `docs/for_human/T85_explanation.md`
- `docs/worker_summary/T85_worker_summary.md`

## Docs To Update

Worker 必须更新：

- `docs/paper_notes/README.md`
- `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`
- `docs/paper_materials/README.md`
- `docs/paper_materials/paper_submission_readiness_preflight_gate.md`
- `docs/paper_materials/paper_submission_blocker_matrix.md`
- `docs/paper_materials/paper_residual_state_lag_sweep.md`
- `docs/review/T85_review.md`
- `docs/for_human/T85_explanation.md`
- `docs/worker_summary/T85_worker_summary.md`

如执行了本地编译，还必须同步更新对应的 note 编译产物。

## Forbidden Scope

Worker 不得：

- 修改任何治理文档 `docs/00_*` 到 `docs/08_*`
- 修改任何 `cnn_fpga/`、`physics/`、`benchmark/`、`tests/` 源码或测试
- 修改任何 `runs/`、`artifacts/`、`docs/evidence_packs/` 历史事实文件
- 运行任何 benchmark、训练、`.tflite` smoke、real-board gate/regeneration/smoke
- 新增或改写任何 stable-ID 结果资产、figure/table、caption、insertion map
- 新建第二份主 note、第二份 manuscript 草稿，或切入 theory 分支大范围重写
- 以 “submission-readiness preflight” 之名顺手扩成 submission pack、cover letter、投递清单、deployment story 升级或 hardware-ready retelling
- 把 `T85` 直接写成 “submission-ready pack 已完成”

## 强制 guardrails

以下口径在 `T85` 中必须继续保留：

1. `T24` 仍是 mainline frozen-set formal software-HIL 历史主锚点。
2. `FR6/FR7` 仍只可写成 descriptive support，而不是 causal closure。
3. `FR8/statcalib` 仍只可写成 extension lane / no-promotion / no unique clean threshold。
4. training/material 仍只可写成 canonical chain intact + one clean CPU-only bounded rerun。
5. `.tflite` 仍只可写成 isolated current-host true runtime for selected preserved artifacts。
6. real-board 仍只可写成 read-only gate / regeneration / provenance with current-host `NO_GO`。
7. 当前暂无 `Linux + FPGA` 硬件宿主，因此任何 hardware-dependent surface 都只可保留为 blocked / future-host requirement。
8. `T85` 的目标是 preflight 与 blocker 明确化，不是 claim promotion；任何 blocked surface 仍必须保持 blocked。

## Section 注释要求

如果 `T85` 修改了某个 section，必须在相邻位置至少保留一条 `T85` 注释，例如：

```tex
% T85-PREFLIGHT: Conclusion
```

最低要求：

- 所有被 `T85` 实际修改的 section，都必须有一条 `% T85-PREFLIGHT: ...` 注释。
- `paper_residual_state_lag_sweep.md` 中列出的 touched location，必须与源码中的 `% T85-PREFLIGHT: ...` 注释一致。

## 推荐执行顺序

1. 先阅读 `T84_review` 与 `T84` 的三份 material 台账，明确哪里已经收口、哪里只是 reader-facing final polish、哪里仍是 blocked surface。
2. 先写 `paper_residual_state_lag_sweep.md` 的初稿，把潜在 state-lag wording 收集出来。
3. 再写 `paper_submission_blocker_matrix.md`，明确当前如果不 ready，到底 blocked 在什么位置。
4. 然后回写 note，只处理 residual wording-lag 与 preflight 需要的最小措辞校准，并为 touched sections 加 `% T85-PREFLIGHT: ...` 注释。
5. 再写 `paper_submission_readiness_preflight_gate.md`，给出唯一 verdict。
6. 更新两个 README。
7. 如本地工具链可用，执行一次 note 编译并记录 log scan 结果。
8. 最后写 `review`、`for_human` 与 `worker_summary`。

## Verification

至少完成以下验证：

1. 必须使用 allowlist-scoped diff 验证，而不是把全仓 `git diff --name-only` 直接当作 `T85` 改动清单。
2. 必须确认：
   - `T80` 的 `% T80-REOPEN` 标记仍保留；
   - `T81` 的 `% T81-CALIBRATION` 标记仍保留；
   - `T82` 的 `% T82-SUPPORT` 标记仍保留；
   - `T83` 的 `% T83-CLOSEOUT` 标记仍保留；
   - `T84` 的 `% T84-POLISH` 标记仍保留。
3. 必须确认源码中不再出现句子：
   - `The remaining writing work is to translate these internal layers into a final reader-facing polish pass`
4. `paper_submission_readiness_preflight_gate.md` 必须只给出一个 verdict，且只能是：
   - `GO_FOR_BOUNDED_SUBMISSION_PACK_ASSEMBLY`
   - `NO_GO_SUBMISSION_PACK_BLOCKERS_EXPLICIT`
5. `paper_submission_blocker_matrix.md` 必须至少列出：
   - blocker_id
   - blocker_type
   - affected_surface
   - why_not_ready
   - next_bounded_task
6. `paper_residual_state_lag_sweep.md` 必须至少列出：
   - location
   - stale_wording_summary
   - action_taken
   - boundary_preserved
7. 如果 note 某个 section 被 `T85` 修改，源码中必须能 grep 到对应的 `% T85-PREFLIGHT: ...` 注释。
8. 如本地工具链可用并执行编译，需要记录：
   - 使用的工具链
   - 编译目标
   - 产物集合
   - `.log` 关键字扫描结果
9. 如工具链不可用，必须在 preflight gate 或 worker summary 中明确写出未编译原因。

## 完成标准

只有同时满足以下条件，`T85` 才算完成：

1. `paper_submission_readiness_preflight_gate.md` 已完成，并只给出一个 verdict。
2. `paper_submission_blocker_matrix.md` 已完成。
3. `paper_residual_state_lag_sweep.md` 已完成。
4. `Conclusion` 中由 `T84_review` 指出的 residual wording-lag 已被处理，或在 residual sweep 文档中明确保留理由。
5. 被 `T85` 实际修改的 section 已全部加上 `% T85-PREFLIGHT: ...` 注释。
6. `docs/paper_notes/README.md` 与 `docs/paper_materials/README.md` 已登记 `T85` 入口。
7. `docs/review/T85_review.md` 已给出正式 review 结论。
8. `docs/for_human/T85_explanation.md` 已向作者说明：
   - 本轮为什么不是新实验而是更强的 preflight gate；
   - 还剩哪些 blocker；
   - 为什么即便 `T85` 成功，也不自动等于 submission-ready pack 已完成。
9. `docs/worker_summary/T85_worker_summary.md` 已总结：
   - 改了什么
   - 怎么验证
   - 剩余风险

## 交付提醒

- 这是一张比 `T84` 更偏“提交前主线诚实度审查”的 docs-only 任务，因为它不仅要做 residual wording-lag 清扫，还要把是否进入下一步 submission-pack assembly 压成一张唯一 gate。
- 但它仍然不是“恢复无界 full-manuscript 扩写”，更不是“直接提交”。
- `T85` 的成功标准不是“把论文写成最终成稿”，而是“把当前 note 能写到哪、不能写到哪、为什么还不能写、下一步是否值得开 bounded submission-pack assembly task”全部变成可审计事实。
