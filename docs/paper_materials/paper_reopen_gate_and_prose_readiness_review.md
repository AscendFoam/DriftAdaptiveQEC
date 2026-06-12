# T79 Paper Reopen Gate And Prose Readiness Review

## Gate Verdict

`GO_FOR_BOUNDED_PROSE_REOPEN`

## Strongest Supported Truth

1. 当前最强主结果仍是 `T24` 的锁定四场景、五模式、`paired_seeds + repeats=2` formal software revalidation，且边界仍是 `mock-backed software-HIL only`。
2. `FR6/FR7` 已经为主结果提供了 paper-facing 的描述性支撑层：六 seed 机制图与 feature/teacher ablation 表都可复用，但机制解释仍只能写成 descriptive，不可写成 causal closure 或 teacher necessity。
3. `FR8/statcalib` 已有一条单独标记的 extension-lane closure，并由 `T70` 明确给出 `no_promotion_keep_extension_lane_only` 与 `no unique clean threshold`；它可以作为补充材料边界存在，但不能升格为并列主结果或成熟 comparator。
4. training/material、`.tflite`、real-board 三条 supporting boundary 都已经有足够 paper-facing 的“最强安全说法”：
   - training/material = canonical chain intact + one clean CPU-only bounded rerun
   - `.tflite` = isolated current-host true runtime for selected preserved artifacts
   - real-board = read-only gate / regeneration / provenance with current-host `NO_GO`
5. `T77` 与 `T78` 已把当前 note 的标题、摘要、引言、Related Work、结果层、讨论、结论及 `statcalib` 层级压回现有 evidence stack；因此，当前缺口已经不再是“结果材料有没有同步进 note”，而是“是否可以在不碰方法章和不升级证据等级的前提下恢复一轮有界 prose 组织”。

## Section-Level Readiness Matrix

| area | status | why | evidence anchors |
| --- | --- | --- | --- |
| 标题 / 摘要 | `ready_for_bounded_reopen` | 标题已去掉把 `statcalib` 写成并列主线的暗示；摘要已同步主结果、机制支撑和 deployment boundary 的保守口径 | `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`; `docs/paper_materials/paper_note_results_sync_manifest.md`; `docs/paper_materials/paper_note_alignment_and_layout_closeout.md` |
| 引言 | `ready_for_bounded_reopen` | `T78` 已把引言中的 evidence framing / contribution positioning 压回当前材料栈，且没有再把 deployment / `statcalib` 读成主结果 | `docs/paper_materials/paper_note_alignment_and_layout_closeout.md`; `docs/review/T78_review.md` |
| Related Work / positioning | `ready_for_bounded_reopen` | `T42/T43` 已给出 position/prose 草稿，`T78` 又把 note 中对应段落压回 `architectural and evidence-bounded` 口径 | `docs/paper_materials/paper_method_positioning_calibration.md`; `docs/paper_materials/paper_background_related_work_draft.md`; `docs/paper_materials/paper_note_alignment_and_layout_closeout.md` |
| 方法相关章节 | `defer_out_of_scope` | `Brief Review of the GKP Code`、`Noise and Drift Model`、`Model Architecture` 未被 `T78` 重校准，但下一轮若保持“有界 narrative/prose reopen”而不碰方法章，这并不阻塞 | `docs/paper_materials/paper_note_alignment_and_layout_closeout.md`; `docs/review/T78_review.md`; `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex` |
| Experimental Setup | `ready_for_bounded_reopen` | `T77` 已把 locked protocol、四场景矩阵和 extension-lane 分层同步进 note；当前口径足以支撑有界 prose 复写 | `docs/paper_materials/paper_note_results_sync_manifest.md`; `docs/review/T77_review.md`; `docs/review/T25_p4_formal_evidence_gate_review.md` |
| Numerical Results | `ready_for_bounded_reopen` | `T74`-`T76` 已锁定 stable ID、成图、callout 和装配顺序；`T77/T78` 已把 note 结果层与 `statcalib` hierarchy 压回当前边界 | `docs/paper_materials/paper_maintext_results_authoring_pack.md`; `docs/paper_materials/paper_results_callout_sheet.md`; `docs/paper_materials/paper_results_section_assembly_pack.md`; `docs/paper_materials/paper_note_results_sync_manifest.md`; `docs/paper_materials/paper_note_alignment_and_layout_closeout.md` |
| Discussion | `ready_for_bounded_reopen` | `T77/T78` 已把 strongest accepted result、`statcalib` no-promotion 与 deployment layered boundary 写回讨论段 | `docs/paper_materials/paper_note_results_sync_manifest.md`; `docs/paper_materials/paper_note_alignment_and_layout_closeout.md` |
| Conclusion | `ready_for_bounded_reopen` | 主结论、机制 hedge、`statcalib` extension lane 与 deployment boundary 已全部校准完成 | `docs/paper_materials/paper_note_results_sync_manifest.md`; `docs/paper_materials/paper_note_alignment_and_layout_closeout.md` |
| 主图 / 主表 / caption / insertion 路由 | `ready_for_bounded_reopen` | `T74`-`T76` 已把 stable IDs、caption、rendered QA 和 section assembly 锁好；当前只需按既有路由落 prose | `docs/paper_materials/paper_simulation_result_table_pack.md`; `docs/paper_materials/paper_figure_caption_pack.md`; `docs/paper_materials/paper_maintext_insertion_map.md`; `docs/paper_materials/paper_results_section_assembly_pack.md`; `docs/review/T74_review.md`; `docs/review/T75_review.md`; `docs/review/T76_review.md` |
| claim/evidence ledger | `ready_for_bounded_reopen` | `supported/partial/blocked` 三层已写清，足以约束 prose 不越界 | `docs/paper_materials/paper_claim_evidence_ledger.md`; `docs/review/T73_review.md` |
| risk table | `ready_for_bounded_reopen` | paper-facing 风险已按 `PR1-PR10` 锁定，足以作为 prose reopen 的 guardrail | `docs/paper_materials/paper_claim_risk_table.md`; `docs/08_risks_and_open_questions.md` |
| training/material supporting boundary | `ready_for_bounded_reopen` | 可以安全写成 supporting boundary，但不能扩写成 full reproducibility | `docs/paper_materials/paper_claim_evidence_ledger.md`; `docs/paper_materials/paper_claim_risk_table.md`; `docs/review/T50_review.md` |
| `.tflite` supporting boundary | `ready_for_bounded_reopen` | 可以安全写成 isolated current-host true runtime supporting boundary，但不能扩写成 default-env / deployment closure | `docs/paper_materials/paper_claim_evidence_ledger.md`; `docs/paper_materials/paper_claim_risk_table.md`; `docs/review/T48_review.md`; `docs/paper_materials/paper_results_callout_sheet.md` |
| real-board supporting boundary | `ready_for_bounded_reopen` | 可以安全写成 read-only gate/provenance with current-host `NO_GO`，但不能扩写成 execution success 或 hardware validation | `docs/paper_materials/paper_claim_evidence_ledger.md`; `docs/paper_materials/paper_claim_risk_table.md`; `docs/review/T72_review.md`; `docs/paper_materials/paper_results_callout_sheet.md` |

## What Is Already Sufficient

1. 已经有一套可直接复用的主结果写作骨架：`T75-FIG-M01` / `T75-FIG-M02` / `T75-FIG-A01`、`T76-CALLOUT-*`、`paper_results_section_assembly_pack.md` 与 `paper_maintext_results_authoring_pack.md` 足以支撑 Results、Discussion、Conclusion 的 bounded prose。
2. 当前 note 已经不是“结果层没同步”的状态：`T77` 负责结果层，`T78` 负责非结果层和 hierarchy/layout；因此标题、摘要、引言、Related Work、Discussion、Conclusion 都已有当前 evidence-aligned 文本基底。
3. claim/evidence ledger 与 risk table 已经足够承担 prose reopen 的 guardrail 角色：作者现在可以明确知道哪些 claim 是 `supported`、哪些只能写成 `partial`、哪些必须保持 `blocked`。
4. deployment-facing supporting material 现在虽然不强，但已经足够支撑一到两句有界收口：`.tflite`、real-board、training/material 三条边界都能诚实地写成支持性说明，而不需要先补新的运行实验。

## What Still Blocks Reopen

对“下一轮 bounded prose reopen”本身，没有新的材料 blocker；真正仍然被阻塞的是更强范围的 reopen：

1. `full-manuscript reopen` 仍然被方法章未校准、paper-grade expanded benchmark 缺失和 deployment closure 缺失所阻塞；
2. `paper-grade benchmark retelling` 仍然被 `C12/PR10` 阻塞；
3. `mechanism-closure retelling` 仍然被 `C4/PR2` 阻塞；
4. `default-env /.tflite / real-board success` 仍然分别被 `C8/C10/PR4/PR7/PR8` 阻塞。

因此，本次 gate 的含义只能是：

- 可以恢复一轮**有界** prose reopen；
- 不可以把这轮 reopen 误读成全文方法章扩写、部署故事放大、或对外 ready-to-submit 结论。

## Single Recommended Next Task

`T80: 主线校准段落的 bounded prose reopen`

推荐范围只包含当前已经 ready 的 narrative / result-facing 区域：

1. `Title`
2. `Abstract`
3. `Introduction`
4. `Related Work / positioning`
5. `Experimental Setup`
6. `Numerical Results`
7. `Discussion`
8. `Conclusion`

明确排除：

- `Brief Review of the GKP Code`
- `Noise and Drift Model`
- `Model Architecture`
- 任何新实验、新图表资产、`.tflite` / real-board retelling 升级、`statcalib` promotion、治理文档更新

这就是为什么本次 verdict 是 `GO_FOR_BOUNDED_PROSE_REOPEN`，而不是 full-manuscript ready。
