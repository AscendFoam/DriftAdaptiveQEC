# T80 Mainline Calibrated Sections Bounded Prose Reopen Manifest

## 1. Scope Verdict

- verdict: `SECTION_BOUNDED_REOPEN_COMPLETED`
- 本轮只覆盖以下 8 个 ready sections：
  - `Title`
  - `Abstract`
  - `Introduction`
  - `Relationship to Existing Work`
  - `Experimental Setup`
  - `Numerical Results and Benchmark Plan`
  - `Discussion`
  - `Conclusion`
- 本轮没有扩展到 methods calibration、benchmark rerun、`.tflite` / real-board retelling 升级，也没有进入 full-manuscript reopen。

## 2. Section Change Ledger

| section | changed_or_not | evidence_anchors | guardrails_preserved |
| --- | --- | --- | --- |
| `Title` | `yes` | `docs/paper_materials/paper_method_positioning_calibration.md`; `docs/paper_materials/paper_reopen_gate_and_prose_readiness_review.md`; `docs/paper_materials/paper_note_alignment_and_layout_closeout.md` | 保持 method-forward / evidence-bounded；不把 `statcalib`、`.tflite`、real-board 写成并列主线 |
| `Abstract` | `yes` | `docs/paper_materials/paper_note_results_sync_manifest.md`; `docs/paper_materials/paper_reopen_gate_and_prose_readiness_review.md`; `docs/paper_materials/paper_claim_evidence_ledger.md`; `docs/paper_materials/paper_claim_risk_table.md` | `T24` 仍是主结果锚点；`FR6/FR7` 仅 descriptive support；`FR8` 仍是 extension lane / no-promotion |
| `Introduction` | `yes` | `docs/paper_materials/paper_note_alignment_and_layout_closeout.md`; `docs/paper_materials/paper_reopen_gate_and_prose_readiness_review.md`; `docs/paper_materials/paper_claim_evidence_ledger.md` | 只重述 framework 与 evidence hierarchy；不把 deployment-facing 材料和 `statcalib` 提升成主结果 |
| `Relationship to Existing Work` | `yes` | `docs/paper_materials/paper_method_positioning_calibration.md`; `docs/paper_materials/paper_background_related_work_draft.md`; `docs/paper_materials/paper_note_alignment_and_layout_closeout.md` | 保持 architectural / evidence-bounded positioning；不暗示已完成 board-level deployment 或 mature comparator closure |
| `Experimental Setup` | `yes` | `docs/paper_materials/paper_note_results_sync_manifest.md`; `docs/review/T24_review.md`; `docs/review/T25_p4_formal_evidence_gate_review.md` | 明确 locked `T24` protocol 是 authoritative frozen anchor；保持 `mock-backed software-HIL only` |
| `Numerical Results and Benchmark Plan` | `yes` | `docs/paper_materials/paper_maintext_results_authoring_pack.md`; `docs/paper_materials/paper_results_callout_sheet.md`; `docs/paper_materials/paper_results_section_assembly_pack.md`; `docs/paper_materials/paper_note_results_sync_manifest.md`; `docs/paper_materials/paper_ablation_result_pack.md`; `docs/review/T48_review.md`; `docs/review/T70_review.md`; `docs/review/T72_review.md` | `T24` 仍是主排名；`FR6/FR7` 不写成 causal closure；`FR8` 不写成 promoted comparator；`.tflite` 只写 isolated current-host true runtime；real-board 只写 gate/provenance with `NO_GO` |
| `Discussion` | `yes` | `docs/paper_materials/paper_note_results_sync_manifest.md`; `docs/paper_materials/paper_claim_evidence_ledger.md`; `docs/paper_materials/paper_claim_risk_table.md` | 明确 strongest accepted result 仍是 `T24`；training/material 只保留 canonical chain + one clean CPU-only bounded rerun；deployment-facing 仍是 layered boundary |
| `Conclusion` | `yes` | `docs/paper_materials/paper_note_results_sync_manifest.md`; `docs/paper_materials/paper_claim_evidence_ledger.md`; `docs/paper_materials/paper_claim_risk_table.md`; `docs/paper_materials/paper_reopen_gap_matrix.md` | 保持两层主结论：mainline frozen result + bounded support/boundary；不外推成 full reproducibility、default-env `.tflite`、real-board success 或 `statcalib` promotion |

## 3. Boundary Checklist

| checklist item | status | note |
| --- | --- | --- |
| `T24` main anchor preserved | `yes` | 主结果段仍把 locked four-scenario frozen benchmark 作为 authoritative ranking |
| `FR6/FR7` descriptive-only preserved | `yes` | 机制段改写仍保留 `descriptive rather than causal` 与 `mixed and mostly harmful` 口径 |
| `FR8` extension-lane no-promotion preserved | `yes` | `statcalib` 仍保留 supplement-side / no-promotion / no unique clean threshold |
| training/material supporting boundary preserved | `yes` | 只写 canonical chain intact + one clean CPU-only bounded rerun |
| `.tflite` supporting boundary preserved | `yes` | 只写 isolated current-host true runtime for selected preserved artifacts |
| real-board supporting boundary preserved | `yes` | 只写 read-only gate / regeneration / provenance with current-host `NO_GO` |
| methods chapters untouched | `yes` | `Brief Review of the GKP Code`、`Noise and Drift Model`、`Model Architecture` 未改写 |

## 4. Compile Status

- status: `compiled`
- effective toolchain: `TeX Live 2024 + latexmk`
- doctor result: `existing-usable`
- compiled target: `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`
- compile output set present after compile:
  - `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.aux`
  - `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.fdb_latexmk`
  - `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.fls`
  - `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.log`
  - `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.out`
  - `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.pdf`
  - `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.synctex.gz`
  - `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.toc`
- log scan result:
  - 未检出 `Underfull`
  - 未检出 `Overfull`
  - 未检出 `LaTeX Warning`
  - 未检出 `undefined` / `Citation`

## 5. Out-of-Scope Sections Left Untouched

- `Summary of Contributions`
- `Brief Review of the GKP Code`
- `Noise and Drift Model`
- `Model Architecture`

说明：

- `Summary of Contributions` 虽然位于引言之后，但它不在 `T80` 允许重写的 8 个 section 列表内，因此本轮刻意不动。
- 三章 methods 是 `T79` 明确标记为 `defer_out_of_scope` 的区域；本轮没有借 prose reopen 名义把 methods calibration 偷带进来。
