# T81 Summary And Methods Calibration Manifest

## 1. Scope Verdict

- verdict: `METHODS_AND_CONTRIBUTION_CALIBRATION_COMPLETED`
- 本轮只改写以下 4 个 target sections：
  - `Summary of Contributions`
  - `Brief Review of the GKP Code`
  - `Noise and Drift Model`
  - `Model Architecture`
- 本轮没有扩展到：
  - `Title` / `Abstract` / `Introduction` / `Relationship to Existing Work`
  - `Experimental Setup` / `Numerical Results` / `Discussion` / `Conclusion`
  - 任何 benchmark、training、`.tflite`、real-board、governance 或 theory-branch 大范围改写

## 2. Section Change Ledger

| section | changed_or_not | evidence_anchors | non_claims_and_guardrails |
| --- | --- | --- | --- |
| `Summary of Contributions` | `yes` | `docs/paper_materials/paper_bounded_prose_reopen_manifest.md`; `docs/paper_materials/paper_claim_evidence_ledger.md`; `docs/paper_materials/paper_claim_risk_table.md`; `docs/review/T80_review.md` | `T24` 仍是主线 frozen-set main anchor；`FR6/FR7` 仍是 descriptive support；`FR8/statcalib` 仍是 extension lane / no-promotion / no unique clean threshold；training/material、`.tflite`、real-board 仍是 layered boundary evidence |
| `Brief Review of the GKP Code` | `yes` | `docs/paper_materials/paper_reopen_gap_matrix.md`; `docs/paper_materials/paper_reopen_gate_and_prose_readiness_review.md`; `docs/paper_materials/paper_claim_risk_table.md` | 只承担 GKP 物理与局部 affine 近似背景；不把局部 affine 近似写成 exact decoder closure；不把 software-HIL 结果、hardware timing 或 deployment 结果偷带进背景章 |
| `Noise and Drift Model` | `yes` | `docs/paper_materials/paper_reopen_gap_matrix.md`; `docs/paper_materials/paper_claim_evidence_ledger.md`; `docs/paper_materials/paper_claim_risk_table.md`; `docs/review/T25_p4_formal_evidence_gate_review.md` | 明确这是 effective model / control-oriented abstraction；不把四场景 effective drift 写成 full circuit-level closure、exhaustive drift coverage 或 hardware-validated noise model |
| `Model Architecture` | `yes` | `docs/paper_materials/paper_method_positioning_calibration.md`; `docs/paper_materials/paper_bounded_prose_reopen_manifest.md`; `docs/paper_materials/paper_claim_evidence_ledger.md`; `docs/paper_materials/paper_claim_risk_table.md`; `docs/review/T48_review.md`; `docs/review/T70_review.md`; `docs/review/T72_review.md` | mainline architecture 仍是 teacher-anchored residual path；`statcalib` 仍是 separately labeled FR8 extension lane；`.tflite` 仍是 isolated current-host true runtime；real-board 仍是 read-only gate/provenance with current-host `NO_GO`；不把 architecture 章写成 board-level closure |

## 3. T80 Ready Sections Left Untouched

| section | status | note |
| --- | --- | --- |
| `Title` | `untouched_in_T81` | `% T80-REOPEN` 标记保留 |
| `Abstract` | `untouched_in_T81` | `% T80-REOPEN` 标记保留 |
| `Introduction` | `untouched_in_T81` | `% T80-REOPEN` 标记保留 |
| `Relationship to Existing Work` | `untouched_in_T81` | `% T80-REOPEN` 标记保留 |
| `Experimental Setup` | `untouched_in_T81` | `% T80-REOPEN` 标记保留 |
| `Numerical Results` | `untouched_in_T81` | `% T80-REOPEN` 标记保留 |
| `Discussion` | `untouched_in_T81` | `% T80-REOPEN` 标记保留 |
| `Conclusion` | `untouched_in_T81` | `% T80-REOPEN` 标记保留 |

说明：

- `T81` 的目标不是再开一轮 ready-section prose reopen，而是补齐 `T80` 刻意保留 untouched 的 contribution/methods 缺口。
- 因此，`T80` 的 8 个 ready sections 在本轮只做范围校验，不做新的大段重写。

## 4. Boundary Checklist

| checklist item | status | note |
| --- | --- | --- |
| `T24` main anchor preserved | `yes` | `Summary of Contributions` 与 methods 叙事都继续把 `T24` 作为主线 frozen-set ranking anchor |
| `FR6/FR7` descriptive-only preserved | `yes` | `Summary of Contributions` 继续把多 seed 机制图与 feature/teacher ablation 写成 descriptive support |
| `FR8/statcalib` extension-lane no-promotion preserved | `yes` | `Summary of Contributions` 与 `Model Architecture` 均继续保留 extension lane / no-promotion / no unique clean threshold |
| training/material supporting boundary preserved | `yes` | 只写 canonical chain intact + one clean CPU-only bounded rerun |
| `.tflite` supporting boundary preserved | `yes` | 只写 isolated current-host true runtime for selected preserved artifacts |
| real-board supporting boundary preserved | `yes` | 只写 read-only gate / regeneration / provenance with current-host `NO_GO` |
| methods kept as calibration, not evidence upgrade | `yes` | `Brief Review` / `Noise and Drift Model` / `Model Architecture` 均未把 theory/model prose 扩写成新实验或部署完成态 |

## 5. Compile Status

- status: `compiled`
- detector status: `existing-usable`
- effective toolchain: `TeX Live 2024 + latexmk`
- doctor smoke result:
  - `latexmk` smoke = `passed`
  - bundled `tectonic` executable = `available`
  - bundled `tectonic` smoke = `failed` with `os error 5`
- compiled target: `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`
- compile output set refreshed/present:
  - `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.aux`
  - `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.fdb_latexmk`
  - `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.fls`
  - `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.log`
  - `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.out`
  - `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.pdf`
  - `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.synctex.gz`
  - `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.toc`
- compiled PDF status:
  - `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.pdf` exists
  - compile command exit code = `0`
- log scan result:
  - 未检出 `Underfull`
  - 未检出 `Overfull`
  - 未检出 `LaTeX Warning`
  - 未检出 `undefined` / `Citation`

## 6. Out-Of-Scope Areas Left Untouched

- `docs/00_*` 至 `docs/08_*` 治理文档
- 任何 `cnn_fpga/`、`physics/`、`benchmark/`、`tests/`
- 任何 `runs/`、`artifacts/`、`docs/evidence_packs/`
- 任何新 figure/table/stable-ID/caption/insertion-map
- 任何 benchmark、training、`.tflite` smoke、real-board 执行
