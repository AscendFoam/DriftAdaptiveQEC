# T82 Supporting-Material Closeout Pack

## 1. Scope Verdict

- verdict: `SUPPORTING_MATERIAL_CLOSEOUT_COMPLETED`
- 本轮只做 supporting-material 与 appendix/supplement 边界整合，不做 full-manuscript reopen。
- 本轮只回写 note 中 4 处 supporting-boundary 段落：
  - `Runtime, quantization, and fixed-point degradation`
  - `Embedded runtime and board-level validation`
  - `Discussion` 中的 deployment/support boundary 段落
  - `Conclusion` 中的 remaining technical gap 段落
- 本轮没有改写：
  - `Title`、`Abstract`、`Introduction`
  - `Summary of Contributions` 与三章 methods
  - `Experimental Setup`、主结果段落、任何 figure/table/stable-ID/caption/insertion map
  - 任何 benchmark、training、`.tflite` smoke、real-board gate/regeneration/execution、治理文档或历史事实文件

## 2. Main Text / Appendix / Supplement / Blocked Routing

| supporting surface | placement | evidence anchors | safest manuscript use | forbidden claims |
| --- | --- | --- | --- | --- |
| `T24` frozen benchmark anchor | `main text` | `docs/paper_materials/paper_claim_evidence_ledger.md` `C2/C3`; `docs/review/T24_review.md`; `docs/review/T25_p4_formal_evidence_gate_review.md` | 作为 frozen-set main result 与主线排名锚点 | 不能写成 expanded benchmark、`.tflite`、real-board 或 deployment closure |
| `FR6/FR7` descriptive support | `main text + appendix` | `docs/paper_materials/paper_claim_evidence_ledger.md` `C4`; `docs/review/T57_review.md`; `docs/review/T58_review.md`; `docs/paper_materials/paper_appendix_bridge_pack.md` | 主文保留保守解读，附录保留查数表与补充说明 | 不能写成 causal closure、teacher necessity 或 mitigation success |
| training/material reproducibility boundary | `appendix` | `docs/paper_materials/paper_claim_evidence_ledger.md` `C5/C6`; `docs/review/T50_review.md`; `docs/paper_materials/paper_maintext_insertion_map.md` `T74-TBL-04` | 只写 canonical chain intact + one clean CPU-only bounded rerun | 不能写成 full reproducibility、repeated-run closure、GPU/CUDA/Linux portability |
| isolated current-host true `.tflite` runtime | `appendix` | `docs/paper_materials/paper_claim_evidence_ledger.md` `C7/C8`; `docs/review/T48_review.md`; `docs/paper_materials/paper_maintext_insertion_map.md` `T74-TBL-05` | 只写 selected preserved artifacts 的 isolated current-host true runtime | 不能写成 default-env recovered、HIL closure、deployment closure |
| `FR8/statcalib` extension-lane closure | `supplement` | `docs/paper_materials/paper_claim_evidence_ledger.md` `C11`; `docs/review/T70_review.md`; `docs/paper_materials/paper_maintext_insertion_map.md` `T74-TBL-07`; `docs/paper_materials/paper_appendix_bridge_pack.md` | 只写 separately labeled extension lane、no-promotion、no unique clean threshold | 不能写成 mature comparator、`T24` 替代表或唯一 clean threshold |
| read-only real-board gate / regeneration / provenance | `supplement` | `docs/paper_materials/paper_claim_evidence_ledger.md` `C9/C10`; `docs/review/T72_review.md`; `docs/paper_materials/paper_maintext_insertion_map.md` `T74-TBL-06` | 只写 checked-in read-only gate/provenance pack with current-host `NO_GO` | 不能写成 execution success、hardware validated、real-board ready |
| hardware-dependent execution surface | `blocked` | `docs/08_risks_and_open_questions.md` `R13/R14/R32/R33`; `docs/review/T72_review.md`; `docs/paper_materials/paper_submission_material_gap_checklist.md` | 只保留 blocked/future-host 需求说明 | 不能写成“现有路径已经可执行”或“只差补一句文案” |

## 3. Surface Integration Notes

### 3.1 `FR8/statcalib`

- 当前最强可引用口径是：`statcalib` 已形成一条 bounded mock-backed software-HIL extension lane，并且 `T70` 已明确给出 `no_promotion_keep_extension_lane_only` 与 `future_selection_task_required`。
- 这条 surface 应保留在 supplement，而不是回挤进主文主结果层。
- 原因不是它“没价值”，而是它的价值恰好在于说明同一 affine runtime contract 可以容纳强 non-neural slow-loop，但当前并没有得到 unique clean threshold，也没有得到 promotion 许可。

### 3.2 training/material reproducibility

- 当前最强可引用口径是：canonical chain intact + one clean CPU-only bounded rerun。
- 这条 surface 适合放在 appendix，用来说明 preserved artifacts、canonical references 与 clean rerun 没有漂移。
- 它不应该被并入主结果表，也不应该被升格成 full reproducibility 或跨主机保证。

### 3.3 isolated current-host true `.tflite` runtime

- 当前最强可引用口径是：selected preserved float/int8 `.tflite` artifacts have real current-host execution under one isolated `tensorflow==2.21.0` environment。
- 这条 surface 适合放在 appendix，与 training/material table 一起构成 deployment-adjacent supporting layer。
- 它只证明“当前主机、隔离环境、选定保留 artifact”这条窄路径，不证明 default-env、HIL 集成、portability 或 deployment closure。

### 3.4 read-only real-board gate / provenance

- 当前最强可引用口径是：仓库已有 checked-in、read-only、role-aware、可 replay/regeneration 的 gate/provenance pack；current-host verdict 仍为 `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`。
- 这条 surface 应留在 supplement，因为它是 gate truth，不是主文或普通附录结果层。
- 它能回答“当前硬件边界在哪里”，但不能回答“真板已经跑通”。

### 3.5 blocked hardware-dependent surface

- 当前明确 blocked 的不是某一张没画出来的表，而是一整层需要未来硬件宿主的 execution surface：
  - `Linux + FPGA` 宿主
  - 可打开的 device path
  - board-level latency/resource rows
  - host-to-device update cost
  - closed-loop timing / feedback evidence
- 在这些前提缺失前，最安全做法是把它们显式保留为 blocked，而不是用 supporting prose 暗示“路径其实已经具备”。

## 4. Note Integration Applied By T82

| note surface | T82 marker | integrated route | non-claim kept explicit |
| --- | --- | --- | --- |
| `Runtime, quantization, and fixed-point degradation` | `% T82-SUPPORT: Runtime, quantization, and fixed-point degradation` | 把 isolated true runtime 固定到 appendix-level supporting boundary | 不把 runtime boundary 写成 deployment closure |
| `Embedded runtime and board-level validation` | `% T82-SUPPORT: Embedded runtime and board-level validation` | 把 appendix-level current-host runtime 与 supplement-only real-board gate 分开 | 不把 gate/provenance 写成 execution path |
| `Discussion` deployment/support boundary paragraph | `% T82-SUPPORT: Discussion deployment/support boundary` | 把 `main text / appendix / supplement / blocked` 四层路由写明 | 不把 layered evidence 扁平化成单一 deployment story |
| `Conclusion` remaining technical gap paragraph | `% T82-SUPPORT: Conclusion remaining technical gap` | 把“supporting-material integration 已完成”与“hardware-dependent surface 仍 blocked”同时写明 | 不把 `T82` 写成 full-manuscript finalization 或 hardware closure |

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
- `.log` scan result:
  - 未检出 `Underfull`
  - 未检出 `Overfull`
  - 未检出 `LaTeX Warning`
  - 未检出 `undefined` / `Citation`
