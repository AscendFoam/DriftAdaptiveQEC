# T82 Manuscript Closeout Readiness Matrix

## 1. 作用

本矩阵回答的是“当前 manuscript-facing surface 能写到哪一层”，而不是“项目还差多少工作才算彻底完成”。

状态只使用三档：

- `ready`
- `support-only`
- `blocked`

## 2. Readiness Matrix

| surface_or_section | readiness_status | blocker_type | evidence anchors | forbidden claims | next bounded action |
| --- | --- | --- | --- | --- | --- |
| frozen benchmark main result layer (`T24` anchor) | `ready` | `none_inside_current_route` | `docs/paper_materials/paper_claim_evidence_ledger.md` `C2/C3`; `docs/review/T24_review.md`; `docs/review/T25_p4_formal_evidence_gate_review.md`; `docs/paper_materials/paper_bounded_prose_reopen_manifest.md` | 不写成 expanded benchmark、`.tflite`、real-board 或 deployment result | 当前 route 内无额外动作；若要扩 benchmark，需新开 protocol-then-execution 任务 |
| contribution + methods mainline calibration | `ready` | `none_inside_current_route` | `docs/paper_materials/paper_methods_and_contribution_calibration_manifest.md`; `docs/review/T81_review.md` | 不把 methods prose 写成新实验或 deployment closure | 当前 route 内无额外动作；若要继续扩章，需另开新的 bounded prose task |
| conservative mechanism / ablation retelling (`FR6/FR7`) | `ready` | `descriptive_only_boundary` | `docs/paper_materials/paper_claim_evidence_ledger.md` `C4`; `docs/review/T57_review.md`; `docs/review/T58_review.md` | 不写 causal closure、teacher necessity、general harmful-instability 命题 | 如果要强化机制结论，需新开 focused mechanism task，而不是扩 prose |
| training/material appendix boundary | `support-only` | `reproducibility_not_closed` | `docs/paper_materials/paper_claim_evidence_ledger.md` `C5/C6`; `docs/review/T50_review.md`; `docs/paper_materials/paper_maintext_insertion_map.md` `T74-TBL-04` | 不写 full reproducibility、cross-host、GPU/CUDA、Linux portability | 若要升级，开 repeated-run / cross-host reproducibility task |
| isolated current-host true `.tflite` appendix boundary | `support-only` | `default_env_and_portability_open` | `docs/paper_materials/paper_claim_evidence_ledger.md` `C7/C8`; `docs/review/T48_review.md`; `docs/paper_materials/paper_maintext_insertion_map.md` `T74-TBL-05` | 不写 default-env recovered、HIL closure、deployment closure | 若要升级，开 default-env / portability / integration audit task |
| `FR8/statcalib` supplement closure | `support-only` | `extension_lane_no_promotion` | `docs/paper_materials/paper_claim_evidence_ledger.md` `C11`; `docs/review/T70_review.md`; `docs/paper_materials/paper_maintext_insertion_map.md` `T74-TBL-07` | 不写 mature comparator、promotion、unique clean threshold | 若要升级，开新的 bounded validation/gate task |
| read-only real-board gate / provenance supplement boundary | `support-only` | `no_current_host_execution_path` | `docs/paper_materials/paper_claim_evidence_ledger.md` `C9/C10`; `docs/review/T72_review.md`; `docs/paper_materials/paper_maintext_insertion_map.md` `T74-TBL-06` | 不写 execution success、hardware validated、ready-to-run board path | 若要升级，先有 future host，再开 read-only-to-execution transition task |
| hardware-dependent execution / board timing / resource rows | `blocked` | `hardware_host_missing + gate_not_passed` | `docs/08_risks_and_open_questions.md` `R13/R14/R32/R33`; `docs/review/T72_review.md` | 不写成“现有 supporting materials 已足以证明硬件执行” | 等待 `Linux + FPGA` host 后，再开 future-host gate / smoke task |
| default-env `.tflite` / deployment portability story | `blocked` | `runtime_portability_not_closed` | `docs/08_risks_and_open_questions.md` `R12`; `docs/review/T48_review.md` | 不写成“true runtime 已自然延伸到 default-env / deployment” | 开 bounded portability/bootstrap task |
| full-manuscript closeout / submission-ready holistic closure | `blocked` | `scope_not_opened + layered_boundaries_still_open` | `docs/paper_materials/paper_reopen_gap_matrix.md`; `docs/paper_materials/paper_supporting_material_closeout_pack.md`; `docs/review/T81_review.md` | 不把 `T82` 回述成 full-manuscript reopen、final closeout 或 deployment-ready paper | 由 Captain 根据 `T82` closeout 另开唯一后续 gate/task |

## 3. Readiness Interpretation

### 3.1 `ready`

- 表示这层内容已经可以进入当前 manuscript-facing route，只要继续保留现有 guardrail。
- `ready` 不等于“以后不会再被修改”，只等于“当前可以诚实使用”。

### 3.2 `support-only`

- 表示这层内容可以被引用，但只能作为 appendix/supplement/supporting boundary，而不能冒充主结果或闭环结论。
- 这些 surface 在 `T82` 中被整合，是为了减少 overclaim，不是为了给它们升级等级。

### 3.3 `blocked`

- 表示当前缺的不是文案，而是证据前提、任务授权或硬件条件。
- 这类 surface 的正确处理方式是显式写 blocked / future-host / future-task，而不是“先写进去，等以后再补”。
