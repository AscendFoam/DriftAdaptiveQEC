# T86 Submission Surface Route Map

本表把当前 mainline claim/material 压成 submission-facing 路由图。它回答“放哪一层、怎么回链、哪里必须停下”，不升级任何证据等级。

| claim_or_section | main_text_route | appendix_route | supplement_route | exclusion_note | source_anchor |
| --- | --- | --- | --- | --- | --- |
| frozen benchmark main result (`T24`) | `T74-TBL-01` / `T75-FIG-M01`；保留在 `Numerical Results` 主结果段 | `T74-FIG-01` 可作为可选 companion，不是更强证据 | `none` | 不写成 expanded benchmark、`.tflite` ranking 或 real-board ranking | `paper_maintext_insertion_map.md`; `paper_maintext_results_authoring_pack.md`; `paper_claim_evidence_ledger.md` `C2/C3` |
| conservative mechanism / ablation interpretation (`FR6/FR7`) | 主文保留 `T75-FIG-M02` 的 descriptive interpretation | `T74-TBL-02`、`T74-TBL-03` 作为查数与 ablation details | `none` | 不写 causal closure、teacher necessity 或 intervention success | `paper_appendix_bridge_pack.md`; `paper_claim_evidence_ledger.md` `C4`; `paper_authoring_do_not_write_list.md` |
| training/material supporting boundary | `none` | `T74-TBL-04` appendix provenance/support table | `T74-SUP-02` 作为 exclusion note 可留在 supplement note 区 | 不写 full reproducibility、repeated-run closure、GPU/CUDA/Linux portability | `paper_supporting_material_closeout_pack.md`; `paper_claim_evidence_ledger.md` `C5/C6` |
| isolated current-host true `.tflite` runtime boundary | `none` | `T74-TBL-05` appendix runtime boundary table；必要时配 `T75-FIG-A01` | `T74-SUP-03` 保留 default-env / deployment exclusion note | 不写 default-env recovered、HIL closure、deployment closure | `paper_supporting_material_closeout_pack.md`; `paper_claim_evidence_ledger.md` `C7/C8`; `paper_authoring_do_not_write_list.md` |
| evidence-boundary schematic | 主文只可一句提及分层结构，不单独承担结果叙事 | `T75-FIG-A01` / `T74-FIG-03` 作为 appendix boundary schematic | `none` | 不写成 deployment success schematic 或 portability closure figure | `paper_result_figure_ledger.md` `F2`; `paper_appendix_bridge_pack.md` |
| `FR8/statcalib` extension-lane closure | `none` | `none` | `T74-TBL-07` + `T74-SUP-01`，保持 extension-lane/no-promotion 边界 | 不写 mature comparator、promotion、unique clean threshold | `paper_supporting_material_closeout_pack.md`; `paper_claim_evidence_ledger.md` `C11`; `paper_claim_risk_table.md` `PR9` |
| read-only real-board gate / provenance | `none` | `none` | `T74-TBL-06` + `T74-SUP-04`，保持 current-host `NO_GO` gate truth | 不写 real-board execution success、hardware validated、board-ready path | `paper_supporting_material_closeout_pack.md`; `paper_claim_evidence_ledger.md` `C9/C10`; `paper_claim_risk_table.md` `PR7/PR8` |
| hardware-dependent execution / timing / resource | `none` | `none` | `none` | 当前 package 只可显式排除，不进入主文/附录/补充的完成态叙事 | `paper_submission_blocker_matrix.md` `SB02`; `docs/08_risks_and_open_questions.md` `R13/R14/R32/R33` |
| default-env `.tflite` / deployment portability | `none` | `none` | `none` | 当前 package 只可显式排除，不进入 closure 叙事 | `paper_submission_blocker_matrix.md` `SB03`; `paper_claim_risk_table.md` `PR4` |
| full training reproducibility | `none` | `none` | `none` | 当前 package 只可显式排除，不进入 completion 叙事 | `paper_submission_blocker_matrix.md` `SB04`; `paper_claim_risk_table.md` `PR3` |
