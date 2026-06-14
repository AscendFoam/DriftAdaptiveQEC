# T86 Submission Pack Assembly Manifest

本表只登记当前 submission-facing package 中可以被实际组装的 surface，以及每个 surface 应如何被作者继续处理。它不是 claim 升级表。

| surface_id | surface_role | included_source | evidence_anchor | author_action |
| --- | --- | --- | --- | --- |
| `PKG-MT-01` | `main_text_core_result` | `T75-FIG-M01` and/or `T74-TBL-01` | `paper_maintext_results_authoring_pack.md`; `paper_maintext_insertion_map.md`; `paper_claim_evidence_ledger.md` `C2/C3` | 保留为主文 frozen benchmark 核心结果；若版面不适合图，可回退到 `T74-TBL-01`，但不得写强证据等级。 |
| `PKG-MT-02` | `main_text_conservative_mechanism_layer` | `T75-FIG-M02` | `paper_maintext_results_authoring_pack.md`; `paper_claim_evidence_ledger.md` `C4`; `paper_authoring_do_not_write_list.md` | 在主文只保留 descriptive interpretation；避免把 figure 或 prose 写成因果闭环。 |
| `PKG-APX-01` | `appendix_ablation_support` | `T74-TBL-02` | `paper_appendix_bridge_pack.md`; `paper_maintext_insertion_map.md` | 作为 frozen-set ablation 细节表保留在 appendix，供 reviewer 查数，不回挤主文。 |
| `PKG-APX-02` | `appendix_cross_seed_companion` | `T74-TBL-03` | `paper_appendix_bridge_pack.md`; `paper_result_figure_ledger.md` `FR6` | 作为 `FR6` 图包 companion table 保留在 appendix，只支撑 descriptive reading。 |
| `PKG-APX-03` | `appendix_training_material_boundary` | `T74-TBL-04` | `paper_supporting_material_closeout_pack.md`; `paper_claim_evidence_ledger.md` `C5/C6` | 只写 canonical chain intact + one clean CPU-only rerun，不写 full reproducibility。 |
| `PKG-APX-04` | `appendix_tflite_runtime_boundary` | `T74-TBL-05` | `paper_supporting_material_closeout_pack.md`; `paper_claim_evidence_ledger.md` `C7/C8` | 只写 isolated current-host true runtime for selected preserved artifacts，不写 portability/deployment closure。 |
| `PKG-APX-05` | `appendix_boundary_schematic` | `T75-FIG-A01` and/or `T74-FIG-03` | `paper_result_figure_ledger.md` `F2`; `paper_appendix_bridge_pack.md` | 用来固定 evidence hierarchy；保持它是 boundary schematic，而不是 result figure。 |
| `PKG-SUP-01` | `supplement_statcalib_extension_lane` | `T74-TBL-07` + `T74-SUP-01` | `paper_supporting_material_closeout_pack.md`; `paper_claim_evidence_ledger.md` `C11`; `paper_claim_risk_table.md` `PR9` | 在 supplement 保留 extension-lane/no-promotion wording，避免 promotion retelling。 |
| `PKG-SUP-02` | `supplement_real_board_gate_boundary` | `T74-TBL-06` + `T74-SUP-04` | `paper_supporting_material_closeout_pack.md`; `paper_claim_evidence_ledger.md` `C9/C10`; `paper_claim_risk_table.md` `PR7/PR8` | 在 supplement 保留 current-host `NO_GO` gate/provenance truth，不写 execution success。 |
| `PKG-SUP-03` | `supplement_exclusion_notes` | `T74-SUP-02` + `T74-SUP-03` | `paper_submission_material_gap_checklist.md`; `paper_authoring_do_not_write_list.md` | 把 training full reproducibility exclusion 与 `.tflite` default-env/deployment exclusion 明确写成 note，不让 supporting table 被读强。 |
