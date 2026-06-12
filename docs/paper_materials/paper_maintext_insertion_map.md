# T74 Paper Maintext Insertion Map

## 1. 使用原则

- 本图只定义 `main text` / `appendix` / `supplement only` 三层放置关系。
- 所有条目都必须使用 `T74-*` stable ID。
- 非 `main text` 项必须写明降级原因，而不是留 `TBD`。
- `blocked` 项可以出现在本图里，但只能作为“不要写”的显式提醒。

## 2. Main Text

| ID | 标题 | 建议稿件位置 | 直接证据 | 边界说明 |
| --- | --- | --- | --- | --- |
| `T74-TBL-01` | `T24` 冻结四场景主结果表 | Results / frozen benchmark subsection | `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/comparison.csv`<br>`docs/review/T24_review.md` | authoritative frozen ranked table；只代表 mock-backed software-HIL |
| `T74-FIG-01` | `T24` 冻结四场景结果汇总图 | Results / frozen benchmark subsection | `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/comparison.csv`<br>`runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/summary.json` | 如果不出图，可由 `T74-TBL-01` 直接替代；不能升级为 expanded benchmark 图 |
| `T74-FIG-02` | `FR6` 六 seed 机制/干预图 | Results / mechanism-or-ablation subsection | `docs/figure_assets/T58_fr6_multi_seed_mechanism_intervention/fr6_multi_seed_mechanism_intervention.svg`<br>`docs/review/T58_review.md` | descriptive only；不写 causal closure |

## 3. Appendix

| ID | 标题 | 建议稿件位置 | 直接证据 | 降级原因 |
| --- | --- | --- | --- | --- |
| `T74-TBL-02` | `FR7` feature/teacher ablation 表 | Appendix / ablation details | `runs/p4_benchmark/T57_fr7_feature_teacher_ablation_20260524_000000/summary_pack/table.csv`<br>`docs/review/T57_review.md` | 重要但细节密度高，且仅限 bounded frozen-set ablation |
| `T74-TBL-03` | `FR6` 六 seed 描述性汇总表 | Appendix / figure companion table | `docs/figure_assets/T58_fr6_multi_seed_mechanism_intervention/figure_data.csv`<br>`docs/review/T58_review.md` | 主文优先放图，表格用于 reviewer 查数 |
| `T74-TBL-04` | training/material reproducibility boundary 表 | Appendix / material provenance | `artifacts/t50_training_repro_pack/training_reproducibility_pack.json`<br>`docs/review/T50_review.md` | 它是材料与 provenance 支撑层，不是主 simulation ranking |
| `T74-TBL-05` | isolated true `.tflite` runtime boundary 表 | Appendix / deployment-adjacent boundary | `artifacts/t48_true_tflite_runtime_gate/t48_true_tflite_runtime_gate.json`<br>`docs/review/T48_review.md` | `.tflite` 真执行已存在，但仍是 isolated current-host boundary，不宜主文升格 |
| `T74-FIG-03` | 证据等级/边界示意图 | Appendix / boundary overview | `docs/03_hil_p4_boundary_audit.md`<br>`docs/evidence_packs/deployment_boundary/t72_real_board_transfer_pack_provenance_hardening.md` | 解释证据层级很有用，但不属于结果图本身 |

## 4. Supplement Only

| ID | 标题 | 建议稿件位置 | 直接证据 | 降级或保留原因 |
| --- | --- | --- | --- | --- |
| `T74-TBL-06` | real-board gate / regeneration / provenance boundary 表 | Supplement / hardware gate boundary | `artifacts/t72_real_board_transfer_pack_provenance_hardening/replay_vs_regeneration_comparison.json`<br>`docs/review/T72_review.md` | 当前还是 `NO_GO` host/device gate，不应进入主文或普通附录结果层 |
| `T74-TBL-07` | `FR8 statcalib` extension-lane closure 表 | Supplement / extension-lane closure | `docs/evidence_packs/statcalib_fr8/fr8_statcalib_bounded_closure_pack.md`<br>`docs/review/T70_review.md` | `statcalib` 仍是 separately labeled extension lane；不可 promotion |
| `T74-FIG-04` | training-to-deployment portability 大闭环图 | Supplement / blocked slot only | `docs/paper_materials/paper_result_figure_ledger.md`<br>`docs/evidence_packs/training_reproducibility/training_reproducibility_and_material_regeneration_pack.md` | 当前证据不支持诚实出图；保留为“不要写”的显式提醒 |
| `T74-SUP-01` | `FR8` no-promotion / no-unique-threshold 说明 | Supplement note | `docs/review/T70_review.md` | 必须跟着 `T74-TBL-07` 一起出现，防止 overclaim |
| `T74-SUP-02` | `T50` full reproducibility exclusion 说明 | Supplement note | `docs/review/T50_review.md` | 保护 training/material 表不被误写成 full reproducibility |
| `T74-SUP-03` | `T48` default-env / deployment closure exclusion 说明 | Supplement note | `docs/review/T48_review.md` | 保护 `.tflite` 表不被误写成 default-env 或 deployment closure |
| `T74-SUP-04` | `T72` future-host / hardware-condition gap 说明 | Supplement note | `docs/review/T72_review.md` | 真实硬件条件仍缺失，且 `R32` 仍在 |

## 5. 最小可提交 simulation/material 组合

如果当前目标是先形成 paper-ready 的 simulation/material 版本，而不等待 real-board host：

1. 主文至少放 `T74-TBL-01`，再在需要时从 `T74-FIG-01` 与 `T74-FIG-02` 中选一到两项。
2. 附录优先放 `T74-TBL-02` 到 `T74-TBL-05`，以及 `T74-FIG-03`。
3. `T74-TBL-06`、`T74-TBL-07` 与 `T74-SUP-*` 保留在补充材料。
4. `T74-FIG-04` 当前只能作为 blocked reminder，不进入稿件成图链。
