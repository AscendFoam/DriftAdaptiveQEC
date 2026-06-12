# Paper Result/Figure Ledger

## 1. 作用域

本台账只回答一件事：当前主线有哪些论文可引用的图、表、result-pack 或边界材料，它们各自绑定到哪些 task / review / run / asset，以及安全表述边界是什么。

它不是论文正文，也不是新实验报告。

状态定义：

- `ready`：现有材料已足够做 bounded paper-facing 引用，不需要新实验。
- `partial`：已有真实来源，但最终图表形态、脚本冻结度或叙事边界仍不足，引用时必须保守。
- `blocked`：当前证据层级不支持把该项写成已完成图表/结果。

## 2. Ledger

| ID | 项目 | 状态 | 当前可直接引用的实体 | 直接来源 | 生成/回填路径 | 安全边界 |
| --- | --- | --- | --- | --- | --- | --- |
| `F1` | `seed=20260429` 单 seed 机制诊断图 | `partial` | `runs/T38_seed20260429_trace_probe_20260513/trace_export/trace_rows.csv` | `docs/evidence_packs/mechanism_ablation/seed20260429_trace_export_diagnosis.md`<br>`docs/review/T38_review.md` | 需要基于 `trace_rows.csv` 手工或 task-local 画图 | 只支持单 seed trace diagnosis；不能把它写成 multi-seed causal proof，也不能拿它支持“high committed-b is harmful”泛化结论 |
| `F2` | 证据等级/边界示意图（software-HIL / frozen benchmark / `.tflite` / real-board gate） | `ready` | `docs/03_hil_p4_boundary_audit.md` | `docs/protocols/benchmark/P4_benchmark_formal_protocol.md`<br>`docs/evidence_packs/deployment_boundary/t48_true_tflite_runtime_gate.md`<br>`docs/evidence_packs/deployment_boundary/t72_real_board_transfer_pack_provenance_hardening.md` | 直接从边界文档整理成 schematic figure | 这是边界图，不是“部署成功图” |
| `F3` | 暗示跨平台训练/部署闭环的 portability 大图 | `blocked` | 无 | `docs/evidence_packs/training_reproducibility/training_reproducibility_and_material_regeneration_pack.md`<br>`docs/evidence_packs/deployment_boundary/t48_true_tflite_runtime_gate.md`<br>`docs/evidence_packs/deployment_boundary/t72_real_board_transfer_pack_provenance_hardening.md` | 当前无诚实成图路径 | `T50/T48/T72` 仍是分层边界证据，不支持把训练、`.tflite`、real-board 串成统一闭环图 |
| `FR1` | 系统结构图（fast loop / slow loop / param bank / gate hierarchy） | `partial` | 源码结构与阶段文档 | `docs/02_experiment_plan.md`<br>`docs/03_hil_p4_boundary_audit.md` | 需要手工整理为架构图 | 可以画概念结构，但必须显式标注 mock-backed software-HIL 与 gate/provenance 边界 |
| `FR4` | T24 四场景结果汇总图 | `partial` | `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/comparison.csv` | `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/summary.json`<br>`docs/review/T24_review.md` | 从 `comparison.csv` 出图；当前没有冻结的 paper-facing 绘图脚本 | 只支持 frozen-set software-HIL 排名，不支持 expanded benchmark retelling |
| `FR6` | 六 seed 机制/干预图包 | `ready` | `docs/figure_assets/T58_fr6_multi_seed_mechanism_intervention/fr6_multi_seed_mechanism_intervention.svg` | `docs/figure_assets/T58_fr6_multi_seed_mechanism_intervention/fr6_multi_seed_mechanism_intervention.png`<br>`docs/figure_assets/T58_fr6_multi_seed_mechanism_intervention/figure_data.csv`<br>`docs/figure_assets/T58_fr6_multi_seed_mechanism_intervention/figure_manifest.json`<br>`docs/evidence_packs/mechanism_ablation/fr6_multi_seed_mechanism_intervention_figure_pack.md`<br>`docs/review/T58_review.md` | `docs/figure_assets/T58_fr6_multi_seed_mechanism_intervention/build_figure.py` | 只支持 descriptive multi-seed figure pack；不能升级 `C4` 为 causal closure |
| `FR7` | 冻结 T24 lane 的 feature/teacher ablation 表 | `ready` | `runs/p4_benchmark/T57_fr7_feature_teacher_ablation_20260524_000000/summary_pack/table.csv` | `runs/p4_benchmark/T57_fr7_feature_teacher_ablation_20260524_000000/summary_pack/summary.json`<br>`runs/p4_benchmark/T57_fr7_feature_teacher_ablation_20260524_000000/provenance_manifest.json`<br>`docs/evidence_packs/mechanism_ablation/fr7_feature_teacher_ablation_reexecution.md`<br>`docs/review/T57_review.md` | 直接引用 `summary_pack/table.csv` 或转成 paper table | 只支持 bounded frozen-set ablation；不能写成 teacher design 必要性或机制闭环 |
| `FR8` | `statcalib` extension-lane closure / no-promotion 素材 | `partial` | `docs/evidence_packs/statcalib_fr8/fr8_statcalib_bounded_closure_pack.md` | `docs/review/T70_review.md`<br>`runs/p4_benchmark/T64_fr8_statcalib_extension_lane_20260527_221658/summary.json`<br>`runs/p4_benchmark/T66_statcalib_sensitivity_20260529_210906/statcalib_sensitivity_summary/summary.json`<br>`runs/p4_benchmark/T67_statcalib_teacher_anchor_20260601_225718/statcalib_teacher_anchor_summary/summary.json`<br>`runs/p4_benchmark/T68_statcalib_generated_only_20260605_205723/statcalib_generated_only_summary/summary.json`<br>`runs/p4_benchmark/T69_statcalib_clean_winner_tiebreak_20260608_160358/statcalib_clean_winner_tiebreak_summary/summary.json` | 需要在 paper 中手工整理成 extension-lane table | 当前只支持 `statcalib` 作为 separately labeled extension lane；必须携带 persistent tie / no-promotion / no-unique-threshold 边界 |
| `FR12` | latency / commit / violation 汇总表 | `partial` | `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/comparison.csv` | `runs/p4_benchmark/T57_fr7_feature_teacher_ablation_20260524_000000/comparison.csv`<br>`runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/summary.json`<br>`runs/p4_benchmark/T57_fr7_feature_teacher_ablation_20260524_000000/summary.json` | 需要额外整理字段与表头 | 只支持 software-HIL 观察表，不支持真板 latency/commit 语义 |
| `T1` | T24 frozen main ranked table | `ready` | `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/comparison.csv` | `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/summary.json`<br>`docs/review/T24_review.md` | 直接从 `comparison.csv` 回填 paper table | 这是 authoritative frozen table；不得被 `statcalib` extension lane 改写 |
| `T2` | benchmark evidence-level / boundary table | `ready` | `docs/03_hil_p4_boundary_audit.md` | `docs/protocols/benchmark/P4_benchmark_formal_protocol.md`<br>`docs/review/T25_p4_formal_evidence_gate_review.md` | 直接整理边界文档 | 最安全的 paper-facing 表之一，因为它明确写出“不支持什么” |
| `T3` | training/material reproducibility boundary table | `ready` | `artifacts/t50_training_repro_pack/training_reproducibility_pack.json` | `docs/evidence_packs/training_reproducibility/training_reproducibility_and_material_regeneration_pack.md`<br>`docs/review/T50_review.md` | 直接引用 pack JSON 与主报告分类表 | 只支持 canonical chain + bounded CPU-only rerun；不支持 full reproducibility / GPU / Linux portability |
| `T4` | isolated true `.tflite` runtime boundary table | `ready` | `artifacts/t48_true_tflite_runtime_gate/t48_true_tflite_runtime_gate.json` | `docs/evidence_packs/deployment_boundary/t48_true_tflite_runtime_gate.md`<br>`docs/review/T48_review.md` | 直接引用 gate JSON、eval/validate reports | 只支持 current-host isolated runtime；不支持 default-env / HIL / deployment closure |
| `T5` | real-board gate / regeneration / provenance boundary table | `ready` | `artifacts/t49_real_board_smoke_execution_gate/t49_real_board_smoke_execution_gate.json` | `artifacts/t71_real_board_gate_regeneration_pack/current_host_regenerated_gate.json`<br>`docs/evidence_packs/deployment_boundary/t49_real_board_smoke_execution_gate.md`<br>`docs/evidence_packs/deployment_boundary/t71_real_board_gate_regeneration_pack.md`<br>`docs/evidence_packs/deployment_boundary/t72_real_board_transfer_pack_provenance_hardening.md`<br>`docs/review/T72_review.md` | 直接整理 `T49/T71/T72` 的 gate/provenance facts | 只支持 read-only gate/provenance boundary 与 current-host `NO_GO`；不支持 real-board execution success |

## 3. 当前最值得直接回填到论文材料的项目

优先级最高、最不容易越界的项目：

1. `T1`：T24 frozen ranked table
2. `T2`：benchmark evidence-level table
3. `T3`：training/material boundary table
4. `T4`：isolated `.tflite` runtime boundary table
5. `T5`：real-board gate/provenance boundary table
6. `FR6`：multi-seed mechanism/intervention figure pack
7. `FR7`：feature/teacher ablation table

当前仍需要明显 caveat 的项目：

- `F1`
- `FR4`
- `FR8`
- `FR12`
