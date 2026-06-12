# Paper Claim/Evidence Ledger

## 1. 作用域与非 claim

本台账只记录当前仓库在 `Phase 2: Controlled Development` 边界下，论文里可以诚实回指到哪些 claim，以及每条 claim 的直接证据路径。

它不是论文正文，也不是新实验报告。它不会：

- 把 draft/prose 当成实验事实；
- 把 mock-backed software-HIL 升格成 `.tflite` HIL 或 real-board validation；
- 把 bounded 机制证据写成 causal proof；
- 把 `T50` 写成 full training reproducibility；
- 把 `T70` 写成 mature `statcalib` comparator promotion。

状态定义：

- `supported`：当前已有直接、具体、可回指的 bounded 证据；
- `partial`：已有真实证据，但 paper wording 必须明显收窄；
- `blocked`：当前证据层级不支持把该 claim 写成已完成事实。

## 2. Claim Ledger

| ID | Claim area | 状态 | 直接证据路径 | 安全表述边界 | Open risk / blocker |
| --- | --- | --- | --- | --- | --- |
| `C1` | 已恢复一条有界 `P3` software-HIL 路径并完成复验 | `supported` | `docs/03_hil_p4_boundary_audit.md`<br>`runs/hil_suite/hardware_hil_recovery_smoke_20260508_172221_3ae9f9176104/hil_summary.json`<br>`runs/hil_suite/hardware_hil_recovery_smoke_20260508_172232_3ae9f9176104/hil_summary.json` | 只能写成 `mock-backed software HIL`，并保留 `artifact_npz + inproc` 边界；不能写成 `.tflite` runtime 或 real-board HIL | `R8` |
| `C2` | `T24` 冻结四场景、五模式、`paired_seeds + repeats=2` 的 formal software revalidation 已完成 | `supported` | `docs/protocols/benchmark/P4_benchmark_formal_protocol.md`<br>`runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/summary.json`<br>`runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/comparison.csv`<br>`docs/review/T24_review.md`<br>`docs/review/T25_p4_formal_evidence_gate_review.md` | 只能写成 frozen-set formal software revalidation，且仍位于 mock-backed software-HIL 内 | `R5`, `R9` |
| `C3` | 在该 frozen set 内，`hybrid_residual_b` 是四个场景的 winner | `supported` | `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/comparison.csv`<br>`runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/summary.json`<br>`docs/review/T24_review.md` | 排名 claim 仅限 `T24` 冻结集合；不能外推到 expanded benchmark、`.tflite` 或 real-board | `R5`, `R9` |
| `C4` | `seed=20260429` failure 已被缩窄为 trace-supported committed-`b` instability story；`FR6` 只把这条证据推进为 descriptive multi-seed figure pack | `partial` | `docs/evidence_packs/mechanism_ablation/seed20260429_failure_diagnosis.md`<br>`docs/evidence_packs/mechanism_ablation/seed20260429_trace_export_diagnosis.md`<br>`runs/T38_seed20260429_trace_probe_20260513/trace_export/trace_rows.csv`<br>`docs/evidence_packs/mechanism_ablation/multi_seed_trace_generalization_probe.md`<br>`docs/evidence_packs/mechanism_ablation/multi_seed_i1_intervention_probe.md`<br>`docs/evidence_packs/mechanism_ablation/post_t55_mechanism_claim_reframing_gate.md`<br>`docs/evidence_packs/mechanism_ablation/fr6_multi_seed_mechanism_intervention_figure_pack.md`<br>`docs/review/T58_review.md` | 最安全写法是：instability pattern 在六 seed 里广泛出现，但 `I1` intervention 结果 mixed/mostly harmful，因此只能保留 descriptive mechanism story；不能写 causal proof、mitigation success 或 “high committed-b is harmful” 泛化命题 | `R10` |
| `C5` | 当前仓库已有 code-backed training/material regeneration pack，且 clean CPU-only lane 已完成一次 bounded train+eval rerun | `supported` | `artifacts/t50_training_repro_pack/training_reproducibility_pack.json`<br>`docs/evidence_packs/training_reproducibility/training_reproducibility_and_material_regeneration_pack.md`<br>`docs/review/T50_review.md` | 可以写 canonical chain 仍在、主线引用未漂、clean CPU-only rerun 已真实完成；不能写成 canonical training quality 被完整重现 | `R11` |
| `C6` | 训练链已经完成跨 host / cross-OS / GPU/CUDA / repeated-run full reproducibility | `blocked` | `docs/evidence_packs/training_reproducibility/training_reproducibility_and_material_regeneration_pack.md`<br>`docs/review/T50_review.md`<br>`docs/08_risks_and_open_questions.md` | `T50` 只支持 bounded pack 与 bounded rerun，不支持 portability / repeated-run closure | `R11` |
| `C7` | 当前机器已确认一条 isolated true `.tflite` runtime 窄路径，可对选定 preserved float / int8 `.tflite` 做真实执行与一致性校验 | `supported` | `artifacts/t48_true_tflite_runtime_gate/t48_true_tflite_runtime_gate.json`<br>`docs/evidence_packs/deployment_boundary/t48_true_tflite_runtime_gate.md`<br>`docs/review/T48_review.md` | 最安全写法是：`current-host isolated true runtime verified for selected preserved float/int8 artifacts`；必须显式区分 true `.tflite` 与 `.tflite.json` stub | `R12` |
| `C8` | 默认环境恢复、`.tflite` HIL closure 或 deployment closure 已完成 | `blocked` | `docs/evidence_packs/deployment_boundary/t48_true_tflite_runtime_gate.md`<br>`docs/review/T48_review.md`<br>`docs/08_risks_and_open_questions.md` | `T48` 不支持 default-env / HIL / deployment closure retelling | `R12` |
| `C9` | 仓库已有 checked-in、只读、role-aware、可 replay / regeneration 的 real-board gate / transfer-pack provenance 包；current-host verdict 仍为 `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE` | `supported` | `artifacts/t49_real_board_smoke_execution_gate/t49_real_board_smoke_execution_gate.json`<br>`artifacts/t71_real_board_gate_regeneration_pack/current_host_regenerated_gate.json`<br>`docs/evidence_packs/deployment_boundary/t49_real_board_smoke_execution_gate.md`<br>`docs/evidence_packs/deployment_boundary/t71_real_board_gate_regeneration_pack.md`<br>`docs/evidence_packs/deployment_boundary/t72_real_board_transfer_pack_provenance_hardening.md`<br>`docs/review/T72_review.md` | 只能写成 read-only gate / regeneration / provenance boundary；不能写成 real-board execution success | `R32`, `R33` |
| `C10` | real-board execution / validation 已完成 | `blocked` | `docs/evidence_packs/deployment_boundary/t49_real_board_smoke_execution_gate.md`<br>`docs/evidence_packs/deployment_boundary/t72_real_board_transfer_pack_provenance_hardening.md`<br>`docs/review/T72_review.md`<br>`docs/08_risks_and_open_questions.md` | 当前 host 仍是 `NO_GO`；future-host provenance 也还不是 fully clean execution-ready 标准入口 | `R13`, `R14`, `R32`, `R33` |
| `C11` | `statcalib` 已形成 bounded mock-backed software-HIL extension-lane closure，并且 `T70` 已明确 `no_promotion_keep_extension_lane_only` / `future_selection_task_required` | `supported` | `docs/evidence_packs/statcalib_fr8/fr8_statcalib_bounded_closure_pack.md`<br>`docs/review/T70_review.md` | 可以写成 separately labeled extension lane 与 no-promotion gate；不能写成 mature comparator、T24 替代表或唯一 clean threshold | `R24` |
| `C12` | 当前仓库证据已经足以支撑 paper-grade expanded benchmark claim | `blocked` | `docs/review/T25_p4_formal_evidence_gate_review.md`<br>`docs/evidence_packs/mechanism_ablation/fr7_feature_teacher_ablation_reexecution.md`<br>`docs/evidence_packs/deployment_boundary/t48_true_tflite_runtime_gate.md`<br>`docs/evidence_packs/deployment_boundary/t72_real_board_transfer_pack_provenance_hardening.md`<br>`docs/evidence_packs/statcalib_fr8/fr8_statcalib_bounded_closure_pack.md` | 当前证据仍是分层 bounded chain：frozen software-HIL、机制/ablation、training/material、isolated `.tflite`、real-board gate、`statcalib` extension lane | `R5`, `R9`, `R10`, `R11`, `R12`, `R24`, `R32`, `R33`, `R15`, `R16` |

## 3. 论文写作 guardrails

1. 把 `hardware validated` 替换成 `mock-backed software-HIL revalidation`，除非引用的是未来真实板级执行产物。
2. 把 `TFLite deployed` 替换成 `isolated current-host true .tflite runtime verified for selected preserved artifacts`，除非后续补齐 default-env / portability / HIL/board integration 证据。
3. 把 `training reproducible` 替换成 `canonical material chain intact + one clean CPU-only bounded rerun completed`，除非后续补齐 repeated-run / cross-host / GPU/Linux 证据。
4. 把 `mechanism proven` 替换成 `trace-supported and multi-seed descriptive mechanism evidence with mixed intervention outcomes`。
5. 把 `statcalib comparator` 拆成两层写：`bounded extension lane` 与 `no-promotion gate`；不要压成一句“新 comparator 已验证”。
6. 把 `real-board ready` 替换成 `checked-in read-only gate/provenance pack with current-host NO_GO verdict`。
