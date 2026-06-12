# Paper Claim Risk Table

## 1. 作用域

本表只记录 paper-facing claim area 的当前风险，不替代 `docs/08_risks_and_open_questions.md`。

目标是让论文写作时能快速回答三件事：

1. 这一段话最容易越界在哪里；
2. 应该回指哪个 `R*` 或 review warning；
3. 当前是已收口、仍开放，还是已被更窄的新风险替代。

## 2. Risk Ledger

| ID | Claim area | 当前最安全的 paper-facing 说法 | 风险/来源 | 状态 | 证据锚点 | 写作约束 |
| --- | --- | --- | --- | --- | --- | --- |
| `PR1` | P3 software-HIL / frozen benchmark | `mock-backed software-HIL` 上的 bounded recovery 与 frozen-set formal revalidation 已完成 | `R8`, `R5`, `R9` | `open` | `docs/03_hil_p4_boundary_audit.md`<br>`docs/review/T24_review.md`<br>`docs/review/T25_p4_formal_evidence_gate_review.md` | 不写成 `.tflite` HIL、real-board HIL 或 expanded benchmark |
| `PR2` | 机制诊断 / `FR6` / `FR7` | 当前只支持 descriptive mechanism story 与 bounded ablation reading | `R10`；`T58` `N3` | `open` | `docs/evidence_packs/mechanism_ablation/post_t55_mechanism_claim_reframing_gate.md`<br>`docs/review/T57_review.md`<br>`docs/review/T58_review.md` | 不写 causal proof；不写 “teacher params 必要”；不写 “high committed-b is harmful” 泛化命题 |
| `PR3` | training/material reproducibility | 当前只支持 canonical material chain + clean CPU-only bounded rerun | `R11` | `open` | `docs/evidence_packs/training_reproducibility/training_reproducibility_and_material_regeneration_pack.md`<br>`docs/review/T50_review.md` | 不写 full reproducibility、repeated-run closure、GPU/CUDA portability 或 Linux portability |
| `PR4` | isolated true `.tflite` runtime | 当前机器已有一条 isolated true `.tflite` runtime 窄路径 | `R12`；`T48` non-blocking issues | `open but narrowed` | `docs/evidence_packs/deployment_boundary/t48_true_tflite_runtime_gate.md`<br>`docs/review/T48_review.md` | 不把 `T48` 写成 default-env 恢复、HIL closure、real-board validation 或 deployment closure |
| `PR5` | real-board role-aware gate / regeneration 缺口 | `T71` 已把 current-host gate 做成 checked-in、role-aware、可 replay / regeneration 的只读入口 | `R30` | `closed by T71` | `docs/evidence_packs/deployment_boundary/t71_real_board_gate_regeneration_pack.md`<br>`docs/review/T71_review.md` | 后续不要再把“缺少 role-aware/regeneration”当成当前主风险 |
| `PR6` | transfer-pack provenance 写死默认文案的旧问题 | `T72` 已把 provenance 改为 execution-derived / override-aware，并保持 verdict 不漂移 | `R31` | `closed by T72` | `docs/evidence_packs/deployment_boundary/t72_real_board_transfer_pack_provenance_hardening.md`<br>`docs/review/T72_review.md` | 后续不要再把 `R31` 当作未收口问题 |
| `PR7` | future-host 最小 config 的 path provenance 标签精确性 | `T72` 之后仍有一个更窄的 future-host carry-forward 风险 | `R32`；`T72` `N1/N3` | `open` | `docs/review/T72_review.md`<br>`docs/evidence_packs/deployment_boundary/t72_real_board_transfer_pack_provenance_hardening.md` | 不把 `T72` 写成 fully provenance-clean future-host 标准入口 |
| `PR8` | 缺少 `Linux + FPGA` 硬件宿主 | 当前 real-board execution 仍无进入条件 | `R33` | `open` | `docs/evidence_packs/deployment_boundary/t49_real_board_smoke_execution_gate.md`<br>`docs/evidence_packs/deployment_boundary/t72_real_board_transfer_pack_provenance_hardening.md` | 论文里只能保留 read-only gate/provenance 边界，不把真板写成近期主结果 |
| `PR9` | `statcalib` overclaim | `statcalib` 只是一条 bounded extension lane，且 `T70` 明确给出 `no_promotion_keep_extension_lane_only` | `R24` | `open` | `docs/evidence_packs/statcalib_fr8/fr8_statcalib_bounded_closure_pack.md`<br>`docs/review/T70_review.md` | 不能把 `statcalib` 写成 mature comparator、T24 替代表、`.tflite` lane 或 real-board lane |
| `PR10` | paper-grade expanded benchmark / prose overclaim | 当前主线仍是分层 bounded evidence，不是 submission-ready expanded benchmark closure | `R15`, `R16` | `open` | `docs/02_experiment_plan.md`<br>`docs/review/T25_p4_formal_evidence_gate_review.md`<br>`docs/paper_materials/paper_claim_evidence_ledger.md` | 在 `T73/T74` 之后仍需保持 claim/result/risk 三本账同步，不把 prose 先行写成事实 |

## 3. `T72` 之后最重要的变化

`T72` 对 paper-facing 风险口径的影响只有两条，必须一起保留：

1. `R31` 已收口：默认文案写死、override provenance 不跟执行上下文走的问题已经被 `T72` 修正。
2. `R32` 新开且更窄：future-host 的最小 config 场景下，path provenance 仍不能区分 `config_field_present` 与 `code_default`。

因此，论文里关于 real-board 的 safest wording 现在是：

- 已有 checked-in、read-only、role-aware、可 replay / regeneration 的 gate/provenance 包；
- current-host verdict 仍为 `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`；
- future-host transfer-pack 更干净了，但还不能写成 fully provenance-clean、ready-to-execute 或 real-board success。
