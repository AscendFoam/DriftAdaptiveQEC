# T74 Paper Figure / Caption Pack

## 1. 任务边界

本文件不给任何新实验“配新故事”。它只把已经存在、并且已经被 `T24/T48/T50/T57/T58/T70/T72` 接受的结果，整理成 paper-facing 的 figure/table/supplement caption 草案。

状态含义：

- `ready`：现有证据已足够支撑 bounded paper-facing caption。
- `partial`：有真实来源，但最终图表形态或人工整理步骤还没冻结。
- `blocked`：当前证据层级不支持诚实出图；只能保留“不要写”的边界说明。

## 2. Stable ID Catalog

| ID | kind | 标题 | 状态 | 放置层级 |
| --- | --- | --- | --- | --- |
| `T74-FIG-01` | `figure` | `T24` 冻结四场景结果汇总图 | `partial` | `main text` |
| `T74-FIG-02` | `figure` | `FR6` 六 seed 机制/干预图 | `ready` | `main text` |
| `T74-FIG-03` | `figure` | 证据等级/边界示意图 | `ready` | `appendix` |
| `T74-FIG-04` | `figure` | training-to-deployment portability 大闭环图 | `blocked` | `supplement only` |
| `T74-TBL-01` | `table` | `T24` 冻结四场景主结果表 | `ready` | `main text` |
| `T74-TBL-02` | `table` | `FR7` feature/teacher ablation 表 | `ready` | `appendix` |
| `T74-TBL-03` | `table` | `FR6` 六 seed 描述性汇总表 | `ready` | `appendix` |
| `T74-TBL-04` | `table` | training/material reproducibility boundary 表 | `ready` | `appendix` |
| `T74-TBL-05` | `table` | isolated true `.tflite` runtime boundary 表 | `ready` | `appendix` |
| `T74-TBL-06` | `table` | real-board gate / regeneration / provenance boundary 表 | `ready` | `supplement only` |
| `T74-TBL-07` | `table` | `FR8 statcalib` extension-lane closure 表 | `partial` | `supplement only` |
| `T74-SUP-01` | `supplement-note` | `FR8` no-promotion / no-unique-threshold 边界说明 | `ready` | `supplement only` |
| `T74-SUP-02` | `supplement-note` | `T50` full reproducibility exclusion 说明 | `ready` | `supplement only` |
| `T74-SUP-03` | `supplement-note` | `T48` default-env / deployment closure exclusion 说明 | `ready` | `supplement only` |
| `T74-SUP-04` | `supplement-note` | `T72` future-host / hardware-condition gap 说明 | `partial` | `supplement only` |

## 3. Figure Captions

### `T74-FIG-01` `T24` 冻结四场景结果汇总图

- 状态：`partial`
- 放置层级：`main text`
- 推荐标题：
  - `Frozen Four-Scenario Benchmark Summary Under the Locked T24 Protocol`
- Caption 草案：
  - `在锁定的四场景、五模式、paired-seeds+repeats=2 的 formal software revalidation 中，hybrid_residual_b 在四个冻结场景均取得最低 mean final LER，ukf 在四个场景均为 runner-up。该图若被绘制，只表示 mock-backed software-HIL 内的 frozen-set ranking，不外推到 expanded benchmark、.tflite 或 real-board。若当前稿件不单独出图，可直接用 T74-TBL-01 作为 authoritative substitute。`
- 核心信息：
  - 主文可视化强调 `T24` 冻结集合里的 winner / runner-up 排名关系。
- 直接证据：
  - `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/comparison.csv`
  - `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/summary.json`
  - `docs/review/T24_review.md`
- gap：
  - 当前没有冻结到治理文档里的 paper-facing 绘图脚本；若要成图，需要人工或 task-local 绘图，但 caption 口径已经可锁定。

### `T74-FIG-02` `FR6` 六 seed 机制/干预图

- 状态：`ready`
- 放置层级：`main text`
- 推荐标题：
  - `Six-Seed Mechanism and Intervention Summary`
- Caption 草案：
  - `图中汇总了 T54/T55 已建立的六 seed bounded mechanism/intervention evidence。Panel A 展示各 seed 上 mean(Gated v5) - mean(Full) 的 baseline gap；负值表示 Gated v5 优于 Full。Panel B 展示各 seed 上 mean(I1) - mean(Gated v5 baseline) 的 intervention delta；正值表示降低 clip 的 I1 干预劣于原始 Gated v5 baseline。该图是 descriptive figure：它说明 instability pattern 在多个 seed 上广泛存在，而本次 lower-clip intervention 的结果是 mixed 且多数更差；它不是 causal proof、mechanism closure、expanded benchmark、.tflite 验证或 real-board 验证。`
- 核心信息：
  - 可作为主文里唯一一张机制/干预图，但必须保留 descriptive-only 边界。
- 直接证据：
  - `docs/figure_assets/T58_fr6_multi_seed_mechanism_intervention/fr6_multi_seed_mechanism_intervention.svg`
  - `docs/figure_assets/T58_fr6_multi_seed_mechanism_intervention/fr6_multi_seed_mechanism_intervention.png`
  - `docs/figure_assets/T58_fr6_multi_seed_mechanism_intervention/figure_data.csv`
  - `docs/figure_assets/T58_fr6_multi_seed_mechanism_intervention/caption.md`
  - `docs/evidence_packs/mechanism_ablation/fr6_multi_seed_mechanism_intervention_figure_pack.md`
  - `docs/review/T58_review.md`
- gap：
  - 无必须补的新实验 gap；当前只需复用现有图包。

### `T74-FIG-03` 证据等级/边界示意图

- 状态：`ready`
- 放置层级：`appendix`
- 推荐标题：
  - `Layered Evidence Boundary for Simulation, Runtime, and Real-Board Gates`
- Caption 草案：
  - `本示意图把当前主线 paper-facing 证据分为四层：mock-backed software-HIL、T24 frozen benchmark、isolated current-host true .tflite runtime，以及 read-only real-board gate / regeneration / provenance boundary。图的用途是解释哪些证据位于哪一层、哪些结论不能跨层升级；它不是 deployment success figure，也不是统一 portability closure 图。`
- 核心信息：
  - 这张图服务于“诚实说明证据边界”，而不是展示性能提升。
- 直接证据：
  - `docs/03_hil_p4_boundary_audit.md`
  - `docs/protocols/benchmark/P4_benchmark_formal_protocol.md`
  - `docs/evidence_packs/deployment_boundary/t48_true_tflite_runtime_gate.md`
  - `docs/evidence_packs/deployment_boundary/t72_real_board_transfer_pack_provenance_hardening.md`
- gap：
  - 无新实验 gap；只需要按边界文档手工整理成 schematic。

### `T74-FIG-04` training-to-deployment portability 大闭环图

- 状态：`blocked`
- 放置层级：`supplement only`
- 推荐标题：
  - `Do Not Author: Unified Portability Closure Figure`
- Caption 草案：
  - `当前证据不支持把 training/material、isolated true .tflite runtime 和 real-board gate/provenance 串成一张“统一 portability 或 deployment closure 大图”。如确需说明该缺口，应直接写明这是一张当前不能诚实出图的 blocked item，而不是用视觉整合去暗示闭环已成立。`
- 核心信息：
  - 这是一个必须显式禁止的图形位，不是待补一张图就能解决的缺项。
- 直接证据：
  - `docs/evidence_packs/training_reproducibility/training_reproducibility_and_material_regeneration_pack.md`
  - `docs/evidence_packs/deployment_boundary/t48_true_tflite_runtime_gate.md`
  - `docs/evidence_packs/deployment_boundary/t72_real_board_transfer_pack_provenance_hardening.md`
  - `docs/paper_materials/paper_result_figure_ledger.md`
- gap：
  - 缺的不是画图脚本，而是诚实的一体化证据路径。

## 4. Table Captions

### `T74-TBL-01` `T24` 冻结四场景主结果表

- 状态：`ready`
- Caption 草案：
  - `锁定 T24 协议下四个冻结场景的最终排名。hybrid_residual_b 在四个场景均为 winner，ukf 在四个场景均为 runner-up。该表只表示 mock-backed software-HIL 里的 frozen-set ranking。`
- 核心信息：
  - 这是主文最稳的 simulation result table。
- 直接证据：
  - `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/comparison.csv`
  - `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/summary.json`
  - `docs/review/T24_review.md`
- gap：
  - 无。

### `T74-TBL-02` `FR7` feature/teacher ablation 表

- 状态：`ready`
- Caption 草案：
  - `锁定 T24 lane 的 bounded feature/teacher ablation 汇总。hybrid_no_teacher_params 在四场景均值上优于 hybrid_full 与 ukf，说明当前 frozen set 下 teacher/feature 配置的主效应值得报告；该表不支持 teacher design 必要性或机制闭环结论。`
- 核心信息：
  - `FR7` 可以稳定进 paper，但应作为 bounded ablation 使用。
- 直接证据：
  - `runs/p4_benchmark/T57_fr7_feature_teacher_ablation_20260524_000000/summary_pack/table.csv`
  - `runs/p4_benchmark/T57_fr7_feature_teacher_ablation_20260524_000000/summary_pack/summary.json`
  - `docs/evidence_packs/mechanism_ablation/fr7_feature_teacher_ablation_reexecution.md`
  - `docs/review/T57_review.md`
- gap：
  - 无。

### `T74-TBL-03` `FR6` 六 seed 描述性汇总表

- 状态：`ready`
- Caption 草案：
  - `对 FR6 六 seed 图包的表格化摘要。baseline gap 与 I1 intervention delta 的多 seed 结果表明 instability pattern 广泛存在，但 intervention 结果是 mixed 且多数 harmful；该表只服务于 descriptive cross-seed reading。`
- 核心信息：
  - 供附录查数，和 `T74-FIG-02` 互相补充。
- 直接证据：
  - `docs/figure_assets/T58_fr6_multi_seed_mechanism_intervention/figure_data.csv`
  - `docs/figure_assets/T58_fr6_multi_seed_mechanism_intervention/caption.md`
  - `docs/review/T58_review.md`
- gap：
  - 无。

### `T74-TBL-04` training/material reproducibility boundary 表

- 状态：`ready`
- Caption 草案：
  - `training/material boundary 摘要：当前仓库保留了 canonical static_theta_v2 与 runtime_b_residual_v1 的材料链，并已在 clean Windows/Python 3.12 CPU-only lane 上完成一次 bounded train+eval rerun。该表不支持 full reproducibility、GPU/CUDA portability、Linux portability、.tflite correctness 或 real-board validation。`
- 核心信息：
  - 给 paper 一个诚实的 training/material 说明入口。
- 直接证据：
  - `artifacts/t50_training_repro_pack/training_reproducibility_pack.json`
  - `docs/evidence_packs/training_reproducibility/training_reproducibility_and_material_regeneration_pack.md`
  - `docs/review/T50_review.md`
- gap：
  - 无。

### `T74-TBL-05` isolated true `.tflite` runtime boundary 表

- 状态：`ready`
- Caption 草案：
  - `isolated true .tflite runtime boundary 摘要：当前机器在 .venvs/t48_tf221 + tensorflow==2.21.0 的隔离环境中，已对选定 preserved float/int8 .tflite 完成真实执行与一致性校验，最终 gate verdict 为 GO_TRUE_TFLITE_RUNTIME_FLOAT_AND_INT8。该表不支持默认环境恢复、HIL closure、deployment closure 或跨环境 portability。`
- 核心信息：
  - paper 可以诚实引用 true runtime 已存在，但不能过度解读。
- 直接证据：
  - `artifacts/t48_true_tflite_runtime_gate/t48_true_tflite_runtime_gate.json`
  - `docs/evidence_packs/deployment_boundary/t48_true_tflite_runtime_gate.md`
  - `docs/review/T48_review.md`
- gap：
  - 无。

### `T74-TBL-06` real-board gate / regeneration / provenance boundary 表

- 状态：`ready`
- Caption 草案：
  - `real-board gate / regeneration / provenance boundary 摘要：仓库已有 checked-in、只读、role-aware、可 replay / regeneration 的 real-board gate 包；current-host replay 与 regeneration 均稳定保持 NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE。该表只支持 gate/provenance 边界，不支持 real-board execution success、hardware validation 或 deployment closure。`
- 核心信息：
  - 给 reviewer 一条完整但不夸大的 deployment-adjacent 事实链。
- 直接证据：
  - `artifacts/t49_real_board_smoke_execution_gate/t49_real_board_smoke_execution_gate.json`
  - `artifacts/t71_real_board_gate_regeneration_pack/current_host_regenerated_gate.json`
  - `artifacts/t72_real_board_transfer_pack_provenance_hardening/replay_vs_regeneration_comparison.json`
  - `docs/review/T72_review.md`
- gap：
  - 无新实验 gap；真实硬件执行仍需等未来 host/device 条件。

### `T74-TBL-07` `FR8 statcalib` extension-lane closure 表

- 状态：`partial`
- Caption 草案：
  - `statcalib extension-lane closure 摘要：T24 仍是 authoritative frozen main table；statcalib 只是一条 separately labeled extension lane；当前最强 clean answer 是 window_variance_t001 = t003 = t005 的 persistent clean tie set，gate verdict 仍为 no_promotion_keep_extension_lane_only 与 future_selection_task_required。`
- 核心信息：
  - 这张表只能作为 supplement-only 的 extension-lane closure。
- 直接证据：
  - `docs/evidence_packs/statcalib_fr8/fr8_statcalib_bounded_closure_pack.md`
  - `docs/review/T70_review.md`
  - `runs/p4_benchmark/T64_fr8_statcalib_extension_lane_20260527_221658/summary.json`
  - `runs/p4_benchmark/T69_statcalib_clean_winner_tiebreak_20260608_160358/statcalib_clean_winner_tiebreak_summary/summary.json`
- gap：
  - 现有 closure pack 已足够支撑 wording，但最终 paper 表格仍需人工按 supplement-only 版式整理。

## 5. Supplement Notes

### `T74-SUP-01` `FR8` no-promotion / no-unique-threshold 说明

- 状态：`ready`
- 插入说明：
  - `statcalib` 相关所有正文和补充材料都必须显式保留 `no_promotion_keep_extension_lane_only` 与 `future_selection_task_required`。
- 直接证据：
  - `docs/evidence_packs/statcalib_fr8/fr8_statcalib_bounded_closure_pack.md`
  - `docs/review/T70_review.md`

### `T74-SUP-02` `T50` full reproducibility exclusion 说明

- 状态：`ready`
- 插入说明：
  - `training/material` 只能写 canonical chain intact + clean CPU-only bounded rerun，不得写 full reproducibility、GPU/CUDA portability 或 Linux portability。
- 直接证据：
  - `artifacts/t50_training_repro_pack/training_reproducibility_pack.json`
  - `docs/review/T50_review.md`

### `T74-SUP-03` `T48` default-env / deployment closure exclusion 说明

- 状态：`ready`
- 插入说明：
  - `true .tflite runtime` 只能写 isolated current-host verified，不得写 default-env recovered、HIL closure 或 deployment closure。
- 直接证据：
  - `artifacts/t48_true_tflite_runtime_gate/t48_true_tflite_runtime_gate.json`
  - `docs/review/T48_review.md`

### `T74-SUP-04` `T72` future-host / hardware-condition gap 说明

- 状态：`partial`
- 插入说明：
  - `real-board` 相关材料目前只能维持 read-only gate/provenance wording；真实 execution 仍等待 Linux + FPGA host/device 条件，且 future-host 最小 config provenance 精确性还有 `R32` 残余风险。
- 直接证据：
  - `docs/review/T72_review.md`
  - `docs/evidence_packs/deployment_boundary/t72_real_board_transfer_pack_provenance_hardening.md`
- gap：
  - 缺的不是补一句 prose，而是真实硬件条件和更窄的 provenance clean-up。
