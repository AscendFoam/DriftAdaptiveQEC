# T74 Paper-Ready Simulation Result Table Pack

## 1. 任务边界

`T74` 是 docs-only 的 paper-facing 打包任务。它只重组已经接受的 simulation/material evidence，不新增 benchmark、training、`.tflite` 或 real-board 执行事实。

本文件里的 stable table ID 必须与以下文件保持一致：

- `docs/paper_materials/paper_figure_caption_pack.md`
- `docs/paper_materials/paper_maintext_insertion_map.md`
- `docs/figure_assets/T74_paper_ready_simulation_result_pack/figure_manifest.json`
- `docs/figure_assets/T74_paper_ready_simulation_result_pack/result_source_map.csv`
- `docs/figure_assets/T74_paper_ready_simulation_result_pack/caption_source_map.csv`
- `docs/figure_assets/T74_paper_ready_simulation_result_pack/table_snapshot.csv`
- `docs/figure_assets/T74_paper_ready_simulation_result_pack/submission_bundle_manifest.json`

## 2. Stable Table Catalog

| ID | 标题 | 建议位置 | 状态 | 直接来源 |
| --- | --- | --- | --- | --- |
| `T74-TBL-01` | `T24` 冻结四场景主结果表 | `main text` | `ready` | `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/comparison.csv` |
| `T74-TBL-02` | `FR7` feature/teacher ablation 表 | `appendix` | `ready` | `runs/p4_benchmark/T57_fr7_feature_teacher_ablation_20260524_000000/summary_pack/table.csv` |
| `T74-TBL-03` | `FR6` 六 seed 描述性汇总表 | `appendix` | `ready` | `docs/figure_assets/T58_fr6_multi_seed_mechanism_intervention/figure_data.csv` |
| `T74-TBL-04` | training/material reproducibility boundary 表 | `appendix` | `ready` | `artifacts/t50_training_repro_pack/training_reproducibility_pack.json` |
| `T74-TBL-05` | isolated true `.tflite` runtime boundary 表 | `appendix` | `ready` | `artifacts/t48_true_tflite_runtime_gate/t48_true_tflite_runtime_gate.json` |
| `T74-TBL-06` | real-board gate / regeneration / provenance boundary 表 | `supplement only` | `ready` | `artifacts/t72_real_board_transfer_pack_provenance_hardening/replay_vs_regeneration_comparison.json` |
| `T74-TBL-07` | `FR8 statcalib` extension-lane closure 表 | `supplement only` | `partial` | `docs/evidence_packs/statcalib_fr8/fr8_statcalib_bounded_closure_pack.md` |

## 3. Table Candidates

### `T74-TBL-01` `T24` 冻结四场景主结果表

- 建议位置：`main text`
- 状态：`ready`
- 推荐表头：`scenario`, `winner`, `winner final LER`, `runner-up`, `runner-up final LER`, `coverage`, `notes`
- 直接来源：
  - `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/comparison.csv`
  - `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/summary.json`
  - `docs/review/T24_review.md`
  - `docs/review/T25_p4_formal_evidence_gate_review.md`
- 安全写法：
  - 在冻结四场景、五模式、`paired_seeds + repeats=2` 的 formal software revalidation 中，`hybrid_residual_b` 在四个场景均为 winner，`ukf` 在四个场景均为 runner-up。
  - 这是 mock-backed software-HIL 内的 authoritative frozen ranked table。
- 禁止写法：
  - 不得把该表改写成 expanded benchmark、`.tflite`、real-board 或 deployment closure 结果。
  - 不得让 `FR8 statcalib` extension lane 回写这张主表。
- 快照锚点：`docs/figure_assets/T74_paper_ready_simulation_result_pack/table_snapshot.csv` 中的 `T74-TBL-01` 行。

| scenario | winner | winner final LER | runner-up | runner-up final LER |
| --- | --- | --- | --- | --- |
| `static_bias_theta` | `hybrid_residual_b` | `0.810902` | `ukf` | `0.825370` |
| `linear_ramp` | `hybrid_residual_b` | `0.787755` | `ukf` | `0.811201` |
| `step_sigma_theta` | `hybrid_residual_b` | `0.788800` | `ukf` | `0.811548` |
| `periodic_drift` | `hybrid_residual_b` | `0.806392` | `ukf` | `0.821558` |

### `T74-TBL-02` `FR7` feature/teacher ablation 表

- 建议位置：`appendix`
- 状态：`ready`
- 推荐表头：`mode`, `description`, `avg LER`, `delta vs UKF`, `delta vs Hybrid Full`, `notes`
- 直接来源：
  - `runs/p4_benchmark/T57_fr7_feature_teacher_ablation_20260524_000000/summary_pack/table.csv`
  - `runs/p4_benchmark/T57_fr7_feature_teacher_ablation_20260524_000000/summary_pack/summary.json`
  - `runs/p4_benchmark/T57_fr7_feature_teacher_ablation_20260524_000000/provenance_manifest.json`
  - `docs/evidence_packs/mechanism_ablation/fr7_feature_teacher_ablation_reexecution.md`
  - `docs/review/T57_review.md`
- 安全写法：
  - 在锁定 `T24` 协议的 bounded frozen set 内，`hybrid_no_teacher_params` 的四场景均值优于 `hybrid_full` 与 `ukf`，因此 feature/teacher 配置差异值得在论文中作为 ablation table 报告。
  - 这是 bounded frozen-set ablation，不是 teacher 设计必要性或机制闭环证明。
- 禁止写法：
  - 不得把该表写成“teacher params 一般性有害”或“teacher design 已被因果否证”。
  - 不得把该表升级成 paper-grade expanded benchmark 论证。
- 进入附录而不是主文的原因：
  - 信息密度高，且语义属于 bounded ablation 解释层，优先服务于主文结果后的展开说明。
- 快照锚点：`table_snapshot.csv` 中的 `T74-TBL-02` 行。

| mode | avg LER | delta vs UKF | delta vs Hybrid Full |
| --- | --- | --- | --- |
| `ukf` | `0.817382` | `0.000000` | `0.018837` |
| `hybrid_full` | `0.798545` | `-0.018837` | `0.000000` |
| `hybrid_no_hist_deltas` | `0.826723` | `0.009341` | `0.028178` |
| `hybrid_no_teacher_prediction` | `0.807251` | `-0.010131` | `0.008706` |
| `hybrid_no_teacher_params` | `0.749621` | `-0.067761` | `-0.048924` |
| `hybrid_no_teacher_deltas` | `0.800329` | `-0.017053` | `0.001784` |

### `T74-TBL-03` `FR6` 六 seed 描述性汇总表

- 建议位置：`appendix`
- 状态：`ready`
- 推荐表头：`seed`, `seed category`, `baseline gap gv5-full`, `I1 delta vs baseline`, `I1 verdict`, `notes`
- 直接来源：
  - `docs/figure_assets/T58_fr6_multi_seed_mechanism_intervention/figure_data.csv`
  - `docs/figure_assets/T58_fr6_multi_seed_mechanism_intervention/caption.md`
  - `docs/evidence_packs/mechanism_ablation/fr6_multi_seed_mechanism_intervention_figure_pack.md`
  - `docs/review/T58_review.md`
- 安全写法：
  - 多个 seed 上都能看到 instability pattern，但具体幅度与方向有差异。
  - `I1` 降 clip intervention 的结果是 mixed，且多数 seed 为 harmful；这张表是 descriptive evidence，不是 causal closure。
- 禁止写法：
  - 不得把六 seed 图表写成因果证明、完整机制闭环、expanded benchmark 或真板验证。
- 进入附录而不是主文的原因：
  - 主文优先使用同一证据的图形版 `T74-FIG-02`；表格版保留在附录供 reviewer 查数。
- 快照锚点：`table_snapshot.csv` 中的 `T74-TBL-03` 行。

| seed | category | `mean(Gated v5) - mean(Full)` | `mean(I1) - mean(Gated v5 baseline)` | verdict |
| --- | --- | --- | --- | --- |
| `20260425` | `quiet` | `0.000907` | `0.163289` | `harmful` |
| `20260427` | `classic` | `-0.145352` | `0.287166` | `harmful` |
| `20260428` | `classic` | `-0.078998` | `0.057395` | `harmful` |
| `20260429` | `classic` | `-0.127948` | `0.322245` | `harmful` |
| `20260430` | `classic` | `-0.170777` | `-0.024372` | `mixed_or_no_clear_effect` |
| `20260510` | `universal` | `-0.003953` | `-0.035533` | `helpful` |

### `T74-TBL-04` training/material reproducibility boundary 表

- 建议位置：`appendix`
- 状态：`ready`
- 推荐表头：`surface`, `evidence`, `supported wording`, `unsupported wording`, `notes`
- 直接来源：
  - `artifacts/t50_training_repro_pack/training_reproducibility_pack.json`
  - `docs/evidence_packs/training_reproducibility/training_reproducibility_and_material_regeneration_pack.md`
  - `docs/review/T50_review.md`
- 安全写法：
  - 仓库中存在完整的 canonical `static_theta_v2` 与 `runtime_b_residual_v1` 材料链。
  - 当前 clean Windows/Python 3.12 CPU-only lane 已完成一次 bounded train+eval rerun。
- 禁止写法：
  - 不得把本表写成 full reproducibility、GPU/CUDA portability、Linux portability、`.tflite` correctness 或 real-board validation。
- 进入附录而不是主文的原因：
  - 这是材料和 provenance 支撑层，不是主 simulation result ranking。
- 快照锚点：`table_snapshot.csv` 中的 `T74-TBL-04` 行。

| surface | key fact |
| --- | --- |
| canonical `static_theta_v2` | dataset / float model / train report chain complete；historical int8/export/eval derivatives 可枚举 |
| canonical `runtime_b_residual_v1` | dataset / float model / train report chain complete；仍是 mainline runtime residual anchor |
| bounded rerun | `n_train=2048`，`n_val=512`，`backend=numpy`，`device=cpu`，`test r2_mean=0.860042` |

### `T74-TBL-05` isolated true `.tflite` runtime boundary 表

- 建议位置：`appendix`
- 状态：`ready`
- 推荐表头：`surface`, `verdict`, `selected artifact`, `runtime result`, `boundary`
- 直接来源：
  - `artifacts/t48_true_tflite_runtime_gate/t48_true_tflite_runtime_gate.json`
  - `docs/evidence_packs/deployment_boundary/t48_true_tflite_runtime_gate.md`
  - `docs/review/T48_review.md`
- 安全写法：
  - 当前机器在 `.venvs/t48_tf221 + tensorflow==2.21.0` 的 isolated 环境中，已对选定 preserved float / int8 `.tflite` 完成真实执行与一致性校验。
  - 最终 gate verdict 是 `GO_TRUE_TFLITE_RUNTIME_FLOAT_AND_INT8`。
- 禁止写法：
  - 不得把该表写成默认环境已恢复、`.tflite` HIL closure、deployment closure 或 cross-host portability。
- 进入附录而不是主文的原因：
  - 它是 deployment-adjacent boundary evidence，而不是主 simulation result。
- 快照锚点：`table_snapshot.csv` 中的 `T74-TBL-05` 行。

| surface | key fact |
| --- | --- |
| runtime env | isolated env runtime available；preferred package `tensorflow==2.21.0` |
| float preserved `.tflite` | `executed=True`；`max_abs_diff=0.119340`；`mean_abs_diff=0.006731` |
| int8 preserved `.tflite` | `executed=True`；`max_abs_diff=0.203893`；`mean_abs_diff=0.008712` |

### `T74-TBL-06` real-board gate / regeneration / provenance boundary 表

- 建议位置：`supplement only`
- 状态：`ready`
- 推荐表头：`surface`, `verdict`, `strongest supported statement`, `remaining risk`, `forbidden upgrade`
- 直接来源：
  - `artifacts/t49_real_board_smoke_execution_gate/t49_real_board_smoke_execution_gate.json`
  - `artifacts/t71_real_board_gate_regeneration_pack/current_host_regenerated_gate.json`
  - `artifacts/t72_real_board_transfer_pack_provenance_hardening/replay_vs_regeneration_comparison.json`
  - `docs/evidence_packs/deployment_boundary/t49_real_board_smoke_execution_gate.md`
  - `docs/evidence_packs/deployment_boundary/t71_real_board_gate_regeneration_pack.md`
  - `docs/evidence_packs/deployment_boundary/t72_real_board_transfer_pack_provenance_hardening.md`
  - `docs/review/T72_review.md`
- 安全写法：
  - 仓库已有 checked-in、只读、role-aware、可 replay / regeneration 的 real-board gate/provenance 包。
  - current-host replay 与 regeneration 均稳定保持 `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`。
- 禁止写法：
  - 不得把该表写成 real-board execution success、hardware validated、`P3 real-board HIL complete` 或 deployment closure。
- 进入补充材料而不是主文的原因：
  - 它是 host/device gating truth，不是 simulation 主结果。
- 快照锚点：`table_snapshot.csv` 中的 `T74-TBL-06` 行。

| surface | key fact |
| --- | --- |
| `T49` replay gate | strongest supported statement 仍是“当前机器没有可读打开的真板设备路径” |
| `T71` regeneration gate | regenerated verdict 与 `T49` replay verdict 一致 |
| `T72` provenance hardening | `R31` 已收口；残余风险收窄为 future-host 最小 config provenance 精确性 `R32` |

### `T74-TBL-07` `FR8 statcalib` extension-lane closure 表

- 建议位置：`supplement only`
- 状态：`partial`
- 推荐表头：`surface`, `supported wording`, `unsupported wording`, `gate verdict`, `notes`
- 直接来源：
  - `docs/evidence_packs/statcalib_fr8/fr8_statcalib_bounded_closure_pack.md`
  - `docs/review/T70_review.md`
  - `runs/p4_benchmark/T64_fr8_statcalib_extension_lane_20260527_221658/summary.json`
  - `runs/p4_benchmark/T66_statcalib_sensitivity_20260529_210906/statcalib_sensitivity_summary/summary.json`
  - `runs/p4_benchmark/T67_statcalib_teacher_anchor_20260601_225718/statcalib_teacher_anchor_summary/summary.json`
  - `runs/p4_benchmark/T68_statcalib_generated_only_20260605_205723/statcalib_generated_only_summary/summary.json`
  - `runs/p4_benchmark/T69_statcalib_clean_winner_tiebreak_20260608_160358/statcalib_clean_winner_tiebreak_summary/summary.json`
- 安全写法：
  - `T24` 仍是 authoritative frozen main table。
  - `statcalib` 只是一条 separately labeled extension lane。
  - 当前最强 clean answer 是 `window_variance_t001 = t003 = t005` 的 persistent clean tie set。
  - gate verdict 仍是 `no_promotion_keep_extension_lane_only` 与 `future_selection_task_required`。
- 禁止写法：
  - 不得把该表写成 mature comparator、唯一 clean threshold、`T24` 替代表、`.tflite` lane 或 real-board lane。
- 之所以仍是 `partial`：
  - 证据和 closure/gate 已存在，但 paper-facing 最终表格仍需要人工整理成 extension-lane-only 的补充材料版式。
- 快照锚点：`table_snapshot.csv` 中的 `T74-TBL-07` 行。

| surface | key fact |
| --- | --- |
| frozen anchor | `T24` winner 仍是 `hybrid_residual_b`，runner-up 仍是 `ukf` |
| strongest clean answer | `statcalib_window_variance_t001 = statcalib_window_variance_t003 = statcalib_window_variance_t005` |
| gate | `no_promotion_keep_extension_lane_only`；`future_selection_task_required` |

## 4. 最安全的主文/附录组合

如果当前目标是“先形成 paper-ready simulation/material pack，而不等待真板条件”，最安全的首版组合是：

1. 主文放 `T74-TBL-01`，必要时再补 `T74-FIG-02`。
2. 附录放 `T74-TBL-02` 到 `T74-TBL-05`。
3. 补充材料放 `T74-TBL-06` 与 `T74-TBL-07`，并显式保留 `T72` 与 `T70` 的边界口径。
