# T74 Paper Submission Material Gap Checklist

## 1. 结论先行

当前主线已经具备一条“simulation/material-complete first”的诚实提交路径：

- `T24` 冻结主结果可以进主文。
- `FR6/FR7` 的机制与 ablation 材料可以进主文或附录。
- `T50/T48` 的 material/runtime boundary 可以进附录。
- `T72` 的 real-board gate/provenance 目前只能停留在补充材料边界位。

这不等于 paper 已经拥有 real-board 主结果，但也不等于“没有真板就完全不能投”。正确口径是：先形成 simulation/material complete route，再把 hardware-dependent surface 明确标成后续层。

## 2. Ready Now

| ID | 状态 | 当前可直接使用的内容 | 直接来源 |
| --- | --- | --- | --- |
| `T74-TBL-01` | `ready` | `T24` 冻结四场景主结果表 | `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/comparison.csv` |
| `T74-FIG-02` | `ready` | `FR6` 六 seed 机制/干预图包 | `docs/figure_assets/T58_fr6_multi_seed_mechanism_intervention/fr6_multi_seed_mechanism_intervention.svg` |
| `T74-TBL-02` | `ready` | `FR7` feature/teacher ablation 表 | `runs/p4_benchmark/T57_fr7_feature_teacher_ablation_20260524_000000/summary_pack/table.csv` |
| `T74-TBL-03` | `ready` | `FR6` 图包对应的表格快照 | `docs/figure_assets/T58_fr6_multi_seed_mechanism_intervention/figure_data.csv` |
| `T74-TBL-04` | `ready` | training/material reproducibility boundary 表 | `artifacts/t50_training_repro_pack/training_reproducibility_pack.json` |
| `T74-TBL-05` | `ready` | isolated true `.tflite` runtime boundary 表 | `artifacts/t48_true_tflite_runtime_gate/t48_true_tflite_runtime_gate.json` |
| `T74-TBL-06` | `ready` | real-board gate / regeneration / provenance boundary 表 | `artifacts/t72_real_board_transfer_pack_provenance_hardening/replay_vs_regeneration_comparison.json` |
| `T74-SUP-01` | `ready` | `FR8` no-promotion/no-unique-threshold 说明 | `docs/review/T70_review.md` |
| `T74-SUP-02` | `ready` | `T50` exclusion note | `docs/review/T50_review.md` |
| `T74-SUP-03` | `ready` | `T48` exclusion note | `docs/review/T48_review.md` |

## 3. Partial But Hardware-Independent

| ID | 状态 | 当前缺口 | 最小补齐动作 | 备注 |
| --- | --- | --- | --- | --- |
| `T74-FIG-01` | `partial` | 缺少冻结到治理文档里的 paper-facing 绘图脚本 | 手工或 task-local 从 `comparison.csv` 出图；如果不画，直接用 `T74-TBL-01` | 不需要新实验 |
| `T74-TBL-07` | `partial` | `FR8` closure pack 已有，但 paper-facing supplement table 还需人工排版 | 把 `T70` closure/gate wording 整理成补充材料表格 | 不需要新 benchmark |
| `T74-SUP-04` | `partial` | future-host 最小 config provenance 精确性仍有 `R32`；真实硬件宿主仍缺 | 在 `T72` 边界说明中保留残余风险；不强行补成完成态 | 这是边界缺口，不是文案缺口 |

## 4. Explicitly Must Not Be Written

| Surface | 不得写成什么 | 为什么 |
| --- | --- | --- |
| `T24` / `T74-TBL-01` | expanded benchmark、`.tflite`、real-board 或 deployment closure 主结果 | `T24` 只代表 frozen-set mock-backed software-HIL |
| `FR6` / `FR7` | causal proof、teacher design 必要性、机制闭环 | `T58/T57` 都只支持 descriptive / bounded ablation wording |
| `T50` | full reproducibility、GPU/CUDA portability、Linux portability | `T50` 只证明 canonical chain + clean CPU-only bounded rerun |
| `T48` | default-env recovered、HIL closure、deployment closure | `T48` 只支持 isolated current-host true `.tflite` runtime |
| `T72` | real-board execution success、hardware validated、`P3 real-board HIL complete` | 当前 replay/regeneration 仍是 `NO_GO` |
| `FR8` | mature comparator、`T24` 替代表、唯一 clean threshold | `T70` 的 gate 仍是 `no_promotion_keep_extension_lane_only` |
| `T74-FIG-04` | 可以补一张图就讲清 portability/deployment 闭环 | 当前缺的是统一诚实证据，不是缺一张图 |

## 5. Must Wait for Hardware Conditions

| Surface | 为什么必须等硬件 | 当前最强替代表述 |
| --- | --- | --- |
| real-board smoke figure/table | 当前 host 缺少可读打开的真板设备路径，`T49/T71/T72` 均为 `NO_GO` | read-only gate / regeneration / provenance boundary only |
| board-level latency / throughput / HIL execution rows | 没有 `Linux + FPGA` 宿主与真实执行记录 | 不写性能或 execution rows，只保留 gate 边界 |
| any paper claim implying deployment closure | 训练、`.tflite`、real-board 仍是分层边界而非统一闭环 | 用 `T74-FIG-03` 和 `T74-SUP-03/04` 显式说明分层 |

## 6. 推荐提交路径

最稳的首版 paper-facing 组合是：

1. 主文：`T74-TBL-01` + `T74-FIG-02`，必要时再补 `T74-FIG-01`。
2. 附录：`T74-TBL-02` 到 `T74-TBL-05` + `T74-FIG-03`。
3. 补充材料：`T74-TBL-06`、`T74-TBL-07` + `T74-SUP-01` 到 `T74-SUP-04`。

这样可以先把 simulation/material route 收完整，同时继续诚实地把 hardware-dependent surface 放在后续层，而不是把 absence of board 写成“paper currently impossible”。
