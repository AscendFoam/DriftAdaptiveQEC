# T74 Paper-Ready Simulation Result Pack

本目录不是新的实验输出目录，而是 `T74` 的 traceability/packaging 目录。它只保存 paper-facing stable ID、caption/result source map、table snapshot 和 submission bundle manifest。

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `figure_manifest.json` | `T74-TBL-*` / `T74-FIG-*` / `T74-SUP-*` 的统一清单 |
| `result_source_map.csv` | 每个 stable ID 的结果来源、边界和禁止写法 |
| `caption_source_map.csv` | 每个 stable ID 的 caption / note 来源 |
| `table_snapshot.csv` | paper table 候选的最小数值快照 |
| `submission_bundle_manifest.json` | 当前 simulation/material-first 提交包的组成、状态和 blocked item |

## 使用规则

1. 本目录不生成新的 run root，不复制历史 `runs/` / `artifacts/`。
2. 真正的图资产如果已经存在，应继续留在原任务目录，例如：
   - `FR6` 真实图资产仍在 `docs/figure_assets/T58_fr6_multi_seed_mechanism_intervention/`
3. 所有 stable ID 必须和以下文档同步：
   - `docs/paper_materials/paper_simulation_result_table_pack.md`
   - `docs/paper_materials/paper_figure_caption_pack.md`
   - `docs/paper_materials/paper_maintext_insertion_map.md`
   - `docs/paper_materials/paper_submission_material_gap_checklist.md`

## 当前边界

- `T48` 仍只是 isolated current-host true `.tflite` runtime。
- `T72` 仍只是 read-only real-board gate / provenance boundary。
- `FR8` 仍只是 extension-lane closure/no-promotion material。
- `T74-FIG-04` 目前是 blocked item，不得画成统一 portability/deployment closure 图。
