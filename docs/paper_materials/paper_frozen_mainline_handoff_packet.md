# T89 Frozen Mainline Handoff Packet

## Packet Status

- upstream gate: `T88`
- sole allowed handoff verdict: `GO_FOR_FROZEN_MAINLINE_HANDOFF_ONLY`
- packet purpose: 把当前 frozen-mainline 的唯一入口、允许引用的材料、必须保留的边界口径与明确 non-claims 收紧到单一 handoff 包
- packet non-purpose: 不新增实验、不改写 note、不解锁 blocked surface、不把当前状态写成 submission-ready completed

## Recommended Handoff Entry

后续作者、Captain 或人工维护者如果只想知道“当前主线到底允许怎样被继续维护”，应按以下顺序阅读：

1. `paper_frozen_mainline_handoff_gate.md`
2. `paper_mainline_surface_freeze_manifest.md`
3. `paper_blocked_surface_disclaimer_table.md`
4. `paper_frozen_mainline_source_of_truth_map.md`
5. `paper_postfreeze_change_control.md`
6. `paper_blocked_surface_reentry_conditions.md`

除这条链路外，不应再从旧任务包、单篇 review、孤立 caption pack 或个人记忆中重建“当前主线可以怎么写”。

## Current Frozen Mainline Surfaces

| surface_family | current_allowed_surface | authoritative_anchor | current_writer_rule |
| --- | --- | --- | --- |
| benchmark main result | `T74-TBL-01` / `Table~\\ref{tab:five-mode-benchmark}` | `paper_mainline_surface_freeze_manifest.md` `FZ01` | 主文 benchmark 主呈现以该表为准，不外推成 expanded benchmark、`.tflite` ranking 或 real-board ranking |
| mechanism / ablation reading | `T75-FIG-M02` descriptive reading + appendix numeric companion | `paper_mainline_surface_freeze_manifest.md` `FZ02` | 只保留 descriptive mechanism reading，不写 causal closure、teacher necessity 或 intervention success |
| appendix support bundle | `T74-TBL-02` 到 `T74-TBL-05` | `paper_mainline_surface_freeze_manifest.md` `FZ03` | appendix 只承担 supporting evidence，不升级为主结论扩写 |
| appendix boundary schematic | `T75-FIG-A01` / `T74-FIG-03` appendix-only optional schematic | `paper_mainline_surface_freeze_manifest.md` `FZ04` | 不重写成 unified deployment closure figure，不在主文再造第二套 caption |
| supplement gated bundle | `T74-TBL-06`、`T74-TBL-07`、`T74-SUP-01` 到 `T74-SUP-04` | `paper_mainline_surface_freeze_manifest.md` `FZ05` | supplement 继续是 gated/excluded surface，不是补强主文的后门 |

## Blocked Or Excluded Surfaces That Must Stay Blocked

- `real-board execution / timing / resource`
- `default-env / cross-host .tflite portability`
- `full training reproducibility`
- `FR8/statcalib` mature comparator promotion / unique clean threshold
- expanded benchmark / stronger oracle baseline route
- unified portability / deployment closure figure or prose
- theory-branch content mergeback into current mainline prose

## Boundary Wording That Must Be Preserved

- `T24` 仍是 mainline frozen-set formal software-HIL 历史主锚点，不外推成真板或 deployment closure。
- `FR6/FR7` 仍只可写成 descriptive support，不写成 causal closure。
- `FR8/statcalib` 仍只可写成 extension lane / no-promotion / no unique clean threshold。
- training/material 仍只可写成 canonical chain intact + one clean CPU-only bounded rerun。
- `.tflite` 仍只可写成 isolated current-host true runtime for selected preserved artifacts。
- real-board 仍只可写成 read-only gate / regeneration / provenance with current-host `NO_GO`。
- 当前没有 `Linux + FPGA` 宿主，因此任何 hardware-dependent surface 都只能继续 blocked。
- `GO_FOR_FROZEN_MAINLINE_HANDOFF_ONLY` 只代表 handoff 可以围绕当前 frozen surface 继续维护，不代表 completed submission state。

## Core Citeable Materials

当前主线 handoff 如果需要引用材料，应优先回指以下文档，而不是重新发明描述：

- `paper_frozen_mainline_handoff_gate.md`
- `paper_mainline_surface_freeze_manifest.md`
- `paper_blocked_surface_disclaimer_table.md`
- `paper_frozen_mainline_source_of_truth_map.md`
- `paper_submission_pack_assembly_manifest.md`
- `paper_submission_surface_route_map.md`
- `paper_submission_exclusion_register.md`
- `paper_submission_author_handoff.md`

## Non-Claims

以下内容在 `T89` 之后仍然不得写成既成事实：

- `submission-ready completed`
- `real-board execution succeeded`
- `hardware-ready finalization`
- `default-environment compatibility closed`
- `cross-host .tflite portability closed`
- `full training reproducibility closed`
- `statcalib mature comparator promoted`
- `unique clean threshold established`
- `expanded benchmark ranking proven beyond the frozen set`
- `theory branch already merged back into main`

## Handoff Use Rule

如果后续维护只涉及错字、链接、文件名或不改变事实边界的排版整理，可以按 `paper_postfreeze_change_control.md` 的 `L0` 规则处理。只要修改会影响：

- frozen surface 选择，
- blocked disclaimer wording，
- source-of-truth 回链，
- verdict 口径，
- 或任何 blocked surface 的写法，

就不能把这份 handoff packet 当作“已经授权自由修改”的依据，而必须按 `L1/L2/L3` 分级重新开任务。
