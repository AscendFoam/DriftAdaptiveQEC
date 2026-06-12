# T75 Main-Text Results Authoring Pack

## 1. 这是什么

本目录存放 `T75` 生成的 task-scoped 论文 authoring 资产。它的用途只有一个：

- 把 `T74` 已冻结的 `stable-ID` 结果包压缩成可直接放入主文 Results/Appendix 的最终成图与追溯清单。

本目录中的 `T75-FIG-*` 不是新的实验事实，也不是新的主线证据层；它们只是对 `T74-*` 已有条目的 publication-facing 重排与压缩。

## 2. 这不是什么

本目录不承担以下语义：

- 不产生新的 benchmark、training、`.tflite` 或 real-board 结果。
- 不替代 `runs/`、`artifacts/`、`docs/evidence_packs/` 中的原始事实来源。
- 不把 `T48` 写成 default-env / HIL / deployment closure。
- 不把 `T49/T71/T72` 写成 real-board execution success。
- 不把 `FR8` 写成 mature comparator promotion。
- 不把 `T74-FIG-04` 从 `blocked` 改写为可用闭环图。

## 3. 目录内容

| 资产 | T75 ID | 上游 `T74` ID | 用途 |
| --- | --- | --- | --- |
| `t75_fig_m01_t24_frozen_summary.svg` | `T75-FIG-M01` | `T74-TBL-01`, `T74-FIG-01` | 主文冻结主结果图 |
| `t75_fig_m02_fr6_multi_seed_mechanism.svg` | `T75-FIG-M02` | `T74-FIG-02`, `T74-TBL-03` | 主文机制/解释图 |
| `t75_fig_a01_boundary_schematic.svg` | `T75-FIG-A01` | `T74-FIG-03`, `T74-TBL-05`, `T74-TBL-06`, `T74-SUP-03`, `T74-SUP-04`, `T74-FIG-04` | 附录边界说明图 |
| `authoring_manifest.json` | - | 全部 | 作者包主清单，锁定 placement/status/boundary |
| `asset_source_map.csv` | - | 全部 | 逐资产追溯到 `T74` 与更上游证据路径 |

## 4. 如何回链

使用顺序应当固定为：

1. 先看 `authoring_manifest.json`，确认每个 `T75-FIG-*` 的 placement、status 和 boundary note。
2. 再看 `asset_source_map.csv`，追溯对应的 `T74-*` 与原始 source path。
3. 如果需要 authoritative 数值或 gate wording，回到 `T74-TBL-*` / `T74-FIG-*` 以及各自的 primary source。

特别说明：

- `T75-FIG-M01` 只是 `T74-TBL-01` 的可视化压缩；若版面不适合，`T74-TBL-01` 仍是 authoritative substitute。
- `T75-FIG-M02` 只承载 `FR6` 的 descriptive mechanism/intervention reading，不升级为 causal proof。
- `T75-FIG-A01` 明确保留 `T74-FIG-04` 的 blocked slot，用来提醒“不能写成统一 portability/deployment closure”。

### T76 Rendered QA Note

- `2026-06-12` 的 `T76` rendered QA 对 `T75-FIG-M01`、`T75-FIG-M02` 和 `T75-FIG-A01` 做了 presentation-only 微调：
  - `T75-FIG-M01` 缩短了图例横条中的 authoritative-source 说明，并把底部 boundary note 改成双行；
  - `T75-FIG-M02` 把底部 boundary note 改成双行。
  - `T75-FIG-A01` 把三层说明中的长句改成两行，避免真实预览时右侧裁切。
- 上述修改只服务于真实渲染后的可读性修正，不改变任何 stable ID、上游映射、数值来源或证据边界。

## 5. 当前最强可支持说法

- `T75` 已经形成一套 bounded 的主文 Results prose + final figure authoring pack。
- 所有 `T75-FIG-*` 都能显式 trace back 到 `T74` stable IDs 与上游 evidence path。
- 作者在写主文时可以直接复用这些图和对应的 caption/placement 锁定说明，而不需要重新组织结果边界。

## 6. 当前仍不能支持的说法

- `T75` 不是 full-manuscript 完成态。
- `T75` 不提升任何 evidence level。
- `T75` 不意味着 deployment closure、real-board success、或 mature statcalib promotion。
