# T75 Appendix Bridge Pack

## 1. 作用

本文件回答一个非常具体的问题：

主文 Results 如果已经由 `T75-FIG-M01`、`T75-FIG-M02` 和 `T74-TBL-01` 锁定，那么其余 `T74` 材料应该如何自然过渡到 appendix / supplement，而不是继续挤进主文。

## 2. 从主文到附录的过渡逻辑

推荐过渡顺序：

1. 主文先完成 `T24` frozen result。
2. 主文再给出 `FR6` descriptive mechanism/intervention layer。
3. 然后用一句 bridge 过渡到附录：
   - `附录进一步提供 frozen-set ablation、cross-seed 数值快照，以及 training/material 与 isolated true runtime 的 supporting boundary tables。`
4. 再在 supplement 中保留 deployment gate 和 `FR8` extension-lane closure。

这条过渡逻辑的核心是：

- 主文讲 `what wins` 与 `how to read it conservatively`
- appendix 讲 `what supports that reading`
- supplement 讲 `what remains bounded, gated, or explicitly not promoted`

## 3. Appendix 建议放置

| 上游 `T74` ID | 建议位置 | 作用 | 为什么不进主文 |
| --- | --- | --- | --- |
| `T74-TBL-02` | `appendix` | `FR7` frozen-set ablation 支撑 | 信息密度高，且只属于 bounded ablation explanation |
| `T74-TBL-03` | `appendix` | `FR6` 的查数版 companion table | 主文已用 `T75-FIG-M02`；附录表供 reviewer 查数 |
| `T74-TBL-04` | `appendix` | training/material provenance 支撑 | 它是 supporting material，不是 main result |
| `T74-TBL-05` | `appendix` | isolated true `.tflite` runtime boundary | 它说明 runtime truth，但不属于主文结果层 |
| `T74-FIG-03` / `T75-FIG-A01` | `appendix` | 分层边界示意图 | 它解释证据边界，而不是结果趋势 |

### 可直接使用的 appendix 过渡句

- `附录 A 给出锁定 frozen set 下的 feature/teacher ablation 与六 seed 数值快照，用以支撑主文中的结果解释。`
- `附录 B 汇总 training/material 与 isolated true runtime 的 supporting boundary tables，说明当前可写到哪一层，而不是暗示部署闭环已经成立。`
- `附录图进一步把 frozen benchmark、true runtime 与 real-board gate 的分层关系固定下来。`

## 4. Supplement-Only 保留逻辑

| 上游 `T74` ID | 为什么必须保留在 supplement |
| --- | --- |
| `T74-TBL-06` | 它描述的是 current-host `NO_GO` 的 real-board gate/provenance truth，不应占用主文或普通附录结果位 |
| `T74-TBL-07` | `FR8` 仍只是 extension lane，且 gate 结论仍是 `no_promotion_keep_extension_lane_only` |
| `T74-SUP-01` | 必须随 `T74-TBL-07` 一起出现，防止把 persistent tie 写成 promoted comparator |
| `T74-SUP-02` | 必须防止把 `T50` supporting material 改写成 full reproducibility |
| `T74-SUP-03` | 必须防止把 `T48` 改写成 default-env / HIL / deployment closure |
| `T74-SUP-04` | 必须保留 `R32` 与硬件条件缺失，防止把 `T72` 写成 future-host ready |

### 可直接使用的 supplement bridge 句

- `补充材料继续保留 deployment-facing gate truth 与 extension-lane closure，但这些材料不改变主文的 frozen benchmark 结果层级。`
- `我们刻意把 real-board gate 与 statcalib extension lane 留在 supplement，以避免把 gate-only 或 no-promotion 证据误写成主线结果。`

## 5. 哪些内容不能被挤进主文

以下内容必须继续留在 appendix / supplement，而不能为了“看起来更完整”硬塞进主文：

1. `FR7` 全表
2. `FR6` 的查数表版
3. `T50` 的 reproducibility/material supporting rows
4. `T48` 的 isolated true runtime rows
5. `T49/T71/T72` 的 real-board gate truth
6. `FR8` 的 extension-lane closure/no-promotion table
7. `T74-FIG-04` 的 blocked portability/deployment closure 图位

## 6. 最安全的 appendix / supplement 套装

若目标是形成一个当前就可投稿的 simulation/material-first 结果包，最安全的组合是：

1. 主文：
   - `T75-FIG-M01`
   - `T74-TBL-01`
   - `T75-FIG-M02`
2. Appendix：
   - `T74-TBL-02`
   - `T74-TBL-03`
   - `T74-TBL-04`
   - `T74-TBL-05`
   - `T75-FIG-A01`
3. Supplement：
   - `T74-TBL-06`
   - `T74-TBL-07`
   - `T74-SUP-01` 到 `T74-SUP-04`

这个分层能最大限度减少 overclaim，同时保留 reviewer 需要的 supporting material。

### T76 rendered QA 备注

- `T76` 对 `T75-FIG-M01` / `T75-FIG-M02` / `T75-FIG-A01` 的修正仅限 rendered-preview 可读性，不改变这里的 appendix / supplement 分层逻辑。
