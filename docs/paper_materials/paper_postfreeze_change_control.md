# T89 Post-Freeze Change Control

## Purpose

`T89` 之后，当前主线进入的是 `frozen-mainline handoff`，不是“可以自由人工润色”的开放态。后续任何改动都必须先判断属于哪一层，再决定是否允许在当前 main 上处理。

## Change Levels

| level | meaning | allowed_without_reopen | required_artifacts |
| --- | --- | --- | --- |
| `L0` | 不改变事实边界的微小整理 | 是 | 至少更新当前变更涉及的 README 或登记文档；必要时补一份小型 review |
| `L1` | 会影响 handoff 组织方式但不引入新证据的 docs-only 改动 | 否，需要新的 bounded docs-only task | 新任务包 + 新 review + 更新 handoff/source-of-truth 文档 |
| `L2` | 会改变证据等级、blocked surface 状态或核心 claim 口径的改动 | 否，需要新的 evidence task | 新协议/新证据/新 review；必要时新 run root 或新主机验证 |
| `L3` | 当前 main 直接禁止的改动 | 否 | 不得直接改；只能由未来 Captain 重新定义边界后另开路线 |

## Concrete Rules

1. `CCR-01`：只修正错字、死链、文件名、标题层级、表述中不改变事实含义的语法问题，且不触碰 note、本地编译产物、stable-ID 选择、blocked disclaimer wording 时，可归入 `L0`。
2. `CCR-02`：任何会改动 `GO_FOR_FROZEN_MAINLINE_HANDOFF_ONLY` verdict 口径的操作，最低也属于 `L2`；不能通过 README、review 或 handoff packet 的手工润色把 verdict 偷换成更强结论。
3. `CCR-03`：任何会改变 `FZ01-FZ05` frozen surface 选择、主文/appendix/supplement route、primary representation 或 explanatory route 的操作，属于 `L1`；必须开新的 bounded docs-only task，并同步更新 source-of-truth map。
4. `CCR-04`：任何会改动 `BD01-BD06` blocked disclaimer wording 的操作，默认属于 `L1`；如果改动意图是弱化免责声明、缩小 blocked 范围或引入更强 claim，则直接提升为 `L2`。
5. `CCR-05`：任何把 `.tflite`、real-board、training reproducibility、`FR8/statcalib`、expanded benchmark、stronger oracle baseline、deployment closure 写得更强的改动，都属于 `L2`；必须有新证据任务，不能靠 prose polish 升级。
6. `CCR-06`：任何会触碰 `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`、其编译产物、stable-ID 图表实体、caption pack、insertion map 的操作，都不属于 `L0`；至少需要 `L1`，而且必须被新的任务包显式授权。
7. `CCR-07`：任何将 theory 分支内容直接 mergeback 到当前 mainline prose 的操作都属于 `L3`，除非未来有单独的 branch-integration task、evidence audit 与 review gate。
8. `CCR-08`：任何新增实验、重跑 benchmark、训练、`.tflite` smoke、real-board gate/regeneration/smoke 的动作都不属于 post-freeze 人工维护，统一归入 `L2` 或更高路线，不能借当前 handoff 包名义直接执行。
9. `CCR-09`：任何 `L1/L2` 改动在落地时都必须同步刷新 `paper_frozen_mainline_handoff_packet.md` 与 `paper_frozen_mainline_source_of_truth_map.md`，否则视为 traceability 不完整。
10. `CCR-10`：任何试图把当前主线写成 `submission-ready completed`、`hardware-ready finalization`、`default-env compatibility closed`、`full reproducibility closed`、`mature statcalib comparator promoted` 的改动，当前 main 直接视为 `L3` 禁止项，除非未来先完成对应 evidence lane 并通过新的 gate。

## Typical Classification Examples

| example | classification | reason |
| --- | --- | --- |
| 修复 `paper_frozen_mainline_handoff_packet.md` 中的错字或坏链接 | `L0` | 不改变事实边界或 route |
| 调整 handoff packet 的阅读顺序，但不改变引用文档集合与边界口径 | `L0` | 只做组织优化 |
| 把 `T75-FIG-M01` 改回主文 benchmark primary representation | `L1` | 改变 frozen surface 选择 |
| 改写 `BD02`，把 `.tflite` 说成 default-env portability | `L2` | 明确升级 blocked surface claim |
| 在当前 main 直接插入 theory 分支大段 prose | `L3` | 违反 main/theory 隔离 |
| 重新编译 note 并顺手改主文结论句 | `L1` 到 `L2` | 触碰 note，且可能改变 claim 边界 |

## Directly Prohibited On Current Main

以下操作在当前 post-freeze main 上默认禁止，不得以“小修”名义直接执行：

- 直接把 blocked surface 改写成已闭环事实
- 直接把 frozen-mainline handoff 说成 submission-ready completed
- 直接把 theory 分支内容 mergeback 到 main
- 直接增删 stable-ID 图表实体、caption pack、insertion map
- 直接重写 note 或编译产物而不经过新任务包
- 直接借 review、README 或 worker summary 文案升格证据结论

## Minimum Recordkeeping Requirement

哪怕只是 `L0` 级微小整理，也必须满足两条：

1. 让后续读者能知道这次改动没有改变 `T89` 的 frozen-mainline 事实边界。
2. 不让 README、handoff packet、source-of-truth map 与 review 之间出现新的口径漂移。
