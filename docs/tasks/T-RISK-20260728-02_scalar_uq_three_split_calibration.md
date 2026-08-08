# T-RISK-20260728-02：scalar UQ 三分裂校准（legacy mirror）

- **日期**：2026-07-28
- **状态**：Done
- **权威完成记录**：
  `docs/new_tasks/T-RISK-20260728-02_scalar_uq_three_split_calibration.md`

## 输入与实际完成

本任务消费原 factor=`1.0`、288-trial scalar-UQ NO-GO，建立两个
selection folds 与一个 untouched confirmation split。V1 在 selection B 的 raw
seed 唯一性门前发现生日碰撞并 fail-closed；V2 改用可证明单射的
`base + cell_index × 2048 + trial`，没有改变 factor grid、统计门或样本量。

## 产物与验证

- 三个 split 共1,179,648 raw rows；
- 最小通过 factor 仍为1.0；
- 三个 split 的 simultaneous coverage 与四组 power IUT 全部通过；
- physics-free verifier 逐行重算，63项组合回归通过；
- V1失败证据永久保留，未产生 selection receipt 或 confirmation 污染。

所有具体路径、哈希、coverage区间、失败 lineage 与反简化检查以权威完成记录为准；
本文件只为旧 `docs/tasks/` 文档地图提供完整的任务身份和终态镜像。

## 风险、插入任务与任务板同步

- R-N188 降为 Mitigated，R-N189 Closed；
- 不插入旁路任务，只恢复父任务的 cutoff32/36 design extension；
- 不提供 twin/LER/lifetime/hardware/official/Puviani/SOTA claim；
- `docs/new_task_board.md` 中本任务状态为 Done。
