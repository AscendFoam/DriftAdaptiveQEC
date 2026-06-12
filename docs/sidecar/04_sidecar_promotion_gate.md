# 04 Sidecar Promotion Gate

## 1. 目的

promotion gate 决定一条 sidecar 是否可以从候选路线进入主线任务候选池。它不直接把结果写成主线事实。

## 2. 晋升前提

从 `S2_bounded_sidecar_benchmark` 进入 `S3_promotion_candidate` 至少需要：

1. 有完整 `sidecar_manifest.json`。
2. 有可复查 summary。
3. 有 clean 或 clearly-recorded dirty provenance。
4. 结果目录只在 `runs/sidecar/<lane_id>/...`。
5. 未改写历史 run root。
6. 未改变主线默认行为，或已用 regression 证明旧路径不变。
7. 明确 negative / inconclusive 结果也被记录。

## 3. 必须回答的问题

promotion review 必须回答：

1. 这条 lane 解决什么主线缺口？
2. 它相对 `T24`、`T57/T58`、`T64-T70` 或 `T48/T50/T72` 的关系是什么？
3. 它是否只是 toy/replay signal？
4. 它是否需要新的正式 benchmark protocol？
5. 它是否触碰 `.tflite` 或 real-board 边界？
6. 它是否要求修改主线 runtime contract？
7. 最小下一任务是什么？

## 4. Gate 结果

允许结果：

| Verdict | 含义 |
| --- | --- |
| `PROMOTE_TO_MAINLINE_TASK_CANDIDATE` | 可以写新主线任务包，但仍不自动执行 |
| `KEEP_AS_SIDECAR` | 继续作为 sidecar，不进入主线 |
| `REWORK_SIDECAR` | 需要修 manifest、provenance、tests 或边界 |
| `RETIRE_LANE` | 路线暂时放弃或只保留历史参考 |

## 5. 红线

即使 gate 通过，也不能直接声称：

- 主线结果已被替代。
- 论文 claim 已经升级。
- `.tflite` 或 real-board 已闭环。
- sidecar 成为 mature comparator。

进入主线必须另开任务包，并接受主线 `Allowed files / Forbidden scope / Verification / Docs to update` 约束。

