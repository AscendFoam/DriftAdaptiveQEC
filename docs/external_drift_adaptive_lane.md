# T6.8.2：一般 drift-adaptive decoder 外部对照 lane

## 结论

本 lane 已完整执行并通过证据完整性审计，但外部 BOCD wrapper **没有通过统一 wall-clock 预算门**。因此可以报告一个受限的描述性结果：在完全相同的 504 条 formal endpoint trajectories、24,772,608 个 joint decisions/method 上，Route-A 的 equal-family/seed average LER 为 `0.01195908`，低于外部 BOCD Window/EWMA router 的 `0.01196909`；paired 24-seed contrast（external − Route-A）为 `1.00184e-5`，95% CI `[7.66793e-6, 1.23689e-5]`。但外部路径 13,104 次 update 中有 1 次 `13,004.1 us`，超过统一 `5,000 us` worst-case ceiling，故不得写成“matched-budget external superiority”或“general drift-adaptive SOTA”。

机器 verdict：`COMPLETE_EXTERNAL_DRIFT_ADAPTIVE_LANE_ROUTE_A_LOWER_LER_BUDGET_FAIL`。12 个门中 11 个通过，唯一失败为 `G08_external_budget_meets_common_caps`；其余 evidence-integrity gates 全通过。

## 外部实现与文献边界

- 真正执行的外部代码是 [Adams–MacKay BOCD](https://arxiv.org/abs/0710.3742) 的 MIT 实现 [y-bar/bocd](https://github.com/y-bar/bocd)，固定 commit `5f272b1f2252b5d396130707a35229757a9e5f18`，未修改上游源码，上游 3 项测试通过。
- wrapper 每 2,000 joint decisions 只消费最近 1,024 个 observed syndrome residuals 的 static-NLL 标量；Window 与 EWMA bank 都按相同 cadence 因果更新，BOCD 只决定下一周期选哪个 bank。hidden truth 只在动作确定后用于评估 detection/lag。
- [Bhardwaj et al. 2025](https://arxiv.org/abs/2511.09491) 只作为 overlapping-window QEC drift 的方法映射；本 lane 没有其 paper-specific official implementation，因此不是 exact reproduction。
- Sivak calibration-prior 与 Stein FiLM 工作只作跨 code-family 背景，不进入 raw-number 排名。

## 预注册选择与 formal 隔离

- pilot：21 个 family endpoint cells × 12 seeds；formal：21 cells × 24 disjoint seeds。rate、amplitude、duration levels 全部不相交。
- 单一全局 grid：hazard `{4,8,16}` updates × short-run `{1,2,4}` × posterior threshold `{0.35,0.55,0.75}`，共 27 个候选。
- eligibility 在 formal 前冻结为：dynamic pilot 至少实际 detection 一次，且 nominal Window-bank rate `<=0.10`。
- 9/27 候选合格；冻结 `h8_r1_p35`。它在 dynamic pilot 上共 1,478 次 detection，53.33% 的动态轨迹至少触发一次；nominal Window-bank rate 为 0。formal 未逐场景调阈。

## Formal 结果

| 方法 | equal-family/seed average LER | p95 window LER | worst window LER |
| --- | ---: | ---: | ---: |
| Route-A | 0.01195908 | 0.10546875 | 0.35156250 |
| external BOCD Window/EWMA router | 0.01196909 | 0.10546875 | 0.35156250 |
| EWMA adaptive MAP | 0.01197252 | 0.10546875 | 0.35156250 |
| Window MAP | 0.01199279 | 0.10546875 | 0.35546875 |
| Kalman adaptive MAP | 0.01267327 | 0.10546875 | 0.45507813 |
| static joint MAP | 0.01460875 | 0.12500000 | 0.21484375 |

四个 paired 24-seed contrasts（left − Route-A）的 95% CI 均为正：

- external BOCD：`1.00184e-5 [7.66793e-6,1.23689e-5]`；
- Window：`3.37158e-5 [1.59524e-5,5.33287e-5]`；
- EWMA：`1.34478e-5 [1.17138e-5,1.51817e-5]`；
- Kalman：`7.14196e-4 [6.69460e-4,7.61245e-4]`。

这不是“所有场景均改善”：Route-A 与 EWMA 在许多 family 完全相同，主要差异集中在 periodic；external wrapper 在 compound 上略低、在 periodic/variance 等处略高。必须保留 family table，不能只引 aggregate CI。

## 外部适应是否真实发生

- formal detection 3,320 次，其中 truth-evaluation-only false detections 530 次；
- 平均 Window-bank rate `0.262665`，nominal/pilot 资格并未保证 formal 全局低切换；
- 177 条轨迹观测到 onset 后 detection，303 条 right-censored；lag median `7,904`、p95 `31,904`、max `43,856` decisions；
- external error trace 在 102/504 条轨迹上与 EWMA 不同，排除了“外部库被调用但动作始终退化为 EWMA”的伪集成。

## 预算失败

BOCD 的 operation proxy 为 696 MAC/update，低于 8,192 上限。实测 Python host update wall-clock：p50 `294.0 us`、p95 `631.5 us`、p99 `896.8 us`、worst `13,004.1 us`；唯一 deadline miss 位于 `formal_evaluation:correlation_drift:05`、seed `202607176212`、update index 2。

统一合同规定任何 host update `>5,000 us` 均失败，不能以 p95 覆盖 worst。因此：

- `EXTERNAL_BOCD_WRAPPER_PAIRED_OUTCOME = ROUTE_A_LOWER_LER` 仅是描述性 paired outcome；
- `EXTERNAL_BOCD_MATCHED_BUDGET = FAILED`；
- `GENERAL_DRIFT_ADAPTIVE_SOTA = PROHIBITED`；
- `BHARDWAJ_EXACT_REPRODUCTION = NOT_ESTABLISHED`。

## 简化实现 / 伪实现复核

1. external 与 T6.7 parent 的 504 条 `(cell,seed)` 逐条比较 input/truth hash：0 missing、0 input mismatch、0 truth mismatch。
2. 全方法 decision count 精确相同，未用少量 external trace 对比完整内部结果。
3. pilot selector 加入 dynamic detection 非零条件，拒绝“永不切 bank”的退化候选。
4. 测试用两套相反 hidden labels 重放相同 observed scores，online bank schedule/error action 保持不变；truth 只改变事后评估字段。
5. 缓存由整个报告文件 hash 改为执行语义 hash；旧 756 条完成缓存仅从明确列出的 reviewed runner hash 原子迁移，算法函数变化仍强制新 key。
6. mutation audit 改成每个 mutation 绑定自己的 target gate，避免已有 G08 失败时“任何 mutation 都假装被捕获”；10/10 target gates 被真实拒绝。
7. current-state verifier 同时重算 gates/verdict、上游 commit/source hash、父报告 hash、runner hash 与 Source Data hash/行数。

## 产物与复现

- runner：`cnn_fpga/benchmark/external_drift_adaptive_lane.py`
- 外部源码：`third_party/bocd/`
- machine report：`docs/t6_8_2_external_drift_adaptive_lane.json`
- Source Data：`docs/t6_8_2_external_drift_adaptive_lane_source_data.csv`
- tests：`tests/test_external_drift_adaptive_lane.py`

```powershell
$env:PYTHONPATH='.'
python -m cnn_fpga.benchmark.external_drift_adaptive_lane
$env:PYTHONPATH='.;third_party/bocd'
python -m pytest -q tests/test_external_drift_adaptive_lane.py third_party/bocd/tests
```

最后一次定向验证：`12 passed`（9 个项目测试 + 3 个上游测试）。
