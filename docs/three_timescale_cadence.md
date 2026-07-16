# T4.3.1 三时间尺度 cadence 与 adaptation lag

## 1. 结论与证据边界

本任务把 T4.2 的逐周期 integer fast path、现有 `DualLoopScheduler`、host window/update 和更慢重标定冻结为
一条可执行的软件 cadence contract。它是配置参考与 software execution evidence，**不是** RTL、综合、
post-route、FPGA 或量子装置时序测量。

冻结的生产参考值如下：

| 层 | owner | 触发/频率 | 本周期动作 | 失败时 |
| --- | --- | --- | --- | --- |
| fast | FPGA/CD1 | 每 `5 us` 配置 epoch，II=1 | 消费一个已锁存 bank version，执行 T4.2 MAP→health→event→frame action | late/invalid action 不顺延；frame hold/fallback |
| urgent event | FPGA/CD1 | 每个 fast epoch 检查，1-cycle action register | `g/e/leakage`、CRC/deadline/OOD 等只走本地状态；source 在下一安全边界产生 action | 不等待 host/window；leakage 立即 hold，持续时 reset request |
| health window | FPGA→host | `2048` valid samples；每 `4000 cycles = 20 ms` 发出 | snapshot 最近 `10.24 ms` 的样本与 health/reason/version 统计 | 整窗拒绝、丢包计数；fast path 继续 |
| slow update | host/CD2 | 每个合格 window，起始周期与 20 ms 发窗同相 | validation/estimate/完整 inactive image；参考均值服务 `995 us = 199 cycles` | 不 stage failed/stale result；保持 last-known-good |
| commit | FPGA/CD1 | slow 完成后 `1 cycle` | 当前 scheduler 在 cycle 开头 commit，随后同 cycle fast callback 首次使用新 version | 冲突/校验失败不切换 |
| recalibration | host administrative | 每 `60 s = 12,000,000 cycles`，并在明确 end-of-run | 只产生 recalibration-due ticket；完整结果仍走 validation/stage/commit | 绝不直接修改 active bank |

窗口的“内容时长”是 `2048×5 us=10.24 ms`，发窗间隔却是 `4000×5 us=20 ms`。两者不可互换；
这里有 `1952-cycle` 的非重叠间隔，但发窗时仍 snapshot 最近 2048 个 valid samples。

## 2. 真实 scheduler 顺序

`DualLoopScheduler.tick_with_fast_path` 的真实顺序为：

1. epoch 增一并采样 fast budget；
2. 若 pending bank 到期，先 `commit_if_ready`；
3. 若 slow job 到期，完成计算并 stage inactive bank；
4. 执行 T4.2 fast callback；
5. 到达窗口边界时发窗；
6. 在发窗后启动下一 slow job。

因此同一 cycle 中，已经到期的 commit 必须先于 fast callback；而在本 cycle 刚完成/stage 的 slow result 最早只能
在下一 cycle commit。T4.3.1 的 reference trace 直接使用真实 scheduler、真实 `ParamBank` 和 T4.2.4
`BitAccurateFastPath`，不是另写一条模拟箭头。

reference onset 为 epoch `2040`：

| 事件 | epoch | 说明 |
| --- | ---: | --- |
| leakage source | 2040 | observed local source；不进入 host critical path |
| local `hold` action | 2041 | 1-cycle event register |
| first influenced window ready / slow start | 2048 | window `[1,2048]` 含 9 个 onset 后样本 |
| slow finish / stage | 2247 | `995 us / 5 us = 199 cycles` |
| commit / first fast use | 2248 | callback 前 commit；version `0→1` |

trace 从 MAP pipeline warm-up 后的每个 cycle 调用 T4.2 integer path；commit 前一周期仍读 version 0，commit
周期读 version 1。两次独立 image compile + scheduler replay 的 trace-row SHA256 一致。

## 3. adaptation lag 的两个口径

对 drift onset epoch `t0`，机器合同逐项保存：

`total = evidence_wait + queue_wait + slow_service + stage_to_commit + commit_to_first_use`。

生产相位锁定且服务时间小于 20 ms，所以本次 nominal sweep 的 queue wait 为 0；这不是对拥塞、通信中断或
jitter 的保证。

必须同时报告两种 evidence policy：

1. `first_influenced_window`：首个含至少 1 个 onset 后样本的 window。它是最乐观的软件响应界，不等于统计
   detector 已可靠识别漂移。
2. `first_full_post_change_window`：首个 2048 个样本全部位于 onset 后的 window。它避免混合 pre/post-change
   evidence，是更保守的完整窗口界。

对全部 `4000` 个 onset phase 逐一穷举，结果为：

| policy | min | median/mean | max | evidence wait |
| --- | ---: | ---: | ---: | ---: |
| first influenced | `200 cycles = 1.000 ms` | `2199.5 cycles = 10.9975 ms` | `4199 cycles = 20.995 ms` | `0--3999 cycles` |
| full post-change | `2247 cycles = 11.235 ms` | `4246.5 cycles = 21.2325 ms` | `6246 cycles = 31.230 ms` | `2047--6046 cycles` |

每个 policy 恰有 4000 个不同 lag，所有 8000 rows 的分解都精确闭合。local urgent event 始终是
`1 cycle = 5 us`，不可拿 host adaptation lag 替换或相加成“物理反馈延迟”。

## 4. 参数 freshness 的集成修复

深审计发现，T4.2.3 独立 fault pilot 的默认 `max_parameter_age_cycles=64` 与 4000-cycle slow cadence 不兼容：
直接串联会在每次更新后 `320 us` 就进入 stale fallback，系统大部分时间无法使用 MAP。

修复不是删除 stale gate，而是：

- `BitAccurateFastPath` 增加显式 `max_parameter_age_cycles`，仍保留 64-cycle 默认以保持 T4.2.3 fault-test
  profile 可复现；
- `hardware_hil.yaml` 与 `hardware_emulation.yaml` 的生产 cadence 冻结为 `8192 cycles = 40.96 ms`；
- 8192 至少覆盖两个 4000-cycle slow interval，即允许一个 nominal update 缺口；第二个缺口后仍 fail closed；
- age 必须小于 `2^16`，与 T4.2.4 16-bit word 一致。

8192 是控制策略配置，不是 device-calibrated SLA。后续 T4.3.3 必须在 timeout/jitter/pause/race stress 下验证
它的 false fallback 与 unsafe-age trade-off。

## 5. clock/domain crossing contract

- fast/event/frame 全在 CD1 epoch domain；event action 不跨 host clock。
- window crossing 是 snapshot/sequence 边界；host serialization 可滞后，但不得改变 source epoch/version。
- host result 只写完整 inactive image；active version 在 commit 边界原子变化。T4.3.1 只验证现有软件顺序，
  version/CRC/timestamp/CAS/readback 与 hysteresis 的完整实现属于 T4.3.2。
- minute/end-of-run recalibration 只是 administrative due signal；它可更新 calibration prior/image，但必须走同一
  validation/stage/commit 路径，不能绕过 CD1 safety authority。

## 6. 产物与复现

- 运行时：`cnn_fpga/runtime/three_timescale_cadence.py`
- 执行验证：`cnn_fpga/benchmark/three_timescale_cadence_validation.py`
- machine artifact：`docs/t4_3_1_three_timescale_cadence_validation.json`
- 8000-row phase Source Data：`docs/t4_3_1_adaptation_lag_phase_sweep.csv`
- scheduler/T4.2 boundary trace：`docs/t4_3_1_cadence_execution_trace.csv`

复现命令：

```powershell
python -m cnn_fpga.benchmark.three_timescale_cadence_validation
python -m pytest -q --basetemp .pytest_tmp_t431 tests/test_three_timescale_cadence.py tests/test_three_timescale_cadence_validation.py
```

production runner 为 14/14 gates；focused tests 为 26 passed，T4.2/T4.3 相邻回归 106 passed；显式全量
`tests/` 为 `1404 passed, 14 skipped, 4 failed`，4 个失败仍仅为 R-N012 缺失的旧 FR8/P4 文档。Source Data
保存配置/实现/trace hash，且每个 lag component 可逐行复算。

## 7. 未完成边界

本任务未实现 T4.3.2 的完整 version/CRC/timestamp/CAS/readback/hysteresis，也未完成 T4.3.3 的 jitter、queue、
communication pause、timeout、race 和 rollback 稳定性 stress。`995 us` 是当前 latency-model 各 stage 均值之和，
不是 p95/p99 或实测；`60 s` 是冻结 policy，不是装置漂移的最优重标定周期。RTL、综合、FPGA、板卡和量子
装置证据继续 fail closed。
