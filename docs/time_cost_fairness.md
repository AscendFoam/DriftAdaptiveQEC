# T5.1.5 物理时间与控制成本公平化报告

## 结论

T5.1.5 已把物理时间、事件负担、算法成本和 latency 分成三条不可混排的证据 lane：

1. `protocol_wallclock_common_700us`：measurement-feedback 与 autonomous sBs 的共同 700 μs 比较；
2. `matched_controller_*_100us`：standard/MF/teacher/recurrence/student 的共同 10 cycles / 100 μs 比较；
3. `host_slow_loop_regime_estimator`：六类 estimator 的 development-host software profile。

报告没有生成总分或全局 leaderboard。第三条 lane 没有 physical lifetime、measurement/reset/gate counts；
T4.4.4 controller 没有同模型 classical latency；target-board core/transport/end-to-end 和 physical frontend
全部保持 `null`。这些空值不能用 0、配置均值或别的模型的 host profile 补齐。

## Protocol wall-clock lane

T3.2.8 提供 cutoff 12/16 × high/medium/low 共 6 个 matched scenarios。measurement-feedback full cycle
为 10 μs，700 μs 内运行 70 cycles；autonomous full cycle 为 7 μs，运行 100 cycles。

| 口径 | Measurement-feedback | Autonomous |
| --- | ---: | ---: |
| Common horizon | 700 μs | 700 μs |
| Full cycles | 70 | 100 |
| Measurements | 140 | 0 |
| Resets | 140 | 200 |
| Active gates | 1,260 | 1,800 |
| Measurements / 100 μs | 20 | 0 |
| Resets / 100 μs | 20 | 28.5714 |
| Active gates / 100 μs | 180 | 257.1429 |

所有 6 个 scenarios 都出现计量单位排序反转：

- 按 protocol-native cycle 的 autonomous/feedback logical-lifetime ratio 为 `1.151287--1.346101`，看似
  autonomous 更好；
- 按共同物理时间的 ratio 为 `0.805901--0.942271`，实际 autonomous 更差；
- 每个 700 μs autonomous 避免 140 次 measurement，却增加 60 次 reset 和 540 次 active gates。

因此不能把 autonomous 每轮更短隐藏掉，也不能用 `0.7` timing ratio 对 lifetime 做事后线性缩放。这里的
7/10 μs 是外部 Puviani model timing，不是当前目标板或量子装置实测。

T5.1.1 的 no-correction idle probe 只有 30 μs。其 measurement/reset/active gate 都精确为 0，但它只作
sanity reference，不能直接塞进 700 μs protocol ranking。

## Matched controller lane

T4.4.4 的 cutoff 12/16 两条 lane 都是 10 full cycles、100 μs。每个策略同时报告 fidelity/logical-Z
lifetime 的 cycles 与 μs；它们共享 measurement-feedback physical protocol，因此每条轨迹均有 20 次
measurement、20 次 reset 和 180 次 active gate applications。

| Cutoff | Strategy | Fidelity lifetime (cycles / μs) | Logical-Z lifetime (cycles / μs) | Stored scalars | MAC/half-cycle |
| ---: | --- | ---: | ---: | ---: | ---: |
| 12 | Standard | 3.571 / 35.706 | 2.772 / 27.722 | 15 | 0 |
| 12 | Exact-budget MF | 8.623 / 86.228 | 6.796 / 67.956 | 72,853 | 72,266 |
| 12 | GRU teacher | 8.440 / 84.399 | 6.757 / 67.570 | 72,853 | 72,266 |
| 12 | Handcrafted recurrence | 6.642 / 66.424 | 6.433 / 64.329 | 105 | 45 |
| 12 | Distilled student | 8.427 / 84.271 | 6.743 / 67.432 | 95 | 87 |
| 16 | Standard | 5.994 / 59.936 | 5.135 / 51.351 | 15 | 0 |
| 16 | Exact-budget MF | 9.156 / 91.557 | 7.508 / 75.076 | 72,853 | 72,266 |
| 16 | GRU teacher | 9.525 / 95.254 | 7.961 / 79.608 | 72,853 | 72,266 |
| 16 | Handcrafted recurrence | 6.381 / 63.807 | 6.268 / 62.681 | 105 | 45 |
| 16 | Distilled student | 9.489 / 94.894 | 7.908 / 79.084 | 95 | 87 |

`observed_e_events` 与 reset 分列：cutoff 12 standard 期望 `e` events 是 `6.6484`，handcrafted 是
`0.4180`，但两者仍各执行 20 次 protocol reset。较低 `e` occupancy 不能写成较低 reset burden。
multilevel leakage events 仍为 `null`。

成本表是 float analytic counts，不是 latency。所有 controller 的 `classical_latency_us` 保持 `null`；95
scalars / 87 MACs 说明 student 压缩程度，但不能推出 FPGA deadline、Fmax 或 board latency。two-cycle
finite-horizon control reference 明确排除在 ten-cycle table 外，不能按 5 倍缩放。

## Host software latency lane

T4.1.1 的六个 estimator 只在共同 32-cycle update task 上报告 development-host batch median：

| Family | Host median μs/update | 5,000 μs ceiling fraction | MAC proxy | Model+state bytes |
| --- | ---: | ---: | ---: | ---: |
| Causal TCN | 4.212 | 0.000842 | 3,556 | 2,508 |
| Small GRU | 6.791 | 0.001358 | 2,300 | 1,916 |
| Gaussian HMM | 724.369 | 0.144874 | 926 | 3,728 |
| Diagonal Kalman | 2.205 | 0.000441 | 1,064 | 1,248 |
| Exponential recurrence | 2.101 | 0.000420 | 504 | 1,084 |
| Run-length FSM | 70.582 | 0.014116 | 1,408 | 1,044 |

这些值来自另一 decision target 和 batch/profile 实现，只说明 development-host software profile 在其
5,000 μs software ceiling 内；不能转移给 T4.4.4 controller，也不是 FPGA core latency。

## Latency provenance 与硬件空值

T2.4.1 中的 fast path `1.0 μs`、slow path `995 μs` 是 `hardware_hil.yaml` 的配置模型均值，不是测量。
当前目标板状态仍为 `not_started`；以下 7 项保持 `null`：

- target-board core、transport、end-to-end latency；
- quantum measurement、ADC acquisition、AWG/DAC output、physical action latency。

外部 Sivak/Puviani timing、项目配置均值、host software profile 和未来 target-board measurement 不能相加、
相减或求一个跨 lane ratio。

## 机器验证与非 demo 审计

- `docs/t5_1_5_time_cost_fairness.json`：12 protocol rows、10 controller rows、6 host rows、18/18 gates；
- `docs/t5_1_5_time_cost_fairness_source_data.csv`：537-row provenance/metric/null ledger；
- `tests/test_time_cost_fairness.py`：25 个 direct 与 mutation tests；
- mutation tests 拒绝丢掉 μs 指标、把 autonomous cycle 改成 10 μs、controller latency 填 0、`e` 当 reset、
  two-cycle oracle 塞进 ten-cycle table、给 estimator 伪造 physical lifetime、填 target core latency、生成
  cross-lane aggregate 或改写 T5.1.4 fallback branch。

该任务是生产 artifact 的只读公平化汇总，不是新增物理采样。它直接消费原始 event/lifetime/cost/profile
字段并做算术一致性验证，避免只写一张手工表或用 `null=0` 的 demo 处理。

