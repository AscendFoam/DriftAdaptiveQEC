# T2.4.2 backlog、jitter 与 deadline 故障模型

## 结论与证据边界

本任务在真实 `DualLoopScheduler` 和 `ParamBank` 路径上加入可执行的 backlog、jitter、
端到端 window deadline、输入 burst、通信停顿、参数更新冲突与 FIFO overflow。正式验证使用
7 个场景、8 个配对 seed、每组 64,000 个 fast cycles，共执行 3,584,000 个 scheduler tick。
所有场景复用同 seed 的物理 drift、观测噪声与 channel noise；slow estimator 只消费
`observed_mean`，不读取 hidden truth。

这是 `paired_synthetic_standard_binning_timing_stress_not_board_measurement`。结果可以证明
软件状态机能检测故障，并在预注册 stress law 下量化 LER/availability 影响；不能证明目标板故障率、
hard-real-time closure、真实量子装置 LER 或经设备标定的通信/脉冲时序。

## Live 配置与指标

模型从 `cnn_fpga/config/hardware_hil.yaml` 读取当前 cadence：

- fast slot：`5 us`；fast budget：`1.5 us`；
- window：2,048 samples；stride：4,000 cycles，即每 `20 ms` 产生一次；
- slow start period/window deadline：`20 ms`；slow service budget：`5 ms`；
- reference slow-stage mean：`10+60+900+20+5=995 us`；
- FIFO 默认容量：2 个待处理 window。

定义三层 availability：

1. `fast_action_availability`：fast latency 未超过 1.5 us；
2. `fresh_parameter_availability`：active bank 的来源 window 距当前周期不超过 20 ms；
3. `end_to_end_control_availability`：上述两项同时成立。

LER 使用同一 synthetic physical trace 上的 standard-binning parity error。fast miss 在本 stress
模型中采用 hold-last action；runtime scheduler 本身仍保持 record-only contract，真正的 late-action
suppress/fallback 属于 T4.2/T4.3，不能从本任务越级声称已实现。

## 可执行故障与失败分支

| 场景 | 注入与隔离目的 | 必须逐 seed 出现的事件 |
| --- | --- | --- |
| `reference` | YAML reference latency，无故障注入 | 正常 start/finish/stage/commit |
| `jitter_deadline` | slow mean ×24、std ×12；fast `1.60 +/- 0.45 us` | fast/slow budget violation、window deadline miss |
| `input_burst` | 两次各 4 个同周期外部 window，FIFO 容量 12 | input burst；隔离 backlog，不强制 overflow |
| `communication_pause` | 一段 16,000-cycle transport pause | pause start/end、window deadline miss |
| `parameter_conflict` | internal update pending 时第二 writer 请求 stage | conflict；第二 writer 必须 0 次落盘 |
| `fifo_overflow` | 两次各 8 个 window，FIFO 容量 2 | burst、overflow、drop-oldest provenance |
| `combined` | jitter、burst、pause、conflict、capacity=2 同时发生 | 上述全部故障事件 |

`ParamBank.stage_update` 不再静默覆盖 pending commit。第二 writer 会抛出
`ParameterUpdateConflictError`；scheduler 转为结构化 `parameter_update_conflict` 事件，保持原 pending
version、staging bank 和 active bank 不变。FIFO overflow 明确记录 capacity、被丢弃/被接纳 window ID
和来源，并采用 drop-oldest。通信 pause 记录起止、周期数和持续时间；slow finish/start 在 pause 中阻塞，
deadline 使用 `finish_time - window.ready_time`，因此包含 queue wait、service 与 pause stall。

## 正式配对结果

下表均为 8-seed 平均；`Delta LER` 和 `Delta availability` 是相对同 seed reference 的配对差，
括号为 10,000 次 cluster bootstrap 的 95% CI。

| 场景 | LER | Delta LER | 端到端 availability | Delta availability |
| --- | ---: | ---: | ---: | ---: |
| reference | 0.07971 | 0 | 0.95005 | 0 |
| jitter + deadline | 0.47564 | +0.39592 [0.38501, 0.40615] | 0.00000 | -0.95005 [-0.95087, -0.94923] |
| input burst | 0.50685 | +0.42714 [0.42585, 0.42834] | 0.29563 | -0.65442 [-0.65519, -0.65367] |
| communication pause | 0.28115 | +0.20144 [0.20081, 0.20212] | 0.30840 | -0.64165 [-0.64242, -0.64090] |
| rejected parameter conflict | 0.07971 | 0 | 0.95005 | 0 |
| FIFO overflow | 0.21285 | +0.13314 [0.13217, 0.13398] | 0.30322 | -0.64683 [-0.64764, -0.64600] |
| combined | 0.35928 | +0.27957 [0.27645, 0.28250] | 0.06526 | -0.88480 [-0.88842, -0.88049] |

被拒绝的孤立参数冲突与 reference 对所有 seed 精确一致，证明 fail-closed rejection 本身没有污染
active control；它不代表未拒绝的 torn/stale/CRC 错误已经解决。组合场景下 fast availability 为
`0.54531`、fresh-parameter availability 为 `0.11897`，端到端交集为 `0.06526`，因此不能只看单一
latency percentile 代替系统 availability。

## 反简化审计

- 事件门在每个 seed 上检查，而不是只检查合并总数；
- 独立 `input_burst` 与 `fifo_overflow` 场景分别隔离 backlog 和容量溢出；
- 参数冲突测试验证 rejection 前后 snapshot 完全不变、version 单调且最大步长为 1；
- pause 测试验证 service `10 us`、window age `15 us`，证明 deadline 计入暂停 stall；
- production JSON 保存每个 seed 的 metrics、事件计数、latency quantile、integrity 和 scheduler
  snapshot；CSV 提供 56 行扁平 Source Data；
- artifact 绑定 timing model、scheduler 和 param bank 的 SHA256：
  `869ddb303bf42ab82f4148f5ead173ef75bdb7af5ed58e5f8c38935eeac65937`；
- 13 个 machine gates 全通过，覆盖逐 seed 故障检测、配对 trace、冲突中性、LER/availability CI、
  数值/版本完整性和 `target_hardware_measured=False`。

## 复现

```powershell
python -m cnn_fpga.runtime.timing_fault_model
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests\test_timing_fault_model.py -q
```

机器结果见 `docs/t2_4_2_timing_fault_validation.json`；逐 seed 表见
`docs/t2_4_2_timing_fault_validation.csv`。

## 未关闭问题

stress schedule、latency multiplier、hold-last action、synthetic drift 与 20 ms freshness 都是项目模型
条件，不是 device calibration。scheduler 尚未实现 CRC、age/CAS/readback/ack wire contract，也未在
runtime 层 suppress late fast action。T2.4.3 继续处理 fixed-point/LUT/bank error；T4.2/T4.3 实现
安全 fallback；T5/T6 负责多场景和实板 core/transport/end-to-end 证据。

