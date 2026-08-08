# T6.2.2 长序列与故障路径软件/RTL 资格验证

## 结论

T6.2.2 通过板卡无关资格门：独立快速整数 golden 生成 10 个 family、每个
`100,000` cycles、合计 `1,000,000` cycles 的逐周期 binary trace；Yosys 展开的
qualification RTL 由 10 个隔离 CXXRTL 进程逐 family 全量执行，对 commit ack、active
bank/version、MAP debug、118-bit output word 和 232-bit state word 逐 bit 比较，结果为
`0 mismatch`、`0 undefined action`、`0 output/state CRC error`、`0 silent overflow`。

机器 verdict 为：

`PASS_BOARD_INDEPENDENT_LONG_RTL_QUALIFICATION_READY_FOR_ROUTE_A`

该结论只表示 software golden + two-state CXXRTL qualification。它不是 bitstream、真实
transport/CDC、vendor timing signoff、板级 latency/deadline/power、SEU/亚稳态或 HIL 证据。

## 被测合同与独立性

- production 参数：OOD threshold `192`、maximum parameter age `8,192`、uint16 maximum
  trusted version；
- core pipeline：5-cycle MAP + 1-cycle registered event/action，II=1；
- 可见比较面：commit ack、active A/B bank/version、MAP valid/address/22-bit LLR、完整
  118-bit output 与 232-bit state；
- 快速 golden 是手写整数实现，不调用 legacy `BitAccurateHardwareReference`、不解析 RTL；
- 正式运行前另用 legacy golden 连续交叉检查 `10,000` rows，结果 `0 mismatch`；
- T6.2.1 production management report 的 SHA256 与 PASS gate 被本报告绑定；完整 514-word
  CRC32/config/session/snapshot 仍由 T6.2.1 审计，T6.2.2 wrapper 不假装替代 management top；
- trace 固定为 82-byte packed row，共 `82,000,000` bytes，SHA256 为
  `1a9481a2e849b0e63f6762411cbfc5bd4e30f66803970d88191acff7ebc1c751`。

## 十个正式 family

| family | cycles | 主要目的 | 关键实测覆盖 |
| --- | ---: | --- | --- |
| nominal_random | 100,000 | 正常随机 syndrome、II=1 与 frame/action | 98,963 outputs；fault mask 全零 |
| boundary_and_frame_wrap | 100,000 | code 0/1/2/3/边界/1023、X/Z recovery 与 frame wrap | 99,003 outputs；X/Z action 均出现 |
| leakage_reset_hysteresis | 100,000 | leakage、hold、reset request/ack、hysteresis | 1,563 次 fault→healthy recovery |
| integrity_ood_stale | 100,000 | invalid/reserved、OOD、CRC、untrusted bank、age、deadline、unexpected ack | bits 0/1/2/3/4/8/9/12 均非零 |
| deadline_pause_recovery | 100,000 | input gap、deadline miss、mid-family synchronous reset | 781 次 fault→healthy recovery |
| version_trust_commit_race | 100,000 | safe-boundary、trust、same-bank、跳号、rollback、cfg/commit race | 61 attempts；11 ack；10 rollback reject；10 untrusted reject |
| fifo_overflow_backpressure | 100,000 | 8-depth FIFO、pause、backpressure、overflow、deadline | 4,793 overflow，全部显式记账；pending=0 |
| drop_duplicate_reorder | 100,000 | pause/drop/duplicate/reorder/sequence check | drop 19、duplicate 15、reorder 12、sequence fault 54 |
| compound_fault_recovery | 100,000 | transport + CRC/OOD/age/deadline/leakage/untrusted compound | 1,224 explicit markers；422 次恢复；pending=0 |
| saturation_extreme_lut | 100,000 | 完整 514-word extreme bank、LLR/counter 边界 | LLR min/max 49,800/49,668；五类 maxima 均 255 |

三个抽象 transport family 的 FIFO 深度均不超过 8，所有 overflow 都在检测时产生显式
fault token，源停止后先排空已接收 packet 和 error marker，再持续 clean observations，最终
mode/health/fault 全部恢复为 `normal/healthy/0`。这里的 FIFO/sequence checker 是 receiver-side
抽象行为模型，不是 UART/USB-SPI/JTAG 实现。

## fault branch 矩阵

| bit | 名称 | 正式计数 | 资格期望 |
| ---: | --- | ---: | --- |
| 0 | observation_invalid | 56,199 | 非零 |
| 1 | ood_score_exceeded | 39,842 | 非零 |
| 2 | input_crc_mismatch | 33,541 | 非零 |
| 3 | image_crc_mismatch / untrusted image projection | 8,755 | 非零 |
| 4 | image_sha256_mismatch / untrusted image projection | 8,755 | 非零 |
| 5 | unknown_bank_version | 0 | production max-version 合同下结构为零 |
| 6 | bank_version_mismatch | 0 | decision version 与 request 同拍锁存，结构为零 |
| 7 | bank_version_rollback | 0 | 有序流水线与 drained commit 不允许新版本越过旧 request；rollback 在 commit CAS 层拒绝 |
| 8 | parameter_stale | 39,842 | 非零 |
| 9 | deadline_miss | 46,435 | 非零 |
| 10 | map_decision_missing | 0 | v4 consume 时 decision 按构造存在 |
| 11 | map_alignment_or_action_invalid | 0 | MAP/action alignment 按构造检查 |
| 12 | unexpected_reset_ack | 11,459 | 非零 |
| 13 | leakage_observed | 56,073 | 非零；非 blocking 但进入 leakage/reset FSM |

没有为了“14/14 非零”而伪造不可达内部状态。rollback 通过 10 次单调 CAS 负向提交被实际拒绝；
bit 7 保持零是 production invariant 的正向证据。

## 模式、动作、状态与反简化审计

- 六种 mode 计数：normal 678,059、X recovery 67,049、Z recovery 54,305、hold 4,695、
  reset request 40,391、fallback 127,887；
- 五种 health 与 I/X/Z 三种 action 全部非零；
- 8-bit `fault_run/good_run/fault_cycle_count/leakage_cycle_count/per-fault-count` 在 extreme
  family 均实际到达 255，没有仅检查类型宽度；
- CXXRTL comparator 对 8 个独立 expected fields 做 shadow bit mutation，8/8 被检测；
- 报告级 8 个语义 mutation 覆盖 mismatch、规模缩小、undefined action、silent overflow、
  fault branch 删除、rollback 删除、越界板测 claim 与 reorder 删除，8/8 被拒绝；
- 首轮正式运行虽然全字段 0 mismatch，但审计发现 `drop_duplicate_reorder` 的 reorder 被
  pause/full-FIFO 时序遮蔽且 family-specific gate 不足。没有保留首轮 PASS：修正注入相位、
  加入明确恢复尾段和 family-specific gates 后重新生成 trace 并完整重跑百万周期。

## 复现

```powershell
$env:PYTHONPATH='.'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
& 'C:\ProgramData\anaconda3\envs\DLEnv\python.exe' `
  -m cnn_fpga.benchmark.long_rtl_qualification
& 'C:\ProgramData\anaconda3\envs\DLEnv\python.exe' `
  -m pytest -q tests\test_long_rtl_qualification.py
```

正式运行使用 Yosys 0.67 与 g++ 15.1.0；trace generation `47.08 s`、优化编译
`101.82 s`、10-family parallel CXXRTL `536.26 s`。并行只利用 family 间 reset 隔离缩短
墙钟，每个进程仍执行自己的全部 100,000 cycles。family 9 是唯一写 RAM 的 family 且独立启动，
不存在跨进程 memory dependency。

## 产物

- `cnn_fpga/runtime/fast_production_core_reference.py`
- `cnn_fpga/rtl/gkp_fast_path_qualification_top.sv`
- `cnn_fpga/rtl/long_qualification_cxxrtl_driver.cc`
- `cnn_fpga/benchmark/long_rtl_qualification.py`
- `tests/test_long_rtl_qualification.py`
- `docs/t6_2_2_long_rtl_qualification.json`
- `docs/t6_2_2_long_rtl_qualification_source_data.csv`
- `build/t6_2_2_long_qualification/{qualification_trace.bin,yosys_cxxrtl.log,gpp_compile.log}`

## 剩余边界

本任务不估计随机软错误率，也不覆盖 four-state/X propagation、CDC/亚稳态、真实 framing、
host driver、pinout、bitstream、真板 deadline/jitter/power。CXXRTL 是确定性的 two-state
functional qualification；这些缺口分别由 T6.1.2—T6.1.3、T6.2.3、T6.4 和 T6.9.2 承接。

