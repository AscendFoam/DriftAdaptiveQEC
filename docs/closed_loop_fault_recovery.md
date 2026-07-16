# T4.3.3 闭环稳定性与故障恢复

## 1. 结论与范围

T4.3.3 用 `ClosedLoopFaultRecoverySupervisor` 将 T4.3.2 完整 image 原子库、T4.2 bit-accurate MAP→health→event→frame fast path 和 T4.3.1 的 `5 us / 4000-cycle / 8192-cycle` cadence 接成同一 software closed loop。每个 fast cycle 都产出确定 action、mode、reason trace、active bank/version/hash 和 bounded frame；blocking fault 下不执行 correction，host/通信失败不阻塞 fast path。

这是 `closed_loop_software_fault_recovery_contract_not_rtl_or_board`。结果不证明 CDC/RTL、FPGA/board、真实装置 leakage/reset efficacy、物理 logical lifetime stability 或 device-calibrated timeout/OOD 阈值。

## 2. 恢复语义

### 2.1 ack/readback

内部 atomic commit 与 host 确认分离。通信中断时 commit 可以在安全 cycle boundary 生效，但 host 状态保持 `awaiting_ack_readback`；超过 400 cycles 后变为 `ack_timeout_awaiting_readback`，仍不盲目 rollback，也不允许下一 writer。通信恢复后必须 readback bank/version/activation epoch/image CRC/SHA 才确认。

### 2.2 post-commit guard 与 LKG

新 candidate commit 后保留 4000-cycle guard。guard 内累计两个 blocking faults 即 `post_commit_guard_fault`，候选不能成为 last-known-good（LKG）。恢复不是把 active version 从 v1 降回 v0，而是把 v0 的操作内容重新编译/封装为 v2 完整 image，经同一 CRC/SHA/CAS/hysteresis/readback 路径发布。这样 rollback 内容恢复可审计，同时 active version 单调，T4.2 FSM 不会遇到 version rollback。

LKG republish 前会计算排除 version/自校验字段的 semantic SHA256；v2 必须与 LKG 内容一致。相同 version 但未注册到 T4.2 fast path 的另一张 image 在 stage 前直接以 `unregistered_fast_path_image` 拒绝，避免 bank 和 decoder 对同一 version 使用不同表。

### 2.3 timeout/stale

host heartbeat age 超过 8192 cycles，或 active image age 超过 8192 cycles，fast path 进入 traceable frame hold，禁止 correction。恢复可走 LKG republish，也可由新的完整、已注册 candidate 在 readback 后经过无故障 guard 清除 stale recovery；`post_commit_guard_fault` 不能通过原候选自我清除。

### 2.4 周期 refresh

首轮 24k-cycle stress 暴露：若只在 6200 周期提交一次，即使 host 一直在线，active image 也会在 8192 cycles 后按设计 stale。最终 campaign 没有放宽 freshness 门，而是在正常通信场景加入 14000/22000 周期的完整 monotonic refresh；guard-republish 场景在 18200 refresh。该修复把 4000-cycle slow cadence 与 8192-cycle age policy 真正闭合。

## 3. 故障 campaign

production run 为 8 场景 × 4 seeds × 每条 23996 cycles，共 `32` runs、`767872` 个逐周期动作；Source Data 用状态段压缩为 436 rows，但每周期 record 都进入 trace SHA256 和门统计。

| 场景 | 注入与关键时序 | 验证结果（每 seed） |
| --- | --- | --- |
| nominal drift | 两频率整数 ADC drift；6200 边界故意 unsafe，下一周期提交 | 670 左右 unique ADC codes；commit `6201/14200/22200`；全程无 fallback |
| burst | 64 cycles `e` burst、OOD 224、deadline false | OOD/deadline 各 64；65 cycles frame hold（含 clear hysteresis）；最终 healthy |
| leakage/reset | 连续 3 cycles leakage，按真实 FSM reset ack | leakage 3、reset request 1、4 个 non-map cycles；最终 healthy |
| host timeout | heartbeat 在 7000 后停止，17000 恢复 | parameter stale 2807、deadline miss 1807、fallback 2808；v2 在 17200 fresh/LKG republish，最终 healthy |
| communication pause + ack loss | 6100–6999 通信中断，v1 在 6200 内部 commit | ack awaiting 401 cycles、timeout-uncertain 399 cycles；7000 readback 确认；未重复写入 |
| corrupt transfer | payload 单字节翻转，随后完整 v1 重试 | 坏包 `transfer_crc_mismatch` 且不激活；仅因重试前 age 边界产生 7 stale cycles；v1 在 8200 提交并在 guard 后清除 recovery |
| update race | 两线程同时 submit 完整 v1 | 恰一 staged winner、恰一 `writer_conflict_pending_commit`；之后 cadence refresh 正常 |
| guard LKG republish | v1 在 6200 commit，6201/6202 报告 image CRC/SHA mismatch | v2 因 4000-cycle residency 延至 10200 commit，内容 semantic hash 回到 v0；v3 在 18200 refresh；最终无 pending recovery |

## 4. 非 demo 门

17/17 gates PASS：

- 所有 767872 cycles 的 undefined action、blocking-fault correction 和 frame out-of-range 计数均为 0；
- active version 在 32/32 runs 单调，不使用版本回退伪装 rollback；
- burst、leakage、host timeout、communication pause/ack loss、bad transfer、race、guard failure 都有显式 reason/action；
- nominal drift 覆盖超过 500 个 ADC codes，不是常数输入；
- online `tick` 无 payload mutation loop，runtime 不读取 truth、oracle、hidden state 或 target params；
- 完整 guard campaign 独立重跑的 per-cycle trace SHA256 和 commit epochs 相同；
- JSON 绑定 runtime/atomic/fast/fallback/FSM/validator 实际执行源码，CSV 绑定 SHA256；public API 另由 direct import test 锁定，避免无关导出使历史 artifact 漂移。

## 5. 产物与复验

- runtime：`cnn_fpga/runtime/closed_loop_fault_recovery.py`
- validator：`cnn_fpga/benchmark/closed_loop_fault_recovery_validation.py`
- machine result：`docs/t4_3_3_closed_loop_fault_recovery_validation.json`
- Source Data：`docs/t4_3_3_closed_loop_fault_recovery_source_data.csv`
- tests：`tests/test_closed_loop_fault_recovery.py`、`tests/test_closed_loop_fault_recovery_validation.py`

```powershell
python -m cnn_fpga.benchmark.closed_loop_fault_recovery_validation
python -m pytest -q tests/test_closed_loop_fault_recovery.py tests/test_closed_loop_fault_recovery_validation.py
```

## 6. 尚未完成

- 4000/8192/400-cycle、OOD 192/193、两 fault guard、leakage/reset 条件均为 software policy，不是 device calibration。
- Source Data 的状态段压缩不等于真实 wire log；transport sequence/CRC/FIFO、CDC/RTL watchdog、板级 readback 与真实时钟抖动留给 T5.5/T6。
- 本任务检验 control action safety/stability，不给出物理 LER/lifetime/fidelity gain；跨模型、OOD 和 long-horizon 结论由 T5.1/T5.4 承接。
- 自动物理 rollback 未实现；这里是 monotonic LKG content republish。论文必须同时说明这一点，不能只写“automatic rollback”。
