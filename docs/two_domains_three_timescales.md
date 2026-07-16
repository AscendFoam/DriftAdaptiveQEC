# T1.4.3 两个计算域、三个时间尺度接口契约

**Task ID：** `T1.4.3`  
**契约版本：** `two-domains-three-timescales-v1`  
**日期：** 2026-07-14  
**状态：** Done  
**机器同源：** `docs/two_domains_three_timescales.json`

## 1. 冻结结论

系统只分两个计算域：

| 域 | 负责 | 绝不负责 |
| --- | --- | --- |
| `CD1` FPGA deterministic/safety | 同周期 bounded action、本地 event FSM、health counter、active bank、cycle-boundary commit、last-known-good/safe-static | 等待 host、在线训练、无界优化、读取 hidden truth、把迟到 action 移到下一周期 |
| `CD2` host estimation/optimization | 验证窗口、observed feature、deployable estimator/teacher replay、完整 inactive-bank proposal、跨窗口/跨 run 重校准 | 进入逐周期 critical path、直接写 active bank、把 send 当 commit、部署模式读取 `target_params` |

系统只定义三个在线时间尺度：`TS1` 逐子周期动作、`TS2` 窗口事件/健康聚合、`TS3` host
慢速估计/优化。事件**被观测**可以发生在每个子周期；“窗口级”只表示计数、趋势和 host
notification 的聚合节拍。leakage/reset/deadline 等安全动作不得等到窗口结束才执行。

## 2. 三时间尺度

| ID | 主 owner | 输入 | 输出 | deadline / failure |
| --- | --- | --- | --- | --- |
| `TS1` per-subcycle fast path | `CD1` | valid flag、wrapped syndrome/event、latched bank/version、local frame/health | bounded correction/frame、action kind、flags/counters、actual-version trace | 同一 protocol boundary；当前配置 5 us subcycle、1.5 us budget，只是 reference。miss 时抑制 late action并安全 hold，不能挪到下一 cycle |
| `TS2` window event/health | `CD1` 主、`CD2` 观察 | raw counts、`n_valid`、event/run-length、saturation/deadline/fallback/lost-trace、version distribution | versioned header/CRC、observed statistics、health transition、ordered notification | urgent transition 下一个安全 cycle boundary 本地执行；serialization 可滞后但 sequence/loss 可追溯；坏整窗丢弃、不阻塞 TS1 |
| `TS3` host slow path | `CD2` | 仅 observed validated window、prior active version/LKG、bounded estimator state | 完整 bank + schema/version/source/timestamp/apply-epoch/CRC，随后 commit/ack/readback | 非逐周期 deadline；当前 20 ms start period、5 ms software budget。失败或 stale 不 stage，LKG 到 max age 后由 FPGA 切 safe-static |

### 2.1 当前 reference timing 的正确身份

| 字段 | 当前值 | 身份 |
| --- | ---: | --- |
| fast subcycle / action budget | 5.0 us / 1.5 us | software-HIL configuration，不是 FPGA measurement |
| window size / content duration | 2,048 valid samples / 10.24 ms | 按 5 us 算术 |
| stride / emission interval | 4,000 cycles / 20 ms | configuration |
| slow start period / job budget | 20 ms / 5 ms | scheduler/latency-model contract |
| pending windows / commit delay | 2 / 1 cycle | software runtime policy |
| max bank age | 未设置 | 必须在 T4.3 前冻结，当前不能声称 stale fallback 已完成 |

所有这些值都不能继承实验论文的装置时序身份，也不能写成 Tang Nano 20K 的板测值。

## 3. 跨域接口

### XIF01：observed window，`CD1 -> CD2`

必须有 schema、sequence、window/epoch range、active version、`n_valid`、flags、length 和 CRC。
接收是 all-or-nothing；坏 header/CRC/count 不允许靠补零进入 estimator。canonical data 是
`32×32 uint16 raw counts`，host adapter 才归一化。

### XIF02：inactive bank staging，`CD2 -> CD1`

必须有 schema、new/expected-active version、source window、created/apply epoch、length、CRC、
`K[2,2]` 和 `b[2]`。payload 先写完、validated header 最后置 valid；host 永不写 active bank。

### XIF03：commit command，`CD2 -> CD1`

目标 bank、expected active version、new version 和 future apply epoch 组成 CAS 请求。commit 只允许
在 cycle boundary 一次切换，不能让一个 cycle 的乘加读取两个版本。

### XIF04：ack/readback，`CD1 -> CD2`

返回 accepted/reason、active bank/version 和 commit epoch。host 只有读回期望 active version 才能
宣布成功；串口 write 完成、USB transaction 完成或 scheduler 已 stage 都不是 commit success。

## 4. 原子更新序列与真实实现状态

| ID | 规则 | 当前状态 |
| --- | --- | --- |
| AP01 | cycle start 锁存一个 active bank/version | software 已有 |
| AP02 | host 完整写 inactive bank | software K/b 已有 |
| AP03 | shape/finite/range/schema/length/CRC 全验证 | 仅 shape + finite |
| AP04 | monotonic version + expected-active CAS | 自动 version 已有，无 CAS |
| AP05 | cycle boundary、该 cycle fast action 前一次切换 | software 已有；direct test 验证 |
| AP06 | ack reason + active-version readback | 未实现真 I/O |
| AP07 | old bank 保留并可 rollback | payload 留在另一 bank，无 rollback FSM |
| AP08 | source freshness/max age | 未实现 |
| AP09 | 最多一个 slow job/一个 pending commit | scheduler 有，direct `stage_update` 可覆盖 pending |

当前 `tick_with_fast_path()` 的真实顺序是：epoch++ -> 记录 fast latency -> `commit_if_ready` ->
finish/stage slow job -> fast callback -> emit window -> start slow job。因而正常 scheduler 路径中，
新 bank 在 fast callback 前切换，并且 slow job 当周期新 stage 的 bank 最早下一周期提交。

## 5. 失败分支

| ID | 事件 | 必须动作 | 当前差距 |
| --- | --- | --- | --- |
| FB01 | invalid/non-finite observation | hold/frame-only/reset-request，不把 NaN 当 0 | validity gate 缺失 |
| FB02 | fast deadline miss | 抑制 late action、reason counter++ | 当前只记录，callback 仍执行 |
| FB03 | input/correction saturation | bounded clip+flag；持续超限降 health/safe-static | 当前 clip/count，无自动降级 |
| FB04 | g/e/leakage/recovery run | 本地 FSM 立即处理，不等 host/window | protocol event FSM 缺失 |
| FB05 | bad window schema/length/CRC/n_valid | 整窗拒绝，不生成 update | wire validation 缺失 |
| FB06 | queue full | 丢最旧、保最新、计 loss，不阻塞 fast | software 已有 |
| FB07 | estimator/DMA/inference failure | 不 stage，active LKG 不变 | exception path 已有 |
| FB08 | slow miss/stale result | freshness/apply epoch 不合格即拒绝 | budget 只记录，无 stale gate |
| FB09 | bad bank/CRC/range/version/CAS | inactive reject，active 不变 | 仅 finite/shape |
| FB10 | ack timeout/readback mismatch | 不假定成功，readback/rollback | 缺失 |
| FB11 | host timeout/max age | LKG 到 age limit，随后本地 safe-static | max-age FSM 缺失 |
| FB12 | sequence gap/duplicate/corruption | reject/retry、计 loss、TS1 继续 | serial adapter 缺失 |
| FB13 | boot/reset/no valid bank | 预验证 safe-static v0 | 当前默认 identity 未被验证为 safe bank |
| FB14 | deployable estimator 读取 hidden truth | 结果失效；仅 explicit oracle/test lane 可读 | mode 有区分，payload channel 尚未物理分离 |

任何失败都必须落到定义动作；不存在“继续使用半包参数”“迟到 correction 顺延”“host 单方面
宣布成功”或“hidden truth 填补缺字段”的路径。

## 6. Hidden-truth 防泄漏

`FastLoopEmulator` 当前 window payload 同时含 observed histogram/diagnostics 和模拟真值
`target_params`。后者只服务明确标注的 `mock` / `oracle_delayed` 上界，不属于 `XIF01` wire。

部署型 window/EKF/UKF/PF/CNN/TCN/GRU/HMM/student 只能读取 observed channel、prior state 和
明确可在线获得的 metadata。训练时 label、simulation hidden state 和 future window 可以保存在
separate truth file，但不得进入 online input tensor。只要 deployable result 消费 `target_params`，
该 run 立即降级为 oracle/test evidence，不能进入性能主表。

## 7. 当前代码证据与非 demo 审计

| 检查 | 结果 |
| --- | --- |
| 是否只画三条箭头 | 否；2 domains、3 scales、4 interfaces、9 atomic steps、14 failure branches 均机器化 |
| 是否验证真实 commit 顺序 | 是；direct scheduler test 证明 epoch 1 commit 后 fast callback 只见 version 1 完整 bank |
| slow failure 是否误写成功 | 否；direct test 证明 `slow_update_failed` 且 active version 保持 0、无 pending commit |
| deadline 是否被虚假写成 enforced | 否；direct test 锁定当前 fast miss 仍执行 callback 的缺口 |
| queue 是否无界 | 否；depth=2、drop-oldest；direct test 验证不阻塞 fast tick |
| event safety 是否错误等待窗口 | 否；contract 明确 urgent action 属于 CD1/TS1，TS2 只聚合 |
| hidden truth 是否混入部署输入 | contract 明确禁止，并以 FB14 fail closed |
| timing 是否冒充硬件 | 全部标为 configuration/software model，CL3 之前不升级 |

本 task 完成的是接口和失败语义冻结，不是 CRC/FSM/serial/RTL/board 实现。缺口被精确路由到
T2 protocol、T4.2/T4.3 safety/atomicity、T5.5 golden/RTL 和 T6 board replay/negative path。

## 8. 论文表述边界

允许写：two-domain, three-timescale digital control architecture；per-cycle deterministic fast path；
window health aggregation；host-proposed atomic parameter updates；software contract and emulation。

不得写：hard real-time deadline achieved、safe fallback fully implemented、atomic board update verified、
real-board transport closed、quantum feedback loop demonstrated。
