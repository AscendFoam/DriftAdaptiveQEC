# T4.1.2 实验式 history 输入合同

## 1. 结论

慢回路输入冻结为 **256 个 causal cycle × 53 个 observed-only features**，另带独立
`mask` 与 `cycle_indices`。每一行只连接仓库中已经存在的四类生产者：

- `ObservedSyndromeStep`：analog/residual syndrome、`g/e/leakage`、X/Z quadrature phase、run length；
- `RunLengthFSMDecision`：同周期实际施加的 correction、mode 与 parameter-bank 行为；
- `llr_1d + DeployableLLRContext`：有显式 observed-calibration provenance 的 q/p LLR；
- `SchedulerEvent + DualLoopScheduler.snapshot()`：deadline、通信、stale/failure/conflict/commit 与 bank 状态。

该 history 在 cycle `t` 结束后形成，只能用于预测 `t+1` 及以后慢状态/参数建议；它不参与
cycle `t` 的 hindsight action，也不接收 `DriftState`、`SyndromeTruthStep`、hidden regime、
logical truth、recovery depth truth 或 evaluation label。

## 2. 冻结 schema

| Group | Features | 数量 | 语义 |
| --- | --- | ---: | --- |
| analog syndrome | `analog_q/p` | 2 | 测量后未折叠的 observed analog 值 |
| residual syndrome | `residual_q/p` | 2 | 当前周期折叠后的 observed residual |
| observed outcome | `syndrome_{x,z}_{g,e,leakage}` | 6 | 两个 constituent 的严格 one-hot 结果 |
| quadrature phase | `phase_{x,z}_{sin,cos}` | 4 | 避免角度 branch cut 的相位编码 |
| recent action | `action_q/p`、5-mode one-hot、bank switch/local-safe/conflict | 10 | 同周期真实 FSM 决策；history 序列自然形成 recent-action trace |
| soft information | `llr_q/p`、两轴 saturation flag | 4 | registered observed calibration 下的 periodic-GKP LLR |
| run length | `x_e/z_e/leakage_run`、三个 saturation flag | 6 | observed event counter，不是 hidden recovery depth |
| deadline health | fast/slow deadline、communication、window age/value-valid | 5 | scheduler 事件和 snapshot 的同周期状态 |
| parameter update | 6-status one-hot、applied/pending、active version/pending windows 及 saturation | 12 | status 与 actuation fact 分离 |
| record health | `valid`、`crc_ok` | 2 | 输入记录与传输健康 |

总计 `53`。`update_status` 的优先级为
`conflict > failed > stale > committed > staged > none`；`update_applied` 是独立事实。例如同一
cycle 先 commit、后遇到第二 writer conflict 时，status 为 `conflict`，但 `update_applied=1`。
这避免高优先级诊断抹去已经发生的 actuation。

## 3. 对齐、padding 与饱和

1. builder 只接受从 cycle 0 开始、无 replay/无 gap 的序列；observed/action/runtime 三者必须同周期。
2. history 不足 256 时只在左侧补全零行，且 `mask=0`、`cycle_indices=-1`；物理零值不能冒充 padding。
3. 有效区必须连续，保存的 sample 是 read-only copy，未来 append 不可回写历史 prefix。
4. LLR、run length、bank version 和 pending-window count 均显式 clipping；每类可能静默截断的量都有 saturation flag。
5. LLR provenance 只允许 registered observed calibration、online observed estimator 或 fixed deployment calibration；action provenance 同样使用 allowlist。

## 4. 信息泄漏审计

`audit_mapping_for_information_leakage()` 对 nested mapping/list/tuple/finite numeric array 做递归检查：

- 字段名和字符串值同时检查 `truth/hidden/oracle/teacher/label/regime/logical/...` denylist；
- `SyndromeTruthStep`、`DriftState` 对象即使藏在安全字段名下也立即拒绝；
- set、任意 dataclass/object、object-dtype 或非有限数组均 fail closed；
- observed record 必须声明 `deployable_observed_syndrome` scope；cycle、valid 和 run counters 做类型/范围检查；
- scheduler snapshot 和每个 event detail 在抽取 status 前完整审计。

metadata 本身不会写入 feature row；上述检查仍用于阻断调用方借 provenance side channel 混入
truth-bearing payload。Source Data 只保存 seed/cycle、对齐诊断、事件类型和 53 个 deployable feature，
不保存 simulator regime、logical label 或 recovery truth。

## 5. 生产式验证

`cnn_fpga.benchmark.experimental_history_validation` 执行 8 seeds × 2,048 cycles，共
16,384 个 cycle-level Source Data 行。它不是手造 feature 表，而是逐周期连接真实 syndrome
generator、run-length FSM、dual-loop scheduler 和 `llr_1d`。stress workload 包含 fast/slow
budget violation、window stale、通信暂停、FIFO overflow/drop、slow-path failure、external update
conflict、commit、CRC fault 和所有 clipping 分支。

关键覆盖：

- 6 类 update status：`none/staged/committed/conflict/failed/stale = 15458/48/413/63/31/371`；
- 5 类 FSM action：`normal/x/z/leakage/fallback = 1600/721/772/422/12869`；
- observed `g/e/leakage = 19633/11437/1698`（X/Z 合计）；
- scheduler 记录 8,282 次 fast violation、422 次 slow violation、387 次 stale、32 次 slow failure、64 次 conflict；
- 17/17 machine gates 通过，所有 one-hot、finite、padding、prefix immutability、denylist negative probe、source hash 和 producer provenance 均通过。

工件：

- `docs/t4_1_2_experimental_history_validation.json`；
- `docs/t4_1_2_experimental_history_source_data.csv`；
- `tests/test_experimental_history.py`；
- `tests/test_experimental_history_validation.py`。

## 6. 非 demo 审计与 claim 边界

深审计中修复/补强了四个容易被 demo 掩盖的点：同周期 commit+conflict 的双事实语义、
bank/pending clipping 的显式 saturation、字符串值/任意对象的泄漏入口、以及伪造 observed
cycle/run/valid 类型。验证不是只跑正常路径，而是要求全部 status、FSM mode、outcome、deadline、
通信、CRC、失败和饱和分支均出现，并用可重算 SHA-256 绑定 Source Data。

允许声称：仓库已有一个 causal、observed-only、producer-connected 的 256-cycle software history
合同。禁止声称：IQ/ADC/device calibration 已完成、LLR calibration 已在真实设备注册、目标板 timing/
wire/RTL 已验证、T4.1.1 HMM 已兼容该 richer schema、learned model 或 logical/control gain 已成立。

