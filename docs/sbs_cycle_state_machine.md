# T2.0.4 sBs Table S3 cycle state machine

**日期：** 2026-07-14  
**实现：** `physics/sbs_cycle_state_machine.py`  
**协议 ID：** `PROTO-SBS-MAIN`  
**时序证据：** `literature_reference_not_target_board_measurement`

## 1. 周期口径

本 FSM 的一个 constituent 是一次 X 或 Z rank-2 dissipator，Table S3 总时长为 `4924 ns`。一个 chronological full cycle 为 X constituent 后接 Z constituent，总参考时长为 `9848 ns`。

所有 trace 都携带 `target_hardware_measured=False`。代码不调用 `sleep`、`perf_counter` 或系统时钟；这里的 ns 是 Sivak 最高增益实验电路的文献时间轴，不是 Python 运行时间、综合 estimate 或本项目低价 FPGA 实测。

## 2. Table S3 逐阶段 FSM

| 顺序 | phase | group | ns |
| ---: | --- | --- | ---: |
| 1 | enter_cycle | cycle_overhead | 24 |
| 2 | enter_sbs | sbs | 24 |
| 3 | sbs_layer_1 | sbs | 502 |
| 4 | sbs_layer_2 | sbs | 708 |
| 5 | sbs_layer_3 | sbs | 262 |
| 6 | sbs_layer_4 | sbs | 76 |
| 7 | exit_sbs | sbs | 24 |
| 8 | enter_reset | reset | 24 |
| 9 | roundtrip_delay | reset | 300 |
| 10 | readout_acquisition | reset | 1400 |
| 11 | signal_processing | reset | 332 |
| 12 | syndrome_distribution | reset | 100 |
| 13 | branch_and_feedback | reset | 200 |
| 14 | exit_reset | reset | 24 |
| 15 | mixer_matrix_calculation | virtual_rotation | 400 |
| 16 | mixer_update | virtual_rotation | 48 |
| 17 | idle | idle | 452 |
| 18 | exit_cycle | cycle_overhead | 24 |

FSM runtime 每次 `advance()` 只产生下一个 phase；未完成时读取 trace、完成后再次 advance 均抛错。不同 runtime 互不共享 transition index。

## 3. Scope-specific 算术

- 四个 layer：`502+708+262+76 = 1548 ns`；正文另报 SBS unitary `1546 ns`，保留 2 ns source discrepancy；
- Table SBS block：`24+1548+24 = 1596 ns`；
- reset 正文 subroutine：`2332 ns`；Table 含 enter/exit 的 block：`2380 ns`；
- virtual rotation：`400+48 = 448 ns`；
- idle：`452 ns`；
- cycle overhead：`24+24 = 48 ns`；
- constituent：`48+1596+2380+448+452 = 4924 ns`；
- full X+Z：`2×4924 = 9848 ns`。

代码不选一个 scope 覆盖另一个。`sivak_table_s3_profile()` 同时保存 prose 和 table 数值，并与 paper-parameter registry 交叉测试。

## 4. Observation、reset、VR 与 frame 接线

`run_full_xz_cycle()` 消费 T2.0.3 的 `SBSObservedCycle`：

- X constituent 使用 observed `syndrome.x` 和 `reset_actions[0]`；
- Z constituent 使用 observed `syndrome.z` 和 `reset_actions[1]`；
- signal processing/distribution event 只携带 observed class；
- branch event 只按 observed class 携带 reset action；
- VR event 使用调用者显式提供的 `g/e/leakage -> calibration key` 和 provenance；
- event metadata 不含 hidden/ideal/truth/carry state。

X constituent 将 Pauli frame 的 x bit 翻转，Z constituent 将 z bit 翻转；一个 full cycle 得到 `(x,z) xor (1,1)`，两个 full cycles 恢复原 frame。VR metadata 保留 square-grid `pi/2` quadrature switch，但不虚构 device-calibrated `theta_g/e/f` 数值。

## 5. 验证与反 demo 审计

`tests/test_sbs_cycle_state_machine.py` 覆盖：

1. 18 phase 名称、顺序、逐项 duration 与 primary source anchor；
2. prose/table scope discrepancy 和所有 group sums；
3. 可逐步推进、未完成/重复推进失败分支；
4. 任意 start offset 下无 overlap、无 gap；
5. branch/VR metadata 只消费 observed inputs；
6. X→Z 组合为连续 9848 ns full cycle；
7. constituent/full-cycle Pauli-frame 更新；
8. 两个 full cycles 的连续调度和 frame cancellation；
9. 全 trace/event 的非实板证据标签；
10. paper-parameter registry 的 4.924/9.848 us、1546/1548 ns、2332/2380 ns 交叉一致；
11. 源码无 wall-clock sleep/measurement；
12. 两个 runtime 状态独立；
13. 坏 control、缺 VR mapping、负 start、重复 phase 与伪 measured scope 全部拒绝。

本任务实现的是事件结构和参考时间轴，不模拟 signal-processing arithmetic、实际分发网络、pulse waveform、jitter、transport 或实板 latency；这些仍需 T4/T6 证据。
