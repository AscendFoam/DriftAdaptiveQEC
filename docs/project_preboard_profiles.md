# T6.19.1 项目原生预板 profile

> 结论边界：这是 CXXRTL 等价、目标器件 P&R 估计与当前主机软件计时；不是板测延迟、功耗、抖动或 deadline 证据。

## 硬件资格表

| 方法 | 资格 | cycles / II | 27 MHz source-to-action | LUT4 / FF / BSRAM / DSP |
|---|---|---:|---:|---:|
| CI_if_rtl | N_A_NO_INDEPENDENT_CI_RTL | N/A | N/A | N/A |
| static_map_lut_if_rtl | ELIGIBLE_EXISTING_SOURCE_BOUND_RTL | 6 / 1 | 222.222 ns | 3379 / 865 / 8 / 2 (seed 7) |
| v5_fast_path_if_rtl | N_A_NO_V5_RTL_EARLY_STOP_AT_T6_10_1 | N/A | N/A | N/A |
| eligible_direct_nn_if_rtl | N_A_NO_SAME_TASK_ELIGIBLE_DIRECT_NN_RTL | N/A | N/A | N/A |

static MAP-LUT 的 CXXRTL 比对覆盖 4316 个有效 action 行，mismatch=0；三种子均通过 27 MHz。

## 软件慢路径（当前主机诊断）

| 方法 | update p50/p99 | compiler p50/p99 | software transfer p50/p99 | software commit p50/p99 | MAC / state / workspace |
|---|---:|---:|---:|---:|---:|
| Window | 105.500/349.754 us | 1005.000/1985.096 us | 1404.700/1931.094 us | 13.300/36.806 us | 128 / 32 B / 512 B |
| EWMA | 136.900/416.012 us | 1244.700/2464.731 us | 1568.350/2571.123 us | 18.000/51.643 us | 136 / 80 B / 512 B |
| Kalman | 313.700/846.339 us | 1178.750/2326.636 us | 1641.650/3189.524 us | 19.400/53.608 us | 7121 / 1740 B / 2048 B |
| V5_if_exists | N/A | N/A | N/A | N/A | N/A |

## 可用与禁用表述

- 可用：现有 static MAP-LUT 在 GW2AR-LV18QN88C8/I7 的三种子 P&R 中满足 27 MHz，并有 6-cycle、II=1 的 source-bound RTL/CXXRTL 证据。
- 资源范围：完整 `gkp_fast_path_synth_top`（MAP-LUT、event/fault/state 与小引脚 harness），不是 MAP ROM 单体面积。
- 禁用：CI、V5 或 Direct NN 已在同一 FPGA 上更快；当前没有相应合格 RTL。
- 禁用：把 Python update/compiler/内存事务时间写成 FPGA latency、真实传输或板级 commit。
- 所有 power/jitter/deadline/board-measured 字段保持 null，等待 T6.9.2。

Gate：12/12；mutation：13/13。
