# GKP fast-path RTL

本目录保存 T5.5.1 Python golden 的可综合 fast-path 对应物。`gkp_fast_path_core.sv`
覆盖 58-bit CRC input、双 bank 257×22-bit phase LUT、精确 ties-to-even 插值、5-cycle
MAP + 1-cycle action pipeline、event/fallback FSM、frame/counter state，以及 118/232-bit
CRC output/state words。

`gkp_fast_path_synth_top.sv` 只是低引脚数的综合 activity harness：它以 LFSR 驱动 core，
并把全部 core output 折叠进 rolling signature，防止综合器把未连接逻辑优化掉。它不是
T6.1 的 UART/USB-SPI transport，不计入 6-cycle core latency，也不能作为板级吞吐证据。

`gkp_fast_path_production_top.sv` 是 T6.2.1 的板卡无关同步管理顶层。它没有假装实现串行
transport，而是在 core 外提供严格 X/Z 514-word 配置事务、CRC32、inactive-bank-only、
16-bit CAS version、safe-boundary deferred/cancel commit、6-cycle retired-bank drain guard，
以及冻结后用 18 cycles 生成 CRC16 的 coherent state snapshot。CRC 只用于偶发损坏检测，
不是 authentication。其独立 reference、CXXRTL driver 和审计入口分别为：

- `cnn_fpga/runtime/production_fast_path_management.py`
- `cnn_fpga/rtl/production_management_cxxrtl_driver.cc`
- `cnn_fpga/benchmark/production_rtl_audit.py`
- `docs/production_rtl_audit.md`

`gkp_fast_path_qualification_top.sv` 与 `long_qualification_cxxrtl_driver.cc` 是 T6.2.2 的
高吞吐资格验证面。它复用 production core 参数，把 10 个 reset-delimited family 各执行
100,000 cycles，并逐周期比较 commit/bank/version、MAP debug、118-bit output 与 232-bit state。
抽象 FIFO/pause/drop/duplicate/reorder 在 Python receiver model 中 fail closed；这不是 CDC、
真实串行 transport、bitstream 或板级时序 harness。报告见 `docs/long_rtl_qualification.md`。

冻结 ROM 由以下命令从 T5.5.1 image registry 机械生成：

```powershell
$env:PYTHONPATH='.'
python -m cnn_fpga.rtl.generate_frozen_memories
```

目标时钟约束为 `tang_nano_20k_27mhz.sdc`；
`tang_nano_20k_synth_harness.cst` 将小引脚 harness 约束到 Sipeed 官方示例已使用的
QN88 clock/reset/LCD pins。后者只用于可重复 P&R，不是最终 transport pinout。器件、
工具版本、综合/P&R 命令和报告由 T5.5.2 的 runner 冻结。

T6.9.2 的真板前 UART 候选由 `route_a_uart_phy.sv` 与
`route_a_uart_board_top.sv` 组成，默认 27 MHz / 3 Mbaud。请求固定 40 bytes，响应固定
96 bytes，均带 CRC32/sequence/version；duplicate 只返回显式状态，不重新执行、不消耗
sequence。commit/config/posterior/fault/reset 等事件位只随 `core_in_valid` 产生单拍脉冲。
`tang_nano_20k_route_a_uart.cst` 仍是待实板 revision 核验的 candidate pinout。
P&R 与 `.fs` 只见 `docs/t6_9_2_preboard_bitstream_candidate.json`，不构成真板 correctness、
source-to-action、deadline 或功耗证据。
