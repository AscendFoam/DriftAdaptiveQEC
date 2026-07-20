# 可综合 fast-path RTL 与 T5.5.1 等价性

## 1. 为什么插入该任务

T5.5.2 启动时仓库没有任何 HDL。对空 top、组合 LUT demo 或 Python resource proxy 报告 Fmax/LUT
都会产生伪综合证据，因此按 R-N104 插入 `T-RISK-20260716-01`，先建立合法被测对象，再恢复目标
器件综合。

## 2. RTL 边界

`cnn_fpga/rtl/gkp_fast_path_core.sv` 实现：

- 58-bit input CRC-16/CCITT-FALSE 重算与 reserved observation fail-closed；
- 两 bank、X/Z 两 phase、257×signed-22-bit MAP tables；
- half-bin linear interpolation、signed round-to-nearest ties-to-even 与 22-bit saturation；
- 5-cycle MAP + 1-cycle action register，II=1；
- safe-boundary bank commit，S0 锁存 bank/version，旧 in-flight request 不被切换污染；
- six-mode event FSM、fallback/health、Pauli/phase frame、6 个 3-bit counters、4 个 8-bit health
  counters、14 个 fault counters；
- 118-bit output 和 232-bit state CRC words。

配置传输的 CRC32/SHA256/manifest/CAS/hysteresis 仍属于 host transaction 层；device core 只接受已经
验证的 `bank_trusted` 与 safe commit。`gkp_fast_path_synth_top.sv` 是小引脚 activity harness，不是
UART/USB-SPI transport。

## 3. 非 demo 修复

首版每张逻辑表需要 `2 read + 1 write`，Yosys 对 GW2A 报 `no valid mapping found`。最终实现把每张
表镜像成 y0/y1 两个物理 copy，配置写广播、每个 copy 只有 1R1W；同时把 8 个 memory output 直接
注册后再选 bank/phase。由此保留 II=1，并能映射为 8 个 Gowin `SDPX9B`，而不是退化为 LUTROM 或
删除写端口。

## 4. CXXRTL 等价验证

固定工具为 YoWASP Yosys 0.67 和 MinGW g++ 15.1。runner 从当前 RTL 重新生成 CXXRTL、编译并运行，
不读取手写软件镜像：

| 场景 | cycles | valid MAP rows | mismatch |
| --- | ---: | ---: | ---: |
| fault + deferred/atomic commit | 226 | 220 | 0 |
| v0/v1 × X/Z × 1,024 codes | 4,102 | 4,096 | 0 |
| 合计 Source Data | 4,328 | 4,316 | 0 |

每行比较 active version、commit ack、map valid/address/signed LLR、118-bit output 和 232-bit state。
CRC corruption、invalid observation、deadline、leakage/reset、OOD、e-run 与 unsafe-boundary commit 均在
连续 trace 内。8 类 row-count/latency/LLR/address/output/state/commit/version mutation 全被 checker 拒绝。

正式 verdict 为
`SYNTHESIZABLE_RTL_EQUIVALENT_FOR_V0_V1_NOT_BOARD_MEASURED`，8/8 gates 通过。

## 5. 证据边界

本任务升级的是“可综合 RTL + CXXRTL 等价”字段，不升级 target synthesis、post-route timing 或 board。
T5.5.2 独立保存器件/工具/约束/report；T6 才能实现 transport、下载 bitstream 并做真板测量。

## 6. 产物

- `cnn_fpga/rtl/gkp_fast_path_core.sv`
- `cnn_fpga/rtl/gkp_fast_path_synth_top.sv`
- `cnn_fpga/rtl/cxxrtl_trace_driver.cc`
- `cnn_fpga/rtl/generate_frozen_memories.py`
- `cnn_fpga/rtl/generated/`
- `cnn_fpga/benchmark/rtl_fast_path_equivalence.py`
- `tests/test_rtl_fast_path_equivalence.py`
- `docs/t_risk_20260716_01_rtl_equivalence.json`
- `docs/t_risk_20260716_01_rtl_equivalence_source_data.csv`

