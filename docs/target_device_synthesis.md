# T5.5.2 目标器件综合与布局布线估计

## 结论

T5.5.1 fast path 已在参考目标 `GW2AR-LV18QN88C8/I7`（Tang Nano 20K、
`GW2A-18C` family）上完成真实 Yosys synthesis 和三次独立 nextpnr place-and-route。
修复 activity-harness 配置地址越界并重跑后，三个 seed 的 Fmax 为
`40.4318 / 39.8661 / 39.7456 MHz`，最差值仍高于冻结的
`27 MHz` 时钟；正式 verdict 为
`TARGET_DEVICE_POST_ROUTE_ESTIMATE_PASSES_27MHZ_NOT_BOARD_MEASURED`。

这是一条 source-bound、target-specific 的开源工具链 post-route estimate。它不是 Gowin
vendor timing signoff、bitstream 下载或板级测量，也没有把 activity harness 冒充 T6 transport。

## 被测对象与前置等价性

- core：`cnn_fpga/rtl/gkp_fast_path_core.sv`；
- synthesis top：`cnn_fpga/rtl/gkp_fast_path_synth_top.sv`；
- 时钟/引脚约束：`tang_nano_20k_27mhz.sdc` 与
  `tang_nano_20k_synth_harness.cst`；
- 固定参数：四个 257×22-bit memory images；RTL 中镜像为八个合法 1R1W physical memories；
- 父证据：`docs/t_risk_20260716_01_rtl_equivalence.json`，4,316 个 valid MAP rows、
  output/state/commit/version 全字逐周期 `0 mismatch`。

小引脚 top 用 LFSR 驱动输入，以分段寄存器和轮转 signature 保持全部 core output/config path
可观测。它只防止综合器把 core 优化为空，不增加或替代 core 的 6-cycle latency contract。

## 工具与命令

正式环境固定为 YoWASP Yosys `0.67.0.0.post1190`、YoWASP nextpnr Himbaechel Gowin
`0.10.0.0.post753`、Apycula `0.31`。runner 保存完整 executable、版本、命令、源文件哈希、
原始 log/report 哈希：

```powershell
$env:PYTHONPATH='.'
python -m cnn_fpga.benchmark.target_device_synthesis --run-tools
```

不加 `--run-tools` 时只从现有 build directory 重建正式摘要和 durable artifacts；缺少任一原始
report 会 fail closed，不会用公式或 Python proxy 补造资源/时序值。

## 三 seed 时序结果

| seed | Fmax (MHz) | critical period (ns) | logic (ns) | routing (ns) | 起点 | 终点 |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 40.4318 | 24.7330 | 8.9390 | 15.5280 | `core.leakage_clean_run_DFFRE_Q_1` | `fold5_DFFR_Q_6` |
| 7 | 39.8661 | 25.0840 | 12.7050 | 12.1130 | `core.leakage_clean_run_DFFRE_Q` | `fold5_DFFR_Q_6` |
| 19 | 39.7456 | 25.1600 | 13.2700 | 11.6240 | `core.phase_frame_x_DFFRE_Q` | `fold5_DFFR_Q_6` |

报告采用全部 seed 的 minimum/median/maximum，而不是选择最有利 seed：
`39.7456 / 39.8661 / 40.4318 MHz`，最差 seed 对 27 MHz 的 margin 为
`12.7456 MHz`。critical path 从 core 状态寄存器穿过状态/CRC 组合逻辑，到 activity
harness 的 `fold5` 观测寄存器，因此是包含观测折叠端点的保守 top-level 路径；不是 transport path。

## 资源结果

下表的 `used` 取三个 seed 的最大值；available 来自目标器件数据库。

| resource | used | available | 最大占用率 |
| --- | ---: | ---: | ---: |
| LUT4-equivalent | 3,362 | 20,736 | 16.21% |
| DFF | 865 | 15,552 | 5.56% |
| BSRAM | 8 | 46 | 17.39% |
| MULT18X18 | 1 | 48 | 2.08% |
| MULT9X9 | 1 | 96 | 1.04% |
| ALU | 340 | 15,552 | 2.19% |
| IOB | 18 | 384 | 4.69% |

Yosys pre-ABC9 structural report 另保留 8×`SDPX9B`、1×`MULT18X18`、1×`MULT9X9` 和
865 个寄存器，且 `check` 为 0 problems。YoWASP 日志在 ABC9 调用处结束，不能从该日志声称 final
LUT1--LUT4 数；LUT 只使用 nextpnr post-route utilization。唯一 warning 是
小型 `fault_counts` register array 被展开为寄存器；参数表本身均映射到 BSRAM。

## latency estimate

core 的位精确父合同仍为 5-stage MAP + 1-stage action：latency `6 cycles`、II `1 cycle`。

- 27 MHz 目标时钟：core latency `222.222 ns`，II `37.037 ns`；
- 三 seed 最差 Fmax：对应 core latency `150.953 ns`；
- 均不包含 ADC/IQ、CDC、host transport、physical actuation 或 quantum cycle latency。

## 非 demo 审计与修复

执行过程中没有接受“空 top 能综合”作为完成：

1. 起始仓库无 HDL，先插入 T-RISK-20260716-01 并完成 full-word CXXRTL 对拍；
2. 初版四个 2R1W memories 无法映射目标 BSRAM，改为八个写广播、读镜像的 1R1W memories；
3. conditional memory read 不能合并同步输出寄存器，改为八路无条件同步读后选择 bank/phase；
4. 初版单体 rolling signature 把 350-bit XOR 树放到关键路径，改为七段寄存器和轮转消费；
5. seed 7/19 的 PowerShell 重定向日志是 UTF-16，runner 增加 BOM-aware 解码并转存 UTF-8；
6. 复核发现 raw 9-bit LFSR 配置地址可能越过 257-entry ROM；改为 modulo-257，并穷举 512 个
   输入码后重新综合/P&R；
7. runner 的 version query 改用 workspace-local YoWASP cache；并行 seed 失败会保存日志并串行 retry；
8. 正式证据使用三个固定 seed、12/12 machine gates、9 类语义 shortcut mutation 和
   20 个 direct tests；删除 seed、伪造 device/Fmax/BRAM/critical path/hash/latency 或冒称板测均被拒绝。

## 正式产物与证据边界

- 摘要：`docs/t5_5_2_target_device_synthesis.json`；
- Source Data：`docs/t5_5_2_target_device_synthesis_source_data.csv`；
- Yosys log：`docs/t5_5_2_yosys_synthesis.log`；
- 三 seed nextpnr detailed reports/logs：`docs/t5_5_2_nextpnr_seed*_report.json` 与
  `docs/t5_5_2_nextpnr_seed*_place_route.log`。

已建立：synthesizable RTL、目标器件 synthesis、目标器件 place-and-route estimate。

仍未建立：vendor timing signoff、bitstream、真实开发板、transport、板上 latency/throughput/power、
量子硬件测量。T5.5.3/T5.5.4 继续比较部署点与模型分支；任何真板 claim 留给 T6。
