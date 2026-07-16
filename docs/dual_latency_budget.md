# T2.4.1 文献系统与项目控制面的双 latency budget

**契约：** `dual-latency-budget-v1`  
**机器同源：** `docs/dual_latency_budget.json`  
**审计快照：** `docs/t2_4_1_dual_latency_budget_validation.json`  
**可执行审计：** `python -m cnn_fpga.runtime.dual_latency_budget --audit`  
**当前结论：** `contract_frozen_no_target_board_measurement`

## 1. 为什么必须是两个 budget

本项目面对的最大时序风险不是“缺一个总数”，而是把不同装置、不同语义和不同证据等级的
数字拼成一个看似完整的闭环。这里冻结两个**不可相加、不可相减、不可求比**的 lane：

1. `literature_system`：外部论文的具名装置实测或数值模型假设；
2. `project_control_plane`：本项目配置假设、容量下界，以及仍为 `null` 的目标板测量字段。

Sivak 的 Virtex-6、ADC/DAC 与 cavity/transmon 装置不等于本项目 Tang Nano 20K；Puviani
的 5/10 us 是数值模型时间轴；本项目配置中的 `t_fast_us=5` 也不是二者的复测。机器合同
把这三种恰好相近的数字永久分开。

## 2. 术语冻结

| 术语 | 本文唯一含义 | 不得混用为 |
| --- | --- | --- |
| measurement chain | 外部装置从 readout pulse 到 outcome bits 可消费的物理/电子链 | 本项目 measurement latency |
| ADC acquisition | ADC 或等价 readout 采集窗口 | FPGA inference |
| FPGA DSP | 具名 FPGA 的 demodulation/integration/thresholding 或数字 core | 整条反馈链 |
| waveform generation / DAC output | 波形产生和 DAC 输出；Sivak 原文只明确 DAC | 低价板已集成 AWG |
| transport latency | host/bridge/serial 与板之间的数据传输 | on-chip core |
| on-chip core latency | 数据已在目标 FPGA 内的 cycle-counted kernel | transport/end-to-end |
| end-to-end digital replay latency | host frame 到 board response/action acknowledgement 的数字全路径 | quantum feedback |
| physical action latency | DAC/AWG/微波链作用到量子系统的时间 | software writeback/commit |

## 3. Lane A：外部文献系统

### 3.1 Sivak 2023 实验装置

来源补充材料明确写的是 VPXI-ePC 上两张 X6-1000M control cards，每张集成 DAC、ADC、
DIO 和 Xilinx Virtex-6 FPGA；DAC 为 500 MS/s、16 bit。它是**外部超导 cavity/transmon
实验装置**，不是 Tang Nano 20K 板测。

#### Measurement 分项

| 分项 | 时间 | 语义 |
| --- | ---: | --- |
| readout pulse | 700 ns | 含原文所述 40 ns ramp-up/down |
| signal travel delay | 300 ns | reflected signal 开始 acquisition 前的传播延迟 |
| acquisition | 1400 ns | readout/ADC acquisition window |
| FPGA DSP | 332 ns | demodulation、filter integration、thresholding |
| bit distribution | 100 ns | `s0/s1` 分发到所有 control cards |

这里**不报告 measurement 总和**。原文没有声明 700 ns pulse、300 ns travel 和 1400 ns
acquisition 是无重叠的串行区间；盲目相加成 2832 ns 会制造不存在的时序语义。需要总链
时，应以原始 timing diagram 或统一 timestamp 重新定义端点。

#### SBS/reset 与 Table S3

| scope | 时间 | 解释 |
| --- | ---: | --- |
| SBS prose | 1546 ns | 最高 gain circuit 的正文值 |
| SBS 四层求和 | 1548 ns | `502+708+262+76`；保留 2 ns 源内差异 |
| SBS table block | 1596 ns | 四层加 24 ns enter + 24 ns exit |
| reset prose subroutine | 2332 ns | 正文子程序 scope |
| reset table block | 2380 ns | `24+300+1400+332+100+200+24` |
| one X or Z constituent | 4924 ns | `24+1596+2380+448+452+24` |
| chronological X+Z full cycle | 9848 ns | `2 x 4924` |

`1546/1548` 和 `2332/2380` 不是可任选的舍入版本：它们的 source scope 不同。正式论文
引用时必须保留该差异，不能为了得到“漂亮总数”擅自覆盖。

### 3.2 Puviani 2025 数值模型

Puviani 的 Table S1 定义 half cycle：

\[
0.1+0.5+0.7+0.3+0.1+2.3+1.0=5.0\ \mu\mathrm{s},
\qquad \tau_{\mathrm{cycle}}=2\times5.0=10\ \mu\mathrm{s}.
\]

其中 measurement+reset 为 2.3 us。但原文同时说明 reset 是数值执行、没有 gates/pulses，
Hamiltonian feedback/virtual-rotation compensation 也未进入数值模拟。因此这是
`external_model_assumption`，不是实验 measurement/ADC/control/AWG 实测。

## 4. Lane B：本项目控制面

### 4.1 cadence 与软件延迟模型

以下均绑定 `cnn_fpga/config/hardware_hil.yaml`，证据等级为
`project_configuration_assumption`：

| 项目 | 数值 | 允许解释 |
| --- | ---: | --- |
| fast cycle period | 5.0 us | scheduler reference slot |
| fast action budget | 1.5 us | software-emulator budget，不是物理 action |
| window | 2048 samples | 每个 valid window 的样本数 |
| window content duration | 10.24 ms | `2048 x 5 us` |
| window stride | 4000 cycles | emission interval 为 `20 ms` |
| slow update period | 20 ms | scheduler reference cadence |
| slow job budget | 5000 us | software slow-path budget |
| modeled fast path | `1.0 +/- 0.12 us` | sampled emulator distribution |
| modeled slow stage means | `10+60+900+20+5=995 us` | DMA/preprocess/inference/writeback/commit-ack mean sum |

`995 us` 只是五个配置 mean 的算术和；它没有 queueing、correlation、tail、jitter、deadline
miss 或 backlog。上述动态属于 T2.4.2，不能在本 task 中用高斯均值假装闭环已验证。

### 4.2 UART/replay 容量下界

115200-baud UART、8N1 在忽略 header/CRC/ack/retry/flow-control 的最乐观条件下：

\[
t_{2048}=\frac{2048\times10}{115200}=177.7778\ \mathrm{ms},
\qquad
t_{4096}=355.5556\ \mathrm{ms}.
\]

2048 B 是目标 `32x32 uint16` raw-count histogram；4096 B 是当前 software backend 的
`32x32 float32` payload。前者已远超 20 ms emission interval，且仅 payload 所需最低 8N1
line rate 为 `1,024,000 bit/s`。因此 115200 UART 只能用于 bring-up/telemetry/fallback，
不能作为全直方图主数据面。USB-to-SPI 只是候选接口，持续吞吐和 latency 仍为 `null`。

### 4.3 必须保持 null 的目标板字段

| 字段 | 当前值 | 原因 |
| --- | --- | --- |
| Tang Nano 20K on-chip core latency | `null` | 无 target-device cycle/timing report 或板测 |
| measured UART/USB-SPI transport latency | `null` | 无 framed adapter 和 timestamp evidence |
| end-to-end digital replay latency | `null` | 当前 gate 为 `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE` |
| quantum measurement latency | `null` | 无 cavity/transmon/readout 装置 |
| high-speed quantum ADC acquisition | `null` | 目标板没有该前端集成证据 |
| AWG/DAC waveform output latency | `null` | 没有 AWG/DAC/微波链集成证据 |
| physical action latency | `null` | software writeback/commit 不是物理 action |

## 5. 来源锚点

- Sivak control wiring：
  `relative_papers/Real-time_quantum_error_correction_beyond_break-even/...md:385`；
- Sivak measurement breakdown：同文件 `:483`；
- Sivak SBS/reset prose 与 Table S3：同文件 `:891`、`:893`、`:903-907`；
- Puviani model cycle/Table S1：
  `relative_papers/Non-Markovian_feedback_for_optimized_quantum_error_correction/...md:451-455`；
- 项目 cadence/latency model：`cnn_fpga/config/hardware_hil.yaml:14-28,145-167`；
- UART 与测量边界：`docs/low_cost_fpga_boundary.md:69-111`。

机器审计会读取上述行区间并验证 expected fragment；来源漂移不会静默通过。

## 6. 反简化审计与论文边界

可执行审计包含 23 个 gate：双 lane、术语、唯一 ID、封闭 evidence class、source anchor、
Sivak/Puviani 算术与 scope 差异、live YAML binding、window/UART 算术、目标板 `null` 字段、
real-board NO-GO 与跨 lane 禁止聚合。负向测试会篡改 target-measured flag、配置数值、来源
fragment、evidence class、ID、measurement 总和及 cross-lane ratio，确认全部 fail closed。

本 task 允许写“已建立 source-grounded dual latency budget”。不得写“本项目闭环为 5/10 us”、
“Tang Nano 20K DSP 为 332 ns”、“UART replay 已实现”、“ADC/AWG 已集成”或“FPGA 已完成
实时量子反馈”。
