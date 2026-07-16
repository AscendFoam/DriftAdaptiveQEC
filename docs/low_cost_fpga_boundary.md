# T1.4.2 约 300 元 FPGA：板卡、I/O、资源与测量边界

**Task ID：** `T1.4.2`  
**契约版本：** `low-cost-fpga-boundary-v1`  
**日期：** 2026-07-14  
**状态：** Done  
**机器同源：** `docs/low_cost_fpga_boundary.json`

## 1. 冻结结论

第一篇论文的低成本数字控制平面**参考目标**冻结为：

- 板卡：Sipeed Tang Nano 20K；
- FPGA：GOWIN `GW2AR-LV18QN88C8/I7`；
- 预算约束：板卡本体优先不超过 300 元，含运费/税费上限暂定 350 元；采购前必须取得
  当日可追溯报价；
- 当前实物状态：`not_procured_or_physically_verified`；
- 当前证据：厂商规格和仓库兼容性审计，不是综合、bitstream、板测或 HIL。

因此，“实际板卡型号”已经作为部署目标冻结，但**不能写成当前已有一块该板、已经采购、已经
烧录或已经测量**。T6.1.1 只有在实物、成本、照片/身份、工具版本和供电记录齐全后，才能把
状态升级为 actual-board provenance。

## 2. 厂商规格快照

| 项目 | 冻结值 | 证据身份 |
| --- | ---: | --- |
| LUT4 | 20,736 | 厂商规格，不是本项目可用/已占用量 |
| FF | 15,552 | 同上 |
| S-SRAM | 41,472 bit | 同上 |
| B-SRAM | 828 Kbit，46 blocks | 同上；`K` 的换算仅作容量估算 |
| 18×18 multiplier | 48 | 使用厂商名称；不擅自换写为其它厂商的 DSP slice |
| PLL | 2 | 以 Tang Nano 20K v1.3 datasheet 为准 |
| I/O bank / free I/O | 8 / 34 | free I/O 来自 v1.3 datasheet |
| SDR SDRAM | 64 Mbit，32-bit | 标称约 8 MiB；不是带宽实测 |
| QSPI Flash | 64 Mbit | 标称约 8 MiB；用于 bitstream/静态数据的能力待设计 |
| 基准晶振 | 27 MHz | 首次 bring-up 以 27 MHz 为保守基线 |
| 额外时钟 | MS5351，文档标 3 路 | 配置、抖动和可用频率必须实测 |
| 供电 | USB-C，5 V ±10%，datasheet 0.5 A | 不是本项目功耗测量 |

一级来源为 [Sipeed 板卡页](https://en.wiki.sipeed.com/hardware/en/tang/tang-nano-20k/nano-20k.html)、
[v1.3 datasheet](https://dl.sipeed.com/fileList/TANG/Nano_20K/1_Datasheet/Sipeed%20Tang%20nano%2020K%20Datasheet%20V1.3-en_US.pdf)
和 [官方上手页](https://wiki.sipeed.com/hardware/en/tang/tang-nano-20k/example/unbox.html)。
官网聚合页对 Flash/PLL 的个别表项存在版本不一致；本 contract 以板卡专页和 v1.3 datasheet
的 64 Mbit Flash、2 PLL 为准，T6.1.1 再按实际 PCB revision 和工具 device database 核对。

## 3. 接口冻结

| ID | 接口 | 本项目允许角色 | 当前不得假定 |
| --- | --- | --- | --- |
| IF01 | USB-C + BL616 | 供电、烧录、host bridge | “480 Mbps USB bridge”就是可持续 payload throughput |
| IF02 | JTAG | bitstream 烧录、器件识别、有限调试 | runtime histogram/action 数据面 |
| IF03 | UART | bring-up、命令、低速 telemetry、保守 fallback transport | 115200 baud 足以承载主数据面 |
| IF04 | USB-to-SPI | syndrome replay 和 parameter bank 的首选候选数据面 | 未测先写持续速率、流控和 deadline closure |
| IF05 | free GPIO | trigger、cycle strobe、logic analyzer、有限数字 I/O | ADC、DAC、微波或量子读出接口 |

板载 BL616 文档支持 JTAG、USB-to-UART、USB-to-SPI、USB-to-I2C，并控制 MS5351。其
480 Mbps 是 USB 链路/桥接能力上限说明，不是本项目 application payload 实测。JTAG 不进入
实时数据面；UART 是必须能工作的保守 bring-up 路径；若要满足窗口级 replay，优先验证
USB-to-SPI，并实现 sequence/schema/length/CRC/flow-control/timeout。

## 4. 容量与传输算术，不是综合结果

`DLIF-v1` 的 `32×32 uint16 raw-count` 单缓冲为 2,048 B，双缓冲为 4,096 B，即 32,768 bit。
若按 828 Ki-bit 换算，仅占厂商 B-SRAM 标称容量约 3.86%。双 affine bank 的 `6×25 bit`
payload 只有 300 bit。以上只能说明**静态容量量级看似可容纳**，不能证明 BRAM 映射、端口数、
路由、Fmax、FSM、CRC、FIFO 和 student 全部装得下；这些必须由 T5.5.2/T5.5.3 综合验证。

官方上手页的 115200-baud 终端只是示例。按 UART 8N1、忽略全部 header/流控，发送一份
2,048 B histogram 的理论下界已经是：

\[
t_{UART} \ge \frac{2048\times 10}{115200}=177.78\ \mathrm{ms}.
\]

而当前 reference window emission 是 20 ms。仅 histogram 就至少需要 1.024 Mbit/s line rate，
尚未计 header、CRC、ack 和重传。因此 115200 UART 不可作为当前 20 ms 全直方图主数据面；
不能靠“USB 480 Mbps”字样跳过 USB-to-SPI 实测，也不能把 host transport 放进 5 us fast loop。

## 5. 当前仓库兼容性

当前 `cnn_fpga/config/hardware_hil.yaml` 仍是：

- `hil.backend=mock`；
- `hil.board=ZCU111`；
- `/dev/uio0` + `/dev/uio1`；
- AXI-Lite/mmap + `32×32 float32` 4,096 B DMA payload。

`cnn_fpga/hwio/board_backend.py` 也只实现 Linux memory-mapped UIO/AXI/DMA scaffolding。T72
当前 gate 为 `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`，repo path 为
`placeholder_only`。Tang Nano 20K 没有该 ZCU111/SoC device-path 假设，故两者**不直接兼容**。

后续不得只把 YAML 的 `board` 字符串改成 Tang Nano 20K 就声称移植完成。至少需要：

1. 新的 BL616 UART/USB-SPI framed transport adapter；
2. `DLIF-v1` raw-count adapter、版本/长度/CRC/sequence；
3. 流控、timeout、retransmit/fallback 与 negative-path tests；
4. Python golden、RTL simulation 和真实板输出的同 trace 对齐；
5. core、transport、end-to-end 三种 latency 分别计时。

## 6. 测量边界

满足相应 gate 后允许报告：目标器件综合/实现资源与时序估计、on-chip core cycles、UART/SPI
transport、数字 replay end-to-end latency/jitter、bit-for-bit agreement、CRC/version/timeout/
overflow/fallback，以及有明确仪表与方法的板级功耗。

没有外部真实微波、DAC/高速量子 ADC、IQ 链路、cavity/transmon 和经授权量子数据时，严格
禁止报告：

- 微波 pulse generation/fidelity；
- 高速 ADC sampling、IQ demodulation 或 readout SNR；
- cavity/transmon control；
- physical GKP squeezing；
- logical lifetime、beyond-break-even 或 closed-loop quantum QEC。

本板在第一篇论文中的唯一身份是 **low-cost FPGA digital control-plane reference target**，
不是 integrated quantum controller。

## 7. 升级门与非 demo 审计

| 检查 | 结果 |
| --- | --- |
| 是否虚构实物、价格或采购 | 否；physical unit 未验证，current quote 为空，采购前刷新 |
| 是否只抄一张规格表 | 否；规格、项目接口、容量算术、现有后端不兼容、测量禁止项分别冻结 |
| 是否把 USB 标称速率当吞吐 | 否；明确为未测 bridge capability，并给出 UART 反例 |
| 是否把容量算术当综合 | 否；要求保存 target-device synthesis/timing report |
| 是否把 ZCU111 placeholder 当板测 | 否；T72 NO_GO 和 adapter 缺口写入 machine contract |
| 是否把数字 replay 当量子实验 | 否；六类量子/模拟前端结论 fail closed |
| 后续如何升级 | T5.5.2/3 综合，T6.1 实物/协议/计时，T6.2/4 correctness/latency/failure |

完成本 task 只把部署目标和可测边界变成可审计 contract；它不晋升 `claim_ladder` 的 CL2、
CL3 或 CL4。
