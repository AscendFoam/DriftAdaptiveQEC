# T49 真实板级 Smoke Execution Gate

## 结论

最终 gate verdict：

`NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`

本轮只完成了当前宿主的只读 host fact probe、只读 device-path probe、AXI/DMA/placeholder 代码审计，以及基于这些事实的 gate 聚合。没有执行任何 MMIO 写、DMA 写、寄存器写，也没有执行 real-board smoke。

## 关键事实

- host probe 解释器：`C:\ProgramData\anaconda3\python.exe`
- Python 版本：`3.12.7`
- 当前宿主：Windows 11，`cmd /c ver` = `Microsoft Windows [Version 10.0.26200.8457]`
- 主机/主板信息：`MECHREVO / JiguangX Series GM6AQ7C / GM6AQ7C`
- `pnputil /enum-devices /connected` 中没有匹配到 `Xilinx|AMD|FPGA|XDMA|UIO` 的已连接设备
- 仓库配置里仍保留 board 侧默认记录：`board=ZCU111`，`bitstream_version=fpga_linear_v1`
- 当前仓库 `fpga/` 目录下未发现 `.bit/.bin/.xsa/.hwh/.bitstream` 文件

## 分层表

| category | summary |
|---|---|
| `host_environment` | host probe 完成；OS/解释器事实已记录；当前宿主本身可被识别 |
| `device_path_truth` | `/dev/uio0`、`/dev/uio1`、`\\\\.\\XilinxDMA`、`\\\\.\\XDMA`、`\\\\.\\uio0`、`\\\\.\\uio1` 全部 `not_found`，`openable_count = 0` |
| `bitstream_and_contract_truth` | 只有 `fpga_linear_v1` 这一条 config-level 版本记录；缺 bitstream 文件、RTL 地址表绑定、DMA contract 和 Q4.20 板侧确认 |
| `repo_execution_path_truth` | `FPGADriver` 能路由到 `board/real`，但 `board_backend.py` 仍是 placeholder-only |
| `supported_claims` | 当前只支持“仓库存在 board-path scaffolding，且代码侧 AXI/DMA 事实可审计；但当前机器没有可读打开的真板设备路径” |
| `unsupported_claims` | 不支持“real-board smoke 已执行成功”“P3 真板 HIL 已完成”“deployment closure” |

## 当前宿主是否具备进入真板 smoke 的基本前提

不具备。

直接阻塞点不是代码入口缺失，而是当前机器上没有找到任何可读打开的真板候选设备路径：

- Linux 风格默认路径 `/dev/uio0`、`/dev/uio1` 不存在
- Windows exploratory 候选 `\\\\.\\XilinxDMA`、`\\\\.\\XDMA`、`\\\\.\\uio0`、`\\\\.\\uio1` 也都不存在
- `pnputil` 没有枚举到明显的 Xilinx/AMD/FPGA/XDMA/UIO 已连接设备

因此，本轮最先触发的是 host/device 层的 `NO_GO`，而不是后面的 contract 层或 repo-path 层。

## 当前到底缺哪一层

按 gate 分层看，当前从强到弱依次缺：

1. `device_path_truth`
   - 当前是最直接的阻塞层
   - 没有任何可读打开的 MMIO/DMA 候选路径
2. `bitstream_and_contract_truth`
   - 仓库里只有 `fpga_linear_v1` 这一条配置记录
   - 没有当前宿主可引用的 bitstream 文件
   - 没有 RTL 地址表与 DMA payload shape/dtype 的 host-bound 证据
3. `repo_execution_path_truth`
   - 即使前两层补齐，当前 `board_backend.py` 仍属于 placeholder-only 语义

## AXI_REGISTER_MAP 与 DMA 的代码侧已知事实

本轮 `code_side_audit.json` 已固定以下代码侧事实：

- AXI 地址表是 concrete 的：
  - `ctrl=0x00`
  - `status=0x04`
  - `hist_meta=0x08`
  - `overflow_count=0x0C`
  - `K/b` 参数寄存器覆盖 `0x10` 到 `0x24`
  - `active_bank=0x30`
  - `epoch_id=0x34`
  - `commit_epoch=0x38`
  - `hist_seq=0x3C`
- 控制/状态 bit mask 也是 concrete 的：
  - `ctrl_start=0x1`
  - `ctrl_reset_hist=0x2`
  - `ctrl_commit_bank=0x4`
  - `status_ready=0x1`
  - `status_hist_ready=0x2`
  - `status_commit_ack=0x4`
  - `status_overflow_alert=0x8`
- DMA 代码约定是 concrete 的：
  - `buffer_bytes = 4096`
  - `buffer_count = 2`
  - `histogram_shape = 32 x 32`
  - `dtype = float32`
  - `DMAReadout` 关注 `buffer_id / byte_count / window.payload["histogram"] / metadata`

但这些都还是“代码已知事实”，不是“当前宿主 + 当前 bitstream 已确认事实”。

## 当前最强可支持的 real-board 表述

当前最强可支持的表述只有：

“当前仓库已经有可审计的 board-path scaffolding、AXI 寄存器映射和 DMA 数据结构；当前宿主也已完成只读事实探测。但这台 Windows 主机上没有发现可读打开的真实板级设备路径，因此还不具备进入 bounded real-board smoke 的最小前提。”

## 当前仍然不能支持的表述

当前仍然不能支持：

- `real-board smoke executed successfully`
- `P3 real-board HIL complete`
- `board backend validated`
- `hardware_validated`
- `deployment closure`
- 任何 real-board benchmark / HIL promotion / 性能对照结论

## 证据文件

- `artifacts/t49_real_board_smoke_execution_gate/host_fact_manifest.json`
- `artifacts/t49_real_board_smoke_execution_gate/device_path_probe.json`
- `artifacts/t49_real_board_smoke_execution_gate/code_side_audit.json`
- `artifacts/t49_real_board_smoke_execution_gate/t49_real_board_smoke_execution_gate.json`
