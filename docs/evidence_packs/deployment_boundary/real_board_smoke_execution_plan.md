# Real-Board Smoke Execution Plan

## 1. Purpose

本文档对应 `T22`，只定义后续真板 smoke 的进入条件、执行顺序、审计清单与量化验收阈值草案。

它不是硬件执行记录，不代表 `real-board HIL complete`，也不代表当前仓库已经具备 `hardware_validated` 证据。

## 2. Scope Boundary

本计划只覆盖最小真板 smoke 的准备与验收边界：

1. 先确认目标主机/板卡拓扑。
2. 先审计 AXI-Lite 地址表与 DMA 缓冲区约定。
3. 先跑 Layer A-D 的最小 smoke，再决定是否有资格进入更高层验证。

本计划不覆盖：

- `board_backend.py` / `fpga_driver.py` 的实现改写
- `backend=board` 的 HIL 或 benchmark 执行
- real-board 与 mock 的性能对比结论
- 正式 benchmark 口径变更

## 3. Platform Decision Points

在任何真板命令执行前，必须先锁定以下四种宿主模型中的一种，并把未选中的路径显式排除：

### 3.1 Linux Local Host

适用条件：

- 板卡设备节点直接出现在本机 Linux
- 可以直接访问 `/dev/uio*`、DMA 设备节点或等价驱动入口
- 允许 `os.open(..., O_RDWR | O_SYNC)` 与 `mmap(...)`

进入前必须确认：

- OS 发行版与内核版本
- UIO / DMA 驱动名称
- 设备节点真实路径
- 当前用户权限模型

### 3.2 Windows Local Host

适用条件：

- 板卡驱动直接安装在 Windows
- MMIO / DMA 访问不是通过 Linux `/dev/uio*` 语义实现

进入前必须确认：

- `board_backend.py` 现有假设是否与 Windows 驱动模型冲突
- 是否存在等价于 `axi_uio_path` / `dma_buffer_path` 的 Windows 设备入口
- 是否需要单独的桥接层或驱动包装

如果上述问题没有明确答案，不得把 Windows 本机视为可直接执行路径。

### 3.3 WSL

适用条件：

- 板卡真实设备由 Windows 托管，但 WSL 负责执行 Python 侧逻辑

进入前必须确认：

- WSL 能否透传目标设备，而不是只继承文件系统
- WSL 内是否能稳定访问 MMIO / DMA 所需驱动接口
- 权限和路径是否与 `board_backend.py` 现有参数模型一致

如果设备访问仍落在 Windows 驱动层而不是 WSL 内核层，WSL 只可作为分析环境，不可直接当作真板执行环境。

### 3.4 Remote Board Host

适用条件：

- 真板接在远端 Linux 主机或嵌入式主机
- 本仓库工作机只负责准备命令与收集日志

进入前必须确认：

- 远端主机 OS / 驱动 / 权限模型
- 仓库版本、bitstream 版本和地址表来源
- 日志、run dir、原始输出如何回传

## 4. AXI/Register Audit Checklist

真板 smoke 前，必须先按 `cnn_fpga/hwio/axi_map.py` 审计以下事实，并把“代码地址表”和“RTL/bitstream 文档来源”逐项对齐：

### 4.1 Control / Status Registers

- `ctrl_addr = 0x00`
- `status_addr = 0x04`
- `hist_meta_addr = 0x08`
- `overflow_count_addr = 0x0C`
- `active_bank_addr = 0x30`
- `epoch_id_addr = 0x34`
- `commit_epoch_addr = 0x38`
- `hist_seq_addr = 0x3C`

### 4.2 Staged Parameter Registers

- `k11_addr = 0x10`
- `k12_addr = 0x14`
- `k21_addr = 0x18`
- `k22_addr = 0x1C`
- `b1_addr = 0x20`
- `b2_addr = 0x24`

### 4.3 Bit Semantics To Confirm

- `ctrl_start_mask`
- `ctrl_reset_hist_mask`
- `ctrl_commit_bank_mask`
- `status_ready_mask`
- `status_hist_ready_mask`
- `status_commit_ack_mask`
- `status_overflow_alert_mask`

### 4.4 Fixed-Point Contract To Confirm

- `fixed_point_spec = Q4.20`
- host 打包/解包是否与 RTL 一致
- 饱和、截断、符号位解释是否一致
- `A/B` bank 编码是否仍为 `A -> 0`、`B -> 1`

### 4.5 Register Audit Exit Condition

只有当以下问题都有确定答案时，才允许进入真板 smoke：

1. 上述地址在当前 bitstream 上逐项可对应。
2. 上述 bit mask 在当前 RTL/driver 文档中有来源。
3. `active_bank`、`commit_epoch`、`hist_seq` 的板级语义有文字说明。
4. `Q4.20` 的 host/RTL 编码约定没有冲突。

## 5. DMA Audit Checklist

真板 smoke 前，必须结合 `cnn_fpga/hwio/dma_client.py` 审计 DMA 缓冲区约定。

### 5.1 Config Items To Freeze

- `MemoryMappedDMAConfig.path`
- `MemoryMappedDMAConfig.buffer_bytes`
- `MemoryMappedDMAConfig.buffer_count`

### 5.2 Runtime Semantics To Freeze

- `histogram_available()` 的触发条件
- `read_histogram()` 对应的真实底层行为
- `DMAReadout.buffer_id`
- `DMAReadout.byte_count`
- `DMAReadout.window.payload["histogram"]`
- `DMAReadout.metadata`

### 5.3 Byte/Shape Audit

以下量必须在执行前写进 smoke 包或执行记录：

1. 期望 histogram dtype。
2. 期望 histogram shape。
3. 期望单个 buffer 的 `byte_count`。
4. `buffer_count` 是否与板侧 ring / ping-pong 设计一致。

如果当前设计沿用 `32 x 32` `float32` histogram，则期望 `byte_count = 4096`。如果真实 bitstream 使用别的 shape 或 dtype，必须在执行前先改写计划记录，而不是在执行时口头修正。

### 5.4 DMA Audit Exit Condition

只有当以下条件都明确后，才允许进入真板 smoke：

1. `buffer_bytes` 与期望 payload 大小一致。
2. `buffer_count` 与板级 buffer 轮换语义一致。
3. `buffer_id` 与 `hist_seq` 的对应关系可解释。
4. 读取路径不依赖 host 侧虚构 `backend` 元数据才能解释 readout。

## 6. Quantitative Acceptance Threshold Draft

以下阈值是 `T22` 给出的计划草案，不是已验证事实。后续硬件任务可以细化，但不得弱化为纯定性描述。

### 6.1 Layer A: Device Presence

- MMIO 路径存在率：`1/1`
- DMA 路径存在率：`1/1`
- 打开设备成功率：`2/2`
  - MMIO open 成功
  - DMA open/mmap 成功
- 单项超时：`<= 5s`

任一项失败即 fail-fast，不进入后续层。

### 6.2 Layer B: Register Liveness

- 连续 `3` 次状态读回无异常
- `start` 写入后 `status.ready` 在 `<= 1s` 内至少观察到 `1` 次有效状态
- `epoch_id` 或 `hist_seq` 在 `<= 5s` 观测窗口内至少发生 `1` 次单调变化
- `active_bank`、`commit_epoch`、`hist_seq` 三项都必须至少成功读回 `1` 次

如果寄存器值全程静止、只返回默认零值或无法解释，不得进入 Layer C。

### 6.3 Layer C: DMA Histogram Readout

- 至少成功读回 `3` 个连续 readout
- `byte_count` 必须 `100%` 等于预期值
- histogram shape 必须 `100%` 等于预期 shape
- `buffer_id` 必须落在 `[0, buffer_count - 1]`
- `hist_seq` 必须单调不减，且在 `3` 次读回中至少推进 `2` 次
- 若存在 overflow 标志，必须与 `hist_meta` / `overflow_count` 可交叉解释

若 payload 全零、未初始化噪声形态明显、shape 错误或 `byte_count` 漂移，立即终止。

### 6.4 Layer D: Commit / Ack Round Trip

- 至少执行 `3` 轮 stage + commit
- 每轮 `commit_ack` 超时阈值：`<= 1s`
- `3/3` 轮都必须观测到可解释的 ack
- `active_bank` 必须至少发生 `2` 次与提交目标一致的切换
- `commit_epoch` 必须与提交序列保持单调不减

如果 ack 仅存在于 host shadow state、寄存器无对应证据，判定失败。

### 6.5 Fail-Fast Budget

- 总 smoke 预算：`<= 15 min`
- 连续设备/寄存器失败次数上限：`2`
- DMA shape/byte_count 失败容忍：`0`
- commit/ack 失败容忍：`0`
- 发生未解释寄存器地址冲突、权限错误或 driver 不匹配时立即停止

## 7. Minimum Command Plan

以下只是后续任务可采用的最小命令骨架，`T22` 未执行这些命令。

### 7.1 Preflight

```powershell
# NOT EXECUTED IN T22
python -c "from cnn_fpga.hwio.axi_map import AXI_REGISTER_MAP as m; print(m)"
python -c "from cnn_fpga.hwio.dma_client import MemoryMappedDMAConfig; print(MemoryMappedDMAConfig(path='TO_FILL', buffer_bytes=TO_FILL, buffer_count=TO_FILL))"
```

### 7.2 Host / Device Evidence Capture

```powershell
# NOT EXECUTED IN T22
python -c "import os; print('host_ready_placeholder')"
```

后续真实任务中，这一步必须替换为目标宿主专用命令，用于记录：

- 主机 OS / kernel / driver
- 设备节点路径
- 访问权限

### 7.3 Smoke Execution Skeleton

```powershell
# NOT EXECUTED IN T22
python -m cnn_fpga.benchmark.run_hil_suite --config <board_smoke_config_placeholder>
```

注意：

- 这里只是说明未来入口仍可能复用现有 orchestration。
- 只有当后续任务显式补齐 `board` backend 运行前提后，才允许把占位符配置替换成真实 board smoke 配置。

## 8. Required Evidence Pack For A Future Hardware Run

后续真实硬件任务至少应产出以下证据：

1. 目标宿主模型结论
   - Linux local / Windows local / WSL / remote board host 四选一
2. 主机与板卡标识
   - OS、kernel、driver、board model、bitstream version
3. AXI/register map 对齐证据
   - 地址表来源
   - mask 语义来源
4. DMA 对齐证据
   - `buffer_bytes`
   - `buffer_count`
   - 期望 shape/dtype
5. Layer A-D 原始日志或摘要日志
6. 至少一份最小 run dir
7. 明确写明 slow-loop artifact 类型
   - `artifact_npz`
   - `tflite_service`
   - `tflite_stub_service`

## 9. Prohibited Wording

在真实硬件证据未出现前，不得写：

- `real-board HIL complete`
- `board backend validated`
- `hardware_validated`
- `true hardware benchmark complete`
- `P3 real-board HIL 已完成`

当前阶段允许写：

- `real-board smoke execution plan exists, but it has not been executed`
- `real-board readiness checklist exists, but hardware validation evidence does not`
- `board path still requires platform confirmation, AXI/DMA audit, and Layer A-D smoke evidence`

## 10. Relationship To Existing Readiness Doc

`docs/evidence_packs/deployment_boundary/real_board_hil_readiness.md` 负责说明：

- 为什么当前仍是 placeholder
- 为什么不能写成真板已完成
- 真板任务的最小前置条件与验收层次

本文档额外补上：

- 目标宿主决策树
- 直接引用 `axi_map.py` / `dma_client.py` 的审计对象
- 量化阈值草案
- 未来执行命令骨架

因此，后续任何真板任务都应同时引用：

- `docs/evidence_packs/deployment_boundary/real_board_hil_readiness.md`
- `docs/evidence_packs/deployment_boundary/real_board_smoke_execution_plan.md`

并明确说明两者都只是计划/前置条件，不是硬件执行结果。
