# Real-Board HIL Readiness Checklist

## 1. Purpose

本文件是 `T20` 的只读 readiness checklist。

它只回答两个问题：

1. 当前真板 HIL 路径距离“可安全启动独立真板任务”还缺什么。
2. 后续真板任务在未补齐哪些证据前，不能写成“real-board HIL 已完成”。

本文件不代表真板 backend 已实现，也不代表当前机器具备硬件可运行条件。

## 2. Current Placeholder Evidence

当前真板路径仍应被标记为 `placeholder_real_board_backend`，证据如下：

### 2.1 `board_backend.py`

- 文件头直接声明：`Placeholder real-board backend using memory-mapped AXI/DMA interfaces.`
- `BoardFPGA` 类注释仍写明它只是占位骨架。
- `BoardFPGAConfig` 默认 `allow_missing_device = True`，说明当前更偏向“允许设备缺失时保留占位入口”，不是已验收部署配置。
- `from_config(...)` 在 MMIO 或 DMA 路径缺失时会抛出 `board_device_missing:...`。
- `schedule_commit(...)` 返回：
  - `target_bank = None`
  - `version = None`
  - `ack_delay_us = None`
- `step(...)` 当前只刷新状态并 `return []`，不会产生可验收的真板事件流。

### 2.2 `fpga_driver.py`

- 文件头写明：`Unified FPGA driver facade for mock and future real HIL backends.`
- `backend_name in {"board", "real"}` 时虽然会走 `BoardFPGA.from_config(...)`，但驱动层整体口径仍是 future-facing。
- `NotImplementedError` 明确写有：`reserved for future real-board integration`。

### 2.3 `run_hil_suite.py`

- `run_hil_session(...)` 只是统一 HIL orchestration 入口，不自动提供真板验收资格。
- 当前已验证路径仍是：
  - `hil.backend=mock`
  - `model_artifact`
  - `artifact_npz`
  - `inproc`
- 因此已有 `hil_events.json` / `hil_summary.json` 只能证明 software HIL 路径存在，不能外推为真板证据。

## 3. Required Preconditions

后续若要单开真板执行任务，至少应先补齐以下前置条件。

### 3.1 Device And Topology

- 明确板卡型号、板上 bitstream 版本、主机 OS。
- 明确 MMIO 设备路径，例如 `axi_uio_path` 对应哪个实际节点。
- 明确 DMA 设备路径，例如 `dma_buffer_path` 对应哪个实际节点。
- 明确寄存器地址表与当前 bitstream 是否一致。
- 明确 histogram buffer 尺寸、dtype、buffer count 与 host 配置是否一致。

### 3.2 Permissions And Host Access

- 当前执行用户是否具备 `/dev/uio*` 或等效设备节点访问权限。
- 是否需要 `sudo`、udev 规则、驱动加载、IOMMU / DMA 映射准备。
- 是否允许在目标主机上执行 `os.open(..., O_RDWR | O_SYNC)` 与 `mmap(...)`。

### 3.3 Driver / Register Evidence

- `AxiRegisterMap` 与 RTL 寄存器布局的一致性证据。
- `ctrl/status/epoch/hist_meta/commit_epoch/hist_seq/overflow_count` 各寄存器的板级定义来源。
- `active_bank`、`commit_ack`、`hist_ready` 的真板语义说明。
- staged params 写入后，板侧实际生效时序的证据。

### 3.4 Runtime And Logging

- 能输出板级启动日志。
- 能输出 MMIO 访问失败日志。
- 能输出 DMA 读缓冲日志。
- 能输出 commit request / ack 对照日志。
- 能保留最小 run dir 证据，而不是只给口头结论。

## 4. Minimum Smoke Acceptance Standard

后续真正执行真板 smoke 时，建议最小验收标准分四层，而不是直接跳到 benchmark。

### 4.1 Layer A: Device Presence

至少需要证明：

1. MMIO 路径存在。
2. DMA 路径存在。
3. 进程具备打开与映射权限。
4. 不依赖 `allow_missing_device=True` 才能继续。

### 4.2 Layer B: Register Liveness

至少需要证明：

1. `start()` 可写控制寄存器。
2. `read_status_fields()` 可返回稳定结构，而不是异常退出。
3. `epoch_id` 会变化，或能证明时钟/状态在推进。
4. `active_bank`、`commit_epoch`、`hist_seq` 至少有一次可读且值域合理。

### 4.3 Layer C: DMA Histogram Readout

至少需要证明：

1. `histogram_available()` 在合理条件下能变为真。
2. `pop_histogram_buffer()` 能返回与配置一致 shape 的 histogram。
3. `buffer_id` 与 `hist_sequence` 有可追踪变化。
4. DMA 读出的 payload 不是空数据、全未初始化噪声或明显错形。

### 4.4 Layer D: Commit / Ack Round Trip

至少需要证明：

1. `stage_params(...)` 后寄存器写入无异常。
2. `commit_bank(...)` 后板侧能给出可解释的 commit 行为。
3. `wait_commit_ack(...)` 至少一次成功返回，不是全靠 host shadow state。
4. active/staged bank 的板级切换证据可被日志或寄存器读回支撑。

只有在 A/B/C/D 四层都具备证据后，才适合继续讨论：

- real-board HIL smoke
- real-board vs mock 对照
- real-board benchmark

## 5. Required Evidence Pack For A Future Real-Board Task

后续真板任务应至少产出以下证据包：

1. 解释器与命令。
2. 主机名、OS、板卡标识。
3. bitstream / RTL 版本标识。
4. MMIO / DMA 路径与权限说明。
5. 寄存器地址表来源。
6. 原始日志或摘要日志。
7. 最小 run dir。
8. 明确 backend=`board`。
9. 若 slow-loop 仍使用 `artifact_npz` 或 `.tflite`，必须同时标注 artifact type。

## 6. Prohibited Wording

在独立真板证据出现前，不应写：

- `real-board HIL complete`
- `board backend validated`
- `true hardware benchmark complete`
- `P3 real-board HIL 已完成`
- `真板路径已恢复`

在当前阶段可以写：

- `real-board backend placeholder exists`
- `board path has config scaffolding but is not yet hardware-validated`
- `T20 only provides readiness checklist and acceptance criteria`

## 7. Recommended Next-Step Boundary

若 Captain 后续决定继续推进真板路径，下一张任务包应是“真板 smoke execution plan”而不是“真板已完成总结”。

该任务至少应继续限制：

1. 不改 benchmark 口径。
2. 不把 mock 结果当作板级结果。
3. 不跳过设备/权限/日志证据。
4. 不在无最小 smoke 证据时更新阶段结论。
