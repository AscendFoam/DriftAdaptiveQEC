# hwio/ — 硬件 I/O 抽象层

本目录实现了 CNN-FPGA 系统的硬件 I/O 抽象层，提供统一的驱动 API (`FPGADriver`)，支持软件模拟后端 (`MockFPGA`) 和真实 FPGA 板卡后端 (`BoardFPGA`)。硬件通信基于 AXI-Lite 内存映射寄存器和 DMA 缓冲区。

## 目录结构

| 文件 | 职责 |
|------|------|
| [axi_map.py](axi_map.py) | AXI-Lite 寄存器映射、定点量化、位级编解码 |
| [dma_client.py](dma_client.py) | DMA 读取数据结构 + 抽象客户端接口 |
| [mock_fpga.py](mock_fpga.py) | 软件 FPGA 仿真（AXI 寄存器/DMA/参数银行/延迟注入） |
| [board_backend.py](board_backend.py) | 真实 FPGA 板卡后端（Linux mmap over /dev/uioX） |
| [fpga_driver.py](fpga_driver.py) | 统一驱动门面，封装 mock/real 后端 |
| [\_\_init\_\_.py](__init__.py) | 惰性导出 19 个公共符号 |

### 模块依赖关系

```
axi_map.py       ← 定义硬件协议（寄存器地址、位域、定点格式）
dma_client.py    ← 定义 DMA 数据结构和抽象接口
    ↓
mock_fpga.py     → axi_map, dma_client + runtime 组件
board_backend.py → axi_map, dma_client + mmap 硬件访问
    ↓
fpga_driver.py   → 统一门面，根据配置选择后端
```

## 核心类与接口

### AXI 寄存器映射 — `axi_map.py`

定义了 FPGA 的硬件协议层，被 MockFPGA 和 BoardFPGA 共同使用。

**寄存器地址映射：**

| 地址 | 寄存器 | 用途 |
|------|--------|------|
| 0x00 | ctrl | 控制字（start / reset_hist / commit_bank） |
| 0x04 | status | 状态字（ready / hist_ready / commit_ack / overflow_alert） |
| 0x08 | hist_meta | 直方图元数据（buffer_id + overflow） |
| 0x0C | overflow_count | 溢出计数 |
| 0x10–0x1C | K11–K22 | 解码器增益矩阵 (2×2) |
| 0x20–0x24 | b1–b2 | 解码器偏置向量 (2) |
| 0x30 | active_bank | 当前活跃参数银行 (A/B) |
| 0x34 | epoch_id | 当前 epoch |
| 0x38 | commit_epoch | 提交目标 epoch |
| 0x3C | hist_seq | 直方图序列号 |

**定点量化：**

- **`FixedPointFormat`** — 描述 Q 格式（如 Q4.20: 4 位整数 + 20 位小数）
  - `quantize(value)` → `(量化值, 饱和标志)`
  - `from_spec("Q4.20")` — 从字符串解析

**编解码方法：**

- `pack_params(DecoderRuntimeParams)` → `{addr: u32_value}` — 将浮点参数打包为寄存器值
- `unpack_params(registers)` → `DecoderRuntimeParams` — 从寄存器值恢复参数
- `build_ctrl_word(...)` / `decode_ctrl_word(word)` — 控制/状态字位域操作

### DMA 客户端 — `dma_client.py`

- **`DMAReadout`** — 一次 DMA 读取的结果（buffer_id, 字节数, 窗口帧, 元数据）
- **`DMAClient`** (ABC) — 抽象接口：`histogram_available()`, `read_histogram()`, `reset()`
- **`BackendDMAClient`** — 适配器，将任何后端对象包装为 `DMAClient`
- **`MemoryMappedDMAClient`** — 基于 mmap 的 DMA 客户端实现

### Mock FPGA — `mock_fpga.py`

完整的 FPGA 软件仿真，是 HIL 验证的主要后端：

- **`MockFPGA`** — 仿真 AXI 寄存器、DMA 缓冲区管理、参数银行提交、延迟注入和快环直方图生成
  - `step(cycles)` — 推进 `cycles` 个快环周期：
    1. 递增 epoch 和时间
    2. 采样快环延迟，检查预算违规
    3. 执行参数银行 commit（如果 epoch 到达目标）
    4. 在窗口边界调用 `FastLoopEmulator` 生成直方图
    5. 推入 DMA 缓冲区，发出结构化事件
  - `schedule_commit(commit_epoch, ack_delay_us)` — 暂存参数并设置提交时序
  - `pop_histogram_buffer()` — 从 DMA 队列弹出一个直方图窗口
- **`MockFPGAEvent`** — 结构化事件日志（kind, epoch_id, time_us, details）

### 真实板卡后端 — `board_backend.py`

通过 Linux `mmap` 访问 `/dev/uioX` 设备文件的 FPGA 后端：

- **`MemoryMappedRegisterIO`** — AXI-Lite 寄存器读写（32 位小端）
- **`MemoryMappedDMARegion`** — DMA 缓冲区读取
- **`BoardFPGA`** — 与 MockFPGA 相同的 API 接口，但操作真实硬件
  - `from_config(config)` — 验证设备路径存在，构建 mmap 连接
  - 当 `allow_missing_device=True` 时，设备不存在时给出明确错误而非崩溃

### FPGA 驱动 — `fpga_driver.py`

统一门面，封装 mock/real 后端：

- **`FPGADriver`** — 核心驱动类：
  - `from_config(config, noise_provider, seed)` — 工厂方法，根据 `hil.backend` 选择后端
    - `"mock"` → `MockFPGA`
    - `"board"` / `"real"` → `BoardFPGA`
  - 便捷方法：`start()`, `advance_cycles()`, `read_status()`, `histogram_available()`, `read_histogram()`, `stage_params()`, `commit_bank()`, `wait_commit_ack()`
- **`FPGADriverConfig`** — 轮询间隔和超时配置

## 使用示例

### 使用 Mock 后端

```python
from cnn_fpga.hwio import FPGADriver

driver = FPGADriver.from_config(config, noise_provider=my_noise_fn, seed=42)
driver.start()

# 推进快环并读取直方图
driver.advance_cycles(2048)
if driver.histogram_available():
    readout = driver.read_histogram()
    histogram = readout.window.payload["histogram"]

# 更新参数
driver.stage_params(new_params)
driver.commit_bank(commit_epoch=1000)
driver.wait_commit_ack()

driver.close()
```

### 使用真实板卡

```yaml
# config.yaml
hil:
  backend: board
  board: ZCU111
  board_io:
    axi_path: /dev/uio0
    dma_path: /dev/uio1
```

```python
driver = FPGADriver.from_config(config)
# 其余 API 与 mock 完全相同
```

### AXI 寄存器操作

```python
from cnn_fpga.hwio import AXI_REGISTER_MAP
from cnn_fpga.runtime.param_bank import DecoderRuntimeParams

# 打包参数到寄存器值
registers = AXI_REGISTER_MAP.pack_params(params)

# 构建控制字
ctrl = AXI_REGISTER_MAP.build_ctrl_word(start=True, commit_bank=True)
```

## 关键设计决策

1. **Mock/Real 统一接口**：MockFPGA 和 BoardFPGA 实现相同的行为契约，FPGADriver 透明切换
2. **参数银行双缓冲**：Stage-then-commit 协议保证参数更新无毛刺
3. **结构化事件日志**：MockFPGA 发出详细事件（窗口就绪、提交确认、预算违规等），便于 HIL 诊断
4. **优雅降级**：板卡后端支持 `allow_missing_device`，设备不可用时给出明确错误而非崩溃
