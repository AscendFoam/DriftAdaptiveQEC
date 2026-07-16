# runtime/ — 双环路运行时框架

本目录实现了 CNN-FPGA 量子纠错系统的双环路运行时框架，负责快环仿真、慢环推理、参数银行管理、时序调度和延迟注入。这是连接物理仿真层与硬件 I/O 层的核心运行时。

## 目录结构

| 文件 | 职责 |
|------|------|
| [param_bank.py](param_bank.py) | 双缓冲参数银行，支持 stage-then-commit 无毛刺更新 |
| [atomic_parameter_bank.py](atomic_parameter_bank.py) | 完整 MAP-LUT image 的 version/CRC/SHA/timestamp/CAS 双 bank 事务、hysteresis、atomic commit 与 readback |
| [three_timescale_cadence.py](three_timescale_cadence.py) | fast/event/window/slow/commit/recalibration exact cadence 与 adaptation-lag 定义 |
| [closed_loop_fault_recovery.py](closed_loop_fault_recovery.py) | atomic bank、ack/readback、post-commit guard、host timeout、monotonic LKG republish 与 bit-accurate fallback 闭环 |
| [latency_injector.py](latency_injector.py) | 随机延迟采样（DMA / 推理 / 写回等阶段） |
| [scheduler.py](scheduler.py) | 双环路周期调度器，协调窗口发射与慢环任务 |
| [fast_loop_emulator.py](fast_loop_emulator.py) | 快环仿真器：采样噪声→综合征→解码→校正→直方图聚合 |
| [slow_loop_runtime.py](slow_loop_runtime.py) | 慢环运行时：直方图→噪声参数预测→解码器参数更新 |
| [feature_builder.py](feature_builder.py) | CNN 输入特征构建（直方图 + teacher 状态 → 张量） |
| [inference_service.py](inference_service.py) | 推理服务抽象层（进程内 / 子进程, NPZ / TFLite） |
| [inference_worker.py](inference_worker.py) | 子进程推理 Worker，通过 JSON lines 通信 |
| [noise_bridge.py](noise_bridge.py) | 物理噪声到运行时噪声参数的桥接层 |
| [\_\_init\_\_.py](__init__.py) | 惰性导出约 30 个公共符号 |

### 模块依赖关系

```
param_bank.py          (legacy 兼容叶节点)
atomic_parameter_bank.py → parametric_map_lut
three_timescale_cadence.py (叶节点)
closed_loop_fault_recovery.py → atomic_parameter_bank, fast_path_fixed_point
latency_injector.py    (叶节点)
noise_bridge.py        (叶节点)
    ↓
feature_builder.py     → param_bank
scheduler.py           → param_bank, latency_injector
fast_loop_emulator.py  → param_bank
inference_service.py   → decoder.param_mapper (外部)
inference_worker.py    → inference_service
    ↓
slow_loop_runtime.py   → feature_builder, inference_service, param_bank, scheduler
```

## 双环路架构

本运行时框架实现了 FPGA 量子纠错的双环路架构：

```
┌─────────────── 快环 (每 5μs) ───────────────┐
│  采样噪声 → 注入误差 → 测量综合征              │
│  → 线性解码 → 施加校正 → 累积直方图            │
└──────────── 每隔 ~2048 周期发射窗口 ──────────┘
        ↓ 直方图窗口
┌─────────────── 慢环 (每 ~20ms) ──────────────┐
│  接收直方图 → 构建特征 → CNN 推理 / 基线估计    │
│  → 预测噪声参数 → 映射为解码器参数 (K, b)       │
│  → Stage 到参数银行 → Commit 到快环            │
└──────────────────────────────────────────────┘
```

## 核心类与接口

### 参数银行 — `param_bank.py`

双缓冲 (Bank A/B) 存储，保证参数更新无毛刺：

- **`DecoderRuntimeParams`** — 解码器运行时参数容器（2×2 矩阵 `K` + 2 元素偏置 `b`）
- **`ParamBank`** — 双缓冲管理：
  - `stage_update(params, commit_epoch)` — 暂存参数到 staging bank
  - `commit_if_ready(epoch_id)` — epoch 到达后交换 active/staging bank
  - `read_active()` — 读取当前生效的参数

### 完整 image 原子参数库 — `atomic_parameter_bank.py`

T4.3.2 production candidate 以完整 parametric MAP-LUT image 为事务单位：

- `observe_selection(...)`：连续两窗同 selection key 的 hysteresis；
- `begin_stage(...)` / `write_chunk(...)` / `finalize_stage(...)`：transfer buffer 与 A/B valid slots 隔离，完整 CRC/SHA/canonical image 验证后才发布 inactive image；
- `commit_if_ready(epoch, safe_boundary=...)`：在 apply epoch、安全 cycle boundary、CAS、freshness 和 minimum residency 通过后交换 active pointer；
- `readback()` / `verify_commit_ack_readback(...)`：以 bank/version/epoch/image CRC/SHA 关闭 host 确认链。

它尚未替换 legacy scheduler 的 `ParamBank`，也不包含自动回滚、真实 transport、CDC/RTL 或板测。

### 闭环故障恢复 — `closed_loop_fault_recovery.py`

T4.3.3 supervisor 将完整 image transaction 与逐周期 fast fallback 组合：commit ack 丢失时保持 host uncertain 并阻止新 writer；post-commit guard 失败时把 prior LKG contents 作为新单调版本重发；host timeout/stale、bad integrity、OOD/deadline 和 leakage 均落到有 reason trace 的 frame hold/reset。它是 software control-safety primitive，不是自动物理 rollback、wire/CDC/RTL 或板测。

### 延迟注入 — `latency_injector.py`

为各流水线阶段提供随机延迟采样：

- **`StageLatencySpec`** — 单阶段延迟规格（mean, std, 分布类型）
- **`LatencyInjector`** — 多阶段延迟采样器，支持负载感知缩放
  - `sample_fast_cycle()` — 采样快环单周期延迟
  - `sample_slow_update(context)` — 采样慢环全流程延迟（DMA + 预处理 + 推理 + 写回 + commit_ack）

### 调度器 — `scheduler.py`

基于周期的双环路调度，发出结构化事件：

- **`SchedulerConfig`** — 调度参数（t_fast_us, window_size, commit_delay_cycles 等）
- **`DualLoopScheduler`** — 核心调度器：
  - `tick_with_fast_path(window_payload, fast_path_fn)` — 推进一个快环周期
  - `run(n_cycles, ...)` — 运行 N 个周期，收集所有事件
  - 事件种类：`window_ready`, `slow_update_finished`, `commit_applied`, `fast_budget_violation` 等

### 快环仿真器 — `fast_loop_emulator.py`

模拟 FPGA 快环的单周期行为：

- **`FastLoopEmulator`** — 驱动闭环快环演化
  - `step(epoch_id, time_us, emit_window)` — 执行一个快环周期
  - 内部流程：采样噪声 → 注入误差 → 测量综合征 → 线性解码 → 应用校正 → 聚合直方图
  - 跟踪：溢出率、饱和率、攻击性参数率、逻辑错误率 (LER)

### 慢环运行时 — `slow_loop_runtime.py`

可插拔的慢环实现，支持多种预测模式：

- **`SlowLoopRuntime`** — 主入口：
  - `__call__(window, active_params) -> DecoderRuntimeParams`
  - 内部根据 mode 分发到对应预测方法

| 模式 | 说明 |
|------|------|
| `fixed_baseline` | 固定参数，不做更新 |
| `oracle_delayed` | 延迟的上帝视角（真实参数 + 延迟窗口） |
| `static_linear` | 静态线性解码器 |
| `window_variance` | 窗口方差估计（单窗口矩估计） |
| `ekf` | 扩展卡尔曼滤波 |
| `ukf` | 无迹卡尔曼滤波（8 维状态，含速度分量） |
| `particle_filter` | 粒子滤波 |
| `model_artifact` | CNN 模型推理（NPZ / TFLite） |
| `hybrid_residual_mu` | Teacher + CNN 残差（位移修正） |
| `hybrid_residual_b` | Teacher + CNN 残差（偏置修正） |
| `rls_residual_b` | Teacher + RLS 在线学习残差 |
| `particle_filter_residual_b` | Teacher + 粒子滤波残差 |

### 特征构建 — `feature_builder.py`

将直方图窗口和 teacher 历史构建为 CNN 输入张量：

- **`RuntimeFeatureConfig`** — 控制：上下文窗口数、直方图增量、teacher 预测/参数/增量特征
- **`RuntimeFeatureSample`** — 输出：`spatial_tensor` (C×H×W) + 可选 `scalar_features`
- 两种布局模式：`broadcast`（teacher 标量铺入空间平面）和 `scalar_branch`（标量保留为独立向量）

### 推理服务 — `inference_service.py`

模型推理的统一抽象层：

- **`InProcInferenceService`** — 进程内推理
- **`SubprocessInferenceService`** — 子进程推理（通过 JSON lines 通信）
- 两个后端：`ArtifactHistogramPredictor`（NPZ 格式）和 `TFLiteHistogramPredictor`（TFLite 格式）
- **`build_inference_service(config)`** — 工厂函数，自动选择正确的服务和预测器

### 噪声桥接 — `noise_bridge.py`

将物理噪声通道参数映射为运行时噪声参数：

- **`PhysicalNoiseBridge`** — 有状态桥接层，支持多种演化模式
  - 输入：gamma (T1), n_bar (温度), sigma_displacement, sigma_phase, mu_q/p_bias
  - 输出：sigma, mu_q, mu_p, theta_deg
  - 演化模式：static / linear / step / sin / random_walk

## 使用示例

### 构建并运行双环路仿真

```python
from cnn_fpga.runtime import (
    DualLoopScheduler, FastLoopEmulator, SlowLoopRuntime,
    ParamBank, LatencyInjector,
)
from cnn_fpga.runtime.param_bank import DecoderRuntimeParams

# 初始化参数银行
initial = DecoderRuntimeParams.identity()
bank = ParamBank(initial_params=initial, initial_epoch=0)

# 从配置构建组件
latency = LatencyInjector.from_config(config, seed=42)
scheduler = DualLoopScheduler.from_config(config, param_bank=bank, latency_injector=latency, slow_path_fn=slow_loop)
fast_loop = FastLoopEmulator(config=config, param_bank=bank, noise_provider=noise_fn)

# 运行
events = scheduler.run(n_cycles=10000, window_payload_factory=None, fast_path_fn=fast_loop)
```

### 慢环推理

```python
slow_loop = SlowLoopRuntime.from_config(config, seed=42)

# 处理一个直方图窗口
updated_params = slow_loop(window_frame, active_params)

slow_loop.close()
```

### 构建推理服务

```python
from cnn_fpga.runtime.inference_service import build_inference_service

service, model_path, inf_config = build_inference_service(config)
prediction = service.predict(histogram)
service.close()
```
