# config/ — 实验配置文件集

本目录包含所有实验阶段使用的 YAML 配置文件，按阶段和用途命名。配置文件之间支持 `base_config` 继承机制，子配置可通过深度合并覆盖父配置的字段。

## 文件命名规则

| 前缀 | 阶段 | 说明 |
|------|------|------|
| `experiment_static*` | P1 | 静态数据集训练配置 |
| `experiment_drift*` | P2 | 漂移仿真配置 |
| `hardware_emulation*` | P2 | 硬件行为仿真配置 |
| `hardware_hil*` | P3 | Hardware-in-the-Loop 配置 |
| `p4_multiscenario*` | P4 | 多场景基准配置 |
| `experiment_runtime_b_residual*` | P4 | 残差-b 模型实验/消融配置 |
| `p4_*` | P4 | P4 阶段专项配置（消融、粒子滤波、UKF 调优等） |
| `hybrid_b_histogram_tuning*` | P3 | Hybrid-b 模式直方图调优 |
| `task_tmp/` | 临时 | 任务相关的临时配置 |

## 配置继承链

```
experiment_static.yaml              (独立, P1)
experiment_drift.yaml               (独立, P2)
hardware_emulation.yaml             (独立, P2)
hardware_hil.yaml                   (独立, P3)
  ← p4_multiscenario.yaml           (继承 hardware_hil.yaml, P4)
      ← experiment_runtime_b_residual.yaml  (继承 p4_multiscenario.yaml, P4 repair)
```

子配置通过顶层 `base_config` 字段引用父文件路径，加载时递归合并：

```yaml
base_config: hardware_hil.yaml
experiment:
  stage: p4_benchmark
```

## 核心配置段说明

### 全局段（所有配置共有）

```yaml
experiment:
  name: static_v1          # 实验运行名称
  stage: p1_static          # 阶段标识
  seed: 42                  # 全局随机种子

paths:
  output_root: runs         # 输出根目录
  dataset_dir: ...          # 数据集目录
  model_dir: ...            # 模型目录
  report_dir: ...           # 报告目录
```

### P1 特有段

```yaml
dataset:
  n_configurations: 50
  samples_per_configuration: 200
  points_per_sample: 2048
  histogram_bins: 32
  split: {train: 0.7, val: 0.15, test: 0.15}
  parameter_ranges:
    sigma: [0.15, 0.50]
    mu_q: [-0.20, 0.20]
    mu_p: [-0.20, 0.20]
    theta_deg: [-10.0, 10.0]

training:
  model_type: tiny_cnn      # 或 linear_regression_baseline
  tiny_cnn:
    conv_channels: 8
    kernel_size: 3
    hidden_dim: 32
    batch_size: 64
    epochs: 20
    learning_rate: 0.005
    patience: 5
```

### P2 特有段

```yaml
simulation:
  n_rounds: 10000
  repeats: 5

drift:                      # 漂移场景定义
  scenarios:
    - name: slow_linear
      type: linear          # linear / step / sin / random_walk
      params:
        sigma0: 0.30
        alpha: 0.0005

latency_model:              # 随机延迟模型
  dma_mean_us: 50
  dma_std_us: 10
  preprocess_mean_us: 100
  inference_mean_us: 200
```

### P3 特有段

```yaml
runtime:                    # 双环路调度参数
  t_fast_us: 5.0
  window_size: 2048
  t_slow_update_ms: 20.0
  window_stride: 4000
  max_pending_windows: 8

hil:
  backend: mock             # mock 或 board
  board: ZCU111
  board_io:
    axi_path: /dev/uio0
    dma_path: /dev/uio1

fast_loop:                  # 快环仿真器参数
  fixed_point: Q4.20
  histogram_bins: 32
  syndrome_limit: 3.0
  correction_limit: 2.5
```

### P4 特有段

```yaml
slow_loop:
  mode: hybrid_residual_b   # 慢环模式
  hybrid_residual_b:
    teacher_mode: ukf
    context_windows: 3
    residual_scale: 1.0
    residual_clip_b: 0.5

p4_benchmark:               # 正式基准协议
  protocol:
    protocol_id: p4_v1
    repeats: 10
    frozen_baseline_set: [static_linear, window_variance, ekf, cnn_fpga]
  modes:
    - name: cnn_fpga
      slow_loop_mode: hybrid_residual_b
  scenarios:
    - name: slow_linear
      overrides: {drift: ...}
```

### 硬件默认值（P1/P2/P3 共有）

```yaml
hardware_defaults:
  fixed_point: Q4.20
  t_fast_us: 5.0
  window_size: 2048
  t_slow_update_ms: 20.0

param_mapping:              # 参数映射配置
  alpha_bias: 0.95
  beta_smoothing: 0.2
  gain_clip: [0.1, 2.0]
  theta_clip_deg: [-45.0, 45.0]
```

## 配置加载方式

所有脚本通过 `cnn_fpga.utils.config.load_yaml_config(path)` 加载配置，该函数自动：

1. 读取 YAML 文件
2. 若存在 `base_config` 字段，递归加载父配置并深度合并
3. 子配置的字段覆盖父配置的同名字段

```python
from cnn_fpga.utils.config import load_yaml_config

config = load_yaml_config("cnn_fpga/config/hardware_hil.yaml")
```
