# data/ — 数据集构建

本目录负责生成 CNN-FPGA 实验所需的训练/验证/测试数据集。提供两种数据构建模式：基于参数采样的合成直方图（P1 静态数据集），以及基于闭环仿真的运行时一致窗口数据集（P4 repair）。

## 文件

| 文件 | 职责 |
|------|------|
| [dataset_builder.py](dataset_builder.py) | P1 合成数据集构建器（参数采样 → 直方图） |
| [runtime_dataset_builder.py](runtime_dataset_builder.py) | P4 运行时窗口数据集构建器（闭环仿真 → 特征 + 残差标签） |

## 两种数据构建模式对比

| 特性 | dataset_builder (P1) | runtime_dataset_builder (P4) |
|------|---------------------|------------------------------|
| 数据来源 | 参数采样 + 解析直方图 | 闭环仿真链（FPGA Driver + 快/慢环） |
| 输入 | 2D 直方图 | 3D 张量 (多帧直方图 + teacher 特征通道) + 标量特征 |
| 标签 | (σ, μq, μp, θ_deg) | 残差标签 (residual_mu 或 residual_b) |
| 噪声 | 合成高斯（各向同性/各向异性） | 运行时真实噪声 |
| 依赖深度 | 低（仅 NumPy + physics.constants） | 高（runtime + decoder + hwio + benchmark） |

## 核心功能 — `dataset_builder.py`

P1 阶段的合成数据集构建器，用于静态 CNN 训练。

### 构建流程

```
配置加载 → 参数采样 (σ, μq, μp, θ) → 各向同性/各向异性高斯噪声
→ 旋转 θ → 位移 (μq, μp) → 模晶格常数取模 → 2D 直方图 (32×32) → 归一化
→ train/val/test 划分 → 保存 .npz + manifest.json
```

### 标签

```python
LABEL_NAMES = ("sigma", "mu_q", "mu_p", "theta_deg")
```

### CLI

```bash
python -m cnn_fpga.data.dataset_builder --config cnn_fpga/config/experiment_static.yaml

# 预览模式（不生成数据）
python -m cnn_fpga.data.dataset_builder --config config.yaml --dry-run
```

### 输出

- `{dataset_dir}/train.npz` — 直方图 (N, 32, 32) + 标签 (N, 4)
- `{dataset_dir}/val.npz`
- `{dataset_dir}/test.npz`
- `{dataset_dir}/manifest.json` — 元数据（种子、样本数、划分比例、参数范围等）

## 核心功能 — `runtime_dataset_builder.py`

P4 阶段的运行时一致窗口数据集构建器，通过实际运行闭环仿真链捕获真实直方图窗口，再计算残差标签。

### 构建流程

```
1. 加载配置（含 base_config 继承）
2. 对每个场景：
   a. 用参考模式（如 static_linear）运行闭环仿真，捕获 n 个窗口
      → 直方图 + 目标参数 + 诊断数据
   b. 对每个窗口运行 teacher 慢环推理
      → teacher 预测 + teacher 参数 (b)
   c. 构建特征张量（多帧直方图 + teacher 特征）
   d. 计算残差标签：
      - residual_mu: teacher_mu - target_mu
      - residual_b:  teacher_b - target_b
3. train/val/test 划分 → 保存 .npz + manifest.json
```

### 特征构建

由 `RuntimeFeatureConfig` 控制，支持：

- 多帧上下文窗口（`context_windows`）
- 直方图增量通道（`include_histogram_deltas`）
- Teacher 预测通道（`include_teacher_prediction`）
- Teacher 参数通道（`include_teacher_params`）
- Teacher 参数增量通道（`include_teacher_deltas`）
- 两种布局：`broadcast`（空间平面）或 `scalar_branch`（独立向量）

### 数据结构

**`CapturedRuntimeWindow`** — 捕获的单个窗口：

- `scenario_name`, `window_id`
- `histogram` (32×32 ndarray)
- `target_params` (真实噪声参数)
- `diagnostics`, `window_stats`

### CLI

```bash
python -m cnn_fpga.data.runtime_dataset_builder \
    --config cnn_fpga/config/experiment_runtime_b_residual.yaml

# 仅生成指定场景
python -m cnn_fpga.data.runtime_dataset_builder \
    --config config.yaml --scenario slow_linear
```

### 输出

比 P1 数据集更丰富：

- `{dataset_dir}/train.npz`:
  - `histograms` — 空间张量 (N, C, 32, 32)
  - `scalar_features` — 标量特征 (N, D)
  - `labels` — 残差标签 (N, 2)
  - `teacher_predictions` — teacher 预测值
  - `target_params` — 目标参数
  - `teacher_runtime_b` / `target_runtime_b` — 解码器偏置
  - `scenario_names`, `window_ids`
- `{dataset_dir}/manifest.json` — 包含特征配置、标签语义、标签统计等详细元数据

## 共同模式

两个构建器共享相同的 CLI 和输出模式：

1. `--config` (必需) — YAML 配置路径
2. `--dry-run` — 仅打印计划不执行
3. `_split_indices()` — 随机划分 train/val/test
4. `_save_split()` — 保存为压缩 `.npz`
5. `manifest.json` — 记录数据集构建的完整元数据

## 依赖

- **dataset_builder**: NumPy, `physics.constants.LATTICE_CONST`, `cnn_fpga.utils.config`
- **runtime_dataset_builder**: 额外依赖 `cnn_fpga.hwio`, `cnn_fpga.runtime`, `cnn_fpga.decoder`, `cnn_fpga.benchmark`
