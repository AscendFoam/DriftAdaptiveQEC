# model/ — CNN 模型训练与部署

本目录实现了 TinyCNN 模型的定义、训练、量化、导出和评估的完整流水线。模型接收直方图窗口（和可选的标量特征），预测 GKP 噪声参数 (σ, μq, μp, θ)，用于慢环的自适应解码器参数更新。

## 目录结构

| 文件 | 职责 |
|------|------|
| [tiny_cnn.py](tiny_cnn.py) | TinyCNN 核心：模型定义（NumPy + PyTorch）、训练、推理、artifact 序列化 |
| [train.py](train.py) | CLI 训练入口（线性回归基线 / TinyCNN） |
| [evaluate.py](evaluate.py) | CLI 评估入口（NPZ artifact，float/int8） |
| [quantize.py](quantize.py) | CLI 量化入口（float → int8 对称量化） |
| [export.py](export.py) | CLI 导出入口（NPZ → TFLite / JSON stub） |
| [evaluate_tflite.py](evaluate_tflite.py) | CLI TFLite 评估入口（通过 TFLite Interpreter 推理） |
| [validate_export.py](validate_export.py) | CLI 导出验证（NPZ vs TFLite 预测一致性） |
| [\_\_init\_\_.py](__init__.py) | 导出 TinyCNN, TinyCNNConfig, fit_tiny_cnn, predict_from_artifact 等 |

### 流水线

```
train.py  →  quantize.py  →  export.py  →  validate_export.py
   │                              │                  │
   ↓                              ↓                  ↓
evaluate.py              evaluate_tflite.py    (对比两条路径)
```

## 核心类与接口

### TinyCNN 模型 — `tiny_cnn.py`

一个极小的卷积神经网络回归器，全量用 NumPy 实现（可选 PyTorch 后端）。

**架构：**

```
输入: (N, 1, 32, 32)  直方图
  → Conv2D(1→8, k=3, same) → ReLU → AvgPool2x2
  → Flatten
  → [可选] 拼接/门控标量特征
  → Dense(8*16*16 → 32) → ReLU
  → Dense(32 → 4)
输出: (N, 4)  [sigma, mu_q, mu_p, theta_deg]
```

**核心组件：**

- **`TinyCNNConfig`** — 超参数配置（conv_channels, kernel_size, hidden_dim, epochs, learning_rate, label_weights, backend, 标量融合模式等）
- **`TinyCNN`** — NumPy 实现：
  - `_forward_internal(x, scalar)` → `(output, cache)` — 前向传播 + 缓存（用于反向传播）
  - `loss_and_grads(x, scalar, y_true)` — 加权 MSE 损失 + L2 正则 + 完整反向传播
  - `apply_adam(grads, state, step)` — Adam 优化器
- **`_TorchTinyCNN`** — PyTorch 镜像实现（仅在 `backend == "torch"` 时使用）

**标量特征融合：**

- `concat` 模式：标量向量直接拼接到卷积展平向量后
- `gated` 模式：通过 sigmoid 门控加权标量特征，可解释性更强

**Artifact 格式：**

训练后保存为 `.npz` 文件，包含：

| 键 | 内容 |
|---|---|
| `conv_w`, `conv_b` | 卷积核权重和偏置 |
| `fc1_w`, `fc1_b`, `fc2_w`, `fc2_b` | 全连接层权重 |
| `scalar_gate_w/b`, `scalar_shift_w/b` | 标量门控参数（可选） |
| `input_mean`, `input_std` | 输入归一化统计 |
| `label_mean`, `label_std` | 标签归一化统计 |
| `model_type` | 模型类型标识 |
| `config` | 训练配置 JSON |

### 训练 — `train.py`

CLI 入口点，支持两种模型：

```bash
# TinyCNN 训练
python -m cnn_fpga.model.train --config cnn_fpga/config/experiment_static.yaml

# 线性回归基线
# (通过 config 中 model_type: linear_regression_baseline 指定)
```

内部流程：
1. 加载数据集 (train.npz / val.npz)
2. 根据 `model_type` 选择训练路径
3. TinyCNN：调用 `fit_tiny_cnn()`，保存 `.npz` artifact + 训练报告 JSON
4. 线性回归：闭式 L2 正则最小二乘，保存权重 artifact

### 量化 — `quantize.py`

将 float32 权重量化为 int8（对称量化）：

- 每个张量独立计算 scale = max_abs / 127
- 保存 `q_{name}` (int8 数组) + `{name}_scale` (float 标量)
- 保留所有元数据字段和归一化统计

### 导出 — `export.py`

将 NPZ artifact 转换为部署格式：

- **TFLite 导出**：重建等价的 `tf.keras.Model`，转换为 `.tflite`（需 TensorFlow）
- **Stub 回退**：TensorFlow 不可用时写入 `.tflite.json` 引用原始 NPZ

### 评估 — `evaluate.py` / `evaluate_tflite.py`

| 脚本 | 推理路径 | 用途 |
|------|----------|------|
| `evaluate.py` | NPZ artifact → NumPy 推理 | 训练后快速评估 |
| `evaluate_tflite.py` | .tflite → TFLite Interpreter | 部署验证 |

两者均输出 MSE, MAE, R² 和逐标签指标。

### 导出验证 — `validate_export.py`

交叉验证 NPZ artifact 推理和 TFLite 推理的一致性，捕获归一化、通道顺序或量化引入的差异。

```bash
python -m cnn_fpga.model.validate_export \
    --config ... \
    --artifact-path model.npz \
    --tflite-path model.tflite \
    --max-samples 128
```

## 使用示例

### 训练

```python
from cnn_fpga.model import fit_tiny_cnn, TinyCNNConfig

config = TinyCNNConfig(epochs=30, learning_rate=0.003, backend="numpy")
artifact, report = fit_tiny_cnn(
    train_histograms, train_labels,
    val_histograms, val_labels,
    config=config,
)
```

### 从 Artifact 推理

```python
from cnn_fpga.model import predict_from_artifact

prediction = predict_from_artifact("model.npz", histogram_2d)
# prediction: NoisePrediction(sigma=..., mu_q=..., mu_p=..., theta_deg=...)
```

### 完整流水线

```bash
# 1. 训练
python -m cnn_fpga.model.train --config config.yaml

# 2. 评估
python -m cnn_fpga.model.evaluate --config config.yaml

# 3. 量化
python -m cnn_fpga.model.quantize --config config.yaml

# 4. 导出
python -m cnn_fpga.model.export --config config.yaml

# 5. TFLite 评估
python -m cnn_fpga.model.evaluate_tflite --config config.yaml

# 6. 导出验证
python -m cnn_fpga.model.validate_export \
    --config config.yaml \
    --artifact-path model.npz \
    --tflite-path model.tflite
```

## 依赖

- **必需**: NumPy
- **可选**: PyTorch (torch 后端), TensorFlow (TFLite 导出/评估), tflite_runtime (轻量 TFLite 推理)
