# decoder/ — 解码器与基线估计器

本目录实现了 FPGA 快环线性解码器的软件模型，以及多种慢环噪声参数估计基线算法。解码器将综合征 (syndrome) 转化为校正位移 (correction)，而基线估计器从直方图窗口中提取噪声参数预测。

## 目录结构

| 文件 | 职责 |
|------|------|
| [linear_runtime.py](linear_runtime.py) | 快环线性解码器软件模型（浮点 + 定点量化） |
| [param_mapper.py](param_mapper.py) | 噪声参数 → 解码器参数映射 (σ,μ,θ → K, b) |
| [window_baseline.py](window_baseline.py) | 窗口方差基线（直方图矩估计） |
| [ekf_baseline.py](ekf_baseline.py) | 扩展卡尔曼滤波基线 |
| [ukf_baseline.py](ukf_baseline.py) | 无迹卡尔曼滤波基线（8 维状态） |
| [particle_filter_baseline.py](particle_filter_baseline.py) | 粒子滤波基线 + 残差-b 粒子滤波 |
| [rls_residual_baseline.py](rls_residual_baseline.py) | 递归最小二乘残差基线 |
| [statcalib.py](statcalib.py) | 统计校准 (StatCalib) 输入/输出契约 |
| [\_\_init\_\_.py](__init__.py) | 惰性导出 22 个公共符号 |

### 模块依赖关系

```
layer 0 (无内部依赖):
  param_mapper.py    ← 定义 NoisePrediction（几乎被所有模块使用）
  statcalib.py       ← 独立契约定义
  linear_runtime.py  ← 独立运行时模型

layer 1 (依赖 param_mapper):
  window_baseline.py        → param_mapper.NoisePrediction
  rls_residual_baseline.py  → param_mapper.NoisePrediction

layer 2 (依赖 param_mapper + window_baseline):
  ekf_baseline.py           → window_baseline.*（作为测量源）
  ukf_baseline.py           → window_baseline.*（作为测量源）
  particle_filter_baseline.py → window_baseline.*（作为测量源）
```

## 核心类与接口

### 快环线性解码器 — `linear_runtime.py`

FPGA 快环解码器的软件模型：

- **`FixedPointFormat`** — 定点格式描述（如 Q4.20），提供 `quantize()` 量化方法
- **`LinearRuntime`** — 核心解码器：
  - 核心公式：`correction = K @ syndrome + b`
  - 浮点模式：直接矩阵乘加，输入/输出裁剪
  - 定点模式：量化 syndrome, K, b → 定点运算 → 反量化 → 裁剪
  - 返回 `LinearRuntimeResult`，包含原始/裁剪/量化各阶段数据及饱和标志

### 噪声参数映射 — `param_mapper.py`

将慢环预测的噪声参数转换为解码器运行时参数：

- **`NoisePrediction`** — 噪声参数预测容器（sigma, mu_q, mu_p, theta_deg, source, metadata）
- **`ParamMapper`** — 核心映射器：
  - 输入：`NoisePrediction` + 可选的上一轮参数
  - 输出：`DecoderRuntimeParams`（K, b）
  - 流程：构建旋转协方差矩阵 → 计算 Kalman 风格增益 K → 裁剪特征值 → 指数平滑 → 输出
- **`analyze_decoder_aggressiveness(K, b)`** — 启发式诊断：检测增益/偏置是否过于激进

### 窗口方差基线 — `window_baseline.py`

非时序单窗口基线，是所有滤波基线的测量源：

- **`HistogramMomentEstimator`** — 从直方图重建加权一阶/二阶矩
  - 输出 `HistogramMomentObservation`：均值、协方差、特征值、主轴、各向异性比
- **`WindowVarianceBaseline`** — 包装矩估计器生成 `NoisePrediction`
  - sigma 从协方差迹减去测量底噪得到
  - theta_deg 从主轴方向提取（各向异性过低时回退到默认值）

### 滤波基线

所有滤波基线共享同一模式：接受 `WindowVarianceBaseline` 作为测量源，调用其 `predict()` 获取观测，再通过各自的递归滤波算法产生平滑的 `NoisePrediction`。

| 基线 | 状态维度 | 核心特点 |
|------|----------|----------|
| **EKFBaseline** | 4 (σ, μq, μp, θ) | 随机游走过程模型 + 对角协方差 EKF 更新 |
| **UKFBaseline** | 8 (σ, μq, μp, θ, vσ, vμq, vμp, vθ) | 恒速度漂移模型 + Sigma 点传播 |
| **ParticleFilterBaseline** | 4 (σ, μq, μp, θ) | Bootstrap 粒子滤波 + 系统重采样 + 再生抖动 |

### 残差基线

残差基线不直接估计绝对噪声参数，而是跟踪 teacher 基线偏置向量的修正量 `delta_b`：

- **`RLSResidualBBaseline`** — 递归最小二乘在线学习
  - 特征向量 17 维：偏置项 + teacher/测量噪声预测 (8) + delta (4) + teacher/测量 b 值 (4)
  - 每个输出维度独立维护权重矩阵和协方差矩阵
- **`ParticleFilterResidualBBaseline`** — 2 维状态 `(delta_b_q, delta_b_p)` 的粒子滤波

### 统计校准契约 — `statcalib.py`

标准化的校准输入/输出接口：

- **`StatCalibInput`** — 输入契约：窗口 ID、当前参数、直方图摘要、校准特征、teacher 信息
- **`StatCalibOutput`** — 输出契约：状态（generated / not_generated / not_applicable / diagnostic_error）+ 可选的 K, b, delta_b

## 使用示例

### 快环解码

```python
from cnn_fpga.decoder import LinearRuntime
from cnn_fpga.decoder.linear_runtime import LinearRuntimeConfig, FixedPointFormat
from cnn_fpga.runtime.param_bank import DecoderRuntimeParams

# 浮点解码
config = LinearRuntimeConfig(enable_fixed_point=False)
runtime = LinearRuntime(config)
result = runtime.decode(syndrome=np.array([0.5, -0.3]), params=params)
print(result.correction_applied)

# 定点解码 (Q4.20)
fp_config = LinearRuntimeConfig(
    fixed_point_spec=FixedPointFormat.from_spec("Q4.20"),
    enable_fixed_point=True,
)
runtime_fp = LinearRuntime(fp_config)
result_fp = runtime_fp.decode(syndrome, params)
```

### 参数映射

```python
from cnn_fpga.decoder import ParamMapper, NoisePrediction

prediction = NoisePrediction(sigma=0.35, mu_q=0.02, mu_p=-0.01, theta_deg=5.0, source="ekf")
mapper = ParamMapper.from_config(config)
params = mapper.map_prediction(prediction, previous_params=None)
```

### 基线估计

```python
from cnn_fpga.decoder import WindowVarianceBaseline, EKFBaseline

# 窗口方差基线
wv = WindowVarianceBaseline.from_config(config)
pred = wv.predict(histogram_2d, window_id=1)

# EKF 基线（以窗口方差为测量源）
ekf = EKFBaseline.from_config(config, measurement_baseline=wv)
smoothed_pred = ekf.predict(histogram_2d, window_id=1)
```
