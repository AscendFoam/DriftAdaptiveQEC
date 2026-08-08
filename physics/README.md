# physics/ — GKP 量子纠错物理仿真库

本目录实现了一套完整的 GKP (Gottesman-Kitaev-Preskill) 连续变量量子纠错仿真链，覆盖从量子态构造、噪声演化、综合征测量、线性解码到逻辑错误率统计的全流程。

## 物理背景

GKP 码将一个逻辑量子比特编码到谐振子的相空间中。理想 GKP 态是相空间中的梳状态：

```
|GKP_0⟩ = Σ_n |nλ⟩_q,   λ = √(2π)
```

实际实现使用有限能量的近似态（高斯包络），用参数 Δ 控制与理想态的偏离程度。纠错的核心思路是：测量位移误差对晶格常数取模得到**综合征 (syndrome)**，再根据综合征施加反向位移完成纠错。

本库基于这一原理，构建了如下仿真管线：

```
状态构造 → 噪声注入 → 综合征测量 → 线性解码 → 残差累积 → 逻辑错误判定
```

## 目录结构

| 文件 | 职责 |
|------|------|
| [constants.py](constants.py) | 共享物理常量 `LATTICE_CONST = √(2π)` |
| [gkp_state.py](gkp_state.py) | GKP 态构造与 Wigner 函数近似计算 |
| [noise_channels.py](noise_channels.py) | 噪声通道建模（光子损失、热噪声、位移噪声、相位噪声） |
| [syndrome_measurement.py](syndrome_measurement.py) | 综合征测量模型（理想 / 带噪声 / 自适应） |
| [error_correction.py](error_correction.py) | 线性解码器、单轮纠错、多轮仿真、漂移场景 |
| [logical_tracking.py](logical_tracking.py) | 逻辑错误追踪、滑窗统计、误差累积仿真入口 |
| [\_\_init\_\_.py](__init__.py) | 统一导出 API（惰性加载） |

### 模块依赖关系

```
constants.py (LATTICE_CONST)
    ↓
gkp_state.py  ←  noise_channels.py
    ↓                   ↓
syndrome_measurement.py → error_correction.py
    ↓                          ↓
    └───── logical_tracking.py ─┘
```

## 核心类与接口速查

### 状态与常量 — `gkp_state.py`

- **`LATTICE_CONST = √(2π) ≈ 2.507`** — GKP 晶格常数，全库共享
- **`ApproximateGKPState`** — 近似 GKP 态对象，支持 Wigner 函数计算和位移操作
- **`GKPStateFactory`** — 工厂类，快速创建 `|0⟩_L, |1⟩_L, |+⟩_L` 等逻辑态

### 噪声通道 — `noise_channels.py`

- **`CombinedNoiseModel`** — 组合噪声模型，统一施加多种噪声并输出等效 σ
- **`PhotonLossChannel`** — 光子损失（超导腔主导噪声），可从 T1 时间构造
- **`ThermalNoiseChannel`** — 热噪声，可从温度构造
- **`DisplacementNoiseChannel`** — 控制不完美导致的位移噪声

噪声施加顺序：光子损失 → 热噪声 → 位移噪声 → 相位噪声。

### 综合征测量 — `syndrome_measurement.py`

- **`MeasurementConfig`** — 测量参数容器（δ、探测效率、辅助比特错误率等）
- **`SyndromeMeasurement`** — 理想取模测量
- **`RealisticSyndromeMeasurement`** — 带真实噪声的测量（有限压缩噪声 + 探测效率损失 + shot noise + ancilla 错误）

测量噪声合并公式：

```
σ_meas = √(Δ² + (1-η)/(2η))
```

最优增益（Wiener 滤波）：

```
g* = σ²_signal / (σ²_signal + σ²_noise)
```

### 纠错与解码 — `error_correction.py`

- **`LinearDecoder`** — 线性解码器，核心公式：`Δ = K @ s + b`
- **`compute_optimal_decoder_params()`** — 根据 σ、Δ、θ 计算近似最优解码参数
- **`GKPErrorCorrector`** — 单轮纠错执行器：测量 → 解码 → 残差 → 成功判定
- **`QECSimulator`** — 多轮仿真器，支持漂移场景与周期重标定

### 逻辑错误统计 — `logical_tracking.py`

- **`LogicalErrorTracker`** — 累积残差追踪，q 方向越界记 X 错误，p 方向越界记 Z 错误
- **`WindowedErrorTracker`** — 滑窗 LER 追踪，用于漂移恶化检测
- **`ExperimentErrorTracker`** — 多配置实验统计
- **`simulate_error_accumulation()`** — 统一仿真入口，支持两种模型

## 两种仿真模型

`simulate_error_accumulation()` 通过 `use_full_qec_model` 参数切换两种模型：

| 特性 | simplified (`False`) | full_qec (`True`, 默认) |
|------|---------------------|------------------------|
| 跨轮残差继承 | 无（每轮独立） | 有（闭环） |
| 测量模型 | 简单加噪 | `RealisticSyndromeMeasurement` |
| 解码器 | 标量增益 | `LinearDecoder`（矩阵 K + 偏置 b） |
| 噪声各向异性 | 不支持 | 支持（θ 旋转） |
| 适用场景 | 历史兼容 / 快速对照 | 物理一致性仿真 |

full_qec 模型的一轮闭环流程：

```
上一轮 wrapped residual r_prev
  → 采样新噪声 n_t
  → total_error = r_prev + n_t (+ bias)
  → syndrome = measurement.measure(total_error)
  → correction = decoder.decode(syndrome)
  → tracker.update(total_error, correction)
  → 累积位移越界判定 + wrap → r_next
```

## 使用示例

### 基本导入

```python
from physics import (
    ApproximateGKPState,
    GKPStateFactory,
    PhotonLossChannel,
    CombinedNoiseModel,
    RealisticSyndromeMeasurement,
    LinearDecoder,
    GKPErrorCorrector,
    LogicalErrorTracker,
)
```

### 单轮纠错

```python
import numpy as np

corrector = GKPErrorCorrector(delta=0.3)
error = np.array([0.5, -0.3])
result = corrector.run_qec_round(error)
# result: {syndrome, correction, residual, success, error}
```

### 多轮逻辑错误率仿真

```python
stats = simulate_error_accumulation(
    n_rounds=10000,
    sigma_error=0.3,
    sigma_measurement=0.1,
    gain=0.8,
    delta=0.3,
    use_full_qec_model=True,
    seed=42,
)
print(f"LER: {stats['total_error_rate']:.4f}")
```

### 漂移场景仿真

```python
from physics import QECSimulator

def drift_model(t):
    sigma = 0.3 + 0.001 * t
    delta = 0.3
    theta = 0.01 * np.sin(0.1 * t)
    return sigma, delta, theta

sim = QECSimulator(delta=0.3)
results = sim.run_with_drift(n_timesteps=500, drift_model=drift_model, recalibrate_every=50)
print(f"平均错误率: {results['mean_error_rate']:.4f}")
```

### 自适应解码器更新

```python
from physics.error_correction import compute_optimal_decoder_params

corrector = GKPErrorCorrector(delta=0.3)
opt_params = compute_optimal_decoder_params(sigma=0.35, delta=0.3, theta=0.02)
corrector.update_decoder(opt_params.K, opt_params.b)
```

## 关键参数说明

| 参数 | 典型值 | 影响 |
|------|--------|------|
| `delta` | 0.2–0.5 | 有限能量参数，越小越理想但能量越高 |
| `sigma_error` | 0.2–0.5 | 每轮位移噪声标准差 |
| `measurement_efficiency` | 0.9–0.99 | 探测器效率，越低测量噪声越大 |
| `gain` | 0.5–1.0 | 解码增益，过大会过补偿 |
| `theta` | 弧度 | 相位漂移旋转角 |
| `error_bias` | [μ_q, μ_p] | 系统位移偏置 |

## 依赖

- **必需**: NumPy, SciPy
- **可选**: Strawberry Fields（精确量子态模拟，不可用时自动退化为解析近似）
## Phase 9 因果数字孪生接口合同

- [`phase9_twin_contract.py`](phase9_twin_contract.py) 实现 T9.2.1 的 pre-backend 有限接口：五个 namespace、observed+memory 唯一 deployable path、24-bit composite K、总 nominal/transition recurrence、80-bit semantic+CRC action、previous-K receipt 重算、fail-closed fault canonicalization、完整 trusted-package nomination validator 与16个 conservative interface probes。
- [`phase9_backend_a.py`](phase9_backend_a.py) 实现 T9.2.2 的 backend A：有限 Fock oscillator × `g/e/f` qutrit joint density、time-dependent action/Ramsey Hamiltonian、GKSL channels、连续 IQ Kraus backaction、条件 reset instrument、`f` persistence、action-conditioned 五维 drift 与 evaluator-only 六态 logical projection。资格检查含完整小系统 Choi CP/TP、joint-state physicality、ideal/reset limits、真实 oscillator syndrome backaction、共同随机数 intervention、seed replay 与 step/cutoff convergence。
- [`phase9_backend_b.py`](phase9_backend_b.py) 实现 T9.2.3 的独立 backend B：dense midpoint Strang unitary、解析 pure-loss/qutrit amplitude/dephasing channels、BLAKE2b+Python Random+手写 Box–Muller、独立 IQ Gaussian likelihood/Kraus backaction、reset/f persistence、action-conditioned leakage/drift 与独立 squeezed-comb 六态 evaluator。静态 AST/token 隔离、闭式 loss/relaxation、Choi/full physicality、non-TP mutation、seed 与 split/cutoff convergence 均纳入资格门。
- T9.2.2/T9.2.3 PASS 分别只资格化 synthetic dimensionless backend A/B 的实现。IQ 仍是 analog pre-frontend，双后端 distributional agreement、LER/lifetime、physical break-even、codebook、硬件与 SOTA/rank 均未资格化。
