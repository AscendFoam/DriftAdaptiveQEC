"""
Error Correction Module

Implements GKP error correction protocols:
- Linear decoder (parameterized)
- Full QEC cycle simulation
- Multiple correction strategies

中文说明：
- 该模块实现 GKP 纠错的核心流程：综合征测量、线性解码、残差统计与多轮仿真。
- 上层实验通常通过 GKPErrorCorrector / QECSimulator 访问这里的能力。
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass, field

from .gkp_state import LATTICE_CONST, ApproximateGKPState
from .syndrome_measurement import RealisticSyndromeMeasurement, MeasurementConfig
from .noise_channels import CombinedNoiseModel


@dataclass
class DecoderParameters:
    """Parameters for linear decoder"""
    K: np.ndarray = field(default_factory=lambda: np.eye(2))  # Gain matrix
    b: np.ndarray = field(default_factory=lambda: np.zeros(2))  # Bias vector

    def to_flat(self) -> np.ndarray:
        """Convert to flat array [K_11, K_12, K_21, K_22, b_1, b_2]"""
        return np.concatenate([self.K.flatten(), self.b])

    @classmethod
    def from_flat(cls, params: np.ndarray) -> 'DecoderParameters':
        """Create from flat array"""
        K = params[:4].reshape(2, 2)
        b = params[4:]
        return cls(K=K, b=b)


class LinearDecoder:
    """
    Parameterized linear decoder for GKP error correction

    Given syndrome s = [sq, sp], computes correction:
    Δ = K @ s + b

    where K is a 2x2 gain matrix and b is a bias vector.

    For optimal decoding under Gaussian noise with variance σ²:
    - K should be close to identity (gain ≈ 1)
    - For noisy measurements, K < I compensates for measurement noise
    - b corrects systematic biases (control drift)
    """

    def __init__(self,
                 K: Optional[np.ndarray] = None,
                 b: Optional[np.ndarray] = None):
        """
        Args:
            K: 2x2 gain matrix (default: identity)
            b: 2D bias vector (default: zero)
        """
        self.K = K if K is not None else np.eye(2)
        self.b = b if b is not None else np.zeros(2)
        self.lattice = LATTICE_CONST

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """
        Compute correction displacement from syndrome

        Args:
            syndrome: [sq, sp] measured syndrome

        Returns:
            correction: [dq, dp] correction to apply
        """
        # 中文注释：线性解码核心公式 Δ = K @ s + b。
        return self.K @ syndrome + self.b

    def update(self, K: np.ndarray, b: np.ndarray):
        """Update decoder parameters"""
        self.K = K
        self.b = b

    def update_from_flat(self, params: np.ndarray):
        """Update from flat parameter array"""
        self.K = params[:4].reshape(2, 2)
        self.b = params[4:]

    def get_params(self) -> DecoderParameters:
        """Get parameters as dataclass"""
        return DecoderParameters(K=self.K.copy(), b=self.b.copy())

    def get_flat_params(self) -> np.ndarray:
        """Get parameters as flat array"""
        return np.concatenate([self.K.flatten(), self.b])


def compute_optimal_decoder_params(sigma: float,
                                    delta: float,
                                    theta: float = 0.0,
                                    meas_efficiency: float = 0.95) -> DecoderParameters:
    """
    Compute optimal linear decoder parameters for given noise

    Args:
        sigma: Displacement noise standard deviation
        delta: GKP finite energy parameter
        theta: Phase space rotation angle
        meas_efficiency: Measurement efficiency

    Returns:
        Optimal decoder parameters
    """
    # 中文注释：使用近似 Wiener 思路估计最优线性增益与旋转矩阵。
    # Total effective noise variance
    # Includes: GKP finite energy, displacement noise, measurement noise
    var_signal = (LATTICE_CONST / 2) ** 2 / 3  # Uniform signal variance

    # Noise contributions
    var_displacement = sigma ** 2
    var_gkp = delta ** 2
    var_meas = (1 - meas_efficiency) / (2 * meas_efficiency)

    var_noise = var_displacement + var_gkp + var_meas

    # Optimal Wiener filter gain
    gain = var_signal / (var_signal + var_noise)

    # Rotation matrix for phase drift
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

    # Combined gain matrix
    K = gain * R

    # Bias (usually zero unless there's systematic drift)
    b = np.zeros(2)

    return DecoderParameters(K=K, b=b)


class GKPErrorCorrector:
    """
    Complete GKP error correction system

    Combines:
    - Syndrome measurement (with noise)
    - Decoder (linear, parameterized)
    - Correction application
    """

    def __init__(self,
                 delta: float = 0.3,
                 decoder: Optional[LinearDecoder] = None,
                 measurement_config: Optional[MeasurementConfig] = None):
        """
        Args:
            delta: GKP finite energy parameter
            decoder: Linear decoder (created automatically if None)
            measurement_config: Measurement configuration
        """
        self.delta = delta
        self.lattice = LATTICE_CONST

        # Setup measurement
        if measurement_config is None:
            measurement_config = MeasurementConfig(delta=delta)
        self.measurement = RealisticSyndromeMeasurement(measurement_config)

        # Setup decoder
        if decoder is None:
            # Create optimal decoder for current noise level
            params = compute_optimal_decoder_params(sigma=0.3, delta=delta)
            decoder = LinearDecoder(K=params.K, b=params.b)
        self.decoder = decoder

    def run_qec_round(self,
                      error: np.ndarray,
                      add_measurement_noise: bool = True) -> Dict[str, Any]:
        """
        Run one round of quantum error correction

        Args:
            error: [eq, ep] displacement error to correct
            add_measurement_noise: Whether to add measurement noise

        Returns:
            Dictionary with:
            - syndrome: Measured syndrome
            - correction: Applied correction
            - residual: Remaining error after correction
            - success: Whether error was successfully corrected
        """
        # 1. 测量综合征（可选加噪）
        syndrome = self.measurement.measure(error, add_noise=add_measurement_noise)

        # 2. 解码得到校正位移
        correction = self.decoder.decode(syndrome)

        # 3. 应用校正并跟踪残差
        residual = error - correction

        # 4. 判定纠错是否成功：残差是否在基本晶胞范围内
        success = (np.abs(residual[0]) < self.lattice / 2 and
                   np.abs(residual[1]) < self.lattice / 2)

        return {
            'syndrome': syndrome,
            'correction': correction,
            'residual': residual,
            'success': success,
            'error': error,
        }

    def evaluate_performance(self,
                             n_samples: int = 10000,
                             error_sigma: float = 0.3) -> Dict[str, float]:
        """
        Evaluate decoder performance

        Args:
            n_samples: Number of error samples
            error_sigma: Standard deviation of errors to test

        Returns:
            Performance metrics
        """
        successes = 0
        residuals = []

        for _ in range(n_samples):
            # Random error
            error = np.random.normal(0, error_sigma, size=2)

            # Run QEC
            result = self.run_qec_round(error)

            if result['success']:
                successes += 1
            residuals.append(result['residual'])

        residuals = np.array(residuals)

        return {
            'logical_error_rate': 1 - successes / n_samples,
            'success_rate': successes / n_samples,
            'mean_residual_q': np.mean(residuals[:, 0]),
            'mean_residual_p': np.mean(residuals[:, 1]),
            'std_residual_q': np.std(residuals[:, 0]),
            'std_residual_p': np.std(residuals[:, 1]),
            'n_samples': n_samples,
        }

    def update_decoder(self, K: np.ndarray, b: np.ndarray):
        """Update decoder parameters"""
        self.decoder.update(K, b)

    def update_decoder_from_fno(self, fno_output: np.ndarray):
        """Update decoder from FNO network output"""
        self.decoder.update_from_flat(fno_output)


class QECSimulator:
    """
    Full QEC simulation over multiple rounds

    Simulates:
    1. State preparation
    2. Noise application
    3. Error correction
    4. Repeat for multiple rounds
    """

    def __init__(self,
                 delta: float = 0.3,
                 noise_model: Optional[CombinedNoiseModel] = None,
                 corrector: Optional[GKPErrorCorrector] = None):
        """
        Args:
            delta: GKP finite energy parameter
            noise_model: Noise model for errors
            corrector: Error corrector
        """
        self.delta = delta
        self.lattice = LATTICE_CONST

        if noise_model is None:
            noise_model = CombinedNoiseModel(gamma=0.05, n_bar=0.01, sigma_disp=0.1)
        self.noise_model = noise_model

        if corrector is None:
            corrector = GKPErrorCorrector(delta=delta)
        self.corrector = corrector

    def simulate_multiple_rounds(self,
                                  n_rounds: int = 100,
                                  error_sigma: float = 0.3) -> Dict[str, Any]:
        """
        Simulate multiple QEC rounds

        Args:
            n_rounds: Number of QEC rounds
            error_sigma: Error standard deviation per round

        Returns:
            Simulation results
        """
        results = []
        cumulative_error = np.zeros(2)

        for round_idx in range(n_rounds):
            # New error this round
            new_error = np.random.normal(0, error_sigma, size=2)

            # Total error = accumulated + new
            total_error = cumulative_error + new_error

            # Run QEC
            round_result = self.corrector.run_qec_round(total_error)
            results.append(round_result)

            # Update cumulative error with residual
            cumulative_error = round_result['residual']

        # Analyze results
        successes = sum(1 for r in results if r['success'])

        return {
            'n_rounds': n_rounds,
            'successes': successes,
            'logical_error_rate': 1 - successes / n_rounds,
            'round_results': results,
        }

    def run_with_drift(self,
                       n_timesteps: int,
                       drift_model,
                       recalibrate_every: int = 50) -> Dict[str, Any]:
        """
        Run QEC with drifting noise parameters

        Args:
            n_timesteps: Number of time steps
            drift_model: Function t -> (sigma, delta, theta)
            recalibrate_every: Recalibration interval

        Returns:
            Results including error rates over time
        """
        error_rates = []

        for t in range(n_timesteps):
            # Get current noise parameters
            sigma_t, delta_t, theta_t = drift_model(t)

            # Evaluate at current noise level
            metrics = self.corrector.evaluate_performance(
                n_samples=1000,
                error_sigma=sigma_t
            )
            error_rates.append(metrics['logical_error_rate'])

            # Periodic recalibration (in real system, would use FNO here)
            if t % recalibrate_every == 0 and t > 0:
                # Update decoder for current noise
                opt_params = compute_optimal_decoder_params(
                    sigma=sigma_t,
                    delta=delta_t,
                    theta=theta_t
                )
                self.corrector.update_decoder(opt_params.K, opt_params.b)

        return {
            'n_timesteps': n_timesteps,
            'error_rates': np.array(error_rates),
            'mean_error_rate': np.mean(error_rates),
        }
        
"""
### 代码核心功能解析
这段代码是**GKP量子纠错（QEC）** 的核心实现模块，专门用于纠正GKP（Gottesman-Kitaev-Preskill）量子比特在演化过程中产生的位移误差，以下分模块解析核心逻辑：

#### 1. 核心数据结构与基础组件
- **DecoderParameters**：数据类，封装线性解码器的核心参数——2x2增益矩阵`K`（控制纠错强度）和2维偏置向量`b`（补偿系统偏移），并提供参数扁平化/还原的方法（适配后续参数优化）。
- **LinearDecoder**：线性解码器核心类，核心逻辑是 `Δ = K @ s + b`（`s`为测量到的综合征，`Δ`为待施加的校正位移）。支持参数更新、最优参数计算，是纠错的“决策核心”。

#### 2. 最优解码器参数计算
`compute_optimal_decoder_params` 函数基于**维纳滤波（Wiener Filter）** 思路，结合噪声来源（位移噪声、GKP有限能量噪声、测量噪声）计算最优增益矩阵：
- 核心逻辑：通过信号方差与总噪声方差的比值确定最优增益`gain`，再结合相位漂移的旋转矩阵`R`得到最终`K`；偏置`b`默认归零（无系统漂移时）。
- 输入：噪声标准差、GKP有限能量参数、相位旋转角、测量效率；输出：最优解码器参数。

#### 3. 单轮纠错核心类 GKPErrorCorrector
这是单轮量子纠错的完整实现，流程为：
1. **综合征测量**：通过`RealisticSyndromeMeasurement`测量误差对应的综合征（可选加入测量噪声，模拟真实实验）；
2. **解码计算校正量**：用`LinearDecoder`将综合征转换为校正位移；
3. **残差计算与成功判定**：残差=原始误差-校正量，若残差落在GKP晶格基本单元内（<晶格常数/2），则判定纠错成功；
4. **性能评估**：通过大量随机误差样本，统计成功概率、逻辑错误率、残差统计特征。

#### 4. 多轮纠错仿真 QECSimulator
模拟真实量子系统的多轮纠错过程，核心能力：
- **多轮仿真**：累计每轮误差→执行单轮纠错→更新累计残差，统计多轮纠错的整体成功率；
- **带漂移的仿真**：模拟噪声参数随时间漂移的场景，支持周期性重新校准解码器参数（适配实时噪声），输出不同时间步的逻辑错误率。

#### 5. 关键概念补充
- **GKP晶格常数（LATTICE_CONST）**：GKP量子比特的相位空间晶格单元大小，是判定纠错是否成功的核心阈值；
- **综合征（syndrome）**：误差的“特征指纹”，通过测量提取，是纠错的依据；
- **残差（residual）**：纠错后未被消除的剩余误差，直接反映纠错效果。

---

### 总结
1. **核心逻辑**：以“测量综合征→线性解码算校正量→应用校正→判定成功”为核心流程，实现GKP量子比特的误差纠正；
2. **优化思路**：基于维纳滤波计算最优解码器参数，平衡信号与噪声，最大化纠错成功率；
3. **仿真能力**：支持单轮/多轮纠错、带噪声/漂移的真实场景仿真，输出错误率、残差等关键性能指标。

简言之，这段代码是GKP量子纠错从“理论解码”到“工程仿真”的完整实现，核心是通过线性解码将测量到的误差特征转换为精准的校正动作，最终降低量子比特的逻辑错误率。
"""