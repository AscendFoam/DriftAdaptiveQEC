"""
Logical Error Tracking Module

Tracks logical errors over multiple QEC rounds.
GKP logical errors occur when:
- Accumulated q displacement exceeds ±√(2π)/2 → Logical X error
- Accumulated p displacement exceeds ±√(2π)/2 → Logical Z error

中文说明：
- 本模块用于把“每轮残差”转成“逻辑错误率”统计结果。
- 支持两种误差累积仿真：简化模型（legacy）与完整闭环模型（full_qec）。
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from .constants import LATTICE_CONST


@dataclass
class LogicalErrorEvent:
    """Record of a logical error event"""
    timestep: int
    error_type: str  # 'X', 'Z', or 'Y'
    accumulated_q: float
    accumulated_p: float


class LogicalErrorTracker:
    """
    Tracks logical errors in GKP error correction

    GKP codes have logical Pauli operators:
    - X_L: Displacement by √(2π) in q
    - Z_L: Displacement by √(2π) in p

    A logical error occurs when the accumulated uncorrected
    displacement crosses the decision boundary at ±√(2π)/2.
    """

    def __init__(self):
        self.lattice = LATTICE_CONST
        self.reset()

    def reset(self):
        """Reset tracking state"""
        self.accumulated_q = 0.0
        self.accumulated_p = 0.0
        self.logical_x_errors = 0
        self.logical_z_errors = 0
        self.total_rounds = 0
        self.error_history: List[LogicalErrorEvent] = []

    def update(self,
               error_q: float,
               error_p: float,
               correction_q: float,
               correction_p: float) -> Tuple[bool, bool]:
        """
        Update accumulated displacement and check for logical errors

        In each QEC round:
        1. We measure syndrome (noisy estimate of error mod lattice)
        2. We apply correction based on syndrome
        3. Residual = true_error - correction accumulates over rounds

        If we make a wrong decision (error was actually closer to a
        different lattice point), we accumulate ±λ/2 extra displacement.

        Args:
            error_q: True q error (before correction)
            error_p: True p error (before correction)
            correction_q: Applied q correction
            correction_p: Applied p correction

        Returns:
            (x_error, z_error): Whether each type of logical error occurred
        """
        # 中文注释：每次 update 都对应一次“真实误差 + 校正后”的逻辑状态推进。
        self.total_rounds += 1

        # The residual is the true error minus the correction
        # This accumulates if corrections are imperfect
        residual_q = error_q - correction_q
        residual_p = error_p - correction_p

        self.accumulated_q += residual_q
        self.accumulated_p += residual_p

        x_error = False
        z_error = False

        # Check for logical X error (accumulated q displacement too large)
        if abs(self.accumulated_q) > self.lattice / 2:
            self.logical_x_errors += 1
            x_error = True
            self.error_history.append(LogicalErrorEvent(
                timestep=self.total_rounds,
                error_type='X',
                accumulated_q=self.accumulated_q,
                accumulated_p=self.accumulated_p
            ))
            # Wrap accumulated value
            self.accumulated_q = np.mod(self.accumulated_q + self.lattice / 2,
                                         self.lattice) - self.lattice / 2

        # Check for logical Z error (accumulated p displacement too large)
        if abs(self.accumulated_p) > self.lattice / 2:
            self.logical_z_errors += 1
            z_error = True
            self.error_history.append(LogicalErrorEvent(
                timestep=self.total_rounds,
                error_type='Z',
                accumulated_q=self.accumulated_q,
                accumulated_p=self.accumulated_p
            ))
            # Wrap accumulated value
            self.accumulated_p = np.mod(self.accumulated_p + self.lattice / 2,
                                         self.lattice) - self.lattice / 2

        return x_error, z_error

    def update_from_qec_result(self, qec_result: Dict) -> Tuple[bool, bool]:
        """
        Update from QEC round result dictionary

        Args:
            qec_result: Output from GKPErrorCorrector.run_qec_round()

        Returns:
            (x_error, z_error)
        """
        error = qec_result['error']
        correction = qec_result['correction']
        return self.update(error[0], error[1], correction[0], correction[1])

    def get_total_logical_errors(self) -> int:
        """Get total number of logical errors"""
        return self.logical_x_errors + self.logical_z_errors

    def get_logical_error_rate(self) -> float:
        """Get logical error rate per round"""
        if self.total_rounds == 0:
            return 0.0
        return self.get_total_logical_errors() / self.total_rounds

    def get_x_error_rate(self) -> float:
        """Get X logical error rate"""
        if self.total_rounds == 0:
            return 0.0
        return self.logical_x_errors / self.total_rounds

    def get_z_error_rate(self) -> float:
        """Get Z logical error rate"""
        if self.total_rounds == 0:
            return 0.0
        return self.logical_z_errors / self.total_rounds

    def get_statistics(self) -> Dict:
        """Get comprehensive statistics"""
        return {
            'total_rounds': self.total_rounds,
            'logical_x_errors': self.logical_x_errors,
            'logical_z_errors': self.logical_z_errors,
            'total_logical_errors': self.get_total_logical_errors(),
            'x_error_rate': self.get_x_error_rate(),
            'z_error_rate': self.get_z_error_rate(),
            'total_error_rate': self.get_logical_error_rate(),
            'accumulated_q': self.accumulated_q,
            'accumulated_p': self.accumulated_p,
        }

    def get_error_times(self) -> List[int]:
        """Get list of timesteps when logical errors occurred"""
        return [event.timestep for event in self.error_history]


class WindowedErrorTracker:
    """
    Tracks logical error rate over sliding windows

    Useful for detecting when performance degrades due to drift.
    """

    def __init__(self, window_size: int = 100):
        """
        Args:
            window_size: Size of sliding window for rate calculation
        """
        self.window_size = window_size
        self.tracker = LogicalErrorTracker()
        self.window_errors: List[int] = []  # Errors in each round (0 or 1)

    def reset(self):
        """Reset all tracking"""
        self.tracker.reset()
        self.window_errors = []

    def update(self,
               syndrome_q: float,
               syndrome_p: float,
               correction_q: float,
               correction_p: float) -> float:
        """
        Update and return windowed error rate

        Returns:
            Current windowed error rate
        """
        x_err, z_err = self.tracker.update(
            syndrome_q, syndrome_p, correction_q, correction_p
        )

        # Record if error occurred this round
        error_occurred = 1 if (x_err or z_err) else 0
        self.window_errors.append(error_occurred)

        # Keep only last window_size rounds
        if len(self.window_errors) > self.window_size:
            self.window_errors.pop(0)

        return self.get_windowed_error_rate()

    def get_windowed_error_rate(self) -> float:
        """Get error rate over current window"""
        if len(self.window_errors) == 0:
            return 0.0
        return sum(self.window_errors) / len(self.window_errors)

    def is_performance_degraded(self, threshold: float = 0.1) -> bool:
        """
        Check if error rate exceeds threshold

        Args:
            threshold: Error rate threshold

        Returns:
            True if current windowed error rate exceeds threshold
        """
        return self.get_windowed_error_rate() > threshold


class ExperimentErrorTracker:
    """
    Tracks errors across an entire experiment with multiple configurations

    Records error rates for different decoder configurations and
    noise conditions.
    """

    def __init__(self):
        self.results: List[Dict] = []
        self.current_tracker: Optional[LogicalErrorTracker] = None

    def start_configuration(self, config_name: str, params: Dict):
        """Start tracking a new configuration"""
        self.current_tracker = LogicalErrorTracker()
        self.current_config = {
            'name': config_name,
            'params': params,
        }

    def update(self, error: np.ndarray, correction: np.ndarray) -> Tuple[bool, bool]:
        """Update current configuration's tracker"""
        if self.current_tracker is None:
            raise RuntimeError("No configuration started")
        return self.current_tracker.update(
            error[0], error[1], correction[0], correction[1]
        )

    def end_configuration(self):
        """End current configuration and save results"""
        if self.current_tracker is not None:
            result = {
                **self.current_config,
                'statistics': self.current_tracker.get_statistics(),
            }
            self.results.append(result)
            self.current_tracker = None

    def get_all_results(self) -> List[Dict]:
        """Get results from all configurations"""
        return self.results

    def get_summary(self) -> Dict:
        """Get summary statistics across all configurations"""
        if not self.results:
            return {}

        error_rates = [r['statistics']['total_error_rate'] for r in self.results]

        return {
            'n_configurations': len(self.results),
            'mean_error_rate': np.mean(error_rates),
            'std_error_rate': np.std(error_rates),
            'min_error_rate': np.min(error_rates),
            'max_error_rate': np.max(error_rates),
            'best_config': self.results[np.argmin(error_rates)]['name'],
            'worst_config': self.results[np.argmax(error_rates)]['name'],
        }


def simulate_error_accumulation(n_rounds: int,
                                 sigma_error: float,
                                 sigma_measurement: float,
                                 gain: float = 1.0,
                                 *,
                                 delta: float = 0.3,
                                 measurement_efficiency: float = 0.95,
                                 ancilla_error_rate: float = 0.01,
                                 add_shot_noise: bool = True,
                                 sigma_error_p: Optional[float] = None,
                                 theta: float = 0.0,
                                 error_bias: Optional[np.ndarray] = None,
                                 use_full_qec_model: bool = True,
                                 return_history: bool = False,
                                 seed: Optional[int] = None) -> Dict:
    """
    Simulate error accumulation over many rounds.

    By default this runs a more realistic closed-loop model:
    1. Residual error from previous round is carried over.
    2. New displacement noise is added.
    3. Syndrome is measured with realistic measurement effects.
    4. Decoder correction is applied.
    5. Logical errors are tracked from accumulated residual displacement.

    Args:
        n_rounds: Number of QEC rounds
        sigma_error: Standard deviation of errors per round
        sigma_measurement: Extra Gaussian readout noise std added to syndrome
        gain: Decoder gain factor
        delta: Finite-energy parameter used by measurement model
        measurement_efficiency: Detector efficiency for measurement model
        ancilla_error_rate: Ancilla error probability
        add_shot_noise: Whether to include measurement shot noise
        sigma_error_p: p-axis displacement noise std (defaults to sigma_error)
        theta: Rotation angle (rad) for anisotropic noise and linear decoder
        error_bias: Mean displacement bias [mu_q, mu_p]
        use_full_qec_model: If False, runs the original simplified model
        return_history: Whether to return per-round history
        seed: Optional random seed for reproducibility

    Returns:
        Simulation statistics
    """
    # 中文注释：该函数是分析入口，可通过 use_full_qec_model 切换严格程度。
    if n_rounds <= 0:
        raise ValueError("n_rounds must be a positive integer")
    if sigma_error < 0 or sigma_measurement < 0:
        raise ValueError("sigma_error and sigma_measurement must be non-negative")
    if sigma_error_p is None:
        sigma_error_p = sigma_error
    if sigma_error_p < 0:
        raise ValueError("sigma_error_p must be non-negative")
    if use_full_qec_model:
        if not (0 < measurement_efficiency <= 1):
            raise ValueError("measurement_efficiency must be in (0, 1]")
        if not (0 <= ancilla_error_rate <= 1):
            raise ValueError("ancilla_error_rate must be in [0, 1]")

    if seed is not None:
        np.random.seed(seed)

    if error_bias is None:
        error_bias_vec = np.zeros(2)
    else:
        error_bias_vec = np.asarray(error_bias, dtype=float)
        if error_bias_vec.shape != (2,):
            raise ValueError("error_bias must have shape (2,)")

    tracker = LogicalErrorTracker()
    history = []

    # Legacy branch kept for backward compatibility.
    if not use_full_qec_model:
        # 中文注释：legacy 分支保留历史行为，便于与旧实验结果做可比对照。
        for _ in range(n_rounds):
            error_q = np.random.normal(0, sigma_error)
            error_p = np.random.normal(0, sigma_error_p)

            syndrome_q = (np.mod(error_q + LATTICE_CONST / 2, LATTICE_CONST) - LATTICE_CONST / 2
                          + np.random.normal(0, sigma_measurement))
            syndrome_p = (np.mod(error_p + LATTICE_CONST / 2, LATTICE_CONST) - LATTICE_CONST / 2
                          + np.random.normal(0, sigma_measurement))

            correction_q = gain * syndrome_q
            correction_p = gain * syndrome_p

            x_err, z_err = tracker.update(error_q, error_p, correction_q, correction_p)

            if return_history:
                history.append({
                    'round': tracker.total_rounds,
                    'new_error': np.array([error_q, error_p]),
                    'syndrome': np.array([syndrome_q, syndrome_p]),
                    'correction': np.array([correction_q, correction_p]),
                    'wrapped_residual': np.array([tracker.accumulated_q, tracker.accumulated_p]),
                    'x_error': x_err,
                    'z_error': z_err,
                })

        stats = tracker.get_statistics()
        stats['model'] = 'simplified'
        if return_history:
            stats['history'] = history
        return stats

    from .error_correction import LinearDecoder
    from .syndrome_measurement import MeasurementConfig, RealisticSyndromeMeasurement

    config = MeasurementConfig(
        delta=delta,
        measurement_efficiency=measurement_efficiency,
        ancilla_error_rate=ancilla_error_rate,
        add_shot_noise=add_shot_noise,
    )
    measurement = RealisticSyndromeMeasurement(config)

    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rotation = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    decoder = LinearDecoder(K=gain * rotation, b=np.zeros(2))

    cumulative_residual = np.zeros(2)

    for _ in range(n_rounds):
        # New round noise can be anisotropic and correlated via rotation.
        base_noise = np.array([
            np.random.normal(0, sigma_error),
            np.random.normal(0, sigma_error_p),
        ])
        new_error = error_bias_vec + rotation @ base_noise

        # Closed-loop QEC: total error = carried residual + new injected noise.
        total_error = cumulative_residual + new_error

        syndrome = measurement.measure(total_error, add_noise=True)
        if sigma_measurement > 0:
            syndrome = syndrome + np.random.normal(0, sigma_measurement, size=2)

        correction = decoder.decode(syndrome)
        residual = total_error - correction
        x_err, z_err = tracker.update(total_error[0], total_error[1], correction[0], correction[1])

        # Keep next round consistent with wrapped logical frame in tracker.
        cumulative_residual = np.array([tracker.accumulated_q, tracker.accumulated_p])

        if return_history:
            history.append({
                'round': tracker.total_rounds,
                'new_error': new_error.copy(),
                'total_error': total_error.copy(),
                'syndrome': syndrome.copy(),
                'correction': correction.copy(),
                'residual': residual.copy(),
                'wrapped_residual': cumulative_residual.copy(),
                'x_error': x_err,
                'z_error': z_err,
            })

    stats = tracker.get_statistics()
    stats['model'] = 'full_qec'
    stats['measurement_sigma_model'] = measurement.sigma_meas
    stats['measurement_sigma_extra'] = sigma_measurement
    if return_history:
        stats['history'] = history
    return stats

"""
### 代码核心功能解析
这段代码是**GKP量子纠错（QEC）系统中逻辑错误的追踪与统计模块**，核心是通过追踪多轮纠错过程中的位移残差累积，判断并统计逻辑错误的发生情况，以下分模块解析：

#### 1. 核心概念与基础类
- **GKP逻辑错误判定规则**：
  GKP编码的逻辑错误由q/p方向的位移累积决定：
  - 累积q位移超过 ±√(2π)/2（即 `LATTICE_CONST/2`）→ 逻辑X错误
  - 累积p位移超过 ±√(2π)/2 → 逻辑Z错误
  `LATTICE_CONST` 是GKP晶格常数，对应 √(2π)。

- **LogicalErrorEvent**：数据类，记录单次逻辑错误的发生时间、类型（X/Z/Y）、错误发生时的q/p累积位移，用于留存错误历史。

- **LogicalErrorTracker**（核心追踪器）：
  - 核心逻辑：`update()` 方法接收「本轮真实误差」和「施加的校正值」，计算**残差（真实误差-校正值）** 并累积；若累积位移超过阈值，判定逻辑错误，同时对累积值做「环绕处理」（避免数值无限增大）。
  - 关键方法：
    - `update()`：单轮误差更新与错误判定
    - `get_logical_error_rate()`：计算每轮平均逻辑错误率
    - `get_statistics()`：返回所有统计指标（总轮数、X/Z错误数、错误率等）

#### 2. 扩展追踪器
- **WindowedErrorTracker**：滑动窗口错误追踪器
  - 基于 `LogicalErrorTracker`，新增「滑动窗口」功能，统计最近N轮的错误率（默认100轮），用于检测性能漂移（如错误率突增）。
  - 核心：`is_performance_degraded()` 对比当前窗口错误率与阈值，判断是否性能退化。

- **ExperimentErrorTracker**：多配置实验追踪器
  - 支持对不同解码器/噪声配置的实验分组追踪，最终输出所有配置的汇总统计（平均/最大/最小错误率、最优/最差配置等）。

#### 3. 核心仿真函数 `simulate_error_accumulation`
这是误差累积的仿真入口，支持两种模型：
- **简化模型（legacy）**：
  每轮独立生成误差，不继承上一轮残差；直接对误差加噪声得到综合征，校正值=增益×综合征，仅用于历史兼容。
- **完整闭环模型（full_qec）**：
  更贴近真实QEC过程，核心逻辑：
  1. 本轮总误差 = 上一轮残差（累积未校正部分） + 本轮新注入噪声
  2. 用真实的综合征测量模型（含探测效率、辅助比特错误、散粒噪声）生成带噪声的综合征
  3. 解码器根据综合征计算校正值，更新残差累积
  4. 判定逻辑错误，并更新下一轮的残差（继承当前环绕后的累积值）

#### 4. 关键逻辑细节
- **残差环绕处理**：当累积位移超过阈值时，执行 `np.mod(accumulated + λ/2, λ) - λ/2`（λ为晶格常数），将累积值拉回 [-λ/2, λ/2] 区间，模拟量子态的周期性。
- **各模块协作**：`simulate_error_accumulation` 调用 `LogicalErrorTracker` 完成错误统计，`WindowedErrorTracker`/`ExperimentErrorTracker` 基于基础追踪器扩展场景化统计能力。

### 总结
1. 核心目标：通过追踪GKP编码中q/p位移的累积残差，判定逻辑X/Z错误，统计多轮纠错的错误率。
2. 核心模型：支持简化（独立轮次）和完整闭环（残差继承）两种误差累积仿真，完整模型更贴近真实QEC过程。
3. 扩展能力：提供滑动窗口监控（性能漂移检测）、多配置实验对比（不同解码器/噪声参数）的工程化能力。

这段代码的本质是**将量子纠错的物理规则（位移累积→逻辑错误）转化为可量化、可仿真的工程实现**，核心价值是输出可对比的逻辑错误率指标，支撑QEC系统的性能评估。
"""
