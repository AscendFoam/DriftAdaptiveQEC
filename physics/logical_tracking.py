"""逻辑错误追踪模块（Logical Error Tracking）。

本模块用于把"每轮残差"转成"逻辑错误率"统计结果，并支持两种误差累积仿真：
- 简化模型（legacy）：每轮独立、不继承残差；
- 完整闭环模型（full_qec）：残差跨轮继承，配合真实测量与线性解码。

GKP 逻辑错误判定规则：
- 累积 q 方向位移越过 ±√(2π)/2 → 逻辑 X 错误；
- 累积 p 方向位移越过 ±√(2π)/2 → 逻辑 Z 错误。
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from .constants import LATTICE_CONST


@dataclass
class LogicalErrorEvent:
    """单次逻辑错误事件记录（数据类）。

    字段:
        timestep: 错误发生的轮次序号。
        error_type: 错误类型 'X' / 'Z' / 'Y'。
        accumulated_q: 错误发生时刻的 q 方向累积位移。
        accumulated_p: 错误发生时刻的 p 方向累积位移。
    """
    timestep: int
    error_type: str  # 'X', 'Z', or 'Y'
    accumulated_q: float
    accumulated_p: float


class LogicalErrorTracker:
    """GKP 纠错中的逻辑错误追踪器。

    GKP 码的逻辑 Pauli 算符：
    - X_L：q 方向位移 √(2π)；
    - Z_L：p 方向位移 √(2π)。

    当累积的未校正位移越过判定边界 ±√(2π)/2 时，即记为一次逻辑错误。
    """

    def __init__(self):
        """初始化追踪器。

        输入:
            无。

        输出:
            无返回值；记录 lattice 并调用 reset 初始化统计状态。
        """
        self.lattice = LATTICE_CONST
        self.reset()

    def reset(self):
        """重置追踪状态。

        输入:
            无。

        输出:
            无返回值；把累积位移、错误计数、轮次计数、错误历史全部清零/清空。
        """
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
        """更新累积位移并判定本轮是否发生逻辑错误。

        功能:
            每个 QEC 轮次：
              1. 测量综合征（误差模晶格的含噪估计）；
              2. 依据综合征施加校正；
              3. 残差 = 真实误差 - 校正，逐轮累积。
            若决策错误（误差实际更靠近另一个晶格点），会额外累积 ±λ/2 位移，
            当累积位移越过 ±λ/2 边界即判定对应方向（q→X、p→Z）的逻辑错误，
            并把累积值 wrap 回 [-λ/2, λ/2] 以避免数值无限增长。

        输入:
            error_q: q 方向真实误差（校正前）。
            error_p: p 方向真实误差（校正前）。
            correction_q: 施加的 q 方向校正量。
            correction_p: 施加的 p 方向校正量。

        输出:
            (x_error, z_error)：本轮是否分别发生 X / Z 逻辑错误。
        """
        # 中文注释：每次 update 都对应一次"真实误差 + 校正后"的逻辑状态推进。
        self.total_rounds += 1

        # 残差 = 真实误差 - 校正量；校正不完美时残差会逐轮累积
        residual_q = error_q - correction_q
        residual_p = error_p - correction_p

        self.accumulated_q += residual_q
        self.accumulated_p += residual_p

        x_error = False
        z_error = False

        # 检查逻辑 X 错误（q 方向累积位移过大）
        if abs(self.accumulated_q) > self.lattice / 2:
            self.logical_x_errors += 1
            x_error = True
            self.error_history.append(LogicalErrorEvent(
                timestep=self.total_rounds,
                error_type='X',
                accumulated_q=self.accumulated_q,
                accumulated_p=self.accumulated_p
            ))
            # wrap 累积值
            self.accumulated_q = np.mod(self.accumulated_q + self.lattice / 2,
                                         self.lattice) - self.lattice / 2

        # 检查逻辑 Z 错误（p 方向累积位移过大）
        if abs(self.accumulated_p) > self.lattice / 2:
            self.logical_z_errors += 1
            z_error = True
            self.error_history.append(LogicalErrorEvent(
                timestep=self.total_rounds,
                error_type='Z',
                accumulated_q=self.accumulated_q,
                accumulated_p=self.accumulated_p
            ))
            # wrap 累积值
            self.accumulated_p = np.mod(self.accumulated_p + self.lattice / 2,
                                         self.lattice) - self.lattice / 2

        return x_error, z_error

    def update_from_qec_result(self, qec_result: Dict) -> Tuple[bool, bool]:
        """从 QEC 单轮结果字典更新追踪状态。

        功能:
            从 ``GKPErrorCorrector.run_qec_round()`` 的返回字典中取出 error 与
            correction，拆分为标量后调用 ``update``。

        输入:
            qec_result: ``run_qec_round()`` 的返回字典，需含 'error' 与 'correction'。

        输出:
            (x_error, z_error)。
        """
        error = qec_result['error']
        correction = qec_result['correction']
        return self.update(error[0], error[1], correction[0], correction[1])

    def get_total_logical_errors(self) -> int:
        """返回累计逻辑错误总数（X + Z）。

        输入:
            无。

        输出:
            logical_x_errors + logical_z_errors。
        """
        return self.logical_x_errors + self.logical_z_errors

    def get_logical_error_rate(self) -> float:
        """返回每轮平均逻辑错误率。

        输入:
            无。

        输出:
            总逻辑错误数 / 总轮数；总轮数为 0 时返回 0.0。
        """
        if self.total_rounds == 0:
            return 0.0
        return self.get_total_logical_errors() / self.total_rounds

    def get_x_error_rate(self) -> float:
        """返回每轮平均 X 逻辑错误率。

        输入:
            无。

        输出:
            logical_x_errors / total_rounds；总轮数为 0 时返回 0.0。
        """
        if self.total_rounds == 0:
            return 0.0
        return self.logical_x_errors / self.total_rounds

    def get_z_error_rate(self) -> float:
        """返回每轮平均 Z 逻辑错误率。

        输入:
            无。

        输出:
            logical_z_errors / total_rounds；总轮数为 0 时返回 0.0。
        """
        if self.total_rounds == 0:
            return 0.0
        return self.logical_z_errors / self.total_rounds

    def get_statistics(self) -> Dict:
        """返回综合统计字典。

        输入:
            无。

        输出:
            字典，含 total_rounds / logical_x_errors / logical_z_errors /
            total_logical_errors / x_error_rate / z_error_rate /
            total_error_rate / accumulated_q / accumulated_p。
        """
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
        """返回所有发生逻辑错误的轮次序号列表。

        输入:
            无。

        输出:
            错误历史中每个事件的 timestep 组成的列表。
        """
        return [event.timestep for event in self.error_history]


class WindowedErrorTracker:
    """滑动窗口逻辑错误率追踪器。

    在 ``LogicalErrorTracker`` 基础上维护一个最近 N 轮的滑动窗口，用于检测
    因漂移导致的性能退化（错误率突增）。
    """

    def __init__(self, window_size: int = 100):
        """初始化滑动窗口追踪器。

        输入:
            window_size: 滑动窗口大小（轮数）。

        输出:
            无返回值；记录 window_size，并内嵌一个 ``LogicalErrorTracker``。
        """
        self.window_size = window_size
        self.tracker = LogicalErrorTracker()
        self.window_errors: List[int] = []  # 每轮是否发生错误（0 或 1）

    def reset(self):
        """重置全部追踪状态。

        输入:
            无。

        输出:
            无返回值；重置内嵌 tracker 与窗口错误列表。
        """
        self.tracker.reset()
        self.window_errors = []

    def update(self,
               syndrome_q: float,
               syndrome_p: float,
               correction_q: float,
               correction_p: float) -> float:
        """更新一轮并返回当前窗口错误率。

        功能:
            先调用内嵌 tracker 的 ``update`` 判定本轮是否发生逻辑错误，
            把"本轮是否出错"（0/1）追加到窗口；窗口超过 window_size 时丢弃最旧一项；
            最后返回当前窗口错误率。

        输入:
            syndrome_q / syndrome_p: 本轮 q/p 误差（作为 tracker 的"误差"输入）。
            correction_q / correction_p: 本轮 q/p 校正量。

        输出:
            当前滑动窗口内的逻辑错误率。
        """
        x_err, z_err = self.tracker.update(
            syndrome_q, syndrome_p, correction_q, correction_p
        )

        # 记录本轮是否出错
        error_occurred = 1 if (x_err or z_err) else 0
        self.window_errors.append(error_occurred)

        # 只保留最近 window_size 轮
        if len(self.window_errors) > self.window_size:
            self.window_errors.pop(0)

        return self.get_windowed_error_rate()

    def get_windowed_error_rate(self) -> float:
        """返回当前窗口内的错误率。

        输入:
            无。

        输出:
            窗口内出错轮数 / 窗口长度；窗口为空时返回 0.0。
        """
        if len(self.window_errors) == 0:
            return 0.0
        return sum(self.window_errors) / len(self.window_errors)

    def is_performance_degraded(self, threshold: float = 0.1) -> bool:
        """判断当前错误率是否超过阈值（即性能退化）。

        输入:
            threshold: 错误率阈值（默认 0.1）。

        输出:
            若当前窗口错误率 > threshold 返回 True，否则 False。
        """
        return self.get_windowed_error_rate() > threshold


class ExperimentErrorTracker:
    """跨多配置实验的错误追踪器。

    记录不同解码器配置 / 噪声条件下的错误率，便于横向对比。
    """

    def __init__(self):
        """初始化实验追踪器。

        输入:
            无。

        输出:
            无返回值；初始化结果列表与"当前配置"追踪器（初始为 None）。
        """
        self.results: List[Dict] = []
        self.current_tracker: Optional[LogicalErrorTracker] = None

    def start_configuration(self, config_name: str, params: Dict):
        """开始追踪一个新的配置。

        功能:
            创建一个新的 ``LogicalErrorTracker``，并记录配置名与参数。

        输入:
            config_name: 配置名称。
            params: 该配置的参数字典。

        输出:
            无返回值；设置 current_tracker 与 current_config。
        """
        self.current_tracker = LogicalErrorTracker()
        self.current_config = {
            'name': config_name,
            'params': params,
        }

    def update(self, error: np.ndarray, correction: np.ndarray) -> Tuple[bool, bool]:
        """更新当前配置的追踪器。

        输入:
            error: 本轮位移误差 [eq, ep]。
            correction: 本轮校正位移 [cq, cp]。

        输出:
            (x_error, z_error)。若尚未 start_configuration 则抛出 RuntimeError。
        """
        if self.current_tracker is None:
            raise RuntimeError("No configuration started")
        return self.current_tracker.update(
            error[0], error[1], correction[0], correction[1]
        )

    def end_configuration(self):
        """结束当前配置并保存其结果。

        功能:
            若存在当前配置，则把配置信息与其统计结果一同存入 results，
            随后清空 current_tracker。

        输入:
            无。

        输出:
            无返回值；把结果追加到 self.results。
        """
        if self.current_tracker is not None:
            result = {
                **self.current_config,
                'statistics': self.current_tracker.get_statistics(),
            }
            self.results.append(result)
            self.current_tracker = None

    def get_all_results(self) -> List[Dict]:
        """返回所有配置的结果列表。

        输入:
            无。

        输出:
            各配置结果组成的列表（每项含 name/params/statistics）。
        """
        return self.results

    def get_summary(self) -> Dict:
        """返回跨所有配置的汇总统计。

        功能:
            汇总各配置的 total_error_rate，给出平均/标准差/最小/最大错误率，
            以及最优/最差配置名。结果为空时返回空字典。

        输入:
            无。

        输出:
            汇总字典；无结果时为 {}。
        """
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
    """仿真多轮误差累积，输出逻辑错误统计。

    功能:
        该函数是误差累积分析入口，默认运行更真实的闭环模型：
          1. 上一轮残差跨轮继承；
          2. 注入新的位移噪声（可各向异性、可通过旋转矩阵 θ 引入相关性）；
          3. 用真实测量模型得到含噪综合征（可选再叠加额外读出噪声）；
          4. 线性解码器计算校正，更新残差与逻辑错误统计。
        当 ``use_full_qec_model=False`` 时，回退到原始简化模型（每轮独立、
        标量增益、简单加噪），用于与历史实验结果做可比对照。

    输入:
        n_rounds: QEC 轮数（必须为正整数）。
        sigma_error: 每轮位移噪声标准差（q 方向，非负）。
        sigma_measurement: 额外叠加到综合征的高斯读出噪声标准差（非负）。
        gain: 解码增益。
        delta: 测量模型使用的有限能量参数。
        measurement_efficiency: 测量模型探测器效率（full_qec 模式需在 (0,1]）。
        ancilla_error_rate: 辅助比特错误率（full_qec 模式需在 [0,1]）。
        add_shot_noise: 是否包含测量散粒噪声。
        sigma_error_p: p 方向位移噪声标准差（默认等于 sigma_error，非负）。
        theta: 各向异性噪声与线性解码的旋转角（弧度）。
        error_bias: 平均位移偏置 [mu_q, mu_p]（形状必须为 (2,)）。
        use_full_qec_model: False 时运行原始简化模型。
        return_history: 是否返回逐轮历史。
        seed: 可选随机种子，用于可复现性。

    输出:
        仿真统计字典。简化模型附 'model'='simplified'；完整模型附
        'model'='full_qec'、'measurement_sigma_model'、'measurement_sigma_extra'；
        return_history=True 时额外含逐轮 'history'。
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

    # 保留 legacy 分支用于向后兼容。
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
        # 本轮新噪声可通过旋转矩阵实现各向异性与相关性
        base_noise = np.array([
            np.random.normal(0, sigma_error),
            np.random.normal(0, sigma_error_p),
        ])
        new_error = error_bias_vec + rotation @ base_noise

        # 闭环 QEC：总误差 = 继承残差 + 本轮新注入噪声
        total_error = cumulative_residual + new_error

        syndrome = measurement.measure(total_error, add_noise=True)
        if sigma_measurement > 0:
            syndrome = syndrome + np.random.normal(0, sigma_measurement, size=2)

        correction = decoder.decode(syndrome)
        residual = total_error - correction
        x_err, z_err = tracker.update(total_error[0], total_error[1], correction[0], correction[1])

        # 下一轮与 tracker 中 wrap 后的逻辑坐标系保持一致
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
