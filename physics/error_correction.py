"""纠错模块（Error Correction）。

本模块实现 GKP 纠错的核心流程：综合征测量、线性解码、残差统计与多轮仿真。
上层实验通常通过 ``GKPErrorCorrector``（单轮纠错）与 ``QECSimulator``（多轮仿真，
支持漂移场景与周期重标定）访问本模块能力。
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass, field

from .constants import LATTICE_CONST
from .gkp_state import ApproximateGKPState
from .syndrome_measurement import RealisticSyndromeMeasurement, MeasurementConfig
from .noise_channels import CombinedNoiseModel


@dataclass
class DecoderParameters:
    """线性解码器参数容器（数据类）。

    字段:
        K: 2×2 增益矩阵（控制纠错强度与各向异性旋转）。
        b: 2 维偏置向量（补偿系统位移偏置）。
    """
    K: np.ndarray = field(default_factory=lambda: np.eye(2))  # Gain matrix
    b: np.ndarray = field(default_factory=lambda: np.zeros(2))  # Bias vector

    def to_flat(self) -> np.ndarray:
        """把参数展平为一维数组。

        功能:
            按 [K_11, K_12, K_21, K_22, b_1, b_2] 顺序拼接成 6 维向量，
            便于作为优化器 / 网络的输入输出。

        输入:
            无。

        输出:
            长度 6 的一维 ndarray。
        """
        return np.concatenate([self.K.flatten(), self.b])

    @classmethod
    def from_flat(cls, params: np.ndarray) -> 'DecoderParameters':
        """由一维数组还原解码器参数。

        输入:
            params: 长度 6 的一维数组，前 4 个为 K 的元素（按行优先），后 2 个为 b。

        输出:
            重建的 ``DecoderParameters``。
        """
        K = params[:4].reshape(2, 2)
        b = params[4:]
        return cls(K=K, b=b)


class LinearDecoder:
    """参数化线性解码器。

    给定综合征 s = [sq, sp]，计算校正位移：

        Δ = K @ s + b

    其中 K 为 2×2 增益矩阵、b 为偏置向量。
    - 高斯噪声下方差最优时 K 接近单位阵（增益≈1）；
    - 测量有噪声时 K < I 以补偿测量噪声；
    - b 用于校正系统偏置（如控制漂移）。
    """

    def __init__(self,
                 K: Optional[np.ndarray] = None,
                 b: Optional[np.ndarray] = None):
        """初始化线性解码器。

        输入:
            K: 2×2 增益矩阵（默认单位阵）。
            b: 2 维偏置向量（默认零向量）。

        输出:
            无返回值；记录 K、b 与 lattice。
        """
        self.K = K if K is not None else np.eye(2)
        self.b = b if b is not None else np.zeros(2)
        self.lattice = LATTICE_CONST

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """由综合征计算校正位移。

        功能:
            线性解码核心公式 Δ = K @ s + b。

        输入:
            syndrome: 测得的综合征 [sq, sp]。

        输出:
            校正位移 [dq, dp]。
        """
        # 中文注释：线性解码核心公式 Δ = K @ s + b。
        return self.K @ syndrome + self.b

    def update(self, K: np.ndarray, b: np.ndarray):
        """更新解码器参数。

        输入:
            K: 新的 2×2 增益矩阵。
            b: 新的 2 维偏置向量。

        输出:
            无返回值；就地更新 self.K、self.b。
        """
        self.K = K
        self.b = b

    def update_from_flat(self, params: np.ndarray):
        """由一维数组更新解码器参数。

        输入:
            params: 长度 6 的一维数组（前 4 个为 K 行优先，后 2 个为 b）。

        输出:
            无返回值；就地更新 self.K、self.b。
        """
        self.K = params[:4].reshape(2, 2)
        self.b = params[4:]

    def get_params(self) -> DecoderParameters:
        """以数据类形式返回当前参数（拷贝）。

        输入:
            无。

        输出:
            ``DecoderParameters``，包含当前 K、b 的拷贝。
        """
        return DecoderParameters(K=self.K.copy(), b=self.b.copy())

    def get_flat_params(self) -> np.ndarray:
        """以一维数组形式返回当前参数。

        输入:
            无。

        输出:
            长度 6 的一维 ndarray（K 行优先 + b）。
        """
        return np.concatenate([self.K.flatten(), self.b])


def compute_optimal_decoder_params(sigma: float,
                                    delta: float,
                                    theta: float = 0.0,
                                    meas_efficiency: float = 0.95) -> DecoderParameters:
    """针对给定噪声计算近似最优的线性解码器参数。

    功能:
        采用近似 Wiener 滤波思路估计最优增益，并叠加相位漂移旋转矩阵：
          - 信号方差 var_signal = (λ/2)²/3（晶格区间内均匀分布）；
          - 噪声方差 var_noise = σ²(位移) + Δ²(GKP 有限能量) + (1-η)/(2η)(测量)；
          - 最优增益 gain = var_signal / (var_signal + var_noise)；
          - 旋转矩阵 R(θ) 表征相位漂移；最终 K = gain·R，偏置 b 默认为零。

    输入:
        sigma: 位移噪声标准差。
        delta: GKP 有限能量参数。
        theta: 相空间旋转角（弧度），表征各向异性/相位漂移。
        meas_efficiency: 测量效率 η。

    输出:
        ``DecoderParameters``（K=gain·R，b=0）。
    """
    # 中文注释：使用近似 Wiener 思路估计最优线性增益与旋转矩阵。
    # 总等效噪声方差，含：GKP 有限能量、位移噪声、测量噪声
    var_signal = (LATTICE_CONST / 2) ** 2 / 3  # 均匀信号方差

    # 各噪声贡献
    var_displacement = sigma ** 2
    var_gkp = delta ** 2
    var_meas = (1 - meas_efficiency) / (2 * meas_efficiency)

    var_noise = var_displacement + var_gkp + var_meas

    # 最优 Wiener 滤波增益
    gain = var_signal / (var_signal + var_noise)

    # 相位漂移的旋转矩阵
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

    # 合成增益矩阵
    K = gain * R

    # 偏置（无系统漂移时通常为零）
    b = np.zeros(2)

    return DecoderParameters(K=K, b=b)


class GKPErrorCorrector:
    """完整的单轮 GKP 纠错系统。

    组合三件事：
    - 综合征测量（带噪声）；
    - 线性（参数化）解码；
    - 校正施加与残差判定。
    """

    def __init__(self,
                 delta: float = 0.3,
                 decoder: Optional[LinearDecoder] = None,
                 measurement_config: Optional[MeasurementConfig] = None):
        """初始化单轮纠错器。

        功能:
            构造真实测量器；若未提供解码器，则用当前噪声水平
            （默认 sigma=0.3）计算近似最优解码器参数并构造线性解码器。

        输入:
            delta: GKP 有限能量参数。
            decoder: 可选的线性解码器（None 时自动创建）。
            measurement_config: 测量配置（None 时按 delta 构造默认配置）。

        输出:
            无返回值；记录 delta / lattice，并组装 measurement 与 decoder。
        """
        self.delta = delta
        self.lattice = LATTICE_CONST

        # 搭建测量器
        if measurement_config is None:
            measurement_config = MeasurementConfig(delta=delta)
        self.measurement = RealisticSyndromeMeasurement(measurement_config)

        # 搭建解码器
        if decoder is None:
            # 为当前噪声水平创建近似最优解码器
            params = compute_optimal_decoder_params(sigma=0.3, delta=delta)
            decoder = LinearDecoder(K=params.K, b=params.b)
        self.decoder = decoder

    def run_qec_round(self,
                      error: np.ndarray,
                      add_measurement_noise: bool = True) -> Dict[str, Any]:
        """执行一轮量子纠错。

        功能:
            1. 测量综合征（可选加噪）；
            2. 解码得到校正位移；
            3. 应用校正并跟踪残差（残差 = 误差 - 校正）；
            4. 判定纠错是否成功：残差是否落在基本晶胞 [-λ/2, λ/2] 范围内。

        输入:
            error: 待纠正的位移误差 [eq, ep]。
            add_measurement_noise: 测量是否加噪。

        输出:
            字典，含 syndrome / correction / residual / success / error。
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
        """评估解码器性能。

        功能:
            采样 n_samples 个高斯位移误差，逐个执行单轮纠错，统计成功率、
            逻辑错误率以及残差的均值/标准差。

        输入:
            n_samples: 误差样本数。
            error_sigma: 测试误差的标准差。

        输出:
            字典，含 logical_error_rate / success_rate / mean_residual_q /
            mean_residual_p / std_residual_q / std_residual_p / n_samples。
        """
        successes = 0
        residuals = []

        for _ in range(n_samples):
            # 随机误差
            error = np.random.normal(0, error_sigma, size=2)

            # 执行纠错
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
        """更新解码器参数。

        输入:
            K: 新的 2×2 增益矩阵。
            b: 新的 2 维偏置向量。

        输出:
            无返回值；委托给内部解码器的 update。
        """
        self.decoder.update(K, b)

    def update_decoder_from_fno(self, fno_output: np.ndarray):
        """由 FNO 网络输出更新解码器。

        功能:
            把 FNO 网络输出的扁平数组解析为解码器参数（前 4 个为 K 行优先，
            后 2 个为 b）并更新。

        输入:
            fno_output: FNO 网络输出的一维数组（长度 6）。

        输出:
            无返回值；委托给内部解码器的 update_from_flat。
        """
        self.decoder.update_from_flat(fno_output)


class QECSimulator:
    """多轮 QEC 仿真器。

    仿真流程：态制备 -> 噪声施加 -> 纠错 -> 重复多轮。
    支持噪声参数随时间漂移的场景，以及周期性重标定解码器。
    """

    def __init__(self,
                 delta: float = 0.3,
                 noise_model: Optional[CombinedNoiseModel] = None,
                 corrector: Optional[GKPErrorCorrector] = None):
        """初始化多轮仿真器。

        功能:
            若未提供噪声模型与纠错器，则使用默认实现（gamma=0.05、n_bar=0.01、
            sigma_disp=0.1 的组合噪声模型；按 delta 构造的纠错器）。

        输入:
            delta: GKP 有限能量参数。
            noise_model: 噪声模型（None 时用默认）。
            corrector: 纠错器（None 时用默认）。

        输出:
            无返回值；记录 delta / lattice / noise_model / corrector。
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
        """仿真多轮 QEC。

        功能:
            每轮：注入新噪声 -> 累加到上一轮残差得到总误差 -> 执行单轮纠错 ->
            用本轮残差更新累计误差。统计多轮总成功率与逻辑错误率。

        输入:
            n_rounds: QEC 轮数。
            error_sigma: 每轮新注入误差的标准差。

        输出:
            字典，含 n_rounds / successes / logical_error_rate / round_results。
        """
        results = []
        cumulative_error = np.zeros(2)

        for round_idx in range(n_rounds):
            # 本轮新误差
            new_error = np.random.normal(0, error_sigma, size=2)

            # 总误差 = 累计残差 + 新误差
            total_error = cumulative_error + new_error

            # 执行纠错
            round_result = self.corrector.run_qec_round(total_error)
            results.append(round_result)

            # 用残差更新累计误差
            cumulative_error = round_result['residual']

        # 分析结果
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
        """在噪声参数漂移场景下运行 QEC。

        功能:
            每个时间步：由 ``drift_model(t)`` 取当前 (sigma, delta, theta)，
            评估当前噪声下的逻辑错误率；每隔 ``recalibrate_every`` 步用
            ``compute_optimal_decoder_params`` 重标定解码器（真实系统中此处可用 FNO）。

        输入:
            n_timesteps: 时间步数。
            drift_model: 可调用对象，t -> (sigma, delta, theta)。
            recalibrate_every: 重标定间隔（步）。

        输出:
            字典，含 n_timesteps / error_rates（每步逻辑错误率数组）/
            mean_error_rate（平均错误率）。
        """
        error_rates = []

        for t in range(n_timesteps):
            # 当前噪声参数
            sigma_t, delta_t, theta_t = drift_model(t)

            # 在当前噪声水平下评估
            metrics = self.corrector.evaluate_performance(
                n_samples=1000,
                error_sigma=sigma_t
            )
            error_rates.append(metrics['logical_error_rate'])

            # 周期性重标定（真实系统中此处会用 FNO）
            if t % recalibrate_every == 0 and t > 0:
                # 按当前噪声更新解码器
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
