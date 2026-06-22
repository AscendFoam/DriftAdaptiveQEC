"""综合征测量模块（Syndrome Measurement）。

本模块模拟 GKP 码的综合征测量过程，从理想的取模测量到带真实物理噪声的测量。
核心思想：位移误差对晶格常数取模即得到综合征，再据此施加反向位移完成纠错。

- ``SyndromeMeasurement``：理想（无噪声）取模测量；
- ``RealisticSyndromeMeasurement``：在理想测量基础上叠加有限压缩噪声、探测效率
  损失、散粒噪声（shot noise）与辅助比特（ancilla）错误，是纠错主流程的默认测量后端；
- ``AdaptiveSyndromeMeasurement``：基于噪声水平自适应调整纠错增益。
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

from .constants import LATTICE_CONST


@dataclass
class MeasurementConfig:
    """综合征测量参数容器（数据类）。

    字段:
        delta: GKP 态有限能量参数，决定有限压缩导致的测量噪声（方差 ∝ Δ²）。
        measurement_efficiency: 探测器效率 η，越低则测量噪声越大。
        ancilla_error_rate: 辅助比特错误率（测量中发生半晶格偏移的概率）。
        add_shot_noise: 是否加入探测器散粒噪声（默认均值 0、std 0.1）。
    """
    delta: float = 0.3  # GKP state finite energy parameter
    measurement_efficiency: float = 0.95  # Detector efficiency
    ancilla_error_rate: float = 0.01  # Probability of ancilla error
    add_shot_noise: bool = True  # Include shot noise


class SyndromeMeasurement:
    """GKP 码的基础（理想）综合征测量。

    综合征即位移误差对晶格常数取模的结果。对位移误差 (e_q, e_p)：

        s_q = e_q mod √(2π)，映射到 [-√(2π)/2, √(2π)/2]
        s_p = e_p mod √(2π)，映射到 [-√(2π)/2, √(2π)/2]
    """

    def __init__(self, lattice: float = LATTICE_CONST):
        """初始化理想测量器。

        输入:
            lattice: 晶格常数（默认 √(2π)）。

        输出:
            无返回值；记录 lattice。
        """
        self.lattice = lattice

    def measure(self, displacement: np.ndarray) -> np.ndarray:
        """理想综合征测量（无噪声）。

        功能:
            把位移误差对晶格常数取模并映射到 [-lattice/2, lattice/2] 区间。

        输入:
            displacement: 位移误差 [dq, dp]。

        输出:
            综合征 [sq, sp]，落在 [-lattice/2, lattice/2]。
        """
        # 映射到 [-lattice/2, lattice/2]
        syndrome = np.mod(displacement + self.lattice / 2, self.lattice) - self.lattice / 2
        return syndrome

    def get_correction(self, syndrome: np.ndarray) -> np.ndarray:
        """由综合征得到校正位移。

        功能:
            理想测量下校正量等于 -syndrome（直接抵消误差）。

        输入:
            syndrome: 测得的综合征 [sq, sp]。

        输出:
            校正位移 [dq, dp] = -syndrome。
        """
        return -syndrome


class RealisticSyndromeMeasurement:
    """带真实物理噪声的综合征测量。

    真实 GKP 纠错的测量流程：
      1. 制备一个辅助 GKP 态；
      2. 用 SUM 门耦合数据态与辅助态；
      3. 对辅助态做 homodyne 探测；
      4. 测量带有有限压缩导致的噪声。

    测量噪声随 GKP delta 参数变化：σ_meas ∝ Δ。
    本类还可接收可选的确定性 RNG（``rng``），供需要可复现的 recovery / runtime 路径使用。
    """

    def __init__(
        self,
        config: Optional[MeasurementConfig] = None,
        *,
        rng: Optional[np.random.Generator] = None,
    ):
        """初始化真实测量器。

        功能:
            记录测量配置与晶格常数，保存可选的确定性随机数发生器（RNG），
            并据此计算测量噪声标准差 sigma_meas。

        输入:
            config: ``MeasurementConfig`` 测量配置（为 None 时用默认配置）。
            rng: 可选的确定性 ``np.random.Generator``，供需要可复现结果的路径使用；
                 为 None 时退化为全局 ``np.random``。

        输出:
            无返回值；设置 config / lattice / sigma_meas / _rng。
        """
        self.config = config or MeasurementConfig()
        self.lattice = LATTICE_CONST
        self._rng = rng

        # 计算测量噪声方差
        self._compute_noise_variance()

    def _normal(self, mean: float, std: float, size: Optional[int | tuple[int, ...]] = None):
        """从高斯分布采样。

        功能:
            若提供了确定性 RNG 则用之，否则用全局 ``np.random.normal``。

        输入:
            mean: 均值。
            std: 标准差。
            size: 采样形状（None 表示标量）。

        输出:
            高斯采样结果（标量或 ndarray）。
        """
        if self._rng is None:
            return np.random.normal(mean, std, size=size)
        return self._rng.normal(mean, std, size=size)

    def _random(self) -> float:
        """从 [0,1) 均匀分布采样一个标量。

        输入:
            无。

        输出:
            [0,1) 区间的随机浮点数。优先用确定性 RNG，否则用全局 ``np.random.random``。
        """
        if self._rng is None:
            return float(np.random.random())
        return float(self._rng.random())

    def _random_sign(self) -> float:
        """随机返回 +1 或 -1。

        功能:
            采样一个标准正态值，按其正负号返回 +1 / -1（>=0 取 +1，<0 取 -1）。

        输入:
            无。

        输出:
            +1.0 或 -1.0。
        """
        sample = float(self._normal(0.0, 1.0))
        return 1.0 if sample >= 0.0 else -1.0

    def _compute_noise_variance(self):
        """由配置计算测量噪声标准差 sigma_meas。

        功能:
            把有限压缩与探测效率损失统一折算为测量噪声：
              - 有限压缩贡献方差 var_squeezing = Δ²；
              - 探测效率损失贡献真空噪声 var_inefficiency = (1-η)/(2η)（η>0 时）；
              - 合成 sigma_meas = √(var_squeezing + var_inefficiency)。

        输入:
            无（参数取自 self.config）。

        输出:
            无返回值；写入 self.sigma_meas。
        """
        # 中文注释：将有限能量与探测效率损失统一折算为测量方差。
        delta = self.config.delta
        eta = self.config.measurement_efficiency

        # 有限压缩贡献测量噪声，方差正比于 Δ²
        var_squeezing = delta ** 2

        # 探测效率损失贡献真空噪声：(1-η)/η × 散粒噪声方差(=1/2)
        var_inefficiency = (1 - eta) / (2 * eta) if eta > 0 else 1.0

        # 总测量方差
        self.sigma_meas = np.sqrt(var_squeezing + var_inefficiency)

    def measure(self,
                true_displacement: np.ndarray,
                add_noise: bool = True) -> np.ndarray:
        """模拟真实（带噪声）的综合征测量。

        功能:
            先计算理想综合征，再依次叠加测量噪声（sigma_meas）、散粒噪声
            （可选，默认均值 0、std 0.1），以及以 ``ancilla_error_rate`` 概率
            发生的辅助比特错误（在 q 或 p 方向随机偏移半个晶格）。

        输入:
            true_displacement: 真实位移误差 [dq, dp]。
            add_noise: 是否叠加噪声；为 False 时直接返回理想综合征。

        输出:
            测得（带噪声）的综合征 [sq, sp]。
        """
        # 中文注释：先得到理想综合征，再叠加测量噪声、shot noise 和 ancilla 扰动。
        # 理想综合征
        syndrome_ideal = np.mod(true_displacement + self.lattice / 2,
                                self.lattice) - self.lattice / 2

        if not add_noise:
            return syndrome_ideal

        # 测量噪声
        measurement_noise = self._normal(0, self.sigma_meas, size=2)

        # 散粒噪声（探测器噪声）
        if self.config.add_shot_noise:
            shot_noise = self._normal(0, 0.1, size=2)
        else:
            shot_noise = 0

        syndrome_noisy = syndrome_ideal + measurement_noise + shot_noise

        # 辅助比特错误（测量中偶发的比特翻转）
        if self._random() < self.config.ancilla_error_rate:
            # 随机偏移半个晶格
            if self._random() > 0.5:
                syndrome_noisy[0] += self.lattice / 2 * self._random_sign()
            else:
                syndrome_noisy[1] += self.lattice / 2 * self._random_sign()

        return syndrome_noisy

    def get_correction(self,
                       syndrome: np.ndarray,
                       gain: float = 1.0) -> np.ndarray:
        """由综合征得到校正位移（带增益）。

        输入:
            syndrome: 测得的综合征 [sq, sp]。
            gain: 校正增益（噪声测量下 <1 以避免过补偿）。

        输出:
            校正位移 [dq, dp] = -gain * syndrome。
        """
        return -gain * syndrome

    def get_optimal_gain(self) -> float:
        """计算最优校正增益（Wiener 滤波思路）。

        功能:
            对带噪声测量，最优增益为
                g* = σ_signal² / (σ_signal² + σ_noise²)
            其中 σ_signal² 为真实综合征的方差（晶格区间内均匀分布，方差 = (λ/2)²/3），
            σ_noise² 为测量噪声方差。

        输入:
            无。

        输出:
            最优增益 g*（float）。
        """
        # 信号方差：[-λ/2, λ/2] 区间均匀分布
        var_signal = (self.lattice / 2) ** 2 / 3

        # 噪声方差
        var_noise = self.sigma_meas ** 2

        return var_signal / (var_signal + var_noise)

    def get_measurement_covariance(self) -> np.ndarray:
        """返回测量噪声的协方差矩阵。

        功能:
            以 sigma_meas² 为对角元；若开启散粒噪声，则额外加上散粒噪声方差 0.01。
            假设 q、p 方向噪声独立，故为对角阵。

        输入:
            无。

        输出:
            shape=(2,2) 的对角协方差矩阵。
        """
        var = self.sigma_meas ** 2
        if self.config.add_shot_noise:
            var += 0.01  # 散粒噪声方差
        return np.diag([var, var])

    def update_delta(self, delta: float):
        """更新 GKP 的 delta 参数并重算测量噪声。

        输入:
            delta: 新的有限能量参数。

        输出:
            无返回值；更新 self.config.delta 并重算 self.sigma_meas。
        """
        self.config.delta = delta
        self._compute_noise_variance()


class AdaptiveSyndromeMeasurement(RealisticSyndromeMeasurement):
    """带自适应增益的综合征测量。

    在 ``RealisticSyndromeMeasurement`` 基础上，依据估计噪声水平动态调整纠错增益。
    """

    def __init__(self, config: Optional[MeasurementConfig] = None):
        """初始化自适应测量器。

        功能:
            复用父类初始化，并用当前配置计算最优增益作为初始增益。

        输入:
            config: 测量配置（为 None 时用默认配置）。

        输出:
            无返回值；额外设置 self.gain 为初始最优增益。
        """
        super().__init__(config)
        self.gain = self.get_optimal_gain()

    def measure_and_correct(self,
                            true_displacement: np.ndarray,
                            add_noise: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """一次性完成"测量综合征 + 计算校正"。

        输入:
            true_displacement: 真实位移误差 [dq, dp]。
            add_noise: 测量是否加噪。

        输出:
            (syndrome, correction)：测得的综合征、以及用当前自适应增益算出的校正位移。
        """
        syndrome = self.measure(true_displacement, add_noise)
        correction = self.get_correction(syndrome, self.gain)
        return syndrome, correction

    def update_gain(self, new_gain: float):
        """更新纠错增益（带安全边界）。

        功能:
            把增益裁剪到 [0.1, 1.5] 区间，避免极端增益导致发散。

        输入:
            new_gain: 期望的新增益值。

        输出:
            无返回值；写入裁剪后的 self.gain。
        """
        self.gain = np.clip(new_gain, 0.1, 1.5)  # 安全边界

    def adapt_to_noise(self, estimated_sigma: float):
        """根据估计的总噪声水平自适应调整增益。

        功能:
            用估计的噪声标准差更新等效测量噪声
                sigma_meas = √(Δ² + estimated_sigma²)，
            然后重算最优增益。

        输入:
            estimated_sigma: 估计的总噪声标准差。

        输出:
            无返回值；更新 self.sigma_meas 与 self.gain。
        """
        # 更新等效测量噪声
        self.sigma_meas = np.sqrt(self.config.delta**2 + estimated_sigma**2)
        self.gain = self.get_optimal_gain()


def simulate_measurement_statistics(n_samples: int = 10000,
                                    true_sigma: float = 0.3,
                                    delta: float = 0.3) -> dict:
    """仿真并统计大量测量样本，用于分析测量噪声影响。

    功能:
        用给定 true_sigma 生成随机位移误差，经真实测量得到带噪声综合征，
        统计其 q/p 均值、标准差、相关性，并附带该配置下的测量噪声 sigma 与最优增益。

    输入:
        n_samples: 采样次数。
        true_sigma: 真实位移误差的标准差。
        delta: GKP 有限能量参数（用于构造测量配置）。

    输出:
        字典，包含 mean_q / mean_p / std_q / std_p / correlation /
        measurement_sigma / optimal_gain 等统计量。
    """
    config = MeasurementConfig(delta=delta)
    measurement = RealisticSyndromeMeasurement(config)

    syndromes = []
    for _ in range(n_samples):
        # 随机位移误差
        error = np.random.normal(0, true_sigma, size=2)
        syndrome = measurement.measure(error, add_noise=True)
        syndromes.append(syndrome)

    syndromes = np.array(syndromes)

    return {
        'mean_q': np.mean(syndromes[:, 0]),
        'mean_p': np.mean(syndromes[:, 1]),
        'std_q': np.std(syndromes[:, 0]),
        'std_p': np.std(syndromes[:, 1]),
        'correlation': np.corrcoef(syndromes[:, 0], syndromes[:, 1])[0, 1],
        'measurement_sigma': measurement.sigma_meas,
        'optimal_gain': measurement.get_optimal_gain(),
    }

"""
### 代码核心功能解析
这段代码是**GKP量子纠错码**中「综合征测量（Syndrome Measurement）」的仿真模块，核心是模拟从「理想无噪声测量」到「带真实物理噪声的测量」全过程，为后续量子纠错流程提供测量数据。

#### 1. 核心概念与类的分工
- **GKP码**：一种用于连续变量量子纠错的编码方案，通过测量「位移误差模晶格常数」得到的**综合征（syndrome）** 来判断量子比特的误差，进而执行纠错。
- **核心类的作用**：
  | 类名 | 核心功能 |
  |---|---|
  | `MeasurementConfig` | 数据类，存储测量的噪声参数（压缩噪声、测量效率、辅助比特误差率等） |
  | `SyndromeMeasurement` | 理想测量：仅计算位移误差模晶格常数，无任何噪声 |
  | `RealisticSyndromeMeasurement` | 真实测量：在理想测量基础上叠加三类噪声（有限压缩噪声、测量低效噪声、辅助比特误差），是纠错主流程的默认后端 |
  | `AdaptiveSyndromeMeasurement` | 自适应测量：基于噪声水平动态调整纠错增益，优化纠错效果 |

#### 2. 关键逻辑拆解
##### （1）理想综合征计算
GKP码的晶格常数为 `√(2π)`（`LATTICE_CONST`），理想综合征的核心公式：
```python
syndrome = np.mod(displacement + lattice/2, lattice) - lattice/2
```
作用：将位移误差 `[dq, dp]` 映射到 `[-晶格/2, 晶格/2]` 区间，得到无噪声的综合征 `[sq, sp]`，纠错时直接取 `-syndrome` 即可抵消误差。

##### （2）真实测量的噪声叠加
`RealisticSyndromeMeasurement.measure()` 是核心方法，噪声叠加逻辑：
1. **有限压缩噪声**：GKP态并非理想无限压缩，用 `delta` 参数表征，噪声方差为 `delta²`；
2. **测量低效噪声**：探测器效率 `measurement_efficiency` 不足，引入真空噪声，方差为 `(1-η)/(2η)`；
3. **散粒噪声（Shot Noise）**：探测器固有噪声，默认添加均值0、标准差0.1的高斯噪声；
4. **辅助比特误差**：以 `ancilla_error_rate` 概率随机翻转综合征（偏移半个晶格）。

##### （3）自适应纠错增益
真实测量存在噪声，直接用 `-syndrome` 纠错会过校正，因此计算**最优增益**：
```python
g* = 信号方差 / (信号方差 + 噪声方差)
```
其中信号方差由晶格区间的均匀分布决定（`(lattice/2)²/3`），噪声方差为测量总噪声的平方。

#### 3. 辅助功能
- `simulate_measurement_statistics`：生成大量测量样本，统计综合征的均值、标准差、相关性等，用于分析测量噪声的影响；
- `get_measurement_covariance`：返回测量噪声的协方差矩阵，用于后续纠错算法的优化。

### 总结
1. 核心目标：仿真GKP码的综合征测量，从「理想模型」到「带物理噪声的真实模型」，为量子纠错提供输入；
2. 关键噪声：有限压缩噪声、测量低效噪声、散粒噪声、辅助比特误差；
3. 优化手段：通过「自适应增益」平衡噪声与信号，避免过校正/欠校正，提升纠错效果。

这段代码是GKP量子纠错流程中「测量环节」的基础模块，输出的带噪声综合征会直接用于后续的误差校正决策。
"""
