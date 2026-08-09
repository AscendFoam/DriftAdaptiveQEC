"""量子噪声通道模块（Quantum Noise Channels）。

本文件提供用于玻色子系统（如超导腔）的多种噪声通道建模，并把不同噪声源
统一折算到可仿真的 Wigner 函数演化上。包含：
- 光子损失（amplitude damping，超导腔主导噪声）；
- 热噪声；
- 位移噪声（控制不完美）；
- 相位噪声 / 退相位（dephasing）。

这些通道作用于 Wigner 函数表示，组合模型可输出等效位移噪声 σ，供上层快速
估计与参数调度使用（不替代完整仿真）。
"""

import numpy as np
from typing import Optional, Tuple, Union
from scipy.ndimage import gaussian_filter
from dataclasses import dataclass

from .constants import LATTICE_CONST
from .gkp_state import ApproximateGKPState

# 尝试导入 Strawberry Fields（部分精确噪声通道依赖）
try:
    import strawberryfields as sf
    from strawberryfields.ops import LossChannel, ThermalLossChannel, Dgate, Rgate
    HAS_STRAWBERRYFIELDS = True
except ImportError:
    HAS_STRAWBERRYFIELDS = False


@dataclass
class NoiseParameters:
    """噪声通道参数容器（数据类）。

    字段:
        gamma: 光子损失率（无量纲，γ=κt）。
        n_bar: 热平均光子数。
        sigma_displacement: 位移噪声标准差（控制不完美）。
        sigma_phase: 相位噪声标准差。
    """
    gamma: float = 0.05  # Photon loss rate
    n_bar: float = 0.01  # Thermal photon number
    sigma_displacement: float = 0.1  # Displacement noise std
    sigma_phase: float = 0.01  # Phase noise std


class QuantumNoiseChannel:
    """组合量子噪声通道（作用于 GKP 态的 Wigner 函数）。

    依次施加多种噪声源：
      1. 光子损失（超导腔主导噪声）；
      2. 热噪声；
      3. 位移噪声（控制不完美）；
      4. 相位噪声 / 退相位。

    为提升效率，各噪声在 Wigner 函数上采用解析近似实现（缩放、高斯卷积、旋转平均）。
    """

    def __init__(self, cutoff: int = 50, use_sf: bool = True):
        """初始化组合噪声通道。

        输入:
            cutoff: Fock 截断维度（仅影响可能的 SF 路径，本类默认走解析近似）。
            use_sf: 是否优先使用 SF（实际启用还取决于 SF 是否可用）。

        输出:
            无返回值；记录 cutoff 与 use_sf（已与 SF 可用性取交集）。
        """
        self.cutoff = cutoff
        self.use_sf = use_sf and HAS_STRAWBERRYFIELDS

    def apply_all(self,
                  wigner: np.ndarray,
                  params: NoiseParameters,
                  grid_range: Tuple[float, float] = (-6, 6)) -> np.ndarray:
        """依次施加全部噪声通道到 Wigner 函数上。

        功能:
            按"光子损失 -> 热噪声 -> 位移噪声 -> 相位噪声"的顺序叠加噪声影响。
            当某噪声参数可忽略时其内部会跳过计算。

        输入:
            wigner: 输入 Wigner 函数（二维 ndarray）。
            params: ``NoiseParameters``，包含各类噪声参数。
            grid_range: 网格坐标范围 (min, max)，用于把物理尺度换算成像素尺度。

        输出:
            施加全部噪声后的 Wigner 函数（与输入同形）。
        """
        # 中文注释：按"损失 -> 热噪声 -> 位移 -> 相位"顺序依次叠加噪声影响。
        W = wigner.copy()

        # 1. 光子损失：向原点收缩 + 高斯扩散
        W = self._apply_photon_loss_wigner(W, params.gamma, grid_range)

        # 2. 热噪声：高斯卷积
        W = self._apply_thermal_noise_wigner(W, params.n_bar, grid_range)

        # 3. 位移噪声：额外展宽
        W = self._apply_displacement_noise_wigner(W, params.sigma_displacement, grid_range)

        # 4. 相位噪声：角度涂抹
        if params.sigma_phase > 0.01:
            W = self._apply_phase_noise_wigner(W, params.sigma_phase)

        return W

    def _apply_photon_loss_wigner(self,
                                   W: np.ndarray,
                                   gamma: float,
                                   grid_range: Tuple[float, float]) -> np.ndarray:
        """对 Wigner 函数施加光子损失通道。

        功能:
            光子损失对 Wigner 函数的变换为
                W(q,p) → (1/η) W(q/√η, p/√η) * G_{σ_η}
            其中 η = exp(-γ) 为透射率，σ_η = √((1-η)/2)。
            具体实现为两步：
              1. 向原点收缩（按 √η 缩放并补零还原尺寸）；
              2. 宽度为 σ_η 的高斯扩散（高斯卷积）。
            最后按最大值重新归一化以稳定幅值。

        输入:
            W: 输入 Wigner 函数。
            gamma: 光子损失率 γ（η = exp(-γ)）。
            grid_range: 网格坐标范围，用于把 σ_η 换算成像素单位。

        输出:
            施加光子损失后的 Wigner 函数（与输入同形）。γ 可忽略时原样返回。
        """
        eta = np.exp(-gamma)  # 透射率
        if eta > 0.999:  # 损失可忽略
            return W

        grid_size = W.shape[0]
        scale_factor = np.sqrt(eta)

        # 1. 通过插值实现向原点收缩
        from scipy.ndimage import zoom, shift
        # 缩放坐标（保持中心对齐）
        center = grid_size // 2

        # 构造收缩后的版本
        if scale_factor < 0.99:
            # 缩放后补零还原到原尺寸
            zoomed_size = int(grid_size * scale_factor)
            if zoomed_size < 3:
                zoomed_size = 3
            W_zoomed = zoom(W, scale_factor, order=1)
            # 补零还原
            pad_size = (grid_size - W_zoomed.shape[0]) // 2
            W_contracted = np.pad(W_zoomed,
                                  ((pad_size, grid_size - W_zoomed.shape[0] - pad_size),
                                   (pad_size, grid_size - W_zoomed.shape[1] - pad_size)),
                                  mode='constant', constant_values=0)
            # 保证形状一致
            W_contracted = W_contracted[:grid_size, :grid_size]
        else:
            W_contracted = W

        # 2. 高斯扩散
        sigma_eta = np.sqrt((1 - eta) / 2)
        # 换算为像素单位
        dx = (grid_range[1] - grid_range[0]) / grid_size
        sigma_pixels = sigma_eta / dx

        if sigma_pixels > 0.1:
            W_final = gaussian_filter(W_contracted, sigma=sigma_pixels)
        else:
            W_final = W_contracted

        # 重新归一化
        if np.max(np.abs(W_final)) > 1e-10:
            W_final = W_final / np.max(np.abs(W_final)) * np.max(np.abs(W))

        return W_final

    def _apply_thermal_noise_wigner(self,
                                     W: np.ndarray,
                                     n_bar: float,
                                     grid_range: Tuple[float, float]) -> np.ndarray:
        """对 Wigner 函数施加热噪声通道。

        功能:
            热噪声用宽度 √(n_bar) 的高斯卷积近似。

        输入:
            W: 输入 Wigner 函数。
            n_bar: 热平均光子数。
            grid_range: 网格坐标范围，用于把 √(n_bar) 换算成像素单位。

        输出:
            施加热噪声后的 Wigner 函数（与输入同形）。n_bar 过小时原样返回。
        """
        if n_bar < 1e-6:
            return W

        grid_size = W.shape[0]
        sigma_thermal = np.sqrt(n_bar)
        dx = (grid_range[1] - grid_range[0]) / grid_size
        sigma_pixels = sigma_thermal / dx

        return gaussian_filter(W, sigma=sigma_pixels)

    def _apply_displacement_noise_wigner(self,
                                          W: np.ndarray,
                                          sigma_disp: float,
                                          grid_range: Tuple[float, float]) -> np.ndarray:
        """对 Wigner 函数施加随机位移噪声（建模为高斯卷积）。

        功能:
            建模控制不完美（AWG 漂移、电缆相位漂移、脉冲校准误差等），
            效果等价为对 Wigner 函数做宽度 sigma_disp 的高斯卷积。

        输入:
            W: 输入 Wigner 函数。
            sigma_disp: 位移噪声标准差。
            grid_range: 网格坐标范围，用于换算像素单位。

        输出:
            施加位移噪声后的 Wigner 函数（与输入同形）。sigma_disp 过小时原样返回。
        """
        if sigma_disp < 1e-6:
            return W

        grid_size = W.shape[0]
        dx = (grid_range[1] - grid_range[0]) / grid_size
        sigma_pixels = sigma_disp / dx

        return gaussian_filter(W, sigma=sigma_pixels)

    def _apply_phase_noise_wigner(self,
                                   W: np.ndarray,
                                   sigma_phase: float) -> np.ndarray:
        """对 Wigner 函数施加相位噪声 / 退相位。

        功能:
            相位噪声在相空间表现为角度涂抹，这里通过对若干小角度旋转取平均来近似。
            采样若干个相位偏移（高斯），分别把 Wigner 旋转后求平均。

        输入:
            W: 输入 Wigner 函数。
            sigma_phase: 相位噪声标准差（弧度）。

        输出:
            角度涂抹后的 Wigner 函数（与输入同形）。
        """
        from scipy.ndimage import rotate

        n_samples = 5
        phases = np.random.normal(0, sigma_phase, n_samples)

        W_avg = np.zeros_like(W)
        for phi in phases:
            W_rotated = rotate(W, np.degrees(phi), reshape=False, mode='constant')
            W_avg += W_rotated

        return W_avg / n_samples


class PhotonLossChannel:
    """光子损失（amplitude damping）通道。

    这是超导腔中的主导噪声源，由腔衰减率 κ 和演化时间 t 决定，总损失 γ = κt。
    """

    def __init__(self, gamma: float):
        """初始化光子损失通道。

        输入:
            gamma: 总光子损失（无量纲），γ = κt；透射率 η = exp(-γ)。

        输出:
            无返回值；记录 gamma 与 eta。
        """
        self.gamma = gamma
        self.eta = np.exp(-gamma)  # 透射率

    def apply_to_wigner(self,
                        W: np.ndarray,
                        grid_range: Tuple[float, float] = (-6, 6)) -> np.ndarray:
        """把光子损失作用到 Wigner 函数表示上。

        功能:
            委托给 ``QuantumNoiseChannel._apply_photon_loss_wigner``，
            仅施加光子损失（其它噪声参数置零）。

        输入:
            W: 输入 Wigner 函数。
            grid_range: 网格坐标范围。

        输出:
            仅施加光子损失后的 Wigner 函数。
        """
        channel = QuantumNoiseChannel()
        params = NoiseParameters(gamma=self.gamma, n_bar=0, sigma_displacement=0, sigma_phase=0)
        return channel._apply_photon_loss_wigner(W, self.gamma, grid_range)

    @classmethod
    def from_t1_and_time(cls, T1: float, t: float) -> 'PhotonLossChannel':
        """由 T1 时间与演化时间构造光子损失通道。

        输入:
            T1: 能量弛豫时间（如微秒）。
            t: 演化时间（与 T1 同量纲）。

        输出:
            新的 ``PhotonLossChannel``，其 gamma = t / T1。
        """
        gamma = t / T1
        return cls(gamma)


class ThermalNoiseChannel:
    """热噪声通道。

    建模温度为 T 的热浴，用平均热光子数 n_bar 表征。
    """

    def __init__(self, n_bar: float):
        """初始化热噪声通道。

        输入:
            n_bar: 平均热光子数。

        输出:
            无返回值；记录 n_bar。
        """
        self.n_bar = n_bar

    def apply_to_wigner(self,
                        W: np.ndarray,
                        grid_range: Tuple[float, float] = (-6, 6)) -> np.ndarray:
        """把热噪声作用到 Wigner 函数表示上。

        功能:
            委托给 ``QuantumNoiseChannel._apply_thermal_noise_wigner``，
            以宽度 √(n_bar) 的高斯卷积近似热噪声。

        输入:
            W: 输入 Wigner 函数。
            grid_range: 网格坐标范围。

        输出:
            施加热噪声后的 Wigner 函数。
        """
        channel = QuantumNoiseChannel()
        return channel._apply_thermal_noise_wigner(W, self.n_bar, grid_range)

    @classmethod
    def from_temperature(cls, T_kelvin: float, omega_hz: float) -> 'ThermalNoiseChannel':
        """由物理温度构造热噪声通道。

        功能:
            利用 Planck 分布 n_bar = 1/(exp(ℏω/kT) - 1) 由温度与腔频率
            计算平均热光子数；当 kT 极小（近乎零温）时取 n_bar = 0。

        输入:
            T_kelvin: 温度（开尔文）。
            omega_hz: 腔频率（Hz）。

        输出:
            新的 ``ThermalNoiseChannel``，其 n_bar 由上式给出。
        """
        import scipy.constants as const
        hbar_omega = const.hbar * omega_hz
        kT = const.k * T_kelvin

        if kT < 1e-30:  # 近乎零温
            n_bar = 0.0
        else:
            n_bar = 1 / (np.exp(hbar_omega / kT) - 1)

        return cls(n_bar)


class DisplacementNoiseChannel:
    """随机位移噪声通道。

    建模控制不完美：
    - AWG 幅度/相位漂移；
    - 电缆热漂移；
    - 脉冲校准误差。
    """

    def __init__(self, sigma_q: float, sigma_p: Optional[float] = None):
        """初始化位移噪声通道。

        输入:
            sigma_q: q 方向位移噪声标准差。
            sigma_p: p 方向位移噪声标准差（默认与 q 相同）。

        输出:
            无返回值；记录 sigma_q 与 sigma_p。
        """
        self.sigma_q = sigma_q
        self.sigma_p = sigma_p if sigma_p is not None else sigma_q

    def sample_displacement(self) -> Tuple[float, float]:
        """采样一次随机位移误差。

        输入:
            无。

        输出:
            (dq, dp)：q 与 p 方向各自独立的高斯位移误差。
        """
        dq = np.random.normal(0, self.sigma_q)
        dp = np.random.normal(0, self.sigma_p)
        return dq, dp

    def apply_to_wigner(self,
                        W: np.ndarray,
                        grid_range: Tuple[float, float] = (-6, 6)) -> np.ndarray:
        """把位移噪声作用到 Wigner 函数（建模为高斯卷积）。

        功能:
            用 q、p 两个方向噪声标准差的平均值作为卷积宽度，
            委托给 ``QuantumNoiseChannel._apply_displacement_noise_wigner``。

        输入:
            W: 输入 Wigner 函数。
            grid_range: 网格坐标范围。

        输出:
            施加位移噪声后的 Wigner 函数。
        """
        sigma_avg = (self.sigma_q + self.sigma_p) / 2
        channel = QuantumNoiseChannel()
        return channel._apply_displacement_noise_wigner(W, sigma_avg, grid_range)


class CombinedNoiseModel:
    """带时变参数的组合噪声模型。

    提供便捷接口，用一组真实噪声参数对 GKP 态的 Wigner 函数一次性施加
    光子损失、热噪声、位移噪声等组合噪声，并支持参数更新与等效 σ 估计。
    """

    def __init__(self,
                 gamma: float = 0.05,
                 n_bar: float = 0.01,
                 sigma_disp: float = 0.1):
        """初始化组合噪声模型。

        输入:
            gamma: 光子损失率。
            n_bar: 热平均光子数。
            sigma_disp: 位移噪声标准差（相位噪声默认置 0）。

        输出:
            无返回值；内部组装 ``NoiseParameters`` 与 ``QuantumNoiseChannel``。
        """
        self.params = NoiseParameters(
            gamma=gamma,
            n_bar=n_bar,
            sigma_displacement=sigma_disp,
            sigma_phase=0.0
        )
        self.channel = QuantumNoiseChannel()

    def apply(self,
              wigner: np.ndarray,
              grid_range: Tuple[float, float] = (-6, 6)) -> np.ndarray:
        """对 Wigner 函数施加组合噪声。

        输入:
            wigner: 输入 Wigner 函数。
            grid_range: 网格坐标范围。

        输出:
            施加全部噪声后的 Wigner 函数。
        """
        return self.channel.apply_all(wigner, self.params, grid_range)

    def update_params(self, **kwargs):
        """更新噪声参数（关键字参数形式）。

        功能:
            仅更新 ``NoiseParameters`` 中已存在的字段，未知字段会被忽略。

        输入:
            **kwargs: 要更新的参数，如 gamma=0.1、n_bar=0.05。

        输出:
            无返回值；就地修改 self.params。
        """
        for key, value in kwargs.items():
            if hasattr(self.params, key):
                setattr(self.params, key, value)

    def get_effective_sigma(self) -> float:
        """计算等效总噪声标准差。

        功能:
            把光子损失、热噪声、位移噪声折算成单一等效位移噪声标准差：

                σ_total = √(σ_loss² + σ_thermal² + σ_disp²)

            其中 σ_loss = √(γ/2)、σ_thermal = √(n_bar)、σ_disp = sigma_displacement。
            该等效 σ 常用于上层快速估计与参数调度，不替代完整仿真。

        输入:
            无。

        输出:
            等效总噪声标准差（float）。
        """
        # 中文注释：该等效 sigma 常用于上层快速估计与参数调度，不替代完整仿真。
        # 光子损失贡献
        sigma_loss = np.sqrt(self.params.gamma / 2) if self.params.gamma > 0 else 0

        # 热噪声贡献
        sigma_thermal = np.sqrt(self.params.n_bar) if self.params.n_bar > 0 else 0

        # 总噪声
        sigma_total = np.sqrt(sigma_loss**2 +
                              sigma_thermal**2 +
                              self.params.sigma_displacement**2)
        return sigma_total
