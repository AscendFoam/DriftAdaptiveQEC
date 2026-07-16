"""GKP 态制备模块（GKP State Preparation）。

本文件负责 GKP（Gottesman-Kitaev-Preskill）量子态的创建与表示，
覆盖"近似（有限能量）GKP 态"的构造、Wigner 函数计算与位移操作。

关键设计与边界：
- 优先使用 Strawberry Fields（SF）做 Fock-backend 量子态仿真；当 SF 不可用或运行
  失败时，旧接口自动退化为启发式 signed-grid visualization。该 fallback 只做 legacy
  smoke/可视化，不是归一化 Wigner，也不能支撑 finite-energy logical-channel claim；
  可归一化 wavefunction、damped-projector、syndrome distribution 与 sampled Wigner
  reference 位于 ``physics.finite_energy_gkp``。
- ``LATTICE_CONST = √(2π)`` 是本模块本地定义的晶格常数，也是后续噪声、测量、
  解码模块共享的基础常量（部分子模块从 ``constants`` 中导入同一常量）。
"""

import numpy as np
from typing import Optional, Tuple, Union, Literal
from dataclasses import dataclass

from .constants import LATTICE_CONST

# 尝试导入 Strawberry Fields（精确仿真后端）；失败则置标志位并降级为解析近似。
try:
    import strawberryfields as sf
    from strawberryfields.ops import GKP, Dgate, Rgate, Sgate
    HAS_STRAWBERRYFIELDS = True
except ImportError:
    HAS_STRAWBERRYFIELDS = False
    import warnings
    warnings.warn("Strawberry Fields not available. Using analytical approximation.", ImportWarning)


@dataclass
class GKPParameters:
    """GKP 态参数容器（数据类）。

    字段:
        delta: 有限能量参数（高斯包络宽度）。
               越小越接近理想 GKP 态，但态能量越高（典型值 0.2~0.5，
               对应约 10~15 dB 压缩）。
        logical_state: 编码的逻辑量子比特态，取值 '0' / '1' / '+' / '-'。
        cutoff: Fock 空间截断维度，控制 SF 仿真的精度与开销。
    """
    delta: float  # Finite energy parameter (envelope width)
    logical_state: str = '0'  # '0', '1', '+', '-'
    cutoff: int = 50  # Fock space cutoff dimension


class ApproximateGKPState:
    """近似（有限能量）GKP 态。

    理想的 GKP 态具有无穷能量，实际实现采用带高斯包络的近似态：

        |GKP_Δ⟩ ∝ Σ_n exp(-Δ² n²) |n√(2π)⟩_q

    其中 Δ 为有限能量参数（越小越理想，但能量越高）。本类支持：
    - 用 SF Fock backend 制备，或退化为 legacy heuristic visualization；
    - 计算 Wigner 函数（相空间分布）；
    - 施加位移操作并估计平均光子数。
    """

    def __init__(self,
                 delta: float = 0.3,
                 logical_state: str = '0',
                 cutoff: int = 50,
                 use_strawberryfields: bool = True):
        """初始化一个近似 GKP 态。

        功能:
            记录参数并据此制备态。若 SF 可用且 ``use_strawberryfields=True``，
            则调用 SF 精确制备；否则切到解析近似模式（仅记录参数，不调用 SF）。
            同时计算等效压缩度（dB）。

        输入:
            delta: 有限能量参数（典型 0.2~0.5），对应约 10~15 dB 压缩。
            logical_state: 逻辑量子比特态，'0' / '1' / '+' / '-'。
            cutoff: Fock 空间截断维度（仅 SF 模式生效）。
            use_strawberryfields: 是否尝试使用 SF 制备态。

        输出:
            无返回值；构造完成后实例持有 delta、logical_state、cutoff、lattice、
            squeezing_db 等属性，以及态表示（SF 状态对象或解析模式标志）。
        """
        self.delta = delta
        self.logical_state = logical_state
        self.cutoff = cutoff
        self.lattice = LATTICE_CONST

        # 计算等效压缩度（dB）：squeezing_db = -10*log10(2*delta^2)
        self.squeezing_db = -10 * np.log10(2 * delta**2)

        # 态表示的缓存占位
        self._wigner_cache = None
        self._sf_state = None

        # 制备态
        if use_strawberryfields and HAS_STRAWBERRYFIELDS:
            self._prepare_state_sf()
        else:
            self._use_analytical = True

    def _prepare_state_sf(self):
        """使用 Strawberry Fields 精确制备 GKP 态（仅在 SF 可用时走此路径）。

        功能:
            构建单模式 SF 程序：用 GKP 门制备 |0⟩_L 或 |1⟩_L；
            对 |+⟩_L / |-⟩_L 则在制备后施加 90° 相空间旋转门（逻辑 Hadamard）。
            随后用 fock 后端以指定截断维度运行，得到态对象并标记为"非解析"模式。
            若 SF 运行抛异常，则降级为解析近似模式。

        输入:
            无（参数取自 self.delta / self.logical_state / self.cutoff）。

        输出:
            无返回值；成功时写入 self._sf_state（SF 态对象）并把
            self._use_analytical 置为 False；失败时把 _use_analytical 置为 True。
        """
        # 中文注释：仅在 SF 可用时走该路径，得到更真实但更慢的状态表示。
        prog = sf.Program(1)

        # 逻辑态 -> GKP 门编码参数 [位移索引, ...]
        state_map = {
            '0': [0, 0],  # |0⟩_L
            '1': [1, 0],  # |1⟩_L
            '+': [0, 0],  # |+⟩_L（之后再施加 Hadamard）
            '-': [1, 0],  # |-⟩_L（之后再施加 Hadamard）
        }

        with prog.context as q:
            # 以有限能量参数 epsilon=delta 制备 GKP 态
            GKP(epsilon=self.delta, state=state_map.get(self.logical_state, [0, 0])) | q[0]

            # |+⟩ / |-⟩ 需要额外 90° 相空间旋转（逻辑 Hadamard）
            if self.logical_state in ['+', '-']:
                Rgate(np.pi / 2) | q[0]

        try:
            eng = sf.Engine("fock", backend_options={"cutoff_dim": self.cutoff})
            result = eng.run(prog)
            self._sf_state = result.state
            self._use_analytical = False
        except Exception as e:
            print(f"SF state preparation failed: {e}. Using analytical approximation.")
            self._use_analytical = True

    def get_wigner(self,
                   q_points: int = 64,
                   p_points: int = 64,
                   q_range: Tuple[float, float] = (-6, 6),
                   p_range: Tuple[float, float] = (-6, 6)) -> np.ndarray:
        """计算 GKP 态的 Wigner 函数。

        功能:
            在指定的 (q, p) 网格上计算态的 Wigner 函数（相空间概率分布表示）。
            优先使用 SF 的 Wigner 计算；若处于解析模式或 SF 计算失败，则退化为
            ``_compute_wigner_analytical`` 的 legacy heuristic signed grid。后者按最大
            绝对值归一化，不是概率归一化 Wigner；论文级 finite-energy 输出应使用
            ``physics.finite_energy_gkp``。

        输入:
            q_points: q 方向网格点数。
            p_points: p 方向网格点数。
            q_range: q 取值范围 (q_min, q_max)。
            p_range: p 取值范围 (p_min, p_max)。

        输出:
            shape=(p_points, q_points) 的二维 ndarray，给出每个网格点的
            Wigner 函数值（解析近似分支已做最大值归一化）。
        """
        q_vec = np.linspace(q_range[0], q_range[1], q_points)
        p_vec = np.linspace(p_range[0], p_range[1], p_points)

        if not self._use_analytical and self._sf_state is not None:
            # 优先使用 Strawberry Fields 的精确 Wigner
            try:
                return self._sf_state.wigner(mode=0, xvec=q_vec, pvec=p_vec)
            except Exception:
                pass

        # 退化为解析近似
        return self._compute_wigner_analytical(q_vec, p_vec)

    def _compute_wigner_analytical(self,
                                   q_vec: np.ndarray,
                                   p_vec: np.ndarray) -> np.ndarray:
        """计算 legacy heuristic signed grid（历史名称保留，不是 normalized Wigner）。

        功能:
            对晶格点求和近似 GKP |0⟩_L 态的 Wigner 函数：

                W(q,p) ∝ Σ_{n,m} (-1)^(n+m) exp(-Δ²(n²+m²))
                                  × exp(-|r - r_{nm}|² / (2Δ²))

            其中 r_{nm} = (n√(2π), m√(2π)) 为晶格点坐标。
            - 高斯包络 exp(-Δ²(n²+m²)) 体现有限能量效应；
            - 棋盘格符号 (-1)^(n+m) 区分 |0⟩_L / |+⟩_L 与 |1⟩_L / |-⟩_L；
            最后做最大值归一化。

        输入:
            q_vec: q 方向一维坐标数组。
            p_vec: p 方向一维坐标数组。

        输出:
            形状为 (len(p_vec), len(q_vec)) 的二维 ndarray，归一化后的 Wigner 值。
        """
        # 中文注释：该函数是"无 SF 依赖"时的核心近似实现，计算速度更高。
        Q, P = np.meshgrid(q_vec, p_vec)
        W = np.zeros_like(Q)

        # 求和覆盖的晶格点范围（按 q 方向最大幅度估计）
        n_max = int(np.ceil(max(abs(q_vec.max()), abs(q_vec.min())) / self.lattice)) + 2

        for nq in range(-n_max, n_max + 1):
            for np_ in range(-n_max, n_max + 1):
                # 当前晶格点的中心坐标
                q_center = nq * self.lattice
                p_center = np_ * self.lattice

                # 高斯包络（有限能量效应）
                envelope = np.exp(-self.delta**2 * (nq**2 + np_**2))

                # 该晶格点处的高斯峰
                gaussian = np.exp(-((Q - q_center)**2 + (P - p_center)**2) / (2 * self.delta**2))

                # GKP |0⟩_L 的交替符号（棋盘格图样）
                if self.logical_state in ['0', '+']:
                    sign = (-1) ** (nq + np_)
                else:  # '1' 或 '-'
                    sign = (-1) ** (nq + np_ + 1)

                W += sign * envelope * gaussian

        # 归一化
        W = W / np.max(np.abs(W))

        return W

    def apply_displacement(self, alpha: complex) -> 'ApproximateGKPState':
        """对态施加位移操作 D(α)。

        功能:
            在解析模式下，位移不会被真正作用于 Wigner，而是以累计复位移的形式
            记录在新态的 ``_displacement`` 属性中，便于上层跟踪。

        输入:
            alpha: 位移复参数 α（实部对应 q，虚部对应 p）。

        输出:
            一个新的 ``ApproximateGKPState``（解析模式，关闭 SF），其
            ``_displacement`` 等于本态已有位移加上 α。
        """
        # 解析模式下单独跟踪位移
        new_state = ApproximateGKPState(
            delta=self.delta,
            logical_state=self.logical_state,
            cutoff=self.cutoff,
            use_strawberryfields=False
        )
        new_state._displacement = getattr(self, '_displacement', 0) + alpha
        return new_state

    @property
    def mean_photon_number(self) -> float:
        """估计态的平均光子数 n̄。

        功能:
            若存在 SF 态对象，则返回其真实平均光子数；否则用理想 GKP 的解析
            估计 n̄ ≈ 1/(2Δ²)。

        输入:
            无。

        输出:
            平均光子数（float）。
        """
        if self._sf_state is not None:
            try:
                return float(self._sf_state.mean_photon(mode=0)[0])
            except Exception:
                pass

        # 解析估计：理想 GKP 的 n̄ ≈ 1/(2Δ²)
        return 1 / (2 * self.delta**2)


class GKPStateFactory:
    """GKP 态工厂类。

    封装常用 GKP 态的创建逻辑，提供便捷接口生成 |0⟩_L / |1⟩_L / |+⟩_L 等
    逻辑态，统一管理默认 Fock 截断维度与是否使用 SF。
    """

    def __init__(self, default_cutoff: int = 50, use_sf: bool = True):
        """初始化工厂。

        输入:
            default_cutoff: 创建态时默认的 Fock 截断维度。
            use_sf: 是否优先使用 SF（实际是否启用还取决于 SF 是否可用）。

        输出:
            无返回值；记录 default_cutoff 与 use_sf（已与 SF 可用性取交集）。
        """
        self.default_cutoff = default_cutoff
        self.use_sf = use_sf and HAS_STRAWBERRYFIELDS

    def create_logical_zero(self, delta: float = 0.3) -> ApproximateGKPState:
        """创建逻辑 |0⟩_L 态。

        输入:
            delta: 有限能量参数（默认 0.3）。

        输出:
            logical_state='0' 的 ``ApproximateGKPState``。
        """
        return ApproximateGKPState(delta=delta, logical_state='0',
                                   cutoff=self.default_cutoff,
                                   use_strawberryfields=self.use_sf)

    def create_logical_one(self, delta: float = 0.3) -> ApproximateGKPState:
        """创建逻辑 |1⟩_L 态。

        输入:
            delta: 有限能量参数（默认 0.3）。

        输出:
            logical_state='1' 的 ``ApproximateGKPState``。
        """
        return ApproximateGKPState(delta=delta, logical_state='1',
                                   cutoff=self.default_cutoff,
                                   use_strawberryfields=self.use_sf)

    def create_logical_plus(self, delta: float = 0.3) -> ApproximateGKPState:
        """创建逻辑 |+⟩_L 态。

        输入:
            delta: 有限能量参数（默认 0.3）。

        输出:
            logical_state='+' 的 ``ApproximateGKPState``（SF 模式下会施加逻辑 Hadamard）。
        """
        return ApproximateGKPState(delta=delta, logical_state='+',
                                   cutoff=self.default_cutoff,
                                   use_strawberryfields=self.use_sf)

    def create_from_params(self, params: GKPParameters) -> ApproximateGKPState:
        """根据 ``GKPParameters`` 数据类创建 GKP 态。

        输入:
            params: ``GKPParameters`` 实例，提供 delta / logical_state / cutoff。

        输出:
            参数对应的 ``ApproximateGKPState``。
        """
        return ApproximateGKPState(
            delta=params.delta,
            logical_state=params.logical_state,
            cutoff=params.cutoff,
            use_strawberryfields=self.use_sf
        )


def delta_to_squeezing_db(delta: float) -> float:
    """把 GKP 的 delta 参数换算为等效压缩度（dB）。

    输入:
        delta: GKP 有限能量参数。

    输出:
        等效压缩度（dB），公式为 -10 * log10(2 * delta^2)。
        delta 越小，压缩度越高。
    """
    return -10 * np.log10(2 * delta**2)


def squeezing_db_to_delta(squeezing_db: float) -> float:
    """把压缩度（dB）反换算为 GKP 的 delta 参数。

    输入:
        squeezing_db: 压缩度（dB）。

    输出:
        等效的 delta 参数，公式为 sqrt(10^(-squeezing_db/10) / 2)。
    """
    return np.sqrt(10**(-squeezing_db / 10) / 2)

"""
### 代码核心功能解析
这段代码是一个**近似GKP量子态的生成与操作模块**，专为有限能量的GKP（Gottesman-Kitaev-Preskill）态设计，核心目标是在量子仿真中高效创建、表示和计算GKP态，以下是分层解析：

#### 1. 基础定义与依赖管理
- **核心常量**：`LATTICE_CONST = √(2π)` 是GKP态的晶格常数，是后续噪声、测量、解码的共享基础，决定了GKP态在相空间中的晶格点间距。
- **依赖兼容**：优先导入量子仿真库Strawberry Fields（SF），若导入失败则触发警告，自动降级为解析近似计算（速度更快，精度稍低）。

#### 2. 数据结构与核心类
##### (1) `GKPParameters` 数据类
用`dataclass`封装GKP态的核心参数，便于参数管理：
- `delta`：有限能量参数（包络宽度），越小越接近理想GKP态，但能量越高（典型值0.2-0.5）；
- `logical_state`：逻辑量子比特态（'0'/'1'/'+/-'）；
- `cutoff`：Fock空间截断维度（仿真精度控制）。

##### (2) `ApproximateGKPState` 核心类
实现近似GKP态的创建、Wigner函数计算、位移操作等核心功能：
- **初始化逻辑**：
  - 计算等效压缩度（dB）：`squeezing_db = -10×log10(2×delta²)`；
  - 优先调用SF创建高精度GKP态，失败则自动切换到解析近似模式。
- **_prepare_state_sf 方法**：
  通过SF的GKP门+旋转门（Rgate）生成不同逻辑态的GKP态（如|+⟩/|-⟩需额外90°相空间旋转），并指定Fock截断维度。
- **get_wigner 方法**：
  计算GKP态的Wigner函数（相空间分布）：优先用SF的高精度计算，失败则调用解析近似；
  解析近似的核心逻辑是对晶格点求和：`W(q,p) ∝ Σ(-1)^(n+m)×exp(-Δ²(n²+m²))×exp(-|r-r_nm|²/(2Δ²))`，其中`r_nm`是晶格点坐标。
- **辅助方法**：
  `apply_displacement`：对GKP态施加位移操作；
  `mean_photon_number`：估算平均光子数（SF模式用真实值，近似模式用`1/(2×delta²)`）。

##### (3) `GKPStateFactory` 工厂类
封装GKP态的创建逻辑，提供便捷的接口生成不同逻辑态的GKP态（如`create_logical_zero`直接生成|0⟩_L），降低使用成本。

#### 3. 工具函数
- `delta_to_squeezing_db`/`squeezing_db_to_delta`：实现delta参数与压缩度（dB）的双向转换，便于和实验参数（如压缩光dB值）对齐。

### 总结
1. 核心目标：生成有限能量的近似GKP态，优先用Strawberry Fields高精度仿真，降级为解析近似保证可用性；
2. 核心逻辑：通过晶格常数定义GKP态的相空间晶格，用delta参数控制能量/理想度，支持不同逻辑态的生成和Wigner函数计算；
3. 设计特点：兼容SF高精度仿真和解析近似两种模式，兼顾精度与速度，通过工厂类简化使用，参数与实验（压缩度dB）对齐。
"""
