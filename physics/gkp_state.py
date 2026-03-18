"""
GKP State Preparation Module

Provides tools for creating and manipulating approximate (finite-energy) GKP states.
Uses Strawberry Fields for quantum state simulation when available,
with analytical fallback for faster computation.

中文说明：
- 本文件负责 GKP 态的创建与表示（优先 SF 精确仿真，失败时退化为解析近似）。
- 其中 LATTICE_CONST 是后续噪声、测量、解码模块共享的基础常量。
"""

import numpy as np
from typing import Optional, Tuple, Union, Literal
from dataclasses import dataclass

# GKP lattice constant
LATTICE_CONST = np.sqrt(2 * np.pi)  # ≈ 2.507

# Try to import Strawberry Fields
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
    """Parameters for GKP state"""
    delta: float  # Finite energy parameter (envelope width)
    logical_state: str = '0'  # '0', '1', '+', '-'
    cutoff: int = 50  # Fock space cutoff dimension


class ApproximateGKPState:
    """
    Approximate (finite-energy) GKP state

    The ideal GKP state has infinite energy. Real implementations use
    approximate GKP states with a Gaussian envelope:

    |GKP_Δ⟩ ∝ Σ_n exp(-Δ² n²) |n√(2π)⟩_q

    where Δ is the finite energy parameter (smaller = more ideal but higher energy)
    """

    def __init__(self,
                 delta: float = 0.3,
                 logical_state: str = '0',
                 cutoff: int = 50,
                 use_strawberryfields: bool = True):
        """
        Initialize an approximate GKP state

        Args:
            delta: Finite energy parameter (typical: 0.2-0.5)
                   Corresponds to ~10-15 dB squeezing
            logical_state: Logical qubit state ('0', '1', '+', '-')
            cutoff: Fock space cutoff for simulation
            use_strawberryfields: Whether to use SF for state preparation
        """
        self.delta = delta
        self.logical_state = logical_state
        self.cutoff = cutoff
        self.lattice = LATTICE_CONST

        # Compute equivalent squeezing in dB
        self.squeezing_db = -10 * np.log10(2 * delta**2)

        # State representation
        self._wigner_cache = None
        self._sf_state = None

        # Prepare the state
        if use_strawberryfields and HAS_STRAWBERRYFIELDS:
            self._prepare_state_sf()
        else:
            self._use_analytical = True

    def _prepare_state_sf(self):
        """Prepare GKP state using Strawberry Fields"""
        # 中文注释：仅在 SF 可用时走该路径，得到更真实但更慢的状态表示。
        prog = sf.Program(1)

        # GKP state encoding
        state_map = {
            '0': [0, 0],  # |0⟩_L
            '1': [1, 0],  # |1⟩_L
            '+': [0, 0],  # |+⟩_L (apply Hadamard after)
            '-': [1, 0],  # |-⟩_L (apply Hadamard after)
        }

        with prog.context as q:
            # Prepare GKP state with finite energy
            GKP(epsilon=self.delta, state=state_map.get(self.logical_state, [0, 0])) | q[0]

            # For |+⟩ and |-⟩, apply logical Hadamard (90° rotation in phase space)
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
        """
        Compute the Wigner function of the GKP state

        Args:
            q_points: Number of points in q direction
            p_points: Number of points in p direction
            q_range: Range of q values
            p_range: Range of p values

        Returns:
            2D numpy array of Wigner function values
        """
        q_vec = np.linspace(q_range[0], q_range[1], q_points)
        p_vec = np.linspace(p_range[0], p_range[1], p_points)

        if not self._use_analytical and self._sf_state is not None:
            # Use Strawberry Fields
            try:
                return self._sf_state.wigner(mode=0, xvec=q_vec, pvec=p_vec)
            except Exception:
                pass

        # Analytical approximation
        return self._compute_wigner_analytical(q_vec, p_vec)

    def _compute_wigner_analytical(self,
                                   q_vec: np.ndarray,
                                   p_vec: np.ndarray) -> np.ndarray:
        """
        Compute Wigner function using analytical approximation

        The Wigner function of an approximate GKP |0⟩_L state is approximately:
        W(q,p) ∝ Σ_{n,m} (-1)^(n+m) exp(-Δ²(n²+m²)) × exp(-|r - r_{nm}|²/(2Δ²))

        where r_{nm} = (n√(2π), m√(2π))
        """
        # 中文注释：该函数是“无 SF 依赖”时的核心近似实现，计算速度更高。
        Q, P = np.meshgrid(q_vec, p_vec)
        W = np.zeros_like(Q)

        # Number of lattice points to sum over
        n_max = int(np.ceil(max(abs(q_vec.max()), abs(q_vec.min())) / self.lattice)) + 2

        for nq in range(-n_max, n_max + 1):
            for np_ in range(-n_max, n_max + 1):
                # Lattice point position
                q_center = nq * self.lattice
                p_center = np_ * self.lattice

                # Gaussian envelope (finite energy effect)
                envelope = np.exp(-self.delta**2 * (nq**2 + np_**2))

                # Peak at this lattice point
                gaussian = np.exp(-((Q - q_center)**2 + (P - p_center)**2) / (2 * self.delta**2))

                # Alternating sign for GKP |0⟩_L (checkerboard pattern)
                if self.logical_state in ['0', '+']:
                    sign = (-1) ** (nq + np_)
                else:  # '1' or '-'
                    sign = (-1) ** (nq + np_ + 1)

                W += sign * envelope * gaussian

        # Normalize
        W = W / np.max(np.abs(W))

        return W

    def apply_displacement(self, alpha: complex) -> 'ApproximateGKPState':
        """Apply a displacement operation D(α) to the state"""
        # For analytical mode, we track displacement separately
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
        """Estimate mean photon number"""
        if self._sf_state is not None:
            try:
                return float(self._sf_state.mean_photon(mode=0)[0])
            except Exception:
                pass

        # Analytical estimate: n̄ ≈ 1/(2Δ²) for ideal GKP
        return 1 / (2 * self.delta**2)


class GKPStateFactory:
    """Factory for creating GKP states with various configurations"""

    def __init__(self, default_cutoff: int = 50, use_sf: bool = True):
        self.default_cutoff = default_cutoff
        self.use_sf = use_sf and HAS_STRAWBERRYFIELDS

    def create_logical_zero(self, delta: float = 0.3) -> ApproximateGKPState:
        """Create |0⟩_L state"""
        return ApproximateGKPState(delta=delta, logical_state='0',
                                   cutoff=self.default_cutoff,
                                   use_strawberryfields=self.use_sf)

    def create_logical_one(self, delta: float = 0.3) -> ApproximateGKPState:
        """Create |1⟩_L state"""
        return ApproximateGKPState(delta=delta, logical_state='1',
                                   cutoff=self.default_cutoff,
                                   use_strawberryfields=self.use_sf)

    def create_logical_plus(self, delta: float = 0.3) -> ApproximateGKPState:
        """Create |+⟩_L state"""
        return ApproximateGKPState(delta=delta, logical_state='+',
                                   cutoff=self.default_cutoff,
                                   use_strawberryfields=self.use_sf)

    def create_from_params(self, params: GKPParameters) -> ApproximateGKPState:
        """Create GKP state from parameters dataclass"""
        return ApproximateGKPState(
            delta=params.delta,
            logical_state=params.logical_state,
            cutoff=params.cutoff,
            use_strawberryfields=self.use_sf
        )


def delta_to_squeezing_db(delta: float) -> float:
    """Convert GKP delta parameter to squeezing in dB"""
    return -10 * np.log10(2 * delta**2)


def squeezing_db_to_delta(squeezing_db: float) -> float:
    """Convert squeezing in dB to GKP delta parameter"""
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