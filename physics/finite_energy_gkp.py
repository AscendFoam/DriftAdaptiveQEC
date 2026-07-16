"""可归一化的 single-mode square approximate-GKP 态族。

本模块提供两种明确区分的 position-wavefunction reference：

1. ``gaussian_envelope``：独立指定 Gaussian peak probability width 与 Gaussian
   envelope inverse width；
2. ``damped_projector``：把 ``N_Delta = exp(-Delta^2 a^dagger a)`` 作用于 ideal
   comb，并用 harmonic-oscillator imaginary-time (Mehler) kernel 写成可计算 comb。

两者都只是一维纯态 wavefunction / syndrome / sampled-Wigner reference；它们不等于
完整 cavity-transmon dynamics，也不替代后续 logical-channel 与 recovery simulation。
默认输出仍使用 decoder-standardized correction-cell spacing
``LATTICE_CONST=sqrt(2*pi)``，因此 logical 0/1 的 q peaks 分别位于偶/奇整数倍
lattice。对 ``damped_projector``，宽度、包络与 Jacobian 现在按
``coordinate_scale=lattice/sqrt(pi)`` 一起缩放；它由 canonical ``[x,p]=i`` 态做
严格坐标 dilation 得到，不再只移动 peaks 而漏掉 peak width/envelope scaling。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, cosh, exp, isfinite, log, pi, sqrt, tanh
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import LATTICE_CONST
from .quadrature_conventions import CANONICAL_LOGICAL_CELL_SPACING


LogicalState = Literal["0", "1", "+", "-"]
FiniteEnergyModel = Literal["gaussian_envelope", "damped_projector"]
_MAX_COMPONENTS_PER_BASIS = 512
_MAX_WIGNER_POINTS = 2049


@dataclass(frozen=True)
class PeakTable:
    """截断后、已经归一化到目标 logical state 的 Gaussian components。"""

    ideal_lattice_indices: NDArray[np.int64]
    centers: NDArray[np.float64]
    coefficients: NDArray[np.float64]
    amplitude_variance: float


@dataclass(frozen=True)
class SyndromeDistribution:
    """一个 correction cell 上的 normalized syndrome density 与截断诊断。"""

    syndrome: NDArray[np.float64]
    density: NDArray[np.float64]
    raw_density: NDArray[np.float64]
    captured_mass: float
    alias_min: int
    alias_max: int


@dataclass(frozen=True)
class WignerLikeGrid:
    """由 wavefunction correlation FFT 得到的 sampled signed quasiprobability。

    ``values`` shape 为 ``(len(q), len(p))``。这是有限 q-window 和离散 y-grid 上的
    Wigner integral approximation；``captured_probability`` 与 ``total_mass`` 显式
    暴露截断/离散误差，不能冒充 Fock-space tomography。
    """

    q: NDArray[np.float64]
    p: NDArray[np.float64]
    values: NDArray[np.float64]
    q_probability: NDArray[np.float64]
    q_marginal: NDArray[np.float64]
    captured_probability: float
    total_mass: float
    negative_volume: float


def _finite_positive(value: float, name: str) -> float:
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _validate_lattice(lattice: float) -> float:
    return _finite_positive(lattice, "lattice")


def _gaussian_cross_integral(
    centers_a: np.ndarray,
    variance_a: float,
    centers_b: np.ndarray,
    variance_b: float,
) -> np.ndarray:
    """返回所有未归一化 amplitude Gaussians 的 pairwise overlap。"""

    denominator = variance_a + variance_b
    prefactor = sqrt(2.0 * pi * variance_a * variance_b / denominator)
    differences = centers_a[:, np.newaxis] - centers_b[np.newaxis, :]
    return prefactor * np.exp(-0.5 * differences * differences / denominator)


def _normalize_components(
    centers: np.ndarray,
    coefficients: np.ndarray,
    amplitude_variance: float,
) -> np.ndarray:
    gram = _gaussian_cross_integral(
        centers,
        amplitude_variance,
        centers,
        amplitude_variance,
    )
    norm_squared = float(coefficients @ gram @ coefficients)
    if not isfinite(norm_squared) or norm_squared <= 0.0:
        raise ValueError("truncated GKP comb has zero or non-finite norm")
    return coefficients / sqrt(norm_squared)


class FiniteEnergyGKPState:
    """归一化 approximate-GKP position wavefunction。

    Parameters
    ----------
    logical_state:
        ``0/1/+/-``。``+/-`` 由各自归一化的 0/1 comb 相干叠加后再次归一化。
    model:
        ``gaussian_envelope`` 或 ``damped_projector``。
    peak_sigma:
        Gaussian-envelope model 中单个隔离 peak 的 *probability* standard deviation。
    envelope_kappa:
        Gaussian-envelope amplitude ``exp[-(kappa*q_peak)^2/2]`` 的 inverse width；
        省略时取 ``peak_sigma``，形成常用 symmetric parameterization。
    projector_delta:
        Damped-projector model 的 Delta，算符为 ``exp(-Delta^2 n)``。
    """

    def __init__(
        self,
        logical_state: LogicalState,
        *,
        model: FiniteEnergyModel,
        peak_sigma: float | None = None,
        envelope_kappa: float | None = None,
        projector_delta: float | None = None,
        lattice: float = LATTICE_CONST,
        tail_tolerance: float = 1.0e-12,
    ) -> None:
        if logical_state not in {"0", "1", "+", "-"}:
            raise ValueError("logical_state must be one of '0', '1', '+', '-'")
        if model not in {"gaussian_envelope", "damped_projector"}:
            raise ValueError("model must be 'gaussian_envelope' or 'damped_projector'")
        tolerance = float(tail_tolerance)
        if not isfinite(tolerance) or not (0.0 < tolerance < 1.0):
            raise ValueError("tail_tolerance must lie strictly between 0 and 1")

        self.logical_state = logical_state
        self.model = model
        self.lattice = _validate_lattice(lattice)
        self.coordinate_scale = self.lattice / CANONICAL_LOGICAL_CELL_SPACING
        self.tail_tolerance = tolerance
        self.peak_sigma: float | None = None
        self.envelope_kappa: float | None = None
        self.projector_delta: float | None = None

        if model == "gaussian_envelope":
            if peak_sigma is None:
                raise ValueError("peak_sigma is required for gaussian_envelope")
            if projector_delta is not None:
                raise ValueError("projector_delta is not used by gaussian_envelope")
            self.peak_sigma = _finite_positive(peak_sigma, "peak_sigma")
            self.envelope_kappa = _finite_positive(
                self.peak_sigma if envelope_kappa is None else envelope_kappa,
                "envelope_kappa",
            )
            self.amplitude_variance = 2.0 * self.peak_sigma * self.peak_sigma
            if not isfinite(self.amplitude_variance) or self.amplitude_variance <= 0.0:
                raise ValueError("peak_sigma produces a non-representable amplitude variance")
            decay_rate = self.envelope_kappa
            contraction = 1.0
        else:
            if projector_delta is None:
                raise ValueError("projector_delta is required for damped_projector")
            if peak_sigma is not None or envelope_kappa is not None:
                raise ValueError(
                    "peak_sigma/envelope_kappa are not used by damped_projector"
                )
            self.projector_delta = _finite_positive(projector_delta, "projector_delta")
            epsilon = self.projector_delta * self.projector_delta
            if not isfinite(epsilon) or epsilon <= 0.0:
                raise ValueError("projector_delta squared must be finite and positive")
            finite_width = tanh(epsilon)
            if not isfinite(finite_width) or finite_width <= 0.0:
                raise ValueError("projector_delta is below the representable finite-width range")
            # N_Delta=exp(-Delta^2 n) 的 Mehler kernel 首先定义在 canonical
            # [x,p]=i chart。输出 lattice 可以是 decoder-standardized chart，
            # 但此时 peak amplitude variance 必须乘 scale^2，envelope inverse
            # width 必须除以 scale，才能保持同一个物理态及归一化 Jacobian。
            self.amplitude_variance = (
                self.coordinate_scale * self.coordinate_scale * finite_width
            )
            decay_rate = sqrt(finite_width) / self.coordinate_scale
            # sech(epsilon) 的稳定形式，避免大 epsilon 的 cosh overflow。
            contraction = (
                1.0 / cosh(epsilon)
                if epsilon < 350.0
                else 2.0 * exp(-epsilon) / (1.0 + exp(-2.0 * epsilon))
            )

        basis_zero = self._build_basis_components(0, decay_rate, contraction)
        basis_one = self._build_basis_components(1, decay_rate, contraction)
        zero_coefficients = _normalize_components(
            basis_zero[1], basis_zero[2], self.amplitude_variance
        )
        one_coefficients = _normalize_components(
            basis_one[1], basis_one[2], self.amplitude_variance
        )

        if logical_state == "0":
            indices, centers, coefficients = (
                basis_zero[0],
                basis_zero[1],
                zero_coefficients,
            )
        elif logical_state == "1":
            indices, centers, coefficients = (
                basis_one[0],
                basis_one[1],
                one_coefficients,
            )
        else:
            sign = 1.0 if logical_state == "+" else -1.0
            indices = np.concatenate((basis_zero[0], basis_one[0]))
            centers = np.concatenate((basis_zero[1], basis_one[1]))
            coefficients = np.concatenate((zero_coefficients, sign * one_coefficients))
            order = np.argsort(centers)
            indices = indices[order]
            centers = centers[order]
            coefficients = _normalize_components(
                centers,
                coefficients[order],
                self.amplitude_variance,
            )

        self._indices = np.asarray(indices, dtype=np.int64)
        self._centers = np.asarray(centers, dtype=np.float64)
        self._coefficients = np.asarray(coefficients, dtype=np.float64)

    def _build_basis_components(
        self,
        parity: int,
        decay_rate: float,
        contraction: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        threshold = sqrt(2.0 * log(1.0 / self.tail_tolerance))
        denominator = decay_rate * self.lattice
        required_index = threshold / denominator
        if not isfinite(required_index) or required_index > _MAX_COMPONENTS_PER_BASIS:
            raise ValueError(
                "finite-energy envelope requires too many peaks; increase tail_tolerance or finite-energy parameter"
            )
        max_index = int(ceil(required_index)) + 2
        indices = np.arange(-max_index, max_index + 1, dtype=np.int64)
        indices = indices[np.mod(indices, 2) == parity]
        if indices.size > _MAX_COMPONENTS_PER_BASIS:
            raise ValueError(
                "finite-energy envelope requires too many peaks; increase tail_tolerance or finite-energy parameter"
            )
        ideal_centers = indices.astype(np.float64) * self.lattice
        coefficients = np.exp(-0.5 * (decay_rate * ideal_centers) ** 2)
        centers = contraction * ideal_centers
        return indices, centers, coefficients

    @property
    def peak_table(self) -> PeakTable:
        return PeakTable(
            ideal_lattice_indices=self._indices.copy(),
            centers=self._centers.copy(),
            coefficients=self._coefficients.copy(),
            amplitude_variance=self.amplitude_variance,
        )

    @property
    def component_count(self) -> int:
        return int(self._centers.size)

    @property
    def support_radius(self) -> float:
        gaussian_tail = sqrt(
            2.0 * self.amplitude_variance * log(1.0 / self.tail_tolerance)
        )
        return float(np.max(np.abs(self._centers)) + gaussian_tail)

    def wavefunction(self, q: ArrayLike) -> float | NDArray[np.float64]:
        """计算 normalized real position wavefunction；支持任意 shape 与标量。"""

        coordinates = np.asarray(q, dtype=np.float64)
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("q must contain only finite values")
        flat = coordinates.reshape(-1)
        values = np.zeros_like(flat)
        for center, coefficient in zip(self._centers, self._coefficients):
            residual = flat - center
            values += coefficient * np.exp(
                -0.5 * residual * residual / self.amplitude_variance
            )
        values = values.reshape(coordinates.shape)
        return float(values.item()) if coordinates.ndim == 0 else values

    def probability_density(self, q: ArrayLike) -> float | NDArray[np.float64]:
        wavefunction = np.asarray(self.wavefunction(q), dtype=np.float64)
        density = wavefunction * wavefunction
        return float(density.item()) if density.ndim == 0 else density

    def inner_product(self, other: "FiniteEnergyGKPState") -> float:
        """解析计算两个截断 Gaussian-comb wavefunctions 的实 inner product。"""

        if not isinstance(other, FiniteEnergyGKPState):
            raise TypeError("other must be a FiniteEnergyGKPState")
        overlap = _gaussian_cross_integral(
            self._centers,
            self.amplitude_variance,
            other._centers,
            other.amplitude_variance,
        )
        return float(self._coefficients @ overlap @ other._coefficients)

    def syndrome_distribution(self, points: int = 1024) -> SyndromeDistribution:
        """把 normalized q density fold 到一个半开 correction cell。

        使用 midpoint grid，因此 normalization 采用 periodic midpoint quadrature；
        ``captured_mass`` 是归一化前的离散质量，用于暴露 alias/grid 截断误差。
        """

        if not isinstance(points, int) or points < 64:
            raise ValueError("points must be an integer >= 64")
        spacing = self.lattice
        step = spacing / points
        syndrome = (-0.5 + (np.arange(points) + 0.5) / points) * spacing
        alias_radius = int(ceil((self.support_radius + spacing / 2.0) / spacing)) + 1
        raw_density = np.zeros(points, dtype=np.float64)
        for alias in range(-alias_radius, alias_radius + 1):
            raw_density += np.asarray(
                self.probability_density(syndrome + alias * spacing),
                dtype=np.float64,
            )
        captured_mass = float(np.sum(raw_density) * step)
        if not isfinite(captured_mass) or captured_mass <= 0.0:
            raise RuntimeError("syndrome distribution has non-finite or zero mass")
        density = raw_density / captured_mass
        return SyndromeDistribution(
            syndrome=syndrome,
            density=density,
            raw_density=raw_density,
            captured_mass=captured_mass,
            alias_min=-alias_radius,
            alias_max=alias_radius,
        )

    def wigner_like_grid(
        self,
        q_points: int = 513,
        q_range: tuple[float, float] | None = None,
    ) -> WignerLikeGrid:
        r"""从 sampled wavefunction 计算 signed Wigner-integral approximation。

        采用 ``W(q,p)=1/pi int dy exp(-2ipy) psi(q+y)psi(q-y)`` 的离散 FFT。
        p-grid 由 q spacing 唯一决定；不插值到任意 p-grid，以保留离散 q marginal
        恒等式。要求奇数 q-points，使 ``y=0`` 有唯一中心 sample。
        """

        if (
            not isinstance(q_points, int)
            or q_points < 33
            or q_points > _MAX_WIGNER_POINTS
            or q_points % 2 == 0
        ):
            raise ValueError("q_points must be an odd integer in [33, 2049]")
        if q_range is None:
            lower, upper = -self.support_radius, self.support_radius
        else:
            try:
                endpoint_count = len(q_range)
            except TypeError as exc:
                raise ValueError("q_range must contain exactly two endpoints") from exc
            if endpoint_count != 2:
                raise ValueError("q_range must contain exactly two endpoints")
            lower, upper = float(q_range[0]), float(q_range[1])
            if not isfinite(lower) or not isfinite(upper) or not lower < upper:
                raise ValueError("q_range endpoints must be finite and increasing")

        q = np.linspace(lower, upper, q_points, dtype=np.float64)
        step_q = float(q[1] - q[0])
        wavefunction = np.asarray(self.wavefunction(q), dtype=np.float64)
        q_probability = wavefunction * wavefunction
        captured_probability = float(np.sum(q_probability) * step_q)

        center = q_points // 2
        correlations = np.zeros((q_points, q_points), dtype=np.float64)
        for q_index in range(q_points):
            radius = min(q_index, q_points - 1 - q_index, center)
            offsets = np.arange(-radius, radius + 1, dtype=np.int64)
            correlations[q_index, center + offsets] = (
                wavefunction[q_index + offsets] * wavefunction[q_index - offsets]
            )
        spectrum = np.fft.fftshift(
            np.fft.fft(np.fft.ifftshift(correlations, axes=1), axis=1),
            axes=1,
        )
        values = (step_q / pi) * np.real(spectrum)
        p = np.fft.fftshift(pi * np.fft.fftfreq(q_points, d=step_q))
        step_p = float(p[1] - p[0])
        q_marginal = np.sum(values, axis=1) * step_p
        total_mass = float(np.sum(values) * step_q * step_p)
        negative_volume = 0.5 * (
            float(np.sum(np.abs(values)) * step_q * step_p) - total_mass
        )
        return WignerLikeGrid(
            q=q,
            p=p,
            values=values,
            q_probability=q_probability,
            q_marginal=q_marginal,
            captured_probability=captured_probability,
            total_mass=total_mass,
            negative_volume=negative_volume,
        )


def gaussian_envelope_state(
    logical_state: LogicalState,
    peak_sigma: float,
    *,
    envelope_kappa: float | None = None,
    lattice: float = LATTICE_CONST,
    tail_tolerance: float = 1.0e-12,
) -> FiniteEnergyGKPState:
    """构造 independent Gaussian peaks + Gaussian envelope 态。"""

    return FiniteEnergyGKPState(
        logical_state,
        model="gaussian_envelope",
        peak_sigma=peak_sigma,
        envelope_kappa=envelope_kappa,
        lattice=lattice,
        tail_tolerance=tail_tolerance,
    )


def damped_projector_state(
    logical_state: LogicalState,
    projector_delta: float,
    *,
    lattice: float = LATTICE_CONST,
    tail_tolerance: float = 1.0e-12,
) -> FiniteEnergyGKPState:
    r"""构造 ``N_Delta=exp(-Delta^2 n)`` damped-projector 态。"""

    return FiniteEnergyGKPState(
        logical_state,
        model="damped_projector",
        projector_delta=projector_delta,
        lattice=lattice,
        tail_tolerance=tail_tolerance,
    )


__all__ = [
    "LogicalState",
    "FiniteEnergyModel",
    "PeakTable",
    "SyndromeDistribution",
    "WignerLikeGrid",
    "FiniteEnergyGKPState",
    "gaussian_envelope_state",
    "damped_projector_state",
]
