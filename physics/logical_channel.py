"""Parity-output decoder 的 Pauli-twirled effective logical-channel metrics。

映射约定：残余 q parity 对应 logical X，残余 p parity 对应 logical Z，两者同时发生
对应 Y（全局相位忽略）。本模块输出的是由 classical parity confusion 得到的 qubit
Pauli channel；它不保留 coherent/non-Pauli/leakage 信息，也不是 Fock-space recovery
tomography。

Finite-energy 路径从 ``FiniteEnergyGKPState`` 的 normalized position density 出发，
可与外加 Gaussian displacement 做解析 Gaussian-mixture convolution，再按 alias parity
fold 到 syndrome cell。它因此不等同于只检查一次 ``abs(x)>lattice/2``。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite, sqrt
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .finite_energy_gkp import FiniteEnergyGKPState


_MAX_RESPONSE_POINTS = 65_536
_MAX_RESPONSE_ALIASES = 10_001
_MAX_DENSITY_PAIR_EVALUATIONS = 100_000_000


@dataclass(frozen=True)
class PauliChannel:
    """Pauli order ``I,X,Y,Z`` 的 normalized qubit channel。"""

    p_i: float
    p_x: float
    p_y: float
    p_z: float
    source: str = "unspecified"

    def __post_init__(self) -> None:
        values = np.array([self.p_i, self.p_x, self.p_y, self.p_z], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("Pauli probabilities must be finite")
        tolerance = 1.0e-12
        if np.any(values < -tolerance) or np.any(values > 1.0 + tolerance):
            raise ValueError("Pauli probabilities must lie in [0, 1]")
        with np.errstate(over="ignore", invalid="ignore"):
            total = float(np.sum(values))
        if not isfinite(total):
            raise ValueError("Pauli probability sum must be finite")
        if abs(total - 1.0) > tolerance:
            raise ValueError("Pauli probabilities must sum to 1")
        values = np.clip(values, 0.0, 1.0) / total
        object.__setattr__(self, "p_i", float(values[0]))
        object.__setattr__(self, "p_x", float(values[1]))
        object.__setattr__(self, "p_y", float(values[2]))
        object.__setattr__(self, "p_z", float(values[3]))

    @property
    def probabilities(self) -> NDArray[np.float64]:
        return np.array([self.p_i, self.p_x, self.p_y, self.p_z], dtype=np.float64)

    @property
    def logical_error_rate(self) -> float:
        return 1.0 - self.p_i

    @property
    def entanglement_fidelity(self) -> float:
        """相对于 identity target 的 ``F_e``；Pauli channel 下等于 ``p_I``。"""

        return self.p_i

    @property
    def average_fidelity(self) -> float:
        """Qubit identity-target average fidelity ``F_avg=(2 F_e+1)/3``。"""

        return (2.0 * self.entanglement_fidelity + 1.0) / 3.0

    @property
    def ptm(self) -> NDArray[np.float64]:
        """Pauli basis ``(I,X,Y,Z)`` 下的 4x4 Pauli transfer matrix。"""

        lambda_x = self.p_i + self.p_x - self.p_y - self.p_z
        lambda_y = self.p_i - self.p_x + self.p_y - self.p_z
        lambda_z = self.p_i - self.p_x - self.p_y + self.p_z
        return np.diag([1.0, lambda_x, lambda_y, lambda_z]).astype(np.float64)


@dataclass(frozen=True)
class ParityResponse1D:
    """Finite-energy alias truth 与 syndrome 的联合 density。"""

    syndrome: NDArray[np.float64]
    joint_density: NDArray[np.float64]
    posterior: NDArray[np.float64]
    map_decision: NDArray[np.int64]
    captured_mass: float
    reference_parity: int
    displacement_sigma: float
    alias_min: int
    alias_max: int

    def __post_init__(self) -> None:
        syndrome = np.asarray(self.syndrome, dtype=np.float64)
        joint_density = np.asarray(self.joint_density, dtype=np.float64)
        posterior = np.asarray(self.posterior, dtype=np.float64)
        map_decision = np.asarray(self.map_decision)
        if syndrome.ndim != 1 or syndrome.size < 2:
            raise ValueError("syndrome must be a one-dimensional grid with at least 2 points")
        if not np.all(np.isfinite(syndrome)) or np.any(np.diff(syndrome) <= 0.0):
            raise ValueError("syndrome grid must be finite and strictly increasing")
        steps = np.diff(syndrome)
        if not np.allclose(steps, steps[0], rtol=1.0e-12, atol=1.0e-15):
            raise ValueError("syndrome grid must be uniform")
        if joint_density.shape != (syndrome.size, 2):
            raise ValueError("joint_density must have shape (len(syndrome), 2)")
        if not np.all(np.isfinite(joint_density)) or np.any(joint_density < 0.0):
            raise ValueError("joint_density must be finite and non-negative")
        if posterior.shape != joint_density.shape or not np.all(np.isfinite(posterior)):
            raise ValueError("posterior must be finite and match joint_density shape")
        if np.any(posterior < -1.0e-12) or not np.allclose(
            np.sum(posterior, axis=-1), 1.0, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError("posterior rows must be normalized probabilities")
        map_decision = _validate_decisions(map_decision, syndrome.shape)
        if not isfinite(self.captured_mass) or self.captured_mass <= 0.0:
            raise ValueError("captured_mass must be finite and positive")
        if self.reference_parity not in (0, 1):
            raise ValueError("reference_parity must be 0 or 1")
        if not isfinite(self.displacement_sigma) or self.displacement_sigma < 0.0:
            raise ValueError("displacement_sigma must be finite and non-negative")
        if not isinstance(self.alias_min, int) or not isinstance(self.alias_max, int):
            raise ValueError("alias bounds must be integers")
        if self.alias_min > self.alias_max:
            raise ValueError("alias_min must not exceed alias_max")
        object.__setattr__(self, "syndrome", syndrome)
        object.__setattr__(self, "joint_density", joint_density)
        object.__setattr__(self, "posterior", posterior)
        object.__setattr__(self, "map_decision", map_decision)


@dataclass(frozen=True)
class ParityConfusion:
    """行是真实 residual parity，列是 decoder decision 的 normalized confusion。"""

    matrix: NDArray[np.float64]
    captured_mass: float
    decoder_name: str

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=np.float64)
        if matrix.shape != (2, 2):
            raise ValueError("parity confusion matrix must have shape (2, 2)")
        if not np.all(np.isfinite(matrix)) or np.any(matrix < -1.0e-12):
            raise ValueError("parity confusion entries must be finite and non-negative")
        total = float(np.sum(matrix))
        if abs(total - 1.0) > 1.0e-12:
            raise ValueError("parity confusion matrix must sum to 1")
        normalized = np.clip(matrix, 0.0, 1.0) / total
        object.__setattr__(self, "matrix", normalized)
        if not isfinite(self.captured_mass) or self.captured_mass <= 0.0:
            raise ValueError("captured_mass must be finite and positive")

    @property
    def error_probability(self) -> float:
        return float(self.matrix[0, 1] + self.matrix[1, 0])

    @property
    def truth_distribution(self) -> NDArray[np.float64]:
        return np.sum(self.matrix, axis=1)

    @property
    def decision_distribution(self) -> NDArray[np.float64]:
        return np.sum(self.matrix, axis=0)


def pauli_channel_from_residual_distribution(
    residual_probabilities: ArrayLike,
    *,
    source: str = "residual_distribution",
) -> PauliChannel:
    """从 ``P(residual_q, residual_p)`` 构造 Pauli channel。

    数组索引 ``[0,0]->I, [1,0]->X, [0,1]->Z, [1,1]->Y``。
    """

    probabilities = np.asarray(residual_probabilities, dtype=np.float64)
    if probabilities.shape != (2, 2):
        raise ValueError("residual_probabilities must have shape (2, 2)")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError("residual probabilities must be finite and non-negative")
    with np.errstate(over="ignore", invalid="ignore"):
        total = float(np.sum(probabilities))
    if not isfinite(total) or total <= 0.0:
        raise ValueError("residual probabilities must have positive finite mass")
    probabilities = probabilities / total
    return PauliChannel(
        p_i=float(probabilities[0, 0]),
        p_x=float(probabilities[1, 0]),
        p_y=float(probabilities[1, 1]),
        p_z=float(probabilities[0, 1]),
        source=source,
    )


def _validate_parity_array(values: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim < 1 or array.shape[-1] != 2:
        raise ValueError(f"{name} must have shape (..., 2)")
    try:
        floating = array.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric binary parities") from exc
    if not np.all(np.isfinite(floating)):
        raise ValueError(f"{name} must contain finite values")
    integer = array.astype(np.int64)
    if not np.all(array == integer) or np.any((integer < 0) | (integer > 1)):
        raise ValueError(f"{name} must contain only binary parities")
    return integer


def pauli_channel_from_parity_samples(
    true_parity: ArrayLike,
    decoded_parity: ArrayLike,
    *,
    weights: ArrayLike | None = None,
    source: str = "parity_samples",
) -> PauliChannel:
    """从 paired truth/decision samples 聚合 residual Pauli probabilities。"""

    truth = _validate_parity_array(true_parity, "true_parity")
    decoded = _validate_parity_array(decoded_parity, "decoded_parity")
    if truth.shape != decoded.shape:
        raise ValueError("true_parity and decoded_parity must have identical shape")
    flat_truth = truth.reshape((-1, 2))
    flat_decoded = decoded.reshape((-1, 2))
    sample_count = flat_truth.shape[0]
    if sample_count == 0:
        raise ValueError("at least one parity sample is required")
    if weights is None:
        sample_weights = np.ones(sample_count, dtype=np.float64)
    else:
        sample_weights = np.asarray(weights, dtype=np.float64)
        try:
            sample_weights = np.broadcast_to(sample_weights, truth.shape[:-1]).reshape(-1)
        except ValueError as exc:
            raise ValueError("weights must broadcast to parity sample shape") from exc
        if not np.all(np.isfinite(sample_weights)) or np.any(sample_weights < 0.0):
            raise ValueError("weights must be finite and non-negative")
    with np.errstate(over="ignore", invalid="ignore"):
        total_weight = float(np.sum(sample_weights))
    if not isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("weights must have positive finite mass")

    residual = np.bitwise_xor(flat_truth, flat_decoded)
    distribution = np.zeros((2, 2), dtype=np.float64)
    for q_parity in (0, 1):
        for p_parity in (0, 1):
            mask = (residual[:, 0] == q_parity) & (residual[:, 1] == p_parity)
            distribution[q_parity, p_parity] = float(np.sum(sample_weights[mask]))
    return pauli_channel_from_residual_distribution(distribution, source=source)


def pauli_channel_from_joint_confusion(
    confusion: ArrayLike,
    *,
    source: str = "joint_confusion",
) -> PauliChannel:
    """从四类 truth x decision confusion 聚合 residual Pauli channel。

    四类编码与 decoder 一致：``class=2*parity_q+parity_p``。
    """

    matrix = np.asarray(confusion, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError("joint confusion must have shape (4, 4)")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError("joint confusion entries must be finite and non-negative")
    with np.errstate(over="ignore", invalid="ignore"):
        total = float(np.sum(matrix))
    if not isfinite(total) or total <= 0.0:
        raise ValueError("joint confusion must have positive finite mass")
    residual = np.zeros((2, 2), dtype=np.float64)
    for true_class in range(4):
        true_q, true_p = divmod(true_class, 2)
        for decoded_class in range(4):
            decoded_q, decoded_p = divmod(decoded_class, 2)
            residual[true_q ^ decoded_q, true_p ^ decoded_p] += matrix[
                true_class, decoded_class
            ]
    return pauli_channel_from_residual_distribution(residual, source=source)


def pauli_channel_from_axis_confusions(
    q_confusion: ParityConfusion,
    p_confusion: ParityConfusion,
    *,
    source: str = "independent_axis_confusions",
) -> PauliChannel:
    """在显式 independent-axis 假设下组合两个 residual error rates。"""

    if not isinstance(q_confusion, ParityConfusion) or not isinstance(
        p_confusion, ParityConfusion
    ):
        raise TypeError("q_confusion and p_confusion must be ParityConfusion instances")
    error_q = q_confusion.error_probability
    error_p = p_confusion.error_probability
    residual = np.array(
        [
            [(1.0 - error_q) * (1.0 - error_p), (1.0 - error_q) * error_p],
            [error_q * (1.0 - error_p), error_q * error_p],
        ],
        dtype=np.float64,
    )
    return pauli_channel_from_residual_distribution(residual, source=source)


def _convolved_state_density(
    state: FiniteEnergyGKPState,
    coordinates: np.ndarray,
    displacement_sigma: float,
) -> np.ndarray:
    if displacement_sigma == 0.0:
        return np.asarray(state.probability_density(coordinates), dtype=np.float64)

    table = state.peak_table
    centers = table.centers
    coefficients = table.coefficients
    amplitude_variance = table.amplitude_variance
    pair_centers = 0.5 * (
        centers[:, np.newaxis] + centers[np.newaxis, :]
    ).reshape(-1)
    differences = centers[:, np.newaxis] - centers[np.newaxis, :]
    pair_weights = (
        coefficients[:, np.newaxis]
        * coefficients[np.newaxis, :]
        * np.exp(-differences * differences / (4.0 * amplitude_variance))
    ).reshape(-1)
    intrinsic_variance = amplitude_variance / 2.0
    broadened_variance = intrinsic_variance + displacement_sigma**2
    pair_weights = pair_weights * sqrt(intrinsic_variance / broadened_variance)

    flat = coordinates.reshape(-1)
    if pair_centers.size > 0 and flat.size > (
        _MAX_DENSITY_PAIR_EVALUATIONS // pair_centers.size
    ):
        raise ValueError(
            "finite-energy density workload exceeds the safety limit; reduce points/noise support"
        )
    density = np.empty_like(flat)
    max_pair_evaluations = 2_000_000
    chunk_size = max(1, max_pair_evaluations // pair_centers.size)
    for start in range(0, flat.size, chunk_size):
        chunk = flat[start : start + chunk_size]
        residual = chunk[:, np.newaxis] - pair_centers[np.newaxis, :]
        density[start : start + chunk_size] = np.exp(
            -0.5 * residual * residual / broadened_variance
        ) @ pair_weights
    return density.reshape(coordinates.shape)


def finite_energy_parity_response_1d(
    state: FiniteEnergyGKPState,
    *,
    displacement_sigma: float = 0.0,
    points: int = 2048,
    tail_sigma: float = 10.0,
) -> ParityResponse1D:
    """从 finite-energy state/noise density 构造 residual-parity soft response。

    只接受 logical ``0/1`` position-basis state；其 nominal lattice parity 被扣除，
    因此 truth=1 表示相对于目标 basis comb 的残余 logical flip。``+/-`` 在 position
    basis 没有单一 nominal parity，必须在更高保真度 channel simulation 中处理。
    """

    if not isinstance(state, FiniteEnergyGKPState):
        raise TypeError("state must be a FiniteEnergyGKPState")
    if state.logical_state not in {"0", "1"}:
        raise ValueError("finite-energy parity response requires logical state '0' or '1'")
    sigma = float(displacement_sigma)
    if not isfinite(sigma) or sigma < 0.0:
        raise ValueError("displacement_sigma must be finite and non-negative")
    if not isinstance(points, int) or not (128 <= points <= _MAX_RESPONSE_POINTS):
        raise ValueError("points must be an integer in [128, 65536]")
    tail = float(tail_sigma)
    if not isfinite(tail) or tail <= 0.0:
        raise ValueError("tail_sigma must be finite and positive")

    spacing = state.lattice
    step = spacing / points
    syndrome = (-0.5 + (np.arange(points) + 0.5) / points) * spacing
    support = state.support_radius + tail * sigma
    if not isfinite(support):
        raise ValueError("displacement_sigma/tail_sigma produces non-finite support")
    alias_radius = int(ceil((support + spacing / 2.0) / spacing)) + 1
    if 2 * alias_radius + 1 > _MAX_RESPONSE_ALIASES:
        raise ValueError(
            "finite-energy alias range exceeds the safety limit; reduce displacement_sigma"
        )
    aliases = np.arange(-alias_radius, alias_radius + 1, dtype=np.int64)
    coordinates = syndrome[np.newaxis, :] + aliases[:, np.newaxis] * spacing
    density = _convolved_state_density(state, coordinates, sigma)
    if not np.all(np.isfinite(density)) or np.any(density < -1.0e-12):
        raise RuntimeError("finite-energy folded density is non-finite or negative")
    density = np.maximum(density, 0.0)

    reference_parity = int(state.logical_state)
    residual_parity = np.bitwise_xor(np.mod(aliases, 2), reference_parity)
    joint_density = np.stack(
        (
            np.sum(density[residual_parity == 0], axis=0),
            np.sum(density[residual_parity == 1], axis=0),
        ),
        axis=-1,
    )
    captured_mass = float(np.sum(joint_density) * step)
    if not isfinite(captured_mass) or captured_mass <= 0.0:
        raise RuntimeError("finite-energy parity response has zero or non-finite mass")
    evidence = np.sum(joint_density, axis=-1)
    if np.any(evidence <= 0.0):
        raise RuntimeError("finite-energy parity response has zero-density syndrome bins")
    posterior = joint_density / evidence[:, np.newaxis]
    map_decision = (joint_density[:, 1] > joint_density[:, 0]).astype(np.int64)
    return ParityResponse1D(
        syndrome=syndrome,
        joint_density=joint_density,
        posterior=posterior,
        map_decision=map_decision,
        captured_mass=captured_mass,
        reference_parity=reference_parity,
        displacement_sigma=sigma,
        alias_min=-alias_radius,
        alias_max=alias_radius,
    )


def _validate_decisions(decisions: ArrayLike, shape: tuple[int, ...]) -> np.ndarray:
    values = np.asarray(decisions)
    if values.shape != shape:
        raise ValueError("decoder decisions must match syndrome grid shape")
    try:
        finite_values = values.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("decoder decisions must be binary") from exc
    if not np.all(np.isfinite(finite_values)):
        raise ValueError("decoder decisions must be finite")
    integer = values.astype(np.int64)
    if not np.all(values == integer) or np.any((integer < 0) | (integer > 1)):
        raise ValueError("decoder decisions must contain only 0 or 1")
    return integer


def parity_confusion_from_response(
    response: ParityResponse1D,
    decoder: Callable[[NDArray[np.float64]], ArrayLike] | ArrayLike | None = None,
    *,
    decoder_name: str = "finite_energy_map",
) -> ParityConfusion:
    """评估任意 deterministic parity-output decoder；``None`` 使用 response MAP。"""

    if not isinstance(response, ParityResponse1D):
        raise TypeError("response must be a ParityResponse1D")
    if decoder is None:
        decisions = response.map_decision
    elif callable(decoder):
        decisions = decoder(response.syndrome.copy())
    else:
        decisions = decoder
    decisions = _validate_decisions(decisions, response.syndrome.shape)
    step = float(response.syndrome[1] - response.syndrome[0])
    matrix = np.zeros((2, 2), dtype=np.float64)
    for truth in (0, 1):
        for decision in (0, 1):
            matrix[truth, decision] = float(
                np.sum(response.joint_density[decisions == decision, truth]) * step
            )
    matrix /= float(np.sum(matrix))
    return ParityConfusion(
        matrix=matrix,
        captured_mass=response.captured_mass,
        decoder_name=decoder_name,
    )


__all__ = [
    "PauliChannel",
    "ParityResponse1D",
    "ParityConfusion",
    "pauli_channel_from_residual_distribution",
    "pauli_channel_from_parity_samples",
    "pauli_channel_from_joint_confusion",
    "pauli_channel_from_axis_confusions",
    "finite_energy_parity_response_1d",
    "parity_confusion_from_response",
]
