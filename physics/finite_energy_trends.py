"""Finite-energy syndrome-level shrinkage trend reproduction harness。

Effective model：data displacement ``x`` 经过 finite-energy syndrome noise ``n_Delta``
后得到 ``s = wrap(x+n_Delta)``；correction action 为 ``c=g*s``，残余 data shift 为
``r=x-c``。standard decoder 使用 ``g=1``；optimized shrinkage 在独立训练样本上
最小化 residual MSE，再在 held-out paired samples 上报告 MSE 与 logical parity error。

``n_Delta`` 的 variance 直接来自 T1.2.1 damped-projector state；canonical
``[x,p]=i`` 中为 ``tanh(Delta^2)/2``，默认 decoder-standardized axis 中为
``tanh(Delta^2)``。这是明确的 syndrome-level effective model，不是完整
finite-energy recovery/channel simulation。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt

import numpy as np
from numpy.typing import NDArray

from .constants import LATTICE_CONST
from .finite_energy_gkp import damped_projector_state
from .ideal_gkp_decoder import standard_binning_1d


@dataclass(frozen=True)
class ShrinkageTrendConfig:
    """Trend sweep protocol；delta_values 必须按大到小排列。"""

    delta_values: tuple[float, ...] = (0.60, 0.50, 0.40, 0.30, 0.22)
    channel_sigma: float = 0.18 * LATTICE_CONST
    train_samples: int = 120_000
    eval_samples: int = 300_000
    seed: int = 20260714
    lattice: float = LATTICE_CONST
    confidence_z: float = 1.96

    def __post_init__(self) -> None:
        if len(self.delta_values) < 3:
            raise ValueError("delta_values must contain at least three points")
        deltas = tuple(float(value) for value in self.delta_values)
        if not all(isfinite(value) and value > 0.0 for value in deltas):
            raise ValueError("all delta_values must be finite and positive")
        if not all(isfinite(value * value) and value * value > 0.0 for value in deltas):
            raise ValueError("all delta_values squared must be representable")
        if not all(left > right for left, right in zip(deltas, deltas[1:])):
            raise ValueError("delta_values must be strictly decreasing")
        if not isfinite(self.channel_sigma) or self.channel_sigma <= 0.0:
            raise ValueError("channel_sigma must be finite and positive")
        if not isfinite(self.channel_sigma * self.channel_sigma):
            raise ValueError("channel_sigma squared must be representable")
        if not isfinite(self.lattice) or self.lattice <= 0.0:
            raise ValueError("lattice must be finite and positive")
        if not isinstance(self.train_samples, int) or self.train_samples < 10_000:
            raise ValueError("train_samples must be an integer >= 10000")
        if not isinstance(self.eval_samples, int) or self.eval_samples < 10_000:
            raise ValueError("eval_samples must be an integer >= 10000")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not isfinite(self.confidence_z) or self.confidence_z <= 0.0:
            raise ValueError("confidence_z must be finite and positive")
        object.__setattr__(self, "delta_values", deltas)


@dataclass(frozen=True)
class ShrinkageTrendPoint:
    delta: float
    intrinsic_sigma: float
    unwrapped_mmse_gain: float
    fitted_gain: float
    standard_mse: float
    shrinkage_mse: float
    standard_logical_error: float
    shrinkage_logical_error: float
    absolute_logical_gain: float
    relative_logical_reduction: float
    standard_only_wrong: int
    shrinkage_only_wrong: int
    paired_standard_error: float
    gain_ci_low: float
    gain_ci_high: float
    mcnemar_z: float


@dataclass(frozen=True)
class ShrinkageTrendResult:
    config: ShrinkageTrendConfig
    points: tuple[ShrinkageTrendPoint, ...]
    fitted_gain_increases_as_delta_decreases: bool
    logical_advantage_shrinks_as_delta_decreases: bool
    mse_advantage_shrinks_as_delta_decreases: bool
    all_shrinkage_mse_not_worse: bool
    evidence_scope: str = "syndrome_level_effective_model"


def _centered_wrap(values: np.ndarray, lattice: float) -> np.ndarray:
    index = np.floor(values / lattice + 0.5)
    return values - index * lattice


def _logical_flip_mask(residual: np.ndarray, lattice: float) -> NDArray[np.bool_]:
    result = standard_binning_1d(residual, lattice=lattice)
    return np.asarray(result.logical_flip, dtype=bool)


def _fit_shrinkage_gain(
    displacement: np.ndarray,
    syndrome: np.ndarray,
) -> float:
    denominator = float(np.dot(syndrome, syndrome))
    if not isfinite(denominator) or denominator <= 0.0:
        raise RuntimeError("training syndrome has zero or non-finite energy")
    numerator = float(np.dot(displacement, syndrome))
    if not isfinite(numerator):
        raise RuntimeError("training displacement-syndrome covariance is non-finite")
    # 任务定义是 shrinkage，不允许把 train noise 拟合成过校正 gain>1 或符号翻转。
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def _nondecreasing(values: list[float], tolerance: float = 1.0e-12) -> bool:
    return all(right + tolerance >= left for left, right in zip(values, values[1:]))


def _nonincreasing(values: list[float], tolerance: float = 1.0e-12) -> bool:
    return all(right <= left + tolerance for left, right in zip(values, values[1:]))


def run_finite_energy_shrinkage_trend(
    config: ShrinkageTrendConfig | None = None,
) -> ShrinkageTrendResult:
    """运行 train/eval 分离、跨 delta common-random-number 的趋势 sweep。"""

    protocol = ShrinkageTrendConfig() if config is None else config
    if not isinstance(protocol, ShrinkageTrendConfig):
        raise TypeError("config must be a ShrinkageTrendConfig")

    rng = np.random.default_rng(protocol.seed)
    train_displacement = rng.normal(
        0.0,
        protocol.channel_sigma,
        size=protocol.train_samples,
    )
    train_standard_normal = rng.normal(0.0, 1.0, size=protocol.train_samples)
    eval_displacement = rng.normal(
        0.0,
        protocol.channel_sigma,
        size=protocol.eval_samples,
    )
    eval_standard_normal = rng.normal(0.0, 1.0, size=protocol.eval_samples)

    points: list[ShrinkageTrendPoint] = []
    for delta in protocol.delta_values:
        state = damped_projector_state(
            "0",
            delta,
            lattice=protocol.lattice,
            tail_tolerance=1.0e-10,
        )
        intrinsic_sigma = sqrt(state.amplitude_variance / 2.0)
        unwrapped_mmse_gain = protocol.channel_sigma**2 / (
            protocol.channel_sigma**2 + intrinsic_sigma**2
        )

        train_observed = train_displacement + intrinsic_sigma * train_standard_normal
        train_syndrome = _centered_wrap(train_observed, protocol.lattice)
        fitted_gain = _fit_shrinkage_gain(train_displacement, train_syndrome)

        eval_observed = eval_displacement + intrinsic_sigma * eval_standard_normal
        eval_syndrome = _centered_wrap(eval_observed, protocol.lattice)
        standard_residual = eval_displacement - eval_syndrome
        shrinkage_residual = eval_displacement - fitted_gain * eval_syndrome
        standard_mse = float(np.mean(standard_residual * standard_residual))
        shrinkage_mse = float(np.mean(shrinkage_residual * shrinkage_residual))
        standard_wrong = _logical_flip_mask(standard_residual, protocol.lattice)
        shrinkage_wrong = _logical_flip_mask(shrinkage_residual, protocol.lattice)
        standard_logical_error = float(np.mean(standard_wrong))
        shrinkage_logical_error = float(np.mean(shrinkage_wrong))
        paired_difference = standard_wrong.astype(np.float64) - shrinkage_wrong.astype(
            np.float64
        )
        absolute_gain = float(np.mean(paired_difference))
        paired_standard_error = float(
            np.std(paired_difference, ddof=1) / sqrt(protocol.eval_samples)
        )
        standard_only_wrong = int(np.sum(standard_wrong & ~shrinkage_wrong))
        shrinkage_only_wrong = int(np.sum(~standard_wrong & shrinkage_wrong))
        discordant = standard_only_wrong + shrinkage_only_wrong
        mcnemar_z = (
            (standard_only_wrong - shrinkage_only_wrong) / sqrt(discordant)
            if discordant > 0
            else 0.0
        )
        relative_reduction = (
            absolute_gain / standard_logical_error
            if standard_logical_error > 0.0
            else 0.0
        )
        points.append(
            ShrinkageTrendPoint(
                delta=delta,
                intrinsic_sigma=intrinsic_sigma,
                unwrapped_mmse_gain=unwrapped_mmse_gain,
                fitted_gain=fitted_gain,
                standard_mse=standard_mse,
                shrinkage_mse=shrinkage_mse,
                standard_logical_error=standard_logical_error,
                shrinkage_logical_error=shrinkage_logical_error,
                absolute_logical_gain=absolute_gain,
                relative_logical_reduction=relative_reduction,
                standard_only_wrong=standard_only_wrong,
                shrinkage_only_wrong=shrinkage_only_wrong,
                paired_standard_error=paired_standard_error,
                gain_ci_low=absolute_gain
                - protocol.confidence_z * paired_standard_error,
                gain_ci_high=absolute_gain
                + protocol.confidence_z * paired_standard_error,
                mcnemar_z=mcnemar_z,
            )
        )

    fitted_gains = [point.fitted_gain for point in points]
    logical_gains = [point.absolute_logical_gain for point in points]
    mse_gains = [point.standard_mse - point.shrinkage_mse for point in points]
    return ShrinkageTrendResult(
        config=protocol,
        points=tuple(points),
        fitted_gain_increases_as_delta_decreases=_nondecreasing(fitted_gains),
        logical_advantage_shrinks_as_delta_decreases=_nonincreasing(logical_gains),
        mse_advantage_shrinks_as_delta_decreases=_nonincreasing(mse_gains),
        all_shrinkage_mse_not_worse=all(
            point.shrinkage_mse <= point.standard_mse for point in points
        ),
    )


__all__ = [
    "ShrinkageTrendConfig",
    "ShrinkageTrendPoint",
    "ShrinkageTrendResult",
    "run_finite_energy_shrinkage_trend",
]
