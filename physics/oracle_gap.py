"""Static/dual/oracle logical-failure gap 与 paired uncertainty metrics。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import NormalDist
from typing import Literal, Optional

import numpy as np
from numpy.typing import ArrayLike


DenominatorStatus = Literal["positive", "zero", "inverted"]
BracketStatus = Literal[
    "within_oracle_static_bracket",
    "dual_worse_than_static",
    "dual_better_than_oracle",
    "zero_reference_gap",
    "reference_order_inverted",
]


@dataclass(frozen=True)
class OracleGapPointEstimate:
    static_error_rate: float
    dual_error_rate: float
    oracle_error_rate: float
    static_oracle_gap: float
    dual_oracle_gap: float
    absolute_improvement: float
    gap_remaining_ratio: Optional[float]
    gap_closed_fraction: Optional[float]
    denominator_status: DenominatorStatus
    bracket_status: BracketStatus
    reference_order_valid: bool
    flags: tuple[str, ...]


@dataclass(frozen=True)
class PairedDifferenceInterval:
    estimate: float
    standard_error: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class OracleGapMetrics:
    point: OracleGapPointEstimate
    n_samples: int
    static_minus_dual: PairedDifferenceInterval
    static_minus_oracle: PairedDifferenceInterval
    dual_minus_oracle: PairedDifferenceInterval
    static_only_failure_count: int
    dual_only_failure_count: int
    mcnemar_z: float
    confidence_level: float
    gap_remaining_ci: Optional[tuple[float, float]]
    gap_closed_ci: Optional[tuple[float, float]]
    denominator_stable: bool
    ratio_ci_reliable: bool
    bootstrap_replicates: int
    bootstrap_valid_replicates: int
    joint_outcome_counts: tuple[int, int, int, int, int, int, int, int]
    seed: int
    evidence_scope: str = "paired_logical_failure_metric"

    @property
    def gap_remaining_ratio(self) -> Optional[float]:
        return self.point.gap_remaining_ratio

    @property
    def gap_closed_fraction(self) -> Optional[float]:
        return self.point.gap_closed_fraction


def _probability(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and lie in [0, 1]")
    return result


def _epsilon(value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("denominator_epsilon must be finite and nonnegative")
    return result


def oracle_gap_from_rates(
    static_error_rate: float,
    dual_error_rate: float,
    oracle_error_rate: float,
    *,
    denominator_epsilon: float = 1.0e-12,
) -> OracleGapPointEstimate:
    """由三个同口径 error rates 计算未截断 oracle-gap 点估计。

    ``gap_remaining_ratio`` 对应任务板公式
    ``(P_dual-P_oracle)/(P_static-P_oracle)``；``gap_closed_fraction`` 等于
    ``1-gap_remaining_ratio``。除分母近零外，异常顺序仍返回原始比值并附 flags，
    不 clip 到 ``[0,1]``。
    """

    static = _probability("static_error_rate", static_error_rate)
    dual = _probability("dual_error_rate", dual_error_rate)
    oracle = _probability("oracle_error_rate", oracle_error_rate)
    epsilon = _epsilon(denominator_epsilon)
    static_oracle_gap = static - oracle
    dual_oracle_gap = dual - oracle
    improvement = static - dual
    flags: list[str] = []

    if abs(static_oracle_gap) <= epsilon:
        denominator_status: DenominatorStatus = "zero"
        bracket_status: BracketStatus = "zero_reference_gap"
        remaining = None
        closed = None
        flags.append("zero_static_oracle_gap")
    else:
        remaining = dual_oracle_gap / static_oracle_gap
        closed = improvement / static_oracle_gap
        if static_oracle_gap < -epsilon:
            denominator_status = "inverted"
            bracket_status = "reference_order_inverted"
            flags.append("oracle_worse_than_static")
        else:
            denominator_status = "positive"
            if dual > static + epsilon:
                bracket_status = "dual_worse_than_static"
                flags.append("dual_worse_than_static")
            elif dual < oracle - epsilon:
                bracket_status = "dual_better_than_oracle"
                flags.append("dual_better_than_oracle_point_estimate")
            else:
                bracket_status = "within_oracle_static_bracket"
                flags.append("within_oracle_static_bracket")

    if remaining is not None and closed is not None:
        if not math.isclose(remaining + closed, 1.0, rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise ArithmeticError("oracle-gap ratio identity was violated")

    return OracleGapPointEstimate(
        static_error_rate=static,
        dual_error_rate=dual,
        oracle_error_rate=oracle,
        static_oracle_gap=static_oracle_gap,
        dual_oracle_gap=dual_oracle_gap,
        absolute_improvement=improvement,
        gap_remaining_ratio=remaining,
        gap_closed_fraction=closed,
        denominator_status=denominator_status,
        bracket_status=bracket_status,
        reference_order_valid=denominator_status == "positive",
        flags=tuple(flags),
    )


def _binary_failures(name: str, values: ArrayLike) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size < 2:
        raise ValueError(f"{name} must contain at least two paired samples")
    if array.dtype == np.bool_:
        return array.astype(bool, copy=True)
    try:
        numeric = np.asarray(array, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain only boolean or 0/1 values") from exc
    if not np.all(np.isfinite(numeric)) or not np.all((numeric == 0.0) | (numeric == 1.0)):
        raise ValueError(f"{name} must contain only boolean or 0/1 values")
    return numeric.astype(bool)


def _nonnegative_integer(name: str, value: int, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0 or result > maximum:
        raise ValueError(f"{name} must lie in [0, {maximum}]")
    return result


def _paired_interval(
    first: np.ndarray,
    second: np.ndarray,
    z_value: float,
) -> PairedDifferenceInterval:
    difference = first.astype(float) - second.astype(float)
    estimate = float(np.mean(difference))
    standard_error = float(np.std(difference, ddof=1) / math.sqrt(difference.size))
    margin = z_value * standard_error
    return PairedDifferenceInterval(
        estimate=estimate,
        standard_error=standard_error,
        ci_low=estimate - margin,
        ci_high=estimate + margin,
    )


def compute_oracle_gap_metrics(
    static_failures: ArrayLike,
    dual_failures: ArrayLike,
    oracle_failures: ArrayLike,
    *,
    confidence_level: float = 0.95,
    bootstrap_replicates: int = 4_000,
    seed: int = 20260714,
    denominator_epsilon: float = 1.0e-12,
) -> OracleGapMetrics:
    """从同一 paired sample set 计算 oracle gap、差值 CI 与 ratio bootstrap CI。

    bootstrap 对八种 ``(static, dual, oracle)`` 联合 outcome 做 multinomial resampling，
    与逐样本 paired bootstrap 等价，但避免分配 ``B x N`` index 矩阵。
    """

    static = _binary_failures("static_failures", static_failures)
    dual = _binary_failures("dual_failures", dual_failures)
    oracle = _binary_failures("oracle_failures", oracle_failures)
    if static.shape != dual.shape or static.shape != oracle.shape:
        raise ValueError("all failure arrays must have the same length")
    confidence = float(confidence_level)
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must lie strictly between 0 and 1")
    replicates = _nonnegative_integer(
        "bootstrap_replicates",
        bootstrap_replicates,
        maximum=1_000_000,
    )
    validated_seed = _nonnegative_integer("seed", seed, maximum=2**64 - 1)
    epsilon = _epsilon(denominator_epsilon)
    z_value = NormalDist().inv_cdf(0.5 + confidence / 2.0)

    point = oracle_gap_from_rates(
        float(np.mean(static)),
        float(np.mean(dual)),
        float(np.mean(oracle)),
        denominator_epsilon=epsilon,
    )
    static_dual = _paired_interval(static, dual, z_value)
    static_oracle = _paired_interval(static, oracle, z_value)
    dual_oracle = _paired_interval(dual, oracle, z_value)
    denominator_stable = static_oracle.ci_low > epsilon

    static_only = int(np.sum(static & ~dual))
    dual_only = int(np.sum(~static & dual))
    discordant = static_only + dual_only
    mcnemar_z = (
        (static_only - dual_only) / math.sqrt(discordant)
        if discordant > 0
        else 0.0
    )

    codes = static.astype(np.int64) * 4 + dual.astype(np.int64) * 2 + oracle.astype(np.int64)
    counts = np.bincount(codes, minlength=8).astype(np.int64)
    count_tuple = tuple(int(value) for value in counts)
    remaining_ci: Optional[tuple[float, float]] = None
    closed_ci: Optional[tuple[float, float]] = None
    valid_replicates = 0

    if replicates > 0 and point.reference_order_valid:
        probabilities = counts.astype(float) / static.size
        rng = np.random.default_rng(validated_seed)
        bootstrap_counts = rng.multinomial(static.size, probabilities, size=replicates)
        code_values = np.arange(8, dtype=np.int64)
        static_mask = ((code_values >> 2) & 1).astype(float)
        dual_mask = ((code_values >> 1) & 1).astype(float)
        oracle_mask = (code_values & 1).astype(float)
        static_rates = bootstrap_counts @ static_mask / static.size
        dual_rates = bootstrap_counts @ dual_mask / static.size
        oracle_rates = bootstrap_counts @ oracle_mask / static.size
        denominator = static_rates - oracle_rates
        valid = denominator > epsilon
        valid_replicates = int(np.sum(valid))
        minimum_valid = max(100, int(math.ceil(0.5 * replicates)))
        if valid_replicates >= minimum_valid:
            remaining_values = (dual_rates[valid] - oracle_rates[valid]) / denominator[valid]
            closed_values = (static_rates[valid] - dual_rates[valid]) / denominator[valid]
            alpha = (1.0 - confidence) / 2.0
            remaining_bounds = np.quantile(remaining_values, [alpha, 1.0 - alpha])
            closed_bounds = np.quantile(closed_values, [alpha, 1.0 - alpha])
            remaining_ci = (float(remaining_bounds[0]), float(remaining_bounds[1]))
            closed_ci = (float(closed_bounds[0]), float(closed_bounds[1]))

    ratio_ci_reliable = (
        denominator_stable
        and replicates > 0
        and valid_replicates >= int(math.ceil(0.95 * replicates))
        and remaining_ci is not None
        and closed_ci is not None
    )

    return OracleGapMetrics(
        point=point,
        n_samples=int(static.size),
        static_minus_dual=static_dual,
        static_minus_oracle=static_oracle,
        dual_minus_oracle=dual_oracle,
        static_only_failure_count=static_only,
        dual_only_failure_count=dual_only,
        mcnemar_z=mcnemar_z,
        confidence_level=confidence,
        gap_remaining_ci=remaining_ci,
        gap_closed_ci=closed_ci,
        denominator_stable=denominator_stable,
        ratio_ci_reliable=ratio_ci_reliable,
        bootstrap_replicates=replicates,
        bootstrap_valid_replicates=valid_replicates,
        joint_outcome_counts=count_tuple,  # type: ignore[arg-type]
        seed=validated_seed,
    )


__all__ = [
    "DenominatorStatus",
    "BracketStatus",
    "OracleGapPointEstimate",
    "PairedDifferenceInterval",
    "OracleGapMetrics",
    "oracle_gap_from_rates",
    "compute_oracle_gap_metrics",
]
