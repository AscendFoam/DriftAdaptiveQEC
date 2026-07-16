"""T2.1.3 向量化多轨迹 Monte Carlo 与稀有事件分层抽样。

核心按 round 循环、在独立 trajectories 维度向量化，保留 T2.1.1 的 residual、
recovery-depth、leakage persistence 与 logical-parity 语义。rare mode 把“每条轨迹
是否含一个额外 burst/leakage episode”定义为已知概率的两层 mixture，分别估计
conditional rates 后按真实层权重组合；allocation 只影响方差，不改变 estimand。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Sequence

import numpy as np

from .constants import LATTICE_CONST
from .drift_processes import DriftProcess, DriftState
from .sbs_error_space import SBS_PROTOCOL_ID


MODEL_SCOPE = "vectorized_multitrajectory_syndrome_level_monte_carlo_not_device_calibrated"
RARE_EVENT_KINDS = ("burst", "leakage", "burst_and_leakage")


def _integer(value: int, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _probability(value: float, name: str, *, strict: bool = False) -> float:
    result = _finite(value, name)
    valid = 0.0 < result < 1.0 if strict else 0.0 <= result <= 1.0
    if not valid:
        interval = "(0, 1)" if strict else "[0, 1]"
        raise ValueError(f"{name} must lie in {interval}")
    return result


def _pair(values: Sequence[float], name: str) -> tuple[float, float]:
    if isinstance(values, (str, bytes)) or len(values) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    return _finite(values[0], f"{name}[0]"), _finite(values[1], f"{name}[1]")


@dataclass(frozen=True)
class FastMonteCarloConfig:
    n_trajectories: int = 1_000
    rounds_per_trajectory: int = 100
    lattice: float = LATTICE_CONST
    loss_environment_variance: float = 0.5
    max_recovery_depth: int = 6
    depth_probability_scale: float = 0.25
    depth_probability_power: float = 2.0
    recovery_probability: float = 0.88
    recovery_gain: float = 0.5
    base_leakage_probability: float = 1.0e-4
    loss_leakage_scale: float = 0.01
    burst_leakage_bonus: float = 0.01
    higher_leakage_fraction: float = 0.2
    higher_leakage_mean_duration: float = 10.0
    leakage_logical_fault_probability: float = 0.0
    confidence_level: float = 0.95
    bootstrap_replicates: int = 500
    seed: int = 2026071413
    model_scope: str = MODEL_SCOPE

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "n_trajectories",
            _integer(self.n_trajectories, "n_trajectories", 4),
        )
        object.__setattr__(
            self,
            "rounds_per_trajectory",
            _integer(self.rounds_per_trajectory, "rounds_per_trajectory", 1),
        )
        if self.total_cycles > 100_000_000:
            raise ValueError("total_cycles must not exceed 100,000,000")
        lattice = _finite(self.lattice, "lattice")
        if lattice <= 0.0:
            raise ValueError("lattice must be positive")
        object.__setattr__(self, "lattice", lattice)
        environment = _finite(self.loss_environment_variance, "loss_environment_variance")
        if environment < 0.0:
            raise ValueError("loss_environment_variance must be non-negative")
        object.__setattr__(self, "loss_environment_variance", environment)
        object.__setattr__(
            self,
            "max_recovery_depth",
            _integer(self.max_recovery_depth, "max_recovery_depth", 1),
        )
        for name in (
            "depth_probability_scale",
            "recovery_probability",
            "recovery_gain",
            "base_leakage_probability",
            "higher_leakage_fraction",
            "leakage_logical_fault_probability",
        ):
            object.__setattr__(self, name, _probability(getattr(self, name), name))
        power = _finite(self.depth_probability_power, "depth_probability_power")
        if power <= 0.0:
            raise ValueError("depth_probability_power must be positive")
        object.__setattr__(self, "depth_probability_power", power)
        for name in ("loss_leakage_scale", "burst_leakage_bonus"):
            value = _finite(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        duration = _finite(self.higher_leakage_mean_duration, "higher_leakage_mean_duration")
        if duration < 2.0:
            raise ValueError("higher_leakage_mean_duration must be at least 2")
        object.__setattr__(self, "higher_leakage_mean_duration", duration)
        object.__setattr__(
            self,
            "confidence_level",
            _probability(self.confidence_level, "confidence_level", strict=True),
        )
        object.__setattr__(
            self,
            "bootstrap_replicates",
            _integer(self.bootstrap_replicates, "bootstrap_replicates", 200),
        )
        object.__setattr__(self, "seed", _integer(self.seed, "seed"))
        if self.seed >= 2**64:
            raise ValueError("seed must be smaller than 2**64")
        if self.model_scope != MODEL_SCOPE:
            raise ValueError(f"model_scope must be {MODEL_SCOPE!r}")

    @property
    def total_cycles(self) -> int:
        return self.n_trajectories * self.rounds_per_trajectory


@dataclass(frozen=True)
class RareEventSpec:
    kind: str = "burst_and_leakage"
    true_trajectory_probability: float = 1.0e-4
    allocation_fraction: float = 0.2
    mean_duration_cycles: float = 4.0
    displacement_scale: float = 3.0
    mean_shift: tuple[float, float] = (0.0, 0.0)
    extra_loss_gamma: float = 0.0
    forced_leakage_fault_probability: float = 0.02

    def __post_init__(self) -> None:
        if self.kind not in RARE_EVENT_KINDS:
            raise ValueError(f"kind must be one of {RARE_EVENT_KINDS}")
        object.__setattr__(
            self,
            "true_trajectory_probability",
            _probability(
                self.true_trajectory_probability,
                "true_trajectory_probability",
                strict=True,
            ),
        )
        object.__setattr__(
            self,
            "allocation_fraction",
            _probability(self.allocation_fraction, "allocation_fraction", strict=True),
        )
        duration = _finite(self.mean_duration_cycles, "mean_duration_cycles")
        if duration < 1.0:
            raise ValueError("mean_duration_cycles must be at least 1")
        object.__setattr__(self, "mean_duration_cycles", duration)
        scale = _finite(self.displacement_scale, "displacement_scale")
        if scale < 1.0:
            raise ValueError("displacement_scale must be at least 1")
        object.__setattr__(self, "displacement_scale", scale)
        object.__setattr__(self, "mean_shift", _pair(self.mean_shift, "mean_shift"))
        extra_loss = _finite(self.extra_loss_gamma, "extra_loss_gamma")
        if extra_loss < 0.0:
            raise ValueError("extra_loss_gamma must be non-negative")
        object.__setattr__(self, "extra_loss_gamma", extra_loss)
        object.__setattr__(
            self,
            "forced_leakage_fault_probability",
            _probability(
                self.forced_leakage_fault_probability,
                "forced_leakage_fault_probability",
            ),
        )


@dataclass(frozen=True)
class MonteCarloStratumResult:
    name: str
    target_weight: float
    n_trajectories: int
    rounds_per_trajectory: int
    simulated_cycles: int
    logical_event_count: int
    q_event_count: int
    p_event_count: int
    failed_trajectory_count: int
    conditional_cycle_error_rate: float
    conditional_q_error_rate: float
    conditional_p_error_rate: float
    conditional_trajectory_failure_rate: float
    trajectory_rate_std: float
    leakage_cycle_count: int
    burst_cycle_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            key: getattr(self, key)
            for key in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class FastMonteCarloResult:
    config: FastMonteCarloConfig
    rare_event: RareEventSpec | None
    strata: tuple[MonteCarloStratumResult, ...]
    logical_error_probability: float
    q_error_probability: float
    p_error_probability: float
    trajectory_failure_probability: float
    ci_low: float
    ci_high: float
    confidence_level: float
    ci_method: str
    bootstrap_standard_error: float
    zero_event_trajectory_upper_bound: float
    elapsed_seconds: float
    cycles_per_second: float
    protocol_id: str = SBS_PROTOCOL_ID
    model_scope: str = MODEL_SCOPE
    device_calibrated: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "t2.1.3-fast-monte-carlo-v1",
            "protocol_id": self.protocol_id,
            "model_scope": self.model_scope,
            "device_calibrated": self.device_calibrated,
            "seed": self.config.seed,
            "n_trajectories": self.config.n_trajectories,
            "rounds_per_trajectory": self.config.rounds_per_trajectory,
            "simulated_cycles": self.config.total_cycles,
            "logical_error_probability": self.logical_error_probability,
            "q_error_probability": self.q_error_probability,
            "p_error_probability": self.p_error_probability,
            "trajectory_failure_probability": self.trajectory_failure_probability,
            "confidence_interval": {
                "level": self.confidence_level,
                "low": self.ci_low,
                "high": self.ci_high,
                "method": self.ci_method,
                "bootstrap_standard_error": self.bootstrap_standard_error,
                "zero_event_trajectory_upper_bound": (
                    self.zero_event_trajectory_upper_bound
                ),
            },
            "rare_event": None
            if self.rare_event is None
            else {
                key: getattr(self.rare_event, key)
                for key in self.rare_event.__dataclass_fields__
            },
            "strata": [stratum.to_dict() for stratum in self.strata],
            "performance": {
                "elapsed_seconds": self.elapsed_seconds,
                "cycles_per_second": self.cycles_per_second,
                "at_least_1e5_cycles": self.config.total_cycles >= 100_000,
                "one_million_cycle_target_met": self.config.total_cycles >= 1_000_000,
            },
            "scope_limits": [
                "syndrome-level effective model",
                "trajectory-level rare-event mixture weights are assumptions",
                "not device calibrated or target-board timed",
            ],
        }


@dataclass(frozen=True)
class _StratumRaw:
    result: MonteCarloStratumResult
    trajectory_rates: np.ndarray
    trajectory_q_rates: np.ndarray
    trajectory_p_rates: np.ndarray
    trajectory_failures: np.ndarray


def _resolve_states(
    source: DriftProcess | Sequence[DriftState], rounds: int
) -> tuple[DriftState, ...]:
    if isinstance(source, DriftProcess):
        states = tuple(source.generate(rounds))
    else:
        if isinstance(source, (str, bytes)):
            raise TypeError("source must be a DriftProcess or sequence of DriftState")
        try:
            states = tuple(source)
        except TypeError as exc:
            raise TypeError("source must be a DriftProcess or sequence of DriftState") from exc
        if len(states) != rounds:
            raise ValueError("DriftState sequence length must equal rounds_per_trajectory")
    if len(states) != rounds or any(not isinstance(state, DriftState) for state in states):
        raise TypeError("source must produce exactly one DriftState per round")
    return states


def _simulate_stratum(
    states: tuple[DriftState, ...],
    *,
    name: str,
    target_weight: float,
    n_trajectories: int,
    config: FastMonteCarloConfig,
    rare_event: RareEventSpec | None,
    rng: np.random.Generator,
) -> _StratumRaw:
    n = n_trajectories
    rounds = config.rounds_per_trajectory
    residual = np.zeros((n, 2), dtype=np.float64)
    recovery_depth = np.zeros(n, dtype=np.int16)
    recovery_axis = np.full(n, -1, dtype=np.int8)
    leakage_remaining = np.zeros(n, dtype=np.int32)
    event_counts = np.zeros(n, dtype=np.int32)
    q_counts = np.zeros(n, dtype=np.int32)
    p_counts = np.zeros(n, dtype=np.int32)
    leakage_counts = np.zeros(n, dtype=np.int32)
    burst_counts = np.zeros(n, dtype=np.int32)

    if rare_event is None:
        rare_starts = np.zeros(n, dtype=np.int32)
        rare_ends = np.zeros(n, dtype=np.int32)
    else:
        rare_starts = rng.integers(0, rounds, size=n, dtype=np.int32)
        if rare_event.mean_duration_cycles == 1.0:
            durations = np.ones(n, dtype=np.int32)
        else:
            durations = rng.geometric(
                1.0 / rare_event.mean_duration_cycles,
                size=n,
            ).astype(np.int32)
        rare_ends = np.minimum(rounds, rare_starts + durations)

    higher_geometric_probability = 1.0 / (config.higher_leakage_mean_duration - 1.0)
    for round_index, state in enumerate(states):
        forced_active = (
            np.zeros(n, dtype=bool)
            if rare_event is None
            else (round_index >= rare_starts) & (round_index < rare_ends)
        )
        forced_burst = forced_active & (
            False if rare_event is None else rare_event.kind in {"burst", "burst_and_leakage"}
        )
        forced_leakage = forced_active & (
            False if rare_event is None else rare_event.kind in {"leakage", "burst_and_leakage"}
        )
        burst_mask = forced_burst | state.burst_active
        burst_counts += burst_mask.astype(np.int32)

        outlier = rng.random(n) < state.p_outlier
        component_scale = np.where(outlier, state.outlier_scale, 1.0)
        z_q = rng.standard_normal(n)
        z_p = rng.standard_normal(n)
        centered_q = state.sigma_q * z_q * component_scale
        centered_p = state.sigma_p * (
            state.rho * z_q + math.sqrt(1.0 - state.rho**2) * z_p
        ) * component_scale
        if rare_event is None:
            burst_scale = np.ones(n, dtype=np.float64)
            rare_mean_q = 0.0
            rare_mean_p = 0.0
            extra_gamma = np.zeros(n, dtype=np.float64)
        else:
            burst_scale = np.where(forced_burst, rare_event.displacement_scale, 1.0)
            rare_mean_q = rare_event.mean_shift[0]
            rare_mean_p = rare_event.mean_shift[1]
            extra_gamma = forced_burst.astype(np.float64) * rare_event.extra_loss_gamma
        channel_q = state.mu_q + centered_q * burst_scale + forced_burst * rare_mean_q
        channel_p = state.mu_p + centered_p * burst_scale + forced_burst * rare_mean_p

        eta = np.exp(-(state.loss_gamma + extra_gamma))
        loss_sigma = np.sqrt((1.0 - eta) * config.loss_environment_variance)
        pre = np.empty_like(residual)
        pre[:, 0] = np.sqrt(eta) * residual[:, 0] + channel_q + rng.standard_normal(n) * loss_sigma
        pre[:, 1] = np.sqrt(eta) * residual[:, 1] + channel_p + rng.standard_normal(n) * loss_sigma
        scaled = pre / config.lattice
        if np.any(np.abs(scaled) > 9.0e15):
            raise RuntimeError("lattice index exceeds exact float-integer conversion range")
        lattice_indices = np.floor(scaled + 0.5).astype(np.int64)
        q_event = np.mod(lattice_indices[:, 0], 2).astype(bool)
        p_event = np.mod(lattice_indices[:, 1], 2).astype(bool)
        folded = pre - lattice_indices * config.lattice

        severity = np.max(np.abs(folded), axis=1) / (0.5 * config.lattice)
        depth_probability = config.depth_probability_scale * np.clip(
            severity, 0.0, 1.0
        ) ** config.depth_probability_power
        injected = rng.binomial(config.max_recovery_depth, depth_probability).astype(
            np.int16
        )
        replace_depth = injected > recovery_depth
        recovery_depth[replace_depth] = injected[replace_depth]
        dominant_axis = np.where(np.abs(folded[:, 0]) >= np.abs(folded[:, 1]), 0, 1)
        recovery_axis[replace_depth] = dominant_axis[replace_depth].astype(np.int8)

        hazard = (
            config.base_leakage_probability
            + config.loss_leakage_scale * (1.0 - eta)
            + config.burst_leakage_bonus * burst_mask.astype(np.float64)
        )
        if np.any((hazard < 0.0) | (hazard > 1.0)):
            raise ValueError("derived leakage hazard must lie in [0, 1]")
        new_leakage = (leakage_remaining == 0) & (rng.random(n) < hazard)
        higher = new_leakage & (rng.random(n) < config.higher_leakage_fraction)
        leakage_remaining[new_leakage] = 1
        higher_count = int(np.count_nonzero(higher))
        if higher_count:
            leakage_remaining[higher] = 1 + rng.geometric(
                higher_geometric_probability,
                size=higher_count,
            ).astype(np.int32)
        background_leakage = leakage_remaining > 0
        active_leakage = background_leakage | forced_leakage
        leakage_counts += active_leakage.astype(np.int32)

        leak_fault_probability = np.full(
            n,
            config.leakage_logical_fault_probability,
            dtype=np.float64,
        )
        if rare_event is not None:
            leak_fault_probability = 1.0 - (
                (1.0 - leak_fault_probability)
                * np.where(
                    forced_leakage,
                    1.0 - rare_event.forced_leakage_fault_probability,
                    1.0,
                )
            )
        leakage_fault = active_leakage & (rng.random(n) < leak_fault_probability)
        logical_event = q_event | p_event | leakage_fault
        event_counts += logical_event.astype(np.int32)
        q_counts += q_event.astype(np.int32)
        p_counts += p_event.astype(np.int32)

        residual = folded
        if np.any(active_leakage):
            recovery_depth[active_leakage] = np.minimum(
                config.max_recovery_depth,
                recovery_depth[active_leakage] + 1,
            )
            needs_axis = active_leakage & (recovery_axis < 0)
            recovery_axis[needs_axis] = dominant_axis[needs_axis].astype(np.int8)
        recoverable = (~active_leakage) & (recovery_depth > 0) & (recovery_axis >= 0)
        recovered = recoverable & (rng.random(n) < config.recovery_probability)
        recovered_q = recovered & (recovery_axis == 0)
        recovered_p = recovered & (recovery_axis == 1)
        residual[recovered_q, 0] *= 1.0 - config.recovery_gain
        residual[recovered_p, 1] *= 1.0 - config.recovery_gain
        recovery_depth[recovered] -= 1
        recovery_axis[recovered & (recovery_depth == 0)] = -1
        leakage_remaining[background_leakage] -= 1

    cycles = n * rounds
    rates = event_counts.astype(np.float64) / rounds
    q_rates = q_counts.astype(np.float64) / rounds
    p_rates = p_counts.astype(np.float64) / rounds
    failures = event_counts > 0
    result = MonteCarloStratumResult(
        name=name,
        target_weight=float(target_weight),
        n_trajectories=n,
        rounds_per_trajectory=rounds,
        simulated_cycles=cycles,
        logical_event_count=int(np.sum(event_counts, dtype=np.int64)),
        q_event_count=int(np.sum(q_counts, dtype=np.int64)),
        p_event_count=int(np.sum(p_counts, dtype=np.int64)),
        failed_trajectory_count=int(np.count_nonzero(failures)),
        conditional_cycle_error_rate=float(np.mean(rates)),
        conditional_q_error_rate=float(np.mean(q_rates)),
        conditional_p_error_rate=float(np.mean(p_rates)),
        conditional_trajectory_failure_rate=float(np.mean(failures)),
        trajectory_rate_std=float(np.std(rates, ddof=1)),
        leakage_cycle_count=int(np.sum(leakage_counts, dtype=np.int64)),
        burst_cycle_count=int(np.sum(burst_counts, dtype=np.int64)),
    )
    return _StratumRaw(
        result=result,
        trajectory_rates=rates,
        trajectory_q_rates=q_rates,
        trajectory_p_rates=p_rates,
        trajectory_failures=failures.astype(np.float64),
    )


def _weighted_mean(raw: Sequence[_StratumRaw], attribute: str) -> float:
    return float(
        sum(
            item.result.target_weight * float(np.mean(getattr(item, attribute)))
            for item in raw
        )
    )


def _bootstrap_interval(
    raw: Sequence[_StratumRaw],
    config: FastMonteCarloConfig,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    estimates = np.empty(config.bootstrap_replicates, dtype=np.float64)
    for replicate in range(config.bootstrap_replicates):
        estimate = 0.0
        for item in raw:
            rates = item.trajectory_rates
            indices = rng.integers(0, rates.size, size=rates.size)
            estimate += item.result.target_weight * float(np.mean(rates[indices]))
        estimates[replicate] = estimate
    alpha = 1.0 - config.confidence_level
    low, high = np.quantile(estimates, [0.5 * alpha, 1.0 - 0.5 * alpha])

    zero_bound_components: list[float] = []
    has_zero_event_stratum = False
    for item in raw:
        if item.result.failed_trajectory_count == 0:
            component = 1.0 - alpha ** (1.0 / item.result.n_trajectories)
            has_zero_event_stratum = True
        else:
            component = item.result.conditional_cycle_error_rate
        zero_bound_components.append(item.result.target_weight * component)
    zero_bound = sum(zero_bound_components) if has_zero_event_stratum else 0.0
    high = max(float(high), zero_bound)
    return (
        max(0.0, float(low)),
        min(1.0, float(high)),
        float(np.std(estimates, ddof=1)),
        min(1.0, float(zero_bound)),
    )


def run_fast_monte_carlo(
    source: DriftProcess | Sequence[DriftState],
    *,
    config: FastMonteCarloConfig | None = None,
    rare_event: RareEventSpec | None = None,
) -> FastMonteCarloResult:
    """运行 vectorized multi-trajectory estimator。

    rare mode 的 estimand 是 trajectory-level two-stratum mixture；不是把过采样后的
    raw event fraction 冒充真实 ``P_L``。
    """

    actual = FastMonteCarloConfig() if config is None else config
    if not isinstance(actual, FastMonteCarloConfig):
        raise TypeError("config must be a FastMonteCarloConfig or None")
    if rare_event is not None and not isinstance(rare_event, RareEventSpec):
        raise TypeError("rare_event must be a RareEventSpec or None")
    states = _resolve_states(source, actual.rounds_per_trajectory)
    start = time.perf_counter()
    child_sequences = np.random.SeedSequence(actual.seed).spawn(3)
    raw: list[_StratumRaw] = []
    if rare_event is None:
        raw.append(
            _simulate_stratum(
                states,
                name="unstratified",
                target_weight=1.0,
                n_trajectories=actual.n_trajectories,
                config=actual,
                rare_event=None,
                rng=np.random.default_rng(child_sequences[0]),
            )
        )
    else:
        rare_count = int(round(actual.n_trajectories * rare_event.allocation_fraction))
        rare_count = min(actual.n_trajectories - 2, max(2, rare_count))
        normal_count = actual.n_trajectories - rare_count
        raw.append(
            _simulate_stratum(
                states,
                name="normal_no_extra_rare_episode",
                target_weight=1.0 - rare_event.true_trajectory_probability,
                n_trajectories=normal_count,
                config=actual,
                rare_event=None,
                rng=np.random.default_rng(child_sequences[0]),
            )
        )
        raw.append(
            _simulate_stratum(
                states,
                name=f"conditional_{rare_event.kind}_episode",
                target_weight=rare_event.true_trajectory_probability,
                n_trajectories=rare_count,
                config=actual,
                rare_event=rare_event,
                rng=np.random.default_rng(child_sequences[1]),
            )
        )

    rate = _weighted_mean(raw, "trajectory_rates")
    q_rate = _weighted_mean(raw, "trajectory_q_rates")
    p_rate = _weighted_mean(raw, "trajectory_p_rates")
    trajectory_failure = _weighted_mean(raw, "trajectory_failures")
    ci_low, ci_high, bootstrap_se, zero_bound = _bootstrap_interval(
        raw,
        actual,
        np.random.default_rng(child_sequences[2]),
    )
    ci_low = min(ci_low, rate)
    ci_high = max(ci_high, rate)
    elapsed = max(time.perf_counter() - start, np.finfo(float).tiny)
    return FastMonteCarloResult(
        config=actual,
        rare_event=rare_event,
        strata=tuple(item.result for item in raw),
        logical_error_probability=rate,
        q_error_probability=q_rate,
        p_error_probability=p_rate,
        trajectory_failure_probability=trajectory_failure,
        ci_low=ci_low,
        ci_high=ci_high,
        confidence_level=actual.confidence_level,
        ci_method=(
            "trajectory_cluster_percentile_bootstrap_with_zero-event_trajectory_upper_bound"
        ),
        bootstrap_standard_error=bootstrap_se,
        zero_event_trajectory_upper_bound=zero_bound,
        elapsed_seconds=elapsed,
        cycles_per_second=actual.total_cycles / elapsed,
    )


def write_fast_monte_carlo_report(
    result: FastMonteCarloResult,
    output_path: str | Path,
) -> Path:
    if not isinstance(result, FastMonteCarloResult):
        raise TypeError("result must be a FastMonteCarloResult")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-trajectories", type=int, default=10_000)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026071413)
    parser.add_argument("--rare", action="store_true")
    args = parser.parse_args()
    config = FastMonteCarloConfig(
        n_trajectories=args.n_trajectories,
        rounds_per_trajectory=args.rounds,
        seed=args.seed,
    )
    base = DriftState(
        sigma_q=0.12 * LATTICE_CONST,
        sigma_p=0.14 * LATTICE_CONST,
        rho=0.25,
        loss_gamma=0.01,
        p_outlier=0.0005,
        outlier_scale=4.0,
        source="t2.1.3-production",
        regime="base",
        seed=args.seed,
    )
    result = run_fast_monte_carlo(
        [base] * args.rounds,
        config=config,
        rare_event=RareEventSpec() if args.rare else None,
    )
    write_fast_monte_carlo_report(result, args.output)
    print(json.dumps(result.to_dict()["performance"], ensure_ascii=False))


if __name__ == "__main__":
    _main()
