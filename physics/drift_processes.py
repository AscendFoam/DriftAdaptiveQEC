"""可复现的非平稳 GKP 噪声状态与 synthetic drift processes。

本模块只生成解码器/实验 harness 可消费的隐藏真值 ``DriftState``，不把
syndrome-level effective model 冒充完整 cavity--ancilla 动力学。随机过程均从固定
seed 重新生成前缀，因此结果与调用顺序无关；telegraph 和 burst 显式保留跨时间状态。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Optional, Protocol, runtime_checkable

import numpy as np


_MAX_SAMPLES = 10_000_000


def _finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive(name: str, value: float) -> float:
    result = _finite(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _seed(value: int) -> int:
    result = _nonnegative_int("seed", value)
    if result >= 2**64:
        raise ValueError("seed must be smaller than 2**64")
    return result


def _steps(value: int) -> int:
    result = _nonnegative_int("steps", value)
    if result > _MAX_SAMPLES:
        raise ValueError(f"steps must not exceed {_MAX_SAMPLES}")
    return result


@dataclass(frozen=True)
class DriftState:
    """单个时间步的隐藏噪声真值。

    ``sigma_q/sigma_p/rho`` 定义 core Gaussian covariance；``p_outlier`` 与
    ``outlier_scale`` 定义同均值、按 covariance scale 放大的第二个 Gaussian
    mixture component。``loss_gamma`` 使用 ``gamma=kappa*t`` 口径，并提供
    ``eta=exp(-gamma)``。它不在这里被悄悄折成位移噪声。
    """

    step: int = 0
    time: float = 0.0
    mu_q: float = 0.0
    mu_p: float = 0.0
    sigma_q: float = 0.3
    sigma_p: float = 0.3
    rho: float = 0.0
    loss_gamma: float = 0.0
    p_outlier: float = 0.0
    outlier_scale: float = 1.0
    burst_active: bool = False
    source: str = "manual"
    regime: str = "base"
    seed: int = 0
    event_id: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "step", _nonnegative_int("step", self.step))
        object.__setattr__(self, "time", _finite("time", self.time))
        if self.time < 0.0:
            raise ValueError("time must be nonnegative")
        for name in ("mu_q", "mu_p"):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))
        for name in ("sigma_q", "sigma_p"):
            object.__setattr__(self, name, _positive(name, getattr(self, name)))
        rho = _finite("rho", self.rho)
        if not -1.0 < rho < 1.0:
            raise ValueError("rho must lie strictly between -1 and 1")
        object.__setattr__(self, "rho", rho)
        gamma = _finite("loss_gamma", self.loss_gamma)
        if gamma < 0.0:
            raise ValueError("loss_gamma must be nonnegative")
        object.__setattr__(self, "loss_gamma", gamma)
        probability = _finite("p_outlier", self.p_outlier)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("p_outlier must lie in [0, 1]")
        object.__setattr__(self, "p_outlier", probability)
        scale = _finite("outlier_scale", self.outlier_scale)
        if scale < 1.0:
            raise ValueError("outlier_scale must be at least 1")
        object.__setattr__(self, "outlier_scale", scale)
        if not isinstance(self.burst_active, (bool, np.bool_)):
            raise TypeError("burst_active must be boolean")
        object.__setattr__(self, "burst_active", bool(self.burst_active))
        for name in ("source", "regime"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
            object.__setattr__(self, name, value.strip())
        object.__setattr__(self, "seed", _seed(self.seed))
        object.__setattr__(
            self,
            "event_id",
            _nonnegative_int("event_id", self.event_id),
        )

    @property
    def mean(self) -> np.ndarray:
        return np.array([self.mu_q, self.mu_p], dtype=float)

    @property
    def covariance(self) -> np.ndarray:
        covariance = self.rho * self.sigma_q * self.sigma_p
        return np.array(
            [
                [self.sigma_q**2, covariance],
                [covariance, self.sigma_p**2],
            ],
            dtype=float,
        )

    @property
    def outlier_covariance(self) -> np.ndarray:
        return self.covariance * self.outlier_scale**2

    @property
    def mixture_covariance(self) -> np.ndarray:
        factor = (1.0 - self.p_outlier) + self.p_outlier * self.outlier_scale**2
        return self.covariance * factor

    @property
    def eta(self) -> float:
        return math.exp(-self.loss_gamma)

    @property
    def principal_angle(self) -> float:
        """Core covariance 主轴角，范围 ``[-pi/2, pi/2]``。"""

        covariance = self.covariance
        numerator = 2.0 * float(covariance[0, 1])
        denominator = float(covariance[0, 0] - covariance[1, 1])
        if abs(numerator) < 1.0e-15 and abs(denominator) < 1.0e-15:
            return 0.0
        angle = 0.5 * math.atan2(numerator, denominator)
        if angle > math.pi / 2.0:
            angle -= math.pi
        if angle <= -math.pi / 2.0:
            angle += math.pi
        return angle

    def legacy_effective_sigma(
        self,
        *,
        include_mean: bool = True,
        include_loss_proxy: bool = True,
        include_outliers: bool = True,
    ) -> float:
        """把完整状态压成旧 ``run_with_drift`` 的 isotropic RMS sigma。

        loss 使用旧 ``CombinedNoiseModel`` 相同的 ``gamma/2`` additive proxy；该值仅
        用于兼容 smoke，不能重建 mixture likelihood 或物理 loss channel。
        """

        covariance = self.mixture_covariance if include_outliers else self.covariance
        second_moment = float(np.trace(covariance)) / 2.0
        if include_mean:
            second_moment += (self.mu_q**2 + self.mu_p**2) / 2.0
        if include_loss_proxy:
            second_moment += self.loss_gamma / 2.0
        sigma = math.sqrt(second_moment)
        if not math.isfinite(sigma):
            raise ValueError("legacy effective sigma overflowed")
        return sigma


@runtime_checkable
class DriftProcess(Protocol):
    """所有 drift process 的最小随机访问/前缀接口。"""

    def state_at(self, step: int) -> DriftState:
        ...

    def generate(self, steps: int) -> tuple[DriftState, ...]:
        ...


def _process_context(
    template: DriftState,
    *,
    step: int,
    dt: float,
    source: str,
    regime: str,
    seed: int,
    event_id: int = 0,
    burst_active: bool = False,
    **updates: float,
) -> DriftState:
    return replace(
        template,
        step=step,
        time=step * dt,
        source=source,
        regime=regime,
        seed=seed,
        event_id=event_id,
        burst_active=burst_active,
        **updates,
    )


@dataclass(frozen=True)
class ConstantDriftProcess:
    base: DriftState = field(default_factory=DriftState)
    dt: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.base, DriftState):
            raise TypeError("base must be a DriftState")
        object.__setattr__(self, "dt", _positive("dt", self.dt))
        object.__setattr__(self, "seed", _seed(self.seed))

    def state_at(self, step: int) -> DriftState:
        step = _nonnegative_int("step", step)
        return _process_context(
            self.base,
            step=step,
            dt=self.dt,
            source="constant",
            regime="base",
            seed=self.seed,
        )

    def generate(self, steps: int) -> tuple[DriftState, ...]:
        return tuple(self.state_at(step) for step in range(_steps(steps)))


@dataclass(frozen=True)
class MeanDriftProcess:
    """线性趋势叠加可选正弦项的 mean drift。"""

    base: DriftState = field(default_factory=DriftState)
    rate_q: float = 0.0
    rate_p: float = 0.0
    amplitude_q: float = 0.0
    amplitude_p: float = 0.0
    period: Optional[float] = None
    phase: float = 0.0
    dt: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.base, DriftState):
            raise TypeError("base must be a DriftState")
        for name in ("rate_q", "rate_p", "amplitude_q", "amplitude_p", "phase"):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))
        if self.period is None:
            if self.amplitude_q != 0.0 or self.amplitude_p != 0.0:
                raise ValueError("period is required for nonzero sinusoidal amplitude")
        else:
            object.__setattr__(self, "period", _positive("period", self.period))
        object.__setattr__(self, "dt", _positive("dt", self.dt))
        object.__setattr__(self, "seed", _seed(self.seed))

    def state_at(self, step: int) -> DriftState:
        step = _nonnegative_int("step", step)
        time = step * self.dt
        oscillation = 0.0
        if self.period is not None:
            oscillation = math.sin(2.0 * math.pi * time / self.period + self.phase)
        return _process_context(
            self.base,
            step=step,
            dt=self.dt,
            source="mean",
            regime="smooth",
            seed=self.seed,
            mu_q=self.base.mu_q + self.rate_q * time + self.amplitude_q * oscillation,
            mu_p=self.base.mu_p + self.rate_p * time + self.amplitude_p * oscillation,
        )

    def generate(self, steps: int) -> tuple[DriftState, ...]:
        return tuple(self.state_at(step) for step in range(_steps(steps)))


@dataclass(frozen=True)
class VarianceDriftProcess:
    """在 log-sigma/Fisher-rho 坐标中线性演化的 variance drift。"""

    base: DriftState = field(default_factory=DriftState)
    log_sigma_rate_q: float = 0.0
    log_sigma_rate_p: float = 0.0
    fisher_rho_rate: float = 0.0
    dt: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.base, DriftState):
            raise TypeError("base must be a DriftState")
        for name in ("log_sigma_rate_q", "log_sigma_rate_p", "fisher_rho_rate"):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))
        object.__setattr__(self, "dt", _positive("dt", self.dt))
        object.__setattr__(self, "seed", _seed(self.seed))

    def state_at(self, step: int) -> DriftState:
        step = _nonnegative_int("step", step)
        time = step * self.dt
        try:
            sigma_q = self.base.sigma_q * math.exp(self.log_sigma_rate_q * time)
            sigma_p = self.base.sigma_p * math.exp(self.log_sigma_rate_p * time)
        except OverflowError as exc:
            raise ValueError("variance drift overflowed") from exc
        fisher = math.atanh(self.base.rho) + self.fisher_rho_rate * time
        rho = math.tanh(fisher)
        return _process_context(
            self.base,
            step=step,
            dt=self.dt,
            source="variance",
            regime="smooth",
            seed=self.seed,
            sigma_q=sigma_q,
            sigma_p=sigma_p,
            rho=rho,
        )

    def generate(self, steps: int) -> tuple[DriftState, ...]:
        return tuple(self.state_at(step) for step in range(_steps(steps)))


@dataclass(frozen=True)
class LossDriftProcess:
    """从 base ``loss_gamma`` 指数松弛到目标值的 loss drift。"""

    base: DriftState = field(default_factory=DriftState)
    target_gamma: float = 0.1
    time_constant: float = 100.0
    dt: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.base, DriftState):
            raise TypeError("base must be a DriftState")
        target = _finite("target_gamma", self.target_gamma)
        if target < 0.0:
            raise ValueError("target_gamma must be nonnegative")
        object.__setattr__(self, "target_gamma", target)
        object.__setattr__(self, "time_constant", _positive("time_constant", self.time_constant))
        object.__setattr__(self, "dt", _positive("dt", self.dt))
        object.__setattr__(self, "seed", _seed(self.seed))

    def state_at(self, step: int) -> DriftState:
        step = _nonnegative_int("step", step)
        time = step * self.dt
        progress = -math.expm1(-time / self.time_constant)
        gamma = self.base.loss_gamma + (self.target_gamma - self.base.loss_gamma) * progress
        return _process_context(
            self.base,
            step=step,
            dt=self.dt,
            source="loss",
            regime="smooth",
            seed=self.seed,
            loss_gamma=gamma,
        )

    def generate(self, steps: int) -> tuple[DriftState, ...]:
        return tuple(self.state_at(step) for step in range(_steps(steps)))


@dataclass(frozen=True)
class OutlierRateDriftProcess:
    """从 base probability 指数松弛到目标值的 outlier-rate drift。"""

    base: DriftState = field(default_factory=DriftState)
    target_probability: float = 0.1
    time_constant: float = 100.0
    dt: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.base, DriftState):
            raise TypeError("base must be a DriftState")
        target = _finite("target_probability", self.target_probability)
        if not 0.0 <= target <= 1.0:
            raise ValueError("target_probability must lie in [0, 1]")
        object.__setattr__(self, "target_probability", target)
        object.__setattr__(self, "time_constant", _positive("time_constant", self.time_constant))
        object.__setattr__(self, "dt", _positive("dt", self.dt))
        object.__setattr__(self, "seed", _seed(self.seed))

    def state_at(self, step: int) -> DriftState:
        step = _nonnegative_int("step", step)
        time = step * self.dt
        progress = -math.expm1(-time / self.time_constant)
        probability = self.base.p_outlier + (
            self.target_probability - self.base.p_outlier
        ) * progress
        return _process_context(
            self.base,
            step=step,
            dt=self.dt,
            source="outlier_rate",
            regime="smooth",
            seed=self.seed,
            p_outlier=probability,
        )

    def generate(self, steps: int) -> tuple[DriftState, ...]:
        return tuple(self.state_at(step) for step in range(_steps(steps)))


@dataclass(frozen=True)
class StepDriftProcess:
    before: DriftState
    after: DriftState
    change_step: int
    dt: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.before, DriftState) or not isinstance(self.after, DriftState):
            raise TypeError("before and after must be DriftState instances")
        object.__setattr__(self, "change_step", _nonnegative_int("change_step", self.change_step))
        object.__setattr__(self, "dt", _positive("dt", self.dt))
        object.__setattr__(self, "seed", _seed(self.seed))

    def state_at(self, step: int) -> DriftState:
        step = _nonnegative_int("step", step)
        is_after = step >= self.change_step
        return _process_context(
            self.after if is_after else self.before,
            step=step,
            dt=self.dt,
            source="step",
            regime="after" if is_after else "before",
            seed=self.seed,
            event_id=1 if is_after else 0,
        )

    def generate(self, steps: int) -> tuple[DriftState, ...]:
        return tuple(self.state_at(step) for step in range(_steps(steps)))


@dataclass(frozen=True)
class TelegraphDriftProcess:
    """具有不对称 transition probabilities 的二态 Markov telegraph。"""

    state_a: DriftState
    state_b: DriftState
    p_a_to_b: float = 0.05
    p_b_to_a: float = 0.05
    initial_regime: str = "a"
    dt: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.state_a, DriftState) or not isinstance(self.state_b, DriftState):
            raise TypeError("state_a and state_b must be DriftState instances")
        for name in ("p_a_to_b", "p_b_to_a"):
            probability = _finite(name, getattr(self, name))
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
            object.__setattr__(self, name, probability)
        if self.initial_regime not in {"a", "b"}:
            raise ValueError("initial_regime must be 'a' or 'b'")
        object.__setattr__(self, "dt", _positive("dt", self.dt))
        object.__setattr__(self, "seed", _seed(self.seed))

    def generate(self, steps: int) -> tuple[DriftState, ...]:
        count = _steps(steps)
        rng = np.random.default_rng(self.seed)
        regime = self.initial_regime
        event_id = 0
        states: list[DriftState] = []
        for step in range(count):
            if step > 0:
                probability = self.p_a_to_b if regime == "a" else self.p_b_to_a
                if float(rng.random()) < probability:
                    regime = "b" if regime == "a" else "a"
                    event_id += 1
            template = self.state_a if regime == "a" else self.state_b
            states.append(
                _process_context(
                    template,
                    step=step,
                    dt=self.dt,
                    source="telegraph",
                    regime=regime,
                    seed=self.seed,
                    event_id=event_id,
                )
            )
        return tuple(states)

    def state_at(self, step: int) -> DriftState:
        index = _nonnegative_int("step", step)
        return self.generate(index + 1)[-1]


@dataclass(frozen=True)
class BurstDriftProcess:
    """带 onset hazard、整数持续时间和 cooldown 的 burst process。"""

    baseline: DriftState
    burst: DriftState
    onset_probability: float = 0.02
    min_duration: int = 2
    max_duration: int = 8
    cooldown_steps: int = 0
    dt: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.baseline, DriftState) or not isinstance(self.burst, DriftState):
            raise TypeError("baseline and burst must be DriftState instances")
        probability = _finite("onset_probability", self.onset_probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("onset_probability must lie in [0, 1]")
        object.__setattr__(self, "onset_probability", probability)
        minimum = _nonnegative_int("min_duration", self.min_duration)
        maximum = _nonnegative_int("max_duration", self.max_duration)
        if minimum < 1 or maximum < minimum:
            raise ValueError("durations must satisfy 1 <= min_duration <= max_duration")
        object.__setattr__(self, "min_duration", minimum)
        object.__setattr__(self, "max_duration", maximum)
        object.__setattr__(
            self,
            "cooldown_steps",
            _nonnegative_int("cooldown_steps", self.cooldown_steps),
        )
        object.__setattr__(self, "dt", _positive("dt", self.dt))
        object.__setattr__(self, "seed", _seed(self.seed))

    def generate(self, steps: int) -> tuple[DriftState, ...]:
        count = _steps(steps)
        rng = np.random.default_rng(self.seed)
        remaining = 0
        cooldown = 0
        event_id = 0
        states: list[DriftState] = []
        for step in range(count):
            if remaining == 0:
                if cooldown > 0:
                    cooldown -= 1
                elif float(rng.random()) < self.onset_probability:
                    event_id += 1
                    remaining = int(rng.integers(self.min_duration, self.max_duration + 1))

            active = remaining > 0
            template = self.burst if active else self.baseline
            states.append(
                _process_context(
                    template,
                    step=step,
                    dt=self.dt,
                    source="burst",
                    regime="burst" if active else "base",
                    seed=self.seed,
                    event_id=event_id,
                    burst_active=active,
                )
            )
            if active:
                remaining -= 1
                if remaining == 0:
                    cooldown = self.cooldown_steps
        return tuple(states)

    def state_at(self, step: int) -> DriftState:
        index = _nonnegative_int("step", step)
        return self.generate(index + 1)[-1]


def sample_displacements(
    state: DriftState,
    n_samples: int,
    *,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """从 ``DriftState`` 的二分量 Gaussian mixture 采样位移和 outlier mask。"""

    if not isinstance(state, DriftState):
        raise TypeError("state must be a DriftState")
    count = _nonnegative_int("n_samples", n_samples)
    if count < 1 or count > _MAX_SAMPLES:
        raise ValueError(f"n_samples must lie in [1, {_MAX_SAMPLES}]")
    if seed is not None and rng is not None:
        raise ValueError("provide either seed or rng, not both")
    if rng is not None and not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")
    generator = rng if rng is not None else np.random.default_rng(0 if seed is None else _seed(seed))
    outlier_mask = generator.random(count) < state.p_outlier
    centered = generator.multivariate_normal(
        np.zeros(2, dtype=float),
        state.covariance,
        size=count,
    )
    centered[outlier_mask] *= state.outlier_scale
    return centered + state.mean[None, :], outlier_mask


@dataclass
class LegacyRunWithDriftAdapter:
    """把 ``DriftProcess`` 适配为旧 ``t -> (sigma, delta, theta)`` 回调。

    该适配器有意保留 ``last_state``/``state_at`` 供审计，但旧 simulator 只能消费
    isotropic RMS 压缩结果，不能用于 oracle 或 mixture-aware benchmark。
    """

    process: DriftProcess
    delta: float = 0.3
    include_mean: bool = True
    include_loss_proxy: bool = True
    include_outliers: bool = True
    last_state: Optional[DriftState] = field(init=False, default=None)

    def __post_init__(self) -> None:
        if not isinstance(self.process, DriftProcess):
            raise TypeError("process must implement state_at() and generate()")
        self.delta = _positive("delta", self.delta)
        for name in ("include_mean", "include_loss_proxy", "include_outliers"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")

    def state_at(self, step: int) -> DriftState:
        state = self.process.state_at(_nonnegative_int("step", step))
        if not isinstance(state, DriftState):
            raise TypeError("process.state_at() must return DriftState")
        return state

    def __call__(self, step: int) -> tuple[float, float, float]:
        state = self.state_at(step)
        self.last_state = state
        return (
            state.legacy_effective_sigma(
                include_mean=self.include_mean,
                include_loss_proxy=self.include_loss_proxy,
                include_outliers=self.include_outliers,
            ),
            self.delta,
            state.principal_angle,
        )


def as_run_with_drift_callback(
    process: DriftProcess,
    *,
    delta: float = 0.3,
    include_mean: bool = True,
    include_loss_proxy: bool = True,
    include_outliers: bool = True,
) -> LegacyRunWithDriftAdapter:
    return LegacyRunWithDriftAdapter(
        process=process,
        delta=delta,
        include_mean=include_mean,
        include_loss_proxy=include_loss_proxy,
        include_outliers=include_outliers,
    )


__all__ = [
    "DriftState",
    "DriftProcess",
    "ConstantDriftProcess",
    "MeanDriftProcess",
    "VarianceDriftProcess",
    "LossDriftProcess",
    "OutlierRateDriftProcess",
    "StepDriftProcess",
    "TelegraphDriftProcess",
    "BurstDriftProcess",
    "sample_displacements",
    "LegacyRunWithDriftAdapter",
    "as_run_with_drift_callback",
]
