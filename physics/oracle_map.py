"""知道真实 ``DriftState`` 的不可部署 periodic-mixture oracle MAP。

oracle 使用每个时间步的真实 mean、2D covariance、outlier probability/scale。它是
synthetic model 内的 Bayes reference upper bound，不是可部署估计器，也不读取 legacy
``run_with_drift`` 的 scalar RMS adapter。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import LATTICE_CONST
from .drift_processes import DriftState
from .ideal_gkp_decoder import map_decode_2d


IntOrArray = Union[int, NDArray[np.int64]]
FloatOrArray = Union[float, NDArray[np.float64]]
ActionOrArray = Union[str, NDArray[np.str_]]
LossModel = Literal["separate", "additive_displacement_proxy"]
_ACTIONS = np.array(["I", "Z", "X", "Y"], dtype="<U1")


@dataclass(frozen=True)
class OracleMAPResult:
    """一个已知 state 下的四陪集 oracle MAP 输出。

    ``analog_correction`` 是应从位移中减去的 centered syndrome；``logical_action``
    是根据 oracle parity 施加/追踪的 Pauli（q parity -> X，p parity -> Z）。
    """

    syndrome: NDArray[np.float64]
    parity: NDArray[np.int64]
    logical_class: IntOrArray
    logical_action: ActionOrArray
    analog_correction: NDArray[np.float64]
    posterior: NDArray[np.float64]
    confidence: FloatOrArray
    log_likelihoods: NDArray[np.float64]
    state_step: int
    state_source: str
    state_regime: str
    burst_active: bool
    mixture_weights: tuple[float, float]
    loss_model: LossModel
    evidence_scope: str = "nondeployable_full_state_oracle"


@dataclass(frozen=True)
class OracleTrajectoryResult:
    """逐时间步 ``DriftState`` 与 syndrome 一一对齐后的 oracle 输出。"""

    syndrome: NDArray[np.float64]
    parity: NDArray[np.int64]
    logical_class: NDArray[np.int64]
    logical_action: NDArray[np.str_]
    analog_correction: NDArray[np.float64]
    posterior: NDArray[np.float64]
    confidence: NDArray[np.float64]
    log_likelihoods: NDArray[np.float64]
    state_steps: NDArray[np.int64]
    state_sources: tuple[str, ...]
    state_regimes: tuple[str, ...]
    burst_active: NDArray[np.bool_]
    loss_model: LossModel
    evidence_scope: str = "nondeployable_full_state_oracle"


def _validate_loss_model(loss_model: str) -> LossModel:
    if loss_model not in {"separate", "additive_displacement_proxy"}:
        raise ValueError(
            "loss_model must be 'separate' or 'additive_displacement_proxy'"
        )
    return loss_model  # type: ignore[return-value]


def _component_covariances(
    state: DriftState,
    loss_model: LossModel,
) -> tuple[np.ndarray, np.ndarray]:
    core = state.covariance
    outlier = state.outlier_covariance
    if loss_model == "additive_displacement_proxy":
        # 与旧 CombinedNoiseModel 的 sigma_loss^2=gamma/2 口径对齐；独立 loss
        # contribution 加到两个 mixture components，而不随 outlier scale 放大。
        loss_covariance = np.eye(2, dtype=float) * (state.loss_gamma / 2.0)
        core = core + loss_covariance
        outlier = outlier + loss_covariance
    return core, outlier


def _validate_prior(prior: ArrayLike | None) -> np.ndarray:
    if prior is None:
        return np.full((2, 2), 0.25, dtype=float)
    probabilities = np.asarray(prior, dtype=float)
    if probabilities.shape != (2, 2):
        raise ValueError("prior must have shape (2, 2)")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities <= 0.0):
        raise ValueError("all prior entries must be finite and strictly positive")
    with np.errstate(over="ignore", invalid="ignore"):
        total = float(np.sum(probabilities))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("prior sum must be finite and positive")
    return probabilities / total


def _logsumexp_last_two(values: np.ndarray) -> np.ndarray:
    maximum = np.max(values, axis=(-2, -1))
    shifted = values - maximum[..., np.newaxis, np.newaxis]
    return maximum + np.log(np.sum(np.exp(shifted), axis=(-2, -1)))


def oracle_log_likelihoods_2d(
    syndrome: ArrayLike,
    state: DriftState,
    *,
    lattice: float = LATTICE_CONST,
    tail_sigma: float = 10.0,
    loss_model: LossModel = "separate",
) -> NDArray[np.float64]:
    r"""返回已知 state 下四个 periodic Gaussian-mixture coset log likelihoods。

    .. math::

       L_b(s)=(1-p)L_b(s;\mu,\Sigma)
              +pL_b(s;\mu,a^2\Sigma).

    两分量在 log domain 用 ``logaddexp`` 合并；``p=0/1`` 和 ``a=1`` 走精确退化
    分支，避免 ``log(0)`` 或无意义重复计算。
    """

    if not isinstance(state, DriftState):
        raise TypeError("state must be a DriftState")
    validated_loss_model = _validate_loss_model(loss_model)
    core_covariance, outlier_covariance = _component_covariances(
        state,
        validated_loss_model,
    )
    core_logs = map_decode_2d(
        syndrome,
        core_covariance,
        mean=state.mean,
        lattice=lattice,
        tail_sigma=tail_sigma,
    ).log_likelihoods
    probability = state.p_outlier
    if probability == 0.0:
        return np.asarray(core_logs, dtype=float)
    if state.outlier_scale == 1.0:
        return np.asarray(core_logs, dtype=float)

    outlier_logs = map_decode_2d(
        syndrome,
        outlier_covariance,
        mean=state.mean,
        lattice=lattice,
        tail_sigma=tail_sigma,
    ).log_likelihoods
    if probability == 1.0:
        return np.asarray(outlier_logs, dtype=float)
    return np.logaddexp(
        math.log1p(-probability) + core_logs,
        math.log(probability) + outlier_logs,
    )


def oracle_map_2d(
    syndrome: ArrayLike,
    state: DriftState,
    *,
    lattice: float = LATTICE_CONST,
    prior: ArrayLike | None = None,
    tail_sigma: float = 10.0,
    loss_model: LossModel = "separate",
) -> OracleMAPResult:
    """在一个真实 ``DriftState`` 下执行四陪集 mixture-aware oracle MAP。"""

    if not isinstance(state, DriftState):
        raise TypeError("state must be a DriftState")
    validated_loss_model = _validate_loss_model(loss_model)
    probabilities = _validate_prior(prior)
    values = np.asarray(syndrome, dtype=float)
    log_likelihoods = oracle_log_likelihoods_2d(
        values,
        state,
        lattice=lattice,
        tail_sigma=tail_sigma,
        loss_model=validated_loss_model,
    )
    log_scores = log_likelihoods + np.log(probabilities)
    log_evidence = _logsumexp_last_two(log_scores)
    posterior = np.exp(
        log_scores - log_evidence[..., np.newaxis, np.newaxis]
    )
    flat = posterior.reshape(posterior.shape[:-2] + (4,))
    logical_class_array = np.argmax(flat, axis=-1).astype(np.int64)
    parity = np.stack(
        (logical_class_array // 2, logical_class_array % 2),
        axis=-1,
    ).astype(np.int64)
    actions = _ACTIONS[logical_class_array]
    ordered = np.sort(flat, axis=-1)
    confidence_array = ordered[..., -1] - ordered[..., -2]
    scalar_input = values.ndim == 1
    logical_class: IntOrArray
    logical_action: ActionOrArray
    confidence: FloatOrArray
    if scalar_input:
        logical_class = int(np.asarray(logical_class_array).item())
        logical_action = str(np.asarray(actions).item())
        confidence = float(np.asarray(confidence_array).item())
    else:
        logical_class = logical_class_array
        logical_action = np.asarray(actions)
        confidence = confidence_array

    return OracleMAPResult(
        syndrome=values.copy(),
        parity=parity,
        logical_class=logical_class,
        logical_action=logical_action,
        analog_correction=values.copy(),
        posterior=posterior,
        confidence=confidence,
        log_likelihoods=log_likelihoods,
        state_step=state.step,
        state_source=state.source,
        state_regime=state.regime,
        burst_active=state.burst_active,
        mixture_weights=(1.0 - state.p_outlier, state.p_outlier),
        loss_model=validated_loss_model,
    )


def oracle_map_trajectory(
    syndrome: ArrayLike,
    states: Sequence[DriftState],
    *,
    lattice: float = LATTICE_CONST,
    prior: ArrayLike | None = None,
    tail_sigma: float = 10.0,
    loss_model: LossModel = "separate",
) -> OracleTrajectoryResult:
    """对齐 ``syndrome[t]`` 与真实 ``states[t]``，逐步执行 oracle MAP。"""

    values = np.asarray(syndrome, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("trajectory syndrome must have shape (timesteps, 2)")
    if values.shape[0] == 0:
        raise ValueError("trajectory must contain at least one timestep")
    try:
        state_tuple = tuple(states)
    except TypeError as exc:
        raise TypeError("states must be a sequence of DriftState") from exc
    if len(state_tuple) != values.shape[0]:
        raise ValueError("states length must match syndrome timesteps")
    if any(not isinstance(state, DriftState) for state in state_tuple):
        raise TypeError("every trajectory state must be a DriftState")
    validated_loss_model = _validate_loss_model(loss_model)
    results = tuple(
        oracle_map_2d(
            values[index],
            state,
            lattice=lattice,
            prior=prior,
            tail_sigma=tail_sigma,
            loss_model=validated_loss_model,
        )
        for index, state in enumerate(state_tuple)
    )
    return OracleTrajectoryResult(
        syndrome=values.copy(),
        parity=np.stack([result.parity for result in results]),
        logical_class=np.array([result.logical_class for result in results], dtype=np.int64),
        logical_action=np.array([result.logical_action for result in results], dtype="<U1"),
        analog_correction=np.stack([result.analog_correction for result in results]),
        posterior=np.stack([result.posterior for result in results]),
        confidence=np.array([result.confidence for result in results], dtype=float),
        log_likelihoods=np.stack([result.log_likelihoods for result in results]),
        state_steps=np.array([state.step for state in state_tuple], dtype=np.int64),
        state_sources=tuple(state.source for state in state_tuple),
        state_regimes=tuple(state.regime for state in state_tuple),
        burst_active=np.array([state.burst_active for state in state_tuple], dtype=bool),
        loss_model=validated_loss_model,
    )


# 简短公共别名，保留文件/任务名与调用语义的一致性。
oracle_map = oracle_map_2d


__all__ = [
    "LossModel",
    "OracleMAPResult",
    "OracleTrajectoryResult",
    "oracle_log_likelihoods_2d",
    "oracle_map_2d",
    "oracle_map",
    "oracle_map_trajectory",
]
