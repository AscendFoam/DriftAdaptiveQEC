"""Feedback-GRAPE reward/score gradient verification for T2.3.5.

For a discrete measurement trajectory ``m`` the exact identity is

    d E[R] / d theta
      = E[d R / d theta] + E[R d log P_theta(m) / d theta].

The module verifies this identity by complete branch enumeration in a small
finite-cutoff model, compares automatic differentiation against central finite
differences, and then checks a genuinely sampled Monte Carlo estimator with a
constant control-variate baseline.  It does not train an RNN or claim the
cutoff/batch/horizon feasibility and NMF ranking reserved for T2.3.6/T2.3.7.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from itertools import product
from math import isfinite
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .differentiable_sbs_trajectory import (
    DIFFERENTIABLE_SBS_SCOPE,
    PARAMETER_NAMES,
    DifferentiableSBSConfig,
    DifferentiableSBSTrajectorySimulator,
)
from .sbs_error_space import SBS_PROTOCOL_ID

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - minimal recovery environment.
    torch = None  # type: ignore[assignment]


FEEDBACK_GRAPE_GRADIENT_SCOPE = (
    "exact-enumeration plus finite-difference and sampled-estimator validation of "
    "Feedback-GRAPE reward and trajectory-score gradients on the finite-cutoff "
    "two-level differentiable SBS model; not teacher training, feasibility envelope, "
    "protocol ranking, device calibration, or hardware evidence"
)
COMPACT_POLICY_PARAMETER_NAMES = (
    "static_control_residual",
    "latest_outcome_response",
    "history_mean_response",
)


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("T2.3.5 requires PyTorch; use the local DLEnv/QuantumEnv")
    return torch


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


@dataclass(frozen=True)
class GradientValidationConfig:
    cutoff: int = 6
    full_cycles: int = 1
    finite_difference_step: float = 1.0e-5
    finite_difference_relative_tolerance: float = 5.0e-5
    finite_difference_absolute_tolerance: float = 2.0e-6
    decomposition_tolerance: float = 2.0e-10
    monte_carlo_batch_size: int = 384
    monte_carlo_repeats: int = 32
    monte_carlo_max_z: float = 3.5
    seed: int = 271828
    device: str = "cpu"
    protocol_id: str = SBS_PROTOCOL_ID
    scope: str = FEEDBACK_GRAPE_GRADIENT_SCOPE

    def __post_init__(self) -> None:
        cutoff = _positive_int(self.cutoff, "cutoff")
        if not 4 <= cutoff <= 16:
            raise ValueError("gradient-validation cutoff must lie in [4, 16]")
        object.__setattr__(self, "cutoff", cutoff)
        cycles = _positive_int(self.full_cycles, "full_cycles")
        if cycles > 2:
            raise ValueError("exact gradient validation is bounded to one or two cycles")
        object.__setattr__(self, "full_cycles", cycles)
        for name in (
            "finite_difference_step",
            "finite_difference_relative_tolerance",
            "finite_difference_absolute_tolerance",
            "decomposition_tolerance",
            "monte_carlo_max_z",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        if self.finite_difference_step >= 1.0e-2:
            raise ValueError("finite_difference_step must be below 1e-2")
        batch = _positive_int(self.monte_carlo_batch_size, "monte_carlo_batch_size")
        repeats = _positive_int(self.monte_carlo_repeats, "monte_carlo_repeats")
        if batch < 64:
            raise ValueError("Monte Carlo batch must be at least 64")
        if repeats < 8:
            raise ValueError("Monte Carlo repeats must be at least 8")
        object.__setattr__(self, "monte_carlo_batch_size", batch)
        object.__setattr__(self, "monte_carlo_repeats", repeats)
        if isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)):
            raise TypeError("seed must be an integer")
        object.__setattr__(self, "seed", int(self.seed))
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if self.protocol_id != SBS_PROTOCOL_ID:
            raise ValueError("T2.3.5 validates only the frozen SBS main protocol")
        if self.scope != FEEDBACK_GRAPE_GRADIENT_SCOPE:
            raise ValueError("scope must preserve the fail-closed gradient boundary")

    @property
    def half_cycles(self) -> int:
        return 2 * self.full_cycles

    @property
    def branch_count(self) -> int:
        return 2**self.half_cycles


class CompactHistoryPolicy:
    """Three-parameter causal policy used only for numerical gradient auditing."""

    def __init__(self, theta: Any) -> None:
        th = _require_torch()
        if not isinstance(theta, th.Tensor):
            raise TypeError("theta must be a torch.Tensor")
        if tuple(theta.shape) != (len(COMPACT_POLICY_PARAMETER_NAMES),):
            raise ValueError("theta must contain the three compact policy parameters")
        if not bool(th.all(th.isfinite(theta)).detach().cpu()):
            raise ValueError("theta must be finite")
        self.theta = theta
        control_index = th.arange(
            len(PARAMETER_NAMES), device=theta.device, dtype=theta.dtype
        )
        self.static_basis = 0.22 * th.sin(0.41 * (control_index + 1.0))
        self.latest_basis = 0.18 * th.cos(0.67 * (control_index + 0.5))
        self.memory_basis = 0.16 * th.sin(0.29 * (control_index + 1.7))

    def __call__(self, history: Any, half_index: int) -> Any:
        th = _require_torch()
        if not isinstance(history, th.Tensor) or history.ndim != 2:
            raise TypeError("history must be a rank-two torch.Tensor")
        if history.shape[1] != half_index:
            raise ValueError("history width must equal the causal half-cycle index")
        batch = history.shape[0]
        if half_index == 0:
            latest = th.zeros((batch, 1), dtype=self.theta.dtype, device=self.theta.device)
            history_mean = latest
        else:
            signed = 2.0 * history.to(self.theta.dtype) - 1.0
            latest = signed[:, -1:]
            history_mean = signed.mean(dim=1, keepdim=True)
        return (
            self.theta[0] * self.static_basis[None, :]
            + self.theta[1] * latest * self.latest_basis[None, :]
            + self.theta[2] * history_mean * self.memory_basis[None, :]
        )


@dataclass(frozen=True)
class ExactGradientDecomposition:
    expected_return: float
    expected_reward: float
    trajectory_probability_sum: float
    exact_gradient: tuple[float, ...]
    reward_path_gradient: tuple[float, ...]
    score_path_gradient: tuple[float, ...]
    decomposed_gradient: tuple[float, ...]
    baseline_score_gradient: tuple[float, ...]
    probability_normalization_score: tuple[float, ...]
    decomposition_absolute_error: float
    baseline_invariance_error: float
    score_normalization_error: float


@dataclass(frozen=True)
class FiniteDifferenceGradientResult:
    step: float
    autograd_gradient: tuple[float, ...]
    finite_difference_gradient: tuple[float, ...]
    reward_path_autograd_gradient: tuple[float, ...]
    reward_path_finite_difference_gradient: tuple[float, ...]
    score_path_autograd_gradient: tuple[float, ...]
    score_path_finite_difference_gradient: tuple[float, ...]
    absolute_error_by_parameter: tuple[float, ...]
    relative_error_by_parameter: tuple[float, ...]
    maximum_absolute_error: float
    relative_l2_error: float
    reward_path_relative_l2_error: float
    score_path_relative_l2_error: float


@dataclass(frozen=True)
class MonteCarloGradientResult:
    batch_size: int
    repeats: int
    total_trajectories: int
    exact_gradient: tuple[float, ...]
    sampled_gradient_mean: tuple[float, ...]
    sampled_gradient_standard_error: tuple[float, ...]
    sampled_gradient_z_score: tuple[float, ...]
    maximum_absolute_z_score: float
    plain_score_trace_variance: float
    baseline_score_trace_variance: float
    baseline_variance_ratio: float
    mean_reward: float
    mean_ground_outcome_fraction: float


def enumerate_binary_trajectories(half_cycles: int, *, device: str = "cpu") -> Any:
    th = _require_torch()
    count = _positive_int(half_cycles, "half_cycles")
    if count > 12:
        raise ValueError("exact binary enumeration is capped at twelve outcomes")
    return th.tensor(
        tuple(product((0, 1), repeat=count)),
        dtype=th.int64,
        device=device,
    )


def default_policy_parameters(*, device: str = "cpu", requires_grad: bool = True) -> Any:
    th = _require_torch()
    return th.tensor(
        (0.11, -0.17, 0.09),
        dtype=th.float64,
        device=device,
        requires_grad=requires_grad,
    )


def _simulator_config(
    config: GradientValidationConfig, *, batch_size: int
) -> DifferentiableSBSConfig:
    return DifferentiableSBSConfig(
        cutoff=config.cutoff,
        full_cycles=config.full_cycles,
        batch_size=batch_size,
        grid_points=2049,
        device=config.device,
        real_dtype="float64",
    )


def exact_gradient_decomposition(
    config: GradientValidationConfig,
    theta: Any | None = None,
) -> ExactGradientDecomposition:
    """Enumerate every branch and split exact dE[R] into both GRAPE terms."""

    th = _require_torch()
    parameters = (
        default_policy_parameters(device=config.device)
        if theta is None
        else theta
    )
    if not isinstance(parameters, th.Tensor) or not parameters.requires_grad:
        raise ValueError("theta must be a requires_grad torch.Tensor")
    outcomes = enumerate_binary_trajectories(config.half_cycles, device=config.device)
    simulator = DifferentiableSBSTrajectorySimulator(
        _simulator_config(config, batch_size=config.branch_count)
    )
    result = simulator.run(
        control_policy=CompactHistoryPolicy(parameters),
        forced_outcomes=outcomes,
        seed=config.seed,
    )
    probability = result.trajectory_probability
    reward = result.reward
    expected_return_tensor = th.sum(probability * reward)
    exact = th.autograd.grad(expected_return_tensor, parameters, retain_graph=True)[0]
    reward_objective = th.sum(probability.detach() * reward)
    reward_path = th.autograd.grad(reward_objective, parameters, retain_graph=True)[0]
    score_objective = th.sum(
        probability.detach() * reward.detach() * result.log_probability
    )
    score_path = th.autograd.grad(score_objective, parameters, retain_graph=True)[0]
    baseline = expected_return_tensor.detach()
    baseline_score_objective = th.sum(
        probability.detach()
        * (reward.detach() - baseline)
        * result.log_probability
    )
    baseline_score = th.autograd.grad(
        baseline_score_objective, parameters, retain_graph=True
    )[0]
    normalization_objective = th.sum(
        probability.detach() * result.log_probability
    )
    normalization_score = th.autograd.grad(normalization_objective, parameters)[0]
    decomposed = reward_path + score_path
    decomposition_error = float(th.max(th.abs(exact - decomposed)).detach().cpu())
    baseline_error = float(th.max(th.abs(score_path - baseline_score)).detach().cpu())
    normalization_error = float(th.max(th.abs(normalization_score)).detach().cpu())

    def values(tensor: Any) -> tuple[float, ...]:
        return tuple(float(value) for value in tensor.detach().cpu().tolist())

    return ExactGradientDecomposition(
        expected_return=float(expected_return_tensor.detach().cpu()),
        expected_reward=float(expected_return_tensor.detach().cpu()),
        trajectory_probability_sum=float(probability.detach().sum().cpu()),
        exact_gradient=values(exact),
        reward_path_gradient=values(reward_path),
        score_path_gradient=values(score_path),
        decomposed_gradient=values(decomposed),
        baseline_score_gradient=values(baseline_score),
        probability_normalization_score=values(normalization_score),
        decomposition_absolute_error=decomposition_error,
        baseline_invariance_error=baseline_error,
        score_normalization_error=normalization_error,
    )


def _expected_return_for_values(
    simulator: DifferentiableSBSTrajectorySimulator,
    outcomes: Any,
    values: Any,
    seed: int,
) -> float:
    th = _require_torch()
    with th.no_grad():
        result = simulator.run(
            control_policy=CompactHistoryPolicy(values),
            forced_outcomes=outcomes,
            seed=seed,
        )
        return float(th.sum(result.trajectory_probability * result.reward).cpu())


def _branch_values_for_values(
    simulator: DifferentiableSBSTrajectorySimulator,
    outcomes: Any,
    values: Any,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    th = _require_torch()
    with th.no_grad():
        result = simulator.run(
            control_policy=CompactHistoryPolicy(values),
            forced_outcomes=outcomes,
            seed=seed,
        )
        return (
            result.trajectory_probability.cpu().numpy().astype(np.float64, copy=True),
            result.reward.cpu().numpy().astype(np.float64, copy=True),
        )


def finite_difference_gradient(
    config: GradientValidationConfig,
    autograd_gradient: Sequence[float] | None = None,
    reward_path_gradient: Sequence[float] | None = None,
    score_path_gradient: Sequence[float] | None = None,
    theta_values: Any | None = None,
) -> FiniteDifferenceGradientResult:
    """Central-difference the fully enumerated expected return."""

    th = _require_torch()
    values = (
        default_policy_parameters(device=config.device, requires_grad=False)
        if theta_values is None
        else theta_values.detach().to(device=config.device, dtype=th.float64)
    )
    outcomes = enumerate_binary_trajectories(config.half_cycles, device=config.device)
    simulator = DifferentiableSBSTrajectorySimulator(
        _simulator_config(config, batch_size=config.branch_count)
    )
    step = config.finite_difference_step
    finite_difference = []
    reward_path_finite_difference = []
    score_path_finite_difference = []
    base_probability, base_reward = _branch_values_for_values(
        simulator, outcomes, values, config.seed
    )
    for index in range(values.numel()):
        plus = values.clone()
        minus = values.clone()
        plus[index] += step
        minus[index] -= step
        plus_probability, plus_reward = _branch_values_for_values(
            simulator, outcomes, plus, config.seed
        )
        minus_probability, minus_reward = _branch_values_for_values(
            simulator, outcomes, minus, config.seed
        )
        upper = float(np.dot(plus_probability, plus_reward))
        lower = float(np.dot(minus_probability, minus_reward))
        finite_difference.append((upper - lower) / (2.0 * step))
        reward_path_finite_difference.append(
            float(np.dot(base_probability, plus_reward - minus_reward) / (2.0 * step))
        )
        score_path_finite_difference.append(
            float(np.dot(base_reward, plus_probability - minus_probability) / (2.0 * step))
        )
    if (
        autograd_gradient is None
        or reward_path_gradient is None
        or score_path_gradient is None
    ):
        theta = values.clone().requires_grad_(True)
        exact = exact_gradient_decomposition(config, theta)
        automatic = np.asarray(exact.exact_gradient, dtype=np.float64)
        reward_automatic = np.asarray(exact.reward_path_gradient, dtype=np.float64)
        score_automatic = np.asarray(exact.score_path_gradient, dtype=np.float64)
    else:
        automatic = np.asarray(tuple(autograd_gradient), dtype=np.float64)
        reward_automatic = np.asarray(tuple(reward_path_gradient), dtype=np.float64)
        score_automatic = np.asarray(tuple(score_path_gradient), dtype=np.float64)
    finite = np.asarray(finite_difference, dtype=np.float64)
    reward_finite = np.asarray(reward_path_finite_difference, dtype=np.float64)
    score_finite = np.asarray(score_path_finite_difference, dtype=np.float64)
    if any(
        item.shape != finite.shape
        for item in (automatic, reward_automatic, score_automatic)
    ):
        raise ValueError("autograd/reward/score gradient has the wrong shape")
    absolute = np.abs(automatic - finite)
    relative = absolute / np.maximum(np.abs(finite), 1.0e-10)
    relative_l2 = float(np.linalg.norm(automatic - finite) / max(np.linalg.norm(finite), 1.0e-12))
    reward_relative_l2 = float(
        np.linalg.norm(reward_automatic - reward_finite)
        / max(np.linalg.norm(reward_finite), 1.0e-12)
    )
    score_relative_l2 = float(
        np.linalg.norm(score_automatic - score_finite)
        / max(np.linalg.norm(score_finite), 1.0e-12)
    )
    return FiniteDifferenceGradientResult(
        step=step,
        autograd_gradient=tuple(float(value) for value in automatic),
        finite_difference_gradient=tuple(float(value) for value in finite),
        reward_path_autograd_gradient=tuple(float(value) for value in reward_automatic),
        reward_path_finite_difference_gradient=tuple(float(value) for value in reward_finite),
        score_path_autograd_gradient=tuple(float(value) for value in score_automatic),
        score_path_finite_difference_gradient=tuple(float(value) for value in score_finite),
        absolute_error_by_parameter=tuple(float(value) for value in absolute),
        relative_error_by_parameter=tuple(float(value) for value in relative),
        maximum_absolute_error=float(np.max(absolute)),
        relative_l2_error=relative_l2,
        reward_path_relative_l2_error=reward_relative_l2,
        score_path_relative_l2_error=score_relative_l2,
    )


def finite_difference_step_sweep(
    config: GradientValidationConfig,
    exact: ExactGradientDecomposition,
    steps: Sequence[float] = (3.0e-4, 1.0e-4, 3.0e-5, 1.0e-5),
) -> tuple[FiniteDifferenceGradientResult, ...]:
    if not steps:
        raise ValueError("finite-difference step sweep must not be empty")
    results = []
    for step in steps:
        results.append(
            finite_difference_gradient(
                replace(config, finite_difference_step=float(step)),
                exact.exact_gradient,
                exact.reward_path_gradient,
                exact.score_path_gradient,
            )
        )
    return tuple(results)


def monte_carlo_gradient_validation(
    config: GradientValidationConfig,
    exact_gradient: Sequence[float],
    baseline: float,
) -> MonteCarloGradientResult:
    """Estimate both stochastic terms repeatedly and compare their mean to exact."""

    th = _require_torch()
    simulator = DifferentiableSBSTrajectorySimulator(
        _simulator_config(config, batch_size=config.monte_carlo_batch_size)
    )
    combined_samples = []
    plain_score_samples = []
    baseline_score_samples = []
    rewards = []
    ground_fractions = []
    for repeat in range(config.monte_carlo_repeats):
        theta = default_policy_parameters(device=config.device, requires_grad=True)
        result = simulator.run(
            control_policy=CompactHistoryPolicy(theta),
            seed=config.seed + repeat,
        )
        reward_path = th.autograd.grad(
            result.reward.mean(), theta, retain_graph=True
        )[0]
        plain_score = th.autograd.grad(
            th.mean(result.reward.detach() * result.log_probability),
            theta,
            retain_graph=True,
        )[0]
        baseline_score = th.autograd.grad(
            th.mean((result.reward.detach() - baseline) * result.log_probability),
            theta,
        )[0]
        combined_samples.append((reward_path + baseline_score).detach().cpu().numpy())
        plain_score_samples.append(plain_score.detach().cpu().numpy())
        baseline_score_samples.append(baseline_score.detach().cpu().numpy())
        rewards.append(float(result.reward.detach().mean().cpu()))
        ground_fractions.append(float((result.outcomes == 0).double().mean().cpu()))
    combined = np.asarray(combined_samples, dtype=np.float64)
    plain_score_array = np.asarray(plain_score_samples, dtype=np.float64)
    baseline_score_array = np.asarray(baseline_score_samples, dtype=np.float64)
    exact = np.asarray(tuple(exact_gradient), dtype=np.float64)
    if exact.shape != (len(COMPACT_POLICY_PARAMETER_NAMES),):
        raise ValueError("exact_gradient has the wrong shape")
    mean = np.mean(combined, axis=0)
    standard_error = np.std(combined, axis=0, ddof=1) / np.sqrt(config.monte_carlo_repeats)
    z_score = (mean - exact) / np.maximum(standard_error, 1.0e-12)
    plain_variance = float(np.sum(np.var(plain_score_array, axis=0, ddof=1)))
    baseline_variance = float(np.sum(np.var(baseline_score_array, axis=0, ddof=1)))
    return MonteCarloGradientResult(
        batch_size=config.monte_carlo_batch_size,
        repeats=config.monte_carlo_repeats,
        total_trajectories=config.monte_carlo_batch_size * config.monte_carlo_repeats,
        exact_gradient=tuple(float(value) for value in exact),
        sampled_gradient_mean=tuple(float(value) for value in mean),
        sampled_gradient_standard_error=tuple(float(value) for value in standard_error),
        sampled_gradient_z_score=tuple(float(value) for value in z_score),
        maximum_absolute_z_score=float(np.max(np.abs(z_score))),
        plain_score_trace_variance=plain_variance,
        baseline_score_trace_variance=baseline_variance,
        baseline_variance_ratio=(
            baseline_variance / plain_variance if plain_variance > 0.0 else 0.0
        ),
        mean_reward=float(np.mean(rewards)),
        mean_ground_outcome_fraction=float(np.mean(ground_fractions)),
    )


def run_feedback_grape_gradient_validation(
    *,
    device: str = "cpu",
    output: str | Path | None = None,
    monte_carlo_batch_size: int = 384,
    monte_carlo_repeats: int = 32,
) -> dict[str, Any]:
    config = GradientValidationConfig(
        device=device,
        monte_carlo_batch_size=monte_carlo_batch_size,
        monte_carlo_repeats=monte_carlo_repeats,
    )
    exact = exact_gradient_decomposition(config)
    finite = finite_difference_gradient(
        config,
        exact.exact_gradient,
        exact.reward_path_gradient,
        exact.score_path_gradient,
    )
    step_sweep = finite_difference_step_sweep(config, exact)
    monte_carlo = monte_carlo_gradient_validation(
        config, exact.exact_gradient, exact.expected_return
    )
    reward_norm = float(np.linalg.norm(exact.reward_path_gradient))
    score_norm = float(np.linalg.norm(exact.score_path_gradient))
    checks = {
        "all_exact_trajectory_probabilities_normalize": abs(
            exact.trajectory_probability_sum - 1.0
        )
        < config.decomposition_tolerance,
        "reward_path_is_nonzero": reward_norm > 1.0e-8,
        "score_path_is_nonzero": score_norm > 1.0e-8,
        "exact_gradient_equals_reward_plus_score": exact.decomposition_absolute_error
        < config.decomposition_tolerance,
        "constant_baseline_preserves_exact_score_gradient": exact.baseline_invariance_error
        < config.decomposition_tolerance,
        "expected_probability_score_is_zero": exact.score_normalization_error
        < config.decomposition_tolerance,
        "autograd_matches_central_finite_difference_relative": finite.relative_l2_error
        < config.finite_difference_relative_tolerance,
        "autograd_matches_central_finite_difference_absolute": finite.maximum_absolute_error
        < config.finite_difference_absolute_tolerance,
        "reward_path_matches_independent_finite_difference": finite.reward_path_relative_l2_error
        < config.finite_difference_relative_tolerance,
        "score_path_matches_independent_finite_difference": finite.score_path_relative_l2_error
        < config.finite_difference_relative_tolerance,
        "finite_difference_step_sweep_is_stable": max(
            item.relative_l2_error for item in step_sweep
        )
        < config.finite_difference_relative_tolerance,
        "sampled_two_term_gradient_matches_exact_within_error": monte_carlo.maximum_absolute_z_score
        < config.monte_carlo_max_z,
        "constant_baseline_reduces_score_variance": monte_carlo.baseline_variance_ratio
        < 1.0,
        "monte_carlo_uses_real_random_trajectories": 0.0
        < monte_carlo.mean_ground_outcome_fraction
        < 1.0,
        "upstream_simulator_scope_is_preserved": DIFFERENTIABLE_SBS_SCOPE.startswith(
            "finite-cutoff joint cavity-two-level-ancilla"
        ),
    }
    payload = {
        "task_id": "T2.3.5",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "source_contract": {
            "paper_equation": (
                "relative_papers/Non-Markovian_feedback_for_optimized_quantum_error_"
                "correction/Non-Markovian_feedback_for_optimized_quantum_error_"
                "correction.md:467-495"
            ),
            "identity": (
                "d E[R]/d theta = E[dR/dtheta] + "
                "E[R d log P_theta(trajectory)/dtheta]"
            ),
        },
        "config": asdict(config),
        "policy_parameters": COMPACT_POLICY_PARAMETER_NAMES,
        "exact_decomposition": asdict(exact),
        "finite_difference": asdict(finite),
        "finite_difference_step_sweep": [asdict(item) for item in step_sweep],
        "monte_carlo": asdict(monte_carlo),
        "checks": checks,
        "scope": FEEDBACK_GRAPE_GRADIENT_SCOPE,
        "forbidden_claims": (
            "not an optimized or trained RNN teacher",
            "not the T2.3.6 cutoff batch horizon feasibility envelope",
            "not the T2.3.7 standard MF NMF ranking",
            "not pulse-level multilevel-transmon or device-calibrated dynamics",
            "not target-board or hardware timing evidence",
        ),
    }
    if output is not None:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--repeats", type=int, default=32)
    parser.add_argument(
        "--output", default="docs/t2_3_5_feedback_grape_gradient_validation.json"
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = run_feedback_grape_gradient_validation(
        device=args.device,
        output=args.output,
        monte_carlo_batch_size=args.batch_size,
        monte_carlo_repeats=args.repeats,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FEEDBACK_GRAPE_GRADIENT_SCOPE",
    "COMPACT_POLICY_PARAMETER_NAMES",
    "GradientValidationConfig",
    "CompactHistoryPolicy",
    "ExactGradientDecomposition",
    "FiniteDifferenceGradientResult",
    "MonteCarloGradientResult",
    "enumerate_binary_trajectories",
    "default_policy_parameters",
    "exact_gradient_decomposition",
    "finite_difference_gradient",
    "finite_difference_step_sweep",
    "monte_carlo_gradient_validation",
    "run_feedback_grape_gradient_validation",
]
