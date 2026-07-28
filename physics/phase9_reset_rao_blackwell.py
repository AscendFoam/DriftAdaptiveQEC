"""Rao--Blackwellized native RESET channel for Phase-9 twin qualification.

The legacy shared-RESET gate compared one independently sampled reset branch
from each backend.  That estimand contains avoidable Bernoulli noise.  This
module replays the *same* native round record twice, forcing the success and
failure branches, and mixes their post-reset densities with the native
pre-reset success probability.  The unmodified sampled branch is returned
only as a nonvoting stress witness.

No transition, measurement, likelihood, reset Kraus operator, or RNG
implementation is shared between backend A and B.  The only common operation
is the linear law of total expectation applied after each native backend has
produced its own conditional branch states.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from physics.phase9_backend_a import backend_a_exogenous
from physics.phase9_backend_b import backend_b_random_record
from physics.phase9_twin_contract import ActionWord


ESTIMAND_ID = "RAO_BLACKWELLIZED_EXPECTED_POST_RESET_DENSITY_AND_LEVELS_V1"
STRESS_ROLE = "SAMPLED_NATIVE_RESET_BRANCH_NONVOTING_STRESS_ONLY"
_ATOL = 2.0e-9


@dataclass(frozen=True)
class RaoBlackwellResetEvidence:
    """One native-backend expected RESET channel evaluation."""

    backend: str
    seed: int
    round_index: int
    success_probability: float
    expected_density: np.ndarray
    expected_levels: tuple[float, float, float]
    success_density: np.ndarray | None
    failure_density: np.ndarray | None
    success_levels: tuple[float, float, float] | None
    failure_levels: tuple[float, float, float] | None
    sampled_density: np.ndarray
    sampled_levels: tuple[float, float, float]
    sampled_hidden_outcome: str
    sampled_matches_forced_branch_trace_distance: float
    branch_trace_distance: float
    pre_reset_levels: tuple[float, float, float]
    sampled_result: object = field(repr=False, compare=False)
    estimand_id: str = ESTIMAND_ID
    sampled_role: str = STRESS_ROLE


def _record(
    backend: str,
    *,
    seed: int,
    round_index: int,
    iq_samples: int,
) -> object:
    if backend == "A":
        return backend_a_exogenous(
            seed=seed,
            round_index=round_index,
            iq_samples=iq_samples,
        )
    if backend == "B":
        return backend_b_random_record(
            seed=seed,
            round_index=round_index,
            iq_samples=iq_samples,
        )
    raise ValueError("backend must be A or B")


def _pre_reset_levels(result: object, backend: str) -> tuple[float, float, float]:
    values = (
        result.truth.pre_reset_level_probabilities
        if backend == "A"
        else result.truth.pre_reset_levels
    )
    return tuple(float(value) for value in values)


def _post_reset_levels(result: object, backend: str) -> tuple[float, float, float]:
    values = (
        result.truth.post_reset_level_probabilities
        if backend == "A"
        else result.truth.post_reset_levels
    )
    return tuple(float(value) for value in values)


def _hidden_outcome(result: object) -> str:
    value = str(result.truth.reset_hidden_outcome)
    if value not in {"success", "failure"}:
        raise RuntimeError("native RESET hidden outcome drift")
    return value


def _half_trace_distance(left: np.ndarray, right: np.ndarray) -> float:
    singular = np.linalg.svd(left - right, compute_uv=False)
    return float(0.5 * np.sum(singular))


def _validated_density(value: Any, *, dimension: int, name: str) -> np.ndarray:
    density = np.asarray(value, dtype=np.complex128)
    if (
        density.shape != (dimension, dimension)
        or not np.all(np.isfinite(density))
        or abs(float(np.trace(density).real) - 1.0) > _ATOL
        or abs(float(np.trace(density).imag)) > _ATOL
        or float(np.linalg.norm(density - density.conj().T, ord="fro")) > _ATOL
        or float(np.min(np.linalg.eigvalsh((density + density.conj().T) / 2.0)))
        < -_ATOL
    ):
        raise RuntimeError(f"{name} physicality drift")
    return density


def _success_probability(
    simulator: object,
    levels: tuple[float, float, float],
) -> float:
    value = (
        levels[0]
        + float(simulator.config.reset_success_e) * levels[1]
        + float(simulator.config.reset_success_f) * levels[2]
    )
    if not -_ATOL <= value <= 1.0 + _ATOL:
        raise RuntimeError("native RESET success probability drift")
    return float(np.clip(value, 0.0, 1.0))


def _run_with_record(
    simulator: object,
    *,
    state: object,
    evaluator: object | None,
    action: ActionWord,
    record: object,
) -> object:
    return simulator.step(state, action, record, evaluator=evaluator)


def evaluate_expected_reset(
    *,
    backend: str,
    simulator: object,
    state: object,
    evaluator: object | None,
    action: ActionWord,
    seed: int,
) -> RaoBlackwellResetEvidence:
    """Evaluate the unconditional native RESET output for one frozen record."""

    if not isinstance(action, ActionWord) or not bool(action.reset_request):
        raise ValueError("Rao-Blackwell RESET estimand requires reset_request")
    round_index = int(state.round_index)
    record = _record(
        backend,
        seed=int(seed),
        round_index=round_index,
        iq_samples=int(simulator.config.iq_samples),
    )
    sampled = _run_with_record(
        simulator,
        state=state,
        evaluator=evaluator,
        action=action,
        record=record,
    )
    sampled_pre = _pre_reset_levels(sampled, backend)
    probability = _success_probability(simulator, sampled_pre)
    dimension = int(state.cutoff) * 3

    success = None
    failure = None
    if probability > 0.0:
        success = _run_with_record(
            simulator,
            state=state,
            evaluator=evaluator,
            action=action,
            record=replace(record, reset_uniform=0.0),
        )
        if _hidden_outcome(success) != "success":
            raise RuntimeError("forced native RESET success branch was not selected")
    if probability < 1.0:
        failure = _run_with_record(
            simulator,
            state=state,
            evaluator=evaluator,
            action=action,
            record=replace(
                record,
                reset_uniform=float(
                    np.nextafter(
                        np.float64(1.0),
                        np.float64(0.0),
                    )
                ),
            ),
        )
        if _hidden_outcome(failure) != "failure":
            raise RuntimeError("forced native RESET failure branch was not selected")

    branch_results = [result for result in (success, failure) if result is not None]
    if not branch_results:
        raise RuntimeError("native RESET produced no admissible branch")
    for result in branch_results:
        if not np.allclose(
            _pre_reset_levels(result, backend),
            sampled_pre,
            rtol=0.0,
            atol=5.0e-13,
        ):
            raise RuntimeError("RESET counterfactual changed the pre-reset state")

    success_density = (
        _validated_density(
            success.state.joint_density,
            dimension=dimension,
            name="success branch",
        )
        if success is not None
        else None
    )
    failure_density = (
        _validated_density(
            failure.state.joint_density,
            dimension=dimension,
            name="failure branch",
        )
        if failure is not None
        else None
    )
    if success_density is None:
        expected = np.array(failure_density, copy=True)
    elif failure_density is None:
        expected = np.array(success_density, copy=True)
    else:
        expected = probability * success_density + (1.0 - probability) * failure_density
    expected = _validated_density(
        expected,
        dimension=dimension,
        name="Rao-Blackwell expected RESET",
    )
    expected_levels = tuple(
        float(value) for value in simulator.level_probabilities(expected)
    )

    success_levels = (
        _post_reset_levels(success, backend) if success is not None else None
    )
    failure_levels = (
        _post_reset_levels(failure, backend) if failure is not None else None
    )
    level_mixture = np.zeros(3, dtype=np.float64)
    if success_levels is not None:
        level_mixture += probability * np.asarray(success_levels)
    if failure_levels is not None:
        level_mixture += (1.0 - probability) * np.asarray(failure_levels)
    if not np.allclose(
        expected_levels,
        level_mixture,
        rtol=0.0,
        atol=5.0e-12,
    ):
        raise RuntimeError("Rao-Blackwell density/level estimand mismatch")

    sampled_density = _validated_density(
        sampled.state.joint_density,
        dimension=dimension,
        name="sampled RESET stress branch",
    )
    sampled_outcome = _hidden_outcome(sampled)
    selected_density = (
        success_density if sampled_outcome == "success" else failure_density
    )
    if selected_density is None:
        raise RuntimeError("sampled RESET selected an impossible branch")
    sampled_match = _half_trace_distance(sampled_density, selected_density)
    if sampled_match > 2.0e-10:
        raise RuntimeError("sampled RESET does not match forced native branch")
    branch_distance = (
        _half_trace_distance(success_density, failure_density)
        if success_density is not None and failure_density is not None
        else 0.0
    )
    return RaoBlackwellResetEvidence(
        backend=backend,
        seed=int(seed),
        round_index=round_index,
        success_probability=probability,
        expected_density=expected,
        expected_levels=expected_levels,
        success_density=success_density,
        failure_density=failure_density,
        success_levels=success_levels,
        failure_levels=failure_levels,
        sampled_density=sampled_density,
        sampled_levels=_post_reset_levels(sampled, backend),
        sampled_hidden_outcome=sampled_outcome,
        sampled_matches_forced_branch_trace_distance=sampled_match,
        branch_trace_distance=branch_distance,
        pre_reset_levels=sampled_pre,
        sampled_result=sampled,
    )


__all__ = [
    "ESTIMAND_ID",
    "RaoBlackwellResetEvidence",
    "STRESS_ROLE",
    "evaluate_expected_reset",
]
