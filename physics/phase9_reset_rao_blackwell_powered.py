"""Additive all-scope Rao--Blackwell RESET adapter for powered T04.

The T03 helper is byte-sealed by historical evidence and must remain unchanged.
This child module consumes that verified branch evaluator, adds a causal
pre-reset receipt, materializes the unconditional primary result, and supports
expected-state continuation through later fault-trajectory rounds.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from hashlib import sha256
import json
from typing import Any

import numpy as np

from physics import phase9_reset_rao_blackwell as _rb
from physics.phase9_reset_rao_blackwell import (
    ESTIMAND_ID,
    RaoBlackwellResetEvidence,
)


ADAPTER_ID = "PHASE9-POWERED-ALL-SCOPE-RB-RESET-ADAPTER-V1"
_BRANCH_PROBABILITY_ATOL = 5.0e-12


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _pre_reset_levels(result: object, backend: str) -> tuple[float, float, float]:
    values = (
        result.truth.pre_reset_level_probabilities
        if backend == "A"
        else result.truth.pre_reset_levels
    )
    return tuple(float(value) for value in values)


def _observation_signature(result: object) -> dict[str, object]:
    observation = result.observation
    return {
        "iq_i": np.asarray(observation.iq_i, dtype="<f8").tolist(),
        "iq_q": np.asarray(observation.iq_q, dtype="<f8").tolist(),
        "integrated_i": float(observation.integrated_i),
        "integrated_q": float(observation.integrated_q),
        "log_evidence_density": float(observation.log_evidence_density),
        "posterior_levels": [
            float(value) for value in observation.posterior_levels
        ],
    }


def _pre_reset_causal_receipt(
    *,
    backend: str,
    state: object,
    record: object,
    sampled: object,
) -> str:
    record_fields = {
        item.name: getattr(record, item.name)
        for item in fields(record)
        if item.name not in {"reset_uniform", "reset_ack_uniform"}
    }
    digest = sha256()
    digest.update(
        np.asarray(state.joint_density, dtype="<c16").tobytes(order="C")
    )
    digest.update(
        _canonical(
            {
                "adapter_id": ADAPTER_ID,
                "estimand_id": ESTIMAND_ID,
                "backend": backend,
                "cutoff": int(state.cutoff),
                "round_index": int(state.round_index),
                "leakage_age": int(state.leakage_age),
                "drift": [float(value) for value in state.drift.vector()],
                "record_without_reset_uniforms": record_fields,
                "observation": _observation_signature(sampled),
                "pre_reset_levels": list(_pre_reset_levels(sampled, backend)),
            }
        )
    )
    return digest.hexdigest()


def _density_diagnostics(density: np.ndarray, backend: str) -> dict[str, float]:
    hermitian = (density + density.conj().T) / 2.0
    eigenvalues = np.linalg.eigvalsh(hermitian)
    diagnostics = {
        "trace_real": float(np.trace(density).real),
        "trace_imag": float(np.trace(density).imag),
        "hermiticity_frobenius": float(
            np.linalg.norm(density - density.conj().T, ord="fro")
        ),
        "minimum_eigenvalue": float(np.min(eigenvalues)),
        "purity": float(np.trace(density @ density).real),
    }
    if backend == "A":
        diagnostics["maximum_eigenvalue"] = float(np.max(eigenvalues))
    return diagnostics


def _run_forced(
    *,
    simulator: object,
    state: object,
    evaluator: object | None,
    action: object,
    record: object,
    success: bool,
) -> object:
    forced = replace(
        record,
        reset_uniform=(
            0.0
            if success
            else float(np.nextafter(np.float64(1.0), np.float64(0.0)))
        ),
    )
    return simulator.step(state, action, forced, evaluator=evaluator)


def _boundary_stable_probability(value: float) -> float:
    """Snap only numerically null RESET branches to their exact boundary."""

    probability = float(value)
    if not np.isfinite(probability):
        raise RuntimeError("native RESET success probability is non-finite")
    if probability < -_BRANCH_PROBABILITY_ATOL or probability > (
        1.0 + _BRANCH_PROBABILITY_ATOL
    ):
        raise RuntimeError("native RESET success probability drift")
    probability = float(np.clip(probability, 0.0, 1.0))
    if probability <= _BRANCH_PROBABILITY_ATOL:
        return 0.0
    if (1.0 - probability) <= _BRANCH_PROBABILITY_ATOL:
        return 1.0
    return probability


def _evaluate_boundary_stable_reset(
    *,
    backend: str,
    simulator: object,
    state: object,
    evaluator: object | None,
    action: object,
    seed: int,
) -> RaoBlackwellResetEvidence:
    """Replay the T03 native branches with an explicit null-branch tolerance.

    T03 deliberately remains byte-immutable.  Its strict ``p < 1`` check can
    request a failure branch when the analytical failure probability is only
    floating-point residue after an earlier expected-RESET continuation.  The
    T04 child contract treats probabilities within 5e-12 of a boundary as an
    exactly null branch, while preserving every non-null branch and all T03
    density, mixture, and sampled-branch checks.
    """

    if not isinstance(action, _rb.ActionWord) or not bool(action.reset_request):
        raise ValueError("Rao-Blackwell RESET estimand requires reset_request")
    round_index = int(state.round_index)
    record = _rb._record(
        backend,
        seed=int(seed),
        round_index=round_index,
        iq_samples=int(simulator.config.iq_samples),
    )
    sampled = _rb._run_with_record(
        simulator,
        state=state,
        evaluator=evaluator,
        action=action,
        record=record,
    )
    sampled_pre = _rb._pre_reset_levels(sampled, backend)
    probability = _boundary_stable_probability(
        _rb._success_probability(simulator, sampled_pre)
    )
    dimension = int(state.cutoff) * 3

    success = None
    failure = None
    if probability > 0.0:
        success = _rb._run_with_record(
            simulator,
            state=state,
            evaluator=evaluator,
            action=action,
            record=replace(record, reset_uniform=0.0),
        )
        if _rb._hidden_outcome(success) != "success":
            raise RuntimeError("forced native RESET success branch was not selected")
    if probability < 1.0:
        failure = _rb._run_with_record(
            simulator,
            state=state,
            evaluator=evaluator,
            action=action,
            record=replace(
                record,
                reset_uniform=float(
                    np.nextafter(np.float64(1.0), np.float64(0.0))
                ),
            ),
        )
        if _rb._hidden_outcome(failure) != "failure":
            raise RuntimeError("forced native RESET failure branch was not selected")

    branch_results = [result for result in (success, failure) if result is not None]
    if not branch_results:
        raise RuntimeError("native RESET produced no admissible branch")
    for result in branch_results:
        if not np.allclose(
            _rb._pre_reset_levels(result, backend),
            sampled_pre,
            rtol=0.0,
            atol=5.0e-13,
        ):
            raise RuntimeError("RESET counterfactual changed the pre-reset state")

    success_density = (
        _rb._validated_density(
            success.state.joint_density,
            dimension=dimension,
            name="success branch",
        )
        if success is not None
        else None
    )
    failure_density = (
        _rb._validated_density(
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
        expected = (
            probability * success_density
            + (1.0 - probability) * failure_density
        )
    expected = _rb._validated_density(
        expected,
        dimension=dimension,
        name="Rao-Blackwell expected RESET",
    )
    expected_levels = tuple(
        float(value) for value in simulator.level_probabilities(expected)
    )
    success_levels = (
        _rb._post_reset_levels(success, backend) if success is not None else None
    )
    failure_levels = (
        _rb._post_reset_levels(failure, backend) if failure is not None else None
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

    sampled_density = _rb._validated_density(
        sampled.state.joint_density,
        dimension=dimension,
        name="sampled RESET stress branch",
    )
    sampled_outcome = _rb._hidden_outcome(sampled)
    selected_density = (
        success_density if sampled_outcome == "success" else failure_density
    )
    if selected_density is None:
        raise RuntimeError("sampled RESET selected a numerically null branch")
    sampled_match = _rb._half_trace_distance(sampled_density, selected_density)
    if sampled_match > 2.0e-10:
        raise RuntimeError("sampled RESET does not match forced native branch")
    branch_distance = (
        _rb._half_trace_distance(success_density, failure_density)
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
        sampled_levels=_rb._post_reset_levels(sampled, backend),
        sampled_hidden_outcome=sampled_outcome,
        sampled_matches_forced_branch_trace_distance=sampled_match,
        branch_trace_distance=branch_distance,
        pre_reset_levels=sampled_pre,
        sampled_result=sampled,
    )


@dataclass(frozen=True)
class PoweredResetEvidence:
    """T04 wrapper around the immutable T03 branch evidence."""

    base: RaoBlackwellResetEvidence
    input_leakage_age: int
    pre_reset_causal_receipt_sha256: str
    adapter_id: str = ADAPTER_ID

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)


def evaluate_expected_reset_powered(
    *,
    backend: str,
    simulator: object,
    state: object,
    evaluator: object | None,
    action: object,
    seed: int,
) -> PoweredResetEvidence:
    """Evaluate branches and independently seal their pre-reset equivalence."""

    base = _evaluate_boundary_stable_reset(
        backend=backend,
        simulator=simulator,
        state=state,
        evaluator=evaluator,
        action=action,
        seed=seed,
    )
    sampled = base.sampled_result
    record = sampled.exogenous if backend == "A" else sampled.random_record
    signatures = [_observation_signature(sampled)]
    if base.success_density is not None:
        success = _run_forced(
            simulator=simulator,
            state=state,
            evaluator=evaluator,
            action=action,
            record=record,
            success=True,
        )
        signatures.append(_observation_signature(success))
        if not np.allclose(
            _pre_reset_levels(success, backend),
            base.pre_reset_levels,
            rtol=0.0,
            atol=5.0e-13,
        ):
            raise RuntimeError("powered RESET success changed pre-reset law")
    if base.failure_density is not None:
        failure = _run_forced(
            simulator=simulator,
            state=state,
            evaluator=evaluator,
            action=action,
            record=record,
            success=False,
        )
        signatures.append(_observation_signature(failure))
        if not np.allclose(
            _pre_reset_levels(failure, backend),
            base.pre_reset_levels,
            rtol=0.0,
            atol=5.0e-13,
        ):
            raise RuntimeError("powered RESET failure changed pre-reset law")
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise RuntimeError("powered RESET branch changed analog observation")
    receipt = _pre_reset_causal_receipt(
        backend=backend,
        state=state,
        record=record,
        sampled=sampled,
    )
    return PoweredResetEvidence(
        base=base,
        input_leakage_age=int(state.leakage_age),
        pre_reset_causal_receipt_sha256=receipt,
    )


def expected_primary_result(
    evidence: PoweredResetEvidence,
    *,
    simulator: object,
) -> object:
    """Construct the unconditional primary row and continuation state."""

    if not isinstance(evidence, PoweredResetEvidence):
        raise TypeError("powered RESET evidence wrapper is required")
    sampled = evidence.sampled_result
    expected_density = np.asarray(evidence.expected_density, dtype=np.complex128)
    leakage_age = (
        min(int(evidence.input_leakage_age) + 1, 65535)
        if evidence.expected_levels[2]
        >= float(simulator.config.leakage_age_threshold)
        else 0
    )
    expected_state = replace(
        sampled.state,
        joint_density=expected_density,
        leakage_age=leakage_age,
    )
    logical = None
    if sampled.logical is not None:
        evaluator = (
            sampled.logical.evaluator_state
            if evidence.backend == "A"
            else sampled.logical.evaluator
        )
        logical = simulator.logical_record(expected_state, evaluator)
    truth_updates: dict[str, object] = {
        "density_diagnostics": _density_diagnostics(
            expected_density, evidence.backend
        )
    }
    if evidence.backend == "A":
        truth_updates["post_reset_level_probabilities"] = evidence.expected_levels
    else:
        truth_updates["post_reset_levels"] = evidence.expected_levels
    expected_truth = replace(sampled.truth, **truth_updates)
    return replace(
        sampled,
        state=expected_state,
        truth=expected_truth,
        logical=logical,
    )


__all__ = [
    "ADAPTER_ID",
    "PoweredResetEvidence",
    "evaluate_expected_reset_powered",
    "expected_primary_result",
]
