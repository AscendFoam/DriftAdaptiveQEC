"""T3.1.4 static sBs observation/reset-aware branch decoder.

The deployable decoder receives only the current classified sBs pair
``(x, z) in {g, e, leakage}^2``.  Its likelihood table is computed exactly
from a frozen preparation/readout/reset kernel and an iid ideal-branch prior.
Persistent hidden ancilla carry is *marginalized* with the unique stationary
distribution; it is never supplied to :meth:`StaticSBSBranchDecoder.decode`.

The target is the ideal four-way sBs Kraus branch, not a logical Pauli class.
Consequently this module is a protocol branch-selection baseline and must not
be described as a full finite-energy quantum recovery optimum.  The optional
``FALLBACK`` action is allowed only after an observed leakage classification,
and its explicit loss is included in every comparison.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from physics.sbs_error_space import SBS_OUTCOMES
from physics.sbs_observation_reset import (
    HIDDEN_ANCILLA_STATES,
    OBSERVED_CLASSES,
    HiddenAncillaMemory,
    PairedSyndrome,
    SBSObservationResetModel,
    SBSObservedCycle,
    ideal_syndrome_from_kraus,
    make_persistent_leakage_model,
)


STATIC_SBS_DECODER_ID = "static_sbs_observation_reset_bayes"
STANDARD_SBS_BRANCH_ID = "direct_observed_sbs_branch"
STATIC_SBS_MAP_ID = "static_sbs_branch_map"
IDEAL_BRANCH_REFERENCE_ID = "ideal_sbs_branch_truth_reference"
FALLBACK_ACTION = "FALLBACK"
_OBSERVATION_PAIRS = tuple(
    (x_value, z_value)
    for x_value in OBSERVED_CLASSES
    for z_value in OBSERVED_CLASSES
)


@dataclass(frozen=True)
class StaticSBSDecoderDescriptor:
    baseline_id: str = STATIC_SBS_DECODER_ID
    label: str = "Static sBs observation/reset-aware Bayes policy"
    task_owner: str = "T3.1.4"
    comparison_role: str = "deployable_static_protocol_branch_decoder_baseline"
    deployable: bool = True
    online_inputs: tuple[str, ...] = (
        "current_observed_sbs_class_x",
        "current_observed_sbs_class_z",
    )
    offline_calibration_inputs: tuple[str, ...] = (
        "ideal_branch_prior",
        "preparation_kernel",
        "readout_confusion",
        "conditional_reset_kernel",
        "observed_leakage_fallback_cost",
    )
    hidden_truth_online_inputs: tuple[str, ...] = ()
    update_during_evaluation: bool = False
    action_space: tuple[str, ...] = (*SBS_OUTCOMES, FALLBACK_ACTION)
    loss_rule: str = (
        "0 for correct ideal-branch action, 1 for wrong branch, explicit fallback "
        "cost; fallback is admissible only when x or z is observed as leakage"
    )
    evidence_scope: str = (
        "assumption_driven_sbs_branch_selection_not_full_quantum_recovery"
    )


STATIC_SBS_DECODER_DESCRIPTOR = StaticSBSDecoderDescriptor()


def _probability_vector(
    values: ArrayLike,
    length: int,
    name: str,
    *,
    strictly_positive: bool,
) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain {length} finite probabilities")
    if np.any(array < 0.0) or (strictly_positive and np.any(array <= 0.0)):
        qualifier = "strictly positive" if strictly_positive else "non-negative"
        raise ValueError(f"{name} entries must be {qualifier}")
    total = float(np.sum(array))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError(f"{name} must have positive finite mass")
    return array / total


def _observation_index(observation: PairedSyndrome | SBSObservedCycle) -> int:
    if isinstance(observation, SBSObservedCycle):
        syndrome = observation.syndrome
    elif isinstance(observation, PairedSyndrome):
        syndrome = observation
    else:
        raise TypeError("observation must be PairedSyndrome or SBSObservedCycle")
    return OBSERVED_CLASSES.index(syndrome.x) * len(OBSERVED_CLASSES) + OBSERVED_CLASSES.index(
        syndrome.z
    )


def _constituent_matrix(
    model: SBSObservationResetModel,
    ideal_value: str,
    observed_value: str | None,
) -> NDArray[np.float64]:
    """Return carry-before -> carry-after probability for one constituent."""

    ideal_index = ("g", "e").index(ideal_value)
    observed_index = (
        None if observed_value is None else OBSERVED_CLASSES.index(observed_value)
    )
    matrix = np.zeros((len(HIDDEN_ANCILLA_STATES), len(HIDDEN_ANCILLA_STATES)))
    for carry in range(len(HIDDEN_ANCILLA_STATES)):
        for hidden_pre in range(len(HIDDEN_ANCILLA_STATES)):
            preparation_probability = model.preparation_kernel[
                carry, ideal_index, hidden_pre
            ]
            if preparation_probability == 0.0:
                continue
            for observed in range(len(OBSERVED_CLASSES)):
                if observed_index is not None and observed != observed_index:
                    continue
                prefix = (
                    preparation_probability
                    * model.readout_confusion[hidden_pre, observed]
                )
                if prefix == 0.0:
                    continue
                matrix[carry] += prefix * model.reset_kernel[observed, hidden_pre]
    if not np.all(np.isfinite(matrix)) or np.any(matrix < -1.0e-15):
        raise RuntimeError("constituent marginalization produced invalid probabilities")
    return np.maximum(matrix, 0.0)


def _stationary_carry_distribution(
    transition: NDArray[np.float64],
    *,
    tolerance: float = 1.0e-14,
    max_iterations: int = 100_000,
) -> NDArray[np.float64]:
    """Find and verify a unique stationary carry distribution.

    Convergence is checked from every pure carry state.  This deliberately
    rejects reducible kernels whose stationary mixture would depend on an
    unrecorded initial hidden state.
    """

    if transition.shape != (4, 4) or not np.all(np.isfinite(transition)):
        raise ValueError("carry transition must be a finite 4x4 matrix")
    if np.any(transition < -1.0e-15) or not np.allclose(
        np.sum(transition, axis=1), 1.0, atol=1.0e-12, rtol=0.0
    ):
        raise ValueError("carry transition must be row stochastic")
    endpoints: list[NDArray[np.float64]] = []
    for initial_index in range(4):
        distribution = np.zeros(4, dtype=np.float64)
        distribution[initial_index] = 1.0
        for _ in range(max_iterations):
            updated = distribution @ transition
            if float(np.max(np.abs(updated - distribution))) <= tolerance:
                distribution = updated
                break
            distribution = updated
        else:
            raise ValueError("carry transition did not converge to a stationary distribution")
        endpoints.append(distribution)
    stacked = np.stack(endpoints)
    if float(np.max(np.ptp(stacked, axis=0))) > 1.0e-10:
        raise ValueError(
            "carry transition has initial-state-dependent stationary distributions"
        )
    stationary = np.mean(stacked, axis=0)
    stationary = np.maximum(stationary, 0.0)
    stationary /= float(np.sum(stationary))
    if not np.allclose(stationary @ transition, stationary, atol=2.0e-13, rtol=0.0):
        raise RuntimeError("stationary carry distribution failed its fixed-point check")
    return stationary


def _model_hash(model: SBSObservationResetModel) -> str:
    payload = {
        "protocol_id": model.protocol_id,
        "model_scope": model.model_scope,
        "preparation_kernel": model.preparation_kernel.tolist(),
        "readout_confusion": model.readout_confusion.tolist(),
        "reset_kernel": model.reset_kernel.tolist(),
        "reset_action_by_observation": dict(model.reset_action_by_observation),
        "counter_max": model.counter_max,
        "preparation_provenance": model.preparation_provenance,
        "readout_provenance": model.readout_provenance,
        "reset_provenance": model.reset_provenance,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class StaticSBSDecoderParameters:
    ideal_branch_prior: tuple[float, float, float, float]
    stationary_carry_prior: tuple[float, float, float, float]
    observation_likelihood: tuple[tuple[float, ...], ...]
    observation_probability: tuple[float, ...]
    posterior_by_observation: tuple[tuple[float, float, float, float], ...]
    fallback_cost: float
    model_sha256: str
    parameter_sha256: str
    calibration_protocol_id: str
    evidence_scope: str = "exact_stationary_hidden_carry_marginalization"

    def __post_init__(self) -> None:
        prior = _probability_vector(
            self.ideal_branch_prior,
            4,
            "ideal_branch_prior",
            strictly_positive=True,
        )
        carry = _probability_vector(
            self.stationary_carry_prior,
            4,
            "stationary_carry_prior",
            strictly_positive=True,
        )
        likelihood = np.asarray(self.observation_likelihood, dtype=np.float64)
        if likelihood.shape != (4, 9) or not np.all(np.isfinite(likelihood)):
            raise ValueError("observation_likelihood must be a finite 4x9 table")
        if np.any(likelihood < 0.0) or not np.allclose(
            np.sum(likelihood, axis=1), 1.0, atol=1.0e-12, rtol=0.0
        ):
            raise ValueError("each observation likelihood row must be normalized")
        evidence = _probability_vector(
            self.observation_probability,
            9,
            "observation_probability",
            strictly_positive=True,
        )
        posterior = np.asarray(self.posterior_by_observation, dtype=np.float64)
        if posterior.shape != (9, 4) or not np.all(np.isfinite(posterior)):
            raise ValueError("posterior_by_observation must be a finite 9x4 table")
        if np.any(posterior < 0.0) or not np.allclose(
            np.sum(posterior, axis=1), 1.0, atol=1.0e-12, rtol=0.0
        ):
            raise ValueError("each posterior row must be normalized")
        joint = prior[:, np.newaxis] * likelihood
        if not np.allclose(np.sum(joint, axis=0), evidence, atol=1.0e-12, rtol=0.0):
            raise ValueError("observation_probability is inconsistent with prior/likelihood")
        if not np.allclose(
            (joint / evidence[np.newaxis, :]).T,
            posterior,
            atol=1.0e-12,
            rtol=0.0,
        ):
            raise ValueError("posterior is inconsistent with Bayes' rule")
        cost = float(self.fallback_cost)
        if not math.isfinite(cost) or not 0.0 < cost < 1.0:
            raise ValueError("fallback_cost must lie strictly between 0 and 1")
        for name in ("model_sha256", "parameter_sha256"):
            value = getattr(self, name)
            if len(value) != 64:
                raise ValueError(f"{name} must be a SHA-256 hex digest")
            try:
                bytes.fromhex(value)
            except ValueError as exc:
                raise ValueError(f"{name} must be a SHA-256 hex digest") from exc
        if not self.calibration_protocol_id.strip():
            raise ValueError("calibration_protocol_id must not be empty")
        object.__setattr__(self, "ideal_branch_prior", tuple(float(x) for x in prior))
        object.__setattr__(
            self, "stationary_carry_prior", tuple(float(x) for x in carry)
        )
        object.__setattr__(self, "fallback_cost", cost)

    def likelihood_array(self) -> NDArray[np.float64]:
        return np.asarray(self.observation_likelihood, dtype=np.float64)

    def posterior_array(self) -> NDArray[np.float64]:
        return np.asarray(self.posterior_by_observation, dtype=np.float64)


def build_static_sbs_parameters(
    model: SBSObservationResetModel,
    *,
    ideal_branch_prior: ArrayLike,
    fallback_cost: float,
    calibration_protocol_id: str,
) -> StaticSBSDecoderParameters:
    """Exactly marginalize current observation and persistent reset carry."""

    if not isinstance(model, SBSObservationResetModel):
        raise TypeError("model must be an SBSObservationResetModel")
    if model.exposes_hidden_truth_to_deployable_view:
        raise ValueError("model must not expose hidden truth to its deployable view")
    prior = _probability_vector(
        ideal_branch_prior,
        4,
        "ideal_branch_prior",
        strictly_positive=True,
    )
    cost = float(fallback_cost)
    if not math.isfinite(cost) or not 0.0 < cost < 1.0:
        raise ValueError("fallback_cost must lie strictly between 0 and 1")
    protocol_id = str(calibration_protocol_id).strip()
    if not protocol_id:
        raise ValueError("calibration_protocol_id must not be empty")

    unconditional = {
        ideal: _constituent_matrix(model, ideal, None) for ideal in ("g", "e")
    }
    conditioned = {
        (ideal, observed): _constituent_matrix(model, ideal, observed)
        for ideal in ("g", "e")
        for observed in OBSERVED_CLASSES
    }
    for matrix in unconditional.values():
        if not np.allclose(np.sum(matrix, axis=1), 1.0, atol=1.0e-12, rtol=0.0):
            raise RuntimeError("unconditioned constituent transition is not stochastic")

    carry_transition = np.zeros((4, 4), dtype=np.float64)
    for label, weight in zip(SBS_OUTCOMES, prior):
        ideal = ideal_syndrome_from_kraus(label)
        carry_transition += float(weight) * (
            unconditional[ideal.x] @ unconditional[ideal.z]
        )
    stationary = _stationary_carry_distribution(carry_transition)

    likelihood = np.zeros((4, len(_OBSERVATION_PAIRS)), dtype=np.float64)
    terminal = np.ones(4, dtype=np.float64)
    for label_index, label in enumerate(SBS_OUTCOMES):
        ideal = ideal_syndrome_from_kraus(label)
        for observation_index, (observed_x, observed_z) in enumerate(
            _OBSERVATION_PAIRS
        ):
            likelihood[label_index, observation_index] = float(
                stationary
                @ conditioned[(ideal.x, observed_x)]
                @ conditioned[(ideal.z, observed_z)]
                @ terminal
            )
    if not np.allclose(np.sum(likelihood, axis=1), 1.0, atol=2.0e-13, rtol=0.0):
        raise RuntimeError("exact sBs observation likelihood is not normalized")
    joint = prior[:, np.newaxis] * likelihood
    evidence = np.sum(joint, axis=0)
    if np.any(evidence <= 0.0):
        raise ValueError(
            "calibration model must assign positive probability to all deployable observations"
        )
    posterior = (joint / evidence[np.newaxis, :]).T
    model_sha = _model_hash(model)
    hash_payload = {
        "model_sha256": model_sha,
        "ideal_branch_prior": prior.tolist(),
        "stationary_carry_prior": stationary.tolist(),
        "observation_likelihood": likelihood.tolist(),
        "fallback_cost": cost,
        "calibration_protocol_id": protocol_id,
    }
    parameter_sha = hashlib.sha256(
        json.dumps(hash_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return StaticSBSDecoderParameters(
        ideal_branch_prior=tuple(float(value) for value in prior),  # type: ignore[arg-type]
        stationary_carry_prior=tuple(float(value) for value in stationary),  # type: ignore[arg-type]
        observation_likelihood=tuple(
            tuple(float(value) for value in row) for row in likelihood
        ),
        observation_probability=tuple(float(value) for value in evidence),
        posterior_by_observation=tuple(
            tuple(float(value) for value in row) for row in posterior
        ),
        fallback_cost=cost,
        model_sha256=model_sha,
        parameter_sha256=parameter_sha,
        calibration_protocol_id=protocol_id,
    )


@dataclass(frozen=True)
class StaticSBSDecision:
    observed_syndrome: tuple[str, str]
    posterior: tuple[float, float, float, float]
    map_label: str
    action: str
    expected_loss: float
    fallback: bool
    fallback_admissible: bool
    evidence_scope: str = "observed_only_static_sbs_branch_decision"


class StaticSBSBranchDecoder:
    """Frozen observed-only lookup decoder built from exact model marginalization."""

    descriptor = STATIC_SBS_DECODER_DESCRIPTOR

    def __init__(self, parameters: StaticSBSDecoderParameters) -> None:
        if not isinstance(parameters, StaticSBSDecoderParameters):
            raise TypeError("parameters must be StaticSBSDecoderParameters")
        self.parameters = parameters

    def decode(
        self, observation: PairedSyndrome | SBSObservedCycle
    ) -> StaticSBSDecision:
        index = _observation_index(observation)
        syndrome = observation.syndrome if isinstance(observation, SBSObservedCycle) else observation
        posterior = self.parameters.posterior_array()[index]
        map_index = int(np.argmax(posterior))
        map_label = SBS_OUTCOMES[map_index]
        map_risk = 1.0 - float(posterior[map_index])
        leakage_observed = "leakage" in syndrome.as_tuple()
        use_fallback = leakage_observed and self.parameters.fallback_cost <= map_risk
        action = FALLBACK_ACTION if use_fallback else map_label
        return StaticSBSDecision(
            observed_syndrome=syndrome.as_tuple(),
            posterior=tuple(float(value) for value in posterior),  # type: ignore[arg-type]
            map_label=map_label,
            action=action,
            expected_loss=(self.parameters.fallback_cost if use_fallback else map_risk),
            fallback=use_fallback,
            fallback_admissible=leakage_observed,
        )


def standard_observed_sbs_action(
    observation: PairedSyndrome | SBSObservedCycle,
) -> str:
    """Direct-classification anchor; observed leakage always emits fallback."""

    _observation_index(observation)  # type and alphabet validation
    syndrome = observation.syndrome if isinstance(observation, SBSObservedCycle) else observation
    if "leakage" in syndrome.as_tuple():
        return FALLBACK_ACTION
    return f"K_{syndrome.z}{syndrome.x}"


def score_sbs_branch_action(
    action: str,
    ideal_branch_truth: str,
    *,
    fallback_cost: float,
) -> float:
    """Evaluator-only 0/1/fallback branch-selection loss."""

    if ideal_branch_truth not in SBS_OUTCOMES:
        raise ValueError(f"ideal_branch_truth must be one of {SBS_OUTCOMES}")
    cost = float(fallback_cost)
    if not math.isfinite(cost) or not 0.0 < cost < 1.0:
        raise ValueError("fallback_cost must lie strictly between 0 and 1")
    if action == FALLBACK_ACTION:
        return cost
    if action not in SBS_OUTCOMES:
        raise ValueError(f"action must be one of {(*SBS_OUTCOMES, FALLBACK_ACTION)}")
    return float(action != ideal_branch_truth)


def validate_static_protocol_comparison_registration() -> tuple[str, ...]:
    """Bind the branch-target comparison without aliasing logical binning."""

    from cnn_fpga.benchmark.standard_binning_baseline import (
        major_comparison_registry,
        validate_major_comparison_registry,
    )

    registry_gates = validate_major_comparison_registry()
    matches = [
        entry
        for entry in major_comparison_registry()
        if entry.comparison_id == "t3_1_4_static_sbs_branch_comparison"
    ]
    if len(matches) != 1:
        raise ValueError("T3.1.4 comparison must be registered exactly once")
    entry = matches[0]
    expected = (
        STANDARD_SBS_BRANCH_ID,
        STATIC_SBS_MAP_ID,
        STATIC_SBS_DECODER_ID,
        IDEAL_BRANCH_REFERENCE_ID,
    )
    if entry.method_ids != expected:
        raise ValueError("T3.1.4 registered method schema does not match the implementation")
    if entry.standard_binning_policy != "not_applicable":
        raise ValueError("sBs branch target must not be aliased to logical standard binning")
    return (*registry_gates, "registry:t3_1_4_target_separation")


@dataclass(frozen=True)
class StaticProtocolScenario:
    scenario_id: str
    readout_confusion: tuple[tuple[float, float, float], ...]
    injection_probabilities: tuple[float, float, float, float]
    reset_success: tuple[float, float, float]
    ideal_branch_prior: tuple[float, float, float, float]
    fallback_cost: float
    parameter_scope: str = "synthetic_assumption_not_device_calibrated"

    def make_model(self) -> SBSObservationResetModel:
        f_g, f_e, higher_g, higher_e = self.injection_probabilities
        e_reset, f_reset, higher_reset = self.reset_success
        provenance = f"T3.1.4 {self.scenario_id} {self.parameter_scope}"
        return make_persistent_leakage_model(
            readout_confusion=np.asarray(self.readout_confusion, dtype=np.float64),
            f_injection_given_g=f_g,
            f_injection_given_e=f_e,
            higher_injection_given_g=higher_g,
            higher_injection_given_e=higher_e,
            e_reset_success=e_reset,
            f_reset_success=f_reset,
            higher_reset_success=higher_reset,
            counter_max=63,
            readout_provenance=provenance,
            parameter_provenance=provenance,
        )


def static_protocol_validation_scenarios() -> tuple[StaticProtocolScenario, ...]:
    return (
        StaticProtocolScenario(
            scenario_id="balanced_low_fault_control",
            readout_confusion=(
                (0.94, 0.04, 0.02),
                (0.03, 0.94, 0.03),
                (0.04, 0.04, 0.92),
                (0.05, 0.05, 0.90),
            ),
            injection_probabilities=(0.025, 0.040, 0.010, 0.015),
            reset_success=(0.90, 0.72, 0.50),
            ideal_branch_prior=(0.25, 0.25, 0.25, 0.25),
            fallback_cost=0.25,
        ),
        StaticProtocolScenario(
            scenario_id="biased_branch_prior",
            readout_confusion=(
                (0.82, 0.15, 0.03),
                (0.20, 0.74, 0.06),
                (0.16, 0.14, 0.70),
                (0.20, 0.15, 0.65),
            ),
            injection_probabilities=(0.035, 0.070, 0.020, 0.035),
            reset_success=(0.78, 0.55, 0.30),
            ideal_branch_prior=(0.68, 0.12, 0.12, 0.08),
            fallback_cost=0.30,
        ),
        StaticProtocolScenario(
            scenario_id="asymmetric_readout",
            readout_confusion=(
                (0.78, 0.18, 0.04),
                (0.28, 0.65, 0.07),
                (0.22, 0.13, 0.65),
                (0.25, 0.15, 0.60),
            ),
            injection_probabilities=(0.040, 0.090, 0.020, 0.040),
            reset_success=(0.72, 0.50, 0.25),
            ideal_branch_prior=(0.12, 0.18, 0.25, 0.45),
            fallback_cost=0.32,
        ),
        StaticProtocolScenario(
            scenario_id="persistent_leakage_stress",
            readout_confusion=(
                (0.86, 0.08, 0.06),
                (0.12, 0.76, 0.12),
                (0.20, 0.12, 0.68),
                (0.24, 0.14, 0.62),
            ),
            injection_probabilities=(0.080, 0.120, 0.050, 0.080),
            reset_success=(0.65, 0.35, 0.15),
            ideal_branch_prior=(0.40, 0.10, 0.10, 0.40),
            fallback_cost=0.35,
        ),
    )


@dataclass(frozen=True)
class StaticProtocolValidationConfig:
    evaluation_seeds: tuple[int, ...] = (
        20260741,
        20260742,
        20260743,
        20260744,
        20260745,
        20260746,
        20260747,
        20260748,
    )
    cycles_per_seed: int = 20_000
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        seeds = tuple(self.evaluation_seeds)
        if len(seeds) < 4 or len(set(seeds)) != len(seeds):
            raise ValueError("evaluation_seeds must contain at least four unique seeds")
        if any(isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) for seed in seeds):
            raise TypeError("evaluation seeds must be integers")
        if any(int(seed) < 0 for seed in seeds):
            raise ValueError("evaluation seeds must be non-negative")
        if isinstance(self.cycles_per_seed, bool) or not isinstance(
            self.cycles_per_seed, (int, np.integer)
        ):
            raise TypeError("cycles_per_seed must be an integer")
        if int(self.cycles_per_seed) < 2_000:
            raise ValueError("cycles_per_seed must be at least 2000")
        confidence = float(self.confidence_level)
        if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
            raise ValueError("confidence_level must lie strictly between 0 and 1")
        object.__setattr__(self, "evaluation_seeds", tuple(int(seed) for seed in seeds))
        object.__setattr__(self, "cycles_per_seed", int(self.cycles_per_seed))
        object.__setattr__(self, "confidence_level", confidence)


def _paired_seed_interval(
    values: Sequence[float], confidence_level: float
) -> dict[str, float]:
    samples = np.asarray(values, dtype=np.float64)
    if samples.ndim != 1 or samples.size < 4 or not np.all(np.isfinite(samples)):
        raise ValueError("paired seed values must contain at least four finite entries")
    estimate = float(np.mean(samples))
    standard_error = float(np.std(samples, ddof=1) / math.sqrt(samples.size))
    z_value = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    return {
        "estimate": estimate,
        "standard_error": standard_error,
        "ci_low": estimate - z_value * standard_error,
        "ci_high": estimate + z_value * standard_error,
        "cluster_unit": "evaluation_seed",
    }


def _policy_expected_costs(
    parameters: StaticSBSDecoderParameters,
) -> dict[str, float]:
    decoder = StaticSBSBranchDecoder(parameters)
    posterior = parameters.posterior_array()
    evidence = np.asarray(parameters.observation_probability)
    standard = 0.0
    forced_map = 0.0
    optimized = 0.0
    fallback_probability = 0.0
    for index, pair in enumerate(_OBSERVATION_PAIRS):
        observation = PairedSyndrome(x=pair[0], z=pair[1])
        standard_action = standard_observed_sbs_action(observation)
        decision = decoder.decode(observation)
        map_index = SBS_OUTCOMES.index(decision.map_label)
        forced_map_risk = 1.0 - posterior[index, map_index]
        if standard_action == FALLBACK_ACTION:
            standard_risk = parameters.fallback_cost
        else:
            standard_risk = 1.0 - posterior[index, SBS_OUTCOMES.index(standard_action)]
        standard += evidence[index] * standard_risk
        forced_map += evidence[index] * forced_map_risk
        optimized += evidence[index] * decision.expected_loss
        fallback_probability += evidence[index] * float(decision.fallback)
    return {
        "direct_observed_expected_cost": float(standard),
        "forced_static_map_expected_cost": float(forced_map),
        "protocol_aware_expected_cost": float(optimized),
        "protocol_aware_fallback_probability": float(fallback_probability),
    }


def _source_sha256(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_static_protocol_validation(
    config: StaticProtocolValidationConfig | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Validate exact tables against independent stationary Markov trajectories."""

    settings = StaticProtocolValidationConfig() if config is None else config
    if not isinstance(settings, StaticProtocolValidationConfig):
        raise TypeError("config must be StaticProtocolValidationConfig")
    registration_gates = validate_static_protocol_comparison_registration()
    scenarios = static_protocol_validation_scenarios()
    rows: list[dict[str, object]] = []
    scenario_payloads: list[dict[str, object]] = []
    aggregate_seed_gains: dict[int, list[float]] = {
        seed: [] for seed in settings.evaluation_seeds
    }
    positive_scenarios = 0
    nonleak_override_scenarios = 0
    maximum_likelihood_error = 0.0
    maximum_cost_error = 0.0
    any_nonleak_fallback = False

    for scenario_index, scenario in enumerate(scenarios):
        model = scenario.make_model()
        parameters = build_static_sbs_parameters(
            model,
            ideal_branch_prior=scenario.ideal_branch_prior,
            fallback_cost=scenario.fallback_cost,
            calibration_protocol_id=f"T3.1.4/{scenario.scenario_id}/stationary-v1",
        )
        decoder = StaticSBSBranchDecoder(parameters)
        likelihood_counts = np.zeros((4, 9), dtype=np.int64)
        label_counts = np.zeros(4, dtype=np.int64)
        carry_counts = np.zeros(4, dtype=np.int64)
        scenario_seed_gains: list[float] = []
        nonleak_override_count = 0
        total_cycles = 0
        summed_standard_cost = 0.0
        summed_map_cost = 0.0
        summed_optimized_cost = 0.0
        summed_oracle_cost = 0.0
        total_fallbacks = 0
        total_observed_leakage = 0
        total_nonleak = 0
        standard_nonleak_errors = 0
        optimized_nonleak_errors = 0

        for seed in settings.evaluation_seeds:
            seed_sequence = np.random.SeedSequence([seed, scenario_index, 314])
            label_sequence, carry_sequence, simulator_sequence = seed_sequence.spawn(3)
            label_rng = np.random.default_rng(label_sequence)
            labels = tuple(
                SBS_OUTCOMES[index]
                for index in label_rng.choice(
                    4,
                    size=settings.cycles_per_seed,
                    p=np.asarray(parameters.ideal_branch_prior),
                )
            )
            carry_rng = np.random.default_rng(carry_sequence)
            initial_carry_index = int(
                carry_rng.choice(4, p=np.asarray(parameters.stationary_carry_prior))
            )
            simulator_seed = int(
                simulator_sequence.generate_state(1, dtype=np.uint64)[0]
            )
            trajectory = model.simulate(
                labels,
                seed=simulator_seed,
                initial_hidden_memory=HiddenAncillaMemory(
                    carry_state=HIDDEN_ANCILLA_STATES[initial_carry_index]
                ),
            )
            seed_standard_cost = 0.0
            seed_map_cost = 0.0
            seed_optimized_cost = 0.0
            seed_fallbacks = 0
            seed_nonleak = 0
            seed_standard_nonleak_errors = 0
            seed_optimized_nonleak_errors = 0
            for label, observed, truth in zip(
                labels, trajectory.observed_cycles, trajectory.truth_cycles
            ):
                observation_index = _observation_index(observed)
                label_index = SBS_OUTCOMES.index(label)
                label_counts[label_index] += 1
                likelihood_counts[label_index, observation_index] += 1
                carry_counts[HIDDEN_ANCILLA_STATES.index(truth.hidden_post_reset[1])] += 1
                standard_action = standard_observed_sbs_action(observed)
                decision = decoder.decode(observed)
                map_action = decision.map_label
                leakage_observed = "leakage" in observed.syndrome.as_tuple()
                standard_loss = score_sbs_branch_action(
                    standard_action, label, fallback_cost=scenario.fallback_cost
                )
                map_loss = score_sbs_branch_action(
                    map_action, label, fallback_cost=scenario.fallback_cost
                )
                optimized_loss = score_sbs_branch_action(
                    decision.action, label, fallback_cost=scenario.fallback_cost
                )
                seed_standard_cost += standard_loss
                seed_map_cost += map_loss
                seed_optimized_cost += optimized_loss
                seed_fallbacks += decision.fallback
                total_observed_leakage += leakage_observed
                if decision.fallback and not leakage_observed:
                    any_nonleak_fallback = True
                if not leakage_observed:
                    seed_nonleak += 1
                    seed_standard_nonleak_errors += standard_action != label
                    seed_optimized_nonleak_errors += decision.action != label
                    nonleak_override_count += decision.action != standard_action

            count = settings.cycles_per_seed
            standard_rate = seed_standard_cost / count
            map_rate = seed_map_cost / count
            optimized_rate = seed_optimized_cost / count
            gain = standard_rate - optimized_rate
            scenario_seed_gains.append(gain)
            aggregate_seed_gains[seed].append(gain)
            summed_standard_cost += seed_standard_cost
            summed_map_cost += seed_map_cost
            summed_optimized_cost += seed_optimized_cost
            total_fallbacks += seed_fallbacks
            total_nonleak += seed_nonleak
            standard_nonleak_errors += seed_standard_nonleak_errors
            optimized_nonleak_errors += seed_optimized_nonleak_errors
            total_cycles += count
            rows.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "evaluation_seed": seed,
                    "cycles": count,
                    "parameter_sha256": parameters.parameter_sha256,
                    "direct_observed_cost": standard_rate,
                    "forced_static_map_cost": map_rate,
                    "protocol_aware_cost": optimized_rate,
                    "ideal_branch_reference_cost": 0.0,
                    "direct_minus_protocol_aware": gain,
                    "protocol_aware_fallback_rate": seed_fallbacks / count,
                    "nonleak_cycles": seed_nonleak,
                    "direct_nonleak_branch_error_rate": (
                        seed_standard_nonleak_errors / seed_nonleak
                    ),
                    "protocol_aware_nonleak_branch_error_rate": (
                        seed_optimized_nonleak_errors / seed_nonleak
                    ),
                }
            )

        empirical_likelihood = likelihood_counts / label_counts[:, np.newaxis]
        likelihood_error = float(
            np.max(np.abs(empirical_likelihood - parameters.likelihood_array()))
        )
        maximum_likelihood_error = max(maximum_likelihood_error, likelihood_error)
        exact_costs = _policy_expected_costs(parameters)
        empirical_costs = {
            "direct_observed_cost": summed_standard_cost / total_cycles,
            "forced_static_map_cost": summed_map_cost / total_cycles,
            "protocol_aware_cost": summed_optimized_cost / total_cycles,
            "ideal_branch_reference_cost": summed_oracle_cost / total_cycles,
        }
        cost_error = max(
            abs(
                empirical_costs["direct_observed_cost"]
                - exact_costs["direct_observed_expected_cost"]
            ),
            abs(
                empirical_costs["forced_static_map_cost"]
                - exact_costs["forced_static_map_expected_cost"]
            ),
            abs(
                empirical_costs["protocol_aware_cost"]
                - exact_costs["protocol_aware_expected_cost"]
            ),
        )
        maximum_cost_error = max(maximum_cost_error, cost_error)
        interval = _paired_seed_interval(
            scenario_seed_gains, settings.confidence_level
        )
        positive_scenarios += interval["ci_low"] > 0.0
        nonleak_override_scenarios += nonleak_override_count > 0
        scenario_payloads.append(
            {
                "scenario": asdict(scenario),
                "parameters": asdict(parameters),
                "observation_pairs_x_then_z": [list(pair) for pair in _OBSERVATION_PAIRS],
                "exact_policy_costs": exact_costs,
                "empirical_costs": empirical_costs,
                "direct_minus_protocol_aware_seed_cluster_ci": interval,
                "empirical_likelihood_max_abs_error": likelihood_error,
                "empirical_stationary_carry_distribution": (
                    carry_counts / float(np.sum(carry_counts))
                ).tolist(),
                "protocol_aware_fallback_rate": total_fallbacks / total_cycles,
                "observed_leakage_rate": total_observed_leakage / total_cycles,
                "direct_nonleak_branch_error_rate": (
                    standard_nonleak_errors / total_nonleak
                ),
                "protocol_aware_nonleak_branch_error_rate": (
                    optimized_nonleak_errors / total_nonleak
                ),
                "nonleak_policy_override_count": nonleak_override_count,
            }
        )

    # Scenarios are fixed benchmark conditions, not independent statistical
    # clusters.  Average their paired gain within each seed first; otherwise
    # treating 4 x 8 scenario-seed rows as 32 clusters would pseudoreplicate
    # the eight independent RNG streams.
    aggregate_seed_means = [
        float(np.mean(aggregate_seed_gains[seed]))
        for seed in settings.evaluation_seeds
    ]
    aggregate_interval = _paired_seed_interval(
        aggregate_seed_means, settings.confidence_level
    )
    gates = {
        "descriptor_uses_no_online_hidden_truth": not STATIC_SBS_DECODER_DESCRIPTOR.hidden_truth_online_inputs,
        "decoder_is_static_during_evaluation": not STATIC_SBS_DECODER_DESCRIPTOR.update_during_evaluation,
        "all_exact_likelihoods_match_independent_markov_mc": maximum_likelihood_error
        <= 0.015,
        "all_exact_policy_costs_match_independent_markov_mc": maximum_cost_error
        <= 0.010,
        "fallback_never_used_without_observed_leakage": not any_nonleak_fallback,
        "nontrivial_nonleak_policy_in_at_least_three_scenarios": nonleak_override_scenarios
        >= 3,
        "three_noncontrol_scenarios_have_resolved_cost_gain": positive_scenarios >= 3,
        "aggregate_protocol_policy_gain_resolved": aggregate_interval["ci_low"] > 0.0,
        "ideal_branch_reference_remains_strict_lower_reference": all(
            payload["empirical_costs"]["protocol_aware_cost"] > 0.0
            for payload in scenario_payloads
        ),
        "scenario_parameter_hashes_are_unique": len(
            {payload["parameters"]["parameter_sha256"] for payload in scenario_payloads}
        )
        == len(scenario_payloads),
        "comparison_is_registered_as_separate_branch_target": bool(registration_gates),
    }
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise AssertionError(f"static protocol validation gates failed: {failed}")
    source_paths = (
        Path(__file__),
        Path(__file__).parents[2] / "physics" / "sbs_observation_reset.py",
        Path(__file__).parents[2] / "physics" / "sbs_error_space.py",
        Path(__file__).with_name("standard_binning_baseline.py"),
    )
    payload: dict[str, object] = {
        "schema_version": "t3.1.4-static-sbs-protocol-bayes-v1",
        "task_id": "T3.1.4",
        "status": "PASS",
        "implementation_sha256": _source_sha256(source_paths),
        "descriptor": asdict(STATIC_SBS_DECODER_DESCRIPTOR),
        "config": asdict(settings),
        "comparison_registry_gates": list(registration_gates),
        "comparison_contract": {
            "standard": STANDARD_SBS_BRANCH_ID,
            "static_map": STATIC_SBS_MAP_ID,
            "protocol_aware": STATIC_SBS_DECODER_ID,
            "nondeployable_truth_reference": IDEAL_BRANCH_REFERENCE_ID,
            "target": "ideal_sbs_kraus_branch",
            "not_target": "logical_Pauli_class_or_full_quantum_recovery",
        },
        "scenarios": scenario_payloads,
        "aggregate": {
            "scenarios": len(scenarios),
            "evaluation_seeds_per_scenario": len(settings.evaluation_seeds),
            "cycles": len(scenarios)
            * len(settings.evaluation_seeds)
            * settings.cycles_per_seed,
            "direct_minus_protocol_aware_seed_cluster_ci": aggregate_interval,
            "maximum_empirical_likelihood_abs_error": maximum_likelihood_error,
            "maximum_empirical_vs_exact_cost_abs_error": maximum_cost_error,
            "resolved_positive_scenarios": positive_scenarios,
            "nonleak_override_scenarios": nonleak_override_scenarios,
        },
        "gate_summary": {
            "passed": len(gates),
            "failed": 0,
            "gates": gates,
        },
        "claim_boundary": {
            "allowed": (
                "a frozen observed-only sBs branch decoder exactly marginalizes the "
                "assumed preparation/readout/reset model and lowers explicit branch-action "
                "cost versus direct classified outcomes in the registered synthetic conditions"
            ),
            "forbidden": (
                "full finite-energy recovery optimum, logical-error-rate improvement, free "
                "post-selection, device calibration, experimental leakage detection, or an "
                "alias of the T3.1.3 full-state decoder oracle"
            ),
        },
    }
    return json.loads(json.dumps(payload, ensure_ascii=False)), rows


def write_static_protocol_validation(
    json_path: str | Path,
    csv_path: str | Path,
    config: StaticProtocolValidationConfig | None = None,
) -> dict[str, object]:
    payload, rows = build_static_protocol_validation(config)
    output_json = Path(json_path)
    output_csv = Path(csv_path)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate the T3.1.4 static sBs protocol-aware decoder"
    )
    parser.add_argument(
        "--json", default="docs/t3_1_4_static_protocol_decoder_validation.json"
    )
    parser.add_argument(
        "--csv", default="docs/t3_1_4_static_protocol_decoder_source_data.csv"
    )
    arguments = parser.parse_args()
    result = write_static_protocol_validation(arguments.json, arguments.csv)
    print(json.dumps(result["gate_summary"], ensure_ascii=False))


__all__ = [
    "STATIC_SBS_DECODER_ID",
    "STANDARD_SBS_BRANCH_ID",
    "STATIC_SBS_MAP_ID",
    "IDEAL_BRANCH_REFERENCE_ID",
    "FALLBACK_ACTION",
    "StaticSBSDecoderDescriptor",
    "STATIC_SBS_DECODER_DESCRIPTOR",
    "StaticSBSDecoderParameters",
    "StaticSBSDecision",
    "StaticSBSBranchDecoder",
    "StaticProtocolScenario",
    "StaticProtocolValidationConfig",
    "build_static_sbs_parameters",
    "standard_observed_sbs_action",
    "score_sbs_branch_action",
    "validate_static_protocol_comparison_registration",
    "static_protocol_validation_scenarios",
    "build_static_protocol_validation",
    "write_static_protocol_validation",
]
