"""T3.1.5 single-mode top-K lattice-coset truncated MAP baseline.

For every two-dimensional logical coset ``(b_q, b_p)``, the decoder ranks
Gaussian lattice-pair aliases by their *joint* correlated likelihood, retains
the K largest terms in that coset, and performs a stable log-sum-exp.  The
candidate rectangle is identical to :func:`physics.ideal_gkp_decoder.map_decode_2d`;
therefore K at or above the largest coset population reproduces that full
periodic-Gaussian reference exactly (up to floating-point reduction order).

This is a single-mode lattice-sum approximation.  It contains no matching
graph, outer surface code, or implementation of the K-MWM algorithm that
motivated the generic top-K idea.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import hashlib
import json
from math import ceil, isfinite, log, pi, sqrt
from pathlib import Path
from statistics import NormalDist
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from physics.constants import LATTICE_CONST
from physics.ideal_gkp_decoder import (
    covariance_from_sigmas,
    map_decode_2d,
    standard_binning_1d,
)


TOPK_LATTICE_COSET_MAP_ID = "topk_lattice_coset_map"
FULL_PERIODIC_GAUSSIAN_MAP_ID = "full_periodic_gaussian_map"
_MAX_ALIAS_EVALUATIONS = 10_000_000


@dataclass(frozen=True)
class TopKDecoderDescriptor:
    baseline_id: str = TOPK_LATTICE_COSET_MAP_ID
    label: str = "Top-K single-mode lattice-coset truncated MAP"
    task_owner: str = "T3.1.5"
    comparison_role: str = "periodic_map_implementation_approximation_baseline"
    deployable: bool = True
    observation_inputs: tuple[str, ...] = (
        "centered_modular_syndrome_q",
        "centered_modular_syndrome_p",
    )
    offline_parameters: tuple[str, ...] = (
        "frozen_mean",
        "frozen_covariance",
        "joint_coset_prior",
        "K",
        "tail_sigma",
    )
    hidden_truth_inputs: tuple[str, ...] = ()
    approximation_rule: str = (
        "rank joint 2D Gaussian lattice-pair aliases separately inside each of "
        "four logical cosets and log-sum-exp the largest K terms"
    )
    excluded_algorithms: tuple[str, ...] = (
        "surface_code_matching",
        "K_minimum_weight_matchings",
        "outer_code_decoding",
    )
    evidence_scope: str = "single_mode_periodic_gaussian_lattice_sum_approximation"


TOPK_DECODER_DESCRIPTOR = TopKDecoderDescriptor()


def _validate_lattice(value: float) -> float:
    spacing = float(value)
    if not isfinite(spacing) or spacing <= 0.0:
        raise ValueError("lattice must be finite and positive")
    return spacing


def _validate_k(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError("K must be an integer")
    integer = int(value)
    if integer < 1:
        raise ValueError("K must be positive")
    return integer


def _validate_covariance(
    values: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float, float, float]:
    covariance = np.asarray(values, dtype=np.float64)
    if covariance.shape != (2, 2) or not np.all(np.isfinite(covariance)):
        raise ValueError("covariance must be a finite 2x2 matrix")
    scale = max(float(np.max(np.abs(covariance))), np.finfo(np.float64).tiny)
    if not np.allclose(covariance, covariance.T, rtol=1.0e-12, atol=1.0e-14 * scale):
        raise ValueError("covariance must be symmetric")
    try:
        cholesky = np.linalg.cholesky(covariance)
        inverse = np.linalg.solve(covariance, np.eye(2))
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance must be strictly positive definite") from exc
    if not np.all(np.isfinite(inverse)):
        raise ValueError("covariance is numerically singular")
    log_determinant = 2.0 * float(np.sum(np.log(np.diag(cholesky))))
    return (
        covariance,
        inverse,
        log_determinant,
        sqrt(float(covariance[0, 0])),
        sqrt(float(covariance[1, 1])),
    )


def _validate_prior(prior: ArrayLike | None) -> NDArray[np.float64]:
    if prior is None:
        return np.full((2, 2), 0.25, dtype=np.float64)
    probabilities = np.asarray(prior, dtype=np.float64)
    if probabilities.shape != (2, 2):
        raise ValueError("prior must have shape (2, 2)")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities <= 0.0):
        raise ValueError("prior entries must be finite and strictly positive")
    total = float(np.sum(probabilities))
    if not isfinite(total) or total <= 0.0:
        raise ValueError("prior must have positive finite mass")
    return probabilities / total


def _nearest_lattice_index(coordinates: NDArray[np.float64]) -> NDArray[np.int64]:
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("mean-syndrome offset is too large relative to lattice")
    safe_limit = float(2**62)
    if np.any(coordinates <= -safe_limit) or np.any(coordinates >= safe_limit):
        raise ValueError("mean-syndrome offset lies outside the supported int64 range")
    return np.floor(coordinates + 0.5).astype(np.int64)


def _logsumexp(values: NDArray[np.float64], axis: int | tuple[int, ...]) -> NDArray[np.float64]:
    maximum = np.max(values, axis=axis, keepdims=True)
    if np.any(~np.isfinite(maximum)):
        raise RuntimeError("log-sum-exp received a coset with no finite alias")
    shifted = np.exp(values - maximum)
    summed = np.sum(shifted, axis=axis, keepdims=True)
    return np.squeeze(maximum + np.log(summed), axis=axis)


@dataclass(frozen=True)
class _PreparedTopKCandidates:
    syndrome: NDArray[np.float64]
    original_shape: tuple[int, ...]
    cumulative_log_terms: tuple[NDArray[np.float64], ...]
    coset_candidate_counts: NDArray[np.int64]
    radius_q: int
    radius_p: int
    candidate_aliases: int


def _prepare_topk_candidates(
    syndrome: ArrayLike,
    covariance: ArrayLike,
    *,
    mean: ArrayLike,
    lattice: float,
    tail_sigma: float,
) -> _PreparedTopKCandidates:
    spacing = _validate_lattice(lattice)
    values = np.asarray(syndrome, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != 2:
        raise ValueError("syndrome must have shape (2,) or (..., 2)")
    if not np.all(np.isfinite(values)):
        raise ValueError("syndrome must contain only finite values")
    half = spacing / 2.0
    if np.any(values < -half) or np.any(values >= half):
        raise ValueError(
            "syndrome coordinates must lie in the half-open interval "
            "[-lattice/2, lattice/2)"
        )
    original_shape = values.shape[:-1]
    flat = values.reshape(-1, 2)
    mean_values = np.asarray(mean, dtype=np.float64)
    if not np.all(np.isfinite(mean_values)):
        raise ValueError("mean must contain only finite values")
    try:
        broadcast_mean = np.broadcast_to(mean_values, values.shape).reshape(-1, 2)
    except ValueError as exc:
        raise ValueError("mean must be broadcast-compatible with syndrome") from exc
    _, inverse, log_determinant, sigma_q, sigma_p = _validate_covariance(covariance)
    tail = float(tail_sigma)
    if not isfinite(tail) or tail <= 0.0:
        raise ValueError("tail_sigma must be finite and positive")
    radius_q = max(2, int(ceil(tail * sigma_q / spacing)) + 2)
    radius_p = max(2, int(ceil(tail * sigma_p / spacing)) + 2)
    candidate_aliases = (2 * radius_q + 1) * (2 * radius_p + 1)
    if candidate_aliases > _MAX_ALIAS_EVALUATIONS or (
        flat.shape[0] > 0
        and candidate_aliases > _MAX_ALIAS_EVALUATIONS // flat.shape[0]
    ):
        raise ValueError(
            "alias workload exceeds the safety limit; reduce noise scale or decode in chunks"
        )
    coordinates = (broadcast_mean - flat) / spacing
    nearest = _nearest_lattice_index(coordinates)
    offsets_q = np.arange(-radius_q, radius_q + 1, dtype=np.int64)
    offsets_p = np.arange(-radius_p, radius_p + 1, dtype=np.int64)
    aliases_q = nearest[:, 0, None, None] + offsets_q[None, :, None]
    aliases_p = nearest[:, 1, None, None] + offsets_p[None, None, :]
    residual_q = (
        flat[:, 0, None, None]
        + aliases_q.astype(np.float64) * spacing
        - broadcast_mean[:, 0, None, None]
    )
    residual_p = (
        flat[:, 1, None, None]
        + aliases_p.astype(np.float64) * spacing
        - broadcast_mean[:, 1, None, None]
    )
    quadratic = (
        inverse[0, 0] * residual_q * residual_q
        + 2.0 * inverse[0, 1] * residual_q * residual_p
        + inverse[1, 1] * residual_p * residual_p
    )
    log_weights = -0.5 * quadratic - log(2.0 * pi) - 0.5 * log_determinant
    cumulative_terms: list[NDArray[np.float64]] = []
    counts: list[NDArray[np.int64]] = []
    for parity_q in (0, 1):
        for parity_p in (0, 1):
            mask = (
                (np.mod(aliases_q, 2) == parity_q)
                & (np.mod(aliases_p, 2) == parity_p)
            )
            counts.append(np.sum(mask, axis=(-2, -1), dtype=np.int64))
            terms = np.where(mask, log_weights, -np.inf).reshape(flat.shape[0], -1)
            sorted_values = np.sort(terms, axis=-1)[:, ::-1]
            cumulative_terms.append(np.logaddexp.accumulate(sorted_values, axis=-1))
    count_array = np.stack(counts, axis=-1)
    if np.any(count_array <= 0) or np.any(np.sum(count_array, axis=-1) != candidate_aliases):
        raise RuntimeError("logical-coset candidate accounting is inconsistent")
    return _PreparedTopKCandidates(
        syndrome=flat,
        original_shape=original_shape,
        cumulative_log_terms=tuple(cumulative_terms),
        coset_candidate_counts=count_array,
        radius_q=radius_q,
        radius_p=radius_p,
        candidate_aliases=candidate_aliases,
    )


def _axis_llrs(log_scores: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    q_even = _logsumexp(log_scores[:, 0, :], axis=-1)
    q_odd = _logsumexp(log_scores[:, 1, :], axis=-1)
    p_even = _logsumexp(log_scores[:, :, 0], axis=-1)
    p_odd = _logsumexp(log_scores[:, :, 1], axis=-1)
    return q_even - q_odd, p_even - p_odd


@dataclass(frozen=True)
class TopKMAP2DResult:
    syndrome: NDArray[np.float64]
    k_requested: int
    k_effective_max: int
    saturated_full_candidate_sum: bool
    parity: NDArray[np.int64]
    logical_class: int | NDArray[np.int64]
    posterior: NDArray[np.float64]
    confidence: float | NDArray[np.float64]
    log_likelihoods: NDArray[np.float64]
    q_llr: float | NDArray[np.float64]
    p_llr: float | NDArray[np.float64]
    candidate_rectangle: tuple[int, int]
    candidate_aliases: int
    coset_candidate_count_min: int
    coset_candidate_count_max: int
    evidence_scope: str = "single_mode_topk_joint_gaussian_alias_sum"


def _result_from_prepared(
    prepared: _PreparedTopKCandidates,
    k: int,
    prior: NDArray[np.float64],
) -> TopKMAP2DResult:
    requested = _validate_k(k)
    selected_logs: list[NDArray[np.float64]] = []
    index = min(requested, prepared.candidate_aliases) - 1
    for cumulative in prepared.cumulative_log_terms:
        selected_logs.append(cumulative[:, index])
    log_likelihoods = np.stack(selected_logs, axis=-1).reshape(-1, 2, 2)
    if not np.all(np.isfinite(log_likelihoods)):
        raise RuntimeError("top-K likelihood contains a non-finite logical coset")
    log_scores = log_likelihoods + np.log(prior)[None, :, :]
    log_evidence = _logsumexp(log_scores, axis=(-2, -1))
    posterior = np.exp(log_scores - log_evidence[:, None, None])
    flat_posterior = posterior.reshape(-1, 4)
    logical_class = np.argmax(flat_posterior, axis=-1).astype(np.int64)
    parity = np.stack((logical_class // 2, logical_class % 2), axis=-1)
    ordered = np.sort(flat_posterior, axis=-1)
    confidence = ordered[:, -1] - ordered[:, -2]
    q_llr, p_llr = _axis_llrs(log_scores)
    max_count = int(np.max(prepared.coset_candidate_counts))
    min_count = int(np.min(prepared.coset_candidate_counts))
    effective_max = min(requested, max_count)
    scalar = prepared.original_shape == ()
    output_shape = prepared.original_shape
    return TopKMAP2DResult(
        syndrome=prepared.syndrome.reshape(output_shape + (2,)).copy(),
        k_requested=requested,
        k_effective_max=effective_max,
        saturated_full_candidate_sum=requested >= max_count,
        parity=parity.reshape(output_shape + (2,)),
        logical_class=(int(logical_class[0]) if scalar else logical_class.reshape(output_shape)),
        posterior=posterior.reshape(output_shape + (2, 2)),
        confidence=(float(confidence[0]) if scalar else confidence.reshape(output_shape)),
        log_likelihoods=log_likelihoods.reshape(output_shape + (2, 2)),
        q_llr=(float(q_llr[0]) if scalar else q_llr.reshape(output_shape)),
        p_llr=(float(p_llr[0]) if scalar else p_llr.reshape(output_shape)),
        candidate_rectangle=(2 * prepared.radius_q + 1, 2 * prepared.radius_p + 1),
        candidate_aliases=prepared.candidate_aliases,
        coset_candidate_count_min=min_count,
        coset_candidate_count_max=max_count,
    )


def topk_map_decode_2d(
    syndrome: ArrayLike,
    covariance: ArrayLike,
    k: int,
    *,
    mean: ArrayLike = (0.0, 0.0),
    lattice: float = LATTICE_CONST,
    prior: ArrayLike | None = None,
    tail_sigma: float = 10.0,
) -> TopKMAP2DResult:
    """Decode with the K highest joint aliases retained inside every coset."""

    requested = _validate_k(k)
    validated_prior = _validate_prior(prior)
    prepared = _prepare_topk_candidates(
        syndrome,
        covariance,
        mean=mean,
        lattice=lattice,
        tail_sigma=tail_sigma,
    )
    return _result_from_prepared(prepared, requested, validated_prior)


def topk_map_sweep_2d(
    syndrome: ArrayLike,
    covariance: ArrayLike,
    k_values: Sequence[int],
    *,
    mean: ArrayLike = (0.0, 0.0),
    lattice: float = LATTICE_CONST,
    prior: ArrayLike | None = None,
    tail_sigma: float = 10.0,
) -> dict[int, TopKMAP2DResult]:
    """Evaluate multiple K values after ranking the candidate aliases once."""

    try:
        requested = tuple(_validate_k(value) for value in k_values)
    except TypeError as exc:
        raise ValueError("k_values must be a sequence of positive integers") from exc
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("k_values must contain unique positive integers")
    if tuple(sorted(requested)) != requested:
        raise ValueError("k_values must be strictly increasing")
    validated_prior = _validate_prior(prior)
    prepared = _prepare_topk_candidates(
        syndrome,
        covariance,
        mean=mean,
        lattice=lattice,
        tail_sigma=tail_sigma,
    )
    return {
        value: _result_from_prepared(prepared, value, validated_prior)
        for value in requested
    }


@dataclass(frozen=True)
class TopKCostProfile:
    k_requested: int
    k_effective_upper: int
    candidate_rectangle_q: int
    candidate_rectangle_p: int
    candidate_aliases: int
    max_coset_candidates: int
    retained_terms_upper: int
    gaussian_multiplications: int
    gaussian_additions: int
    exponential_lut_queries: int
    streaming_topk_comparisons_upper: int
    likelihood_accumulations_upper: int
    alias_index_bits: int
    retained_state_bits: int
    serial_cycle_upper_proxy: int
    candidate_pipeline_cycle_lower_proxy: int
    saturated_full_candidate_sum: bool
    target_lut: int | None = None
    target_ff: int | None = None
    target_bram: int | None = None
    target_dsp: int | None = None
    target_fmax_hz: float | None = None
    target_measured: bool = False
    scope: str = "deterministic_operation_storage_proxy_not_synthesis"


def topk_cost_profile(
    covariance: ArrayLike,
    k: int,
    *,
    mean: ArrayLike = (0.0, 0.0),
    lattice: float = LATTICE_CONST,
    tail_sigma: float = 10.0,
    value_bits: int = 24,
) -> TopKCostProfile:
    """Return a deterministic serial-streaming cost proxy.

    One candidate belongs to exactly one coset.  The comparison upper bound
    assumes a length-K ordered insertion list; it is deliberately conservative
    and is not a measured RTL latency or synthesis utilization result.  The
    proxy describes a probability-domain hardware mapping: evaluate each
    Gaussian quadratic form, issue one exponential-LUT query per alias, retain
    aliases by the equivalent log-weight ordering, then add retained weights.
    It therefore does not pretend that NumPy's log-domain ``logaddexp`` call is
    itself an RTL implementation or a one-cycle primitive.
    """

    requested = _validate_k(k)
    spacing = _validate_lattice(lattice)
    _, _, _, sigma_q, sigma_p = _validate_covariance(covariance)
    tail = float(tail_sigma)
    if not isfinite(tail) or tail <= 0.0:
        raise ValueError("tail_sigma must be finite and positive")
    if isinstance(value_bits, bool) or not isinstance(value_bits, (int, np.integer)):
        raise TypeError("value_bits must be an integer")
    if int(value_bits) < 8:
        raise ValueError("value_bits must be at least 8")
    radius_q = max(2, int(ceil(tail * sigma_q / spacing)) + 2)
    radius_p = max(2, int(ceil(tail * sigma_p / spacing)) + 2)
    count_q = 2 * radius_q + 1
    count_p = 2 * radius_p + 1
    candidates = count_q * count_p
    max_coset = ceil(count_q / 2) * ceil(count_p / 2)
    effective = min(requested, max_coset)
    retained = 4 * effective
    mean_array = np.asarray(mean, dtype=np.float64)
    if mean_array.shape != (2,) or not np.all(np.isfinite(mean_array)):
        raise ValueError("cost-model mean must contain two finite values")
    max_center = int(ceil(float(np.max(np.abs(mean_array))) / spacing + 0.5))
    max_alias_magnitude = max_center + max(radius_q, radius_p)
    alias_bits = max(1, int(ceil(np.log2(2 * max_alias_magnitude + 1))))
    comparison_upper = candidates * effective
    accumulations = max(0, retained - 4)
    serial_cycles = candidates * (1 + effective) + accumulations + 7
    pipeline_lower = candidates + effective + max(0, effective - 1) + 7
    return TopKCostProfile(
        k_requested=requested,
        k_effective_upper=effective,
        candidate_rectangle_q=count_q,
        candidate_rectangle_p=count_p,
        candidate_aliases=candidates,
        max_coset_candidates=max_coset,
        retained_terms_upper=retained,
        gaussian_multiplications=6 * candidates,
        gaussian_additions=4 * candidates,
        exponential_lut_queries=candidates,
        streaming_topk_comparisons_upper=comparison_upper,
        likelihood_accumulations_upper=accumulations,
        alias_index_bits=alias_bits,
        retained_state_bits=retained * (int(value_bits) + 2 * alias_bits),
        serial_cycle_upper_proxy=serial_cycles,
        candidate_pipeline_cycle_lower_proxy=pipeline_lower,
        saturated_full_candidate_sum=requested >= max_coset,
    )


@dataclass(frozen=True)
class TopKValidationScenario:
    scenario_id: str
    sigma_q_lattice: float
    sigma_p_lattice: float
    rho: float
    mean_lattice: tuple[float, float]

    def covariance(self) -> NDArray[np.float64]:
        return covariance_from_sigmas(
            self.sigma_q_lattice * LATTICE_CONST,
            self.sigma_p_lattice * LATTICE_CONST,
            self.rho,
        )

    def mean(self) -> NDArray[np.float64]:
        return np.asarray(self.mean_lattice, dtype=np.float64) * LATTICE_CONST


def topk_validation_scenarios() -> tuple[TopKValidationScenario, ...]:
    return (
        TopKValidationScenario("narrow_isotropic", 0.16, 0.16, 0.0, (0.18, -0.10)),
        TopKValidationScenario("anisotropic", 0.45, 0.18, 0.25, (0.22, -0.16)),
        TopKValidationScenario("positive_correlation", 0.38, 0.32, 0.85, (0.20, -0.17)),
        TopKValidationScenario("negative_correlation", 0.38, 0.32, -0.85, (-0.19, 0.16)),
        TopKValidationScenario("broad_correlated", 0.65, 0.55, 0.65, (0.31, -0.27)),
        TopKValidationScenario("large_bias", 0.50, 0.42, -0.55, (0.48, -0.42)),
    )


@dataclass(frozen=True)
class TopKValidationConfig:
    k_values: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
    evaluation_seeds: tuple[int, ...] = (20260751, 20260752, 20260753, 20260754)
    samples_per_seed: int = 12_000
    tail_sigma: float = 8.0
    chunk_size: int = 1_500
    confidence_level: float = 0.95
    value_bits: int = 24

    def __post_init__(self) -> None:
        values = tuple(_validate_k(value) for value in self.k_values)
        if len(values) < 5 or tuple(sorted(set(values))) != values or values[0] != 1:
            raise ValueError("k_values must be unique increasing values beginning at K=1")
        seeds = tuple(self.evaluation_seeds)
        if len(seeds) < 4 or len(set(seeds)) != len(seeds):
            raise ValueError("evaluation_seeds must contain at least four unique values")
        if any(isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) for seed in seeds):
            raise TypeError("evaluation seeds must be integers")
        if any(int(seed) < 0 for seed in seeds):
            raise ValueError("evaluation seeds must be non-negative")
        for name, minimum in (("samples_per_seed", 4_000), ("chunk_size", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer")
            if int(value) < minimum:
                raise ValueError(f"{name} must be at least {minimum}")
        if not isfinite(self.tail_sigma) or self.tail_sigma <= 0.0:
            raise ValueError("tail_sigma must be finite and positive")
        if not isfinite(self.confidence_level) or not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must lie strictly between 0 and 1")
        if isinstance(self.value_bits, bool) or not isinstance(self.value_bits, (int, np.integer)):
            raise TypeError("value_bits must be an integer")
        if int(self.value_bits) < 8:
            raise ValueError("value_bits must be at least 8")
        object.__setattr__(self, "k_values", values)
        object.__setattr__(self, "evaluation_seeds", tuple(int(seed) for seed in seeds))
        object.__setattr__(self, "samples_per_seed", int(self.samples_per_seed))
        object.__setattr__(self, "chunk_size", int(self.chunk_size))


def _paired_seed_interval(values: Sequence[float], confidence_level: float) -> dict[str, float]:
    samples = np.asarray(values, dtype=np.float64)
    if samples.shape[0] < 4 or not np.all(np.isfinite(samples)):
        raise ValueError("seed-cluster interval requires at least four finite values")
    estimate = float(np.mean(samples))
    standard_error = float(np.std(samples, ddof=1) / sqrt(samples.size))
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    return {
        "estimate": estimate,
        "standard_error": standard_error,
        "ci_low": estimate - z * standard_error,
        "ci_high": estimate + z * standard_error,
        "cluster_unit": "evaluation_seed",
    }


def _full_axis_llrs(log_likelihoods: NDArray[np.float64], prior: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    scores = log_likelihoods + np.log(prior)[None, :, :]
    return _axis_llrs(scores)


def _source_sha256(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def validate_topk_comparison_registration() -> tuple[str, ...]:
    from cnn_fpga.benchmark.standard_binning_baseline import (
        major_comparison_registry,
        validate_major_comparison_registry,
    )

    gates = validate_major_comparison_registry()
    matches = [
        entry
        for entry in major_comparison_registry()
        if entry.comparison_id == "t3_1_5_topk_periodic_map_sensitivity"
    ]
    if len(matches) != 1:
        raise ValueError("T3.1.5 approximation comparison must be registered exactly once")
    entry = matches[0]
    if entry.method_ids != (FULL_PERIODIC_GAUSSIAN_MAP_ID, TOPK_LATTICE_COSET_MAP_ID):
        raise ValueError("T3.1.5 comparison method schema is inconsistent")
    if entry.standard_binning_policy != "not_applicable":
        raise ValueError("same-decoder K sensitivity must not be an algorithm ranking table")
    return (*gates, "registry:t3_1_5_topk_sensitivity")


def build_topk_validation(
    config: TopKValidationConfig | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    settings = TopKValidationConfig() if config is None else config
    if not isinstance(settings, TopKValidationConfig):
        raise TypeError("config must be TopKValidationConfig")
    registry_gates = validate_topk_comparison_registration()
    prior = np.full((2, 2), 0.25)
    rows: list[dict[str, object]] = []
    scenario_payloads: list[dict[str, object]] = []
    k1_nonexact_scenarios = 0
    final_exact_scenarios = 0
    all_convergence_k: list[int] = []

    for scenario_index, scenario in enumerate(topk_validation_scenarios()):
        covariance = scenario.covariance()
        mean = scenario.mean()
        costs = {
            k: topk_cost_profile(
                covariance,
                k,
                mean=mean,
                tail_sigma=settings.tail_sigma,
                value_bits=settings.value_bits,
            )
            for k in settings.k_values
        }
        aggregates: dict[int, dict[str, object]] = {
            k: {
                "seed_ler_deltas": [],
                "decision_disagreements": 0,
                "topk_errors": 0,
                "full_errors": 0,
                "q_llr_errors": [],
                "p_llr_errors": [],
                "posterior_tv_sum": 0.0,
                "posterior_tv_max": 0.0,
                "samples": 0,
            }
            for k in settings.k_values
        }
        trace_hashes: set[str] = set()

        for seed in settings.evaluation_seeds:
            sequence = np.random.SeedSequence([seed, scenario_index, 315])
            rng = np.random.default_rng(sequence)
            factor = np.linalg.cholesky(covariance)
            displacements = mean + rng.standard_normal((settings.samples_per_seed, 2)) @ factor.T
            q_standard = standard_binning_1d(displacements[:, 0])
            p_standard = standard_binning_1d(displacements[:, 1])
            syndrome = np.stack((np.asarray(q_standard.syndrome), np.asarray(p_standard.syndrome)), axis=-1)
            truth = 2 * np.asarray(q_standard.logical_parity) + np.asarray(p_standard.logical_parity)
            trace_hash = hashlib.sha256(
                np.ascontiguousarray(displacements).view(np.uint8).tobytes()
            ).hexdigest()
            trace_hashes.add(trace_hash)
            seed_metrics = {
                k: {
                    "disagreement": 0,
                    "topk_errors": 0,
                    "full_errors": 0,
                    "q_llr": [],
                    "p_llr": [],
                    "tv_sum": 0.0,
                    "tv_max": 0.0,
                }
                for k in settings.k_values
            }
            for start in range(0, settings.samples_per_seed, settings.chunk_size):
                stop = min(start + settings.chunk_size, settings.samples_per_seed)
                chunk = syndrome[start:stop]
                chunk_truth = truth[start:stop]
                full = map_decode_2d(
                    chunk,
                    covariance,
                    mean=mean,
                    prior=prior,
                    tail_sigma=settings.tail_sigma,
                )
                full_class = np.asarray(full.logical_class, dtype=np.int64)
                full_posterior = np.asarray(full.posterior)
                full_q_llr, full_p_llr = _full_axis_llrs(full.log_likelihoods, prior)
                sweep = topk_map_sweep_2d(
                    chunk,
                    covariance,
                    settings.k_values,
                    mean=mean,
                    prior=prior,
                    tail_sigma=settings.tail_sigma,
                )
                full_failure = full_class != chunk_truth
                for k, result in sweep.items():
                    topk_class = np.asarray(result.logical_class, dtype=np.int64)
                    topk_failure = topk_class != chunk_truth
                    q_error = np.abs(np.asarray(result.q_llr) - full_q_llr)
                    p_error = np.abs(np.asarray(result.p_llr) - full_p_llr)
                    tv = 0.5 * np.sum(
                        np.abs(np.asarray(result.posterior) - full_posterior), axis=(-2, -1)
                    )
                    metrics = seed_metrics[k]
                    metrics["disagreement"] += int(np.sum(topk_class != full_class))
                    metrics["topk_errors"] += int(np.sum(topk_failure))
                    metrics["full_errors"] += int(np.sum(full_failure))
                    metrics["q_llr"].append(q_error)
                    metrics["p_llr"].append(p_error)
                    metrics["tv_sum"] += float(np.sum(tv))
                    metrics["tv_max"] = max(float(metrics["tv_max"]), float(np.max(tv)))

            for k in settings.k_values:
                metrics = seed_metrics[k]
                q_errors = np.concatenate(metrics["q_llr"])
                p_errors = np.concatenate(metrics["p_llr"])
                count = settings.samples_per_seed
                ler_delta = (metrics["topk_errors"] - metrics["full_errors"]) / count
                aggregate = aggregates[k]
                aggregate["seed_ler_deltas"].append(ler_delta)
                aggregate["decision_disagreements"] += metrics["disagreement"]
                aggregate["topk_errors"] += metrics["topk_errors"]
                aggregate["full_errors"] += metrics["full_errors"]
                aggregate["q_llr_errors"].append(q_errors)
                aggregate["p_llr_errors"].append(p_errors)
                aggregate["posterior_tv_sum"] += metrics["tv_sum"]
                aggregate["posterior_tv_max"] = max(
                    float(aggregate["posterior_tv_max"]), float(metrics["tv_max"])
                )
                aggregate["samples"] += count
                rows.append(
                    {
                        "scenario_id": scenario.scenario_id,
                        "evaluation_seed": seed,
                        "trace_sha256": trace_hash,
                        "samples": count,
                        "K": k,
                        "full_map_ler": metrics["full_errors"] / count,
                        "topk_map_ler": metrics["topk_errors"] / count,
                        "topk_minus_full_ler": ler_delta,
                        "decision_disagreement_rate": metrics["disagreement"] / count,
                        "q_llr_mean_abs_error": float(np.mean(q_errors)),
                        "p_llr_mean_abs_error": float(np.mean(p_errors)),
                        "axis_llr_max_abs_error": float(max(np.max(q_errors), np.max(p_errors))),
                        "posterior_tv_mean": metrics["tv_sum"] / count,
                        "posterior_tv_max": metrics["tv_max"],
                        "candidate_aliases": costs[k].candidate_aliases,
                        "max_coset_candidates": costs[k].max_coset_candidates,
                        "retained_state_bits": costs[k].retained_state_bits,
                        "streaming_topk_comparisons_upper": costs[k].streaming_topk_comparisons_upper,
                        "serial_cycle_upper_proxy": costs[k].serial_cycle_upper_proxy,
                    }
                )

        sweep_payload: list[dict[str, object]] = []
        convergence_k: int | None = None
        for k in settings.k_values:
            aggregate = aggregates[k]
            count = int(aggregate["samples"])
            q_errors = np.concatenate(aggregate["q_llr_errors"])
            p_errors = np.concatenate(aggregate["p_llr_errors"])
            axis_errors = np.concatenate((q_errors, p_errors))
            interval = _paired_seed_interval(
                aggregate["seed_ler_deltas"], settings.confidence_level
            )
            disagreement = aggregate["decision_disagreements"] / count
            llr_p99 = float(np.quantile(axis_errors, 0.99))
            row = {
                "K": k,
                "full_map_ler": aggregate["full_errors"] / count,
                "topk_map_ler": aggregate["topk_errors"] / count,
                "topk_minus_full_ler_seed_cluster_ci": interval,
                "decision_disagreement_rate": disagreement,
                "q_llr_mean_abs_error": float(np.mean(q_errors)),
                "p_llr_mean_abs_error": float(np.mean(p_errors)),
                "axis_llr_p99_abs_error": llr_p99,
                "axis_llr_max_abs_error": float(np.max(axis_errors)),
                "posterior_tv_mean": aggregate["posterior_tv_sum"] / count,
                "posterior_tv_max": aggregate["posterior_tv_max"],
                "cost": asdict(costs[k]),
            }
            sweep_payload.append(row)
            if convergence_k is None and (
                disagreement <= 1.0e-4
                and llr_p99 <= 1.0e-3
                and abs(float(interval["estimate"])) <= 1.0e-4
            ):
                convergence_k = k
        if convergence_k is None:
            raise AssertionError(f"{scenario.scenario_id} did not converge within the K scan")
        all_convergence_k.append(convergence_k)
        # K=1 often preserves the hard class in narrow Gaussian conditions,
        # but it is not the full soft decoder: require a resolved LLR/posterior
        # difference rather than manufacturing a hard-decision failure.
        k1_nonexact_scenarios += sweep_payload[0]["axis_llr_p99_abs_error"] > 1.0e-3
        final = sweep_payload[-1]
        final_exact = (
            final["decision_disagreement_rate"] == 0.0
            and final["axis_llr_max_abs_error"] <= 5.0e-13
            and abs(final["topk_minus_full_ler_seed_cluster_ci"]["estimate"]) <= 1.0e-15
        )
        final_exact_scenarios += final_exact
        scenario_payloads.append(
            {
                "scenario": asdict(scenario),
                "covariance": covariance.tolist(),
                "mean": mean.tolist(),
                "samples": len(settings.evaluation_seeds) * settings.samples_per_seed,
                "unique_trace_hashes": len(trace_hashes),
                "convergence_K": convergence_k,
                "sweep": sweep_payload,
            }
        )

    final_costs = [
        scenario["sweep"][-1]["cost"] for scenario in scenario_payloads
    ]
    gates = {
        "descriptor_is_single_mode_and_excludes_matching": (
            "K_minimum_weight_matchings" in TOPK_DECODER_DESCRIPTOR.excluded_algorithms
            and "single_mode" in TOPK_DECODER_DESCRIPTOR.evidence_scope
        ),
        "comparison_registered_as_implementation_sensitivity": bool(registry_gates),
        "K1_is_not_silently_treated_as_full_map": k1_nonexact_scenarios >= 3,
        "all_scenarios_reach_registered_convergence_region": len(all_convergence_k)
        == len(topk_validation_scenarios()),
        "largest_K_exactly_matches_full_candidate_sum": final_exact_scenarios
        == len(topk_validation_scenarios()),
        "all_evaluation_traces_are_unique": all(
            scenario["unique_trace_hashes"] == len(settings.evaluation_seeds)
            for scenario in scenario_payloads
        ),
        "cost_storage_and_comparisons_are_monotone_in_K": all(
            all(
                sweep[index]["cost"][field] <= sweep[index + 1]["cost"][field]
                for index in range(len(sweep) - 1)
                for field in ("retained_state_bits", "streaming_topk_comparisons_upper")
            )
            for sweep in (scenario["sweep"] for scenario in scenario_payloads)
        ),
        "hardware_utilization_fields_remain_unmeasured": all(
            cost["target_lut"] is None
            and cost["target_ff"] is None
            and cost["target_bram"] is None
            and cost["target_dsp"] is None
            and cost["target_fmax_hz"] is None
            and not cost["target_measured"]
            for cost in final_costs
        ),
    }
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise AssertionError(f"top-K validation gates failed: {failed}")
    source_paths = (
        Path(__file__),
        Path(__file__).parents[2] / "physics" / "ideal_gkp_decoder.py",
        Path(__file__).with_name("standard_binning_baseline.py"),
    )
    payload: dict[str, object] = {
        "schema_version": "t3.1.5-topk-lattice-coset-map-v1",
        "task_id": "T3.1.5",
        "status": "PASS",
        "implementation_sha256": _source_sha256(source_paths),
        "descriptor": asdict(TOPK_DECODER_DESCRIPTOR),
        "config": asdict(settings),
        "comparison_registry_gates": list(registry_gates),
        "comparison_contract": {
            "approximation": TOPK_LATTICE_COSET_MAP_ID,
            "reference": FULL_PERIODIC_GAUSSIAN_MAP_ID,
            "same_candidate_rectangle": True,
            "logical_class_encoding": "2*parity_q+parity_p",
            "full_reference_scope": "tail_sigma_truncated_periodic_gaussian_map",
        },
        "scenarios": scenario_payloads,
        "aggregate": {
            "scenarios": len(scenario_payloads),
            "evaluation_seeds_per_scenario": len(settings.evaluation_seeds),
            "samples": len(scenario_payloads)
            * len(settings.evaluation_seeds)
            * settings.samples_per_seed,
            "source_data_rows": len(rows),
            "convergence_K_min": min(all_convergence_k),
            "convergence_K_max": max(all_convergence_k),
            "K1_nonexact_scenarios": k1_nonexact_scenarios,
            "largest_K_exact_scenarios": final_exact_scenarios,
        },
        "gate_summary": {"passed": len(gates), "failed": 0, "gates": gates},
        "claim_boundary": {
            "allowed": (
                "per-coset top-K joint lattice likelihood provides a deterministic "
                "accuracy-operation-storage tradeoff and converges to the registered full "
                "periodic Gaussian MAP candidate sum"
            ),
            "forbidden": (
                "surface-code K-MWM implementation, hardware synthesis/utilization/latency "
                "measurement, universal optimal K, finite-energy recovery, or experimental LER"
            ),
        },
    }
    return json.loads(json.dumps(payload, ensure_ascii=False)), rows


def write_topk_validation(
    json_path: str | Path,
    csv_path: str | Path,
    config: TopKValidationConfig | None = None,
) -> dict[str, object]:
    payload, rows = build_topk_validation(config)
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
        description="Validate the T3.1.5 top-K lattice-coset MAP baseline"
    )
    parser.add_argument("--json", default="docs/t3_1_5_topk_map_validation.json")
    parser.add_argument("--csv", default="docs/t3_1_5_topk_map_source_data.csv")
    arguments = parser.parse_args()
    result = write_topk_validation(arguments.json, arguments.csv)
    print(json.dumps(result["gate_summary"], ensure_ascii=False))


__all__ = [
    "TOPK_LATTICE_COSET_MAP_ID",
    "FULL_PERIODIC_GAUSSIAN_MAP_ID",
    "TopKDecoderDescriptor",
    "TOPK_DECODER_DESCRIPTOR",
    "TopKMAP2DResult",
    "TopKCostProfile",
    "TopKValidationScenario",
    "TopKValidationConfig",
    "topk_map_decode_2d",
    "topk_map_sweep_2d",
    "topk_cost_profile",
    "topk_validation_scenarios",
    "validate_topk_comparison_registration",
    "build_topk_validation",
    "write_topk_validation",
]
