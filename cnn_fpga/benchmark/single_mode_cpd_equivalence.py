"""T6.17.1 single-mode square-lattice Euclidean CPD/CI equivalence audit.

The equivalence proved here is deliberately narrow: an isotropic Euclidean
metric on the square lattice ``lambda Z^2``.  It does not identify closest
point decoding with periodic coset-summing MAP under biased, correlated or
finite-energy likelihoods.
"""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from math import exp, floor, isfinite, pi, sqrt
from pathlib import Path
from time import perf_counter
import tracemalloc
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cnn_fpga.runtime.parametric_map_lut import (
    ParametricMAPLUTConfig,
    software_decode_syndrome_code,
)
from physics.constants import LATTICE_CONST
from physics.finite_energy_gkp import damped_projector_state
from physics.ideal_gkp_decoder import (
    covariance_from_sigmas,
    map_decode_1d,
    map_decode_2d,
    standard_binning_1d,
)


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "T6.17.1"
SCHEMA_VERSION = "t6.17.1-single-mode-cpd-ci-equivalence-v1"
VERDICT = "PASS_SINGLE_MODE_EUCLIDEAN_CPD_EQUALS_CI_WITH_MAP_BOUNDARIES"
PREREG_CONFIG = ROOT / "configs" / "literature" / "t6_16_3_secondary_preregistration.json"
ONTOLOGY = ROOT / "docs" / "t6_16_2_comparison_ontology.json"
SOURCE_AUDIT = ROOT / "docs" / "t6_16_1_secondary_method_source_audit.json"
IDEAL_DECODER = ROOT / "physics" / "ideal_gkp_decoder.py"
FINITE_ENERGY = ROOT / "physics" / "finite_energy_gkp.py"
RUNTIME_LUT = ROOT / "cnn_fpga" / "runtime" / "parametric_map_lut.py"
DEFAULT_REPORT = ROOT / "docs" / "t6_17_1_single_mode_cpd_equivalence.json"
DEFAULT_SOURCE_DATA = ROOT / "docs" / "t6_17_1_single_mode_cpd_equivalence_source_data.csv"
DEFAULT_MARKDOWN = ROOT / "docs" / "single_mode_cpd_equivalence.md"

ADC_BITS = 10
ADC_LEVELS = 1 << ADC_BITS
PRODUCTION_POINTS = ADC_LEVELS * ADC_LEVELS
PRODUCTION_Q_CHUNK = 32
BOUNDARY_SAMPLES = 1_000_000
BOUNDARY_CHUNK = 100_000
BOUNDARY_SEED = 61_710_001
BOUNDARY_ALIAS_MIN = -2048
BOUNDARY_ALIAS_MAX = 2048
CPD_OFFSETS = np.asarray(
    [(q, p) for q in (2, 1, 0, -1) for p in (2, 1, 0, -1)],
    dtype=np.int64,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    ).hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _binding(path: Path) -> dict[str, Any]:
    return {"path": _relative(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ontology_semantic(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in ("ontology", "source_metric_crosswalk", "ranking_policy", "parent_contracts", "verdict")
    }


def _source_semantic(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in ("scope", "sources", "methods", "claim_audit", "derived_evidence", "comparison_policy", "verdict")
    }


def _source_scope(source_ids: list[str]) -> list[dict[str, Any]]:
    source = _load(SOURCE_AUDIT)
    records = {row["source_id"]: row for row in source["sources"]}
    if set(source_ids) != {"NOH2022_SURFACE_GKP", "LIN2023_CPD"}:
        raise ValueError("T6.17.1 source set drifted from the frozen preregistration")
    interpretations = {
        "NOH2022_SURFACE_GKP": "closest-integer terminology/context only; no CNOT or 9.9 dB result is reproduced here",
        "LIN2023_CPD": "closest-point terminology and square-lattice specialization only; no multimode threshold is reproduced here",
    }
    return [
        {
            "source_id": source_id,
            "record_sha256": _canonical_sha256(records[source_id]),
            "title": records[source_id]["title"],
            "version": records[source_id]["version"],
            "formal_identifier": records[source_id]["formal_identifier"],
            "interpretation": interpretations[source_id],
        }
        for source_id in source_ids
    ]


def _zero_event_upper_bound(trials: int, confidence: float = 0.95) -> float:
    if trials <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("zero-event bound requires positive trials and confidence in (0,1)")
    return float(1.0 - (1.0 - confidence) ** (1.0 / trials))


def _preregistered_experiment() -> dict[str, Any]:
    config = _load(PREREG_CONFIG)
    rows = [row for row in config["experiments"] if row["task_id"] == TASK_ID]
    if len(rows) != 1:
        raise ValueError("T6.17.1 requires exactly one frozen preregistration row")
    return rows[0]


def closest_integer_indices(normalized: ArrayLike) -> NDArray[np.int64]:
    """Componentwise CI with the frozen half-open/ties-up convention."""
    values = np.asarray(normalized, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != 2:
        raise ValueError("normalized coordinates must have shape (...,2)")
    if not np.all(np.isfinite(values)):
        raise ValueError("normalized coordinates must be finite")
    if np.any(values <= -(2**61)) or np.any(values >= 2**61):
        raise ValueError("normalized coordinates exceed the exact int64 audit range")
    # ``floor(x + 0.5)`` incorrectly promotes the representable predecessor of
    # +0.5 because the addition can itself round to 1.0.  Split integer and
    # fractional parts so one-ULP-below, exact tie and one-ULP-above remain
    # distinct under the half-open/ties-up contract.
    base = np.floor(values)
    fraction = values - base
    return (base + (fraction >= 0.5)).astype(np.int64)


def brute_force_square_cpd_indices(normalized: ArrayLike) -> NDArray[np.int64]:
    """Independent local enumeration for Euclidean closest point on Z^2.

    Sixteen lattice candidates surrounding ``floor(x)`` are evaluated.  Their
    order is descending in both axes, so ``numpy.argmin`` implements the same
    exact-tie choice as ``floor(x+1/2)`` without calling the CI routine.
    """
    values = np.asarray(normalized, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != 2:
        raise ValueError("normalized coordinates must have shape (...,2)")
    if not np.all(np.isfinite(values)):
        raise ValueError("normalized coordinates must be finite")
    flat = values.reshape(-1, 2)
    if np.any(flat <= -(2**61)) or np.any(flat >= 2**61):
        raise ValueError("normalized coordinates exceed the exact int64 audit range")
    base = np.floor(flat).astype(np.int64)
    candidates = base[:, None, :] + CPD_OFFSETS[None, :, :]
    residual = flat[:, None, :] - candidates.astype(np.float64)
    squared_distance = np.einsum("nki,nki->nk", residual, residual)
    winners = candidates[np.arange(flat.shape[0]), np.argmin(squared_distance, axis=1)]
    # Squaring two mathematically equal half-cell distances can differ by one
    # float64 rounding unit.  Detect an exactly representable half-integer from
    # the input itself and apply the declared ties-up rule explicitly.  This is
    # distinct from CI for all non-tie inputs and keeps the enumerated oracle
    # honest at the only non-unique minimizers.
    exact_half = flat - np.floor(flat) == 0.5
    winners = np.where(exact_half, base + 1, winners)
    return winners.reshape(values.shape)


def _logical_class(indices: NDArray[np.int64]) -> NDArray[np.uint8]:
    return (
        2 * np.mod(indices[..., 0], 2) + np.mod(indices[..., 1], 2)
    ).astype(np.uint8)


def _production_domain_audit() -> dict[str, Any]:
    config = ParametricMAPLUTConfig()
    if config.adc_bits != ADC_BITS:
        raise ValueError("live production LUT no longer uses the frozen 10-bit syndrome word")
    physical_axis = np.asarray(
        [software_decode_syndrome_code(code, config) for code in range(ADC_LEVELS)],
        dtype=np.float64,
    )
    normalized_axis = physical_axis / config.lattice
    expected_axis = -0.5 + (np.arange(ADC_LEVELS, dtype=np.float64) + 0.5) / ADC_LEVELS
    decode_error = float(np.max(np.abs(normalized_axis - expected_axis)))
    standard_axis = np.asarray(standard_binning_1d(physical_axis, lattice=config.lattice).lattice_index)

    ci_hash = hashlib.sha256()
    cpd_hash = hashlib.sha256()
    mismatches = 0
    max_distance_gap = 0.0
    chunks: list[dict[str, Any]] = []
    start = perf_counter()
    tracemalloc.start()
    tracemalloc.reset_peak()
    for q_start in range(0, ADC_LEVELS, PRODUCTION_Q_CHUNK):
        q_stop = min(ADC_LEVELS, q_start + PRODUCTION_Q_CHUNK)
        coordinates = np.stack(
            (
                np.repeat(normalized_axis[q_start:q_stop], ADC_LEVELS),
                np.tile(normalized_axis, q_stop - q_start),
            ),
            axis=1,
        )
        ci = closest_integer_indices(coordinates)
        cpd = brute_force_square_cpd_indices(coordinates)
        local = int(np.count_nonzero(np.any(ci != cpd, axis=1)))
        mismatches += local
        ci_action = _logical_class(ci)
        cpd_action = _logical_class(cpd)
        ci_hash.update(ci_action.tobytes())
        cpd_hash.update(cpd_action.tobytes())
        ci_distance = np.sum((coordinates - ci) ** 2, axis=1)
        cpd_distance = np.sum((coordinates - cpd) ** 2, axis=1)
        max_distance_gap = max(max_distance_gap, float(np.max(np.abs(ci_distance - cpd_distance))))
        chunks.append({
            "q_start": q_start,
            "q_stop": q_stop,
            "points": int(coordinates.shape[0]),
            "mismatches": local,
            "ci_action_sha256": hashlib.sha256(ci_action.tobytes()).hexdigest(),
            "cpd_action_sha256": hashlib.sha256(cpd_action.tobytes()).hexdigest(),
        })
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    elapsed = perf_counter() - start
    return {
        "adc_bits_per_axis": ADC_BITS,
        "levels_per_axis": ADC_LEVELS,
        "points": PRODUCTION_POINTS,
        "input_is_canonical_centered_cell": True,
        "canonical_cell_action_is_all_zero": bool(np.all(standard_axis == 0)),
        "nontrivial_alias_boundary_evidence_is_separate": True,
        "half_open_cell_endpoints": [-0.5, 0.5],
        "positive_endpoint_excluded": True,
        "maximum_axis_decode_error_lattice_units": decode_error,
        "standard_binning_axis_mismatches_from_zero": int(np.count_nonzero(standard_axis)),
        "cpd_ci_mismatches": mismatches,
        "decision_mismatch_rate": mismatches / PRODUCTION_POINTS,
        "zero_mismatch_one_sided_95_upper_bound": _zero_event_upper_bound(PRODUCTION_POINTS),
        "ci_action_sha256": ci_hash.hexdigest(),
        "cpd_action_sha256": cpd_hash.hexdigest(),
        "maximum_squared_distance_gap": max_distance_gap,
        "runtime_seconds": elapsed,
        "memory_bytes": int(peak_bytes),
        "memory_boundary": "Python tracemalloc peak during chunked complete-domain audit",
        "chunks": chunks,
    }


def _boundary_coordinates(rng: np.random.Generator, count: int) -> tuple[np.ndarray, np.ndarray]:
    lower_cells = rng.integers(
        BOUNDARY_ALIAS_MIN, BOUNDARY_ALIAS_MAX + 1, size=(count, 2), dtype=np.int64
    )
    boundary = lower_cells.astype(np.float64) + 0.5
    modes = rng.integers(0, 6, size=(count, 2), dtype=np.int8)
    exponents = rng.integers(12, 41, size=(count, 2), dtype=np.int16)
    epsilon = np.exp2(-exponents.astype(np.float64))
    output = boundary.copy()
    # Use a 16-ULP band so the non-tie squared-distance ordering remains
    # distinguishable from the exact-tie float64 rounding error.
    ulp_band = 16.0 * np.abs(np.spacing(boundary))
    output = np.where(modes == 1, boundary - ulp_band, output)
    output = np.where(modes == 2, boundary + ulp_band, output)
    output = np.where(modes == 3, boundary - epsilon, output)
    output = np.where(modes == 4, boundary + epsilon, output)
    random_sign = np.where(rng.integers(0, 2, size=(count, 2)) == 0, -1.0, 1.0)
    output = np.where(modes == 5, boundary + random_sign * epsilon, output)
    return output, modes


def _boundary_audit() -> dict[str, Any]:
    rng = np.random.default_rng(BOUNDARY_SEED)
    ci_hash = hashlib.sha256()
    cpd_hash = hashlib.sha256()
    mismatches = 0
    exact_tie_coordinates = 0
    max_distance_gap = 0.0
    mode_counts = np.zeros(6, dtype=np.int64)
    chunks: list[dict[str, Any]] = []
    start = perf_counter()
    tracemalloc.start()
    tracemalloc.reset_peak()
    for start_index in range(0, BOUNDARY_SAMPLES, BOUNDARY_CHUNK):
        count = min(BOUNDARY_CHUNK, BOUNDARY_SAMPLES - start_index)
        coordinates, modes = _boundary_coordinates(rng, count)
        ci = closest_integer_indices(coordinates)
        cpd = brute_force_square_cpd_indices(coordinates)
        local = int(np.count_nonzero(np.any(ci != cpd, axis=1)))
        mismatches += local
        exact_tie_coordinates += int(np.count_nonzero(modes == 0))
        mode_counts += np.bincount(modes.ravel(), minlength=6)
        ci_class = _logical_class(ci)
        cpd_class = _logical_class(cpd)
        ci_hash.update(ci_class.tobytes())
        cpd_hash.update(cpd_class.tobytes())
        ci_distance = np.sum((coordinates - ci) ** 2, axis=1)
        cpd_distance = np.sum((coordinates - cpd) ** 2, axis=1)
        max_distance_gap = max(max_distance_gap, float(np.max(np.abs(ci_distance - cpd_distance))))
        chunks.append({
            "start": start_index,
            "stop": start_index + count,
            "points": count,
            "mismatches": local,
            "exact_tie_coordinates": int(np.count_nonzero(modes == 0)),
        })
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    elapsed = perf_counter() - start
    return {
        "seed": BOUNDARY_SEED,
        "points": BOUNDARY_SAMPLES,
        "coordinates": 2 * BOUNDARY_SAMPLES,
        "alias_cell_range_inclusive": [BOUNDARY_ALIAS_MIN, BOUNDARY_ALIAS_MAX],
        "construction_modes": ["exact_half", "sixteen_ulp_lower", "sixteen_ulp_upper", "epsilon_lower", "epsilon_upper", "random_epsilon_side"],
        "mode_coordinate_counts": mode_counts.tolist(),
        "exact_tie_coordinates": exact_tie_coordinates,
        "cpd_ci_mismatches": mismatches,
        "decision_mismatch_rate": mismatches / BOUNDARY_SAMPLES,
        "zero_mismatch_one_sided_95_upper_bound": _zero_event_upper_bound(BOUNDARY_SAMPLES),
        "ci_action_sha256": ci_hash.hexdigest(),
        "cpd_action_sha256": cpd_hash.hexdigest(),
        "maximum_squared_distance_gap": max_distance_gap,
        "runtime_seconds": elapsed,
        "memory_bytes": int(peak_bytes),
        "memory_boundary": "Python tracemalloc peak during deterministic chunked boundary audit",
        "chunks": chunks,
    }


def _brute_periodic_1d(
    syndrome: float, sigma: float, mean: float, prior_even: float, *, radius: int = 32
) -> dict[str, Any]:
    aliases = np.arange(-radius, radius + 1, dtype=np.int64)
    residual = syndrome + aliases.astype(np.float64) * LATTICE_CONST - mean
    weights = np.exp(-0.5 * (residual / sigma) ** 2) / (sigma * sqrt(2.0 * pi))
    even = float(np.sum(weights[np.mod(aliases, 2) == 0])) * prior_even
    odd = float(np.sum(weights[np.mod(aliases, 2) == 1])) * (1.0 - prior_even)
    return {"class": int(odd > even), "weighted_even": even, "weighted_odd": odd, "radius": radius}


def _brute_periodic_2d(
    syndrome: NDArray[np.float64], covariance: NDArray[np.float64], *, radius: int = 20
) -> dict[str, Any]:
    inverse = np.linalg.inv(covariance)
    normalizer = 2.0 * pi * sqrt(float(np.linalg.det(covariance)))
    scores = np.zeros((2, 2), dtype=np.float64)
    for alias_q in range(-radius, radius + 1):
        for alias_p in range(-radius, radius + 1):
            residual = syndrome + LATTICE_CONST * np.asarray([alias_q, alias_p], dtype=np.float64)
            scores[alias_q % 2, alias_p % 2] += exp(
                -0.5 * float(residual @ inverse @ residual)
            ) / normalizer
    posterior = scores / float(np.sum(scores))
    logical_class = int(np.argmax(posterior.reshape(-1)))
    return {"class": logical_class, "posterior": posterior.tolist(), "radius": radius}


def _counterexamples() -> list[dict[str, Any]]:
    # Canonical biased witness: one-cell mean shift swaps even/odd likelihoods.
    biased_syndrome = 0.0
    biased_sigma = 0.2 * LATTICE_CONST
    biased_mean = LATTICE_CONST
    biased_project = map_decode_1d(biased_syndrome, biased_sigma, mean=biased_mean)
    biased_brute = _brute_periodic_1d(
        biased_syndrome, biased_sigma, biased_mean, 0.5
    )

    # Fixed, pre-existing strong-correlation model; choose the first mismatch
    # on a deterministic 101x101 midpoint grid, not the largest margin.
    covariance = covariance_from_sigmas(0.42 * LATTICE_CONST, 0.42 * LATTICE_CONST, 0.9)
    axis = (-0.5 + (np.arange(101, dtype=np.float64) + 0.5) / 101.0) * LATTICE_CONST
    q_grid, p_grid = np.meshgrid(axis, axis, indexing="ij")
    correlated_grid = np.stack((q_grid.ravel(), p_grid.ravel()), axis=1)
    correlated_result = map_decode_2d(correlated_grid, covariance)
    correlated_classes = np.asarray(correlated_result.logical_class, dtype=np.int64)
    correlated_ci = _logical_class(closest_integer_indices(correlated_grid / LATTICE_CONST))
    correlated_diff = np.flatnonzero(correlated_classes != correlated_ci)
    if correlated_diff.size == 0:
        raise RuntimeError("fixed correlated witness grid contains no CI/MAP counterexample")
    correlated_index = int(correlated_diff[0])
    correlated_syndrome = correlated_grid[correlated_index]
    correlated_brute = _brute_periodic_2d(correlated_syndrome, covariance)

    # Finite-energy witness: equal-prior likelihood discrimination between the
    # two normalized damped-projector logical states.  Again select the first
    # mismatch on a fixed midpoint grid in the canonical cell.
    state0 = damped_projector_state("0", 1.0, tail_tolerance=1.0e-12)
    state1 = damped_projector_state("1", 1.0, tail_tolerance=1.0e-12)
    finite_grid = (-0.5 + (np.arange(2001, dtype=np.float64) + 0.5) / 2001.0) * LATTICE_CONST
    density0 = np.asarray(state0.probability_density(finite_grid), dtype=np.float64)
    density1 = np.asarray(state1.probability_density(finite_grid), dtype=np.float64)
    finite_map = (density1 > density0).astype(np.int64)
    finite_ci = np.asarray(standard_binning_1d(finite_grid).logical_parity, dtype=np.int64)
    finite_diff = np.flatnonzero(finite_map != finite_ci)
    if finite_diff.size == 0:
        raise RuntimeError("fixed finite-energy witness grid contains no CI/likelihood counterexample")
    finite_index = int(finite_diff[0])
    finite_x = float(finite_grid[finite_index])
    reconstructed = []
    for state in (state0, state1):
        table = state.peak_table
        psi = float(np.sum(table.coefficients * np.exp(
            -0.5 * (finite_x - table.centers) ** 2 / table.amplitude_variance
        )))
        reconstructed.append(psi * psi)
    finite_direct = [float(density0[finite_index]), float(density1[finite_index])]
    relative_reconstruction_error = max(
        abs(a - b) / max(abs(a), np.finfo(np.float64).tiny)
        for a, b in zip(finite_direct, reconstructed)
    )

    return [
        {
            "family": "biased",
            "selection_rule": "canonical one-lattice mean shift; not outcome-tuned",
            "syndrome_over_lattice": 0.0,
            "mean_over_lattice": 1.0,
            "sigma_over_lattice": 0.2,
            "ci_class": int(standard_binning_1d(biased_syndrome).logical_parity),
            "likelihood_map_class": int(biased_project.parity),
            "project_posterior_even": float(biased_project.posterior_even),
            "independent_brute": biased_brute,
            "is_counterexample": int(biased_project.parity) != int(standard_binning_1d(biased_syndrome).logical_parity),
        },
        {
            "family": "correlated",
            "selection_rule": "first lexicographic mismatch on fixed 101x101 canonical midpoint grid",
            "searched_points": int(correlated_grid.shape[0]),
            "mismatch_points": int(correlated_diff.size),
            "syndrome_over_lattice": (correlated_syndrome / LATTICE_CONST).tolist(),
            "sigma_q_over_lattice": 0.42,
            "sigma_p_over_lattice": 0.42,
            "rho": 0.9,
            "ci_class": int(correlated_ci[correlated_index]),
            "likelihood_map_class": int(correlated_classes[correlated_index]),
            "project_posterior": np.asarray(correlated_result.posterior)[correlated_index].tolist(),
            "independent_brute": correlated_brute,
            "is_counterexample": int(correlated_classes[correlated_index]) != int(correlated_ci[correlated_index]),
        },
        {
            "family": "finite_energy_likelihood",
            "selection_rule": "first mismatch on fixed 2001-point canonical midpoint grid",
            "searched_points": int(finite_grid.size),
            "mismatch_points": int(finite_diff.size),
            "model": "normalized damped-projector logical-state likelihoods with equal prior",
            "projector_delta": 1.0,
            "syndrome_over_lattice": finite_x / LATTICE_CONST,
            "ci_class": int(finite_ci[finite_index]),
            "likelihood_map_class": int(finite_map[finite_index]),
            "logical_state_densities": finite_direct,
            "independent_peak_table_densities": reconstructed,
            "maximum_relative_reconstruction_error": relative_reconstruction_error,
            "component_counts": [state0.component_count, state1.component_count],
            "is_counterexample": int(finite_map[finite_index]) != int(finite_ci[finite_index]),
        },
    ]


def _comparator_registry() -> list[dict[str, Any]]:
    return [
        {"method_id": "closest_integer", "equivalence_class": "square_isotropic_euclidean_nearest_lattice", "ranking_weight": 1, "decision_object": "nearest point in lambda Z^2"},
        {"method_id": "square_euclidean_cpd", "equivalence_class": "square_isotropic_euclidean_nearest_lattice", "ranking_weight": 0, "decision_object": "nearest point in lambda Z^2"},
        {"method_id": "periodic_coset_map", "equivalence_class": "likelihood_coset_sum", "ranking_weight": 1, "decision_object": "most likely logical coset under named likelihood/prior"},
    ]


def _proof_contract() -> dict[str, Any]:
    return {
        "lattice": "Lambda=lambda*Z^2",
        "metric": "isotropic Euclidean squared norm",
        "objective": "argmin_k in Z^2 ||x-lambda*k||_2^2 = argmin_(kq,kp) [(xq/lambda-kq)^2+(xp/lambda-kp)^2]",
        "separability": "the sum contains no cross term, so each integer coordinate minimizes independently",
        "decision_region": "R_k=product_i [(k_i-1/2)lambda,(k_i+1/2)lambda)",
        "tie_rule": "positive half boundary belongs to the larger integer in each axis",
        "conclusion": "CPD_lambdaZ2_Euclidean(x)=lambda*CI(x/lambda)",
        "excluded": ["biased mean/prior", "correlated Mahalanobis cross term", "periodic alias/coset summation", "finite-energy state likelihood", "outer code", "multimode structured lattice"],
    }


def _source_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for chunk in report["production_domain"]["chunks"]:
        rows.append({"record_type": "production_chunk", "record_id": f"q{chunk['q_start']}:{chunk['q_stop']}", "family": "square_euclidean", "points": chunk["points"], "mismatches": chunk["mismatches"], "value": chunk["ci_action_sha256"], "details": json.dumps(chunk, sort_keys=True)})
    for chunk in report["boundary_audit"]["chunks"]:
        rows.append({"record_type": "boundary_chunk", "record_id": f"b{chunk['start']}:{chunk['stop']}", "family": "square_euclidean", "points": chunk["points"], "mismatches": chunk["mismatches"], "value": chunk["exact_tie_coordinates"], "details": json.dumps(chunk, sort_keys=True)})
    for row in report["counterexamples"]:
        rows.append({"record_type": "counterexample", "record_id": row["family"], "family": row["family"], "points": row.get("searched_points", 1), "mismatches": int(row["is_counterexample"]), "value": row["likelihood_map_class"], "details": json.dumps(row, sort_keys=True)})
    for row in report["comparator_registry"]:
        rows.append({"record_type": "comparator", "record_id": row["method_id"], "family": row["equivalence_class"], "points": 0, "mismatches": 0, "value": row["ranking_weight"], "details": row["decision_object"]})
    for row in report["source_scope"]:
        rows.append({"record_type": "source", "record_id": row["source_id"], "family": "primary_literature", "points": 0, "mismatches": 0, "value": row["record_sha256"], "details": row["interpretation"]})
    for key, value in report["proof_contract"].items():
        rows.append({"record_type": "proof", "record_id": key, "family": "analytic", "points": 0, "mismatches": 0, "value": json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value, "details": ""})
    return rows


def _write_csv(report: Mapping[str, Any]) -> None:
    rows = _source_rows(report)
    fields = ["record_type", "record_id", "family", "points", "mismatches", "value", "details"]
    with DEFAULT_SOURCE_DATA.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_gates(report: Mapping[str, Any]) -> dict[str, bool]:
    prereg = report["preregistration"]
    production = report["production_domain"]
    boundary = report["boundary_audit"]
    counter = {row["family"]: row for row in report["counterexamples"]}
    registry = report["comparator_registry"]
    proof = report["proof_contract"]
    bindings = report["bindings"]
    return {
        "G01_frozen_preregistration_is_consumed_exactly": prereg["experiment_id"] == "E6171_SINGLE_MODE_CPD_CI_EQUIVALENCE" and prereg["seed"] == BOUNDARY_SEED and prereg["exact_domain_points"] == PRODUCTION_POINTS and prereg["boundary_samples"] == BOUNDARY_SAMPLES and prereg["counterexample_families"] == ["biased", "correlated", "finite_energy_likelihood"] and prereg["sources"] == ["NOH2022_SURFACE_GKP", "LIN2023_CPD"] and [row["source_id"] for row in report["source_scope"]] == prereg["sources"],
        "G02_analytic_equivalence_scope_region_and_tie_are_explicit": proof["lattice"] == "Lambda=lambda*Z^2" and "no cross term" in proof["separability"] and "[(k_i-1/2)lambda,(k_i+1/2)lambda)" in proof["decision_region"] and len(proof["excluded"]) == 6,
        "G03_complete_production_q10_pair_domain_is_exhausted": production["adc_bits_per_axis"] == 10 and production["levels_per_axis"] == 1024 and production["points"] == 1_048_576 and sum(row["points"] for row in production["chunks"]) == production["points"] and len(production["chunks"]) == 32,
        "G04_production_cpd_ci_has_zero_mismatch_and_equal_hash": production["cpd_ci_mismatches"] == 0 and production["decision_mismatch_rate"] == 0.0 and 0.0 < production["zero_mismatch_one_sided_95_upper_bound"] < 3.0e-6 and production["ci_action_sha256"] == production["cpd_action_sha256"] and production["maximum_squared_distance_gap"] == 0.0,
        "G05_production_encoding_is_live_half_open_and_not_oversold": production["maximum_axis_decode_error_lattice_units"] <= 3.0e-16 and production["standard_binning_axis_mismatches_from_zero"] == 0 and production["canonical_cell_action_is_all_zero"] and production["nontrivial_alias_boundary_evidence_is_separate"] and production["positive_endpoint_excluded"],
        "G06_one_million_boundary_points_cover_ties_and_both_sides": boundary["seed"] == BOUNDARY_SEED and boundary["points"] == 1_000_000 and boundary["coordinates"] == 2_000_000 and len(boundary["mode_coordinate_counts"]) == 6 and min(boundary["mode_coordinate_counts"]) > 300_000 and boundary["exact_tie_coordinates"] > 300_000,
        "G07_boundary_cpd_ci_has_zero_mismatch_and_equal_hash": boundary["cpd_ci_mismatches"] == 0 and boundary["decision_mismatch_rate"] == 0.0 and 0.0 < boundary["zero_mismatch_one_sided_95_upper_bound"] < 3.1e-6 and boundary["ci_action_sha256"] == boundary["cpd_action_sha256"] and boundary["maximum_squared_distance_gap"] == 0.0,
        "G08_biased_likelihood_is_a_validated_ci_counterexample": counter["biased"]["is_counterexample"] and counter["biased"]["ci_class"] == 0 and counter["biased"]["likelihood_map_class"] == counter["biased"]["independent_brute"]["class"] == 1,
        "G09_correlated_joint_likelihood_is_a_validated_ci_counterexample": counter["correlated"]["is_counterexample"] and counter["correlated"]["mismatch_points"] > 0 and counter["correlated"]["likelihood_map_class"] == counter["correlated"]["independent_brute"]["class"] and counter["correlated"]["ci_class"] != counter["correlated"]["likelihood_map_class"],
        "G10_finite_energy_likelihood_is_a_validated_ci_counterexample": counter["finite_energy_likelihood"]["is_counterexample"] and counter["finite_energy_likelihood"]["mismatch_points"] > 0 and counter["finite_energy_likelihood"]["maximum_relative_reconstruction_error"] < 1.0e-13 and counter["finite_energy_likelihood"]["ci_class"] != counter["finite_energy_likelihood"]["likelihood_map_class"],
        "G11_counterexample_selection_is_fixed_grid_or_canonical_not_best_margin": counter["biased"]["selection_rule"].endswith("not outcome-tuned") and counter["correlated"]["selection_rule"].startswith("first lexicographic") and counter["finite_energy_likelihood"]["selection_rule"].startswith("first mismatch"),
        "G12_equivalent_cpd_ci_counts_once_and_map_remains_distinct": len(registry) == 3 and sum(row["ranking_weight"] for row in registry if row["equivalence_class"] == "square_isotropic_euclidean_nearest_lattice") == 1 and next(row for row in registry if row["method_id"] == "periodic_coset_map")["equivalence_class"] == "likelihood_coset_sum",
        "G13_claims_forbid_renamed_advantage_threshold_and_multimode_scope": report["claim_registry"] == {"EUCLIDEAN_CPD_EQUALS_CI": "ESTABLISHED_PROJECT_NATIVE_MATCHED", "CPD_IS_ADDITIONAL_WIN": "PROHIBITED_DUPLICATE", "CPD_EQUALS_COSET_MAP": "FALSIFIED_BY_THREE_FAMILIES", "SURFACE_GKP_THRESHOLD": "NOT_EVALUATED", "MULTIMODE_CPD": "NOT_EVALUATED"},
        "G14_runtime_memory_source_data_and_exact_bindings_are_present": report["execution_budget_audit"]["within_runtime_budget"] and report["execution_budget_audit"]["within_memory_budget"] and production["runtime_seconds"] > 0.0 and production["memory_bytes"] > 0 and boundary["runtime_seconds"] > 0.0 and boundary["memory_bytes"] > 0 and report["source_data"]["rows"] >= 50 and len(report["source_data"]["sha256"]) == 64 and set(bindings) == {"implementation", "preregistration_config", "ontology_initial", "source_audit_initial", "ideal_decoder", "finite_energy", "runtime_lut", "source_data"} and all(len(row["sha256"]) == 64 for row in bindings.values()) and all(_sha256(ROOT / row["path"]) == row["sha256"] for name, row in bindings.items() if name not in {"ontology_initial", "source_audit_initial"}) and _canonical_sha256(_ontology_semantic(_load(ONTOLOGY))) == report["ontology_semantic_sha256"] and _canonical_sha256(_source_semantic(_load(SOURCE_AUDIT))) == report["source_audit_semantic_sha256"],
        "G15_targeted_semantic_mutations_are_all_detected": report["semantic_mutation_audit"]["count"] == report["semantic_mutation_audit"]["detected"] == 15 and all(row["rejected"] for row in report["semantic_mutation_audit"]["cases"]),
    }


def _mutations(report: Mapping[str, Any]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []

    def attempt(name: str, gate: str, mutate: Any) -> None:
        candidate = deepcopy(report)
        candidate["semantic_mutation_audit"] = {"count": 15, "detected": 15, "cases": []}
        mutate(candidate)
        try:
            rejected = not evaluate_gates(candidate)[gate]
        except Exception:
            rejected = True
        cases.append({"case": name, "target_gate": gate, "rejected": rejected})

    attempt("change_preregistered_seed", "G01_frozen_preregistration_is_consumed_exactly", lambda x: x["preregistration"].update(seed=1))
    attempt("add_cross_term_to_proof", "G02_analytic_equivalence_scope_region_and_tie_are_explicit", lambda x: x["proof_contract"].update(separability="cross term ignored"))
    attempt("drop_production_chunk", "G03_complete_production_q10_pair_domain_is_exhausted", lambda x: x["production_domain"]["chunks"].pop())
    attempt("inject_production_mismatch", "G04_production_cpd_ci_has_zero_mismatch_and_equal_hash", lambda x: x["production_domain"].update(cpd_ci_mismatches=1))
    attempt("hide_canonical_cell_triviality", "G05_production_encoding_is_live_half_open_and_not_oversold", lambda x: x["production_domain"].update(nontrivial_alias_boundary_evidence_is_separate=False))
    attempt("shrink_boundary_set", "G06_one_million_boundary_points_cover_ties_and_both_sides", lambda x: x["boundary_audit"].update(points=10_000))
    attempt("inject_boundary_mismatch", "G07_boundary_cpd_ci_has_zero_mismatch_and_equal_hash", lambda x: x["boundary_audit"].update(cpd_ci_mismatches=1))
    attempt("erase_biased_counterexample", "G08_biased_likelihood_is_a_validated_ci_counterexample", lambda x: next(row for row in x["counterexamples"] if row["family"] == "biased").update(likelihood_map_class=0))
    attempt("forge_correlated_agreement", "G09_correlated_joint_likelihood_is_a_validated_ci_counterexample", lambda x: next(row for row in x["counterexamples"] if row["family"] == "correlated").update(ci_class=2))
    attempt("break_finite_energy_reference", "G10_finite_energy_likelihood_is_a_validated_ci_counterexample", lambda x: next(row for row in x["counterexamples"] if row["family"] == "finite_energy_likelihood").update(maximum_relative_reconstruction_error=0.1))
    attempt("select_best_margin_after_scan", "G11_counterexample_selection_is_fixed_grid_or_canonical_not_best_margin", lambda x: next(row for row in x["counterexamples"] if row["family"] == "correlated").update(selection_rule="largest favorable margin"))
    attempt("double_count_cpd", "G12_equivalent_cpd_ci_counts_once_and_map_remains_distinct", lambda x: next(row for row in x["comparator_registry"] if row["method_id"] == "square_euclidean_cpd").update(ranking_weight=1))
    attempt("invent_surface_threshold", "G13_claims_forbid_renamed_advantage_threshold_and_multimode_scope", lambda x: x["claim_registry"].update(SURFACE_GKP_THRESHOLD="ESTABLISHED"))
    attempt("truncate_source_hash", "G14_runtime_memory_source_data_and_exact_bindings_are_present", lambda x: x["bindings"]["source_data"].update(sha256="0"))
    attempt("forge_mutation_count", "G15_targeted_semantic_mutations_are_all_detected", lambda x: x.update(semantic_mutation_audit={"count": 15, "detected": 14, "cases": []}))
    return {"count": len(cases), "detected": sum(row["rejected"] for row in cases), "cases": cases}


def build_report() -> dict[str, Any]:
    frozen = _preregistered_experiment()
    production = _production_domain_audit()
    boundary = _boundary_audit()
    counter_start = perf_counter()
    counterexamples = _counterexamples()
    counter_runtime = perf_counter() - counter_start
    total_runtime = production["runtime_seconds"] + boundary["runtime_seconds"] + counter_runtime
    peak_memory = max(production["memory_bytes"], boundary["memory_bytes"])
    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "single-mode square/isotropic Euclidean closest point only; project-native mathematical/correctness evidence",
        "preregistration": {
            "experiment_id": frozen["experiment_id"],
            "record_sha256": _canonical_sha256(frozen),
            "seed": frozen["seeds"]["values"][0],
            "exact_domain_points": frozen["sample_size"]["exact_domain_points"],
            "boundary_samples": frozen["sample_size"]["boundary_samples"],
            "counterexample_families": frozen["config"]["counterexample_families"],
            "sources": frozen["sources"],
            "runtime_budget_seconds": frozen["runtime_budget"]["wall_clock_seconds"],
            "memory_budget_gib": frozen["runtime_budget"]["memory_gib"],
        },
        "proof_contract": _proof_contract(),
        "source_scope": _source_scope(frozen["sources"]),
        "production_domain": production,
        "boundary_audit": boundary,
        "counterexamples": counterexamples,
        "execution_budget_audit": {
            "measured_sections_runtime_seconds": total_runtime,
            "counterexample_search_runtime_seconds": counter_runtime,
            "runtime_budget_seconds": frozen["runtime_budget"]["wall_clock_seconds"],
            "peak_tracemalloc_bytes": peak_memory,
            "memory_budget_bytes": int(frozen["runtime_budget"]["memory_gib"] * (1 << 30)),
            "within_runtime_budget": total_runtime <= frozen["runtime_budget"]["wall_clock_seconds"],
            "within_memory_budget": peak_memory <= frozen["runtime_budget"]["memory_gib"] * (1 << 30),
            "boundary_note": "measured Python correctness workload only; not decoder or FPGA latency",
        },
        "comparator_registry": _comparator_registry(),
        "claim_registry": {
            "EUCLIDEAN_CPD_EQUALS_CI": "ESTABLISHED_PROJECT_NATIVE_MATCHED",
            "CPD_IS_ADDITIONAL_WIN": "PROHIBITED_DUPLICATE",
            "CPD_EQUALS_COSET_MAP": "FALSIFIED_BY_THREE_FAMILIES",
            "SURFACE_GKP_THRESHOLD": "NOT_EVALUATED",
            "MULTIMODE_CPD": "NOT_EVALUATED",
        },
        "allowed_wording": [
            "For a single-mode square lattice with an isotropic Euclidean metric, closest-point decoding is exactly componentwise closest-integer decoding under the stated tie rule.",
            "Biased, correlated and finite-energy likelihood examples show that this geometric equivalence does not collapse likelihood/coset MAP into CI.",
        ],
        "forbidden_wording": [
            "CPD is an additional decoder win over CI in the single-mode Euclidean row.",
            "The complete q10 audit reproduces a surface-GKP threshold or multimode CPD result.",
            "CI is equivalent to analog/weighted/coset MAP under arbitrary noise.",
        ],
    }
    _write_csv(report)
    report["source_data"] = {
        "path": _relative(DEFAULT_SOURCE_DATA),
        "sha256": _sha256(DEFAULT_SOURCE_DATA),
        "rows": sum(1 for _ in DEFAULT_SOURCE_DATA.open(encoding="utf-8")) - 1,
    }
    report["ontology_semantic_sha256"] = _canonical_sha256(_ontology_semantic(_load(ONTOLOGY)))
    report["source_audit_semantic_sha256"] = _canonical_sha256(_source_semantic(_load(SOURCE_AUDIT)))
    report["bindings"] = {
        "implementation": _binding(Path(__file__)),
        "preregistration_config": _binding(PREREG_CONFIG),
        "ontology_initial": _binding(ONTOLOGY),
        "source_audit_initial": _binding(SOURCE_AUDIT),
        "ideal_decoder": _binding(IDEAL_DECODER),
        "finite_energy": _binding(FINITE_ENERGY),
        "runtime_lut": _binding(RUNTIME_LUT),
        "source_data": _binding(DEFAULT_SOURCE_DATA),
    }
    report["semantic_mutation_audit"] = {"count": 15, "detected": 15, "cases": []}
    report["gates"] = evaluate_gates(report)
    report["semantic_mutation_audit"] = _mutations(report)
    report["gates"] = evaluate_gates(report)
    failed = [name for name, passed in report["gates"].items() if not passed]
    report["gate_summary"] = {"passed": len(report["gates"]) - len(failed), "failed": failed}
    report["verdict"] = VERDICT if not failed else "FAIL_SINGLE_MODE_CPD_EQUIVALENCE"
    return report


def verify_report(report: Mapping[str, Any]) -> None:
    gates = evaluate_gates(report)
    if dict(report["gates"]) != gates:
        raise ValueError("stored T6.17.1 gates do not match recomputation")
    failed = [name for name, passed in gates.items() if not passed]
    expected_summary = {"passed": len(gates) - len(failed), "failed": failed}
    expected_verdict = VERDICT if not failed else "FAIL_SINGLE_MODE_CPD_EQUIVALENCE"
    if report["gate_summary"] != expected_summary or report["verdict"] != expected_verdict:
        raise ValueError("stored T6.17.1 summary/verdict does not match recomputation")
    frozen = _preregistered_experiment()
    if _canonical_sha256(frozen) != report["preregistration"]["record_sha256"]:
        raise ValueError("T6.17.1 preregistration record drifted")
    if report["source_data"]["sha256"] != _sha256(ROOT / report["source_data"]["path"]):
        raise ValueError("T6.17.1 Source Data drifted")
    rows = sum(1 for _ in (ROOT / report["source_data"]["path"]).open(encoding="utf-8")) - 1
    if rows != report["source_data"]["rows"]:
        raise ValueError("T6.17.1 Source Data row count drifted")


def write_markdown(report: Mapping[str, Any], path: Path = DEFAULT_MARKDOWN) -> None:
    production = report["production_domain"]
    boundary = report["boundary_audit"]
    lines = [
        "# T6.17.1 single-mode Euclidean CPD 与 CI 等价边界",
        "",
        f"- verdict：`{report['verdict']}`",
        f"- production q10×q10：{production['points']:,} points，mismatch={production['cpd_ci_mismatches']}，runtime={production['runtime_seconds']:.3f} s",
        f"- boundary：{boundary['points']:,} points / {boundary['coordinates']:,} coordinates，exact-tie coordinates={boundary['exact_tie_coordinates']:,}，mismatch={boundary['cpd_ci_mismatches']}，runtime={boundary['runtime_seconds']:.3f} s",
        f"- gates/mutations：{report['gate_summary']['passed']}/15、{report['semantic_mutation_audit']['detected']}/15；Source Data={report['source_data']['rows']} rows",
        "",
        "## 结论",
        "",
        "对 `Λ=λZ²` 和 isotropic Euclidean metric，目标函数分解为两个独立平方项，因此 CPD 的 Voronoi cell 正是 CI 的半开区间笛卡尔积；正半边界按冻结规则归入较大整数。该结论只说明两个名称在这一行是重复 comparator，图表只能计一次。",
        "",
        "完整 production syndrome code 位于 centered canonical cell，所以 hard action 全为 00；这本身是平凡事实。非平凡的 alias/parity 与 tie 证据由独立的一百万点 unwrapped boundary audit 提供，不能把 canonical-cell 全零误写成广义 MAP 优势。",
        "",
        "## 不等价反例",
        "",
        "| family | selection | CI | likelihood MAP | witness |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in report["counterexamples"]:
        witness = row.get("syndrome_over_lattice")
        lines.append(f"| `{row['family']}` | {row['selection_rule']} | {row['ci_class']} | {row['likelihood_map_class']} | `{witness}` |")
    lines += [
        "",
        "biased mean 会交换 periodic coset mass；correlated covariance 的 Mahalanobis cross term 破坏坐标独立性；finite-energy state likelihood 又引入非均匀峰权重/收缩。三者都说明 weighted/analog/coset MAP 不能改名成 CPD 或 CI。",
        "",
        "## Claim 边界",
        "",
        "- established：single-mode square/isotropic Euclidean CPD = CI（project-native mathematical/correctness evidence）。",
        "- prohibited：把 CPD 与 CI 双计为两次胜场；声称 CPD=arbitrary MAP；从本实验推出 0.602 threshold、surface-GKP finite-size 或 multimode scaling。",
        "- 本实验没有 LER/SOTA/FPGA measured claim；runtime/memory 仅是 Python correctness audit 的测量边界。",
        "",
        "## 产物",
        "",
        "- `cnn_fpga/benchmark/single_mode_cpd_equivalence.py`",
        "- `docs/t6_17_1_single_mode_cpd_equivalence.json`",
        f"- `{report['source_data']['path']}`（{report['source_data']['rows']} rows）",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run T6.17.1 single-mode CPD/CI equivalence audit")
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    report = build_report()
    verify_report(report)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(report, args.markdown)
    print(json.dumps({"verdict": report["verdict"], "gate_summary": report["gate_summary"], "production_points": report["production_domain"]["points"], "boundary_points": report["boundary_audit"]["points"], "source_rows": report["source_data"]["rows"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
