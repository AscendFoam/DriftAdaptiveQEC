"""T6.17.2 independent reproduction of Noh et al. Table-I CNOT results.

This module implements the paper's eight-shift finite-squeezing model, its
componentwise closest-integer (CI) decoder and the correlated two-dimensional
maximum-likelihood (ML) Voronoi algorithms.  The reproduced object is one
error-corrected CNOT between two square-lattice GKP qubits.  It is not the
finite-size surface-GKP threshold experiment from the same paper.
"""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from math import floor, pi, sqrt
from pathlib import Path
from statistics import NormalDist
from time import perf_counter
import tracemalloc
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "T6.17.2"
SCHEMA_VERSION = "t6.17.2-noh-cnot-ci-ml-reproduction-v1"
VERDICT = "PASS_PROJECT_NATIVE_MATCHED_NOH_CNOT_CI_ML_REPRODUCTION"
FAIL_VERDICT = "NEGATIVE_ANCHOR_TOLERANCE_FAIL"
PREREG_CONFIG = ROOT / "configs" / "literature" / "t6_16_3_secondary_preregistration.json"
ONTOLOGY = ROOT / "docs" / "t6_16_2_comparison_ontology.json"
SOURCE_AUDIT = ROOT / "docs" / "t6_16_1_secondary_method_source_audit.json"
PRIMARY_READER = ROOT / "relative_papers" / "08_Low overhead fault-tolerant quantum error correction with the surface-GK.md"
DEFAULT_REPORT = ROOT / "docs" / "t6_17_2_noh_cnot_ci_ml_reproduction.json"
DEFAULT_SOURCE_DATA = ROOT / "docs" / "t6_17_2_noh_cnot_ci_ml_reproduction_source_data.csv"
DEFAULT_MARKDOWN = ROOT / "docs" / "noh_cnot_ci_ml_reproduction.md"

LATTICE = sqrt(pi)
LAMBDA = 1.0
SEED_BATCH = 2048
BRUTE_RADIUS = 2
BRUTE_OFFSETS = np.asarray(
    [(i, j) for i in range(-BRUTE_RADIUS, BRUTE_RADIUS + 1) for j in range(-BRUTE_RADIUS, BRUTE_RADIUS + 1)],
    dtype=np.int64,
)
CONFIDENCE = 0.95
BOOTSTRAP_RESAMPLES = 20_000
BOUNDARY_SEED = 61_729_001


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


def _preregistered_experiment() -> dict[str, Any]:
    rows = [row for row in _load(PREREG_CONFIG)["experiments"] if row["task_id"] == TASK_ID]
    if len(rows) != 1:
        raise ValueError("T6.17.2 requires exactly one frozen preregistration row")
    return rows[0]


def _source_record() -> dict[str, Any]:
    rows = [row for row in _load(SOURCE_AUDIT)["sources"] if row["source_id"] == "NOH2022_SURFACE_GKP"]
    if len(rows) != 1:
        raise ValueError("Noh 2022 primary source record is missing or duplicated")
    return rows[0]


def source_sufficiency_contract() -> dict[str, Any]:
    """Freeze the primary-paper ingredients needed for an exact gate model."""
    ingredients = [
        {"ingredient": "finite_squeezing_variance", "locator": "Eq. (24)", "content": "sigma_gkp^2=(1/2)*10^(-squeezing_dB/10)"},
        {"ingredient": "eight_iid_primitive_shifts", "locator": "Appendix C1d, Eq. (C19)", "content": "xi_1,...,xi_8 iid N(0,sigma_gkp^2)"},
        {"ingredient": "q_net_shifts_lambda_1", "locator": "Eq. (25); Eq. (C19)", "content": "q1=xi1+xi2; q2=xi1+xi3+xi4"},
        {"ingredient": "p_net_shifts_lambda_1", "locator": "Eq. (25); Eq. (C19)", "content": "p1=-xi5+xi6+xi7; p2=xi5+xi8"},
        {"ingredient": "closest_integer_rule", "locator": "Eq. (27)", "content": "n=floor(shift/lattice_spacing+1/2) componentwise"},
        {"ingredient": "q_joint_ml", "locator": "Eqs. (29)-(30); Algorithm 1", "content": "min 3*x1^2+2*x2^2-2*x1*x2 at lambda=1"},
        {"ingredient": "p_joint_ml", "locator": "Eqs. (31)-(33); Algorithm 2", "content": "min 2*x1^2+3*x2^2+2*x1*x2 at lambda=1"},
        {"ingredient": "logical_failure_mapping", "locator": "Appendix C, Eqs. (C20)-(C21)", "content": "failure iff any of nq1,nq2,np1,np2 is odd"},
        {"ingredient": "anchors", "locator": "Table I", "content": "CI/ML failure at 9,12,13 dB"},
    ]
    return {
        "status": "PASS_SOURCE_SUFFICIENCY",
        "primary_source_id": "NOH2022_SURFACE_GKP",
        "paper_version": "PRX Quantum 3, 010315 / arXiv:2103.06994",
        "local_primary_reader": _relative(PRIMARY_READER),
        "ingredients": ingredients,
        "complete_ingredients": len(ingredients),
        "missing_ingredients": [],
        "model_scope": "one error-corrected CNOT between two square-lattice GKP qubits, lambda=1, finite-squeezing-only noise",
        "explicit_exclusions": [
            "photon loss and other physical channels",
            "surface-code lattice, matching graph and finite-size scaling",
            "the 9.9 dB outer surface-GKP threshold",
            "hardware or closed-loop decoder latency",
        ],
    }


def squeezing_variance(squeezing_db: float) -> float:
    if not np.isfinite(squeezing_db):
        raise ValueError("squeezing_db must be finite")
    return float(0.5 * 10.0 ** (-float(squeezing_db) / 10.0))


def _round_lattice(values: ArrayLike, spacing: float = LATTICE) -> NDArray[np.int64]:
    x = np.asarray(values, dtype=np.float64)
    if spacing <= 0.0 or not np.isfinite(spacing) or not np.all(np.isfinite(x)):
        raise ValueError("finite values and positive finite spacing are required")
    normalized = x / spacing
    if np.any(normalized <= -(2**61)) or np.any(normalized >= 2**61):
        raise ValueError("lattice coordinate exceeds exact int64 audit range")
    base = np.floor(normalized)
    fraction = normalized - base
    return (base + (fraction >= 0.5)).astype(np.int64)


def closest_integer_decode(shifts: ArrayLike) -> NDArray[np.int64]:
    values = np.asarray(shifts, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != 2:
        raise ValueError("shift pairs must have shape (...,2)")
    return _round_lattice(values)


def q_ml_decode(shifts: ArrayLike) -> NDArray[np.int64]:
    """Paper Algorithm 1, specialized to the frozen square lattice."""
    values = np.asarray(shifts, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or not np.all(np.isfinite(values)):
        raise ValueError("q shifts must be a finite (n,2) array")
    q1, q2 = values[:, 0], values[:, 1]
    b1, b2 = _round_lattice(q1), _round_lattice(q2)
    r1 = q1 - b1 * LATTICE
    v1 = 0.5 * r1 - 0.5 * LATTICE
    v2 = q2 - b2 * LATTICE
    v3 = 0.5 * r1 + 0.5 * LATTICE
    v4 = 3.0 * r1 - 1.5 * LATTICE
    v5 = 3.0 * r1 + 1.5 * LATTICE
    v6 = -2.0 * (r1 + LATTICE) + 1.5 * LATTICE
    v7 = -2.0 * (r1 - LATTICE) - 1.5 * LATTICE
    out = np.stack((b1, b2), axis=1)
    middle = (v1 < v2) & (v2 < v3)
    upper = (~middle) & (v2 >= v3)
    lower = ~(middle | upper)
    mask = middle & (v2 >= v5)
    out[mask, 0] = b1[mask] - 1
    mask = middle & (v2 < v5) & ~(v4 < v2)
    out[mask, 0] = b1[mask] + 1
    mask = upper & (v2 > v6)
    out[mask, 1] = b2[mask] + 1
    mask = upper & ~(v2 > v6)
    out[mask, 0] = b1[mask] - 1
    mask = lower & (v2 > v7)
    out[mask, 0] = b1[mask] + 1
    mask = lower & ~(v2 > v7)
    out[mask, 1] = b2[mask] - 1
    return out


def p_ml_decode(shifts: ArrayLike) -> NDArray[np.int64]:
    """Paper Algorithm 2, specialized to the frozen square lattice."""
    values = np.asarray(shifts, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or not np.all(np.isfinite(values)):
        raise ValueError("p shifts must be a finite (n,2) array")
    p1, p2 = values[:, 0], values[:, 1]
    b1, b2 = _round_lattice(p1), _round_lattice(p2)
    r1 = p1 - b1 * LATTICE
    v1 = -r1 / 3.0 - 0.5 * LATTICE
    v2 = p2 - b2 * LATTICE
    v3 = -r1 / 3.0 + 0.5 * LATTICE
    v4 = -2.0 * r1 - LATTICE
    v5 = -2.0 * r1 + LATTICE
    v6 = 0.5 * (r1 - LATTICE) + 0.75 * LATTICE
    v7 = 0.5 * (r1 + LATTICE) - 0.75 * LATTICE
    out = np.stack((b1, b2), axis=1)
    middle = (v1 < v2) & (v2 < v3)
    upper = (~middle) & (v2 >= v3)
    lower = ~(middle | upper)
    mask = middle & (v2 >= v5)
    out[mask, 0] = b1[mask] + 1
    mask = middle & (v2 < v5) & ~(v4 < v2)
    out[mask, 0] = b1[mask] - 1
    mask = upper & (v2 > v6)
    out[mask, 1] = b2[mask] + 1
    mask = upper & ~(v2 > v6)
    out[mask, 0] = b1[mask] + 1
    mask = lower & (v2 > v7)
    out[mask, 0] = b1[mask] - 1
    mask = lower & ~(v2 > v7)
    out[mask, 1] = b2[mask] - 1
    return out


def brute_likelihood_decode(shifts: ArrayLike, quadrature: str) -> NDArray[np.int64]:
    """Independent 25-candidate likelihood oracle for the lambda=1 model."""
    values = np.asarray(shifts, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or not np.all(np.isfinite(values)):
        raise ValueError("shifts must be a finite (n,2) array")
    if quadrature not in {"q", "p"}:
        raise ValueError("quadrature must be 'q' or 'p'")
    base = closest_integer_decode(values)
    candidates = base[:, None, :] + BRUTE_OFFSETS[None, :, :]
    residual = values[:, None, :] - candidates.astype(np.float64) * LATTICE
    x1, x2 = residual[:, :, 0], residual[:, :, 1]
    if quadrature == "q":
        cost = 3.0 * x1 * x1 + 2.0 * x2 * x2 - 2.0 * x1 * x2
    else:
        cost = 2.0 * x1 * x1 + 3.0 * x2 * x2 + 2.0 * x1 * x2
    return candidates[np.arange(values.shape[0]), np.argmin(cost, axis=1)]


def _failure(indices_q: NDArray[np.int64], indices_p: NDArray[np.int64]) -> NDArray[np.bool_]:
    indices = np.concatenate((indices_q, indices_p), axis=1)
    return np.any(np.mod(indices, 2) != 0, axis=1)


def net_cnot_shifts(standard_normal_draws: ArrayLike, sigma_gkp: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply Appendix-C Eq. (C19) to eight standard-normal draws."""
    standard = np.asarray(standard_normal_draws, dtype=np.float64)
    if standard.ndim != 2 or standard.shape[1] != 8 or not np.all(np.isfinite(standard)):
        raise ValueError("primitive draws must be a finite (n,8) array")
    if sigma_gkp <= 0.0 or not np.isfinite(sigma_gkp):
        raise ValueError("sigma_gkp must be positive and finite")
    xi = float(sigma_gkp) * standard
    q = np.stack((xi[:, 0] + xi[:, 1], xi[:, 0] + xi[:, 2] + xi[:, 3]), axis=1)
    p = np.stack((-xi[:, 4] + xi[:, 5] + xi[:, 6], xi[:, 4] + xi[:, 7]), axis=1)
    return q, p


def _wilson(failures: int, trials: int, confidence: float = CONFIDENCE) -> list[float]:
    if trials <= 0 or not 0 <= failures <= trials:
        raise ValueError("invalid binomial count")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    p = failures / trials
    den = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / den
    half = z * sqrt((p * (1.0 - p) + z * z / (4.0 * trials)) / trials) / den
    return [max(0.0, center - half), min(1.0, center + half)]


def _cluster_bootstrap(clusters: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    trials = np.asarray([row["trials"] for row in clusters], dtype=np.int64)
    ci = np.asarray([row["ci_failures"] for row in clusters], dtype=np.int64)
    ml = np.asarray([row["ml_failures"] for row in clusters], dtype=np.int64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(clusters), size=(BOOTSTRAP_RESAMPLES, len(clusters)))
    denominator = trials[indices].sum(axis=1)
    difference = (ci[indices].sum(axis=1) - ml[indices].sum(axis=1)) / denominator
    reduction = 1.0 - ml[indices].sum(axis=1) / ci[indices].sum(axis=1)
    return {
        "resamples": BOOTSTRAP_RESAMPLES,
        "unit": "independent_seed_cluster",
        "seed": seed,
        "ci_minus_ml_95_interval": np.quantile(difference, [0.025, 0.975]).tolist(),
        "relative_reduction_95_interval": np.quantile(reduction, [0.025, 0.975]).tolist(),
    }


def simulate_point(
    squeezing_db: float,
    seeds: list[int],
    target_failures: int,
    n_max: int,
    seed_batch: int = SEED_BATCH,
) -> dict[str, Any]:
    if len(seeds) == 0 or len(set(seeds)) != len(seeds):
        raise ValueError("nonempty unique seed list required")
    if target_failures <= 0 or n_max <= 0 or seed_batch <= 0:
        raise ValueError("positive stopping parameters required")
    rngs = [np.random.default_rng(seed) for seed in seeds]
    clusters = [
        {"seed": seed, "trials": 0, "ci_failures": 0, "ml_failures": 0, "both_fail": 0, "ci_only": 0, "ml_only": 0}
        for seed in seeds
    ]
    ci_hash = hashlib.sha256()
    ml_hash = hashlib.sha256()
    trials = ci_failures = ml_failures = both_fail = ci_only = ml_only = 0
    sigma = sqrt(squeezing_variance(squeezing_db))
    start = perf_counter()
    tracemalloc.start()
    tracemalloc.reset_peak()
    completed_sweeps = 0
    while trials < n_max and (ci_failures < target_failures or ml_failures < target_failures):
        for rng, cluster in zip(rngs, clusters, strict=True):
            remaining = n_max - trials
            if remaining <= 0:
                break
            size = min(seed_batch, remaining)
            standard = rng.standard_normal((size, 8))
            primitive_bytes = standard.tobytes()
            ci_hash.update(primitive_bytes)
            ml_hash.update(primitive_bytes)
            q, p = net_cnot_shifts(standard, sigma)
            ci = _failure(closest_integer_decode(q), closest_integer_decode(p))
            ml = _failure(q_ml_decode(q), p_ml_decode(p))
            local_ci = int(np.count_nonzero(ci))
            local_ml = int(np.count_nonzero(ml))
            local_both = int(np.count_nonzero(ci & ml))
            local_ci_only = int(np.count_nonzero(ci & ~ml))
            local_ml_only = int(np.count_nonzero(~ci & ml))
            cluster["trials"] += size
            cluster["ci_failures"] += local_ci
            cluster["ml_failures"] += local_ml
            cluster["both_fail"] += local_both
            cluster["ci_only"] += local_ci_only
            cluster["ml_only"] += local_ml_only
            trials += size
            ci_failures += local_ci
            ml_failures += local_ml
            both_fail += local_both
            ci_only += local_ci_only
            ml_only += local_ml_only
        completed_sweeps += 1
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    runtime = perf_counter() - start
    stop_reason = "TARGET_FAILURES_BOTH_METHODS" if ci_failures >= target_failures and ml_failures >= target_failures else "N_MAX"
    p_ci = ci_failures / trials
    p_ml = ml_failures / trials
    discordant = ci_only + ml_only
    mcnemar = float(binomtest(ci_only, discordant, p=0.5, alternative="two-sided").pvalue) if discordant else 1.0
    bootstrap_seed = int(seeds[-1] ^ int(round(squeezing_db * 1000.0)))
    return {
        "squeezing_db": squeezing_db,
        "sigma_gkp_squared": sigma * sigma,
        "sigma_gkp": sigma,
        "trials": trials,
        "completed_seed_sweeps": completed_sweeps,
        "seed_batch": seed_batch,
        "stop_reason": stop_reason,
        "target_failures_per_method": target_failures,
        "n_max": n_max,
        "ci_failures": ci_failures,
        "ml_failures": ml_failures,
        "both_fail": both_fail,
        "ci_only": ci_only,
        "ml_only": ml_only,
        "neither": trials - both_fail - ci_only - ml_only,
        "ci_probability": p_ci,
        "ml_probability": p_ml,
        "ci_wilson_95": _wilson(ci_failures, trials),
        "ml_wilson_95": _wilson(ml_failures, trials),
        "paired_difference": p_ci - p_ml,
        "relative_failure_reduction": 1.0 - p_ml / p_ci,
        "mcnemar_exact_two_sided_p": mcnemar,
        "cluster_bootstrap": _cluster_bootstrap(clusters, bootstrap_seed),
        "clusters": clusters,
        "ci_primitive_draw_sha256": ci_hash.hexdigest(),
        "ml_primitive_draw_sha256": ml_hash.hexdigest(),
        "runtime_seconds": runtime,
        "peak_tracemalloc_bytes": peak_bytes,
    }


def _holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(np.asarray(p_values))
    adjusted = np.empty(len(p_values), dtype=np.float64)
    running = 0.0
    m = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p_values[int(index)])
        adjusted[int(index)] = min(1.0, running)
    return adjusted.tolist()


def _boundary_audit(samples: int) -> dict[str, Any]:
    if samples < 100_000 or samples % 2:
        raise ValueError("boundary audit requires an even sample count of at least 100000")
    rng = np.random.default_rng(BOUNDARY_SEED)
    pairs_needed = samples // 2
    epsilon = LATTICE * 2.0 ** -30
    chunks: list[dict[str, Any]] = []
    total_mismatch = total_pairs = generated = crossing_pairs = 0
    max_cost_gap = 0.0
    production_hash = hashlib.sha256()
    brute_hash = hashlib.sha256()
    start = perf_counter()
    tracemalloc.start()
    tracemalloc.reset_peak()
    for kind, matrix, directions in (
        ("q", np.asarray([[3.0, -1.0], [-1.0, 2.0]]), np.asarray([[1, 0], [0, 1], [1, 1]])),
        ("p", np.asarray([[2.0, 1.0], [1.0, 3.0]]), np.asarray([[1, 0], [0, 1], [1, -1]])),
    ):
        kind_pairs = pairs_needed // 2
        accepted: list[NDArray[np.float64]] = []
        accepted_pairs = 0
        attempts = 0
        while accepted_pairs < kind_pairs:
            batch_pairs = min(20_000, max(2048, kind_pairs - accepted_pairs))
            cells = rng.integers(-64, 65, size=(batch_pairs, 2))
            direction = directions[rng.integers(0, len(directions), size=batch_pairs)].astype(np.float64)
            normal = direction @ matrix
            tangent = np.stack((-normal[:, 1], normal[:, 0]), axis=1)
            tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
            midpoint = (cells + 0.5 * direction) * LATTICE
            on_plane = midpoint + tangent * rng.uniform(-0.22 * LATTICE, 0.22 * LATTICE, size=(batch_pairs, 1))
            unit_normal = normal / np.linalg.norm(normal, axis=1, keepdims=True)
            minus = on_plane - epsilon * unit_normal
            plus = on_plane + epsilon * unit_normal
            brute_minus = brute_likelihood_decode(minus, kind)
            brute_plus = brute_likelihood_decode(plus, kind)
            crosses = np.any(brute_minus != brute_plus, axis=1)
            if np.any(crosses):
                pair_points = np.stack((minus[crosses], plus[crosses]), axis=1)
                take = min(kind_pairs - accepted_pairs, pair_points.shape[0])
                accepted.append(pair_points[:take])
                accepted_pairs += take
            attempts += batch_pairs
            if attempts > 20 * kind_pairs:
                raise RuntimeError("failed to construct enough actual ML boundary pairs")
        points = np.concatenate(accepted, axis=0).reshape(-1, 2)
        production = q_ml_decode(points) if kind == "q" else p_ml_decode(points)
        brute = brute_likelihood_decode(points, kind)
        mismatch = int(np.count_nonzero(np.any(production != brute, axis=1)))
        residual_prod = points - production * LATTICE
        residual_brute = points - brute * LATTICE
        prod_cost = np.einsum("ni,ij,nj->n", residual_prod, matrix, residual_prod)
        brute_cost = np.einsum("ni,ij,nj->n", residual_brute, matrix, residual_brute)
        cost_gap = float(np.max(np.abs(prod_cost - brute_cost)))
        max_cost_gap = max(max_cost_gap, cost_gap)
        production_hash.update(production.tobytes())
        brute_hash.update(brute.tobytes())
        local_crossing = int(np.count_nonzero(np.any(production[0::2] != production[1::2], axis=1)))
        chunks.append({
            "quadrature": kind,
            "points": int(points.shape[0]),
            "boundary_pairs": accepted_pairs,
            "candidate_attempt_pairs": attempts,
            "production_brute_mismatches": mismatch,
            "one_sided_crossing_pairs": local_crossing,
            "maximum_likelihood_cost_gap": cost_gap,
        })
        total_mismatch += mismatch
        total_pairs += accepted_pairs
        generated += points.shape[0]
        crossing_pairs += local_crossing
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "seed": BOUNDARY_SEED,
        "points": generated,
        "boundary_pairs": total_pairs,
        "epsilon_lattice_units": epsilon / LATTICE,
        "facet_tangent_span_lattice_units": 0.22,
        "exact_ties_excluded_due_nonunique_argmin": True,
        "all_pairs_cross_decision_boundary": crossing_pairs == total_pairs,
        "one_sided_crossing_pairs": crossing_pairs,
        "production_brute_mismatches": total_mismatch,
        "decision_mismatch_rate": total_mismatch / generated,
        "maximum_likelihood_cost_gap": max_cost_gap,
        "production_action_sha256": production_hash.hexdigest(),
        "brute_action_sha256": brute_hash.hexdigest(),
        "brute_candidate_count_per_pair": int(BRUTE_OFFSETS.shape[0]),
        "brute_radius": BRUTE_RADIUS,
        "enclosure_proof": "lambda_min(A)*2.5^2 > lambda_max(A)*0.5 proves candidates outside radius 2 cannot beat componentwise CI base for A eigenvalues (5+-sqrt(5))/2",
        "chunks": chunks,
        "runtime_seconds": perf_counter() - start,
        "peak_tracemalloc_bytes": peak_bytes,
    }


def _anchor_rows(points: list[dict[str, Any]], frozen: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, point in enumerate(points):
        absolute_tolerance = float(frozen["config"]["absolute_tolerance"][index])
        for method in ("ci", "ml"):
            anchor = float(frozen["config"][f"{method}_anchors"][index])
            estimate = float(point[f"{method}_probability"])
            absolute = abs(estimate - anchor)
            relative = absolute / anchor
            rows.append({
                "squeezing_db": point["squeezing_db"],
                "method": method.upper(),
                "literature_anchor": anchor,
                "estimate": estimate,
                "absolute_discrepancy": absolute,
                "relative_discrepancy": relative,
                "absolute_tolerance": absolute_tolerance,
                "relative_tolerance": frozen["config"]["relative_tolerance"],
                "absolute_pass": absolute <= absolute_tolerance,
                "relative_pass": relative <= frozen["config"]["relative_tolerance"],
                "joint_pass": absolute <= absolute_tolerance and relative <= frozen["config"]["relative_tolerance"],
            })
    return rows


def _source_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ingredient in report["source_sufficiency"]["ingredients"]:
        rows.append({"record_type": "source", "squeezing_db": "", "seed": "", "method": ingredient["ingredient"], "trials": 0, "failures": 0, "probability": "", "ci_low": "", "ci_high": "", "details": f"{ingredient['locator']}: {ingredient['content']}"})
    for point in report["points"]:
        for cluster in point["clusters"]:
            rows.append({"record_type": "seed_cluster", "squeezing_db": point["squeezing_db"], "seed": cluster["seed"], "method": "PAIRED_CI_ML", "trials": cluster["trials"], "failures": f"{cluster['ci_failures']}/{cluster['ml_failures']}", "probability": "", "ci_low": "", "ci_high": "", "details": json.dumps({key: cluster[key] for key in ("both_fail", "ci_only", "ml_only")}, sort_keys=True)})
        for method in ("ci", "ml"):
            interval = point[f"{method}_wilson_95"]
            rows.append({"record_type": "point_summary", "squeezing_db": point["squeezing_db"], "seed": "all", "method": method.upper(), "trials": point["trials"], "failures": point[f"{method}_failures"], "probability": point[f"{method}_probability"], "ci_low": interval[0], "ci_high": interval[1], "details": point["stop_reason"]})
    for row in report["anchor_discrepancies"]:
        rows.append({"record_type": "anchor", "squeezing_db": row["squeezing_db"], "seed": "", "method": row["method"], "trials": 0, "failures": 0, "probability": row["estimate"], "ci_low": "", "ci_high": "", "details": json.dumps(row, sort_keys=True)})
    for chunk in report["boundary_audit"]["chunks"]:
        rows.append({"record_type": "boundary", "squeezing_db": "", "seed": BOUNDARY_SEED, "method": chunk["quadrature"].upper(), "trials": chunk["points"], "failures": chunk["production_brute_mismatches"], "probability": 0.0, "ci_low": "", "ci_high": "", "details": json.dumps(chunk, sort_keys=True)})
    return rows


def _write_csv(report: Mapping[str, Any]) -> None:
    fields = ["record_type", "squeezing_db", "seed", "method", "trials", "failures", "probability", "ci_low", "ci_high", "details"]
    with DEFAULT_SOURCE_DATA.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(_source_rows(report))


def evaluate_gates(report: Mapping[str, Any]) -> dict[str, bool]:
    frozen = _preregistered_experiment()
    prereg = report["preregistration"]
    source = report["source_sufficiency"]
    points = report["points"]
    boundary = report["boundary_audit"]
    bindings = report["bindings"]
    raw_trials = sum(point["trials"] for point in points)
    expected_anchor_keys = {
        (float(db), method.upper()): (float(anchor), float(absolute_tolerance))
        for method in ("ci", "ml")
        for db, anchor, absolute_tolerance in zip(
            frozen["config"]["squeezing_db"],
            frozen["config"][f"{method}_anchors"],
            frozen["config"]["absolute_tolerance"],
            strict=True,
        )
    }
    anchor_recomputed = True
    for row in report["anchor_discrepancies"]:
        key = (float(row["squeezing_db"]), row["method"])
        if key not in expected_anchor_keys:
            anchor_recomputed = False
            continue
        expected_anchor, expected_absolute_tolerance = expected_anchor_keys[key]
        absolute = abs(float(row["estimate"]) - expected_anchor)
        relative = absolute / expected_anchor
        absolute_pass = absolute <= expected_absolute_tolerance
        relative_pass = relative <= float(frozen["config"]["relative_tolerance"])
        anchor_recomputed &= (
            row["literature_anchor"] == expected_anchor
            and row["absolute_tolerance"] == expected_absolute_tolerance
            and row["relative_tolerance"] == frozen["config"]["relative_tolerance"]
            and np.isclose(row["absolute_discrepancy"], absolute, rtol=0.0, atol=1e-18)
            and np.isclose(row["relative_discrepancy"], relative, rtol=0.0, atol=1e-16)
            and row["absolute_pass"] == absolute_pass
            and row["relative_pass"] == relative_pass
            and row["joint_pass"] == (absolute_pass and relative_pass)
        )
    expected_raw_p = [
        float(binomtest(row["ci_only"], row["ci_only"] + row["ml_only"], p=0.5, alternative="two-sided").pvalue)
        if row["ci_only"] + row["ml_only"] else 1.0
        for row in points
    ]
    expected_holm = _holm_adjust(expected_raw_p)
    return {
        "G01_frozen_preregistration_and_primary_source_are_consumed_exactly": prereg["experiment_id"] == "E6172_NOH_CNOT_CI_ML_REPRODUCTION" and prereg["record_sha256"] == _canonical_sha256(_preregistered_experiment()) and prereg["source_record_sha256"] == _canonical_sha256(_source_record()) and prereg["seeds"] == list(range(61720001, 61720033)),
        "G02_source_sufficiency_covers_noise_circuit_decoders_and_failure_mapping": source["status"] == "PASS_SOURCE_SUFFICIENCY" and source["complete_ingredients"] == len(source["ingredients"]) == 9 and not source["missing_ingredients"] and len(source["explicit_exclusions"]) == 4,
        "G03_frozen_square_lattice_variance_and_three_anchor_points_are_exact": report["model_contract"] == {"lambda": 1.0, "lattice_spacing": "sqrt(pi)", "primitive_shift_count": 8, "primitive_distribution": "iid N(0,sigma_gkp^2)", "sigma_gkp_squared": "0.5*10^(-squeezing_dB/10)", "q_covariance_over_sigma_squared": [[2.0, 1.0], [1.0, 3.0]], "p_covariance_over_sigma_squared": [[3.0, -1.0], [-1.0, 2.0]], "logical_failure": "any decoded integer odd"} and [row["squeezing_db"] for row in points] == [9.0, 12.0, 13.0] and all(np.isclose(row["sigma_gkp_squared"], squeezing_variance(row["squeezing_db"]), rtol=2e-15, atol=0.0) for row in points),
        "G04_every_point_obeys_failure_target_or_nmax_stopping": len(points) == 3 and all(row["trials"] <= row["n_max"] and ((row["stop_reason"] == "TARGET_FAILURES_BOTH_METHODS" and min(row["ci_failures"], row["ml_failures"]) >= row["target_failures_per_method"]) or (row["stop_reason"] == "N_MAX" and row["trials"] == row["n_max"])) for row in points),
        "G05_common_random_numbers_and_seed_clusters_are_complete": all(row["ci_primitive_draw_sha256"] == row["ml_primitive_draw_sha256"] and len(row["clusters"]) == 32 and sum(cluster["trials"] for cluster in row["clusters"]) == row["trials"] for row in points),
        "G06_raw_paired_counts_are_internally_conservative": all(row["both_fail"] + row["ci_only"] == row["ci_failures"] and row["both_fail"] + row["ml_only"] == row["ml_failures"] and row["both_fail"] + row["ci_only"] + row["ml_only"] + row["neither"] == row["trials"] and sum(cluster["ci_failures"] for cluster in row["clusters"]) == row["ci_failures"] and sum(cluster["ml_failures"] for cluster in row["clusters"]) == row["ml_failures"] and np.isclose(row["ci_probability"], row["ci_failures"] / row["trials"]) and np.isclose(row["ml_probability"], row["ml_failures"] / row["trials"]) and np.isclose(row["paired_difference"], (row["ci_only"] - row["ml_only"]) / row["trials"]) and np.isclose(row["relative_failure_reduction"], 1.0 - row["ml_failures"] / row["ci_failures"]) for row in points),
        "G07_all_six_table_i_anchor_discrepancies_pass_frozen_dual_tolerance": len(report["anchor_discrepancies"]) == len(expected_anchor_keys) == 6 and set((float(row["squeezing_db"]), row["method"]) for row in report["anchor_discrepancies"]) == set(expected_anchor_keys) and anchor_recomputed and all(row["joint_pass"] for row in report["anchor_discrepancies"]),
        "G08_ml_reduction_is_paired_significant_at_all_points_after_holm": len(report["multiplicity"]["holm_adjusted_mcnemar_p"]) == 3 and np.allclose(report["multiplicity"]["raw_mcnemar_p"], expected_raw_p, rtol=0.0, atol=0.0) and np.allclose(report["multiplicity"]["holm_adjusted_mcnemar_p"], expected_holm, rtol=0.0, atol=0.0) and all(np.isclose(row["mcnemar_exact_two_sided_p"], expected_raw_p[index], rtol=0.0, atol=0.0) and row["paired_difference"] > 0.0 and row["cluster_bootstrap"]["ci_minus_ml_95_interval"][0] > 0.0 for index, row in enumerate(points)) and all(value < 0.05 for value in expected_holm),
        "G09_wilson_and_cluster_bootstrap_statistics_are_complete": all(np.allclose(row["ci_wilson_95"], _wilson(row["ci_failures"], row["trials"]), rtol=0.0, atol=1e-18) and np.allclose(row["ml_wilson_95"], _wilson(row["ml_failures"], row["trials"]), rtol=0.0, atol=1e-18) and 0.0 <= row["ci_wilson_95"][0] <= row["ci_probability"] <= row["ci_wilson_95"][1] <= 1.0 and 0.0 <= row["ml_wilson_95"][0] <= row["ml_probability"] <= row["ml_wilson_95"][1] <= 1.0 and row["cluster_bootstrap"]["resamples"] == 20_000 and row["cluster_bootstrap"]["unit"] == "independent_seed_cluster" for row in points),
        "G10_boundary_likelihood_oracle_has_100k_actual_crossings_and_zero_mismatch": boundary["points"] == 100_000 and boundary["boundary_pairs"] == 50_000 and boundary["all_pairs_cross_decision_boundary"] and boundary["production_brute_mismatches"] == 0 and boundary["decision_mismatch_rate"] == 0.0 and boundary["production_action_sha256"] == boundary["brute_action_sha256"],
        "G11_independent_brute_enclosure_and_likelihood_optimality_are_explicit": boundary["brute_radius"] == 2 and boundary["brute_candidate_count_per_pair"] == 25 and "lambda_min" in boundary["enclosure_proof"] and boundary["maximum_likelihood_cost_gap"] <= 1e-12,
        "G12_operation_cost_is_fixed_and_not_mislabeled_as_latency": report["operation_cost"]["ci"]["correlated_2d_classifiers"] == 0 and report["operation_cost"]["ml"]["correlated_2d_classifiers"] == 2 and report["operation_cost"]["ml"]["asymptotic_per_gate"] == "O(1)" and report["operation_cost"]["measured_latency_ns"] is None and "not latency" in report["operation_cost"]["python_runtime_boundary"],
        "G13_outer_threshold_and_hardware_claims_remain_literature_only_or_null": report["claim_registry"] == {"NOH_TABLE_I_GATE_REPRODUCTION": "PROJECT_NATIVE_MATCHED", "NOH_9P9DB_OUTER_THRESHOLD": "LITERATURE_ONLY_NULL", "CI_LT50NS": "NULL_UNSOURCED", "ML_GT1MS": "NULL_UNSOURCED", "HARDWARE_RESOURCES": "NULL_NOT_IMPLEMENTED", "PHASE6B_V5_VERDICT": "READ_ONLY_NO_GO_UNCHANGED"},
        "G14_runtime_memory_raw_source_data_and_live_bindings_are_valid": report["execution_budget_audit"]["within_runtime_budget"] and report["execution_budget_audit"]["within_memory_budget"] and raw_trials > 0 and report["source_data"]["rows"] >= 100 and report["source_data"]["sha256"] == _sha256(ROOT / report["source_data"]["path"]) and set(bindings) == {"implementation", "preregistration_config", "ontology_initial", "source_audit_initial", "primary_reader", "source_data"} and bindings["source_data"]["sha256"] == report["source_data"]["sha256"] and all(_sha256(ROOT / row["path"]) == row["sha256"] for name, row in bindings.items() if name not in {"ontology_initial", "source_audit_initial"}) and _canonical_sha256(_ontology_semantic(_load(ONTOLOGY))) == report["ontology_semantic_sha256"] and _canonical_sha256(_source_semantic(_load(SOURCE_AUDIT))) == report["source_audit_semantic_sha256"],
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

    attempt("change_preregistration_hash", "G01_frozen_preregistration_and_primary_source_are_consumed_exactly", lambda x: x["preregistration"].update(record_sha256="0" * 64))
    attempt("remove_failure_mapping_source", "G02_source_sufficiency_covers_noise_circuit_decoders_and_failure_mapping", lambda x: x["source_sufficiency"]["ingredients"].pop())
    attempt("change_noise_variance", "G03_frozen_square_lattice_variance_and_three_anchor_points_are_exact", lambda x: x["points"][0].update(sigma_gkp_squared=0.5))
    attempt("claim_early_target_stop", "G04_every_point_obeys_failure_target_or_nmax_stopping", lambda x: x["points"][2].update(ml_failures=1999))
    attempt("break_crn_hash", "G05_common_random_numbers_and_seed_clusters_are_complete", lambda x: x["points"][0].update(ml_primitive_draw_sha256="0" * 64))
    attempt("forge_paired_partition", "G06_raw_paired_counts_are_internally_conservative", lambda x: x["points"][0].update(ci_only=x["points"][0]["ci_only"] + 1))
    attempt("forge_anchor_pass", "G07_all_six_table_i_anchor_discrepancies_pass_frozen_dual_tolerance", lambda x: x["anchor_discrepancies"][0].update(joint_pass=False))
    attempt("erase_paired_advantage", "G08_ml_reduction_is_paired_significant_at_all_points_after_holm", lambda x: x["points"][0].update(paired_difference=-0.1))
    attempt("individual_round_bootstrap", "G09_wilson_and_cluster_bootstrap_statistics_are_complete", lambda x: x["points"][0]["cluster_bootstrap"].update(unit="individual_round"))
    attempt("inject_boundary_mismatch", "G10_boundary_likelihood_oracle_has_100k_actual_crossings_and_zero_mismatch", lambda x: x["boundary_audit"].update(production_brute_mismatches=1))
    attempt("shrink_brute_radius", "G11_independent_brute_enclosure_and_likelihood_optimality_are_explicit", lambda x: x["boundary_audit"].update(brute_radius=1))
    attempt("invent_nanosecond_latency", "G12_operation_cost_is_fixed_and_not_mislabeled_as_latency", lambda x: x["operation_cost"].update(measured_latency_ns=12.0))
    attempt("promote_outer_threshold", "G13_outer_threshold_and_hardware_claims_remain_literature_only_or_null", lambda x: x["claim_registry"].update(NOH_9P9DB_OUTER_THRESHOLD="PROJECT_REPRODUCED"))
    attempt("forge_well_formed_source_data_hash", "G14_runtime_memory_raw_source_data_and_live_bindings_are_valid", lambda x: x["source_data"].update(sha256="0" * 64))
    attempt("forge_mutation_count", "G15_targeted_semantic_mutations_are_all_detected", lambda x: x.update(semantic_mutation_audit={"count": 15, "detected": 14, "cases": []}))
    return {"count": len(cases), "detected": sum(row["rejected"] for row in cases), "cases": cases}


def build_report() -> dict[str, Any]:
    frozen = _preregistered_experiment()
    seeds = list(map(int, frozen["seeds"]["values"]))
    start = perf_counter()
    points = [
        simulate_point(
            float(db),
            seeds,
            int(frozen["sample_size"]["target_failures_per_method_point"]),
            int(frozen["sample_size"]["n_max_per_squeezing_point"]),
        )
        for db in frozen["config"]["squeezing_db"]
    ]
    raw_p = [row["mcnemar_exact_two_sided_p"] for row in points]
    boundary = _boundary_audit(int(frozen["config"]["boundary_likelihood_samples"]))
    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "project-native matched reproduction of Noh 2022 Table-I gate-level CI versus analog ML only",
        "preregistration": {
            "experiment_id": frozen["experiment_id"],
            "record_sha256": _canonical_sha256(frozen),
            "source_record_sha256": _canonical_sha256(_source_record()),
            "seeds": seeds,
            "seed_namespace": frozen["seeds"]["namespace"],
            "target_failures_per_method_point": frozen["sample_size"]["target_failures_per_method_point"],
            "n_max_per_squeezing_point": frozen["sample_size"]["n_max_per_squeezing_point"],
            "runtime_budget_seconds": frozen["runtime_budget"]["wall_clock_seconds"],
            "memory_budget_gib": frozen["runtime_budget"]["memory_gib"],
        },
        "source_sufficiency": source_sufficiency_contract(),
        "model_contract": {
            "lambda": LAMBDA,
            "lattice_spacing": "sqrt(pi)",
            "primitive_shift_count": 8,
            "primitive_distribution": "iid N(0,sigma_gkp^2)",
            "sigma_gkp_squared": "0.5*10^(-squeezing_dB/10)",
            "q_covariance_over_sigma_squared": [[2.0, 1.0], [1.0, 3.0]],
            "p_covariance_over_sigma_squared": [[3.0, -1.0], [-1.0, 2.0]],
            "logical_failure": "any decoded integer odd",
        },
        "points": points,
        "anchor_discrepancies": _anchor_rows(points, frozen),
        "multiplicity": {
            "family": "three squeezing-point paired CI-vs-ML tests",
            "raw_mcnemar_p": raw_p,
            "holm_adjusted_mcnemar_p": _holm_adjust(raw_p),
            "alpha": 0.05,
        },
        "boundary_audit": boundary,
        "operation_cost": {
            "ci": {"scalar_round_to_nearest": 4, "correlated_2d_classifiers": 0, "asymptotic_per_gate": "O(1)"},
            "ml": {"scalar_round_to_nearest": 4, "correlated_2d_classifiers": 2, "classifier_facet_tests": "at most 9 scalar comparisons per quadrature", "asymptotic_per_gate": "O(1)", "exponentials_or_alias_sums": 0},
            "independent_oracle": {"candidates_per_quadrature_pair": 25, "used_for_correctness_only": True},
            "measured_latency_ns": None,
            "python_runtime_boundary": "host vectorized Monte Carlo correctness workload; not latency, FPGA timing or real-time throughput",
        },
        "claim_registry": {
            "NOH_TABLE_I_GATE_REPRODUCTION": "PROJECT_NATIVE_MATCHED",
            "NOH_9P9DB_OUTER_THRESHOLD": "LITERATURE_ONLY_NULL",
            "CI_LT50NS": "NULL_UNSOURCED",
            "ML_GT1MS": "NULL_UNSOURCED",
            "HARDWARE_RESOURCES": "NULL_NOT_IMPLEMENTED",
            "PHASE6B_V5_VERDICT": "READ_ONLY_NO_GO_UNCHANGED",
        },
        "allowed_wording": [
            "Under the Noh 2022 finite-squeezing-only two-GKP CNOT model, an independent project implementation reproduces the Table-I CI and analog-ML gate failure rates within the frozen tolerance.",
            "The correlated ML decoder significantly reduces gate failure relative to CI at 9, 12 and 13 dB in paired common-random-number simulation.",
        ],
        "forbidden_wording": [
            "The project reproduces or exceeds the 9.9 dB surface-GKP threshold.",
            "The host Monte Carlo runtime is a decoder latency or FPGA measurement.",
            "Gate-level CNOT failure proves superiority on the project's repeated-memory drift task.",
        ],
    }
    elapsed = perf_counter() - start
    peak = max([boundary["peak_tracemalloc_bytes"], *[row["peak_tracemalloc_bytes"] for row in points]])
    report["execution_budget_audit"] = {
        "runtime_seconds": elapsed,
        "runtime_budget_seconds": frozen["runtime_budget"]["wall_clock_seconds"],
        "peak_tracemalloc_bytes": peak,
        "memory_budget_bytes": int(frozen["runtime_budget"]["memory_gib"] * (1 << 30)),
        "within_runtime_budget": elapsed <= frozen["runtime_budget"]["wall_clock_seconds"],
        "within_memory_budget": peak <= frozen["runtime_budget"]["memory_gib"] * (1 << 30),
        "measurement_boundary": "Python simulation/correctness workload only",
    }
    _write_csv(report)
    report["source_data"] = {"path": _relative(DEFAULT_SOURCE_DATA), "sha256": _sha256(DEFAULT_SOURCE_DATA), "rows": sum(1 for _ in DEFAULT_SOURCE_DATA.open(encoding="utf-8")) - 1}
    report["ontology_semantic_sha256"] = _canonical_sha256(_ontology_semantic(_load(ONTOLOGY)))
    report["source_audit_semantic_sha256"] = _canonical_sha256(_source_semantic(_load(SOURCE_AUDIT)))
    report["bindings"] = {
        "implementation": _binding(Path(__file__)),
        "preregistration_config": _binding(PREREG_CONFIG),
        "ontology_initial": _binding(ONTOLOGY),
        "source_audit_initial": _binding(SOURCE_AUDIT),
        "primary_reader": _binding(PRIMARY_READER),
        "source_data": _binding(DEFAULT_SOURCE_DATA),
    }
    report["semantic_mutation_audit"] = {"count": 15, "detected": 15, "cases": []}
    report["gates"] = evaluate_gates(report)
    report["semantic_mutation_audit"] = _mutations(report)
    report["gates"] = evaluate_gates(report)
    failed = [name for name, passed in report["gates"].items() if not passed]
    report["gate_summary"] = {"passed": len(report["gates"]) - len(failed), "failed": failed}
    report["verdict"] = VERDICT if not failed else FAIL_VERDICT
    return report


def verify_report(report: Mapping[str, Any]) -> None:
    gates = evaluate_gates(report)
    if dict(report["gates"]) != gates:
        raise ValueError("stored T6.17.2 gates do not match recomputation")
    failed = [name for name, passed in gates.items() if not passed]
    expected_summary = {"passed": len(gates) - len(failed), "failed": failed}
    expected_verdict = VERDICT if not failed else FAIL_VERDICT
    if report["gate_summary"] != expected_summary or report["verdict"] != expected_verdict:
        raise ValueError("stored T6.17.2 summary/verdict does not match recomputation")
    if report["source_data"]["sha256"] != _sha256(ROOT / report["source_data"]["path"]):
        raise ValueError("T6.17.2 Source Data drifted")
    row_count = sum(1 for _ in (ROOT / report["source_data"]["path"]).open(encoding="utf-8")) - 1
    if row_count != report["source_data"]["rows"]:
        raise ValueError("T6.17.2 Source Data row count drifted")


def write_markdown(report: Mapping[str, Any], path: Path = DEFAULT_MARKDOWN) -> None:
    lines = [
        "# T6.17.2 Noh 2022 双 GKP CNOT：CI / analog ML 独立复现",
        "",
        f"- verdict：`{report['verdict']}`",
        f"- source-sufficiency：`{report['source_sufficiency']['status']}`",
        f"- gates / mutations：{report['gate_summary']['passed']}/15 / {report['semantic_mutation_audit']['detected']}/15",
        f"- correctness boundary：{report['boundary_audit']['production_brute_mismatches']}/{report['boundary_audit']['points']:,} mismatch",
        f"- runtime：{report['execution_budget_audit']['runtime_seconds']:.3f} s（Python correctness workload，非 decoder latency）",
        "",
        "## Table I 锚点复现",
        "",
        "| dB | method | failures / trials | estimate | literature | abs diff | rel diff | Wilson 95% |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    anchors = {(row["squeezing_db"], row["method"]): row for row in report["anchor_discrepancies"]}
    for point in report["points"]:
        for method in ("CI", "ML"):
            key = method.lower()
            anchor = anchors[(point["squeezing_db"], method)]
            interval = point[f"{key}_wilson_95"]
            lines.append(f"| {point['squeezing_db']:.0f} | {method} | {point[f'{key}_failures']:,} / {point['trials']:,} | {point[f'{key}_probability']:.8g} | {anchor['literature_anchor']:.8g} | {anchor['absolute_discrepancy']:.3g} | {anchor['relative_discrepancy']:.2%} | [{interval[0]:.8g}, {interval[1]:.8g}] |")
    lines += [
        "",
        "## 证据边界",
        "",
        "本 task 只复现 Table I 的 gate-level 对象：两个 square-lattice GKP qubit、`lambda=1`、有限 squeezing 高斯位移是唯一噪声。8 个独立位移和四个净位移按 Appendix C Eq. (C19) 直接采样；CI 按 Eq. (27)，ML 按 Eqs. (30)/(33) 与 Algorithms 1/2；任一整数为奇数即是非平凡 Pauli failure。",
        "",
        "100,000 个样本由 50,000 对实际跨越 q/p ML Voronoi facet 的单侧点组成；paper algorithm 与独立 25-candidate likelihood oracle 为零 mismatch。exact tie 的 argmin 非唯一，因此不拿 tie convention 制造假 mismatch。",
        "",
        "`9.9 dB` 仍只是文献中 full surface–GKP finite-size threshold。本 task 没有 outer-code lattice、matching graph 或 code-distance crossing，故该值保持 `LITERATURE_ONLY_NULL`；CI `<50 ns`、ML `>1 ms` 和硬件资源也没有被本仿真补写。",
        "",
        "## 产物",
        "",
        f"- report：`{_relative(DEFAULT_REPORT)}`",
        f"- Source Data：`{report['source_data']['path']}`（{report['source_data']['rows']} rows）",
        f"- implementation：`{_relative(Path(__file__))}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="verify the stored report instead of regenerating")
    args = parser.parse_args()
    if args.verify:
        verify_report(_load(DEFAULT_REPORT))
        print(f"verified {DEFAULT_REPORT}")
        return
    report = build_report()
    DEFAULT_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(report)
    verify_report(report)
    print(json.dumps({"verdict": report["verdict"], "gates": report["gate_summary"], "source_data_rows": report["source_data"]["rows"], "runtime_seconds": report["execution_budget_audit"]["runtime_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
