"""Observed-cluster joint maxT engine for the powered T04 outcome.

Unlike the synthetic T05/T06 power design, every multiplier replicate here is
constructed from real paired-cluster contributions.  A single deterministic
Rademacher sign is keyed by ``(frozen_seed, replicate, cluster_root_id)`` and is
therefore reused across every gate that observes the same physical cluster.
Independent roots never share a sign merely because their local position is
equal.

Nonlinear estimands are bootstrapped in their native environment:

* scalar difference: ``abs(mean(z))``;
* vector difference: ``||mean(z)||`` (L2 or L-infinity);
* density difference: ``0.5 * ||mean(delta_rho)||_1``.

There is no null-gradient jackknife, mean-of-norms substitution, family-wise
split, pointwise-z rescue, or synthetic influence path.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import ceil, isfinite
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


STATISTICS_SCHEMA = "PHASE9-POWERED-OBSERVED-JOINT-MAXT-V1"
SUPPORTED_KINDS = {
    "scalar_abs_mean",
    "scalar_nonnegative_mean",
    "vector_l2_mean",
    "vector_l1_mean",
    "vector_linf_mean",
    "density_half_trace_mean",
}


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: object) -> str:
    return sha256(_canonical(value)).hexdigest()


def _higher_quantile(values: np.ndarray, probability: float) -> float:
    array = np.sort(np.asarray(values, dtype=np.float64))
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("quantile values must be one finite vector")
    if not 0.0 < probability < 1.0:
        raise ValueError("quantile probability must lie in (0,1)")
    index = max(0, min(array.size - 1, ceil(probability * array.size) - 1))
    return float(array[index])


def rademacher_sign(seed: int, replicate: int, cluster_root_id: str) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Rademacher seed must be a nonnegative integer")
    if isinstance(replicate, bool) or not isinstance(replicate, int) or replicate < 0:
        raise ValueError("replicate must be a nonnegative integer")
    if not isinstance(cluster_root_id, str) or not cluster_root_id:
        raise ValueError("cluster_root_id is required")
    payload = f"{seed}|{replicate}|{cluster_root_id}".encode("utf-8")
    return 1 if (sha256(payload).digest()[0] & 1) else -1


def rademacher_matrix(
    *,
    seed: int,
    replicates: int,
    cluster_root_ids: Sequence[str],
) -> np.ndarray:
    if replicates != 199:
        raise ValueError("formal joint maxT requires exactly B=199")
    roots = tuple(cluster_root_ids)
    if len(roots) == 0 or len(set(roots)) != len(roots):
        raise ValueError("cluster roots must be non-empty and unique")
    result = np.empty((replicates, len(roots)), dtype=np.int8)
    for replicate in range(replicates):
        result[replicate] = [
            rademacher_sign(seed, replicate, root) for root in roots
        ]
    return result


def _validated_values(kind: str, values: np.ndarray) -> np.ndarray:
    if kind not in SUPPORTED_KINDS:
        raise ValueError(f"unsupported observed estimand kind: {kind}")
    array = np.asarray(values)
    if array.shape[0] < 2 or not np.all(np.isfinite(array)):
        raise ValueError("cluster contribution array must be finite with n>=2")
    if kind.startswith("scalar_"):
        if array.ndim != 1:
            raise ValueError("scalar contribution must have shape (n,)")
        array = np.asarray(array, dtype=np.float64)
        if kind == "scalar_nonnegative_mean" and np.any(array < 0.0):
            raise ValueError("nonnegative scalar contribution contains negatives")
        return array
    if kind.startswith("vector_"):
        if array.ndim != 2 or array.shape[1] == 0:
            raise ValueError("vector contribution must have shape (n,d)")
        return np.asarray(array, dtype=np.float64)
    if (
        array.ndim != 3
        or array.shape[1] != array.shape[2]
        or array.shape[1] == 0
    ):
        raise ValueError("density contribution must have shape (n,d,d)")
    density = np.asarray(array, dtype=np.complex128)
    hermiticity = np.max(
        np.linalg.norm(
            density - np.swapaxes(density.conj(), 1, 2),
            axis=(1, 2),
        )
    )
    if float(hermiticity) > 5.0e-10:
        raise ValueError("density differences must be Hermitian")
    return density


def _functional(kind: str, mean: np.ndarray | np.generic | float) -> float:
    if kind == "scalar_abs_mean":
        return abs(float(mean))
    if kind == "scalar_nonnegative_mean":
        value = float(mean)
        if value < -1.0e-12:
            raise ValueError("nonnegative scalar mean became negative")
        return max(value, 0.0)
    array = np.asarray(mean)
    if kind == "vector_l2_mean":
        return float(np.linalg.norm(array, ord=2))
    if kind == "vector_l1_mean":
        return float(np.linalg.norm(array, ord=1))
    if kind == "vector_linf_mean":
        return float(np.linalg.norm(array, ord=np.inf))
    if kind == "density_half_trace_mean":
        hermitian = (array + array.conj().T) / 2.0
        return float(0.5 * np.sum(np.abs(np.linalg.eigvalsh(hermitian))))
    raise ValueError(f"unsupported functional kind: {kind}")


@dataclass(frozen=True)
class ObservedGateInput:
    gate_id: str
    family: str
    direction: str
    margin: float
    kind: str
    cluster_root_ids: tuple[str, ...]
    values: np.ndarray
    quantization_radius: float


@dataclass(frozen=True)
class ObservedConservativeGateInput:
    """Two cutoff legs sharing roots/signs but retaining separate uncertainty."""

    gate_id: str
    family: str
    direction: str
    margin: float
    kind: str
    cluster_root_ids: tuple[str, ...]
    leg_a_values: np.ndarray
    leg_b_values: np.ndarray
    leg_a_quantization_radius: float
    leg_b_quantization_radius: float


@dataclass(frozen=True)
class ObservedGateResult:
    gate_id: str
    family: str
    direction: str
    margin: float
    kind: str
    cluster_count: int
    point: float
    standard_error: float
    quantization_radius: float
    replicate_errors: np.ndarray
    standardized_replicate_errors: np.ndarray
    bound_components: tuple[tuple[float, float, float], ...]
    leg_details: tuple[Mapping[str, object], ...]
    replicate_values_sha256: str


def evaluate_gate(
    gate: ObservedGateInput,
    *,
    signs: np.ndarray,
) -> ObservedGateResult:
    if (
        not isinstance(gate.gate_id, str)
        or not gate.gate_id
        or not isinstance(gate.family, str)
        or not gate.family
    ):
        raise ValueError("gate identity/family missing")
    if gate.direction not in {"upper", "lower"}:
        raise ValueError("gate direction must be upper/lower")
    if not isfinite(float(gate.margin)):
        raise ValueError("gate margin must be finite")
    quantization = float(gate.quantization_radius)
    if not isfinite(quantization) or quantization < 0.0:
        raise ValueError("quantization radius must be finite and nonnegative")
    values = _validated_values(gate.kind, gate.values)
    roots = tuple(gate.cluster_root_ids)
    if len(roots) != values.shape[0] or len(set(roots)) != len(roots):
        raise ValueError("gate roots do not align one-to-one with contributions")
    matrix = np.asarray(signs, dtype=np.int8)
    if matrix.shape != (199, values.shape[0]) or not np.all(
        (matrix == -1) | (matrix == 1)
    ):
        raise ValueError("gate sign matrix must have shape (199,n) with ±1")
    mean = np.mean(values, axis=0)
    point = _functional(gate.kind, mean)
    centered = values - mean
    flattened = centered.reshape(values.shape[0], -1)
    perturbations = (
        matrix.astype(np.float64) @ flattened / float(values.shape[0])
    ).reshape((199, *np.asarray(mean).shape))
    replicate_values = np.asarray(
        [
            _functional(gate.kind, mean + perturbation)
            for perturbation in perturbations
        ],
        dtype=np.float64,
    )
    errors = np.abs(replicate_values - point)
    standard_error = float(np.sqrt(np.mean(np.square(errors))))
    if (
        not np.all(np.isfinite(replicate_values))
        or not isfinite(standard_error)
        or standard_error <= 1.0e-12
    ):
        raise RuntimeError(
            f"INCOMPLETE_FAIL_CLOSED: degenerate observed SE for {gate.gate_id}"
        )
    fingerprint = sha256(
        np.asarray(replicate_values, dtype="<f8").tobytes(order="C")
    ).hexdigest()
    return ObservedGateResult(
        gate_id=gate.gate_id,
        family=gate.family,
        direction=gate.direction,
        margin=float(gate.margin),
        kind=gate.kind,
        cluster_count=values.shape[0],
        point=point,
        standard_error=standard_error,
        quantization_radius=quantization,
        replicate_errors=errors,
        standardized_replicate_errors=errors / standard_error,
        bound_components=((point, standard_error, quantization),),
        leg_details=(),
        replicate_values_sha256=fingerprint,
    )


def evaluate_conservative_gate(
    gate: ObservedConservativeGateInput,
    *,
    signs: np.ndarray,
) -> ObservedGateResult:
    """Evaluate both cutoff legs and retain replicate-wise conservative max."""

    if gate.direction != "upper":
        raise ValueError("conservative cutoff leg gate must be upper-directed")
    common = {
        "gate_id": gate.gate_id,
        "family": gate.family,
        "direction": gate.direction,
        "margin": gate.margin,
        "kind": gate.kind,
        "cluster_root_ids": gate.cluster_root_ids,
    }
    first = evaluate_gate(
        ObservedGateInput(
            **common,
            values=gate.leg_a_values,
            quantization_radius=gate.leg_a_quantization_radius,
        ),
        signs=signs,
    )
    second = evaluate_gate(
        ObservedGateInput(
            **common,
            values=gate.leg_b_values,
            quantization_radius=gate.leg_b_quantization_radius,
        ),
        signs=signs,
    )
    standardized = np.maximum(
        first.standardized_replicate_errors,
        second.standardized_replicate_errors,
    )
    fingerprint = _sha(
        {
            "leg_a": first.replicate_values_sha256,
            "leg_b": second.replicate_values_sha256,
            "replicatewise_standardized_max_sha256": sha256(
                np.asarray(standardized, dtype="<f8").tobytes(order="C")
            ).hexdigest(),
        }
    )
    details = tuple(
        {
            "leg": name,
            "point": result.point,
            "standard_error": result.standard_error,
            "quantization_radius": result.quantization_radius,
            "replicate_values_sha256": result.replicate_values_sha256,
        }
        for name, result in (("A", first), ("B", second))
    )
    return ObservedGateResult(
        gate_id=gate.gate_id,
        family=gate.family,
        direction=gate.direction,
        margin=float(gate.margin),
        kind=gate.kind,
        cluster_count=len(gate.cluster_root_ids),
        point=max(first.point, second.point),
        standard_error=max(first.standard_error, second.standard_error),
        quantization_radius=max(
            first.quantization_radius,
            second.quantization_radius,
        ),
        replicate_errors=np.maximum(
            first.replicate_errors,
            second.replicate_errors,
        ),
        standardized_replicate_errors=standardized,
        bound_components=(
            first.bound_components[0],
            second.bound_components[0],
        ),
        leg_details=details,
        replicate_values_sha256=fingerprint,
    )


def exact_mapping_gate_results(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, object]]:
    """Validate the six deterministic c36/c40/c44 mapping gates."""

    expected_ids = {
        *(f"mapping/c{cutoff}/principal_singular" for cutoff in (36, 40, 44)),
        *(f"mapping/c{cutoff}/projector_frobenius" for cutoff in (36, 40, 44)),
    }
    by_id = {str(row.get("gate_id")): row for row in rows}
    if set(by_id) != expected_ids or len(rows) != 6:
        raise RuntimeError("exact mapping gate set must contain exactly six IDs")
    results: list[dict[str, object]] = []
    for gate_id in sorted(expected_ids):
        row = by_id[gate_id]
        value = float(row["point"])
        if not isfinite(value):
            raise ValueError("exact mapping point must be finite")
        if gate_id.endswith("principal_singular"):
            direction, margin = "lower", 0.95
            passed = value >= margin
        else:
            direction, margin = "upper", 0.3
            passed = value <= margin
        results.append(
            {
                "gate_id": gate_id,
                "direction": direction,
                "margin": margin,
                "point": value,
                "passed": bool(passed),
                "stochastic": False,
            }
        )
    return results


def joint_maxt(
    gates: Sequence[ObservedGateInput],
    *,
    rademacher_seed: int,
    confidence: float = 0.95,
    multiplier_replicates: int = 199,
    expected_stochastic_gate_count: int = 3037,
) -> dict[str, Any]:
    """Evaluate an in-memory closed family with one common maxT critical."""

    identifiers = [gate.gate_id for gate in gates]
    if len(gates) != expected_stochastic_gate_count:
        raise RuntimeError(
            f"closed family gate count drift: {len(gates)} "
            f"!= {expected_stochastic_gate_count}"
        )
    if len(set(identifiers)) != len(identifiers):
        raise RuntimeError("duplicate stochastic gate ID")
    return joint_maxt_stream(
        iter(gates),
        expected_gate_ids=identifiers,
        rademacher_seed=rademacher_seed,
        confidence=confidence,
        multiplier_replicates=multiplier_replicates,
        expected_stochastic_gate_count=expected_stochastic_gate_count,
    )


def joint_maxt_stream(
    gates: Iterable[ObservedGateInput | ObservedConservativeGateInput],
    *,
    expected_gate_ids: Sequence[str],
    rademacher_seed: int,
    confidence: float = 0.95,
    multiplier_replicates: int = 199,
    expected_stochastic_gate_count: int = 3037,
) -> dict[str, Any]:
    """Stream the closed family so raw density tensors need not coexist.

    Only the 199-value replicate error vector for each gate and the sign
    matrices for distinct cluster-root sets remain resident.  The caller must
    supply the outcome-blind closed gate-ID universe independently of the
    observed iterator; deletion, duplication, or substitution is rejected
    after the iterator is exhausted.
    """

    if multiplier_replicates != 199:
        raise ValueError("formal joint maxT requires exactly B=199")
    if confidence != 0.95:
        raise ValueError("formal joint maxT confidence must remain 0.95")
    expected = tuple(expected_gate_ids)
    if (
        len(expected) != expected_stochastic_gate_count
        or len(set(expected)) != len(expected)
    ):
        raise RuntimeError(
            "outcome-blind expected stochastic gate universe drifted"
        )
    sign_cache: dict[tuple[str, ...], np.ndarray] = {}
    results: list[ObservedGateResult] = []
    observed_ids: list[str] = []
    observed_set: set[str] = set()
    for gate in gates:
        if gate.gate_id in observed_set:
            raise RuntimeError("duplicate stochastic gate ID")
        observed_ids.append(gate.gate_id)
        observed_set.add(gate.gate_id)
        roots = tuple(gate.cluster_root_ids)
        if roots not in sign_cache:
            sign_cache[roots] = rademacher_matrix(
                seed=rademacher_seed,
                replicates=multiplier_replicates,
                cluster_root_ids=roots,
            )
        results.append(
            evaluate_conservative_gate(gate, signs=sign_cache[roots])
            if isinstance(gate, ObservedConservativeGateInput)
            else evaluate_gate(gate, signs=sign_cache[roots])
        )
    if (
        len(results) != expected_stochastic_gate_count
        or observed_set != set(expected)
    ):
        missing = sorted(set(expected) - set(observed_ids))
        extra = sorted(set(observed_ids) - set(expected))
        raise RuntimeError(
            "closed family gate-ID coverage drift "
            f"missing={missing[:3]} extra={extra[:3]}"
        )
    results.sort(key=lambda value: value.gate_id)
    standardized = np.stack(
        [
            result.standardized_replicate_errors
            for result in results
        ],
        axis=0,
    )
    maxima = np.max(standardized, axis=0)
    critical = _higher_quantile(maxima, confidence)
    gate_rows: list[dict[str, object]] = []
    for result in results:
        if result.direction == "upper":
            bound = max(
                point + critical * standard_error + quantization
                for point, standard_error, quantization in result.bound_components
            )
            passed = bound <= result.margin
        else:
            bound = min(
                point - critical * standard_error - quantization
                for point, standard_error, quantization in result.bound_components
            )
            passed = bound >= result.margin
        gate_rows.append(
            {
                "gate_id": result.gate_id,
                "family": result.family,
                "direction": result.direction,
                "margin": result.margin,
                "kind": result.kind,
                "cluster_count": result.cluster_count,
                "point": result.point,
                "standard_error": result.standard_error,
                "joint_critical": critical,
                "quantization_radius": result.quantization_radius,
                "simultaneous_bound": bound,
                "passed": bool(passed),
                "replicate_values_sha256": result.replicate_values_sha256,
                "leg_details": list(result.leg_details),
            }
        )
    maximum_digest = sha256(
        np.asarray(maxima, dtype="<f8").tobytes(order="C")
    ).hexdigest()
    replicate_fingerprint = _sha(
        {
            row["gate_id"]: row["replicate_values_sha256"]
            for row in gate_rows
        }
    )
    report: dict[str, Any] = {
        "schema_version": STATISTICS_SCHEMA,
        "stochastic_gate_count": len(gate_rows),
        "multiplier_replicates": 199,
        "confidence": confidence,
        "quantile_method": "higher",
        "rademacher_seed": rademacher_seed,
        "joint_critical": critical,
        "maxima": maxima.tolist(),
        "maxima_sha256": maximum_digest,
        "replicate_fingerprint_sha256": replicate_fingerprint,
        "all_stochastic_gates_passed": all(
            bool(row["passed"]) for row in gate_rows
        ),
        "gates": gate_rows,
        "synthetic_influence_used": False,
        "pointwise_z_substitution_used": False,
        "family_split_used": False,
        "aggregate_rescue_used": False,
    }
    report["analysis_sha256"] = _sha(report)
    return report


__all__ = [
    "ObservedGateInput",
    "ObservedConservativeGateInput",
    "ObservedGateResult",
    "STATISTICS_SCHEMA",
    "SUPPORTED_KINDS",
    "evaluate_gate",
    "evaluate_conservative_gate",
    "exact_mapping_gate_results",
    "joint_maxt",
    "joint_maxt_stream",
    "rademacher_matrix",
    "rademacher_sign",
]
