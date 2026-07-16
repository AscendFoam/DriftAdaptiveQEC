"""T3.1.2 training-average static periodic-Gaussian MAP baseline.

The baseline is fit once from an evaluation-independent sequence of training
noise states.  It moment-matches the marginal displacement distribution using
the law of total covariance, freezes the resulting mean/covariance, and then
decodes every evaluation syndrome without updates or hidden truth.
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

from physics.constants import LATTICE_CONST
from physics.drift_processes import DriftState
from physics.ideal_gkp_decoder import map_decode_2d


STATIC_MAP_ID = "static_training_average_map"


@dataclass(frozen=True)
class StaticMAPDescriptor:
    baseline_id: str = STATIC_MAP_ID
    label: str = "Static MAP (training-average moment-matched Gaussian)"
    task_owner: str = "T3.1.2"
    comparison_role: str = "deployable_static_decoder_baseline"
    deployable: bool = True
    training_inputs: tuple[str, ...] = (
        "training_state_mean",
        "training_state_mixture_covariance",
        "training_state_weight",
    )
    evaluation_inputs: tuple[str, ...] = (
        "centered_modular_syndrome_q",
        "centered_modular_syndrome_p",
    )
    evaluation_hidden_truth_inputs: tuple[str, ...] = ()
    update_during_evaluation: bool = False
    fit_rule: str = "weighted marginal first moment plus law-of-total-covariance"
    evidence_scope: str = "static_periodic_gaussian_map_model_baseline"


STATIC_MAP_DESCRIPTOR = StaticMAPDescriptor()


@dataclass(frozen=True)
class StaticMAPParameters:
    mean: tuple[float, float]
    covariance: tuple[tuple[float, float], tuple[float, float]]
    training_windows: int
    effective_training_weight: float
    training_state_sha256: str
    training_protocol_id: str
    fit_rule: str = "moment_matched_marginal_law_of_total_covariance"
    loss_handling: str = "reject_nonzero_loss_gamma"
    outlier_handling: str = "moment_match_same_mean_gaussian_mixture_covariance"

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=float)
        covariance = np.asarray(self.covariance, dtype=float)
        if mean.shape != (2,) or not np.all(np.isfinite(mean)):
            raise ValueError("mean must contain two finite values")
        if covariance.shape != (2, 2) or not np.all(np.isfinite(covariance)):
            raise ValueError("covariance must be a finite 2x2 matrix")
        if not np.allclose(covariance, covariance.T, atol=1.0e-12, rtol=0.0):
            raise ValueError("covariance must be symmetric")
        if float(np.min(np.linalg.eigvalsh(covariance))) <= 0.0:
            raise ValueError("covariance must be positive definite")
        if isinstance(self.training_windows, bool) or not isinstance(
            self.training_windows, (int, np.integer)
        ):
            raise TypeError("training_windows must be an integer")
        if int(self.training_windows) < 2:
            raise ValueError("training_windows must be at least 2")
        weight = float(self.effective_training_weight)
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("effective_training_weight must be finite and positive")
        if len(self.training_state_sha256) != 64:
            raise ValueError("training_state_sha256 must be a SHA-256 hex digest")
        try:
            bytes.fromhex(self.training_state_sha256)
        except ValueError as exc:
            raise ValueError("training_state_sha256 must be a SHA-256 hex digest") from exc
        if not self.training_protocol_id.strip():
            raise ValueError("training_protocol_id must not be empty")

    def mean_array(self) -> NDArray[np.float64]:
        return np.asarray(self.mean, dtype=np.float64)

    def covariance_array(self) -> NDArray[np.float64]:
        return np.asarray(self.covariance, dtype=np.float64)


@dataclass(frozen=True)
class StaticMAPValidationConfig:
    evaluation_seeds: tuple[int, ...] = (
        20260721,
        20260722,
        20260723,
        20260724,
        20260725,
        20260726,
        20260727,
        20260728,
    )
    training_seed: int = 20260312
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        seeds = tuple(self.evaluation_seeds)
        if len(seeds) < 4 or len(set(seeds)) != len(seeds):
            raise ValueError("evaluation_seeds must contain at least four unique seeds")
        for seed in (*seeds, self.training_seed):
            if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
                raise TypeError("all seeds must be integers")
            if not 0 <= int(seed) <= 2**64 - 4:
                raise ValueError("all seeds must lie in [0, 2**64-4]")
        if int(self.training_seed) in set(int(seed) for seed in seeds):
            raise ValueError("training_seed must be disjoint from evaluation_seeds")
        confidence = float(self.confidence_level)
        if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
            raise ValueError("confidence_level must lie strictly between 0 and 1")
        object.__setattr__(self, "evaluation_seeds", tuple(int(seed) for seed in seeds))
        object.__setattr__(self, "training_seed", int(self.training_seed))
        object.__setattr__(self, "confidence_level", confidence)


def _state_training_hash(
    states: Sequence[DriftState],
    weights: NDArray[np.float64],
    training_protocol_id: str,
) -> str:
    payload = {
        "training_protocol_id": training_protocol_id,
        "states": [
            {
                "mu": [state.mu_q, state.mu_p],
                "mixture_covariance": state.mixture_covariance.tolist(),
                "loss_gamma": state.loss_gamma,
                "p_outlier": state.p_outlier,
                "outlier_scale": state.outlier_scale,
                "source": state.source,
                "regime": state.regime,
                "weight": float(weight),
            }
            for state, weight in zip(states, weights)
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def fit_static_map_from_training_states(
    states: Sequence[DriftState],
    *,
    weights: ArrayLike | None = None,
    training_protocol_id: str,
) -> StaticMAPParameters:
    """Fit one frozen Gaussian prior from training states only.

    For state ``j`` with mean ``mu_j`` and mixture covariance ``Sigma_j``,
    the fitted covariance is
    ``sum_j w_j [Sigma_j + (mu_j-mu_bar)(mu_j-mu_bar)^T]``.
    Nonzero physical loss is rejected rather than silently converted to
    additive displacement noise.
    """

    sequence = tuple(states)
    if len(sequence) < 2:
        raise ValueError("states must contain at least two training windows")
    if not all(isinstance(state, DriftState) for state in sequence):
        raise TypeError("states must contain only DriftState values")
    protocol_id = str(training_protocol_id).strip()
    if not protocol_id:
        raise ValueError("training_protocol_id must not be empty")
    if any(state.loss_gamma != 0.0 for state in sequence):
        raise ValueError(
            "static Gaussian MAP cannot silently absorb nonzero loss_gamma into displacement noise"
        )
    if weights is None:
        raw_weights = np.ones(len(sequence), dtype=np.float64)
    else:
        raw_weights = np.asarray(weights, dtype=np.float64)
        if raw_weights.shape != (len(sequence),):
            raise ValueError("weights must have one entry per training state")
    if not np.all(np.isfinite(raw_weights)) or np.any(raw_weights <= 0.0):
        raise ValueError("weights must be finite and strictly positive")
    normalized = raw_weights / float(np.sum(raw_weights))
    means = np.stack([state.mean for state in sequence], axis=0)
    mean = np.sum(normalized[:, np.newaxis] * means, axis=0)
    covariance = np.zeros((2, 2), dtype=np.float64)
    for state, state_mean, weight in zip(sequence, means, normalized):
        delta = state_mean - mean
        covariance += float(weight) * (
            state.mixture_covariance + np.outer(delta, delta)
        )
    covariance = 0.5 * (covariance + covariance.T)
    if float(np.min(np.linalg.eigvalsh(covariance))) <= 0.0:
        raise ValueError("training-average covariance is not positive definite")
    return StaticMAPParameters(
        mean=(float(mean[0]), float(mean[1])),
        covariance=(
            (float(covariance[0, 0]), float(covariance[0, 1])),
            (float(covariance[1, 0]), float(covariance[1, 1])),
        ),
        training_windows=len(sequence),
        effective_training_weight=float(np.sum(raw_weights)),
        training_state_sha256=_state_training_hash(sequence, raw_weights, protocol_id),
        training_protocol_id=protocol_id,
    )


def static_map_logical_class(
    centered_syndrome: ArrayLike,
    parameters: StaticMAPParameters,
    *,
    lattice: float = LATTICE_CONST,
    chunk_size: int = 2_000,
) -> NDArray[np.int64]:
    """Decode centered syndromes using frozen parameters and no updates."""

    if not isinstance(parameters, StaticMAPParameters):
        raise TypeError("parameters must be StaticMAPParameters")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, (int, np.integer)):
        raise TypeError("chunk_size must be an integer")
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive")
    values = np.asarray(centered_syndrome, dtype=np.float64)
    if values.ndim == 0 or values.shape[-1] != 2:
        raise ValueError("centered_syndrome must have shape (..., 2)")
    if not np.all(np.isfinite(values)):
        raise ValueError("centered_syndrome must contain only finite values")
    original_shape = values.shape[:-1]
    flat = values.reshape(-1, 2)
    outputs: list[NDArray[np.int64]] = []
    for start in range(0, len(flat), int(chunk_size)):
        result = map_decode_2d(
            flat[start : start + int(chunk_size)],
            parameters.covariance_array(),
            mean=parameters.mean_array(),
            lattice=lattice,
        )
        outputs.append(np.asarray(result.logical_class, dtype=np.int64).reshape(-1))
    if not outputs:
        return np.empty(original_shape, dtype=np.int64)
    return np.concatenate(outputs).reshape(original_shape)


def validate_static_map_major_comparisons() -> tuple[str, ...]:
    """Validate schemas that explicitly select this formal static MAP anchor.

    Decoder tasks may instead declare a task-specific static anchor, such as
    an H-step final-outcome Bayesian comparator.  The central registry checks
    that every decoder comparison has exactly one explicit static anchor; this
    function must not overwrite that role contract with this module's method.
    """

    from cnn_fpga.benchmark.standard_binning_baseline import (
        major_comparison_registry,
        validate_major_comparison_registry,
    )

    validate_major_comparison_registry()
    gates: list[str] = []
    for entry in major_comparison_registry():
        if entry.static_anchor_method_id != STATIC_MAP_ID:
            continue
        occurrences = entry.method_ids.count(STATIC_MAP_ID)
        if occurrences != 1:
            raise ValueError(f"{entry.comparison_id} must contain {STATIC_MAP_ID} exactly once")
        gates.append(f"static_registry:{entry.comparison_id}")
    if not gates:
        raise ValueError("no decoder schemas select the formal static MAP anchor")
    return tuple(gates)


def _paired_interval_from_discordant(
    first_only: int,
    second_only: int,
    n_samples: int,
    confidence_level: float,
) -> dict[str, float]:
    if n_samples <= 1:
        raise ValueError("n_samples must exceed one")
    estimate = (first_only - second_only) / n_samples
    second_moment = (first_only + second_only) / n_samples
    variance = (second_moment - estimate**2) * n_samples / (n_samples - 1)
    standard_error = math.sqrt(max(variance, 0.0) / n_samples)
    z_value = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    return {
        "estimate": estimate,
        "standard_error": standard_error,
        "ci_low": estimate - z_value * standard_error,
        "ci_high": estimate + z_value * standard_error,
    }


def _source_sha256(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_static_map_validation(
    config: StaticMAPValidationConfig | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Run the eight-seed current main-comparison validation."""

    from cnn_fpga.benchmark.adaptive_drift_alignment import (
        AdaptiveAlignmentConfig,
        run_adaptive_drift_alignment,
    )
    from cnn_fpga.benchmark.standard_binning_baseline import (
        major_comparison_registry,
        validate_major_comparison_registry,
    )

    settings = StaticMAPValidationConfig() if config is None else config
    if not isinstance(settings, StaticMAPValidationConfig):
        raise TypeError("config must be StaticMAPValidationConfig")
    validate_major_comparison_registry()
    static_registry_gates = validate_static_map_major_comparisons()
    rows: list[dict[str, object]] = []
    training_hashes: set[str] = set()
    total_samples = 0
    total_standard_failures = 0
    total_static_failures = 0
    total_oracle_failures = 0
    standard_only = 0
    static_only = 0
    trace_hashes: set[str] = set()
    frozen_parameters: StaticMAPParameters | None = None
    for seed in settings.evaluation_seeds:
        result = run_adaptive_drift_alignment(
            AdaptiveAlignmentConfig(
                seed=seed,
                static_training_seed=settings.training_seed,
                bootstrap_replicates=0,
            )
        )
        training_hashes.add(result.static_training_state_sha256)
        if frozen_parameters is None:
            frozen_parameters = result.static_parameters
        elif result.static_parameters != frozen_parameters:
            raise AssertionError("static parameters changed across evaluation seeds")
        trace_hashes.add(result.trace_sha256)
        n_samples = result.paired_samples
        total_samples += n_samples
        total_standard_failures += sum(row.standard_failures for row in result.records)
        total_static_failures += sum(row.static_failures for row in result.records)
        total_oracle_failures += sum(row.oracle_failures for row in result.records)
        standard_only += result.standard_gap.static_only_failure_count
        static_only += result.standard_gap.dual_only_failure_count
        rows.append(
            {
                "evaluation_seed": seed,
                "training_seed": settings.training_seed,
                "training_state_sha256": result.static_training_state_sha256,
                "evaluation_trace_sha256": result.trace_sha256,
                "paired_samples": n_samples,
                "standard_binning_error_rate": result.standard_error_rate,
                "static_training_average_map_error_rate": result.static_error_rate,
                "full_state_model_oracle_map_error_rate": result.oracle_error_rate,
                "standard_minus_static": result.standard_gap.static_minus_dual.estimate,
                "standard_minus_static_ci_low": result.standard_gap.static_minus_dual.ci_low,
                "standard_minus_static_ci_high": result.standard_gap.static_minus_dual.ci_high,
                "standard_only_failure_count": result.standard_gap.static_only_failure_count,
                "static_only_failure_count": result.standard_gap.dual_only_failure_count,
            }
        )
    paired = _paired_interval_from_discordant(
        standard_only,
        static_only,
        total_samples,
        settings.confidence_level,
    )
    gates = {
        "training_evaluation_seed_disjoint": settings.training_seed
        not in settings.evaluation_seeds,
        "single_frozen_training_parameter_hash": len(training_hashes) == 1,
        "unique_materialized_evaluation_traces": len(trace_hashes)
        == len(settings.evaluation_seeds),
        "formal_static_map_present_in_declared_schemas": all(
            STATIC_MAP_ID in entry.method_ids and "standard_binning" in entry.method_ids
            for entry in major_comparison_registry()
            if entry.static_anchor_method_id == STATIC_MAP_ID
        ),
        "static_better_than_standard_each_seed": all(
            float(row["standard_minus_static_ci_low"]) > 0.0 for row in rows
        ),
        "aggregate_static_improvement_resolved": paired["ci_low"] > 0.0,
        "static_remains_above_model_oracle": (
            total_static_failures > total_oracle_failures
        ),
        "descriptor_has_no_evaluation_truth": not STATIC_MAP_DESCRIPTOR.evaluation_hidden_truth_inputs,
        "no_evaluation_updates": not STATIC_MAP_DESCRIPTOR.update_during_evaluation,
    }
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise AssertionError(f"static MAP validation gates failed: {failed}")
    if frozen_parameters is None:
        raise AssertionError("static MAP validation produced no frozen parameters")
    source_paths = (
        Path(__file__),
        Path(__file__).with_name("adaptive_drift_alignment.py"),
        Path(__file__).with_name("standard_binning_baseline.py"),
        Path(__file__).parents[2] / "physics" / "ideal_gkp_decoder.py",
    )
    payload: dict[str, object] = {
        "schema_version": "t3.1.2-static-map-v2",
        "task_id": "T3.1.2",
        "status": "PASS",
        "evidence_scope": "multi_seed_synthetic_static_periodic_gaussian_map",
        "implementation_sha256": _source_sha256(source_paths),
        "descriptor": asdict(STATIC_MAP_DESCRIPTOR),
        "config": asdict(settings),
        "static_registry_gates": list(static_registry_gates),
        "frozen_static_parameters": asdict(frozen_parameters),
        "aggregate": {
            "evaluation_seeds": len(settings.evaluation_seeds),
            "paired_samples": total_samples,
            "training_state_sha256": next(iter(training_hashes)),
            "standard_binning_error_rate": total_standard_failures / total_samples,
            "static_training_average_map_error_rate": total_static_failures / total_samples,
            "full_state_model_oracle_map_error_rate": total_oracle_failures / total_samples,
            "standard_minus_static": paired,
            "standard_only_failure_count": standard_only,
            "static_only_failure_count": static_only,
        },
        "gate_summary": {
            "passed": len(gates),
            "failed": 0,
            "gates": gates,
        },
        "claim_boundary": {
            "allowed": (
                "evaluation-independent training-average moment-matched static Gaussian MAP "
                "outperforms standard binning on the registered multi-seed synthetic step scenario"
            ),
            "forbidden": (
                "static MAP is universally optimal, consumes evaluation truth, handles physical "
                "loss/protocol leakage, or is an FPGA/quantum-hardware measurement"
            ),
        },
    }
    normalized = json.loads(json.dumps(payload, ensure_ascii=False))
    return normalized, rows


def write_static_map_validation(
    json_path: str | Path,
    csv_path: str | Path,
    config: StaticMAPValidationConfig | None = None,
) -> dict[str, object]:
    payload, rows = build_static_map_validation(config)
    output_json = Path(json_path)
    output_csv = Path(csv_path)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate T3.1.2 static MAP baseline")
    parser.add_argument("--json", default="docs/t3_1_2_static_map_validation.json")
    parser.add_argument("--csv", default="docs/t3_1_2_static_map_source_data.csv")
    arguments = parser.parse_args()
    result = write_static_map_validation(arguments.json, arguments.csv)
    print(json.dumps(result["gate_summary"], ensure_ascii=False))


__all__ = [
    "STATIC_MAP_ID",
    "StaticMAPDescriptor",
    "STATIC_MAP_DESCRIPTOR",
    "StaticMAPParameters",
    "StaticMAPValidationConfig",
    "fit_static_map_from_training_states",
    "static_map_logical_class",
    "validate_static_map_major_comparisons",
    "build_static_map_validation",
    "write_static_map_validation",
]
