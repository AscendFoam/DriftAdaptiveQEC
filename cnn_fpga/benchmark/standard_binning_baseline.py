"""T3.1.1 standard-binning baseline contract and integration audit.

The deployed decision receives only a centered modular syndrome.  Fixed
half-cell nearest-lattice recovery therefore always selects the central even
coset; the hidden lattice-cell parity is used only by the evaluator to score
that decision.  Keeping these two operations separate prevents full
displacement truth from leaking into the baseline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from physics.constants import LATTICE_CONST
from physics.ideal_gkp_decoder import standard_binning_1d


STANDARD_BINNING_ID = "standard_binning"


@dataclass(frozen=True)
class DecoderBaselineDescriptor:
    baseline_id: str
    label: str
    task_owner: str
    comparison_role: str
    deployable: bool
    observation_inputs: tuple[str, ...]
    hidden_truth_inputs: tuple[str, ...]
    decision_rule: str
    tunable_parameters: tuple[str, ...]
    evidence_scope: str


@dataclass(frozen=True)
class MajorComparisonRegistration:
    comparison_id: str
    code_path: str
    comparison_kind: Literal[
        "decoder_algorithm_comparison",
        "implementation_sensitivity",
        "timing_sensitivity",
        "legacy_parameter_estimator_comparison",
    ]
    lifecycle: Literal["active", "future_contract", "frozen_legacy"]
    method_ids: tuple[str, ...]
    standard_binning_policy: Literal["required", "not_applicable"]
    rationale: str
    static_anchor_method_id: str | None = None
    reference_anchor_method_id: str | None = None


STANDARD_BINNING_DESCRIPTOR = DecoderBaselineDescriptor(
    baseline_id=STANDARD_BINNING_ID,
    label="Standard binning (fixed half-cell recovery)",
    task_owner="T3.1.1",
    comparison_role="deployable_static_decoder_baseline",
    deployable=True,
    observation_inputs=("centered_modular_syndrome_q", "centered_modular_syndrome_p"),
    hidden_truth_inputs=(),
    decision_rule=(
        "apply minus-centered-syndrome recovery and select the central even-even "
        "logical coset; half-open cells are inherited from standard_binning_1d"
    ),
    tunable_parameters=(),
    evidence_scope="ideal_square_gkp_syndrome_level_fixed_recovery",
)


_MAJOR_COMPARISONS = (
    MajorComparisonRegistration(
        comparison_id="t1_3_4_adaptive_drift_alignment",
        code_path="cnn_fpga/benchmark/adaptive_drift_alignment.py",
        comparison_kind="decoder_algorithm_comparison",
        lifecycle="active",
        method_ids=(
            STANDARD_BINNING_ID,
            "static_training_average_map",
            "window_variance_map",
            "ekf_map",
            "full_state_model_oracle_map",
        ),
        standard_binning_policy="required",
        rationale="current paired logical-decoder comparison",
        static_anchor_method_id="static_training_average_map",
        reference_anchor_method_id="full_state_model_oracle_map",
    ),
    MajorComparisonRegistration(
        comparison_id="phase5_main_decoder_benchmark",
        code_path="future:T5.1",
        comparison_kind="decoder_algorithm_comparison",
        lifecycle="future_contract",
        method_ids=(
            STANDARD_BINNING_ID,
            "static_training_average_map",
            "full_state_model_oracle_map",
        ),
        standard_binning_policy="required",
        rationale="schema guard: every future main decoder table must retain this anchor",
        static_anchor_method_id="static_training_average_map",
        reference_anchor_method_id="full_state_model_oracle_map",
    ),
    MajorComparisonRegistration(
        comparison_id="t2_4_3_fixed_point_precision_sensitivity",
        code_path="cnn_fpga/runtime/fixed_point_chain.py",
        comparison_kind="implementation_sensitivity",
        lifecycle="active",
        method_ids=("float_periodic_map", "quantized_periodic_map"),
        standard_binning_policy="not_applicable",
        rationale="same-decoder precision sweep, not an algorithm-ranking table",
    ),
    MajorComparisonRegistration(
        comparison_id="t3_1_4_static_sbs_branch_comparison",
        code_path="cnn_fpga/benchmark/static_protocol_decoder.py",
        comparison_kind="decoder_algorithm_comparison",
        lifecycle="active",
        method_ids=(
            "direct_observed_sbs_branch",
            "static_sbs_branch_map",
            "static_sbs_observation_reset_bayes",
            "ideal_sbs_branch_truth_reference",
        ),
        standard_binning_policy="not_applicable",
        rationale=(
            "separate ideal-sBs-Kraus-branch target with g/e/leakage observations; "
            "it must not be mixed with the logical-coset standard-binning schema"
        ),
        static_anchor_method_id="static_sbs_observation_reset_bayes",
        reference_anchor_method_id="ideal_sbs_branch_truth_reference",
    ),
    MajorComparisonRegistration(
        comparison_id="t3_1_5_topk_periodic_map_sensitivity",
        code_path="cnn_fpga/benchmark/topk_lattice_coset_map.py",
        comparison_kind="implementation_sensitivity",
        lifecycle="active",
        method_ids=("full_periodic_gaussian_map", "topk_lattice_coset_map"),
        standard_binning_policy="not_applicable",
        rationale=(
            "same single-mode periodic-Gaussian decoder and candidate rectangle with only "
            "the per-coset likelihood truncation K changed; this is not an algorithm ranking"
        ),
    ),
    MajorComparisonRegistration(
        comparison_id="t3_2_1_memory_bayesian_episode_comparison",
        code_path="cnn_fpga/benchmark/memory_assisted_bayesian_decoder.py",
        comparison_kind="decoder_algorithm_comparison",
        lifecycle="active",
        method_ids=(
            STANDARD_BINNING_ID,
            "final_outcome_static_periodic_bayes",
            "periodic_memory_assisted_bayes",
            "full_episode_logical_truth_reference",
        ),
        standard_binning_policy="required",
        rationale=(
            "bounded no-intermediate-correction modular-syndrome episodes with a known "
            "start; all deployable methods use the same traces and the memory method is "
            "limited to the registered causal observation/history budget"
        ),
        static_anchor_method_id="final_outcome_static_periodic_bayes",
        reference_anchor_method_id="full_episode_logical_truth_reference",
    ),
    MajorComparisonRegistration(
        comparison_id="t3_2_2_continuous_adaptive_map_comparison",
        code_path="cnn_fpga/benchmark/continuous_adaptive_map.py",
        comparison_kind="decoder_algorithm_comparison",
        lifecycle="active",
        method_ids=(
            STANDARD_BINNING_ID,
            "static_training_average_map",
            "latest_window_periodic_moment_map",
            "ewma_periodic_moment_map",
            "kalman_constant_velocity_periodic_map",
            "full_state_model_oracle_map",
        ),
        standard_binning_policy="required",
        rationale=(
            "continuous wrapped-Gaussian drift comparison with one-window causal delay, "
            "shared residual observations, frozen training-only hyperparameters and the "
            "formal static/full-state reference anchors"
        ),
        static_anchor_method_id="static_training_average_map",
        reference_anchor_method_id="full_state_model_oracle_map",
    ),
    MajorComparisonRegistration(
        comparison_id="t3_2_3_sliding_window_syndrome_comparison",
        code_path="cnn_fpga/benchmark/sliding_window_syndrome_estimator.py",
        comparison_kind="decoder_algorithm_comparison",
        lifecycle="active",
        method_ids=(
            STANDARD_BINNING_ID,
            "static_training_average_map",
            "latest_window_periodic_moment_map",
            "training_selected_sliding_window_periodic_map",
            "full_state_model_oracle_map",
        ),
        standard_binning_policy="required",
        rationale=(
            "uniform overlapping-window syndrome estimator with training-only history-"
            "length selection, one-window delay and identical observation/update budget"
        ),
        static_anchor_method_id="static_training_average_map",
        reference_anchor_method_id="full_state_model_oracle_map",
    ),
    MajorComparisonRegistration(
        comparison_id="t2_4_2_timing_fault_sensitivity",
        code_path="cnn_fpga/runtime/timing_fault_model.py",
        comparison_kind="timing_sensitivity",
        lifecycle="active",
        method_ids=("nominal_runtime", "timing_fault_runtime"),
        standard_binning_policy="not_applicable",
        rationale="runtime fault sensitivity, not a decoder-method comparison",
    ),
    MajorComparisonRegistration(
        comparison_id="legacy_p4_frozen_software_hil",
        code_path="cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py",
        comparison_kind="legacy_parameter_estimator_comparison",
        lifecycle="frozen_legacy",
        method_ids=("static_linear", "window_variance", "ekf", "cnn_fpga"),
        standard_binning_policy="not_applicable",
        rationale=(
            "frozen historical slow-loop parameter-estimator benchmark; static_linear is "
            "not renamed or reinterpreted as a GKP standard decoder"
        ),
    ),
)


def standard_binning_logical_class(
    centered_syndrome: ArrayLike,
    *,
    lattice: float = LATTICE_CONST,
) -> NDArray[np.int64]:
    """Return the fixed even-even decision from an observed 2D syndrome.

    The input must have shape ``(..., 2)`` and lie in the half-open centered
    cell.  No displacement, cell index, drift state, noise parameter, history,
    or oracle field is accepted.
    """

    spacing = float(lattice)
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("lattice must be finite and positive")
    values = np.asarray(centered_syndrome, dtype=np.float64)
    if values.ndim == 0 or values.shape[-1] != 2:
        raise ValueError("centered_syndrome must have shape (..., 2)")
    if not np.all(np.isfinite(values)):
        raise ValueError("centered_syndrome must contain only finite values")
    half = spacing / 2.0
    if np.any(values < -half) or np.any(values >= half):
        raise ValueError(
            "centered_syndrome must lie in the half-open interval [-lattice/2, lattice/2)"
        )
    return np.zeros(values.shape[:-1], dtype=np.int64)


def standard_binning_paired_outcomes(
    displacements: ArrayLike,
    *,
    lattice: float = LATTICE_CONST,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.bool_]]:
    """Build observed decisions, hidden truth labels, and paired failures.

    Truth construction is confined to this evaluator.  The decision is made
    exclusively by :func:`standard_binning_logical_class` from the centered
    syndrome buffer.
    """

    values = np.asarray(displacements, dtype=np.float64)
    if values.ndim == 0 or values.shape[-1] != 2:
        raise ValueError("displacements must have shape (..., 2)")
    if not np.all(np.isfinite(values)):
        raise ValueError("displacements must contain only finite values")
    q_result = standard_binning_1d(values[..., 0], lattice=lattice)
    p_result = standard_binning_1d(values[..., 1], lattice=lattice)
    syndrome = np.stack(
        (np.asarray(q_result.syndrome), np.asarray(p_result.syndrome)),
        axis=-1,
    )
    truth = (
        2 * np.asarray(q_result.logical_parity, dtype=np.int64)
        + np.asarray(p_result.logical_parity, dtype=np.int64)
    )
    decision = standard_binning_logical_class(syndrome, lattice=lattice)
    return decision, truth, decision != truth


def major_comparison_registry() -> tuple[MajorComparisonRegistration, ...]:
    return _MAJOR_COMPARISONS


def validate_major_comparison_registry(
    registrations: Sequence[MajorComparisonRegistration] | None = None,
) -> tuple[str, ...]:
    """Fail closed if an algorithm-comparison schema omits standard binning."""

    entries = tuple(_MAJOR_COMPARISONS if registrations is None else registrations)
    if not entries:
        raise ValueError("major comparison registry must not be empty")
    comparison_ids = [entry.comparison_id for entry in entries]
    if len(set(comparison_ids)) != len(comparison_ids):
        raise ValueError("major comparison registry contains duplicate comparison_id values")
    gates: list[str] = []
    for entry in entries:
        if not entry.rationale.strip():
            raise ValueError(f"{entry.comparison_id} must provide an explicit rationale")
        occurrences = entry.method_ids.count(STANDARD_BINNING_ID)
        if entry.standard_binning_policy == "required":
            if entry.comparison_kind != "decoder_algorithm_comparison":
                raise ValueError(
                    f"{entry.comparison_id} requires standard binning but is not a decoder comparison"
                )
            if occurrences != 1:
                raise ValueError(
                    f"{entry.comparison_id} must contain standard_binning exactly once"
                )
        elif occurrences:
            raise ValueError(
                f"{entry.comparison_id} marks standard binning not applicable but includes it"
            )
        if entry.comparison_kind == "decoder_algorithm_comparison":
            anchor = entry.static_anchor_method_id
            if anchor is None or not anchor.strip():
                raise ValueError(
                    f"{entry.comparison_id} must declare one task-specific static anchor"
                )
            if entry.method_ids.count(anchor) != 1:
                raise ValueError(
                    f"{entry.comparison_id} must contain its static anchor exactly once"
                )
            reference = entry.reference_anchor_method_id
            if reference is None or not reference.strip():
                raise ValueError(
                    f"{entry.comparison_id} must declare one task-specific reference anchor"
                )
            if entry.method_ids.count(reference) != 1:
                raise ValueError(
                    f"{entry.comparison_id} must contain its reference anchor exactly once"
                )
        elif (
            entry.static_anchor_method_id is not None
            or entry.reference_anchor_method_id is not None
        ):
            raise ValueError(
                f"{entry.comparison_id} is not a decoder comparison and must not declare "
                "static/reference anchors"
            )
        gates.append(f"registry:{entry.comparison_id}")
    if not any(
        entry.lifecycle == "active" and entry.standard_binning_policy == "required"
        for entry in entries
    ):
        raise ValueError("registry must contain an active decoder comparison with standard binning")
    return tuple(gates)


def _source_sha256(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_standard_binning_validation() -> dict[str, object]:
    """Run the production integration audit and return a JSON-ready payload."""

    from cnn_fpga.benchmark.adaptive_drift_alignment import run_adaptive_drift_alignment

    registry_gates = validate_major_comparison_registry()
    alignment = run_adaptive_drift_alignment()
    active_registration = next(
        entry
        for entry in _MAJOR_COMPARISONS
        if entry.comparison_id == "t1_3_4_adaptive_drift_alignment"
    )
    if alignment.comparison_method_ids != active_registration.method_ids:
        raise AssertionError("adaptive comparison methods drifted from the registered schema")
    sample_count = sum(record.evaluation_samples for record in alignment.records)
    if sample_count != alignment.paired_samples:
        raise AssertionError("adaptive comparison sample accounting mismatch")
    standard_failures = sum(record.standard_failures for record in alignment.records)
    if standard_failures / sample_count != alignment.standard_error_rate:
        raise AssertionError("standard-binning failure accounting mismatch")

    source_paths = (
        Path(__file__),
        Path(__file__).with_name("adaptive_drift_alignment.py"),
        Path(__file__).parents[2] / "physics" / "ideal_gkp_decoder.py",
    )
    gates = [
        *registry_gates,
        "descriptor:no_hidden_truth_inputs",
        "adaptive:method_schema_bound",
        "adaptive:paired_failure_accounting",
        "adaptive:standard_vs_static_paired_difference_resolved",
        "legacy_p4:semantic_non_alias",
    ]
    if STANDARD_BINNING_DESCRIPTOR.hidden_truth_inputs:
        raise AssertionError("standard-binning descriptor must not expose hidden truth")
    standard_minus_static = alignment.standard_gap.static_minus_dual
    if not (
        standard_minus_static.ci_high < 0.0
        or standard_minus_static.ci_low > 0.0
    ):
        raise AssertionError("standard-vs-static paired difference is not statistically resolved")
    payload = {
        "schema_version": "t3.1.1-standard-binning-v1",
        "task_id": "T3.1.1",
        "status": "PASS",
        "evidence_scope": "paired_synthetic_ideal_square_gkp_decoder_integration",
        "implementation_sha256": _source_sha256(source_paths),
        "baseline_descriptor": asdict(STANDARD_BINNING_DESCRIPTOR),
        "major_comparison_registry": [asdict(entry) for entry in _MAJOR_COMPARISONS],
        "adaptive_alignment": {
            "method_ids": list(alignment.comparison_method_ids),
            "paired_samples": alignment.paired_samples,
            "trace_sha256": alignment.trace_sha256,
            "standard_binning_error_rate": alignment.standard_error_rate,
            "static_calibration_map_error_rate": alignment.static_error_rate,
            "window_variance_map_error_rate": alignment.window_error_rate,
            "ekf_map_error_rate": alignment.ekf_error_rate,
            "full_state_model_oracle_map_error_rate": alignment.oracle_error_rate,
            "standard_minus_static": asdict(alignment.standard_gap.static_minus_dual),
            "standard_only_failure_count": alignment.standard_gap.static_only_failure_count,
            "static_only_failure_count": alignment.standard_gap.dual_only_failure_count,
            "mcnemar_z": alignment.standard_gap.mcnemar_z,
            "counterevidence": (
                (
                    "standard binning is better than the current static row on this registered "
                    "trace; no dominance assumption is imposed"
                    if standard_minus_static.estimate < 0.0
                    else "the current static row is better than standard binning on this registered "
                    "trace; no cross-scenario dominance assumption is imposed"
                )
            ),
        },
        "gate_summary": {
            "passed": len(gates),
            "failed": 0,
            "gates": gates,
        },
        "claim_boundary": {
            "allowed": (
                "standard binning is an explicit, paired, no-tuning row in the current "
                "synthetic ideal square-GKP decoder comparison; any observed ranking against "
                "the current static row remains scenario-specific"
            ),
            "forbidden": (
                "legacy static_linear is standard binning, or the integration is a "
                "finite-energy/protocol/FPGA/quantum-hardware performance result"
            ),
        },
    }
    # Normalize dataclass tuples to their on-disk JSON representation so the
    # in-memory validator and persisted artifact are exactly comparable.
    return json.loads(json.dumps(payload, ensure_ascii=False))


def write_standard_binning_validation(path: str | Path) -> dict[str, object]:
    payload = build_standard_binning_validation()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate T3.1.1 standard-binning integration")
    parser.add_argument(
        "--output",
        default="docs/t3_1_1_standard_binning_validation.json",
    )
    arguments = parser.parse_args()
    result = write_standard_binning_validation(arguments.output)
    print(json.dumps(result["gate_summary"], ensure_ascii=False))


__all__ = [
    "STANDARD_BINNING_ID",
    "DecoderBaselineDescriptor",
    "MajorComparisonRegistration",
    "STANDARD_BINNING_DESCRIPTOR",
    "standard_binning_logical_class",
    "standard_binning_paired_outcomes",
    "major_comparison_registry",
    "validate_major_comparison_registry",
    "build_standard_binning_validation",
    "write_standard_binning_validation",
]
