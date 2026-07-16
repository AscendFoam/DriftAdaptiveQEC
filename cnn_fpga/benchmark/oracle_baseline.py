"""T3.1.3 nondeployable full-state/regime/leakage oracle integration.

Two truth-qualified upper references are kept separate:

* the periodic Gaussian-mixture decoder oracle reuses ``physics.oracle_map``
  and consumes the exact per-sample ``DriftState``;
* the protocol leakage oracle exposes only a perfect hidden-leakage *flag*.
  It never invents a Pauli correction for leakage.  Its optimistic and
  conservative error envelopes are reported separately.

Neither interface is a deployable decoder input path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cnn_fpga.benchmark.static_map_baseline import (
    STATIC_MAP_ID,
    fit_static_map_from_training_states,
    static_map_logical_class,
)
from physics.constants import LATTICE_CONST
from physics.drift_processes import DriftState, sample_displacements
from physics.ideal_gkp_decoder import standard_binning_1d
from physics.oracle_map import OracleMAPResult, oracle_map_2d
from physics.syndrome_stream import (
    MODEL_SCOPE,
    SyndromeStreamConfig,
    SyndromeTruthStep,
    generate_syndrome_stream,
)


FULL_STATE_ORACLE_ID = "full_state_model_oracle_map"
LEAKAGE_FLAG_ORACLE_ID = "hidden_leakage_flag_oracle"
LeakageKind = Literal["none", "f", "higher"]


@dataclass(frozen=True)
class OracleBaselineDescriptor:
    baseline_id: str = FULL_STATE_ORACLE_ID
    label: str = "Full-state model oracle MAP"
    task_owner: str = "T3.1.3"
    comparison_role: str = "nondeployable_decoder_model_upper_reference"
    deployable: bool = False
    observed_inputs: tuple[str, ...] = (
        "centered_modular_syndrome_q",
        "centered_modular_syndrome_p",
    )
    hidden_inputs: tuple[str, ...] = (
        "exact_DriftState_mean_covariance_outlier_mixture",
        "hidden_regime",
        "burst_active",
        "hidden_leakage_kind_for_flag_lane",
    )
    forbidden_deployable_aliases: tuple[str, ...] = (
        "oracle_delayed",
        "teacher_mode",
        "target_params",
    )
    evidence_scope: str = "nondeployable_assumed_model_upper_reference"


ORACLE_BASELINE_DESCRIPTOR = OracleBaselineDescriptor()


@dataclass(frozen=True)
class OracleHiddenContext:
    drift_state: DriftState
    hidden_regime: str
    leakage_kind: LeakageKind
    truth_scope: str = "simulator_hidden_truth_not_deployable_input"

    def __post_init__(self) -> None:
        if not isinstance(self.drift_state, DriftState):
            raise TypeError("drift_state must be a DriftState")
        if self.hidden_regime != self.drift_state.regime:
            raise ValueError("hidden_regime must match drift_state.regime")
        if self.leakage_kind not in {"none", "f", "higher"}:
            raise ValueError("leakage_kind must be 'none', 'f', or 'higher'")
        if self.truth_scope != "simulator_hidden_truth_not_deployable_input":
            raise ValueError("truth_scope must identify simulator hidden truth")

    @classmethod
    def from_truth_step(cls, truth: SyndromeTruthStep) -> "OracleHiddenContext":
        if not isinstance(truth, SyndromeTruthStep):
            raise TypeError("truth must be a SyndromeTruthStep, not a deployable record")
        return cls(
            drift_state=truth.drift_state,
            hidden_regime=truth.hidden_regime,
            leakage_kind=truth.leakage_kind,  # type: ignore[arg-type]
            truth_scope=truth.truth_scope,
        )


@dataclass(frozen=True)
class OracleUpperReferenceDecision:
    reference_id: str
    logical_class: int | None
    logical_action: str
    erasure_flag: bool
    hidden_regime: str
    leakage_kind: LeakageKind
    map_result: OracleMAPResult | None
    deployable: bool = False
    evidence_scope: str = "nondeployable_truth_qualified_oracle_decision"


def oracle_upper_reference_decision(
    centered_syndrome: ArrayLike,
    context: OracleHiddenContext,
) -> OracleUpperReferenceDecision:
    """Decode normal cycles and emit a truth-only erasure flag on leakage."""

    if not isinstance(context, OracleHiddenContext):
        raise TypeError("context must be an OracleHiddenContext")
    syndrome = np.asarray(centered_syndrome, dtype=float)
    if syndrome.shape != (2,) or not np.all(np.isfinite(syndrome)):
        raise ValueError("centered_syndrome must contain exactly two finite values")
    if context.leakage_kind != "none":
        return OracleUpperReferenceDecision(
            reference_id=LEAKAGE_FLAG_ORACLE_ID,
            logical_class=None,
            logical_action="FLAG_LEAKAGE",
            erasure_flag=True,
            hidden_regime=context.hidden_regime,
            leakage_kind=context.leakage_kind,
            map_result=None,
        )
    result = oracle_map_2d(syndrome, context.drift_state)
    return OracleUpperReferenceDecision(
        reference_id=FULL_STATE_ORACLE_ID,
        logical_class=int(result.logical_class),
        logical_action=str(result.logical_action),
        erasure_flag=False,
        hidden_regime=context.hidden_regime,
        leakage_kind=context.leakage_kind,
        map_result=result,
    )


def validate_oracle_major_comparisons() -> tuple[str, ...]:
    """Validate schemas that explicitly select the full-state model oracle.

    Other decoder tasks can use a task-specific nondeployable reference (for
    example exact episode truth or an ideal sBs branch).  The central registry
    validates that role; this module only owns ``full_state_model_oracle_map``.
    """

    from cnn_fpga.benchmark.standard_binning_baseline import (
        major_comparison_registry,
        validate_major_comparison_registry,
    )

    validate_major_comparison_registry()
    gates: list[str] = []
    for entry in major_comparison_registry():
        if entry.reference_anchor_method_id != FULL_STATE_ORACLE_ID:
            continue
        if entry.method_ids.count(FULL_STATE_ORACLE_ID) != 1:
            raise ValueError(
                f"{entry.comparison_id} must contain {FULL_STATE_ORACLE_ID} exactly once"
            )
        gates.append(f"oracle_registry:{entry.comparison_id}")
    if not gates:
        raise ValueError("no decoder schemas select the full-state model oracle")
    return tuple(gates)


def _scenario_states() -> tuple[tuple[str, DriftState], ...]:
    lam = LATTICE_CONST
    return (
        (
            "quiet",
            DriftState(
                mu_q=0.0,
                mu_p=0.0,
                sigma_q=0.18 * lam,
                sigma_p=0.10 * lam,
                regime="quiet",
                source="t3.1.3-regime-matrix",
            ),
        ),
        (
            "shifted",
            DriftState(
                mu_q=0.22 * lam,
                mu_p=-0.16 * lam,
                sigma_q=0.22 * lam,
                sigma_p=0.13 * lam,
                rho=0.20,
                regime="shifted",
                source="t3.1.3-regime-matrix",
            ),
        ),
        (
            "correlated",
            DriftState(
                mu_q=-0.17 * lam,
                mu_p=0.19 * lam,
                sigma_q=0.25 * lam,
                sigma_p=0.16 * lam,
                rho=0.75,
                regime="correlated",
                source="t3.1.3-regime-matrix",
            ),
        ),
        (
            "burst_mixture",
            DriftState(
                mu_q=0.12 * lam,
                mu_p=0.07 * lam,
                sigma_q=0.23 * lam,
                sigma_p=0.15 * lam,
                rho=-0.40,
                p_outlier=0.12,
                outlier_scale=2.5,
                burst_active=True,
                regime="burst",
                source="t3.1.3-regime-matrix",
            ),
        ),
    )


def _paired_interval(
    first_failures: NDArray[np.bool_],
    second_failures: NDArray[np.bool_],
) -> dict[str, float | int]:
    if first_failures.shape != second_failures.shape or first_failures.ndim != 1:
        raise ValueError("paired failure arrays must be one-dimensional and aligned")
    difference = first_failures.astype(float) - second_failures.astype(float)
    standard_error = float(np.std(difference, ddof=1) / math.sqrt(len(difference)))
    margin = NormalDist().inv_cdf(0.975) * standard_error
    first_only = int(np.sum(first_failures & ~second_failures))
    second_only = int(np.sum(~first_failures & second_failures))
    estimate = float(np.mean(difference))
    return {
        "estimate": estimate,
        "standard_error": standard_error,
        "ci_low": estimate - margin,
        "ci_high": estimate + margin,
        "first_only_failure_count": first_only,
        "second_only_failure_count": second_only,
    }


def _truth_and_syndrome(
    displacements: NDArray[np.float64],
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    q_result = standard_binning_1d(displacements[:, 0])
    p_result = standard_binning_1d(displacements[:, 1])
    truth = (
        2 * np.asarray(q_result.logical_parity, dtype=np.int64)
        + np.asarray(p_result.logical_parity, dtype=np.int64)
    )
    syndrome = np.column_stack((q_result.syndrome, p_result.syndrome))
    return truth, syndrome


def _oracle_batch(
    syndrome: NDArray[np.float64],
    state: DriftState,
) -> NDArray[np.int64]:
    chunks: list[NDArray[np.int64]] = []
    for start in range(0, len(syndrome), 2_000):
        result = oracle_map_2d(syndrome[start : start + 2_000], state)
        chunks.append(np.asarray(result.logical_class, dtype=np.int64))
    return np.concatenate(chunks)


def _source_sha256(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_oracle_validation() -> tuple[dict[str, object], list[dict[str, object]]]:
    """Run the regime matrix and protocol leakage-flag audit."""

    oracle_registry_gates = validate_oracle_major_comparisons()
    scenarios = _scenario_states()
    static_parameters = fit_static_map_from_training_states(
        tuple(state for _, state in scenarios),
        training_protocol_id="t3.1.3-four-regime-static-anchor-v1",
    )
    rows: list[dict[str, object]] = []
    all_static_failures: list[NDArray[np.bool_]] = []
    all_oracle_failures: list[NDArray[np.bool_]] = []
    evaluation_seeds = (20260731, 20260732, 20260733, 20260734)
    samples_per_row = 20_000
    for scenario_index, (scenario_name, state) in enumerate(scenarios):
        for seed in evaluation_seeds:
            rng = np.random.default_rng(seed + 100 * scenario_index)
            displacement, _ = sample_displacements(state, samples_per_row, rng=rng)
            truth, syndrome = _truth_and_syndrome(displacement)
            static_decision = static_map_logical_class(syndrome, static_parameters)
            oracle_decision = _oracle_batch(syndrome, state)
            standard_failure = truth != 0
            static_failure = static_decision != truth
            oracle_failure = oracle_decision != truth
            interval = _paired_interval(static_failure, oracle_failure)
            all_static_failures.append(static_failure)
            all_oracle_failures.append(oracle_failure)
            trace_digest = hashlib.sha256()
            trace_digest.update(np.ascontiguousarray(displacement, dtype="<f8").tobytes())
            rows.append(
                {
                    "lane": "regime_matrix",
                    "scenario": scenario_name,
                    "evaluation_seed": seed,
                    "samples": samples_per_row,
                    "trace_sha256": trace_digest.hexdigest(),
                    "regime": state.regime,
                    "burst_active": state.burst_active,
                    "p_outlier": state.p_outlier,
                    "standard_error_rate": float(np.mean(standard_failure)),
                    "static_error_rate": float(np.mean(static_failure)),
                    "oracle_error_rate": float(np.mean(oracle_failure)),
                    "static_minus_oracle": interval["estimate"],
                    "static_minus_oracle_ci_low": interval["ci_low"],
                    "static_minus_oracle_ci_high": interval["ci_high"],
                    "leakage_cycles": "",
                    "leakage_flag_sensitivity": "",
                    "leakage_flag_specificity": "",
                    "optimistic_flagged_error_lower_bound": "",
                    "conservative_leakage_as_failure_rate": "",
                }
            )

    # Protocol-truth leakage lane. The oracle receives the exact hidden kind
    # only to emit an erasure flag; it does not invent a corrective Pauli.
    leakage_total = 0
    leakage_flagged = 0
    nonleak_total = 0
    nonleak_not_flagged = 0
    nonleak_errors = 0
    leakage_rows: list[dict[str, object]] = []
    for seed in (20260741, 20260742, 20260743, 20260744):
        base = scenarios[1][1]
        states = tuple(
            replace(
                base,
                step=index,
                time=float(index),
                burst_active=(index % 7) < 3,
                regime="burst" if (index % 7) < 3 else "quiet",
                source="t3.1.3-leakage-stream",
            )
            for index in range(2_000)
        )
        stream = generate_syndrome_stream(
            states,
            config=SyndromeStreamConfig(
                seed=seed,
                measurement_sigma=(0.01, 0.01),
                base_leakage_probability=0.03,
                burst_leakage_bonus=0.12,
                loss_leakage_scale=0.0,
                higher_leakage_fraction=0.5,
                higher_leakage_mean_duration=5.0,
            ),
        )
        seed_leakage = 0
        seed_flagged = 0
        seed_nonleak = 0
        seed_nonleak_errors = 0
        for step in stream.steps:
            context = OracleHiddenContext.from_truth_step(step.truth)
            decision = oracle_upper_reference_decision(
                step.observed.residual_syndrome,
                context,
            )
            if context.leakage_kind != "none":
                seed_leakage += 1
                seed_flagged += int(decision.erasure_flag)
                if decision.logical_class is not None or decision.logical_action != "FLAG_LEAKAGE":
                    raise AssertionError("leakage oracle issued a fabricated logical correction")
            else:
                seed_nonleak += 1
                nonleak_not_flagged += int(not decision.erasure_flag)
                truth_class = 2 * step.truth.logical_increment[0] + step.truth.logical_increment[1]
                seed_nonleak_errors += int(decision.logical_class != truth_class)
        leakage_total += seed_leakage
        leakage_flagged += seed_flagged
        nonleak_total += seed_nonleak
        nonleak_errors += seed_nonleak_errors
        leakage_rows.append(
            {
                "lane": "protocol_leakage_flag",
                "scenario": "mixed_burst_leakage",
                "evaluation_seed": seed,
                "samples": len(stream.steps),
                "trace_sha256": hashlib.sha256(
                    json.dumps(stream.truth_records(), default=str).encode("utf-8")
                ).hexdigest(),
                "regime": "mixed",
                "burst_active": "mixed",
                "p_outlier": base.p_outlier,
                "standard_error_rate": "",
                "static_error_rate": "",
                "oracle_error_rate": "",
                "static_minus_oracle": "",
                "static_minus_oracle_ci_low": "",
                "static_minus_oracle_ci_high": "",
                "leakage_cycles": seed_leakage,
                "leakage_flag_sensitivity": seed_flagged / seed_leakage,
                "leakage_flag_specificity": 1.0,
                "optimistic_flagged_error_lower_bound": seed_nonleak_errors / len(stream.steps),
                "conservative_leakage_as_failure_rate": (
                    seed_nonleak_errors + seed_leakage
                )
                / len(stream.steps),
            }
        )
    rows.extend(leakage_rows)

    static_all = np.concatenate(all_static_failures)
    oracle_all = np.concatenate(all_oracle_failures)
    aggregate_interval = _paired_interval(static_all, oracle_all)
    regime_rows = [row for row in rows if row["lane"] == "regime_matrix"]
    gates = {
        "descriptor_explicitly_nondeployable": not ORACLE_BASELINE_DESCRIPTOR.deployable,
        "hidden_inputs_explicit": len(ORACLE_BASELINE_DESCRIPTOR.hidden_inputs) >= 4,
        "full_state_oracle_present_in_declared_schemas": len(oracle_registry_gates) >= 2,
        "all_regime_rows_static_oracle_ci_positive": all(
            float(row["static_minus_oracle_ci_low"]) > 0.0 for row in regime_rows
        ),
        "aggregate_static_oracle_gap_positive": float(aggregate_interval["ci_low"]) > 0.0,
        "regime_matrix_includes_burst_and_outlier": any(
            bool(row["burst_active"]) and float(row["p_outlier"]) > 0.0
            for row in regime_rows
        ),
        "leakage_flag_sensitivity_one": leakage_flagged == leakage_total and leakage_total > 0,
        "leakage_flag_specificity_one": nonleak_not_flagged == nonleak_total and nonleak_total > 0,
        "leakage_lane_reports_nonzero_cost_envelope": 0 < nonleak_errors < nonleak_total,
        "legacy_oracle_delayed_not_canonical_oracle": "oracle_delayed"
        in ORACLE_BASELINE_DESCRIPTOR.forbidden_deployable_aliases,
    }
    if not all(gates.values()):
        failed = [name for name, value in gates.items() if not value]
        raise AssertionError(f"oracle validation gates failed: {failed}")
    source_paths = (
        Path(__file__),
        Path(__file__).parents[2] / "physics" / "oracle_map.py",
        Path(__file__).with_name("adaptive_drift_alignment.py"),
        Path(__file__).with_name("standard_binning_baseline.py"),
    )
    payload: dict[str, object] = {
        "schema_version": "t3.1.3-oracle-integration-v2",
        "task_id": "T3.1.3",
        "status": "PASS",
        "evidence_scope": "nondeployable_synthetic_model_oracle_upper_reference",
        "implementation_sha256": _source_sha256(source_paths),
        "descriptor": asdict(ORACLE_BASELINE_DESCRIPTOR),
        "oracle_registry_gates": list(oracle_registry_gates),
        "regime_matrix": {
            "scenarios": len(scenarios),
            "evaluation_seeds": len(evaluation_seeds),
            "samples": int(static_all.size),
            "static_error_rate": float(np.mean(static_all)),
            "oracle_error_rate": float(np.mean(oracle_all)),
            "static_minus_oracle": aggregate_interval,
            "static_training_state_sha256": static_parameters.training_state_sha256,
        },
        "protocol_leakage_flag": {
            "model_scope": MODEL_SCOPE,
            "cycles": leakage_total + nonleak_total,
            "leakage_cycles": leakage_total,
            "nonleakage_cycles": nonleak_total,
            "flag_sensitivity": leakage_flagged / leakage_total,
            "flag_specificity": nonleak_not_flagged / nonleak_total,
            "nonleakage_map_errors": nonleak_errors,
            "optimistic_perfect_erasure_lower_bound": nonleak_errors
            / (leakage_total + nonleak_total),
            "conservative_leakage_as_failure_rate": (nonleak_errors + leakage_total)
            / (leakage_total + nonleak_total),
            "cost_interpretation": (
                "the interval brackets unknown leakage recovery cost; flagged cycles are not "
                "silently counted as corrected"
            ),
        },
        "gate_summary": {
            "passed": len(gates),
            "failed": 0,
            "gates": gates,
        },
        "claim_boundary": {
            "allowed": (
                "exact-state periodic-mixture oracle is a nondeployable assumed-model upper "
                "reference; hidden leakage supplies only a perfect erasure flag with explicit cost bounds"
            ),
            "forbidden": (
                "oracle is deployable, oracle_delayed is the decoder oracle, leakage is perfectly "
                "corrected for free, or this is a finite-energy/device/channel-recovery optimum"
            ),
        },
    }
    return json.loads(json.dumps(payload, ensure_ascii=False)), rows


def write_oracle_validation(
    json_path: str | Path,
    csv_path: str | Path,
) -> dict[str, object]:
    payload, rows = build_oracle_validation()
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

    parser = argparse.ArgumentParser(description="Validate T3.1.3 oracle integration")
    parser.add_argument("--json", default="docs/t3_1_3_oracle_validation.json")
    parser.add_argument("--csv", default="docs/t3_1_3_oracle_source_data.csv")
    arguments = parser.parse_args()
    result = write_oracle_validation(arguments.json, arguments.csv)
    print(json.dumps(result["gate_summary"], ensure_ascii=False))


__all__ = [
    "FULL_STATE_ORACLE_ID",
    "LEAKAGE_FLAG_ORACLE_ID",
    "OracleBaselineDescriptor",
    "ORACLE_BASELINE_DESCRIPTOR",
    "OracleHiddenContext",
    "OracleUpperReferenceDecision",
    "oracle_upper_reference_decision",
    "validate_oracle_major_comparisons",
    "build_oracle_validation",
    "write_oracle_validation",
]
