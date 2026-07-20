"""T5.3.2 leakage-inclusive fidelity and short-time rate report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from cnn_fpga.benchmark.logical_channel_reconstruction import (
    DEFAULT_ARTIFACT as PARENT_ARTIFACT,
    implementation_sha256 as parent_implementation_sha256,
    validate_artifact_payload as validate_parent_payload,
)
from physics.fock_logical_channel import STATE_LABELS, logical_eigenstate_density
from physics.logical_channel_fidelity import (
    cptni_identity_fidelity,
    short_time_effective_depolarization,
    terminal_cutoff_interval,
)


TASK_ID = "T5.3.2"
CONTRACT_ID = "T532-CPTNI-FIDELITY-SHORT-TIME-RATE-V1"
DEFAULT_ARTIFACT = Path("docs/t5_3_2_logical_channel_fidelity.json")
DEFAULT_SOURCE_DATA = Path("docs/t5_3_2_logical_channel_fidelity_source_data.csv")
PAPER_SOURCE = Path(
    "relative_papers/Real-time_quantum_error_correction_beyond_break-even/"
    "Real-time_quantum_error_correction_beyond_break-even.md"
)
PAPER_FRAGMENTS = (
    "The average channel fidelity of a quantum channel",
    "often called the entanglement fidelity",
    "where $ \\Gamma $ is an effective depolarization rate",
)
TERMINAL_CUTOFFS = (36, 40)


@dataclass(frozen=True)
class FidelityReportConfig:
    parent_artifact: str = PARENT_ARTIFACT.as_posix()
    terminal_cutoffs: tuple[int, int] = TERMINAL_CUTOFFS
    primary_rate_estimator: str = "three_point_second_order_forward_difference"

    def __post_init__(self) -> None:
        if Path(self.parent_artifact).as_posix() != PARENT_ARTIFACT.as_posix():
            raise ValueError("formal T5.3.2 must remain bound to the T5.3.1 artifact")
        cutoffs = tuple(int(value) for value in self.terminal_cutoffs)
        if cutoffs != TERMINAL_CUTOFFS:
            raise ValueError("formal terminal cutoffs must remain 36 and 40")
        object.__setattr__(self, "terminal_cutoffs", cutoffs)
        if self.primary_rate_estimator != "three_point_second_order_forward_difference":
            raise ValueError("primary short-time estimator must remain frozen")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def implementation_sha256() -> str:
    digest = hashlib.sha256()
    paths = (
        Path(__file__).resolve(),
        Path(__file__).resolve().parents[2] / "physics" / "logical_channel_fidelity.py",
    )
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _deep_close(left: Any, right: Any, *, tolerance: float = 2.0e-9) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _deep_close(left[key], right[key], tolerance=tolerance) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _deep_close(a, b, tolerance=tolerance) for a, b in zip(left, right, strict=True)
        )
    if left is None or right is None or isinstance(left, (str, bool)) or isinstance(right, (str, bool)):
        return left == right
    try:
        return bool(np.isclose(float(left), float(right), rtol=2.0e-9, atol=tolerance))
    except (TypeError, ValueError):
        return left == right


def _load_parent() -> dict[str, Any]:
    payload = json.loads(PARENT_ARTIFACT.read_text(encoding="utf-8"))
    validate_parent_payload(payload)
    return payload


def _parent_audit(parent: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": PARENT_ARTIFACT.as_posix(),
        "sha256": _sha256(PARENT_ARTIFACT),
        "task_id": parent.get("task_id"),
        "status": parent.get("status"),
        "gate_count": len(parent.get("gates", {})),
        "all_gates_passed": all(parent.get("gates", {}).values()),
        "stored_implementation_sha256": parent.get("implementation_sha256"),
        "live_implementation_sha256": parent_implementation_sha256(),
        "implementation_hash_matches": (
            parent.get("implementation_sha256") == parent_implementation_sha256()
        ),
    }


def _paper_audit() -> dict[str, Any]:
    text = PAPER_SOURCE.read_text(encoding="utf-8")
    return {
        "path": PAPER_SOURCE.as_posix(),
        "sha256": _sha256(PAPER_SOURCE),
        "required_fragments": list(PAPER_FRAGMENTS),
        "fragment_matches": {fragment: fragment in text for fragment in PAPER_FRAGMENTS},
        "sections": ["main text average channel fidelity and leading-order rate", "Supplement K equations S22-S28"],
        "source_scope": "TP experimental formula motivates six-state metric; CPTNI leakage extension is derived and independently tested in this project",
    }


def _raw_outputs(parent_lane: Mapping[str, Any], cycle_index: int) -> dict[str, np.ndarray]:
    real = np.asarray(parent_lane["projected_output_real"][cycle_index], dtype=np.float64)
    imag = np.asarray(parent_lane["projected_output_imag"][cycle_index], dtype=np.float64)
    matrices = real + 1.0j * imag
    return {label: matrices[index] for index, label in enumerate(STATE_LABELS)}


def _derive_lanes(parent: Mapping[str, Any]) -> dict[str, Any]:
    derived: dict[str, Any] = {}
    for lane_id, parent_lane in parent["lanes"].items():
        cycle_metrics = []
        average_curve = []
        for cycle_index, tomography in enumerate(parent_lane["tomography"]):
            metrics = cptni_identity_fidelity(
                tomography["ptm"], outputs=_raw_outputs(parent_lane, cycle_index)
            )
            cycle_metrics.append(metrics.to_dict())
            average_curve.append(metrics.average_fidelity)
        short_time = short_time_effective_depolarization(
            parent_lane["time_us"], average_curve
        ).to_dict()
        derived[lane_id] = {
            "lane_id": lane_id,
            "parent_lane_config": dict(parent_lane["config"]),
            "noise_profile": parent_lane["noise_profile"],
            "cycles": list(parent_lane["cycles"]),
            "time_us": list(parent_lane["time_us"]),
            "cycle_metrics": cycle_metrics,
            "short_time_effective_depolarization": short_time,
            "final_metrics": dict(cycle_metrics[-1]),
        }
    return derived


def _terminal_intervals(lanes: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    lower, higher = TERMINAL_CUTOFFS
    metric_getters = {
        "final_entanglement_fidelity": lambda lane: lane["final_metrics"]["entanglement_fidelity"],
        "final_average_fidelity": lambda lane: lane["final_metrics"]["average_fidelity"],
        "final_mean_code_survival": lambda lane: lane["final_metrics"]["mean_code_survival"],
        "short_time_primary_rate_per_us": lambda lane: lane["short_time_effective_depolarization"]["primary_rate_per_us"],
        "short_time_discretization_spread_per_us": lambda lane: lane["short_time_effective_depolarization"]["discretization_spread_per_us"],
    }
    for noise in ("high", "medium", "low"):
        for mode in ("qec_off", "qec_on"):
            low_lane = lanes[f"cutoff{lower}:{noise}:{mode}"]
            high_lane = lanes[f"cutoff{higher}:{noise}:{mode}"]
            for metric, getter in metric_getters.items():
                rows.append(
                    {
                        "noise_profile": noise,
                        "mode": mode,
                        "metric": metric,
                        **terminal_cutoff_interval(
                            getter(low_lane),
                            getter(high_lane),
                            lower_cutoff=lower,
                            higher_cutoff=higher,
                        ),
                    }
                )
    return rows


def _matched_differences(lanes: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for noise in ("high", "medium", "low"):
        on = lanes[f"cutoff40:{noise}:qec_on"]
        off = lanes[f"cutoff40:{noise}:qec_off"]
        rows.append(
            {
                "cutoff": 40,
                "noise_profile": noise,
                "qec_on_minus_off_final_entanglement_fidelity": float(
                    on["final_metrics"]["entanglement_fidelity"]
                    - off["final_metrics"]["entanglement_fidelity"]
                ),
                "qec_on_minus_off_final_average_fidelity": float(
                    on["final_metrics"]["average_fidelity"]
                    - off["final_metrics"]["average_fidelity"]
                ),
                "qec_on_minus_off_final_mean_survival": float(
                    on["final_metrics"]["mean_code_survival"]
                    - off["final_metrics"]["mean_code_survival"]
                ),
                "qec_on_minus_off_short_time_rate_per_us": float(
                    on["short_time_effective_depolarization"]["primary_rate_per_us"]
                    - off["short_time_effective_depolarization"]["primary_rate_per_us"]
                ),
                "ratio_or_gain_reported": False,
                "operational_boundary_claim": False,
            }
        )
    return rows


def _cutoff_direction_audit(lanes: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for noise in ("high", "medium", "low"):
        low_on = lanes[f"cutoff12:{noise}:qec_on"]["final_metrics"]["average_fidelity"]
        low_off = lanes[f"cutoff12:{noise}:qec_off"]["final_metrics"]["average_fidelity"]
        high_on = lanes[f"cutoff40:{noise}:qec_on"]["final_metrics"]["average_fidelity"]
        high_off = lanes[f"cutoff40:{noise}:qec_off"]["final_metrics"]["average_fidelity"]
        rows.append(
            {
                "noise_profile": noise,
                "cutoff12_qec_on_minus_off_final_average_fidelity": float(low_on - low_off),
                "cutoff40_qec_on_minus_off_final_average_fidelity": float(high_on - high_off),
                "direction_reversal_preserved": bool((low_on - low_off) * (high_on - high_off) < 0.0),
                "used_to_select_or_drop_data": False,
            }
        )
    return rows


def _semantic_gates(payload: Mapping[str, Any], parent: Mapping[str, Any]) -> dict[str, bool]:
    lanes = payload.get("lanes", {})
    all_points = [point for lane in lanes.values() for point in lane.get("cycle_metrics", [])]
    rates = [lane.get("short_time_effective_depolarization", {}) for lane in lanes.values()]
    recomputed_lanes = _derive_lanes(parent)
    paper = payload.get("paper_audit", {})
    audit = payload.get("parent_audit", {})
    expected_lane_ids = set(parent["lanes"])
    return {
        "t531_parent_artifact_and_implementation_are_live": bool(
            audit.get("path") == PARENT_ARTIFACT.as_posix()
            and audit.get("sha256") == _sha256(PARENT_ARTIFACT)
            and audit.get("task_id") == "T5.3.1"
            and audit.get("status") == "PASS"
            and audit.get("all_gates_passed")
            and audit.get("implementation_hash_matches")
        ),
        "primary_paper_average_fidelity_and_rate_anchors_are_live": bool(
            paper.get("path") == PAPER_SOURCE.as_posix()
            and paper.get("sha256") == _sha256(PAPER_SOURCE)
            and all(paper.get("fragment_matches", {}).values())
        ),
        "all_24_parent_lanes_and_31_cycle_points_are_derived": (
            set(lanes) == expected_lane_ids
            and len(lanes) == 24
            and all(len(lane["cycle_metrics"]) == 31 for lane in lanes.values())
        ),
        "identity_endpoint_has_unit_entanglement_and_average_fidelity": all(
            abs(lane["cycle_metrics"][0]["entanglement_fidelity"] - 1.0) <= 2.0e-9
            and abs(lane["cycle_metrics"][0]["average_fidelity"] - 1.0) <= 2.0e-9
            for lane in lanes.values()
        ) if lanes else False,
        "all_leakage_inclusive_fidelities_are_finite_and_bounded": all(
            -2.0e-9 <= point["entanglement_fidelity"] <= 1.0 + 2.0e-9
            and -2.0e-9 <= point["average_fidelity"] <= 1.0 + 2.0e-9
            and point["average_fidelity"] <= point["mean_code_survival"] + 2.0e-9
            for point in all_points
        ) if all_points else False,
        "six_state_direct_average_matches_ptm_fidelity": all(
            point["six_state_ptm_residual"] <= 2.0e-9 for point in all_points
        ) if all_points else False,
        "six_state_mean_survival_matches_ptm_r_ii": all(
            point["six_state_survival_residual"] <= 2.0e-9 for point in all_points
        ) if all_points else False,
        "tp_formula_overstatement_equals_leakage_over_three": all(
            abs(
                point["tp_formula_overstatement"]
                - (1.0 - point["mean_code_survival"]) / 3.0
            ) <= 2.0e-9
            for point in all_points
        ) if all_points else False,
        "tp_formula_is_not_silently_used_after_leakage": any(
            point["tp_formula_overstatement"] > 1.0e-3 for point in all_points
        ),
        "conditional_state_fidelity_is_diagnostic_only": all(
            point["conditional_metric_role"]
            == "diagnostic_only_not_a_linear_channel_fidelity"
            for point in all_points
        ) if all_points else False,
        "short_time_rate_uses_uniform_10us_raw_fidelity_points": all(
            rate.get("step_us") == 10.0
            and abs(rate.get("initial_average_fidelity") - 1.0) <= 2.0e-9
            for rate in rates
        ) if rates else False,
        "short_time_primary_rate_is_three_point_derivative_not_exponential_fit": all(
            rate.get("exponential_fit_used") is False
            and "three-point forward derivative" in rate.get("primary_definition", "")
            for rate in rates
        ) if rates else False,
        "rate_cycle_and_microsecond_units_are_consistent": all(
            abs(rate["primary_rate_per_cycle"] - 10.0 * rate["primary_rate_per_us"])
            <= 2.0e-12
            for rate in rates
        ) if rates else False,
        "time_discretization_spread_is_preserved_not_called_ci": all(
            rate["discretization_spread_per_us"] >= 0.0
            and "not a statistical confidence interval" in rate["uncertainty_role"]
            for rate in rates
        ) if rates else False,
        "passive_short_time_rates_meet_reliability_gate": all(
            lane["short_time_effective_depolarization"]["reliability_status"]
            == "reliable_discrete_short_time_proxy"
            for lane in lanes.values()
            if lane["parent_lane_config"]["mode"] == "qec_off"
        ) if lanes else False,
        "active_short_time_rates_are_flagged_unreliable_not_fit_away": all(
            lane["short_time_effective_depolarization"]["reliability_status"]
            == "unreliable_cycle_scale_transient"
            and lane["short_time_effective_depolarization"]["primary_lifetime_us"] is None
            and lane["short_time_effective_depolarization"]["algebraic_inverse_rate_us"] is not None
            for lane in lanes.values()
            if lane["parent_lane_config"]["mode"] == "qec_on"
        ) if lanes else False,
        "statistical_uncertainty_is_explicitly_null_for_exact_channels": bool(
            payload.get("uncertainty_contract", {}).get("statistical_standard_error") is None
            and payload.get("uncertainty_contract", {}).get("statistical_confidence_interval") is None
            and payload.get("uncertainty_contract", {}).get("reason")
            == "deterministic exact-channel evaluation has no stochastic sample population"
        ),
        "terminal_cutoff_intervals_are_systematic_not_confidence_intervals": all(
            row["lower_cutoff"] == 36
            and row["higher_cutoff"] == 40
            and row["is_confidence_interval"] is False
            and row["statistical_confidence_level"] is None
            and row["infinite_cutoff_claim"] is False
            for row in payload.get("terminal_cutoff_intervals", [])
        ) and len(payload.get("terminal_cutoff_intervals", [])) == 30,
        "low_cutoff_performance_direction_reversal_is_preserved": all(
            row["direction_reversal_preserved"] is True
            and row["used_to_select_or_drop_data"] is False
            for row in payload.get("cutoff_direction_audit", [])
        ) and len(payload.get("cutoff_direction_audit", [])) == 3,
        "matched_on_off_rows_do_not_report_gain_or_boundary": all(
            row["ratio_or_gain_reported"] is False
            and row["operational_boundary_claim"] is False
            for row in payload.get("matched_on_off_differences", [])
        ) and len(payload.get("matched_on_off_differences", [])) == 3,
        "all_derived_lane_metrics_recompute_from_parent_raw_outputs": _deep_close(
            recomputed_lanes, lanes
        ),
        "terminal_comparison_and_direction_tables_recompute": (
            _deep_close(_terminal_intervals(lanes), payload.get("terminal_cutoff_intervals", []))
            and _deep_close(_matched_differences(lanes), payload.get("matched_on_off_differences", []))
            and _deep_close(_cutoff_direction_audit(lanes), payload.get("cutoff_direction_audit", []))
        ) if lanes else False,
        "claim_boundary_remains_simulation_only": bool(
            payload.get("claim_boundary", {}).get("experimental_tomography") is False
            and payload.get("claim_boundary", {}).get("break_even_claim") is False
            and payload.get("claim_boundary", {}).get("target_hardware_measured") is False
        ),
    }


def validate_artifact_payload(payload: Mapping[str, Any]) -> dict[str, bool]:
    if payload.get("task_id") != TASK_ID or payload.get("contract_id") != CONTRACT_ID:
        raise ValueError("artifact task/contract identity mismatch")
    parent = _load_parent()
    gates = _semantic_gates(payload, parent)
    if payload.get("gates") != gates:
        raise ValueError("stored gates do not match recomputed semantic gates")
    if payload.get("required_gates") != list(gates):
        raise ValueError("required gate order/schema mismatch")
    expected = "PASS" if all(gates.values()) else "FAIL"
    if payload.get("status") != expected:
        raise ValueError("artifact status does not match gates")
    return gates


def run_report(config: FidelityReportConfig = FidelityReportConfig()) -> dict[str, Any]:
    parent = _load_parent()
    lanes = _derive_lanes(parent)
    payload: dict[str, Any] = {
        "task_id": TASK_ID,
        "contract_id": CONTRACT_ID,
        "status": "PENDING",
        "implementation_sha256": implementation_sha256(),
        "config": {
            "parent_artifact": config.parent_artifact,
            "terminal_cutoffs": list(config.terminal_cutoffs),
            "primary_rate_estimator": config.primary_rate_estimator,
        },
        "parent_audit": _parent_audit(parent),
        "paper_audit": _paper_audit(),
        "metric_contract": {
            "entanglement_fidelity": "Tr(PTM)/4 for the unnormalized CPTNI identity-target subchannel",
            "average_fidelity": "(2 F_e + R_II)/3, equal to direct Haar/six-state unnormalized overlap",
            "tp_only_formula": "(2 F_e + 1)/3 is allowed only when R_II=1",
            "leakage_erasure_semantics": "leaked weight has zero overlap with the target code state",
            "conditional_fidelity": "per-state diagnostic only; not a linear channel metric",
            "short_time_rate": "Gamma=-2 dF_avg/dt at t=0 via three-point second-order forward difference",
            "short_time_reliability": "qualified only when cycles 0-2 are nonincreasing and one/three/four-point relative spread is <=0.25",
        },
        "uncertainty_contract": {
            "statistical_standard_error": None,
            "statistical_confidence_interval": None,
            "reason": "deterministic exact-channel evaluation has no stochastic sample population",
            "numerical_systematic": "36/40 deterministic terminal-cutoff interval",
            "time_discretization_systematic": "one/three/four-point forward-rate envelope",
            "cutoff_interval_is_ci": False,
            "discretization_envelope_is_ci": False,
        },
        "lanes": lanes,
        "terminal_cutoff_intervals": _terminal_intervals(lanes),
        "matched_on_off_differences": _matched_differences(lanes),
        "cutoff_direction_audit": _cutoff_direction_audit(lanes),
        "claim_boundary": {
            "allowed": "leakage-inclusive finite-cutoff CPTNI F_e/F_avg, short-time effective rate and deterministic numerical sensitivity",
            "forbidden": [
                "TP-only fidelity formula after leakage",
                "conditional postselected channel fidelity",
                "statistical CI from deterministic states or cutoffs",
                "single-exponential replacement of average channel fidelity",
                "qualified active short-time lifetime when cycle-scale transient fails reliability",
                "operational break-even or coherence gain",
                "experimental tomography, device calibration or hardware result",
            ],
            "experimental_tomography": False,
            "break_even_claim": False,
            "target_hardware_measured": False,
        },
    }
    gates = _semantic_gates(payload, parent)
    payload["gates"] = gates
    payload["required_gates"] = list(gates)
    payload["status"] = "PASS" if all(gates.values()) else "FAIL"
    validate_artifact_payload(payload)
    return payload


def _source_rows(payload: Mapping[str, Any], parent: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "category": "contract",
            "task_id": TASK_ID,
            "contract_id": CONTRACT_ID,
            "status": payload["status"],
            "implementation_sha256": payload["implementation_sha256"],
        },
        {"category": "parent", **payload["parent_audit"]},
        {"category": "paper", **{key: value for key, value in payload["paper_audit"].items() if key != "fragment_matches"}},
    ]
    for lane_id, lane in payload["lanes"].items():
        parent_lane = parent["lanes"][lane_id]
        for cycle_index, metric in enumerate(lane["cycle_metrics"]):
            rows.append(
                {
                    "category": "channel_fidelity",
                    "lane_id": lane_id,
                    "cutoff": lane["parent_lane_config"]["cutoff"],
                    "noise_profile": lane["noise_profile"],
                    "mode": lane["parent_lane_config"]["mode"],
                    "cycle": lane["cycles"][cycle_index],
                    "time_us": lane["time_us"][cycle_index],
                    **{key: value for key, value in metric.items() if not key.endswith("_role") and key != "metric_scope"},
                }
            )
            outputs = _raw_outputs(parent_lane, cycle_index)
            for label in STATE_LABELS:
                state = logical_eigenstate_density(label)
                output = outputs[label]
                survival = float(np.trace(output).real)
                overlap = float(np.trace(state @ output).real)
                rows.append(
                    {
                        "category": "state_fidelity",
                        "lane_id": lane_id,
                        "cutoff": lane["parent_lane_config"]["cutoff"],
                        "noise_profile": lane["noise_profile"],
                        "mode": lane["parent_lane_config"]["mode"],
                        "cycle": lane["cycles"][cycle_index],
                        "time_us": lane["time_us"][cycle_index],
                        "state_label": label,
                        "survival": survival,
                        "leakage_inclusive_state_fidelity": overlap,
                        "conditional_state_fidelity": overlap / survival,
                    }
                )
        rows.append(
            {
                "category": "short_time_rate",
                "lane_id": lane_id,
                "cutoff": lane["parent_lane_config"]["cutoff"],
                "noise_profile": lane["noise_profile"],
                "mode": lane["parent_lane_config"]["mode"],
                **lane["short_time_effective_depolarization"],
            }
        )
    rows.extend({"category": "terminal_cutoff_interval", **row} for row in payload["terminal_cutoff_intervals"])
    rows.extend({"category": "matched_on_off_difference", **row} for row in payload["matched_on_off_differences"])
    rows.extend({"category": "cutoff_direction_audit", **row} for row in payload["cutoff_direction_audit"])
    rows.extend({"category": "gate", "gate": gate, "passed": passed} for gate, passed in payload["gates"].items())
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_artifacts(
    config: FidelityReportConfig = FidelityReportConfig(),
    *,
    artifact_path: Path = DEFAULT_ARTIFACT,
    source_data_path: Path = DEFAULT_SOURCE_DATA,
) -> dict[str, Any]:
    payload = run_report(config)
    parent = _load_parent()
    rows = _source_rows(payload, parent)
    _write_csv(source_data_path, rows)
    payload["source_data"] = {
        "path": source_data_path.as_posix(),
        "sha256": _sha256(source_data_path),
        "row_count": len(rows),
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--source-data", type=Path, default=DEFAULT_SOURCE_DATA)
    args = parser.parse_args()
    payload = write_artifacts(artifact_path=args.artifact, source_data_path=args.source_data)
    print(
        json.dumps(
            {
                "task_id": payload["task_id"],
                "status": payload["status"],
                "gates": f"{sum(payload['gates'].values())}/{len(payload['gates'])}",
                "source_rows": payload["source_data"]["row_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "CONTRACT_ID",
    "DEFAULT_ARTIFACT",
    "DEFAULT_SOURCE_DATA",
    "FidelityReportConfig",
    "implementation_sha256",
    "run_report",
    "validate_artifact_payload",
    "write_artifacts",
]
