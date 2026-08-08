"""T5.3.4 online-QEC and post-selection cost accounting report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from cnn_fpga.benchmark import postselection_diagnostic as postselection_module
from cnn_fpga.benchmark.experimental_feasibility import (
    implementation_sha256 as feasibility_implementation_sha256,
    validate_payload as validate_feasibility_payload,
)
from cnn_fpga.benchmark.logical_channel_fidelity import (
    implementation_sha256 as fidelity_implementation_sha256,
    validate_artifact_payload as validate_fidelity_payload,
)
from cnn_fpga.benchmark.logical_channel_reconstruction import (
    implementation_sha256 as channel_implementation_sha256,
    validate_artifact_payload as validate_channel_payload,
)
from cnn_fpga.benchmark.logical_operational_boundary import (
    implementation_sha256 as boundary_implementation_sha256,
    validate_artifact_payload as validate_boundary_payload,
)
from cnn_fpga.benchmark.time_cost_fairness import (
    implementation_sha256 as fairness_implementation_sha256,
    validate_payload as validate_fairness_payload,
)
from physics.qec_cost_accounting import (
    postselection_cost,
    scale_measurement_feedback_cost,
    squeezing_db_from_projector_delta,
)


TASK_ID = "T5.3.4"
CONTRACT_ID = "T534-QEC-POSTSELECTION-COST-LEDGER-V1"
DEFAULT_ARTIFACT = Path("docs/t5_3_4_qec_postselection_cost.json")
DEFAULT_SOURCE_DATA = Path("docs/t5_3_4_qec_postselection_cost_source_data.csv")
PARENT_PATHS = {
    "T3.2.4": Path("docs/t3_2_4_postselection_validation.json"),
    "T5.1.5": Path("docs/t5_1_5_time_cost_fairness.json"),
    "T5.1.6": Path("docs/t5_1_6_experimental_feasibility.json"),
    "T5.3.1": Path("docs/t5_3_1_logical_channel_reconstruction.json"),
    "T5.3.2": Path("docs/t5_3_2_logical_channel_fidelity.json"),
    "T5.3.3": Path("docs/t5_3_3_logical_operational_boundary.json"),
}
POSTSELECTION_SOURCE = Path("docs/t3_2_4_postselection_source_data.csv")
TERMINAL_CUTOFFS = (36, 40)
NOISE_PROFILES = ("high", "medium", "low")
REJECTION_PENALTIES = (0.0, 0.25, 0.5, 1.0)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def implementation_sha256() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__).resolve(),
        Path(__file__).resolve().parents[2] / "physics" / "qec_cost_accounting.py",
    ):
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


def _load_parents() -> dict[str, dict[str, Any]]:
    parents = {
        task: json.loads(path.read_text(encoding="utf-8"))
        for task, path in PARENT_PATHS.items()
    }
    post = parents["T3.2.4"]
    if post.get("status") != "PASS" or not all(
        post.get("gate_summary", {}).get("gates", {}).values()
    ):
        raise ValueError("T3.2.4 post-selection parent is not a live PASS artifact")
    if post.get("implementation_sha256") != postselection_module._implementation_sha256():
        raise ValueError("T3.2.4 implementation hash is stale")
    if validate_fairness_payload(parents["T5.1.5"]):
        raise ValueError("T5.1.5 time/cost parent failed validation")
    if validate_feasibility_payload(parents["T5.1.6"]):
        raise ValueError("T5.1.6 feasibility parent failed validation")
    validate_channel_payload(parents["T5.3.1"])
    validate_fidelity_payload(parents["T5.3.2"])
    validate_boundary_payload(parents["T5.3.3"])
    return parents


def _live_implementation_hash(task: str) -> str:
    return {
        "T3.2.4": postselection_module._implementation_sha256(),
        "T5.1.5": fairness_implementation_sha256(),
        "T5.1.6": feasibility_implementation_sha256(),
        "T5.3.1": channel_implementation_sha256(),
        "T5.3.2": fidelity_implementation_sha256(),
        "T5.3.3": boundary_implementation_sha256(),
    }[task]


def _parent_audits(parents: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    audits = {}
    for task, path in PARENT_PATHS.items():
        payload = parents[task]
        stored = payload.get("implementation_sha256")
        live = _live_implementation_hash(task)
        if task == "T3.2.4":
            gates = payload.get("gate_summary", {}).get("gates", {})
        else:
            gates = payload.get("gates", {})
        audits[task] = {
            "path": path.as_posix(),
            "sha256": _sha256(path),
            "task_id": payload.get("task_id"),
            "status": payload.get("status"),
            "all_gates_passed": bool(gates) and all(gates.values()),
            "stored_implementation_sha256": stored,
            "live_implementation_sha256": live,
            "implementation_hash_matches": stored == live,
        }
    audits["T3.2.4"]["source_data_path"] = POSTSELECTION_SOURCE.as_posix()
    audits["T3.2.4"]["source_data_sha256"] = _sha256(POSTSELECTION_SOURCE)
    audits["T3.2.4"]["source_data_rows"] = parents["T3.2.4"]["aggregate"][
        "source_data_rows"
    ]
    return audits


def _standard_cost_reference(fairness: Mapping[str, Any]) -> dict[str, Any]:
    rows = [
        row
        for row in fairness["matched_controller_rows"]
        if row["strategy"] == "standard"
    ]
    if len(rows) != 2:
        raise ValueError("T5.1.5 must contain the two standard fixed-control references")
    fields = (
        "cycle_duration_us",
        "measurement_events",
        "reset_events",
        "active_gate_applications",
        "stored_scalars",
        "persistent_state_scalars",
        "analytic_macs_per_half_cycle",
        "classical_latency_us",
    )
    if any(any(row[field] != rows[0][field] for field in fields) for row in rows[1:]):
        raise ValueError("standard fixed-control cost reference drifts across cutoffs")
    reference = rows[0]
    if reference["full_cycles"] != 10 or reference["simulated_physical_time_us"] != 100.0:
        raise ValueError("standard cost reference must remain the 100 us ten-cycle lane")
    return {
        "source_lane_ids": [row["lane_id"] for row in rows],
        "source_cutoffs": [row["cutoff"] for row in rows],
        "reference_horizon_us": 100.0,
        "reference_full_cycles": 10,
        "measurements_per_full_cycle": reference["measurement_events"] // 10,
        "resets_per_full_cycle": reference["reset_events"] // 10,
        "active_gates_per_full_cycle": reference["active_gate_applications"] // 10,
        "stored_control_scalars": reference["stored_scalars"],
        "persistent_state_scalars": reference["persistent_state_scalars"],
        "analytic_macs_per_half_cycle": reference["analytic_macs_per_half_cycle"],
        "matched_controller_classical_latency_us": reference["classical_latency_us"],
        "resource_scope": "fixed nominal control constants; analytic counts only",
    }


def _online_rows(parents: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    fidelity = parents["T5.3.2"]
    boundary = parents["T5.3.3"]
    channel = parents["T5.3.1"]
    reference = _standard_cost_reference(parents["T5.1.5"])
    scaled = scale_measurement_feedback_cost(
        horizon_us=300.0,
        cycle_duration_us=10.0,
        measurements_per_full_cycle=reference["measurements_per_full_cycle"],
        resets_per_full_cycle=reference["resets_per_full_cycle"],
        active_gates_per_full_cycle=reference["active_gates_per_full_cycle"],
    ).to_dict()
    rows = []
    for cutoff in TERMINAL_CUTOFFS:
        for noise in NOISE_PROFILES:
            active = fidelity["lanes"][f"cutoff{cutoff}:{noise}:qec_on"]
            passive = fidelity["lanes"][f"cutoff{cutoff}:{noise}:qec_off"]
            comparison = boundary["comparisons"][f"cutoff{cutoff}:{noise}"]
            active_events = channel["lanes"][f"cutoff{cutoff}:{noise}:qec_on"][
                "event_accounting"
            ]
            passive_events = channel["lanes"][f"cutoff{cutoff}:{noise}:qec_off"][
                "event_accounting"
            ]
            delta = float(active["parent_lane_config"]["projector_delta"])
            rows.append(
                {
                    "row_id": f"cutoff{cutoff}:{noise}:online_qec",
                    "lane_role": "online_nonpostselected_qec",
                    "cutoff": cutoff,
                    "noise_profile": noise,
                    **scaled,
                    "postselection_applied": False,
                    "postselection_acceptance_fraction": 1.0,
                    "postselection_rejection_fraction": 0.0,
                    "code_space_survival_is_acceptance": False,
                    "projector_delta": delta,
                    "equivalent_squeezing_db": squeezing_db_from_projector_delta(delta),
                    "squeezing_convention": "-10 log10(2 Delta^2)",
                    "stored_control_scalars": reference["stored_control_scalars"],
                    "persistent_state_scalars": reference["persistent_state_scalars"],
                    "analytic_macs_per_half_cycle": reference[
                        "analytic_macs_per_half_cycle"
                    ],
                    "matched_controller_classical_latency_us": reference[
                        "matched_controller_classical_latency_us"
                    ],
                    "target_board_core_latency_us": None,
                    "target_board_transport_latency_us": None,
                    "target_board_end_to_end_latency_us": None,
                    "active_pulse_or_gate_count": active_events[
                        "active_gate_applications"
                    ],
                    "active_pulse_duration_us": None,
                    "active_pulse_energy": None,
                    "measurement_events": active_events["measurement_events"],
                    "reset_events": active_events["reset_events"],
                    "passive_measurement_events": passive_events["measurement_events"],
                    "passive_reset_events": passive_events["reset_events"],
                    "passive_active_gate_applications": passive_events[
                        "active_gate_applications"
                    ],
                    "scaled_reference_matches_channel_events": (
                        active_events["full_cycles"] == scaled["full_cycles"]
                        and active_events["total_physical_time_us"] == scaled["horizon_us"]
                        and active_events["measurement_events"]
                        == scaled["measurement_events"]
                        and active_events["reset_events"] == scaled["reset_events"]
                        and active_events["active_gate_applications"]
                        == scaled["active_gate_applications"]
                    ),
                    "event_count_provenance": (
                        "T5.3.1 native event_accounting cross-checked against T5.1.5 "
                        "standard measurement-feedback protocol"
                    ),
                    "final_active_average_fidelity": active["final_metrics"][
                        "average_fidelity"
                    ],
                    "final_passive_average_fidelity": passive["final_metrics"][
                        "average_fidelity"
                    ],
                    "final_average_fidelity_advantage": comparison["boundary"][
                        "terminal_advantage"
                    ],
                    "final_active_code_space_survival": active["final_metrics"][
                        "mean_code_survival"
                    ],
                    "final_passive_code_space_survival": passive["final_metrics"][
                        "mean_code_survival"
                    ],
                    "sustained_boundary_time_us": comparison["boundary"][
                        "sustained_dominance_time_us"
                    ],
                    "cumulative_payback_time_us": comparison["boundary"][
                        "cumulative_breakeven_time_us"
                    ],
                    "achieved_logical_error_rate": None,
                    "logical_error_rate_scope": "not_defined_for_this_cptni_channel_lane",
                    "full_cost_complete": False,
                    "full_cost_operational_boundary_qualified": False,
                    "target_hardware_measured": False,
                }
            )
    return rows


def _postselection_rows(post: Mapping[str, Any]) -> list[dict[str, Any]]:
    summary_by_target: dict[float, list[Mapping[str, Any]]] = {}
    for summary in post["scenario_survival_summaries"]:
        summary_by_target.setdefault(float(summary["target_survival"]), []).append(summary)
    rows = []
    for aggregate in post["aggregate_by_target_survival"]:
        target = float(aggregate["target_survival"])
        metric = postselection_cost(
            acceptance_fraction=aggregate["realized_survival_fraction"],
            raw_error_rate=aggregate["raw_error_rate"],
            conditional_error_rate=aggregate["conditional_error_rate"],
            rejection_penalties=REJECTION_PENALTIES,
        ).to_dict()
        truth_upper = float(
            np.mean(
                [
                    row["truth_upper_conditional_error_rate"]
                    for row in summary_by_target[target]
                ]
            )
        )
        rows.append(
            {
                "row_id": f"postselection_target_{target:.3f}",
                "lane_role": "offline_postselection_diagnostic",
                "target_survival": target,
                "training_threshold": aggregate["training_threshold"],
                **metric,
                "raw_minus_conditional_seed_cluster_ci": aggregate[
                    "raw_minus_conditional_seed_cluster_ci"
                ],
                "truth_upper_conditional_error_rate": truth_upper,
                "truth_upper_deployable": False,
                "online_correction_eligible": False,
                "primary_metric_eligible": False,
                "achieved_average_fidelity": None,
                "achieved_logical_error_rate": None,
                "conditional_error_scope": (
                    "synthetic wrapped-Gaussian decoder decision error, not CPTNI F_avg or "
                    "physical-memory LER"
                ),
                "measurement_events": None,
                "reset_events": None,
                "active_pulse_or_gate_count": None,
                "classical_latency_us": None,
                "cross_lane_cost_aggregate_eligible": False,
            }
        )
    return rows


def _safety_row(feasibility: Mapping[str, Any]) -> dict[str, Any]:
    source = feasibility["safety_summary"]
    return {
        "lane_role": "separate_deterministic_software_safety_campaign",
        **source,
        "joined_to_online_channel_cost": False,
        "joined_to_postselection_cost": False,
        "device_population_rate": False,
    }


def _missing_fields(fairness: Mapping[str, Any]) -> list[dict[str, Any]]:
    latency = fairness["latency_contract"]
    rows = [
        {
            "field": "matched_controller_classical_latency_us",
            "value": None,
            "evidence": "not_measured_for_fixed_nominal_controller_and_target",
        },
        {
            "field": "active_pulse_duration_and_energy",
            "value": None,
            "evidence": "only active gate counts are available",
        },
        {
            "field": "device_calibrated_reset_fidelity_and_energy",
            "value": None,
            "evidence": "software event counts and effective reset assumptions only",
        },
        {
            "field": "matched_physical_memory_logical_error_rate",
            "value": None,
            "evidence": "T5.3 channel reports CPTNI F_avg/F_e, not physical-memory LER",
        },
        {
            "field": "best_passive_physical_qubit_reference",
            "value": None,
            "evidence": "qec_off is matched encoded idle only",
        },
    ]
    for source in latency["target_board_latency"] + latency["physical_frontend"]:
        rows.append(
            {
                "field": source["name"],
                "value": source["value_us"],
                "evidence": source["evidence_class"],
            }
        )
    return rows


def _derive(parents: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    fairness = parents["T5.1.5"]
    post_rows = _postselection_rows(parents["T3.2.4"])
    online = _online_rows(parents)
    return {
        "standard_cost_reference": _standard_cost_reference(fairness),
        "online_qec_cost_rows": online,
        "postselection_cost_rows": post_rows,
        "software_safety_cost_row": _safety_row(parents["T5.1.6"]),
        "missing_cost_fields": _missing_fields(fairness),
    }


def _semantic_gates(
    payload: Mapping[str, Any], parents: Mapping[str, Mapping[str, Any]]
) -> dict[str, bool]:
    derived = _derive(parents)
    audits = payload.get("parent_audits", {})
    online = payload.get("online_qec_cost_rows", [])
    post = payload.get("postselection_cost_rows", [])
    missing = payload.get("missing_cost_fields", [])
    verdict = payload.get("verdict", {})
    claim = payload.get("claim_boundary", {})
    expected_tasks = set(PARENT_PATHS)
    parent_gate = lambda task: bool(
        audits.get(task, {}).get("path") == PARENT_PATHS[task].as_posix()
        and audits.get(task, {}).get("sha256") == _sha256(PARENT_PATHS[task])
        and audits.get(task, {}).get("task_id") == task
        and audits.get(task, {}).get("status") == "PASS"
        and audits.get(task, {}).get("all_gates_passed")
        and audits.get(task, {}).get("implementation_hash_matches")
    )
    return {
        "all_six_parent_artifacts_and_implementations_are_live": (
            set(audits) == expected_tasks and all(parent_gate(task) for task in expected_tasks)
        ),
        "postselection_source_data_is_bound_and_complete": bool(
            audits.get("T3.2.4", {}).get("source_data_path")
            == POSTSELECTION_SOURCE.as_posix()
            and audits.get("T3.2.4", {}).get("source_data_sha256")
            == _sha256(POSTSELECTION_SOURCE)
            and audits.get("T3.2.4", {}).get("source_data_rows") == 256
        ),
        "six_terminal_online_qec_cost_rows_are_present": len(online) == 6,
        "online_rows_use_300us_30cycle_60halfcycle_horizon": all(
            row["horizon_us"] == 300.0
            and row["full_cycles"] == 30
            and row["half_cycles"] == 60
            for row in online
        ) if online else False,
        "online_event_and_active_gate_counts_scale_exactly": all(
            row["measurement_events"] == 60
            and row["reset_events"] == 60
            and row["active_gate_applications"] == 540
            and row["active_pulse_or_gate_count"] == 540
            and row["passive_measurement_events"] == 0
            and row["passive_reset_events"] == 0
            and row["passive_active_gate_applications"] == 0
            and row["scaled_reference_matches_channel_events"] is True
            for row in online
        ) if online else False,
        "projector_delta_and_squeezing_conversion_are_traceable": all(
            row["projector_delta"] == 0.34
            and abs(row["equivalent_squeezing_db"] - squeezing_db_from_projector_delta(0.34))
            <= 2.0e-12
            and row["squeezing_convention"] == "-10 log10(2 Delta^2)"
            for row in online
        ) if online else False,
        "online_qec_is_nonpostselected_and_code_survival_is_not_acceptance": all(
            row["postselection_applied"] is False
            and row["postselection_acceptance_fraction"] == 1.0
            and row["postselection_rejection_fraction"] == 0.0
            and row["code_space_survival_is_acceptance"] is False
            for row in online
        ) if online else False,
        "online_fidelity_survival_and_boundaries_match_native_parents": _deep_close(
            online, derived["online_qec_cost_rows"]
        ),
        "online_ler_is_null_not_inferred_from_fidelity": all(
            row["achieved_logical_error_rate"] is None
            and row["logical_error_rate_scope"]
            == "not_defined_for_this_cptni_channel_lane"
            for row in online
        ) if online else False,
        "fixed_nominal_classical_resource_is_analytic_only": all(
            row["stored_control_scalars"] == 15
            and row["persistent_state_scalars"] == 0
            and row["analytic_macs_per_half_cycle"] == 0
            for row in online
        ) if online else False,
        "matched_and_target_board_latency_remain_null": all(
            row["matched_controller_classical_latency_us"] is None
            and row["target_board_core_latency_us"] is None
            and row["target_board_transport_latency_us"] is None
            and row["target_board_end_to_end_latency_us"] is None
            for row in online
        ) if online else False,
        "active_gate_count_is_known_but_pulse_duration_energy_remain_null": all(
            row["active_pulse_or_gate_count"] == 540
            and row["active_pulse_duration_us"] is None
            and row["active_pulse_energy"] is None
            for row in online
        ) if online else False,
        "online_rows_do_not_claim_complete_cost_or_hardware": all(
            row["full_cost_complete"] is False
            and row["full_cost_operational_boundary_qualified"] is False
            and row["target_hardware_measured"] is False
            for row in online
        ) if online else False,
        "eight_training_only_postselection_targets_are_present": (
            len(post) == 8
            and [row["target_survival"] for row in post]
            == [0.995, 0.99, 0.98, 0.95, 0.9, 0.8, 0.7, 0.5]
        ),
        "postselection_acceptance_rejection_and_failures_are_complete": all(
            abs(row["acceptance_fraction"] + row["rejection_fraction"] - 1.0) <= 1.0e-12
            and abs(
                row["accepted_failures_per_input"]
                - row["acceptance_fraction"] * row["conditional_error_rate"]
            )
            <= 1.0e-12
            for row in post
        ) if post else False,
        "postselection_penalty_grid_and_total_cost_identity_are_complete": all(
            row["rejection_penalties"] == list(REJECTION_PENALTIES)
            and all(
                abs(
                    cost
                    - (
                        row["accepted_failures_per_input"]
                        + penalty * row["rejection_fraction"]
                    )
                )
                <= 1.0e-12
                for penalty, cost in zip(
                    row["rejection_penalties"], row["total_costs"], strict=True
                )
            )
            for row in post
        ) if post else False,
        "unit_rejection_penalty_reverses_all_conditional_improvements": all(
            row["conditional_error_rate"] < row["raw_error_rate"]
            and row["total_cost_by_rejection_penalty"]["1.00"] > row["raw_error_rate"]
            for row in post
        ) if post else False,
        "postselection_ci_and_truth_upper_are_preserved": all(
            row["raw_minus_conditional_seed_cluster_ci"]["ci_low"] > 0.0
            and row["truth_upper_conditional_error_rate"] <= row["conditional_error_rate"]
            and row["truth_upper_deployable"] is False
            for row in post
        ) if post else False,
        "conditional_postselection_metric_is_never_promoted_online": all(
            row["online_correction_eligible"] is False
            and row["primary_metric_eligible"] is False
            and row["conditional_metric_online_eligible"] is False
            and row["cross_lane_cost_aggregate_eligible"] is False
            for row in post
        ) if post else False,
        "postselection_does_not_invent_favg_ler_events_or_latency": all(
            row["achieved_average_fidelity"] is None
            and row["achieved_logical_error_rate"] is None
            and row["measurement_events"] is None
            and row["reset_events"] is None
            and row["active_pulse_or_gate_count"] is None
            and row["classical_latency_us"] is None
            for row in post
        ) if post else False,
        "software_safety_cost_remains_a_separate_deterministic_campaign": bool(
            payload.get("software_safety_cost_row", {}).get("campaign_cycles") == 767872
            and payload.get("software_safety_cost_row", {}).get("fallback_cycles") == 11552
            and payload.get("software_safety_cost_row", {}).get("reset_request_cycles") == 4
            and payload.get("software_safety_cost_row", {}).get(
                "statistical_population_upper_bound"
            ) is None
            and payload.get("software_safety_cost_row", {}).get(
                "joined_to_online_channel_cost"
            ) is False
            and payload.get("software_safety_cost_row", {}).get(
                "joined_to_postselection_cost"
            ) is False
        ),
        "all_required_unmeasured_cost_fields_remain_explicit_nulls": (
            len(missing) == 12
            and all(row.get("value") is None and row.get("evidence") for row in missing)
        ),
        "all_derived_cost_ledgers_recompute_from_native_parents": _deep_close(
            {
                "standard_cost_reference": payload.get("standard_cost_reference"),
                "online_qec_cost_rows": online,
                "postselection_cost_rows": post,
                "software_safety_cost_row": payload.get("software_safety_cost_row"),
                "missing_cost_fields": missing,
            },
            derived,
        ),
        "cost_dimensions_are_reported_without_cross_lane_scalarization": bool(
            payload.get("cost_contract", {}).get("global_cost_score") is None
            and payload.get("cost_contract", {}).get("cross_lane_total") is None
            and payload.get("cost_contract", {}).get("postselection_joined_to_qec") is False
        ),
        "full_cost_boundary_and_coherence_gain_remain_not_established": bool(
            verdict.get("full_cost_operational_boundary") == "NOT_ESTABLISHED"
            and verdict.get("paper_defined_coherence_gain") == "NOT_ESTABLISHED"
            and verdict.get("postselected_break_even") == "NOT_ESTABLISHED"
        ),
        "wall_clock_boundary_is_retained_without_cost_promotion": (
            verdict.get("wall_clock_operational_boundary_parent")
            == "ESTABLISHED_WITHIN_300US_FINITE_CUTOFF_MODEL"
        ),
        "claim_boundary_remains_diagnostic_simulation_only": bool(
            claim.get("conditional_postselection_is_online_gain") is False
            and claim.get("full_cost_complete") is False
            and claim.get("experimental_break_even") is False
            and claim.get("target_hardware_measured") is False
        ),
    }


def validate_artifact_payload(payload: Mapping[str, Any]) -> dict[str, bool]:
    if payload.get("task_id") != TASK_ID or payload.get("contract_id") != CONTRACT_ID:
        raise ValueError("artifact task/contract identity mismatch")
    parents = _load_parents()
    gates = _semantic_gates(payload, parents)
    if payload.get("gates") != gates:
        raise ValueError("stored gates do not match recomputed semantic gates")
    if payload.get("required_gates") != list(gates):
        raise ValueError("required gate order/schema mismatch")
    expected = "PASS" if all(gates.values()) else "FAIL"
    if payload.get("status") != expected:
        raise ValueError("artifact status does not match gates")
    return gates


def run_report() -> dict[str, Any]:
    parents = _load_parents()
    derived = _derive(parents)
    post = derived["postselection_cost_rows"]
    payload: dict[str, Any] = {
        "task_id": TASK_ID,
        "contract_id": CONTRACT_ID,
        "status": "PENDING",
        "implementation_sha256": implementation_sha256(),
        "parent_audits": _parent_audits(parents),
        "cost_contract": {
            "online_qec_scope": (
                "300 us finite-cutoff nominal-SBS CPTNI channel with scaled protocol event counts"
            ),
            "postselection_scope": (
                "separate synthetic wrapped-Gaussian offline confidence diagnostic"
            ),
            "safety_scope": "separate deterministic software fault campaign",
            "global_cost_score": None,
            "cross_lane_total": None,
            "postselection_joined_to_qec": False,
            "null_rule": "unmeasured or model-incompatible cost fields remain null, never zero",
        },
        **derived,
        "verdict": {
            "online_qec_cost_ledger": "COMPLETE_WITH_EXPLICIT_UNMEASURED_FIELDS",
            "postselection_cost_ledger": "COMPLETE_DIAGNOSTIC_ONLY",
            "postselection_targets_with_lower_conditional_error": sum(
                row["conditional_error_rate"] < row["raw_error_rate"] for row in post
            ),
            "postselection_targets_worse_at_unit_rejection_penalty": sum(
                row["total_cost_by_rejection_penalty"]["1.00"] > row["raw_error_rate"]
                for row in post
            ),
            "wall_clock_operational_boundary_parent": parents["T5.3.3"]["verdict"][
                "wall_clock_operational_boundary"
            ],
            "full_cost_operational_boundary": "NOT_ESTABLISHED",
            "paper_defined_coherence_gain": "NOT_ESTABLISHED",
            "postselected_break_even": "NOT_ESTABLISHED",
        },
        "claim_boundary": {
            "allowed": (
                "separate simulation-derived online event/resource counts, CPTNI fidelity/survival, "
                "offline postselection acceptance/rejection/penalty costs, and explicit null fields"
            ),
            "forbidden": [
                "conditional postselection as online correction gain",
                "postselected or full-cost break-even",
                "CPTNI code survival as trajectory acceptance",
                "configuration latency as measurement",
                "analytic MAC/scalar counts as RTL or board resource",
                "cross-lane total score",
                "physical-memory LER, device reset rate or experimental result",
            ],
            "conditional_postselection_is_online_gain": False,
            "full_cost_complete": False,
            "experimental_break_even": False,
            "target_hardware_measured": False,
        },
    }
    gates = _semantic_gates(payload, parents)
    payload["gates"] = gates
    payload["required_gates"] = list(gates)
    payload["status"] = "PASS" if all(gates.values()) else "FAIL"
    validate_artifact_payload(payload)
    return payload


def _source_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "category": "contract",
            "task_id": TASK_ID,
            "contract_id": CONTRACT_ID,
            "status": payload["status"],
            "implementation_sha256": payload["implementation_sha256"],
        }
    ]
    rows.extend(
        {"category": "parent", "parent_task": task, **audit}
        for task, audit in payload["parent_audits"].items()
    )
    rows.append({"category": "standard_cost_reference", **payload["standard_cost_reference"]})
    rows.extend({"category": "online_qec_cost", **row} for row in payload["online_qec_cost_rows"])
    for row in payload["postselection_cost_rows"]:
        rows.append(
            {
                "category": "postselection_summary",
                **{
                    key: value
                    for key, value in row.items()
                    if key
                    not in {
                        "rejection_penalties",
                        "total_costs",
                        "total_cost_by_rejection_penalty",
                        "raw_minus_conditional_seed_cluster_ci",
                    }
                },
                "ci_low": row["raw_minus_conditional_seed_cluster_ci"]["ci_low"],
                "ci_high": row["raw_minus_conditional_seed_cluster_ci"]["ci_high"],
            }
        )
        rows.extend(
            {
                "category": "postselection_penalty",
                "row_id": row["row_id"],
                "target_survival": row["target_survival"],
                "rejection_penalty": penalty,
                "total_cost": cost,
                "raw_error_rate": row["raw_error_rate"],
                "cost_minus_raw": cost - row["raw_error_rate"],
            }
            for penalty, cost in zip(
                row["rejection_penalties"], row["total_costs"], strict=True
            )
        )
    rows.append({"category": "software_safety_cost", **payload["software_safety_cost_row"]})
    rows.extend({"category": "missing_cost_field", **row} for row in payload["missing_cost_fields"])
    rows.extend(
        {"category": "gate", "gate": gate, "passed": passed}
        for gate, passed in payload["gates"].items()
    )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_artifacts(
    *,
    artifact_path: Path = DEFAULT_ARTIFACT,
    source_data_path: Path = DEFAULT_SOURCE_DATA,
) -> dict[str, Any]:
    payload = run_report()
    rows = _source_rows(payload)
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
                "verdict": payload["verdict"],
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
    "implementation_sha256",
    "run_report",
    "validate_artifact_payload",
    "write_artifacts",
]
