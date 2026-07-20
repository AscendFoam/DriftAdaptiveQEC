"""T5.1.5 physical-time and control-cost fairness consolidation.

The report is read-only: it consolidates protocol-native wall-clock evidence,
matched-controller cost evidence, host-software estimator profiles and explicit
target-board null fields.  It never rescales one protocol's lifetime by another
protocol's cycle count and never fills an unmeasured latency with zero.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from cnn_fpga.benchmark.algorithm_success_falsification import (
    FALLBACK_BRANCH_ID,
)


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "T5.1.5"
SCHEMA_VERSION = "t5.1.5-time-cost-fairness-v1"
PROTOCOL_ID = "LANE-LOCAL-PHYSICAL-TIME-CONTROL-COST-V1"
DEFAULT_ARTIFACT = Path("docs/t5_1_5_time_cost_fairness.json")
DEFAULT_SOURCE_DATA = Path("docs/t5_1_5_time_cost_fairness_source_data.csv")

PARENT_ARTIFACTS: dict[str, Path] = {
    "T3.2.8": Path("docs/t3_2_8_autonomous_sbs_wallclock_validation.json"),
    "T5.1.1": Path("docs/t5_1_1_comparison_set_registry.json"),
    "T4.4.4": Path("docs/t4_4_4_teacher_student_gain_retention.json"),
    "T4.1.1": Path("docs/t4_1_1_slow_loop_model_selection_validation.json"),
    "T2.4.1": Path("docs/t2_4_1_dual_latency_budget_validation.json"),
    "T5.1.4": Path("docs/t5_1_4_algorithm_branch_verdict.json"),
}

DUAL_LATENCY_ARTIFACT = Path("docs/dual_latency_budget.json")

CONTROLLER_STRATEGIES = (
    "standard",
    "exact_budget_mf",
    "fresh_gru_teacher",
    "handcrafted_recurrence",
    "distilled_student",
)


def _repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(_repo_path(path).read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def implementation_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def load_parent_artifacts(
    paths: Mapping[str, str | Path] = PARENT_ARTIFACTS,
) -> dict[str, dict[str, Any]]:
    parents: dict[str, dict[str, Any]] = {}
    for task_id, path in paths.items():
        payload = json.loads(_repo_path(path).read_text(encoding="utf-8"))
        if payload.get("task_id") != task_id:
            raise ValueError(
                f"{path} has task_id {payload.get('task_id')!r}, expected {task_id}"
            )
        parents[task_id] = payload
    return parents


def _gate_entries(payload: Mapping[str, Any]) -> list[tuple[str, bool]]:
    gates = payload.get("gates")
    if isinstance(gates, Mapping):
        return [(str(name), value is True) for name, value in gates.items()]
    if isinstance(gates, list):
        return [
            (str(row.get("id", index)), row.get("passed") is True)
            for index, row in enumerate(gates)
            if isinstance(row, Mapping)
        ]
    summary = payload.get("gate_summary")
    if isinstance(summary, Mapping) and isinstance(summary.get("gates"), Mapping):
        return [
            (str(name), value is True) for name, value in summary["gates"].items()
        ]
    return []


def _machine_pass(payload: Mapping[str, Any]) -> bool:
    gates = _gate_entries(payload)
    return bool(payload.get("status") == "PASS" and gates and all(v for _, v in gates))


def _declared_file_bindings(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    bindings: list[dict[str, str]] = []

    def add(role: str, record: Any, *, sha_key: str = "sha256") -> None:
        if isinstance(record, Mapping) and record.get("path") and record.get(sha_key):
            bindings.append(
                {
                    "role": role,
                    "path": str(record["path"]),
                    "declared_sha256": str(record[sha_key]),
                }
            )

    add("source_data", payload.get("source_data"))
    for index, record in enumerate(payload.get("artifact_bindings", ())):
        add(f"artifact:{index}", record)
    for index, record in enumerate(payload.get("implementation_bindings", ())):
        add(f"implementation:{index}", record)
    if payload.get("artifact_path") and payload.get("artifact_sha256"):
        add(
            "audited_artifact",
            {"path": payload["artifact_path"], "sha256": payload["artifact_sha256"]},
        )
    return bindings


def current_parent_implementation_hashes() -> dict[str, str]:
    from cnn_fpga.benchmark.algorithm_success_falsification import (
        implementation_sha256 as t514_hash,
    )
    from cnn_fpga.benchmark.autonomous_sbs_wallclock_baseline import (
        implementation_sha256 as t328_hash,
    )
    from cnn_fpga.benchmark.slow_loop_model_selection import (
        _implementation_sha256 as t411_hash,
    )
    from cnn_fpga.benchmark.teacher_student_gain_retention import (
        implementation_sha256 as t444_hash,
    )

    return {
        "T3.2.8": t328_hash(),
        "T4.4.4": t444_hash(),
        "T4.1.1": t411_hash(),
        "T5.1.4": t514_hash(),
    }


def inspect_parent_integrity(
    parents: Mapping[str, Mapping[str, Any]],
    implementation_hashes: Mapping[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    hashes = dict(
        current_parent_implementation_hashes()
        if implementation_hashes is None
        else implementation_hashes
    )
    result: dict[str, dict[str, Any]] = {}
    for task_id, payload in parents.items():
        checks: list[dict[str, Any]] = []
        for binding in _declared_file_bindings(payload):
            path = _repo_path(binding["path"])
            actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            checks.append(
                {
                    **binding,
                    "actual_sha256": actual,
                    "passed": actual == binding["declared_sha256"],
                }
            )
        implementation_current = True
        if task_id in hashes:
            implementation_current = payload.get("implementation_sha256") == hashes[task_id]
        record = {
            "machine_pass": _machine_pass(payload),
            "machine_gate_count": len(_gate_entries(payload)),
            "declared_file_bindings": checks,
            "all_declared_files_current": all(row["passed"] for row in checks),
            "declared_implementation_sha256": payload.get("implementation_sha256"),
            "current_implementation_sha256": hashes.get(task_id),
            "implementation_current": implementation_current,
        }
        record["passed"] = bool(
            record["machine_pass"]
            and record["all_declared_files_current"]
            and implementation_current
        )
        result[task_id] = record
    return result


def _protocol_rows(t328: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    reversals: list[dict[str, Any]] = []
    for lane_id, lane in t328["lanes"].items():
        for strategy in ("measurement_feedback", "autonomous"):
            record = lane[strategy]
            timing = record["config"]["timing"]
            cycle_us = 2.0 * float(
                sum(
                    timing[name]
                    for name in (
                        "entering_cycle_ns",
                        "layer_1_ns",
                        "layer_2_ns",
                        "layer_3_ns",
                        "layer_4_ns",
                        "measurement_and_or_reset_ns",
                        "virtual_rotation_and_idle_ns",
                    )
                )
            ) / 1000.0
            event = record["event_accounting"]
            fidelity = record["metrics"]["fidelity"]
            logical = record["metrics"]["logical_z_signal"]
            rows.append(
                {
                    "lane_id": lane_id,
                    "cutoff": lane["cutoff"],
                    "noise_profile": lane["noise_profile"],
                    "strategy": strategy,
                    "cycle_duration_us": cycle_us,
                    "common_horizon_us": event["total_physical_time_us"],
                    "full_cycles": event["full_cycles"],
                    "fidelity_area_lifetime_protocol_cycles": fidelity[
                        "area_equivalent_lifetime_protocol_cycles"
                    ],
                    "fidelity_area_lifetime_us": fidelity["area_equivalent_lifetime_us"],
                    "logical_z_area_lifetime_protocol_cycles": logical[
                        "area_equivalent_lifetime_protocol_cycles"
                    ],
                    "logical_z_area_lifetime_us": logical[
                        "area_equivalent_lifetime_us"
                    ],
                    "measurement_events": event["measurement_events"],
                    "reset_events": event["reset_events"],
                    "active_gate_applications": event["active_gate_applications"],
                    "measurements_per_100us": event["measurements_per_100us"],
                    "resets_per_100us": event["resets_per_100us"],
                    "active_gates_per_100us": event["active_gates_per_100us"],
                    "outcome_dependent_parameter_updates": event[
                        "outcome_dependent_parameter_updates"
                    ],
                    "online_classical_latency_us": None,
                    "classical_latency_evidence": "not_measured_nominal_nonselective_protocol",
                    "target_hardware_measured": False,
                }
            )
        feedback = lane["measurement_feedback"]
        autonomous = lane["autonomous"]
        comparison = lane["comparison"]
        reversals.append(
            {
                "lane_id": lane_id,
                "autonomous_to_feedback_logical_lifetime_protocol_cycle_ratio": comparison[
                    "autonomous_to_measurement_logical_lifetime_protocol_cycle_ratio"
                ],
                "autonomous_to_feedback_logical_lifetime_us_ratio": comparison[
                    "autonomous_to_measurement_logical_lifetime_us_ratio"
                ],
                "protocol_cycle_favors_autonomous": autonomous["metrics"][
                    "logical_z_signal"
                ]["area_equivalent_lifetime_protocol_cycles"]
                > feedback["metrics"]["logical_z_signal"][
                    "area_equivalent_lifetime_protocol_cycles"
                ],
                "wallclock_favors_autonomous": autonomous["metrics"]["logical_z_signal"][
                    "area_equivalent_lifetime_us"
                ]
                > feedback["metrics"]["logical_z_signal"][
                    "area_equivalent_lifetime_us"
                ],
                "measurement_events_avoided": comparison[
                    "measurement_events_avoided_at_common_horizon"
                ],
                "additional_autonomous_resets": comparison[
                    "additional_autonomous_resets_at_common_horizon"
                ],
                "additional_autonomous_active_gates": comparison[
                    "additional_autonomous_active_gates_at_common_horizon"
                ],
            }
        )
    return rows, reversals


def _idle_reference(t511: Mapping[str, Any]) -> dict[str, Any]:
    probe = t511["no_correction_probe"]
    event = probe["event_accounting"]
    return {
        "reference_id": "no_correction_idle_30us_probe",
        "scope": "separate sanity reference; not a 700us ranked protocol row",
        "reporting_interval_us": 10.0,
        "horizon_us": event["total_physical_time_us"],
        "full_intervals": event["full_cycles"],
        "measurement_events": event["measurement_events"],
        "reset_events": event["reset_events"],
        "active_gate_applications": event["active_gate_applications"],
        "classical_latency_us": None,
        "classical_latency_evidence": "not_applicable_zero_active_operations",
        "ranked_with_700us_protocol_lane": False,
    }


def _controller_rows(t444: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cost_by_strategy = {row["strategy"]: row for row in t444["costs"]}
    burden_keys = {
        "standard": "standard",
        "exact_budget_mf": "exact_budget_mf_agent_mean",
        "fresh_gru_teacher": "teacher",
        "handcrafted_recurrence": "handcrafted_recurrence",
        "distilled_student": "distilled_student",
    }
    payload_keys = {
        "standard": "standard",
        "exact_budget_mf": "mf_all_agents",
        "fresh_gru_teacher": "teacher",
        "handcrafted_recurrence": "handcrafted_recurrence",
        "distilled_student": "distilled_student",
    }
    rows: list[dict[str, Any]] = []
    for lane_name, cutoff in (("primary", 12), ("confirmation", 16)):
        lane = t444["stochastic_ten_cycle"][lane_name]
        burden_lane = t444["burden_summary"]["stochastic"][lane_name]
        for strategy in CONTROLLER_STRATEGIES:
            payload = lane[payload_keys[strategy]]
            if strategy == "exact_budget_mf":
                metrics = payload["metric_mean_across_agents"]
                auxiliary = payload["auxiliary_mean_across_agents"]
                physical_times = {
                    row["simulated_physical_time_us"] for row in payload["agents"]
                }
                physical_time_us = next(iter(physical_times)) if len(physical_times) == 1 else None
            else:
                metrics = payload["metric_means"]
                auxiliary = payload["auxiliary_means"]
                physical_time_us = payload["simulated_physical_time_us"]
            burden = burden_lane[burden_keys[strategy]]
            cost = cost_by_strategy[strategy]
            fidelity_cycles = metrics["fidelity_effective_lifetime_cycles"]
            logical_cycles = metrics["logical_z_effective_lifetime_cycles"]
            rows.append(
                {
                    "lane_id": f"matched_controller_cutoff{cutoff}_100us",
                    "cutoff": cutoff,
                    "strategy": strategy,
                    "full_cycles": 10,
                    "cycle_duration_us": 10.0,
                    "simulated_physical_time_us": physical_time_us,
                    "fidelity_effective_lifetime_cycles": fidelity_cycles,
                    "fidelity_effective_lifetime_us": 10.0 * fidelity_cycles,
                    "logical_z_effective_lifetime_cycles": logical_cycles,
                    "logical_z_effective_lifetime_us": 10.0 * logical_cycles,
                    "measurement_events": 20,
                    "reset_events": 20,
                    "active_gate_applications": 180,
                    "physical_event_count_provenance": (
                        "T3.2.8 measurement-feedback timing: 2 half-cycles/full-cycle, "
                        "1 measurement, 1 reset and 9 active gates/half-cycle"
                    ),
                    "observed_e_events": burden[
                        "expected_e_events_from_observed_fraction"
                    ],
                    "multilevel_leakage_events": burden["multilevel_leakage_events"],
                    "mean_control_residual_rms": auxiliary[
                        "mean_control_residual_rms"
                    ],
                    "mean_control_slew_rms": auxiliary["mean_control_slew_rms"],
                    "online_policy_evaluations": 0 if strategy == "standard" else 20,
                    "stored_scalars": cost["stored_scalars"],
                    "persistent_state_scalars": cost["persistent_state_scalars"],
                    "analytic_macs_per_half_cycle": cost[
                        "analytic_macs_per_half_cycle"
                    ],
                    "classical_latency_us": None,
                    "classical_latency_evidence": (
                        "not_measured_for_this_controller_model_or_target_board"
                    ),
                    "deployable_in_parent": cost["deployable"],
                    "cost_scope": cost["cost_scope"],
                    "target_hardware_measured": False,
                }
            )
    exclusion = {
        "strategy": "finite_horizon_control_oracle",
        "excluded_from_ten_cycle_lane": True,
        "available_horizon_cycles": 2,
        "reason": (
            "exact finite-horizon reference has a different objective/horizon and cannot "
            "be rescaled or extrapolated into the ten-cycle controller ranking"
        ),
    }
    return rows, exclusion


def _slow_loop_rows(t411: Mapping[str, Any]) -> list[dict[str, Any]]:
    budget = t411["validation_config"]["budget"]
    rows: list[dict[str, Any]] = []
    for family, profile in t411["resource_profiles"].items():
        latency = profile["host_batch_median_us_per_update"]
        rows.append(
            {
                "lane_id": "host_slow_loop_regime_estimator",
                "family": family,
                "decision_target": "four_regime_posterior_per_32_cycle_update",
                "update_period_cycles": budget["update_period_cycles"],
                "macs_per_update_proxy": profile["macs_per_update_proxy"],
                "model_and_state_bytes": profile["model_and_state_bytes"],
                "transient_workspace_bytes": profile["transient_workspace_bytes"],
                "host_profile_rows_per_batch": profile["host_profile_rows_per_batch"],
                "host_profile_repeats": profile["host_profile_repeats"],
                "host_batch_median_us_per_update": latency,
                "host_software_latency_ceiling_us": budget[
                    "host_software_latency_ceiling_us"
                ],
                "latency_to_ceiling_fraction": latency
                / budget["host_software_latency_ceiling_us"],
                "physical_lifetime_cycles": None,
                "physical_lifetime_us": None,
                "measurement_events": None,
                "reset_events": None,
                "active_gate_applications": None,
                "evidence_class": "development_host_batch_profile_not_controller_or_target_board",
                "target_hardware_measured": False,
            }
        )
    return rows


def _latency_contract(dual_budget: Mapping[str, Any]) -> dict[str, Any]:
    project = dual_budget["lanes"]["project_control_plane"]
    software = project["software_latency_model"]
    return {
        "project_target_board": project["target_board"],
        "real_board_gate": project["real_board_gate"],
        "board_measurement_status": project["board_measurement_status"],
        "configured_cadence_assumption": project["cadence"],
        "configured_software_latency_model": {
            "fast_path_mean_us": software["fast_path"]["mean_us"],
            "slow_path_mean_total_us": software["slow_path_mean_total_us"],
            "evidence_class": software["evidence_class"],
            "measured_on_target_board": software["measured_on_target_board"],
            "scope_note": software["scope_note"],
        },
        "target_board_latency": project["target_board_latency"],
        "physical_frontend": project["physical_frontend"],
        "cross_lane_aggregate_latency_us": dual_budget["cross_lane_comparison"][
            "aggregate_latency_us"
        ],
        "cross_lane_comparison_status": dual_budget["cross_lane_comparison"]["status"],
    }


def validate_payload(payload: Mapping[str, Any]) -> tuple[str, ...]:
    errors: list[str] = []
    protocol_rows = payload.get("protocol_wallclock_rows", ())
    if len(protocol_rows) != 12:
        errors.append("protocol wall-clock lane must contain 12 rows")
    for row in protocol_rows:
        if row.get("common_horizon_us") != 700.0:
            errors.append("protocol rows do not share 700 us")
        cycle = row.get("cycle_duration_us")
        cycles = row.get("full_cycles")
        if cycle is None or cycles is None or abs(cycle * cycles - 700.0) > 1e-9:
            errors.append("protocol cycle arithmetic is inconsistent")
        for prefix in ("fidelity", "logical_z"):
            per_cycle = row.get(f"{prefix}_area_lifetime_protocol_cycles")
            per_us = row.get(f"{prefix}_area_lifetime_us")
            if per_cycle is None or per_us is None or abs(per_cycle * cycle - per_us) > 1e-8:
                errors.append("protocol cycle/us lifetime pair is incomplete or rescaled")
        if row.get("online_classical_latency_us") is not None:
            errors.append("nominal protocol latency must remain unmeasured")
    reversals = payload.get("protocol_ordering_reversal", ())
    if len(reversals) != 6 or not all(
        row.get("protocol_cycle_favors_autonomous") is True
        and row.get("wallclock_favors_autonomous") is False
        for row in reversals
    ):
        errors.append("six protocol cycle/wall-clock reversals are not retained")

    idle = payload.get("idle_reference", {})
    if any(idle.get(name) != 0 for name in ("measurement_events", "reset_events", "active_gate_applications")):
        errors.append("idle event counts are not exact zero")
    if idle.get("ranked_with_700us_protocol_lane") is not False:
        errors.append("30 us idle probe was mixed into 700 us ranking")

    controller = payload.get("matched_controller_rows", ())
    if len(controller) != 10:
        errors.append("matched-controller lane must contain 10 rows")
    if any(row.get("strategy") == "finite_horizon_control_oracle" for row in controller):
        errors.append("two-cycle control reference entered ten-cycle lane")
    for row in controller:
        if row.get("full_cycles") != 10 or row.get("cycle_duration_us") != 10.0:
            errors.append("controller cycle contract changed")
        if row.get("simulated_physical_time_us") != 100.0:
            errors.append("controller matched physical horizon is not 100 us")
        if abs(
            row.get("fidelity_effective_lifetime_cycles", 0.0) * 10.0
            - row.get("fidelity_effective_lifetime_us", -1.0)
        ) > 1e-8:
            errors.append("controller fidelity cycle/us pair is inconsistent")
        if abs(
            row.get("logical_z_effective_lifetime_cycles", 0.0) * 10.0
            - row.get("logical_z_effective_lifetime_us", -1.0)
        ) > 1e-8:
            errors.append("controller logical cycle/us pair is inconsistent")
        if (
            row.get("measurement_events") != 20
            or row.get("reset_events") != 20
            or row.get("active_gate_applications") != 180
        ):
            errors.append("controller physical event count was changed or confused with e burden")
        if row.get("classical_latency_us") is not None:
            errors.append("unmeasured controller latency was filled")
        if row.get("multilevel_leakage_events") is not None:
            errors.append("unavailable multilevel leakage burden was filled")
    exclusion = payload.get("excluded_control_reference", {})
    if exclusion.get("excluded_from_ten_cycle_lane") is not True:
        errors.append("two-cycle control reference exclusion is missing")

    slow = payload.get("slow_loop_host_latency_rows", ())
    if len(slow) != 6:
        errors.append("slow-loop host lane must contain six families")
    for row in slow:
        if row.get("physical_lifetime_cycles") is not None or row.get("physical_lifetime_us") is not None:
            errors.append("estimator host profile was given a physical lifetime")
        if any(row.get(name) is not None for name in ("measurement_events", "reset_events", "active_gate_applications")):
            errors.append("estimator host profile was given physical event counts")
        latency = row.get("host_batch_median_us_per_update")
        ceiling = row.get("host_software_latency_ceiling_us")
        if not isinstance(latency, (float, int)) or not 0.0 < latency < ceiling:
            errors.append("host latency is nonfinite or outside its software ceiling")
        if row.get("target_hardware_measured") is not False:
            errors.append("host software profile was promoted to target hardware")

    latency = payload.get("latency_contract", {})
    target_rows = list(latency.get("target_board_latency", ())) + list(
        latency.get("physical_frontend", ())
    )
    if len(target_rows) != 7 or any(
        row.get("value_us") is not None or row.get("measured_on_target_board") is not False
        for row in target_rows
    ):
        errors.append("target-board or physical-frontend null fields were filled")
    if latency.get("configured_software_latency_model", {}).get("measured_on_target_board") is not False:
        errors.append("configured software latency was promoted to measurement")
    if latency.get("cross_lane_aggregate_latency_us") is not None:
        errors.append("cross-lane aggregate latency was invented")
    if payload.get("active_algorithm_branch") != FALLBACK_BRANCH_ID:
        errors.append("T5.1.4 fallback branch is not preserved")
    return tuple(errors)


def build_report(
    parents: Mapping[str, Mapping[str, Any]],
    integrity: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    missing = set(PARENT_ARTIFACTS) - set(parents)
    if missing:
        raise ValueError(f"missing parent artifacts: {sorted(missing)}")
    if set(PARENT_ARTIFACTS) - set(integrity):
        raise ValueError("missing parent integrity")
    dual_budget = json.loads(_repo_path(DUAL_LATENCY_ARTIFACT).read_text(encoding="utf-8"))
    protocol_rows, reversals = _protocol_rows(parents["T3.2.8"])
    controller_rows, exclusion = _controller_rows(parents["T4.4.4"])
    slow_rows = _slow_loop_rows(parents["T4.1.1"])
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "protocol_id": PROTOCOL_ID,
        "pass_semantics": (
            "cycle/wall-clock, physical-event, algorithmic-cost and latency evidence are "
            "reported lane-locally with nulls preserved; no global performance ranking or "
            "target-board measurement is implied"
        ),
        "active_algorithm_branch": parents["T5.1.4"]["active_branch"]["branch_id"],
        "parent_integrity": dict(integrity),
        "protocol_wallclock_rows": protocol_rows,
        "protocol_ordering_reversal": reversals,
        "idle_reference": _idle_reference(parents["T5.1.1"]),
        "matched_controller_rows": controller_rows,
        "excluded_control_reference": exclusion,
        "slow_loop_host_latency_rows": slow_rows,
        "latency_contract": _latency_contract(dual_budget),
        "nonmixing_contract": {
            "global_score_or_leaderboard": None,
            "protocol_wallclock_scope": "finite-cutoff nonselective sBs common 700 us",
            "controller_scope": "matched two-level measurement-feedback sBs ten-cycle/100 us",
            "slow_loop_scope": "synthetic regime-estimator host software profile only",
            "idle_scope": "30 us sanity reference only",
            "e_outcomes_are_reset_events": False,
            "host_profile_applies_to_t4_4_controller_models": False,
            "configured_latency_is_measured_latency": False,
        },
    }
    semantic_errors = validate_payload(result)
    gates = {
        "all_parent_artifacts_are_passed_and_current": all(
            record["passed"] for record in integrity.values()
        ),
        "twelve_protocol_rows_cover_two_methods_six_lanes": len(protocol_rows) == 12,
        "protocol_rows_share_exact_700us_horizon": all(
            row["common_horizon_us"] == 700.0 for row in protocol_rows
        ),
        "protocol_cycle_and_microsecond_lifetimes_are_both_reported": all(
            row["logical_z_area_lifetime_protocol_cycles"] is not None
            and row["logical_z_area_lifetime_us"] is not None
            for row in protocol_rows
        ),
        "all_six_cycle_to_wallclock_ordering_reversals_are_retained": len(reversals) == 6
        and all(
            row["protocol_cycle_favors_autonomous"]
            and not row["wallclock_favors_autonomous"]
            for row in reversals
        ),
        "protocol_measurement_reset_and_active_gate_counts_are_explicit": all(
            all(
                row[name] is not None
                for name in (
                    "measurement_events",
                    "reset_events",
                    "active_gate_applications",
                )
            )
            for row in protocol_rows
        ),
        "idle_probe_is_zero_cost_and_not_ranked_at_700us": result["idle_reference"][
            "ranked_with_700us_protocol_lane"
        ]
        is False
        and all(
            result["idle_reference"][name] == 0
            for name in (
                "measurement_events",
                "reset_events",
                "active_gate_applications",
            )
        ),
        "ten_matched_controller_rows_report_cycle_us_events_and_cost": len(controller_rows)
        == 10,
        "controller_e_outcomes_are_not_substituted_for_resets": all(
            row["reset_events"] == 20 and row["observed_e_events"] != row["reset_events"]
            for row in controller_rows
        ),
        "controller_classical_latency_remains_unmeasured": all(
            row["classical_latency_us"] is None for row in controller_rows
        ),
        "two_cycle_control_reference_is_excluded_from_ten_cycle_lane": exclusion[
            "excluded_from_ten_cycle_lane"
        ],
        "six_host_profiles_are_estimator_only_and_below_software_ceiling": len(slow_rows)
        == 6
        and all(
            0.0
            < row["host_batch_median_us_per_update"]
            < row["host_software_latency_ceiling_us"]
            and row["physical_lifetime_us"] is None
            for row in slow_rows
        ),
        "configured_latency_is_labelled_assumption_not_measurement": result[
            "latency_contract"
        ]["configured_software_latency_model"]["measured_on_target_board"]
        is False,
        "all_target_board_and_physical_frontend_latency_fields_remain_null": all(
            row["value_us"] is None and row["measured_on_target_board"] is False
            for row in result["latency_contract"]["target_board_latency"]
            + result["latency_contract"]["physical_frontend"]
        ),
        "no_cross_lane_latency_or_performance_aggregate_exists": result[
            "latency_contract"
        ]["cross_lane_aggregate_latency_us"]
        is None
        and result["nonmixing_contract"]["global_score_or_leaderboard"] is None,
        "t5_1_4_fallback_claim_scope_is_preserved": result["active_algorithm_branch"]
        == FALLBACK_BRANCH_ID,
        "semantic_validator_accepts_only_complete_lane_local_report": semantic_errors == (),
        "report_is_read_only_without_new_performance_sampling": True,
    }
    result["gates"] = gates
    result["gate_summary"] = {
        "passed": sum(gates.values()),
        "total": len(gates),
        "failed": [name for name, value in gates.items() if not value],
    }
    result["status"] = "PASS" if all(gates.values()) else "FAIL"
    result["contract_sha256"] = _canonical_sha256(
        {
            "protocol_id": PROTOCOL_ID,
            "active_algorithm_branch": result["active_algorithm_branch"],
            "protocol_wallclock_rows": protocol_rows,
            "protocol_ordering_reversal": reversals,
            "idle_reference": result["idle_reference"],
            "matched_controller_rows": controller_rows,
            "excluded_control_reference": exclusion,
            "slow_loop_host_latency_rows": slow_rows,
            "latency_contract": result["latency_contract"],
            "nonmixing_contract": result["nonmixing_contract"],
        }
    )
    return result


CSV_FIELDS = (
    "row_type",
    "item_id",
    "lane_id",
    "strategy",
    "metric",
    "value",
    "unit_or_scope",
    "passed",
    "source_task",
    "source_artifact_sha256",
)


def _source_rows(
    result: Mapping[str, Any],
    parents: Mapping[str, Mapping[str, Any]],
    artifact_hashes: Mapping[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(row_type: str, item_id: str, **values: Any) -> None:
        row = {field: "" for field in CSV_FIELDS}
        row.update({"row_type": row_type, "item_id": item_id, **values})
        rows.append(row)

    for task_id, parent in parents.items():
        integrity = result["parent_integrity"][task_id]
        add(
            "parent_artifact",
            task_id,
            value=integrity["passed"],
            passed=integrity["passed"],
            source_task=task_id,
            source_artifact_sha256=artifact_hashes[task_id],
        )
        for name, value in _gate_entries(parent):
            add(
                "parent_gate",
                f"{task_id}:{name}",
                metric=name,
                value=value,
                passed=value,
                source_task=task_id,
                source_artifact_sha256=artifact_hashes[task_id],
            )
        for index, binding in enumerate(integrity["declared_file_bindings"]):
            add(
                "file_binding",
                f"{task_id}:{index}",
                metric=binding["path"],
                value=binding["actual_sha256"],
                unit_or_scope=binding["declared_sha256"],
                passed=binding["passed"],
                source_task=task_id,
                source_artifact_sha256=artifact_hashes[task_id],
            )
    for row in result["protocol_wallclock_rows"]:
        for metric in (
            "cycle_duration_us",
            "common_horizon_us",
            "full_cycles",
            "fidelity_area_lifetime_protocol_cycles",
            "fidelity_area_lifetime_us",
            "logical_z_area_lifetime_protocol_cycles",
            "logical_z_area_lifetime_us",
            "measurement_events",
            "reset_events",
            "active_gate_applications",
            "measurements_per_100us",
            "resets_per_100us",
            "active_gates_per_100us",
            "online_classical_latency_us",
        ):
            add(
                "protocol_metric",
                f"{row['lane_id']}:{row['strategy']}:{metric}",
                lane_id=row["lane_id"],
                strategy=row["strategy"],
                metric=metric,
                value=row[metric],
                source_task="T3.2.8",
                source_artifact_sha256=artifact_hashes["T3.2.8"],
            )
    for row in result["protocol_ordering_reversal"]:
        add(
            "ordering_audit",
            row["lane_id"],
            lane_id=row["lane_id"],
            metric="cycle_favors_autonomous_but_wallclock_does_not",
            value=f"{row['protocol_cycle_favors_autonomous']}/{row['wallclock_favors_autonomous']}",
            passed=row["protocol_cycle_favors_autonomous"]
            and not row["wallclock_favors_autonomous"],
            source_task="T3.2.8",
            source_artifact_sha256=artifact_hashes["T3.2.8"],
        )
    for row in result["matched_controller_rows"]:
        for metric in (
            "simulated_physical_time_us",
            "fidelity_effective_lifetime_cycles",
            "fidelity_effective_lifetime_us",
            "logical_z_effective_lifetime_cycles",
            "logical_z_effective_lifetime_us",
            "measurement_events",
            "reset_events",
            "active_gate_applications",
            "observed_e_events",
            "multilevel_leakage_events",
            "stored_scalars",
            "persistent_state_scalars",
            "analytic_macs_per_half_cycle",
            "classical_latency_us",
        ):
            add(
                "controller_metric",
                f"{row['lane_id']}:{row['strategy']}:{metric}",
                lane_id=row["lane_id"],
                strategy=row["strategy"],
                metric=metric,
                value=row[metric],
                source_task="T4.4.4",
                source_artifact_sha256=artifact_hashes["T4.4.4"],
            )
    for row in result["slow_loop_host_latency_rows"]:
        for metric in (
            "macs_per_update_proxy",
            "model_and_state_bytes",
            "transient_workspace_bytes",
            "host_batch_median_us_per_update",
            "host_software_latency_ceiling_us",
            "latency_to_ceiling_fraction",
            "physical_lifetime_us",
        ):
            add(
                "host_estimator_metric",
                f"{row['family']}:{metric}",
                lane_id=row["lane_id"],
                strategy=row["family"],
                metric=metric,
                value=row[metric],
                source_task="T4.1.1",
                source_artifact_sha256=artifact_hashes["T4.1.1"],
            )
    for row in result["latency_contract"]["target_board_latency"] + result[
        "latency_contract"
    ]["physical_frontend"]:
        add(
            "target_latency_null",
            row["field_id"],
            metric=row["name"],
            value=row["value_us"],
            unit_or_scope=row["evidence_class"],
            passed=row["value_us"] is None and row["measured_on_target_board"] is False,
            source_task="T2.4.1",
            source_artifact_sha256=artifact_hashes["T2.4.1"],
        )
    for name, value in result["gates"].items():
        add(
            "contract_gate",
            name,
            metric=name,
            value=value,
            passed=value,
            source_task=TASK_ID,
        )
    return rows


def write_artifacts(
    *,
    artifact_path: str | Path = DEFAULT_ARTIFACT,
    source_data_path: str | Path = DEFAULT_SOURCE_DATA,
) -> dict[str, Any]:
    parents = load_parent_artifacts()
    integrity = inspect_parent_integrity(parents)
    result = build_report(parents, integrity)
    artifact_hashes = {
        task_id: _sha256(path) for task_id, path in PARENT_ARTIFACTS.items()
    }
    result["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["implementation_sha256"] = implementation_sha256()
    result["artifact_bindings"] = [
        {
            "task_id": task_id,
            "path": path.as_posix(),
            "sha256": artifact_hashes[task_id],
            "machine_and_integrity_pass": integrity[task_id]["passed"],
        }
        for task_id, path in PARENT_ARTIFACTS.items()
    ]
    rows = _source_rows(result, parents, artifact_hashes)
    source = _repo_path(source_data_path)
    source.parent.mkdir(parents=True, exist_ok=True)
    with source.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    result["source_data"] = {
        "path": Path(source_data_path).as_posix(),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "row_count": len(rows),
        "row_types": sorted({row["row_type"] for row in rows}),
    }
    artifact = _repo_path(artifact_path)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--source-data", type=Path, default=DEFAULT_SOURCE_DATA)
    args = parser.parse_args(argv)
    result = write_artifacts(
        artifact_path=args.artifact, source_data_path=args.source_data
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "protocol_rows": len(result["protocol_wallclock_rows"]),
                "controller_rows": len(result["matched_controller_rows"]),
                "host_latency_rows": len(result["slow_loop_host_latency_rows"]),
                "gate_summary": result["gate_summary"],
                "source_rows": result["source_data"]["row_count"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONTROLLER_STRATEGIES",
    "DEFAULT_ARTIFACT",
    "DEFAULT_SOURCE_DATA",
    "PARENT_ARTIFACTS",
    "build_report",
    "current_parent_implementation_hashes",
    "implementation_sha256",
    "inspect_parent_integrity",
    "load_parent_artifacts",
    "validate_payload",
    "write_artifacts",
]
