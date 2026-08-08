"""T5.1.6 fail-closed experimental-feasibility consolidation.

This read-only report binds controller burden/cost evidence to software fallback
campaigns while preserving missing leakage, saturation, latency and device fields.
A PASS means feasibility constraints are reported completely; deployment or device
feasibility remains explicitly NOT_ESTABLISHED.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from cnn_fpga.benchmark.algorithm_success_falsification import FALLBACK_BRANCH_ID


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "T5.1.6"
SCHEMA_VERSION = "t5.1.6-experimental-feasibility-v1"
PROTOCOL_ID = "LANE-LOCAL-FEASIBILITY-BURDEN-SAFETY-V1"
DEFAULT_ARTIFACT = Path("docs/t5_1_6_experimental_feasibility.json")
DEFAULT_SOURCE_DATA = Path("docs/t5_1_6_experimental_feasibility_source_data.csv")

PARENT_ARTIFACTS: dict[str, Path] = {
    "T4.4.4": Path("docs/t4_4_4_teacher_student_gain_retention.json"),
    "T5.1.5": Path("docs/t5_1_5_time_cost_fairness.json"),
    "T4.3.3": Path("docs/t4_3_3_closed_loop_fault_recovery_validation.json"),
    "T4.2.3": Path("docs/t4_2_3_conservative_fallback_validation.json"),
    "T4.4.3": Path("docs/t4_4_3_low_dimensional_student_validation.json"),
    "T5.1.4": Path("docs/t5_1_4_algorithm_branch_verdict.json"),
}

BURDEN_KEYS = {
    "standard": "standard",
    "exact_budget_mf": "exact_budget_mf_agent_mean",
    "fresh_gru_teacher": "teacher",
    "handcrafted_recurrence": "handcrafted_recurrence",
    "distilled_student": "distilled_student",
}

CONTROLLER_STRATEGIES = tuple(BURDEN_KEYS)

EXPECTED_COMPONENT_STATUS_COUNTS = {
    "degraded": 64,
    "fallback": 1296,
    "healthy": 2050,
    "recovering": 622,
    "reset_required": 64,
}

EXPECTED_NONMIXING_CONTRACT = {
    "controller_occupancy_is_device_occupancy": False,
    "injected_leakage_fault_rate_is_physical_leakage_occupancy": False,
    "e_events_are_reset_events": False,
    "hard_bound_compliance_is_saturation_rate": False,
    "software_fault_campaign_is_device_safety_rate": False,
    "peak_lifetime_is_deployment_readiness": False,
}


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


def _declared_bindings(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    bindings: list[dict[str, str]] = []

    def add(role: str, record: Any) -> None:
        if isinstance(record, Mapping) and record.get("path") and record.get("sha256"):
            bindings.append(
                {
                    "role": role,
                    "path": str(record["path"]),
                    "declared_sha256": str(record["sha256"]),
                }
            )

    add("source_data", payload.get("source_data"))
    add("checkpoint", payload.get("checkpoint"))
    for index, row in enumerate(payload.get("artifact_bindings", ())):
        add(f"artifact:{index}", row)
    for index, row in enumerate(payload.get("implementation_bindings", ())):
        add(f"implementation:{index}", row)
    return bindings


def current_parent_implementation_hashes() -> dict[str, str]:
    from cnn_fpga.benchmark.algorithm_success_falsification import (
        implementation_sha256 as t514_hash,
    )
    from cnn_fpga.benchmark.closed_loop_fault_recovery_validation import (
        _implementation_sha256 as t433_hash,
    )
    from cnn_fpga.benchmark.conservative_fallback_validation import (
        _implementation_sha256 as t423_hash,
    )
    from cnn_fpga.benchmark.low_dimensional_student_distillation import (
        implementation_sha256 as t443_hash,
    )
    from cnn_fpga.benchmark.teacher_student_gain_retention import (
        implementation_sha256 as t444_hash,
    )
    from cnn_fpga.benchmark.time_cost_fairness import implementation_sha256 as t515_hash

    return {
        "T4.4.4": t444_hash(),
        "T5.1.5": t515_hash(),
        "T4.3.3": t433_hash(),
        "T4.2.3": t423_hash(),
        "T4.4.3": t443_hash(),
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
        for binding in _declared_bindings(payload):
            path = _repo_path(binding["path"])
            actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            checks.append(
                {
                    **binding,
                    "actual_sha256": actual,
                    "passed": actual == binding["declared_sha256"],
                }
            )
        current = payload.get("implementation_sha256") == hashes.get(task_id)
        record = {
            "machine_pass": _machine_pass(payload),
            "machine_gate_count": len(_gate_entries(payload)),
            "declared_file_bindings": checks,
            "all_declared_files_current": all(row["passed"] for row in checks),
            "declared_implementation_sha256": payload.get("implementation_sha256"),
            "current_implementation_sha256": hashes.get(task_id),
            "implementation_current": current,
        }
        record["passed"] = bool(
            record["machine_pass"]
            and record["all_declared_files_current"]
            and current
        )
        result[task_id] = record
    return result


def _controller_feasibility_rows(
    t444: Mapping[str, Any], t515: Mapping[str, Any]
) -> list[dict[str, Any]]:
    burden = t444["burden_summary"]["stochastic"]
    rows: list[dict[str, Any]] = []
    for source in t515["matched_controller_rows"]:
        lane = "primary" if source["cutoff"] == 12 else "confirmation"
        occupancy = burden[lane][BURDEN_KEYS[source["strategy"]]]
        row = {
            "lane_id": source["lane_id"],
            "cutoff": source["cutoff"],
            "strategy": source["strategy"],
            "p_g": occupancy["observed_ground_fraction"],
            "p_e": occupancy["observed_e_fraction"],
            "expected_e_events": occupancy[
                "expected_e_events_from_observed_fraction"
            ],
            "multilevel_leakage_occupancy": None,
            "multilevel_leakage_events": occupancy["multilevel_leakage_events"],
            "measurement_events": source["measurement_events"],
            "reset_events": source["reset_events"],
            "active_gate_applications": source["active_gate_applications"],
            "mean_control_residual_rms": source["mean_control_residual_rms"],
            "mean_parameter_slew_rms": source["mean_control_slew_rms"],
            "hard_residual_bounds_obeyed": True,
            "parameter_saturation_rate": None,
            "parameter_saturation_evidence": (
                "not_applicable_fixed_nominal_residual"
                if source["strategy"] == "standard"
                else "hard_bound_compliance_only_no_bound_hit_count"
            ),
            "fidelity_lifetime_cycles": source[
                "fidelity_effective_lifetime_cycles"
            ],
            "fidelity_lifetime_us": source["fidelity_effective_lifetime_us"],
            "logical_z_lifetime_cycles": source[
                "logical_z_effective_lifetime_cycles"
            ],
            "logical_z_lifetime_us": source["logical_z_effective_lifetime_us"],
            "stored_scalars": source["stored_scalars"],
            "analytic_macs_per_half_cycle": source[
                "analytic_macs_per_half_cycle"
            ],
            "classical_latency_us": source["classical_latency_us"],
            "target_hardware_measured": source["target_hardware_measured"],
            "parent_deployable_flag": source["deployable_in_parent"],
            "device_feasibility_status": "NOT_ESTABLISHED",
        }
        rows.append(row)
    for cutoff in (12, 16):
        lane_rows = [row for row in rows if row["cutoff"] == cutoff]
        peak = max(row["fidelity_lifetime_us"] for row in lane_rows)
        for row in lane_rows:
            row["peak_fidelity_lifetime_in_lane"] = row["fidelity_lifetime_us"] == peak
            row["peak_can_support_deployment_claim"] = False
    return rows


def _aggregate_action_counts(runs: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for run in runs:
        counter.update({str(key): int(value) for key, value in run["action_counts"].items()})
    return dict(sorted(counter.items()))


def _fault_campaign_rows(t433: Mapping[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for run in t433["summary"]["per_run"]:
        grouped.setdefault(str(run["scenario"]), []).append(run)
    rows: list[dict[str, Any]] = []
    for scenario in t433["summary"]["scenarios"]:
        runs = grouped[scenario]
        cycles = sum(int(run["cycles_executed"]) for run in runs)
        fallback = sum(int(run["fallback_cycles"]) for run in runs)
        resets = sum(int(run["reset_request_cycles"]) for run in runs)
        unsafe = sum(
            int(run["blocking_fault_with_correction_count"]) for run in runs
        )
        undefined = sum(int(run["undefined_action_count"]) for run in runs)
        ack = sum(int(run["ack_timeout_cycles"]) for run in runs)
        readback = sum(int(run["awaiting_readback_cycles"]) for run in runs)
        rows.append(
            {
                "scenario": scenario,
                "run_count": len(runs),
                "cycles": cycles,
                "fallback_cycles": fallback,
                "fallback_rate": fallback / cycles,
                "reset_request_cycles": resets,
                "reset_request_rate": resets / cycles,
                "unsafe_action_cycles": unsafe,
                "unsafe_action_rate": unsafe / cycles,
                "undefined_action_cycles": undefined,
                "undefined_action_rate": undefined / cycles,
                "ack_timeout_cycles": ack,
                "awaiting_readback_cycles": readback,
                "action_counts": _aggregate_action_counts(runs),
                "scope": "closed_loop_software_fault_recovery_not_rtl_board_or_device",
            }
        )
    return rows


def _safety_summary(t433: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    cycles = sum(int(row["cycles"]) for row in rows)
    fallback = sum(int(row["fallback_cycles"]) for row in rows)
    resets = sum(int(row["reset_request_cycles"]) for row in rows)
    unsafe = sum(int(row["unsafe_action_cycles"]) for row in rows)
    undefined = sum(int(row["undefined_action_cycles"]) for row in rows)
    return {
        "campaign_cycles": cycles,
        "scenario_count": len(rows),
        "run_count": t433["summary"]["runs"],
        "fallback_cycles": fallback,
        "fallback_rate": fallback / cycles,
        "reset_request_cycles": resets,
        "reset_request_rate": resets / cycles,
        "unsafe_action_cycles": unsafe,
        "observed_unsafe_action_rate": unsafe / cycles,
        "undefined_action_cycles": undefined,
        "observed_undefined_action_rate": undefined / cycles,
        "statistical_population_upper_bound": None,
        "upper_bound_reason": (
            "targeted deterministic software campaigns are coverage evidence, not iid "
            "samples from a declared device-fault population"
        ),
        "blocking_fault_with_correction_count": t433["summary"][
            "blocking_fault_with_correction_count"
        ],
        "frame_out_of_range_count": t433["summary"]["frame_out_of_range_count"],
        "claim_scope": "observed rate in registered software campaigns only",
    }


def _component_fallback(t423: Mapping[str, Any]) -> dict[str, Any]:
    diagnostics = t423["diagnostics"]
    statuses = diagnostics["status_counts"]
    total = sum(int(value) for value in statuses.values())
    return {
        "cycles": total,
        "status_counts": statuses,
        "fault_flag_counts": diagnostics["fault_flag_counts"],
        "healthy_fraction": statuses["healthy"] / total,
        "nonhealthy_fraction": (total - statuses["healthy"]) / total,
        "scope": "component_fallback_taxonomy_not_controller_lifetime_or_device_rate",
    }


MISSING_EVIDENCE = (
    {
        "field": "controller_multilevel_leakage_occupancy_and_events",
        "status": "MISSING",
        "required_task": "T5.2/T5.3",
    },
    {
        "field": "controller_parameter_bound_hit_or_saturation_rate",
        "status": "MISSING",
        "required_task": "T5.4/T5.5",
    },
    {
        "field": "matched_controller_classical_latency",
        "status": "MISSING",
        "required_task": "T5.5",
    },
    {
        "field": "target_board_core_transport_and_end_to_end_latency",
        "status": "MISSING",
        "required_task": "T6",
    },
    {
        "field": "physical_measurement_adc_awg_and_action_latency",
        "status": "MISSING",
        "required_task": "T6/device integration",
    },
    {
        "field": "device_calibrated_reset_fidelity_and_reset_storm_burden",
        "status": "MISSING",
        "required_task": "T5.2/T6.4.3",
    },
    {
        "field": "single_matched_finite_energy_closed_loop_joining_lifetime_and_fault_rates",
        "status": "MISSING",
        "required_task": "T5.2--T5.4",
    },
)


def validate_payload(payload: Mapping[str, Any]) -> tuple[str, ...]:
    errors: list[str] = []
    rows = payload.get("controller_feasibility_rows", ())
    if len(rows) != 10:
        errors.append("controller feasibility table must contain ten rows")
    if {
        (row.get("cutoff"), row.get("strategy")) for row in rows
    } != {
        (cutoff, strategy)
        for cutoff in (12, 16)
        for strategy in CONTROLLER_STRATEGIES
    }:
        errors.append("controller feasibility lane membership changed")
    for row in rows:
        if abs(row.get("p_g", 0.0) + row.get("p_e", 0.0) - 1.0) > 1e-12:
            errors.append("p(g) and p(e) do not sum to one")
        if abs(
            float(row.get("expected_e_events", -1.0))
            - float(row.get("p_e", -1.0)) * 20.0
        ) > 1e-12:
            errors.append("expected e-event burden is inconsistent")
        if row.get("multilevel_leakage_occupancy") is not None or row.get(
            "multilevel_leakage_events"
        ) is not None:
            errors.append("missing multilevel leakage evidence was filled")
        if row.get("reset_events") != 20:
            errors.append("protocol reset burden was hidden or replaced")
        if row.get("mean_parameter_slew_rms") is None:
            errors.append("parameter slew is missing")
        if row.get("parameter_saturation_rate") is not None:
            errors.append("unmeasured parameter saturation rate was filled")
        if row.get("classical_latency_us") is not None:
            errors.append("unmeasured matched controller latency was filled")
        if row.get("device_feasibility_status") != "NOT_ESTABLISHED":
            errors.append("controller was promoted to device feasible")
        if row.get("peak_can_support_deployment_claim") is not False:
            errors.append("peak lifetime was allowed to hide feasibility costs")
    peaks = [row for row in rows if row.get("peak_fidelity_lifetime_in_lane")]
    if len(peaks) != 2 or {row["cutoff"] for row in peaks} != {12, 16}:
        errors.append("one peak row per cutoff is not retained")
    for cutoff in (12, 16):
        lane_rows = [row for row in rows if row.get("cutoff") == cutoff]
        marked = [row for row in lane_rows if row.get("peak_fidelity_lifetime_in_lane")]
        if lane_rows and (
            len(marked) != 1
            or marked[0].get("fidelity_lifetime_us")
            != max(row.get("fidelity_lifetime_us", float("-inf")) for row in lane_rows)
        ):
            errors.append("marked controller peak is not the lane maximum")
    if any(
        peak.get("stored_scalars") is None
        or peak.get("analytic_macs_per_half_cycle") is None
        or peak.get("classical_latency_us") is not None
        for peak in peaks
    ):
        errors.append("peak lifetime cost/null fields are incomplete")

    campaign = payload.get("fault_campaign_rows", ())
    if len(campaign) != 8 or any(row.get("run_count") != 4 for row in campaign):
        errors.append("fault campaign must retain eight scenarios and four runs each")
    for row in campaign:
        cycles = row.get("cycles", 0)
        if cycles != 95984:
            errors.append("fault scenario cycle denominator changed")
        if abs(row.get("fallback_rate", -1.0) - row.get("fallback_cycles", 0) / cycles) > 1e-15:
            errors.append("fallback rate denominator is inconsistent")
        if row.get("unsafe_action_rate") != 0.0 or row.get("undefined_action_rate") != 0.0:
            errors.append("unsafe or undefined action counter changed")
        if row.get("unsafe_action_cycles") != 0 or row.get("undefined_action_cycles") != 0:
            errors.append("unsafe or undefined action count changed")
    by_scenario = {row["scenario"]: row for row in campaign}
    if by_scenario.get("host_timeout", {}).get("fallback_cycles") != 11232:
        errors.append("host-timeout fallback burden was hidden")
    if by_scenario.get("leakage_reset", {}).get("reset_request_cycles") != 4:
        errors.append("leakage reset request burden was hidden")
    pause = by_scenario.get("communication_pause_ack_loss", {})
    if pause.get("ack_timeout_cycles") != 1596 or pause.get(
        "awaiting_readback_cycles"
    ) != 1604:
        errors.append("ack/readback uncertainty burden was hidden")

    safety = payload.get("safety_summary", {})
    if safety.get("campaign_cycles") != 767872:
        errors.append("global safety denominator changed")
    if safety.get("observed_unsafe_action_rate") != 0.0 or safety.get(
        "observed_undefined_action_rate"
    ) != 0.0:
        errors.append("global observed safety rate changed")
    if safety.get("statistical_population_upper_bound") is not None:
        errors.append("deterministic campaign was given an iid population bound")
    if safety.get("fallback_cycles") != 11552 or safety.get("reset_request_cycles") != 4:
        errors.append("global fallback or reset burden changed")
    if abs(
        safety.get("fallback_rate", -1.0)
        - safety.get("fallback_cycles", 0) / safety.get("campaign_cycles", 1)
    ) > 1e-15 or abs(
        safety.get("reset_request_rate", -1.0)
        - safety.get("reset_request_cycles", 0) / safety.get("campaign_cycles", 1)
    ) > 1e-15:
        errors.append("global fallback or reset rate denominator is inconsistent")

    component = payload.get("component_fallback", {})
    if component.get("cycles") != 4096 or sum(
        component.get("status_counts", {}).values()
    ) != 4096:
        errors.append("component fallback taxonomy is incomplete")
    if component.get("status_counts") != EXPECTED_COMPONENT_STATUS_COUNTS:
        errors.append("component fallback status distribution changed")
    student = payload.get("student_fail_closed_contract", {})
    if not (
        student.get("safe_baseline") == "reset state and exact zero physical residual"
        and student.get("leakage_resets_initial_state") is True
        and student.get("target_latency_cycles") is None
        and student.get("rtl_measured") is False
        and student.get("board_measured") is False
    ):
        errors.append("student fail-closed or unmeasured contract changed")
    if payload.get("nonmixing_contract") != EXPECTED_NONMIXING_CONTRACT:
        errors.append("nonmixing feasibility contract changed")
    if payload.get("active_algorithm_branch") != FALLBACK_BRANCH_ID:
        errors.append("T5.1.4 fallback branch was rewritten")
    if payload.get("deployment_readiness") != "NOT_ESTABLISHED":
        errors.append("overall deployment readiness was promoted")
    if tuple(payload.get("missing_evidence", ())) != MISSING_EVIDENCE:
        errors.append("missing-evidence ledger changed")
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
    controller = _controller_feasibility_rows(parents["T4.4.4"], parents["T5.1.5"])
    campaign = _fault_campaign_rows(parents["T4.3.3"])
    safety = _safety_summary(parents["T4.3.3"], campaign)
    student_contract = parents["T4.4.3"]["student_artifact"]["runtime_replay"][
        "online_contract"
    ]
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "protocol_id": PROTOCOL_ID,
        "pass_semantics": (
            "p(g), e/leakage availability, reset, slew/saturation availability, fallback "
            "and observed unsafe-action rates are reported with provenance and missing "
            "fields preserved; deployment or device feasibility is not established"
        ),
        "active_algorithm_branch": parents["T5.1.4"]["active_branch"]["branch_id"],
        "deployment_readiness": "NOT_ESTABLISHED",
        "parent_integrity": dict(integrity),
        "controller_feasibility_rows": controller,
        "fault_campaign_rows": campaign,
        "safety_summary": safety,
        "component_fallback": _component_fallback(parents["T4.2.3"]),
        "student_fail_closed_contract": {
            "safe_baseline": student_contract["safe_baseline"],
            "leakage_resets_initial_state": parents["T4.4.3"]["student_artifact"][
                "runtime_replay"
            ]["leakage_resets_initial_state"],
            "target_latency_cycles": student_contract["target_latency_cycles"],
            "rtl_measured": student_contract["rtl_measured"],
            "board_measured": student_contract["board_measured"],
        },
        "missing_evidence": list(MISSING_EVIDENCE),
        "nonmixing_contract": {
            "controller_occupancy_is_device_occupancy": False,
            "injected_leakage_fault_rate_is_physical_leakage_occupancy": False,
            "e_events_are_reset_events": False,
            "hard_bound_compliance_is_saturation_rate": False,
            "software_fault_campaign_is_device_safety_rate": False,
            "peak_lifetime_is_deployment_readiness": False,
        },
    }
    errors = validate_payload(result)
    gates = {
        "all_parent_artifacts_are_passed_and_current": all(
            row["passed"] for row in integrity.values()
        ),
        "ten_controller_rows_report_pg_pe_reset_slew_and_cost": len(controller) == 10,
        "all_controller_pg_pe_pairs_are_normalized": all(
            abs(row["p_g"] + row["p_e"] - 1.0) <= 1e-12 for row in controller
        ),
        "multilevel_leakage_occupancy_remains_unavailable": all(
            row["multilevel_leakage_occupancy"] is None
            and row["multilevel_leakage_events"] is None
            for row in controller
        ),
        "reset_burden_is_explicit_and_not_replaced_by_e_events": all(
            row["reset_events"] == 20 and row["expected_e_events"] != 20
            for row in controller
        ),
        "parameter_slew_is_reported_but_saturation_rate_remains_unavailable": all(
            row["mean_parameter_slew_rms"] is not None
            and row["parameter_saturation_rate"] is None
            for row in controller
        ),
        "one_peak_lifetime_row_per_cutoff_retains_full_cost_and_nulls": len(
            [row for row in controller if row["peak_fidelity_lifetime_in_lane"]]
        )
        == 2
        and all(
            row["peak_can_support_deployment_claim"] is False
            for row in controller
            if row["peak_fidelity_lifetime_in_lane"]
        ),
        "controller_latency_and_device_feasibility_are_not_invented": all(
            row["classical_latency_us"] is None
            and row["device_feasibility_status"] == "NOT_ESTABLISHED"
            for row in controller
        ),
        "fault_campaign_covers_eight_scenarios_four_runs_and_767872_cycles": len(campaign)
        == 8
        and all(row["run_count"] == 4 for row in campaign)
        and safety["campaign_cycles"] == 767872,
        "fallback_and_reset_rates_use_full_scenario_denominators": all(
            row["fallback_rate"] == row["fallback_cycles"] / row["cycles"]
            and row["reset_request_rate"] == row["reset_request_cycles"] / row["cycles"]
            for row in campaign
        ),
        "observed_unsafe_and_undefined_action_rates_are_zero": safety[
            "observed_unsafe_action_rate"
        ]
        == 0.0
        and safety["observed_undefined_action_rate"] == 0.0,
        "deterministic_campaign_is_not_given_fake_population_ci": safety[
            "statistical_population_upper_bound"
        ]
        is None,
        "host_timeout_fallback_burden_is_retained": next(
            row for row in campaign if row["scenario"] == "host_timeout"
        )["fallback_cycles"]
        == 11232,
        "leakage_reset_and_ack_uncertainty_burdens_are_retained": next(
            row for row in campaign if row["scenario"] == "leakage_reset"
        )["reset_request_cycles"]
        == 4
        and next(
            row
            for row in campaign
            if row["scenario"] == "communication_pause_ack_loss"
        )["ack_timeout_cycles"]
        == 1596,
        "component_fallback_taxonomy_retains_all_4096_cycles": result[
            "component_fallback"
        ]["cycles"]
        == 4096,
        "student_safe_baseline_is_fail_closed_and_unmeasured": result[
            "student_fail_closed_contract"
        ]["safe_baseline"]
        == "reset state and exact zero physical residual"
        and result["student_fail_closed_contract"]["target_latency_cycles"] is None
        and not result["student_fail_closed_contract"]["rtl_measured"]
        and not result["student_fail_closed_contract"]["board_measured"],
        "t5_1_4_fallback_branch_is_preserved": result["active_algorithm_branch"]
        == FALLBACK_BRANCH_ID,
        "seven_missing_feasibility_fields_are_explicit": len(result["missing_evidence"])
        == 7
        and all(row["status"] == "MISSING" for row in result["missing_evidence"]),
        "overall_deployment_readiness_remains_not_established": result[
            "deployment_readiness"
        ]
        == "NOT_ESTABLISHED",
        "semantic_validator_accepts_only_complete_fail_closed_report": errors == (),
        "report_is_read_only_without_new_physical_or_fault_sampling": True,
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
            "deployment_readiness": result["deployment_readiness"],
            "controller_feasibility_rows": controller,
            "fault_campaign_rows": campaign,
            "safety_summary": safety,
            "component_fallback": result["component_fallback"],
            "student_fail_closed_contract": result["student_fail_closed_contract"],
            "missing_evidence": result["missing_evidence"],
            "nonmixing_contract": result["nonmixing_contract"],
        }
    )
    return result


CSV_FIELDS = (
    "row_type",
    "item_id",
    "lane_or_scenario",
    "strategy",
    "metric",
    "value",
    "status_or_scope",
    "passed",
    "source_task",
    "source_artifact_sha256",
)


def _source_rows(
    result: Mapping[str, Any],
    parents: Mapping[str, Mapping[str, Any]],
    hashes: Mapping[str, str],
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
            source_artifact_sha256=hashes[task_id],
        )
        for name, value in _gate_entries(parent):
            add(
                "parent_gate",
                f"{task_id}:{name}",
                metric=name,
                value=value,
                passed=value,
                source_task=task_id,
                source_artifact_sha256=hashes[task_id],
            )
        for index, binding in enumerate(integrity["declared_file_bindings"]):
            add(
                "file_binding",
                f"{task_id}:{index}",
                metric=binding["path"],
                value=binding["actual_sha256"],
                status_or_scope=binding["declared_sha256"],
                passed=binding["passed"],
                source_task=task_id,
                source_artifact_sha256=hashes[task_id],
            )
    for row in result["controller_feasibility_rows"]:
        for metric in (
            "p_g",
            "p_e",
            "expected_e_events",
            "multilevel_leakage_occupancy",
            "multilevel_leakage_events",
            "reset_events",
            "mean_control_residual_rms",
            "mean_parameter_slew_rms",
            "parameter_saturation_rate",
            "fidelity_lifetime_us",
            "logical_z_lifetime_us",
            "stored_scalars",
            "analytic_macs_per_half_cycle",
            "classical_latency_us",
            "peak_fidelity_lifetime_in_lane",
            "device_feasibility_status",
        ):
            add(
                "controller_feasibility",
                f"{row['lane_id']}:{row['strategy']}:{metric}",
                lane_or_scenario=row["lane_id"],
                strategy=row["strategy"],
                metric=metric,
                value=row[metric],
                status_or_scope=row["device_feasibility_status"],
                source_task="T4.4.4/T5.1.5",
                source_artifact_sha256=hashes["T5.1.5"],
            )
    for row in result["fault_campaign_rows"]:
        for metric in (
            "cycles",
            "fallback_cycles",
            "fallback_rate",
            "reset_request_cycles",
            "reset_request_rate",
            "unsafe_action_cycles",
            "unsafe_action_rate",
            "undefined_action_cycles",
            "undefined_action_rate",
            "ack_timeout_cycles",
            "awaiting_readback_cycles",
        ):
            add(
                "fault_campaign",
                f"{row['scenario']}:{metric}",
                lane_or_scenario=row["scenario"],
                metric=metric,
                value=row[metric],
                status_or_scope=row["scope"],
                source_task="T4.3.3",
                source_artifact_sha256=hashes["T4.3.3"],
            )
    for field in result["missing_evidence"]:
        add(
            "missing_evidence",
            field["field"],
            metric=field["field"],
            value="",
            status_or_scope=f"{field['status']}:{field['required_task']}",
            passed=False,
            source_task=TASK_ID,
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
    hashes = {task: _sha256(path) for task, path in PARENT_ARTIFACTS.items()}
    result["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["implementation_sha256"] = implementation_sha256()
    result["artifact_bindings"] = [
        {
            "task_id": task,
            "path": path.as_posix(),
            "sha256": hashes[task],
            "machine_and_integrity_pass": integrity[task]["passed"],
        }
        for task, path in PARENT_ARTIFACTS.items()
    ]
    rows = _source_rows(result, parents, hashes)
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
                "deployment_readiness": result["deployment_readiness"],
                "controller_rows": len(result["controller_feasibility_rows"]),
                "fault_scenarios": len(result["fault_campaign_rows"]),
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
    "MISSING_EVIDENCE",
    "EXPECTED_COMPONENT_STATUS_COUNTS",
    "EXPECTED_NONMIXING_CONTRACT",
    "PARENT_ARTIFACTS",
    "build_report",
    "current_parent_implementation_hashes",
    "implementation_sha256",
    "inspect_parent_integrity",
    "load_parent_artifacts",
    "validate_payload",
    "write_artifacts",
]
