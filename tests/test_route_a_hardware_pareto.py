from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path

import pytest

from cnn_fpga.benchmark import route_a_hardware_pareto as pareto


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "docs" / "t6_9_1_route_a_hardware_pareto.json"


def _report() -> dict:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _profiles() -> dict[str, dict]:
    return {row["profile_id"]: row for row in _report()["profiles"]}


def test_current_report_recomputes_and_passes_every_gate() -> None:
    report = _report()
    pareto.verify_report(report)
    assert report["verdict"] == "PASS_ROUTE_A_INTEGRATED_THREE_SEED_PR_ESTIMATE_NOT_BOARD_MEASURED"
    assert report["gate_summary"] == {"passed": 15, "failed": 0}
    assert report["target"]["device"] == "GW2AR-LV18QN88C8/I7"
    assert report["target"]["seeds"] == [1, 7, 19]


def test_both_real_profiles_have_three_passing_routes_and_expected_resources() -> None:
    profiles = _profiles()
    assert set(profiles) == set(pareto.PROFILES)
    for profile in profiles.values():
        assert [row["seed"] for row in profile["place_route"]] == [1, 7, 19]
        assert all(row["route_status"] == "PASS" and row["timing_pass"] for row in profile["place_route"])
        assert min(row["achieved_fmax_mhz"] for row in profile["place_route"]) > 27.0
        resources = profile["summary"]["resources_max_across_seeds"]
        assert resources["BSRAM"]["used"] == 8
        assert all(0 < resources[name]["used"] <= resources[name]["available"] for name in pareto.RESOURCE_NAMES)

    core = profiles["route_a_core_no_student"]
    student = profiles["route_a_plus_student_sidecar"]
    assert core["summary"]["resources_max_across_seeds"]["LUT4"]["used"] == 3859
    assert core["summary"]["resources_max_across_seeds"]["DFF"]["used"] == 1069
    assert student["summary"]["resources_max_across_seeds"]["LUT4"]["used"] == 4889
    assert student["summary"]["resources_max_across_seeds"]["DFF"]["used"] == 1210


def test_structural_audit_proves_banks_and_optional_student_survive() -> None:
    profiles = _profiles()
    core = profiles["route_a_core_no_student"]
    student = profiles["route_a_plus_student_sidecar"]
    assert core["structural_netlist"]["sdpx9b_cells"] == 8
    assert student["structural_netlist"]["sdpx9b_cells"] == 8
    assert core["structural_netlist"]["student_hierarchy_present"] is False
    assert student["structural_netlist"]["student_hierarchy_present"] is True
    assert core["synthesis"]["cell_counts"]["MULT18X18"] == 1
    assert student["synthesis"]["cell_counts"]["MULT18X18"] == 2
    assert all(profile["student_drives_fast_action"] is False for profile in profiles.values())
    assert _report()["pareto_decision"]["selected_profile"] == "route_a_core_no_student"


def test_latency_deadline_and_power_remain_estimates_not_measurements() -> None:
    for profile in _profiles().values():
        latency = profile["source_to_action_latency_model"]
        assert latency["cycles"] == 6
        assert latency["initiation_interval_cycles"] == 1
        assert latency["at_enforced_27mhz_ns"] == pytest.approx(222.22222222222223)
        assert latency["deadline_margin_us_at_27mhz"] == pytest.approx(1.2777777777777777)
        assert latency["deadline_miss_count"] is None
        assert "P&R" in latency["evidence_level"]

        power = profile["dynamic_power_estimate"]
        values = power["dynamic_power_mw_sensitivity"]
        assert values["low"] < values["nominal"] < values["high"]
        assert power["evidence_level"] == "analytic_switching_capacitance_sensitivity_not_vendor_power"
        assert power["static_power_mw"] is None
        assert power["vendor_power_mw"] is None
        assert power["board_measured_power_mw"] is None

    boundary = _report()["evidence_boundary"]
    assert boundary["board_measured"] is False
    assert boundary["board_deadline_miss"] is None
    assert boundary["measured_source_to_action_ns"] is None
    assert boundary["measured_power_mw"] is None
    assert boundary["speed_advantage"] == "PROHIBITED_PENDING_T6.9.2"


def test_all_sources_tool_reports_and_csv_are_live_hash_bound() -> None:
    report = _report()
    for binding in report["source_bindings"] + report["durable_artifacts"] + [report["source_data"]]:
        path = ROOT / binding["path"]
        assert path.is_file()
        assert _sha256(path) == binding["sha256"]
        assert path.stat().st_size == binding["bytes"]
    with (ROOT / report["source_data"]["path"]).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert {(row["profile_id"], int(row["seed"])) for row in rows} == {
        (profile, seed) for profile in pareto.PROFILES for seed in pareto.SEEDS
    }


def test_semantic_mutations_fail_closed_and_board_promotion_is_rejected() -> None:
    report = _report()
    audit = report["semantic_mutation_audit"]
    assert audit["count"] == audit["detected"] == 15
    assert {row["target_gate"] for row in audit["cases"]} == set(report["gates"])
    assert all(row["rejected"] for row in audit["cases"])

    forged = deepcopy(report)
    forged["evidence_boundary"]["board_measured"] = True
    forged["evidence_boundary"]["speed_advantage"] = "ESTABLISHED"
    with pytest.raises(ValueError, match="gates/verdict"):
        pareto.verify_report(forged)
