from __future__ import annotations

import copy
import csv
import json

from cnn_fpga.benchmark import experiment_relevance_reviewer_contract as contract


def test_report_passes_all_gates_and_mutations() -> None:
    report = contract.build_report()
    assert report["verdict"] == contract.VERDICT
    assert report["gate_summary"] == {"passed": 24, "total": 24}
    assert all(report["gates"].values())
    audit = report["semantic_mutation_audit"]
    assert audit["count"] == audit["detected"] == 24
    assert {case["target_gate"] for case in audit["cases"]} == set(report["gates"])


def test_package_is_preemptive_and_not_falsely_submission_ready() -> None:
    report = contract.build_report()
    assert report["reviewer_context"]["comment_id"] == "PRQ-HW-1"
    assert report["response_package"]["package_readiness"] == "draft_with_placeholders"
    assert report["response_package"]["missing_information"] == ["ACTUAL_REVIEWER_ID_AND_VERBATIM_WORDING"]


def test_evidence_ladder_is_seven_level_and_ordered() -> None:
    report = contract.build_report()
    assert [row["level"] for row in report["evidence_ladder"]] == [
        "LITERATURE_FACT",
        "OFFICIAL_CODE_REPRODUCTION",
        "PROJECT_NATIVE_SIMULATION",
        "MOCK_SOFTWARE_HIL",
        "PREBOARD_DIGITAL_QUALIFICATION",
        "PHYSICAL_BOARD_MEASUREMENT",
        "QUANTUM_HARDWARE_OR_REAL_GKP_DATA",
    ]
    assert all(row["allowed"] and row["forbidden"] for row in report["evidence_ladder"])


def test_literature_and_official_code_are_not_project_measurements() -> None:
    ladder = {row["level"]: row for row in contract.build_report()["evidence_ladder"]}
    assert ladder["LITERATURE_FACT"]["evidence"] == {
        "literature_value_cells": 57,
        "literature_only_cells": 162,
        "null_not_reported_cells": 107,
    }
    official = ladder["OFFICIAL_CODE_REPRODUCTION"]["evidence"]
    assert official["count"] == 2
    assert official["physical_measurement"] is False


def test_project_native_simulation_preserves_depth_and_negative_result() -> None:
    evidence = contract.build_report()["evidence_ladder"][2]["evidence"]
    assert evidence["aqec_seed_clusters"] == 144
    assert evidence["aqec_source_rows"] == 144_152
    assert evidence["multimode_verdict"] == "NO_GO_MULTIMODE_CAUSAL_HEADROOM"
    assert evidence["multimode_relative_improvement"] == 0.0


def test_mock_hil_is_not_real_board_hil() -> None:
    evidence = contract.build_report()["evidence_ladder"][3]["evidence"]
    assert evidence["software_orchestrator"]
    assert evidence["mock_backend"]
    assert evidence["placeholder_board_backend"]
    assert evidence["real_board_hil"] is False


def test_preboard_digital_evidence_is_exact_but_not_measured() -> None:
    evidence = contract.build_report()["evidence_ladder"][4]["evidence"]
    assert evidence["formal_gates"] == 17
    assert evidence["formal_mutants"] == 21
    assert evidence["cxxrtl_cycles"] == 1_000_000
    assert evidence["latency_violations"] == evidence["undefined_actions"] == evidence["silent_overflow"] == 0
    assert evidence["place_route_seeds"] == 3
    assert evidence["measured"] is False


def test_physical_board_and_quantum_hardware_are_absent() -> None:
    ladder = {row["level"]: row for row in contract.build_report()["evidence_ladder"]}
    physical = ladder["PHYSICAL_BOARD_MEASUREMENT"]["evidence"]
    assert physical["all_null"]
    assert physical["field_count"] == physical["null_count"]
    assert physical["historical_candidate_programmed"] is False
    assert physical["historical_measurements_collected"] is False
    quantum = ladder["QUANTUM_HARDWARE_OR_REAL_GKP_DATA"]["evidence"]
    assert set(quantum["phase8_statuses"].values()) == {"Todo"}
    assert quantum["real_gkp_data"] is quantum["quantum_control_chain"] is False


def test_nontransfer_and_terminology_contracts_are_fail_closed() -> None:
    report = contract.build_report()
    assert set(report["nontransfer_contract"].values()) == {False}
    assert report["terminology_contract"]["experimental_gkp_qec_claim"] is False
    assert not any(report["manuscript_audit"]["forbidden_phrase_presence"].values())
    mutated = copy.deepcopy(report)
    mutated["nontransfer_contract"]["mock_hil_to_board_measurement"] = True
    assert not contract.evaluate_gates(mutated)["G14_all_cross_level_evidence_transfers_are_forbidden"]


def test_response_directly_concedes_boundary_and_keeps_contribution() -> None:
    report = contract.build_report()
    text = report["response_package"]["english_response"]
    assert "do not describe this work as experimental GKP quantum error correction" in text
    assert "hardware contribution is narrower but substantive" in text
    assert "every physical measurement field is null" in text
    assert not any(phrase.lower() in text.lower() for phrase in report["forbidden_response_phrases"])


def test_rows_are_lossless_unique_and_cover_every_state() -> None:
    report = contract.build_report()
    rows = report["response_rows"]
    assert len(rows) == 24
    assert len({row["row_id"] for row in rows}) == 24
    assert {row["response_state"] for row in rows} == contract.RESPONSE_STATES
    assert all(row["source_ids"] and row["claim"] and row["boundary"] for row in rows)


def test_written_report_source_data_and_markdown_verify() -> None:
    report = contract.build_report()
    contract.write_outputs(report)
    with contract.DEFAULT_SOURCE_DATA.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows == contract._source_rows(report)
    stored = json.loads(contract.DEFAULT_REPORT.read_text(encoding="utf-8"))
    assert stored["source_data"]["rows"] == 24
    ok, checks = contract.verify_report()
    assert ok, checks
    assert all(checks.values())
