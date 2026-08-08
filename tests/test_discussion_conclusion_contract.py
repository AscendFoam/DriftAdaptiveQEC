from __future__ import annotations

import copy
import csv
import json

from cnn_fpga.benchmark import discussion_conclusion_contract as contract


def test_report_passes_all_gates() -> None:
    report = contract.build_report()
    assert report["verdict"] == contract.VERDICT
    assert report["gate_summary"] == {"passed": 22, "total": 22}
    assert all(report["gates"].values())


def test_discussion_rows_are_complete_and_unique() -> None:
    report = contract.build_report()
    rows = report["discussion_rows"]
    assert len(rows) == 27
    assert len({row["row_id"] for row in rows}) == 27
    assert {row["discussion_state"] for row in rows} == set(contract.DISCUSSION_STATES)


def test_subsections_and_transition_stages_are_ordered() -> None:
    manuscript = contract.build_report()["manuscript"]
    assert manuscript["subsections"] == list(contract.REQUIRED_SUBSECTIONS)
    assert manuscript["transition_stages"] == list(contract.REQUIRED_TRANSITION_STAGES)


def test_physical_and_training_nonclaims_are_explicit() -> None:
    checks = contract.build_report()["manuscript"]["checks"]
    assert checks["external_validity"]
    assert checks["explicit_nonclaims"]
    assert checks["prohibited_assertions_absent"]


def test_board_and_v5_absence_are_live() -> None:
    parent = contract.build_report()["parent_state"]
    assert parent["board"]["measured_field_count"] == 42
    assert parent["board"]["nonnull_field_count"] == 0
    assert len(parent["board"]["false_external_prerequisites"]) == 6
    assert parent["v5"] == {"dropped_tasks": 20, "downstream_outputs": 0, "formal_artifacts_exist": False}


def test_previous_prose_contracts_remain_live() -> None:
    parent = contract.build_report()["parent_state"]
    assert all(parent["previous_contracts_live"].values())
    assert parent["verdicts"]["results"] == "PASS_RESULTS_COMPLETE_NEGATIVE_AND_SECONDARY_BOUNDARIES"


def test_each_targeted_mutation_is_rejected() -> None:
    audit = contract.build_report()["semantic_mutation_audit"]
    assert audit["count"] == audit["detected"] == 22
    assert all(case["rejected"] for case in audit["cases"])


def test_board_null_mutation_fails_closed() -> None:
    report = contract.build_report()
    mutated = copy.deepcopy(report)
    mutated["parent_state"]["board"]["nonnull_field_count"] = 1
    assert not contract.evaluate_gates(mutated)["G18_board_null_and_blocked"]


def test_source_data_matches_rows() -> None:
    report = contract.build_report()
    contract.write_outputs(report)
    with contract.DEFAULT_SOURCE_DATA.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows == report["discussion_rows"]
    stored = json.loads(contract.DEFAULT_REPORT.read_text(encoding="utf-8"))
    assert stored["analysis_sha256"] == report["analysis_sha256"]


def test_generated_report_verifies() -> None:
    contract.write_outputs(contract.build_report())
    ok, checks = contract.verify_report()
    assert ok, checks
    assert all(checks.values())
