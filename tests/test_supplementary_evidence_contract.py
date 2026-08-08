from __future__ import annotations

import copy
import csv
import json

from cnn_fpga.benchmark import supplementary_evidence_contract as contract


def test_report_passes_all_gates() -> None:
    report = contract.build_report()
    assert report["verdict"] == contract.VERDICT
    assert report["gate_summary"] == {"passed": 24, "total": 24}
    assert all(report["gates"].values())


def test_appendix_sections_are_complete_and_ordered() -> None:
    manuscript = contract.build_report()["manuscript"]
    assert manuscript["sections"] == list(contract.REQUIRED_APPENDIX_SECTIONS)
    assert manuscript["characters"] > 30_000


def test_source_rows_cover_every_supplement_state() -> None:
    rows = contract.build_report()["supplement_rows"]
    assert len(rows) == 46
    assert len({row["row_id"] for row in rows}) == 46
    assert {row["supplement_state"] for row in rows} == set(contract.SUPPLEMENT_STATES)


def test_value_states_remain_lossless() -> None:
    parent = contract.build_report()["parent_state"]
    assert len(parent["ontology"]["signature_fields"]) == 13
    assert set(parent["ontology"]["value_states"]) == contract.REQUIRED_VALUE_STATES


def test_both_one_million_cycle_qualifications_keep_denominators() -> None:
    parent = contract.build_report()["parent_state"]
    assert parent["long_rtl"] == {
        "families": 10,
        "cycles": 1_000_000,
        "valid": 972_386,
        "commit_attempts": 61,
        "undefined": 0,
        "silent_overflow": 0,
        "cxx_mismatches": 0,
    }
    assert parent["integrated_rtl"] == {
        "families": 10,
        "cycles": 1_000_000,
        "replay": 995_802,
        "directed": 4_198,
        "host_attempts": 75,
        "rollback_attempts": 25,
        "undefined": 0,
        "silent_overflow": 0,
        "cxx_mismatches": 0,
    }


def test_preboard_profile_does_not_fill_board_fields() -> None:
    parent = contract.build_report()["parent_state"]
    assert parent["preboard"]["eligible_profiles"] == 1
    assert parent["preboard"]["equivalence_rows"] == 4_316
    assert parent["preboard"]["cycles"] == 6
    assert parent["preboard"]["ii"] == 1
    assert parent["preboard"]["pr_seeds"] == 3
    assert parent["board"] == {"fields": 42, "nonnull": 0}


def test_phase6c_correctness_and_reproduction_counts_are_exact() -> None:
    phase = contract.build_report()["parent_state"]["phase6c"]
    assert phase["single_domain"] == 1_048_576
    assert phase["single_boundary"] == 1_000_000
    assert phase["single_mismatches"] == 0
    assert phase["cnot_trials"] == 3_080_192
    assert phase["structured_upstream"] == 2_005
    assert phase["structured_cells"] == phase["structured_cpd_wins"] == 27


def test_phase6c_positive_and_absent_rows_cannot_mix() -> None:
    phase = contract.build_report()["parent_state"]["phase6c"]
    assert phase["multimode_cycles"] == 9_600_000
    assert phase["multimode_decodes"] == 38_400_000
    assert phase["multimode_adaptive"] < phase["multimode_static"]
    assert phase["learned_candidates"] == 16
    assert phase["learned_eligible"] == 0
    assert phase["external_rows"] == 18
    assert phase["external_same_task"] == 0
    assert phase["global_score"] is False
    assert phase["global_winner"] is None


def test_previous_prose_contracts_remain_live() -> None:
    parent = contract.build_report()["parent_state"]
    assert all(parent["previous_contracts_live"].values())


def test_every_targeted_mutation_is_rejected() -> None:
    audit = contract.build_report()["semantic_mutation_audit"]
    assert audit["count"] == audit["detected"] == 24
    assert all(case["rejected"] for case in audit["cases"])


def test_null_to_value_and_global_ranking_mutations_fail_closed() -> None:
    report = contract.build_report()
    board_mutation = copy.deepcopy(report)
    board_mutation["parent_state"]["board"]["nonnull"] = 1
    assert not contract.evaluate_gates(board_mutation)["G18_preboard_parent_and_board_null"]
    rank_mutation = copy.deepcopy(report)
    rank_mutation["parent_state"]["phase6c"]["global_score"] = True
    assert not contract.evaluate_gates(rank_mutation)["G20_phase6c_positive_is_lane_local"]


def test_source_data_and_stored_report_are_lossless() -> None:
    report = contract.build_report()
    contract.write_outputs(report)
    with contract.DEFAULT_SOURCE_DATA.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows == report["supplement_rows"]
    stored = json.loads(contract.DEFAULT_REPORT.read_text(encoding="utf-8"))
    assert stored["analysis_sha256"] == report["analysis_sha256"]
    ok, checks = contract.verify_report()
    assert ok, checks
