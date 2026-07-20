from __future__ import annotations

import csv
import json

from cnn_fpga.benchmark import results_evidence_contract as contract


def _report() -> dict:
    return json.loads(contract.DEFAULT_REPORT.read_text(encoding="utf-8"))


def test_live_results_report_and_all_gates_pass() -> None:
    report = _report()
    assert report["verdict"] == contract.VERDICT
    assert all(report["gates"].values())
    assert all(contract.verify_report().values())


def test_results_order_is_v4_then_v5_then_phase6c_then_extensions() -> None:
    manuscript = _report()["manuscript"]
    assert manuscript["subsections"] == list(contract.REQUIRED_SUBSECTIONS)
    assert manuscript["section_order"].index("Results") < manuscript["section_order"].index(
        "Where the current data show an advantage"
    )
    assert manuscript["characters"] > 25_000


def test_every_result_has_an_explicit_state_and_boundary() -> None:
    rows = _report()["result_rows"]
    assert len(rows) == 27
    assert {row["result_state"] for row in rows} == set(contract.RESULT_STATES)
    assert len({row["row_id"] for row in rows}) == len(rows)
    assert len({row["result_id"] for row in rows}) == len(rows)
    assert all(row["boundary"] and row["source_ids"] for row in rows)


def test_v4_positive_is_reported_with_stronger_negative_comparators() -> None:
    parent = _report()["parent_state"]
    assert parent["smooth"]["primary_ci_low"] > 0.0
    assert parent["smooth"]["holm_families"] == ["periodic_drift"]
    assert parent["smooth"]["strongest_deployable"] == "window_map"
    assert parent["smooth"]["route_a_beats_static"] is False
    assert parent["smooth"]["route_a_beats_window"] is False
    assert parent["smooth"]["gap_closure"] < 0.0


def test_tail_and_operational_cost_negatives_remain_visible() -> None:
    parent = _report()["parent_state"]
    assert parent["tail"]["broad_improvement"] is False
    assert parent["tail"]["confirmed_improvement_families"] == []
    assert len(parent["tail"]["exact_equal_families"]) == 5
    assert parent["tail"]["minimum_fallback"] > 0.59
    assert parent["tail"]["maximum_false_updates"] == 3365
    assert all("all 38" in text for text in parent["failed_policy_families"].values())


def test_v4_preboard_does_not_fill_board_measurements() -> None:
    parent = _report()["parent_state"]
    assert parent["rtl"]["cycles"] == 1_000_000
    assert parent["rtl"]["mismatches"] == 0
    assert parent["rtl"]["host_commit_attempts"] == 75
    assert parent["rtl"]["rollback_attempts"] == 25
    assert parent["rtl"]["measured_board_latency"] is False
    assert parent["hardware"]["seed_counts"] == [3, 3]
    assert parent["board"] == {"field_count": 42, "nonnull_count": 0}


def test_v5_is_diagnostic_stop_with_no_downstream_results() -> None:
    v5 = _report()["parent_state"]["v5"]
    assert v5["formal_decisions"] == 71_958_528
    assert v5["development_decisions"] == 4_571_136
    assert v5["strict_causal_headroom"] < 0.0
    assert 0.0 <= v5["incremental_action_headroom"] < 0.001
    assert v5["dropped_tasks"] == 20
    assert v5["downstream_outputs"] == 0
    assert v5["formal_manifest"] is False
    assert v5["formal_output"] is False


def test_phase6c_results_contain_only_the_five_eligible_secondary_claims() -> None:
    phase6c = _report()["parent_state"]["phase6c"]
    assert set(phase6c["eligible_result_ids"]) == contract.EXPECTED_PHASE6C_RESULTS
    assert phase6c["literature_result_ids"] == []
    assert phase6c["global_score"] is False
    assert phase6c["global_winner"] is None
    assert phase6c["cells"] == 206
    assert phase6c["gates_passed"] == 24
    assert phase6c["parent_bindings_live"] is True


def test_restart_selection_cap_hits_and_hindsight_are_not_hidden() -> None:
    selection = _report()["parent_state"]["selection"]
    assert selection["episodes"] == 6
    assert selection["active_evaluation_selection"] == 0
    assert selection["hindsight_disagreements"] == 2
    assert selection["teacher_restart_seeds"] == [601, 709, 811]
    assert selection["teacher_cap_indices"] == [0, 2]
    assert selection["teacher_failed_restarts"] == []
    assert selection["student_training_records"] == 9
    assert selection["student_cap_count"] == 6


def test_source_data_is_lossless_and_each_gate_rejects_its_mutation() -> None:
    report = _report()
    with contract.DEFAULT_SOURCE_DATA.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == report["result_rows"]
    audit = report["semantic_mutation_audit"]
    assert audit["count"] == audit["detected"] == len(report["gates"]) == 20
    assert {case["target_gate"] for case in audit["cases"]} == set(report["gates"])
    assert all(case["rejected"] for case in audit["cases"])
