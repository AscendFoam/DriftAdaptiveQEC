from __future__ import annotations

import csv
import json

from cnn_fpga.benchmark import methods_evidence_contract as contract


def _report() -> dict:
    return json.loads(contract.DEFAULT_REPORT.read_text(encoding="utf-8"))


def test_live_methods_report_and_all_gates_pass() -> None:
    report = _report()
    assert report["verdict"] == contract.VERDICT
    assert all(report["gates"].values())
    assert all(contract.verify_report().values())


def test_methods_section_has_complete_hierarchy_before_results() -> None:
    manuscript = _report()["manuscript"]
    assert manuscript["section_order"].index("Contract-centric dual-loop method") < manuscript["section_order"].index("Results")
    assert all(title in manuscript["subsections"] for title in contract.REQUIRED_SUBSECTIONS)
    assert all(title in manuscript["subsubsections"] for title in contract.REQUIRED_SUBSUBSECTIONS)
    assert manuscript["characters"] > 14_000


def test_every_component_has_one_of_four_explicit_evidence_states() -> None:
    rows = _report()["method_rows"]
    assert len(rows) == 18
    assert {row["evidence_state"] for row in rows} == set(contract.METHOD_STATES)
    assert len({row["row_id"] for row in rows}) == len(rows)
    assert len({row["component_id"] for row in rows}) == len(rows)


def test_online_observed_and_offline_truth_roles_are_not_conflated() -> None:
    by_id = {row["component_id"]: row for row in _report()["method_rows"]}
    assert by_id["observation_truth_split"]["online_privilege"] == "ONLINE_OBSERVED_ONLY"
    assert by_id["hidden_oracle"]["online_privilege"] == "NONE"
    assert by_id["hidden_oracle"]["offline_truth_role"] == "OFFLINE_TRUTH_ONLY_SCORING"
    assert by_id["physical_board"]["online_privilege"] == "PHYSICAL_INPUT_PENDING"
    assert all("TRUTH" not in row["online_privilege"] for row in by_id.values())


def test_v4_implemented_components_are_distinct_from_stopped_v5() -> None:
    by_id = {row["component_id"]: row for row in _report()["method_rows"]}
    for component_id in ("v4_hmm", "v4_window_ewma", "v4_typed_bank", "v4_integer_cxxrtl", "v4_post_route"):
        assert by_id[component_id]["evidence_state"] == "IMPLEMENTED_EVALUATED"
    assert by_id["v5_headroom"]["evidence_state"] == "DIAGNOSTIC_ONLY_EXECUTED"
    for component_id in ("v5_four_split", "v5_posterior", "v5_map_risk", "v5_typed_policy", "v5_qualification"):
        assert by_id[component_id]["evidence_state"] == "CONDITIONALLY_REGISTERED_STOPPED"
        assert by_id[component_id]["online_privilege"] == "NONE_NOT_RUN"


def test_parent_state_preserves_early_stop_and_board_null() -> None:
    parent = _report()["parent_state"]
    assert parent["v5_dropped_task_count"] == 20
    assert parent["v5_downstream_output_count"] == 0
    assert parent["v5_formal_manifest_exists"] is False
    assert parent["v5_formal_output_exists"] is False
    assert parent["strict_causal_router_headroom"] < 0.0
    assert parent["incremental_action_space_headroom"] < 0.001
    assert parent["board_measured_field_count"] == 42
    assert parent["board_measured_nonnull_count"] == 0


def test_source_data_is_lossless_and_traceable() -> None:
    report = _report()
    with contract.DEFAULT_SOURCE_DATA.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == report["method_rows"]
    assert all(row["source_ids"] and row["boundary"] for row in rows)


def test_each_gate_detects_its_targeted_semantic_mutation() -> None:
    report = _report()
    audit = report["semantic_mutation_audit"]
    assert audit["count"] == audit["detected"] == len(report["gates"]) == 18
    assert {case["target_gate"] for case in audit["cases"]} == set(report["gates"])
    assert all(case["rejected"] for case in audit["cases"])


def test_methods_contains_no_assertive_v5_or_board_promotion() -> None:
    manuscript = _report()["manuscript"]
    assert manuscript["prohibited_hits"] == []
    assert manuscript["checks"]["no_assertive_overclaim"] is True
