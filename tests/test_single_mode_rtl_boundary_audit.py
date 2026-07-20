from __future__ import annotations

import copy
import json

import pytest

from cnn_fpga.benchmark import single_mode_rtl_boundary_audit as subject


@pytest.fixture(scope="module")
def report():
    return subject.build_report()


def test_boundary_verdict_and_all_gates_pass(report):
    assert report["verdict"] == "PASS_BOUNDARY_FROZEN_CONVERGED_PRODUCTION_TOP_REQUIRED"
    assert report["gate_summary"]["passed"] == report["gate_summary"]["total"] == 15


def test_no_current_top_falsely_combines_all_capabilities(report):
    assert len(report["module_audit"]) == 5
    assert not any(row["is_converged_production_top"] for row in report["module_audit"])
    assert report["convergence_gap"]["present"] is True
    assert report["convergence_gap"]["next_task"] == "T6.25.2"


def test_production_and_policy_tops_are_not_conflated(report):
    modules = {row["module"]: row for row in report["module_audit"]}
    production = modules["gkp_fast_path_production_top"]
    integrated = modules["route_a_integrated_qualification_top"]
    assert "complete_image_crc32" in production["capabilities"]
    assert "regime_policy_overlay" not in production["capabilities"]
    assert "regime_policy_overlay" in integrated["capabilities"]
    assert "complete_image_crc32" not in integrated["capabilities"]
    assert "no_raw_trust_or_cfg_bypass" in integrated["missing_converged_capabilities"]


def test_module_graph_and_required_source_tokens_are_live(report):
    assert all(row["module_declaration_count"] == 1 for row in report["module_audit"])
    assert all(row["children_present"] for row in report["module_audit"])
    assert all(row["passed"] for row in report["required_token_audit"])
    assert all(report["structural_findings"].values())


def test_multimode_decoder_is_absent_from_rtl_graph(report):
    scope = report["transitive_rtl_scope"]
    assert scope["contains_multimode_graph_or_exact_mld"] is False
    assert scope["forbidden_multimode_module_hits"] == []
    assert scope["forbidden_multimode_source_hits"] == []


def test_live_parent_hashes_and_legacy_binding_gap_are_both_preserved(report):
    parents = report["parent_evidence"]
    assert parents["T6.7.3"]["direct_sources_live"] is True
    assert parents["T6.9.1"]["direct_sources_live"] is True
    assert parents["T6.19.1"]["direct_sources_live"] is True
    assert parents["T6.20.2"]["direct_sources_live"] is True
    assert parents["T6.2.1"]["direct_source_bindings"] == 0
    assert parents["T6.2.2"]["direct_source_bindings"] == 0
    assert parents["T6.2.2"]["trace_live"] is True


def test_old_long_run_and_pr_are_regression_only(report):
    decisions = {row["task_id"]: row["decision"] for row in report["reuse_decisions"]}
    assert decisions["T6.2.2"] == "CORE_LONG_RUN_REGRESSION_ONLY"
    assert decisions["T6.7.3"] == "POLICY_CORE_LONG_RUN_REGRESSION_ONLY"
    assert decisions["T6.9.1"] == "OLD_HARNESS_PR_REFERENCE_ONLY"


def test_only_four_narrow_contract_bridges_are_allowed(report):
    bridge_ids = {row["bridge_id"] for row in report["allowed_contract_bridges"]}
    assert bridge_ids == {
        "BRIDGE-CANDIDATE-IMAGE",
        "BRIDGE-ATOMIC-ACTIVE-VIEW",
        "BRIDGE-REGIME-COMMAND",
        "BRIDGE-EVENT-ACTION",
    }


def test_board_fastest_and_multimode_deployment_claims_remain_closed(report):
    boundary = report["claim_boundary"]
    assert boundary["board_measurement"] is None
    assert boundary["fastest_or_speed_advantage"] is False
    assert boundary["multimode_decoder_deployed_in_rtl"] is False


def test_semantic_mutations_are_all_caught(report):
    assert len(report["semantic_mutation_audit"]) >= 20
    assert all(row["caught"] for row in report["semantic_mutation_audit"])


def test_tampered_report_fails_closed(report):
    candidate = copy.deepcopy(report)
    candidate["reuse_decisions"][1]["decision"] = "FULL_REUSE"
    with pytest.raises(subject.IntegrityError):
        subject._validate_report(candidate, check_files=False)


def test_stored_report_rebuilds_from_live_sources():
    result = subject.verify()
    assert result["verdict"] == "PASS_BOUNDARY_FROZEN_CONVERGED_PRODUCTION_TOP_REQUIRED"
    assert result["gates"]["passed"] == result["gates"]["total"]


def test_source_data_is_lossless_enough_for_key_rows(report):
    rows = subject._source_rows(report)
    assert sum(row["section"] == "capability" for row in rows) == 55
    assert sum(row["section"] == "module" for row in rows) == 5
    assert sum(row["section"] == "reuse" for row in rows) == 5
    assert sum(row["section"] == "mutation" for row in rows) == len(report["semantic_mutation_audit"])
