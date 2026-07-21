from __future__ import annotations

import copy
import json

import pytest

from cnn_fpga.benchmark import converged_rtl_formal as subject


@pytest.fixture(scope="module")
def report():
    return json.loads(subject.REPORT.read_text(encoding="utf-8"))


def test_verdict_and_all_seventeen_gates_pass(report):
    assert report["verdict"] == "PASS_CONVERGED_PRODUCTION_TOP_PROPERTY_COVER_MUTATION_CLOSED"
    assert report["gate_summary"] == {"passed": 17, "total": 17}


def test_unique_top_contains_all_four_children_and_no_raw_bypass(report):
    structural = report["structural_audit"]
    assert structural["module_declaration_count"] == 1
    assert structural["all_required_children_exactly_once"] is True
    assert structural["child_instantiation_counts"] == {
        "route_a_commit_admission": 1,
        "gkp_parameter_bank_manager": 1,
        "gkp_fast_path_core": 1,
        "route_a_policy_overlay": 1,
    }
    assert not any(structural["forbidden_external_port_hits"].values())
    assert structural["manager_is_sole_core_cfg_and_trust_driver"] is True


def test_image_generation_is_not_conflated_with_activation_epoch(report):
    assert report["structural_audit"]["image_and_activation_versions_separated"] is True
    assert report["structural_audit"]["production_words_per_phase"] == 257
    assert report["structural_audit"]["formal_words_per_phase"] == 2
    assert "reachability only" in report["structural_audit"]["formal_depth_reduction_scope"]


def test_all_state_and_reachable_transition_scopes_are_not_conflated(report):
    scope = report["proof_scope"]
    assert set(scope["all_state_present_state_guards"]) == {
        "prop_all_state_management_guards",
        "prop_all_state_admission_guards",
        "prop_all_state_policy_output_guards",
    }
    assert scope["reset_reachable_transition_bound_cycles"] == 20
    assert scope["actual_core_fail_closed_bound_cycles"] == 10
    assert scope["actual_core_atomic_commit_arbitrary_state_steps"] == 3
    assert scope["compositional_unbounded_safety_closed"] is True
    assert scope["monolithic_induction_attempt_promoted"] is False
    assert "unbounded liveness" in scope["induction_boundary"]
    assert report["formal_results"]["inductive_invariants"]["returncode"] == 0
    assert report["formal_results"]["all_state_transitions"]["returncode"] == 0


def test_every_required_reachable_witness_exists(report):
    expected = {row["name"] for row in subject._load(subject.CONFIG)["reachable_covers"]}
    results = report["formal_results"]
    assert report["cover_summary"] == {"reachable": 14, "total": 14}
    assert all(results[name]["returncode"] == 0 and results[name]["model_found"] for name in expected)
    assert results["actual_core_fail_closed_cover"]["model_found"] is True


def test_actual_core_deadline_and_age_fail_closed_proof_is_clean(report):
    result = report["formal_results"]["actual_core_fail_closed"]
    assert result["returncode"] == 0
    assert result["proof_failed"] is False
    assert "gkp_fast_path_fail_closed_formal" in result["command"]
    assert "memory_map" in result["command"]


def test_actual_core_atomic_commit_refines_the_abstract_contract(report):
    result = report["formal_results"]["actual_core_atomic_commit"]
    assert result["returncode"] == 0
    assert result["proof_failed"] is False
    assert "gkp_fast_path_atomic_commit_formal" in result["command"]
    assert "-set reset_n 1" in result["command"]


def test_all_twenty_one_mutants_are_killed_by_formal_counterexamples(report):
    mutations = report["mutation_results"]
    assert report["mutation_summary"] == {"killed": 21, "total": 21, "minimum": 18}
    assert len({row["mutation"] for row in mutations}) == 21
    assert "core_accepts_wrong_activation_version" in {row["mutation"] for row in mutations}
    assert all(row["killed"] for row in mutations)
    assert all(row["kill_mechanism"] == "independent_formal_counterexample" for row in mutations)
    assert all(row["tool_result"]["proof_failed"] for row in mutations)
    assert all(not row["tool_result"]["error"] or row["tool_result"]["proof_failed"] for row in mutations)


def test_formal_discovered_duplicate_present_bug_remains_documented(report):
    correction = report["implementation_correction"]
    assert correction["found_by_formal"] is True
    assert "re-presented" in correction["problem"]
    assert "plus-one" in correction["correction"]
    assert correction["regression_property"] == "prop_all_state_management_guards"


def test_board_fastest_and_multimode_claims_remain_closed(report):
    boundary = report["claim_boundary"]
    assert boundary["board_measurement"] is None
    assert boundary["fastest_or_speed_advantage"] is False
    assert boundary["multimode_decoder_deployed_in_rtl"] is False
    assert "unbounded liveness" in boundary["not_allowed_now"]


def test_tampered_mutation_or_induction_promotion_fails_closed(report):
    candidate = copy.deepcopy(report)
    candidate["mutation_results"][0]["killed"] = False
    with pytest.raises(subject.IntegrityError):
        subject._validate_report(candidate, check_files=False)
    candidate = copy.deepcopy(report)
    candidate["proof_scope"]["monolithic_induction_attempt_promoted"] = True
    with pytest.raises(subject.IntegrityError):
        subject._validate_report(candidate, check_files=False)


def test_source_data_preserves_formal_mutation_gate_and_binding_rows(report):
    rows = subject._source_rows(report)
    assert sum(row["section"] == "formal" for row in rows) == len(report["formal_results"])
    assert sum(row["section"] == "mutation" for row in rows) == 21
    assert sum(row["section"] == "gate" for row in rows) == 17
    assert sum(row["section"] == "binding" for row in rows) == len(report["bindings"])


def test_stored_report_hash_and_live_sources_verify():
    result = subject.verify()
    assert result["verdict"] == "PASS_CONVERGED_PRODUCTION_TOP_PROPERTY_COVER_MUTATION_CLOSED"
    assert result["gates"] == {"passed": 17, "total": 17}
