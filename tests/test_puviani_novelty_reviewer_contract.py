from __future__ import annotations

import copy
import csv
import json

from cnn_fpga.benchmark import puviani_novelty_reviewer_contract as contract


def test_report_passes_all_gates_and_targeted_mutations() -> None:
    report = contract.build_report()
    assert report["verdict"] == contract.VERDICT
    assert report["gate_summary"] == {"passed": 24, "total": 24}
    assert all(report["gates"].values())
    assert report["semantic_mutation_audit"]["detected"] == 24
    assert report["semantic_mutation_audit"]["count"] == 24


def test_package_is_preemptive_and_does_not_invent_reviewer_wording() -> None:
    package = contract.build_report()["response_package"]
    assert package["package_readiness"] == "draft_with_placeholders"
    assert package["missing_information"] == ["ACTUAL_REVIEWER_ID_AND_VERBATIM_WORDING"]


def test_parent_contract_binding_ignores_timestamp_but_detects_analysis_change() -> None:
    payload = json.loads(contract.LEARNING_RESPONSE.read_text(encoding="utf-8"))
    parent = contract._contract_projection(json.dumps(payload))
    payload["generated_at_utc"] = "timestamp-only-change"
    assert contract._contract_projection(json.dumps(payload)) == parent
    payload["analysis_sha256"] = "0" * 64
    assert contract._contract_projection(json.dumps(payload)) != parent


def test_official_intake_is_not_misreported_as_exact_reproduction() -> None:
    report = contract.build_report()
    intake = report["official_source_intake"]
    exact = report["official_exact_status"]
    assert intake["commit"] == "c9ab1ef2b3ff6fa6d6d24cd95fbd06e2872e016d"
    assert intake["trained_checkpoints_present"] is False
    assert intake["paper_exact_reproduction"] is False
    assert exact["qualification"]["passed"] == 0
    assert exact["qualification"]["failed"] == 15
    assert exact["guessed_fields"] == []


def test_twenty_agent_and_all_exact_outcomes_remain_null() -> None:
    exact = contract.build_report()["official_exact_status"]
    assert exact["agent_rows"] == 20
    assert exact["agent_rows_all_null"]
    assert exact["all_exact_outcomes_null"]
    for method in ("standard", "MF", "NMF"):
        row = exact["paper_exact_outcomes"][method]
        assert all(row[field] is None for field in ("T_X", "T_Y", "T_Z", "T_ch", "F_avg"))


def test_reduced_probe_and_matched_negative_branch_are_not_promoted() -> None:
    report = contract.build_report()
    reduced = report["reduced_diagnostic"]
    matched = report["matched_comparison_status"]
    assert reduced["scope"] == "REDUCED_STANDARD_PATH_DIAGNOSTIC_NOT_PAPER_REPRODUCTION"
    assert reduced["coverage"]["rows"] == 756
    assert reduced["coverage"]["trajectories"] == 36
    assert reduced["coverage"]["environment_steps"] == 378
    assert not reduced["contains_mf_or_nmf"]
    assert matched["execution_branch"] == "INELIGIBLE_NEGATIVE_BRANCH_NO_MATCHED_RUN"
    assert matched["comparison_run_manifest"] is None
    assert matched["comparison_raw_data"] is None
    assert matched["metric_count"] == 13
    assert matched["all_metrics_null"]


def test_project_native_directional_result_keeps_counterexample() -> None:
    project = contract.build_report()["project_native_directional"]
    assert project["primary_logical_z_lifetime_cycles"] == {
        "standard": 2.7476620716328606,
        "mf": 6.534670655440108,
        "nmf": 6.740784780540096,
        "nmf_latest_only": 6.031675171829197,
    }
    assert project["nmf_minus_mf"]["ci95_low"] == 0.08416109825708099
    assert project["nmf_minus_mf"]["ci95_high"] == 0.32806715194289604
    assert project["confirmation_logical_z_lifetime_cycles"]["nmf_latest_only"] == 8.271987493616864
    assert project["confirmation_reset_counterexample"]
    assert project["used_as_official_replacement"] is False


def test_task_signatures_forbid_cross_lane_ranking() -> None:
    signatures = contract.build_report()["task_signatures"]
    assert len(signatures["axes"]) == 9
    for name in (
        "puviani_physical_controller",
        "project_multimode_decoder",
        "project_single_mode_rtl",
    ):
        assert set(signatures[name]) == set(signatures["axes"])
    assert signatures["same_task"] is False
    assert signatures["numeric_global_leaderboard_allowed"] is False


def test_current_algorithm_no_go_and_tail_cost_are_visible() -> None:
    report = contract.build_report()
    current = report["current_phase6d"]
    tail = report["tail_safety"]
    assert current["physical_rounds"] == 79_872
    assert current["baseline_p_L"] == current["proposed_p_L"] == 0.11197916666666667
    assert current["relative_improvement"] == current["relative_improvement_lcb"] == 0.0
    assert current["verdict"] == "NO_GO_MULTIMODE_CAUSAL_HEADROOM"
    assert tail["fallback_rates"]["nominal_static"] == 0.001193576388888889
    assert tail["fallback_rates"]["telegraph_drift"] == 0.5944959852430556
    assert tail["fallback_rates"]["step_calibration_shift"] == 0.9585458260995371
    assert tail["calibration"]["baseline_global_worst_error_count"] == 181
    assert tail["calibration"]["proposed_global_worst_error_count"] == 181


def test_learning_is_scoped_ablation_absent_from_current_rtl() -> None:
    learning = contract.build_report()["learning_extension"]
    assert learning["candidate_families"] == 16
    assert learning["same_task_eligible"] == 0
    assert learning["teacher_parameters"] == 72_853
    assert learning["student_state_dimension"] == 4
    assert learning["student_scalars"] == 95
    assert learning["minimum_retention_lcb"] == 0.9445014278749587
    assert learning["student_present_in_current_rtl"] is False
    assert learning["current_disposition"] == "DROPPED_ABLATION_ONLY"


def test_rtl_result_is_exact_preboard_and_not_nmf_or_measured() -> None:
    report = contract.build_report()
    rtl = report["rtl_contribution"]
    assert rtl["formal_gates"] == {"passed": 17, "total": 17}
    assert rtl["cover_witnesses"] == {"reachable": 14, "total": 14}
    assert rtl["formal_mutations"]["killed"] == rtl["formal_mutations"]["total"] == 21
    assert rtl["cycles"] == 1_000_000
    assert rtl["ii1_input_pairs"] == rtl["ii1_output_pairs"] == 998_435
    assert rtl["mismatch_count"] == rtl["undefined_actions"] == rtl["silent_overflow"] == 0
    assert rtl["latency_cycles"] == 6
    assert rtl["initiation_interval_cycles"] == 1
    assert all(value is None for value in rtl["measured_fields"].values())
    assert rtl["evidence_boundary"]["board_measured"] is False
    assert rtl["student_drives_fast_action"] is False
    assert set(report["nontransfer_contract"].values()) == {False}


def test_response_is_numeric_direct_and_forbidden_claim_mutation_is_killed() -> None:
    report = contract.build_report()
    text = report["response_package"]["english_response"]
    for token in ("0/15", "all thirteen result fields remain null", "0% relative improvement", "8.271987"):
        assert token in text
    assert not any(phrase.lower() in text.lower() for phrase in report["forbidden_response_phrases"])
    mutated = copy.deepcopy(report)
    mutated["response_package"]["english_response"] += " We surpass Puviani NMF."
    assert not contract.evaluate_gates(mutated)["G21_response_and_manuscript_are_direct_without_overclaim"]


def test_rows_are_lossless_and_written_artifacts_verify() -> None:
    report = contract.build_report()
    rows = report["response_rows"]
    assert len(rows) == len({row["row_id"] for row in rows}) == 24
    assert {row["response_state"] for row in rows} == contract.RESPONSE_STATES
    contract.write_outputs(report)
    with contract.DEFAULT_SOURCE_DATA.open(encoding="utf-8", newline="") as stream:
        assert list(csv.DictReader(stream)) == contract._source_rows(report)
    stored = json.loads(contract.DEFAULT_REPORT.read_text(encoding="utf-8"))
    assert stored["source_data"]["rows"] == 24
    ok, checks = contract.verify_report()
    assert ok, checks
    assert all(checks.values())
