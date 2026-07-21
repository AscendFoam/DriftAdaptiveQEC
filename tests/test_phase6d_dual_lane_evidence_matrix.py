from __future__ import annotations

import copy
import csv
import hashlib
import json

import pytest

from cnn_fpga.benchmark import phase6d_dual_lane_evidence_matrix as subject


@pytest.fixture(scope="module")
def report() -> dict:
    return json.loads(subject.REPORT.read_text(encoding="utf-8"))


def test_frozen_config_has_three_noninterchangeable_panels_and_ten_claims() -> None:
    config = json.loads(subject.CONFIG.read_text(encoding="utf-8"))
    assert [row["panel_id"] for row in config["panels"]] == [
        "MULTIMODE_SOFTWARE", "SINGLE_MODE_RTL", "LEARNING_EXTENSION"
    ]
    assert len(config["claim_ids"]) == 10
    assert len(config["forbidden_transfer_ids"]) == 10
    assert config["global_weighted_score"] == "PROHIBITED"
    assert config["one_lane_cannot_satisfy_another_lane_gate"] is True


def test_all_live_parent_verifiers_pass_and_verdicts_are_exact(report: dict) -> None:
    checks = subject._parent_verification()
    assert set(checks) == set(report["parent_verification"])
    assert all(report["parent_verification"].values())
    config = json.loads(subject.CONFIG.read_text(encoding="utf-8"))
    assert report["parent_verdicts"] == config["required_parent_verdicts"]


def test_board_snapshot_preserves_dropped_and_active_or_done_states(report: dict) -> None:
    snapshot = subject._board_snapshot()
    assert report["board_snapshot"] == snapshot
    assert snapshot["statuses"]["T6.24.5"] == "Dropped"
    assert snapshot["statuses"]["T6.26.1"] == "Dropped"
    assert snapshot["statuses"]["T6.26.2"] == "Dropped"
    assert snapshot["statuses"]["T6.26.3"] == "ACTIVE_OR_DONE"


def test_opened_multimode_positive_and_primary_no_go_are_both_present(report: dict) -> None:
    claims = {row["claim_id"]: row for row in report["claims"]}
    opened = claims["MM_OPENED_TASK_LOCAL_GAIN"]
    no_go = claims["MM_V1_CAUSAL_HEADROOM_NO_GO"]
    blocked = claims["MM_FROZEN_BENCHMARK_SOTA_BLOCKED"]
    assert opened["state"] == "RESULTS_ONLY_NONRANKING"
    assert opened["current_result"]["candidate_p_L"] < opened["current_result"]["static_euclidean_p_L"]
    assert no_go["state"] == "MANDATORY_NEGATIVE"
    assert no_go["current_result"]["strongest_baseline"] == "static_mixture_exact_mld"
    assert no_go["current_result"]["baseline_p_L"] == no_go["current_result"]["proposed_p_L"]
    assert no_go["current_result"]["relative_improvement_point"] == 0.0
    assert no_go["current_result"]["relative_improvement_lcb"] == 0.0
    assert blocked["state"] == "BLOCKED_NOT_RUN"
    assert blocked["current_result"] is None


def test_multimode_figure_separates_ler_tail_compute_and_evidence_state(report: dict) -> None:
    elements = {
        row["element_id"]: row for row in report["figure_contract"]["elements"]
        if row["panel_id"] == "MULTIMODE_SOFTWARE"
    }
    assert {row["metric_namespace"] for row in elements.values()} == {
        "LER", "TAIL", "COMPUTE", "EVIDENCE_STATE"
    }
    assert set(elements["MM-E2"]["value"]) == {
        "candidate_worst_window_ler", "candidate_cvar95_window_ler"
    }
    assert set(elements["MM-E3"]["value"]) == {
        "candidate_runtime_seconds", "candidate_seconds_per_decode",
        "candidate_allocated_bytes_first_decode_max",
    }
    assert elements["MM-E5"]["value"] is None


def test_rtl_claim_is_same_top_property_longrun_and_three_seed_postroute(report: dict) -> None:
    summary = report["parent_summaries"]
    assert summary["rtl_formal"]["gates"] == {"passed": 17, "total": 17}
    assert summary["rtl_formal"]["mutations"] == {"killed": 21, "total": 21, "minimum": 18}
    assert summary["rtl_long"]["cycles"] == 1_000_000
    assert summary["rtl_long"]["mismatches"] == 0
    assert summary["rtl_long"]["ii1_input_pairs"] == summary["rtl_long"]["ii1_output_pairs"]
    assert summary["rtl_hardware"]["cycles"] == 6
    assert summary["rtl_hardware"]["ii"] == 1
    assert summary["rtl_hardware"]["seeds"] == [1, 7, 19]
    assert summary["rtl_hardware"]["all_timing_pass"] is True


def test_postroute_contract_preserves_whole_harness_critical_path_caveat(report: dict) -> None:
    claim = next(row for row in report["claims"] if row["claim_id"] == "RTL_POST_ROUTE_ESTIMATE")
    result = claim["current_result"]
    assert result["fmax_mhz"]["minimum"] == pytest.approx(36.79446792602539)
    assert result["fmax_mhz"]["minimum"] > 27.0
    assert all(row["wrapper_may_dominate"] for row in result["critical_paths"])
    assert all(row["end_component"] == "observability_fold" for row in result["critical_paths"])
    elements = {row["element_id"]: row for row in report["figure_contract"]["elements"]}
    assert elements["RTL-E4"]["metric_namespace"] == "POST_ROUTE"
    assert "resource_summary" not in elements["RTL-E4"]["value"]
    assert elements["RTL-E5"]["metric_namespace"] == "RESOURCE"


def test_board_and_speed_claims_remain_null_or_prohibited(report: dict) -> None:
    claims = {row["claim_id"]: row for row in report["claims"]}
    board = claims["RTL_BOARD_MEASUREMENT_BLOCKED"]
    speed = claims["RTL_SPEED_ADVANTAGE_PROHIBITED"]
    assert board["state"] == "BLOCKED_NOT_RUN"
    assert all(value is None for value in board["current_result"].values())
    assert speed["state"] == "PROHIBITED_POSITIVE"
    assert speed["current_result"] == {"fastest_or_sota": False}
    assert report["evidence_boundary"]["board_measured"] is False
    assert report["evidence_boundary"]["fastest_or_sota_hardware"] is False


def test_learning_is_status_only_absent_and_not_a_primary_lane(report: dict) -> None:
    claim = next(row for row in report["claims"] if row["claim_id"] == "LEARNING_APPROXIMATION_DROPPED")
    assert claim["state"] == "DROPPED_ABSENT"
    assert claim["current_result"] == {
        "T6.26.1": "Dropped", "T6.26.2": "Dropped", "present_in_primary_rtl": False
    }
    element = next(row for row in report["figure_contract"]["elements"] if row["element_id"] == "ML-E1")
    assert element["metric_namespace"] == "STATUS"
    assert report["lane_outcomes"]["LEARNED_APPROXIMATION_EXTENSION"] == "DROPPED_ABSENT"


def test_no_cross_lane_arrow_score_or_gate_substitution(report: dict) -> None:
    figure = report["figure_contract"]
    assert figure["global_weighted_score"] is None
    assert figure["ranking_policy"] == "NO_CROSS_LANE_RANKING_OR_GATE_SUBSTITUTION"
    assert all(edge["source_lane"] == edge["target_lane"] for edge in figure["edges"])
    assert len(report["forbidden_transfers"]) == 10
    assert all(row["disposition"] == "REJECT" for row in report["forbidden_transfers"])


def test_every_claim_and_figure_element_has_live_report_raw_config_code_source(report: dict) -> None:
    artifacts = report["artifact_registry"]
    assert all(subject._live(binding) for binding in artifacts.values())
    for row in [*report["claims"], *report["figure_contract"]["elements"]]:
        evidence = row["evidence"]
        assert set(evidence) == {*subject.EVIDENCE_CATEGORIES, "selectors"}
        assert evidence["selectors"]
        for category in subject.EVIDENCE_CATEGORIES:
            assert evidence[category]
            assert all(key in artifacts for key in evidence[category])
    for element in report["figure_contract"]["elements"]:
        assert element["allowed_wording"]
        assert element["forbidden_interpretation"]


def test_source_data_losslessly_reconstructs_all_contract_records(report: dict) -> None:
    with subject.SOURCE_DATA.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows == subject._source_rows(report)
    assert report["source_data"]["rows"] == len(rows) == 67
    assert sum(row["section"] == "claim" for row in rows) == 10
    assert sum(row["section"] == "figure_element" for row in rows) == 12
    assert sum(row["section"] == "forbidden_transfer" for row in rows) == 10
    assert sum(row["section"] == "artifact" for row in rows) == len(report["artifact_registry"])
    for row in rows:
        assert hashlib.sha256(row["payload_json"].encode()).hexdigest() == row["payload_sha256"]


def test_all_twenty_one_mutations_are_independently_recomputed(report: dict) -> None:
    audit = subject.semantic_mutation_audit(report)
    assert report["semantic_mutations"] == {"detected": 21, "total": 21}
    assert audit["detected"] == audit["total"] == 21
    assert report["semantic_mutation_results"] == audit["mutations"]
    assert all(row["rejected"] for row in audit["mutations"])


def test_validator_rejects_cross_lane_promotion_and_board_forgery(report: dict) -> None:
    candidate = copy.deepcopy(report)
    candidate["figure_contract"]["edges"][0]["target_lane"] = "SINGLE_MODE_DETERMINISTIC_RTL"
    with pytest.raises(subject.IntegrityError):
        subject._validate(candidate, check_live_files=False)
    candidate = copy.deepcopy(report)
    candidate["parent_summaries"]["rtl_hardware"]["measured_fields"]["board_power_mw"] = 1.0
    with pytest.raises(subject.IntegrityError):
        subject._validate(candidate, check_live_files=False)


def test_live_report_verification_is_fail_closed() -> None:
    verified = subject.verify()
    assert verified["verdict"] == subject.VERDICT
    assert verified["gates"] == {"passed": 21, "total": 21}
    assert verified["mutations"] == {"detected": 21, "total": 21}
