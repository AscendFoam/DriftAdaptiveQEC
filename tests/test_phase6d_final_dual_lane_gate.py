from __future__ import annotations

import copy
import csv
import hashlib
import json

import pytest

from cnn_fpga.benchmark import phase6d_final_dual_lane_gate as subject


@pytest.fixture(scope="module")
def report() -> dict:
    return json.loads(subject.REPORT.read_text(encoding="utf-8"))


def test_frozen_truth_table_has_all_four_boolean_outcomes() -> None:
    config = json.loads(subject.CONFIG.read_text(encoding="utf-8"))
    assert config["verdict_truth_table"] == {
        "multimode=true,rtl=true": "GO_TWO_LANE",
        "multimode=true,rtl=false": "GO_MULTIMODE_ONLY",
        "multimode=false,rtl=true": "GO_RTL_ONLY",
        "multimode=false,rtl=false": "NO_GO",
    }
    assert config["expected_current_verdict"] == "GO_RTL_ONLY"
    assert config["global_weighted_score"] == "PROHIBITED"
    assert config["learning_cannot_change_verdict"] is True


def test_all_five_direct_parent_verifiers_pass(report: dict) -> None:
    assert report["parent_verification"] == {
        "matrix": True, "headroom": True, "formal": True, "long": True, "hardware": True
    }
    assert report["matrix_anchor"]["verdict"] == "PASS_NONTRANSFERABLE_DUAL_LANE_EVIDENCE_AND_FIGURE_CONTRACT"
    assert report["matrix_anchor"]["gates"] == {"passed": 21, "total": 21}
    assert report["matrix_anchor"]["mutations"] == {"detected": 21, "total": 21}


def test_board_snapshot_directly_consumes_t6_24_5_t6_25_4_and_t6_26_2(report: dict) -> None:
    snapshot = subject._board_snapshot()
    assert report["board_snapshot"] == snapshot
    assert snapshot["statuses"]["T6.24.5"] == "Dropped"
    assert snapshot["statuses"]["T6.25.4"] == "Done"
    assert snapshot["statuses"]["T6.26.2"] == "Dropped"
    assert snapshot["statuses"]["T6.26.4"] == "ACTIVE_OR_DONE"
    assert snapshot["statuses"]["T6.9.2"] == "Blocked"
    assert snapshot["statuses"]["T7.1.5"] == "TODO_OR_ACTIVE"


def test_multimode_lane_is_no_go_from_direct_headroom_and_task_state(report: dict) -> None:
    lane = report["lane_decisions"]["MULTIMODE_SOFTWARE_ALGORITHM"]
    evidence = lane["direct_evidence"]
    assert lane["required_task"] == "T6.24.5"
    assert lane["gate_passed"] is False
    assert lane["decision"] == "NO_GO"
    assert evidence["task_status"] == "Dropped"
    assert evidence["headroom_verdict"] == "NO_GO_MULTIMODE_CAUSAL_HEADROOM"
    assert evidence["strongest_baseline"] == "static_mixture_exact_mld"
    assert evidence["baseline_p_L"] == evidence["proposed_p_L"]
    assert evidence["relative_improvement_point"] == evidence["relative_improvement_lcb"] == 0.0
    assert evidence["formal_or_pilot_accessed"] is False


def test_rtl_lane_is_go_only_from_exact_top_formal_long_and_hardware(report: dict) -> None:
    lane = report["lane_decisions"]["SINGLE_MODE_DETERMINISTIC_RTL"]
    evidence = lane["direct_evidence"]
    assert lane["required_task"] == "T6.25.4"
    assert lane["gate_passed"] is True
    assert lane["decision"] == "GO"
    assert evidence["hardware_verdict"] == "PASS_EXACT_CONVERGED_TOP_THREE_SEED_PREBOARD_HARDWARE_LANE"
    assert evidence["formal_verdict"] == subject.FORMAL_VERDICT
    assert evidence["long_verdict"] == "PASS_EXACT_CONVERGED_TOP_MILLION_CYCLE_CXXRTL_QUALIFICATION"
    assert evidence["latency_cycles"] == 6
    assert evidence["initiation_interval_cycles"] == 1
    assert evidence["minimum_fmax_mhz"] == pytest.approx(36.79446792602539)
    assert evidence["wrapper_may_dominate_all"] is True
    assert all(value is None for value in evidence["measured_fields"].values())


def test_learning_is_excluded_from_boolean_truth_table(report: dict) -> None:
    lane = report["lane_decisions"]["LEARNED_APPROXIMATION_EXTENSION"]
    assert lane["required_task"] == "T6.26.2"
    assert lane["gate_passed"] is False
    assert lane["decision"] == "DROPPED_ABLATION_ONLY"
    assert lane["direct_evidence"] == {"task_status": "Dropped", "changes_overall_verdict": False}


def test_truth_table_independently_recomputes_go_rtl_only(report: dict) -> None:
    mm = report["lane_decisions"]["MULTIMODE_SOFTWARE_ALGORITHM"]["gate_passed"]
    rtl = report["lane_decisions"]["SINGLE_MODE_DETERMINISTIC_RTL"]["gate_passed"]
    key = subject._truth_key(mm, rtl)
    assert key == report["truth_key"] == "multimode=false,rtl=true"
    assert report["verdict_truth_table"][key] == report["verdict"] == "GO_RTL_ONLY"
    assert report["global_weighted_score"] is None
    assert report["decision_policy"] == "INDEPENDENT_BOOLEAN_LANES_NO_WEIGHTED_SCORE_NO_GATE_SUBSTITUTION"


def test_all_ten_claim_dispositions_match_frozen_config(report: dict) -> None:
    config = json.loads(subject.CONFIG.read_text(encoding="utf-8"))
    actual = {row["claim_id"]: row["final_disposition"] for row in report["final_claims"]}
    assert actual == config["claim_dispositions"]
    assert actual["MM_OPENED_TASK_LOCAL_GAIN"] == "RETAIN_CONTEXT_ONLY"
    assert actual["MM_V1_CAUSAL_HEADROOM_NO_GO"] == "MANDATORY_NEGATIVE"
    assert actual["MM_FROZEN_BENCHMARK_SOTA_BLOCKED"] == "BLOCKED"
    assert actual["RTL_SPEED_ADVANTAGE_PROHIBITED"] == "PROHIBITED_POSITIVE"
    assert actual["LEARNING_APPROXIMATION_DROPPED"] == "DROPPED_ABLATION_ONLY"


def test_every_claim_has_gap_revocation_placement_and_parent_payload_anchor(report: dict) -> None:
    anchor = report["matrix_anchor"]["claim_payload_sha256"]
    assert len(report["final_claims"]) == 10
    for row in report["final_claims"]:
        assert "current_evidence" in row
        assert row["blocking_gaps"]
        assert row["revocation_conditions"]
        assert row["paper_placements"]
        assert row["final_wording"]
        assert row["forbidden_wording"]
        assert row["parent_evidence_keys"]
        assert row["parent_payload_sha256"] == anchor[row["claim_id"]]


def test_board_speed_multimode_deployment_and_learning_boundaries_remain_closed(report: dict) -> None:
    assert report["publication_boundary"] == {
        "phase6d_verdict": "GO_RTL_ONLY",
        "multimode_frozen_benchmark_sota": False,
        "multimode_opened_context_only": True,
        "single_mode_preboard_deterministic_atomic_fail_closed": True,
        "board_measured": False,
        "hardware_fastest_or_sota": False,
        "multimode_decoder_in_rtl": False,
        "learning_primary": False,
    }


def test_phase7_handoff_preserves_snapshot_and_requires_delta(report: dict) -> None:
    config = json.loads(subject.CONFIG.read_text(encoding="utf-8"))
    handoff = report["phase7_handoff"]
    assert handoff["next_task"] == "T7.1.5"
    assert handoff["tasks"] == config["phase7_handoff"]
    assert handoff["historical_snapshot_policy"] == "PRESERVE_T7_1_1_TO_T7_1_4_AND_ADD_DELTA"
    assert handoff["old_bundle_publishable_without_delta"] is False


def test_all_thirty_one_artifacts_are_live_hash_bound(report: dict) -> None:
    assert len(report["artifact_registry"]) == 31
    assert all(subject._live(binding) for binding in report["artifact_registry"].values())


def test_source_data_losslessly_reconstructs_decisions_claims_handoff_truth_and_artifacts(report: dict) -> None:
    with subject.SOURCE_DATA.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows == subject._source_rows(report)
    assert report["source_data"]["rows"] == len(rows) == 54
    assert sum(row["section"] == "lane_decision" for row in rows) == 3
    assert sum(row["section"] == "final_claim" for row in rows) == 10
    assert sum(row["section"] == "phase7_handoff" for row in rows) == 6
    assert sum(row["section"] == "truth_table" for row in rows) == 4
    assert sum(row["section"] == "artifact" for row in rows) == 31
    for row in rows:
        assert hashlib.sha256(row["payload_json"].encode()).hexdigest() == row["payload_sha256"]


def test_all_twenty_two_mutations_are_independently_recomputed(report: dict) -> None:
    audit = subject.semantic_mutation_audit(report)
    assert report["semantic_mutations"] == {"detected": 22, "total": 22}
    assert audit["detected"] == audit["total"] == 22
    assert report["semantic_mutation_results"] == audit["mutations"]
    assert all(row["rejected"] for row in audit["mutations"])


def test_validator_rejects_lane_rescue_weighted_score_and_snapshot_bypass(report: dict) -> None:
    candidate = copy.deepcopy(report)
    candidate["lane_decisions"]["MULTIMODE_SOFTWARE_ALGORITHM"]["gate_passed"] = True
    with pytest.raises(subject.IntegrityError):
        subject._validate(candidate, check_live_files=False)
    candidate = copy.deepcopy(report)
    candidate["global_weighted_score"] = 1.0
    with pytest.raises(subject.IntegrityError):
        subject._validate(candidate, check_live_files=False)
    candidate = copy.deepcopy(report)
    candidate["phase7_handoff"]["old_bundle_publishable_without_delta"] = True
    with pytest.raises(subject.IntegrityError):
        subject._validate(candidate, check_live_files=False)


def test_live_final_report_verification_is_fail_closed() -> None:
    verified = subject.verify()
    assert verified["verdict"] == "GO_RTL_ONLY"
    assert verified["gates"] == {"passed": 21, "total": 21}
    assert verified["mutations"] == {"detected": 22, "total": 22}
