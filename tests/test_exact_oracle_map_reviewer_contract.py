from __future__ import annotations

import copy
import csv
import json

import pytest

from cnn_fpga.benchmark import exact_oracle_map_reviewer_contract as contract


def test_report_passes_all_gates() -> None:
    report = contract.build_report()
    assert report["verdict"] == contract.VERDICT
    assert report["gate_summary"] == {"passed": 20, "total": 20}
    assert all(report["gates"].values())


def test_response_rows_cover_every_state_without_duplicates() -> None:
    rows = contract.build_report()["response_rows"]
    assert len(rows) == 20
    assert len({row["row_id"] for row in rows}) == 20
    assert {row["response_state"] for row in rows} == set(contract.RESPONSE_STATES)


def test_smooth_oracle_gap_is_negative_and_not_clipped() -> None:
    smooth = contract.build_report()["evidence"]["smooth"]
    assert smooth["static_ler"] < smooth["proposed_ler"]
    assert smooth["window_ler"] < smooth["proposed_ler"]
    assert smooth["oracle_ler"] < smooth["static_ler"]
    assert smooth["gap_closure"] == pytest.approx(-0.030462415077799714)
    assert max(smooth["gap_ci95"]) < 0.0


def test_oracle_is_non_deployable_nonzero_and_model_bounded() -> None:
    evidence = contract.build_report()["evidence"]
    assert evidence["smooth"]["oracle_deployable"] is False
    assert evidence["oracle_reference"]["oracle_ler"] == pytest.approx(0.025103125)
    assert "nondeployable" in evidence["oracle_reference"]["allowed"]
    assert "channel-recovery optimum" in evidence["oracle_reference"]["forbidden"]


def test_static_and_calibration_counterevidence_remain_visible() -> None:
    evidence = contract.build_report()["evidence"]
    assert evidence["claim_states"]["STATIC_GKP_SUPERIORITY"] == "FALSIFIED"
    assert evidence["calibration_shift"]["static_worst_errors_per_512"] == 32
    assert evidence["calibration_shift"]["proposed_worst_errors_per_512"] == 181


def test_truth_oracle_is_not_relabelled_as_causal_headroom() -> None:
    causal = contract.build_report()["evidence"]["causal_headroom"]
    assert causal["truth_privileged"] is True
    assert causal["selector_relative_headroom"] < 0.0
    assert causal["incremental_errors"] == 9
    assert causal["incremental_action_space_headroom"] == pytest.approx(0.0002548564308772725)
    assert causal["verdict"] == "NO_GO_V5_INSUFFICIENT_ACTION_SPACE_HEADROOM"


def test_each_targeted_mutation_is_rejected() -> None:
    audit = contract.build_report()["semantic_mutation_audit"]
    assert audit["count"] == audit["detected"] == 20
    assert all(case["rejected"] for case in audit["cases"])


def test_oracle_deployability_and_gap_sign_mutations_fail_closed() -> None:
    report = contract.build_report()
    deployable = copy.deepcopy(report)
    deployable["evidence"]["smooth"]["oracle_deployable"] = True
    assert not contract.evaluate_gates(deployable)["G08_smooth_values_and_nondeployability"]
    crossed = copy.deepcopy(report)
    crossed["evidence"]["smooth"]["gap_ci95"] = [-0.04, 0.01]
    assert not contract.evaluate_gates(crossed)["G09_gap_formula_sign_and_interval"]


def test_task_board_requires_terminal_task_and_next_pointer() -> None:
    report = contract.build_report()
    assert report["task_status"] == {"T7.3.1": "Done", "T7.3.2": "In Progress"}
    mutated = copy.deepcopy(report)
    mutated["task_status"]["T7.3.2"] = "Todo"
    assert not contract.evaluate_gates(mutated)["G19_task_board_terminal_and_next"]


def test_source_data_and_stored_report_are_lossless() -> None:
    report = contract.build_report()
    contract.write_outputs(report)
    with contract.DEFAULT_SOURCE_DATA.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows == report["response_rows"]
    stored = json.loads(contract.DEFAULT_REPORT.read_text(encoding="utf-8"))
    assert stored["analysis_sha256"] == report["analysis_sha256"]
    ok, checks = contract.verify_report()
    assert ok, checks
