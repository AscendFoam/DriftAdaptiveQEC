from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path

import pytest

from cnn_fpga.benchmark import gqf_paper_exact_reproduction as reproduction


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "docs" / "t6_8_4_gqf_paper_exact_reproduction.json"


def _report() -> dict:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def test_current_no_go_report_recomputes_with_full_integrity() -> None:
    report = _report()
    reproduction.verify_report(report)
    assert report["task_id"] == "T6.8.4"
    assert report["integrity_summary"] == {"passed": 13, "failed": 0}
    assert report["verdict"] == "COMPLETE_GQF_PAPER_EXACT_ATTEMPT_NO_GO_SOURCE_INCOMPLETE"


def test_paper_protocol_and_numeric_anchors_are_frozen() -> None:
    prereg = _report()["preregistration"]
    protocol = prereg["paper_protocol"]
    assert protocol["fock_cutoff"] == 100
    assert protocol["training_Delta"] == 0.2
    assert protocol["training_epochs"] == 1000
    assert protocol["agents"] == 20
    assert protocol["rnn_architecture"] == [10, 256, 256, 15]
    assert protocol["evaluation_full_cycles"] == 1000
    assert len(protocol["evaluation_logical_states"]) == 6
    anchors = prereg["published_numeric_anchors"]
    assert anchors["low_noise_T_Z_standard_cycles"] == 700.0
    assert anchors["low_noise_T_Z_NMF_cycles"] == 1500.0
    assert anchors["low_noise_T_minus_Y_NMF_cycles"] == 770.0
    assert anchors["status_of_complete_TX_TY_TZ_Tch_table"] == "NOT_TABULATED_NUMERICALLY_IN_SOURCE"


def test_official_source_conflicts_are_not_silently_patched_into_exact() -> None:
    report = _report()
    discrepancies = {row["id"]: row for row in report["source_discrepancies"]}
    assert len(discrepancies) == 18
    assert all(row["blocking"] for row in discrepancies.values())
    assert discrepancies["D04"]["paper"] == [10, 256, 256, 15]
    assert discrepancies["D04"]["official"] == [30, 30, 30, 15]
    assert discrepancies["D06"]["paper"] == 1000
    assert discrepancies["D06"]["official_runner"] == 101
    assert discrepancies["D10"]["official"] == "previous_reward initialized to 0 and never updated"
    assert discrepancies["D16"]["official"] == "max_steps=21 executes 21 env.step calls"
    assert report["guessed_fields"] == []


def test_reduced_probe_is_real_six_state_three_seed_twenty_one_step_data() -> None:
    report = _report()
    probe = report["reduced_probe"]
    assert probe["scope"] == "REDUCED_STANDARD_PATH_DIAGNOSTIC_NOT_PAPER_REPRODUCTION"
    assert probe["coverage"] == {
        "rows": 756,
        "expected_rows": 756,
        "trajectories": 36,
        "environment_steps": 378,
    }
    assert all(probe["checks"].values())
    assert probe["configuration"]["cutoff"] == 8
    assert probe["configuration"]["paper_ten_cycle_prefix_half_steps"] == 20
    assert probe["configuration"]["official_terminal_half_steps"] == 21
    assert probe["elapsed_s"] > 0.0

    csv_path = ROOT / "docs" / "t6_8_4_gqf_reproduction_source_data.csv"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 756
    assert len({row["state_id"] for row in rows}) == 6
    assert len({row["seed"] for row in rows}) == 3
    assert {int(row["half_cycle"]) for row in rows} == set(range(1, 22))
    assert all(0.0 <= float(row["measurement_probability"]) <= 1.0 for row in rows)
    assert max(abs(float(row["rho_trace_real"]) - 1.0) for row in rows) <= 5.0e-5


def test_all_exact_metrics_and_all_twenty_agents_remain_explicitly_missing() -> None:
    report = _report()
    exact = report["exact_qualification"]
    assert exact["passed"] == 0
    assert exact["failed"] == 15
    assert not any(exact["gates"].values())
    for strategy in ("standard", "MF", "NMF"):
        outcome = report["paper_exact_outcomes"][strategy]
        assert outcome["status"] == "NOT_RUN_EXACT_PREREQUISITE_FAIL"
        assert all(outcome[metric] is None for metric in ("T_X", "T_Y", "T_Z", "T_ch", "F_avg"))
    assert len(report["agent_ledger"]) == 20
    assert [row["agent_index"] for row in report["agent_ledger"]] == list(range(1, 21))
    assert all(row["seed"] is None and row["checkpoint_sha256"] is None for row in report["agent_ledger"])


def test_claims_and_t6_8_5_fail_closed() -> None:
    report = _report()
    assert report["t6_8_5_eligible"] is False
    assert report["claim_boundary"] == {
        "paper_exact_reproduction": "PROHIBITED",
        "directional_MF_NMF_ordering": "NOT_ESTABLISHED",
        "surpass_puviani_nmf": "PROHIBITED",
        "reduced_official_standard_path": "ESTABLISHED_DIAGNOSTIC_ONLY",
    }
    assert report["preregistration"]["scope"]["project_T2_3_7_may_substitute_official"] is False


def test_all_integrity_gates_have_targeted_mutations_and_hashes_fail_live() -> None:
    report = _report()
    audit = report["semantic_mutation_audit"]
    assert audit["count"] == audit["detected"] == 13
    assert {row["target_gate"] for row in audit["cases"]} == set(report["integrity_gates"])
    assert all(row["rejected"] for row in audit["cases"])

    forged = deepcopy(report)
    forged["bindings"]["source_csv"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="bound artifact drifted"):
        reproduction.verify_report(forged)


def test_exact_promotion_is_rejected_even_if_verdict_string_is_changed() -> None:
    forged = deepcopy(_report())
    forged["exact_qualification"]["gates"]["E15_published_ordering_passes"] = True
    forged["exact_qualification"]["passed"] = 1
    forged["exact_qualification"]["failed"] = 14
    forged["exact_reproduction_status"] = "PASS_EXACT"
    with pytest.raises(ValueError):
        reproduction.verify_report(forged)

