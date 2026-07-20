from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path

import pytest

from cnn_fpga.benchmark import route_a_claim_matrix as matrix


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "docs" / "t6_8_7_route_a_claim_matrix.json"


def _report() -> dict:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _claims() -> dict[str, dict]:
    return {row["claim_id"]: row for row in _report()["claims"]}


def test_current_report_recomputes_and_all_atomic_gates_pass() -> None:
    report = _report()
    matrix.verify_report(report)
    assert report["gate_summary"] == {"passed": 14, "failed": 0}
    assert report["verdict"] == "PASS_ROUTE_A_ATOMIC_CLAIM_MATRIX_WITH_RESTRICTED_POSITIVE_CLAIMS"
    assert len(report["claims"]) == 10
    assert {row["opponent_class"] for row in report["claims"]} == matrix.OPPONENT_CLASSES


def test_negative_results_cannot_be_hidden_by_the_claim_matrix() -> None:
    claims = _claims()
    static = claims["STATIC_GKP_SUPERIORITY"]
    assert static["state"] == "FALSIFIED"
    assert static["current_result"]["static_minus_route_a"]["ci95_high"] < 0.0
    assert static["current_result"]["static_average_ler"] < static["current_result"]["route_a_average_ler"]

    drift = claims["GENERAL_DRIFT_MATCHED_BUDGET_SOTA"]
    assert drift["state"] == "NOT_ESTABLISHED"
    assert drift["current_result"]["qualified_external_comparator_count"] == 0
    assert drift["current_result"]["worst_update_us"] > drift["current_result"]["cap_us"]

    gqf = claims["PUVIANI_NMF_SURPASS"]
    assert gqf["state"] == "PROHIBITED"
    assert gqf["current_result"]["paper_exact_passed"] == 0
    assert gqf["current_result"]["matched_metric_non_null_count"] == 0

    fpga = claims["FPGA_SPEED_ADVANTAGE"]
    assert fpga["state"] == "PROHIBITED"
    assert fpga["current_result"]["same_task_external_comparator_count"] == 0
    assert fpga["current_result"]["real_board_source_to_action"] == "PENDING_T6.9.2"


def test_supported_claims_are_restricted_and_have_explicit_retraction_rules() -> None:
    claims = _claims()
    positive = [row for row in claims.values() if row["state"] in matrix.POSITIVE_STATES]
    assert {row["claim_id"] for row in positive} == {
        "CONTRACT_SYSTEM_INTEGRATION",
        "SMOOTH_LOCKED_EWMA_ADVANTAGE",
        "STATIC_K4_HARD_ACTION_EQUIVALENCE",
        "GENERAL_DRIFT_BOCD_OUTCOME",
        "FPGA_DETERMINISTIC_ARCHITECTURE",
    }
    text = " ".join(row["strongest_supported_wording"].lower() for row in positive)
    assert not any(token in text for token in matrix.FORBIDDEN_RANKING_TOKENS)
    assert all(row["remaining_gaps"] and row["revocation_conditions"] for row in positive)
    assert claims["CNN_PRIMARY_ROLE"]["state"] == "ABLATION_ONLY"


def test_every_claim_has_live_report_source_config_seed_and_pending_t69_bindings() -> None:
    for claim in _report()["claims"]:
        evidence = claim["evidence"]
        assert len(evidence["config"]["threshold_lock_sha256"]) == 64
        assert evidence["seeds"]["external_drift_formal"] == list(range(202607176201, 202607176225))
        assert evidence["seeds"]["gqf_reduced_probe"] == [68401, 68407, 68419]
        assert evidence["selectors"]
        for artifact in evidence["current_artifacts"]:
            report_binding = artifact["report"]
            report_path = ROOT / report_binding["path"]
            assert report_path.stat().st_size == report_binding["bytes"]
            assert _sha256(report_path) == report_binding["sha256"]
            source = artifact["source_data"]
            if source:
                source_path = ROOT / source["path"]
                assert source_path.is_file()
                assert _sha256(source_path) == source["sha256"]
        assert [item["task_id"] for item in evidence["t6_9_dependencies"]] == ["T6.9.1", "T6.9.2", "T6.9.3"]
        assert all(item["status"].startswith("PENDING") and item["sha256"] is None for item in evidence["t6_9_dependencies"])


def test_source_csv_is_lossless_and_semantic_mutations_fail_closed() -> None:
    report = _report()
    path = ROOT / report["source_data"]["path"]
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == report["source_data"]["rows"] == 10
    assert {row["claim_id"] for row in rows} == matrix.CLAIM_IDS
    assert report["semantic_mutation_audit"]["count"] == report["semantic_mutation_audit"]["detected"] == 14
    assert all(row["rejected"] for row in report["semantic_mutation_audit"]["cases"])

    forged = deepcopy(report)
    next(row for row in forged["claims"] if row["claim_id"] == "FPGA_SPEED_ADVANTAGE")["state"] = "SUPPORTED_RESTRICTED"
    with pytest.raises(ValueError, match="gates/verdict"):
        matrix.verify_report(forged)
