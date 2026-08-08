from __future__ import annotations

from copy import deepcopy
import json

import pytest

from cnn_fpga.benchmark import static_gkp_same_model_lane as lane


@pytest.fixture(scope="module")
def report() -> dict:
    return json.loads(lane.DEFAULT_ARTIFACT.read_text(encoding="utf-8"))


def test_stored_same_model_report_is_current(report: dict) -> None:
    lane.verify_report(report)
    assert report["verdict"] == lane.VERDICT
    assert report["gate_summary"] == {"passed": 11, "failed": 0}


def test_all_methods_share_full_formal_scale_and_metrics(report: dict) -> None:
    for row in report["method_table"]:
        assert row["decisions"] == 28_311_552
        assert set(("p_L", "p_X", "p_Y", "p_Z", "p95_window_ler", "worst_window_ler", "family_ler", "cost")) <= set(row)
    oracle = next(row for row in report["method_table"] if row["method_id"] == "hidden_state_oracle")
    assert oracle["deployable"] is False


def test_topk_k4_inherits_full_static_counts_only_after_complete_domain_proof(report: dict) -> None:
    eq = report["topk_full_exhaustive_equivalence"]
    assert eq["grid_points"] == 1024 * 1024
    assert eq["hard_disagreements"] == 0
    assert eq["topk_action_sha256"] == eq["full_action_sha256"]
    assert max(eq["q_llr_max_abs_error"], eq["p_llr_max_abs_error"]) > 0.0
    rows = {row["method_id"]: row for row in report["method_table"]}
    for metric in ("p_I", "p_X", "p_Y", "p_Z", "p_L", "average_ler_equal_family_seed", "p95_window_ler", "worst_window_ler", "family_ler"):
        assert rows["topk_k4_static_map"][metric] == rows["static_joint_map"][metric]


def test_topk_operation_storage_proxy_is_smaller_but_not_measured(report: dict) -> None:
    rows = {row["method_id"]: row for row in report["method_table"]}
    topk = rows["topk_k4_static_map"]["cost"]
    full = rows["static_joint_map"]["cost"]
    assert topk["target_measured"] is False
    assert full["target_measured"] is False
    assert topk["operation_storage_proxy"]["serial_cycle_upper_proxy"] < full["operation_storage_proxy"]["serial_cycle_upper_proxy"]
    assert topk["operation_storage_proxy"]["retained_state_bits"] < full["operation_storage_proxy"]["retained_state_bits"]


def test_raw_trajectory_cluster_contrast_recomputes_and_falsifies_superiority(report: dict) -> None:
    recomputed = lane._paired_static_contrast(lane._load(lane.PARENT))
    assert recomputed == report["paired_static_contrast"]
    assert recomputed["ci95_high"] < 0.0
    assert recomputed["route_a_superiority_passes_lcb_gt_zero"] is False


def test_literature_registry_never_imports_cross_model_numbers(report: dict) -> None:
    registry = report["literature_registry"]
    assert len(registry) >= 4
    assert all(row["primary_url"].startswith("https://") for row in registry)
    assert all(row["numeric_cross_model_comparison_allowed"] is False for row in registry)


def test_negative_claims_are_machine_frozen(report: dict) -> None:
    claims = {row["claim_id"]: row["state"] for row in report["claim_registry"]}
    assert claims["STATIC_GKP_ROUTE_A_SUPERIORITY"] == "FALSIFIED"
    assert claims["TOPK_K4_HARD_ACTION_EQUIVALENCE"] == "ESTABLISHED_PREBOARD"
    assert claims["GLOBAL_GKP_SOTA"] == "PROHIBITED"
    assert claims["PHYSICAL_BREAK_EVEN"] == "PROHIBITED"


def test_mutations_and_direct_evaluator_fail_closed(report: dict) -> None:
    audit = report["semantic_mutation_audit"]
    assert audit["detected"] == audit["count"] == 8
    assert all(row["rejected"] for row in audit["cases"])
    candidate = deepcopy(report)
    candidate["topk_full_exhaustive_equivalence"]["hard_disagreements"] = 1
    assert not lane.evaluate_gates(candidate)["G05_topk_k4_is_exhaustively_hard_action_equivalent"]
