from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path

import pytest

from cnn_fpga.benchmark import secondary_method_source_audit as audit


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "docs" / "t6_16_1_secondary_method_source_audit.json"


def _report() -> dict:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def _methods(report: dict) -> dict[str, dict]:
    return {row["method_id"]: row for row in report["methods"]}


def _metrics(method: dict) -> dict[str, dict]:
    return {row["metric_id"]: row for row in method["metrics"]}


def test_report_recomputes_all_gates_and_has_no_global_rank() -> None:
    report = _report()
    audit.verify_report(report)
    assert report["verdict"] == "PASS_SECONDARY_METHOD_SOURCE_AUDIT_NO_GLOBAL_RANKING"
    assert report["gate_summary"] == {"passed": 15, "failed": []}
    assert report["comparison_policy"]["global_leaderboard"] == "PROHIBITED"
    assert report["comparison_policy"]["cross_lane_score"] == "PROHIBITED"
    assert report["comparison_policy"]["same_task_external_fpga_comparator_count"] == 0


def test_source_and_method_schema_covers_all_distinct_decision_objects() -> None:
    report = _report()
    assert len(report["sources"]) == 11
    assert len(report["methods"]) == 12
    assert {row["lane_id"] for row in report["methods"]} == audit.LANES
    assert {row["screenshot_category"] for row in report["methods"]} == audit.REQUIRED_SCREENSHOT_CATEGORIES
    assert {row["normalized_category"] for row in report["methods"]} == audit.REQUIRED_NORMALIZED_CATEGORIES
    methods = _methods(report)
    assert methods["wang_direct_nn_surface_gkp"]["normalized_category"] == "direct_neural_decoder"
    assert methods["sivak2023_rl_controller"]["normalized_category"] == "offline_experiment_in_the_loop_rl_controller"
    assert methods["puviani_nmf_controller"]["normalized_category"] == "model_based_feedback_grape_recurrent_controller"


def test_numerical_claims_keep_denominator_and_source_locator() -> None:
    report = _report()
    for method in report["methods"]:
        for metric in method["metrics"]:
            if metric["value"] is not None:
                assert metric["unit"]
                assert metric["direction"]
                assert metric["denominator"]
                assert metric["statistic"]
                assert metric["source_locator"]
                assert metric["evidence_grade"] in audit.EVIDENCE_GRADES
            if metric["ranking_eligible"]:
                assert metric["value"] is not None


def test_9p9db_and_cnot_reductions_are_scoped_and_recomputed() -> None:
    report = _report()
    methods = _methods(report)
    threshold = _metrics(methods["noh_ml_cnot_and_outer"])["surface_gkp_threshold_db"]
    assert threshold["value"] == 9.9
    assert threshold["ranking_eligible"] is False
    assert "outer-code" in threshold["ineligibility_reason"]
    reductions = report["derived_evidence"]["noh_cnot_failure_reduction_by_squeezing"]
    assert reductions["9_dB"] == pytest.approx(0.3178217821782179)
    assert reductions["12_dB"] == pytest.approx(0.5845799769850403)
    assert reductions["13_dB"] == pytest.approx(0.6719230769230769)
    assert report["derived_evidence"]["noh_about_50_percent_is_not_universal"] is True
    claim = next(row for row in report["claim_audit"] if row["claim_id"] == "C05")
    assert claim["normalized_value"] is None


def test_aqec_is_physical_protocol_and_latency_is_na_not_zero() -> None:
    report = _report()
    method = _methods(report)["lachance_aqec_experiment"]
    metrics = _metrics(method)
    assert metrics["method_a_gain"]["value"] == 1.14
    assert metrics["method_a_gain"]["uncertainty"] == 0.18
    assert metrics["method_b_gain"]["value"] == 1.14
    assert metrics["method_b_gain"]["uncertainty"] == 0.16
    assert method["latency"]["core_ns"] is None
    assert method["latency"]["source_to_action_ns"] is None
    assert method["latency"]["closed_loop_ns"] is None
    assert method["latency"]["statistic"] == "N/A, not zero"
    claim = next(row for row in report["claim_audit"] if row["claim_id"] == "C15")
    assert claim["normalized_value"] is None
    assert claim["verdict"] == "N_A_NOT_ZERO"


def test_nn_latency_boundaries_are_concrete_not_a_class_range() -> None:
    report = _report()
    methods = _methods(report)
    assert methods["overwater_fpga_nn_d5"]["latency"]["core_ns"] == 87.6
    assert methods["overwater_fpga_nn_d5"]["latency"]["closed_loop_ns"] is None
    yang = methods["yang_fpga_nn_d3"]["latency"]
    assert yang["core_ns"] == 124.0
    assert yang["source_to_action_ns"] == 550.0
    assert yang["closed_loop_ns"] == 550.0
    claim = next(row for row in report["claim_audit"] if row["claim_id"] == "C12")
    assert claim["normalized_value"] is None
    assert claim["verdict"] == "NULL_FALSE_CLASS_RANGE"


def test_cpd_units_and_project_negative_boundaries_are_preserved() -> None:
    report = _report()
    methods = _methods(report)
    cpd = _metrics(methods["lin_structured_cpd"])
    assert cpd["surface_gkp_cpd_threshold"]["value"] == 0.602
    assert cpd["surface_gkp_analog_mwpm_threshold"]["value"] == 0.599
    assert cpd["surface_gkp_cpd_threshold"]["unit"] == "paper_sigma"
    assert report["derived_evidence"]["cpd_same_task_absolute_threshold_delta"] == pytest.approx(0.003)
    fast = methods["project_t5_hybrid_fast_path"]
    assert fast["latency"]["latency_cycles"] == 6
    assert fast["latency"]["core_ns"] == pytest.approx(222.22222222222223)
    assert fast["latency"]["source_to_action_ns"] is None
    assert fast["resources"]["lut"] == 3377
    v5 = methods["project_v5_route_a"]
    assert v5["latency"]["evidence_grade"] == "NEGATIVE"
    assert v5["resources"]["evidence_grade"] == "NEGATIVE"
    assert report["project_anchor_summary"]["v5_verdict"] == "NO_GO_V5_EARLY_HEADROOM_STOP"


def test_null_claims_csv_bindings_and_semantic_mutations_are_complete() -> None:
    report = _report()
    for claim in report["claim_audit"]:
        if claim["verdict"].startswith("NULL_"):
            assert claim["normalized_value"] is None
    csv_path = ROOT / report["source_data"]["path"]
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == report["source_data"]["rows"] == 82
    assert {row["record_type"] for row in rows} == {"method", "metric", "latency", "resources", "claim_audit"}
    for binding in report["bindings"].values():
        path = ROOT / binding["path"]
        assert path.exists()
        assert audit._sha256(path) == binding["sha256"]
    mutations = report["semantic_mutation_audit"]
    assert mutations["count"] == mutations["detected"] == 15
    assert {case["target_gate"] for case in mutations["cases"]} == set(report["gates"])
    assert all(case["rejected"] for case in mutations["cases"])


def test_forged_cross_lane_or_zero_latency_claim_fails_closed() -> None:
    report = _report()
    forged = deepcopy(report)
    forged["comparison_policy"]["global_leaderboard"] = "HYBRID_WINS"
    with pytest.raises(ValueError, match="gates"):
        audit.verify_report(forged)

    forged = deepcopy(report)
    _methods(forged)["lachance_aqec_experiment"]["latency"].update(core_ns=0.0, statistic="zero")
    with pytest.raises(ValueError, match="gates"):
        audit.verify_report(forged)
