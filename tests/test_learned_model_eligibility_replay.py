from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path

import numpy as np

from cnn_fpga.benchmark import learned_model_eligibility_replay as audit


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "docs" / "t6_17_3_learned_model_eligibility_replay.json"


def _report() -> dict:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def _by_id(report: dict, candidate_id: str) -> dict:
    return next(row for row in report["candidates"] if row["candidate_id"] == candidate_id)


def test_report_recomputes_all_fail_closed_gates() -> None:
    report = _report()
    audit.verify_report(report)
    assert report["verdict"] == audit.VERDICT
    assert report["gate_summary"] == {"passed": 16, "failed": []}
    assert report["semantic_mutation_audit"]["count"] == 16
    assert report["semantic_mutation_audit"]["detected"] == 16
    assert all(row["rejected"] for row in report["semantic_mutation_audit"]["cases"])


def test_candidate_universe_is_complete_and_preserves_family_multiplicity() -> None:
    report = _report()
    assert report["eligibility_summary"] == {
        "candidate_families": 16,
        "same_task_eligible": 0,
        "eligible_replayed": 0,
        "ineligible": 16,
        "diagnostic_replays": 1,
    }
    assert _by_id(report, "t327_latest_outcome_fnn")["member_count"] == 5
    assert _by_id(report, "t3210_exponential_recurrence")["member_count"] == 3
    assert _by_id(report, "t415_distilled_recurrence_student")["member_count"] == 3
    assert _by_id(report, "t441_bounded_residual_gru_teacher")["member_count"] == 3
    assert _by_id(report, "t443_distilled_state4_student")["member_count"] == 9
    assert _by_id(report, "t545_horizon_student_family")["member_count"] == 10
    assert _by_id(report, "t237_project_nmf_controller")["member_count"] == 5


def test_all_signatures_are_recomputed_from_required_contract() -> None:
    report = _report()
    for row in report["candidates"]:
        assert set(row["signature"]) == set(audit.SIGNATURE_FIELDS)
        expected_checks = {
            field: {
                "required": audit.REQUIRED_SIGNATURE[field],
                "actual": row["signature"][field],
                "match": row["signature"][field] == audit.REQUIRED_SIGNATURE[field],
            }
            for field in audit.SIGNATURE_FIELDS
        }
        assert row["signature_checks"] == expected_checks
        assert row["mismatch_fields"] == [
            field for field in audit.SIGNATURE_FIELDS
            if row["signature"][field] != audit.REQUIRED_SIGNATURE[field]
        ]
        assert not row["same_task_eligible"]
        assert row["mismatch_fields"]


def test_budget_schema_is_complete_without_fabricating_unknown_values() -> None:
    report = _report()
    expected = set(audit.REQUIRED_BUDGET_FIELDS) | {"provenance"}
    assert all(set(row["budget"]) == expected for row in report["candidates"])
    assert _by_id(report, "t411_causal_tcn")["budget"]["MAC"] == 3556
    assert _by_id(report, "t411_small_gru")["budget"]["MAC"] == 2300
    assert _by_id(report, "t327_latest_outcome_fnn")["budget"]["MAC"] == 72_266
    assert _by_id(report, "t3210_exponential_recurrence")["budget"]["MAC"] == 15
    assert _by_id(report, "t415_distilled_recurrence_student")["budget"]["MAC"] == 15
    assert _by_id(report, "t443_distilled_state4_student")["budget"]["MAC"] == 87
    assert _by_id(report, "wang2022_direct_nn")["budget"]["MAC"] is None
    assert _by_id(report, "gqf_official_nmf_controller")["budget"]["workspace_bytes"] is None


def test_ineligible_candidates_have_null_unranked_primary_metrics() -> None:
    report = _report()
    for row in report["candidates"]:
        assert set(row["metrics"]) == set(audit.METRICS)
        for metric in row["metrics"].values():
            assert metric == {
                "value": None,
                "value_state": "N_A_NOT_APPLICABLE",
                "ranking_eligible": False,
                "reason": "INELIGIBLE_TASK_SIGNATURE",
            }
    assert report["cross_lane_aggregate"] is None
    assert report["global_ranking"] is None


def test_legacy_cnn_diagnostic_replay_is_exact_but_not_promoted_to_ler() -> None:
    replay = _report()["diagnostic_replay"]
    assert replay["state"] == "DIAGNOSTIC_REPLAY_EXACT_NOT_RANKED"
    assert replay["samples"] == len(replay["rows"]) == 206
    assert replay["repeat_count"] == len(replay["repeat_output_sha256s"]) == 5
    assert len(set(replay["repeat_output_sha256s"])) == 1
    assert replay["output_sha256"] == replay["repeat_output_sha256s"][0]
    assert replay["bit_exact_across_repeats"]
    assert replay["bit_exact_with_t5_4_3_preserved_predictions"]
    assert replay["maximum_abs_difference_from_t5_4_3"] == 0.0
    assert "not decoder latency_ns" in replay["host_timing_boundary"]


def test_legacy_diagnostic_metrics_recompute_from_raw_rows() -> None:
    replay = _report()["diagnostic_replay"]
    errors = np.asarray([row["squared_error"] for row in replay["rows"]], dtype=np.float64)
    assert np.isclose(replay["mse"], errors.mean(), rtol=0.0, atol=1e-20)
    assert np.isclose(replay["mse"], replay["parent_mse"], rtol=0.0, atol=1e-20)
    assert np.isclose(replay["mse"], replay["parent_evaluation_report_mse"], rtol=0.0, atol=1e-20)
    assert np.isclose(
        replay["mae"],
        np.mean([
            abs(pred - target)
            for row in replay["rows"]
            for pred, target in zip(row["prediction"], row["target"], strict=True)
        ]),
        rtol=0.0,
        atol=1e-20,
    )


def test_external_methods_remain_in_native_lanes_without_checkpoint_replay() -> None:
    report = _report()
    assert _by_id(report, "wang2022_direct_nn")["native_lane"] == "surface_gkp_gate_outer_code"
    for candidate_id in ("sivak2023_rl_controller", "sivak2026_rl_drift"):
        row = _by_id(report, candidate_id)
        assert row["native_lane"] == "controller_rl_nmf"
        assert row["artifact"] is None
        assert row["replay_state"] == "NOT_REPLAYED_INELIGIBLE"
    for candidate_id in ("t237_project_nmf_controller", "gqf_official_nmf_controller"):
        assert _by_id(report, candidate_id)["native_lane"] == "controller_rl_nmf"


def test_execution_contract_proves_zero_training_and_preserves_phase6b_verdict() -> None:
    report = _report()
    assert report["execution_contract"] == {
        "training_executed": False,
        "hyperparameter_search_executed": False,
        "checkpoint_reselection_executed": False,
        "new_checkpoint_written": False,
        "performance_p_value_computed": False,
        "phase6b_outputs_modified": False,
    }
    assert report["claim_registry"]["PHASE6B_V5_VERDICT"] == "READ_ONLY_NO_GO_UNCHANGED"


def test_live_gate_rejects_forged_signature_and_output_hash() -> None:
    report = _report()
    forged_signature = deepcopy(report)
    _by_id(forged_signature, "legacy_residual_tinycnn")["signature_checks"]["output_action"]["match"] = True
    assert not audit.evaluate_gates(forged_signature)[
        "G04_every_candidate_has_complete_13_field_signature_and_live_mismatch_projection"
    ]

    forged_hash = deepcopy(report)
    forged_hash["diagnostic_replay"]["output_sha256"] = "0" * 64
    assert not audit.evaluate_gates(forged_hash)[
        "G10_checkpoint_input_output_and_parent_hashes_are_complete"
    ]


def test_source_data_has_exact_semantic_cardinality_and_states() -> None:
    report = _report()
    path = ROOT / report["source_data"]["path"]
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == report["source_data"]["rows"] == 434
    counts = {kind: sum(row["record_type"] == kind for row in rows) for kind in {
        "candidate", "signature", "diagnostic_replay", "source"
    }}
    assert counts == {"candidate": 16, "signature": 208, "diagnostic_replay": 206, "source": 4}
    assert all(row["value_state"] == "DIAGNOSTIC_NOT_RANKED" for row in rows if row["record_type"] == "diagnostic_replay")


def test_every_bound_artifact_hash_is_live() -> None:
    report = _report()
    for binding in report["bindings"].values():
        path = ROOT / binding["path"]
        assert path.is_file()
        assert audit._sha256(path) == binding["sha256"]
        assert path.stat().st_size == binding["bytes"]


def test_markdown_keeps_diagnostic_and_latency_boundaries_explicit() -> None:
    text = (ROOT / "docs" / "learned_model_eligibility_replay.md").read_text(encoding="utf-8")
    assert "same-task eligible=0" in text
    assert "不能被包装成" not in text  # report stays technical rather than rhetorical
    assert "不证明 logical decoding" in text
    assert "不转换为 `latency_ns`" in text
    assert "不与 single-mode Pauli decoder 合并" in text
