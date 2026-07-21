from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import pytest

from cnn_fpga.benchmark import phase6d_claim_figure_delta as delta


ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _report() -> dict:
    return _load(delta.REPORT)


def _element(report: dict, element_id: str) -> dict:
    return next(row for row in report["elements"] if row["element_id"] == element_id)


def _claim(report: dict, claim_id: str) -> dict:
    return next(row for row in report["claims"] if row["claim_id"] == claim_id)


def test_repository_contract_and_rendered_bundle_verify_end_to_end() -> None:
    result = delta.verify()
    bundle = delta.verify_bundle()
    assert result["verdict"] == delta.VERDICT
    assert result["gates"] == {"passed": 22, "total": 22, "failed": []}
    assert result["mutations"] == {"detected": 22, "total": 22}
    assert result["source_rows"] == 94
    assert all(bundle.values())


def test_two_figures_are_lane_separate_without_global_score_or_cross_edge() -> None:
    report = _report()
    figure5 = [row for row in report["elements"] if row["figure_id"] == "Figure 5"]
    figure6 = [row for row in report["elements"] if row["figure_id"] == "Figure 6"]
    assert {row["lane_id"] for row in figure5} == {delta.MM_LANE, delta.LEARNING_LANE}
    assert {row["lane_id"] for row in figure6} == {delta.RTL_LANE}
    assert report["bundle_boundary"]["global_weighted_score"] is None
    assert report["bundle_boundary"]["cross_lane_visual_edges"] == 0
    assert "pL=0.111979" not in report["figures"]["Figure 6"]["caption"]
    assert "36.794 MHz" not in report["figures"]["Figure 5"]["caption"]


def test_every_parent_element_is_consumed_exactly_once_and_scaling_null_is_explicit() -> None:
    report = _report()
    matrix = _load(delta.MATRIX_REPORT)
    parent = {row["element_id"]: row for row in matrix["figure_contract"]["elements"]}
    elements = {row["element_id"]: row for row in report["elements"]}
    assert set(elements) == {*parent, "MM-D6"}
    assert len(elements) == 13
    for element_id, source in parent.items():
        assert elements[element_id]["source_element_payload_sha256"] == delta._canonical_sha256(source)
    assert elements["MM-D6"]["metric_namespace"] == "SCALING"
    assert elements["MM-D6"]["value"] == {
        "distance_scaling": None,
        "sigma_scaling": None,
        "pilot_accessed": False,
        "formal_accessed": False,
    }


def test_multimode_hero_retains_strongest_denominator_and_mandatory_no_go() -> None:
    report = _report()
    value = _element(report, "MM-E4")["value"]
    assert value["strongest_baseline"] == "static_mixture_exact_mld"
    assert value["baseline_p_L"] == value["proposed_p_L"] == 0.11197916666666667
    assert value["relative_improvement_point"] == value["relative_improvement_lcb"] == 0.0
    assert value["formal_or_pilot_accessed"] is False
    claim = _claim(report, "MM_V1_CAUSAL_HEADROOM_NO_GO")
    assert claim["final_disposition"] == "MANDATORY_NEGATIVE"
    assert "rescued by RTL safety" in claim["forbidden_wording"]


def test_opened_multimode_context_has_ler_tail_compute_but_no_promotion() -> None:
    report = _report()
    assert _element(report, "MM-E1")["metric_namespace"] == "LER"
    assert _element(report, "MM-E2")["metric_namespace"] == "TAIL"
    assert _element(report, "MM-E3")["metric_namespace"] == "COMPUTE"
    assert _element(report, "MM-E2")["value"]["candidate_worst_window_ler"] == 0.291015625
    assert _element(report, "MM-E3")["value"]["candidate_seconds_per_decode"] > 0
    claim = _claim(report, "MM_OPENED_TASK_LOCAL_GAIN")
    assert claim["final_disposition"] == "RETAIN_CONTEXT_ONLY"
    assert "not the strongest eligible deployable denominator" in claim["blocking_gaps"]


def test_rtl_panels_are_exact_single_mode_preboard_and_board_values_stay_null() -> None:
    report = _report()
    assert _element(report, "RTL-E1")["value"] == {"cycles": 6, "II": 1}
    assert _element(report, "RTL-E2")["value"]["formal_gates"] == {"passed": 17, "total": 17}
    assert _element(report, "RTL-E3")["value"]["cycles_qualified"] == 1_000_000
    assert _element(report, "RTL-E3")["value"]["mismatches"] == 0
    assert all(row["wrapper_may_dominate"] for row in _element(report, "RTL-E4")["value"]["critical_paths"])
    assert all(value is None for value in _element(report, "RTL-E6")["value"].values())
    assert _claim(report, "RTL_SPEED_ADVANTAGE_PROHIBITED")["final_disposition"] == "PROHIBITED_POSITIVE"


def test_learning_is_only_a_dropped_inset_and_never_votes() -> None:
    report = _report()
    learning = _element(report, "ML-E1")
    assert (learning["figure_id"], learning["panel"], learning["lane_id"]) == ("Figure 5", "d", delta.LEARNING_LANE)
    assert learning["value"] == {"T6.26.1": "Dropped", "T6.26.2": "Dropped", "present_in_primary_rtl": False}
    claim = _claim(report, "LEARNING_APPROXIMATION_DROPPED")
    assert claim["final_disposition"] == "DROPPED_ABLATION_ONLY"
    assert "use legacy CNN to alter either lane verdict" in claim["revocation_conditions"]


def test_all_claims_and_elements_have_full_live_evidence_and_revocation() -> None:
    report = _report()
    artifacts = report["artifact_registry"]
    for row in [*report["claims"], *report["elements"]]:
        evidence = row["evidence"]
        assert all(evidence[category] for category in delta.EVIDENCE_CATEGORIES)
        ids = {artifact_id for category in delta.EVIDENCE_CATEGORIES for artifact_id in evidence[category]}
        assert ids == set(evidence["hashes"])
        assert all(evidence["hashes"][artifact_id] == artifacts[artifact_id]["sha256"] for artifact_id in ids)
        assert row["revocation_conditions"]
    assert all(delta._live(binding) for binding in artifacts.values())


def test_source_data_is_lossless_for_figures_elements_claims_snapshots_and_artifacts() -> None:
    report = _report()
    with delta.SOURCE_DATA.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == delta._source_rows(report)
    assert len(rows) == report["source_data"]["rows"] == 94
    assert report["source_data"]["record_type_counts"] == {
        "artifact": 58,
        "claim": 10,
        "figure_contract": 2,
        "figure_element": 13,
        "historical_snapshot": 11,
    }
    assert all(row["payload_sha256"] == delta._canonical_sha256(json.loads(row["payload_json"])) for row in rows)


def test_historical_snapshot_manifests_are_live_and_new_outputs_are_not_copies() -> None:
    report = _report()
    manifest = _load(delta.MANIFEST)
    assert len(report["historical_snapshots"]) == 11
    assert all(delta._live(binding) for binding in report["historical_snapshots"].values())
    historical = set(manifest["historical_output_sha256"])
    assert historical
    assert not ({row["sha256"] for row in manifest["outputs"].values()} & historical)
    assert all(name.startswith(("figure5_", "figure6_")) for name in manifest["outputs"])


def test_visual_bundle_is_editable_high_resolution_nonblank_and_margin_safe() -> None:
    manifest = _load(delta.MANIFEST)
    qa = manifest["qa"]
    assert qa["manual_visual_qa"] == "PASS"
    assert qa["backend_exclusive"] is True
    assert qa["svg_embedded_raster_count"] == 0
    assert min(qa["svg_text_nodes"].values()) >= 25
    assert min(qa["tiff_min_dimension_px"].values()) >= 2950
    assert min(qa["png_min_dimension_px"].values()) >= 1450
    assert all(0.015 <= value <= 0.55 for value in qa["nonwhite_fraction"].values())
    assert all(value <= 0.03 for value in qa["edge_ink_fraction"].values())


def test_one_semantic_mutation_targets_every_contract_gate() -> None:
    report = _report()
    audit = report["semantic_mutation_audit"]
    assert audit["count"] == audit["detected"] == 22
    assert len(audit["cases"]) == 22
    assert len({row["target_gate"] for row in audit["cases"]}) == 22
    assert all(row["rejected"] for row in audit["cases"])


@pytest.mark.parametrize(
    ("mutation", "gate"),
    [
        (lambda report: _element(report, "MM-E4")["value"].update(proposed_p_L=0.05), "G11_multimode_strongest_baseline_values_and_no_go_are_exact"),
        (lambda report: _element(report, "MM-D6")["value"].update(distance_scaling=0.5), "G12_multimode_pilot_formal_scaling_and_sota_remain_unavailable"),
        (lambda report: _element(report, "RTL-E6")["value"].update(board_latency_ns=1.0), "G13_rtl_six_cycle_ii1_atomic_longrun_and_board_null_are_exact"),
        (lambda report: _claim(report, "LEARNING_APPROXIMATION_DROPPED").update(final_disposition="PRIMARY"), "G15_learning_is_dropped_absent_and_cannot_change_any_verdict"),
        (lambda report: report["bundle_boundary"].update(global_weighted_score=0.8), "G08_no_cross_lane_edge_global_score_or_common_performance_denominator"),
    ],
)
def test_high_risk_manual_mutations_fail_closed(mutation, gate: str) -> None:
    report = _report()
    mutation(report)
    assert delta.evaluate_gates(report, check_live_files=False)[gate] is False
