from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path

import pytest

from cnn_fpga.benchmark import main_result_figure_contract as figures


ROOT = Path(__file__).resolve().parents[1]


def _report() -> dict:
    return json.loads(figures.DEFAULT_REPORT.read_text(encoding="utf-8"))


def _records(report: dict) -> dict[str, dict]:
    return {row["record_id"]: row for row in report["records"]}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_repository_result_figure_contract_and_bundle_verify_end_to_end() -> None:
    assert figures.verify_report() == {"identity": True, "gates": True, "verdict": True, "analysis_hash": True}
    assert figures.verify_bundle() == {
        "identity": True, "contract_live": True, "source_data_live": True,
        "outputs_exact": True, "outputs_live": True, "editable_svg": True,
        "raster_dimensions": True, "visual_contract": True,
    }
    report = _report()
    assert report["gate_summary"] == {"passed": 16, "failed": []}
    assert report["verdict"] == figures.VERDICT
    assert all(report["parent_verification"].values())


def test_all_fifty_five_records_are_unique_source_bound_and_claim_bound() -> None:
    report = _report()
    records = _records(report)
    assert len(records) == len(report["records"]) == 55
    assert {row["figure"] for row in records.values()} == {"Figure 3", "Figure 4"}
    assert all(row["source_ids"] and row["selector"] and row["claim_ids"] for row in records.values())
    assert report["figures"]["Figure 3"]["height_mm"] == 137
    assert report["figures"]["Figure 4"]["height_mm"] == 127


def test_smooth_figure_preserves_narrow_positive_and_mandatory_negative_results() -> None:
    report = _report()
    records = _records(report)
    method_ler = {row["method"]: row["value"] for row in records.values() if row["panel"] == "a" and row["figure"] == "Figure 3"}
    assert min((value, method) for method, value in method_ler.items() if method != "hidden_state_oracle")[1] == "window_map"
    assert method_ler["window_map"] < method_ler["proposed_route_a"] < method_ler["ewma_adaptive_map"]
    assert report["result_boundary"]["route_a_global_best"] is False
    assert records["f3b_ewma_minus_route"]["lower"] > 0
    assert records["f3b_static_minus_route"]["upper"] < 0
    assert records["f3b_oracle_gap"]["value"] < 0 and records["f3b_oracle_gap"]["upper"] < 0


def test_tail_figure_reports_noninferiority_with_fallback_and_recovery_cost() -> None:
    report = _report()
    records = _records(report)
    assert report["result_boundary"]["broad_tail_improvement"] is False
    for family in report["tail_families"]:
        assert records[f"f3c_{family}_ewma_adaptive_map"]["value"] == records[f"f3c_{family}_proposed_route_a"]["value"]
        assert records[f"f3d_{family}_fallback"]["value"] > 0.5
    assert records["f3d_step_calibration_shift_fallback"]["value"] > 0.95
    assert records["f3d_step_calibration_shift_fallback"]["metadata"]["recovery_p95_decisions"] is None
    assert records["f3d_telegraph_drift_fallback"]["metadata"]["recovery_p95_decisions"] == 288.0
    assert records["f3d_nominal_fallback"]["value"] == pytest.approx(0.001193576388888889)
    assert records["f3e_secondary_supplement"]["status"] == "EXCLUDED_FROM_MAIN_RANKING"


def test_hardware_figure_separates_cxxrtl_clock_model_pr_and_board_measurement() -> None:
    records = _records(_report())
    assert records["f4a_qualified_cycles"]["value"] == 1_000_000
    assert all(records[key]["value"] == 0 for key in ("f4a_rtl_mismatches", "f4a_undefined_actions", "f4a_silent_overflow"))
    assert records["f4b_latency_cycles"]["value"] == 6
    assert records["f4b_ii"]["value"] == 1
    assert records["f4b_27mhz_ns"]["value"] == pytest.approx(222.22222222222223)
    assert all(records[key]["status"] == "POST_ROUTE_ESTIMATE" for key in records if key.startswith("f4c_") or key.startswith("f4d_"))
    assert records["f4e_board_null"]["value"] == 42
    assert records["f4e_board_null"]["status"] == "BLOCKED_ALL_NULL"
    assert records["f4e_v5_dropped"]["value"] is None
    assert records["f4e_v5_dropped"]["status"] == "NOT_RUN_DROPPED"


def test_source_csv_is_lossless_and_hashes_every_live_artifact() -> None:
    report = _report()
    artifacts = report["artifact_registry"]
    with figures.DEFAULT_SOURCE_DATA.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 55
    assert {row["record_id"] for row in rows} == {row["record_id"] for row in report["records"]}
    for row in rows:
        source_ids = json.loads(row["source_ids_json"])
        hashes = json.loads(row["source_hashes_json"])
        assert set(hashes) == set(source_ids)
        for source_id in source_ids:
            binding = artifacts[source_id]
            path = ROOT / binding["path"]
            assert path.stat().st_size == binding["bytes"]
            assert _sha256(path) == hashes[source_id] == binding["sha256"]


def test_bundle_is_editable_multiformat_and_publication_resolution() -> None:
    report = _report()
    manifest = json.loads(figures.DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    assert report["export_contract"] == {
        "backend": "Python/matplotlib only", "width_mm": 183,
        "svg_text": "editable", "pdf_fonttype": 42,
        "tiff_dpi": 600, "png_dpi": 300, "outputs": list(figures.FIGURE_OUTPUTS),
    }
    assert manifest["qa"]["manual_visual_qa"] == "PASS"
    assert manifest["qa"]["svg_text_nodes"] >= 50
    assert manifest["qa"]["svg_path_text_promotion"] is False
    assert min(manifest["qa"]["tiff_min_dimension_px"].values()) >= 3000
    assert {Path(name).suffix for name in manifest["outputs"]} == {".svg", ".pdf", ".png", ".tiff"}


def test_each_gate_has_a_detected_mutation_and_promotions_fail_closed() -> None:
    report = _report()
    audit = report["semantic_mutation_audit"]
    assert audit["count"] == audit["detected"] == len(report["gates"]) == 16
    assert {row["target_gate"] for row in audit["cases"]} == set(report["gates"])
    assert all(row["rejected"] for row in audit["cases"])

    promoted = deepcopy(report)
    promoted["result_boundary"]["route_a_global_best"] = True
    assert figures.evaluate_gates(promoted, check_live_files=False)["G05_smooth_all_methods_and_strongest_window_are_explicit"] is False
    with pytest.raises(ValueError, match="verification failed"):
        figures.verify_report(promoted)
