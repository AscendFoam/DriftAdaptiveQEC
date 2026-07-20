from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path

import pytest

from cnn_fpga.benchmark import main_figure_contract as figures


ROOT = Path(__file__).resolve().parents[1]


def _report() -> dict:
    return json.loads(figures.DEFAULT_REPORT.read_text(encoding="utf-8"))


def _manifest() -> dict:
    return json.loads(figures.DEFAULT_MANIFEST.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _elements(report: dict) -> dict[str, dict]:
    return {row["element_id"]: row for row in report["elements"]}


def test_repository_figure_contract_and_rendered_bundle_verify_end_to_end() -> None:
    assert figures.verify_report() == {
        "identity": True,
        "gates": True,
        "verdict": True,
        "analysis_hash": True,
    }
    assert figures.verify_bundle() == {
        "identity": True,
        "contract_live": True,
        "source_data_live": True,
        "outputs_exact": True,
        "outputs_live": True,
        "editable_svg": True,
        "raster_dimensions": True,
        "visual_contract": True,
    }
    report = _report()
    assert report["gate_summary"] == {"passed": 16, "failed": []}
    assert report["verdict"] == figures.VERDICT


def test_figure_elements_are_complete_unique_and_separate_execution_layers() -> None:
    report = _report()
    rows = report["elements"]
    elements = _elements(report)
    assert len(rows) == len(elements) == 38
    assert {row["figure"] for row in rows} == {"Figure 1", "Figure 2"}
    assert report["figures"]["Figure 1"]["height_mm"] == 127
    assert report["figures"]["Figure 2"]["height_mm"] == 137
    assert elements["f1a_fast"]["evidence_layer"] == "CXXRTL_PREBOARD"
    assert elements["f1a_host"]["status"] == "SOFTWARE_SLOW_LOOP"
    assert elements["f1a_learning_sidecar"]["status"] == "NOT_PRIMARY_NOT_FAST_ACTION"
    assert elements["f1a_plant"]["status"] == "CONTEXT_NOT_HARDWARE"
    assert report["forbidden_promotions"] == [
        "HMM or CNN in RTL",
        "V5 module implemented",
        "post-route estimate as board measurement",
        "measured speed or power advantage",
    ]


def test_timing_and_board_boundary_are_exact_and_not_interchangeable() -> None:
    elements = _elements(_report())
    assert elements["f1c_pipeline"]["value"] == 6
    assert elements["f1c_pipeline"]["status"] == "CLOCK_MODEL_NOT_BOARD"
    assert elements["f1c_ii"]["value"] == 1
    assert elements["f1c_update"]["value"] == 4000
    assert elements["f1b_pr"]["evidence_layer"] == "POST_ROUTE_ESTIMATE"
    assert elements["f1b_board"]["value"] == 42
    assert elements["f1b_board"]["status"] == "BLOCKED_ALL_FIELDS_NULL"
    assert elements["f1c_board_latency"]["value"] is None
    assert elements["f1c_board_latency"]["evidence_layer"] == "BOARD_MEASURED"


def test_safe_adaptation_uses_observed_inputs_typed_actions_and_atomic_recovery() -> None:
    elements = _elements(_report())
    observed = ("f2a_syndrome", "f2a_health", "f2a_version")
    regimes = ("f2b_smooth", "f2b_tail", "f2b_leakage", "f2b_integrity")
    actions = ("f2c_update", "f2c_trusted", "f2c_reset", "f2c_rollback")
    transaction = ("f2d_candidate", "f2d_inactive", "f2d_commit", "f2d_recover")
    assert all(elements[key]["status"] == "OBSERVED_ONLY" for key in observed)
    assert all(elements[key]["status"] == "IMPLEMENTED_POLICY_BRANCH" for key in regimes)
    assert all(elements[key]["status"] == "TYPED_ACTION" for key in actions)
    assert all(elements[key]["status"] == "ATOMIC_PRECONDITION" for key in transaction)
    assert elements["f2d_recover"]["value"] == 8


def test_dropped_v5_and_null_board_results_are_visible_boundaries_only() -> None:
    elements = _elements(_report())
    assert elements["f2e_v5"]["status"] == "NOT_RUN_DROPPED"
    assert elements["f2e_v5_rtl"]["status"] == "NOT_RUN_DROPPED"
    assert elements["f2e_board"]["status"] == "BLOCKED_NULL"
    labels = " ".join(elements[key]["label"].lower() for key in ("f2e_v5", "f2e_v5_rtl"))
    assert all(token in labels for token in ("imm", "bocpd", "posterior-mixture", "risk compiler", "quantized", "formal", "cxxrtl", "p&r"))


def test_source_csv_is_lossless_and_each_element_carries_live_source_hashes() -> None:
    report = _report()
    artifacts = report["artifact_registry"]
    with figures.DEFAULT_SOURCE_DATA.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 38
    assert {row["element_id"] for row in rows} == {row["element_id"] for row in report["elements"]}
    for row in rows:
        source_ids = json.loads(row["source_ids_json"])
        hashes = json.loads(row["source_hashes_json"])
        assert set(hashes) == set(source_ids)
        for source_id in source_ids:
            binding = artifacts[source_id]
            source_path = ROOT / binding["path"]
            assert source_path.stat().st_size == binding["bytes"]
            assert _sha256(source_path) == hashes[source_id] == binding["sha256"]


def test_bundle_is_python_only_editable_multiformat_and_publication_resolution() -> None:
    report = _report()
    manifest = _manifest()
    assert report["export_contract"] == {
        "backend": "Python/matplotlib only",
        "width_mm": 183,
        "svg_text": "editable",
        "pdf_fonttype": 42,
        "tiff_dpi": 600,
        "png_dpi": 300,
        "outputs": list(figures.FIGURE_OUTPUTS),
    }
    assert manifest["backend"] == "Python/matplotlib only"
    assert manifest["qa"]["manual_visual_qa"] == "PASS"
    assert manifest["qa"]["svg_text_nodes"] >= 50
    assert manifest["qa"]["svg_path_text_promotion"] is False
    assert min(manifest["qa"]["tiff_min_dimension_px"].values()) >= 3000
    assert set(manifest["outputs"]) == set(figures.FIGURE_OUTPUTS)
    assert {Path(name).suffix for name in manifest["outputs"]} == {".svg", ".pdf", ".png", ".tiff"}


def test_promotion_mutations_fail_closed_and_every_gate_has_a_mutation() -> None:
    report = _report()
    audit = report["semantic_mutation_audit"]
    assert audit["count"] == audit["detected"] == len(report["gates"]) == 16
    assert {row["target_gate"] for row in audit["cases"]} == set(report["gates"])
    assert all(row["rejected"] for row in audit["cases"])

    promoted = deepcopy(report)
    next(row for row in promoted["elements"] if row["element_id"] == "f1b_pr")["evidence_layer"] = "BOARD_MEASURED"
    assert figures.evaluate_gates(promoted, check_live_files=False)["G07_postroute_and_board_measurement_are_never_merged"] is False
    with pytest.raises(ValueError, match="verification failed"):
        figures.verify_report(promoted)
