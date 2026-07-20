from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path

from PIL import Image

from cnn_fpga.benchmark import secondary_evidence_integrity_gate as gate


def _report() -> dict:
    return json.loads(gate.DEFAULT_REPORT.read_text(encoding="utf-8"))


def test_repository_report_verifies_end_to_end() -> None:
    assert all(gate.verify_report().values())


def test_six_lanes_and_all_explicit_nonvalue_states_are_present() -> None:
    report = _report()
    assert {cell["lane_id"] for cell in report["cells"]} == set(gate.LANES)
    assert gate.NO_VALUE_STATES <= {cell["value_state"] for cell in report["cells"]}
    assert len(report["cells"]) == len({cell["cell_id"] for cell in report["cells"]}) == 206
    external_metric_cells = [cell for cell in report["cells"] if cell["cell_id"].startswith("fpga_external_") and cell["cell_id"] not in {"fpga_external_same_task"}]
    assert len(external_metric_cells) == 18 * 9


def test_every_cell_has_current_three_way_provenance_and_csv_row() -> None:
    report = _report()
    with gate.DEFAULT_SOURCE_DATA.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(report["cells"])
    assert {row["cell_id"] for row in rows} == {cell["cell_id"] for cell in report["cells"]}
    for cell in report["cells"]:
        assert gate._bindings_current({"report": cell["source_report"], "raw": cell["raw_source"], "config": cell["config"]})


def test_raw_recomputations_close_counts_ci_threshold_timing_and_hashes() -> None:
    report = _report()
    live = gate._live_recomputations()
    assert gate._recomputations_match(report, live) == {
        "single": True, "cnot": True, "threshold": True, "multimode": True,
        "aqec": True, "hardware": True, "external": True,
    }
    assert live["surface_cnot"]["13.0"]["trials"] == 2_424_832
    assert live["structured_cpd"]["independent_total_paired_trials"] == 1_728_000
    assert live["multimode"]["observed_only_posterior_predictive_weighted"]["seed_clusters"] == 96
    assert live["external_fpga"] == {"row_count": 18, "direct_nn_count": 5, "same_task_comparable_count": 0, "physical_board_count": 10}


def test_no_cross_lane_or_evidence_promotion_is_possible() -> None:
    report = _report()
    assert report["ranking_policy"] == {"global_score": False, "global_winner": None, "cross_lane_ranking": False}
    assert all(not cell["ranking_eligible_within_signature"] for cell in report["cells"] if "oracle" in cell["method_id"])
    assert not any(cell["value_state"] == "MEASURED_VALUE" for cell in report["cells"])
    assert report["phase6b_snapshot"]["verdict"] == "NO_GO_V5_EARLY_HEADROOM_STOP"
    assert report["board_blocker_snapshot"]["verdict"].startswith("BLOCKED_T6_9_2")


def test_semantic_mutation_audit_has_one_detected_mutation_per_gate() -> None:
    report = _report()
    rows = report["semantic_mutation_audit"]
    assert len(rows) == len(report["gates"]) == 24
    assert {row["target_gate"] for row in rows} == set(report["gates"])
    assert all(row["detected"] and row["target_gate"] in row["failed_gates"] for row in rows)


def test_material_tampering_fails_the_targeted_gates() -> None:
    report = _report()
    tampered = deepcopy(report)
    tampered["cells"][0]["raw_source"]["sha256"] = "0" * 64
    assert gate.evaluate_gates(tampered, verify_parents=False)["G06_every_cell_has_current_report_raw_config_hash"] is False

    tampered = deepcopy(report)
    tampered["recomputations"]["hardware"]["source_to_action_ns_at_27mhz"] = 6.0
    assert gate.evaluate_gates(tampered, verify_parents=False)["G21_fpga_timing_and_resource_profiles_recompute"] is False

    tampered = deepcopy(report)
    next(cell for cell in tampered["cells"] if cell["value_state"] == "ESTIMATE_VALUE")["value_state"] = "MEASURED_VALUE"
    assert gate.evaluate_gates(tampered, verify_parents=False)["G12_estimate_never_promoted_to_measured"] is False


def test_figure_bundle_is_editable_and_high_resolution() -> None:
    report = _report()
    svg = gate.ROOT / report["figures"]["svg"]["path"]
    assert "<text" in svg.read_text(encoding="utf-8")
    with Image.open(gate.ROOT / report["figures"]["png"]["path"]) as image:
        assert image.width >= 1600 and image.height >= 1600
    with Image.open(gate.ROOT / report["figures"]["tiff"]["path"]) as image:
        assert image.info.get("dpi", (0, 0))[0] >= 590


def test_temporary_build_produces_a_self_consistent_bundle(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    source_path = tmp_path / "source.csv"
    markdown_path = tmp_path / "atlas.md"
    stem = tmp_path / "atlas"
    report = gate.build_report(report_path, source_path, markdown_path, stem)
    assert report["verdict"] == gate.VERDICT
    assert all(gate.verify_report(report, report_path).values())
    assert report_path.is_file() and source_path.is_file() and markdown_path.is_file()
    assert all(stem.with_suffix(f".{suffix}").is_file() for suffix in ("svg", "pdf", "tiff", "png"))
