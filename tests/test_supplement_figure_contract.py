from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image

from cnn_fpga.benchmark import supplement_figure_contract as contract


def _report() -> dict:
    return json.loads(contract.DEFAULT_REPORT.read_text(encoding="utf-8"))


def _categories(report: dict) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    for row in report["records"]:
        result.setdefault(row["category"], []).append(row)
    return result


def test_report_and_bundle_verify_live() -> None:
    assert all(contract.verify_report().values())
    assert all(contract.verify_bundle().values())


def test_source_data_is_lossless_and_hash_bound() -> None:
    report = _report()
    with contract.DEFAULT_SOURCE_DATA.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(report["records"]) == 792
    assert {row["record_id"] for row in rows} == {row["record_id"] for row in report["records"]}
    assert all(json.loads(row["source_ids_json"]) for row in rows)
    assert all(json.loads(row["source_hashes_json"]) for row in rows)


def test_physics_validity_domains_are_not_promoted() -> None:
    categories = _categories(_report())
    assert len(categories["gradient"]) == 4
    assert len(categories["cutoff_feasibility"]) == 65
    assert sum(row["status"] == "MEMORY_EXCEEDED" for row in categories["cutoff_feasibility"]) == 1
    assert sum(row["status"] == "RUNTIME_EXCEEDED" for row in categories["cutoff_feasibility"]) == 1
    failed_db = {row["metadata"]["squeezing_db"] for row in categories["noise_transfer_domain"] if row["status"] == "FAILURE_DOMAIN"}
    valid_db = {row["metadata"]["squeezing_db"] for row in categories["noise_transfer_domain"] if row["status"] == "VALID_DOMAIN"}
    assert failed_db == {3.0, 5.0, 8.0}
    assert valid_db == {10.0, 12.0}


def test_petz_topk_and_six_pauli_keep_distinct_semantics() -> None:
    categories = _categories(_report())
    assert len(categories["petz_small_sdp"]) == 15
    assert len(categories["petz_cutoff_extension"]) == 15
    assert all(row["status"].startswith("NONDEPLOYABLE_BOUND") for row in categories["petz_small_sdp"] + categories["petz_cutoff_extension"])
    assert len(categories["topk_pareto"]) == 48
    assert {row["metadata"]["K"] for row in categories["topk_pareto"]} == {1, 2, 4, 8, 16, 32, 64, 128}
    assert all(row["metadata"]["target_measured"] is False for row in categories["topk_pareto"])
    assert len(categories["six_pauli_states"]) == 372
    assert {row["family"] for row in categories["six_pauli_states"]} == {"x_plus", "x_minus", "y_plus", "y_minus", "z_plus", "z_minus"}


def test_all_seed_and_full_ood_evidence_is_complete() -> None:
    report = _report()
    categories = _categories(report)
    assert len(categories["all_seed_distribution"]) == 24 * 7
    assert len({row["metadata"]["seed"] for row in categories["all_seed_distribution"]}) == 24
    assert (len(categories["ood_drift"]), len(categories["ood_measurement"]), len(categories["ood_leakage"]), len(categories["ood_communication"])) == (18, 3, 3, 8)
    assert report["result_boundary"]["ood_system_robustness"] == "NOT_ESTABLISHED_LANE_LOCAL_ONLY"
    assert report["result_boundary"]["ood_device_robustness"] == "NOT_ESTABLISHED_NO_TARGET_HARDWARE"


def test_fixed_point_and_failure_ledger_fail_closed() -> None:
    report = _report()
    categories = _categories(report)
    assert len(categories["fixed_point_oat"]) == 46
    assert len(categories["fixed_point_production"]) == 4
    assert all(row["metadata"]["target_synthesis_measured"] is False for row in categories["fixed_point_oat"])
    assert len(categories["failure_mode"]) == 9
    assert {row["status"] for row in categories["failure_mode"]} >= {"FAILURE_DOMAIN", "RESOURCE_BOUNDARY", "NOT_SYNTHESIZED", "INCOMPARABLE", "NOT_ESTABLISHED", "DROPPED", "BLOCKED_ALL_NULL"}
    assert report["result_boundary"]["fixed_point_board_measured"] is False


def test_phase6c_atlas_is_linked_without_global_ranking() -> None:
    report = _report()
    rows = _categories(report)["phase6c_linked_atlas"]
    assert len(rows) == 6
    assert sum(row["value"] for row in rows) == 206
    assert report["result_boundary"]["phase6c_global_ranking"] is False
    assert set(report["linked_outputs"]) == set(contract.LINKED_OUTPUTS)
    for name, source_id in contract.LINKED_OUTPUTS.items():
        assert report["linked_outputs"][name] == contract._binding(contract.SOURCES[source_id])


def test_each_gate_has_a_detected_substantive_mutation() -> None:
    report = _report()
    audit = report["semantic_mutation_audit"]
    assert audit["count"] == audit["detected"] == len(report["gates"]) == 17
    assert {row["target_gate"] for row in audit["cases"]} == set(report["gates"])
    assert all(row["rejected"] for row in audit["cases"])


def test_multiformat_outputs_are_editable_and_publication_resolution() -> None:
    manifest = json.loads(contract.DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    assert set(manifest["outputs"]) == set(contract.ALL_OUTPUTS)
    assert manifest["qa"]["manual_visual_qa"] == "PASS"
    assert manifest["qa"]["svg_text_nodes"] >= 120
    for name, binding in manifest["outputs"].items():
        path = contract.ROOT / binding["path"]
        assert path.is_file() and path.stat().st_size == binding["bytes"]
        if name.endswith(".svg"):
            assert "<text" in path.read_text(encoding="utf-8")
        if name.endswith(".tiff"):
            with Image.open(path) as image:
                assert min(image.size) >= 2800
