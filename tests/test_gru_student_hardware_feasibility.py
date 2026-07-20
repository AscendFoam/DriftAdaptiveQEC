from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from cnn_fpga.benchmark import gru_student_hardware_feasibility as feasibility


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "docs/t5_5_4_gru_student_hardware_feasibility.json"
CSV_PATH = ROOT / "docs/t5_5_4_gru_student_hardware_feasibility_source_data.csv"


def _report() -> dict:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def test_formal_report_recomputes_all_fifteen_gates() -> None:
    report = _report()
    gates = feasibility.evaluate_gates(report)
    assert all(gates.values())
    assert report["gates"]["all_shortcut_mutations_are_rejected"]
    assert report["gate_summary"] == {"passed": 16, "total": 16}
    assert report["status"] == "PASS"


def test_selected_checkpoint_parameter_accounting_is_exact() -> None:
    report = _report()
    accounting = report["parameter_accounting"]
    assert accounting["architecture"] == "GRU10-DENSE256-DENSE256-OUT15"
    assert accounting["weight_macs"] == 72_266
    assert accounting["bias_scalars"] == 587
    assert accounting["total_parameters"] == 72_853
    assert accounting["weight_macs"] + accounting["bias_scalars"] == accounting["total_parameters"]


def test_quantized_memory_covers_every_parameter_and_is_hash_bound() -> None:
    manifest = json.loads(feasibility.QUANT_MANIFEST.read_text(encoding="utf-8"))
    weights = feasibility._load_signed_mem(ROOT / manifest["weight_file"]["path"], 8)
    biases = feasibility._load_signed_mem(ROOT / manifest["bias_file"]["path"], 18)
    assert len(weights) == 72_266
    assert len(biases) == 587
    assert min(weights) >= -128 and max(weights) <= 127
    assert min(biases) >= -(1 << 17) and max(biases) < (1 << 17)
    assert feasibility._sha256(ROOT / manifest["weight_file"]["path"]) == manifest["weight_file"]["sha256"]
    assert feasibility._sha256(ROOT / manifest["bias_file"]["path"]) == manifest["bias_file"]["sha256"]


def test_manual_gru_equation_matches_torch_and_shadow_stays_bounded() -> None:
    shadow = _report()["quantized_functional_shadow"]
    assert shadow["torch_grucell_equation_max_abs_error"] <= 1e-12
    for lane in ("exhaustive_length8_histories_all_prefixes", "long_random_sequences"):
        metrics = shadow[lane]
        assert metrics["all_quantized_values_finite"]
        assert metrics["all_quantized_actions_bounded"]
        assert metrics["action_maximum_absolute_error"] <= 5e-3
    assert shadow["physical_gain_retention"] is None


def test_small_functional_shadow_is_deterministic_and_finite() -> None:
    floating, quantized, _ = feasibility.load_teacher_states()
    outcomes = np.asarray([[0, 1, 0, 1], [1, 1, 0, 0]], dtype=np.float64)
    first = feasibility._evaluate_sequences(outcomes, floating, quantized)
    second = feasibility._evaluate_sequences(outcomes, floating, quantized)
    assert first == second
    assert first["sequence_count"] == 2
    assert first["action_scalar_comparisons"] == 2 * 4 * 15
    assert first["all_quantized_values_finite"]


def test_full_gru_storage_failure_is_computed_before_synthesis() -> None:
    rows = _report()["candidates"][:2]
    expected = {64: 261, 32: 135}
    for row in rows:
        assert row["integrated_bram_blocks_lower_bound"] == expected[row["weight_bits"]]
        assert row["integrated_bram_blocks_lower_bound"] > 46
        assert row["actual_target_synthesis"] is False
        assert not row["capacity_pass"]
        assert not row["enhanced_route_eligible"]


def test_quantized_gru_workload_is_exact_but_only_a_lower_bound() -> None:
    report = _report()
    trace = report["quantized_lower_bound_cxxrtl"]
    quantized = report["candidates"][2]
    assert trace["cycles_after_start"] == 72_854
    assert trace["weight_macs_completed"] == 72_266
    assert trace["biases_consumed"] == 587
    assert trace["done"] and not trace["busy"]
    assert trace["signature_matches_independent_reference"]
    assert trace["signature"] == trace["independent_reference"]["signature"]
    assert quantized["functional_model"] is False
    assert quantized["worst_case_latency_us"] is None
    assert quantized["latency_us_at_27mhz_lower_bound"] > 2_000
    assert not quantized["deadline_pass"]


def test_three_post_route_seeds_fit_target_but_do_not_qualify_gru() -> None:
    report = _report()
    routes = report["quantized_lower_bound_place_route"]
    assert [row["seed"] for row in routes] == [1, 7, 19]
    assert all(row["route_status"] == "PASS" and row["timing_pass"] for row in routes)
    assert min(row["achieved_fmax_mhz"] for row in routes) == report["quantized_lower_bound_fmax_mhz"]["minimum"]
    assert report["quantized_lower_bound_resources"]["BSRAM"] == {"used": 41, "available": 46}
    assert report["selection"]["quantized_gru_enhanced_route"] == "dropped"


def test_student_is_unique_route_with_measured_retention() -> None:
    report = _report()
    selected = report["candidates"][3]
    assert selected["candidate_id"] == report["selection"]["candidate_id"]
    assert selected["functional_model"] and selected["deadline_pass"] and selected["capacity_pass"]
    assert selected["cxxrtl_cycles"] == 64
    assert selected["worst_case_latency_us_at_27mhz"] < 5.0
    assert selected["physical_gain_retention"]["minimum_point"] >= 0.9
    assert selected["physical_gain_retention"]["minimum_ci_lower"] >= 0.9
    assert [row["candidate_id"] for row in report["candidates"] if row["enhanced_route_eligible"]] == [selected["candidate_id"]]


def test_all_eleven_shortcut_mutations_are_rejected() -> None:
    report = _report()
    mutations = feasibility.mutation_audit(report)
    assert len(mutations) == 12
    assert all(row["rejected"] and row["failed_gates"] for row in mutations)


def test_source_data_has_all_four_candidate_rows() -> None:
    report = _report()
    with CSV_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert {row["candidate_id"] for row in rows} == {row["candidate_id"] for row in report["candidates"]}
    assert feasibility._sha256(CSV_PATH) == report["source_data"]["sha256"]


def test_evidence_boundary_never_claims_functional_gru_or_board_data() -> None:
    boundary = _report()["evidence_boundary"]
    assert boundary["quantized_gru_lower_bound_target_post_route"]
    assert not boundary["quantized_gru_functional_rtl"]
    assert not boundary["quantized_gru_physical_gain_retention"]
    assert boundary["student_physical_gain_retention"]
    assert not boundary["vendor_timing_signoff"]
    assert not boundary["board_measured"]


def test_all_durable_artifacts_and_sources_still_match_hashes() -> None:
    report = _report()
    assert all(feasibility._matches(row) for row in report["parents"] for row in [row["artifact"]])
    assert all(feasibility._matches(row) for row in report["source_bindings"])
    assert all(feasibility._matches(row) for row in report["durable_artifacts"])
