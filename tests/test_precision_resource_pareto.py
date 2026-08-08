from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import pytest

from cnn_fpga.benchmark.precision_resource_pareto import (
    DEFAULT_CSV,
    DEFAULT_JSON,
    PRECISION_IDS,
    STATE_DIMENSIONS,
    TOPK_VALUES,
    VERDICT,
    evaluate_gates,
)


ROOT = Path(__file__).resolve().parents[1]


def _report() -> dict:
    return json.loads(DEFAULT_JSON.read_text(encoding="utf-8"))


def test_formal_pareto_report_passes_all_gates() -> None:
    report = _report()
    assert report["status"] == "PASS"
    assert report["verdict"] == VERDICT
    assert report["gate_summary"] == {"passed": 16, "total": 16}
    assert all(report["gates"].values())
    assert all(evaluate_gates(report).values())


def test_precision_axis_uses_parent_ler_and_smallest_thresholded_profile() -> None:
    rows = _report()["axes"]["precision"]
    assert [row["profile_id"] for row in rows] == list(PRECISION_IDS)
    assert [row["profile_id"] for row in rows if row["quality_pass"]] == [
        "selected_p10_a8_q9_12", "dense_p12_a10_q10_14"
    ]
    selected = rows[2]
    assert selected["action_disagreement_mean"] < 1e-4
    assert selected["estimated_bram_blocks_for_eight_mirrors"] == 8
    assert rows[3]["estimated_bram_blocks_for_eight_mirrors"] == 16


def test_topk_axis_selects_k4_but_keeps_it_off_device() -> None:
    rows = _report()["axes"]["topk"]
    assert [row["k"] for row in rows] == list(TOPK_VALUES)
    assert [row["k"] for row in rows if row["all_scenarios_converged"]] == [4]
    assert rows[-1]["scenario_count"] == 6
    assert rows[-1]["maximum_absolute_ler_delta"] <= 1e-4
    assert all(row["hardware_role"].startswith("off_device") for row in rows)


def test_student_axis_rejects_smaller_dimensions_from_parent_evidence() -> None:
    rows = _report()["axes"]["student_dimension"]
    assert [row["dimension"] for row in rows] == list(STATE_DIMENSIONS)
    assert [row["dimension"] for row in rows if row["parent_dimension_eligible"]] == [4]
    assert rows[2]["evaluation_mse"] == pytest.approx(6.08313615636731e-06)
    assert rows[2]["physical_gain_retention_ci_lower_minimum"] == pytest.approx(0.9445014278749587)
    assert rows[0]["physical_gain_retention_ci_lower_minimum"] is None


def test_parallelism_axis_distinguishes_real_serial_rtl_from_extrapolation() -> None:
    rows = _report()["axes"]["parallelism"]
    assert [row["multipliers"] for row in rows] == [1, 2, 4]
    assert [row["selected_state4_latency_cycles"] for row in rows] == [64, 32, 16]
    assert rows[0]["evidence_level"] == "actual_fixed_rtl"
    assert all("not_rtl" in row["evidence_level"] for row in rows[1:])


def test_integrated_student_survives_synthesis_as_real_state_and_dsp() -> None:
    report = _report()
    synthesis = report["integrated_synthesis"]
    assert synthesis["zero_structural_problems"] is True
    assert synthesis["cell_counts"]["SDPX9B"] == 8
    assert synthesis["cell_counts"]["MULT18X18"] == 2
    delta = report["increment_over_t5_5_2_maxima"]
    assert delta["LUT4"] == 440
    assert delta["DFF"] == 157
    assert delta["BSRAM"] == 0
    assert delta["MULT18X18"] == 1


def test_three_integrated_seeds_pass_and_worst_is_reported() -> None:
    report = _report()
    routes = report["integrated_place_route"]
    assert [row["seed"] for row in routes] == [1, 7, 19]
    assert all(row["timing_pass"] and row["route_status"] == "PASS" for row in routes)
    assert report["integrated_fmax_mhz"]["minimum"] == pytest.approx(39.5726165771)
    assert report["integrated_fmax_mhz"]["median"] == pytest.approx(40.3225822449)
    assert report["integrated_fmax_mhz"]["maximum"] == pytest.approx(40.5350608826)


def test_integrated_resource_maxima_fit_target() -> None:
    resources = _report()["integrated_post_route_resources"]
    expected = {"LUT4": 3802, "DFF": 1022, "BSRAM": 8, "MULT18X18": 2, "MULT9X9": 1, "ALU": 616, "IOB": 18}
    for name, used in expected.items():
        assert resources[name]["used"] == used
        assert used <= resources[name]["available"]


def test_joint_grid_has_108_rows_and_one_final_candidate() -> None:
    rows = _report()["candidates"]
    assert len(rows) == 4 * 3 * 3 * 3 == 108
    assert len({row["candidate_id"] for row in rows}) == 108
    selected = [row for row in rows if row["final_eligible"]]
    assert len(selected) == 1
    assert selected[0]["candidate_id"] == "selected_p10_a8_q9_12__k4__d4__p1"


def test_selected_candidate_is_actual_not_formula_proxy() -> None:
    selected = _report()["selection"]
    assert selected["resource_evidence_level"] == "actual_three_seed_integrated_post_route"
    assert selected["measured_resources"] == {
        "LUT4": 3802, "DFF": 1022, "BSRAM": 8,
        "MULT18X18": 2, "MULT9X9": 1, "ALU": 616, "IOB": 18,
    }
    assert selected["measured_fmax_mhz_minimum"] == pytest.approx(39.5726165771)
    assert selected["topk_hardware_resources_included"] is False


def test_serial_student_meets_model_deadline_without_parallel_dsp_claim() -> None:
    selected = _report()["selection"]
    assert selected["student_latency_cycles"] == 64
    assert selected["student_latency_us_at_27mhz"] == pytest.approx(64 / 27)
    assert selected["student_latency_us_at_27mhz"] < 5.0


def test_all_unsynthesized_rows_are_explicit_estimates() -> None:
    for row in _report()["candidates"]:
        actual = row["resource_evidence_level"] == "actual_three_seed_integrated_post_route"
        assert (row["measured_resources"] is not None) == actual
        if not actual:
            assert row["measured_fmax_mhz_minimum"] is None
            assert row["resource_evidence_level"] == "calibrated_estimate_not_synthesis"


def test_source_data_is_the_full_candidate_matrix() -> None:
    report = _report()
    with DEFAULT_CSV.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == report["source_data"]["candidate_rows"] == 108
    assert sum(row["final_eligible"] == "True" for row in rows) == 1


def test_all_sources_parents_and_tool_artifacts_are_live() -> None:
    report = _report()
    for parent in report["parents"]:
        assert parent["status"] == "PASS"
        assert parent["gate_summary"]["passed"] == parent["gate_summary"]["total"]
    for row in report["source_bindings"] + report["durable_artifacts"]:
        path = ROOT / row["path"]
        assert path.is_file() and path.stat().st_size == row["bytes"]


def test_every_shortcut_mutation_is_rejected() -> None:
    rows = _report()["mutation_audit"]
    assert len(rows) == 10
    assert all(row["rejected"] and row["failed_gates"] for row in rows)


def test_recomputation_rejects_false_board_claim() -> None:
    report = copy.deepcopy(_report())
    report["evidence_boundary"]["board_measured"] = True
    gates = evaluate_gates(report)
    assert gates["post_route_pareto_is_not_mislabeled_as_vendor_or_board_evidence"] is False


def test_evidence_boundary_does_not_claim_online_topk_or_board() -> None:
    boundary = _report()["evidence_boundary"]
    assert boundary["student_cxxrtl_equivalence"] is True
    assert boundary["integrated_target_post_route"] is True
    assert boundary["online_topk_rtl"] is False
    assert boundary["parallelism_two_or_four_rtl"] is False
    assert boundary["vendor_timing_signoff"] is False
    assert boundary["bitstream_generated"] is False
    assert boundary["board_measured"] is False
