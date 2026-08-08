from __future__ import annotations

import copy
import csv
import json

from cnn_fpga.benchmark import phase6d_manuscript_delta as contract


def test_report_passes_all_gates_and_mutations() -> None:
    report = contract.build_report()
    assert report["verdict"] == contract.VERDICT
    assert report["gate_summary"] == {"passed": 27, "total": 27}
    audit = report["semantic_mutation_audit"]
    assert audit["detected"] == audit["total"] == 27


def test_final_claims_are_exact_and_complete() -> None:
    report = contract.build_report()
    expected = sorted(report["config_snapshot"]["required_claim_ids"])
    assert report["parent_state"]["claim_ids"] == expected
    assert len(report["parent_state"]["final_claims"]) == 10


def test_multimode_no_go_is_exact_and_downstream_unopened() -> None:
    mm = contract.build_report()["parent_state"]["multimode"]
    assert mm["strongest_baseline"] == "static_mixture_exact_mld"
    assert mm["baseline_p_L"] == mm["proposed_p_L"] == 0.11197916666666667
    assert mm["relative_improvement_point"] == mm["relative_improvement_lcb"] == 0.0
    assert mm["formal_or_pilot_accessed"] is False


def test_rtl_positive_is_preboard_and_not_multimode() -> None:
    rtl = contract.build_report()["parent_state"]["rtl"]
    assert (rtl["latency_cycles"], rtl["ii_cycles"]) == (6, 1)
    assert rtl["cycles"] == 1_000_000
    assert rtl["mismatches"] == 0
    assert rtl["ii1_input_pairs"] == rtl["ii1_output_pairs"] == 998_435
    assert rtl["board_measured"] is False
    assert rtl["multimode_decoder_in_rtl"] is False
    assert all(value is None for value in rtl["measured_fields"].values())


def test_every_manuscript_section_contract_is_present() -> None:
    manuscript = contract.build_report()["manuscript"]
    assert manuscript["section_order_exact"]
    assert all(manuscript["section_checks"].values())
    assert all(manuscript["canonical_terms_present"].values())


def test_citations_and_figures_resolve() -> None:
    manuscript = contract.build_report()["manuscript"]
    assert manuscript["required_citations_present"]
    assert manuscript["all_citations_resolved"]
    assert all(all((item["exists"], item["included"], item["label_present"])) for item in manuscript["figures"].values())


def test_cross_lane_and_learning_rescue_mutations_fail_closed() -> None:
    report = contract.build_report()
    cross_lane = copy.deepcopy(report)
    cross_lane["parent_state"]["nontransfer"]["global_weighted_score"] = 0.5
    assert not contract.evaluate_gates(cross_lane)["G21_nontransferability_explicit"]
    learning = copy.deepcopy(report)
    learning["parent_state"]["learning"]["primary"] = True
    assert not contract.evaluate_gates(learning)["G23_learning_dropped_only"]


def test_board_null_and_headroom_mutations_fail_closed() -> None:
    report = contract.build_report()
    board = copy.deepcopy(report)
    board["parent_state"]["rtl"]["measured_fields"]["board_power_mw"] = 1.0
    assert not contract.evaluate_gates(board)["G22_board_values_are_null"]
    headroom = copy.deepcopy(report)
    headroom["exact_parent_number_checks"]["mm_zero_headroom"] = False
    assert not contract.evaluate_gates(headroom)["G20_parent_numbers_exact"]


def test_source_data_is_lossless_and_matches_report() -> None:
    report = contract.build_report()
    assert len(report["source_data"]) == 57
    assert contract._rows_lossless(report["source_data"])
    contract.write_outputs(report)
    with contract.DEFAULT_SOURCE_DATA.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows == report["source_data"]


def test_task_board_binding_uses_narrow_semantic_projection() -> None:
    board = contract.BOARD_PATH.read_text(encoding="utf-8")
    projection = contract._task_board_projection(board)
    assert set(projection["tasks"]) == set(contract.BOARD_PROJECTION_TASKS)
    assert projection["tasks"]["T7.2.6"] == "Done"
    assert contract._task_board_projection(board + "\n| unrelated future log row |\n") == projection

    mutated = board.replace(
        "| T7.2.6 | Done |",
        "| T7.2.6 | Blocked |",
        1,
    )
    assert contract._task_board_projection(mutated) != projection


def test_generated_outputs_verify() -> None:
    contract.write_outputs(contract.build_report())
    ok, checks = contract.verify_report()
    assert ok, checks
    assert all(checks.values())
    stored = json.loads(contract.DEFAULT_REPORT.read_text(encoding="utf-8"))
    assert stored["verdict"] == contract.VERDICT


def test_compiled_pdf_and_visual_qa_are_hash_bound() -> None:
    qa = contract.build_report()["visual_qa"]
    assert qa["pdf_hash_live"]
    assert qa["tex_hash_live"]
    assert qa["log_hash_live"]
    assert qa["page_count_exact"]
    assert qa["zero_log_diagnostics"]
    assert qa["manual_pass"]
    assert qa["raster_bounds"]
    assert qa["text_scan_pass"]
