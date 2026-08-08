from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from cnn_fpga.benchmark.phase6c_preboard_profiles import (
    ATOMIC_BANK_VALIDATION,
    DEFAULT_CONFIG,
    DEFAULT_CSV,
    DEFAULT_JSON,
    DEFAULT_MARKDOWN,
    EQUIVALENCE_REPORT,
    LEARNED_ELIGIBILITY,
    PHASE6B_TERMINAL,
    PREREGISTRATION,
    SYNTHESIS_REPORT,
    recompute_gates,
    verify_report,
)


ROOT = Path(__file__).resolve().parents[1]


def _report() -> dict:
    return json.loads(DEFAULT_JSON.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_formal_report_passes_live_gate_recomputation() -> None:
    report = verify_report(DEFAULT_JSON)
    assert report["status"] == "PASS"
    assert report["gate_summary"] == {"passed": 12, "total": 12, "failed": []}
    assert recompute_gates(report) == report["gates"]


def test_only_static_map_lut_has_hardware_ranking_eligibility() -> None:
    rows = {row["method_id"]: row for row in _report()["hardware_profiles"]}
    assert set(rows) == {
        "CI_if_rtl",
        "static_map_lut_if_rtl",
        "v5_fast_path_if_rtl",
        "eligible_direct_nn_if_rtl",
    }
    assert rows["static_map_lut_if_rtl"]["ranking_eligible_project_preboard"] is True
    for method in ("CI_if_rtl", "v5_fast_path_if_rtl", "eligible_direct_nn_if_rtl"):
        assert rows[method]["ranking_eligible_project_preboard"] is False
        assert rows[method]["core_cycles"] is None
        assert rows[method]["place_route"] is None


def test_static_hardware_row_is_exact_current_rtl_evidence() -> None:
    report = _report()
    row = next(item for item in report["hardware_profiles"] if item["method_id"] == "static_map_lut_if_rtl")
    equivalence = json.loads(EQUIVALENCE_REPORT.read_text(encoding="utf-8"))
    assert row["cxxrtl_action_mismatch_count"] == 0
    assert row["equivalence_map_valid_rows"] == 4316
    assert all(item["exact"] and item["mismatch_count"] == 0 for item in equivalence["scenarios"])
    for binding in equivalence["source_bindings"]:
        assert _sha256(ROOT / binding["path"]) == binding["sha256"]


def test_three_place_route_seeds_retain_resources_and_critical_paths() -> None:
    row = next(item for item in _report()["hardware_profiles"] if item["method_id"] == "static_map_lut_if_rtl")
    assert [item["seed"] for item in row["place_route"]] == [1, 7, 19]
    for item in row["place_route"]:
        assert item["timing_pass"] is True
        assert item["achieved_fmax_mhz"] > 27.0
        assert min(item["lut4_count"], item["ff_count"], item["bram_count"], item["dsp_count"]) > 0
        assert item["critical_path"]["period_ns"] > 0.0
        assert item["critical_path"]["start_cell"].startswith("core.")
    assert "not map-ROM-only area" in row["resource_scope"]


def test_cycle_and_clock_latency_are_exact_but_not_board_measured() -> None:
    row = next(item for item in _report()["hardware_profiles"] if item["method_id"] == "static_map_lut_if_rtl")
    assert row["core_cycles"] == 6
    assert row["initiation_interval_cycles"] == 1
    assert row["source_to_action_ns"] == 6 * 1000 / 27
    assert row["initiation_interval_ns"] == 1000 / 27
    assert row["board_measured_latency_ns"] is None
    assert "NOT_BOARD_MEASURED" in row["evidence_boundary"]


def test_real_host_profiles_have_all_four_stage_distributions() -> None:
    rows = {row["method_id"]: row for row in _report()["host_profiles"]}
    for method in ("Window", "EWMA", "Kalman"):
        row = rows[method]
        assert row["repeats"] == 1000
        assert row["update_macs"] >= 0
        assert row["private_model_state_bytes"] > 0
        assert row["transient_workspace_bytes"] > 0
        for stage in (
            "update",
            "compiler",
            "software_transactional_transfer",
            "software_commit_readback",
        ):
            timing = row[stage]
            assert 0 < timing["minimum_us"] <= timing["p50_us"] <= timing["p95_us"] <= timing["p99_us"] <= timing["worst_us"]
    atomic = json.loads(ATOMIC_BANK_VALIDATION.read_text(encoding="utf-8"))
    assert atomic["status"] == "PASS"
    assert all(row["passed"] for row in atomic["gates"])


def test_v5_host_profile_remains_na_after_early_stop() -> None:
    report = _report()
    row = next(item for item in report["host_profiles"] if item["method_id"] == "V5_if_exists")
    phase6b = json.loads(PHASE6B_TERMINAL.read_text(encoding="utf-8"))
    assert phase6b["execution_path"] == "EARLY_STOP_AT_T6.10.1_HEADROOM_GATE"
    assert phase6b["v5_downstream_outputs_found"] == []
    assert row["eligibility_state"] == "N_A_NO_V5_IMPLEMENTATION_EARLY_STOP"
    assert row["update"] is None


def test_direct_nn_na_is_bound_to_same_task_eligibility_replay() -> None:
    learned = json.loads(LEARNED_ELIGIBILITY.read_text(encoding="utf-8"))
    row = next(item for item in _report()["hardware_profiles"] if item["method_id"] == "eligible_direct_nn_if_rtl")
    assert learned["eligibility_summary"]["same_task_eligible"] == 0
    assert learned["eligibility_summary"]["eligible_replayed"] == 0
    assert row["eligibility_state"] == "N_A_NO_SAME_TASK_ELIGIBLE_DIRECT_NN_RTL"


def test_all_board_power_jitter_deadline_and_physical_transfer_fields_are_null() -> None:
    report = _report()
    fields = report["config"]["must_remain_null_until_t6_9_2"]
    for row in report["hardware_profiles"] + report["host_profiles"]:
        assert all(row[field] is None for field in fields)
    assert report["board_measurement_state"] == "BLOCKED_UNTIL_T6.9.2"
    assert report["cross_table_ranking"] is None
    assert report["global_fastest_claim"] is None


def test_raw_source_data_contains_all_host_repeats_and_hardware_seeds() -> None:
    report = _report()
    with DEFAULT_CSV.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == report["source_data"]["rows"] == 3003
    assert sum(row["row_type"] == "hardware_seed" for row in rows) == 3
    assert sum(row["row_type"] == "host_repeat" for row in rows) == 3000
    for method in ("Window", "EWMA", "Kalman"):
        selected = [row for row in rows if row["method_id"] == method]
        assert len(selected) == 1000
        assert len({int(row["repeat"]) for row in selected}) == 1000
        assert all(row["image_sha256"] == row["readback_sha256"] for row in selected)
    assert report["source_data"]["sha256"] == _sha256(DEFAULT_CSV)


def test_preregistration_and_live_bindings_are_current() -> None:
    report = _report()
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    prereg = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    row = next(item for item in prereg["experiments"] if item["task_id"] == "T6.19.1")
    assert row["config"]["methods"] == config["hardware_methods"]
    assert row["config"]["host_profiles"] == config["host_methods"]
    for binding in report["bindings"].values():
        if isinstance(binding, dict) and "path" in binding:
            path = ROOT / binding["path"]
            assert path.stat().st_size == binding["bytes"]
            assert _sha256(path) == binding["sha256"]
    assert _sha256(SYNTHESIS_REPORT) == report["bindings"]["synthesis_report"]["sha256"]


def test_every_semantic_shortcut_mutation_is_detected() -> None:
    audit = _report()["semantic_mutation_audit"]
    assert audit["count"] == audit["detected"] == 13
    assert all(row["rejected"] for row in audit["cases"])


def test_markdown_keeps_preboard_and_host_boundaries_visible() -> None:
    text = DEFAULT_MARKDOWN.read_text(encoding="utf-8")
    assert "不是板测延迟" in text
    assert "N_A_NO_INDEPENDENT_CI_RTL" in text
    assert "software transfer" in text
    assert "等待 T6.9.2" in text
