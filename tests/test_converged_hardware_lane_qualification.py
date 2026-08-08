from __future__ import annotations

import copy
import csv
import hashlib
import json

import pytest

from cnn_fpga.benchmark import converged_hardware_lane_qualification as subject


@pytest.fixture(scope="module")
def report() -> dict:
    return json.loads(subject.REPORT.read_text(encoding="utf-8"))


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_frozen_target_and_commands_use_exact_converged_top() -> None:
    config = json.loads(subject.CONFIG.read_text(encoding="utf-8"))
    assert config["device"] == subject.DEVICE == "GW2AR-LV18QN88C8/I7"
    assert config["family"] == subject.FAMILY == "GW2A-18C"
    assert config["synthesis_top"] == subject.TOP_MODULE
    assert config["production_top"] == "gkp_route_a_converged_production_top"
    assert config["target_mhz"] == subject.TARGET_MHZ == 27.0
    assert config["seeds"] == list(subject.SEEDS) == [1, 7, 19]
    commands = subject.tool_commands(subject.DEFAULT_BUILD, subject.discover_tools())
    synthesis = " ".join(commands["synthesis"])
    assert f"-top {subject.TOP_MODULE}" in synthesis
    assert subject._relative(subject.PRODUCTION_TOP) in synthesis
    assert all("--freq 27" in " ".join(command) for command in commands["place_route"].values())


def test_wrapper_preserves_one_production_top_without_raw_or_learning_bypass() -> None:
    assert subject._structural_source_audit() == {
        "production_top_instance_count": 1,
        "observable_payload_bits": 922,
        "registered_fold_words": 29,
        "status_signature_lanes": 8,
        "all_high_level_management_inputs_driven": True,
        "all_public_outputs_folded": True,
        "forbidden_raw_child_instantiation_count": 0,
        "learning_module_tokens": 0,
    }


def test_parent_is_exact_live_million_cycle_qualification(report: dict) -> None:
    parent = report["parent_long_qualification"]
    assert parent["verdict"] == subject.LONG_VERDICT
    assert parent["gate_summary"] == {"passed": 19, "total": 19}
    assert parent["cycles"] == 1_000_000
    assert parent["mismatches"] == 0
    assert parent["exact_required_source_bindings_live"] is True


def test_report_recomputes_all_sixteen_gates(report: dict) -> None:
    assert report["verdict"] == subject.VERDICT
    assert report["gate_summary"] == {"passed": 16, "total": 16}
    assert all(row["passed"] for row in report["gates"])
    assert report["gates"][:-1] == subject.evaluate_gates(report)
    assert report["toolchain"]["yosys"].strip()
    assert report["toolchain"]["nextpnr"].strip()


def test_three_independent_seeds_pass_target_and_report_resources(report: dict) -> None:
    routes = report["place_route"]
    assert [row["seed"] for row in routes] == [1, 7, 19]
    assert all(row["route_status"] == "PASS" for row in routes)
    assert all(row["timing_pass"] for row in routes)
    assert min(row["achieved_fmax_mhz"] for row in routes) > subject.TARGET_MHZ
    assert all(row["utilization"]["BSRAM"]["used"] == 8 for row in routes)
    assert all(row["utilization"]["MULT18X18"]["used"] == 1 for row in routes)
    assert all(row["utilization"]["MULT9X9"]["used"] == 1 for row in routes)


def test_fmax_and_resource_statistics_are_losslessly_recomputed(report: dict) -> None:
    routes = report["place_route"]
    fmax = [float(row["achieved_fmax_mhz"]) for row in routes]
    assert report["fmax_mhz"] == {
        "minimum": min(fmax),
        "median": sorted(fmax)[1],
        "maximum": max(fmax),
        "spread": max(fmax) - min(fmax),
    }
    for name in subject.RESOURCE_NAMES:
        used = [int(row["utilization"][name]["used"]) for row in routes]
        summary = report["resource_summary"][name]
        assert summary["minimum"] == min(used)
        assert summary["median"] == sorted(used)[1]
        assert summary["maximum"] == max(used)
        assert summary["maximum"] <= summary["available"]


def test_critical_path_decomposition_is_complete_and_conservatively_labeled(report: dict) -> None:
    paths = report["critical_paths"]
    assert len(paths) == 3
    for row in paths:
        assert row["period_ns"] == pytest.approx(
            row["clock_to_q_ns"] + row["logic_ns"] + row["routing_ns"] + row["setup_ns"],
            abs=1e-9,
        )
        assert row["start_component"] == "production_core"
        assert row["end_component"] == "observability_fold"
        assert row["wrapper_may_dominate"] is True


def test_clock_model_is_six_cycle_ii1_and_not_board_measurement(report: dict) -> None:
    clock = report["clock_model"]
    assert clock["cycles"] == 6
    assert clock["initiation_interval_cycles"] == 1
    assert clock["at_27mhz_ns"] == pytest.approx(6 * 1000 / 27)
    assert clock["at_minimum_fmax_ns"] == pytest.approx(
        6 * 1000 / report["fmax_mhz"]["minimum"]
    )
    assert clock["deadline_miss_count"] is None
    assert clock["jitter_ns"] is None


def test_power_is_only_monotone_analytic_sensitivity(report: dict) -> None:
    power = report["analytic_power_sensitivity"]
    activity = power["dynamic_power_mw_activity_sensitivity"]
    assert activity["low"] < activity["nominal"] < activity["high"]
    assert power["evidence_level"] == "analytic_switching_capacitance_sensitivity_not_vendor_power"
    assert power["static_power_mw"] is None
    assert power["vendor_power_mw"] is None
    assert power["board_measured_power_mw"] is None
    assert all(value is None for value in report["measured_fields"].values())


def test_netlist_and_every_durable_artifact_are_live_hash_bound(report: dict) -> None:
    assert report["structural_netlist"]["top_cell_count"] > 1000
    assert report["structural_netlist"]["cell_type_counts"]["SDPX9B"] == 8
    for binding in [*report["source_bindings"], *report["durable_artifacts"]]:
        path = subject.ROOT / binding["path"]
        assert path.stat().st_size == binding["bytes"] > 0
        assert _sha256(path) == binding["sha256"]


def test_source_data_has_one_lossless_row_per_seed(report: dict) -> None:
    with subject.SOURCE_DATA.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert report["source_data"]["rows"] == len(rows) == 3
    assert [int(row["seed"]) for row in rows] == [1, 7, 19]
    for source, route in zip(rows, report["place_route"], strict=True):
        assert float(source["achieved_fmax_mhz"]) == route["achieved_fmax_mhz"]
        assert int(source["lut4"]) == route["utilization"]["LUT4"]["used"]
        assert int(source["dff"]) == route["utilization"]["DFF"]["used"]
        assert int(source["bsram"]) == route["utilization"]["BSRAM"]["used"]
        assert source["board_measured"] == "False"


def test_all_semantic_mutations_are_independently_rejected(report: dict) -> None:
    audit = subject.semantic_mutation_audit(report)
    assert report["semantic_mutations"] == {"detected": 19, "total": 19}
    assert audit["detected"] == audit["total"] == 19
    assert report["semantic_mutation_results"] == audit["mutations"]
    assert all(row["rejected"] for row in audit["mutations"])


def test_validator_rejects_result_forgery_and_claim_promotion(report: dict) -> None:
    candidate = copy.deepcopy(report)
    candidate["place_route"][0]["timing_pass"] = False
    with pytest.raises(subject.IntegrityError):
        subject._validate(candidate, check_files=False)
    candidate = copy.deepcopy(report)
    candidate["evidence_boundary"]["fastest_or_sota"] = True
    with pytest.raises(subject.IntegrityError):
        subject._validate(candidate, check_files=False)


def test_live_report_verification_is_fail_closed() -> None:
    verified = subject.verify()
    assert verified["verdict"] == subject.VERDICT
    assert verified["gates"] == {"passed": 16, "total": 16}
