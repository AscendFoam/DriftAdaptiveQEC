from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path

import pytest

from cnn_fpga.benchmark import comparison_metric_ontology as ontology


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "docs" / "t6_16_2_comparison_ontology.json"


def _report() -> dict:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def _ontology() -> dict:
    return _report()["ontology"]


def test_report_recomputes_all_gates() -> None:
    report = _report()
    ontology.verify_report(report)
    assert report["verdict"] == "PASS_FAIL_CLOSED_COMPARISON_ONTOLOGY"
    assert report["gate_summary"] == {"passed": 16, "failed": []}
    assert report["ranking_policy"]["global_score"] == "PROHIBITED"
    assert report["contract_self_tests"]["global_score_api"] is None


def test_six_lanes_and_required_metrics_are_exactly_typed() -> None:
    payload = _ontology()
    assert {row["lane_id"] for row in payload["lanes"]} == ontology.LANE_IDS
    assert len(payload["lanes"]) == 6
    metrics = {row["metric_id"]: row for row in payload["metrics"]}
    assert len(metrics) == 46
    assert ontology.REQUIRED_METRICS <= set(metrics)
    assert metrics["p_Y"]["conversion_policy"] == "never infer from pX+pZ"
    assert metrics["squeezing_threshold_db"]["direction"] == "lower_required_squeezing_is_better"
    assert metrics["noise_threshold_sigma"]["direction"] == "higher_tolerable_noise_is_better"
    assert metrics["adaptation_lag_epochs"]["conversion_policy"] == "never convert to inference microseconds without epoch workload"


def test_value_states_preserve_null_na_failed_and_negative() -> None:
    states = _ontology()["value_states"]
    assert set(states) == ontology.VALUE_STATES
    for state in ("NULL_NOT_REPORTED", "N_A_NOT_APPLICABLE", "FAILED", "NEGATIVE"):
        assert states[state]["value_required"] is False
        assert states[state]["ranking_allowed"] is False
        row = ontology._observation(observation_id=state, state=state, value=None)
        assert ontology.validate_observation(row, _ontology()) == []
        row["value"] = 0.0
        assert "null_na_failed_negative_must_not_have_value" in ontology.validate_observation(row, _ontology())


def test_value_state_must_match_task_signature_evidence_level() -> None:
    payload = _ontology()
    estimate = ontology._observation(observation_id="estimate", state="ESTIMATE_VALUE", value=1.0)
    assert ontology.validate_observation(estimate, payload) == []
    estimate["task_signature"]["evidence_level"] = "measured_hardware"
    assert "value_state_evidence_level_mismatch" in ontology.validate_observation(estimate, payload)

    measured = ontology._observation(observation_id="measured", state="MEASURED_VALUE", value=1.0)
    estimate = ontology._observation(observation_id="estimate2", state="ESTIMATE_VALUE", value=1.0)
    reasons = ontology.compare_observations(measured, estimate, payload)
    assert "mismatch:task_signature" in reasons


def test_wrong_lane_and_incomplete_signature_fail_closed() -> None:
    payload = _ontology()
    row = ontology._observation(observation_id="wrong", lane_id="aqec_wallclock", metric_id="p_L")
    assert "wrong_lane" in ontology.validate_observation(row, payload)
    row = ontology._observation(observation_id="signature")
    row["task_signature"].pop("precision")
    assert "task_signature_incomplete" in ontology.validate_observation(row, payload)


def test_comparison_requires_same_denominator_signature_and_timing_boundary() -> None:
    payload = _ontology()
    left = ontology._observation(observation_id="left")
    right = deepcopy(left)
    right["observation_id"] = "right"
    assert ontology.compare_observations(left, right, payload) == []
    right["denominator"] = "different denominator"
    assert "mismatch:denominator" in ontology.compare_observations(left, right, payload)
    right = deepcopy(left)
    right["task_signature"]["noise_model"] = "different noise"
    assert "mismatch:task_signature" in ontology.compare_observations(left, right, payload)

    core = ontology._observation(
        observation_id="core", lane_id="fpga_implementation", metric_id="latency_ns",
        value=124.0, unit="ns", denominator="one action", statistic="deterministic",
        timing_boundary="decoder_core",
    )
    loop = deepcopy(core)
    loop.update(observation_id="loop", value=550.0, timing_boundary="closed_loop")
    assert "mismatch:timing_boundary" in ontology.compare_observations(core, loop, payload)


def test_timing_and_throughput_boundaries_are_not_conflated() -> None:
    payload = _ontology()
    boundaries = {row["boundary_id"]: row for row in payload["timing_boundaries"]}
    assert set(boundaries) == ontology.TIMING_BOUNDARIES
    assert boundaries["source_to_action"]["composable_from"] == ["transport", "decoder_core"]
    assert boundaries["closed_loop"]["composable_from"] == ["source_to_action"]
    metrics = {row["metric_id"]: row for row in payload["metrics"]}
    assert metrics["latency_ns"]["family"] == "timing"
    assert metrics["initiation_interval_ns"]["family"] == "throughput"
    assert "not end-to-end latency" in metrics["initiation_interval_ns"]["conversion_policy"]


def test_resource_qualifiers_prevent_cross_device_primitive_ranking() -> None:
    resources = {row["resource_id"]: row for row in _ontology()["resource_dimensions"]}
    assert set(resources) == ontology.RESOURCE_IDS
    assert set(resources["BRAM"]["required_qualifiers"]) == {"device", "primitive_bits", "tool", "stage", "seed/profile"}
    assert set(resources["power"]["required_qualifiers"]) == {"device", "voltage", "clock", "activity", "method", "stage"}
    assert resources["power"]["allowed_evidence"] == ["analytic_sensitivity", "tool_estimate", "board_measured"]


def test_qualitative_complexity_requires_numeric_or_asymptotic_locator() -> None:
    payload = _ontology()
    valid = ontology._observation(
        observation_id="complexity", lane_id="multimode_structured_lattice_cpd",
        metric_id="asymptotic_time_complexity", value="O(n)", unit="big_O",
        denominator="structured repetition-rectangular lattice", statistic="source theorem",
        qualitative_complexity={"claim":"linear", "basis":"source_asymptotic", "source_locator":"Lin 2023 theorem"},
    )
    assert ontology.validate_observation(valid, payload) == []
    invalid = deepcopy(valid)
    invalid["qualitative_complexity"] = {"claim":"low", "basis":"opinion", "source_locator":None}
    assert "qualitative_complexity_unsupported" in ontology.validate_observation(invalid, payload)


def test_t6_16_1_metric_crosswalk_is_complete_and_rejects_only_undefined_denominator() -> None:
    rows = _report()["source_metric_crosswalk"]
    assert len(rows) == len(ontology.SOURCE_METRIC_CROSSWALK) == 30
    assert sum(row["status"] == "MAPPED" for row in rows) == 29
    rejected = [row for row in rows if row["status"] == "REJECTED_UNDEFINED_DENOMINATOR"]
    assert len(rejected) == 1
    assert rejected[0]["method_id"] == "wang_direct_nn_surface_gkp"
    assert rejected[0]["source_metric_id"] == "decoding_rate_improvement"
    assert rejected[0]["ontology_metric_id"] is None


def test_source_data_bindings_and_mutations_are_complete() -> None:
    report = _report()
    csv_path = ROOT / report["source_data"]["path"]
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == report["source_data"]["rows"] == 101
    assert {row["record_type"] for row in rows} == {"lane", "metric", "timing_boundary", "resource", "value_state", "source_crosswalk"}
    for binding in report["bindings"].values():
        path = ROOT / binding["path"]
        assert path.exists()
        assert ontology._sha256(path) == binding["sha256"]
    mutations = report["semantic_mutation_audit"]
    assert mutations["count"] == mutations["detected"] == 16
    assert {row["target_gate"] for row in mutations["cases"]} == set(report["gates"])
    assert all(row["rejected"] for row in mutations["cases"])


def test_forged_global_score_or_rankable_null_invalidates_report() -> None:
    report = _report()
    forged = deepcopy(report)
    forged["ranking_policy"]["global_score"] = "WEIGHTED_SUM"
    with pytest.raises(ValueError, match="gates"):
        ontology.verify_report(forged)
    forged = deepcopy(report)
    forged["ontology"]["value_states"]["NULL_NOT_REPORTED"]["ranking_allowed"] = True
    with pytest.raises(ValueError, match="gates"):
        ontology.verify_report(forged)
