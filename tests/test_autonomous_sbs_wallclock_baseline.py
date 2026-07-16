from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from cnn_fpga.benchmark.autonomous_sbs_wallclock_baseline import (
    DEFAULT_ARTIFACT,
    DEFAULT_SOURCE_DATA,
    NOISE_PROFILES,
    implementation_sha256,
)
from physics.autonomous_sbs import PAPER_SOURCE

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads((ROOT / DEFAULT_ARTIFACT).read_text(encoding="utf-8"))


def test_artifact_passes_all_non_directional_gates(artifact: dict) -> None:
    assert artifact["task_id"] == "T3.2.8"
    assert artifact["status"] == "PASS"
    assert artifact["required_gates"] == list(artifact["gates"])
    assert all(artifact["gates"].values())
    assert artifact["implementation_sha256"] == implementation_sha256()
    assert artifact["gates"]["no_desired_performance_direction_is_required"] is True


def test_literature_and_source_data_hashes_are_live(artifact: dict) -> None:
    assert artifact["literature"]["source_sha256"] == sha(ROOT / PAPER_SOURCE)
    assert artifact["source_data"]["sha256"] == sha(ROOT / DEFAULT_SOURCE_DATA)
    assert artifact["literature"]["timing_is_literature_simulation_not_target_hardware"] is True


def test_all_noise_cutoff_protocol_lanes_are_present(artifact: dict) -> None:
    assert set(artifact["config"]["cutoffs"]) == {12, 16}
    assert set(artifact["noise_profiles_us"]) == set(NOISE_PROFILES)
    assert len(artifact["lanes"]) == 6
    assert artifact["workload"]["total_full_cycles"] == 1020
    assert artifact["workload"]["deterministic_nonselective_no_monte_carlo_ci"] is True


def test_each_lane_uses_common_700us_but_protocol_native_cycle_count(artifact: dict) -> None:
    for lane in artifact["lanes"].values():
        measurement = lane["measurement_feedback"]
        autonomous = lane["autonomous"]
        assert measurement["event_accounting"]["full_cycles"] == 70
        assert autonomous["event_accounting"]["full_cycles"] == 100
        assert measurement["time_us"][-1] == autonomous["time_us"][-1] == 700.0
        assert len(measurement["time_us"]) == 71
        assert len(autonomous["time_us"]) == 101


def test_event_costs_are_raw_counts_not_hidden_weighted_scalar(artifact: dict) -> None:
    for lane in artifact["lanes"].values():
        m = lane["measurement_feedback"]["event_accounting"]
        a = lane["autonomous"]["event_accounting"]
        assert (m["measurement_events"], a["measurement_events"]) == (140, 0)
        assert (m["reset_events"], a["reset_events"]) == (140, 200)
        assert (m["active_gate_applications"], a["active_gate_applications"]) == (1260, 1800)
        assert lane["comparison"]["measurement_events_avoided_at_common_horizon"] == 140
        assert lane["comparison"]["additional_autonomous_resets_at_common_horizon"] == 60
        assert lane["comparison"]["additional_autonomous_active_gates_at_common_horizon"] == 540


def test_per_cycle_and_per_microsecond_lifetimes_are_both_retained(artifact: dict) -> None:
    for lane in artifact["lanes"].values():
        for mode in ("measurement_feedback", "autonomous"):
            metrics = lane[mode]["metrics"]["logical_z_signal"]
            assert set(metrics) == {
                "normalized_signed_auc",
                "area_equivalent_lifetime_us",
                "area_equivalent_lifetime_protocol_cycles",
                "area_equivalent_lifetime_standard_10us_cycles",
                "horizon_us",
            }
            assert all(np.isfinite(value) for value in metrics.values())
            cycle_us = 10.0 if mode == "measurement_feedback" else 7.0
            assert metrics["area_equivalent_lifetime_us"] == pytest.approx(
                metrics["area_equivalent_lifetime_protocol_cycles"] * cycle_us
            )


def test_cycle_normalization_and_physical_time_reach_opposite_rankings(artifact: dict) -> None:
    comparisons = [lane["comparison"] for lane in artifact["lanes"].values()]
    assert all(item["autonomous_to_measurement_logical_lifetime_protocol_cycle_ratio"] > 1.0 for item in comparisons)
    assert all(item["autonomous_to_measurement_logical_lifetime_us_ratio"] < 1.0 for item in comparisons)

    cutoff_12 = [
        lane
        for lane in artifact["lanes"].values()
        if lane["cutoff"] == 12
    ]
    assert all(lane["comparison"]["autonomous_minus_measurement_final_logical_z"] > 0.0 for lane in cutoff_12)
    assert all(lane["comparison"]["autonomous_to_measurement_logical_lifetime_us_ratio"] < 1.0 for lane in cutoff_12)


def test_source_data_recomputes_every_curve_row_count(artifact: dict) -> None:
    with (ROOT / DEFAULT_SOURCE_DATA).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == artifact["source_data"]["row_count"]
    assert len(rows) > 4000
    assert {row["row_type"] for row in rows} == {"curve", "event_accounting", "comparison"}
    curve_rows = [row for row in rows if row["row_type"] == "curve"]
    expected = 6 * 4 * (71 + 101)
    assert len(curve_rows) == expected


def test_no_claim_promotes_literature_timing_or_autonomous_optimality(artifact: dict) -> None:
    forbidden = artifact["claim_boundary"]["forbidden"]
    assert "trained autonomous optimum" in forbidden
    assert "target-board or device measured timing" in forbidden
    for profile in artifact["timing_profiles"].values():
        assert profile["target_hardware_measured"] is False


def test_method_audits_are_strict(artifact: dict) -> None:
    equivalence = artifact["method_audits"]["nonselective_measurement_equivalence"]
    zero_noise = artifact["method_audits"]["zero_noise_duration_invariance"]
    assert equivalence["maximum_density_difference"] <= 2.0e-12
    assert equivalence["branch_probability_sum"] == pytest.approx(1.0, abs=2.0e-12)
    assert zero_noise["maximum_density_difference"] <= 2.0e-12
