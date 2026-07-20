from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from cnn_fpga.benchmark.logical_channel_reconstruction import (
    CONTRACT_ID,
    DEFAULT_ARTIFACT,
    DEFAULT_SOURCE_DATA,
    LogicalChannelBenchmarkConfig,
    implementation_sha256,
    run_benchmark,
    validate_artifact_payload,
)
from physics.fock_logical_channel import AXIS_LABELS, STATE_LABELS


@pytest.fixture(scope="module")
def small_payload() -> dict:
    return run_benchmark(
        LogicalChannelBenchmarkConfig(
            full_cycles=3, cutoffs=(12, 24, 36, 40), device="cpu"
        )
    )


def test_small_campaign_is_complete_and_all_semantic_gates_pass(small_payload: dict) -> None:
    assert small_payload["task_id"] == "T5.3.1"
    assert small_payload["contract_id"] == CONTRACT_ID
    assert small_payload["status"] == "PASS"
    assert len(small_payload["gates"]) == 26
    assert all(small_payload["gates"].values())
    assert len(small_payload["lanes"]) == 24
    assert len(small_payload["matched_comparisons"]) == 12
    assert len(small_payload["cutoff_diagnostics"]) == 18
    assert validate_artifact_payload(small_payload) == small_payload["gates"]


def test_each_lane_contains_six_states_full_ptm_and_raw_lifetimes(small_payload: dict) -> None:
    for lane in small_payload["lanes"].values():
        assert lane["state_labels"] == list(STATE_LABELS)
        assert np.asarray(lane["projected_output_real"]).shape == (4, 6, 2, 2)
        assert np.asarray(lane["projected_output_imag"]).shape == (4, 6, 2, 2)
        assert np.asarray(lane["survival"]).shape == (4, 6)
        assert len(lane["tomography"]) == 4
        assert all(np.asarray(point["ptm"]).shape == (4, 4) for point in lane["tomography"])
        assert set(lane["pauli_lifetimes"]) == set(AXIS_LABELS)
        for metric in lane["pauli_lifetimes"].values():
            assert metric["e_fold_status"] in {"observed", "right_censored"}
            assert "no exponential fit or postselection" in metric["definition"]
            assert metric["truncated_signed_area_us"] == pytest.approx(
                10.0 * metric["truncated_signed_area_cycles"], abs=2.0e-9
            )


def test_on_off_pairs_are_exactly_matched_except_mode(small_payload: dict) -> None:
    for comparison in small_payload["matched_comparisons"]:
        cutoff = comparison["cutoff"]
        noise = comparison["noise_profile"]
        on = small_payload["lanes"][f"cutoff{cutoff}:{noise}:qec_on"]["config"]
        off = small_payload["lanes"][f"cutoff{cutoff}:{noise}:qec_off"]["config"]
        on_without_mode = {key: value for key, value in on.items() if key != "mode"}
        off_without_mode = {key: value for key, value in off.items() if key != "mode"}
        assert on_without_mode == off_without_mode
        assert comparison["performance_direction_required"] is False


def test_pass_does_not_hide_negative_qec_result(small_payload: dict) -> None:
    signed_differences = [
        axis["qec_on_minus_off_area_cycles"]
        for comparison in small_payload["matched_comparisons"]
        for axis in comparison["axes"].values()
    ]
    assert any(value < 0.0 for value in signed_differences)
    assert small_payload["gates"]["no_desired_qec_performance_direction_is_required"]


@pytest.mark.parametrize(
    "mutation",
    [
        "cycle_duration",
        "postselection",
        "hardcode_nonpauli_zero",
        "stale_parent",
        "promote_hardware",
        "drop_state",
        "tamper_ptm",
        "tamper_lifetime",
        "tamper_comparison",
    ],
)
def test_semantic_mutations_cannot_keep_stored_pass(
    small_payload: dict, mutation: str
) -> None:
    bad = copy.deepcopy(small_payload)
    lane = next(iter(bad["lanes"].values()))
    if mutation == "cycle_duration":
        lane["config"]["cycle_duration_us"] = 9.0
    elif mutation == "postselection":
        lane["event_accounting"]["discarded_trajectories"] = 1
    elif mutation == "hardcode_nonpauli_zero":
        for candidate in bad["lanes"].values():
            for point in candidate["tomography"]:
                point["off_diagonal_pauli_norm"] = 0.0
                point["state_dependent_survival_norm"] = 0.0
    elif mutation == "stale_parent":
        bad["parent_audit"]["implementation_hash_matches"] = False
    elif mutation == "promote_hardware":
        bad["claim_boundary"]["target_hardware_measured"] = True
    elif mutation == "drop_state":
        lane["state_labels"] = lane["state_labels"][:-1]
    elif mutation == "tamper_ptm":
        lane["tomography"][1]["ptm"][1][1] += 0.1
    elif mutation == "tamper_lifetime":
        lane["pauli_lifetimes"]["X"]["truncated_signed_area_cycles"] += 0.1
    elif mutation == "tamper_comparison":
        bad["matched_comparisons"][0]["axes"]["X"]["qec_on_minus_off_area_cycles"] += 0.1
    with pytest.raises(ValueError, match="stored gates"):
        validate_artifact_payload(bad)


def test_gate_and_status_rewrites_are_recomputed_not_trusted(small_payload: dict) -> None:
    bad_gate = copy.deepcopy(small_payload)
    first = next(iter(bad_gate["gates"]))
    bad_gate["gates"][first] = False
    with pytest.raises(ValueError, match="stored gates"):
        validate_artifact_payload(bad_gate)
    bad_status = copy.deepcopy(small_payload)
    bad_status["status"] = "FAIL"
    with pytest.raises(ValueError, match="status"):
        validate_artifact_payload(bad_status)


def test_formal_artifact_source_hash_schema_and_live_implementation() -> None:
    payload = json.loads(DEFAULT_ARTIFACT.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["implementation_sha256"] == implementation_sha256()
    assert payload["config"] == {
        "full_cycles": 30,
        "cutoffs": [12, 24, 36, 40],
        "projector_delta": 0.34,
        "device": "cuda",
        "real_dtype": "float64",
    }
    assert validate_artifact_payload(payload) == payload["gates"]
    source = payload["source_data"]
    assert source["path"] == DEFAULT_SOURCE_DATA.as_posix()
    assert hashlib.sha256(DEFAULT_SOURCE_DATA.read_bytes()).hexdigest() == source["sha256"]
    with DEFAULT_SOURCE_DATA.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == source["row_count"]
    categories = {row["category"] for row in rows}
    assert {
        "contract",
        "parent",
        "state_output",
        "ptm",
        "tomography_diagnostic",
        "pauli_lifetime",
        "matched_comparison",
        "cutoff_diagnostic",
        "gate",
    } <= categories


def test_formal_artifact_is_not_parity_twirl_or_cross_lane_stitching() -> None:
    payload = json.loads(DEFAULT_ARTIFACT.read_text(encoding="utf-8"))
    contract = payload["tomography_contract"]
    assert contract["channel_class"] == "completely_positive_trace_nonincreasing_code_subchannel"
    assert contract["ptm_reconstruction_source"] == "unnormalized_code_space_outputs"
    assert contract["postselection"] is False
    excluded = payload["evidence_routing"]["excluded_heterogeneous_inputs"]
    assert len(excluded) == 3
    assert all(row["status"] == "EXCLUDED" for row in excluded)
    assert all(
        comparison["only_intervention_difference"] == "mode:qec_on_vs_qec_off"
        for comparison in payload["matched_comparisons"]
    )


def test_config_rejects_demo_and_mismatched_formal_profiles() -> None:
    with pytest.raises(ValueError, match=">=3"):
        LogicalChannelBenchmarkConfig(
            full_cycles=2, cutoffs=(12, 24, 36, 40), device="cpu"
        )
    with pytest.raises(ValueError, match="exactly four"):
        LogicalChannelBenchmarkConfig(cutoffs=(6,), device="cpu")
    with pytest.raises(ValueError, match="unique"):
        LogicalChannelBenchmarkConfig(cutoffs=(6, 8, 10, 10), device="cpu")
    with pytest.raises(ValueError, match="strictly increasing"):
        LogicalChannelBenchmarkConfig(cutoffs=(6, 10, 8, 12), device="cpu")
    with pytest.raises(ValueError, match="float64"):
        LogicalChannelBenchmarkConfig(
            cutoffs=(6, 8, 10, 12), device="cpu", real_dtype="float32"
        )
