from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from cnn_fpga.benchmark.unified_comparator_runner import (
    COMMON_TRACE_METHODS,
    CNN_MODEL,
    CNN_TEST_SPLIT,
    DEFAULT_ARTIFACT,
    DEFAULT_SOURCE_DATA,
    HMM_CHECKPOINT,
    RunnerConfig,
    _load_hmm,
    derive_method_costs,
    materialize_qualification_trace,
    paired_residuals_from_packets,
    recompute_gates,
    verify_report,
)
from cnn_fpga.model.tiny_cnn import predict_from_artifact
from cnn_fpga.runtime.unified_execution_contract import (
    ContractViolation,
    MatchedBudget,
    validate_observed_mapping_for_deployable,
)


def _sha256_array(value: np.ndarray, dtype: str) -> str:
    return hashlib.sha256(np.asarray(value, dtype=dtype).tobytes()).hexdigest()


def test_trace_uses_exact_packet_cadence_and_every_adapter_rebuilds_same_pairs() -> None:
    config = RunnerConfig()
    observed, truth = materialize_qualification_trace(config)
    assert len(observed.windows) == len(truth.windows) == 16
    assert observed.trace_id == truth.trace_id
    assert observed.calibration_residuals.shape == (
        config.calibration_windows * config.pairs_per_parameter_window,
        2,
    )
    for window in observed.windows:
        assert len(window.packets) == 2048
        assert window.packets[-1].cycle_index - window.packets[0].cycle_index == 2047
        arrays = [
            paired_residuals_from_packets(window.packets, method_id=method_id)
            for method_id in COMMON_TRACE_METHODS
        ]
        assert arrays[0].shape == (1024, 2)
        assert all(np.array_equal(arrays[0], item) for item in arrays[1:])


def test_packet_bridge_rejects_reordering_missing_phase_nonadjacency_and_truth() -> None:
    observed, _ = materialize_qualification_trace()
    packets = observed.windows[0].packets
    with pytest.raises(ContractViolation, match="phase_pair_order_mismatch"):
        paired_residuals_from_packets(
            (packets[1], packets[0], *packets[2:]), method_id="static_joint_map"
        )
    with pytest.raises(ContractViolation, match="incomplete_phase_pair"):
        paired_residuals_from_packets(packets[:-1], method_id="static_joint_map")
    with pytest.raises(ContractViolation, match="phase_pair_not_adjacent"):
        paired_residuals_from_packets(
            (packets[0], replace(packets[1], cycle_index=999999), *packets[2:]),
            method_id="static_joint_map",
        )
    with pytest.raises(ContractViolation, match="hidden_truth_key_rejected"):
        validate_observed_mapping_for_deployable(
            "kalman_adaptive_map", {**asdict(packets[0]), "logical_truth": 0}
        )


def test_costs_are_dimension_derived_and_route_worst_collision_stays_under_cap() -> None:
    budget = MatchedBudget()
    costs = derive_method_costs()
    assert tuple(costs) == COMMON_TRACE_METHODS
    assert costs["kalman_adaptive_map"].update_macs == 7121
    assert costs["proposed_route_a"].update_macs == 8047
    assert costs["proposed_route_a"].private_model_state_bytes == 5468
    for cost in costs.values():
        assert cost.update_macs <= budget.max_algorithm_macs_per_parameter_update
        assert cost.private_model_state_bytes <= budget.max_private_model_state_bytes
        assert cost.transient_workspace_bytes <= budget.max_transient_workspace_bytes


def test_exported_hmm_is_hash_bound_to_selected_t411_checkpoint() -> None:
    model, temperature, payload = _load_hmm()
    assert HMM_CHECKPOINT.exists()
    assert payload["selected_family_from_validation"] == "gaussian_hmm"
    assert model.parameter_count == 896
    assert model.macs_per_update_proxy == 800
    assert temperature > 0.0
    source = Path(payload["source_checkpoint"])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == payload["source_checkpoint_sha256"]


def test_legacy_cnn_witness_reproduces_real_checkpoint_output_without_labels() -> None:
    report = json.loads(DEFAULT_ARTIFACT.read_text(encoding="utf-8"))
    witness = report["ablation_table"][0]
    with np.load(CNN_TEST_SPLIT, allow_pickle=True) as split:
        histograms = np.asarray(
            split["histograms"][: witness["witness_samples"]], dtype=np.float32
        )
    prediction = np.asarray(predict_from_artifact(CNN_MODEL, histograms), dtype=np.float64)
    assert _sha256_array(histograms, "<f4") == witness["witness_input_sha256"]
    assert _sha256_array(prediction, "<f8") == witness["witness_prediction_sha256"]
    assert witness["adapter_consumed_keys"] == ["histograms"]
    assert witness["labels_or_target_consumed_online"] is False
    assert witness["macs_per_inference"] == 3_489_984
    assert witness["private_model_state_bytes"] == 1_586_368
    assert witness["ranking_status"].startswith("ablation_only")


def test_committed_report_has_separate_tables_real_outputs_and_recomputable_gates() -> None:
    report = json.loads(DEFAULT_ARTIFACT.read_text(encoding="utf-8"))
    verify_report(report)
    assert report["gate_summary"] == {"all_passed": True, "passed": 18, "total": 18}
    deployable = report["deployable_common_trace_table"]
    assert tuple(row["method_id"] for row in deployable) == COMMON_TRACE_METHODS
    assert len({row["trace_sha256"] for row in deployable}) == 1
    assert len({row["decision_sha256"] for row in deployable}) > 3
    assert report["oracle_table"]["included_in_deployable_ranking"] is False
    assert report["oracle_table"]["accounting"] is None
    assert report["qualification_metrics"]["ranking_prohibited"] is True
    assert recompute_gates(report) == report["gates"]
    assert len(DEFAULT_SOURCE_DATA.read_text(encoding="utf-8").splitlines()) > 100


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("deployable_common_trace_table", 5, "ranking_status"), "formal_winner"),
        (("deployable_common_trace_table", 0, "accounting", 0, "contract_passed"), False),
        (("ablation_table", 0, "matched_budget_conforms"), True),
        (("oracle_table", "included_in_deployable_ranking"), True),
        (("prefix_causality_audit", "rows", 0, "prefix_equal"), False),
        (("periodic_feature_grid_equivalence", "max_absolute_complex_error"), 1.0),
        (("mutation_audit", 0, "rejected"), False),
        (("adapter_bindings", 0, "t6_5_2_manifest_mutated"), True),
        (("gate_summary", "total"), 999),
    ),
)
def test_semantic_mutations_fail_verification(path: tuple[object, ...], value: object) -> None:
    report = json.loads(DEFAULT_ARTIFACT.read_text(encoding="utf-8"))
    mutated = deepcopy(report)
    target: object = mutated
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]
    with pytest.raises(ValueError):
        verify_report(mutated, verify_sources=False)


def test_prefix_audit_is_nonvacuous_for_data_dependent_methods() -> None:
    report = json.loads(DEFAULT_ARTIFACT.read_text(encoding="utf-8"))
    rows = {row["method_id"]: row for row in report["prefix_causality_audit"]["rows"]}
    assert all(row["prefix_equal"] and row["future_input_changed"] for row in rows.values())
    assert rows["standard_binning"]["future_decision_changed"] is False
    for method_id in set(COMMON_TRACE_METHODS) - {"standard_binning"}:
        assert rows[method_id]["future_decision_changed"] is True


def test_standard_lut_and_periodic_frontend_equivalences_are_exhaustive() -> None:
    report = json.loads(DEFAULT_ARTIFACT.read_text(encoding="utf-8"))
    standard = report["standard_lut_equivalence"]
    periodic = report["periodic_feature_grid_equivalence"]
    assert standard["phase_code_cases"] == 2048
    assert standard["mismatches"] == 0
    assert periodic["exhaustive_qp_pairs"] == 1024**2
    assert periodic["checked_complex_product_identities"] == 2 * 1024**2
    assert periodic["max_absolute_complex_error"] <= periodic["tolerance"]
