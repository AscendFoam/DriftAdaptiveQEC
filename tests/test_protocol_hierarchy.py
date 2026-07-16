from __future__ import annotations

import json
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "docs" / "protocol_hierarchy.json"
MARKDOWN_PATH = ROOT / "docs" / "protocol_hierarchy.md"
PARAMETER_REGISTRY_PATH = ROOT / "docs" / "paper_parameter_registry.json"
LITERATURE_TREND_PATH = ROOT / "docs" / "t5_0_1_literature_trend_reproduction.json"
INDEPENDENT_HOLDOUT_PATH = ROOT / "docs" / "t5_0_2_independent_cross_fidelity_holdout.json"


@pytest.fixture(scope="module")
def registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def _protocols(registry: dict) -> dict[str, dict]:
    return {item["protocol_id"]: item for item in registry["protocols"]}


def test_literature_trend_registry_is_top_level_nonmixing_contract(registry: dict) -> None:
    contract = registry["literature_trend_reproduction_contract"]
    assert contract["task_id"] == "T5.0.1"
    assert contract["target_count"] == 14
    assert "not that every row has been reproduced" in contract["pass_semantics"]
    assert "secondary" in contract["nonmixing_rule"]
    assert contract["future_gate"].startswith("T5.0.2")
    payload = json.loads(LITERATURE_TREND_PATH.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert len(payload["targets"]) == contract["target_count"]
    assert ROOT / contract["source_data"] == ROOT / payload["source_data"]["path"]


def test_t5_0_2_holdout_contract_preserves_failed_main_and_secondary_boundary(
    registry: dict,
) -> None:
    contract = registry["independent_cross_fidelity_holdout_contract"]
    artifact = json.loads(INDEPENDENT_HOLDOUT_PATH.read_text(encoding="utf-8"))
    assert contract["task_id"] == "T5.0.2"
    assert contract["formal_squeezing_db"] == [2.5, 10.25, 11.75]
    assert not set(contract["formal_squeezing_db"]) & set(
        contract["excluded_calibration_and_pilot_db"]
    )
    assert contract["main_cross_fidelity_status"].startswith("FAIL_")
    assert contract["secondary_psteane_status"].startswith("PASS_")
    assert "does not promote" in contract["acceptance_semantics"]
    assert "outside sBs main ranking" in contract["nonmixing_rule"]
    assert artifact["status"] == "PASS"
    assert artifact["main_cross_fidelity_holdout"]["status"] == "FAIL"
    assert artifact["secondary_psteane_holdout"]["status"] == "PASS"
    assert artifact["main_cross_fidelity_holdout"]["failed_gates"] == [
        "pooled_effective_holdout_matches_noise_transfer_within_two_sigma"
    ]


def test_hierarchy_has_one_main_one_cross_validation_and_two_secondary(registry: dict) -> None:
    assert registry["task_id"] == "T2.0.1"
    assert registry["status"] == (
        "frozen_contract_with_t2_0_2_through_t2_3_8_and_quadrature_contract_implementation"
    )
    roles = [item["hierarchy_role"] for item in registry["protocols"]]
    assert roles.count("main_digital_twin") == 1
    assert roles.count("primary_cross_validation") == 1
    assert roles.count("secondary_reproduction") == 2
    assert len(registry["protocols"]) == 4


def test_protocol_records_are_complete_unique_and_fail_closed(registry: dict) -> None:
    protocols = registry["protocols"]
    ids = [item["protocol_id"] for item in protocols]
    assert len(ids) == len(set(ids))
    required = {
        "protocol_id",
        "name",
        "hierarchy_role",
        "evidence_class",
        "source_ids",
        "anchor_ids",
        "current_status",
        "mathematical_object",
        "cycle_contract",
        "observation_contract",
        "action_contract",
        "required_future_implementation",
        "explicitly_unmodeled",
        "allowed_use",
        "forbidden_transfers",
        "future_tasks",
    }
    for protocol in protocols:
        extra = set(protocol) - required
        if protocol["protocol_id"] == "PROTO-SBS-MAIN":
            assert extra == {
                "fault_diagnostic_contract",
                "occupancy_correlation_contract",
                "syndrome_stream_contract",
                "control_memory_contract",
                "fast_monte_carlo_contract",
                "finite_squeezing_noise_contract",
                "ancilla_fault_contract",
                    "control_imperfection_contract",
                    "finite_cutoff_fock_contract",
                    "fock_sbs_cycle_contract",
                    "noise_transfer_surrogate_contract",
                    "cross_fidelity_contract",
                    "differentiable_trajectory_contract",
                    "feedback_grape_gradient_contract",
                    "differentiable_training_feasibility_contract",
                    "nmf_directional_ranking_contract",
                    "bounded_residual_rnn_teacher_contract",
                    "teacher_hidden_control_analysis_contract",
                    "low_dimensional_student_contract",
                    "teacher_student_gain_retention_contract",
                    "teacher_student_branch_freeze_contract",
                }
        elif protocol["protocol_id"] == "PROTO-SHARPEN-TRIM-XVAL":
            assert extra == {"ancilla_fault_contract"}
        else:
            assert not extra
        assert required <= set(protocol), protocol["protocol_id"]
        if protocol["protocol_id"] == "PROTO-SBS-MAIN":
            assert protocol["current_status"] == (
                "error_space_through_finite_cutoff_nmf_directional_ranking_"
                "implemented_long_horizon_memory_robustness_and_device_fidelity_blocked"
            )
        elif protocol["protocol_id"] == "PROTO-SHARPEN-TRIM-XVAL":
            assert protocol["current_status"] == (
                "native_ancilla_readout_reset_effective_model_implemented_"
                "fock_and_device_calibration_pending"
            )
        else:
            assert protocol["current_status"] == "contract_only_not_implemented"
        assert protocol["source_ids"] and protocol["anchor_ids"]
        assert protocol["action_contract"]
        assert protocol["required_future_implementation"]
        assert protocol["explicitly_unmodeled"]
        assert protocol["forbidden_transfers"]
        assert protocol["cycle_contract"]["target_hardware_measured_us"] is None


def test_source_evidence_tiers_match_protocol_roles(registry: dict) -> None:
    sources = {item["source_id"]: item for item in registry["sources"]}
    assert len(sources) == len(registry["sources"])
    for source in sources.values():
        assert (ROOT / source["local_path"]).is_file()

    for protocol in registry["protocols"]:
        tiers = {sources[source_id]["evidence_tier"] for source_id in protocol["source_ids"]}
        if protocol["hierarchy_role"] in {"main_digital_twin", "primary_cross_validation"}:
            assert tiers == {"primary_local"}
            assert protocol["evidence_class"] == "primary_local"
        else:
            assert tiers == {"secondary_local"}
            assert protocol["evidence_class"] == "secondary_local"


def test_every_anchor_matches_its_exact_local_source(registry: dict) -> None:
    sources = {item["source_id"]: item for item in registry["sources"]}
    anchors = registry["source_anchors"]
    anchor_ids = [item["anchor_id"] for item in anchors]
    assert len(anchor_ids) == len(set(anchor_ids))

    for anchor in anchors:
        source = sources[anchor["source_id"]]
        lines = (ROOT / source["local_path"]).read_text(encoding="utf-8").splitlines()
        start = anchor["line_start"]
        end = anchor["line_end"]
        assert 1 <= start <= end <= len(lines), anchor["anchor_id"]
        text = "\n".join(lines[start - 1 : end])
        assert anchor["expected_fragment"] in text, anchor["anchor_id"]

    used = {anchor_id for protocol in registry["protocols"] for anchor_id in protocol["anchor_ids"]}
    assert used == set(anchor_ids)


def test_sbs_contract_preserves_rank_cycle_pair_and_frame_semantics(registry: dict) -> None:
    sbs = _protocols(registry)["PROTO-SBS-MAIN"]
    obj = sbs["mathematical_object"]
    cycle = sbs["cycle_contract"]
    obs = sbs["observation_contract"]

    assert obj["full_channel_kraus_rank"] == 4
    assert len(obj["constituent_channels"]) == 2
    assert obj["kraus_labels"] == ["K_gg", "K_ge", "K_eg", "K_ee"]
    assert cycle["cycle_unit"] == "full_X_plus_Z_composite"
    assert cycle["constituent_steps"] == 2
    assert cycle["constituent_reference_us"] == pytest.approx(4.924)
    assert cycle["full_cycle_reference_us"] == pytest.approx(9.848)
    assert cycle["full_cycle_reference_us"] == pytest.approx(
        cycle["constituent_steps"] * cycle["constituent_reference_us"]
    )
    assert obs["full_cycle_observation"] == "ordered_pair"
    assert obs["full_cycle_alphabet"] == ["gg", "ge", "eg", "ee"]
    assert obs["hidden_state_separate_from_observation"] is True
    assert any("Pauli-frame" in action for action in sbs["action_contract"])


def test_sbs_f_is_reset_leakage_class_not_fifth_ideal_kraus_label(registry: dict) -> None:
    sbs = _protocols(registry)["PROTO-SBS-MAIN"]
    obs = sbs["observation_contract"]
    assert obs["experimental_reset_classes"] == ["g", "e", "f"]
    assert "f" not in sbs["mathematical_object"]["kraus_labels"]
    assert len(sbs["mathematical_object"]["kraus_labels"]) == 4
    assert "not to the four ideal" in obs["leakage_rule"]
    assert any("fifth ideal Kraus" in rule for rule in sbs["forbidden_transfers"])


def test_mixed_state_stream_contract_separates_observed_and_truth_lanes(registry: dict) -> None:
    contract = _protocols(registry)["PROTO-SBS-MAIN"]["syndrome_stream_contract"]
    assert "complete per-cycle DriftState" in contract["input"]
    assert contract["causal_order"].startswith("loss attenuation")
    assert "analog/residual" in contract["observed_output"]
    assert "logical Pauli label" in contract["truth_output"]
    assert "never expose hidden regime" in contract["schema_rule"]
    assert contract["implemented_model_scope"] == (
        "protocol_aligned_mixed_state_syndrome_stream_not_device_calibrated"
    )
    assert (ROOT / contract["artifact"]).is_file()


def test_control_memory_contract_is_observed_only_and_matches_runtime_semantics(registry: dict) -> None:
    contract = _protocols(registry)["PROTO-SBS-MAIN"]["control_memory_contract"]
    assert "simulator truth is type-rejected" in contract["input"]
    assert "previous correction" in contract["state"]
    assert "subtract the actual correction command" in contract["residual_update"]
    assert "rollback is rejected" in contract["version_rule"]
    assert "local fallback correction" in contract["deadline_rule"]
    assert contract["implemented_model_scope"] == (
        "observed_only_multiround_control_memory_not_fallback_policy"
    )
    assert (ROOT / contract["artifact"]).is_file()


def test_fast_monte_carlo_contract_uses_cluster_ci_and_target_stratum_weights(registry: dict) -> None:
    contract = _protocols(registry)["PROTO-SBS-MAIN"]["fast_monte_carlo_contract"]
    assert "per-trajectory residual" in contract["execution_shape"]
    assert "cycle logical-error probability" in contract["primary_metric"]
    assert "trajectory-cluster" in contract["confidence_interval"]
    assert "known-probability trajectory mixture" in contract["rare_event_estimand"]
    assert "target mixture weights" in contract["rare_event_rule"]
    assert contract["implemented_model_scope"] == (
        "vectorized_multitrajectory_syndrome_level_monte_carlo_not_device_calibrated"
    )
    artifact = ROOT / contract["production_artifact"]
    assert artifact.is_file()
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["simulated_cycles"] == 1_000_000
    assert payload["performance"]["one_million_cycle_target_met"] is True


def test_finite_squeezing_contract_separates_physical_observation_and_ideal_limit(registry: dict) -> None:
    contract = _protocols(registry)["PROTO-SBS-MAIN"]["finite_squeezing_noise_contract"]
    assert contract["decomposition"].startswith("physical=channel+data_gkp_peak")
    assert "tanh(Delta^2)/2" in contract["peak_variance"]
    assert "not an added Gaussian" in contract["envelope_model"]
    assert "never silently added to physical data truth" in contract["schema_rule"]
    assert "exact zero finite-squeezing excess" in contract["ideal_limit"]
    assert contract["implemented_model_scope"] == (
        "decomposed_finite_squeezing_syndrome_level_effective_model_not_fock_or_device_calibrated"
    )
    artifact = ROOT / contract["production_artifact"]
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["checks"]["analytic_excess_strictly_decreases"] is True
    assert payload["checks"]["ideal_endpoint_exact"] is True
    assert payload["checks"]["broad_finite_squeezing_rate_above_ideal"] is True


def test_sbs_cycle_values_match_the_frozen_parameter_registry(registry: dict) -> None:
    params = json.loads(PARAMETER_REGISTRY_PATH.read_text(encoding="utf-8"))
    parameter = next(item for item in params["parameters"] if item["name"] == "sivak_composite_xz_cycle")
    sbs_cycle = _protocols(registry)["PROTO-SBS-MAIN"]["cycle_contract"]
    assert parameter["classification"] == "literature_fact"
    assert sbs_cycle["constituent_reference_us"] == parameter["value"]["constituent_step_us"]
    assert sbs_cycle["full_cycle_reference_us"] == parameter["value"]["full_xz_cycle_us"]


def test_sharpen_trim_keeps_native_rounds_observations_and_unknown_timing(registry: dict) -> None:
    xval = _protocols(registry)["PROTO-SHARPEN-TRIM-XVAL"]
    cycle = xval["cycle_contract"]
    obs = xval["observation_contract"]
    assert cycle["cycle_unit"] == "four_round_sharpen_trim_block"
    assert cycle["peak_sharpen_rounds"] == 2
    assert cycle["envelope_trim_rounds"] == 2
    assert cycle["round_reference_us"] is None
    assert cycle["full_cycle_reference_us"] is None
    assert obs["per_round_basis"] == "sigma_y"
    assert obs["per_round_alphabet"] == ["+y", "-y"]
    assert not set(obs["per_round_alphabet"]) & {"gg", "ge", "eg", "ee"}
    assert any("do not assign the SBS" in rule for rule in xval["forbidden_transfers"])


def test_secondary_protocols_cannot_enter_main_or_hardware_claims(registry: dict) -> None:
    protocols = _protocols(registry)
    secondary = [
        protocols["PROTO-KNILL-QUNAUGHT-SECONDARY"],
        protocols["PROTO-ME-PSTEANE-SECONDARY"],
    ]
    for protocol in secondary:
        assert protocol["hierarchy_role"] == "secondary_reproduction"
        assert protocol["evidence_class"] == "secondary_local"
        assert protocol["cycle_contract"]["round_reference_us"] is None
        assert protocol["cycle_contract"]["full_cycle_reference_us"] is None
        rules = " ".join(protocol["forbidden_transfers"])
        assert "do not enter the SBS main ranking" in rules
        assert "FPGA" in rules or "hardware" in rules

    psteane = protocols["PROTO-ME-PSTEANE-SECONDARY"]
    assert any("do not claim FPGA implements physical squeezing" in item for item in psteane["forbidden_transfers"])
    assert "physical squeezing operation" in psteane["explicitly_unmodeled"]


def test_nonmixing_rules_and_promotion_gates_cover_negative_paths(registry: dict) -> None:
    rule_ids = {item["rule_id"] for item in registry["nonmixing_rules"]}
    assert rule_ids == {f"NM-{index:02d}" for index in range(1, 7)}
    rules = " ".join(item["rule"] for item in registry["nonmixing_rules"])
    assert "paired gg/ge/eg/ee" in rules
    assert "fifth ideal Kraus" in rules
    assert "never transfer" in rules
    assert "contract-only" in rules.lower()

    gates = {item["gate_id"]: item for item in registry["promotion_gates"]}
    assert set(gates) == {
        "PG-PRIMARY-SOURCE",
        "PG-PROTOCOL-TEST",
        "PG-MAIN-RANKING",
        "PG-HARDWARE-CLAIM",
    }
    assert "real-board" in gates["PG-HARDWARE-CLAIM"]["pass_condition"]
    assert "primary paper" in gates["PG-PRIMARY-SOURCE"]["pass_condition"]


def test_markdown_and_json_protocol_ids_are_synchronized(registry: dict) -> None:
    markdown = MARKDOWN_PATH.read_text(encoding="utf-8")
    markdown_ids = re.findall(r"<!-- protocol-id: ([A-Z0-9-]+) -->", markdown)
    json_ids = [item["protocol_id"] for item in registry["protocols"]]
    assert len(markdown_ids) == len(set(markdown_ids))
    assert set(markdown_ids) == set(json_ids)
    assert "## 8. 非 demo 审计结论" in markdown
    assert "contract_only_not_implemented" in markdown
    assert "T2.3.7 strict-split directional ranking" in markdown
    assert "axis-resolved cross-fidelity" in markdown


def test_t2_1_1_implementation_update_is_artifact_backed(registry: dict) -> None:
    update = next(item for item in registry["implementation_updates"] if item["task_id"] == "T2.1.1")
    assert update["protocol_id"] == "PROTO-SBS-MAIN"
    assert update["evidence_scope"] == (
        "protocol_aligned_mixed_state_syndrome_stream_not_device_calibrated"
    )
    assert len(update["verified"]) >= 8
    assert any("schema" in item for item in update["verified"])
    assert any("deadlock" in item for item in update["verified"])
    for artifact in update["artifacts"]:
        assert (ROOT / artifact).is_file()


def test_t2_1_2_implementation_update_preserves_unimplemented_policy_boundaries(registry: dict) -> None:
    update = next(item for item in registry["implementation_updates"] if item["task_id"] == "T2.1.2")
    assert update["protocol_id"] == "PROTO-SBS-MAIN"
    assert update["evidence_scope"] == (
        "observed_only_multiround_control_memory_not_fallback_policy"
    )
    assert any("FastLoopEmulator" in item for item in update["verified"])
    assert any("type-rejected" in item for item in update["verified"])
    remaining = " ".join(update["remaining"])
    assert "cycle-slip" in remaining
    assert "CRC" in remaining and "fallback" in remaining
    for artifact in update["artifacts"]:
        assert (ROOT / artifact).is_file()


def test_t2_1_3_implementation_update_is_million_cycle_and_rare_event_backed(registry: dict) -> None:
    update = next(item for item in registry["implementation_updates"] if item["task_id"] == "T2.1.3")
    assert update["protocol_id"] == "PROTO-SBS-MAIN"
    assert update["evidence_scope"] == (
        "vectorized_multitrajectory_syndrome_level_monte_carlo_not_device_calibrated"
    )
    verified = " ".join(update["verified"])
    assert "1,000,000-cycle" in verified
    assert "periodic analytic" in verified
    assert "target-weighted" in verified
    assert "allocation" in verified
    remaining = " ".join(update["remaining"])
    assert "device calibration" in remaining
    assert "target-board" in remaining
    for artifact in update["artifacts"]:
        assert (ROOT / artifact).is_file()


def test_t2_2_1_implementation_update_is_component_and_limit_backed(registry: dict) -> None:
    update = next(item for item in registry["implementation_updates"] if item["task_id"] == "T2.2.1")
    assert update["protocol_id"] == "PROTO-SBS-MAIN"
    assert update["evidence_scope"] == (
        "decomposed_finite_squeezing_syndrome_level_effective_model_not_fock_or_device_calibrated"
    )
    verified = " ".join(update["verified"])
    assert "damped-projector" in verified
    assert "separately auditable" in verified
    assert "periodic analytic" in verified
    assert "exact Delta-zero endpoint" in verified
    remaining = " ".join(update["remaining"])
    assert "Fock-space" in remaining
    assert "squeezing-dB" in remaining
    for artifact in update["artifacts"]:
        assert (ROOT / artifact).is_file()


def test_t2_2_2_implementation_update_is_native_schema_and_limit_backed(
    registry: dict,
) -> None:
    update = next(
        item for item in registry["implementation_updates"] if item["task_id"] == "T2.2.2"
    )
    assert update["protocol_id"] == "PROTO-SBS-MAIN+PROTO-SHARPEN-TRIM-XVAL"
    assert update["evidence_scope"] == (
        "protocol_native_ancilla_error_effective_validation_not_device_calibrated"
    )
    verified = " ".join(update["verified"])
    assert "small-CD, big-CD and readout" in verified
    assert "native four-round +y/-y" in verified
    assert "not silently promoted" in verified
    assert "80,000 SBS cycles and 80,000 sharpen-trim rounds" in verified
    remaining = " ".join(update["remaining"])
    assert "device calibration" in remaining
    assert "Fock/master-equation" in remaining
    for artifact in update["artifacts"]:
        assert (ROOT / artifact).is_file()


def test_t2_2_3_implementation_update_is_causal_quantized_and_limit_backed(
    registry: dict,
) -> None:
    update = next(
        item for item in registry["implementation_updates"] if item["task_id"] == "T2.2.3"
    )
    assert update["protocol_id"] == "PROTO-SBS-MAIN"
    assert update["evidence_scope"] == (
        "causal_control_imperfection_effective_model_not_device_calibrated"
    )
    verified = " ".join(update["verified"])
    assert "Q4.20" in verified
    assert "noncommuting action orders" in verified
    assert "80,000-sample Monte Carlo" in verified
    assert "strictly decreasing RMS" in verified
    remaining = " ".join(update["remaining"])
    assert "target DAC/AWG" in remaining
    assert "hard-real-time" in remaining
    for artifact in update["artifacts"]:
        assert (ROOT / artifact).is_file()


def test_t2_3_1_implementation_update_is_converged_analytic_and_fail_closed(
    registry: dict,
) -> None:
    update = next(
        item for item in registry["implementation_updates"] if item["task_id"] == "T2.3.1"
    )
    assert update["protocol_id"] == "GENERIC-SINGLE-OSCILLATOR-FOCK-REFERENCE"
    assert update["evidence_scope"] == (
        "finite_cutoff_single_oscillator_density_reference_not_protocol_cycle_or_device_model"
    )
    verified = " ".join(update["verified"])
    assert "0.99999961" in verified
    assert "complete sqrt(2) wavefunction" in verified
    assert "pure-loss Kraus" in verified
    assert "thermal" in verified
    assert "positive and complete" in verified
    remaining = " ".join(update["remaining"])
    assert "protocol agnostic" in remaining
    assert "T2.3.2" in remaining
    assert "cavity-transmon" in remaining
    assert "device calibration" in remaining
    for artifact in update["artifacts"]:
        assert (ROOT / artifact).is_file()


def test_t2_3_2_implementation_update_is_formula_completed_and_fail_closed(
    registry: dict,
) -> None:
    update = next(
        item for item in registry["implementation_updates"] if item["task_id"] == "T2.3.2"
    )
    assert update["protocol_id"] == "PROTO-SBS-MAIN"
    assert update["evidence_scope"] == (
        "finite_cutoff_completed_analytic_sbs_one_round_not_pulse_transmon_or_device_model"
    )
    verified = " ".join(update["verified"])
    assert "official arXiv TeX" in verified
    assert "sqrt(2)" in verified
    assert "shared right inverse-square-root completion" in verified
    assert "sixteen hidden-observed branches" in verified
    assert "photon-loss and photon-gain" in verified
    assert "100,000 sampled branches" in verified
    remaining = " ".join(update["remaining"])
    assert "completed axis-resolved cross-fidelity" in remaining
    assert "ancilla/transmon" in remaining
    assert "infinite-dimensional" in remaining
    for artifact in update["artifacts"]:
        assert (ROOT / artifact).is_file()


def test_t2_3_8_implementation_update_is_independent_aligned_and_fail_closed(
    registry: dict,
) -> None:
    update = next(
        item for item in registry["implementation_updates"] if item["task_id"] == "T2.3.8"
    )
    assert update["protocol_id"] == "PROTO-SBS-MAIN"
    assert update["evidence_scope"] == (
        "heisenberg_inspired_decoder_standardized_classical_axis_noise_transfer_"
        "surrogate_not_sbs_or_device_model"
    )
    verified = " ".join(update["verified"])
    assert "independent numerical quadrature" in verified
    assert "Frechet bounds" in verified
    assert "10 and 12 dB" in verified
    assert "clipping-dominated falsification" in verified
    assert "200,000 two-axis Monte Carlo" in verified
    remaining = " ".join(update["remaining"])
    assert "completed axis-resolved T2.3.3" in remaining
    assert "coherent joint-axis" in remaining
    assert "device calibration" in remaining
    for artifact in update["artifacts"]:
        assert (ROOT / artifact).is_file()


def test_t2_3_3_implementation_update_is_four_lane_and_negative_evidence_backed(
    registry: dict,
) -> None:
    update = next(
        item for item in registry["implementation_updates"] if item["task_id"] == "T2.3.3"
    )
    assert update["protocol_id"] == "PROTO-SBS-MAIN"
    assert update["evidence_scope"] == (
        "four_lane_axis_resolved_cross_fidelity_with_verified_quadrature_contract_"
        "not_joint_coherent_process_fidelity"
    )
    verified = " ".join(update["verified"])
    assert "all four common independent-axis" in verified
    assert "3.93e-5" in verified
    assert "cutoffs 24 to 48" in verified
    assert "negative provenance" in verified
    assert "structured attribution" in verified
    remaining = " ".join(update["remaining"])
    assert "coherent joint-axis" in remaining
    assert "infinite-cutoff" in remaining
    for artifact in update["artifacts"]:
        assert (ROOT / artifact).is_file()


def test_inserted_quadrature_contract_is_artifact_backed_and_fail_closed(
    registry: dict,
) -> None:
    update = next(
        item
        for item in registry["implementation_updates"]
        if item["task_id"] == "T-RISK-20260714-01"
    )
    assert update["protocol_id"] == "PROTO-SBS-MAIN"
    assert update["evidence_scope"] == (
        "verified_quadrature_and_axis_resolved_fock_contract_not_device_"
        "calibration_or_joint_coherent_process_fidelity"
    )
    verified = " ".join(update["verified"])
    assert "determinant two" in verified
    assert "diag(sqrt(2),1/sqrt(2))" in verified
    assert "wavefunction Jacobian" in verified
    assert "legacy ambiguous Fourier path" in verified
    remaining = " ".join(update["remaining"])
    assert "source-device envelope" in remaining
    assert "coherent correlated joint-axis" in remaining
    assert "infinite-cutoff" in remaining
    for artifact in update["artifacts"]:
        assert (ROOT / artifact).is_file()


def test_t2_3_4_implementation_update_is_joint_history_and_autograd_backed(
    registry: dict,
) -> None:
    update = next(
        item for item in registry["implementation_updates"] if item["task_id"] == "T2.3.4"
    )
    assert update["protocol_id"] == "PROTO-SBS-MAIN"
    assert update["evidence_scope"] == (
        "finite_cutoff_two_level_joint_cavity_ancilla_differentiable_trajectory_"
        "not_gradient_feasibility_ranking_device_or_hardware_closure"
    )
    verified = " ".join(update["verified"])
    assert "four two-measurement branches normalize" in verified
    assert "independent SciPy" in verified
    assert "cumulative trajectory log-probability" in verified
    assert "no future outcome" in verified
    assert "CPU and CUDA cutoff-eight" in verified
    remaining = " ".join(update["remaining"])
    assert "directional ranking" in remaining
    assert "pulse Hamiltonian" in remaining
    for artifact in update["artifacts"]:
        assert (ROOT / artifact).is_file()


def test_t2_3_5_implementation_update_has_separate_fd_and_random_estimator_evidence(
    registry: dict,
) -> None:
    update = next(
        item for item in registry["implementation_updates"] if item["task_id"] == "T2.3.5"
    )
    assert update["protocol_id"] == "PROTO-SBS-MAIN"
    assert update["evidence_scope"] == (
        "compact_policy_short_horizon_feedback_grape_gradient_validated_not_rnn_"
        "training_feasibility_ranking_device_or_hardware_closure"
    )
    verified = " ".join(update["verified"])
    assert "nonzero reward and score" in verified
    assert "independently match central finite differences" in verified
    assert "four-point finite-difference step sweep" in verified
    assert "12288 samples" in verified
    assert "0.04443" in verified
    remaining = " ".join(update["remaining"])
    assert "GRU/RNN optimization" in remaining
    for artifact in update["artifacts"]:
        assert (ROOT / artifact).is_file()


def test_t2_3_6_implementation_update_has_real_adam_resource_frontiers(
    registry: dict,
) -> None:
    update = next(
        item for item in registry["implementation_updates"] if item["task_id"] == "T2.3.6"
    )
    assert update["protocol_id"] == "PROTO-SBS-MAIN"
    assert update["evidence_scope"] == (
        "current_host_finite_cutoff_two_level_recurrent_adam_kernel_feasibility_"
        "not_training_convergence_ranking_device_or_hardware_closure"
    )
    verified = " ".join(update["verified"])
    assert "72913-parameter" in verified
    assert "fifty-six isolated CUDA points and nine CPU points" in verified
    assert "Adam parameter updates" in verified
    assert "batch 576 exceeds" in verified
    assert "cutoff forty-eight" in verified
    assert "sixty-five points" in verified
    remaining = " ".join(update["remaining"])
    assert "directional ranking" in remaining
    assert "optimization convergence" in remaining
    assert "physical cutoff convergence" in remaining
    for artifact in update["artifacts"]:
        assert (ROOT / artifact).is_file()
    payload = json.loads(
        (ROOT / "docs/t2_3_6_differentiable_sbs_feasibility.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["status"] == "PASS"
    assert payload["measurement_contract"]["optimizer"] == "Adam"
    assert len(payload["points"]) == 65
    assert payload["summary"]["resource_frontier_observed"] is True


def test_t2_3_7_implementation_update_is_strict_split_multiagent_and_fail_closed(
    registry: dict,
) -> None:
    main = _protocols(registry)["PROTO-SBS-MAIN"]
    contract = main["nmf_directional_ranking_contract"]
    assert contract["analysis_contract"].startswith("T237-V3-STRICT-SPLIT")
    assert "five train agent seeds" in contract["split_rule"]
    assert "all five paired agents" in contract["primary_result"]
    assert "cutoff sixteen" in contract["confirmation_result"]
    assert "reset view rises above" in contract["counterevidence"]
    assert "not_paper_exact" in contract["implemented_model_scope"]
    for artifact in contract["production_artifacts"]:
        assert (ROOT / artifact).is_file()

    update = next(
        item for item in registry["implementation_updates"] if item["task_id"] == "T2.3.7"
    )
    assert update["protocol_id"] == "PROTO-SBS-MAIN"
    assert update["evidence_scope"] == (
        "finite_cutoff_two_level_ten_cycle_directional_nmf_ranking_not_paper_"
        "exact_long_horizon_memory_robustness_device_or_hardware_closure"
    )
    verified = " ".join(update["verified"])
    assert "five independently trained" in verified
    assert "train-only warm-up" in verified
    assert "0.0842 to 0.3281" in verified
    assert "counterevidence" in verified
    assert "thirty-one focused" in verified
    remaining = " ".join(update["remaining"])
    assert "one-thousand-cycle six-state" in remaining
    assert "memory-specific robustness" in remaining
    assert "device calibration" in remaining
    for artifact in update["artifacts"]:
        assert (ROOT / artifact).is_file()


def test_t4_4_1_protocol_contract_is_fresh_bounded_and_fail_closed(
    registry: dict,
) -> None:
    main = _protocols(registry)["PROTO-SBS-MAIN"]
    contract = main["bounded_residual_rnn_teacher_contract"]
    assert contract["task_id"] == "T4.4.1"
    assert contract["training_protocol"].startswith("T441-FRESH")
    assert "three fresh 72853-parameter" in contract["architecture_rule"]
    assert "validation only" in contract["split_rule"]
    assert "no parent state dict" in contract["non_reuse_rule"]
    assert "0.253603" in contract["primary_result"]
    assert "0.141557" in contract["confirmation_result"]
    assert "global optimizer convergence is not claimed" in contract["counterevidence"]
    assert "not_hidden_state_student" in contract["implemented_model_scope"]
    for artifact in contract["production_artifacts"]:
        assert (ROOT / artifact).is_file()


def test_t4_4_2_protocol_contract_keeps_analysis_post_hoc_and_leakage_ood(
    registry: dict,
) -> None:
    contract = _protocols(registry)["PROTO-SBS-MAIN"][
        "teacher_hidden_control_analysis_contract"
    ]
    assert contract["task_id"] == "T4.4.2"
    assert contract["analysis_protocol"].startswith("T442-FROZEN")
    assert "zero optimizer steps" in contract["freeze_rule"]
    assert "128 half-cycles" in contract["trajectory_rule"]
    assert "one principal component" in contract["low_dimensional_result"]
    assert "0.667797" in contract["belief_proxy_result"]
    assert "10 and 12 half-cycles" in contract["memory_result"]
    assert "falsifying a universal" in contract["counterevidence"]
    assert "never passed as a third teacher token" in contract["leakage_rule"]
    assert "not_unique_belief_student_gain" in contract["implemented_model_scope"]
    for artifact in contract["production_artifacts"]:
        assert (ROOT / artifact).is_file()


def test_t4_4_3_protocol_contract_is_strict_split_small_and_gain_bounded(
    registry: dict,
) -> None:
    contract = _protocols(registry)["PROTO-SBS-MAIN"][
        "low_dimensional_student_contract"
    ]
    assert contract["task_id"] == "T4.4.3"
    assert contract["training_protocol"].startswith("T443-LOWDIM")
    assert "one two and four-state" in contract["architecture_rule"]
    assert "validation alone selects" in contract["split_rule"]
    assert "four states" in contract["selection_result"]
    assert "6.083136e-6" in contract["selection_result"]
    assert "legacy T4.1.5" in contract["comparison_result"]
    assert "no torch physics or teacher dependency" in contract["online_rule"]
    assert "global optimizer convergence is not claimed" in contract["counterevidence"]
    assert "not_physical_gain_retention" in contract["implemented_model_scope"]
    for artifact in contract["production_artifacts"]:
        assert (ROOT / artifact).is_file()


def test_t4_4_4_protocol_contract_retains_gain_without_universal_mf_claim(
    registry: dict,
) -> None:
    contract = _protocols(registry)["PROTO-SBS-MAIN"][
        "teacher_student_gain_retention_contract"
    ]
    assert contract["task_id"] == "T4.4.4"
    assert contract["evaluation_protocol"].startswith("T444-PHYSICAL")
    assert "eight new cutoff-twelve seeds" in contract["split_rule"]
    assert "all five MF agents" in contract["split_rule"]
    assert "twenty-thousand-repeat paired bootstrap" in contract["retention_rule"]
    assert "minimum confidence lower 0.982442" in contract["retention_result"]
    assert "minimum lower 0.944501" in contract["retention_result"]
    assert "only in a separate exact two-cycle" in contract["oracle_rule"]
    assert "not RTL synthesis or board" in contract["burden_cost_rule"]
    assert "ordering reverses" in contract["counterevidence"]
    assert "not_universal_nmf_over_mf" in contract["implemented_model_scope"]
    for artifact in contract["production_artifacts"]:
        assert (ROOT / artifact).is_file()


def test_t4_4_5_protocol_contract_freezes_qualified_and_fallback_branches(
    registry: dict,
) -> None:
    contract = _protocols(registry)["PROTO-SBS-MAIN"]["teacher_student_branch_freeze_contract"]
    assert contract["task_id"] == "T4.4.5"
    assert contract["decision_protocol"].startswith("T445-FAIL-CLOSED")
    assert "without rerunning or retuning evaluation" in contract["input_rule"]
    assert "all eight" in contract["strong_rule"]
    assert "remove teacher and distillation claims" in contract["fallback_rule"]
    assert "0.557115" in contract["counterevidence"]
    assert "universal NMF-over-MF remains falsified" in contract["counterevidence"]
    assert "leakage OOD long-horizon RTL FPGA board" in contract["claim_rule"]
    assert "T5.2" in contract["revocation_rule"] and "T6" in contract["revocation_rule"]
    assert "not_universal_nmf" in contract["implemented_model_scope"]
    for artifact in contract["production_artifacts"]:
        assert (ROOT / artifact).is_file()


def test_deferred_variants_do_not_silently_replace_main_protocol(registry: dict) -> None:
    variants = {item["variant"]: item for item in registry["deferred_variants"]}
    assert variants["autonomous_sbs"]["scheduled_task"] == "T3.2.8"
    assert "rather than silently replacing" in variants["autonomous_sbs"]["reason"]
    assert variants["multimode_trapped_ion_qec"]["scheduled_task"] == "T7.1.4"
