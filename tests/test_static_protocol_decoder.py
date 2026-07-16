from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from cnn_fpga.benchmark.standard_binning_baseline import major_comparison_registry
from cnn_fpga.benchmark.static_protocol_decoder import (
    FALLBACK_ACTION,
    IDEAL_BRANCH_REFERENCE_ID,
    STANDARD_SBS_BRANCH_ID,
    STATIC_SBS_DECODER_DESCRIPTOR,
    STATIC_SBS_DECODER_ID,
    STATIC_SBS_MAP_ID,
    StaticSBSBranchDecoder,
    build_static_sbs_parameters,
    score_sbs_branch_action,
    standard_observed_sbs_action,
    static_protocol_validation_scenarios,
    validate_static_protocol_comparison_registration,
)
from physics.sbs_error_space import SBS_OUTCOMES
from physics.sbs_observation_reset import (
    PairedSyndrome,
    make_persistent_leakage_model,
)


ROOT = Path(__file__).resolve().parents[1]
JSON_ARTIFACT = ROOT / "docs" / "t3_1_4_static_protocol_decoder_validation.json"
CSV_ARTIFACT = ROOT / "docs" / "t3_1_4_static_protocol_decoder_source_data.csv"


def _parameters(scenario_index: int = 0):
    scenario = static_protocol_validation_scenarios()[scenario_index]
    return build_static_sbs_parameters(
        scenario.make_model(),
        ideal_branch_prior=scenario.ideal_branch_prior,
        fallback_cost=scenario.fallback_cost,
        calibration_protocol_id=f"test/{scenario.scenario_id}",
    )


def test_descriptor_is_static_deployable_and_has_no_truth_input() -> None:
    descriptor = STATIC_SBS_DECODER_DESCRIPTOR

    assert descriptor.baseline_id == STATIC_SBS_DECODER_ID
    assert descriptor.deployable
    assert descriptor.hidden_truth_online_inputs == ()
    assert not descriptor.update_during_evaluation
    assert descriptor.action_space == (*SBS_OUTCOMES, FALLBACK_ACTION)
    assert "not_full_quantum_recovery" in descriptor.evidence_scope


def test_exact_stationary_likelihood_and_bayes_posterior_are_normalized() -> None:
    parameters = _parameters()
    prior = np.asarray(parameters.ideal_branch_prior)
    likelihood = parameters.likelihood_array()
    evidence = np.asarray(parameters.observation_probability)
    posterior = parameters.posterior_array()

    np.testing.assert_allclose(likelihood.sum(axis=1), 1.0, atol=2.0e-15)
    np.testing.assert_allclose(evidence.sum(), 1.0, atol=2.0e-15)
    np.testing.assert_allclose(posterior.sum(axis=1), 1.0, atol=2.0e-15)
    np.testing.assert_allclose(
        posterior,
        (prior[:, None] * likelihood / evidence[None, :]).T,
        atol=2.0e-15,
    )
    assert all(value > 0.0 for value in parameters.stationary_carry_prior)


def test_decoder_minimizes_registered_action_loss_for_every_observation() -> None:
    parameters = _parameters(3)
    decoder = StaticSBSBranchDecoder(parameters)

    for x_value in ("g", "e", "leakage"):
        for z_value in ("g", "e", "leakage"):
            observation = PairedSyndrome(x=x_value, z=z_value)
            decision = decoder.decode(observation)
            risks = {
                label: 1.0 - decision.posterior[index]
                for index, label in enumerate(SBS_OUTCOMES)
            }
            if "leakage" in observation.as_tuple():
                risks[FALLBACK_ACTION] = parameters.fallback_cost
            assert decision.expected_loss == pytest.approx(min(risks.values()), abs=1.0e-15)
            assert decision.action in {
                action for action, risk in risks.items() if risk == min(risks.values())
            }
            assert decision.fallback <= decision.fallback_admissible


def test_direct_anchor_preserves_xz_execution_and_zx_kraus_label_order() -> None:
    assert standard_observed_sbs_action(PairedSyndrome(x="g", z="g")) == "K_gg"
    assert standard_observed_sbs_action(PairedSyndrome(x="e", z="g")) == "K_ge"
    assert standard_observed_sbs_action(PairedSyndrome(x="g", z="e")) == "K_eg"
    assert standard_observed_sbs_action(PairedSyndrome(x="e", z="e")) == "K_ee"
    assert (
        standard_observed_sbs_action(PairedSyndrome(x="leakage", z="g"))
        == FALLBACK_ACTION
    )


def test_nonleak_observation_can_override_direct_class_without_hidden_truth() -> None:
    parameters = _parameters(1)
    decoder = StaticSBSBranchDecoder(parameters)
    observation = PairedSyndrome(x="g", z="e")

    direct = standard_observed_sbs_action(observation)
    decision = decoder.decode(observation)

    assert direct == "K_eg"
    assert decision.action == "K_gg"
    assert not decision.fallback
    assert decision.expected_loss < 1.0 - decision.posterior[SBS_OUTCOMES.index(direct)]


def test_reset_kernel_changes_frozen_likelihood_and_parameter_hash() -> None:
    base = static_protocol_validation_scenarios()[3]
    model_a = base.make_model()
    parameters_a = build_static_sbs_parameters(
        model_a,
        ideal_branch_prior=base.ideal_branch_prior,
        fallback_cost=base.fallback_cost,
        calibration_protocol_id="reset-a",
    )
    f_g, f_e, h_g, h_e = base.injection_probabilities
    e_reset, f_reset, _ = base.reset_success
    model_b = make_persistent_leakage_model(
        readout_confusion=np.asarray(base.readout_confusion),
        f_injection_given_g=f_g,
        f_injection_given_e=f_e,
        higher_injection_given_g=h_g,
        higher_injection_given_e=h_e,
        e_reset_success=e_reset,
        f_reset_success=f_reset,
        higher_reset_success=0.90,
        counter_max=63,
        readout_provenance="test reset sensitivity",
        parameter_provenance="test reset sensitivity",
    )
    parameters_b = build_static_sbs_parameters(
        model_b,
        ideal_branch_prior=base.ideal_branch_prior,
        fallback_cost=base.fallback_cost,
        calibration_protocol_id="reset-b",
    )

    assert parameters_a.model_sha256 != parameters_b.model_sha256
    assert parameters_a.parameter_sha256 != parameters_b.parameter_sha256
    assert not np.allclose(
        parameters_a.likelihood_array(), parameters_b.likelihood_array(), atol=1.0e-6
    )


def test_reducible_hidden_carry_model_fails_closed() -> None:
    readout = np.array(
        [
            [0.96, 0.03, 0.01],
            [0.03, 0.95, 0.02],
            [0.05, 0.05, 0.90],
            [0.05, 0.05, 0.90],
        ]
    )
    model = make_persistent_leakage_model(
        readout_confusion=readout,
        f_injection_given_g=0.02,
        f_injection_given_e=0.02,
        higher_injection_given_g=0.02,
        higher_injection_given_e=0.02,
        e_reset_success=1.0,
        f_reset_success=0.0,
        higher_reset_success=0.0,
        counter_max=31,
        readout_provenance="reducible test",
        parameter_provenance="reducible test",
    )

    with pytest.raises(ValueError, match="initial-state-dependent"):
        build_static_sbs_parameters(
            model,
            ideal_branch_prior=(0.25, 0.25, 0.25, 0.25),
            fallback_cost=0.3,
            calibration_protocol_id="reducible",
        )


@pytest.mark.parametrize(
    ("prior", "cost", "message"),
    [
        ((0.5, 0.5, 0.0, 0.0), 0.2, "strictly positive"),
        ((0.25, 0.25, 0.25), 0.2, "4"),
        ((0.25, 0.25, 0.25, 0.25), 0.0, "strictly between"),
        ((0.25, 0.25, 0.25, 0.25), 1.0, "strictly between"),
    ],
)
def test_invalid_prior_and_loss_fail_closed(prior, cost, message) -> None:
    scenario = static_protocol_validation_scenarios()[0]
    with pytest.raises(ValueError, match=message):
        build_static_sbs_parameters(
            scenario.make_model(),
            ideal_branch_prior=prior,
            fallback_cost=cost,
            calibration_protocol_id="invalid",
        )


def test_scoring_keeps_fallback_cost_separate_from_branch_error() -> None:
    assert score_sbs_branch_action("K_ge", "K_ge", fallback_cost=0.3) == 0.0
    assert score_sbs_branch_action("K_gg", "K_ge", fallback_cost=0.3) == 1.0
    assert score_sbs_branch_action(FALLBACK_ACTION, "K_ge", fallback_cost=0.3) == 0.3
    with pytest.raises(ValueError, match="action"):
        score_sbs_branch_action("CORRECT_LEAKAGE", "K_ge", fallback_cost=0.3)


def test_comparison_registry_keeps_branch_target_separate_from_logical_binning() -> None:
    gates = validate_static_protocol_comparison_registration()
    entry = next(
        item
        for item in major_comparison_registry()
        if item.comparison_id == "t3_1_4_static_sbs_branch_comparison"
    )

    assert entry.method_ids == (
        STANDARD_SBS_BRANCH_ID,
        STATIC_SBS_MAP_ID,
        STATIC_SBS_DECODER_ID,
        IDEAL_BRANCH_REFERENCE_ID,
    )
    assert entry.standard_binning_policy == "not_applicable"
    assert "target" in entry.rationale
    assert gates[-1] == "registry:t3_1_4_target_separation"


def _implementation_hash() -> str:
    paths = (
        ROOT / "cnn_fpga" / "benchmark" / "static_protocol_decoder.py",
        ROOT / "physics" / "sbs_observation_reset.py",
        ROOT / "physics" / "sbs_error_space.py",
        ROOT / "cnn_fpga" / "benchmark" / "standard_binning_baseline.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def test_production_artifact_is_source_bound_and_non_demo_scale() -> None:
    payload = json.loads(JSON_ARTIFACT.read_text(encoding="utf-8"))

    assert payload["status"] == "PASS"
    assert payload["implementation_sha256"] == _implementation_hash()
    assert payload["aggregate"]["cycles"] == 640_000
    assert payload["aggregate"]["resolved_positive_scenarios"] == 3
    assert payload["aggregate"]["nonleak_override_scenarios"] == 3
    assert payload["aggregate"]["direct_minus_protocol_aware_seed_cluster_ci"][
        "ci_low"
    ] > 0.0
    assert payload["aggregate"]["maximum_empirical_likelihood_abs_error"] <= 0.015
    assert payload["gate_summary"]["failed"] == 0
    assert payload["comparison_contract"]["not_target"] == (
        "logical_Pauli_class_or_full_quantum_recovery"
    )


def test_source_data_has_all_scenario_seed_rows_and_recomputes_costs() -> None:
    payload = json.loads(JSON_ARTIFACT.read_text(encoding="utf-8"))
    with CSV_ARTIFACT.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 4 * 8
    assert len({(row["scenario_id"], row["evaluation_seed"]) for row in rows}) == 32
    assert all(int(row["cycles"]) == 20_000 for row in rows)
    assert all(len(row["parameter_sha256"]) == 64 for row in rows)
    by_seed: dict[str, list[float]] = {}
    for row in rows:
        by_seed.setdefault(row["evaluation_seed"], []).append(
            float(row["direct_minus_protocol_aware"])
        )
    seed_means = np.asarray([np.mean(values) for values in by_seed.values()])
    assert float(np.mean(seed_means)) == pytest.approx(
        payload["aggregate"]["direct_minus_protocol_aware_seed_cluster_ci"][
            "estimate"
        ],
        abs=1.0e-15,
    )
    assert all(len(values) == 4 for values in by_seed.values())


def test_claim_boundary_forbids_postselection_and_oracle_aliases() -> None:
    payload = json.loads(JSON_ARTIFACT.read_text(encoding="utf-8"))
    forbidden = payload["claim_boundary"]["forbidden"]

    assert "free post-selection" in forbidden
    assert "full finite-energy recovery optimum" in forbidden
    assert "T3.1.3 full-state decoder oracle" in forbidden
    assert all(
        scenario["protocol_aware_fallback_rate"]
        <= scenario["observed_leakage_rate"] + 1.0e-15
        for scenario in payload["scenarios"]
    )
