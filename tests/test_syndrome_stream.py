from __future__ import annotations

from dataclasses import fields
import math

import numpy as np
import pytest

from physics.constants import LATTICE_CONST
from physics.drift_processes import ConstantDriftProcess, DriftState
from physics.sbs_error_space import SBS_PROTOCOL_ID
from physics.syndrome_stream import (
    CONSTITUENT_PHASES_RAD,
    MODEL_SCOPE,
    ObservedSyndromeStep,
    SyndromeStreamConfig,
    SyndromeTruthStep,
    generate_syndrome_stream,
)


def _state(**updates: object) -> DriftState:
    values: dict[str, object] = {
        "sigma_q": 1.0e-12,
        "sigma_p": 1.0e-12,
        "source": "unit-test",
        "regime": "quiet",
    }
    values.update(updates)
    return DriftState(**values)


def _quiet_config(**updates: object) -> SyndromeStreamConfig:
    values: dict[str, object] = {
        "measurement_sigma": (0.0, 0.0),
        "loss_environment_variance": 0.0,
        "depth_probability_scale": 0.0,
        "recovery_probability": 0.0,
        "recovery_gain": 0.0,
        "base_leakage_probability": 0.0,
        "loss_leakage_scale": 0.0,
        "burst_leakage_bonus": 0.0,
        "readout_fidelity_g": 1.0,
        "readout_fidelity_e": 1.0,
        "seed": 17,
    }
    values.update(updates)
    return SyndromeStreamConfig(**values)


def test_config_fails_closed_on_invalid_probabilities_geometry_and_seed() -> None:
    invalid = [
        lambda: SyndromeStreamConfig(lattice=0.0),
        lambda: SyndromeStreamConfig(measurement_sigma=(0.1,)),
        lambda: SyndromeStreamConfig(measurement_sigma=(-0.1, 0.1)),
        lambda: SyndromeStreamConfig(loss_environment_variance=-1.0),
        lambda: SyndromeStreamConfig(max_recovery_depth=0),
        lambda: SyndromeStreamConfig(max_recovery_depth=True),
        lambda: SyndromeStreamConfig(depth_probability_scale=1.1),
        lambda: SyndromeStreamConfig(depth_probability_power=0.0),
        lambda: SyndromeStreamConfig(loss_leakage_scale=-0.1),
        lambda: SyndromeStreamConfig(higher_leakage_mean_duration=1.99),
        lambda: SyndromeStreamConfig(seed=-1),
        lambda: SyndromeStreamConfig(seed=2**64),
        lambda: SyndromeStreamConfig(model_scope="device calibrated"),
    ]
    for call in invalid:
        with pytest.raises((TypeError, ValueError)):
            call()


def test_process_and_explicit_sequence_have_fail_closed_step_contracts() -> None:
    process = ConstantDriftProcess(base=_state())
    with pytest.raises(ValueError, match="steps is required"):
        generate_syndrome_stream(process)
    with pytest.raises(ValueError, match="must equal"):
        generate_syndrome_stream([_state()], steps=2)
    with pytest.raises(TypeError, match="every stream state"):
        generate_syndrome_stream([_state(), object()])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="source"):
        generate_syndrome_stream("bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="config"):
        generate_syndrome_stream([], config=object())  # type: ignore[arg-type]


def test_empty_sequence_returns_typed_empty_stream() -> None:
    result = generate_syndrome_stream([], config=_quiet_config())
    assert result.steps == ()
    assert result.observed_records() == ()
    assert result.truth_records() == ()
    assert result.final_physical_residual == (0.0, 0.0)
    assert result.final_logical_bits == (0, 0)
    assert result.protocol_id == SBS_PROTOCOL_ID
    assert result.model_scope == MODEL_SCOPE
    assert result.device_calibrated is False


def test_each_step_outputs_required_observed_and_truth_fields() -> None:
    state = _state(
        step=8,
        time=1.25,
        mu_q=0.11,
        mu_p=-0.17,
        sigma_q=0.03,
        sigma_p=0.05,
        rho=0.4,
        loss_gamma=0.2,
        p_outlier=0.3,
        outlier_scale=4.0,
        burst_active=True,
        source="burst",
        regime="burst-active",
        seed=99,
        event_id=3,
    )
    step = generate_syndrome_stream([state], config=_quiet_config(seed=23)).steps[0]
    assert isinstance(step.observed, ObservedSyndromeStep)
    assert isinstance(step.truth, SyndromeTruthStep)
    assert step.observed.drift_step == state.step
    assert step.observed.time == state.time
    assert step.observed.quadrature_phases_rad == CONSTITUENT_PHASES_RAD
    assert step.observed.syndrome.x in {"g", "e", "leakage"}
    assert step.observed.syndrome.z in {"g", "e", "leakage"}
    assert step.truth.drift_state == state
    assert step.truth.hidden_regime == state.regime
    assert step.truth.burst_active is True
    assert step.truth.true_logical_label in {"I", "X", "Z", "Y"}


def test_deployable_schema_is_structurally_isolated_from_hidden_truth() -> None:
    result = generate_syndrome_stream(
        [_state(regime="secret", p_outlier=1.0, outlier_scale=2.0)],
        config=_quiet_config(),
    )
    record = result.observed_records()[0]
    forbidden_fragments = (
        "truth",
        "hidden",
        "regime",
        "outlier",
        "leakage_kind",
        "recovery_depth",
        "logical",
        "drift_state",
    )
    assert not any(fragment in key for key in record for fragment in forbidden_fragments)
    assert {field.name for field in fields(ObservedSyndromeStep)}.isdisjoint(
        {"hidden_regime", "true_logical_label", "recovery_depth"}
    )


def test_truth_records_are_complete_enough_for_audit_and_read_only() -> None:
    records = generate_syndrome_stream([_state()], config=_quiet_config()).truth_records()
    record = records[0]
    assert {
        "drift_state",
        "channel_displacement",
        "outlier_component",
        "loss_environment_noise",
        "true_folded_syndrome",
        "lattice_indices",
        "true_logical_label",
        "hidden_regime",
        "leakage_hazard",
        "recovery_depth_before_action",
        "recovery_depth",
        "physical_residual_after_action",
    } <= set(record)
    with pytest.raises(TypeError):
        record["recovery_depth"] = 10  # type: ignore[index]


def test_analog_measurement_and_modular_residual_obey_half_open_cell() -> None:
    config = _quiet_config(measurement_sigma=(0.8, 0.9), seed=71)
    result = generate_syndrome_stream([_state(mu_q=0.4, mu_p=-0.3)] * 200, config=config)
    for step in result.steps:
        analog = np.asarray(step.observed.analog_syndrome)
        residual = np.asarray(step.observed.residual_syndrome)
        expected = (analog + 0.5 * LATTICE_CONST) % LATTICE_CONST - 0.5 * LATTICE_CONST
        np.testing.assert_allclose(residual, expected, atol=2.0e-15, rtol=0.0)
        assert np.all(residual >= -0.5 * LATTICE_CONST)
        assert np.all(residual < 0.5 * LATTICE_CONST)


def test_logical_truth_tracks_q_as_x_p_as_z_and_composes_pauli_bits() -> None:
    states = [
        _state(mu_q=LATTICE_CONST),
        _state(mu_p=LATTICE_CONST),
        _state(mu_q=LATTICE_CONST),
        _state(mu_p=LATTICE_CONST),
    ]
    labels = [
        step.truth.true_logical_label
        for step in generate_syndrome_stream(states, config=_quiet_config()).steps
    ]
    assert labels == ["X", "Y", "Z", "I"]


def test_channel_displacement_consumes_full_correlated_gaussian_mixture() -> None:
    state = DriftState(
        mu_q=0.2,
        mu_p=-0.1,
        sigma_q=0.2,
        sigma_p=0.3,
        rho=0.4,
        p_outlier=0.2,
        outlier_scale=3.0,
        regime="mixture",
    )
    count = 20_000
    result = generate_syndrome_stream([state] * count, config=_quiet_config(seed=909))
    samples = np.asarray([step.truth.channel_displacement for step in result.steps])
    masks = np.asarray([step.truth.outlier_component for step in result.steps])
    np.testing.assert_allclose(samples.mean(axis=0), state.mean, atol=0.012, rtol=0.0)
    np.testing.assert_allclose(np.cov(samples, rowvar=False), state.mixture_covariance, rtol=0.08, atol=0.008)
    assert float(masks.mean()) == pytest.approx(state.p_outlier, abs=0.012)


def test_loss_attenuates_carried_residual_without_silently_becoming_sigma_proxy() -> None:
    first = _state(mu_q=0.4, mu_p=-0.2)
    second = _state(loss_gamma=math.log(4.0))
    result = generate_syndrome_stream([first, second], config=_quiet_config(seed=41))
    first_residual = np.asarray(result.steps[0].truth.physical_residual_after_action)
    second_shift = np.asarray(result.steps[1].truth.pre_measurement_shift)
    expected = 0.5 * first_residual
    np.testing.assert_allclose(second_shift, expected, atol=8.0e-12, rtol=0.0)
    assert result.steps[1].truth.loss_environment_noise == (0.0, 0.0)


def test_loss_environment_variance_and_loss_leakage_hazard_are_explicit() -> None:
    gamma = math.log(2.0)
    config = _quiet_config(
        loss_environment_variance=0.5,
        loss_leakage_scale=0.4,
        seed=812,
    )
    result = generate_syndrome_stream([_state(loss_gamma=gamma)] * 8_000, config=config)
    noise = np.asarray([step.truth.loss_environment_noise for step in result.steps])
    np.testing.assert_allclose(noise.var(axis=0), [0.25, 0.25], rtol=0.07, atol=0.01)
    assert result.steps[0].truth.leakage_hazard == pytest.approx(0.2)


def test_derived_leakage_hazard_fails_closed_instead_of_clipping() -> None:
    config = _quiet_config(base_leakage_probability=0.8, burst_leakage_bonus=0.3)
    with pytest.raises(ValueError, match="derived leakage hazard"):
        generate_syndrome_stream([_state(burst_active=True)], config=config)


def test_burst_regime_can_trigger_observed_leakage_without_exposing_regime() -> None:
    config = _quiet_config(burst_leakage_bonus=1.0, higher_leakage_fraction=0.0)
    states = [_state(burst_active=True, regime="burst"), _state(regime="quiet")]
    result = generate_syndrome_stream(states, config=config)
    assert result.steps[0].observed.syndrome.as_tuple() == ("leakage", "leakage")
    assert result.steps[0].truth.leakage_kind == "f"
    assert result.steps[1].observed.syndrome.as_tuple() != ("leakage", "leakage")
    assert "regime" not in result.observed_records()[0]


def test_leakage_depth_has_pending_quadrature_and_recovers_after_leakage_ends() -> None:
    config = _quiet_config(
        burst_leakage_bonus=1.0,
        higher_leakage_fraction=0.0,
        recovery_probability=1.0,
        recovery_gain=0.5,
    )
    states = [_state(burst_active=True, mu_q=0.3), _state(mu_q=0.1)]
    result = generate_syndrome_stream(states, config=config)
    leaked = result.steps[0]
    recovered = result.steps[1]
    assert leaked.truth.recovery_quadrature is None
    assert leaked.truth.recovery_quadrature_after_action == "X"
    assert leaked.truth.recovery_depth_after_action == 1
    assert recovered.truth.recovery_quadrature == "X"
    assert recovered.observed.syndrome.as_tuple() == ("e", "g")
    assert recovered.truth.recovery_succeeded is True
    assert recovered.truth.recovery_depth_after_action == 0


def test_higher_leakage_persists_for_at_least_two_cycles() -> None:
    config = _quiet_config(
        burst_leakage_bonus=1.0,
        higher_leakage_fraction=1.0,
        higher_leakage_mean_duration=4.0,
        seed=124,
    )
    states = [_state(burst_active=True)] + [_state()] * 20
    result = generate_syndrome_stream(states, config=config)
    run = 0
    for step in result.steps:
        if step.observed.syndrome.x != "leakage":
            break
        run += 1
        assert step.truth.leakage_kind == "higher"
    assert 2 <= run < len(states)
    assert result.steps[run - 1].observed.leakage_run == run


def test_recovery_is_constituent_specific_not_two_axis_shrinkage() -> None:
    config = _quiet_config(
        max_recovery_depth=1,
        depth_probability_scale=1.0,
        depth_probability_power=1.0,
        recovery_probability=1.0,
        recovery_gain=0.5,
        seed=3,
    )
    state = _state(mu_q=0.5 * LATTICE_CONST - 1.0e-7, mu_p=0.2 * LATTICE_CONST)
    step = generate_syndrome_stream([state], config=config).steps[0]
    assert step.truth.injected_recovery_depth == 1
    assert step.truth.recovery_quadrature == "X"
    assert step.observed.syndrome.as_tuple() == ("e", "g")
    before = np.asarray(step.truth.true_folded_syndrome)
    after = np.asarray(step.truth.physical_residual_after_action)
    assert after[0] == pytest.approx(0.5 * before[0])
    assert after[1] == pytest.approx(before[1])


def test_configured_g_e_confusion_is_visible_in_long_recovery_run() -> None:
    config = _quiet_config(
        max_recovery_depth=1,
        depth_probability_scale=1.0,
        recovery_probability=0.0,
        readout_fidelity_e=0.8,
        readout_fidelity_g=0.9,
        seed=445,
    )
    states = [_state(mu_q=0.5 * LATTICE_CONST - 1.0e-7)] + [_state()] * 4_999
    result = generate_syndrome_stream(states, config=config)
    x_e = np.mean([step.observed.syndrome.x == "e" for step in result.steps])
    z_g = np.mean([step.observed.syndrome.z == "g" for step in result.steps])
    assert x_e == pytest.approx(0.8, abs=0.025)
    assert z_g == pytest.approx(0.9, abs=0.025)


def test_recovery_depth_and_observed_run_lengths_are_causal() -> None:
    config = _quiet_config(
        max_recovery_depth=2,
        depth_probability_scale=1.0,
        recovery_probability=0.0,
        seed=5,
    )
    states = [_state(mu_q=0.5 * LATTICE_CONST - 1.0e-7)] + [_state()] * 4
    result = generate_syndrome_stream(states, config=config)
    assert [step.truth.recovery_depth_after_action for step in result.steps] == [2] * 5
    assert [step.observed.x_e_run for step in result.steps] == [1, 2, 3, 4, 5]
    assert [step.observed.z_e_run for step in result.steps] == [0] * 5


def test_fixed_seed_is_exactly_reproducible_and_prefix_stable() -> None:
    base = DriftState(sigma_q=0.2, sigma_p=0.4, rho=-0.3, p_outlier=0.1, outlier_scale=4.0)
    process = ConstantDriftProcess(base=base, seed=21)
    config = SyndromeStreamConfig(seed=777)
    short = generate_syndrome_stream(process, steps=25, config=config)
    repeated = generate_syndrome_stream(process, steps=25, config=config)
    long = generate_syndrome_stream(process, steps=60, config=config)
    assert short == repeated
    assert short.steps == long.steps[:25]


def test_different_stream_seed_changes_stochastic_trajectory_not_drift_truth() -> None:
    states = ConstantDriftProcess(base=DriftState(), seed=9).generate(10)
    first = generate_syndrome_stream(states, config=SyndromeStreamConfig(seed=1))
    second = generate_syndrome_stream(states, config=SyndromeStreamConfig(seed=2))
    assert [step.truth.drift_state for step in first.steps] == list(states)
    assert [step.truth.drift_state for step in second.steps] == list(states)
    assert [step.truth.channel_displacement for step in first.steps] != [
        step.truth.channel_displacement for step in second.steps
    ]


def test_public_physics_exports_resolve_without_eager_import_contract_breakage() -> None:
    from physics import SyndromeStreamConfig as PublicConfig
    from physics import generate_syndrome_stream as public_generate

    assert PublicConfig is SyndromeStreamConfig
    assert public_generate is generate_syndrome_stream
