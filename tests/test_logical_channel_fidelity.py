from __future__ import annotations

import math

import numpy as np
import pytest

from physics.fock_logical_channel import STATE_LABELS, logical_eigenstate_density
from physics.logical_channel_fidelity import (
    cptni_identity_fidelity,
    short_time_effective_depolarization,
    terminal_cutoff_interval,
)


def _outputs(channel) -> dict[str, np.ndarray]:
    return {label: channel(logical_eigenstate_density(label)) for label in STATE_LABELS}


def test_identity_and_depolarizing_tp_channels_match_known_formulas() -> None:
    identity = cptni_identity_fidelity(np.eye(4), outputs=_outputs(lambda rho: rho))
    assert identity.entanglement_fidelity == pytest.approx(1.0)
    assert identity.average_fidelity == pytest.approx(1.0)
    assert identity.mean_code_survival == pytest.approx(1.0)
    assert identity.tp_formula_overstatement == pytest.approx(0.0)
    shrink = 0.7
    depolarizing = cptni_identity_fidelity(np.diag([1.0, shrink, shrink, shrink]))
    assert depolarizing.entanglement_fidelity == pytest.approx((1.0 + 3.0 * shrink) / 4.0)
    assert depolarizing.average_fidelity == pytest.approx((1.0 + shrink) / 2.0)


def test_uniform_erasure_is_leakage_inclusive_and_tp_formula_overstates() -> None:
    survival = 0.72
    metrics = cptni_identity_fidelity(
        survival * np.eye(4), outputs=_outputs(lambda rho: survival * rho)
    )
    assert metrics.entanglement_fidelity == pytest.approx(survival)
    assert metrics.average_fidelity == pytest.approx(survival)
    assert metrics.direct_six_state_average_fidelity == pytest.approx(survival)
    assert metrics.mean_conditional_state_fidelity == pytest.approx(1.0)
    assert metrics.tp_formula_overstatement == pytest.approx((1.0 - survival) / 3.0)
    assert metrics.tp_assuming_average_fidelity > metrics.average_fidelity


def test_trace_preserving_amplitude_damping_has_known_fidelity() -> None:
    gamma = 0.2
    root = math.sqrt(1.0 - gamma)
    ptm = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, root, 0.0, 0.0],
            [0.0, 0.0, root, 0.0],
            [gamma, 0.0, 0.0, 1.0 - gamma],
        ]
    )
    k0 = np.diag([1.0, root])
    k1 = np.array([[0.0, math.sqrt(gamma)], [0.0, 0.0]])
    metrics = cptni_identity_fidelity(
        ptm, outputs=_outputs(lambda rho: k0 @ rho @ k0.T + k1 @ rho @ k1.T)
    )
    expected_fe = (1.0 + 2.0 * root + 1.0 - gamma) / 4.0
    assert metrics.entanglement_fidelity == pytest.approx(expected_fe)
    assert metrics.average_fidelity == pytest.approx((2.0 * expected_fe + 1.0) / 3.0)
    assert metrics.mean_code_survival == pytest.approx(1.0)


def test_state_dependent_survival_keeps_conditional_diagnostic_separate() -> None:
    # |0> survives fully, |1> survives with amplitude s; this is one no-jump Kraus map.
    s = 0.6
    kraus = np.diag([1.0, math.sqrt(s)])
    outputs = _outputs(lambda rho: kraus @ rho @ kraus)
    # Reconstruct the PTM from the analytic map.
    root = math.sqrt(s)
    ptm = np.array(
        [
            [(1.0 + s) / 2.0, 0.0, 0.0, (1.0 - s) / 2.0],
            [0.0, root, 0.0, 0.0],
            [0.0, 0.0, root, 0.0],
            [(1.0 - s) / 2.0, 0.0, 0.0, (1.0 + s) / 2.0],
        ]
    )
    metrics = cptni_identity_fidelity(ptm, outputs=outputs)
    assert metrics.minimum_state_survival == pytest.approx(s)
    assert metrics.maximum_state_survival == pytest.approx(1.0)
    assert metrics.mean_conditional_state_fidelity != pytest.approx(metrics.average_fidelity)
    assert metrics.six_state_ptm_residual < 1.0e-14


def test_invalid_ptm_outputs_and_tp_assumption_fail_closed() -> None:
    with pytest.raises(ValueError, match="4x4"):
        cptni_identity_fidelity(np.eye(3))
    bad = np.eye(4)
    bad[0, 0] = 1.2
    with pytest.raises(ValueError, match="R_II"):
        cptni_identity_fidelity(bad)
    with pytest.raises(ValueError, match="exactly"):
        cptni_identity_fidelity(np.eye(4), outputs={"z_plus": np.eye(2) / 2.0})
    outputs = _outputs(lambda rho: rho)
    outputs["z_plus"] = np.zeros((2, 2))
    with pytest.raises(ValueError, match="zero survival"):
        cptni_identity_fidelity(np.eye(4), outputs=outputs)


def test_short_time_linear_decay_recovers_exact_rate_without_exponential_fit() -> None:
    gamma = 0.012
    times = np.arange(5, dtype=float) * 10.0
    fidelity = 1.0 - 0.5 * gamma * times
    result = short_time_effective_depolarization(times, fidelity)
    assert result.one_step_rate_per_us == pytest.approx(gamma)
    assert result.three_point_rate_per_us == pytest.approx(gamma)
    assert result.four_point_rate_per_us == pytest.approx(gamma)
    assert result.primary_rate_per_cycle == pytest.approx(10.0 * gamma)
    assert result.primary_lifetime_us == pytest.approx(1.0 / gamma)
    assert result.algebraic_inverse_rate_us == pytest.approx(1.0 / gamma)
    assert result.reliability_status == "reliable_discrete_short_time_proxy"
    assert result.discretization_spread_per_us < 1.0e-14
    assert result.to_dict()["exponential_fit_used"] is False


def test_short_time_curvature_keeps_discretization_spread_and_negative_rate_null() -> None:
    times = np.arange(5, dtype=float) * 10.0
    fidelity = np.exp(-0.01 * times)
    result = short_time_effective_depolarization(times, fidelity)
    assert result.discretization_spread_per_us > 0.0
    assert result.initial_curvature_per_us2 > 0.0
    transient = short_time_effective_depolarization(
        times, [1.0, 0.6, 0.82, 0.7, 0.75]
    )
    assert transient.primary_rate_per_us > 0.0
    assert transient.reliability_status == "unreliable_cycle_scale_transient"
    assert transient.first_three_monotone_nonincreasing is False
    assert transient.primary_lifetime_us is None
    assert transient.primary_lifetime_cycles is None
    assert transient.algebraic_inverse_rate_us is not None


@pytest.mark.parametrize(
    "times,values,match",
    [
        ([0, 1, 2], [1, 0.9, 0.8], ">=4"),
        ([0, 1, 3, 4], [1, 0.9, 0.8, 0.7], "uniform"),
        ([1, 2, 3, 4], [1, 0.9, 0.8, 0.7], "start at zero"),
        ([0, 1, 2, 3], [0.9, 0.8, 0.7, 0.6], "must equal one"),
        ([0, 1, 2, 3], [1, 0.9, 1.1, 0.6], r"\[0,1\]"),
    ],
)
def test_short_time_failure_paths(times, values, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        short_time_effective_depolarization(times, values)


def test_terminal_cutoff_interval_is_systematic_not_statistical() -> None:
    interval = terminal_cutoff_interval(0.81, 0.83, lower_cutoff=36, higher_cutoff=40)
    assert interval["numerical_interval_min"] == pytest.approx(0.81)
    assert interval["numerical_interval_max"] == pytest.approx(0.83)
    assert interval["absolute_spread"] == pytest.approx(0.02)
    assert interval["statistical_confidence_level"] is None
    assert interval["is_confidence_interval"] is False
    assert interval["infinite_cutoff_claim"] is False
    with pytest.raises(ValueError, match="strictly increasing"):
        terminal_cutoff_interval(0.8, 0.9, lower_cutoff=40, higher_cutoff=36)
