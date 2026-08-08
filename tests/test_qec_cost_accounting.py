from __future__ import annotations

import pytest

from physics.qec_cost_accounting import (
    postselection_cost,
    scale_measurement_feedback_cost,
    squeezing_db_from_projector_delta,
)


def test_projector_delta_uses_registered_squeezing_convention() -> None:
    assert squeezing_db_from_projector_delta(0.34) == pytest.approx(6.360121703, abs=1e-9)


def test_measurement_feedback_counts_scale_as_integers() -> None:
    cost = scale_measurement_feedback_cost(
        horizon_us=300.0,
        cycle_duration_us=10.0,
        measurements_per_full_cycle=2,
        resets_per_full_cycle=2,
        active_gates_per_full_cycle=18,
    )
    assert cost.full_cycles == 30
    assert cost.half_cycles == 60
    assert cost.measurement_events == 60
    assert cost.reset_events == 60
    assert cost.active_gate_applications == 540


def test_noninteger_horizon_fails_closed() -> None:
    with pytest.raises(ValueError, match="integer number"):
        scale_measurement_feedback_cost(
            horizon_us=305.0,
            cycle_duration_us=10.0,
            measurements_per_full_cycle=2,
            resets_per_full_cycle=2,
            active_gates_per_full_cycle=18,
        )


def test_postselection_prices_accepted_failures_and_rejection() -> None:
    result = postselection_cost(
        acceptance_fraction=0.9,
        raw_error_rate=0.02,
        conditional_error_rate=0.005,
    )
    assert result.accepted_failures_per_input == pytest.approx(0.0045)
    assert result.rejection_fraction == pytest.approx(0.1)
    assert result.total_costs == pytest.approx((0.0045, 0.0295, 0.0545, 0.1045))
    assert result.break_even_rejection_penalty == pytest.approx(0.155)
    assert result.to_dict()["conditional_metric_online_eligible"] is False


def test_unit_rejection_penalty_can_reverse_conditional_improvement() -> None:
    result = postselection_cost(
        acceptance_fraction=0.5,
        raw_error_rate=0.014,
        conditional_error_rate=0.0001,
    )
    assert result.conditional_error_rate < result.raw_error_rate
    assert result.total_costs[-1] > result.raw_error_rate


@pytest.mark.parametrize("delta", [0.0, -0.1, float("nan")])
def test_invalid_projector_delta_is_rejected(delta: float) -> None:
    with pytest.raises(ValueError, match="projector_delta"):
        squeezing_db_from_projector_delta(delta)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"acceptance_fraction": 1.1, "raw_error_rate": 0.1, "conditional_error_rate": 0.05}, "acceptance"),
        ({"acceptance_fraction": 0.9, "raw_error_rate": 0.1, "conditional_error_rate": 0.2}, "cannot exceed"),
        ({"acceptance_fraction": 0.9, "raw_error_rate": 0.1, "conditional_error_rate": 0.05, "rejection_penalties": (0.5, 0.25)}, "sorted"),
    ],
)
def test_invalid_postselection_cost_inputs_fail_closed(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        postselection_cost(**kwargs)

