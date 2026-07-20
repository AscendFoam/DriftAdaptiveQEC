from __future__ import annotations

import numpy as np
import pytest

from physics.operational_boundary import matched_operational_boundary


def test_sustained_boundary_rejects_transient_early_advantage() -> None:
    result = matched_operational_boundary(
        [0.0, 1.0, 2.0, 3.0, 4.0],
        [1.0, 0.75, 0.82, 0.72, 0.71],
        [1.0, 0.80, 0.78, 0.70, 0.65],
    )
    assert result.first_positive_index == 2
    assert result.last_negative_index == 1
    assert result.sustained_dominance_index == 2
    assert result.sustained_dominance_time_us == 2.0
    assert result.sign_reversal_count == 1
    assert result.terminal_advantage == pytest.approx(0.06)


def test_late_disadvantage_moves_boundary_after_last_negative() -> None:
    result = matched_operational_boundary(
        np.arange(6.0),
        [1.0, 0.70, 0.83, 0.65, 0.72, 0.70],
        [1.0, 0.80, 0.78, 0.68, 0.66, 0.62],
    )
    assert result.first_positive_index == 2
    assert result.last_negative_index == 3
    assert result.sustained_dominance_index == 4
    assert result.sign_reversal_count == 3


def test_cumulative_boundary_repays_initial_deficit() -> None:
    result = matched_operational_boundary(
        np.arange(6.0),
        [1.0, 0.70, 0.78, 0.80, 0.80, 0.80],
        [1.0, 0.80, 0.74, 0.70, 0.65, 0.60],
    )
    assert result.sustained_dominance_index == 2
    assert result.cumulative_breakeven_index == 4
    assert result.cumulative_advantage_us[3] < 0.0
    assert result.cumulative_advantage_us[4] > 0.0
    assert 3.0 < result.cumulative_linear_crossing_time_us < 4.0


def test_no_boundary_when_terminal_sample_remains_worse() -> None:
    result = matched_operational_boundary(
        np.arange(4.0),
        [1.0, 0.8, 0.7, 0.6],
        [1.0, 0.85, 0.75, 0.65],
    )
    assert result.sustained_dominance_index is None
    assert result.cumulative_breakeven_index is None
    assert result.terminal_cumulative_advantage_us < 0.0


def test_never_worse_curve_has_zero_boundary() -> None:
    result = matched_operational_boundary(
        np.arange(4.0),
        [1.0, 0.9, 0.85, 0.8],
        [1.0, 0.8, 0.7, 0.6],
    )
    assert result.sustained_dominance_index == 0
    assert result.cumulative_breakeven_index == 0
    assert result.to_dict()["ratio_reported"] is False


@pytest.mark.parametrize(
    "time,active,passive,match",
    [
        ([0.0, 1.0], [1.0, 0.9], [1.0, 0.8], "at least 3"),
        ([0.0, 1.0, 1.0], [1.0, 0.9, 0.8], [1.0, 0.8, 0.7], "increase strictly"),
        ([0.0, 1.0, 3.0], [1.0, 0.9, 0.8], [1.0, 0.8, 0.7], "uniform"),
        ([0.0, 1.0, 2.0], [1.0, 1.1, 0.8], [1.0, 0.8, 0.7], "active fidelity"),
        ([0.0, 1.0, 2.0], [0.9, 0.8, 0.7], [1.0, 0.8, 0.7], "same unit"),
        ([0.0, 1.0, 2.0], [1.0, 0.8, 0.7], [1.0, 0.8], "equal length"),
    ],
)
def test_invalid_boundary_inputs_fail_closed(time, active, passive, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        matched_operational_boundary(time, active, passive)


def test_negative_tolerance_is_rejected() -> None:
    with pytest.raises(ValueError, match="tolerance"):
        matched_operational_boundary(
            [0.0, 1.0, 2.0], [1.0, 0.8, 0.7], [1.0, 0.8, 0.7], tolerance=-1.0
        )
