from __future__ import annotations

import numpy as np
import pytest

from cnn_fpga.benchmark.phase9_paired_cluster_uq import (
    half_trace_norm,
    paired_density_trace_ucb,
    paired_vector_norm_ucb,
)


def _diagonal_density_stack(
    rng: np.random.Generator, count: int, dimension: int
) -> np.ndarray:
    probabilities = rng.dirichlet(np.ones(dimension) * 3.0, size=count)
    return np.asarray([np.diag(row) for row in probabilities], dtype=np.complex128)


def test_half_trace_norm_matches_diagonal_total_variation() -> None:
    diagonal = np.asarray([0.4, -0.3, -0.1])
    assert half_trace_norm(np.diag(diagonal)) == pytest.approx(0.4)


def test_density_ucb_is_deterministic_and_above_point() -> None:
    rng = np.random.default_rng(123)
    left = _diagonal_density_stack(rng, 96, 8)
    right = _diagonal_density_stack(rng, 96, 8)
    kwargs = {
        "confidence": 0.95,
        "multiplier_replicates": 199,
        "seed": 9001,
        "calibration_factor": 1.1,
        "quantization_bounds": np.full(96, 2e-7),
    }
    first = paired_density_trace_ucb(left, right, **kwargs)
    second = paired_density_trace_ucb(left, right, **kwargs)
    assert first == second
    assert first.upper_bound > first.estimate
    assert first.quantization_bound == pytest.approx(2e-7)
    assert first.cluster_count == 96


def test_density_ucb_detects_registered_shift() -> None:
    rng = np.random.default_rng(456)
    left = _diagonal_density_stack(rng, 128, 6)
    right = left.copy()
    shift = 0.08
    right[:, 0, 0] -= shift
    right[:, 1, 1] += shift
    result = paired_density_trace_ucb(
        left,
        right,
        confidence=0.95,
        multiplier_replicates=199,
        seed=9002,
        calibration_factor=1.0,
    )
    assert result.estimate == pytest.approx(shift)
    assert result.raw_radius < 1e-12
    assert result.upper_bound == pytest.approx(shift)


def test_vector_ucb_uses_norm_of_mean_not_mean_absolute_difference() -> None:
    left = np.asarray([[1.0], [-1.0]] * 64)
    right = np.zeros_like(left)
    result = paired_vector_norm_ucb(
        left,
        right,
        ord_value=1,
        confidence=0.95,
        multiplier_replicates=199,
        seed=9003,
        calibration_factor=1.0,
    )
    assert result.estimate == pytest.approx(0.0)
    assert result.upper_bound > 0.0


@pytest.mark.parametrize("factor", [0.0, 0.99, np.inf])
def test_invalid_calibration_factor_fails_closed(factor: float) -> None:
    values = np.stack([np.eye(2) / 2] * 8)
    with pytest.raises(ValueError, match="calibration_factor"):
        paired_density_trace_ucb(
            values,
            values,
            confidence=0.95,
            multiplier_replicates=99,
            seed=1,
            calibration_factor=factor,
        )


def test_nonhermitian_density_fails_closed() -> None:
    left = np.stack([np.eye(2, dtype=np.complex128) / 2] * 8)
    right = left.copy()
    right[0, 0, 1] = 0.2j
    with pytest.raises(ValueError, match="Hermitian"):
        paired_density_trace_ucb(
            left,
            right,
            confidence=0.95,
            multiplier_replicates=99,
            seed=2,
            calibration_factor=1.0,
        )
