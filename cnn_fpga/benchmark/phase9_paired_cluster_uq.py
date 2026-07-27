"""Paired-cluster uncertainty bounds for Phase-9 density qualification.

The primary density estimand is

``0.5 * || E[rho_left - rho_right] ||_1``.

Directly bootstrapping that non-smooth norm is unreliable near the null.
Instead, this module bootstraps the *matrix mean error* after cluster
centering and takes its trace norm.  The resulting radius is then added to
the observed norm.  A separately frozen calibration factor may inflate the
raw multiplier quantile; selecting that factor from formal outcomes is
forbidden.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class NormUCB:
    estimate: float
    raw_radius: float
    calibrated_radius: float
    quantization_bound: float
    upper_bound: float
    confidence: float
    multiplier_replicates: int
    cluster_count: int
    calibration_factor: float
    seed: int


def _validated_hermitian_stack(
    values: Sequence[np.ndarray] | np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    stack = np.asarray(values, dtype=np.complex128)
    if (
        stack.ndim != 3
        or stack.shape[0] < 2
        or stack.shape[1] != stack.shape[2]
        or not np.all(np.isfinite(stack.real))
        or not np.all(np.isfinite(stack.imag))
    ):
        raise ValueError(f"{name} must be a finite square matrix stack")
    hermitian = 0.5 * (stack + np.swapaxes(stack.conj(), -1, -2))
    residual = float(np.max(np.abs(stack - hermitian)))
    if residual > 1e-9:
        raise ValueError(f"{name} is not Hermitian within tolerance")
    return hermitian


def half_trace_norm(matrix: np.ndarray) -> float:
    value = np.asarray(matrix, dtype=np.complex128)
    if (
        value.ndim != 2
        or value.shape[0] != value.shape[1]
        or not np.all(np.isfinite(value.real))
        or not np.all(np.isfinite(value.imag))
    ):
        raise ValueError("matrix must be finite and square")
    hermitian = 0.5 * (value + value.conj().T)
    if float(np.max(np.abs(value - hermitian))) > 1e-9:
        raise ValueError("matrix is not Hermitian within tolerance")
    return 0.5 * float(np.sum(np.abs(np.linalg.eigvalsh(hermitian))))


def _higher_quantile(values: np.ndarray, confidence: float) -> float:
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0.5 and 1")
    return float(np.quantile(values, confidence, method="higher"))


def paired_density_trace_ucb(
    left: Sequence[np.ndarray] | np.ndarray,
    right: Sequence[np.ndarray] | np.ndarray,
    *,
    confidence: float,
    multiplier_replicates: int,
    seed: int,
    calibration_factor: float,
    quantization_bounds: Sequence[float] | np.ndarray | None = None,
    batch_size: int = 64,
) -> NormUCB:
    """Return a calibrated one-sided trace-distance upper bound.

    ``left[i]`` and ``right[i]`` form one paired cluster.  Pairing is required
    even when the physical random streams are independent because the
    registered seed position and logical-state schedule define the common
    cluster denominator.
    """

    left_stack = _validated_hermitian_stack(left, name="left")
    right_stack = _validated_hermitian_stack(right, name="right")
    if left_stack.shape != right_stack.shape:
        raise ValueError("left and right density stacks must have equal shape")
    if multiplier_replicates < 99:
        raise ValueError("at least 99 multiplier replicates are required")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not math.isfinite(calibration_factor) or calibration_factor < 1.0:
        raise ValueError("calibration_factor must be finite and at least one")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    count = left_stack.shape[0]
    differences = left_stack - right_stack
    mean_difference = np.mean(differences, axis=0)
    centered = differences - mean_difference[None, :, :]
    estimate = half_trace_norm(mean_difference)

    if quantization_bounds is None:
        quantization = np.zeros(count, dtype=np.float64)
    else:
        quantization = np.asarray(quantization_bounds, dtype=np.float64)
        if (
            quantization.shape != (count,)
            or not np.all(np.isfinite(quantization))
            or np.any(quantization < 0.0)
        ):
            raise ValueError(
                "quantization_bounds must be one finite non-negative value per cluster"
            )
    quantization_bound = float(np.mean(quantization))

    rng = np.random.default_rng(seed)
    radii = np.empty(multiplier_replicates, dtype=np.float64)
    for start in range(0, multiplier_replicates, batch_size):
        stop = min(start + batch_size, multiplier_replicates)
        weights = rng.integers(
            0,
            2,
            size=(stop - start, count),
            endpoint=False,
            dtype=np.int8,
        )
        signed = weights.astype(np.float64) * 2.0 - 1.0
        perturbations = np.einsum(
            "bn,nij->bij", signed, centered, optimize=True
        ) / float(count)
        perturbations = 0.5 * (
            perturbations + np.swapaxes(perturbations.conj(), -1, -2)
        )
        eigenvalues = np.linalg.eigvalsh(perturbations)
        radii[start:stop] = 0.5 * np.sum(np.abs(eigenvalues), axis=1)

    raw_radius = _higher_quantile(radii, confidence)
    calibrated_radius = calibration_factor * raw_radius
    upper_bound = estimate + quantization_bound + calibrated_radius
    if not all(
        math.isfinite(value)
        for value in (
            estimate,
            raw_radius,
            calibrated_radius,
            quantization_bound,
            upper_bound,
        )
    ):
        raise ValueError("non-finite UCB result")
    return NormUCB(
        estimate=estimate,
        raw_radius=raw_radius,
        calibrated_radius=calibrated_radius,
        quantization_bound=quantization_bound,
        upper_bound=upper_bound,
        confidence=confidence,
        multiplier_replicates=multiplier_replicates,
        cluster_count=count,
        calibration_factor=calibration_factor,
        seed=seed,
    )


def paired_vector_norm_ucb(
    left: np.ndarray,
    right: np.ndarray,
    *,
    ord_value: float,
    confidence: float,
    multiplier_replicates: int,
    seed: int,
    calibration_factor: float,
    batch_size: int = 256,
) -> NormUCB:
    """Multiplier UCB for an absolute scalar or vector mean difference."""

    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    if left_values.ndim == 1:
        left_values = left_values[:, None]
    if right_values.ndim == 1:
        right_values = right_values[:, None]
    if (
        left_values.shape != right_values.shape
        or left_values.ndim != 2
        or left_values.shape[0] < 2
        or not np.all(np.isfinite(left_values))
        or not np.all(np.isfinite(right_values))
    ):
        raise ValueError("left/right values must be equal finite cluster matrices")
    if ord_value not in (1, 2, np.inf):
        raise ValueError("ord_value must be 1, 2, or infinity")
    if multiplier_replicates < 99:
        raise ValueError("at least 99 multiplier replicates are required")
    if not math.isfinite(calibration_factor) or calibration_factor < 1.0:
        raise ValueError("calibration_factor must be finite and at least one")

    differences = left_values - right_values
    mean_difference = np.mean(differences, axis=0)
    centered = differences - mean_difference[None, :]
    estimate = float(np.linalg.norm(mean_difference, ord=ord_value))
    rng = np.random.default_rng(seed)
    radii = np.empty(multiplier_replicates, dtype=np.float64)
    count = len(differences)
    for start in range(0, multiplier_replicates, batch_size):
        stop = min(start + batch_size, multiplier_replicates)
        weights = rng.integers(
            0,
            2,
            size=(stop - start, count),
            endpoint=False,
            dtype=np.int8,
        )
        signed = weights.astype(np.float64) * 2.0 - 1.0
        perturbations = signed @ centered / float(count)
        radii[start:stop] = np.linalg.norm(
            perturbations, ord=ord_value, axis=1
        )
    raw_radius = _higher_quantile(radii, confidence)
    calibrated_radius = calibration_factor * raw_radius
    upper_bound = estimate + calibrated_radius
    return NormUCB(
        estimate=estimate,
        raw_radius=raw_radius,
        calibrated_radius=calibrated_radius,
        quantization_bound=0.0,
        upper_bound=upper_bound,
        confidence=confidence,
        multiplier_replicates=multiplier_replicates,
        cluster_count=count,
        calibration_factor=calibration_factor,
        seed=seed,
    )


__all__ = [
    "NormUCB",
    "half_trace_norm",
    "paired_density_trace_ucb",
    "paired_vector_norm_ucb",
]
