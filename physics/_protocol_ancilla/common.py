"""Shared protocol vocabularies and exact validation helpers."""

from __future__ import annotations

from math import isfinite

import numpy as np
from numpy.typing import ArrayLike, NDArray

SBS_FAULT_STAGES = ("small_cd", "big_cd", "readout")
SBS_CONSTITUENTS = ("X", "Z")
SHARPEN_TRIM_PROTOCOL_ID = "PROTO-SHARPEN-TRIM-XVAL"
SHARPEN_TRIM_ROUND_TYPES = (
    "q_peak_sharpen",
    "p_peak_sharpen",
    "q_envelope_trim",
    "p_envelope_trim",
)
SHARPEN_TRIM_HIDDEN_STATES = ("+y", "-y", "leakage")
SHARPEN_TRIM_OBSERVED_CLASSES = ("+y", "-y")
SHARPEN_TRIM_CARRY_STATES = ("g", "+y", "-y", "leakage")


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value.strip()


def _counter(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    integer = int(value)
    if integer < 0:
        raise ValueError(f"{name} must be non-negative")
    return integer


def _probability(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real probability")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real probability") from exc
    if not isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return number


def _probability_matrix(
    values: object,
    shape: tuple[int, int],
    name: str,
) -> tuple[tuple[float, ...], ...]:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric matrix") from exc
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have finite shape {shape}")
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} entries must lie in [0, 1]")
    return tuple(tuple(float(value) for value in row) for row in array)


def _probability_vector(values: object, length: int, name: str) -> tuple[float, ...]:
    try:
        vector = tuple(_probability(value, name) for value in values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"{name} must contain {length} probabilities") from exc
    if len(vector) != length:
        raise ValueError(f"{name} must contain {length} probabilities")
    return vector


def _row_stochastic(values: ArrayLike, shape: tuple[int, int], name: str) -> NDArray[np.float64]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError(f"{name} must contain finite non-negative values")
    if not np.allclose(np.sum(matrix, axis=1), 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"{name} rows must sum to 1")
    result = np.array(matrix, copy=True)
    result.setflags(write=False)
    return result


def _seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError("seed must be an integer")
    integer = int(value)
    if integer < 0:
        raise ValueError("seed must be non-negative")
    return integer

