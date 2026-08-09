"""Vectorized sampling primitives shared by protocol models."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def categorical_rows(
    probabilities: NDArray[np.float64], rng: np.random.Generator
) -> NDArray[np.int64]:
    if probabilities.ndim != 2 or probabilities.shape[0] == 0:
        raise ValueError("probabilities must be a non-empty 2D matrix")
    if np.any(probabilities < 0.0) or not np.allclose(
        probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("probability rows must be non-negative and normalized")
    cumulative = np.cumsum(probabilities, axis=1)
    choices = np.sum(rng.random(probabilities.shape[0])[:, None] > cumulative, axis=1)
    return np.minimum(choices, probabilities.shape[1] - 1).astype(np.int64)
