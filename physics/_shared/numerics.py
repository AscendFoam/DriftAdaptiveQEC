"""Numerical kernels shared by otherwise independent models."""

from __future__ import annotations

from math import pi, sqrt

import numpy as np
from numpy.typing import NDArray


def hermite_functions(
    coordinate: NDArray[np.float64], cutoff: int
) -> NDArray[np.float64]:
    functions = np.empty((cutoff, coordinate.size), dtype=np.float64)
    previous = np.zeros_like(coordinate)
    current = pi ** (-0.25) * np.exp(-0.5 * coordinate * coordinate)
    for n in range(cutoff):
        functions[n] = current
        following = sqrt(2.0 / (n + 1.0)) * coordinate * current
        if n:
            following -= sqrt(n / (n + 1.0)) * previous
        previous, current = current, following
    return functions
