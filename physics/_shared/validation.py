"""Validation primitives shared by physics configuration objects."""

from __future__ import annotations

from math import isfinite
from numbers import Integral
from typing import Sequence


def integer(value: int, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def finite(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def finite_pair(values: Sequence[float], name: str) -> tuple[float, float]:
    if isinstance(values, (str, bytes)) or len(values) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    return finite(values[0], f"{name}[0]"), finite(values[1], f"{name}[1]")


def positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return int(value)


def nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return int(value)


def finite_positive(value: float, name: str) -> float:
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result
