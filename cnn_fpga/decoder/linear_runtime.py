"""Fast-loop linear decoder runtime with optional fixed-point emulation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from physics.constants import LATTICE_CONST

from cnn_fpga.runtime.param_bank import DecoderRuntimeParams


@dataclass(frozen=True)
class FixedPointFormat:
    """Signed fixed-point format used by the fast loop."""

    integer_bits: int = 4
    fractional_bits: int = 20

    def __post_init__(self) -> None:
        for name, value in (
            ("integer_bits", self.integer_bits),
            ("fractional_bits", self.fractional_bits),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.word_bits > 63:
            raise ValueError("fixed-point word width above 63 bits is unsupported")

    @classmethod
    def from_spec(cls, spec: str) -> "FixedPointFormat":
        match = re.fullmatch(r"Q(\d+)\.(\d+)", spec.strip(), flags=re.IGNORECASE)
        if match is None:
            raise ValueError(f"Unsupported fixed-point spec: {spec}")
        return cls(integer_bits=int(match.group(1)), fractional_bits=int(match.group(2)))

    @property
    def step(self) -> float:
        return 2.0 ** (-self.fractional_bits)

    @property
    def word_bits(self) -> int:
        """One sign bit plus the configured integer and fractional fields."""

        return 1 + self.integer_bits + self.fractional_bits

    @property
    def min_code(self) -> int:
        return -(1 << (self.word_bits - 1))

    @property
    def max_code(self) -> int:
        return (1 << (self.word_bits - 1)) - 1

    @property
    def min_value(self) -> float:
        # 中文注释：这里按“1 个符号位 + integer_bits 个整数位 + fractional_bits 个小数位”解释。
        return -(2.0 ** self.integer_bits)

    @property
    def max_value(self) -> float:
        return (2.0 ** self.integer_bits) - self.step

    def encode(
        self,
        value: np.ndarray | list[float] | float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode to signed integer words using explicit ties-to-even rounding."""

        array = np.asarray(value, dtype=float)
        if not np.all(np.isfinite(array)):
            raise ValueError("fixed-point input must contain only finite values")
        scaled = np.rint(array / self.step)
        saturated = np.logical_or(scaled < self.min_code, scaled > self.max_code)
        codes = np.clip(scaled, self.min_code, self.max_code).astype(np.int64)
        return codes, saturated

    def decode(self, codes: np.ndarray | list[int] | int) -> np.ndarray:
        """Decode signed integer words and reject out-of-format codes."""

        array = np.asarray(codes)
        if not np.issubdtype(array.dtype, np.integer):
            raise TypeError("fixed-point codes must be integers")
        signed = array.astype(np.int64, copy=False)
        if np.any(signed < self.min_code) or np.any(signed > self.max_code):
            raise ValueError("fixed-point code lies outside the configured word width")
        return signed.astype(float) * self.step

    def quantize(self, value: np.ndarray | list[float] | float) -> tuple[np.ndarray, np.ndarray]:
        codes, saturated = self.encode(value)
        return self.decode(codes), saturated


@dataclass(frozen=True)
class LinearRuntimeConfig:
    """Configuration for one-cycle linear decoder emulation."""

    fixed_point_spec: str = "Q4.20"
    enable_fixed_point: bool = True
    syndrome_limit: float = LATTICE_CONST / 2.0
    correction_limit: float = LATTICE_CONST

    def __post_init__(self) -> None:
        if self.syndrome_limit <= 0:
            raise ValueError("syndrome_limit must be positive")
        if self.correction_limit <= 0:
            raise ValueError("correction_limit must be positive")

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LinearRuntimeConfig":
        hardware = config.get("hardware_defaults", {})
        fast_cfg = config.get("fast_loop", {})
        return cls(
            fixed_point_spec=str(fast_cfg.get("fixed_point", hardware.get("fixed_point", "Q4.20"))),
            enable_fixed_point=bool(fast_cfg.get("enable_fixed_point", True)),
            syndrome_limit=float(fast_cfg.get("syndrome_limit", LATTICE_CONST / 2.0)),
            correction_limit=float(fast_cfg.get("correction_limit", LATTICE_CONST)),
        )


@dataclass
class LinearRuntimeResult:
    """Outputs of one fast-loop decode cycle."""

    syndrome_raw: np.ndarray
    syndrome_used: np.ndarray
    correction_unclipped: np.ndarray
    correction_reference: np.ndarray
    correction_applied: np.ndarray
    params_used: DecoderRuntimeParams
    syndrome_saturated: np.ndarray
    param_saturated: Dict[str, np.ndarray]
    correction_clip_saturated: np.ndarray
    correction_fixed_point_saturated: np.ndarray
    correction_saturated: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        return {
            "syndrome_raw": self.syndrome_raw.tolist(),
            "syndrome_used": self.syndrome_used.tolist(),
            "correction_unclipped": self.correction_unclipped.tolist(),
            "correction_reference": self.correction_reference.tolist(),
            "correction_applied": self.correction_applied.tolist(),
            "params_used": self.params_used.to_dict(),
            "syndrome_saturated": self.syndrome_saturated.astype(int).tolist(),
            "param_saturated": {
                "K": self.param_saturated["K"].astype(int).tolist(),
                "b": self.param_saturated["b"].astype(int).tolist(),
            },
            "correction_clip_saturated": self.correction_clip_saturated.astype(int).tolist(),
            "correction_fixed_point_saturated": self.correction_fixed_point_saturated.astype(int).tolist(),
            "correction_saturated": self.correction_saturated.astype(int).tolist(),
        }


class LinearRuntime:
    """Cycle-accurate-ish software model of the FPGA linear decoder."""

    def __init__(self, config: LinearRuntimeConfig) -> None:
        self.config = config
        self.fixed_point = FixedPointFormat.from_spec(config.fixed_point_spec)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LinearRuntime":
        return cls(LinearRuntimeConfig.from_config(config))

    def _clip_syndrome(self, syndrome: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(syndrome, dtype=float), -self.config.syndrome_limit, self.config.syndrome_limit)

    def _clip_correction(self, correction: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(correction, dtype=float), -self.config.correction_limit, self.config.correction_limit)

    def decode(self, syndrome: np.ndarray, params: DecoderRuntimeParams) -> LinearRuntimeResult:
        """Decode one syndrome sample using the current active runtime parameters."""
        syndrome_raw = np.asarray(syndrome, dtype=float).reshape(2)
        syndrome_clipped = self._clip_syndrome(syndrome_raw)
        correction_unclipped = params.K @ syndrome_clipped + params.b
        correction_reference = self._clip_correction(correction_unclipped)
        correction_clip_saturated = np.abs(correction_unclipped) > self.config.correction_limit

        if not self.config.enable_fixed_point:
            return LinearRuntimeResult(
                syndrome_raw=syndrome_raw,
                syndrome_used=syndrome_clipped,
                correction_unclipped=correction_unclipped,
                correction_reference=correction_reference,
                correction_applied=correction_reference.copy(),
                params_used=params.copy(),
                syndrome_saturated=np.abs(syndrome_raw) > self.config.syndrome_limit,
                param_saturated={"K": np.zeros_like(params.K, dtype=bool), "b": np.zeros_like(params.b, dtype=bool)},
                correction_clip_saturated=correction_clip_saturated,
                correction_fixed_point_saturated=np.zeros_like(correction_reference, dtype=bool),
                correction_saturated=correction_clip_saturated,
            )

        syndrome_used, syndrome_sat = self.fixed_point.quantize(syndrome_clipped)
        k_used, k_sat = self.fixed_point.quantize(params.K)
        b_used, b_sat = self.fixed_point.quantize(params.b)
        correction_hw_unclipped = k_used @ syndrome_used + b_used
        correction_hw_clip_sat = np.abs(correction_hw_unclipped) > self.config.correction_limit
        correction_hw = self._clip_correction(correction_hw_unclipped)
        correction_applied, correction_quant_sat = self.fixed_point.quantize(correction_hw)

        params_used = DecoderRuntimeParams(K=k_used, b=b_used, metadata=dict(params.metadata))
        return LinearRuntimeResult(
            syndrome_raw=syndrome_raw,
            syndrome_used=syndrome_used,
            correction_unclipped=correction_hw_unclipped,
            correction_reference=correction_reference,
            correction_applied=correction_applied,
            params_used=params_used,
            syndrome_saturated=np.abs(syndrome_raw) > self.config.syndrome_limit,
            param_saturated={"K": k_sat, "b": b_sat},
            correction_clip_saturated=correction_hw_clip_sat,
            correction_fixed_point_saturated=correction_quant_sat,
            correction_saturated=np.logical_or(correction_hw_clip_sat, correction_quant_sat),
        )
