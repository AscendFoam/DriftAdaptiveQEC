"""Statcalib comparator contract helpers.

This module defines a separate statcalib comparator lane contract without
changing existing ParamMapper or benchmark semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping

import numpy as np

from cnn_fpga.runtime.param_bank import DecoderRuntimeParams


STATCALIB_STATUS_GENERATED = "generated"
STATCALIB_STATUS_NOT_GENERATED = "not_generated"
STATCALIB_STATUS_NOT_APPLICABLE = "not_applicable"
STATCALIB_STATUS_DIAGNOSTIC_ERROR = "diagnostic_error"

STATCALIB_REASON_PARAMS_EMITTED = "statcalib_params_emitted"
STATCALIB_REASON_SIGNAL_INSUFFICIENT = "insufficient_calibration_signal"
STATCALIB_REASON_MODE_NOT_APPLICABLE = "mode_does_not_emit_statcalib"
STATCALIB_REASON_INTERFACE_VALIDATION_FAILED = "interface_validation_failed"
STATCALIB_REASON_DIAGNOSTIC_ERROR = "statcalib_diagnostic_error"

STATCALIB_STATUS_VALUES = (
    STATCALIB_STATUS_GENERATED,
    STATCALIB_STATUS_NOT_GENERATED,
    STATCALIB_STATUS_NOT_APPLICABLE,
    STATCALIB_STATUS_DIAGNOSTIC_ERROR,
)

STATCALIB_REASON_VALUES = (
    STATCALIB_REASON_PARAMS_EMITTED,
    STATCALIB_REASON_SIGNAL_INSUFFICIENT,
    STATCALIB_REASON_MODE_NOT_APPLICABLE,
    STATCALIB_REASON_INTERFACE_VALIDATION_FAILED,
    STATCALIB_REASON_DIAGNOSTIC_ERROR,
)


def _coerce_vector2(value: np.ndarray | list[float], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (2,):
        raise ValueError(f"{name} must have shape (2,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


def _coerce_matrix22(value: np.ndarray | list[list[float]], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (2, 2):
        raise ValueError(f"{name} must have shape (2, 2), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


def _coerce_float_mapping(value: Mapping[str, Any], name: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, raw in value.items():
        number = float(raw)
        if not np.isfinite(number):
            raise ValueError(f"{name}[{key!r}] must be finite")
        out[str(key)] = number
    return out


def _coerce_prediction_mapping(value: Mapping[str, Any] | None) -> Dict[str, float] | None:
    if value is None:
        return None
    out = _coerce_float_mapping(value, "teacher_prediction")
    required = {"sigma", "mu_q", "mu_p", "theta_deg"}
    missing = required.difference(out)
    if missing:
        raise ValueError(f"teacher_prediction missing required keys: {sorted(missing)}")
    return out


@dataclass(frozen=True)
class StatCalibInput:
    """Typed statcalib comparator input contract."""

    window_id: int
    slow_update_index: int
    prior_decoder_params: DecoderRuntimeParams
    histogram_summary: Dict[str, float]
    calibration_features: Dict[str, float]
    source: str = "statcalib"
    teacher_prediction: Dict[str, float] | None = None
    teacher_decoder_params: DecoderRuntimeParams | None = None
    provenance: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.window_id) < 0:
            raise ValueError("window_id must be non-negative")
        if int(self.slow_update_index) < 0:
            raise ValueError("slow_update_index must be non-negative")
        if not str(self.source).strip():
            raise ValueError("source must be non-empty")
        object.__setattr__(self, "window_id", int(self.window_id))
        object.__setattr__(self, "slow_update_index", int(self.slow_update_index))
        object.__setattr__(self, "prior_decoder_params", self.prior_decoder_params.copy())
        object.__setattr__(self, "histogram_summary", _coerce_float_mapping(self.histogram_summary, "histogram_summary"))
        object.__setattr__(self, "calibration_features", _coerce_float_mapping(self.calibration_features, "calibration_features"))
        object.__setattr__(self, "teacher_prediction", _coerce_prediction_mapping(self.teacher_prediction))
        object.__setattr__(
            self,
            "teacher_decoder_params",
            None if self.teacher_decoder_params is None else self.teacher_decoder_params.copy(),
        )
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "provenance", dict(self.provenance))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_id": self.window_id,
            "slow_update_index": self.slow_update_index,
            "source": self.source,
            "histogram_summary": dict(self.histogram_summary),
            "calibration_features": dict(self.calibration_features),
            "teacher_prediction": None if self.teacher_prediction is None else dict(self.teacher_prediction),
            "teacher_decoder_params": None if self.teacher_decoder_params is None else self.teacher_decoder_params.to_dict(),
            "prior_decoder_params": self.prior_decoder_params.to_dict(),
            "provenance": dict(self.provenance),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class StatCalibOutput:
    """Typed statcalib comparator output contract."""

    status: str
    reason: str
    source: str = "statcalib"
    K: np.ndarray | None = None
    b: np.ndarray | None = None
    delta_b: np.ndarray | None = None
    provenance: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in STATCALIB_STATUS_VALUES:
            raise ValueError(f"unsupported statcalib status: {self.status}")
        if self.reason not in STATCALIB_REASON_VALUES:
            raise ValueError(f"unsupported statcalib reason: {self.reason}")
        if not str(self.source).strip():
            raise ValueError("source must be non-empty")
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "provenance", dict(self.provenance))
        object.__setattr__(self, "metadata", dict(self.metadata))

        if self.status == STATCALIB_STATUS_GENERATED:
            if self.K is None or self.b is None or self.delta_b is None:
                raise ValueError("generated statcalib output must include K, b, and delta_b")
            object.__setattr__(self, "K", _coerce_matrix22(self.K, "K"))
            object.__setattr__(self, "b", _coerce_vector2(self.b, "b"))
            object.__setattr__(self, "delta_b", _coerce_vector2(self.delta_b, "delta_b"))
            return

        if self.K is not None or self.b is not None or self.delta_b is not None:
            raise ValueError("non-generated statcalib output must not include K, b, or delta_b")

    @classmethod
    def from_delta_b(
        cls,
        statcalib_input: StatCalibInput,
        delta_b: np.ndarray | list[float],
        *,
        source: str = "statcalib",
        K: np.ndarray | list[list[float]] | None = None,
        provenance: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "StatCalibOutput":
        delta_b_array = _coerce_vector2(delta_b, "delta_b")
        prior = statcalib_input.prior_decoder_params
        next_K = prior.K.copy() if K is None else _coerce_matrix22(K, "K")
        next_b = prior.b + delta_b_array
        merged_provenance = {
            "window_id": statcalib_input.window_id,
            "slow_update_index": statcalib_input.slow_update_index,
            "input_source": statcalib_input.source,
            **dict(statcalib_input.provenance),
            **dict(provenance or {}),
        }
        merged_metadata = {
            "histogram_summary": dict(statcalib_input.histogram_summary),
            "calibration_features": dict(statcalib_input.calibration_features),
            "teacher_prediction_present": bool(statcalib_input.teacher_prediction is not None),
            "teacher_decoder_params_present": bool(statcalib_input.teacher_decoder_params is not None),
            **dict(statcalib_input.metadata),
            **dict(metadata or {}),
        }
        return cls(
            status=STATCALIB_STATUS_GENERATED,
            reason=STATCALIB_REASON_PARAMS_EMITTED,
            source=source,
            K=next_K,
            b=next_b,
            delta_b=delta_b_array,
            provenance=merged_provenance,
            metadata=merged_metadata,
        )

    @classmethod
    def not_generated(
        cls,
        *,
        source: str = "statcalib",
        reason: str = STATCALIB_REASON_SIGNAL_INSUFFICIENT,
        provenance: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "StatCalibOutput":
        return cls(
            status=STATCALIB_STATUS_NOT_GENERATED,
            reason=reason,
            source=source,
            provenance=dict(provenance or {}),
            metadata=dict(metadata or {}),
        )

    @classmethod
    def not_applicable(
        cls,
        *,
        source: str = "statcalib",
        provenance: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "StatCalibOutput":
        return cls(
            status=STATCALIB_STATUS_NOT_APPLICABLE,
            reason=STATCALIB_REASON_MODE_NOT_APPLICABLE,
            source=source,
            provenance=dict(provenance or {}),
            metadata=dict(metadata or {}),
        )

    @classmethod
    def diagnostic_error(
        cls,
        *,
        source: str = "statcalib",
        reason: str = STATCALIB_REASON_DIAGNOSTIC_ERROR,
        provenance: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "StatCalibOutput":
        return cls(
            status=STATCALIB_STATUS_DIAGNOSTIC_ERROR,
            reason=reason,
            source=source,
            provenance=dict(provenance or {}),
            metadata=dict(metadata or {}),
        )

    def to_runtime_params(self) -> DecoderRuntimeParams:
        if self.status != STATCALIB_STATUS_GENERATED:
            raise ValueError(f"cannot convert statcalib output with status={self.status!r} to DecoderRuntimeParams")
        return DecoderRuntimeParams(
            K=self.K,
            b=self.b,
            metadata={
                "runtime_mode": "statcalib",
                "statcalib_status": self.status,
                "statcalib_reason": self.reason,
                "statcalib_source": self.source,
                "statcalib_provenance": dict(self.provenance),
                "applied_delta_b": None if self.delta_b is None else self.delta_b.tolist(),
                "statcalib_metadata": dict(self.metadata),
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "source": self.source,
            "K": None if self.K is None else self.K.tolist(),
            "b": None if self.b is None else self.b.tolist(),
            "delta_b": None if self.delta_b is None else self.delta_b.tolist(),
            "provenance": dict(self.provenance),
            "metadata": dict(self.metadata),
        }

