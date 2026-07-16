"""Typed, observed-only hybrid slow-state output contract for T4.1.3.

The output describes a future slow-loop state and an inactive parameter-bank
proposal.  It deliberately contains no cycle-critical correction, frame update
or pulse action.  Recovery depth is represented as an observed-data posterior
over recovery *burden*, never as simulator truth copied into the payload.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from math import isfinite, log
from types import MappingProxyType
from typing import Any, Mapping, Sequence
import zlib

import numpy as np

from cnn_fpga.data.experimental_history import (
    ExperimentalHistorySample,
    FEATURE_NAMES,
    audit_mapping_for_information_leakage,
)
from cnn_fpga.decoder.param_mapper import NoisePrediction, ParamMapper, ParamMapperConfig
from cnn_fpga.decoder.periodic_adaptive_map import (
    PeriodicMomentConfig,
    estimate_periodic_gaussian,
)
from cnn_fpga.decoder.regime_hmm import REGIME_CLASSES
from cnn_fpga.runtime.param_bank import DecoderRuntimeParams, ParamBank, PendingCommit
from cnn_fpga.runtime.run_length_fsm import (
    FALLBACK,
    FSM_MODES,
    LEAKAGE_HOLD,
    NORMAL,
    X_RECOVERY,
    Z_RECOVERY,
)
from physics.constants import LATTICE_CONST


SCHEMA_VERSION = "t4.1.3-hybrid-state-output-v1"
CONTINUOUS_PARAMETER_NAMES = (
    "mean_q",
    "mean_p",
    "sigma_q",
    "sigma_p",
    "rho_qp",
    "tail_rate",
    "x_e_rate",
    "z_e_rate",
    "leakage_rate",
)
CONTINUOUS_PARAMETER_UNITS = MappingProxyType(
    {
        "mean_q": "lattice_coordinate",
        "mean_p": "lattice_coordinate",
        "sigma_q": "lattice_coordinate",
        "sigma_p": "lattice_coordinate",
        "rho_qp": "dimensionless",
        "tail_rate": "probability",
        "x_e_rate": "probability",
        "z_e_rate": "probability",
        "leakage_rate": "probability",
    }
)
REGIME_POSTERIOR_SOURCES = (
    "t4.1.1_registered_gaussian_hmm",
    "registered_matched_budget_estimator",
    "observed_fallback_prior",
)
RISK_SOURCES = ("observed_beta_run_burden_proxy",)
CALIBRATION_SCOPES = ("uncalibrated_contract", "registered_synthetic_pilot")
BANK_ACTIONS = ("stage_candidate", "hold_active")
FORBIDDEN_DIRECT_OUTPUT_TOKENS = (
    "correction",
    "frameupdate",
    "pulseaction",
    "gatepulse",
    "fastaction",
    "cycleaction",
    "logicaltruth",
    "hiddentruth",
)


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _finite(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be real")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be real") from exc
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _probability(value: object, name: str) -> float:
    result = _finite(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0,1]")
    return result


def _tuple_floats(values: object, length: int, name: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a numeric sequence")
    try:
        result = tuple(_finite(item, f"{name}[{index}]") for index, item in enumerate(values))  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{name} must be a numeric sequence") from exc
    if len(result) != length:
        raise ValueError(f"{name} must contain {length} values")
    return result


def _entropy(probabilities: Sequence[float]) -> float:
    return float(-sum(value * log(value) for value in probabilities if value > 0.0))


def _normalized_key(value: object) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def _audit_output_provenance(provenance: Mapping[str, Any]) -> None:
    """Reuse the input denylist while allowing the legitimate output role name.

    ``regime`` is forbidden in model *inputs* but is a required T4.1.3 output.
    Only the registered top-level provenance key is exempted; its nested value
    is still recursively audited, so hidden-regime payloads remain rejected.
    """

    for key, value in provenance.items():
        if key == "regime_source":
            audit_mapping_for_information_leakage(value, path="hybrid_output.provenance.regime_source")
        else:
            audit_mapping_for_information_leakage(
                {key: value}, path="hybrid_output.provenance"
            )


@dataclass(frozen=True)
class ContinuousNoiseCalibration:
    mean_q: float
    mean_p: float
    sigma_q: float
    sigma_p: float
    rho_qp: float
    tail_rate: float
    x_e_rate: float
    z_e_rate: float
    leakage_rate: float
    source: str = "periodic_observed_moments"

    def __post_init__(self) -> None:
        half = 0.5 * LATTICE_CONST
        for name in ("mean_q", "mean_p"):
            value = _finite(getattr(self, name), name)
            if not -half <= value < half:
                raise ValueError(f"{name} must lie in the centered lattice cell")
            object.__setattr__(self, name, value)
        for name in ("sigma_q", "sigma_p"):
            value = _finite(getattr(self, name), name)
            if not 0.0 < value <= LATTICE_CONST:
                raise ValueError(f"{name} must lie in (0,lattice]")
            object.__setattr__(self, name, value)
        rho = _finite(self.rho_qp, "rho_qp")
        if not -1.0 < rho < 1.0:
            raise ValueError("rho_qp must lie strictly inside (-1,1)")
        object.__setattr__(self, "rho_qp", rho)
        for name in ("tail_rate", "x_e_rate", "z_e_rate", "leakage_rate"):
            object.__setattr__(self, name, _probability(getattr(self, name), name))
        if self.source != "periodic_observed_moments":
            raise ValueError("continuous source must be periodic_observed_moments")

    def as_vector(self) -> np.ndarray:
        return np.asarray([getattr(self, name) for name in CONTINUOUS_PARAMETER_NAMES], dtype=np.float64)

    @property
    def covariance_qp(self) -> np.ndarray:
        cross = self.rho_qp * self.sigma_q * self.sigma_p
        return np.asarray(
            [[self.sigma_q**2, cross], [cross, self.sigma_p**2]], dtype=np.float64
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "values": {name: float(getattr(self, name)) for name in CONTINUOUS_PARAMETER_NAMES},
            "units": dict(CONTINUOUS_PARAMETER_UNITS),
            "source": self.source,
        }


@dataclass(frozen=True)
class RegimePosteriorOutput:
    probabilities: tuple[float, ...]
    source: str

    def __post_init__(self) -> None:
        values = _tuple_floats(self.probabilities, len(REGIME_CLASSES), "probabilities")
        if any(value < 0.0 for value in values) or not np.isclose(
            sum(values), 1.0, atol=1.0e-12, rtol=0.0
        ):
            raise ValueError("regime probabilities must be nonnegative and normalized")
        if self.source not in REGIME_POSTERIOR_SOURCES:
            raise ValueError(f"source must be one of {REGIME_POSTERIOR_SOURCES}")
        object.__setattr__(self, "probabilities", values)

    @property
    def entropy_nats(self) -> float:
        return _entropy(self.probabilities)

    @property
    def confidence(self) -> float:
        return float(max(self.probabilities))

    @property
    def most_likely(self) -> str:
        return REGIME_CLASSES[int(np.argmax(self.probabilities))]

    def to_dict(self) -> dict[str, object]:
        return {
            "classes": list(REGIME_CLASSES),
            "probabilities": list(self.probabilities),
            "most_likely": self.most_likely,
            "entropy_nats": self.entropy_nats,
            "source": self.source,
        }


@dataclass(frozen=True)
class LeakageRecoveryOutput:
    leakage_probability_next_cycle: float
    leakage_probability_horizon: float
    horizon_cycles: int
    recovery_burden_posterior: tuple[float, ...]
    max_recovery_depth: int
    source: str = "observed_beta_run_burden_proxy"

    def __post_init__(self) -> None:
        next_risk = _probability(self.leakage_probability_next_cycle, "leakage_probability_next_cycle")
        horizon_risk = _probability(self.leakage_probability_horizon, "leakage_probability_horizon")
        if horizon_risk + 1.0e-15 < next_risk:
            raise ValueError("horizon leakage risk cannot be below next-cycle risk")
        horizon = _integer(self.horizon_cycles, "horizon_cycles", 1)
        depth = _integer(self.max_recovery_depth, "max_recovery_depth", 1)
        values = _tuple_floats(
            self.recovery_burden_posterior, depth + 1, "recovery_burden_posterior"
        )
        if any(value < 0.0 for value in values) or not np.isclose(
            sum(values), 1.0, atol=1.0e-12, rtol=0.0
        ):
            raise ValueError("recovery burden posterior must be nonnegative and normalized")
        if self.source not in RISK_SOURCES:
            raise ValueError(f"source must be one of {RISK_SOURCES}")
        object.__setattr__(self, "leakage_probability_next_cycle", next_risk)
        object.__setattr__(self, "leakage_probability_horizon", horizon_risk)
        object.__setattr__(self, "horizon_cycles", horizon)
        object.__setattr__(self, "max_recovery_depth", depth)
        object.__setattr__(self, "recovery_burden_posterior", values)

    @property
    def expected_recovery_depth(self) -> float:
        return float(sum(index * value for index, value in enumerate(self.recovery_burden_posterior)))

    @property
    def recovery_entropy_nats(self) -> float:
        return _entropy(self.recovery_burden_posterior)

    @property
    def leakage_entropy_nats(self) -> float:
        p = self.leakage_probability_next_cycle
        return _entropy((p, 1.0 - p))

    def to_dict(self) -> dict[str, object]:
        return {
            "leakage_probability_next_cycle": self.leakage_probability_next_cycle,
            "leakage_probability_horizon": self.leakage_probability_horizon,
            "horizon_cycles": self.horizon_cycles,
            "estimated_recovery_depth_classes": list(range(self.max_recovery_depth + 1)),
            "recovery_burden_posterior": list(self.recovery_burden_posterior),
            "expected_recovery_depth": self.expected_recovery_depth,
            "source": self.source,
            "truth_semantics": False,
        }


@dataclass(frozen=True)
class UncertaintyOutput:
    continuous_covariance: tuple[tuple[float, ...], ...]
    sample_count: int
    bootstrap_replicates: int
    block_length_cycles: int
    regime_entropy_nats: float
    leakage_entropy_nats: float
    recovery_entropy_nats: float
    ood_score: float
    recommendation_confidence: float
    calibration_scope: str
    source: str = "observed_moving_block_bootstrap"

    def __post_init__(self) -> None:
        matrix = np.asarray(self.continuous_covariance, dtype=np.float64)
        size = len(CONTINUOUS_PARAMETER_NAMES)
        if matrix.shape != (size, size) or not np.all(np.isfinite(matrix)):
            raise ValueError(f"continuous_covariance must be finite with shape {(size, size)}")
        if not np.allclose(matrix, matrix.T, atol=1.0e-12, rtol=0.0):
            raise ValueError("continuous_covariance must be symmetric")
        if float(np.min(np.linalg.eigvalsh(matrix))) < -1.0e-10:
            raise ValueError("continuous_covariance must be positive semidefinite")
        for name, minimum in (
            ("sample_count", 2),
            ("bootstrap_replicates", 32),
            ("block_length_cycles", 2),
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum))
        for name in ("regime_entropy_nats", "leakage_entropy_nats", "recovery_entropy_nats"):
            value = _finite(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "ood_score", _probability(self.ood_score, "ood_score"))
        object.__setattr__(
            self,
            "recommendation_confidence",
            _probability(self.recommendation_confidence, "recommendation_confidence"),
        )
        if self.calibration_scope not in CALIBRATION_SCOPES:
            raise ValueError(f"calibration_scope must be one of {CALIBRATION_SCOPES}")
        if self.source != "observed_moving_block_bootstrap":
            raise ValueError("uncertainty source must be observed_moving_block_bootstrap")
        immutable = tuple(tuple(float(value) for value in row) for row in matrix)
        object.__setattr__(self, "continuous_covariance", immutable)

    @property
    def standard_errors(self) -> tuple[float, ...]:
        diagonal = np.maximum(np.diag(np.asarray(self.continuous_covariance)), 0.0)
        return tuple(float(value) for value in np.sqrt(diagonal))

    def to_dict(self) -> dict[str, object]:
        return {
            "continuous_parameter_order": list(CONTINUOUS_PARAMETER_NAMES),
            "continuous_covariance": [list(row) for row in self.continuous_covariance],
            "continuous_standard_errors": list(self.standard_errors),
            "sample_count": self.sample_count,
            "bootstrap_replicates": self.bootstrap_replicates,
            "block_length_cycles": self.block_length_cycles,
            "regime_entropy_nats": self.regime_entropy_nats,
            "leakage_entropy_nats": self.leakage_entropy_nats,
            "recovery_entropy_nats": self.recovery_entropy_nats,
            "ood_score": self.ood_score,
            "recommendation_confidence": self.recommendation_confidence,
            "calibration_scope": self.calibration_scope,
            "source": self.source,
        }


@dataclass(frozen=True)
class ParameterBankRecommendation:
    bank_action: str
    recommended_mode: str
    gain_matrix: tuple[tuple[float, float], tuple[float, float]]
    bias: tuple[float, float]
    base_active_version: int
    valid_from_cycle: int
    expires_after_cycle: int
    recommendation_id: str
    payload_crc32: int
    calibration_scope: str
    mapping_method: str = "param_mapper_covariance_bridge_v1"
    hold_reason: str | None = None

    @staticmethod
    def _payload(
        bank_action: str,
        recommended_mode: str,
        gain_matrix: Sequence[Sequence[float]],
        bias: Sequence[float],
        base_active_version: int,
        valid_from_cycle: int,
        expires_after_cycle: int,
        recommendation_id: str,
        calibration_scope: str,
        mapping_method: str,
        hold_reason: str | None,
    ) -> dict[str, object]:
        return {
            "bank_action": bank_action,
            "recommended_mode": recommended_mode,
            "gain_matrix": [[float(value) for value in row] for row in gain_matrix],
            "bias": [float(value) for value in bias],
            "base_active_version": int(base_active_version),
            "valid_from_cycle": int(valid_from_cycle),
            "expires_after_cycle": int(expires_after_cycle),
            "recommendation_id": recommendation_id,
            "calibration_scope": calibration_scope,
            "mapping_method": mapping_method,
            "hold_reason": hold_reason,
        }

    @classmethod
    def create(
        cls,
        *,
        bank_action: str,
        recommended_mode: str,
        gain_matrix: Sequence[Sequence[float]],
        bias: Sequence[float],
        base_active_version: int,
        valid_from_cycle: int,
        expires_after_cycle: int,
        recommendation_id: str,
        calibration_scope: str,
        hold_reason: str | None = None,
    ) -> "ParameterBankRecommendation":
        payload = cls._payload(
            bank_action,
            recommended_mode,
            gain_matrix,
            bias,
            base_active_version,
            valid_from_cycle,
            expires_after_cycle,
            recommendation_id,
            calibration_scope,
            "param_mapper_covariance_bridge_v1",
            hold_reason,
        )
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return cls(payload_crc32=zlib.crc32(canonical) & 0xFFFFFFFF, **payload)  # type: ignore[arg-type]

    def __post_init__(self) -> None:
        if self.bank_action not in BANK_ACTIONS:
            raise ValueError(f"bank_action must be one of {BANK_ACTIONS}")
        if self.recommended_mode not in FSM_MODES:
            raise ValueError(f"recommended_mode must be one of {FSM_MODES}")
        matrix = np.asarray(self.gain_matrix, dtype=np.float64)
        bias = np.asarray(self.bias, dtype=np.float64)
        if matrix.shape != (2, 2) or bias.shape != (2,) or not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(bias)):
            raise ValueError("gain_matrix/bias must be finite with shapes (2,2)/(2,)")
        if not np.allclose(matrix, matrix.T, atol=1.0e-12, rtol=0.0):
            raise ValueError("gain_matrix must be symmetric")
        if float(np.min(np.linalg.eigvalsh(matrix))) < -1.0e-12 or float(np.max(np.linalg.eigvalsh(matrix))) > 1.2 + 1.0e-12:
            raise ValueError("gain_matrix eigenvalues must lie in [0,1.2]")
        for name in ("base_active_version", "valid_from_cycle", "expires_after_cycle"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        if self.expires_after_cycle < self.valid_from_cycle:
            raise ValueError("expires_after_cycle must not precede valid_from_cycle")
        if not isinstance(self.recommendation_id, str) or not self.recommendation_id.strip():
            raise ValueError("recommendation_id must be nonempty")
        if self.calibration_scope not in CALIBRATION_SCOPES:
            raise ValueError(f"calibration_scope must be one of {CALIBRATION_SCOPES}")
        if self.mapping_method != "param_mapper_covariance_bridge_v1":
            raise ValueError("mapping_method is not registered")
        if self.bank_action == "hold_active" and not self.hold_reason:
            raise ValueError("hold_active requires hold_reason")
        if self.bank_action == "stage_candidate" and self.hold_reason is not None:
            raise ValueError("stage_candidate must not carry hold_reason")
        crc = _integer(self.payload_crc32, "payload_crc32")
        if crc >= 2**32:
            raise ValueError("payload_crc32 must fit uint32")
        payload = self._payload(
            self.bank_action,
            self.recommended_mode,
            matrix,
            bias,
            self.base_active_version,
            self.valid_from_cycle,
            self.expires_after_cycle,
            self.recommendation_id,
            self.calibration_scope,
            self.mapping_method,
            self.hold_reason,
        )
        expected = zlib.crc32(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ) & 0xFFFFFFFF
        if crc != expected:
            raise ValueError("payload_crc32 does not match recommendation payload")
        object.__setattr__(self, "gain_matrix", tuple(tuple(float(value) for value in row) for row in matrix))
        object.__setattr__(self, "bias", tuple(float(value) for value in bias))

    def to_runtime_params(self) -> DecoderRuntimeParams:
        return DecoderRuntimeParams(
            K=np.asarray(self.gain_matrix, dtype=np.float64),
            b=np.asarray(self.bias, dtype=np.float64),
            metadata={
                "output_schema": SCHEMA_VERSION,
                "recommended_mode": self.recommended_mode,
                "recommendation_id": self.recommendation_id,
                "payload_crc32": self.payload_crc32,
                "mapping_method": self.mapping_method,
                "calibration_scope": self.calibration_scope,
            },
        )

    def to_dict(self) -> dict[str, object]:
        payload = self._payload(
            self.bank_action,
            self.recommended_mode,
            self.gain_matrix,
            self.bias,
            self.base_active_version,
            self.valid_from_cycle,
            self.expires_after_cycle,
            self.recommendation_id,
            self.calibration_scope,
            self.mapping_method,
            self.hold_reason,
        )
        payload["payload_crc32"] = self.payload_crc32
        return payload


@dataclass(frozen=True)
class HybridStateOutput:
    as_of_cycle: int
    history_start_cycle: int
    output_sequence: int
    continuous: ContinuousNoiseCalibration
    regime: RegimePosteriorOutput
    risk: LeakageRecoveryOutput
    uncertainty: UncertaintyOutput
    parameter_bank_recommendation: ParameterBankRecommendation
    provenance: Mapping[str, Any]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        as_of = _integer(self.as_of_cycle, "as_of_cycle")
        start = _integer(self.history_start_cycle, "history_start_cycle")
        if start > as_of:
            raise ValueError("history_start_cycle cannot exceed as_of_cycle")
        object.__setattr__(self, "output_sequence", _integer(self.output_sequence, "output_sequence"))
        for name, expected in (
            ("continuous", ContinuousNoiseCalibration),
            ("regime", RegimePosteriorOutput),
            ("risk", LeakageRecoveryOutput),
            ("uncertainty", UncertaintyOutput),
            ("parameter_bank_recommendation", ParameterBankRecommendation),
        ):
            if not isinstance(getattr(self, name), expected):
                raise TypeError(f"{name} must be {expected.__name__}")
        recommendation = self.parameter_bank_recommendation
        if recommendation.valid_from_cycle <= as_of:
            raise ValueError("bank recommendation must be valid only after as_of_cycle")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
        if not isinstance(self.provenance, Mapping):
            raise TypeError("provenance must be a mapping")
        _audit_output_provenance(self.provenance)
        if abs(self.uncertainty.regime_entropy_nats - self.regime.entropy_nats) > 1.0e-10:
            raise ValueError("uncertainty regime entropy does not match posterior")
        if abs(self.uncertainty.leakage_entropy_nats - self.risk.leakage_entropy_nats) > 1.0e-10:
            raise ValueError("uncertainty leakage entropy does not match risk")
        if abs(self.uncertainty.recovery_entropy_nats - self.risk.recovery_entropy_nats) > 1.0e-10:
            raise ValueError("uncertainty recovery entropy does not match burden posterior")
        object.__setattr__(self, "as_of_cycle", as_of)
        object.__setattr__(self, "history_start_cycle", start)
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    def to_deployable_dict(self) -> dict[str, object]:
        payload = {
            "schema_version": self.schema_version,
            "as_of_cycle": self.as_of_cycle,
            "history_start_cycle": self.history_start_cycle,
            "output_sequence": self.output_sequence,
            "continuous_noise_calibration": self.continuous.to_dict(),
            "regime_posterior": self.regime.to_dict(),
            "leakage_recovery_estimate": self.risk.to_dict(),
            "uncertainty": self.uncertainty.to_dict(),
            "parameter_bank_recommendation": self.parameter_bank_recommendation.to_dict(),
            "provenance": dict(self.provenance),
        }
        normalized = tuple(_normalized_key(name) for name in _walk_keys(payload))
        hit = next(
            (
                (token, name)
                for token in FORBIDDEN_DIRECT_OUTPUT_TOKENS
                for name in normalized
                if token in name
            ),
            None,
        )
        if hit is not None:
            raise RuntimeError(f"direct fast-path output token {hit[0]!r} entered payload")
        return payload


def _walk_keys(value: object) -> tuple[str, ...]:
    keys: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            keys.append(str(key))
            keys.extend(_walk_keys(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            keys.extend(_walk_keys(item))
    return tuple(keys)


@dataclass(frozen=True)
class HybridStateEstimatorConfig:
    minimum_samples: int = 128
    bootstrap_replicates: int = 128
    block_length_cycles: int = 16
    tail_threshold: float = 0.35 * LATTICE_CONST
    risk_horizon_cycles: int = 32
    max_recovery_depth: int = 6
    beta_prior_alpha: float = 0.5
    beta_prior_beta: float = 0.5
    ood_hold_threshold: float = 0.55
    minimum_regime_confidence: float = 0.38
    recommendation_ttl_cycles: int = 32
    calibration_scope: str = "uncalibrated_contract"
    bootstrap_seed: int = 2026071503
    param_mapper_config: ParamMapperConfig = ParamMapperConfig(
        alpha_bias=1.0,
        beta_smoothing=0.2,
        gain_clip=(0.2, 1.2),
        gain_scale=1.0,
        theta_clip_deg=(-90.0, 90.0),
        sigma_meas=0.03,
        delta_eff=0.30,
        sigma_ratio_p=1.0,
    )

    def __post_init__(self) -> None:
        for name, minimum in (
            ("minimum_samples", 8),
            ("bootstrap_replicates", 32),
            ("block_length_cycles", 2),
            ("risk_horizon_cycles", 1),
            ("max_recovery_depth", 1),
            ("recommendation_ttl_cycles", 1),
            ("bootstrap_seed", 0),
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum))
        if self.block_length_cycles > self.minimum_samples:
            raise ValueError("block_length_cycles cannot exceed minimum_samples")
        threshold = _finite(self.tail_threshold, "tail_threshold")
        if threshold <= 0.0:
            raise ValueError("tail_threshold must be positive")
        object.__setattr__(self, "tail_threshold", threshold)
        for name in ("beta_prior_alpha", "beta_prior_beta"):
            value = _finite(getattr(self, name), name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "ood_hold_threshold", _probability(self.ood_hold_threshold, "ood_hold_threshold"))
        object.__setattr__(
            self,
            "minimum_regime_confidence",
            _probability(self.minimum_regime_confidence, "minimum_regime_confidence"),
        )
        if self.calibration_scope not in CALIBRATION_SCOPES:
            raise ValueError(f"calibration_scope must be one of {CALIBRATION_SCOPES}")
        if not isinstance(self.param_mapper_config, ParamMapperConfig):
            raise TypeError("param_mapper_config must be ParamMapperConfig")


def _continuous_from_rows(rows: np.ndarray, config: HybridStateEstimatorConfig) -> ContinuousNoiseCalibration:
    index = {name: FEATURE_NAMES.index(name) for name in FEATURE_NAMES}
    residual = rows[:, [index["residual_q"], index["residual_p"]]]
    estimate = estimate_periodic_gaussian(
        residual,
        PeriodicMomentConfig(minimum_samples=min(config.minimum_samples, len(rows))),
        source="periodic_observed_moments",
        window_id=-1,
    )
    covariance = estimate.covariance_array()
    sigma_q = float(np.sqrt(covariance[0, 0]))
    sigma_p = float(np.sqrt(covariance[1, 1]))
    rho = float(covariance[0, 1] / (sigma_q * sigma_p))
    tail = float(np.mean(np.max(np.abs(residual), axis=1) >= config.tail_threshold))
    return ContinuousNoiseCalibration(
        mean_q=estimate.mean[0],
        mean_p=estimate.mean[1],
        sigma_q=sigma_q,
        sigma_p=sigma_p,
        rho_qp=float(np.clip(rho, -0.999999, 0.999999)),
        tail_rate=tail,
        x_e_rate=float(np.mean(rows[:, index["syndrome_x_e"]])),
        z_e_rate=float(np.mean(rows[:, index["syndrome_z_e"]])),
        leakage_rate=float(
            np.mean(
                np.maximum(
                    rows[:, index["syndrome_x_leakage"]],
                    rows[:, index["syndrome_z_leakage"]],
                )
            )
        ),
    )


def _moving_block_bootstrap(
    rows: np.ndarray,
    config: HybridStateEstimatorConfig,
    *,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    count = len(rows)
    block = min(config.block_length_cycles, count)
    starts_max = count - block
    replicates = np.empty((config.bootstrap_replicates, len(CONTINUOUS_PARAMETER_NAMES)))
    blocks_needed = int(np.ceil(count / block))
    offsets = np.arange(block)
    for replicate in range(config.bootstrap_replicates):
        starts = rng.integers(0, starts_max + 1, size=blocks_needed)
        indices = (starts[:, None] + offsets[None, :]).reshape(-1)[:count]
        replicates[replicate] = _continuous_from_rows(rows[indices], config).as_vector()
    return replicates


def _risk_from_rows(rows: np.ndarray, config: HybridStateEstimatorConfig) -> LeakageRecoveryOutput:
    index = {name: FEATURE_NAMES.index(name) for name in FEATURE_NAMES}
    leakage = np.maximum(
        rows[:, index["syndrome_x_leakage"]], rows[:, index["syndrome_z_leakage"]]
    )
    p = float(
        (np.sum(leakage) + config.beta_prior_alpha)
        / (len(rows) + config.beta_prior_alpha + config.beta_prior_beta)
    )
    horizon = float(1.0 - (1.0 - p) ** config.risk_horizon_cycles)
    last = rows[-1]
    current_burden = max(
        last[index["x_e_run"]],
        last[index["z_e_run"]],
        2.0 * last[index["leakage_run"]],
    )
    observed_rate = float(
        np.mean(
            np.maximum(rows[:, index["syndrome_x_e"]], rows[:, index["syndrome_z_e"]])
        )
    )
    center = min(
        config.max_recovery_depth,
        current_burden + 2.0 * observed_rate + 3.0 * p,
    )
    scale = 0.65 + 2.0 * p + observed_rate
    depths = np.arange(config.max_recovery_depth + 1, dtype=np.float64)
    weights = np.exp(-np.abs(depths - center) / scale)
    weights /= np.sum(weights)
    return LeakageRecoveryOutput(
        leakage_probability_next_cycle=p,
        leakage_probability_horizon=horizon,
        horizon_cycles=config.risk_horizon_cycles,
        recovery_burden_posterior=tuple(float(value) for value in weights),
        max_recovery_depth=config.max_recovery_depth,
    )


def _ood_score(rows: np.ndarray, regime: RegimePosteriorOutput) -> float:
    index = {name: FEATURE_NAMES.index(name) for name in FEATURE_NAMES}
    saturation_names = (
        "llr_q_saturated",
        "llr_p_saturated",
        "x_e_run_saturated",
        "z_e_run_saturated",
        "leakage_run_saturated",
        "active_bank_version_saturated",
        "pending_window_count_saturated",
    )
    saturation = float(np.max([np.mean(rows[:, index[name]]) for name in saturation_names]))
    health = float(
        max(
            1.0 - np.mean(rows[:, index["valid"]]),
            1.0 - np.mean(rows[:, index["crc_ok"]]),
            1.0 - np.mean(rows[:, index["fast_deadline_ok"]]),
            1.0 - np.mean(rows[:, index["slow_deadline_ok"]]),
            1.0 - np.mean(rows[:, index["communication_available"]]),
        )
    )
    posterior_uncertainty = 1.0 - regime.confidence
    return float(np.clip(max(saturation, 2.0 * health, posterior_uncertainty), 0.0, 1.0))


def _recommended_mode(rows: np.ndarray, risk: LeakageRecoveryOutput) -> str:
    index = {name: FEATURE_NAMES.index(name) for name in FEATURE_NAMES}
    last = rows[-1]
    # The horizon probability is a union over many future cycles and is not an
    # instantaneous hold signal.  Profile selection therefore uses current
    # observed runs plus next-cycle risk; otherwise even a small per-cycle rate
    # would spuriously force nearly every 32-cycle recommendation into hold.
    if risk.leakage_probability_next_cycle >= 0.15 or last[index["leakage_run"]] > 0.0:
        return LEAKAGE_HOLD
    x_run = last[index["x_e_run"]]
    z_run = last[index["z_e_run"]]
    if max(x_run, z_run) >= 2.0 or risk.expected_recovery_depth >= 3.0:
        if x_run >= z_run:
            return X_RECOVERY
        return Z_RECOVERY
    return NORMAL


class HybridStateEstimator:
    def __init__(self, config: HybridStateEstimatorConfig | None = None) -> None:
        self.config = HybridStateEstimatorConfig() if config is None else config
        if not isinstance(self.config, HybridStateEstimatorConfig):
            raise TypeError("config must be HybridStateEstimatorConfig")
        self._output_sequence = 0

    def estimate(
        self,
        history: ExperimentalHistorySample,
        regime_probabilities: Sequence[float],
        *,
        regime_source: str,
        active_params: DecoderRuntimeParams,
        active_bank_version: int,
    ) -> HybridStateOutput:
        if not isinstance(history, ExperimentalHistorySample):
            raise TypeError("history must be ExperimentalHistorySample")
        if not isinstance(active_params, DecoderRuntimeParams):
            raise TypeError("active_params must be DecoderRuntimeParams")
        active_version = _integer(active_bank_version, "active_bank_version")
        regime = RegimePosteriorOutput(tuple(regime_probabilities), regime_source)
        index = {name: FEATURE_NAMES.index(name) for name in FEATURE_NAMES}
        selected = history.values[history.mask == 1.0]
        selected = selected[selected[:, index["valid"]] == 1.0]
        if len(selected) < self.config.minimum_samples:
            raise ValueError("history contains fewer valid rows than minimum_samples")
        continuous = _continuous_from_rows(selected, self.config)
        risk = _risk_from_rows(selected, self.config)
        bootstrap = _moving_block_bootstrap(
            selected,
            self.config,
            seed=self.config.bootstrap_seed + history.end_cycle + 1_000_003 * self._output_sequence,
        )
        covariance = np.cov(bootstrap, rowvar=False, ddof=1)
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        covariance = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
        covariance = 0.5 * (covariance + covariance.T)
        ood = _ood_score(selected, regime)
        confidence = float(np.clip(regime.confidence * (1.0 - ood), 0.0, 1.0))
        uncertainty = UncertaintyOutput(
            continuous_covariance=tuple(tuple(float(value) for value in row) for row in covariance),
            sample_count=len(selected),
            bootstrap_replicates=self.config.bootstrap_replicates,
            block_length_cycles=self.config.block_length_cycles,
            regime_entropy_nats=regime.entropy_nats,
            leakage_entropy_nats=risk.leakage_entropy_nats,
            recovery_entropy_nats=risk.recovery_entropy_nats,
            ood_score=ood,
            recommendation_confidence=confidence,
            calibration_scope=self.config.calibration_scope,
        )

        physical_covariance = continuous.covariance_qp
        eigenvalues, eigenvectors = np.linalg.eigh(physical_covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        principal = eigenvectors[:, order[0]]
        theta_deg = float(np.degrees(np.arctan2(principal[1], principal[0])))
        while theta_deg > 90.0:
            theta_deg -= 180.0
        while theta_deg <= -90.0:
            theta_deg += 180.0
        ratio = float(np.sqrt(max(eigenvalues[1], 1.0e-12) / max(eigenvalues[0], 1.0e-12)))
        mapper = ParamMapper(replace(self.config.param_mapper_config, sigma_ratio_p=ratio))
        mapped = mapper.map_prediction(
            NoisePrediction(
                sigma=float(np.sqrt(eigenvalues[0])),
                mu_q=continuous.mean_q,
                mu_p=continuous.mean_p,
                theta_deg=theta_deg,
                source="hybrid_state_observed_estimate",
                metadata={"as_of_cycle": history.end_cycle, "schema": SCHEMA_VERSION},
            ),
            previous_params=active_params,
        )
        can_stage = (
            self.config.calibration_scope == "registered_synthetic_pilot"
            and ood <= self.config.ood_hold_threshold
            and regime.confidence >= self.config.minimum_regime_confidence
        )
        reasons = []
        if self.config.calibration_scope != "registered_synthetic_pilot":
            reasons.append("calibration_scope_not_registered")
        if ood > self.config.ood_hold_threshold:
            reasons.append("ood_gate")
        if regime.confidence < self.config.minimum_regime_confidence:
            reasons.append("regime_confidence_gate")
        identifier_payload = np.concatenate(
            (continuous.as_vector(), np.asarray(regime.probabilities), np.asarray([history.end_cycle, active_version]))
        ).astype("<f8")
        recommendation_id = "hs-" + hashlib.sha256(identifier_payload.tobytes()).hexdigest()[:20]
        recommendation = ParameterBankRecommendation.create(
            bank_action="stage_candidate" if can_stage else "hold_active",
            recommended_mode=_recommended_mode(selected, risk) if can_stage else FALLBACK,
            gain_matrix=mapped.K,
            bias=mapped.b,
            base_active_version=active_version,
            valid_from_cycle=history.end_cycle + 1,
            expires_after_cycle=history.end_cycle + self.config.recommendation_ttl_cycles,
            recommendation_id=recommendation_id,
            calibration_scope=self.config.calibration_scope,
            hold_reason=None if can_stage else "+".join(reasons),
        )
        valid_cycles = history.cycle_indices[history.mask == 1.0]
        output = HybridStateOutput(
            as_of_cycle=history.end_cycle,
            history_start_cycle=int(valid_cycles[0]),
            output_sequence=self._output_sequence,
            continuous=continuous,
            regime=regime,
            risk=risk,
            uncertainty=uncertainty,
            parameter_bank_recommendation=recommendation,
            provenance={
                "history_schema": history.schema_version,
                "continuous_source": continuous.source,
                "regime_source": regime.source,
                "risk_source": risk.source,
                "uncertainty_source": uncertainty.source,
                "parameter_mapping": recommendation.mapping_method,
                "future_only": True,
            },
        )
        self._output_sequence += 1
        return output


def stage_parameter_bank_recommendation(
    output: HybridStateOutput,
    param_bank: ParamBank,
    *,
    staged_cycle: int,
) -> PendingCommit:
    if not isinstance(output, HybridStateOutput):
        raise TypeError("output must be HybridStateOutput")
    if not isinstance(param_bank, ParamBank):
        raise TypeError("param_bank must be ParamBank")
    cycle = _integer(staged_cycle, "staged_cycle")
    recommendation = output.parameter_bank_recommendation
    if recommendation.bank_action != "stage_candidate":
        raise ValueError("hold_active recommendation must not be staged")
    if cycle != output.as_of_cycle:
        raise ValueError("recommendation must be staged at its as_of_cycle")
    if param_bank.active_version != recommendation.base_active_version:
        raise ValueError("base_active_version is stale relative to active parameter bank")
    if recommendation.valid_from_cycle > recommendation.expires_after_cycle:
        raise ValueError("recommendation validity interval is empty")
    return param_bank.stage_update(
        recommendation.to_runtime_params(),
        commit_epoch=recommendation.valid_from_cycle,
        staged_epoch=cycle,
        metadata={
            "recommendation_id": recommendation.recommendation_id,
            "payload_crc32": recommendation.payload_crc32,
            "output_sequence": output.output_sequence,
            "as_of_cycle": output.as_of_cycle,
        },
    )


def schema_provenance() -> dict[str, object]:
    fields = {
        "continuous": list(CONTINUOUS_PARAMETER_NAMES),
        "regime_posterior": list(REGIME_CLASSES),
        "leakage_recovery": [
            "leakage_probability_next_cycle",
            "leakage_probability_horizon",
            "recovery_burden_posterior",
        ],
        "uncertainty": [
            "continuous_covariance",
            "regime_entropy_nats",
            "leakage_entropy_nats",
            "recovery_entropy_nats",
            "ood_score",
            "recommendation_confidence",
        ],
        "parameter_bank_recommendation": [
            "bank_action",
            "recommended_mode",
            "gain_matrix",
            "bias",
            "base_active_version",
            "valid_from_cycle",
            "expires_after_cycle",
            "recommendation_id",
            "payload_crc32",
        ],
    }
    flattened = tuple(name for names in fields.values() for name in names)
    return {
        "schema_version": SCHEMA_VERSION,
        "fields": fields,
        "continuous_units": dict(CONTINUOUS_PARAMETER_UNITS),
        "forbidden_direct_output_tokens": list(FORBIDDEN_DIRECT_OUTPUT_TOKENS),
        "has_forbidden_direct_output": any(
            token in _normalized_key(name)
            for name in flattened
            for token in FORBIDDEN_DIRECT_OUTPUT_TOKENS
        ),
        "recovery_semantics": "observed-data recovery-burden posterior, not simulator recovery-depth truth",
        "bank_semantics": "future inactive-bank proposal with base version, validity interval and CRC; atomic commit remains ParamBank-owned",
        "hardware_measured": False,
    }


__all__ = [
    "SCHEMA_VERSION",
    "CONTINUOUS_PARAMETER_NAMES",
    "CONTINUOUS_PARAMETER_UNITS",
    "REGIME_POSTERIOR_SOURCES",
    "RISK_SOURCES",
    "CALIBRATION_SCOPES",
    "BANK_ACTIONS",
    "FORBIDDEN_DIRECT_OUTPUT_TOKENS",
    "ContinuousNoiseCalibration",
    "RegimePosteriorOutput",
    "LeakageRecoveryOutput",
    "UncertaintyOutput",
    "ParameterBankRecommendation",
    "HybridStateOutput",
    "HybridStateEstimatorConfig",
    "HybridStateEstimator",
    "stage_parameter_bank_recommendation",
    "schema_provenance",
]
