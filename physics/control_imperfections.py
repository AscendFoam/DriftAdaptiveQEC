"""控制链与 active-correction imperfection 的因果 effective model。

因果顺序：

1. requested Cartesian displacement -> AWG polar amplitude/phase codes；
2. polar command -> DAC signed I/Q codes；
3. pulse gain/crosstalk/bias -> mean physical displacement；
4. latency drift/diffusion accumulates before the action；
5. multiplicative/additive active-displacement error is sampled；
6. virtual-rotation code, systematic calibration and stochastic angle error act
   in an explicitly selected order.

Controller-visible command/codes 与 simulator-only physical realization 使用不同
dataclass。所有参数均为显式 scenario assumptions；没有 device-calibrated defaults。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, exp, isfinite, pi, sin
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._shared.validation import integer as _integer
from .sbs_error_space import SBS_PROTOCOL_ID


CONTROL_IMPERFECTION_SCOPE = (
    "causal_control_imperfection_effective_model_not_device_calibrated"
)
ACTION_ORDERS = (
    "displacement_then_virtual_rotation",
    "virtual_rotation_then_displacement",
)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value.strip()


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite real number") from exc
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative(value: object, name: str) -> float:
    result = _finite(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _positive(value: object, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _pair(value: object, name: str) -> tuple[float, float]:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain two finite values") from exc
    if array.shape != (2,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain two finite values")
    return float(array[0]), float(array[1])


def _matrix(value: object, name: str) -> NDArray[np.float64]:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite 2x2 matrix") from exc
    if array.shape != (2, 2) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite 2x2 matrix")
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


def _covariance(value: object, name: str) -> NDArray[np.float64]:
    matrix = _matrix(value, name)
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"{name} must be symmetric")
    eigenvalues = np.linalg.eigvalsh(matrix)
    if float(np.min(eigenvalues)) < -1.0e-12:
        raise ValueError(f"{name} must be positive semidefinite")
    return matrix


def _seed(value: object) -> int:
    return _integer(value, "seed")


def _wrap_signed(angle: float) -> float:
    return float((angle + pi) % (2.0 * pi) - pi)


def _rotation(angle: float) -> NDArray[np.float64]:
    c = cos(angle)
    s = sin(angle)
    return np.asarray([[c, -s], [s, c]], dtype=np.float64)


def _freeze(array: NDArray[np.float64] | NDArray[np.int64]) -> None:
    array.setflags(write=False)


@dataclass(frozen=True)
class ControlImperfectionConfig:
    enable_quantization: bool
    awg_amplitude_bits: int
    awg_phase_bits: int
    awg_amplitude_full_scale: float
    dac_bits: int
    dac_full_scale: float
    virtual_rotation_bits: int
    pulse_gain_matrix: ArrayLike
    pulse_bias: tuple[float, float]
    active_relative_gain_sigma: float
    active_displacement_covariance: ArrayLike
    virtual_rotation_gain: float
    virtual_rotation_bias_rad: float
    virtual_rotation_noise_sigma_rad: float
    latency_drift_per_us: tuple[float, float]
    latency_diffusion_covariance_per_us: ArrayLike
    max_latency_us: float
    action_order: str
    quantization_provenance: str
    pulse_provenance: str
    latency_provenance: str
    model_scope: str = CONTROL_IMPERFECTION_SCOPE

    def __post_init__(self) -> None:
        if not isinstance(self.enable_quantization, bool):
            raise TypeError("enable_quantization must be bool")
        for name in (
            "awg_amplitude_bits",
            "awg_phase_bits",
            "dac_bits",
            "virtual_rotation_bits",
        ):
            bits = _integer(getattr(self, name), name, 2)
            if bits > 30:
                raise ValueError(f"{name} must be at most 30")
            object.__setattr__(self, name, bits)
        for name in ("awg_amplitude_full_scale", "dac_full_scale", "max_latency_us"):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        object.__setattr__(
            self,
            "pulse_gain_matrix",
            _matrix(self.pulse_gain_matrix, "pulse_gain_matrix"),
        )
        object.__setattr__(self, "pulse_bias", _pair(self.pulse_bias, "pulse_bias"))
        object.__setattr__(
            self,
            "active_relative_gain_sigma",
            _nonnegative(
                self.active_relative_gain_sigma,
                "active_relative_gain_sigma",
            ),
        )
        object.__setattr__(
            self,
            "active_displacement_covariance",
            _covariance(
                self.active_displacement_covariance,
                "active_displacement_covariance",
            ),
        )
        object.__setattr__(
            self,
            "virtual_rotation_gain",
            _finite(self.virtual_rotation_gain, "virtual_rotation_gain"),
        )
        object.__setattr__(
            self,
            "virtual_rotation_bias_rad",
            _finite(self.virtual_rotation_bias_rad, "virtual_rotation_bias_rad"),
        )
        object.__setattr__(
            self,
            "virtual_rotation_noise_sigma_rad",
            _nonnegative(
                self.virtual_rotation_noise_sigma_rad,
                "virtual_rotation_noise_sigma_rad",
            ),
        )
        object.__setattr__(
            self,
            "latency_drift_per_us",
            _pair(self.latency_drift_per_us, "latency_drift_per_us"),
        )
        object.__setattr__(
            self,
            "latency_diffusion_covariance_per_us",
            _covariance(
                self.latency_diffusion_covariance_per_us,
                "latency_diffusion_covariance_per_us",
            ),
        )
        if self.action_order not in ACTION_ORDERS:
            raise ValueError(f"action_order must be one of {ACTION_ORDERS}")
        for name in (
            "quantization_provenance",
            "pulse_provenance",
            "latency_provenance",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if self.model_scope != CONTROL_IMPERFECTION_SCOPE:
            raise ValueError(f"model_scope must be {CONTROL_IMPERFECTION_SCOPE!r}")


def ideal_control_imperfection_config(
    *,
    provenance: str = "explicit ideal-control ablation",
) -> ControlImperfectionConfig:
    source = _text(provenance, "provenance")
    return ControlImperfectionConfig(
        enable_quantization=False,
        awg_amplitude_bits=16,
        awg_phase_bits=16,
        awg_amplitude_full_scale=16.0,
        dac_bits=16,
        dac_full_scale=16.0,
        virtual_rotation_bits=16,
        pulse_gain_matrix=np.eye(2),
        pulse_bias=(0.0, 0.0),
        active_relative_gain_sigma=0.0,
        active_displacement_covariance=np.zeros((2, 2)),
        virtual_rotation_gain=1.0,
        virtual_rotation_bias_rad=0.0,
        virtual_rotation_noise_sigma_rad=0.0,
        latency_drift_per_us=(0.0, 0.0),
        latency_diffusion_covariance_per_us=np.zeros((2, 2)),
        max_latency_us=1.0e9,
        action_order="displacement_then_virtual_rotation",
        quantization_provenance=source,
        pulse_provenance=source,
        latency_provenance=source,
    )


@dataclass(frozen=True)
class ControlActionRequest:
    cycle_index: int
    correction_command: tuple[float, float]
    virtual_rotation_command_rad: float
    latency_us: float
    parameter_bank_version: int = 0
    protocol_id: str = SBS_PROTOCOL_ID

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "cycle_index",
            _integer(self.cycle_index, "cycle_index"),
        )
        object.__setattr__(
            self,
            "correction_command",
            _pair(self.correction_command, "correction_command"),
        )
        object.__setattr__(
            self,
            "virtual_rotation_command_rad",
            _finite(
                self.virtual_rotation_command_rad,
                "virtual_rotation_command_rad",
            ),
        )
        object.__setattr__(self, "latency_us", _nonnegative(self.latency_us, "latency_us"))
        object.__setattr__(
            self,
            "parameter_bank_version",
            _integer(self.parameter_bank_version, "parameter_bank_version"),
        )
        if self.protocol_id != SBS_PROTOCOL_ID:
            raise ValueError(f"protocol_id must be {SBS_PROTOCOL_ID!r}")


@dataclass(frozen=True)
class ControlActionRecord:
    cycle_index: int
    correction_requested: tuple[float, float]
    awg_amplitude_requested: float
    awg_amplitude_code: int | None
    awg_amplitude_command: float
    awg_phase_requested_rad: float
    awg_phase_code: int | None
    awg_phase_command_rad: float
    dac_iq_codes: tuple[int | None, int | None]
    correction_commanded: tuple[float, float]
    displacement_saturated: bool
    virtual_rotation_requested_rad: float
    virtual_rotation_code: int | None
    virtual_rotation_commanded_rad: float
    latency_us: float
    parameter_bank_version: int
    protocol_id: str = SBS_PROTOCOL_ID
    record_scope: str = "controller_visible_encoded_command_not_physical_realization"

    def as_deployable_dict(self) -> dict[str, object]:
        return {
            "cycle_index": self.cycle_index,
            "correction_requested": list(self.correction_requested),
            "awg_amplitude_requested": self.awg_amplitude_requested,
            "awg_amplitude_code": self.awg_amplitude_code,
            "awg_amplitude_command": self.awg_amplitude_command,
            "awg_phase_requested_rad": self.awg_phase_requested_rad,
            "awg_phase_code": self.awg_phase_code,
            "awg_phase_command_rad": self.awg_phase_command_rad,
            "dac_iq_codes": list(self.dac_iq_codes),
            "correction_commanded": list(self.correction_commanded),
            "displacement_saturated": self.displacement_saturated,
            "virtual_rotation_requested_rad": self.virtual_rotation_requested_rad,
            "virtual_rotation_code": self.virtual_rotation_code,
            "virtual_rotation_commanded_rad": self.virtual_rotation_commanded_rad,
            "latency_us": self.latency_us,
            "parameter_bank_version": self.parameter_bank_version,
            "protocol_id": self.protocol_id,
            "record_scope": self.record_scope,
        }


@dataclass(frozen=True)
class ControlActionTruth:
    residual_before_latency: tuple[float, float]
    latency_drift: tuple[float, float]
    latency_diffusion: tuple[float, float]
    residual_at_action: tuple[float, float]
    pulse_mean_displacement: tuple[float, float]
    active_relative_gain_error: float
    active_additive_error: tuple[float, float]
    actual_displacement: tuple[float, float]
    virtual_rotation_noise_rad: float
    actual_virtual_rotation_rad: float
    virtual_rotation_error_rad: float
    residual_after_action: tuple[float, float]
    quantization_provenance: str
    pulse_provenance: str
    latency_provenance: str
    truth_scope: str = "simulator_physical_control_truth_not_deployable_input"


@dataclass(frozen=True)
class ControlImperfectionStep:
    record: ControlActionRecord
    truth: ControlActionTruth

    def deployable_record(self) -> dict[str, object]:
        return self.record.as_deployable_dict()


@dataclass(frozen=True)
class ControlImperfectionTrajectory:
    steps: tuple[ControlImperfectionStep, ...]
    final_residual: tuple[float, float]
    seed: int
    protocol_id: str = SBS_PROTOCOL_ID

    def deployable_records(self) -> tuple[dict[str, object], ...]:
        return tuple(step.deployable_record() for step in self.steps)


@dataclass(frozen=True)
class ControlImperfectionBatch:
    record: ControlActionRecord
    residual_before_latency: NDArray[np.float64]
    latency_diffusion: NDArray[np.float64]
    active_relative_gain_error: NDArray[np.float64]
    active_additive_error: NDArray[np.float64]
    actual_displacement: NDArray[np.float64]
    virtual_rotation_noise_rad: NDArray[np.float64]
    residual_after_action: NDArray[np.float64]
    seed: int
    truth_scope: str = "vectorized_simulator_physical_control_truth"

    def __post_init__(self) -> None:
        samples = self.residual_after_action.shape[0]
        shapes = {
            "residual_before_latency": (samples, 2),
            "latency_diffusion": (samples, 2),
            "active_relative_gain_error": (samples,),
            "active_additive_error": (samples, 2),
            "actual_displacement": (samples, 2),
            "virtual_rotation_noise_rad": (samples,),
            "residual_after_action": (samples, 2),
        }
        for name, shape in shapes.items():
            value = getattr(self, name)
            if not isinstance(value, np.ndarray) or value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain finite values")
            _freeze(value)

    @property
    def samples(self) -> int:
        return int(self.residual_after_action.shape[0])

    @property
    def empirical_mean(self) -> NDArray[np.float64]:
        return np.mean(self.residual_after_action, axis=0)

    @property
    def empirical_covariance(self) -> NDArray[np.float64]:
        return np.cov(self.residual_after_action, rowvar=False, ddof=1)


class ControlImperfectionModel:
    """Requested command 到 hidden physical residual 的协议级 effective model。"""

    protocol_id = SBS_PROTOCOL_ID
    device_calibrated = False

    def __init__(self, config: ControlImperfectionConfig) -> None:
        if not isinstance(config, ControlImperfectionConfig):
            raise TypeError("config must be a ControlImperfectionConfig")
        self.config = config

    @staticmethod
    def _unsigned_quantize(
        value: float,
        *,
        bits: int,
        full_scale: float,
    ) -> tuple[int, float, bool]:
        maximum_code = (1 << bits) - 1
        clipped = min(max(value, 0.0), full_scale)
        code = int(np.rint(clipped / full_scale * maximum_code))
        dequantized = code * full_scale / maximum_code
        return code, float(dequantized), value > full_scale or value < 0.0

    @staticmethod
    def _phase_quantize(angle: float, *, bits: int) -> tuple[int, float]:
        levels = 1 << bits
        wrapped = angle % (2.0 * pi)
        code = int(np.rint(wrapped / (2.0 * pi) * levels)) % levels
        dequantized = code * (2.0 * pi) / levels
        return code, _wrap_signed(dequantized)

    @staticmethod
    def _signed_quantize(
        value: float,
        *,
        bits: int,
        full_scale: float,
    ) -> tuple[int, float, bool]:
        minimum_code = -(1 << (bits - 1))
        maximum_code = (1 << (bits - 1)) - 1
        step = full_scale / (1 << (bits - 1))
        raw_code = int(np.rint(value / step))
        code = min(max(raw_code, minimum_code), maximum_code)
        return code, float(code * step), raw_code != code

    def encode(self, request: ControlActionRequest) -> ControlActionRecord:
        if not isinstance(request, ControlActionRequest):
            raise TypeError("request must be a ControlActionRequest")
        if request.latency_us > self.config.max_latency_us:
            raise ValueError("request latency_us exceeds configured max_latency_us")
        requested = np.asarray(request.correction_command, dtype=np.float64)
        amplitude = float(np.linalg.norm(requested))
        phase = 0.0 if amplitude == 0.0 else _wrap_signed(atan2(requested[1], requested[0]))
        if self.config.enable_quantization:
            amp_code, amp_command, amp_sat = self._unsigned_quantize(
                amplitude,
                bits=self.config.awg_amplitude_bits,
                full_scale=self.config.awg_amplitude_full_scale,
            )
            phase_code, phase_command = self._phase_quantize(
                phase,
                bits=self.config.awg_phase_bits,
            )
            polar = np.asarray(
                [
                    amp_command * cos(phase_command),
                    amp_command * sin(phase_command),
                ],
                dtype=np.float64,
            )
            q_code, q_command, q_sat = self._signed_quantize(
                float(polar[0]),
                bits=self.config.dac_bits,
                full_scale=self.config.dac_full_scale,
            )
            p_code, p_command, p_sat = self._signed_quantize(
                float(polar[1]),
                bits=self.config.dac_bits,
                full_scale=self.config.dac_full_scale,
            )
            vr_code, vr_command = self._phase_quantize(
                request.virtual_rotation_command_rad,
                bits=self.config.virtual_rotation_bits,
            )
        else:
            amp_code = phase_code = q_code = p_code = vr_code = None
            amp_command = amplitude
            phase_command = phase
            q_command, p_command = request.correction_command
            vr_command = request.virtual_rotation_command_rad
            amp_sat = q_sat = p_sat = False
        return ControlActionRecord(
            cycle_index=request.cycle_index,
            correction_requested=request.correction_command,
            awg_amplitude_requested=amplitude,
            awg_amplitude_code=amp_code,
            awg_amplitude_command=amp_command,
            awg_phase_requested_rad=phase,
            awg_phase_code=phase_code,
            awg_phase_command_rad=phase_command,
            dac_iq_codes=(q_code, p_code),
            correction_commanded=(float(q_command), float(p_command)),
            displacement_saturated=bool(amp_sat or q_sat or p_sat),
            virtual_rotation_requested_rad=request.virtual_rotation_command_rad,
            virtual_rotation_code=vr_code,
            virtual_rotation_commanded_rad=vr_command,
            latency_us=request.latency_us,
            parameter_bank_version=request.parameter_bank_version,
        )

    def _deterministic_terms(
        self,
        record: ControlActionRecord,
        residual_before_latency: tuple[float, float],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float]:
        pre = np.asarray(residual_before_latency, dtype=np.float64)
        drift = (
            np.asarray(self.config.latency_drift_per_us, dtype=np.float64)
            * record.latency_us
        )
        command = np.asarray(record.correction_commanded, dtype=np.float64)
        pulse_mean = (
            np.asarray(self.config.pulse_gain_matrix) @ command
            + np.asarray(self.config.pulse_bias)
        )
        angle_mean = (
            self.config.virtual_rotation_gain * record.virtual_rotation_commanded_rad
            + self.config.virtual_rotation_bias_rad
            - record.virtual_rotation_requested_rad
        )
        return pre, drift, pulse_mean, float(angle_mean)

    def _step_with_rng(
        self,
        request: ControlActionRequest,
        residual_before_latency: tuple[float, float],
        rng: np.random.Generator,
    ) -> ControlImperfectionStep:
        pre_pair = _pair(residual_before_latency, "residual_before_latency")
        record = self.encode(request)
        pre, drift, pulse_mean, angle_mean = self._deterministic_terms(record, pre_pair)
        latency_covariance = (
            np.asarray(self.config.latency_diffusion_covariance_per_us)
            * record.latency_us
        )
        latency_diffusion = rng.multivariate_normal(np.zeros(2), latency_covariance)
        relative_gain = float(rng.normal(0.0, self.config.active_relative_gain_sigma))
        active_additive = rng.multivariate_normal(
            np.zeros(2),
            np.asarray(self.config.active_displacement_covariance),
        )
        actual_displacement = pulse_mean * (1.0 + relative_gain) + active_additive
        residual_at_action = pre + drift + latency_diffusion
        rotation_noise = float(
            rng.normal(0.0, self.config.virtual_rotation_noise_sigma_rad)
        )
        angle_error = angle_mean + rotation_noise
        if self.config.action_order == "displacement_then_virtual_rotation":
            residual_after = _rotation(angle_error) @ (
                residual_at_action - actual_displacement
            )
        else:
            residual_after = (
                _rotation(angle_error) @ residual_at_action - actual_displacement
            )
        truth = ControlActionTruth(
            residual_before_latency=pre_pair,
            latency_drift=(float(drift[0]), float(drift[1])),
            latency_diffusion=(
                float(latency_diffusion[0]),
                float(latency_diffusion[1]),
            ),
            residual_at_action=(
                float(residual_at_action[0]),
                float(residual_at_action[1]),
            ),
            pulse_mean_displacement=(
                float(pulse_mean[0]),
                float(pulse_mean[1]),
            ),
            active_relative_gain_error=relative_gain,
            active_additive_error=(
                float(active_additive[0]),
                float(active_additive[1]),
            ),
            actual_displacement=(
                float(actual_displacement[0]),
                float(actual_displacement[1]),
            ),
            virtual_rotation_noise_rad=rotation_noise,
            actual_virtual_rotation_rad=(
                self.config.virtual_rotation_gain
                * record.virtual_rotation_commanded_rad
                + self.config.virtual_rotation_bias_rad
                + rotation_noise
            ),
            virtual_rotation_error_rad=angle_error,
            residual_after_action=(
                float(residual_after[0]),
                float(residual_after[1]),
            ),
            quantization_provenance=self.config.quantization_provenance,
            pulse_provenance=self.config.pulse_provenance,
            latency_provenance=self.config.latency_provenance,
        )
        return ControlImperfectionStep(record=record, truth=truth)

    def step(
        self,
        request: ControlActionRequest,
        *,
        residual_before_latency: tuple[float, float],
        seed: int,
    ) -> ControlImperfectionStep:
        return self._step_with_rng(
            request,
            residual_before_latency,
            np.random.default_rng(_seed(seed)),
        )

    def simulate(
        self,
        requests: Sequence[ControlActionRequest],
        *,
        initial_residual: tuple[float, float],
        seed: int,
    ) -> ControlImperfectionTrajectory:
        if isinstance(requests, (str, bytes)):
            raise TypeError("requests must be a sequence, not text")
        items = tuple(requests)
        if any(not isinstance(item, ControlActionRequest) for item in items):
            raise TypeError("all requests must be ControlActionRequest")
        if any(
            later.cycle_index != earlier.cycle_index + 1
            for earlier, later in zip(items, items[1:])
        ):
            raise ValueError("request cycle_index values must be consecutive")
        residual = _pair(initial_residual, "initial_residual")
        rng = np.random.default_rng(_seed(seed))
        steps: list[ControlImperfectionStep] = []
        for request in items:
            step = self._step_with_rng(request, residual, rng)
            steps.append(step)
            residual = step.truth.residual_after_action
        return ControlImperfectionTrajectory(
            steps=tuple(steps),
            final_residual=residual,
            seed=_seed(seed),
        )

    def sample_fixed_request(
        self,
        request: ControlActionRequest,
        *,
        residual_before_latency: tuple[float, float],
        samples: int,
        seed: int,
    ) -> ControlImperfectionBatch:
        count = _integer(samples, "samples", 1)
        pre_pair = _pair(residual_before_latency, "residual_before_latency")
        record = self.encode(request)
        pre, drift, pulse_mean, angle_mean = self._deterministic_terms(record, pre_pair)
        sequences = np.random.SeedSequence(_seed(seed)).spawn(4)
        latency = np.random.default_rng(sequences[0]).multivariate_normal(
            np.zeros(2),
            np.asarray(self.config.latency_diffusion_covariance_per_us)
            * record.latency_us,
            size=count,
        )
        gain = np.random.default_rng(sequences[1]).normal(
            0.0,
            self.config.active_relative_gain_sigma,
            size=count,
        )
        active = np.random.default_rng(sequences[2]).multivariate_normal(
            np.zeros(2),
            np.asarray(self.config.active_displacement_covariance),
            size=count,
        )
        rotation_noise = np.random.default_rng(sequences[3]).normal(
            0.0,
            self.config.virtual_rotation_noise_sigma_rad,
            size=count,
        )
        actual = pulse_mean[None, :] * (1.0 + gain[:, None]) + active
        residual_at_action = pre[None, :] + drift[None, :] + latency
        angle = angle_mean + rotation_noise
        c = np.cos(angle)
        s = np.sin(angle)
        if self.config.action_order == "displacement_then_virtual_rotation":
            before_rotation = residual_at_action - actual
            after = np.column_stack(
                (
                    c * before_rotation[:, 0] - s * before_rotation[:, 1],
                    s * before_rotation[:, 0] + c * before_rotation[:, 1],
                )
            )
        else:
            rotated_state = np.column_stack(
                (
                    c * residual_at_action[:, 0] - s * residual_at_action[:, 1],
                    s * residual_at_action[:, 0] + c * residual_at_action[:, 1],
                )
            )
            after = rotated_state - actual
        before = np.broadcast_to(pre, (count, 2)).copy()
        return ControlImperfectionBatch(
            record=record,
            residual_before_latency=before,
            latency_diffusion=np.asarray(latency),
            active_relative_gain_error=np.asarray(gain),
            active_additive_error=np.asarray(active),
            actual_displacement=np.asarray(actual),
            virtual_rotation_noise_rad=np.asarray(rotation_noise),
            residual_after_action=np.asarray(after),
            seed=_seed(seed),
        )

    @staticmethod
    def _rotated_gaussian_moments(
        mean: NDArray[np.float64],
        covariance: NDArray[np.float64],
        angle_mean: float,
        angle_sigma: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        rotation_mean = exp(-0.5 * angle_sigma**2) * _rotation(angle_mean)
        output_mean = rotation_mean @ mean
        second = covariance + np.outer(mean, mean)
        a = float(second[0, 0])
        b = float(second[0, 1])
        d = float(second[1, 1])
        cos2 = exp(-2.0 * angle_sigma**2) * cos(2.0 * angle_mean)
        sin2 = exp(-2.0 * angle_sigma**2) * sin(2.0 * angle_mean)
        half_trace = 0.5 * (a + d)
        half_difference = 0.5 * (a - d)
        rotated_second = np.asarray(
            [
                [
                    half_trace + half_difference * cos2 - b * sin2,
                    half_difference * sin2 + b * cos2,
                ],
                [
                    half_difference * sin2 + b * cos2,
                    half_trace - half_difference * cos2 + b * sin2,
                ],
            ]
        )
        output_covariance = rotated_second - np.outer(output_mean, output_mean)
        return output_mean, output_covariance

    def analytic_residual_moments(
        self,
        request: ControlActionRequest,
        *,
        residual_before_latency: tuple[float, float],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        pre_pair = _pair(residual_before_latency, "residual_before_latency")
        record = self.encode(request)
        pre, drift, pulse_mean, angle_mean = self._deterministic_terms(record, pre_pair)
        latency_covariance = (
            np.asarray(self.config.latency_diffusion_covariance_per_us)
            * record.latency_us
        )
        active_covariance = (
            np.asarray(self.config.active_displacement_covariance)
            + self.config.active_relative_gain_sigma**2
            * np.outer(pulse_mean, pulse_mean)
        )
        angle_sigma = self.config.virtual_rotation_noise_sigma_rad
        if self.config.action_order == "displacement_then_virtual_rotation":
            mean_before_rotation = pre + drift - pulse_mean
            covariance_before_rotation = latency_covariance + active_covariance
            return self._rotated_gaussian_moments(
                mean_before_rotation,
                covariance_before_rotation,
                angle_mean,
                angle_sigma,
            )
        rotated_mean, rotated_covariance = self._rotated_gaussian_moments(
            pre + drift,
            latency_covariance,
            angle_mean,
            angle_sigma,
        )
        return rotated_mean - pulse_mean, rotated_covariance + active_covariance


@dataclass(frozen=True)
class ControlImperfectionValidationResult:
    samples: int
    seed: int
    empirical_mean: tuple[float, float]
    analytic_mean: tuple[float, float]
    maximum_mean_z_score: float
    empirical_covariance: tuple[tuple[float, float], tuple[float, float]]
    analytic_covariance: tuple[tuple[float, float], tuple[float, float]]
    covariance_relative_frobenius_error: float
    quantization_bits: tuple[int, ...]
    quantization_rms_error: tuple[float, ...]
    pulse_systematic_displacement_error_norm: float
    virtual_rotation_systematic_error_rad: float
    latency_covariance_trace: tuple[float, ...]
    ideal_endpoint_max_abs_residual: float
    checks: Mapping[str, bool]
    evidence_scope: str = CONTROL_IMPERFECTION_SCOPE

    def as_dict(self) -> dict[str, object]:
        return {
            "samples": self.samples,
            "seed": self.seed,
            "mean": {
                "empirical": list(self.empirical_mean),
                "analytic": list(self.analytic_mean),
                "maximum_z_score": self.maximum_mean_z_score,
            },
            "covariance": {
                "empirical": [list(row) for row in self.empirical_covariance],
                "analytic": [list(row) for row in self.analytic_covariance],
                "relative_frobenius_error": self.covariance_relative_frobenius_error,
            },
            "quantization_sweep": {
                "bits": list(self.quantization_bits),
                "rms_displacement_error": list(self.quantization_rms_error),
            },
            "pulse_systematic_displacement_error_norm": (
                self.pulse_systematic_displacement_error_norm
            ),
            "virtual_rotation_systematic_error_rad": (
                self.virtual_rotation_systematic_error_rad
            ),
            "latency_covariance_trace": list(self.latency_covariance_trace),
            "ideal_endpoint_max_abs_residual": self.ideal_endpoint_max_abs_residual,
            "checks": {name: bool(value) for name, value in self.checks.items()},
            "evidence_scope": self.evidence_scope,
            "claim_boundary": {
                "allowed": "causal command-encoding and physical-control imperfection sensitivity",
                "forbidden": "device-calibrated DAC/AWG precision, pulse fidelity, hard-real-time latency or microwave-chain measurement",
            },
        }


def run_control_imperfection_validation(
    *,
    samples: int = 100_000,
    seed: int = 2026071423,
) -> ControlImperfectionValidationResult:
    from ._control_imperfections.validation import (
        run_control_imperfection_validation as run_validation,
    )

    return run_validation(samples=samples, seed=seed)


def write_control_imperfection_validation(
    result: ControlImperfectionValidationResult,
    output_path: str | Path,
) -> Path:
    from ._control_imperfections.validation import (
        write_control_imperfection_validation as write_validation,
    )

    return write_validation(result, output_path)


def _main() -> None:
    from ._control_imperfections.validation import main

    main()


if __name__ == "__main__":
    _main()
