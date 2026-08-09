"""连续 ``DriftState`` + 离散 regime 的 protocol-aligned syndrome stream。

T2.1.1 的 generator 在同一 causal cycle 中显式执行：loss attenuator -> correlated
Gaussian-mixture displacement -> modular syndrome -> measurement noise -> coarse sBs
recovery/leakage observation -> residual state 与 logical-truth update。

``ObservedSyndromeStep`` 与 ``SyndromeTruthStep`` 分离。deployable record 不包含
``DriftState``、outlier mask、leakage kind、recovery depth 或 logical truth。该模型是
syndrome-level effective stream，不是 cavity--transmon/Fock/device-calibrated simulator。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from .constants import LATTICE_CONST
from .drift_processes import DriftProcess, DriftState, sample_displacements
from .ideal_gkp_decoder import standard_binning_1d
from .sbs_error_space import SBS_PROTOCOL_ID
from .sbs_observation_reset import OBSERVED_CLASSES, PairedSyndrome
from ._shared.validation import finite as _finite
from ._shared.validation import finite_pair as _pair
from ._shared.validation import integer as _integer


MODEL_SCOPE = "protocol_aligned_mixed_state_syndrome_stream_not_device_calibrated"
CONSTITUENT_PHASES_RAD = (0.0, math.pi / 2.0)


def _probability(value: float, name: str) -> float:
    result = _finite(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def _wrap(values: np.ndarray, lattice: float) -> np.ndarray:
    wrapped = np.mod(values + 0.5 * lattice, lattice) - 0.5 * lattice
    # Preserve the repository half-open convention [-lattice/2, lattice/2).
    return np.where(wrapped >= 0.5 * lattice, wrapped - lattice, wrapped)


@dataclass(frozen=True)
class SyndromeStreamConfig:
    lattice: float = LATTICE_CONST
    measurement_sigma: tuple[float, float] = (0.02, 0.02)
    loss_environment_variance: float = 0.5
    max_recovery_depth: int = 6
    depth_probability_scale: float = 0.25
    depth_probability_power: float = 2.0
    recovery_probability: float = 0.88
    recovery_gain: float = 0.5
    base_leakage_probability: float = 1.0e-4
    loss_leakage_scale: float = 0.01
    burst_leakage_bonus: float = 0.01
    higher_leakage_fraction: float = 0.2
    higher_leakage_mean_duration: float = 10.0
    readout_fidelity_g: float = 0.9997
    readout_fidelity_e: float = 0.9914
    seed: int = 2026071411
    model_scope: str = MODEL_SCOPE

    def __post_init__(self) -> None:
        lattice = _finite(self.lattice, "lattice")
        if lattice <= 0.0:
            raise ValueError("lattice must be positive")
        object.__setattr__(self, "lattice", lattice)
        measurement = _pair(self.measurement_sigma, "measurement_sigma")
        if any(value < 0.0 for value in measurement):
            raise ValueError("measurement_sigma values must be non-negative")
        object.__setattr__(self, "measurement_sigma", measurement)
        environment = _finite(self.loss_environment_variance, "loss_environment_variance")
        if environment < 0.0:
            raise ValueError("loss_environment_variance must be non-negative")
        object.__setattr__(self, "loss_environment_variance", environment)
        object.__setattr__(
            self,
            "max_recovery_depth",
            _integer(self.max_recovery_depth, "max_recovery_depth", 1),
        )
        for name in (
            "depth_probability_scale",
            "recovery_probability",
            "recovery_gain",
            "base_leakage_probability",
            "higher_leakage_fraction",
            "readout_fidelity_g",
            "readout_fidelity_e",
        ):
            object.__setattr__(self, name, _probability(getattr(self, name), name))
        power = _finite(self.depth_probability_power, "depth_probability_power")
        if power <= 0.0:
            raise ValueError("depth_probability_power must be positive")
        object.__setattr__(self, "depth_probability_power", power)
        for name in ("loss_leakage_scale", "burst_leakage_bonus"):
            value = _finite(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        duration = _finite(self.higher_leakage_mean_duration, "higher_leakage_mean_duration")
        if duration < 2.0:
            raise ValueError("higher_leakage_mean_duration must be at least 2")
        object.__setattr__(self, "higher_leakage_mean_duration", duration)
        object.__setattr__(self, "seed", _integer(self.seed, "seed"))
        if self.seed >= 2**64:
            raise ValueError("seed must be smaller than 2**64")
        if self.model_scope != MODEL_SCOPE:
            raise ValueError(f"model_scope must be {MODEL_SCOPE!r}")


@dataclass(frozen=True)
class ObservedSyndromeStep:
    cycle_index: int
    drift_step: int
    time: float
    analog_syndrome: tuple[float, float]
    residual_syndrome: tuple[float, float]
    syndrome: PairedSyndrome
    quadrature_phases_rad: tuple[float, float]
    x_e_run: int
    z_e_run: int
    leakage_run: int
    valid: bool = True
    observation_scope: str = "deployable_observed_syndrome"

    def as_deployable_dict(self) -> dict[str, object]:
        return {
            "cycle_index": self.cycle_index,
            "drift_step": self.drift_step,
            "time": self.time,
            "analog_q": self.analog_syndrome[0],
            "analog_p": self.analog_syndrome[1],
            "residual_q": self.residual_syndrome[0],
            "residual_p": self.residual_syndrome[1],
            "syndrome_x": self.syndrome.x,
            "syndrome_z": self.syndrome.z,
            "phase_x_rad": self.quadrature_phases_rad[0],
            "phase_z_rad": self.quadrature_phases_rad[1],
            "x_e_run": self.x_e_run,
            "z_e_run": self.z_e_run,
            "leakage_run": self.leakage_run,
            "valid": self.valid,
            "observation_scope": self.observation_scope,
        }


@dataclass(frozen=True)
class SyndromeTruthStep:
    cycle_index: int
    drift_state: DriftState
    channel_displacement: tuple[float, float]
    outlier_component: bool
    loss_environment_noise: tuple[float, float]
    pre_measurement_shift: tuple[float, float]
    true_folded_syndrome: tuple[float, float]
    lattice_indices: tuple[int, int]
    logical_increment: tuple[int, int]
    true_logical_bits: tuple[int, int]
    true_logical_label: str
    hidden_regime: str
    burst_active: bool
    leakage_kind: str
    leakage_hazard: float
    recovery_quadrature: str | None
    recovery_quadrature_after_action: str | None
    previous_recovery_depth: int
    injected_recovery_depth: int
    recovery_depth_before_action: int
    recovery_depth_after_action: int
    recovery_succeeded: bool
    physical_residual_after_action: tuple[float, float]
    truth_scope: str = "simulator_hidden_truth_not_deployable_input"


@dataclass(frozen=True)
class SyndromeStreamStep:
    observed: ObservedSyndromeStep
    truth: SyndromeTruthStep


@dataclass(frozen=True)
class SyndromeStream:
    steps: tuple[SyndromeStreamStep, ...]
    seed: int
    final_physical_residual: tuple[float, float]
    final_logical_bits: tuple[int, int]
    protocol_id: str = SBS_PROTOCOL_ID
    model_scope: str = MODEL_SCOPE
    device_calibrated: bool = False

    def observed_records(self) -> tuple[dict[str, object], ...]:
        return tuple(step.observed.as_deployable_dict() for step in self.steps)

    def truth_records(self) -> tuple[Mapping[str, object], ...]:
        records: list[Mapping[str, object]] = []
        for step in self.steps:
            truth = step.truth
            records.append(
                MappingProxyType(
                    {
                        "cycle_index": truth.cycle_index,
                        "drift_state": truth.drift_state,
                        "channel_displacement": truth.channel_displacement,
                        "outlier_component": truth.outlier_component,
                        "loss_environment_noise": truth.loss_environment_noise,
                        "pre_measurement_shift": truth.pre_measurement_shift,
                        "true_folded_syndrome": truth.true_folded_syndrome,
                        "lattice_indices": truth.lattice_indices,
                        "logical_increment": truth.logical_increment,
                        "true_logical_bits": truth.true_logical_bits,
                        "true_logical_label": truth.true_logical_label,
                        "hidden_regime": truth.hidden_regime,
                        "burst_active": truth.burst_active,
                        "leakage_kind": truth.leakage_kind,
                        "leakage_hazard": truth.leakage_hazard,
                        "recovery_quadrature": truth.recovery_quadrature,
                        "recovery_quadrature_after_action": (
                            truth.recovery_quadrature_after_action
                        ),
                        "previous_recovery_depth": truth.previous_recovery_depth,
                        "injected_recovery_depth": truth.injected_recovery_depth,
                        "recovery_depth_before_action": (
                            truth.recovery_depth_before_action
                        ),
                        "recovery_depth": truth.recovery_depth_after_action,
                        "recovery_succeeded": truth.recovery_succeeded,
                        "physical_residual_after_action": (
                            truth.physical_residual_after_action
                        ),
                        "truth_scope": truth.truth_scope,
                    }
                )
            )
        return tuple(records)


def _logical_label(bits: tuple[int, int]) -> str:
    mapping = {(0, 0): "I", (1, 0): "X", (0, 1): "Z", (1, 1): "Y"}
    return mapping[bits]


def _resolve_states(
    source: DriftProcess | Sequence[DriftState], steps: int | None
) -> tuple[DriftState, ...]:
    if isinstance(source, DriftProcess):
        if steps is None:
            raise ValueError("steps is required when source is a DriftProcess")
        count = _integer(steps, "steps")
        states = tuple(source.generate(count))
    else:
        if isinstance(source, (str, bytes)):
            raise TypeError("source must be a DriftProcess or sequence of DriftState")
        try:
            states = tuple(source)
        except TypeError as exc:
            raise TypeError("source must be a DriftProcess or sequence of DriftState") from exc
        if steps is not None and _integer(steps, "steps") != len(states):
            raise ValueError("steps must equal the provided DriftState sequence length")
    if any(not isinstance(state, DriftState) for state in states):
        raise TypeError("every stream state must be a DriftState")
    return states


def _readout_class(
    ideal: str, config: SyndromeStreamConfig, rng: np.random.Generator
) -> str:
    if ideal == "g":
        return "g" if float(rng.random()) < config.readout_fidelity_g else "e"
    if ideal == "e":
        return "e" if float(rng.random()) < config.readout_fidelity_e else "g"
    raise ValueError("ideal readout class must be g or e")


def generate_syndrome_stream(
    source: DriftProcess | Sequence[DriftState],
    *,
    steps: int | None = None,
    config: SyndromeStreamConfig | None = None,
) -> SyndromeStream:
    """生成 causal mixed-state syndrome stream。

    ``source`` 可为完整 ``DriftProcess`` 或显式 ``DriftState`` sequence。process
    入口必须给 ``steps``；sequence 若同时给 ``steps`` 必须精确一致。
    """

    actual = SyndromeStreamConfig() if config is None else config
    if not isinstance(actual, SyndromeStreamConfig):
        raise TypeError("config must be a SyndromeStreamConfig or None")
    states = _resolve_states(source, steps)
    rng = np.random.default_rng(actual.seed)
    physical_residual = np.zeros(2, dtype=np.float64)
    logical_bits = np.zeros(2, dtype=np.int64)
    recovery_depth = 0
    recovery_quadrature: str | None = None
    leakage_remaining = 0
    leakage_kind = "none"
    x_e_run = 0
    z_e_run = 0
    leakage_run = 0
    records: list[SyndromeStreamStep] = []

    measurement_sigma = np.asarray(actual.measurement_sigma, dtype=np.float64)
    geometric_probability = 1.0 / (actual.higher_leakage_mean_duration - 1.0)

    for cycle_index, state in enumerate(states):
        channel, outlier_mask = sample_displacements(state, 1, rng=rng)
        eta = state.eta
        loss_sigma = math.sqrt((1.0 - eta) * actual.loss_environment_variance)
        loss_noise = rng.normal(0.0, loss_sigma, size=2)
        pre_measurement = math.sqrt(eta) * physical_residual + channel[0] + loss_noise

        q_decode = standard_binning_1d(pre_measurement[0], lattice=actual.lattice)
        p_decode = standard_binning_1d(pre_measurement[1], lattice=actual.lattice)
        true_folded = np.asarray([q_decode.syndrome, p_decode.syndrome], dtype=np.float64)
        lattice_indices = (int(q_decode.lattice_index), int(p_decode.lattice_index))
        logical_increment = (
            int(q_decode.logical_parity),
            int(p_decode.logical_parity),
        )
        logical_bits ^= np.asarray(logical_increment, dtype=np.int64)

        analog = true_folded + rng.normal(0.0, measurement_sigma, size=2)
        observed_residual = _wrap(analog, actual.lattice)

        severity = np.abs(true_folded) / (0.5 * actual.lattice)
        severity = np.clip(severity, 0.0, 1.0)
        depth_probability = actual.depth_probability_scale * float(np.max(severity)) ** (
            actual.depth_probability_power
        )
        if not 0.0 <= depth_probability <= 1.0:
            raise RuntimeError("derived recovery-depth probability left [0, 1]")
        injected_depth = int(
            rng.binomial(actual.max_recovery_depth, depth_probability)
        )
        previous_depth = recovery_depth
        if injected_depth > recovery_depth:
            recovery_depth = injected_depth
            recovery_quadrature = "X" if severity[0] >= severity[1] else "Z"

        leakage_hazard = (
            actual.base_leakage_probability
            + actual.loss_leakage_scale * (1.0 - eta)
            + (actual.burst_leakage_bonus if state.burst_active else 0.0)
        )
        if not 0.0 <= leakage_hazard <= 1.0:
            raise ValueError(
                "derived leakage hazard must lie in [0, 1]; reduce loss/burst leakage scales"
            )
        if leakage_remaining == 0 and float(rng.random()) < leakage_hazard:
            if float(rng.random()) < actual.higher_leakage_fraction:
                leakage_kind = "higher"
                leakage_remaining = 1 + int(rng.geometric(geometric_probability))
            else:
                leakage_kind = "f"
                leakage_remaining = 1

        active_leakage = leakage_remaining > 0
        depth_before_action = recovery_depth
        action_quadrature = recovery_quadrature
        recovery_succeeded = False
        if active_leakage:
            recovery_depth = min(actual.max_recovery_depth, recovery_depth + 1)
            if recovery_quadrature is None:
                recovery_quadrature = "X" if severity[0] >= severity[1] else "Z"
            syndrome = PairedSyndrome(x="leakage", z="leakage")
            leakage_remaining -= 1
            if leakage_remaining == 0:
                leakage_kind_after_record = leakage_kind
                leakage_kind = "none"
            else:
                leakage_kind_after_record = leakage_kind
        else:
            leakage_kind_after_record = "none"
            if recovery_depth > 0 and recovery_quadrature is not None:
                ideal_x = "e" if recovery_quadrature == "X" else "g"
                ideal_z = "e" if recovery_quadrature == "Z" else "g"
                syndrome = PairedSyndrome(
                    x=_readout_class(ideal_x, actual, rng),
                    z=_readout_class(ideal_z, actual, rng),
                )
                recovery_succeeded = float(rng.random()) < actual.recovery_probability
                if recovery_succeeded:
                    recovery_depth -= 1
                    if recovery_depth == 0:
                        recovery_quadrature = None
            else:
                syndrome = PairedSyndrome(
                    x=_readout_class("g", actual, rng),
                    z=_readout_class("g", actual, rng),
                )

        if syndrome.x == "e":
            x_e_run += 1
        else:
            x_e_run = 0
        if syndrome.z == "e":
            z_e_run += 1
        else:
            z_e_run = 0
        if "leakage" in syndrome.as_tuple():
            leakage_run += 1
        else:
            leakage_run = 0

        physical_residual = np.array(true_folded, dtype=np.float64, copy=True)
        if recovery_succeeded:
            corrected_axis = 0 if action_quadrature == "X" else 1
            physical_residual[corrected_axis] *= 1.0 - actual.recovery_gain
        bits_tuple = (int(logical_bits[0]), int(logical_bits[1]))
        observed = ObservedSyndromeStep(
            cycle_index=cycle_index,
            drift_step=state.step,
            time=state.time,
            analog_syndrome=(float(analog[0]), float(analog[1])),
            residual_syndrome=(float(observed_residual[0]), float(observed_residual[1])),
            syndrome=syndrome,
            quadrature_phases_rad=CONSTITUENT_PHASES_RAD,
            x_e_run=x_e_run,
            z_e_run=z_e_run,
            leakage_run=leakage_run,
        )
        truth = SyndromeTruthStep(
            cycle_index=cycle_index,
            drift_state=state,
            channel_displacement=(float(channel[0, 0]), float(channel[0, 1])),
            outlier_component=bool(outlier_mask[0]),
            loss_environment_noise=(float(loss_noise[0]), float(loss_noise[1])),
            pre_measurement_shift=(float(pre_measurement[0]), float(pre_measurement[1])),
            true_folded_syndrome=(float(true_folded[0]), float(true_folded[1])),
            lattice_indices=lattice_indices,
            logical_increment=logical_increment,
            true_logical_bits=bits_tuple,
            true_logical_label=_logical_label(bits_tuple),
            hidden_regime=state.regime,
            burst_active=state.burst_active,
            leakage_kind=leakage_kind_after_record,
            leakage_hazard=leakage_hazard,
            recovery_quadrature=action_quadrature,
            recovery_quadrature_after_action=recovery_quadrature,
            previous_recovery_depth=previous_depth,
            injected_recovery_depth=injected_depth,
            recovery_depth_before_action=depth_before_action,
            recovery_depth_after_action=recovery_depth,
            recovery_succeeded=recovery_succeeded,
            physical_residual_after_action=(
                float(physical_residual[0]),
                float(physical_residual[1]),
            ),
        )
        records.append(SyndromeStreamStep(observed=observed, truth=truth))

    return SyndromeStream(
        steps=tuple(records),
        seed=actual.seed,
        final_physical_residual=(float(physical_residual[0]), float(physical_residual[1])),
        final_logical_bits=(int(logical_bits[0]), int(logical_bits[1])),
    )
