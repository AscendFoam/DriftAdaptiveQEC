"""Protocol-native autonomous versus measurement-feedback sBs channels.

The implementation shares the explicit finite-cutoff gates and idle CPTP maps
from :mod:`physics.differentiable_sbs_trajectory`, but it does not estimate an
autonomous curve by rescaling a measurement-feedback lifetime.  Each protocol
is propagated through its own reset window and number of cycles at a common
wall-clock horizon.

For the fixed nominal sBs controls used here, the measurement outcome is not
consumed by the next control.  Therefore the measurement-feedback expectation
can be evaluated exactly as a nonselective z measurement followed by reset,
rather than by noisy Monte Carlo trajectory averaging.  Autonomous sBs omits
the measurement event but still traces and resets the ancilla after layer 4.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, Literal, Mapping

import numpy as np

from physics.differentiable_sbs_trajectory import (
    PARAMETER_NAMES,
    POROTTI_S1_PROFILE_ID,
    DifferentiableSBSConfig,
    DifferentiableSBSTrajectorySimulator,
)

try:  # Minimal recovery interpreter has no torch.
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


ProtocolMode = Literal["measurement_feedback", "autonomous"]

MEASUREMENT_PROFILE_ID = POROTTI_S1_PROFILE_ID
AUTONOMOUS_PROFILE_ID = "PUVIANI-S4A-AUTONOMOUS-0P35-HALF-CYCLE"
PAPER_SOURCE = (
    "relative_papers/Non-Markovian_feedback_for_optimized_quantum_error_correction/"
    "Non-Markovian_feedback_for_optimized_quantum_error_correction.md"
)
MODEL_SCOPE = (
    "finite-cutoff nominal-control nonselective sBs channel with paper timing; "
    "instantaneous gates plus analytic idle CPTP maps and numerical trace-reset; "
    "not pulse-Hamiltonian, multilevel leakage, device-calibrated, target-board, "
    "or trained autonomous-controller evidence"
)


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("autonomous sBs requires the local DLEnv PyTorch environment")
    return torch


@dataclass(frozen=True)
class ProtocolTiming:
    profile_id: str
    entering_cycle_ns: int
    layer_1_ns: int
    layer_2_ns: int
    layer_3_ns: int
    layer_4_ns: int
    measurement_and_or_reset_ns: int
    virtual_rotation_and_idle_ns: int
    measurement_events_per_half_cycle: int
    reset_events_per_half_cycle: int = 1
    source: str = PAPER_SOURCE
    target_hardware_measured: bool = False

    def __post_init__(self) -> None:
        values = (
            self.entering_cycle_ns,
            self.layer_1_ns,
            self.layer_2_ns,
            self.layer_3_ns,
            self.layer_4_ns,
            self.measurement_and_or_reset_ns,
            self.virtual_rotation_and_idle_ns,
        )
        if any(isinstance(value, bool) or int(value) <= 0 for value in values):
            raise ValueError("all protocol timing phases must be positive integer nanoseconds")
        if self.measurement_events_per_half_cycle not in {0, 1}:
            raise ValueError("measurement_events_per_half_cycle must be zero or one")
        if self.reset_events_per_half_cycle != 1:
            raise ValueError("both registered protocols reset once per half-cycle")
        if self.target_hardware_measured:
            raise ValueError("literature timing must not be marked target-hardware measured")

    @property
    def half_cycle_duration_ns(self) -> int:
        return int(
            self.entering_cycle_ns
            + self.layer_1_ns
            + self.layer_2_ns
            + self.layer_3_ns
            + self.layer_4_ns
            + self.measurement_and_or_reset_ns
            + self.virtual_rotation_and_idle_ns
        )

    @property
    def full_cycle_duration_ns(self) -> int:
        return 2 * self.half_cycle_duration_ns


MEASUREMENT_TIMING = ProtocolTiming(
    profile_id=MEASUREMENT_PROFILE_ID,
    entering_cycle_ns=100,
    layer_1_ns=500,
    layer_2_ns=700,
    layer_3_ns=300,
    layer_4_ns=100,
    measurement_and_or_reset_ns=2300,
    virtual_rotation_and_idle_ns=1000,
    measurement_events_per_half_cycle=1,
)

AUTONOMOUS_TIMING = ProtocolTiming(
    profile_id=AUTONOMOUS_PROFILE_ID,
    entering_cycle_ns=100,
    layer_1_ns=500,
    layer_2_ns=700,
    layer_3_ns=300,
    layer_4_ns=100,
    measurement_and_or_reset_ns=800,
    virtual_rotation_and_idle_ns=1000,
    measurement_events_per_half_cycle=0,
)


@dataclass(frozen=True)
class NonselectiveSBSConfig:
    mode: ProtocolMode
    full_cycles: int
    cutoff: int = 12
    projector_delta: float = 0.34
    cavity_lifetime_us: float = 245.0
    ancilla_t1_us: float = 50.0
    ancilla_t2_us: float = 60.0
    device: Literal["cpu", "cuda"] = "cpu"
    real_dtype: Literal["float32", "float64"] = "float64"
    scope: str = MODEL_SCOPE

    def __post_init__(self) -> None:
        if self.mode not in {"measurement_feedback", "autonomous"}:
            raise ValueError("mode must be measurement_feedback or autonomous")
        for name in ("full_cycles", "cutoff"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        if not 4 <= self.cutoff <= 48:
            raise ValueError("cutoff must lie in [4, 48]")
        if self.full_cycles > 10_000:
            raise ValueError("full_cycles exceeds the explicit long-horizon safety guard")
        for name in ("projector_delta", "cavity_lifetime_us", "ancilla_t1_us", "ancilla_t2_us"):
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        if self.ancilla_t2_us > 2.0 * self.ancilla_t1_us + 1.0e-12:
            raise ValueError("ancilla_t2_us must not exceed 2*T1")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if self.real_dtype not in {"float32", "float64"}:
            raise ValueError("real_dtype must be float32 or float64")
        if self.scope != MODEL_SCOPE:
            raise ValueError("model scope must remain fail closed")

    @property
    def timing(self) -> ProtocolTiming:
        return MEASUREMENT_TIMING if self.mode == "measurement_feedback" else AUTONOMOUS_TIMING


@dataclass
class NonselectiveSBSResult:
    config: NonselectiveSBSConfig
    time_us: np.ndarray
    fidelity: np.ndarray
    code_survival: np.ndarray
    logical_z_signal: np.ndarray
    conditional_logical_z: np.ndarray
    final_cavity_density: Any
    physical_controls: Any
    event_accounting: Mapping[str, float | int]
    maximum_trace_error: float
    maximum_hermiticity_error: float
    minimum_final_eigenvalue: float

    def to_dict(self, *, include_density: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "config": {**asdict(self.config), "timing": asdict(self.config.timing)},
            "time_us": self.time_us.tolist(),
            "fidelity": self.fidelity.tolist(),
            "code_survival": self.code_survival.tolist(),
            "logical_z_signal": self.logical_z_signal.tolist(),
            "conditional_logical_z": self.conditional_logical_z.tolist(),
            "physical_controls": self.physical_controls.detach().cpu().tolist(),
            "event_accounting": dict(self.event_accounting),
            "maximum_trace_error": self.maximum_trace_error,
            "maximum_hermiticity_error": self.maximum_hermiticity_error,
            "minimum_final_eigenvalue": self.minimum_final_eigenvalue,
        }
        if include_density:
            density = self.final_cavity_density.detach().cpu().numpy()
            payload["final_cavity_density_real"] = density.real.tolist()
            payload["final_cavity_density_imag"] = density.imag.tolist()
        return payload


@dataclass(frozen=True)
class IdleMemoryConfig:
    """No-correction memory anchor on the same finite-cutoff cavity model.

    One ``full_cycle`` is a reporting interval only.  The channel contains
    cavity loss during that interval and deliberately contains no sBs gate,
    measurement, reset, frame update, or outcome-dependent action.
    """

    full_cycles: int
    cutoff: int = 12
    cycle_duration_us: float = 10.0
    projector_delta: float = 0.34
    cavity_lifetime_us: float = 245.0
    ancilla_t1_us: float = 50.0
    ancilla_t2_us: float = 60.0
    device: Literal["cpu", "cuda"] = "cpu"
    real_dtype: Literal["float32", "float64"] = "float64"

    def __post_init__(self) -> None:
        for name in ("full_cycles", "cutoff"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or int(value) <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        if not 4 <= self.cutoff <= 48:
            raise ValueError("cutoff must lie in [4, 48]")
        if self.full_cycles > 10_000:
            raise ValueError("full_cycles exceeds the explicit long-horizon safety guard")
        for name in (
            "cycle_duration_us",
            "projector_delta",
            "cavity_lifetime_us",
            "ancilla_t1_us",
            "ancilla_t2_us",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        if self.ancilla_t2_us > 2.0 * self.ancilla_t1_us + 1.0e-12:
            raise ValueError("ancilla_t2_us must not exceed 2*T1")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if self.real_dtype not in {"float32", "float64"}:
            raise ValueError("real_dtype must be float32 or float64")


@dataclass
class IdleMemoryResult:
    config: IdleMemoryConfig
    time_us: np.ndarray
    fidelity: np.ndarray
    code_survival: np.ndarray
    logical_z_signal: np.ndarray
    conditional_logical_z: np.ndarray
    final_cavity_density: Any
    event_accounting: Mapping[str, float | int]
    maximum_trace_error: float
    maximum_hermiticity_error: float
    minimum_final_eigenvalue: float

    def to_dict(self, *, include_density: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "config": asdict(self.config),
            "time_us": self.time_us.tolist(),
            "fidelity": self.fidelity.tolist(),
            "code_survival": self.code_survival.tolist(),
            "logical_z_signal": self.logical_z_signal.tolist(),
            "conditional_logical_z": self.conditional_logical_z.tolist(),
            "event_accounting": dict(self.event_accounting),
            "maximum_trace_error": self.maximum_trace_error,
            "maximum_hermiticity_error": self.maximum_hermiticity_error,
            "minimum_final_eigenvalue": self.minimum_final_eigenvalue,
        }
        if include_density:
            density = self.final_cavity_density.detach().cpu().numpy()
            payload["final_cavity_density_real"] = density.real.tolist()
            payload["final_cavity_density_imag"] = density.imag.tolist()
        return payload


class NonselectiveSBSSimulator:
    """Exact expected channel for fixed-control measurement or autonomous sBs."""

    def __init__(self, config: NonselectiveSBSConfig) -> None:
        th = _require_torch()
        if not isinstance(config, NonselectiveSBSConfig):
            raise TypeError("config must be a NonselectiveSBSConfig")
        self.config = config
        base = DifferentiableSBSConfig(
            cutoff=config.cutoff,
            full_cycles=1,
            batch_size=1,
            projector_delta=config.projector_delta,
            cavity_lifetime_us=config.cavity_lifetime_us,
            ancilla_t1_us=config.ancilla_t1_us,
            ancilla_t2_us=config.ancilla_t2_us,
            device=config.device,
            real_dtype=config.real_dtype,
        )
        self.engine = DifferentiableSBSTrajectorySimulator(base)
        self.reset_window_kraus = self.engine._joint_idle_kraus(
            config.timing.measurement_and_or_reset_ns / 1000.0
        )
        self.device = th.device(config.device)

    def _apply_reset_window(self, state: Any) -> Any:
        for operators in self.reset_window_kraus:
            state = self.engine._apply_kraus(state, operators)
        return self.engine._stabilize_density(state)

    def _trace_and_reset(self, state: Any) -> Any:
        th = _require_torch()
        cavity = self.engine._reduce_cavity(state)
        reset = th.einsum(
            "bij,kl->bikjl", cavity, self.engine.g_projector
        ).reshape(1, self.engine.joint_dimension, self.engine.joint_dimension)
        return self.engine._stabilize_density(reset)

    def run(self) -> NonselectiveSBSResult:
        th = _require_torch()
        state = self.engine._initial_joint_density()
        controls = self.engine.bounded_physical_controls(None)[:, 0, :]
        fidelity: list[Any] = []
        survival: list[Any] = []
        logical: list[Any] = []
        conditional: list[Any] = []

        def record() -> None:
            metrics = self.engine._cavity_evaluation_metrics(
                self.engine._reduce_cavity(state)
            )
            fidelity.append(metrics[0][0])
            survival.append(metrics[1][0])
            logical.append(metrics[2][0])
            conditional.append(metrics[3][0])

        maximum_trace = 0.0
        maximum_hermiticity = 0.0
        with th.no_grad():
            record()
            for half_index in range(2 * self.config.full_cycles):
                state = self.engine._apply_idle(state, "entering_cycle")
                for layer in range(1, 5):
                    state = self.engine._layer(state, controls, layer)
                state = self._apply_reset_window(state)
                state = self._trace_and_reset(state)
                state = self.engine._virtual_rotation(state, controls[:, 14])
                state = self.engine._apply_idle(state, "virtual_rotation_and_idle")
                trace_error, hermiticity_error, _ = self.engine._diagnostics(state)
                maximum_trace = max(maximum_trace, trace_error)
                maximum_hermiticity = max(maximum_hermiticity, hermiticity_error)
                if (half_index + 1) % 2 == 0:
                    record()
            final_trace, final_hermiticity, minimum_eigenvalue = self.engine._diagnostics(state)
        maximum_trace = max(maximum_trace, final_trace)
        maximum_hermiticity = max(maximum_hermiticity, final_hermiticity)

        cycles = self.config.full_cycles
        duration_us = self.config.timing.full_cycle_duration_ns / 1000.0
        half_cycles = 2 * cycles
        gate_breakdown = {
            "qubit_rotations": 8 * cycles,
            "echoed_conditional_displacements": 6 * cycles,
            "fixed_cavity_displacements": 2 * cycles,
            "virtual_rotations": 2 * cycles,
        }
        active_gates = int(sum(gate_breakdown.values()))
        total_time_us = duration_us * cycles
        measurements = self.config.timing.measurement_events_per_half_cycle * half_cycles
        resets = self.config.timing.reset_events_per_half_cycle * half_cycles
        accounting: dict[str, float | int] = {
            "full_cycles": cycles,
            "half_cycles": half_cycles,
            "total_physical_time_us": total_time_us,
            "measurement_events": measurements,
            "reset_events": resets,
            "active_gate_applications": active_gates,
            **gate_breakdown,
            "cycles_per_100us": 100.0 / duration_us,
            "measurements_per_100us": 100.0 * measurements / total_time_us,
            "resets_per_100us": 100.0 * resets / total_time_us,
            "active_gates_per_100us": 100.0 * active_gates / total_time_us,
            "outcome_dependent_parameter_updates": 0,
        }
        return NonselectiveSBSResult(
            config=self.config,
            time_us=np.arange(cycles + 1, dtype=np.float64) * duration_us,
            fidelity=th.stack(fidelity).detach().cpu().numpy().astype(np.float64),
            code_survival=th.stack(survival).detach().cpu().numpy().astype(np.float64),
            logical_z_signal=th.stack(logical).detach().cpu().numpy().astype(np.float64),
            conditional_logical_z=th.stack(conditional).detach().cpu().numpy().astype(np.float64),
            final_cavity_density=self.engine._reduce_cavity(state)[0],
            physical_controls=controls[0],
            event_accounting=accounting,
            maximum_trace_error=maximum_trace,
            maximum_hermiticity_error=maximum_hermiticity,
            minimum_final_eigenvalue=minimum_eigenvalue,
        )


class IdleMemorySimulator:
    """Exact no-correction cavity-memory anchor at a fixed time grid."""

    def __init__(self, config: IdleMemoryConfig) -> None:
        if not isinstance(config, IdleMemoryConfig):
            raise TypeError("config must be an IdleMemoryConfig")
        self.config = config
        base = DifferentiableSBSConfig(
            cutoff=config.cutoff,
            full_cycles=1,
            batch_size=1,
            projector_delta=config.projector_delta,
            cavity_lifetime_us=config.cavity_lifetime_us,
            ancilla_t1_us=config.ancilla_t1_us,
            ancilla_t2_us=config.ancilla_t2_us,
            device=config.device,
            real_dtype=config.real_dtype,
        )
        self.engine = DifferentiableSBSTrajectorySimulator(base)
        self.cycle_idle_kraus = self.engine._joint_idle_kraus(
            config.cycle_duration_us
        )

    def run(self) -> IdleMemoryResult:
        th = _require_torch()
        state = self.engine._initial_joint_density()
        fidelity: list[Any] = []
        survival: list[Any] = []
        logical: list[Any] = []
        conditional: list[Any] = []

        def record() -> None:
            metrics = self.engine._cavity_evaluation_metrics(
                self.engine._reduce_cavity(state)
            )
            fidelity.append(metrics[0][0])
            survival.append(metrics[1][0])
            logical.append(metrics[2][0])
            conditional.append(metrics[3][0])

        maximum_trace = 0.0
        maximum_hermiticity = 0.0
        with th.no_grad():
            record()
            for _ in range(self.config.full_cycles):
                for operators in self.cycle_idle_kraus:
                    state = self.engine._apply_kraus(state, operators)
                state = self.engine._stabilize_density(state)
                trace_error, hermiticity_error, _ = self.engine._diagnostics(state)
                maximum_trace = max(maximum_trace, trace_error)
                maximum_hermiticity = max(
                    maximum_hermiticity, hermiticity_error
                )
                record()
            final_trace, final_hermiticity, minimum_eigenvalue = (
                self.engine._diagnostics(state)
            )
        maximum_trace = max(maximum_trace, final_trace)
        maximum_hermiticity = max(maximum_hermiticity, final_hermiticity)

        cycles = self.config.full_cycles
        duration = self.config.cycle_duration_us
        total_time = cycles * duration
        accounting: dict[str, float | int] = {
            "full_cycles": cycles,
            "total_physical_time_us": total_time,
            "measurement_events": 0,
            "reset_events": 0,
            "active_gate_applications": 0,
            "frame_updates": 0,
            "outcome_dependent_parameter_updates": 0,
            "cycles_per_100us": 100.0 / duration,
            "measurements_per_100us": 0.0,
            "resets_per_100us": 0.0,
            "active_gates_per_100us": 0.0,
        }
        return IdleMemoryResult(
            config=self.config,
            time_us=np.arange(cycles + 1, dtype=np.float64) * duration,
            fidelity=th.stack(fidelity).detach().cpu().numpy().astype(np.float64),
            code_survival=th.stack(survival).detach().cpu().numpy().astype(np.float64),
            logical_z_signal=th.stack(logical).detach().cpu().numpy().astype(np.float64),
            conditional_logical_z=th.stack(conditional)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64),
            final_cavity_density=self.engine._reduce_cavity(state)[0],
            event_accounting=accounting,
            maximum_trace_error=maximum_trace,
            maximum_hermiticity_error=maximum_hermiticity,
            minimum_final_eigenvalue=minimum_eigenvalue,
        )


def finite_horizon_area_lifetime(time_us: np.ndarray, curve: np.ndarray) -> dict[str, float]:
    """Area-equivalent exponential lifetime on an explicit physical-time grid."""

    times = np.asarray(time_us, dtype=np.float64)
    values = np.asarray(curve, dtype=np.float64)
    if times.ndim != 1 or values.shape != times.shape or times.size < 3:
        raise ValueError("time and curve must be aligned rank-one arrays with >=3 points")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(values)):
        raise ValueError("time and curve must be finite")
    if times[0] != 0.0 or np.any(np.diff(times) <= 0.0):
        raise ValueError("time must start at zero and increase strictly")
    if abs(values[0]) <= 1.0e-14:
        raise ValueError("curve initial value must be nonzero")
    normalized = values / values[0]
    horizon = float(times[-1])
    normalized_auc = float(np.trapezoid(normalized, times) / horizon)
    if not 0.0 < normalized_auc <= 1.0 + 1.0e-9:
        raise ValueError("normalized signed area lies outside (0,1]")
    normalized_auc = min(normalized_auc, 1.0)
    if normalized_auc >= 1.0 - 1.0e-12:
        lifetime = 1.0e12 * horizon
    else:
        lower = 1.0e-12 * horizon
        upper = 1.0e12 * horizon
        for _ in range(160):
            middle = 0.5 * (lower + upper)
            area = (middle / horizon) * (1.0 - np.exp(-horizon / middle))
            if area < normalized_auc:
                lower = middle
            else:
                upper = middle
        lifetime = 0.5 * (lower + upper)
    return {
        "normalized_signed_auc": normalized_auc,
        "area_equivalent_lifetime_us": float(lifetime),
        "area_equivalent_lifetime_protocol_cycles": float(
            lifetime / (times[1] - times[0])
        ),
        "area_equivalent_lifetime_standard_10us_cycles": float(lifetime / 10.0),
        "horizon_us": horizon,
    }


def validate_timing_contract() -> Mapping[str, bool]:
    return {
        "measurement_half_cycle_is_5us": MEASUREMENT_TIMING.half_cycle_duration_ns == 5000,
        "autonomous_half_cycle_is_3p5us": AUTONOMOUS_TIMING.half_cycle_duration_ns == 3500,
        "autonomous_full_cycle_is_0p7_measurement_cycle": (
            AUTONOMOUS_TIMING.full_cycle_duration_ns * 10
            == MEASUREMENT_TIMING.full_cycle_duration_ns * 7
        ),
        "measurement_window_is_2p3us": MEASUREMENT_TIMING.measurement_and_or_reset_ns == 2300,
        "autonomous_reset_window_is_0p8us": AUTONOMOUS_TIMING.measurement_and_or_reset_ns == 800,
        "autonomous_has_no_measurement_but_keeps_reset": (
            AUTONOMOUS_TIMING.measurement_events_per_half_cycle == 0
            and AUTONOMOUS_TIMING.reset_events_per_half_cycle == 1
        ),
        "literature_timing_is_not_target_hardware_measurement": (
            not MEASUREMENT_TIMING.target_hardware_measured
            and not AUTONOMOUS_TIMING.target_hardware_measured
        ),
    }


__all__ = [
    "AUTONOMOUS_PROFILE_ID",
    "AUTONOMOUS_TIMING",
    "MEASUREMENT_PROFILE_ID",
    "MEASUREMENT_TIMING",
    "MODEL_SCOPE",
    "IdleMemoryConfig",
    "IdleMemoryResult",
    "IdleMemorySimulator",
    "NonselectiveSBSConfig",
    "NonselectiveSBSResult",
    "NonselectiveSBSSimulator",
    "PAPER_SOURCE",
    "ProtocolTiming",
    "finite_horizon_area_lifetime",
    "validate_timing_contract",
]
