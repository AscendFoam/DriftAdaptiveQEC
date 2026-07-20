"""Six-state finite-cutoff logical-channel tomography for matched QEC on/off lanes.

The reconstructed object is the *unnormalized code-space subchannel*

``rho_L -> V^dagger E(V rho_L V^dagger) V``.

It is completely positive and trace non-increasing (CPTNI), so leakage remains
in the missing trace instead of being hidden by post-selection.  The Pauli
transfer matrix (PTM) is therefore linear and can expose non-Pauli mixing,
non-unital code-space flow and state-dependent survival.  Conditional Bloch
vectors are reported only as diagnostics and are never used to reconstruct the
channel or its lifetimes.

Both ``qec_on`` and ``qec_off`` use the same orthonormal finite-cutoff GKP code
basis, cavity-loss model, cutoff, initial states and 10 us reporting interval.
The on lane executes the explicit nominal sBs gate/reset channel; the off lane
contains only the matched-duration idle channel.  This is a simulation channel,
not a multilevel device model, experimental tomography or hardware timing result.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp, isfinite
from typing import Any, Literal, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from physics.autonomous_sbs import MEASUREMENT_TIMING
from physics.differentiable_sbs_trajectory import (
    DifferentiableSBSConfig,
    DifferentiableSBSTrajectorySimulator,
)

try:  # Minimal recovery interpreter does not require PyTorch.
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


ChannelMode = Literal["qec_on", "qec_off"]

STATE_LABELS = (
    "x_plus",
    "x_minus",
    "y_plus",
    "y_minus",
    "z_plus",
    "z_minus",
)
AXIS_LABELS = ("X", "Y", "Z")
PAULI_LABELS = ("I", "X", "Y", "Z")
AXIS_STATE_PAIRS: Mapping[str, tuple[str, str]] = {
    "X": ("x_plus", "x_minus"),
    "Y": ("y_plus", "y_minus"),
    "Z": ("z_plus", "z_minus"),
}

_I2 = np.eye(2, dtype=np.complex128)
_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
PAULIS = np.stack((_I2, _X, _Y, _Z), axis=0)

MODEL_SCOPE = (
    "finite-cutoff six-state CPTNI logical subchannel tomography on a matched "
    "nominal-sBs versus idle cavity model; no postselection, multilevel leakage, "
    "device calibration, experimental tomography, target-board timing or LER claim"
)


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("finite-Fock logical tomography requires PyTorch")
    return torch


def logical_eigenstate_density(label: str) -> NDArray[np.complex128]:
    states = {
        "x_plus": 0.5 * (_I2 + _X),
        "x_minus": 0.5 * (_I2 - _X),
        "y_plus": 0.5 * (_I2 + _Y),
        "y_minus": 0.5 * (_I2 - _Y),
        "z_plus": 0.5 * (_I2 + _Z),
        "z_minus": 0.5 * (_I2 - _Z),
    }
    if label not in states:
        raise ValueError(f"unknown logical eigenstate label: {label}")
    return states[label].copy()


def _validated_output_matrix(value: ArrayLike, label: str) -> NDArray[np.complex128]:
    matrix = np.asarray(value, dtype=np.complex128)
    if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"output {label} must be a finite 2x2 matrix")
    if np.linalg.norm(matrix - matrix.conj().T, ord="fro") > 2.0e-9:
        raise ValueError(f"output {label} must be Hermitian")
    matrix = 0.5 * (matrix + matrix.conj().T)
    eigenvalues = np.linalg.eigvalsh(matrix)
    trace = float(np.trace(matrix).real)
    if float(np.min(eigenvalues)) < -2.0e-9:
        raise ValueError(f"output {label} must be positive semidefinite")
    if trace < -2.0e-9 or trace > 1.0 + 2.0e-9:
        raise ValueError(f"output {label} trace must lie in [0,1]")
    return matrix


@dataclass(frozen=True)
class SubchannelTomography:
    ptm: NDArray[np.float64]
    choi: NDArray[np.complex128]
    pair_sum_linearity_residual: float
    minimum_choi_eigenvalue: float
    tni_effect_eigenvalues: tuple[float, float]
    maximum_output_hermiticity_error: float
    minimum_output_eigenvalue: float
    minimum_survival: float
    maximum_survival: float
    mean_leakage: float
    survival_spread: float
    off_diagonal_pauli_norm: float
    coherent_rotation_norm: float
    nonunital_code_flow_norm: float
    state_dependent_survival_norm: float

    @property
    def passed_physicality(self) -> bool:
        return (
            self.pair_sum_linearity_residual <= 2.0e-8
            and self.minimum_choi_eigenvalue >= -2.0e-8
            and self.tni_effect_eigenvalues[0] >= -2.0e-8
            and self.tni_effect_eigenvalues[1] <= 1.0 + 2.0e-8
            and self.maximum_output_hermiticity_error <= 2.0e-9
            and self.minimum_output_eigenvalue >= -2.0e-9
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ptm": self.ptm.tolist(),
            "choi_real": self.choi.real.tolist(),
            "choi_imag": self.choi.imag.tolist(),
            "pair_sum_linearity_residual": self.pair_sum_linearity_residual,
            "minimum_choi_eigenvalue": self.minimum_choi_eigenvalue,
            "tni_effect_eigenvalues": list(self.tni_effect_eigenvalues),
            "maximum_output_hermiticity_error": self.maximum_output_hermiticity_error,
            "minimum_output_eigenvalue": self.minimum_output_eigenvalue,
            "minimum_survival": self.minimum_survival,
            "maximum_survival": self.maximum_survival,
            "mean_leakage": self.mean_leakage,
            "survival_spread": self.survival_spread,
            "off_diagonal_pauli_norm": self.off_diagonal_pauli_norm,
            "coherent_rotation_norm": self.coherent_rotation_norm,
            "nonunital_code_flow_norm": self.nonunital_code_flow_norm,
            "state_dependent_survival_norm": self.state_dependent_survival_norm,
            "passed_physicality": self.passed_physicality,
        }


def reconstruct_code_subchannel(
    outputs: Mapping[str, ArrayLike],
) -> SubchannelTomography:
    """Reconstruct a linear CPTNI logical subchannel from six unnormalized outputs."""

    if set(outputs) != set(STATE_LABELS):
        missing = sorted(set(STATE_LABELS) - set(outputs))
        extra = sorted(set(outputs) - set(STATE_LABELS))
        raise ValueError(f"six-state outputs mismatch; missing={missing}, extra={extra}")
    matrices = {
        label: _validated_output_matrix(outputs[label], label) for label in STATE_LABELS
    }
    pair_sums = []
    for axis in AXIS_LABELS:
        plus, minus = AXIS_STATE_PAIRS[axis]
        pair_sums.append(matrices[plus] + matrices[minus])
    e_i = sum(pair_sums) / 3.0
    pair_residual = max(
        float(np.linalg.norm(value - e_i, ord="fro")) for value in pair_sums
    )
    e_paulis = [e_i]
    for axis in AXIS_LABELS:
        plus, minus = AXIS_STATE_PAIRS[axis]
        e_paulis.append(matrices[plus] - matrices[minus])

    ptm = np.empty((4, 4), dtype=np.float64)
    for row, pauli in enumerate(PAULIS):
        for column, output in enumerate(e_paulis):
            value = 0.5 * np.trace(pauli @ output)
            if abs(float(value.imag)) > 2.0e-9:
                raise RuntimeError("PTM reconstruction produced a non-real coefficient")
            ptm[row, column] = float(value.real)

    e00 = 0.5 * (e_paulis[0] + e_paulis[3])
    e11 = 0.5 * (e_paulis[0] - e_paulis[3])
    e01 = 0.5 * (e_paulis[1] + 1.0j * e_paulis[2])
    e10 = 0.5 * (e_paulis[1] - 1.0j * e_paulis[2])
    choi = np.block([[e00, e01], [e10, e11]])
    choi = 0.5 * (choi + choi.conj().T)
    min_choi = float(np.min(np.linalg.eigvalsh(choi)))

    survival_effect = sum(ptm[0, index] * PAULIS[index] for index in range(4))
    survival_effect = 0.5 * (survival_effect + survival_effect.conj().T)
    tni_eigenvalues = np.linalg.eigvalsh(survival_effect)

    traces = np.array([np.trace(matrices[label]).real for label in STATE_LABELS])
    hermiticity = max(
        float(np.linalg.norm(matrices[label] - matrices[label].conj().T, ord="fro"))
        for label in STATE_LABELS
    )
    min_output_eigenvalue = min(
        float(np.min(np.linalg.eigvalsh(matrices[label]))) for label in STATE_LABELS
    )
    pauli_block = ptm[1:, 1:]
    off_diagonal = pauli_block - np.diag(np.diag(pauli_block))
    coherent = 0.5 * (pauli_block - pauli_block.T)
    return SubchannelTomography(
        ptm=ptm,
        choi=choi,
        pair_sum_linearity_residual=pair_residual,
        minimum_choi_eigenvalue=min_choi,
        tni_effect_eigenvalues=(float(tni_eigenvalues[0]), float(tni_eigenvalues[1])),
        maximum_output_hermiticity_error=hermiticity,
        minimum_output_eigenvalue=min_output_eigenvalue,
        minimum_survival=float(np.min(traces)),
        maximum_survival=float(np.max(traces)),
        mean_leakage=float(1.0 - np.mean(traces)),
        survival_spread=float(np.max(traces) - np.min(traces)),
        off_diagonal_pauli_norm=float(np.linalg.norm(off_diagonal, ord="fro")),
        coherent_rotation_norm=float(np.linalg.norm(coherent, ord="fro")),
        nonunital_code_flow_norm=float(np.linalg.norm(ptm[1:, 0])),
        state_dependent_survival_norm=float(np.linalg.norm(ptm[0, 1:])),
    )


def finite_horizon_pauli_lifetime(
    cycles: ArrayLike,
    time_us: ArrayLike,
    signal: ArrayLike,
) -> dict[str, Any]:
    """Return raw finite-horizon Pauli-signal lifetime diagnostics without fitting.

    The signed area is intentionally not clipped or forced through an exponential.
    An interpolated e-fold crossing is reported only when the raw signal crosses
    ``exp(-1)`` of its initial value.
    """

    cycle_grid = np.asarray(cycles, dtype=np.float64)
    times = np.asarray(time_us, dtype=np.float64)
    values = np.asarray(signal, dtype=np.float64)
    if cycle_grid.ndim != 1 or times.shape != cycle_grid.shape or values.shape != cycle_grid.shape:
        raise ValueError("cycles, time_us and signal must be aligned one-dimensional arrays")
    if cycle_grid.size < 3 or not np.all(np.isfinite(cycle_grid)):
        raise ValueError("at least three finite cycle points are required")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(values)):
        raise ValueError("time and signal must be finite")
    if cycle_grid[0] != 0.0 or times[0] != 0.0:
        raise ValueError("cycle and time grids must start at zero")
    if np.any(np.diff(cycle_grid) <= 0.0) or np.any(np.diff(times) <= 0.0):
        raise ValueError("cycle and time grids must increase strictly")
    if abs(values[0]) <= 1.0e-12:
        raise ValueError("initial Pauli signal must be nonzero")
    normalized = values / values[0]
    area_cycles = float(np.trapezoid(normalized, cycle_grid))
    area_us = float(np.trapezoid(normalized, times))
    threshold = exp(-1.0)
    crossing_cycle: float | None = None
    crossing_us: float | None = None
    for index in range(1, normalized.size):
        if normalized[index] <= threshold < normalized[index - 1]:
            denominator = normalized[index - 1] - normalized[index]
            fraction = 0.0 if denominator <= 0.0 else (
                normalized[index - 1] - threshold
            ) / denominator
            crossing_cycle = float(cycle_grid[index - 1] + fraction * np.diff(cycle_grid)[index - 1])
            crossing_us = float(times[index - 1] + fraction * np.diff(times)[index - 1])
            break
    positive_steps = np.diff(normalized)
    return {
        "definition": "finite-horizon integral of the raw code-weighted Pauli contrast; no exponential fit or postselection",
        "truncated_signed_area_cycles": area_cycles,
        "truncated_signed_area_us": area_us,
        "e_fold_crossing_cycles": crossing_cycle,
        "e_fold_crossing_us": crossing_us,
        "e_fold_status": "observed" if crossing_cycle is not None else "right_censored",
        "horizon_cycles": float(cycle_grid[-1]),
        "horizon_us": float(times[-1]),
        "final_normalized_signal": float(normalized[-1]),
        "minimum_normalized_signal": float(np.min(normalized)),
        "maximum_normalized_signal": float(np.max(normalized)),
        "revival_step_count": int(np.count_nonzero(positive_steps > 1.0e-10)),
        "negative_point_count": int(np.count_nonzero(normalized < 0.0)),
    }


@dataclass(frozen=True)
class FockLogicalChannelConfig:
    mode: ChannelMode
    full_cycles: int = 30
    cutoff: int = 12
    projector_delta: float = 0.34
    cavity_lifetime_us: float = 245.0
    ancilla_t1_us: float = 50.0
    ancilla_t2_us: float = 60.0
    cycle_duration_us: float = MEASUREMENT_TIMING.full_cycle_duration_ns / 1000.0
    device: Literal["cpu", "cuda"] = "cpu"
    real_dtype: Literal["float32", "float64"] = "float64"
    scope: str = MODEL_SCOPE

    def __post_init__(self) -> None:
        if self.mode not in {"qec_on", "qec_off"}:
            raise ValueError("mode must be qec_on or qec_off")
        for name in ("full_cycles", "cutoff"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        if not 4 <= self.cutoff <= 48:
            raise ValueError("cutoff must lie in [4,48]")
        if self.full_cycles > 10_000:
            raise ValueError("full_cycles exceeds the explicit safety guard")
        for name in (
            "projector_delta",
            "cavity_lifetime_us",
            "ancilla_t1_us",
            "ancilla_t2_us",
            "cycle_duration_us",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        expected_duration = MEASUREMENT_TIMING.full_cycle_duration_ns / 1000.0
        if abs(self.cycle_duration_us - expected_duration) > 1.0e-12:
            raise ValueError("matched T5.3.1 lanes require the registered 10 us cycle")
        if self.ancilla_t2_us > 2.0 * self.ancilla_t1_us + 1.0e-12:
            raise ValueError("ancilla_t2_us must not exceed 2*T1")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if self.real_dtype != "float64":
            raise ValueError("formal logical tomography requires float64")
        if self.scope != MODEL_SCOPE:
            raise ValueError("model scope must remain fail closed")


@dataclass
class FockLogicalChannelResult:
    config: FockLogicalChannelConfig
    cycles: NDArray[np.int64]
    time_us: NDArray[np.float64]
    projected_outputs: NDArray[np.complex128]
    survival: NDArray[np.float64]
    leakage: NDArray[np.float64]
    conditional_bloch: NDArray[np.float64]
    tomography: tuple[SubchannelTomography, ...]
    pauli_lifetimes: Mapping[str, Mapping[str, Any]]
    event_accounting: Mapping[str, float | int]
    maximum_physical_trace_error: float
    maximum_physical_hermiticity_error: float
    minimum_physical_eigenvalue: float

    @property
    def ptm(self) -> NDArray[np.float64]:
        return np.stack([point.ptm for point in self.tomography], axis=0)

    def to_dict(self, *, include_projected_outputs: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "config": asdict(self.config),
            "state_labels": list(STATE_LABELS),
            "cycles": self.cycles.tolist(),
            "time_us": self.time_us.tolist(),
            "survival": self.survival.tolist(),
            "leakage": self.leakage.tolist(),
            "conditional_bloch_xyz": self.conditional_bloch.tolist(),
            "tomography": [point.to_dict() for point in self.tomography],
            "pauli_lifetimes": {key: dict(value) for key, value in self.pauli_lifetimes.items()},
            "event_accounting": dict(self.event_accounting),
            "maximum_physical_trace_error": self.maximum_physical_trace_error,
            "maximum_physical_hermiticity_error": self.maximum_physical_hermiticity_error,
            "minimum_physical_eigenvalue": self.minimum_physical_eigenvalue,
        }
        if include_projected_outputs:
            payload["projected_output_real"] = self.projected_outputs.real.tolist()
            payload["projected_output_imag"] = self.projected_outputs.imag.tolist()
        return payload


class FockLogicalChannelSimulator:
    """Propagate all six code states through one matched repeated channel."""

    def __init__(self, config: FockLogicalChannelConfig) -> None:
        th = _require_torch()
        if not isinstance(config, FockLogicalChannelConfig):
            raise TypeError("config must be a FockLogicalChannelConfig")
        if config.device == "cuda" and not th.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.config = config
        base = DifferentiableSBSConfig(
            cutoff=config.cutoff,
            full_cycles=1,
            batch_size=len(STATE_LABELS),
            projector_delta=config.projector_delta,
            cavity_lifetime_us=config.cavity_lifetime_us,
            ancilla_t1_us=config.ancilla_t1_us,
            ancilla_t2_us=config.ancilla_t2_us,
            device=config.device,
            real_dtype=config.real_dtype,
        )
        self.engine = DifferentiableSBSTrajectorySimulator(base)
        self.controls = self.engine.bounded_physical_controls(None)[:, 0, :]
        self.reset_window_kraus = self.engine._joint_idle_kraus(
            MEASUREMENT_TIMING.measurement_and_or_reset_ns / 1000.0
        )
        self.cycle_idle_kraus = self.engine._joint_idle_kraus(config.cycle_duration_us)

    def _initial_joint_density(self) -> Any:
        th = _require_torch()
        logical = th.tensor(
            np.stack([logical_eigenstate_density(label) for label in STATE_LABELS]),
            dtype=self.engine.complex_dtype,
            device=self.engine.device,
        )
        isometry = self.engine.logical_isometry
        cavity = isometry.unsqueeze(0) @ logical @ isometry.mH.unsqueeze(0)
        joint = th.einsum("bij,kl->bikjl", cavity, self.engine.g_projector).reshape(
            len(STATE_LABELS), self.engine.joint_dimension, self.engine.joint_dimension
        )
        return self.engine._stabilize_density(joint)

    def _trace_and_reset(self, state: Any) -> Any:
        th = _require_torch()
        cavity = self.engine._reduce_cavity(state)
        reset = th.einsum("bij,kl->bikjl", cavity, self.engine.g_projector).reshape(
            len(STATE_LABELS), self.engine.joint_dimension, self.engine.joint_dimension
        )
        return self.engine._stabilize_density(reset)

    def _projected_outputs(self, state: Any) -> NDArray[np.complex128]:
        cavity = self.engine._reduce_cavity(state)
        isometry = self.engine.logical_isometry
        projected = isometry.mH.unsqueeze(0) @ cavity @ isometry.unsqueeze(0)
        return projected.detach().cpu().numpy().astype(np.complex128)

    def _active_cycle(self, state: Any) -> Any:
        state = self.engine._apply_idle(state, "entering_cycle")
        for layer in range(1, 5):
            state = self.engine._layer(state, self.controls, layer)
        for operators in self.reset_window_kraus:
            state = self.engine._apply_kraus(state, operators)
        state = self.engine._stabilize_density(state)
        state = self._trace_and_reset(state)
        state = self.engine._virtual_rotation(state, self.controls[:, 14])
        state = self.engine._apply_idle(state, "virtual_rotation_and_idle")

        state = self.engine._apply_idle(state, "entering_cycle")
        for layer in range(1, 5):
            state = self.engine._layer(state, self.controls, layer)
        for operators in self.reset_window_kraus:
            state = self.engine._apply_kraus(state, operators)
        state = self.engine._stabilize_density(state)
        state = self._trace_and_reset(state)
        state = self.engine._virtual_rotation(state, self.controls[:, 14])
        state = self.engine._apply_idle(state, "virtual_rotation_and_idle")
        return state

    def _idle_cycle(self, state: Any) -> Any:
        for operators in self.cycle_idle_kraus:
            state = self.engine._apply_kraus(state, operators)
        return self.engine._stabilize_density(state)

    def run(self) -> FockLogicalChannelResult:
        th = _require_torch()
        state = self._initial_joint_density()
        outputs = [self._projected_outputs(state)]
        maximum_trace = 0.0
        maximum_hermiticity = 0.0
        minimum_eigenvalue = 1.0
        step = self._active_cycle if self.config.mode == "qec_on" else self._idle_cycle
        with th.no_grad():
            for _ in range(self.config.full_cycles):
                state = step(state)
                trace_error, hermiticity_error, eigenvalue = self.engine._diagnostics(state)
                maximum_trace = max(maximum_trace, trace_error)
                maximum_hermiticity = max(maximum_hermiticity, hermiticity_error)
                minimum_eigenvalue = min(minimum_eigenvalue, eigenvalue)
                outputs.append(self._projected_outputs(state))
        projected = np.stack(outputs, axis=0)
        survival = np.trace(projected, axis1=-2, axis2=-1).real
        leakage = 1.0 - survival
        conditional_bloch = np.empty((projected.shape[0], len(STATE_LABELS), 3))
        for cycle_index in range(projected.shape[0]):
            for state_index in range(len(STATE_LABELS)):
                weight = float(survival[cycle_index, state_index])
                if weight <= 1.0e-14:
                    conditional_bloch[cycle_index, state_index] = np.nan
                else:
                    conditional_bloch[cycle_index, state_index] = [
                        float(np.trace(pauli @ projected[cycle_index, state_index]).real / weight)
                        for pauli in PAULIS[1:]
                    ]
        tomography = tuple(
            reconstruct_code_subchannel(
                {label: projected[index, state_index] for state_index, label in enumerate(STATE_LABELS)}
            )
            for index in range(projected.shape[0])
        )
        cycles = np.arange(self.config.full_cycles + 1, dtype=np.int64)
        time_us = cycles.astype(np.float64) * self.config.cycle_duration_us
        ptm = np.stack([point.ptm for point in tomography])
        lifetimes = {
            axis: finite_horizon_pauli_lifetime(cycles, time_us, ptm[:, index, index])
            for index, axis in enumerate(AXIS_LABELS, start=1)
        }
        active = self.config.mode == "qec_on"
        cycle_count = self.config.full_cycles
        event_accounting: dict[str, float | int] = {
            "full_cycles": cycle_count,
            "total_physical_time_us": cycle_count * self.config.cycle_duration_us,
            "measurement_events": 2 * cycle_count if active else 0,
            "reset_events": 2 * cycle_count if active else 0,
            "active_gate_applications": 18 * cycle_count if active else 0,
            "outcome_dependent_parameter_updates": 0,
            "postselected_trajectories": 0,
            "discarded_trajectories": 0,
            "target_hardware_measured": 0,
        }
        return FockLogicalChannelResult(
            config=self.config,
            cycles=cycles,
            time_us=time_us,
            projected_outputs=projected,
            survival=survival,
            leakage=leakage,
            conditional_bloch=conditional_bloch,
            tomography=tomography,
            pauli_lifetimes=lifetimes,
            event_accounting=event_accounting,
            maximum_physical_trace_error=maximum_trace,
            maximum_physical_hermiticity_error=maximum_hermiticity,
            minimum_physical_eigenvalue=minimum_eigenvalue,
        )


__all__ = [
    "AXIS_LABELS",
    "AXIS_STATE_PAIRS",
    "FockLogicalChannelConfig",
    "FockLogicalChannelResult",
    "FockLogicalChannelSimulator",
    "MODEL_SCOPE",
    "PAULI_LABELS",
    "STATE_LABELS",
    "SubchannelTomography",
    "finite_horizon_pauli_lifetime",
    "logical_eigenstate_density",
    "reconstruct_code_subchannel",
]
