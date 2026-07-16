"""PyTorch differentiable short-horizon SBS trajectory simulator.

This module implements the joint cavity--two-level-ancilla model used by the
Feedback-GRAPE feasibility lane.  It deliberately does *not* wrap the fixed
analytic Kraus pair from :mod:`physics.fock_sbs_cycle`: all fifteen controls of
one measurement half-cycle enter explicit differentiable ``R/ECD/D/VR`` gates,
and the g/e trajectory probability is accumulated from the projective
measurement probabilities.

The fixed timing profile is Table S1 of Puviani *et al.* (5 us per half-cycle,
10 us per full X/Z cycle).  It is a literature simulation profile, not the
4.924 us Sivak Table S3 constituent timeline and not a target-board timing
claim.  Gates are instantaneous and each listed interval is represented by an
analytic CPTP idle channel.  The model omits pulse Hamiltonians, a multilevel
transmon, leakage, SPAM and device calibration; those omissions are surfaced in
the result scope and validation artifact.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from math import comb, exp, isfinite, pi, sqrt
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from .fock_density_model import FiniteCutoffFockModel
from .sbs_error_space import SBS_PROTOCOL_ID

try:  # The repository's minimal recovery interpreter intentionally lacks torch.
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised by the minimal env.
    torch = None  # type: ignore[assignment]


TorchDeviceName = Literal["cpu", "cuda"]
OutcomeName = Literal["g", "e"]

DIFFERENTIABLE_SBS_SCOPE = (
    "finite-cutoff joint cavity-two-level-ancilla differentiable trajectory model; "
    "instantaneous paper-defined gates plus analytic idle CPTP channels; no pulse "
    "Hamiltonian, transmon leakage, SPAM, device calibration, or hardware timing claim"
)
POROTTI_S1_PROFILE_ID = "PUVIANI-2024-TABLE-S1-FEEDBACK-GRAPE"
POROTTI_S1_SOURCE = (
    "relative_papers/Non-Markovian_feedback_for_optimized_quantum_error_correction/"
    "Non-Markovian_feedback_for_optimized_quantum_error_correction.md:451-459"
)
PARAMETER_NAMES = (
    "layer1_phi",
    "layer1_theta",
    "layer1_beta_real",
    "layer1_beta_imag",
    "layer2_phi",
    "layer2_theta",
    "layer2_beta_real",
    "layer2_beta_imag",
    "layer3_phi",
    "layer3_theta",
    "layer3_beta_real",
    "layer3_beta_imag",
    "layer4_phi",
    "layer4_theta",
    "virtual_rotation",
)


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError(
            "T2.3.4 requires PyTorch; use the local DLEnv/QuantumEnv interpreter "
            "or install torch in the selected environment"
        )
    return torch


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _finite_positive(value: float, name: str) -> float:
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


@dataclass(frozen=True)
class TrajectoryTimingPhase:
    phase_id: str
    duration_ns: int
    gate_before_idle: str


@dataclass(frozen=True)
class DifferentiableSBSTimingProfile:
    profile_id: str = POROTTI_S1_PROFILE_ID
    phases: tuple[TrajectoryTimingPhase, ...] = (
        TrajectoryTimingPhase("entering_cycle", 100, "none"),
        TrajectoryTimingPhase("layer_1", 500, "R1_then_ECD1"),
        TrajectoryTimingPhase("layer_2", 700, "R2_then_ECD2"),
        TrajectoryTimingPhase("layer_3", 300, "R3_then_ECD3"),
        TrajectoryTimingPhase("layer_4", 100, "R4_then_fixed_D_alpha"),
        TrajectoryTimingPhase("measurement_and_reset", 2300, "measure_g_or_e_then_reset"),
        TrajectoryTimingPhase("virtual_rotation_and_idle", 1000, "VR"),
    )
    source: str = POROTTI_S1_SOURCE
    evidence_scope: str = "literature_simulation_timing_not_target_hardware_measurement"

    def __post_init__(self) -> None:
        if self.profile_id != POROTTI_S1_PROFILE_ID:
            raise ValueError("T2.3.4 freezes the Puviani Table S1 timing profile")
        if len(self.phases) != 7 or len({item.phase_id for item in self.phases}) != 7:
            raise ValueError("timing profile must contain the seven unique Table S1 phases")
        if any(item.duration_ns <= 0 for item in self.phases):
            raise ValueError("all timing phases must have positive duration")
        if self.half_cycle_duration_ns != 5000:
            raise ValueError("Puviani Table S1 half-cycle must sum to 5000 ns")
        if self.evidence_scope != "literature_simulation_timing_not_target_hardware_measurement":
            raise ValueError("timing evidence scope must remain fail closed")

    @property
    def half_cycle_duration_ns(self) -> int:
        return sum(item.duration_ns for item in self.phases)

    @property
    def full_cycle_duration_ns(self) -> int:
        return 2 * self.half_cycle_duration_ns


@dataclass(frozen=True)
class DifferentiableSBSConfig:
    cutoff: int = 12
    full_cycles: int = 1
    batch_size: int = 2
    projector_delta: float = 0.34
    grid_points: int = 4097
    fixed_layer4_displacement: float = sqrt(pi / 2.0)
    cavity_lifetime_us: float = 245.0
    ancilla_t1_us: float = 50.0
    ancilla_t2_us: float = 60.0
    device: TorchDeviceName = "cpu"
    real_dtype: str = "float64"
    probability_floor: float = 1.0e-14
    timing: DifferentiableSBSTimingProfile = DifferentiableSBSTimingProfile()
    protocol_id: str = SBS_PROTOCOL_ID
    scope: str = DIFFERENTIABLE_SBS_SCOPE

    def __post_init__(self) -> None:
        cutoff = _positive_int(self.cutoff, "cutoff")
        if not 4 <= cutoff <= 48:
            raise ValueError("cutoff must lie in [4, 48]")
        object.__setattr__(self, "cutoff", cutoff)
        cycles = _positive_int(self.full_cycles, "full_cycles")
        if cycles > 10:
            raise ValueError("T2.3.4 short-horizon full_cycles must lie in [1, 10]")
        object.__setattr__(self, "full_cycles", cycles)
        batch = _positive_int(self.batch_size, "batch_size")
        if batch > 4096:
            raise ValueError("batch_size exceeds the explicit feasibility guard")
        object.__setattr__(self, "batch_size", batch)
        delta = _finite_positive(self.projector_delta, "projector_delta")
        object.__setattr__(self, "projector_delta", delta)
        grid = _positive_int(self.grid_points, "grid_points")
        if grid < 1025 or grid % 2 == 0:
            raise ValueError("grid_points must be an odd integer >= 1025")
        object.__setattr__(self, "grid_points", grid)
        object.__setattr__(
            self,
            "fixed_layer4_displacement",
            _finite_positive(self.fixed_layer4_displacement, "fixed_layer4_displacement"),
        )
        for name in ("cavity_lifetime_us", "ancilla_t1_us", "ancilla_t2_us"):
            object.__setattr__(self, name, _finite_positive(getattr(self, name), name))
        if self.ancilla_t2_us > 2.0 * self.ancilla_t1_us + 1.0e-12:
            raise ValueError("ancilla_t2_us must not exceed 2*T1 for a nonnegative pure-dephasing rate")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")
        if self.real_dtype not in {"float32", "float64"}:
            raise ValueError("real_dtype must be float32 or float64")
        floor = float(self.probability_floor)
        if not isfinite(floor) or not 0.0 < floor < 1.0e-4:
            raise ValueError("probability_floor must lie in (0, 1e-4)")
        object.__setattr__(self, "probability_floor", floor)
        if not isinstance(self.timing, DifferentiableSBSTimingProfile):
            raise TypeError("timing must be a DifferentiableSBSTimingProfile")
        if self.protocol_id != SBS_PROTOCOL_ID:
            raise ValueError("T2.3.4 implements only the frozen SBS main protocol")
        if self.scope != DIFFERENTIABLE_SBS_SCOPE:
            raise ValueError("scope must preserve the fail-closed model boundary")

    @property
    def half_cycles(self) -> int:
        return 2 * self.full_cycles


@dataclass(frozen=True)
class TrajectoryResourceProfile:
    device: str
    real_dtype: str
    complex_dtype: str
    cutoff: int
    joint_dimension: int
    batch_size: int
    full_cycles: int
    half_cycles: int
    control_source: str
    trainable_controls: int
    matrix_exponentials: int
    unitary_applications: int
    idle_windows: int
    cptp_channel_applications: int
    state_tensor_bytes: int
    autograd_state_lower_bound_bytes: int
    wall_time_seconds: float
    cuda_peak_allocated_bytes: int | None
    timing_profile_id: str
    simulated_physical_time_ns: int
    target_hardware_measured: bool = False


@dataclass
class DifferentiableTrajectoryResult:
    final_joint_density: Any
    final_cavity_density: Any
    outcomes: Any
    conditional_probabilities: Any
    log_probability: Any
    trajectory_probability: Any
    reward: Any
    physical_controls: Any
    cycle_fidelities: Any | None
    cycle_code_survival: Any | None
    cycle_logical_z_signal: Any | None
    cycle_conditional_logical_z: Any | None
    resource_profile: TrajectoryResourceProfile
    maximum_trace_error: float
    maximum_hermiticity_error: float
    minimum_final_eigenvalue: float
    protocol_id: str = SBS_PROTOCOL_ID
    scope: str = DIFFERENTIABLE_SBS_SCOPE

    def detached_summary(self) -> dict[str, Any]:
        return {
            "outcomes": self.outcomes.detach().cpu().tolist(),
            "conditional_probabilities": (
                self.conditional_probabilities.detach().cpu().tolist()
            ),
            "log_probability": self.log_probability.detach().cpu().tolist(),
            "trajectory_probability": self.trajectory_probability.detach().cpu().tolist(),
            "reward": self.reward.detach().cpu().tolist(),
            "cycle_fidelities": (
                None
                if self.cycle_fidelities is None
                else self.cycle_fidelities.detach().cpu().tolist()
            ),
            "cycle_code_survival": (
                None
                if self.cycle_code_survival is None
                else self.cycle_code_survival.detach().cpu().tolist()
            ),
            "cycle_logical_z_signal": (
                None
                if self.cycle_logical_z_signal is None
                else self.cycle_logical_z_signal.detach().cpu().tolist()
            ),
            "cycle_conditional_logical_z": (
                None
                if self.cycle_conditional_logical_z is None
                else self.cycle_conditional_logical_z.detach().cpu().tolist()
            ),
            "resource_profile": asdict(self.resource_profile),
            "maximum_trace_error": self.maximum_trace_error,
            "maximum_hermiticity_error": self.maximum_hermiticity_error,
            "minimum_final_eigenvalue": self.minimum_final_eigenvalue,
            "protocol_id": self.protocol_id,
            "scope": self.scope,
        }


def nominal_sbs_parameters(*, device: str = "cpu", dtype: Any | None = None) -> Any:
    """Return the paper Table S4 nominal 15-vector for one half-cycle."""

    th = _require_torch()
    actual_dtype = th.float64 if dtype is None else dtype
    values = (
        pi / 2.0,
        pi / 2.0,
        0.0,
        0.2,
        0.0,
        -pi / 2.0,
        sqrt(2.0 * pi),
        0.0,
        0.0,
        pi / 2.0,
        0.0,
        0.2,
        pi / 2.0,
        -pi / 2.0,
        pi / 2.0,
    )
    return th.tensor(values, dtype=actual_dtype, device=device)


class DifferentiableSBSTrajectorySimulator:
    """Explicit batched joint-state simulator with stochastic g/e trajectories."""

    def __init__(self, config: DifferentiableSBSConfig) -> None:
        th = _require_torch()
        if not isinstance(config, DifferentiableSBSConfig):
            raise TypeError("config must be a DifferentiableSBSConfig")
        if config.device == "cuda" and not th.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        self.config = config
        self.device = th.device(config.device)
        self.real_dtype = th.float64 if config.real_dtype == "float64" else th.float32
        self.complex_dtype = th.complex128 if self.real_dtype == th.float64 else th.complex64
        self.cutoff = config.cutoff
        self.joint_dimension = 2 * config.cutoff
        self.identity_cavity = th.eye(
            config.cutoff, dtype=self.complex_dtype, device=self.device
        )
        self.identity_ancilla = th.eye(2, dtype=self.complex_dtype, device=self.device)
        self.identity_joint = th.eye(
            self.joint_dimension, dtype=self.complex_dtype, device=self.device
        )
        self.a = self._annihilation()
        self.adag = self.a.mH
        self.number = self.adag @ self.a
        self.g_projector = th.tensor(
            [[1.0, 0.0], [0.0, 0.0]], dtype=self.complex_dtype, device=self.device
        )
        self.e_projector = th.tensor(
            [[0.0, 0.0], [0.0, 1.0]], dtype=self.complex_dtype, device=self.device
        )
        self.sigma_z = th.tensor(
            [[1.0, 0.0], [0.0, -1.0]], dtype=self.complex_dtype, device=self.device
        )
        self._idle_kraus = {
            phase.phase_id: self._joint_idle_kraus(phase.duration_ns / 1000.0)
            for phase in config.timing.phases
        }
        self.initial_state_vector = self._prepare_initial_gkp_vector()
        self.target_cavity_density = (
            self.initial_state_vector[:, None] * self.initial_state_vector.conj()[None, :]
        )
        self.logical_isometry, self.logical_projector, self.logical_z_fock = (
            self._build_logical_code_basis()
        )

    def _annihilation(self) -> Any:
        th = _require_torch()
        matrix = th.zeros(
            (self.cutoff, self.cutoff), dtype=self.complex_dtype, device=self.device
        )
        indices = th.arange(1, self.cutoff, device=self.device)
        matrix[indices - 1, indices] = th.sqrt(indices.to(self.real_dtype)).to(
            self.complex_dtype
        )
        return matrix

    def _prepare_initial_gkp_vector(self) -> Any:
        th = _require_torch()
        model = FiniteCutoffFockModel(self.cutoff)
        preparation = model.prepare_damped_projector_gkp(
            "0",
            self.config.projector_delta,
            grid_points=self.config.grid_points,
            source_coordinate_scale=sqrt(2.0),
        )
        coefficients = np.asarray(preparation.coefficients, dtype=np.complex128)
        coefficients = coefficients / np.linalg.norm(coefficients)
        return th.tensor(coefficients, dtype=self.complex_dtype, device=self.device)

    def _build_logical_code_basis(self) -> tuple[Any, Any, Any]:
        """Build an orthonormal finite-cutoff ``|0_L>, |1_L>`` projection.

        The projected logical-Z signal is an evaluation diagnostic, not a claim
        that finite-cutoff code-space projection equals an experimental logical
        Pauli measurement.  Code leakage remains visible because the signal is
        not divided by the code-survival probability.
        """

        th = _require_torch()
        model = FiniteCutoffFockModel(self.cutoff)
        one = model.prepare_damped_projector_gkp(
            "1",
            self.config.projector_delta,
            grid_points=self.config.grid_points,
            source_coordinate_scale=sqrt(2.0),
        )
        raw = np.column_stack(
            (
                self.initial_state_vector.detach().cpu().numpy(),
                np.asarray(one.coefficients, dtype=np.complex128)
                / np.linalg.norm(one.coefficients),
            )
        )
        gram = raw.conj().T @ raw
        values, vectors = np.linalg.eigh(gram)
        if float(np.min(values)) <= 1.0e-10:
            raise RuntimeError("finite-cutoff logical codewords are linearly dependent")
        inverse_sqrt = (vectors * (1.0 / np.sqrt(values))) @ vectors.conj().T
        isometry_np = raw @ inverse_sqrt
        projector_np = isometry_np @ isometry_np.conj().T
        logical_z_np = isometry_np @ np.diag([1.0, -1.0]) @ isometry_np.conj().T
        return (
            th.tensor(isometry_np, dtype=self.complex_dtype, device=self.device),
            th.tensor(projector_np, dtype=self.complex_dtype, device=self.device),
            th.tensor(logical_z_np, dtype=self.complex_dtype, device=self.device),
        )

    def _cavity_evaluation_metrics(self, cavity: Any) -> tuple[Any, Any, Any, Any]:
        """Return physical-state fidelity and code-aware logical diagnostics."""

        th = _require_torch()
        fidelity = th.einsum("ij,bji->b", self.target_cavity_density, cavity).real
        fidelity = th.clamp(fidelity, min=0.0, max=1.0)
        survival = th.einsum("ij,bji->b", self.logical_projector, cavity).real
        survival = th.clamp(survival, min=0.0, max=1.0)
        logical_z_signal = th.einsum("ij,bji->b", self.logical_z_fock, cavity).real
        conditional = logical_z_signal / th.clamp(
            survival, min=self.config.probability_floor
        )
        return fidelity, survival, logical_z_signal, conditional

    def _cavity_loss_kraus(self, duration_us: float) -> Any:
        th = _require_torch()
        eta = exp(-duration_us / self.config.cavity_lifetime_us)
        operators = []
        for lost in range(self.cutoff):
            matrix = np.zeros((self.cutoff, self.cutoff), dtype=np.complex128)
            for incoming in range(lost, self.cutoff):
                outgoing = incoming - lost
                matrix[outgoing, incoming] = (
                    sqrt(comb(incoming, lost))
                    * ((1.0 - eta) ** (lost / 2.0))
                    * (eta ** (outgoing / 2.0))
                )
            operators.append(matrix)
        return th.tensor(
            np.stack(operators), dtype=self.complex_dtype, device=self.device
        )

    def _ancilla_kraus(self, duration_us: float) -> Any:
        th = _require_torch()
        damping_probability = 1.0 - exp(-duration_us / self.config.ancilla_t1_us)
        amp0 = th.tensor(
            [[1.0, 0.0], [0.0, sqrt(1.0 - damping_probability)]],
            dtype=self.complex_dtype,
            device=self.device,
        )
        amp1 = th.tensor(
            [[0.0, sqrt(damping_probability)], [0.0, 0.0]],
            dtype=self.complex_dtype,
            device=self.device,
        )
        inverse_tphi = 1.0 / self.config.ancilla_t2_us - 0.5 / self.config.ancilla_t1_us
        if inverse_tphi <= 1.0e-15:
            dephasing = self.identity_ancilla[None, :, :]
        else:
            tphi = 1.0 / inverse_tphi
            coherence = exp(-duration_us / tphi)
            dephasing = th.stack(
                (
                    sqrt((1.0 + coherence) / 2.0) * self.identity_ancilla,
                    sqrt((1.0 - coherence) / 2.0) * self.sigma_z,
                )
            )
        amplitude = th.stack((amp0, amp1))
        return amplitude, dephasing

    def _joint_idle_kraus(self, duration_us: float) -> tuple[Any, Any, Any]:
        th = _require_torch()
        cavity = self._cavity_loss_kraus(duration_us)
        cavity_joint = th.stack(
            tuple(th.kron(item, self.identity_ancilla) for item in cavity)
        )
        amplitude, dephasing = self._ancilla_kraus(duration_us)
        amplitude_joint = th.stack(
            tuple(th.kron(self.identity_cavity, item) for item in amplitude)
        )
        dephasing_joint = th.stack(
            tuple(th.kron(self.identity_cavity, item) for item in dephasing)
        )
        return cavity_joint, amplitude_joint, dephasing_joint

    @staticmethod
    def _apply_kraus(state: Any, operators: Any) -> Any:
        th = _require_torch()
        return th.einsum("kij,bjl,knl->bin", operators, state, operators.conj())

    def _apply_idle(self, state: Any, phase_id: str) -> Any:
        for operators in self._idle_kraus[phase_id]:
            state = self._apply_kraus(state, operators)
        return self._stabilize_density(state)

    def _stabilize_density(self, state: Any) -> Any:
        th = _require_torch()
        hermitian = 0.5 * (state + state.mH)
        traces = th.diagonal(hermitian, dim1=-2, dim2=-1).sum(-1).real
        if not bool(th.all(th.isfinite(traces)).detach().cpu()):
            raise RuntimeError("density trace became non-finite")
        if bool(th.any(traces <= 0.0).detach().cpu()):
            raise RuntimeError("density trace became non-positive")
        return hermitian / traces[:, None, None].to(self.complex_dtype)

    def _batch_displacement(self, amplitude: Any) -> Any:
        th = _require_torch()
        if amplitude.ndim != 1 or amplitude.shape[0] != self.config.batch_size:
            raise ValueError("amplitude must have shape (batch_size,)")
        generator = (
            amplitude[:, None, None].to(self.complex_dtype) * self.adag[None, :, :]
            - amplitude.conj()[:, None, None].to(self.complex_dtype) * self.a[None, :, :]
        )
        return th.matrix_exp(generator)

    def _batch_qubit_rotation(self, phi: Any, theta: Any) -> Any:
        th = _require_torch()
        cosine = th.cos(theta / 2.0).to(self.complex_dtype)
        sine = th.sin(theta / 2.0).to(self.complex_dtype)
        phase = th.exp(1.0j * phi.to(self.complex_dtype))
        result = th.zeros(
            (self.config.batch_size, 2, 2),
            dtype=self.complex_dtype,
            device=self.device,
        )
        result[:, 0, 0] = cosine
        result[:, 1, 1] = cosine
        result[:, 0, 1] = -1.0j * sine * phase.conj()
        result[:, 1, 0] = -1.0j * sine * phase
        return result

    def _batch_joint_qubit_rotation(self, phi: Any, theta: Any) -> Any:
        th = _require_torch()
        rotation = self._batch_qubit_rotation(phi, theta)
        return th.einsum("ij,bkl->bikjl", self.identity_cavity, rotation).reshape(
            self.config.batch_size, self.joint_dimension, self.joint_dimension
        )

    def _batch_ecd(self, beta: Any) -> Any:
        th = _require_torch()
        plus = self._batch_displacement(beta / 2.0)
        minus = plus.mH
        blocks = th.zeros(
            (
                self.config.batch_size,
                self.cutoff,
                2,
                self.cutoff,
                2,
            ),
            dtype=self.complex_dtype,
            device=self.device,
        )
        blocks[:, :, 0, :, 1] = plus
        blocks[:, :, 1, :, 0] = minus
        return blocks.reshape(
            self.config.batch_size, self.joint_dimension, self.joint_dimension
        )

    def _batch_cavity_unitary(self, cavity_unitary: Any) -> Any:
        th = _require_torch()
        return th.einsum(
            "bij,kl->bikjl", cavity_unitary, self.identity_ancilla
        ).reshape(self.config.batch_size, self.joint_dimension, self.joint_dimension)

    @staticmethod
    def _apply_unitary(state: Any, unitary: Any) -> Any:
        return unitary @ state @ unitary.mH

    def bounded_physical_controls(self, raw_corrections: Any | None = None) -> Any:
        """Map unconstrained corrections to the paper's bounded 15 controls."""

        th = _require_torch()
        expected = (self.config.batch_size, self.config.half_cycles, len(PARAMETER_NAMES))
        if raw_corrections is None:
            raw = th.zeros(expected, dtype=self.real_dtype, device=self.device)
        else:
            if not isinstance(raw_corrections, th.Tensor):
                raise TypeError("raw_corrections must be a torch.Tensor")
            raw = raw_corrections.to(device=self.device, dtype=self.real_dtype)
            if raw.ndim == 2 and tuple(raw.shape) == expected[1:]:
                raw = raw.unsqueeze(0).expand(expected)
            if tuple(raw.shape) != expected:
                raise ValueError(f"raw_corrections must have shape {expected} or {expected[1:]}")
            if not bool(th.all(th.isfinite(raw)).detach().cpu()):
                raise ValueError("raw_corrections must be finite")
        return self._map_bounded_corrections(raw)

    def _map_bounded_corrections(self, raw: Any) -> Any:
        """Apply the nominal-plus-bounded-residual map to any trailing 15-vector."""

        th = _require_torch()
        if raw.shape[-1] != len(PARAMETER_NAMES):
            raise ValueError("raw correction tensor must end in the 15 control parameters")
        nominal = nominal_sbs_parameters(device=str(self.device), dtype=self.real_dtype)
        bounds = th.full(
            (len(PARAMETER_NAMES),), 2.0, dtype=self.real_dtype, device=self.device
        )
        bounds[-1] = 1.0
        return nominal + bounds * th.tanh(raw)

    def _policy_controls(self, policy: Any, history: Any, half_index: int) -> Any:
        """Evaluate a causal history-conditioned policy for one half-cycle."""

        th = _require_torch()
        if not callable(policy):
            raise TypeError("control_policy must be callable")
        step_rollout = getattr(policy, "step_rollout", None)
        raw = (
            step_rollout(history, half_index)
            if callable(step_rollout)
            else policy(history, half_index)
        )
        if not isinstance(raw, th.Tensor):
            raise TypeError("control_policy must return a torch.Tensor")
        raw = raw.to(device=self.device, dtype=self.real_dtype)
        if raw.ndim == 1 and tuple(raw.shape) == (len(PARAMETER_NAMES),):
            raw = raw[None, :].expand(self.config.batch_size, -1)
        expected = (self.config.batch_size, len(PARAMETER_NAMES))
        if tuple(raw.shape) != expected:
            raise ValueError(f"control_policy must return shape {expected} or (15,)")
        if not bool(th.all(th.isfinite(raw)).detach().cpu()):
            raise ValueError("control_policy returned non-finite corrections")
        return self._map_bounded_corrections(raw)

    def _initial_joint_density(self) -> Any:
        th = _require_torch()
        cavity = self.target_cavity_density[None, :, :].expand(
            self.config.batch_size, -1, -1
        )
        joint = th.einsum("bij,kl->bikjl", cavity, self.g_projector).reshape(
            self.config.batch_size, self.joint_dimension, self.joint_dimension
        )
        return joint.clone()

    def _layer(self, state: Any, controls: Any, layer: int) -> Any:
        offset = (layer - 1) * 4
        if layer <= 3:
            phi = controls[:, offset]
            theta = controls[:, offset + 1]
            beta = controls[:, offset + 2].to(self.complex_dtype) + 1.0j * controls[
                :, offset + 3
            ].to(self.complex_dtype)
            state = self._apply_unitary(
                state, self._batch_joint_qubit_rotation(phi, theta)
            )
            state = self._apply_unitary(state, self._batch_ecd(beta))
        elif layer == 4:
            phi = controls[:, 12]
            theta = controls[:, 13]
            state = self._apply_unitary(
                state, self._batch_joint_qubit_rotation(phi, theta)
            )
            fixed = torch.full(
                (self.config.batch_size,),
                self.config.fixed_layer4_displacement,
                dtype=self.complex_dtype,
                device=self.device,
            )
            state = self._apply_unitary(
                state, self._batch_cavity_unitary(self._batch_displacement(fixed))
            )
        else:
            raise ValueError("layer must lie in [1, 4]")
        return self._apply_idle(state, f"layer_{layer}")

    def _measurement_probabilities(self, state: Any) -> tuple[Any, Any]:
        th = _require_torch()
        blocks = state.reshape(
            self.config.batch_size, self.cutoff, 2, self.cutoff, 2
        )
        g_state = blocks[:, :, 0, :, 0]
        e_state = blocks[:, :, 1, :, 1]
        p_g = th.diagonal(g_state, dim1=-2, dim2=-1).sum(-1).real
        p_e = th.diagonal(e_state, dim1=-2, dim2=-1).sum(-1).real
        probabilities = th.stack((p_g, p_e), dim=-1)
        probabilities = probabilities / probabilities.sum(-1, keepdim=True)
        return probabilities, th.stack((g_state, e_state), dim=1)

    def _measure_and_reset(
        self,
        state: Any,
        forced_outcome: Any | None,
        generator: Any,
    ) -> tuple[Any, Any, Any]:
        th = _require_torch()
        probabilities, unnormalized_cavity = self._measurement_probabilities(state)
        if forced_outcome is None:
            uniforms = th.rand(
                (self.config.batch_size,), generator=generator, dtype=th.float64
            ).to(self.device)
            outcomes = (uniforms >= probabilities[:, 0].detach()).to(th.int64)
        else:
            outcomes = forced_outcome.to(device=self.device, dtype=th.int64)
            if tuple(outcomes.shape) != (self.config.batch_size,):
                raise ValueError("forced outcome slice must have shape (batch_size,)")
            if not bool(th.all((outcomes == 0) | (outcomes == 1)).detach().cpu()):
                raise ValueError("forced outcomes must encode g=0 or e=1")
        batch_index = th.arange(self.config.batch_size, device=self.device)
        selected_probability = probabilities[batch_index, outcomes]
        if bool(
            th.any(selected_probability.detach() <= self.config.probability_floor).cpu()
        ):
            raise RuntimeError("sampled/forced trajectory contains a numerically impossible branch")
        cavity = unnormalized_cavity[batch_index, outcomes]
        cavity = cavity / selected_probability[:, None, None].to(self.complex_dtype)
        reset = th.einsum("bij,kl->bikjl", cavity, self.g_projector).reshape(
            self.config.batch_size, self.joint_dimension, self.joint_dimension
        )
        return self._stabilize_density(reset), outcomes, selected_probability

    def _virtual_rotation(self, state: Any, angles: Any) -> Any:
        th = _require_torch()
        levels = th.arange(self.cutoff, device=self.device, dtype=self.real_dtype)
        phases = th.exp(1.0j * angles[:, None].to(self.complex_dtype) * levels[None, :])
        cavity = th.diag_embed(phases)
        return self._apply_unitary(state, self._batch_cavity_unitary(cavity))

    def _reduce_cavity(self, joint: Any) -> Any:
        blocks = joint.reshape(
            self.config.batch_size, self.cutoff, 2, self.cutoff, 2
        )
        return blocks[:, :, 0, :, 0] + blocks[:, :, 1, :, 1]

    def _diagnostics(self, state: Any) -> tuple[float, float, float]:
        th = _require_torch()
        detached = state.detach()
        traces = th.diagonal(detached, dim1=-2, dim2=-1).sum(-1)
        trace_error = float(th.max(th.abs(traces - 1.0)).cpu())
        hermiticity = float(th.max(th.linalg.matrix_norm(detached - detached.mH)).cpu())
        eigenvalues = th.linalg.eigvalsh(0.5 * (detached + detached.mH))
        minimum = float(th.min(eigenvalues).cpu())
        return trace_error, hermiticity, minimum

    def gate_unitarity_residuals(self, controls: Any | None = None) -> dict[str, float]:
        """Audit each differentiable gate family without changing a state."""

        th = _require_torch()
        physical = self.bounded_physical_controls(controls)[:, 0, :]
        samples = {
            "R1": self._batch_joint_qubit_rotation(physical[:, 0], physical[:, 1]),
            "ECD1": self._batch_ecd(
                physical[:, 2].to(self.complex_dtype)
                + 1.0j * physical[:, 3].to(self.complex_dtype)
            ),
            "fixed_D": self._batch_cavity_unitary(
                self._batch_displacement(
                    th.full(
                        (self.config.batch_size,),
                        self.config.fixed_layer4_displacement,
                        dtype=self.complex_dtype,
                        device=self.device,
                    )
                )
            ),
        }
        levels = th.arange(self.cutoff, device=self.device, dtype=self.real_dtype)
        phases = th.exp(1.0j * physical[:, 14, None].to(self.complex_dtype) * levels)
        samples["VR"] = self._batch_cavity_unitary(th.diag_embed(phases))
        result = {}
        for name, unitary in samples.items():
            residual = unitary.mH @ unitary - self.identity_joint[None, :, :]
            result[name] = float(th.max(th.linalg.matrix_norm(residual)).detach().cpu())
        return result

    def idle_completeness_residuals(self) -> dict[str, float]:
        th = _require_torch()
        result: dict[str, float] = {}
        for phase_id, channel_groups in self._idle_kraus.items():
            for channel_index, operators in enumerate(channel_groups):
                gram = th.einsum("kji,kjl->il", operators.conj(), operators)
                residual = th.linalg.matrix_norm(gram - self.identity_joint)
                result[f"{phase_id}:{channel_index}"] = float(residual.detach().cpu())
        return result

    def run(
        self,
        raw_corrections: Any | None = None,
        *,
        control_policy: Any | None = None,
        forced_outcomes: Any | Sequence[Sequence[int]] | None = None,
        seed: int = 0,
        record_cycle_metrics: bool = False,
    ) -> DifferentiableTrajectoryResult:
        th = _require_torch()
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise TypeError("seed must be an integer")
        if raw_corrections is not None and control_policy is not None:
            raise ValueError("provide raw_corrections or control_policy, not both")
        controls = (
            None
            if control_policy is not None
            else self.bounded_physical_controls(raw_corrections)
        )
        if forced_outcomes is None:
            forced = None
        else:
            forced = th.as_tensor(forced_outcomes, dtype=th.int64, device=self.device)
            expected = (self.config.batch_size, self.config.half_cycles)
            if tuple(forced.shape) != expected:
                raise ValueError(f"forced_outcomes must have shape {expected}")
            if not bool(th.all((forced == 0) | (forced == 1)).detach().cpu()):
                raise ValueError("forced outcomes must encode g=0 or e=1")
        generator = th.Generator(device="cpu")
        generator.manual_seed(int(seed))
        if self.config.device == "cuda":
            th.cuda.synchronize(self.device)
            th.cuda.reset_peak_memory_stats(self.device)
        start = time.perf_counter()
        state = self._initial_joint_density()
        if control_policy is not None:
            reset_rollout = getattr(control_policy, "reset_rollout", None)
            if callable(reset_rollout):
                reset_rollout(
                    batch_size=self.config.batch_size,
                    device=self.device,
                    dtype=self.real_dtype,
                )
        cycle_fidelities = []
        cycle_code_survival = []
        cycle_logical_z_signal = []
        cycle_conditional_logical_z = []
        if record_cycle_metrics:
            initial_cavity = self._reduce_cavity(state)
            initial_metrics = self._cavity_evaluation_metrics(initial_cavity)
            cycle_fidelities.append(initial_metrics[0])
            cycle_code_survival.append(initial_metrics[1])
            cycle_logical_z_signal.append(initial_metrics[2])
            cycle_conditional_logical_z.append(initial_metrics[3])
        log_probability = th.zeros(
            (self.config.batch_size,), dtype=self.real_dtype, device=self.device
        )
        outcomes = []
        conditional_probabilities = []
        applied_controls = []
        maximum_trace_error = 0.0
        maximum_hermiticity_error = 0.0
        for half_index in range(self.config.half_cycles):
            state = self._apply_idle(state, "entering_cycle")
            if control_policy is None:
                half_controls = controls[:, half_index, :]
            else:
                history = (
                    th.empty(
                        (self.config.batch_size, 0),
                        dtype=th.int64,
                        device=self.device,
                    )
                    if not outcomes
                    else th.stack(outcomes, dim=1)
                )
                half_controls = self._policy_controls(
                    control_policy, history, half_index
                )
            applied_controls.append(half_controls)
            for layer in range(1, 5):
                state = self._layer(state, half_controls, layer)
            state = self._apply_idle(state, "measurement_and_reset")
            state, outcome, probability = self._measure_and_reset(
                state,
                None if forced is None else forced[:, half_index],
                generator,
            )
            outcomes.append(outcome)
            conditional_probabilities.append(probability)
            log_probability = log_probability + th.log(
                th.clamp(probability, min=self.config.probability_floor)
            )
            state = self._virtual_rotation(state, half_controls[:, 14])
            state = self._apply_idle(state, "virtual_rotation_and_idle")
            trace_error, hermiticity_error, _ = self._diagnostics(state)
            maximum_trace_error = max(maximum_trace_error, trace_error)
            maximum_hermiticity_error = max(
                maximum_hermiticity_error, hermiticity_error
            )
            if record_cycle_metrics and (half_index + 1) % 2 == 0:
                checkpoint_cavity = self._reduce_cavity(state)
                checkpoint_metrics = self._cavity_evaluation_metrics(checkpoint_cavity)
                cycle_fidelities.append(checkpoint_metrics[0])
                cycle_code_survival.append(checkpoint_metrics[1])
                cycle_logical_z_signal.append(checkpoint_metrics[2])
                cycle_conditional_logical_z.append(checkpoint_metrics[3])
        cavity = self._reduce_cavity(state)
        reward = self._cavity_evaluation_metrics(cavity)[0]
        trajectory_probability = th.exp(log_probability)
        if self.config.device == "cuda":
            th.cuda.synchronize(self.device)
            peak = int(th.cuda.max_memory_allocated(self.device))
        else:
            peak = None
        wall_time = time.perf_counter() - start
        final_trace_error, final_hermiticity_error, minimum_eigenvalue = self._diagnostics(
            state
        )
        maximum_trace_error = max(maximum_trace_error, final_trace_error)
        maximum_hermiticity_error = max(
            maximum_hermiticity_error, final_hermiticity_error
        )
        complex_bytes = 16 if self.complex_dtype == th.complex128 else 8
        state_bytes = (
            self.config.batch_size
            * self.joint_dimension
            * self.joint_dimension
            * complex_bytes
        )
        half_cycles = self.config.half_cycles
        resources = TrajectoryResourceProfile(
            device=str(self.device),
            real_dtype=self.config.real_dtype,
            complex_dtype=str(self.complex_dtype).replace("torch.", ""),
            cutoff=self.cutoff,
            joint_dimension=self.joint_dimension,
            batch_size=self.config.batch_size,
            full_cycles=self.config.full_cycles,
            half_cycles=half_cycles,
            control_source=(
                "history_conditioned_policy"
                if control_policy is not None
                else "open_loop_raw_corrections"
            ),
            trainable_controls=self.config.batch_size * half_cycles * len(PARAMETER_NAMES),
            matrix_exponentials=self.config.batch_size * half_cycles * 4,
            unitary_applications=self.config.batch_size * half_cycles * 9,
            idle_windows=self.config.batch_size * half_cycles * 7,
            cptp_channel_applications=self.config.batch_size * half_cycles * 21,
            state_tensor_bytes=state_bytes,
            autograd_state_lower_bound_bytes=state_bytes * (1 + 7 * half_cycles),
            wall_time_seconds=wall_time,
            cuda_peak_allocated_bytes=peak,
            timing_profile_id=self.config.timing.profile_id,
            simulated_physical_time_ns=(
                half_cycles * self.config.timing.half_cycle_duration_ns
            ),
        )
        return DifferentiableTrajectoryResult(
            final_joint_density=state,
            final_cavity_density=cavity,
            outcomes=th.stack(outcomes, dim=1),
            conditional_probabilities=th.stack(conditional_probabilities, dim=1),
            log_probability=log_probability,
            trajectory_probability=trajectory_probability,
            reward=reward,
            physical_controls=th.stack(applied_controls, dim=1),
            cycle_fidelities=(
                th.stack(cycle_fidelities, dim=1) if record_cycle_metrics else None
            ),
            cycle_code_survival=(
                th.stack(cycle_code_survival, dim=1)
                if record_cycle_metrics
                else None
            ),
            cycle_logical_z_signal=(
                th.stack(cycle_logical_z_signal, dim=1)
                if record_cycle_metrics
                else None
            ),
            cycle_conditional_logical_z=(
                th.stack(cycle_conditional_logical_z, dim=1)
                if record_cycle_metrics
                else None
            ),
            resource_profile=resources,
            maximum_trace_error=maximum_trace_error,
            maximum_hermiticity_error=maximum_hermiticity_error,
            minimum_final_eigenvalue=minimum_eigenvalue,
        )


def run_differentiable_sbs_validation(
    *,
    device: TorchDeviceName = "cpu",
    cutoff: int = 8,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Run deterministic production gates and optionally write their JSON artifact."""

    th = _require_torch()
    config = DifferentiableSBSConfig(
        cutoff=cutoff,
        full_cycles=1,
        batch_size=4,
        grid_points=4097,
        device=device,
        real_dtype="float64",
    )
    simulator = DifferentiableSBSTrajectorySimulator(config)
    raw = th.linspace(
        -0.08,
        0.08,
        steps=config.half_cycles * len(PARAMETER_NAMES),
        dtype=simulator.real_dtype,
        device=simulator.device,
    ).reshape(1, config.half_cycles, len(PARAMETER_NAMES))
    raw = raw.expand(config.batch_size, -1, -1).clone().requires_grad_(True)
    all_branches = th.tensor(
        [[0, 0], [0, 1], [1, 0], [1, 1]],
        dtype=th.int64,
        device=simulator.device,
    )
    result = simulator.run(raw, forced_outcomes=all_branches, seed=314159)
    branch_probability_sum = float(result.trajectory_probability.sum().detach().cpu())
    scalar = result.reward.mean() + 0.01 * result.log_probability.mean()
    gradient = th.autograd.grad(scalar, raw, retain_graph=False)[0]
    gradient_norm = float(th.linalg.vector_norm(gradient).detach().cpu())
    policy_bias = th.zeros(
        (len(PARAMETER_NAMES),),
        dtype=simulator.real_dtype,
        device=simulator.device,
        requires_grad=True,
    )
    history_direction = th.linspace(
        -0.12,
        0.12,
        len(PARAMETER_NAMES),
        dtype=simulator.real_dtype,
        device=simulator.device,
    )

    def history_policy(history: Any, half_index: int) -> Any:
        if half_index == 0:
            signed_latest = th.zeros(
                (config.batch_size, 1),
                dtype=simulator.real_dtype,
                device=simulator.device,
            )
        else:
            signed_latest = (2.0 * history[:, -1:].to(simulator.real_dtype) - 1.0)
        return policy_bias[None, :] + signed_latest * history_direction[None, :]

    policy_result = simulator.run(
        control_policy=history_policy,
        forced_outcomes=all_branches,
        seed=314159,
    )
    policy_scalar = policy_result.reward.mean() + 0.01 * policy_result.log_probability.mean()
    policy_gradient = th.autograd.grad(policy_scalar, policy_bias)[0]
    policy_gradient_norm = float(th.linalg.vector_norm(policy_gradient).detach().cpu())
    history_control_separation = float(
        th.linalg.vector_norm(
            policy_result.physical_controls[0, 1]
            - policy_result.physical_controls[2, 1]
        )
        .detach()
        .cpu()
    )
    replay = simulator.run(raw.detach(), forced_outcomes=all_branches, seed=9)
    gate_residuals = simulator.gate_unitarity_residuals(raw.detach())
    idle_residuals = simulator.idle_completeness_residuals()
    max_gate_residual = max(gate_residuals.values())
    max_idle_residual = max(idle_residuals.values())
    checks = {
        "fixed_half_cycle_is_5000_ns": config.timing.half_cycle_duration_ns == 5000,
        "fixed_full_cycle_is_10000_ns": config.timing.full_cycle_duration_ns == 10000,
        "all_four_two_measurement_branches_normalize": abs(branch_probability_sum - 1.0)
        < 2.0e-9,
        "trajectory_probability_matches_log_probability": bool(
            th.allclose(
                result.trajectory_probability,
                th.exp(result.log_probability),
                atol=2.0e-12,
                rtol=2.0e-12,
            )
        ),
        "forced_replay_is_exact": bool(
            th.allclose(result.reward.detach(), replay.reward, atol=2.0e-12, rtol=2.0e-12)
            and th.allclose(
                result.trajectory_probability.detach(),
                replay.trajectory_probability,
                atol=2.0e-12,
                rtol=2.0e-12,
            )
        ),
        "reward_is_physical": bool(
            th.all((result.reward.detach() >= 0.0) & (result.reward.detach() <= 1.0))
        ),
        "autograd_graph_is_connected": bool(
            th.all(th.isfinite(gradient)).detach().cpu() and gradient_norm > 1.0e-10
        ),
        "history_conditioned_policy_changes_future_controls": history_control_separation
        > 1.0e-8,
        "history_conditioned_policy_gradient_is_connected": bool(
            th.all(th.isfinite(policy_gradient)).detach().cpu()
            and policy_gradient_norm > 1.0e-10
        ),
        "history_conditioned_branch_tree_normalizes": abs(
            float(policy_result.trajectory_probability.sum().detach().cpu()) - 1.0
        )
        < 2.0e-9,
        "gate_families_are_unitary": max_gate_residual < 2.0e-10,
        "idle_channels_are_trace_preserving": max_idle_residual < 2.0e-10,
        "trajectory_trace_is_preserved": result.maximum_trace_error < 2.0e-10,
        "trajectory_is_hermitian": result.maximum_hermiticity_error < 2.0e-10,
        "final_density_is_positive": result.minimum_final_eigenvalue > -2.0e-10,
        "resource_profile_counts_fifteen_controls_per_half_cycle": (
            result.resource_profile.trainable_controls
            == config.batch_size * config.half_cycles * 15
        ),
        "resource_profile_is_not_hardware_measurement": not result.resource_profile.target_hardware_measured,
    }
    payload = {
        "task_id": "T2.3.4",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "source_contract": {
            "primary_source": POROTTI_S1_SOURCE,
            "gate_definitions": (
                "relative_papers/Non-Markovian_feedback_for_optimized_quantum_error_"
                "correction/Non-Markovian_feedback_for_optimized_quantum_error_"
                "correction.md:411-434"
            ),
            "feedback_grape_probability_gradient": (
                "relative_papers/Non-Markovian_feedback_for_optimized_quantum_error_"
                "correction/Non-Markovian_feedback_for_optimized_quantum_error_"
                "correction.md:467-495"
            ),
            "parameter_contract": (
                "relative_papers/Non-Markovian_feedback_for_optimized_quantum_error_"
                "correction/Non-Markovian_feedback_for_optimized_quantum_error_"
                "correction.md:511-527"
            ),
        },
        "config": {
            **asdict(config),
            "timing": asdict(config.timing),
        },
        "metrics": {
            "branch_probability_sum": branch_probability_sum,
            "gradient_norm": gradient_norm,
            "policy_gradient_norm": policy_gradient_norm,
            "history_control_separation": history_control_separation,
            "maximum_gate_unitarity_residual": max_gate_residual,
            "maximum_idle_completeness_residual": max_idle_residual,
            "maximum_trace_error": result.maximum_trace_error,
            "maximum_hermiticity_error": result.maximum_hermiticity_error,
            "minimum_final_eigenvalue": result.minimum_final_eigenvalue,
            "reward_minimum": float(th.min(result.reward).detach().cpu()),
            "reward_maximum": float(th.max(result.reward).detach().cpu()),
            "trajectory_probability_minimum": float(
                th.min(result.trajectory_probability).detach().cpu()
            ),
            "trajectory_probability_maximum": float(
                th.max(result.trajectory_probability).detach().cpu()
            ),
        },
        "resource_profile": asdict(result.resource_profile),
        "gate_unitarity_residuals": gate_residuals,
        "idle_completeness_residuals": idle_residuals,
        "checks": checks,
        "scope": DIFFERENTIABLE_SBS_SCOPE,
        "forbidden_claims": (
            "not a pulse-level cavity-transmon Hamiltonian simulation",
            "not a leakage/SPAM/device-calibrated model",
            "not a Feedback-GRAPE gradient validation (reserved for T2.3.5)",
            "not a cutoff/batch/horizon feasibility envelope (reserved for T2.3.6)",
            "not a standard/MF/NMF ranking (reserved for T2.3.7)",
            "not target-board latency or hardware evidence",
        ),
    }
    if output is not None:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cutoff", type=int, default=8)
    parser.add_argument(
        "--output", default="docs/t2_3_4_differentiable_trajectory_validation.json"
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = run_differentiable_sbs_validation(
        device=args.device, cutoff=args.cutoff, output=args.output
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DIFFERENTIABLE_SBS_SCOPE",
    "POROTTI_S1_PROFILE_ID",
    "PARAMETER_NAMES",
    "TrajectoryTimingPhase",
    "DifferentiableSBSTimingProfile",
    "DifferentiableSBSConfig",
    "TrajectoryResourceProfile",
    "DifferentiableTrajectoryResult",
    "nominal_sbs_parameters",
    "DifferentiableSBSTrajectorySimulator",
    "run_differentiable_sbs_validation",
]
