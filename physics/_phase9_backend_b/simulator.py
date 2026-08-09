"""Independent dense Strang/Kraus simulator for Phase-9 backend B."""

from __future__ import annotations

from dataclasses import replace
from math import comb, exp, isfinite, log, pi, sin, sqrt
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import expm

from ..phase9_twin_contract import (
    ActionWord,
    NominalAction,
    execute_representative_probe,
    representative_action_probes,
)
from .models import (
    ANCILLA_LEVELS,
    MAX_EXACT_CHOI_CUTOFF,
    BackendBConfig,
    BackendBDrift,
    BackendBEvaluator,
    BackendBLogical,
    BackendBObservation,
    BackendBRandomRecord,
    BackendBRound,
    BackendBState,
    BackendBTrajectory,
    BackendBTruth,
    ComplexMatrix,
    _density,
    _diagnostics,
    _readonly,
    backend_b_random_record,
)


def diagnostic_action_word_b(action_name: str) -> ActionWord:
    if action_name not in {
        item.name for item in NominalAction if item != NominalAction.INVALID
    }:
        raise ValueError("unknown action")
    for probe in representative_action_probes():
        if probe.expected_terminal == action_name:
            return execute_representative_probe(probe)[-1].recurrence.action_word
    raise ValueError("T9.2.1 has no matching diagnostic action")


class Phase9BackendBSimulator:
    def __init__(self, config: BackendBConfig) -> None:
        if not isinstance(config, BackendBConfig):
            raise TypeError("config must be BackendBConfig")
        self.config = config
        self.cutoff = config.cutoff
        self.dimension = self.cutoff * 3
        self.i_o = np.eye(self.cutoff, dtype=np.complex128)
        self.i_a = np.eye(3, dtype=np.complex128)
        self.i_joint = np.eye(self.dimension, dtype=np.complex128)
        a = np.zeros((self.cutoff, self.cutoff), dtype=np.complex128)
        for number in range(1, self.cutoff):
            a[number - 1, number] = sqrt(float(number))
        self.a = a
        self.adag = a.conj().T
        self.number = self.adag @ self.a
        self.q = (self.a + self.adag) / sqrt(2.0)
        self.p = 1.0j * (self.adag - self.a) / sqrt(2.0)
        self.level_kets = tuple(
            np.eye(3, dtype=np.complex128)[:, index]
            for index in range(3)
        )
        self.level_projectors = tuple(
            np.outer(ket, ket.conj()) for ket in self.level_kets
        )
        g, e, f = self.level_kets
        self.ge_lower = np.outer(g, e.conj())
        self.ge_raise = self.ge_lower.conj().T
        self.ef_lower = np.outer(e, f.conj())
        self.ef_raise = self.ef_lower.conj().T
        self.y_ge = -1.0j * self.ge_lower + 1.0j * self.ge_raise
        self.x_ef = self.ef_lower + self.ef_raise
        self.joint_a = np.kron(self.a, self.i_a)
        self.joint_number = np.kron(self.number, self.i_a)
        self.joint_q = np.kron(self.q, self.i_a)
        self.joint_p = np.kron(self.p, self.i_a)
        self.joint_y_ge = np.kron(self.i_o, self.y_ge)
        self.joint_x_ef = np.kron(self.i_o, self.x_ef)
        self.joint_projectors = tuple(
            np.kron(self.i_o, projector)
            for projector in self.level_projectors
        )
        self._logical_isometry: ComplexMatrix | None = None

    @staticmethod
    def _matrix_exponential(matrix: ComplexMatrix) -> ComplexMatrix:
        return expm(matrix)

    @staticmethod
    def _apply_kraus(
        matrix: ComplexMatrix,
        operators: Sequence[ComplexMatrix],
    ) -> ComplexMatrix:
        result = np.zeros_like(matrix)
        for operator in operators:
            result += operator @ matrix @ operator.conj().T
        return result

    def _pure_loss_operators(self, duration: float) -> tuple[ComplexMatrix, ...]:
        transmissivity = exp(-self.config.oscillator_loss_rate * duration)
        rows: list[ComplexMatrix] = []
        for lost in range(self.cutoff):
            operator = np.zeros(
                (self.cutoff, self.cutoff),
                dtype=np.complex128,
            )
            for initial in range(lost, self.cutoff):
                remaining = initial - lost
                operator[remaining, initial] = (
                    sqrt(comb(initial, lost))
                    * (1.0 - transmissivity) ** (0.5 * lost)
                    * transmissivity ** (0.5 * remaining)
                )
            rows.append(np.kron(operator, self.i_a))
        return tuple(rows)

    def _local_amplitude_operators(
        self,
        source: int,
        target: int,
        probability: float,
    ) -> tuple[ComplexMatrix, ComplexMatrix]:
        no_jump = np.eye(3, dtype=np.complex128)
        no_jump[source, source] = sqrt(1.0 - probability)
        jump = np.zeros((3, 3), dtype=np.complex128)
        jump[target, source] = sqrt(probability)
        return np.kron(self.i_o, no_jump), np.kron(self.i_o, jump)

    def _dephase_oscillator(
        self,
        matrix: ComplexMatrix,
        duration: float,
    ) -> ComplexMatrix:
        if self.config.oscillator_dephasing_rate == 0.0:
            return matrix
        tensor = matrix.reshape(self.cutoff, 3, self.cutoff, 3)
        indices = np.arange(self.cutoff, dtype=np.float64)
        factor = np.exp(
            -0.5
            * self.config.oscillator_dephasing_rate
            * duration
            * (indices[:, None] - indices[None, :]) ** 2
        )
        tensor = tensor * factor[:, None, :, None]
        return tensor.reshape(self.dimension, self.dimension)

    def _dephase_ancilla(
        self,
        matrix: ComplexMatrix,
        duration: float,
    ) -> ComplexMatrix:
        if self.config.ancilla_dephasing_rate == 0.0:
            return matrix
        weights = np.array([-1.0, 1.0, 2.0], dtype=np.float64)
        factor = np.exp(
            -0.5
            * self.config.ancilla_dephasing_rate
            * duration
            * (weights[:, None] - weights[None, :]) ** 2
        )
        tensor = matrix.reshape(self.cutoff, 3, self.cutoff, 3)
        tensor = tensor * factor[None, :, None, :]
        return tensor.reshape(self.dimension, self.dimension)

    def _noise_channels(
        self,
        matrix: ComplexMatrix,
        duration: float,
    ) -> ComplexMatrix:
        result = matrix
        if self.config.oscillator_loss_rate > 0.0:
            result = self._apply_kraus(
                result,
                self._pure_loss_operators(duration),
            )
        result = self._dephase_oscillator(result, duration)
        local_channels = (
            (1, 0, self.config.ancilla_ge_relax_rate),
            (2, 1, self.config.ancilla_fe_relax_rate),
            (0, 1, self.config.ancilla_ge_excitation_rate),
        )
        for source, target, rate in local_channels:
            if rate > 0.0:
                probability = 1.0 - exp(-rate * duration)
                result = self._apply_kraus(
                    result,
                    self._local_amplitude_operators(
                        source,
                        target,
                        probability,
                    ),
                )
        result = self._dephase_ancilla(result, duration)
        return result

    def channel_completeness_errors(self, duration: float) -> dict[str, float]:
        errors: dict[str, float] = {}
        gram = np.zeros_like(self.i_joint)
        for operator in self._pure_loss_operators(duration):
            gram += operator.conj().T @ operator
        errors["pure_loss"] = float(
            np.linalg.norm(gram - self.i_joint, ord="fro")
        )
        for label, source, target, rate in (
            ("ge_relax", 1, 0, self.config.ancilla_ge_relax_rate),
            ("fe_relax", 2, 1, self.config.ancilla_fe_relax_rate),
            ("ge_excite", 0, 1, self.config.ancilla_ge_excitation_rate),
        ):
            probability = 1.0 - exp(-rate * duration)
            operators = self._local_amplitude_operators(
                source,
                target,
                probability,
            )
            local_gram = sum(
                operator.conj().T @ operator for operator in operators
            )
            errors[label] = float(
                np.linalg.norm(local_gram - self.i_joint, ord="fro")
            )
        return errors

    def _base_hamiltonian(self, drift: BackendBDrift) -> ComplexMatrix:
        dispersion = self.level_projectors[1] + 2.0 * self.level_projectors[2]
        kerr = self.number @ (self.number - self.i_o)
        return (
            self.config.self_kerr * np.kron(kerr, self.i_a)
            + self.config.dispersive_chi * np.kron(self.number, dispersion)
            + drift.drive_q * self.joint_q
            + drift.drive_p * self.joint_p
            + drift.leakage_detuning * self.joint_projectors[2]
        )

    @staticmethod
    def _envelope(fraction: float) -> float:
        return 0.5 * pi * sin(pi * fraction)

    def _split_segment(
        self,
        density: ComplexMatrix,
        duration: float,
        hamiltonian_at: Any,
    ) -> ComplexMatrix:
        if duration == 0.0:
            return _readonly(density)
        steps = self.config.split_steps_per_segment
        dt = duration / steps
        result = np.asarray(density, dtype=np.complex128)
        for index in range(steps):
            midpoint = (index + 0.5) / steps
            hamiltonian = np.asarray(
                hamiltonian_at(midpoint),
                dtype=np.complex128,
            )
            if np.linalg.norm(
                hamiltonian - hamiltonian.conj().T,
                ord="fro",
            ) > 1.0e-10:
                raise ValueError("Hamiltonian must be Hermitian")
            half = self._matrix_exponential(-0.5j * dt * hamiltonian)
            result = half @ result @ half.conj().T
            result = self._noise_channels(result, dt)
            result = half @ result @ half.conj().T
        raw_trace = complex(np.trace(result))
        if (
            abs(raw_trace.imag) > 5.0e-9
            or abs(raw_trace.real - 1.0) > 1.0e-8
        ):
            raise RuntimeError(
                "dense split propagation violated trace preservation"
            )
        result = result / raw_trace.real
        return _density(result, self.dimension, "split_output", tolerance=1.0e-8)

    def _action_alpha(self, action: ActionWord) -> complex:
        scale = self.config.action_displacement / sqrt(2.0)
        return complex(
            scale * int(action.pauli_dx),
            scale * int(action.pauli_dz),
        )

    def _action_energy(self, action: ActionWord) -> float:
        code = NominalAction(action.action_code)
        if code == NominalAction.RESET:
            return self.config.reset_action_energy
        if code == NominalAction.XZ:
            return 2.0
        if code in {NominalAction.X, NominalAction.Z}:
            return 1.0
        if code == NominalAction.HOLD:
            return 0.05
        if code == NominalAction.LKG_HOLD:
            return 0.02
        return 0.0

    def _apply_action(
        self,
        density: ComplexMatrix,
        drift: BackendBDrift,
        action: ActionWord,
    ) -> ComplexMatrix:
        base = self._base_hamiltonian(drift)
        duration = self.config.action_duration
        alpha = self._action_alpha(action)
        drive = np.zeros_like(base)
        leakage = np.zeros_like(base)
        if duration > 0.0:
            drive = 1.0j * (
                alpha * np.kron(self.adag, self.i_a)
                - alpha.conjugate() * np.kron(self.a, self.i_a)
            ) / duration
            leakage = (
                self.config.action_leakage_coupling
                * self._action_energy(action)
                * self.joint_x_ef
                / duration
            )
        return self._split_segment(
            density,
            duration,
            lambda fraction: base
            + self._envelope(fraction) * (drive + leakage),
        )

    def _pulse(
        self,
        density: ComplexMatrix,
        drift: BackendBDrift,
        angle: float,
    ) -> ComplexMatrix:
        duration = self.config.ramsey_pulse_duration
        if duration == 0.0 or angle == 0.0:
            return _readonly(density)
        base = self._base_hamiltonian(drift)
        ge = angle * self.joint_y_ge / (2.0 * duration)
        ef = (
            abs(angle)
            * self.config.pulse_leakage_crosstalk
            * self.joint_x_ef
            / (2.0 * duration)
        )
        return self._split_segment(
            density,
            duration,
            lambda fraction: base + self._envelope(fraction) * (ge + ef),
        )

    def _sense(
        self,
        density: ComplexMatrix,
        drift: BackendBDrift,
    ) -> ComplexMatrix:
        base = self._base_hamiltonian(drift)
        return self._split_segment(
            density,
            self.config.sense_duration,
            lambda _fraction: base
            + self.config.measurement_leakage_coupling * self.joint_x_ef,
        )

    def ancilla_density(self, density: ArrayLike) -> ComplexMatrix:
        matrix = np.asarray(density, dtype=np.complex128)
        tensor = matrix.reshape(self.cutoff, 3, self.cutoff, 3)
        result = np.trace(tensor, axis1=0, axis2=2)
        return _readonly(0.5 * (result + result.conj().T))

    def oscillator_density(self, density: ArrayLike) -> ComplexMatrix:
        matrix = np.asarray(density, dtype=np.complex128)
        tensor = matrix.reshape(self.cutoff, 3, self.cutoff, 3)
        result = np.trace(tensor, axis1=1, axis2=3)
        result = 0.5 * (result + result.conj().T)
        trace = float(np.trace(result).real)
        if abs(trace - 1.0) > 1.0e-8:
            raise RuntimeError("oscillator partial trace is not normalized")
        return _readonly(result / trace)

    def level_probabilities(
        self,
        density: ArrayLike,
    ) -> tuple[float, float, float]:
        diagonal = np.real(np.diag(self.ancilla_density(density)))
        diagonal = np.maximum(diagonal, 0.0)
        diagonal /= np.sum(diagonal)
        return tuple(float(item) for item in diagonal)

    @staticmethod
    def _categorical(probabilities: Sequence[float], uniform: float) -> int:
        cumulative = 0.0
        for index, probability in enumerate(probabilities):
            cumulative += probability
            if uniform < cumulative or index == len(probabilities) - 1:
                return index
        raise AssertionError("categorical failure")

    def _measure(
        self,
        density: ComplexMatrix,
        drift: BackendBDrift,
        record: BackendBRandomRecord,
    ) -> tuple[
        ComplexMatrix,
        BackendBObservation,
        str,
        tuple[float, float, float],
        tuple[float, float, float],
    ]:
        if len(record.iq_normal_i) != self.config.iq_samples:
            raise ValueError("random IQ length mismatch")
        prior = self.level_probabilities(density)
        component = self._categorical(prior, record.component_uniform)
        centers = np.asarray(self.config.iq_centers, dtype=np.float64).copy()
        centers[:, 0] += drift.readout_i
        centers[:, 1] += drift.readout_q
        sigma = self.config.iq_sigma
        i_values = (
            centers[component, 0]
            + sigma * np.asarray(record.iq_normal_i, dtype=np.float64)
        )
        q_values = (
            centers[component, 1]
            + sigma * np.asarray(record.iq_normal_q, dtype=np.float64)
        )
        log_likelihood: list[float] = []
        normalization = -log(2.0 * pi * sigma * sigma)
        for center_i, center_q in centers:
            total = 0.0
            for sample_i, sample_q in zip(i_values, q_values):
                total += normalization - (
                    (sample_i - center_i) ** 2
                    + (sample_q - center_q) ** 2
                ) / (2.0 * sigma * sigma)
            log_likelihood.append(total)
        maximum = max(log_likelihood)
        amplitudes = np.array(
            [exp(0.5 * (item - maximum)) for item in log_likelihood],
            dtype=np.complex128,
        )
        operator = np.kron(self.i_o, np.diag(amplitudes))
        unnormalized = operator @ density @ operator.conj().T
        evidence_scaled = float(np.trace(unnormalized).real)
        if evidence_scaled <= 0.0 or not isfinite(evidence_scaled):
            raise RuntimeError("measurement evidence invalid")
        post = _density(
            unnormalized / evidence_scaled,
            self.dimension,
            "measurement_output",
            tolerance=1.0e-8,
        )
        posterior = self.level_probabilities(post)
        observation = BackendBObservation(
            iq_i=i_values,
            iq_q=q_values,
            integrated_i=float(np.mean(i_values)),
            integrated_q=float(np.mean(q_values)),
            log_evidence_density=maximum + log(evidence_scaled),
            posterior_levels=posterior,
            leakage_confidence_analog=posterior[2],
            reset_ack="none",
        )
        return post, observation, ANCILLA_LEVELS[component], prior, posterior

    def measurement_completeness_error(self) -> float:
        return float(
            np.linalg.norm(
                sum(self.joint_projectors) - self.i_joint,
                ord="fro",
            )
        )

    def _reset_operators(self) -> dict[str, tuple[ComplexMatrix, ...]]:
        g, e, f = self.level_kets
        success = (
            np.kron(self.i_o, np.outer(g, g.conj())),
            sqrt(self.config.reset_success_e)
            * np.kron(self.i_o, np.outer(g, e.conj())),
            sqrt(self.config.reset_success_f)
            * np.kron(self.i_o, np.outer(g, f.conj())),
        )
        failed_local = (
            sqrt(1.0 - self.config.reset_success_e)
            * np.outer(e, e.conj())
            + sqrt(1.0 - self.config.reset_success_f)
            * np.outer(f, f.conj())
        )
        return {
            "success": success,
            "failure": (np.kron(self.i_o, failed_local),),
        }

    def reset_completeness_error(self) -> float:
        gram = np.zeros_like(self.i_joint)
        for operators in self._reset_operators().values():
            for operator in operators:
                gram += operator.conj().T @ operator
        return float(np.linalg.norm(gram - self.i_joint, ord="fro"))

    def _reset(
        self,
        density: ComplexMatrix,
        record: BackendBRandomRecord,
    ) -> tuple[ComplexMatrix, str, str]:
        branches: dict[str, ComplexMatrix] = {}
        probabilities: dict[str, float] = {}
        for outcome, operators in self._reset_operators().items():
            branch = self._apply_kraus(density, operators)
            branches[outcome] = branch
            probabilities[outcome] = max(float(np.trace(branch).real), 0.0)
        if abs(sum(probabilities.values()) - 1.0) > 2.0e-9:
            raise RuntimeError("reset probability normalization failed")
        hidden = (
            "success"
            if record.reset_uniform < probabilities["success"]
            else "failure"
        )
        selected = probabilities[hidden]
        if selected <= 0.0:
            raise RuntimeError("selected impossible reset outcome")
        post = _density(
            branches[hidden] / selected,
            self.dimension,
            "reset_output",
            tolerance=1.0e-8,
        )
        observed = hidden
        if record.ack_uniform < self.config.reset_ack_error:
            observed = "failure" if hidden == "success" else "success"
        return post, hidden, observed

    def _drift_update(
        self,
        drift: BackendBDrift,
        action: ActionWord,
        record: BackendBRandomRecord,
    ) -> BackendBDrift:
        retention = np.asarray(self.config.drift_retention)
        noise = np.asarray(self.config.drift_noise_std) * np.asarray(
            record.drift_normal
        )
        energy = self._action_energy(action)
        forcing = np.array(
            [
                self.config.drift_action_kick * int(action.pauli_dx),
                self.config.drift_action_kick * int(action.pauli_dz),
                self.config.drift_readout_heating * energy,
                -0.5 * self.config.drift_readout_heating * energy,
                self.config.drift_leakage_heating * energy,
            ]
        )
        return BackendBDrift.from_vector(
            retention * drift.vector() + forcing + noise
        )

    def _comb_isometry(self) -> ComplexMatrix:
        if self.cutoff < 8:
            raise ValueError("logical comb requires cutoff >= 8")
        if self._logical_isometry is not None:
            return self._logical_isometry
        squeeze_generator = 0.5 * self.config.comb_squeezing * (
            self.a @ self.a - self.adag @ self.adag
        )
        squeezed_vacuum = self._matrix_exponential(squeeze_generator)[:, 0]
        columns: list[ComplexMatrix] = []
        for bit in (0, 1):
            vector = np.zeros(self.cutoff, dtype=np.complex128)
            for index in range(
                -self.config.comb_half_width,
                self.config.comb_half_width + 1,
            ):
                q_position = (2 * index + bit) * sqrt(pi)
                alpha = q_position / sqrt(2.0)
                displacement = self._matrix_exponential(
                    alpha * self.adag - alpha * self.a
                )
                weight = exp(
                    -0.5 * self.config.comb_envelope * q_position**2
                )
                vector += weight * (displacement @ squeezed_vacuum)
            vector /= sqrt(float(np.vdot(vector, vector).real))
            columns.append(vector)
        raw = np.column_stack(columns)
        gram = raw.conj().T @ raw
        values, vectors = np.linalg.eigh(gram)
        if float(np.min(values)) <= 1.0e-10:
            raise RuntimeError("independent logical comb is singular")
        inverse_root = (vectors * (1.0 / np.sqrt(values))) @ vectors.conj().T
        self._logical_isometry = _readonly(raw @ inverse_root)
        return self._logical_isometry

    @staticmethod
    def _logical_target(label: str) -> ComplexMatrix:
        identity = np.eye(2, dtype=np.complex128)
        x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
        z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
        values = {
            "0": 0.5 * (identity + z),
            "1": 0.5 * (identity - z),
            "+": 0.5 * (identity + x),
            "-": 0.5 * (identity - x),
            "+i": 0.5 * (identity + y),
            "-i": 0.5 * (identity - y),
        }
        if label not in values:
            raise ValueError("unknown logical label")
        return values[label]

    def initialize_logical(
        self,
        label: str,
        *,
        ancilla_level: str = "g",
        drift: BackendBDrift | None = None,
    ) -> tuple[BackendBState, BackendBEvaluator]:
        if ancilla_level not in ANCILLA_LEVELS:
            raise ValueError("ancilla level invalid")
        logical = self._logical_target(label)
        isometry = self._comb_isometry()
        oscillator = isometry @ logical @ isometry.conj().T
        ancilla = self.level_projectors[ANCILLA_LEVELS.index(ancilla_level)]
        return (
            BackendBState(
                joint_density=np.kron(oscillator, ancilla),
                cutoff=self.cutoff,
                drift=BackendBDrift() if drift is None else drift,
            ),
            BackendBEvaluator(
                target_label=label,
                target_density=logical,
            ),
        )

    def initialize_fock(
        self,
        *,
        oscillator_ket: ArrayLike | None = None,
        ancilla_state: str | ArrayLike = "g",
        drift: BackendBDrift | None = None,
    ) -> BackendBState:
        if oscillator_ket is None:
            ket = np.zeros(self.cutoff, dtype=np.complex128)
            ket[0] = 1.0
        else:
            ket = np.asarray(oscillator_ket, dtype=np.complex128)
        if ket.shape != (self.cutoff,) or not np.all(np.isfinite(ket)):
            raise ValueError("oscillator ket shape/finite failure")
        ket_norm = float(np.vdot(ket, ket).real)
        if ket_norm <= 1.0e-15:
            raise ValueError("oscillator ket must have nonzero norm")
        ket = ket / sqrt(ket_norm)
        oscillator = np.outer(ket, ket.conj())
        if isinstance(ancilla_state, str):
            if ancilla_state not in ANCILLA_LEVELS:
                raise ValueError("ancilla state invalid")
            ancilla = self.level_projectors[
                ANCILLA_LEVELS.index(ancilla_state)
            ]
        else:
            value = np.asarray(ancilla_state, dtype=np.complex128)
            if value.shape == (3,):
                value_norm = float(np.vdot(value, value).real)
                if not np.all(np.isfinite(value)) or value_norm <= 1.0e-15:
                    raise ValueError(
                        "ancilla ket must be finite and have nonzero norm"
                    )
                value = value / sqrt(value_norm)
                ancilla = np.outer(value, value.conj())
            else:
                ancilla = _density(value, 3, "ancilla_state")
        return BackendBState(
            joint_density=np.kron(oscillator, ancilla),
            cutoff=self.cutoff,
            drift=BackendBDrift() if drift is None else drift,
        )

    def logical_record(
        self,
        state: BackendBState,
        evaluator: BackendBEvaluator,
    ) -> BackendBLogical:
        isometry = self._comb_isometry()
        oscillator = self.oscillator_density(state.joint_density)
        encoded = isometry.conj().T @ oscillator @ isometry
        survival = float(np.trace(encoded).real)
        if survival <= 1.0e-12:
            raise RuntimeError("logical support vanished")
        raw = encoded / survival
        x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
        z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
        frame = np.eye(2, dtype=np.complex128)
        if evaluator.pauli_x:
            frame = x @ frame
        if evaluator.pauli_z:
            frame = z @ frame
        corrected = frame.conj().T @ raw @ frame
        fidelity = min(
            max(float(np.trace(corrected @ evaluator.target_density).real), 0.0),
            1.0,
        )
        return BackendBLogical(
            code_survival=min(max(survival, 0.0), 1.0),
            raw_density=_density(raw, 2, "raw_logical"),
            corrected_density=_density(corrected, 2, "corrected_logical"),
            bloch_xyz=tuple(
                float(np.trace(corrected @ operator).real)
                for operator in (x, y, z)
            ),
            target_fidelity=fidelity,
            logical_error=bool(fidelity < 0.5),
            evaluator=evaluator,
        )

    def step(
        self,
        state: BackendBState,
        action: ActionWord,
        random_record: BackendBRandomRecord,
        *,
        evaluator: BackendBEvaluator | None = None,
    ) -> BackendBRound:
        if state.cutoff != self.cutoff:
            raise ValueError("state cutoff mismatch")
        if not isinstance(action, ActionWord):
            raise TypeError("action must be ActionWord")
        if random_record.round_index != state.round_index:
            raise ValueError("random record round mismatch")
        density = self._apply_action(
            state.joint_density,
            state.drift,
            action,
        )
        density = self._pulse(
            density,
            state.drift,
            self.config.ramsey_angle,
        )
        density = self._sense(density, state.drift)
        density = self._pulse(
            density,
            state.drift,
            -self.config.ramsey_angle,
        )
        (
            density,
            observation,
            sampled,
            pre_measurement,
            post_measurement,
        ) = self._measure(density, state.drift, random_record)
        pre_reset = self.level_probabilities(density)
        hidden_reset = "none"
        observed_reset = "none"
        if action.reset_request:
            density, hidden_reset, observed_reset = self._reset(
                density,
                random_record,
            )
            observation = replace(
                observation,
                reset_ack=observed_reset,
            )
        post_reset = self.level_probabilities(density)
        drift_after = self._drift_update(
            state.drift,
            action,
            random_record,
        )
        leakage_age = (
            min(state.leakage_age + 1, 65535)
            if post_reset[2] >= self.config.leakage_age_threshold
            else 0
        )
        next_state = BackendBState(
            joint_density=density,
            cutoff=self.cutoff,
            drift=drift_after,
            leakage_age=leakage_age,
            round_index=state.round_index + 1,
        )
        next_evaluator = evaluator.after(action) if evaluator is not None else None
        logical = (
            self.logical_record(next_state, next_evaluator)
            if next_evaluator is not None
            else None
        )
        truth = BackendBTruth(
            sampled_component=sampled,
            pre_measurement_levels=pre_measurement,
            post_measurement_levels=post_measurement,
            reset_hidden_outcome=hidden_reset,
            pre_reset_levels=pre_reset,
            post_reset_levels=post_reset,
            action_code=NominalAction(action.action_code).name,
            drift_before=tuple(float(item) for item in state.drift.vector()),
            drift_after=tuple(float(item) for item in drift_after.vector()),
            density_diagnostics=_diagnostics(density),
        )
        return BackendBRound(
            state=next_state,
            observation=observation,
            truth=truth,
            logical=logical,
            action=action,
            random_record=random_record,
        )

    def simulate(
        self,
        initial: BackendBState,
        actions: Sequence[ActionWord],
        *,
        seed: int,
        evaluator: BackendBEvaluator | None = None,
    ) -> BackendBTrajectory:
        state = initial
        active_evaluator = evaluator
        rows: list[BackendBRound] = []
        for action in actions:
            record = backend_b_random_record(
                seed=seed,
                round_index=state.round_index,
                iq_samples=self.config.iq_samples,
            )
            result = self.step(
                state,
                action,
                record,
                evaluator=active_evaluator,
            )
            rows.append(result)
            state = result.state
            if result.logical is not None:
                active_evaluator = result.logical.evaluator
        return BackendBTrajectory(
            rounds=tuple(rows),
            initial_state=initial,
            final_state=state,
            seed=seed,
        )

    def split_channel_choi(
        self,
        hamiltonian: ComplexMatrix,
        duration: float,
    ) -> tuple[float, float, float]:
        if self.config.cutoff > MAX_EXACT_CHOI_CUTOFF:
            raise RuntimeError(
                "exact Choi construction is restricted to cutoff "
                f"<= {MAX_EXACT_CHOI_CUTOFF}; use scalable state/channel "
                "diagnostics at high cutoff"
            )
        dimension = self.dimension
        half = self._matrix_exponential(-0.5j * duration * hamiltonian)

        def channel(matrix: ComplexMatrix) -> ComplexMatrix:
            result = half @ matrix @ half.conj().T
            result = self._noise_channels(result, duration)
            return half @ result @ half.conj().T

        choi = np.zeros(
            (dimension * dimension, dimension * dimension),
            dtype=np.complex128,
        )
        for row in range(dimension):
            for column in range(dimension):
                basis = np.zeros(
                    (dimension, dimension),
                    dtype=np.complex128,
                )
                basis[row, column] = 1.0
                input_basis = np.zeros_like(basis)
                input_basis[row, column] = 1.0
                choi += np.kron(input_basis, channel(basis))
        choi = 0.5 * (choi + choi.conj().T)
        tensor = choi.reshape(
            dimension,
            dimension,
            dimension,
            dimension,
        )
        partial = np.trace(tensor, axis1=1, axis2=3)
        return (
            float(np.min(np.linalg.eigvalsh(choi))),
            float(np.linalg.norm(partial - np.eye(dimension), ord="fro")),
            float(np.linalg.norm(choi - choi.conj().T, ord="fro")),
        )
