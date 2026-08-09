"""Joint Fock×qutrit/GKSL/IQ/reset/drift simulator for backend A."""

from __future__ import annotations

from dataclasses import replace
from math import exp, isfinite, log, pi, sqrt
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import expm
from scipy.sparse import csr_matrix, eye as sparse_eye, kron as sparse_kron
from scipy.sparse.linalg import expm_multiply

from ..fock_density_model import FiniteCutoffDensity, FiniteCutoffFockModel
from ..fock_sbs_cycle import SBSFockCycleConfig, SBSFockOneRoundSimulator, logical_density
from ..phase9_twin_contract import ActionWord, NominalAction
from .schema import (
    ANCILLA_LEVELS, MAX_EXACT_CHOI_CUTOFF, BackendAConfig, BackendADriftState, BackendAEvaluatorState,
    BackendAExogenous, BackendALogicalRecord, BackendAObservation, BackendARoundResult,
    BackendAState, BackendATrajectory, BackendATruthRecord, ChannelDiagnostics, ComplexMatrix,
    _density_diagnostics, _nonnegative, _readonly, _validated_density, backend_a_exogenous,
)


class Phase9BackendASimulator:
    """Joint oscillator-qutrit trajectory simulator."""

    def __init__(self, config: BackendAConfig) -> None:
        if not isinstance(config, BackendAConfig):
            raise TypeError("config must be BackendAConfig")
        self.config = config
        self.cutoff = config.cutoff
        self.ancilla_dimension = 3
        self.dimension = self.cutoff * self.ancilla_dimension
        self.oscillator = FiniteCutoffFockModel(self.cutoff)

        self.i_o = self.oscillator.identity
        self.i_a = np.eye(3, dtype=np.complex128)
        self.i_joint = np.eye(self.dimension, dtype=np.complex128)
        self.a = self.oscillator.a
        self.adag = self.oscillator.adag
        self.number = self.oscillator.number
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
        self.sigma_ge = np.outer(g, e.conj())
        self.sigma_eg = self.sigma_ge.conj().T
        self.sigma_ef = np.outer(e, f.conj())
        self.sigma_fe = self.sigma_ef.conj().T
        self.y_ge = -1.0j * self.sigma_ge + 1.0j * self.sigma_eg
        self.x_ef = self.sigma_ef + self.sigma_fe

        self._joint_a = self._tensor(self.a, self.i_a)
        self._joint_number = self._tensor(self.number, self.i_a)
        self._joint_q = self._tensor(self.q, self.i_a)
        self._joint_p = self._tensor(self.p, self.i_a)
        self._joint_level_projectors = tuple(
            self._tensor(self.i_o, projector)
            for projector in self.level_projectors
        )
        self._joint_y_ge = self._tensor(self.i_o, self.y_ge)
        self._joint_x_ef = self._tensor(self.i_o, self.x_ef)

        self._collapse_operators = self._build_collapse_operators()
        self._dissipator = self._build_dissipator(self._collapse_operators)
        self._logical_simulator: SBSFockOneRoundSimulator | None = None

    @staticmethod
    def _tensor(left: ComplexMatrix, right: ComplexMatrix) -> ComplexMatrix:
        return np.kron(left, right)

    def _build_collapse_operators(self) -> tuple[ComplexMatrix, ...]:
        config = self.config
        rows: list[ComplexMatrix] = []

        def add(rate: float, operator: ComplexMatrix) -> None:
            if rate > 0.0:
                rows.append(sqrt(rate) * operator)

        add(config.oscillator_loss_rate, self._joint_a)
        add(config.oscillator_dephasing_rate, self._joint_number)
        add(
            config.ancilla_ge_relax_rate,
            self._tensor(self.i_o, self.sigma_ge),
        )
        add(
            config.ancilla_fe_relax_rate,
            self._tensor(self.i_o, self.sigma_ef),
        )
        add(
            config.ancilla_ge_excitation_rate,
            self._tensor(self.i_o, self.sigma_eg),
        )
        ancilla_phase = (
            -self.level_projectors[0]
            + self.level_projectors[1]
            + 2.0 * self.level_projectors[2]
        )
        add(
            config.ancilla_dephasing_rate,
            self._tensor(self.i_o, ancilla_phase),
        )
        return tuple(_readonly(operator) for operator in rows)

    def _build_dissipator(
        self,
        collapse_operators: Sequence[ComplexMatrix],
    ) -> csr_matrix:
        dimension = self.dimension
        identity = sparse_eye(dimension, dtype=np.complex128, format="csr")
        dissipator = csr_matrix(
            (dimension * dimension, dimension * dimension),
            dtype=np.complex128,
        )
        for operator in collapse_operators:
            collapse = csr_matrix(operator)
            gram = collapse.getH() @ collapse
            dissipator = (
                dissipator
                + sparse_kron(
                    collapse.conjugate(),
                    collapse,
                    format="csr",
                )
                - 0.5 * sparse_kron(identity, gram, format="csr")
                - 0.5
                * sparse_kron(gram.transpose(), identity, format="csr")
            )
        return dissipator.tocsr()

    def liouvillian(self, hamiltonian: ArrayLike) -> csr_matrix:
        h = np.asarray(hamiltonian, dtype=np.complex128)
        if h.shape != (self.dimension, self.dimension):
            raise ValueError("hamiltonian shape mismatch")
        if not np.all(np.isfinite(h)):
            raise ValueError("hamiltonian must be finite")
        if np.linalg.norm(h - h.conj().T, ord="fro") > 1.0e-10:
            raise ValueError("hamiltonian must be Hermitian")
        identity = sparse_eye(
            self.dimension,
            dtype=np.complex128,
            format="csr",
        )
        h_sparse = csr_matrix(h)
        commutator = -1.0j * (
            sparse_kron(identity, h_sparse, format="csr")
            - sparse_kron(h_sparse.transpose(), identity, format="csr")
        )
        return (commutator + self._dissipator).tocsr()

    def _base_hamiltonian(self, drift: BackendADriftState) -> ComplexMatrix:
        ancilla_dispersion = (
            self.level_projectors[1] + 2.0 * self.level_projectors[2]
        )
        kerr_operator = self.number @ (self.number - self.i_o)
        return _readonly(
            self.config.self_kerr * self._tensor(kerr_operator, self.i_a)
            + self.config.dispersive_chi
            * self._tensor(self.number, ancilla_dispersion)
            + drift.drive_q * self._joint_q
            + drift.drive_p * self._joint_p
            + drift.leakage_detuning * self._joint_level_projectors[2]
        )

    @staticmethod
    def _pulse_envelope(fraction: float) -> float:
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("pulse fraction must lie in [0,1]")
        # Integral over [0,1] equals one.
        return 0.5 * pi * np.sin(pi * fraction)

    def _action_alpha(self, action: ActionWord) -> complex:
        if not isinstance(action, ActionWord):
            raise TypeError("action must be ActionWord")
        amplitude = self.config.action_displacement / sqrt(2.0)
        return complex(
            amplitude * int(action.pauli_dx),
            amplitude * int(action.pauli_dz),
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

    def _evolve_segment(
        self,
        density: ComplexMatrix,
        duration: float,
        hamiltonian_at_fraction: Any,
    ) -> ComplexMatrix:
        if duration == 0.0:
            return _readonly(density)
        substeps = self.config.substeps_per_segment
        step = duration / substeps
        vector = np.asarray(density, dtype=np.complex128).reshape(
            self.dimension * self.dimension,
            order="F",
        )
        for index in range(substeps):
            midpoint = (index + 0.5) / substeps
            hamiltonian = np.asarray(
                hamiltonian_at_fraction(midpoint),
                dtype=np.complex128,
            )
            generator = self.liouvillian(hamiltonian)
            vector = expm_multiply(generator * step, vector)
        matrix = vector.reshape(
            (self.dimension, self.dimension),
            order="F",
        )
        matrix = 0.5 * (matrix + matrix.conj().T)
        trace = complex(np.trace(matrix))
        if abs(trace.imag) > 2.0e-9 or trace.real <= 0.0:
            raise RuntimeError("GKSL propagation produced an invalid trace")
        if abs(trace.real - 1.0) > 5.0e-8:
            raise RuntimeError(
                "GKSL propagation violated trace preservation before cleanup"
            )
        matrix = matrix / trace.real
        return _validated_density(
            matrix,
            self.dimension,
            "evolved_density",
            trace_tolerance=5.0e-9,
            positivity_tolerance=5.0e-9,
        )

    def _apply_action(
        self,
        density: ComplexMatrix,
        drift: BackendADriftState,
        action: ActionWord,
    ) -> ComplexMatrix:
        base = self._base_hamiltonian(drift)
        alpha = self._action_alpha(action)
        duration = self.config.action_duration
        drive = np.zeros_like(base)
        if duration > 0.0 and alpha != 0.0j:
            drive = (
                1.0j
                * (
                    alpha * self._tensor(self.adag, self.i_a)
                    - alpha.conjugate() * self._tensor(self.a, self.i_a)
                )
                / duration
            )
        leakage = np.zeros_like(base)
        if duration > 0.0:
            leakage = (
                self.config.action_leakage_coupling
                * self._action_energy(action)
                * self._joint_x_ef
                / duration
            )
        return self._evolve_segment(
            density,
            duration,
            lambda fraction: base
            + self._pulse_envelope(fraction) * (drive + leakage),
        )

    def _ramsey_pulse(
        self,
        density: ComplexMatrix,
        drift: BackendADriftState,
        angle: float,
    ) -> ComplexMatrix:
        duration = self.config.ramsey_pulse_duration
        if duration == 0.0 or angle == 0.0:
            return _readonly(density)
        base = self._base_hamiltonian(drift)
        ge_drive = angle * self._joint_y_ge / (2.0 * duration)
        leakage_drive = (
            abs(angle)
            * self.config.pulse_leakage_crosstalk
            * self._joint_x_ef
            / (2.0 * duration)
        )
        return self._evolve_segment(
            density,
            duration,
            lambda fraction: base
            + self._pulse_envelope(fraction)
            * (ge_drive + leakage_drive),
        )

    def _sense(
        self,
        density: ComplexMatrix,
        drift: BackendADriftState,
    ) -> ComplexMatrix:
        base = self._base_hamiltonian(drift)
        interaction = (
            self.config.measurement_leakage_coupling * self._joint_x_ef
        )
        return self._evolve_segment(
            density,
            self.config.sense_duration,
            lambda _fraction: base + interaction,
        )

    def ancilla_density(self, density: ArrayLike) -> ComplexMatrix:
        matrix = np.asarray(density, dtype=np.complex128)
        if matrix.shape != (self.dimension, self.dimension):
            raise ValueError("joint density shape mismatch")
        tensor = matrix.reshape(
            self.cutoff,
            3,
            self.cutoff,
            3,
        )
        result = np.trace(tensor, axis1=0, axis2=2)
        return _readonly(0.5 * (result + result.conj().T))

    def oscillator_density(self, density: ArrayLike) -> FiniteCutoffDensity:
        matrix = np.asarray(density, dtype=np.complex128)
        if matrix.shape != (self.dimension, self.dimension):
            raise ValueError("joint density shape mismatch")
        tensor = matrix.reshape(
            self.cutoff,
            3,
            self.cutoff,
            3,
        )
        result = np.trace(tensor, axis1=1, axis2=3)
        result = 0.5 * (result + result.conj().T)
        result /= float(np.trace(result).real)
        return FiniteCutoffDensity(result, self.cutoff)

    def level_probabilities(
        self,
        density: ArrayLike,
    ) -> tuple[float, float, float]:
        ancilla = self.ancilla_density(density)
        probabilities = np.real(np.diag(ancilla))
        probabilities = np.maximum(probabilities, 0.0)
        probabilities /= np.sum(probabilities)
        return tuple(float(item) for item in probabilities)

    @staticmethod
    def _sample_categorical(
        probabilities: Sequence[float],
        uniform: float,
    ) -> int:
        cumulative = 0.0
        for index, probability in enumerate(probabilities):
            cumulative += float(probability)
            if uniform < cumulative or index == len(probabilities) - 1:
                return index
        raise AssertionError("categorical sampler failed")

    def _measure_iq(
        self,
        density: ComplexMatrix,
        drift: BackendADriftState,
        exogenous: BackendAExogenous,
    ) -> tuple[
        ComplexMatrix,
        BackendAObservation,
        str,
        tuple[float, float, float],
        tuple[float, float, float],
    ]:
        if len(exogenous.iq_standard_i) != self.config.iq_samples:
            raise ValueError("exogenous IQ sample count does not match config")
        pre_probabilities = self.level_probabilities(density)
        sampled_index = self._sample_categorical(
            pre_probabilities,
            exogenous.emission_uniform,
        )
        centers = np.asarray(self.config.iq_centers, dtype=np.float64).copy()
        centers[:, 0] += drift.readout_i
        centers[:, 1] += drift.readout_q
        sigma = self.config.iq_sigma
        iq_i = (
            centers[sampled_index, 0]
            + sigma * np.asarray(exogenous.iq_standard_i, dtype=np.float64)
        )
        iq_q = (
            centers[sampled_index, 1]
            + sigma * np.asarray(exogenous.iq_standard_q, dtype=np.float64)
        )
        squared = (
            (iq_i[None, :] - centers[:, 0, None]) ** 2
            + (iq_q[None, :] - centers[:, 1, None]) ** 2
        )
        log_likelihood = (
            -0.5 * np.sum(squared, axis=1) / (sigma * sigma)
            - self.config.iq_samples * log(2.0 * pi * sigma * sigma)
        )
        maximum = float(np.max(log_likelihood))
        amplitudes = np.exp(0.5 * (log_likelihood - maximum))
        ancilla_kraus = np.diag(amplitudes.astype(np.complex128))
        kraus = self._tensor(self.i_o, ancilla_kraus)
        unnormalized = kraus @ density @ kraus.conj().T
        scaled_evidence = float(np.trace(unnormalized).real)
        if not isfinite(scaled_evidence) or scaled_evidence <= 0.0:
            raise RuntimeError("IQ measurement produced zero/invalid evidence")
        post = unnormalized / scaled_evidence
        post = _validated_density(
            post,
            self.dimension,
            "post_measurement_density",
            positivity_tolerance=5.0e-9,
        )
        posterior = self.level_probabilities(post)
        observation = BackendAObservation(
            iq_i=iq_i,
            iq_q=iq_q,
            integrated_i=float(np.mean(iq_i)),
            integrated_q=float(np.mean(iq_q)),
            log_evidence_density=maximum + log(scaled_evidence),
            posterior_levels=posterior,
            leakage_confidence_analog=posterior[2],
            reset_ack="none",
        )
        return (
            post,
            observation,
            ANCILLA_LEVELS[sampled_index],
            pre_probabilities,
            posterior,
        )

    def measurement_completeness_error(self) -> float:
        # Each level-conditioned 2D Gaussian integrates to one.  The remaining
        # operator identity is computed rather than asserted.
        completeness = sum(self._joint_level_projectors)
        return float(
            np.linalg.norm(completeness - self.i_joint, ord="fro")
        )

    def reset_kraus(self) -> dict[str, tuple[ComplexMatrix, ...]]:
        g, e, f = self.level_kets
        p_e = self.config.reset_success_e
        p_f = self.config.reset_success_f
        success = (
            self._tensor(self.i_o, np.outer(g, g.conj())),
            sqrt(p_e) * self._tensor(self.i_o, np.outer(g, e.conj())),
            sqrt(p_f) * self._tensor(self.i_o, np.outer(g, f.conj())),
        )
        failure_operator = (
            sqrt(1.0 - p_e) * np.outer(e, e.conj())
            + sqrt(1.0 - p_f) * np.outer(f, f.conj())
        )
        failure = (self._tensor(self.i_o, failure_operator),)
        return {
            "success": tuple(_readonly(item) for item in success),
            "failure": tuple(_readonly(item) for item in failure),
        }

    def reset_completeness_error(self) -> float:
        gram = np.zeros_like(self.i_joint)
        for group in self.reset_kraus().values():
            for operator in group:
                gram += operator.conj().T @ operator
        return float(np.linalg.norm(gram - self.i_joint, ord="fro"))

    def _reset(
        self,
        density: ComplexMatrix,
        exogenous: BackendAExogenous,
    ) -> tuple[ComplexMatrix, str, str]:
        branches: dict[str, ComplexMatrix] = {}
        probabilities: dict[str, float] = {}
        for outcome, operators in self.reset_kraus().items():
            branch = sum(
                operator @ density @ operator.conj().T
                for operator in operators
            )
            probability = float(np.trace(branch).real)
            branches[outcome] = branch
            probabilities[outcome] = max(probability, 0.0)
        total = sum(probabilities.values())
        if abs(total - 1.0) > 2.0e-9:
            raise RuntimeError("reset instrument probabilities do not sum to one")
        hidden = (
            "success"
            if exogenous.reset_uniform < probabilities["success"]
            else "failure"
        )
        probability = probabilities[hidden]
        if probability <= 0.0:
            raise RuntimeError("selected zero-probability reset branch")
        post = _validated_density(
            branches[hidden] / probability,
            self.dimension,
            "post_reset_density",
            positivity_tolerance=5.0e-9,
        )
        observed = hidden
        if exogenous.reset_ack_uniform < self.config.reset_ack_error:
            observed = "failure" if hidden == "success" else "success"
        return post, hidden, observed

    def _update_drift(
        self,
        drift: BackendADriftState,
        action: ActionWord,
        exogenous: BackendAExogenous,
    ) -> BackendADriftState:
        before = drift.vector()
        retention = np.asarray(
            self.config.drift_retention,
            dtype=np.float64,
        )
        noise = np.asarray(
            self.config.drift_noise_std,
            dtype=np.float64,
        ) * np.asarray(exogenous.drift_standard, dtype=np.float64)
        energy = self._action_energy(action)
        kick = np.array(
            [
                self.config.drift_action_kick * int(action.pauli_dx),
                self.config.drift_action_kick * int(action.pauli_dz),
                self.config.drift_readout_heating * energy,
                -0.5 * self.config.drift_readout_heating * energy,
                self.config.drift_leakage_heating * energy,
            ],
            dtype=np.float64,
        )
        return BackendADriftState.from_vector(retention * before + kick + noise)

    def _logical_engine(self) -> SBSFockOneRoundSimulator:
        if self.cutoff < 8:
            raise ValueError("logical GKP projection requires cutoff >= 8")
        if self._logical_simulator is None:
            self._logical_simulator = SBSFockOneRoundSimulator(
                SBSFockCycleConfig(
                    cutoff=self.cutoff,
                    projector_delta=self.config.logical_projector_delta,
                    grid_points=self.config.logical_grid_points,
                )
            )
        return self._logical_simulator

    def initialize_logical(
        self,
        label: str,
        *,
        ancilla_level: str = "g",
        drift: BackendADriftState | None = None,
    ) -> tuple[BackendAState, BackendAEvaluatorState]:
        if ancilla_level not in ANCILLA_LEVELS:
            raise ValueError("ancilla_level must be g/e/f")
        logical_engine = self._logical_engine()
        oscillator_state = logical_engine.initialize(label)
        ancilla = self.level_projectors[ANCILLA_LEVELS.index(ancilla_level)]
        state = BackendAState(
            joint_density=self._tensor(oscillator_state.matrix, ancilla),
            cutoff=self.cutoff,
            drift=BackendADriftState() if drift is None else drift,
        )
        evaluator = BackendAEvaluatorState(
            target_label=label,
            target_density=logical_density(label),
        )
        return state, evaluator

    def initialize_fock(
        self,
        *,
        oscillator_ket: ArrayLike | None = None,
        ancilla_state: str | ArrayLike = "g",
        drift: BackendADriftState | None = None,
    ) -> BackendAState:
        if oscillator_ket is None:
            ket = np.zeros(self.cutoff, dtype=np.complex128)
            ket[0] = 1.0
        else:
            ket = np.asarray(oscillator_ket, dtype=np.complex128)
        if ket.shape != (self.cutoff,) or not np.all(np.isfinite(ket)):
            raise ValueError("oscillator_ket must be finite with shape (cutoff,)")
        norm = float(np.vdot(ket, ket).real)
        if norm <= 0.0:
            raise ValueError("oscillator_ket must have nonzero norm")
        ket = ket / sqrt(norm)
        oscillator = np.outer(ket, ket.conj())
        if isinstance(ancilla_state, str):
            if ancilla_state not in ANCILLA_LEVELS:
                raise ValueError("ancilla_state must be g/e/f")
            ancilla = self.level_projectors[
                ANCILLA_LEVELS.index(ancilla_state)
            ]
        else:
            value = np.asarray(ancilla_state, dtype=np.complex128)
            if value.shape == (3,):
                value = value / sqrt(float(np.vdot(value, value).real))
                ancilla = np.outer(value, value.conj())
            else:
                ancilla = _validated_density(value, 3, "ancilla_state")
        return BackendAState(
            joint_density=self._tensor(oscillator, ancilla),
            cutoff=self.cutoff,
            drift=BackendADriftState() if drift is None else drift,
        )

    def logical_record(
        self,
        state: BackendAState,
        evaluator: BackendAEvaluatorState,
    ) -> BackendALogicalRecord:
        if state.cutoff != self.cutoff:
            raise ValueError("state cutoff mismatch")
        if not isinstance(evaluator, BackendAEvaluatorState):
            raise TypeError("evaluator must be BackendAEvaluatorState")
        oscillator = self.oscillator_density(state.joint_density)
        engine = self._logical_engine()
        encoded = (
            engine.code_basis.isometry.conj().T
            @ oscillator.matrix
            @ engine.code_basis.isometry
        )
        survival = float(np.trace(encoded).real)
        if survival <= 1.0e-12:
            raise RuntimeError("logical code survival is numerically zero")
        raw = 0.5 * (encoded / survival + (encoded / survival).conj().T)
        pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        pauli_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
        frame = np.eye(2, dtype=np.complex128)
        if evaluator.pauli_x:
            frame = pauli_x @ frame
        if evaluator.pauli_z:
            frame = pauli_z @ frame
        corrected = frame.conj().T @ raw @ frame
        pauli_y = np.array(
            [[0.0, -1.0j], [1.0j, 0.0]],
            dtype=np.complex128,
        )
        bloch = tuple(
            float(np.trace(corrected @ operator).real)
            for operator in (pauli_x, pauli_y, pauli_z)
        )
        fidelity = float(
            np.trace(corrected @ evaluator.target_density).real
        )
        fidelity = min(max(fidelity, 0.0), 1.0)
        return BackendALogicalRecord(
            code_survival_probability=min(max(survival, 0.0), 1.0),
            raw_logical_density=raw,
            frame_corrected_logical_density=corrected,
            bloch_xyz=bloch,
            target_fidelity=fidelity,
            logical_error=bool(fidelity < 0.5),
            evaluator_state=evaluator,
        )

    def step(
        self,
        state: BackendAState,
        action: ActionWord,
        exogenous: BackendAExogenous,
        *,
        evaluator: BackendAEvaluatorState | None = None,
    ) -> BackendARoundResult:
        if not isinstance(state, BackendAState):
            raise TypeError("state must be BackendAState")
        if state.cutoff != self.cutoff:
            raise ValueError("state cutoff mismatch")
        if not isinstance(action, ActionWord):
            raise TypeError("action must be a semantic+CRC-valid ActionWord")
        # Round-addressed randomness is part of the causal contract.
        if exogenous.round_index != state.round_index:
            raise ValueError("exogenous round index must equal state round index")
        if len(exogenous.iq_standard_i) != self.config.iq_samples:
            raise ValueError("exogenous IQ length mismatch")

        density = self._apply_action(
            state.joint_density,
            state.drift,
            action,
        )
        density = self._ramsey_pulse(
            density,
            state.drift,
            self.config.ramsey_angle,
        )
        density = self._sense(density, state.drift)
        density = self._ramsey_pulse(
            density,
            state.drift,
            -self.config.ramsey_angle,
        )
        (
            density,
            observation,
            sampled_level,
            pre_measurement,
            post_measurement,
        ) = self._measure_iq(density, state.drift, exogenous)
        pre_reset = self.level_probabilities(density)
        reset_hidden = "none"
        reset_observed = "none"
        if action.reset_request:
            density, reset_hidden, reset_observed = self._reset(
                density,
                exogenous,
            )
            observation = replace(observation, reset_ack=reset_observed)
        post_reset = self.level_probabilities(density)
        drift_after = self._update_drift(state.drift, action, exogenous)
        leakage_age = (
            min(state.leakage_age + 1, 65535)
            if post_reset[2] >= self.config.leakage_age_threshold
            else 0
        )
        next_state = BackendAState(
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
        alpha = self._action_alpha(action)
        truth = BackendATruthRecord(
            sampled_emission_level=sampled_level,
            pre_measurement_level_probabilities=pre_measurement,
            post_measurement_level_probabilities=post_measurement,
            reset_hidden_outcome=reset_hidden,
            pre_reset_level_probabilities=pre_reset,
            post_reset_level_probabilities=post_reset,
            action_code=NominalAction(action.action_code).name,
            action_alpha_real=float(alpha.real),
            action_alpha_imag=float(alpha.imag),
            drift_before=tuple(float(item) for item in state.drift.vector()),
            drift_after=tuple(float(item) for item in drift_after.vector()),
            density_diagnostics=_density_diagnostics(density),
        )
        return BackendARoundResult(
            state=next_state,
            observation=observation,
            truth=truth,
            logical=logical,
            action_word=action,
            exogenous=exogenous,
        )

    def simulate(
        self,
        initial_state: BackendAState,
        actions: Sequence[ActionWord],
        *,
        seed: int,
        evaluator: BackendAEvaluatorState | None = None,
    ) -> BackendATrajectory:
        state = initial_state
        active_evaluator = evaluator
        rounds: list[BackendARoundResult] = []
        for action in actions:
            exogenous = backend_a_exogenous(
                seed=seed,
                round_index=state.round_index,
                iq_samples=self.config.iq_samples,
            )
            result = self.step(
                state,
                action,
                exogenous,
                evaluator=active_evaluator,
            )
            rounds.append(result)
            state = result.state
            if result.logical is not None:
                active_evaluator = result.logical.evaluator_state
        return BackendATrajectory(
            rounds=tuple(rounds),
            initial_state=initial_state,
            final_state=state,
            seed=seed,
        )

    def channel_diagnostics(
        self,
        hamiltonian: ArrayLike,
        duration: float,
    ) -> ChannelDiagnostics:
        """Numerically construct the Choi matrix of one GKSL segment.

        This is intentionally used with a small cutoff in qualification.  It is
        an implementation check of vectorization, trace preservation and CP,
        not a statement that one successful output-state test proves CP.
        """

        if self.config.cutoff > MAX_EXACT_CHOI_CUTOFF:
            raise RuntimeError(
                "exact Choi construction is restricted to cutoff "
                f"<= {MAX_EXACT_CHOI_CUTOFF}; use scalable state/channel "
                "diagnostics at high cutoff"
            )
        time = _nonnegative(duration, "duration")
        generator = self.liouvillian(hamiltonian)
        superoperator = expm(generator.toarray() * time)
        dimension = self.dimension
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
                output = (
                    superoperator
                    @ basis.reshape(dimension * dimension, order="F")
                ).reshape((dimension, dimension), order="F")
                input_basis = np.zeros_like(basis)
                input_basis[row, column] = 1.0
                choi += np.kron(input_basis, output)
        hermiticity = float(
            np.linalg.norm(choi - choi.conj().T, ord="fro")
        )
        choi_hermitian = 0.5 * (choi + choi.conj().T)
        tensor = choi_hermitian.reshape(
            dimension,
            dimension,
            dimension,
            dimension,
        )
        partial_output = np.trace(tensor, axis1=1, axis2=3)
        return ChannelDiagnostics(
            dimension=dimension,
            choi_minimum_eigenvalue=float(
                np.min(np.linalg.eigvalsh(choi_hermitian))
            ),
            choi_trace=float(np.trace(choi_hermitian).real),
            trace_preservation_frobenius=float(
                np.linalg.norm(
                    partial_output - np.eye(dimension),
                    ord="fro",
                )
            ),
            hermiticity_frobenius=hermiticity,
        )
