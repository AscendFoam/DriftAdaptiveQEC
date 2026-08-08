"""Finite-dimensional Petz/QEC-matrix and SDP channel-recovery bounds.

The target channel is an encoded logical channel ``A = N o E`` with Kraus
operators of shape ``(physical_dimension, logical_dimension)``.  Recovery is
allowed to be any CPTP map from the physical output back to the logical space.
This module therefore evaluates an encoding--noise property; it does not model
or claim a deployable decoder, controller, pulse sequence, or hardware result.

The QEC-matrix convention follows Zheng et al., PRL 132, 250602 (2024):

``M[mu,l; nu,k] = <mu| A_l^dagger A_k |nu>``.

The transpose/Petz recovery fidelity is

``F_tilde = ||Tr_L sqrt(M)||_F^2 / d_L^2``

and obeys ``F_tilde <= F_opt <= (1 + F_tilde) / 2``.  For small physical
dimensions, :func:`solve_optimal_recovery_sdp` additionally produces a repaired
primal CPTP lower certificate and a shifted dual-feasible upper certificate.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb, exp, isfinite, sqrt
from typing import Any, Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from physics.fock_density_model import FiniteCutoffFockModel


ComplexMatrix = NDArray[np.complex128]


def _positive_integer(value: int, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _finite_positive(value: float, name: str) -> float:
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _matrix(value: ArrayLike, name: str) -> ComplexMatrix:
    result = np.asarray(value, dtype=np.complex128)
    if result.ndim != 2 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite matrix")
    return result


def _psd_eigh(
    value: ArrayLike,
    name: str,
    *,
    negative_tolerance: float = 2.0e-8,
) -> tuple[NDArray[np.float64], ComplexMatrix, float, float]:
    matrix = _matrix(value, name)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square")
    hermiticity = float(np.linalg.norm(matrix - matrix.conj().T, ord="fro"))
    if hermiticity > 2.0e-8:
        raise ValueError(f"{name} must be Hermitian")
    hermitian = 0.5 * (matrix + matrix.conj().T)
    values, vectors = np.linalg.eigh(hermitian)
    minimum = float(values[0])
    if minimum < -negative_tolerance:
        raise ValueError(f"{name} is not positive semidefinite")
    return np.maximum(values, 0.0), vectors, minimum, hermiticity


def finite_cutoff_gkp_isometry(
    cutoff: int,
    projector_delta: float,
    *,
    grid_points: int = 8193,
) -> ComplexMatrix:
    """Return the registered orthonormal finite-cutoff ``|0_L>,|1_L>`` basis."""

    dimension = _positive_integer(cutoff, "cutoff", minimum=4)
    if dimension > 48:
        raise ValueError("formal finite-cutoff recovery scan is limited to cutoff <= 48")
    delta = _finite_positive(projector_delta, "projector_delta")
    points = _positive_integer(grid_points, "grid_points", minimum=1025)
    if points % 2 == 0:
        raise ValueError("grid_points must be odd")
    model = FiniteCutoffFockModel(dimension)
    columns = []
    for logical_state in ("0", "1"):
        preparation = model.prepare_damped_projector_gkp(
            logical_state,
            delta,
            grid_points=points,
            source_coordinate_scale=sqrt(2.0),
        )
        coefficients = np.asarray(preparation.coefficients, dtype=np.complex128)
        norm = float(np.linalg.norm(coefficients))
        if not isfinite(norm) or norm <= 1.0e-12:
            raise RuntimeError("finite-cutoff GKP projection has zero or invalid norm")
        columns.append(coefficients / norm)
    raw = np.column_stack(columns)
    gram = raw.conj().T @ raw
    values, vectors = np.linalg.eigh(0.5 * (gram + gram.conj().T))
    if float(values[0]) <= 1.0e-10:
        raise RuntimeError("finite-cutoff logical codewords are linearly dependent")
    inverse_sqrt = (vectors * (1.0 / np.sqrt(values))) @ vectors.conj().T
    isometry = raw @ inverse_sqrt
    residual = np.linalg.norm(isometry.conj().T @ isometry - np.eye(2), ord="fro")
    if float(residual) > 2.0e-10:
        raise RuntimeError("logical isometry orthonormalization failed")
    return np.asarray(isometry, dtype=np.complex128)


def pure_loss_kraus(
    cutoff: int,
    *,
    duration_us: float,
    cavity_lifetime_us: float,
) -> tuple[ComplexMatrix, ...]:
    """Exact finite-Fock pure-loss Kraus family for the registered idle channel."""

    dimension = _positive_integer(cutoff, "cutoff", minimum=1)
    duration = _finite_positive(duration_us, "duration_us")
    lifetime = _finite_positive(cavity_lifetime_us, "cavity_lifetime_us")
    eta = exp(-duration / lifetime)
    operators: list[ComplexMatrix] = []
    for lost in range(dimension):
        operator = np.zeros((dimension, dimension), dtype=np.complex128)
        for incoming in range(lost, dimension):
            outgoing = incoming - lost
            operator[outgoing, incoming] = (
                sqrt(float(comb(incoming, lost)))
                * (1.0 - eta) ** (0.5 * lost)
                * eta ** (0.5 * outgoing)
            )
        operators.append(operator)
    effect = sum((operator.conj().T @ operator for operator in operators), start=np.zeros((dimension, dimension), dtype=np.complex128))
    if float(np.linalg.norm(effect - np.eye(dimension), ord="fro")) > 2.0e-10:
        raise RuntimeError("finite-cutoff pure-loss Kraus family is not trace preserving")
    return tuple(operators)


def encoded_channel_kraus(
    isometry: ArrayLike,
    noise_kraus: Iterable[ArrayLike],
) -> tuple[ComplexMatrix, ...]:
    """Compose a physical noise channel with an isometric logical encoding."""

    encoding = _matrix(isometry, "isometry")
    physical_dimension, logical_dimension = encoding.shape
    if logical_dimension < 2:
        raise ValueError("logical dimension must be at least two")
    if np.linalg.norm(encoding.conj().T @ encoding - np.eye(logical_dimension), ord="fro") > 2.0e-8:
        raise ValueError("isometry columns must be orthonormal")
    noise = tuple(_matrix(value, f"noise_kraus[{index}]") for index, value in enumerate(noise_kraus))
    if not noise:
        raise ValueError("at least one noise Kraus operator is required")
    if any(value.shape != (physical_dimension, physical_dimension) for value in noise):
        raise ValueError("noise Kraus dimensions do not match the encoding output")
    noise_effect = sum((value.conj().T @ value for value in noise), start=np.zeros((physical_dimension, physical_dimension), dtype=np.complex128))
    if np.linalg.norm(noise_effect - np.eye(physical_dimension), ord="fro") > 2.0e-8:
        raise ValueError("noise Kraus family must be trace preserving")
    encoded = tuple(value @ encoding for value in noise)
    encoded_effect = sum((value.conj().T @ value for value in encoded), start=np.zeros((logical_dimension, logical_dimension), dtype=np.complex128))
    if np.linalg.norm(encoded_effect - np.eye(logical_dimension), ord="fro") > 2.0e-8:
        raise RuntimeError("encoded channel is not trace preserving")
    return encoded


def qec_matrix(encoded_kraus: Sequence[ArrayLike]) -> ComplexMatrix:
    """Build ``M[mu,l;nu,k]`` with logical-major, Kraus-minor ordering."""

    operators = tuple(_matrix(value, f"encoded_kraus[{index}]") for index, value in enumerate(encoded_kraus))
    if not operators:
        raise ValueError("at least one encoded Kraus operator is required")
    shape = operators[0].shape
    if any(value.shape != shape for value in operators):
        raise ValueError("encoded Kraus operators must have equal shapes")
    _, logical_dimension = shape
    effect = sum((value.conj().T @ value for value in operators), start=np.zeros((logical_dimension, logical_dimension), dtype=np.complex128))
    if np.linalg.norm(effect - np.eye(logical_dimension), ord="fro") > 2.0e-8:
        raise ValueError("encoded Kraus family must be trace preserving")
    count = len(operators)
    matrix = np.empty((logical_dimension * count, logical_dimension * count), dtype=np.complex128)
    for logical_left in range(logical_dimension):
        for left_index, left in enumerate(operators):
            for logical_right in range(logical_dimension):
                for right_index, right in enumerate(operators):
                    matrix[
                        logical_left * count + left_index,
                        logical_right * count + right_index,
                    ] = np.vdot(left[:, logical_left], right[:, logical_right])
    return 0.5 * (matrix + matrix.conj().T)


def near_optimal_fidelity_from_qec(
    matrix: ArrayLike,
    *,
    logical_dimension: int,
    kraus_count: int,
) -> dict[str, Any]:
    """Evaluate the exact QEC-matrix/Petz fidelity and theorem interval."""

    logical = _positive_integer(logical_dimension, "logical_dimension", minimum=2)
    count = _positive_integer(kraus_count, "kraus_count")
    qec = _matrix(matrix, "qec_matrix")
    expected = logical * count
    if qec.shape != (expected, expected):
        raise ValueError("qec_matrix shape does not match logical_dimension*kraus_count")
    values, vectors, minimum, hermiticity = _psd_eigh(qec, "qec_matrix")
    root = (vectors * np.sqrt(values)) @ vectors.conj().T
    partial = np.zeros((count, count), dtype=np.complex128)
    for logical_index in range(logical):
        start = logical_index * count
        partial += root[start : start + count, start : start + count]
    fidelity = float(np.linalg.norm(partial, ord="fro") ** 2 / logical**2)
    if not -2.0e-9 <= fidelity <= 1.0 + 2.0e-9:
        raise RuntimeError("near-optimal fidelity lies outside [0,1]")
    rank_tolerance = max(1.0e-13, float(values[-1]) * 1.0e-12)
    return {
        "petz_fidelity": min(max(fidelity, 0.0), 1.0),
        "petz_infidelity": 1.0 - min(max(fidelity, 0.0), 1.0),
        "theorem_optimal_lower": min(max(fidelity, 0.0), 1.0),
        "theorem_optimal_upper": min(max(0.5 * (1.0 + fidelity), 0.0), 1.0),
        "qec_matrix_minimum_eigenvalue": minimum,
        "qec_matrix_rank": int(np.count_nonzero(values > rank_tolerance)),
        "qec_matrix_dimension": expected,
        "qec_matrix_trace_residual": abs(float(np.trace(qec).real) - logical),
        "qec_matrix_hermiticity_residual": hermiticity,
        "partial_trace_root_frobenius_norm": float(np.linalg.norm(partial, ord="fro")),
    }


def _support_inverse_sqrt(value: ArrayLike) -> tuple[ComplexMatrix, ComplexMatrix, int, float]:
    values, vectors, minimum, _ = _psd_eigh(value, "support_operator")
    threshold = max(1.0e-13, float(values[-1]) * 1.0e-12)
    active = values > threshold
    inverse = (vectors * np.where(active, 1.0 / np.sqrt(np.where(active, values, 1.0)), 0.0)) @ vectors.conj().T
    projector = (vectors * active.astype(np.float64)) @ vectors.conj().T
    return inverse, projector, int(np.count_nonzero(active)), minimum


def petz_recovery_diagnostics(encoded_kraus: Sequence[ArrayLike]) -> dict[str, Any]:
    """Construct the transpose/Petz recovery on the noise-output support."""

    operators = tuple(_matrix(value, f"encoded_kraus[{index}]") for index, value in enumerate(encoded_kraus))
    if not operators:
        raise ValueError("at least one encoded Kraus operator is required")
    physical, logical = operators[0].shape
    if any(value.shape != (physical, logical) for value in operators):
        raise ValueError("encoded Kraus operators must have equal shapes")
    sigma = sum((value @ value.conj().T for value in operators), start=np.zeros((physical, physical), dtype=np.complex128))
    inverse, support, support_rank, sigma_minimum = _support_inverse_sqrt(sigma)
    recovery = tuple(value.conj().T @ inverse for value in operators)
    recovery_effect = sum((value.conj().T @ value for value in recovery), start=np.zeros((physical, physical), dtype=np.complex128))
    fidelity = sum(
        abs(np.trace(recovery_operator @ encoded_operator)) ** 2
        for recovery_operator in recovery
        for encoded_operator in operators
    ) / logical**2
    support_action_residual = max(
        float(np.linalg.norm((np.eye(physical) - support) @ value, ord="fro"))
        for value in operators
    )
    return {
        "direct_petz_fidelity": float(fidelity.real),
        "noise_output_support_rank": support_rank,
        "noise_output_dimension": physical,
        "noise_output_minimum_eigenvalue": sigma_minimum,
        "petz_support_tp_residual": float(np.linalg.norm(recovery_effect - support, ord="fro")),
        "encoded_output_outside_support_residual": support_action_residual,
        "support_completion_required": support_rank < physical,
        "support_completion_fidelity_contribution": 0.0,
    }


def recovery_objective_matrix(encoded_kraus: Sequence[ArrayLike]) -> ComplexMatrix:
    """Return ``C`` such that ``F_e(R o A) = Tr(C J_R)``."""

    operators = tuple(_matrix(value, f"encoded_kraus[{index}]") for index, value in enumerate(encoded_kraus))
    if not operators:
        raise ValueError("at least one encoded Kraus operator is required")
    physical, logical = operators[0].shape
    if any(value.shape != (physical, logical) for value in operators):
        raise ValueError("encoded Kraus operators must have equal shapes")
    objective = np.zeros((logical * physical, logical * physical), dtype=np.complex128)
    for operator in operators:
        coefficient = operator.T.reshape(-1, order="F")
        objective += np.outer(coefficient.conj(), coefficient) / logical**2
    return 0.5 * (objective + objective.conj().T)


def partial_trace_recovery_output(
    choi: ArrayLike,
    *,
    logical_dimension: int,
    physical_dimension: int,
) -> ComplexMatrix:
    """Trace the logical output of an output-fast recovery Choi matrix."""

    logical = _positive_integer(logical_dimension, "logical_dimension", minimum=2)
    physical = _positive_integer(physical_dimension, "physical_dimension")
    matrix = _matrix(choi, "choi")
    if matrix.shape != (logical * physical, logical * physical):
        raise ValueError("choi shape does not match dimensions")
    return sum(
        (matrix[index::logical, index::logical] for index in range(logical)),
        start=np.zeros((physical, physical), dtype=np.complex128),
    )


def _repair_primal_choi(
    raw_choi: ArrayLike,
    objective: ComplexMatrix,
    *,
    logical_dimension: int,
    physical_dimension: int,
) -> tuple[ComplexMatrix, dict[str, float]]:
    raw = _matrix(raw_choi, "raw_primal_choi")
    hermitian = 0.5 * (raw + raw.conj().T)
    raw_eigenvalues, raw_vectors = np.linalg.eigh(hermitian)
    positive = (raw_vectors * np.maximum(raw_eigenvalues, 0.0)) @ raw_vectors.conj().T
    effect = partial_trace_recovery_output(
        positive,
        logical_dimension=logical_dimension,
        physical_dimension=physical_dimension,
    )
    effect = 0.5 * (effect + effect.conj().T)
    values, vectors = np.linalg.eigh(effect)
    if float(values[0]) <= 1.0e-10:
        raise RuntimeError("projected SDP primal has singular trace-preserving effect")
    inverse_sqrt = (vectors * (1.0 / np.sqrt(values))) @ vectors.conj().T
    transform = np.kron(inverse_sqrt, np.eye(logical_dimension))
    repaired = transform @ positive @ transform.conj().T
    repaired = 0.5 * (repaired + repaired.conj().T)
    repaired_effect = partial_trace_recovery_output(
        repaired,
        logical_dimension=logical_dimension,
        physical_dimension=physical_dimension,
    )
    return repaired, {
        "raw_primal_minimum_eigenvalue": float(raw_eigenvalues[0]),
        "raw_primal_tp_residual": float(
            np.linalg.norm(
                partial_trace_recovery_output(
                    hermitian,
                    logical_dimension=logical_dimension,
                    physical_dimension=physical_dimension,
                )
                - np.eye(physical_dimension),
                ord="fro",
            )
        ),
        "repaired_primal_minimum_eigenvalue": float(np.linalg.eigvalsh(repaired)[0]),
        "repaired_primal_tp_residual": float(
            np.linalg.norm(repaired_effect - np.eye(physical_dimension), ord="fro")
        ),
        "repaired_primal_fidelity_lower": float(np.trace(objective @ repaired).real),
    }


def solve_optimal_recovery_sdp(
    encoded_kraus: Sequence[ArrayLike],
    *,
    solver: str = "CLARABEL",
    tolerance: float = 1.0e-9,
    certificate_margin: float = 1.0e-10,
) -> dict[str, Any]:
    """Solve primal and dual recovery SDPs and repair both certificates.

    The repaired primal Choi matrix is PSD and normalized to be exactly CPTP up
    to reported floating residual.  The dual variable is shifted by a scalar
    identity until ``kron(Y, I_L) - C`` has the requested positive margin.
    """

    try:
        import cvxpy as cp
    except ModuleNotFoundError as exc:  # pragma: no cover - environment contract
        raise RuntimeError("CVXPY is required for the small-cutoff recovery SDP") from exc

    operators = tuple(_matrix(value, f"encoded_kraus[{index}]") for index, value in enumerate(encoded_kraus))
    if not operators:
        raise ValueError("at least one encoded Kraus operator is required")
    physical, logical = operators[0].shape
    if any(value.shape != (physical, logical) for value in operators):
        raise ValueError("encoded Kraus operators must have equal shapes")
    tol = _finite_positive(tolerance, "tolerance")
    margin = _finite_positive(certificate_margin, "certificate_margin")
    solver_name = str(solver).upper()
    if solver_name != "CLARABEL":
        raise ValueError("formal small-cutoff certificate currently requires CLARABEL")
    if solver_name not in cp.installed_solvers():
        raise RuntimeError(f"required CVXPY solver {solver_name} is unavailable")
    objective = recovery_objective_matrix(operators)
    size = logical * physical

    primal_choi = cp.Variable((size, size), hermitian=True)
    tp_expression = sum(primal_choi[index::logical, index::logical] for index in range(logical))
    primal_problem = cp.Problem(
        cp.Maximize(cp.real(cp.trace(objective @ primal_choi))),
        [primal_choi >> 0, tp_expression == np.eye(physical)],
    )
    solver_options = {
        "tol_gap_abs": tol,
        "tol_gap_rel": tol,
        "tol_feas": tol,
        "max_iter": 500,
    }
    primal_value = primal_problem.solve(solver=solver_name, **solver_options)
    if primal_problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or primal_choi.value is None:
        raise RuntimeError(f"recovery primal SDP failed: {primal_problem.status}")
    _, primal_certificate = _repair_primal_choi(
        primal_choi.value,
        objective,
        logical_dimension=logical,
        physical_dimension=physical,
    )

    dual_variable = cp.Variable((physical, physical), hermitian=True)
    dual_slack = cp.kron(dual_variable, np.eye(logical)) - objective
    dual_problem = cp.Problem(
        cp.Minimize(cp.real(cp.trace(dual_variable))),
        [dual_slack >> 0],
    )
    dual_value = dual_problem.solve(solver=solver_name, **solver_options)
    if dual_problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or dual_variable.value is None:
        raise RuntimeError(f"recovery dual SDP failed: {dual_problem.status}")
    raw_dual = 0.5 * (dual_variable.value + dual_variable.value.conj().T)
    raw_slack = np.kron(raw_dual, np.eye(logical)) - objective
    raw_dual_minimum = float(np.linalg.eigvalsh(0.5 * (raw_slack + raw_slack.conj().T))[0])
    shift = max(0.0, margin - raw_dual_minimum)
    repaired_dual = raw_dual + shift * np.eye(physical)
    repaired_slack = np.kron(repaired_dual, np.eye(logical)) - objective
    repaired_dual_minimum = float(np.linalg.eigvalsh(0.5 * (repaired_slack + repaired_slack.conj().T))[0])
    dual_upper = float(np.trace(repaired_dual).real)
    primal_lower = primal_certificate["repaired_primal_fidelity_lower"]
    if primal_lower > dual_upper + 2.0e-7:
        raise RuntimeError("repaired SDP primal exceeds repaired dual certificate")
    return {
        "solver": solver_name,
        "primal_status": primal_problem.status,
        "dual_status": dual_problem.status,
        "primal_raw_objective": float(primal_value),
        "dual_raw_objective": float(dual_value),
        **primal_certificate,
        "raw_dual_minimum_slack_eigenvalue": raw_dual_minimum,
        "dual_identity_shift": shift,
        "repaired_dual_minimum_slack_eigenvalue": repaired_dual_minimum,
        "repaired_dual_fidelity_upper": dual_upper,
        "repaired_certificate_width": dual_upper - primal_lower,
        "raw_solver_duality_gap": float(dual_value - primal_value),
        "primal_solve_time_seconds": float(primal_problem.solver_stats.solve_time or 0.0),
        "dual_solve_time_seconds": float(dual_problem.solver_stats.solve_time or 0.0),
        "primal_iterations": int(primal_problem.solver_stats.num_iters or 0),
        "dual_iterations": int(dual_problem.solver_stats.num_iters or 0),
        "solver_tolerance": tol,
        "certificate_margin": margin,
    }


@dataclass(frozen=True)
class ChannelRecoveryBound:
    physical_dimension: int
    logical_dimension: int
    kraus_count: int
    mean_code_photon_number: float
    encoding_orthonormality_residual: float
    encoded_channel_tp_residual: float
    petz: dict[str, Any]
    petz_recovery: dict[str, Any]
    sdp: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "physical_dimension": self.physical_dimension,
            "logical_dimension": self.logical_dimension,
            "kraus_count": self.kraus_count,
            "mean_code_photon_number": self.mean_code_photon_number,
            "encoding_orthonormality_residual": self.encoding_orthonormality_residual,
            "encoded_channel_tp_residual": self.encoded_channel_tp_residual,
            "petz": dict(self.petz),
            "petz_recovery": dict(self.petz_recovery),
            "sdp": None if self.sdp is None else dict(self.sdp),
        }


def evaluate_encoded_channel_recovery(
    isometry: ArrayLike,
    noise_kraus: Sequence[ArrayLike],
    *,
    solve_sdp: bool = False,
    solver: str = "CLARABEL",
    solver_tolerance: float = 1.0e-9,
) -> ChannelRecoveryBound:
    """Evaluate Petz/theorem bounds and optionally the small-cutoff SDP."""

    encoding = _matrix(isometry, "isometry")
    encoded = encoded_channel_kraus(encoding, noise_kraus)
    physical, logical = encoding.shape
    number = np.diag(np.arange(physical, dtype=np.float64))
    mean_photon = float(np.trace(encoding.conj().T @ number @ encoding).real / logical)
    effect = sum((value.conj().T @ value for value in encoded), start=np.zeros((logical, logical), dtype=np.complex128))
    matrix = qec_matrix(encoded)
    near = near_optimal_fidelity_from_qec(
        matrix,
        logical_dimension=logical,
        kraus_count=len(encoded),
    )
    direct = petz_recovery_diagnostics(encoded)
    near["qec_vs_direct_petz_residual"] = abs(
        near["petz_fidelity"] - direct["direct_petz_fidelity"]
    )
    if near["qec_vs_direct_petz_residual"] > 2.0e-7:
        raise RuntimeError("QEC-matrix and direct Petz fidelities disagree")
    sdp = None
    if solve_sdp:
        sdp = solve_optimal_recovery_sdp(
            encoded,
            solver=solver,
            tolerance=solver_tolerance,
        )
        sdp["intersection_certified_lower"] = max(
            near["theorem_optimal_lower"],
            sdp["repaired_primal_fidelity_lower"],
        )
        sdp["intersection_certified_upper"] = min(
            near["theorem_optimal_upper"],
            sdp["repaired_dual_fidelity_upper"],
        )
        sdp["intersection_width"] = (
            sdp["intersection_certified_upper"]
            - sdp["intersection_certified_lower"]
        )
        if sdp["intersection_width"] < -2.0e-7:
            raise RuntimeError("Petz theorem and SDP certificate intervals do not intersect")
    return ChannelRecoveryBound(
        physical_dimension=physical,
        logical_dimension=logical,
        kraus_count=len(encoded),
        mean_code_photon_number=mean_photon,
        encoding_orthonormality_residual=float(
            np.linalg.norm(encoding.conj().T @ encoding - np.eye(logical), ord="fro")
        ),
        encoded_channel_tp_residual=float(np.linalg.norm(effect - np.eye(logical), ord="fro")),
        petz=near,
        petz_recovery=direct,
        sdp=sdp,
    )


__all__ = [
    "ChannelRecoveryBound",
    "encoded_channel_kraus",
    "evaluate_encoded_channel_recovery",
    "finite_cutoff_gkp_isometry",
    "near_optimal_fidelity_from_qec",
    "partial_trace_recovery_output",
    "petz_recovery_diagnostics",
    "pure_loss_kraus",
    "qec_matrix",
    "recovery_objective_matrix",
    "solve_optimal_recovery_sdp",
]
