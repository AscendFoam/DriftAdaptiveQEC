"""Public façade for independent Phase-9 backend B.

Its dense Strang/Kraus, RNG, IQ, reset/drift, and logical kernels stay private.
"""

from __future__ import annotations

from scipy.linalg import expm

from ._phase9_backend_b.models import (
    ANCILLA_LEVELS,
    BACKEND_B_ID,
    BACKEND_B_LIKELIHOOD_ID,
    BACKEND_B_LOGICAL_ID,
    BACKEND_B_RNG_ID,
    BACKEND_B_SCOPE,
    BACKEND_B_SOLVER_ID,
    MAX_EXACT_CHOI_CUTOFF,
    MAX_SUPPORTED_CUTOFF,
    BackendBConfig,
    BackendBDrift,
    BackendBEvaluator,
    BackendBLogical,
    BackendBObservation,
    BackendBQualification,
    BackendBQualificationThresholds,
    BackendBRandomRecord,
    BackendBRound,
    BackendBState,
    BackendBTrajectory,
    BackendBTruth,
    ComplexMatrix,
    backend_b_random_record,
)
from ._phase9_backend_b.qualification import (
    run_backend_b_qualification,
)
from ._phase9_backend_b.simulator import (
    Phase9BackendBSimulator as _Phase9BackendBSimulator,
    diagnostic_action_word_b,
)
from .phase9_twin_contract import ActionWord as _ActionWord


class Phase9BackendBSimulator(_Phase9BackendBSimulator):
    """Stable public simulator type backed by the independent B kernel."""

    @staticmethod
    def _matrix_exponential(matrix: ComplexMatrix) -> ComplexMatrix:
        return expm(matrix)

__all__ = [
    "ANCILLA_LEVELS", "BACKEND_B_ID", "BACKEND_B_SCOPE",
    "BACKEND_B_SOLVER_ID", "BACKEND_B_RNG_ID",
    "BACKEND_B_LIKELIHOOD_ID", "BACKEND_B_LOGICAL_ID",
    "MAX_SUPPORTED_CUTOFF", "MAX_EXACT_CHOI_CUTOFF",
    "BackendBConfig", "BackendBQualificationThresholds", "BackendBDrift",
    "BackendBState", "BackendBRandomRecord", "BackendBObservation",
    "BackendBTruth", "BackendBEvaluator", "BackendBLogical",
    "BackendBRound", "BackendBTrajectory", "BackendBQualification",
    "Phase9BackendBSimulator", "backend_b_random_record",
    "diagnostic_action_word_b", "run_backend_b_qualification",
]
