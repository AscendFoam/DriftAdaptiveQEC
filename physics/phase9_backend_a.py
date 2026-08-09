"""Stable façade for the independent Phase-9 backend-A implementation.
The private implementation retains Fock×qutrit/GKSL/IQ/reset/drift dynamics,
logical evaluation, qualification gates, and the audited runtime-cap surface.
"""

from ._phase9_backend_a.qualification import run_backend_a_qualification
from ._phase9_backend_a.schema import *  # noqa: F403
from ._phase9_backend_a.simulator import Phase9BackendASimulator

__all__ = [
    "ANCILLA_LEVELS", "BACKEND_A_ID", "BACKEND_A_SCOPE", "BackendAConfig", "BackendADriftState",
    "BackendAEvaluatorState", "BackendAExogenous", "BackendALogicalRecord", "BackendAObservation",
    "BackendAQualification", "BackendAQualificationThresholds", "BackendARoundResult", "BackendAState",
    "BackendATrajectory", "BackendATruthRecord", "ChannelDiagnostics", "DEFAULT_PARAMETER_PROVENANCE",
    "MAX_SUPPORTED_CUTOFF", "MAX_EXACT_CHOI_CUTOFF", "Phase9BackendASimulator", "backend_a_exogenous",
    "diagnostic_action_word", "run_backend_a_qualification",
]
