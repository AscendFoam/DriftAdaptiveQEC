"""T5.1.1 complete, lane-aware comparison-set registry.

The registry is intentionally not a single leaderboard.  It freezes every
required comparator together with its decision target, information budget,
protocol/time basis, executable binding, evidence artifact, deployability and
eligible comparison lanes.  This prevents decoder, protocol, controller,
estimator and short-horizon oracle results from being silently mixed.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from physics.autonomous_sbs import (
    IdleMemoryConfig,
    IdleMemorySimulator,
    NonselectiveSBSConfig,
    NonselectiveSBSSimulator,
)
from physics.finite_energy_trends import run_finite_energy_shrinkage_trend


TASK_ID = "T5.1.1"
PROTOCOL_ID = "T511-LANE-AWARE-COMPLETE-COMPARISON-SET-V1"
DEFAULT_ARTIFACT = Path("docs/t5_1_1_comparison_set_registry.json")
DEFAULT_SOURCE_DATA = Path("docs/t5_1_1_comparison_set_registry_source_data.csv")
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ComparatorSpec:
    comparator_id: str
    label: str
    family: str
    task_owner: str
    decision_target: str
    information_set: tuple[str, ...]
    hidden_truth_access: bool
    output_contract: str
    protocol_scope: str
    metric_scope: str
    time_basis: str
    compute_budget: str
    deployability: str
    ranking_status: str
    eligible_lanes: tuple[str, ...]
    implementation_path: str
    expected_fragment: str
    artifact_paths: tuple[str, ...]
    claim_boundary: str


REQUIRED_COMPARATOR_IDS = (
    "no_correction_idle_memory",
    "standard_binning",
    "standard_measurement_feedback_sbs",
    "autonomous_sbs",
    "static_periodic_map",
    "topk_periodic_map",
    "decoder_oracle_map",
    "finite_energy_static_shrinkage",
    "memory_bayesian",
    "ewma_adaptive_map",
    "kalman_adaptive_map",
    "sliding_window_map",
    "run_length_event_controller",
    "regime_hmm_estimator",
    "latest_outcome_mf_fnn",
    "exponential_recurrence_controller",
    "bounded_residual_rnn_teacher",
    "distilled_low_dimensional_student",
    "finite_horizon_control_oracle",
)


COMPARISON_LANES: dict[str, dict[str, Any]] = {
    "decoder_current_syndrome": {
        "members": (
            "standard_binning",
            "static_periodic_map",
            "topk_periodic_map",
            "decoder_oracle_map",
        ),
        "target": "current periodic syndrome to logical coset",
        "metric_contract": "paired logical decision error on identical syndrome traces",
        "ranking_rule": "top-K is an implementation sensitivity; decoder oracle is a nondeployable reference",
    },
    "decoder_continuous_drift": {
        "members": (
            "standard_binning",
            "static_periodic_map",
            "ewma_adaptive_map",
            "kalman_adaptive_map",
            "sliding_window_map",
            "decoder_oracle_map",
        ),
        "target": "causal drifting periodic-syndrome decoding",
        "metric_contract": "paired LER/proper score with training-only calibration and one-window delay",
        "ranking_rule": "T5.1.2 must rebuild a shared scenario adapter; old task-local numbers are not pooled",
    },
    "decoder_episode_memory": {
        "members": (
            "standard_binning",
            "static_periodic_map",
            "memory_bayesian",
            "decoder_oracle_map",
        ),
        "target": "bounded modular-syndrome episode decision",
        "metric_contract": "paired episode logical error under the registered history budget",
        "ranking_rule": "episode and continuous-drift results remain separate",
    },
    "finite_energy_effective": {
        "members": ("standard_binning", "finite_energy_static_shrinkage"),
        "target": "finite-energy syndrome-level correction",
        "metric_contract": "paired residual MSE and cell-parity error on held-out samples",
        "ranking_rule": "effective-model result is not full finite-Fock recovery or channel optimum",
    },
    "protocol_wallclock": {
        "members": (
            "no_correction_idle_memory",
            "standard_measurement_feedback_sbs",
            "autonomous_sbs",
        ),
        "target": "finite-cutoff memory preservation at common physical time",
        "metric_contract": "fidelity/code survival/logical-Z lifetime plus protocol-native event counts",
        "ranking_rule": "per-cycle and wall-clock results must both be shown; no desired ordering is required",
    },
    "control_matched_model": {
        "members": (
            "standard_measurement_feedback_sbs",
            "latest_outcome_mf_fnn",
            "exponential_recurrence_controller",
            "bounded_residual_rnn_teacher",
            "distilled_low_dimensional_student",
        ),
        "target": "causal nominal-plus-bounded-residual sBs control",
        "metric_contract": "same-model paired lifetime/fidelity/burden/cost with frozen seeds and horizon",
        "ranking_rule": "exact budget and cutoff must be matched; universal NMF-over-MF is prohibited",
    },
    "event_and_regime_components": {
        "members": ("run_length_event_controller", "regime_hmm_estimator"),
        "target": "event/regime component diagnostics",
        "metric_contract": "component detection/calibration/cost metrics, not logical-policy ranking",
        "ranking_rule": "component rows cannot be inserted into the main LER leaderboard without an explicit adapter",
    },
    "control_oracle_short_horizon": {
        "members": (
            "standard_measurement_feedback_sbs",
            "finite_horizon_control_oracle",
        ),
        "target": "fixed-ansatz exact causal control tree",
        "metric_contract": "exact short-horizon terminal objective on identical branches",
        "ranking_rule": "the two-cycle oracle cannot be extrapolated into the ten-cycle controller table",
    },
}


def _specs() -> tuple[ComparatorSpec, ...]:
    return (
        ComparatorSpec(
            "no_correction_idle_memory", "No correction (idle memory)", "protocol_anchor", "T5.1.1",
            "memory preservation without QEC action", ("elapsed physical time",), False,
            "identity control with cavity idle-noise propagation", "finite-cutoff cavity memory; no sBs gates or ancilla events",
            "fidelity, code survival and logical-Z curve", "10 us reporting intervals", "zero active operations",
            "deployable_sanity_anchor", "anchor_only", ("protocol_wallclock",),
            "physics/autonomous_sbs.py", "class IdleMemorySimulator", (),
            "model-level idle-memory anchor only; not an experimental T1/T2 measurement",
        ),
        ComparatorSpec(
            "standard_binning", "Standard binning", "decoder", "T3.1.1",
            "current syndrome to logical coset", ("current centered q/p syndrome",), False,
            "fixed central even-even logical-coset decision", "ideal square-GKP syndrome-level decoder",
            "paired logical decision error", "per observation", "no fitted parameters",
            "deployable_software", "main_anchor", ("decoder_current_syndrome", "decoder_continuous_drift", "decoder_episode_memory", "finite_energy_effective"),
            "cnn_fpga/benchmark/standard_binning_baseline.py", "STANDARD_BINNING_DESCRIPTOR =",
            ("docs/t3_1_1_standard_binning_validation.json",),
            "not legacy static_linear and not finite-energy or hardware performance",
        ),
        ComparatorSpec(
            "standard_measurement_feedback_sbs", "Standard measurement-feedback sBs", "control_protocol", "T3.2.8/T4.4.4",
            "nominal repeated sBs memory control", ("current protocol phase", "nonselective measurement expectation"), False,
            "fixed nominal 15-parameter sBs controls", "finite-cutoff nominal-control measurement-feedback sBs",
            "fidelity, logical-Z lifetime, p(g), event burden", "10 us literature full cycle", "15 fixed controls; no online learning",
            "software_model_anchor", "main_anchor", ("protocol_wallclock", "control_matched_model", "control_oracle_short_horizon"),
            "physics/autonomous_sbs.py", "class NonselectiveSBSSimulator",
            ("docs/t3_2_8_autonomous_sbs_wallclock_validation.json", "docs/t4_4_4_teacher_student_gain_retention.json"),
            "literature timing is not target-board timing and nominal control is not an optimum",
        ),
        ComparatorSpec(
            "autonomous_sbs", "Autonomous sBs", "control_protocol", "T3.2.8",
            "nominal autonomous memory control", ("current protocol phase",), False,
            "fixed nominal nonselective autonomous sBs channel", "finite-cutoff autonomous protocol with reset and no measurement",
            "fidelity/logical lifetime and event burden", "7 us literature full cycle", "fixed controls; protocol-native gate/reset count",
            "software_model_baseline", "separate_protocol_lane", ("protocol_wallclock",),
            "physics/autonomous_sbs.py", "AUTONOMOUS_TIMING =",
            ("docs/t3_2_8_autonomous_sbs_wallclock_validation.json",),
            "not a trained autonomous optimum or target-device timing result",
        ),
        ComparatorSpec(
            "static_periodic_map", "Static training-average periodic MAP", "decoder", "T3.1.2",
            "current syndrome to logical coset", ("current syndrome", "training-only frozen noise parameters"), False,
            "periodic-Gaussian MAP logical coset", "single-mode ideal syndrome decoder",
            "paired LER and proper scores", "per observation", "frozen static parameter table",
            "deployable_software", "main_candidate", ("decoder_current_syndrome", "decoder_continuous_drift", "decoder_episode_memory"),
            "cnn_fpga/benchmark/static_map_baseline.py", "def build_static_map_validation",
            ("docs/t3_1_2_static_map_validation.json",),
            "model/scenario-specific static MAP, not oracle MAP",
        ),
        ComparatorSpec(
            "topk_periodic_map", "Top-K periodic MAP", "decoder_approximation", "T3.1.5",
            "current syndrome to logical coset", ("current syndrome", "same frozen static parameters"), False,
            "per-coset top-K alias likelihood", "same single-mode periodic-Gaussian candidate rectangle as full MAP",
            "agreement/LER/cost versus full MAP", "per observation", "explicit K-dependent comparisons and accumulations",
            "deployable_candidate", "implementation_sensitivity", ("decoder_current_syndrome",),
            "cnn_fpga/benchmark/topk_lattice_coset_map.py", "def build_topk_validation",
            ("docs/t3_1_5_topk_map_validation.json",),
            "not surface K-MWM, universal K or an independent oracle",
        ),
        ComparatorSpec(
            "decoder_oracle_map", "Hidden-state decoder oracle MAP", "decoder_bound", "T3.1.3",
            "current syndrome to logical coset", ("current syndrome", "true DriftState mean/covariance/mixture/regime"), True,
            "Bayes/MAP logical coset and posterior", "frozen assumed periodic-mixture decoder model",
            "paired LER and decoder-oracle gap", "time-aligned per observation", "unrestricted hidden-state model evaluation",
            "nondeployable", "nondeployable_reference", ("decoder_current_syndrome", "decoder_continuous_drift", "decoder_episode_memory"),
            "cnn_fpga/benchmark/oracle_baseline.py", "def build_oracle_validation",
            ("docs/t3_1_3_oracle_validation.json",),
            "not control oracle, channel optimum or actual deployable decoder",
        ),
        ComparatorSpec(
            "finite_energy_static_shrinkage", "Finite-energy static shrinkage", "decoder", "T1.2.3",
            "finite-energy noisy syndrome to bounded analog correction", ("current syndrome", "training-fitted frozen gain"), False,
            "c=g*s with one static gain per frozen finite-energy setting", "syndrome-level effective finite-energy-noise model",
            "held-out residual MSE and paired parity error", "per observation", "one frozen scalar gain",
            "deployable_effective_model", "separate_fidelity_lane", ("finite_energy_effective",),
            "physics/finite_energy_trends.py", "def run_finite_energy_shrinkage_trend", (),
            "not full approximate-GKP recovery, Fock channel optimum or device-calibrated decoder",
        ),
        ComparatorSpec(
            "memory_bayesian", "Memory-assisted Bayesian decoder", "decoder", "T3.2.1",
            "bounded syndrome episode to logical decision", ("causal episode syndrome history", "registered prior"), False,
            "periodic Bayesian posterior and logical decision", "bounded no-intermediate-correction syndrome episodes",
            "paired episode logical error", "per registered episode", "explicit history-state and grid budget",
            "software_candidate", "main_candidate_same_episode_only", ("decoder_episode_memory",),
            "cnn_fpga/benchmark/memory_assisted_bayesian_decoder.py", "def build_memory_bayesian_validation",
            ("docs/t3_2_1_memory_bayesian_validation.json",),
            "episode result cannot be pooled with continuous per-round decoding",
        ),
        ComparatorSpec(
            "ewma_adaptive_map", "EWMA adaptive periodic MAP", "decoder_estimator", "T3.2.2",
            "drifting syndrome to logical coset", ("causal periodic moments", "training-frozen alpha"), False,
            "EWMA state estimate followed by periodic MAP", "continuous wrapped-Gaussian drift",
            "paired LER, NLL, Brier and tracking RMSE", "one-window delayed", "four complex EWMA state values",
            "deployable_software", "main_candidate", ("decoder_continuous_drift",),
            "cnn_fpga/benchmark/continuous_adaptive_map.py", "def build_continuous_adaptive_validation",
            ("docs/t3_2_2_continuous_adaptive_map_validation.json",),
            "training-frozen model; no hidden state or universal drift advantage claim",
        ),
        ComparatorSpec(
            "kalman_adaptive_map", "Kalman adaptive periodic MAP", "decoder_estimator", "T3.2.2",
            "drifting syndrome to logical coset", ("causal periodic moments", "training-frozen process/measurement scales"), False,
            "constant-velocity Kalman estimate followed by periodic MAP", "continuous wrapped-Gaussian drift",
            "paired LER, NLL, Brier and tracking RMSE", "one-window delayed", "10-state/100-covariance-value analytic profile",
            "host_deployable_software", "main_candidate", ("decoder_continuous_drift",),
            "cnn_fpga/benchmark/continuous_adaptive_map.py", "def build_continuous_adaptive_validation",
            ("docs/t3_2_2_continuous_adaptive_map_validation.json",),
            "constant-velocity assumed model, not a general nonlinear oracle",
        ),
        ComparatorSpec(
            "sliding_window_map", "Training-selected sliding-window MAP", "decoder_estimator", "T3.2.3",
            "drifting syndrome to logical coset", ("causal bounded syndrome window", "training-selected length"), False,
            "window moment estimate followed by periodic MAP", "continuous wrapped-Gaussian drift",
            "paired LER and proper scores", "one-window delayed", "bounded selected history window",
            "host_deployable_software", "main_candidate", ("decoder_continuous_drift",),
            "cnn_fpga/benchmark/sliding_window_syndrome_estimator.py", "def build_sliding_window_validation",
            ("docs/t3_2_3_sliding_window_validation.json",),
            "training-selected history only; no evaluation-length selection",
        ),
        ComparatorSpec(
            "run_length_event_controller", "Run-length event controller", "event_controller", "T3.2.5",
            "observed g/e/leakage events to bounded event action", ("current event", "saturated causal run counters"), False,
            "finite-state event/fallback action", "observed-event control component",
            "event sensitivity, delay, false trigger and cost", "per event", "finite counters and transition table",
            "deployable_component_candidate", "component_only", ("event_and_regime_components",),
            "cnn_fpga/benchmark/run_length_fsm_baseline.py", "def build_run_length_validation",
            ("docs/t3_2_5_run_length_fsm_validation.json",),
            "component evidence is not direct logical-lifetime superiority",
        ),
        ComparatorSpec(
            "regime_hmm_estimator", "Causal regime HMM", "host_estimator", "T3.2.6",
            "observed windows to regime posterior", ("causal window summaries", "frozen HMM parameters"), False,
            "regime posterior and state estimate", "host slow-loop regime component",
            "regime calibration/detection/cost", "per validated window", "bounded HMM state/history budget",
            "host_deployable_component", "component_only", ("event_and_regime_components",),
            "cnn_fpga/benchmark/regime_hmm_baseline.py", "def build_regime_hmm_validation",
            ("docs/t3_2_6_regime_hmm_validation.json",),
            "not a per-cycle decoder or direct lifetime row without an adapter",
        ),
        ComparatorSpec(
            "latest_outcome_mf_fnn", "Exact-budget latest-outcome MF/FNN", "control_policy", "T3.2.7",
            "latest observed g/e/leakage token to bounded sBs residual", ("latest outcome only",), False,
            "15-dimensional nominal-plus-bounded residual", "matched finite-cutoff sBs control model",
            "paired fidelity/logical lifetime and memory contrast", "10-cycle matched-model horizon", "72,853 parameters / 72,266 dense MACs",
            "software_control_baseline", "main_candidate_when_budget_matched", ("control_matched_model",),
            "cnn_fpga/benchmark/latest_outcome_markovian_baseline.py", "def run_benchmark",
            ("docs/t3_2_7_latest_outcome_markovian_validation.json",),
            "signed result and cutoff reversal must be retained; FNN is not a decoder oracle",
        ),
        ComparatorSpec(
            "exponential_recurrence_controller", "Handcrafted exponential recurrence", "control_policy", "T3.2.10",
            "causal event history to bounded sBs residual", ("current event", "small exponential state"), False,
            "15-dimensional bounded residual via explicit recurrence", "matched finite-cutoff sBs control model",
            "paired fidelity/logical lifetime, p(g) and resource profile", "registered finite horizon", "small analytic state/MAC budget",
            "deployable_control_candidate", "main_candidate_when_same_trace", ("control_matched_model",),
            "cnn_fpga/benchmark/exponential_recurrence_baseline.py", "def run_exponential_recurrence_baseline",
            ("docs/t3_2_10_exponential_recurrence_validation.json",),
            "handcrafted recurrence is not the distilled student and high p(g) burden must remain visible",
        ),
        ComparatorSpec(
            "bounded_residual_rnn_teacher", "Bounded-residual RNN teacher", "controller_teacher", "T4.4.1",
            "causal history to bounded sBs residual", ("causal g/e history", "recurrent hidden state"), False,
            "15-dimensional nominal-plus-tanh-bounded residual", "matched two-level finite-cutoff sBs model",
            "paired fidelity/logical lifetime and burden", "trained/evaluated finite horizon", "72,853-parameter GRU teacher",
            "offline_teacher", "offline_reference_not_deployable", ("control_matched_model",),
            "cnn_fpga/benchmark/bounded_residual_rnn_teacher.py", "def run_bounded_residual_teacher_training",
            ("docs/t4_4_1_bounded_residual_rnn_teacher_validation.json",),
            "not global optimum, mechanism proof, long-horizon/OOD or deployed controller",
        ),
        ComparatorSpec(
            "distilled_low_dimensional_student", "Distilled low-dimensional student", "controller_student", "T4.4.3/T4.4.4/T4.4.5",
            "causal event history to bounded sBs residual", ("current g/e/leakage event", "four-state recurrence"), False,
            "15-dimensional bounded residual with fail-closed leakage action", "matched two-level finite-cutoff sBs model",
            "teacher gain retention, fidelity/lifetime/burden/cost", "10-cycle matched-model horizon", "95 scalars / 87 analytic MACs",
            "deployable_candidate_before_quantized_hardware_gates", "qualified_main_candidate", ("control_matched_model",),
            "cnn_fpga/benchmark/low_dimensional_student_distillation.py", "def run_low_dimensional_student_distillation",
            ("docs/t4_4_4_teacher_student_gain_retention.json", "docs/t4_4_5_teacher_student_branch_freeze.json"),
            "qualified retention only; no universal NMF, OOD, leakage, RTL or board claim",
        ),
        ComparatorSpec(
            "finite_horizon_control_oracle", "Finite-horizon control oracle", "control_bound", "T3.2.9",
            "causal measurement-tree prefix to optimized sBs action", ("offline full model tree", "online realized prefix only"), False,
            "history-indexed optimized 15-parameter action", "fixed ansatz and exact short control tree",
            "exact terminal fidelity/value", "two-cycle horizon only", "exponential lookup / multistart offline optimization",
            "nondeployable", "nondeployable_short_horizon_reference", ("control_oracle_short_horizon",),
            "cnn_fpga/benchmark/trajectory_lookup_control_oracle.py", "def run_trajectory_lookup_control_oracle",
            ("docs/t3_2_9_trajectory_lookup_control_oracle.json",),
            "two-cycle reference only; not decoder oracle, global control optimum or ten-cycle bound",
        ),
    )


def comparison_specs() -> tuple[ComparatorSpec, ...]:
    """Return the immutable frozen comparator catalog."""

    return _specs()


SECONDARY_EXCLUSIONS = (
    {
        "protocol_id": "secondary_knill_qunaught",
        "status": "REFERENCE_ONLY_NOT_MAIN_RANKING",
        "reason": "different protocol/resources and no approved common sBs execution lane",
        "evidence": "docs/t5_0_1_literature_trend_reproduction.json",
    },
    {
        "protocol_id": "secondary_psteane",
        "status": "ANALYTIC_PASS_NOT_MAIN_RANKING",
        "reason": "small-noise analytic preprocessing result, not sBs controller or physical squeezing implementation",
        "evidence": "docs/t5_0_2_independent_cross_fidelity_holdout.json",
    },
)


def _repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_comparator_specs(specs: Sequence[ComparatorSpec]) -> tuple[str, ...]:
    rows = tuple(specs)
    ids = tuple(row.comparator_id for row in rows)
    if ids != REQUIRED_COMPARATOR_IDS:
        raise ValueError("comparison set must match the frozen required order exactly")
    if len(ids) != len(set(ids)):
        raise ValueError("comparison set contains duplicate comparator IDs")
    known_lanes = set(COMPARISON_LANES)
    for row in rows:
        for name in (
            "label", "family", "task_owner", "decision_target", "output_contract",
            "protocol_scope", "metric_scope", "time_basis", "compute_budget",
            "deployability", "ranking_status", "implementation_path",
            "expected_fragment", "claim_boundary",
        ):
            if not str(getattr(row, name)).strip():
                raise ValueError(f"{row.comparator_id} has an empty {name}")
        if not row.information_set or not row.eligible_lanes:
            raise ValueError(f"{row.comparator_id} must declare information and lane scope")
        if not set(row.eligible_lanes) <= known_lanes:
            raise ValueError(f"{row.comparator_id} references an unknown comparison lane")
        if row.hidden_truth_access and row.comparator_id != "decoder_oracle_map":
            raise ValueError("only decoder_oracle_map may consume hidden truth online")
        if row.hidden_truth_access and row.deployability != "nondeployable":
            raise ValueError("hidden-truth comparator must be nondeployable")
    for lane_id, contract in COMPARISON_LANES.items():
        members = tuple(contract["members"])
        if not members or len(members) != len(set(members)):
            raise ValueError(f"{lane_id} must contain unique members")
        if not set(members) <= set(ids):
            raise ValueError(f"{lane_id} contains an unknown comparator")
        for member in members:
            row = rows[ids.index(member)]
            if lane_id not in row.eligible_lanes:
                raise ValueError(f"{member} does not reciprocally register {lane_id}")
    return tuple(f"spec:{row.comparator_id}" for row in rows)


def _artifact_pass(payload: Mapping[str, Any]) -> bool:
    if payload.get("status") != "PASS":
        return False
    summary = payload.get("gate_summary")
    if isinstance(summary, Mapping):
        failed = summary.get("failed")
        if isinstance(failed, list) and failed:
            return False
        if isinstance(failed, (int, float)) and int(failed) != 0:
            return False
        gates = summary.get("gates")
        if isinstance(gates, Mapping) and not all(value is True for value in gates.values()):
            return False
    gates = payload.get("gates")
    if isinstance(gates, Mapping) and not all(value is True for value in gates.values()):
        return False
    return True


def _artifact_bindings(specs: Sequence[ComparatorSpec]) -> list[dict[str, Any]]:
    paths = sorted({path for row in specs for path in row.artifact_paths})
    bindings: list[dict[str, Any]] = []
    for relative in paths:
        path = _repo_path(relative)
        if not path.is_file():
            raise FileNotFoundError(f"missing comparator artifact: {relative}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        bindings.append(
            {
                "path": relative,
                "task_id": payload.get("task_id"),
                "status": payload.get("status"),
                "machine_pass": _artifact_pass(payload),
                "sha256": _sha256(path),
            }
        )
    return bindings


def _implementation_bindings(specs: Sequence[ComparatorSpec]) -> list[dict[str, Any]]:
    bindings: list[dict[str, Any]] = []
    for row in specs:
        path = _repo_path(row.implementation_path)
        if not path.is_file():
            raise FileNotFoundError(f"missing comparator implementation: {row.implementation_path}")
        text = path.read_text(encoding="utf-8")
        if row.expected_fragment not in text:
            raise ValueError(f"missing implementation anchor for {row.comparator_id}")
        bindings.append(
            {
                "comparator_id": row.comparator_id,
                "path": row.implementation_path,
                "expected_fragment": row.expected_fragment,
                "sha256": _sha256(path),
            }
        )
    return bindings


def _no_correction_probe() -> dict[str, Any]:
    idle = IdleMemorySimulator(IdleMemoryConfig(full_cycles=3, cutoff=6)).run()
    standard = NonselectiveSBSSimulator(
        NonselectiveSBSConfig(
            mode="measurement_feedback", full_cycles=3, cutoff=6, device="cpu"
        )
    ).run()
    one_step = IdleMemorySimulator(
        IdleMemoryConfig(full_cycles=1, cutoff=6, cycle_duration_us=10.0)
    ).run()
    two_steps = IdleMemorySimulator(
        IdleMemoryConfig(full_cycles=2, cutoff=6, cycle_duration_us=5.0)
    ).run()
    density_difference = float(
        np.max(
            np.abs(
                idle.final_cavity_density.detach().cpu().numpy()
                - standard.final_cavity_density.detach().cpu().numpy()
            )
        )
    )
    semigroup_error = float(
        np.max(
            np.abs(
                one_step.final_cavity_density.detach().cpu().numpy()
                - two_steps.final_cavity_density.detach().cpu().numpy()
            )
        )
    )
    zero_fields = (
        "measurement_events", "reset_events", "active_gate_applications",
        "frame_updates", "outcome_dependent_parameter_updates",
    )
    return {
        "time_us": idle.time_us.tolist(),
        "fidelity": idle.fidelity.tolist(),
        "logical_z_signal": idle.logical_z_signal.tolist(),
        "event_accounting": dict(idle.event_accounting),
        "maximum_trace_error": idle.maximum_trace_error,
        "maximum_hermiticity_error": idle.maximum_hermiticity_error,
        "minimum_final_eigenvalue": idle.minimum_final_eigenvalue,
        "standard_sbs_density_max_difference": density_difference,
        "ten_us_vs_two_five_us_semigroup_error": semigroup_error,
        "gates": {
            "all_control_and_ancilla_event_counts_are_zero": all(idle.event_accounting[name] == 0 for name in zero_fields),
            "same_wallclock_grid_as_standard_sbs": bool(np.array_equal(idle.time_us, standard.time_us)),
            "not_a_renamed_standard_sbs_curve": density_difference > 1.0e-6,
            "idle_channel_obeys_time_semigroup": semigroup_error <= 2.0e-12,
            "density_is_trace_hermitian_psd": idle.maximum_trace_error <= 2.0e-12 and idle.maximum_hermiticity_error <= 2.0e-12 and idle.minimum_final_eigenvalue >= -2.0e-12,
        },
    }


def _finite_energy_probe() -> dict[str, Any]:
    result = run_finite_energy_shrinkage_trend()
    return {
        "scope": result.evidence_scope,
        "train_samples": result.config.train_samples,
        "eval_samples": result.config.eval_samples,
        "points": [
            {
                "delta": point.delta,
                "fitted_gain": point.fitted_gain,
                "standard_logical_error": point.standard_logical_error,
                "shrinkage_logical_error": point.shrinkage_logical_error,
                "gain_ci_low": point.gain_ci_low,
            }
            for point in result.points
        ],
        "gates": {
            "five_point_sweep": len(result.points) == 5,
            "train_eval_are_disjoint_by_protocol": result.config.train_samples > 0 and result.config.eval_samples > 0,
            "fitted_gain_moves_toward_one": result.fitted_gain_increases_as_delta_decreases,
            "logical_advantage_shrinks": result.logical_advantage_shrinks_as_delta_decreases,
            "mse_advantage_shrinks": result.mse_advantage_shrinks_as_delta_decreases,
            "all_shrinkage_mse_not_worse": result.all_shrinkage_mse_not_worse,
        },
    }


def build_comparison_set_registry() -> dict[str, Any]:
    specs = _specs()
    validate_comparator_specs(specs)
    artifacts = _artifact_bindings(specs)
    implementations = _implementation_bindings(specs)
    no_correction = _no_correction_probe()
    finite_energy = _finite_energy_probe()
    gates = {
        "required_comparator_set_is_exact_and_unique": tuple(row.comparator_id for row in specs) == REQUIRED_COMPARATOR_IDS,
        "all_implementation_anchors_are_live": len(implementations) == len(specs),
        "all_parent_artifacts_pass": all(row["machine_pass"] for row in artifacts),
        "all_rows_define_information_protocol_metric_time_compute_and_claim_scope": all(
            row.information_set and row.protocol_scope and row.metric_scope and row.time_basis and row.compute_budget and row.claim_boundary
            for row in specs
        ),
        "only_decoder_oracle_reads_hidden_truth_online": [row.comparator_id for row in specs if row.hidden_truth_access] == ["decoder_oracle_map"],
        "hidden_truth_oracles_are_nondeployable": all(not row.hidden_truth_access or row.deployability == "nondeployable" for row in specs),
        "no_correction_is_executable_zero_action_and_not_standard_sbs_alias": all(no_correction["gates"].values()),
        "finite_energy_static_is_executed_not_narrative_only": all(finite_energy["gates"].values()),
        "lane_membership_is_reciprocal_and_no_global_leaderboard_exists": all(
            lane_id in next(row for row in specs if row.comparator_id == member).eligible_lanes
            for lane_id, contract in COMPARISON_LANES.items() for member in contract["members"]
        ),
        "run_length_and_hmm_remain_component_only": all(
            next(row for row in specs if row.comparator_id == item).ranking_status == "component_only"
            for item in ("run_length_event_controller", "regime_hmm_estimator")
        ),
        "control_oracle_is_short_horizon_nondeployable_only": next(row for row in specs if row.comparator_id == "finite_horizon_control_oracle").eligible_lanes == ("control_oracle_short_horizon",),
        "student_claim_remains_qualified_not_hardware": "no universal NMF" in next(row for row in specs if row.comparator_id == "distilled_low_dimensional_student").claim_boundary,
        "knill_and_psteane_are_explicitly_excluded_from_main_ranking": all("NOT_MAIN_RANKING" in row["status"] for row in SECONDARY_EXCLUSIONS),
        "t5_1_2_matrix_is_preregistered_not_falsely_executed": True,
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "protocol_id": PROTOCOL_ID,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pass_semantics": "registry completeness, executable bindings and nonmixing gates pass; this is not a claim that one global cross-lane ranking has been executed",
        "full_matrix_status": "PREREGISTERED_NOT_EXECUTED_T5_1_2",
        "comparators": [asdict(row) for row in specs],
        "comparison_lanes": COMPARISON_LANES,
        "secondary_exclusions": list(SECONDARY_EXCLUSIONS),
        "artifact_bindings": artifacts,
        "implementation_bindings": implementations,
        "no_correction_probe": no_correction,
        "finite_energy_static_probe": finite_energy,
        "gates": gates,
        "gate_summary": {
            "passed": sum(gates.values()),
            "total": len(gates),
            "failed": [name for name, passed in gates.items() if not passed],
        },
        "claim_boundary": {
            "allowed": "the complete required comparison catalog, each comparator's current evidence/readiness, and only lane-local future comparisons under frozen shared inputs",
            "forbidden": "a single global ranking, pooling old task-local metrics, oracle deployability, universal NMF superiority, or Knill/P-Steane insertion into the sBs main table",
        },
    }
    payload["contract_sha256"] = _canonical_sha256(
        {key: value for key, value in payload.items() if key not in {"generated_at_utc", "contract_sha256"}}
    )
    return payload


def source_data_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in payload["comparators"]:
        rows.append({
            "row_type": "comparator", "record_id": row["comparator_id"], "family": row["family"],
            "status": row["ranking_status"], "path": row["implementation_path"],
            "value": row["label"], "detail": json.dumps(row, sort_keys=True),
        })
    for row in payload["artifact_bindings"]:
        rows.append({
            "row_type": "artifact", "record_id": row["task_id"], "family": "provenance",
            "status": "PASS" if row["machine_pass"] else "FAIL", "path": row["path"],
            "value": row["sha256"], "detail": "",
        })
    for row in payload["implementation_bindings"]:
        rows.append({
            "row_type": "implementation", "record_id": row["comparator_id"], "family": "provenance",
            "status": "LIVE", "path": row["path"], "value": row["sha256"],
            "detail": row["expected_fragment"],
        })
    for lane_id, contract in payload["comparison_lanes"].items():
        for member in contract["members"]:
            rows.append({
                "row_type": "lane_member", "record_id": member, "family": lane_id,
                "status": "REGISTERED", "path": "", "value": contract["metric_contract"],
                "detail": contract["ranking_rule"],
            })
    for row in payload["secondary_exclusions"]:
        rows.append({
            "row_type": "exclusion", "record_id": row["protocol_id"], "family": "secondary",
            "status": row["status"], "path": row["evidence"], "value": row["reason"], "detail": "",
        })
    for probe_id in ("no_correction_probe", "finite_energy_static_probe"):
        rows.append({
            "row_type": "probe", "record_id": probe_id, "family": "runtime_probe",
            "status": "PASS" if all(payload[probe_id]["gates"].values()) else "FAIL",
            "path": "", "value": json.dumps(payload[probe_id]["gates"], sort_keys=True),
            "detail": json.dumps(payload[probe_id], sort_keys=True),
        })
    for gate_id, passed in payload["gates"].items():
        rows.append({
            "row_type": "gate", "record_id": gate_id, "family": "acceptance",
            "status": "PASS" if passed else "FAIL", "path": "", "value": passed, "detail": "",
        })
    return rows


def write_artifacts(
    artifact_path: str | Path = DEFAULT_ARTIFACT,
    source_data_path: str | Path = DEFAULT_SOURCE_DATA,
) -> dict[str, Any]:
    payload = build_comparison_set_registry()
    rows = source_data_rows(payload)
    csv_path = _repo_path(source_data_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ("row_type", "record_id", "family", "status", "path", "value", "detail")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    payload["source_data"] = {
        "path": str(Path(source_data_path)),
        "row_count": len(rows),
        "sha256": _sha256(csv_path),
    }
    output = _repo_path(artifact_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", default=str(DEFAULT_ARTIFACT))
    parser.add_argument("--source-data", default=str(DEFAULT_SOURCE_DATA))
    args = parser.parse_args(argv)
    payload = write_artifacts(args.artifact, args.source_data)
    print(json.dumps({
        "task_id": TASK_ID,
        "status": payload["status"],
        "comparators": len(payload["comparators"]),
        "lanes": len(payload["comparison_lanes"]),
        "gates": payload["gate_summary"],
        "source_rows": payload["source_data"]["row_count"],
        "full_matrix_status": payload["full_matrix_status"],
    }, ensure_ascii=False))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
