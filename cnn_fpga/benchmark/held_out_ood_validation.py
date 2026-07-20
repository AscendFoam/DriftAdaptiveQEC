"""T5.4.1 pre-registered held-out and out-of-distribution validation.

The campaign deliberately keeps four native evidence lanes separate:

* a frozen T5.1.2 decoder is replayed on unseen drift families/ranges;
* protocol-native sBs readout is replayed with unseen confusion matrices;
* the T5.2.3 leakage/reset kernel is replayed at unseen leakage rates; and
* the T2.4.2 scheduler is replayed under unseen communication patterns.

``PASS`` means that the registered OOD cells were actually executed with
disjoint seeds, frozen parent parameters, complete observables and fail-closed
integrity checks.  It does not mean that one decoder dominates every lane and
does not establish device robustness.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from math import isfinite, pi, sin, cos, sqrt
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cnn_fpga.benchmark import ancilla_readout_causal as ancilla_parent
from cnn_fpga.benchmark import leakage_reset_causal as leakage_parent
from cnn_fpga.benchmark.continuous_adaptive_map import (
    ContinuousAdaptiveValidationConfig,
    FrozenAdaptiveHyperparameters,
    _evaluate_seed,
    _mean_interval,
)
from cnn_fpga.benchmark.mixed_scenario_matrix import (
    DECODER_METHODS,
    decoder_scenarios,
)
from cnn_fpga.benchmark.static_map_baseline import StaticMAPParameters
from cnn_fpga.decoder.periodic_adaptive_map import PeriodicMomentConfig
from cnn_fpga.runtime import timing_fault_model as timing_parent
from physics.constants import LATTICE_CONST
from physics.drift_processes import DriftState, TelegraphDriftProcess
from physics.protocol_ancilla_errors import SBSAncillaFaultOverlay, SBSFaultOverlayConfig
from physics.sbs_observation_reset import make_persistent_leakage_model


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "T5.4.1"
SCHEMA_VERSION = "t5.4.1-held-out-ood-validation-v1"
PROTOCOL_ID = "HELD-OUT-OOD-LANE-LOCAL-PREREG-V1"
DEFAULT_ARTIFACT = Path("docs/t5_4_1_held_out_ood_validation.json")
DEFAULT_SOURCE_DATA = Path("docs/t5_4_1_held_out_ood_source_data.csv")

PARENT_ARTIFACTS: dict[str, Path] = {
    "T5.1.2": Path("docs/t5_1_2_mixed_scenario_matrix.json"),
    "T5.1.4": Path("docs/t5_1_4_algorithm_branch_verdict.json"),
    "T5.1.6": Path("docs/t5_1_6_experimental_feasibility.json"),
    "T5.2.2": Path("docs/t5_2_2_ancilla_readout_causal.json"),
    "T5.2.3": Path("docs/t5_2_3_leakage_reset_causal.json"),
    "T4.2.3": Path("docs/t4_2_3_conservative_fallback_validation.json"),
    "T4.3.3": Path("docs/t4_3_3_closed_loop_fault_recovery_validation.json"),
    "T2.4.2": Path("docs/t2_4_2_timing_fault_validation.json"),
}

IMPLEMENTATION_PATHS = (
    Path("cnn_fpga/benchmark/held_out_ood_validation.py"),
    Path("cnn_fpga/benchmark/mixed_scenario_matrix.py"),
    Path("cnn_fpga/benchmark/continuous_adaptive_map.py"),
    Path("cnn_fpga/benchmark/ancilla_readout_causal.py"),
    Path("cnn_fpga/benchmark/leakage_reset_causal.py"),
    Path("cnn_fpga/runtime/timing_fault_model.py"),
    Path("physics/drift_processes.py"),
    Path("physics/protocol_ancilla_errors.py"),
    Path("physics/sbs_observation_reset.py"),
)

DRIFT_SCENARIOS = (
    "joint_sinusoidal_rotation_unseen_family",
    "stochastic_telegraph_unseen_family",
    "compound_range_extrapolation",
)
MEASUREMENT_SCENARIOS = (
    "asymmetric_g_to_e_confusion",
    "asymmetric_e_to_g_confusion",
    "high_symmetric_confusion",
)
COMMUNICATION_SCENARIOS = (
    "periodic_micro_outages",
    "increasing_duration_flaps",
    "communication_jitter_burst_compound",
)
MEASUREMENT_CONFUSION: dict[str, tuple[tuple[float, float, float], ...]] = {
    "asymmetric_g_to_e_confusion": (
        (0.85, 0.15, 0.0),
        (0.03, 0.97, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0),
    ),
    "asymmetric_e_to_g_confusion": (
        (0.96, 0.04, 0.0),
        (0.18, 0.82, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0),
    ),
    "high_symmetric_confusion": (
        (0.82, 0.18, 0.0),
        (0.18, 0.82, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0),
    ),
}
LEAKAGE_OOD_RATES = (0.003, 0.006, 0.012)
DRIFT_EVALUATION_SEEDS = tuple(202607154101 + index for index in range(8))
MEASUREMENT_EVALUATION_SEEDS = tuple(202607154201 + index for index in range(8))
LEAKAGE_EVALUATION_SEEDS = tuple(202607154301 + index for index in range(8))
COMMUNICATION_EVALUATION_SEEDS = tuple(202607154401 + index for index in range(8))


@dataclass(frozen=True)
class HeldOutOODConfig:
    drift_scenarios: tuple[str, ...] = DRIFT_SCENARIOS
    measurement_scenarios: tuple[str, ...] = MEASUREMENT_SCENARIOS
    communication_scenarios: tuple[str, ...] = COMMUNICATION_SCENARIOS
    leakage_ood_rates: tuple[float, ...] = LEAKAGE_OOD_RATES
    drift_evaluation_seeds: tuple[int, ...] = DRIFT_EVALUATION_SEEDS
    measurement_evaluation_seeds: tuple[int, ...] = MEASUREMENT_EVALUATION_SEEDS
    leakage_evaluation_seeds: tuple[int, ...] = LEAKAGE_EVALUATION_SEEDS
    communication_evaluation_seeds: tuple[int, ...] = COMMUNICATION_EVALUATION_SEEDS
    drift_windows: int = 64
    measurement_cycles_per_seed: int = 8192
    communication_cycles_per_seed: int = 24_000
    communication_bootstrap_replicates: int = 10_000
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        frozen = {
            "drift_scenarios": DRIFT_SCENARIOS,
            "measurement_scenarios": MEASUREMENT_SCENARIOS,
            "communication_scenarios": COMMUNICATION_SCENARIOS,
            "leakage_ood_rates": LEAKAGE_OOD_RATES,
            "drift_evaluation_seeds": DRIFT_EVALUATION_SEEDS,
            "measurement_evaluation_seeds": MEASUREMENT_EVALUATION_SEEDS,
            "leakage_evaluation_seeds": LEAKAGE_EVALUATION_SEEDS,
            "communication_evaluation_seeds": COMMUNICATION_EVALUATION_SEEDS,
        }
        for name, expected in frozen.items():
            if tuple(getattr(self, name)) != expected:
                raise ValueError(f"formal {name} changed")
        exact = {
            "drift_windows": 64,
            "measurement_cycles_per_seed": 8192,
            "communication_cycles_per_seed": 24_000,
            "communication_bootstrap_replicates": 10_000,
            "confidence_level": 0.95,
        }
        for name, expected in exact.items():
            if getattr(self, name) != expected:
                raise ValueError(f"formal {name} changed")
        groups = (
            self.drift_evaluation_seeds,
            self.measurement_evaluation_seeds,
            self.leakage_evaluation_seeds,
            self.communication_evaluation_seeds,
        )
        if any(len(group) != 8 or len(set(group)) != len(group) for group in groups):
            raise ValueError("each OOD lane requires eight unique seed clusters")
        union = set().union(*(set(group) for group in groups))
        if len(union) != sum(len(group) for group in groups):
            raise ValueError("OOD lane seed clusters must be pairwise disjoint")


def _repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(_repo_path(path).read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    data = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(data).hexdigest()


def _seed_stream(seed: int, stream: str) -> int:
    digest = hashlib.sha256(f"{seed}:{stream}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "little")


def _parent_pass(payload: Mapping[str, Any]) -> bool:
    if payload.get("status") == "PASS" or payload.get("passed") is True:
        return True
    gate = payload.get("gate")
    if isinstance(gate, Mapping) and gate.get("passed") is True:
        return True
    checks = payload.get("checks")
    return bool(isinstance(checks, Mapping) and checks and all(checks.values()))


def _extract_seed_values(value: Any, *, key_path: str = "") -> set[int]:
    found: set[int] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{key_path}.{key}" if key_path else str(key)
            found.update(_extract_seed_values(child, key_path=child_path))
    elif isinstance(value, (list, tuple)):
        for child in value:
            found.update(_extract_seed_values(child, key_path=key_path))
    elif "seed" in key_path.lower() and isinstance(value, int) and not isinstance(value, bool):
        found.add(int(value))
    return found


def load_parent_artifacts() -> dict[str, dict[str, Any]]:
    return {
        task_id: json.loads(_repo_path(path).read_text(encoding="utf-8"))
        for task_id, path in PARENT_ARTIFACTS.items()
    }


def parent_bindings(
    parents: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    missing = set(PARENT_ARTIFACTS) - set(parents)
    if missing:
        raise ValueError(f"missing parent artifacts: {sorted(missing)}")
    return [
        {
            "task_id": task_id,
            "path": path.as_posix(),
            "sha256": _sha256(path),
            "machine_pass": _parent_pass(parents[task_id]),
            "registered_seed_count": len(_extract_seed_values(parents[task_id])),
        }
        for task_id, path in PARENT_ARTIFACTS.items()
    ]


def implementation_bindings() -> list[dict[str, str]]:
    return [
        {"path": path.as_posix(), "sha256": _sha256(path)}
        for path in IMPLEMENTATION_PATHS
    ]


def _parent_seed_union(parents: Mapping[str, Mapping[str, Any]]) -> set[int]:
    return set().union(*(_extract_seed_values(payload) for payload in parents.values()))


def _state_envelope(states: Sequence[DriftState]) -> dict[str, float]:
    if not states:
        raise ValueError("state envelope requires at least one state")
    fields = {
        "abs_mu_q_max": max(abs(state.mu_q) for state in states),
        "abs_mu_p_max": max(abs(state.mu_p) for state in states),
        "sigma_q_min": min(state.sigma_q for state in states),
        "sigma_q_max": max(state.sigma_q for state in states),
        "sigma_p_min": min(state.sigma_p for state in states),
        "sigma_p_max": max(state.sigma_p for state in states),
        "abs_rho_max": max(abs(state.rho) for state in states),
        "p_outlier_max": max(state.p_outlier for state in states),
        "outlier_scale_max": max(state.outlier_scale for state in states),
    }
    return {name: float(value) for name, value in fields.items()}


@dataclass(frozen=True)
class OODDriftScenario:
    scenario_id: str
    dynamic_seed: int

    def __post_init__(self) -> None:
        if self.scenario_id not in DRIFT_SCENARIOS:
            raise ValueError(f"unknown OOD drift scenario: {self.scenario_id}")
        if isinstance(self.dynamic_seed, bool) or not isinstance(self.dynamic_seed, int):
            raise TypeError("dynamic_seed must be an integer")

    def states(self, windows: int) -> tuple[DriftState, ...]:
        if isinstance(windows, bool) or not isinstance(windows, int) or windows < 16:
            raise ValueError("windows must be an integer >= 16")
        lam = float(LATTICE_CONST)
        if self.scenario_id == "stochastic_telegraph_unseen_family":
            state_a = DriftState(
                mu_q=-0.15 * lam,
                mu_p=0.11 * lam,
                sigma_q=0.12 * lam,
                sigma_p=0.20 * lam,
                rho=-0.55,
                source="t5.4.1-preregistered-telegraph-a",
                regime="a",
            )
            state_b = DriftState(
                mu_q=0.16 * lam,
                mu_p=-0.13 * lam,
                sigma_q=0.21 * lam,
                sigma_p=0.115 * lam,
                rho=0.58,
                source="t5.4.1-preregistered-telegraph-b",
                regime="b",
            )
            return TelegraphDriftProcess(
                state_a=state_a,
                state_b=state_b,
                p_a_to_b=0.12,
                p_b_to_a=0.08,
                initial_regime="a" if self.dynamic_seed % 2 == 0 else "b",
                seed=self.dynamic_seed,
            ).generate(windows)

        states: list[DriftState] = []
        for step in range(windows):
            progress = step / (windows - 1)
            phase = 2.0 * pi * (3.0 * progress + (self.dynamic_seed % 17) / 17.0)
            if self.scenario_id == "joint_sinusoidal_rotation_unseen_family":
                mu_q = (0.145 * sin(phase) + 0.025 * sin(3.0 * phase)) * lam
                mu_p = (0.135 * cos(phase) - 0.02 * sin(2.0 * phase)) * lam
                sigma_q = (0.155 + 0.045 * sin(phase + pi / 4.0)) * lam
                sigma_p = (0.155 - 0.04 * sin(phase + pi / 4.0)) * lam
                rho = 0.58 * sin(2.0 * phase)
                p_outlier = 0.0
                outlier_scale = 1.0
                event_id = int(phase // (2.0 * pi))
            else:
                # All extrema exceed the T5.1.2 fitting/evaluation envelope.  The
                # compound nonstationarity is frozen before seeing OOD results.
                triangle = 2.0 * abs(2.0 * ((2.0 * progress) % 1.0) - 1.0) - 1.0
                mu_q = (0.30 * triangle + 0.04 * sin(5.0 * phase)) * lam
                mu_p = (-0.27 * triangle + 0.05 * cos(3.0 * phase)) * lam
                sigma_q = (0.12 + 0.20 * (0.5 + 0.5 * sin(phase))) * lam
                sigma_p = (0.10 + 0.19 * (0.5 + 0.5 * cos(phase))) * lam
                rho = 0.86 * sin(phase)
                p_outlier = 0.18 if step % 11 in (0, 1, 2) else 0.04
                outlier_scale = 6.0 if p_outlier > 0.1 else 3.5
                event_id = 1 + step // 11
            states.append(
                DriftState(
                    step=step,
                    time=float(step),
                    mu_q=float(mu_q),
                    mu_p=float(mu_p),
                    sigma_q=float(sigma_q),
                    sigma_p=float(sigma_p),
                    rho=float(rho),
                    p_outlier=float(p_outlier),
                    outlier_scale=float(outlier_scale),
                    burst_active=p_outlier > 0.1,
                    source="t5.4.1-held-out-ood",
                    regime=self.scenario_id,
                    seed=self.dynamic_seed,
                    event_id=event_id,
                )
            )
        return tuple(states)


def _restore_decoder_parent(
    parent: Mapping[str, Any], config: HeldOutOODConfig
) -> tuple[
    ContinuousAdaptiveValidationConfig,
    FrozenAdaptiveHyperparameters,
    StaticMAPParameters,
    dict[str, Any],
]:
    lane = parent["decoder_lane"]
    parent_config = lane["config"]
    settings = ContinuousAdaptiveValidationConfig(
        training_seeds=tuple(parent_config["training_seeds"]),
        evaluation_seeds=config.drift_evaluation_seeds,
        windows=config.drift_windows,
        calibration_windows=int(parent_config["calibration_windows"]),
        observation_samples_per_window=int(parent_config["observation_samples_per_window"]),
        training_score_samples_per_window=int(
            parent_config["training_score_samples_per_window"]
        ),
        evaluation_samples_per_window=int(parent_config["evaluation_samples_per_window"]),
        ewma_alpha_candidates=tuple(parent_config["ewma_alpha_candidates"]),
        kalman_process_scale_candidates=tuple(
            parent_config["kalman_process_scale_candidates"]
        ),
        kalman_measurement_scale_candidates=tuple(
            parent_config["kalman_measurement_scale_candidates"]
        ),
        confidence_level=float(parent_config["confidence_level"]),
    )
    hyper = lane["frozen_hyperparameters"]
    frozen = FrozenAdaptiveHyperparameters(
        ewma_alpha=float(hyper["ewma_alpha"]),
        kalman_process_scale=float(hyper["kalman_process_scale"]),
        kalman_measurement_scale=float(hyper["kalman_measurement_scale"]),
        ewma_candidate_scores=tuple(tuple(row) for row in hyper["ewma_candidate_scores"]),
        kalman_candidate_scores=tuple(
            tuple(row) for row in hyper["kalman_candidate_scores"]
        ),
        training_trace_sha256=str(hyper["training_trace_sha256"]),
        selection_objective=str(hyper["selection_objective"]),
    )
    static_map = StaticMAPParameters(**lane["static_training_parameters"])
    frozen_binding = {
        "parent_task": "T5.1.2",
        "parent_training_seeds": list(parent_config["training_seeds"]),
        "parent_evaluation_seeds_excluded": list(parent_config["evaluation_seeds"]),
        "frozen_hyperparameters_sha256": _canonical_sha256(hyper),
        "static_parameters_sha256": _canonical_sha256(lane["static_training_parameters"]),
        "hyperparameters_reselected_on_ood": False,
        "static_parameters_refit_on_ood": False,
    }
    return settings, frozen, static_map, frozen_binding


def _run_drift_lane(
    parent: Mapping[str, Any], config: HeldOutOODConfig
) -> dict[str, Any]:
    settings, frozen, static_map, frozen_binding = _restore_decoder_parent(parent, config)
    moment = PeriodicMomentConfig(
        minimum_samples=min(64, settings.observation_samples_per_window)
    )
    parent_states = [
        state
        for scenario in decoder_scenarios()
        for state in scenario.states(int(parent["decoder_lane"]["config"]["windows"]))
    ]
    parent_envelope = _state_envelope(parent_states)
    seed_rows: list[dict[str, Any]] = []
    scenario_states: dict[str, list[DriftState]] = {name: [] for name in DRIFT_SCENARIOS}
    for scenario_index, scenario_id in enumerate(DRIFT_SCENARIOS, start=70):
        for base_seed in config.drift_evaluation_seeds:
            scenario = OODDriftScenario(
                scenario_id=scenario_id,
                dynamic_seed=_seed_stream(base_seed, f"drift-dynamics:{scenario_id}"),
            )
            states = scenario.states(settings.windows)
            scenario_states[scenario_id].extend(states)
            row = dict(
                _evaluate_seed(
                    scenario,
                    scenario_index,
                    base_seed,
                    settings,
                    frozen,
                    static_map,
                    moment,
                )
            )
            row["base_evaluation_seed"] = base_seed
            row["dynamic_seed"] = scenario.dynamic_seed
            row["shared_trace_for_all_methods"] = True
            seed_rows.append(row)

    aggregates: list[dict[str, Any]] = []
    for scenario_id in DRIFT_SCENARIOS:
        rows = [row for row in seed_rows if row["scenario_id"] == scenario_id]
        methods = {
            method: {
                "error_rate_seed_cluster_ci": _mean_interval(
                    [float(row[f"{method}_error_rate"]) for row in rows],
                    config.confidence_level,
                ),
                **(
                    {}
                    if method == "standard"
                    else {
                        "nll_seed_cluster_ci": _mean_interval(
                            [float(row[f"{method}_nll"]) for row in rows],
                            config.confidence_level,
                        ),
                        "brier_seed_cluster_ci": _mean_interval(
                            [float(row[f"{method}_brier"]) for row in rows],
                            config.confidence_level,
                        ),
                        "oracle_gap_error_rate_seed_cluster_ci": _mean_interval(
                            [
                                float(row[f"{method}_error_rate"])
                                - float(row["oracle_error_rate"])
                                for row in rows
                            ],
                            config.confidence_level,
                        ),
                    }
                ),
            }
            for method in DECODER_METHODS
        }
        envelope = _state_envelope(scenario_states[scenario_id])
        aggregates.append(
            {
                "scenario_id": scenario_id,
                "seed_clusters": len(rows),
                "evaluation_samples": sum(int(row["evaluation_samples"]) for row in rows),
                "unique_trace_hashes": len({row["trace_sha256"] for row in rows}),
                "state_envelope": envelope,
                "exceeds_parent_envelope": {
                    name: bool(
                        (name.endswith("_min") and envelope[name] < parent_envelope[name])
                        or (not name.endswith("_min") and envelope[name] > parent_envelope[name])
                    )
                    for name in parent_envelope
                },
                "methods": methods,
                "paired_contrasts": {
                    "static_minus_ewma_error_rate": _mean_interval(
                        [float(row["static_minus_ewma_error_rate"]) for row in rows],
                        config.confidence_level,
                    ),
                    "static_minus_kalman_error_rate": _mean_interval(
                        [float(row["static_minus_kalman_error_rate"]) for row in rows],
                        config.confidence_level,
                    ),
                    "window_minus_best_recursive_error_rate": _mean_interval(
                        [
                            float(row["window_minus_best_recursive_error_rate"])
                            for row in rows
                        ],
                        config.confidence_level,
                    ),
                },
            }
        )
    return {
        "lane_id": "frozen_decoder_drift_ood",
        "scope": "paired syndrome-decoder decisions under unseen synthetic drift; not physical-memory/device robustness",
        "parent_scenario_ids": list(parent["decoder_lane"]["executed_scenarios"]),
        "registered_ood_scenario_ids": list(DRIFT_SCENARIOS),
        "parent_state_envelope": parent_envelope,
        "frozen_parent_binding": frozen_binding,
        "config": asdict(settings),
        "seed_rows": seed_rows,
        "scenario_aggregates": aggregates,
        "selection_used_ood_results": False,
    }


def _measurement_overlay(matrix: Sequence[Sequence[float]]) -> SBSAncillaFaultOverlay:
    base = make_persistent_leakage_model(
        readout_confusion=np.asarray(matrix, dtype=np.float64),
        f_injection_given_g=0.0,
        f_injection_given_e=0.0,
        higher_injection_given_g=0.0,
        higher_injection_given_e=0.0,
        e_reset_success=1.0,
        f_reset_success=1.0,
        higher_reset_success=1.0,
        counter_max=2**31 - 1,
        readout_provenance="T5.4.1 pre-registered OOD confusion matrix",
        parameter_provenance="T5.4.1 no-leakage/no-ancilla-flip isolation",
    )
    return SBSAncillaFaultOverlay(
        base,
        SBSFaultOverlayConfig(
            bit_flip_probabilities=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            phase_flip_probabilities=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            logical_fault_given_big_cd_bit=(0.0, 0.0),
            phase_backaction_scale=(0.0, 0.0),
            small_cd_bit_backaction_scale=(0.0, 0.0),
            misclassification_rotation_max_rad=ancilla_parent.VIRTUAL_ROTATION_MAX_RAD,
            parameter_provenance="T5.4.1 measurement-only OOD intervention",
        ),
    )


def _run_measurement_seed(
    scenario_id: str, matrix: Sequence[Sequence[float]], seed: int, cycles: int
) -> dict[str, Any]:
    trajectory = _measurement_overlay(matrix).simulate(
        ancilla_parent._ideal_labels(cycles),
        seed=_seed_stream(seed, "measurement-confusion-paired-stream"),
    )
    counts = Counter()
    mismatches = 0
    rotations: list[float] = []
    label_changes = 0
    deployable_keys: set[str] | None = None
    digest = hashlib.sha256()
    expected_keys = {
        "cycle_index",
        "syndrome_x",
        "syndrome_z",
        "reset_action_x",
        "reset_action_z",
        "x_e_run",
        "z_e_run",
        "leakage_constituent_run",
        "leakage_cycle_run",
        "observation_scope",
    }
    for step in trajectory.steps:
        hidden = step.observation_reset.truth.hidden_pre_readout
        observed = step.observation_reset.observed.syndrome.as_tuple()
        for hidden_value, observed_value, rotation in zip(
            hidden, observed, step.fault_truth.virtual_rotation_error_rad
        ):
            if hidden_value in {"g", "e"}:
                counts[f"{hidden_value}_total"] += 1
                counts[f"{hidden_value}_to_{observed_value}"] += 1
                mismatches += int(hidden_value != observed_value)
            rotations.append(abs(float(rotation)))
            digest.update(hidden_value.encode("ascii"))
            digest.update(observed_value.encode("ascii"))
            digest.update(np.asarray(rotation, dtype="<f8").tobytes())
        label_changes += int(
            step.fault_truth.original_ideal_kraus_label
            != step.fault_truth.faulted_ideal_kraus_label
        )
        keys = set(step.deployable_record())
        deployable_keys = keys if deployable_keys is None else deployable_keys | keys
    g_to_e = counts["g_to_e"] / counts["g_total"]
    e_to_g = counts["e_to_g"] / counts["e_total"]
    return {
        "scenario_id": scenario_id,
        "seed": seed,
        "cycles": cycles,
        "constituent_observations": 2 * cycles,
        "confusion_matrix": [list(row) for row in matrix],
        "g_observation_count": counts["g_total"],
        "e_observation_count": counts["e_total"],
        "empirical_g_to_e": g_to_e,
        "empirical_e_to_g": e_to_g,
        "misclassification_rate": mismatches / (2 * cycles),
        "nonzero_virtual_rotation_rate": float(np.mean(np.asarray(rotations) > 0.0)),
        "mean_abs_virtual_rotation_rad": float(np.mean(rotations)),
        "faulted_label_change_count": label_changes,
        "ancilla_bit_phase_event_count": sum(
            len(step.fault_truth.events) for step in trajectory.steps
        ),
        "deployable_schema_exact": deployable_keys == expected_keys,
        "truth_used_only_for_scoring": True,
        "trace_sha256": digest.hexdigest(),
    }


def _binomial_tolerance(probability: float, trials: int) -> float:
    return 5.0 * sqrt(max(probability * (1.0 - probability), 1.0e-12) / trials) + 1.0 / trials


def _run_measurement_lane(config: HeldOutOODConfig) -> dict[str, Any]:
    rows = [
        _run_measurement_seed(
            scenario_id,
            MEASUREMENT_CONFUSION[scenario_id],
            seed,
            config.measurement_cycles_per_seed,
        )
        for scenario_id in MEASUREMENT_SCENARIOS
        for seed in config.measurement_evaluation_seeds
    ]
    aggregates: list[dict[str, Any]] = []
    for scenario_id in MEASUREMENT_SCENARIOS:
        selected = [row for row in rows if row["scenario_id"] == scenario_id]
        matrix = MEASUREMENT_CONFUSION[scenario_id]
        g_target = float(matrix[0][1])
        e_target = float(matrix[1][0])
        g_trials = sum(int(row["g_observation_count"]) for row in selected)
        e_trials = sum(int(row["e_observation_count"]) for row in selected)
        g_empirical = sum(
            float(row["empirical_g_to_e"]) * int(row["g_observation_count"])
            for row in selected
        ) / g_trials
        e_empirical = sum(
            float(row["empirical_e_to_g"]) * int(row["e_observation_count"])
            for row in selected
        ) / e_trials
        aggregates.append(
            {
                "scenario_id": scenario_id,
                "seed_clusters": len(selected),
                "target_g_to_e": g_target,
                "target_e_to_g": e_target,
                "empirical_g_to_e": g_empirical,
                "empirical_e_to_g": e_empirical,
                "g_to_e_tolerance": _binomial_tolerance(g_target, g_trials),
                "e_to_g_tolerance": _binomial_tolerance(e_target, e_trials),
                "confusion_rates_within_preregistered_tolerance": (
                    abs(g_empirical - g_target) <= _binomial_tolerance(g_target, g_trials)
                    and abs(e_empirical - e_target) <= _binomial_tolerance(e_target, e_trials)
                ),
                "misclassification_rate_seed_cluster_ci": _mean_interval(
                    [float(row["misclassification_rate"]) for row in selected],
                    config.confidence_level,
                ),
                "mean_abs_virtual_rotation_seed_cluster_ci": _mean_interval(
                    [float(row["mean_abs_virtual_rotation_rad"]) for row in selected],
                    config.confidence_level,
                ),
                "unique_trace_hashes": len({row["trace_sha256"] for row in selected}),
            }
        )
    return {
        "lane_id": "sbs_measurement_confusion_ood",
        "scope": "protocol-native effective readout-confusion sensitivity; not device-calibrated measurement fidelity",
        "parent_rate_grid": list(ancilla_parent.INJECTION_RATES),
        "registered_confusion_matrices": {
            key: [list(row) for row in value]
            for key, value in MEASUREMENT_CONFUSION.items()
        },
        "seed_rows": rows,
        "scenario_aggregates": aggregates,
        "selection_used_ood_results": False,
    }


def _run_leakage_lane(config: HeldOutOODConfig) -> dict[str, Any]:
    kernel_config = leakage_parent.CampaignConfig()
    rows = [
        leakage_parent._run_seed_cell(
            "higher_leakage_injection",
            rate,
            seed,
            config=kernel_config,
        )
        for rate in config.leakage_ood_rates
        for seed in config.leakage_evaluation_seeds
    ]
    aggregates: list[dict[str, Any]] = []
    metrics = (
        "empirical_higher_injection_probability",
        "detection_probability",
        "false_alarm_rate_per_healthy_step",
        "hidden_leakage_occupancy",
        "observed_leakage_alarm_rate",
        "declared_normal_action_availability",
        "safe_normal_action_availability",
        "unsafe_declared_available_fraction",
        "reset_attempts_per_1000_cycles",
        "reset_failures_per_1000_cycles",
        "mean_hidden_leakage_run_steps",
        "p95_hidden_leakage_run_steps",
    )
    for rate in config.leakage_ood_rates:
        selected = [row for row in rows if row["intervention_rate"] == rate]
        aggregate: dict[str, Any] = {
            "intervention_rate": rate,
            "seed_clusters": len(selected),
            "unique_trace_hashes": len({row["trace_sha256"] for row in selected}),
        }
        for metric in metrics:
            values = [row[metric] for row in selected]
            if any(value is None for value in values):
                aggregate[metric] = {
                    "status": "NOT_APPLICABLE_OR_PARTIAL_NULL",
                    "values": values,
                }
            else:
                aggregate[f"{metric}_seed_cluster_ci"] = _mean_interval(
                    [float(value) for value in values], config.confidence_level
                )
        trials = sum(
            int(round(float(row["injection_episode_count"]) / float(row["empirical_higher_injection_probability"])))
            for row in selected
            if float(row["empirical_higher_injection_probability"]) > 0.0
        )
        empirical = aggregate[
            "empirical_higher_injection_probability_seed_cluster_ci"
        ]["estimate"]
        aggregate["injection_rate_tolerance"] = _binomial_tolerance(rate, max(trials, 1))
        aggregate["injection_rate_within_preregistered_tolerance"] = bool(
            abs(float(empirical) - rate) <= aggregate["injection_rate_tolerance"]
        )
        aggregates.append(aggregate)
    return {
        "lane_id": "persistent_leakage_rate_ood",
        "scope": "effective higher-level leakage/reset sensitivity with hidden truth only for scoring; not cavity-device leakage calibration",
        "parent_rate_grid": list(leakage_parent.LEAKAGE_INJECTION_RATES),
        "registered_ood_rates": list(config.leakage_ood_rates),
        "fixed_reset_failure_probability": leakage_parent.FIXED_RESET_FAILURE_FOR_LEAKAGE_FAMILY,
        "kernel_config": asdict(kernel_config),
        "seed_rows": rows,
        "rate_aggregates": aggregates,
        "selection_used_ood_results": False,
    }


def _communication_patterns(n_cycles: int) -> tuple[timing_parent.TimingFaultScenario, ...]:
    if n_cycles != 24_000:
        raise ValueError("formal communication OOD campaign requires 24,000 cycles")
    micro = tuple((start, start + 200) for start in range(3_200, 21_201, 2_000))
    increasing = ((3_000, 3_100), (6_000, 6_400), (10_000, 11_600), (16_000, 19_200))
    compound_pauses = ((4_000, 4_500), (9_000, 10_200), (17_000, 19_400))
    return (
        timing_parent.TimingFaultScenario(name="reference"),
        timing_parent.TimingFaultScenario(
            name="periodic_micro_outages", communication_pauses=micro
        ),
        timing_parent.TimingFaultScenario(
            name="increasing_duration_flaps", communication_pauses=increasing
        ),
        timing_parent.TimingFaultScenario(
            name="communication_jitter_burst_compound",
            slow_mean_scale=10.0,
            slow_std_scale=8.0,
            fast_mean_us=1.35,
            fast_std_us=0.38,
            burst_epochs=(7_500, 15_500, 21_000),
            burst_size=6,
            communication_pauses=compound_pauses,
            max_pending_windows=3,
        ),
    )


def _run_communication_lane(config: HeldOutOODConfig) -> dict[str, Any]:
    timing_config = timing_parent.TimingStressConfig(
        n_cycles=config.communication_cycles_per_seed,
        seeds=config.communication_evaluation_seeds,
        measurement_noise_sigma=0.10,
        channel_noise_sigma=0.34,
        evaluation_warmup_windows=2,
        bootstrap_replicates=config.communication_bootstrap_replicates,
        bootstrap_seed=202607154499,
    )
    scenarios = _communication_patterns(timing_config.n_cycles)
    source_config = timing_parent.load_yaml_config(timing_parent.DEFAULT_CONFIG)
    rows = [
        timing_parent.simulate_scenario(
            scenario,
            config=timing_config,
            seed=seed,
            yaml_config=source_config,
        )
        for scenario in scenarios
        for seed in timing_config.seeds
    ]
    by_scenario = {
        scenario.name: [row for row in rows if row["scenario"] == scenario.name]
        for scenario in scenarios
    }
    reference_by_seed = {
        int(row["seed"]): row for row in by_scenario["reference"]
    }
    aggregates = [
        timing_parent._aggregate_results(
            by_scenario[scenario.name],
            reference_by_seed=reference_by_seed,
            config=timing_config,
            scenario_index=index,
        )
        for index, scenario in enumerate(scenarios)
    ]
    return {
        "lane_id": "scheduler_communication_disturbance_ood",
        "scope": "paired software timing/scheduler stress; not transport or target-board measurement",
        "config": asdict(timing_config),
        "scenarios": [asdict(scenario) for scenario in scenarios],
        "per_seed_results": rows,
        "scenario_aggregates": aggregates,
        "selection_used_ood_results": False,
        "target_hardware_measured": False,
    }


def _all_ood_seed_groups(config: HeldOutOODConfig) -> tuple[tuple[int, ...], ...]:
    return (
        config.drift_evaluation_seeds,
        config.measurement_evaluation_seeds,
        config.leakage_evaluation_seeds,
        config.communication_evaluation_seeds,
    )


def _finite_metric_rows(rows: Sequence[Mapping[str, Any]], suffixes: Sequence[str]) -> bool:
    for row in rows:
        for key, value in row.items():
            if any(key.endswith(suffix) for suffix in suffixes):
                # Vector diagnostics (for example correlation_lags_steps) have
                # their own shape checks in the parent kernel.  This helper is
                # intentionally a scalar-finiteness audit and must not mistake
                # a finite vector for a malformed scalar.
                if isinstance(value, (Mapping, list, tuple)):
                    continue
                if value is not None and (not isinstance(value, (int, float)) or not isfinite(float(value))):
                    return False
    return True


def _forbidden_cross_lane_key(value: Any) -> bool:
    forbidden = {
        "universal_rank",
        "universal_score",
        "cross_lane_score",
        "system_robustness_established",
    }
    if isinstance(value, Mapping):
        return any(key in forbidden or _forbidden_cross_lane_key(child) for key, child in value.items())
    if isinstance(value, list):
        return any(_forbidden_cross_lane_key(child) for child in value)
    return False


def _compute_gates(
    report: Mapping[str, Any], parents: Mapping[str, Mapping[str, Any]]
) -> dict[str, bool]:
    config = HeldOutOODConfig()
    parent_seed_set = _parent_seed_union(parents)
    ood_seed_set = set().union(*(set(group) for group in _all_ood_seed_groups(config)))
    drift = report["drift_lane"]
    measurement = report["measurement_confusion_lane"]
    leakage = report["leakage_rate_lane"]
    communication = report["communication_lane"]
    parent_drift_ids = set(drift["parent_scenario_ids"])
    range_row = next(
        row
        for row in drift["scenario_aggregates"]
        if row["scenario_id"] == "compound_range_extrapolation"
    )
    communication_rows = communication["per_seed_results"]
    expected_comm_transitions = {
        scenario: all(
            row["event_counts"].get("communication_pause_started", 0) > 0
            and row["event_counts"].get("communication_pause_ended", 0) > 0
            for row in communication_rows
            if row["scenario"] == scenario
        )
        for scenario in COMMUNICATION_SCENARIOS
    }
    leakage_occupancy = [
        row["hidden_leakage_occupancy_seed_cluster_ci"]["estimate"]
        for row in leakage["rate_aggregates"]
    ]
    expected_measurement_matrices = {
        key: [list(row) for row in value]
        for key, value in MEASUREMENT_CONFUSION.items()
    }
    expected_communication_scenarios = [
        asdict(scenario)
        for scenario in _communication_patterns(config.communication_cycles_per_seed)
    ]
    expected_parent_states = [
        state
        for scenario in decoder_scenarios()
        for state in scenario.states(32)
    ]
    expected_range_states = [
        state
        for seed in config.drift_evaluation_seeds
        for state in OODDriftScenario(
            scenario_id="compound_range_extrapolation",
            dynamic_seed=_seed_stream(seed, "drift-dynamics:compound_range_extrapolation"),
        ).states(config.drift_windows)
    ]
    return {
        "all_parent_artifacts_hash_bound_and_pass": all(
            binding["machine_pass"] and binding["sha256"] == _sha256(binding["path"])
            for binding in report["parent_bindings"]
        ),
        "implementation_files_hash_bound": len(report["implementation_bindings"])
        == len(IMPLEMENTATION_PATHS)
        and all(binding["sha256"] == _sha256(binding["path"]) for binding in report["implementation_bindings"]),
        "ood_seed_groups_pairwise_disjoint_and_parent_disjoint": len(ood_seed_set) == 32
        and not (ood_seed_set & parent_seed_set),
        "exact_four_native_lane_coverage": set(report["lane_ids"])
        == {
            "frozen_decoder_drift_ood",
            "sbs_measurement_confusion_ood",
            "persistent_leakage_rate_ood",
            "scheduler_communication_disturbance_ood",
        },
        "unseen_drift_families_are_not_parent_scenarios": not (
            set(DRIFT_SCENARIOS[:2]) & parent_drift_ids
        ),
        "compound_drift_exceeds_parent_parameter_envelope": sum(
            bool(value) for value in range_row["exceeds_parent_envelope"].values()
        )
        >= 5
        and drift["parent_state_envelope"] == _state_envelope(expected_parent_states)
        and range_row["state_envelope"] == _state_envelope(expected_range_states),
        "decoder_hyperparameters_and_static_map_restored_without_ood_selection": not drift[
            "frozen_parent_binding"
        ]["hyperparameters_reselected_on_ood"]
        and not drift["frozen_parent_binding"]["static_parameters_refit_on_ood"]
        and not drift["selection_used_ood_results"],
        "drift_cells_complete_shared_trace_and_finite": len(drift["seed_rows"])
        == len(DRIFT_SCENARIOS) * 8
        and all(row["shared_trace_for_all_methods"] for row in drift["seed_rows"])
        and all(row["unique_trace_hashes"] == 8 for row in drift["scenario_aggregates"])
        and _finite_metric_rows(drift["seed_rows"], ("_rate", "_nll", "_brier", "_lattice", "_lattice2")),
        "measurement_confusion_matrices_are_new_row_stochastic_interventions": measurement[
            "registered_confusion_matrices"
        ]
        == expected_measurement_matrices
        and all(
            np.allclose(np.sum(np.asarray(matrix), axis=1), 1.0, rtol=0.0, atol=1.0e-12)
            and max(float(matrix[0][1]), float(matrix[1][0]))
            > max(ancilla_parent.INJECTION_RATES)
            for matrix in measurement["registered_confusion_matrices"].values()
        ),
        "measurement_confusion_realized_within_preregistered_tolerance": all(
            row["confusion_rates_within_preregistered_tolerance"]
            and abs(float(row["empirical_g_to_e"]) - float(row["target_g_to_e"]))
            <= float(row["g_to_e_tolerance"])
            and abs(float(row["empirical_e_to_g"]) - float(row["target_e_to_g"]))
            <= float(row["e_to_g_tolerance"])
            for row in measurement["scenario_aggregates"]
        ),
        "measurement_only_isolation_and_truth_boundary_hold": len(measurement["seed_rows"])
        == len(MEASUREMENT_SCENARIOS) * 8
        and all(
            row["ancilla_bit_phase_event_count"] == 0
            and row["faulted_label_change_count"] == 0
            and row["deployable_schema_exact"]
            and row["truth_used_only_for_scoring"]
            for row in measurement["seed_rows"]
        ),
        "leakage_rates_are_unseen_and_include_extrapolation": not (
            set(config.leakage_ood_rates) & set(leakage_parent.LEAKAGE_INJECTION_RATES)
        )
        and max(config.leakage_ood_rates) > max(leakage_parent.LEAKAGE_INJECTION_RATES),
        "leakage_cells_complete_rate_calibrated_and_finite": len(leakage["seed_rows"])
        == len(LEAKAGE_OOD_RATES) * 8
        and all(
            row["injection_rate_within_preregistered_tolerance"]
            and abs(
                float(
                    row["empirical_higher_injection_probability_seed_cluster_ci"][
                        "estimate"
                    ]
                )
                - float(row["intervention_rate"])
            )
            <= float(row["injection_rate_tolerance"])
            for row in leakage["rate_aggregates"]
        )
        and _finite_metric_rows(leakage["seed_rows"], ("_probability", "_rate", "_occupancy", "_availability", "_fraction", "_steps", "_cycles")),
        "leakage_observed_only_boundary_and_monotone_burden_hold": all(
            row["truth_used_only_for_scoring"]
            and not any(
                "hidden" in key or "truth" in key
                for key in row["deployable_observation_fields"]
            )
            for row in leakage["seed_rows"]
        )
        and all(a < b for a, b in zip(leakage_occupancy, leakage_occupancy[1:])),
        "communication_patterns_are_unseen_and_exact": _canonical_sha256(
            communication["scenarios"]
        )
        == _canonical_sha256(expected_communication_scenarios),
        "communication_transitions_detected_in_every_seed": all(
            expected_comm_transitions.values()
        ),
        "communication_state_numeric_and_conflict_integrity_hold": len(communication_rows)
        == (len(COMMUNICATION_SCENARIOS) + 1) * 8
        and all(
            row["integrity"]["active_version_monotonic"]
            and row["integrity"]["maximum_version_step"] <= 1
            and row["integrity"]["all_arrays_finite"]
            and not row["integrity"]["slow_estimator_uses_hidden_truth"]
            and row["integrity"]["external_conflicting_updates_applied"] == 0
            for row in communication_rows
        ),
        "communication_impact_reported_against_paired_reference": all(
            "paired_availability_minus_reference" in row
            and "paired_ler_minus_reference" in row
            for row in communication["scenario_aggregates"]
            if row["scenario"] != "reference"
        ),
        "no_cross_lane_ranking_or_system_robustness_claim": not _forbidden_cross_lane_key(report)
        and report["system_robustness_status"] == "NOT_ESTABLISHED_LANE_LOCAL_ONLY",
        "all_ood_thresholds_and_scenarios_declared_selection_free": all(
            not lane["selection_used_ood_results"]
            for lane in (drift, measurement, leakage, communication)
        ),
    }


def _contract_view(report: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {
        "generated_at_utc",
        "contract_sha256",
        "source_data",
        "gate_summary",
    }
    return {key: value for key, value in report.items() if key not in excluded}


def validate_artifact(report: Mapping[str, Any]) -> tuple[str, ...]:
    errors: list[str] = []
    if report.get("task_id") != TASK_ID or report.get("schema_version") != SCHEMA_VERSION:
        errors.append("task/schema identity mismatch")
    if report.get("protocol_id") != PROTOCOL_ID:
        errors.append("protocol identity mismatch")
    if tuple(report.get("lane_ids", ())) != (
        "frozen_decoder_drift_ood",
        "sbs_measurement_confusion_ood",
        "persistent_leakage_rate_ood",
        "scheduler_communication_disturbance_ood",
    ):
        errors.append("lane order/membership mismatch")
    gates = report.get("gates")
    if not isinstance(gates, Mapping) or not gates or not all(value is True for value in gates.values()):
        errors.append("one or more gates failed")
    if report.get("status") != "PASS":
        errors.append("artifact status is not PASS")
    expected_contract = _canonical_sha256(_contract_view(report))
    if report.get("contract_sha256") != expected_contract:
        errors.append("contract hash mismatch")
    source = report.get("source_data")
    if not isinstance(source, Mapping) or int(source.get("row_count", -1)) <= 0:
        errors.append("source-data binding missing")
    if _forbidden_cross_lane_key(report):
        errors.append("forbidden cross-lane robustness/ranking field present")
    try:
        parents = load_parent_artifacts()
        recomputed_gates = _compute_gates(report, parents)
        if gates != recomputed_gates:
            errors.append("stored gates do not match recomputed evidence gates")
        config = report["config"]
        if _canonical_sha256(config) != _canonical_sha256(asdict(HeldOutOODConfig())):
            errors.append("pre-registered config drifted")
        if len(report["drift_lane"]["seed_rows"]) != 24:
            errors.append("drift row count mismatch")
        if len(report["measurement_confusion_lane"]["seed_rows"]) != 24:
            errors.append("measurement row count mismatch")
        if len(report["leakage_rate_lane"]["seed_rows"]) != 24:
            errors.append("leakage row count mismatch")
        if len(report["communication_lane"]["per_seed_results"]) != 32:
            errors.append("communication row count mismatch")
        if not all(
            row["confusion_rates_within_preregistered_tolerance"]
            for row in report["measurement_confusion_lane"]["scenario_aggregates"]
        ):
            errors.append("measurement confusion calibration mismatch")
        if not all(
            row["injection_rate_within_preregistered_tolerance"]
            for row in report["leakage_rate_lane"]["rate_aggregates"]
        ):
            errors.append("leakage injection calibration mismatch")
        for row in report["communication_lane"]["per_seed_results"]:
            if row["scenario"] != "reference" and (
                row["event_counts"].get("communication_pause_started", 0) <= 0
                or row["event_counts"].get("communication_pause_ended", 0) <= 0
            ):
                errors.append("communication transition evidence missing")
                break
        computed_source_rows = source_rows(report)
        if int(source["row_count"]) != len(computed_source_rows):
            errors.append("source-data row count mismatch")
        if source["rows_sha256"] != _canonical_sha256(computed_source_rows):
            errors.append("source-data canonical row hash mismatch")
        source_path = _repo_path(source["path"])
        if source_path.exists() and source.get("csv_sha256") != _sha256(source_path):
            errors.append("source-data CSV byte hash mismatch")
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"malformed artifact: {exc}")
    return tuple(errors)


def build_report(config: HeldOutOODConfig | None = None) -> dict[str, Any]:
    actual = HeldOutOODConfig() if config is None else config
    if not isinstance(actual, HeldOutOODConfig):
        raise TypeError("config must be HeldOutOODConfig")
    parents = load_parent_artifacts()
    bindings = parent_bindings(parents)
    drift = _run_drift_lane(parents["T5.1.2"], actual)
    measurement = _run_measurement_lane(actual)
    leakage = _run_leakage_lane(actual)
    communication = _run_communication_lane(actual)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "protocol_id": PROTOCOL_ID,
        "pass_semantics": (
            "all pre-registered OOD cells were executed with frozen parent parameters, "
            "disjoint seeds, lane-native observables and integrity gates; no universal "
            "performance or device-robustness claim is implied"
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(actual),
        "lane_ids": [
            drift["lane_id"],
            measurement["lane_id"],
            leakage["lane_id"],
            communication["lane_id"],
        ],
        "parent_bindings": bindings,
        "implementation_bindings": implementation_bindings(),
        "split_contract": {
            "registered_before_run": True,
            "ood_results_used_for_model_or_threshold_selection": False,
            "parent_seed_count": len(_parent_seed_union(parents)),
            "ood_seed_groups": {
                "drift": list(actual.drift_evaluation_seeds),
                "measurement_confusion": list(actual.measurement_evaluation_seeds),
                "leakage_rate": list(actual.leakage_evaluation_seeds),
                "communication": list(actual.communication_evaluation_seeds),
            },
            "lane_local_observables_only": True,
        },
        "drift_lane": drift,
        "measurement_confusion_lane": measurement,
        "leakage_rate_lane": leakage,
        "communication_lane": communication,
        "system_robustness_status": "NOT_ESTABLISHED_LANE_LOCAL_ONLY",
        "device_robustness_status": "NOT_ESTABLISHED_NO_TARGET_HARDWARE",
        "claim_boundary": {
            "allowed": (
                "pre-registered synthetic OOD coverage and lane-local degradation/safety "
                "diagnostics for frozen software/effective-model components"
            ),
            "forbidden": (
                "universal robustness, cross-lane leaderboard, experimental/device OOD "
                "robustness, or uncertainty-gated fallback benefit before T5.4.2"
            ),
        },
    }
    report["gates"] = _compute_gates(report, parents)
    report["gate_summary"] = {
        "passed": sum(bool(value) for value in report["gates"].values()),
        "total": len(report["gates"]),
    }
    report["status"] = "PASS" if all(report["gates"].values()) else "FAIL"
    report["source_data"] = {
        "path": DEFAULT_SOURCE_DATA.as_posix(),
        "row_count": len(source_rows(report)),
        "rows_sha256": _canonical_sha256(source_rows(report)),
        "csv_sha256": None,
    }
    report["contract_sha256"] = _canonical_sha256(_contract_view(report))
    return report


def source_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for binding in report["parent_bindings"]:
        rows.append(
            {
                "row_type": "parent_binding",
                "lane_id": "provenance",
                "record_id": binding["task_id"],
                "seed": "",
                "method": "",
                "metric": "machine_pass",
                "value": int(binding["machine_pass"]),
                "detail": json.dumps(binding, sort_keys=True),
            }
        )
    for row in report["drift_lane"]["seed_rows"]:
        for method in DECODER_METHODS:
            rows.append(
                {
                    "row_type": "drift_seed_method",
                    "lane_id": report["drift_lane"]["lane_id"],
                    "record_id": row["scenario_id"],
                    "seed": row["base_evaluation_seed"],
                    "method": method,
                    "metric": "error_rate",
                    "value": row[f"{method}_error_rate"],
                    "detail": json.dumps(row, sort_keys=True),
                }
            )
    for aggregate in report["drift_lane"]["scenario_aggregates"]:
        for method, metrics in aggregate["methods"].items():
            rows.append(
                {
                    "row_type": "drift_aggregate",
                    "lane_id": report["drift_lane"]["lane_id"],
                    "record_id": aggregate["scenario_id"],
                    "seed": "",
                    "method": method,
                    "metric": "error_rate_mean",
                    "value": metrics["error_rate_seed_cluster_ci"]["estimate"],
                    "detail": json.dumps(metrics, sort_keys=True),
                }
            )
    for row in report["measurement_confusion_lane"]["seed_rows"]:
        rows.append(
            {
                "row_type": "measurement_seed",
                "lane_id": report["measurement_confusion_lane"]["lane_id"],
                "record_id": row["scenario_id"],
                "seed": row["seed"],
                "method": "sbs_fault_overlay",
                "metric": "misclassification_rate",
                "value": row["misclassification_rate"],
                "detail": json.dumps(row, sort_keys=True),
            }
        )
    for row in report["measurement_confusion_lane"]["scenario_aggregates"]:
        rows.append(
            {
                "row_type": "measurement_aggregate",
                "lane_id": report["measurement_confusion_lane"]["lane_id"],
                "record_id": row["scenario_id"],
                "seed": "",
                "method": "sbs_fault_overlay",
                "metric": "confusion_rates_within_tolerance",
                "value": int(row["confusion_rates_within_preregistered_tolerance"]),
                "detail": json.dumps(row, sort_keys=True),
            }
        )
    for row in report["leakage_rate_lane"]["seed_rows"]:
        rows.append(
            {
                "row_type": "leakage_seed",
                "lane_id": report["leakage_rate_lane"]["lane_id"],
                "record_id": row["intervention_rate"],
                "seed": row["seed"],
                "method": "persistent_leakage_reset_kernel",
                "metric": "hidden_leakage_occupancy",
                "value": row["hidden_leakage_occupancy"],
                "detail": json.dumps(row, sort_keys=True),
            }
        )
    for row in report["leakage_rate_lane"]["rate_aggregates"]:
        rows.append(
            {
                "row_type": "leakage_aggregate",
                "lane_id": report["leakage_rate_lane"]["lane_id"],
                "record_id": row["intervention_rate"],
                "seed": "",
                "method": "persistent_leakage_reset_kernel",
                "metric": "injection_rate_within_tolerance",
                "value": int(row["injection_rate_within_preregistered_tolerance"]),
                "detail": json.dumps(row, sort_keys=True),
            }
        )
    for row in report["communication_lane"]["per_seed_results"]:
        rows.append(
            {
                "row_type": "communication_seed",
                "lane_id": report["communication_lane"]["lane_id"],
                "record_id": row["scenario"],
                "seed": row["seed"],
                "method": "dual_loop_scheduler",
                "metric": "end_to_end_control_availability",
                "value": row["metrics"]["end_to_end_control_availability"],
                "detail": json.dumps(row, sort_keys=True),
            }
        )
    for row in report["communication_lane"]["scenario_aggregates"]:
        rows.append(
            {
                "row_type": "communication_aggregate",
                "lane_id": report["communication_lane"]["lane_id"],
                "record_id": row["scenario"],
                "seed": "",
                "method": "dual_loop_scheduler",
                "metric": "availability_minus_reference",
                "value": row["paired_availability_minus_reference"]["mean"],
                "detail": json.dumps(row, sort_keys=True),
            }
        )
    for gate, passed in report["gates"].items():
        rows.append(
            {
                "row_type": "gate",
                "lane_id": "governance",
                "record_id": gate,
                "seed": "",
                "method": "",
                "metric": "passed",
                "value": int(passed),
                "detail": "",
            }
        )
    return rows


def write_report(
    report: Mapping[str, Any],
    *,
    artifact_path: str | Path = DEFAULT_ARTIFACT,
    source_path: str | Path = DEFAULT_SOURCE_DATA,
) -> dict[str, Any]:
    artifact = dict(report)
    rows = source_rows(artifact)
    csv_path = _repo_path(source_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("row_type", "lane_id", "record_id", "seed", "method", "metric", "value", "detail")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    artifact["source_data"] = {
        "path": Path(source_path).as_posix(),
        "row_count": len(rows),
        "rows_sha256": _canonical_sha256(rows),
        "csv_sha256": _sha256(csv_path),
    }
    artifact["contract_sha256"] = _canonical_sha256(_contract_view(artifact))
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("artifact validation failed: " + "; ".join(errors))
    path = _repo_path(artifact_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--source-data", type=Path, default=DEFAULT_SOURCE_DATA)
    args = parser.parse_args(argv)
    report = write_report(
        build_report(), artifact_path=args.artifact, source_path=args.source_data
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "gates": report["gate_summary"],
                "source_rows": report["source_data"]["row_count"],
                "artifact": str(args.artifact),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COMMUNICATION_EVALUATION_SEEDS",
    "COMMUNICATION_SCENARIOS",
    "DEFAULT_ARTIFACT",
    "DEFAULT_SOURCE_DATA",
    "DRIFT_EVALUATION_SEEDS",
    "DRIFT_SCENARIOS",
    "HeldOutOODConfig",
    "LEAKAGE_EVALUATION_SEEDS",
    "LEAKAGE_OOD_RATES",
    "MEASUREMENT_CONFUSION",
    "MEASUREMENT_EVALUATION_SEEDS",
    "MEASUREMENT_SCENARIOS",
    "OODDriftScenario",
    "build_report",
    "source_rows",
    "validate_artifact",
    "write_report",
]
