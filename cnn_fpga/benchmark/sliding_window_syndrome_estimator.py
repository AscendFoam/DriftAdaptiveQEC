"""T3.2.3 causal sliding-window periodic-syndrome benchmark.

Window length is selected only on training traces.  Every candidate receives
384 new residuals and one update per benchmark window; longer candidates differ
only in retained history, not observation or update bandwidth.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import hashlib
import json
from math import isfinite, sqrt
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from cnn_fpga.benchmark.continuous_adaptive_map import (
    FULL_STATE_ORACLE_ID,
    WINDOW_PERIODIC_MAP_ID,
    ContinuousDriftScenario,
    _calibration_residuals,
    _estimate_score,
    _mean_interval,
    _parameter_errors,
    _residuals_and_truth,
    _scores,
    continuous_drift_scenarios,
)
from cnn_fpga.benchmark.static_map_baseline import (
    STATIC_MAP_ID,
    StaticMAPParameters,
    fit_static_map_from_training_states,
)
from cnn_fpga.decoder.periodic_adaptive_map import PeriodicMomentConfig, estimate_periodic_gaussian
from cnn_fpga.decoder.sliding_window_syndrome import (
    SlidingWindowConfig,
    SlidingWindowPeriodicEstimator,
    validate_window_candidates,
)
from physics.drift_processes import DriftState
from physics.ideal_gkp_decoder import map_decode_2d
from physics.oracle_map import oracle_map_2d


SLIDING_WINDOW_MAP_ID = "training_selected_sliding_window_periodic_map"
COMPARISON_ID = "t3_2_3_sliding_window_syndrome_comparison"


def _integer(value: object, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _finite(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class SlidingWindowDescriptor:
    task_id: str = "T3.2.3"
    comparison_id: str = COMPARISON_ID
    label: str = "Training-selected uniform sliding-window periodic syndrome estimator"
    consumed_observation_fields: tuple[str, ...] = ("residual_q", "residual_p")
    hidden_truth_inputs: tuple[str, ...] = ()
    update_timing: str = "one_window_delay_update_after_current_evaluation"
    budget_rule: str = "384_new_observations_and_one_update_per_window_for_every_candidate"
    state_rule: str = "incremental_add_remove_of_four_joint_circular_feature_sums"
    evidence_scope: str = "continuous_synthetic_wrapped_gaussian_syndrome_level"
    excluded_claims: tuple[str, ...] = (
        "universal_optimal_window",
        "loss_outlier_or_leakage_identification",
        "finite_energy_protocol_fidelity",
        "device_calibration",
        "FPGA_synthesis_or_measured_latency",
    )


SLIDING_DESCRIPTOR = SlidingWindowDescriptor()


@dataclass(frozen=True)
class SlidingWindowValidationConfig:
    training_seeds: tuple[int, ...] = (20260911, 20260912, 20260913)
    evaluation_seeds: tuple[int, ...] = tuple(range(20260931, 20260939))
    windows: int = 48
    observation_samples_per_window: int = 384
    training_score_samples_per_window: int = 384
    evaluation_samples_per_window: int = 1024
    feature_chunk_samples: int = 96
    window_sample_candidates: tuple[int, ...] = (384, 480, 576, 768, 1152, 1536)
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        training = tuple(self.training_seeds)
        evaluation = tuple(self.evaluation_seeds)
        if len(training) < 3 or len(set(training)) != len(training):
            raise ValueError("training_seeds must contain at least three unique seeds")
        if len(evaluation) < 6 or len(set(evaluation)) != len(evaluation):
            raise ValueError("evaluation_seeds must contain at least six unique seeds")
        if set(training) & set(evaluation):
            raise ValueError("training and evaluation seeds must be disjoint")
        if any(
            isinstance(seed, bool)
            or not isinstance(seed, (int, np.integer))
            or int(seed) < 0
            or int(seed) >= 2**64 - 100_000
            for seed in training + evaluation
        ):
            raise ValueError("seeds must be nonnegative uint64-safe integers")
        object.__setattr__(self, "training_seeds", tuple(int(seed) for seed in training))
        object.__setattr__(self, "evaluation_seeds", tuple(int(seed) for seed in evaluation))
        object.__setattr__(self, "windows", _integer(self.windows, "windows", 16))
        for name, minimum in (
            ("observation_samples_per_window", 128),
            ("training_score_samples_per_window", 128),
            ("evaluation_samples_per_window", 256),
            ("feature_chunk_samples", 32),
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum))
        candidates = validate_window_candidates(
            self.window_sample_candidates,
            update_stride_samples=self.observation_samples_per_window,
            feature_chunk_samples=self.feature_chunk_samples,
        )
        if candidates[0] != self.observation_samples_per_window:
            raise ValueError("the first candidate must be the latest-window anchor")
        object.__setattr__(self, "window_sample_candidates", candidates)
        confidence = _finite(self.confidence_level, "confidence_level")
        if not 0.5 < confidence < 1.0:
            raise ValueError("confidence_level must lie in (0.5,1)")
        object.__setattr__(self, "confidence_level", confidence)
        workload = len(evaluation) * len(continuous_drift_scenarios()) * self.windows * (
            self.observation_samples_per_window + self.evaluation_samples_per_window
        )
        if workload > 5_000_000:
            raise ValueError("evaluation workload must not exceed 5,000,000 samples")

    @property
    def calibration_samples(self) -> int:
        return max(self.window_sample_candidates)

    @property
    def calibration_windows(self) -> int:
        return self.calibration_samples // self.observation_samples_per_window


@dataclass(frozen=True)
class FrozenSlidingWindow:
    selected_window_samples: int
    candidate_scores: tuple[tuple[int, float], ...]
    training_trace_sha256: str
    selection_objective: str = "observation_only_independent_next_window_periodic_moment_score"


@dataclass(frozen=True)
class _TrainingTrace:
    calibration: NDArray[np.float64]
    updates: tuple[NDArray[np.float64], ...]
    scores: tuple[NDArray[np.float64], ...]
    trace_sha256: str


def _materialize_training_traces(settings: SlidingWindowValidationConfig) -> tuple[_TrainingTrace, ...]:
    traces: list[_TrainingTrace] = []
    for scenario_index, scenario in enumerate(continuous_drift_scenarios()):
        states = scenario.states(settings.windows)
        for base_seed in settings.training_seeds:
            seed = int(base_seed + 100_000 * scenario_index)
            rng = np.random.default_rng(seed)
            calibration = _calibration_residuals(
                states[0],
                settings.calibration_windows,
                settings.observation_samples_per_window,
                rng,
            )
            updates: list[NDArray[np.float64]] = []
            scores: list[NDArray[np.float64]] = []
            digest = hashlib.sha256()
            digest.update(scenario.scenario_id.encode("utf-8"))
            digest.update(seed.to_bytes(8, "little", signed=False))
            digest.update(np.asarray(calibration, dtype="<f8").tobytes())
            for state in states:
                update = _residuals_and_truth(state, settings.observation_samples_per_window, rng)[0]
                score = _residuals_and_truth(state, settings.training_score_samples_per_window, rng)[0]
                updates.append(update)
                scores.append(score)
                digest.update(np.asarray(update, dtype="<f8").tobytes())
                digest.update(np.asarray(score, dtype="<f8").tobytes())
            traces.append(
                _TrainingTrace(
                    calibration=calibration,
                    updates=tuple(updates),
                    scores=tuple(scores),
                    trace_sha256=digest.hexdigest(),
                )
            )
    return tuple(traces)


def select_frozen_window(
    config: SlidingWindowValidationConfig | None = None,
) -> FrozenSlidingWindow:
    settings = SlidingWindowValidationConfig() if config is None else config
    if not isinstance(settings, SlidingWindowValidationConfig):
        raise TypeError("config must be SlidingWindowValidationConfig")
    moment_config = PeriodicMomentConfig(minimum_samples=settings.feature_chunk_samples)
    traces = _materialize_training_traces(settings)
    candidate_scores: list[tuple[int, float]] = []
    for candidate in settings.window_sample_candidates:
        scores: list[float] = []
        for trace in traces:
            estimator = SlidingWindowPeriodicEstimator(
                trace.calibration,
                sliding_config=SlidingWindowConfig(
                    window_samples=candidate,
                    update_stride_samples=settings.observation_samples_per_window,
                    feature_chunk_samples=settings.feature_chunk_samples,
                ),
                moment_config=moment_config,
            )
            for window_id, (update, score_samples) in enumerate(zip(trace.updates, trace.scores)):
                target = estimate_periodic_gaussian(
                    score_samples,
                    moment_config,
                    source="training_scoring_observation",
                    window_id=window_id,
                )
                scores.append(_estimate_score(estimator.prediction(), target))
                estimator.update(update, window_id=window_id)
        candidate_scores.append((candidate, float(np.mean(scores))))
    selected = min(candidate_scores, key=lambda row: (row[1], row[0]))
    digest = hashlib.sha256()
    for trace in traces:
        digest.update(bytes.fromhex(trace.trace_sha256))
    return FrozenSlidingWindow(
        selected_window_samples=int(selected[0]),
        candidate_scores=tuple(candidate_scores),
        training_trace_sha256=digest.hexdigest(),
    )


def _fit_static_parameters(settings: SlidingWindowValidationConfig) -> StaticMAPParameters:
    states: list[DriftState] = []
    for scenario_index, scenario in enumerate(continuous_drift_scenarios()):
        for state in scenario.states(settings.windows):
            states.append(
                DriftState(
                    **{
                        **state.__dict__,
                        "step": len(states),
                        "time": float(len(states)),
                        "seed": settings.training_seeds[0] + scenario_index,
                    }
                )
            )
    return fit_static_map_from_training_states(
        tuple(states),
        training_protocol_id=(
            "t3.2.3-continuous-scenario-average-v1:"
            f"windows={settings.windows}:training_seeds={settings.training_seeds}"
        ),
    )


def validate_sliding_window_registration() -> tuple[str, ...]:
    from cnn_fpga.benchmark.standard_binning_baseline import (
        major_comparison_registry,
        validate_major_comparison_registry,
    )

    gates = validate_major_comparison_registry()
    matches = [entry for entry in major_comparison_registry() if entry.comparison_id == COMPARISON_ID]
    if len(matches) != 1:
        raise ValueError("T3.2.3 comparison must be registered exactly once")
    entry = matches[0]
    expected = (
        "standard_binning",
        STATIC_MAP_ID,
        WINDOW_PERIODIC_MAP_ID,
        SLIDING_WINDOW_MAP_ID,
        FULL_STATE_ORACLE_ID,
    )
    if entry.method_ids != expected:
        raise ValueError("T3.2.3 method order/roles drifted from the comparison contract")
    if entry.static_anchor_method_id != STATIC_MAP_ID:
        raise ValueError("T3.2.3 must use formal static MAP as static anchor")
    if entry.reference_anchor_method_id != FULL_STATE_ORACLE_ID:
        raise ValueError("T3.2.3 must use full-state model oracle as reference")
    return gates


def _implementation_sha256() -> str:
    paths = (
        Path(__file__),
        Path(__file__).parents[1] / "decoder" / "sliding_window_syndrome.py",
        Path(__file__).with_name("continuous_adaptive_map.py"),
        Path(__file__).with_name("standard_binning_baseline.py"),
        Path(__file__).with_name("static_map_baseline.py"),
        Path(__file__).parents[2] / "physics" / "ideal_gkp_decoder.py",
        Path(__file__).parents[2] / "physics" / "drift_processes.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _evaluate_seed(
    scenario: ContinuousDriftScenario,
    scenario_index: int,
    base_seed: int,
    settings: SlidingWindowValidationConfig,
    frozen: FrozenSlidingWindow,
    static_parameters: StaticMAPParameters,
    moment_config: PeriodicMomentConfig,
) -> dict[str, object]:
    seed = int(base_seed + 100_000 * scenario_index)
    rng = np.random.default_rng(seed)
    states = scenario.states(settings.windows)
    calibration = _calibration_residuals(
        states[0],
        settings.calibration_windows,
        settings.observation_samples_per_window,
        rng,
    )
    estimators = {
        candidate: SlidingWindowPeriodicEstimator(
            calibration,
            sliding_config=SlidingWindowConfig(
                window_samples=candidate,
                update_stride_samples=settings.observation_samples_per_window,
                feature_chunk_samples=settings.feature_chunk_samples,
            ),
            moment_config=moment_config,
        )
        for candidate in settings.window_sample_candidates
    }
    failures = {"standard": 0, "static": 0, "oracle": 0}
    failures.update({f"window_{candidate}": 0 for candidate in settings.window_sample_candidates})
    nll_sums = {"static": 0.0, "oracle": 0.0}
    brier_sums = {"static": 0.0, "oracle": 0.0}
    tracking = {candidate: [] for candidate in settings.window_sample_candidates}
    for candidate in settings.window_sample_candidates:
        nll_sums[f"window_{candidate}"] = 0.0
        brier_sums[f"window_{candidate}"] = 0.0
    digest = hashlib.sha256()
    digest.update(scenario.scenario_id.encode("utf-8"))
    digest.update(seed.to_bytes(8, "little", signed=False))
    digest.update(np.asarray(calibration, dtype="<f8").tobytes())
    total = 0
    for window_id, state in enumerate(states):
        predictions = {candidate: estimator.prediction() for candidate, estimator in estimators.items()}
        residual, truth, displacements = _residuals_and_truth(
            state, settings.evaluation_samples_per_window, rng
        )
        digest.update(window_id.to_bytes(4, "little", signed=False))
        digest.update(np.asarray(displacements, dtype="<f8").tobytes())
        total += truth.size
        failures["standard"] += int(np.sum(truth != 0))
        results = {
            "static": map_decode_2d(
                residual,
                static_parameters.covariance_array(),
                mean=static_parameters.mean_array(),
            ),
            "oracle": oracle_map_2d(residual, state),
        }
        for candidate, prediction in predictions.items():
            key = f"window_{candidate}"
            results[key] = map_decode_2d(
                residual,
                prediction.covariance_array(),
                mean=prediction.mean_array(),
            )
            tracking[candidate].append(_parameter_errors(prediction, state))
        for key, result in results.items():
            decision = np.asarray(result.logical_class, dtype=np.int64)
            failures[key] += int(np.sum(decision != truth))
            nll, brier = _scores(np.asarray(result.posterior), truth)
            nll_sums[key] += nll * truth.size
            brier_sums[key] += brier * truth.size
        observation = _residuals_and_truth(
            state, settings.observation_samples_per_window, rng
        )[0]
        digest.update(np.asarray(observation, dtype="<f8").tobytes())
        for estimator in estimators.values():
            estimator.update(observation, window_id=window_id)
    row: dict[str, object] = {
        "scenario_id": scenario.scenario_id,
        "base_evaluation_seed": int(base_seed),
        "evaluation_seed": seed,
        "windows": settings.windows,
        "observation_samples_per_window": settings.observation_samples_per_window,
        "updates_per_window": 1,
        "evaluation_samples": total,
        "trace_sha256": digest.hexdigest(),
        "standard_error_rate": failures["standard"] / total,
        "static_error_rate": failures["static"] / total,
        "oracle_error_rate": failures["oracle"] / total,
        "static_nll": nll_sums["static"] / total,
        "static_brier": brier_sums["static"] / total,
        "oracle_nll": nll_sums["oracle"] / total,
        "oracle_brier": brier_sums["oracle"] / total,
    }
    for candidate in settings.window_sample_candidates:
        key = f"window_{candidate}"
        values = np.asarray(tracking[candidate], dtype=np.float64)
        row[f"{key}_error_rate"] = failures[key] / total
        row[f"{key}_nll"] = nll_sums[key] / total
        row[f"{key}_brier"] = brier_sums[key] / total
        row[f"{key}_mean_tracking_rmse_lattice"] = float(sqrt(np.mean(np.square(values[:, 0]))))
        row[f"{key}_covariance_tracking_rmse_lattice2"] = float(
            sqrt(np.mean(np.square(values[:, 1])))
        )
    selected_key = f"window_{frozen.selected_window_samples}"
    anchor_key = f"window_{settings.observation_samples_per_window}"
    row["selected_window_samples"] = frozen.selected_window_samples
    row["selected_error_rate"] = row[f"{selected_key}_error_rate"]
    row["latest_window_error_rate"] = row[f"{anchor_key}_error_rate"]
    row["static_minus_selected_error_rate"] = float(row["static_error_rate"]) - float(
        row["selected_error_rate"]
    )
    row["latest_minus_selected_error_rate"] = float(row["latest_window_error_rate"]) - float(
        row["selected_error_rate"]
    )
    return row


def _scenario_summary(
    scenario_id: str,
    rows: Sequence[dict[str, object]],
    settings: SlidingWindowValidationConfig,
    frozen: FrozenSlidingWindow,
) -> dict[str, object]:
    selected = frozen.selected_window_samples
    anchor = settings.observation_samples_per_window
    result: dict[str, object] = {
        "scenario_id": scenario_id,
        "seeds": len(rows),
        "windows": settings.windows,
        "evaluation_samples": int(sum(int(row["evaluation_samples"]) for row in rows)),
        "unique_trace_hashes": len({str(row["trace_sha256"]) for row in rows}),
        "standard_error_rate": float(np.mean([row["standard_error_rate"] for row in rows])),
        "static_error_rate": float(np.mean([row["static_error_rate"] for row in rows])),
        "latest_window_error_rate": float(
            np.mean([row[f"window_{anchor}_error_rate"] for row in rows])
        ),
        "selected_window_samples": selected,
        "selected_error_rate": float(
            np.mean([row[f"window_{selected}_error_rate"] for row in rows])
        ),
        "oracle_error_rate": float(np.mean([row["oracle_error_rate"] for row in rows])),
        "static_minus_selected_seed_cluster_ci": _mean_interval(
            [float(row["static_minus_selected_error_rate"]) for row in rows],
            settings.confidence_level,
        ),
        "latest_minus_selected_seed_cluster_ci": _mean_interval(
            [float(row["latest_minus_selected_error_rate"]) for row in rows],
            settings.confidence_level,
        ),
        "candidate_error_rates": {
            str(candidate): float(np.mean([row[f"window_{candidate}_error_rate"] for row in rows]))
            for candidate in settings.window_sample_candidates
        },
        "candidate_mean_tracking_rmse_lattice": {
            str(candidate): float(
                np.mean([row[f"window_{candidate}_mean_tracking_rmse_lattice"] for row in rows])
            )
            for candidate in settings.window_sample_candidates
        },
        "candidate_covariance_tracking_rmse_lattice2": {
            str(candidate): float(
                np.mean(
                    [row[f"window_{candidate}_covariance_tracking_rmse_lattice2"] for row in rows]
                )
            )
            for candidate in settings.window_sample_candidates
        },
        "static_nll": float(np.mean([row["static_nll"] for row in rows])),
        "static_brier": float(np.mean([row["static_brier"] for row in rows])),
        "selected_nll": float(np.mean([row[f"window_{selected}_nll"] for row in rows])),
        "selected_brier": float(np.mean([row[f"window_{selected}_brier"] for row in rows])),
    }
    return result


def build_sliding_window_validation(
    config: SlidingWindowValidationConfig | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    settings = SlidingWindowValidationConfig() if config is None else config
    if not isinstance(settings, SlidingWindowValidationConfig):
        raise TypeError("config must be SlidingWindowValidationConfig")
    registry_gates = validate_sliding_window_registration()
    frozen = select_frozen_window(settings)
    static_parameters = _fit_static_parameters(settings)
    moment_config = PeriodicMomentConfig(minimum_samples=settings.feature_chunk_samples)
    rows: list[dict[str, object]] = []
    for scenario_index, scenario in enumerate(continuous_drift_scenarios()):
        for base_seed in settings.evaluation_seeds:
            rows.append(
                _evaluate_seed(
                    scenario,
                    scenario_index,
                    base_seed,
                    settings,
                    frozen,
                    static_parameters,
                    moment_config,
                )
            )
    scenario_summaries = [
        _scenario_summary(
            scenario.scenario_id,
            [row for row in rows if row["scenario_id"] == scenario.scenario_id],
            settings,
            frozen,
        )
        for scenario in continuous_drift_scenarios()
    ]
    seed_cluster_static = []
    seed_cluster_latest = []
    for seed in settings.evaluation_seeds:
        cluster = [row for row in rows if row["base_evaluation_seed"] == seed]
        seed_cluster_static.append(float(np.mean([row["static_minus_selected_error_rate"] for row in cluster])))
        seed_cluster_latest.append(float(np.mean([row["latest_minus_selected_error_rate"] for row in cluster])))
    costs = []
    for candidate in settings.window_sample_candidates:
        estimator = SlidingWindowPeriodicEstimator(
            np.zeros((settings.calibration_samples, 2), dtype=np.float64),
            sliding_config=SlidingWindowConfig(
                window_samples=candidate,
                update_stride_samples=settings.observation_samples_per_window,
                feature_chunk_samples=settings.feature_chunk_samples,
            ),
            moment_config=moment_config,
        )
        costs.append(asdict(estimator.cost_profile()) | {"stored_complex_values": estimator.cost_profile().stored_complex_values})
    static_gain = _mean_interval(seed_cluster_static, settings.confidence_level)
    latest_gain = _mean_interval(seed_cluster_latest, settings.confidence_level)
    selected_is_latest = frozen.selected_window_samples == settings.observation_samples_per_window
    aggregate_candidate_error_rates = {
        str(candidate): float(
            np.mean([float(row[f"window_{candidate}_error_rate"]) for row in rows])
        )
        for candidate in settings.window_sample_candidates
    }
    evaluation_best_window_samples = min(
        settings.window_sample_candidates,
        key=lambda candidate: (aggregate_candidate_error_rates[str(candidate)], candidate),
    )
    gates = {
        "training_and_evaluation_are_disjoint": not bool(
            set(settings.training_seeds) & set(settings.evaluation_seeds)
        ),
        "window_selected_before_evaluation": bool(frozen.training_trace_sha256),
        "comparison_roles_are_registered": len(registry_gates) > 0,
        "all_evaluation_traces_are_unique": len({row["trace_sha256"] for row in rows}) == len(rows),
        "candidate_matrix_has_latest_and_overlapping_windows": (
            len(settings.window_sample_candidates) >= 4
            and settings.window_sample_candidates[0] == settings.observation_samples_per_window
            and settings.window_sample_candidates[-1] > settings.observation_samples_per_window
        ),
        "observation_and_update_budget_is_identical": all(
            row["observation_samples_per_window"] == settings.observation_samples_per_window
            and row["updates_per_window"] == 1
            for row in rows
        ),
        "selected_alias_matches_candidate_column": all(
            row["selected_error_rate"]
            == row[f"window_{frozen.selected_window_samples}_error_rate"]
            for row in rows
        ),
        "selected_improves_static_in_every_scenario": all(
            summary["static_minus_selected_seed_cluster_ci"]["ci_low"] > 0.0
            for summary in scenario_summaries
        ),
        "selected_proper_scores_improve_static": all(
            summary["selected_nll"] < summary["static_nll"]
            and summary["selected_brier"] < summary["static_brier"]
            for summary in scenario_summaries
        ),
        "aggregate_static_gain_resolved": static_gain["ci_low"] > 0.0,
        "oracle_remains_strict_reference": all(
            summary["oracle_error_rate"] < summary["selected_error_rate"]
            for summary in scenario_summaries
        ),
        "storage_proxy_is_monotone": all(
            costs[index]["stored_complex_values"] < costs[index + 1]["stored_complex_values"]
            for index in range(len(costs) - 1)
        ),
        "hardware_fields_remain_unmeasured": all(
            cost["target_lut"] is None
            and cost["target_ff"] is None
            and cost["target_bram"] is None
            and cost["target_dsp"] is None
            and cost["target_fmax_hz"] is None
            and cost["target_measured"] is False
            for cost in costs
        ),
        "latest_comparison_is_complete_without_forced_win": (
            all(
                isfinite(float(latest_gain[key]))
                for key in ("estimate", "standard_error", "ci_low", "ci_high")
            )
            and (
                not selected_is_latest
                or abs(float(latest_gain["estimate"])) < 1.0e-15
            )
        ),
    }
    failed = [name for name, passed in gates.items() if not passed]
    payload: dict[str, object] = {
        "schema_version": "t3.2.3-sliding-window-syndrome-v1",
        "task_id": "T3.2.3",
        "status": "PASS" if not failed else "FAIL",
        "implementation_sha256": _implementation_sha256(),
        "descriptor": asdict(SLIDING_DESCRIPTOR),
        "validation_config": asdict(settings),
        "frozen_window_selection": asdict(frozen),
        "static_training_parameters": asdict(static_parameters),
        "scenarios": scenario_summaries,
        "aggregate": {
            "scenarios": len(scenario_summaries),
            "evaluation_seeds_per_scenario": len(settings.evaluation_seeds),
            "windows": len(rows) * settings.windows,
            "evaluation_samples": int(sum(int(row["evaluation_samples"]) for row in rows)),
            "source_data_rows": len(rows),
            "candidate_error_rates": aggregate_candidate_error_rates,
            "evaluation_best_window_samples_diagnostic_only": evaluation_best_window_samples,
            "static_minus_selected_seed_cluster_ci": static_gain,
            "latest_minus_selected_seed_cluster_ci": latest_gain,
        },
        "cost_profiles": costs,
        "gate_summary": {
            "passed": sum(bool(value) for value in gates.values()),
            "failed": len(failed),
            "gates": gates,
        },
        "claim_boundary": {
            "selection_result": (
                "latest_window_selected" if selected_is_latest else "longer_overlapping_window_selected"
            ),
            "evaluation_best_is_diagnostic_not_a_selector": True,
            "latest_comparison_resolved": bool(latest_gain["ci_low"] > 0.0),
            "allowed": (
                "training-selected uniform sliding-window baseline on the registered continuous "
                "wrapped-Gaussian scenarios with the frozen observation/update budget"
            ),
            "forbidden": (
                "universal optimal window, evaluation-tuned window, CNN superiority, loss/outlier/"
                "leakage identification, device calibration, or FPGA synthesis/measurement"
            ),
        },
    }
    return payload, rows


def write_sliding_window_validation(
    json_path: str | Path = "docs/t3_2_3_sliding_window_validation.json",
    csv_path: str | Path = "docs/t3_2_3_sliding_window_source_data.csv",
    config: SlidingWindowValidationConfig | None = None,
) -> dict[str, object]:
    payload, rows = build_sliding_window_validation(config)
    json_target = Path(json_path)
    csv_target = Path(csv_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    csv_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not rows:
        raise RuntimeError("sliding-window validation produced no source rows")
    with csv_target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return payload


def main() -> int:
    payload = write_sliding_window_validation()
    summary = payload["gate_summary"]
    print(
        json.dumps(
            {
                "passed": summary["passed"],
                "failed": summary["failed"],
                "selected_window_samples": payload["frozen_window_selection"]["selected_window_samples"],
                "gates": summary["gates"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SLIDING_WINDOW_MAP_ID",
    "COMPARISON_ID",
    "SlidingWindowDescriptor",
    "SLIDING_DESCRIPTOR",
    "SlidingWindowValidationConfig",
    "FrozenSlidingWindow",
    "select_frozen_window",
    "validate_sliding_window_registration",
    "build_sliding_window_validation",
    "write_sliding_window_validation",
]
