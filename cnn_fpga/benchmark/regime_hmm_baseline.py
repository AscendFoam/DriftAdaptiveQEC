"""T3.2.6 causal four-state Gaussian-HMM regime baseline validation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
import csv
import hashlib
import json
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np

from cnn_fpga.benchmark.continuous_adaptive_map import _mean_interval
from cnn_fpga.decoder.regime_hmm import (
    RAW_FEATURE_NAMES,
    REGIME_CLASSES,
    SUMMARY_FEATURE_NAMES,
    GaussianRegimeHMM,
    RegimeEstimatorBudget,
    RegimeObservationWindow,
    fit_supervised_gaussian_hmm,
    summarize_regime_window,
)
from physics.drift_processes import DriftState
from physics.syndrome_stream import SyndromeStreamConfig, generate_syndrome_stream


ROOT = Path(__file__).resolve().parents[2]
ESTIMATORS = ("static_prior", "memoryless_emission", "causal_hmm")


def _integer(value: object, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


@dataclass(frozen=True)
class RegimeHMMDescriptor:
    task_id: str = "T3.2.6"
    role: str = "host_regime_estimator_baseline"
    online_input: tuple[str, ...] = RAW_FEATURE_NAMES
    online_hidden_truth_input: tuple[str, ...] = ()
    training_only_labels: tuple[str, ...] = REGIME_CLASSES
    output: str = "normalized_normal_burst_leakage_calibration_shift_posterior"
    filtering: str = "strictly_causal_forward_recursion_no_viterbi_smoothing"
    future_cnn_fairness: str = "same_raw_window_shape_update_period_and_reserved_MAC_state_budget"
    logical_decoder: bool = False
    controller: bool = False
    hardware_measured: bool = False
    evidence_scope: str = "protocol_aligned_synthetic_window_level_regime_estimation"


REGIME_HMM_DESCRIPTOR = RegimeHMMDescriptor()


@dataclass(frozen=True)
class RegimeHMMValidationConfig:
    training_seeds: tuple[int, ...] = (20261201, 20261202, 20261203)
    validation_seeds: tuple[int, ...] = (20261211, 20261212, 20261213)
    evaluation_seeds: tuple[int, ...] = tuple(range(20261231, 20261239))
    windows_per_trajectory: int = 512
    budget: RegimeEstimatorBudget = RegimeEstimatorBudget()
    covariance_regularization_grid: tuple[float, ...] = (0.01, 0.03, 0.10, 0.30, 1.0, 3.0)
    transition_smoothing_grid: tuple[float, ...] = (0.05, 0.10, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0)
    temperature_grid: tuple[float, ...] = (0.30, 0.45, 0.60, 0.80, 1.0, 1.25, 1.60, 2.0, 3.0, 4.0)
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        groups = {
            "training_seeds": tuple(self.training_seeds),
            "validation_seeds": tuple(self.validation_seeds),
            "evaluation_seeds": tuple(self.evaluation_seeds),
        }
        for name, seeds in groups.items():
            minimum = 6 if name == "evaluation_seeds" else 3
            if len(seeds) < minimum or len(set(seeds)) != len(seeds):
                raise ValueError(f"{name} must contain at least {minimum} unique values")
            if any(
                isinstance(seed, bool)
                or not isinstance(seed, (int, np.integer))
                or int(seed) < 0
                or int(seed) >= 2**64 - 1_000_000
                for seed in seeds
            ):
                raise ValueError(f"{name} must contain nonnegative uint64-safe integers")
            object.__setattr__(self, name, tuple(int(seed) for seed in seeds))
        if any(set(left) & set(right) for left, right in ((groups["training_seeds"], groups["validation_seeds"]), (groups["training_seeds"], groups["evaluation_seeds"]), (groups["validation_seeds"], groups["evaluation_seeds"]))):
            raise ValueError("training, validation and evaluation seeds must be pairwise disjoint")
        object.__setattr__(
            self,
            "windows_per_trajectory",
            _integer(self.windows_per_trajectory, "windows_per_trajectory", 128),
        )
        if not isinstance(self.budget, RegimeEstimatorBudget):
            raise TypeError("budget must be RegimeEstimatorBudget")
        for name in (
            "covariance_regularization_grid",
            "transition_smoothing_grid",
            "temperature_grid",
        ):
            values = tuple(float(value) for value in getattr(self, name))
            if not values or len(set(values)) != len(values) or any(
                not isfinite(value) or value <= 0.0 for value in values
            ):
                raise ValueError(f"{name} must contain unique positive finite values")
            object.__setattr__(self, name, values)
        confidence = float(self.confidence_level)
        if not isfinite(confidence) or not 0.0 < confidence < 1.0:
            raise ValueError("confidence_level must lie in (0,1)")
        object.__setattr__(self, "confidence_level", confidence)
        cycles = (
            len(self.training_seeds)
            + len(self.validation_seeds)
            + len(self.evaluation_seeds)
        ) * self.windows_per_trajectory * self.budget.window_cycles
        if cycles > 500_000:
            raise ValueError("total simulator workload must not exceed 500,000 cycles")


@dataclass(frozen=True)
class RegimeTrajectory:
    base_seed: int
    windows: tuple[RegimeObservationWindow, ...]
    features: np.ndarray
    labels: tuple[str, ...]
    deployable_trace_sha256: str
    truth_trace_sha256: str


def _regime_schedule(windows: int, seed: int) -> tuple[str, ...]:
    rng = np.random.default_rng(seed + 17)
    labels: list[str] = []
    last: str | None = None
    forced = list(REGIME_CLASSES)
    rng.shuffle(forced)
    while len(labels) < windows:
        if forced:
            state = forced.pop(0)
        else:
            candidates = [name for name in REGIME_CLASSES if name != last]
            state = candidates[int(rng.integers(0, len(candidates)))]
        duration = int(rng.integers(8, 25))
        labels.extend([state] * min(duration, windows - len(labels)))
        last = state
    return tuple(labels)


def _state_for_cycle(label: str, cycle: int, seed: int, event_id: int) -> DriftState:
    common = {
        "step": cycle,
        "time": float(cycle),
        "source": "t3.2.6-four-regime-semi-markov",
        "regime": label,
        "seed": seed,
        "event_id": event_id,
    }
    if label == "normal":
        return DriftState(sigma_q=0.28, sigma_p=0.30, rho=0.08, loss_gamma=0.01, **common)
    if label == "burst":
        return DriftState(
            sigma_q=0.46,
            sigma_p=0.43,
            rho=0.38,
            loss_gamma=0.025,
            p_outlier=0.06,
            outlier_scale=3.0,
            burst_active=True,
            **common,
        )
    if label == "leakage":
        return DriftState(
            sigma_q=0.31,
            sigma_p=0.32,
            rho=0.05,
            loss_gamma=0.75,
            **common,
        )
    if label == "calibration_shift":
        return DriftState(
            mu_q=0.12,
            mu_p=-0.10,
            sigma_q=0.29,
            sigma_p=0.30,
            rho=-0.05,
            loss_gamma=0.015,
            **common,
        )
    raise ValueError(f"unknown regime {label!r}")


def _trajectory(seed: int, settings: RegimeHMMValidationConfig) -> RegimeTrajectory:
    labels = _regime_schedule(settings.windows_per_trajectory, seed)
    states = tuple(
        _state_for_cycle(
            labels[cycle // settings.budget.window_cycles],
            cycle,
            seed,
            cycle // settings.budget.window_cycles,
        )
        for cycle in range(settings.windows_per_trajectory * settings.budget.window_cycles)
    )
    stream = generate_syndrome_stream(
        states,
        config=SyndromeStreamConfig(
            measurement_sigma=(0.035, 0.035),
            depth_probability_scale=0.20,
            recovery_probability=0.72,
            base_leakage_probability=0.0005,
            loss_leakage_scale=0.12,
            burst_leakage_bonus=0.004,
            higher_leakage_fraction=0.75,
            higher_leakage_mean_duration=4.0,
            readout_fidelity_g=0.985,
            readout_fidelity_e=0.975,
            seed=seed,
        ),
    )
    health_rng = np.random.default_rng(seed + 999_983)
    window_records = []
    summaries = []
    deployable_digest = hashlib.sha256()
    truth_digest = hashlib.sha256()
    for window_index, label in enumerate(labels):
        start = window_index * settings.budget.window_cycles
        raw = np.empty((settings.budget.window_cycles, len(RAW_FEATURE_NAMES)), dtype=np.float64)
        for local, step in enumerate(stream.steps[start : start + settings.budget.window_cycles]):
            observed = step.observed
            raw[local] = (
                observed.residual_syndrome[0],
                observed.residual_syndrome[1],
                float(observed.syndrome.x == "e"),
                float(observed.syndrome.z == "e"),
                float("leakage" in observed.syndrome.as_tuple()),
                float((start + local) & 1),
                float(health_rng.random() >= 0.004),
                float(health_rng.random() >= 0.002),
            )
        window = RegimeObservationWindow(window_index, start, raw)
        summary = summarize_regime_window(window)
        window_records.append(window)
        summaries.append(summary)
        deployable_digest.update(np.asarray(raw, dtype="<f8").tobytes())
        truth_digest.update(label.encode("ascii"))
    feature_array = np.asarray(summaries, dtype=np.float64)
    feature_array.setflags(write=False)
    return RegimeTrajectory(
        base_seed=seed,
        windows=tuple(window_records),
        features=feature_array,
        labels=labels,
        deployable_trace_sha256=deployable_digest.hexdigest(),
        truth_trace_sha256=truth_digest.hexdigest(),
    )


def _label_indices(labels: Sequence[str]) -> np.ndarray:
    lookup = {name: index for index, name in enumerate(REGIME_CLASSES)}
    return np.asarray([lookup[label] for label in labels], dtype=np.int64)


def _nll(labels: Sequence[str], posterior: np.ndarray) -> float:
    truth = _label_indices(labels)
    probability = np.clip(posterior[np.arange(len(truth)), truth], 1.0e-300, 1.0)
    return float(-np.mean(np.log(probability)))


def _ece(labels: Sequence[str], posterior: np.ndarray, bins: int = 10) -> float:
    truth = _label_indices(labels)
    prediction = np.argmax(posterior, axis=1)
    confidence = np.max(posterior, axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        mask = (confidence > edges[index]) & (confidence <= edges[index + 1])
        if index == 0:
            mask |= confidence == 0.0
        if np.any(mask):
            accuracy = np.mean(prediction[mask] == truth[mask])
            result += np.mean(mask) * abs(float(accuracy - np.mean(confidence[mask])))
    return float(result)


def _transition_diagnostics(labels: Sequence[str], prediction: np.ndarray) -> tuple[float, float]:
    truth = _label_indices(labels)
    delays = []
    for start in range(1, len(truth)):
        if truth[start] == truth[start - 1]:
            continue
        end = start + 1
        while end < len(truth) and truth[end] == truth[start]:
            end += 1
        detection = next(
            (index for index in range(start, end) if prediction[index] == truth[start]),
            end,
        )
        delays.append(detection - start)
    false_switches = sum(
        prediction[index] != prediction[index - 1] and truth[index] == truth[index - 1]
        for index in range(1, len(truth))
    )
    stable_boundaries = max(1, sum(truth[index] == truth[index - 1] for index in range(1, len(truth))))
    return (float(np.mean(delays)) if delays else 0.0, false_switches / stable_boundaries)


def _metrics(labels: Sequence[str], posterior: np.ndarray) -> dict[str, object]:
    truth = _label_indices(labels)
    prediction = np.argmax(posterior, axis=1)
    one_hot = np.eye(len(REGIME_CLASSES))[truth]
    recalls = {}
    f1_values = []
    for index, name in enumerate(REGIME_CLASSES):
        true_positive = np.sum((prediction == index) & (truth == index))
        false_positive = np.sum((prediction == index) & (truth != index))
        false_negative = np.sum((prediction != index) & (truth == index))
        recall = true_positive / max(1, true_positive + false_negative)
        precision = true_positive / max(1, true_positive + false_positive)
        f1 = 2.0 * precision * recall / max(1.0e-15, precision + recall)
        recalls[name] = float(recall)
        f1_values.append(float(f1))
    delay, false_switch_rate = _transition_diagnostics(labels, prediction)
    return {
        "accuracy": float(np.mean(prediction == truth)),
        "macro_f1": float(np.mean(f1_values)),
        "negative_log_likelihood": _nll(labels, posterior),
        "brier_score": float(np.mean(np.sum((posterior - one_hot) ** 2, axis=1))),
        "expected_calibration_error": _ece(labels, posterior),
        "mean_transition_detection_delay_windows": delay,
        "false_switch_rate": float(false_switch_rate),
        "class_recall": recalls,
    }


def _temperature_selection(
    model: GaussianRegimeHMM,
    trajectories: Sequence[RegimeTrajectory],
    temperatures: Sequence[float],
    *,
    hmm: bool,
) -> tuple[float, list[dict[str, float]]]:
    rows = []
    for temperature in temperatures:
        values = []
        for trajectory in trajectories:
            posterior = (
                model.filter_sequence(trajectory.features, temperature=temperature)
                if hmm
                else model.memoryless_posterior(trajectory.features, temperature=temperature)
            )
            values.append(_nll(trajectory.labels, posterior))
        rows.append({"temperature": float(temperature), "validation_nll": float(np.mean(values))})
    selected = min(rows, key=lambda row: (row["validation_nll"], row["temperature"]))
    return float(selected["temperature"]), rows


def _select_model(
    training: Sequence[RegimeTrajectory],
    validation: Sequence[RegimeTrajectory],
    settings: RegimeHMMValidationConfig,
) -> tuple[GaussianRegimeHMM, float, float, list[dict[str, object]]]:
    feature_sequences = [trajectory.features for trajectory in training]
    label_sequences = [trajectory.labels for trajectory in training]
    grid_rows = []
    models: dict[tuple[float, float], GaussianRegimeHMM] = {}
    for regularization in settings.covariance_regularization_grid:
        for smoothing in settings.transition_smoothing_grid:
            model = fit_supervised_gaussian_hmm(
                feature_sequences,
                label_sequences,
                covariance_regularization=regularization,
                transition_smoothing=smoothing,
            )
            temperature, temperature_rows = _temperature_selection(
                model, validation, settings.temperature_grid, hmm=True
            )
            grid_rows.append(
                {
                    "covariance_regularization": regularization,
                    "transition_smoothing": smoothing,
                    "selected_hmm_temperature": temperature,
                    "validation_hmm_nll": min(row["validation_nll"] for row in temperature_rows),
                    "temperature_scan": temperature_rows,
                }
            )
            models[(regularization, smoothing)] = model
    selected = min(
        grid_rows,
        key=lambda row: (
            row["validation_hmm_nll"],
            row["covariance_regularization"],
            row["transition_smoothing"],
        ),
    )
    model = models[(selected["covariance_regularization"], selected["transition_smoothing"])]
    hmm_temperature = float(selected["selected_hmm_temperature"])
    memoryless_temperature, memoryless_scan = _temperature_selection(
        model, validation, settings.temperature_grid, hmm=False
    )
    selected["memoryless_temperature_scan_same_emissions"] = memoryless_scan
    selected["selected_memoryless_temperature"] = memoryless_temperature
    return model, hmm_temperature, memoryless_temperature, grid_rows


def _implementation_sha256() -> str:
    digest = hashlib.sha256()
    for relative in (
        "cnn_fpga/decoder/regime_hmm.py",
        "cnn_fpga/benchmark/regime_hmm_baseline.py",
        "physics/syndrome_stream.py",
    ):
        digest.update(relative.encode("utf-8"))
        digest.update((ROOT / relative).read_bytes())
    return digest.hexdigest()


def _profile_filter(model: GaussianRegimeHMM, features: np.ndarray) -> dict[str, float | int]:
    repeats = 20
    timings = []
    for _ in range(repeats):
        start = perf_counter()
        model.filter_sequence(features)
        timings.append(perf_counter() - start)
    return {
        "host_profile_repeats": repeats,
        "host_median_us_per_update": float(np.median(timings) * 1.0e6 / len(features)),
        "parameter_count_float_values": model.parameter_count,
        "float32_state_bytes_proxy": model.parameter_count * 4,
        "macs_per_update_proxy": model.macs_per_update_proxy,
    }


def build_regime_hmm_validation(
    config: RegimeHMMValidationConfig | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    settings = RegimeHMMValidationConfig() if config is None else config
    if not isinstance(settings, RegimeHMMValidationConfig):
        raise TypeError("config must be RegimeHMMValidationConfig")
    training = [_trajectory(seed, settings) for seed in settings.training_seeds]
    validation = [_trajectory(seed, settings) for seed in settings.validation_seeds]
    evaluation = [_trajectory(seed, settings) for seed in settings.evaluation_seeds]
    model, hmm_temperature, memoryless_temperature, selection_grid = _select_model(
        training, validation, settings
    )
    source_rows: list[dict[str, object]] = []
    seed_summaries: list[dict[str, object]] = []
    for trajectory in evaluation:
        posterior_by_estimator = {
            "static_prior": np.tile(model.class_prior_probabilities, (len(trajectory.labels), 1)),
            "memoryless_emission": model.memoryless_posterior(
                trajectory.features, temperature=memoryless_temperature
            ),
            "causal_hmm": model.filter_sequence(
                trajectory.features, temperature=hmm_temperature
            ),
        }
        summary = {"evaluation_seed": trajectory.base_seed}
        for estimator, posterior in posterior_by_estimator.items():
            summary[estimator] = _metrics(trajectory.labels, posterior)
        seed_summaries.append(summary)
        for window_index, label in enumerate(trajectory.labels):
            row: dict[str, object] = {
                "evaluation_seed": trajectory.base_seed,
                "window_index": window_index,
                "start_cycle": trajectory.windows[window_index].start_cycle,
                "end_cycle": trajectory.windows[window_index].end_cycle,
                "truth_regime": label,
                "deployable_trace_sha256": trajectory.deployable_trace_sha256,
                "truth_trace_sha256": trajectory.truth_trace_sha256,
            }
            for estimator, posterior in posterior_by_estimator.items():
                row[f"{estimator}_prediction"] = REGIME_CLASSES[int(np.argmax(posterior[window_index]))]
                for state_index, state in enumerate(REGIME_CLASSES):
                    row[f"{estimator}_p_{state}"] = float(posterior[window_index, state_index])
            source_rows.append(row)

    aggregate = {}
    for estimator in ESTIMATORS:
        aggregate[estimator] = {
            metric: float(np.mean([summary[estimator][metric] for summary in seed_summaries]))
            for metric in (
                "accuracy",
                "macro_f1",
                "negative_log_likelihood",
                "brier_score",
                "expected_calibration_error",
                "mean_transition_detection_delay_windows",
                "false_switch_rate",
            )
        }
        aggregate[estimator]["class_recall"] = {
            state: float(np.mean([summary[estimator]["class_recall"][state] for summary in seed_summaries]))
            for state in REGIME_CLASSES
        }
    nll_gain = [
        summary["memoryless_emission"]["negative_log_likelihood"]
        - summary["causal_hmm"]["negative_log_likelihood"]
        for summary in seed_summaries
    ]
    brier_gain = [
        summary["memoryless_emission"]["brier_score"]
        - summary["causal_hmm"]["brier_score"]
        for summary in seed_summaries
    ]
    accuracy_gain = [
        summary["causal_hmm"]["accuracy"] - summary["memoryless_emission"]["accuracy"]
        for summary in seed_summaries
    ]
    comparisons = {
        "memoryless_minus_hmm_nll": _mean_interval(nll_gain, settings.confidence_level),
        "memoryless_minus_hmm_brier": _mean_interval(brier_gain, settings.confidence_level),
        "hmm_minus_memoryless_accuracy": _mean_interval(accuracy_gain, settings.confidence_level),
    }
    profile = _profile_filter(model, evaluation[0].features)
    input_fields = {field.name for field in fields(RegimeObservationWindow)}
    grid_expected = len(settings.covariance_regularization_grid) * len(settings.transition_smoothing_grid)
    class_counts = {
        state: sum(trajectory.labels.count(state) for trajectory in evaluation)
        for state in REGIME_CLASSES
    }
    gates = {
        "train_validation_evaluation_seeds_disjoint": not (
            set(settings.training_seeds) & set(settings.validation_seeds)
            or set(settings.training_seeds) & set(settings.evaluation_seeds)
            or set(settings.validation_seeds) & set(settings.evaluation_seeds)
        ),
        "online_window_schema_has_no_truth_or_regime": (
            REGIME_HMM_DESCRIPTOR.online_hidden_truth_input == ()
            and not any("truth" in name or "regime" in name or "label" in name for name in input_fields)
        ),
        "selection_grid_complete_and_evaluation_blind": len(selection_grid) == grid_expected,
        "selected_hyperparameters_are_not_search_boundary": (
            min(settings.covariance_regularization_grid) < model.covariance_regularization < max(settings.covariance_regularization_grid)
            and min(settings.transition_smoothing_grid) < model.transition_smoothing < max(settings.transition_smoothing_grid)
            and min(settings.temperature_grid) < hmm_temperature < max(settings.temperature_grid)
            and min(settings.temperature_grid) < memoryless_temperature < max(settings.temperature_grid)
        ),
        "all_four_regimes_have_evaluation_support": min(class_counts.values()) >= 0.10 * len(source_rows),
        "all_posteriors_normalized": all(
            abs(sum(float(row[f"{estimator}_p_{state}"]) for state in REGIME_CLASSES) - 1.0) < 1.0e-10
            for row in source_rows
            for estimator in ESTIMATORS
        ),
        "source_rows_and_trace_hashes_complete": (
            len(source_rows) == len(settings.evaluation_seeds) * settings.windows_per_trajectory
            and len({row["deployable_trace_sha256"] for row in source_rows}) == len(settings.evaluation_seeds)
            and len({row["truth_trace_sha256"] for row in source_rows}) == len(settings.evaluation_seeds)
        ),
        "hmm_nll_improvement_over_same_emission_memoryless_resolved": comparisons["memoryless_minus_hmm_nll"]["ci_low"] > 0.0,
        "hmm_brier_improvement_over_same_emission_memoryless_resolved": comparisons["memoryless_minus_hmm_brier"]["ci_low"] > 0.0,
        "hmm_accuracy_not_worse_than_memoryless": comparisons["hmm_minus_memoryless_accuracy"]["ci_low"] >= 0.0,
        "every_hmm_class_recall_exceeds_half": min(aggregate["causal_hmm"]["class_recall"].values()) > 0.5,
        "causal_prefix_invariance": all(
            np.array_equal(
                model.filter_sequence(evaluation[0].features[:stop], temperature=hmm_temperature),
                model.filter_sequence(evaluation[0].features, temperature=hmm_temperature)[:stop],
            )
            for stop in (1, 17, 129, settings.windows_per_trajectory)
        ),
        "shared_future_cnn_budget_respected": (
            profile["macs_per_update_proxy"] <= settings.budget.max_macs_per_update
            and profile["float32_state_bytes_proxy"] <= settings.budget.max_float32_state_bytes
            and all(window.cycles == settings.budget.window_cycles for trajectory in evaluation for window in trajectory.windows)
        ),
        "host_profile_is_finite_but_not_hardware_claim": (
            isfinite(float(profile["host_median_us_per_update"]))
            and float(profile["host_median_us_per_update"]) > 0.0
            and not REGIME_HMM_DESCRIPTOR.hardware_measured
        ),
        "role_is_estimator_not_decoder_or_controller": (
            not REGIME_HMM_DESCRIPTOR.logical_decoder and not REGIME_HMM_DESCRIPTOR.controller
        ),
    }
    failed = [name for name, passed in gates.items() if not passed]
    payload: dict[str, object] = {
        "schema_version": "t3.2.6-causal-regime-hmm-v1",
        "task_id": "T3.2.6",
        "status": "PASS" if not failed else "FAIL",
        "implementation_sha256": _implementation_sha256(),
        "descriptor": asdict(REGIME_HMM_DESCRIPTOR),
        "validation_config": asdict(settings),
        "shared_input_budget": asdict(settings.budget),
        "training_selection": {
            "evaluation_truth_used": False,
            "training_trajectories": len(training),
            "validation_trajectories": len(validation),
            "grid": selection_grid,
            "selected_covariance_regularization": model.covariance_regularization,
            "selected_transition_smoothing": model.transition_smoothing,
            "selected_hmm_temperature": hmm_temperature,
            "selected_memoryless_temperature": memoryless_temperature,
        },
        "model": {
            "regime_classes": list(REGIME_CLASSES),
            "raw_feature_names": list(RAW_FEATURE_NAMES),
            "summary_feature_names": list(SUMMARY_FEATURE_NAMES),
            "transition_matrix": model.transition_matrix.tolist(),
            "initial_probabilities": model.initial_probabilities.tolist(),
            "class_prior_probabilities": model.class_prior_probabilities.tolist(),
            "emission_means_standardized": model.emission_means.tolist(),
            "emission_covariances_standardized": model.emission_covariances.tolist(),
            "profile": profile,
        },
        "evaluation": {
            "trajectories": len(evaluation),
            "windows": len(source_rows),
            "cycles": len(source_rows) * settings.budget.window_cycles,
            "class_counts": class_counts,
            "per_seed": seed_summaries,
            "aggregate": aggregate,
            "paired_seed_cluster_comparisons": comparisons,
            "source_data_rows": len(source_rows),
        },
        "gate_summary": {
            "passed": sum(bool(value) for value in gates.values()),
            "failed": len(failed),
            "gates": gates,
        },
        "claim_boundary": {
            "allowed": (
                "causal observed-window four-regime posterior estimation under the registered synthetic process, "
                "with same-emission memoryless ablation and a reserved future CNN input/operation budget"
            ),
            "forbidden": (
                "device-calibrated regime identification, logical-decoding or control gain, future-CNN measured "
                "latency parity, bit-accurate implementation, synthesis, FPGA latency, or quantum experiment"
            ),
        },
    }
    return payload, source_rows


def write_regime_hmm_validation(
    json_path: str | Path = "docs/t3_2_6_regime_hmm_validation.json",
    csv_path: str | Path = "docs/t3_2_6_regime_hmm_source_data.csv",
    config: RegimeHMMValidationConfig | None = None,
) -> dict[str, object]:
    payload, rows = build_regime_hmm_validation(config)
    json_target = Path(json_path)
    csv_target = Path(csv_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    csv_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not rows:
        raise RuntimeError("regime HMM validation produced no Source Data")
    with csv_target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--json", default="docs/t3_2_6_regime_hmm_validation.json")
    parser.add_argument("--csv", default="docs/t3_2_6_regime_hmm_source_data.csv")
    args = parser.parse_args(argv)
    config = RegimeHMMValidationConfig(windows_per_trajectory=128) if args.smoke else None
    payload = write_regime_hmm_validation(args.json, args.csv, config)
    print(json.dumps(payload["gate_summary"], ensure_ascii=False))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "REGIME_HMM_DESCRIPTOR",
    "RegimeHMMDescriptor",
    "RegimeHMMValidationConfig",
    "build_regime_hmm_validation",
    "write_regime_hmm_validation",
]
