"""T4.1.1 matched-budget slow-loop model-family selection.

The comparison deliberately reuses the T3.2.6 observed-window regime task so
that every family sees identical causal information, history, split and metric.
It is a host-software/synthetic selection study, not an FPGA latency result.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import csv
import hashlib
import json
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Sequence

import numpy as np

from cnn_fpga.benchmark.continuous_adaptive_map import _mean_interval
from cnn_fpga.benchmark.regime_hmm_baseline import (
    RegimeHMMValidationConfig,
    RegimeTrajectory,
    _metrics,
    _nll,
    _trajectory,
)
from cnn_fpga.decoder.regime_hmm import (
    REGIME_CLASSES,
    GaussianRegimeHMM,
    fit_supervised_gaussian_hmm,
)
from cnn_fpga.decoder.slow_loop_model_selection import (
    MODEL_FAMILIES,
    CausalTCN,
    DiagonalGaussianHead,
    FeatureStandardizer,
    SlowLoopSelectionBudget,
    SmallGRU,
    bounded_histories,
    diagonal_kalman_states,
    exponential_states,
    labels_for_histories,
    RollingGaussianHMMAdapter,
    resource_profiles,
    run_length_fsm_posterior,
    softmax_logits,
    temper_posterior,
)

try:
    import torch
except ImportError:  # pragma: no cover - production requires DLEnv.
    torch = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class SlowLoopSelectionDescriptor:
    task_id: str = "T4.1.1"
    role: str = "matched_budget_host_slow_loop_regime_model_selection"
    online_input: str = "last_8_canonical_T3.2.6_observed_32_cycle_summary_windows"
    online_hidden_truth_input: tuple[str, ...] = ()
    output: str = "normal_burst_leakage_calibration_shift_posterior"
    model_family_prior: str = "none_validation_NLL_lexicographic_selection"
    evaluation_used_for_selection: bool = False
    controller: bool = False
    logical_decoder: bool = False
    hardware_measured: bool = False


DESCRIPTOR = SlowLoopSelectionDescriptor()


@dataclass(frozen=True)
class SlowLoopSelectionConfig:
    training_seeds: tuple[int, ...] = (20261201, 20261202, 20261203)
    validation_seeds: tuple[int, ...] = (20261211, 20261212, 20261213)
    evaluation_seeds: tuple[int, ...] = tuple(range(20261231, 20261239))
    windows_per_trajectory: int = 512
    budget: SlowLoopSelectionBudget = SlowLoopSelectionBudget()
    neural_restarts: tuple[int, ...] = (41101, 41102, 41103, 41104, 41105)
    neural_epochs: int = 240
    neural_patience: int = 36
    neural_batch_size: int = 128
    neural_learning_rate: float = 0.003
    neural_weight_decay: float = 0.0001
    temperature_grid: tuple[float, ...] = (0.45, 0.65, 0.85, 1.0, 1.25, 1.6, 2.0, 3.0)
    hmm_covariance_grid: tuple[float, ...] = (0.01, 0.03, 0.10, 0.30, 1.0)
    hmm_transition_grid: tuple[float, ...] = (0.10, 0.25, 0.5, 1.0, 5.0, 30.0)
    diagonal_variance_floor_grid: tuple[float, ...] = (0.01, 0.03, 0.10, 0.30)
    recurrence_decay_grid: tuple[float, ...] = (0.0, 0.25, 0.50, 0.70, 0.85, 0.93)
    kalman_process_grid: tuple[float, ...] = (0.01, 0.03, 0.10, 0.30)
    kalman_measurement_grid: tuple[float, ...] = (0.30, 1.0, 3.0)
    fsm_enter_run_grid: tuple[int, ...] = (1, 2, 3, 4)
    fsm_confidence_grid: tuple[float, ...] = (0.55, 0.70, 0.85)
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
                raise ValueError(f"{name} must contain at least {minimum} unique seeds")
            if any(isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or int(seed) < 0 for seed in seeds):
                raise ValueError(f"{name} must contain non-negative integers")
            object.__setattr__(self, name, tuple(int(seed) for seed in seeds))
        if any(
            set(left) & set(right)
            for left, right in (
                (groups["training_seeds"], groups["validation_seeds"]),
                (groups["training_seeds"], groups["evaluation_seeds"]),
                (groups["validation_seeds"], groups["evaluation_seeds"]),
            )
        ):
            raise ValueError("training, validation and evaluation seeds must be pairwise disjoint")
        if isinstance(self.windows_per_trajectory, bool) or int(self.windows_per_trajectory) < 128:
            raise ValueError("windows_per_trajectory must be at least 128")
        object.__setattr__(self, "windows_per_trajectory", int(self.windows_per_trajectory))
        if not isinstance(self.budget, SlowLoopSelectionBudget):
            raise TypeError("budget must be SlowLoopSelectionBudget")
        for name in ("neural_epochs", "neural_patience", "neural_batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 1:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        if self.neural_patience >= self.neural_epochs:
            raise ValueError("neural_patience must be smaller than neural_epochs")
        restarts = tuple(self.neural_restarts)
        if len(restarts) < 2 or len(set(restarts)) != len(restarts) or any(
            isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) for seed in restarts
        ):
            raise ValueError("neural_restarts must contain at least two unique integer seeds")
        object.__setattr__(self, "neural_restarts", tuple(int(seed) for seed in restarts))
        for name in ("neural_learning_rate", "neural_weight_decay", "confidence_level"):
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        if self.neural_weight_decay >= 1.0:
            raise ValueError("neural_weight_decay must be below one")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must lie in (0,1)")
        for name in (
            "temperature_grid",
            "hmm_covariance_grid",
            "hmm_transition_grid",
            "diagonal_variance_floor_grid",
            "kalman_process_grid",
            "kalman_measurement_grid",
            "fsm_confidence_grid",
        ):
            values = tuple(float(value) for value in getattr(self, name))
            if not values or len(set(values)) != len(values) or any(not isfinite(value) or value <= 0.0 for value in values):
                raise ValueError(f"{name} must contain unique positive finite values")
            object.__setattr__(self, name, values)
        decays = tuple(float(value) for value in self.recurrence_decay_grid)
        if not decays or len(set(decays)) != len(decays) or any(not 0.0 <= value < 1.0 for value in decays):
            raise ValueError("recurrence_decay_grid must contain unique values in [0,1)")
        object.__setattr__(self, "recurrence_decay_grid", decays)
        runs = tuple(self.fsm_enter_run_grid)
        if not runs or len(set(runs)) != len(runs) or any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 1 for value in runs
        ):
            raise ValueError("fsm_enter_run_grid must contain unique positive integers")
        object.__setattr__(self, "fsm_enter_run_grid", tuple(int(value) for value in runs))
        if any(not 0.25 < value < 1.0 for value in self.fsm_confidence_grid):
            raise ValueError("fsm confidence must lie between uniform and one")


@dataclass(frozen=True)
class HistoryDataset:
    raw_histories: np.ndarray
    standardized_histories: np.ndarray
    labels: np.ndarray
    sequence_indices: np.ndarray
    local_indices: np.ndarray


@dataclass
class FittedCandidate:
    family: str
    predict: Callable[[np.ndarray, np.ndarray], np.ndarray]
    validation_posterior: np.ndarray
    validation_metrics: dict[str, object]
    selected_hyperparameters: dict[str, object]
    selection_scan: list[dict[str, object]]
    checkpoint: dict[str, object]
    online_profile: Callable[[np.ndarray], np.ndarray] | None = None


def _dataset(
    trajectories: Sequence[RegimeTrajectory],
    standardizer: FeatureStandardizer,
    history_windows: int,
) -> HistoryDataset:
    raw, sequence_indices, local_indices = bounded_histories(
        [trajectory.features for trajectory in trajectories], history_windows=history_windows
    )
    labels = labels_for_histories(
        [trajectory.labels for trajectory in trajectories], sequence_indices, local_indices
    )
    return HistoryDataset(
        raw_histories=raw,
        standardized_histories=standardizer.transform(raw),
        labels=labels,
        sequence_indices=sequence_indices,
        local_indices=local_indices,
    )


def _labels_as_names(labels: np.ndarray) -> tuple[str, ...]:
    return tuple(REGIME_CLASSES[int(index)] for index in labels)


def _select_temperature(
    logits_or_posterior: np.ndarray,
    labels: np.ndarray,
    temperatures: Sequence[float],
    *,
    already_posterior: bool,
) -> tuple[float, np.ndarray, list[dict[str, float]]]:
    rows = []
    predictions: dict[float, np.ndarray] = {}
    names = _labels_as_names(labels)
    for temperature in temperatures:
        posterior = (
            temper_posterior(logits_or_posterior, temperature)
            if already_posterior
            else softmax_logits(logits_or_posterior, temperature)
        )
        predictions[float(temperature)] = posterior
        rows.append({"temperature": float(temperature), "validation_nll": _nll(names, posterior)})
    selected = min(rows, key=lambda row: (row["validation_nll"], row["temperature"]))
    temperature = float(selected["temperature"])
    return temperature, predictions[temperature], rows


def _state_dict_cpu(model: Any) -> dict[str, Any]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _fit_neural(
    family: str,
    training: HistoryDataset,
    validation: HistoryDataset,
    settings: SlowLoopSelectionConfig,
) -> FittedCandidate:
    if torch is None or CausalTCN is None or SmallGRU is None:
        raise RuntimeError("T4.1.1 neural training requires torch in DLEnv")
    torch.set_num_threads(1)
    factory = CausalTCN if family == "causal_tcn" else SmallGRU
    train_x = torch.as_tensor(training.standardized_histories, dtype=torch.float32)
    train_y = torch.as_tensor(training.labels, dtype=torch.long)
    validation_x = torch.as_tensor(validation.standardized_histories, dtype=torch.float32)
    records: list[dict[str, object]] = []
    snapshots: dict[int, dict[str, Any]] = {}
    validation_logits: dict[int, np.ndarray] = {}
    for restart in settings.neural_restarts:
        torch.manual_seed(restart)
        np_rng = np.random.default_rng(restart)
        model = factory()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=settings.neural_learning_rate,
            weight_decay=settings.neural_weight_decay,
        )
        best_loss = float("inf")
        best_epoch = 0
        best_state: dict[str, Any] | None = None
        stale = 0
        for epoch in range(1, settings.neural_epochs + 1):
            model.train()
            order = np_rng.permutation(len(train_x))
            for start in range(0, len(order), settings.neural_batch_size):
                index = torch.as_tensor(order[start : start + settings.neural_batch_size], dtype=torch.long)
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(model(train_x[index]), train_y[index])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            model.eval()
            with torch.no_grad():
                val_logits = model(validation_x)
                val_loss = float(torch.nn.functional.cross_entropy(val_logits, torch.as_tensor(validation.labels)).item())
            if val_loss < best_loss - 1.0e-7:
                best_loss = val_loss
                best_epoch = epoch
                best_state = _state_dict_cpu(model)
                stale = 0
            else:
                stale += 1
            if stale >= settings.neural_patience:
                break
        if best_state is None:
            raise RuntimeError("neural restart produced no checkpoint")
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            logits = model(validation_x).cpu().numpy().astype(np.float64)
        temperature, posterior, scan = _select_temperature(
            logits, validation.labels, settings.temperature_grid, already_posterior=False
        )
        snapshots[restart] = best_state
        validation_logits[restart] = logits
        records.append(
            {
                "restart_seed": restart,
                "best_epoch": best_epoch,
                "epochs_executed": epoch,
                "best_raw_validation_cross_entropy": best_loss,
                "selected_temperature": temperature,
                "calibrated_validation_nll": _nll(_labels_as_names(validation.labels), posterior),
                "temperature_scan": scan,
            }
        )
    selected = min(
        records,
        key=lambda row: (
            row["calibrated_validation_nll"],
            row["best_raw_validation_cross_entropy"],
            row["restart_seed"],
        ),
    )
    selected_seed = int(selected["restart_seed"])
    selected_temperature = float(selected["selected_temperature"])
    model = factory()
    model.load_state_dict(snapshots[selected_seed])
    model.eval()

    def predict(_raw: np.ndarray, standardized: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            logits = model(torch.as_tensor(standardized, dtype=torch.float32)).cpu().numpy()
        return softmax_logits(logits, selected_temperature)

    posterior = softmax_logits(validation_logits[selected_seed], selected_temperature)
    return FittedCandidate(
        family=family,
        predict=predict,
        validation_posterior=posterior,
        validation_metrics=_metrics(_labels_as_names(validation.labels), posterior),
        selected_hyperparameters={
            "restart_seed": selected_seed,
            "temperature": selected_temperature,
            "architecture": "14x7-k3-d1/k3-d2" if family == "causal_tcn" else "14x5-GRU",
        },
        selection_scan=records,
        checkpoint={
            "family": family,
            "selected_restart_seed": selected_seed,
            "temperature": selected_temperature,
            "state_dict": snapshots[selected_seed],
            "restart_records": records,
        },
    )


def _bounded_hmm_posterior(model: GaussianRegimeHMM, histories: np.ndarray, temperature: float) -> np.ndarray:
    return np.vstack(
        [model.filter_sequence(history, temperature=temperature)[-1] for history in histories]
    )


def _serialize_hmm(model: GaussianRegimeHMM) -> dict[str, object]:
    return {
        name: np.asarray(getattr(model, name)).tolist()
        for name in (
            "standardization_mean",
            "standardization_scale",
            "emission_means",
            "emission_covariances",
            "emission_precisions",
            "emission_log_determinants",
            "transition_matrix",
            "initial_probabilities",
            "class_prior_probabilities",
        )
    } | {
        "covariance_regularization": model.covariance_regularization,
        "transition_smoothing": model.transition_smoothing,
    }


def _fit_hmm(
    trajectories: Sequence[RegimeTrajectory],
    validation: HistoryDataset,
    settings: SlowLoopSelectionConfig,
) -> FittedCandidate:
    rows: list[dict[str, object]] = []
    models: dict[tuple[float, float], GaussianRegimeHMM] = {}
    for regularization in settings.hmm_covariance_grid:
        for smoothing in settings.hmm_transition_grid:
            model = fit_supervised_gaussian_hmm(
                [trajectory.features for trajectory in trajectories],
                [trajectory.labels for trajectory in trajectories],
                covariance_regularization=regularization,
                transition_smoothing=smoothing,
            )
            raw = _bounded_hmm_posterior(model, validation.raw_histories, 1.0)
            temperature, posterior, scan = _select_temperature(
                raw, validation.labels, settings.temperature_grid, already_posterior=True
            )
            rows.append(
                {
                    "covariance_regularization": regularization,
                    "transition_smoothing": smoothing,
                    "temperature": temperature,
                    "validation_nll": _nll(_labels_as_names(validation.labels), posterior),
                    "temperature_scan": scan,
                }
            )
            models[(regularization, smoothing)] = model
    selected = min(
        rows,
        key=lambda row: (
            row["validation_nll"], row["covariance_regularization"], row["transition_smoothing"]
        ),
    )
    model = models[(float(selected["covariance_regularization"]), float(selected["transition_smoothing"]))]
    temperature = float(selected["temperature"])

    def predict(raw: np.ndarray, _standardized: np.ndarray) -> np.ndarray:
        return _bounded_hmm_posterior(model, raw, temperature)

    def online_profile(features: np.ndarray) -> np.ndarray:
        adapter = RollingGaussianHMMAdapter(
            model,
            history_windows=settings.budget.history_windows,
            temperature=temperature,
        )
        return np.vstack([adapter.step(row) for row in features])

    posterior = predict(validation.raw_histories, validation.standardized_histories)
    return FittedCandidate(
        family="gaussian_hmm",
        predict=predict,
        validation_posterior=posterior,
        validation_metrics=_metrics(_labels_as_names(validation.labels), posterior),
        selected_hyperparameters={key: selected[key] for key in ("covariance_regularization", "transition_smoothing", "temperature")},
        selection_scan=rows,
        checkpoint={"family": "gaussian_hmm", "model": _serialize_hmm(model), "temperature": temperature},
        online_profile=online_profile,
    )


def _serialize_head(head: DiagonalGaussianHead) -> dict[str, object]:
    return {"means": head.means.tolist(), "variances": head.variances.tolist(), "priors": head.priors.tolist()}


def _fit_filtered_diagonal(
    family: str,
    training: HistoryDataset,
    validation: HistoryDataset,
    settings: SlowLoopSelectionConfig,
) -> FittedCandidate:
    rows: list[dict[str, object]] = []
    candidates: dict[tuple[float, ...], tuple[DiagonalGaussianHead, dict[str, float], np.ndarray]] = {}
    if family == "exponential_recurrence":
        dynamics = [(decay,) for decay in settings.recurrence_decay_grid]
    else:
        dynamics = [
            (process, measurement)
            for process in settings.kalman_process_grid
            for measurement in settings.kalman_measurement_grid
        ]
    for dynamic in dynamics:
        train_states = (
            exponential_states(training.standardized_histories, dynamic[0])
            if family == "exponential_recurrence"
            else diagonal_kalman_states(training.standardized_histories, dynamic[0], dynamic[1])
        )
        validation_states = (
            exponential_states(validation.standardized_histories, dynamic[0])
            if family == "exponential_recurrence"
            else diagonal_kalman_states(validation.standardized_histories, dynamic[0], dynamic[1])
        )
        for floor in settings.diagonal_variance_floor_grid:
            head = DiagonalGaussianHead.fit(train_states, training.labels, variance_floor=floor)
            logits = head.logits(validation_states)
            temperature, posterior, scan = _select_temperature(
                logits, validation.labels, settings.temperature_grid, already_posterior=False
            )
            parameters = (
                {"decay": dynamic[0]}
                if family == "exponential_recurrence"
                else {"process_variance": dynamic[0], "measurement_variance": dynamic[1]}
            )
            parameters |= {"variance_floor": floor, "temperature": temperature}
            key = tuple(dynamic) + (floor,)
            rows.append(parameters | {"validation_nll": _nll(_labels_as_names(validation.labels), posterior), "temperature_scan": scan})
            candidates[key] = (head, parameters, logits)
    selected = min(rows, key=lambda row: tuple([row["validation_nll"]] + [row[key] for key in sorted(row) if key not in {"validation_nll", "temperature_scan"}]))
    if family == "exponential_recurrence":
        key = (float(selected["decay"]), float(selected["variance_floor"]))
    else:
        key = (
            float(selected["process_variance"]),
            float(selected["measurement_variance"]),
            float(selected["variance_floor"]),
        )
    head, parameters, _ = candidates[key]
    temperature = float(parameters["temperature"])

    def predict(_raw: np.ndarray, standardized: np.ndarray) -> np.ndarray:
        states = (
            exponential_states(standardized, float(parameters["decay"]))
            if family == "exponential_recurrence"
            else diagonal_kalman_states(
                standardized,
                float(parameters["process_variance"]),
                float(parameters["measurement_variance"]),
            )
        )
        return softmax_logits(head.logits(states), temperature)

    posterior = predict(validation.raw_histories, validation.standardized_histories)
    return FittedCandidate(
        family=family,
        predict=predict,
        validation_posterior=posterior,
        validation_metrics=_metrics(_labels_as_names(validation.labels), posterior),
        selected_hyperparameters=dict(parameters),
        selection_scan=rows,
        checkpoint={"family": family, "head": _serialize_head(head), "hyperparameters": dict(parameters)},
    )


def _fit_fsm(
    training: HistoryDataset,
    validation: HistoryDataset,
    settings: SlowLoopSelectionConfig,
) -> FittedCandidate:
    rows: list[dict[str, object]] = []
    heads: dict[float, DiagonalGaussianHead] = {}
    selected_posteriors: dict[tuple[float, int, float, float], np.ndarray] = {}
    for floor in settings.diagonal_variance_floor_grid:
        head = DiagonalGaussianHead.fit(
            training.standardized_histories[:, -1, :], training.labels, variance_floor=floor
        )
        heads[floor] = head
        instantaneous = softmax_logits(
            head.logits(validation.standardized_histories.reshape(-1, 14))
        ).reshape(len(validation.labels), settings.budget.history_windows, len(REGIME_CLASSES))
        for enter_run in settings.fsm_enter_run_grid:
            for confidence in settings.fsm_confidence_grid:
                raw = run_length_fsm_posterior(instantaneous, enter_run=enter_run, confidence=confidence)
                temperature, posterior, scan = _select_temperature(
                    raw, validation.labels, settings.temperature_grid, already_posterior=True
                )
                key = (floor, enter_run, confidence, temperature)
                selected_posteriors[key] = posterior
                rows.append(
                    {
                        "variance_floor": floor,
                        "enter_run": enter_run,
                        "confidence": confidence,
                        "temperature": temperature,
                        "validation_nll": _nll(_labels_as_names(validation.labels), posterior),
                        "temperature_scan": scan,
                    }
                )
    selected = min(rows, key=lambda row: (row["validation_nll"], row["enter_run"], row["variance_floor"], row["confidence"]))
    floor = float(selected["variance_floor"])
    enter_run = int(selected["enter_run"])
    confidence = float(selected["confidence"])
    temperature = float(selected["temperature"])
    head = heads[floor]

    def predict(_raw: np.ndarray, standardized: np.ndarray) -> np.ndarray:
        instantaneous = softmax_logits(head.logits(standardized.reshape(-1, 14))).reshape(
            len(standardized), standardized.shape[1], len(REGIME_CLASSES)
        )
        raw = run_length_fsm_posterior(instantaneous, enter_run=enter_run, confidence=confidence)
        return temper_posterior(raw, temperature)

    posterior = predict(validation.raw_histories, validation.standardized_histories)
    parameters = {key: selected[key] for key in ("variance_floor", "enter_run", "confidence", "temperature")}
    return FittedCandidate(
        family="run_length_fsm",
        predict=predict,
        validation_posterior=posterior,
        validation_metrics=_metrics(_labels_as_names(validation.labels), posterior),
        selected_hyperparameters=parameters,
        selection_scan=rows,
        checkpoint={"family": "run_length_fsm", "head": _serialize_head(head), "hyperparameters": parameters},
    )


def _profile(
    candidate: FittedCandidate,
    dataset: HistoryDataset,
    raw_sequence: np.ndarray,
    repeats: int = 7,
) -> dict[str, float | int]:
    take = min(256, len(dataset.labels))
    raw = dataset.raw_histories[:take]
    standardized = dataset.standardized_histories[:take]
    if candidate.online_profile is not None:
        profile_input = np.asarray(raw_sequence[:take], dtype=np.float64)
        candidate.online_profile(profile_input)
    else:
        profile_input = None
        candidate.predict(raw, standardized)
    timings = []
    for _ in range(repeats):
        start = perf_counter()
        if candidate.online_profile is not None:
            candidate.online_profile(profile_input)
        else:
            candidate.predict(raw, standardized)
        timings.append(perf_counter() - start)
    return {
        "host_profile_rows_per_batch": take,
        "host_profile_repeats": repeats,
        "host_batch_median_us_per_update": float(np.median(timings) * 1.0e6 / take),
    }


def _implementation_sha256() -> str:
    digest = hashlib.sha256()
    for relative in (
        "cnn_fpga/decoder/slow_loop_model_selection.py",
        "cnn_fpga/benchmark/slow_loop_model_selection.py",
        "cnn_fpga/decoder/regime_hmm.py",
        "cnn_fpga/benchmark/regime_hmm_baseline.py",
        "physics/syndrome_stream.py",
    ):
        digest.update(relative.encode("utf-8"))
        digest.update((ROOT / relative).read_bytes())
    return digest.hexdigest()


def _checkpoint_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _aggregate(seed_rows: Sequence[dict[str, object]], family: str) -> dict[str, object]:
    rows = [row for row in seed_rows if row["family"] == family]
    scalar_names = (
        "accuracy",
        "macro_f1",
        "negative_log_likelihood",
        "brier_score",
        "expected_calibration_error",
        "mean_transition_detection_delay_windows",
        "false_switch_rate",
    )
    result: dict[str, object] = {
        name: float(np.mean([row["metrics"][name] for row in rows])) for name in scalar_names
    }
    result["class_recall"] = {
        state: float(np.mean([row["metrics"]["class_recall"][state] for row in rows]))
        for state in REGIME_CLASSES
    }
    return result


def build_slow_loop_selection(
    config: SlowLoopSelectionConfig | None = None,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    settings = SlowLoopSelectionConfig() if config is None else config
    if not isinstance(settings, SlowLoopSelectionConfig):
        raise TypeError("config must be SlowLoopSelectionConfig")
    if torch is None:
        raise RuntimeError("production model selection requires torch in DLEnv")
    trajectory_config = RegimeHMMValidationConfig(
        training_seeds=settings.training_seeds,
        validation_seeds=settings.validation_seeds,
        evaluation_seeds=settings.evaluation_seeds,
        windows_per_trajectory=settings.windows_per_trajectory,
    )
    training_trajectories = [_trajectory(seed, trajectory_config) for seed in settings.training_seeds]
    validation_trajectories = [_trajectory(seed, trajectory_config) for seed in settings.validation_seeds]
    evaluation_trajectories = [_trajectory(seed, trajectory_config) for seed in settings.evaluation_seeds]
    standardizer = FeatureStandardizer.fit([trajectory.features for trajectory in training_trajectories])
    training = _dataset(training_trajectories, standardizer, settings.budget.history_windows)
    validation = _dataset(validation_trajectories, standardizer, settings.budget.history_windows)
    evaluation = _dataset(evaluation_trajectories, standardizer, settings.budget.history_windows)

    candidates = {
        "causal_tcn": _fit_neural("causal_tcn", training, validation, settings),
        "small_gru": _fit_neural("small_gru", training, validation, settings),
        "gaussian_hmm": _fit_hmm(training_trajectories, validation, settings),
        "diagonal_kalman": _fit_filtered_diagonal("diagonal_kalman", training, validation, settings),
        "exponential_recurrence": _fit_filtered_diagonal("exponential_recurrence", training, validation, settings),
        "run_length_fsm": _fit_fsm(training, validation, settings),
    }
    profiles = resource_profiles(settings.budget)
    selection_table = []
    for family in MODEL_FAMILIES:
        metrics = candidates[family].validation_metrics
        profile = profiles[family]
        selection_table.append(
            {
                "family": family,
                "eligible": profile.within(settings.budget),
                "validation_negative_log_likelihood": metrics["negative_log_likelihood"],
                "validation_brier_score": metrics["brier_score"],
                "validation_accuracy": metrics["accuracy"],
                "validation_macro_f1": metrics["macro_f1"],
                "macs_per_update_proxy": profile.macs_per_update_proxy,
                "model_and_state_bytes": profile.model_and_state_bytes,
                "selection_key": [
                    metrics["negative_log_likelihood"],
                    metrics["brier_score"],
                    profile.macs_per_update_proxy,
                    profile.trainable_or_fitted_float_values,
                    family,
                ],
            }
        )
    eligible = [row for row in selection_table if row["eligible"]]
    selected_row = min(eligible, key=lambda row: tuple(row["selection_key"]))
    selected_family = str(selected_row["family"])

    evaluation_posteriors = {
        family: candidates[family].predict(evaluation.raw_histories, evaluation.standardized_histories)
        for family in MODEL_FAMILIES
    }
    source_rows: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []
    histories_per_trajectory = settings.windows_per_trajectory - settings.budget.history_windows + 1
    for sequence_index, trajectory in enumerate(evaluation_trajectories):
        mask = evaluation.sequence_indices == sequence_index
        labels = evaluation.labels[mask]
        local_indices = evaluation.local_indices[mask]
        for family in MODEL_FAMILIES:
            posterior = evaluation_posteriors[family][mask]
            metrics = _metrics(_labels_as_names(labels), posterior)
            seed_rows.append({"evaluation_seed": trajectory.base_seed, "family": family, "metrics": metrics})
            for row_index, local_index in enumerate(local_indices):
                probabilities = posterior[row_index]
                source_rows.append(
                    {
                        "evaluation_seed": trajectory.base_seed,
                        "window_index": int(local_index),
                        "history_start_window": int(local_index) - settings.budget.history_windows + 1,
                        "truth_regime": REGIME_CLASSES[int(labels[row_index])],
                        "family": family,
                        "prediction": REGIME_CLASSES[int(np.argmax(probabilities))],
                        **{f"p_{state}": float(probabilities[index]) for index, state in enumerate(REGIME_CLASSES)},
                        "deployable_trace_sha256": trajectory.deployable_trace_sha256,
                        "truth_trace_sha256": trajectory.truth_trace_sha256,
                    }
                )
    aggregate = {family: _aggregate(seed_rows, family) for family in MODEL_FAMILIES}
    evaluation_ranking = sorted(
        MODEL_FAMILIES,
        key=lambda family: (
            aggregate[family]["negative_log_likelihood"],
            aggregate[family]["brier_score"],
            profiles[family].macs_per_update_proxy,
            family,
        ),
    )
    validation_ranking = [row["family"] for row in sorted(eligible, key=lambda row: tuple(row["selection_key"]))]
    runner_up = str(validation_ranking[1])
    selected_seed_nll = {
        int(row["evaluation_seed"]): float(row["metrics"]["negative_log_likelihood"])
        for row in seed_rows
        if row["family"] == selected_family
    }
    runner_seed_nll = {
        int(row["evaluation_seed"]): float(row["metrics"]["negative_log_likelihood"])
        for row in seed_rows
        if row["family"] == runner_up
    }
    runner_minus_selected = [
        runner_seed_nll[seed] - selected_seed_nll[seed] for seed in settings.evaluation_seeds
    ]
    host_profiles = {
        family: _profile(candidates[family], evaluation, evaluation_trajectories[0].features)
        for family in MODEL_FAMILIES
    }
    checkpoint: dict[str, object] = {
        "schema_version": "t4.1.1-slow-loop-model-selection-checkpoints-v1",
        "implementation_sha256": _implementation_sha256(),
        "standardizer": {"mean": standardizer.mean.tolist(), "scale": standardizer.scale.tolist()},
        "selected_family_from_validation": selected_family,
        "models": {family: candidates[family].checkpoint for family in MODEL_FAMILIES},
    }
    class_counts = {
        state: int(sum(label == REGIME_CLASSES.index(state) for label in evaluation.labels))
        for state in REGIME_CLASSES
    }
    gates = {
        "all_six_required_families_present": tuple(candidates) == MODEL_FAMILIES,
        "train_validation_evaluation_seeds_pairwise_disjoint": not (
            set(settings.training_seeds) & set(settings.validation_seeds)
            or set(settings.training_seeds) & set(settings.evaluation_seeds)
            or set(settings.validation_seeds) & set(settings.evaluation_seeds)
        ),
        "common_observation_history_and_output_contract": (
            training.raw_histories.shape[1:] == validation.raw_histories.shape[1:] == evaluation.raw_histories.shape[1:]
            == (settings.budget.history_windows, settings.budget.summary_feature_count)
            and DESCRIPTOR.online_hidden_truth_input == ()
        ),
        "all_families_within_MAC_and_model_state_budget": all(profile.within(settings.budget) for profile in profiles.values()),
        "selection_is_validation_only_without_family_prior": (
            not DESCRIPTOR.evaluation_used_for_selection
            and DESCRIPTOR.model_family_prior.startswith("none")
            and selected_family == min(eligible, key=lambda row: tuple(row["selection_key"]))["family"]
        ),
        "neural_families_use_registered_independent_restarts": all(
            len(candidates[family].selection_scan) == len(settings.neural_restarts)
            and len(settings.neural_restarts) >= 2
            for family in ("causal_tcn", "small_gru")
        ),
        "evaluation_source_rows_complete": len(source_rows) == len(settings.evaluation_seeds) * histories_per_trajectory * len(MODEL_FAMILIES),
        "all_evaluation_posteriors_positive_and_normalized": all(
            np.all(posterior > 0.0) and np.allclose(np.sum(posterior, axis=1), 1.0, atol=1.0e-9, rtol=0.0)
            for posterior in evaluation_posteriors.values()
        ),
        "all_regimes_have_evaluation_support": min(class_counts.values()) >= 0.10 * len(evaluation.labels),
        "all_metrics_and_host_profiles_finite": all(
            all(isfinite(float(aggregate[family][name])) for name in ("accuracy", "negative_log_likelihood", "brier_score", "macro_f1"))
            and isfinite(float(host_profiles[family]["host_batch_median_us_per_update"]))
            and 0.0 < float(host_profiles[family]["host_batch_median_us_per_update"]) < settings.budget.host_software_latency_ceiling_us
            for family in MODEL_FAMILIES
        ),
        "selected_model_evaluated_on_all_registered_seeds": len(selected_seed_nll) == len(settings.evaluation_seeds),
        "checkpoint_contains_every_family_and_source_binding": (
            tuple(checkpoint["models"]) == MODEL_FAMILIES
            and checkpoint["implementation_sha256"] == _implementation_sha256()
        ),
        "role_remains_estimator_not_decoder_or_controller": not DESCRIPTOR.controller and not DESCRIPTOR.logical_decoder,
    }
    failed = [name for name, passed in gates.items() if not passed]
    payload: dict[str, object] = {
        "schema_version": "t4.1.1-slow-loop-model-selection-v1",
        "task_id": "T4.1.1",
        "status": "PASS" if not failed else "FAIL",
        "implementation_sha256": _implementation_sha256(),
        "descriptor": asdict(DESCRIPTOR),
        "validation_config": asdict(settings),
        "common_task_contract": {
            "source_task": "T3.2.6 canonical observed-window regime estimation",
            "observation": "14 training-defined summary features per non-overlapping 32-cycle window",
            "history": f"exactly the last {settings.budget.history_windows} windows; reset/replay at each prediction",
            "output_classes": list(REGIME_CLASSES),
            "primary_selection_metric": "validation negative log likelihood",
            "tie_breakers": ["validation Brier", "MAC proxy", "fitted floats", "family name"],
            "memory_accounting": (
                "model_and_state_bytes counts float32 fitted parameters plus persistent online state; "
                "transient_workspace_bytes separately counts reusable inference scratch"
            ),
            "evaluation_used_for_model_or_hyperparameter_selection": False,
            "excluded_inputs": ["hidden regime", "future windows", "evaluation labels", "logical truth"],
        },
        "resource_profiles": {
            family: asdict(profiles[family]) | host_profiles[family] for family in MODEL_FAMILIES
        },
        "training_and_validation_selection": {
            "training_histories": len(training.labels),
            "validation_histories": len(validation.labels),
            "selection_table": selection_table,
            "validation_ranking": validation_ranking,
            "selected_family": selected_family,
            "runner_up_family": runner_up,
            "family_details": {
                family: {
                    "validation_metrics": candidates[family].validation_metrics,
                    "selected_hyperparameters": candidates[family].selected_hyperparameters,
                    "selection_scan": candidates[family].selection_scan,
                }
                for family in MODEL_FAMILIES
            },
        },
        "evaluation": {
            "trajectories": len(evaluation_trajectories),
            "histories_per_trajectory": histories_per_trajectory,
            "histories": len(evaluation.labels),
            "source_data_rows": len(source_rows),
            "class_counts": class_counts,
            "aggregate": aggregate,
            "evaluation_ranking_diagnostic_not_used_for_selection": evaluation_ranking,
            "validation_winner_rank_on_evaluation": evaluation_ranking.index(selected_family) + 1,
            "paired_seed_runner_minus_selected_NLL": _mean_interval(runner_minus_selected, settings.confidence_level),
            "per_seed": seed_rows,
        },
        "gate_summary": {"passed": sum(bool(value) for value in gates.values()), "failed": len(failed), "gates": gates},
        "claim_boundary": {
            "allowed": (
                "under the registered synthetic four-regime pilot and common 8-window/4096-MAC/4096-byte envelope, "
                f"validation selected {selected_family}; all evaluation results and ranking reversals are reported"
            ),
            "forbidden": (
                "universal architecture superiority, T4.1.2 rich-history completion, logical-error/control gain, "
                "device-calibrated robustness, bit-accurate inference, synthesis, FPGA resource/latency or experiment"
            ),
        },
    }
    return payload, source_rows, checkpoint


def write_slow_loop_selection(
    json_path: str | Path = "docs/t4_1_1_slow_loop_model_selection_validation.json",
    csv_path: str | Path = "docs/t4_1_1_slow_loop_model_selection_source_data.csv",
    checkpoint_path: str | Path = "docs/t4_1_1_slow_loop_model_selection_checkpoints.pt",
    config: SlowLoopSelectionConfig | None = None,
) -> dict[str, object]:
    if torch is None:
        raise RuntimeError("production model selection requires torch in DLEnv")
    payload, rows, checkpoint = build_slow_loop_selection(config)
    json_target = Path(json_path)
    csv_target = Path(csv_path)
    checkpoint_target = Path(checkpoint_path)
    for target in (json_target, csv_target, checkpoint_target):
        target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_target)
    checkpoint_hash = _checkpoint_sha256(checkpoint_target)
    payload["checkpoint"] = {
        "path": checkpoint_target.as_posix(),
        "sha256": checkpoint_hash,
        "schema_version": checkpoint["schema_version"],
    }
    if not rows:
        raise RuntimeError("model selection produced no Source Data")
    with csv_target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload["source_data"] = {
        "path": csv_target.as_posix(),
        "rows": len(rows),
        "sha256": hashlib.sha256(csv_target.read_bytes()).hexdigest(),
    }
    json_target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--json", default="docs/t4_1_1_slow_loop_model_selection_validation.json")
    parser.add_argument("--csv", default="docs/t4_1_1_slow_loop_model_selection_source_data.csv")
    parser.add_argument("--checkpoint", default="docs/t4_1_1_slow_loop_model_selection_checkpoints.pt")
    args = parser.parse_args(argv)
    config = (
        SlowLoopSelectionConfig(
            windows_per_trajectory=128,
            neural_restarts=(41101, 41102),
            neural_epochs=30,
            neural_patience=8,
        )
        if args.smoke
        else None
    )
    payload = write_slow_loop_selection(args.json, args.csv, args.checkpoint, config)
    print(json.dumps({"gate_summary": payload["gate_summary"], "selected_family": payload["training_and_validation_selection"]["selected_family"]}, ensure_ascii=False))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DESCRIPTOR",
    "SlowLoopSelectionDescriptor",
    "SlowLoopSelectionConfig",
    "build_slow_loop_selection",
    "write_slow_loop_selection",
    "_implementation_sha256",
]
