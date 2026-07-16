"""T3.2.2 continuous-drift EWMA/Kalman adaptive MAP benchmark.

The benchmark uses one materialized observation window to update predictors
only after the current evaluation window has been decoded.  Static, latest-
window, EWMA, constant-velocity Kalman, and exact-state oracle MAP therefore
share the same evaluation trace and a one-window causal delay.  Hyperparameters
are selected on separate training seeds with an observation-only next-window
moment score and remain frozen for evaluation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import hashlib
import json
from math import atanh, isfinite, log, pi, sqrt, tanh
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.stats import t as student_t

from cnn_fpga.benchmark.static_map_baseline import (
    STATIC_MAP_ID,
    StaticMAPParameters,
    fit_static_map_from_training_states,
)
from cnn_fpga.decoder.periodic_adaptive_map import (
    ConstantVelocityPeriodicKalman,
    LatestWindowPeriodicPredictor,
    PeriodicGaussianEstimate,
    PeriodicKalmanConfig,
    PeriodicMomentConfig,
    PeriodicMomentEWMA,
    estimate_periodic_gaussian,
    scaled_periodic_kalman_config,
)
from physics.constants import LATTICE_CONST
from physics.drift_processes import DriftState, sample_displacements
from physics.ideal_gkp_decoder import map_decode_2d, standard_binning_1d
from physics.oracle_map import oracle_map_2d


EWMA_ADAPTIVE_MAP_ID = "ewma_periodic_moment_map"
KALMAN_ADAPTIVE_MAP_ID = "kalman_constant_velocity_periodic_map"
WINDOW_PERIODIC_MAP_ID = "latest_window_periodic_moment_map"
FULL_STATE_ORACLE_ID = "full_state_model_oracle_map"
COMPARISON_ID = "t3_2_2_continuous_adaptive_map_comparison"


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


def _integer(value: object, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _wrap(values: NDArray[np.float64], period: float) -> NDArray[np.float64]:
    return np.mod(values + 0.5 * period, period) - 0.5 * period


@dataclass(frozen=True)
class ContinuousAdaptiveDescriptor:
    task_id: str = "T3.2.2"
    comparison_id: str = COMPARISON_ID
    label: str = "Causal full-covariance EWMA/Kalman adaptive periodic MAP"
    deployable_algorithm_family: bool = True
    consumed_observation_fields: tuple[str, ...] = ("residual_q", "residual_p")
    hidden_truth_inputs: tuple[str, ...] = ()
    update_timing: str = "one_window_delay_update_after_current_evaluation"
    state_rule: str = (
        "periodic circular moments estimate mean/full covariance; EWMA filters four "
        "complex moments and Kalman filters five transformed parameters plus velocity"
    )
    evidence_scope: str = "continuous_synthetic_wrapped_gaussian_syndrome_level"
    excluded_claims: tuple[str, ...] = (
        "finite_energy_protocol_fidelity",
        "loss_outlier_or_leakage_identification",
        "device_calibration",
        "FPGA_synthesis_or_measured_latency",
    )


ADAPTIVE_DESCRIPTOR = ContinuousAdaptiveDescriptor()


@dataclass(frozen=True)
class ContinuousDriftScenario:
    scenario_id: str

    def states(self, windows: int) -> tuple[DriftState, ...]:
        count = _integer(windows, "windows", 8)
        lam = LATTICE_CONST
        states: list[DriftState] = []
        for step in range(count):
            progress = step / (count - 1)
            phase = 2.0 * pi * step / 32.0
            if self.scenario_id == "linear_mean":
                mu_q = (-0.18 + 0.35 * progress) * lam
                mu_p = (0.14 - 0.26 * progress) * lam
                sigma_q, sigma_p, rho = 0.155 * lam, 0.105 * lam, 0.25
            elif self.scenario_id == "variance_correlation_ramp":
                mu_q, mu_p = 0.08 * lam, -0.05 * lam
                sigma_q = 0.11 * lam * np.exp(log(0.23 / 0.11) * progress)
                sigma_p = 0.20 * lam * np.exp(log(0.12 / 0.20) * progress)
                rho = tanh(atanh(-0.60) + (atanh(0.60) - atanh(-0.60)) * progress)
            elif self.scenario_id == "sinusoidal_joint":
                mu_q = 0.18 * lam * np.sin(phase)
                mu_p = 0.14 * lam * np.cos(phase + 0.35)
                sigma_q = (0.165 + 0.035 * np.sin(phase + 0.40)) * lam
                sigma_p = (0.145 + 0.030 * np.cos(phase - 0.25)) * lam
                rho = 0.55 * np.sin(0.70 * phase - 0.20)
            elif self.scenario_id == "smooth_mixed":
                mu_q = (-0.12 + 0.24 * progress + 0.07 * np.sin(phase)) * lam
                mu_p = (0.10 - 0.18 * progress + 0.05 * np.cos(phase + 0.2)) * lam
                sigma_q = (0.13 + 0.075 * progress) * lam
                sigma_p = (0.20 - 0.060 * progress) * lam
                rho = -0.48 + 0.90 * progress
            else:
                raise ValueError(f"unknown continuous scenario {self.scenario_id!r}")
            states.append(
                DriftState(
                    step=step,
                    time=float(step),
                    mu_q=float(mu_q),
                    mu_p=float(mu_p),
                    sigma_q=float(sigma_q),
                    sigma_p=float(sigma_p),
                    rho=float(rho),
                    source="t3.2.2-continuous-drift",
                    regime=self.scenario_id,
                )
            )
        return tuple(states)


def continuous_drift_scenarios() -> tuple[ContinuousDriftScenario, ...]:
    return tuple(
        ContinuousDriftScenario(name)
        for name in (
            "linear_mean",
            "variance_correlation_ramp",
            "sinusoidal_joint",
            "smooth_mixed",
        )
    )


@dataclass(frozen=True)
class ContinuousAdaptiveValidationConfig:
    training_seeds: tuple[int, ...] = (20260811, 20260812, 20260813)
    evaluation_seeds: tuple[int, ...] = tuple(range(20260831, 20260839))
    windows: int = 48
    calibration_windows: int = 4
    observation_samples_per_window: int = 384
    training_score_samples_per_window: int = 384
    evaluation_samples_per_window: int = 1024
    ewma_alpha_candidates: tuple[float, ...] = (
        0.15,
        0.25,
        0.35,
        0.50,
        0.70,
        0.85,
        1.00,
    )
    kalman_process_scale_candidates: tuple[float, ...] = (
        0.50,
        1.00,
        1.50,
        2.00,
        3.00,
    )
    kalman_measurement_scale_candidates: tuple[float, ...] = (
        0.50,
        0.75,
        1.00,
        1.50,
    )
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
        for name, minimum in (
            ("windows", 16),
            ("calibration_windows", 2),
            ("observation_samples_per_window", 128),
            ("training_score_samples_per_window", 128),
            ("evaluation_samples_per_window", 256),
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum))
        if self.calibration_windows >= self.windows:
            raise ValueError("calibration_windows must be smaller than windows")
        alphas = tuple(_finite(value, "ewma alpha") for value in self.ewma_alpha_candidates)
        process = tuple(
            _finite(value, "Kalman process scale")
            for value in self.kalman_process_scale_candidates
        )
        measurement = tuple(
            _finite(value, "Kalman measurement scale")
            for value in self.kalman_measurement_scale_candidates
        )
        if len(alphas) < 3 or any(not 0.0 < value <= 1.0 for value in alphas):
            raise ValueError("EWMA candidates must contain at least three values in (0,1]")
        if len(process) < 2 or len(measurement) < 2:
            raise ValueError("Kalman tuning requires at least a 2x2 scale grid")
        if any(value <= 0.0 for value in process + measurement):
            raise ValueError("Kalman candidate scales must be positive")
        object.__setattr__(self, "ewma_alpha_candidates", alphas)
        object.__setattr__(self, "kalman_process_scale_candidates", process)
        object.__setattr__(self, "kalman_measurement_scale_candidates", measurement)
        confidence = _finite(self.confidence_level, "confidence_level")
        if not 0.5 < confidence < 1.0:
            raise ValueError("confidence_level must lie in (0.5,1)")
        object.__setattr__(self, "confidence_level", confidence)
        workload = len(evaluation) * len(continuous_drift_scenarios()) * self.windows * (
            self.observation_samples_per_window + self.evaluation_samples_per_window
        )
        if workload > 5_000_000:
            raise ValueError("evaluation workload must not exceed 5,000,000 samples")


@dataclass(frozen=True)
class FrozenAdaptiveHyperparameters:
    ewma_alpha: float
    kalman_process_scale: float
    kalman_measurement_scale: float
    ewma_candidate_scores: tuple[tuple[float, float], ...]
    kalman_candidate_scores: tuple[tuple[float, float, float], ...]
    training_trace_sha256: str
    selection_objective: str = (
        "observation_only_independent_window_periodic_moment_forecast_score"
    )


@dataclass(frozen=True)
class AdaptiveCostProfile:
    observation_samples_per_window: int
    updates_per_window: int
    causal_delay_windows: int
    complex_exponentials_per_observation: int
    complex_products_per_observation: int
    ewma_complex_state_values: int
    ewma_complex_blends_per_update: int
    kalman_state_values: int
    kalman_covariance_values: int
    kalman_innovation_dimension: int
    target_lut: int | None = None
    target_ff: int | None = None
    target_bram: int | None = None
    target_dsp: int | None = None
    target_fmax_hz: float | None = None
    target_measured: bool = False
    scope: str = "deterministic_operation_storage_proxy_not_synthesis"


def adaptive_cost_profile(config: ContinuousAdaptiveValidationConfig) -> AdaptiveCostProfile:
    if not isinstance(config, ContinuousAdaptiveValidationConfig):
        raise TypeError("config must be ContinuousAdaptiveValidationConfig")
    return AdaptiveCostProfile(
        observation_samples_per_window=config.observation_samples_per_window,
        updates_per_window=1,
        causal_delay_windows=1,
        complex_exponentials_per_observation=2,
        complex_products_per_observation=2,
        ewma_complex_state_values=4,
        ewma_complex_blends_per_update=4,
        kalman_state_values=10,
        kalman_covariance_values=100,
        kalman_innovation_dimension=5,
    )


def _residuals_and_truth(
    state: DriftState,
    samples: int,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
    displacements, _ = sample_displacements(state, samples, rng=rng)
    q = standard_binning_1d(displacements[:, 0])
    p = standard_binning_1d(displacements[:, 1])
    residual = np.column_stack((q.syndrome, p.syndrome)).astype(np.float64)
    truth = (
        2 * np.asarray(q.logical_parity, dtype=np.int64)
        + np.asarray(p.logical_parity, dtype=np.int64)
    )
    return residual, truth, np.asarray(displacements, dtype=np.float64)


def _calibration_residuals(
    state: DriftState,
    windows: int,
    samples: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    return np.concatenate(
        [_residuals_and_truth(state, samples, rng)[0] for _ in range(windows)],
        axis=0,
    )


def _estimate_score(
    prediction: PeriodicGaussianEstimate,
    target: PeriodicGaussianEstimate,
) -> float:
    lattice = LATTICE_CONST
    mean_error = _wrap(prediction.mean_array() - target.mean_array(), lattice)
    prediction_cov = prediction.covariance_array()
    target_cov = target.covariance_array()
    prediction_rho = prediction.rho
    target_rho = target.rho
    components = np.asarray(
        [
            mean_error[0] / (0.05 * lattice),
            mean_error[1] / (0.05 * lattice),
            (log(prediction_cov[0, 0]) - log(target_cov[0, 0])) / 0.25,
            (log(prediction_cov[1, 1]) - log(target_cov[1, 1])) / 0.25,
            (atanh(prediction_rho) - atanh(target_rho)) / 0.25,
        ],
        dtype=np.float64,
    )
    return float(np.mean(components * components))


@dataclass(frozen=True)
class _TrainingTrace:
    calibration: NDArray[np.float64]
    updates: tuple[NDArray[np.float64], ...]
    scores: tuple[NDArray[np.float64], ...]
    trace_sha256: str


def _materialize_training_traces(
    config: ContinuousAdaptiveValidationConfig,
) -> tuple[_TrainingTrace, ...]:
    traces: list[_TrainingTrace] = []
    for scenario_index, scenario in enumerate(continuous_drift_scenarios()):
        states = scenario.states(config.windows)
        for seed in config.training_seeds:
            derived_seed = int(seed + 100_000 * scenario_index)
            rng = np.random.default_rng(derived_seed)
            calibration = _calibration_residuals(
                states[0],
                config.calibration_windows,
                config.observation_samples_per_window,
                rng,
            )
            updates: list[NDArray[np.float64]] = []
            scores: list[NDArray[np.float64]] = []
            digest = hashlib.sha256()
            digest.update(scenario.scenario_id.encode("utf-8"))
            digest.update(derived_seed.to_bytes(8, "little", signed=False))
            digest.update(np.asarray(calibration, dtype="<f8").tobytes())
            for state in states:
                update = _residuals_and_truth(
                    state,
                    config.observation_samples_per_window,
                    rng,
                )[0]
                score = _residuals_and_truth(
                    state,
                    config.training_score_samples_per_window,
                    rng,
                )[0]
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


def _score_predictor(
    traces: Sequence[_TrainingTrace],
    *,
    kind: str,
    alpha: float | None = None,
    kalman_config: PeriodicKalmanConfig | None = None,
    moment_config: PeriodicMomentConfig,
) -> float:
    values: list[float] = []
    for trace in traces:
        if kind == "ewma":
            if alpha is None:
                raise ValueError("EWMA score requires alpha")
            predictor: PeriodicMomentEWMA | ConstantVelocityPeriodicKalman = (
                PeriodicMomentEWMA(trace.calibration, alpha=alpha, config=moment_config)
            )
        elif kind == "kalman":
            if kalman_config is None:
                raise ValueError("Kalman score requires kalman_config")
            predictor = ConstantVelocityPeriodicKalman(
                trace.calibration,
                moment_config=moment_config,
                kalman_config=kalman_config,
            )
        else:
            raise ValueError("kind must be 'ewma' or 'kalman'")
        for window_id, (update, scoring) in enumerate(zip(trace.updates, trace.scores)):
            target = estimate_periodic_gaussian(
                scoring,
                moment_config,
                source="training_scoring_observation",
                window_id=window_id,
            )
            values.append(_estimate_score(predictor.prediction(), target))
            predictor.update(update, window_id=window_id)
    return float(np.mean(values))


def select_frozen_hyperparameters(
    config: ContinuousAdaptiveValidationConfig | None = None,
) -> FrozenAdaptiveHyperparameters:
    settings = ContinuousAdaptiveValidationConfig() if config is None else config
    if not isinstance(settings, ContinuousAdaptiveValidationConfig):
        raise TypeError("config must be ContinuousAdaptiveValidationConfig")
    moment_config = PeriodicMomentConfig(
        minimum_samples=min(64, settings.observation_samples_per_window)
    )
    traces = _materialize_training_traces(settings)
    ewma_scores = tuple(
        (
            alpha,
            _score_predictor(
                traces,
                kind="ewma",
                alpha=alpha,
                moment_config=moment_config,
            ),
        )
        for alpha in settings.ewma_alpha_candidates
    )
    kalman_scores = tuple(
        (
            process_scale,
            measurement_scale,
            _score_predictor(
                traces,
                kind="kalman",
                kalman_config=scaled_periodic_kalman_config(
                    process_scale=process_scale,
                    measurement_scale=measurement_scale,
                ),
                moment_config=moment_config,
            ),
        )
        for process_scale in settings.kalman_process_scale_candidates
        for measurement_scale in settings.kalman_measurement_scale_candidates
    )
    selected_ewma = min(ewma_scores, key=lambda row: (row[1], row[0]))
    selected_kalman = min(kalman_scores, key=lambda row: (row[2], row[0], row[1]))
    digest = hashlib.sha256()
    for trace in traces:
        digest.update(bytes.fromhex(trace.trace_sha256))
    return FrozenAdaptiveHyperparameters(
        ewma_alpha=float(selected_ewma[0]),
        kalman_process_scale=float(selected_kalman[0]),
        kalman_measurement_scale=float(selected_kalman[1]),
        ewma_candidate_scores=tuple((float(a), float(score)) for a, score in ewma_scores),
        kalman_candidate_scores=tuple(
            (float(p), float(m), float(score)) for p, m, score in kalman_scores
        ),
        training_trace_sha256=digest.hexdigest(),
    )


def _static_training_parameters(
    config: ContinuousAdaptiveValidationConfig,
) -> StaticMAPParameters:
    states: list[DriftState] = []
    for scenario_index, scenario in enumerate(continuous_drift_scenarios()):
        for state in scenario.states(config.windows):
            states.append(
                DriftState(
                    **{
                        **state.__dict__,
                        "step": len(states),
                        "time": float(len(states)),
                        "seed": config.training_seeds[0] + scenario_index,
                    }
                )
            )
    return fit_static_map_from_training_states(
        tuple(states),
        training_protocol_id=(
            "t3.2.2-continuous-scenario-average-v1:"
            f"windows={config.windows}:training_seeds={config.training_seeds}"
        ),
    )


def _scores(
    posterior: NDArray[np.float64],
    truth: NDArray[np.int64],
) -> tuple[float, float]:
    probabilities = posterior.reshape((-1, 4))
    selected = probabilities[np.arange(truth.size), truth]
    nll = float(np.mean(-np.log(np.clip(selected, np.finfo(float).tiny, 1.0))))
    one_hot = np.eye(4, dtype=np.float64)[truth]
    brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=-1)))
    return nll, brier


def _parameter_errors(
    estimate: PeriodicGaussianEstimate,
    state: DriftState,
) -> tuple[float, float]:
    mean_error = _wrap(estimate.mean_array() - state.mean, LATTICE_CONST)
    covariance_error = estimate.covariance_array() - state.covariance
    return (
        float(np.linalg.norm(mean_error) / LATTICE_CONST),
        float(np.linalg.norm(covariance_error, ord="fro") / LATTICE_CONST**2),
    )


def _mean_interval(values: Sequence[float], confidence: float) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    estimate = float(np.mean(array))
    degrees_freedom = int(array.size - 1)
    standard_error = float(np.std(array, ddof=1) / sqrt(array.size))
    critical = float(student_t.ppf(0.5 + confidence / 2.0, degrees_freedom))
    return {
        "estimate": estimate,
        "standard_error": standard_error,
        "ci_low": estimate - critical * standard_error,
        "ci_high": estimate + critical * standard_error,
        "cluster_unit": "evaluation_seed",
        "interval_method": "two_sided_student_t_cluster_mean",
        "degrees_freedom": degrees_freedom,
    }


def validate_continuous_adaptive_registration() -> tuple[str, ...]:
    from cnn_fpga.benchmark.standard_binning_baseline import (
        major_comparison_registry,
        validate_major_comparison_registry,
    )

    gates = validate_major_comparison_registry()
    matches = [
        entry for entry in major_comparison_registry() if entry.comparison_id == COMPARISON_ID
    ]
    if len(matches) != 1:
        raise ValueError("T3.2.2 comparison must be registered exactly once")
    entry = matches[0]
    expected = (
        "standard_binning",
        STATIC_MAP_ID,
        WINDOW_PERIODIC_MAP_ID,
        EWMA_ADAPTIVE_MAP_ID,
        KALMAN_ADAPTIVE_MAP_ID,
        FULL_STATE_ORACLE_ID,
    )
    if entry.method_ids != expected:
        raise ValueError("T3.2.2 method order/roles drifted from the comparison contract")
    if entry.static_anchor_method_id != STATIC_MAP_ID:
        raise ValueError("T3.2.2 must use formal static MAP as its static anchor")
    if entry.reference_anchor_method_id != FULL_STATE_ORACLE_ID:
        raise ValueError("T3.2.2 must use full-state model oracle as its reference")
    return gates


def _implementation_sha256() -> str:
    paths = (
        Path(__file__),
        Path(__file__).parents[1] / "decoder" / "periodic_adaptive_map.py",
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
    evaluation_seed: int,
    settings: ContinuousAdaptiveValidationConfig,
    hyperparameters: FrozenAdaptiveHyperparameters,
    static_parameters: StaticMAPParameters,
    moment_config: PeriodicMomentConfig,
) -> dict[str, object]:
    derived_seed = int(evaluation_seed + 100_000 * scenario_index)
    rng = np.random.default_rng(derived_seed)
    states = scenario.states(settings.windows)
    calibration = _calibration_residuals(
        states[0],
        settings.calibration_windows,
        settings.observation_samples_per_window,
        rng,
    )
    latest = LatestWindowPeriodicPredictor(calibration, moment_config)
    ewma = PeriodicMomentEWMA(
        calibration,
        alpha=hyperparameters.ewma_alpha,
        config=moment_config,
    )
    kalman = ConstantVelocityPeriodicKalman(
        calibration,
        moment_config=moment_config,
        kalman_config=scaled_periodic_kalman_config(
            process_scale=hyperparameters.kalman_process_scale,
            measurement_scale=hyperparameters.kalman_measurement_scale,
        ),
    )
    method_names = ("standard", "static", "window", "ewma", "kalman", "oracle")
    failures = {name: 0 for name in method_names}
    nll_sums = {name: 0.0 for name in method_names if name != "standard"}
    brier_sums = {name: 0.0 for name in method_names if name != "standard"}
    parameter_errors = {
        name: [] for name in ("window", "ewma", "kalman")
    }
    digest = hashlib.sha256()
    digest.update(scenario.scenario_id.encode("utf-8"))
    digest.update(derived_seed.to_bytes(8, "little", signed=False))
    digest.update(np.asarray(calibration, dtype="<f8").tobytes())
    total_samples = 0
    for window_id, state in enumerate(states):
        predictions = {
            "window": latest.prediction(),
            "ewma": ewma.prediction(),
            "kalman": kalman.prediction(),
        }
        residual, truth, displacements = _residuals_and_truth(
            state,
            settings.evaluation_samples_per_window,
            rng,
        )
        digest.update(window_id.to_bytes(4, "little", signed=False))
        digest.update(np.asarray(displacements, dtype="<f8").tobytes())
        total_samples += truth.size
        failures["standard"] += int(np.sum(truth != 0))
        static_result = map_decode_2d(
            residual,
            static_parameters.covariance_array(),
            mean=static_parameters.mean_array(),
        )
        oracle_result = oracle_map_2d(residual, state)
        results = {
            "static": static_result,
            "oracle": oracle_result,
        }
        for name, prediction in predictions.items():
            results[name] = map_decode_2d(
                residual,
                prediction.covariance_array(),
                mean=prediction.mean_array(),
            )
            mean_error, covariance_error = _parameter_errors(prediction, state)
            parameter_errors[name].append((mean_error, covariance_error))
        for name, result in results.items():
            decision = np.asarray(result.logical_class, dtype=np.int64)
            failures[name] += int(np.sum(decision != truth))
            nll, brier = _scores(np.asarray(result.posterior), truth)
            nll_sums[name] += nll * truth.size
            brier_sums[name] += brier * truth.size
        observation = _residuals_and_truth(
            state,
            settings.observation_samples_per_window,
            rng,
        )[0]
        digest.update(np.asarray(observation, dtype="<f8").tobytes())
        # Causal ordering: the current observation is applied only after every
        # current-window decoder and score has finished.
        latest.update(observation, window_id=window_id)
        ewma.update(observation, window_id=window_id)
        kalman.update(observation, window_id=window_id)
    row: dict[str, object] = {
        "scenario_id": scenario.scenario_id,
        "evaluation_seed": derived_seed,
        "windows": settings.windows,
        "observation_samples_per_window": settings.observation_samples_per_window,
        "evaluation_samples": total_samples,
        "trace_sha256": digest.hexdigest(),
    }
    for name in method_names:
        row[f"{name}_error_rate"] = failures[name] / total_samples
    for name in nll_sums:
        row[f"{name}_nll"] = nll_sums[name] / total_samples
        row[f"{name}_brier"] = brier_sums[name] / total_samples
    for name, values in parameter_errors.items():
        array = np.asarray(values, dtype=np.float64)
        row[f"{name}_mean_tracking_rmse_lattice"] = float(
            sqrt(np.mean(np.square(array[:, 0])))
        )
        row[f"{name}_covariance_tracking_rmse_lattice2"] = float(
            sqrt(np.mean(np.square(array[:, 1])))
        )
    row["static_minus_ewma_error_rate"] = (
        float(row["static_error_rate"]) - float(row["ewma_error_rate"])
    )
    row["static_minus_kalman_error_rate"] = (
        float(row["static_error_rate"]) - float(row["kalman_error_rate"])
    )
    row["window_minus_best_recursive_error_rate"] = float(row["window_error_rate"]) - min(
        float(row["ewma_error_rate"]), float(row["kalman_error_rate"])
    )
    return row


def build_continuous_adaptive_validation(
    config: ContinuousAdaptiveValidationConfig | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    settings = ContinuousAdaptiveValidationConfig() if config is None else config
    if not isinstance(settings, ContinuousAdaptiveValidationConfig):
        raise TypeError("config must be ContinuousAdaptiveValidationConfig")
    registry_gates = validate_continuous_adaptive_registration()
    hyperparameters = select_frozen_hyperparameters(settings)
    static_parameters = _static_training_parameters(settings)
    moment_config = PeriodicMomentConfig(
        minimum_samples=min(64, settings.observation_samples_per_window)
    )
    rows: list[dict[str, object]] = []
    scenario_payloads: list[dict[str, object]] = []
    for scenario_index, scenario in enumerate(continuous_drift_scenarios()):
        scenario_rows = [
            _evaluate_seed(
                scenario,
                scenario_index,
                seed,
                settings,
                hyperparameters,
                static_parameters,
                moment_config,
            )
            for seed in settings.evaluation_seeds
        ]
        rows.extend(scenario_rows)
        ewma_gain = [float(row["static_minus_ewma_error_rate"]) for row in scenario_rows]
        kalman_gain = [float(row["static_minus_kalman_error_rate"]) for row in scenario_rows]
        recursive_gain = [
            float(row["window_minus_best_recursive_error_rate"]) for row in scenario_rows
        ]
        payload: dict[str, object] = {
            "scenario_id": scenario.scenario_id,
            "seeds": len(scenario_rows),
            "windows": settings.windows,
            "evaluation_samples": sum(int(row["evaluation_samples"]) for row in scenario_rows),
            "unique_trace_hashes": len({str(row["trace_sha256"]) for row in scenario_rows}),
            "static_minus_ewma_seed_cluster_ci": _mean_interval(
                ewma_gain, settings.confidence_level
            ),
            "static_minus_kalman_seed_cluster_ci": _mean_interval(
                kalman_gain, settings.confidence_level
            ),
            "window_minus_best_recursive_seed_cluster_ci": _mean_interval(
                recursive_gain, settings.confidence_level
            ),
        }
        for name in ("standard", "static", "window", "ewma", "kalman", "oracle"):
            payload[f"{name}_error_rate"] = float(
                np.mean([float(row[f"{name}_error_rate"]) for row in scenario_rows])
            )
        for name in ("static", "window", "ewma", "kalman", "oracle"):
            payload[f"{name}_nll"] = float(
                np.mean([float(row[f"{name}_nll"]) for row in scenario_rows])
            )
            payload[f"{name}_brier"] = float(
                np.mean([float(row[f"{name}_brier"]) for row in scenario_rows])
            )
        for name in ("window", "ewma", "kalman"):
            for metric in (
                "mean_tracking_rmse_lattice",
                "covariance_tracking_rmse_lattice2",
            ):
                payload[f"{name}_{metric}"] = float(
                    np.mean([float(row[f"{name}_{metric}"]) for row in scenario_rows])
                )
        scenario_payloads.append(payload)
    aggregate_ewma = [
        float(
            np.mean(
                [
                    row["static_minus_ewma_error_rate"]
                    for row in rows
                    if int(row["evaluation_seed"]) % 100_000 == int(seed) % 100_000
                ]
            )
        )
        for seed in settings.evaluation_seeds
    ]
    aggregate_kalman = [
        float(
            np.mean(
                [
                    row["static_minus_kalman_error_rate"]
                    for row in rows
                    if int(row["evaluation_seed"]) % 100_000 == int(seed) % 100_000
                ]
            )
        )
        for seed in settings.evaluation_seeds
    ]
    cost = adaptive_cost_profile(settings)
    gates = {
        "training_and_evaluation_are_disjoint": not (
            set(settings.training_seeds) & set(settings.evaluation_seeds)
        ),
        "hyperparameters_selected_before_evaluation": (
            len(hyperparameters.training_trace_sha256) == 64
            and hyperparameters.selection_objective.startswith("observation_only")
        ),
        "selected_hyperparameters_are_not_grid_boundaries": (
            min(settings.ewma_alpha_candidates)
            < hyperparameters.ewma_alpha
            < max(settings.ewma_alpha_candidates)
            and min(settings.kalman_process_scale_candidates)
            < hyperparameters.kalman_process_scale
            < max(settings.kalman_process_scale_candidates)
            and min(settings.kalman_measurement_scale_candidates)
            < hyperparameters.kalman_measurement_scale
            < max(settings.kalman_measurement_scale_candidates)
        ),
        "comparison_roles_are_registered": any(
            item == f"registry:{COMPARISON_ID}" for item in registry_gates
        ),
        "all_evaluation_traces_are_unique": len({str(row["trace_sha256"]) for row in rows})
        == len(rows),
        "observation_and_update_budget_is_identical": all(
            int(row["observation_samples_per_window"])
            == settings.observation_samples_per_window
            for row in rows
        ),
        "ewma_improves_static_in_every_scenario": all(
            item["static_minus_ewma_seed_cluster_ci"]["ci_low"] > 0.0
            for item in scenario_payloads
        ),
        "kalman_improves_static_in_every_scenario": all(
            item["static_minus_kalman_seed_cluster_ci"]["ci_low"] > 0.0
            for item in scenario_payloads
        ),
        "best_recursive_point_not_worse_than_latest_window": all(
            min(item["ewma_error_rate"], item["kalman_error_rate"])
            <= item["window_error_rate"] + 1.0e-15
            for item in scenario_payloads
        ),
        "recursive_tracking_improves_mean_and_covariance": all(
            min(
                item["ewma_mean_tracking_rmse_lattice"],
                item["kalman_mean_tracking_rmse_lattice"],
            )
            < item["window_mean_tracking_rmse_lattice"]
            and min(
                item["ewma_covariance_tracking_rmse_lattice2"],
                item["kalman_covariance_tracking_rmse_lattice2"],
            )
            < item["window_covariance_tracking_rmse_lattice2"]
            for item in scenario_payloads
        ),
        "adaptive_proper_scores_improve_static_in_every_scenario": all(
            min(item["ewma_nll"], item["kalman_nll"]) < item["static_nll"]
            and min(item["ewma_brier"], item["kalman_brier"]) < item["static_brier"]
            for item in scenario_payloads
        ),
        "aggregate_ewma_gain_resolved": _mean_interval(
            aggregate_ewma, settings.confidence_level
        )["ci_low"]
        > 0.0,
        "aggregate_kalman_gain_resolved": _mean_interval(
            aggregate_kalman, settings.confidence_level
        )["ci_low"]
        > 0.0,
        "oracle_remains_strict_reference": all(
            min(item["ewma_error_rate"], item["kalman_error_rate"])
            > item["oracle_error_rate"]
            for item in scenario_payloads
        ),
        "cost_profile_remains_not_synthesis": not cost.target_measured,
    }
    failures = [name for name, passed in gates.items() if not passed]
    payload = {
        "schema_version": "t3.2.2-continuous-adaptive-map-v1",
        "task_id": "T3.2.2",
        "status": "PASS" if not failures else "FAIL",
        "implementation_sha256": _implementation_sha256(),
        "descriptor": asdict(ADAPTIVE_DESCRIPTOR),
        "observation_budget": {
            "fields": ["residual_q", "residual_p"],
            "samples_per_window": settings.observation_samples_per_window,
            "updates_per_window": 1,
            "causal_delay_windows": 1,
            "evaluation_buffer_separate_from_observation_buffer": True,
            "hidden_truth_inputs": [],
        },
        "validation_config": asdict(settings),
        "frozen_hyperparameters": asdict(hyperparameters),
        "static_training_parameters": asdict(static_parameters),
        "scenarios": scenario_payloads,
        "aggregate": {
            "scenarios": len(scenario_payloads),
            "evaluation_seeds_per_scenario": len(settings.evaluation_seeds),
            "windows": len(rows) * settings.windows,
            "evaluation_samples": sum(int(row["evaluation_samples"]) for row in rows),
            "source_data_rows": len(rows),
            "static_minus_ewma_seed_cluster_ci": _mean_interval(
                aggregate_ewma, settings.confidence_level
            ),
            "static_minus_kalman_seed_cluster_ci": _mean_interval(
                aggregate_kalman, settings.confidence_level
            ),
        },
        "cost_profile": asdict(cost),
        "gate_summary": {
            "passed": len(gates) - len(failures),
            "failed": len(failures),
            "gates": gates,
        },
        "claim_boundary": {
            "allowed": (
                "on the registered continuous wrapped-Gaussian syndrome scenarios, "
                "full-covariance EWMA and constant-velocity Kalman predictors are causal "
                "strong traditional adaptive MAP baselines"
            ),
            "forbidden": (
                "universal adaptive superiority, loss/outlier/leakage identification, "
                "finite-energy protocol fidelity, device calibration, CNN superiority, "
                "or FPGA synthesis/measurement"
            ),
        },
    }
    return json.loads(json.dumps(payload, ensure_ascii=False)), rows


def write_continuous_adaptive_validation(
    json_path: str | Path,
    csv_path: str | Path,
    config: ContinuousAdaptiveValidationConfig | None = None,
) -> dict[str, object]:
    payload, rows = build_continuous_adaptive_validation(config)
    output_json = Path(json_path)
    output_csv = Path(csv_path)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate T3.2.2 continuous EWMA/Kalman adaptive MAP baselines"
    )
    parser.add_argument(
        "--json", default="docs/t3_2_2_continuous_adaptive_map_validation.json"
    )
    parser.add_argument(
        "--csv", default="docs/t3_2_2_continuous_adaptive_map_source_data.csv"
    )
    arguments = parser.parse_args()
    result = write_continuous_adaptive_validation(arguments.json, arguments.csv)
    print(json.dumps(result["gate_summary"], ensure_ascii=False))


__all__ = [
    "EWMA_ADAPTIVE_MAP_ID",
    "KALMAN_ADAPTIVE_MAP_ID",
    "WINDOW_PERIODIC_MAP_ID",
    "COMPARISON_ID",
    "ContinuousAdaptiveDescriptor",
    "ADAPTIVE_DESCRIPTOR",
    "ContinuousDriftScenario",
    "continuous_drift_scenarios",
    "ContinuousAdaptiveValidationConfig",
    "FrozenAdaptiveHyperparameters",
    "AdaptiveCostProfile",
    "adaptive_cost_profile",
    "select_frozen_hyperparameters",
    "validate_continuous_adaptive_registration",
    "build_continuous_adaptive_validation",
    "write_continuous_adaptive_validation",
]
