"""T1.3.4：现有 Window/EKF baseline 与 full-state oracle 的 causal alignment harness。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Optional

import numpy as np

from cnn_fpga.decoder.ekf_baseline import EKFBaseline, EKFBaselineConfig
from cnn_fpga.decoder.param_mapper import NoisePrediction
from cnn_fpga.decoder.window_baseline import WindowVarianceBaseline, WindowVarianceConfig
from cnn_fpga.benchmark.standard_binning_baseline import (
    STANDARD_BINNING_ID,
    standard_binning_paired_outcomes,
)
from cnn_fpga.benchmark.static_map_baseline import (
    STATIC_MAP_ID,
    StaticMAPParameters,
    fit_static_map_from_training_states,
    static_map_logical_class,
)
from physics.constants import LATTICE_CONST
from physics.drift_processes import DriftState, StepDriftProcess, sample_displacements
from physics.ideal_gkp_decoder import map_decode_2d, standard_binning_1d
from physics.oracle_gap import OracleGapMetrics, compute_oracle_gap_metrics
from physics.oracle_map import oracle_map_2d


@dataclass(frozen=True)
class AdaptiveAlignmentConfig:
    windows: int = 24
    change_step: int = 8
    calibration_windows: int = 6
    observation_samples_per_window: int = 3_000
    evaluation_samples_per_window: int = 3_000
    histogram_bins: int = 48
    sigma_ratio_p: float = 0.55
    before_mu_q_fraction: float = 0.0
    before_mu_p_fraction: float = 0.0
    after_mu_q_fraction: float = 0.20
    after_mu_p_fraction: float = -0.15
    before_sigma_fraction: float = 0.18
    after_sigma_fraction: float = 0.22
    before_theta_deg: float = 0.0
    after_theta_deg: float = 15.0
    bootstrap_replicates: int = 3_000
    seed: int = 20260714
    static_training_windows: int = 24
    static_training_change_step: int = 8
    static_training_seed: int = 20260312

    def __post_init__(self) -> None:
        for name in (
            "windows",
            "change_step",
            "calibration_windows",
            "observation_samples_per_window",
            "evaluation_samples_per_window",
            "histogram_bins",
            "bootstrap_replicates",
            "seed",
            "static_training_windows",
            "static_training_change_step",
            "static_training_seed",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer")
            object.__setattr__(self, name, int(value))
        if self.windows < 4:
            raise ValueError("windows must be at least 4")
        if self.windows > 10_000:
            raise ValueError("windows must not exceed 10000")
        if not 1 <= self.change_step < self.windows:
            raise ValueError("change_step must lie in [1, windows-1]")
        if self.calibration_windows < 2:
            raise ValueError("calibration_windows must be at least 2")
        if self.calibration_windows > 10_000:
            raise ValueError("calibration_windows must not exceed 10000")
        if self.observation_samples_per_window < 200:
            raise ValueError("observation_samples_per_window must be at least 200")
        if self.evaluation_samples_per_window < 200:
            raise ValueError("evaluation_samples_per_window must be at least 200")
        if self.histogram_bins < 16:
            raise ValueError("histogram_bins must be at least 16")
        if self.histogram_bins > 512:
            raise ValueError("histogram_bins must not exceed 512")
        if self.bootstrap_replicates < 0 or self.bootstrap_replicates > 1_000_000:
            raise ValueError("bootstrap_replicates must lie in [0, 1000000]")
        if self.seed < 0 or self.seed > 2**64 - 4:
            raise ValueError("seed must lie in [0, 2**64-4] so child seeds remain valid")
        if self.static_training_windows < 4:
            raise ValueError("static_training_windows must be at least 4")
        if not 1 <= self.static_training_change_step < self.static_training_windows:
            raise ValueError(
                "static_training_change_step must lie in [1, static_training_windows-1]"
            )
        if self.static_training_seed < 0 or self.static_training_seed > 2**64 - 1:
            raise ValueError("static_training_seed must lie in [0, 2**64-1]")
        if self.static_training_seed == self.seed:
            raise ValueError("static_training_seed must differ from evaluation seed")
        total_samples = (
            self.calibration_windows * self.observation_samples_per_window
            + self.windows
            * (
                self.observation_samples_per_window
                + self.evaluation_samples_per_window
            )
        )
        if total_samples > 10_000_000:
            raise ValueError("configured workload must not exceed 10000000 samples")
        ratio = float(self.sigma_ratio_p)
        if not math.isfinite(ratio) or not 0.0 < ratio <= 1.0:
            raise ValueError("sigma_ratio_p must lie in (0, 1] for principal-axis alignment")
        object.__setattr__(self, "sigma_ratio_p", ratio)
        for name in (
            "before_mu_q_fraction",
            "before_mu_p_fraction",
            "after_mu_q_fraction",
            "after_mu_p_fraction",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or abs(value) >= 0.5:
                raise ValueError(f"{name} must be finite and lie in (-0.5, 0.5)")
            object.__setattr__(self, name, value)
        for name in ("before_sigma_fraction", "after_sigma_fraction"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.01 <= value <= 0.45:
                raise ValueError(f"{name} must lie in [0.01, 0.45]")
            object.__setattr__(self, name, value)
        for name in ("before_theta_deg", "after_theta_deg"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not -89.0 <= value <= 89.0:
                raise ValueError(f"{name} must lie in [-89, 89]")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class PredictionSnapshot:
    sigma: float
    mu_q: float
    mu_p: float
    theta_deg: float
    source: str
    covariance: tuple[tuple[float, float], tuple[float, float]]

    def covariance_array(self) -> np.ndarray:
        return np.asarray(self.covariance, dtype=float)


@dataclass(frozen=True)
class AlignmentWindowRecord:
    window_id: int
    regime: str
    truth_mu_q: float
    truth_mu_p: float
    truth_sigma_major: float
    truth_theta_deg: float
    window_prediction_used: PredictionSnapshot
    ekf_prediction_used: PredictionSnapshot
    standard_failures: int
    static_failures: int
    window_failures: int
    ekf_failures: int
    oracle_failures: int
    evaluation_samples: int
    observation_alias_rate: float
    evaluation_trace_sha256: str


@dataclass(frozen=True)
class AdaptiveAlignmentResult:
    config: AdaptiveAlignmentConfig
    static_prediction: PredictionSnapshot
    static_parameters: StaticMAPParameters
    static_training_state_sha256: str
    records: tuple[AlignmentWindowRecord, ...]
    standard_error_rate: float
    static_error_rate: float
    window_error_rate: float
    ekf_error_rate: float
    oracle_error_rate: float
    standard_gap: OracleGapMetrics
    window_gap: OracleGapMetrics
    ekf_gap: OracleGapMetrics
    primary_method: str
    static_oracle_gap_exploitable: bool
    primary_alignment_gate_passed: bool
    causal_delay_windows: int
    paired_samples: int
    trace_sha256: str
    comparison_method_ids: tuple[str, ...] = (
        STANDARD_BINNING_ID,
        STATIC_MAP_ID,
        "window_variance_map",
        "ekf_map",
        "full_state_model_oracle_map",
    )
    evidence_scope: str = "causal_synthetic_existing_baseline_alignment"


def _ellipse_covariance(sigma: float, sigma_ratio_p: float, theta_deg: float) -> np.ndarray:
    theta = math.radians(float(theta_deg))
    rotation = np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
        dtype=float,
    )
    covariance = rotation @ np.diag([sigma**2, (sigma * sigma_ratio_p) ** 2]) @ rotation.T
    return 0.5 * (covariance + covariance.T)


def _state_from_ellipse(
    *,
    mu_q: float,
    mu_p: float,
    sigma: float,
    sigma_ratio_p: float,
    theta_deg: float,
) -> DriftState:
    covariance = _ellipse_covariance(sigma, sigma_ratio_p, theta_deg)
    sigma_q = math.sqrt(float(covariance[0, 0]))
    sigma_p = math.sqrt(float(covariance[1, 1]))
    rho = float(covariance[0, 1] / (sigma_q * sigma_p))
    return DriftState(
        mu_q=mu_q,
        mu_p=mu_p,
        sigma_q=sigma_q,
        sigma_p=sigma_p,
        rho=rho,
    )


def _snapshot(
    prediction: NoisePrediction,
    sigma_ratio_p: float,
    *,
    source: Optional[str] = None,
) -> PredictionSnapshot:
    covariance = _ellipse_covariance(
        float(prediction.sigma),
        sigma_ratio_p,
        float(prediction.theta_deg),
    )
    return PredictionSnapshot(
        sigma=float(prediction.sigma),
        mu_q=float(prediction.mu_q),
        mu_p=float(prediction.mu_p),
        theta_deg=float(prediction.theta_deg),
        source=prediction.source if source is None else source,
        covariance=(
            (float(covariance[0, 0]), float(covariance[0, 1])),
            (float(covariance[1, 0]), float(covariance[1, 1])),
        ),
    )


def _histogram_and_alias_rate(
    samples: np.ndarray,
    *,
    bins: int,
) -> tuple[np.ndarray, float]:
    q_result = standard_binning_1d(samples[:, 0])
    p_result = standard_binning_1d(samples[:, 1])
    syndrome = np.column_stack((q_result.syndrome, p_result.syndrome))
    histogram, _, _ = np.histogram2d(
        syndrome[:, 0],
        syndrome[:, 1],
        bins=bins,
        range=[
            [-LATTICE_CONST / 2.0, LATTICE_CONST / 2.0],
            [-LATTICE_CONST / 2.0, LATTICE_CONST / 2.0],
        ],
    )
    alias_rate = float(
        np.mean(
            (np.asarray(q_result.lattice_index) != 0)
            | (np.asarray(p_result.lattice_index) != 0)
        )
    )
    return histogram, alias_rate


def _decode_failures(
    samples: np.ndarray,
    state: DriftState,
    static_parameters: StaticMAPParameters,
    window_prediction: PredictionSnapshot,
    ekf_prediction: PredictionSnapshot,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q_truth = standard_binning_1d(samples[:, 0])
    p_truth = standard_binning_1d(samples[:, 1])
    truth = 2 * np.asarray(q_truth.logical_parity) + np.asarray(p_truth.logical_parity)
    syndrome = np.column_stack((q_truth.syndrome, p_truth.syndrome))
    standard_decision, standard_truth, standard_failure = standard_binning_paired_outcomes(
        samples
    )
    if not np.array_equal(standard_truth, truth):
        raise AssertionError("standard-binning evaluator truth disagrees with alignment truth")
    if not np.all(standard_decision == 0):
        raise AssertionError("fixed half-cell recovery must select the central logical class")
    outputs: dict[str, list[np.ndarray]] = {
        "static": [],
        "window": [],
        "ekf": [],
        "oracle": [],
    }
    for start in range(0, len(samples), 2_000):
        chunk = syndrome[start : start + 2_000]
        outputs["static"].append(static_map_logical_class(chunk, static_parameters))
        outputs["window"].append(
            np.asarray(
                map_decode_2d(
                    chunk,
                    window_prediction.covariance_array(),
                    mean=(window_prediction.mu_q, window_prediction.mu_p),
                ).logical_class
            )
        )
        outputs["ekf"].append(
            np.asarray(
                map_decode_2d(
                    chunk,
                    ekf_prediction.covariance_array(),
                    mean=(ekf_prediction.mu_q, ekf_prediction.mu_p),
                ).logical_class
            )
        )
        outputs["oracle"].append(
            np.asarray(oracle_map_2d(chunk, state).logical_class)
        )
    decisions = {name: np.concatenate(chunks) for name, chunks in outputs.items()}
    return (
        standard_failure,
        decisions["static"] != truth,
        decisions["window"] != truth,
        decisions["ekf"] != truth,
        decisions["oracle"] != truth,
    )


def run_adaptive_drift_alignment(
    config: AdaptiveAlignmentConfig | None = None,
) -> AdaptiveAlignmentResult:
    """运行 pre-registered step-drift、one-window-delay、paired logical benchmark。"""

    if config is None:
        config = AdaptiveAlignmentConfig()
    if not isinstance(config, AdaptiveAlignmentConfig):
        raise TypeError("config must be AdaptiveAlignmentConfig")
    lam = LATTICE_CONST
    before = _state_from_ellipse(
        mu_q=config.before_mu_q_fraction * lam,
        mu_p=config.before_mu_p_fraction * lam,
        sigma=config.before_sigma_fraction * lam,
        sigma_ratio_p=config.sigma_ratio_p,
        theta_deg=config.before_theta_deg,
    )
    after = _state_from_ellipse(
        mu_q=config.after_mu_q_fraction * lam,
        mu_p=config.after_mu_p_fraction * lam,
        sigma=config.after_sigma_fraction * lam,
        sigma_ratio_p=config.sigma_ratio_p,
        theta_deg=config.after_theta_deg,
    )
    states = StepDriftProcess(
        before,
        after,
        change_step=config.change_step,
        seed=config.seed,
    ).generate(config.windows)
    static_training_states = StepDriftProcess(
        before,
        after,
        change_step=config.static_training_change_step,
        seed=config.static_training_seed,
    ).generate(config.static_training_windows)
    static_parameters = fit_static_map_from_training_states(
        static_training_states,
        training_protocol_id=(
            "t3.1.2-step-training-average-v1:"
            f"seed={config.static_training_seed}:windows={config.static_training_windows}:"
            f"change={config.static_training_change_step}"
        ),
    )
    window_config = WindowVarianceConfig(
        histogram_bins=config.histogram_bins,
        histogram_range_limit=lam / 2.0,
        sigma_clip=(0.03 * lam, 0.45 * lam),
        mu_clip=(-0.49 * lam, 0.49 * lam),
        theta_clip_deg=(-89.0, 89.0),
        theta_default_deg=0.0,
        min_anisotropy_ratio=0.02,
        sigma_ratio_p=config.sigma_ratio_p,
        measurement_var_floor=0.0,
    )
    window_baseline = WindowVarianceBaseline(window_config)
    rng = np.random.default_rng(config.seed)

    calibration_histogram = np.zeros(
        (config.histogram_bins, config.histogram_bins),
        dtype=float,
    )
    for _ in range(config.calibration_windows):
        calibration_samples, _ = sample_displacements(
            before,
            config.observation_samples_per_window,
            rng=rng,
        )
        histogram, _ = _histogram_and_alias_rate(
            calibration_samples,
            bins=config.histogram_bins,
        )
        calibration_histogram += histogram
    calibration_raw = window_baseline.predict(calibration_histogram, window_id=-1)
    calibration_prediction = _snapshot(
        calibration_raw,
        config.sigma_ratio_p,
        source="static_calibration_window_variance",
    )
    static_covariance = static_parameters.covariance_array()
    eigenvalues, eigenvectors = np.linalg.eigh(static_covariance)
    major_index = int(np.argmax(eigenvalues))
    major_vector = eigenvectors[:, major_index]
    static_theta_deg = math.degrees(
        math.atan2(float(major_vector[1]), float(major_vector[0]))
    )
    if static_theta_deg >= 90.0:
        static_theta_deg -= 180.0
    if static_theta_deg < -90.0:
        static_theta_deg += 180.0
    static_prediction = PredictionSnapshot(
        sigma=math.sqrt(float(eigenvalues[major_index])),
        mu_q=static_parameters.mean[0],
        mu_p=static_parameters.mean[1],
        theta_deg=static_theta_deg,
        source=STATIC_MAP_ID,
        covariance=static_parameters.covariance,
    )
    initial_raw = NoisePrediction(
        sigma=calibration_prediction.sigma,
        mu_q=calibration_prediction.mu_q,
        mu_p=calibration_prediction.mu_p,
        theta_deg=calibration_prediction.theta_deg,
        source="causal_initial_calibration",
    )
    window_prediction_used = _snapshot(initial_raw, config.sigma_ratio_p)
    ekf_prediction_used = _snapshot(initial_raw, config.sigma_ratio_p)
    ekf_config = EKFBaselineConfig(
        sigma_clip=window_config.sigma_clip,
        mu_clip=window_config.mu_clip,
        theta_clip_deg=window_config.theta_clip_deg,
        initial_sigma=calibration_prediction.sigma,
        initial_mu_q=calibration_prediction.mu_q,
        initial_mu_p=calibration_prediction.mu_p,
        initial_theta_deg=calibration_prediction.theta_deg,
        process_std_sigma=0.03 * lam,
        process_std_mu_q=0.04 * lam,
        process_std_mu_p=0.04 * lam,
        process_std_theta_deg=4.0,
        measurement_std_sigma=0.08 * lam,
        measurement_std_mu_q=0.08 * lam,
        measurement_std_mu_p=0.08 * lam,
        measurement_std_theta_deg=8.0,
        covariance_floor=1.0e-8,
    )
    ekf_baseline = EKFBaseline(
        ekf_config,
        measurement_baseline=WindowVarianceBaseline(window_config),
    )

    standard_failures: list[np.ndarray] = []
    static_failures: list[np.ndarray] = []
    window_failures: list[np.ndarray] = []
    ekf_failures: list[np.ndarray] = []
    oracle_failures: list[np.ndarray] = []
    records: list[AlignmentWindowRecord] = []
    trace_hasher = hashlib.sha256()
    trace_hasher.update(b"T1.3.4-paired-evaluation-trace-v1\0")

    for window_id, state in enumerate(states):
        # 决策只使用 calibration 或上一窗口已完成的 prediction；当前窗口 histogram
        # 在本窗口评估后才生成，严格形成一窗口因果延迟。
        evaluation_samples, _ = sample_displacements(
            state,
            config.evaluation_samples_per_window,
            rng=rng,
        )
        # Fingerprint the single materialized evaluation trace before any decoder
        # consumes it.  All four failure arrays below are derived from this exact
        # buffer, rather than from same-seed reruns that may silently diverge.
        window_trace_hasher = hashlib.sha256()
        window_trace_hasher.update(
            int(window_id).to_bytes(8, byteorder="little", signed=False)
        )
        window_trace_hasher.update(state.regime.encode("utf-8"))
        window_trace_hasher.update(b"\0")
        evaluation_bytes = np.ascontiguousarray(
            evaluation_samples,
            dtype="<f8",
        ).tobytes()
        window_trace_hasher.update(evaluation_bytes)
        window_trace_sha256 = window_trace_hasher.hexdigest()
        trace_hasher.update(bytes.fromhex(window_trace_sha256))
        failures = _decode_failures(
            evaluation_samples,
            state,
            static_parameters,
            window_prediction_used,
            ekf_prediction_used,
        )
        for destination, values in zip(
            (
                standard_failures,
                static_failures,
                window_failures,
                ekf_failures,
                oracle_failures,
            ),
            failures,
        ):
            destination.append(values)

        observation_samples, _ = sample_displacements(
            state,
            config.observation_samples_per_window,
            rng=rng,
        )
        histogram, alias_rate = _histogram_and_alias_rate(
            observation_samples,
            bins=config.histogram_bins,
        )
        next_window_raw = window_baseline.predict(histogram, window_id=window_id)
        next_ekf_raw = ekf_baseline.predict(histogram, window_id=window_id)

        truth_angle = math.degrees(state.principal_angle)
        truth_eigenvalues = np.linalg.eigvalsh(state.covariance)
        records.append(
            AlignmentWindowRecord(
                window_id=window_id,
                regime=state.regime,
                truth_mu_q=state.mu_q,
                truth_mu_p=state.mu_p,
                truth_sigma_major=math.sqrt(float(np.max(truth_eigenvalues))),
                truth_theta_deg=truth_angle,
                window_prediction_used=window_prediction_used,
                ekf_prediction_used=ekf_prediction_used,
                standard_failures=int(np.sum(failures[0])),
                static_failures=int(np.sum(failures[1])),
                window_failures=int(np.sum(failures[2])),
                ekf_failures=int(np.sum(failures[3])),
                oracle_failures=int(np.sum(failures[4])),
                evaluation_samples=config.evaluation_samples_per_window,
                observation_alias_rate=alias_rate,
                evaluation_trace_sha256=window_trace_sha256,
            )
        )
        window_prediction_used = _snapshot(next_window_raw, config.sigma_ratio_p)
        ekf_prediction_used = _snapshot(next_ekf_raw, config.sigma_ratio_p)

    standard_array = np.concatenate(standard_failures)
    static_array = np.concatenate(static_failures)
    window_array = np.concatenate(window_failures)
    ekf_array = np.concatenate(ekf_failures)
    oracle_array = np.concatenate(oracle_failures)
    standard_gap = compute_oracle_gap_metrics(
        standard_array,
        static_array,
        oracle_array,
        bootstrap_replicates=config.bootstrap_replicates,
        seed=config.seed + 3,
    )
    window_gap = compute_oracle_gap_metrics(
        static_array,
        window_array,
        oracle_array,
        bootstrap_replicates=config.bootstrap_replicates,
        seed=config.seed + 1,
    )
    ekf_gap = compute_oracle_gap_metrics(
        static_array,
        ekf_array,
        oracle_array,
        bootstrap_replicates=config.bootstrap_replicates,
        seed=config.seed + 2,
    )
    exploitable = (
        window_gap.static_minus_oracle.estimate > 0.01
        and window_gap.static_minus_oracle.ci_low > 0.0
        and window_gap.denominator_stable
    )
    primary_gate = (
        exploitable
        and ekf_gap.static_minus_dual.ci_low > 0.0
        and ekf_gap.gap_closed_fraction is not None
        and ekf_gap.gap_closed_fraction > 0.0
        and ekf_gap.ratio_ci_reliable
    )
    return AdaptiveAlignmentResult(
        config=config,
        static_prediction=static_prediction,
        static_parameters=static_parameters,
        static_training_state_sha256=static_parameters.training_state_sha256,
        records=tuple(records),
        standard_error_rate=float(np.mean(standard_array)),
        static_error_rate=float(np.mean(static_array)),
        window_error_rate=float(np.mean(window_array)),
        ekf_error_rate=float(np.mean(ekf_array)),
        oracle_error_rate=float(np.mean(oracle_array)),
        standard_gap=standard_gap,
        window_gap=window_gap,
        ekf_gap=ekf_gap,
        primary_method="ekf_baseline",
        static_oracle_gap_exploitable=exploitable,
        primary_alignment_gate_passed=primary_gate,
        causal_delay_windows=1,
        paired_samples=int(static_array.size),
        trace_sha256=trace_hasher.hexdigest(),
    )


__all__ = [
    "AdaptiveAlignmentConfig",
    "PredictionSnapshot",
    "AlignmentWindowRecord",
    "AdaptiveAlignmentResult",
    "run_adaptive_drift_alignment",
]
