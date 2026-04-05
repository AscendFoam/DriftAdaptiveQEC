"""Fast-loop runtime emulation with physical-noise feedback and window accumulation."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, Optional

import numpy as np

from cnn_fpga.decoder.param_mapper import analyze_decoder_aggressiveness
from physics.gkp_state import LATTICE_CONST
from physics.logical_tracking import LogicalErrorTracker
from physics.syndrome_measurement import MeasurementConfig, RealisticSyndromeMeasurement

from cnn_fpga.decoder.linear_runtime import LinearRuntime
from cnn_fpga.runtime.param_bank import ParamBank


NoiseProvider = Callable[[int, float], Dict[str, Any]]


@dataclass(frozen=True)
class FastLoopConfig:
    """Configuration of the fast-loop emulator."""

    window_size: int = 2048
    histogram_bins: int = 32
    syndrome_limit: float = LATTICE_CONST / 2.0
    histogram_range_limit: float = LATTICE_CONST / 2.0
    sigma_measurement: float = 0.05
    sigma_ratio_p: float = 1.0
    use_full_qec_model: bool = True
    overflow_alert_ratio: float = 0.05
    correction_limit: float = LATTICE_CONST
    gain_upper_bound: float = 1.2
    aggressive_gain_ratio: float = 0.9
    aggressive_bias_ratio: float = 0.25
    aggressive_correction_ratio: float = 0.7
    delta: float = 0.3
    measurement_efficiency: float = 0.95
    ancilla_error_rate: float = 0.01
    add_shot_noise: bool = True

    def __post_init__(self) -> None:
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.histogram_bins <= 0:
            raise ValueError("histogram_bins must be positive")
        if self.syndrome_limit <= 0:
            raise ValueError("syndrome_limit must be positive")
        if self.histogram_range_limit <= 0:
            raise ValueError("histogram_range_limit must be positive")
        if self.correction_limit <= 0:
            raise ValueError("correction_limit must be positive")
        if self.sigma_measurement < 0:
            raise ValueError("sigma_measurement must be non-negative")
        if self.sigma_ratio_p < 0:
            raise ValueError("sigma_ratio_p must be non-negative")
        if self.gain_upper_bound <= 0:
            raise ValueError("gain_upper_bound must be positive")
        if self.aggressive_gain_ratio < 0 or self.aggressive_bias_ratio < 0 or self.aggressive_correction_ratio < 0:
            raise ValueError("aggressive ratios must be non-negative")

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FastLoopConfig":
        hardware = config.get("hardware_defaults", {})
        runtime_cfg = config.get("runtime", {})
        fast_cfg = config.get("fast_loop", {})
        measurement_cfg = config.get("measurement", {})
        model_cfg = config.get("model", {})
        mapping_cfg = config.get("param_mapping", {})
        gain_clip_raw = mapping_cfg.get("gain_clip", [0.2, 1.2])
        histogram_range_limit = float(fast_cfg.get("histogram_range_limit", fast_cfg.get("syndrome_limit", LATTICE_CONST / 2.0)))
        return cls(
            window_size=int(runtime_cfg.get("window_size", hardware.get("window_size", 2048))),
            histogram_bins=int(fast_cfg.get("histogram_bins", 32)),
            syndrome_limit=float(fast_cfg.get("syndrome_limit", LATTICE_CONST / 2.0)),
            histogram_range_limit=histogram_range_limit,
            sigma_measurement=float(model_cfg.get("sigma_measurement", fast_cfg.get("sigma_measurement", 0.05))),
            sigma_ratio_p=float(model_cfg.get("sigma_ratio_p", 1.0)),
            use_full_qec_model=bool(fast_cfg.get("use_full_qec_model", True)),
            overflow_alert_ratio=float(fast_cfg.get("overflow_alert_ratio", 0.05)),
            correction_limit=float(fast_cfg.get("correction_limit", LATTICE_CONST)),
            gain_upper_bound=float(gain_clip_raw[1]),
            aggressive_gain_ratio=float(mapping_cfg.get("aggressive_gain_ratio", 0.9)),
            aggressive_bias_ratio=float(mapping_cfg.get("aggressive_bias_ratio", 0.25)),
            aggressive_correction_ratio=float(mapping_cfg.get("aggressive_correction_ratio", 0.7)),
            delta=float(measurement_cfg.get("delta", 0.3)),
            measurement_efficiency=float(measurement_cfg.get("measurement_efficiency", 0.95)),
            ancilla_error_rate=float(measurement_cfg.get("ancilla_error_rate", 0.01)),
            add_shot_noise=bool(measurement_cfg.get("add_shot_noise", True)),
        )


@dataclass(frozen=True)
class NoiseState:
    """Instantaneous physical-noise state for one fast cycle."""

    sigma: float
    mu_q: float
    mu_p: float
    theta_deg: float
    metadata: Dict[str, Any]

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "NoiseState":
        return cls(
            sigma=max(0.0, float(payload.get("sigma", 0.25))),
            mu_q=float(payload.get("mu_q", 0.0)),
            mu_p=float(payload.get("mu_p", 0.0)),
            theta_deg=float(payload.get("theta_deg", 0.0)),
            metadata=dict(payload.get("metadata", {})),
        )

    def to_target_dict(self) -> Dict[str, float]:
        return {
            "sigma": self.sigma,
            "mu_q": self.mu_q,
            "mu_p": self.mu_p,
            "theta_deg": self.theta_deg,
        }


@dataclass
class FastCycleRecord:
    """Per-cycle record produced by the fast-loop emulator."""

    epoch_id: int
    time_us: float
    active_version: int
    active_bank: str
    noise_state: NoiseState
    total_error: np.ndarray
    syndrome_raw: np.ndarray
    syndrome_used: np.ndarray
    syndrome_saturated: np.ndarray
    histogram_input_saturated: np.ndarray
    correction: np.ndarray
    correction_saturated: np.ndarray
    correction_clip_saturated: np.ndarray
    correction_fixed_point_saturated: np.ndarray
    param_aggressive: bool
    param_aggressive_gain_flag: bool
    param_aggressive_bias_flag: bool
    param_aggressive_correction_flag: bool
    param_max_gain: float
    param_bias_norm: float
    correction_utilization: float
    wrapped_residual: np.ndarray
    overflow: bool
    overflow_any: bool
    x_error: bool
    z_error: bool


def _dominant_source(hist_ratio: float, corr_ratio: float, aggressive_ratio: float) -> str:
    candidates = [
        ("histogram_input", float(hist_ratio)),
        ("correction_saturation", float(corr_ratio)),
        ("aggressive_param", float(aggressive_ratio)),
    ]
    ordered = sorted(candidates, key=lambda item: item[1], reverse=True)
    if ordered[0][1] <= 0.0:
        return "none"
    if len(ordered) > 1 and abs(ordered[0][1] - ordered[1][1]) <= 1.0e-3:
        return f"mixed:{ordered[0][0]}+{ordered[1][0]}"
    return ordered[0][0]


class FastLoopEmulator:
    """Drive closed-loop fast-cycle evolution and build slow-loop windows."""

    def __init__(
        self,
        config: FastLoopConfig,
        *,
        param_bank: ParamBank,
        linear_runtime: Optional[LinearRuntime] = None,
        noise_provider: Optional[NoiseProvider] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.config = config
        self.param_bank = param_bank
        self.linear_runtime = linear_runtime or LinearRuntime.from_config({})
        self.noise_provider = noise_provider or (lambda _epoch_id, _time_us: {})
        self._rng = np.random.default_rng(seed)

        self.measurement = RealisticSyndromeMeasurement(
            MeasurementConfig(
                delta=config.delta,
                measurement_efficiency=config.measurement_efficiency,
                ancilla_error_rate=config.ancilla_error_rate,
                add_shot_noise=config.add_shot_noise,
            )
        )
        self.tracker = LogicalErrorTracker()
        self._window_records: Deque[FastCycleRecord] = deque(maxlen=config.window_size)
        self._cumulative_residual = np.zeros(2, dtype=float)
        self._total_cycles = 0
        # 中文注释：为兼容旧报告，`_overflow_count` 仍保留“直方图输入超范围”这一原始定义。
        self._overflow_count = 0
        self._overflow_any_count = 0
        self._histogram_sat_q_count = 0
        self._histogram_sat_p_count = 0
        self._correction_sat_count = 0
        self._correction_sat_q_count = 0
        self._correction_sat_p_count = 0
        self._correction_clip_sat_count = 0
        self._correction_fixed_point_sat_count = 0
        self._aggressive_param_count = 0
        self._aggressive_gain_count = 0
        self._aggressive_bias_count = 0
        self._aggressive_correction_count = 0
        self._aggressive_with_histogram_sat_count = 0
        self._aggressive_with_correction_sat_count = 0
        self._latest_record: Optional[FastCycleRecord] = None
        self._cumulative_ler_curve: list[float] = []

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        *,
        param_bank: ParamBank,
        noise_provider: Optional[NoiseProvider] = None,
        seed: Optional[int] = None,
    ) -> "FastLoopEmulator":
        return cls(
            FastLoopConfig.from_config(config),
            param_bank=param_bank,
            linear_runtime=LinearRuntime.from_config(config),
            noise_provider=noise_provider,
            seed=seed,
        )

    @property
    def total_cycles(self) -> int:
        return self._total_cycles

    @property
    def latest_record(self) -> Optional[FastCycleRecord]:
        return self._latest_record

    def _wrap(self, value: np.ndarray) -> np.ndarray:
        return np.mod(value + LATTICE_CONST / 2.0, LATTICE_CONST) - LATTICE_CONST / 2.0

    def _sample_noise_state(self, epoch_id: int, time_us: float) -> NoiseState:
        raw = self.noise_provider(epoch_id, time_us)
        if not isinstance(raw, dict):
            raise TypeError("noise_provider must return a dict payload")
        return NoiseState.from_payload(raw)

    def _sample_new_error(self, state: NoiseState) -> np.ndarray:
        theta = np.deg2rad(state.theta_deg)
        rotation = np.array(
            [
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],
            ],
            dtype=float,
        )
        base_noise = np.array(
            [
                self._rng.normal(0.0, state.sigma),
                self._rng.normal(0.0, state.sigma * self.config.sigma_ratio_p),
            ],
            dtype=float,
        )
        return np.array([state.mu_q, state.mu_p], dtype=float) + rotation @ base_noise

    def _measure_syndrome(self, total_error: np.ndarray) -> np.ndarray:
        if self.config.use_full_qec_model:
            syndrome = self.measurement.measure(total_error, add_noise=True)
        else:
            syndrome = self._wrap(total_error)
        if self.config.sigma_measurement > 0:
            syndrome = syndrome + self._rng.normal(0.0, self.config.sigma_measurement, size=2)
        return syndrome.astype(float)

    def _build_window_payload(self) -> Dict[str, Any]:
        records = list(self._window_records)
        if len(records) < self.config.window_size:
            return {
                "histogram": np.zeros((self.config.histogram_bins, self.config.histogram_bins), dtype=np.float32),
                "target_params": {"sigma": 0.0, "mu_q": 0.0, "mu_p": 0.0, "theta_deg": 0.0},
                "diagnostics": {"valid_window": False},
            }

        syndromes = np.vstack([record.syndrome_used for record in records])
        hist, _, _ = np.histogram2d(
            syndromes[:, 0],
            syndromes[:, 1],
            bins=self.config.histogram_bins,
            range=[
                [-self.config.histogram_range_limit, self.config.histogram_range_limit],
                [-self.config.histogram_range_limit, self.config.histogram_range_limit],
            ],
        )
        hist_sum = float(hist.sum())
        if hist_sum > 0.0:
            hist = hist / hist_sum

        target_array = np.array(
            [[r.noise_state.sigma, r.noise_state.mu_q, r.noise_state.mu_p, r.noise_state.theta_deg] for r in records],
            dtype=float,
        )
        correction_array = np.vstack([record.correction for record in records])
        residual_array = np.vstack([record.wrapped_residual for record in records])
        overflow_ratio = float(np.mean([record.overflow for record in records]))
        decoder_input_sat_q_ratio = float(np.mean([record.syndrome_saturated[0] for record in records]))
        decoder_input_sat_p_ratio = float(np.mean([record.syndrome_saturated[1] for record in records]))
        decoder_input_sat_ratio = float(np.mean([np.any(record.syndrome_saturated) for record in records]))
        overflow_any_ratio = float(np.mean([record.overflow_any for record in records]))
        histogram_sat_q_ratio = float(np.mean([record.histogram_input_saturated[0] for record in records]))
        histogram_sat_p_ratio = float(np.mean([record.histogram_input_saturated[1] for record in records]))
        histogram_sat_ratio = float(np.mean([np.any(record.histogram_input_saturated) for record in records]))
        correction_sat_q_ratio = float(np.mean([record.correction_saturated[0] for record in records]))
        correction_sat_p_ratio = float(np.mean([record.correction_saturated[1] for record in records]))
        correction_sat_ratio = float(np.mean([np.any(record.correction_saturated) for record in records]))
        correction_clip_sat_ratio = float(np.mean([np.any(record.correction_clip_saturated) for record in records]))
        correction_fixed_point_sat_ratio = float(
            np.mean([np.any(record.correction_fixed_point_saturated) for record in records])
        )
        aggressive_param_ratio = float(np.mean([record.param_aggressive for record in records]))
        aggressive_gain_ratio = float(np.mean([record.param_aggressive_gain_flag for record in records]))
        aggressive_bias_ratio = float(np.mean([record.param_aggressive_bias_flag for record in records]))
        aggressive_correction_ratio = float(np.mean([record.param_aggressive_correction_flag for record in records]))
        aggressive_with_hist_ratio = float(
            np.mean([record.param_aggressive and np.any(record.syndrome_saturated) for record in records])
        )
        aggressive_with_corr_ratio = float(
            np.mean([record.param_aggressive and np.any(record.correction_saturated) for record in records])
        )
        logical_flags = np.array([record.x_error or record.z_error for record in records], dtype=float)
        version_counts = Counter(str(record.active_version) for record in records)

        return {
            "histogram": hist.astype(np.float32),
            "target_params": {
                "sigma": float(np.mean(target_array[:, 0])),
                "mu_q": float(np.mean(target_array[:, 1])),
                "mu_p": float(np.mean(target_array[:, 2])),
                "theta_deg": float(np.mean(target_array[:, 3])),
            },
            "diagnostics": {
                "valid_window": True,
                "overflow_ratio": overflow_ratio,
                "overflow_any_ratio": overflow_any_ratio,
                "overflow_alert": bool(overflow_ratio > self.config.overflow_alert_ratio),
                "histogram_input_saturation_ratio": histogram_sat_ratio,
                "histogram_input_saturation_q_ratio": histogram_sat_q_ratio,
                "histogram_input_saturation_p_ratio": histogram_sat_p_ratio,
                "decoder_input_saturation_ratio": decoder_input_sat_ratio,
                "decoder_input_saturation_q_ratio": decoder_input_sat_q_ratio,
                "decoder_input_saturation_p_ratio": decoder_input_sat_p_ratio,
                "correction_saturation_ratio": correction_sat_ratio,
                "correction_saturation_q_ratio": correction_sat_q_ratio,
                "correction_saturation_p_ratio": correction_sat_p_ratio,
                "correction_clip_saturation_ratio": correction_clip_sat_ratio,
                "correction_fixed_point_saturation_ratio": correction_fixed_point_sat_ratio,
                "aggressive_param_ratio": aggressive_param_ratio,
                "aggressive_param_gain_ratio": aggressive_gain_ratio,
                "aggressive_param_bias_ratio": aggressive_bias_ratio,
                "aggressive_param_correction_ratio": aggressive_correction_ratio,
                "aggressive_with_histogram_saturation_ratio": aggressive_with_hist_ratio,
                "aggressive_with_correction_saturation_ratio": aggressive_with_corr_ratio,
                "dominant_overflow_source": _dominant_source(
                    histogram_sat_ratio,
                    correction_sat_ratio,
                    aggressive_param_ratio,
                ),
                "window_ler": float(np.mean(logical_flags)),
                "mean_abs_residual_q": float(np.mean(np.abs(residual_array[:, 0]))),
                "mean_abs_residual_p": float(np.mean(np.abs(residual_array[:, 1]))),
                "mean_correction_norm": float(np.mean(np.linalg.norm(correction_array, axis=1))),
                "mean_correction_utilization": float(np.mean([record.correction_utilization for record in records])),
                "mean_active_param_max_gain": float(np.mean([record.param_max_gain for record in records])),
                "mean_active_param_bias_norm": float(np.mean([record.param_bias_norm for record in records])),
                "active_version_counts": dict(version_counts),
            },
            "window_stats": {
                "n_cycles": len(records),
                "epoch_start": records[0].epoch_id,
                "epoch_end": records[-1].epoch_id,
                "mean_syndrome_q": float(np.mean(syndromes[:, 0])),
                "mean_syndrome_p": float(np.mean(syndromes[:, 1])),
                "std_syndrome_q": float(np.std(syndromes[:, 0])),
                "std_syndrome_p": float(np.std(syndromes[:, 1])),
            },
        }

    def step(self, epoch_id: int, time_us: float, emit_window: bool = False) -> Optional[Dict[str, Any]]:
        """Advance one fast cycle and optionally emit a slow-loop window payload."""
        noise_state = self._sample_noise_state(epoch_id, time_us)
        new_error = self._sample_new_error(noise_state)
        if self.config.use_full_qec_model:
            total_error = self._cumulative_residual + new_error
        else:
            total_error = new_error

        syndrome_raw = self._measure_syndrome(total_error)
        params = self.param_bank.read_active()
        decode_result = self.linear_runtime.decode(syndrome_raw, params)
        correction = decode_result.correction_applied
        aggressiveness = analyze_decoder_aggressiveness(
            decode_result.params_used.K,
            decode_result.params_used.b,
            gain_upper_bound=self.config.gain_upper_bound,
            correction_limit=self.config.correction_limit,
            aggressive_gain_ratio=self.config.aggressive_gain_ratio,
            aggressive_bias_ratio=self.config.aggressive_bias_ratio,
        )
        correction_utilization = float(
            np.max(np.abs(correction)) / max(1.0e-8, self.config.correction_limit)
        )
        correction_aggressive = bool(correction_utilization >= self.config.aggressive_correction_ratio)
        param_aggressive = bool(aggressiveness["aggressive"] or correction_aggressive)
        x_error, z_error = self.tracker.update(
            total_error[0],
            total_error[1],
            correction[0],
            correction[1],
        )
        wrapped_residual = np.array([self.tracker.accumulated_q, self.tracker.accumulated_p], dtype=float)
        self._cumulative_residual = wrapped_residual.copy()

        histogram_input_saturated = np.abs(syndrome_raw) > self.config.histogram_range_limit
        overflow = bool(np.any(histogram_input_saturated))
        correction_overflow = bool(np.any(decode_result.correction_saturated))
        overflow_any = bool(overflow or correction_overflow)
        if overflow:
            self._overflow_count += 1
        if overflow_any:
            self._overflow_any_count += 1
        self._histogram_sat_q_count += int(bool(histogram_input_saturated[0]))
        self._histogram_sat_p_count += int(bool(histogram_input_saturated[1]))
        self._correction_sat_count += int(correction_overflow)
        self._correction_sat_q_count += int(bool(decode_result.correction_saturated[0]))
        self._correction_sat_p_count += int(bool(decode_result.correction_saturated[1]))
        self._correction_clip_sat_count += int(bool(np.any(decode_result.correction_clip_saturated)))
        self._correction_fixed_point_sat_count += int(bool(np.any(decode_result.correction_fixed_point_saturated)))
        self._aggressive_param_count += int(param_aggressive)
        self._aggressive_gain_count += int(bool(aggressiveness["gain_flag"]))
        self._aggressive_bias_count += int(bool(aggressiveness["bias_flag"]))
        self._aggressive_correction_count += int(correction_aggressive)
        if param_aggressive and overflow:
            self._aggressive_with_histogram_sat_count += 1
        if param_aggressive and correction_overflow:
            self._aggressive_with_correction_sat_count += 1

        record = FastCycleRecord(
            epoch_id=epoch_id,
            time_us=time_us,
            active_version=self.param_bank.active_version,
            active_bank=self.param_bank.active_bank_name,
            noise_state=noise_state,
            total_error=total_error.copy(),
            syndrome_raw=syndrome_raw.copy(),
            syndrome_used=decode_result.syndrome_used.copy(),
            syndrome_saturated=decode_result.syndrome_saturated.copy(),
            histogram_input_saturated=histogram_input_saturated.copy(),
            correction=correction.copy(),
            correction_saturated=decode_result.correction_saturated.copy(),
            correction_clip_saturated=decode_result.correction_clip_saturated.copy(),
            correction_fixed_point_saturated=decode_result.correction_fixed_point_saturated.copy(),
            param_aggressive=param_aggressive,
            param_aggressive_gain_flag=bool(aggressiveness["gain_flag"]),
            param_aggressive_bias_flag=bool(aggressiveness["bias_flag"]),
            param_aggressive_correction_flag=correction_aggressive,
            param_max_gain=float(aggressiveness["max_gain"]),
            param_bias_norm=float(aggressiveness["bias_norm"]),
            correction_utilization=correction_utilization,
            wrapped_residual=wrapped_residual.copy(),
            overflow=overflow,
            overflow_any=overflow_any,
            x_error=x_error,
            z_error=z_error,
        )
        self._window_records.append(record)
        self._latest_record = record
        self._total_cycles += 1
        self._cumulative_ler_curve.append(self.tracker.get_logical_error_rate())

        if not emit_window:
            return None
        return self._build_window_payload()

    def __call__(self, epoch_id: int, time_us: float, emit_window: bool = False) -> Optional[Dict[str, Any]]:
        return self.step(epoch_id, time_us, emit_window=emit_window)

    def summary(self) -> Dict[str, Any]:
        """Return high-level diagnostics of the fast-loop run."""
        tracker_stats = self.tracker.get_statistics()
        n_cycles = max(1, self._total_cycles)
        histogram_sat_ratio = float(self._overflow_count / n_cycles)
        correction_sat_ratio = float(self._correction_sat_count / n_cycles)
        aggressive_param_ratio = float(self._aggressive_param_count / n_cycles)
        return {
            "n_fast_cycles": self._total_cycles,
            "overflow_count": self._overflow_count,
            "overflow_rate": histogram_sat_ratio,
            "overflow_any_count": self._overflow_any_count,
            "overflow_any_rate": float(self._overflow_any_count / n_cycles),
            "histogram_input_saturation_count": self._overflow_count,
            "histogram_input_saturation_rate": histogram_sat_ratio,
            "histogram_input_saturation_q_rate": float(self._histogram_sat_q_count / n_cycles),
            "histogram_input_saturation_p_rate": float(self._histogram_sat_p_count / n_cycles),
            "correction_saturation_count": self._correction_sat_count,
            "correction_saturation_rate": correction_sat_ratio,
            "correction_saturation_q_rate": float(self._correction_sat_q_count / n_cycles),
            "correction_saturation_p_rate": float(self._correction_sat_p_count / n_cycles),
            "correction_clip_saturation_rate": float(self._correction_clip_sat_count / n_cycles),
            "correction_fixed_point_saturation_rate": float(self._correction_fixed_point_sat_count / n_cycles),
            "aggressive_param_count": self._aggressive_param_count,
            "aggressive_param_rate": aggressive_param_ratio,
            "aggressive_param_gain_rate": float(self._aggressive_gain_count / n_cycles),
            "aggressive_param_bias_rate": float(self._aggressive_bias_count / n_cycles),
            "aggressive_param_correction_rate": float(self._aggressive_correction_count / n_cycles),
            "aggressive_with_histogram_saturation_rate": float(
                self._aggressive_with_histogram_sat_count / n_cycles
            ),
            "aggressive_with_correction_saturation_rate": float(
                self._aggressive_with_correction_sat_count / n_cycles
            ),
            "dominant_overflow_source": _dominant_source(
                histogram_sat_ratio,
                correction_sat_ratio,
                aggressive_param_ratio,
            ),
            "final_logical_error_rate": float(self._cumulative_ler_curve[-1]) if self._cumulative_ler_curve else 0.0,
            "tracker": tracker_stats,
            "latest_record": None
            if self._latest_record is None
            else {
                "epoch_id": self._latest_record.epoch_id,
                "active_version": self._latest_record.active_version,
                "noise_state": self._latest_record.noise_state.to_target_dict(),
                "wrapped_residual": self._latest_record.wrapped_residual.tolist(),
                "syndrome_saturated": self._latest_record.syndrome_saturated.astype(int).tolist(),
                "histogram_input_saturated": self._latest_record.histogram_input_saturated.astype(int).tolist(),
                "correction_saturated": self._latest_record.correction_saturated.astype(int).tolist(),
                "param_aggressive": bool(self._latest_record.param_aggressive),
            },
        }

    def get_ler_curve(self) -> np.ndarray:
        return np.asarray(self._cumulative_ler_curve, dtype=float)
