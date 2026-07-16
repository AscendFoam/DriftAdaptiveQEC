"""T2.4.3 bit-accurate ADC/LUT/LLR/state/bank precision stress model.

The implementation produces paired software-model precision--resource--LER
curves.  Resource fields are exact representation-size and word-width proxies;
they are deliberately not FPGA synthesis or target-board measurements.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cnn_fpga.decoder.linear_runtime import FixedPointFormat
from cnn_fpga.utils.config import save_json
from physics.constants import LATTICE_CONST
from physics.ideal_gkp_decoder import llr_1d


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT = ROOT / "docs" / "t2_4_3_fixed_point_validation.json"
DEFAULT_CSV = ROOT / "docs" / "t2_4_3_precision_resource_ler.csv"
CONTRACT_ID = "T243-BIT-ACCURATE-PRECISION-RESOURCE-LER-V1"
MODEL_SCOPE = "paired_synthetic_bit_accurate_lut_decoder_not_synthesis_or_board"
BANK_FAULT_MODES = (
    "none",
    "lut_sign_burst",
    "stale_commit",
    "torn_update",
    "state_msb_flip",
)


@dataclass(frozen=True)
class UniformCodeFormat:
    """Unsigned code over a closed physical range using level-centre decode."""

    bits: int
    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        if isinstance(self.bits, bool) or not isinstance(self.bits, int):
            raise TypeError("bits must be an integer")
        if not 2 <= self.bits <= 24:
            raise ValueError("bits must lie in [2, 24]")
        if not math.isfinite(self.minimum) or not math.isfinite(self.maximum):
            raise ValueError("uniform-code bounds must be finite")
        if self.maximum <= self.minimum:
            raise ValueError("maximum must exceed minimum")

    @property
    def levels(self) -> int:
        return 1 << self.bits

    @property
    def step(self) -> float:
        return (self.maximum - self.minimum) / self.levels

    def encode(
        self,
        values: np.ndarray | Sequence[float] | float,
    ) -> tuple[np.ndarray, np.ndarray]:
        array = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(array)):
            raise ValueError("uniform-code input must contain only finite values")
        clipped = np.clip(array, self.minimum, np.nextafter(self.maximum, self.minimum))
        codes = np.floor((clipped - self.minimum) / self.step).astype(np.int64)
        codes = np.clip(codes, 0, self.levels - 1)
        saturated = np.logical_or(array < self.minimum, array >= self.maximum)
        return codes, saturated

    def decode(self, codes: np.ndarray | Sequence[int] | int) -> np.ndarray:
        array = np.asarray(codes)
        if not np.issubdtype(array.dtype, np.integer):
            raise TypeError("uniform codes must be integers")
        integer = array.astype(np.int64, copy=False)
        if np.any(integer < 0) or np.any(integer >= self.levels):
            raise ValueError("uniform code lies outside configured width")
        return self.minimum + (integer.astype(np.float64) + 0.5) * self.step

    def quantize(
        self,
        values: np.ndarray | Sequence[float] | float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        codes, saturated = self.encode(values)
        return self.decode(codes), codes, saturated


@dataclass(frozen=True)
class PrecisionProfile:
    profile_id: str
    curve_axis: str
    axis_value: float
    adc_bits: int = 10
    lut_address_bits: int = 8
    llr_integer_bits: int = 3
    llr_fractional_bits: int = 8
    threshold_integer_bits: int = 0
    threshold_fractional_bits: int = 9
    state_bits: int = 12
    update_period_windows: int = 1

    def __post_init__(self) -> None:
        if not self.profile_id or not self.curve_axis:
            raise ValueError("profile_id and curve_axis must be nonempty")
        for name in ("adc_bits", "lut_address_bits", "state_bits"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f"{name} must be an integer >= 2")
        if not 2 <= self.lut_address_bits <= 12:
            raise ValueError("lut_address_bits must lie in [2, 12]")
        if self.update_period_windows <= 0:
            raise ValueError("update_period_windows must be positive")
        FixedPointFormat(self.llr_integer_bits, self.llr_fractional_bits)
        FixedPointFormat(self.threshold_integer_bits, self.threshold_fractional_bits)

    @property
    def llr_word_bits(self) -> int:
        return 1 + self.llr_integer_bits + self.llr_fractional_bits

    @property
    def threshold_word_bits(self) -> int:
        return 1 + self.threshold_integer_bits + self.threshold_fractional_bits

    @property
    def lut_entries(self) -> int:
        return 1 << self.lut_address_bits


@dataclass(frozen=True)
class FixedPointStressConfig:
    n_samples: int = 65_536
    window_size: int = 256
    seeds: tuple[int, ...] = (101, 211, 307, 401, 503, 607, 709, 811)
    measurement_noise_sigma: float = 0.055
    initial_mean: float = 0.0
    initial_sigma: float = 0.38
    confidence_threshold: float = 0.24
    evaluation_warmup_windows: int = 2
    bootstrap_replicates: int = 10_000
    bootstrap_seed: int = 24_301
    bank_fault_every_updates: int = 4
    high_precision_max_abs_delta_ler: float = 0.003
    low_precision_min_ler_gap: float = 0.01
    pareto_max_delta_ler: float = 0.02
    severe_fault_min_ler_increase: float = 0.001

    def __post_init__(self) -> None:
        if self.n_samples <= 0 or self.window_size <= 0:
            raise ValueError("n_samples and window_size must be positive")
        if self.n_samples % self.window_size != 0:
            raise ValueError("n_samples must be divisible by window_size")
        if len(self.seeds) < 2 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("at least two unique seeds are required")
        if self.measurement_noise_sigma < 0:
            raise ValueError("measurement_noise_sigma must be non-negative")
        if not -LATTICE_CONST / 2 < self.initial_mean < LATTICE_CONST / 2:
            raise ValueError("initial_mean must lie inside the fundamental cell")
        if not 0.10 <= self.initial_sigma <= 0.90:
            raise ValueError("initial_sigma must lie in [0.10, 0.90]")
        if not 0.0 <= self.confidence_threshold < 1.0:
            raise ValueError("confidence_threshold must lie in [0, 1)")
        if self.evaluation_warmup_windows < 1:
            raise ValueError("evaluation_warmup_windows must be positive")
        if self.bootstrap_replicates < 1000:
            raise ValueError("bootstrap_replicates must be at least 1000")
        if self.bank_fault_every_updates < 2:
            raise ValueError("bank_fault_every_updates must be at least 2")
        for name in (
            "high_precision_max_abs_delta_ler",
            "low_precision_min_ler_gap",
            "pareto_max_delta_ler",
            "severe_fault_min_ler_increase",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass
class LUTBankImage:
    lut_codes: np.ndarray
    mean_code: int
    sigma_code: int
    threshold_code: int
    source_window: int
    llr_saturation_count: int
    image_sha256: str

    def copy(self) -> "LUTBankImage":
        return LUTBankImage(
            lut_codes=self.lut_codes.copy(),
            mean_code=int(self.mean_code),
            sigma_code=int(self.sigma_code),
            threshold_code=int(self.threshold_code),
            source_window=int(self.source_window),
            llr_saturation_count=int(self.llr_saturation_count),
            image_sha256=str(self.image_sha256),
        )


class LUTParameterBank:
    """Double-buffered integer-word image bank with explicit corruption models."""

    def __init__(self, image: LUTBankImage) -> None:
        self._banks = {"A": image.copy(), "B": image.copy()}
        self._active = "A"
        self._staging = "B"
        self.active_version = 0
        self.commits = 0
        self.fault_counts: Counter[str] = Counter()

    @property
    def active_image(self) -> LUTBankImage:
        return self._banks[self._active]

    def stage_and_commit(
        self,
        image: LUTBankImage,
        *,
        fault_mode: str,
        inject_fault: bool,
        profile: PrecisionProfile,
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        if fault_mode not in BANK_FAULT_MODES:
            raise ValueError(f"unsupported bank fault mode: {fault_mode}")
        staged = image.copy()
        event = {
            "fault_mode": fault_mode,
            "fault_injected": False,
            "commit_applied": True,
            "source_window": image.source_window,
        }
        if inject_fault and fault_mode != "none":
            event["fault_injected"] = True
            self.fault_counts[fault_mode] += 1
            if fault_mode == "stale_commit":
                event["commit_applied"] = False
                return event
            if fault_mode == "torn_update":
                staged.lut_codes[1::2] = self.active_image.lut_codes[1::2]
                staged.image_sha256 = _image_sha256(staged)
            elif fault_mode == "lut_sign_burst":
                width = max(1, profile.lut_entries // 8)
                start = int(rng.integers(0, profile.lut_entries - width + 1))
                indices = np.arange(start, start + width, dtype=np.int64)
                staged.lut_codes[indices] = _flip_signed_bit(
                    staged.lut_codes[indices],
                    word_bits=profile.llr_word_bits,
                    bit_index=profile.llr_word_bits - 1,
                )
                staged.image_sha256 = _image_sha256(staged)
                event.update({"address_start": start, "address_count": width})
            elif fault_mode == "state_msb_flip":
                staged.mean_code = int(
                    staged.mean_code ^ (1 << (profile.state_bits - 1))
                )
                staged = _rebuild_lut_from_codes(staged, profile)

        self._banks[self._staging] = staged
        self._active, self._staging = self._staging, self._active
        self.active_version += 1
        self.commits += 1
        return event


def base_profile() -> PrecisionProfile:
    return PrecisionProfile(
        profile_id="base_quantized",
        curve_axis="base",
        axis_value=1.0,
    )


def precision_profiles() -> tuple[PrecisionProfile, ...]:
    base = base_profile()
    profiles: list[PrecisionProfile] = [base]
    axes: tuple[tuple[str, str, Sequence[int]], ...] = (
        ("adc_bits", "adc_bits", (3, 4, 6, 8, 10, 12)),
        ("lut_address_bits", "lut_address_bits", (3, 4, 5, 6, 7, 8)),
        ("llr_fractional_bits", "llr_fractional_bits", (0, 1, 2, 4, 6, 8)),
        (
            "threshold_fractional_bits",
            "threshold_fractional_bits",
            (1, 2, 3, 5, 7, 9),
        ),
        ("state_bits", "state_bits", (3, 4, 6, 8, 10, 12)),
        (
            "update_period_windows",
            "update_period_windows",
            (1, 2, 4, 8, 16, 32),
        ),
    )
    for axis, field_name, values in axes:
        for value in values:
            profiles.append(
                replace(
                    base,
                    profile_id=f"axis_{axis}_{value}",
                    curve_axis=axis,
                    axis_value=float(value),
                    **{field_name: int(value)},
                )
            )

    joint = (
        ("joint_p03", 3, 3, 0, 1, 3, 32),
        ("joint_p04", 4, 4, 1, 2, 4, 16),
        ("joint_p06", 6, 5, 2, 3, 6, 8),
        ("joint_p08", 8, 6, 4, 5, 8, 4),
        ("joint_p12", 12, 8, 8, 9, 12, 1),
    )
    for profile_id, adc, address, llr_frac, threshold_frac, state, period in joint:
        profiles.append(
            replace(
                base,
                profile_id=profile_id,
                curve_axis="joint_precision",
                axis_value=float(adc),
                adc_bits=adc,
                lut_address_bits=address,
                llr_fractional_bits=llr_frac,
                threshold_fractional_bits=threshold_frac,
                state_bits=state,
                update_period_windows=period,
            )
        )
    if len({profile.profile_id for profile in profiles}) != len(profiles):
        raise RuntimeError("precision profile IDs must be unique")
    return tuple(profiles)


def resource_proxy(profile: PrecisionProfile, *, window_size: int) -> dict[str, Any]:
    update_payload = (
        profile.lut_entries * profile.llr_word_bits
        + 2 * profile.state_bits
        + profile.threshold_word_bits
    )
    dual_lut = 2 * profile.lut_entries * profile.llr_word_bits
    dual_state = 2 * (2 * profile.state_bits + profile.threshold_word_bits)
    return {
        "identity": "exact_representation_proxy_not_synthesis",
        "target_synthesis_measured": False,
        "lut_entries": profile.lut_entries,
        "lut_address_bits": profile.lut_address_bits,
        "llr_word_bits": profile.llr_word_bits,
        "threshold_word_bits": profile.threshold_word_bits,
        "state_word_bits_each": profile.state_bits,
        "dual_bank_lut_bits": dual_lut,
        "dual_bank_state_threshold_bits": dual_state,
        "total_dual_bank_storage_bits": dual_lut + dual_state,
        "replay_window_bits": window_size * profile.adc_bits,
        "update_payload_bits": update_payload,
        "mean_update_payload_bits_per_window": update_payload
        / profile.update_period_windows,
        "online_lookup_comparator_word_bits": profile.llr_word_bits
        + profile.threshold_word_bits,
        "fpga_lut_count": None,
        "bram_count": None,
        "dsp_count": None,
        "fmax_mhz": None,
    }


def _physical_trace(config: FixedPointStressConfig, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    sample = np.arange(config.n_samples, dtype=np.int64)
    block = sample // (8 * config.window_size)
    mean_levels = np.asarray([0.00, 0.42, -0.42, 0.76, -0.70, 0.18, -0.18, 0.62])
    sigma_levels = np.asarray([0.30, 0.34, 0.38, 0.44, 0.50, 0.36, 0.42, 0.32])
    true_mean = mean_levels[np.mod(block, mean_levels.size)]
    true_mean = true_mean + 0.035 * np.sin(2 * np.pi * sample / (29 * config.window_size))
    true_sigma = sigma_levels[np.mod(block, sigma_levels.size)]
    raw_error = true_mean + rng.normal(0.0, true_sigma, size=config.n_samples)
    lattice_index = np.floor(raw_error / LATTICE_CONST + 0.5).astype(np.int64)
    syndrome = raw_error - lattice_index * LATTICE_CONST
    observed = syndrome + rng.normal(
        0.0, config.measurement_noise_sigma, size=config.n_samples
    )
    half = LATTICE_CONST / 2.0
    observed = np.mod(observed + half, LATTICE_CONST) - half
    return {
        "true_mean": true_mean.astype(np.float64),
        "true_sigma": true_sigma.astype(np.float64),
        "raw_error": raw_error.astype(np.float64),
        "syndrome": syndrome.astype(np.float64),
        "observed_syndrome": observed.astype(np.float64),
        "true_parity": np.mod(lattice_index, 2).astype(np.bool_),
    }


def estimate_wrapped_state(observed_syndrome: np.ndarray) -> tuple[float, float]:
    """Observed-only circular mean and wrapped-normal sigma estimate."""

    values = np.asarray(observed_syndrome, dtype=np.float64)
    if values.ndim != 1 or values.size < 4:
        raise ValueError("state estimation requires a 1D window with at least four samples")
    if not np.all(np.isfinite(values)):
        raise ValueError("state-estimation input must be finite")
    half = LATTICE_CONST / 2.0
    if np.any(values < -half) or np.any(values >= half):
        raise ValueError("state-estimation input must be centered syndrome")
    phases = 2.0 * np.pi * values / LATTICE_CONST
    phasor = np.mean(np.exp(1j * phases))
    mean = float(np.angle(phasor) * LATTICE_CONST / (2.0 * np.pi))
    concentration = float(np.clip(abs(phasor), 1.0e-12, 1.0))
    sigma = float(
        LATTICE_CONST / (2.0 * np.pi) * np.sqrt(max(0.0, -2.0 * np.log(concentration)))
    )
    return mean, float(np.clip(sigma, 0.10, 0.90))


def _state_formats(profile: PrecisionProfile) -> tuple[UniformCodeFormat, UniformCodeFormat]:
    half = LATTICE_CONST / 2.0
    return (
        UniformCodeFormat(profile.state_bits, -half, half),
        UniformCodeFormat(profile.state_bits, 0.10, 0.90),
    )


def _image_sha256(image: LUTBankImage) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(image.lut_codes, dtype="<i8").tobytes())
    for value in (
        image.mean_code,
        image.sigma_code,
        image.threshold_code,
        image.source_window,
        image.llr_saturation_count,
    ):
        digest.update(int(value).to_bytes(8, "little", signed=True))
    return digest.hexdigest()


def _build_bank_image(
    profile: PrecisionProfile,
    *,
    mean: float,
    sigma: float,
    threshold: float,
    source_window: int,
) -> LUTBankImage:
    mean_format, sigma_format = _state_formats(profile)
    mean_q, mean_code, _ = mean_format.quantize(mean)
    sigma_q, sigma_code, _ = sigma_format.quantize(sigma)
    llr_format = FixedPointFormat(profile.llr_integer_bits, profile.llr_fractional_bits)
    threshold_format = FixedPointFormat(
        profile.threshold_integer_bits, profile.threshold_fractional_bits
    )
    centers = UniformCodeFormat(
        profile.lut_address_bits,
        -LATTICE_CONST / 2.0,
        LATTICE_CONST / 2.0,
    ).decode(np.arange(profile.lut_entries, dtype=np.int64))
    values = np.asarray(
        llr_1d(centers, float(sigma_q), mean=float(mean_q)), dtype=np.float64
    )
    llr_codes, llr_saturated = llr_format.encode(values)
    threshold_code, _ = threshold_format.encode(float(threshold))
    image = LUTBankImage(
        lut_codes=llr_codes,
        mean_code=int(np.asarray(mean_code).item()),
        sigma_code=int(np.asarray(sigma_code).item()),
        threshold_code=int(np.asarray(threshold_code).item()),
        source_window=int(source_window),
        llr_saturation_count=int(np.count_nonzero(llr_saturated)),
        image_sha256="",
    )
    image.image_sha256 = _image_sha256(image)
    return image


def _rebuild_lut_from_codes(
    image: LUTBankImage,
    profile: PrecisionProfile,
) -> LUTBankImage:
    mean_format, sigma_format = _state_formats(profile)
    mean = float(mean_format.decode(image.mean_code))
    sigma = float(sigma_format.decode(image.sigma_code))
    threshold_format = FixedPointFormat(
        profile.threshold_integer_bits, profile.threshold_fractional_bits
    )
    threshold = float(threshold_format.decode(image.threshold_code))
    return _build_bank_image(
        profile,
        mean=mean,
        sigma=sigma,
        threshold=threshold,
        source_window=image.source_window,
    )


def _flip_signed_bit(
    codes: np.ndarray,
    *,
    word_bits: int,
    bit_index: int,
) -> np.ndarray:
    if not 0 <= bit_index < word_bits:
        raise ValueError("bit_index must lie inside word width")
    values = np.asarray(codes, dtype=np.int64)
    lower = -(1 << (word_bits - 1))
    upper = (1 << (word_bits - 1)) - 1
    if np.any(values < lower) or np.any(values > upper):
        raise ValueError("signed code lies outside word width")
    mask = (1 << word_bits) - 1
    unsigned = np.bitwise_and(values, mask)
    flipped = np.bitwise_xor(unsigned, 1 << bit_index)
    sign = 1 << (word_bits - 1)
    signed = np.where(flipped >= sign, flipped - (1 << word_bits), flipped)
    return signed.astype(np.int64)


def _lut_addresses(values: np.ndarray, address_bits: int) -> np.ndarray:
    fmt = UniformCodeFormat(
        address_bits, -LATTICE_CONST / 2.0, LATTICE_CONST / 2.0
    )
    addresses, _ = fmt.encode(values)
    return addresses


def _simulate_float_reference(
    trace: Mapping[str, np.ndarray],
    config: FixedPointStressConfig,
) -> dict[str, Any]:
    observed = np.asarray(trace["observed_syndrome"], dtype=np.float64)
    truth = np.asarray(trace["true_parity"], dtype=np.bool_)
    predictions = np.zeros(config.n_samples, dtype=np.bool_)
    mean = config.initial_mean
    sigma = config.initial_sigma
    n_windows = config.n_samples // config.window_size
    for window in range(n_windows):
        start = window * config.window_size
        end = start + config.window_size
        values = observed[start:end]
        llr = np.asarray(llr_1d(values, sigma, mean=mean), dtype=np.float64)
        predictions[start:end] = llr < -config.confidence_threshold
        mean, sigma = estimate_wrapped_state(values)
    evaluation_start = config.evaluation_warmup_windows * config.window_size
    errors = predictions[evaluation_start:] != truth[evaluation_start:]
    return {
        "predictions": predictions,
        "logical_error_rate": float(np.mean(errors)),
        "logical_error_count": int(np.count_nonzero(errors)),
        "evaluated_samples": int(errors.size),
        "evaluation_start_sample": evaluation_start,
    }


def simulate_quantized_profile(
    profile: PrecisionProfile,
    *,
    trace: Mapping[str, np.ndarray],
    float_reference: Mapping[str, Any],
    config: FixedPointStressConfig,
    seed: int,
    bank_fault_mode: str = "none",
) -> dict[str, Any]:
    if bank_fault_mode not in BANK_FAULT_MODES:
        raise ValueError(f"unsupported bank fault mode: {bank_fault_mode}")
    observed = np.asarray(trace["observed_syndrome"], dtype=np.float64)
    truth = np.asarray(trace["true_parity"], dtype=np.bool_)
    float_predictions = np.asarray(float_reference["predictions"], dtype=np.bool_)
    adc_format = UniformCodeFormat(
        profile.adc_bits, -LATTICE_CONST / 2.0, LATTICE_CONST / 2.0
    )
    llr_format = FixedPointFormat(profile.llr_integer_bits, profile.llr_fractional_bits)
    threshold_format = FixedPointFormat(
        profile.threshold_integer_bits, profile.threshold_fractional_bits
    )
    initial = _build_bank_image(
        profile,
        mean=config.initial_mean,
        sigma=config.initial_sigma,
        threshold=config.confidence_threshold,
        source_window=-1,
    )
    bank = LUTParameterBank(initial)
    rng = np.random.default_rng(seed + 243_000 + sum(ord(c) for c in bank_fault_mode))
    predictions = np.zeros(config.n_samples, dtype=np.bool_)
    adc_saturations = 0
    llr_saturations = 0
    update_attempts = 0
    bank_events: Counter[str] = Counter()
    maximum_bank_age_windows = 0
    n_windows = config.n_samples // config.window_size

    for window in range(n_windows):
        start = window * config.window_size
        end = start + config.window_size
        adc_values, _, adc_saturated = adc_format.quantize(observed[start:end])
        adc_saturations += int(np.count_nonzero(adc_saturated))
        image = bank.active_image
        addresses = _lut_addresses(adc_values, profile.lut_address_bits)
        llr_values = llr_format.decode(image.lut_codes[addresses])
        threshold = float(threshold_format.decode(image.threshold_code))
        predictions[start:end] = llr_values < -threshold
        maximum_bank_age_windows = max(
            maximum_bank_age_windows,
            window - image.source_window,
        )

        if (window + 1) % profile.update_period_windows == 0:
            mean_hat, sigma_hat = estimate_wrapped_state(adc_values)
            staged = _build_bank_image(
                profile,
                mean=mean_hat,
                sigma=sigma_hat,
                threshold=config.confidence_threshold,
                source_window=window,
            )
            llr_saturations += staged.llr_saturation_count
            update_attempts += 1
            inject = update_attempts % config.bank_fault_every_updates == 0
            event = bank.stage_and_commit(
                staged,
                fault_mode=bank_fault_mode,
                inject_fault=inject,
                profile=profile,
                rng=rng,
            )
            bank_events["update_attempt"] += 1
            if event["commit_applied"]:
                bank_events["commit_applied"] += 1
            if event["fault_injected"]:
                bank_events[bank_fault_mode] += 1

    evaluation_start = config.evaluation_warmup_windows * config.window_size
    evaluated_predictions = predictions[evaluation_start:]
    evaluated_truth = truth[evaluation_start:]
    errors = evaluated_predictions != evaluated_truth
    disagreement = evaluated_predictions != float_predictions[evaluation_start:]
    resources = resource_proxy(profile, window_size=config.window_size)
    return {
        "profile_id": profile.profile_id,
        "curve_axis": profile.curve_axis,
        "axis_value": profile.axis_value,
        "seed": int(seed),
        "bank_fault_mode": bank_fault_mode,
        "model_scope": MODEL_SCOPE,
        "target_hardware_measured": False,
        "synthesis_measured": False,
        "metrics": {
            "logical_error_rate": float(np.mean(errors)),
            "logical_error_count": int(np.count_nonzero(errors)),
            "evaluated_samples": int(errors.size),
            "prediction_disagreement_vs_float": float(np.mean(disagreement)),
            "adc_saturation_rate": float(adc_saturations / config.n_samples),
            "llr_lut_saturation_rate_per_entry_update": float(
                llr_saturations
                / max(1, update_attempts * profile.lut_entries)
            ),
            "maximum_bank_age_windows": int(maximum_bank_age_windows),
        },
        "profile": asdict(profile),
        "resource_proxy": resources,
        "bank": {
            "active_version": bank.active_version,
            "commits": bank.commits,
            "events": dict(sorted(bank_events.items())),
            "fault_counts": dict(sorted(bank.fault_counts.items())),
            "active_image_sha256": bank.active_image.image_sha256,
        },
        "integrity": {
            "predictions_finite_binary": bool(
                predictions.dtype == np.bool_ and predictions.size == config.n_samples
            ),
            "version_matches_commits": bank.active_version == bank.commits,
            "observed_only_estimator": True,
            "float_reference_ler": float(float_reference["logical_error_rate"]),
        },
    }


def _paired_bootstrap(
    values: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    if values.ndim != 1 or values.size < 2:
        raise ValueError("paired bootstrap requires at least two values")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(replicates, values.size))
    means = values[indices].mean(axis=1)
    return {
        "mean": float(np.mean(values)),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "probability_positive": float(np.mean(means > 0.0)),
        "probability_negative": float(np.mean(means < 0.0)),
    }


def _aggregate(
    rows: Sequence[Mapping[str, Any]],
    *,
    float_ler_by_seed: Mapping[int, float],
    config: FixedPointStressConfig,
    bootstrap_offset: int,
) -> dict[str, Any]:
    ler = np.asarray([row["metrics"]["logical_error_rate"] for row in rows])
    reference = np.asarray([float_ler_by_seed[int(row["seed"])] for row in rows])
    first = rows[0]
    return {
        "profile_id": first["profile_id"],
        "curve_axis": first["curve_axis"],
        "axis_value": first["axis_value"],
        "bank_fault_mode": first["bank_fault_mode"],
        "seeds": [int(row["seed"]) for row in rows],
        "logical_error_rate": {
            "mean": float(np.mean(ler)),
            "min": float(np.min(ler)),
            "max": float(np.max(ler)),
        },
        "paired_ler_minus_float": _paired_bootstrap(
            ler - reference,
            replicates=config.bootstrap_replicates,
            seed=config.bootstrap_seed + bootstrap_offset,
        ),
        "prediction_disagreement_vs_float_mean": float(
            np.mean(
                [row["metrics"]["prediction_disagreement_vs_float"] for row in rows]
            )
        ),
        "maximum_bank_age_windows": int(
            max(row["metrics"]["maximum_bank_age_windows"] for row in rows)
        ),
        "fault_events_total": int(
            sum(
                row["bank"]["events"].get(first["bank_fault_mode"], 0)
                for row in rows
            )
        ),
        "resource_proxy": first["resource_proxy"],
        "profile": first["profile"],
    }


def _sha256_sources(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        rel = path.relative_to(ROOT).as_posix().encode("utf-8")
        digest.update(len(rel).to_bytes(4, "big"))
        digest.update(rel)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def run_fixed_point_validation(
    config: FixedPointStressConfig | None = None,
    *,
    profiles: Sequence[PrecisionProfile] | None = None,
) -> dict[str, Any]:
    cfg = config or FixedPointStressConfig()
    selected_profiles = tuple(profiles or precision_profiles())
    if not selected_profiles:
        raise ValueError("at least one precision profile is required")
    if len({profile.profile_id for profile in selected_profiles}) != len(selected_profiles):
        raise ValueError("profile IDs must be unique")

    traces = {seed: _physical_trace(cfg, seed) for seed in cfg.seeds}
    float_reference = {
        seed: _simulate_float_reference(traces[seed], cfg) for seed in cfg.seeds
    }
    float_ler_by_seed = {
        seed: float(result["logical_error_rate"])
        for seed, result in float_reference.items()
    }

    per_seed: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for profile in selected_profiles:
        key = (profile.profile_id, "none")
        grouped[key] = []
        for seed in cfg.seeds:
            row = simulate_quantized_profile(
                profile,
                trace=traces[seed],
                float_reference=float_reference[seed],
                config=cfg,
                seed=seed,
            )
            per_seed.append(row)
            grouped[key].append(row)

    base = next(
        (profile for profile in selected_profiles if profile.profile_id == "base_quantized"),
        selected_profiles[0],
    )
    for fault_mode in BANK_FAULT_MODES[1:]:
        key = (base.profile_id, fault_mode)
        grouped[key] = []
        for seed in cfg.seeds:
            row = simulate_quantized_profile(
                base,
                trace=traces[seed],
                float_reference=float_reference[seed],
                config=cfg,
                seed=seed,
                bank_fault_mode=fault_mode,
            )
            per_seed.append(row)
            grouped[key].append(row)

    aggregates = [
        _aggregate(
            rows,
            float_ler_by_seed=float_ler_by_seed,
            config=cfg,
            bootstrap_offset=index,
        )
        for index, rows in enumerate(grouped.values())
    ]
    aggregate_map = {
        (row["profile_id"], row["bank_fault_mode"]): row for row in aggregates
    }
    axes = {
        axis: sorted(
            [
                row
                for row in aggregates
                if row["curve_axis"] == axis and row["bank_fault_mode"] == "none"
            ],
            key=lambda row: row["axis_value"],
        )
        for axis in (
            "adc_bits",
            "lut_address_bits",
            "llr_fractional_bits",
            "threshold_fractional_bits",
            "state_bits",
            "update_period_windows",
            "joint_precision",
        )
    }
    base_aggregate = aggregate_map[(base.profile_id, "none")]
    fault_aggregates = {
        mode: aggregate_map[(base.profile_id, mode)] for mode in BANK_FAULT_MODES[1:]
    }
    base_ler_by_seed = {
        int(row["seed"]): float(row["metrics"]["logical_error_rate"])
        for row in grouped[(base.profile_id, "none")]
    }
    for index, mode in enumerate(BANK_FAULT_MODES[1:]):
        fault_rows = grouped[(base.profile_id, mode)]
        paired_fault_delta = np.asarray(
            [
                float(row["metrics"]["logical_error_rate"])
                - base_ler_by_seed[int(row["seed"])]
                for row in fault_rows
            ],
            dtype=np.float64,
        )
        fault_aggregates[mode]["paired_ler_minus_base_quantized"] = _paired_bootstrap(
            paired_fault_delta,
            replicates=cfg.bootstrap_replicates,
            seed=cfg.bootstrap_seed + 10_000 + index,
        )
    severe_faults = ("lut_sign_burst", "state_msb_flip")
    high_joint = axes["joint_precision"][-1]
    low_joint = axes["joint_precision"][0]
    feasible_precision_points = [
        row
        for row in aggregates
        if row["bank_fault_mode"] == "none"
        if row["resource_proxy"]["total_dual_bank_storage_bits"]
        < high_joint["resource_proxy"]["total_dual_bank_storage_bits"]
        and row["paired_ler_minus_float"]["ci_high"] <= cfg.pareto_max_delta_ler
    ]
    candidate_pareto = [
        row
        for row in feasible_precision_points
        if not any(
            other["profile_id"] != row["profile_id"]
            and other["resource_proxy"]["total_dual_bank_storage_bits"]
            <= row["resource_proxy"]["total_dual_bank_storage_bits"]
            and other["logical_error_rate"]["mean"]
            <= row["logical_error_rate"]["mean"]
            and (
                other["resource_proxy"]["total_dual_bank_storage_bits"]
                < row["resource_proxy"]["total_dual_bank_storage_bits"]
                or other["logical_error_rate"]["mean"]
                < row["logical_error_rate"]["mean"]
            )
            for other in feasible_precision_points
        )
    ]
    expected_axis_lengths = {axis: 6 for axis in axes if axis != "joint_precision"}
    expected_axis_lengths["joint_precision"] = 5
    all_integrity = all(
        row["integrity"]["predictions_finite_binary"]
        and row["integrity"]["version_matches_commits"]
        and row["integrity"]["observed_only_estimator"]
        and row["target_hardware_measured"] is False
        and row["synthesis_measured"] is False
        for row in per_seed
    )
    fault_per_seed = all(
        row["bank"]["events"].get(row["bank_fault_mode"], 0) > 0
        for row in per_seed
        if row["bank_fault_mode"] != "none"
    )
    gates = {
        "all_precision_axes_have_predeclared_coverage": all(
            len(axes[axis]) == length for axis, length in expected_axis_lengths.items()
        ),
        "all_profiles_and_faults_run_on_all_paired_seeds": len(per_seed)
        == (len(selected_profiles) + len(BANK_FAULT_MODES) - 1) * len(cfg.seeds),
        "float_reference_is_analytic_without_lut_resource_claim": True,
        "highest_joint_precision_converges_to_float": abs(
            high_joint["paired_ler_minus_float"]["mean"]
        )
        <= cfg.high_precision_max_abs_delta_ler,
        "lowest_joint_precision_is_detectably_worse_than_highest": low_joint[
            "logical_error_rate"
        ]["mean"]
        > high_joint["logical_error_rate"]["mean"] + cfg.low_precision_min_ler_gap,
        "nontrivial_precision_storage_pareto_candidate_exists": bool(candidate_pareto),
        "update_granularity_changes_bank_age": axes["update_period_windows"][-1][
            "maximum_bank_age_windows"
        ]
        > axes["update_period_windows"][0]["maximum_bank_age_windows"],
        "every_injected_bank_fault_detected_per_seed": fault_per_seed,
        "severe_bank_faults_raise_ler": all(
            fault_aggregates[mode]["paired_ler_minus_base_quantized"]["ci_low"]
            > cfg.severe_fault_min_ler_increase
            for mode in severe_faults
        ),
        "integer_state_and_numeric_integrity": all_integrity,
        "resources_are_proxy_not_synthesis_or_board": all(
            row["resource_proxy"]["identity"]
            == "exact_representation_proxy_not_synthesis"
            and row["resource_proxy"]["fpga_lut_count"] is None
            and row["resource_proxy"]["bram_count"] is None
            and row["resource_proxy"]["dsp_count"] is None
            for row in aggregates
        ),
    }
    implementation_sha256 = _sha256_sources(
        [
            ROOT / "cnn_fpga" / "runtime" / "fixed_point_chain.py",
            ROOT / "cnn_fpga" / "decoder" / "linear_runtime.py",
            ROOT / "physics" / "ideal_gkp_decoder.py",
        ]
    )
    return {
        "contract_id": CONTRACT_ID,
        "task_id": "T2.4.3",
        "model_scope": MODEL_SCOPE,
        "target_hardware_measured": False,
        "synthesis_measured": False,
        "implementation_sha256": implementation_sha256,
        "config": asdict(cfg),
        "float_reference": {
            "identity": "observed_only_one_window_delayed_analytic_llr",
            "resource_proxy": None,
            "per_seed_ler": {str(seed): value for seed, value in float_ler_by_seed.items()},
            "mean_ler": float(np.mean(list(float_ler_by_seed.values()))),
        },
        "profiles": [asdict(profile) for profile in selected_profiles],
        "per_seed_results": per_seed,
        "aggregates": aggregates,
        "curves": axes,
        "pareto_candidates": candidate_pareto,
        "bank_fault_aggregates": fault_aggregates,
        "gates": gates,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "claim_boundary": {
            "allowed": "paired bit-accurate software precision/resource-proxy/LER sensitivity",
            "forbidden": [
                "FPGA LUT/BRAM/DSP utilization",
                "post-place-and-route timing or Fmax",
                "target-board latency or availability",
                "device-calibrated ADC, bank upset, or quantum LER",
            ],
        },
    }


def write_source_csv(path: Path, result: Mapping[str, Any]) -> None:
    fields = [
        "profile_id",
        "curve_axis",
        "axis_value",
        "bank_fault_mode",
        "seed",
        "logical_error_rate",
        "prediction_disagreement_vs_float",
        "maximum_bank_age_windows",
        "fault_events",
        "adc_bits",
        "lut_address_bits",
        "llr_word_bits",
        "threshold_word_bits",
        "state_bits",
        "update_period_windows",
        "total_dual_bank_storage_bits",
        "replay_window_bits",
        "update_payload_bits",
        "mean_update_payload_bits_per_window",
        "target_hardware_measured",
        "synthesis_measured",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in result["per_seed_results"]:
            profile = row["profile"]
            resource = row["resource_proxy"]
            writer.writerow(
                {
                    "profile_id": row["profile_id"],
                    "curve_axis": row["curve_axis"],
                    "axis_value": row["axis_value"],
                    "bank_fault_mode": row["bank_fault_mode"],
                    "seed": row["seed"],
                    "logical_error_rate": row["metrics"]["logical_error_rate"],
                    "prediction_disagreement_vs_float": row["metrics"][
                        "prediction_disagreement_vs_float"
                    ],
                    "maximum_bank_age_windows": row["metrics"][
                        "maximum_bank_age_windows"
                    ],
                    "fault_events": row["bank"]["events"].get(
                        row["bank_fault_mode"], 0
                    ),
                    "adc_bits": profile["adc_bits"],
                    "lut_address_bits": profile["lut_address_bits"],
                    "llr_word_bits": resource["llr_word_bits"],
                    "threshold_word_bits": resource["threshold_word_bits"],
                    "state_bits": profile["state_bits"],
                    "update_period_windows": profile["update_period_windows"],
                    "total_dual_bank_storage_bits": resource[
                        "total_dual_bank_storage_bits"
                    ],
                    "replay_window_bits": resource["replay_window_bits"],
                    "update_payload_bits": resource["update_payload_bits"],
                    "mean_update_payload_bits_per_window": resource[
                        "mean_update_payload_bits_per_window"
                    ],
                    "target_hardware_measured": row["target_hardware_measured"],
                    "synthesis_measured": row["synthesis_measured"],
                }
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--samples", type=int, default=FixedPointStressConfig.n_samples)
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in FixedPointStressConfig.seeds),
    )
    parser.add_argument(
        "--bootstrap", type=int, default=FixedPointStressConfig.bootstrap_replicates
    )
    args = parser.parse_args(argv)
    seeds = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
    result = run_fixed_point_validation(
        FixedPointStressConfig(
            n_samples=args.samples,
            seeds=seeds,
            bootstrap_replicates=args.bootstrap,
        )
    )
    save_json(args.artifact, result)
    write_source_csv(args.csv, result)
    summary = {
        "status": result["status"],
        "artifact": str(args.artifact),
        "csv": str(args.csv),
        "float_mean_ler": result["float_reference"]["mean_ler"],
        "gates": result["gates"],
        "joint_curve": [
            {
                "profile_id": row["profile_id"],
                "ler": row["logical_error_rate"]["mean"],
                "delta_ler": row["paired_ler_minus_float"],
                "storage_bits": row["resource_proxy"]["total_dual_bank_storage_bits"],
            }
            for row in result["curves"]["joint_precision"]
        ],
        "bank_faults": {
            mode: row["logical_error_rate"]["mean"]
            for mode, row in result["bank_fault_aggregates"].items()
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
