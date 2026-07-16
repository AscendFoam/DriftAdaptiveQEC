"""Slow-path compiler from a live DecoderRuntimeParams bank to integer MAP ROMs."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np

from cnn_fpga.runtime.param_bank import DecoderRuntimeParams, ParamBank
from cnn_fpga.runtime.parametric_map_lut import (
    ParametricMAPLUTConfig,
    ParametricMAPLUTImage,
    software_decode_syndrome_code,
)
from physics.ideal_gkp_decoder import llr_1d


@dataclass(frozen=True)
class AxisMAPModel:
    mean: tuple[float, float]
    sigma: tuple[float, float]
    covariance: tuple[tuple[float, float], tuple[float, float]]
    measurement_covariance: tuple[tuple[float, float], tuple[float, float]]
    alpha_bias: float
    derivation_residual_max_abs: float
    scope: str = "active_K_b_effective_marginal_periodic_gaussian"


def source_params_sha256(params: DecoderRuntimeParams) -> str:
    if not isinstance(params, DecoderRuntimeParams):
        raise TypeError("params must be DecoderRuntimeParams")
    payload = json.dumps(
        params.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def derive_axis_map_model(params: DecoderRuntimeParams) -> AxisMAPModel:
    """Recover the effective Gaussian model represented by active K/b.

    ParamMapper uses K=C(C+R)^-1 and b=alpha(I-K)mu. Compilation performs the
    inverse algebra on the slow path, so every ROM is bound to actual active
    K/b rather than merely copying a stale prediction from metadata.
    """

    if not isinstance(params, DecoderRuntimeParams):
        raise TypeError("params must be DecoderRuntimeParams")
    metadata = params.metadata
    if "measurement_cov" not in metadata or "alpha_bias" not in metadata:
        raise ValueError(
            "parameter bank lacks measurement_cov/alpha_bias required by MAP-LUT compiler"
        )
    k = np.asarray(params.K, dtype=np.float64)
    b = np.asarray(params.b, dtype=np.float64)
    r = np.asarray(metadata["measurement_cov"], dtype=np.float64)
    alpha = float(metadata["alpha_bias"])
    if r.shape != (2, 2) or not np.all(np.isfinite(r)):
        raise ValueError("measurement_cov must be a finite 2x2 matrix")
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha_bias must be finite and positive")
    if not np.allclose(k, k.T, rtol=0.0, atol=2.0e-12):
        raise ValueError("active K must be symmetric for marginal MAP compilation")
    if not np.allclose(r, r.T, rtol=0.0, atol=2.0e-12):
        raise ValueError("measurement_cov must be symmetric")
    if float(np.min(np.linalg.eigvalsh(r))) <= 0.0:
        raise ValueError("measurement_cov must be positive definite")
    gain_eigenvalues = np.linalg.eigvalsh(k)
    if float(np.min(gain_eigenvalues)) <= 0.0 or float(np.max(gain_eigenvalues)) >= 1.0:
        raise ValueError("active K eigenvalues must lie strictly inside (0,1)")

    complement = np.eye(2, dtype=np.float64) - k
    mean = np.linalg.solve(complement, b / alpha)
    covariance = np.linalg.solve(complement, k @ r)
    covariance = 0.5 * (covariance + covariance.T)
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(covariance)):
        raise ValueError("derived MAP model is non-finite")
    if float(np.min(np.linalg.eigvalsh(covariance))) <= 0.0:
        raise ValueError("derived MAP covariance must be positive definite")
    reconstruction = covariance @ np.linalg.inv(covariance + r)
    residual = float(np.max(np.abs(reconstruction - k)))
    if residual > 2.0e-10:
        raise ValueError("active K is inconsistent with derived Gaussian model")
    sigma = np.sqrt(np.diag(covariance))
    return AxisMAPModel(
        mean=(float(mean[0]), float(mean[1])),
        sigma=(float(sigma[0]), float(sigma[1])),
        covariance=(
            (float(covariance[0, 0]), float(covariance[0, 1])),
            (float(covariance[1, 0]), float(covariance[1, 1])),
        ),
        measurement_covariance=(
            (float(r[0, 0]), float(r[0, 1])),
            (float(r[1, 0]), float(r[1, 1])),
        ),
        alpha_bias=alpha,
        derivation_residual_max_abs=residual,
    )


def _encode_llr(
    values: np.ndarray, config: ParametricMAPLUTConfig
) -> tuple[np.ndarray, int]:
    scale = 1 << config.llr_fractional_bits
    rounded = np.rint(np.asarray(values, dtype=np.float64) * scale)
    saturated = np.logical_or(
        rounded < config.llr_min_code, rounded > config.llr_max_code
    )
    codes = np.clip(rounded, config.llr_min_code, config.llr_max_code).astype(np.int64)
    return codes, int(np.count_nonzero(saturated))


def compile_parametric_map_lut(
    params: DecoderRuntimeParams,
    *,
    active_bank_version: int,
    config: ParametricMAPLUTConfig | None = None,
) -> ParametricMAPLUTImage:
    actual = ParametricMAPLUTConfig() if config is None else config
    if not isinstance(actual, ParametricMAPLUTConfig):
        raise TypeError("config must be ParametricMAPLUTConfig")
    if isinstance(active_bank_version, bool) or not isinstance(active_bank_version, int):
        raise TypeError("active_bank_version must be an integer")
    if active_bank_version < 0:
        raise ValueError("active_bank_version must be non-negative")
    model = derive_axis_map_model(params)

    node_codes = np.arange(actual.table_entries, dtype=np.int64) << actual.fraction_bits
    node_values = -0.5 * actual.lattice + node_codes.astype(np.float64) * (
        actual.lattice / actual.adc_levels
    )
    node_values[-1] = np.nextafter(0.5 * actual.lattice, -np.inf)
    tables: list[tuple[int, ...]] = []
    saturation_count = 0
    for phase in (0, 1):
        exact = np.asarray(
            llr_1d(
                node_values,
                model.sigma[phase],
                mean=model.mean[phase],
                lattice=actual.lattice,
            ),
            dtype=np.float64,
        )
        codes, count = _encode_llr(exact, actual)
        tables.append(tuple(int(value) for value in codes))
        saturation_count += count
    return ParametricMAPLUTImage.create(
        config=actual,
        active_bank_version=active_bank_version,
        source_params_sha256=source_params_sha256(params),
        model_mean=model.mean,
        model_sigma=model.sigma,
        table_codes=(tables[0], tables[1]),
        llr_saturation_count=saturation_count,
    )


def compile_active_param_bank(
    param_bank: ParamBank,
    config: ParametricMAPLUTConfig | None = None,
) -> ParametricMAPLUTImage:
    if not isinstance(param_bank, ParamBank):
        raise TypeError("param_bank must be ParamBank")
    return compile_parametric_map_lut(
        param_bank.read_active(),
        active_bank_version=param_bank.active_version,
        config=config,
    )


def exact_quantized_llr_code(
    syndrome_code: int,
    quadrature_phase_bit: int,
    image: ParametricMAPLUTImage,
) -> int:
    if quadrature_phase_bit not in (0, 1):
        raise ValueError("quadrature_phase_bit must be 0 or 1")
    syndrome = software_decode_syndrome_code(syndrome_code, image.config)
    exact = float(
        llr_1d(
            syndrome,
            image.model_sigma[quadrature_phase_bit],
            mean=image.model_mean[quadrature_phase_bit],
            lattice=image.config.lattice,
        )
    )
    code, _ = _encode_llr(np.asarray([exact]), image.config)
    return int(code[0])


__all__ = [
    "AxisMAPModel",
    "compile_active_param_bank",
    "compile_parametric_map_lut",
    "derive_axis_map_model",
    "exact_quantized_llr_code",
    "source_params_sha256",
]
