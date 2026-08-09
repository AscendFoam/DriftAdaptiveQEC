"""Private validation runner and CLI for the control-imperfection model."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from math import cos, pi, sin
from pathlib import Path
from types import MappingProxyType

import numpy as np

from .. import control_imperfections as _model_module
from ..control_imperfections import (
    ControlActionRequest,
    ControlImperfectionConfig,
    ControlImperfectionModel,
    ControlImperfectionValidationResult,
    _seed,
    ideal_control_imperfection_config,
)


def _validation_config() -> ControlImperfectionConfig:
    return ControlImperfectionConfig(
        enable_quantization=True,
        awg_amplitude_bits=10,
        awg_phase_bits=11,
        awg_amplitude_full_scale=3.0,
        dac_bits=12,
        dac_full_scale=2.5,
        virtual_rotation_bits=12,
        pulse_gain_matrix=np.asarray([[1.012, 0.007], [-0.004, 0.991]]),
        pulse_bias=(0.002, -0.003),
        active_relative_gain_sigma=0.015,
        active_displacement_covariance=np.asarray(
            [[4.0e-4, 1.0e-4], [1.0e-4, 3.0e-4]]
        ),
        virtual_rotation_gain=1.003,
        virtual_rotation_bias_rad=0.002,
        virtual_rotation_noise_sigma_rad=0.008,
        latency_drift_per_us=(0.001, -0.0005),
        latency_diffusion_covariance_per_us=np.asarray(
            [[1.0e-4, 2.0e-5], [2.0e-5, 8.0e-5]]
        ),
        max_latency_us=50.0,
        action_order="displacement_then_virtual_rotation",
        quantization_provenance="T2.2.3 seeded sensitivity assumptions",
        pulse_provenance="T2.2.3 seeded affine and stochastic pulse assumptions",
        latency_provenance="T2.2.3 seeded drift-diffusion latency assumptions",
    )


def run_control_imperfection_validation(
    *,
    samples: int = 100_000,
    seed: int = 2026071423,
) -> ControlImperfectionValidationResult:
    if not isinstance(samples, int) or samples < 10_000:
        raise ValueError("samples must be an integer >= 10000")
    normalized_seed = _seed(seed)
    model = ControlImperfectionModel(_validation_config())
    request = ControlActionRequest(
        cycle_index=0,
        correction_command=(0.63, -0.41),
        virtual_rotation_command_rad=0.37,
        latency_us=4.0,
        parameter_bank_version=7,
    )
    pre = (0.72, -0.33)
    record = model.encode(request)
    batch = model.sample_fixed_request(
        request,
        residual_before_latency=pre,
        samples=samples,
        seed=normalized_seed,
    )
    analytic_mean, analytic_covariance = model.analytic_residual_moments(
        request,
        residual_before_latency=pre,
    )
    empirical_mean = batch.empirical_mean
    empirical_covariance = batch.empirical_covariance
    standard_errors = np.sqrt(np.diag(analytic_covariance) / samples)
    maximum_z = float(
        np.max(np.abs(empirical_mean - analytic_mean) / standard_errors)
    )
    relative_covariance = float(
        np.linalg.norm(empirical_covariance - analytic_covariance)
        / np.linalg.norm(analytic_covariance)
    )
    quantization_bits = (6, 8, 10, 12)
    radii = np.linspace(0.05, 1.8, 12)
    angles = np.linspace(-pi, pi, 73, endpoint=False)
    quantization_errors: list[float] = []
    for bits in quantization_bits:
        quantization_model = ControlImperfectionModel(
            replace(
                _validation_config(),
                awg_amplitude_bits=bits,
                awg_phase_bits=bits,
                dac_bits=bits,
                virtual_rotation_bits=bits,
                awg_amplitude_full_scale=2.0,
                dac_full_scale=2.0,
            )
        )
        squared_error: list[float] = []
        cycle_index = 0
        for radius in radii:
            for angle in angles:
                requested = (float(radius * cos(angle)), float(radius * sin(angle)))
                encoded = quantization_model.encode(
                    ControlActionRequest(
                        cycle_index=cycle_index,
                        correction_command=requested,
                        virtual_rotation_command_rad=angle,
                        latency_us=0.0,
                    )
                )
                squared_error.append(
                    float(
                        np.sum(
                            (
                                np.asarray(encoded.correction_commanded)
                                - np.asarray(requested)
                            )
                            ** 2
                        )
                    )
                )
                cycle_index += 1
        quantization_errors.append(float(np.sqrt(np.mean(squared_error))))

    record_pulse_mean = (
        np.asarray(model.config.pulse_gain_matrix)
        @ np.asarray(record.correction_commanded)
        + np.asarray(model.config.pulse_bias)
    )
    pulse_systematic_norm = float(
        np.linalg.norm(record_pulse_mean - np.asarray(record.correction_commanded))
    )
    virtual_systematic_error = float(
        model.config.virtual_rotation_gain * record.virtual_rotation_commanded_rad
        + model.config.virtual_rotation_bias_rad
        - record.virtual_rotation_requested_rad
    )

    latency_model = ControlImperfectionModel(
        replace(
            _validation_config(),
            virtual_rotation_noise_sigma_rad=0.0,
            active_relative_gain_sigma=0.0,
            active_displacement_covariance=np.zeros((2, 2)),
            pulse_gain_matrix=np.eye(2),
            pulse_bias=(0.0, 0.0),
            virtual_rotation_gain=1.0,
            virtual_rotation_bias_rad=0.0,
        )
    )
    latency_traces: list[float] = []
    for latency in (0.0, 2.0, 5.0, 10.0):
        _, covariance = latency_model.analytic_residual_moments(
            replace(request, latency_us=latency),
            residual_before_latency=pre,
        )
        latency_traces.append(float(np.trace(covariance)))

    ideal_model = ControlImperfectionModel(ideal_control_imperfection_config())
    ideal_request = ControlActionRequest(
        cycle_index=0,
        correction_command=pre,
        virtual_rotation_command_rad=0.37,
        latency_us=0.0,
    )
    ideal_batch = ideal_model.sample_fixed_request(
        ideal_request,
        residual_before_latency=pre,
        samples=10_000,
        seed=normalized_seed + 1,
    )
    ideal_max = float(np.max(np.abs(ideal_batch.residual_after_action)))
    checks = MappingProxyType(
        {
            "mean_within_5_standard_errors": maximum_z <= 5.0,
            "covariance_relative_error_below_2_percent": relative_covariance <= 0.02,
            "quantization_rms_error_strictly_decreases_with_bits": all(
                later < earlier
                for earlier, later in zip(
                    quantization_errors,
                    quantization_errors[1:],
                )
            ),
            "latency_covariance_trace_strictly_increases": all(
                later > earlier
                for earlier, later in zip(latency_traces, latency_traces[1:])
            ),
            "ideal_endpoint_is_exact": ideal_max == 0.0,
            "quantization_produces_integer_codes": all(
                code is not None
                for code in (
                    record.awg_amplitude_code,
                    record.awg_phase_code,
                    *record.dac_iq_codes,
                    record.virtual_rotation_code,
                )
            ),
            "production_command_is_not_saturated": not record.displacement_saturated,
        }
    )
    return ControlImperfectionValidationResult(
        samples=samples,
        seed=normalized_seed,
        empirical_mean=(float(empirical_mean[0]), float(empirical_mean[1])),
        analytic_mean=(float(analytic_mean[0]), float(analytic_mean[1])),
        maximum_mean_z_score=maximum_z,
        empirical_covariance=tuple(
            tuple(float(value) for value in row) for row in empirical_covariance
        ),  # type: ignore[arg-type]
        analytic_covariance=tuple(
            tuple(float(value) for value in row) for row in analytic_covariance
        ),  # type: ignore[arg-type]
        covariance_relative_frobenius_error=relative_covariance,
        quantization_bits=quantization_bits,
        quantization_rms_error=tuple(quantization_errors),
        pulse_systematic_displacement_error_norm=pulse_systematic_norm,
        virtual_rotation_systematic_error_rad=virtual_systematic_error,
        latency_covariance_trace=tuple(latency_traces),
        ideal_endpoint_max_abs_residual=ideal_max,
        checks=checks,
    )


def write_control_imperfection_validation(
    result: ControlImperfectionValidationResult,
    output_path: str | Path,
) -> Path:
    if not isinstance(result, ControlImperfectionValidationResult):
        raise TypeError("result must be a ControlImperfectionValidationResult")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.as_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=_model_module.__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=2026071423)
    arguments = parser.parse_args()
    result = run_control_imperfection_validation(
        samples=arguments.samples,
        seed=arguments.seed,
    )
    write_control_imperfection_validation(result, arguments.output)
    print(json.dumps(result.as_dict()["checks"], ensure_ascii=False))


if __name__ == "__main__":
    main()


__all__ = [
    "ControlImperfectionValidationResult",
    "run_control_imperfection_validation",
    "write_control_imperfection_validation",
]
