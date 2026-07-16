"""T4.2.1 exhaustive validation for the version-bound parametric MAP-LUT."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import inspect
import json
import textwrap
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from cnn_fpga.decoder.parametric_map_lut import (
    compile_active_param_bank,
    derive_axis_map_model,
    exact_quantized_llr_code,
)
from cnn_fpga.runtime.param_bank import DecoderRuntimeParams, ParamBank
from cnn_fpga.runtime.parametric_map_lut import (
    ParametricMAPLUTConfig,
    ParametricMAPLUTImage,
    ParametricMAPLUTInput,
    ParametricMAPLUTPipeline,
    ParametricMAPLUTRuntime,
    _rounded_signed_shift,
    resource_contract,
    software_decode_syndrome_code,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON = ROOT / "docs" / "t4_2_1_parametric_map_lut_validation.json"
DEFAULT_CSV = ROOT / "docs" / "t4_2_1_parametric_map_lut_source_data.csv"
DEFAULT_IMAGES = ROOT / "docs" / "t4_2_1_parametric_map_lut_bank_images.json"
SCHEMA_VERSION = "t4.2.1-parametric-map-lut-validation-v1"


@dataclass(frozen=True)
class ValidationConfig:
    adc_bits: int = 10
    selected_address_bits: int = 8
    convergence_address_bits: tuple[int, ...] = (5, 6, 7, 8)
    llr_integer_bits: int = 9
    llr_fractional_bits: int = 12
    maximum_action_mismatch_rate: float = 0.0
    maximum_mean_abs_llr_code_error: float = 0.75


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _implementation_sha256() -> str:
    digest = hashlib.sha256()
    for relative in (
        "cnn_fpga/runtime/parametric_map_lut.py",
        "cnn_fpga/decoder/parametric_map_lut.py",
        "cnn_fpga/decoder/param_mapper.py",
        "cnn_fpga/benchmark/parametric_map_lut_validation.py",
    ):
        digest.update(relative.encode("utf-8"))
        digest.update((ROOT / relative).read_bytes())
    return digest.hexdigest()


def _params_from_model(
    *,
    mean_fraction: tuple[float, float],
    sigma_fraction: tuple[float, float],
    rho: float,
    measurement_sigma_fraction: float,
    lattice: float,
    profile_id: str,
) -> tuple[DecoderRuntimeParams, dict[str, Any]]:
    mean = np.asarray(mean_fraction, dtype=np.float64) * lattice
    sigma = np.asarray(sigma_fraction, dtype=np.float64) * lattice
    covariance = np.asarray(
        [
            [sigma[0] ** 2, rho * sigma[0] * sigma[1]],
            [rho * sigma[0] * sigma[1], sigma[1] ** 2],
        ],
        dtype=np.float64,
    )
    measurement_sigma = measurement_sigma_fraction * lattice
    measurement = np.eye(2, dtype=np.float64) * measurement_sigma**2
    gain = covariance @ np.linalg.inv(covariance + measurement)
    bias = (np.eye(2, dtype=np.float64) - gain) @ mean
    params = DecoderRuntimeParams(
        K=gain,
        b=bias,
        metadata={
            "measurement_cov": measurement.tolist(),
            "alpha_bias": 1.0,
            "profile_id": profile_id,
            "scope": "registered_t4.2.1_effective_axis_model",
        },
    )
    truth = {
        "profile_id": profile_id,
        "mean": mean.tolist(),
        "sigma": sigma.tolist(),
        "rho": rho,
        "covariance": covariance.tolist(),
        "measurement_covariance": measurement.tolist(),
    }
    return params, truth


def registered_parameter_profiles(
    config: ParametricMAPLUTConfig,
) -> tuple[tuple[DecoderRuntimeParams, dict[str, Any]], ...]:
    rows = (
        ("center_balanced", (0.00, 0.00), (0.15, 0.18), 0.00, 0.12),
        ("small_drift_posneg", (0.12, -0.08), (0.20, 0.16), 0.35, 0.10),
        ("large_drift_posneg", (0.30, -0.25), (0.25, 0.22), -0.45, 0.14),
        ("large_drift_negpos", (-0.35, 0.28), (0.18, 0.30), 0.60, 0.11),
        ("near_edge_q", (0.42, 0.10), (0.32, 0.24), -0.55, 0.13),
        ("near_edge_p", (-0.18, -0.40), (0.14, 0.28), 0.50, 0.09),
        ("anisotropic_q", (0.05, 0.33), (0.27, 0.19), -0.30, 0.15),
        ("anisotropic_p", (-0.28, -0.15), (0.23, 0.31), 0.42, 0.12),
    )
    return tuple(
        _params_from_model(
            profile_id=profile_id,
            mean_fraction=mean,
            sigma_fraction=sigma,
            rho=rho,
            measurement_sigma_fraction=measurement,
            lattice=config.lattice,
        )
        for profile_id, mean, sigma, rho, measurement in rows
    )


def _nearest_code(
    syndrome_code: int, phase: int, image: ParametricMAPLUTImage
) -> int:
    config = image.config
    address = syndrome_code >> config.fraction_bits
    fraction = syndrome_code & ((1 << config.fraction_bits) - 1)
    numerator = (fraction << 1) + 1
    choose_upper = numerator >= (1 << config.fraction_bits)
    return image.table_codes[phase][address + int(choose_upper)]


def _evaluate_images(
    images: Sequence[ParametricMAPLUTImage],
    profile_ids: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_rows: list[dict[str, Any]] = []
    interpolation_errors: list[int] = []
    nearest_errors: list[int] = []
    action_mismatches = 0
    phase_table_difference_count = 0
    for image, profile_id in zip(images, profile_ids):
        runtime = ParametricMAPLUTRuntime(image)
        phase_table_difference_count += sum(
            left != right for left, right in zip(*image.table_codes)
        )
        for phase in (0, 1):
            for code in range(image.config.adc_levels):
                decision = runtime.decode_code(
                    ParametricMAPLUTInput(code, code, phase, image.active_bank_version)
                )
                exact = exact_quantized_llr_code(code, phase, image)
                nearest = _nearest_code(code, phase, image)
                error = abs(decision.llr_code - exact)
                nearest_error = abs(nearest - exact)
                mismatch = (decision.llr_code < 0) != (exact < 0)
                interpolation_errors.append(error)
                nearest_errors.append(nearest_error)
                action_mismatches += int(mismatch)
                source_rows.append(
                    {
                        "profile_id": profile_id,
                        "active_bank_version": image.active_bank_version,
                        "phase_bit": phase,
                        "phase_label": decision.phase_label,
                        "syndrome_code": code,
                        "syndrome_center": software_decode_syndrome_code(code, image.config),
                        "address": decision.address,
                        "fraction_code": decision.fraction_code,
                        "interpolated_llr_code": decision.llr_code,
                        "exact_quantized_llr_code": exact,
                        "nearest_llr_code": nearest,
                        "abs_interpolation_error_code": error,
                        "abs_nearest_error_code": nearest_error,
                        "logical_action": decision.logical_action,
                        "action_mismatch": int(mismatch),
                        "image_sha256": image.image_sha256,
                    }
                )
    count = len(interpolation_errors)
    metrics = {
        "rows": count,
        "mean_abs_llr_code_error": float(np.mean(interpolation_errors)),
        "max_abs_llr_code_error": int(max(interpolation_errors)),
        "nearest_mean_abs_llr_code_error": float(np.mean(nearest_errors)),
        "interpolation_to_nearest_error_ratio": float(
            np.mean(interpolation_errors) / np.mean(nearest_errors)
        ),
        "action_mismatch_count": action_mismatches,
        "action_mismatch_rate": action_mismatches / count,
        "phase_table_difference_count": phase_table_difference_count,
    }
    return source_rows, metrics


def _convergence_audit(
    bank_params: Sequence[DecoderRuntimeParams],
    config: ValidationConfig,
) -> list[dict[str, Any]]:
    rows = []
    for address_bits in config.convergence_address_bits:
        runtime_config = ParametricMAPLUTConfig(
            adc_bits=config.adc_bits,
            address_bits=address_bits,
            llr_integer_bits=config.llr_integer_bits,
            llr_fractional_bits=config.llr_fractional_bits,
        )
        errors: list[int] = []
        mismatches = 0
        for version, params in enumerate(bank_params):
            bank = ParamBank(params)
            image = compile_active_param_bank(bank, runtime_config)
            runtime = ParametricMAPLUTRuntime(image)
            for phase in (0, 1):
                for code in range(runtime_config.adc_levels):
                    actual = runtime.decode_code(
                        ParametricMAPLUTInput(code, code, phase, 0)
                    ).llr_code
                    exact = exact_quantized_llr_code(code, phase, image)
                    errors.append(abs(actual - exact))
                    mismatches += int((actual < 0) != (exact < 0))
        rows.append(
            {
                "address_bits": address_bits,
                "table_entries_per_phase_with_guard": runtime_config.table_entries,
                "mean_abs_llr_code_error": float(np.mean(errors)),
                "max_abs_llr_code_error": int(max(errors)),
                "action_mismatch_count": mismatches,
            }
        )
    return rows


def _pipeline_audit(images: Sequence[ParametricMAPLUTImage]) -> dict[str, Any]:
    old, new = images[0], images[1]
    probe_code = min(700, old.config.adc_levels - 1)
    pipeline = ParametricMAPLUTPipeline(old)
    pipeline.step(
        0, ParametricMAPLUTInput(0, probe_code, 0, old.active_bank_version)
    )
    pipeline.load_image(new)
    pipeline.step(
        1, ParametricMAPLUTInput(1, probe_code, 0, new.active_bank_version)
    )
    outputs = []
    for cycle in range(2, 7):
        output = pipeline.step(cycle)
        if output is not None:
            outputs.append(output)

    ii_pipeline = ParametricMAPLUTPipeline(old)
    ii_outputs = []
    for cycle in range(21):
        request = (
            ParametricMAPLUTInput(cycle, 100 + cycle, cycle % 2, old.active_bank_version)
            if cycle < 16
            else None
        )
        output = ii_pipeline.step(cycle, request)
        if output is not None:
            ii_outputs.append(output)
    return {
        "old_inflight_image_latched": (
            len(outputs) == 2
            and outputs[0].image_sha256 == old.image_sha256
            and outputs[1].image_sha256 == new.image_sha256
        ),
        "old_inflight_version_latched": (
            len(outputs) == 2
            and outputs[0].active_bank_version == old.active_bank_version
            and outputs[1].active_bank_version == new.active_bank_version
        ),
        "latencies": [item.valid_cycle - item.input_cycle for item in outputs],
        "ii_one_input_cycles": [item.input_cycle for item in ii_outputs],
        "ii_one_valid_cycles": [item.valid_cycle for item in ii_outputs],
        "ii_one_output_count": len(ii_outputs),
    }


def _negative_audit(image: ParametricMAPLUTImage) -> dict[str, bool]:
    results: dict[str, bool] = {}
    runtime = ParametricMAPLUTRuntime(image)
    probes = {
        "stale_version_rejected": lambda: runtime.decode_code(
            ParametricMAPLUTInput(0, 0, 0, image.active_bank_version + 1)
        ),
        "bad_adc_code_rejected": lambda: runtime.decode_code(
            ParametricMAPLUTInput(
                0, image.config.adc_levels, 0, image.active_bank_version
            )
        ),
        "bad_phase_rejected": lambda: ParametricMAPLUTInput(
            0, 0, 2, image.active_bank_version
        ),
    }
    for name, probe in probes.items():
        try:
            probe()
        except (TypeError, ValueError):
            results[name] = True
        else:
            results[name] = False
    table = list(image.table_codes[0])
    table[3] += 1
    tampered = replace(image, table_codes=(tuple(table), image.table_codes[1]))
    try:
        ParametricMAPLUTRuntime(tampered)
    except ValueError:
        results["crc_sha_tamper_rejected"] = True
    else:
        results["crc_sha_tamper_rejected"] = False
    return results


def _integer_kernel_audit() -> dict[str, Any]:
    trees = [
        ast.parse(textwrap.dedent(inspect.getsource(function)))
        for function in (ParametricMAPLUTRuntime.decode_code, _rounded_signed_shift)
    ]
    division_nodes = sum(
        isinstance(node, ast.Div) for tree in trees for node in ast.walk(tree)
    )
    forbidden_calls = sorted(
        {
            node.func.id
            for tree in trees
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"exp", "log", "sqrt", "pow"}
        }
    )
    return {
        "float_division_ast_nodes": division_nodes,
        "forbidden_math_calls": forbidden_calls,
        "integer_only": division_nodes == 0 and not forbidden_calls,
    }


def _write_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_validation(
    config: ValidationConfig | None = None,
    *,
    json_path: Path | None = None,
    csv_path: Path | None = None,
    images_path: Path | None = None,
) -> dict[str, Any]:
    actual = ValidationConfig() if config is None else config
    runtime_config = ParametricMAPLUTConfig(
        adc_bits=actual.adc_bits,
        address_bits=actual.selected_address_bits,
        llr_integer_bits=actual.llr_integer_bits,
        llr_fractional_bits=actual.llr_fractional_bits,
    )
    profiles = registered_parameter_profiles(runtime_config)
    bank_params = [params for params, _ in profiles]
    profile_truth = [truth for _, truth in profiles]
    profile_ids = [row["profile_id"] for row in profile_truth]

    bank = ParamBank(bank_params[0])
    images = [compile_active_param_bank(bank, runtime_config)]
    for index, params in enumerate(bank_params[1:], start=1):
        bank.stage_update(params, commit_epoch=index, staged_epoch=index - 1)
        committed = bank.commit_if_ready(index)
        if committed is None or committed.version != index:
            raise RuntimeError("real ParamBank failed registered atomic commit sequence")
        images.append(compile_active_param_bank(bank, runtime_config))

    derivation = []
    for params, truth in profiles:
        model = derive_axis_map_model(params)
        derivation.append(
            {
                "profile_id": truth["profile_id"],
                "mean_max_abs_error": float(
                    np.max(np.abs(np.asarray(model.mean) - np.asarray(truth["mean"])))
                ),
                "covariance_max_abs_error": float(
                    np.max(
                        np.abs(
                            np.asarray(model.covariance)
                            - np.asarray(truth["covariance"])
                        )
                    )
                ),
                "K_reconstruction_max_abs_error": model.derivation_residual_max_abs,
            }
        )

    source_rows, selected_metrics = _evaluate_images(images, profile_ids)
    convergence = _convergence_audit(bank_params, actual)
    pipeline = _pipeline_audit(images)
    negatives = _negative_audit(images[0])
    integer_kernel = _integer_kernel_audit()
    resources = resource_contract(runtime_config)

    output_csv = DEFAULT_CSV if csv_path is None else Path(csv_path)
    output_images = DEFAULT_IMAGES if images_path is None else Path(images_path)
    _write_csv(source_rows, output_csv)
    image_payload = {
        "schema_version": "t4.2.1-parametric-map-lut-bank-images-v1",
        "images": [image.to_dict(include_tables=True) for image in images],
    }
    output_images.write_text(
        json.dumps(image_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    convergence_errors = [row["mean_abs_llr_code_error"] for row in convergence]
    gates = {
        "eight_registered_parameter_banks_compile": len(images) == 8,
        "real_param_bank_versions_are_atomic_and_monotonic": [
            image.active_bank_version for image in images
        ]
        == list(range(8)),
        "active_K_b_inverse_recovers_registered_models": max(
            max(row["mean_max_abs_error"], row["covariance_max_abs_error"])
            for row in derivation
        )
        < 2.0e-12,
        "K_reconstruction_is_machine_precision": max(
            row["K_reconstruction_max_abs_error"] for row in derivation
        )
        < 2.0e-12,
        "all_adc_codes_and_both_phases_are_exhaustive": selected_metrics["rows"]
        == 8 * 2 * runtime_config.adc_levels,
        "phase_tables_are_not_silently_shared": selected_metrics[
            "phase_table_difference_count"
        ]
        > runtime_config.table_entries,
        "selected_lut_has_zero_action_mismatch": selected_metrics[
            "action_mismatch_rate"
        ]
        <= actual.maximum_action_mismatch_rate,
        "selected_lut_meets_llr_code_error_budget": selected_metrics[
            "mean_abs_llr_code_error"
        ]
        <= actual.maximum_mean_abs_llr_code_error,
        "linear_interpolation_beats_nearest_rom": selected_metrics[
            "interpolation_to_nearest_error_ratio"
        ]
        < 0.25,
        "address_grid_error_decreases_strictly": all(
            left > right for left, right in zip(convergence_errors, convergence_errors[1:])
        ),
        "all_convergence_grids_preserve_actions": all(
            row["action_mismatch_count"] == 0 for row in convergence
        ),
        "image_crc_and_sha_verify": all(
            image.verify() is None for image in images
        ),
        "image_hashes_bind_each_param_bank": len(
            {image.source_params_sha256 for image in images}
        )
        == len(images),
        "pipeline_latency_is_exactly_five_cycles": pipeline["latencies"] == [5, 5],
        "pipeline_accepts_one_input_per_cycle": pipeline["ii_one_output_count"] == 16
        and pipeline["ii_one_valid_cycles"] == list(range(5, 21)),
        "inflight_request_latches_old_bank": pipeline["old_inflight_image_latched"]
        and pipeline["old_inflight_version_latched"],
        "all_negative_paths_fail_closed": all(negatives.values()),
        "online_kernel_has_no_float_division_exp_or_log": integer_kernel["integer_only"],
        "resource_fields_remain_non_measured": resources["target_lut_count"] is None
        and resources["target_bram_count"] is None
        and resources["fmax_mhz"] is None
        and not resources["rtl_measured"]
        and not resources["board_measured"],
        "scope_is_axis_marginal_not_joint_or_hardware": True,
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "task_id": "T4.2.1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "implementation_sha256": _implementation_sha256(),
        "validation_config": actual.__dict__,
        "runtime_config": runtime_config.to_dict(),
        "profiles": profile_truth,
        "derivation_audit": derivation,
        "selected_metrics": selected_metrics,
        "convergence": convergence,
        "pipeline_audit": pipeline,
        "negative_audit": negatives,
        "integer_kernel_audit": integer_kernel,
        "resource_contract": resources,
        "image_artifact": {
            "path": str(output_images.relative_to(ROOT)).replace("\\", "/"),
            "sha256": _sha256_file(output_images),
            "image_sha256s": [image.image_sha256 for image in images],
        },
        "source_data": {
            "path": str(output_csv.relative_to(ROOT)).replace("\\", "/"),
            "rows": len(source_rows),
            "sha256": _sha256_file(output_csv),
        },
        "gate_summary": {
            "passed": sum(gates.values()),
            "failed": len(gates) - sum(gates.values()),
            "gates": gates,
        },
        "claim_boundary": {
            "allowed": "version-bound integer axis-marginal periodic MAP-LUT software pipeline contract with exhaustive ADC/phase validation",
            "forbidden": "full correlated 2D MAP optimality, event-FSM/frame integration, fixed-point end-to-end LER, RTL synthesis, FPGA timing, or board/device measurement",
        },
    }
    output_json = DEFAULT_JSON if json_path is None else Path(json_path)
    output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    args = parser.parse_args(argv)
    payload = run_validation(
        json_path=args.json, csv_path=args.csv, images_path=args.images
    )
    print(json.dumps(payload["gate_summary"], indent=2))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
