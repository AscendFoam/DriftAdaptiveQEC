"""T4.2.4 end-to-end bit-accurate fast-path and paired LER validation."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import inspect
import json
import textwrap
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from cnn_fpga.benchmark.parametric_map_lut_validation import registered_parameter_profiles
from cnn_fpga.decoder.parametric_map_lut import (
    compile_parametric_map_lut,
    exact_quantized_llr_code,
)
from cnn_fpga.runtime.fast_path_fixed_point import (
    BitAccurateFastPath,
    FastPathCodeInput,
    FastPathFixedPointContract,
    build_code_input_from_replay,
    encode_unit_interval_replay,
    encode_unsigned_age_replay,
)
from cnn_fpga.runtime.parametric_map_lut import (
    ParametricMAPLUTConfig,
    ParametricMAPLUTImage,
    ParametricMAPLUTInput,
    ParametricMAPLUTRuntime,
)
from physics.ideal_gkp_decoder import llr_1d


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON = ROOT / "docs" / "t4_2_4_fast_path_fixed_point_validation.json"
DEFAULT_LER_CSV = ROOT / "docs" / "t4_2_4_fast_path_fixed_point_ler.csv"
DEFAULT_CODE_CSV = ROOT / "docs" / "t4_2_4_fast_path_fixed_point_exhaustive_codes.csv"
SCHEMA_VERSION = "t4.2.4-fast-path-fixed-point-validation-v1"
PROFILE_CONFIGS = {
    "low_p6_a4_q5_6": ParametricMAPLUTConfig(
        adc_bits=6, address_bits=4, llr_integer_bits=5, llr_fractional_bits=6
    ),
    "medium_p8_a6_q7_10": ParametricMAPLUTConfig(
        adc_bits=8, address_bits=6, llr_integer_bits=7, llr_fractional_bits=10
    ),
    "selected_p10_a8_q9_12": ParametricMAPLUTConfig(),
    "dense_p12_a10_q10_14": ParametricMAPLUTConfig(
        adc_bits=12, address_bits=10, llr_integer_bits=10, llr_fractional_bits=14
    ),
}
SEEDS = (42401, 42403, 42407, 42409)


def _implementation_sha256() -> str:
    digest = hashlib.sha256()
    for relative in (
        "cnn_fpga/runtime/fast_path_fixed_point.py",
        "cnn_fpga/runtime/parametric_map_lut.py",
        "cnn_fpga/runtime/conservative_fallback.py",
        "cnn_fpga/runtime/experimental_event_fsm.py",
        "cnn_fpga/benchmark/fast_path_fixed_point_validation.py",
    ):
        digest.update(relative.encode("utf-8"))
        digest.update((ROOT / relative).read_bytes())
    return digest.hexdigest()


def _images(config: ParametricMAPLUTConfig) -> tuple[ParametricMAPLUTImage, ...]:
    return tuple(
        compile_parametric_map_lut(params, active_bank_version=index, config=config)
        for index, (params, _) in enumerate(registered_parameter_profiles(config))
    )


def _profile_images() -> dict[str, tuple[ParametricMAPLUTImage, ...]]:
    return {name: _images(config) for name, config in PROFILE_CONFIGS.items()}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _row_hash(rows: Sequence[dict[str, Any]]) -> str:
    payload = json.dumps(list(rows), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _exhaustive_code_audit(
    profiles: dict[str, tuple[ParametricMAPLUTImage, ...]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for profile_name, images in profiles.items():
        errors: list[float] = []
        mismatches = 0
        for image in images:
            runtime = ParametricMAPLUTRuntime(image)
            scale = float(1 << image.config.llr_fractional_bits)
            for phase in (0, 1):
                for code in range(image.config.adc_levels):
                    decision = runtime.decode_code(
                        ParametricMAPLUTInput(0, code, phase, image.active_bank_version)
                    )
                    exact_code = exact_quantized_llr_code(code, phase, image)
                    error_code = abs(decision.llr_code - exact_code)
                    mismatch = decision.logical_flip != (exact_code < 0)
                    errors.append(error_code / scale)
                    mismatches += int(mismatch)
                    rows.append(
                        {
                            "profile": profile_name,
                            "active_bank_version": image.active_bank_version,
                            "phase_bit": phase,
                            "syndrome_code": code,
                            "runtime_llr_code": decision.llr_code,
                            "exact_quantized_llr_code": exact_code,
                            "absolute_llr_value_error": error_code / scale,
                            "hard_action_mismatch": int(mismatch),
                        }
                    )
        summaries[profile_name] = {
            "rows": sum(
                2 * image.config.adc_levels for image in images
            ),
            "mean_abs_llr_value_error": float(np.mean(errors)),
            "max_abs_llr_value_error": float(np.max(errors)),
            "hard_action_mismatch_count": mismatches,
            "hard_action_mismatch_rate": mismatches / len(errors),
        }
    return rows, summaries


def _trace(image: ParametricMAPLUTImage, *, seed: int, samples: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed + 1009 * image.active_bank_version)
    phase = np.arange(samples, dtype=np.int64) % 2
    means = np.asarray(image.model_mean)[phase]
    sigmas = np.asarray(image.model_sigma)[phase]
    unwrapped = rng.normal(means, sigmas)
    half = 0.5 * image.config.lattice
    cell = np.floor((unwrapped + half) / image.config.lattice).astype(np.int64)
    syndrome = unwrapped - cell.astype(np.float64) * image.config.lattice
    truth = np.mod(cell, 2) != 0
    exact_llr = np.empty(samples, dtype=np.float64)
    for axis in (0, 1):
        mask = phase == axis
        exact_llr[mask] = np.asarray(
            llr_1d(
                syndrome[mask],
                image.model_sigma[axis],
                mean=image.model_mean[axis],
                lattice=image.config.lattice,
            ),
            dtype=np.float64,
        )
    return {
        "phase": phase,
        "syndrome": syndrome,
        "truth": truth,
        "float_action": exact_llr < 0.0,
    }


def _simulate_ler(
    profiles: dict[str, tuple[ParametricMAPLUTImage, ...]],
    *,
    samples_per_run: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = profiles["selected_p10_a8_q9_12"]
    traces = {
        (bank, seed): _trace(selected[bank], seed=seed, samples=samples_per_run)
        for bank in range(8)
        for seed in SEEDS
    }
    rows: list[dict[str, Any]] = []
    latency_exact = True
    ii_exact = True
    both_axes_updated: set[int] = set()
    for profile_name, images in profiles.items():
        for bank, image in enumerate(images):
            for seed in SEEDS:
                trace = traces[(bank, seed)]
                path = BitAccurateFastPath(images)
                predictions = np.zeros(samples_per_run, dtype=np.bool_)
                fallback_count = 0
                accepted_count = 0
                action_cycles: list[int] = []
                for index in range(samples_per_run):
                    code_input, saturation = build_code_input_from_replay(
                        cycle_index=5 + index,
                        syndrome=float(trace["syndrome"][index]),
                        syndrome_x="g",
                        syndrome_z="g",
                        quadrature_phase_bit=int(trace["phase"][index]),
                        image=image,
                        parameter_age_cycles=index % 65,
                        ood_score=0.0,
                    )
                    if any(saturation.values()):
                        raise RuntimeError("model-matched LER trace unexpectedly saturated")
                    result = path.step_codes(code_input)
                    action = result.fallback_action
                    hardware = action.hardware_action
                    axis = int(trace["phase"][index])
                    predictions[index] = (
                        hardware.pauli_frame_delta_x if axis == 0 else hardware.pauli_frame_delta_z
                    )
                    if hardware.pauli_frame_delta_x:
                        both_axes_updated.add(0)
                    if hardware.pauli_frame_delta_z:
                        both_axes_updated.add(1)
                    fallback_count += int(action.status != "healthy")
                    accepted_count += int(action.map_decision_accepted)
                    action_cycles.append(hardware.action_cycle)
                    latency_exact &= hardware.action_cycle - hardware.source_cycle == 6
                ii_exact &= action_cycles == list(range(6, 6 + samples_per_run))
                truth = trace["truth"]
                float_action = trace["float_action"]
                quant_errors = predictions != truth
                float_errors = float_action != truth
                rows.append(
                    {
                        "profile": profile_name,
                        "active_bank_version": bank,
                        "seed": seed,
                        "samples": samples_per_run,
                        "quantized_error_count": int(np.count_nonzero(quant_errors)),
                        "quantized_ler": float(np.mean(quant_errors)),
                        "float_error_count": int(np.count_nonzero(float_errors)),
                        "float_ler": float(np.mean(float_errors)),
                        "quantized_minus_float_ler": float(np.mean(quant_errors) - np.mean(float_errors)),
                        "action_disagreement_rate": float(np.mean(predictions != float_action)),
                        "fallback_count": fallback_count,
                        "map_accepted_count": accepted_count,
                    }
                )
    diagnostics = {
        "latency_exact": latency_exact,
        "initiation_interval_exact": ii_exact,
        "both_axes_frame_updated": sorted(both_axes_updated),
        "samples_per_run": samples_per_run,
        "seeds": list(SEEDS),
    }
    return rows, diagnostics


def _paired_bootstrap(values: np.ndarray, *, seed: int = 42424) -> dict[str, float]:
    if values.ndim != 1 or values.size < 2:
        raise ValueError("paired bootstrap requires at least two cluster values")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(10_000, values.size))
    means = values[indices].mean(axis=1)
    return {
        "mean": float(np.mean(values)),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "clusters": int(values.size),
    }


def _aggregate_ler(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for profile in PROFILE_CONFIGS:
        selected = [row for row in rows if row["profile"] == profile]
        differences = np.asarray(
            [row["quantized_minus_float_ler"] for row in selected], dtype=np.float64
        )
        result[profile] = {
            "quantized_ler_mean": float(np.mean([row["quantized_ler"] for row in selected])),
            "float_ler_mean": float(np.mean([row["float_ler"] for row in selected])),
            "action_disagreement_mean": float(
                np.mean([row["action_disagreement_rate"] for row in selected])
            ),
            "paired_quantized_minus_float": _paired_bootstrap(differences),
            "fallback_count": int(sum(row["fallback_count"] for row in selected)),
            "map_accepted_count": int(sum(row["map_accepted_count"] for row in selected)),
        }
    return result


def _negative_audit(images: Sequence[ParametricMAPLUTImage]) -> dict[str, bool]:
    image = images[0]
    invalid_path = BitAccurateFastPath(images)
    invalid_input, _ = build_code_input_from_replay(
        cycle_index=5,
        syndrome=float("nan"),
        syndrome_x="g",
        syndrome_z="g",
        quadrature_phase_bit=0,
        image=image,
        parameter_age_cycles=0,
        ood_score=0.0,
    )
    invalid = invalid_path.step_codes(invalid_input).fallback_action

    below_code, _ = encode_unit_interval_replay(192.0 / 255.0, 8)
    above_code, _ = encode_unit_interval_replay(193.0 / 255.0, 8)
    age_code, age_sat = encode_unsigned_age_replay(65_536, 16)
    return {
        "nonfinite_syndrome_becomes_frame_hold": (
            invalid.conservative_action == "frame_hold"
            and "observation_invalid" in invalid.fault_flags
            and "map_decision_missing" in invalid.fault_flags
        ),
        "ood_boundary_codes_are_exact": below_code == 192 and above_code == 193,
        "age_overflow_saturates_to_stale_code": age_code == 65_535 and age_sat,
    }


def _online_integer_audit() -> dict[str, Any]:
    source = textwrap.dedent(inspect.getsource(BitAccurateFastPath.step_codes))
    tree = ast.parse(source)
    forbidden_divisions = sum(
        isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div) for node in ast.walk(tree)
    )
    forbidden_calls = [
        token for token in ("exp(", "log(", "sqrt(", "float(") if token in source
    ]
    return {
        "float_division_ast_nodes": forbidden_divisions,
        "forbidden_float_math_tokens": forbidden_calls,
        "integer_only": forbidden_divisions == 0 and not forbidden_calls,
    }


def run_validation(
    *,
    samples_per_run: int = 2_048,
    json_path: Path | None = None,
    ler_csv_path: Path | None = None,
    code_csv_path: Path | None = None,
) -> dict[str, Any]:
    if samples_per_run < 1_024:
        raise ValueError("samples_per_run must be at least 1024")
    profiles = _profile_images()
    code_rows, code_summary = _exhaustive_code_audit(profiles)
    ler_rows, timing = _simulate_ler(profiles, samples_per_run=samples_per_run)
    repeat_profiles = _profile_images()
    repeat_code_rows, _ = _exhaustive_code_audit(repeat_profiles)
    repeat_ler_rows, repeat_timing = _simulate_ler(
        repeat_profiles, samples_per_run=samples_per_run
    )
    deterministic = (
        _row_hash(code_rows) == _row_hash(repeat_code_rows)
        and _row_hash(ler_rows) == _row_hash(repeat_ler_rows)
        and timing == repeat_timing
    )
    ler_summary = _aggregate_ler(ler_rows)
    selected_code = code_summary["selected_p10_a8_q9_12"]
    dense_code = code_summary["dense_p12_a10_q10_14"]
    low_code = code_summary["low_p6_a4_q5_6"]
    selected_ler = ler_summary["selected_p10_a8_q9_12"]
    dense_ler = ler_summary["dense_p12_a10_q10_14"]
    low_ler = ler_summary["low_p6_a4_q5_6"]
    contract = FastPathFixedPointContract()
    negatives = _negative_audit(profiles["selected_p10_a8_q9_12"])
    integer_audit = _online_integer_audit()
    hardware_fields = {
        "target_lut_count": None,
        "target_ff_count": None,
        "target_bram_count": None,
        "target_dsp_count": None,
        "fmax_mhz": None,
        "rtl_measured": False,
        "board_measured": False,
    }
    arithmetic_rules = {
        "adc_replay": "floor to one of 2^B half-open-cell bins; out-of-cell/nonfinite is clipped only for a code and separately invalidates observation",
        "lut_compile": "float LLR multiplied by 2^F, round-to-nearest ties-to-even, then signed-word saturation",
        "lut_interpolation": "integer half-bin numerator with signed right shift round-to-nearest ties-to-even, then signed-word saturation",
        "logical_action": "strict signed LLR code < 0; zero maps to identity",
        "ood_replay": "unit interval multiplied by 2^B-1, round-to-nearest ties-to-even, unsigned saturation",
        "parameter_age": "unsigned B-bit saturation; saturated maximum remains stale",
        "event_health_counters": "unsigned saturating increment, no wrap",
        "pauli_phase_frames": "GF(2) XOR and unsigned modulo-2^B half-turn addition",
        "crc_sha_version": "exact digital equality and monotonic version comparison; no numeric rounding",
    }
    gates = {
        "selected_contract_freezes_complete_word_and_timing_table": contract.llr_word_bits == 22
        and contract.source_to_action_cycles == 6
        and contract.representation_proxy()["live_event_state_bits"] == 55
        and contract.representation_proxy()["live_health_state_and_input_bits"] == 182,
        "four_precision_profiles_compile_all_eight_registered_banks": set(profiles)
        == set(PROFILE_CONFIGS)
        and all(len(images) == 8 for images in profiles.values()),
        "exhaustive_code_audit_covers_every_profile_bank_phase_and_adc_code": len(code_rows)
        == sum(8 * 2 * config.adc_levels for config in PROFILE_CONFIGS.values()),
        "selected_and_dense_profiles_have_zero_hard_action_mismatch": selected_code[
            "hard_action_mismatch_count"
        ]
        == 0
        and dense_code["hard_action_mismatch_count"] == 0,
        "selected_interpolated_llr_value_error_is_below_one_code": selected_code[
            "mean_abs_llr_value_error"
        ]
        < 1.0 / (1 << PROFILE_CONFIGS["selected_p10_a8_q9_12"].llr_fractional_bits),
        "dense_profile_reduces_llr_value_error_below_selected": dense_code[
            "mean_abs_llr_value_error"
        ]
        < selected_code["mean_abs_llr_value_error"],
        "low_precision_retains_measured_nonzero_quantization_effect": low_code[
            "mean_abs_llr_value_error"
        ]
        > selected_code["mean_abs_llr_value_error"],
        "paired_ler_matrix_covers_four_profiles_eight_banks_four_seeds": len(ler_rows)
        == 4 * 8 * len(SEEDS),
        "selected_ler_impact_is_bounded_to_two_per_mille": abs(
            selected_ler["paired_quantized_minus_float"]["mean"]
        )
        <= 0.002,
        "dense_action_disagreement_does_not_exceed_selected": dense_ler[
            "action_disagreement_mean"
        ]
        <= selected_ler["action_disagreement_mean"],
        "low_action_disagreement_is_not_better_than_selected": low_ler[
            "action_disagreement_mean"
        ]
        >= selected_ler["action_disagreement_mean"],
        "healthy_ler_replay_has_no_fallback_and_accepts_every_map": all(
            row["fallback_count"] == 0 and row["map_accepted_count"] == samples_per_run
            for row in ler_rows
        ),
        "healthy_ler_replay_updates_both_frame_axes": timing["both_axes_frame_updated"]
        == [0, 1],
        "source_to_action_latency_is_exactly_six_cycles": timing["latency_exact"],
        "integer_fast_path_has_initiation_interval_one": timing[
            "initiation_interval_exact"
        ],
        "replay_boundary_saturation_and_fallback_are_explicit": all(negatives.values()),
        "online_step_contains_no_float_division_or_math": integer_audit["integer_only"],
        "online_code_input_contains_no_truth_or_hidden_fields": not any(
            token in field
            for field in FastPathCodeInput.__dataclass_fields__
            for token in ("truth", "hidden", "drift", "recovery_depth")
        ),
        "rounding_saturation_and_comparison_rules_are_explicit": set(arithmetic_rules)
        == {
            "adc_replay",
            "lut_compile",
            "lut_interpolation",
            "logical_action",
            "ood_replay",
            "parameter_age",
            "event_health_counters",
            "pauli_phase_frames",
            "crc_sha_version",
        },
        "source_rows_are_bit_deterministic": deterministic,
        "resource_fields_remain_representation_only_not_hardware": all(
            hardware_fields[field] is None
            for field in (
                "target_lut_count",
                "target_ff_count",
                "target_bram_count",
                "target_dsp_count",
                "fmax_mhz",
            )
        )
        and not hardware_fields["rtl_measured"]
        and not hardware_fields["board_measured"],
    }
    ler_csv = DEFAULT_LER_CSV if ler_csv_path is None else Path(ler_csv_path)
    code_csv = DEFAULT_CODE_CSV if code_csv_path is None else Path(code_csv_path)
    _write_csv(ler_rows, ler_csv)
    _write_csv(code_rows, code_csv)
    representation = {
        name: FastPathFixedPointContract(
            adc_bits=config.adc_bits,
            address_bits=config.address_bits,
            interpolation_fraction_bits=config.fraction_bits,
            llr_integer_bits=config.llr_integer_bits,
            llr_fractional_bits=config.llr_fractional_bits,
        ).representation_proxy()
        for name, config in PROFILE_CONFIGS.items()
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "task_id": "T4.2.4",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "implementation_sha256": _implementation_sha256(),
        "selected_contract": asdict(contract),
        "arithmetic_rules": arithmetic_rules,
        "profile_configs": {name: config.to_dict() for name, config in PROFILE_CONFIGS.items()},
        "representation_proxy": representation,
        "exhaustive_code_summary": code_summary,
        "ler_summary": ler_summary,
        "timing_diagnostics": timing,
        "negative_audit": negatives,
        "online_integer_audit": integer_audit,
        "source_data": {
            "ler": {
                "path": str(ler_csv.relative_to(ROOT)).replace("\\", "/"),
                "rows": len(ler_rows),
                "sha256": _sha256_file(ler_csv),
                "canonical_rows_sha256": _row_hash(ler_rows),
            },
            "exhaustive_codes": {
                "path": str(code_csv.relative_to(ROOT)).replace("\\", "/"),
                "rows": len(code_rows),
                "sha256": _sha256_file(code_csv),
                "canonical_rows_sha256": _row_hash(code_rows),
            },
        },
        "gate_summary": {
            "passed": sum(gates.values()),
            "failed": len(gates) - sum(gates.values()),
            "gates": gates,
        },
        "hardware_fields": hardware_fields,
        "claim_boundary": {
            "allowed": "paired model-matched bit-accurate software MAP-health-event-frame quantization and LER sensitivity",
            "forbidden": "device-calibrated ADC/OOD/LER, correlated 2D optimum, synthesis/post-route resource or timing, FPGA, or board measurement",
        },
    }
    output_json = DEFAULT_JSON if json_path is None else Path(json_path)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-run", type=int, default=2_048)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--ler-csv", type=Path, default=DEFAULT_LER_CSV)
    parser.add_argument("--code-csv", type=Path, default=DEFAULT_CODE_CSV)
    args = parser.parse_args(argv)
    payload = run_validation(
        samples_per_run=args.samples_per_run,
        json_path=args.json,
        ler_csv_path=args.ler_csv,
        code_csv_path=args.code_csv,
    )
    print(json.dumps(payload["gate_summary"], indent=2))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
