"""T6.19.1 matched project-native pre-board and host-path profiles.

The hardware table is deliberately eligibility-first: only already existing,
source-bound synthesizable RTL may receive numbers.  The host table executes
the real Window/EWMA/Kalman estimator, MAP-LUT compiler and transactional
software bank; those timings are never relabelled as FPGA or board latency.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Mapping, Sequence

import numpy as np

from cnn_fpga.benchmark.unified_comparator_runner import derive_method_costs
from cnn_fpga.benchmark.atomic_parameter_bank_validation import (
    _implementation_sha256 as _atomic_implementation_sha256,
)
from cnn_fpga.decoder.parametric_map_lut import compile_parametric_map_lut
from cnn_fpga.decoder.periodic_adaptive_map import (
    ConstantVelocityPeriodicKalman,
    LatestWindowPeriodicPredictor,
    PeriodicGaussianEstimate,
    PeriodicMomentConfig,
    PeriodicMomentEWMA,
    scaled_periodic_kalman_config,
)
from cnn_fpga.runtime.atomic_parameter_bank import (
    AtomicParameterBankConfig,
    AtomicParameterImageBank,
    build_parameter_image_manifest,
    verify_commit_ack_readback,
)
from cnn_fpga.runtime.param_bank import DecoderRuntimeParams
from cnn_fpga.runtime.parametric_map_lut import ParametricMAPLUTConfig
from physics.constants import LATTICE_CONST


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/literature/t6_19_1_preboard_profiles.json"
DEFAULT_JSON = ROOT / "docs/t6_19_1_project_preboard_profiles.json"
DEFAULT_CSV = ROOT / "docs/t6_19_1_project_preboard_profiles_source_data.csv"
DEFAULT_MARKDOWN = ROOT / "docs/project_preboard_profiles.md"
PREREGISTRATION = ROOT / "configs/literature/t6_16_3_secondary_preregistration.json"
SYNTHESIS_REPORT = ROOT / "docs/t5_5_2_target_device_synthesis.json"
EQUIVALENCE_REPORT = ROOT / "docs/t_risk_20260716_01_rtl_equivalence.json"
LEARNED_ELIGIBILITY = ROOT / "docs/t6_17_3_learned_model_eligibility_replay.json"
PHASE6B_TERMINAL = ROOT / "docs/t6_15_5_route_a_v5_final_evidence_gate.json"
ATOMIC_BANK_VALIDATION = ROOT / "docs/t4_3_2_atomic_parameter_bank_validation.json"
SCHEMA_VERSION = "t6.19.1-project-preboard-profiles-v1"


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    return {"path": _relative(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _experiment(preregistration: Mapping[str, Any]) -> dict[str, Any]:
    rows = [
        row
        for row in preregistration.get("experiments", [])
        if row.get("experiment_id") == "E6191_PROJECT_PREBOARD_PROFILES"
    ]
    if len(rows) != 1:
        raise ValueError("E6191 preregistration must exist exactly once")
    return rows[0]


def _file_bindings_current(rows: Sequence[Mapping[str, Any]]) -> bool:
    for row in rows:
        path = ROOT / str(row.get("path", ""))
        if not path.is_file() or row.get("sha256") != _sha256(path):
            return False
        if "bytes" in row and row.get("bytes") != path.stat().st_size:
            return False
    return True


def _percentiles_us(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("timing samples must be a non-empty finite vector")
    return {
        "p50_us": float(np.percentile(array, 50)),
        "p95_us": float(np.percentile(array, 95)),
        "p99_us": float(np.percentile(array, 99)),
        "worst_us": float(np.max(array)),
        "minimum_us": float(np.min(array)),
        "mean_us": float(np.mean(array)),
    }


def _wrap(values: np.ndarray) -> np.ndarray:
    return np.mod(values + 0.5 * LATTICE_CONST, LATTICE_CONST) - 0.5 * LATTICE_CONST


def _observed_inputs(config: Mapping[str, Any]) -> tuple[np.ndarray, list[np.ndarray], str]:
    repeats = int(config["host_runtime_repeats"])
    warmup = int(config["warmup_repeats"])
    count = int(config["residual_samples_per_update"])
    rng = np.random.default_rng(int(config["seed"]))
    calibration = _wrap(
        rng.normal(
            np.asarray([0.015, -0.012]) * LATTICE_CONST,
            np.asarray([0.155, 0.145]) * LATTICE_CONST,
            size=(count, 2),
        )
    )
    windows: list[np.ndarray] = []
    for index in range(warmup + repeats):
        phase = 2.0 * np.pi * index / 127.0
        mean = np.asarray([0.030 * np.sin(phase), -0.024 * np.cos(phase)]) * LATTICE_CONST
        sigma = np.asarray(
            [0.150 + 0.010 * np.sin(phase / 2.0), 0.145 + 0.012 * np.cos(phase / 3.0)]
        ) * LATTICE_CONST
        windows.append(_wrap(rng.normal(mean, sigma, size=(count, 2))))
    digest = hashlib.sha256()
    digest.update(np.asarray(calibration, dtype="<f8").tobytes())
    for window in windows:
        digest.update(np.asarray(window, dtype="<f8").tobytes())
    return calibration, windows, digest.hexdigest()


def _params_from_estimate(
    estimate: PeriodicGaussianEstimate, *, estimator_id: str
) -> DecoderRuntimeParams:
    """Use the same effective-Gaussian mapping as the Route-A policy validator."""

    covariance = estimate.covariance_array()
    measurement_sigma = 0.04 * LATTICE_CONST
    measurement = np.eye(2, dtype=np.float64) * measurement_sigma**2
    gain = covariance @ np.linalg.inv(covariance + measurement)
    bias = (np.eye(2, dtype=np.float64) - gain) @ estimate.mean_array()
    return DecoderRuntimeParams(
        K=gain,
        b=bias,
        metadata={
            "measurement_cov": measurement.tolist(),
            "alpha_bias": 1.0,
            "estimator_id": estimator_id,
            "estimator_source": estimate.source,
            "estimator_window_id": estimate.window_id,
            "observed_only": True,
        },
    )


def _predictors(calibration: np.ndarray) -> dict[str, Any]:
    moment = PeriodicMomentConfig(minimum_samples=64)
    return {
        "Window": LatestWindowPeriodicPredictor(calibration, moment),
        "EWMA": PeriodicMomentEWMA(calibration, alpha=0.20, config=moment),
        "Kalman": ConstantVelocityPeriodicKalman(
            calibration,
            moment_config=moment,
            kalman_config=scaled_periodic_kalman_config(
                process_scale=1.0, measurement_scale=1.0
            ),
        ),
    }


def _transaction_once(
    initial_image: Any,
    candidate_image: Any,
    *,
    chunk_bytes: int,
    transaction_id: str,
) -> tuple[float, float, int, str]:
    bank = AtomicParameterImageBank(
        initial_image,
        AtomicParameterBankConfig(
            fast_cycle_ns=5_000,
            promotion_good_windows=2,
            min_residency_cycles=4_000,
            max_payload_age_cycles=8_192,
            safe_boundary_period_cycles=1,
        ),
    )
    bank.observe_selection(window_id=1, selection_key="profile", eligible=True)
    bank.observe_selection(window_id=2, selection_key="profile", eligible=True)
    start = perf_counter_ns()
    manifest, payload = build_parameter_image_manifest(
        candidate_image,
        transaction_id=transaction_id,
        selection_key="profile",
        expected_active_version=0,
        source_window_id=2,
        created_epoch=4_000,
        apply_epoch=4_001,
        fast_cycle_ns=5_000,
    )
    bank.begin_stage(manifest, current_epoch=4_000)
    for offset in range(0, len(payload), chunk_bytes):
        bank.write_chunk(
            transaction_id,
            offset=offset,
            chunk=payload[offset : offset + chunk_bytes],
        )
    staged = bank.finalize_stage(transaction_id, current_epoch=4_000)
    transfer_us = (perf_counter_ns() - start) / 1000.0
    start = perf_counter_ns()
    ack = bank.commit_if_ready(4_001, safe_boundary=True)
    if ack is None or not ack.accepted:
        raise RuntimeError("software atomic-bank commit did not complete")
    readback = bank.readback(epoch=4_001)
    if not verify_commit_ack_readback(ack, readback):
        raise RuntimeError("software atomic-bank readback mismatch")
    commit_us = (perf_counter_ns() - start) / 1000.0
    if staged.image_sha256 != candidate_image.image_sha256:
        raise RuntimeError("staged image digest mismatch")
    return transfer_us, commit_us, len(payload), readback.image_sha256


def _host_profiles(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    repeats = int(config["host_runtime_repeats"])
    warmup = int(config["warmup_repeats"])
    chunk_bytes = int(config["transaction_chunk_bytes"])
    calibration, windows, input_sha256 = _observed_inputs(config)
    costs = derive_method_costs()
    cost_keys = {"Window": "window_map", "EWMA": "ewma_adaptive_map", "Kalman": "kalman_adaptive_map"}
    predictors = _predictors(calibration)
    lut_config = ParametricMAPLUTConfig()
    initial_estimate = predictors["Window"].prediction()
    initial_image = compile_parametric_map_lut(
        _params_from_estimate(initial_estimate, estimator_id="initial"),
        active_bank_version=0,
        config=lut_config,
    )
    raw_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for method_id in ("Window", "EWMA", "Kalman"):
        predictor = predictors[method_id]
        estimates: list[PeriodicGaussianEstimate] = []
        update_times: list[float] = []
        for index, residuals in enumerate(windows):
            start = perf_counter_ns()
            estimate = predictor.update(residuals, window_id=index + 1)
            elapsed = (perf_counter_ns() - start) / 1000.0
            if index >= warmup:
                update_times.append(elapsed)
                estimates.append(estimate)
        params = [
            _params_from_estimate(estimate, estimator_id=method_id.lower())
            for estimate in estimates
        ]
        for _ in range(warmup):
            compile_parametric_map_lut(params[0], active_bank_version=1, config=lut_config)
        images: list[Any] = []
        compile_times: list[float] = []
        for item in params:
            start = perf_counter_ns()
            image = compile_parametric_map_lut(item, active_bank_version=1, config=lut_config)
            compile_times.append((perf_counter_ns() - start) / 1000.0)
            images.append(image)
        for index in range(warmup):
            _transaction_once(
                initial_image,
                images[index % len(images)],
                chunk_bytes=chunk_bytes,
                transaction_id=f"warmup-{method_id.lower()}-{index}",
            )
        transfer_times: list[float] = []
        commit_times: list[float] = []
        payload_lengths: list[int] = []
        readback_hashes: list[str] = []
        for index, image in enumerate(images):
            transfer_us, commit_us, payload_length, readback_hash = _transaction_once(
                initial_image,
                image,
                chunk_bytes=chunk_bytes,
                transaction_id=f"profile-{method_id.lower()}-{index}",
            )
            transfer_times.append(transfer_us)
            commit_times.append(commit_us)
            payload_lengths.append(payload_length)
            readback_hashes.append(readback_hash)
            raw_rows.append(
                {
                    "row_type": "host_repeat",
                    "method_id": method_id,
                    "repeat": index,
                    "seed": config["seed"],
                    "update_us": update_times[index],
                    "compiler_us": compile_times[index],
                    "software_transfer_us": transfer_us,
                    "software_commit_us": commit_us,
                    "payload_bytes": payload_length,
                    "image_sha256": image.image_sha256,
                    "readback_sha256": readback_hash,
                    "input_trace_sha256": input_sha256,
                }
            )
        if len(update_times) != repeats or len(images) != repeats:
            raise RuntimeError("host repeat count drifted from preregistration")
        if any(left != right for left, right in zip([image.image_sha256 for image in images], readback_hashes, strict=True)):
            raise RuntimeError("transactional readback digest mismatch")
        cost = costs[cost_keys[method_id]]
        summaries.append(
            {
                "method_id": method_id,
                "eligibility_state": "EXECUTED_REAL_PROJECT_IMPLEMENTATION",
                "repeats": repeats,
                "warmup_repeats": warmup,
                "residual_samples_per_update": int(config["residual_samples_per_update"]),
                "update": _percentiles_us(update_times),
                "compiler": _percentiles_us(compile_times),
                "software_transactional_transfer": _percentiles_us(transfer_times),
                "software_commit_readback": _percentiles_us(commit_times),
                "update_macs": cost.update_macs,
                "update_macs_scope": "method-private real MAC ledger after the shared periodic-characteristic frontend; not a count of NumPy exp/reduction operations",
                "private_model_state_bytes": cost.private_model_state_bytes,
                "transient_workspace_bytes": cost.transient_workspace_bytes,
                "cost_derivation": cost.derivation,
                "payload_bytes": {
                    "minimum": min(payload_lengths),
                    "maximum": max(payload_lengths),
                },
                "host_precision": "NumPy float64 estimator/compiler; software-bank canonical integer image",
                "physical_transfer_latency_us": None,
                "physical_commit_latency_us": None,
                "board_measured_latency_ns": None,
                "power_w": None,
                "jitter_ns": None,
                "deadline_miss_rate": None,
            }
        )
    summaries.append(
        {
            "method_id": "V5_if_exists",
            "eligibility_state": "N_A_NO_V5_IMPLEMENTATION_EARLY_STOP",
            "repeats": 0,
            "warmup_repeats": 0,
            "residual_samples_per_update": None,
            "update": None,
            "compiler": None,
            "software_transactional_transfer": None,
            "software_commit_readback": None,
            "update_macs": None,
            "update_macs_scope": None,
            "private_model_state_bytes": None,
            "transient_workspace_bytes": None,
            "cost_derivation": None,
            "payload_bytes": None,
            "host_precision": None,
            "physical_transfer_latency_us": None,
            "physical_commit_latency_us": None,
            "board_measured_latency_ns": None,
            "power_w": None,
            "jitter_ns": None,
            "deadline_miss_rate": None,
        }
    )
    clock_info = __import__("time").get_clock_info("perf_counter")
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "perf_counter": {
            "implementation": clock_info.implementation,
            "monotonic": clock_info.monotonic,
            "adjustable": clock_info.adjustable,
            "resolution_seconds": clock_info.resolution,
        },
        "input_trace_sha256": input_sha256,
        "timing_boundary": "single-process current-host Python wall-clock diagnostic; not FPGA latency",
    }
    return summaries, raw_rows, environment


def _na_hardware_row(method_id: str, reason: str) -> dict[str, Any]:
    return {
        "method_id": method_id,
        "eligibility_state": reason,
        "ranking_eligible_project_preboard": False,
        "rtl_path": None,
        "integer_reference_path": None,
        "cxxrtl_action_mismatch_count": None,
        "decision_mismatch_rate": None,
        "core_cycles": None,
        "initiation_interval_cycles": None,
        "clock_mhz": None,
        "source_to_action_ns": None,
        "initiation_interval_ns": None,
        "place_route": None,
        "power_w": None,
        "jitter_ns": None,
        "deadline_miss_rate": None,
        "board_measured_latency_ns": None,
        "physical_transfer_latency_us": None,
        "physical_commit_latency_us": None,
        "evidence_boundary": "N_A_NO_ELIGIBLE_SYNTHESIZABLE_RTL",
    }


def _hardware_profiles(
    config: Mapping[str, Any],
    synthesis: Mapping[str, Any],
    equivalence: Mapping[str, Any],
    learned: Mapping[str, Any],
    phase6b: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seeds = [int(value) for value in config["target"]["place_route_seeds"]]
    place_route = synthesis["place_route"]
    seed_rows: list[dict[str, Any]] = []
    compact: list[dict[str, Any]] = []
    for row in place_route:
        seed = int(row["seed"])
        utilization = row["utilization"]
        critical = row["critical_path"]
        item = {
            "seed": seed,
            "target_mhz": float(row["target_mhz"]),
            "achieved_fmax_mhz": float(row["achieved_fmax_mhz"]),
            "timing_pass": bool(row["timing_pass"]),
            "route_status": row["route_status"],
            "lut4_count": int(utilization["LUT4"]["used"]),
            "ff_count": int(utilization["DFF"]["used"]),
            "bram_count": int(utilization["BSRAM"]["used"]),
            "dsp_count": int(utilization["MULT18X18"]["used"] + utilization["MULT9X9"]["used"]),
            "dsp_breakdown": {
                "MULT18X18": int(utilization["MULT18X18"]["used"]),
                "MULT9X9": int(utilization["MULT9X9"]["used"]),
            },
            "critical_path": {
                "period_ns": float(critical["period_ns"]),
                "logic_ns": float(critical["logic_ns"]),
                "routing_ns": float(critical["routing_ns"]),
                "start_cell": critical["start_cell"],
                "end_cell": critical["end_cell"],
                "segment_count": int(critical["segment_count"]),
            },
            "report_artifact": row["report_artifact"],
            "log_artifact": row["log_artifact"],
        }
        compact.append(item)
        seed_rows.append(
            {
                "row_type": "hardware_seed",
                "method_id": "static_map_lut_if_rtl",
                "repeat": "",
                "seed": seed,
                "update_us": "",
                "compiler_us": "",
                "software_transfer_us": "",
                "software_commit_us": "",
                "payload_bytes": "",
                "image_sha256": "",
                "readback_sha256": "",
                "input_trace_sha256": "",
                "achieved_fmax_mhz": item["achieved_fmax_mhz"],
                "lut4_count": item["lut4_count"],
                "ff_count": item["ff_count"],
                "bram_count": item["bram_count"],
                "dsp_count": item["dsp_count"],
                "critical_path_ns": item["critical_path"]["period_ns"],
            }
        )
    if sorted(row["seed"] for row in compact) != sorted(seeds):
        raise ValueError("place-and-route seed set does not match T6.19.1 config")
    latency = synthesis["latency_estimate"]
    scenarios = equivalence["scenarios"]
    hardware = [
        _na_hardware_row("CI_if_rtl", "N_A_NO_INDEPENDENT_CI_RTL"),
        {
            "method_id": "static_map_lut_if_rtl",
            "eligibility_state": "ELIGIBLE_EXISTING_SOURCE_BOUND_RTL",
            "ranking_eligible_project_preboard": True,
            "rtl_path": "cnn_fpga/rtl/gkp_fast_path_core.sv",
            "integer_reference_path": "cnn_fpga/runtime/parametric_map_lut.py",
            "cxxrtl_action_mismatch_count": int(equivalence["scenarios"][0]["mismatch_count"] + equivalence["scenarios"][1]["mismatch_count"]),
            "decision_mismatch_rate": 0.0,
            "equivalence_scenarios": scenarios,
            "equivalence_map_valid_rows": int(sum(row["map_valid_rows"] for row in scenarios)),
            "core_cycles": int(latency["core_cycles"]),
            "initiation_interval_cycles": int(latency["initiation_interval_cycles"]),
            "clock_mhz": float(config["target"]["clock_mhz"]),
            "source_to_action_ns": float(latency["at_target_27mhz_ns"]),
            "initiation_interval_ns": float(latency["initiation_interval_at_target_ns"]),
            "place_route": compact,
            "fmax_summary_mhz": synthesis["summary"]["fmax_mhz"],
            "power_w": None,
            "jitter_ns": None,
            "deadline_miss_rate": None,
            "board_measured_latency_ns": None,
            "physical_transfer_latency_us": None,
            "physical_commit_latency_us": None,
            "evidence_boundary": "CXXRTL_EQUIVALENCE_AND_TARGET_DEVICE_POST_ROUTE_ESTIMATE_NOT_BOARD_MEASURED",
            "resource_scope": "complete gkp_fast_path_synth_top including MAP-LUT, event/fault/state logic and small-pin activity/observability harness; excludes transport and is not map-ROM-only area",
        },
        _na_hardware_row("v5_fast_path_if_rtl", "N_A_NO_V5_RTL_EARLY_STOP_AT_T6_10_1"),
        _na_hardware_row("eligible_direct_nn_if_rtl", "N_A_NO_SAME_TASK_ELIGIBLE_DIRECT_NN_RTL"),
    ]
    if phase6b.get("execution_path") != "EARLY_STOP_AT_T6.10.1_HEADROOM_GATE":
        raise ValueError("V5 early-stop evidence changed")
    summary = learned.get("eligibility_summary", {})
    if summary.get("same_task_eligible") != 0:
        raise ValueError("learned eligibility no longer supports the frozen N/A row")
    return hardware, seed_rows


def _all_null_boundaries(report: Mapping[str, Any]) -> bool:
    fields = report["config"]["must_remain_null_until_t6_9_2"]
    for table in (report["hardware_profiles"], report["host_profiles"]):
        for row in table:
            if any(row.get(field) is not None for field in fields):
                return False
    return True


def recompute_gates(report: Mapping[str, Any]) -> dict[str, bool]:
    config = report["config"]
    hardware = {row["method_id"]: row for row in report["hardware_profiles"]}
    host = {row["method_id"]: row for row in report["host_profiles"]}
    eligible = hardware["static_map_lut_if_rtl"]
    synthesis = _load_json(SYNTHESIS_REPORT)
    equivalence = _load_json(EQUIVALENCE_REPORT)
    learned = _load_json(LEARNED_ELIGIBILITY)
    phase6b = _load_json(PHASE6B_TERMINAL)
    atomic_validation = _load_json(ATOMIC_BANK_VALIDATION)
    prereg = _experiment(_load_json(PREREGISTRATION))
    source_data = report["source_data"]
    source_path = ROOT / source_data["path"]
    raw_rows: list[dict[str, str]] = []
    if source_path.is_file():
        with source_path.open(newline="", encoding="utf-8") as stream:
            raw_rows = list(csv.DictReader(stream))
    n_a_ids = {"CI_if_rtl", "v5_fast_path_if_rtl", "eligible_direct_nn_if_rtl"}
    na_numeric = (
        "decision_mismatch_rate",
        "core_cycles",
        "initiation_interval_cycles",
        "clock_mhz",
        "source_to_action_ns",
        "initiation_interval_ns",
        "place_route",
    )
    seed_rows = eligible.get("place_route") or []
    expected_costs = derive_method_costs()
    cost_keys = {"Window": "window_map", "EWMA": "ewma_adaptive_map", "Kalman": "kalman_adaptive_map"}
    expected_latency = 6.0 / 27.0 * 1000.0
    expected_ii = 1.0 / 27.0 * 1000.0
    bindings = report["bindings"]
    binding_rows = [row for row in bindings.values() if isinstance(row, Mapping) and "path" in row]

    synthesis_by_seed = {int(row["seed"]): row for row in synthesis["place_route"]}
    compact_matches_synthesis = True
    for row in seed_rows:
        parent = synthesis_by_seed.get(int(row["seed"]))
        if parent is None:
            compact_matches_synthesis = False
            break
        utilization = parent["utilization"]
        expected_dsp = int(utilization["MULT18X18"]["used"] + utilization["MULT9X9"]["used"])
        compact_matches_synthesis = compact_matches_synthesis and (
            row["achieved_fmax_mhz"] == parent["achieved_fmax_mhz"]
            and row["lut4_count"] == utilization["LUT4"]["used"]
            and row["ff_count"] == utilization["DFF"]["used"]
            and row["bram_count"] == utilization["BSRAM"]["used"]
            and row["dsp_count"] == expected_dsp
            and row["critical_path"]["period_ns"] == parent["critical_path"]["period_ns"]
            and row["report_artifact"] == parent["report_artifact"]
            and row["log_artifact"] == parent["log_artifact"]
        )

    raw_host = [row for row in raw_rows if row.get("row_type") == "host_repeat"]
    stage_columns = {
        "update": "update_us",
        "compiler": "compiler_us",
        "software_transactional_transfer": "software_transfer_us",
        "software_commit_readback": "software_commit_us",
    }
    host_raw_exact = True
    for method in cost_keys:
        rows = [row for row in raw_host if row.get("method_id") == method]
        host_raw_exact = host_raw_exact and (
            len(rows) == config["host_runtime_repeats"]
            and {int(row["repeat"]) for row in rows} == set(range(config["host_runtime_repeats"]))
            and all(row["image_sha256"] == row["readback_sha256"] for row in rows)
            and len({row["input_trace_sha256"] for row in rows}) == 1
            and rows[0]["input_trace_sha256"] == report["host_environment"]["input_trace_sha256"]
        )
        for stage, column in stage_columns.items():
            values = [float(row[column]) for row in rows]
            host_raw_exact = host_raw_exact and _percentiles_us(values) == host[method][stage]
            timing = host[method][stage]
            host_raw_exact = host_raw_exact and (
                0.0 < timing["minimum_us"]
                <= timing["p50_us"]
                <= timing["p95_us"]
                <= timing["p99_us"]
                <= timing["worst_us"]
            )
    return {
        "G01_frozen_preregistration_and_config_are_exact": (
            prereg["task_id"] == "T6.19.1"
            and prereg["config"]["methods"] == config["hardware_methods"]
            and prereg["config"]["host_profiles"] == config["host_methods"]
            and prereg["sample_size"]["host_runtime_repeats"] == config["host_runtime_repeats"]
            and bindings["preregistration"]["sha256"] == _sha256(PREREGISTRATION)
        ),
        "G02_only_existing_static_map_rtl_enters_hardware_table": (
            set(hardware) == set(config["hardware_methods"])
            and eligible["ranking_eligible_project_preboard"]
            and all(not hardware[item]["ranking_eligible_project_preboard"] for item in n_a_ids)
            and all(all(hardware[item].get(field) is None for field in na_numeric) for item in n_a_ids)
        ),
        "G03_integer_cxxrtl_equivalence_is_exact_and_source_bound": (
            equivalence["status"] == "PASS"
            and eligible["cxxrtl_action_mismatch_count"] == 0
            and eligible["equivalence_map_valid_rows"] == 4316
            and all(row["exact"] and row["mismatch_count"] == 0 for row in equivalence["scenarios"])
            and _file_bindings_current(equivalence["source_bindings"])
        ),
        "G04_three_seed_target_device_place_route_is_complete": (
            [row["seed"] for row in seed_rows] == config["target"]["place_route_seeds"]
            and all(row["timing_pass"] and row["route_status"] == "PASS" for row in seed_rows)
            and all(min(row["lut4_count"], row["ff_count"], row["bram_count"], row["dsp_count"]) > 0 for row in seed_rows)
            and all(row["critical_path"]["period_ns"] > 0.0 for row in seed_rows)
            and synthesis["status"] == "PASS"
            and compact_matches_synthesis
            and _file_bindings_current(synthesis["source_bindings"])
            and _file_bindings_current(synthesis["durable_artifacts"])
        ),
        "G05_six_cycle_ii_one_clock_arithmetic_is_exact": (
            eligible["core_cycles"] == 6
            and eligible["initiation_interval_cycles"] == 1
            and eligible["clock_mhz"] == 27.0
            and abs(eligible["source_to_action_ns"] - expected_latency) < 1.0e-12
            and abs(eligible["initiation_interval_ns"] - expected_ii) < 1.0e-12
        ),
        "G06_window_ewma_kalman_execute_1000_real_updates_per_stage": (
            set(host) == set(config["host_methods"])
            and all(host[item]["repeats"] == 1000 for item in cost_keys)
            and all(host[item]["eligibility_state"] == "EXECUTED_REAL_PROJECT_IMPLEMENTATION" for item in cost_keys)
            and all(
                all(
                    host[item][stage][metric] > 0.0
                    for stage in ("update", "compiler", "software_transactional_transfer", "software_commit_readback")
                    for metric in ("p50_us", "p95_us", "p99_us", "worst_us")
                )
                for item in cost_keys
            )
            and host_raw_exact
            and atomic_validation["status"] == "PASS"
            and all(row["passed"] for row in atomic_validation["gates"])
            and atomic_validation["implementation_sha256"]
            == _atomic_implementation_sha256()
        ),
        "G07_host_operation_state_and_workspace_ledgers_match_implemented_methods": all(
            host[method]["update_macs"] == expected_costs[key].update_macs
            and host[method]["private_model_state_bytes"] == expected_costs[key].private_model_state_bytes
            and host[method]["transient_workspace_bytes"] == expected_costs[key].transient_workspace_bytes
            for method, key in cost_keys.items()
        ),
        "G08_v5_and_direct_nn_remain_na_without_rescue_implementation": (
            phase6b["execution_path"] == "EARLY_STOP_AT_T6.10.1_HEADROOM_GATE"
            and phase6b["v5_downstream_outputs_found"] == []
            and host["V5_if_exists"]["eligibility_state"] == "N_A_NO_V5_IMPLEMENTATION_EARLY_STOP"
            and learned["eligibility_summary"]["same_task_eligible"] == 0
            and learned["eligibility_summary"]["eligible_replayed"] == 0
        ),
        "G09_host_and_hardware_boundaries_are_separate_and_board_fields_null": (
            _all_null_boundaries(report)
            and report["cross_table_ranking"] is None
            and report["board_measurement_state"] == "BLOCKED_UNTIL_T6.9.2"
        ),
        "G10_raw_rows_hash_and_count_are_current": (
            source_path.is_file()
            and source_data["sha256"] == _sha256(source_path)
            and source_data["rows"] == 3 + 3 * config["host_runtime_repeats"]
            and len(raw_rows) == source_data["rows"]
            and sum(row.get("row_type") == "hardware_seed" for row in raw_rows) == 3
            and len(raw_host) == 3 * config["host_runtime_repeats"]
        ),
        "G11_all_live_source_bindings_and_phase6b_readonly_hash_are_current": (
            _file_bindings_current(binding_rows)
            and bindings["phase6b_terminal"]["sha256"] == _sha256(PHASE6B_TERMINAL)
            and report["phase6b_effect"] == "READ_ONLY_UNCHANGED"
        ),
        "G12_execution_respects_frozen_wallclock_budget": (
            0.0 < report["runtime_budget"]["wall_clock_seconds"]
            <= report["runtime_budget"]["limit_seconds"]
            and report["runtime_budget"]["within_wall_clock_budget"] is True
        ),
    }


def _mutation_audit(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    cases: list[tuple[str, str, Any]] = []
    target = "G02_only_existing_static_map_rtl_enters_hardware_table"
    cases.append(("promote_ci_without_rtl", target, lambda row: row["hardware_profiles"][0].update({"ranking_eligible_project_preboard": True})))
    cases.append(("fill_ci_latency", target, lambda row: row["hardware_profiles"][0].update({"core_cycles": 1})))
    cases.append(("forge_cxxrtl_mismatch", "G03_integer_cxxrtl_equivalence_is_exact_and_source_bound", lambda row: row["hardware_profiles"][1].update({"cxxrtl_action_mismatch_count": 1})))
    cases.append(("drop_place_route_seed", "G04_three_seed_target_device_place_route_is_complete", lambda row: row["hardware_profiles"][1].update({"place_route": row["hardware_profiles"][1]["place_route"][:-1]})))
    cases.append(("break_latency_arithmetic", "G05_six_cycle_ii_one_clock_arithmetic_is_exact", lambda row: row["hardware_profiles"][1].update({"source_to_action_ns": 6.0})))
    cases.append(("reduce_host_repeats", "G06_window_ewma_kalman_execute_1000_real_updates_per_stage", lambda row: row["host_profiles"][0].update({"repeats": 999})))
    cases.append(("forge_kalman_macs", "G07_host_operation_state_and_workspace_ledgers_match_implemented_methods", lambda row: row["host_profiles"][2].update({"update_macs": 1})))
    cases.append(("invent_v5_host_result", "G08_v5_and_direct_nn_remain_na_without_rescue_implementation", lambda row: row["host_profiles"][3].update({"eligibility_state": "EXECUTED_REAL_PROJECT_IMPLEMENTATION"})))
    cases.append(("invent_board_power", "G09_host_and_hardware_boundaries_are_separate_and_board_fields_null", lambda row: row["hardware_profiles"][1].update({"power_w": 0.1})))
    cases.append(("create_cross_table_ranking", "G09_host_and_hardware_boundaries_are_separate_and_board_fields_null", lambda row: row.update({"cross_table_ranking": ["static", "Kalman"]})))
    cases.append(("forge_source_row_count", "G10_raw_rows_hash_and_count_are_current", lambda row: row["source_data"].update({"rows": 1})))
    cases.append(("claim_phase6b_upgrade", "G11_all_live_source_bindings_and_phase6b_readonly_hash_are_current", lambda row: row.update({"phase6b_effect": "UPGRADED"})))
    cases.append(("exceed_wallclock_budget", "G12_execution_respects_frozen_wallclock_budget", lambda row: row["runtime_budget"].update({"wall_clock_seconds": row["runtime_budget"]["limit_seconds"] + 1.0})))
    output = []
    for name, gate, mutate in cases:
        candidate = copy.deepcopy(report)
        candidate.pop("semantic_mutation_audit", None)
        candidate.pop("gates", None)
        candidate.pop("gate_summary", None)
        mutate(candidate)
        output.append({"case": name, "target_gate": gate, "rejected": not recompute_gates(candidate)[gate]})
    return output


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(report: Mapping[str, Any], path: Path) -> None:
    hardware = {row["method_id"]: row for row in report["hardware_profiles"]}
    static = hardware["static_map_lut_if_rtl"]
    lines = [
        "# T6.19.1 项目原生预板 profile",
        "",
        "> 结论边界：这是 CXXRTL 等价、目标器件 P&R 估计与当前主机软件计时；不是板测延迟、功耗、抖动或 deadline 证据。",
        "",
        "## 硬件资格表",
        "",
        "| 方法 | 资格 | cycles / II | 27 MHz source-to-action | LUT4 / FF / BSRAM / DSP |",
        "|---|---|---:|---:|---:|",
    ]
    for row in report["hardware_profiles"]:
        resources = "N/A"
        if row["place_route"]:
            med = sorted(row["place_route"], key=lambda item: item["seed"])[1]
            resources = f"{med['lut4_count']} / {med['ff_count']} / {med['bram_count']} / {med['dsp_count']} (seed 7)"
        cycles = "N/A" if row["core_cycles"] is None else f"{row['core_cycles']} / {row['initiation_interval_cycles']}"
        latency = "N/A" if row["source_to_action_ns"] is None else f"{row['source_to_action_ns']:.3f} ns"
        lines.append(f"| {row['method_id']} | {row['eligibility_state']} | {cycles} | {latency} | {resources} |")
    lines.extend(
        [
            "",
            f"static MAP-LUT 的 CXXRTL 比对覆盖 {static['equivalence_map_valid_rows']} 个有效 action 行，mismatch={static['cxxrtl_action_mismatch_count']}；三种子均通过 27 MHz。",
            "",
            "## 软件慢路径（当前主机诊断）",
            "",
            "| 方法 | update p50/p99 | compiler p50/p99 | software transfer p50/p99 | software commit p50/p99 | MAC / state / workspace |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["host_profiles"]:
        if row["update"] is None:
            lines.append(f"| {row['method_id']} | N/A | N/A | N/A | N/A | N/A |")
            continue
        lines.append(
            f"| {row['method_id']} | {row['update']['p50_us']:.3f}/{row['update']['p99_us']:.3f} us | "
            f"{row['compiler']['p50_us']:.3f}/{row['compiler']['p99_us']:.3f} us | "
            f"{row['software_transactional_transfer']['p50_us']:.3f}/{row['software_transactional_transfer']['p99_us']:.3f} us | "
            f"{row['software_commit_readback']['p50_us']:.3f}/{row['software_commit_readback']['p99_us']:.3f} us | "
            f"{row['update_macs']} / {row['private_model_state_bytes']} B / {row['transient_workspace_bytes']} B |"
        )
    lines.extend(
        [
            "",
            "## 可用与禁用表述",
            "",
            "- 可用：现有 static MAP-LUT 在 GW2AR-LV18QN88C8/I7 的三种子 P&R 中满足 27 MHz，并有 6-cycle、II=1 的 source-bound RTL/CXXRTL 证据。",
            "- 资源范围：完整 `gkp_fast_path_synth_top`（MAP-LUT、event/fault/state 与小引脚 harness），不是 MAP ROM 单体面积。",
            "- 禁用：CI、V5 或 Direct NN 已在同一 FPGA 上更快；当前没有相应合格 RTL。",
            "- 禁用：把 Python update/compiler/内存事务时间写成 FPGA latency、真实传输或板级 commit。",
            "- 所有 power/jitter/deadline/board-measured 字段保持 null，等待 T6.9.2。",
            "",
            f"Gate：{report['gate_summary']['passed']}/{report['gate_summary']['total']}；mutation：{report['semantic_mutation_audit']['detected']}/{report['semantic_mutation_audit']['count']}。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_profile(
    *,
    config_path: Path = DEFAULT_CONFIG,
    json_path: Path = DEFAULT_JSON,
    csv_path: Path = DEFAULT_CSV,
    markdown_path: Path = DEFAULT_MARKDOWN,
) -> dict[str, Any]:
    started = perf_counter_ns()
    config = _load_json(config_path)
    preregistration = _load_json(PREREGISTRATION)
    prereg = _experiment(preregistration)
    synthesis = _load_json(SYNTHESIS_REPORT)
    equivalence = _load_json(EQUIVALENCE_REPORT)
    learned = _load_json(LEARNED_ELIGIBILITY)
    phase6b = _load_json(PHASE6B_TERMINAL)
    hardware, hardware_rows = _hardware_profiles(config, synthesis, equivalence, learned, phase6b)
    host, host_rows, environment = _host_profiles(config)
    _write_csv([*hardware_rows, *host_rows], csv_path)
    report: dict[str, Any] = {
        "task_id": "T6.19.1",
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PENDING_GATES",
        "verdict": "PENDING_GATES",
        "scope": "matched existing-RTL pre-board estimates plus separately reported current-host software slow-path profiles",
        "preregistration": {
            "experiment_id": prereg["experiment_id"],
            "record_sha256": hashlib.sha256(json.dumps(prereg, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")).hexdigest(),
            "execution_type": prereg["execution_type"],
            "forbidden_actions": prereg["forbidden_actions"],
        },
        "config": config,
        "hardware_profiles": hardware,
        "host_profiles": host,
        "host_environment": environment,
        "board_measurement_state": "BLOCKED_UNTIL_T6.9.2",
        "cross_table_ranking": None,
        "global_fastest_claim": None,
        "phase6b_effect": "READ_ONLY_UNCHANGED",
        "source_data": {
            "path": _relative(csv_path),
            "sha256": _sha256(csv_path),
            "rows": len(hardware_rows) + len(host_rows),
            "hardware_seed_rows": len(hardware_rows),
            "host_repeat_rows": len(host_rows),
        },
        "bindings": {
            "implementation": _binding(Path(__file__)),
            "config": _binding(config_path),
            "preregistration": _binding(PREREGISTRATION),
            "synthesis_report": _binding(SYNTHESIS_REPORT),
            "equivalence_report": _binding(EQUIVALENCE_REPORT),
            "learned_eligibility": _binding(LEARNED_ELIGIBILITY),
            "phase6b_terminal": _binding(PHASE6B_TERMINAL),
            "periodic_estimators": _binding(ROOT / "cnn_fpga/decoder/periodic_adaptive_map.py"),
            "map_lut_compiler": _binding(ROOT / "cnn_fpga/decoder/parametric_map_lut.py"),
            "atomic_parameter_bank": _binding(ROOT / "cnn_fpga/runtime/atomic_parameter_bank.py"),
            "atomic_parameter_bank_validation": _binding(ATOMIC_BANK_VALIDATION),
            "unified_cost_ledger": _binding(ROOT / "cnn_fpga/benchmark/unified_comparator_runner.py"),
            "source_data": _binding(csv_path),
        },
        "claim_boundary": {
            "allowed": [
                "static MAP-LUT has exact source-bound integer/CXXRTL action equivalence and three-seed 27 MHz target-device P&R estimates",
                "Window/EWMA/Kalman host stages use real project implementations and are reported as current-host diagnostics",
            ],
            "forbidden": [
                "CI, V5 or Direct NN hardware ranking without eligible RTL",
                "Python timing as FPGA latency",
                "post-route estimate as vendor signoff or board measurement",
                "non-null power, jitter, deadline or physical transfer/commit before T6.9.2",
            ],
        },
        "runtime_budget": {
            "wall_clock_seconds": (perf_counter_ns() - started) / 1.0e9,
            "limit_seconds": prereg["runtime_budget"]["wall_clock_seconds"],
            "memory_limit_gib": prereg["runtime_budget"]["memory_gib"],
        },
    }
    report["runtime_budget"]["within_wall_clock_budget"] = (
        report["runtime_budget"]["wall_clock_seconds"]
        <= report["runtime_budget"]["limit_seconds"]
    )
    gates = recompute_gates(report)
    report["gates"] = gates
    audit = _mutation_audit(report)
    report["semantic_mutation_audit"] = {
        "count": len(audit),
        "detected": sum(row["rejected"] for row in audit),
        "cases": audit,
    }
    report["gate_summary"] = {
        "passed": sum(gates.values()),
        "total": len(gates),
        "failed": [name for name, passed in gates.items() if not passed],
    }
    passed = all(gates.values()) and all(row["rejected"] for row in audit)
    report["status"] = "PASS" if passed else "FAIL"
    report["verdict"] = (
        "PASS_STATIC_MAP_LUT_PREBOARD_PROFILE_OTHERS_NA_HOST_STAGES_SEPARATE"
        if passed
        else "FAIL_T6_19_1_PROFILE_INTEGRITY"
    )
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_markdown(report, markdown_path)
    return report


def verify_report(path: Path = DEFAULT_JSON) -> dict[str, Any]:
    report = _load_json(path)
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("T6.19.1 schema mismatch")
    gates = recompute_gates(report)
    if gates != report.get("gates") or not all(gates.values()):
        raise ValueError("T6.19.1 gate recomputation failed")
    audit = report.get("semantic_mutation_audit", {})
    if audit.get("count") != 13 or audit.get("detected") != 13 or not all(
        row.get("rejected") for row in audit.get("cases", [])
    ):
        raise ValueError("T6.19.1 mutation audit is incomplete")
    if report.get("status") != "PASS":
        raise ValueError("T6.19.1 report is not PASS")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    if args.verify_only:
        report = verify_report(args.json)
    else:
        report = run_profile(
            config_path=args.config,
            json_path=args.json,
            csv_path=args.csv,
            markdown_path=args.markdown,
        )
    print(json.dumps({"status": report["status"], "verdict": report["verdict"], "gate_summary": report["gate_summary"]}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
