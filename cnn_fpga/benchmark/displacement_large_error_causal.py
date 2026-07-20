"""T5.2.1 independent causal displacement / large-distance fault campaign.

The campaign keeps two estimands separate:

* protocol-native recovery depth and same-quadrature e-run from the registered
  coarse sBs error-space/observation model;
* evaluator-only logical classification under a frozen injection-jitter model,
  reported both relative to the nearest nominal logical operation and relative
  to the identity frame.

The latter distinction prevents the logical-flip endpoint at ``epsilon/l_S=0.5``
from being relabelled as harmless.  No quantity is device calibrated or a
physical-memory LER.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from math import erfc, sqrt
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cnn_fpga.benchmark.algorithm_success_falsification import FALLBACK_BRANCH_ID
from physics.ideal_gkp_decoder import standard_binning_1d
from physics.sbs_displacement_fault import (
    PRIMARY_SOURCE_ANCHORS,
    PRIMARY_SOURCE_PATH,
    DisplacementFaultSweepConfig,
    _make_observation_model,
    _make_recovery_instrument,
    _simulate_amplitude,
    distance_to_closest_logical_operation,
)


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "T5.2.1"
SCHEMA_VERSION = "t5.2.1-causal-displacement-large-error-v1"
PROTOCOL_ID = "CAUSAL-DISPLACEMENT-LARGE-DISTANCE-INJECTION-V1"
DEFAULT_ARTIFACT = Path("docs/t5_2_1_displacement_large_error_causal.json")
DEFAULT_SOURCE_DATA = Path("docs/t5_2_1_displacement_large_error_source_data.csv")

PARENT_ARTIFACTS: dict[str, Path] = {
    "T2.0.5": Path("docs/t2_0_5_displacement_fault_trend.json"),
    "T5.1.2": Path("docs/t5_1_2_mixed_scenario_matrix.json"),
    "T2.3.3": Path("docs/t2_3_3_cross_fidelity_validation.json"),
    "T5.1.6": Path("docs/t5_1_6_experimental_feasibility.json"),
}

IMPLEMENTATION_PATHS = (
    Path("cnn_fpga/benchmark/displacement_large_error_causal.py"),
    Path("physics/sbs_displacement_fault.py"),
    Path("physics/ideal_gkp_decoder.py"),
)

AMPLITUDES = tuple(index / 32.0 for index in range(17))
EVALUATION_SEEDS = tuple(202607162101 + index for index in range(8))
CALIBRATION_SEEDS = (2026071405, 2026071406, 20260716401, 20260716402)


@dataclass(frozen=True)
class LogicalNoiseProfile:
    profile_id: str
    injection_sigma_over_lattice: float


LOGICAL_NOISE_PROFILES = (
    LogicalNoiseProfile("primary_sigma_0p040", 0.040),
    LogicalNoiseProfile("confirmation_sigma_0p025", 0.025),
)


@dataclass(frozen=True)
class CampaignConfig:
    amplitudes_over_lattice: tuple[float, ...] = AMPLITUDES
    evaluation_seeds: tuple[int, ...] = EVALUATION_SEEDS
    shots_per_seed_amplitude: int = 4096
    cycles: int = 20
    max_recovery_depth: int = 6
    one_step_recovery_probability: float = 0.88
    fault_quadrature: str = "Z"
    false_e_given_g: float = 0.005
    e_detection_probability: float = 0.98
    seed_cluster_bootstrap_replicates: int = 20000
    bootstrap_seed: int = 202607162199
    confidence_level: float = 0.95
    logical_operation_spacing_over_lattice: float = 0.5
    logical_noise_profiles: tuple[LogicalNoiseProfile, ...] = LOGICAL_NOISE_PROFILES

    def __post_init__(self) -> None:
        if tuple(self.amplitudes_over_lattice) != AMPLITUDES:
            raise ValueError("formal amplitudes must remain the preregistered 17-point grid")
        if tuple(self.evaluation_seeds) != EVALUATION_SEEDS:
            raise ValueError("formal evaluation seed clusters changed")
        if set(self.evaluation_seeds) & set(CALIBRATION_SEEDS):
            raise ValueError("evaluation seeds overlap calibration/pilot seeds")
        if self.shots_per_seed_amplitude < 1024:
            raise ValueError("shots_per_seed_amplitude must be at least 1024")
        if self.cycles < self.max_recovery_depth:
            raise ValueError("cycles must cover max_recovery_depth")
        if self.seed_cluster_bootstrap_replicates < 10000:
            raise ValueError("seed-cluster bootstrap must use at least 10000 replicates")
        if self.logical_operation_spacing_over_lattice != 0.5:
            raise ValueError("logical operation spacing is frozen at l_S/2")
        if tuple(self.logical_noise_profiles) != LOGICAL_NOISE_PROFILES:
            raise ValueError("logical injection-noise confirmation profiles changed")


def _repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(_repo_path(path).read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def implementation_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _machine_pass(task_id: str, payload: Mapping[str, Any]) -> bool:
    if task_id == "T2.0.5":
        return payload.get("gate", {}).get("passed") is True
    if task_id == "T2.3.3":
        return payload.get("passed") is True
    gates = payload.get("gates")
    return bool(
        payload.get("status") == "PASS"
        and isinstance(gates, Mapping)
        and gates
        and all(value is True for value in gates.values())
    )


def load_parent_artifacts(
    paths: Mapping[str, str | Path] = PARENT_ARTIFACTS,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for task_id, path in paths.items():
        result[task_id] = json.loads(_repo_path(path).read_text(encoding="utf-8"))
    return result


def inspect_parent_integrity(
    parents: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    missing = set(PARENT_ARTIFACTS) - set(parents)
    if missing:
        raise ValueError(f"missing parent artifacts: {sorted(missing)}")
    return {
        task_id: {
            "path": PARENT_ARTIFACTS[task_id].as_posix(),
            "sha256": _sha256(PARENT_ARTIFACTS[task_id]),
            "machine_pass": _machine_pass(task_id, parents[task_id]),
        }
        for task_id in PARENT_ARTIFACTS
    }


def _seed_stream(seed: int, stream: str) -> int:
    digest = hashlib.sha256(f"{seed}:{stream}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "little")


def _array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _base_recovery_config(config: CampaignConfig, seed: int) -> DisplacementFaultSweepConfig:
    return DisplacementFaultSweepConfig(
        amplitudes_over_lattice=config.amplitudes_over_lattice,
        shots=config.shots_per_seed_amplitude,
        cycles=config.cycles,
        seed=seed,
        bootstrap_seed=seed + 10_000,
        bootstrap_replicates=100,
        max_recovery_depth=config.max_recovery_depth,
        one_step_recovery_probability=config.one_step_recovery_probability,
        fault_quadrature=config.fault_quadrature,
        false_e_given_g=config.false_e_given_g,
        e_detection_probability=config.e_detection_probability,
    )


def _run_recovery_seed_rows(config: CampaignConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in config.evaluation_seeds:
        local = _base_recovery_config(config, seed)
        instrument = _make_recovery_instrument(local)
        observation = _make_observation_model(local)
        # Resetting the stream at every amplitude is deliberate common-random-number
        # pairing; only displacement severity changes between paired runs.
        for amplitude in config.amplitudes_over_lattice:
            raw = _simulate_amplitude(
                amplitude,
                config=local,
                instrument=instrument,
                observation_model=observation,
                rng=np.random.default_rng(_seed_stream(seed, "recovery_crn")),
            )
            rows.append(
                {
                    "seed": seed,
                    "amplitude_over_lattice": amplitude,
                    "logical_distance": float(
                        distance_to_closest_logical_operation(amplitude)
                    ),
                    "mean_initial_recovery_depth": float(np.mean(raw.initial_depth)),
                    "mean_observed_same_quadrature_max_e_run": float(
                        np.mean(raw.observed_max_run)
                    ),
                    "mean_ideal_same_quadrature_max_e_run": float(
                        np.mean(raw.ideal_max_run)
                    ),
                    "mean_restricted_recovery_cycles": float(
                        np.mean(raw.restricted_recovery_cycles)
                    ),
                    "recovered_fraction_by_horizon": float(np.mean(raw.recovered)),
                    "unaffected_e_probability_max": float(
                        np.max(np.mean(raw.unaffected_e, axis=0))
                    ),
                    "trace_sha256": _array_hash(
                        raw.initial_depth,
                        raw.observed_max_run,
                        raw.restricted_recovery_cycles,
                        raw.recovered,
                        raw.affected_e,
                        raw.unaffected_e,
                    ),
                    "paired_stream_id": f"recovery-crn-{seed}",
                }
            )
    return rows


def _nearest_target_parity(amplitude: float, spacing: float) -> int:
    return int(standard_binning_1d(amplitude, lattice=spacing).logical_parity)


def _analytic_nearest_operation_failure(amplitude: float, sigma: float) -> float:
    boundary_distance = abs(0.25 - amplitude)
    return 0.5 * erfc(boundary_distance / (sqrt(2.0) * sigma))


def _run_logical_seed_rows(config: CampaignConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    spacing = config.logical_operation_spacing_over_lattice
    for seed in config.evaluation_seeds:
        for profile in config.logical_noise_profiles:
            rng = np.random.default_rng(
                _seed_stream(seed, f"logical:{profile.profile_id}")
            )
            standardized_jitter = rng.normal(size=config.shots_per_seed_amplitude)
            jitter_hash = _array_hash(standardized_jitter)
            for amplitude in config.amplitudes_over_lattice:
                actual = amplitude + profile.injection_sigma_over_lattice * standardized_jitter
                parity = np.asarray(
                    standard_binning_1d(actual, lattice=spacing).logical_parity,
                    dtype=np.int64,
                )
                target = _nearest_target_parity(amplitude, spacing)
                nearest_failure = parity != target
                rows.append(
                    {
                        "seed": seed,
                        "profile_id": profile.profile_id,
                        "injection_sigma_over_lattice": profile.injection_sigma_over_lattice,
                        "amplitude_over_lattice": amplitude,
                        "logical_distance": float(
                            distance_to_closest_logical_operation(amplitude)
                        ),
                        "nearest_nominal_logical_parity": target,
                        "nearest_operation_logical_failure_rate": float(
                            np.mean(nearest_failure)
                        ),
                        "identity_reference_logical_flip_rate": float(np.mean(parity != 0)),
                        "analytic_nearest_operation_failure_rate": (
                            _analytic_nearest_operation_failure(
                                amplitude, profile.injection_sigma_over_lattice
                            )
                        ),
                        "jitter_trace_sha256": jitter_hash,
                        "paired_stream_id": f"logical-{profile.profile_id}-{seed}",
                    }
                )
    return rows


def _cluster_summary(
    values: Sequence[float],
    *,
    config: CampaignConfig,
    key: str,
) -> dict[str, Any]:
    data = np.asarray(values, dtype=np.float64)
    if data.shape != (len(config.evaluation_seeds),) or not np.all(np.isfinite(data)):
        raise ValueError("cluster summary requires one finite value per evaluation seed")
    rng = np.random.default_rng(_seed_stream(config.bootstrap_seed, key))
    indices = rng.integers(
        0,
        data.size,
        size=(config.seed_cluster_bootstrap_replicates, data.size),
    )
    bootstrap = np.mean(data[indices], axis=1)
    tail = 0.5 * (1.0 - config.confidence_level)
    low, high = np.quantile(bootstrap, [tail, 1.0 - tail])
    return {
        "mean": float(np.mean(data)),
        "ci_low": float(low),
        "ci_high": float(high),
        "paired_seed_cluster_count": int(data.size),
        "bootstrap_replicates": config.seed_cluster_bootstrap_replicates,
        "confidence_level": config.confidence_level,
        "resampling_unit": "whole_seed_cluster",
    }


RECOVERY_METRICS = (
    "mean_initial_recovery_depth",
    "mean_observed_same_quadrature_max_e_run",
    "mean_ideal_same_quadrature_max_e_run",
    "mean_restricted_recovery_cycles",
    "recovered_fraction_by_horizon",
    "unaffected_e_probability_max",
)

LOGICAL_METRICS = (
    "nearest_operation_logical_failure_rate",
    "identity_reference_logical_flip_rate",
    "analytic_nearest_operation_failure_rate",
)


def _summarize_recovery(
    rows: Sequence[Mapping[str, Any]], config: CampaignConfig
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for amplitude in config.amplitudes_over_lattice:
        selected = [row for row in rows if row["amplitude_over_lattice"] == amplitude]
        record: dict[str, Any] = {
            "amplitude_over_lattice": amplitude,
            "logical_distance": float(distance_to_closest_logical_operation(amplitude)),
        }
        for metric in RECOVERY_METRICS:
            record[metric] = _cluster_summary(
                [float(row[metric]) for row in selected],
                config=config,
                key=f"recovery:{amplitude}:{metric}",
            )
        summaries.append(record)
    return summaries


def _summarize_logical(
    rows: Sequence[Mapping[str, Any]], config: CampaignConfig
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for profile in config.logical_noise_profiles:
        for amplitude in config.amplitudes_over_lattice:
            selected = [
                row
                for row in rows
                if row["profile_id"] == profile.profile_id
                and row["amplitude_over_lattice"] == amplitude
            ]
            record: dict[str, Any] = {
                "profile_id": profile.profile_id,
                "injection_sigma_over_lattice": profile.injection_sigma_over_lattice,
                "amplitude_over_lattice": amplitude,
                "logical_distance": float(
                    distance_to_closest_logical_operation(amplitude)
                ),
                "nearest_nominal_logical_parity": _nearest_target_parity(
                    amplitude, config.logical_operation_spacing_over_lattice
                ),
            }
            for metric in LOGICAL_METRICS:
                record[metric] = _cluster_summary(
                    [float(row[metric]) for row in selected],
                    config=config,
                    key=f"logical:{profile.profile_id}:{amplitude}:{metric}",
                )
            summaries.append(record)
    return summaries


def _means(records: Sequence[Mapping[str, Any]], metric: str) -> np.ndarray:
    return np.asarray([row[metric]["mean"] for row in records], dtype=np.float64)


def _strict_peak_at_midpoint(records: Sequence[Mapping[str, Any]], metric: str) -> bool:
    values = _means(records, metric)
    amplitudes = np.asarray([row["amplitude_over_lattice"] for row in records])
    return bool(amplitudes[int(np.argmax(values))] == 0.25)


def _branch_monotone(records: Sequence[Mapping[str, Any]], metric: str) -> bool:
    values = _means(records, metric)
    midpoint = len(values) // 2
    return bool(
        np.all(np.diff(values[: midpoint + 1]) >= -1.0e-12)
        and np.all(np.diff(values[midpoint:]) <= 1.0e-12)
    )


def _logical_profile_records(
    summaries: Sequence[Mapping[str, Any]], profile_id: str
) -> list[Mapping[str, Any]]:
    return [row for row in summaries if row["profile_id"] == profile_id]


def _evaluate_gates(result: Mapping[str, Any]) -> dict[str, bool]:
    config = result["config"]
    recovery = result["recovery_summary"]
    logical = result["logical_summary"]
    midpoint = recovery[len(recovery) // 2]
    max_analytic_gap = max(
        abs(
            row["nearest_operation_logical_failure_rate"]["mean"]
            - row["analytic_nearest_operation_failure_rate"]["mean"]
        )
        for row in logical
    )
    logical_profile_checks = []
    identity_checks = []
    for profile in LOGICAL_NOISE_PROFILES:
        records = _logical_profile_records(logical, profile.profile_id)
        logical_profile_checks.append(
            _strict_peak_at_midpoint(
                records, "nearest_operation_logical_failure_rate"
            )
            and _branch_monotone(
                records, "nearest_operation_logical_failure_rate"
            )
        )
        identity = _means(records, "identity_reference_logical_flip_rate")
        identity_checks.append(bool(np.all(np.diff(identity) >= -1.0e-12)))
    recovery_seed_rows = result["recovery_seed_rows"]
    logical_seed_rows = result["logical_seed_rows"]
    return {
        "all_parent_artifacts_are_current_and_machine_pass": all(
            row["machine_pass"] for row in result["parent_integrity"].values()
        ),
        "implementation_bindings_are_current": all(
            _sha256(row["path"]) == row["sha256"]
            for row in result["implementation_bindings"]
        ),
        "formal_grid_has_17_points_and_midpoint": tuple(
            config["amplitudes_over_lattice"]
        )
        == AMPLITUDES,
        "eight_disjoint_evaluation_seed_clusters": tuple(config["evaluation_seeds"])
        == EVALUATION_SEEDS
        and not (set(config["evaluation_seeds"]) & set(CALIBRATION_SEEDS)),
        "every_recovery_seed_executes_every_amplitude": len(recovery_seed_rows)
        == len(EVALUATION_SEEDS) * len(AMPLITUDES),
        "every_logical_profile_seed_executes_every_amplitude": len(logical_seed_rows)
        == len(EVALUATION_SEEDS) * len(AMPLITUDES) * len(LOGICAL_NOISE_PROFILES),
        "recovery_depth_peaks_at_large_distance_midpoint": _strict_peak_at_midpoint(
            recovery, "mean_initial_recovery_depth"
        )
        and _branch_monotone(recovery, "mean_initial_recovery_depth"),
        "same_quadrature_e_run_peaks_at_large_distance_midpoint": _strict_peak_at_midpoint(
            recovery, "mean_observed_same_quadrature_max_e_run"
        )
        and _branch_monotone(
            recovery, "mean_observed_same_quadrature_max_e_run"
        ),
        "midpoint_depth_matches_frozen_six_level_injection": abs(
            midpoint["mean_initial_recovery_depth"]["mean"]
            - config["max_recovery_depth"]
        )
        <= 0.02,
        "unaffected_quadrature_remains_negative_control": max(
            row["unaffected_e_probability_max"]["ci_high"] for row in recovery
        )
        <= 0.06,
        "nearest_operation_logical_failure_is_midpoint_peaked_in_both_profiles": all(
            logical_profile_checks
        ),
        "identity_reference_logical_flip_rate_is_monotone_not_relabelled": all(
            identity_checks
        ),
        "logical_monte_carlo_matches_independent_gaussian_boundary_formula": max_analytic_gap
        <= 0.012,
        "logical_failure_and_recovery_censoring_are_separate_estimands": result[
            "estimand_contract"
        ]["logical_failure_is_recovery_censoring"]
        is False,
        "logical_truth_is_evaluator_only": result["causal_contract"][
            "logical_truth_visibility"
        ]
        == "evaluator_only_not_controller_input",
        "only_displacement_channel_changes_across_paired_runs": result[
            "causal_contract"
        ]["changed_channel"]
        == "nominal_displacement_amplitude_only",
        "active_fallback_branch_is_preserved": result["active_algorithm_branch"]
        == FALLBACK_BRANCH_ID,
        "device_and_experimental_claims_remain_false": result["device_calibrated"]
        is False
        and result["experimental_hardware_used"] is False
        and result["physical_memory_ler_established"] is False,
        "source_anchors_are_hash_bound": result["source_binding"]["sha256"]
        == _sha256(result["source_binding"]["path"]),
        "semantic_validator_accepts_only_complete_nonmixing_campaign": validate_payload(
            result
        )
        == (),
    }


def _summary_seed_values(
    seed_rows: Sequence[Mapping[str, Any]],
    *,
    amplitude: float,
    metric: str,
    profile_id: str | None = None,
) -> list[float]:
    selected = [
        row
        for row in seed_rows
        if row["amplitude_over_lattice"] == amplitude
        and (profile_id is None or row.get("profile_id") == profile_id)
    ]
    return [float(row[metric]) for row in selected]


def validate_payload(payload: Mapping[str, Any]) -> tuple[str, ...]:
    errors: list[str] = []
    if payload.get("task_id") != TASK_ID:
        errors.append("task identity changed")
    config = payload.get("config", {})
    if tuple(config.get("amplitudes_over_lattice", ())) != AMPLITUDES:
        errors.append("formal amplitude grid changed")
    if tuple(config.get("evaluation_seeds", ())) != EVALUATION_SEEDS:
        errors.append("formal seed clusters changed")
    if set(config.get("evaluation_seeds", ())) & set(CALIBRATION_SEEDS):
        errors.append("evaluation/calibration seed overlap")
    recovery_rows = payload.get("recovery_seed_rows", ())
    logical_rows = payload.get("logical_seed_rows", ())
    if len(recovery_rows) != 136:
        errors.append("recovery seed-amplitude matrix incomplete")
    if len(logical_rows) != 272:
        errors.append("logical profile-seed-amplitude matrix incomplete")
    expected_recovery_keys = {
        (seed, amplitude) for seed in EVALUATION_SEEDS for amplitude in AMPLITUDES
    }
    if {
        (row.get("seed"), row.get("amplitude_over_lattice"))
        for row in recovery_rows
    } != expected_recovery_keys:
        errors.append("recovery seed-amplitude membership changed")
    expected_logical_keys = {
        (seed, profile.profile_id, amplitude)
        for seed in EVALUATION_SEEDS
        for profile in LOGICAL_NOISE_PROFILES
        for amplitude in AMPLITUDES
    }
    if {
        (row.get("seed"), row.get("profile_id"), row.get("amplitude_over_lattice"))
        for row in logical_rows
    } != expected_logical_keys:
        errors.append("logical seed-profile-amplitude membership changed")
    for row in (*recovery_rows, *logical_rows):
        expected_distance = float(
            distance_to_closest_logical_operation(row["amplitude_over_lattice"])
        )
        if abs(float(row.get("logical_distance", -1.0)) - expected_distance) > 1e-15:
            errors.append("nearest-logical-operation distance changed")
            break
    for row in logical_rows:
        for metric in (
            "nearest_operation_logical_failure_rate",
            "identity_reference_logical_flip_rate",
            "analytic_nearest_operation_failure_rate",
        ):
            if not 0.0 <= float(row.get(metric, -1.0)) <= 1.0:
                errors.append("logical rate is outside [0,1]")
                break
        expected_target = _nearest_target_parity(
            row["amplitude_over_lattice"],
            config.get("logical_operation_spacing_over_lattice", 0.5),
        )
        if row.get("nearest_nominal_logical_parity") != expected_target:
            errors.append("nearest nominal logical target changed")
            break
    recovery_summary = payload.get("recovery_summary", ())
    logical_summary = payload.get("logical_summary", ())
    if len(recovery_summary) != 17 or len(logical_summary) != 34:
        errors.append("summary matrices are incomplete")
    for summary in recovery_summary:
        amplitude = summary["amplitude_over_lattice"]
        for metric in RECOVERY_METRICS:
            values = _summary_seed_values(
                recovery_rows, amplitude=amplitude, metric=metric
            )
            if not values or abs(summary[metric]["mean"] - float(np.mean(values))) > 1e-15:
                errors.append("recovery summary no longer matches seed clusters")
                break
    for summary in logical_summary:
        amplitude = summary["amplitude_over_lattice"]
        profile_id = summary["profile_id"]
        for metric in LOGICAL_METRICS:
            values = _summary_seed_values(
                logical_rows,
                amplitude=amplitude,
                metric=metric,
                profile_id=profile_id,
            )
            if not values or abs(summary[metric]["mean"] - float(np.mean(values))) > 1e-15:
                errors.append("logical summary no longer matches seed clusters")
                break
    if len(recovery_summary) == 17:
        if not (
            _strict_peak_at_midpoint(
                recovery_summary, "mean_initial_recovery_depth"
            )
            and _branch_monotone(
                recovery_summary, "mean_initial_recovery_depth"
            )
        ):
            errors.append("recovery-depth causal trend changed")
        if not (
            _strict_peak_at_midpoint(
                recovery_summary, "mean_observed_same_quadrature_max_e_run"
            )
            and _branch_monotone(
                recovery_summary, "mean_observed_same_quadrature_max_e_run"
            )
        ):
            errors.append("same-quadrature e-run causal trend changed")
        if max(
            row["unaffected_e_probability_max"]["ci_high"]
            for row in recovery_summary
        ) > 0.06:
            errors.append("unaffected-quadrature negative control changed")
    if len(logical_summary) == 34:
        max_analytic_gap = 0.0
        for profile in LOGICAL_NOISE_PROFILES:
            records = _logical_profile_records(logical_summary, profile.profile_id)
            if len(records) != 17 or not (
                _strict_peak_at_midpoint(
                    records, "nearest_operation_logical_failure_rate"
                )
                and _branch_monotone(
                    records, "nearest_operation_logical_failure_rate"
                )
            ):
                errors.append("nearest-operation logical-failure trend changed")
                continue
            identity = _means(records, "identity_reference_logical_flip_rate")
            if not np.all(np.diff(identity) >= -1.0e-12):
                errors.append("identity-reference logical-flip trend changed")
            max_analytic_gap = max(
                max_analytic_gap,
                max(
                    abs(
                        row["nearest_operation_logical_failure_rate"]["mean"]
                        - row["analytic_nearest_operation_failure_rate"]["mean"]
                    )
                    for row in records
                ),
            )
        if max_analytic_gap > 0.012:
            errors.append("logical Monte Carlo no longer matches boundary formula")
    estimand = payload.get("estimand_contract", {})
    if not (
        estimand.get("nearest_operation_logical_failure")
        == "actual jittered parity differs from nearest nominal logical operation"
        and estimand.get("identity_reference_logical_flip")
        == "actual jittered parity differs from identity frame"
        and estimand.get("logical_failure_is_recovery_censoring") is False
        and estimand.get("physical_memory_ler") == "NOT_ESTABLISHED"
    ):
        errors.append("logical estimand contract changed")
    causal = payload.get("causal_contract", {})
    if causal.get("changed_channel") != "nominal_displacement_amplitude_only":
        errors.append("causal channel isolation changed")
    if causal.get("logical_truth_visibility") != "evaluator_only_not_controller_input":
        errors.append("logical truth entered controller input")
    if payload.get("active_algorithm_branch") != FALLBACK_BRANCH_ID:
        errors.append("active algorithm branch changed")
    if (
        payload.get("device_calibrated") is not False
        or payload.get("experimental_hardware_used") is not False
        or payload.get("physical_memory_ler_established") is not False
    ):
        errors.append("simulation was promoted to device or physical LER evidence")
    integrity = payload.get("parent_integrity", {})
    if set(integrity) != set(PARENT_ARTIFACTS):
        errors.append("parent integrity membership changed")
    else:
        for task_id, path in PARENT_ARTIFACTS.items():
            record = integrity[task_id]
            if not (
                record.get("path") == path.as_posix()
                and record.get("sha256") == _sha256(path)
                and record.get("machine_pass") is True
            ):
                errors.append("parent artifact binding is stale or failed")
                break
    bindings = payload.get("implementation_bindings", ())
    if len(bindings) != len(IMPLEMENTATION_PATHS) or any(
        row.get("path") != path.as_posix()
        or row.get("sha256") != _sha256(path)
        for row, path in zip(bindings, IMPLEMENTATION_PATHS)
    ):
        errors.append("implementation binding is stale or incomplete")
    source = payload.get("source_binding", {})
    if not (
        source.get("path") == PRIMARY_SOURCE_PATH
        and source.get("sha256") == _sha256(PRIMARY_SOURCE_PATH)
        and tuple(source.get("anchors", ())) == PRIMARY_SOURCE_ANCHORS
    ):
        errors.append("primary source binding changed")
    if "gates" in payload and (
        payload.get("status") != "PASS"
        or not payload.get("gates")
        or not all(value is True for value in payload["gates"].values())
    ):
        errors.append("committed machine gate status is not PASS")
    if "implementation_sha256" in payload and payload.get(
        "implementation_sha256"
    ) != implementation_sha256():
        errors.append("campaign implementation hash is stale")
    if "source_data" in payload:
        source_data = payload["source_data"]
        source_path = _repo_path(source_data.get("path", ""))
        if not source_path.is_file() or source_data.get("sha256") != hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest():
            errors.append("source-data binding is stale")
    return tuple(errors)


def build_report(
    parents: Mapping[str, Mapping[str, Any]],
    integrity: Mapping[str, Mapping[str, Any]],
    config: CampaignConfig | None = None,
) -> dict[str, Any]:
    actual = CampaignConfig() if config is None else config
    if not isinstance(actual, CampaignConfig):
        raise TypeError("config must be CampaignConfig or None")
    missing = set(PARENT_ARTIFACTS) - set(parents)
    if missing:
        raise ValueError(f"missing parent artifacts: {sorted(missing)}")
    recovery_seed_rows = _run_recovery_seed_rows(actual)
    logical_seed_rows = _run_logical_seed_rows(actual)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "protocol_id": PROTOCOL_ID,
        "pass_semantics": (
            "independent causal displacement trends and nonmixing logical estimands pass; "
            "not a device-calibrated physical-memory LER or hardware result"
        ),
        "active_algorithm_branch": FALLBACK_BRANCH_ID,
        "config": {
            **asdict(actual),
            "logical_noise_profiles": [
                asdict(profile) for profile in actual.logical_noise_profiles
            ],
        },
        "parent_integrity": dict(integrity),
        "implementation_bindings": [
            {"path": path.as_posix(), "sha256": _sha256(path)}
            for path in IMPLEMENTATION_PATHS
        ],
        "source_binding": {
            "path": PRIMARY_SOURCE_PATH,
            "sha256": _sha256(PRIMARY_SOURCE_PATH),
            "anchors": list(PRIMARY_SOURCE_ANCHORS),
        },
        "causal_contract": {
            "changed_channel": "nominal_displacement_amplitude_only",
            "paired_randomness": (
                "within each seed cluster, recovery and logical jitter streams are common "
                "across amplitudes; seeds are independent clusters"
            ),
            "fixed_channels": [
                "recovery transition kernel",
                "readout confusion",
                "reset kernel",
                "fault quadrature",
                "horizon",
                "shot count",
            ],
            "logical_truth_visibility": "evaluator_only_not_controller_input",
            "controller_observation": "same-quadrature g/e observation trajectory only",
        },
        "estimand_contract": {
            "recovery_depth": "initial coarse error-hierarchy depth",
            "e_run": "maximum observed consecutive e outcomes in affected quadrature",
            "nearest_operation_logical_failure": (
                "actual jittered parity differs from nearest nominal logical operation"
            ),
            "identity_reference_logical_flip": (
                "actual jittered parity differs from identity frame"
            ),
            "logical_failure_is_recovery_censoring": False,
            "physical_memory_ler": "NOT_ESTABLISHED",
        },
        "recovery_seed_rows": recovery_seed_rows,
        "logical_seed_rows": logical_seed_rows,
        "recovery_summary": _summarize_recovery(recovery_seed_rows, actual),
        "logical_summary": _summarize_logical(logical_seed_rows, actual),
        "device_calibrated": False,
        "experimental_hardware_used": False,
        "physical_memory_ler_established": False,
        "limitations": [
            "coarse population-level error-space recovery, not coherent Fock-space injection",
            "recovery/readout and injection-jitter values are frozen project assumptions",
            "logical classification assay is evaluator-only and is not a repeated-memory LER",
            "no waveform, transmon, target-board, ADC/AWG or quantum-device calibration",
        ],
    }
    errors = validate_payload(result)
    result["validation_errors"] = list(errors)
    gates = _evaluate_gates(result)
    result["gates"] = gates
    result["gate_summary"] = {
        "passed": sum(gates.values()),
        "total": len(gates),
        "failed": [name for name, value in gates.items() if not value],
    }
    result["status"] = "PASS" if all(gates.values()) else "FAIL"
    result["contract_sha256"] = _canonical_sha256(
        {
            "protocol_id": PROTOCOL_ID,
            "config": result["config"],
            "causal_contract": result["causal_contract"],
            "estimand_contract": result["estimand_contract"],
            "recovery_summary": result["recovery_summary"],
            "logical_summary": result["logical_summary"],
            "limitations": result["limitations"],
        }
    )
    return result


CSV_FIELDS = (
    "row_type",
    "seed",
    "profile_id",
    "amplitude_over_lattice",
    "logical_distance",
    "metric",
    "value",
    "ci_low",
    "ci_high",
    "status_or_scope",
    "trace_sha256",
    "source_task",
)


def _source_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(row_type: str, **values: Any) -> None:
        row = {field: "" for field in CSV_FIELDS}
        row.update({"row_type": row_type, **values})
        rows.append(row)

    for task_id, parent in result["parent_integrity"].items():
        add(
            "parent_artifact",
            metric=parent["path"],
            value=parent["sha256"],
            status_or_scope=parent["machine_pass"],
            source_task=task_id,
        )
    for binding in result["implementation_bindings"]:
        add(
            "implementation_binding",
            metric=binding["path"],
            value=binding["sha256"],
            status_or_scope="current",
            source_task=TASK_ID,
        )
    for row in result["recovery_seed_rows"]:
        for metric in RECOVERY_METRICS:
            add(
                "recovery_seed_metric",
                seed=row["seed"],
                amplitude_over_lattice=row["amplitude_over_lattice"],
                logical_distance=row["logical_distance"],
                metric=metric,
                value=row[metric],
                status_or_scope="protocol_aligned_component",
                trace_sha256=row["trace_sha256"],
                source_task=TASK_ID,
            )
    for row in result["logical_seed_rows"]:
        for metric in LOGICAL_METRICS:
            add(
                "logical_seed_metric",
                seed=row["seed"],
                profile_id=row["profile_id"],
                amplitude_over_lattice=row["amplitude_over_lattice"],
                logical_distance=row["logical_distance"],
                metric=metric,
                value=row[metric],
                status_or_scope="evaluator_only_not_physical_memory_ler",
                trace_sha256=row["jitter_trace_sha256"],
                source_task=TASK_ID,
            )
    for row in result["recovery_summary"]:
        for metric in RECOVERY_METRICS:
            summary = row[metric]
            add(
                "recovery_cluster_summary",
                amplitude_over_lattice=row["amplitude_over_lattice"],
                logical_distance=row["logical_distance"],
                metric=metric,
                value=summary["mean"],
                ci_low=summary["ci_low"],
                ci_high=summary["ci_high"],
                status_or_scope="whole_seed_cluster_bootstrap",
                source_task=TASK_ID,
            )
    for row in result["logical_summary"]:
        for metric in LOGICAL_METRICS:
            summary = row[metric]
            add(
                "logical_cluster_summary",
                profile_id=row["profile_id"],
                amplitude_over_lattice=row["amplitude_over_lattice"],
                logical_distance=row["logical_distance"],
                metric=metric,
                value=summary["mean"],
                ci_low=summary["ci_low"],
                ci_high=summary["ci_high"],
                status_or_scope="whole_seed_cluster_bootstrap",
                source_task=TASK_ID,
            )
    for name, passed in result["gates"].items():
        add(
            "contract_gate",
            metric=name,
            value=passed,
            status_or_scope="PASS" if passed else "FAIL",
            source_task=TASK_ID,
        )
    return rows


def write_artifacts(
    *,
    artifact_path: str | Path = DEFAULT_ARTIFACT,
    source_data_path: str | Path = DEFAULT_SOURCE_DATA,
    config: CampaignConfig | None = None,
) -> dict[str, Any]:
    parents = load_parent_artifacts()
    integrity = inspect_parent_integrity(parents)
    result = build_report(parents, integrity, config)
    result["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["implementation_sha256"] = implementation_sha256()
    result["artifact_bindings"] = [
        {
            "task_id": task_id,
            "path": record["path"],
            "sha256": record["sha256"],
            "machine_pass": record["machine_pass"],
        }
        for task_id, record in integrity.items()
    ]
    rows = _source_rows(result)
    source = _repo_path(source_data_path)
    source.parent.mkdir(parents=True, exist_ok=True)
    with source.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    result["source_data"] = {
        "path": Path(source_data_path).as_posix(),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "row_count": len(rows),
        "row_types": sorted({row["row_type"] for row in rows}),
    }
    artifact = _repo_path(artifact_path)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--source-data", type=Path, default=DEFAULT_SOURCE_DATA)
    args = parser.parse_args(argv)
    result = write_artifacts(
        artifact_path=args.artifact,
        source_data_path=args.source_data,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "recovery_seed_rows": len(result["recovery_seed_rows"]),
                "logical_seed_rows": len(result["logical_seed_rows"]),
                "gate_summary": result["gate_summary"],
                "source_rows": result["source_data"]["row_count"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AMPLITUDES",
    "CALIBRATION_SEEDS",
    "CampaignConfig",
    "DEFAULT_ARTIFACT",
    "DEFAULT_SOURCE_DATA",
    "EVALUATION_SEEDS",
    "LOGICAL_NOISE_PROFILES",
    "PARENT_ARTIFACTS",
    "build_report",
    "implementation_sha256",
    "inspect_parent_integrity",
    "load_parent_artifacts",
    "validate_payload",
    "write_artifacts",
]
