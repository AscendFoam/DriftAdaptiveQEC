"""T5.1.3 average/tail and dual-oracle-gap reporting.

The decoder lane deterministically replays the T5.1.2 scenario/seed protocol
and retains window-level error counts.  Seed is the independent cluster for
paired bootstrap and multiple comparisons.  The control-oracle lane consumes
the separate exact two-cycle branch enumeration and never mixes its fidelity
metrics with syndrome-decoder logical-error probabilities.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import itertools
import json
from math import isfinite
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cnn_fpga.benchmark.continuous_adaptive_map import (
    _calibration_residuals,
    _residuals_and_truth,
    _static_training_parameters,
    select_frozen_hyperparameters,
)
from cnn_fpga.benchmark.mixed_scenario_matrix import (
    DECODER_METHODS,
    DECODER_SCENARIO_IDS,
    decoder_scenarios,
    production_decoder_config,
)
from cnn_fpga.decoder.periodic_adaptive_map import (
    ConstantVelocityPeriodicKalman,
    LatestWindowPeriodicPredictor,
    PeriodicMomentConfig,
    PeriodicMomentEWMA,
    scaled_periodic_kalman_config,
)
from physics.ideal_gkp_decoder import map_decode_2d
from physics.oracle_map import oracle_map_2d


TASK_ID = "T5.1.3"
SCHEMA_VERSION = "t5.1.3-average-tail-dual-oracle-gap-v1"
PROTOCOL_ID = "PAIRED-SEED-TAIL-DUAL-ORACLE-GAP-V1"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT = Path("docs/t5_1_3_oracle_gap_tail_report.json")
DEFAULT_SOURCE_DATA = Path("docs/t5_1_3_oracle_gap_tail_source_data.csv")

BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20260716601
CONFIDENCE_LEVEL = 0.95
MULTIPLICITY_ALPHA = 0.05
MULTIPLICITY_METHOD = "Holm-Bonferroni_two_sided_exact_seed_sign_flip"
CHALLENGER_METHODS = ("standard", "window", "ewma", "kalman")
CONTROL_METRICS = (
    "selection_score",
    "terminal_fidelity",
    "fidelity_effective_lifetime_cycles",
    "logical_z_effective_lifetime_cycles",
)

PARENT_ARTIFACTS = (
    ("T5.1.2", "docs/t5_1_2_mixed_scenario_matrix.json"),
    ("T4.4.4", "docs/t4_4_4_teacher_student_gain_retention.json"),
    ("T3.2.9", "docs/t3_2_9_trajectory_lookup_control_oracle.json"),
)
IMPLEMENTATION_PATHS = (
    "cnn_fpga/benchmark/oracle_gap_tail_report.py",
    "cnn_fpga/benchmark/mixed_scenario_matrix.py",
    "cnn_fpga/benchmark/continuous_adaptive_map.py",
    "cnn_fpga/decoder/periodic_adaptive_map.py",
    "physics/ideal_gkp_decoder.py",
    "physics/oracle_map.py",
    "physics/oracle_gap.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _load_json(relative: str) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _parent_pass(payload: Mapping[str, Any]) -> bool:
    if payload.get("status") == "PASS":
        return True
    gates = payload.get("gates")
    return isinstance(gates, Mapping) and bool(gates) and all(gates.values())


def _artifact_bindings() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_id, relative in PARENT_ARTIFACTS:
        payload = _load_json(relative)
        rows.append(
            {
                "task_id": task_id,
                "path": relative,
                "sha256": _sha256(ROOT / relative),
                "machine_pass": _parent_pass(payload),
            }
        )
    return rows


def _implementation_bindings() -> list[dict[str, str]]:
    return [
        {"path": relative, "sha256": _sha256(ROOT / relative)}
        for relative in IMPLEMENTATION_PATHS
    ]


def _evaluate_seed_windows(
    scenario: Any,
    scenario_index: int,
    evaluation_seed: int,
    settings: Any,
    hyperparameters: Any,
    static_parameters: Any,
    moment_config: PeriodicMomentConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Mirror T3.2.2/T5.1.2 RNG order while retaining window counts."""

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
        calibration, alpha=hyperparameters.ewma_alpha, config=moment_config
    )
    kalman = ConstantVelocityPeriodicKalman(
        calibration,
        moment_config=moment_config,
        kalman_config=scaled_periodic_kalman_config(
            process_scale=hyperparameters.kalman_process_scale,
            measurement_scale=hyperparameters.kalman_measurement_scale,
        ),
    )
    digest = hashlib.sha256()
    digest.update(scenario.scenario_id.encode("utf-8"))
    digest.update(derived_seed.to_bytes(8, "little", signed=False))
    digest.update(np.asarray(calibration, dtype="<f8").tobytes())
    totals = {method: 0 for method in DECODER_METHODS}
    windows: list[dict[str, Any]] = []
    for window_id, state in enumerate(states):
        predictions = {
            "window": latest.prediction(),
            "ewma": ewma.prediction(),
            "kalman": kalman.prediction(),
        }
        residual, truth, displacements = _residuals_and_truth(
            state, settings.evaluation_samples_per_window, rng
        )
        digest.update(window_id.to_bytes(4, "little", signed=False))
        digest.update(np.asarray(displacements, dtype="<f8").tobytes())
        decisions: dict[str, np.ndarray] = {
            "standard": np.zeros(truth.size, dtype=np.int64),
            "static": np.asarray(
                map_decode_2d(
                    residual,
                    static_parameters.covariance_array(),
                    mean=static_parameters.mean_array(),
                ).logical_class,
                dtype=np.int64,
            ),
            "oracle": np.asarray(
                oracle_map_2d(residual, state).logical_class, dtype=np.int64
            ),
        }
        for name, prediction in predictions.items():
            decisions[name] = np.asarray(
                map_decode_2d(
                    residual,
                    prediction.covariance_array(),
                    mean=prediction.mean_array(),
                ).logical_class,
                dtype=np.int64,
            )
        row: dict[str, Any] = {
            "scenario_id": scenario.scenario_id,
            "evaluation_seed": derived_seed,
            "base_evaluation_seed": int(evaluation_seed),
            "window_id": window_id,
            "samples": int(truth.size),
            "burst_active": bool(state.burst_active),
            "event_id": int(state.event_id),
        }
        for method in DECODER_METHODS:
            failures = int(np.sum(decisions[method] != truth))
            totals[method] += failures
            row[f"{method}_failures"] = failures
            row[f"{method}_ler"] = failures / truth.size
        windows.append(row)
        observation = _residuals_and_truth(
            state, settings.observation_samples_per_window, rng
        )[0]
        digest.update(np.asarray(observation, dtype="<f8").tobytes())
        latest.update(observation, window_id=window_id)
        ewma.update(observation, window_id=window_id)
        kalman.update(observation, window_id=window_id)
    samples = settings.windows * settings.evaluation_samples_per_window
    seed_row: dict[str, Any] = {
        "scenario_id": scenario.scenario_id,
        "evaluation_seed": derived_seed,
        "base_evaluation_seed": int(evaluation_seed),
        "windows": settings.windows,
        "evaluation_samples": samples,
        "trace_sha256": digest.hexdigest(),
    }
    for method in DECODER_METHODS:
        seed_row[f"{method}_error_rate"] = totals[method] / samples
    return seed_row, windows


def _interval(values: np.ndarray, confidence: float = CONFIDENCE_LEVEL) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    tail = 0.5 * (1.0 - confidence)
    return {
        "estimate": float(np.mean(array)),
        "ci_low": float(np.quantile(array, tail)),
        "ci_high": float(np.quantile(array, 1.0 - tail)),
        "confidence_level": confidence,
        "replicates": int(array.size),
        "method": "paired_seed_cluster_percentile_bootstrap",
    }


def _method_metrics(
    rates: np.ndarray,
    bootstrap_indices: np.ndarray,
) -> dict[str, Any]:
    if rates.shape != (6, 32):
        raise ValueError("window-rate matrix must have shape (6,32)")
    seed_average = np.mean(rates, axis=1)
    seed_worst = np.max(rates, axis=1)
    average_boot = np.mean(seed_average[bootstrap_indices], axis=1)
    worst_mean_boot = np.mean(seed_worst[bootstrap_indices], axis=1)
    flattened = rates[bootstrap_indices].reshape((bootstrap_indices.shape[0], -1))
    p95_boot = np.quantile(flattened, 0.95, axis=1, method="linear")
    worst_index = np.unravel_index(int(np.argmax(rates)), rates.shape)
    return {
        "p_l": float(np.mean(seed_average)),
        "p_l_bootstrap_ci": _interval(average_boot),
        "window_ler_p95": float(np.quantile(rates, 0.95, method="linear")),
        "window_ler_p95_bootstrap_ci": _interval(p95_boot),
        "observed_worst_window_ler": float(np.max(rates)),
        "observed_worst_seed_index": int(worst_index[0]),
        "observed_worst_window_id": int(worst_index[1]),
        "mean_per_seed_worst_window_ler": float(np.mean(seed_worst)),
        "mean_per_seed_worst_bootstrap_ci": _interval(worst_mean_boot),
        "independent_cluster_unit": "evaluation_seed",
        "window_count": int(rates.size),
    }


def _oracle_gap_metrics(
    method_seed_rates: np.ndarray,
    static_seed_rates: np.ndarray,
    oracle_seed_rates: np.ndarray,
    bootstrap_indices: np.ndarray,
) -> dict[str, Any]:
    for array in (method_seed_rates, static_seed_rates, oracle_seed_rates):
        if np.asarray(array).shape != (6,):
            raise ValueError("oracle-gap inputs must contain six paired seed rates")
    static_gap = float(np.mean(static_seed_rates - oracle_seed_rates))
    method_gap = float(np.mean(method_seed_rates - oracle_seed_rates))
    improvement = float(np.mean(static_seed_rates - method_seed_rates))
    closed = improvement / static_gap if static_gap > 0.0 else None
    remaining = method_gap / static_gap if static_gap > 0.0 else None
    static_boot = np.mean(
        static_seed_rates[bootstrap_indices] - oracle_seed_rates[bootstrap_indices], axis=1
    )
    method_boot = np.mean(
        method_seed_rates[bootstrap_indices] - oracle_seed_rates[bootstrap_indices], axis=1
    )
    improvement_boot = np.mean(
        static_seed_rates[bootstrap_indices] - method_seed_rates[bootstrap_indices], axis=1
    )
    valid = static_boot > 0.0
    closed_boot = improvement_boot[valid] / static_boot[valid]
    remaining_boot = method_boot[valid] / static_boot[valid]
    payload = {
        "static_oracle_gap": static_gap,
        "method_oracle_gap": method_gap,
        "static_minus_method_improvement": improvement,
        "gap_closed_fraction": closed,
        "gap_remaining_fraction": remaining,
        "static_oracle_gap_bootstrap_ci": _interval(static_boot),
        "method_oracle_gap_bootstrap_ci": _interval(method_boot),
        "static_minus_method_bootstrap_ci": _interval(improvement_boot),
        "bootstrap_valid_fraction": float(np.mean(valid)),
        "bootstrap_valid_replicates": int(np.sum(valid)),
        "bootstrap_total_replicates": int(valid.size),
        "denominator_rule": "ratio reported only when paired bootstrap static-minus-oracle denominator is positive",
    }
    if closed_boot.size:
        payload["gap_closed_fraction_bootstrap_ci"] = _interval(closed_boot)
        payload["gap_remaining_fraction_bootstrap_ci"] = _interval(remaining_boot)
    else:
        payload["gap_closed_fraction_bootstrap_ci"] = None
        payload["gap_remaining_fraction_bootstrap_ci"] = None
    return payload


def _exact_sign_flip_pvalue(differences: Sequence[float]) -> float:
    values = np.asarray(differences, dtype=np.float64)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ValueError("exact sign-flip test requires six finite paired differences")
    observed = abs(float(np.mean(values)))
    permuted = np.asarray(
        [
            abs(float(np.mean(values * np.asarray(signs, dtype=np.float64))))
            for signs in itertools.product((-1.0, 1.0), repeat=values.size)
        ],
        dtype=np.float64,
    )
    return float(np.mean(permuted >= observed - 1.0e-15))


def _holm_adjust(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(enumerate(rows), key=lambda item: (item[1]["raw_p_value"], item[0]))
    adjusted = [1.0] * len(rows)
    running = 0.0
    total = len(rows)
    for rank, (original, row) in enumerate(ordered):
        value = min(1.0, (total - rank) * float(row["raw_p_value"]))
        running = max(running, value)
        adjusted[original] = running
    result: list[dict[str, Any]] = []
    for row, value in zip(rows, adjusted):
        result.append(
            {
                **row,
                "holm_adjusted_p_value": value,
                "reject_at_familywise_alpha_0_05": value <= MULTIPLICITY_ALPHA,
            }
        )
    return result


def _decoder_lane() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    parent = _load_json("docs/t5_1_2_mixed_scenario_matrix.json")
    settings = production_decoder_config()
    hyperparameters = select_frozen_hyperparameters(settings)
    static_parameters = _static_training_parameters(settings)
    moment_config = PeriodicMomentConfig(
        minimum_samples=min(64, settings.observation_samples_per_window)
    )
    parent_rows = {
        (row["scenario_id"], int(row["evaluation_seed"])): row
        for row in parent["decoder_lane"]["seed_rows"]
    }
    seed_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    for scenario_index, scenario in enumerate(decoder_scenarios()):
        for seed in settings.evaluation_seeds:
            seed_row, windows = _evaluate_seed_windows(
                scenario,
                20 + scenario_index,
                seed,
                settings,
                hyperparameters,
                static_parameters,
                moment_config,
            )
            parent_row = parent_rows[(scenario.scenario_id, seed_row["evaluation_seed"])]
            seed_row["t5_1_2_trace_match"] = seed_row["trace_sha256"] == parent_row["trace_sha256"]
            seed_row["t5_1_2_max_error_rate_difference"] = max(
                abs(float(seed_row[f"{method}_error_rate"]) - float(parent_row[f"{method}_error_rate"]))
                for method in DECODER_METHODS
            )
            seed_rows.append(seed_row)
            window_rows.extend(windows)

    scenario_reports: list[dict[str, Any]] = []
    multiplicity_rows: list[dict[str, Any]] = []
    for scenario_index, scenario_id in enumerate(DECODER_SCENARIO_IDS):
        scenario_seed_rows = [row for row in seed_rows if row["scenario_id"] == scenario_id]
        scenario_window_rows = [row for row in window_rows if row["scenario_id"] == scenario_id]
        seed_order = [int(row["evaluation_seed"]) for row in scenario_seed_rows]
        bootstrap_rng = np.random.default_rng(BOOTSTRAP_SEED + scenario_index)
        bootstrap_indices = bootstrap_rng.integers(
            0, len(seed_order), size=(BOOTSTRAP_REPLICATES, len(seed_order))
        )
        by_seed_window = {
            (int(row["evaluation_seed"]), int(row["window_id"])): row
            for row in scenario_window_rows
        }
        method_reports: dict[str, Any] = {}
        seed_rates: dict[str, np.ndarray] = {}
        static_seed = np.asarray(
            [row["static_error_rate"] for row in scenario_seed_rows], dtype=np.float64
        )
        oracle_seed = np.asarray(
            [row["oracle_error_rate"] for row in scenario_seed_rows], dtype=np.float64
        )
        for method in DECODER_METHODS:
            rates = np.asarray(
                [
                    [by_seed_window[(seed, window)][f"{method}_ler"] for window in range(settings.windows)]
                    for seed in seed_order
                ],
                dtype=np.float64,
            )
            seed_rates[method] = np.mean(rates, axis=1)
            report = _method_metrics(rates, bootstrap_indices)
            report["decoder_oracle_gap"] = _oracle_gap_metrics(
                seed_rates[method], static_seed, oracle_seed, bootstrap_indices
            )
            method_reports[method] = report
        for method in CHALLENGER_METHODS:
            differences = static_seed - seed_rates[method]
            diff_boot = np.mean(differences[bootstrap_indices], axis=1)
            multiplicity_rows.append(
                {
                    "scenario_id": scenario_id,
                    "comparison": f"{method}_vs_static",
                    "challenger_method": method,
                    "effect_definition": "static_P_L_minus_challenger_P_L_positive_favors_challenger",
                    "paired_seed_effect": float(np.mean(differences)),
                    "paired_seed_effect_bootstrap_ci": _interval(diff_boot),
                    "raw_p_value": _exact_sign_flip_pvalue(differences),
                    "paired_seed_count": 6,
                }
            )
        scenario_reports.append(
            {
                "scenario_id": scenario_id,
                "seed_order": seed_order,
                "window_count": len(scenario_window_rows),
                "samples_per_window": settings.evaluation_samples_per_window,
                "methods": method_reports,
            }
        )
    multiplicity = _holm_adjust(multiplicity_rows)
    return (
        {
            "lane_id": "decoder_syndrome_level_paired",
            "estimand": "syndrome_level_logical_class_decision_error_probability_P_L",
            "independent_unit": "evaluation_seed",
            "paired_unit": "same_scenario_seed_window_displacement_trace",
            "bootstrap": {
                "replicates": BOOTSTRAP_REPLICATES,
                "seed": BOOTSTRAP_SEED,
                "method": "paired_nonparametric_seed_cluster_bootstrap",
                "p95_rule": "resample six whole seed trajectories then recompute empirical 95th percentile across 192 windows",
                "worst_rule": "report observed maximum plus bootstrap CI for mean per-seed maximum; do not present a naive iid-window CI for the global maximum",
            },
            "multiplicity": {
                "family": "four deployable challengers versus static across six scenarios",
                "hypotheses": len(multiplicity),
                "raw_test": "two_sided_exact_sign_flip_over_six_paired_seed_effects",
                "adjustment": MULTIPLICITY_METHOD,
                "familywise_alpha": MULTIPLICITY_ALPHA,
                "discoveries": sum(row["reject_at_familywise_alpha_0_05"] for row in multiplicity),
                "rows": multiplicity,
            },
            "config": asdict(settings),
            "frozen_hyperparameters": asdict(hyperparameters),
            "scenario_reports": scenario_reports,
            "seed_rows": seed_rows,
            "window_rows": window_rows,
        },
        window_rows,
    )


def _control_metric(record: Mapping[str, Any], metric: str) -> float:
    if metric in ("selection_score", "terminal_fidelity"):
        return float(record[metric])
    if metric == "fidelity_effective_lifetime_cycles":
        return float(record["fidelity"]["effective_lifetime_cycles"])
    if metric == "logical_z_effective_lifetime_cycles":
        return float(record["logical_z"]["effective_lifetime_cycles"])
    raise ValueError(f"unknown control metric {metric}")


def _control_lane() -> dict[str, Any]:
    parent = _load_json("docs/t4_4_4_teacher_student_gain_retention.json")
    cutoff_reports: list[dict[str, Any]] = []
    for cutoff in ("12", "16"):
        source = parent["exact_two_cycle"][cutoff]
        oracle = source["control_oracle"]
        records: list[tuple[str, Mapping[str, Any]]] = [
            ("standard", source["standard"]),
            *[
                (str(agent["strategy"]), agent)
                for agent in source["mf_all_agents"]["agents"]
            ],
            ("teacher", source["teacher"]),
            ("handcrafted_recurrence", source["handcrafted_recurrence"]),
            ("distilled_student", source["distilled_student"]),
            ("finite_horizon_control_oracle", oracle),
        ]
        method_rows: list[dict[str, Any]] = []
        for method_id, record in records:
            metrics: dict[str, Any] = {}
            for metric in CONTROL_METRICS:
                value = _control_metric(record, metric)
                reference = _control_metric(oracle, metric)
                metrics[metric] = {
                    "value": value,
                    "control_oracle_value": reference,
                    "control_oracle_minus_method_gap": reference - value,
                }
            method_rows.append({"method_id": method_id, "metrics": metrics})
        cutoff_reports.append(
            {
                "cutoff": int(cutoff),
                "full_cycles": int(source["full_cycles"]),
                "terminal_branches": int(source["control_oracle"]["branch_count"]),
                "trajectory_probability_sum": float(
                    source["control_oracle"]["trajectory_probability_sum"]
                ),
                "methods": method_rows,
            }
        )
    return {
        "lane_id": "control_oracle_short_horizon_exact",
        "role_id": "finite_horizon_control_oracle",
        "scope": "exact_two_cycle_finite_cutoff_two_level_matched_model_expectation",
        "uncertainty_status": "EXACT_BRANCH_EXPECTATION_NO_SAMPLING_CI",
        "why_no_bootstrap_ci": "all 16 terminal branches are enumerated with policy-dependent exact probabilities; resampling branches or treating optimization restarts as experimental seeds would create a false sampling interval",
        "optimization_boundary": "best of the registered finite multistart optimization, not a globally certified control optimum and never a ten-cycle bound",
        "metric_direction": "higher_is_better; gap is control_oracle minus method",
        "cutoffs": cutoff_reports,
    }


def _all_finite_or_none(value: object) -> bool:
    if value is None or isinstance(value, (bool, str)):
        return True
    if isinstance(value, Mapping):
        return all(_all_finite_or_none(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite_or_none(item) for item in value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return isfinite(float(value))
    return True


def validate_report_payload(payload: Mapping[str, Any]) -> tuple[str, ...]:
    if payload.get("task_id") != TASK_ID or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("task/schema identity drifted")
    if payload.get("status") != "PASS":
        raise ValueError("report status must be PASS")
    decoder = payload.get("decoder_lane")
    if not isinstance(decoder, Mapping):
        raise ValueError("decoder lane is missing")
    if len(decoder.get("seed_rows", ())) != 36 or len(decoder.get("window_rows", ())) != 1152:
        raise ValueError("decoder report must retain 36 seed and 1152 window rows")
    if not all(row.get("t5_1_2_trace_match") for row in decoder["seed_rows"]):
        raise ValueError("T5.1.2 replay trace mismatch")
    if max(float(row["t5_1_2_max_error_rate_difference"]) for row in decoder["seed_rows"]) > 1.0e-15:
        raise ValueError("T5.1.2 replay aggregate mismatch")
    if decoder["multiplicity"].get("hypotheses") != 24:
        raise ValueError("multiplicity family must contain exactly 24 hypotheses")
    if decoder["multiplicity"].get("adjustment") != MULTIPLICITY_METHOD:
        raise ValueError("multiplicity adjustment drifted")
    for scenario in decoder.get("scenario_reports", ()):
        if scenario.get("window_count") != 192 or set(scenario.get("methods", {})) != set(DECODER_METHODS):
            raise ValueError("scenario window/method report drifted")
        for metrics in scenario["methods"].values():
            if metrics["window_ler_p95"] > metrics["observed_worst_window_ler"]:
                raise ValueError("p95 window LER exceeds observed worst")
            gap = metrics["decoder_oracle_gap"]
            if gap["bootstrap_valid_fraction"] < 0.95:
                raise ValueError("decoder oracle-gap denominator is not bootstrap reliable")
    control = payload.get("control_oracle_lane")
    if not isinstance(control, Mapping) or control.get("uncertainty_status") != "EXACT_BRANCH_EXPECTATION_NO_SAMPLING_CI":
        raise ValueError("control-oracle exact/no-CI boundary drifted")
    if [row["cutoff"] for row in control.get("cutoffs", ())] != [12, 16]:
        raise ValueError("control-oracle cutoffs drifted")
    if any(row["full_cycles"] != 2 or row["terminal_branches"] != 16 for row in control["cutoffs"]):
        raise ValueError("control oracle must remain exact two-cycle/16-branch")
    bindings = payload.get("artifact_bindings")
    if not isinstance(bindings, list) or len(bindings) != len(PARENT_ARTIFACTS):
        raise ValueError("parent artifact binding count drifted")
    for binding in bindings:
        path = ROOT / binding["path"]
        if not binding["machine_pass"] or _sha256(path) != binding["sha256"]:
            raise ValueError("parent artifact binding is stale or failed")
    implementations = payload.get("implementation_bindings")
    if not isinstance(implementations, list) or len(implementations) != len(IMPLEMENTATION_PATHS):
        raise ValueError("implementation binding count drifted")
    for binding in implementations:
        if _sha256(ROOT / binding["path"]) != binding["sha256"]:
            raise ValueError("implementation binding is stale")
    if not _all_finite_or_none({"decoder": decoder, "control": control}):
        raise ValueError("report contains nonfinite numeric values")
    gates = payload.get("gates")
    if not isinstance(gates, Mapping) or len(gates) != 15 or not all(gates.values()):
        raise ValueError("all fifteen reporting gates must pass")
    source = payload.get("source_data")
    if source is not None:
        path = Path(source["path"])
        if not path.is_absolute():
            path = ROOT / path
        if _sha256(path) != source["sha256"]:
            raise ValueError("source-data hash is stale")
    return (
        "window_level_replay",
        "paired_seed_bootstrap",
        "decoder_oracle_denominator_reliability",
        "holm_familywise_multiplicity",
        "exact_short_horizon_control_oracle_boundary",
        "provenance_and_reporting_gates",
    )


def build_oracle_gap_tail_report() -> dict[str, Any]:
    artifacts = _artifact_bindings()
    implementations = _implementation_bindings()
    decoder, _ = _decoder_lane()
    control = _control_lane()
    gates = {
        "all_parent_artifacts_current_and_pass": all(row["machine_pass"] for row in artifacts),
        "all_implementation_bindings_present": len(implementations) == len(IMPLEMENTATION_PATHS),
        "exact_six_scenario_six_seed_32_window_replay": len(decoder["window_rows"]) == 1152,
        "all_t5_1_2_trace_hashes_reproduced": all(row["t5_1_2_trace_match"] for row in decoder["seed_rows"]),
        "all_t5_1_2_seed_rates_reproduced": max(row["t5_1_2_max_error_rate_difference"] for row in decoder["seed_rows"]) <= 1.0e-15,
        "paired_seed_cluster_bootstrap_20000": decoder["bootstrap"]["replicates"] == 20_000,
        "average_p_l_p95_and_worst_reported": all(
            set(("p_l", "window_ler_p95", "observed_worst_window_ler")) <= set(method)
            for scenario in decoder["scenario_reports"]
            for method in scenario["methods"].values()
        ),
        "p95_never_exceeds_observed_worst": all(
            method["window_ler_p95"] <= method["observed_worst_window_ler"]
            for scenario in decoder["scenario_reports"]
            for method in scenario["methods"].values()
        ),
        "all_decoder_oracle_gap_denominators_reliable": all(
            method["decoder_oracle_gap"]["bootstrap_valid_fraction"] >= 0.95
            for scenario in decoder["scenario_reports"]
            for method in scenario["methods"].values()
        ),
        "holm_family_contains_exact_24_hypotheses": decoder["multiplicity"]["hypotheses"] == 24,
        "holm_adjusted_p_values_monotone_and_bounded": all(
            0.0 <= row["raw_p_value"] <= row["holm_adjusted_p_value"] <= 1.0
            for row in decoder["multiplicity"]["rows"]
        ),
        "control_oracle_exact_two_cycle_dual_cutoff": all(
            row["full_cycles"] == 2 and row["terminal_branches"] == 16
            for row in control["cutoffs"]
        ),
        "control_oracle_probability_normalized": all(
            abs(row["trajectory_probability_sum"] - 1.0) <= 1.0e-12
            for row in control["cutoffs"]
        ),
        "control_oracle_no_fake_sampling_ci": control["uncertainty_status"] == "EXACT_BRANCH_EXPECTATION_NO_SAMPLING_CI",
        "decoder_and_control_metrics_not_cross_ranked": decoder["lane_id"] != control["lane_id"],
    }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "protocol_id": PROTOCOL_ID,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pass_semantics": "average/tail/dual-oracle reporting, paired-seed uncertainty, multiplicity and nonmixing gates pass; this is not the T5.1.4 algorithm-success verdict",
        "artifact_bindings": artifacts,
        "implementation_bindings": implementations,
        "decoder_lane": decoder,
        "control_oracle_lane": control,
        "gates": gates,
        "gate_summary": {
            "passed": sum(gates.values()),
            "total": len(gates),
            "failed": [name for name, passed in gates.items() if not passed],
        },
        "claim_boundary": {
            "allowed": "syndrome-level P_L average/p95/worst and paired decoder-oracle gaps on the frozen T5.1.2 traces, plus exact two-cycle matched-model control-oracle gaps in a separate lane",
            "forbidden": "iid-window uncertainty, post-hoc seed selection, cross-lane oracle ranking, a globally optimal or ten-cycle control oracle, finite-energy/device P_L equivalence, or a T5.1.4 success claim",
        },
    }
    payload["contract_sha256"] = _canonical_sha256(
        {
            key: value
            for key, value in payload.items()
            if key not in {"generated_at_utc", "contract_sha256"}
        }
    )
    normalized = json.loads(json.dumps(payload, ensure_ascii=False))
    validate_report_payload(normalized)
    return normalized


def source_data_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    decoder = payload["decoder_lane"]
    for row in decoder["window_rows"]:
        for method in DECODER_METHODS:
            rows.append(
                {
                    "row_type": "decoder_window",
                    "lane_id": decoder["lane_id"],
                    "scenario_id": row["scenario_id"],
                    "method_id": method,
                    "seed": row["evaluation_seed"],
                    "window_id": row["window_id"],
                    "metric": "window_ler",
                    "value": row[f"{method}_ler"],
                    "detail": f"failures={row[f'{method}_failures']};samples={row['samples']}",
                }
            )
    for scenario in decoder["scenario_reports"]:
        for method, metrics in scenario["methods"].items():
            for metric in ("p_l", "window_ler_p95", "observed_worst_window_ler"):
                rows.append(
                    {
                        "row_type": "decoder_summary",
                        "lane_id": decoder["lane_id"],
                        "scenario_id": scenario["scenario_id"],
                        "method_id": method,
                        "seed": "",
                        "window_id": "",
                        "metric": metric,
                        "value": metrics[metric],
                        "detail": json.dumps(metrics, sort_keys=True),
                    }
                )
    for row in decoder["multiplicity"]["rows"]:
        rows.append(
            {
                "row_type": "multiplicity",
                "lane_id": decoder["lane_id"],
                "scenario_id": row["scenario_id"],
                "method_id": row["challenger_method"],
                "seed": "",
                "window_id": "",
                "metric": "holm_adjusted_p_value",
                "value": row["holm_adjusted_p_value"],
                "detail": json.dumps(row, sort_keys=True),
            }
        )
    for cutoff in payload["control_oracle_lane"]["cutoffs"]:
        for method in cutoff["methods"]:
            for metric, record in method["metrics"].items():
                rows.append(
                    {
                        "row_type": "control_oracle_gap",
                        "lane_id": payload["control_oracle_lane"]["lane_id"],
                        "scenario_id": f"cutoff_{cutoff['cutoff']}_two_cycle",
                        "method_id": method["method_id"],
                        "seed": "exact",
                        "window_id": "",
                        "metric": metric,
                        "value": record["control_oracle_minus_method_gap"],
                        "detail": json.dumps(record, sort_keys=True),
                    }
                )
    for gate_id, passed in payload["gates"].items():
        rows.append(
            {
                "row_type": "gate",
                "lane_id": "acceptance",
                "scenario_id": "",
                "method_id": "",
                "seed": "",
                "window_id": "",
                "metric": gate_id,
                "value": passed,
                "detail": "",
            }
        )
    return rows


def write_artifacts(
    artifact_path: str | Path = DEFAULT_ARTIFACT,
    source_data_path: str | Path = DEFAULT_SOURCE_DATA,
) -> dict[str, Any]:
    payload = build_oracle_gap_tail_report()
    rows = source_data_rows(payload)
    csv_path = Path(source_data_path)
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "row_type",
        "lane_id",
        "scenario_id",
        "method_id",
        "seed",
        "window_id",
        "metric",
        "value",
        "detail",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    try:
        source_label = str(csv_path.relative_to(ROOT))
    except ValueError:
        source_label = str(csv_path)
    payload["source_data"] = {
        "path": source_label,
        "row_count": len(rows),
        "sha256": _sha256(csv_path),
    }
    validate_report_payload(payload)
    output = Path(artifact_path)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", default=str(DEFAULT_ARTIFACT))
    parser.add_argument("--source-data", default=str(DEFAULT_SOURCE_DATA))
    args = parser.parse_args(argv)
    payload = write_artifacts(args.artifact, args.source_data)
    print(
        json.dumps(
            {
                "task_id": TASK_ID,
                "status": payload["status"],
                "window_rows": len(payload["decoder_lane"]["window_rows"]),
                "multiplicity_discoveries": payload["decoder_lane"]["multiplicity"]["discoveries"],
                "source_rows": payload["source_data"]["row_count"],
                "gates": payload["gate_summary"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "TASK_ID",
    "SCHEMA_VERSION",
    "PROTOCOL_ID",
    "DEFAULT_ARTIFACT",
    "DEFAULT_SOURCE_DATA",
    "BOOTSTRAP_REPLICATES",
    "MULTIPLICITY_METHOD",
    "_exact_sign_flip_pvalue",
    "_holm_adjust",
    "validate_report_payload",
    "build_oracle_gap_tail_report",
    "source_data_rows",
    "write_artifacts",
]
