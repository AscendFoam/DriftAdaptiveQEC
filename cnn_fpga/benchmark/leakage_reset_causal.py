"""T5.2.3 independent leakage-injection and reset-failure campaign.

The implementation vectorizes the registered T2.0.3 hidden/observed/reset
kernel over independent trajectory clusters.  Exactly one physical assumption
changes in each family: either higher-level leakage injection or higher-level
reset failure.  Truth is used only to score detection and false alarms; no
truth field enters the deployable observation or reset request.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cnn_fpga.benchmark.algorithm_success_falsification import FALLBACK_BRANCH_ID
from physics.sbs_observation_reset import (
    HIDDEN_ANCILLA_STATES,
    IDEAL_ANCILLA_STATES,
    OBSERVED_CLASSES,
    make_persistent_leakage_model,
)
from physics.sbs_occupancy_correlation import (
    PRIMARY_SOURCE_ANCHORS,
    PRIMARY_SOURCE_PATH,
)


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "T5.2.3"
SCHEMA_VERSION = "t5.2.3-independent-leakage-reset-causal-v1"
PROTOCOL_ID = "SBS-INDEPENDENT-LEAKAGE-RESET-INJECTION-V1"
DEFAULT_ARTIFACT = Path("docs/t5_2_3_leakage_reset_causal.json")
DEFAULT_SOURCE_DATA = Path("docs/t5_2_3_leakage_reset_source_data.csv")

PARENT_ARTIFACTS: dict[str, Path] = {
    "T2.0.6": Path("docs/t2_0_6_occupancy_correlation.json"),
    "T5.1.2": Path("docs/t5_1_2_mixed_scenario_matrix.json"),
    "T5.1.6": Path("docs/t5_1_6_experimental_feasibility.json"),
    "T5.2.2": Path("docs/t5_2_2_ancilla_readout_causal.json"),
}

IMPLEMENTATION_PATHS = (
    Path("cnn_fpga/benchmark/leakage_reset_causal.py"),
    Path("physics/sbs_observation_reset.py"),
    Path("physics/sbs_occupancy_correlation.py"),
)

FAMILIES = ("higher_leakage_injection", "higher_reset_failure")
LEAKAGE_INJECTION_RATES = (0.0, 0.00025, 0.0005, 0.001, 0.002, 0.004)
RESET_FAILURE_RATES = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95)
EVALUATION_SEEDS = tuple(202607162301 + index for index in range(8))
CALIBRATION_SEEDS = (
    2026071407,
    2026071408,
    20260716501,
    20260716502,
    *tuple(202607162201 + index for index in range(8)),
)

FIXED_RESET_FAILURE_FOR_LEAKAGE_FAMILY = 0.9
FIXED_LEAKAGE_INJECTION_FOR_RESET_FAMILY = 0.002
FALSE_LEAKAGE_ALARM_PROBABILITY = 0.0002
LEAKAGE_DETECTION_PROBABILITY = 0.95
CORRELATION_LAGS = (1, 2, 4, 8, 16, 32)

SCALAR_METRICS = (
    "injection_episode_rate_per_1000_cycles",
    "empirical_higher_injection_probability",
    "detection_probability",
    "mean_detection_delay_steps",
    "p95_detection_delay_steps",
    "false_alarm_rate_per_healthy_step",
    "false_negative_rate_per_leakage_step",
    "hidden_leakage_occupancy",
    "observed_leakage_alarm_rate",
    "declared_normal_action_availability",
    "safe_normal_action_availability",
    "unsafe_declared_available_fraction",
    "reset_requests_per_1000_cycles",
    "reset_attempts_per_1000_cycles",
    "empirical_reset_failure_probability",
    "reset_failures_per_1000_cycles",
    "successful_resets_per_1000_cycles",
    "mean_hidden_leakage_run_steps",
    "p95_hidden_leakage_run_steps",
    "mean_short_lag_correlation",
    "mean_long_lag_correlation",
    "mean_long_lag_covariance",
)


@dataclass(frozen=True)
class CampaignConfig:
    families: tuple[str, ...] = FAMILIES
    leakage_injection_rates: tuple[float, ...] = LEAKAGE_INJECTION_RATES
    reset_failure_rates: tuple[float, ...] = RESET_FAILURE_RATES
    evaluation_seeds: tuple[int, ...] = EVALUATION_SEEDS
    trajectories_per_seed: int = 256
    burn_in_cycles: int = 128
    evaluation_cycles: int = 512
    seed_cluster_bootstrap_replicates: int = 20000
    bootstrap_seed: int = 202607162399
    confidence_level: float = 0.95
    false_leakage_alarm_probability: float = FALSE_LEAKAGE_ALARM_PROBABILITY
    leakage_detection_probability: float = LEAKAGE_DETECTION_PROBABILITY
    fixed_reset_failure_for_leakage_family: float = (
        FIXED_RESET_FAILURE_FOR_LEAKAGE_FAMILY
    )
    fixed_leakage_injection_for_reset_family: float = (
        FIXED_LEAKAGE_INJECTION_FOR_RESET_FAMILY
    )
    correlation_lags: tuple[int, ...] = CORRELATION_LAGS

    def __post_init__(self) -> None:
        frozen = {
            "families": FAMILIES,
            "leakage_injection_rates": LEAKAGE_INJECTION_RATES,
            "reset_failure_rates": RESET_FAILURE_RATES,
            "evaluation_seeds": EVALUATION_SEEDS,
            "correlation_lags": CORRELATION_LAGS,
        }
        for name, expected in frozen.items():
            if tuple(getattr(self, name)) != expected:
                raise ValueError(f"formal {name} changed")
        if set(self.evaluation_seeds) & set(CALIBRATION_SEEDS):
            raise ValueError("evaluation seeds overlap calibration/pilot seeds")
        exact_scalars = {
            "trajectories_per_seed": 256,
            "burn_in_cycles": 128,
            "evaluation_cycles": 512,
            "seed_cluster_bootstrap_replicates": 20000,
            "confidence_level": 0.95,
            "false_leakage_alarm_probability": FALSE_LEAKAGE_ALARM_PROBABILITY,
            "leakage_detection_probability": LEAKAGE_DETECTION_PROBABILITY,
            "fixed_reset_failure_for_leakage_family": FIXED_RESET_FAILURE_FOR_LEAKAGE_FAMILY,
            "fixed_leakage_injection_for_reset_family": FIXED_LEAKAGE_INJECTION_FOR_RESET_FAMILY,
        }
        for name, expected in exact_scalars.items():
            if getattr(self, name) != expected:
                raise ValueError(f"formal {name} changed")


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
    if task_id == "T2.0.6":
        return payload.get("gate", {}).get("passed") is True
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
    return {
        task_id: json.loads(_repo_path(path).read_text(encoding="utf-8"))
        for task_id, path in paths.items()
    }


def inspect_parent_integrity(
    parents: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    missing = set(PARENT_ARTIFACTS) - set(parents)
    if missing:
        raise ValueError(f"missing parent artifacts: {sorted(missing)}")
    return {
        task_id: {
            "path": path.as_posix(),
            "sha256": _sha256(path),
            "machine_pass": _machine_pass(task_id, parents[task_id]),
        }
        for task_id, path in PARENT_ARTIFACTS.items()
    }


def _source_anchors_current() -> bool:
    lines = _repo_path(PRIMARY_SOURCE_PATH).read_text(encoding="utf-8").splitlines()
    return all(
        0 < int(anchor["line"]) <= len(lines)
        and str(anchor["fragment"]) in lines[int(anchor["line"]) - 1]
        for anchor in PRIMARY_SOURCE_ANCHORS
    )


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


def _family_rates(family: str) -> tuple[float, ...]:
    if family == "higher_leakage_injection":
        return LEAKAGE_INJECTION_RATES
    if family == "higher_reset_failure":
        return RESET_FAILURE_RATES
    raise ValueError(f"unknown family: {family}")


def _channel_spec(family: str, intervention_rate: float) -> dict[str, Any]:
    if family == "higher_leakage_injection":
        injection = intervention_rate
        reset_failure = FIXED_RESET_FAILURE_FOR_LEAKAGE_FAMILY
    elif family == "higher_reset_failure":
        injection = FIXED_LEAKAGE_INJECTION_FOR_RESET_FAMILY
        reset_failure = intervention_rate
    else:
        raise ValueError(f"unknown family: {family}")
    return {
        "changed_channel": family,
        "higher_injection_given_g": injection,
        "higher_injection_given_e": injection,
        "f_injection_given_g": 0.0,
        "f_injection_given_e": 0.0,
        "e_reset_failure_probability": 0.0,
        "f_reset_failure_probability": 0.0,
        "higher_reset_failure_probability": reset_failure,
        "false_leakage_alarm_probability": FALSE_LEAKAGE_ALARM_PROBABILITY,
        "leakage_detection_probability": LEAKAGE_DETECTION_PROBABILITY,
    }


def _observation_model(spec: Mapping[str, Any]):
    false_alarm = float(spec["false_leakage_alarm_probability"])
    detection = float(spec["leakage_detection_probability"])
    confusion = np.asarray(
        [
            [1.0 - false_alarm - 0.0003, 0.0003, false_alarm],
            [0.005, 1.0 - false_alarm - 0.005, false_alarm],
            [0.02, 1.0 - detection - 0.02, detection],
            [0.02, 1.0 - detection - 0.02, detection],
        ],
        dtype=np.float64,
    )
    return make_persistent_leakage_model(
        readout_confusion=confusion,
        f_injection_given_g=float(spec["f_injection_given_g"]),
        f_injection_given_e=float(spec["f_injection_given_e"]),
        higher_injection_given_g=float(spec["higher_injection_given_g"]),
        higher_injection_given_e=float(spec["higher_injection_given_e"]),
        e_reset_success=1.0 - float(spec["e_reset_failure_probability"]),
        f_reset_success=1.0 - float(spec["f_reset_failure_probability"]),
        higher_reset_success=1.0
        - float(spec["higher_reset_failure_probability"]),
        counter_max=2**31 - 1,
        readout_provenance=(
            "T5.2.3 frozen imperfect leakage classifier sensitivity assumption"
        ),
        parameter_provenance=(
            "T5.2.3 one-channel-at-a-time higher-leakage/reset assumptions"
        ),
    )


def _sample_rows(
    probabilities: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    cumulative = np.cumsum(probabilities, axis=1)
    draws = rng.random(probabilities.shape[0])
    return np.minimum(
        np.sum(draws[:, None] > cumulative, axis=1), probabilities.shape[1] - 1
    ).astype(np.int64)


def _ideal_constituent(step: int) -> int:
    # Four-cycle K_gg/K_ge/K_eg/K_ee schedule in chronological X,Z order.
    values = ((0, 0), (1, 0), (0, 1), (1, 1))
    cycle, constituent = divmod(step, 2)
    return values[cycle % 4][constituent]


def _run_lengths(matrix: np.ndarray) -> np.ndarray:
    lengths: list[int] = []
    for trajectory in matrix:
        padded = np.concatenate(([False], trajectory, [False])).astype(np.int8)
        edges = np.diff(padded)
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        lengths.extend((ends - starts).tolist())
    return np.asarray(lengths, dtype=np.int64)


def _lag_diagnostics(
    activity: np.ndarray, lags: Sequence[int]
) -> tuple[list[float], list[float]]:
    correlations: list[float] = []
    covariances: list[float] = []
    for lag in lags:
        x = activity[:, :-lag].reshape(-1).astype(np.float64)
        y = activity[:, lag:].reshape(-1).astype(np.float64)
        mean_x = float(np.mean(x))
        mean_y = float(np.mean(y))
        covariance = float(np.mean(x * y) - mean_x * mean_y)
        denominator = np.sqrt(mean_x * (1.0 - mean_x) * mean_y * (1.0 - mean_y))
        correlations.append(0.0 if denominator == 0.0 else covariance / denominator)
        covariances.append(covariance)
    return correlations, covariances


def _run_seed_cell(
    family: str,
    intervention_rate: float,
    seed: int,
    *,
    config: CampaignConfig,
) -> dict[str, Any]:
    spec = _channel_spec(family, intervention_rate)
    model = _observation_model(spec)
    rng = np.random.default_rng(_seed_stream(seed, f"{family}:paired-rate-stream"))
    shots = config.trajectories_per_seed
    burn_steps = 2 * config.burn_in_cycles
    evaluation_steps = 2 * config.evaluation_cycles
    total_steps = burn_steps + evaluation_steps
    carry = np.zeros(shots, dtype=np.int64)
    g_hidden = HIDDEN_ANCILLA_STATES.index("g")
    higher_hidden = HIDDEN_ANCILLA_STATES.index("higher")
    leakage_observed = OBSERVED_CLASSES.index("leakage")
    hidden_matrix = np.zeros((shots, evaluation_steps), dtype=np.bool_)
    observed_matrix = np.zeros((shots, evaluation_steps), dtype=np.bool_)
    reset_success_matrix = np.zeros((shots, evaluation_steps), dtype=np.bool_)
    reset_failure_matrix = np.zeros((shots, evaluation_steps), dtype=np.bool_)

    episode_start = np.full(shots, -1, dtype=np.int64)
    episode_detected = np.zeros(shots, dtype=np.bool_)
    detection_delays: list[int] = []
    injection_episodes = 0
    detected_episodes = 0
    false_alarms = false_negatives = 0
    healthy_steps = leakage_steps = 0
    reset_requests = reset_attempts = reset_failures = reset_successes = 0
    injection_opportunities = 0

    for absolute_step in range(total_steps):
        ideal = _ideal_constituent(absolute_step)
        entering = carry.copy()
        pre = _sample_rows(model.preparation_kernel[entering, ideal], rng)
        observed = _sample_rows(model.readout_confusion[pre], rng)
        post = _sample_rows(model.reset_kernel[observed, pre], rng)
        if absolute_step >= burn_steps:
            step = absolute_step - burn_steps
            hidden = pre == higher_hidden
            alarm = observed == leakage_observed
            onset = (entering == g_hidden) & hidden
            injection_opportunities += int(np.count_nonzero(entering == g_hidden))
            injection_episodes += int(np.count_nonzero(onset))
            episode_start[onset] = step
            episode_detected[onset] = False
            newly_detected = (
                (episode_start >= 0) & ~episode_detected & hidden & alarm
            )
            if np.any(newly_detected):
                delays = step - episode_start[newly_detected]
                detection_delays.extend(delays.tolist())
                detected_episodes += int(delays.size)
                episode_detected[newly_detected] = True

            healthy = ~hidden
            false_alarms += int(np.count_nonzero(healthy & alarm))
            false_negatives += int(np.count_nonzero(hidden & ~alarm))
            healthy_steps += int(np.count_nonzero(healthy))
            leakage_steps += int(np.count_nonzero(hidden))
            reset_requests += int(np.count_nonzero(alarm))
            attempt = hidden & alarm
            success = attempt & (post == g_hidden)
            failure = attempt & (post == higher_hidden)
            reset_attempts += int(np.count_nonzero(attempt))
            reset_successes += int(np.count_nonzero(success))
            reset_failures += int(np.count_nonzero(failure))
            closed = (episode_start >= 0) & (post == g_hidden)
            episode_start[closed] = -1
            episode_detected[closed] = False
            hidden_matrix[:, step] = hidden
            observed_matrix[:, step] = alarm
            reset_success_matrix[:, step] = success
            reset_failure_matrix[:, step] = failure
        carry = post

    censored_undetected = int(
        np.count_nonzero((episode_start >= 0) & ~episode_detected)
    )
    delays = np.asarray(detection_delays, dtype=np.float64)
    runs = _run_lengths(hidden_matrix)
    correlations, covariances = _lag_diagnostics(
        observed_matrix, config.correlation_lags
    )
    short = slice(0, 3)
    long = slice(3, None)
    denominators = shots * config.evaluation_cycles
    constituent_denominator = shots * evaluation_steps
    detection_probability = (
        detected_episodes / injection_episodes if injection_episodes else None
    )
    mean_delay = float(np.mean(delays)) if delays.size else None
    p95_delay = float(np.quantile(delays, 0.95)) if delays.size else None
    empirical_reset_failure = (
        reset_failures / reset_attempts if reset_attempts else None
    )
    mean_run = float(np.mean(runs)) if runs.size else None
    p95_run = float(np.quantile(runs, 0.95)) if runs.size else None
    return {
        "family": family,
        "intervention_rate": intervention_rate,
        "seed": seed,
        "paired_stream_id": f"{family}-crn-{seed}",
        "channel_spec": spec,
        "trajectories": shots,
        "evaluation_cycles": config.evaluation_cycles,
        "injection_episode_count": injection_episodes,
        "detected_episode_count": detected_episodes,
        "censored_undetected_episode_count": censored_undetected,
        "injection_episode_rate_per_1000_cycles": 1000.0
        * injection_episodes
        / denominators,
        "empirical_higher_injection_probability": injection_episodes
        / injection_opportunities,
        "detection_probability": detection_probability,
        "mean_detection_delay_steps": mean_delay,
        "p95_detection_delay_steps": p95_delay,
        "false_alarm_rate_per_healthy_step": false_alarms / healthy_steps,
        "false_negative_rate_per_leakage_step": (
            false_negatives / leakage_steps if leakage_steps else None
        ),
        "hidden_leakage_occupancy": leakage_steps / constituent_denominator,
        "observed_leakage_alarm_rate": reset_requests / constituent_denominator,
        "declared_normal_action_availability": 1.0
        - reset_requests / constituent_denominator,
        "safe_normal_action_availability": (healthy_steps - false_alarms)
        / constituent_denominator,
        "unsafe_declared_available_fraction": false_negatives
        / constituent_denominator,
        "reset_requests_per_1000_cycles": 1000.0
        * reset_requests
        / denominators,
        "reset_attempts_per_1000_cycles": 1000.0
        * reset_attempts
        / denominators,
        "empirical_reset_failure_probability": empirical_reset_failure,
        "reset_failures_per_1000_cycles": 1000.0
        * reset_failures
        / denominators,
        "successful_resets_per_1000_cycles": 1000.0
        * reset_successes
        / denominators,
        "mean_hidden_leakage_run_steps": mean_run,
        "p95_hidden_leakage_run_steps": p95_run,
        "correlation_lags_steps": list(config.correlation_lags),
        "observed_leakage_correlations": correlations,
        "observed_leakage_covariances": covariances,
        "mean_short_lag_correlation": float(np.mean(correlations[short])),
        "mean_long_lag_correlation": float(np.mean(correlations[long])),
        "mean_long_lag_covariance": float(np.mean(covariances[long])),
        "deployable_observation_fields": [
            "observed_x",
            "observed_z",
            "conditional_reset_action",
            "observed_leakage_run",
        ],
        "truth_used_only_for_scoring": True,
        "trace_sha256": _array_hash(
            hidden_matrix,
            observed_matrix,
            reset_success_matrix,
            reset_failure_matrix,
        ),
    }


def _run_seed_rows(config: CampaignConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in config.families:
        for seed in config.evaluation_seeds:
            for rate in _family_rates(family):
                rows.append(
                    _run_seed_cell(family, rate, seed, config=config)
                )
    return rows


def _cluster_summary(
    values: Sequence[float | None],
    *,
    config: CampaignConfig,
    key: str,
) -> dict[str, Any]:
    if len(values) != len(config.evaluation_seeds):
        raise ValueError("summary requires one value per seed cluster")
    present = [float(value) for value in values if value is not None]
    if not present:
        return {
            "mean": None,
            "ci_low": None,
            "ci_high": None,
            "paired_seed_cluster_count": len(values),
            "nonnull_seed_cluster_count": 0,
            "bootstrap_replicates": config.seed_cluster_bootstrap_replicates,
            "confidence_level": config.confidence_level,
            "resampling_unit": "whole_seed_cluster",
            "status": "NOT_APPLICABLE_NO_TRUE_EPISODES",
        }
    if len(present) != len(values):
        raise ValueError("partial null cluster summary is forbidden")
    data = np.asarray(present, dtype=np.float64)
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
        "paired_seed_cluster_count": len(values),
        "nonnull_seed_cluster_count": len(present),
        "bootstrap_replicates": config.seed_cluster_bootstrap_replicates,
        "confidence_level": config.confidence_level,
        "resampling_unit": "whole_seed_cluster",
        "status": "ESTIMATED",
    }


def _summarize(
    seed_rows: Sequence[Mapping[str, Any]], config: CampaignConfig
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for family in config.families:
        for rate in _family_rates(family):
            selected = [
                row
                for row in seed_rows
                if row["family"] == family and row["intervention_rate"] == rate
            ]
            record: dict[str, Any] = {
                "family": family,
                "intervention_rate": rate,
            }
            for metric in SCALAR_METRICS:
                record[metric] = _cluster_summary(
                    [row[metric] for row in selected],
                    config=config,
                    key=f"{family}:{rate}:{metric}",
                )
            summaries.append(record)
    return summaries


def _family_summary(
    rows: Sequence[Mapping[str, Any]], family: str
) -> list[Mapping[str, Any]]:
    return [row for row in rows if row["family"] == family]


def _means(rows: Sequence[Mapping[str, Any]], metric: str) -> np.ndarray:
    return np.asarray([row[metric]["mean"] for row in rows], dtype=np.float64)


def _strict_increase(rows: Sequence[Mapping[str, Any]], metric: str) -> bool:
    return bool(np.all(np.diff(_means(rows, metric)) > 0.0))


def _strict_decrease(rows: Sequence[Mapping[str, Any]], metric: str) -> bool:
    return bool(np.all(np.diff(_means(rows, metric)) < 0.0))


def _channel_specs_exact(rows: Sequence[Mapping[str, Any]]) -> bool:
    return all(
        _canonical_sha256(row.get("channel_spec"))
        == _canonical_sha256(
            _channel_spec(
                str(row.get("family")), float(row.get("intervention_rate", -1.0))
            )
        )
        for row in rows
    )


def _evaluate_gates(result: Mapping[str, Any]) -> dict[str, bool]:
    summaries = result["summary_rows"]
    leakage = _family_summary(summaries, "higher_leakage_injection")
    reset = _family_summary(summaries, "higher_reset_failure")
    ablation = leakage[0]
    return {
        "all_parent_artifacts_are_current_and_machine_pass": all(
            record["machine_pass"] for record in result["parent_integrity"].values()
        ),
        "implementation_bindings_are_current": all(
            row["sha256"] == _sha256(row["path"])
            for row in result["implementation_bindings"]
        ),
        "primary_source_anchors_are_line_and_hash_bound": (
            result["source_binding"]["sha256"]
            == _sha256(result["source_binding"]["path"])
            and _source_anchors_current()
        ),
        "two_families_and_disjoint_seed_clusters_are_frozen": (
            tuple(result["config"]["families"]) == FAMILIES
            and tuple(result["config"]["evaluation_seeds"]) == EVALUATION_SEEDS
            and not (set(EVALUATION_SEEDS) & set(CALIBRATION_SEEDS))
        ),
        "every_family_seed_rate_cell_executes": len(result["seed_rows"]) == 96,
        "only_one_registered_channel_changes_per_family": _channel_specs_exact(
            result["seed_rows"]
        ),
        "paired_common_random_numbers_cover_all_rates": all(
            len(
                {
                    row["paired_stream_id"]
                    for row in result["seed_rows"]
                    if row["family"] == family and row["seed"] == seed
                }
            )
            == 1
            for family in FAMILIES
            for seed in EVALUATION_SEEDS
        ),
        "leakage_free_ablation_has_no_hidden_leakage_or_true_reset_attempt": (
            ablation["hidden_leakage_occupancy"]["mean"] == 0.0
            and ablation["reset_attempts_per_1000_cycles"]["mean"] == 0.0
            and ablation["reset_failures_per_1000_cycles"]["mean"] == 0.0
            and ablation["detection_probability"]["status"]
            == "NOT_APPLICABLE_NO_TRUE_EPISODES"
        ),
        "leakage_free_ablation_retains_measured_false_alarm_not_fake_zero": (
            5.0e-5
            <= ablation["false_alarm_rate_per_healthy_step"]["mean"]
            <= 4.0e-4
            and ablation["reset_requests_per_1000_cycles"]["mean"] > 0.0
        ),
        "leakage_injection_increases_occupancy_cost_and_unavailability": (
            _strict_increase(leakage, "hidden_leakage_occupancy")
            and _strict_increase(leakage, "reset_attempts_per_1000_cycles")
            and _strict_decrease(leakage, "safe_normal_action_availability")
        ),
        "empirical_leakage_injection_and_fixed_reset_channels_are_calibrated": (
            all(
                abs(
                    row["empirical_higher_injection_probability"]["mean"]
                    - row["intervention_rate"]
                )
                <= 8.0e-5
                and abs(
                    row["empirical_reset_failure_probability"]["mean"]
                    - FIXED_RESET_FAILURE_FOR_LEAKAGE_FAMILY
                )
                <= 0.02
                for row in leakage[1:]
            )
            and leakage[0]["empirical_higher_injection_probability"]["mean"]
            == 0.0
            and all(
                abs(
                    row["empirical_higher_injection_probability"]["mean"]
                    - FIXED_LEAKAGE_INJECTION_FOR_RESET_FAMILY
                )
                <= 2.0e-4
                for row in reset
            )
        ),
        "leakage_injection_creates_observed_correlation_tail_vs_ablation": (
            leakage[-1]["mean_long_lag_covariance"]["mean"]
            > ablation["mean_long_lag_covariance"]["mean"] + 1.0e-5
            and leakage[-1]["mean_short_lag_correlation"]["mean"] > 0.1
        ),
        "reset_failure_increases_persistence_cost_and_unavailability": (
            _strict_increase(reset, "hidden_leakage_occupancy")
            and _strict_increase(reset, "mean_hidden_leakage_run_steps")
            and _strict_increase(reset, "reset_failures_per_1000_cycles")
            and _strict_decrease(reset, "safe_normal_action_availability")
        ),
        "reset_failure_creates_longer_observed_correlation_tail": (
            reset[-1]["mean_long_lag_correlation"]["mean"]
            > reset[0]["mean_long_lag_correlation"]["mean"] + 0.05
        ),
        "detector_probability_delay_and_false_negative_are_explicit": all(
            row["detection_probability"]["mean"] is None
            or (
                row["detection_probability"]["mean"] >= 0.99
                and row["mean_detection_delay_steps"]["mean"] <= 0.12
                and 0.03
                <= row["false_negative_rate_per_leakage_step"]["mean"]
                <= 0.07
            )
            for row in summaries
        ),
        "false_alarm_probability_remains_fixed_across_both_interventions": all(
            5.0e-5 <= row["false_alarm_rate_per_healthy_step"]["mean"] <= 4.0e-4
            for row in summaries
        ),
        "empirical_reset_failure_matches_injected_channel": all(
            row["empirical_reset_failure_probability"]["mean"] is not None
            and abs(
                row["empirical_reset_failure_probability"]["mean"]
                - row["intervention_rate"]
            )
            <= 0.02
            for row in reset
        ),
        "availability_and_recovery_cost_remain_separate_raw_estimands": (
            result["estimand_contract"]["combined_availability_cost_score"]
            == "FORBIDDEN"
            and result["estimand_contract"]["postselection_used_for_primary_metrics"]
            is False
        ),
        "truth_is_evaluator_only_and_deployable_schema_is_fixed": all(
            row["truth_used_only_for_scoring"] is True
            and row["deployable_observation_fields"]
            == [
                "observed_x",
                "observed_z",
                "conditional_reset_action",
                "observed_leakage_run",
            ]
            for row in result["seed_rows"]
        ),
        "uncertainty_resamples_whole_seed_clusters": all(
            row[metric]["resampling_unit"] == "whole_seed_cluster"
            and row[metric]["paired_seed_cluster_count"] == 8
            for row in summaries
            for metric in SCALAR_METRICS
        ),
        "active_fallback_branch_is_preserved": result["active_algorithm_branch"]
        == FALLBACK_BRANCH_ID,
        "device_and_physical_memory_claims_remain_false": (
            result["device_calibrated"] is False
            and result["experimental_hardware_used"] is False
            and result["physical_memory_ler_established"] is False
        ),
        "semantic_validator_accepts_only_complete_nonmixing_campaign": validate_payload(
            result
        )
        == (),
    }


def _expected_config_dict() -> dict[str, Any]:
    return asdict(CampaignConfig())


def _summary_values(
    rows: Sequence[Mapping[str, Any]], family: str, rate: float, metric: str
) -> list[float | None]:
    return [
        row[metric]
        for row in rows
        if row["family"] == family and row["intervention_rate"] == rate
    ]


def _has_forbidden_collapsed_or_postselected_field(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key)
            in {
                "global_score",
                "combined_score",
                "availability_cost_score",
                "postselected_primary_metrics",
                "postselected_seed_rows",
            }
            or _has_forbidden_collapsed_or_postselected_field(child)
            for key, child in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(
            _has_forbidden_collapsed_or_postselected_field(child) for child in value
        )
    return False


def validate_payload(payload: Mapping[str, Any]) -> tuple[str, ...]:
    errors: list[str] = []
    if payload.get("task_id") != TASK_ID or payload.get("protocol_id") != PROTOCOL_ID:
        errors.append("task or protocol identity changed")
    config = dict(payload.get("config", {}))
    for key in (
        "families",
        "leakage_injection_rates",
        "reset_failure_rates",
        "evaluation_seeds",
        "correlation_lags",
    ):
        if key in config:
            config[key] = tuple(config[key])
    if config != _expected_config_dict():
        errors.append("formal campaign configuration changed")
    if set(config.get("evaluation_seeds", ())) & set(CALIBRATION_SEEDS):
        errors.append("evaluation/calibration seed overlap")

    seed_rows = payload.get("seed_rows", ())
    expected_keys = {
        (family, seed, rate)
        for family in FAMILIES
        for seed in EVALUATION_SEEDS
        for rate in _family_rates(family)
    }
    if len(seed_rows) != len(expected_keys) or {
        (row.get("family"), row.get("seed"), row.get("intervention_rate"))
        for row in seed_rows
    } != expected_keys:
        errors.append("family-seed-rate matrix is incomplete or duplicated")
    if seed_rows and not _channel_specs_exact(seed_rows):
        errors.append("one-channel-at-a-time configuration changed")
    for row in seed_rows:
        if row.get("paired_stream_id") != f"{row.get('family')}-crn-{row.get('seed')}":
            errors.append("paired stream identity changed")
            break
        if row.get("truth_used_only_for_scoring") is not True:
            errors.append("truth entered deployable path")
            break
        trace = row.get("trace_sha256", "")
        if not isinstance(trace, str) or len(trace) != 64:
            errors.append("trace hash is missing or malformed")
            break
        for metric in SCALAR_METRICS:
            value = row.get(metric)
            if value is not None and not np.isfinite(float(value)):
                errors.append("seed metric is non-finite")
                break

    summaries = payload.get("summary_rows", ())
    expected_summary_keys = {
        (family, rate) for family in FAMILIES for rate in _family_rates(family)
    }
    if len(summaries) != len(expected_summary_keys) or {
        (row.get("family"), row.get("intervention_rate")) for row in summaries
    } != expected_summary_keys:
        errors.append("family-rate summary matrix is incomplete or duplicated")
    for summary in summaries:
        family = summary.get("family")
        rate = summary.get("intervention_rate")
        for metric in SCALAR_METRICS:
            values = _summary_values(seed_rows, family, rate, metric)
            record = summary.get(metric, {})
            present = [float(value) for value in values if value is not None]
            if not present:
                if not (
                    record.get("mean") is None
                    and record.get("ci_low") is None
                    and record.get("ci_high") is None
                    and record.get("status") == "NOT_APPLICABLE_NO_TRUE_EPISODES"
                ):
                    errors.append("not-applicable summary was converted to a fake zero")
                    break
            elif len(present) != len(values) or abs(
                float(record.get("mean", np.nan)) - float(np.mean(present))
            ) > 1e-15:
                errors.append("summary mean no longer matches complete seed clusters")
                break
            if not (
                record.get("paired_seed_cluster_count") == 8
                and record.get("bootstrap_replicates") == 20000
                and record.get("confidence_level") == 0.95
                and record.get("resampling_unit") == "whole_seed_cluster"
            ):
                errors.append("cluster uncertainty contract changed")
                break

    if len(summaries) == len(expected_summary_keys):
        leakage = _family_summary(summaries, "higher_leakage_injection")
        reset = _family_summary(summaries, "higher_reset_failure")
        ablation = leakage[0]
        if not (
            ablation["hidden_leakage_occupancy"]["mean"] == 0.0
            and ablation["detection_probability"]["mean"] is None
            and ablation["false_alarm_rate_per_healthy_step"]["mean"] > 0.0
        ):
            errors.append("leakage-free ablation contract changed")
        if not (
            _strict_increase(leakage, "hidden_leakage_occupancy")
            and _strict_increase(leakage, "reset_attempts_per_1000_cycles")
            and _strict_decrease(leakage, "safe_normal_action_availability")
        ):
            errors.append("leakage-injection causal direction changed")
        if not (
            all(
                abs(
                    row["empirical_higher_injection_probability"]["mean"]
                    - row["intervention_rate"]
                )
                <= 8.0e-5
                and abs(
                    row["empirical_reset_failure_probability"]["mean"]
                    - FIXED_RESET_FAILURE_FOR_LEAKAGE_FAMILY
                )
                <= 0.02
                for row in leakage[1:]
            )
            and leakage[0]["empirical_higher_injection_probability"]["mean"]
            == 0.0
            and all(
                abs(
                    row["empirical_higher_injection_probability"]["mean"]
                    - FIXED_LEAKAGE_INJECTION_FOR_RESET_FAMILY
                )
                <= 2.0e-4
                for row in reset
            )
        ):
            errors.append("leakage injection or fixed reset channel calibration changed")
        if not (
            _strict_increase(reset, "hidden_leakage_occupancy")
            and _strict_increase(reset, "mean_hidden_leakage_run_steps")
            and _strict_increase(reset, "reset_failures_per_1000_cycles")
            and _strict_decrease(reset, "safe_normal_action_availability")
        ):
            errors.append("reset-failure causal direction changed")
        if any(
            row["empirical_reset_failure_probability"]["mean"] is None
            or abs(
                row["empirical_reset_failure_probability"]["mean"]
                - row["intervention_rate"]
            )
            > 0.02
            for row in reset
        ):
            errors.append("reset-failure empirical rate changed")

    causal = payload.get("causal_contract", {})
    if not (
        causal.get("intervention_rule") == "exactly_one_physical_channel_changes"
        and causal.get("truth_visibility")
        == "truth_only_scores_detection_false_alarm_and_persistence"
        and causal.get("leakage_free_ablation")
        == "higher_leakage_injection family at rate zero"
    ):
        errors.append("causal isolation or truth contract changed")
    estimand = payload.get("estimand_contract", {})
    if not (
        estimand.get("detection_delay")
        == "constituent steps from hidden higher-level injection onset to first observed leakage"
        and estimand.get("false_alarm")
        == "observed leakage while hidden pre-readout state is g/e"
        and estimand.get("postselection_used_for_primary_metrics") is False
        and estimand.get("combined_availability_cost_score") == "FORBIDDEN"
        and estimand.get("physical_memory_ler") == "NOT_ESTABLISHED"
    ):
        errors.append("estimand or nonmixing contract changed")
    if _has_forbidden_collapsed_or_postselected_field(
        {key: value for key, value in payload.items() if key != "estimand_contract"}
    ):
        errors.append("forbidden collapsed score or postselected primary field was introduced")
    if payload.get("active_algorithm_branch") != FALLBACK_BRANCH_ID:
        errors.append("active fallback branch changed")
    if (
        payload.get("device_calibrated") is not False
        or payload.get("experimental_hardware_used") is not False
        or payload.get("physical_memory_ler_established") is not False
    ):
        errors.append("effective simulation was promoted to device or physical LER evidence")

    integrity = payload.get("parent_integrity", {})
    if set(integrity) != set(PARENT_ARTIFACTS):
        errors.append("parent artifact membership changed")
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
        row.get("path") != path.as_posix() or row.get("sha256") != _sha256(path)
        for row, path in zip(bindings, IMPLEMENTATION_PATHS)
    ):
        errors.append("implementation binding is stale or incomplete")
    source = payload.get("source_binding", {})
    if not (
        source.get("path") == PRIMARY_SOURCE_PATH
        and source.get("sha256") == _sha256(PRIMARY_SOURCE_PATH)
        and tuple(source.get("anchors", ())) == PRIMARY_SOURCE_ANCHORS
        and _source_anchors_current()
    ):
        errors.append("primary source binding or line anchor changed")
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
    seed_rows = _run_seed_rows(actual)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "protocol_id": PROTOCOL_ID,
        "pass_semantics": (
            "independent effective leakage/reset causal directions, online observed "
            "detection diagnostics and raw availability/cost reporting pass; not a "
            "device-calibrated reset/leakage rate or physical-memory LER"
        ),
        "active_algorithm_branch": FALLBACK_BRANCH_ID,
        "config": asdict(actual),
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
            "intervention_rule": "exactly_one_physical_channel_changes",
            "paired_randomness": (
                "common random numbers across rates within family/seed; whole independent "
                "seed clusters are the inference unit"
            ),
            "fixed_channels": [
                "balanced ideal g/e constituent schedule",
                "4x3 hidden-to-observed classifier",
                "g/e/f reset success",
                "counter and trajectory horizons",
            ],
            "truth_visibility": (
                "truth_only_scores_detection_false_alarm_and_persistence"
            ),
            "leakage_free_ablation": (
                "higher_leakage_injection family at rate zero"
            ),
        },
        "estimand_contract": {
            "detection_delay": (
                "constituent steps from hidden higher-level injection onset to first observed leakage"
            ),
            "false_alarm": "observed leakage while hidden pre-readout state is g/e",
            "correlation_tail": (
                "pooled observed-leakage lag correlation/covariance without trajectory removal"
            ),
            "availability": (
                "declared observed availability and truth-scored safe availability are separate"
            ),
            "recovery_cost": (
                "raw reset requests, true attempts, successes and failures per 1000 cycles"
            ),
            "postselection_used_for_primary_metrics": False,
            "combined_availability_cost_score": "FORBIDDEN",
            "physical_memory_ler": "NOT_ESTABLISHED",
        },
        "seed_rows": seed_rows,
        "summary_rows": _summarize(seed_rows, actual),
        "device_calibrated": False,
        "experimental_hardware_used": False,
        "physical_memory_ler_established": False,
        "limitations": [
            "effective hidden-state kernel, not cavity-transmon or coherent Fock dynamics",
            "classifier, injection and reset probabilities are frozen project assumptions",
            "false alarms and delays are truth-scored evaluator diagnostics",
            "availability is a software/protocol action fraction, not target-board uptime",
            "no postselection, physical logical channel, device calibration or hardware measurement",
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
            "summary_rows": result["summary_rows"],
            "limitations": result["limitations"],
        }
    )
    return result


CSV_FIELDS = (
    "row_type",
    "family",
    "seed",
    "intervention_rate",
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
    for anchor in result["source_binding"]["anchors"]:
        add(
            "primary_source_anchor",
            metric=f"line:{anchor['line']}:{anchor['role']}",
            value=anchor["fragment"],
            status_or_scope="exact_line_fragment",
            source_task="Sivak2023",
        )
    for row in result["seed_rows"]:
        add(
            "channel_intervention",
            family=row["family"],
            seed=row["seed"],
            intervention_rate=row["intervention_rate"],
            metric="channel_spec_sha256",
            value=_canonical_sha256(row["channel_spec"]),
            status_or_scope="exactly_one_physical_channel_changes",
            trace_sha256=row["trace_sha256"],
            source_task=TASK_ID,
        )
        for metric in SCALAR_METRICS:
            add(
                "seed_metric",
                family=row["family"],
                seed=row["seed"],
                intervention_rate=row["intervention_rate"],
                metric=metric,
                value=row[metric],
                status_or_scope=(
                    "NOT_APPLICABLE_NO_TRUE_EPISODES"
                    if row[metric] is None
                    else "whole_seed_trajectory_cluster"
                ),
                trace_sha256=row["trace_sha256"],
                source_task=TASK_ID,
            )
    for row in result["summary_rows"]:
        for metric in SCALAR_METRICS:
            summary = row[metric]
            add(
                "seed_cluster_summary",
                family=row["family"],
                intervention_rate=row["intervention_rate"],
                metric=metric,
                value=summary["mean"],
                ci_low=summary["ci_low"],
                ci_high=summary["ci_high"],
                status_or_scope=summary["status"],
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
                "seed_rows": len(result["seed_rows"]),
                "summary_rows": len(result["summary_rows"]),
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
    "CALIBRATION_SEEDS",
    "CampaignConfig",
    "DEFAULT_ARTIFACT",
    "DEFAULT_SOURCE_DATA",
    "EVALUATION_SEEDS",
    "FAMILIES",
    "IMPLEMENTATION_PATHS",
    "LEAKAGE_INJECTION_RATES",
    "PARENT_ARTIFACTS",
    "RESET_FAILURE_RATES",
    "SCALAR_METRICS",
    "build_report",
    "implementation_sha256",
    "inspect_parent_integrity",
    "load_parent_artifacts",
    "validate_payload",
    "write_artifacts",
]
