"""T5.4.2 matched uncertainty-gated fallback validation.

The primary action is the frozen T5.1.2 EWMA MAP decision.  When an
observation-only uncertainty score crosses a threshold, the gated system uses
the frozen T5.1.2 static MAP image as a last-known-good fallback.  A no-fallback
system always uses EWMA.  Both consume the same syndrome sample and are scored
against the same logical-class truth.

The threshold is selected only on the already completed T5.4.1 development
seeds.  T5.4.2 confirmation uses twelve fresh seeds.  A fallback may avoid a
logical-class failure, do nothing, or induce a new failure; all three outcomes
are counted, so a safety benefit is not true by construction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from cnn_fpga.benchmark import held_out_ood_validation as held_parent
from cnn_fpga.benchmark.continuous_adaptive_map import (
    _calibration_residuals,
    _mean_interval,
    _residuals_and_truth,
)
from cnn_fpga.decoder.periodic_adaptive_map import (
    ConstantVelocityPeriodicKalman,
    LatestWindowPeriodicPredictor,
    PeriodicMomentConfig,
    PeriodicMomentEWMA,
    scaled_periodic_kalman_config,
)
from physics.drift_processes import DriftState
from physics.ideal_gkp_decoder import map_decode_2d


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "T5.4.2"
SCHEMA_VERSION = "t5.4.2-uncertainty-gated-fallback-v1"
PROTOCOL_ID = "MATCHED-EWMA-STATIC-UNCERTAINTY-GATE-V1"
DEFAULT_ARTIFACT = Path("docs/t5_4_2_uncertainty_gated_fallback.json")
DEFAULT_SOURCE_DATA = Path("docs/t5_4_2_uncertainty_gated_fallback_source_data.csv")

PARENT_ARTIFACTS: dict[str, Path] = {
    "T5.1.2": Path("docs/t5_1_2_mixed_scenario_matrix.json"),
    "T5.1.4": Path("docs/t5_1_4_algorithm_branch_verdict.json"),
    "T4.1.4": Path("docs/t4_1_4_hybrid_multiobjective_validation.json"),
    "T4.2.3": Path("docs/t4_2_3_conservative_fallback_validation.json"),
    "T5.4.1": Path("docs/t5_4_1_held_out_ood_validation.json"),
}

IMPLEMENTATION_PATHS = (
    Path("cnn_fpga/benchmark/uncertainty_gated_fallback.py"),
    Path("cnn_fpga/benchmark/held_out_ood_validation.py"),
    Path("cnn_fpga/benchmark/continuous_adaptive_map.py"),
    Path("cnn_fpga/decoder/periodic_adaptive_map.py"),
    Path("physics/ideal_gkp_decoder.py"),
)

CALIBRATION_SCENARIOS = held_parent.DRIFT_SCENARIOS
CONFIRMATION_OOD_SCENARIOS = held_parent.DRIFT_SCENARIOS
NOMINAL_SCENARIO = "nominal_static_holdout"
CALIBRATION_SEEDS = held_parent.DRIFT_EVALUATION_SEEDS
CONFIRMATION_SEEDS = tuple(202607154501 + index for index in range(12))
THRESHOLD_GRID = tuple(index / 40.0 for index in range(41))
SCORE_COMPONENTS = (
    "ewma_posterior_risk",
    "safe_ensemble_jensen_shannon",
    "hard_decision_disagreement",
)
PRIMARY_METHOD = "frozen_ewma_periodic_map"
FALLBACK_METHOD = "frozen_static_map_last_known_good"
FALLBACK_PROFILE_ID = "t5.1.2-static-map-parent-hash"


@dataclass(frozen=True)
class FallbackValidationConfig:
    calibration_scenarios: tuple[str, ...] = CALIBRATION_SCENARIOS
    confirmation_ood_scenarios: tuple[str, ...] = CONFIRMATION_OOD_SCENARIOS
    nominal_scenario: str = NOMINAL_SCENARIO
    calibration_seeds: tuple[int, ...] = CALIBRATION_SEEDS
    confirmation_seeds: tuple[int, ...] = CONFIRMATION_SEEDS
    threshold_grid: tuple[float, ...] = THRESHOLD_GRID
    windows: int = 64
    calibration_windows: int = 4
    observation_samples_per_window: int = 256
    evaluation_samples_per_window: int = 512
    bootstrap_replicates: int = 20_000
    bootstrap_seed: int = 202607154599
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        frozen = {
            "calibration_scenarios": CALIBRATION_SCENARIOS,
            "confirmation_ood_scenarios": CONFIRMATION_OOD_SCENARIOS,
            "calibration_seeds": CALIBRATION_SEEDS,
            "confirmation_seeds": CONFIRMATION_SEEDS,
            "threshold_grid": THRESHOLD_GRID,
        }
        for name, expected in frozen.items():
            if tuple(getattr(self, name)) != expected:
                raise ValueError(f"formal {name} changed")
        if self.nominal_scenario != NOMINAL_SCENARIO:
            raise ValueError("formal nominal_scenario changed")
        exact = {
            "windows": 64,
            "calibration_windows": 4,
            "observation_samples_per_window": 256,
            "evaluation_samples_per_window": 512,
            "bootstrap_replicates": 20_000,
            "bootstrap_seed": 202607154599,
            "confidence_level": 0.95,
        }
        for name, expected in exact.items():
            if getattr(self, name) != expected:
                raise ValueError(f"formal {name} changed")
        if set(self.calibration_seeds) & set(self.confirmation_seeds):
            raise ValueError("calibration and confirmation seeds overlap")
        if len(self.calibration_seeds) != 8 or len(self.confirmation_seeds) != 12:
            raise ValueError("formal calibration/confirmation cluster counts changed")


class Scenario(Protocol):
    scenario_id: str

    def states(self, windows: int) -> tuple[DriftState, ...]: ...


@dataclass(frozen=True)
class NominalStaticScenario:
    scenario_id: str = NOMINAL_SCENARIO
    dynamic_seed: int = 0

    def states(self, windows: int) -> tuple[DriftState, ...]:
        if windows < 16:
            raise ValueError("windows must be at least 16")
        # Same in-distribution stationary family as the T5.1.2 anchor, with no
        # access to confirmation truth or threshold-dependent parameters.
        from cnn_fpga.benchmark.mixed_scenario_matrix import MixedDecoderScenario

        return MixedDecoderScenario("static_gaussian").states(windows)


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


def _seed_stream(seed: int, stream: str) -> int:
    digest = hashlib.sha256(f"{seed}:{stream}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "little")


def _parent_pass(payload: Mapping[str, Any]) -> bool:
    if payload.get("status") == "PASS" or payload.get("passed") is True:
        return True
    gate = payload.get("gate")
    if isinstance(gate, Mapping) and gate.get("passed") is True:
        return True
    checks = payload.get("checks")
    return bool(isinstance(checks, Mapping) and checks and all(checks.values()))


def load_parent_artifacts() -> dict[str, dict[str, Any]]:
    return {
        task_id: json.loads(_repo_path(path).read_text(encoding="utf-8"))
        for task_id, path in PARENT_ARTIFACTS.items()
    }


def _parent_bindings(
    parents: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if set(parents) != set(PARENT_ARTIFACTS):
        raise ValueError("parent artifact membership changed")
    return [
        {
            "task_id": task_id,
            "path": path.as_posix(),
            "sha256": _sha256(path),
            "machine_pass": _parent_pass(parents[task_id]),
        }
        for task_id, path in PARENT_ARTIFACTS.items()
    ]


def _implementation_bindings() -> list[dict[str, str]]:
    return [
        {"path": path.as_posix(), "sha256": _sha256(path)}
        for path in IMPLEMENTATION_PATHS
    ]


def _uncertainty_score(
    ewma_posterior: np.ndarray,
    safe_ensemble_posteriors: Sequence[np.ndarray],
    ewma_decision: np.ndarray,
    safe_ensemble_decisions: Sequence[np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return an observed-only score; logical truth is deliberately absent."""

    primary = np.asarray(ewma_posterior, dtype=np.float64).reshape((-1, 4))
    ensemble = np.stack(
        [np.asarray(value, dtype=np.float64).reshape((-1, 4)) for value in safe_ensemble_posteriors]
    )
    decisions = np.column_stack(
        [np.asarray(value, dtype=np.int64).reshape(-1) for value in safe_ensemble_decisions]
    )
    primary_decision = np.asarray(ewma_decision, dtype=np.int64).reshape(-1)
    if ensemble.shape[1:] != primary.shape or decisions.shape != (
        primary.shape[0],
        len(safe_ensemble_decisions),
    ):
        raise ValueError("uncertainty ensemble shapes do not align")
    posterior_risk = 1.0 - np.max(primary, axis=1)
    mean_posterior = np.mean(ensemble, axis=0)
    ratio = np.clip(ensemble, 1.0e-15, 1.0) / np.clip(
        mean_posterior[None, :, :], 1.0e-15, 1.0
    )
    js = np.mean(
        np.sum(ensemble * np.log(ratio), axis=2), axis=0
    ) / math.log(4.0)
    disagreement = np.mean(decisions != primary_decision[:, None], axis=1)
    components = {
        "ewma_posterior_risk": np.clip(posterior_risk, 0.0, 1.0),
        "safe_ensemble_jensen_shannon": np.clip(js, 0.0, 1.0),
        "hard_decision_disagreement": np.clip(disagreement, 0.0, 1.0),
    }
    score = np.maximum.reduce([components[name] for name in SCORE_COMPONENTS])
    return np.clip(score, 0.0, 1.0), components


def _scenario_for(
    scenario_id: str, seed: int, *, split: str
) -> Scenario:
    if scenario_id == NOMINAL_SCENARIO:
        return NominalStaticScenario(dynamic_seed=_seed_stream(seed, f"{split}:nominal"))
    return held_parent.OODDriftScenario(
        scenario_id=scenario_id,
        dynamic_seed=_seed_stream(seed, f"drift-dynamics:{scenario_id}"),
    )


def _count_threshold(
    threshold: float,
    score: np.ndarray,
    primary_failed: np.ndarray,
    fallback_failed: np.ndarray,
) -> dict[str, int | float]:
    gate = score >= threshold
    gated_failed = np.where(gate, fallback_failed, primary_failed)
    avoided = gate & primary_failed & ~fallback_failed
    induced = gate & ~primary_failed & fallback_failed
    unnecessary = gate & ~primary_failed
    selected_but_no_benefit = gate & ~(primary_failed & ~fallback_failed)
    return {
        "threshold": float(threshold),
        "decisions": int(score.size),
        "fallback_count": int(np.count_nonzero(gate)),
        "primary_failure_count": int(np.count_nonzero(primary_failed)),
        "static_failure_count": int(np.count_nonzero(fallback_failed)),
        "gated_failure_count": int(np.count_nonzero(gated_failed)),
        "avoided_failure_count": int(np.count_nonzero(avoided)),
        "induced_failure_count": int(np.count_nonzero(induced)),
        "unnecessary_fallback_count": int(np.count_nonzero(unnecessary)),
        "selected_without_benefit_count": int(np.count_nonzero(selected_but_no_benefit)),
    }


def _sum_count_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, int | float]:
    if not rows:
        raise ValueError("count aggregation requires nonempty rows")
    threshold = float(rows[0]["threshold"])
    if any(float(row["threshold"]) != threshold for row in rows):
        raise ValueError("cannot aggregate different thresholds")
    keys = (
        "decisions",
        "fallback_count",
        "primary_failure_count",
        "static_failure_count",
        "gated_failure_count",
        "avoided_failure_count",
        "induced_failure_count",
        "unnecessary_fallback_count",
        "selected_without_benefit_count",
    )
    return {
        "threshold": threshold,
        **{key: sum(int(row[key]) for row in rows) for key in keys},
    }


def _rates(counts: Mapping[str, int | float]) -> dict[str, float]:
    decisions = int(counts["decisions"])
    if decisions <= 0:
        raise ValueError("decision count must be positive")
    primary = int(counts["primary_failure_count"])
    gated = int(counts["gated_failure_count"])
    opportunity = int(counts["avoided_failure_count"]) + int(
        counts["induced_failure_count"]
    )
    return {
        "primary_failure_rate": primary / decisions,
        "static_failure_rate": int(counts["static_failure_count"]) / decisions,
        "gated_failure_rate": gated / decisions,
        "absolute_catastrophic_reduction": (primary - gated) / decisions,
        "relative_catastrophic_reduction": (
            (primary - gated) / primary if primary else 0.0
        ),
        "fallback_rate": int(counts["fallback_count"]) / decisions,
        "avoided_failure_rate": int(counts["avoided_failure_count"]) / decisions,
        "induced_failure_rate": int(counts["induced_failure_count"]) / decisions,
        "unnecessary_fallback_rate": int(counts["unnecessary_fallback_count"])
        / decisions,
        "selected_without_benefit_rate": int(
            counts["selected_without_benefit_count"]
        )
        / decisions,
        "fallback_precision_for_avoided_failure": (
            int(counts["avoided_failure_count"]) / int(counts["fallback_count"])
            if int(counts["fallback_count"])
            else 0.0
        ),
        "net_avoidable_opportunity_fraction": opportunity / decisions,
    }


def _run_cell(
    scenario: Scenario,
    *,
    scenario_index: int,
    base_seed: int,
    split: str,
    thresholds: Sequence[float],
    settings: Any,
    frozen: Any,
    static_map: Any,
    moment_config: PeriodicMomentConfig,
) -> dict[str, Any]:
    derived_seed = int(base_seed + 100_000 * scenario_index)
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
        calibration, alpha=frozen.ewma_alpha, config=moment_config
    )
    kalman = ConstantVelocityPeriodicKalman(
        calibration,
        moment_config=moment_config,
        kalman_config=scaled_periodic_kalman_config(
            process_scale=frozen.kalman_process_scale,
            measurement_scale=frozen.kalman_measurement_scale,
        ),
    )
    count_rows = {
        float(threshold): {
            "threshold": float(threshold),
            "decisions": 0,
            "fallback_count": 0,
            "primary_failure_count": 0,
            "static_failure_count": 0,
            "gated_failure_count": 0,
            "avoided_failure_count": 0,
            "induced_failure_count": 0,
            "unnecessary_fallback_count": 0,
            "selected_without_benefit_count": 0,
        }
        for threshold in thresholds
    }
    component_values = {name: [] for name in SCORE_COMPONENTS}
    score_values: list[np.ndarray] = []
    window_counts: dict[float, list[dict[str, Any]]] = {
        float(threshold): [] for threshold in thresholds
    }
    digest = hashlib.sha256()
    digest.update(split.encode("ascii"))
    digest.update(scenario.scenario_id.encode("utf-8"))
    digest.update(derived_seed.to_bytes(8, "little", signed=False))
    digest.update(np.asarray(calibration, dtype="<f8").tobytes())
    for window_id, state in enumerate(states):
        predictions = {
            "window": latest.prediction(),
            "ewma": ewma.prediction(),
            "kalman": kalman.prediction(),
        }
        residual, truth, displacements = _residuals_and_truth(
            state, settings.evaluation_samples_per_window, rng
        )
        results = {
            name: map_decode_2d(
                residual,
                prediction.covariance_array(),
                mean=prediction.mean_array(),
            )
            for name, prediction in predictions.items()
        }
        results["static"] = map_decode_2d(
            residual,
            static_map.covariance_array(),
            mean=static_map.mean_array(),
        )
        posterior = {
            name: np.asarray(result.posterior, dtype=np.float64).reshape((-1, 4))
            for name, result in results.items()
        }
        decisions = {
            name: np.asarray(result.logical_class, dtype=np.int64).reshape(-1)
            for name, result in results.items()
        }
        score, components = _uncertainty_score(
            posterior["ewma"],
            (posterior["static"], posterior["window"], posterior["kalman"]),
            decisions["ewma"],
            (decisions["static"], decisions["window"], decisions["kalman"]),
        )
        primary_failed = decisions["ewma"] != truth
        fallback_failed = decisions["static"] != truth
        for name, values in components.items():
            component_values[name].append(values)
        score_values.append(score)
        for threshold in thresholds:
            threshold = float(threshold)
            window = _count_threshold(
                threshold, score, primary_failed, fallback_failed
            )
            for key, value in window.items():
                if key != "threshold":
                    count_rows[threshold][key] += int(value)
            window_counts[threshold].append(
                {
                    "window_id": window_id,
                    **window,
                    **_rates(window),
                }
            )
        digest.update(window_id.to_bytes(4, "little", signed=False))
        digest.update(np.asarray(displacements, dtype="<f8").tobytes())
        digest.update(np.asarray(score, dtype="<f8").tobytes())
        observation = _residuals_and_truth(
            state, settings.observation_samples_per_window, rng
        )[0]
        digest.update(np.asarray(observation, dtype="<f8").tobytes())
        # Current-window action/score is complete before the observation update.
        latest.update(observation, window_id=window_id)
        ewma.update(observation, window_id=window_id)
        kalman.update(observation, window_id=window_id)

    all_scores = np.concatenate(score_values)
    component_summary = {
        name: {
            "mean": float(np.mean(np.concatenate(values))),
            "p50": float(np.quantile(np.concatenate(values), 0.50)),
            "p95": float(np.quantile(np.concatenate(values), 0.95)),
            "max": float(np.max(np.concatenate(values))),
        }
        for name, values in component_values.items()
    }
    threshold_rows = []
    for threshold in thresholds:
        threshold = float(threshold)
        counts = count_rows[threshold]
        windows = window_counts[threshold]
        threshold_rows.append(
            {
                **counts,
                **_rates(counts),
                "window_primary_error_p95": float(
                    np.quantile([row["primary_failure_rate"] for row in windows], 0.95)
                ),
                "window_gated_error_p95": float(
                    np.quantile([row["gated_failure_rate"] for row in windows], 0.95)
                ),
                "window_primary_error_max": max(
                    float(row["primary_failure_rate"]) for row in windows
                ),
                "window_gated_error_max": max(
                    float(row["gated_failure_rate"]) for row in windows
                ),
                "window_rows": windows,
            }
        )
    return {
        "split": split,
        "scenario_id": scenario.scenario_id,
        "base_seed": base_seed,
        "derived_seed": derived_seed,
        "dynamic_seed": getattr(scenario, "dynamic_seed", 0),
        "windows": settings.windows,
        "evaluation_samples": settings.windows
        * settings.evaluation_samples_per_window,
        "score_inputs": [
            "current modular syndrome posterior under frozen EWMA/static/window/Kalman",
            "past-only one-window-delayed predictor states",
        ],
        "score_hidden_truth_inputs": [],
        "component_summary": component_summary,
        "score_summary": {
            "mean": float(np.mean(all_scores)),
            "p50": float(np.quantile(all_scores, 0.50)),
            "p95": float(np.quantile(all_scores, 0.95)),
            "p99": float(np.quantile(all_scores, 0.99)),
            "max": float(np.max(all_scores)),
        },
        "threshold_rows": threshold_rows,
        "trace_sha256": digest.hexdigest(),
    }


def _threshold_seed_rows(
    cells: Sequence[Mapping[str, Any]], threshold_grid: Sequence[float]
) -> list[dict[str, Any]]:
    seeds = sorted({int(cell["base_seed"]) for cell in cells})
    rows: list[dict[str, Any]] = []
    for threshold in threshold_grid:
        for seed in seeds:
            selected = [
                next(
                    row
                    for row in cell["threshold_rows"]
                    if float(row["threshold"]) == float(threshold)
                )
                for cell in cells
                if int(cell["base_seed"]) == seed
            ]
            counts = _sum_count_rows(selected)
            rows.append(
                {
                    "threshold": float(threshold),
                    "seed": seed,
                    **counts,
                    **_rates(counts),
                }
            )
    return rows


def _select_threshold(
    seed_rows: Sequence[Mapping[str, Any]], config: FallbackValidationConfig
) -> tuple[float, list[dict[str, Any]]]:
    curve: list[dict[str, Any]] = []
    for threshold in config.threshold_grid:
        selected = [
            row for row in seed_rows if float(row["threshold"]) == float(threshold)
        ]
        reductions = [float(row["absolute_catastrophic_reduction"]) for row in selected]
        fallbacks = [float(row["fallback_rate"]) for row in selected]
        induced = [float(row["induced_failure_rate"]) for row in selected]
        curve.append(
            {
                "threshold": float(threshold),
                "seed_clusters": len(selected),
                "catastrophic_reduction_seed_cluster_ci": _mean_interval(
                    reductions, config.confidence_level
                ),
                "fallback_rate_seed_cluster_ci": _mean_interval(
                    fallbacks, config.confidence_level
                ),
                "induced_failure_rate_seed_cluster_ci": _mean_interval(
                    induced, config.confidence_level
                ),
            }
        )
    # Validation-only objective: maximize mean paired logical-failure reduction;
    # tie-break by fewer fallback actions, fewer induced errors, then the higher
    # threshold.  Confirmation rows are not materialized until after selection.
    chosen = max(
        curve,
        key=lambda row: (
            row["catastrophic_reduction_seed_cluster_ci"]["estimate"],
            -row["fallback_rate_seed_cluster_ci"]["estimate"],
            -row["induced_failure_rate_seed_cluster_ci"]["estimate"],
            row["threshold"],
        ),
    )
    return float(chosen["threshold"]), curve


def _bootstrap_interval(
    values: Sequence[float],
    *,
    config: FallbackValidationConfig,
    key: str,
    resampling_unit: str,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (len(config.confirmation_seeds),) or not np.all(np.isfinite(array)):
        raise ValueError("confirmation bootstrap requires one finite value per seed")
    rng = np.random.default_rng(_seed_stream(config.bootstrap_seed, key))
    indices = rng.integers(0, array.size, size=(config.bootstrap_replicates, array.size))
    means = np.mean(array[indices], axis=1)
    tail = 0.5 * (1.0 - config.confidence_level)
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return {
        "estimate": float(np.mean(array)),
        "ci_low": float(low),
        "ci_high": float(high),
        "seed_clusters": int(array.size),
        "bootstrap_replicates": config.bootstrap_replicates,
        "resampling_unit": resampling_unit,
    }


def _confirmation_summary(
    cells: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
    config: FallbackValidationConfig,
    key_prefix: str,
    resampling_unit: str,
) -> dict[str, Any]:
    seed_rows = _threshold_seed_rows(cells, (threshold,))
    metrics = (
        "primary_failure_rate",
        "static_failure_rate",
        "gated_failure_rate",
        "absolute_catastrophic_reduction",
        "fallback_rate",
        "avoided_failure_rate",
        "induced_failure_rate",
        "unnecessary_fallback_rate",
        "selected_without_benefit_rate",
        "fallback_precision_for_avoided_failure",
    )
    return {
        "threshold": threshold,
        "seed_rows": seed_rows,
        "metrics": {
            metric: _bootstrap_interval(
                [float(row[metric]) for row in seed_rows],
                config=config,
                key=f"{key_prefix}:{metric}",
                resampling_unit=resampling_unit,
            )
            for metric in metrics
        },
        "sample_accounting": _sum_count_rows(
            [
                next(
                    row
                    for row in cell["threshold_rows"]
                    if float(row["threshold"]) == threshold
                )
                for cell in cells
            ]
        ),
    }


def _count_row_is_valid(row: Mapping[str, Any]) -> bool:
    """Recompute every count/rate identity for one threshold or window row."""

    count_keys = (
        "fallback_count",
        "primary_failure_count",
        "static_failure_count",
        "gated_failure_count",
        "avoided_failure_count",
        "induced_failure_count",
        "unnecessary_fallback_count",
        "selected_without_benefit_count",
    )
    try:
        decisions = int(row["decisions"])
        counts = {key: int(row[key]) for key in count_keys}
        if decisions <= 0 or any(value < 0 or value > decisions for value in counts.values()):
            return False
        if counts["gated_failure_count"] != (
            counts["primary_failure_count"]
            - counts["avoided_failure_count"]
            + counts["induced_failure_count"]
        ):
            return False
        if counts["selected_without_benefit_count"] != (
            counts["fallback_count"] - counts["avoided_failure_count"]
        ):
            return False
        if counts["unnecessary_fallback_count"] > counts["selected_without_benefit_count"]:
            return False
        expected = _rates({"decisions": decisions, **counts})
        return all(
            math.isfinite(float(row[name]))
            and abs(float(row[name]) - value) <= 1.0e-15
            for name, value in expected.items()
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _cell_accounting_is_valid(cell: Mapping[str, Any]) -> bool:
    try:
        for threshold_row in cell["threshold_rows"]:
            if not _count_row_is_valid(threshold_row):
                return False
            windows = threshold_row["window_rows"]
            if len(windows) != int(cell["windows"]):
                return False
            if not all(_count_row_is_valid(row) for row in windows):
                return False
            summed = _sum_count_rows(windows)
            for key, value in summed.items():
                if key == "threshold":
                    if float(value) != float(threshold_row[key]):
                        return False
                elif int(value) != int(threshold_row[key]):
                    return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def _contract_view(report: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {
        "generated_at_utc",
        "contract_sha256",
        "source_data",
        "gate_summary",
    }
    return {key: value for key, value in report.items() if key not in excluded}


def _compute_gates(
    report: Mapping[str, Any], parents: Mapping[str, Mapping[str, Any]]
) -> dict[str, bool]:
    config = FallbackValidationConfig()
    calibration = report["calibration"]
    confirmation = report["confirmation_ood"]
    nominal = report["confirmation_nominal"]
    threshold = float(calibration["selected_threshold"])
    selected_curve = next(
        row
        for row in calibration["threshold_curve"]
        if float(row["threshold"]) == threshold
    )
    best = max(
        float(row["catastrophic_reduction_seed_cluster_ci"]["estimate"])
        for row in calibration["threshold_curve"]
    )
    accounting = confirmation["sample_accounting"]
    reduction = confirmation["metrics"]["absolute_catastrophic_reduction"]
    nominal_metrics = nominal["metrics"]
    scenario_summaries = confirmation["scenario_summaries"]
    function_source = inspect.getsource(_uncertainty_score)
    parent_seed_union = set().union(
        *(held_parent._extract_seed_values(parent) for parent in parents.values())
    )
    all_cells = (
        *calibration["cells"],
        *confirmation["cells"],
        *nominal["cells"],
    )
    # T5.4.1 calibration seeds are intentionally reused as development data;
    # only fresh confirmation seeds must be absent from every parent.
    confirmation_seed_set = set(config.confirmation_seeds)
    return {
        "all_parent_artifacts_hash_bound_and_pass": all(
            binding["machine_pass"] and binding["sha256"] == _sha256(binding["path"])
            for binding in report["parent_bindings"]
        ),
        "implementation_files_hash_bound": len(report["implementation_bindings"])
        == len(IMPLEMENTATION_PATHS)
        and all(binding["sha256"] == _sha256(binding["path"]) for binding in report["implementation_bindings"]),
        "calibration_reuses_only_declared_t5_4_1_development_seeds": tuple(
            calibration["seeds"]
        )
        == CALIBRATION_SEEDS
        and report["split_contract"]["t5_4_1_role"] == "development_calibration_only",
        "confirmation_seeds_are_fresh_parent_disjoint_clusters": len(
            confirmation_seed_set
        )
        == 12
        and not (confirmation_seed_set & parent_seed_union),
        "threshold_grid_and_selection_objective_are_frozen": tuple(
            calibration["threshold_grid"]
        )
        == THRESHOLD_GRID
        and len(calibration["threshold_curve"]) == len(THRESHOLD_GRID)
        and abs(
            float(selected_curve["catastrophic_reduction_seed_cluster_ci"]["estimate"])
            - best
        )
        <= 1.0e-15,
        "confirmation_not_used_for_threshold_selection": report["split_contract"][
            "confirmation_used_for_selection"
        ]
        is False
        and calibration["selection_scope"]
        == "t5.4.1_development_seed_cluster_mean_only",
        "matched_primary_fallback_and_no_fallback_actions_are_explicit": report[
            "action_contract"
        ]["primary_method"]
        == PRIMARY_METHOD
        and report["action_contract"]["fallback_method"] == FALLBACK_METHOD
        and report["action_contract"]["no_fallback_method"] == PRIMARY_METHOD,
        "uncertainty_score_is_observed_only_and_truth_absent": "truth" not in inspect.signature(
            _uncertainty_score
        ).parameters
        and "logical_truth" not in function_source
        and report["uncertainty_contract"]["hidden_truth_inputs"] == [],
        "calibration_and_confirmation_cells_are_complete": len(calibration["cells"])
        == len(CALIBRATION_SCENARIOS) * 8
        and len(confirmation["cells"]) == len(CONFIRMATION_OOD_SCENARIOS) * 12
        and len(nominal["cells"]) == 12,
        "confirmation_traces_are_unique_and_threshold_fixed": len(
            {cell["trace_sha256"] for cell in confirmation["cells"]}
        )
        == len(confirmation["cells"])
        and all(
            len(cell["threshold_rows"]) == 1
            and float(cell["threshold_rows"][0]["threshold"]) == threshold
            for cell in (*confirmation["cells"], *nominal["cells"])
        ),
        "catastrophic_definition_is_logical_class_error_not_proxy": report[
            "outcome_contract"
        ]["catastrophic_failure"]
        == "decoded_logical_class_differs_from_hidden_evaluator_class_on_same_sample",
        "confirmation_sample_accounting_identity_holds": int(
            accounting["primary_failure_count"]
        )
        - int(accounting["gated_failure_count"])
        == int(accounting["avoided_failure_count"])
        - int(accounting["induced_failure_count"]),
        "every_cell_threshold_and_window_accounting_recomputes": all(
            _cell_accounting_is_valid(cell) for cell in all_cells
        ),
        "gated_fallback_reduces_confirmatory_ood_catastrophic_failure": reduction[
            "ci_low"
        ]
        > 0.0
        and confirmation["metrics"]["gated_failure_rate"]["estimate"]
        < confirmation["metrics"]["primary_failure_rate"]["estimate"],
        "scenario_heterogeneity_and_negative_lane_are_reported": set(
            scenario_summaries
        )
        == set(CONFIRMATION_OOD_SCENARIOS)
        and any(
            summary["metrics"]["absolute_catastrophic_reduction"]["ci_high"]
            < 0.0
            for summary in scenario_summaries.values()
        )
        and report["claim_boundary"]["scenario_universal_benefit"]
        == "NOT_ESTABLISHED",
        "avoided_failures_exceed_induced_failures": accounting[
            "avoided_failure_count"
        ]
        > accounting["induced_failure_count"],
        "unnecessary_fallback_and_induced_cost_are_reported": all(
            name in confirmation["metrics"]
            for name in (
                "unnecessary_fallback_rate",
                "selected_without_benefit_rate",
                "induced_failure_rate",
                "fallback_rate",
            )
        ),
        "nominal_fallback_burden_and_performance_cost_are_reported": all(
            name in nominal_metrics
            for name in (
                "fallback_rate",
                "unnecessary_fallback_rate",
                "induced_failure_rate",
                "absolute_catastrophic_reduction",
            )
        )
        and nominal["scope"] == "in_distribution_negative_control",
        "always_static_comparator_is_reported_without_being_called_gate": "static_failure_rate"
        in confirmation["metrics"]
        and report["action_contract"]["always_static_role"]
        == "ungated_last_known_good_comparator_not_selected_policy",
        "t4_1_4_all_fallback_negative_result_is_not_relabelled": report[
            "parent_negative_evidence"
        ]["t4_1_4_evaluation_false_fallback_rate"]
        == 1.0
        and report["parent_negative_evidence"]["score_contract_reused"] is False,
        "no_device_or_physical_memory_claim": report["device_calibrated"] is False
        and report["physical_memory_ler_established"] is False
        and report["claim_boundary"]["fallback_scope"]
        == "syndrome_decision_level_last_known_good_map_selection",
    }


def validate_artifact(report: Mapping[str, Any]) -> tuple[str, ...]:
    errors: list[str] = []
    if report.get("task_id") != TASK_ID or report.get("schema_version") != SCHEMA_VERSION:
        errors.append("task/schema identity mismatch")
    if report.get("protocol_id") != PROTOCOL_ID:
        errors.append("protocol identity mismatch")
    if _canonical_sha256(report.get("config")) != _canonical_sha256(
        asdict(FallbackValidationConfig())
    ):
        errors.append("pre-registered config drifted")
    gates = report.get("gates")
    if not isinstance(gates, Mapping) or not gates or not all(value is True for value in gates.values()):
        errors.append("one or more gates failed")
    try:
        recomputed = _compute_gates(report, load_parent_artifacts())
        if gates != recomputed:
            errors.append("stored gates do not match recomputed evidence gates")
        if report.get("contract_sha256") != _canonical_sha256(_contract_view(report)):
            errors.append("contract hash mismatch")
        if report.get("status") != "PASS":
            errors.append("artifact status is not PASS")
        source = report["source_data"]
        rows = source_rows(report)
        if int(source["row_count"]) != len(rows):
            errors.append("source-data row count mismatch")
        if source["rows_sha256"] != _canonical_sha256(rows):
            errors.append("source-data canonical row hash mismatch")
        path = _repo_path(source["path"])
        if path.exists() and source.get("csv_sha256") != _sha256(path):
            errors.append("source-data CSV byte hash mismatch")
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"malformed artifact: {exc}")
    return tuple(errors)


def build_report(config: FallbackValidationConfig | None = None) -> dict[str, Any]:
    actual = FallbackValidationConfig() if config is None else config
    if not isinstance(actual, FallbackValidationConfig):
        raise TypeError("config must be FallbackValidationConfig")
    parents = load_parent_artifacts()
    held_config = held_parent.HeldOutOODConfig()
    base_settings, frozen, static_map, frozen_binding = held_parent._restore_decoder_parent(
        parents["T5.1.2"], held_config
    )
    settings = replace(
        base_settings,
        evaluation_seeds=actual.confirmation_seeds,
        windows=actual.windows,
        calibration_windows=actual.calibration_windows,
        observation_samples_per_window=actual.observation_samples_per_window,
        evaluation_samples_per_window=actual.evaluation_samples_per_window,
    )
    moment = PeriodicMomentConfig(
        minimum_samples=min(64, settings.observation_samples_per_window)
    )

    calibration_cells = [
        _run_cell(
            _scenario_for(scenario_id, seed, split="calibration"),
            scenario_index=70 + scenario_index,
            base_seed=seed,
            split="calibration",
            thresholds=actual.threshold_grid,
            settings=settings,
            frozen=frozen,
            static_map=static_map,
            moment_config=moment,
        )
        for scenario_index, scenario_id in enumerate(actual.calibration_scenarios)
        for seed in actual.calibration_seeds
    ]
    calibration_seed_rows = _threshold_seed_rows(
        calibration_cells, actual.threshold_grid
    )
    selected_threshold, threshold_curve = _select_threshold(
        calibration_seed_rows, actual
    )

    confirmation_cells = [
        _run_cell(
            _scenario_for(scenario_id, seed, split="confirmation"),
            scenario_index=80 + scenario_index,
            base_seed=seed,
            split="confirmation_ood",
            thresholds=(selected_threshold,),
            settings=settings,
            frozen=frozen,
            static_map=static_map,
            moment_config=moment,
        )
        for scenario_index, scenario_id in enumerate(actual.confirmation_ood_scenarios)
        for seed in actual.confirmation_seeds
    ]
    nominal_cells = [
        _run_cell(
            _scenario_for(NOMINAL_SCENARIO, seed, split="confirmation"),
            scenario_index=90,
            base_seed=seed,
            split="confirmation_nominal",
            thresholds=(selected_threshold,),
            settings=settings,
            frozen=frozen,
            static_map=static_map,
            moment_config=moment,
        )
        for seed in actual.confirmation_seeds
    ]

    confirmation_summary = _confirmation_summary(
        confirmation_cells,
        threshold=selected_threshold,
        config=actual,
        key_prefix="confirmation-ood",
        resampling_unit="base_seed_cluster_aggregated_across_three_registered_scenarios",
    )
    nominal_summary = _confirmation_summary(
        nominal_cells,
        threshold=selected_threshold,
        config=actual,
        key_prefix="confirmation-nominal",
        resampling_unit="base_seed_cluster_within_nominal_scenario",
    )
    scenario_summaries = {
        scenario_id: _confirmation_summary(
            [
                cell
                for cell in confirmation_cells
                if cell["scenario_id"] == scenario_id
            ],
            threshold=selected_threshold,
            config=actual,
            key_prefix=f"confirmation-ood:{scenario_id}",
            resampling_unit="base_seed_cluster_within_single_registered_scenario",
        )
        for scenario_id in actual.confirmation_ood_scenarios
    }
    confirmation_summary.update(
        {
            "scope": "three_registered_ood_scenarios_aggregated_by_base_seed",
            "scenario_ids": list(actual.confirmation_ood_scenarios),
            "scenario_summaries": scenario_summaries,
            "cells": confirmation_cells,
        }
    )
    nominal_summary.update(
        {
            "scope": "in_distribution_negative_control",
            "scenario_ids": [NOMINAL_SCENARIO],
            "cells": nominal_cells,
        }
    )
    t414_eval = parents["T4.1.4"]["evaluation_frozen"]["diagnostics"]
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "protocol_id": PROTOCOL_ID,
        "pass_semantics": (
            "the validation-only selected observed uncertainty gate lowers paired "
            "confirmatory OOD logical-class failure with a positive seed-cluster CI; "
            "unnecessary and induced fallback costs remain explicit"
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(actual),
        "parent_bindings": _parent_bindings(parents),
        "implementation_bindings": _implementation_bindings(),
        "split_contract": {
            "t5_4_1_role": "development_calibration_only",
            "calibration_seeds": list(actual.calibration_seeds),
            "confirmation_seeds": list(actual.confirmation_seeds),
            "confirmation_used_for_selection": False,
            "selection_precedes_confirmation_materialization": True,
            "resampling_unit": "base_seed_cluster",
        },
        "frozen_decoder_binding": frozen_binding,
        "uncertainty_contract": {
            "score_components": list(SCORE_COMPONENTS),
            "combination": "maximum_of_three_unit_interval_components",
            "current_inputs": "current observed modular syndrome posteriors",
            "state_inputs": "past-only one-window-delayed frozen predictors",
            "hidden_truth_inputs": [],
            "threshold_rule": (
                "maximize T5.4.1 development seed-cluster mean primary-minus-gated "
                "logical failure; tie-break lower fallback, lower induced failure, higher threshold"
            ),
        },
        "action_contract": {
            "primary_method": PRIMARY_METHOD,
            "fallback_method": FALLBACK_METHOD,
            "fallback_profile_id": FALLBACK_PROFILE_ID,
            "no_fallback_method": PRIMARY_METHOD,
            "always_static_role": "ungated_last_known_good_comparator_not_selected_policy",
            "same_sample_and_truth_for_all_actions": True,
            "causal_order": "score_and_action_before_current_window_predictor_update",
        },
        "outcome_contract": {
            "catastrophic_failure": "decoded_logical_class_differs_from_hidden_evaluator_class_on_same_sample",
            "avoided_failure": "gate_and_primary_wrong_and_static_correct",
            "induced_failure": "gate_and_primary_correct_and_static_wrong",
            "unnecessary_fallback": "gate_while_primary_would_be_correct",
            "selected_without_benefit": "gate_without_primary_wrong_static_correct",
            "truth_use": "offline_scoring_only_never_score_or_action_input",
        },
        "parent_negative_evidence": {
            "t4_1_4_evaluation_false_fallback_rate": float(
                t414_eval["false_fallback_rate"]
            ),
            "t4_1_4_required_fallback_recall": float(
                t414_eval["required_fallback_recall"]
            ),
            "score_contract_reused": False,
            "reason": (
                "T4.1.4 future-horizon hybrid score selected all evaluation records; "
                "T5.4.2 tests a distinct matched per-decision observed ensemble gate"
            ),
        },
        "calibration": {
            "role": "development_only_no_confirmatory_claim",
            "selection_scope": "t5.4.1_development_seed_cluster_mean_only",
            "scenario_ids": list(actual.calibration_scenarios),
            "seeds": list(actual.calibration_seeds),
            "threshold_grid": list(actual.threshold_grid),
            "selected_threshold": selected_threshold,
            "threshold_curve": threshold_curve,
            "threshold_seed_rows": calibration_seed_rows,
            "cells": calibration_cells,
        },
        "confirmation_ood": confirmation_summary,
        "confirmation_nominal": nominal_summary,
        "device_calibrated": False,
        "physical_memory_ler_established": False,
        "claim_boundary": {
            "allowed": (
                "paired syndrome-decision evidence that an observed uncertainty gate "
                "selecting a frozen static MAP reduces confirmatory synthetic OOD logical-class errors"
            ),
            "forbidden": (
                "physical-memory LER, device catastrophic-failure probability, universal "
                "OOD safety, controller/RTL/board fallback, or reuse of T4.1.4 calibration claims"
            ),
            "fallback_scope": "syndrome_decision_level_last_known_good_map_selection",
            "scenario_universal_benefit": "NOT_ESTABLISHED",
        },
    }
    report["gates"] = _compute_gates(report, parents)
    report["gate_summary"] = {
        "passed": sum(bool(value) for value in report["gates"].values()),
        "total": len(report["gates"]),
    }
    report["status"] = "PASS" if all(report["gates"].values()) else "FAIL"
    report["source_data"] = {
        "path": DEFAULT_SOURCE_DATA.as_posix(),
        "row_count": len(source_rows(report)),
        "rows_sha256": _canonical_sha256(source_rows(report)),
        "csv_sha256": None,
    }
    report["contract_sha256"] = _canonical_sha256(_contract_view(report))
    return report


def source_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for binding in report["parent_bindings"]:
        rows.append(
            {
                "row_type": "parent_binding",
                "record_id": binding["task_id"],
                "split": "provenance",
                "scenario": "",
                "seed": "",
                "threshold": "",
                "metric": "machine_pass",
                "value": int(binding["machine_pass"]),
                "detail": json.dumps(binding, sort_keys=True),
            }
        )
    for row in report["calibration"]["threshold_seed_rows"]:
        rows.append(
            {
                "row_type": "calibration_threshold_seed",
                "record_id": f"t{row['threshold']}-s{row['seed']}",
                "split": "calibration",
                "scenario": "all_registered_development",
                "seed": row["seed"],
                "threshold": row["threshold"],
                "metric": "absolute_catastrophic_reduction",
                "value": row["absolute_catastrophic_reduction"],
                "detail": json.dumps(row, sort_keys=True),
            }
        )
    for row in report["calibration"]["threshold_curve"]:
        rows.append(
            {
                "row_type": "calibration_threshold_aggregate",
                "record_id": f"t{row['threshold']}",
                "split": "calibration",
                "scenario": "all_registered_development",
                "seed": "",
                "threshold": row["threshold"],
                "metric": "absolute_catastrophic_reduction",
                "value": row["catastrophic_reduction_seed_cluster_ci"]["estimate"],
                "detail": json.dumps(row, sort_keys=True),
            }
        )
    for section_name in ("confirmation_ood", "confirmation_nominal"):
        section = report[section_name]
        for row in section["seed_rows"]:
            rows.append(
                {
                    "row_type": "confirmation_seed",
                    "record_id": f"{section_name}-s{row['seed']}",
                    "split": section_name,
                    "scenario": "|".join(section["scenario_ids"]),
                    "seed": row["seed"],
                    "threshold": row["threshold"],
                    "metric": "absolute_catastrophic_reduction",
                    "value": row["absolute_catastrophic_reduction"],
                    "detail": json.dumps(row, sort_keys=True),
                }
            )
        for metric, interval in section["metrics"].items():
            rows.append(
                {
                    "row_type": "confirmation_aggregate",
                    "record_id": section_name,
                    "split": section_name,
                    "scenario": "|".join(section["scenario_ids"]),
                    "seed": "",
                    "threshold": section["threshold"],
                    "metric": metric,
                    "value": interval["estimate"],
                    "detail": json.dumps(interval, sort_keys=True),
                }
            )
        for scenario_id, summary in section.get("scenario_summaries", {}).items():
            for metric, interval in summary["metrics"].items():
                rows.append(
                    {
                        "row_type": "confirmation_scenario_aggregate",
                        "record_id": scenario_id,
                        "split": section_name,
                        "scenario": scenario_id,
                        "seed": "",
                        "threshold": section["threshold"],
                        "metric": metric,
                        "value": interval["estimate"],
                        "detail": json.dumps(interval, sort_keys=True),
                    }
                )
        for cell in section["cells"]:
            selected = cell["threshold_rows"][0]
            rows.append(
                {
                    "row_type": "confirmation_cell",
                    "record_id": f"{cell['scenario_id']}-s{cell['base_seed']}",
                    "split": section_name,
                    "scenario": cell["scenario_id"],
                    "seed": cell["base_seed"],
                    "threshold": selected["threshold"],
                    "metric": "gated_failure_rate",
                    "value": selected["gated_failure_rate"],
                    "detail": json.dumps(
                        {
                            key: value
                            for key, value in cell.items()
                            if key != "threshold_rows"
                        }
                        | {"selected_threshold_row": {k: v for k, v in selected.items() if k != "window_rows"}},
                        sort_keys=True,
                    ),
                }
            )
    for gate, passed in report["gates"].items():
        rows.append(
            {
                "row_type": "gate",
                "record_id": gate,
                "split": "governance",
                "scenario": "",
                "seed": "",
                "threshold": "",
                "metric": "passed",
                "value": int(passed),
                "detail": "",
            }
        )
    return rows


def write_report(
    report: Mapping[str, Any],
    *,
    artifact_path: str | Path = DEFAULT_ARTIFACT,
    source_path: str | Path = DEFAULT_SOURCE_DATA,
) -> dict[str, Any]:
    artifact = dict(report)
    rows = source_rows(artifact)
    csv_path = _repo_path(source_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "row_type",
        "record_id",
        "split",
        "scenario",
        "seed",
        "threshold",
        "metric",
        "value",
        "detail",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    artifact["source_data"] = {
        "path": Path(source_path).as_posix(),
        "row_count": len(rows),
        "rows_sha256": _canonical_sha256(rows),
        "csv_sha256": _sha256(csv_path),
    }
    artifact["contract_sha256"] = _canonical_sha256(_contract_view(artifact))
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("artifact validation failed: " + "; ".join(errors))
    path = _repo_path(artifact_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--source-data", type=Path, default=DEFAULT_SOURCE_DATA)
    args = parser.parse_args(argv)
    report = write_report(
        build_report(), artifact_path=args.artifact, source_path=args.source_data
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "gates": report["gate_summary"],
                "selected_threshold": report["calibration"]["selected_threshold"],
                "ood_reduction": report["confirmation_ood"]["metrics"][
                    "absolute_catastrophic_reduction"
                ],
                "source_rows": report["source_data"]["row_count"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CALIBRATION_SEEDS",
    "CALIBRATION_SCENARIOS",
    "CONFIRMATION_OOD_SCENARIOS",
    "CONFIRMATION_SEEDS",
    "DEFAULT_ARTIFACT",
    "DEFAULT_SOURCE_DATA",
    "FALLBACK_METHOD",
    "FallbackValidationConfig",
    "NOMINAL_SCENARIO",
    "PRIMARY_METHOD",
    "SCORE_COMPONENTS",
    "THRESHOLD_GRID",
    "build_report",
    "source_rows",
    "validate_artifact",
    "write_report",
]
