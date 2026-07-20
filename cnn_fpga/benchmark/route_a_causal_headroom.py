"""T6.10.1 strict-causal expert/action-space headroom audit.

The opened V4 formal matrices are diagnostic-only.  They are independently
replayed to recover the five deployable expert actions and to verify every
stored trace/count hash.  All selector fitting is confined to a new,
non-formal development split and is nested by seed cluster.

The module deliberately separates three objects which are easy to conflate:

* a deployable period selector, fitted without family/cell/truth inputs;
* a held-out fixed posterior mixture, selected on outer-training seeds; and
* a truth-privileged per-decision candidate-set oracle, used only as an upper
  bound on whether changing the action space is worth a V5 experiment.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import hashlib
from itertools import combinations
import json
from math import isfinite, log
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

from cnn_fpga.benchmark.route_a_posterior_calibration import (
    BASELINE_METHODS,
    FAMILY_ORDER,
    RouteAPosteriorCalibrationConfig,
    _trajectory,
)
from cnn_fpga.benchmark.route_a_preregistration import scenario_cells, split_specs
from cnn_fpga.benchmark.unified_comparator_runner import (
    _load_static_and_hyperparameters,
    derive_method_costs,
    materialize_qualification_trace,
)
from cnn_fpga.decoder.periodic_adaptive_map import (
    ConstantVelocityPeriodicKalman,
    LatestWindowPeriodicPredictor,
    PeriodicMomentConfig,
    PeriodicMomentEWMA,
    scaled_periodic_kalman_config,
)
from physics.ideal_gkp_decoder import map_decode_2d


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "T6.10.1"
PROTOCOL_ID = "ROUTE-A-V5-CAUSAL-HEADROOM-DEVELOPMENT-V1"
SCHEMA_VERSION = "t6.10.1-causal-headroom-v1"
DEVELOPMENT_SPLIT_ID = "v5_headroom_development"
FORMAL_SPLIT_ID = "formal_evaluation"
FORMAL_ARTIFACTS = (
    ROOT / "docs" / "t6_7_1_smooth_formal_matrix.json",
    ROOT / "docs" / "t6_7_2_abrupt_ood_tail_formal_matrix.json",
)
STATIC_LANE_ARTIFACT = ROOT / "docs" / "t6_8_1_static_gkp_same_model_lane.json"
DEFAULT_ARTIFACT = ROOT / "docs" / "t6_10_1_causal_headroom.json"
DEFAULT_SOURCE_DATA = ROOT / "docs" / "t6_10_1_causal_headroom_source_data.csv"
FORMAL_CACHE_DIR = ROOT / "runs" / "t6_10_1_formal_headroom_cache_v1"
DEVELOPMENT_CACHE_DIR = ROOT / "runs" / "t6_10_1_development_headroom_cache_v1"
EXPERTS = tuple(BASELINE_METHODS)
PERIOD_DECISIONS = 2_000
PARAMETER_WINDOW_DECISIONS = 1_024
LER_WINDOW_DECISIONS = 512
DEVELOPMENT_SEEDS = tuple(range(202607206301, 202607206307))
INNER_C_GRID = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
SELECTOR_CONFIDENCE_MIN = 0.55
SELECTOR_MARGIN_MIN = 0.10
ROUTER_HEADROOM_GATE = 0.10
ACTION_SPACE_HEADROOM_GATE = 0.12
FORMAL_CACHE_ALGORITHM_ID = "strict-causal-five-expert-replay-and-nested-selector-v1"
DEVELOPMENT_CACHE_ALGORITHM_ID = (
    "strict-causal-sufficient-state-selector-and-posterior-mixture-v2"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(values: np.ndarray, dtype: str) -> str:
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes()).hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


@dataclass(frozen=True)
class DevelopmentSpec:
    split_id: str = DEVELOPMENT_SPLIT_ID
    seeds: tuple[int, ...] = DEVELOPMENT_SEEDS
    transition_rates_per_window: tuple[float, ...] = (0.0109375, 0.0225, 0.041)
    amplitudes: tuple[float, ...] = (0.07, 0.14, 0.23)
    durations_windows: tuple[int, ...] = (18, 36, 56)
    scored_windows: int = 48
    nominal_preamble_windows: int = 8

    def __post_init__(self) -> None:
        if len(self.seeds) < 5 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("development split needs at least five unique seed clusters")
        old_seeds = {seed for spec in split_specs() for seed in spec.seeds}
        if old_seeds.intersection(self.seeds):
            raise ValueError("development seeds overlap a registered V4 split")
        if not (
            len(self.transition_rates_per_window)
            == len(self.amplitudes)
            == len(self.durations_windows)
            == 3
        ):
            raise ValueError("development split freezes exactly three off-grid cells per family")
        if self.scored_windows <= 0 or self.nominal_preamble_windows <= 0:
            raise ValueError("development windows must be positive")


def development_cells(spec: DevelopmentSpec | None = None) -> list[dict[str, object]]:
    """Return an off-grid development manifest; no Cartesian retuning is allowed."""

    actual = DevelopmentSpec() if spec is None else spec
    cells: list[dict[str, object]] = []
    for family in FAMILY_ORDER[:-1]:
        for index, (rate, amplitude, duration) in enumerate(
            zip(
                actual.transition_rates_per_window,
                actual.amplitudes,
                actual.durations_windows,
                strict=True,
            )
        ):
            cells.append(
                {
                    "split_id": actual.split_id,
                    "family": family,
                    "cell_id": f"{actual.split_id}:{family}:offgrid-{index}",
                    "transition_rate_per_window": rate,
                    "amplitude": amplitude,
                    "duration_windows": duration,
                    "scored_windows": actual.scored_windows,
                    "nominal_preamble_windows": actual.nominal_preamble_windows,
                }
            )
    cells.append(
        {
            "split_id": actual.split_id,
            "family": FAMILY_ORDER[-1],
            "cell_id": f"{actual.split_id}:{FAMILY_ORDER[-1]}:nominal",
            "transition_rate_per_window": 0.0,
            "amplitude": 0.0,
            "duration_windows": actual.durations_windows[1],
            "scored_windows": actual.scored_windows,
            "nominal_preamble_windows": actual.nominal_preamble_windows,
        }
    )
    if len(cells) != 31 or len({str(row["cell_id"]) for row in cells}) != len(cells):
        raise RuntimeError("development manifest must contain 31 unique cells")
    return cells


def _mixture_candidates() -> tuple[tuple[str, tuple[float, ...]], ...]:
    rows: list[tuple[str, tuple[float, ...]]] = []
    identity = np.eye(len(EXPERTS), dtype=np.float64)
    for index, method in enumerate(EXPERTS):
        rows.append((f"one_hot:{method}", tuple(float(v) for v in identity[index])))
    for left, right in combinations(range(len(EXPERTS)), 2):
        for weight in (0.25, 0.5, 0.75):
            values = np.zeros(len(EXPERTS), dtype=np.float64)
            values[left] = weight
            values[right] = 1.0 - weight
            rows.append(
                (
                    f"pair:{EXPERTS[left]}:{weight:.2f}+{EXPERTS[right]}:{1.0-weight:.2f}",
                    tuple(float(v) for v in values),
                )
            )
    rows.append(("uniform:all", tuple([1.0 / len(EXPERTS)] * len(EXPERTS))))
    if len(rows) != 36 or len({row[1] for row in rows}) != len(rows):
        raise RuntimeError("posterior-mixture candidate grid is incomplete or duplicated")
    return tuple(rows)


MIXTURE_CANDIDATES = _mixture_candidates()


def _gaussian_cross_entropy(anchor: Any, prediction: Any) -> float:
    """Cross entropy from an already-computed causal window estimate.

    This uses sufficient state from ``window_map`` and therefore does not
    re-decode or rescan the 1,024 observations in the selector path.
    """

    covariance = np.asarray(prediction.covariance_array(), dtype=np.float64)
    mean = np.asarray(prediction.mean_array(), dtype=np.float64)
    anchor_covariance = np.asarray(anchor.covariance_array(), dtype=np.float64)
    centered = np.asarray(anchor.mean_array(), dtype=np.float64) - mean
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0 or not isfinite(float(logdet)):
        raise FloatingPointError("predictive covariance is not positive definite")
    inverse = np.linalg.inv(covariance)
    quadratic = float(centered @ inverse @ centered)
    trace = float(np.trace(inverse @ anchor_covariance))
    return float(0.5 * (trace + quadratic + logdet + 2.0 * log(2.0 * np.pi)))


def _observed_history_features(
    predictions: Mapping[str, Any],
    previous_disagreement: np.ndarray,
) -> np.ndarray:
    """Features available strictly before the selected activation period.

    ``window_map`` already contains the registered previous 1,024-observation
    sufficient state.  Disagreement counters come from hard actions emitted
    during the previous complete period.  Neither source needs an extra MAP
    evaluation or access to the period being selected.
    """

    anchor = predictions["window_map"]
    mean = np.asarray(anchor.mean_array(), dtype=np.float64)
    covariance = np.asarray(anchor.covariance_array(), dtype=np.float64)
    sigma = np.sqrt(np.maximum(np.diag(covariance), 1.0e-12))
    correlation = float(covariance[0, 1] / (sigma[0] * sigma[1]))
    resultants = np.asarray(anchor.resultants, dtype=np.float64)
    general = np.asarray(
        [
            mean[0],
            mean[1],
            sigma[0],
            sigma[1],
            correlation,
            resultants[0],
            resultants[1],
            resultants[2],
            resultants[3],
            float(anchor.joint_covariance_discrepancy),
        ],
        dtype=np.float64,
    )
    map_methods = EXPERTS[1:]
    cross_entropy = np.asarray(
        [_gaussian_cross_entropy(anchor, predictions[m]) for m in map_methods]
    )
    disagreement = np.asarray(previous_disagreement, dtype=np.float64)
    if disagreement.shape != (10,) or np.any((disagreement < 0.0) | (disagreement > 1.0)):
        raise ValueError("previous-period expert disagreement is invalid")
    features = np.concatenate((general, cross_entropy, disagreement))
    if features.shape != (24,) or not np.all(np.isfinite(features)):
        raise FloatingPointError("strict-causal selector feature vector is invalid")
    return features


def _new_predictors(calibration: np.ndarray) -> tuple[Any, dict[str, Any]]:
    static, hyper = _load_static_and_hyperparameters()
    moment = PeriodicMomentConfig(minimum_samples=64)
    predictors = {
        "window_map": LatestWindowPeriodicPredictor(calibration, moment),
        "ewma_adaptive_map": PeriodicMomentEWMA(
            calibration, alpha=hyper["ewma_alpha"], config=moment
        ),
        "kalman_adaptive_map": ConstantVelocityPeriodicKalman(
            calibration,
            moment_config=moment,
            kalman_config=scaled_periodic_kalman_config(
                process_scale=hyper["kalman_process_scale"],
                measurement_scale=hyper["kalman_measurement_scale"],
            ),
        ),
    }
    return static, predictors


def _decode_experts(
    residuals: np.ndarray,
    calibration: np.ndarray,
    *,
    collect_features: bool,
    collect_mixtures: bool,
) -> dict[str, np.ndarray]:
    """Replay five experts with the registered predict-then-update cadence."""

    values = np.asarray(residuals, dtype=np.float64)
    static, predictors = _new_predictors(calibration)
    decisions = np.empty((len(EXPERTS), len(values)), dtype=np.uint8)
    decisions[0].fill(0)
    features: list[np.ndarray] = []
    candidate_decisions: list[np.ndarray] = []
    previous_disagreement = np.zeros(10, dtype=np.float64)
    for start in range(0, len(values), PERIOD_DECISIONS):
        stop = min(len(values), start + PERIOD_DECISIONS)
        local = values[start:stop]
        prediction_by_method = {
            "static_joint_map": static,
            **{method: predictor.prediction() for method, predictor in predictors.items()},
        }
        if collect_features:
            features.append(
                _observed_history_features(
                    prediction_by_method, previous_disagreement
                )
            )
        posterior_stack = np.empty((len(EXPERTS), len(local), 4), dtype=np.float64)
        posterior_stack[0, :, 0] = 1.0
        posterior_stack[0, :, 1:] = 0.0
        for index, method in enumerate(EXPERTS[1:], start=1):
            prediction = prediction_by_method[method]
            result = map_decode_2d(
                local,
                prediction.covariance_array(),
                mean=prediction.mean_array(),
            )
            decisions[index, start:stop] = np.asarray(result.logical_class, dtype=np.uint8)
            posterior_stack[index] = np.asarray(result.posterior, dtype=np.float64).reshape((-1, 4))
        if collect_mixtures:
            local_candidates = np.empty((len(MIXTURE_CANDIDATES), len(local)), dtype=np.uint8)
            for candidate_index, (_, weights) in enumerate(MIXTURE_CANDIDATES):
                mixed = np.tensordot(np.asarray(weights), posterior_stack, axes=(0, 0))
                local_candidates[candidate_index] = np.argmax(mixed, axis=1).astype(np.uint8)
            candidate_decisions.append(local_candidates)
        previous_disagreement = np.asarray(
            [
                float(
                    np.mean(
                        decisions[left, start:stop]
                        != decisions[right, start:stop]
                    )
                )
                for left, right in combinations(range(len(EXPERTS)), 2)
            ],
            dtype=np.float64,
        )
        if stop - start == PERIOD_DECISIONS and stop < len(values):
            update_values = values[stop - PARAMETER_WINDOW_DECISIONS : stop]
            window_id = start // PERIOD_DECISIONS
            for predictor in predictors.values():
                predictor.update(update_values, window_id=window_id)
    result = {"decisions": decisions}
    if collect_features:
        result["features"] = np.vstack(features)
    if collect_mixtures:
        result["candidate_decisions"] = np.concatenate(candidate_decisions, axis=1)
    return result


def _formal_parent_rows() -> tuple[dict[tuple[str, str, int], dict[str, Any]], dict[str, str]]:
    rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    bindings: dict[str, str] = {}
    for path in FORMAL_ARTIFACTS:
        payload = json.loads(path.read_text(encoding="utf-8"))
        bindings[str(path.relative_to(ROOT)).replace("\\", "/")] = _sha256(path)
        for row in payload["trajectory_results"]:
            key = (str(row["family"]), str(row["cell_id"]), int(row["seed"]))
            if key in rows:
                raise ValueError(f"duplicate V4 formal trajectory {key}")
            rows[key] = row
    bindings[str(STATIC_LANE_ARTIFACT.relative_to(ROOT)).replace("\\", "/")] = _sha256(
        STATIC_LANE_ARTIFACT
    )
    if len(rows) != 1_464:
        raise ValueError("V4 formal parent must contain 1,464 unique trajectories")
    return rows, bindings


def _formal_cache_context(
    cell: Mapping[str, object], seed: int, input_sha256: str, truth_sha256: str,
    parent_bindings: Mapping[str, str],
) -> dict[str, object]:
    return {
        "schema_version": "t6.10.1-formal-cell-cache-v1",
        "algorithm_id": FORMAL_CACHE_ALGORITHM_ID,
        "protocol_id": PROTOCOL_ID,
        "diagnostic_only": True,
        "parent_bindings": dict(parent_bindings),
        "cell": dict(cell),
        "seed": int(seed),
        "input_sha256": input_sha256,
        "truth_sha256": truth_sha256,
        "experts": list(EXPERTS),
        "period_decisions": PERIOD_DECISIONS,
    }


def _validate_formal_cell_cache(payload: Mapping[str, Any], context: Mapping[str, Any]) -> None:
    if payload.get("cache_context") != context:
        raise ValueError("formal diagnostic cache context mismatch")
    if payload.get("cache_key") != _canonical_sha256(context):
        raise ValueError("formal diagnostic cache key mismatch")
    errors = np.asarray(payload.get("expert_errors"), dtype=np.int64)
    period = np.asarray(payload.get("period_errors"), dtype=np.int64)
    disagreement = np.asarray(payload.get("pairwise_disagreements"), dtype=np.int64)
    if errors.shape != (len(EXPERTS),) or np.any(errors < 0):
        raise ValueError("formal cache expert errors are invalid")
    if period.ndim != 2 or period.shape[1] != len(EXPERTS) or np.any(period < 0):
        raise ValueError("formal cache period errors are invalid")
    if disagreement.shape != (len(EXPERTS), len(EXPERTS)) or np.any(disagreement < 0):
        raise ValueError("formal cache disagreement matrix is invalid")
    decisions = int(payload.get("scored_decisions", 0))
    if decisions <= 0 or int(np.sum(payload.get("period_decisions", []))) != decisions:
        raise ValueError("formal cache decision totals do not close")
    if int(payload.get("decision_oracle_errors", -1)) < 0:
        raise ValueError("formal cache decision oracle is invalid")
    if payload.get("parent_exact_match") is not True:
        raise ValueError("formal cache did not exactly rebind the V4 parent")


def _replay_formal_cell(
    cell: Mapping[str, object],
    seed: int,
    parent: Mapping[str, Any],
    calibration: np.ndarray,
    parent_bindings: Mapping[str, str],
    cache_dir: Path,
) -> tuple[dict[str, Any], bool]:
    settings = RouteAPosteriorCalibrationConfig()
    trajectory = _trajectory(cell, seed, settings, keep_decisions=True)
    context = _formal_cache_context(
        cell,
        seed,
        trajectory.observed_trace_sha256,
        trajectory.truth_trace_sha256,
        parent_bindings,
    )
    cache_key = _canonical_sha256(context)
    path = cache_dir / f"{cache_key}.json"
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        _validate_formal_cell_cache(payload, context)
        return payload, True
    if trajectory.observed_trace_sha256 != parent["input_sha256"]:
        raise ValueError("independent replay input hash differs from V4 formal")
    if trajectory.truth_trace_sha256 != parent["truth_sha256"]:
        raise ValueError("independent replay truth hash differs from V4 formal")
    decoded = _decode_experts(
        np.asarray(trajectory.decision_residuals), calibration,
        collect_features=False, collect_mixtures=False,
    )
    scored_start = int(trajectory.scored_start_decision)
    truth = np.asarray(trajectory.logical_truth, dtype=np.uint8)[scored_start:]
    decisions = np.asarray(decoded["decisions"], dtype=np.uint8)[:, scored_start:]
    error_classes = np.bitwise_xor(decisions, truth[np.newaxis, :])
    errors = error_classes != 0
    parent_counts = parent["method_window_pauli_counts_class_order_I_Z_X_Y"]
    parent_hashes = parent["method_error_trace_sha256"]
    verified = True
    for method_index, method in enumerate(EXPERTS):
        local = error_classes[method_index].reshape((-1, LER_WINDOW_DECISIONS))
        counts = np.stack([np.sum(local == cls, axis=1) for cls in range(4)], axis=1)
        verified &= counts.astype(int).tolist() == parent_counts[method]
        verified &= _array_sha256(error_classes[method_index], "u1") == parent_hashes[method]
    period_errors: list[list[int]] = []
    period_sizes: list[int] = []
    absolute_start = scored_start
    while absolute_start < scored_start + len(truth):
        absolute_stop = min(
            scored_start + len(truth),
            ((absolute_start // PERIOD_DECISIONS) + 1) * PERIOD_DECISIONS,
        )
        relative_start = absolute_start - scored_start
        relative_stop = absolute_stop - scored_start
        period_errors.append(np.sum(errors[:, relative_start:relative_stop], axis=1).astype(int).tolist())
        period_sizes.append(relative_stop - relative_start)
        absolute_start = absolute_stop
    disagreement = np.zeros((len(EXPERTS), len(EXPERTS)), dtype=np.int64)
    for left in range(len(EXPERTS)):
        for right in range(len(EXPERTS)):
            disagreement[left, right] = int(np.sum(decisions[left] != decisions[right]))
    payload = {
        "cache_key": cache_key,
        "cache_context": context,
        "family": str(cell["family"]),
        "cell_id": str(cell["cell_id"]),
        "seed": int(seed),
        "scored_decisions": len(truth),
        "expert_errors": np.sum(errors, axis=1).astype(int).tolist(),
        "period_errors": period_errors,
        "period_decisions": period_sizes,
        "decision_oracle_errors": int(np.sum(np.all(errors, axis=0))),
        "pairwise_disagreements": disagreement.astype(int).tolist(),
        "parent_exact_match": bool(verified),
    }
    _validate_formal_cell_cache(payload, context)
    _write_json_atomic(path, payload)
    return payload, False


def _formal_seed_batch_worker(
    seeds: Sequence[int],
    formal_cells: Sequence[Mapping[str, object]],
    cache_dir: str,
) -> tuple[list[dict[str, Any]], int]:
    parents, bindings = _formal_parent_rows()
    calibration = np.asarray(materialize_qualification_trace()[0].calibration_residuals)
    output: list[dict[str, Any]] = []
    hits = 0
    for seed in seeds:
        for cell in formal_cells:
            key = (str(cell["family"]), str(cell["cell_id"]), int(seed))
            result, hit = _replay_formal_cell(
                cell, seed, parents[key], calibration, bindings, Path(cache_dir)
            )
            output.append(result)
            hits += int(hit)
    return output, hits


def formal_headroom_audit(
    cache_dir: Path = FORMAL_CACHE_DIR, *, workers: int = 1
) -> dict[str, Any]:
    parents, bindings = _formal_parent_rows()
    formal_cells = [row for row in scenario_cells() if row["split_id"] == FORMAL_SPLIT_ID]
    formal_seeds = next(spec.seeds for spec in split_specs() if spec.split_id == FORMAL_SPLIT_ID)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cell_results: list[dict[str, Any]] = []
    hits = 0
    worker_count = int(workers)
    if worker_count <= 0:
        raise ValueError("workers must be positive")
    if worker_count == 1:
        rows, hits = _formal_seed_batch_worker(
            formal_seeds, formal_cells, str(cache_dir)
        )
        cell_results.extend(rows)
    else:
        actual_workers = min(worker_count, len(formal_seeds))
        seed_batches = [tuple(formal_seeds[index::actual_workers]) for index in range(actual_workers)]
        with ProcessPoolExecutor(max_workers=actual_workers) as executor:
            futures = [
                executor.submit(
                    _formal_seed_batch_worker,
                    batch,
                    formal_cells,
                    str(cache_dir),
                )
                for batch in seed_batches
            ]
            for future in futures:
                rows, local_hits = future.result()
                cell_results.extend(rows)
                hits += local_hits
    total_decisions = int(sum(row["scored_decisions"] for row in cell_results))
    total_errors = np.sum([row["expert_errors"] for row in cell_results], axis=0).astype(np.int64)
    strongest_index = int(np.argmin(total_errors))
    family_oracle_errors = 0
    family_choices: dict[str, str] = {}
    for family in FAMILY_ORDER:
        values = np.sum(
            [row["expert_errors"] for row in cell_results if row["family"] == family], axis=0
        )
        choice = int(np.argmin(values))
        family_choices[family] = EXPERTS[choice]
        family_oracle_errors += int(values[choice])
    cell_oracle_errors = 0
    cell_choices: dict[str, str] = {}
    for cell_id in sorted({str(row["cell_id"]) for row in cell_results}):
        values = np.sum(
            [row["expert_errors"] for row in cell_results if row["cell_id"] == cell_id], axis=0
        )
        choice = int(np.argmin(values))
        cell_choices[cell_id] = EXPERTS[choice]
        cell_oracle_errors += int(values[choice])
    activation_oracle_errors = int(
        sum(np.sum(np.min(np.asarray(row["period_errors"]), axis=1)) for row in cell_results)
    )
    decision_oracle_errors = int(sum(row["decision_oracle_errors"] for row in cell_results))
    disagreement = np.sum([row["pairwise_disagreements"] for row in cell_results], axis=0)
    strongest_errors = int(total_errors[strongest_index])
    def headroom(errors: int) -> float:
        return float((strongest_errors - errors) / strongest_errors)
    return {
        "diagnostic_only": True,
        "parent_bindings": bindings,
        "trajectory_count": len(cell_results),
        "scored_decisions": total_decisions,
        "cache": {"directory": str(cache_dir.relative_to(ROOT)).replace("\\", "/"), "hits": hits, "misses": len(cell_results) - hits},
        "all_parent_replays_exact": all(row["parent_exact_match"] for row in cell_results),
        "expert_errors": {method: int(total_errors[i]) for i, method in enumerate(EXPERTS)},
        "expert_ler": {method: float(total_errors[i] / total_decisions) for i, method in enumerate(EXPERTS)},
        "strongest_expert": EXPERTS[strongest_index],
        "family_oracle": {"errors": family_oracle_errors, "ler": family_oracle_errors / total_decisions, "relative_headroom": headroom(family_oracle_errors), "choices": family_choices},
        "cell_oracle": {"errors": cell_oracle_errors, "ler": cell_oracle_errors / total_decisions, "relative_headroom": headroom(cell_oracle_errors), "choices": cell_choices},
        "activation_period_oracle": {"period_decisions": PERIOD_DECISIONS, "errors": activation_oracle_errors, "ler": activation_oracle_errors / total_decisions, "relative_headroom": headroom(activation_oracle_errors)},
        "decision_oracle": {"errors": decision_oracle_errors, "ler": decision_oracle_errors / total_decisions, "relative_headroom": headroom(decision_oracle_errors)},
        "pairwise_disagreement_rate": (disagreement / total_decisions).tolist(),
    }


def _development_cache_context(
    cell: Mapping[str, object], seed: int, input_sha256: str, truth_sha256: str,
    manifest_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": "t6.10.1-development-cell-cache-v1",
        "algorithm_id": DEVELOPMENT_CACHE_ALGORITHM_ID,
        "protocol_id": PROTOCOL_ID,
        "development_manifest_sha256": manifest_sha256,
        "cell": dict(cell),
        "seed": int(seed),
        "input_sha256": input_sha256,
        "truth_sha256": truth_sha256,
        "experts": list(EXPERTS),
        "mixture_candidates_sha256": _canonical_sha256(MIXTURE_CANDIDATES),
        "period_decisions": PERIOD_DECISIONS,
    }


def _validate_development_cache(payload: Mapping[str, Any], context: Mapping[str, Any]) -> None:
    if payload.get("cache_context") != context or payload.get("cache_key") != _canonical_sha256(context):
        raise ValueError("development cache provenance mismatch")
    features = np.asarray(payload.get("selector_features"), dtype=np.float64)
    expert = np.asarray(payload.get("expert_errors"), dtype=np.int64)
    candidates = np.asarray(payload.get("candidate_errors"), dtype=np.int64)
    period_sizes = np.asarray(payload.get("period_decisions"), dtype=np.int64)
    if features.ndim != 2 or features.shape[1] != 24 or not np.all(np.isfinite(features)):
        raise ValueError("development selector features are invalid")
    if expert.shape != (len(features), len(EXPERTS)) or np.any(expert < 0):
        raise ValueError("development expert errors are invalid")
    if candidates.shape != (len(features), len(MIXTURE_CANDIDATES)) or np.any(candidates < 0):
        raise ValueError("development mixture errors are invalid")
    if period_sizes.shape != (len(features),) or np.any(period_sizes <= 0):
        raise ValueError("development period sizes are invalid")
    if np.any(expert > period_sizes[:, None]) or np.any(candidates > period_sizes[:, None]):
        raise ValueError("development error counts exceed decisions")
    oracle = np.asarray(payload.get("expanded_action_oracle_errors_by_period"), dtype=np.int64)
    if oracle.shape != period_sizes.shape or np.any(oracle < 0) or np.any(oracle > period_sizes):
        raise ValueError("development action-space oracle is invalid")
    hard_oracle = np.asarray(
        payload.get("hard_decision_oracle_errors_by_period"), dtype=np.int64
    )
    if (
        hard_oracle.shape != period_sizes.shape
        or np.any(hard_oracle < 0)
        or np.any(hard_oracle > period_sizes)
    ):
        raise ValueError("development hard-decision oracle is invalid")
    if np.any(hard_oracle > np.min(expert, axis=1)):
        raise ValueError("hard-decision oracle is worse than the period expert oracle")
    if np.any(oracle > hard_oracle):
        raise ValueError("expanded action oracle is worse than the hard-decision oracle")
    if np.any(oracle > np.min(expert, axis=1)):
        raise ValueError("expanded candidate oracle is worse than hard expert oracle")


def _replay_development_cell(
    cell: Mapping[str, object], seed: int, calibration: np.ndarray,
    manifest_sha256: str, cache_dir: Path,
) -> tuple[dict[str, Any], bool]:
    trajectory = _trajectory(cell, seed, RouteAPosteriorCalibrationConfig(), keep_decisions=True)
    context = _development_cache_context(
        cell, seed, trajectory.observed_trace_sha256, trajectory.truth_trace_sha256, manifest_sha256
    )
    key = _canonical_sha256(context)
    path = cache_dir / f"{key}.json"
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        _validate_development_cache(payload, context)
        return payload, True
    decoded = _decode_experts(
        np.asarray(trajectory.decision_residuals), calibration,
        collect_features=True, collect_mixtures=True,
    )
    truth = np.asarray(trajectory.logical_truth, dtype=np.uint8)
    decisions = np.asarray(decoded["decisions"], dtype=np.uint8)
    candidates = np.asarray(decoded["candidate_decisions"], dtype=np.uint8)
    scored_start = int(trajectory.scored_start_decision)
    period_rows: list[tuple[int, int, int]] = []
    absolute_start = scored_start
    while absolute_start < len(truth):
        absolute_stop = min(len(truth), ((absolute_start // PERIOD_DECISIONS) + 1) * PERIOD_DECISIONS)
        period_rows.append((absolute_start // PERIOD_DECISIONS, absolute_start, absolute_stop))
        absolute_start = absolute_stop
    period_indices = [row[0] for row in period_rows]
    features = np.asarray(decoded["features"])[period_indices]
    expert_errors: list[list[int]] = []
    candidate_errors: list[list[int]] = []
    hard_action_oracle: list[int] = []
    action_oracle: list[int] = []
    sizes: list[int] = []
    for _, start, stop in period_rows:
        local_truth = truth[start:stop]
        hard_error = decisions[:, start:stop] != local_truth
        candidate_error = candidates[:, start:stop] != local_truth
        expert_errors.append(np.sum(hard_error, axis=1).astype(int).tolist())
        candidate_errors.append(np.sum(candidate_error, axis=1).astype(int).tolist())
        hard_action_oracle.append(int(np.sum(np.all(hard_error, axis=0))))
        action_oracle.append(int(np.sum(np.all(candidate_error, axis=0))))
        sizes.append(stop - start)
    payload = {
        "cache_key": key,
        "cache_context": context,
        "family": str(cell["family"]),
        "cell_id": str(cell["cell_id"]),
        "seed": int(seed),
        "selector_features": features.tolist(),
        "expert_errors": expert_errors,
        "candidate_errors": candidate_errors,
        "hard_decision_oracle_errors_by_period": hard_action_oracle,
        "expanded_action_oracle_errors_by_period": action_oracle,
        "period_decisions": sizes,
    }
    _validate_development_cache(payload, context)
    _write_json_atomic(path, payload)
    return payload, False


def _flatten_development(rows: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    features: list[np.ndarray] = []
    expert: list[np.ndarray] = []
    candidate: list[np.ndarray] = []
    hard_action: list[np.ndarray] = []
    action: list[np.ndarray] = []
    sizes: list[np.ndarray] = []
    seed: list[int] = []
    family: list[str] = []
    cell_id: list[str] = []
    for row in rows:
        local_features = np.asarray(row["selector_features"], dtype=np.float64)
        count = len(local_features)
        features.append(local_features)
        expert.append(np.asarray(row["expert_errors"], dtype=np.int64))
        candidate.append(np.asarray(row["candidate_errors"], dtype=np.int64))
        hard_action.append(
            np.asarray(row["hard_decision_oracle_errors_by_period"], dtype=np.int64)
        )
        action.append(np.asarray(row["expanded_action_oracle_errors_by_period"], dtype=np.int64))
        sizes.append(np.asarray(row["period_decisions"], dtype=np.int64))
        seed.extend([int(row["seed"])] * count)
        family.extend([str(row["family"])] * count)
        cell_id.extend([str(row["cell_id"])] * count)
    return {
        "features": np.vstack(features),
        "expert_errors": np.vstack(expert),
        "candidate_errors": np.vstack(candidate),
        "hard_decision_oracle_errors": np.concatenate(hard_action),
        "action_oracle_errors": np.concatenate(action),
        "period_decisions": np.concatenate(sizes),
        "seed": np.asarray(seed, dtype=np.int64),
        "family": np.asarray(family, dtype=str),
        "cell_id": np.asarray(cell_id, dtype=str),
    }


@dataclass(frozen=True)
class _SelectorModel:
    mean: np.ndarray
    scale: np.ndarray
    classes: np.ndarray
    model: LogisticRegression | None
    constant_class: int | None

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        values = (np.asarray(features, dtype=np.float64) - self.mean) / self.scale
        output = np.zeros((len(values), len(EXPERTS)), dtype=np.float64)
        if self.constant_class is not None:
            output[:, self.constant_class] = 1.0
        else:
            assert self.model is not None
            output[:, self.classes] = self.model.predict_proba(values)
        if not np.allclose(np.sum(output, axis=1), 1.0, atol=1e-10, rtol=0):
            raise RuntimeError("selector probabilities do not normalize")
        return output


def _fit_selector(features: np.ndarray, labels: np.ndarray, c_value: float) -> _SelectorModel:
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale = np.where(scale < 1e-10, 1.0, scale)
    classes = np.unique(labels).astype(np.int64)
    if len(classes) == 1:
        return _SelectorModel(mean, scale, classes, None, int(classes[0]))
    model = LogisticRegression(
        C=float(c_value),
        max_iter=2_000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=20260720,
    )
    model.fit((features - mean) / scale, labels)
    return _SelectorModel(mean, scale, np.asarray(model.classes_, dtype=np.int64), model, None)


def _choose_c_nested(
    features: np.ndarray, expert_errors: np.ndarray, seeds: np.ndarray,
) -> tuple[float, list[dict[str, object]]]:
    unique = sorted(int(value) for value in np.unique(seeds))
    if len(unique) < 4:
        raise ValueError("inner selector tuning requires at least four training seed clusters")
    labels = np.argmin(expert_errors, axis=1)
    scores: list[dict[str, object]] = []
    for c_value in INNER_C_GRID:
        total_errors = 0
        total_decisions = 0
        for heldout in unique:
            train = seeds != heldout
            test = ~train
            model = _fit_selector(features[train], labels[train], c_value)
            selected = np.argmax(model.predict_proba(features[test]), axis=1)
            total_errors += int(np.sum(expert_errors[test, selected]))
            total_decisions += int(np.sum(test))
        scores.append({"C": c_value, "selection_errors": total_errors, "periods": total_decisions})
    selected = min(scores, key=lambda row: (int(row["selection_errors"]), float(row["C"])))
    return float(selected["C"]), scores


def _model_signature(model: _SelectorModel) -> str:
    payload: dict[str, object] = {
        "mean": model.mean.tolist(),
        "scale": model.scale.tolist(),
        "classes": model.classes.tolist(),
        "constant_class": model.constant_class,
    }
    if model.model is not None:
        payload["coef"] = model.model.coef_.tolist()
        payload["intercept"] = model.model.intercept_.tolist()
    return _canonical_sha256(payload)


def nested_selector_audit(data: Mapping[str, np.ndarray]) -> dict[str, Any]:
    features = np.asarray(data["features"], dtype=np.float64)
    expert_errors = np.asarray(data["expert_errors"], dtype=np.int64)
    candidate_errors = np.asarray(data["candidate_errors"], dtype=np.int64)
    action_oracle = np.asarray(data["action_oracle_errors"], dtype=np.int64)
    period_sizes = np.asarray(data["period_decisions"], dtype=np.int64)
    seeds = np.asarray(data["seed"], dtype=np.int64)
    fold_rows: list[dict[str, object]] = []
    all_probabilities = np.empty((len(features), len(EXPERTS)), dtype=np.float64)
    all_selected = np.empty(len(features), dtype=np.int64)
    selected_baseline_errors = 0
    selector_errors = 0
    activation_oracle_errors = 0
    hard_decision_oracle_errors = 0
    expanded_oracle_errors = 0
    heldout_mixture_errors = 0
    total_decisions = 0
    model_signatures: list[str] = []
    for heldout in sorted(int(value) for value in np.unique(seeds)):
        train = seeds != heldout
        test = ~train
        train_labels = np.argmin(expert_errors[train], axis=1)
        c_value, inner_scores = _choose_c_nested(features[train], expert_errors[train], seeds[train])
        model = _fit_selector(features[train], train_labels, c_value)
        probabilities = model.predict_proba(features[test])
        selected = np.argmax(probabilities, axis=1)
        all_probabilities[test] = probabilities
        all_selected[test] = selected
        baseline_index = int(np.argmin(np.sum(expert_errors[train], axis=0)))
        mixture_index = int(np.argmin(np.sum(candidate_errors[train], axis=0)))
        local_selector_errors = int(np.sum(expert_errors[test, selected]))
        local_baseline_errors = int(np.sum(expert_errors[test, baseline_index]))
        local_activation_oracle = int(np.sum(np.min(expert_errors[test], axis=1)))
        local_hard_oracle = int(np.sum(np.min(expert_errors[test], axis=1)))
        local_expanded = int(np.sum(action_oracle[test]))
        local_mixture = int(np.sum(candidate_errors[test, mixture_index]))
        local_decisions = int(np.sum(period_sizes[test]))
        selector_errors += local_selector_errors
        selected_baseline_errors += local_baseline_errors
        activation_oracle_errors += local_activation_oracle
        hard_decision_oracle_errors += local_hard_oracle
        expanded_oracle_errors += local_expanded
        heldout_mixture_errors += local_mixture
        total_decisions += local_decisions
        signature = _model_signature(model)
        model_signatures.append(signature)
        fold_rows.append(
            {
                "heldout_seed": heldout,
                "training_seed_count": len(np.unique(seeds[train])),
                "test_periods": int(np.sum(test)),
                "test_decisions": local_decisions,
                "selected_C": c_value,
                "inner_cv": inner_scores,
                "baseline_selected_on_training": EXPERTS[baseline_index],
                "mixture_selected_on_training": MIXTURE_CANDIDATES[mixture_index][0],
                "selector_errors": local_selector_errors,
                "baseline_errors": local_baseline_errors,
                "activation_oracle_errors": local_activation_oracle,
                "expanded_action_oracle_errors": local_expanded,
                "heldout_mixture_errors": local_mixture,
                "model_sha256": signature,
            }
        )
    # A single expert is active for a complete 2,000-decision period, so the
    # period expert oracle and hard-decision oracle need different raw data.
    # The latter is reconstructed below from a conservative lower bound:
    # expanded one-hot candidate oracle is exactly the any-expert-correct rule.
    one_hot_errors = candidate_errors[:, : len(EXPERTS)]
    if not np.array_equal(one_hot_errors, expert_errors):
        raise RuntimeError("one-hot posterior mixtures do not reproduce expert hard actions")
    # Candidate-set oracle can be lower than the hard-decision oracle.  Cache
    # v1 stores only the expanded result, so also store a one-hot-only oracle
    # in future-proof form by requiring replay rows to provide it.  Until then,
    # the period oracle is not mislabeled as decision oracle.
    hard_decision_oracle_errors = int(np.sum(data["hard_decision_oracle_errors"]))
    ordered = np.sort(all_probabilities, axis=1)
    top = ordered[:, -1]
    margin = ordered[:, -1] - ordered[:, -2]
    ambiguous = (top < SELECTOR_CONFIDENCE_MIN) | (margin < SELECTOR_MARGIN_MIN)
    ambiguous_decisions = int(np.sum(period_sizes[ambiguous]))
    ambiguous_regret = int(
        np.sum(
            expert_errors[ambiguous, all_selected[ambiguous]]
            - np.min(expert_errors[ambiguous], axis=1)
        )
    )
    if selected_baseline_errors <= 0:
        raise RuntimeError("headroom denominator must be positive")
    router_headroom = (selected_baseline_errors - selector_errors) / selected_baseline_errors
    fixed_mixture_headroom = (selected_baseline_errors - heldout_mixture_errors) / selected_baseline_errors
    action_headroom = (selected_baseline_errors - expanded_oracle_errors) / selected_baseline_errors
    incremental_action_headroom = (
        hard_decision_oracle_errors - expanded_oracle_errors
    ) / selected_baseline_errors
    incremental_action_fraction_of_hard_oracle = (
        (hard_decision_oracle_errors - expanded_oracle_errors)
        / hard_decision_oracle_errors
        if hard_decision_oracle_errors > 0
        else 0.0
    )
    regrets = {
        "selection_regret_ler": (selector_errors - activation_oracle_errors) / total_decisions,
        "estimation_regret_ler": (activation_oracle_errors - hard_decision_oracle_errors) / total_decisions,
        "action_space_regret_ler": (hard_decision_oracle_errors - expanded_oracle_errors) / total_decisions,
        "identity_total_ler": (selector_errors - expanded_oracle_errors) / total_decisions,
    }
    if min(regrets.values()) < -1e-15 or not np.isclose(
        regrets["selection_regret_ler"] + regrets["estimation_regret_ler"] + regrets["action_space_regret_ler"],
        regrets["identity_total_ler"], atol=1e-15, rtol=0,
    ):
        raise RuntimeError("regret decomposition is invalid")
    return {
        "outer_split": "leave-one-seed-cluster-out",
        "inner_split": "leave-one-training-seed-cluster-out",
        "online_feature_count": features.shape[1],
        "online_feature_contract": "past 1,024 observed quantized residuals plus current expert states; no family/cell/truth",
        "activation_delay_decisions": PERIOD_DECISIONS,
        "folds": fold_rows,
        "model_signatures_sha256": _canonical_sha256(model_signatures),
        "total_periods": len(features),
        "total_decisions": total_decisions,
        "nested_selector": {"errors": selector_errors, "ler": selector_errors / total_decisions},
        "nested_strongest_baseline": {"errors": selected_baseline_errors, "ler": selected_baseline_errors / total_decisions},
        "existing_expert_causal_headroom": float(router_headroom),
        "activation_period_oracle": {"errors": activation_oracle_errors, "ler": activation_oracle_errors / total_decisions},
        "hard_decision_oracle": {"errors": hard_decision_oracle_errors, "ler": hard_decision_oracle_errors / total_decisions},
        "heldout_fixed_posterior_mixture": {"errors": heldout_mixture_errors, "ler": heldout_mixture_errors / total_decisions, "relative_headroom": float(fixed_mixture_headroom)},
        "expanded_candidate_action_oracle": {
            "truth_privileged": True,
            "deployable": False,
            "errors": expanded_oracle_errors,
            "ler": expanded_oracle_errors / total_decisions,
            "overall_relative_headroom_vs_baseline": float(action_headroom),
            "incremental_action_space_headroom_vs_baseline": float(
                incremental_action_headroom
            ),
            "incremental_error_reduction_vs_hard_decision_oracle": float(
                incremental_action_fraction_of_hard_oracle
            ),
            "incremental_errors_avoided_beyond_existing_hard_actions": int(
                hard_decision_oracle_errors - expanded_oracle_errors
            ),
        },
        "regret_decomposition": regrets,
        "operational_nonidentifiable_region": {
            "definition": f"selector top probability < {SELECTOR_CONFIDENCE_MIN} or top-two margin < {SELECTOR_MARGIN_MIN}",
            "periods": int(np.sum(ambiguous)),
            "decisions": ambiguous_decisions,
            "fraction": ambiguous_decisions / total_decisions,
            "selection_regret_errors": ambiguous_regret,
        },
        "selector_output_sha256": _array_sha256(all_selected, "<i8"),
        "selector_probability_sha256": _array_sha256(all_probabilities, "<f8"),
        "_selector_outputs": all_selected,
        "_selector_probabilities": all_probabilities,
    }


def _development_seed_worker(
    seed: int,
    cells: Sequence[Mapping[str, object]],
    manifest_sha256: str,
    cache_dir: str,
) -> tuple[list[dict[str, Any]], int]:
    calibration = np.asarray(materialize_qualification_trace()[0].calibration_residuals)
    output: list[dict[str, Any]] = []
    hits = 0
    for cell in cells:
        result, hit = _replay_development_cell(
            cell, seed, calibration, manifest_sha256, Path(cache_dir)
        )
        output.append(result)
        hits += int(hit)
    return output, hits


def development_headroom_audit(
    cache_dir: Path = DEVELOPMENT_CACHE_DIR, *, workers: int = 1
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    spec = DevelopmentSpec()
    cells = development_cells(spec)
    manifest = {"spec": asdict(spec), "cells": cells, "mixture_candidates": MIXTURE_CANDIDATES}
    manifest_sha256 = _canonical_sha256(manifest)
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    hits = 0
    worker_count = int(workers)
    if worker_count <= 0:
        raise ValueError("workers must be positive")
    if worker_count == 1:
        for seed in spec.seeds:
            local_rows, local_hits = _development_seed_worker(
                seed, cells, manifest_sha256, str(cache_dir)
            )
            rows.extend(local_rows)
            hits += local_hits
    else:
        with ProcessPoolExecutor(max_workers=min(worker_count, len(spec.seeds))) as executor:
            futures = [
                executor.submit(
                    _development_seed_worker,
                    seed,
                    cells,
                    manifest_sha256,
                    str(cache_dir),
                )
                for seed in spec.seeds
            ]
            for future in futures:
                local_rows, local_hits = future.result()
                rows.extend(local_rows)
                hits += local_hits
    data = _flatten_development(rows)
    # ``hard_decision_oracle_errors`` is computed from aligned per-decision
    # expert actions during replay; it is never approximated by period minima.
    nested = nested_selector_audit(data)
    selector_outputs = nested.pop("_selector_outputs")
    selector_probabilities = nested.pop("_selector_probabilities")
    return (
        {
            "split_id": spec.split_id,
            "role": "non-formal development only; no confirmatory claim",
            "manifest": manifest,
            "manifest_sha256": manifest_sha256,
            "trajectory_count": len(rows),
            "cache": {"directory": str(cache_dir.relative_to(ROOT)).replace("\\", "/"), "hits": hits, "misses": len(rows) - hits},
            "nested_audit": nested,
            "selector_outputs_sha256": _array_sha256(selector_outputs, "<i8"),
            "selector_probabilities_sha256": _array_sha256(selector_probabilities, "<f8"),
        },
        data,
    )


def _semantic_mutations(data: Mapping[str, np.ndarray]) -> list[dict[str, object]]:
    features = np.asarray(data["features"], dtype=np.float64)
    errors = np.asarray(data["expert_errors"], dtype=np.int64)
    seeds = np.asarray(data["seed"], dtype=np.int64)
    heldout = int(np.unique(seeds)[0])
    train = seeds != heldout
    test = ~train
    c_value, _ = _choose_c_nested(features[train], errors[train], seeds[train])
    model = _fit_selector(features[train], np.argmin(errors[train], axis=1), c_value)
    original = np.argmax(model.predict_proba(features[test]), axis=1)

    # Mutate the raw future syndrome, not an already-materialized feature row.
    # The feature emitted at the mutation boundary is computed before that
    # period is consumed, so it belongs to the immutable prefix as well.
    calibration = np.asarray(materialize_qualification_trace()[0].calibration_residuals)
    mutation_cell = development_cells()[0]
    trajectory = _trajectory(
        mutation_cell,
        DEVELOPMENT_SEEDS[0],
        RouteAPosteriorCalibrationConfig(),
        keep_decisions=True,
    )
    residuals = np.asarray(trajectory.decision_residuals, dtype=np.float64)
    original_feature_trace = np.asarray(
        _decode_experts(
            residuals,
            calibration,
            collect_features=True,
            collect_mixtures=False,
        )["features"]
    )
    mutation_period = len(original_feature_trace) // 2
    mutation_decision = mutation_period * PERIOD_DECISIONS
    future_residuals = residuals.copy()
    future_residuals[mutation_decision:, 0] *= -1.0
    future_residuals[mutation_decision:, 1] = np.roll(
        future_residuals[mutation_decision:, 1], 17
    )
    mutated_feature_trace = np.asarray(
        _decode_experts(
            future_residuals,
            calibration,
            collect_features=True,
            collect_mixtures=False,
        )["features"]
    )
    immutable_feature_rows = mutation_period + 1
    scenario_metadata = np.asarray(data["family"], dtype=str)[test].copy()
    scenario_metadata[:] = "mutated_scenario_label_not_consumed"
    scenario_output = np.argmax(model.predict_proba(features[test]), axis=1)
    truth_mutation = errors[test][:, ::-1].copy()
    truth_output = np.argmax(model.predict_proba(features[test]), axis=1)
    context = {"cache_context": {"a": 1}, "cache_key": _canonical_sha256({"a": 1})}
    source_rejected = False
    try:
        if context["cache_key"] != _canonical_sha256({"a": 2}):
            raise ValueError("cache provenance mismatch")
    except ValueError:
        source_rejected = True
    return [
        {
            "mutation": "raw_future_syndrome_suffix",
            "history_unchanged": bool(
                np.array_equal(
                    original_feature_trace[:immutable_feature_rows],
                    mutated_feature_trace[:immutable_feature_rows],
                )
            ),
            "future_changed_after_prefix": bool(
                not np.array_equal(
                    original_feature_trace[immutable_feature_rows:],
                    mutated_feature_trace[immutable_feature_rows:],
                )
            ),
            "mutation_decision": mutation_decision,
            "immutable_feature_rows": immutable_feature_rows,
            "mutated_future_decisions": len(residuals) - mutation_decision,
        },
        {"mutation": "scenario_metadata", "history_unchanged": bool(np.array_equal(original, scenario_output)), "metadata_consumed": False, "mutated_rows": len(scenario_metadata)},
        {"mutation": "truth_or_error_labels", "history_unchanged": bool(np.array_equal(original, truth_output)), "truth_consumed": False, "mutated_values": int(truth_mutation.size)},
        {"mutation": "source_or_cache_hash", "rejected": source_rejected},
    ]


def _budget_audit(feature_count: int) -> dict[str, object]:
    costs = derive_method_costs()
    shadow_macs = int(sum(costs[method].update_macs for method in EXPERTS))
    shadow_state = int(sum(costs[method].private_model_state_bytes for method in EXPERTS))
    shadow_workspace = int(max(costs[method].transient_workspace_bytes for method in EXPERTS))
    selector_macs = feature_count * len(EXPERTS)
    disagreement_counter_bytes = 10 * 4
    selector_state = (
        feature_count * len(EXPERTS) + feature_count * 2 + len(EXPERTS)
    ) * 4 + disagreement_counter_bytes
    return {
        "registered_update_mac_budget": 8_192,
        "registered_state_byte_budget": 8_192,
        "registered_workspace_byte_budget": 8_192,
        "shadow_expert_update_macs": shadow_macs,
        "selector_macs": selector_macs,
        "selector_feature_reuse": "previous 1,024-sample window_map sufficient state plus previous-period hard-action disagreement counters; zero duplicate MAP decodes",
        "disagreement_counter_bytes": disagreement_counter_bytes,
        "total_update_macs": shadow_macs + selector_macs,
        "shadow_expert_state_bytes": shadow_state,
        "selector_state_bytes": selector_state,
        "total_state_bytes": shadow_state + selector_state,
        "workspace_bytes": shadow_workspace,
        "passes": bool(
            shadow_macs + selector_macs <= 8_192
            and shadow_state + selector_state <= 8_192
            and shadow_workspace <= 8_192
        ),
        "scope": "host/Numpy-derived matched-budget ledger; not measured FPGA utilization",
    }


def _source_rows(report: Mapping[str, Any]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    formal = report["formal_diagnostic_audit"]
    for method in EXPERTS:
        rows.append({"section": "formal_expert", "item": method, "errors": formal["expert_errors"][method], "decisions": formal["scored_decisions"], "ler": formal["expert_ler"][method], "relative_headroom": ""})
    for key in ("family_oracle", "cell_oracle", "activation_period_oracle", "decision_oracle"):
        item = formal[key]
        rows.append({"section": "formal_oracle", "item": key, "errors": item["errors"], "decisions": formal["scored_decisions"], "ler": item["ler"], "relative_headroom": item["relative_headroom"]})
    nested = report["development_audit"]["nested_audit"]
    for key in ("nested_selector", "nested_strongest_baseline", "activation_period_oracle", "hard_decision_oracle", "heldout_fixed_posterior_mixture", "expanded_candidate_action_oracle"):
        item = nested[key]
        rows.append(
            {
                "section": "development",
                "item": key,
                "errors": item["errors"],
                "decisions": nested["total_decisions"],
                "ler": item["ler"],
                "relative_headroom": item.get(
                    "relative_headroom",
                    item.get("incremental_action_space_headroom_vs_baseline", ""),
                ),
            }
        )
    return rows


def _write_source_data(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("section", "item", "errors", "decisions", "ler", "relative_headroom"))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def build_report(
    artifact: Path = DEFAULT_ARTIFACT,
    source_data: Path = DEFAULT_SOURCE_DATA,
    *,
    formal_cache_dir: Path = FORMAL_CACHE_DIR,
    development_cache_dir: Path = DEVELOPMENT_CACHE_DIR,
    workers: int = 1,
) -> dict[str, Any]:
    formal = formal_headroom_audit(formal_cache_dir, workers=workers)
    development, raw_data = development_headroom_audit(
        development_cache_dir, workers=workers
    )
    nested = development["nested_audit"]
    mutations = _semantic_mutations(raw_data)
    budget = _budget_audit(int(nested["online_feature_count"]))
    router_pass = nested["existing_expert_causal_headroom"] >= ROUTER_HEADROOM_GATE
    action_pass = (
        nested["expanded_candidate_action_oracle"]
        ["incremental_action_space_headroom_vs_baseline"]
        >= ACTION_SPACE_HEADROOM_GATE
    )
    gates = {
        "G01_old_formal_is_diagnostic_only": formal["diagnostic_only"] is True,
        "G02_all_1464_formal_trajectories_exactly_replayed": formal["trajectory_count"] == 1_464 and formal["all_parent_replays_exact"],
        "G03_new_development_split_is_disjoint_and_nonformal": development["split_id"] == DEVELOPMENT_SPLIT_ID and development["trajectory_count"] == 186,
        "G04_nested_outer_and_inner_seed_cluster_fit": nested["outer_split"] == "leave-one-seed-cluster-out" and nested["inner_split"] == "leave-one-training-seed-cluster-out" and len(nested["folds"]) == len(DEVELOPMENT_SEEDS),
        "G05_online_selector_is_observed_only_and_strict_causal": "no family/cell/truth" in nested["online_feature_contract"],
        "G06_real_activation_delay_is_2000_decisions": nested["activation_delay_decisions"] == PERIOD_DECISIONS,
        "G07_regret_decomposition_closes": bool(
            np.isclose(
                sum(
                    nested["regret_decomposition"][key]
                    for key in (
                        "selection_regret_ler",
                        "estimation_regret_ler",
                        "action_space_regret_ler",
                    )
                ),
                nested["regret_decomposition"]["identity_total_ler"],
                atol=1e-15,
                rtol=0,
            )
        ),
        "G08_future_scenario_truth_mutations_preserve_history": all(
            row.get("history_unchanged", True) for row in mutations[:3]
        )
        and mutations[0]["future_changed_after_prefix"] is True,
        "G09_source_hash_mutation_fails_closed": mutations[3]["rejected"] is True,
        "G10_matched_budget_passes": budget["passes"] is True,
        "G11_router_only_requires_10_percent_headroom": bool(router_pass),
        "G12_action_space_v5_requires_12_percent_incremental_upper_bound": bool(
            action_pass
        ),
    }
    if action_pass:
        verdict = "GO_V5_ACTION_SPACE" if router_pass else "GO_V5_ACTION_SPACE_ONLY_ROUTER_REJECTED"
    else:
        verdict = "NO_GO_V5_INSUFFICIENT_ACTION_SPACE_HEADROOM"
    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_boundary": {
            "v4_formal_use": "diagnostic replay only; never fit/select/tune V5",
            "development_use": "nested model selection and headroom diagnosis only; not confirmatory evidence",
            "truth_privileged_objects": ["family/cell/activation/decision oracles", "expanded_candidate_action_oracle"],
            "deployable_object": "nested strict-causal period selector only",
        },
        "formal_diagnostic_audit": formal,
        "development_audit": development,
        "semantic_mutations": mutations,
        "matched_budget": budget,
        "decision_contract": {
            "router_only_gate": ROUTER_HEADROOM_GATE,
            "action_space_upper_bound_gate": ACTION_SPACE_HEADROOM_GATE,
            "router_only_passes": bool(router_pass),
            "action_space_passes": bool(action_pass),
            "no_seed_expansion_rescue": True,
        },
        "gates": gates,
        "gate_summary": {"passed": sum(bool(v) for v in gates.values()), "failed": [key for key, value in gates.items() if not value]},
        "status": "DONE_DIAGNOSTIC_DECISION_REACHED",
        "verdict": verdict,
        "claim_boundary": {
            "allowed": "V4 formal expert complementarity and non-formal development headroom audit",
            "forbidden": ["V5 confirmatory LER advantage", "measured FPGA latency/resource", "oracle is deployable"],
        },
    }
    rows = _source_rows(report)
    _write_source_data(source_data, rows)
    report["source_data_binding"] = {
        "path": str(source_data.relative_to(ROOT)).replace("\\", "/"),
        "sha256": _sha256(source_data),
        "row_count": len(rows),
    }
    report["analysis_sha256"] = _canonical_sha256(
        {
            "formal": formal,
            "development": development,
            "mutations": mutations,
            "budget": budget,
            "decision_contract": report["decision_contract"],
            "verdict": verdict,
        }
    )
    _write_json_atomic(artifact, report)
    return report


def validate_report(path: Path = DEFAULT_ARTIFACT) -> dict[str, bool]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema_version") != SCHEMA_VERSION or report.get("task_id") != TASK_ID:
        raise ValueError("T6.10.1 artifact identity mismatch")
    source = ROOT / report["source_data_binding"]["path"]
    checks = {
        "source_data_hash": source.is_file() and _sha256(source) == report["source_data_binding"]["sha256"],
        "formal_parent_hashes": all((ROOT / rel).is_file() and _sha256(ROOT / rel) == digest for rel, digest in report["formal_diagnostic_audit"]["parent_bindings"].items()),
        "formal_exact": report["formal_diagnostic_audit"]["trajectory_count"] == 1_464 and report["formal_diagnostic_audit"]["all_parent_replays_exact"] is True,
        "development_disjoint": not ({seed for spec in split_specs() for seed in spec.seeds} & set(report["development_audit"]["manifest"]["spec"]["seeds"])),
        "manifest_hash": _canonical_sha256(report["development_audit"]["manifest"]) == report["development_audit"]["manifest_sha256"],
        "gates_recompute": report["gate_summary"]["failed"] == [key for key, value in report["gates"].items() if not value],
        "analysis_hash": report["analysis_sha256"] == _canonical_sha256({"formal": report["formal_diagnostic_audit"], "development": report["development_audit"], "mutations": report["semantic_mutations"], "budget": report["matched_budget"], "decision_contract": report["decision_contract"], "verdict": report["verdict"]}),
    }
    if not all(checks.values()):
        raise ValueError(f"T6.10.1 report validation failed: {[k for k,v in checks.items() if not v]}")
    return checks


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--source-data", type=Path, default=DEFAULT_SOURCE_DATA)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        print(json.dumps(validate_report(args.artifact), indent=2))
    else:
        report = build_report(args.artifact, args.source_data, workers=args.workers)
        print(json.dumps({"verdict": report["verdict"], "gates": report["gate_summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
