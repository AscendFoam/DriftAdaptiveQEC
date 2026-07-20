"""T6.7.2 untouched abrupt/OOD, tail-safety and nominal formal matrix.

The runner consumes the same immutable V4 posterior/router lock as T6.7.1 and
the previously untouched formal abrupt/OOD plus nominal cells.  All decisions
are produced from observed, quantized syndrome information.  Simulator regime
labels and logical outcomes are used only after the online trace closes, for
false-update, lag and logical-error evaluation.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Mapping, Sequence

import numpy as np

from cnn_fpga.benchmark.route_a_posterior_calibration import (
    CLASS_TO_INDEX,
    DUAL_BANK_EXPERTS,
    PREQUENTIAL_SCORE_MEMORY,
    SMOOTH_BANK_POSTERIOR_MIN,
    RouteAPosteriorCalibrationConfig,
    _causal_gaussian_predictive_score,
    _load_static_and_hyperparameters,
    _prediction_classes,
    _selected_action_trace,
    _trajectory,
)
from cnn_fpga.benchmark.route_a_preregistration import (
    ABRUPT_OOD_FAMILIES,
    DEFAULT_ARTIFACT as PREREG_ARTIFACT,
    NOMINAL_FAMILY,
    protocol_payload,
    scenario_cells,
    split_specs,
    validate_protocol,
)
from cnn_fpga.benchmark.route_a_smooth_formal import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    DEFAULT_LOCK,
    LER_WINDOW_DECISIONS,
    PARAMETER_PERIOD_DECISIONS,
    PARAMETER_WINDOW_DECISIONS,
    POSTERIOR_WINDOW_DECISIONS,
    PRIMARY_BASELINE,
    ROOT,
    _array_sha256,
    _bootstrap_mean,
    _json_sha256,
    _load_models,
    _load_parents,
    _sha256,
    _window_class_counts,
)
from cnn_fpga.benchmark.unified_comparator_runner import (
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
from cnn_fpga.decoder.route_a_regime_posterior import temperature_scale
from physics.ideal_gkp_decoder import map_decode_2d


TASK_ID = "T6.7.2"
PROTOCOL_ID = "ROUTE-A-TAIL-FORMAL-V1"
SCHEMA_VERSION = "t6.7.2-route-a-tail-formal-v1"
DEFAULT_ARTIFACT = ROOT / "docs" / "t6_7_2_abrupt_ood_tail_formal_matrix.json"
DEFAULT_SOURCE_DATA = ROOT / "docs" / "t6_7_2_abrupt_ood_tail_formal_matrix_source_data.csv"
DEFAULT_ACCESS_LEDGER = ROOT / "runs" / "t6_7_2_formal_access_ledger.json"
DEFAULT_CACHE_DIR = ROOT / "runs" / "t6_7_2_tail_formal_cache_v1"

METHODS = (
    "standard_binning",
    "static_joint_map",
    "window_map",
    "ewma_adaptive_map",
    "kalman_adaptive_map",
    "proposed_route_a",
)
TAIL_CLASS_INDICES = (CLASS_TO_INDEX["calibration_shift"], CLASS_TO_INDEX["burst"])
CALIBRATION_FAMILY = "step_calibration_shift"


def _algorithm_contract() -> dict[str, object]:
    return {
        "schema_version": "t6.7.2-tail-formal-cell-cache-v1",
        "methods": list(METHODS),
        "parameter_period_decisions": PARAMETER_PERIOD_DECISIONS,
        "parameter_window_decisions": PARAMETER_WINDOW_DECISIONS,
        "posterior_window_decisions": POSTERIOR_WINDOW_DECISIONS,
        "ler_window_decisions": LER_WINDOW_DECISIONS,
        "router": (
            "continuously update Window/EWMA shadows; promote Window only after "
            "causal pre-update score win plus OPEN/smooth evidence; otherwise EWMA"
        ),
        "truth_use": "evaluation-only after complete online decisions",
        "aggregation": "cell mean within seed/family; formal seed is cluster",
        "tail_metrics": [
            "average",
            "p95_window",
            "seed_worst_window",
            "single_window_excess",
            "false_update",
            "fallback",
            "avoided_induced",
            "detection_recovery_lag",
        ],
    }


ALGORITHM_CONTRACT_SHA256 = _json_sha256(_algorithm_contract())


def _formal_cells_and_seeds() -> tuple[tuple[dict[str, object], ...], tuple[int, ...]]:
    cells = tuple(
        row
        for row in scenario_cells()
        if row["split_id"] == "formal_evaluation"
        and (row["family"] in ABRUPT_OOD_FAMILIES or row["family"] == NOMINAL_FAMILY)
    )
    formal = next(row for row in split_specs() if row.split_id == "formal_evaluation")
    dynamic = [row for row in cells if row["family"] in ABRUPT_OOD_FAMILIES]
    nominal = [row for row in cells if row["family"] == NOMINAL_FAMILY]
    if len(dynamic) != 36 or len(nominal) != 1 or len(cells) != 37:
        raise ValueError("formal tail matrix is not the frozen 6x6 plus nominal design")
    if len(formal.seeds) != 24:
        raise ValueError("formal seed cluster count is not 24")
    return cells, formal.seeds


def _record_formal_access(parents: Mapping[str, Any], path: Path) -> dict[str, object]:
    lock = parents["threshold_lock"]
    payload = {
        "schema_version": "t6.7.2-formal-access-ledger-v1",
        "task_id": TASK_ID,
        "first_access_is_irreversible": True,
        "formal_split": "formal_evaluation/abrupt_ood_and_nominal",
        "preregistration_sha256": _sha256(PREREG_ARTIFACT),
        "threshold_lock_artifact_sha256": _sha256(DEFAULT_LOCK),
        "threshold_lock_sha256": lock["threshold_lock"]["lock_sha256"],
        "primary_baseline": PRIMARY_BASELINE,
        "families": [*ABRUPT_OOD_FAMILIES, NOMINAL_FAMILY],
        "prohibitions": [
            "no formal baseline reselection",
            "no threshold/router retuning",
            "no family/cell/seed deletion",
            "no smooth-gain offset against tail or nominal failure",
        ],
    }
    if path.is_file():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if {key: stored[key] for key in payload} != payload:
            raise ValueError("T6.7.2 formal access ledger conflicts with frozen parents")
        return stored
    payload["first_accessed_at_utc"] = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return payload


def _transition_events(labels: np.ndarray, actions: np.ndarray) -> list[dict[str, object]]:
    truth = np.isin(np.asarray(labels, dtype=np.int8), TAIL_CLASS_INDICES)
    selected = np.asarray(actions, dtype=np.uint8)
    if truth.shape != selected.shape:
        raise ValueError("tail truth and action traces are not aligned")
    rows: list[dict[str, object]] = []
    for index in range(len(truth)):
        previous = bool(truth[index - 1]) if index else False
        current = bool(truth[index])
        if current and not previous:
            match = np.flatnonzero(selected[index:] != 0)
            detected = None if not len(match) else index + int(match[0])
            rows.append(
                {
                    "event_type": "tail_onset_to_fallback",
                    "truth_update": index,
                    "action_update": detected,
                    "lag_posterior_windows": None if detected is None else detected - index,
                    "lag_decisions": None if detected is None else (detected - index) * POSTERIOR_WINDOW_DECISIONS,
                    "right_censored": detected is None,
                }
            )
        if not current and previous:
            match = np.flatnonzero(selected[index:] == 0)
            recovered = None if not len(match) else index + int(match[0])
            rows.append(
                {
                    "event_type": "tail_recovery_to_open",
                    "truth_update": index,
                    "action_update": recovered,
                    "lag_posterior_windows": None if recovered is None else recovered - index,
                    "lag_decisions": None if recovered is None else (recovered - index) * POSTERIOR_WINDOW_DECISIONS,
                    "right_censored": recovered is None,
                }
            )
    return rows


def _cache_context(
    trajectory: Any,
    cell: Mapping[str, object],
    parents: Mapping[str, Any],
    calibration_sha256: str,
) -> dict[str, object]:
    lock = parents["threshold_lock"]
    return {
        **_algorithm_contract(),
        "algorithm_contract_sha256": ALGORITHM_CONTRACT_SHA256,
        "protocol_id": PROTOCOL_ID,
        "preregistration_sha256": _sha256(PREREG_ARTIFACT),
        "threshold_lock_artifact_sha256": _sha256(DEFAULT_LOCK),
        "threshold_lock_sha256": lock["threshold_lock"]["lock_sha256"],
        "baseline_selection_sha256": lock["threshold_lock"]["lock_core"]["baseline_selection_sha256"],
        "calibration_sha256": calibration_sha256,
        "cell": dict(cell),
        "seed": int(trajectory.seed),
        "observed_trace_sha256": trajectory.observed_trace_sha256,
        "truth_trace_sha256": trajectory.truth_trace_sha256,
        "scored_start_decision": int(trajectory.scored_start_decision),
    }


def _run_trajectory(
    cell: Mapping[str, object],
    seed: int,
    parents: Mapping[str, Any],
    calibration: np.ndarray,
    cache_dir: Path,
) -> tuple[dict[str, object], bool]:
    settings = RouteAPosteriorCalibrationConfig()
    trajectory = _trajectory(cell, seed, settings, keep_decisions=True)
    residuals = np.asarray(trajectory.decision_residuals, dtype=np.float64)
    truth = np.asarray(trajectory.logical_truth, dtype=np.uint8)
    context = _cache_context(
        trajectory, cell, parents, _array_sha256(calibration, "<f8")
    )
    cache_key = _json_sha256(context)
    cache_path = cache_dir / f"{cache_key}.npz"
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cached:
            expected_context = json.dumps(
                context, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            )
            if str(cached["context_json"].item()) != expected_context:
                raise ValueError(f"T6.7.2 cache context mismatch: {cache_path}")
            result = json.loads(str(cached["result_json"].item()))
        _validate_trajectory_result(result)
        return result, True

    lock = parents["threshold_lock"]
    model, event_model, temperature, thresholds = _load_models(lock)
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
    decisions = {method: np.empty(len(truth), dtype=np.uint8) for method in METHODS}
    decisions["standard_binning"].fill(0)
    wallclock_ns = {method: 0 for method in METHODS}

    route_begin = perf_counter_ns()
    posterior = temperature_scale(model.filter_base(trajectory.features), temperature)
    ood = np.maximum(
        np.asarray(trajectory.ood_score_codes, dtype=np.uint8),
        event_model.score_codes(trajectory.features),
    )
    actions = _selected_action_trace(
        posterior,
        ood,
        np.asarray([0], dtype=np.int64),
        np.asarray([len(posterior)], dtype=np.int64),
        thresholds,
    )
    route_model_ns = perf_counter_ns() - route_begin

    preferred_next: list[int] = []
    parameter_window_id = 0
    for start in range(0, len(truth), PARAMETER_PERIOD_DECISIONS):
        stop = min(len(truth), start + PARAMETER_PERIOD_DECISIONS)
        local = residuals[start:stop]
        begin = perf_counter_ns()
        decisions["static_joint_map"][start:stop] = np.asarray(
            map_decode_2d(
                local, static.covariance_array(), mean=static.mean_array()
            ).logical_class,
            dtype=np.uint8,
        )
        wallclock_ns["static_joint_map"] += perf_counter_ns() - begin
        predictions: dict[str, Any] = {}
        for method, predictor in predictors.items():
            prediction = predictor.prediction()
            predictions[method] = prediction
            begin = perf_counter_ns()
            decisions[method][start:stop] = _prediction_classes(local, prediction)
            wallclock_ns[method] += perf_counter_ns() - begin
        if stop - start == PARAMETER_PERIOD_DECISIONS and stop < len(truth):
            update_values = residuals[stop - PARAMETER_WINDOW_DECISIONS : stop]
            scores = np.asarray(
                [
                    _causal_gaussian_predictive_score(
                        update_values, predictions[method]
                    )
                    for method in DUAL_BANK_EXPERTS
                ]
            )
            preferred_next.append(int(np.argmin(scores)))
            for method, predictor in predictors.items():
                begin = perf_counter_ns()
                predictor.update(update_values, window_id=parameter_window_id)
                wallclock_ns[method] += perf_counter_ns() - begin
            parameter_window_id += 1

    period_count = (
        len(truth) + PARAMETER_PERIOD_DECISIONS - 1
    ) // PARAMETER_PERIOD_DECISIONS
    active_expert = np.ones(period_count, dtype=np.uint8)
    commit_rows: list[dict[str, object]] = []
    for period_index, preferred in enumerate(preferred_next):
        stop = (period_index + 1) * PARAMETER_PERIOD_DECISIONS
        boundary_update = min(
            len(actions) - 1,
            max(0, stop // POSTERIOR_WINDOW_DECISIONS - 1),
        )
        window_allowed = bool(
            preferred == 0
            and posterior[boundary_update, CLASS_TO_INDEX["smooth"]]
            >= SMOOTH_BANK_POSTERIOR_MIN
            and actions[boundary_update] == 0
        )
        selected_expert = 0 if window_allowed else 1
        active_expert[period_index + 1] = selected_expert
        commit_rows.append(
            {
                "commit_decision": stop,
                "boundary_posterior_update": boundary_update,
                "preferred_expert": DUAL_BANK_EXPERTS[preferred],
                "selected_expert": DUAL_BANK_EXPERTS[selected_expert],
                "policy_action": int(actions[boundary_update]),
                "truth_class": int(trajectory.labels[boundary_update]),
                "accepted_and_acknowledged": True,
            }
        )
    router_begin = perf_counter_ns()
    for period_index, start in enumerate(
        range(0, len(truth), PARAMETER_PERIOD_DECISIONS)
    ):
        stop = min(len(truth), start + PARAMETER_PERIOD_DECISIONS)
        source = DUAL_BANK_EXPERTS[int(active_expert[period_index])]
        decisions["proposed_route_a"][start:stop] = decisions[source][start:stop]
    wallclock_ns["proposed_route_a"] = (
        route_model_ns
        + wallclock_ns["window_map"]
        + wallclock_ns["ewma_adaptive_map"]
        + perf_counter_ns()
        - router_begin
    )

    scored_start = int(trajectory.scored_start_decision)
    scored_slice = slice(scored_start, len(truth))
    scored_truth = truth[scored_slice]
    method_counts: dict[str, list[list[int]]] = {}
    method_hashes: dict[str, str] = {}
    method_errors: dict[str, np.ndarray] = {}
    for method in METHODS:
        error_classes = np.bitwise_xor(decisions[method][scored_slice], scored_truth)
        method_errors[method] = error_classes != 0
        method_counts[method] = _window_class_counts(error_classes).astype(int).tolist()
        method_hashes[method] = _array_sha256(error_classes, "u1")
    off_error = method_errors[PRIMARY_BASELINE]
    on_error = method_errors["proposed_route_a"]
    scored_actions = np.repeat(actions, POSTERIOR_WINDOW_DECISIONS)[scored_slice]
    fallback = scored_actions != 0
    paired = np.column_stack(
        (
            ~(off_error | on_error),
            off_error & ~on_error,
            ~off_error & on_error,
            off_error & on_error,
        )
    ).reshape((-1, LER_WINDOW_DECISIONS, 4)).sum(axis=1).astype(np.int16)
    fallback_windows = fallback.reshape((-1, LER_WINDOW_DECISIONS)).sum(axis=1).astype(np.int16)
    unnecessary_windows = (
        fallback & ~off_error & ~on_error
    ).reshape((-1, LER_WINDOW_DECISIONS)).sum(axis=1).astype(np.int16)
    scored_commits = [
        row for row in commit_rows if int(row["commit_decision"]) >= scored_start
    ]
    false_updates = sum(
        int(row["truth_class"])
        not in (CLASS_TO_INDEX["normal"], CLASS_TO_INDEX["smooth"])
        for row in scored_commits
    )
    events = _transition_events(trajectory.labels, actions)
    result = {
        "seed": int(seed),
        "cell_id": str(cell["cell_id"]),
        "family": str(cell["family"]),
        "cell_parameters": {
            key: cell[key]
            for key in (
                "transition_rate_per_window",
                "amplitude",
                "duration_windows",
                "scored_windows",
                "nominal_preamble_windows",
            )
        },
        "input_sha256": trajectory.observed_trace_sha256,
        "truth_sha256": trajectory.truth_trace_sha256,
        "scored_start_decision": scored_start,
        "scored_decisions": len(scored_truth),
        "method_window_pauli_counts_class_order_I_Z_X_Y": method_counts,
        "method_error_trace_sha256": method_hashes,
        "paired_ewma_proposed_window_counts_order_both_correct_avoided_induced_both_wrong": paired.astype(int).tolist(),
        "fallback_window_decision_counts": fallback_windows.astype(int).tolist(),
        "unnecessary_fallback_window_decision_counts": unnecessary_windows.astype(int).tolist(),
        "posterior_update_count": len(actions),
        "scored_posterior_update_count": len(actions) - scored_start // POSTERIOR_WINDOW_DECISIONS,
        "scored_fallback_update_count": int(
            np.sum(actions[scored_start // POSTERIOR_WINDOW_DECISIONS :] != 0)
        ),
        "scored_tail_action_count": int(
            np.sum(actions[scored_start // POSTERIOR_WINDOW_DECISIONS :] == 1)
        ),
        "scored_uncertain_action_count": int(
            np.sum(actions[scored_start // POSTERIOR_WINDOW_DECISIONS :] == 2)
        ),
        "commit_count": len(commit_rows),
        "scored_commit_count": len(scored_commits),
        "window_bank_scored_commit_count": sum(
            row["selected_expert"] == "window_map" for row in scored_commits
        ),
        "ewma_bank_scored_commit_count": sum(
            row["selected_expert"] == PRIMARY_BASELINE for row in scored_commits
        ),
        "false_update_count": int(false_updates),
        "commit_rows": commit_rows,
        "transition_events": events,
        "wallclock_ns": {key: int(value) for key, value in wallclock_ns.items()},
        "route_model_and_action_wallclock_ns": int(route_model_ns),
        "cache_key": cache_key,
        "cache_context_sha256": _json_sha256(context),
    }
    _validate_trajectory_result(result)
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            context_json=np.asarray(
                json.dumps(
                    context,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                )
            ),
            result_json=np.asarray(
                json.dumps(
                    result,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                )
            ),
        )
    temporary.replace(cache_path)
    return result, False


def _validate_trajectory_result(row: Mapping[str, Any]) -> None:
    if row.get("family") not in (*ABRUPT_OOD_FAMILIES, NOMINAL_FAMILY):
        raise ValueError("T6.7.2 trajectory family is invalid")
    if int(row.get("scored_decisions", 0)) != 96 * LER_WINDOW_DECISIONS:
        raise ValueError("T6.7.2 scored decision count is invalid")
    tables = row.get("method_window_pauli_counts_class_order_I_Z_X_Y")
    if not isinstance(tables, Mapping) or set(tables) != set(METHODS):
        raise ValueError("T6.7.2 method table is incomplete")
    for method in METHODS:
        counts = np.asarray(tables[method], dtype=np.int64)
        if (
            counts.shape != (96, 4)
            or np.any(counts < 0)
            or not np.all(np.sum(counts, axis=1) == LER_WINDOW_DECISIONS)
        ):
            raise ValueError(f"T6.7.2 Pauli counts invalid for {method}")
    paired = np.asarray(
        row[
            "paired_ewma_proposed_window_counts_order_both_correct_avoided_induced_both_wrong"
        ],
        dtype=np.int64,
    )
    if paired.shape != (96, 4) or not np.all(
        np.sum(paired, axis=1) == LER_WINDOW_DECISIONS
    ):
        raise ValueError("T6.7.2 paired outcome table is invalid")
    for field in (
        "fallback_window_decision_counts",
        "unnecessary_fallback_window_decision_counts",
    ):
        values = np.asarray(row[field], dtype=np.int64)
        if (
            values.shape != (96,)
            or np.any(values < 0)
            or np.any(values > LER_WINDOW_DECISIONS)
        ):
            raise ValueError(f"T6.7.2 {field} is invalid")
    for event in row["transition_events"]:
        if event["event_type"] not in (
            "tail_onset_to_fallback",
            "tail_recovery_to_open",
        ):
            raise ValueError("T6.7.2 transition event type is invalid")
        if event["right_censored"] != (event["lag_decisions"] is None):
            raise ValueError("T6.7.2 transition censoring is inconsistent")


def _selected_rows(
    rows: Sequence[Mapping[str, Any]], family: str, seed: int
) -> list[Mapping[str, Any]]:
    selected = [
        row
        for row in rows
        if row["family"] == family and int(row["seed"]) == int(seed)
    ]
    expected = 1 if family == NOMINAL_FAMILY else 6
    if len(selected) != expected:
        raise ValueError(
            f"T6.7.2 {family}/{seed} has {len(selected)} cells, expected {expected}"
        )
    return selected


def _seed_metric(
    rows: Sequence[Mapping[str, Any]],
    method: str,
    family: str,
    seeds: Sequence[int],
    metric: str,
) -> np.ndarray:
    output = []
    for seed in seeds:
        selected = _selected_rows(rows, family, seed)
        if metric == "average":
            cell_rates = []
            for row in selected:
                counts = np.asarray(
                    row["method_window_pauli_counts_class_order_I_Z_X_Y"][method]
                )
                cell_rates.append(float(np.sum(counts[:, 1:]) / np.sum(counts)))
            output.append(float(np.mean(cell_rates)))
        else:
            window_counts = np.concatenate(
                [
                    np.sum(
                        np.asarray(
                            row[
                                "method_window_pauli_counts_class_order_I_Z_X_Y"
                            ][method],
                            dtype=np.int64,
                        )[:, 1:],
                        axis=1,
                    )
                    for row in selected
                ]
            )
            if metric == "p95":
                output.append(
                    float(
                        np.quantile(
                            window_counts / LER_WINDOW_DECISIONS,
                            0.95,
                            method="higher",
                        )
                    )
                )
            elif metric == "worst":
                output.append(float(np.max(window_counts) / LER_WINDOW_DECISIONS))
            else:
                raise ValueError(f"unknown T6.7.2 seed metric: {metric}")
    return np.asarray(output, dtype=np.float64)


def _family_method_summary(
    rows: Sequence[Mapping[str, Any]], family: str, method: str, seeds: Sequence[int]
) -> dict[str, object]:
    average = _seed_metric(rows, method, family, seeds, "average")
    p95 = _seed_metric(rows, method, family, seeds, "p95")
    worst = _seed_metric(rows, method, family, seeds, "worst")
    selected = [row for row in rows if row["family"] == family]
    global_count = -1
    locator: dict[str, object] | None = None
    for row in selected:
        counts = np.asarray(
            row["method_window_pauli_counts_class_order_I_Z_X_Y"][method]
        )
        errors = np.sum(counts[:, 1:], axis=1)
        index = int(np.argmax(errors))
        if int(errors[index]) > global_count:
            global_count = int(errors[index])
            locator = {
                "seed": row["seed"],
                "cell_id": row["cell_id"],
                "window_id": index,
                "errors": global_count,
                "denominator": LER_WINDOW_DECISIONS,
            }
    return {
        "family": family,
        "method_id": method,
        "average_ler": float(np.mean(average)),
        "seed_mean_p95_window_ler": float(np.mean(p95)),
        "seed_mean_worst_window_ler": float(np.mean(worst)),
        "global_worst_window_error_count": global_count,
        "global_worst_window_ler": global_count / LER_WINDOW_DECISIONS,
        "global_worst_locator": locator,
    }


def _event_summary(rows: Sequence[Mapping[str, Any]], family: str) -> dict[str, object]:
    selected = [row for row in rows if row["family"] == family]
    output: dict[str, object] = {}
    for event_type in ("tail_onset_to_fallback", "tail_recovery_to_open"):
        events = [
            event
            for row in selected
            for event in row["transition_events"]
            if event["event_type"] == event_type
        ]
        observed = [
            int(event["lag_decisions"])
            for event in events
            if event["lag_decisions"] is not None
        ]
        output[event_type] = {
            "events": len(events),
            "observed": len(observed),
            "right_censored": sum(bool(event["right_censored"]) for event in events),
            "mean_decisions": float(np.mean(observed)) if observed else None,
            "p95_higher_decisions": (
                float(np.quantile(observed, 0.95, method="higher"))
                if observed
                else None
            ),
            "max_decisions": max(observed) if observed else None,
        }
    return output


def _bootstrap_indices(seeds: Sequence[int]) -> np.ndarray:
    return np.random.default_rng(BOOTSTRAP_SEED).integers(
        0,
        len(seeds),
        size=(BOOTSTRAP_REPLICATES, len(seeds)),
    )


def _paired_interval(values: np.ndarray, indices: np.ndarray) -> dict[str, float]:
    samples = np.mean(np.asarray(values, dtype=np.float64)[indices], axis=1)
    return {
        "estimate": float(np.mean(values)),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
    }


def _analyze(
    rows: Sequence[Mapping[str, Any]], seeds: Sequence[int]
) -> dict[str, object]:
    for row in rows:
        _validate_trajectory_result(row)
    indices = _bootstrap_indices(seeds)
    prereg = protocol_payload()
    margins = prereg["acceptance_gates"]
    family_results = []
    method_summaries = []
    for family in (*ABRUPT_OOD_FAMILIES, NOMINAL_FAMILY):
        for method in METHODS:
            method_summaries.append(
                _family_method_summary(rows, family, method, seeds)
            )
        if family == NOMINAL_FAMILY:
            continue
        baseline = {
            metric: _seed_metric(rows, PRIMARY_BASELINE, family, seeds, metric)
            for metric in ("average", "p95", "worst")
        }
        proposed = {
            metric: _seed_metric(rows, "proposed_route_a", family, seeds, metric)
            for metric in ("average", "p95", "worst")
        }
        intervals = {
            metric: _paired_interval(proposed[metric] - baseline[metric], indices)
            for metric in ("average", "p95", "worst")
        }
        selected = [row for row in rows if row["family"] == family]
        max_excess = -10**9
        locator = None
        for row in selected:
            baseline_counts = np.sum(
                np.asarray(
                    row["method_window_pauli_counts_class_order_I_Z_X_Y"][
                        PRIMARY_BASELINE
                    ]
                )[:, 1:],
                axis=1,
            )
            proposed_counts = np.sum(
                np.asarray(
                    row["method_window_pauli_counts_class_order_I_Z_X_Y"][
                        "proposed_route_a"
                    ]
                )[:, 1:],
                axis=1,
            )
            excess = proposed_counts - baseline_counts
            index = int(np.argmax(excess))
            if int(excess[index]) > max_excess:
                max_excess = int(excess[index])
                locator = {
                    "seed": row["seed"],
                    "cell_id": row["cell_id"],
                    "window_id": index,
                    "proposed_minus_baseline_errors": max_excess,
                }
        catastrophic = margins["catastrophic_degradation_each_abrupt_ood_family"]
        passes = {
            "average_ucb": intervals["average"]["ci95_high"]
            <= catastrophic[
                "average_ler_proposed_minus_baseline_95_ucb_max"
            ],
            "p95_ucb": intervals["p95"]["ci95_high"]
            <= catastrophic[
                "p95_window_ler_proposed_minus_baseline_95_ucb_max"
            ],
            "seed_worst_ucb": intervals["worst"]["ci95_high"]
            <= catastrophic[
                "seed_worst_window_ler_proposed_minus_baseline_95_ucb_max"
            ],
            "single_window_excess": max_excess
            <= catastrophic["any_single_window_excess_error_count_max"],
        }
        family_results.append(
            {
                "family": family,
                "average_proposed_minus_baseline": intervals["average"],
                "p95_proposed_minus_baseline": intervals["p95"],
                "seed_worst_proposed_minus_baseline": intervals["worst"],
                "max_single_window_excess_error_count": max_excess,
                "max_single_window_excess_locator": locator,
                "catastrophic_gate_components": passes,
                "passes_all_catastrophic_gates": all(passes.values()),
                "events": _event_summary(rows, family),
            }
        )

    # Calibration-shift strict counterexample gate.
    calibration_methods = {
        row["method_id"]: row
        for row in method_summaries
        if row["family"] == CALIBRATION_FAMILY
    }
    calibration_baseline_worst = calibration_methods[PRIMARY_BASELINE][
        "global_worst_window_error_count"
    ]
    calibration_proposed_worst = calibration_methods["proposed_route_a"][
        "global_worst_window_error_count"
    ]
    calibration_seed_worst = _seed_metric(
        rows, "proposed_route_a", CALIBRATION_FAMILY, seeds, "worst"
    ) - _seed_metric(rows, PRIMARY_BASELINE, CALIBRATION_FAMILY, seeds, "worst")
    calibration_interval = _paired_interval(calibration_seed_worst, indices)
    calibration_gate = {
        "baseline_global_worst_error_count": calibration_baseline_worst,
        "proposed_global_worst_error_count": calibration_proposed_worst,
        "global_worst_proposed_not_greater": calibration_proposed_worst
        <= calibration_baseline_worst,
        "seed_worst_proposed_minus_baseline": calibration_interval,
        "seed_worst_95_ucb_not_greater_than_zero": calibration_interval[
            "ci95_high"
        ]
        <= 0.0,
    }
    calibration_gate["passes"] = bool(
        calibration_gate["global_worst_proposed_not_greater"]
        and calibration_gate["seed_worst_95_ucb_not_greater_than_zero"]
    )

    # Nominal negative control.
    baseline_nominal = _seed_metric(
        rows, PRIMARY_BASELINE, NOMINAL_FAMILY, seeds, "average"
    )
    proposed_nominal = _seed_metric(
        rows, "proposed_route_a", NOMINAL_FAMILY, seeds, "average"
    )
    nominal_average = _paired_interval(proposed_nominal - baseline_nominal, indices)
    nominal_rows = [row for row in rows if row["family"] == NOMINAL_FAMILY]
    nominal_updates = sum(
        int(row["scored_posterior_update_count"]) for row in nominal_rows
    )
    nominal_fallback = sum(
        int(row["scored_fallback_update_count"]) for row in nominal_rows
    )
    nominal_decisions = sum(int(row["scored_decisions"]) for row in nominal_rows)
    nominal_unnecessary = sum(
        sum(row["unnecessary_fallback_window_decision_counts"])
        for row in nominal_rows
    )
    seed_induced_minus_avoided = []
    for seed in seeds:
        row = _selected_rows(rows, NOMINAL_FAMILY, seed)[0]
        paired = np.sum(
            np.asarray(
                row[
                    "paired_ewma_proposed_window_counts_order_both_correct_avoided_induced_both_wrong"
                ]
            ),
            axis=0,
        )
        seed_induced_minus_avoided.append(
            (int(paired[2]) - int(paired[1])) / int(row["scored_decisions"])
        )
    induced_interval = _paired_interval(
        np.asarray(seed_induced_minus_avoided), indices
    )
    nominal_margin = margins["nominal_non_inferiority"]
    nominal_components = {
        "average_ucb": nominal_average["ci95_high"]
        <= nominal_margin[
            "average_ler_proposed_minus_policy_off_95_ucb_max"
        ],
        "fallback_rate": nominal_fallback / nominal_updates
        <= nominal_margin["fallback_rate_max"],
        "unnecessary_fallback_rate": nominal_unnecessary / nominal_decisions
        <= nominal_margin["unnecessary_fallback_rate_max"],
        "induced_minus_avoided_ucb": induced_interval["ci95_high"]
        <= nominal_margin["induced_minus_avoided_rate_95_ucb_max"],
    }
    nominal_gate = {
        "average_proposed_minus_policy_off": nominal_average,
        "fallback_rate": nominal_fallback / nominal_updates,
        "unnecessary_fallback_rate": nominal_unnecessary / nominal_decisions,
        "induced_minus_avoided_rate": induced_interval,
        "components": nominal_components,
        "passes": all(nominal_components.values()),
    }

    action_by_family = []
    for family in (*ABRUPT_OOD_FAMILIES, NOMINAL_FAMILY):
        selected = [row for row in rows if row["family"] == family]
        paired = np.sum(
            [
                np.sum(
                    np.asarray(
                        row[
                            "paired_ewma_proposed_window_counts_order_both_correct_avoided_induced_both_wrong"
                        ]
                    ),
                    axis=0,
                )
                for row in selected
            ],
            axis=0,
        )
        updates = sum(int(row["scored_posterior_update_count"]) for row in selected)
        decisions = sum(int(row["scored_decisions"]) for row in selected)
        action_by_family.append(
            {
                "family": family,
                "fallback_rate": sum(
                    int(row["scored_fallback_update_count"]) for row in selected
                )
                / updates,
                "false_updates": sum(int(row["false_update_count"]) for row in selected),
                "commits": sum(int(row["scored_commit_count"]) for row in selected),
                "unnecessary_fallback_rate": sum(
                    sum(row["unnecessary_fallback_window_decision_counts"])
                    for row in selected
                )
                / decisions,
                "avoided_errors": int(paired[1]),
                "induced_errors": int(paired[2]),
                "events": _event_summary(rows, family),
            }
        )
    all_catastrophic = all(
        bool(row["passes_all_catastrophic_gates"]) for row in family_results
    )
    return {
        "family_method_summaries": method_summaries,
        "family_paired_safety": family_results,
        "calibration_shift_strict_gate": calibration_gate,
        "nominal_noninferiority_gate": nominal_gate,
        "action_metrics_by_family": action_by_family,
        "promotion_components": {
            "all_six_catastrophic_gates_pass": all_catastrophic,
            "calibration_shift_strict_gate_pass": calibration_gate["passes"],
            "nominal_noninferiority_gate_pass": nominal_gate["passes"],
        },
        "tail_safety_gate_passes": bool(
            all_catastrophic and calibration_gate["passes"] and nominal_gate["passes"]
        ),
        "bootstrap_contract": {
            "independent_cluster": "formal seed",
            "cluster_count": 24,
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
        },
    }


def _source_rows(report: Mapping[str, Any]) -> list[dict[str, object]]:
    output = []
    for row in report["trajectory_results"]:
        for method in METHODS:
            for window_id, values in enumerate(
                row["method_window_pauli_counts_class_order_I_Z_X_Y"][method]
            ):
                output.append(
                    {
                        "row_type": "formal_window",
                        "seed": row["seed"],
                        "cell_id": row["cell_id"],
                        "family": row["family"],
                        "method_id": method,
                        "window_id": window_id,
                        "n_I": values[0],
                        "n_X": values[2],
                        "n_Y": values[3],
                        "n_Z": values[1],
                        "denominator": LER_WINDOW_DECISIONS,
                        "input_sha256": row["input_sha256"],
                        "detail": row["method_error_trace_sha256"][method],
                    }
                )
        paired_rows = row[
            "paired_ewma_proposed_window_counts_order_both_correct_avoided_induced_both_wrong"
        ]
        for window_id, (paired, fallback, unnecessary) in enumerate(
            zip(
                paired_rows,
                row["fallback_window_decision_counts"],
                row["unnecessary_fallback_window_decision_counts"],
                strict=True,
            )
        ):
            output.append(
                {
                    "row_type": "paired_outcome_window",
                    "seed": row["seed"],
                    "cell_id": row["cell_id"],
                    "family": row["family"],
                    "method_id": "ewma_adaptive_map_vs_proposed_route_a",
                    "window_id": window_id,
                    "n_I": paired[0],
                    "n_X": paired[1],
                    "n_Y": paired[2],
                    "n_Z": paired[3],
                    "denominator": LER_WINDOW_DECISIONS,
                    "input_sha256": row["input_sha256"],
                    "detail": "both_correct|avoided|induced|both_wrong",
                }
            )
            output.append(
                {
                    "row_type": "action_window",
                    "seed": row["seed"],
                    "cell_id": row["cell_id"],
                    "family": row["family"],
                    "method_id": "proposed_route_a",
                    "window_id": window_id,
                    "n_I": fallback,
                    "n_X": unnecessary,
                    "n_Y": 0,
                    "n_Z": 0,
                    "denominator": LER_WINDOW_DECISIONS,
                    "input_sha256": row["input_sha256"],
                    "detail": "fallback|unnecessary_fallback",
                }
            )
        output.append(
            {
                "row_type": "trajectory_action",
                "seed": row["seed"],
                "cell_id": row["cell_id"],
                "family": row["family"],
                "method_id": "proposed_route_a",
                "window_id": -1,
                "n_I": row["scored_fallback_update_count"],
                "n_X": row["false_update_count"],
                "n_Y": row["window_bank_scored_commit_count"],
                "n_Z": row["ewma_bank_scored_commit_count"],
                "denominator": row["scored_posterior_update_count"],
                "input_sha256": row["input_sha256"],
                "detail": "fallback_updates|false_updates|window_commits|ewma_commits",
            }
        )
        for event_index, event in enumerate(row["transition_events"]):
            output.append(
                {
                    "row_type": "transition_event",
                    "seed": row["seed"],
                    "cell_id": row["cell_id"],
                    "family": row["family"],
                    "method_id": "proposed_route_a",
                    "window_id": event_index,
                    "n_I": event["truth_update"],
                    "n_X": -1 if event["action_update"] is None else event["action_update"],
                    "n_Y": -1 if event["lag_decisions"] is None else event["lag_decisions"],
                    "n_Z": int(event["right_censored"]),
                    "denominator": POSTERIOR_WINDOW_DECISIONS,
                    "input_sha256": row["input_sha256"],
                    "detail": event["event_type"],
                }
            )
    return output


def _validate_core(report: Mapping[str, Any], *, verify_source: bool = True) -> None:
    rows = report.get("trajectory_results")
    if not isinstance(rows, list) or len(rows) != 888:
        raise ValueError("T6.7.2 formal trajectory matrix is incomplete")
    for row in rows:
        _validate_trajectory_result(row)
    analysis = _analyze(rows, report["formal_design"]["seeds"])
    if report.get("analysis") != analysis or report.get("analysis_sha256") != _json_sha256(analysis):
        raise ValueError("T6.7.2 analysis does not recompute from raw trajectories")
    if report["primary_baseline"] != PRIMARY_BASELINE or report["formal_baseline_reselection"] is not False:
        raise ValueError("T6.7.2 primary baseline was reselected")
    if report["parent_bindings"]["threshold_lock_sha256"] != "9347edb270bbeb3f50d8bd8aceaeefd8003e118f1e88712dd5265519bb0f67aa":
        raise ValueError("T6.7.2 V4 threshold lock was replaced")
    for binding in report["source_bindings"]:
        path = ROOT / binding["path"]
        if not path.is_file() or _sha256(path) != binding["sha256"]:
            raise ValueError(f"T6.7.2 source binding stale: {binding['path']}")
    if verify_source:
        source = ROOT / report["source_data_binding"]["path"]
        if not source.is_file() or _sha256(source) != report["source_data_binding"]["sha256"]:
            raise ValueError("T6.7.2 Source Data binding is stale")


def _semantic_mutations(report: Mapping[str, Any]) -> list[dict[str, object]]:
    cases = (
        ("replace_lock", ("parent_bindings", "threshold_lock_sha256"), "0" * 64),
        ("reselect_baseline", ("primary_baseline",), "window_map"),
        ("claim_reselection", ("formal_baseline_reselection",), True),
        ("drop_trajectory", ("trajectory_results",), report["trajectory_results"][:-1]),
        ("flip_tail_gate", ("analysis", "tail_safety_gate_passes"), not report["analysis"]["tail_safety_gate_passes"]),
        ("delete_failed_family", ("analysis", "family_paired_safety"), report["analysis"]["family_paired_safety"][:-1]),
    )
    output = []
    for mutation, path, value in cases:
        mutated = deepcopy(report)
        target: Any = mutated
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        try:
            _validate_core(mutated, verify_source=False)
        except (TypeError, ValueError, KeyError) as exc:
            output.append({"mutation": mutation, "rejected": True, "detail": str(exc)})
        else:
            output.append({"mutation": mutation, "rejected": False, "detail": "accepted"})
    return output


def recompute_gates(report: Mapping[str, Any]) -> dict[str, bool]:
    rows = report["trajectory_results"]
    seeds = report["formal_design"]["seeds"]
    analysis = _analyze(rows, seeds)
    expected_pairs = {
        (int(seed), str(cell["cell_id"]))
        for seed in seeds
        for cell in report["formal_design"]["cells"]
    }
    actual_pairs = {(int(row["seed"]), str(row["cell_id"])) for row in rows}
    source = ROOT / report["source_data_binding"]["path"]
    return {
        "G01_parent_v4_lock_and_sources_current": report["parent_bindings"]["preregistration_sha256"] == _sha256(PREREG_ARTIFACT) and report["parent_bindings"]["threshold_lock_artifact_sha256"] == _sha256(DEFAULT_LOCK) and report["parent_bindings"]["threshold_lock_sha256"] == "9347edb270bbeb3f50d8bd8aceaeefd8003e118f1e88712dd5265519bb0f67aa" and all(_sha256(ROOT / row["path"]) == row["sha256"] for row in report["source_bindings"]),
        "G02_primary_baseline_remains_pilot_locked_ewma": report["primary_baseline"] == PRIMARY_BASELINE and report["formal_baseline_reselection"] is False,
        "G03_exact_24_seed_by_37_cell_matrix_executed": len(rows) == 888 and actual_pairs == expected_pairs,
        "G04_six_abrupt_families_and_nominal_complete": set(row["family"] for row in rows) == set((*ABRUPT_OOD_FAMILIES, NOMINAL_FAMILY)) and all(sum(row["family"] == family for row in rows) == 144 for family in ABRUPT_OOD_FAMILIES) and sum(row["family"] == NOMINAL_FAMILY for row in rows) == 24,
        "G05_every_trajectory_has_six_methods_and_96_windows": all(set(row["method_window_pauli_counts_class_order_I_Z_X_Y"]) == set(METHODS) and int(row["scored_decisions"]) == 96 * 512 for row in rows),
        "G06_analysis_recomputes_from_raw_counts": report["analysis"] == analysis and report["analysis_sha256"] == _json_sha256(analysis),
        "G07_cluster_bootstrap_contract_frozen": analysis["bootstrap_contract"] == {"independent_cluster": "formal seed", "cluster_count": 24, "replicates": 20000, "seed": 202607176999},
        "G08_source_data_complete_and_hash_bound": source.is_file() and _sha256(source) == report["source_data_binding"]["sha256"] and int(report["source_data_binding"]["row_count"]) == int(report["source_data_binding"]["expected_row_count"]),
        "G09_cache_replay_binds_all_888_traces": len(report["cache_audit"]["cache_keys"]) == len(set(report["cache_audit"]["cache_keys"])) == 888 and report["cache_audit"]["algorithm_contract_sha256"] == ALGORITHM_CONTRACT_SHA256,
        "G10_all_lag_events_and_censoring_are_defined": all(event["right_censored"] == (event["lag_decisions"] is None) for row in rows for event in row["transition_events"]),
        "G11_semantic_mutations_all_rejected": len(report.get("semantic_mutations", [])) == 6 and all(row["rejected"] for row in report.get("semantic_mutations", [])),
    }


def build_report(
    *,
    access_ledger: Path = DEFAULT_ACCESS_LEDGER,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> dict[str, Any]:
    parents = _load_parents(DEFAULT_LOCK)
    prereg_protocol = {
        key: parents["preregistration"][key] for key in protocol_payload()
    }
    validate_protocol(prereg_protocol)
    ledger = _record_formal_access(parents, access_ledger)
    cells, seeds = _formal_cells_and_seeds()
    calibration = materialize_qualification_trace()[0].calibration_residuals
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    hits = 0
    misses = 0
    total = len(cells) * len(seeds)
    completed = 0
    for seed in seeds:
        for cell in cells:
            row, hit = _run_trajectory(cell, seed, parents, calibration, cache_dir)
            rows.append(row)
            hits += int(hit)
            misses += int(not hit)
            completed += 1
            if completed % 12 == 0 or completed == total:
                print(
                    json.dumps(
                        {
                            "task": TASK_ID,
                            "completed": completed,
                            "total": total,
                            "cache_hits": hits,
                            "cache_misses": misses,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    analysis = _analyze(rows, seeds)
    lock = parents["threshold_lock"]
    report = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_access_ledger": ledger,
        "parent_bindings": {
            "preregistration_sha256": _sha256(PREREG_ARTIFACT),
            "preregistration_protocol_sha256": parents["preregistration"]["protocol_sha256"],
            "threshold_lock_artifact_sha256": _sha256(DEFAULT_LOCK),
            "threshold_lock_sha256": lock["threshold_lock"]["lock_sha256"],
            "posterior_model_sha256": lock["posterior_model_sha256"],
            "event_model_sha256": lock["event_model_sha256"],
            "baseline_selection_sha256": lock["pilot_baseline_qualification"]["selection_sha256"],
        },
        "primary_baseline": PRIMARY_BASELINE,
        "formal_baseline_reselection": False,
        "formal_design": {
            "split_id": "formal_evaluation",
            "families": [*ABRUPT_OOD_FAMILIES, NOMINAL_FAMILY],
            "cells": list(cells),
            "seeds": list(seeds),
        },
        "method_contract": {
            "methods": list(METHODS),
            "same_quantized_syndrome": True,
            "hidden_truth_online": False,
            "pauli_class_encoding": {"0": "I", "1": "Z", "2": "X", "3": "Y"},
        },
        "trajectory_results": rows,
        "analysis": analysis,
        "analysis_sha256": _json_sha256(analysis),
        "cache_audit": {
            "directory": cache_dir.relative_to(ROOT).as_posix(),
            "schema_version": _algorithm_contract()["schema_version"],
            "algorithm_contract_sha256": ALGORITHM_CONTRACT_SHA256,
            "hits": hits,
            "misses": misses,
            "cache_keys": [row["cache_key"] for row in rows],
        },
        "source_bindings": [
            {"path": relative, "sha256": _sha256(ROOT / relative)}
            for relative in (
                "cnn_fpga/benchmark/route_a_tail_formal.py",
                "cnn_fpga/benchmark/route_a_smooth_formal.py",
                "cnn_fpga/benchmark/route_a_posterior_calibration.py",
                "cnn_fpga/benchmark/route_a_preregistration.py",
                "cnn_fpga/decoder/route_a_regime_posterior.py",
                "cnn_fpga/decoder/periodic_adaptive_map.py",
                "physics/ideal_gkp_decoder.py",
            )
        ],
        "source_data_binding": {
            "path": DEFAULT_SOURCE_DATA.relative_to(ROOT).as_posix(),
            "sha256": None,
            "row_count": 0,
            "expected_row_count": 0,
        },
        "semantic_mutations": [],
        "claim_boundary": {
            "tail_safety_gate_is_falsifiable": True,
            "smooth_gain_cannot_offset_tail_failure": True,
            "not_admitted": [
                "physical-device tail guarantee",
                "board deadline or measured latency",
                "static GKP superiority",
                "Puviani NMF lifetime superiority",
            ],
        },
    }
    return report


def write_report(
    artifact: Path = DEFAULT_ARTIFACT,
    source_data: Path = DEFAULT_SOURCE_DATA,
    *,
    access_ledger: Path = DEFAULT_ACCESS_LEDGER,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> dict[str, Any]:
    report = build_report(access_ledger=access_ledger, cache_dir=cache_dir)
    source_rows = _source_rows(report)
    source_data.parent.mkdir(parents=True, exist_ok=True)
    with source_data.open("w", encoding="utf-8", newline="") as handle:
        fields = (
            "row_type",
            "seed",
            "cell_id",
            "family",
            "method_id",
            "window_id",
            "n_I",
            "n_X",
            "n_Y",
            "n_Z",
            "denominator",
            "input_sha256",
            "detail",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(source_rows)
    report["source_data_binding"] = {
        "path": source_data.relative_to(ROOT).as_posix(),
        "sha256": _sha256(source_data),
        "row_count": len(source_rows),
        "expected_row_count": len(source_rows),
    }
    report["semantic_mutations"] = _semantic_mutations(report)
    gates = recompute_gates(report)
    report["gate_summary"] = {
        "passed": sum(gates.values()),
        "failed": sum(not value for value in gates.values()),
        "gates": gates,
    }
    valid = all(gates.values())
    scientific = bool(report["analysis"]["tail_safety_gate_passes"])
    report["status"] = "PASS" if valid else "FAIL"
    report["verdict"] = (
        "PASS_VALID_FORMAL_RESULT_TAIL_SAFETY_GATE_PASSED"
        if valid and scientific
        else "PASS_VALID_FORMAL_RESULT_TAIL_SAFETY_GATE_FAILED"
        if valid
        else "FAIL_INVALID_FORMAL_RESULT"
    )
    _validate_core(report)
    if not valid:
        raise ValueError(
            f"T6.7.2 evidence gates failed: {[key for key, value in gates.items() if not value]}"
        )
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def verify_report(report: Mapping[str, Any]) -> None:
    _validate_core(report)
    gates = recompute_gates(report)
    if report.get("gate_summary", {}).get("gates") != gates or not all(gates.values()):
        raise ValueError("T6.7.2 evidence gates do not recompute")
    if len(report.get("semantic_mutations", [])) != 6 or not all(
        row["rejected"] for row in report["semantic_mutations"]
    ):
        raise ValueError("T6.7.2 semantic mutation audit incomplete")
    scientific = bool(report["analysis"]["tail_safety_gate_passes"])
    expected = (
        "PASS_VALID_FORMAL_RESULT_TAIL_SAFETY_GATE_PASSED"
        if scientific
        else "PASS_VALID_FORMAL_RESULT_TAIL_SAFETY_GATE_FAILED"
    )
    if report.get("status") != "PASS" or report.get("verdict") != expected:
        raise ValueError("T6.7.2 verdict does not match recomputed tail gate")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--source-data", type=Path, default=DEFAULT_SOURCE_DATA)
    parser.add_argument("--access-ledger", type=Path, default=DEFAULT_ACCESS_LEDGER)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    if args.verify_only:
        report = json.loads(args.artifact.read_text(encoding="utf-8"))
        verify_report(report)
    else:
        report = write_report(
            args.artifact,
            args.source_data,
            access_ledger=args.access_ledger,
            cache_dir=args.cache_dir,
        )
    print(
        json.dumps(
            {
                "status": report["status"],
                "verdict": report["verdict"],
                "promotion": report["analysis"]["promotion_components"],
                "calibration": report["analysis"]["calibration_shift_strict_gate"],
                "nominal": report["analysis"]["nominal_noninferiority_gate"],
                "gates": report["gate_summary"],
                "cache": {
                    key: report["cache_audit"][key] for key in ("hits", "misses")
                },
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
