"""T3.2.4 post-selection diagnostic upper bound and rejection-cost audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import hashlib
import json
from math import isfinite
from pathlib import Path
from typing import Sequence

import numpy as np

from cnn_fpga.benchmark.continuous_adaptive_map import (
    _mean_interval,
    _residuals_and_truth,
    continuous_drift_scenarios,
)
from cnn_fpga.benchmark.static_map_baseline import (
    StaticMAPParameters,
    fit_static_map_from_training_states,
)
from cnn_fpga.decoder.postselection_diagnostic import (
    binary_score_auc,
    calibrate_survival_thresholds,
    evaluate_postselection,
    posterior_error_risk,
)
from physics.drift_processes import DriftState
from physics.ideal_gkp_decoder import map_decode_2d


def _integer(value: object, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _probability(value: object, name: str, *, strict: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a probability")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a probability") from exc
    lower = result > 0.0 if strict else result >= 0.0
    upper = result < 1.0 if strict else result <= 1.0
    if not isfinite(result) or not lower or not upper:
        interval = "(0,1)" if strict else "[0,1]"
        raise ValueError(f"{name} must lie in {interval}")
    return result


@dataclass(frozen=True)
class PostselectionDescriptor:
    task_id: str = "T3.2.4"
    label: str = "Offline observed-confidence post-selection diagnostic and truth upper bound"
    online_decoder: bool = False
    observed_score_inputs: tuple[str, ...] = ("static_map_logical_posterior",)
    hidden_truth_score_inputs: tuple[str, ...] = ()
    truth_only_evaluator_fields: tuple[str, ...] = ("logical_failure",)
    threshold_rule: str = "training_only_target_survival_quantile"
    reported_cost_rule: str = "accepted_failures_per_input_plus_penalty_times_rejection_fraction"
    primary_metric_eligible: bool = False
    evidence_scope: str = "continuous_synthetic_wrapped_gaussian_diagnostic_only"
    forbidden_claims: tuple[str, ...] = (
        "online_error_correction_gain",
        "postselected_break_even",
        "free_rejection",
        "device_calibrated_anomaly_detector",
        "FPGA_synthesis_or_measured_latency",
    )


POSTSELECTION_DESCRIPTOR = PostselectionDescriptor()


@dataclass(frozen=True)
class PostselectionValidationConfig:
    training_seeds: tuple[int, ...] = (20261011, 20261012, 20261013)
    evaluation_seeds: tuple[int, ...] = tuple(range(20261031, 20261039))
    windows: int = 48
    training_samples_per_window: int = 512
    evaluation_samples_per_window: int = 1024
    target_survivals: tuple[float, ...] = (
        0.995,
        0.99,
        0.98,
        0.95,
        0.90,
        0.80,
        0.70,
        0.50,
    )
    rejection_penalties: tuple[float, ...] = (0.0, 0.25, 0.50, 1.0)
    primary_diagnostic_survival: float = 0.90
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        training = tuple(self.training_seeds)
        evaluation = tuple(self.evaluation_seeds)
        if len(training) < 3 or len(set(training)) != len(training):
            raise ValueError("training_seeds must contain at least three unique values")
        if len(evaluation) < 6 or len(set(evaluation)) != len(evaluation):
            raise ValueError("evaluation_seeds must contain at least six unique values")
        if set(training) & set(evaluation):
            raise ValueError("training and evaluation seeds must be disjoint")
        if any(
            isinstance(seed, bool)
            or not isinstance(seed, (int, np.integer))
            or int(seed) < 0
            or int(seed) >= 2**64 - 100_000
            for seed in training + evaluation
        ):
            raise ValueError("seeds must be nonnegative uint64-safe integers")
        object.__setattr__(self, "training_seeds", tuple(int(seed) for seed in training))
        object.__setattr__(self, "evaluation_seeds", tuple(int(seed) for seed in evaluation))
        object.__setattr__(self, "windows", _integer(self.windows, "windows", 16))
        object.__setattr__(
            self,
            "training_samples_per_window",
            _integer(self.training_samples_per_window, "training_samples_per_window", 256),
        )
        object.__setattr__(
            self,
            "evaluation_samples_per_window",
            _integer(self.evaluation_samples_per_window, "evaluation_samples_per_window", 512),
        )
        targets = tuple(_probability(value, "target survival", strict=True) for value in self.target_survivals)
        if len(targets) < 5 or len(set(targets)) != len(targets):
            raise ValueError("target_survivals must contain at least five unique values")
        if tuple(sorted(targets, reverse=True)) != targets:
            raise ValueError("target_survivals must be strictly decreasing")
        object.__setattr__(self, "target_survivals", targets)
        penalties = tuple(_probability(value, "rejection penalty") for value in self.rejection_penalties)
        if len(penalties) < 4 or tuple(sorted(penalties)) != penalties:
            raise ValueError("rejection_penalties must contain at least four increasing values")
        if penalties[0] != 0.0 or penalties[-1] != 1.0:
            raise ValueError("rejection_penalties must span 0 to 1")
        object.__setattr__(self, "rejection_penalties", penalties)
        primary = _probability(self.primary_diagnostic_survival, "primary_diagnostic_survival", strict=True)
        if primary not in targets:
            raise ValueError("primary_diagnostic_survival must be a registered target")
        object.__setattr__(self, "primary_diagnostic_survival", primary)
        confidence = _probability(self.confidence_level, "confidence_level", strict=True)
        object.__setattr__(self, "confidence_level", confidence)
        workload = len(evaluation) * len(continuous_drift_scenarios()) * self.windows * self.evaluation_samples_per_window
        if workload > 3_000_000:
            raise ValueError("evaluation workload must not exceed 3,000,000 samples")


def _fit_static_parameters(settings: PostselectionValidationConfig) -> StaticMAPParameters:
    states: list[DriftState] = []
    for scenario_index, scenario in enumerate(continuous_drift_scenarios()):
        for state in scenario.states(settings.windows):
            states.append(
                DriftState(
                    **{
                        **state.__dict__,
                        "step": len(states),
                        "time": float(len(states)),
                        "seed": settings.training_seeds[0] + scenario_index,
                    }
                )
            )
    return fit_static_map_from_training_states(
        tuple(states),
        training_protocol_id=(
            "t3.2.4-postselection-static-v1:"
            f"windows={settings.windows}:training_seeds={settings.training_seeds}"
        ),
    )


def _score_trace(
    settings: PostselectionValidationConfig,
    static_parameters: StaticMAPParameters,
    *,
    seeds: Sequence[int],
    samples_per_window: int,
) -> tuple[np.ndarray, str]:
    scores: list[np.ndarray] = []
    digest = hashlib.sha256()
    for scenario_index, scenario in enumerate(continuous_drift_scenarios()):
        states = scenario.states(settings.windows)
        for base_seed in seeds:
            seed = int(base_seed + 100_000 * scenario_index)
            rng = np.random.default_rng(seed)
            digest.update(scenario.scenario_id.encode("utf-8"))
            digest.update(seed.to_bytes(8, "little", signed=False))
            for window_id, state in enumerate(states):
                residual, _, displacements = _residuals_and_truth(state, samples_per_window, rng)
                result = map_decode_2d(
                    residual,
                    static_parameters.covariance_array(),
                    mean=static_parameters.mean_array(),
                )
                risk = posterior_error_risk(
                    np.asarray(result.posterior, dtype=np.float64).reshape((-1, 4))
                ).reshape(-1)
                scores.append(risk)
                digest.update(window_id.to_bytes(4, "little", signed=False))
                digest.update(np.asarray(displacements, dtype="<f8").tobytes())
                digest.update(np.asarray(risk, dtype="<f8").tobytes())
    return np.concatenate(scores), digest.hexdigest()


def _evaluate_seed(
    scenario_index: int,
    base_seed: int,
    settings: PostselectionValidationConfig,
    static_parameters: StaticMAPParameters,
    thresholds: Sequence[tuple[float, float]],
) -> tuple[list[dict[str, object]], float]:
    scenario = continuous_drift_scenarios()[scenario_index]
    seed = int(base_seed + 100_000 * scenario_index)
    rng = np.random.default_rng(seed)
    score_chunks: list[np.ndarray] = []
    failure_chunks: list[np.ndarray] = []
    digest = hashlib.sha256()
    digest.update(scenario.scenario_id.encode("utf-8"))
    digest.update(seed.to_bytes(8, "little", signed=False))
    for window_id, state in enumerate(scenario.states(settings.windows)):
        residual, truth, displacements = _residuals_and_truth(
            state, settings.evaluation_samples_per_window, rng
        )
        result = map_decode_2d(
            residual,
            static_parameters.covariance_array(),
            mean=static_parameters.mean_array(),
        )
        decision = np.asarray(result.logical_class, dtype=np.int64).reshape(-1)
        score = posterior_error_risk(
            np.asarray(result.posterior, dtype=np.float64).reshape((-1, 4))
        ).reshape(-1)
        score_chunks.append(score)
        failure_chunks.append(decision != truth.reshape(-1))
        digest.update(window_id.to_bytes(4, "little", signed=False))
        digest.update(np.asarray(displacements, dtype="<f8").tobytes())
        digest.update(np.asarray(score, dtype="<f8").tobytes())
    scores = np.concatenate(score_chunks)
    failures = np.concatenate(failure_chunks).astype(np.bool_)
    auc = binary_score_auc(scores, failures)
    rows: list[dict[str, object]] = []
    for target, threshold in thresholds:
        metrics = evaluate_postselection(
            scores,
            failures,
            threshold=threshold,
            rejection_penalties=settings.rejection_penalties,
        )
        row: dict[str, object] = {
            "scenario_id": scenario.scenario_id,
            "base_evaluation_seed": int(base_seed),
            "evaluation_seed": seed,
            "target_survival": target,
            "score_threshold": threshold,
            "score_auc": auc,
            "trace_sha256": digest.hexdigest(),
            **{
                key: value
                for key, value in asdict(metrics).items()
                if key != "total_cost_by_rejection_penalty"
            },
        }
        for penalty, cost in metrics.total_cost_by_rejection_penalty.items():
            row[f"total_cost_penalty_{penalty}"] = cost
        row["raw_minus_conditional_error_rate"] = (
            metrics.raw_error_rate - metrics.conditional_error_rate
        )
        rows.append(row)
    return rows, auc


def _implementation_sha256() -> str:
    paths = (
        Path(__file__),
        Path(__file__).parents[1] / "decoder" / "postselection_diagnostic.py",
        Path(__file__).with_name("continuous_adaptive_map.py"),
        Path(__file__).with_name("static_map_baseline.py"),
        Path(__file__).parents[2] / "physics" / "ideal_gkp_decoder.py",
        Path(__file__).parents[2] / "physics" / "drift_processes.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_postselection_validation(
    config: PostselectionValidationConfig | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    settings = PostselectionValidationConfig() if config is None else config
    if not isinstance(settings, PostselectionValidationConfig):
        raise TypeError("config must be PostselectionValidationConfig")
    static_parameters = _fit_static_parameters(settings)
    training_scores, training_trace_sha256 = _score_trace(
        settings,
        static_parameters,
        seeds=settings.training_seeds,
        samples_per_window=settings.training_samples_per_window,
    )
    thresholds = calibrate_survival_thresholds(training_scores, settings.target_survivals)
    rows: list[dict[str, object]] = []
    for scenario_index, _ in enumerate(continuous_drift_scenarios()):
        for base_seed in settings.evaluation_seeds:
            seed_rows, _ = _evaluate_seed(
                scenario_index,
                base_seed,
                settings,
                static_parameters,
                thresholds,
            )
            rows.extend(seed_rows)

    summaries: list[dict[str, object]] = []
    for scenario in continuous_drift_scenarios():
        for target, threshold in thresholds:
            selected = [
                row
                for row in rows
                if row["scenario_id"] == scenario.scenario_id
                and row["target_survival"] == target
            ]
            summaries.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "target_survival": target,
                    "training_threshold": threshold,
                    "seeds": len(selected),
                    "realized_survival_fraction": float(
                        np.mean([row["survival_fraction"] for row in selected])
                    ),
                    "raw_error_rate": float(np.mean([row["raw_error_rate"] for row in selected])),
                    "conditional_error_rate": float(
                        np.mean([row["conditional_error_rate"] for row in selected])
                    ),
                    "accepted_failures_per_input": float(
                        np.mean([row["accepted_failures_per_input"] for row in selected])
                    ),
                    "truth_upper_conditional_error_rate": float(
                        np.mean([row["truth_upper_conditional_error_rate"] for row in selected])
                    ),
                    "random_rejection_expected_conditional_error_rate": float(
                        np.mean(
                            [row["random_rejection_expected_conditional_error_rate"] for row in selected]
                        )
                    ),
                    "rejected_failure_capture_fraction": float(
                        np.mean([row["rejected_failure_capture_fraction"] for row in selected])
                    ),
                    "break_even_rejection_penalty": float(
                        np.mean([row["break_even_rejection_penalty"] for row in selected])
                    ),
                    "score_auc_seed_cluster_ci": _mean_interval(
                        [float(row["score_auc"]) for row in selected], settings.confidence_level
                    ),
                    "raw_minus_conditional_seed_cluster_ci": _mean_interval(
                        [float(row["raw_minus_conditional_error_rate"]) for row in selected],
                        settings.confidence_level,
                    ),
                    "mean_total_cost_by_rejection_penalty": {
                        f"{penalty:.2f}": float(
                            np.mean([row[f"total_cost_penalty_{penalty:.2f}"] for row in selected])
                        )
                        for penalty in settings.rejection_penalties
                    },
                }
            )
    primary = [
        summary
        for summary in summaries
        if summary["target_survival"] == settings.primary_diagnostic_survival
    ]
    aggregate_by_target = []
    for target, threshold in thresholds:
        seed_differences = []
        for seed in settings.evaluation_seeds:
            selected = [
                row
                for row in rows
                if row["base_evaluation_seed"] == seed and row["target_survival"] == target
            ]
            seed_differences.append(
                float(np.mean([row["raw_minus_conditional_error_rate"] for row in selected]))
            )
        target_rows = [row for row in rows if row["target_survival"] == target]
        aggregate_by_target.append(
            {
                "target_survival": target,
                "training_threshold": threshold,
                "realized_survival_fraction": float(
                    np.mean([row["survival_fraction"] for row in target_rows])
                ),
                "raw_error_rate": float(np.mean([row["raw_error_rate"] for row in target_rows])),
                "conditional_error_rate": float(
                    np.mean([row["conditional_error_rate"] for row in target_rows])
                ),
                "raw_minus_conditional_seed_cluster_ci": _mean_interval(
                    seed_differences, settings.confidence_level
                ),
            }
        )

    target_values = [row[0] for row in thresholds]
    threshold_values = [row[1] for row in thresholds]
    gates = {
        "training_and_evaluation_are_disjoint": not bool(
            set(settings.training_seeds) & set(settings.evaluation_seeds)
        ),
        "thresholds_are_training_only_and_monotone": (
            bool(training_trace_sha256)
            and target_values == sorted(target_values, reverse=True)
            and all(threshold_values[index] > threshold_values[index + 1] for index in range(len(threshold_values) - 1))
        ),
        "score_contract_has_no_hidden_truth_input": not POSTSELECTION_DESCRIPTOR.hidden_truth_score_inputs,
        "all_evaluation_traces_are_unique": (
            len({row["trace_sha256"] for row in rows})
            == len(continuous_drift_scenarios()) * len(settings.evaluation_seeds)
        ),
        "source_grid_is_complete": len(rows)
        == len(continuous_drift_scenarios())
        * len(settings.evaluation_seeds)
        * len(settings.target_survivals),
        "realized_survival_tracks_training_targets": all(
            abs(float(row["survival_fraction"]) - float(row["target_survival"])) < 0.10
            for row in rows
        ),
        "observed_score_is_informative_in_every_scenario": all(
            summary["score_auc_seed_cluster_ci"]["ci_low"] > 0.5 for summary in primary
        ),
        "primary_conditional_improvement_resolved_every_scenario": all(
            summary["raw_minus_conditional_seed_cluster_ci"]["ci_low"] > 0.0
            for summary in primary
        ),
        "truth_upper_bound_never_worse_than_observed": all(
            float(row["truth_upper_conditional_error_rate"])
            <= float(row["conditional_error_rate"]) + 1.0e-15
            for row in rows
        ),
        "truth_upper_frontier_has_nonzero_and_zero_regions": (
            any(float(row["truth_upper_conditional_error_rate"]) > 0.0 for row in rows)
            and any(float(row["truth_upper_conditional_error_rate"]) == 0.0 for row in rows)
        ),
        "random_rejection_does_not_fake_conditional_gain": all(
            float(row["random_rejection_expected_conditional_error_rate"])
            == float(row["raw_error_rate"])
            for row in rows
        ),
        "unit_rejection_cost_prevents_free_postselection_gain": all(
            float(row["total_cost_penalty_1.00"]) >= float(row["raw_error_rate"]) - 1.0e-15
            for row in rows
        ),
        "rejection_cost_and_survival_are_complete": all(
            0.0 < float(row["survival_fraction"]) < 1.0
            and abs(
                float(row["survival_fraction"])
                + float(row["rejection_fraction"])
                - 1.0
            )
            < 1.0e-12
            and 0.0 <= float(row["break_even_rejection_penalty"]) <= 1.0
            for row in rows
        ),
        "diagnostic_is_excluded_from_primary_metric": (
            not POSTSELECTION_DESCRIPTOR.online_decoder
            and not POSTSELECTION_DESCRIPTOR.primary_metric_eligible
        ),
    }
    failed = [name for name, passed in gates.items() if not passed]
    payload: dict[str, object] = {
        "schema_version": "t3.2.4-postselection-diagnostic-v1",
        "task_id": "T3.2.4",
        "status": "PASS" if not failed else "FAIL",
        "implementation_sha256": _implementation_sha256(),
        "descriptor": asdict(POSTSELECTION_DESCRIPTOR),
        "validation_config": asdict(settings),
        "static_training_parameters": asdict(static_parameters),
        "training_calibration": {
            "training_trace_sha256": training_trace_sha256,
            "training_samples": int(training_scores.size),
            "thresholds": [
                {"target_survival": target, "score_threshold": threshold}
                for target, threshold in thresholds
            ],
            "evaluation_truth_used": False,
        },
        "scenario_survival_summaries": summaries,
        "aggregate_by_target_survival": aggregate_by_target,
        "aggregate": {
            "scenarios": len(continuous_drift_scenarios()),
            "evaluation_seeds_per_scenario": len(settings.evaluation_seeds),
            "evaluation_samples": len(continuous_drift_scenarios())
            * len(settings.evaluation_seeds)
            * settings.windows
            * settings.evaluation_samples_per_window,
            "source_data_rows": len(rows),
        },
        "gate_summary": {
            "passed": sum(bool(value) for value in gates.values()),
            "failed": len(failed),
            "gates": gates,
        },
        "claim_boundary": {
            "allowed": (
                "offline evidence that static-MAP posterior confidence contains failure information, "
                "with survival, rejection and penalty costs reported"
            ),
            "forbidden": (
                "online correction gain, postselected break-even, free rejection, truth-score deployment, "
                "device calibration, or FPGA synthesis/measurement"
            ),
        },
    }
    return payload, rows


def write_postselection_validation(
    json_path: str | Path = "docs/t3_2_4_postselection_validation.json",
    csv_path: str | Path = "docs/t3_2_4_postselection_source_data.csv",
    config: PostselectionValidationConfig | None = None,
) -> dict[str, object]:
    payload, rows = build_postselection_validation(config)
    json_target = Path(json_path)
    csv_target = Path(csv_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    csv_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not rows:
        raise RuntimeError("post-selection validation produced no source rows")
    with csv_target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return payload


def main() -> int:
    payload = write_postselection_validation()
    summary = payload["gate_summary"]
    print(json.dumps({"passed": summary["passed"], "failed": summary["failed"], "gates": summary["gates"]}))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PostselectionDescriptor",
    "POSTSELECTION_DESCRIPTOR",
    "PostselectionValidationConfig",
    "build_postselection_validation",
    "write_postselection_validation",
]
