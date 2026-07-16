"""T4.1.4 strict-split multi-objective calibration benchmark.

The benchmark binds the T4.1.3 future-only outputs to the next 32 cycles of
the registered synthetic syndrome stream.  Future observations and simulator
truth are used only in this offline evaluator.  The deployable history/output
contracts remain truth-free.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from cnn_fpga.benchmark.experimental_history_validation import _make_stream
from cnn_fpga.decoder.hybrid_multiobjective import (
    OBJECTIVE_NAMES,
    CalibrationRecord,
    FrozenCalibration,
    MultiObjectiveWeights,
    calibration_manifest,
    evaluate_multiobjective_loss,
    fit_training_normalizers,
    fit_validation_calibration,
    score_calibration_records,
)
from cnn_fpga.decoder.hybrid_state_output import CONTINUOUS_PARAMETER_NAMES
from cnn_fpga.decoder.periodic_adaptive_map import PeriodicMomentConfig, estimate_periodic_gaussian
from cnn_fpga.decoder.regime_hmm import REGIME_CLASSES
from physics.constants import LATTICE_CONST
from physics.ideal_gkp_decoder import map_decode_2d
from physics.oracle_map import oracle_map_2d


@dataclass(frozen=True)
class HybridMultiObjectiveValidationConfig:
    source_csv_path: str = "docs/t4_1_3_hybrid_state_output_source_data.csv"
    source_manifest_path: str = "docs/t4_1_3_hybrid_state_output_validation.json"
    cycles_per_seed: int = 2048
    future_horizon_cycles: int = 32
    training_seed_count: int = 3
    validation_seed_count: int = 2
    minimum_unsafe_recall: float = 0.90
    uncertainty_floor_continuous: float = 1.0e-4
    uncertainty_floor_rate: float = 1.0 / 258.0

    def __post_init__(self) -> None:
        for name, minimum in (
            ("cycles_per_seed", 512),
            ("future_horizon_cycles", 8),
            ("training_seed_count", 2),
            ("validation_seed_count", 1),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer at least {minimum}")
        if self.cycles_per_seed % self.future_horizon_cycles:
            raise ValueError("cycles_per_seed must be divisible by future_horizon_cycles")
        if self.training_seed_count + self.validation_seed_count >= 8:
            raise ValueError("the registered eight seeds must leave evaluation seeds")
        if not 0.0 < float(self.minimum_unsafe_recall) <= 1.0:
            raise ValueError("minimum_unsafe_recall must lie in (0, 1]")
        for name in ("uncertainty_floor_continuous", "uncertainty_floor_rate"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("T4.1.3 Source Data is empty")
    required = {
        "seed",
        "as_of_cycle",
        "bank_action",
        "ood_score",
        "recommendation_confidence",
        "leakage_probability_horizon",
        "gain_qq",
        "gain_qp",
        "gain_pq",
        "gain_pp",
        "bias_q",
        "bias_p",
        *CONTINUOUS_PARAMETER_NAMES,
        *(f"continuous_se_{name}" for name in CONTINUOUS_PARAMETER_NAMES),
        *(f"regime_p_{name}" for name in REGIME_CLASSES),
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"T4.1.3 Source Data is missing {missing}")
    return rows


def _future_target(steps) -> tuple[float, ...]:
    residual = np.asarray([step.observed.residual_syndrome for step in steps], dtype=np.float64)
    estimate = estimate_periodic_gaussian(
        residual,
        PeriodicMomentConfig(minimum_samples=min(16, len(residual))),
        source="offline_future_observed_target",
        window_id=-1,
    )
    covariance = estimate.covariance_array()
    sigma_q = float(np.sqrt(covariance[0, 0]))
    sigma_p = float(np.sqrt(covariance[1, 1]))
    rho = float(covariance[0, 1] / (sigma_q * sigma_p))
    x_e = np.asarray([step.observed.syndrome.x == "e" for step in steps], dtype=np.float64)
    z_e = np.asarray([step.observed.syndrome.z == "e" for step in steps], dtype=np.float64)
    leakage = np.asarray(
        [
            step.observed.syndrome.x == "leakage" or step.observed.syndrome.z == "leakage"
            for step in steps
        ],
        dtype=np.float64,
    )
    return (
        float(estimate.mean[0]),
        float(estimate.mean[1]),
        sigma_q,
        sigma_p,
        float(np.clip(rho, -0.999999, 0.999999)),
        float(np.mean(np.max(np.abs(residual), axis=1) >= 0.35 * LATTICE_CONST)),
        float(np.mean(x_e)),
        float(np.mean(z_e)),
        float(np.mean(leakage)),
    )


def _regime_label(previous_step, future_steps) -> str:
    previous_regime = previous_step.truth.hidden_regime
    if any(step.truth.hidden_regime != previous_regime for step in future_steps):
        return "calibration_shift"
    leakage_fraction = np.mean([step.truth.leakage_kind != "none" for step in future_steps])
    if leakage_fraction >= 0.10:
        return "leakage"
    burst_fraction = np.mean([step.truth.burst_active for step in future_steps])
    if burst_fraction >= 0.25:
        return "burst"
    return "normal"


def _paired_oracle_counts(row: dict[str, str], future_steps) -> tuple[int, int, int]:
    sigma_q = float(row["sigma_q"])
    sigma_p = float(row["sigma_p"])
    rho = float(row["rho_qp"])
    covariance = np.asarray(
        [
            [sigma_q * sigma_q, rho * sigma_q * sigma_p],
            [rho * sigma_q * sigma_p, sigma_p * sigma_p],
        ],
        dtype=np.float64,
    )
    eligible = [step for step in future_steps if step.truth.leakage_kind == "none"]
    if not eligible:
        raise RuntimeError("future horizon contains no nonleakage oracle trials")
    syndrome = np.asarray([step.observed.residual_syndrome for step in eligible], dtype=np.float64)
    candidate = np.asarray(
        map_decode_2d(
            syndrome,
            covariance,
            mean=(float(row["mean_q"]), float(row["mean_p"])),
        ).logical_class,
        dtype=np.int64,
    )
    oracle = np.asarray(
        [
            int(oracle_map_2d(step.observed.residual_syndrome, step.truth.drift_state).logical_class)
            for step in eligible
        ],
        dtype=np.int64,
    )
    truth = np.asarray(
        [2 * step.truth.logical_increment[0] + step.truth.logical_increment[1] for step in eligible],
        dtype=np.int64,
    )
    return int(np.sum(candidate != truth)), int(np.sum(oracle != truth)), len(eligible)


def _fallback_required(future_steps, candidate_failures: int, oracle_failures: int, trials: int) -> bool:
    # A single nonzero recovery-depth sample is normal protocol activity in
    # this stream and must not label every horizon unsafe.  Require persistent
    # higher leakage or a high *mean* recovery burden instead.
    higher_fraction = float(
        np.mean([step.truth.leakage_kind == "higher" for step in future_steps])
    )
    mean_recovery_burden = float(
        np.mean(
            [
                max(
                    step.truth.recovery_depth_before_action,
                    step.truth.recovery_depth_after_action,
                )
                for step in future_steps
            ]
        )
    )
    truth_risk = higher_fraction >= 2.0 / len(future_steps) or mean_recovery_burden >= 2.5
    model_risk = candidate_failures - oracle_failures >= max(3, int(np.ceil(0.10 * trials)))
    return bool(truth_risk or model_risk)


def _split_for_seed(
    seed: int,
    seeds: Sequence[int],
    config: HybridMultiObjectiveValidationConfig,
) -> str:
    index = tuple(seeds).index(seed)
    if index < config.training_seed_count:
        return "training"
    if index < config.training_seed_count + config.validation_seed_count:
        return "validation"
    return "evaluation"


def _implementation_sha256() -> str:
    digest = hashlib.sha256()
    for path in (
        "cnn_fpga/decoder/hybrid_multiobjective.py",
        "cnn_fpga/benchmark/hybrid_multiobjective_calibration.py",
        "cnn_fpga/decoder/hybrid_state_output.py",
        "cnn_fpga/decoder/periodic_adaptive_map.py",
        "physics/oracle_map.py",
        "physics/syndrome_stream.py",
    ):
        digest.update(path.encode("utf-8"))
        digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def build_hybrid_multiobjective_validation(
    config: HybridMultiObjectiveValidationConfig | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    settings = HybridMultiObjectiveValidationConfig() if config is None else config
    if not isinstance(settings, HybridMultiObjectiveValidationConfig):
        raise TypeError("config must be HybridMultiObjectiveValidationConfig")
    source_rows = _load_rows(settings.source_csv_path)
    source_manifest = json.loads(Path(settings.source_manifest_path).read_text(encoding="utf-8"))
    seeds = tuple(sorted({int(row["seed"]) for row in source_rows}))
    if len(seeds) != 8:
        raise ValueError("T4.1.4 requires exactly the registered eight T4.1.3 seeds")
    streams = {seed: _make_stream(seed, settings.cycles_per_seed) for seed in seeds}
    previous_parameters: dict[int, np.ndarray] = {}
    records: list[CalibrationRecord] = []
    rows: list[dict[str, object]] = []

    for source in sorted(source_rows, key=lambda item: (int(item["seed"]), int(item["as_of_cycle"]))):
        seed = int(source["seed"])
        as_of_cycle = int(source["as_of_cycle"])
        future_end = as_of_cycle + 1 + settings.future_horizon_cycles
        if future_end > settings.cycles_per_seed:
            continue
        future_steps = streams[seed].steps[as_of_cycle + 1 : future_end]
        if len(future_steps) != settings.future_horizon_cycles:
            raise RuntimeError("future alignment produced a short horizon")
        prediction = tuple(float(source[name]) for name in CONTINUOUS_PARAMETER_NAMES)
        target = _future_target(future_steps)
        standard_errors = []
        for name in CONTINUOUS_PARAMETER_NAMES:
            floor = (
                settings.uncertainty_floor_rate
                if name in {"tail_rate", "x_e_rate", "z_e_rate", "leakage_rate"}
                else settings.uncertainty_floor_continuous
            )
            standard_errors.append(max(float(source[f"continuous_se_{name}"]), floor))
        probabilities = np.asarray(
            [float(source[f"regime_p_{name}"]) for name in REGIME_CLASSES], dtype=np.float64
        )
        probabilities = np.clip(probabilities, 1.0e-15, 1.0)
        probabilities /= np.sum(probabilities)
        candidate_failures, oracle_failures, trials = _paired_oracle_counts(source, future_steps)
        fallback_score = max(
            float(source["ood_score"]),
            float(source["leakage_probability_horizon"]),
            1.0 - float(source["recommendation_confidence"]),
        )
        current_parameters = np.asarray(
            [
                float(source["gain_qq"]),
                float(source["gain_qp"]),
                float(source["gain_pq"]),
                float(source["gain_pp"]),
                float(source["bias_q"]),
                float(source["bias_p"]),
            ],
            dtype=np.float64,
        )
        previous = previous_parameters.get(seed, current_parameters)
        parameter_churn = float(np.mean(np.abs(current_parameters - previous)))
        staged = source["bank_action"] == "stage_candidate"
        # Every output incurs slow-loop scoring/serialization work even when a
        # safety gate holds the active bank.  Atomic staging adds one unit;
        # recommendation churn adds a smaller payload-dependent term.
        update_cost = 0.05 + 0.10 * parameter_churn + (1.0 if staged else 0.0)
        previous_parameters[seed] = current_parameters
        split = _split_for_seed(seed, seeds, settings)
        regime_label = _regime_label(streams[seed].steps[as_of_cycle], future_steps)
        required = _fallback_required(
            future_steps, candidate_failures, oracle_failures, trials
        )
        record = CalibrationRecord(
            record_id=f"seed{seed}-asof{as_of_cycle}",
            split=split,
            seed=seed,
            prediction=prediction,
            target=target,
            uncertainty_standard_errors=tuple(standard_errors),
            regime_probabilities=tuple(float(value) for value in probabilities),
            regime_label=regime_label,
            candidate_failures=candidate_failures,
            oracle_failures=oracle_failures,
            oracle_trials=trials,
            fallback_score=fallback_score,
            fallback_required=required,
            update_cost=update_cost,
        )
        records.append(record)
        output_row: dict[str, object] = {
            "record_id": record.record_id,
            "split": split,
            "seed": seed,
            "as_of_cycle": as_of_cycle,
            "future_start_cycle": as_of_cycle + 1,
            "future_end_cycle_exclusive": future_end,
            "offline_regime_label": regime_label,
            "candidate_failures": candidate_failures,
            "oracle_failures": oracle_failures,
            "oracle_trials": trials,
            "fallback_score": fallback_score,
            "offline_fallback_required": int(required),
            "update_cost": update_cost,
            "source_bank_action": source["bank_action"],
            "scope": record.scope,
        }
        for index, name in enumerate(CONTINUOUS_PARAMETER_NAMES):
            output_row[f"prediction_{name}"] = prediction[index]
            output_row[f"offline_future_target_{name}"] = target[index]
            output_row[f"uncertainty_se_{name}"] = standard_errors[index]
        for index, name in enumerate(REGIME_CLASSES):
            output_row[f"regime_p_{name}"] = probabilities[index]
        rows.append(output_row)

    split_records = {
        split: tuple(record for record in records if record.split == split)
        for split in ("training", "validation", "evaluation")
    }
    normalizers = fit_training_normalizers(split_records["training"])
    calibration = fit_validation_calibration(
        split_records["validation"],
        normalizers,
        minimum_unsafe_recall=settings.minimum_unsafe_recall,
    )
    weights = MultiObjectiveWeights()
    evaluation = evaluate_multiobjective_loss(
        split_records["evaluation"], normalizers, calibration, weights
    )
    validation_calibrated = score_calibration_records(
        split_records["validation"], normalizers, calibration
    )
    identity = FrozenCalibration(
        regime_temperature=1.0,
        regime_uniform_mix=0.0,
        uncertainty_scale=1.0,
        fallback_threshold=0.58,
        minimum_unsafe_recall=settings.minimum_unsafe_recall,
        training_record_ids_sha256=normalizers.training_record_ids_sha256,
        validation_record_ids_sha256=calibration.validation_record_ids_sha256,
        training_seeds=calibration.training_seeds,
        validation_seeds=calibration.validation_seeds,
    )
    validation_identity = score_calibration_records(
        split_records["validation"], normalizers, identity
    )
    evaluation_identity = score_calibration_records(
        split_records["evaluation"], normalizers, identity
    )
    manifest = calibration_manifest(normalizers, calibration, weights)

    split_counts = {split: len(values) for split, values in split_records.items()}
    split_seeds = {
        split: sorted({record.seed for record in values}) for split, values in split_records.items()
    }
    label_counts = {
        split: {
            label: sum(record.regime_label == label for record in values)
            for label in REGIME_CLASSES
        }
        for split, values in split_records.items()
    }
    required_counts = {
        split: sum(record.fallback_required for record in values)
        for split, values in split_records.items()
    }
    source_digest = hashlib.sha256()
    for row in rows:
        source_digest.update(
            (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        )
    validation_cal_diag = validation_calibrated["diagnostics"]
    validation_id_diag = validation_identity["diagnostics"]
    evaluation_values = list(evaluation["raw_objectives"].values()) + list(
        evaluation["diagnostics"].values()
    )
    gates = {
        "registered_t4_1_3_source_manifest_passes": (
            source_manifest.get("status") == "PASS"
            and source_manifest.get("gate_summary", {}).get("failed") == 0
            and source_manifest.get("aggregate", {}).get("outputs") == len(source_rows)
        ),
        "future_alignment_drops_only_terminal_output_per_seed": len(rows) == len(source_rows) - len(seeds),
        "three_two_three_seed_split_is_disjoint": (
            [len(split_seeds[name]) for name in ("training", "validation", "evaluation")] == [3, 2, 3]
            and not (set(split_seeds["training"]) & set(split_seeds["validation"]))
            and not (set(split_seeds["training"]) & set(split_seeds["evaluation"]))
            and not (set(split_seeds["validation"]) & set(split_seeds["evaluation"]))
        ),
        "every_split_has_all_four_regime_targets": all(
            min(counts.values()) > 0 for counts in label_counts.values()
        ),
        "normalizers_are_training_only": normalizers.source_split == "training_only",
        "calibration_is_validation_only": calibration.selection_scope == "validation_only_after_training_normalizers",
        "evaluation_is_not_used_for_selection": evaluation["selection_provenance"]["evaluation_used_for_selection"] is False,
        "validation_temperature_does_not_worsen_nll": (
            validation_cal_diag["regime_nll"] <= validation_id_diag["regime_nll"] + 1.0e-12
        ),
        "validation_uncertainty_scale_does_not_worsen_proper_nll": (
            validation_calibrated["raw_objectives"]["uncertainty_calibration"]
            <= validation_identity["raw_objectives"]["uncertainty_calibration"] + 1.0e-12
        ),
        "validation_fallback_meets_registered_unsafe_recall": (
            validation_cal_diag["required_fallback_recall"] + 1.0e-12
            >= settings.minimum_unsafe_recall
        ),
        "safe_and_required_fallback_targets_exist_in_every_split": all(
            0 < required_counts[split] < split_counts[split] for split in split_counts
        ),
        "all_six_objectives_are_finite": (
            set(evaluation["raw_objectives"]) == set(OBJECTIVE_NAMES)
            and np.all(np.isfinite(evaluation_values))
        ),
        "all_six_leave_one_out_diagnostics_are_reported": (
            set(evaluation["leave_one_objective_out"]) == set(OBJECTIVE_NAMES)
            and all(
                entry["interpretation"]
                == "frozen-output_leave_one_objective_out_not_retrained_causal_ablation"
                for entry in evaluation["leave_one_objective_out"].values()
            )
        ),
        "paired_candidate_oracle_trials_are_nonempty_and_vary": (
            min(record.oracle_trials for record in records) > 0
            and len({(record.candidate_failures, record.oracle_failures) for record in records}) > 8
        ),
        "update_cost_exercises_stage_hold_and_churn": (
            {row["source_bank_action"] for row in rows} == {"stage_candidate", "hold_active"}
            and np.std([record.update_cost for record in records]) > 0.0
            and min(record.update_cost for record in records) > 0.0
        ),
        "offline_truth_never_becomes_deployable_payload": (
            manifest["deployable"] is False
            and manifest["truth_use"] == "offline_targets_and_scores_only"
            and all(row["scope"] == "offline_future_aligned_calibration_record" for row in rows)
        ),
        "calibration_manifest_is_hash_bound": len(str(manifest["manifest_sha256"])) == 64,
        "source_rows_are_unique_and_hash_bound": (
            len({row["record_id"] for row in rows}) == len(rows)
            and len(source_digest.hexdigest()) == 64
        ),
        "scope_remains_synthetic_host_not_hardware": source_manifest.get("output_schema", {}).get("hardware_measured") is False,
    }
    gates = {name: bool(value) for name, value in gates.items()}
    failed = [name for name, passed in gates.items() if not passed]
    payload: dict[str, object] = {
        "schema_version": "t4.1.4-hybrid-multiobjective-validation-v1",
        "task_id": "T4.1.4",
        "status": "PASS" if not failed else "FAIL",
        "implementation_sha256": _implementation_sha256(),
        "source_rows_sha256": source_digest.hexdigest(),
        "input_source": {
            "csv_path": settings.source_csv_path,
            "csv_sha256": _sha256(settings.source_csv_path),
            "manifest_path": settings.source_manifest_path,
            "manifest_sha256": _sha256(settings.source_manifest_path),
        },
        "validation_config": asdict(settings),
        "split_counts": split_counts,
        "split_seeds": split_seeds,
        "regime_target_counts": label_counts,
        "required_fallback_counts": required_counts,
        "calibration_manifest": manifest,
        "validation_identity": validation_identity,
        "validation_calibrated": validation_calibrated,
        "evaluation_identity": evaluation_identity,
        "evaluation_frozen": evaluation,
        "gate_summary": {
            "passed": sum(gates.values()),
            "failed": len(failed),
            "gates": gates,
        },
        "claim_boundary": {
            "allowed": "registered synthetic offline six-objective loss, strict-split calibration, proper-score diagnostics, and frozen-output leave-one-objective-out accounting",
            "forbidden": "evaluation-tuned weights, retrained causal ablation, physical risk calibration, logical/control gain claim, RTL synthesis, FPGA timing, or board/device measurement",
        },
    }
    return payload, rows


def write_hybrid_multiobjective_validation(
    json_path: str | Path = "docs/t4_1_4_hybrid_multiobjective_validation.json",
    csv_path: str | Path = "docs/t4_1_4_hybrid_multiobjective_source_data.csv",
    config: HybridMultiObjectiveValidationConfig | None = None,
) -> dict[str, object]:
    payload, rows = build_hybrid_multiobjective_validation(config)
    if not rows:
        raise RuntimeError("T4.1.4 validation produced no Source Data")
    json_target = Path(json_path)
    csv_target = Path(csv_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    csv_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with csv_target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default="docs/t4_1_4_hybrid_multiobjective_validation.json")
    parser.add_argument("--csv", default="docs/t4_1_4_hybrid_multiobjective_source_data.csv")
    args = parser.parse_args(argv)
    payload = write_hybrid_multiobjective_validation(args.json, args.csv)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "HybridMultiObjectiveValidationConfig",
    "build_hybrid_multiobjective_validation",
    "write_hybrid_multiobjective_validation",
]
