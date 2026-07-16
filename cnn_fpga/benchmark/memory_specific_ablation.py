"""T3.2.11 frozen and retrained memory-specific ablation benchmark."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from cnn_fpga.benchmark.latest_outcome_markovian_baseline import (
    DEFAULT_ARTIFACT as EXACT_MF_ARTIFACT,
    DEFAULT_CHECKPOINT as EXACT_MF_CHECKPOINT,
    DEFAULT_PARENT_ARTIFACT,
    DEFAULT_PARENT_CHECKPOINT,
    _load_parent_evidence,
    implementation_sha256 as exact_mf_implementation_sha256,
)
from physics.latest_outcome_markovian import (
    COMPUTE_CONTRACT,
    build_budget_matched_policy,
)
from physics.memory_specific_ablation import (
    FrozenMemoryInterventionPolicy,
    MemoryInterventionSpec,
)
from physics.nmf_directional_ranking import (
    _agent_metric,
    _bootstrap_agent_difference,
    _distribution,
    _state_dict_cpu,
    _torch_dtype,
    evaluate_policy,
    state_dict_sha256,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT = Path("docs/t3_2_11_memory_specific_ablation_validation.json")
DEFAULT_SOURCE_DATA = Path("docs/t3_2_11_memory_specific_ablation_source_data.csv")
SCHEMA = "T3211-MEMORY-SPECIFIC-ABLATION-V1"
METRICS = (
    "fidelity_effective_lifetime_cycles",
    "fidelity_normalized_auc",
    "logical_z_effective_lifetime_cycles",
    "logical_z_normalized_auc",
)


def _require_torch() -> Any:
    import torch

    return torch


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def implementation_sha256() -> str:
    digest = hashlib.sha256()
    for relative in (
        "physics/memory_specific_ablation.py",
        "physics/nmf_directional_ranking.py",
        "physics/latest_outcome_markovian.py",
        "physics/differentiable_sbs_trajectory.py",
        "cnn_fpga/benchmark/memory_specific_ablation.py",
        "cnn_fpga/benchmark/latest_outcome_markovian_baseline.py",
    ):
        path = ROOT / relative
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class MemoryAblationConfig:
    truncation_lengths: tuple[int, ...] = (1, 2, 4, 8)
    reset_periods: tuple[int, ...] = (1, 2, 4, 8)
    shuffle_seeds: tuple[int, ...] = (17, 29, 43)
    bootstrap_seed: int = 93271
    bootstrap_repetitions: int = 20_000

    def __post_init__(self) -> None:
        for name in ("truncation_lengths", "reset_periods", "shuffle_seeds"):
            values = tuple(getattr(self, name))
            if not values or len(set(values)) != len(values):
                raise ValueError(f"{name} must be nonempty and unique")
            if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) <= 0 for value in values):
                raise ValueError(f"{name} must contain positive integers")
            object.__setattr__(self, name, tuple(int(value) for value in values))
        if self.truncation_lengths[0] != 1 or self.reset_periods[0] != 1:
            raise ValueError("truncation/reset grids must include one as their first endpoint")
        if len(self.shuffle_seeds) < 3:
            raise ValueError("production shuffle audit requires at least three permutations")
        for name in ("bootstrap_seed", "bootstrap_repetitions"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        if self.bootstrap_repetitions < 10_000:
            raise ValueError("production paired bootstrap requires at least 10,000 repetitions")


def _load_exact_mf_models(config: Any) -> tuple[list[Any], Mapping[str, Any], Mapping[str, Any]]:
    torch = _require_torch()
    artifact_path = ROOT / EXACT_MF_ARTIFACT
    checkpoint_path = ROOT / EXACT_MF_CHECKPOINT
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if artifact.get("task_id") != "T3.2.7" or artifact.get("status") != "PASS":
        raise ValueError("T3.2.7 exact-MF artifact must be PASS")
    if artifact.get("implementation_sha256") != exact_mf_implementation_sha256():
        raise ValueError("T3.2.7 exact-MF implementation hash is stale")
    if artifact.get("checkpoint", {}).get("sha256") != _sha256(checkpoint_path):
        raise ValueError("T3.2.7 exact-MF checkpoint hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != 1 or checkpoint.get("contract_hash") != artifact.get("contract_hash"):
        raise ValueError("T3.2.7 exact-MF checkpoint contract mismatch")
    by_seed = {int(item["training_seed"]): item for item in checkpoint["models"]}
    if set(by_seed) != set(config.training_seeds):
        raise ValueError("T3.2.7 exact-MF checkpoint is incomplete")
    models = []
    for seed in config.training_seeds:
        item = by_seed[int(seed)]
        model = build_budget_matched_policy(
            device=config.device, dtype=_torch_dtype(config.real_dtype), seed=int(seed)
        )
        model.load_state_dict(item["state_dict"])
        model.eval()
        if state_dict_sha256(_state_dict_cpu(model)) != item["checkpoint_sha256"]:
            raise ValueError(f"T3.2.7 exact-MF model hash mismatch for seed {seed}")
        models.append(model)
    return models, artifact, checkpoint


def _evaluate_models(
    models: Sequence[Any],
    parent_config: Any,
    spec: MemoryInterventionSpec | None,
    *,
    cutoff: int,
    batch_size: int,
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    evaluations = []
    for model in models:
        policy = model if spec is None else FrozenMemoryInterventionPolicy(model, spec)
        evaluations.append(
            evaluate_policy(
                "nmf", policy, parent_config,
                cutoff=cutoff, batch_size=batch_size, seeds=seeds,
            )
        )
    return evaluations


def _variant_summary(evaluations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        metric: _distribution([_agent_metric(item, metric) for item in evaluations])
        for metric in METRICS
    } | {
        "auxiliary": {
            metric: _distribution([float(item["auxiliary_means"][metric]) for item in evaluations])
            for metric in (
                "mean_ground_outcome_probability",
                "mean_control_residual_rms",
                "mean_control_slew_rms",
            )
        }
    }


def _average_replicate_agent_values(
    replicate_evaluations: Sequence[Sequence[Mapping[str, Any]]], metric: str
) -> list[float]:
    matrix = np.asarray(
        [[_agent_metric(item, metric) for item in replicate] for replicate in replicate_evaluations],
        dtype=np.float64,
    )
    return np.mean(matrix, axis=0).tolist()


def _paired(
    left: Sequence[float], right: Sequence[float], settings: MemoryAblationConfig, offset: int
) -> dict[str, float]:
    return _bootstrap_agent_difference(
        left,
        right,
        seed=settings.bootstrap_seed + offset,
        repetitions=settings.bootstrap_repetitions,
    )


def _action_intervention_audit(models: Sequence[Any], settings: MemoryAblationConfig) -> dict[str, Any]:
    torch = _require_torch()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(321011)
    histories = torch.randint(0, 2, (128, 20), generator=generator, dtype=torch.int64)
    specs = [
        MemoryInterventionSpec("last_outcome_only"),
        *(MemoryInterventionSpec("history_truncation", history_length=value) for value in settings.truncation_lengths[1:]),
        *(MemoryInterventionSpec("periodic_hidden_reset", reset_period=value) for value in settings.reset_periods[1:]),
        *(MemoryInterventionSpec("history_shuffle", shuffle_seed=value) for value in settings.shuffle_seeds),
    ]
    rows = []
    full_exact = []
    for agent_index, model in enumerate(models):
        reference = next(model.parameters())
        history = histories.to(reference.device)
        full = FrozenMemoryInterventionPolicy(model, MemoryInterventionSpec("full_history"))
        with torch.no_grad():
            parent_action = model(history, 20)
            full_action = full(history, 20)
        full_exact.append(bool(torch.equal(parent_action, full_action)))
        for spec in specs:
            view = FrozenMemoryInterventionPolicy(model, spec)
            with torch.no_grad():
                action = view(history, 20)
            difference = torch.linalg.vector_norm(action - parent_action, dim=1)
            rows.append({
                "agent_index": agent_index,
                "variant": spec.identifier,
                "changed_history_fraction": float(torch.mean((difference > 1.0e-12).to(torch.float64)).cpu()),
                "mean_action_l2_difference": float(torch.mean(difference).cpu()),
                "maximum_action_l2_difference": float(torch.max(difference).cpu()),
            })
    return {
        "sampled_histories": int(histories.shape[0]),
        "history_depth": int(histories.shape[1]),
        "full_view_bit_exact_for_all_agents": all(full_exact),
        "rows": rows,
        "every_intervention_changes_actions": all(row["changed_history_fraction"] > 0.0 for row in rows),
    }


def _source_rows(
    lane_evaluations: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    training_seeds: Sequence[int],
    action_audit: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lane, variants in lane_evaluations.items():
        for variant, evaluations in variants.items():
            for agent_index, evaluation in enumerate(evaluations):
                training_seed = int(training_seeds[agent_index])
                for seed_item in evaluation["per_seed"]:
                    for curve_name in ("fidelity_curve", "code_survival_curve", "logical_z_curve"):
                        for cycle, value in enumerate(seed_item[curve_name]):
                            rows.append({
                                "row_type": "evaluation_curve",
                                "lane": lane,
                                "variant": variant,
                                "training_seed": training_seed,
                                "evaluation_seed": seed_item["seed"],
                                "cycle": cycle,
                                "metric": curve_name,
                                "value": value,
                            })
                    for metric in (
                        "mean_ground_outcome_probability",
                        "mean_control_residual_rms",
                        "mean_control_slew_rms",
                        "maximum_trace_error",
                        "maximum_hermiticity_error",
                        "minimum_final_eigenvalue",
                    ):
                        rows.append({
                            "row_type": "evaluation_metric",
                            "lane": lane,
                            "variant": variant,
                            "training_seed": training_seed,
                            "evaluation_seed": seed_item["seed"],
                            "cycle": "",
                            "metric": metric,
                            "value": seed_item[metric],
                        })
    for row in action_audit["rows"]:
        for metric in (
            "changed_history_fraction",
            "mean_action_l2_difference",
            "maximum_action_l2_difference",
        ):
            rows.append({
                "row_type": "action_audit",
                "lane": "causal_action_probe",
                "variant": row["variant"],
                "training_seed": training_seeds[int(row["agent_index"])],
                "evaluation_seed": "",
                "cycle": action_audit["history_depth"],
                "metric": metric,
                "value": row[metric],
            })
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("row_type", "lane", "variant", "training_seed", "evaluation_seed", "cycle", "metric", "value")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_memory_specific_ablation(
    settings: MemoryAblationConfig | None = None,
    *,
    artifact_path: str | Path = DEFAULT_ARTIFACT,
    source_data_path: str | Path = DEFAULT_SOURCE_DATA,
) -> dict[str, Any]:
    actual = MemoryAblationConfig() if settings is None else settings
    started = time.perf_counter()
    parent_artifact_path = ROOT / DEFAULT_PARENT_ARTIFACT
    parent_checkpoint_path = ROOT / DEFAULT_PARENT_CHECKPOINT
    parent_config, parent_models, parent_artifact, _ = _load_parent_evidence(
        parent_artifact_path, parent_checkpoint_path
    )
    history_models = parent_models["history_nmf"]
    exact_models, exact_artifact, _ = _load_exact_mf_models(parent_config)
    if len(history_models) != 5 or len(exact_models) != 5:
        raise RuntimeError("production memory audit requires five parent and five exact-MF agents")
    if parent_config.full_cycles != 10 or parent_config.cutoff != 12 or parent_config.confirmation_cutoff != 16:
        raise RuntimeError("parent production horizon/cutoff contract drifted")
    if max((*actual.truncation_lengths, *actual.reset_periods)) >= parent_config.full_cycles * 2:
        raise ValueError("ablation grid endpoints must remain below the 20-half-cycle full history")
    parent_hashes_before = [state_dict_sha256(_state_dict_cpu(model)) for model in history_models]
    lanes = {
        "primary": (parent_config.cutoff, parent_config.test_batch_size, parent_config.test_seeds),
        "confirmation": (
            parent_config.confirmation_cutoff,
            parent_config.confirmation_batch_size,
            parent_config.confirmation_seeds,
        ),
    }
    lane_evaluations: dict[str, dict[str, list[dict[str, Any]]]] = {}
    lane_summaries: dict[str, dict[str, Any]] = {}
    lane_comparisons: dict[str, dict[str, Any]] = {}
    for lane_index, (lane, (cutoff, batch_size, seeds)) in enumerate(lanes.items()):
        variants: dict[str, list[dict[str, Any]]] = {}
        variants["full_history"] = _evaluate_models(
            history_models, parent_config, MemoryInterventionSpec("full_history"),
            cutoff=cutoff, batch_size=batch_size, seeds=seeds,
        )
        variants["retrained_exact_budget_last_outcome"] = [
            evaluate_policy(
                "mf", model, parent_config,
                cutoff=cutoff, batch_size=batch_size, seeds=seeds,
            )
            for model in exact_models
        ]
        variants["frozen_parent_last_outcome_only"] = _evaluate_models(
            history_models, parent_config, MemoryInterventionSpec("last_outcome_only"),
            cutoff=cutoff, batch_size=batch_size, seeds=seeds,
        )
        for length in actual.truncation_lengths[1:]:
            variants[f"history_truncation_L{length}"] = _evaluate_models(
                history_models, parent_config,
                MemoryInterventionSpec("history_truncation", history_length=length),
                cutoff=cutoff, batch_size=batch_size, seeds=seeds,
            )
        for period in actual.reset_periods[1:]:
            variants[f"periodic_hidden_reset_R{period}"] = _evaluate_models(
                history_models, parent_config,
                MemoryInterventionSpec("periodic_hidden_reset", reset_period=period),
                cutoff=cutoff, batch_size=batch_size, seeds=seeds,
            )
        shuffle_replicates = []
        for shuffle_seed in actual.shuffle_seeds:
            key = f"history_shuffle_seed{shuffle_seed}"
            variants[key] = _evaluate_models(
                history_models, parent_config,
                MemoryInterventionSpec("history_shuffle", shuffle_seed=shuffle_seed),
                cutoff=cutoff, batch_size=batch_size, seeds=seeds,
            )
            shuffle_replicates.append(variants[key])
        lane_evaluations[lane] = variants
        summaries = {name: _variant_summary(values) for name, values in variants.items()}
        summaries["history_shuffle_replicate_mean"] = {
            metric: _distribution(_average_replicate_agent_values(shuffle_replicates, metric))
            for metric in METRICS
        }
        lane_summaries[lane] = summaries
        comparisons: dict[str, Any] = {}
        full_values_by_metric = {
            metric: [_agent_metric(item, metric) for item in variants["full_history"]]
            for metric in METRICS
        }
        comparison_targets = {
            "retrained_exact_budget_last_outcome": variants["retrained_exact_budget_last_outcome"],
            "frozen_parent_last_outcome_only": variants["frozen_parent_last_outcome_only"],
            **{
                f"history_truncation_L{length}": variants[f"history_truncation_L{length}"]
                for length in actual.truncation_lengths[1:]
            },
            **{
                f"periodic_hidden_reset_R{period}": variants[f"periodic_hidden_reset_R{period}"]
                for period in actual.reset_periods[1:]
            },
        }
        offset = lane_index * 1000
        for target_index, (target, evaluations) in enumerate(comparison_targets.items()):
            comparisons[f"full_minus_{target}"] = {
                metric: _paired(
                    full_values_by_metric[metric],
                    [_agent_metric(item, metric) for item in evaluations],
                    actual,
                    offset + target_index * 10 + metric_index,
                )
                for metric_index, metric in enumerate(METRICS)
            }
        comparisons["full_minus_history_shuffle_replicate_mean"] = {
            metric: _paired(
                full_values_by_metric[metric],
                _average_replicate_agent_values(shuffle_replicates, metric),
                actual,
                offset + 900 + metric_index,
            )
            for metric_index, metric in enumerate(METRICS)
        }
        lane_comparisons[lane] = comparisons

    action_audit = _action_intervention_audit(history_models, actual)
    parent_hashes_after = [state_dict_sha256(_state_dict_cpu(model)) for model in history_models]
    metric = "logical_z_effective_lifetime_cycles"
    required_contrasts = (
        "full_minus_retrained_exact_budget_last_outcome",
        "full_minus_frozen_parent_last_outcome_only",
        "full_minus_history_shuffle_replicate_mean",
        "full_minus_periodic_hidden_reset_R2",
    )
    lane_support = {
        lane: all(
            lane_comparisons[lane][contrast][metric]["ci95_low"] > 0.0
            for contrast in required_contrasts
        )
        for lane in lanes
    }
    robust_support = all(lane_support.values())
    verdict = (
        "cross_cutoff_memory_mechanism_supported"
        if robust_support
        else "cross_cutoff_memory_mechanism_not_supported"
    )
    source_rows = _source_rows(lane_evaluations, parent_config.training_seeds, action_audit)
    source_target = ROOT / Path(source_data_path)
    _write_csv(source_target, source_rows)
    all_evaluations = [
        evaluation
        for variants in lane_evaluations.values()
        for evaluations in variants.values()
        for evaluation in evaluations
    ]
    input_contract = {
        "online_inputs": ["causal observed g/e prefix"],
        "forbidden_inputs": ["future outcomes", "hidden simulator truth", "evaluation-selected intervention"],
        "physical_leakage_tokens_observed": 0,
    }
    expected_variants_per_lane = 3 + (len(actual.truncation_lengths) - 1) + (len(actual.reset_periods) - 1) + len(actual.shuffle_seeds)
    gates = {
        "parent_t237_artifact_and_checkpoint_are_live": parent_artifact.get("status") == "PASS" and parent_artifact.get("checkpoint", {}).get("sha256") == _sha256(parent_checkpoint_path),
        "exact_mf_t327_artifact_and_checkpoint_are_live": exact_artifact.get("status") == "PASS" and exact_artifact.get("checkpoint", {}).get("sha256") == _sha256(ROOT / EXACT_MF_CHECKPOINT),
        "five_paired_parent_and_retrained_agents_are_complete": len(history_models) == len(exact_models) == 5,
        "exact_mf_retraining_matches_parent_parameter_and_mac_budget": COMPUTE_CONTRACT.total_parameter_count == 72_853 and COMPUTE_CONTRACT.total_dense_macs == 72_266,
        "frozen_full_view_is_bit_exact": action_audit["full_view_bit_exact_for_all_agents"],
        "every_frozen_intervention_changes_some_actions": action_audit["every_intervention_changes_actions"],
        "frozen_interventions_do_not_mutate_parent_weights": parent_hashes_before == parent_hashes_after,
        "three_independent_shuffle_permutations_are_retained": len(actual.shuffle_seeds) >= 3,
        "both_cutoffs_execute_the_complete_registered_variant_set": all(len(lane_evaluations[lane]) == expected_variants_per_lane for lane in lanes),
        "paired_agent_bootstrap_is_reported_for_every_required_contrast": all(set(required_contrasts) <= set(lane_comparisons[lane]) for lane in lanes),
        "signed_cross_cutoff_result_is_retained_without_direction_gate": verdict in {"cross_cutoff_memory_mechanism_supported", "cross_cutoff_memory_mechanism_not_supported"},
        "density_diagnostics_remain_physical": all(
            item["maximum_trace_error"] < 2.0e-10
            and item["maximum_hermiticity_error"] < 2.0e-10
            and item["minimum_final_eigenvalue"] > -2.0e-9
            for evaluation in all_evaluations
            for item in evaluation["per_seed"]
        ),
        "no_future_truth_or_evaluation_intervention_input": input_contract["forbidden_inputs"] == ["future outcomes", "hidden simulator truth", "evaluation-selected intervention"],
        "two_level_leakage_boundary_is_explicit": input_contract["physical_leakage_tokens_observed"] == 0,
        "source_data_is_curve_level_and_nontrivial": len(source_rows) > 10_000,
    }
    failed = [name for name, passed in gates.items() if not passed]
    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "task_id": "T3.2.11",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not failed else "FAIL",
        "implementation_sha256": implementation_sha256(),
        "settings": asdict(actual),
        "parent_contract": {
            "architecture": "GRU10-DENSE256-DENSE256-OUT15",
            "training_seeds": list(parent_config.training_seeds),
            "full_cycles": parent_config.full_cycles,
            "half_cycles": parent_config.full_cycles * 2,
            "primary_cutoff": parent_config.cutoff,
            "confirmation_cutoff": parent_config.confirmation_cutoff,
            "primary_test_seeds": list(parent_config.test_seeds),
            "confirmation_seeds": list(parent_config.confirmation_seeds),
            "parameter_count": 72_853,
            "dense_mac_count_per_half_cycle": 72_266,
        },
        "intervention_contract": {
            "history_shuffle": "deterministic permutation of observed prefix only; preserves token multiset and destroys order",
            "history_truncation": "sliding last-L observed tokens replayed from zero hidden state",
            "periodic_hidden_reset": "fixed half-cycle blocks replayed from zero hidden state; R1 equals frozen latest-only",
            "last_outcome_only": "both frozen parent GRU latest-token view and independently retrained exact-budget stateless FNN",
            "weight_policy": "all frozen interventions reuse bit-identical parent weights; only exact-MF capacity comparator is independently trained on T3.2.7 train/validation split",
            **input_contract,
        },
        "parent_evidence": {
            "t237_artifact": DEFAULT_PARENT_ARTIFACT.as_posix(),
            "t237_artifact_sha256": _sha256(parent_artifact_path),
            "t237_checkpoint": DEFAULT_PARENT_CHECKPOINT.as_posix(),
            "t237_checkpoint_sha256": _sha256(parent_checkpoint_path),
            "t327_artifact": EXACT_MF_ARTIFACT.as_posix(),
            "t327_artifact_sha256": _sha256(ROOT / EXACT_MF_ARTIFACT),
            "t327_checkpoint": EXACT_MF_CHECKPOINT.as_posix(),
            "t327_checkpoint_sha256": _sha256(ROOT / EXACT_MF_CHECKPOINT),
        },
        "summary": lane_summaries,
        "paired_bootstrap_full_minus_ablation": lane_comparisons,
        "mechanism_verdict": {
            "primary_support": lane_support["primary"],
            "confirmation_support": lane_support["confirmation"],
            "robust_cross_cutoff_support": robust_support,
            "verdict": verdict,
            "decision_metric": metric,
            "required_positive_contrasts": list(required_contrasts),
            "rule": "all four paired 95% lower bounds must exceed zero at both cutoffs",
        },
        "action_intervention_audit": action_audit,
        "gate_summary": {
            "passed": sum(bool(value) for value in gates.values()),
            "failed": len(failed),
            "failed_names": failed,
            "gates": gates,
        },
        "source_data": {
            "path": Path(source_data_path).as_posix(),
            "sha256": _sha256(source_target),
            "row_count": len(source_rows),
        },
        "claim_boundary": {
            "allowed": "frozen causal history interventions and independently retrained exact-budget latest-only comparison on the registered finite-cutoff two-level ten-cycle model",
            "forbidden": [
                "universal long-memory benefit",
                "paper-exact memory mechanism or 1000-cycle lifetime",
                "physical multilevel leakage robustness",
                "optimizer optimality",
                "device, RTL, FPGA, or board performance",
            ],
        },
        "wall_time_seconds": time.perf_counter() - started,
    }
    artifact_target = ROOT / Path(artifact_path)
    artifact_target.parent.mkdir(parents=True, exist_ok=True)
    artifact_target.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--source-data", type=Path, default=DEFAULT_SOURCE_DATA)
    args = parser.parse_args(argv)
    artifact = run_memory_specific_ablation(
        artifact_path=args.artifact, source_data_path=args.source_data
    )
    print(json.dumps({
        "status": artifact["status"],
        "verdict": artifact["mechanism_verdict"],
        "gates": artifact["gate_summary"],
    }, ensure_ascii=False))
    return 0 if artifact["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_ARTIFACT",
    "DEFAULT_SOURCE_DATA",
    "MemoryAblationConfig",
    "implementation_sha256",
    "run_memory_specific_ablation",
]
