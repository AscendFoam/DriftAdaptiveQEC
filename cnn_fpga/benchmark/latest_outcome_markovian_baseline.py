"""T3.2.7 exact-budget latest-outcome Markovian feedback benchmark.

This runner trains five independent static policies with the *same* seed split,
Feedback-GRAPE objective, 15-dimensional action, parameter count, dense-MAC
budget, validation selection, and held-out evaluation traces as the frozen
T2.3.7 history GRU.  It intentionally does not require a positive memory gain:
the signed paired result is evidence even if the Markovian comparator wins.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from physics.differentiable_sbs_trajectory import PARAMETER_NAMES, nominal_sbs_parameters
from physics.latest_outcome_markovian import (
    ACTION_CONTRACT_ID,
    ARCHITECTURE_ID,
    COMPUTE_CONTRACT,
    OUTCOME_ALPHABET_ID,
    TRAINING_PROTOCOL_ID,
    audit_latest_only_behavior,
    build_budget_matched_policy,
)
from physics.nmf_directional_ranking import (
    DirectionalRankingConfig,
    PAPER_MF_ARCHITECTURE,
    PAPER_RNN_ARCHITECTURE,
    _agent_metric,
    _bootstrap_agent_difference,
    _distribution,
    _simulator,
    _state_dict_cpu,
    _torch_dtype,
    _training_trajectory_seed,
    build_policy,
    evaluate_policy,
    implementation_sha256 as parent_implementation_sha256,
    state_dict_sha256,
    validate_production_design,
)

try:  # Minimal recovery environment deliberately has no torch.
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


TASK_ID = "T3.2.7"
BENCHMARK_CONTRACT_ID = "T327-EXACT-BUDGET-LATEST-OUTCOME-V1"
SCOPE = (
    "finite-cutoff two-level-ancilla 10-cycle comparison of an exact-budget "
    "latest-observation FNN against the frozen T2.3.7 history GRU; the learned "
    "input API supports g/e/leakage, but the production physics lane emits only "
    "g/e; not multilevel leakage, device, FPGA, or optimizer-optimal evidence"
)
DEFAULT_PARENT_ARTIFACT = Path("docs/t2_3_7_nmf_directional_ranking.json")
DEFAULT_PARENT_CHECKPOINT = Path("docs/t2_3_7_nmf_directional_ranking_checkpoints.pt")
DEFAULT_ARTIFACT = Path("docs/t3_2_7_latest_outcome_markovian_validation.json")
DEFAULT_CHECKPOINT = Path("docs/t3_2_7_latest_outcome_markovian_checkpoints.pt")
DEFAULT_SOURCE_DATA = Path("docs/t3_2_7_latest_outcome_markovian_source_data.csv")


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("T3.2.7 requires the local DLEnv PyTorch environment")
    return torch


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def implementation_sha256() -> str:
    digest = hashlib.sha256()
    paths = (
        Path(__file__).resolve(),
        Path(__file__).resolve().parents[2] / "physics" / "latest_outcome_markovian.py",
        Path(__file__).resolve().parents[2] / "physics" / "nmf_directional_ranking.py",
        Path(__file__).resolve().parents[2] / "physics" / "differentiable_sbs_trajectory.py",
    )
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _config_from_parent(payload: Mapping[str, Any]) -> DirectionalRankingConfig:
    raw = dict(payload["config"])
    for key in ("training_seeds", "validation_seeds", "test_seeds", "confirmation_seeds"):
        raw[key] = tuple(int(value) for value in raw[key])
    return DirectionalRankingConfig(**raw)


def _load_parent_evidence(
    artifact_path: Path, checkpoint_path: Path
) -> tuple[DirectionalRankingConfig, dict[str, list[Any]], Mapping[str, Any], Mapping[str, Any]]:
    th = _require_torch()
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if artifact.get("task_id") != "T2.3.7" or artifact.get("status") != "PASS":
        raise ValueError("parent T2.3.7 artifact must be a PASS artifact")
    if artifact.get("implementation_sha256") != parent_implementation_sha256():
        raise ValueError("parent T2.3.7 implementation hash is stale")
    if artifact.get("checkpoint", {}).get("sha256") != _sha256(checkpoint_path):
        raise ValueError("parent T2.3.7 checkpoint hash mismatch")
    config = _config_from_parent(artifact)
    if artifact.get("contract_hash") != config.contract_hash:
        raise ValueError("parent config contract hash mismatch")
    checkpoint = th.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != 3:
        raise ValueError("parent checkpoint must use schema version 3")
    if checkpoint.get("config_contract_hash") != config.contract_hash:
        raise ValueError("parent checkpoint config hash mismatch")
    if checkpoint.get("implementation_sha256") != artifact["implementation_sha256"]:
        raise ValueError("parent checkpoint implementation hash mismatch")

    models: dict[str, list[Any]] = {"legacy_mf": [], "history_nmf": []}
    for source_name, target_name in (("mf", "legacy_mf"), ("nmf", "history_nmf")):
        by_seed = {int(item["training_seed"]): item for item in checkpoint["models"][source_name]}
        if set(by_seed) != set(config.training_seeds):
            raise ValueError(f"parent {source_name} checkpoint is incomplete")
        for seed in config.training_seeds:
            item = by_seed[int(seed)]
            model = build_policy(source_name, config, int(seed))
            model.load_state_dict(item["state_dict"])
            model.to(device=config.device, dtype=_torch_dtype(config.real_dtype))
            model.eval()
            if state_dict_sha256(_state_dict_cpu(model)) != item["checkpoint_sha256"]:
                raise ValueError(f"parent {source_name} model hash mismatch for seed {seed}")
            models[target_name].append(model)
    return config, models, artifact, checkpoint


def _gradient_coverage(model: Any, covered: dict[str, Any]) -> None:
    th = _require_torch()
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        mask = parameter.grad.detach().ne(0).cpu()
        if name not in covered:
            covered[name] = th.zeros_like(mask, dtype=th.bool)
        covered[name] |= mask


def train_agent(seed: int, config: DirectionalRankingConfig) -> tuple[Any, dict[str, Any]]:
    """Train one exact-budget MF agent with the frozen T2.3.7 protocol."""

    th = _require_torch()
    model = build_budget_matched_policy(
        device=config.device, dtype=_torch_dtype(config.real_dtype), seed=int(seed)
    )
    optimizer = th.optim.Adam(model.parameters(), lr=config.learning_rate)
    simulator = _simulator(config, cutoff=config.cutoff, batch_size=config.train_batch_size)
    nominal = nominal_sbs_parameters(device=config.device, dtype=_torch_dtype(config.real_dtype))

    initial = evaluate_policy(
        "mf", model, config, cutoff=config.cutoff,
        batch_size=config.validation_batch_size, seeds=config.validation_seeds,
    )
    best_score = float(initial["selection_score_mean"])
    best_epoch = 0
    best_state = _state_dict_cpu(model)
    validation_history = [{
        "epoch": 0,
        "selection_score": best_score,
        "metric_means": initial["metric_means"],
    }]
    print(json.dumps({"event": "validation", "strategy": "exact_mf", "training_seed": int(seed), "epoch": 0, "selection_score": best_score}), flush=True)

    warmup_seed = _training_trajectory_seed(int(seed), 0)
    with th.no_grad():
        warmup = simulator.run(control_policy=model, seed=warmup_seed, record_cycle_metrics=False)
    baseline = float(warmup.reward.detach().mean().cpu())
    training_curve: list[dict[str, Any]] = []
    covered: dict[str, Any] = {}
    started = time.perf_counter()

    for epoch in range(1, config.train_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        result = simulator.run(
            control_policy=model,
            seed=_training_trajectory_seed(int(seed), epoch),
            record_cycle_metrics=False,
        )
        advantage = result.reward.detach() - baseline
        reward_path = result.reward.mean()
        score_path = th.mean(advantage * result.log_probability)
        residual = result.physical_controls - nominal[None, None, :]
        slew = result.physical_controls[:, 1:, :] - result.physical_controls[:, :-1, :]
        residual_penalty = th.mean(residual**2)
        slew_penalty = th.mean(slew**2)
        loss = (
            -(reward_path + score_path)
            + config.residual_l2_weight * residual_penalty
            + config.slew_l2_weight * slew_penalty
        )
        if not bool(th.isfinite(loss).detach().cpu()):
            raise RuntimeError(f"non-finite exact-MF loss for seed {seed}, epoch {epoch}")
        loss.backward()
        if epoch in {1, config.train_epochs}:
            _gradient_coverage(model, covered)
        gradient_norm = float(
            th.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm).detach().cpu()
        )
        if not np.isfinite(gradient_norm) or gradient_norm <= 0.0:
            raise RuntimeError(f"invalid exact-MF gradient for seed {seed}, epoch {epoch}")
        optimizer.step()
        reward_mean = float(result.reward.detach().mean().cpu())
        baseline = config.score_baseline_decay * baseline + (1.0 - config.score_baseline_decay) * reward_mean
        training_curve.append({
            "epoch": int(epoch),
            "mean_reward": reward_mean,
            "score_path": float(score_path.detach().cpu()),
            "loss": float(loss.detach().cpu()),
            "gradient_norm_before_clip": gradient_norm,
            "baseline": baseline,
            "ground_outcome_fraction": float(th.mean((result.outcomes == 0).to(th.float64)).detach().cpu()),
            "control_residual_rms": float(th.sqrt(residual_penalty).detach().cpu()),
            "control_slew_rms": float(th.sqrt(slew_penalty).detach().cpu()),
        })
        if epoch % config.validation_interval == 0 or epoch == config.train_epochs:
            validation = evaluate_policy(
                "mf", model, config, cutoff=config.cutoff,
                batch_size=config.validation_batch_size, seeds=config.validation_seeds,
            )
            score = float(validation["selection_score_mean"])
            validation_history.append({
                "epoch": int(epoch), "selection_score": score,
                "metric_means": validation["metric_means"],
            })
            print(json.dumps({"event": "validation", "strategy": "exact_mf", "training_seed": int(seed), "epoch": int(epoch), "selection_score": score}), flush=True)
            if score > best_score:
                best_score = score
                best_epoch = int(epoch)
                best_state = _state_dict_cpu(model)

    model.load_state_dict(best_state)
    model.eval()
    total_elements = int(sum(parameter.numel() for parameter in model.parameters()))
    covered_elements = int(sum(int(mask.sum()) for mask in covered.values()))
    uncovered = {
        name: int(mask.numel() - int(mask.sum()))
        for name, mask in covered.items()
        if int(mask.sum()) != mask.numel()
    }
    record = {
        "strategy": "exact_mf",
        "training_seed": int(seed),
        "architecture": ARCHITECTURE_ID,
        "parameter_count": int(model.parameter_count),
        "dense_mac_count": int(model.dense_mac_count),
        "epochs_executed": int(config.train_epochs),
        "training_trajectories": int(config.train_epochs * config.train_batch_size),
        "baseline_warmup_trajectories": int(config.train_batch_size),
        "baseline_initialization": "train-only no-gradient warm-up reward mean",
        "baseline_warmup_seed": int(warmup_seed),
        "training_trajectory_seeds": [
            _training_trajectory_seed(int(seed), epoch)
            for epoch in range(1, config.train_epochs + 1)
        ],
        "validation_seeds_used_for_checkpoint_selection_only": list(config.validation_seeds),
        "best_validation_epoch": best_epoch,
        "best_validation_score": best_score,
        "initial_validation_score": float(initial["selection_score_mean"]),
        "validation_history": validation_history,
        "training_curve": training_curve,
        "gradient_coverage": {
            "audited_epochs": [1, int(config.train_epochs)],
            "covered_parameter_elements": covered_elements,
            "total_parameter_elements": total_elements,
            "coverage_fraction": covered_elements / total_elements,
            "uncovered_by_parameter": uncovered,
            "expected_unobserved_leakage_column_elements": 10,
        },
        "checkpoint_sha256": state_dict_sha256(best_state),
        "wall_time_seconds": time.perf_counter() - started,
    }
    return model, record


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    th = _require_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    th.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _summarize(evaluations: Mapping[str, Any]) -> dict[str, Any]:
    metrics = (
        "fidelity_effective_lifetime_cycles",
        "fidelity_normalized_auc",
        "logical_z_effective_lifetime_cycles",
        "logical_z_normalized_auc",
    )
    auxiliary = (
        "mean_ground_outcome_probability",
        "mean_control_residual_rms",
        "mean_control_slew_rms",
        "fidelity_log_linear_fit_r_squared_diagnostic",
        "logical_z_log_linear_fit_r_squared_diagnostic",
    )
    summary: dict[str, Any] = {}
    for strategy, values in evaluations.items():
        items = [values] if strategy == "standard" else values
        summary[strategy] = {
            metric: _distribution([_agent_metric(item, metric) for item in items])
            for metric in metrics
        }
        summary[strategy]["auxiliary"] = {
            metric: _distribution([float(item["auxiliary_means"][metric]) for item in items])
            for metric in auxiliary
        }
    return summary


def _source_rows(
    records: Sequence[Mapping[str, Any]], evaluations: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        seed = int(record["training_seed"])
        for item in record["training_curve"]:
            for metric in (
                "mean_reward", "score_path", "loss", "gradient_norm_before_clip",
                "ground_outcome_fraction", "control_residual_rms", "control_slew_rms",
            ):
                rows.append({"row_type": "training", "lane": "train", "strategy": "exact_mf", "training_seed": seed, "evaluation_seed": "", "epoch": item["epoch"], "cycle": "", "metric": metric, "value": item[metric]})
        for item in record["validation_history"]:
            rows.append({"row_type": "validation", "lane": "validation", "strategy": "exact_mf", "training_seed": seed, "evaluation_seed": "", "epoch": item["epoch"], "cycle": "", "metric": "selection_score", "value": item["selection_score"]})
    for lane, lane_values in evaluations.items():
        for strategy, values in lane_values.items():
            items = [values] if strategy == "standard" else values
            for agent_index, evaluation in enumerate(items):
                training_seed = -1 if strategy == "standard" else int(records[agent_index]["training_seed"])
                for seed_item in evaluation["per_seed"]:
                    for curve_name in ("fidelity_curve", "code_survival_curve", "logical_z_curve"):
                        for cycle, value in enumerate(seed_item[curve_name]):
                            rows.append({"row_type": "evaluation_curve", "lane": lane, "strategy": strategy, "training_seed": training_seed, "evaluation_seed": seed_item["seed"], "epoch": "", "cycle": cycle, "metric": curve_name, "value": value})
                    for metric in ("mean_ground_outcome_probability", "mean_control_residual_rms", "mean_control_slew_rms", "maximum_trace_error", "maximum_hermiticity_error", "minimum_final_eigenvalue"):
                        rows.append({"row_type": "evaluation_metric", "lane": lane, "strategy": strategy, "training_seed": training_seed, "evaluation_seed": seed_item["seed"], "epoch": "", "cycle": "", "metric": metric, "value": seed_item[metric]})
    return rows


def _write_source_data(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("row_type", "lane", "strategy", "training_seed", "evaluation_seed", "epoch", "cycle", "metric", "value"))
        writer.writeheader()
        writer.writerows(rows)


def run_benchmark(
    *,
    parent_artifact_path: Path = DEFAULT_PARENT_ARTIFACT,
    parent_checkpoint_path: Path = DEFAULT_PARENT_CHECKPOINT,
    artifact_path: Path = DEFAULT_ARTIFACT,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    source_data_path: Path = DEFAULT_SOURCE_DATA,
    resume: bool = True,
) -> dict[str, Any]:
    th = _require_torch()
    started = time.perf_counter()
    config, parent_models, parent_artifact, _ = _load_parent_evidence(
        parent_artifact_path, parent_checkpoint_path
    )
    validate_production_design(config)
    if config.device == "cuda" and not th.cuda.is_available():
        raise RuntimeError("the frozen parent protocol requires CUDA")

    source_hash = implementation_sha256()
    parent_checkpoint_hash = _sha256(parent_checkpoint_path)
    contract_hash = hashlib.sha256(json.dumps({
        "benchmark_contract_id": BENCHMARK_CONTRACT_ID,
        "config_contract_hash": config.contract_hash,
        "implementation_sha256": source_hash,
        "parent_checkpoint_sha256": parent_checkpoint_hash,
        "parameter_contract": asdict(COMPUTE_CONTRACT),
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    payload: dict[str, Any] = {
        "schema_version": 1,
        "contract_hash": contract_hash,
        "implementation_sha256": source_hash,
        "models": [],
        "training_records": [],
    }
    resumed = 0
    if resume and checkpoint_path.exists():
        loaded = th.load(checkpoint_path, map_location="cpu", weights_only=False)
        if loaded.get("schema_version") != 1 or loaded.get("contract_hash") != contract_hash:
            raise ValueError("T3.2.7 checkpoint contract mismatch")
        payload = loaded

    existing = {int(item["training_seed"]): item for item in payload["models"]}
    record_by_seed = {int(item["training_seed"]): item for item in payload["training_records"]}
    exact_models: list[Any] = []
    records: list[dict[str, Any]] = []
    for seed in config.training_seeds:
        if int(seed) in existing:
            item = existing[int(seed)]
            model = build_budget_matched_policy(device=config.device, dtype=_torch_dtype(config.real_dtype), seed=int(seed))
            model.load_state_dict(item["state_dict"])
            if state_dict_sha256(_state_dict_cpu(model)) != item["checkpoint_sha256"]:
                raise ValueError(f"T3.2.7 resumed model hash mismatch for seed {seed}")
            record = record_by_seed[int(seed)]
            resumed += 1
        else:
            model, record = train_agent(int(seed), config)
            item = {"training_seed": int(seed), "checkpoint_sha256": record["checkpoint_sha256"], "state_dict": _state_dict_cpu(model)}
            payload["models"].append(item)
            payload["training_records"].append(record)
            _atomic_torch_save(payload, checkpoint_path)
        model.eval()
        exact_models.append(model)
        records.append(record)

    def evaluate_lane(*, cutoff: int, batch_size: int, seeds: Sequence[int]) -> dict[str, Any]:
        return {
            "standard": evaluate_policy("standard", None, config, cutoff=cutoff, batch_size=batch_size, seeds=seeds),
            "legacy_mf": [evaluate_policy("mf", model, config, cutoff=cutoff, batch_size=batch_size, seeds=seeds) for model in parent_models["legacy_mf"]],
            "exact_mf": [evaluate_policy("mf", model, config, cutoff=cutoff, batch_size=batch_size, seeds=seeds) for model in exact_models],
            "history_nmf": [evaluate_policy("nmf", model, config, cutoff=cutoff, batch_size=batch_size, seeds=seeds) for model in parent_models["history_nmf"]],
        }

    evaluations = {
        "primary": evaluate_lane(cutoff=config.cutoff, batch_size=config.test_batch_size, seeds=config.test_seeds),
        "confirmation": evaluate_lane(cutoff=config.confirmation_cutoff, batch_size=config.confirmation_batch_size, seeds=config.confirmation_seeds),
    }
    summary = {lane: _summarize(values) for lane, values in evaluations.items()}
    metric = "logical_z_effective_lifetime_cycles"
    exact_values = summary["primary"]["exact_mf"][metric]["values"]
    nmf_values = summary["primary"]["history_nmf"][metric]["values"]
    legacy_values = summary["primary"]["legacy_mf"][metric]["values"]
    paired = {
        "history_nmf_minus_exact_mf": _bootstrap_agent_difference(nmf_values, exact_values, seed=config.bootstrap_seed + 327, repetitions=config.bootstrap_repetitions),
        "exact_mf_minus_legacy_mf": _bootstrap_agent_difference(exact_values, legacy_values, seed=config.bootstrap_seed + 328, repetitions=config.bootstrap_repetitions),
    }

    behavior = [dict(audit_latest_only_behavior(model)) for model in exact_models]
    checkpoint_models_complete = len(payload["models"]) == len(config.training_seeds)
    checkpoint_hashes_match = checkpoint_models_complete and all(
        state_dict_sha256(item["state_dict"]) == item["checkpoint_sha256"]
        for item in payload["models"]
    )
    density_ok = all(
        seed_item["maximum_trace_error"] <= 2.0e-9
        and seed_item["maximum_hermiticity_error"] <= 2.0e-9
        and seed_item["minimum_final_eigenvalue"] >= -2.0e-8
        for lane in evaluations.values()
        for strategy, values in lane.items()
        for evaluation in ([values] if strategy == "standard" else values)
        for seed_item in evaluation["per_seed"]
    )
    gates = {
        "parent_pass_source_and_checkpoint_are_live": True,
        "seed_splits_and_training_protocol_are_identical": all(
            record["epochs_executed"] == config.train_epochs
            and record["validation_seeds_used_for_checkpoint_selection_only"] == list(config.validation_seeds)
            and set(record["training_trajectory_seeds"]).isdisjoint(
                set(config.validation_seeds) | set(config.test_seeds) | set(config.confirmation_seeds)
            )
            for record in records
        ),
        "parameter_count_exactly_matches_history_model": all(model.parameter_count == 72_853 for model in exact_models) and all(model.parameter_count == 72_853 for model in parent_models["history_nmf"]),
        "dense_mac_budget_exactly_matches_history_model": all(model.dense_mac_count == COMPUTE_CONTRACT.total_dense_macs for model in exact_models),
        "same_fifteen_dimensional_action_contract": len(PARAMETER_NAMES) == 15 and ACTION_CONTRACT_ID.endswith("15"),
        "strict_latest_only_behavior_is_bit_exact": all(item["earlier_history_invariant_bit_exact"] and item["stateless_repeat_bit_exact"] and item["has_no_recurrent_state_attribute"] for item in behavior),
        "all_g_e_leakage_tokens_are_executable_and_distinct": all(item["all_three_tokens_have_distinct_outputs"] for item in behavior),
        "no_dummy_parameter_tensor_in_observed_training_path": all(record["gradient_coverage"]["coverage_fraction"] >= (COMPUTE_CONTRACT.total_parameter_count - 10) / COMPUTE_CONTRACT.total_parameter_count for record in records),
        "checkpoint_contains_all_seed_models_with_matching_hashes": checkpoint_hashes_match,
        "all_evaluations_use_frozen_held_out_seeds": all(set(record["training_trajectory_seeds"]).isdisjoint(set(config.test_seeds) | set(config.confirmation_seeds)) for record in records),
        "finite_density_diagnostics_pass": density_ok,
        "signed_memory_comparison_is_reported_without_direction_gate": set(paired) == {"history_nmf_minus_exact_mf", "exact_mf_minus_legacy_mf"},
        "two_level_leakage_evidence_boundary_is_explicit": OUTCOME_ALPHABET_ID.endswith("V1") and "emits only g/e" in SCOPE,
    }
    required = tuple(gates)
    status = "PASS" if all(gates[name] for name in required) else "FAIL"

    rows = _source_rows(records, evaluations)
    _write_source_data(rows, source_data_path)
    checkpoint_sha = _sha256(checkpoint_path)
    artifact: dict[str, Any] = {
        "task_id": TASK_ID,
        "status": status,
        "scope": SCOPE,
        "benchmark_contract_id": BENCHMARK_CONTRACT_ID,
        "outcome_alphabet_id": OUTCOME_ALPHABET_ID,
        "action_contract_id": ACTION_CONTRACT_ID,
        "training_protocol_id": TRAINING_PROTOCOL_ID,
        "architectures": {
            "legacy_mf": PAPER_MF_ARCHITECTURE,
            "exact_mf": ARCHITECTURE_ID,
            "history_nmf": PAPER_RNN_ARCHITECTURE,
        },
        "compute_contract": asdict(COMPUTE_CONTRACT),
        "implementation_sha256": source_hash,
        "contract_hash": contract_hash,
        "config": asdict(config),
        "parent_evidence": {
            "artifact_path": parent_artifact_path.as_posix(),
            "artifact_sha256": _sha256(parent_artifact_path),
            "checkpoint_path": parent_checkpoint_path.as_posix(),
            "checkpoint_sha256": parent_checkpoint_hash,
            "parent_implementation_sha256": parent_artifact["implementation_sha256"],
        },
        "execution": {"resume_enabled": bool(resume), "resumed_agents": resumed, "newly_trained_agents": len(config.training_seeds) - resumed},
        "training_records": records,
        "evaluation": evaluations,
        "summary": summary,
        "paired_bootstrap": paired,
        "latest_only_behavior_audit": behavior,
        "leakage_evidence": {
            "interface_token_supported": True,
            "production_two_level_simulator_token_count": 0,
            "multilevel_leakage_training_or_evaluation": False,
        },
        "gates": gates,
        "required_gates": list(required),
        "checkpoint": {"path": checkpoint_path.as_posix(), "sha256": checkpoint_sha, "schema_version": 1, "contains_all_training_seed_models": checkpoint_models_complete, "all_model_hashes_match": checkpoint_hashes_match},
        "source_data": {"path": source_data_path.as_posix(), "sha256": _sha256(source_data_path), "row_count": len(rows)},
        "claim_boundary": {
            "allowed": "same-budget finite-cutoff latest-outcome Markovian comparator and signed memory contrast",
            "forbidden": ["multilevel leakage robustness", "optimizer optimality", "paper-exact 1000-cycle lifetime", "device or FPGA performance"],
        },
        "wall_time_seconds": time.perf_counter() - started,
    }
    _atomic_json(artifact, artifact_path)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-artifact", type=Path, default=DEFAULT_PARENT_ARTIFACT)
    parser.add_argument("--parent-checkpoint", type=Path, default=DEFAULT_PARENT_CHECKPOINT)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--source-data", type=Path, default=DEFAULT_SOURCE_DATA)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    artifact = run_benchmark(
        parent_artifact_path=args.parent_artifact,
        parent_checkpoint_path=args.parent_checkpoint,
        artifact_path=args.artifact,
        checkpoint_path=args.checkpoint,
        source_data_path=args.source_data,
        resume=not args.no_resume,
    )
    print(json.dumps({"task_id": TASK_ID, "status": artifact["status"], "paired_bootstrap": artifact["paired_bootstrap"], "gates": artifact["gates"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

