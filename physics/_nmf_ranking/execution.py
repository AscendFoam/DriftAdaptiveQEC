"""Private checkpoint, orchestration and CLI for NMF directional ranking."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..differentiable_sbs_trajectory import DIFFERENTIABLE_SBS_SCOPE, POROTTI_S1_PROFILE_ID
from ..nmf_directional_ranking import (
    ANALYSIS_CONTRACT_ID,
    NMF_RANKING_SCOPE,
    OFFICIAL_CODE_URL,
    PAPER_ARXIV,
    PAPER_DOI,
    POLICY_INITIALIZATION,
    TRAINING_PROTOCOL_ID,
    DirectionalRankingConfig,
    _agent_metric,
    _bootstrap_agent_difference,
    _distribution,
    _require_torch,
    _state_dict_cpu,
    build_policy,
    evaluate_policy,
    implementation_sha256,
    state_dict_sha256,
    train_agent,
    validate_production_design,
)
from ..sbs_error_space import SBS_PROTOCOL_ID


def _atomic_torch_save(payload: Any, path: Path) -> None:
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


def run_directional_ranking(
    config: DirectionalRankingConfig,
    *,
    artifact_path: str | Path,
    checkpoint_path: str | Path,
    production: bool = True,
    resume: bool = True,
) -> dict[str, Any]:
    """Train all agents, evaluate held-out seeds, and persist auditable evidence."""

    th = _require_torch()
    if production:
        validate_production_design(config)
    if config.device == "cuda" and not th.cuda.is_available():
        raise RuntimeError("CUDA production run requested but torch.cuda.is_available() is false")
    start = time.perf_counter()
    implementation_hash = implementation_sha256()
    checkpoint_contract_hash = hashlib.sha256(
        f"{config.contract_hash}:{implementation_hash}:{TRAINING_PROTOCOL_ID}".encode(
            "utf-8"
        )
    ).hexdigest()
    trained_models: dict[str, list[Any]] = {"mf": [], "nmf": []}
    training_records: dict[str, list[dict[str, Any]]] = {"mf": [], "nmf": []}
    checkpoint_payload: dict[str, Any] = {
        "schema_version": 3,
        "config_contract_hash": config.contract_hash,
        "checkpoint_contract_hash": checkpoint_contract_hash,
        "implementation_sha256": implementation_hash,
        "training_protocol_id": TRAINING_PROTOCOL_ID,
        "config": asdict(config),
        "models": {"mf": [], "nmf": []},
        "training_records": {"mf": [], "nmf": []},
    }
    checkpoint = Path(checkpoint_path)
    resumed_agents = 0
    if resume and checkpoint.exists():
        loaded = th.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(loaded, dict) or loaded.get("schema_version") != 3:
            raise ValueError("checkpoint exists but is not a resumable T2.3.7 schema-v3 file")
        if loaded.get("checkpoint_contract_hash") != checkpoint_contract_hash:
            raise ValueError(
                "checkpoint contract hash differs from the requested config/source/training protocol"
            )
        checkpoint_payload = loaded
    for strategy in ("mf", "nmf"):
        existing_models = {
            int(item["training_seed"]): item
            for item in checkpoint_payload["models"][strategy]
        }
        existing_records = {
            int(item["training_seed"]): item
            for item in checkpoint_payload["training_records"][strategy]
        }
        for seed in config.training_seeds:
            if seed in existing_models and seed in existing_records:
                model = build_policy(strategy, config, seed)
                model.load_state_dict(existing_models[seed]["state_dict"])
                model.eval()
                record = existing_records[seed]
                if state_dict_sha256(_state_dict_cpu(model)) != record["checkpoint_sha256"]:
                    raise ValueError("resumed model hash does not match its training record")
                resumed_agents += 1
            else:
                model, record = train_agent(strategy, seed, config)
                checkpoint_payload["models"][strategy].append(
                    {
                        "training_seed": int(seed),
                        "checkpoint_sha256": record["checkpoint_sha256"],
                        "state_dict": _state_dict_cpu(model),
                    }
                )
                checkpoint_payload["training_records"][strategy].append(record)
                _atomic_torch_save(checkpoint_payload, checkpoint)
                print(
                    json.dumps(
                        {
                            "event": "agent_complete",
                            "strategy": strategy,
                            "training_seed": int(seed),
                            "best_validation_epoch": record["best_validation_epoch"],
                            "best_validation_score": record["best_validation_score"],
                            "checkpoint_sha256": record["checkpoint_sha256"],
                        }
                    ),
                    flush=True,
                )
            trained_models[strategy].append(model)
            training_records[strategy].append(record)

    primary: dict[str, Any] = {
        "standard": evaluate_policy(
            "standard",
            None,
            config,
            cutoff=config.cutoff,
            batch_size=config.test_batch_size,
            seeds=config.test_seeds,
        ),
        "mf": [],
        "nmf": [],
        "nmf_latest_only": [],
    }
    confirmation: dict[str, Any] = {
        "standard": evaluate_policy(
            "standard",
            None,
            config,
            cutoff=config.confirmation_cutoff,
            batch_size=config.confirmation_batch_size,
            seeds=config.confirmation_seeds,
        ),
        "mf": [],
        "nmf": [],
        "nmf_latest_only": [],
    }
    for strategy in ("mf", "nmf"):
        for model in trained_models[strategy]:
            primary[strategy].append(
                evaluate_policy(
                    strategy,
                    model,
                    config,
                    cutoff=config.cutoff,
                    batch_size=config.test_batch_size,
                    seeds=config.test_seeds,
                )
            )
            confirmation[strategy].append(
                evaluate_policy(
                    strategy,
                    model,
                    config,
                    cutoff=config.confirmation_cutoff,
                    batch_size=config.confirmation_batch_size,
                    seeds=config.confirmation_seeds,
                )
            )
            if strategy == "nmf":
                primary["nmf_latest_only"].append(
                    evaluate_policy(
                        "nmf_latest_only",
                        model,
                        config,
                        cutoff=config.cutoff,
                        batch_size=config.test_batch_size,
                        seeds=config.test_seeds,
                    )
                )
                confirmation["nmf_latest_only"].append(
                    evaluate_policy(
                        "nmf_latest_only",
                        model,
                        config,
                        cutoff=config.confirmation_cutoff,
                        batch_size=config.confirmation_batch_size,
                        seeds=config.confirmation_seeds,
                    )
                )

    metric_names = (
        "fidelity_effective_lifetime_cycles",
        "fidelity_normalized_auc",
        "logical_z_effective_lifetime_cycles",
        "logical_z_normalized_auc",
    )
    summary: dict[str, Any] = {"primary": {}, "confirmation": {}}
    for lane_name, lane in (("primary", primary), ("confirmation", confirmation)):
        for strategy in ("standard", "mf", "nmf", "nmf_latest_only"):
            evaluations = [lane[strategy]] if strategy == "standard" else lane[strategy]
            summary[lane_name][strategy] = {
                metric: _distribution([_agent_metric(item, metric) for item in evaluations])
                for metric in metric_names
            }
            summary[lane_name][strategy]["auxiliary"] = {
                name: _distribution(
                    [float(item["auxiliary_means"][name]) for item in evaluations]
                )
                for name in (
                    "mean_ground_outcome_probability",
                    "mean_control_residual_rms",
                    "mean_control_slew_rms",
                    "fidelity_log_linear_fit_r_squared_diagnostic",
                    "logical_z_log_linear_fit_r_squared_diagnostic",
                )
            }

    standard_lifetime = summary["primary"]["standard"][
        "logical_z_effective_lifetime_cycles"
    ]["mean"]
    mf_lifetimes = summary["primary"]["mf"]["logical_z_effective_lifetime_cycles"][
        "values"
    ]
    nmf_lifetimes = summary["primary"]["nmf"]["logical_z_effective_lifetime_cycles"][
        "values"
    ]
    ablated_lifetimes = summary["primary"]["nmf_latest_only"][
        "logical_z_effective_lifetime_cycles"
    ]["values"]
    nmf_auc = summary["primary"]["nmf"]["logical_z_normalized_auc"]["values"]
    mf_auc = summary["primary"]["mf"]["logical_z_normalized_auc"]["values"]
    ablated_auc = summary["primary"]["nmf_latest_only"]["logical_z_normalized_auc"][
        "values"
    ]
    confirmation_standard = summary["confirmation"]["standard"][
        "logical_z_effective_lifetime_cycles"
    ]["mean"]
    confirmation_mf = summary["confirmation"]["mf"][
        "logical_z_effective_lifetime_cycles"
    ]["median"]
    confirmation_nmf = summary["confirmation"]["nmf"][
        "logical_z_effective_lifetime_cycles"
    ]["median"]
    paired_nmf_mf = _bootstrap_agent_difference(
        nmf_lifetimes,
        mf_lifetimes,
        seed=config.bootstrap_seed,
        repetitions=config.bootstrap_repetitions,
    )
    paired_nmf_ablation = _bootstrap_agent_difference(
        nmf_lifetimes,
        ablated_lifetimes,
        seed=config.bootstrap_seed + 1,
        repetitions=config.bootstrap_repetitions,
    )
    configured_evaluation_seeds = (
        set(config.validation_seeds)
        | set(config.test_seeds)
        | set(config.confirmation_seeds)
    )
    actual_training_rollout_seeds = {
        int(seed_value)
        for strategy in ("mf", "nmf")
        for record in training_records[strategy]
        for seed_value in (
            [record["baseline_warmup_seed"]]
            + list(record["training_trajectory_seeds"])
        )
    }
    checkpoint_selection_audits: list[bool] = []
    for strategy in ("mf", "nmf"):
        for record in training_records[strategy]:
            history = record["validation_history"]
            best_entry = max(history, key=lambda item: float(item["selection_score"]))
            checkpoint_selection_audits.append(
                record["baseline_initialization"]
                == "train-only no-gradient warm-up reward mean"
                and list(record["validation_seeds_used_for_checkpoint_selection_only"])
                == list(config.validation_seeds)
                and int(record["best_validation_epoch"]) == int(best_entry["epoch"])
                and np.isclose(
                    float(record["best_validation_score"]),
                    float(best_entry["selection_score"]),
                    rtol=0.0,
                    atol=1.0e-14,
                )
            )
    expected_agent_seeds = set(config.training_seeds)
    checkpoint_models_complete = all(
        {
            int(item["training_seed"])
            for item in checkpoint_payload["models"][strategy]
        }
        == expected_agent_seeds
        and {
            int(item["training_seed"])
            for item in checkpoint_payload["training_records"][strategy]
        }
        == expected_agent_seeds
        for strategy in ("mf", "nmf")
    )
    checkpoint_model_hashes_valid = all(
        state_dict_sha256(item["state_dict"]) == item["checkpoint_sha256"]
        for strategy in ("mf", "nmf")
        for item in checkpoint_payload["models"][strategy]
    )
    all_evaluations = {
        "primary": primary,
        "confirmation": confirmation,
    }
    gates = {
        "all_seed_sets_are_disjoint": not (
            set(config.training_seeds)
            & (set(config.validation_seeds) | set(config.test_seeds) | set(config.confirmation_seeds))
            or set(config.validation_seeds)
            & (set(config.test_seeds) | set(config.confirmation_seeds))
            or set(config.test_seeds) & set(config.confirmation_seeds)
        )
        and not (actual_training_rollout_seeds & configured_evaluation_seeds),
        "paper_scale_architectures_used": (
            all(item["parameter_count"] == 70_159 for item in training_records["mf"])
            and all(item["parameter_count"] == 72_853 for item in training_records["nmf"])
        ),
        "checkpoint_selection_uses_validation_only": all(
            checkpoint_selection_audits
        ),
        "primary_logical_z_lifetime_ranks_nmf_above_mf_and_standard": (
            float(np.median(nmf_lifetimes)) > max(float(np.median(mf_lifetimes)), standard_lifetime)
        ),
        "primary_logical_z_auc_ranks_nmf_above_mf_and_standard": (
            float(np.median(nmf_auc))
            > max(
                float(np.median(mf_auc)),
                summary["primary"]["standard"]["logical_z_normalized_auc"]["mean"],
            )
        ),
        "primary_fidelity_lifetime_has_same_nmf_direction": (
            summary["primary"]["nmf"]["fidelity_effective_lifetime_cycles"]["median"]
            > max(
                summary["primary"]["mf"]["fidelity_effective_lifetime_cycles"]["median"],
                summary["primary"]["standard"]["fidelity_effective_lifetime_cycles"]["mean"],
            )
        ),
        "at_least_80_percent_of_nmf_agents_beat_both_comparators": (
            int(
                np.sum(
                    (np.asarray(nmf_lifetimes) > np.asarray(mf_lifetimes))
                    & (np.asarray(nmf_lifetimes) > standard_lifetime)
                )
            )
            >= max(1, int(np.ceil(0.8 * len(nmf_lifetimes))))
        ),
        "history_reset_reduces_nmf_lifetime_and_auc": (
            float(np.median(nmf_lifetimes)) > float(np.median(ablated_lifetimes))
            and float(np.median(nmf_auc)) > float(np.median(ablated_auc))
        ),
        "cutoff16_confirmation_preserves_direction": (
            confirmation_nmf > max(confirmation_mf, confirmation_standard)
        ),
        "nmf_ground_outcome_probability_stays_above_0_75": (
            summary["primary"]["nmf"]["auxiliary"][
                "mean_ground_outcome_probability"
            ]["minimum"]
            >= 0.75
            and summary["confirmation"]["nmf"]["auxiliary"][
                "mean_ground_outcome_probability"
            ]["minimum"]
            >= 0.75
        ),
        "bounded_control_residual_and_slew_are_finite": all(
            np.isfinite(
                summary[lane][strategy]["auxiliary"][metric]["maximum"]
            )
            for lane in ("primary", "confirmation")
            for strategy in ("mf", "nmf", "nmf_latest_only")
            for metric in ("mean_control_residual_rms", "mean_control_slew_rms")
        )
        and all(
            summary[lane][strategy]["auxiliary"]["mean_control_residual_rms"][
                "maximum"
            ]
            <= 2.0
            and summary[lane][strategy]["auxiliary"]["mean_control_slew_rms"][
                "maximum"
            ]
            <= 4.0
            for lane in ("primary", "confirmation")
            for strategy in ("mf", "nmf", "nmf_latest_only")
        ),
        "nonexponential_fit_is_diagnostic_only": all(
            bool(item[metric]["exponential_fit_is_diagnostic_only"])
            and item[metric]["effective_lifetime_method"]
            == "finite_horizon_area_equivalent_exponential"
            for lane in ("primary", "confirmation")
            for strategy in ("standard", "mf", "nmf", "nmf_latest_only")
            for evaluation in (
                [all_evaluations[lane][strategy]]
                if strategy == "standard"
                else all_evaluations[lane][strategy]
            )
            for item in evaluation["per_seed"]
            for metric in ("fidelity", "logical_z")
        ),
        "finite_density_diagnostics_pass": all(
            item["maximum_trace_error"] <= 2.0e-9
            and item["maximum_hermiticity_error"] <= 2.0e-9
            and item["minimum_final_eigenvalue"] >= -2.0e-8
            for lane in ("primary", "confirmation")
            for strategy in ("standard", "mf", "nmf", "nmf_latest_only")
            for evaluation in (
                [all_evaluations[lane][strategy]]
                if strategy == "standard"
                else all_evaluations[lane][strategy]
            )
            for item in evaluation["per_seed"]
        ),
        "checkpoint_contains_all_models_with_matching_hashes": (
            checkpoint_models_complete and checkpoint_model_hashes_valid
        ),
    }
    required_directional_gates = (
        "all_seed_sets_are_disjoint",
        "paper_scale_architectures_used",
        "checkpoint_selection_uses_validation_only",
        "primary_logical_z_lifetime_ranks_nmf_above_mf_and_standard",
        "primary_logical_z_auc_ranks_nmf_above_mf_and_standard",
        "primary_fidelity_lifetime_has_same_nmf_direction",
        "at_least_80_percent_of_nmf_agents_beat_both_comparators",
        "history_reset_reduces_nmf_lifetime_and_auc",
        "cutoff16_confirmation_preserves_direction",
        "nmf_ground_outcome_probability_stays_above_0_75",
        "bounded_control_residual_and_slew_are_finite",
        "nonexponential_fit_is_diagnostic_only",
        "finite_density_diagnostics_pass",
        "checkpoint_contains_all_models_with_matching_hashes",
    )
    status = "PASS" if all(gates[name] for name in required_directional_gates) else "FAIL"
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    artifact: dict[str, Any] = {
        "task_id": "T2.3.7",
        "status": status,
        "scope": NMF_RANKING_SCOPE,
        "protocol_id": SBS_PROTOCOL_ID,
        "simulator_scope": DIFFERENTIABLE_SBS_SCOPE,
        "timing_profile_id": POROTTI_S1_PROFILE_ID,
        "literature": {
            "doi": PAPER_DOI,
            "arxiv": PAPER_ARXIV,
            "official_code": OFFICIAL_CODE_URL,
            "zotero_item_key": "IJ5EDGZF",
            "bibtex_key": "puviani_non-markovian_2025",
        },
        "policy_initialization": POLICY_INITIALIZATION,
        "analysis_contract_id": ANALYSIS_CONTRACT_ID,
        "training_protocol_id": TRAINING_PROTOCOL_ID,
        "implementation_sha256": implementation_hash,
        "config": asdict(config),
        "contract_hash": config.contract_hash,
        "checkpoint_contract_hash": checkpoint_contract_hash,
        "production_design": bool(production),
        "execution": {
            "resume_enabled": bool(resume),
            "resumed_agents": resumed_agents,
            "newly_trained_agents": 2 * len(config.training_seeds) - resumed_agents,
        },
        "training_records": training_records,
        "evaluation": {"primary": primary, "confirmation": confirmation},
        "summary": summary,
        "paired_bootstrap": {
            "nmf_minus_mf_logical_z_lifetime": paired_nmf_mf,
            "nmf_minus_latest_only_ablation_logical_z_lifetime": paired_nmf_ablation,
        },
        "gates": gates,
        "required_directional_gates": list(required_directional_gates),
        "checkpoint": {
            "path": str(checkpoint).replace("\\", "/"),
            "sha256": checkpoint_sha256,
            "contains_all_training_seed_models": checkpoint_models_complete,
            "all_model_hashes_match": checkpoint_model_hashes_valid,
            "schema_version": 3,
        },
        "claim_boundary": {
            "allowed": (
                "directional 10-cycle state-fidelity and projected-logical-Z finite-horizon "
                "area-equivalent lifetime ranking within the finite-cutoff two-level "
                "high-noise simulator"
            ),
            "forbidden": (
                "paper-exact 1000-cycle six-state channel lifetime, optimizer optimality, "
                "experimental/device/FPGA lifetime, or leakage/SPAM robustness"
            ),
            "failure_branch": (
                "if status is FAIL, NMF remains a literature baseline and teacher-distillation "
                "claims are disabled"
            ),
        },
        "wall_time_seconds": time.perf_counter() - start,
    }
    _atomic_json(artifact, Path(artifact_path))
    return artifact


def _parse_seed_list(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--cutoff", type=int, default=12)
    parser.add_argument("--confirmation-cutoff", type=int, default=16)
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=320)
    parser.add_argument("--train-batch", type=int, default=8)
    parser.add_argument("--validation-batch", type=int, default=32)
    parser.add_argument("--test-batch", type=int, default=64)
    parser.add_argument("--confirmation-batch", type=int, default=32)
    parser.add_argument("--validation-interval", type=int, default=40)
    parser.add_argument("--training-seeds", type=_parse_seed_list, default=(101, 211, 307, 401, 503))
    parser.add_argument("--validation-seeds", type=_parse_seed_list, default=(10007, 10009))
    parser.add_argument(
        "--test-seeds",
        type=_parse_seed_list,
        default=(20011, 20021, 20023, 20029, 20047, 20051, 20063, 20071),
    )
    parser.add_argument(
        "--confirmation-seeds", type=_parse_seed_list, default=(30011, 30013, 30029, 30047)
    )
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    config = DirectionalRankingConfig(
        cutoff=arguments.cutoff,
        confirmation_cutoff=arguments.confirmation_cutoff,
        full_cycles=arguments.cycles,
        train_epochs=arguments.epochs,
        train_batch_size=arguments.train_batch,
        validation_batch_size=arguments.validation_batch,
        test_batch_size=arguments.test_batch,
        confirmation_batch_size=arguments.confirmation_batch,
        validation_interval=arguments.validation_interval,
        training_seeds=arguments.training_seeds,
        validation_seeds=arguments.validation_seeds,
        test_seeds=arguments.test_seeds,
        confirmation_seeds=arguments.confirmation_seeds,
        device=arguments.device,
    )
    artifact = run_directional_ranking(
        config,
        artifact_path=arguments.artifact,
        checkpoint_path=arguments.checkpoint,
        production=not arguments.pilot,
        resume=not arguments.no_resume,
    )
    print(json.dumps({"status": artifact["status"], "gates": artifact["gates"]}, indent=2))
    return 0 if artifact["status"] == "PASS" else 2
