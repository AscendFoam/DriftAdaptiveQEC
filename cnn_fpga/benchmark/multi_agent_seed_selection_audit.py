"""T5.4.4 multi-agent/restart/seed selection-bias audit.

The audit is read-only: it reconstructs every current learned-candidate selection
from validation evidence, retains every registered evaluation unit, and reports
median/IQR/worst-quartile summaries.  Hindsight test-best diagnostics are shown
only to quantify the bias that would have resulted from post-selection; they
never alter the frozen candidate chosen by the parent campaign.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cnn_fpga.benchmark import algorithm_success_falsification as branch_parent
from cnn_fpga.benchmark import bounded_residual_rnn_teacher as teacher_parent
from cnn_fpga.benchmark import low_dimensional_student_distillation as student_parent
from cnn_fpga.benchmark import offline_teacher_student_distillation as legacy_parent
from cnn_fpga.benchmark import slow_loop_model_selection as slow_parent
from cnn_fpga.benchmark import teacher_student_gain_retention as retention_parent
import physics.nmf_directional_ranking as nmf_parent


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "T5.4.4"
SCHEMA_VERSION = "t5.4.4-multi-agent-seed-selection-audit-v1"
PROTOCOL_ID = "VALIDATION-ONLY-ALL-UNIT-SELECTION-AUDIT-V1"
DEFAULT_ARTIFACT = Path("docs/t5_4_4_multi_agent_seed_selection_audit.json")
DEFAULT_SOURCE_DATA = Path(
    "docs/t5_4_4_multi_agent_seed_selection_audit_source_data.csv"
)

PARENT_ARTIFACTS: dict[str, Path] = {
    "T2.3.7": Path("docs/t2_3_7_nmf_directional_ranking.json"),
    "T4.1.1": Path("docs/t4_1_1_slow_loop_model_selection_validation.json"),
    "T4.1.5": Path("docs/t4_1_5_teacher_student_validation.json"),
    "T4.4.1": Path("docs/t4_4_1_bounded_residual_rnn_teacher_validation.json"),
    "T4.4.3": Path("docs/t4_4_3_low_dimensional_student_validation.json"),
    "T4.4.4": Path("docs/t4_4_4_teacher_student_gain_retention.json"),
    "T5.1.4": Path("docs/t5_1_4_algorithm_branch_verdict.json"),
}

PARENT_SOURCE_FILES: dict[str, Path] = {
    "T2.3.7": Path("docs/t2_3_7_nmf_directional_ranking.csv"),
    "T4.1.1": Path("docs/t4_1_1_slow_loop_model_selection_source_data.csv"),
    "T4.1.5": Path("docs/t4_1_5_teacher_student_source_data.csv"),
    "T4.4.1": Path("docs/t4_4_1_bounded_residual_rnn_teacher_source_data.csv"),
    "T4.4.3": Path("docs/t4_4_3_low_dimensional_student_source_data.csv"),
    "T4.4.4": Path("docs/t4_4_4_teacher_student_gain_retention_source_data.csv"),
    "T5.1.4": Path("docs/t5_1_4_algorithm_branch_verdict_source_data.csv"),
}

IMPLEMENTATION_PATHS = (
    Path("cnn_fpga/benchmark/multi_agent_seed_selection_audit.py"),
    Path("physics/nmf_directional_ranking.py"),
    Path("cnn_fpga/benchmark/slow_loop_model_selection.py"),
    Path("cnn_fpga/benchmark/offline_teacher_student_distillation.py"),
    Path("cnn_fpga/benchmark/bounded_residual_rnn_teacher.py"),
    Path("cnn_fpga/benchmark/low_dimensional_student_distillation.py"),
    Path("cnn_fpga/benchmark/teacher_student_gain_retention.py"),
)


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


def _machine_pass(payload: Mapping[str, Any]) -> bool:
    if payload.get("status") != "PASS" and payload.get("passed") is not True:
        return False
    gate = payload.get("gate_summary")
    if not isinstance(gate, Mapping):
        return True
    failed = gate.get("failed", gate.get("failed_names", 0))
    if isinstance(failed, Sequence) and not isinstance(failed, (str, bytes)):
        return len(failed) == 0
    return int(failed or 0) == 0


def load_parent_artifacts() -> dict[str, dict[str, Any]]:
    return {
        task_id: json.loads(_repo_path(path).read_text(encoding="utf-8"))
        for task_id, path in PARENT_ARTIFACTS.items()
    }


def _bindings(
    paths: Mapping[str, Path] | Sequence[Path], *, kind: str
) -> list[dict[str, Any]]:
    if isinstance(paths, Mapping):
        items = paths.items()
    else:
        items = ((path.as_posix(), path) for path in paths)
    return [
        {
            "binding_id": str(binding_id),
            "kind": kind,
            "path": path.as_posix(),
            "sha256": _sha256(path),
        }
        for binding_id, path in items
    ]


def _parent_bindings(
    parents: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "task_id": task_id,
            "path": path.as_posix(),
            "sha256": _sha256(path),
            "machine_pass": _machine_pass(parents[task_id]),
        }
        for task_id, path in PARENT_ARTIFACTS.items()
    ]


def _distribution(
    rows: Sequence[Mapping[str, Any]],
    *,
    higher_is_better: bool,
    unit_key: str = "unit_id",
    value_key: str = "value",
) -> dict[str, Any]:
    if not rows:
        raise ValueError("distribution requires at least one row")
    values = np.asarray([float(row[value_key]) for row in rows], dtype=np.float64)
    order = np.argsort(values)
    if higher_is_better:
        worst_indices = order[: max(1, math.ceil(values.size / 4.0))]
        direction = "lower_is_worse"
    else:
        worst_indices = order[::-1][: max(1, math.ceil(values.size / 4.0))]
        direction = "higher_is_worse"
    q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75], method="linear")
    return {
        "count": int(values.size),
        "higher_is_better": higher_is_better,
        "minimum": float(np.min(values)),
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "maximum": float(np.max(values)),
        "worst_quartile_rule": direction,
        "worst_quartile_count": int(len(worst_indices)),
        "worst_quartile": [
            {
                "unit_id": str(rows[int(index)][unit_key]),
                "value": float(rows[int(index)][value_key]),
            }
            for index in worst_indices
        ],
    }


def _logical_z_lifetime(row: Mapping[str, Any]) -> float:
    return float(row["logical_z"]["effective_lifetime_cycles"])


def _nmf_lane(parent: Mapping[str, Any]) -> dict[str, Any]:
    metric = "logical_z_effective_lifetime_cycles"
    splits: dict[str, Any] = {}
    for split in ("primary", "confirmation"):
        mf_results = parent["evaluation"][split]["mf"]
        nmf_results = parent["evaluation"][split]["nmf"]
        agent_rows: list[dict[str, Any]] = []
        seed_rows: list[dict[str, Any]] = []
        for index, (mf_result, nmf_result) in enumerate(
            zip(mf_results, nmf_results, strict=True)
        ):
            mf_training = parent["training_records"]["mf"][index]
            nmf_training = parent["training_records"]["nmf"][index]
            mf_value = float(mf_result["metric_means"][metric])
            nmf_value = float(nmf_result["metric_means"][metric])
            agent_rows.append(
                {
                    "unit_id": f"agent-{index}",
                    "agent_index": index,
                    "mf_training_seed": int(mf_training["training_seed"]),
                    "nmf_training_seed": int(nmf_training["training_seed"]),
                    "mf_validation_score": float(mf_training["best_validation_score"]),
                    "nmf_validation_score": float(nmf_training["best_validation_score"]),
                    "mf_test_value": mf_value,
                    "nmf_test_value": nmf_value,
                    "value": nmf_value - mf_value,
                    "selected_after_test": False,
                }
            )
            mf_by_seed = {int(row["seed"]): row for row in mf_result["per_seed"]}
            nmf_by_seed = {int(row["seed"]): row for row in nmf_result["per_seed"]}
            for seed in sorted(mf_by_seed):
                mf_seed_value = _logical_z_lifetime(mf_by_seed[seed])
                nmf_seed_value = _logical_z_lifetime(nmf_by_seed[seed])
                seed_rows.append(
                    {
                        "unit_id": f"agent-{index}-seed-{seed}",
                        "agent_index": index,
                        "evaluation_seed": seed,
                        "mf_value": mf_seed_value,
                        "nmf_value": nmf_seed_value,
                        "value": nmf_seed_value - mf_seed_value,
                    }
                )
        distribution = _distribution(agent_rows, higher_is_better=True)
        best = max(agent_rows, key=lambda row: row["value"])
        splits[split] = {
            "cutoff": int(nmf_results[0]["cutoff"]),
            "registered_evaluation_seeds": [
                int(seed)
                for seed in parent["config"][
                    "test_seeds" if split == "primary" else "confirmation_seeds"
                ]
            ],
            "agent_rows": agent_rows,
            "agent_seed_rows": seed_rows,
            "agent_distribution": distribution,
            "agent_seed_distribution_descriptive_only": _distribution(
                seed_rows, higher_is_better=True
            ),
            "hypothetical_test_best_agent": {
                "agent_index": int(best["agent_index"]),
                "value": float(best["value"]),
                "inflation_over_all_agent_median": float(
                    best["value"] - distribution["median"]
                ),
                "used_for_claim_or_selection": False,
            },
        }
    return {
        "lane_id": "nmf_directional_all_agent",
        "parent_task": "T2.3.7",
        "selection_unit": "checkpoint_within_each_pre_registered_agent",
        "agent_selection_rule": "none_all_five_paired_agents_retained",
        "checkpoint_selection_split": "validation_only",
        "evaluation_used_for_agent_selection": False,
        "metric": metric,
        "metric_direction": "higher_is_better",
        "splits": splits,
        "claim_use": "all-agent directional distribution; never best-agent performance",
    }


def _slow_loop_lane(parent: Mapping[str, Any]) -> dict[str, Any]:
    selection = parent["training_and_validation_selection"]
    evaluation = parent["evaluation"]
    selected = str(selection["selected_family"])
    candidate_rows: list[dict[str, Any]] = []
    for row in selection["selection_table"]:
        family = str(row["family"])
        candidate_rows.append(
            {
                "unit_id": family,
                "family": family,
                "validation_nll": float(row["validation_negative_log_likelihood"]),
                "evaluation_nll": float(
                    evaluation["aggregate"][family]["negative_log_likelihood"]
                ),
                "selected_on_validation": family == selected,
            }
        )
    neural_restart_rows: list[dict[str, Any]] = []
    for family in ("causal_tcn", "small_gru"):
        detail = selection["family_details"][family]
        chosen = int(detail["selected_hyperparameters"]["restart_seed"])
        for row in detail["selection_scan"]:
            neural_restart_rows.append(
                {
                    "unit_id": f"{family}-{row['restart_seed']}",
                    "family": family,
                    "restart_seed": int(row["restart_seed"]),
                    "value": float(row["calibrated_validation_nll"]),
                    "selected_on_validation": int(row["restart_seed"]) == chosen,
                    "evaluation_metric_available_for_this_restart": False,
                }
            )
    seed_rows = [
        {
            "unit_id": f"{row['family']}-seed-{row['evaluation_seed']}",
            "family": str(row["family"]),
            "evaluation_seed": int(row["evaluation_seed"]),
            "value": float(row["metrics"]["negative_log_likelihood"]),
        }
        for row in evaluation["per_seed"]
    ]
    per_family_distributions = {
        family: _distribution(
            [row for row in seed_rows if row["family"] == family],
            higher_is_better=False,
        )
        for family in sorted(evaluation["aggregate"])
    }
    hindsight = min(candidate_rows, key=lambda row: row["evaluation_nll"])
    selected_row = next(row for row in candidate_rows if row["family"] == selected)
    return {
        "lane_id": "matched_budget_slow_loop_family_selection",
        "parent_task": "T4.1.1",
        "selection_metric": "validation_negative_log_likelihood",
        "evaluation_metric": "evaluation_negative_log_likelihood",
        "metric_direction": "lower_is_better",
        "candidate_rows": candidate_rows,
        "neural_restart_validation_rows": neural_restart_rows,
        "neural_restart_validation_distribution": _distribution(
            neural_restart_rows, higher_is_better=False
        ),
        "evaluation_seed_rows": seed_rows,
        "per_family_evaluation_distributions": per_family_distributions,
        "selected_family": selected,
        "evaluation_used_for_selection": False,
        "hindsight_test_best_diagnostic": {
            "family": hindsight["family"],
            "evaluation_nll": hindsight["evaluation_nll"],
            "agrees_with_validation_selection": hindsight["family"] == selected,
            "postselection_optimism": float(
                selected_row["evaluation_nll"] - hindsight["evaluation_nll"]
            ),
            "used_to_change_selection": False,
        },
    }


def _legacy_student_lane(parent: Mapping[str, Any]) -> dict[str, Any]:
    selected = int(parent["selected_restart"])
    candidate_rows = [
        {
            "unit_id": f"restart-{row['restart_index']}",
            "restart_index": int(row["restart_index"]),
            "restart_seed": int(row["seed"]),
            "value": float(row["validation_mse"]),
            "selected_on_validation": int(row["restart_index"]) == selected,
            "evaluation_metric_available_for_this_restart": int(row["restart_index"])
            == selected,
        }
        for row in parent["training_restarts"]
    ]
    selected_evaluation = parent["metrics"]["evaluation"]["student"]
    return {
        "lane_id": "legacy_offline_student_predecessor",
        "parent_task": "T4.1.5",
        "role": "superseded_predecessor_not_current_selected_student",
        "selection_metric": "validation_teacher_action_mse",
        "metric_direction": "lower_is_better",
        "candidate_validation_rows": candidate_rows,
        "candidate_validation_distribution": _distribution(
            candidate_rows, higher_is_better=False
        ),
        "selected_restart": selected,
        "evaluation_used_for_training_or_selection": bool(
            parent["dataset"]["evaluation_used_for_training_or_selection"]
        ),
        "selected_evaluation_metrics": selected_evaluation,
        "all_candidate_evaluation_available": False,
        "missing_evidence": (
            "nonselected legacy restart weights/evaluation metrics were not retained; "
            "no hindsight test-best restart can be computed"
        ),
        "claim_decision": (
            "retain as validation-only historical method evidence; active low-dimensional "
            "student selection is audited separately from T4.4.3"
        ),
    }


def _teacher_lane(parent: Mapping[str, Any]) -> dict[str, Any]:
    selected = int(parent["selected_restart_index"])
    primary = parent["evaluation"]["teacher_primary_all_restarts"]
    confirmation = parent["evaluation"]["teacher_confirmation_all_restarts"]
    candidate_rows: list[dict[str, Any]] = []
    for index, (training, primary_result, confirmation_result) in enumerate(
        zip(parent["training_restarts"], primary, confirmation, strict=True)
    ):
        candidate_rows.append(
            {
                "unit_id": f"restart-{index}",
                "restart_index": index,
                "training_seed": int(training["training_seed"]),
                "validation_score": float(training["best_validation_score"]),
                "primary_selection_score": float(primary_result["selection_score_mean"]),
                "confirmation_selection_score": float(
                    confirmation_result["selection_score_mean"]
                ),
                "primary_logical_z_lifetime": float(
                    primary_result["metric_means"][
                        "logical_z_effective_lifetime_cycles"
                    ]
                ),
                "confirmation_logical_z_lifetime": float(
                    confirmation_result["metric_means"][
                        "logical_z_effective_lifetime_cycles"
                    ]
                ),
                "selected_on_validation": index == selected,
            }
        )
    split_results: dict[str, Any] = {}
    for split, results in (("primary", primary), ("confirmation", confirmation)):
        seed_rows: list[dict[str, Any]] = []
        for restart_index, result in enumerate(results):
            for row in result["per_seed"]:
                seed_rows.append(
                    {
                        "unit_id": f"restart-{restart_index}-seed-{row['seed']}",
                        "restart_index": restart_index,
                        "evaluation_seed": int(row["seed"]),
                        "value": _logical_z_lifetime(row),
                    }
                )
        score_rows = [
            {
                "unit_id": row["unit_id"],
                "restart_index": row["restart_index"],
                "value": float(row[f"{split}_selection_score"]),
            }
            for row in candidate_rows
        ]
        selected_row = next(
            row for row in score_rows if row["restart_index"] == selected
        )
        hindsight = max(score_rows, key=lambda row: row["value"])
        split_results[split] = {
            "registered_evaluation_seeds": sorted(
                {row["evaluation_seed"] for row in seed_rows}
            ),
            "all_restart_seed_rows": seed_rows,
            "all_restart_seed_distribution_descriptive_only": _distribution(
                seed_rows, higher_is_better=True
            ),
            "restart_score_distribution": _distribution(
                score_rows, higher_is_better=True
            ),
            "hindsight_test_best_diagnostic": {
                "restart_index": int(hindsight["restart_index"]),
                "value": float(hindsight["value"]),
                "validation_selected_restart": selected,
                "validation_selected_value": float(selected_row["value"]),
                "agrees_with_validation_selection": int(hindsight["restart_index"])
                == selected,
                "postselection_optimism": float(
                    hindsight["value"] - selected_row["value"]
                ),
                "used_to_change_selection": False,
            },
        }
    return {
        "lane_id": "fresh_bounded_teacher_restart_selection",
        "parent_task": "T4.4.1",
        "selection_metric": "best_validation_selection_score",
        "metric_direction": "higher_is_better",
        "candidate_rows": candidate_rows,
        "candidate_validation_distribution": _distribution(
            [
                {"unit_id": row["unit_id"], "value": row["validation_score"]}
                for row in candidate_rows
            ],
            higher_is_better=True,
        ),
        "selected_restart": selected,
        "selection_rule": str(parent["selection_rule"]),
        "evaluation_used_for_selection": False,
        "splits": split_results,
    }


def _student_lane(parent: Mapping[str, Any]) -> dict[str, Any]:
    selected_dimension = int(parent["selection"]["selected_dimension"])
    selected_restart = int(parent["selection"]["selected_restart"])
    training_rows = [
        {
            "unit_id": f"d{row['dimension']}-r{row['restart_index']}",
            "dimension": int(row["dimension"]),
            "restart_index": int(row["restart_index"]),
            "restart_seed": int(row["restart_seed"]),
            "value": float(row["best_validation_mse"]),
            "selected_on_validation": int(row["dimension"]) == selected_dimension
            and int(row["restart_index"]) == selected_restart,
        }
        for row in parent["training_records"]
    ]
    dimension_rows: list[dict[str, Any]] = []
    for dimension_text, metrics in parent["candidate_metrics"].items():
        dimension = int(dimension_text)
        dimension_rows.append(
            {
                "unit_id": f"dimension-{dimension}",
                "dimension": dimension,
                "restart_index": int(metrics["restart_index"]),
                "validation_mse": float(metrics["validation"]["mse"]),
                "evaluation_mse": float(metrics["evaluation"]["mse"]),
                "selected_on_validation": dimension == selected_dimension,
            }
        )
    selected_row = next(
        row for row in dimension_rows if row["dimension"] == selected_dimension
    )
    hindsight = min(dimension_rows, key=lambda row: row["evaluation_mse"])
    return {
        "lane_id": "low_dimensional_student_candidate_selection",
        "parent_task": "T4.4.3",
        "selection_metric": "validation_teacher_action_mse_with_dimension_tolerance",
        "metric_direction": "lower_is_better",
        "training_candidate_rows": training_rows,
        "all_nine_validation_distribution": _distribution(
            training_rows, higher_is_better=False
        ),
        "best_per_dimension_rows": dimension_rows,
        "best_per_dimension_evaluation_distribution": _distribution(
            [
                {"unit_id": row["unit_id"], "value": row["evaluation_mse"]}
                for row in dimension_rows
            ],
            higher_is_better=False,
        ),
        "selected_dimension": selected_dimension,
        "selected_restart": selected_restart,
        "selection_rule": str(parent["selection"]["rule"]),
        "evaluation_used_for_selection": not bool(
            parent["selection"]["evaluation_blind"]
        ),
        "hindsight_test_best_diagnostic": {
            "dimension": int(hindsight["dimension"]),
            "restart_index": int(hindsight["restart_index"]),
            "evaluation_mse": float(hindsight["evaluation_mse"]),
            "agrees_with_validation_selection": int(hindsight["dimension"])
            == selected_dimension,
            "postselection_optimism": float(
                selected_row["evaluation_mse"] - hindsight["evaluation_mse"]
            ),
            "used_to_change_selection": False,
        },
    }


def _retention_lane(parent: Mapping[str, Any]) -> dict[str, Any]:
    split_results: dict[str, Any] = {}
    for split in ("primary", "confirmation"):
        section = parent["stochastic_ten_cycle"][split]
        strategies: list[Mapping[str, Any]] = [section["standard"]]
        strategies.extend(section["mf_all_agents"]["agents"])
        strategies.extend(
            [
                section["teacher"],
                section["handcrafted_recurrence"],
                section["distilled_student"],
            ]
        )
        seed_rows: list[dict[str, Any]] = []
        distributions: dict[str, Any] = {}
        for strategy in strategies:
            name = str(strategy["strategy"])
            rows = [
                {
                    "unit_id": f"{name}-seed-{row['seed']}",
                    "strategy": name,
                    "evaluation_seed": int(row["seed"]),
                    "value": _logical_z_lifetime(row),
                }
                for row in strategy["per_seed"]
            ]
            seed_rows.extend(rows)
            distributions[name] = _distribution(rows, higher_is_better=True)
        mf_agent_rows = [
            {
                "unit_id": str(agent["strategy"]),
                "value": float(
                    agent["metric_means"]["logical_z_effective_lifetime_cycles"]
                ),
            }
            for agent in section["mf_all_agents"]["agents"]
        ]
        split_results[split] = {
            "cutoff": int(section["cutoff"]),
            "registered_evaluation_seeds": [int(seed) for seed in section["seeds"]],
            "strategy_seed_rows": seed_rows,
            "per_strategy_distributions": distributions,
            "mf_all_agent_distribution": _distribution(
                mf_agent_rows, higher_is_better=True
            ),
            "student_gain_retention": parent["stochastic_retention"][split][
                "logical_z_effective_lifetime_cycles"
            ],
        }
    return {
        "lane_id": "frozen_teacher_student_gain_retention",
        "parent_task": "T4.4.4",
        "metric": "logical_z_effective_lifetime_cycles",
        "metric_direction": "higher_is_better",
        "selection_stage": "none_frozen_parent_candidates_only",
        "evaluation_used_for_selection": False,
        "all_five_mf_agents_retained": bool(
            parent["gates"]["all_five_mf_agents_are_reported_without_test_postselection"]
        ),
        "splits": split_results,
    }


def _build_lanes(parents: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "nmf_directional": _nmf_lane(parents["T2.3.7"]),
        "slow_loop_model_selection": _slow_loop_lane(parents["T4.1.1"]),
        "legacy_student_predecessor": _legacy_student_lane(parents["T4.1.5"]),
        "fresh_teacher": _teacher_lane(parents["T4.4.1"]),
        "low_dimensional_student": _student_lane(parents["T4.4.3"]),
        "gain_retention": _retention_lane(parents["T4.4.4"]),
    }


def _current_parent_implementations() -> dict[str, str]:
    return {
        "T2.3.7": nmf_parent.implementation_sha256(),
        "T4.1.1": slow_parent._implementation_sha256(),
        "T4.1.5": legacy_parent._implementation_sha256(),
        "T4.4.1": teacher_parent.implementation_sha256(),
        "T4.4.3": student_parent.implementation_sha256(),
        "T4.4.4": retention_parent.implementation_sha256(),
        "T5.1.4": branch_parent.implementation_sha256(),
    }


def _selection_registry(lanes: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "lane_id": "nmf_directional",
            "parent_task": "T2.3.7",
            "candidate_census": "5 MF plus 5 paired NMF agents",
            "selection_stage": "validation checkpoint within each agent; no agent selection",
            "evaluation_scope": "all five pairs on all primary and confirmation seeds",
            "evaluation_used_for_selection": False,
            "active_claim_role": "all-agent directional distribution",
        },
        {
            "lane_id": "slow_loop_model_selection",
            "parent_task": "T4.1.1",
            "candidate_census": "6 families plus all 5x2 neural restart scans",
            "selection_stage": "validation NLL lexicographic family selection",
            "evaluation_scope": "all six families on eight seeds",
            "evaluation_used_for_selection": False,
            "active_claim_role": "matched synthetic host-estimator pilot",
        },
        {
            "lane_id": "legacy_student_predecessor",
            "parent_task": "T4.1.5",
            "candidate_census": "3 validation-selected recurrence restarts",
            "selection_stage": "minimum validation teacher-action MSE",
            "evaluation_scope": "selected restart only; nonselected evaluation unavailable",
            "evaluation_used_for_selection": False,
            "active_claim_role": "superseded historical method detail only",
        },
        {
            "lane_id": "fresh_teacher",
            "parent_task": "T4.4.1",
            "candidate_census": "3 fresh GRU restarts",
            "selection_stage": "maximum validation selection score",
            "evaluation_scope": "all three restarts on primary and confirmation seeds",
            "evaluation_used_for_selection": False,
            "active_claim_role": "frozen offline teacher",
        },
        {
            "lane_id": "low_dimensional_student",
            "parent_task": "T4.4.3",
            "candidate_census": "1/2/4-state x 3 restarts",
            "selection_stage": "validation-only restart then smallest eligible dimension",
            "evaluation_scope": "all validation candidates and best-per-dimension evaluation",
            "evaluation_used_for_selection": False,
            "active_claim_role": "frozen distilled student candidate",
        },
        {
            "lane_id": "gain_retention",
            "parent_task": "T4.4.4",
            "candidate_census": "frozen standard, all 5 MF, teacher, recurrence, student",
            "selection_stage": "none after parent freeze",
            "evaluation_scope": "all strategies on all 8 primary and 4 confirmation seeds",
            "evaluation_used_for_selection": False,
            "active_claim_role": "frozen physical retention audit",
        },
    ]


def _iter_distributions(value: Any, path: str = "") -> list[tuple[str, Mapping[str, Any]]]:
    found: list[tuple[str, Mapping[str, Any]]] = []
    if isinstance(value, Mapping):
        if {
            "count",
            "q1",
            "median",
            "q3",
            "iqr",
            "worst_quartile",
        }.issubset(value):
            found.append((path, value))
        for key, child in value.items():
            found.extend(_iter_distributions(child, f"{path}.{key}" if path else str(key)))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            found.extend(_iter_distributions(child, f"{path}[{index}]"))
    return found


def _distribution_contract_valid(distribution: Mapping[str, Any]) -> bool:
    count = int(distribution["count"])
    worst = distribution["worst_quartile"]
    numeric = [
        distribution["minimum"],
        distribution["q1"],
        distribution["median"],
        distribution["q3"],
        distribution["iqr"],
        distribution["maximum"],
    ]
    return (
        count > 0
        and int(distribution["worst_quartile_count"]) == max(1, math.ceil(count / 4.0))
        and len(worst) == int(distribution["worst_quartile_count"])
        and all(np.isfinite(float(value)) for value in numeric)
        and float(distribution["minimum"])
        <= float(distribution["q1"])
        <= float(distribution["median"])
        <= float(distribution["q3"])
        <= float(distribution["maximum"])
        and abs(
            float(distribution["iqr"])
            - (float(distribution["q3"]) - float(distribution["q1"]))
        )
        <= 1.0e-15
    )


def _audit_summary(lanes: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    nmf = lanes["nmf_directional"]
    slow = lanes["slow_loop_model_selection"]
    teacher = lanes["fresh_teacher"]
    student = lanes["low_dimensional_student"]
    retention = lanes["gain_retention"]
    evaluation_unit_rows = (
        sum(len(nmf["splits"][split]["agent_seed_rows"]) for split in nmf["splits"])
        + len(slow["evaluation_seed_rows"])
        + sum(
            len(teacher["splits"][split]["all_restart_seed_rows"])
            for split in teacher["splits"]
        )
        + len(student["best_per_dimension_rows"])
        + sum(
            len(retention["splits"][split]["strategy_seed_rows"])
            for split in retention["splits"]
        )
    )
    hindsight = [
        slow["hindsight_test_best_diagnostic"],
        teacher["splits"]["primary"]["hindsight_test_best_diagnostic"],
        teacher["splits"]["confirmation"]["hindsight_test_best_diagnostic"],
        student["hindsight_test_best_diagnostic"],
    ]
    return {
        "selection_episode_count": 6,
        "evaluation_unit_row_count": evaluation_unit_rows,
        "distribution_count": len(_iter_distributions(lanes)),
        "active_selection_episodes_using_evaluation": 0,
        "hindsight_diagnostic_count": len(hindsight),
        "hindsight_selection_disagreement_count": sum(
            not bool(row["agrees_with_validation_selection"]) for row in hindsight
        ),
        "hindsight_diagnostics_used_to_change_selection": sum(
            bool(row["used_to_change_selection"]) for row in hindsight
        ),
        "legacy_all_candidate_evaluation_gap": (
            "T4.1.5 nonselected restart evaluation unavailable; predecessor is "
            "superseded and does not support a best-restart performance claim"
        ),
        "audit_verdict": "PASS_WITH_WARNINGS",
    }


def _contract_view(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in report.items()
        if key not in {"generated_at_utc", "contract_sha256"}
    }


def _compute_gates(
    report: Mapping[str, Any], parents: Mapping[str, Mapping[str, Any]]
) -> dict[str, bool]:
    lanes = report["lanes"]
    nmf = lanes["nmf_directional"]
    slow = lanes["slow_loop_model_selection"]
    legacy = lanes["legacy_student_predecessor"]
    teacher = lanes["fresh_teacher"]
    student = lanes["low_dimensional_student"]
    retention = lanes["gain_retention"]
    implementations = _current_parent_implementations()
    distributions = _iter_distributions(lanes)
    selected_teacher = int(parents["T4.4.1"]["selected_restart_index"])
    best_teacher_validation = max(
        range(len(parents["T4.4.1"]["training_restarts"])),
        key=lambda index: float(
            parents["T4.4.1"]["training_restarts"][index]["best_validation_score"]
        ),
    )
    return {
        "all_parent_artifacts_are_hash_bound_and_pass": all(
            row["machine_pass"] and row["sha256"] == _sha256(row["path"])
            for row in report["parent_bindings"]
        ),
        "all_parent_source_and_implementation_files_are_hash_bound": all(
            row["sha256"] == _sha256(row["path"])
            for row in (
                *report["parent_source_bindings"],
                *report["implementation_bindings"],
            )
        ),
        "all_parent_implementation_composites_are_current": all(
            parents[task_id]["implementation_sha256"] == implementation
            for task_id, implementation in implementations.items()
        ),
        "selection_registry_covers_six_declared_episodes": len(
            report["selection_registry"]
        )
        == 6
        and {row["lane_id"] for row in report["selection_registry"]}
        == set(lanes),
        "nmf_retains_all_five_paired_agents_without_test_selection": all(
            len(nmf["splits"][split]["agent_rows"]) == 5
            and len(nmf["splits"][split]["agent_seed_rows"])
            == 5 * len(nmf["splits"][split]["registered_evaluation_seeds"])
            for split in ("primary", "confirmation")
        )
        and nmf["agent_selection_rule"] == "none_all_five_paired_agents_retained"
        and nmf["evaluation_used_for_agent_selection"] is False,
        "nmf_checkpoint_selection_is_validation_only": bool(
            parents["T2.3.7"]["gates"]["checkpoint_selection_uses_validation_only"]
        )
        and nmf["checkpoint_selection_split"] == "validation_only",
        "nmf_best_agent_inflation_is_diagnostic_only": all(
            nmf["splits"][split]["hypothetical_test_best_agent"][
                "inflation_over_all_agent_median"
            ]
            >= 0.0
            and nmf["splits"][split]["hypothetical_test_best_agent"][
                "used_for_claim_or_selection"
            ]
            is False
            for split in ("primary", "confirmation")
        ),
        "slow_loop_reports_all_families_restarts_and_seeds": len(
            slow["candidate_rows"]
        )
        == 6
        and len(slow["neural_restart_validation_rows"]) == 10
        and len(slow["evaluation_seed_rows"]) == 48,
        "slow_loop_selection_is_validation_only": parents["T4.1.1"][
            "descriptor"
        ]["evaluation_used_for_selection"]
        is False
        and slow["evaluation_used_for_selection"] is False
        and slow["hindsight_test_best_diagnostic"]["used_to_change_selection"] is False,
        "legacy_student_all_restarts_and_missing_counterfactual_are_explicit": len(
            legacy["candidate_validation_rows"]
        )
        == 3
        and legacy["evaluation_used_for_training_or_selection"] is False
        and legacy["all_candidate_evaluation_available"] is False
        and bool(legacy["missing_evidence"]),
        "teacher_retains_all_three_restarts_and_all_registered_seeds": len(
            teacher["candidate_rows"]
        )
        == 3
        and len(teacher["splits"]["primary"]["all_restart_seed_rows"]) == 24
        and len(teacher["splits"]["confirmation"]["all_restart_seed_rows"]) == 12,
        "teacher_selection_recomputes_from_validation_only": teacher[
            "evaluation_used_for_selection"
        ]
        is False
        and selected_teacher == best_teacher_validation == teacher["selected_restart"],
        "teacher_test_best_reversal_is_retained_without_reselection": any(
            not teacher["splits"][split]["hindsight_test_best_diagnostic"][
                "agrees_with_validation_selection"
            ]
            for split in ("primary", "confirmation")
        )
        and all(
            teacher["splits"][split]["hindsight_test_best_diagnostic"][
                "used_to_change_selection"
            ]
            is False
            for split in ("primary", "confirmation")
        ),
        "student_reports_all_nine_validation_candidates": len(
            student["training_candidate_rows"]
        )
        == 9
        and {(row["dimension"], row["restart_index"]) for row in student["training_candidate_rows"]}
        == {(dimension, restart) for dimension in (1, 2, 4) for restart in range(3)},
        "student_selection_is_validation_only_and_evaluation_blind": student[
            "evaluation_used_for_selection"
        ]
        is False
        and parents["T4.4.3"]["selection"]["evaluation_blind"] is True
        and student["hindsight_test_best_diagnostic"]["used_to_change_selection"]
        is False,
        "retention_reports_all_five_mf_agents_and_all_seeds": retention[
            "all_five_mf_agents_retained"
        ]
        and len(retention["splits"]["primary"]["strategy_seed_rows"]) == 72
        and len(retention["splits"]["confirmation"]["strategy_seed_rows"]) == 36,
        "gain_retention_uses_only_frozen_parent_candidates": retention[
            "selection_stage"
        ]
        == "none_frozen_parent_candidates_only"
        and retention["evaluation_used_for_selection"] is False,
        "every_distribution_has_median_iqr_and_worst_quartile": len(distributions)
        >= 30
        and all(_distribution_contract_valid(value) for _, value in distributions),
        "every_active_selection_episode_is_evaluation_blind": report[
            "audit_summary"
        ]["active_selection_episodes_using_evaluation"]
        == 0
        and all(
            row["evaluation_used_for_selection"] is False
            for row in report["selection_registry"]
        ),
        "hindsight_diagnostics_never_change_frozen_selection": report[
            "audit_summary"
        ]["hindsight_diagnostics_used_to_change_selection"]
        == 0,
        "learned_decoder_performance_branch_remains_revoked": parents["T5.1.4"][
            "active_branch"
        ]["strong_branch_activated"]
        is False,
        "legacy_coverage_gap_is_warning_not_silent_pass": report["audit_summary"][
            "audit_verdict"
        ]
        == "PASS_WITH_WARNINGS"
        and bool(report["audit_summary"]["legacy_all_candidate_evaluation_gap"]),
        "no_physical_memory_device_or_hardware_claim": report["claim_boundary"][
            "physical_memory_ler_established"
        ]
        is False
        and report["claim_boundary"]["device_calibrated"] is False
        and report["claim_boundary"]["hardware_measured"] is False,
    }


def build_report() -> dict[str, Any]:
    parents = load_parent_artifacts()
    lanes = _build_lanes(parents)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pass_semantics": (
            "every current selection-bearing learned lane is traced to validation, "
            "every registered agent/restart/seed available to the active claim is "
            "reported with median IQR and worst quartile, and hindsight test-best "
            "diagnostics never alter the frozen candidate"
        ),
        "parent_bindings": _parent_bindings(parents),
        "parent_source_bindings": _bindings(PARENT_SOURCE_FILES, kind="parent_source"),
        "implementation_bindings": _bindings(
            IMPLEMENTATION_PATHS, kind="implementation"
        ),
        "selection_contract": {
            "allowed_selection_split": "validation_only",
            "independent_test_role": "evaluation_and_hindsight_bias_diagnostic_only",
            "agent_reporting_rule": "all_registered_agents_restarts_and_seeds",
            "distribution_rule": (
                "linear q1 median q3; IQR=q3-q1; worst quartile contains "
                "ceil(n/4) lowest higher-is-better or highest lower-is-better units"
            ),
            "postselection_rule": "test_best_may_be_reported_but_never_selected",
        },
        "selection_registry": _selection_registry(lanes),
        "lanes": lanes,
        "audit_summary": _audit_summary(lanes),
        "claim_boundary": {
            "allowed": (
                "validation-only model selection, all-unit distribution summaries, "
                "worst-quartile outcomes, and explicit hindsight selection-bias diagnostics"
            ),
            "forbidden": (
                "best-of-N test selection, selected-agent-only reporting, optimizer "
                "optimality, universal memory benefit, physical-memory LER, device, "
                "RTL, FPGA, board, or experiment"
            ),
            "physical_memory_ler_established": False,
            "device_calibrated": False,
            "hardware_measured": False,
        },
    }
    report["gates"] = _compute_gates(report, parents)
    report["gate_summary"] = {
        "passed": sum(bool(value) for value in report["gates"].values()),
        "total": len(report["gates"]),
    }
    report["status"] = "PASS" if all(report["gates"].values()) else "FAIL"
    rows = source_rows(report)
    report["source_data"] = {
        "path": DEFAULT_SOURCE_DATA.as_posix(),
        "row_count": len(rows),
        "rows_sha256": _canonical_sha256(rows),
        "csv_sha256": None,
    }
    report["contract_sha256"] = _canonical_sha256(_contract_view(report))
    return report


def source_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def append(
        row_type: str,
        lane_id: str,
        unit_id: str,
        metric: str,
        value: Any,
        detail: Any,
    ) -> None:
        rows.append(
            {
                "row_type": row_type,
                "lane_id": lane_id,
                "unit_id": unit_id,
                "metric": metric,
                "value": value,
                "detail": "" if detail == "" else json.dumps(detail, sort_keys=True),
            }
        )

    for binding in report["parent_bindings"]:
        append(
            "parent_binding",
            "provenance",
            binding["task_id"],
            "machine_pass",
            int(binding["machine_pass"]),
            binding,
        )
    for binding in (
        *report["parent_source_bindings"],
        *report["implementation_bindings"],
    ):
        append(
            "file_binding",
            "provenance",
            binding["binding_id"],
            "sha256_bound",
            1,
            binding,
        )
    for registry in report["selection_registry"]:
        append(
            "selection_registry",
            registry["lane_id"],
            registry["parent_task"],
            "evaluation_used_for_selection",
            int(registry["evaluation_used_for_selection"]),
            registry,
        )

    nmf = report["lanes"]["nmf_directional"]
    for split, section in nmf["splits"].items():
        for row in section["agent_rows"]:
            append(
                "nmf_agent",
                f"nmf_directional/{split}",
                row["unit_id"],
                "nmf_minus_mf_logical_z_lifetime",
                row["value"],
                row,
            )
        for row in section["agent_seed_rows"]:
            append(
                "nmf_agent_seed",
                f"nmf_directional/{split}",
                row["unit_id"],
                "nmf_minus_mf_logical_z_lifetime",
                row["value"],
                row,
            )
        append(
            "distribution",
            f"nmf_directional/{split}",
            "agents",
            "nmf_minus_mf_logical_z_lifetime",
            section["agent_distribution"]["median"],
            section["agent_distribution"],
        )
        append(
            "hindsight_diagnostic",
            f"nmf_directional/{split}",
            "hypothetical-best-agent",
            "inflation_over_all_agent_median",
            section["hypothetical_test_best_agent"][
                "inflation_over_all_agent_median"
            ],
            section["hypothetical_test_best_agent"],
        )

    slow = report["lanes"]["slow_loop_model_selection"]
    for row in slow["candidate_rows"]:
        append(
            "model_family",
            "slow_loop_model_selection",
            row["unit_id"],
            "validation_nll",
            row["validation_nll"],
            row,
        )
    for row in slow["neural_restart_validation_rows"]:
        append(
            "neural_restart",
            "slow_loop_model_selection",
            row["unit_id"],
            "validation_nll",
            row["value"],
            row,
        )
    for row in slow["evaluation_seed_rows"]:
        append(
            "model_seed",
            "slow_loop_model_selection",
            row["unit_id"],
            "evaluation_nll",
            row["value"],
            row,
        )
    for family, distribution in slow["per_family_evaluation_distributions"].items():
        append(
            "distribution",
            "slow_loop_model_selection",
            family,
            "evaluation_nll",
            distribution["median"],
            distribution,
        )
    append(
        "hindsight_diagnostic",
        "slow_loop_model_selection",
        "test-best-family",
        "postselection_optimism",
        slow["hindsight_test_best_diagnostic"]["postselection_optimism"],
        slow["hindsight_test_best_diagnostic"],
    )

    legacy = report["lanes"]["legacy_student_predecessor"]
    for row in legacy["candidate_validation_rows"]:
        append(
            "legacy_student_restart",
            "legacy_student_predecessor",
            row["unit_id"],
            "validation_mse",
            row["value"],
            row,
        )
    append(
        "missing_evidence",
        "legacy_student_predecessor",
        "nonselected-evaluation",
        "available",
        0,
        legacy["missing_evidence"],
    )

    teacher = report["lanes"]["fresh_teacher"]
    for row in teacher["candidate_rows"]:
        append(
            "teacher_restart",
            "fresh_teacher",
            row["unit_id"],
            "validation_score",
            row["validation_score"],
            row,
        )
    for split, section in teacher["splits"].items():
        for row in section["all_restart_seed_rows"]:
            append(
                "teacher_restart_seed",
                f"fresh_teacher/{split}",
                row["unit_id"],
                "logical_z_lifetime",
                row["value"],
                row,
            )
        append(
            "distribution",
            f"fresh_teacher/{split}",
            "restart-score",
            "selection_score",
            section["restart_score_distribution"]["median"],
            section["restart_score_distribution"],
        )
        append(
            "hindsight_diagnostic",
            f"fresh_teacher/{split}",
            "test-best-restart",
            "postselection_optimism",
            section["hindsight_test_best_diagnostic"]["postselection_optimism"],
            section["hindsight_test_best_diagnostic"],
        )

    student = report["lanes"]["low_dimensional_student"]
    for row in student["training_candidate_rows"]:
        append(
            "student_candidate",
            "low_dimensional_student",
            row["unit_id"],
            "validation_mse",
            row["value"],
            row,
        )
    for row in student["best_per_dimension_rows"]:
        append(
            "student_dimension",
            "low_dimensional_student",
            row["unit_id"],
            "evaluation_mse",
            row["evaluation_mse"],
            row,
        )
    append(
        "hindsight_diagnostic",
        "low_dimensional_student",
        "test-best-dimension",
        "postselection_optimism",
        student["hindsight_test_best_diagnostic"]["postselection_optimism"],
        student["hindsight_test_best_diagnostic"],
    )

    retention = report["lanes"]["gain_retention"]
    for split, section in retention["splits"].items():
        for row in section["strategy_seed_rows"]:
            append(
                "retention_strategy_seed",
                f"gain_retention/{split}",
                row["unit_id"],
                "logical_z_lifetime",
                row["value"],
                row,
            )
        for strategy, distribution in section["per_strategy_distributions"].items():
            append(
                "distribution",
                f"gain_retention/{split}",
                strategy,
                "logical_z_lifetime",
                distribution["median"],
                distribution,
            )

    for path, distribution in _iter_distributions(report["lanes"]):
        append(
            "distribution_contract",
            "all_lanes",
            path,
            "valid",
            int(_distribution_contract_valid(distribution)),
            {
                "count": distribution["count"],
                "median": distribution["median"],
                "iqr": distribution["iqr"],
                "worst_quartile_count": distribution["worst_quartile_count"],
            },
        )
    for gate, passed in report["gates"].items():
        append("gate", "governance", gate, "passed", int(bool(passed)), "")
    return rows


def validate_artifact(report: Mapping[str, Any]) -> tuple[str, ...]:
    errors: list[str] = []
    if report.get("task_id") != TASK_ID or report.get("schema_version") != SCHEMA_VERSION:
        errors.append("task/schema identity mismatch")
    if report.get("protocol_id") != PROTOCOL_ID:
        errors.append("protocol identity mismatch")
    try:
        parents = load_parent_artifacts()
        expected_lanes = _build_lanes(parents)
        if _canonical_sha256(report["lanes"]) != _canonical_sha256(expected_lanes):
            errors.append("stored audit lanes do not match current parent evidence")
        if report["selection_registry"] != _selection_registry(expected_lanes):
            errors.append("selection registry drifted from current lane census")
        if report["audit_summary"] != _audit_summary(expected_lanes):
            errors.append("audit summary drifted from current lane evidence")
        recomputed = _compute_gates(report, parents)
        if report.get("gates") != recomputed:
            errors.append("stored gates do not match recomputed evidence gates")
        if not recomputed or not all(bool(value) for value in recomputed.values()):
            errors.append("one or more selection-audit evidence gates failed")
        if report.get("status") != "PASS":
            errors.append("artifact status is not PASS")
        if report.get("contract_sha256") != _canonical_sha256(_contract_view(report)):
            errors.append("contract hash mismatch")
        expected_rows = source_rows(report)
        source = report["source_data"]
        if int(source["row_count"]) != len(expected_rows):
            errors.append("source-data row count mismatch")
        if source["rows_sha256"] != _canonical_sha256(expected_rows):
            errors.append("source-data canonical row hash mismatch")
        source_path = _repo_path(source["path"])
        if source_path.exists() and source.get("csv_sha256") != _sha256(source_path):
            errors.append("source-data CSV byte hash mismatch")
    except (KeyError, TypeError, ValueError, OSError) as exc:
        errors.append(f"malformed artifact: {exc}")
    return tuple(errors)


def write_report(
    report: Mapping[str, Any],
    *,
    artifact_path: str | Path = DEFAULT_ARTIFACT,
    source_path: str | Path = DEFAULT_SOURCE_DATA,
) -> dict[str, Any]:
    artifact = dict(report)
    rows = source_rows(artifact)
    target = _repo_path(source_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = ("row_type", "lane_id", "unit_id", "metric", "value", "detail")
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    artifact["source_data"] = {
        "path": Path(source_path).as_posix(),
        "row_count": len(rows),
        "rows_sha256": _canonical_sha256(rows),
        "csv_sha256": _sha256(target),
    }
    artifact["contract_sha256"] = _canonical_sha256(_contract_view(artifact))
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("artifact validation failed: " + "; ".join(errors))
    artifact_target = _repo_path(artifact_path)
    artifact_target.parent.mkdir(parents=True, exist_ok=True)
    artifact_target.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
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
                "audit_verdict": report["audit_summary"]["audit_verdict"],
                "gates": report["gate_summary"],
                "source_rows": report["source_data"]["row_count"],
                "evaluation_unit_rows": report["audit_summary"][
                    "evaluation_unit_row_count"
                ],
                "hindsight_disagreements": report["audit_summary"][
                    "hindsight_selection_disagreement_count"
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_ARTIFACT",
    "DEFAULT_SOURCE_DATA",
    "PROTOCOL_ID",
    "SCHEMA_VERSION",
    "TASK_ID",
    "build_report",
    "load_parent_artifacts",
    "source_rows",
    "validate_artifact",
    "write_report",
]
